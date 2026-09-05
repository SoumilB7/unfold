"""C-7 gates over the persisted 29+10 reconciliation denominator."""
from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.reconciliation import (
    unresolved_axis_findings, unresolved_reason_class_counts,
)
from scripts.generate_s7_shadow import (
    RecipeAttemptBundle, RecipeResolution, _bf16_retry,
    _stable_observation_payload, check,
)
from physics.execution_observation import ExecutionRecipe, ObservationResult
from physics.instance_inventory import Failure


ROOT = Path(__file__).resolve().parents[1]
S7 = ROOT / "verification" / "s7"


def _matrix():
    return json.loads((S7 / "matrix.json").read_text(encoding="utf-8"))


def _artifact(slug: str):
    with gzip.open(S7 / "models" / f"{slug}.json.gz", "rt",
                   encoding="utf-8") as stream:
        return json.load(stream)


def test_observation_diagnostics_drop_only_volatile_torch_prefix_metadata():
    class Result:
        def __init__(self, stderr: str):
            self.stderr = stderr

        def to_dict(self):
            return {"stdout": "", "stderr": self.stderr,
                    "failure": {"detail": "RuntimeError: shape mismatch"}}

    first = _stable_observation_payload(Result(
        "E0905 01:12:46.242000 2526 site.py:7] shape mismatch\n"))
    second = _stable_observation_payload(Result(
        "E0905 03:16:59.411000 11254 site.py:7] shape mismatch\n"))
    changed = _stable_observation_payload(Result(
        "E0905 03:16:59.411000 11254 site.py:7] dtype mismatch\n"))
    assert first == second
    assert first != changed
    assert first["failure"]["detail"] == "RuntimeError: shape mismatch"


def test_shadow_denominator_and_artifact_hashes_are_current():
    check()
    matrix = _matrix()
    assert matrix["denominator"] == {"corpus": 29, "to_serve": 10}
    assert len(matrix["models"]) == 39
    assert len(matrix["artifacts"]) == 39
    assert len(matrix["observation_artifacts"]) == 39


def test_every_model_has_a_typed_signature_attempt_and_split_execution_counts():
    for summary in _matrix()["models"]:
        assert summary["no_recipe_attempted"] == 0, summary["slug"]
        assert summary["execution_unresolved"] == (
            summary["no_recipe_attempted"]
            + summary["unobserved_no_static_proof"])
        with gzip.open(
                S7 / "observations" / f"{summary['slug']}.json.gz", "rt",
                encoding="utf-8") as stream:
            result = json.load(stream)
        assert result["schema_version"] == 2
        assert len(result["attempts"]) in {1, 2}
        assert result["attempts"][0]["recipe"]["flags"]["source"] == \
            "resolved_callable_signature"
        assert result["final_status"] in {"ok", "failed"}
        assert result["retry_count"] == len(result["attempts"]) - 1


def test_known_dtype_retry_is_single_and_never_rewrites_checkpoint_dtype():
    recipe = ExecutionRecipe(
        "r", "callable_signature", "eval", "disabled", "unspecified", False,
        "float32", {"torch": "1"},
        flags={"source": "resolved_callable_signature",
               "checkpoint_dtype": {"value": None, "state": "absent"}})
    resolution = RecipeResolution(
        "ok", {"value": None, "state": "absent"}, "float32", recipe, {})
    first = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure(
            "ExecutionFailed", "execute",
            "RuntimeError: Expected inputs of BF16 type but got float32"))
    retry_recipe = _bf16_retry(recipe)
    retry = ObservationResult(
        "failed", recipe=retry_recipe,
        failure=Failure("ExecutionFailed", "execute", "later shape failure"))
    bundle = RecipeAttemptBundle(resolution, (first, retry))
    assert bundle.to_dict()["retry_count"] == 1
    assert bundle.resolution.checkpoint_dtype == {
        "value": None, "state": "absent"}
    assert retry.recipe.dtype == "bfloat16"
    with pytest.raises(ValueError, match="one attempt and at most one retry"):
        RecipeAttemptBundle(resolution, (first, retry, retry))

    forged_recipe = dataclasses.replace(
        retry_recipe,
        flags={**retry_recipe.flags,
               "checkpoint_dtype": {"value": "bfloat16", "state": "present"}})
    forged = ObservationResult(
        "failed", recipe=forged_recipe,
        failure=Failure("ExecutionFailed", "execute", "later shape failure"))
    with pytest.raises(ValueError, match="change only execution dtype"):
        RecipeAttemptBundle(resolution, (first, forged))


def test_retry_for_a_non_dtype_error_is_rejected():
    recipe = ExecutionRecipe(
        "r", "callable_signature", "eval", "disabled", "unspecified", False,
        "float32", {"torch": "1"},
        flags={"source": "resolved_callable_signature"})
    resolution = RecipeResolution("ok", {}, "float32", recipe, {})
    first = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure("ExecutionFailed", "execute", "shape mismatch"))
    retry = ObservationResult(
        "failed", recipe=_bf16_retry(recipe),
        failure=Failure("ExecutionFailed", "execute", "shape mismatch"))
    with pytest.raises(ValueError, match="known dtype error"):
        RecipeAttemptBundle(resolution, (first, retry))


def test_failed_recipe_resolution_cannot_carry_an_execution_result():
    recipe = ExecutionRecipe(
        "r", "callable_signature", "eval", "disabled", "unspecified", False,
        "float32", {"torch": "1"}, flags={"resolution_status": "failed"})
    resolution = RecipeResolution(
        "failed", {}, "float32", recipe, {}, "forward callable unresolved")
    executed = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure("ExecutionFailed", "execute", "runtime failure"))
    with pytest.raises(ValueError, match="typed resolution failure"):
        RecipeAttemptBundle(resolution, (executed,))


def test_projection_categories_partition_denominator_and_keep_unknown_exact():
    for summary in _matrix()["models"]:
        assert summary["occurrences"] == (
            summary["rendered"] + summary["grouped"]
            + summary["non_architectural_container"]
            + summary["projection_unresolved"]), summary["slug"]
        for row in _artifact(summary["slug"])["table"]["occurrences"]:
            projection = row["projection"]
            if projection["kind"] == "projection_unresolved":
                assert projection["reason"] == \
                    "no product block or fact cites this occurrence"
                assert projection["reason_class"] == "structure_unaccounted"
            assert [item[0] for item in projection["fact_claim_kinds"]] == \
                projection["fact_keys"]
            assert all(item[1] in {
                "existence", "connection", "applied_function", "value", "relation"}
                       for item in projection["fact_claim_kinds"])

    llama = next(row for row in _matrix()["models"]
                 if row["slug"] == "llama-7b")
    assert llama["rendered"] > 0
    assert llama["grouped"] > 0
    assert llama["projection_unresolved"] < llama["occurrences"]


def test_every_unresolved_axis_has_exactly_one_v26_reason_class():
    classes = {
        "investigation_missing", "structure_unaccounted",
        "mechanism_unresolved",
    }
    unresolved_kinds = {
        "construction": "construction_conflict",
        "execution": "execution_unresolved",
        "projection": "projection_unresolved",
    }
    for summary in _matrix()["models"]:
        table = _artifact(summary["slug"])["table"]
        counts = unresolved_reason_class_counts(table)
        assert counts == {name: summary[name] for name in sorted(classes)}
        for row in table["occurrences"]:
            for axis, unresolved_kind in unresolved_kinds.items():
                value = row[axis]
                if value["kind"] == unresolved_kind:
                    assert value["reason_class"] in classes
                    if value["reason_class"] == "mechanism_unresolved":
                        assert value["investigation"] is not None
                else:
                    assert value["reason_class"] is None
                    assert value["investigation"] is None
        for relation in table["relations"]:
            if relation["kind"] == "relation_unresolved":
                assert relation["reason_class"] in classes
                if relation["reason_class"] == "mechanism_unresolved":
                    assert relation["investigation"] is not None
            else:
                assert relation["reason_class"] is None
                assert relation["investigation"] is None


def test_closed_torch_types_always_carry_exact_framework_meaning():
    for summary in _matrix()["models"]:
        for row in _artifact(summary["slug"])["table"]["occurrences"]:
            runtime = row["provenance"]["runtime_class"]
            if runtime is None or not runtime["module"].startswith(
                    "torch.nn.modules."):
                continue
            meaning = row["provenance"]["meaning"]
            assert meaning["framework_primitive"] == runtime


def test_every_inventory_occurrence_has_exactly_three_axes_and_no_silent_drop():
    for summary in _matrix()["models"]:
        artifact = _artifact(summary["slug"])
        occurrences = artifact["table"]["occurrences"]
        assert len(occurrences) == summary["occurrences"] > 0
        assert all(set(row) == {
            "provenance", "construction", "execution", "projection"}
                   for row in occurrences)
        assert artifact["blocking_findings"] == unresolved_axis_findings(
            artifact["table"])
        assert artifact["blocking_findings"], summary["slug"]


def test_hard_relation_witnesses_use_the_expected_existing_evidence():
    gemma2 = _artifact("gemma-2-2b-it")["table"]["relations"]
    assert [row["kind"] for row in gemma2] == ["param_share"]

    assert _artifact("stable-diffusion-3-5-large")["table"]["relations"] == []

    gemma = _artifact("gemma-3n-e2b")["table"]["relations"]
    assert {row["kind"] for row in gemma} == {
        "param_share", "activation_reuse", "multi_stream_residual",
        "per_layer_side_input"}
    multi = next(row for row in gemma
                 if row["kind"] == "multi_stream_residual")
    assert multi["detail"]["stream_count"] == 4
    reuse = {(row["detail"]["from_layer"], row["detail"]["to_layer"])
             for row in gemma if row["kind"] == "activation_reuse"}
    assert reuse == {
        (18, 20), (18, 21), (18, 22), (18, 23), (19, 24),
        (18, 25), (18, 26), (18, 27), (18, 28), (19, 29),
    }

    deepseek = _artifact("deepseek-v4-flash")["table"]["relations"]
    assert {row["kind"] for row in deepseek} == {
        "multi_stream_residual", "side_head"}
    assert next(row for row in deepseek
                if row["kind"] == "multi_stream_residual")["detail"][
                    "stream_count"] == 4
    head = next(row for row in deepseek if row["kind"] == "side_head")
    assert head["targets"] == ["model.hc_head"]
    assert head["detail"] == {
        "input_shape": [1, 8, 4, 4096],
        "output_shape": [1, 8, 4096],
    }


def test_nemotron_aperiodic_construction_order_is_not_forced_into_a_cycle():
    schedules = _matrix()["models"]
    nemotron = next(row for row in schedules if row["slug"] == "nemotron-h-8b")
    stack = next(row for row in nemotron["construction_schedules"]
                 if row["container_path"] == "model.layers")
    assert stack["member_count"] == 52
    assert stack["period"] is None
    assert [group[0] for group in stack["groups_in_encounter_order"]] == [0, 1, 7]
    assert sorted(index for group in stack["groups_in_encounter_order"]
                  for index in group) == list(range(52))


def test_source_change_poison_makes_the_artifact_gate_red(tmp_path):
    matrix = _matrix()
    source = next(iter(matrix["sources"]))
    matrix["sources"][source] = "0" * 64
    (tmp_path / "matrix.json").write_text(
        json.dumps(matrix), encoding="utf-8")
    with pytest.raises(ValueError, match="stale for source"):
        check(tmp_path)
