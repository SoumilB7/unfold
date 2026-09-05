"""C-7 gates over the persisted 29+10 reconciliation denominator."""
from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.generate_s7_shadow as shadow_generator

from model_unfolder.evidence.reconciliation import (
    unresolved_axis_findings, unresolved_reason_class_counts,
)
from scripts.generate_s7_shadow import (
    RecipeAttemptBundle, RecipeResolution, _assert_live_shadow_matches,
    _bf16_retry, _generation_sources, _semantic_payload, _signature_recipe,
    _require_schema_version, _source_hashes, _stable_observation_payload, _targets,
    _validate_relation_cross_file, _validate_relation_payload,
    _validate_target_metadata, check,
)
from model_unfolder.evidence.relation_probe import RelationProbePlanReceipt
from physics.execution_observation import ExecutionRecipe, ObservationResult
from physics.instance_inventory import Failure
from physics.relation_observation import RelationObservationResult
from model_unfolder.evidence.document import PreparedDocument


ROOT = Path(__file__).resolve().parents[1]
S7 = ROOT / "verification" / "s7"


_ABSENT_CHECKPOINT_DTYPE = {
    "state": "absent", "path": None, "spelling": None,
    "provenance": "", "source_kind": "checkpoint", "value": None,
}
_PROBE_FLOAT32 = {
    "kind": "probe_default", "value": "float32",
    "reason": "checkpoint dtype is absent",
}


def _recipe_resolution(*, status: str = "ok", failure_detail: str = ""):
    argument_sources = {}
    flags = {
        "source": "resolved_callable_signature",
        "checkpoint_dtype": dict(_ABSENT_CHECKPOINT_DTYPE),
        "execution_dtype": "float32",
        "execution_dtype_source": dict(_PROBE_FLOAT32),
        "argument_sources": argument_sources,
        "resolution_status": status,
    }
    recipe = ExecutionRecipe(
        "r", "callable_signature", "eval", "disabled", "unspecified", False,
        "float32", {"torch": "1"}, flags=flags)
    return recipe, RecipeResolution(
        status, dict(_ABSENT_CHECKPOINT_DTYPE), "float32", recipe,
        argument_sources, failure_detail, dict(_PROBE_FLOAT32))


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
    assert len(matrix["relation_artifacts"]) == 39


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
    recipe, resolution = _recipe_resolution()
    first = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure(
            "ExecutionFailed", "execute",
            "RuntimeError: aten._grouped_mm Expected inputs of BF16 type "
            "but got float32"))
    retry_recipe = _bf16_retry(recipe)
    retry = ObservationResult(
        "failed", recipe=retry_recipe,
        failure=Failure("ExecutionFailed", "execute", "later shape failure"))
    bundle = RecipeAttemptBundle(resolution, (first, retry))
    assert bundle.to_dict()["retry_count"] == 1
    assert bundle.resolution.checkpoint_dtype == _ABSENT_CHECKPOINT_DTYPE
    assert retry.recipe.dtype == "bfloat16"
    assert retry.recipe.flags["execution_dtype_source"] == {
        "kind": "known_grouped_mm_retry", "value": "bfloat16",
        "from": "float32",
    }
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
    recipe, resolution = _recipe_resolution()
    first = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure("ExecutionFailed", "execute", "shape mismatch"))
    retry = ObservationResult(
        "failed", recipe=_bf16_retry(recipe),
        failure=Failure("ExecutionFailed", "execute", "shape mismatch"))
    with pytest.raises(ValueError, match="known dtype error"):
        RecipeAttemptBundle(resolution, (first, retry))


def test_generic_bf16_error_without_grouped_mm_cannot_trigger_retry():
    recipe, resolution = _recipe_resolution()
    first = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure(
            "ExecutionFailed", "execute",
            "unrelated op: Expected inputs of BF16 type but got float32"))
    retry = ObservationResult("failed", recipe=_bf16_retry(recipe),
                              failure=first.failure)
    with pytest.raises(ValueError, match="known dtype error"):
        RecipeAttemptBundle(resolution, (first, retry))


def test_grouped_mm_traceback_and_unrelated_dtype_line_cannot_trigger_retry():
    recipe, resolution = _recipe_resolution()
    first = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure(
            "ExecutionFailed", "execute",
            "File grouped_mm_helper.py, line 7\n"
            "RuntimeError: Expected inputs of BF16 type but got float32"))
    retry = ObservationResult("failed", recipe=_bf16_retry(recipe),
                              failure=first.failure)
    with pytest.raises(ValueError, match="known dtype error"):
        RecipeAttemptBundle(resolution, (first, retry))


def test_same_line_grouped_mm_helper_name_cannot_trigger_retry():
    recipe, resolution = _recipe_resolution()
    first = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure(
            "ExecutionFailed", "execute",
            "grouped_mm_helper: Expected inputs of BF16 type but got float32"))
    retry = ObservationResult("failed", recipe=_bf16_retry(recipe),
                              failure=first.failure)
    with pytest.raises(ValueError, match="known dtype error"):
        RecipeAttemptBundle(resolution, (first, retry))


def test_torch_grouped_mm_meta_trace_authorizes_exactly_one_retry():
    recipe, resolution = _recipe_resolution()
    first = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure(
            "ExecutionFailed", "execute",
            "RuntimeError: Expected inputs of BF16 type but got float32"),
        stderr=(
            "E<date> <time> <pid> fake_tensor.py:1] failed while attempting "
            "to run meta for aten._grouped_mm.default\n"
            "E<date> <time> <pid> fake_tensor.py:1] Traceback (most recent call last):\n"
            "E<date> <time> <pid> fake_tensor.py:1] RuntimeError: Expected "
            "inputs of BF16 type but got float32\n"))
    retry = ObservationResult(
        "failed", recipe=_bf16_retry(recipe),
        failure=Failure("ExecutionFailed", "execute", "later shape failure"))
    assert RecipeAttemptBundle(
        resolution, (first, retry)).to_dict()["retry_count"] == 1


@pytest.mark.parametrize("stderr", (
    "E] failed while attempting to run meta for aten.mm.default\n"
    "E] RuntimeError: Expected inputs of BF16 type but got float32\n",
    "E] failed while attempting to run meta for aten._grouped_mm.default\n"
    "E] RuntimeError: unrelated shape failure\n",
    "E] failed while attempting to run meta for aten._grouped_mm.default\n"
    "E] RuntimeError: Expected inputs of BF16 type but got float32\n"
    "later unrelated diagnostic\n",
))
def test_incomplete_or_noncausal_meta_trace_cannot_trigger_retry(stderr):
    recipe, resolution = _recipe_resolution()
    first = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure(
            "ExecutionFailed", "execute",
            "RuntimeError: Expected inputs of BF16 type but got float32"),
        stderr=stderr)
    retry = ObservationResult(
        "failed", recipe=_bf16_retry(recipe), failure=first.failure)
    with pytest.raises(ValueError, match="known dtype error"):
        RecipeAttemptBundle(resolution, (first, retry))


def test_failed_recipe_resolution_cannot_carry_an_execution_result():
    recipe, resolution = _recipe_resolution(
        status="failed", failure_detail="forward callable unresolved")
    executed = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure("ExecutionFailed", "execute", "runtime failure"))
    with pytest.raises(ValueError, match="typed resolution failure"):
        RecipeAttemptBundle(resolution, (executed,))


def test_conflicting_checkpoint_dtype_spellings_are_a_typed_recipe_failure():
    symbol = SimpleNamespace(qualified_name="Model")
    forward = SimpleNamespace(
        symbol=SimpleNamespace(qualified_name="Model.forward"),
        params=(SimpleNamespace(name="self", kind="positional",
                                has_default=False),
                SimpleNamespace(name="input_ids", kind="positional",
                                has_default=False)))
    index = SimpleNamespace(callables_of=lambda _symbol: (forward,))
    root = SimpleNamespace(
        address_resolved=True,
        graph=SimpleNamespace(root=SimpleNamespace(symbol=symbol)))
    inventory = SimpleNamespace(
        provenance=SimpleNamespace(
            packages=(SimpleNamespace(package="torch", version="1"),)))
    result = _signature_recipe(
        index, root, inventory,
        {"dtype": "float16", "torch_dtype": "bfloat16"})
    assert result.status == "failed"
    assert result.checkpoint_dtype["state"] == "ambiguous"
    assert result.failure_detail == "ambiguous checkpoint dtype declarations"


def test_recipe_shape_records_exact_resolved_operand_and_result():
    symbol = SimpleNamespace(qualified_name="Model")
    forward = SimpleNamespace(
        symbol=SimpleNamespace(qualified_name="Model.forward"),
        params=(SimpleNamespace(name="self", kind="positional",
                                has_default=False),
                SimpleNamespace(name="input_ids", kind="positional",
                                has_default=False)))
    index = SimpleNamespace(callables_of=lambda _symbol: (forward,))
    root = SimpleNamespace(
        address_resolved=True,
        graph=SimpleNamespace(root=SimpleNamespace(symbol=symbol)))
    inventory = SimpleNamespace(
        provenance=SimpleNamespace(
            packages=(SimpleNamespace(package="torch", version="1"),)))
    result = _signature_recipe(
        index, root, inventory, {"max_position_embeddings": 1})
    assert result.recipe.tensor_arguments[0].shape == (1, 1)
    formula = result.argument_sources["input_ids"]["calculation"]
    assert formula == {
        "operation": "min_positive_or_fallback",
        "operands": {"max_position_embeddings": 1, "upper_bound": 2},
        "fallback": 2, "result": 1,
    }


def test_resolution_recipe_flags_cannot_rewrite_checkpoint_dtype():
    recipe, _resolution = _recipe_resolution()
    declared = {
        "state": "present", "path": "dtype", "spelling": "dtype",
        "provenance": "checkpoint_declared", "source_kind": "checkpoint",
        "value": "float32",
    }
    with pytest.raises(ValueError, match="complete recipe resolution"):
        RecipeResolution(
            "ok", declared, "float32", recipe, {},
            execution_dtype_source={
                "kind": "checkpoint_declared", "value": "float32",
                "path": "dtype", "spelling": "dtype"})


def test_dtype_channels_have_closed_state_discriminated_provenance():
    recipe, _resolution = _recipe_resolution()
    for checkpoint in (
        {**_ABSENT_CHECKPOINT_DTYPE, "value": "bfloat16"},
        {**_ABSENT_CHECKPOINT_DTYPE, "source_kind": "class_default"},
        {"state": "present", "path": None, "spelling": None,
         "provenance": "checkpoint_declared", "source_kind": "checkpoint",
         "value": "float32"},
    ):
        with pytest.raises(ValueError, match="checkpoint dtype"):
            RecipeResolution(
                "ok", checkpoint, "float32", recipe, {},
                execution_dtype_source=_PROBE_FLOAT32)

    with pytest.raises(ValueError, match="cannot become checkpoint evidence"):
        RecipeResolution(
            "ok", {
                "state": "present", "path": "dtype", "spelling": "dtype",
                "provenance": "checkpoint_declared",
                "source_kind": "checkpoint", "value": "float32"},
            "float32", recipe, {}, execution_dtype_source={
                "kind": "class_default", "value": "float32",
                "path": None, "spelling": None})


def test_ok_resolution_cannot_deserialize_a_resolution_stage_failure():
    recipe, resolution = _recipe_resolution()
    failure = ObservationResult(
        "failed", recipe=recipe,
        failure=Failure(
            "ConfigurationFailed", "recipe_resolution", "fabricated"))
    with pytest.raises(ValueError, match="resolved recipe cannot carry"):
        RecipeAttemptBundle(resolution, (failure,))


def test_s7_freshness_surface_contains_every_input_and_production_dependency():
    targets = _targets()
    paths = {path.relative_to(ROOT).as_posix()
             for path in _generation_sources(targets)}
    assert {row["input"] for row in targets} <= paths
    assert "physics/execution_observation.py" in paths
    assert "physics/instance_inventory.py" in paths
    assert "model_unfolder/evidence/document.py" in paths
    assert "model_unfolder/evidence/program_index.py" in paths
    assert any(path.endswith(".yaml") for path in paths)


def test_live_shadow_must_equal_committed_logical_artifacts():
    committed = {
        "denominator": {"corpus": 29, "to_serve": 10},
        "sources": {"a": "1"}, "models": [{"slug": "x"}],
        "logical_artifacts": {"models/x.json.gz": "2"},
        "logical_observation_artifacts": {"observations/x.json.gz": "4"},
    }
    _assert_live_shadow_matches(committed, committed)
    changed = {**committed, "logical_artifacts": {
        "models/x.json.gz": "3"}}
    with pytest.raises(ValueError, match="logical_artifacts"):
        _assert_live_shadow_matches(changed, committed)
    changed_observation = {**committed, "logical_observation_artifacts": {
        "observations/x.json.gz": "5"}}
    with pytest.raises(ValueError, match="logical_observation_artifacts"):
        _assert_live_shadow_matches(changed_observation, committed)


def test_semantic_live_hash_normalizes_only_host_metadata_and_diagnostics():
    base = {
        "inventory_provenance": {"environment": {
            "platform": "mac", "network": "sandbox",
            "python": "3.12.1", "hash_seed": "0"}},
        "source_hash": "a" * 64, "span": {"line": 7},
        "ops": ["aten.add"],
    }
    host_changed = {**base, "inventory_provenance": {"environment": {
        **base["inventory_provenance"]["environment"],
        "platform": "linux", "network": "netns", "python": "3.12.9"}}}
    assert _semantic_payload(base) == _semantic_payload(host_changed)
    for changed in (
        {**base, "source_hash": "b" * 64},
        {**base, "span": {"line": 8}},
        {**base, "ops": ["aten.mul"]},
    ):
        assert _semantic_payload(base) != _semantic_payload(changed)

    # A familiar field spelling outside the exact schema path remains evidence.
    nested = {**base, "architecture": {
        "environment": {"platform": "mechanism-a"}, "stderr": "fact-a"}}
    nested_changed = {**base, "architecture": {
        "environment": {"platform": "mechanism-b"}, "stderr": "fact-b"}}
    assert _semantic_payload(nested) != _semantic_payload(nested_changed)

    attempts = {"attempts": [{"stdout": "host noise", "stderr": "host error"}]}
    attempts_changed = {
        "attempts": [{"stdout": "other noise", "stderr": "other error"}]}
    assert _semantic_payload(attempts) == _semantic_payload(attempts_changed)


def test_class_default_dtype_is_not_recorded_as_checkpoint_deployment_fact(
        monkeypatch):
    symbol = SimpleNamespace(qualified_name="Model")
    forward = SimpleNamespace(
        symbol=SimpleNamespace(qualified_name="Model.forward"),
        params=(SimpleNamespace(name="self", kind="positional",
                                has_default=False),
                SimpleNamespace(name="input_ids", kind="positional",
                                has_default=False)))
    index = SimpleNamespace(callables_of=lambda _symbol: (forward,))
    root = SimpleNamespace(
        address_resolved=True,
        graph=SimpleNamespace(root=SimpleNamespace(symbol=symbol)))
    inventory = SimpleNamespace(provenance=SimpleNamespace(
        packages=(SimpleNamespace(package="torch", version="1"),)))
    prepared = PreparedDocument(
        document={}, checkpoint={}, class_overlay={"dtype": "bfloat16"},
        provenance={})
    monkeypatch.setattr(shadow_generator, "prepare_document",
                        lambda *_args, **_kwargs: prepared)

    result = _signature_recipe(index, root, inventory, {})

    assert result.checkpoint_dtype["state"] == "absent"
    assert result.checkpoint_dtype["value"] is None
    assert result.execution_dtype == "bfloat16"
    assert result.execution_dtype_source == {
        "kind": "class_default", "value": "bfloat16",
        "path": None, "spelling": None,
    }


def test_relation_bundle_deserializes_and_exactly_partitions_plans_and_results():
    base = ExecutionRecipe(
        "base", "callable_signature", "eval", "disabled", "unspecified",
        False, "float32", {"torch": "1"})
    recipe = dataclasses.replace(
        base, recipe_id="relation-stack",
        flags={"relation_stack_path": "layers", "base_recipe_id": "base"})
    result = RelationObservationResult(
        "failed", recipe=recipe,
        failure=Failure("ExecutionFailed", "relation_observe", "typed failure"))
    plan = {
        "stack_path": "layers", "recipe": recipe.to_dict(),
        "base_recipe": base.to_dict(), "container": {},
        "container_class": {}, "member_paths": [], "member_calls": [],
        "observation_sha256": "a" * 64,
        "inventory_config_sha256": "b" * 64,
        "index_fingerprint": "c" * 64,
    }
    plan["receipt"] = RelationProbePlanReceipt.from_payload(plan).to_dict()
    payload = {
        "schema_version": 3,
        "probe_resolution": {
            "status": "resolved", "plans": [plan], "issues": [],
            "failure_kind": "", "failure_detail": "",
            "semantic_negative": False},
        "results": [result.to_dict()],
    }
    _validate_relation_payload(payload, "relations/x.json.gz", base)

    missing = {**payload, "results": []}
    with pytest.raises(ValueError, match="partition drifted"):
        _validate_relation_payload(missing, "relations/x.json.gz", base)

    wrong_stack_recipe = dataclasses.replace(
        recipe, flags={**recipe.flags, "relation_stack_path": "other"})
    wrong_stack = {**payload, "results": [dataclasses.replace(
        result, recipe=wrong_stack_recipe).to_dict()]}
    with pytest.raises(ValueError, match="partition drifted"):
        _validate_relation_payload(wrong_stack, "relations/x.json.gz", base)

    wrong_recipe = dataclasses.replace(recipe, dtype="float16")
    recipe_drift = {**payload, "results": [dataclasses.replace(
        result, recipe=wrong_recipe).to_dict()]}
    with pytest.raises(ValueError, match="recipe drifted"):
        _validate_relation_payload(recipe_drift, "relations/x.json.gz", base)

    malformed = {**payload, "results": [{"status": "ok"}]}
    with pytest.raises(ValueError, match="result is malformed"):
        _validate_relation_payload(malformed, "relations/x.json.gz", base)

    failed_without_failure = {
        **payload,
        "probe_resolution": {
            "status": "failed", "plans": [], "issues": [],
            "failure_kind": "", "failure_detail": "",
            "semantic_negative": False,
        },
        "results": [],
    }
    with pytest.raises(ValueError, match="failure payload is invalid"):
        _validate_relation_payload(
            failed_without_failure, "relations/x.json.gz", base)

    wrong_base = json.loads(json.dumps(payload))
    wrong_base_plan = wrong_base["probe_resolution"]["plans"][0]
    wrong_base_plan["base_recipe"]["recipe_id"] = "other-base"
    wrong_base_plan["receipt"] = RelationProbePlanReceipt.from_payload(
        wrong_base_plan).to_dict()
    with pytest.raises(ValueError, match="base-recipe identity drifted"):
        _validate_relation_payload(wrong_base, "relations/x.json.gz", base)

    other_base = dataclasses.replace(base, recipe_id="unrelated")
    with pytest.raises(ValueError, match="wrong observation recipe"):
        _validate_relation_payload(payload, "relations/x.json.gz", other_base)

    results = _validate_relation_payload(payload, "relations/x.json.gz", base)
    artifact = {"relation_probe_resolution": payload["probe_resolution"]}
    summary = {
        "relation_probe_status": "resolved", "relation_recipe_attempts": 1,
        "relation_recipe_ok": 0, "relation_recipe_failed": 1,
    }
    _validate_relation_cross_file(
        payload=payload, results=results, artifact=artifact, summary=summary,
        relative="relations/x.json.gz")
    with pytest.raises(ValueError, match="relation summary drifted"):
        _validate_relation_cross_file(
            payload=payload, results=results, artifact=artifact,
            summary={**summary, "relation_recipe_attempts": 0},
            relative="relations/x.json.gz")


def test_persisted_recipe_bundle_reconstructs_every_closed_invariant():
    recipe, resolution = _recipe_resolution()
    first = ObservationResult(
        "failed", recipe=recipe, failure=Failure(
            "ExecutionFailed", "execute",
            "RuntimeError: aten._grouped_mm expected inputs of bf16 type but got fp32"))
    retry = ObservationResult(
        "failed", recipe=_bf16_retry(recipe),
        failure=Failure("ExecutionFailed", "execute", "later shape failure"))
    persisted = RecipeAttemptBundle(resolution, (first, retry)).to_dict()
    assert RecipeAttemptBundle.from_dict(persisted).to_dict() == persisted

    wrong_final = {**persisted, "final_status": "ok"}
    with pytest.raises(ValueError, match="final status drifted"):
        RecipeAttemptBundle.from_dict(wrong_final)

    wrong_retry = json.loads(json.dumps(persisted))
    wrong_retry["attempts"][1]["recipe"]["literal_arguments"]["extra"] = True
    with pytest.raises(ValueError, match="change only execution dtype"):
        RecipeAttemptBundle.from_dict(wrong_retry)

    wrong_checkpoint = json.loads(json.dumps(persisted))
    wrong_checkpoint["resolution"]["checkpoint_dtype"]["value"] = "bfloat16"
    with pytest.raises(ValueError, match="absent checkpoint dtype"):
        RecipeAttemptBundle.from_dict(wrong_checkpoint)

    surplus = json.loads(json.dumps(persisted))
    surplus["attempts"][0]["unexpected"] = "self-certified"
    with pytest.raises(ValueError, match="canonical typed JSON"):
        RecipeAttemptBundle.from_dict(surplus)

    null_detail = json.loads(json.dumps(persisted))
    null_detail["resolution"]["failure_detail"] = None
    with pytest.raises(TypeError, match="native types"):
        RecipeAttemptBundle.from_dict(null_detail)

    bool_retry = json.loads(json.dumps(persisted))
    bool_retry["retry_count"] = True
    with pytest.raises(TypeError, match="native scalar types"):
        RecipeAttemptBundle.from_dict(bool_retry)


def test_target_metadata_is_exact_across_matrix_and_targets_artifact():
    targets = (
        {"cohort": "corpus", "slug": "a", "model": "org/a",
         "input": "a.json"},
        {"cohort": "to_serve", "slug": "b", "model": "org/b",
         "input": "b.json"},
    )
    matrix = {"models": [
        {**row, "occurrences": 1} for row in targets
    ]}
    targets_payload = {"models": [dict(row) for row in targets]}
    _validate_target_metadata(matrix, targets_payload, targets)

    with pytest.raises(ValueError, match="targets artifact"):
        _validate_target_metadata(
            matrix, {"models": [{**targets[0], "model": "wrong"}, targets[1]]},
            targets)
    with pytest.raises(ValueError, match="matrix target set drifted"):
        _validate_target_metadata(
            {"models": [{**matrix["models"][0], "input": "wrong.json"},
                        matrix["models"][1]]}, targets_payload, targets)


def test_matrix_and_model_schema_versions_are_native_and_exact():
    _require_schema_version({"schema_version": 1}, 1, "S7 matrix")
    for version in (True, "1", 2, None):
        with pytest.raises(ValueError, match="schema version drifted"):
            _require_schema_version(
                {"schema_version": version}, 1, "S7 model artifact")


def test_projection_categories_partition_denominator_and_keep_unknown_exact():
    for summary in _matrix()["models"]:
        assert summary["occurrences"] == (
            summary["rendered"] + summary["grouped"]
            + summary["non_architectural_container"]
            + summary["projection_unresolved"]), summary["slug"]
        for row in _artifact(summary["slug"])["table"]["occurrences"]:
            projection = row["projection"]
            if projection["kind"] == "projection_unresolved":
                assert projection["reason"] in {
                    "no product block or fact cites this occurrence",
                    "product cites facts without typed semantic proof",
                }
                if projection["unqualified_fact_keys"]:
                    assert projection["reason_class"] == "mechanism_unresolved"
                    assert projection["investigation"] is not None
                else:
                    assert projection["reason_class"] == "structure_unaccounted"
            declared = dict(projection["fact_claim_kinds"])
            assert sorted(
                projection["undeclared_fact_keys"]
                + projection["declared_unproven_fact_keys"]) == \
                projection["unqualified_fact_keys"]
            assert not (set(projection["undeclared_fact_keys"])
                        & set(projection["declared_unproven_fact_keys"]))
            assert all(key not in declared
                       for key in projection["undeclared_fact_keys"])
            assert all(key in declared
                       for key in projection["declared_unproven_fact_keys"])
            assert [item["fact_id"] for item in
                    projection["fact_claim_proofs"]] == projection["fact_keys"]
            assert all(item[1] in {
                "existence", "connection", "applied_function", "value", "relation"}
                       for item in projection["fact_claim_kinds"])
            assert all(proof["claim_kind"] == declared[proof["fact_id"]]
                       for proof in projection["fact_claim_proofs"])

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
        # The generic signature resolver's exact minimal sequence is 2; S7
        # records that recipe, never the earlier hand-written probe length 8.
        "input_shape": [1, 2, 4, 4096],
        "output_shape": [1, 2, 4096],
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
    (tmp_path / "targets.json").write_bytes(
        (S7 / "targets.json").read_bytes())
    with pytest.raises(ValueError, match="dependency surface is stale"):
        check(tmp_path)


def test_yaml_only_change_makes_dependency_surface_gate_red(monkeypatch):
    targets = _targets()
    original = _source_hashes(targets)
    yaml_path = next(path for path in _generation_sources(targets)
                     if path.suffix == ".yaml")
    relative = yaml_path.relative_to(ROOT).as_posix()
    changed = dict(original)
    changed[relative] = "0" * 64
    monkeypatch.setattr(shadow_generator, "_source_hashes",
                        lambda _targets: changed)
    with pytest.raises(ValueError, match="dependency surface is stale"):
        check()


def test_quality_workflow_checks_examples_without_platform_rasterization():
    """Linux checks source-bound artifacts without re-rasterizing them."""
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(
        encoding="utf-8")
    install = "sudo apt-get install --yes librsvg2-bin"
    example_check = "python scripts/generate_examples.py --check"
    assert install not in workflow
    assert workflow.count(example_check) == 1
    source = (ROOT / "scripts" / "generate_examples.py").read_text(
        encoding="utf-8")
    check_body = source.split("def check(", 1)[1].split("def main(", 1)[0]
    assert "rasterize_hero=False" in check_body
    assert "svg_to_png(" not in check_body
