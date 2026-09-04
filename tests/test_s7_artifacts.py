"""C-7 gates over the persisted 29+10 reconciliation denominator."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.reconciliation import unresolved_axis_findings
from scripts.generate_s7_shadow import check


ROOT = Path(__file__).resolve().parents[1]
S7 = ROOT / "verification" / "s7"


def _matrix():
    return json.loads((S7 / "matrix.json").read_text(encoding="utf-8"))


def _artifact(slug: str):
    with gzip.open(S7 / "models" / f"{slug}.json.gz", "rt",
                   encoding="utf-8") as stream:
        return json.load(stream)


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
        assert result["recipe"]["flags"]["source"] == \
            "resolved_callable_signature"
        assert result["status"] in {"ok", "failed"}


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

    llama = next(row for row in _matrix()["models"]
                 if row["slug"] == "llama-7b")
    assert llama["rendered"] > 0
    assert llama["grouped"] > 0
    assert llama["projection_unresolved"] < llama["occurrences"]


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
