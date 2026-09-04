"""S4 anti-vacuous recall, support-denominator and ship-receipt laws."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_unfolder import unfold
from model_unfolder.evidence.coverage import coverage_problems


ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    return json.loads((ROOT / "coverage.json").read_text(encoding="utf-8"))


def test_coverage_manifest_is_closed_and_every_model_has_zero_silent():
    assert coverage_problems(_manifest()) == []


def test_coverage_poison_one_silent_model_turns_the_gate_red():
    poisoned = _manifest()
    poisoned["models"][0]["silent"] = 1
    problems = coverage_problems(poisoned)
    assert any("silent=1" in problem for problem in problems)


def test_coverage_poison_cannot_shrink_the_declared_support_set():
    poisoned = _manifest()
    removed = poisoned["models"].pop()
    problems = coverage_problems(poisoned)
    assert any(removed["cohort"] in problem and "denominator changed" in problem
               for problem in problems)


def test_census_and_coverage_checks_are_wired_into_ci():
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(
        encoding="utf-8")
    assert "python scripts/census.py --check" in workflow
    assert "python scripts/coverage.py --check" in workflow


@pytest.mark.parametrize("path", sorted(
    (ROOT / "tests" / "unseen_model_configs").glob("*.json")))
def test_unseen_support_input_names_an_exact_model_and_config(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"config", "model_id"}
    assert isinstance(payload["model_id"], str) and "/" in payload["model_id"]
    assert isinstance(payload["config"], dict) and payload["config"]


def _unseen(slug: str) -> dict:
    return json.loads((ROOT / "tests" / "unseen_model_configs" /
                       f"{slug}.json").read_text(encoding="utf-8"))["config"]


@pytest.mark.parametrize("slug", [
    "qwen3-5-27b-full",
    "qwen3-6-35b-a3b",
    "qwen3-vl-235b",
])
def test_qwen_side_reader_failures_keep_the_main_stack_and_projector_chip(slug):
    diagram = unfold(_unseen(slug))
    ir = diagram.to_ir()
    assert ir["layers"], "a side-reader failure erased the proven text stack"
    assert any("projector evidence unresolved" in warning
               for warning in ir["warnings"])
    assert "Projector evidence unresolved" in diagram.to_html()


@pytest.mark.parametrize(("slug", "check"), [
    ("jamba-v0-1", "op_conformance"),
    ("deepseek-coder-v2-lite", "fact_conformance"),
])
def test_hard_model_conformance_findings_are_visible_in_plain_unfold(slug, check):
    diagram = unfold(_unseen(slug))
    rows = diagram.to_ir()["extras"]["ship_findings"]
    assert any(row["check"] == check for row in rows)
    html = diagram.to_html()
    assert "unresolved evidence" in html
    exact = [row["message"] for row in rows if row["check"] == check]
    assert exact and all(message in html for message in exact)


def test_typed_ship_finding_is_never_mislabeled_as_partial_config():
    diagram = unfold(_unseen("jamba-v0-1"))
    html = diagram.to_html()
    assert diagram.to_ir()["extras"]["ship_findings"]
    assert "unresolved evidence" in html
    assert "partial config" not in html
