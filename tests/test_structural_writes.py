"""H2 Part B — the StructuralWrite census across every author surface (§16.4).

These tests ARE the exit criteria (no separate doc).  The census makes a new
structural author on ANY representation — a spec field, an ``ir.extras`` leaf, an
opgraph ``Op``/``Region`` kind, a block/card key, a parameter-estimator
assumption — a conscious, reviewed act, so a structural claim cannot bypass the
fact registry by choosing a different representation.

Static half here (scan the production tree + tmp poisons); the full-corpus
runtime gate lives with the corpus fixtures in ``test_fact_registry.py``.
"""
from __future__ import annotations

from collections import Counter

from model_unfolder.evidence.structural_writes import (
    STRUCTURAL_SURFACE,
    new_structural_writes,
    runtime_structural_targets,
    scan_structural_writes,
    stale_surface_entries,
)


# --------------------------------------------------------------------------- #
# The headline: no structural write bypasses the registry via a new form
# --------------------------------------------------------------------------- #

def test_static_surface_exactly_matches_the_tree():
    """No new structural write, and no stale pinned entry — the surface is the
    complete current set, and growth on any sink is a conscious re-pin."""
    assert new_structural_writes() == []
    assert stale_surface_entries() == []


def test_no_structural_write_bypasses_via_a_new_representation():
    new = new_structural_writes()
    assert not new, (
        "a structural author bypassed the registry through a new representation: "
        + ", ".join(f"{w.sink}:{w.target} @ {w.module}:{w.symbol}" for w in new))


def test_scanner_covers_every_sink_never_vacuous():
    """Anti-vacuous: the scanner actually detects writes on every sink, so a
    green census means coverage, not dead detection."""
    by_sink = Counter(w.sink for w in scan_structural_writes())
    for sink in ("ledger", "spec", "spec_field", "extras", "opgraph", "card", "params"):
        assert by_sink[sink] > 0, f"{sink} detection found nothing — it is dead"
    assert sum(by_sink.values()) == len(STRUCTURAL_SURFACE)


# --------------------------------------------------------------------------- #
# The ONE StructuralDebt register (U2-R6) — live blocking gates
# --------------------------------------------------------------------------- #

def test_structural_debt_register_is_lawful_and_exact():
    """§R6 blocking report on the LIVE register: no duplicate row, every row's
    writer joins the live census (or exists, for non-census sinks), every
    consumer symbol exists, no deletion condition already satisfied, and every
    raw extras census target — top-level AND nested, family excuses are gone —
    has its own exact row."""
    from model_unfolder.evidence.structural_debt import debt_problems
    assert debt_problems() == []


def test_structural_debt_units_are_u3_to_u14_by_construction():
    """The H7/H8/'scoped' vocabulary is DEAD — rows carry U3–U14 only, and the
    constructor (not this test) is the enforcement; here we pin that the live
    register is non-empty and every row parses its checkable condition."""
    from model_unfolder.evidence.structural_debt import (
        MIGRATION_UNITS, STRUCTURAL_DEBT)
    assert len(STRUCTURAL_DEBT) >= 60, "the register lost rows unexpectedly"
    assert {r.migration_unit for r in STRUCTURAL_DEBT} <= MIGRATION_UNITS


def test_shrink_rule_deleting_a_writer_strands_its_row():
    """§R6: deleting/migrating a writer must shrink the register in the same
    commit — with the writer's census key gone, the row is DEAD (unbacked)
    and the blocking report names it."""
    from model_unfolder.evidence.structural_debt import (
        STRUCTURAL_DEBT, unbacked_debt_rows)
    row = next(r for r in STRUCTURAL_DEBT
               if r.sink_kind == "extras"
               and r.structural_target == "block_diffusion")
    from model_unfolder.evidence.structural_writes import (
        scan_structural_write_multiset)
    keys = {(k.module, k.enclosing_symbol, k.sink_kind, k.normalized_target)
            for k in scan_structural_write_multiset()}
    assert unbacked_debt_rows((row,), census_keys=keys) == []
    keys.discard(row.writer_key)
    assert unbacked_debt_rows((row,), census_keys=keys) == [row]


# --------------------------------------------------------------------------- #
# Runtime mechanism (fast single fixture; the full-corpus gate is elsewhere)
# --------------------------------------------------------------------------- #

def test_runtime_targets_are_top_level_and_covered():
    from test_support import LLAMA
    import model_unfolder as mu
    from model_unfolder.evidence.structural_debt import debt_targets

    targets = runtime_structural_targets(mu.unfold(LLAMA).ir)
    assert all(sink == "extras" and "." not in t for sink, t in targets), \
        "runtime gate unit must be top-level extras keys"
    uncovered = {t for _, t in targets if t not in debt_targets("extras")}
    assert not uncovered, f"runtime top-level extras not in the register: {uncovered}"


# --------------------------------------------------------------------------- #
# Five poisons — one per sink §16.4 names (each fires on a NEW target)
# --------------------------------------------------------------------------- #

def _pkg(tmp_path):
    package = tmp_path / "model_unfolder"
    package.mkdir()
    return package


def test_poison_new_nested_extras_field(tmp_path):
    pkg = _pkg(tmp_path)
    (pkg / "poison.py").write_text(
        "def f(ir):\n    ir.extras['moe'] = {'fabricated_leaf': 1}\n")
    new = new_structural_writes(root=pkg)
    assert any(w.sink == "extras" and w.target == "moe.fabricated_leaf" for w in new)


def test_poison_new_spec_field(tmp_path):
    pkg = _pkg(tmp_path)
    (pkg / "poison.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\nclass AttentionSpec:\n"
        "    kind: str\n"
        "    fabricated_field: int = 0\n")
    new = new_structural_writes(root=pkg)
    assert any(w.sink == "spec_field" and w.target == "AttentionSpec.fabricated_field"
               for w in new)


def test_poison_new_opgraph_default(tmp_path):
    pkg = _pkg(tmp_path)
    (pkg / "poison.py").write_text(
        "def draw():\n    return Op(kind='fabricated_mha_default')\n")
    new = new_structural_writes(root=pkg)
    assert any(w.sink == "opgraph" and w.target == "Op:fabricated_mha_default"
               for w in new)


def test_poison_new_card_claim(tmp_path):
    pkg = _pkg(tmp_path)
    views = pkg / "renderers" / "html" / "block_views"
    views.mkdir(parents=True)
    (views / "poison.py").write_text(
        "def card():\n    return {'lane': 'fabricated_lane'}\n")
    new = new_structural_writes(root=pkg)
    assert any(w.sink == "card" and w.target == "lane" for w in new)


def test_poison_new_param_formula(tmp_path):
    pkg = _pkg(tmp_path)
    (pkg / "params.py").write_text(
        "def new_estimator():\n    out = []\n    out.append('assume fabricated')\n")
    new = new_structural_writes(root=pkg)
    assert any(w.sink == "params" and "new_estimator" in w.target for w in new)


def test_poison_controls_do_not_fire_on_known_targets(tmp_path):
    """Anti-false-positive: writing a KNOWN target (an existing nested extras
    leaf, an existing op kind) is not flagged — only genuinely new ones are."""
    pkg = _pkg(tmp_path)
    (pkg / "poison.py").write_text(
        "def f(ir):\n    ir.extras['block_diffusion'] = {'canvas_length': 8}\n"
        "def g():\n    return Op(kind='q_proj')\n")
    new = new_structural_writes(root=pkg)
    assert not any(
        w.target in ("block_diffusion.canvas_length", "Op:q_proj") for w in new)
