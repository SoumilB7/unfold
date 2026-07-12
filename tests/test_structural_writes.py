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

import pytest

from model_unfolder.evidence.structural_writes import (
    LEGACY_EXTRAS,
    STRUCTURAL_SURFACE,
    legacy_extras_targets,
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
# The structured legacy register (owner · reason · unit · deletion)
# --------------------------------------------------------------------------- #

def test_legacy_extras_register_is_fully_qualified():
    """§16.4: every legacy entry carries owner, reason, migration unit and
    intended deletion — never a bare string in an allowlist."""
    units = {"H7", "H8", "scoped"}
    for entry in LEGACY_EXTRAS:
        assert entry.target and entry.owner and entry.reason and entry.deletion, entry
        assert entry.unit in units, f"{entry.target}: unit {entry.unit!r} not in {units}"
    targets = [entry.target for entry in LEGACY_EXTRAS]
    assert len(targets) == len(set(targets)), "duplicate legacy-extras targets"


def test_every_raw_extras_write_has_a_structured_debt_row():
    """No top-level raw ``extras`` structural write may exist without a
    structured debt row — the bare-string allowlist is gone."""
    top_level = {t for s, t in STRUCTURAL_SURFACE if s == "extras" and "." not in t}
    missing = top_level - legacy_extras_targets()
    assert not missing, f"raw extras writes without a structured register row: {missing}"


# --------------------------------------------------------------------------- #
# Runtime mechanism (fast single fixture; the full-corpus gate is elsewhere)
# --------------------------------------------------------------------------- #

def test_runtime_targets_are_top_level_and_covered():
    from test_support import LLAMA
    import model_unfolder as mu

    targets = runtime_structural_targets(mu.unfold(LLAMA).ir)
    assert all(sink == "extras" and "." not in t for sink, t in targets), \
        "runtime gate unit must be top-level extras keys"
    uncovered = {t for _, t in targets if t not in legacy_extras_targets()}
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
        "def f(ir):\n    ir.extras['moe'] = {'num_experts': 8}\n"
        "def g():\n    return Op(kind='q_proj')\n")
    new = new_structural_writes(root=pkg)
    assert not any(w.target in ("moe.num_experts", "Op:q_proj") for w in new)
