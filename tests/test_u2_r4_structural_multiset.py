"""U2-R4 — the structural-writer census is a multiset keyed on writer identity.

A set keyed by ``(sink, target)`` dropped a SECOND author of one target, so a
writer could reach the same structural sink from a new symbol and never appear
in the census. Keying on the full writer identity — module, enclosing symbol,
sink kind, normalized target — and counting a multiset makes every author
visible, so moving or renaming a writer cannot hide it.
"""
from __future__ import annotations

import pathlib
from collections import Counter

from model_unfolder.evidence import structural_writes as sw
from model_unfolder.evidence.structural_writes import (
    StructuralWriteKey,
    scan_structural_write_multiset,
)


def test_the_multiset_distinguishes_two_writers_of_one_target():
    a = StructuralWriteKey("mod_a.py", "sym_a", "spec", "AttentionSpec")
    b = StructuralWriteKey("mod_b.py", "sym_b", "spec", "AttentionSpec")
    same = StructuralWriteKey("mod_a.py", "sym_a", "spec", "AttentionSpec")
    assert len({a, b}) == 2         # two authors, two keys
    assert a == same                # same author, one key (line-insensitive)


def test_the_live_multiset_exceeds_the_deduped_set():
    """The real corpus: the multiset must reveal authors the sink+target set
    hides — otherwise the two are the same and the fix is cosmetic."""
    multiset = scan_structural_write_multiset()
    deduped = sw.scan_structural_writes()
    assert len(multiset) > len(deduped), (
        "the writer-identity multiset must surface more authors than the "
        "sink+target set; if equal, no target has a second writer and the "
        "scanner is not seeing them")


def test_targets_with_multiple_writers_are_all_counted():
    multiset = scan_structural_write_multiset()
    by_target = Counter((k.sink_kind, k.normalized_target) for k in multiset)
    multi = {kt: c for kt, c in by_target.items() if c > 1}
    # these were single rows before the multiset; each must now show every author
    assert multi, "no multi-writer target found — the census still collapses them"
    assert ("spec", "AttentionSpec") in multi


# -------------------------------------------------------------------------
# Poison — a second writer of an existing target must GROW the multiset
# -------------------------------------------------------------------------

def test_a_second_symbol_writing_one_target_grows_the_multiset(tmp_path):
    """The census counts the SECOND author; the set would not."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    # AttentionSpec is a scanner-recognized spec vocabulary (a real structural
    # sink); two symbols constructing it are two authors.
    one = pkg / "one.py"
    one.write_text("def a():\n    return AttentionSpec(kind='mha')\n")
    before = scan_structural_write_multiset(pkg)

    two = pkg / "two.py"
    two.write_text("def b():\n    return AttentionSpec(kind='gqa')\n")
    after = scan_structural_write_multiset(pkg)

    a_writers = {k for k in after if k.normalized_target == "AttentionSpec"}
    b_writers = {k for k in before if k.normalized_target == "AttentionSpec"}
    assert len(a_writers) == len(b_writers) + 1, (
        "a second symbol authoring the same spec must add a distinct writer key")


def test_the_writer_gate_is_clean_at_baseline():
    """U2-R4 vet: the multiset is THE gate.  At baseline no writer is new or
    stale — the pinned surface matches the live tree exactly."""
    assert sw.new_structural_writers() == []
    assert sw.stale_structural_writers() == []


def test_the_gate_flags_a_second_writer_of_an_ALREADY_PINNED_target(monkeypatch):
    """The load-bearing replacement: the OLD (sink,target) gate could not see a
    second author of an already-pinned target.  The writer-identity gate does.

    Simulated by pinning a baseline that lacks one live writer of AttentionSpec
    — the gate must report that writer as new even though AttentionSpec is
    already an authored target."""
    live = set(sw.scan_structural_write_multiset())
    attn_writers = sorted(
        (k for k in live if k.normalized_target == "AttentionSpec"),
        key=lambda k: (k.module, k.enclosing_symbol))
    assert len(attn_writers) >= 2, "need >=2 AttentionSpec authors for this test"
    # pin a baseline (count-aware) missing exactly ONE live author
    missing = attn_writers[-1]
    pinned = frozenset(live - {missing})
    monkeypatch.setattr(sw, "STRUCTURAL_WRITERS", pinned)
    monkeypatch.setattr(sw, "_STRUCTURAL_WRITERS_MULTI", {})
    new = sw.new_structural_writers()
    assert missing in new, (
        "a second author of an already-pinned target must be flagged — the "
        "identity gate sees it where the sink+target set could not")


def test_runtime_surface_recurses_into_the_real_ir():
    """§R4 runtime: the recursive surface reaches deep structural nodes a
    top-level-only scan cannot — a per-layer attention spec, a nested group."""
    import json
    import model_unfolder as mu
    from model_unfolder.evidence.structural_writes import runtime_structural_surface
    cfg = json.loads(
        (pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
         / "llama-7b.json").read_text())["config"]
    nodes = runtime_structural_surface(mu.unfold(cfg).to_ir())
    assert len(nodes) > 100, "runtime surface did not recurse"
    owners = {n.owner for n in nodes}
    # a value deep inside the layer stack must be reached (not just top-level)
    assert any("layers" in o and "attention" in o for o in owners), (
        "runtime surface stayed shallow — a per-layer attention node was not "
        "visited")
    # every node carries a value hash so drift is detectable
    assert all(n.value_hash for n in nodes)


def test_moving_a_writer_within_its_symbol_is_not_growth(tmp_path):
    """Line-insensitive: re-indenting or reordering a call inside its function
    is not a new author."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    mod = pkg / "m.py"
    mod.write_text("def a():\n    x = 1\n    return AttentionSpec(kind='mha')\n")
    before = scan_structural_write_multiset(pkg)
    mod.write_text(
        "def a():\n    y = 2\n    z = 3\n    return AttentionSpec(kind='mha')\n")
    after = scan_structural_write_multiset(pkg)
    assert before == after
