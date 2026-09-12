"""H9-core — prove the reusable metamorphic harness on reference models.

Each relation is exercised so that H7/H8 migrated readers can trust it: the
rename/collision/provenance relations hold on real decoders and multimodal
models, the equivalent-control relation holds across an alias spelling, and the
missing-source relation actually FIRES on a source-less config (not just skipped).
"""
from __future__ import annotations

import pytest

from test_support import LLAMA, PIXTRAL_STYLE, metamorphic


def test_rename_invariant_holds_on_a_decoder():
    metamorphic.assert_rename_invariant(LLAMA)


def test_rename_invariant_holds_on_a_multimodal():
    metamorphic.assert_rename_invariant(PIXTRAL_STYLE)


def test_collision_invariant_separates_multimodal_siblings():
    metamorphic.assert_collision_invariant(PIXTRAL_STYLE)


def test_partial_source_provenance_integrity_on_a_decoder():
    """Every code_proven fact cites a source — no fabricated proof."""
    metamorphic.assert_partial_source_invariant(LLAMA)


def test_equivalent_control_across_an_alias_spelling():
    """LLAMA vs the same config with ``hidden_size`` spelled as its alias
    ``n_embd`` → identical structure (the alias vocabulary, not identity, absorbs
    the spelling)."""
    twin = {k: v for k, v in LLAMA.items() if k != "hidden_size"}
    twin["n_embd"] = LLAMA["hidden_size"]
    metamorphic.assert_equivalent_control(LLAMA, twin)


def test_missing_source_relation_actually_fires():
    """MISSING-SOURCE must FIRE (not just skip): a config whose model_type has no
    installed modeling file resolves no source, so it may carry NO code_proven
    fact.  We assert the relation holds AND that the source really was absent
    (else the test would be vacuous)."""
    from model_unfolder.parser import _coerce
    from model_unfolder.evidence.context import ParseContext

    # Source resolves from BOTH model_type and architectures, so fake both to a
    # non-existent (but causal-LM-shaped, so the adapter still accepts it) name.
    sourceless = {**LLAMA, "model_type": "no_such_arch_zzz",
                  "architectures": ["NoSuchInstalledModelForCausalLM"]}
    ctx = ParseContext.build(_coerce(sourceless), source="local")
    assert not (getattr(ctx.source_bundle, "files", ()) or ()), \
        "test setup: expected no source for an unknown architecture"
    metamorphic.assert_missing_source_invariant(sourceless)


def test_run_all_bundles_the_relations():
    """The convenience entry a migrated reader calls."""
    twin = {k: v for k, v in LLAMA.items() if k != "hidden_size"}
    twin["n_embd"] = LLAMA["hidden_size"]
    metamorphic.run_all(LLAMA, equivalent=twin)
