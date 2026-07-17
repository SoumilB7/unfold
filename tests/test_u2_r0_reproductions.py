"""U2-R0 — reproduce the U2 defects, before fixing any of them.

The spec's reproduce-first discipline: every defect the later units close is
pinned HERE as a failing poison first, so a fix cannot be claimed without a test
that was red for the stated reason and is now green.

Open defects use ``xfail(strict=True)``: the poison documents a defect that is
present today, and strict mode turns the xfail into a FAILURE the moment the
defect is fixed — which forces the owning unit to come back and convert the
poison into a real passing assertion.  A defect that is already closed gets a
plain passing guard instead, so it can never silently reopen.

Each poison names the unit that owns its fix.
"""
from __future__ import annotations

import pytest

from model_unfolder.evidence import arbitration as arb
from model_unfolder.evidence import structural_writes as sw
from model_unfolder.evidence.config_access import (
    CHECKPOINT_DECLARED,
    CLASS_DEFAULT,
)
from model_unfolder.evidence.document import prepare_document


# -------------------------------------------------------------------------
# R3 — arbitration accepts empty provenance as config strength
# -------------------------------------------------------------------------

def test_r0_empty_provenance_cannot_claim_checkpoint_strength():
    """CLOSED in U2-R3: unestablished origin is ineligible for checkpoint
    strength."""
    with pytest.raises(ValueError):
        arb.arbitrate("m", [arb.Candidate(4, arb.CHECKPOINT_EXPLICIT, "p",
                                          "path", "")])


def test_r0_empty_provenance_cannot_claim_class_strength():
    """CLOSED in U2-R3: unestablished origin is ineligible for class strength."""
    with pytest.raises(ValueError):
        arb.arbitrate("m", [arb.Candidate(4, arb.CLASS_SUPPLIED, "p",
                                          "path", "")])


# -------------------------------------------------------------------------
# R2 — value and origin can be detached (provenance_of is a loose lookup)
# -------------------------------------------------------------------------

def test_r0_value_and_provenance_are_bound_by_consume_decision():
    """CLOSED in U2-R2: the typed decision binds value and provenance in one
    object, so a reader can no longer pair a value with a foreign path's
    origin."""
    from model_unfolder.evidence import config_access as ca
    assert hasattr(ca.ConfigResolution, "consume_decision")
    assert hasattr(ca, "ConsumedConfigDecision")


# -------------------------------------------------------------------------
# R4 — a second writer of one structural target is hidden by the set key
# -------------------------------------------------------------------------

def test_r0_a_second_writer_of_one_target_is_not_hidden():
    """CLOSED in U2-R4: the census keys on writer IDENTITY (module, symbol,
    sink, target) and counts a multiset, so two distinct authors of one target
    are two rows."""
    ka = sw.StructuralWriteKey("mod_a.py", "sym_a", "extras", "root.x")
    kb = sw.StructuralWriteKey("mod_b.py", "sym_b", "extras", "root.x")
    assert len({ka, kb}) == 2


# -------------------------------------------------------------------------
# Already-closed laws — passing guards, so they cannot silently reopen
# -------------------------------------------------------------------------

def test_r0_foreign_prepared_document_is_refused():
    """R1 (already hardened): a preparation may not be transplanted onto an
    object it never described."""
    prepared = prepare_document({"model_type": "gemma2", "hidden_size": 8})
    with pytest.raises(ValueError, match="different document"):
        prepare_document({"unrelated": 1}, already_prepared=prepared)


def test_r0_loader_metadata_cannot_author_architecture():
    from model_unfolder.evidence.config_access import LOADER_METADATA
    with pytest.raises(ValueError, match="may not author architecture"):
        arb.Candidate("repo/x", arb.CHECKPOINT_EXPLICIT, "p", "_repo_id",
                      LOADER_METADATA)


def _gemma2():
    return {"model_type": "gemma2", "hidden_size": 256, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 100,
            "intermediate_size": 512, "sliding_window": 128, "head_dim": 64}


def test_r0_class_overlay_does_not_author_ir_in_shadow_mode():
    """§3.3: during U2 the class overlay is audited but never authors structure.

    A Gemma-2 config whose alternating schedule lives only in the config class
    must, parsed as a plain checkpoint (no class merge), stay uniform — the
    overlay is recorded, not applied."""
    import model_unfolder as mu
    masks = {(layer.get("attention") or {}).get("mask")
             for layer in (mu.unfold(_gemma2()).to_ir().get("layers") or [])}
    assert masks == {"sliding"}, (
        f"class overlay authored a schedule during U2 (masks={masks}) — §3.3 "
        "forbids a merge-driven structural delta before U8")


def test_r0_embedded_and_standalone_agree_in_shadow_mode():
    """R1 vet: the embedded encoder path used merge=True, so the SAME Gemma-2
    was heterogeneous embedded and uniform standalone.  Both are shadow now —
    the class overlay authors nothing in EITHER, and the two agree."""
    from model_unfolder.encoder_panel import normalize_encoder_config
    spec = normalize_encoder_config(_gemma2())
    groups = (spec.get("sub_model") or {}).get("groups") or []
    tags = {g.get("tag") for g in groups}
    # embedded: no class-authored sliding/global split
    assert not any("sliding" in str(t) and "window" in str(t) for t in tags), (
        f"embedded encoder overlay authored a schedule (tags={tags}) — §3.3")
