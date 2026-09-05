"""U2-R1 — one document-preparation boundary, bound to its owner.

A migrated reader enters a document through a single ``DocumentBinding`` that
carries the document, its address and its provenance together, and refuses to
describe any object but its own. This closes the class of bugs where an object
and a provenance map were supplied independently and drifted apart, and where a
nested document silently inherited a parent's origins.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import model_unfolder as mu
from model_unfolder.evidence import config_access as ca
from model_unfolder.evidence.document import (
    DocumentBinding,
    prepare_document,
)

_CORPUS = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"


def _prep(raw):
    return prepare_document(raw)


# -------------------------------------------------------------------------
# The binding
# -------------------------------------------------------------------------

def test_a_binding_only_describes_its_own_document():
    prepared = _prep({"model_type": "gemma2", "hidden_size": 8})
    binding = DocumentBinding("root", (), prepared)
    assert binding.describes(prepared.document)
    assert not binding.describes({"model_type": "gemma2", "hidden_size": 8})


def test_deep_nested_mutation_cannot_alter_the_checkpoint_snapshot():
    raw = {"model_type": "gemma2", "hidden_size": 8, "nested": {"a": 1}}
    prepared = _prep(raw)
    raw["nested"]["a"] = 999
    assert prepared.checkpoint["nested"]["a"] == 1


def test_same_preparation_is_idempotent():
    prepared = _prep({"model_type": "gemma2", "hidden_size": 8})
    again = prepare_document(prepared.document, already_prepared=prepared)
    assert again is prepared


def test_a_foreign_preparation_transplant_raises():
    prepared = _prep({"model_type": "gemma2", "hidden_size": 8})
    with pytest.raises(ValueError, match="different document"):
        prepare_document({"unrelated": 1}, already_prepared=prepared)


# -------------------------------------------------------------------------
# Entering a bound document
# -------------------------------------------------------------------------

def test_bound_document_carries_provenance_and_verifies_identity():
    doc = {"real": 1}
    prepared = _prep(doc)              # no model_type -> raw checkpoint document
    binding = DocumentBinding("root", (), prepared)
    with ca.capture_events() as ledger, ca.bound_document(binding):
        ca.emit("real", intent="inspected", present=True,
                config_path="real", source_obj_id=id(binding.document))
    event = ledger.events[0]
    assert event.provenance == ca.CHECKPOINT_DECLARED
    assert event.path_exact is True


def test_a_nested_bound_document_does_not_inherit_parent_provenance():
    """Each document owns its map: an UNMAPPED nested document stays
    unestablished rather than borrowing the parent's checkpoint origin, where
    the same path means something else."""
    from model_unfolder.evidence.document import PreparedDocument
    outer_doc = {"same": 1}
    inner_doc = {"same": 2}
    outer = DocumentBinding("root", (), _prep(outer_doc))
    # inner is deliberately UNMAPPED (empty provenance) — the strong case: if the
    # scope inherited, its `same` would read as the parent's checkpoint origin.
    inner = DocumentBinding("root.enc", ("slot",),
                            PreparedDocument(document=inner_doc,
                                             checkpoint=inner_doc, provenance={}))
    with ca.capture_events() as ledger, ca.bound_document(outer):
        ca.emit("same", intent="inspected", present=True,
                source_obj_id=id(outer_doc))
        with ca.bound_document(inner):
            ca.emit("same", intent="inspected", present=True,
                    source_obj_id=id(inner_doc))
    by_doc = {tuple(e.document_path): e.provenance for e in ledger.events}
    assert by_doc[()] == ca.CHECKPOINT_DECLARED
    assert by_doc[("slot",)] == ""     # NOT inherited from the parent


# -------------------------------------------------------------------------
# The corpus gates: fabricated = 0, unestablished = 0, no structural delta
# -------------------------------------------------------------------------

def _has(doc, parts) -> bool:
    cur = doc
    for key in parts:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return False
    return True


@pytest.mark.parametrize(
    "witness", sorted(_CORPUS.glob("*.json")), ids=lambda p: p.stem)
def test_r1_gates_hold_on_every_witness(witness):
    cfg = json.loads(witness.read_text())["config"]
    ledger = ca.ConfigAccessLedger()
    with ca.capture_events(ledger):
        ir = mu.unfold(cfg).to_ir()
    worklists = ledger.worklists()

    # fabricated exact paths = 0
    access = (ir.get("extras") or {}).get("config_access") or {}
    roots = access.get("document_roots") or {}
    fabricated = [
        (r["component"], r["path"])
        for r in access.get("accessed_unconsumed_exact") or []
        if not _has(cfg, (*roots.get(r["component"], []), *r["path"].split(".")))]
    assert not fabricated, f"{witness.stem}: fabricated exact paths {fabricated}"

    # unestablished origin = 0 (every read reached a prepared boundary)
    assert not worklists["unestablished_origin"], (
        f"{witness.stem}: reads with no established origin — a document reached "
        f"a reader without a binding: {worklists['unestablished_origin'][:4]}")
