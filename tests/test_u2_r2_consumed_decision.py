"""U2-R2 — value and origin are inseparable on a migrated decision.

``consume_decision`` returns one immutable object carrying the value, its exact
occurrence, its provenance, and the exact target it feeds. A structural reader
no longer goes back to ``provenance_of(path)`` to recover the origin, so it can
never pair one path's value with another path's provenance.
"""
from __future__ import annotations

import pytest

from model_unfolder.evidence import config_access as ca
from test_support import bind_document


def _resolve(doc, canonical, *, provenance=None, path=()):
    prov = provenance or {canonical: ca.CHECKPOINT_DECLARED}
    with ca.bound_document(bind_document(doc, prov)):
        return ca.resolve(doc, canonical, (), path=path)


def _in_scope(doc, provenance):
    return ca.bound_document(bind_document(doc, provenance))


# -------------------------------------------------------------------------
# value + origin are one object
# -------------------------------------------------------------------------

def test_present_decision_carries_value_provenance_and_target():
    doc = {"num_key_value_heads": 4}
    with ca.capture_events() as ledger, _in_scope(
            doc, {"num_key_value_heads": ca.CHECKPOINT_DECLARED}):
        decision = ca.resolve(doc, "num_key_value_heads", ()).consume_decision(
            mechanism="kv", fact_owner="decoder.attention",
            fact_key="num_kv_heads", reader="t")
    assert decision.value == 4
    assert decision.provenance == ca.CHECKPOINT_DECLARED
    assert decision.target.owner == "decoder.attention"
    assert decision.target.fact_key == "num_kv_heads"
    assert decision.occurrence.config_path == "num_key_value_heads"
    # exactly one consumed event, sharing the decision's occurrence
    consumed = [e for e in ledger.events if e.intent == "consumed"]
    assert len(consumed) == 1
    assert consumed[0].config_path == decision.occurrence.config_path


def test_a_class_supplied_read_carries_class_provenance():
    doc = {"n": 8}
    with ca.capture_events(), _in_scope(doc, {"n": ca.CLASS_DEFAULT}):
        decision = ca.resolve(doc, "n", ()).consume_decision(
            mechanism="m", fact_owner="o", fact_key="k", reader="t")
    assert decision.provenance == ca.CLASS_DEFAULT


def test_explicit_null_is_present_and_distinct_from_absent():
    doc = {"n": None}
    with ca.capture_events(), _in_scope(doc, {"n": ca.CHECKPOINT_DECLARED}):
        decision = ca.resolve(doc, "n", ()).consume_decision(
            mechanism="m", fact_owner="o", fact_key="k", reader="t")
    assert decision.state == "present"
    assert decision.value_state == "explicit_null"


def test_absent_returns_a_typed_decision_without_an_occurrence():
    doc = {"other": 1}
    with ca.capture_events(), _in_scope(doc, {}):
        decision = ca.resolve(doc, "n", ()).consume_decision(
            mechanism="m", fact_owner="o", fact_key="k", reader="t")
    assert decision.state == "absent"
    assert decision.occurrence is None
    assert decision.value_state == "missing"


def test_ambiguous_cannot_return_a_structural_value():
    doc = {"a": 1, "b": 2}       # two present accepted aliases, unequal
    with ca.capture_events(), _in_scope(
            doc, {"a": ca.CHECKPOINT_DECLARED, "b": ca.CHECKPOINT_DECLARED}):
        resolution = ca.resolve(doc, "a", ("b",))
        assert resolution.state == "ambiguous"
        with pytest.raises(ValueError, match="ambiguous"):
            resolution.consume_decision(mechanism="m", fact_owner="o",
                                        fact_key="k", reader="t")


# -------------------------------------------------------------------------
# a decision cannot be recombined with a foreign owner/path
# -------------------------------------------------------------------------

def test_sibling_owners_with_the_same_leaf_never_discharge_each_other():
    text_doc = {"hidden_size": 4096}
    vision_doc = {"hidden_size": 1280}
    with ca.capture_events() as ledger:
        with ca.owner_scope("root.text"), _in_scope(
                text_doc, {"hidden_size": ca.CHECKPOINT_DECLARED}):
            ca.resolve(text_doc, "hidden_size", ()).consume_decision(
                mechanism="w", fact_owner="root.text", fact_key="hidden_size",
                reader="t")
        with ca.owner_scope("root.vision"), _in_scope(
                vision_doc, {"hidden_size": ca.CHECKPOINT_DECLARED}):
            ca.resolve(vision_doc, "hidden_size", ()).consume_decision(
                mechanism="w", fact_owner="root.vision", fact_key="hidden_size",
                reader="t")
    consumed = [(e.component, e.fact_owner, e.value_status_hash)
                for e in ledger.events if e.intent == "consumed"]
    owners = {c[1] for c in consumed}
    assert owners == {"root.text", "root.vision"}      # two distinct owners
    # the two are distinct occurrences, neither clears the other
    assert len({(e.component, e.config_path)
                for e in ledger.events if e.intent == "consumed"}) == 2


def test_the_same_path_in_two_documents_is_two_occurrences():
    """U2-R2 vet: the DOCUMENT is part of the occurrence identity.  A pipeline
    root ``hidden_size`` and an embedded encoder ``hidden_size`` are different
    occurrences — merging them would let one document's consumption discharge
    the other's obligation."""
    occs = []
    for doc, dp in (({"hidden_size": 4096}, ()),
                    ({"hidden_size": 1280},
                     ("_text_encoder_configs", "text_encoder"))):
        with ca.capture_events(), ca.bound_document(bind_document(
                doc, {"hidden_size": ca.CHECKPOINT_DECLARED}, path=dp)):
            occs.append(ca.resolve(doc, "hidden_size", ()).consume_decision(
                mechanism="w", fact_owner="o", fact_key="hidden_size",
                reader="r").occurrence)
    assert occs[0] != occs[1]
    assert occs[0].document_path == ()
    assert occs[1].document_path == ("_text_encoder_configs", "text_encoder")


def test_one_source_two_readers_are_two_occurrences():
    """U2-R2 vet: the READER is part of the identity.  One source consumed by
    two readers into two targets is two obligations, not one."""
    doc = {"n": 4}
    occs = []
    for reader in ("reader_x", "reader_y"):
        with ca.capture_events(), _in_scope(doc, {"n": ca.CHECKPOINT_DECLARED}):
            occs.append(ca.resolve(doc, "n", ()).consume_decision(
                mechanism="w", fact_owner="o", fact_key="k",
                reader=reader).occurrence)
    assert occs[0] != occs[1]
    assert {o.reader for o in occs} == {"reader_x", "reader_y"}


def test_the_decision_value_matches_its_own_occurrence_not_a_foreign_path():
    """The R2 point: value and origin come from ONE resolution — there is no
    seam at which a value could acquire a different path's provenance."""
    doc = {"declared": 4, "class_added": 8}
    with ca.capture_events(), _in_scope(
            doc, {"declared": ca.CHECKPOINT_DECLARED,
                  "class_added": ca.CLASS_DEFAULT}):
        d1 = ca.resolve(doc, "declared", ()).consume_decision(
            mechanism="m", fact_owner="o", fact_key="k1", reader="t")
        d2 = ca.resolve(doc, "class_added", ()).consume_decision(
            mechanism="m", fact_owner="o", fact_key="k2", reader="t")
    assert (d1.value, d1.provenance) == (4, ca.CHECKPOINT_DECLARED)
    assert (d2.value, d2.provenance) == (8, ca.CLASS_DEFAULT)
