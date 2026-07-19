"""U3-C — the generic ReaderResult[T] substrate and its failure laws.

A migrated reader may never return a bare value or a bare None/False; it returns
a ReaderResult whose status makes "no evidence", "ambiguous", "incomplete" and
"failed" distinguishable. These tests pin the laws so an ill-formed result cannot
be constructed, and prove the wrapper is generic over the real domain evidence
dataclasses and bridges the resolver's typed conflicts.
"""
from __future__ import annotations

import pytest

from model_unfolder.evidence.program_index import (
    ConflictRecord,
    SourceId,
    SourceSpan,
    SymbolId,
)
from model_unfolder.evidence.reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderResult,
    ambiguity_from_conflicts,
)


def _owner():
    sid = SourceId("/m.py", "fp", component_key="root")
    return SymbolId(sid, "Block.attention")


# --------------------------------------------------------------------------- #
# Lawful constructors produce the right shape
# --------------------------------------------------------------------------- #

def test_resolved_carries_value_and_is_complete():
    r = ReaderResult.resolved(_owner(), value={"kind": "mha"}, provenance=("ast",))
    assert r.ok and r.has_value and r.status == "resolved"
    assert r.completeness == "complete" and r.ambiguity is None
    assert r.provenance == ("ast",)


def test_incomplete_carries_partial_value():
    r = ReaderResult.incomplete(_owner(), value={"kind": "mha"},
                                failures=(ReaderFailure("missing_source", "no rope"),))
    assert r.status == "incomplete" and r.completeness == "partial"
    assert r.has_value and not r.ok


def test_absent_is_honest_no_evidence():
    r = ReaderResult.absent(_owner())
    assert r.status == "absent" and not r.has_value and r.completeness == "none"
    assert r.value_or("dflt") == "dflt"


def test_failed_requires_typed_failures():
    r = ReaderResult.failed(_owner(), failures=(ReaderFailure("parse_failure", "boom"),))
    assert r.status == "failed" and not r.has_value and r.failures


def test_ambiguous_carries_rivals_and_no_value():
    amb = Ambiguity(rival_owner_chains=((("EagerAttn", "registry"),
                                         ("FlashAttn", "registry")),))
    r = ReaderResult.ambiguous(_owner(), amb)
    assert r.status == "ambiguous" and not r.has_value
    assert r.ambiguity is amb


# --------------------------------------------------------------------------- #
# The failure LAWS reject ill-formed results at construction
# --------------------------------------------------------------------------- #

def test_unknown_status_rejected():
    with pytest.raises(ValueError):
        ReaderResult(status="maybe")


def test_resolved_without_value_rejected():
    with pytest.raises(ValueError):
        ReaderResult(status="resolved", value=None, completeness="complete")


def test_resolved_with_ambiguity_rejected():
    with pytest.raises(ValueError):
        ReaderResult(status="resolved", value={"x": 1}, completeness="complete",
                     ambiguity=Ambiguity(rival_config_prefixes=((("a",), ("b",)),)))


def test_ambiguous_without_ambiguity_rejected():
    with pytest.raises(ValueError):
        ReaderResult(status="ambiguous", value=None)


def test_ambiguous_with_empty_ambiguity_rejected():
    with pytest.raises(ValueError):
        ReaderResult(status="ambiguous", ambiguity=Ambiguity())


def test_ambiguous_with_value_rejected():
    amb = Ambiguity(rival_config_prefixes=((("a",), ("b",)),))
    with pytest.raises(ValueError):
        ReaderResult(status="ambiguous", value={"x": 1}, ambiguity=amb)


def test_failed_without_failures_rejected():
    with pytest.raises(ValueError):
        ReaderResult(status="failed", failures=())


def test_absent_with_value_rejected():
    with pytest.raises(ValueError):
        ReaderResult(status="absent", value={"x": 1})


def test_incomplete_must_be_partial():
    with pytest.raises(ValueError):
        ReaderResult(status="incomplete", value={"x": 1}, completeness="complete")


def test_failures_must_be_typed():
    with pytest.raises(TypeError):
        ReaderResult(status="failed", failures=("just a string",))


# --------------------------------------------------------------------------- #
# Generic over the real domain evidence dataclasses
# --------------------------------------------------------------------------- #

def test_wraps_a_real_domain_evidence_dataclass():
    from model_unfolder.evidence.models import PositionalEvidence
    ev = PositionalEvidence(status="proven")
    r: ReaderResult[PositionalEvidence] = ReaderResult.resolved(_owner(), ev)
    assert r.ok and isinstance(r.value, PositionalEvidence)
    assert r.value.status == "proven"


# --------------------------------------------------------------------------- #
# The U3-B -> reader bridge: build an Ambiguity from typed ConflictRecords
# --------------------------------------------------------------------------- #

def test_ambiguity_from_conflicts_bridge():
    sid = SourceId("/m.py", "fp", component_key="root")
    span = SourceSpan(sid, 10)
    conflicts = (
        ConflictRecord("rival_owner_chain",
                       (("EagerAttn", "registry"), ("FlashAttn", "registry")),
                       (span,)),
        ConflictRecord("rival_config_prefix", (("text_config",), ("vision_config",)),
                       (span,)),
    )
    amb = ambiguity_from_conflicts(conflicts)
    assert len(amb.rival_owner_chains) == 1
    assert len(amb.rival_config_prefixes) == 1
    assert span in amb.sites and len(amb.conflicts) == 2
    # and it can be handed straight into an ambiguous result
    r = ReaderResult.ambiguous(_owner(), amb)
    assert r.status == "ambiguous"


def test_ambiguity_from_conflicts_rejects_non_conflicts():
    with pytest.raises(TypeError):
        ambiguity_from_conflicts(["not a conflict record"])
