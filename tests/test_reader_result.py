"""U3-C — exact-owner ReaderResult failure and provenance laws."""
from __future__ import annotations

import pytest

from model_unfolder.evidence.component_owner import (
    ConfigPrefixRival,
    OwnerOccurrenceId,
    OwnerRival,
)
from model_unfolder.evidence.program_index import (
    ConflictRecord,
    ConstructionSiteId,
    SourceId,
    SourceSpan,
    SymbolId,
)
from model_unfolder.evidence.reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
    ReaderValueUnavailable,
    ambiguity_from_conflicts,
)


def _symbol(name="Block"):
    return SymbolId(SourceId("/m.py", "fp", component_key="root"), name)


def _span(line=10):
    return SourceSpan(_symbol().source, line)


def _site(line=10, ordinal=0):
    owner = _symbol("Root")
    span = SourceSpan(owner.source, line)
    return ConstructionSiteId(owner, SymbolId(owner.source, "Root.__init__"),
                              span, ordinal)


def _owner():
    return OwnerOccurrenceId(_symbol("Root"), (_site(),))


def _provenance():
    return (ReaderProvenance("source", spans=(_span(),)),)


def _owner_rivals():
    parent = OwnerOccurrenceId(_symbol("Root"))
    site = _site()
    return (
        OwnerRival(parent, site, _symbol("EagerAttn"), "EagerAttn", "registry"),
        OwnerRival(parent, site, _symbol("FlashAttn"), "FlashAttn", "registry"),
    )


def _prefix_rivals():
    parent = OwnerOccurrenceId(_symbol("Root"))
    site = _site()
    return (
        ConfigPrefixRival(parent, site, "config", ("text_config",)),
        ConfigPrefixRival(parent, site, "config", ("vision_config",)),
    )


def test_resolved_carries_exact_owner_value_and_structured_provenance():
    result = ReaderResult.resolved(
        _owner(), {"kind": "mha"}, provenance=_provenance())
    assert result.ok and result.has_value
    assert result.completeness == "complete"
    assert isinstance(result.owner, OwnerOccurrenceId)
    assert result.require_value() == {"kind": "mha"}


def test_incomplete_requires_partial_value_failure_and_provenance():
    result = ReaderResult.incomplete(
        _owner(), {"kind": "mha"},
        failures=(ReaderFailure("incomplete_graph", "position path unresolved"),),
        provenance=_provenance(),
    )
    assert result.status == "incomplete" and result.has_value
    assert not result.ok and result.require_value()["kind"] == "mha"


def test_absent_is_honest_no_evidence_and_cannot_default():
    result = ReaderResult.absent(_owner())
    assert result.status == "absent" and not result.has_value
    with pytest.raises(ReaderValueUnavailable):
        result.require_value()
    assert not hasattr(result, "value_or")


def test_failed_carries_closed_typed_failure():
    result = ReaderResult.failed(
        _owner(), (ReaderFailure("parse_failure", "invalid syntax", _span()),))
    assert result.status == "failed" and not result.has_value
    with pytest.raises(ReaderValueUnavailable):
        result.require_value()


def test_ambiguous_carries_exact_rivals_and_no_value():
    ambiguity = Ambiguity(rival_owner_chains=(_owner_rivals(),))
    result = ReaderResult.ambiguous(_owner(), ambiguity)
    assert result.status == "ambiguous" and result.ambiguity is ambiguity
    with pytest.raises(ReaderValueUnavailable):
        result.require_value()


def test_ambiguity_can_be_proven_by_exact_rival_sites_without_owner_guessing():
    ambiguity = Ambiguity(sites=(_span(10), _span(20)))
    result = ReaderResult.ambiguous(_owner(), ambiguity)
    assert result.status == "ambiguous"
    assert result.ambiguity.sites == (_span(10), _span(20))


@pytest.mark.parametrize("status", ["maybe", "proven", "unknown"])
def test_unknown_status_rejected(status):
    with pytest.raises(ValueError):
        ReaderResult(status=status)


def test_resolved_requires_occurrence_owner():
    with pytest.raises(TypeError):
        ReaderResult.resolved(_symbol(), {"x": 1}, provenance=_provenance())


def test_resolved_requires_value():
    with pytest.raises(ValueError):
        ReaderResult("resolved", _owner(), None, "complete",
                     provenance=_provenance())


def test_resolved_requires_structured_provenance():
    with pytest.raises(ValueError):
        ReaderResult.resolved(_owner(), {"x": 1}, provenance=())
    with pytest.raises(TypeError):
        ReaderResult.resolved(_owner(), {"x": 1}, provenance=("ast",))


def test_resolved_cannot_hide_failure_or_ambiguity():
    with pytest.raises(ValueError):
        ReaderResult("resolved", _owner(), {"x": 1}, "complete",
                     failures=(ReaderFailure("incomplete_graph", "missing"),),
                     provenance=_provenance())
    with pytest.raises(ValueError):
        ReaderResult("resolved", _owner(), {"x": 1}, "complete",
                     provenance=_provenance(),
                     ambiguity=Ambiguity(rival_owner_chains=(_owner_rivals(),)))


def test_incomplete_requires_failure_and_provenance():
    with pytest.raises(ValueError):
        ReaderResult("incomplete", _owner(), {"x": 1}, "partial",
                     provenance=_provenance())
    with pytest.raises(ValueError):
        ReaderResult("incomplete", _owner(), {"x": 1}, "partial",
                     failures=(ReaderFailure("incomplete_graph", "missing"),))


def test_incomplete_cannot_also_be_ambiguous():
    with pytest.raises(ValueError):
        ReaderResult(
            "incomplete", _owner(), {"x": 1}, "partial",
            failures=(ReaderFailure("conflict", "rivals"),),
            provenance=_provenance(),
            ambiguity=Ambiguity(rival_owner_chains=(_owner_rivals(),)),
        )


def test_ambiguous_requires_owner_and_at_least_two_typed_rivals():
    with pytest.raises(ValueError):
        ReaderResult.ambiguous(None,
                               Ambiguity(rival_owner_chains=(_owner_rivals(),)))
    with pytest.raises(ValueError):
        Ambiguity(rival_owner_chains=((_owner_rivals()[0],),))
    with pytest.raises(ValueError):
        Ambiguity(rival_owner_chains=(("a", "b"),))


def test_ambiguous_cannot_carry_value_or_failure():
    ambiguity = Ambiguity(rival_config_prefixes=(_prefix_rivals(),))
    with pytest.raises(ValueError):
        ReaderResult("ambiguous", _owner(), {"x": 1}, ambiguity=ambiguity)
    with pytest.raises(ValueError):
        ReaderResult("ambiguous", _owner(), failures=(
            ReaderFailure("conflict", "rivals"),), ambiguity=ambiguity)


def test_failed_requires_failure_and_rejects_unknown_failure_kind():
    with pytest.raises(ValueError):
        ReaderResult.failed(_owner(), ())
    with pytest.raises(ValueError):
        ReaderFailure("whatever", "not a closed kind")
    with pytest.raises(ValueError):
        ReaderFailure("parse_failure", "")


def test_absent_cannot_carry_value_failure_or_ambiguity():
    with pytest.raises(ValueError):
        ReaderResult("absent", _owner(), {"x": 1})
    with pytest.raises(ValueError):
        ReaderResult("absent", _owner(), failures=(
            ReaderFailure("missing_source", "none"),))
    with pytest.raises(ValueError):
        ReaderResult("absent", _owner(),
                     ambiguity=Ambiguity(rival_owner_chains=(_owner_rivals(),)))


def test_provenance_is_structural_and_nonempty():
    with pytest.raises(ValueError):
        ReaderProvenance("source")
    with pytest.raises(ValueError):
        ReaderProvenance("invented", detail="x")
    with pytest.raises(TypeError):
        ReaderProvenance("source", config_paths=(["hidden_size"],))
    config = ReaderProvenance(
        "code_and_config", spans=(_span(),),
        config_paths=(("text_config", "hidden_size"),))
    assert config.config_paths[0][-1] == "hidden_size"


def test_each_provenance_kind_requires_its_real_evidence_channel():
    with pytest.raises(ValueError):
        ReaderProvenance("source", detail="a label is not a source span")
    with pytest.raises(ValueError):
        ReaderProvenance("external", detail="module name only")
    with pytest.raises(ValueError):
        ReaderProvenance("config", detail="field name only")
    with pytest.raises(ValueError):
        ReaderProvenance("code_and_config", spans=(_span(),))
    with pytest.raises(ValueError):
        ReaderProvenance("code_and_config",
                         config_paths=(("hidden_size",),))
    with pytest.raises(ValueError):
        ReaderProvenance("derived", spans=(_span(),))
    assert ReaderProvenance("derived", detail="premises: attention.kind")


def test_ambiguity_from_conflicts_preserves_exact_records():
    owner_conflict = ConflictRecord(
        "rival_owner_chain", _owner_rivals(), (_span(),))
    prefix_conflict = ConflictRecord(
        "rival_config_prefix", _prefix_rivals(), (_span(),))
    ambiguity = ambiguity_from_conflicts((owner_conflict, prefix_conflict))
    assert ambiguity.rival_owner_chains == (_owner_rivals(),)
    assert ambiguity.rival_config_prefixes == (_prefix_rivals(),)
    assert ambiguity.sites == (_span(),)
    result = ReaderResult.ambiguous(_owner(), ambiguity)
    assert result.status == "ambiguous"


def test_ambiguity_bridge_rejects_untyped_or_unknown_conflicts():
    with pytest.raises(TypeError):
        ambiguity_from_conflicts(("not a conflict",))
    with pytest.raises(ValueError):
        ambiguity_from_conflicts((ConflictRecord("other", (), ()),))
    with pytest.raises(ValueError):
        ambiguity_from_conflicts((
            ConflictRecord("rival_owner_chain", (("a",), ("b",)), (_span(),)),
        ))
