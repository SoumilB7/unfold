"""H1 — typed evidence primitives and the negative-proof law.

Covers the plan's H1.6 matrix: proven true; proven false with complete
inspection; illegal false from presence-only inspection; ambiguous; oracle
missing; derived strength no greater than premises — plus the H1.2 type
boundary (observations are not facts), the H1.5 truthiness ban, exact
round-trip representability of every current FactLedger record shape, and
the single-author ledger extension.
"""
from __future__ import annotations

import pytest

from model_unfolder.evidence.context import FACT_STATUSES, FactLedger, FactRecord
from model_unfolder.evidence.facts import (
    EvidenceFact,
    NegativeProofError,
    BoundObservation,
    RawObservation,
    SourceSpan,
    is_negative_value,
    resolve_strength,
    strength_of,
)


def _fact(**kw) -> EvidenceFact:
    base = dict(key="gated", owner="root.decoder.ffn", value=True,
                status="code_proven", completeness="presence_only")
    base.update(kw)
    return EvidenceFact(**base)


# --- the negative-proof law (I-3) -------------------------------------------

def test_proven_true_needs_only_presence():
    fact = _fact(value=True, completeness="presence_only")
    assert fact.value is True and not fact.is_negative()


def test_proven_false_with_complete_inspection_is_lawful():
    fact = _fact(value=False, completeness="complete")
    assert fact.is_negative()


@pytest.mark.parametrize("value", [False, None, "none"])
@pytest.mark.parametrize("status", ["code_proven", "code_and_config"])
@pytest.mark.parametrize("completeness", ["presence_only", "partial", "uninspected"])
def test_code_claimed_negative_from_incomplete_inspection_is_illegal(
        value, status, completeness):
    with pytest.raises(NegativeProofError):
        _fact(value=value, status=status, completeness=completeness)


def test_config_declared_negative_is_checkpoint_truth_not_inspection_claim():
    fact = _fact(value=False, status="config_declared", completeness="uninspected")
    assert fact.is_negative()


def test_negative_vocabulary():
    assert is_negative_value(False) and is_negative_value(None)
    assert is_negative_value("none") and is_negative_value("Absent")
    assert not is_negative_value(0) and not is_negative_value("")
    assert not is_negative_value("nope-like-string")


# --- typed failures (I-9) ----------------------------------------------------

def test_ambiguous_and_oracle_missing_carry_typed_failures():
    amb = _fact(value=None, status="ambiguous", completeness="partial",
                failure_kind="unsupported_syntax")
    missing = _fact(value=None, status="oracle_missing",
                    failure_kind="source_missing")
    assert amb.failure_kind == "unsupported_syntax"
    assert missing.failure_kind == "source_missing"


def test_failure_kind_cannot_decorate_a_proven_claim():
    with pytest.raises(ValueError):
        _fact(value=True, status="code_proven", failure_kind="reader_error")
    with pytest.raises(ValueError):
        _fact(value=None, status="ambiguous", failure_kind="not-a-kind")


# --- derived facts and strength (I-5) ----------------------------------------

def test_derived_requires_premises_and_only_derived_carries_them():
    with pytest.raises(ValueError):
        _fact(status="derived", premises=())
    with pytest.raises(ValueError):
        _fact(status="code_proven", value=True, premises=("root.x.y",))
    with pytest.raises(ValueError):
        EvidenceFact.derive("combined", "root", True, premises=[])


def test_derived_strength_is_the_weakest_premise():
    strong = _fact(key="a", value=True)                              # code_proven: 5
    weak = _fact(key="b", value="silu", status="class_default",
                 completeness="uninspected")                          # 2
    derived = EvidenceFact.derive("c", "root.decoder.ffn", True,
                                  premises=[strong, weak])
    facts = {f.ledger_key(): f for f in (strong, weak, derived)}
    assert resolve_strength(derived.ledger_key(), facts) == strength_of("class_default")
    assert resolve_strength(strong.ledger_key(), facts) == strength_of("code_proven")


def test_derived_strength_collapses_on_missing_premise_or_cycle():
    orphan = EvidenceFact(key="c", owner="root", value=1, status="derived",
                          premises=("root.gone",))
    assert resolve_strength("root.c", {"root.c": orphan}) == 0

    a = EvidenceFact(key="a", owner="root", value=1, status="derived",
                     premises=("root.b",))
    b = EvidenceFact(key="b", owner="root", value=1, status="derived",
                     premises=("root.a",))
    facts = {"root.a": a, "root.b": b}
    assert resolve_strength("root.a", facts) == 0
    assert resolve_strength("root.b", facts) == 0


def test_strength_of_refuses_derived():
    with pytest.raises(ValueError):
        strength_of("derived")


# --- the type boundary (H1.2) and truthiness ban (H1.5) -----------------------

def test_observations_are_not_facts():
    raw = RawObservation(kind="call", token="chunk",
                         span=SourceSpan(component="decoder", file="m.py", line=7))
    bound = BoundObservation(observation=raw, owner="root.decoder.ffn",
                             via="construction")
    with pytest.raises(TypeError):
        _fact(value=raw)
    with pytest.raises(TypeError):
        _fact(value=bound)


def test_facts_are_not_truthy():
    fact = _fact(value=False, completeness="complete")
    with pytest.raises(TypeError):
        bool(fact)
    with pytest.raises(TypeError):
        _ = fact or "default"   # the exact idiom the ban exists to break


# --- representability: every current ledger record round-trips (H1.1) --------

@pytest.mark.parametrize("status", sorted(FACT_STATUSES))
@pytest.mark.parametrize("value", [True, False, None, "silu", 4096, 0,
                                   ["a", "b"], {"k": 1}])
@pytest.mark.parametrize("source", [None, "decoder_ffn_mechanism_for_path",
                                    "config:hidden_act", "file.py:123"])
def test_every_legacy_record_shape_round_trips_exactly(status, value, source):
    record = FactRecord(value, status, source)
    for key in ("layers[3].attention.qk_norm", "decoder.ffn.gated", "mask"):
        lifted = EvidenceFact.from_record(key, record)
        assert lifted.to_record() == record
        assert lifted.ledger_key() == key


def test_lifted_premiseless_derived_rows_carry_the_sentinel_and_zero_strength():
    """Legacy 'derived' rows recorded no premises (eps-spelling norm tier).
    The lift marks them with the explicit sentinel — countable H2 debt — and
    I-5 resolution gives them strength 0, never an invented rank."""
    from model_unfolder.evidence.facts import LEGACY_UNRECORDED_PREMISE

    record = FactRecord("rmsnorm", "derived", "eps-spelling rms_norm_eps")
    lifted = EvidenceFact.from_record("decoder.layer.norm_kind", record)
    assert lifted.premises == (LEGACY_UNRECORDED_PREMISE,)
    assert lifted.to_record() == record
    assert resolve_strength("decoder.layer.norm_kind",
                            {"decoder.layer.norm_kind": lifted}) == 0


def test_lift_never_manufactures_complete_and_marks_migrated_debt():
    """H1 item-1 amendment: a legacy code-negative lift must NOT invent
    completeness='complete' from its status.  It stays uninspected, is flagged
    migrated_legacy (exempt from the native negative-proof law), and is counted
    as debt."""
    from model_unfolder.evidence.facts import migrated_legacy_debt

    lifted = EvidenceFact.from_record(
        "decoder.attention.sinks", FactRecord(False, "code_proven", "reader"))
    assert lifted.completeness == "uninspected"      # NOT "complete"
    assert lifted.migrated_legacy is True
    assert lifted.to_record() == FactRecord(False, "code_proven", "reader")

    positive = EvidenceFact.from_record(
        "decoder.ffn.gated", FactRecord(True, "code_proven", "reader"))
    assert positive.completeness == "uninspected"
    assert positive.migrated_legacy is False          # positive: no law to escape

    debt = migrated_legacy_debt({lifted.ledger_key(): lifted,
                                 positive.ledger_key(): positive})
    assert debt == ("decoder.attention.sinks",)


def test_native_code_negative_still_must_prove_complete():
    """The native negative-proof law is UNCHANGED for non-lifted construction:
    a code-negative built natively without complete inspection still raises."""
    with pytest.raises(NegativeProofError):
        EvidenceFact(key="sinks", owner="decoder.attention", value=False,
                     status="code_proven", completeness="uninspected")


def test_public_legacy_bypass_is_impossible():
    """§16.3 item 1: ``migrated_legacy`` is NOT a public initializer argument, so
    a native caller cannot construct a law-exempt fact.  The ONLY law-exempt path
    is the internal lift, reached through ``from_record``."""
    with pytest.raises(TypeError):
        EvidenceFact(key="sinks", owner="decoder.attention", value=False,
                     status="code_proven", completeness="uninspected",
                     migrated_legacy=True)  # type: ignore[call-arg]
    # the sole lawful lift path — a code-negative from a legacy record — succeeds
    # and is flagged as migration debt, never as a native fact.
    lifted = EvidenceFact.from_record(
        "decoder.attention.sinks", FactRecord(False, "code_proven", "reader"))
    assert lifted.migrated_legacy is True
    assert lifted.completeness == "uninspected"


def test_derived_false_from_presence_only_raises():
    """§16.3 item 2: a derived NEGATIVE from a presence-only premise cannot prove
    absence, so ``derive()`` raises; the same derivation from a COMPLETE premise
    is lawful."""
    presence = EvidenceFact(key="a", owner="root.decoder.ffn", value=True,
                            status="code_proven", completeness="presence_only")
    with pytest.raises(NegativeProofError):
        EvidenceFact.derive("gated", "root.decoder.ffn", False, premises=[presence])

    complete = EvidenceFact(key="a", owner="root.decoder.ffn", value=True,
                            status="code_proven", completeness="complete")
    ok = EvidenceFact.derive("gated", "root.decoder.ffn", False, premises=[complete])
    assert ok.is_negative() and ok.completeness == "complete"


def test_reason_and_source_do_not_alias():
    """§16.3 item 3: a human ``reason`` is NEVER serialized as source provenance;
    the flat serialization source comes only from the stable ``legacy_source``
    label."""
    with_both = _fact(reason="because the forward gates via chunk",
                      legacy_source="decoder_ffn_mechanism_for_path")
    assert with_both.to_record().source == "decoder_ffn_mechanism_for_path"

    reason_only = _fact(reason="a human note, not provenance")
    assert reason_only.to_record().source is None


def test_native_source_spans_survive_typed_ledger_storage():
    """§16.3 item 4: structured SourceSpan / config paths survive the typed
    channel — record a native fact carrying spans, read it back, spans intact."""
    span = SourceSpan(component="decoder", class_name="LlamaMLP",
                      method="forward", file="modeling_llama.py", line=210)
    native = _fact(key="gated", owner="decoder.ffn", value=True,
                   completeness="presence_only", source_spans=(span,),
                   config_paths=("hidden_act",))
    ledger = FactLedger()
    ledger.record_typed(native)
    back = ledger.typed_records()["decoder.ffn.gated"]
    assert back.source_spans == (span,)
    assert back.config_paths == ("hidden_act",)


def test_legacy_asserted_serializes_as_live_asserted_spelling():
    fact = _fact(value="split", status="legacy_asserted")
    assert fact.to_record().status == "asserted"


# --- ledger extension: one author, stable serialization ----------------------

def test_record_typed_single_author_and_stable_serialization():
    legacy = FactLedger()
    legacy.record("decoder.ffn", "gated", True, "code_proven", "reader")

    typed = FactLedger()
    typed.record_typed(_fact(key="gated", owner="decoder.ffn", value=True,
                             status="code_proven", completeness="presence_only",
                             legacy_source="reader"))

    assert typed.to_dict() == legacy.to_dict()
    assert typed.records["decoder.ffn.gated"] == \
        typed.typed["decoder.ffn.gated"].to_record()

    with pytest.raises(TypeError):
        typed.record_typed({"not": "a fact"})


def test_typed_records_lifts_legacy_rows_and_keeps_native_ones():
    ledger = FactLedger()
    ledger.record("decoder.ffn", "activation", "silu", "class_default", None)
    native = _fact(key="gated", owner="decoder.ffn", value=True)
    ledger.record_typed(native)

    lifted = ledger.typed_records()
    assert lifted["decoder.ffn.gated"] is native
    assert lifted["decoder.ffn.activation"].status == "class_default"
    assert ledger.strength("decoder.ffn.gated") == strength_of("code_proven")


def test_asserted_census_unaffected_by_typed_channel():
    ledger = FactLedger()
    ledger.record("decoder.ffn", "ffn_storage", "split", "asserted", None)
    ledger.record_typed(_fact(key="projection_mode", owner="decoder.attention",
                              value="split", status="legacy_asserted"))
    assert ledger.asserted() == ("decoder.attention.projection_mode",
                                 "decoder.ffn.ffn_storage")


# --- integration: a real parse's whole ledger is representable ----------------

def test_real_parse_ledger_is_fully_representable_and_round_trips():
    from model_unfolder import unfold
    from model_unfolder.sable import load_corpus

    config = next(fix["config"] for name, fix in load_corpus()
                  if "llama-7b" in name)
    ir = unfold(config).ir
    provenance = (ir.extras or {}).get("fact_provenance") or {}
    assert provenance, "llama fixture produced no fact_provenance"

    for key, row in provenance.items():
        record = FactRecord(row["value"], row["status"], row.get("source"))
        lifted = EvidenceFact.from_record(key, record)
        assert lifted.to_record() == record, key
