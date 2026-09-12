"""Whole schedules bind a real reader and the exact repeated-child count."""
from dataclasses import replace
import copy
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.facts import EvidenceFact, SourceSpan
from model_unfolder.evidence.reader_claims import (
    ReaderProjectionClaimProof, qualify_reader_fact,
)
from model_unfolder.evidence.receipts import value_status_hash
from test_support.s9_fixtures.position_linear_bias import _pipeline, _source


def _reader_case(tmp_path, family, raw=None):
    if family == "additive":
        from model_unfolder.evidence.cross_attention_schedule import decoder_cross_attention_all_layers_for_path
        from test_support.s9_fixtures.cross_attention_schedule import _SOURCE
        index, bundle = _pipeline(tmp_path, raw=raw or _SOURCE, architecture="Shell")
        return index, bundle, decoder_cross_attention_all_layers_for_path, "cross_attention_schedule"
    from model_unfolder.evidence.position_linear_bias import decoder_alibi_score_bias_for_path
    index, bundle = _pipeline(tmp_path, raw=raw)
    return index, bundle, decoder_alibi_score_bias_for_path, "position_schedule"


def _fact(result, key, value, status, paths=()):
    return EvidenceFact(key=key, owner="decoder.attention", value=value,
        status=status, completeness="complete", config_paths=tuple(".".join(p) for p in paths),
        source_spans=tuple(dict.fromkeys(SourceSpan(
            component=span.source.component_key or "root",
            file=span.source.canonical_path, line=span.line)
            for origin in result.provenance for span in origin.spans)))


@pytest.mark.parametrize("family", ["alibi", "additive"])
@pytest.mark.parametrize("count", [3, 7])
@pytest.mark.parametrize("defaulted", [False, True])
def test_actual_retained_schedule_qualifies_its_exact_count_and_tier(tmp_path, family, count, defaulted):
    index, bundle, reader, key = _reader_case(tmp_path, family)
    checkpoint = {} if defaulted else {"layers": count}
    document = PreparedDocument(copy.deepcopy(checkpoint), copy.deepcopy(checkpoint),
        class_overlay={"layers": count} if defaulted else {})
    with bound_document(DocumentBinding("root", (), document)):
        result = reader(index, bundle, (), allow_root_stage=True)
        assert result.status == "resolved", result.failures
        proof = ReaderProjectionClaimProof(f"decoder.attention.{key}", result, index, document)
        projected = proof.projection()
        assert len(projected.value) == count
        assert projected.status == ("class_default" if defaulted else "code_and_config")
        assert projected.config_paths == (() if defaulted else (("layers",),))
        fact = qualify_reader_fact(_fact(result, key, projected.value,
            projected.status, projected.config_paths), result, index, document)
        validate_fact_claim(fact, fact.claim_evidence)
        assert fact.claim_kind == "relation"
        assert fact.claim_readers == (result.claim_reader_symbol,)
        assert fact.claim_document_token
        if family == "alibi":
            assert all(row == {"position_kind": "alibi", "position_application": "attention_bias",
                               "rope_dim": None} for row in fact.value)
        else:
            assert fact.value == ("additive_cross",) * count
        # Same mechanism or a known count does not authorize another schedule/tier/path.
        for mutation in ({"value": fact.value[:-1]}, {"status": "code_proven"},
                         {"config_paths": ("foreign_count",)}):
            with pytest.raises(ValueError):
                validate_fact_claim(replace(fact, **mutation), fact.claim_evidence)
        with pytest.raises(ValueError, match="actual result"):
            ReaderProjectionClaimProof(fact.ledger_key(), replace(result), index, document)
        other = PreparedDocument(copy.deepcopy(checkpoint), copy.deepcopy(checkpoint),
            class_overlay={"layers": count} if defaulted else {})
        with pytest.raises(ValueError, match="prepared document"):
            ReaderProjectionClaimProof(fact.ledger_key(), result, index, other)


@pytest.mark.parametrize("family", ["alibi", "additive"])
def test_unbound_schedule_reader_declares_kind_without_count_proof(tmp_path, family):
    index, bundle, reader, key = _reader_case(tmp_path, family)
    result = reader(index, bundle, (), allow_root_stage=True)
    assert result.status == "resolved"
    value = (("additive_cross",) * 3 if family == "additive" else
             tuple({"position_kind": "alibi", "position_application": "attention_bias",
                    "rope_dim": None} for _ in range(3)))
    fact = qualify_reader_fact(_fact(result, key, value, "code_proven"), result, index, None)
    assert fact.claim_kind == "relation"
    assert fact.claim_evidence is None and not fact.claim_document_token
    with pytest.raises(ValueError, match="prepared document"):
        ReaderProjectionClaimProof(fact.ledger_key(), result, index, None)


@pytest.mark.parametrize("mutation", ["missing", "arithmetic", "shadow", "conflicting_count"])
def test_unavailable_repetition_never_qualifies_an_existing_schedule(tmp_path, mutation):
    raw = _source()
    if mutation == "arithmetic":
        raw = raw.replace("range(config.layers)", "range(config.layers + 1)")
    if mutation == "shadow":
        raw += "\nrange = lambda count: (0,)\n"
    index, bundle, reader, key = _reader_case(tmp_path, "alibi", raw)
    checkpoint = {} if mutation == "missing" else {"layers": 0 if mutation == "conflicting_count" else 3}
    document = PreparedDocument(copy.deepcopy(checkpoint), copy.deepcopy(checkpoint))
    with bound_document(DocumentBinding("root", (), document)):
        result = reader(index, bundle, (), allow_root_stage=True)
        assert result.status == "resolved", result.failures
        value = tuple({"position_kind": "alibi", "position_application": "attention_bias",
                       "rope_dim": None} for _ in range(3))
        fact = qualify_reader_fact(_fact(result, key, value, "code_proven"), result, index, document)
        assert fact.value == value and fact.claim_kind == "relation"
        assert fact.claim_evidence is None and not fact.claim_document_token
        assert fact.unknown_reason is None  # no manufactured exhaustion/class3


@pytest.mark.parametrize("slug,key,count,path", [
    ("bloom", "position_schedule", 70, "n_layer"),
    ("musicgen-small", "cross_attention_schedule", 24, "decoder.num_hidden_layers"),
])
def test_actual_corpus_schedule_count_supplies_fact_and_consumption_hash(slug, key, count, path):
    from model_unfolder import config_to_ir
    from model_unfolder.evidence.context import ParseContext
    cfg = json.loads((Path(__file__).parent / "sable_test_corpus" / (slug + ".json")).read_text())["config"]
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    assert len(ir.layers) == count
    fact = context.facts.typed[f"decoder.attention.{key}"]
    assert fact.status == "code_and_config" and fact.config_paths == (path,)
    assert fact.claim_kind == "relation" and fact.claim_evidence is not None
    validate_fact_claim(fact, fact.claim_evidence)
    assert len(fact.value) == count
    if key == "position_schedule":
        assert all(layer.attention.position_kind == "alibi" and
                   layer.attention.position_application == "attention_bias" for layer in ir.layers)
    else:
        assert fact.value == ("additive_cross",) * count
    decisions = [event for event in context.config_access.events
                 if event.intent == "consumed" and event.fact_owner == "decoder.attention"
                 and event.fact_key == key and event.mechanism == key]
    assert len(decisions) == 1
    assert decisions[0].config_path == path
    assert decisions[0].value_status_hash == value_status_hash(fact.value, fact.status)


def test_actual_nested_class_default_count_keeps_the_schedule_and_weaker_tier():
    from model_unfolder import config_to_ir
    from model_unfolder.evidence.context import ParseContext
    cfg = json.loads((Path(__file__).parent / "sable_test_corpus" / "musicgen-small.json").read_text())["config"]
    assert cfg["decoder"].pop("num_hidden_layers") == 24
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    fact = context.facts.typed["decoder.attention.cross_attention_schedule"]
    assert len(ir.layers) == 24 and fact.value == ("additive_cross",) * 24
    assert fact.status == "class_default" and fact.config_paths == ()
    validate_fact_claim(fact, fact.claim_evidence)
    decisions = [event for event in context.config_access.events
                 if event.intent == "absent_default" and event.fact_owner == "decoder.attention"
                 and event.fact_key == "cross_attention_schedule"
                 and event.mechanism == "cross_attention_schedule"]
    assert len(decisions) == 1
    assert decisions[0].config_path == "decoder.num_hidden_layers"
    assert decisions[0].provenance == "class_default" and not decisions[0].present
    assert decisions[0].value_status_hash == value_status_hash(fact.value, fact.status)
