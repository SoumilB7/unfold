"""Scalar typed decisions preserve their supplying value, path and event tier."""

from test_support.s9_fixtures.s9_scalar_decisions import _config, _parse
from dataclasses import replace

import pytest

from model_unfolder.evidence import config_access
from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.receipts import value_status_hash






def _decisions(events, key):
    return [event for event in events if event.fact_owner == "model"
            and event.fact_key == key and event.intent in {"consumed", "absent_default"}]


@pytest.mark.parametrize("spelling", ["hidden_size", "d_model"])
def test_explicit_scalar_decisions_keep_exact_alias_and_legacy_tie_event(spelling):
    checkpoint = _config()
    checkpoint[spelling] = checkpoint.pop("hidden_size")
    ir, facts, events = _parse(checkpoint)
    assert ir.hidden_size == 64 and ir.tie_word_embeddings is False
    for key, path, value, reader, mechanism in (
        ("hidden_size", spelling, 64, "adapters.transformer.parser.parse", "model_hidden_size_value"),
        ("tie_word_embeddings", "tie_word_embeddings", False,
         "model_unfolder.adapters.transformer.parser.parse", ""),
    ):
        fact = facts["model." + key]
        assert fact.value == value and fact.status == "config_declared"
        assert fact.claim_kind == "value" and fact.config_paths == (path,)
        validate_fact_claim(fact, fact.claim_evidence)
        decisions = _decisions(events, key)
        assert len(decisions) == 1
        event = decisions[0]
        assert event.config_path == path and event.path_exact and event.present
        assert event.reader == reader and event.mechanism == mechanism
        assert event.intent == "consumed" and event.value_state == "value"
        assert event.provenance == "checkpoint_declared"
        assert event.document_fingerprint and event.document_token
        assert event.value_status_hash == value_status_hash(value, "config_declared")


def test_sparse_scalar_uses_source_default_without_inventing_checkpoint_path(tmp_path):
    from test_support.s9_fixtures.s9_class_default_value import _read

    checkpoint = _config()
    del checkpoint["hidden_size"]
    result, index, document = _read(tmp_path, checkpoint=checkpoint)
    assert result.status == "resolved"
    defaults = {"hidden_size": 64}
    context = ParseContext(result.claim_witness.bundle, class_defaults=defaults,
                           class_defaults_by_path={(): defaults})
    context._program_index = index
    ir, facts, events = _parse(checkpoint, context=context, document=document)
    fact = facts["model.hidden_size"]
    assert ir.hidden_size == fact.value == 64
    assert fact.status == "class_default" and fact.claim_kind == "value"
    assert fact.config_paths == ()
    validate_fact_claim(fact, fact.claim_evidence)
    decisions = _decisions(events, "hidden_size")
    assert decisions and all(event.intent == "absent_default" and not event.present for event in decisions)
    initial = [event for event in decisions if event.reader == "adapters.transformer.parser.parse"]
    assert len(initial) == 1 and initial[0].value_status_hash == ""
    final = [event for event in decisions if event.reader == result.claim_reader_symbol]
    assert len(final) == 1
    assert final[0].mechanism == "model_hidden_size_value"
    assert final[0].provenance == "class_default"
    assert final[0].value_status_hash == value_status_hash(64, "class_default")
    # Absence remains an absent-default event; its source-named path is not a
    # claim that the checkpoint contains that operand.
    assert "hidden_size" not in document.checkpoint


def test_wrapper_tie_fallback_keeps_wrapper_occurrence_and_child_width():
    child = _config()
    del child["tie_word_embeddings"]
    checkpoint = {"text_config": child, "tie_word_embeddings": False}
    ir, facts, events = _parse(checkpoint)
    assert ir.hidden_size == 64 and ir.tie_word_embeddings is False
    width = facts["model.hidden_size"]
    tie = facts["model.tie_word_embeddings"]
    assert width.config_paths == ("text_config.hidden_size",)
    assert tie.config_paths == ("tie_word_embeddings",)
    validate_fact_claim(width, width.claim_evidence)
    validate_fact_claim(tie, tie.claim_evidence)
    consumed = [event for event in _decisions(events, "tie_word_embeddings") if event.present]
    assert len(consumed) == 1
    assert consumed[0].config_path == "tie_word_embeddings" and consumed[0].path_exact
    assert consumed[0].mechanism == "" and consumed[0].provenance == "checkpoint_declared"
    assert consumed[0].value_status_hash == value_status_hash(False, "config_declared")
    assert any(event.intent == "absent_default" and not event.present
               for event in _decisions(events, "tie_word_embeddings"))


@pytest.mark.parametrize("changed", [{"selected_path": "unrelated_width"}, {"value": 96}])
def test_tampered_consumed_width_decision_cannot_borrow_original_event_proof(monkeypatch, changed):
    original = config_access.ConfigResolution.consume_decision
    supplied = []

    def altered(resolution, **kwargs):
        decision = original(resolution, **kwargs)
        if kwargs.get("fact_owner") == "model" and kwargs.get("fact_key") == "hidden_size" \
                and kwargs.get("reader") == "adapters.transformer.parser.parse":
            supplied.append(decision)
            return replace(decision, **changed)
        return decision

    monkeypatch.setattr(config_access.ConfigResolution, "consume_decision", altered)
    checkpoint = _config()
    checkpoint["unrelated_width"] = 64  # Equal scalar is still a different supplying occurrence.
    _ir, facts, events = _parse(checkpoint)
    assert len(supplied) == 1
    assert supplied[0].value == 64 and supplied[0].selected_path == "hidden_size"
    fact = facts["model.hidden_size"]
    assert fact.value == changed.get("value", 64)
    assert fact.config_paths == (changed.get("selected_path", "hidden_size"),)
    assert fact.claim_kind == "value" and fact.claim_evidence is None
    assert fact.claim_document_token == ""
    # The original real consumption remains intact. Its path/hash cannot
    # certify a changed value or an equal value taken from another address.
    decisions = _decisions(events, "hidden_size")
    assert len(decisions) == 1 and decisions[0].config_path == "hidden_size"
    assert decisions[0].value_status_hash == value_status_hash(64, "config_declared")
