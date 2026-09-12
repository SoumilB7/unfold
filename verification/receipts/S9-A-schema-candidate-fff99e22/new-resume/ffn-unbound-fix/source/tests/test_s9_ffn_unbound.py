"""Actual FFN source without a prepared document cannot issue operand proof."""
from dataclasses import replace

import pytest

from model_unfolder.adapters.transformer import parser
from model_unfolder.evidence.config_access import (
    bound_document, capture_events, current_prepared_document,
)
from model_unfolder.evidence.context import ParseContext, capture_facts
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.ffn_mechanism import decoder_ffn_mechanism_for_path
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reader_claims import (
    ReaderClaimUnavailable, ReaderProjectionClaimProof, qualify_reader_fact,
)
from model_unfolder.evidence.receipts import value_status_hash
from test_attention_geometry import _config
from test_ffn_mechanism import _reader, _config_selected_wrapper_reader
from test_reader_claims import _native_fact


def _bundle(path):
    source = str(path)
    return SourceBundle(source="local", files=(source,), architecture="Wrapper",
                        component_files={"root": (source,)},
                        component_architectures={"root": "Wrapper"})


@pytest.mark.parametrize("dispatch,activation", [(True, "silu"), (False, "gelu")])
def test_direct_source_ffn_parse_retains_value_and_event_without_proof(tmp_path, dispatch, activation):
    expression = "ACT2FN[config.hidden_act]" if dispatch else "nn.GELU()"
    raw = _reader(tmp_path, f'''
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = {expression}
    def forward(self, x):
        return self.down(self.act(self.up(x)))
''')
    assert raw.status == "resolved"
    context = ParseContext(_bundle(tmp_path / "model.py"))
    config = _config(hidden=64, wide=128, intermediate_size=128, hidden_act="silu")
    assert current_prepared_document.get() is None
    with capture_events(context.config_access), capture_facts(context.facts):
        ir = parser.parse(config, context=context)
    fact = context.facts.typed_records()["decoder.ffn.activation"]
    assert ir.layers[0].ffn.activation == fact.value == activation
    assert fact.status == ("code_and_config" if dispatch else "code_proven")
    assert fact.claim_kind == "applied_function"
    assert fact.claim_readers == (
        "model_unfolder.evidence.ffn_mechanism.decoder_ffn_mechanism_for_path",)
    assert fact.claim_evidence is None and fact.claim_document_token == ""
    assert fact.occurrence_citation is None
    events = [event for event in context.config_access.events
              if event.fact_owner == "decoder.ffn" and event.fact_key == "activation"
              and event.mechanism == "ffn_activation"]
    assert len(events) == (1 if dispatch else 0)
    if dispatch:
        assert events[0].config_path == "hidden_act"
        assert events[0].value_status_hash == value_status_hash(fact.value, fact.status)
    result = decoder_ffn_mechanism_for_path(
        context.program_index(), context.source_bundle, (), allow_root_stage=True)
    assert result.status == "resolved"
    if dispatch:
        with pytest.raises(ReaderClaimUnavailable, match="no bound supplying document"):
            result.claim_witness.project("decoder.ffn", "activation", None)
        with pytest.raises(ReaderClaimUnavailable, match="no bound supplying document"):
            result.claim_witness.validate_legacy_projection_value(fact, None, None)
    else:
        # A literal source projection needs no operand lookup, yet without a
        # document it still cannot become a retained positive proof above.
        projection = result.claim_witness.project("decoder.ffn", "activation", None)
        assert projection.value == "gelu" and projection.status == "code_proven"
    assert current_prepared_document.get() is None
    assert context.prepared_documents == {}


@pytest.mark.parametrize("selected", [True, False])
@pytest.mark.parametrize("key,kind", [
    ("gated", "connection"), ("activation", "applied_function"),
    ("projection_mode", "relation"),
])
def test_source_selected_ffn_needs_document_for_every_dependent_claim(tmp_path, selected, key, kind):
    _config_selected_wrapper_reader(tmp_path, lambda path: selected)
    bundle = _bundle(tmp_path / "selected.py")
    index = build_program_index(bundle)
    assert current_prepared_document.get() is None
    result = decoder_ffn_mechanism_for_path(
        index, bundle, (), allow_root_stage=True, config_selector=lambda path: selected)
    assert result.status == "resolved"
    expected = {"gated": selected, "activation": "silu" if selected else "gelu",
                "projection_mode": "split" if selected else "dense"}[key]
    with pytest.raises(ReaderClaimUnavailable, match="no bound supplying document"):
        result.claim_witness.project("decoder.ffn", key, None)
    fact = replace(_native_fact(result, owner="decoder.ffn", key=key, value=expected),
                   status="code_and_config")
    limited = qualify_reader_fact(fact, result, index, None)
    assert limited.value == expected and limited.claim_kind == kind
    assert limited.claim_evidence is None and limited.claim_document_token == ""
    assert limited.occurrence_citation is None
    with pytest.raises(ValueError, match="prepared document"):
        ReaderProjectionClaimProof(fact.ledger_key(), result, index, None)

    checkpoint = {"choose_split": selected}
    document = PreparedDocument(dict(checkpoint), dict(checkpoint))
    with bound_document(DocumentBinding("root", (), document)):
        rebound = decoder_ffn_mechanism_for_path(
            index, bundle, (), allow_root_stage=True, config_selector=lambda path: selected)
        projection = ReaderProjectionClaimProof(fact.ledger_key(), rebound, index, document).projection()
    assert projection.value == expected and projection.claim_kind == kind
    assert projection.status == "code_and_config"
    assert projection.config_paths == (("choose_split",),)
