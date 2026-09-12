"""Actual reader kind declarations carry no fabricated claim proof."""
from dataclasses import replace

import pytest

from model_unfolder.evidence.config_access import bound_document, capture_events, owner_scope
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.facts import EvidenceFact
from model_unfolder.evidence.reader_claims import qualify_reader_fact
from test_ffn_schedule import _config


@pytest.mark.parametrize("kind", ["router", "shared_count"])
def test_exact_reader_declares_its_pending_projection_without_proving_legacy_value(kind):
    from model_unfolder.evidence.router import decoder_router_selection_for_path
    from model_unfolder.evidence.expert_width import decoder_shared_expert_count_for_path
    config = _config("dbrx-base" if kind == "router" else "deepseek-v3")
    context = ParseContext.build(config)
    index = context.program_index()
    document = PreparedDocument(dict(config), dict(config))
    with bound_document(DocumentBinding("root", (), document)):
        if kind == "router":
            result = decoder_router_selection_for_path(index, context.source_bundle, (), allow_root_stage=True)
            fact = EvidenceFact(owner="decoder.ffn", key="routing_policy", value={"selection_kind": "topk"},
                                status="code_and_config", completeness="complete")
            intended = "relation"
        else:
            result = decoder_shared_expert_count_for_path(index, context.source_bundle, (), config,
                                                         allow_root_stage=True)
            fact = EvidenceFact(owner="decoder.ffn.expert", key="shared_expert_count", value=1,
                                status="code_and_config", completeness="complete")
            intended = "value"
        assert result.status == "resolved", result.failures
        declared = qualify_reader_fact(fact, result, index, document)
        assert declared.claim_kind == intended
        assert declared.claim_readers == (result.claim_witness.reader_symbol,)
        assert declared.claim_evidence is None and declared.claim_document_token == ""
        assert declared.value == fact.value and declared.status == fact.status
        with pytest.raises(ValueError, match="actual result"):
            qualify_reader_fact(fact, replace(result), index, document)
        with pytest.raises(ValueError, match="intended projection"):
            qualify_reader_fact(replace(fact, key="unrelated"), result, index, document)
        with pytest.raises(ValueError, match="another.*document"):
            qualify_reader_fact(fact, result, index, PreparedDocument(dict(config), dict(config)))


def test_projector_declaration_keeps_original_result_through_actual_width_writer(tmp_path):
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.evidence.context import capture_facts
    from model_unfolder.evidence.projector import projector_result_for_context
    from model_unfolder.adapters.transformer.special_parts.modalities.vision import apply_projector_evidence
    source = tmp_path / "modeling_projector.py"
    source.write_text('''from torch.nn import Linear
class Root:
    def __init__(self): self.bridge = Linear(4, 6)
    def forward(self, inputs_embeds, image_features, mask):
        image_features = self.bridge(image_features)
        return inputs_embeds.masked_scatter(mask, image_features)
''')
    bundle = SourceBundle(source="test", files=(str(source),), architecture="Root",
                          component_files={"root": (str(source),)},
                          component_architectures={"root": "Root"})
    context = ParseContext(bundle)
    unbound = projector_result_for_context(context)
    assert unbound.status == "resolved", unbound.failures
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)), capture_facts(context.facts):
        result = projector_result_for_context(context)
        assert result is not unbound  # Original producer reruns under the document.
        assert projector_result_for_context(context) is result
        payload = {"modalities": {"inputs": {"vision": {"projector": {}, "pipeline": []}}}}
        apply_projector_evidence(payload, result, {})
        for key, value in (("projector_in_features", 4), ("projector_out_features", 6)):
            fact = context.facts.typed["root.vision." + key]
            assert fact.value == value and fact.status == "code_proven"
            assert fact.claim_kind == "value"
            assert fact.claim_evidence is None and fact.claim_document_token == ""
            assert fact.claim_readers == (result.claim_witness.reader_symbol,)
        fact = context.facts.typed["root.vision.projector_out_features"]
        with pytest.raises(ValueError, match="actual result"):
            qualify_reader_fact(fact, replace(result), context.program_index(), document)
        with pytest.raises(ValueError, match="destination-qualified"):
            qualify_reader_fact(replace(fact, owner="root.video"), result,
                                context.program_index(), document)


def test_gated_delta_presence_does_not_stamp_unfinished_live_geometry_relation(tmp_path):
    from test_attention_mechanism import _pipeline, _gated_delta_mixer
    from model_unfolder.evidence.attention import decoder_gated_delta_geometry_for_path
    index, bundle, _root, _block = _pipeline(tmp_path, _gated_delta_mixer())
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_gated_delta_geometry_for_path(index, bundle, (), allow_root_stage=True)
        assert result.status == "resolved", result.failures
        assert result.value.key_heads_path == ("kh",)
        # This older positive has repeats after the terminal. Its sites stay
        # useful, but the stronger live role relation is intentionally unproved.
        fact = EvidenceFact(owner="decoder.attention", key="gated_delta_geometry",
                            value=(16, 48, 128, 128, 4), status="code_and_config",
                            completeness="presence_only")
        declared = qualify_reader_fact(fact, result, index, document)
        assert declared.value == fact.value and declared.status == fact.status
        assert declared.claim_kind == "relation"
        assert declared.claim_evidence is None and declared.claim_document_token == ""
        with pytest.raises(ValueError, match="actual result"):
            qualify_reader_fact(fact, replace(result), index, document)


@pytest.mark.parametrize("kind", ["learned_position", "relative_bias", "geometry_schedule", "replacement_cross", "mtp"])
def test_additional_actual_producers_declare_only_their_finite_intended_kind(tmp_path, kind):
    from model_unfolder.evidence.reader_claims import retained_reader_attempt

    config = {}
    if kind == "geometry_schedule":
        from transformers import AutoConfig
        from test_attention_geometry import _real_schedule
        config = AutoConfig.for_model("gemma4_text").to_dict()
        read = lambda: _real_schedule("gemma4_text", config)
        owner, key, intended = "decoder.attention", "head_geometry_schedule", "relation"
    elif kind == "relative_bias":
        from test_position_relative_bias import _t5_context
        from model_unfolder.evidence.position_relative_bias import decoder_relative_position_bias_for_path
        context = _t5_context()
        read = lambda: decoder_relative_position_bias_for_path(
            context.program_index(), context.source_bundle, (), allow_root_stage=True)
        owner, key, intended = "decoder.attention", "position_schedule", "relation"
    elif kind == "replacement_cross":
        from test_cross_attention_replacement import _case, _selector
        from model_unfolder.evidence.cross_attention_replacement import decoder_replacement_cross_attention_schedule_for_path
        index, bundle = _case(tmp_path)
        config = {"num_hidden_layers": 4, "cross_layers": [1, 3]}
        read = lambda: decoder_replacement_cross_attention_schedule_for_path(
            index, bundle, (), 4, allow_root_stage=True,
            config_selector=_selector({(name,): value for name, value in config.items()}))
        owner, key, intended = "decoder.attention", "cross_attention_schedule", "relation"
    elif kind == "learned_position":
        from test_position_absolute import _result
        read = lambda: _result(tmp_path)
        owner, key, intended = "decoder.input", "position_addition", "connection"
    else:
        from test_mtp import _read
        read = lambda: _read(tmp_path)
        owner, key, intended = "decoder", "mtp_modules", "relation"

    document = PreparedDocument(dict(config), dict(config))
    with bound_document(DocumentBinding("root", (), document)):
        result = read()
        assert result.status == "resolved", result.failures
        _symbol, index, actual_document, _occurrence = retained_reader_attempt(result)
        assert actual_document is document
        # Intentionally no successful proof is inferred from the older aggregate.
        # This declaration describes the question even when its value is unproved.
        fact = EvidenceFact(owner=owner, key=key, value={"legacy": "unchanged"},
                            status="code_and_config", completeness="presence_only")
        declared = qualify_reader_fact(fact, result, index, document)
        assert declared.claim_kind == intended
        assert declared.claim_readers == (result.claim_witness.reader_symbol,)
        assert declared.claim_evidence is None and declared.claim_document_token == ""
        assert declared.value == fact.value and declared.status == fact.status
        assert declared.completeness == fact.completeness
        with pytest.raises(ValueError, match="actual result"):
            qualify_reader_fact(fact, replace(result), index, document)
        with pytest.raises(ValueError, match="intended projection"):
            qualify_reader_fact(replace(fact, key="unrelated"), result, index, document)
        with pytest.raises(ValueError, match="intended projection"):
            qualify_reader_fact(replace(fact, owner="foreign.owner"), result, index, document)
        with pytest.raises(ValueError, match="another.*document"):
            qualify_reader_fact(fact, result, index, PreparedDocument(dict(config), dict(config)))


def test_actual_failed_reader_declares_question_without_claiming_exhaustion(tmp_path):
    from test_embedding_bookend import _pipeline
    from model_unfolder.evidence.final_bookend import final_stage_norm_evidence
    from model_unfolder.evidence.reader_claims import declare_reader_fact
    from model_unfolder.evidence.reader_result import ReaderResult, ReaderFailure
    bundle, index, *_ = _pipeline(tmp_path)
    document = PreparedDocument({}, {})
    fact = EvidenceFact(owner="model", key="final_norm_kind", value=None,
                        status="ambiguous", completeness="complete")
    with bound_document(DocumentBinding("root", (), document)):
        result = final_stage_norm_evidence(index, bundle)
        assert result.status != "resolved" and result.claim_witness is None
        declared = qualify_reader_fact(fact, result, index, document)
        assert declared.claim_kind == "applied_function"
        assert declared.value is None and declared.status == "ambiguous"
        assert declared.claim_evidence is None and declared.claim_document_token == ""
        assert declared.unknown_reason.investigation is None
        assert declared.unknown_reason.reason_class == "investigation_missing"
        with pytest.raises(ValueError, match="actual retained reader invocation"):
            declare_reader_fact(fact, replace(result), index, document)
        with pytest.raises(ValueError, match="intended projection"):
            declare_reader_fact(replace(fact, owner="decoder.layer"), result, index, document)
        with pytest.raises(ValueError, match="intended projection"):
            declare_reader_fact(replace(fact, key="embedding_norm_kind"), result, index, document)
        with pytest.raises(ValueError, match="another source index or prepared document"):
            declare_reader_fact(fact, result, index, PreparedDocument({}, {}))
        fabricated = ReaderResult.failed(result.owner, (ReaderFailure("incomplete_graph", "unissued synthetic failure"),))
        assert qualify_reader_fact(fact, fabricated, index, document) is fact
        with pytest.raises(ValueError, match="actual retained reader invocation"):
            declare_reader_fact(fact, replace(fabricated, claim_reader_symbol=result.claim_reader_symbol), index, document)


def test_intended_declaration_cannot_erase_an_existing_qualified_proof(tmp_path):
    from test_reader_claims import _qualified_final
    from model_unfolder.evidence.reader_claims import declare_reader_fact
    _bundle, index, document, result, qualified = _qualified_final(tmp_path)
    with pytest.raises(ValueError, match="cannot erase an already qualified proof"):
        declare_reader_fact(qualified, result, index, document)


def test_producer_intended_vocabulary_cannot_be_replaced_by_writable_wrapper_attributes(tmp_path):
    from test_embedding_bookend import _pipeline
    from model_unfolder.evidence.final_bookend import final_stage_norm_evidence
    from model_unfolder.evidence.reader_claims import declare_reader_fact
    bundle, index, *_ = _pipeline(tmp_path)
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)):
        result = final_stage_norm_evidence(index, bundle)
        final_stage_norm_evidence.intended_claims = (("model", "final_norm_kind", "existence"),)
        try:
            fact = EvidenceFact(owner="model", key="final_norm_kind", value=None,
                                status="ambiguous", completeness="complete")
            assert declare_reader_fact(fact, result, index, document).claim_kind == "applied_function"
        finally:
            del final_stage_norm_evidence.intended_claims


def test_manual_weight_assignment_declares_existing_scalar_without_sharing_proof(tmp_path):
    from test_weight_tying import _reader
    from model_unfolder.evidence.reader_claims import retained_reader_attempt
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)):
        result = _reader(tmp_path)
        assert result.status == "resolved"
        _, index, _, _ = retained_reader_attempt(result)
        fact = EvidenceFact(owner="model", key="tie_word_embeddings", value=True,
                            status="code_proven", completeness="complete")
        declared = qualify_reader_fact(fact, result, index, document)
        assert declared.value is True and declared.status == "code_proven"
        assert declared.claim_kind == "value"
        assert declared.claim_evidence is None and declared.claim_document_token == ""
        with pytest.raises(ValueError, match="intended claim kind"):
            qualify_reader_fact(replace(fact, claim_kind="relation",
                                       claim_readers=(result.claim_reader_symbol,)), result, index, document)


def test_actual_missing_source_rows_keep_values_statuses_and_declared_questions():
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.adapters.transformer.parser import parse
    config = {"model_type": "synthetic_missing", "hidden_size": 8,
              "num_hidden_layers": 1, "num_attention_heads": 2,
              "num_key_value_heads": 2, "intermediate_size": 16,
              "vocab_size": 32, "hidden_act": "relu"}
    context = ParseContext(SourceBundle(source="missing", files=()))
    document = PreparedDocument(config, dict(config))
    with capture_events(context.config_access), owner_scope("root"), \
            bound_document(DocumentBinding("root", (), document)):
        parse(config, context=context)
    for key, kind in (("bias", "existence"), ("scores_scale", "applied_function"),
                      ("projection_mode", "relation")):
        fact = context.facts.typed["decoder.attention." + key]
        assert fact.value is None and fact.status == "oracle_missing"
        assert fact.claim_kind == kind
        assert fact.claim_evidence is None and fact.claim_document_token == ""
