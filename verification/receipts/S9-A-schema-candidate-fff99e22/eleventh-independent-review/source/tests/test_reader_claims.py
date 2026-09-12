"""S9-A: kind qualification requires the actual producer's whole proof."""
from dataclasses import replace

import pytest

from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.facts import EvidenceFact, SourceSpan
from model_unfolder.evidence.final_bookend import final_stage_norm_evidence
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reader_claims import (
    ReaderProjectionClaimProof, qualify_reader_fact, reader_claim_scope_matches,
)
from model_unfolder.evidence.reader_result import ReaderResult
from test_embedding_bookend import _pipeline, _with_final_norm


def _native_fact(result, *, key="final_norm_kind", owner="model", value="rmsnorm"):
    return EvidenceFact(
        key=key, owner=owner, value=value, status="code_proven",
        completeness="complete",
        source_spans=tuple(dict.fromkeys(SourceSpan(
            component=span.source.component_key or "root",
            file=span.source.canonical_path, line=span.line)
            for origin in result.provenance for span in origin.spans)))


def _qualified_final(tmp_path):
    bundle, index, *_ = _pipeline(tmp_path, _with_final_norm())
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)):
        result = final_stage_norm_evidence(index, bundle)
        assert result.status == "resolved", result.failures
        fact = qualify_reader_fact(_native_fact(result), result, index, document)
    return bundle, index, document, result, fact


def test_final_norm_keeps_actual_applied_function_lineage(tmp_path):
    _bundle, index, document, result, fact = _qualified_final(tmp_path)
    assert fact.value == "rmsnorm"
    assert fact.claim_kind == "applied_function"
    assert result.value == "RMSNorm"  # existing public reader contract
    assert result.claim_witness.returns
    assert result.claim_witness.upstream_paths
    assert reader_claim_scope_matches(result, index, document)
    validate_fact_claim(fact, fact.claim_evidence)
    summary = fact.claim_evidence.summary()
    assert summary.index_fingerprints
    assert all("/" not in value for value in summary.evidence_refs)


@pytest.mark.parametrize("change", [
    {"value": "layernorm"}, {"value": True}, {"value": 1},
    {"status": "class_default"}, {"claim_kind": "connection"},
    {"owner": "decoder.layer"}, {"key": "norm_placement"},
    {"completeness": "presence_only"}, {"source_spans": ()},
    {"config_paths": ("invented",)}, {"claim_document_token": "a" * 64},
])
def test_mutating_any_qualified_fact_axis_is_rejected(tmp_path, change):
    *_unused, fact = _qualified_final(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        replace(fact, **change)


def test_reconstructed_result_cannot_borrow_a_real_witness(tmp_path):
    _bundle, index, document, result, _fact = _qualified_final(tmp_path)
    copied = replace(result)
    assert copied == result
    assert not reader_claim_scope_matches(copied, index, document)
    with pytest.raises(ValueError, match="actual result"):
        qualify_reader_fact(_native_fact(copied), copied, index, document)


def test_writable_function_names_cannot_impersonate_a_producer(tmp_path):
    from model_unfolder.evidence.reader_claims import retained_claim_reader
    _bundle, _index, _document, result, _fact = _qualified_final(tmp_path)
    def fake(*args, **kwargs):
        return replace(result, value="LayerNorm",
                       claim_witness=replace(result.claim_witness, norm_kind="layernorm"))
    fake.__module__ = final_stage_norm_evidence.__module__
    fake.__qualname__ = final_stage_norm_evidence.__qualname__
    with pytest.raises(ValueError, match="exact original producer"):
        retained_claim_reader(fake)


def test_successful_string_and_spans_cannot_replace_semantic_witness(tmp_path):
    _bundle, index, document, result, _fact = _qualified_final(tmp_path)
    reconstructed = ReaderResult.resolved(
        result.owner, result.value, provenance=result.provenance)
    raw = _native_fact(reconstructed)
    assert qualify_reader_fact(raw, reconstructed, index, document) is raw
    assert raw.claim_evidence is None


def test_preparse_result_requires_one_real_read_in_bound_document(tmp_path):
    bundle, index, *_ = _pipeline(tmp_path, _with_final_norm())
    unbound = final_stage_norm_evidence(index, bundle)
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)):
        assert not reader_claim_scope_matches(unbound, index, document)
        with pytest.raises(ValueError, match="prepared document"):
            qualify_reader_fact(_native_fact(unbound), unbound, index, document)
        rebound = final_stage_norm_evidence(index, bundle)
        assert rebound.value == unbound.value
        assert reader_claim_scope_matches(rebound, index, document)
        assert qualify_reader_fact(_native_fact(rebound), rebound, index, document).claim_evidence


def test_another_index_cannot_borrow_the_original_read(tmp_path):
    bundle, _index, document, result, _fact = _qualified_final(tmp_path)
    other_index = build_program_index(bundle)
    with pytest.raises(ValueError, match="source index"):
        ReaderProjectionClaimProof("model.final_norm_kind", result, other_index, document)


def test_identical_checkpoint_in_another_document_cannot_borrow_read(tmp_path):
    _bundle, index, _document, result, _fact = _qualified_final(tmp_path)
    other = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), other)):
        with pytest.raises(ValueError, match="prepared document"):
            ReaderProjectionClaimProof("model.final_norm_kind", result, index, other)


def test_checkpoint_mutation_invalidates_existing_proof(tmp_path):
    _bundle, index, document, result, fact = _qualified_final(tmp_path)
    document.checkpoint["changed"] = True
    assert not reader_claim_scope_matches(result, index, document)
    with pytest.raises(ValueError, match="prepared document"):
        validate_fact_claim(fact, fact.claim_evidence)


@pytest.mark.parametrize("channel", ["class_overlay", "document", "provenance"])
def test_prepared_channel_mutation_invalidates_existing_proof(tmp_path, channel):
    _bundle, index, document, result, fact = _qualified_final(tmp_path)
    getattr(document, channel)["changed"] = True
    assert not reader_claim_scope_matches(result, index, document)
    with pytest.raises(ValueError, match="prepared document"):
        validate_fact_claim(fact, fact.claim_evidence)


def test_final_norm_cannot_author_another_finite_projection(tmp_path):
    _bundle, index, document, result, _fact = _qualified_final(tmp_path)
    with pytest.raises(ValueError, match="another fact projection"):
        ReaderProjectionClaimProof("decoder.layer.norm_placement", result, index, document)


def test_proof_operand_inspection_never_clears_real_config_obligations():
    from model_unfolder.evidence.config_access import capture_events, resolve
    from model_unfolder.evidence.reader_claims import reader_operand
    document = PreparedDocument({"width": 12}, {"width": 12})
    with bound_document(DocumentBinding("root", (), document)), capture_events() as ledger:
        operand = reader_operand(document, ("width",), allow_aliases=False)
        assert operand.value == 12
        assert ledger.events == []
        ordinary = resolve(document.document, "width")
        assert ordinary.value == operand.value
        assert [event.intent for event in ledger.events] == ["inspected"]
        before = tuple(ledger.events)
        reader_operand(document, ("width",), allow_aliases=False)
        assert tuple(ledger.events) == before
        assert ledger.accessed_but_unconsumed() == {("root", "width")}


def test_nested_flat_default_uses_the_same_prepared_channel():
    from model_unfolder.evidence.reader_claims import reader_operand
    document = PreparedDocument({"text": {}}, {"text": {}},
                                class_overlay={"text.width": 12})
    with bound_document(DocumentBinding("root", (), document)):
        operand = reader_operand(document, ("text", "width"), allow_aliases=False)
        assert operand.value == 12
        assert operand.source_kind == "class_default"
        assert operand.checkpoint_path is None


def test_checkpoint_label_keys_keep_lossless_types_and_stable_order():
    from model_unfolder.evidence.config_access import checkpoint_fingerprint
    first = {"labels": {0: "zero", "0": "string zero", -1: "negative"}}
    reordered = {"labels": {-1: "negative", "0": "string zero", 0: "zero"}}
    assert checkpoint_fingerprint(first) == checkpoint_fingerprint(reordered)
    assert checkpoint_fingerprint({"labels": {0: "same"}}) != checkpoint_fingerprint({"labels": {"0": "same"}})
    assert checkpoint_fingerprint({"labels": {-1: "same"}}) != checkpoint_fingerprint({"labels": {1: "same"}})
    assert checkpoint_fingerprint({"labels": {0: "a", "0": "b"}}) != checkpoint_fingerprint({"labels": {0: "b", "0": "a"}})


def test_all_string_checkpoint_fingerprint_keeps_existing_encoding():
    import hashlib
    import json
    from model_unfolder.evidence.config_access import checkpoint_fingerprint
    payload = ["dict", [["flag", ["bool", True]], ["width", ["int", "12"]]]]
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert checkpoint_fingerprint({"width": 12, "flag": True}) == expected


@pytest.mark.parametrize("foreign_selector", [False, True])
def test_topology_checks_actual_selector_values_against_prepared_document(tmp_path, foreign_selector):
    from test_cell_topology import _read, _GUARDED_AUGMENTED_CELL, _GUARDED_NORMS
    declared = {"new_decoder_architecture": False, "parallel": False, "num_norms": 1}
    selected = {**declared, "parallel": foreign_selector}
    document = PreparedDocument(dict(declared), dict(declared))
    with bound_document(DocumentBinding("root", (), document)):
        result = _read(tmp_path, _GUARDED_AUGMENTED_CELL,
                       selected_config=selected, norm_fields=_GUARDED_NORMS)
        assert result.status == "resolved", result.failures
        if foreign_selector:
            with pytest.raises(ValueError, match="selector value/origin"):
                result.claim_witness.project("decoder.layer", "residual_topology", document)
        else:
            projection = result.claim_witness.project("decoder.layer", "residual_topology", document)
            assert projection.value == "sequential"
            assert projection.status == "code_and_config"


@pytest.mark.parametrize("mismatch", [None, "count", "selector"])
@pytest.mark.parametrize("family", ["mixer", "ffn"])
def test_real_schedule_deciding_operands_cannot_borrow_another_document(family, mismatch):
    import copy
    if family == "mixer":
        from test_mixer_schedule import _config, _real_result
        owner, key, selector_key = "decoder.attention", "mixer_schedule", "layer_types"
    else:
        from test_ffn_schedule import _config, _real_result
        owner, key, selector_key = "decoder.ffn", "ffn_schedule", "first_k_dense_replace"
    selected = _config()
    declared = copy.deepcopy(selected)
    if mismatch == "count":
        declared["num_hidden_layers"] += 1
    elif mismatch == "selector":
        if family == "mixer":
            declared[selector_key][0] = "full_attention"
        else:
            declared[selector_key] += 1
    document = PreparedDocument(copy.deepcopy(declared), copy.deepcopy(declared))
    with bound_document(DocumentBinding("root", (), document)):
        result = _real_result(config=selected)
        assert result.status == "resolved", result.failures
        witness = result.claim_witness
        if mismatch:
            with pytest.raises(ValueError, match="deciding operand"):
                witness.project(owner, key, document)
            with pytest.raises(ValueError, match="selector value/origin|deciding operand"):
                ReaderProjectionClaimProof(f"{owner}.{key}", result, witness.index, document)
        else:
            projection = witness.project(owner, key, document)
            assert len(projection.value) == selected["num_hidden_layers"]
            ReaderProjectionClaimProof(f"{owner}.{key}", result, witness.index, document)


@pytest.mark.parametrize("mismatch", [None, "count", "selector"])
def test_position_schedule_checks_retained_values_without_callback_trust(tmp_path, mismatch):
    from test_position_schedule import _result
    declared = {"num_hidden_layers": 4, "position_schedule": [1, 0, 1, 0]}
    if mismatch == "count":
        declared["num_hidden_layers"] = 5
    elif mismatch == "selector":
        declared["position_schedule"] = [0, 1, 0, 1]
    document = PreparedDocument(dict(declared), dict(declared))
    with bound_document(DocumentBinding("root", (), document)):
        result = _result(tmp_path)
        assert result.status == "resolved", result.failures
        witness = result.claim_witness
        if mismatch:
            with pytest.raises(ValueError, match="deciding operand"):
                witness.project("decoder.attention", "position_schedule", document)
        else:
            ReaderProjectionClaimProof("decoder.attention.position_schedule", result,
                                       witness.index, document)


@pytest.mark.parametrize("foreign_base", [False, True])
def test_frequency_initializer_checks_actual_base_against_sealed_document(tmp_path, foreign_base):
    from test_position_initialization import _result
    declared = {"num_hidden_layers": 4, "position_schedule": [1, 1, 1, 1],
                "hidden_size": 32, "head_width": 8,
                "frequency": {"kind": "plain", "base": 54321.0 if foreign_base else 12345.0}}
    document = PreparedDocument(dict(declared), dict(declared))
    with bound_document(DocumentBinding("root", (), document)):
        result = _result(tmp_path)
        assert result.status == "resolved", result.failures
        witness = result.claim_witness
        if foreign_base:
            with pytest.raises(ValueError, match="deciding operand"):
                witness.project("decoder.attention", "rope_theta", document)
        else:
            ReaderProjectionClaimProof("decoder.attention.rope_theta", result,
                                       witness.index, document)


def test_actual_failed_attempt_needs_current_scope_and_copies_are_not_authentic(tmp_path):
    from model_unfolder.evidence.reader_claims import reader_requires_claim_scope
    bundle, index, *_ = _pipeline(tmp_path)
    unbound = final_stage_norm_evidence(index, bundle)
    assert unbound.status == "failed"
    assert reader_requires_claim_scope(unbound)
    document = PreparedDocument({}, {})
    assert not reader_claim_scope_matches(unbound, index, document)
    with bound_document(DocumentBinding("root", (), document)):
        bound = final_stage_norm_evidence(index, bundle)
    assert reader_claim_scope_matches(bound, index, document)
    copied = replace(bound)
    assert reader_requires_claim_scope(copied)
    assert not reader_claim_scope_matches(copied, index, document)


@pytest.mark.parametrize("family", ["alibi", "additive"])
@pytest.mark.parametrize("count", [3, 7])
def test_repetition_capability_derives_count_from_actual_source_operand(tmp_path, family, count):
    from model_unfolder.evidence.decoder_block import decoder_block_candidates_for_config, decoder_block_path_for_config
    from model_unfolder.evidence.cross_attention_schedule import AdditiveCrossClaimWitness, decoder_cross_attention_all_layers_for_path
    from model_unfolder.evidence.position_linear_bias import AlibiClaimWitness, decoder_alibi_score_bias_for_path
    from test_position_linear_bias import _pipeline
    if family == "additive":
        from test_cross_attention_schedule import _SOURCE
        index, bundle = _pipeline(tmp_path, raw=_SOURCE, architecture="Shell")
        result = decoder_cross_attention_all_layers_for_path(index, bundle, (), allow_root_stage=True)
        blocks = decoder_block_path_for_config(index, bundle, (), allow_root_stage=True).value
        witness_type, key = AdditiveCrossClaimWitness, "cross_attention_schedule"
    else:
        index, bundle = _pipeline(tmp_path)
        result = decoder_alibi_score_bias_for_path(index, bundle, (), allow_root_stage=True)
        blocks = decoder_block_candidates_for_config(index, bundle, (), allow_root_stage=True).value
        witness_type, key = AlibiClaimWitness, "position_schedule"
    assert result.status == "resolved", result.failures
    # These count provenance changes are held for the owner's scope ruling.
    assert result.claim_witness is None
    document = PreparedDocument({"layers": count}, {"layers": count})
    witness = witness_type(index, result.value, blocks)
    witness.validate_result(result)
    projected = witness.project("decoder.attention", key, document)
    assert len(projected.value) == count
    assert projected.status == "code_and_config"
    assert projected.config_paths == (("layers",),)
    with pytest.raises(ValueError, match="operand"):
        witness.project("decoder.attention", key, PreparedDocument({}, {}))


@pytest.mark.parametrize("mutation", ["arithmetic", "shadow"])
def test_repetition_capability_never_treats_a_range_operand_as_a_whole_count(tmp_path, mutation):
    from model_unfolder.evidence.decoder_block import decoder_block_candidates_for_config
    from model_unfolder.evidence.reader_claims import repeated_count_operand
    from test_position_linear_bias import _pipeline, _source
    source = _source()
    if mutation == "arithmetic":
        source = source.replace("range(config.layers)", "range(config.layers + 1)")
    else:
        source += "\nrange = lambda count: (0,)\n"
    index, bundle = _pipeline(tmp_path, raw=source)
    blocks = decoder_block_candidates_for_config(index, bundle, (), allow_root_stage=True)
    assert blocks.status == "resolved", blocks.failures
    document = PreparedDocument({"layers": 3}, {"layers": 3})
    with pytest.raises(ValueError, match="range|source operand"):
        repeated_count_operand(index, blocks.value, blocks.value.occurrences[0], document)


def test_real_ffn_projects_mechanism_and_dispatch_as_distinct_claim_kinds():
    from test_ffn_schedule import _config
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.ffn_mechanism import decoder_ffn_mechanism_for_path
    config = _config("llama-7b")
    context = ParseContext.build(config)
    document = PreparedDocument(dict(config), dict(config))
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_ffn_mechanism_for_path(context.program_index(), context.source_bundle,
                                               (), allow_root_stage=True)
        assert result.status == "resolved", result.failures
        proofs = {key: ReaderProjectionClaimProof(f"decoder.ffn.{key}", result,
                    context.program_index(), document) for key in ("gated", "projection_mode")}
        from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
        with pytest.raises(ReaderClaimUnavailable, match="selected class has dynamic declaration"):
            ReaderProjectionClaimProof("decoder.ffn.activation", result,
                                       context.program_index(), document)
        assert proofs["gated"].claim_kind == "connection"
        assert proofs["gated"].projection().value is True
        assert proofs["projection_mode"].claim_kind == "relation"



def test_exact_source_span_membership_is_cached_and_rejects_foreign_locations(tmp_path):
    from model_unfolder.evidence.reader_claims import _indexed_spans
    _bundle, index, _document, result, _fact = _qualified_final(tmp_path)
    actual = result.claim_witness.norm_call.span
    spans = _indexed_spans(index)
    assert actual in spans
    assert _indexed_spans(index) is spans
    assert replace(actual, line=999999, end_line=999999) not in spans
    assert replace(actual, source=replace(actual.source,
                   content_fingerprint="0" * 64)) not in spans


def test_sink_connection_cannot_be_qualified_by_presence_or_changed_value(tmp_path):
    from test_attention_sinks import _reader
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)):
        result = _reader(tmp_path)
        assert result.status == "resolved", result.failures
        witness = result.claim_witness
        fact = qualify_reader_fact(_native_fact(result, key="sinks", owner="decoder.attention", value=True),
                                   result, witness.index, document)
        assert fact.claim_kind == "connection"
        with pytest.raises(ValueError):
            validate_fact_claim(replace(fact, claim_kind="existence"), fact.claim_evidence)
        with pytest.raises(ValueError):
            validate_fact_claim(replace(fact, value=False), fact.claim_evidence)


@pytest.mark.parametrize("defaulted", [False, True])
def test_clip_keeps_exact_applied_operand_and_its_declared_or_default_channel(tmp_path, defaulted):
    from test_attention_clip import _bundle, _read
    document = PreparedDocument({} if defaulted else {"clip_qkv": 8},
                                {} if defaulted else {"clip_qkv": 8},
                                class_overlay={"clip_qkv": 8} if defaulted else {})
    with bound_document(DocumentBinding("root", (), document)):
        result = _read(_bundle(tmp_path, "qkv = qkv.clamp(max=self.limit)"))
        assert result.status == "resolved", result.failures
        proof = ReaderProjectionClaimProof("decoder.attention.qkv_clip", result,
                                           result.claim_witness.index, document)
        assert proof.claim_kind == "applied_function"
        assert proof.projection().value == 8
        assert proof.projection().status == ("class_default" if defaulted else "code_and_config")
        assert proof.projection().config_paths == (() if defaulted else (("clip_qkv",),))


@pytest.mark.parametrize("raw", [{"name": "silu"}, "SILU", "gated-silu", "missing_activation_key"])
def test_actual_ffn_dispatch_never_normalizes_or_invents_a_registry_key(raw):
    from test_ffn_schedule import _config
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.ffn_mechanism import decoder_ffn_mechanism_for_path
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    config = _config("llama-7b")
    config["hidden_act"] = raw
    context = ParseContext.build(config)
    document = PreparedDocument(dict(config), dict(config))
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_ffn_mechanism_for_path(context.program_index(), context.source_bundle,
                                               (), allow_root_stage=True)
        assert result.status == "resolved", result.failures
        assert result.value.activation_config_path == ("hidden_act",)
        with pytest.raises(ReaderClaimUnavailable):
            ReaderProjectionClaimProof("decoder.ffn.activation", result,
                                       context.program_index(), document)
        # Failure to identify the callable does not erase the proven affine graph.
        assert ReaderProjectionClaimProof("decoder.ffn.gated", result,
            context.program_index(), document).projection().value is True


def test_neutral_class_decorators_are_retained_without_interpreting_them(tmp_path):
    from test_program_index import _index, _class
    index = _index(tmp_path, "decorated.py", """
@wrapped("kernel")
class Cell(metaclass=meta):
    @another
    def forward(self, value):
        return value
""")
    cls = _class(index, "Cell")
    assert cls.keywords[0][0] == "metaclass"
    assert cls.keywords[0][1].name == "meta"
    assert len(cls.decorators) == 1
    assert cls.decorators[0].kind == "call"
    assert cls.decorators[0].children[0].name == "wrapped"
    assert cls.decorators[0].children[1].const_value == "kernel"
    assert index.callables_of(cls.symbol)[0].decorators[0].name == "another"


@pytest.mark.parametrize("raw,primitive", [("relu", "relu"), ("swish", "silu")])
def test_exact_imported_registry_key_retains_instantiated_primitive(raw, primitive):
    from test_ffn_schedule import _config
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.ffn_mechanism import decoder_ffn_mechanism_for_path
    config = _config("llama-7b")
    config["hidden_act"] = raw
    context = ParseContext.build(config)
    document = PreparedDocument(dict(config), dict(config))
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_ffn_mechanism_for_path(context.program_index(), context.source_bundle,
                                               (), allow_root_stage=True)
        proof = ReaderProjectionClaimProof("decoder.ffn.activation", result,
                                           context.program_index(), document)
        assert proof.claim_kind == "applied_function"
        assert proof.projection().value == raw
        assert result.claim_witness.activation_applications[0].mechanism == primitive
        assert any(span.source.canonical_path.endswith("/activations.py")
                   for origin in result.provenance for span in origin.spans)


@pytest.mark.parametrize("foreign", [False, True])
def test_ffn_width_binds_actual_shape_operands_and_upstream_reader(foreign):
    from test_ffn_schedule import _config
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.ffn_width import decoder_ffn_intermediate_width_for_path
    config = _config("llama-7b")
    context = ParseContext.build(config)
    document = PreparedDocument(dict(config), dict(config))
    supplied = dict(config)
    if foreign:
        supplied["intermediate_size"] += 8
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_ffn_intermediate_width_for_path(context.program_index(),
            context.source_bundle, (), supplied, allow_root_stage=True)
        assert result.status == "resolved", result.failures
        if foreign:
            with pytest.raises(ValueError, match="operand"):
                ReaderProjectionClaimProof("decoder.ffn.intermediate_size", result,
                    context.program_index(), document)
        else:
            proof = ReaderProjectionClaimProof("decoder.ffn.intermediate_size", result,
                context.program_index(), document)
            assert proof.claim_kind == "value"
            assert proof.projection().value == config["intermediate_size"]
            assert proof.projection().status == "code_and_config"
            upstream = result.claim_witness.mechanism_result
            copied = decoder_ffn_intermediate_width_for_path(context.program_index(),
                context.source_bundle, (), supplied, allow_root_stage=True,
                mechanism_result=replace(upstream))
            with pytest.raises(ValueError, match="actual|retained"):
                ReaderProjectionClaimProof("decoder.ffn.intermediate_size", copied,
                    context.program_index(), document)


@pytest.mark.parametrize("foreign", [False, True])
def test_expert_width_requires_exact_fused_parameter_operand(foreign):
    from test_ffn_schedule import _config
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.expert_width import decoder_expert_intermediate_width_for_path
    config = _config("deepseek-v3")
    context = ParseContext.build(config)
    document = PreparedDocument(dict(config), dict(config))
    supplied = dict(config)
    if foreign:
        supplied["moe_intermediate_size"] += 8
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_expert_intermediate_width_for_path(context.program_index(),
            context.source_bundle, (), supplied, allow_root_stage=True)
        assert result.status == "resolved", result.failures
        if foreign:
            with pytest.raises(ValueError, match="operand"):
                ReaderProjectionClaimProof("decoder.ffn.expert.expert_intermediate_size", result,
                    context.program_index(), document)
        else:
            proof = ReaderProjectionClaimProof("decoder.ffn.expert.expert_intermediate_size", result,
                context.program_index(), document)
            assert proof.claim_kind == "value"
            assert proof.projection().value == config["moe_intermediate_size"]
            upstream = result.claim_witness.storage_result
            assert ReaderProjectionClaimProof("decoder.ffn.expert.expert_projection_mode", upstream,
                context.program_index(), document).projection().value == "fused_gate_up"


def _registry_use_fixture(tmp_path, *, registry_tail="", consumer_tail="", namespace=False,
                          registry_edit=None, full=False):
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.evidence.construction_calls import resolve_import_reference
    from model_unfolder.evidence.activation_registry import _registry_use_closure
    package = tmp_path / "package"
    package.mkdir()
    registry = package / "registry.py"
    registry_source = '''from collections import OrderedDict
from torch import nn
class Instantiate(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)
ENTRIES = {"relu": nn.ReLU}
ACTIVATIONS = Instantiate(ENTRIES)
''' + registry_tail
    registry.write_text(registry_edit(registry_source) if registry_edit else registry_source)
    consumer = package / "consumer.py"
    consumer.write_text(('''from . import registry as catalog
def choose(key):
    return catalog.ACTIVATIONS[key]
''' if namespace else '''from .registry import ACTIVATIONS
def choose(key):
    return ACTIVATIONS[key]
''') + consumer_tail)
    bundle = SourceBundle(source="path", files=(str(registry), str(consumer)),
                          component_files={"root": (str(registry), str(consumer))})
    index = build_program_index(bundle)
    assignment = next(item for item in index.module_assignments
                      if item.symbol.qualified_name == "ACTIVATIONS")
    backing = next(item for item in index.dispatch_registries
                   if item.symbol.qualified_name == "ENTRIES")
    module = next(item for item in index.modules
                  if item.source.source_id.canonical_path == str(consumer))
    access = next(item for item in module.name_accesses
                  if item.expression.kind == "subscript" and item.context == "load")
    imported = resolve_import_reference(index, module.source.source_id, None,
                                         access.expression.children[0])
    assert imported is not None
    def check():
        _registry_use_closure(index, assignment, backing, assignment.value.children[0], imported)
        if full:
            from model_unfolder.evidence.activation_registry import (
                _instantiated_lookup, _returned_activation, _selected_activation_use_closure)
            selected, arguments, _lookup, _spans = _instantiated_lookup(index,
                assignment.symbol.source, assignment.value.children[0], backing.entries[0][1])
            _selected_activation_use_closure(index, assignment.symbol.source, selected, backing)
            return _returned_activation(index, assignment.symbol.source, selected, arguments)
    return check


@pytest.mark.parametrize("namespace", [False, True])
def test_registry_use_census_accepts_exact_read_only_imports(tmp_path, namespace):
    _registry_use_fixture(tmp_path, namespace=namespace)()


@pytest.mark.parametrize("where,mutation", [
    ("registry", '\nENTRIES["relu"] = nn.SiLU\n'),
    ("registry", '\nACTIVATIONS["relu"] = nn.SiLU\n'),
    ("registry", '\nACTIVATIONS.update({"relu": nn.SiLU})\n'),
    ("registry", '\nalias = ACTIVATIONS\n'),
    ("registry", '\nInstantiate.__getitem__ = lambda self, key: nn.SiLU()\n'),
    ("consumer", '\nACTIVATIONS["relu"] = object()\n'),
    ("consumer", '\ndef change():\n    ACTIVATIONS["relu"] = object()\n'),
    ("consumer", '\nACTIVATIONS.update({})\n'),
    ("consumer", '\nstashed = ACTIVATIONS\n'),
])
def test_registry_writes_and_alias_escapes_are_not_erased_by_literal_entries(tmp_path, where, mutation):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, **{where + "_tail": mutation})
    with pytest.raises(ReaderClaimUnavailable, match="write|escapes|unsupported use"):
        check()


@pytest.mark.parametrize("mutation", [
    '\ncatalog.ACTIVATIONS["relu"] = object()\n',
    '\ncatalog.ACTIVATIONS.update({})\n',
    '\nstashed = catalog\n',
    '\nsetattr(catalog, "ACTIVATIONS", {})\n',
])
def test_registry_namespace_import_cannot_hide_writes_or_escape(tmp_path, mutation):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, namespace=True, consumer_tail=mutation)
    with pytest.raises(ReaderClaimUnavailable, match="write|escapes|unsupported use"):
        check()


def test_unavailable_activation_declares_intended_kind_without_proof_or_document_seal():
    from test_ffn_schedule import _config
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.ffn_mechanism import decoder_ffn_mechanism_for_path
    config = _config("llama-7b")
    config["hidden_act"] = "not_a_registry_key"
    context = ParseContext.build(config)
    document = PreparedDocument(dict(config), dict(config))
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_ffn_mechanism_for_path(context.program_index(), context.source_bundle,
                                               (), allow_root_stage=True)
        fact = EvidenceFact(key="activation", owner="decoder.ffn", value=config["hidden_act"],
                            status="code_and_config", completeness="complete")
        declared = qualify_reader_fact(fact, result, context.program_index(), document)
        assert declared.claim_kind == "applied_function"
        assert declared.claim_readers == (result.claim_witness.reader_symbol,)
        assert declared.claim_evidence is None
        assert not declared.claim_document_token
        assert declared.value == fact.value and declared.status == fact.status
        with pytest.raises(ValueError, match="actual result"):
            qualify_reader_fact(fact, replace(result), context.program_index(), document)
        with pytest.raises(ValueError, match="cannot author"):
            qualify_reader_fact(replace(fact, key="unrelated"), result,
                                 context.program_index(), document)


def test_registry_factory_has_a_complete_supported_lookup_body(tmp_path):
    kind, _target, _spans = _registry_use_fixture(tmp_path, full=True)()
    assert kind == "relu"


@pytest.mark.parametrize("old,new", [
    ("class Instantiate(OrderedDict):", "class Instantiate(OrderedDict, metaclass=Other):"),
    ("class Instantiate(OrderedDict):", "@wrapped\nclass Instantiate(OrderedDict):"),
    ("        return cls(**kwargs)", "        return cls(**kwargs)\n    __getitem__ = replacement"),
    ("        return cls(**kwargs)", "        del self[key]\n        return cls(**kwargs)"),
    ("        return cls(**kwargs)", "        yield None\n        return cls(**kwargs)"),
])
def test_registry_factory_cannot_hide_metaclass_override_or_unaccounted_statements(tmp_path, old, new):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, full=True,
        registry_edit=lambda source: source.replace(old, new))
    with pytest.raises(ReaderClaimUnavailable):
        check()


def _with_local_registry_activation(source):
    return source.replace('ENTRIES = {"relu": nn.ReLU}', '''class Chosen(nn.Module):
    def forward(self, value):
        return nn.functional.relu(value)
ENTRIES = {"relu": Chosen}''')


def test_selected_local_activation_has_one_unchanged_callable_binding(tmp_path):
    kind, target, _spans = _registry_use_fixture(tmp_path, full=True,
        registry_edit=_with_local_registry_activation)()
    assert kind == "relu"
    assert target.qualified_name == "Chosen.forward"


@pytest.mark.parametrize("tail", [
    '\nChosen.forward = replacement\n',
    '\nChosen = nn.SiLU\n',
    '\nstashed = Chosen\n',
])
def test_selected_local_activation_rebinding_mutation_or_escape_cannot_borrow_old_forward(tmp_path, tail):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, full=True, registry_tail=tail,
                                  registry_edit=_with_local_registry_activation)
    with pytest.raises(ReaderClaimUnavailable, match="selected activation"):
        check()


@pytest.mark.parametrize("tail", [
    '\nfrom .registry import Chosen\nChosen.forward = replacement\n',
    '\nfrom .registry import Chosen\nstashed = Chosen\n',
])
def test_consumer_cannot_mutate_or_export_the_selected_local_activation_class(tmp_path, tail):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, full=True, consumer_tail=tail,
                                  registry_edit=_with_local_registry_activation)
    with pytest.raises(ReaderClaimUnavailable, match="selected activation"):
        check()


@pytest.mark.parametrize("tail", [
    '\nnn.ReLU.forward = replacement\n',
    '\nnn.ReLU = nn.SiLU\n',
    '\nstashed = nn.ReLU\n',
])
def test_selected_imported_activation_cannot_be_mutated_or_escape(tmp_path, tail):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, full=True, registry_tail=tail)
    with pytest.raises(ReaderClaimUnavailable, match="selected activation"):
        check()


@pytest.mark.parametrize("old,new", [
    ("def __getitem__(self, key):", "async def __getitem__(self, key):"),
    ("def __getitem__(self, key):", "def __getitem__(self, *, key):"),
    ("def forward(self, value):", "async def forward(self, value):"),
    ("def forward(self, value):", "def forward(self, *, value):"),
])
def test_registry_lookup_and_selected_activation_require_synchronous_positional_calls(tmp_path, old, new):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, full=True,
        registry_edit=lambda source: _with_local_registry_activation(source).replace(old, new))
    with pytest.raises(ReaderClaimUnavailable):
        check()


@pytest.mark.parametrize("scale", ["", ", scale=None", ", scale=0.5", ", scale=self.width ** -0.5", ", **extra",
    ", None, attn_mask=None", ", None, 0.0, dropout_p=0.0", ", None, 0.0, False, is_causal=False"])
def test_score_claim_proves_only_actual_unpacked_sdpa_default(tmp_path, scale):
    from test_attention_mechanism import _pipeline, _split_attention
    from model_unfolder.evidence.attention import decoder_attention_score_scaling_for_path
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    source = _split_attention("config.query_groups", "config.query_groups").replace(
        "        score = torch.matmul(one, two.transpose(-1, -2))\n"
        "        return torch.matmul(F.softmax(score, dim=-1), three)",
        "        extra = {}\n"
        f"        return F.scaled_dot_product_attention(one, two, three{scale})")
    index, bundle, _root, _block = _pipeline(tmp_path, source)
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_attention_score_scaling_for_path(index, bundle, (), allow_root_stage=True)
        assert result.status == "resolved", result.failures
        if scale not in {"", ", scale=None"}:
            with pytest.raises(ReaderClaimUnavailable):
                ReaderProjectionClaimProof("decoder.attention.scores_scale", result, index, document)
        else:
            proof = ReaderProjectionClaimProof("decoder.attention.scores_scale", result, index, document)
            assert proof.claim_kind == "applied_function"
            assert proof.projection().value == "sqrt(head_dim)"  # existing denominator label
            assert proof.projection().status == "code_proven"


@pytest.mark.parametrize("multiplier", ["", " * 0.5"])
def test_raw_score_proof_does_not_promote_an_arbitrary_multiplier_to_square_root(tmp_path, multiplier):
    from test_attention_mechanism import _pipeline, _split_attention
    from model_unfolder.evidence.attention import decoder_attention_score_scaling_for_path
    source = _split_attention("config.query_groups", "config.query_groups").replace(
        "score = torch.matmul(one, two.transpose(-1, -2))",
        "score = torch.matmul(one, two.transpose(-1, -2))" + multiplier)
    index, bundle, _root, _block = _pipeline(tmp_path, source)
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_attention_score_scaling_for_path(index, bundle, (), allow_root_stage=True)
        assert result.status == "resolved", result.failures
        value = "sqrt(head_dim)" if multiplier else "unscaled (raw QK^T)"
        fact = _native_fact(result, owner="decoder.attention", key="scores_scale", value=value)
        qualified = qualify_reader_fact(fact, result, index, document)
        assert qualified.value == value
        assert qualified.claim_kind == "applied_function"
        assert (qualified.claim_evidence is None) is bool(multiplier)
        if multiplier:
            assert not qualified.claim_document_token


@pytest.mark.parametrize("tail", [
    '\nnn.functional.relu = nn.functional.silu\n',
    '\nprimitive_alias = nn.functional.relu\n',
])
def test_selected_forward_terminal_primitive_cannot_change_or_escape(tmp_path, tail):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, full=True, registry_tail=tail,
                                  registry_edit=_with_local_registry_activation)
    with pytest.raises(ReaderClaimUnavailable, match="selected activation"):
        check()


def test_selected_forward_terminal_does_not_accept_invented_constant_keywords(tmp_path):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, full=True,
        registry_edit=lambda source: _with_local_registry_activation(source).replace(
            "nn.functional.relu(value)", "nn.functional.relu(value, invented=True)"))
    with pytest.raises(ReaderClaimUnavailable, match="unsupported arguments"):
        check()


def test_unrelated_broad_torch_attribute_read_does_not_claim_selected_callable_escape(tmp_path):
    check = _registry_use_fixture(tmp_path, full=True, consumer_tail=
        '\nimport torch\ndef dtype_lookup(name):\n    dtype = getattr(torch, name)\n    return dtype\n')
    assert check()[0] == "relu"


@pytest.mark.parametrize("tail", [
    '\nimport torch\ngetattr(torch, "nn").ReLU = replacement\n',
    '\nimport torch\nnamespace = getattr(torch, "nn")\nnamespace.ReLU = replacement\n',
    '\nimport torch\nnamespace = getattr(torch, requested)\nnamespace.ReLU = replacement\n',
    '\nimport torch\nnamespace = getattr(torch, requested)\nother = namespace\nother.ReLU = replacement\n',
    '\nfrom torch import nn\nchosen = getattr(nn, "ReLU")\nchosen.forward = replacement\n',
])
def test_reflected_selected_namespace_or_callable_cannot_hide_a_receiver_write(tmp_path, tail):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, full=True, consumer_tail=tail)
    with pytest.raises(ReaderClaimUnavailable, match="activation.*(namespace|ancestor)|reflected activation"):
        check()


@pytest.mark.parametrize("key,target,canonical,qualified", [
    ("relu", "ReLU", "relu", True),
    ("relu", "SiLU", "silu", False),
    ("neutral_slot", "SiLU", "silu", False),
    ("swish", "SiLU", "silu", True),
])
def test_registry_target_establishes_function_not_its_selected_key(tmp_path, key, target, canonical, qualified):
    from test_ffn_mechanism import _reader
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.evidence.ffn_mechanism import decoder_ffn_mechanism_for_path
    _reader(tmp_path, '''
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.down(self.act(self.up(x)))
''')
    registry = tmp_path / "transformers" / "activations.py"
    registry.parent.mkdir()
    registry.write_text('''from collections import OrderedDict
from torch import nn
class Instantiate(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)
''' + f'ENTRIES = {{{key!r}: nn.{target}}}\nACT2FN = Instantiate(ENTRIES)\n')
    source = str(tmp_path / "model.py")
    bundle = SourceBundle(source="local", files=(source,),
        component_files={"root": (source,)},
        supporting_files={"root": (str(registry),)},
        architecture="Wrapper", component_architectures={"root": "Wrapper"})
    index = build_program_index(bundle)
    config = {"hidden_act": key, "hidden": 8, "wide": 16, "layers": 2}
    document = PreparedDocument(dict(config), dict(config))
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_ffn_mechanism_for_path(index, bundle, (), allow_root_stage=True)
        assert result.status == "resolved", result.failures
        applications = result.claim_witness.activation_applications
        assert len(applications) == 1, result.claim_witness.activation_gap
        assert applications[0].mechanism == canonical
        assert applications[0].key == key
        proved = ReaderProjectionClaimProof("decoder.ffn.activation", result, index, document).projection()
        assert proved.value == ("swish" if key == "swish" else canonical)
        fact = replace(_native_fact(result, key="activation", owner="decoder.ffn", value=key),
                       status="code_and_config", config_paths=("hidden_act",))
        output = qualify_reader_fact(fact, result, index, document)
        assert output.value == key  # Legacy output correction requires owner approval.
        assert output.claim_kind == "applied_function"
        assert bool(output.claim_evidence) is qualified
        if not qualified:
            assert not output.claim_document_token
        assert ReaderProjectionClaimProof("decoder.ffn.gated", result, index, document).projection().value is False


@pytest.mark.parametrize("mutation", [
    "F.scaled_dot_product_attention = replacement",
    "F = replacement",
    "terminal_alias = F.scaled_dot_product_attention",
])
def test_sdpa_default_claim_refuses_changed_or_escaped_imported_terminal(tmp_path, mutation):
    from test_attention_mechanism import _pipeline, _split_attention
    from model_unfolder.evidence.attention import decoder_attention_score_scaling_for_path
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    source = _split_attention("config.query_groups", "config.query_groups").replace(
        "        score = torch.matmul(one, two.transpose(-1, -2))\n"
        "        return torch.matmul(F.softmax(score, dim=-1), three)",
        "        return F.scaled_dot_product_attention(one, two, three)")
    index, bundle, _root, _block = _pipeline(tmp_path, source + "\n" + mutation + "\n")
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)):
        result = decoder_attention_score_scaling_for_path(index, bundle, (), allow_root_stage=True)
        if result.status == "resolved":
            with pytest.raises(ReaderClaimUnavailable, match="selected activation"):
                ReaderProjectionClaimProof("decoder.attention.scores_scale", result, index, document)
        else:
            # Earlier source resolution may already reject a rebound import.
            assert result.claim_witness is None


@pytest.mark.parametrize("tail", [
    '\nimport torch\nrequested = "nn"\nalias = getattr(torch, requested)\nsetattr(alias, "ReLU", replacement)\n',
    '\nimport torch\nalias = getattr(torch, requested)\nother = alias\ndelattr(other, "ReLU")\n',
    '\nimport torch\nholder.member = getattr(torch, requested)\nsetattr(holder.member, "ReLU", replacement)\n',
    '\nimport torch\nholder["member"] = getattr(torch, requested)\nholder["member"].ReLU = replacement\n',
    '\nimport torch\nalias = getattr(torch, requested)\nholder.member = alias\nsetattr(holder.member, "ReLU", replacement)\n',
    '\nimport torch\nalias = getattr(torch, requested)\nchosen = getattr(alias, "ReLU")\nchosen.forward = replacement\n',
])
def test_reflected_ancestor_alias_cannot_hide_builtin_mutation_or_receiver_storage(tmp_path, tail):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, full=True, consumer_tail=tail)
    with pytest.raises(ReaderClaimUnavailable, match="reflected activation ancestor"):
        check()


@pytest.mark.parametrize("body", [
    'def lookup(getattr, name):\n    return getattr(torch, name)\n',
    'def lookup(name):\n    getattr = replacement\n    return getattr(torch, name)\n',
    'def lookup(hasattr, name):\n    return hasattr(torch, name)\n',
    'def outer(getattr):\n    def lookup(name):\n        return getattr(torch, name)\n    return lookup\n',
])
def test_broad_ancestor_getter_is_bound_against_lexical_parameters_and_stores(tmp_path, body):
    from model_unfolder.evidence.reader_claims import ReaderClaimUnavailable
    check = _registry_use_fixture(tmp_path, full=True, consumer_tail='\nimport torch\n' + body)
    with pytest.raises(ReaderClaimUnavailable, match="opaque call"):
        check()
