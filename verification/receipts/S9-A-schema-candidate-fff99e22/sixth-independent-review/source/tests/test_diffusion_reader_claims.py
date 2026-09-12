"""S9-A diffusion proofs retain actual source calls and whole finite claims."""
from dataclasses import replace

import pytest

from model_unfolder.adapters.diffusor.projection_ir import project_diffusion_ir
from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.context import FactLedger, capture_facts
from model_unfolder.evidence.diffusion_stream import read_diffusion_stream_graph
from model_unfolder.evidence.document import DocumentBinding
from model_unfolder.evidence.reader_claims import qualify_reader_fact
from test_diffusion_config_binding import _bind


def _projected(tmp_path):
    result, _config_ledger = _bind(tmp_path)
    bound = result.require_value()
    facts = FactLedger()
    with capture_facts(facts):
        projection = project_diffusion_ir(bound)
    return bound, projection, facts.typed_records()


def test_actual_diffusion_facts_retain_finite_producer_proofs(tmp_path):
    bound, projection, facts = _projected(tmp_path)
    assert projection.layers
    expected = {
        "root.denoiser.diffusion_root_topology": "connection",
        "root.denoiser.stacks[0].diffusion_stack_depth": "value",
        "root.denoiser.stacks[0].attention[0].diffusion_attention_head_protocol": "connection",
        "root.denoiser.stacks[0].cell.diffusion_norm_mechanism": "applied_function",
        "root.denoiser.stacks[0].ffn.diffusion_ffn_mechanism": "connection",
    }
    for key, kind in expected.items():
        fact = facts[key]
        assert fact.claim_kind == kind
        validate_fact_claim(fact, fact.claim_evidence)
    block_result, stream_result, conditioning_result, bookend_result = bound.source.reader_results
    assert block_result.value is bound.source.block_inventory
    assert stream_result.value is bound.source.stream_inventory
    assert stream_result.claim_witness.block_result is block_result
    assert conditioning_result.claim_witness.stream_result is stream_result
    assert bookend_result.claim_witness.dependencies[1] is stream_result


@pytest.mark.parametrize("mutation", [
    {"kind": "mqa"}, {"num_heads": 7}, {"projection_mode": "fused"},
])
def test_compound_attention_claim_rejects_changed_member(tmp_path, mutation):
    _bound, _projection, facts = _projected(tmp_path)
    fact = facts["root.denoiser.stacks[0].attention[0].diffusion_attention_head_protocol"]
    with pytest.raises(ValueError):
        replace(fact, value={**fact.value, **mutation})


@pytest.mark.parametrize("fact_id", [
    "root.denoiser.diffusion_root_topology",
    "root.denoiser.stacks[0].attention[0].diffusion_attention_head_protocol",
])
def test_finite_question_rejects_missing_decisive_local_evidence(tmp_path, fact_id):
    _bound, _projection, facts = _projected(tmp_path)
    fact = facts[fact_id]
    required = fact.claim_evidence._required_spans
    assert required
    dropped = required[0]
    kept = tuple(span for span in fact.source_spans
                 if (span.component, span.file, span.line) != (
                     dropped.source.component_key, dropped.source.canonical_path, dropped.line))
    assert len(kept) < len(fact.source_spans)
    with pytest.raises(ValueError):
        replace(fact, source_spans=kept)


def test_real_stream_call_cannot_launder_a_reconstructed_upstream_result(tmp_path):
    bound, _projection, facts = _projected(tmp_path)
    original = facts["root.denoiser.stacks[0].ffn.diffusion_ffn_mechanism"]
    block_result = bound.source.reader_results[0]
    index = block_result.claim_witness.index
    with bound_document(DocumentBinding("root", (), bound.prepared_document)):
        reconstructed = replace(block_result)
        stream = read_diffusion_stream_graph(index, bound.config_root, reconstructed)
        assert stream.has_value
        raw = replace(original, claim_kind=None, claim_readers=(), claim_evidence=None,
                      claim_document_token="")
        with pytest.raises(ValueError, match="actual scoped block reader"):
            qualify_reader_fact(raw, stream, index, bound.prepared_document)


def test_connection_claim_cannot_be_relabelled_existence(tmp_path):
    _bound, _projection, facts = _projected(tmp_path)
    fact = facts["root.denoiser.diffusion_root_topology"]
    with pytest.raises(ValueError):
        replace(fact, claim_kind="existence")


def test_whole_supplying_document_is_sealed_for_diffusion(tmp_path):
    bound, _projection, facts = _projected(tmp_path)
    fact = facts["root.denoiser.stacks[0].attention[0].diffusion_attention_head_protocol"]
    bound.prepared_document.document["query_heads"] = 16
    with pytest.raises(ValueError):
        validate_fact_claim(fact, fact.claim_evidence)
