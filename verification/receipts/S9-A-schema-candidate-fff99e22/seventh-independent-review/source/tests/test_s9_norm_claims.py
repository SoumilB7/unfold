"""Norm facts retain actual primitive/call/order proofs, not plausible labels."""
from dataclasses import replace

import pytest

from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.embedding_bookend import embedding_stage_norm_evidence
from model_unfolder.evidence.reader_claims import ReaderProjectionClaimProof, qualify_reader_fact
from model_unfolder.evidence.reader_result import ReaderResult
from test_decoder_norm import _read
from test_embedding_bookend import _pipeline, _SOURCE
from test_reader_claims import _native_fact


def _decoder(tmp_path):
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding('root', (), document)):
        result = _read(tmp_path, '', block_norms='''
        self.n1 = nn.LayerNorm(config.hidden)
        self.n2 = nn.LayerNorm(config.hidden)
''')
        assert result.status == 'resolved' and result.value == 'layernorm'
        index = result.claim_witness.index
        fact = qualify_reader_fact(_native_fact(result, owner='decoder.layer',
            key='norm_kind', value='layernorm'), result, index, document)
    return result, index, document, fact


def _embedding(tmp_path):
    bundle, index, *_ = _pipeline(tmp_path)
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding('root', (), document)):
        result = embedding_stage_norm_evidence(index, bundle)
        assert result.status == 'resolved' and result.value == 'RMSNorm'
        fact = qualify_reader_fact(_native_fact(result, key='embedding_norm_kind'),
                                   result, index, document)
    return result, index, document, fact


@pytest.mark.parametrize('reader', [_decoder, _embedding])
def test_actual_norm_claims_keep_original_values_and_source_lineage(tmp_path, reader):
    result, index, document, fact = reader(tmp_path)
    assert fact.status == 'code_proven' and fact.claim_kind == 'applied_function'
    assert fact.completeness == ('presence_only' if reader is _decoder else 'complete')
    validate_fact_claim(fact, fact.claim_evidence)
    summary = fact.claim_evidence.summary()
    assert summary.reader_symbols == (result.claim_witness.reader_symbol,)
    assert summary.index_fingerprints and summary.document_fingerprints
    assert summary.evidence_refs
    assert all('/' not in ref for ref in summary.evidence_refs)
    assert fact.source_spans
    if reader is _decoder:
        retained = result.claim_witness.owner_claims
        assert retained and all(claim.census.completeness == 'partial' for claim in retained)
        assert all(claim.examined and claim.invocations.call_sites for claim in retained)
    else:
        retained = result.claim_witness
        assert retained.edge.proof_kind == 'versioned_def_use'
        assert retained.call.guard == ()
        assert retained.edge.target.call_site in {proof.template.call_site for proof in retained.repeated.proofs}


@pytest.mark.parametrize('reader', [_decoder, _embedding])
@pytest.mark.parametrize('change', [
    {'value': 'invented'}, {'status': 'class_default'}, {'source_spans': ()},
    {'owner': 'other'}, {'key': 'other'}, {'claim_kind': 'value'},
])
def test_qualified_norm_fact_cannot_change_any_claim_axis(tmp_path, reader, change):
    *_, fact = reader(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        replace(fact, **change)


@pytest.mark.parametrize('reader', [_decoder, _embedding])
def test_norm_strings_spans_and_copied_witnesses_cannot_issue_claims(tmp_path, reader):
    result, index, document, fact = reader(tmp_path)
    copied = replace(result)
    with pytest.raises(ValueError, match='actual result'):
        ReaderProjectionClaimProof(fact.ledger_key(), copied, index, document)
    fabricated = ReaderResult.resolved(result.owner, result.value, provenance=result.provenance)
    raw = _native_fact(result, key=fact.key, owner=fact.owner, value=fact.value)
    assert qualify_reader_fact(raw, fabricated, index, document) is raw
    with pytest.raises(ValueError, match='another fact projection'):
        ReaderProjectionClaimProof('unrelated.kind', result, index, document)
    other_document = PreparedDocument({}, {})
    with pytest.raises(ValueError, match='prepared document'):
        ReaderProjectionClaimProof(fact.ledger_key(), result, index, other_document)
    document.document['changed'] = True
    with pytest.raises(ValueError, match='prepared document'):
        validate_fact_claim(fact, fact.claim_evidence)


def test_decoder_proof_cannot_remove_a_candidate_or_change_primitive_family(tmp_path):
    result, _, _, _ = _decoder(tmp_path)
    proof = result.claim_witness
    with pytest.raises(ValueError, match='every selected candidate'):
        replace(proof, owner_claims=()).validate_result(result)
    owner = proof.owner_claims[0]
    with pytest.raises(ValueError, match='omitted or changed an examined invocation'):
        replace(owner, examined=owner.examined[:-1]).validate()
    call, construction, primitive = next(row for row in owner.examined
                                        if row[2].status == 'resolved' and row[2].value == 'layernorm')
    poisoned = tuple((call, construction, replace(primitive, value='rmsnorm'))
                     if row[0] is call else row for row in owner.examined)
    with pytest.raises(ValueError, match='actual source derivation'):
        replace(owner, examined=poisoned).validate()
    foreign_call = replace(call)
    poisoned = tuple((foreign_call, construction, primitive) if row[0] is call else row
                     for row in owner.examined)
    with pytest.raises(ValueError):
        replace(owner, examined=poisoned).validate()


def test_decoder_rival_or_unproved_norms_stay_explicit_and_unqualified(tmp_path):
    result = _read(tmp_path, '', block_norms='''
        self.n1 = nn.LayerNorm(config.hidden)
        self.n2 = nn.RMSNorm(config.hidden)
''')
    assert result.status == 'ambiguous' and result.claim_witness is None


def test_embedding_proof_cannot_reverse_edge_guard_call_or_change_primitive(tmp_path):
    result, _, _, _ = _embedding(tmp_path)
    proof = result.claim_witness
    reversed_edge = replace(proof.edge, source=proof.edge.target, target=proof.edge.source)
    with pytest.raises(ValueError):
        replace(proof, edge=reversed_edge).validate_result(result)
    with pytest.raises(ValueError):
        replace(proof, call=replace(proof.call, guard=('conditional',))).validate_result(result)
    with pytest.raises(ValueError, match='actual source derivation'):
        replace(proof, primitive=replace(proof.primitive, value='layernorm')).validate_result(result)
    with pytest.raises(ValueError, match='provenance'):
        proof.validate_result(replace(result, provenance=()))


def test_embedding_non_norm_input_keeps_original_failure_without_claim(tmp_path):
    bundle, index, *_ = _pipeline(tmp_path, _SOURCE.replace(
        'self.entry = CustomNorm(config)', 'self.entry = Unit(config)'))
    result = embedding_stage_norm_evidence(index, bundle)
    assert result.status == 'failed' and result.claim_witness is None
    assert result.failures
