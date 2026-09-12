"""Unbound direct parses declare questions without inventing checkpoint proof."""
from dataclasses import replace

import pytest

from model_unfolder.evidence.claim_evidence import qualify_config_value_fact
from model_unfolder.evidence.config_access import bound_document, current_prepared_document
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.facts import EvidenceFact
from model_unfolder.evidence.final_bookend import final_stage_norm_evidence
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.reader_claims import (
    ReaderProjectionClaimProof, declare_reader_fact, qualify_reader_fact,
    validate_reader_investigation,
)
from test_support.s9_fixtures.reader_claims import _native_fact, _pipeline, _with_final_norm


@pytest.mark.parametrize('readable_source', [False, True])
def test_direct_parse_keeps_explicit_values_without_forging_a_document(tmp_path, readable_source):
    from model_unfolder.adapters.transformer import parser
    assert current_prepared_document.get() is None
    if readable_source:
        bundle, index, *_ = _pipeline(tmp_path, _with_final_norm())
        context = ParseContext(bundle)
        context._program_index = index
    else:
        context = ParseContext(SourceBundle(source='local', files=()))
    config = {'hidden_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2,
              'num_key_value_heads': 2, 'intermediate_size': 16, 'vocab_size': 32,
              'tie_word_embeddings': True}
    ir = parser.parse(config, context=context)
    assert ir.hidden_size == 8 and len(ir.layers) == 2
    for key, expected in [('model.hidden_size', 8), ('model.tie_word_embeddings', True)]:
        fact = context.facts.typed_records()[key]
        assert fact.value == expected and fact.status == 'config_declared'
        assert fact.claim_kind == 'value' and fact.claim_readers
        assert fact.claim_evidence is None and not fact.claim_document_token
        assert fact.occurrence_citation is None
    assert current_prepared_document.get() is None
    assert context.prepared_documents == {}


def test_unbound_reader_declares_only_its_actual_question_and_cannot_gain_document_authority(tmp_path):
    assert current_prepared_document.get() is None
    bundle, index, *_ = _pipeline(tmp_path, _with_final_norm())
    result = final_stage_norm_evidence(index, bundle)
    assert result.status == 'resolved'
    fact = _native_fact(result)
    declared = qualify_reader_fact(fact, result, index, None)
    assert declared.claim_kind == 'applied_function'
    assert declared.claim_readers == (result.claim_reader_symbol,)
    assert declared.claim_evidence is None and not declared.claim_document_token
    assert declared.occurrence_citation is None
    with pytest.raises(ValueError, match='actual retained reader invocation'):
        declare_reader_fact(fact, replace(result), index, None)
    with pytest.raises(ValueError, match='another source index or prepared document'):
        declare_reader_fact(fact, result, object(), None)
    forged = PreparedDocument({}, {})
    with pytest.raises(ValueError, match='another source index or prepared document'):
        declare_reader_fact(fact, result, index, forged)
    with pytest.raises(ValueError, match='prepared document'):
        ReaderProjectionClaimProof(fact.ledger_key(), result, index, None)
    with bound_document(DocumentBinding('root', (), forged)):
        bound = final_stage_norm_evidence(index, bundle)
    with pytest.raises(ValueError, match='another source index or prepared document'):
        declare_reader_fact(_native_fact(bound), bound, index, None)


def test_missing_document_still_cannot_qualify_a_value_or_investigated_exhaustion(tmp_path):
    value = EvidenceFact(owner='model', key='hidden_size', value=8,
                         status='config_declared', completeness='complete',
                         claim_kind='value', claim_readers=('adapters.transformer.parser.parse',))
    with pytest.raises(TypeError, match='exact prepared document'):
        qualify_config_value_fact(value, (), None)
    bundle, index, *_ = _pipeline(tmp_path)
    result = final_stage_norm_evidence(index, bundle)
    assert result.status in {'failed', 'ambiguous', 'incomplete', 'absent'}
    with pytest.raises(ValueError, match='no exact prepared-document mapping binding'):
        validate_reader_investigation(result, owner='model', key='final_norm_kind',
                                      claim_kind='applied_function')
