"""Exact embedding state width, its default channel, and refusal boundaries."""
from dataclasses import replace

import pytest

from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.facts import EvidenceFact, SourceSpan
from model_unfolder.evidence.model_state_width import model_state_width_for_path
from model_unfolder.evidence.reader_claims import qualify_reader_fact
from test_embedding_bookend import _pipeline, _SOURCE


SOURCE = _SOURCE.replace('hidden = self.entry(inputs_embeds)', 'hidden = inputs_embeds')


def _read(tmp_path, source=SOURCE, *, omitted=False, width=64):
    bundle, index, *_ = _pipeline(tmp_path, source)
    checkpoint = {'vocab': 100, 'layers': 2}
    if not omitted:
        checkpoint['hidden'] = width
    document = PreparedDocument(dict(checkpoint), dict(checkpoint),
                                {'hidden': width} if omitted else {})
    with bound_document(DocumentBinding('root', (), document)):
        result = model_state_width_for_path(index, bundle, (), allow_root_stage=False)
    return result, index, document


def _fact(result, index, document):
    witness = result.claim_witness
    projection = witness.project('model', 'hidden_size', document)
    fact = EvidenceFact(key='hidden_size', owner='model', value=projection.value,
        status=projection.status, completeness='presence_only',
        config_paths=tuple('.'.join(path) for path in projection.config_paths),
        source_spans=tuple(dict.fromkeys(SourceSpan(component=span.source.component_key or 'root',
                             file=span.source.canonical_path, line=span.line)
                             for origin in result.provenance for span in origin.spans)))
    return qualify_reader_fact(fact, result, index, document)


@pytest.mark.parametrize('omitted,status', [(False, 'code_and_config'), (True, 'class_default')])
def test_actual_embedding_operand_proves_width_in_both_value_channels(tmp_path, omitted, status):
    result, index, document = _read(tmp_path, omitted=omitted)
    assert result.status == 'resolved', result.failures
    assert result.value.value == 64
    assert result.value.source_path == ('hidden',)  # no hidden_size spelling assumption
    assert result.value.external_input_unmeasured is True
    fact = _fact(result, index, document)
    assert (fact.value, fact.status, fact.claim_kind, fact.completeness) == (64, status, 'value', 'presence_only')
    assert fact.config_paths == (() if omitted else ('hidden',))
    validate_fact_claim(fact, fact.claim_evidence)
    assert fact.claim_evidence.summary().reader_symbols == (result.claim_witness.reader_symbol,)
    assert result.value.spans and result.value.repeated_sites


def test_unconditional_embedding_route_has_no_external_input_claim(tmp_path):
    source = SOURCE.replace('            if inputs_embeds is None:\n                inputs_embeds = self.embedding(token_ids)',
                            '            inputs_embeds = self.embedding(token_ids)')
    result, index, document = _read(tmp_path, source)
    assert result.status == 'resolved', result.failures
    assert result.value.external_input_unmeasured is False
    assert _fact(result, index, document).value == 64


@pytest.mark.parametrize('replacement', [
    'hidden = self.entry(inputs_embeds)',  # actual intervening operation not a direct alias
    'hidden = inputs_embeds[..., :16]',
    'hidden = inputs_embeds.reshape(1, -1)',
    'hidden = inputs_embeds * 0',
    'hidden = token_ids',
    'inputs_embeds.resize_(1, 16)\n            hidden = inputs_embeds',
    'inputs_embeds[...] = 0\n            hidden = inputs_embeds',
    'alias = inputs_embeds\n            alias.resize_(1, 16)\n            hidden = inputs_embeds',
])
def test_transformation_missing_connection_and_alias_mutation_stay_unavailable(tmp_path, replacement):
    result, _index, _document = _read(tmp_path, SOURCE.replace('hidden = inputs_embeds', replacement))
    assert result.status == 'failed'
    assert result.claim_witness is None


@pytest.mark.parametrize('constructor', [
    'nn.Linear(config.vocab, config.hidden)',
    'nn.Embedding(config.vocab, config.hidden // 2)',
    'nn.Embedding(config.vocab, config.hidden, embedding_dim=17)',
    'nn.Embedding(config.vocab, config.hidden, num_embeddings=100)',
    'nn.Embedding(config.vocab, config.hidden, _weight=config.weight)',
])
def test_bottleneck_wrong_primitive_or_conflicting_constructor_is_not_width_proof(tmp_path, constructor):
    source = SOURCE.replace('nn.Embedding(config.vocab, config.hidden)', constructor)
    result, _index, _document = _read(tmp_path, source)
    assert result.status == 'failed'


@pytest.mark.parametrize('value', [None, True, 0, -1])
def test_explicit_invalid_width_is_never_filled_from_a_default(tmp_path, value):
    result, _index, _document = _read(tmp_path, width=value)
    assert result.status == 'failed'


@pytest.mark.parametrize('change', [
    {'value': 32}, {'status': 'code_proven'}, {'owner': 'other'},
    {'key': 'other'}, {'claim_kind': 'relation'}, {'completeness': 'complete'},
    {'source_spans': ()},
])
def test_qualified_width_rejects_changed_value_owner_scope_or_completeness(tmp_path, change):
    result, index, document = _read(tmp_path, omitted=True)
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, document)
    with pytest.raises((ValueError, TypeError)):
        replace(fact, **change)


def test_default_change_after_read_invalidates_original_proof(tmp_path):
    result, index, document = _read(tmp_path, omitted=True)
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, document)
    document.class_overlay['hidden'] = 32
    with pytest.raises((ValueError, TypeError)):
        validate_fact_claim(fact, fact.claim_evidence)


def test_copied_resolved_dto_cannot_supply_width_authority(tmp_path):
    result, index, document = _read(tmp_path)
    assert result.status == 'resolved', result.failures
    with pytest.raises((ValueError, TypeError)):
        _fact(replace(result), index, document)


def test_same_width_explicitly_different_source_operand_does_not_get_guessed(tmp_path):
    source = SOURCE.replace('config.hidden)', 'config.other_width)')
    result, _index, _document = _read(tmp_path, source)
    assert result.status == 'failed'  # actual source property has no supplied/default value


@pytest.mark.parametrize('source', [
    SOURCE.replace('hidden = unit(hidden)', 'hidden = unit(hidden, x=hidden)'),
    SOURCE.replace('hidden = unit(hidden)', 'other = unit(hidden)'),
    SOURCE.replace('def forward(self, token_ids, inputs_embeds=None):', 'async def forward(self, token_ids, inputs_embeds=None):'),
])
def test_recurrence_argument_conflict_and_coroutine_are_not_state_width(tmp_path, source):
    result, _index, _document = _read(tmp_path, source)
    assert result.status == 'failed'


def test_registered_default_width_attaches_only_a_cited_presentation_chip(tmp_path):
    from model_unfolder.evidence.context import FactLedger
    from model_unfolder.evidence.presentation_projection import attach_fact_chips
    from model_unfolder.evidence.registry import fact_definition, validate_typed_write
    result, index, document = _read(tmp_path, omitted=True)
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, document)
    assert validate_typed_write(fact) == []
    definition = fact_definition('hidden_size')
    assert definition.owner_patterns == frozenset({'model'})
    assert definition.parameter_consumer
    ledger = FactLedger()
    ledger.record_typed(fact)
    block = {'id': 'embed', 'kind': 'embedding', 'source_fact_keys': ['model.hidden_size']}
    ir = {'layers': [], 'extras': {'fact_provenance': ledger.to_dict(),
                                  'render': {'model_blocks': [block]}}}
    assert attach_fact_chips(ir, ledger.typed_records()) == ()
    assert block['presentation_chips'][0]['chip_kind'] == 'class_default'
    assert len(ir['extras']['render']['model_blocks']) == 1
