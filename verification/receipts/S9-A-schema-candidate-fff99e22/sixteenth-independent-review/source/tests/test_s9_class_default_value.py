"""Source-class scalar defaults do not claim model execution or connections."""
from dataclasses import replace

import pytest

from model_unfolder.evidence.class_default_value import model_hidden_size_class_default
from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.context import FactLedger
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument, checkpoint_provenance
from model_unfolder.evidence.facts import EvidenceFact, SourceSpan
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.presentation_projection import attach_fact_chips
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reader_claims import qualify_reader_fact


def _read(tmp_path, *, default='64', annotation='ExactConfig', support_component='root',
          checkpoint=None, overlay=None, config_path=(), model_suffix='', inherited=False,
          reader_document=None, failure=None):
    model = tmp_path / 'modeling_default.py'
    support = tmp_path / 'configuration_default.py'
    model.write_text(f'''from transformers.modeling_utils import PreTrainedModel
from .configuration_default import ExactConfig
class Base(PreTrainedModel):
    config: {annotation}
class Wrapper(Base):
    def __init__(self, config):
        super().__init__(config)
{model_suffix}
''')
    support.write_text(f'''class {'Parent' if inherited else 'ExactConfig'}:
    hidden_size: int = {default}
''' + ('class ExactConfig(Parent):\n    pass\n' if inherited else ''))
    bundle = SourceBundle(source='local', files=(str(model),),
        component_files={'root': (str(model),)}, supporting_files={support_component: (str(support),)},
        component_architectures={'root': 'Wrapper'}, architecture='Wrapper')
    index = build_program_index(bundle)
    checkpoint = {} if checkpoint is None else checkpoint
    document = PreparedDocument(dict(checkpoint) if reader_document is None else reader_document,
                                dict(checkpoint),
                                {'hidden_size': 64} if overlay is None else overlay,
                                provenance=checkpoint_provenance(checkpoint), failure=failure)
    with bound_document(DocumentBinding('root', (), document)):
        result = model_hidden_size_class_default(index, bundle, config_path)
    return result, index, document


def _fact(result, index, document):
    projection = result.claim_witness.project('model', 'hidden_size', document)
    fact = EvidenceFact(key='hidden_size', owner='model', value=projection.value,
        status='class_default', completeness='complete', config_paths=(),
        source_spans=tuple(dict.fromkeys(SourceSpan(component=span.source.component_key or 'root',
                        file=span.source.canonical_path, line=span.line)
                        for origin in result.provenance for span in origin.spans)))
    return qualify_reader_fact(fact, result, index, document)


def test_actual_annotated_class_literal_supplies_only_scalar_default_value(tmp_path):
    result, index, document = _read(tmp_path)
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, document)
    assert (fact.value, fact.status, fact.claim_kind, fact.completeness) == (64, 'class_default', 'value', 'complete')
    assert fact.config_paths == ()
    assert result.value.config_class.config_class.qualified_name == 'ExactConfig'
    assert result.value.declaration.assignment.attr == 'hidden_size'
    validate_fact_claim(fact, fact.claim_evidence)
    assert fact.claim_evidence.summary().reader_symbols == (result.claim_witness.reader_symbol,)


@pytest.mark.parametrize('checkpoint', [
    {'hidden_size': 64}, {'hidden_size': None}, {'d_model': 64}, {'d_model': None},
    {'hidden_size': 32, 'd_model': 64}, {'d_model': 32, 'n_embd': 64},
])
def test_explicit_values_nulls_and_alias_conflicts_never_get_class_default_proof(tmp_path, checkpoint):
    result, _index, _document = _read(tmp_path, checkpoint=checkpoint)
    assert result.status == 'failed'


@pytest.mark.parametrize('kwargs', [
    {'default': '32'}, {'default': '32 * 2'}, {'default': 'None'}, {'default': 'True'},
    {'annotation': 'ForeignConfig'}, {'support_component': 'other'},
    {'inherited': True}, {'overlay': {}}, {'overlay': {'hidden_size': 32}},
    {'config_path': ('text_config',)},
])
def test_missing_foreign_computed_or_disagreeing_default_is_unavailable(tmp_path, kwargs):
    result, _index, _document = _read(tmp_path, **kwargs)
    assert result.status == 'failed'
    assert result.claim_witness is None


def test_forward_behavior_cannot_change_a_config_default_value_declaration(tmp_path):
    result, index, document = _read(tmp_path, model_suffix='''    def forward(self, x):
        return self.other(x).resize_(1, 16)
''')
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, document)
    assert fact.value == 64 and fact.claim_kind == 'value'
    with pytest.raises((ValueError, TypeError)):
        replace(fact, claim_kind='connection')


@pytest.mark.parametrize('change', [
    {'value': 32}, {'status': 'code_proven'}, {'owner': 'foreign'}, {'key': 'other'},
    {'claim_kind': 'applied_function'}, {'source_spans': ()}, {'config_paths': ('hidden_size',)},
])
def test_scalar_claim_cannot_change_value_kind_owner_or_checkpoint_occurrence(tmp_path, change):
    result, index, document = _read(tmp_path)
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, document)
    with pytest.raises((ValueError, TypeError)):
        replace(fact, **change)


def test_foreign_document_and_copied_result_are_not_actual_reader_authority(tmp_path):
    result, index, document = _read(tmp_path)
    assert result.status == 'resolved', result.failures
    with pytest.raises((ValueError, TypeError)):
        _fact(result, index, PreparedDocument({}, {}, {'hidden_size': 64}))
    with pytest.raises((ValueError, TypeError)):
        _fact(replace(result), index, document)


def test_overlay_change_after_original_read_invalidates_qualified_default(tmp_path):
    result, index, document = _read(tmp_path)
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, document)
    document.class_overlay['hidden_size'] = 32
    with pytest.raises((ValueError, TypeError)):
        validate_fact_claim(fact, fact.claim_evidence)


def test_foreign_source_class_result_cannot_be_transplanted_into_witness(tmp_path):
    other = tmp_path / 'other'
    other.mkdir()
    result, _index, _document = _read(tmp_path)
    foreign, _other_index, _other_doc = _read(other)
    assert result.status == foreign.status == 'resolved'
    with pytest.raises((ValueError, TypeError)):
        replace(result.claim_witness, default=foreign.value).validate_result(result)


def test_qualified_scalar_default_is_a_chip_not_an_extra_architectural_node(tmp_path):
    from model_unfolder.evidence.registry import fact_definition, validate_typed_write
    result, index, document = _read(tmp_path)
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, document)
    assert validate_typed_write(fact) == []
    assert fact_definition('hidden_size').owner_patterns == frozenset({'model'})
    ledger = FactLedger()
    ledger.record_typed(fact)
    block = {'id': 'embed', 'kind': 'embedding', 'source_fact_keys': ['model.hidden_size']}
    ir = {'layers': [], 'extras': {'fact_provenance': ledger.to_dict(),
                                  'render': {'model_blocks': [block]}}}
    assert attach_fact_chips(ir, ledger.typed_records()) == ()
    assert block['presentation_chips'][0]['chip_kind'] == 'class_default'
    assert len(ir['extras']['render']['model_blocks']) == 1


@pytest.mark.parametrize('checkpoint', [{'hidden_size': 64}, {'hidden_size': None},
                                        {'d_model': 64}, {'d_model': None}])
def test_reader_document_cannot_hide_a_checkpoint_operand(tmp_path, checkpoint):
    result, _index, _document = _read(tmp_path, checkpoint=checkpoint, reader_document={})
    assert result.status == 'failed'


def test_failed_preparation_cannot_supply_a_default_even_when_overlay_agrees(tmp_path):
    from model_unfolder.evidence.document import PreparationFailure
    result, _index, _document = _read(tmp_path, failure=PreparationFailure('class_rejected', 'hydrate'))
    assert result.status == 'failed'


@pytest.mark.parametrize('omitted', [False, True])
def test_actual_parser_records_same_scalar_on_checkpoint_and_source_default_rungs(tmp_path, monkeypatch, omitted):
    from model_unfolder.adapters.transformer import parser
    from model_unfolder.evidence import class_default_value, config_access
    from model_unfolder.evidence.context import ParseContext, capture_facts
    checkpoint = {'architectures': ['Wrapper'], 'num_hidden_layers': 2, 'vocab_size': 100,
                  'num_attention_heads': 4, 'intermediate_size': 128, 'hidden_act': 'silu',
                  'tie_word_embeddings': False}
    if not omitted:
        checkpoint['hidden_size'] = 64
    result, index, document = _read(tmp_path, checkpoint=checkpoint)
    if omitted:
        assert result.status == 'resolved', result.failures
        bundle = result.claim_witness.bundle
    else:
        # This public source descriptor is data only. No source-default reader
        # is needed to prove the ordinary checkpoint scalar consumption.
        bundle = SourceBundle(source='local', files=(str(tmp_path / 'modeling_default.py'),),
            component_files={'root': (str(tmp_path / 'modeling_default.py'),)},
            supporting_files={'root': (str(tmp_path / 'configuration_default.py'),)},
            component_architectures={'root': 'Wrapper'}, architecture='Wrapper')
        def forbidden(*args, **kwargs):
            raise AssertionError('ordinary width must not invoke the default source reader')
        monkeypatch.setattr(class_default_value, 'model_hidden_size_class_default', forbidden)
    context = ParseContext(bundle, class_defaults={'hidden_size': 64},
                           class_defaults_by_path={(): {'hidden_size': 64}})
    context._program_index = index
    with bound_document(DocumentBinding('root', (), document)), \
            config_access.capture_events(context.config_access), capture_facts(context.facts):
        ir = parser.parse(document.document, context=context)
    fact = context.facts.typed_records()['model.hidden_size']
    assert ir.hidden_size == fact.value == 64
    assert fact.status == ('class_default' if omitted else 'config_declared')
    assert fact.claim_kind == 'value' and fact.completeness == 'complete'
    assert fact.claim_evidence is not None
    assert fact.config_paths == (() if omitted else ('hidden_size',))
    validate_fact_claim(fact, fact.claim_evidence)
