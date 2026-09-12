"""Source-class scalar defaults do not claim model execution or connections."""

from test_support.s9_fixtures.s9_class_default_value import _fact, _read
from dataclasses import replace

import pytest

from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.context import FactLedger
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.presentation_projection import attach_fact_chips






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


@pytest.mark.parametrize('where, statement', [
    ('before_super', 'config.hidden_size = 32'),
    ('after_super', 'config.hidden_size = 32'),
    ('after_super', 'config.hidden_size += 1'),
    ('after_super', 'config = other'),
    ('after_super', 'alias = config\n        alias.hidden_size = 32'),
    ('after_super', 'alias = config\n        other = alias\n        other.hidden_size = 32'),
    ('after_super', 'alias = config\n        if flag:\n            alias.hidden_size = 32'),
    ('after_super', 'self.config.hidden_size = 32'),
    ('after_super', 'alias = self.config\n        alias.hidden_size = 32'),
    ('after_super', 'config["hidden_size"] = 32'),
])
def test_relevant_root_constructor_override_refuses_default_proof(tmp_path, where, statement):
    result, _index, _document = _read(tmp_path, **{where: '        ' + statement + '\n'})
    assert result.status == 'failed'
    assert result.claim_witness is None


@pytest.mark.parametrize('statement', [
    'config.other = 32',
    'alias = config\n        alias.other = 32',
    'width = config.hidden_size',
])
def test_unrelated_constructor_fields_and_reads_do_not_override_scalar_default(tmp_path, statement):
    result, index, document = _read(tmp_path, after_super='        ' + statement + '\n')
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, document)
    assert fact.value == 64 and fact.claim_kind == 'value'


@pytest.mark.parametrize('statement', [
    "setattr(config, 'hidden_size', 32)",
    "config.__setattr__('hidden_size', 32)",
    "object.__setattr__(config, 'hidden_size', 32)",
    "delattr(config, 'hidden_size')",
    "config.__delattr__('hidden_size')",
    'del config.hidden_size',
    'del config["hidden_size"]',
    '(config.hidden_size, other) = (32, 0)',
    'self.saved = config\n        self.saved.hidden_size = 32',
    'saved = [config]\n        saved[0].hidden_size = 32',
    'alias = alias.hidden_size = config',
    'config.__dict__["hidden_size"] = 32',
    'alias = config\n        for item in items:\n            del alias.hidden_size',
])
def test_all_direct_override_forms_and_untracked_config_transport_refuse(tmp_path, statement):
    result, _index, _document = _read(tmp_path, after_super='        ' + statement + '\n')
    assert result.status == 'failed'
    assert result.claim_witness is None


def test_parser_does_not_keep_provisional_shape_default_after_override_refusal(tmp_path):
    from model_unfolder.adapters.transformer import parser
    from model_unfolder.evidence import config_access
    from model_unfolder.evidence.context import ParseContext, capture_facts
    checkpoint = {'architectures': ['Wrapper'], 'vocab_size': 100}
    result, index, document = _read(tmp_path, checkpoint=checkpoint,
        after_super='        config.hidden_size = 32\n')
    assert result.status == 'failed'
    bundle = SourceBundle(source='local', files=(str(tmp_path / 'modeling_default.py'),),
        component_files={'root': (str(tmp_path / 'modeling_default.py'),)},
        supporting_files={'root': (str(tmp_path / 'configuration_default.py'),)},
        component_architectures={'root': 'Wrapper'}, architecture='Wrapper')
    defaults = {'hidden_size': 64, 'num_hidden_layers': 2, 'num_attention_heads': 4}
    context = ParseContext(bundle, class_defaults=defaults, class_defaults_by_path={(): defaults})
    context._program_index = index
    # No checkpoint shape operand: this enters the pre-existing broad shape
    # default channel. The provisional width must be cleared on failed proof.
    assert not parser._has_transformer_shape(checkpoint)
    with bound_document(DocumentBinding('root', (), document)), \
            config_access.capture_events(context.config_access), capture_facts(context.facts):
        ir = parser.parse(document.document, context=context)
    assert ir.hidden_size is None
    assert 'model.hidden_size' not in context.facts.typed_records()


@pytest.mark.parametrize('statement', [
    "vars(config)['hidden_size'] = 32",
    "for item in items:\n            vars(config)['hidden_size'] = 32",
    "del vars(config)['hidden_size']",
    "config.__dict__.__setitem__('hidden_size', 32)",
    "config.__dict__.__delitem__('hidden_size')",
    "config.__dict__.update(hidden_size=32)",
    "config.__dict__.clear()",
    "config.__dict__.pop('hidden_size')",
    "config.__dict__.popitem()",
    "config.__dict__.setdefault('hidden_size', 32)",
    "config.__dict__.__ior__({'hidden_size': 32})",
    "config.__dict__.unknown_method()",
    "namespace = config.__dict__\n        namespace.clear()",
    "dict.__setitem__(config.__dict__, 'hidden_size', 32)",
    "escape(config.__dict__)",
])
def test_direct_config_namespace_protocols_cannot_preserve_default_proof(tmp_path, statement):
    result, _index, _document = _read(tmp_path, after_super='        ' + statement + '\n')
    assert result.status == 'failed'
    assert result.claim_witness is None
