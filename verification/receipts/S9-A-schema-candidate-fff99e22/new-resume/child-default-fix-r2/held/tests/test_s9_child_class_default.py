"""A selected child's own literal default requires the complete caller address."""
from copy import deepcopy
from dataclasses import replace

import pytest

from model_unfolder.evidence.class_default_value import model_hidden_size_class_default
from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument, checkpoint_provenance
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from test_support.s9_fixtures.s9_class_default_value import _fact


def _child(tmp_path, *, before='', after='', child_after='', factory=True,
           checkpoint=None, overlay=None, selected=('text_config',), child_component=True):
    model = tmp_path / 'modeling_child.py'
    support = tmp_path / 'configuration_child.py'
    call = 'Child._from_config' if factory else 'Child'
    model.write_text(f'''from transformers.modeling_utils import PreTrainedModel
from .configuration_child import ExactConfig
class Child(PreTrainedModel):
    def __init__(self, config: ExactConfig):
        super().__init__(config)
{child_after}class Wrapper(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
{before}        self.slot = {call}(config.text_config)
{after}''')
    support.write_text('class ExactConfig:\n    hidden_size: int = 64\n')
    components = {'root': (str(model),)}
    if child_component:
        components['text_config'] = (str(model),)
    bundle = SourceBundle(source='local', files=(str(model),), component_files=components,
        supporting_files={key: (str(support),) for key in components},
        component_architectures={'root': 'Wrapper', 'text_config': 'Child'}, architecture='Wrapper')
    index = build_program_index(bundle)
    checkpoint = {'hidden_size': 999, 'text_config': {}} if checkpoint is None else checkpoint
    document = PreparedDocument(deepcopy(checkpoint), deepcopy(checkpoint),
        {'text_config.hidden_size': 64} if overlay is None else overlay,
        provenance=checkpoint_provenance(checkpoint))
    with bound_document(DocumentBinding('root', (), document)):
        result = model_hidden_size_class_default(index, bundle, selected)
    return result, index, document, bundle


@pytest.mark.parametrize('factory', [False, True])
def test_exact_constructed_child_literal_is_qualified_without_root_borrowing(tmp_path, factory):
    result, index, document, _ = _child(tmp_path, factory=factory)
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, document)
    assert (fact.value, fact.status, fact.claim_kind) == (64, 'class_default', 'value')
    assert fact.config_paths == ()
    assert result.value.construction.config_path == ('text_config',)
    assert result.value.alias.owner_symbol.source.component_key == 'text_config'
    assert result.value.config_class.config_class.qualified_name == 'ExactConfig'
    assert set(result.value.construction.spans) <= set(result.value.spans)
    assert any(span.source.component_key == 'root' for span in result.value.spans)
    assert document.document['hidden_size'] == 999
    # Receipt replay must re-establish the actual caller/child path.
    with pytest.raises(ValueError):
        replace(result.claim_witness, config_path=('other_config',))._checked()
    changed = deepcopy(document.document)
    changed['text_config']['hidden_size'] = 32
    document.document['text_config'] = changed['text_config']
    with pytest.raises(ValueError):
        result.claim_witness._checked()


@pytest.mark.parametrize('statement', [
    'config.text_config.hidden_size = 32',
    'config.text_config = replacement',
    'del config.text_config.hidden_size',
    'setattr(config.text_config, "hidden_size", 32)',
    'config.text_config.__dict__.update(hidden_size=32)',
    'escape(config.text_config)',
    'escape(config)',
    'self.leaked = config.text_config',
    'self.leaked = config.text_config if enabled else None',
    'return config.text_config',
    'alias = config.text_config\n        alias.hidden_size = 32',
    'alias = config\n        alias = config.text_config\n        alias.hidden_size = 32',
    'escape(config.text_config).field = 32',
])
@pytest.mark.parametrize('where', ['before', 'after'])
def test_parent_mutation_replacement_or_escape_cannot_keep_the_child_default(tmp_path, statement, where):
    result, _, _, _ = _child(tmp_path, **{where: '        ' + statement + '\n'})
    assert result.status == 'failed'
    assert result.claim_witness is None


@pytest.mark.parametrize('kwargs', [
    {'checkpoint': {'text_config': {'hidden_size': None}}},
    {'checkpoint': {'text_config': {'hidden_size': 64}}},
    {'checkpoint': {'text_config': {'d_model': 64}}},
    {'checkpoint': {'text_config': {'d_model': 64, 'n_embd': 32}}},
    {'checkpoint': {'text_config': None}},
    {'checkpoint': {}},
    {'overlay': {'hidden_size': 64}},
    {'overlay': {'text_config.hidden_size': 32}},
    {'selected': ('other_config',)},
    {'child_component': False},
    {'child_after': '        config.hidden_size = 32\n'},
])
def test_absent_address_or_foreign_value_never_falls_back_to_root_default(tmp_path, kwargs):
    result, _, _, _ = _child(tmp_path, **kwargs)
    assert result.status == 'failed'
    assert result.claim_witness is None


def test_unrelated_root_scalar_write_does_not_replace_child_default(tmp_path):
    result, index, document, _ = _child(tmp_path, before='        config.hidden_size = 32\n')
    assert result.status == 'resolved', result.failures
    assert _fact(result, index, document).value == 64


def test_constructor_relay_allowance_uses_each_exact_whole_call(tmp_path, monkeypatch):
    from model_unfolder.evidence.class_default_value import _constructor_preserves_operand
    from model_unfolder.evidence.program_index import SymbolId

    _, index, _, _ = _child(tmp_path, factory=False,
        after='        self.other = Child(config.text_config)\n')
    owner = next(record.symbol for record in index.classes
                 if record.symbol.source.component_key == 'root'
                 and record.symbol.qualified_name == 'Wrapper')
    constructor = SymbolId(owner.source, 'Wrapper.__init__')
    calls = tuple(index.calls_in(constructor))
    twins = tuple(call for call in calls if call.callee.name == 'Child')
    assert len(twins) == 2
    assert twins[0].span != twins[1].span
    assert all(call.span != call.callee.span for call in twins)
    allowed = frozenset(call.span for call in calls)

    def kept(addresses):
        return _constructor_preserves_operand(index, owner, 'config', None,
            ('text_config', 'hidden_size'), allowed_config_calls=addresses)

    assert kept(allowed)
    # An identical callee spelling at another source address cannot inherit
    # the first construction's allowance. A shorter token is not a call ID.
    assert not kept(allowed - {twins[1].span})
    assert not kept(frozenset(call.callee.span for call in calls))
    assert not kept(frozenset({None}))
    original_calls_in = type(index).calls_in

    def missing_call_identity(active_index, symbol):
        return tuple(replace(call, span=None) if symbol == constructor and call == twins[0]
                     else call for call in original_calls_in(active_index, symbol))

    monkeypatch.setattr(type(index), 'calls_in', missing_call_identity)
    # Even an allowlist containing None cannot admit an unstamped call.
    assert not kept(allowed | {None})


def test_unknown_wrapper_cannot_reintroduce_locally_hydrated_bias_as_bound_default():
    from model_unfolder import unfold
    nested = {'model_type': 'composite_xyz', 'architectures': ['CompositeForConditionalGeneration'],
              'thinker_config': {'model_type': 'inner_xyz', 'audio_token_index': 151646,
                'vision_config': {'model_type': 'inner_vit', 'depth': 4, 'hidden_size': 128,
                                  'num_heads': 4, 'patch_size': 14, 'spatial_merge_size': 2},
                'audio_config': {'model_type': 'inner_audio', 'd_model': 128,
                                 'encoder_layers': 2, 'encoder_attention_heads': 4, 'num_mel_bins': 128},
                'text_config': {'model_type': 'llama', 'hidden_size': 256, 'num_hidden_layers': 2,
                                'num_attention_heads': 4, 'intermediate_size': 512, 'vocab_size': 1000}}}
    ir = unfold(nested).to_ir()
    assert ir['hidden_size'] == 256
    assert ir['layers'] and all(layer['attention']['bias'] is None for layer in ir['layers'])
    row = ir['extras']['fact_provenance']['decoder.attention.bias']
    assert row['status'] in ('ambiguous', 'oracle_missing', 'unknown')
    assert row['unknown_reason']['reason_class'] == 'investigation_missing'
    # A failed enclosing class address does not invalidate a real checkpoint
    # value; the exact same source path becomes a lawful explicit operand.
    nested['thinker_config']['text_config']['attention_bias'] = False
    declared = unfold(nested).to_ir()
    assert all(layer['attention']['bias'] is False for layer in declared['layers'])
    assert declared['extras']['fact_provenance']['decoder.attention.bias']['status'] == 'code_and_config'
