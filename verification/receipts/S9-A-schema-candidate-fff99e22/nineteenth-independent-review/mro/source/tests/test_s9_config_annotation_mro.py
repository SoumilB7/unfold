"""Exact first-direct-base annotation ordering, without general MRO inference."""
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.framework_config import framework_config_alias, framework_config_class
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _read(tmp_path, *, bases='First, GenerationMixin', root_annotation='',
          first_annotation='config: ExactConfig', first_base='PreTrainedModel',
          root_keywords='', first_keywords='', rebinding=''):
    source = tmp_path / 'modeling_order.py'
    support = tmp_path / 'configuration_order.py'
    source.write_text(f'''from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from .configuration_order import ExactConfig, OtherConfig
class Ancestor(PreTrainedModel):
    config: OtherConfig
class First({first_base}{first_keywords}):
    {first_annotation or 'pass'}
class OtherFirst(PreTrainedModel):
    config: OtherConfig
{rebinding}
class Wrapper({bases}{root_keywords}):
    {root_annotation or 'pass'}
    def __init__(self, config):
        super().__init__(config)
''')
    support.write_text('class ExactConfig:\n    hidden_size: int = 64\n'
                       'class OtherConfig:\n    hidden_size: int = 32\n')
    bundle = SourceBundle(source='local', files=(str(source),),
        component_files={'root': (str(source),)}, supporting_files={'root': (str(support),)},
        component_architectures={'root': 'Wrapper'}, architecture='Wrapper')
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, 'root')
    if root.status != 'resolved':
        return root
    alias = framework_config_alias(index, root, root.graph.root.occurrence)
    if alias.status != 'resolved':
        return alias
    return framework_config_class(index, alias.value)


@pytest.mark.parametrize('base, expected', [('First', 'ExactConfig'), ('OtherFirst', 'OtherConfig')])
def test_first_direct_local_base_own_annotation_wins(tmp_path, base, expected):
    result = _read(tmp_path, bases=base + ', GenerationMixin')
    assert result.status == 'resolved', result.failures
    assert result.value.config_class.qualified_name == expected
    assert result.value.annotation_owner.qualified_name == base
    assert tuple(item.qualified_name for item in result.value.inheritance_symbols) == ('Wrapper', base)


def test_root_own_annotation_precedes_first_base(tmp_path):
    result = _read(tmp_path, root_annotation='config: OtherConfig')
    assert result.status == 'resolved', result.failures
    assert result.value.annotation_owner.qualified_name == 'Wrapper'
    assert result.value.config_class.qualified_name == 'OtherConfig'


@pytest.mark.parametrize('kwargs', [
    {'bases': 'GenerationMixin, First'},
    {'bases': 'UnknownFirst, GenerationMixin'},
    {'first_annotation': ''},
    {'first_annotation': '', 'first_base': 'Ancestor'},
    {'root_keywords': ', metaclass=CustomMeta'},
    {'first_keywords': ', metaclass=CustomMeta'},
    {'rebinding': 'First = replacement'},
    {'first_annotation': 'if flag:\n        config: ExactConfig'},
])
def test_fork_does_not_guess_unknown_first_base_ancestor_or_metaclass_order(tmp_path, kwargs):
    result = _read(tmp_path, **kwargs)
    assert result.status != 'resolved'


def test_existing_single_base_annotation_chain_is_preserved(tmp_path):
    result = _read(tmp_path, bases='First', first_base='Ancestor', first_annotation='')
    assert result.status == 'resolved', result.failures
    assert result.value.annotation_owner.qualified_name == 'Ancestor'
    assert result.value.config_class.qualified_name == 'OtherConfig'


def test_actual_installed_llama_sparse_width_reaches_first_base_own_annotation():
    from model_unfolder.evidence.class_default_value import model_hidden_size_class_default
    from model_unfolder.evidence.config_access import bound_document
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.document import DocumentBinding
    from model_unfolder.parser import _coerce_prepared
    from test_s9_class_default_value import _fact

    config = json.loads((Path(__file__).parent / 'sable_test_corpus' / 'llama-7b.json').read_text())['config']
    expected = config.pop('hidden_size')
    prepared = _coerce_prepared(config)
    assert prepared.class_overlay['hidden_size'] == expected
    context = ParseContext.build(config)
    index = context.program_index()
    with bound_document(DocumentBinding('root', (), prepared)):
        result = model_hidden_size_class_default(index, context.source_bundle, ())
    assert result.status == 'resolved', result.failures
    fact = _fact(result, index, prepared)
    assert (fact.value, fact.status, fact.claim_kind) == (expected, 'class_default', 'value')
    annotation = result.value.config_class
    owner = index.class_by_symbol(annotation.alias.owner_symbol)
    assert len(owner.bases) > 1
    assert annotation.annotation_owner.qualified_name == owner.bases[0].name
    assert annotation.annotation_assignment in index.class_by_symbol(annotation.annotation_owner).body_assigns
    assert fact.claim_evidence is not None
