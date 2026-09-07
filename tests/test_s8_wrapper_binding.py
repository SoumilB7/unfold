"""Actual wrapper delegation needs clause and receiver-binding evidence."""
import importlib.util

import pytest

from physics.attribute_bindings import (
    AttributeBindingWitness, _function_witness, capture_attribute_lookup_types,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_wrapper_binding import read_wrapper_binding


def _read(tmp_path, body):
    source = ('class Cell:\n    def forward(owner, value):\n        return owner.child(value)\n\n'
              'def wrap(captured):\n    def wrapper(owner, value):\n' +
              ''.join('        ' + line + '\n' for line in body.splitlines()) +
              '    return wrapper\n')
    path = tmp_path / 'wrapper_fixture.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('s8_wrapper_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = build_program_index(SourceBundle(source='path', component_files={'root': (str(path),)}))
    function, reason = _function_witness(module.wrap(module.Cell.forward), capture_attribute_lookup_types())
    assert function is not None, reason
    witness = AttributeBindingWitness('', 'forward', 'plain_bound_method',
                                      function=function, lookup_kind='object_getattribute')
    target = next(row.symbol for row in index.callables if row.symbol.qualified_name == 'Cell.forward')
    return read_wrapper_binding(index, witness, target)


def test_finally_delegation_retains_each_parent_effect_condition(tmp_path):
    result = _read(tmp_path, '''if enabled:
    before(owner)
try:
    result = captured(owner, value)
    return result
finally:
    if enabled:
        after(owner)''')
    assert result.kind == 'conditional_wrapper'
    assert len(result.conditions) == 2
    assert all(row['branch'] is False for row in result.conditions)
    assert len({row['source'] for row in result.conditions}) == 2


@pytest.mark.parametrize('body', [
    '''if enabled:
    before(owner)
    return captured(owner, value)''',
    '''try:
    unknown()
except Exception as owner:
    return captured(owner, value)''',
    '''try:
    return captured(owner, value)
finally:
    return replacement''',
    '''try:
    return captured(owner, value)
finally:
    owner = replacement''',
    '''try:
    return captured(owner, value)
finally:
    after(owner)''',
    '''with manager() as owner:
    return captured(owner, value)''',
    '''for owner in others:
    pass
return captured(owner, value)''',
    '''del owner.child
return captured(owner, value)''',
])
def test_unclosed_clause_or_owner_effect_refuses_delegation(tmp_path, body):
    assert _read(tmp_path, body).kind == 'unresolved'
