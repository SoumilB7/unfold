"""Exact immutable value hashes; no dataclass/identity framework replacement."""
from copy import copy, deepcopy
from dataclasses import asdict, fields, replace
import pickle

import pytest

from model_unfolder.evidence.program_index import ExprNode, SourceId, SourceSpan


def _values():
    source = SourceId('/source.py', 'content-fingerprint', 'root')
    span = SourceSpan(source, 3, 2, 4, 10)
    leaf = ExprNode('name', name='x', span=span, source_segment='x')
    expression = ExprNode('call', children=(leaf,), keyword_children=(('arg', leaf),), span=span)
    return source, span, expression


def _field_hash(value):
    return hash(tuple(getattr(value, field.name) for field in fields(value)
                      if (field.compare if field.hash is None else field.hash)))


@pytest.mark.parametrize('position', range(3))
def test_hash_is_exact_declared_field_tuple_and_not_a_dataclass_field(position):
    value = _values()[position]
    before = asdict(value)
    assert hash(value) == _field_hash(value)
    assert '_structural_hash' in value.__dict__
    assert '_structural_hash' not in {field.name for field in fields(value)}
    assert asdict(value) == before
    for _ in range(100):
        assert hash(value) == _field_hash(value)


@pytest.mark.parametrize('position', range(3))
def test_pickle_copy_and_replace_omit_derived_hash_and_preserve_exact_state(position):
    value = _values()[position]
    original_pickle = pickle.dumps(value)
    original_dict = asdict(value)
    expected_hash = hash(value)
    assert pickle.dumps(value) == original_pickle
    for restored in (copy(value), deepcopy(value), pickle.loads(pickle.dumps(value)), replace(value)):
        assert '_structural_hash' not in restored.__dict__
        assert restored == value and asdict(restored) == original_dict
        assert hash(restored) == expected_hash


@pytest.mark.parametrize('position', range(3))
def test_replace_changed_field_resets_hash_and_does_not_alias_old_cache(position):
    value = _values()[position]
    hash(value)
    updates = ({'content_fingerprint': 'changed'}, {'line': 20}, {'name': 'changed'})
    changed = replace(value, **updates[position])
    assert changed != value and '_structural_hash' not in changed.__dict__
    assert hash(changed) == _field_hash(changed)


class _Collision(str):
    def __hash__(self):
        return 7


def _collisions():
    left = SourceId(_Collision('left'), 'same', 'root')
    right = SourceId(_Collision('right'), 'same', 'root')
    return [(left, right), (SourceSpan(left, 1), SourceSpan(right, 1)),
            (ExprNode('name', name=_Collision('left')), ExprNode('name', name=_Collision('right')))]


@pytest.mark.parametrize('position', range(3))
def test_equal_hashes_never_replace_complete_value_equality(position):
    left, right = _collisions()[position]
    assert left != right and hash(left) == hash(right)
    values = {left: 'left', right: 'right'}
    assert len(values) == 2 and values[left] == 'left' and values[right] == 'right'
    assert replace(left) == left and values[replace(left)] == 'left'


@pytest.mark.parametrize('value', [SourceId([], 'hash'), SourceSpan([], 1),
                                    ExprNode('constant', const_value=[])])
def test_malformed_unhashable_fields_raise_on_every_attempt_without_cache(value):
    for _ in range(2):
        with pytest.raises(TypeError):
            hash(value)
        assert '_structural_hash' not in value.__dict__


def test_each_value_computes_field_hash_once_including_shared_nested_children(monkeypatch):
    counts = {cls: [] for cls in (SourceId, SourceSpan, ExprNode)}
    for cls, calls in counts.items():
        descriptor = cls.__dict__['_structural_hash']
        original = descriptor.func
        def counted(value, _original=original, _calls=calls):
            _calls.append(value)
            return _original(value)
        monkeypatch.setattr(descriptor, 'func', counted)
    source, span, expression = _values()
    expected = _field_hash(expression)
    for _ in range(100):
        assert hash(expression) == expected
        hash(source)
        hash(span)
    assert len(counts[SourceId]) == 1
    assert len(counts[SourceSpan]) == 1
    assert len(counts[ExprNode]) == 2  # one parent + one shared leaf
    assert len({id(value) for value in counts[ExprNode]}) == 2


def test_pickle_does_not_transport_even_a_poisoned_process_local_cache():
    # A distinct process uses a different hash salt. Force a foreign cache
    # sentinel here; serialization must discard it, never bless its integer.
    for position in range(3):
        value = _values()[position]
        expected = _field_hash(value)
        object.__setattr__(value, '_structural_hash', expected + 12345)
        restored = pickle.loads(pickle.dumps(value))
        assert '_structural_hash' not in restored.__dict__
        assert hash(restored) == _field_hash(restored) == expected
