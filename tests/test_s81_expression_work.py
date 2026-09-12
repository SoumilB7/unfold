"""One walker run normalizes each exact AST object once, retaining all records."""
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import pickle

import pytest

from model_unfolder.everchanging import (
    load_constructor_classmethods, load_program_index_vocab,
)
from model_unfolder.evidence import program_index as program


SOURCE = '''
from somewhere import Leaf
class Model:
    def __init__(self, config):
        self.items = [Leaf(config.width) for i in range(config.depth) if i > 0]
        self.options = {"width": config.width, **config.extra}
    def forward(self, value):
        café = value[:, :2] if value is not None else (1, 2)
        result = self.items[0](café, scale=2 + 3)
        if result and not value:
            return {i: x for i, x in enumerate(result)}
        else:
            return (lambda x: f"item:{x}")(result)
'''


def walker(source=SOURCE, *, component="root", tree=None, vocab=None):
    sid = program.SourceId('/same.py', program.content_fingerprint(source), component)
    return program._SourceWalker(
        sid, source, ast.parse(source) if tree is None else tree,
        config_vocab=load_program_index_vocab() if vocab is None else vocab,
        factory_names=frozenset(load_constructor_classmethods()))


def records(value):
    return tuple((name, tuple(getattr(value, attr)))
                 for name, attr in program._WALKER_RECORDS)


def serialized(value):
    return tuple((name, tuple(asdict(item) for item in rows))
                 for name, rows in records(value))


def test_all_records_equal_uncached_walk_ast_unchanged_and_once_per_object(monkeypatch):
    original = program._SourceWalker._build_expr
    counts = Counter()
    retained = {}

    def counted(self, node):
        if node is not None:
            counts[id(node)] += 1
            retained[id(node)] = node
        return original(self, node)

    monkeypatch.setattr(program._SourceWalker, '_build_expr', counted)
    actual = walker()
    before = ast.dump(actual.tree, include_attributes=True)
    actual.run()
    assert counts and set(counts.values()) == {1}
    assert ast.dump(actual.tree, include_attributes=True) == before
    assert actual._expr_memo is None
    once = sum(counts.values())
    counts.clear()
    monkeypatch.setattr(program._SourceWalker, '_expr', counted)
    reference = walker().run()
    assert sum(counts.values()) > once
    assert records(actual) == records(reference)
    assert serialized(actual) == serialized(reference)
    assert deepcopy(records(actual)) == records(reference)
    assert pickle.loads(pickle.dumps(records(actual))) == records(reference)


def test_new_walkers_keep_source_component_vocabulary_and_current_ast(monkeypatch):
    first = walker().run()
    variants = [
        walker(SOURCE.replace('config.width', 'config.changed')),
        walker(component='other'),
    ]
    vocab = dict(load_program_index_vocab())
    vocab['config_roots'] = frozenset({'different'})
    variants.append(walker(vocab=vocab))
    for value in variants:
        value.run()
        assert records(value) != records(first)
        assert value._expr_memo is None
    # Direct calls are deliberately uncached because callers may mutate an AST.
    direct = walker()
    node = ast.parse('old', mode='eval').body
    assert direct._expr(node).name == 'old'
    node.id = 'new'
    assert direct._expr(node).name == 'new'
    original = program._SourceWalker._expr
    with monkeypatch.context() as patch:
        patch.setattr(program._SourceWalker, '_expr', program._SourceWalker._build_expr)
        for value in variants:
            reference = walker(value.text, component=value.sid.component_key,
                               vocab=vocab if value is variants[-1] else None).run()
            assert serialized(value) == serialized(reference)
    assert program._SourceWalker._expr is original


def test_construction_exception_is_identical_and_run_cache_always_released(monkeypatch):
    failure = RuntimeError('normalization failed')
    original = program._SourceWalker._build_expr
    observed = []

    def failing(self, node):
        if isinstance(node, ast.Call):
            observed.append(self._expr_memo)
            raise failure
        return original(self, node)

    monkeypatch.setattr(program._SourceWalker, '_build_expr', failing)
    value = walker()
    with pytest.raises(RuntimeError) as caught:
        value.run()
    assert caught.value is failure
    assert observed and value._expr_memo is None
