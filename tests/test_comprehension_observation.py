"""U3-A3 — exact, neutral comprehension binding observations."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import (
    ComprehensionClause,
    ExprNode,
    SourceId,
    SourceSpan,
    SymbolId,
)


def _index(tmp_path, source: str):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)})
    return pi.build_program_index(bundle)


def _forward(index, owner="Model"):
    return next(
        item.symbol for item in index.callables
        if item.symbol.qualified_name == f"{owner}.forward")


def test_all_comprehension_forms_preserve_exact_targets_iterables_and_outputs(
        tmp_path):
    index = _index(tmp_path, """
        class Model:
            def forward(self, xs, ys, keep):
                a = [(i, j) for i in xs if keep(i) for j in ys]
                b = {i for i in xs}
                c = {i: keep(i) for i in xs}
                d = (i for i in xs)
                return a, b, c, d
    """)
    observations = index.comprehensions_in(_forward(index))

    assert [item.expression_kind for item in observations] == [
        "list", "set", "dict", "generator"]
    first = observations[0]
    assert [clause.target.name for clause in first.clauses] == ["i", "j"]
    assert [clause.iterable.name for clause in first.clauses] == ["xs", "ys"]
    assert first.clauses[0].filters[0].kind == "call"
    assert first.clauses[0].filters[0].children[0].name == "keep"
    assert first.outputs[0].kind == "tuple"
    assert len(observations[2].outputs) == 2
    assert observations[2].outputs[0].name == "i"
    assert observations[2].outputs[1].kind == "call"
    assert all(item.span.source == first.span.source for item in observations)


def test_comprehension_retains_enclosing_guard_without_calling_it_execution(
        tmp_path):
    index = _index(tmp_path, """
        class Model:
            def forward(self, xs, enabled):
                if enabled:
                    value = [x for x in xs]
                return value
    """)
    observation = index.comprehensions_in(_forward(index))[0]
    assert len(observation.guard) == 1
    assert observation.guard[0].kind == "if"
    assert not hasattr(observation, "execution_order")
    assert not hasattr(observation, "happens_before")


def test_nested_comprehensions_are_separate_exact_observations(tmp_path):
    index = _index(tmp_path, """
        class Model:
            def forward(self, rows):
                return [[x for x in row] for row in rows]
    """)
    observations = index.comprehensions_in(_forward(index))
    assert len(observations) == 2
    assert {
        (item.clauses[0].target.name, item.clauses[0].iterable.name)
        for item in observations
    } == {("row", "rows"), ("x", "row")}
    assert observations[0].span != observations[1].span


def test_async_clause_is_observed_not_interpreted(tmp_path):
    index = _index(tmp_path, """
        class Model:
            async def forward(self, stream):
                return [item async for item in stream]
    """)
    observation = index.comprehensions_in(_forward(index))[0]
    assert observation.clauses[0].async_flag is True
    assert observation.clauses[0].target.name == "item"
    assert observation.clauses[0].iterable.name == "stream"


def test_same_spelling_in_sibling_callable_cannot_enter_owner_query(tmp_path):
    index = _index(tmp_path, """
        class Model:
            def forward(self, bank):
                return [item for item in bank]

        class Sibling:
            def forward(self, bank):
                return [item for item in bank if item]
    """)
    model = index.comprehensions_in(_forward(index, "Model"))
    sibling = index.comprehensions_in(_forward(index, "Sibling"))
    assert len(model) == len(sibling) == 1
    assert model[0].owner.qualified_name == "Model"
    assert sibling[0].owner.qualified_name == "Sibling"
    assert model[0].clauses[0].filters == ()
    assert len(sibling[0].clauses[0].filters) == 1


def test_observation_closure_rejects_incomplete_and_cross_source_payloads(
        tmp_path):
    index = _index(tmp_path, """
        class Model:
            def forward(self, xs):
                return [x for x in xs]
    """)
    observation = index.comprehensions_in(_forward(index))[0]

    with pytest.raises(ValueError):
        replace(observation, expression_kind="unknown")
    with pytest.raises(ValueError):
        replace(observation, outputs=())
    with pytest.raises(ValueError):
        replace(observation, clauses=())
    with pytest.raises(ValueError):
        replace(observation, expression_kind="dict")

    clause = observation.clauses[0]
    with pytest.raises(TypeError):
        replace(clause, filters=("not-an-expression",))
    with pytest.raises(TypeError):
        replace(clause, async_flag=1)

    other_source = SourceId(
        "/tmp/foreign.py", "foreign", component_key="root")
    foreign = ExprNode(
        "name", name="xs", span=SourceSpan(other_source, 1, 0, 1, 2))
    with pytest.raises(ValueError):
        ComprehensionClause(clause.target, foreign)
    with pytest.raises(ValueError):
        replace(
            observation,
            enclosing_callable=SymbolId(
                other_source, observation.enclosing_callable.qualified_name))


def test_program_index_round_trips_the_observation_tuple(tmp_path):
    index = _index(tmp_path, """
        class Model:
            def forward(self, xs):
                return sum(x for x in xs)
    """)
    forward = _forward(index)
    assert index.comprehensions == index.comprehensions_in(forward)
    observation = index.comprehensions[0]
    aggregate = next(
        call for call in index.calls_in(forward)
        if call.callee.kind == "name" and call.callee.name == "sum")
    assert aggregate.args[0].span == observation.span

