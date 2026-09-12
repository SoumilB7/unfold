"""Call-local work controls; actual readers still decide source obligations."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import textwrap

from model_unfolder.evidence import unet_selected_spatial as spatial
from model_unfolder.evidence.expression_eval import EvaluatedExpression
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_call_binding import demanded_root_lookups
from model_unfolder.evidence.unet_lookup_closure import read_lookup_closure
from physics.attribute_bindings import AttributeBindingWitness, PythonFunctionWitness


def _index(tmp_path, source, other_source=None):
    paths = []
    for name, text in (("selected.py", source), ("unrelated.py", other_source)):
        if text is not None:
            path = tmp_path / name
            path.write_text(textwrap.dedent(text))
            paths.append(str(path))
    return build_program_index(SourceBundle(
        source="local", files=tuple(paths),
        component_files={"root": tuple(paths)}))


class _AddressQueriesOnly:
    """Forbid a reader's global census without changing the real index/maps."""

    def __init__(self, index):
        self.index = index
        self.visits = []

    def __getattr__(self, name):
        if name == "attribute_accesses":
            raise AssertionError("reader rescanned the entire attribute census")
        return getattr(self.index, name)

    def attribute_accesses_in(self, symbol):
        self.visits.append(symbol)
        return self.index.attribute_accesses_in(symbol)


def test_root_lookup_demand_keeps_source_identity_and_nested_storage(tmp_path):
    index = _index(tmp_path, """
        class Cell:
            def forward(self, value):
                value = self.helper(value)
                return self.child(value)
            def helper(self, value):
                if self.config.enabled:
                    return self.other(value)
                return value
            def unreachable(self):
                return self.unreachable_member
    """, """
        class Cell:
            def forward(self, value):
                return self.foreign(value)
            def helper(self, value):
                return self.foreign_helper(value)
    """)
    root = next(row.symbol for row in index.classes
                if Path(row.symbol.source.canonical_path).name == "selected.py")
    queries = _AddressQueriesOnly(index)
    requests = demanded_root_lookups(queries, root)
    assert [(row.attribute, row.storage_attributes) for row in requests] == [
        ("child", ()), ("config", ("enabled",)), ("forward", ()),
        ("helper", ()), ("other", ())]
    assert [row.qualified_name for row in queries.visits] == [
        "Cell.forward", "Cell.helper"]
    assert all(row.source == root.source for row in queries.visits)


def test_property_lookup_keeps_own_write_and_new_index_refusal(tmp_path):
    index = _index(tmp_path, """
        class Cell:
            @property
            def config(self):
                return self._storage
    """, """
        class Cell:
            @property
            def config(self):
                self._storage = None
                return self._storage
    """)
    method = next(row for row in index.callables
                  if Path(row.symbol.source.canonical_path).name == "selected.py")
    function = PythonFunctionWitness(
        "selected", method.symbol.qualified_name, "selected.py",
        method.symbol.source.content_fingerprint,
        min(method.span.line, *(row.span.line for row in method.decorators)), "a" * 64)
    witness = AttributeBindingWitness(
        "", "config", "property_getter", lookup_kind="object_getattribute",
        function=function, instance_storage_names=("_storage",))
    bindings = SimpleNamespace(_modules={})
    queries = _AddressQueriesOnly(index)
    result = read_lookup_closure(bindings, witness, index=queries)
    assert result.kind == "property_storage"
    assert result.storage_name == "_storage" and result.method == method
    assert queries.visits == [method.symbol]

    own_read = next(row for row in index.attribute_accesses
                    if row.enclosing_callable == method.symbol)
    changed = replace(index, attribute_accesses=(
        *index.attribute_accesses, replace(own_read, mode="write")))
    refused = read_lookup_closure(bindings, witness, index=_AddressQueriesOnly(changed))
    assert not refused.established
    assert refused.reason == "property_getter_is_not_direct_instance_storage_read"
    assert read_lookup_closure(bindings, witness, index=queries) == result


def test_spatial_environment_built_once_per_invocation_with_fresh_guards(
        tmp_path, monkeypatch):
    index = _index(tmp_path, """
        class Cell:
            def forward(self, value):
                if self.enabled:
                    value = self.first(value)
                    value = self.second(value)
                return value
    """)
    mechanism = SimpleNamespace(occurrence_id=SimpleNamespace(
        symbol=index.classes[0].symbol))
    operands = object()
    environments, routes = [], []
    selected_values = iter((True, False, None, True))

    def environment(actual_index, actual_operands):
        assert actual_index is index and actual_operands is operands
        value = next(selected_values)
        env = {} if value is None else {"self.enabled": EvaluatedExpression(value)}
        environments.append(env)
        return env

    def route(actual_index, actual_mechanism, actual_operands, call):
        assert actual_index is index and actual_mechanism is mechanism
        assert actual_operands is operands
        routes.append(call.callee.name)
        return None, ("exact route remains unresolved", (call.span,))

    monkeypatch.setattr(spatial, "_child_env", environment)
    monkeypatch.setattr(spatial, "_invoked_field_constructor_route", route)
    results = []
    for number in range(4):
        results.append(spatial._selected_external_effect(index, mechanism, operands))
        assert len(environments) == number + 1
    assert routes == ["first", "second", "first", "second"]
    assert results[0] == results[3] and len(results[0][1]) == 2
    assert results[1] == results[2] == ((), ())
    assert environments == [
        {"self.enabled": EvaluatedExpression(True)},
        {"self.enabled": EvaluatedExpression(False)}, {},
        {"self.enabled": EvaluatedExpression(True)}]
