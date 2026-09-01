"""Neutral exact self-method return transport counterexamples."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.self_method_return import (
    SelfMethodReturnLane,
    SelfMethodReturnTransport,
    resolve_self_method_return_transport,
)


_SOURCE = """
class Owner:
    def helper(self, left, right):
        if left:
            left = make_a(left)
        else:
            left = make_b(left)
        return left, right

    def forward(self, x, y):
        first, second = self.helper(left=x, right=y)
        return use(first, second)
"""


def _index(tmp_path, source=_SOURCE):
    path = tmp_path / "modeling_helper.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Owner"}, architecture="Owner")
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    return index, root


def _result(tmp_path, source=_SOURCE):
    index, root = _index(tmp_path, source)
    caller = next(item.symbol for item in index.callables
                  if item.symbol.qualified_name == "Owner.forward")
    call = next(item for item in index.calls_in(caller)
                if item.callee.source_segment == "self.helper")
    return index, resolve_self_method_return_transport(
        index, root, root.graph.root.occurrence, caller, call)


def test_exact_helper_call_return_and_unpack_lanes_resolve(tmp_path):
    _index_value, result = _result(tmp_path)
    assert result.status == "resolved", result.failures
    value = result.value
    assert value.helper.symbol.qualified_name == "Owner.helper"
    assert [(item.formal.name, item.actual.source_segment)
            for item in value.arguments] == [("left", "x"), ("right", "y")]
    assert [(item.caller_target.source_segment,
             item.returned_value.source_segment)
            for item in value.lanes] == [
                ("first", "left"), ("second", "right")]


def test_unrelated_optional_slice_children_do_not_crash_expression_walk(
        tmp_path):
    source = _SOURCE.replace(
        "first, second = self.helper(left=x, right=y)",
        "window = x[:]\n        first, second = self.helper(left=x, right=y)")
    _index_value, result = _result(tmp_path, source)
    assert result.status == "resolved", result.failures


@pytest.mark.parametrize("old,new", [
    ("return left, right", "return left"),
    ("return left, right", "if right:\n            return left, right"),
    ("self.helper(left=x, right=y)", "self.helper(x, *y)"),
    ("self.helper(left=x, right=y)", "self.helper(left=x, **y)"),
])
def test_arity_guarded_return_and_starred_calls_do_not_resolve(
        tmp_path, old, new):
    _index_value, result = _result(tmp_path, _SOURCE.replace(old, new))
    assert result.status == "failed"


def test_dynamic_or_inherited_helper_is_not_selected_by_spelling(tmp_path):
    source = _SOURCE.replace(
        "class Owner:\n    def helper(self, left, right):",
        "class Base:\n    def helper(self, left, right):") \
        .replace("\n    def forward(self, x, y):", "\n\nclass Owner(Base):\n    def forward(self, x, y):")
    _index_value, result = _result(tmp_path, source)
    assert result.status == "failed"


def test_transport_closure_rejects_foreign_lane_and_helper(tmp_path):
    _index_value, result = _result(tmp_path)
    value = result.value
    with pytest.raises(ValueError):
        replace(value, lanes=(SelfMethodReturnLane(
            0, value.lanes[0].caller_target,
            value.lanes[1].returned_value), *value.lanes[1:]))
    with pytest.raises(ValueError):
        SelfMethodReturnTransport(
            value.owner_occurrence, value.caller, value.caller, value.call,
            value.caller_definition, value.returned,
            value.arguments, value.lanes, value.spans)


def test_unsupported_execution_is_strict_unless_a_consumer_explicitly_defers(
        tmp_path):
    source = _SOURCE.replace(
        "    def helper(self, left, right):\n",
        "    def helper(self, left, right):\n"
        "        try:\n"
        "            left = left\n"
        "        finally:\n"
        "            left = left\n")
    index, strict = _result(tmp_path, source)
    assert strict.status == "failed"
    root_bundle = SourceBundle(
        source="local", files=tuple(item.source_id.canonical_path
                                    for item in index.source_nodes),
        component_files={"root": tuple(
            item.source_id.canonical_path for item in index.source_nodes)},
        component_architectures={"root": "Owner"}, architecture="Owner")
    root = resolve_component_root(index, root_bundle, "root")
    caller = next(item.symbol for item in index.callables
                  if item.symbol.qualified_name == "Owner.forward")
    call = next(item for item in index.calls_in(caller)
                if item.callee.source_segment == "self.helper")
    deferred = resolve_self_method_return_transport(
        index, root, root.occurrence, caller, call,
        defer_unsupported_to_consumer=True)
    assert deferred.status == "resolved"
    with pytest.raises(TypeError):
        resolve_self_method_return_transport(
            index, root, root.occurrence, caller, call,
            defer_unsupported_to_consumer=1)
