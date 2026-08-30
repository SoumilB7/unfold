"""Occurrence-exact constructor-argument propagation controls."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_owner_graph,
)
from model_unfolder.evidence.expression_eval import (
    ConfigExpressionEvaluator,
    EvaluatedExpression,
    constructor_argument_env,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _graph(tmp_path, source_text):
    source = tmp_path / "modeling_constructor_env.py"
    source.write_text(textwrap.dedent(source_text), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(source),), architecture="Root",
        component_files={"root": (str(source),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    return index, root.graph


def test_constructor_formal_is_resolved_through_two_exact_occurrence_hops(
        tmp_path):
    index, graph = _graph(tmp_path, """
        class Block:
            def __init__(self, config, gated):
                self.gated = gated

        class Stage:
            def __init__(self, config, gated):
                self.block = Block(config, gated)

        class Root:
            def __init__(self, config):
                self.local = Stage(config, False)
                self.global_stage = Stage(config, True)
    """)
    blocks = tuple(
        node for node in graph.walk()
        if node.symbol.qualified_name == "Block")
    assert len(blocks) == 2
    values = {
        constructor_argument_env(
            index, graph, node.occurrence, {}).get("gated").value
        for node in blocks
    }
    assert values == {False, True}


def test_same_class_occurrences_do_not_launder_constructor_arguments(
        tmp_path):
    index, graph = _graph(tmp_path, """
        from torch.nn import ModuleList
        class Block:
            def __init__(self, config, switch): pass
        class Stage:
            def __init__(self, config, switch):
                self.blocks = ModuleList(
                    [Block(config, switch) for _ in range(config.depth)])
        class Root:
            def __init__(self, config):
                self.a = Stage(config, config.first)
                self.b = Stage(config, config.second)
    """)
    blocks = tuple(
        node for node in graph.walk()
        if node.symbol.qualified_name == "Block")
    assert len(blocks) == 2
    values = sorted(
        constructor_argument_env(
            index, graph, node.occurrence,
            {"first": 3, "second": 9})["switch"].value
        for node in blocks)
    assert values == [3, 9]


def test_missing_runtime_actual_falls_through_to_exact_config_binding(tmp_path):
    """A symbolic occurrence may lack an evaluable runtime argument.

    That absence must not hide an independently exact OwnerGraph config-path
    binding for the same formal.  Conversely, an explicit runtime value still
    wins over the document binding for its exact occurrence.
    """
    index, base_graph = _graph(tmp_path, """
        class Root:
            def __init__(self, enabled):
                if enabled:
                    self.active = True
    """)
    symbol = base_graph.root.symbol
    graph = resolve_owner_graph(
        index, symbol, root_param_prefixes={"enabled": ("enabled",)})
    node = graph.root
    control = next(item for item in index.controls
                   if item.enclosing_callable.qualified_name == "Root.__init__")

    from_binding = ConfigExpressionEvaluator(
        node.config_bindings, {"enabled": True}, env={})
    assert from_binding.expression(control.controlling).value is True
    assert from_binding.expression(control.controlling).premises == (
        (("enabled",), True),)

    from_actual = ConfigExpressionEvaluator(
        node.config_bindings, {"enabled": True},
        env={"enabled": ConfigExpressionEvaluator(
            node.config_bindings, {"enabled": False}, env={})
             .expression(control.controlling)})
    assert from_actual.expression(control.controlling).value is False


def test_literal_membership_guard_is_evaluated_without_name_semantics(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, choice):
                if choice in ("first", "second"):
                    self.active = True
    """)
    test = next(item.controlling for item in index.controls
                if item.kind == "if")
    inside = ConfigExpressionEvaluator(
        (), {}, {"choice": EvaluatedExpression("second")},
        allow_control_literals=True).expression(test)
    outside = ConfigExpressionEvaluator(
        (), {}, {"choice": EvaluatedExpression("other")},
        allow_control_literals=True).expression(test)
    assert inside is not None and inside.value is True
    assert outside is not None and outside.value is False


def test_membership_with_an_unknown_collection_item_stays_unknown(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, choice):
                if choice in ("first", runtime_value):
                    self.active = True
    """)
    test = next(item.controlling for item in index.controls
                if item.kind == "if")
    assert ConfigExpressionEvaluator(
        (), {}, {"choice": EvaluatedExpression("first")},
        allow_control_literals=True) \
        .expression(test) is None


def test_boolean_guard_uses_only_exact_python_short_circuit(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, present, mode):
                if present and mode != "disabled":
                    self.active = True
    """)
    test = next(item.controlling for item in index.controls
                if item.kind == "if")
    stopped = ConfigExpressionEvaluator(
        (), {}, {"present": EvaluatedExpression(False)},
        allow_control_literals=True).expression(test)
    assert stopped is not None and stopped.value is False
    # Once execution reaches an unknown operand, later syntax cannot be used
    # to manufacture a decision.
    assert ConfigExpressionEvaluator(
        (), {}, {"present": EvaluatedExpression(True)},
        allow_control_literals=True).expression(test) is None
    decided = ConfigExpressionEvaluator((), {}, {
        "present": EvaluatedExpression(True),
        "mode": EvaluatedExpression("enabled"),
    }, allow_control_literals=True).expression(test)
    assert decided is not None and decided.value is True


def test_control_literal_extension_is_opt_in(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, choice):
                if choice in ("first", "second"):
                    self.active = True
    """)
    test = next(item.controlling for item in index.controls
                if item.kind == "if")
    assert ConfigExpressionEvaluator(
        (), {}, {"choice": EvaluatedExpression("first")}).expression(test) \
        is None


def test_string_prefix_and_slice_protocols_are_separately_opt_in(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, choice):
                choice = choice[2:] if choice.startswith("X:") else choice
                if choice == "dense":
                    self.active = True
    """)
    binding = next(item for item in index.bindings
                   if item.value is not None and item.value.kind == "ifexp")
    env = {"choice": EvaluatedExpression("X:dense")}
    assert ConfigExpressionEvaluator(
        (), {}, env, allow_control_literals=True).expression(binding.value) \
        is None
    evaluated = ConfigExpressionEvaluator(
        (), {}, env, allow_control_literals=True,
        allow_string_protocols=True).expression(binding.value)
    assert evaluated is not None and evaluated.value == "dense"


def test_string_protocol_rejects_dynamic_prefix_and_noninteger_slice(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, choice, prefix, offset):
                a = choice.startswith(prefix)
                b = choice[offset:]
    """)
    values = [item.value for item in index.bindings if item.value is not None]
    evaluator = ConfigExpressionEvaluator(
        (), {}, {"choice": EvaluatedExpression("X:dense")},
        allow_string_protocols=True)
    assert all(evaluator.expression(item) is None for item in values)


def test_exact_environment_integer_can_select_sequence_position(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, values, position):
                selected = values[position]
    """)
    expression = next(item.value for item in index.bindings
                      if item.value is not None and item.value.kind == "subscript")
    env = {
        "values": EvaluatedExpression([11, 29]),
        "position": EvaluatedExpression(1),
    }
    assert ConfigExpressionEvaluator((), {}, env).expression(expression) is None
    assert ConfigExpressionEvaluator(
        (), {}, env, allow_string_protocols=True).expression(expression) is None
    result = ConfigExpressionEvaluator(
        (), {}, env, allow_string_protocols=True,
        allow_dynamic_sequence_index=True).expression(expression)
    assert result is not None and result.value == 29


def test_boolean_or_out_of_range_selector_stays_unknown(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, values, position):
                selected = values[position]
    """)
    expression = next(item.value for item in index.bindings
                      if item.value is not None and item.value.kind == "subscript")
    for selector in (True, 8):
        assert ConfigExpressionEvaluator((), {}, {
            "values": EvaluatedExpression([11, 29]),
            "position": EvaluatedExpression(selector),
        }, allow_string_protocols=True,
           allow_dynamic_sequence_index=True).expression(expression) is None


def test_closed_builtin_sequence_protocols_are_separately_opt_in(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, values):
                a = isinstance(values, tuple)
                b = len(values)
                c = list(reversed(values))
                d = min(1, b)
    """)
    expressions = {next(iter(item.targets)).name: item.value
                   for item in index.bindings if item.value is not None}
    env = {"values": EvaluatedExpression((7, 9))}
    assert all(ConfigExpressionEvaluator(
        (), {}, env, allow_control_literals=True).expression(item) is None
               for item in expressions.values())
    evaluator = ConfigExpressionEvaluator(
        (), {}, env, allow_control_literals=True,
        builtin_protocols={
            "isinstance", "len", "list", "tuple", "reversed", "min"})
    values = {}
    for name in ("a", "b", "c", "d"):
        result = evaluator.expression(expressions[name])
        assert result is not None
        evaluator.env[name] = result
        values[name] = result.value
    assert values == {"a": True, "b": 2, "c": [9, 7], "d": 1}


def test_isinstance_type_spelling_requires_its_own_lexical_proof(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, value):
                active = isinstance(value, int)
    """)
    expression = next(item.value for item in index.bindings
                      if item.value is not None)
    env = {"value": EvaluatedExpression(4)}
    assert ConfigExpressionEvaluator(
        (), {}, env, builtin_protocols={"isinstance"}) \
        .expression(expression) is None
    result = ConfigExpressionEvaluator(
        (), {}, env, builtin_protocols={"isinstance", "int"}) \
        .expression(expression)
    assert result is not None and result.value is True


def test_unknown_builtin_protocol_name_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="closed set"):
        ConfigExpressionEvaluator((), {}, builtin_protocols={"sorted"})


def test_boolean_not_is_control_literal_opt_in(tmp_path):
    index, _graph_value = _graph(tmp_path, """
        class Root:
            def __init__(self, final):
                active = not final
    """)
    expression = next(item.value for item in index.bindings
                      if item.value is not None)
    env = {"final": EvaluatedExpression(False)}
    assert ConfigExpressionEvaluator((), {}, env).expression(expression) is None
    assert ConfigExpressionEvaluator(
        (), {}, env, allow_control_literals=True).expression(expression) is None
    result = ConfigExpressionEvaluator(
        (), {}, env, allow_control_literals=True,
        allow_boolean_not=True).expression(expression)
    assert result is not None and result.value is True
