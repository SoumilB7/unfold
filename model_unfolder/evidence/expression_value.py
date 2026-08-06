"""Evaluate one exact owner expression from code and typed config premises.

This is a deliberately small interpreter for architecture values computed in
an exact constructor.  It follows unique unguarded ``self`` fields and local
assignments, exact owner-qualified config paths, arithmetic, and unshadowed
numeric casts.  Unsupported syntax, rival writes, missing config values and
runtime-dependent control flow return ``None`` rather than a default.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator

from .attention import exact_config_path_for_expression
from .config_guard import NormalizedConfigValue
from .program_index import ExprNode, ProgramIndex, SourceSpan, SymbolId


@dataclass(frozen=True)
class EvaluatedExpressionValue:
    value: object
    premises: tuple[tuple[tuple[str, ...], str, object], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if tuple(dict.fromkeys(self.premises)) != self.premises \
                or any(not path or kind not in {
                    "config_declared", "class_default"}
                    for path, kind, _value in self.premises):
            raise ValueError("evaluated values carry exact typed premises")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("evaluated values carry exact source spans")


def evaluate_owner_expression(
    index: ProgramIndex,
    owner_node,
    expression: ExprNode,
    config_selector,
    *,
    config_prefix: tuple[str, ...] = (),
) -> EvaluatedExpressionValue | None:
    """Evaluate one expression through its exact owner constructor only."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("expression evaluation requires a ProgramIndex")
    if not isinstance(expression, ExprNode):
        raise TypeError("expression evaluation requires an ExprNode")
    if config_selector is None:
        return None
    evaluator = _Evaluator(
        index, owner_node, config_selector, tuple(config_prefix))
    return evaluator.expression(expression, expression.span, frozenset())


class _Evaluator:
    def __init__(self, index, node, selector, prefix):
        self._program_index = index
        self._node = node
        self._selector = selector
        self._prefix = prefix
        self._constructor = SymbolId(
            node.symbol.source, f"{node.symbol.qualified_name}.__init__")

    def expression(self, expression, before, seen):
        if expression is None or expression.span is None:
            return None
        path = exact_config_path_for_expression(
            self._program_index, self._node, expression,
            config_prefix=self._prefix)
        if path is not None:
            return self._config(path, expression.span)
        if expression.kind == "constant":
            return EvaluatedExpressionValue(
                expression.const_value, (), (expression.span,))
        if expression.kind == "attribute" and _self_field(expression):
            field = expression.name
            key = ("field", field)
            if key in seen:
                return None
            assigns = tuple(item for item in self._program_index.field_assigns_of(
                self._node.symbol) if item.field == field and not item.guard
                and item.enclosing_callable == self._constructor)
            all_assigns = tuple(
                item for item in self._program_index.field_assigns_of(
                    self._node.symbol) if item.field == field
                and item.enclosing_callable == self._constructor)
            if len(assigns) != 1 or len(all_assigns) != 1 \
                    or assigns[0].value is None:
                return None
            value = self.expression(
                assigns[0].value, assigns[0].span, seen | {key})
            return _with_span(value, assigns[0].span)
        if expression.kind == "name" and expression.name:
            key = ("local", expression.name)
            if key in seen or before is None:
                return None
            bindings = tuple(item for item in self._program_index.bindings_in(
                self._constructor) if item.span is not None
                and _span_before(item.span, before)
                and any(target.kind == "name"
                        and target.name == expression.name
                        for target in item.targets))
            if len(bindings) != 1 or bindings[0].guard \
                    or bindings[0].value is None:
                return None
            value = self.expression(
                bindings[0].value, bindings[0].span, seen | {key})
            return _with_span(value, bindings[0].span)
        if expression.kind == "binop" and len(expression.children) == 2:
            left = self.expression(expression.children[0], expression.span, seen)
            right = self.expression(expression.children[1], expression.span, seen)
            functions = {
                "+": operator.add, "-": operator.sub, "*": operator.mul,
                "/": operator.truediv, "//": operator.floordiv,
                "%": operator.mod,
            }
            if left is None or right is None \
                    or expression.operator not in functions:
                return None
            try:
                value = functions[expression.operator](left.value, right.value)
            except (ArithmeticError, TypeError, ValueError):
                return None
            return _combined(value, expression.span, left, right)
        if expression.kind == "unaryop" and len(expression.children) == 1 \
                and expression.operator in {"+", "-"}:
            child = self.expression(expression.children[0], expression.span, seen)
            if child is None:
                return None
            try:
                value = +child.value if expression.operator == "+" else -child.value
            except TypeError:
                return None
            return _combined(value, expression.span, child)
        if expression.kind == "call" and len(expression.children) == 2:
            callee, argument = expression.children
            if callee.kind != "name" or callee.name not in {"int", "float"} \
                    or _name_is_shadowed(
                        self._program_index, self._constructor, callee.name):
                return None
            child = self.expression(argument, expression.span, seen)
            if child is None:
                return None
            try:
                value = int(child.value) if callee.name == "int" \
                    else float(child.value)
            except (TypeError, ValueError, OverflowError):
                return None
            return _combined(value, expression.span, child)
        return None

    def _config(self, path, span):
        selected = self._selector(path)
        # The normalized wrapper carries dependency paths/kinds but not each
        # dependency's original value.  Inventing those values would make the
        # numeric proof self-certifying, so this evaluator refuses it until a
        # value-bearing normalized-premise contract exists.
        if isinstance(selected, NormalizedConfigValue):
            return None
        kind = "config_declared"
        if isinstance(selected, tuple) and len(selected) in {2, 3} \
                and isinstance(selected[0], bool):
            present, value = selected[:2]
            if len(selected) == 3:
                kind = selected[2]
        else:
            present, value = selected is not None, selected
        if not present or kind not in {"config_declared", "class_default"}:
            return None
        frozen = _freeze(value)
        return EvaluatedExpressionValue(
            frozen, ((tuple(path), kind, frozen),), (span,))


def _self_field(expression):
    return (expression.kind == "attribute" and len(expression.children) == 1
            and expression.children[0].kind == "name"
            and expression.children[0].name == "self")


def _name_is_shadowed(index, callable_symbol, name):
    return any(item.name == name and item.context in {
        "parameter", "store", "del"} for item in index.identifiers_in(
            callable_symbol)) or any(
        item.name == name for item in index.module_bindings_in(
            callable_symbol.source))


def _with_span(value, span):
    if value is None:
        return None
    return EvaluatedExpressionValue(
        value.value, value.premises,
        tuple(dict.fromkeys((*value.spans, span))))


def _combined(value, span, *parts):
    return EvaluatedExpressionValue(
        value,
        tuple(dict.fromkeys(
            premise for part in parts for premise in part.premises)),
        tuple(dict.fromkeys((
            span, *(item for part in parts for item in part.spans)))))


def _span_before(left, right):
    return left.source == right.source and (
        left.end_line or left.line, left.end_col or left.col) < (
        right.line, right.col)


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


__all__ = ["EvaluatedExpressionValue", "evaluate_owner_expression"]
