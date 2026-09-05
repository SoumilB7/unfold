"""Unanimous code proof for a container's default attention input interface.

The container and its exact construction frame are supplied by the caller.
This boundary discovers the callable field delegated by ``forward``, proves how
that field is installed by the constructor, preserves every exact default
implementation alternative, and requires each alternative to prove the same
primary/context QKV interface independently.

Runtime implementation selection (for example an available optimized kernel)
may remain unknown because it does not change the architectural input role;
unanimity across the complete exact constructor expression is required.  No
class, field, setter, local, or formal spelling carries semantics.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_input_interface import ContextFallbackAttentionInterface
from .attention_input_interface import (
    context_fallback_attention_interface_at_symbol,
)
from .constructor_values import (
    ConstructorFrame,
    EffectiveConstructorValue,
    resolve_effective_constructor_parameter,
)
from .expression_eval import ConfigExpressionEvaluator, EvaluatedExpression
from .expression_eval import guard_path_evidence
from .program_index import (
    BindingObservation,
    CallObservation,
    CallableRecord,
    ClassRecord,
    FieldAssignRecord,
    ParamRecord,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class DefaultAttentionImplementation:
    """One exact constructor-expression leaf and its independent interface."""

    call: CallObservation
    class_record: ClassRecord
    symbol: SymbolId
    interface: ContextFallbackAttentionInterface

    def __post_init__(self) -> None:
        if not isinstance(self.call, CallObservation) \
                or not isinstance(self.class_record, ClassRecord) \
                or not isinstance(self.symbol, SymbolId) \
                or not isinstance(
                    self.interface, ContextFallbackAttentionInterface):
            raise TypeError("a default implementation retains typed evidence")
        if self.class_record.symbol != self.symbol \
                or self.call.callee.kind != "name" \
                or self.call.callee.name != self.symbol.qualified_name \
                or self.call.owner is None \
                or self.call.owner.source != self.symbol.source \
                or self.interface.owner_symbol != self.symbol:
            raise ValueError("implementation leaf closes call/class/symbol/interface")


@dataclass(frozen=True)
class DefaultAttentionContainerInterface:
    """One container whose complete default alternatives share one interface."""

    frame: ConstructorFrame
    forward: CallableRecord
    delegate: CallObservation
    returned: ReturnObservation
    setter: CallableRecord
    setter_assignment: FieldAssignRecord
    setter_call: CallObservation
    selector_parameter: ParamRecord
    selector_value: EffectiveConstructorValue
    selector_binding: BindingObservation
    implementations: tuple[DefaultAttentionImplementation, ...]
    primary_formal: ParamRecord
    context_formal: ParamRecord
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        symbol = self.frame.target.symbol
        if not isinstance(self.frame, ConstructorFrame) \
                or not isinstance(self.forward, CallableRecord) \
                or self.forward.owner != symbol \
                or not isinstance(self.delegate, CallObservation) \
                or self.delegate.enclosing_callable != self.forward.symbol \
                or not isinstance(self.returned, ReturnObservation) \
                or self.returned.enclosing_callable != self.forward.symbol \
                or self.returned.value is None \
                or self.returned.value.span != self.delegate.span:
            raise ValueError("container interface closes one exact returned delegate")
        if not isinstance(self.setter, CallableRecord) \
                or self.setter.owner != symbol \
                or not isinstance(self.setter_assignment, FieldAssignRecord) \
                or self.setter_assignment.enclosing_callable != self.setter.symbol \
                or not isinstance(self.setter_call, CallObservation) \
                or self.setter_call.enclosing_callable \
                != self.frame.constructor.symbol:
            raise ValueError("container interface closes its exact field installer")
        field = _self_field(self.delegate.callee)
        setter_params = tuple(
            item for item in self.setter.params
            if item.name != "self" and item.kind not in {"vararg", "kwarg"})
        if not field or self.setter_assignment.field != field \
                or self.setter_assignment.value.kind != "name" \
                or len(setter_params) != 1 \
                or self.setter_assignment.value.name != setter_params[0].name \
                or len(self.setter_call.args) != 1 or self.setter_call.kwargs \
                or self.setter_call.args[0].kind != "name" \
                or _self_field(self.setter_call.callee) \
                != self.setter.symbol.qualified_name.rsplit(".", 1)[-1]:
            raise ValueError("delegate field is installed by the exact helper call")
        if not isinstance(self.selector_parameter, ParamRecord) \
                or self.selector_parameter not in self.frame.constructor.params \
                or not isinstance(self.selector_value, EffectiveConstructorValue) \
                or self.selector_value.frame != self.frame \
                or self.selector_value.parameter != self.selector_parameter \
                or self.selector_value.value is not None \
                or not isinstance(self.selector_binding, BindingObservation) \
                or self.selector_binding.enclosing_callable \
                != self.frame.constructor.symbol \
                or len(self.selector_binding.targets) != 1 \
                or not _target_is_name(
                    self.selector_binding.targets[0],
                    self.selector_parameter.name) \
                or self.setter_call.args[0].name \
                != self.selector_parameter.name \
                or not _exact_none_guard(
                    self.selector_binding, self.selector_parameter.name):
            raise ValueError("default selection closes one exact None constructor formal")
        if not self.implementations \
                or tuple(sorted(self.implementations,
                                key=lambda item: _span_key(item.call.span))) \
                != self.implementations \
                or len({item.call.span for item in self.implementations}) \
                != len(self.implementations) \
                or _constructor_call_spans(self.selector_binding.value) \
                != tuple(item.call.span for item in self.implementations):
            raise ValueError("default implementation alternatives are complete and ordered")
        if any(item.interface.container_class is None
               or item.interface.container_class.symbol != symbol
               for item in self.implementations):
            raise ValueError("every default interface names this exact container")
        if not isinstance(self.primary_formal, ParamRecord) \
                or not isinstance(self.context_formal, ParamRecord) \
                or self.primary_formal == self.context_formal \
                or self.primary_formal not in self.forward.params \
                or self.context_formal not in self.forward.params \
                or not self.context_formal.has_default \
                or self.context_formal.default.kind != "constant" \
                or self.context_formal.default.const_value is not None:
            raise ValueError("container forward retains primary + optional context formals")
        required = {
            self.delegate.span, self.returned.span, self.setter_assignment.span,
            self.setter_call.span, self.selector_binding.span,
            *self.selector_value.spans,
            *(item.call.span for item in self.implementations),
            *(item.class_record.span for item in self.implementations),
            *(span for item in self.implementations
              for span in item.interface.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan)
                       for item in self.spans):
            raise ValueError("container-interface provenance closes every edge")


def default_attention_container_interface(
        index: ProgramIndex,
        frame: ConstructorFrame,
) -> ReaderResult[DefaultAttentionContainerInterface]:
    """Prove unanimous default processor semantics for one exact container."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(frame, ConstructorFrame):
        raise TypeError("default attention interface requires index + frame")
    owner = frame.graph.root.occurrence
    forward = index.callable_by_symbol(SymbolId(
        frame.target.symbol.source,
        f"{frame.target.symbol.qualified_name}.forward"))
    if forward is None:
        return _failed(owner, "the container has no exact forward")
    returned_rows = tuple(item for item in index.return_observations_in(
        forward.symbol) if not item.guard and item.value is not None
                         and item.value.kind == "call")
    if len(returned_rows) != 1:
        return _failed(owner, "forward has no unique unguarded returned call")
    returned = returned_rows[0]
    delegates = tuple(item for item in index.calls_in(forward.symbol)
                      if item.span == returned.value.span
                      and _self_field(item.callee))
    if len(delegates) != 1:
        return _failed(owner, "the returned call is not one exact self delegate")
    delegate = delegates[0]
    field = _self_field(delegate.callee)

    setters = []
    for callable_record in index.callables_of(frame.target.symbol):
        if callable_record.symbol in {
                frame.constructor.symbol, forward.symbol}:
            continue
        assignments = tuple(
            item for item in index.field_assigns_of(frame.target.symbol)
            if item.enclosing_callable == callable_record.symbol
            and item.field == field and not item.guard
            and item.value.kind == "name")
        if len(assignments) != 1:
            continue
        ordinary = tuple(item for item in callable_record.params
                         if item.name != "self"
                         and item.kind not in {"vararg", "kwarg"})
        if len(ordinary) == 1 and assignments[0].value.name == ordinary[0].name:
            setters.append((callable_record, assignments[0]))
    if len(setters) != 1:
        return _failed(owner, "delegate field has no unique exact installer")
    setter, setter_assignment = setters[0]
    setter_leaf = setter.symbol.qualified_name.rsplit(".", 1)[-1]
    setter_calls = tuple(
        item for item in index.calls_in(frame.constructor.symbol)
        if not item.guard and _self_field(item.callee) == setter_leaf
        and len(item.args) == 1 and not item.kwargs
        and item.args[0].kind == "name" and item.args[0].name)
    if len(setter_calls) != 1:
        return _failed(owner, "initializer has no exact installer call")
    setter_call = setter_calls[0]
    local = setter_call.args[0].name
    params = tuple(item for item in frame.constructor.params
                   if item.name == local and item.name != "self"
                   and item.kind not in {"vararg", "kwarg"})
    if len(params) != 1:
        return _failed(owner, "installer input is not one constructor formal")
    selector_parameter = params[0]
    selector_result = resolve_effective_constructor_parameter(
        index, frame, selector_parameter.name)
    if selector_result.status != "resolved" \
            or selector_result.require_value().value is not None:
        return _failed(owner, "installer formal is not exactly defaulted to None")
    selector_value = selector_result.require_value()
    env = {selector_parameter.name: EvaluatedExpression(
        selector_value.value, spans=selector_value.spans)}
    bindings = tuple(
        item for item in index.bindings_in(frame.constructor.symbol)
        if _span_before(item.span, setter_call.span)
        and any(_target_is_name(target, local) for target in item.targets))
    states = []
    for binding in bindings:
        evidence = guard_path_evidence(
            index, frame.constructor.symbol, binding.guard,
            ConfigExpressionEvaluator(
                (), {}, dict(env), allow_control_literals=True),
            binding.span)
        if evidence is None or not isinstance(evidence.value, bool):
            return _failed(owner, "one installer-input write has an unknown guard")
        states.append((binding, evidence.value))
    active = tuple(binding for binding, state in states if state is True)
    if len(active) != 1 or active[0].value is None:
        return _failed(owner, "default selection has no unique active binding")
    selector_binding = active[0]
    leaves = _constructor_leaves(
        index, frame, selector_binding.value)
    if not leaves:
        return _failed(owner, "default expression has no exact constructor leaves")
    implementations = []
    for call, class_record, symbol in leaves:
        interface_result = context_fallback_attention_interface_at_symbol(
            index, symbol, frame.target.symbol)
        if interface_result.status != "resolved":
            detail = "; ".join(
                item.detail for item in interface_result.failures
                if item.detail) or "unknown input-interface failure"
            return _failed(
                owner,
                "default implementation "
                f"{symbol.qualified_name!r} lacks input proof: {detail}")
        implementations.append(DefaultAttentionImplementation(
            call, class_record, symbol, interface_result.require_value()))
    implementations = tuple(sorted(
        implementations, key=lambda item: _span_key(item.call.span)))

    roles = []
    for implementation in implementations:
        interface = implementation.interface
        actuals = _bind_delegate(delegate, interface.callable)
        container_actual = actuals.get(interface.container_formal.name)
        primary_actual = actuals.get(interface.primary_formal.name)
        context_actual = actuals.get(interface.context_formal.name)
        if container_actual is None or container_actual.kind != "name" \
                or container_actual.name != "self" \
                or primary_actual is None or primary_actual.kind != "name" \
                or context_actual is None or context_actual.kind != "name":
            return _failed(owner, "delegate does not forward exact interface formals")
        roles.append((primary_actual.name, context_actual.name))
    if len(set(roles)) != 1:
        return _failed(owner, "default implementations disagree on input interface")
    primary_name, context_name = roles[0]
    forward_params = {item.name: item for item in forward.params
                      if item.name != "self"
                      and item.kind not in {"vararg", "kwarg"}}
    primary = forward_params.get(primary_name)
    context = forward_params.get(context_name)
    if primary is None or context is None or primary == context \
            or not context.has_default \
            or context.default.kind != "constant" \
            or context.default.const_value is not None:
        return _failed(owner, "delegate inputs are not container-forward formals")
    spans = tuple(dict.fromkeys(span for span in (
        delegate.span, returned.span, setter_assignment.span, setter_call.span,
        selector_binding.span, *selector_value.spans,
        *(item.call.span for item in implementations),
        *(item.class_record.span for item in implementations),
        *(span for item in implementations for span in item.interface.spans),
    ) if isinstance(span, SourceSpan)))
    value = DefaultAttentionContainerInterface(
        frame, forward, delegate, returned, setter, setter_assignment,
        setter_call, selector_parameter, selector_value, selector_binding,
        implementations, primary, context, spans)
    return ReaderResult.resolved(owner, value, provenance=(ReaderProvenance(
        "source", spans=spans,
        detail="exact container delegate + unanimous default Q/K/V interfaces"),))


def _constructor_leaves(index, frame, expression):
    expressions = []

    def visit(item):
        if item.kind == "ifexp" and len(item.children) == 3:
            return visit(item.children[0]) and visit(item.children[2])
        if item.kind == "call":
            expressions.append(item)
            return True
        return False

    if not visit(expression):
        return ()
    rows = []
    for expression in expressions:
        calls = tuple(item for item in index.calls_in(frame.constructor.symbol)
                      if item.span == expression.span)
        if len(calls) != 1 or calls[0].callee.kind != "name" \
                or not calls[0].callee.name:
            return ()
        classes = tuple(item for item in index.classes
                        if item.symbol.source == frame.target.symbol.source
                        and item.symbol.qualified_name == calls[0].callee.name)
        if len(classes) != 1:
            return ()
        rows.append((calls[0], classes[0], classes[0].symbol))
    return tuple(rows)


def _constructor_call_spans(expression):
    if expression is None:
        return None
    if expression.kind == "call":
        return (expression.span,)
    if expression.kind != "ifexp" or len(expression.children) != 3:
        return None
    left = _constructor_call_spans(expression.children[0])
    right = _constructor_call_spans(expression.children[2])
    if left is None or right is None:
        return None
    return tuple(sorted((*left, *right), key=_span_key))


def _bind_delegate(call, callable_record):
    params = list(callable_record.params)
    if callable_record.kind == "method":
        if not params:
            return {}
        params = params[1:]
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    by_name = {item.name: item for item in params
               if item.kind not in {"vararg", "kwarg"}}
    if len(call.args) > len(positional):
        return {}
    bound = {param.name: actual
             for param, actual in zip(positional, call.args)}
    for name, actual in call.kwargs:
        if name == "**":
            continue
        if name not in by_name or name in bound:
            return {}
        bound[name] = actual
    return bound


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" \
        else None


def _target_is_name(target, name):
    return target.kind == "name" and target.name == name


def _exact_none_guard(binding, name):
    if len(binding.guard) != 1:
        return False
    step = binding.guard[0]
    test = step.test
    if step.kind != "if" or test is None or test.kind != "compare" \
            or test.operator != "is" or len(test.children) != 2:
        return False
    left, right = test.children
    return (
        left.kind == "name" and left.name == name
        and right.kind == "constant" and right.const_value is None
    ) or (
        right.kind == "name" and right.name == name
        and left.kind == "constant" and left.const_value is None
    )


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) \
        < (right.line, right.col)


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


def _failed(owner, detail):
    return ReaderResult.failed(owner, (ReaderFailure(
        "incomplete_graph", detail),))


__all__ = [
    "DefaultAttentionContainerInterface",
    "DefaultAttentionImplementation",
    "default_attention_container_interface",
]
