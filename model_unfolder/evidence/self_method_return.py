"""Neutral exact transport across one indexed ``self.method(...)`` return.

This boundary proves Python address/dataflow shape only:

    caller call -> exact same-class helper -> exact explicit arguments
      -> one exact unguarded return -> exact caller assignment lanes

It assigns no meaning to the helper, the returned values, or target spellings.
Branch-local definitions inside the helper remain for the consuming reader to
interpret.  Inherited/dynamic helpers, starred calls, rival definitions,
guarded/multiple returns and arity disagreement remain unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .program_index import (
    BindingObservation,
    CallObservation,
    CallableRecord,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class SelfMethodArgumentBinding:
    """One explicit actual bound to one exact helper formal."""

    formal: ParamRecord
    actual: ExprNode

    def __post_init__(self) -> None:
        if not isinstance(self.formal, ParamRecord) \
                or not isinstance(self.actual, ExprNode):
            raise TypeError("a helper argument binding is expression-typed")
        if self.formal.kind in {"vararg", "kwarg"}:
            raise ValueError("variadic formals are never exact helper bindings")


@dataclass(frozen=True)
class SelfMethodReturnLane:
    """One exact helper-return lane installed into one caller target lane."""

    lane_index: int
    caller_target: ExprNode
    returned_value: ExprNode

    def __post_init__(self) -> None:
        if isinstance(self.lane_index, bool) or not isinstance(
                self.lane_index, int) or self.lane_index < 0:
            raise ValueError("a helper return lane has a non-negative index")
        if not isinstance(self.caller_target, ExprNode) \
                or not isinstance(self.returned_value, ExprNode):
            raise TypeError("helper return lanes carry exact expressions")


@dataclass(frozen=True)
class SelfMethodReturnTransport:
    """Closed exact call/return/unpack transport for one same-class helper."""

    owner_occurrence: OwnerOccurrenceId
    caller: CallableRecord
    helper: CallableRecord
    call: CallObservation
    caller_definition: BindingObservation
    returned: ReturnObservation
    arguments: tuple[SelfMethodArgumentBinding, ...]
    lanes: tuple[SelfMethodReturnLane, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.caller, CallableRecord) \
                or not isinstance(self.helper, CallableRecord) \
                or self.caller.owner is None \
                or self.helper.owner != self.caller.owner:
            raise TypeError("self-method transport stays in one exact class")
        if self.caller.owner.source != self.owner_occurrence.root.source:
            raise ValueError("helper transport belongs to the exact owner root")
        if not isinstance(self.call, CallObservation) \
                or self.call.enclosing_callable != self.caller.symbol:
            raise ValueError("transport carries the exact caller call")
        if _self_method_name(self.call.callee) != _method_leaf(self.helper.symbol):
            raise ValueError("the call names the exact indexed helper")
        if not isinstance(self.caller_definition, BindingObservation) \
                or self.caller_definition.enclosing_callable != self.caller.symbol \
                or self.caller_definition.value is None \
                or not _contains_span(
                    self.caller_definition.value, self.call.span):
            raise ValueError("the helper call belongs to its exact caller definition")
        if not isinstance(self.returned, ReturnObservation) \
                or self.returned.enclosing_callable != self.helper.symbol \
                or self.returned.value is None or self.returned.guard:
            raise ValueError("transport carries one unguarded exact helper return")
        if any(not isinstance(item, SelfMethodArgumentBinding)
               for item in self.arguments):
            raise TypeError("transport arguments are exact typed bindings")
        expected_formals = tuple(
            item for item in self.helper.params
            if item.name != "self" and item.kind not in {"vararg", "kwarg"})
        if tuple(item.formal for item in self.arguments) != expected_formals:
            raise ValueError("every exact helper formal is bound in declaration order")
        if not self.lanes or any(
                not isinstance(item, SelfMethodReturnLane)
                for item in self.lanes) \
                or tuple(item.lane_index for item in self.lanes) \
                != tuple(range(len(self.lanes))):
            raise ValueError("transport lanes form one exact contiguous partition")
        targets = _lanes(self.caller_definition.targets)
        returned = _lanes((self.returned.value,))
        if tuple(item.caller_target for item in self.lanes) != targets \
                or tuple(item.returned_value for item in self.lanes) != returned:
            raise ValueError("transport lanes round-trip to caller targets and return")
        required = {
            self.call.span,
            self.caller_definition.span,
            self.returned.span,
            *(item.actual.span for item in self.arguments),
            *(item.caller_target.span for item in self.lanes),
            *(item.returned_value.span for item in self.lanes),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("helper transport retains every exact boundary span")


def resolve_self_method_return_transport(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    owner_occurrence: OwnerOccurrenceId,
    caller: SymbolId,
    call: CallObservation,
    *,
    defer_unsupported_to_consumer: bool = False,
) -> ReaderResult[SelfMethodReturnTransport]:
    """Resolve one exact same-class helper call and its return-lane mapping.

    The default remains execution-strict.  A consumer that has an independent,
    exact path evaluator may explicitly defer unsupported-region handling, but
    must then inspect every indexed unsupported region in the helper.  This
    flag proves transport only; it never upgrades callable completeness.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("self-method return transport requires a ProgramIndex")
    if type(defer_unsupported_to_consumer) is not bool:
        raise TypeError("unsupported-region deferral is an explicit boolean")
    root = require_resolved_component_root(
        root, caller="self-method return transport")
    if not isinstance(owner_occurrence, OwnerOccurrenceId):
        raise TypeError("self-method return transport is occurrence-qualified")
    owner_node = root.graph.node_for(owner_occurrence)
    if owner_node is None:
        return _failed(owner_occurrence, "the exact owner is absent from its graph")
    caller_record = index.callable_by_symbol(caller)
    if caller_record is None or caller_record.owner != owner_node.symbol:
        return _failed(owner_occurrence, "the caller is not the exact owner's method")
    if not isinstance(call, CallObservation) or call.enclosing_callable != caller:
        raise ValueError("the call belongs to the requested exact caller")
    helper_name = _self_method_name(call.callee)
    if helper_name is None:
        return _failed(owner_occurrence, "the call is not an exact self-method call")
    helper_symbol = SymbolId(
        caller.source, f"{caller_record.owner.qualified_name}.{helper_name}")
    helper = index.callable_by_symbol(helper_symbol)
    if helper is None or helper.owner != caller_record.owner:
        return _failed(owner_occurrence, "the exact same-class helper is not indexed")

    definitions = tuple(
        item for item in index.bindings_in(caller)
        if item.value is not None and _contains_span(item.value, call.span))
    if len(definitions) != 1:
        return _failed(owner_occurrence, "the helper call has no unique caller definition")
    returns = index.return_observations_in(helper_symbol)
    if len(returns) != 1 or returns[0].value is None or returns[0].guard:
        return _failed(owner_occurrence, "the helper has no unique unguarded value return")
    unsupported = tuple(
        item for item in index.unsupported_execution
        if item.enclosing_callable == helper_symbol)
    if unsupported and not defer_unsupported_to_consumer:
        return _failed(owner_occurrence, "unsupported helper execution prevents exact transport")

    arguments = _bind_arguments(helper, call)
    if arguments is None:
        return _failed(owner_occurrence, "the helper arguments are not exactly bound")
    targets = _lanes(definitions[0].targets)
    returned_values = _lanes((returns[0].value,))
    if not targets or len(targets) != len(returned_values):
        return _failed(owner_occurrence, "caller targets and helper return lanes disagree")
    lanes = tuple(
        SelfMethodReturnLane(i, target, returned)
        for i, (target, returned) in enumerate(zip(targets, returned_values)))
    spans = tuple(dict.fromkeys((
        call.span, definitions[0].span, returns[0].span,
        *(item.actual.span for item in arguments),
        *(item.caller_target.span for item in lanes),
        *(item.returned_value.span for item in lanes),
    )))
    value = SelfMethodReturnTransport(
        owner_occurrence, caller_record, helper, call, definitions[0], returns[0],
        arguments, lanes, spans)
    return ReaderResult.resolved(
        owner_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact same-class helper call/return/unpack transport"),))


def _failed(owner, detail):
    return ReaderResult.failed(
        owner, (ReaderFailure("incomplete_graph", detail),))


def _method_leaf(symbol: SymbolId) -> str:
    return symbol.qualified_name.rsplit(".", 1)[-1]


def _self_method_name(expr: ExprNode) -> str | None:
    if expr.kind != "attribute" or len(expr.children) != 1:
        return None
    base = expr.children[0]
    if base.kind == "name" and base.name == "self" and expr.name:
        return expr.name
    return None


def _contains_span(expr: ExprNode | None, span: SourceSpan | None) -> bool:
    if not isinstance(expr, ExprNode) or span is None:
        return False
    if expr.span == span:
        return True
    return any(_contains_span(child, span) for child in expr.children) \
        or any(_contains_span(child, span)
               for _name, child in expr.keyword_children)


def _lanes(expressions: tuple[ExprNode, ...]) -> tuple[ExprNode, ...]:
    if len(expressions) != 1:
        return tuple(expressions)
    expression = expressions[0]
    if expression.kind in {"tuple", "list"}:
        return tuple(expression.children)
    return (expression,)


def _bind_arguments(helper: CallableRecord, call: CallObservation):
    params = list(helper.params)
    if helper.kind != "method" or not params or params[0].name != "self":
        return None
    params = params[1:]
    if any(name == "**" for name, _value in call.kwargs):
        return None
    positional = tuple(
        item for item in params if item.kind in {"positional", "posonly"})
    by_name = {
        item.name: item for item in params
        if item.kind not in {"vararg", "kwarg", "posonly"}
    }
    if len(call.args) > len(positional):
        return None
    if any(item.kind == "starred" for item in call.args):
        return None
    bound = dict(zip((item.name for item in positional), call.args))
    for name, value in call.kwargs:
        if name not in by_name or name in bound:
            return None
        bound[name] = value
    exact = tuple(
        item for item in params if item.kind not in {"vararg", "kwarg"})
    if any(item.name not in bound and item.default is None for item in exact):
        return None
    # Omitted literal defaults are execution facts, but they have no caller
    # expression and cannot form an exact cross-boundary binding.  This rail is
    # intentionally stricter and requires an explicit actual for every lane.
    if any(item.name not in bound for item in exact):
        return None
    return tuple(
        SelfMethodArgumentBinding(item, bound[item.name]) for item in exact)


__all__ = [
    "SelfMethodArgumentBinding",
    "SelfMethodReturnLane",
    "SelfMethodReturnTransport",
    "resolve_self_method_return_transport",
]
