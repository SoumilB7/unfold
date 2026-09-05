"""U9-E — exact multimodal multi-axis position-construction route.

This reader proves a narrow positive relation:

* an exact wrapper occurrence already has code-proven modality fusion;
* that same callable passes a value through the framework ``position_ids``
  keyword to one exactly constructed child;
* the value's exact return-dependency closure reaches a framework-bound
  ``torch.stack`` of at least three axes.

No class, field, model, helper, or local-variable spelling selects the result.
The keyword is a closed execution protocol; the mechanism is completed only
by the helper bodies.  Missing positive evidence is never interpreted as a
one-dimensional or absent position scheme.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId, resolve_component_root
from .construction_calls import (
    resolve_construction_call_in_graph,
    resolve_import_reference,
)
from .fusion import FusionExecutionObservation, fusion_execution_observations
from .models import SourceBundle
from .program_index import (
    CallObservation, ExprNode, ProgramIndex, SourceSpan, SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


POSITION_INPUT_PROTOCOL = "position_ids"
MULTIAXIS_STACK_PROTOCOL = "torch.stack"


@dataclass(frozen=True)
class MultiaxisPositionRoute:
    owner_occurrence: OwnerOccurrenceId
    fusion: FusionExecutionObservation
    consumer: CallObservation
    producer: CallObservation
    helper_trace: tuple[SymbolId, ...]
    stack_call: CallObservation
    axis_count: int
    conditional: bool
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.fusion, FusionExecutionObservation):
            raise TypeError("a multi-axis route is exact fusion-owner qualified")
        if self.fusion.occurrence != self.owner_occurrence:
            raise ValueError("the position route and fusion share one occurrence")
        if any(not isinstance(item, CallObservation)
               for item in (self.consumer, self.producer, self.stack_call)):
            raise TypeError("a multi-axis route carries exact calls")
        if self.consumer.enclosing_callable != self.fusion.callable_symbol \
                or self.producer.enclosing_callable != self.fusion.callable_symbol:
            raise ValueError("consumer and producer belong to the fusion callable")
        if POSITION_INPUT_PROTOCOL not in {
                name for name, _value in self.consumer.kwargs}:
            raise ValueError("the consumer uses the closed position-input protocol")
        if not self.helper_trace \
                or self.helper_trace[0] != _self_method_symbol(
                    self.producer, self.fusion.owner) \
                or self.helper_trace[-1] != self.stack_call.enclosing_callable:
            raise ValueError("the helper trace is contiguous from the producer")
        if self.axis_count < 3 or _stack_axis_count(self.stack_call) != self.axis_count:
            raise ValueError("multi-axis construction stacks at least three axes")
        required = {
            self.consumer.span, self.producer.span, self.stack_call.span,
            *(call.span for call in self.fusion.operation_calls),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("multi-axis route provenance closes every join")


def multimodal_multiaxis_position_result(
    index: ProgramIndex,
    bundle: SourceBundle,
) -> ReaderResult[tuple[MultiaxisPositionRoute, ...]]:
    """Prove exact wrapper routes that construct >=3-axis position ids."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("multi-axis position reading requires ProgramIndex + SourceBundle")
    root = resolve_component_root(index, bundle, "root")
    if root.status != "resolved":
        return ReaderResult.failed(None, (ReaderFailure(
            "incomplete_graph", f"root component address is {root.status}"),))

    routes = []
    blockers = []
    for fusion in fusion_execution_observations(index, bundle):
        if fusion.evidence.kind not in {
                "placeholder_replace", "unified_multimodal_stream"}:
            continue
        for consumer in index.calls_in(fusion.callable_symbol):
            values = tuple(value for name, value in consumer.kwargs
                           if name == POSITION_INPUT_PROTOCOL)
            if len(values) != 1:
                continue
            child = resolve_construction_call_in_graph(
                index, root.graph, fusion.occurrence, consumer)
            if child.status != "resolved" or child.selected.kind != "internal":
                blockers.append(ReaderFailure(
                    "incomplete_graph",
                    "a position-input consumer is not one exact constructed child",
                    consumer.span))
                continue
            producers = _dependent_calls(
                index, fusion.callable_symbol, values[0], consumer.span)
            for producer in producers:
                first = _self_method_symbol(producer, fusion.owner)
                if first is None:
                    continue
                proofs = _multiaxis_helpers(index, fusion.owner, first)
                for helper_trace, stack_call in proofs:
                    spans = tuple(dict.fromkeys(
                        span for span in (
                            *(call.span for call in fusion.operation_calls),
                            consumer.span, producer.span, stack_call.span,
                        ) if isinstance(span, SourceSpan)))
                    routes.append(MultiaxisPositionRoute(
                        fusion.occurrence, fusion, consumer, producer,
                        helper_trace, stack_call, _stack_axis_count(stack_call),
                        bool(producer.guard or stack_call.guard), spans))

    if not routes:
        return ReaderResult.failed(root.occurrence, tuple(blockers) or (ReaderFailure(
            "incomplete_graph",
            "no exact fusion wrapper constructs multi-axis position ids"),))
    routes = tuple(sorted(
        dict.fromkeys(routes),
        key=lambda item: (
            item.consumer.span.source.canonical_path,
            item.consumer.span.line, item.producer.span.line,
            item.stack_call.span.line)))
    spans = tuple(dict.fromkeys(
        span for route in routes for span in route.spans))
    return ReaderResult.resolved(
        root.occurrence, routes,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=("exact fusion wrapper position-input dependency reaches "
                    "a framework-bound >=3-axis stack")),))


def _multiaxis_helpers(index, owner, first):
    out = []
    queue = [((first,), first)]
    seen = set()
    while queue:
        trace, callable_symbol = queue.pop(0)
        if callable_symbol in seen:
            continue
        seen.add(callable_symbol)
        record = index.callable_by_symbol(callable_symbol)
        if record is None or record.owner != owner:
            continue
        returns = index.return_observations_in(callable_symbol)
        calls = tuple(dict.fromkeys(
            call for returned in returns if returned.value is not None
            for call in _dependent_calls(
                index, callable_symbol, returned.value, returned.span)))
        for call in calls:
            proof = resolve_import_reference(
                index, callable_symbol.source, callable_symbol, call.callee)
            if proof is not None \
                    and proof.qualified_target == MULTIAXIS_STACK_PROTOCOL \
                    and _stack_axis_count(call) >= 3:
                out.append((trace, call))
            child = _self_method_symbol(call, owner)
            if child is not None and child not in trace:
                queue.append(((*trace, child), child))
    return tuple(out)


def _dependent_calls(index, callable_symbol, expression, cutoff, visiting=()):
    """Positive return/use dependencies; rivals are all retained, never picked."""
    if not isinstance(expression, ExprNode) or expression.span is None:
        return ()
    key = (expression.kind, expression.name, expression.span, cutoff)
    if key in visiting:
        return ()
    visiting = (*visiting, key)
    out = []
    if expression.kind == "name" and expression.name:
        # Exact accumulator provenance: a returned ``cat(list)`` depends on
        # every value appended to that same local list before the use.  Only
        # the closed one-argument ``append`` form is accepted; other mutations
        # are not silently treated as equivalent.
        mutations = tuple(
            call for call in index.calls_in(callable_symbol)
            if _before(call.span, cutoff)
            and call.receiver is not None
            and call.receiver.kind == "name"
            and call.receiver.name == expression.name
            and call.callee.name in {
                "append", "extend", "insert", "add", "update"})
        if mutations:
            if any(call.callee.name != "append" or len(call.args) != 1
                   for call in mutations):
                return ()
            for call in mutations:
                out.extend(_dependent_calls(
                    index, callable_symbol, call.args[0], call.span, visiting))
        bindings = tuple(
            binding for binding in index.bindings_in(callable_symbol)
            if _before(binding.span, cutoff)
            and any(expression.name in _target_names(target)
                    for target in binding.targets))
        for binding in _reaching_bindings(bindings):
            out.extend(_dependent_calls(
                index, callable_symbol, binding.value, binding.span, visiting))
        return _unique_calls(out)
    for child in expression.children:
        if isinstance(child, ExprNode):
            out.extend(_dependent_calls(
                index, callable_symbol, child, cutoff, visiting))
    for _name, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            out.extend(_dependent_calls(
                index, callable_symbol, child, cutoff, visiting))
    if expression.kind == "call":
        out.extend(call for call in index.calls_in(callable_symbol)
                   if call.span == expression.span)
    return _unique_calls(out)


def _self_method_symbol(call, owner):
    callee = call.callee
    if callee.kind != "attribute" or len(callee.children) != 1 \
            or callee.children[0].kind != "name" \
            or callee.children[0].name != "self":
        return None
    symbol = SymbolId(owner.source, f"{owner.qualified_name}.{callee.name}")
    record = call.enclosing_callable
    return symbol if symbol != record else None


def _stack_axis_count(call):
    if not call.args:
        return 0
    axes = call.args[0]
    return len(axes.children) if axes.kind in {"list", "tuple"} else 0


def _target_names(expression):
    if not isinstance(expression, ExprNode):
        return set()
    out = {expression.name} if expression.kind == "name" and expression.name else set()
    for child in expression.children:
        out.update(_target_names(child))
    return out


def _reaching_bindings(bindings):
    """Conservative source-order versions for one exact local spelling.

    A later unconditional assignment kills every earlier version.  Conditional
    alternatives after it remain possible, and repeated writes under the same
    exact guard keep only their latest version.  This is deliberately narrower
    than a CFG claim but prevents a dead overwritten multi-axis producer from
    certifying the final value.
    """
    ordered = sorted(
        bindings,
        key=lambda item: (
            item.span.line, item.span.col,
            item.span.end_line or item.span.line,
            item.span.end_col or item.span.col))
    last_unconditional = max(
        (index for index, item in enumerate(ordered) if not item.guard),
        default=-1)
    eligible = ordered[last_unconditional:] if last_unconditional >= 0 else ordered
    latest = {}
    for item in eligible:
        latest[tuple(item.guard)] = item
    return tuple(sorted(
        latest.values(),
        key=lambda item: (item.span.line, item.span.col)))


def _before(first, second):
    if first is None or second is None or first.source != second.source:
        return False
    return (first.end_line or first.line, first.end_col or first.col) <= \
        (second.line, second.col)


def _unique_calls(calls):
    out, seen = [], set()
    for call in calls:
        if call.span in seen:
            continue
        seen.add(call.span)
        out.append(call)
    return tuple(out)


__all__ = [
    "POSITION_INPUT_PROTOCOL", "MULTIAXIS_STACK_PROTOCOL",
    "MultiaxisPositionRoute", "multimodal_multiaxis_position_result",
]
