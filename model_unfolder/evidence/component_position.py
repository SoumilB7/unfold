"""Exact learned multi-axis position lookup before a nested repeated stage.

This reader covers a component shape that model-stage U8 readers cannot: a
separate input child performs learned coordinate lookups, adds them to the
input stream, and its result then feeds a nested repeated encoder.  The proof
is entirely structural and occurrence-qualified.  Names are addresses only.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import producer_sources_reaching_expressions
from .component_owner import OwnerOccurrenceId, require_resolved_component_root
from .construction_calls import resolve_import_reference
from .container_inventory import resolve_container_inventory
from .execution_flow import (
    InvocationNodeId,
    resolve_addressed_invocations,
    resolve_execution_flow,
)
from .program_index import (
    CallObservation,
    CallSiteId,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult


_PARAMETER_PROTOCOLS = frozenset({
    "torch.nn.Parameter", "torch.nn.parameter.Parameter",
})
_EMBEDDING_PROTOCOL = "torch.nn.functional.embedding"


@dataclass(frozen=True)
class ComponentLearnedPositionEvidence:
    component_occurrence: OwnerOccurrenceId
    stage_occurrence: OwnerOccurrenceId
    producer_occurrence: OwnerOccurrenceId
    field: str
    coordinate_parameter: str
    lookup_calls: tuple[CallObservation, ...]
    addition_span: SourceSpan
    producer_call: CallObservation
    stage_call: CallObservation
    spans: tuple[SourceSpan, ...]
    kind: str = "learned_absolute"
    application: str = "embedding_add"

    def __post_init__(self):
        if any(not isinstance(item, OwnerOccurrenceId) for item in (
                self.component_occurrence, self.stage_occurrence,
                self.producer_occurrence)):
            raise TypeError("component position evidence is occurrence-qualified")
        if not self.producer_occurrence.sites or not self.field \
                or not self.coordinate_parameter:
            raise ValueError("learned position evidence retains exact addresses")
        if len(self.lookup_calls) < 2 or len(set(self.lookup_calls)) \
                != len(self.lookup_calls):
            raise ValueError("multi-axis lookup carries two or more exact calls")
        if any(not isinstance(call, CallObservation) for call in self.lookup_calls) \
                or not isinstance(self.producer_call, CallObservation) \
                or not isinstance(self.stage_call, CallObservation):
            raise TypeError("learned position evidence retains exact calls")
        required = {
            self.addition_span, self.producer_call.span, self.stage_call.span,
            *(item.span for item in self.lookup_calls),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("component position provenance closes every join")
        if self.kind != "learned_absolute" \
                or self.application != "embedding_add":
            raise ValueError("this protocol has one closed architectural meaning")


def read_component_learned_position(index, root, stage):
    if not isinstance(index, ProgramIndex):
        raise TypeError("component position reading requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="read_component_learned_position")
    if not isinstance(stage, OwnerOccurrenceId) or root.graph.node_for(stage) is None:
        raise ValueError("component position reading requires an exact stage")
    component = root.graph.root.occurrence
    inventory = resolve_container_inventory(index, root, component)
    invocations = resolve_addressed_invocations(index, root, component, inventory)
    flow = resolve_execution_flow(index, root, component, inventory)
    if invocations.status != "resolved" or flow.status != "partial":
        return ReaderResult.failed(stage, (ReaderFailure(
            "incomplete_graph", "component invocation flow is unavailable"),))
    stage_invocations = tuple(
        item for item in invocations.addressed
        if item.callee_owner_occurrence == stage)
    if len(stage_invocations) != 1:
        sites = tuple(item.call.span for item in stage_invocations)
        if len(sites) > 1:
            return ReaderResult.ambiguous(stage, Ambiguity(sites=sites))
        return ReaderResult.absent(stage)
    stage_invocation = stage_invocations[0]
    stage_node = InvocationNodeId(stage_invocation.call_site, "addressed")
    predecessors = _predecessors(
        (*flow.proven_edges, *flow.conditional_edges), stage_node)
    candidates = []
    for invocation in invocations.addressed:
        node_id = InvocationNodeId(invocation.call_site, "addressed")
        if node_id not in predecessors:
            continue
        item = _learned_position_in_child(
            index, root, invocation, stage, stage_invocation.call)
        if item is not None:
            candidates.append(item)
    unique = {
        (item.producer_occurrence, item.field,
         tuple(call.span for call in item.lookup_calls)): item
        for item in candidates}
    if len(unique) > 1:
        return ReaderResult.ambiguous(
            stage, Ambiguity(sites=tuple(
                call.span for item in unique.values()
                for call in item.lookup_calls)))
    if not unique:
        return ReaderResult.absent(stage)
    value = next(iter(unique.values()))
    return ReaderResult.resolved(
        stage, value,
        provenance=(ReaderProvenance(
            "source", spans=value.spans,
            detail=("exact learned multi-axis lookup is added to a component "
                    "stream that reaches the exact repeated stage")),))


def _learned_position_in_child(index, root, invocation, stage, stage_call):
    occurrence = invocation.callee_owner_occurrence
    node = root.graph.node_for(occurrence)
    if node is None:
        return None
    callable_symbol = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.forward")
    record = index.callable_by_symbol(callable_symbol)
    if record is None:
        return None
    parameter_fields = {}
    for site in index.construction_sites_of(node.symbol):
        if site.target_kind != "field" or len(site.candidates) != 1:
            continue
        candidate = site.candidates[0]
        if candidate.symbol is not None:
            continue
        proof = resolve_import_reference(
            index, site.owner.source, site.enclosing_callable,
            candidate.reference)
        if proof is not None and proof.qualified_target in _PARAMETER_PROTOCOLS:
            parameter_fields[site.target] = site.span
    if not parameter_fields:
        return None

    formals = {item.name for item in record.params
               if item.name not in {"self", "cls"}}
    routes = []
    direct = _lookup_protocol(
        index, callable_symbol, parameter_fields)
    if direct is not None:
        field, origin, calls, lookup_return = direct
        producers = {CallSiteId.of(call): call for call in calls}
        addition = _stream_addition(
            index, callable_symbol, producers, formals - {origin},
            lookup_return)
        if addition is not None:
            routes.append((
                field, origin, calls, addition,
                (*tuple(call.span for call in calls), lookup_return.span)))

    # A coordinate helper may be a self-method whose returned table lookups
    # are then added to the data stream in ``forward``.  Bind the helper by its
    # exact owner/call address and require its call result to be the side of
    # the outer addition; a merely reachable helper body is insufficient.
    outer_returns = tuple(index.return_observations_in(callable_symbol))
    if len(outer_returns) == 1 and not outer_returns[0].guard \
            and outer_returns[0].value is not None:
        for helper_call in index.calls_in(callable_symbol):
            helper = _self_method(index, node.symbol, helper_call)
            if helper is None:
                continue
            protocol = _lookup_protocol(index, helper, parameter_fields)
            if protocol is None:
                continue
            field, origin, calls, helper_return = protocol
            producer = {CallSiteId.of(helper_call): helper_call}
            addition = _stream_addition(
                index, callable_symbol, producer, formals,
                outer_returns[0])
            if addition is None:
                continue
            routes.append((
                field, f"{helper.qualified_name}:{origin}", calls, addition,
                (helper_call.span, helper_return.span,
                 *(call.span for call in calls))))

    for field, origin, calls, addition, route_spans in routes:
        spans = tuple(dict.fromkeys(
            span for span in (
                parameter_fields[field], *route_spans,
                addition, invocation.call.span, stage_call.span,
            ) if isinstance(span, SourceSpan)))
        return ComponentLearnedPositionEvidence(
            root.graph.root.occurrence, stage,
            occurrence, field, origin, calls, addition,
            invocation.call, stage_call, spans)
    return None


def _lookup_protocol(index, callable_symbol, parameter_fields):
    record = index.callable_by_symbol(callable_symbol)
    if record is None:
        return None
    formals = {item.name for item in record.params
               if item.name not in {"self", "cls"}}
    groups = {}
    for call in index.calls_in(callable_symbol):
        proof = resolve_import_reference(
            index, callable_symbol.source, callable_symbol, call.callee)
        if proof is None or proof.qualified_target != _EMBEDDING_PROTOCOL \
                or len(call.args) < 2:
            continue
        field_axis = _parameter_field_axis(call.args[1])
        index_axis = _last_constant_index(call.args[0])
        origins = _parameter_origins(
            index, callable_symbol, call.args[0], call.span, set())
        if field_axis is None or index_axis is None or len(origins & formals) != 1:
            continue
        field, weight_axis = field_axis
        if field not in parameter_fields or weight_axis != index_axis:
            continue
        origin = next(iter(origins & formals))
        groups.setdefault((field, origin), []).append((index_axis, call))

    returns = tuple(index.return_observations_in(callable_symbol))
    if len(returns) != 1 or returns[0].guard or returns[0].value is None:
        return None
    for (field, origin), axes_calls in groups.items():
        by_axis = {axis: call for axis, call in axes_calls}
        if len(by_axis) < 2:
            continue
        calls = tuple(by_axis[axis] for axis in sorted(by_axis))
        producers = {CallSiteId.of(call): call for call in calls}
        all_sources, _, _, uncertain = producer_sources_reaching_expressions(
            index, callable_symbol,
            ((returns[0].span, (returns[0].value,)),), producers)
        if uncertain or set(producers) - set(all_sources):
            continue
        return field, origin, calls, returns[0]
    return None


def _self_method(index, owner_symbol, call):
    callee = call.callee
    if callee.kind != "attribute" or len(callee.children) != 1 \
            or callee.children[0].kind != "name" \
            or callee.children[0].name != "self":
        return None
    symbol = SymbolId(
        owner_symbol.source, f"{owner_symbol.qualified_name}.{callee.name}")
    record = index.callable_by_symbol(symbol)
    return symbol if record is not None and record.owner == owner_symbol else None


def _stream_addition(index, callable_symbol, producers, data_formals, returned):
    expressions = [returned.value]
    expressions.extend(
        binding.value for binding in index.bindings_in(callable_symbol)
        if not binding.guard and binding.value is not None)
    for expression in expressions:
        for addition in _additions(expression):
            left, right = addition.children
            left_sources, _, _, left_uncertain = producer_sources_reaching_expressions(
                index, callable_symbol, ((addition.span, (left,)),), producers)
            right_sources, _, _, right_uncertain = producer_sources_reaching_expressions(
                index, callable_symbol, ((addition.span, (right,)),), producers)
            all_ids = set(producers)
            for position_side, data_side, uncertain in (
                    (set(left_sources), right, left_uncertain),
                    (set(right_sources), left, right_uncertain)):
                if uncertain or position_side != all_ids:
                    continue
                if _parameter_origins(
                        index, callable_symbol, data_side,
                        addition.span, set()) & data_formals:
                    return addition.span
    return None


def _parameter_field_axis(expression):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return None
    base, selector = expression.children
    if base.kind != "attribute" or len(base.children) != 1 \
            or base.children[0].kind != "name" \
            or base.children[0].name != "self":
        return None
    axis = _constant_int(selector)
    return (base.name, axis) if axis is not None else None


def _last_constant_index(expression):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return None
    selector = expression.children[1]
    values = selector.children if selector.kind == "tuple" else (selector,)
    return _constant_int(values[-1]) if values else None


def _constant_int(expression):
    return (expression.const_value
            if expression.kind == "constant"
            and isinstance(expression.const_value, int)
            and not isinstance(expression.const_value, bool) else None)


def _parameter_origins(index, callable_symbol, expression, cutoff, seen):
    if expression is None or expression.span is None:
        return set()
    if expression.kind == "name" and expression.name:
        if expression.name in seen:
            return {expression.name}
        bindings = tuple(
            item for item in index.bindings_in(callable_symbol)
            if not item.guard and item.span is not None
            and _before(item.span, cutoff)
            and any(target.kind == "name" and target.name == expression.name
                    for target in item.targets))
        if not bindings:
            return {expression.name}
        binding = max(bindings, key=lambda item: _span_key(item.span))
        return _parameter_origins(
            index, callable_symbol, binding.value, binding.span,
            {*seen, expression.name})
    out = set()
    children = list(expression.children)
    children.extend(value for _name, value in expression.keyword_children)
    for child in children:
        if isinstance(child, ExprNode):
            out.update(_parameter_origins(
                index, callable_symbol, child, cutoff, seen))
    return out


def _additions(expression):
    out = []
    if expression.kind == "binop" and expression.operator == "+" \
            and len(expression.children) == 2:
        out.append(expression)
    for child in expression.children:
        if isinstance(child, ExprNode):
            out.extend(_additions(child))
    return tuple(out)


def _predecessors(edges, anchor):
    reached = set()
    frontier = [anchor]
    while frontier:
        target = frontier.pop()
        for edge in edges:
            if edge.target != target or edge.source in reached:
                continue
            reached.add(edge.source)
            frontier.append(edge.source)
    return reached


def _before(first, second):
    if first is None or second is None or first.source != second.source:
        return False
    return (first.end_line or first.line, first.end_col or first.col) <= \
        (second.line, second.col)


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line, span.end_col or span.col)


__all__ = [
    "ComponentLearnedPositionEvidence", "read_component_learned_position",
]
