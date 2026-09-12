"""U9-D/E — exact stage-boundary and repeated-block operation projections.

These readers start from owner occurrences that U3/U9 already proved.  They
never discover a component, infer a role from a field/class name, or claim a
complete control-flow graph.  They answer two deliberately smaller questions:

* which registered operations occur before/after the exact repeated-call loop;
* which registered operations and U6/U7 mechanism calls occur in one exact
  repeated block, in source order.

The result is positive evidence only.  An unclassified call remains visible as
an incomplete reason; absence of an operation is never a negative proof.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .attention_child import AttentionChildEvidence
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .construction_calls import resolve_construction_call_in_graph
from .container_inventory import resolve_container_inventory
from .execution_flow import RepeatedInvocationTemplate
from .ffn_mechanism import OrdinaryFFNPositiveCensus
from .models import SourceOp
from .program_index import (
    CallObservation,
    LoopObservation,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .projector_chain import projector_call_operation_in_graph
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class StageBoundaryOperations:
    """Positive operations around one exact repeated-call loop."""

    stage_occurrence: OwnerOccurrenceId
    stage_symbol: SymbolId
    callable_symbol: SymbolId
    repeated_calls: tuple[CallObservation, ...]
    repeated_loop: LoopObservation
    frontend: tuple[SourceOp, ...]
    post: tuple[SourceOp, ...]
    operation_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.stage_symbol, SymbolId) \
                or not isinstance(self.callable_symbol, SymbolId):
            raise TypeError("stage-boundary operations are exact-owner qualified")
        if not self.repeated_calls or any(
                not isinstance(call, CallObservation)
                or call.enclosing_callable != self.callable_symbol
                or not _contains(self.repeated_loop.body_span, call.span)
                for call in self.repeated_calls):
            raise ValueError("the exact repeated calls lie inside one loop body")
        if not isinstance(self.repeated_loop, LoopObservation) \
                or self.repeated_loop.enclosing_callable != self.callable_symbol:
            raise ValueError("the repeated loop belongs to the stage callable")
        operations = (*self.frontend, *self.post)
        if not operations or any(not isinstance(item, SourceOp)
                                 for item in operations):
            raise ValueError("a boundary result carries positive operations")
        if len(self.operation_spans) != len(operations) \
                or any(not isinstance(span, SourceSpan)
                       for span in self.operation_spans):
            raise ValueError("every projected operation retains its exact span")
        if any(span.source != self.stage_symbol.source
               for span in self.operation_spans):
            raise ValueError("boundary operation spans stay in the component")


@dataclass(frozen=True)
class BlockOperationInventory:
    """Positive source-ordered operations for one exact repeated block."""

    block_occurrence: OwnerOccurrenceId
    block_symbol: SymbolId
    callable_symbol: SymbolId
    operations: tuple[SourceOp, ...]
    operation_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.block_symbol, SymbolId) \
                or not isinstance(self.callable_symbol, SymbolId):
            raise TypeError("block operations are exact-owner qualified")
        if not self.operations or len(self.operations) != len(self.operation_spans):
            raise ValueError("block operations carry one exact span per operation")
        if any(not isinstance(item, SourceOp) for item in self.operations) \
                or any(not isinstance(span, SourceSpan)
                       for span in self.operation_spans):
            raise TypeError("block operations retain typed operations and spans")
        if any(span.source != self.block_symbol.source
               for span in self.operation_spans):
            raise ValueError("block operation spans stay in the exact component")


@dataclass(frozen=True)
class ComponentBoundaryOperations:
    """Registered operations around the exact root-to-stage invocation route."""

    component_occurrence: OwnerOccurrenceId
    stage_occurrence: OwnerOccurrenceId
    route_calls: tuple[CallObservation, ...]
    frontend: tuple[SourceOp, ...]
    post: tuple[SourceOp, ...]
    operation_spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not self.route_calls or self.component_occurrence == self.stage_occurrence:
            raise ValueError("component boundaries retain a non-empty stage route")
        if any(not isinstance(call, CallObservation) or call.span is None
               for call in self.route_calls):
            raise TypeError("component boundaries retain exact route calls")
        operations = (*self.frontend, *self.post)
        if not operations or len(operations) != len(self.operation_spans):
            raise ValueError("component boundary operations retain exact spans")


def read_component_boundary_operations(index, root, stage):
    """Read positive operations before/after an exact nested stage route."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("component-boundary reading requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="read_component_boundary_operations")
    component = root.graph.root.occurrence
    if stage == component:
        return ReaderResult.absent(stage)
    if root.graph.node_for(stage) is None:
        return ReaderResult.failed(stage, (ReaderFailure(
            "out_of_owner", "stage is absent from the component graph"),))

    from .execution_flow import resolve_addressed_invocations
    current = component
    route = []
    boundaries = []
    seen = set()
    while current != stage:
        if current in seen:
            return ReaderResult.failed(stage, (ReaderFailure(
                "incomplete_graph", "component-to-stage route cycles"),))
        seen.add(current)
        inventory = resolve_container_inventory(index, root, current)
        invocations = resolve_addressed_invocations(index, root, current, inventory)
        if invocations.status != "resolved":
            return ReaderResult.failed(stage, (ReaderFailure(
                "incomplete_graph", "component invocation census unavailable"),))
        matches = tuple(
            item for item in invocations.addressed
            if _occurrence_prefix(item.callee_owner_occurrence, stage))
        if len(matches) != 1:
            sites = tuple(item.call.span for item in matches)
            if len(matches) > 1:
                return ReaderResult.ambiguous(stage, Ambiguity(sites=sites))
            return ReaderResult.failed(stage, (ReaderFailure(
                "incomplete_graph", "no exact invoked route reaches the stage"),))
        invocation = matches[0]
        route.append(invocation.call)
        boundaries.append(_operations_around_call(
            index, root, current, inventory, invocation.call))
        current = invocation.callee_owner_occurrence

    front_pairs = [pair for front, _post, _failures in boundaries for pair in front]
    post_pairs = [pair for _front, post, _failures in reversed(boundaries)
                  for pair in post]
    failures = tuple(problem for _front, _post, problems in boundaries
                     for problem in problems)
    pairs = (*front_pairs, *post_pairs)
    if not pairs:
        return ReaderResult.failed(stage, failures or (ReaderFailure(
            "incomplete_graph", "no registered operation surrounds the stage"),))
    operations = _sequence(tuple(op for _span, op in pairs), "component")
    spans = tuple(span for span, _op in pairs)
    value = ComponentBoundaryOperations(
        component, stage, tuple(route),
        operations[:len(front_pairs)], operations[len(front_pairs):], spans)
    provenance = (ReaderProvenance(
        "source", spans=tuple(dict.fromkeys((
            *(call.span for call in route), *spans))),
        detail="registered operations around exact component-to-stage route"),)
    if failures:
        return ReaderResult.incomplete(
            stage, value, failures=failures, provenance=provenance)
    return ReaderResult.resolved(stage, value, provenance=provenance)


def _operations_around_call(index, root, owner, inventory, anchor):
    node = root.graph.node_for(owner)
    if node is None:
        return (), (), (ReaderFailure(
            "out_of_owner", "route caller is absent from the graph"),)
    from .execution_flow import resolve_execution_flow
    from .program_index import CallSiteId

    flow = resolve_execution_flow(index, root, owner, inventory)
    if flow.status != "partial":
        return (), (), (ReaderFailure(
            "incomplete_graph", "component execution flow is unavailable"),)
    anchor_node = next((
        item for item in flow.nodes if item.call_site == CallSiteId.of(anchor)), None)
    if anchor_node is None:
        return (), (), (ReaderFailure(
            "incomplete_graph", "the stage invocation has no flow node"),)
    edges = (*flow.proven_edges, *flow.conditional_edges)
    predecessors, front_failure = _linear_primary_predecessors(
        edges, anchor_node, anchor)
    successors, post_failure = _linear_successors(edges, anchor_node)
    callable_symbol = anchor.enclosing_callable
    call_by_site = {
        CallSiteId.of(call): call for call in index.calls_in(callable_symbol)
        if call.span is not None
    }
    front, post = [], []
    failures = [
        failure for failure in (front_failure, post_failure)
        if failure is not None
    ]
    for flow_node in sorted(
            (*predecessors, *successors),
            key=lambda item: _span_key(item.call_site.span)):
        call = call_by_site.get(flow_node.call_site)
        if call is None:
            continue
        side = front if flow_node in predecessors else post
        classified = projector_call_operation_in_graph(
            index, root.graph, owner, call)
        if classified is None:
            continue
        operations, spans, failure = classified
        if failure is not None:
            failures.append(failure)
        side.extend(zip(spans, operations))
    return tuple(front), tuple(post), tuple(failures)


def _linear_primary_predecessors(edges, anchor_node, anchor):
    """Return the one proven primary-tensor chain entering ``anchor``.

    A component call commonly consumes architectural side inputs beside its
    tensor stream (Qwen2-VL passes rotary positions into each visual block).
    The former implementation took the transitive union of *all* predecessors
    and then serialized that union as a linear operation list.  Independent
    branches consequently became fictional chains, and a real two-input concat
    could be drawn with only one input.

    The first positional argument is the established primary tensor convention
    for this boundary.  Its exact consumer-use span selects the direct producer;
    a branch/merge anywhere in that ancestry is not representable by the flat
    ``SourceOp`` boundary DTO and therefore remains typed-incomplete rather than
    being linearized.
    """
    primary = anchor.args[0] if anchor.args else None
    if primary is None or primary.span is None:
        return (), ReaderFailure(
            "incomplete_graph",
            "the stage invocation has no exact primary-tensor argument span",
            anchor.span)
    direct_edges = tuple(
        edge for edge in edges
        if edge.target == anchor_node and edge.supporting_spans
        and _contains(primary.span, edge.supporting_spans[-1]))
    if not direct_edges:
        return (), None
    direct = {edge.source for edge in direct_edges}
    selected = set(direct)
    for node in tuple(direct):
        selected.update(_reachable_nodes(edges, node, reverse=True))
    if not _is_linear_relation(
            edges, selected, anchor_node, boundary_edges=direct_edges):
        return (), ReaderFailure(
            "incomplete_graph",
            "the primary-tensor route branches or merges and cannot be "
            "projected as a flat operation chain",
            anchor.span)
    return tuple(sorted(selected, key=lambda item: _span_key(item.call_site.span))), None


def _linear_successors(edges, anchor_node):
    """Return one non-branching output chain, or a typed incompleteness."""
    direct_edges = tuple(edge for edge in edges if edge.source == anchor_node)
    if not direct_edges:
        return (), None
    direct = {edge.target for edge in direct_edges}
    selected = set(direct)
    for node in tuple(direct):
        selected.update(_reachable_nodes(edges, node, reverse=False))
    if not _is_linear_relation(
            edges, selected, anchor_node, boundary_edges=direct_edges,
            reverse=True):
        return (), ReaderFailure(
            "incomplete_graph",
            "the stage-output route branches or merges and cannot be "
            "projected as a flat operation chain",
            anchor_node.call_site.span)
    return tuple(sorted(selected, key=lambda item: _span_key(item.call_site.span))), None


def _is_linear_relation(
        edges, selected, boundary, *, boundary_edges, reverse=False):
    """Whether ``selected`` plus its boundary edge is exactly one chain."""
    if len(boundary_edges) != 1 or not selected:
        return False
    relation_edges = [
        edge for edge in edges
        if edge.source in selected and edge.target in selected
    ]
    incoming = {node: 0 for node in selected}
    outgoing = {node: 0 for node in selected}
    for edge in relation_edges:
        outgoing[edge.source] += 1
        incoming[edge.target] += 1
    for edge in boundary_edges:
        if reverse:
            outgoing[boundary] = outgoing.get(boundary, 0) + 1
            incoming[edge.target] += 1
        else:
            outgoing[edge.source] += 1
            incoming[boundary] = incoming.get(boundary, 0) + 1
    # Every retained invocation has at most one predecessor and successor.
    # The boundary itself is included only to make a multi-edge fan-in/fan-out
    # fail the same predicate; it is never returned as a projected operation.
    return all(value <= 1 for value in incoming.values()) \
        and all(value <= 1 for value in outgoing.values())


def _reachable_nodes(edges, anchor, *, reverse):
    """Transitive positive local relations into/out of one exact invocation."""
    reached = set()
    frontier = [anchor]
    while frontier:
        current = frontier.pop()
        for edge in edges:
            source, target = ((edge.target, edge.source)
                              if reverse else (edge.source, edge.target))
            if source != current or target == anchor or target in reached:
                continue
            reached.add(target)
            frontier.append(target)
    return reached


def read_stage_boundary_operations(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    stage: OwnerOccurrenceId,
    targets: tuple[RepeatedInvocationTemplate, ...],
) -> ReaderResult[StageBoundaryOperations]:
    """Project registered calls before/after the exact repeated-call loop."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("stage-boundary reading requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="read_stage_boundary_operations")
    node = root.graph.node_for(stage)
    if node is None:
        return ReaderResult.failed(stage, (ReaderFailure(
            "out_of_owner", "the stage is absent from its component graph"),))
    if not targets or any(
            not isinstance(item, RepeatedInvocationTemplate)
            or item.caller_occurrence != stage for item in targets):
        raise ValueError("boundary targets belong to the exact stage")
    repeated_calls = tuple(dict.fromkeys(item.call for item in targets))
    callables = {item.enclosing_callable for item in repeated_calls}
    if len(callables) != 1:
        return ReaderResult.failed(stage, (ReaderFailure(
            "conflict", "repeated calls do not share one stage callable"),))
    callable_symbol = next(iter(callables))
    if callable_symbol != SymbolId(
            node.symbol.source, f"{node.symbol.qualified_name}.forward"):
        return ReaderResult.failed(stage, (ReaderFailure(
            "out_of_owner", "repeated calls are not in the exact stage forward"),))

    loops = tuple(
        loop for loop in index.loops_in(callable_symbol)
        if all(_contains(loop.body_span, call.span) for call in repeated_calls))
    loop = _unique_innermost_loop(loops)
    if loop is None:
        return ReaderResult.failed(stage, (ReaderFailure(
            "conflict" if loops else "incomplete_graph",
            "the repeated calls do not identify one exact innermost loop"),))

    # The execution-flow substrate proves which calls actually feed the exact
    # repeated invocation.  Lexical position alone is insufficient: visual
    # towers routinely compute masks/rotary positions beside the hidden-state
    # route before entering the loop.  Project only a consensus, non-branching
    # primary-tensor chain; never serialize the union as if it were sequential.
    from .execution_flow import resolve_execution_flow
    from .program_index import CallSiteId

    inventory = resolve_container_inventory(index, root, stage)
    flow = resolve_execution_flow(index, root, stage, inventory)
    if flow.status != "partial":
        return ReaderResult.failed(stage, (ReaderFailure(
            "incomplete_graph", "stage execution flow is unavailable"),))
    node_by_site = {item.call_site: item for item in flow.nodes}
    anchors = tuple(node_by_site.get(CallSiteId.of(call))
                    for call in repeated_calls)
    if any(item is None for item in anchors):
        return ReaderResult.failed(stage, (ReaderFailure(
            "incomplete_graph", "a repeated call has no execution-flow node"),))
    edges = (*flow.proven_edges, *flow.conditional_edges)
    front_routes = []
    post_routes = []
    blockers: list[ReaderFailure] = []
    for anchor_node, repeated_call in zip(anchors, repeated_calls):
        front, front_failure = _linear_primary_predecessors(
            edges, anchor_node, repeated_call)
        post_route, post_failure = _linear_successors(edges, anchor_node)
        front_routes.append(front)
        post_routes.append(post_route)
        blockers.extend(item for item in (front_failure, post_failure)
                        if item is not None)
    if len(set(front_routes)) != 1:
        blockers.append(ReaderFailure(
            "conflict",
            "repeated-call variants do not share one exact primary frontend",
            loop.span))
        front_nodes = ()
    else:
        front_nodes = front_routes[0]
    if len(set(post_routes)) != 1:
        blockers.append(ReaderFailure(
            "conflict",
            "repeated-call variants do not share one exact output route",
            loop.span))
        post_nodes = ()
    else:
        post_nodes = post_routes[0]

    call_by_site = {
        CallSiteId.of(call): call for call in index.calls_in(callable_symbol)
        if call.span is not None
    }
    frontend: list[SourceOp] = []
    front_spans: list[SourceSpan] = []
    post: list[SourceOp] = []
    post_spans: list[SourceSpan] = []
    for side, nodes in (("front", front_nodes), ("post", post_nodes)):
        for flow_node in nodes:
            call = call_by_site.get(flow_node.call_site)
            if call is None:
                continue
            classified = projector_call_operation_in_graph(
                index, root.graph, stage, call)
            if classified is None:
                continue
            operations, spans, failure = classified
            if failure is not None:
                blockers.append(failure)
            if side == "front":
                frontend.extend(operations)
                front_spans.extend(spans)
            else:
                post.extend(operations)
                post_spans.extend(spans)
    if not frontend and not post:
        return ReaderResult.failed(stage, tuple(blockers) or (ReaderFailure(
            "incomplete_graph", "no registered boundary operation is proven"),))
    frontend = list(_sequence(frontend, "front"))
    post = list(_sequence(post, "post"))
    spans = tuple((*front_spans, *post_spans))
    value = StageBoundaryOperations(
        stage, node.symbol, callable_symbol, repeated_calls, loop,
        tuple(frontend), tuple(post), spans)
    provenance = (ReaderProvenance(
        "source", spans=tuple(dict.fromkeys((loop.span, *spans))),
        detail="registered operations around one exact repeated-call loop"),)
    if blockers:
        return ReaderResult.incomplete(
            stage, value, failures=tuple(blockers), provenance=provenance)
    return ReaderResult.resolved(stage, value, provenance=provenance)


def read_block_operations(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block: OwnerOccurrenceId,
    attention: ReaderResult[AttentionChildEvidence],
    ffn_census: ReaderResult[OrdinaryFFNPositiveCensus],
) -> ReaderResult[BlockOperationInventory]:
    """Project exact U6/U7 mechanisms and registered calls in source order."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("block-operation reading requires a ProgramIndex")
    root = require_resolved_component_root(root, caller="read_block_operations")
    node = root.graph.node_for(block)
    if node is None:
        return ReaderResult.failed(block, (ReaderFailure(
            "out_of_owner", "the block is absent from its component graph"),))
    callable_symbol = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.forward")
    if index.callable_by_symbol(callable_symbol) is None:
        return ReaderResult.absent(block)

    overrides: dict[SourceSpan, tuple[int, SourceOp]] = {}
    suppressed: set[SourceSpan] = set()
    if attention.status == "resolved":
        evidence = attention.value
        call = evidence.invocation.call
        overrides[call.span] = (0, SourceOp(
            "attention", "Self-attention", evidence.compute_owner_symbol.qualified_name,
            evidence.compute_owner_symbol.source.canonical_path, call.span.line))

    ffn_values = (
        ffn_census.value.candidates
        if ffn_census.status in {"resolved", "incomplete"}
        and ffn_census.value is not None else ())
    calls = tuple(sorted(index.calls_in(callable_symbol), key=_call_key))
    for number, mechanism in enumerate(ffn_values, 1):
        matched: set[SourceSpan] = {
            item.call.span for item in mechanism.invocations
            if item.caller_occurrence == block
        }
        projections = set(mechanism.projections)
        for call in calls:
            if _self_field(call) is None:
                continue
            resolution = resolve_construction_call_in_graph(
                index, root.graph, block, call)
            if resolution.status == "resolved" \
                    and resolution.selected.occurrence in projections:
                matched.add(call.span)
        if not matched:
            continue
        anchor = min(matched, key=_span_key)
        label = "Feed-forward" if len(ffn_values) == 1 \
            else f"Feed-forward {number}"
        overrides[anchor] = (number, SourceOp(
            "ffn", label, mechanism.owner_symbol.qualified_name,
            mechanism.owner_symbol.source.canonical_path, anchor.line))
        suppressed.update(matched - {anchor})

    projected: list[tuple[SourceSpan, SourceOp]] = []
    blockers: list[ReaderFailure] = []
    for call in calls:
        if call.span in suppressed:
            continue
        override = overrides.get(call.span)
        if override is not None:
            projected.append((call.span, override[1]))
            continue
        classified = projector_call_operation_in_graph(
            index, root.graph, block, call)
        if classified is None:
            continue
        operations, spans, failure = classified
        if failure is not None:
            blockers.append(failure)
        projected.extend(zip(spans, operations))
    if not projected:
        return ReaderResult.failed(block, tuple(blockers) or (ReaderFailure(
            "incomplete_graph", "no registered block operation is proven"),))
    projected.sort(key=lambda item: _span_key(item[0]))
    spans = tuple(item[0] for item in projected)
    operations = _sequence(tuple(item[1] for item in projected), "block")
    value = BlockOperationInventory(
        block, node.symbol, callable_symbol, operations, spans)
    provenance = (ReaderProvenance(
        "source", spans=tuple(dict.fromkeys(spans)),
        detail="source-ordered exact block operations"),)
    if blockers:
        return ReaderResult.incomplete(
            block, value, failures=tuple(blockers), provenance=provenance)
    return ReaderResult.resolved(block, value, provenance=provenance)


def _unique_innermost_loop(loops):
    if not loops:
        return None
    innermost = tuple(
        item for item in loops
        if not any(item != other and _contains(item.body_span, other.body_span)
                   for other in loops))
    return innermost[0] if len(innermost) == 1 else None


def _sequence(operations, prefix):
    result = []
    previous = ""
    for index, operation in enumerate(operations):
        op_id = f"{prefix}_{index}"
        result.append(replace(
            operation, op_id=op_id,
            inputs=(previous,) if previous else ()))
        previous = op_id
    return tuple(result)


def _contains(outer, inner):
    if not isinstance(outer, SourceSpan) or not isinstance(inner, SourceSpan) \
            or outer.source != inner.source:
        return False
    return (outer.line, outer.col) <= (inner.line, inner.col) \
        and (inner.end_line, inner.end_col) <= (outer.end_line, outer.end_col)


def _before(left, right):
    return _span_key(left) < _span_key(right)


def _span_key(span):
    return span.line, span.col, span.end_line, span.end_col


def _call_key(call):
    return _span_key(call.span)


def _self_field(call):
    callee = getattr(call, "callee", None)
    if callee is None or callee.kind != "attribute" \
            or len(callee.children) != 1:
        return None
    receiver = callee.children[0]
    return callee.name if receiver.kind == "name" \
        and receiver.name == "self" else None


def _occurrence_prefix(parent, child):
    return parent.root == child.root \
        and child.sites[:len(parent.sites)] == parent.sites


__all__ = [
    "StageBoundaryOperations", "BlockOperationInventory",
    "ComponentBoundaryOperations", "read_stage_boundary_operations",
    "read_component_boundary_operations", "read_block_operations",
]
