"""U9-E — positive code-proven operation routes into a repeated stage.

The caller supplies the exact stage occurrence and the exact repeated-call
templates already established by U3/U9-D.  This reader does not decide that a
container is a layer stack and it does not call an operation a patch embed,
projector, or token reducer.  It only proves supported operations whose result
reaches one supplied repeated invocation through U3's local def-use edges.

The execution substrate is deliberately open.  Consequently this boundary can
prove positive local routes, but it never interprets a missing route as proof
that the stage has no frontend operation.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .container_inventory import resolve_container_inventory
from .execution_flow import (
    HappensBeforeEdge,
    InvocationNodeId,
    RepeatedInvocationTemplate,
    resolve_addressed_invocations,
    resolve_execution_flow,
)
from .models import SourceOp
from .program_index import (
    CallObservation, CallSiteId, ProgramIndex, SourceSpan, SymbolId,
)
from .projector_chain import (
    projector_call_operation_in_graph,
    projector_operation_chain_in_graph,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class StageOperationRoute:
    """One exact operation-bearing invocation reaching one repeated call."""

    stage_occurrence: OwnerOccurrenceId
    source_node: InvocationNodeId
    target: RepeatedInvocationTemplate
    call: CallObservation
    operations: tuple[SourceOp, ...]
    operation_spans: tuple[SourceSpan, ...]
    paths: tuple[tuple[HappensBeforeEdge, ...], ...]
    callee_occurrence: OwnerOccurrenceId | None = None

    def __post_init__(self):
        if not isinstance(self.stage_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.source_node, InvocationNodeId) \
                or not isinstance(self.target, RepeatedInvocationTemplate) \
                or not isinstance(self.call, CallObservation):
            raise TypeError("a stage route is exact-owner/call qualified")
        if self.target.caller_occurrence != self.stage_occurrence \
                or self.call.enclosing_callable \
                != self.target.call.enclosing_callable:
            raise ValueError("source and repeated target belong to one stage forward")
        if self.source_node.call_site != CallSiteId.of(self.call):
            raise ValueError("source node round-trips to the exact call")
        if self.source_node.kind not in {"addressed", "external"}:
            raise ValueError("a proven operation route starts at an addressed call")
        if self.callee_occurrence is not None:
            if self.source_node.kind != "addressed" \
                    or self.callee_occurrence.root != self.stage_occurrence.root:
                raise ValueError("an internal operation cites its exact graph child")
        elif self.source_node.kind != "external":
            raise ValueError("an external operation carries no fabricated child")
        if not self.operations or len(self.operations) != len(self.operation_spans):
            raise ValueError("an operation route carries one span per operation")
        if any(not isinstance(item, SourceOp) for item in self.operations) \
                or any(not isinstance(span, SourceSpan)
                       for span in self.operation_spans):
            raise TypeError("operation routes retain typed operations and spans")
        if not self.paths:
            raise ValueError("a route requires a positive local path")
        target_node = InvocationNodeId(self.target.call_site, "template")
        for path in self.paths:
            if not path or path[0].source != self.source_node \
                    or path[-1].target != target_node \
                    or any(left.target != right.source
                           for left, right in zip(path, path[1:])):
                raise ValueError("every carried path is contiguous source-to-target proof")


@dataclass(frozen=True)
class StageOperationInventory:
    """Positive routes for an explicitly supplied repeated-stage boundary."""

    stage_occurrence: OwnerOccurrenceId
    stage_symbol: SymbolId
    targets: tuple[RepeatedInvocationTemplate, ...]
    routes: tuple[StageOperationRoute, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.stage_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.stage_symbol, SymbolId):
            raise TypeError("a stage operation inventory is occurrence-qualified")
        if not self.targets or any(
                not isinstance(item, RepeatedInvocationTemplate)
                or item.caller_occurrence != self.stage_occurrence
                for item in self.targets):
            raise ValueError("the inventory retains every supplied exact target")
        if not self.routes or any(
                not isinstance(item, StageOperationRoute)
                or item.stage_occurrence != self.stage_occurrence
                or item.target not in self.targets
                for item in self.routes):
            raise ValueError("the inventory contains positive routes to supplied targets")
        required = {
            *(item.call.span for item in self.routes),
            *(span for item in self.routes for span in item.operation_spans),
            *(span for item in self.routes for path in item.paths
              for edge in path for span in edge.supporting_spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("inventory provenance closes every positive route")


def stage_operation_inventory_at_owner(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    stage: OwnerOccurrenceId,
    targets: tuple[RepeatedInvocationTemplate, ...],
) -> ReaderResult[StageOperationInventory]:
    """Prove supported operation calls that feed exact repeated invocations."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("stage operation reading requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="stage_operation_inventory_at_owner")
    if not isinstance(stage, OwnerOccurrenceId) or not isinstance(targets, tuple):
        raise TypeError("stage operations require an exact stage and target tuple")
    if not targets or any(
            not isinstance(item, RepeatedInvocationTemplate)
            or item.caller_occurrence != stage for item in targets):
        raise ValueError("stage operation targets belong to the supplied stage")
    node = root.graph.node_for(stage)
    if node is None:
        return ReaderResult.failed(stage, (ReaderFailure(
            "out_of_owner", "the stage is absent from its component graph"),))

    containers = resolve_container_inventory(index, root, stage)
    invocations = resolve_addressed_invocations(index, root, stage, containers)
    flow = resolve_execution_flow(index, root, stage, containers)
    if invocations.status != "resolved" or flow.status != "partial":
        return ReaderResult.failed(stage, (ReaderFailure(
            "incomplete_graph", "the stage's local invocation graph is unavailable"),))
    carried = {item.call_site: item for item in invocations.templates}
    if any(item.call_site not in carried or carried[item.call_site] != item
           for item in targets):
        return ReaderResult.failed(stage, (ReaderFailure(
            "out_of_owner", "a supplied repeated target does not round-trip"),))

    edges = (*flow.proven_edges, *flow.conditional_edges)
    routes = []
    blockers = []
    sources = (
        *((item, "addressed") for item in invocations.addressed),
        *((item, "external") for item in invocations.external_addressed),
    )
    for invocation, kind in sources:
        source_node = InvocationNodeId(invocation.call_site, kind)
        for target in targets:
            target_node = InvocationNodeId(target.call_site, "template")
            paths = _paths(source_node, target_node, edges)
            if not paths:
                continue
            if kind == "addressed":
                operations, spans, failure = projector_operation_chain_in_graph(
                    index, root.graph, invocation.callee_owner_occurrence)
                callee = invocation.callee_owner_occurrence
            else:
                item = projector_call_operation_in_graph(
                    index, root.graph, stage, invocation.call)
                if item is None:
                    operations, spans, failure = (), (), None
                else:
                    operations, spans, failure = item
                callee = None
            if failure is not None:
                blockers.append(failure)
            if not operations:
                continue
            routes.append(StageOperationRoute(
                stage, source_node, target, invocation.call,
                tuple(operations), tuple(spans), paths, callee))

    for unresolved in invocations.unresolved:
        source = InvocationNodeId(unresolved.call_site, "observed")
        if any(_paths(
                source, InvocationNodeId(target.call_site, "template"), edges)
               for target in targets):
            blockers.append(ReaderFailure(
                "incomplete_graph",
                f"an unresolved invocation may feed the repeated stage: "
                f"{unresolved.reason}", unresolved.call.span))

    if not routes:
        return ReaderResult.failed(stage, tuple(blockers) or (ReaderFailure(
            "incomplete_graph",
            "no supported operation has a positive path to the repeated stage"),))
    spans = tuple(dict.fromkeys(
        span for route in routes for span in (
            route.call.span,
            *route.operation_spans,
            *(item for path in route.paths for edge in path
              for item in edge.supporting_spans),
        ) if isinstance(span, SourceSpan)))
    value = StageOperationInventory(
        stage, node.symbol, targets, tuple(routes), spans)
    provenance = (ReaderProvenance(
        "source", spans=spans,
        detail="positive exact operation routes into repeated invocations"),)
    if blockers:
        return ReaderResult.incomplete(
            stage, value, failures=tuple(blockers), provenance=provenance)
    return ReaderResult.resolved(stage, value, provenance=provenance)


def _paths(start, end, edges):
    adjacency = {}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge)
    out = []

    def visit(node, path, seen):
        if node == end:
            out.append(tuple(path))
            return
        for edge in adjacency.get(node, ()):
            if edge.target in seen:
                continue
            visit(edge.target, (*path, edge), {*seen, edge.target})

    visit(start, (), {start})
    return tuple(out)


__all__ = [
    "StageOperationRoute", "StageOperationInventory",
    "stage_operation_inventory_at_owner",
]
