"""U11-C — partial, exact U-shaped stage-execution evidence.

U10 proves the outer repeated calls and skip dataflow. U11-B proves how those
exact repeated containers are populated. This boundary adds only one further
positive fact: exact root-owned constructed fields that are invoked textually
between the two U10 repeated sides.

The interval is an address filter, not a whole-CFG ordering proof. Every result
therefore remains partial; direct calls retain their guards, U10's skip route is
the only proven inter-stage dataflow edge, and all other relations remain typed
unresolved. Field/class/factory spellings never assign a mid/bookend/attention
role. A later projection may call a direct node "bottleneck" only because its
exact invocation occupies this proven U-shaped interval, never because its
field happened to be named ``mid_block``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .diffusion_root import DiffusionRootTopology, RepeatedRootStage, SkipRoute
from .execution_flow import execution_taint_reason
from .models import SourceBundle
from .program_index import CallObservation, ExprNode, ProgramIndex, SourceSpan, SymbolId
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_stage_construction import (
    DirectFieldConstructionInventory,
    DirectFieldInvocationAddress,
    RepeatedStageConstruction,
    UNetStageConstructionInventory,
    UnresolvedStageConstruction,
    read_direct_field_construction,
)


UNRESOLVED_RELATION_KINDS = frozenset({
    "whole_callable_open",
    "direct_stage_order_open",
    "direct_construction_incomplete",
    "direct_call_tainted",
})


def _self_field(expr: ExprNode | None) -> str | None:
    if expr is None or expr.kind != "attribute" or not expr.children:
        return None
    root = expr.children[0]
    return (expr.name if root.kind == "name" and root.name == "self"
            else None)


def _point(span: SourceSpan) -> tuple[int, int]:
    return span.line, span.col


def _end(span: SourceSpan) -> tuple[int, int]:
    return span.end_line or span.line, span.end_col or span.col


@dataclass(frozen=True, order=True)
class UNetStageNodeId:
    """Occurrence identity at the U11-C altitude; ``kind`` is neutral syntax."""

    source_position: tuple[int, int]
    kind: str                   # repeated | direct
    field: str
    owner: OwnerOccurrenceId

    def __post_init__(self) -> None:
        if len(self.source_position) != 2 \
                or any(not isinstance(item, int) or item < 0
                       for item in self.source_position) \
                or self.kind not in {"repeated", "direct"}:
            raise ValueError("a stage node has source order + neutral kind")
        if not self.field or not isinstance(self.owner, OwnerOccurrenceId):
            raise ValueError("a stage node retains exact owner + field address")


@dataclass(frozen=True)
class RepeatedStageExecution:
    node_id: UNetStageNodeId
    topology_stage: RepeatedRootStage
    constructions: tuple[RepeatedStageConstruction, ...]
    unresolved_construction: UnresolvedStageConstruction | None = None

    def __post_init__(self) -> None:
        if self.node_id.kind != "repeated" \
                or self.node_id.owner != self.topology_stage.owner \
                or self.node_id.field != self.topology_stage.field:
            raise ValueError("a repeated node round-trips to exact U10 evidence")
        if self.constructions and self.unresolved_construction is not None:
            raise ValueError("construction evidence is resolved templates xor unresolved")
        if not self.constructions and self.unresolved_construction is None:
            raise ValueError("a repeated execution retains construction disposition")
        if any(item.owner != self.node_id.owner
               or item.topology_stage != self.topology_stage
               for item in self.constructions):
            raise ValueError("repeated construction templates belong to this node")
        if self.unresolved_construction is not None \
                and (self.unresolved_construction.owner != self.node_id.owner
                     or self.unresolved_construction.topology_stage !=
                     self.topology_stage):
            raise ValueError("unresolved construction belongs to this node")


@dataclass(frozen=True)
class DirectStageExecution:
    """A constructed root field invoked inside the U10 inter-loop interval.

    ``direct`` means direct ``self.<field>(...)`` syntax, not a semantic role.
    """

    node_id: UNetStageNodeId
    call: CallObservation
    construction: DirectFieldConstructionInventory

    def __post_init__(self) -> None:
        if self.node_id.kind != "direct" \
                or not isinstance(self.call, CallObservation) \
                or self.call.span is None:
            raise ValueError("a direct node carries one exact direct call")
        if self.construction.owner != self.node_id.owner \
                or self.construction.field != self.node_id.field:
            raise ValueError("direct execution and construction share owner + field")
        if _self_field(self.call.callee) != self.node_id.field \
                or self.node_id.source_position != _point(self.call.span) \
                or self.call not in self.construction.address.calls:
            raise ValueError("the node identity round-trips to its exact call site")


@dataclass(frozen=True)
class StageDataflowEdge:
    """A positive inter-node dataflow proof; never inferred from source order."""

    source: RepeatedStageExecution
    target: RepeatedStageExecution
    proof_kind: str             # u10_skip_route
    route: SkipRoute

    def __post_init__(self) -> None:
        if self.proof_kind != "u10_skip_route" \
                or self.source.node_id.owner != self.target.node_id.owner:
            raise ValueError("an inter-stage edge is the exact U10 skip proof")
        if not isinstance(self.route, SkipRoute) \
                or self.route.producer not in self.source.topology_stage.calls \
                or self.route.consumer not in self.target.topology_stage.calls:
            raise ValueError("skip endpoints require graph-bound stage membership")


@dataclass(frozen=True)
class UnresolvedStageRelation:
    kind: str
    nodes: tuple[UNetStageNodeId, ...]
    detail: str
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in UNRESOLVED_RELATION_KINDS or not self.detail:
            raise ValueError("an unresolved relation has closed kind + detail")
        if not self.nodes or any(not isinstance(node, UNetStageNodeId)
                                 for node in self.nodes):
            raise TypeError("unresolved relations cite exact nodes")
        if len({node.owner for node in self.nodes}) != 1:
            raise ValueError("unresolved relation nodes share one owner")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("unresolved relation spans are typed")


@dataclass(frozen=True)
class UNetStageExecutionGraph:
    """Positive nodes + U10 dataflow, with all non-proven relations unresolved."""

    owner: OwnerOccurrenceId
    topology: DiffusionRootTopology
    construction: UNetStageConstructionInventory
    repeated: tuple[RepeatedStageExecution, ...]
    direct: tuple[DirectStageExecution, ...]
    edges: tuple[StageDataflowEdge, ...]
    unresolved: tuple[UnresolvedStageRelation, ...]
    index: ProgramIndex

    def __post_init__(self) -> None:
        if self.topology.owner != self.owner or self.construction.owner != self.owner:
            raise ValueError("execution graph consumes exact U10+B owner evidence")
        if not set(self.construction.index.source_nodes) <= set(self.index.source_nodes):
            raise ValueError("the graph index contains the U11-B evidence universe")
        if len(self.repeated) != len(self.topology.stages):
            raise ValueError("every U10 repeated stage has one execution node")
        if tuple(item.topology_stage for item in self.repeated) != \
                self.topology.stages:
            raise ValueError("repeated nodes retain exact U10 topology order")
        nodes = tuple(item.node_id for item in (*self.repeated, *self.direct))
        if len(set(nodes)) != len(nodes) or any(node.owner != self.owner for node in nodes):
            raise ValueError("stage node identities are unique and owner-qualified")
        for item in self.direct:
            if item.call not in self.index.calls_in(item.call.enclosing_callable):
                raise ValueError("every direct invocation belongs to the carried index")
            if not set(item.construction.index.source_nodes) <= set(
                    self.index.source_nodes):
                raise ValueError("the graph index contains every direct construction source")
            for construction in item.construction.constructions:
                if construction.producer_call not in self.index.calls_in(
                        construction.producer_call.enclosing_callable):
                    raise ValueError("the final index contains each direct constructor call")
                if any(self.index.class_by_symbol(candidate.symbol) is None
                       for candidate in construction.candidates):
                    raise ValueError("the final index contains each direct candidate class")
            if item.construction.address.earlier_stage != self.topology.stages[0] \
                    or item.construction.address.later_stage != \
                    self.topology.stages[-1]:
                raise ValueError("direct nodes retain the exact U10 interval")
        repeated_by_id = {item.node_id: item for item in self.repeated}
        for edge in self.edges:
            if edge.source.node_id not in repeated_by_id \
                    or edge.target.node_id not in repeated_by_id:
                raise ValueError("edge endpoints belong to repeated graph nodes")
            if repeated_by_id[edge.source.node_id] != edge.source \
                    or repeated_by_id[edge.target.node_id] != edge.target:
                raise ValueError("edge endpoints are the authoritative graph nodes")
            if edge.route.producer not in edge.source.topology_stage.calls \
                    or edge.route.consumer not in edge.target.topology_stage.calls:
                raise ValueError("edge endpoints round-trip to the exact U10 calls")
        if len(self.edges) != 1 or self.edges[0].route != self.topology.skip_route:
            raise ValueError("the graph carries exactly the one positive U10 skip edge")
        known_nodes = set(nodes)
        if any(not set(item.nodes) <= known_nodes for item in self.unresolved):
            raise ValueError("unresolved relations cite only authoritative graph nodes")
        forward_source = self.topology.stages[0].loop.enclosing_callable.source
        if any(span.source != forward_source for item in self.unresolved
               for span in item.spans):
            raise ValueError("unresolved relation spans belong to the root forward")
        if sum(item.kind == "whole_callable_open"
               for item in self.unresolved) != 1:
            raise ValueError("the partial graph carries one open-CFG disposition")
        if not self.unresolved:
            raise ValueError("open CFG coverage is never vacuously complete")


def _construction_rows(construction: UNetStageConstructionInventory, order: int):
    rows = tuple(item for item in construction.stages
                 if item.topology_order == order)
    unresolved = tuple(item for item in construction.unresolved_stages
                       if item.topology_order == order)
    if rows:
        return rows, None
    return (), unresolved[0]


def _direct_calls_between(index: ProgramIndex, callable_symbol: SymbolId,
                          first: RepeatedRootStage,
                          last: RepeatedRootStage):
    if first.loop.span is None or last.loop.span is None:
        return ()
    lower, upper = _end(first.loop.span), _point(last.loop.span)
    calls = tuple(call for call in index.calls_in(callable_symbol)
                  if call.span is not None and lower < _point(call.span) < upper
                  and _self_field(call.callee) is not None)
    return tuple(sorted(calls, key=lambda item: _point(item.span)))


def read_unet_stage_execution(
        construction: UNetStageConstructionInventory,
        bundle: SourceBundle,
        root_resolution: ComponentRootResolution,
        ) -> ReaderResult[UNetStageExecutionGraph]:
    """Build a partial exact stage graph without interpreting field names."""
    if not isinstance(construction, UNetStageConstructionInventory) \
            or not isinstance(bundle, SourceBundle) \
            or not isinstance(root_resolution, ComponentRootResolution):
        raise TypeError("U11-C requires exact U11-B, bundle and D0 evidence")
    owner = construction.owner
    if root_resolution.status != "resolved" \
            or root_resolution.occurrence != owner \
            or construction.topology.kind != "u_shaped":
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "U11-C inputs do not share one resolved U-shaped root"),))
    topology = construction.topology
    if len(topology.stages) != 2:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the exact U10 proof does not expose exactly two route stages"),))
    index = construction.index
    node = root_resolution.graph.node_for(owner)
    forward = SymbolId(node.symbol.source, f"{node.symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "missing_source", "the exact U-shaped root has no indexed forward"),))

    repeated = []
    for order, stage in enumerate(topology.stages):
        rows, unresolved = _construction_rows(construction, order)
        repeated.append(RepeatedStageExecution(
            UNetStageNodeId(_point(stage.loop.span), "repeated", stage.field, owner),
            stage, rows, unresolved))

    direct_calls = _direct_calls_between(
        index, forward, topology.stages[0], topology.stages[-1])
    eligible_calls = tuple(
        call for call in direct_calls
        if execution_taint_reason(index, forward, call) is None)
    tainted_calls = tuple(call for call in direct_calls
                          if call not in eligible_calls)
    calls_by_field = {
        field: tuple(call for call in eligible_calls
                     if _self_field(call.callee) == field)
        for field in dict.fromkeys(_self_field(call.callee)
                                   for call in eligible_calls)
    }
    direct = []
    unresolved_relations = []
    expanded = index
    field_results = {}
    for call in eligible_calls:
        field = _self_field(call.callee)
        if field not in field_results:
            address = DirectFieldInvocationAddress(
                owner, field, calls_by_field[field],
                topology.stages[0], topology.stages[-1])
            field_results[field] = read_direct_field_construction(
                expanded, bundle, root_resolution, address)
        result = field_results[field]
        if result.status == "absent":
            continue  # exact self method / non-field call, not a constructed stage
        if not result.has_value:
            unresolved_relations.append(UnresolvedStageRelation(
                "direct_construction_incomplete",
                (repeated[0].node_id, repeated[-1].node_id),
                f"constructed-field lookup for {field!r} is {result.status}",
                (call.span,)))
            continue
        field_construction = result.require_value()
        if field_construction.index != expanded:
            expanded = field_construction.index
        direct.append(DirectStageExecution(
            UNetStageNodeId(_point(call.span), "direct", field, owner),
            call, field_construction))

    for call in tainted_calls:
        reason = execution_taint_reason(index, forward, call)
        unresolved_relations.append(UnresolvedStageRelation(
            "direct_call_tainted",
            (repeated[0].node_id, repeated[-1].node_id),
            f"inter-loop call is observed but cannot prove execution: {reason}",
            (call.span,)))

    edge = StageDataflowEdge(
        repeated[0], repeated[-1], "u10_skip_route", topology.skip_route)
    unresolved_relations.append(UnresolvedStageRelation(
        "whole_callable_open",
        tuple(item.node_id for item in (*repeated, *direct)),
        "positive local stage evidence; whole-callable CFG coverage is open",
        tuple(stage.loop.span for stage in topology.stages
              if stage.loop.span is not None)))
    if direct:
        unresolved_relations.append(UnresolvedStageRelation(
            "direct_stage_order_open",
            tuple(item.node_id for item in (*repeated, *direct)),
            "inter-loop source position does not prove unconditional execution order",
            tuple(item.call.span for item in direct)))
    graph = UNetStageExecutionGraph(
        owner, topology, construction, tuple(repeated), tuple(direct),
        (edge,), tuple(unresolved_relations), expanded)
    spans = tuple(dict.fromkeys((
        *(stage.loop.span for stage in topology.stages
          if stage.loop.span is not None),
        *(item.call.span for item in direct),
        *topology.skip_route.spans,
    )))
    return ReaderResult.incomplete(
        owner, graph,
        failures=(ReaderFailure(
            "incomplete_graph",
            "positive U-shaped stage graph; non-proven relations remain unresolved"),),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact U10+B stage execution and skip evidence"),))


__all__ = [
    "DirectStageExecution",
    "RepeatedStageExecution",
    "StageDataflowEdge",
    "UNetStageExecutionGraph",
    "UNetStageNodeId",
    "UNRESOLVED_RELATION_KINDS",
    "UnresolvedStageRelation",
    "read_unet_stage_execution",
]
