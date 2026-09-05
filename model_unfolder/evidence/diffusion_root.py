"""U10-A — exact diffusion-root topology evidence.

This reader publishes two source-proven root *shape candidates*:

``repeated_stack``
    The exact root forward iterates an exact root-owned module container and
    invokes the loop-bound element.  This does not call the element a
    Transformer block; U10-C must prove its mechanisms.

``u_shaped``
    Two exact repeated-container loops are joined by a bypass value: an output
    of the earlier loop invocation is accumulated, a later binding derives a
    value from that accumulator, and the later loop invocation consumes that
    value alongside another output of the earlier invocation.

Class names, field names, config fields, constructor names and source strings
never select either result.  The reader consumes the one ProgramIndex and a
resolved D0 component root, and retains every exact loop/call/binding span used
by the proof.  Because U3 deliberately exposes no whole-callable CFG coverage
certificate, a positive value is returned as ``ReaderResult.incomplete``: it
proves the candidate exists but never claims no other root route exists. It
makes no diagram decision; U10-A publishes the result in parser shadow state.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .container_inventory import (
    ContainerAddress,
    ContainerInventory,
    ContainerRival,
    resolve_container_inventory,
)
from .program_index import (
    BindingObservation,
    CallObservation,
    ExprNode,
    LoopObservation,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult


ROOT_TOPOLOGIES = frozenset({"repeated_stack", "u_shaped"})


def _within(inner: SourceSpan | None, outer: SourceSpan | None) -> bool:
    if inner is None or outer is None or inner.source != outer.source:
        return False
    inner_end = (inner.end_line or inner.line, inner.end_col or inner.col)
    outer_end = (outer.end_line or outer.line, outer.end_col or outer.col)
    return ((inner.line, inner.col) >= (outer.line, outer.col)
            and inner_end <= outer_end)


def _source_order(span: SourceSpan | None) -> tuple[int, int, int, int]:
    if span is None:
        return (-1, -1, -1, -1)
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


def _self_field(expr: ExprNode | None) -> str | None:
    if expr is None or expr.kind != "attribute" or not expr.children:
        return None
    root = expr.children[0]
    return (expr.name if root.kind == "name" and root.name == "self"
            else None)


def _iteration_field(expr: ExprNode | None) -> str | None:
    """Exact ``self.x`` / ``enumerate(self.x)`` / sliced equivalents.

    The field is an address joining the loop to a ContainerInventory record;
    its spelling never supplies architectural meaning.
    """
    if expr is None:
        return None
    direct = _self_field(expr)
    if direct is not None:
        return direct
    if expr.kind == "subscript" and expr.children:
        return _self_field(expr.children[0])
    if expr.kind != "call" or len(expr.children) < 2:
        return None
    callee, arg = expr.children[0], expr.children[1]
    if callee.kind != "name" or callee.name not in {"enumerate", "reversed"}:
        return None
    return _iteration_field(arg)


def _names(expr: ExprNode | None) -> frozenset[str]:
    if expr is None:
        return frozenset()
    found = {expr.name} if expr.kind == "name" and expr.name else set()
    for child in expr.children:
        found.update(_names(child))
    for _key, child in expr.keyword_children:
        found.update(_names(child))
    return frozenset(found)


def _target_names(expr: ExprNode | None) -> tuple[str, ...]:
    if expr is None:
        return ()
    if expr.kind == "name" and expr.name:
        return (expr.name,)
    if expr.kind in {"tuple", "list"}:
        return tuple(name for child in expr.children
                     for name in _target_names(child))
    return ()


def _binding_targets(binding: BindingObservation) -> tuple[str, ...]:
    return tuple(name for target in binding.targets
                 for name in _target_names(target))


def _call_inputs(call: CallObservation) -> frozenset[str]:
    found: set[str] = set()
    for arg in call.args:
        found.update(_names(arg))
    for _key, value in call.kwargs:
        found.update(_names(value))
    return frozenset(found)


@dataclass(frozen=True)
class RepeatedRootStage:
    """One exact container loop and every call through its bound element.

    ``container_records`` preserves either the one ContainerAddress or every
    guarded ContainerRival record.  Rival element construction is not resolved
    or discarded merely to prove that the container is iterated.
    """

    owner: OwnerOccurrenceId
    field: str
    container_records: tuple
    loop: LoopObservation
    element_target: str
    calls: tuple[CallObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner, OwnerOccurrenceId):
            raise TypeError("a repeated root stage is owner-qualified")
        if not self.field or not self.element_target:
            raise ValueError("a repeated root stage retains exact address names")
        if not self.container_records or any(
                not isinstance(item, (ContainerAddress, ContainerRival))
                for item in self.container_records):
            raise TypeError("a repeated root stage cites exact container records")
        if any(item.owner_occurrence != self.owner
               or item.field != self.field for item in self.container_records):
            raise ValueError("container records belong to the exact owner and field")
        if not isinstance(self.loop, LoopObservation) or self.loop.span is None:
            raise TypeError("a repeated root stage cites its exact loop")
        if _iteration_field(self.loop.iterable) != self.field:
            raise ValueError("the loop iterates the cited container field")
        target_names = set(_target_names(self.loop.target))
        if self.element_target not in target_names:
            raise ValueError("the invoked element is bound by the cited loop")
        if not self.calls:
            raise ValueError("a repeated root stage requires an element invocation")
        for call in self.calls:
            if not isinstance(call, CallObservation) or call.span is None:
                raise TypeError("stage calls are exact CallObservations")
            if call.callee.kind != "name" \
                    or call.callee.name != self.element_target:
                raise ValueError("stage calls invoke the exact loop-bound element")
            if not _within(call.span, self.loop.body_span):
                raise ValueError("stage calls lie inside the exact loop body")


@dataclass(frozen=True)
class SkipRoute:
    """One exact bypass from an earlier repeated call into a later one."""

    producer: CallObservation
    producer_binding: BindingObservation
    bypass_output: str
    carried_output: str
    accumulator_binding: BindingObservation
    accumulator: str
    derived_binding: BindingObservation
    derived_value: str
    consumer: CallObservation

    def __post_init__(self) -> None:
        typed = (self.producer_binding, self.accumulator_binding,
                 self.derived_binding)
        if not isinstance(self.producer, CallObservation) \
                or not isinstance(self.consumer, CallObservation) \
                or any(not isinstance(item, BindingObservation)
                       for item in typed):
            raise TypeError("a skip route retains exact calls and bindings")
        produced = set(_binding_targets(self.producer_binding))
        if self.bypass_output not in produced \
                or self.carried_output not in produced \
                or self.bypass_output == self.carried_output:
            raise ValueError("the bypass and carried values are distinct call outputs")
        if self.accumulator_binding.assignment_kind != "augassign" \
                or self.accumulator not in _binding_targets(
                    self.accumulator_binding) \
                or self.bypass_output not in _names(
                    self.accumulator_binding.value):
            raise ValueError("the bypass output is accumulated explicitly")
        if self.derived_value not in _binding_targets(self.derived_binding) \
                or self.accumulator not in _names(self.derived_binding.value):
            raise ValueError("the later value is derived from the exact accumulator")
        consumed = _call_inputs(self.consumer)
        if self.derived_value not in consumed or self.carried_output not in consumed:
            raise ValueError("the later invocation consumes bypass and carried values")

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(item.span for item in (
            self.producer, self.producer_binding,
            self.accumulator_binding, self.derived_binding, self.consumer)
                     if item.span is not None)


@dataclass(frozen=True)
class DiffusionRootTopology:
    """Strongest positively-proven root shape, never a closed-world verdict."""
    kind: str
    owner: OwnerOccurrenceId
    stages: tuple[RepeatedRootStage, ...]
    skip_route: SkipRoute | None = None

    def __post_init__(self) -> None:
        if self.kind not in ROOT_TOPOLOGIES:
            raise ValueError(f"unknown diffusion-root topology {self.kind!r}")
        if not isinstance(self.owner, OwnerOccurrenceId):
            raise TypeError("a diffusion-root topology is owner-qualified")
        if not self.stages or any(not isinstance(stage, RepeatedRootStage)
                                  for stage in self.stages):
            raise ValueError("a diffusion-root topology carries repeated stages")
        if any(stage.owner != self.owner for stage in self.stages):
            raise ValueError("all repeated stages belong to the exact root owner")
        orders = tuple(_source_order(stage.loop.span) for stage in self.stages)
        if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
            raise ValueError("repeated stages are in strict source order")
        if self.kind == "u_shaped":
            if len(self.stages) < 2 or self.skip_route is None:
                raise ValueError("a U-shaped root requires two stages and a skip route")
        elif self.skip_route is not None:
            raise ValueError("a repeated-stack root carries no U-shaped skip claim")


def _forward(index: ProgramIndex, owner: SymbolId):
    return index.callable_by_symbol(SymbolId(
        owner.source, f"{owner.qualified_name}.forward"))


def _container_records(inventory: ContainerInventory) -> dict[str, tuple]:
    out: dict[str, list] = {}
    for item in inventory.containers:
        out.setdefault(item.field, []).append(item)
    for item in inventory.rivals:
        out.setdefault(item.field, []).append(item)
    return {field: tuple(rows) for field, rows in out.items()}


def _repeated_stages(index: ProgramIndex, callable_symbol: SymbolId,
                     owner: OwnerOccurrenceId,
                     inventory: ContainerInventory) -> tuple[RepeatedRootStage, ...]:
    records = _container_records(inventory)
    calls = index.calls_in(callable_symbol)
    found: list[RepeatedRootStage] = []
    for loop in index.loops_in(callable_symbol):
        field = _iteration_field(loop.iterable)
        if field is None or field not in records or loop.body_span is None:
            continue
        targets = set(_target_names(loop.target))
        through: dict[str, list[CallObservation]] = {}
        for call in calls:
            if call.span is None or not _within(call.span, loop.body_span) \
                    or call.callee.kind != "name" \
                    or call.callee.name not in targets:
                continue
            through.setdefault(call.callee.name, []).append(call)
        # A loop with two different bound targets invoked is not one exact
        # repeated stage; retain no guess.  enumerate(index, element) has one.
        if len(through) != 1:
            continue
        element_target, invoked = next(iter(through.items()))
        found.append(RepeatedRootStage(
            owner, field, records[field], loop, element_target,
            tuple(sorted(invoked, key=lambda item: _source_order(item.span)))))
    return tuple(sorted(found, key=lambda item: _source_order(item.loop.span)))


def _binding_containing(bindings, call: CallObservation):
    candidates = tuple(binding for binding in bindings
                       if binding.value is not None
                       and _within(call.span, binding.value.span))
    return min(candidates, key=lambda item: (
        ((item.span.end_line or item.span.line) - item.span.line
         if item.span is not None else 10**9),
        ((item.span.end_col or item.span.col) - item.span.col
         if item.span is not None else 10**9)), default=None)


def _lineage_survives(bindings, name: str, start: SourceSpan,
                      end: SourceSpan) -> bool:
    """A positive source-path proof that ``name`` is not unconditionally killed.

    A self-dependent assignment/augmentation preserves the route. A guarded
    overwrite has a path on which it does not run, so it cannot disprove an
    existence claim. An unconditional overwrite that does not consume the
    previous value kills the route.
    """
    for binding in sorted(bindings, key=lambda item: _source_order(item.span)):
        if binding.span is None \
                or _source_order(binding.span) <= _source_order(start) \
                or _source_order(binding.span) >= _source_order(end) \
                or name not in _binding_targets(binding):
            continue
        if binding.assignment_kind == "augassign" \
                or name in _names(binding.value) \
                or binding.guard:
            continue
        return False
    return True


def _skip_route(stages, bindings) -> SkipRoute | None:
    for left_index, earlier in enumerate(stages[:-1]):
        for later in stages[left_index + 1:]:
            if _source_order(earlier.loop.span) >= _source_order(later.loop.span):
                continue
            for producer in earlier.calls:
                produced_by = _binding_containing(bindings, producer)
                produced = _binding_targets(produced_by) if produced_by else ()
                if len(produced) < 2:
                    continue
                for bridge in bindings:
                    if bridge.assignment_kind != "augassign" \
                            or bridge.span is None \
                            or _source_order(bridge.span) <= _source_order(producer.span) \
                            or _source_order(bridge.span) >= _source_order(later.loop.span):
                        continue
                    targets = _binding_targets(bridge)
                    if len(targets) != 1:
                        continue
                    accumulator = targets[0]
                    bypasses = tuple(name for name in produced
                                     if name in _names(bridge.value))
                    if len(bypasses) != 1:
                        continue
                    bypass = bypasses[0]
                    for derived in bindings:
                        if derived.span is None \
                                or _source_order(derived.span) <= _source_order(bridge.span):
                            continue
                        derived_targets = _binding_targets(derived)
                        if len(derived_targets) != 1 \
                                or accumulator not in _names(derived.value):
                            continue
                        derived_name = derived_targets[0]
                        for consumer in later.calls:
                            inputs = _call_inputs(consumer)
                            carried = tuple(name for name in produced
                                            if name != bypass and name in inputs)
                            if derived.span is None or consumer.span is None \
                                    or _source_order(derived.span) >= _source_order(
                                        consumer.span) \
                                    or derived_name not in inputs \
                                    or len(carried) != 1 \
                                    or not _lineage_survives(
                                        bindings, accumulator, bridge.span,
                                        derived.span) \
                                    or not _lineage_survives(
                                        bindings, carried[0], producer.span,
                                        consumer.span):
                                continue
                            return SkipRoute(
                                producer, produced_by, bypass, carried[0],
                                bridge, accumulator, derived, derived_name,
                                consumer)
    return None


def _provenance(topology: DiffusionRootTopology) -> tuple[ReaderProvenance, ...]:
    spans: list[SourceSpan] = []
    for stage in topology.stages:
        if stage.loop.span is not None:
            spans.append(stage.loop.span)
        spans.extend(call.span for call in stage.calls if call.span is not None)
        for record in stage.container_records:
            rows = (record.records if isinstance(record, ContainerRival)
                    else (record.record,))
            spans.extend(row.span for row in rows if row.span is not None)
    if topology.skip_route is not None:
        spans.extend(topology.skip_route.spans)
    unique = tuple(dict.fromkeys(sorted(spans, key=_source_order)))
    return (ReaderProvenance(
        "source", spans=unique,
        detail="exact root container loops and local value-flow proof"),)


def read_diffusion_root_topology(
        index: ProgramIndex,
        root_resolution: ComponentRootResolution,
        ) -> ReaderResult[DiffusionRootTopology]:
    """Read one exact diffusion-root topology without config or identity input."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("read_diffusion_root_topology requires a ProgramIndex")
    if not isinstance(root_resolution, ComponentRootResolution):
        raise TypeError("read_diffusion_root_topology requires a D0 root resolution")
    if not root_resolution.address_resolved:
        kind = ("parse_failure" if root_resolution.status == "failed"
                else "conflict" if root_resolution.status == "ambiguous"
                else "missing_source")
        return ReaderResult.failed(None, (ReaderFailure(
            kind, f"component root is {root_resolution.status}"),))

    owner = root_resolution.occurrence
    graph = root_resolution.graph
    node = graph.node_for(owner)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the resolved root does not belong to this index"),))
    callable_record = _forward(index, node.symbol)
    if callable_record is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "missing_source", "the exact root class has no indexed forward"),))

    inventory = resolve_container_inventory(index, root_resolution, owner)
    if inventory.status == "failed":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", inventory.failure_detail),))
    stages = _repeated_stages(index, callable_record.symbol, owner, inventory)
    if not stages:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no exact root-owned container invocation is proven"),))

    route = _skip_route(stages, index.bindings_in(callable_record.symbol))
    if route is not None:
        route_stages = tuple(stage for stage in stages
                             if route.producer in stage.calls
                             or route.consumer in stage.calls)
        if len(route_stages) != len(stages):
            # A proved U-shaped route plus an independent repeated root stack
            # are two rival root-topology candidates.  U10-A does not demote
            # the extra stack to a side role or pick the larger candidate.
            sites = tuple(stage.loop.span for stage in stages
                          if stage.loop.span is not None)
            return ReaderResult.ambiguous(
                owner, Ambiguity(sites=sites),
                provenance=(ReaderProvenance(
                    "source", spans=sites,
                    detail="rival U-shaped and independent repeated-stack routes"),))
    topology = DiffusionRootTopology(
        "u_shaped" if route is not None else "repeated_stack",
        owner, stages, route)
    return ReaderResult.incomplete(
        owner, topology,
        failures=(ReaderFailure(
            "incomplete_graph",
            "positive local topology proof; whole-callable coverage is open"),),
        provenance=_provenance(topology))


__all__ = [
    "DiffusionRootTopology",
    "RepeatedRootStage",
    "ROOT_TOPOLOGIES",
    "SkipRoute",
    "read_diffusion_root_topology",
]
