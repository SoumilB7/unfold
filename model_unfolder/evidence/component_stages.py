"""Exact repeated-stage inventory below one active component occurrence.

The boundary is deliberately address-only.  It walks graph-authoritative child
invocations from the caller-authorized component root and retains every exact
owner occurrence that positively invokes a repeated container.  A component
may therefore contain one stage, several sequential stages, or none.  Class,
field, component and modality spellings never select a stage.

The inventory does not claim execution order.  ``source_order`` is only the
lexical position of the exact invocation in its parent callable.  Consumers
that need happens-before must join the U3 execution-flow relation separately.
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
from .execution_flow import AddressedInvocation, resolve_addressed_invocations
from .program_index import ProgramIndex, SourceSpan, SymbolId
from .repeated_child import RepeatedChildResolution, resolve_repeated_child_at_owner


@dataclass(frozen=True)
class ComponentRepeatedStage:
    """One exact invoked owner that positively contains repetition."""

    component_occurrence: OwnerOccurrenceId
    parent_occurrence: OwnerOccurrenceId
    stage_occurrence: OwnerOccurrenceId
    stage_symbol: SymbolId
    invocation: AddressedInvocation | None
    repeated_child: RepeatedChildResolution
    source_order: int
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not all(isinstance(item, OwnerOccurrenceId) for item in (
                self.component_occurrence, self.parent_occurrence,
                self.stage_occurrence)):
            raise TypeError("component stages are exact-occurrence qualified")
        if not isinstance(self.stage_symbol, SymbolId):
            raise TypeError("a component stage retains its exact symbol")
        if self.repeated_child.status not in {"resolved", "ambiguous"} \
                or self.repeated_child.model_stage != self.stage_occurrence:
            raise ValueError("a component stage positively carries repetition")
        if not isinstance(self.source_order, int) or self.source_order < 0:
            raise ValueError("stage source order is a non-negative integer")
        if self.invocation is None:
            if self.parent_occurrence != self.stage_occurrence \
                    or self.stage_occurrence != self.component_occurrence \
                    or self.source_order != 0:
                raise ValueError("an uninvoked stage is exactly the component root")
        elif not isinstance(self.invocation, AddressedInvocation) \
                or self.invocation.caller_occurrence != self.parent_occurrence \
                or self.invocation.callee_owner_occurrence != self.stage_occurrence:
            raise ValueError("the exact parent invocation addresses the stage")
        required = {
            *(proof.template.call.span for proof in (
                self.repeated_child.proofs or self.repeated_child.rivals)),
            *((self.invocation.call.span,) if self.invocation is not None else ()),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("component-stage provenance closes decisive sites")


@dataclass(frozen=True)
class ComponentStageInventory:
    """Complete positive stage disposition under the observed owner graph."""

    status: str                 # resolved | incomplete | absent | failed
    component_occurrence: OwnerOccurrenceId
    stages: tuple[ComponentRepeatedStage, ...] = ()
    unresolved: tuple[str, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self):
        if self.status not in {"resolved", "incomplete", "absent", "failed"}:
            raise ValueError("unknown component-stage inventory status")
        if not isinstance(self.component_occurrence, OwnerOccurrenceId):
            raise TypeError("a stage inventory is component-occurrence qualified")
        if any(not isinstance(item, ComponentRepeatedStage)
               or item.component_occurrence != self.component_occurrence
               for item in self.stages):
            raise ValueError("every repeated stage belongs to this component")
        identities = tuple(item.stage_occurrence for item in self.stages)
        if len(identities) != len(set(identities)):
            raise ValueError("stage occurrences are unique")
        orders = tuple(item.source_order for item in self.stages)
        if orders != tuple(sorted(orders)) or len(orders) != len(set(orders)):
            raise ValueError("stage source order is strict and canonical")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("failure detail requires a failure kind")
        if self.status == "resolved":
            if not self.stages or self.unresolved or self.failure_kind:
                raise ValueError("resolved inventory carries exact stages only")
        elif self.status == "incomplete":
            if not self.unresolved or self.failure_kind:
                raise ValueError("incomplete inventory names every blocker")
        elif self.status == "absent":
            if self.stages or self.unresolved or self.failure_kind:
                raise ValueError("absent inventory carries no positive payload")
        elif not self.failure_kind or self.stages or self.unresolved:
            raise ValueError("failed inventory carries failure only")


def resolve_component_stages(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    component_occurrence: OwnerOccurrenceId,
) -> ComponentStageInventory:
    """Inventory every positively repeated stage below one exact component."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("component-stage inventory requires a ProgramIndex")
    root = require_resolved_component_root(root, caller="resolve_component_stages")
    if not isinstance(component_occurrence, OwnerOccurrenceId):
        raise TypeError("component-stage inventory requires an exact occurrence")
    node = root.graph.node_for(component_occurrence)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ComponentStageInventory(
            "failed", component_occurrence,
            failure_kind="owner_not_in_graph",
            failure_detail="component occurrence is absent from graph/index")

    stages: list[ComponentRepeatedStage] = []
    blockers: list[str] = []
    visited: set[OwnerOccurrenceId] = set()
    order = 0

    def visit(owner: OwnerOccurrenceId, entry: AddressedInvocation | None):
        nonlocal order
        if owner in visited:
            blockers.append(f"cycle:{_occurrence_label(owner)}")
            return
        visited.add(owner)
        inventory = resolve_container_inventory(index, root, owner)
        if inventory.status == "failed":
            blockers.append(f"container:{_occurrence_label(owner)}:{inventory.failure_kind}")
            return
        repeated = resolve_repeated_child_at_owner(index, root, owner, inventory)
        if repeated.status in {"resolved", "ambiguous"}:
            stage_node = root.graph.node_for(owner)
            spans = tuple(dict.fromkeys(
                span for span in (
                    *((entry.call.span,) if entry is not None else ()),
                    *(proof.template.call.span for proof in (
                        repeated.proofs or repeated.rivals)),
                ) if isinstance(span, SourceSpan)))
            stages.append(ComponentRepeatedStage(
                component_occurrence,
                entry.caller_occurrence if entry is not None else owner,
                owner, stage_node.symbol, entry, repeated, order, spans))
            order += 1
            return

        invocations = resolve_addressed_invocations(index, root, owner, inventory)
        if invocations.status == "failed":
            blockers.append(
                f"invocations:{_occurrence_label(owner)}:{invocations.failure_kind}")
            return
        if invocations.status == "absent":
            return
        owner_node = root.graph.node_for(owner)
        child_fields = ({child.via_field for child in owner_node.children}
                        if owner_node is not None else set())
        unresolved_fields = ({item.field for item in owner_node.unresolved}
                             if owner_node is not None else set())
        for unresolved in invocations.unresolved:
            field = _self_call_field(unresolved.call)
            if field in child_fields or field in unresolved_fields:
                blockers.append(
                    f"unresolved_child_call:{_occurrence_label(owner)}:"
                    f"{unresolved.call.span.line}:{unresolved.reason}")
        for invocation in sorted(
                invocations.addressed,
                key=lambda item: _span_key(item.call.span)):
            visit(invocation.callee_owner_occurrence, invocation)

    visit(component_occurrence, None)
    stages.sort(key=lambda item: item.source_order)
    blockers = list(dict.fromkeys(blockers))
    if blockers:
        return ComponentStageInventory(
            "incomplete", component_occurrence, tuple(stages), tuple(blockers))
    if stages:
        return ComponentStageInventory(
            "resolved", component_occurrence, tuple(stages))
    return ComponentStageInventory("absent", component_occurrence)


def _self_call_field(call):
    callee = call.callee
    if callee.kind != "attribute" or not callee.children:
        return None
    receiver = callee.children[0]
    return callee.name if receiver.kind == "name" and receiver.name == "self" else None


def _span_key(span):
    return (span.source.canonical_path, span.line, span.col,
            span.end_line or span.line, span.end_col or span.col)


def _occurrence_label(occurrence):
    return "/".join(
        f"{site.owner.qualified_name}:{site.field}:{site.ordinal}"
        for site in occurrence.sites) or occurrence.root.qualified_name


__all__ = [
    "ComponentRepeatedStage", "ComponentStageInventory",
    "resolve_component_stages",
]
