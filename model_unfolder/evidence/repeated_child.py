"""U3-F2 — exact repeated-child occurrence boundary.

This module joins already-authoritative address evidence:

* D0 resolved component root;
* B1 resolved declared model-stage occurrence;
* B2 container inventory for that exact occurrence;
* a positively proven repeated invocation template;
* the OwnerGraph child created at that exact element construction site.

It is deliberately neutral.  It does not call the result a decoder layer, layer
stack, block, attention, FFN, or any other architectural role.  It never treats
missing positive evidence as absence because the execution substrate is open and
non-exhaustive.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    DeclaredModelStageResolution,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .container_inventory import ContainerInventory
from .execution_flow import (
    RepeatedInvocationTemplate,
    resolve_addressed_invocations,
)
from .program_index import ProgramIndex, SymbolId


@dataclass(frozen=True)
class RepeatedChildProof:
    """One positive template -> exact immediate graph-child join."""

    model_stage: OwnerOccurrenceId
    template: RepeatedInvocationTemplate
    child_occurrence: OwnerOccurrenceId
    child_symbol: SymbolId

    def __post_init__(self) -> None:
        if not isinstance(self.model_stage, OwnerOccurrenceId):
            raise TypeError("a repeated-child proof names its model-stage occurrence")
        if not isinstance(self.template, RepeatedInvocationTemplate):
            raise TypeError("a repeated-child proof carries a typed invocation template")
        if not isinstance(self.child_occurrence, OwnerOccurrenceId):
            raise TypeError("a repeated-child proof names its exact child occurrence")
        if not isinstance(self.child_symbol, SymbolId):
            raise TypeError("a repeated-child proof names its exact child symbol")
        if self.template.caller_occurrence != self.model_stage:
            raise ValueError("the template is called by the exact model-stage occurrence")
        if self.child_occurrence.root != self.model_stage.root:
            raise ValueError("the child and model stage share the component root")
        if self.child_occurrence.sites[:-1] != self.model_stage.sites:
            raise ValueError("the repeated child is an immediate model-stage child")
        if self.child_occurrence.sites[-1] != self.template.element_template.site_id:
            raise ValueError("the child occurrence ends at the exact element construction site")
        candidates = self.template.element_template.candidates
        if len(candidates) != 1 or candidates[0].symbol != self.child_symbol:
            raise ValueError("the unique element candidate is the carried child symbol")


@dataclass(frozen=True)
class RepeatedChildResolution:
    """Neutral repeated-child address result.

    ``incomplete`` means positive execution/address evidence was insufficient; it
    is intentionally not ``absent``.
    """

    status: str                 # resolved | ambiguous | incomplete | failed
    model_stage: OwnerOccurrenceId
    model_stage_symbol: SymbolId | None = None
    child_occurrence: OwnerOccurrenceId | None = None
    child_symbol: SymbolId | None = None
    proofs: tuple[RepeatedChildProof, ...] = ()
    rivals: tuple[RepeatedChildProof, ...] = ()
    incomplete_reasons: tuple[str, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "ambiguous", "incomplete", "failed"}:
            raise ValueError(f"unknown repeated-child status {self.status!r}")
        if not isinstance(self.model_stage, OwnerOccurrenceId):
            raise TypeError("a repeated-child result is anchored at a model-stage occurrence")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("a failure detail requires a failure kind")
        for proof in self.proofs + self.rivals:
            if not isinstance(proof, RepeatedChildProof):
                raise TypeError("proof/rival entries are RepeatedChildProof values")
            if proof.model_stage != self.model_stage:
                raise ValueError("every proof/rival belongs to the requested model stage")
        if self.status == "resolved":
            if self.model_stage_symbol is None or self.child_occurrence is None \
                    or self.child_symbol is None or not self.proofs:
                raise ValueError("a resolved result carries stage, child and positive proof")
            if self.rivals or self.incomplete_reasons or self.failure_kind:
                raise ValueError("a resolved result carries no rivals/incompleteness/failure")
            if any(p.child_occurrence != self.child_occurrence
                   or p.child_symbol != self.child_symbol for p in self.proofs):
                raise ValueError("every proof resolves to the one carried child occurrence")
        elif self.status == "ambiguous":
            identities = {(p.child_occurrence, p.child_symbol) for p in self.rivals}
            if len(identities) < 2:
                raise ValueError("an ambiguous result preserves at least two exact rivals")
            if (self.child_occurrence is not None or self.child_symbol is not None
                    or self.proofs or self.incomplete_reasons or self.failure_kind):
                raise ValueError("an ambiguous result carries rivals only")
            if self.model_stage_symbol is None:
                raise ValueError("an ambiguous result names its resolved stage symbol")
        elif self.status == "incomplete":
            if not self.incomplete_reasons:
                raise ValueError("an incomplete result names why proof is unavailable")
            if (self.child_occurrence is not None or self.child_symbol is not None
                    or self.proofs or self.rivals or self.failure_kind):
                raise ValueError("an incomplete result carries no child/rivals/failure")
            if self.model_stage_symbol is None:
                raise ValueError("an incomplete result names its resolved stage symbol")
        else:
            if self.failure_kind not in {
                    "index_mismatch", "model_stage_not_in_graph",
                    "inventory_mismatch", "invocation_failure",
                    "template_graph_mismatch"}:
                raise ValueError("a failed result carries a known failure kind")
            if (self.model_stage_symbol is not None or self.child_occurrence is not None
                    or self.child_symbol is not None or self.proofs or self.rivals
                    or self.incomplete_reasons):
                raise ValueError("a failed result carries no resolved address payload")


def resolve_repeated_child(
    index: ProgramIndex,
    root_resolution: ComponentRootResolution | ConstructedComponentRoot,
    stage_resolution: DeclaredModelStageResolution,
    inventory: ContainerInventory,
) -> RepeatedChildResolution:
    """Resolve one exact repeatedly invoked child occurrence, without roles."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_repeated_child requires a ProgramIndex")
    root_resolution = require_resolved_component_root(
        root_resolution, caller="resolve_repeated_child")
    if not isinstance(stage_resolution, DeclaredModelStageResolution):
        raise TypeError("resolve_repeated_child requires a DeclaredModelStageResolution (B1)")
    if stage_resolution.status != "resolved":
        raise ValueError("resolve_repeated_child requires a resolved B1 model stage")
    if not isinstance(inventory, ContainerInventory):
        raise TypeError("resolve_repeated_child requires a ContainerInventory (B2)")
    return resolve_repeated_child_at_owner(
        index, root_resolution, stage_resolution.occurrence, inventory)


def resolve_repeated_child_at_owner(
    index: ProgramIndex,
    root_resolution: ComponentRootResolution | ConstructedComponentRoot,
    stage: OwnerOccurrenceId,
    inventory: ContainerInventory,
) -> RepeatedChildResolution:
    """Resolve repetition for an explicitly authorized exact owner.

    B1 is the normal caller through :func:`resolve_repeated_child`.  This lower
    boundary also supports adapters whose selected architecture *is already*
    the model stage (the D0 root), without manufacturing a B1 declaration or
    interpreting a failed B1 result as permission to fall back.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_repeated_child_at_owner requires a ProgramIndex")
    root_resolution = require_resolved_component_root(
        root_resolution, caller="resolve_repeated_child_at_owner")
    if not isinstance(stage, OwnerOccurrenceId):
        raise TypeError(
            "resolve_repeated_child_at_owner requires an explicit OwnerOccurrenceId")
    if not isinstance(inventory, ContainerInventory):
        raise TypeError(
            "resolve_repeated_child_at_owner requires a ContainerInventory")
    if inventory.owner_occurrence != stage:
        return RepeatedChildResolution(
            "failed", stage, failure_kind="inventory_mismatch",
            failure_detail="the B2 inventory is not owned by the explicit stage")

    graph = root_resolution.graph
    stage_node = graph.node_for(stage)
    if stage_node is None:
        return RepeatedChildResolution(
            "failed", stage, failure_kind="model_stage_not_in_graph",
            failure_detail="the stage occurrence does not round-trip through D0")
    if index.class_by_symbol(stage_node.symbol) is None:
        return RepeatedChildResolution(
            "failed", stage, failure_kind="index_mismatch",
            failure_detail="the model-stage symbol is absent from this ProgramIndex")
    authoritative_containers = index.containers
    if (any(item.record not in authoritative_containers
            or item.record.owner != stage_node.symbol for item in inventory.containers)
            or any(record not in authoritative_containers
                   or record.owner != stage_node.symbol
                   for rival in inventory.rivals for record in rival.records)):
        return RepeatedChildResolution(
            "failed", stage, failure_kind="inventory_mismatch",
            failure_detail="the B2 inventory does not round-trip to this ProgramIndex")

    invocation = resolve_addressed_invocations(
        index, root_resolution, stage, inventory)
    if invocation.status == "failed":
        return RepeatedChildResolution(
            "failed", stage, failure_kind="invocation_failure",
            failure_detail=f"{invocation.failure_kind}: {invocation.failure_detail}")
    if invocation.status == "absent":
        return RepeatedChildResolution(
            "incomplete", stage, stage_node.symbol,
            incomplete_reasons=("model_stage_forward_absent",))
    if not invocation.templates:
        reasons = tuple(sorted({
            item.reason for item in invocation.unresolved
        })) or ("no_proven_repeated_invocation",)
        return RepeatedChildResolution(
            "incomplete", stage, stage_node.symbol,
            incomplete_reasons=reasons)

    proofs: list[RepeatedChildProof] = []
    missing: list[str] = []
    for template in invocation.templates:
        matches = [
            child for child in stage_node.children
            if child.via_field == template.container.field
            and child.via_site == template.element_template.site_id
            and child.symbol == template.element_template.candidates[0].symbol
            and graph.node_for(child.occurrence) is child
        ]
        if len(matches) != 1:
            missing.append(
                f"{template.container.field}@{template.element_template.site_id.span.line}:"
                f"{len(matches)}_graph_matches")
            continue
        child = matches[0]
        proofs.append(RepeatedChildProof(
            stage, template, child.occurrence, child.symbol))

    if missing:
        return RepeatedChildResolution(
            "failed", stage, failure_kind="template_graph_mismatch",
            failure_detail="; ".join(sorted(missing)))

    by_identity: dict[tuple, list[RepeatedChildProof]] = {}
    for proof in proofs:
        by_identity.setdefault(
            (proof.child_occurrence, proof.child_symbol), []).append(proof)
    if len(by_identity) == 1:
        grouped = next(iter(by_identity.values()))
        first = grouped[0]
        return RepeatedChildResolution(
            "resolved", stage, stage_node.symbol,
            first.child_occurrence, first.child_symbol, tuple(grouped))
    rivals = tuple(proof for group in by_identity.values() for proof in group)
    return RepeatedChildResolution(
        "ambiguous", stage, stage_node.symbol, rivals=rivals)


__all__ = [
    "RepeatedChildProof",
    "RepeatedChildResolution",
    "resolve_repeated_child",
    "resolve_repeated_child_at_owner",
]
