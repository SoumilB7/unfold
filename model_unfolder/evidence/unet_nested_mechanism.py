"""U11-E1 — exact nested mechanism inventory for U-Net cell occurrences.

This boundary begins at U11-D2's exact cell-construction occurrences and walks
only child calls proven by each cell's isolated :class:`OwnerGraph`.  It does
not pretend that a nested cell is a decoder/component root.  Graph-local
container and invocation resolvers preserve direct and symbolic repeated calls
without fabricating runtime repetitions.

Each invoked child is classified independently from positive implementation
evidence already owned by U6/U7:

* an exact attention-compute proof;
* an exact ordinary FFN dataflow/storage proof; or
* unresolved/overlapping evidence, never a spelling-based choice.

Argument lineage is retained as exact caller-formal origins.  U11-E1 does not
call those origins query/context, self/cross, or conditioning; those semantic
roles require a separate interface join.  Whole-callable coverage remains open
and an absent positive mechanism is not a negative fact.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import (
    AttentionComputeProof,
    attention_compute_positive_proof_for_symbol,
)
from .attention_lane import (
    FrameworkAttentionLaneEvidence,
    framework_attention_lane_positive_proof_in_graph,
)
from .component_owner import (
    OwnerOccurrenceId,
    resolve_owner_graph,
)
from .container_inventory import ContainerRival, resolve_container_inventory_in_graph
from .diffusion_stream import local_lineage_at_callable
from .execution_flow import (
    AddressedInvocation,
    RepeatedInvocationTemplate,
    resolve_addressed_invocations_in_graph,
)
from .ffn_mechanism import FFNMechanism, ordinary_ffn_mechanism_at_symbol
from .import_source import (
    canonical_called_import_target,
    resolve_called_import_source,
)
from .program_index import ConstructionSite, ExprNode, ProgramIndex, SourceSpan, SymbolId
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_cell_mechanism import (
    CellCandidateOccurrenceId,
    UNetCellMechanism,
    UNetCellMechanismInventory,
)
from .unet_stage_construction import (
    StageClassCandidate,
    resolve_stage_constructor_candidates,
)


MECHANISM_KINDS = frozenset({"attention", "ffn", "overlap"})
ISSUE_KINDS = frozenset({
    "invocation_unresolved",
    "external_unclassified",
    "mechanism_overlap",
    "owner_graph_conflict",
    "whole_callable_open",
})


@dataclass(frozen=True)
class NestedCellOccurrenceId:
    """One exact local owner occurrence qualified by its outer U11 cell."""

    parent: CellCandidateOccurrenceId
    local: OwnerOccurrenceId

    def __post_init__(self) -> None:
        if not isinstance(self.parent, CellCandidateOccurrenceId) \
                or not isinstance(self.local, OwnerOccurrenceId):
            raise TypeError("a nested occurrence retains outer + local identity")
        if self.local.root != self.parent.symbol or not self.local.sites:
            raise ValueError("a nested occurrence is an exact child of the cell graph")


@dataclass(frozen=True)
class AlternativeCellRoot:
    """One exact rival construction route, preserved without selecting it."""

    parent: CellCandidateOccurrenceId
    field: str
    site: ConstructionSite
    symbol: SymbolId
    rival_sites: tuple[ConstructionSite, ...]
    candidate: StageClassCandidate

    def __post_init__(self) -> None:
        if not isinstance(self.parent, CellCandidateOccurrenceId) \
                or not self.field or not isinstance(self.site, ConstructionSite) \
                or not isinstance(self.symbol, SymbolId) \
                or not isinstance(self.candidate, StageClassCandidate):
            raise TypeError("an alternative root retains cell/field/site/symbol")
        if len(self.rival_sites) < 2 or self.site not in self.rival_sites:
            raise ValueError("an alternative root preserves the complete rival set")
        if any(not isinstance(item, ConstructionSite)
               or item.owner != self.parent.symbol
               or item.target != self.field for item in self.rival_sites):
            raise ValueError("all alternative sites write one exact cell field")
        if self.candidate.symbol != self.symbol \
                or self.candidate.call is None \
                or self.candidate.call.span != self.site.span:
            raise ValueError("the alternative carries its exact U11-A1 route")


@dataclass(frozen=True)
class AlternativeNestedOccurrenceId:
    """An invoked child inside one preserved rival construction root."""

    alternative: AlternativeCellRoot
    local: OwnerOccurrenceId

    def __post_init__(self) -> None:
        if not isinstance(self.alternative, AlternativeCellRoot) \
                or not isinstance(self.local, OwnerOccurrenceId):
            raise TypeError("alternative nested identity retains both boundaries")
        if self.local.root != self.alternative.symbol or not self.local.sites:
            raise ValueError("the nested occurrence is a child of the rival root")


NestedOccurrenceId = NestedCellOccurrenceId | AlternativeNestedOccurrenceId


def _parent_of(occurrence_id: NestedOccurrenceId) -> CellCandidateOccurrenceId:
    return (occurrence_id.parent
            if isinstance(occurrence_id, NestedCellOccurrenceId)
            else occurrence_id.alternative.parent)


def _graph_root_of(occurrence_id: NestedOccurrenceId) -> SymbolId:
    return (occurrence_id.parent.symbol
            if isinstance(occurrence_id, NestedCellOccurrenceId)
            else occurrence_id.alternative.symbol)


@dataclass(frozen=True)
class InvocationInputLineage:
    """Exact caller-formal origins of one invocation argument."""

    slot: str                 # positional:<n> | keyword:<spelling>
    expression: ExprNode
    roots: tuple[str, ...]
    unresolved: bool
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not (self.slot.startswith("positional:")
                or self.slot.startswith("keyword:")):
            raise ValueError("an input slot is exact Python call syntax")
        if not isinstance(self.expression, ExprNode) \
                or self.expression.span is None:
            raise TypeError("an input lineage retains its exact expression")
        if tuple(sorted(set(self.roots))) != self.roots \
                or any(not isinstance(item, str) or not item
                       for item in self.roots):
            raise ValueError("lineage roots are unique sorted caller formals")
        if self.expression.span not in self.spans \
                or any(not isinstance(item, SourceSpan)
                       or item.source != self.expression.span.source
                       for item in self.spans):
            raise ValueError("lineage provenance closes the exact argument")


@dataclass(frozen=True)
class NestedMechanismEvidence:
    """One exact invoked child with independently proven mechanism evidence."""

    occurrence_id: NestedOccurrenceId
    caller_occurrence: OwnerOccurrenceId
    invocation: AddressedInvocation | RepeatedInvocationTemplate
    kind: str
    inputs: tuple[InvocationInputLineage, ...]
    attention: AttentionComputeProof | FrameworkAttentionLaneEvidence | None = None
    ffn: FFNMechanism | None = None

    def __post_init__(self) -> None:
        if self.kind not in MECHANISM_KINDS:
            raise ValueError("nested mechanism kind has a closed vocabulary")
        if self.caller_occurrence.root != _graph_root_of(self.occurrence_id) \
                or self.invocation.caller_occurrence != self.caller_occurrence:
            raise ValueError("the invocation belongs to this exact cell graph")
        if isinstance(self.invocation, AddressedInvocation):
            callee = self.invocation.callee_owner_occurrence
        else:
            callee = self.occurrence_id.local
        if callee != self.occurrence_id.local:
            raise ValueError("the invocation closes its exact callee occurrence")
        expected = (
            self.attention is not None,
            self.ffn is not None,
        )
        if self.kind == "attention" and expected != (True, False):
            raise ValueError("attention kind carries only attention evidence")
        if self.kind == "ffn" and expected != (False, True):
            raise ValueError("FFN kind carries only FFN evidence")
        if self.kind == "overlap" and expected != (True, True):
            raise ValueError("overlap preserves both rival positive proofs")
        if self.ffn is not None and self.ffn.owner_symbol.source.component_key \
                != self.occurrence_id.local.root.source.component_key:
            raise ValueError("FFN proof belongs to the same source component")
        slots = tuple(item.slot for item in self.inputs)
        if len(slots) != len(set(slots)):
            raise ValueError("each call argument has one lineage row")


@dataclass(frozen=True)
class NestedMechanismIssue:
    parent: CellCandidateOccurrenceId
    kind: str
    detail: str
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parent, CellCandidateOccurrenceId) \
                or self.kind not in ISSUE_KINDS or not self.detail:
            raise ValueError("a nested issue is cell-qualified and typed")
        if any(not isinstance(item, SourceSpan) for item in self.spans):
            raise TypeError("nested issues retain typed spans")


@dataclass(frozen=True)
class UNetNestedMechanismInventory:
    cells: UNetCellMechanismInventory
    mechanisms: tuple[NestedMechanismEvidence, ...]
    issues: tuple[NestedMechanismIssue, ...]
    index: ProgramIndex

    def __post_init__(self) -> None:
        if not isinstance(self.cells, UNetCellMechanismInventory) \
                or not isinstance(self.index, ProgramIndex):
            raise TypeError("U11-E1 consumes the exact U11-D2 inventory")
        if any(node not in self.index.source_nodes
               for node in self.cells.index.source_nodes):
            raise ValueError("the U11-E index is a monotonic source expansion")
        parents = {item.occurrence_id for item in self.cells.mechanisms}
        if any(_parent_of(item.occurrence_id) not in parents
               for item in self.mechanisms) \
                or any(item.parent not in parents for item in self.issues):
            raise ValueError("all nested evidence belongs to an exact D2 cell")
        identities = tuple((item.occurrence_id, item.invocation.call_site)
                           for item in self.mechanisms)
        if len(identities) != len(set(identities)):
            raise ValueError("one nested mechanism row exists per exact call")
        for item in self.mechanisms:
            graph = resolve_owner_graph(self.index, _graph_root_of(item.occurrence_id))
            node = graph.node_for(item.occurrence_id.local)
            if node is None:
                raise ValueError("nested occurrence round-trips through its owner graph")
            if isinstance(item.attention, AttentionComputeProof) \
                    and item.attention.child_symbol != node.symbol:
                raise ValueError("attention compute names the exact nested symbol")
            if isinstance(item.attention, FrameworkAttentionLaneEvidence) \
                    and item.attention.child_symbol != node.symbol:
                raise ValueError("framework attention names the exact nested symbol")
            if item.ffn is not None and item.ffn.owner_symbol != node.symbol:
                raise ValueError("FFN proof names the exact nested symbol")
        for parent in parents:
            if sum(item.parent == parent and item.kind == "whole_callable_open"
                   for item in self.issues) != 1:
                raise ValueError("every D2 cell retains one open-callable issue")


def _lineages(index, callable_symbol, call):
    record = index.callable_by_symbol(callable_symbol)
    if record is None:
        return ()
    lineage = local_lineage_at_callable(index, record)
    rows = []
    arguments = tuple(
        [(f"positional:{number}", value)
         for number, value in enumerate(call.args)]
        + [(f"keyword:{name}", value) for name, value in call.kwargs])
    for slot, expression in arguments:
        traced = lineage.trace(expression, call.span, call.guard)
        spans = tuple(dict.fromkeys((expression.span, *traced.spans)))
        rows.append(InvocationInputLineage(
            slot, expression, tuple(sorted(traced.roots)), traced.unresolved,
            tuple(item for item in spans if isinstance(item, SourceSpan))))
    return tuple(rows)


def _template_child(graph, template):
    symbol = template.element_template.candidates[0].symbol
    matches = tuple(
        child for child in graph.node_for(template.caller_occurrence).children
        if child.via_site == template.element_template.site_id
        and child.symbol == symbol and graph.node_for(child.occurrence) is child)
    return matches[0] if len(matches) == 1 else None


def _occurrence_id(scope, child):
    return (NestedCellOccurrenceId(scope, child.occurrence)
            if isinstance(scope, CellCandidateOccurrenceId)
            else AlternativeNestedOccurrenceId(scope, child.occurrence))


def _canonical_child_import(index, bundle, graph, invocation, child):
    if not isinstance(invocation, AddressedInvocation) or child.via_site is None:
        return None
    caller = graph.node_for(invocation.caller_occurrence)
    if caller is None:
        return None
    sites = tuple(
        site for site in index.construction_sites_of(caller.symbol)
        if site.site_id == child.via_site)
    if len(sites) != 1:
        return None
    call = _call_for_site(index, sites[0])
    if call is None:
        return None
    resolved = resolve_called_import_source(
        index, bundle, child.symbol.source.component_key, call)
    if resolved.status != "resolved" or resolved.imported_symbol != child.symbol:
        return None
    return canonical_called_import_target(bundle, resolved)


def _classify(index, bundle, scope, graph, invocation, child):
    attention = attention_compute_positive_proof_for_symbol(index, child.symbol)
    if attention is None and isinstance(invocation, AddressedInvocation):
        attention = framework_attention_lane_positive_proof_in_graph(
            index, graph, invocation,
            canonical_import=_canonical_child_import(
                index, bundle, graph, invocation, child))
    ffn_result = ordinary_ffn_mechanism_at_symbol(index, child.symbol)
    ffn = ffn_result.value if ffn_result.status == "resolved" else None
    if attention is None and ffn is None:
        return None
    kind = ("overlap" if attention is not None and ffn is not None
            else "attention" if attention is not None else "ffn")
    return NestedMechanismEvidence(
        _occurrence_id(scope, child),
        invocation.caller_occurrence, invocation, kind,
        _lineages(index, invocation.call.enclosing_callable, invocation.call),
        attention, ffn)


def _call_for_site(index, site):
    matches = tuple(
        call for call in index.calls_in(site.enclosing_callable)
        if call.span == site.span)
    return matches[0] if len(matches) == 1 else None


def _alternative_roots(index, bundle, parent, unresolved):
    """Exact alternatives behind one rival-container invocation.

    No occurrence is manufactured.  Each result retains its authoritative
    construction site plus the complete rival set; semantic agreement may be
    evaluated only after every alternative is independently scanned.
    """
    if unresolved.reason != "rival_container_records" \
            or len(unresolved.evidence) != 1 \
            or not isinstance(unresolved.evidence[0], ContainerRival):
        return (), index
    rival = unresolved.evidence[0]
    sites = tuple(site for record in rival.records for site in record.elements)
    rows = []
    expanded = index
    for site in sites:
        call = _call_for_site(expanded, site)
        if call is None:
            continue
        candidates, _issues, expanded = resolve_stage_constructor_candidates(
            expanded, bundle, parent.symbol.source.component_key, call)
        rows.extend(AlternativeCellRoot(
            parent, rival.field, site, candidate.symbol, sites, candidate)
            for candidate in candidates)
    return tuple(rows), expanded


def _scan_graph(index: ProgramIndex, bundle,
                parent: CellCandidateOccurrenceId, scope, graph):
    rows = []
    issues = []
    if graph.conflicts:
        issues.append(NestedMechanismIssue(
            parent, "owner_graph_conflict",
            "the exact cell graph retains rival child owners",
            tuple(rival.site.span for conflict in graph.conflicts
                  for rival in conflict.rivals)))

    queue = [graph.root]
    visited = set()
    alternatives = []
    while queue:
        owner = queue.pop(0)
        if owner.occurrence in visited:
            continue
        visited.add(owner.occurrence)
        inventory = resolve_container_inventory_in_graph(
            index, graph, owner.occurrence)
        invocation_result = resolve_addressed_invocations_in_graph(
            index, graph, owner.occurrence, inventory)
        if invocation_result.status != "resolved":
            continue
        for unresolved in invocation_result.unresolved:
            issues.append(NestedMechanismIssue(
                parent, "invocation_unresolved", unresolved.reason,
                tuple(item for item in (unresolved.call.span,)
                      if isinstance(item, SourceSpan))))
            new_alternatives, index = _alternative_roots(
                index, bundle, parent, unresolved)
            alternatives.extend(new_alternatives)
        for external in invocation_result.external_addressed:
            issues.append(NestedMechanismIssue(
                parent, "external_unclassified",
                "an exact external invocation has no U11 nested mechanism proof",
                tuple(item for item in (
                    external.call.span, external.construction.site.span)
                    if isinstance(item, SourceSpan))))
        candidates = []
        for invocation in invocation_result.addressed:
            child = graph.node_for(invocation.callee_owner_occurrence)
            if child is not None:
                candidates.append((invocation, child))
        for template in invocation_result.templates:
            child = _template_child(graph, template)
            if child is None:
                issues.append(NestedMechanismIssue(
                    parent, "invocation_unresolved",
                    "a repeated template does not close one owner occurrence",
                    (template.call.span, template.element_template.span)))
            else:
                candidates.append((template, child))

        for invocation, child in candidates:
            classified = _classify(
                index, bundle, scope, graph, invocation, child)
            if classified is not None:
                rows.append(classified)
                if classified.kind == "overlap":
                    issues.append(NestedMechanismIssue(
                        parent, "mechanism_overlap",
                        "one exact child positively proves attention and FFN",
                        (invocation.call.span,)))
                # A positive mechanism is this traversal's semantic leaf.  Its
                # own U6/U7 evidence already owns the internal implementation.
                continue
            queue.append(child)

    return (tuple(rows), tuple(issues), tuple(dict.fromkeys(alternatives)),
            index)


def _cell_inventory(index: ProgramIndex, bundle, cell: UNetCellMechanism):
    parent = cell.occurrence_id
    graph = resolve_owner_graph(index, parent.symbol)
    rows, issues, alternatives, index = _scan_graph(
        index, bundle, parent, parent, graph)
    rows = list(rows)
    issues = list(issues)
    # Rival field constructions remain alternatives, never a chosen owner.
    # Scan each exact candidate as its own isolated graph and preserve that
    # qualification on every emitted nested occurrence.
    for alternative in alternatives:
        child_rows, child_issues, nested_alternatives, index = _scan_graph(
            index, bundle, parent, alternative,
            resolve_owner_graph(index, alternative.symbol))
        rows.extend(child_rows)
        issues.extend(child_issues)
        if nested_alternatives:
            issues.append(NestedMechanismIssue(
                parent, "invocation_unresolved",
                "nested rival construction requires another qualification level",
                tuple(item.site.span for item in nested_alternatives)))

    issues.append(NestedMechanismIssue(
        parent, "whole_callable_open",
        "positive nested mechanisms; whole-callable coverage remains open",
        tuple(item.invocation.call.span for item in rows)))
    return tuple(rows), tuple(issues), index


def read_unet_nested_mechanisms(cells: UNetCellMechanismInventory) \
        -> ReaderResult[UNetNestedMechanismInventory]:
    """Derive positive nested attention/FFN evidence per exact U11 cell."""
    if not isinstance(cells, UNetCellMechanismInventory):
        raise TypeError("U11-E1 requires the exact U11-D2 inventory")
    mechanisms = []
    issues = []
    expanded = cells.index
    bundle = cells.cells.bundle
    for cell in cells.mechanisms:
        child_rows, child_issues, expanded = _cell_inventory(
            expanded, bundle, cell)
        mechanisms.extend(child_rows)
        issues.extend(child_issues)
    value = UNetNestedMechanismInventory(
        cells, tuple(mechanisms), tuple(issues), expanded)
    spans = tuple(dict.fromkeys(
        span for item in value.mechanisms
        for span in (
            item.invocation.call.span,
            *(item.attention.spans if item.attention is not None else ()),
            *(item.ffn.spans if item.ffn is not None else ()),
        ) if isinstance(span, SourceSpan)))
    if not spans:
        spans = tuple(dict.fromkeys(
            item.occurrence_id.candidate_span for item in cells.mechanisms))
    return ReaderResult.incomplete(
        cells.cells.graph.owner, value,
        failures=(ReaderFailure(
            "incomplete_graph",
            "positive nested mechanisms; stream roles and CFG remain open"),),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact U11 cell→nested invocation→U6/U7 positive evidence"),))


__all__ = [
    "AlternativeCellRoot",
    "AlternativeNestedOccurrenceId",
    "ISSUE_KINDS",
    "MECHANISM_KINDS",
    "InvocationInputLineage",
    "NestedCellOccurrenceId",
    "NestedMechanismEvidence",
    "NestedMechanismIssue",
    "UNetNestedMechanismInventory",
    "read_unet_nested_mechanisms",
]
