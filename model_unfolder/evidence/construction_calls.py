"""U3-F3a — exact construction-call addresses, including external primitives.

An external library construction is not an indexed class and therefore must not
receive a fabricated :class:`OwnerOccurrenceId`.  This boundary gives it a
separate occurrence identity: exact parent occurrence + exact construction site.
It resolves import references lexically and preserves every rival/dynamic site.
No primitive or architectural semantics are assigned here.
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
    CallObservation,
    ConstructionSite,
    ConstructionSiteId,
    ExprNode,
    ImportRecord,
    ProgramIndex,
    SymbolId,
)


@dataclass(frozen=True)
class ConstructionOccurrenceId:
    parent: OwnerOccurrenceId
    site: ConstructionSiteId

    def __post_init__(self) -> None:
        if not isinstance(self.parent, OwnerOccurrenceId):
            raise TypeError("a construction occurrence has an OwnerOccurrenceId parent")
        if not isinstance(self.site, ConstructionSiteId):
            raise TypeError("a construction occurrence has an exact ConstructionSiteId")
        if self.site.owner.source.component_key != self.parent.root.source.component_key:
            raise ValueError("construction and parent belong to the same component")


@dataclass(frozen=True)
class ExternalReferenceProof:
    """Exact import binding proving an external constructor target."""

    reference: ExprNode
    binding: ImportRecord
    qualified_target: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ExprNode):
            raise TypeError("an external reference proof carries an ExprNode")
        if not isinstance(self.binding, ImportRecord):
            raise TypeError("an external reference proof carries its ImportRecord")
        flattened = _flatten_reference(self.reference)
        if flattened is None:
            raise ValueError("an external reference is an exact name/attribute chain")
        root, suffix = flattened
        if self.binding.alias != root:
            raise ValueError("the import binding owns the reference's exact root name")
        expected = ".".join((self.binding.target, *suffix))
        if self.qualified_target != expected:
            raise ValueError("the qualified target is derived from import target + attributes")
        if self.reference.span is None \
                or self.reference.span.source != self.binding.source:
            raise ValueError("the reference and import binding share one exact source")


@dataclass(frozen=True)
class ConstructionAlternative:
    """One exact construction-site interpretation.

    `internal` carries a real graph occurrence; `external` carries an exact
    import proof; `unresolved` carries neither and names the blocking evidence.
    """

    occurrence: ConstructionOccurrenceId
    field: str
    site: ConstructionSite
    kind: str                  # internal | external | unresolved
    internal_occurrence: OwnerOccurrenceId | None = None
    internal_symbol: SymbolId | None = None
    external_reference: ExternalReferenceProof | None = None
    unresolved_kind: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.occurrence, ConstructionOccurrenceId):
            raise TypeError("a construction alternative carries its occurrence identity")
        if not isinstance(self.site, ConstructionSite):
            raise TypeError("a construction alternative carries its ConstructionSite")
        if self.occurrence.site != self.site.site_id:
            raise ValueError("the occurrence identity cites the carried construction site")
        if not self.field or self.site.target != self.field \
                or self.site.target_kind != "field":
            raise ValueError("the alternative is for one exact constructed field")
        if self.kind not in {"internal", "external", "unresolved"}:
            raise ValueError(f"unknown construction-alternative kind {self.kind!r}")
        if self.kind == "internal":
            if self.internal_occurrence is None or self.internal_symbol is None:
                raise ValueError("an internal alternative carries graph occurrence + symbol")
            if self.external_reference is not None or self.unresolved_kind:
                raise ValueError("an internal alternative carries no external/unresolved payload")
            if self.internal_occurrence.root != self.occurrence.parent.root \
                    or self.internal_occurrence.sites[:-1] != self.occurrence.parent.sites \
                    or self.internal_occurrence.sites[-1] != self.site.site_id:
                raise ValueError("the internal occurrence is the exact immediate site child")
            if len(self.site.candidates) != 1 \
                    or self.site.candidates[0].symbol != self.internal_symbol:
                raise ValueError("the unique construction candidate is the internal symbol")
        elif self.kind == "external":
            if self.external_reference is None:
                raise ValueError("an external alternative carries exact import proof")
            if self.internal_occurrence is not None or self.internal_symbol is not None \
                    or self.unresolved_kind:
                raise ValueError("an external alternative carries no internal/unresolved payload")
            if len(self.site.candidates) != 1 \
                    or self.site.candidates[0].symbol is not None \
                    or self.site.candidates[0].reference != self.external_reference.reference:
                raise ValueError("the external proof belongs to the unique unresolved candidate")
        else:
            if not self.unresolved_kind:
                raise ValueError("an unresolved alternative names its unresolved kind")
            if self.internal_occurrence is not None or self.internal_symbol is not None \
                    or self.external_reference is not None:
                raise ValueError("an unresolved alternative carries no resolved target")


@dataclass(frozen=True)
class ConstructionCallResolution:
    status: str                # resolved | ambiguous | incomplete | failed
    caller: OwnerOccurrenceId
    call: CallObservation
    field: str
    caller_symbol: SymbolId | None = None
    alternatives: tuple[ConstructionAlternative, ...] = ()
    incomplete_reasons: tuple[str, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "ambiguous", "incomplete", "failed"}:
            raise ValueError(f"unknown construction-call status {self.status!r}")
        if not isinstance(self.caller, OwnerOccurrenceId):
            raise TypeError("a construction-call result is owner-qualified")
        if not isinstance(self.call, CallObservation):
            raise TypeError("a construction-call result carries its CallObservation")
        if _self_field(self.call.callee) != self.field or not self.field:
            raise ValueError("the carried field is the call's exact self-field")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("a failure detail requires a failure kind")
        for alternative in self.alternatives:
            if not isinstance(alternative, ConstructionAlternative):
                raise TypeError("alternatives are ConstructionAlternative values")
            if alternative.occurrence.parent != self.caller \
                    or alternative.field != self.field:
                raise ValueError("every alternative belongs to the exact caller field")
        if self.status == "resolved":
            if self.caller_symbol is None or len(self.alternatives) != 1 \
                    or self.alternatives[0].kind not in {"internal", "external"}:
                raise ValueError("a resolved call carries one resolved construction alternative")
            if self.incomplete_reasons or self.failure_kind:
                raise ValueError("a resolved call carries no incompleteness/failure")
        elif self.status == "ambiguous":
            if self.caller_symbol is None or len(self.alternatives) < 2:
                raise ValueError("an ambiguous call preserves at least two exact alternatives")
            if self.incomplete_reasons or self.failure_kind:
                raise ValueError("an ambiguous call carries rivals only")
        elif self.status == "incomplete":
            if self.caller_symbol is None or not self.incomplete_reasons:
                raise ValueError("an incomplete call names caller + missing proof")
            if any(a.kind != "unresolved" for a in self.alternatives):
                raise ValueError("incomplete alternatives are unresolved")
            if self.failure_kind:
                raise ValueError("an incomplete call carries no failure")
        else:
            if self.failure_kind not in {
                    "caller_not_in_graph", "index_mismatch", "call_not_in_index"}:
                raise ValueError("a failed call carries a known failure kind")
            if self.caller_symbol is not None or self.alternatives \
                    or self.incomplete_reasons:
                raise ValueError("a failed call carries no resolved address payload")

    @property
    def selected(self) -> ConstructionAlternative | None:
        return self.alternatives[0] if self.status == "resolved" else None


def resolve_construction_call(
    index: ProgramIndex,
    root_resolution: ComponentRootResolution | ConstructedComponentRoot,
    caller: OwnerOccurrenceId,
    call: CallObservation,
) -> ConstructionCallResolution:
    """Resolve one exact ``self.<field>(...)`` call to its construction site."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_construction_call requires a ProgramIndex")
    root_resolution = require_resolved_component_root(
        root_resolution, caller="resolve_construction_call")
    if not isinstance(caller, OwnerOccurrenceId):
        raise TypeError("resolve_construction_call requires an OwnerOccurrenceId caller")
    if not isinstance(call, CallObservation):
        raise TypeError("resolve_construction_call requires a CallObservation")
    field = _self_field(call.callee)
    if field is None:
        raise ValueError("resolve_construction_call requires an exact self.<field> call")

    graph = root_resolution.graph
    node = graph.node_for(caller)
    if node is None:
        return ConstructionCallResolution(
            "failed", caller, call, field, failure_kind="caller_not_in_graph",
            failure_detail="the caller occurrence is absent from the D0 graph")
    if index.class_by_symbol(node.symbol) is None:
        return ConstructionCallResolution(
            "failed", caller, call, field, failure_kind="index_mismatch",
            failure_detail="the caller symbol is absent from this ProgramIndex")
    if call.owner != node.symbol or call not in index.calls_in(call.enclosing_callable):
        return ConstructionCallResolution(
            "failed", caller, call, field, failure_kind="call_not_in_index",
            failure_detail="the call does not round-trip to the exact caller's index")

    sites = tuple(site for site in index.construction_sites_of(node.symbol)
                  if site.target_kind == "field" and site.target == field)
    if not sites:
        return ConstructionCallResolution(
            "incomplete", caller, call, field, node.symbol,
            incomplete_reasons=("no_exact_construction_site",))

    alternatives = tuple(
        _site_alternative(index, graph, node, caller, field, site)
        for site in sites)
    if len(alternatives) > 1:
        return ConstructionCallResolution(
            "ambiguous", caller, call, field, node.symbol, alternatives)
    alternative = alternatives[0]
    if alternative.kind == "unresolved":
        return ConstructionCallResolution(
            "incomplete", caller, call, field, node.symbol, alternatives,
            (alternative.unresolved_kind,))
    return ConstructionCallResolution(
        "resolved", caller, call, field, node.symbol, alternatives)


def _site_alternative(index, graph, node, caller, field, site):
    occurrence = ConstructionOccurrenceId(caller, site.site_id)
    children = tuple(child for child in node.children
                     if child.via_field == field and child.via_site == site.site_id)
    if len(children) == 1 and graph.node_for(children[0].occurrence) is children[0]:
        child = children[0]
        return ConstructionAlternative(
            occurrence, field, site, "internal",
            internal_occurrence=child.occurrence, internal_symbol=child.symbol)
    if len(children) > 1:
        return ConstructionAlternative(
            occurrence, field, site, "unresolved",
            unresolved_kind="rival_internal_occurrences")

    unresolved = tuple(item for item in node.unresolved
                       if item.field == field and item.site == site.site_id)
    if len(unresolved) != 1:
        return ConstructionAlternative(
            occurrence, field, site, "unresolved",
            unresolved_kind=("missing_graph_edge" if not unresolved
                             else "rival_unresolved_edges"))
    if unresolved[0].kind != "external":
        return ConstructionAlternative(
            occurrence, field, site, "unresolved",
            unresolved_kind=unresolved[0].kind)
    proof = _external_reference(index, site)
    if proof is None:
        return ConstructionAlternative(
            occurrence, field, site, "unresolved",
            unresolved_kind="external_reference_unproven")
    return ConstructionAlternative(
        occurrence, field, site, "external", external_reference=proof)


def _external_reference(index, site) -> ExternalReferenceProof | None:
    if len(site.candidates) != 1 or site.candidates[0].symbol is not None:
        return None
    reference = site.candidates[0].reference
    return resolve_import_reference(
        index, site.owner.source, site.enclosing_callable, reference)


def resolve_import_reference(index, source, callable_symbol,
                             reference, *,
                             allow_guarded: bool = False,
                             ) -> ExternalReferenceProof | None:
    """Resolve one exact name/attribute reference through one unshadowed import.

    This is lexical address evidence only.  It assigns no framework or model
    semantics and refuses duplicate/conditional/local rebinding.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_import_reference requires a ProgramIndex")
    if not isinstance(reference, ExprNode):
        raise TypeError("resolve_import_reference requires an ExprNode")
    if not isinstance(allow_guarded, bool):
        raise TypeError("allow_guarded is an explicit evidence-policy flag")
    flattened = _flatten_reference(reference)
    if flattened is None:
        return None
    root, suffix = flattened
    imports = tuple(
        item for item in index.imports
        if item.source == source and item.alias == root
        and (allow_guarded or not item.guard))
    if len(imports) != 1:
        return None
    # A later/conditional module rebinding or local binding makes the import
    # reference non-unique.  Familiar spelling never repairs that ambiguity.
    if any(binding.name == root and binding.kind != "import"
           for binding in index.module_bindings_in(source)):
        return None
    if callable_symbol is not None:
        if any(identifier.name == root
               and identifier.context in {"parameter", "store", "del"}
               for identifier in index.identifiers_in(callable_symbol)):
            return None
        if any(region.construct_kind in {"try", "match"}
               for region in index.unsupported_execution_in(callable_symbol)):
            return None
    binding = imports[0]
    target = ".".join((binding.target, *suffix))
    return ExternalReferenceProof(reference, binding, target)


def _flatten_reference(reference: ExprNode):
    suffix: list[str] = []
    current = reference
    while current.kind == "attribute" and len(current.children) == 1 \
            and current.children[0] is not None:
        suffix.append(current.name)
        current = current.children[0]
    if current.kind != "name" or not current.name:
        return None
    suffix.reverse()
    return current.name, tuple(suffix)


def _self_field(expr: ExprNode) -> str | None:
    if expr.kind == "attribute" and len(expr.children) == 1:
        base = expr.children[0]
        if base is not None and base.kind == "name" and base.name == "self":
            return expr.name
    return None


__all__ = [
    "ConstructionOccurrenceId",
    "ExternalReferenceProof",
    "ConstructionAlternative",
    "ConstructionCallResolution",
    "resolve_construction_call",
    "resolve_import_reference",
]
