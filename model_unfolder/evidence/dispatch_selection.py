"""Exact code-registry + config-value construction selection.

This boundary keeps the project's code/config division explicit:

* source code proves that one construction site indexes one literal registry;
* an owner-qualified :class:`ConsumedConfigDecision` supplies the runtime key;
* the result addresses the selected class candidate at that exact site.

It does not create an ``OwnerOccurrenceId``.  The authoritative owner graph did
not contain the dynamically selected child, so manufacturing an occurrence
would be a lie.  A later mechanism reader may consume this address directly or
an owner-graph extension may incorporate it under its own closed contract.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .config_access import ConsumedConfigDecision
from .program_index import (
    CallObservation,
    ChildCandidate,
    ConstructionSite,
    DispatchRegistryRecord,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


@dataclass(frozen=True)
class SelectedDispatchConstruction:
    """One selected registry candidate without a fabricated child occurrence."""

    parent_occurrence: OwnerOccurrenceId
    parent_symbol: SymbolId
    call: CallObservation
    site: ConstructionSite
    registry: DispatchRegistryRecord
    key_expression: ExprNode
    decision: ConsumedConfigDecision
    candidate: ChildCandidate

    def __post_init__(self) -> None:
        if not isinstance(self.parent_occurrence, OwnerOccurrenceId):
            raise TypeError("a dispatch selection is parent-occurrence qualified")
        if not isinstance(self.parent_symbol, SymbolId):
            raise TypeError("a dispatch selection carries its exact parent symbol")
        if not isinstance(self.call, CallObservation):
            raise TypeError("a dispatch selection carries its exact call")
        if not isinstance(self.site, ConstructionSite):
            raise TypeError("a dispatch selection carries its construction site")
        if not isinstance(self.registry, DispatchRegistryRecord):
            raise TypeError("a dispatch selection carries its literal registry")
        if not isinstance(self.key_expression, ExprNode):
            raise TypeError("a dispatch selection carries the exact key expression")
        if not isinstance(self.decision, ConsumedConfigDecision):
            raise TypeError("a dispatch selection carries a consumed config decision")
        if not isinstance(self.candidate, ChildCandidate) \
                or self.candidate.symbol is None:
            raise TypeError("a dispatch selection carries one exact class candidate")
        if self.site.owner != self.call.owner \
                or self.site.enclosing_callable.source != self.call.enclosing_callable.source:
            raise ValueError("call and construction site belong to the same parent")
        if self.site.owner != self.parent_symbol:
            raise ValueError("the construction site belongs to the exact parent")
        if self.registry.symbol.source != self.site.owner.source:
            raise ValueError("registry and construction site share one source")
        if self.candidate not in self.site.candidates:
            raise ValueError("the selected candidate is from the exact site")
        selected_values = tuple(
            value for key, value in self.registry.entries
            if key.kind == "constant" and key.const_value == self.decision.value)
        if self.candidate.reference not in selected_values:
            raise ValueError("the candidate is selected by the consumed registry key")
        if not self.decision.present or self.decision.occurrence is None:
            raise ValueError("dispatch selection requires a present exact occurrence")
        component = self.parent_occurrence.root.source.component_key
        if self.decision.component != component \
                or self.decision.occurrence.component_path != component \
                or self.decision.document_path:
            raise ValueError(
                "dispatch decision belongs to the exact component document")
        if not self.decision.mechanism or not self.decision.reader \
                or not self.decision.target.owner \
                or not self.decision.target.fact_key:
            raise ValueError("dispatch decision carries its exact consumer target")


def resolve_dispatch_construction(
    index: ProgramIndex,
    root: ComponentRootResolution,
    parent_occurrence: OwnerOccurrenceId,
    call: CallObservation,
    decision: ConsumedConfigDecision,
) -> ReaderResult[SelectedDispatchConstruction]:
    """Select one exact literal-registry construction candidate."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_dispatch_construction requires a ProgramIndex")
    if not isinstance(root, ComponentRootResolution) or root.status != "resolved":
        raise ValueError("resolve_dispatch_construction requires a resolved root")
    if not isinstance(parent_occurrence, OwnerOccurrenceId):
        raise TypeError("resolve_dispatch_construction requires an exact parent")
    if not isinstance(call, CallObservation):
        raise TypeError("resolve_dispatch_construction requires an exact call")
    if not isinstance(decision, ConsumedConfigDecision):
        raise TypeError("resolve_dispatch_construction requires a consumed decision")

    node = root.graph.node_for(parent_occurrence)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "out_of_owner", "the parent does not round-trip through this index"),))
    if call.owner != node.symbol or call not in index.calls_in(call.enclosing_callable):
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "out_of_owner", "the call does not belong to the exact parent"),))
    component = parent_occurrence.root.source.component_key
    if decision.component != component \
            or decision.occurrence is None \
            or decision.occurrence.component_path != component \
            or decision.document_path:
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "out_of_owner",
            "the consumed decision belongs to another component document"),))
    field = _self_field(call.callee)
    if field is None:
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "unsupported_syntax", "dispatch construction requires self.<field>(...)"),))
    sites = tuple(
        site for site in index.construction_sites_of(node.symbol)
        if site.target_kind == "field" and site.target == field)
    if len(sites) != 1:
        spans = tuple(
            site.span for site in sites if isinstance(site.span, SourceSpan))
        if len(sites) > 1:
            return ReaderResult.ambiguous(
                parent_occurrence, Ambiguity(sites=spans))
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "incomplete_graph", "no exact construction site for the called field"),))
    site = sites[0]
    parsed = _registry_subscript(site.constructor)
    if parsed is None:
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "unsupported_syntax",
            "the construction target is not a literal-registry subscript",
            site.span),))
    registry_name, key_expression = parsed

    expected_path = _decision_path(node, key_expression)
    if expected_path is None or decision.selected_path != ".".join(expected_path):
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "conflict",
            "the consumed occurrence is not the registry key's exact owner path",
            key_expression.span),))
    if decision.state != "present" or decision.value_state != "value" \
            or decision.occurrence is None:
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the registry key has no present non-null config occurrence",
            key_expression.span),))

    registries = tuple(
        record for record in index.dispatch_registries
        if record.symbol.source == site.owner.source
        and record.symbol.qualified_name == registry_name)
    if len(registries) != 1:
        spans = tuple(
            record.span for record in registries
            if isinstance(record.span, SourceSpan))
        if len(registries) > 1:
            return ReaderResult.ambiguous(
                parent_occurrence, Ambiguity(sites=spans))
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact literal registry is absent",
            key_expression.span),))
    registry = registries[0]
    matching_values = tuple(
        value for key, value in registry.entries
        if key.kind == "constant" and key.const_value == decision.value)
    matching_candidates = tuple(dict.fromkeys(
        candidate
        for value in matching_values
        for candidate in site.candidates
        if candidate.reference == value and candidate.symbol is not None))
    symbols = {candidate.symbol for candidate in matching_candidates}
    if len(symbols) > 1:
        spans = tuple(
            candidate.reference.span for candidate in matching_candidates
            if isinstance(candidate.reference.span, SourceSpan))
        return ReaderResult.ambiguous(
            parent_occurrence, Ambiguity(sites=spans))
    if len(symbols) != 1 or not matching_candidates:
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "incomplete_graph",
            f"registry key {decision.value!r} has no unique indexed class",
            key_expression.span),))
    candidate = matching_candidates[0]
    value = SelectedDispatchConstruction(
        parent_occurrence, node.symbol, call, site, registry, key_expression,
        decision, candidate)
    spans = tuple(dict.fromkeys(
        span for span in (
            call.span, site.span, registry.span,
            key_expression.span, candidate.reference.span,
        ) if isinstance(span, SourceSpan)))
    return ReaderResult.resolved(
        parent_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=(tuple(decision.selected_path.split(".")),),
            detail="literal registry entry selected by its exact config occurrence"),))


def _registry_subscript(expression):
    if expression.kind != "call" or not expression.children:
        return None
    target = expression.children[0]
    if target.kind != "subscript" or len(target.children) != 2:
        return None
    registry, key = target.children
    if registry.kind != "name" or not registry.name:
        return None
    return registry.name, key


def _decision_path(node, expression):
    root_name, segments = _config_expression_path(expression)
    if root_name is None:
        return None
    bindings = tuple(
        binding for binding in node.config_bindings
        if binding.parameter == root_name)
    if len(bindings) != 1 or bindings[0].resolved_prefix is None:
        return None
    return (*bindings[0].resolved_prefix, *segments)


def _config_expression_path(expression):
    segments = []
    current = expression
    while current.kind == "attribute" and current.name and current.children:
        segments.append(current.name)
        current = current.children[0]
    if current.kind == "subscript" and len(current.children) == 2:
        base, key = current.children
        if key.kind != "constant" or not isinstance(key.const_value, str):
            return None, ()
        segments.append(key.const_value)
        current = base
    if current.kind != "name" or not current.name:
        return None, ()
    return current.name, tuple(reversed(segments))


def _self_field(expression):
    if expression.kind != "attribute" or not expression.children:
        return None
    root = expression.children[0]
    return (expression.name if root.kind == "name" and root.name == "self"
            else None)


__all__ = [
    "SelectedDispatchConstruction",
    "resolve_dispatch_construction",
]
