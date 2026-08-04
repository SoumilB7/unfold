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

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
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
class DispatchCandidateAddress:
    """One literal registry candidate at one exact construction site."""

    candidate: ChildCandidate
    keys: tuple[object, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ChildCandidate) \
                or self.candidate.symbol is None:
            raise TypeError("a dispatch candidate address carries one exact class")
        if not self.keys:
            raise ValueError("a dispatch candidate address carries >=1 literal key")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("dispatch candidate keys are unique")


@dataclass(frozen=True)
class DispatchConstructionCensus:
    """The complete literal candidate set for one registry construction.

    This is address evidence only.  Several keys may address the same class,
    but one key may never silently address rival classes.
    """

    parent_occurrence: OwnerOccurrenceId
    parent_symbol: SymbolId
    call: CallObservation
    site: ConstructionSite
    registry: DispatchRegistryRecord
    key_expression: ExprNode
    candidates: tuple[DispatchCandidateAddress, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parent_occurrence, OwnerOccurrenceId):
            raise TypeError("a dispatch census is parent-occurrence qualified")
        if not isinstance(self.parent_symbol, SymbolId):
            raise TypeError("a dispatch census carries its exact parent symbol")
        if not isinstance(self.call, CallObservation):
            raise TypeError("a dispatch census carries its exact call")
        if not isinstance(self.site, ConstructionSite):
            raise TypeError("a dispatch census carries its construction site")
        if not isinstance(self.registry, DispatchRegistryRecord):
            raise TypeError("a dispatch census carries its literal registry")
        if not isinstance(self.key_expression, ExprNode):
            raise TypeError("a dispatch census carries the exact key expression")
        if not self.candidates or any(
                not isinstance(item, DispatchCandidateAddress)
                for item in self.candidates):
            raise TypeError("a dispatch census carries exact candidate addresses")
        if self.site.owner != self.call.owner \
                or self.site.owner != self.parent_symbol:
            raise ValueError("call and construction site belong to the exact parent")
        if self.registry.symbol.source != self.site.owner.source:
            raise ValueError("registry and construction site share one source")
        site_candidates = set(self.site.candidates)
        if any(item.candidate not in site_candidates for item in self.candidates):
            raise ValueError("every census candidate comes from the exact site")
        symbols = tuple(item.candidate.symbol for item in self.candidates)
        if len(set(symbols)) != len(symbols):
            raise ValueError("dispatch census candidates are symbol-unique")
        keys = tuple(key for item in self.candidates for key in item.keys)
        if len(set(keys)) != len(keys):
            raise ValueError("one literal dispatch key cannot address rival classes")
        registry_keys = tuple(
            key.const_value for key, _ in self.registry.entries
            if key.kind == "constant")
        if set(keys) != set(registry_keys) or len(keys) != len(registry_keys):
            raise ValueError("the census covers every literal registry entry exactly")


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
    root: ComponentRootResolution | ConstructedComponentRoot,
    parent_occurrence: OwnerOccurrenceId,
    call: CallObservation,
    decision: ConsumedConfigDecision,
) -> ReaderResult[SelectedDispatchConstruction]:
    """Select one exact literal-registry construction candidate."""
    if not isinstance(decision, ConsumedConfigDecision):
        raise TypeError("resolve_dispatch_construction requires a consumed decision")
    census = resolve_dispatch_candidates(
        index, root, parent_occurrence, call)
    if census.status == "ambiguous":
        return ReaderResult.ambiguous(
            parent_occurrence, census.ambiguity,
            provenance=census.provenance)
    if census.status != "resolved":
        return ReaderResult.failed(
            parent_occurrence, census.failures,
            provenance=census.provenance)
    value = census.value
    component = parent_occurrence.root.source.component_key
    if decision.component != component \
            or decision.occurrence is None \
            or decision.occurrence.component_path != component \
            or decision.document_path:
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "out_of_owner",
            "the consumed decision belongs to another component document"),))
    expected_path = _decision_path(
        root.graph.node_for(parent_occurrence), value.key_expression)
    if expected_path is None or decision.selected_path != ".".join(expected_path):
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "conflict",
            "the consumed occurrence is not the registry key's exact owner path",
            value.key_expression.span),))
    if decision.state != "present" or decision.value_state != "value" \
            or decision.occurrence is None:
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the registry key has no present non-null config occurrence",
            value.key_expression.span),))
    matching = tuple(
        item for item in value.candidates if decision.value in item.keys)
    if len(matching) != 1:
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "incomplete_graph",
            f"registry key {decision.value!r} has no unique indexed class",
            value.key_expression.span),))
    selected = matching[0].candidate
    result = SelectedDispatchConstruction(
        parent_occurrence, value.parent_symbol, call, value.site,
        value.registry, value.key_expression, decision, selected)
    spans = tuple(dict.fromkeys(
        span for span in (
            call.span, value.site.span, value.registry.span,
            value.key_expression.span, selected.reference.span,
        ) if isinstance(span, SourceSpan)))
    return ReaderResult.resolved(
        parent_occurrence, result,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=(tuple(decision.selected_path.split(".")),),
            detail="literal registry entry selected by its exact config occurrence"),))


def resolve_dispatch_candidates(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    parent_occurrence: OwnerOccurrenceId,
    call: CallObservation,
) -> ReaderResult[DispatchConstructionCensus]:
    """Census every exact class addressed by one literal registry call."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_dispatch_candidates requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="resolve_dispatch_candidates")
    if not isinstance(parent_occurrence, OwnerOccurrenceId):
        raise TypeError("resolve_dispatch_candidates requires an exact parent")
    if not isinstance(call, CallObservation):
        raise TypeError("resolve_dispatch_candidates requires an exact call")

    node = root.graph.node_for(parent_occurrence)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "out_of_owner", "the parent does not round-trip through this index"),))
    if call.owner != node.symbol or call not in index.calls_in(call.enclosing_callable):
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "out_of_owner", "the call does not belong to the exact parent"),))
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
    if not registry.entries or any(
            key.kind != "constant" for key, _ in registry.entries):
        return ReaderResult.failed(parent_occurrence, (ReaderFailure(
            "unsupported_syntax",
            "the dispatch registry contains a non-literal key",
            registry.span),))
    grouped: dict[SymbolId, tuple[ChildCandidate, list[object]]] = {}
    key_symbols: dict[object, set[SymbolId]] = {}
    for key, expression in registry.entries:
        matches = tuple(dict.fromkeys(
            candidate for candidate in site.candidates
            if candidate.reference == expression and candidate.symbol is not None))
        if not matches:
            return ReaderResult.failed(parent_occurrence, (ReaderFailure(
                "incomplete_graph",
                f"registry key {key.const_value!r} has no indexed class",
                expression.span),))
        for candidate in matches:
            key_symbols.setdefault(key.const_value, set()).add(candidate.symbol)
            entry = grouped.setdefault(candidate.symbol, (candidate, []))
            entry[1].append(key.const_value)
    rival_keys = tuple(
        key for key, symbols in key_symbols.items() if len(symbols) > 1)
    if rival_keys:
        spans = tuple(
            expression.span for key, expression in registry.entries
            if key.const_value in rival_keys
            and isinstance(expression.span, SourceSpan))
        return ReaderResult.ambiguous(
            parent_occurrence, Ambiguity(sites=spans))
    candidates = tuple(
        DispatchCandidateAddress(candidate, tuple(dict.fromkeys(keys)))
        for candidate, keys in grouped.values())
    value = DispatchConstructionCensus(
        parent_occurrence, node.symbol, call, site, registry, key_expression,
        candidates)
    spans = tuple(dict.fromkeys(
        span for span in (
            call.span, site.span, registry.span,
            key_expression.span,
            *(item.candidate.reference.span for item in candidates),
        ) if isinstance(span, SourceSpan)))
    return ReaderResult.resolved(
        parent_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="complete literal dispatch registry candidate census"),))


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
    if len(bindings) != 1:
        return None
    return bindings[0].resolved_path(tuple(segments))


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
    "DispatchCandidateAddress",
    "DispatchConstructionCensus",
    "SelectedDispatchConstruction",
    "resolve_dispatch_candidates",
    "resolve_dispatch_construction",
]
