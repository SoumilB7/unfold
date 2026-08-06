"""Neutral exact construction-argument to ``__init__``-formal binding.

This boundary transports syntax across an already-authoritative OwnerGraph
construction edge.  It assigns no architectural role to a field, class,
parameter, or value.  Defaults, variadics, dynamic/rival candidates and
arguments that do not bind exactly remain typed partial/failure.
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
    ConstructionSite,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)


@dataclass(frozen=True)
class ConstructionArgumentBinding:
    """One explicit construction expression bound to one child formal."""

    site: ConstructionSite
    parent_occurrence: OwnerOccurrenceId
    child_occurrence: OwnerOccurrenceId
    child_symbol: SymbolId
    constructor_callable: SymbolId
    formal: ParamRecord
    actual: ExprNode
    binding_kind: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.site, ConstructionSite) \
                or self.site.span is None:
            raise TypeError("construction binding carries its exact site")
        if not isinstance(self.parent_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.child_occurrence, OwnerOccurrenceId):
            raise TypeError("construction binding is occurrence-qualified")
        if self.child_occurrence.sites != (
                *self.parent_occurrence.sites, self.site.site_id):
            raise ValueError("construction binding names the exact graph edge")
        if not isinstance(self.child_symbol, SymbolId) \
                or not isinstance(self.constructor_callable, SymbolId) \
                or self.constructor_callable.source != self.child_symbol.source \
                or self.constructor_callable.qualified_name \
                != f"{self.child_symbol.qualified_name}.__init__":
            raise ValueError("construction binding names the exact initializer")
        if not isinstance(self.formal, ParamRecord) \
                or self.formal.kind in {"vararg", "kwarg"}:
            raise TypeError("construction binding names one ordinary formal")
        if not isinstance(self.actual, ExprNode) or self.actual.span is None:
            raise TypeError("construction binding carries one exact actual")
        if self.binding_kind not in {"positional", "keyword"}:
            raise ValueError("construction binding kind is closed")
        if self.binding_kind == "positional":
            if self.actual not in self.site.args:
                raise ValueError("positional actual belongs to the exact site")
        else:
            matches = tuple(value for name, value in self.site.kwargs
                            if name == self.formal.name)
            if len(matches) != 1 or matches[0] != self.actual:
                raise ValueError("keyword actual belongs to the exact site")
        required = {self.site.span, self.actual.span}
        if not required <= set(self.spans) or any(
                not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("construction binding retains exact provenance")


@dataclass(frozen=True)
class ConstructionBindingResolution:
    """Closed result for one exact parent->child construction occurrence."""

    status: str
    root: ComponentRootResolution | ConstructedComponentRoot
    site: ConstructionSite
    parent_occurrence: OwnerOccurrenceId
    child_occurrence: OwnerOccurrenceId
    child_symbol: SymbolId | None = None
    constructor_callable: SymbolId | None = None
    bindings: tuple[ConstructionArgumentBinding, ...] = ()
    unresolved: tuple[str, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "partial", "failed"}:
            raise ValueError("unknown construction-binding status")
        root = require_resolved_component_root(
            self.root, caller="ConstructionBindingResolution")
        if not isinstance(self.site, ConstructionSite) \
                or not isinstance(self.parent_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.child_occurrence, OwnerOccurrenceId):
            raise TypeError("construction result retains exact address inputs")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("failure detail requires failure kind")
        names = tuple(item.formal.name for item in self.bindings)
        if len(names) != len(set(names)):
            raise ValueError("each constructor formal is bound at most once")
        for item in self.bindings:
            if not isinstance(item, ConstructionArgumentBinding) \
                    or item.site != self.site \
                    or item.parent_occurrence != self.parent_occurrence \
                    or item.child_occurrence != self.child_occurrence \
                    or item.child_symbol != self.child_symbol \
                    or item.constructor_callable != self.constructor_callable:
                raise ValueError("every binding belongs to this exact edge")
        if self.status in {"resolved", "partial"}:
            node = root.graph.node_for(self.child_occurrence)
            if not isinstance(self.child_symbol, SymbolId) \
                    or not isinstance(self.constructor_callable, SymbolId) \
                    or node is None or node.symbol != self.child_symbol \
                    or self.failure_kind:
                raise ValueError("bound construction round-trips through D0")
            if self.status == "resolved" and self.unresolved:
                raise ValueError("resolved construction has no unresolved lanes")
            if self.status == "partial" and not self.unresolved:
                raise ValueError("partial construction names unresolved lanes")
        elif self.failure_kind not in {
                "site_not_owned", "child_not_in_graph", "index_mismatch",
                "candidate_not_unique", "callable_unavailable",
                "duplicate_argument"}:
            raise ValueError("failed construction carries a known failure")
        elif self.child_symbol is not None \
                or self.constructor_callable is not None \
                or self.bindings or self.unresolved:
            raise ValueError("failed construction carries failure only")

    def for_formal(self, name: str) -> ConstructionArgumentBinding | None:
        matches = tuple(item for item in self.bindings
                        if item.formal.name == name)
        return matches[0] if len(matches) == 1 else None


def bind_construction_site(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    parent_occurrence: OwnerOccurrenceId,
    site: ConstructionSite,
) -> ConstructionBindingResolution:
    """Bind one graph-authoritative construction site to child formals."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("construction binding requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="bind_construction_site")
    if not isinstance(parent_occurrence, OwnerOccurrenceId) \
            or not isinstance(site, ConstructionSite):
        raise TypeError("construction binding requires an occurrence and site")
    parent = root.graph.node_for(parent_occurrence)
    child_occurrence = OwnerOccurrenceId(
        parent_occurrence.root, (*parent_occurrence.sites, site.site_id))
    indexed_sites = tuple(
        item for item in (
            *index.construction_sites,
            *(element for record in index.containers
              for element in record.elements))
        if item.site_id == site.site_id)
    if len(indexed_sites) != 1 or indexed_sites[0] != site:
        return _failure(
            root, site, parent_occurrence, child_occurrence,
            "index_mismatch", "site is not the authoritative ProgramIndex record")
    if parent is None or site.owner != parent.symbol:
        return _failure(
            root, site, parent_occurrence, child_occurrence,
            "site_not_owned", "site owner is not the exact parent occurrence")
    child = root.graph.node_for(child_occurrence)
    if child is None:
        return _failure(
            root, site, parent_occurrence, child_occurrence,
            "child_not_in_graph", "child occurrence does not round-trip")
    if len(site.candidates) != 1 \
            or site.candidates[0].symbol != child.symbol:
        return _failure(
            root, site, parent_occurrence, child_occurrence,
            "candidate_not_unique", "site has no unique graph-matching candidate")
    if index.class_by_symbol(child.symbol) is None:
        return _failure(
            root, site, parent_occurrence, child_occurrence,
            "index_mismatch", "child class is absent from this ProgramIndex")
    callable_symbol = SymbolId(
        child.symbol.source, f"{child.symbol.qualified_name}.__init__")
    record = index.callable_by_symbol(callable_symbol)
    if record is None or record.owner != child.symbol \
            or record.kind != "method" or not record.params \
            or record.params[0].kind not in {"positional", "posonly"}:
        return _failure(
            root, site, parent_occurrence, child_occurrence,
            "callable_unavailable", "child initializer is not exactly bindable")
    params = tuple(record.params[1:])
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    by_name = {item.name: item for item in params
               if item.kind not in {"vararg", "kwarg", "posonly"}}
    bindings = []
    unresolved = []
    bound = set()
    for number, actual in enumerate(site.args):
        if number >= len(positional):
            unresolved.append(f"extra_positional:{number}")
            continue
        formal = positional[number]
        bindings.append(_binding(
            site, parent_occurrence, child_occurrence, child.symbol,
            callable_symbol, formal, actual, "positional"))
        bound.add(formal.name)
    has_kwarg = any(item.kind == "kwarg" for item in params)
    for name, actual in site.kwargs:
        if name == "**":
            unresolved.append("expanded_kwargs")
            continue
        formal = by_name.get(name)
        if formal is None:
            unresolved.append(
                f"unknown_keyword:{name}" if has_kwarg
                else f"invalid_keyword:{name}")
            continue
        if formal.name in bound:
            return _failure(
                root, site, parent_occurrence, child_occurrence,
                "duplicate_argument", f"formal {formal.name!r} is bound twice")
        bindings.append(_binding(
            site, parent_occurrence, child_occurrence, child.symbol,
            callable_symbol, formal, actual, "keyword"))
        bound.add(formal.name)
    status = "partial" if unresolved else "resolved"
    return ConstructionBindingResolution(
        status, root, site, parent_occurrence, child_occurrence,
        child.symbol, callable_symbol, tuple(bindings), tuple(unresolved))


def _binding(site, parent, child, child_symbol, callable_symbol,
             formal, actual, kind):
    return ConstructionArgumentBinding(
        site, parent, child, child_symbol, callable_symbol, formal, actual,
        kind, tuple(dict.fromkeys((site.span, actual.span))))


def _failure(root, site, parent, child, kind, detail):
    return ConstructionBindingResolution(
        "failed", root, site, parent, child,
        failure_kind=kind, failure_detail=detail)


__all__ = [
    "ConstructionArgumentBinding",
    "ConstructionBindingResolution",
    "bind_construction_site",
]
