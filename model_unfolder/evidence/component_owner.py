"""U3-B — exact construction-occurrence ownership over ``ProgramIndex``.

``ProgramIndex`` records syntax and candidate construction edges.  This module
resolves only what those edges prove.  Its identity is an
``OwnerOccurrenceId`` (root symbol + the exact chain of construction sites),
never a class name or class symbol by itself.  Consequently two fields that
construct the same class remain two owners.

Config-prefix propagation is equally strict: a prefix is published only when
one exact prefix survives.  Rival prefixes remain on the node and become typed
conflicts; they never fall back to the parent's prefix.

This resolver performs address work, not mechanism classification.  Names may
resolve an exact import binding, but no substring or role vocabulary can choose
an architectural owner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .program_index import (
    ConflictRecord,
    ConstructionSiteId,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)


ConfigPrefix = tuple[str, ...]


@dataclass(frozen=True)
class OwnerOccurrenceId:
    """Identity of one constructed occurrence.

    ``sites`` contains the complete root-to-child construction chain.  Helper
    folds retain both the field's helper-call site and the helper's return site,
    so two fields calling the same helper cannot collapse.
    """

    root: SymbolId
    sites: tuple[ConstructionSiteId, ...] = ()

    def child(self, *sites: ConstructionSiteId) -> "OwnerOccurrenceId":
        if not sites:
            raise ValueError("an owner child requires at least one construction site")
        if any(not isinstance(site, ConstructionSiteId) for site in sites):
            raise TypeError("owner occurrence chains contain ConstructionSiteId values")
        return OwnerOccurrenceId(self.root, self.sites + tuple(sites))


@dataclass(frozen=True)
class ConfigBinding:
    """All config-prefix candidates proven for one child parameter."""

    parameter: str
    prefixes: tuple[ConfigPrefix, ...]
    origin: str = "constructor_argument"

    def __post_init__(self) -> None:
        if not self.parameter:
            raise ValueError("a config binding requires a parameter")
        if any(not isinstance(prefix, tuple) or
               any(not isinstance(part, str) for part in prefix)
               for prefix in self.prefixes):
            raise TypeError("config prefixes must be tuple[str, ...] values")
        normalized = _unique_prefixes(*self.prefixes)
        if normalized != self.prefixes:
            object.__setattr__(self, "prefixes", normalized)

    @property
    def resolved_prefix(self) -> ConfigPrefix | None:
        return self.prefixes[0] if len(self.prefixes) == 1 else None


@dataclass(frozen=True)
class OwnerRival:
    """One rival child attached to its exact parent occurrence and site."""

    parent: OwnerOccurrenceId
    site: ConstructionSiteId
    candidate: SymbolId | None
    reference: str
    provenance: str


@dataclass(frozen=True)
class ConfigPrefixRival:
    """One rival config prefix for one exact child parameter/site."""

    parent: OwnerOccurrenceId
    site: ConstructionSiteId
    parameter: str
    prefix: ConfigPrefix


@dataclass(frozen=True)
class UnresolvedChild:
    """A construction edge the resolver could not prove uniquely."""

    parent: OwnerOccurrenceId
    field: str
    kind: str
    site: ConstructionSiteId | None
    span: SourceSpan | None
    detail: str = ""


@dataclass(frozen=True)
class OwnerNode:
    """One exact owner occurrence in the resolved construction graph."""

    occurrence: OwnerOccurrenceId
    symbol: SymbolId
    config_bindings: tuple[ConfigBinding, ...]
    config_prefix_candidates: tuple[ConfigPrefix, ...]
    via_site: ConstructionSiteId | None
    via_field: str
    via_kind: str
    children: tuple["OwnerNode", ...] = ()
    unresolved: tuple[UnresolvedChild, ...] = ()

    @property
    def config_prefix(self) -> ConfigPrefix | None:
        """The unique component prefix, or ``None`` when absent/ambiguous."""
        return (self.config_prefix_candidates[0]
                if len(self.config_prefix_candidates) == 1 else None)


@dataclass(frozen=True)
class OwnerGraph:
    """Resolved occurrence tree plus every exact typed conflict."""

    root: OwnerNode
    conflicts: tuple[ConflictRecord, ...] = ()

    def walk(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def node_for(self, occurrence: OwnerOccurrenceId) -> OwnerNode | None:
        """Lookup by occurrence identity only; class-symbol lookup is unsafe."""
        if not isinstance(occurrence, OwnerOccurrenceId):
            raise TypeError("node_for requires OwnerOccurrenceId; use nodes_for_symbol")
        return next((node for node in self.walk()
                     if node.occurrence == occurrence), None)

    def nodes_for_symbol(self, symbol: SymbolId) -> tuple[OwnerNode, ...]:
        """Return every occurrence of a class symbol without choosing one."""
        if not isinstance(symbol, SymbolId):
            raise TypeError("nodes_for_symbol requires SymbolId")
        return tuple(node for node in self.walk() if node.symbol == symbol)


def resolve_owner_graph(
    index: ProgramIndex,
    root_symbol: SymbolId,
    *,
    root_param_prefixes: Mapping[str, ConfigPrefix] | None = None,
    max_depth: int = 64,
) -> OwnerGraph:
    """Resolve the construction graph rooted at ``root_symbol``.

    When the root has exactly one ordinary constructor parameter, that parameter
    is structurally unique and may be bound to the root document.  A root with
    several possible parameters is never guessed: callers must provide exact
    ``root_param_prefixes`` or the root receives an explicit unresolved record.
    """
    if index.class_by_symbol(root_symbol) is None:
        raise ValueError("root_symbol must name a class in this ProgramIndex")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    resolver = _Resolver(index, max_depth)
    bindings, unresolved = resolver.root_bindings(root_symbol, root_param_prefixes)
    occurrence = OwnerOccurrenceId(root_symbol)
    root = resolver.build(
        root_symbol,
        occurrence=occurrence,
        config_bindings=bindings,
        prefix_candidates=((),),
        via_site=None,
        via_field="",
        via_kind="root",
        ancestor_symbols=(),
        inherited_unresolved=unresolved,
    )
    return OwnerGraph(root=root, conflicts=tuple(resolver.conflicts))


class _Resolver:
    def __init__(self, index: ProgramIndex, max_depth: int):
        self.program_index = index
        self.max_depth = max_depth
        self.conflicts: list[ConflictRecord] = []

    # -- callable/parameter bindings ------------------------------------- #

    def _init_of(self, class_symbol: SymbolId):
        return self.program_index.callable_by_symbol(
            SymbolId(class_symbol.source, f"{class_symbol.qualified_name}.__init__"))

    def _params(self, class_symbol: SymbolId) -> tuple:
        init = self._init_of(class_symbol)
        return tuple(p for p in (init.params if init else ()) if p.name != "self")

    def _positional_params(self, class_symbol: SymbolId) -> tuple:
        return tuple(p for p in self._params(class_symbol)
                     if p.kind == "positional")

    def root_bindings(self, root_symbol, supplied):
        params = self._params(root_symbol)
        names = {p.name for p in params}
        occurrence = OwnerOccurrenceId(root_symbol)
        if supplied is not None:
            unknown = set(supplied) - names
            if unknown:
                raise ValueError(f"unknown root constructor parameters: {sorted(unknown)}")
            bindings = tuple(ConfigBinding(name, (tuple(prefix),), "root_argument")
                             for name, prefix in supplied.items())
            return bindings, ()
        ordinary = [p for p in params if p.kind not in {"vararg", "kwarg"}]
        if len(ordinary) == 1:
            return (ConfigBinding(ordinary[0].name, ((),), "root_argument"),), ()
        if not ordinary:
            return (), ()
        unresolved = UnresolvedChild(
            occurrence, "", "root_config_binding", None,
            self._init_of(root_symbol).span if self._init_of(root_symbol) else None,
            "multiple root constructor parameters; provide root_param_prefixes",
        )
        return (), (unresolved,)

    # -- recursive graph ------------------------------------------------- #

    def build(self, owner_symbol, *, occurrence, config_bindings,
              prefix_candidates, via_site, via_field, via_kind,
              ancestor_symbols, inherited_unresolved=()):
        unresolved = list(inherited_unresolved)
        if len(occurrence.sites) > self.max_depth:
            unresolved.append(UnresolvedChild(
                occurrence, via_field, "depth_limit", via_site,
                via_site.span if via_site else None,
                f"owner graph exceeded max_depth={self.max_depth}",
            ))
            return OwnerNode(occurrence, owner_symbol, tuple(config_bindings),
                             tuple(prefix_candidates), via_site, via_field,
                             via_kind, unresolved=tuple(unresolved))
        if owner_symbol in ancestor_symbols:
            unresolved.append(UnresolvedChild(
                occurrence, via_field, "cycle", via_site,
                via_site.span if via_site else None,
                f"recursive construction of {owner_symbol.qualified_name}",
            ))
            return OwnerNode(occurrence, owner_symbol, tuple(config_bindings),
                             tuple(prefix_candidates), via_site, via_field,
                             via_kind, unresolved=tuple(unresolved))

        children: list[OwnerNode] = []
        param_prefixes = {binding.parameter: binding.prefixes
                          for binding in config_bindings}
        next_ancestors = ancestor_symbols + (owner_symbol,)
        sites = self._owner_sites(owner_symbol)
        rival_fields = self._rival_field_sites(sites, occurrence, unresolved)
        for site in sites:
            if site.target_kind == "field" and site.target in rival_fields:
                continue
            self._process_site(site, owner_symbol, occurrence, param_prefixes,
                               children, unresolved, next_ancestors)
        return OwnerNode(occurrence, owner_symbol, tuple(config_bindings),
                         tuple(prefix_candidates), via_site, via_field, via_kind,
                         tuple(children), tuple(unresolved))

    def _owner_sites(self, owner_symbol: SymbolId) -> tuple:
        unique = {site.site_id: site
                  for site in self.program_index.construction_sites_of(owner_symbol)
                  if site.target_kind in {"field", "element"}}
        for container in self.program_index.containers:
            if container.owner == owner_symbol:
                for site in container.elements:
                    if site.target_kind in {"field", "element"}:
                        unique.setdefault(site.site_id, site)
        return tuple(sorted(unique.values(), key=_site_sort_key))

    def _rival_field_sites(self, sites, parent_occurrence, unresolved) -> set[str]:
        """Several construction sites writing one field are rival occurrences.

        Selecting the last textual assignment or a guard branch would require a
        control-flow decision. U3-B preserves the sites and lets a mechanism
        reader evaluate a proven branch later; it never publishes both as if the
        parent owned two simultaneous children.
        """
        grouped: dict[str, list] = {}
        for site in sites:
            if site.target_kind == "field":
                grouped.setdefault(site.target, []).append(site)
        rivals = {field for field, choices in grouped.items() if len(choices) > 1}
        for field in sorted(rivals):
            choices = grouped[field]
            owner_rivals = []
            for choice in choices:
                for candidate in choice.candidates or (None,):
                    owner_rivals.append(OwnerRival(
                        parent_occurrence,
                        choice.site_id,
                        candidate.symbol if candidate is not None else None,
                        ((candidate.reference.name
                          or candidate.reference.source_segment
                          or candidate.reference.kind)
                         if candidate is not None else _ref_detail(choice)),
                        candidate.provenance if candidate is not None else "dynamic",
                    ))
            self.conflicts.append(ConflictRecord(
                "rival_owner_chain", tuple(owner_rivals),
                tuple(choice.span for choice in choices if choice.span),
            ))
            unresolved.append(UnresolvedChild(
                parent_occurrence, field, "rival_owner", None,
                choices[0].span,
                "several construction occurrences write the same field",
            ))
        return rivals

    def _process_site(self, site, owner_symbol, parent_occurrence,
                      param_prefixes, children, unresolved,
                      ancestor_symbols) -> None:
        method = self._self_helper(site, owner_symbol)
        if method is None:
            self._resolve_and_recurse(
                site, field=site.target, via_kind=site.target_kind or "field",
                owner_symbol=owner_symbol, parent_occurrence=parent_occurrence,
                occurrence_sites=(site.site_id,), param_prefixes=param_prefixes,
                children=children, unresolved=unresolved,
                ancestor_symbols=ancestor_symbols,
            )
            return

        method_symbol = SymbolId(
            owner_symbol.source, f"{owner_symbol.qualified_name}.{method}")
        return_sites = tuple(sorted(
            (candidate for candidate in self.program_index.construction_sites
             if candidate.enclosing_callable == method_symbol
             and candidate.target_kind == "return"),
            key=_site_sort_key,
        ))
        if not return_sites:
            unresolved.append(UnresolvedChild(
                parent_occurrence, site.target, "external", site.site_id,
                site.span, f"helper {method} has no resolvable return construction",
            ))
            return
        if len(return_sites) > 1:
            rivals = tuple(OwnerRival(
                parent_occurrence, returned.site_id,
                (returned.candidates[0].symbol
                 if len(returned.candidates) == 1 else None),
                _ref_detail(returned), f"helper_return:{method}",
            ) for returned in return_sites)
            self.conflicts.append(ConflictRecord(
                "rival_owner_chain", rivals,
                tuple(returned.span for returned in return_sites if returned.span),
            ))
            unresolved.append(UnresolvedChild(
                parent_occurrence, site.target, "rival_owner", site.site_id,
                site.span, f"helper {method} has several return constructions",
            ))
            return
        returned = return_sites[0]
        self._resolve_and_recurse(
            returned, field=site.target, via_kind="return",
            owner_symbol=owner_symbol, parent_occurrence=parent_occurrence,
            occurrence_sites=(site.site_id, returned.site_id),
            param_prefixes=param_prefixes, children=children,
            unresolved=unresolved, ancestor_symbols=ancestor_symbols,
        )

    def _resolve_and_recurse(self, site, *, field, via_kind, owner_symbol,
                             parent_occurrence, occurrence_sites,
                             param_prefixes, children, unresolved,
                             ancestor_symbols) -> None:
        child_symbol, kind = self._resolve_child(site)
        if child_symbol is None:
            unresolved.append(UnresolvedChild(
                parent_occurrence, field, kind, site.site_id, site.span,
                _ref_detail(site),
            ))
            if kind == "rival_owner":
                chain_site = occurrence_sites[-1]
                rivals = tuple(OwnerRival(
                    parent_occurrence, chain_site, candidate.symbol,
                    candidate.reference.name or candidate.reference.source_segment
                    or candidate.reference.kind,
                    candidate.provenance,
                ) for candidate in site.candidates)
                self.conflicts.append(ConflictRecord(
                    "rival_owner_chain", rivals,
                    (site.span,) if site.span else (),
                ))
            return

        child_occurrence = parent_occurrence.child(*occurrence_sites)
        bindings = self._child_bindings(
            site, child_symbol, param_prefixes, parent_occurrence)
        prefix_candidates = _unique_prefixes(*(
            prefix for binding in bindings for prefix in binding.prefixes))
        children.append(self.build(
            child_symbol,
            occurrence=child_occurrence,
            config_bindings=bindings,
            prefix_candidates=prefix_candidates,
            via_site=occurrence_sites[-1],
            via_field=field,
            via_kind=via_kind,
            ancestor_symbols=ancestor_symbols,
        ))

    # -- exact child symbol resolution ---------------------------------- #

    def _resolve_child(self, site):
        candidates = site.candidates
        if len(candidates) >= 2:
            return None, "rival_owner"
        if not candidates:
            return None, "dynamic"
        candidate = candidates[0]
        if candidate.symbol is not None:
            return candidate.symbol, "resolved"
        imported = self._resolve_import_binding(site.owner.source, candidate)
        if len(imported) == 1:
            return imported[0], "resolved_import"
        if len(imported) > 1:
            return None, "ambiguous_import"
        return None, "external"

    def _resolve_import_binding(self, source, candidate) -> tuple[SymbolId, ...]:
        reference = candidate.reference
        chain = _attribute_chain(reference)
        if not chain:
            return ()
        alias, *attributes = chain
        imports = tuple(record for record in self.program_index.imports
                        if record.source == source and record.alias == alias)
        matches: list[SymbolId] = []
        for record in imports:
            parts = tuple(part for part in record.target.lstrip(".").split(".") if part)
            parts = parts + tuple(attributes)
            if not parts:
                continue
            class_name = parts[-1]
            module_name = parts[-2] if len(parts) >= 2 else ""
            for class_record in self.program_index.classes:
                path_stem = _module_stem(class_record.symbol.source.canonical_path)
                if class_record.symbol.qualified_name == class_name and \
                        (not module_name or path_stem == module_name):
                    matches.append(class_record.symbol)
        return tuple(dict.fromkeys(matches))

    def _self_helper(self, site, owner_symbol) -> str | None:
        constructor = site.constructor
        if constructor is None or constructor.kind != "call" or not constructor.children:
            return None
        callee = constructor.children[0]
        if callee.kind != "attribute" or not callee.children:
            return None
        receiver = callee.children[0]
        if receiver.kind != "name" or receiver.name != "self":
            return None
        method_symbol = SymbolId(
            owner_symbol.source, f"{owner_symbol.qualified_name}.{callee.name}")
        return (callee.name
                if self.program_index.callable_by_symbol(method_symbol) is not None
                else None)

    # -- config-prefix propagation -------------------------------------- #

    def _child_bindings(self, site, child_symbol, param_prefixes,
                        parent_occurrence) -> tuple[ConfigBinding, ...]:
        # A class factory proves the component input prefix at the call site,
        # but not how that factory forwards it into __init__. Do not map factory
        # arguments onto constructor parameters merely because their positions
        # happen to align.
        if site.via.startswith("factory:"):
            prefixes = (_arg_prefixes(site.args[0], param_prefixes)
                        if site.args else ())
            bindings = ((ConfigBinding("@factory_input", prefixes,
                                       "factory_argument"),)
                        if prefixes else ())
            self._record_prefix_conflicts(
                site, bindings, parent_occurrence)
            return bindings

        positional = self._positional_params(child_symbol)
        all_params = {param.name: param for param in self._params(child_symbol)}
        found: dict[str, tuple[ConfigPrefix, ...]] = {}

        for index, argument in enumerate(site.args):
            if index >= len(positional):
                break
            prefixes = _arg_prefixes(argument, param_prefixes)
            if prefixes:
                found[positional[index].name] = prefixes
        for name, argument in site.kwargs:
            if name in all_params:
                prefixes = _arg_prefixes(argument, param_prefixes)
                if prefixes:
                    found[name] = prefixes

        bindings = tuple(ConfigBinding(name, prefixes,
                                       "constructor_argument")
                         for name, prefixes in found.items())
        self._record_prefix_conflicts(site, bindings, parent_occurrence)
        return bindings

    def _record_prefix_conflicts(self, site, bindings,
                                 parent_occurrence) -> None:
        for binding in bindings:
            if len(binding.prefixes) < 2:
                continue
            rivals = tuple(ConfigPrefixRival(
                parent_occurrence, site.site_id, binding.parameter, prefix)
                for prefix in binding.prefixes)
            self.conflicts.append(ConflictRecord(
                "rival_config_prefix", rivals,
                (site.span,) if site.span else (),
            ))


def _arg_prefixes(expr, param_prefixes) -> tuple[ConfigPrefix, ...]:
    """Return every config prefix structurally reachable from ``expr``."""
    if expr is None:
        return ()
    if expr.kind == "ifexp" and len(expr.children) == 3:
        return _unique_prefixes(
            *_arg_prefixes(expr.children[0], param_prefixes),
            *_arg_prefixes(expr.children[2], param_prefixes),
        )
    if expr.kind == "boolop":
        return _unique_prefixes(*(
            prefix for child in expr.children
            for prefix in _arg_prefixes(child, param_prefixes)))

    segments: list[str] = []
    current = expr
    while current is not None and current.kind == "attribute":
        segments.append(current.name)
        current = current.children[0] if current.children else None
    if current is not None and current.kind == "name" \
            and current.name in param_prefixes:
        return _unique_prefixes(*(
            (*prefix, *reversed(segments))
            for prefix in param_prefixes[current.name]))
    return ()


def _unique_prefixes(*prefixes) -> tuple[ConfigPrefix, ...]:
    normalized: list[ConfigPrefix] = []
    for prefix in prefixes:
        value = tuple(prefix)
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _site_sort_key(site):
    span = site.span
    return (span.source.canonical_path if span else "",
            span.line if span else -1, span.col if span else -1,
            site.site_id.ordinal)


def _module_stem(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name[:-3] if name.endswith(".py") else name


def _attribute_chain(expr) -> tuple[str, ...]:
    """Return ``(root, *attrs)`` for an exact Name/Attribute expression."""
    attrs: list[str] = []
    current = expr
    while current is not None and current.kind == "attribute":
        attrs.append(current.name)
        current = current.children[0] if current.children else None
    if current is None or current.kind != "name" or not current.name:
        return ()
    return (current.name, *reversed(attrs))


def _ref_detail(site) -> str:
    if not site.candidates:
        return site.constructor.source_segment if site.constructor else ""
    return ", ".join(candidate.reference.name
                     or candidate.reference.source_segment
                     or candidate.reference.kind
                     for candidate in site.candidates)


__all__ = [
    "ConfigPrefix", "OwnerOccurrenceId", "ConfigBinding", "OwnerRival",
    "ConfigPrefixRival", "UnresolvedChild", "OwnerNode", "OwnerGraph",
    "resolve_owner_graph",
]
