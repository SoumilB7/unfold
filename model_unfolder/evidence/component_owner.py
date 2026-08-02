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
    ParseFailure,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)


ConfigPrefix = tuple[str, ...]


def _span_before(left: SourceSpan | None,
                 right: SourceSpan | None) -> bool:
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) \
        <= (right.line, right.col)


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


@dataclass(frozen=True)
class ConstructedComponentRoot:
    """A component-root ADDRESS proven by an exact outer construction.

    This is deliberately distinct from :class:`ComponentRootResolution`: the
    latter proves a bundle-declared architecture, while this type proves
    ``config.path -> construction -> exact self.field installation`` (either a
    direct field construction or a proven local-to-field alias).  Downstream
    ownership readers may consume either closed proof route, but no caller may
    relabel construction evidence as a declared architecture merely to satisfy
    a type.
    """

    component_key: str
    occurrence: OwnerOccurrenceId
    graph: OwnerGraph
    outer_graph: OwnerGraph
    outer_root: OwnerOccurrenceId
    outer_owner_symbol: SymbolId
    config_path: ConfigPrefix
    installation_field: str
    construction_site: ConstructionSiteId
    installation_kind: str
    construction_span: SourceSpan
    installation_span: SourceSpan

    def __post_init__(self) -> None:
        if not self.component_key \
                or self.component_key != ".".join(self.config_path):
            raise ValueError(
                "a constructed component key is its exact non-empty config path")
        if not isinstance(self.occurrence, OwnerOccurrenceId) \
                or not isinstance(self.outer_root, OwnerOccurrenceId):
            raise TypeError("constructed roots carry exact inner + outer occurrences")
        if not isinstance(self.outer_owner_symbol, SymbolId):
            raise TypeError(
                "a constructed root carries the exact outer owner symbol")
        if not isinstance(self.graph, OwnerGraph) \
                or not isinstance(self.outer_graph, OwnerGraph):
            raise TypeError(
                "a constructed root carries inner and outer OwnerGraphs")
        if self.occurrence != self.graph.root.occurrence \
                or self.graph.root.symbol != self.occurrence.root \
                or self.occurrence.sites:
            raise ValueError("the constructed component occurrence is the graph root")
        if self.graph.root.symbol.source.component_key != self.component_key:
            raise ValueError("the constructed graph belongs to the selected component")
        if self.outer_graph.root.occurrence.root != self.outer_root.root:
            raise ValueError(
                "the outer occurrence belongs to the carried outer graph")
        outer_node = self.outer_graph.node_for(self.outer_root)
        if outer_node is None or outer_node.symbol != self.outer_owner_symbol:
            raise ValueError(
                "the outer graph proves the exact construction-owner occurrence")
        if not self.installation_field:
            raise ValueError("a constructed component is installed on an exact field")
        if self.installation_kind not in {"direct_field", "local_alias"}:
            raise ValueError(
                f"unknown component installation kind {self.installation_kind!r}")
        if not isinstance(self.construction_site, ConstructionSiteId) \
                or self.construction_site.owner != self.outer_owner_symbol:
            raise ValueError(
                "the construction belongs to the exact outer occurrence")
        if not isinstance(self.construction_span, SourceSpan) \
                or not isinstance(self.installation_span, SourceSpan):
            raise TypeError("constructed-root proof carries exact source spans")
        if self.construction_site.span != self.construction_span \
                or self.construction_span.source != self.installation_span.source:
            raise ValueError(
                "construction and installation spans close the exact proof")
        construction_point = (
            self.construction_span.line, self.construction_span.col)
        installation_point = (
            self.installation_span.line, self.installation_span.col)
        if self.installation_kind == "direct_field" \
                and installation_point != construction_point:
            raise ValueError(
                "a direct field construction installs at the construction site")
        if self.installation_kind == "local_alias" \
                and installation_point <= construction_point:
            raise ValueError(
                "a local-alias field installation follows the construction")

    @property
    def status(self) -> str:
        return "resolved"


def require_resolved_component_root(value, *, caller: str):
    """Validate either lawful component-root address proof.

    The helper centralizes the sole widening from D0-declared roots to exact
    construction-derived roots.  Unknown objects and non-resolved D0 outcomes
    remain rejected at every downstream boundary.
    """
    if isinstance(value, ConstructedComponentRoot):
        return value
    if not isinstance(value, ComponentRootResolution):
        raise TypeError(
            f"{caller} requires a ComponentRootResolution (D0) or "
            "ConstructedComponentRoot")
    if value.status != "resolved":
        raise ValueError(
            f"{caller} requires a resolved component root; got {value.status!r}")
    return value


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


def resolve_construction_candidate_symbols(
    index: ProgramIndex,
    site,
) -> tuple[SymbolId, ...]:
    """Resolve one construction site's class candidates as ADDRESS evidence.

    A directly indexed candidate is returned verbatim.  An imported candidate
    is bound through the same exact lexical import resolver used by
    :class:`OwnerGraph`; duplicate component-index copies and genuinely rival
    bindings remain visible to the caller.  This helper assigns no mechanism or
    architectural role.
    """
    from .program_index import ConstructionSite

    if not isinstance(index, ProgramIndex):
        raise TypeError("construction candidate resolution requires a ProgramIndex")
    if not isinstance(site, ConstructionSite):
        raise TypeError("construction candidate resolution requires a ConstructionSite")
    if len(site.candidates) != 1:
        return ()
    candidate = site.candidates[0]
    if candidate.symbol is not None:
        return (candidate.symbol,)
    return _Resolver(index, 64)._resolve_import_binding(site.owner.source, candidate)


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
        imported = resolve_construction_candidate_symbols(self.program_index, site)
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
        # Resolve straight-line local aliases at the exact construction site.
        # ``local = config; self.child = Child(local)`` preserves the same
        # address.  Guarded/rival/reassigned locals are killed rather than
        # guessed.  This is address propagation only; no field/class spelling
        # selects an architectural role.
        param_prefixes = self._local_prefixes_at_site(site, param_prefixes)
        # A class factory proves the component input prefix at the call site,
        # but not, by itself, how that factory forwards it into __init__.  When
        # the exact factory body is indexed we may prove the two call bindings
        # (caller -> factory formal -> constructor formal).  External/inherited
        # factories remain an opaque @factory_input; positions at the outer call
        # site never get relabelled as constructor parameters by resemblance.
        if site.via.startswith("factory:"):
            forwarded = self._indexed_factory_bindings(
                site, child_symbol, param_prefixes)
            if forwarded:
                self._record_prefix_conflicts(
                    site, forwarded, parent_occurrence)
                return forwarded
            prefixes = (_arg_prefixes(site.args[0], param_prefixes)
                        if site.args else ())
            bindings = ((ConfigBinding("@factory_input", prefixes,
                                       "factory_input_unproven_forwarding"),)
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

    def _local_prefixes_at_site(self, site, initial) -> dict:
        environment = dict(initial)
        if site.span is None:
            return environment
        bindings = tuple(sorted(
            (item for item in self.program_index.bindings_in(
                site.enclosing_callable)
             if item.span is not None
             and _span_before(item.span, site.span)),
            key=lambda item: (
                item.span.line, item.span.col,
                item.span.end_line or item.span.line,
                item.span.end_col or item.span.col)))
        for binding in bindings:
            targets = tuple(
                target.name for target in binding.targets
                if target.kind == "name" and target.name)
            if len(targets) != 1:
                continue
            target = targets[0]
            if binding.guard or binding.value is None:
                environment.pop(target, None)
                continue
            prefixes = _arg_prefixes(binding.value, environment)
            if prefixes:
                environment[target] = prefixes
            else:
                environment.pop(target, None)
        return environment

    def _indexed_factory_bindings(
            self, site, child_symbol, parent_prefixes) -> tuple[ConfigBinding, ...]:
        """Prove factory-formal -> constructor-formal config forwarding.

        This is deliberately a small Python call-binding proof, not a framework
        protocol.  It applies only to a directly indexed ``@classmethod`` on the
        exact constructed class, with one exact ``return cls(...)``.  A bare
        class-name call could be shadowed in the function scope and is therefore
        not an address proof.  Inherited/external factories, rival returns,
        dynamic forwarding and ``**kwargs``-only forwarding remain opaque.
        """
        method = site.via.partition(":")[2]
        if not method:
            return ()
        factory = self.program_index.callable_by_symbol(SymbolId(
            child_symbol.source, f"{child_symbol.qualified_name}.{method}"))
        if factory is None or factory.owner != child_symbol \
                or not _is_unshadowed_classmethod(
                    self.program_index, child_symbol, factory):
            return ()
        params = tuple(factory.params)
        if not params or params[0].name != "cls":
            return ()
        factory_prefixes = _bind_call_prefixes(
            site.args, site.kwargs, params[1:], parent_prefixes)
        if not factory_prefixes or len(factory.returns) != 1:
            return ()
        returned = factory.returns[0]
        return_observations = self.program_index.return_observations_in(
            factory.symbol)
        if len(return_observations) != 1 \
                or return_observations[0].guard \
                or return_observations[0].value != returned:
            return ()
        forwarded_formals = frozenset(factory_prefixes)
        if self.program_index.unsupported_execution_in(factory.symbol) \
                or any(identifier.name in forwarded_formals
                       and identifier.context in {"store", "del"}
                       for identifier in self.program_index.identifiers_in(
                           factory.symbol)):
            return ()
        if returned.kind != "call" or not returned.children:
            return ()
        callee = returned.children[0]
        if not _is_exact_factory_constructor(callee, child_symbol, params[0].name):
            return ()
        init_params = self._params(child_symbol)
        if not init_params:
            return ()
        returned_args = tuple(returned.children[1:])
        returned_kwargs = tuple(returned.keyword_children)
        found = _bind_call_prefixes(
            returned_args, returned_kwargs, init_params, factory_prefixes)
        return tuple(ConfigBinding(
            name, prefixes, "indexed_factory_forwarding")
            for name, prefixes in found.items())

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


def _bind_call_prefixes(args, kwargs, params, source_prefixes):
    """Bind config prefixes across one exact Python call surface.

    The result contains only positively-proven bindings.  Varargs, ``**kwargs``
    expansion, missing/defaulted arguments and unknown keyword names do not
    fabricate a prefix.
    """
    if any(argument.kind == "starred" for argument in args) \
            or any(name == "**" for name, _argument in kwargs):
        return {}
    positional = tuple(param for param in params
                       if param.kind == "positional")
    by_name = {param.name: param for param in params
               if param.kind not in {"vararg", "kwarg"}}
    found: dict[str, tuple[ConfigPrefix, ...]] = {}
    for index, argument in enumerate(args):
        if index >= len(positional):
            break
        prefixes = _arg_prefixes(argument, source_prefixes)
        if prefixes:
            found[positional[index].name] = prefixes
    for name, argument in kwargs:
        if name == "**" or name not in by_name:
            continue
        prefixes = _arg_prefixes(argument, source_prefixes)
        if prefixes:
            found[name] = prefixes
    return found


def _is_unshadowed_classmethod(index, child_symbol, callable_record) -> bool:
    decorators = tuple(callable_record.decorators)
    if len(decorators) != 1 \
            or decorators[0].kind != "name" \
            or decorators[0].name != "classmethod":
        return False
    class_record = index.class_by_symbol(child_symbol)
    if class_record is None \
            or any(item.attr == "classmethod"
                   for item in class_record.body_assigns):
        return False
    return not any(binding.source == child_symbol.source
                   and binding.name == "classmethod"
                   for binding in index.module_bindings)


def _is_exact_factory_constructor(callee, child_symbol, cls_parameter) -> bool:
    del child_symbol  # the exact class is carried by the bound ``cls`` formal
    return callee.kind == "name" and callee.name == cls_parameter


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


# --------------------------------------------------------------------------- #
# U3-D0 — the component-root address boundary
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ComponentRootCandidate:
    """One class exactly addressed by a component's declared architecture.

    Exact identity is an ADDRESS (component key + declared spelling -> the class
    of that qualified name in that component), never a mechanism claim.  The
    invariants below make the candidate self-verifying: a forged component,
    spelling or span source cannot be constructed.
    """

    component_key: str
    declared_architecture: str
    symbol: SymbolId
    span: SourceSpan               # exact class span, required (never optional)

    def __post_init__(self) -> None:
        if not self.component_key:
            raise ValueError("a component-root candidate requires a component key")
        if not self.declared_architecture:
            raise ValueError("a component-root candidate requires a declared architecture")
        if not isinstance(self.symbol, SymbolId):
            raise TypeError("a component-root candidate symbol must be a SymbolId")
        if not isinstance(self.span, SourceSpan):
            raise TypeError("a component-root candidate requires an exact SourceSpan")
        if self.symbol.source.component_key != self.component_key:
            raise ValueError("candidate symbol is not in the addressed component")
        if self.symbol.qualified_name != self.declared_architecture:
            raise ValueError("candidate symbol is not the declared architecture")
        if self.span.source != self.symbol.source:
            raise ValueError("candidate span does not belong to the symbol's source")


def _candidate_sort_key(candidate: "ComponentRootCandidate"):
    """Canonical, file-order-independent ordering for rival candidates."""
    span = candidate.span
    return (candidate.component_key,
            candidate.symbol.source.canonical_path,
            candidate.symbol.source.content_fingerprint,
            candidate.symbol.qualified_name,
            span.line, span.col)


def _parse_failure_sort_key(failure: ParseFailure):
    """Canonical, file-order-independent ordering for component parse failures."""
    return (failure.source.component_key,
            failure.source.canonical_path,
            failure.source.content_fingerprint,
            failure.kind, failure.detail)


@dataclass(frozen=True)
class ComponentRootResolution:
    """Typed outcome of addressing a SourceBundle component's root class.

    ``resolved`` carries the exact OwnerOccurrenceId and its OwnerGraph;
    ``ambiguous`` preserves every exact rival candidate; ``failed`` carries the
    indexed ProgramIndex parse failures that prevented the lookup; ``absent`` is
    an honest no-match with nothing further.  ``resolved`` is an ADDRESS claim:
    the class was located; any unresolved constructor/config binding stays
    explicitly inside ``graph.root.unresolved`` and is not implied by this status.
    Every closure invariant is enforced so a cross-field forgery cannot be built.
    """

    status: str                       # resolved | absent | ambiguous | failed
    component_key: str
    declared_architecture: str | None = None
    occurrence: OwnerOccurrenceId | None = None
    graph: OwnerGraph | None = None
    candidates: tuple[ComponentRootCandidate, ...] = ()
    parse_failures: tuple[ParseFailure, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "absent", "ambiguous", "failed"}:
            raise ValueError(f"unknown component-root status {self.status!r}")
        if not self.component_key:
            raise ValueError("a component-root resolution requires a component key")
        # closed parse-failure membership: every failure is typed and belongs here
        for failure in self.parse_failures:
            if not isinstance(failure, ParseFailure):
                raise TypeError("parse_failures entries must be ParseFailure records")
            if failure.source.component_key != self.component_key:
                raise ValueError("a component-root failure must belong to this component")
        # closed candidate membership: every rival matches this component + address
        for candidate in self.candidates:
            if not isinstance(candidate, ComponentRootCandidate):
                raise TypeError("candidates entries must be ComponentRootCandidate values")
            if candidate.component_key != self.component_key:
                raise ValueError("a rival candidate must belong to this component")
            if candidate.declared_architecture != self.declared_architecture:
                raise ValueError("a rival candidate must match the declared architecture")
        if self.status != "absent" and not self.declared_architecture:
            raise ValueError(
                f"a {self.status} component root requires a declared architecture")
        if self.status == "resolved":
            if self.occurrence is None or self.graph is None:
                raise ValueError("a resolved component root carries its occurrence and graph")
            if self.candidates or self.parse_failures:
                raise ValueError("a resolved component root carries no rivals or failures")
            if self.occurrence != self.graph.root.occurrence:
                raise ValueError("the resolved occurrence must equal the graph root occurrence")
            # a manually inconsistent OwnerGraph must not certify an unrelated
            # occurrence root: the graph root's own occurrence must name the graph
            # root's class symbol, and a component root has no construction chain.
            if self.graph.root.occurrence.root != self.graph.root.symbol:
                raise ValueError("the resolved occurrence root must be the graph root symbol")
            if self.graph.root.occurrence.sites != ():
                raise ValueError("a component root occurrence carries no construction-site chain")
            root_symbol = self.graph.root.symbol
            if root_symbol.source.component_key != self.component_key:
                raise ValueError("the resolved root is not in the requested component")
            if root_symbol.qualified_name != self.declared_architecture:
                raise ValueError("the resolved root is not the declared architecture")
        elif self.status == "ambiguous":
            if len(self.candidates) < 2:
                raise ValueError("an ambiguous component root preserves >=2 exact rival candidates")
            if self.occurrence is not None or self.graph is not None or self.parse_failures:
                raise ValueError("an ambiguous component root carries rival candidates only")
        elif self.status == "failed":
            if not self.parse_failures:
                raise ValueError("a failed component root carries the indexed parse failures")
            if self.occurrence is not None or self.graph is not None or self.candidates:
                raise ValueError("a failed component root carries parse failures only")
        else:  # absent
            if (self.occurrence is not None or self.graph is not None
                    or self.candidates or self.parse_failures):
                raise ValueError("an absent component root carries nothing further")

    @property
    def address_resolved(self) -> bool:
        """True iff the root ADDRESS resolved.  This is NOT a claim that the whole
        OwnerGraph is resolved — check ``graph.root.unresolved`` for that."""
        return self.status == "resolved"


def resolve_component_root(index, bundle, component_key, *,
                           root_param_prefixes=None):
    """Bridge a ``SourceBundle`` component address to an exact root occurrence.

    Address law (U3-D0):

    1. The declared architecture is read ONLY from
       ``bundle.component_architectures[component_key]``; ``bundle.architecture``
       is a lawful compatibility address for the ``root`` component only.
    2. HIDDEN-RIVAL LAW: an unreadable file in this component can hide another
       exact class definition, so uniqueness is unprovable while any indexed
       parse/read failure exists for the component.  Such failures are collected
       BEFORE the candidate count and return ``failed`` ahead of
       resolved/ambiguous/absent.
    3. Otherwise resolution searches ONLY ``ProgramIndex`` classes whose
       ``SourceId.component_key`` equals the requested component and matches the
       qualified class name EXACTLY (an address, not a mechanism claim).
    4. Exactly one candidate resolves; zero is ``absent``; more than one is
       ``ambiguous`` with every rival kept in a canonical, file-order-independent
       order.
    5. Never uses model_type, suffixes, substrings, shortest name, field count,
       file order, import order or role markers.
    """
    architectures = getattr(bundle, "component_architectures", None) or {}
    declared = architectures.get(component_key)
    if declared is None and component_key == "root":
        declared = getattr(bundle, "architecture", None)   # root-only compat
    if not declared:
        # empty / absent architecture never "picks the only class"
        return ComponentRootResolution(
            status="absent", component_key=component_key,
            declared_architecture=declared)

    # Hidden-rival law: a parse/read failure anywhere in this component makes
    # uniqueness unprovable (a broken file could define another exact class), so
    # fail before counting visible candidates.  Sorted canonically so bundle file
    # iteration order never leaks into the failed result.
    component_failures = tuple(sorted((
        failure for failure in index.parse_failures
        if failure.source.component_key == component_key),
        key=_parse_failure_sort_key))
    if component_failures:
        return ComponentRootResolution(
            status="failed", component_key=component_key,
            declared_architecture=declared, parse_failures=component_failures)

    candidates = tuple(sorted((
        ComponentRootCandidate(component_key, declared, record.symbol, record.span)
        for record in index.classes
        if record.symbol.source.component_key == component_key
        and record.symbol.qualified_name == declared),
        key=_candidate_sort_key))

    if len(candidates) == 1:
        graph = resolve_owner_graph(index, candidates[0].symbol,
                                    root_param_prefixes=root_param_prefixes)
        return ComponentRootResolution(
            status="resolved", component_key=component_key,
            declared_architecture=declared,
            occurrence=graph.root.occurrence, graph=graph)
    if len(candidates) >= 2:
        return ComponentRootResolution(
            status="ambiguous", component_key=component_key,
            declared_architecture=declared, candidates=candidates)
    return ComponentRootResolution(
        status="absent", component_key=component_key,
        declared_architecture=declared)


# --------------------------------------------------------------------------- #
# U3-B1 — the declared model-stage address boundary
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FrameworkAddressProtocol:
    """A closed, code-declared framework model-stage ADDRESS contract.  The
    declaration attribute names WHERE a wrapper binds its base/model-stage
    submodule (transformers' ``base_model_prefix = "model"``) — ADDRESS metadata
    only, proving no mechanism.  This registry is CODE, not data: adding a
    protocol changes which occurrence becomes architectural authority, so a new
    protocol must arrive with its own semantics and poison tests, never a mere
    config/YAML edit."""

    declaration_attr: str

    def __post_init__(self) -> None:
        if not self.declaration_attr:
            raise ValueError("a framework address protocol needs a declaration attribute")


# The closed registry of framework model-stage address protocols.
_FRAMEWORK_ADDRESS_PROTOCOLS: tuple = (
    FrameworkAddressProtocol("base_model_prefix"),
)


@dataclass(frozen=True)
class ModelStageDeclaration:
    """One exact class-body model-stage ADDRESS declaration
    (``base_model_prefix = "<attr>"``) proven at an exact declaring class —
    directly on the root or through exact inheritance.  Address metadata only.

    ``proof_trace`` is the exact MRO-prefix / precedence trace
    (root -> ... -> declaring_class) that made this declaration decisive; in a
    diamond its pre-declaring elements are C3-order predecessors, NOT necessarily a
    parent-child chain.  ``precedence_basis`` names WHY it is decisive
    (``root-direct`` | ``first-base-direct`` | ``c3``).  Together with ``span``
    they are the resolver's proof that the lookup is exact, never guessed."""

    declaring_class: SymbolId
    attribute: str                # the declared literal (may be "" for self)
    span: SourceSpan
    inherited: bool
    proof_trace: tuple[SymbolId, ...] = ()   # MRO-prefix / precedence trace
    precedence_basis: str = ""

    def __post_init__(self) -> None:
        # Provenance closure: a declaration must PROVE where it came from.  The
        # proof_trace is an MRO-PREFIX (precedence) trace root -> ... ->
        # declaring_class; in a diamond the elements before declaring_class are the
        # C3-order predecessors, NOT necessarily a parent-child chain.
        if not self.proof_trace:
            raise ValueError("a model-stage declaration must carry a non-empty proof trace")
        if self.proof_trace[-1] != self.declaring_class:
            raise ValueError("a proof trace must end at the declaring class")
        if self.span is None or self.span.source != self.declaring_class.source:
            raise ValueError("the declaration span must live in the declaring class's source")
        if self.precedence_basis not in {"root-direct", "first-base-direct", "c3"}:
            raise ValueError(f"unknown precedence basis {self.precedence_basis!r}")
        if self.precedence_basis == "root-direct":
            if len(self.proof_trace) != 1:
                raise ValueError("a root-direct trace is exactly the root itself")
            if self.inherited:
                raise ValueError("a root-direct declaration is not inherited")
        else:
            # first-base-direct / c3 name a declaration on an inherited base.
            if not self.inherited:
                raise ValueError(f"a {self.precedence_basis} declaration is inherited")


@dataclass(frozen=True)
class DeclaredModelStageResolution:
    """Typed outcome of resolving the declared model-stage occurrence against the
    authoritative root ``OwnerGraph``.

    ``resolved`` carries the EXACT existing child occurrence from the graph (or
    the root itself only on an explicit empty-prefix self proof); ``ambiguous``
    preserves >=2 rival declarations or >=2 complete rival occurrences with
    spans; ``absent`` is a missing declaration or a declaration pointing to no
    constructed field; ``failed`` is broken source / unresolved inheritance /
    dynamic declaration / unsupported or ambiguous construction / external
    stage / unavailable framework contract.
    """

    status: str                   # resolved | absent | ambiguous | failed
    root: SymbolId
    attribute: str | None = None
    declaration: ModelStageDeclaration | None = None
    occurrence: OwnerOccurrenceId | None = None
    self_stage: bool = False
    rival_declarations: tuple[ModelStageDeclaration, ...] = ()
    rival_occurrences: tuple[OwnerOccurrenceId, ...] = ()
    rival_owners: tuple[OwnerRival, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "absent", "ambiguous", "failed"}:
            raise ValueError(f"unknown model-stage status {self.status!r}")
        if not isinstance(self.root, SymbolId):
            raise TypeError("model-stage resolution requires a root SymbolId")
        root_occurrence = OwnerOccurrenceId(self.root)
        # ---- typed membership + cross-root closure over EVERY rival channel ---- #
        for rival in self.rival_declarations:
            if not isinstance(rival, ModelStageDeclaration):
                raise TypeError("rival_declarations must be ModelStageDeclaration values")
            if rival.proof_trace[0] != self.root:
                raise ValueError("a rival declaration's proof trace must begin at the root")
        for rival in self.rival_occurrences:
            if not isinstance(rival, OwnerOccurrenceId):
                raise TypeError("rival_occurrences must be complete OwnerOccurrenceId values")
            if rival.root != self.root:
                raise ValueError("a rival occurrence must be rooted at the requested root")
            if not rival.sites:
                raise ValueError("a rival occurrence carries a non-empty child chain")
        for rival in self.rival_owners:
            if not isinstance(rival, OwnerRival):
                raise TypeError("rival_owners must be authoritative OwnerRival records")
            if rival.parent != root_occurrence:
                raise ValueError("a rival owner must attach to the exact root occurrence")
        # ---- carried declaration provenance (any status that carries one) ------ #
        if self.declaration is not None and self.declaration.proof_trace[0] != self.root:
            raise ValueError("a carried declaration's proof trace must begin at the root")
        if (self.declaration is not None and self.attribute is not None
                and self.attribute != self.declaration.attribute):
            raise ValueError("declaration and attribute must agree when both are present")
        # ---- failure fields are inseparable ------------------------------------ #
        if self.failure_detail and not self.failure_kind:
            raise ValueError("a failure detail requires a failure kind")
        # ---- self_stage is legal ONLY for a resolved self-stage ---------------- #
        if self.self_stage and self.status != "resolved":
            raise ValueError("self_stage is legal only for a resolved self-stage")

        if self.status == "resolved":
            if self.declaration is None:
                raise ValueError("a resolved model stage carries its declaration")
            if self.attribute != self.declaration.attribute:
                raise ValueError("the resolved attribute must equal the declaration attribute")
            if (self.rival_declarations or self.rival_occurrences
                    or self.rival_owners or self.failure_kind):
                raise ValueError("a resolved model stage carries no rivals or failure")
            if self.occurrence is None:
                raise ValueError("a resolved model stage carries an occurrence")
            if self.self_stage:
                if self.attribute != "":
                    raise ValueError("a self-stage resolution requires an explicit empty-prefix proof")
                if self.occurrence != root_occurrence:
                    raise ValueError("a self-stage occurrence is exactly OwnerOccurrenceId(root)")
            else:
                if not self.occurrence.sites:
                    raise ValueError("a non-self model-stage occurrence is a non-empty child chain")
                if self.occurrence.root != self.root:
                    raise ValueError("the occurrence must be rooted at the requested root")
        elif self.status == "ambiguous":
            if (len(self.rival_declarations) < 2 and len(self.rival_occurrences) < 2
                    and len(self.rival_owners) < 2):
                raise ValueError("an ambiguous model stage preserves >=2 rival "
                                 "declarations, occurrences, or owner records")
            if self.occurrence is not None or self.failure_kind:
                raise ValueError("an ambiguous model stage carries no resolved occurrence or failure")
            # occurrence-side ambiguity must name the applicable declaration.
            if (self.rival_occurrences or self.rival_owners) and self.declaration is None:
                raise ValueError("occurrence-side ambiguity carries its applicable declaration")
        elif self.status == "failed":
            if not self.failure_kind:
                raise ValueError("a failed model stage carries a typed failure kind")
            if (self.occurrence is not None or self.rival_declarations
                    or self.rival_occurrences or self.rival_owners):
                raise ValueError("a failed model stage carries no occurrence or rivals")
        else:  # absent
            if (self.occurrence is not None or self.rival_declarations
                    or self.rival_occurrences or self.rival_owners or self.failure_kind):
                raise ValueError("an absent model stage carries nothing further")

    @property
    def address_resolved(self) -> bool:
        return self.status == "resolved"


def resolve_declared_model_stage(index: ProgramIndex,
                                 root_resolution,
                                 ) -> DeclaredModelStageResolution:
    """Resolve the model-stage occurrence a root DECLARES via a closed framework
    address protocol (``base_model_prefix``), matched against the ALREADY-RESOLVED
    root ``OwnerGraph`` — never manufacturing an occurrence.

    B1 consumes either a resolved D0 declaration or a closed
    :class:`ConstructedComponentRoot`; both routes already proved component
    isolation and hidden-rival handling.  A resolved result carries the exact
    existing graph child occurrence (``graph.node_for`` returns it).  Uses ONLY
    the code-declared literal + exact reference binding through inheritance +
    the graph's construction occurrences: never class names, model types, role
    vocabulary, embedding/layer/norm evidence, call ordering, return flow, a
    most-plausible-child heuristic, or a family table."""
    root_resolution = require_resolved_component_root(
        root_resolution, caller="resolve_declared_model_stage")
    graph = root_resolution.graph
    root_node = graph.root
    root = root_node.symbol
    # D0's own closure guarantees graph.root.occurrence has an empty site chain and
    # names the indexed declared-architecture class; B1 never re-derives the root.

    attrs = frozenset(p.declaration_attr for p in _FRAMEWORK_ADDRESS_PROTOCOLS)
    effective, decl_status, payload = _resolve_stage_declaration(index, root, attrs)
    # Declaration resolution is a TOTAL Python precedence order (exact C3 / provable
    # lazy shortcut): it yields resolved / failed / absent, never a declaration-side
    # tie.  Occurrence-side rivals (below) are the only ambiguity.
    if decl_status == "failed":
        return DeclaredModelStageResolution(
            "failed", root, failure_kind=payload[0], failure_detail=payload[1])
    if decl_status == "absent":
        return DeclaredModelStageResolution("absent", root)

    attribute = effective.attribute
    if attribute == "":
        # explicit self-fallback: base_model_prefix == "" declares the class IS
        # its own base (getattr(self, "", self) -> self).
        return DeclaredModelStageResolution(
            "resolved", root, attribute="", declaration=effective,
            occurrence=OwnerOccurrenceId(root), self_stage=True)

    # Match the declared attribute against the AUTHORITATIVE graph.  Every branch
    # below is EXHAUSTIVE over the graph's own occurrence facts: a resolved child,
    # a preserved conflict, or a typed unresolved entry.  `absent` is reachable
    # ONLY when the field has NO resolved child, NO unresolved entry, and NO owner
    # conflict — a matching unresolved entry may never degrade to absent.
    resolved_children = [c for c in root_node.children if c.via_field == attribute]
    unresolved_here = [u for u in root_node.unresolved if u.field == attribute]
    field_site_ids = {s.site_id for s in index.construction_sites_of(root)
                      if s.target == attribute and s.target_kind == "field"}
    field_site_ids |= {u.site for u in unresolved_here if u.site is not None}
    # Authoritative rivals already preserved in graph.conflicts (never fabricated):
    # an OwnerRival attaches to THIS field's root occurrence and construction site.
    field_rivals = tuple(
        rival
        for conflict in graph.conflicts if conflict.kind == "rival_owner_chain"
        for rival in conflict.rivals
        if rival.parent == root_node.occurrence and rival.site in field_site_ids)

    # >=2 resolved children: rival REAL occurrences (each round-trips via node_for).
    if len(resolved_children) >= 2:
        occurrences = tuple(c.occurrence for c in resolved_children
                            if graph.node_for(c.occurrence) is not None)
        return DeclaredModelStageResolution(
            "ambiguous", root, attribute=attribute, declaration=effective,
            rival_occurrences=occurrences)

    # Any matching unresolved entry -> never absent; classify EXHAUSTIVELY.
    if unresolved_here:
        kinds = {u.kind for u in unresolved_here}
        if kinds & {"rival_owner", "ambiguous_import"}:
            if len(field_rivals) >= 2:
                # ambiguity with the authoritative OwnerRival records preserved.
                return DeclaredModelStageResolution(
                    "ambiguous", root, attribute=attribute, declaration=effective,
                    rival_owners=field_rivals)
            # rivals exist but are not attributable to an exact root occurrence
            # (e.g. helper-return chains) or were not preserved (ambiguous import):
            # typed failure, never a fabricated occurrence, never absent.
            return DeclaredModelStageResolution(
                "failed", root, attribute=attribute, declaration=effective,
                failure_kind="unresolved_construction",
                failure_detail=f"self.{attribute} has rival constructions without an "
                               f"exact authoritative occurrence")
        if "dynamic" in kinds:
            return DeclaredModelStageResolution(
                "failed", root, attribute=attribute, declaration=effective,
                failure_kind="unsupported_construction",
                failure_detail=f"self.{attribute} is constructed dynamically")
        if "external" in kinds:
            return DeclaredModelStageResolution(
                "failed", root, attribute=attribute, declaration=effective,
                failure_kind="external_model_stage",
                failure_detail=f"self.{attribute} constructs an external/symbol-less child")
        # depth_limit / cycle / unbindable / any UNKNOWN future kind -> typed
        # failure, never absent.
        return DeclaredModelStageResolution(
            "failed", root, attribute=attribute, declaration=effective,
            failure_kind="unresolved_construction",
            failure_detail=f"self.{attribute} is unresolved ({sorted(kinds)})")

    # A preserved owner conflict for the field but no unresolved entry (defensive):
    # still never absent.
    if len(field_rivals) >= 2:
        return DeclaredModelStageResolution(
            "ambiguous", root, attribute=attribute, declaration=effective,
            rival_owners=field_rivals)

    if len(resolved_children) == 1:
        return DeclaredModelStageResolution(
            "resolved", root, attribute=attribute, declaration=effective,
            occurrence=resolved_children[0].occurrence)

    # No resolved child, no unresolved entry, no owner conflict: legally absent.
    return DeclaredModelStageResolution(
        "absent", root, attribute=attribute, declaration=effective)


class _MROIncomplete(Exception):
    """The exact C3 linearization required to decide precedence needs a class the
    index does not (yet) contain, so precedence is unprovable."""


def _resolve_stage_declaration(index, root, attrs):
    """Resolve the effective model-stage declaration for ``root`` from the CLOSED
    protocol attrs using LAZY, EXACT precedence — never a guessed order.  Returns
    (declaration|None, status, payload): ``resolved`` -> the declaration with a
    proof trace; ``failed`` -> (kind, detail); ``absent`` -> None.

    Precedence (each step is an EXACT Python-semantics fact, applied only when
    provable from the indexed closure):
      1. the root's OWN class-body declaration is decisive (final in-class
         assignment wins; a decisive dynamic assignment fails);
      2. a DIRECTLY declaring class is decisive before its ancestors are inspected;
      3. a direct declaration on the FIRST exactly-bound base is decisive over any
         later base (C3 places the first direct base at MRO position 1);
      4. otherwise the exact C3 linearization decides — computed ONLY when its
         required closure is fully indexed;
      5. an unresolved earlier base that can affect the lookup -> failed
         (``mro_incomplete``); such a base is NEVER skipped to reach a later
         declaration."""
    value, span, dynamic = _own_declaration(index, root, attrs)
    if dynamic:
        return None, "failed", ("dynamic_declaration",
                                f"{root.qualified_name} declares a non-literal model-stage address")
    if value is not None:
        return (ModelStageDeclaration(root, value, span, False,
                                      proof_trace=(root,), precedence_basis="root-direct"),
                "resolved", None)

    bases = _direct_bindings(index, root)
    if bases:
        first_display, first_binding = bases[0]
        if first_binding is None:
            return None, "failed", ("mro_incomplete",
                                    f"earliest base {first_display!r} is unresolved and can affect precedence")
        if first_binding is _RIVAL_BASE:
            return None, "failed", ("mro_incomplete",
                                    f"earliest base {first_display!r} binds rival exact classes")
        value, span, dynamic = _own_declaration(index, first_binding, attrs)
        if value is not None or dynamic:
            # The first-base shortcut relies on C3 placing the first direct base at
            # MRO position 1.  When the WHOLE closure is indexed we can prove the
            # hierarchy is C3-consistent; an INVALID fully-indexed hierarchy (one
            # Python itself would reject) must fail, not resolve.  When the closure
            # is incomplete (external bases such as GenerationMixin), we rely on the
            # runtime-valid-class premise: the class exists at runtime, so Python
            # already computed a consistent MRO for it, and the first direct base is
            # necessarily first among the bases.
            if _closure_fully_indexed(index, root):
                try:
                    _c3_linearization(index, root, set())
                except _MROIncomplete as exc:
                    return None, "failed", ("mro_incomplete", str(exc))
            if dynamic:
                return None, "failed", ("dynamic_declaration",
                                        f"{first_binding.qualified_name} declares a non-literal model-stage address")
            return (ModelStageDeclaration(first_binding, value, span, True,
                                          proof_trace=(root, first_binding),
                                          precedence_basis="first-base-direct"),
                    "resolved", None)

    # The lazy shortcuts did not decide.  Fall back to the EXACT C3 linearization,
    # which requires the full closure to be indexed.
    try:
        mro = _c3_linearization(index, root, set())
    except _MROIncomplete as exc:
        return None, "failed", ("mro_incomplete", str(exc))
    for offset, cls in enumerate(mro):
        if cls == root:
            continue                          # root already proven non-declaring
        value, span, dynamic = _own_declaration(index, cls, attrs)
        if dynamic:
            return None, "failed", ("dynamic_declaration",
                                    f"{cls.qualified_name} declares a non-literal model-stage address")
        if value is not None:
            return (ModelStageDeclaration(cls, value, span, True,
                                          proof_trace=tuple(mro[:offset + 1]),
                                          precedence_basis="c3"),
                    "resolved", None)
    return None, "absent", None


def _own_declaration(index, symbol, attrs):
    """The class's OWN-body effective model-stage declaration: the LAST assignment
    of a protocol attr in class-body (source) order.  Returns (literal|None,
    span|None, is_dynamic)."""
    record = index.class_by_symbol(symbol)
    if record is None:
        return None, None, False
    assigns = [ba for ba in record.body_assigns if ba.attr in attrs]
    if not assigns:
        return None, None, False
    last = max(assigns, key=lambda ba: (ba.span.line if ba.span else 0,
                                        ba.span.col if ba.span else 0))
    value = last.value
    if value.kind == "constant" and isinstance(value.const_value, str):
        return value.const_value, last.span, False
    return None, last.span, True


# Sentinel for a direct base whose reference binds >=2 rival exact classes: the
# exact binding is ambiguous, so precedence over it is unprovable.
_RIVAL_BASE = object()


def _direct_bindings(index, symbol):
    """The direct bases of ``symbol`` in listed (Python precedence) order, each
    bound EXACTLY.  Returns [(display, SymbolId | None | _RIVAL_BASE), ...]; a
    None binding is an unresolved base, ``_RIVAL_BASE`` a rival binding.  An
    unbindable base EXPRESSION is reported as unresolved (it can still affect the
    MRO, so it must never be silently skipped)."""
    record = index.class_by_symbol(symbol)
    out: list = []
    if record is None:
        return out
    for base in record.bases:
        reference = _base_reference(base)
        if reference is None:
            out.append(("<unbindable base>", None))
            continue
        bindings = _resolve_base_binding(index, symbol.source, reference)
        display = _reference_display(reference)
        if not bindings:
            out.append((display, None))
        elif len(bindings) >= 2:
            out.append((display, _RIVAL_BASE))
        else:
            out.append((display, bindings[0]))
    return out


def _closure_fully_indexed(index, root) -> bool:
    """True iff every class in ``root``'s transitive inheritance closure is indexed
    and every base binds to exactly one indexed class — the precondition for
    computing (and validating) the exact C3 linearization."""
    seen: set = set()
    stack = [root]
    while stack:
        symbol = stack.pop()
        if symbol in seen:
            continue
        seen.add(symbol)
        if index.class_by_symbol(symbol) is None:
            return False
        for _display, binding in _direct_bindings(index, symbol):
            if binding is None or binding is _RIVAL_BASE:
                return False
            stack.append(binding)
    return True


def _c3_linearization(index, symbol, stack):
    """The EXACT C3 linearization (Python MRO) of ``symbol`` as SymbolIds.  Raises
    :class:`_MROIncomplete` if any class in the required closure is not indexed,
    a base binds rival exact classes, a base expression is unbindable, the
    hierarchy is cyclic, or no consistent linearization exists — every case in
    which the true Python order cannot be proven from the index."""
    if symbol in stack:
        raise _MROIncomplete(f"cyclic inheritance at {symbol.qualified_name!r}")
    record = index.class_by_symbol(symbol)
    if record is None:
        raise _MROIncomplete(f"class {symbol.qualified_name!r} is not indexed")
    base_symbols: list = []
    for display, binding in _direct_bindings(index, symbol):
        if binding is None:
            raise _MROIncomplete(f"base {display!r} of {symbol.qualified_name!r} is unresolved")
        if binding is _RIVAL_BASE:
            raise _MROIncomplete(f"base {display!r} of {symbol.qualified_name!r} binds rival exact classes")
        base_symbols.append(binding)
    sequences = [_c3_linearization(index, base, stack | {symbol}) for base in base_symbols]
    sequences.append(list(base_symbols))
    return [symbol] + _c3_merge(sequences)


def _c3_merge(sequences):
    """The C3 merge: repeatedly take a head that appears in no sequence's tail."""
    sequences = [list(seq) for seq in sequences if seq]
    result: list = []
    while sequences:
        for seq in sequences:
            head = seq[0]
            if not any(head in later[1:] for later in sequences):
                break
        else:
            raise _MROIncomplete("no consistent C3 linearization")
        result.append(head)
        sequences = [[c for c in seq if c != head] for seq in sequences]
        sequences = [seq for seq in sequences if seq]
    return result


def _base_reference(expr):
    """The Name/Attribute reference a base expression binds to (unwrapping a
    ``Generic[...]``-style subscript), or None for a base we cannot bind."""
    if expr is None:
        return None
    if expr.kind in ("name", "attribute"):
        return expr
    if expr.kind == "subscript" and expr.children:
        return _base_reference(expr.children[0])
    return None


def _resolve_base_binding(index, source, reference):
    """Bind a base reference EXACTLY, relative to its declaring source: a class of
    that name in the SAME source, else an exact import binding (alias -> import
    target -> the class in the addressed module).  Never a global qualified-name
    search; returns >=2 SymbolIds only for genuine rival bindings."""
    chain = _attribute_chain(reference)
    if not chain:
        return ()
    if len(chain) == 1:
        local = tuple(record.symbol for record in index.classes
                      if record.symbol.source == source
                      and record.symbol.qualified_name == chain[0])
        if local:
            return local
    alias, *attributes = chain
    matches: list = []
    for record in index.imports:
        if record.source != source or record.alias != alias:
            continue
        parts = tuple(part for part in record.target.lstrip(".").split(".") if part)
        parts = parts + tuple(attributes)
        if not parts:
            continue
        class_name = parts[-1]
        module_name = parts[-2] if len(parts) >= 2 else ""
        for class_record in index.classes:
            if class_record.symbol.qualified_name == class_name and (
                    not module_name
                    or _module_stem(class_record.symbol.source.canonical_path) == module_name):
                matches.append(class_record.symbol)
    return tuple(dict.fromkeys(matches))


def _reference_display(reference) -> str:
    chain = _attribute_chain(reference)
    return ".".join(chain) if chain else (reference.name or reference.kind)


__all__ = [
    "ConfigPrefix", "OwnerOccurrenceId", "ConfigBinding", "OwnerRival",
    "ConfigPrefixRival", "UnresolvedChild", "OwnerNode", "OwnerGraph",
    "ConstructedComponentRoot", "require_resolved_component_root",
    "resolve_owner_graph", "resolve_construction_candidate_symbols",
    "ComponentRootCandidate", "ComponentRootResolution", "resolve_component_root",
    "FrameworkAddressProtocol", "ModelStageDeclaration",
    "DeclaredModelStageResolution", "resolve_declared_model_stage",
]
