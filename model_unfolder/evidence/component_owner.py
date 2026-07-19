"""U3-B — the ComponentOwner resolver over the raw ProgramIndex.

The ProgramIndex (U3-A) OBSERVES: it records 0 / 1 / several construction
candidates per site and never picks a winner.  This resolver RESOLVES: it walks
the construction graph from a root class, following the chain

    parent owner -> construction site -> field/slot -> child class

to build an owner tree, propagating the exact config-path prefix down each edge
(the ``projector._config_param_chains`` prototype, generalized and made per-
occurrence).  Where resolution cannot prove uniqueness it emits a TYPED
:class:`~.program_index.ConflictRecord` — never a silent drop, never a guess:

  * ``rival_owner_chain``   — one slot has >=2 candidate child classes (a
    registry dispatch the index kept as rivals);
  * ``rival_config_prefix`` — one construction passes a config argument that
    resolves to >=2 distinct prefixes (an ``if/else`` or ``a or b`` over config
    paths).

Boundary (Soumil 2026-07-19): conflicts are RECORDS carrying the rival chains,
the conflicting prefixes, the exact sites and spans.  The resolver reads the
index by ADDRESS (SymbolId) only; it performs no name/substring architectural
selection beyond resolving a construction REFERENCE to the class of that name
(cross-file only when that name is unique in the bundle).  This module migrates
no reader and changes no rendering — readers move onto it from U3-D.
"""
from __future__ import annotations

from dataclasses import dataclass

from .program_index import (
    ConflictRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)


@dataclass(frozen=True)
class UnresolvedChild:
    """A slot whose child the resolver could not prove uniquely."""

    field: str
    kind: str            # dynamic | external | ambiguous_crossfile | rival_owner |
    #                      cycle | depth_limit
    span: SourceSpan | None
    detail: str = ""


@dataclass(frozen=True)
class OwnerNode:
    """One resolved owner in the construction graph."""

    owner: SymbolId
    config_prefix: tuple          # config path from the ROOT config to this owner
    via_field: str                # the parent field/slot that constructed it ('' root)
    via_kind: str                 # root | field | element | return
    children: tuple = ()          # tuple[OwnerNode]
    unresolved: tuple = ()        # tuple[UnresolvedChild]


@dataclass(frozen=True)
class OwnerGraph:
    """The resolved owner tree + every typed conflict found while resolving."""

    root: OwnerNode
    conflicts: tuple = ()         # tuple[ConflictRecord]

    def walk(self):
        """Yield every OwnerNode depth-first (root first)."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def node_for(self, owner: SymbolId) -> OwnerNode | None:
        for node in self.walk():
            if node.owner == owner:
                return node
        return None


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

def resolve_owner_graph(index: ProgramIndex, root_symbol: SymbolId,
                        *, max_depth: int = 64) -> OwnerGraph:
    """Resolve the owner graph rooted at ``root_symbol`` from the raw index."""
    resolver = _Resolver(index, max_depth)
    root = resolver.build(root_symbol, config_prefix=(), via_field="",
                          via_kind="root",
                          param_prefixes=resolver.root_param_prefixes(root_symbol),
                          path=())
    return OwnerGraph(root=root, conflicts=tuple(resolver.conflicts))


class _Resolver:
    def __init__(self, index: ProgramIndex, max_depth: int):
        self.program_index = index
        self.max_depth = max_depth
        self.conflicts: list = []
        # class name -> the ClassRecord symbols defining it, for bounded cross-file
        # resolution of a construction reference (unique-name only).
        self._by_name: dict = {}
        for cr in index.classes:
            self._by_name.setdefault(cr.symbol.qualified_name, []).append(cr.symbol)

    # -- init/param helpers ------------------------------------------------- #

    def _init_of(self, class_symbol: SymbolId):
        return self.program_index.callable_by_symbol(
            SymbolId(class_symbol.source, f"{class_symbol.qualified_name}.__init__"))

    def _config_params(self, class_symbol: SymbolId) -> list:
        init = self._init_of(class_symbol)
        if init is None:
            return []
        return [p.name for p in init.params if p.name != "self"]

    def root_param_prefixes(self, root_symbol: SymbolId) -> dict:
        params = self._config_params(root_symbol)
        return {params[0]: ()} if params else {}

    # -- the recursive build ------------------------------------------------ #

    def build(self, owner_symbol, config_prefix, via_field, via_kind,
              param_prefixes, path) -> OwnerNode:
        if len(path) > self.max_depth:
            return OwnerNode(owner_symbol, config_prefix, via_field, via_kind,
                             unresolved=(UnresolvedChild(via_field, "depth_limit",
                                                         None),))
        if owner_symbol in path:                      # cycle guard
            return OwnerNode(owner_symbol, config_prefix, via_field, via_kind)
        path = path + (owner_symbol,)
        children: list = []
        unresolved: list = []
        for site in self._owner_sites(owner_symbol):
            self._process_site(site, owner_symbol, config_prefix, param_prefixes,
                               children, unresolved, path)
        return OwnerNode(owner_symbol, config_prefix, via_field, via_kind,
                         tuple(children), tuple(unresolved))

    def _owner_sites(self, owner_symbol: SymbolId) -> list:
        sites = list(self.program_index.construction_sites_of(owner_symbol))
        for cont in self.program_index.containers:
            if cont.owner == owner_symbol:
                sites.extend(cont.elements)
        # field + element sites; return-sites are reached only via a helper fold
        return [s for s in sites if s.target_kind in ("field", "element")]

    def _process_site(self, site, owner_symbol, config_prefix, param_prefixes,
                      children, unresolved, path) -> None:
        # helper fold: self.<field> = self._build(...) -> follow the helper's
        # return construction sites (the field's real children).
        method = self._self_helper(site, owner_symbol)
        if method is not None:
            msym = SymbolId(owner_symbol.source,
                            f"{owner_symbol.qualified_name}.{method}")
            ret_sites = [s for s in self.program_index.construction_sites
                         if s.enclosing_callable == msym
                         and s.target_kind == "return"]
            if not ret_sites:
                unresolved.append(UnresolvedChild(site.target, "external", site.span,
                                                  f"helper {method} not resolvable"))
                return
            for rs in ret_sites:
                self._resolve_and_recurse(rs, site.target, "return", owner_symbol,
                                          config_prefix, param_prefixes, children,
                                          unresolved, path)
            return
        self._resolve_and_recurse(site, site.target, site.target_kind or "field",
                                  owner_symbol, config_prefix, param_prefixes,
                                  children, unresolved, path)

    def _resolve_and_recurse(self, site, field, via_kind, owner_symbol,
                             config_prefix, param_prefixes, children, unresolved,
                             path) -> None:
        child_symbol, kind = self._resolve_child(site)
        if child_symbol is None:
            unresolved.append(UnresolvedChild(field, kind, site.span,
                                              _ref_detail(site)))
            if kind == "rival_owner":
                self.conflicts.append(ConflictRecord(
                    "rival_owner_chain",
                    tuple((c.reference.name, c.provenance) for c in site.candidates),
                    (site.span,)))
            return
        child_prefixes, rival = self._child_param_prefixes(
            site, child_symbol, param_prefixes, config_prefix)
        if rival is not None:
            self.conflicts.append(ConflictRecord(
                "rival_config_prefix", tuple(rival), (site.span,)))
        first = self._config_params(child_symbol)
        child_prefix = (child_prefixes.get(first[0], config_prefix)
                        if first else config_prefix)
        children.append(self.build(child_symbol, child_prefix, field, via_kind,
                                   child_prefixes, path))

    # -- child symbol resolution ------------------------------------------- #

    def _resolve_child(self, site):
        cands = site.candidates
        if len(cands) >= 2:
            return None, "rival_owner"
        if len(cands) == 1:
            c = cands[0]
            if c.symbol is not None:
                return c.symbol, "resolved"
            name = c.reference.name if c.reference.kind in ("name", "attribute") else None
            if name:
                matches = self._by_name.get(name, [])
                if len(matches) == 1:
                    return matches[0], "resolved_crossfile"
                if len(matches) >= 2:
                    return None, "ambiguous_crossfile"
            return None, "external"
        return None, "dynamic"

    def _self_helper(self, site, owner_symbol) -> str | None:
        """If the site's constructor is ``self.<method>(...)`` and <method> is a
        method of this owner, return the method name (a helper fold)."""
        ctor = site.constructor
        if ctor is None or ctor.kind != "call" or not ctor.children:
            return None
        callee = ctor.children[0]
        if callee.kind != "attribute" or not callee.children:
            return None
        recv = callee.children[0]
        if recv.kind != "name" or recv.name != "self":
            return None
        method = callee.name
        msym = SymbolId(owner_symbol.source,
                        f"{owner_symbol.qualified_name}.{method}")
        return method if self.program_index.callable_by_symbol(msym) is not None else None

    # -- config prefix propagation ----------------------------------------- #

    def _child_param_prefixes(self, site, child_symbol, param_prefixes,
                              config_prefix):
        child_params = self._config_params(child_symbol)
        out: dict = {}
        rival = None
        if site.via.startswith("factory:"):
            if site.args and child_params:
                pfx, rivals = _arg_prefix(site.args[0], param_prefixes)
                if rivals:
                    rival = rivals
                elif pfx is not None:
                    out[child_params[0]] = pfx
            return out, rival
        for i, arg in enumerate(site.args):
            if i >= len(child_params):
                break
            pfx, rivals = _arg_prefix(arg, param_prefixes)
            if rivals:
                rival = rivals
            elif pfx is not None:
                out[child_params[i]] = pfx
        for name, arg in site.kwargs:
            if name in child_params:
                pfx, rivals = _arg_prefix(arg, param_prefixes)
                if rivals:
                    rival = rivals
                elif pfx is not None:
                    out[name] = pfx
        return out, rival


def _ref_detail(site) -> str:
    if not site.candidates:
        return site.constructor.source_segment if site.constructor else ""
    return ", ".join(c.reference.name or c.reference.kind for c in site.candidates)


def _arg_prefix(expr, param_prefixes):
    """The config prefix a construction ARGUMENT resolves to, given the enclosing
    owner's ``param -> prefix`` map.  Returns ``(prefix|None, rivals)``; ``rivals``
    is a non-empty list of >=2 distinct prefixes when the argument is an
    ``if/else`` or ``a or b`` over DIFFERENT config paths (a rival_config_prefix)."""
    if expr is None:
        return None, []
    if expr.kind == "ifexp" and len(expr.children) == 3:
        a, _ = _arg_prefix(expr.children[0], param_prefixes)
        b, _ = _arg_prefix(expr.children[2], param_prefixes)
        branch = [p for p in (a, b) if p is not None]
        if len(set(branch)) >= 2:
            return None, branch
        return (branch[0] if branch else None), []
    if expr.kind == "boolop":
        prefixes = []
        for child in expr.children:
            p, _ = _arg_prefix(child, param_prefixes)
            if p is not None:
                prefixes.append(p)
        if len(set(prefixes)) >= 2:
            return None, prefixes
        return (prefixes[0] if prefixes else None), []
    # an attribute chain on a config-bound parameter: config.text_config.x
    segs: list = []
    cur = expr
    while cur is not None and cur.kind == "attribute":
        segs.append(cur.name)
        cur = cur.children[0] if cur.children else None
    if cur is not None and cur.kind == "name" and cur.name in param_prefixes:
        return (*param_prefixes[cur.name], *reversed(segs)), []
    return None, []


__all__ = [
    "UnresolvedChild", "OwnerNode", "OwnerGraph", "resolve_owner_graph",
]
