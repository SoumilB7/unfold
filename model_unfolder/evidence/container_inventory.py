"""U3-B2 — neutral container address inventory.

Enumerates every :class:`ContainerElementsRecord` owned by an EXACT resolved owner
occurrence (normally B1's model-stage occurrence, not merely the D0 root).  B2
consumes a RESOLVED :class:`ComponentRootResolution` (D0) — inheriting D0's
component isolation and hidden-rival / parse-failure law — and an EXPLICIT owner
occurrence.  B2 NEVER selects an owner: there is no B1-failure→root fallback; a
caller that wants the diffusion root must pass it explicitly.

The inventory is strictly NEUTRAL and observation-only:

- it NEVER calls a container a "layer stack", "block", or any role; container kind
  is OBSERVED SYNTAX ONLY (the lowercased constructor: ``modulelist`` |
  ``sequential`` | ``moduledict`` | ...);
- it preserves the AUTHORITATIVE :class:`ContainerElementsRecord` and each original
  :class:`ConstructionSite` / ``ConstructionSiteId`` — it never rebuilds lossy
  element identities;
- multiple container records targeting the SAME exact owner+field (e.g. guarded
  construction) are grouped as typed :class:`ContainerRival` s; B2 never picks one.
  Each authoritative record and its spans are preserved, and guards SURVIVE ONLY
  where ProgramIndex emitted element sites (element-site guards).  The
  container-ASSIGNMENT guard (the branch around ``self.x = ModuleList(...)``) is
  not carried by ContainerElementsRecord and is NOT yet available — that guard
  completeness is a U3-A2 gap, not something B2 claims;
- it emits SOURCE order only (``source_order``), NEVER execution order;
- it CITES a count's :class:`ConfigPathObservation` only when a recorded
  observation in the SAME callable EXACTLY matches an attribute chain in the count
  AND its span lies INSIDE the count expression's span; a sibling occurrence is
  never acceptable.  If the current ProgramIndex cannot emit a count-contained
  observation, the citation is ``None`` (it does not compensate);
- it reads no config VALUE and never touches the U1 ledger; it infers no role,
  migrates no reader, changes no rendering.

Element-coverage honesty: element extraction follows ProgramIndex exactly.  Direct
positional arguments to a container constructor — ``Sequential(A(), B())`` — are
NOT currently emitted as elements by ProgramIndex, so ``record.elements`` can be
empty even when the source lists elements.  B2 reports the record honestly and
makes NO claim of complete element coverage.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .program_index import (
    ConfigPathObservation,
    ConstructionSite,
    ContainerElementsRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)


def _element_child(site: ConstructionSite) -> SymbolId | None:
    """The exact bound child class of an element site, only when a single candidate
    carries a resolved symbol; otherwise None (dynamic / symbol-less / rivals)."""
    if len(site.candidates) == 1 and site.candidates[0].symbol is not None:
        return site.candidates[0].symbol
    return None


def _span_within(inner: SourceSpan | None, outer: SourceSpan | None) -> bool:
    """True iff ``inner`` lies textually inside ``outer`` (same source, start >=
    outer start, end <= outer end)."""
    if inner is None or outer is None or inner.source != outer.source:
        return False
    inner_end = (inner.end_line or inner.line, inner.end_col or inner.col)
    outer_end = (outer.end_line or outer.line, outer.end_col or outer.col)
    return ((inner.line, inner.col) >= (outer.line, outer.col)
            and inner_end <= outer_end)


@dataclass(frozen=True)
class ContainerAddress:
    """One container record owned by the resolved owner occurrence, carrying the
    AUTHORITATIVE :class:`ContainerElementsRecord` (original ConstructionSites +
    ConstructionSiteIds preserved).  Observed syntax + source order only; no roles,
    no execution order.  See the module docstring on incomplete element coverage
    for direct constructor arguments."""

    owner_occurrence: OwnerOccurrenceId
    record: ContainerElementsRecord
    source_order: int
    count_config_path: ConfigPathObservation | None = None

    @property
    def field(self) -> str:
        return self.record.field

    @property
    def syntactic_kind(self) -> str:
        return self.record.kind

    @property
    def count_expression(self):
        return self.record.count

    @property
    def element_sites(self) -> tuple:
        return self.record.elements

    @property
    def unresolved_sites(self) -> tuple:
        return tuple(s for s in self.record.elements if _element_child(s) is None)

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("a container address is anchored at an OwnerOccurrenceId")
        if not isinstance(self.record, ContainerElementsRecord):
            raise TypeError("a container address carries the authoritative record")
        record = self.record
        sources = {record.owner.source, record.enclosing_callable.source}
        if record.span is not None:
            sources.add(record.span.source)
        if len(sources) != 1:
            raise ValueError("record owner / callable / span must share a source")
        for site in record.elements:
            if site.owner != record.owner or site.site_id.owner != record.owner:
                raise ValueError("every element site belongs to the exact owner symbol")
        if not record.kind:
            raise ValueError("a container carries its observed syntactic kind")
        if self.source_order < 0:
            raise ValueError("source order is a non-negative source position")
        if self.count_config_path is not None:
            if record.count is None:
                raise ValueError("a count citation requires a count expression")
            obs = self.count_config_path
            if obs.enclosing_callable != record.enclosing_callable:
                raise ValueError("a count citation belongs to the same callable")
            if not _span_within(obs.span, record.count.span):
                raise ValueError("a count citation lies inside the count expression span")
            root = (obs.root_binding.name
                    if obs.root_binding is not None and obs.root_binding.kind == "name"
                    else None)
            if root is None:
                raise ValueError("a count citation is rooted at a config-root name")
            if any(seg.dynamic for seg in obs.segments):
                raise ValueError("a count citation carries only exact non-dynamic segments")
            segments = tuple(seg.name for seg in obs.segments)
            if (root, segments) not in _config_chains(record.count):
                raise ValueError("a count citation path must appear structurally inside the count")


@dataclass(frozen=True)
class ContainerRival:
    """>=2 container records target the SAME exact owner+field (e.g. guarded
    construction): typed rivals with each authoritative record preserved (its
    spans, and the element-site guards ProgramIndex emitted, intact).  B2 never
    picks one.  The container-ASSIGNMENT guard (the branch condition around the
    ``self.x = ...`` statement) is NOT carried by ContainerElementsRecord — that
    guard completeness is a U3-A2 gap, not something this type claims."""

    owner_occurrence: OwnerOccurrenceId
    field: str
    records: tuple[ContainerElementsRecord, ...]
    source_order: int

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("a container rival is anchored at an OwnerOccurrenceId")
        if len(self.records) < 2:
            raise ValueError("a container rival preserves >=2 rival records")
        if any(not isinstance(r, ContainerElementsRecord) for r in self.records):
            raise TypeError("rival records must be ContainerElementsRecord values")
        if {r.field for r in self.records} != {self.field}:
            raise ValueError("all rival records share the exact field")
        if len({r.owner for r in self.records}) != 1:
            raise ValueError("all rival records share the exact owner symbol")
        if self.source_order < 0:
            raise ValueError("source order is a non-negative source position")


@dataclass(frozen=True)
class ContainerInventory:
    """The neutral inventory of every container record owned by an EXACT resolved
    owner occurrence.  Source evidence only: no roles, no execution order, no config
    values, no U1 ledger consumption.

    ``resolved`` carries >=1 container and/or rival in source order; ``absent`` is a
    resolved owner with no container records; ``failed`` is an owner
    occurrence not present in the supplied graph."""

    status: str                  # resolved | absent | failed
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId | None = None
    containers: tuple[ContainerAddress, ...] = ()
    rivals: tuple[ContainerRival, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "absent", "failed"}:
            raise ValueError(f"unknown container-inventory status {self.status!r}")
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("a container inventory is anchored at an OwnerOccurrenceId")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("a failure detail requires a failure kind")
        # Every carried record is bound to the inventory's exact owner symbol, so a
        # cross-owner record (even one in the same source file) cannot be laundered
        # into this inventory.
        for container in self.containers:
            if not isinstance(container, ContainerAddress):
                raise TypeError("containers must be ContainerAddress values")
            if container.owner_occurrence != self.owner_occurrence:
                raise ValueError("every container is owned by the requested occurrence")
            if self.owner_symbol is not None and container.record.owner != self.owner_symbol:
                raise ValueError("every container record is owned by the inventory owner symbol")
        for rival in self.rivals:
            if not isinstance(rival, ContainerRival):
                raise TypeError("rivals must be ContainerRival values")
            if rival.owner_occurrence != self.owner_occurrence:
                raise ValueError("every rival is owned by the requested occurrence")
            if self.owner_symbol is not None and any(
                    r.owner != self.owner_symbol for r in rival.records):
                raise ValueError("every rival record is owned by the inventory owner symbol")
        if self.status == "resolved":
            if self.owner_symbol is None:
                raise ValueError("a resolved inventory names its owner symbol")
            if not (self.containers or self.rivals):
                raise ValueError("a resolved inventory carries >=1 container or rival")
            if self.failure_kind:
                raise ValueError("a resolved inventory carries no failure")
            orders = ([c.source_order for c in self.containers]
                      + [r.source_order for r in self.rivals])
            if sorted(orders) != list(range(len(orders))):
                raise ValueError("containers and rivals occupy a strict 0..n source order")
        elif self.status == "absent":
            if self.owner_symbol is None:
                raise ValueError("an absent inventory still names its resolved owner symbol")
            if self.containers or self.rivals or self.failure_kind:
                raise ValueError("an absent inventory carries nothing further")
        else:  # failed
            if not self.failure_kind:
                raise ValueError("a failed inventory carries a typed failure kind")
            if self.containers or self.rivals:
                raise ValueError("a failed inventory carries no containers or rivals")


def resolve_container_inventory(index: ProgramIndex,
                                root_resolution: ComponentRootResolution,
                                owner_occurrence: OwnerOccurrenceId,
                                ) -> ContainerInventory:
    """Inventory every container record owned by ``owner_occurrence`` — an EXACT
    resolved owner occurrence in the RESOLVED D0 ``root_resolution``'s graph
    (normally B1's model-stage occurrence, not merely the D0 root).  B2 rejects a
    non-D0 argument (TypeError) and any non-resolved D0 result (ValueError),
    inheriting D0's component isolation + hidden-rival law; it never selects an
    owner and never falls back to the root.  Neutral and source-evidence-only (see
    the module docstring)."""
    if not isinstance(root_resolution, ComponentRootResolution):
        raise TypeError("resolve_container_inventory requires a ComponentRootResolution (D0)")
    if root_resolution.status != "resolved":
        raise ValueError(
            "resolve_container_inventory requires a RESOLVED component root; D0 "
            f"returned {root_resolution.status!r} (component isolation + hidden-rival law)")
    if not isinstance(owner_occurrence, OwnerOccurrenceId):
        raise TypeError("resolve_container_inventory requires an explicit OwnerOccurrenceId owner")

    graph = root_resolution.graph
    # The D0 resolution must have been built from THIS index: the graph root symbol
    # (with its exact SourceId content fingerprint) must resolve in `index`.  A D0
    # from a different ProgramIndex would otherwise silently enumerate mismatched
    # (or zero) container records.
    if index.class_by_symbol(graph.root.symbol) is None:
        return ContainerInventory(
            "failed", owner_occurrence, failure_kind="index_mismatch",
            failure_detail="the D0 resolution was built from a different ProgramIndex")

    node = graph.node_for(owner_occurrence)
    if node is None:
        return ContainerInventory(
            "failed", owner_occurrence, failure_kind="owner_not_in_graph",
            failure_detail="the owner occurrence is not an exact node of the resolved graph")

    # The resolved OWNER class (not only the graph root) must also exist in THIS
    # index: an index that has the identical wrapper file but omits the child file
    # would otherwise enumerate zero records and masquerade as `absent`.
    if index.class_by_symbol(node.symbol) is None:
        return ContainerInventory(
            "failed", owner_occurrence, failure_kind="index_mismatch",
            failure_detail="the owner class is absent from this ProgramIndex")

    owner_symbol = node.symbol
    by_field: dict = {}
    for record in index.containers:
        if record.owner == owner_symbol:
            by_field.setdefault(record.field, []).append(record)

    entries = []
    for field, records in by_field.items():
        records_sorted = sorted(records, key=_record_sort_key)
        entries.append((_record_sort_key(records_sorted[0]), field, records_sorted))
    entries.sort(key=lambda entry: entry[0])

    containers: list[ContainerAddress] = []
    rivals: list[ContainerRival] = []
    for source_order, (_key, field, records) in enumerate(entries):
        if len(records) == 1:
            record = records[0]
            containers.append(ContainerAddress(
                owner_occurrence, record, source_order,
                _matching_config_path(index, record)))
        else:
            rivals.append(ContainerRival(
                owner_occurrence, field, tuple(records), source_order))

    if not entries:
        return ContainerInventory("absent", owner_occurrence, owner_symbol)
    return ContainerInventory(
        "resolved", owner_occurrence, owner_symbol,
        tuple(containers), tuple(rivals))


def _record_sort_key(record):
    span = record.span
    return (span.line if span else 0, span.col if span else 0, record.field)


def _matching_config_path(index, record) -> ConfigPathObservation | None:
    """Cite the count's config path ONLY when a recorded ConfigPathObservation in
    the SAME callable exactly matches an attribute chain in the count AND its span
    lies INSIDE the count expression's span.  A sibling read (same path, span
    outside the count) is never acceptable.  When the current substrate emits no
    count-contained observation, returns None — it does not compensate."""
    count = record.count
    if count is None or count.span is None:
        return None
    chains = _config_chains(count)
    if not chains:
        return None
    matches: list = []
    for obs in index.config_paths_in(record.enclosing_callable):
        if not _span_within(obs.span, count.span):
            continue
        root = (obs.root_binding.name
                if obs.root_binding is not None and obs.root_binding.kind == "name"
                else None)
        if root is None or any(seg.dynamic for seg in obs.segments):
            continue
        segments = tuple(seg.name for seg in obs.segments)
        if (root, segments) in chains:
            matches.append(obs)
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _config_chains(expr) -> set:
    """Every ``<name>.<attr>...`` attribute chain (root Name + segment names) that
    appears structurally inside ``expr``.  Purely structural; reads no value."""
    found: set = set()

    def walk(node) -> None:
        if node is None:
            return
        if getattr(node, "kind", None) == "attribute":
            segments: list[str] = []
            current = node
            while current is not None and current.kind == "attribute":
                segments.append(current.name)
                current = current.children[0] if current.children else None
            if current is not None and current.kind == "name" and current.name:
                found.add((current.name, tuple(reversed(segments))))
        for child in getattr(node, "children", ()) or ():
            walk(child)
        for _key, child in getattr(node, "keyword_children", ()) or ():
            walk(child)

    walk(expr)
    return found


__all__ = [
    "ContainerAddress",
    "ContainerRival",
    "ContainerInventory",
    "resolve_container_inventory",
]
