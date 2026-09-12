"""U11-B — exact U-shaped repeated-stage construction inventory.

This boundary connects the repeated root containers proven by U10 to the
constructor/factory calls that populate those exact containers.  It is an
address and construction-evidence reader only:

* it consumes a resolved D0 root and an exact positive U10 ``u_shaped`` proof;
* it follows an append only when the appended value is tied to one exact
  producer call in the root constructor;
* it expands only the exact called import through U11-A1;
* it records every exact returned class-constructor branch from a factory;
* a string/token operand is retained as an operand and never interpreted as a
  mechanism;
* a comprehension is one symbolic construction template, never N fabricated
  runtime occurrences; and
* source order, construction, execution order and architectural roles remain
  distinct.

The result deliberately remains ``ReaderResult.incomplete``.  U3 exposes no
whole-callable CFG coverage certificate, and this unit does not read config
values to select a factory branch.  Positive candidates are nevertheless exact
and usable by later U11 readers; uncertainty is carried rather than defaulted.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .container_inventory import ContainerAddress, ContainerRival
from .diffusion_root import DiffusionRootTopology, RepeatedRootStage
from .import_source import (
    CalledImportSourceResolution,
    resolve_called_import_source,
)
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    CallObservation,
    ConfigPathObservation,
    ConstructionSite,
    ExprNode,
    FieldAssignRecord,
    GuardStep,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


ISSUE_KINDS = frozenset({
    "container_rival",
    "storage_route_absent",
    "storage_route_rival",
    "dynamic_constructor",
    "import_incomplete",
    "import_ambiguous",
    "parse_failure",
    "factory_without_exact_return",
    "factory_cycle",
})


def _same_span(left: SourceSpan | None, right: SourceSpan | None) -> bool:
    return left is not None and right is not None and left == right


def _within(inner: SourceSpan | None, outer: SourceSpan | None) -> bool:
    if inner is None or outer is None or inner.source != outer.source:
        return False
    inner_end = (inner.end_line or inner.line, inner.end_col or inner.col)
    outer_end = (outer.end_line or outer.line, outer.end_col or outer.col)
    return ((inner.line, inner.col) >= (outer.line, outer.col)
            and inner_end <= outer_end)


def _self_field(expr: ExprNode | None) -> str | None:
    if expr is None or expr.kind != "attribute" or not expr.children:
        return None
    root = expr.children[0]
    return (expr.name if root.kind == "name" and root.name == "self"
            else None)


def _target_names(expr: ExprNode) -> tuple[str, ...]:
    if expr.kind == "name" and expr.name:
        return (expr.name,)
    if expr.kind in {"tuple", "list"}:
        return tuple(name for child in expr.children
                     for name in _target_names(child))
    return ()


def _binding_names(binding: BindingObservation) -> tuple[str, ...]:
    return tuple(name for target in binding.targets
                 for name in _target_names(target))


def _call_for_expr(index: ProgramIndex, callable_symbol: SymbolId,
                   expression: ExprNode) -> CallObservation | None:
    if expression.kind != "call" or expression.span is None:
        return None
    matches = tuple(call for call in index.calls_in(callable_symbol)
                    if _same_span(call.span, expression.span))
    return matches[0] if len(matches) == 1 else None


def _config_paths_inside(index: ProgramIndex, call: CallObservation) \
        -> tuple[ConfigPathObservation, ...]:
    paths = tuple(path for path in index.config_paths_in(call.enclosing_callable)
                  if _within(path.span, call.span))
    return tuple(sorted(paths, key=lambda item: (
        item.span.line if item.span else -1,
        item.span.col if item.span else -1,
    )))


@dataclass(frozen=True)
class StageConstructionIssue:
    """One exact reason a construction template is not uniquely closed."""

    kind: str
    detail: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.kind not in ISSUE_KINDS:
            raise ValueError(f"unknown stage-construction issue {self.kind!r}")
        if not self.detail:
            raise ValueError("a stage-construction issue requires detail")
        if self.span is not None and not isinstance(self.span, SourceSpan):
            raise TypeError("an issue span is a SourceSpan")


@dataclass(frozen=True)
class StageClassCandidate:
    """One exact class constructor reachable from one construction template.

    ``factory_chain`` is an address chain only.  The symbol/class spelling and
    any string dispatch operand carry no architectural semantics.
    """

    symbol: SymbolId
    factory_chain: tuple[SymbolId, ...]
    import_chain: tuple[CalledImportSourceResolution, ...] = ()
    call: CallObservation | None = None
    site: ConstructionSite | None = None
    returned_by: ReturnObservation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, SymbolId) or not self.factory_chain:
            raise TypeError("a stage candidate carries a symbol and factory chain")
        if any(not isinstance(item, SymbolId) for item in self.factory_chain):
            raise TypeError("factory-chain entries are exact SymbolIds")
        if any(not isinstance(item, CalledImportSourceResolution)
               or item.status != "resolved" for item in self.import_chain):
            raise TypeError("candidate import-chain entries are resolved U11-A1 proofs")
        if (self.call is None) == (self.site is None):
            raise ValueError("a candidate carries exactly one call or construction site")
        if self.call is not None:
            if self.call.span is None:
                raise ValueError("a call candidate carries an exact span")
            if self.returned_by is not None:
                if self.call.enclosing_callable != self.factory_chain[-1] \
                        or self.returned_by.enclosing_callable != \
                        self.factory_chain[-1] \
                        or not _within(self.call.span, self.returned_by.span):
                    raise ValueError("the constructor is returned by the exact factory")
            if self.symbol.source == self.call.enclosing_callable.source \
                    and (self.call.callee.kind != "name"
                         or self.call.callee.name != self.symbol.qualified_name):
                raise ValueError("a same-source candidate closes the exact called class")
            if self.symbol.source != self.call.enclosing_callable.source:
                if not self.import_chain \
                        or self.import_chain[-1].imported_symbol != self.symbol:
                    raise ValueError("a cross-source candidate carries its exact import proof")
        elif self.returned_by is not None:
            raise ValueError("a direct construction site has no factory return")
        if self.site is not None and self.site.span is None:
            raise ValueError("a direct construction site carries an exact span")

    @property
    def span(self) -> SourceSpan:
        return self.call.span if self.call is not None else self.site.span

    @property
    def guard(self) -> tuple[GuardStep, ...]:
        return (self.returned_by.guard if self.returned_by is not None
                else self.call.guard if self.call is not None
                else self.site.guard)


@dataclass(frozen=True)
class RepeatedStageConstruction:
    """One symbolic construction template feeding one exact U10 stage."""

    owner: OwnerOccurrenceId
    topology_stage: RepeatedRootStage
    topology_order: int
    producer_call: CallObservation | None
    producer_binding: BindingObservation | None
    storage_call: CallObservation | None
    direct_site: ConstructionSite | None
    config_paths: tuple[ConfigPathObservation, ...]
    candidates: tuple[StageClassCandidate, ...]
    issues: tuple[StageConstructionIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner, OwnerOccurrenceId) \
                or not isinstance(self.topology_stage, RepeatedRootStage):
            raise TypeError("a repeated construction is anchored to exact U10 evidence")
        if self.topology_stage.owner != self.owner or self.topology_order < 0:
            raise ValueError("the template belongs to the requested ordered topology")
        if (self.direct_site is None) == (self.producer_call is None):
            raise ValueError("a template is direct-site xor producer-call based")
        if self.direct_site is not None:
            if self.producer_binding is not None or self.storage_call is not None:
                raise ValueError("a direct container element needs no append route")
        else:
            if self.producer_binding is None or self.storage_call is None:
                raise ValueError("an appended template carries producer/binding/storage")
            if self.producer_binding.enclosing_callable != \
                    self.producer_call.enclosing_callable \
                    or self.storage_call.enclosing_callable != \
                    self.producer_call.enclosing_callable:
                raise ValueError("the append route remains in one exact callable")
            if not _same_span(self.producer_binding.value.span,
                              self.producer_call.span):
                raise ValueError("the producer binding cites the exact call expression")
            receiver = (self.storage_call.receiver
                        if self.storage_call.receiver is not None
                        else (self.storage_call.callee.children[0]
                              if self.storage_call.callee.children else None))
            if _self_field(receiver) != self.topology_stage.field:
                raise ValueError("the storage call targets the exact U10 container")
        invalid_paths = bool(self.config_paths) if self.producer_call is None else any(
            path.enclosing_callable != self.producer_call.enclosing_callable
            or not _within(path.span, self.producer_call.span)
            for path in self.config_paths)
        if invalid_paths:
            raise ValueError("config paths occur exactly inside the producer call")
        if not self.candidates and not self.issues:
            raise ValueError("a template carries positive candidates or typed uncertainty")
        if any(not isinstance(item, StageClassCandidate)
               for item in self.candidates):
            raise TypeError("template candidates are typed")
        if any(not isinstance(item, StageConstructionIssue) for item in self.issues):
            raise TypeError("template issues are typed")


@dataclass(frozen=True)
class UnresolvedStageConstruction:
    """One U10 stage whose construction route could not be uniquely proven."""

    owner: OwnerOccurrenceId
    topology_stage: RepeatedRootStage
    topology_order: int
    issues: tuple[StageConstructionIssue, ...]

    def __post_init__(self) -> None:
        if self.topology_stage.owner != self.owner or self.topology_order < 0:
            raise ValueError("unresolved construction belongs to the exact U10 stage")
        if not self.issues or any(not isinstance(item, StageConstructionIssue)
                                  for item in self.issues):
            raise ValueError("unresolved construction carries typed issues")


@dataclass(frozen=True)
class UNetStageConstructionInventory:
    """All exact repeated-stage construction templates for one U-shaped root.

    The carried ProgramIndex is the single immutable evidence universe after
    exact demand expansion.  Later U11 readers must consume it rather than
    reopening source files.
    """

    owner: OwnerOccurrenceId
    topology: DiffusionRootTopology
    stages: tuple[RepeatedStageConstruction, ...]
    unresolved_stages: tuple[UnresolvedStageConstruction, ...]
    index: ProgramIndex

    def __post_init__(self) -> None:
        if self.topology.kind != "u_shaped" or self.topology.owner != self.owner:
            raise ValueError("the construction inventory consumes one exact U-shape")
        if self.index.class_by_symbol(self.owner.root) is None:
            raise ValueError("the carried index contains the exact U10 root class")
        if any(not isinstance(item, UnresolvedStageConstruction)
               for item in self.unresolved_stages):
            raise TypeError("unresolved stages are typed")
        represented = {stage.topology_order for stage in self.stages}
        unresolved = {stage.topology_order for stage in self.unresolved_stages}
        expected = set(range(len(self.topology.stages)))
        if represented & unresolved or represented | unresolved != expected:
            raise ValueError("resolved templates and unresolved rows partition U10 stages")
        for stage in (*self.stages, *self.unresolved_stages):
            if self.topology.stages[stage.topology_order] != stage.topology_stage:
                raise ValueError("every row cites the exact U10 stage at its order")
        orders = tuple(stage.topology_order for stage in self.stages)
        if orders != tuple(sorted(orders)):
            raise ValueError("construction templates preserve U10 topology order")
        for stage in self.stages:
            if stage.producer_call is not None and stage.producer_call not in \
                    self.index.calls_in(stage.producer_call.enclosing_callable):
                raise ValueError("a producer call belongs to the carried index")
            if stage.producer_binding is not None and stage.producer_binding not in \
                    self.index.bindings_in(stage.producer_binding.enclosing_callable):
                raise ValueError("a producer binding belongs to the carried index")
            if stage.storage_call is not None and stage.storage_call not in \
                    self.index.calls_in(stage.storage_call.enclosing_callable):
                raise ValueError("a storage call belongs to the carried index")
            authoritative_sites = tuple(
                site for record in stage.topology_stage.container_records
                for raw in ((record.record,) if isinstance(record, ContainerAddress)
                            else record.records)
                for site in raw.elements)
            if stage.direct_site is not None \
                    and stage.direct_site not in authoritative_sites:
                raise ValueError("a direct site belongs to the authoritative container")
            for candidate in stage.candidates:
                if self.index.class_by_symbol(candidate.symbol) is None:
                    raise ValueError("every candidate class belongs to the carried index")
                if candidate.call is not None and candidate.call not in \
                        self.index.calls_in(candidate.call.enclosing_callable):
                    raise ValueError("every candidate call belongs to the carried index")
                if candidate.site is not None \
                        and candidate.site not in authoritative_sites \
                        and candidate.site not in self.index.construction_sites_in(
                            candidate.site.enclosing_callable):
                    raise ValueError("every candidate site belongs to indexed construction evidence")


@dataclass(frozen=True)
class DirectFieldConstruction:
    """One exact ``self.<field> = constructor(...)`` address.

    The field is supplied by a later execution reader from a positively observed
    invocation.  This DTO never decides that a field is a mid/bookend/conditioner
    from its spelling.
    """

    owner: OwnerOccurrenceId
    field_assign: FieldAssignRecord
    producer_call: CallObservation
    config_paths: tuple[ConfigPathObservation, ...]
    candidates: tuple[StageClassCandidate, ...]
    issues: tuple[StageConstructionIssue, ...]

    def __post_init__(self) -> None:
        if self.field_assign.owner != self.owner.root:
            raise ValueError("a direct-field construction belongs to the exact root")
        if self.field_assign.value.kind != "call" \
                or not _same_span(self.field_assign.value.span,
                                  self.producer_call.span):
            raise ValueError("the exact field assignment contains the producer call")
        if self.producer_call.enclosing_callable != \
                self.field_assign.enclosing_callable:
            raise ValueError("field assignment and call share one callable")
        if any(path.enclosing_callable != self.producer_call.enclosing_callable
               or not _within(path.span, self.producer_call.span)
               for path in self.config_paths):
            raise ValueError("direct-field config paths occur inside the call")
        if not self.candidates and not self.issues:
            raise ValueError("a direct-field construction carries evidence or uncertainty")

    @property
    def field(self) -> str:
        return self.field_assign.field


@dataclass(frozen=True)
class DirectFieldInvocationAddress:
    """One field address derived from exact calls inside a U10 inter-loop span.

    Requiring this proof object prevents callers from supplying a familiar field
    spelling and laundering it into construction evidence.
    """

    owner: OwnerOccurrenceId
    field: str
    calls: tuple[CallObservation, ...]
    earlier_stage: RepeatedRootStage
    later_stage: RepeatedRootStage

    def __post_init__(self) -> None:
        if not self.field or not self.calls:
            raise ValueError("an invocation address carries field + exact calls")
        if self.earlier_stage.owner != self.owner \
                or self.later_stage.owner != self.owner \
                or self.earlier_stage.loop.span is None \
                or self.later_stage.loop.span is None:
            raise ValueError("the invocation interval belongs to exact U10 stages")
        callable_symbol = self.earlier_stage.loop.enclosing_callable
        if self.later_stage.loop.enclosing_callable != callable_symbol \
                or self.earlier_stage == self.later_stage:
            raise ValueError("the invocation interval has two stages in one callable")
        lower = (self.earlier_stage.loop.span.end_line
                 or self.earlier_stage.loop.span.line,
                 self.earlier_stage.loop.span.end_col
                 or self.earlier_stage.loop.span.col)
        upper = (self.later_stage.loop.span.line,
                 self.later_stage.loop.span.col)
        for call in self.calls:
            if not isinstance(call, CallObservation) or call.span is None \
                    or _self_field(call.callee) != self.field \
                    or call.owner != self.owner.root \
                    or call.enclosing_callable != callable_symbol \
                    or call.span.source != callable_symbol.source \
                    or not lower < (call.span.line, call.span.col) < upper:
                raise ValueError("every call is the exact field inside the U10 interval")
        points = tuple((call.span.line, call.span.col) for call in self.calls)
        if points != tuple(sorted(points)) or len(set(points)) != len(points):
            raise ValueError("invocation addresses retain strict source order")


@dataclass(frozen=True)
class UnresolvedDirectFieldConstruction:
    """A matching field assignment whose constructor is not exactly callable."""

    owner: OwnerOccurrenceId
    field_assign: FieldAssignRecord
    issue: StageConstructionIssue

    def __post_init__(self) -> None:
        if self.field_assign.owner != self.owner.root:
            raise ValueError("unresolved direct field belongs to the exact root")
        if not isinstance(self.issue, StageConstructionIssue):
            raise TypeError("unresolved direct field carries a typed issue")


@dataclass(frozen=True)
class DirectFieldConstructionInventory:
    """All assignments to one exact field selected by positive execution evidence."""

    owner: OwnerOccurrenceId
    address: DirectFieldInvocationAddress
    constructions: tuple[DirectFieldConstruction, ...]
    unresolved: tuple[UnresolvedDirectFieldConstruction, ...]
    index: ProgramIndex

    def __post_init__(self) -> None:
        if self.address.owner != self.owner \
                or self.index.class_by_symbol(self.owner.root) is None:
            raise ValueError("a direct-field inventory has an indexed owner + field")
        if not (self.constructions or self.unresolved):
            raise ValueError("a direct-field inventory carries at least one assignment")
        if any(call not in self.index.calls_in(call.enclosing_callable)
               for call in self.address.calls):
            raise ValueError("the invocation address belongs to the carried index")
        if any(item.owner != self.owner or item.field != self.field
               for item in self.constructions):
            raise ValueError("every construction targets the requested owner + field")
        if any(item.owner != self.owner
               or item.field_assign.field != self.field
               for item in self.unresolved):
            raise ValueError("every unresolved row targets the requested owner + field")
        for item in self.constructions:
            if item.field_assign not in self.index.field_assigns_of(self.owner.root) \
                    or item.producer_call not in self.index.calls_in(
                        item.producer_call.enclosing_callable):
                raise ValueError("direct-field evidence belongs to the carried index")
            for candidate in item.candidates:
                if self.index.class_by_symbol(candidate.symbol) is None:
                    raise ValueError("direct-field candidates belong to the carried index")

    @property
    def field(self) -> str:
        return self.address.field


def _factory_candidates(
        index: ProgramIndex,
        bundle: SourceBundle,
        component: str,
        call: CallObservation,
        chain: tuple[SymbolId, ...] = (),
        imports: tuple[CalledImportSourceResolution, ...] = (),
        ) -> tuple[tuple[StageClassCandidate, ...],
                   tuple[StageConstructionIssue, ...], ProgramIndex]:
    """Resolve one exact constructor/factory call without interpreting names."""
    if call.callee.kind != "name" or not call.callee.name:
        return (), (StageConstructionIssue(
            "dynamic_constructor", "the producer call has no exact name binding",
            call.span),), index

    local = SymbolId(call.enclosing_callable.source, call.callee.name)
    if index.class_by_symbol(local) is not None:
        sites = tuple(site for site in index.construction_sites_in(
            call.enclosing_callable) if _same_span(site.span, call.span))
        if len(sites) == 1:
            return (StageClassCandidate(
                local, (call.enclosing_callable,), imports,
                site=sites[0]),), (), index
        return (StageClassCandidate(
            local, (local,), imports, call=call),), (), index

    callable_record = index.callable_by_symbol(local)
    expanded = index
    factory_symbol = local if callable_record is not None else None
    if factory_symbol is None:
        imported = resolve_called_import_source(index, bundle, component, call)
        if imported.status == "ambiguous":
            return (), (StageConstructionIssue(
                "import_ambiguous", "the called import has rival bindings",
                call.span),), index
        if imported.status != "resolved":
            kind = ("parse_failure" if imported.status == "failed"
                    else "import_incomplete")
            return (), (StageConstructionIssue(
                kind, imported.failure_detail or imported.failure_kind,
                call.span),), index
        expanded = imported.index
        factory_symbol = imported.imported_symbol
        imports = (*imports, imported)
        if expanded.class_by_symbol(factory_symbol) is not None:
            return (StageClassCandidate(
                factory_symbol, (factory_symbol,), imports,
                call=call),), (), expanded
        callable_record = expanded.callable_by_symbol(factory_symbol)

    if callable_record is None:
        return (), (StageConstructionIssue(
            "dynamic_constructor", "the exact called symbol is not a class or callable",
            call.span),), expanded
    if factory_symbol in chain:
        return (), (StageConstructionIssue(
            "factory_cycle", "the exact factory-return chain is cyclic",
            call.span),), expanded

    next_chain = (*chain, factory_symbol)
    candidates: list[StageClassCandidate] = []
    issues: list[StageConstructionIssue] = []
    returns = expanded.return_observations_in(factory_symbol)
    for returned in returns:
        if returned.value is None or returned.value.kind != "call":
            issues.append(StageConstructionIssue(
                "factory_without_exact_return",
                "a factory return is not an exact constructor call",
                returned.span))
            continue
        returned_call = _call_for_expr(expanded, factory_symbol, returned.value)
        if returned_call is None or returned_call.callee.kind != "name" \
                or not returned_call.callee.name:
            issues.append(StageConstructionIssue(
                "dynamic_constructor",
                "a returned constructor has no unique exact call binding",
                returned.span))
            continue
        symbol = SymbolId(factory_symbol.source, returned_call.callee.name)
        if expanded.class_by_symbol(symbol) is not None:
            candidates.append(StageClassCandidate(
                symbol, next_chain, imports,
                call=returned_call, returned_by=returned))
            continue
        child_candidates, child_issues, expanded = _factory_candidates(
            expanded, bundle, component, returned_call, next_chain, imports)
        candidates.extend(child_candidates)
        issues.extend(child_issues)
    if not returns:
        issues.append(StageConstructionIssue(
            "factory_without_exact_return",
            "the exact factory has no indexed return observations",
            callable_record.span))
    return tuple(candidates), tuple(issues), expanded


def resolve_stage_constructor_candidates(
        index: ProgramIndex,
        bundle: SourceBundle,
        component: str,
        call: CallObservation,
        ) -> tuple[tuple[StageClassCandidate, ...],
                   tuple[StageConstructionIssue, ...], ProgramIndex]:
    """Expand one exact stage/cell constructor call without role semantics.

    U11-B and later U11 child readers share this ONE demand-driven factory/import
    boundary.  The call must belong to the supplied index and component; callers
    cannot pass a class spelling or reopen source through another parser.
    """
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("stage-constructor expansion requires ProgramIndex + bundle")
    if not isinstance(call, CallObservation) \
            or call not in index.calls_in(call.enclosing_callable):
        raise ValueError("stage-constructor expansion requires one exact indexed call")
    if not component or call.enclosing_callable.source.component_key != component:
        raise ValueError("the constructor call belongs to the requested component")
    return _factory_candidates(index, bundle, component, call)


def _direct_templates(stage: RepeatedRootStage, order: int, index: ProgramIndex):
    templates = []
    for record in stage.container_records:
        if isinstance(record, ContainerRival):
            return (), StageConstructionIssue(
                "container_rival",
                "multiple container constructions target the exact stage field",
                next(iter(record.records)).span)
        if not isinstance(record, ContainerAddress):
            continue
        for site in record.element_sites:
            candidates = tuple(StageClassCandidate(
                child.symbol, (site.owner,), (), site=site)
                for child in site.candidates if child.symbol is not None
                and index.class_by_symbol(child.symbol) is not None)
            issues = (() if candidates and len(candidates) == len(site.candidates)
                      else (StageConstructionIssue(
                          "dynamic_constructor",
                          "the symbolic element has unresolved/rival class candidates",
                          site.span),))
            templates.append(RepeatedStageConstruction(
                stage.owner, stage, order, None, None, None, site,
                (), candidates, issues))
    return tuple(templates), None


def _append_routes(index: ProgramIndex, stage: RepeatedRootStage,
                   init_symbol: SymbolId):
    calls = index.calls_in(init_symbol)
    bindings = index.bindings_in(init_symbol)
    routes = []
    issues = []
    for storage in calls:
        receiver = (storage.receiver if storage.receiver is not None
                    else (storage.callee.children[0]
                          if storage.callee.children else None))
        if storage.callee.kind != "attribute" or storage.callee.name != "append" \
                or _self_field(receiver) != stage.field \
                or len(storage.args) != 1:
            continue
        stored = storage.args[0]
        if stored.kind != "name" or not stored.name:
            continue
        producers = tuple(binding for binding in bindings
                          if stored.name in _binding_names(binding)
                          and binding.value.kind == "call"
                          and binding.span is not None and storage.span is not None
                          and (binding.span.line, binding.span.col)
                          < (storage.span.line, storage.span.col)
                          and binding.guard[:len(storage.guard)] == storage.guard)
        exact = tuple((producer, binding, storage)
                      for binding in producers
                      for producer in (_call_for_expr(
                          index, init_symbol, binding.value),)
                      if producer is not None)
        if len(exact) == 1:
            routes.extend(exact)
        elif len(exact) > 1:
            issues.append(StageConstructionIssue(
                "storage_route_rival",
                "rival guarded producers can reach one exact append",
                storage.span))
    return tuple(routes), tuple(issues)


def read_unet_stage_construction(
        index: ProgramIndex,
        bundle: SourceBundle,
        root_resolution: ComponentRootResolution,
        topology: DiffusionRootTopology,
        ) -> ReaderResult[UNetStageConstructionInventory]:
    """Inventory exact construction templates for every U10 repeated stage."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("U11-B requires the one ProgramIndex and SourceBundle")
    root_resolution = require_resolved_component_root(
        root_resolution, caller="read_unet_stage_construction")
    if not isinstance(root_resolution, ComponentRootResolution):
        raise TypeError("U11-B currently consumes a D0 component root")
    owner = root_resolution.occurrence
    if not isinstance(topology, DiffusionRootTopology) \
            or topology.owner != owner or topology.kind != "u_shaped":
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner",
            "U11-B requires the exact U10 u_shaped topology for this root"),))
    node = root_resolution.graph.node_for(owner)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the exact U10 root is not in this ProgramIndex"),))
    init_symbol = SymbolId(node.symbol.source, f"{node.symbol.qualified_name}.__init__")
    if index.callable_by_symbol(init_symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "missing_source", "the exact U-shaped root has no indexed constructor"),))

    expanded = index
    templates: list[RepeatedStageConstruction] = []
    unresolved: list[UnresolvedStageConstruction] = []
    for order, stage in enumerate(topology.stages):
        direct, direct_issue = _direct_templates(stage, order, expanded)
        if direct_issue is not None:
            unresolved.append(UnresolvedStageConstruction(
                owner, stage, order, (direct_issue,)))
            continue
        if direct:
            templates.extend(direct)
            continue
        routes, route_issues = _append_routes(expanded, stage, init_symbol)
        if route_issues or not routes:
            issues = route_issues or (StageConstructionIssue(
                "storage_route_absent",
                "no exact producer→append route feeds the U10 container",
                stage.loop.span),)
            unresolved.append(UnresolvedStageConstruction(
                owner, stage, order, issues))
            continue
        for producer, binding, storage in routes:
            candidates, issues, expanded = resolve_stage_constructor_candidates(
                expanded, bundle, root_resolution.component_key, producer)
            templates.append(RepeatedStageConstruction(
                owner, stage, order, producer, binding, storage, None,
                _config_paths_inside(expanded, producer), candidates, issues))

    inventory = UNetStageConstructionInventory(
        owner, topology, tuple(templates), tuple(unresolved), expanded)
    spans = tuple(dict.fromkeys(
        (
            *(span for stage in inventory.stages
              for span in (
                  stage.producer_call.span
                  if stage.producer_call else stage.direct_site.span,
                  stage.storage_call.span if stage.storage_call else None,
                  *(candidate.span for candidate in stage.candidates),
              ) if span is not None),
            *(issue.span for stage in inventory.unresolved_stages
              for issue in stage.issues if issue.span is not None),
        )))
    return ReaderResult.incomplete(
        owner, inventory,
        failures=(ReaderFailure(
            "incomplete_graph",
            "positive exact construction inventory; config branch selection and "
            "whole-callable coverage remain open"),),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact U10 container→producer→factory-return construction evidence"),))


def read_direct_field_construction(
        index: ProgramIndex,
        bundle: SourceBundle,
        root_resolution: ComponentRootResolution,
        address: DirectFieldInvocationAddress,
        ) -> ReaderResult[DirectFieldConstructionInventory]:
    """Expand one field already selected by positive execution evidence.

    A spelling is not accepted. U11-C supplies a closed invocation-address
    object derived from exact direct ``self.<field>(...)`` calls between the U10
    repeated sides.
    """
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("direct-field construction requires ProgramIndex + bundle")
    root_resolution = require_resolved_component_root(
        root_resolution, caller="read_direct_field_construction")
    if not isinstance(root_resolution, ComponentRootResolution) \
            or not isinstance(address, DirectFieldInvocationAddress):
        raise TypeError("direct-field construction requires D0 + invocation address")
    owner = root_resolution.occurrence
    if address.owner != owner:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the invocation address belongs to another root"),))
    node = root_resolution.graph.node_for(owner)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the exact root is not in this ProgramIndex"),))
    all_assignments = tuple(item for item in index.field_assigns_of(node.symbol)
                            if item.field == address.field)
    if not all_assignments:
        return ReaderResult.absent(owner)
    init_symbol = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.__init__")
    assignments = tuple(item for item in all_assignments
                        if item.enclosing_callable == init_symbol)
    if not assignments:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the invoked field is assigned only outside the exact root constructor"),))

    expanded = index
    constructions = []
    unresolved = []
    for assignment in assignments:
        producer = _call_for_expr(
            expanded, assignment.enclosing_callable, assignment.value)
        if producer is None:
            unresolved.append(UnresolvedDirectFieldConstruction(
                owner, assignment, StageConstructionIssue(
                    "dynamic_constructor",
                    "the selected field assignment is not one exact constructor call",
                    assignment.span)))
            continue
        candidates, issues, expanded = _factory_candidates(
            expanded, bundle, root_resolution.component_key, producer)
        constructions.append(DirectFieldConstruction(
            owner, assignment, producer,
            _config_paths_inside(expanded, producer), candidates, issues))
    inventory = DirectFieldConstructionInventory(
        owner, address, tuple(constructions), tuple(unresolved), expanded)
    spans = tuple(dict.fromkeys((
        *(item.producer_call.span for item in constructions
          if item.producer_call.span is not None),
        *(item.field_assign.span for item in unresolved
          if item.field_assign.span is not None),
        *(candidate.span for item in constructions
          for candidate in item.candidates),
    )))
    return ReaderResult.incomplete(
        owner, inventory,
        failures=(ReaderFailure(
            "incomplete_graph",
            "positive field-construction evidence; whole-callable coverage is open"),),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact execution-selected field construction evidence"),))


__all__ = [
    "ISSUE_KINDS",
    "DirectFieldConstruction",
    "DirectFieldConstructionInventory",
    "DirectFieldInvocationAddress",
    "RepeatedStageConstruction",
    "StageClassCandidate",
    "StageConstructionIssue",
    "UNetStageConstructionInventory",
    "UnresolvedDirectFieldConstruction",
    "UnresolvedStageConstruction",
    "read_unet_stage_construction",
    "read_direct_field_construction",
    "resolve_stage_constructor_candidates",
]
