"""U11-D1 — neutral child-invocation inventory for exact U-Net stages.

This boundary starts from the U11-C graph and discovers the constructed child
objects that its exact stage-class candidates invoke.  It supports direct
container iteration, sliced/reversed/enumerated wrappers, zip/enumerate(zip)
siblings, and direct ``self.<field>(...)`` calls.  Every result retains the
construction/call/loop evidence and every factory rival.

Nothing here classifies a child as residual, attention, sampler, temporal, or
any other architectural role.  Field/class/factory spellings are addresses
only.  Source order is not execution order; unsupported/unreachable calls stay
typed unresolved and whole-callable coverage remains open.
"""
from __future__ import annotations

from dataclasses import dataclass

from .execution_flow import execution_taint_reason, unshadowed_builtin
from .component_owner import OwnerOccurrenceId
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    CallObservation,
    ConstructionSite,
    ContainerElementsRecord,
    ExprNode,
    FieldAssignRecord,
    LoopObservation,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_stage_construction import (
    DirectFieldConstruction,
    RepeatedStageConstruction,
    StageClassCandidate,
    StageConstructionIssue,
    resolve_stage_constructor_candidates,
)
from .unet_stage_execution import (
    UNetStageExecutionGraph,
    UNetStageNodeId,
)


UNRESOLVED_CHILD_KINDS = frozenset({
    "missing_forward",
    "unsupported_iteration",
    "container_absent",
    "container_rival",
    "construction_incomplete",
    "call_tainted",
    "whole_callable_open",
})


def _within(inner: SourceSpan | None, outer: SourceSpan | None) -> bool:
    if inner is None or outer is None or inner.source != outer.source:
        return False
    return ((inner.line, inner.col) >= (outer.line, outer.col)
            and (inner.end_line or inner.line, inner.end_col or inner.col)
            <= (outer.end_line or outer.line, outer.end_col or outer.col))


def _self_field(expr: ExprNode | None) -> str | None:
    if expr is None or expr.kind != "attribute" or not expr.children:
        return None
    root = expr.children[0]
    return expr.name if root.kind == "name" and root.name == "self" else None


def _target_names(expr: ExprNode | None) -> tuple[str, ...]:
    if expr is None:
        return ()
    if expr.kind == "name" and expr.name:
        return (expr.name,)
    if expr.kind in {"tuple", "list"}:
        return tuple(name for child in expr.children
                     for name in _target_names(child))
    return ()


def _callee_name(expr: ExprNode | None) -> str | None:
    return expr.name if expr is not None and expr.kind == "name" else None


def _before(left: SourceSpan | None, right: SourceSpan | None) -> bool:
    return bool(left is not None and right is not None
                and left.source == right.source
                and (left.end_line or left.line, left.end_col or left.col)
                <= (right.line, right.col))


def _binding_target(binding: BindingObservation) -> str | None:
    names = tuple(name for target in binding.targets
                  for name in _target_names(target))
    return names[0] if len(names) == 1 else None


def _iteration_bindings(iterable: ExprNode | None, target: ExprNode | None,
                        aliases: dict[str, BindingObservation] | None = None,
                        seen=frozenset()) \
        -> tuple[tuple[str, str], ...] | None:
    """Exact ``container field -> loop target`` bindings, or unsupported.

    Wrapper spellings below describe Python iteration syntax only.  They never
    carry a module role or execution claim.
    """
    if iterable is None or target is None:
        return None
    aliases = aliases or {}
    if iterable.kind == "name" and iterable.name in aliases:
        if iterable.name in seen:
            return None
        value = aliases[iterable.name].value
        return _iteration_bindings(
            value, target, aliases, seen | {iterable.name})
    field = _self_field(iterable)
    targets = _target_names(target)
    if field is not None:
        return ((field, targets[-1]),) if len(targets) == 1 else None
    if iterable.kind == "subscript" and iterable.children:
        return _iteration_bindings(
            iterable.children[0], target, aliases, seen)
    if iterable.kind != "call" or not iterable.children:
        return None
    callee = _callee_name(iterable.children[0])
    args = iterable.children[1:]
    if callee in {"enumerate", "reversed", "list", "tuple"} and args:
        inner_target = target
        if callee == "enumerate" and target.kind in {"tuple", "list"} \
                and len(target.children) >= 2:
            inner_target = target.children[-1]
        return _iteration_bindings(args[0], inner_target, aliases, seen)
    if callee != "zip" or target.kind not in {"tuple", "list"} \
            or len(args) != len(target.children):
        return None
    rows: list[tuple[str, str]] = []
    for arg, child_target in zip(args, target.children):
        bound = _iteration_bindings(arg, child_target, aliases, seen)
        if bound is None:
            return None
        rows.extend(bound)
    return tuple(rows)


def _referenced_aliases(expression: ExprNode | None,
                        aliases: dict[str, BindingObservation],
                        seen=frozenset()) -> frozenset[str]:
    if expression is None:
        return frozenset()
    if expression.kind == "name" and expression.name in aliases:
        name = expression.name
        if name in seen:
            return frozenset((name,))
        return frozenset((name,)) | _referenced_aliases(
            aliases[name].value, aliases, seen | {name})
    return frozenset().union(*(
        [_referenced_aliases(child, aliases, seen)
         for child in expression.children]
        + [_referenced_aliases(child, aliases, seen)
           for _name, child in expression.keyword_children]))


def _iteration_builtin_names(
        expression: ExprNode | None,
        aliases: dict[str, BindingObservation], seen=frozenset()) \
        -> frozenset[str]:
    if expression is None:
        return frozenset()
    if expression.kind == "name" and expression.name in aliases:
        if expression.name in seen:
            return frozenset()
        return _iteration_builtin_names(
            aliases[expression.name].value, aliases,
            seen | {expression.name})
    own = frozenset()
    if expression.kind == "call" and expression.children:
        callee = _callee_name(expression.children[0])
        if callee in {"enumerate", "reversed", "list", "tuple", "zip"}:
            own = frozenset((callee,))
    return own | frozenset().union(*(
        [_iteration_builtin_names(child, aliases, seen)
         for child in expression.children]
        + [_iteration_builtin_names(child, aliases, seen)
           for _name, child in expression.keyword_children]))


def _iteration_aliases(index: ProgramIndex, loop: LoopObservation) \
        -> tuple[BindingObservation, ...] | None:
    """Resolve only exact callable-local iterable aliases reaching ``loop``."""
    selected = []
    visiting = set()

    def visit(expression):
        if expression is None:
            return True
        if expression.kind == "name" and expression.name:
            name = expression.name
            definitions = tuple(
                item for item in index.bindings_in(loop.enclosing_callable)
                if _binding_target(item) == name
                and _before(item.span, loop.span))
            if not definitions:
                return True
            unguarded = tuple(item for item in definitions if not item.guard)
            if not unguarded:
                return False
            winner = unguarded[-1]
            if any(_before(winner.span, item.span)
                   for item in definitions if item != winner):
                return False
            if name in visiting or winner.value is None:
                return False
            visiting.add(name)
            if not visit(winner.value):
                return False
            visiting.remove(name)
            selected.append(winner)
            return True
        return all(visit(child) for child in expression.children) \
            and all(visit(child) for _name, child in expression.keyword_children)

    if not visit(loop.iterable):
        return None
    return tuple(dict.fromkeys(selected))


def _call_for_span(index: ProgramIndex, callable_symbol: SymbolId,
                   span: SourceSpan | None) -> CallObservation | None:
    matches = tuple(call for call in index.calls_in(callable_symbol)
                    if span is not None and call.span == span)
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class StageClassOccurrenceId:
    """One construction occurrence of one stage candidate."""

    owner: OwnerOccurrenceId
    parent_field: str
    construction_span: SourceSpan
    candidate_span: SourceSpan
    symbol: SymbolId

    def __post_init__(self) -> None:
        if not isinstance(self.owner, OwnerOccurrenceId) \
                or not self.parent_field or not isinstance(self.symbol, SymbolId):
            raise ValueError("a stage-class occurrence retains field + symbol")
        if not isinstance(self.construction_span, SourceSpan) \
                or not isinstance(self.candidate_span, SourceSpan):
            raise TypeError("a stage-class occurrence retains exact spans")


@dataclass(frozen=True)
class StageClassOccurrence:
    occurrence_id: StageClassOccurrenceId
    parent_nodes: tuple[UNetStageNodeId, ...]
    candidate: StageClassCandidate
    construction: RepeatedStageConstruction | DirectFieldConstruction

    def __post_init__(self) -> None:
        if not self.parent_nodes or len(set(self.parent_nodes)) != len(self.parent_nodes):
            raise ValueError("a stage occurrence retains unique parent invocations")
        if any(node.owner != self.occurrence_id.owner
               or node.field != self.occurrence_id.parent_field
               for node in self.parent_nodes):
            raise ValueError("stage occurrence parent nodes retain owner + field")
        if self.candidate.symbol != self.occurrence_id.symbol \
                or self.candidate.span != self.occurrence_id.candidate_span:
            raise ValueError("stage occurrence round-trips to the candidate proof")
        if self.candidate not in self.construction.candidates:
            raise ValueError("the candidate belongs to the exact construction")
        field = (self.construction.topology_stage.field
                 if isinstance(self.construction, RepeatedStageConstruction)
                 else self.construction.field)
        if field != self.occurrence_id.parent_field:
            raise ValueError("stage occurrence retains its exact parent field")


@dataclass(frozen=True)
class ChildConstructionEvidence:
    field: str
    site: ConstructionSite | None
    field_assign: FieldAssignRecord | None
    producer_call: CallObservation | None
    candidates: tuple[StageClassCandidate, ...]
    issues: tuple[StageConstructionIssue, ...]

    def __post_init__(self) -> None:
        if not self.field or (self.site is None) == (self.field_assign is None):
            raise ValueError("child construction is container-site xor direct-field")
        expected = (self.site.span if self.site is not None
                    else self.field_assign.value.span)
        if self.producer_call is not None \
                and self.producer_call.span != expected:
            raise ValueError("child construction retains the exact producer call")
        if self.producer_call is None and self.candidates:
            raise ValueError("positive candidates require an exact producer call")
        if not self.candidates and not self.issues:
            raise ValueError("child construction retains evidence or uncertainty")
        for candidate in self.candidates:
            if candidate.import_chain:
                bound = candidate.import_chain[0].call == self.producer_call
            elif candidate.site is not None:
                bound = (candidate.site == self.site
                         if self.site is not None else
                         self.field_assign is not None
                         and candidate.site.owner == self.field_assign.owner
                         and candidate.site.enclosing_callable
                         == self.field_assign.enclosing_callable
                         and candidate.site.target == self.field
                         and candidate.site.span == self.field_assign.value.span)
            elif candidate.returned_by is not None:
                local = SymbolId(
                    self.producer_call.enclosing_callable.source,
                    self.producer_call.callee.name or "")
                bound = bool(candidate.factory_chain) \
                    and candidate.factory_chain[0] == local
            else:
                bound = candidate.call == self.producer_call
            if not bound:
                raise ValueError(
                    "every child candidate closes this exact constructor route")


@dataclass(frozen=True)
class StageChildInvocation:
    parent: StageClassOccurrence
    kind: str                    # repeated | direct
    field: str
    call: CallObservation
    loop: LoopObservation | None
    target: str
    iteration_aliases: tuple[BindingObservation, ...]
    iteration_builtins: tuple[str, ...]
    constructions: tuple[ChildConstructionEvidence, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"repeated", "direct"} or not self.field:
            raise ValueError("child invocation kind is neutral and closed")
        if self.call.owner != self.parent.occurrence_id.symbol \
                or self.call.span is None:
            raise ValueError("child invocation belongs to the exact stage class")
        if self.kind == "repeated":
            if self.loop is None or self.loop.body_span is None \
                    or not _within(self.call.span, self.loop.body_span) \
                    or self.call.callee.kind != "name" \
                    or self.call.callee.name != self.target:
                raise ValueError("repeated child call is bound by the exact loop target")
            if any(item.enclosing_callable != self.loop.enclosing_callable
                   or item.guard or item.value is None
                   or not _before(item.span, self.loop.span)
                   for item in self.iteration_aliases):
                raise ValueError("iteration aliases are exact reaching definitions")
            alias_map = {_binding_target(item): item
                         for item in self.iteration_aliases}
            if None in alias_map or len(alias_map) != len(self.iteration_aliases):
                raise ValueError("iteration aliases have unique simple targets")
            if _referenced_aliases(
                    self.loop.iterable, alias_map) != frozenset(alias_map):
                raise ValueError("iteration aliases are the exact used route")
            expected_builtins = tuple(sorted(_iteration_builtin_names(
                self.loop.iterable, alias_map)))
            if self.iteration_builtins != expected_builtins:
                raise ValueError("iteration builtin protocols are exact syntax")
            bindings = _iteration_bindings(
                self.loop.iterable, self.loop.target, alias_map)
            if bindings is None or (self.field, self.target) not in bindings:
                raise ValueError("the loop structurally binds target to this container")
        elif self.loop is not None or self.iteration_aliases \
                or self.iteration_builtins \
                or _self_field(self.call.callee) != self.field:
            raise ValueError("direct child call targets exact self field")
        if not self.constructions or any(item.field != self.field
                                         for item in self.constructions):
            raise ValueError("child invocation retains all exact constructions")
        for item in self.constructions:
            if item.site is not None and item.site.owner != \
                    self.parent.occurrence_id.symbol:
                raise ValueError("container construction belongs to the stage class")
            if item.field_assign is not None and item.field_assign.owner != \
                    self.parent.occurrence_id.symbol:
                raise ValueError("direct construction belongs to the stage class")


@dataclass(frozen=True)
class UnresolvedStageChild:
    parent: StageClassOccurrence
    kind: str
    detail: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.kind not in UNRESOLVED_CHILD_KINDS or not self.detail:
            raise ValueError("unresolved child evidence has closed kind + detail")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("unresolved child evidence retains typed spans")


@dataclass(frozen=True)
class UNetStageCellInventory:
    graph: UNetStageExecutionGraph
    stages: tuple[StageClassOccurrence, ...]
    invocations: tuple[StageChildInvocation, ...]
    unresolved: tuple[UnresolvedStageChild, ...]
    index: ProgramIndex
    bundle: SourceBundle

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a cell inventory retains exact stage candidates")
        if not isinstance(self.index, ProgramIndex) \
                or not isinstance(self.bundle, SourceBundle):
            raise TypeError("a cell inventory retains its exact index + bundle")
        ids = tuple(item.occurrence_id for item in self.stages)
        if len(ids) != len(set(ids)):
            raise ValueError("stage construction occurrence identities are unique")
        if any(item.parent not in self.stages for item in self.invocations) \
                or any(item.parent not in self.stages for item in self.unresolved):
            raise ValueError("all child evidence belongs to an authoritative stage")
        graph_nodes = {
            item.node_id for item in (*self.graph.repeated, *self.graph.direct)
        }
        if any(not set(item.parent_nodes) <= graph_nodes for item in self.stages):
            raise ValueError("stage occurrences belong to exact U11-C nodes")
        call_ids = tuple((item.parent.occurrence_id, item.call)
                         for item in self.invocations)
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("each stage-class occurrence carries a call once")
        if any(item.call not in self.index.calls_in(item.call.enclosing_callable)
               for item in self.invocations):
            raise ValueError("all child calls belong to the final index")
        if any(alias not in self.index.bindings_in(alias.enclosing_callable)
               for item in self.invocations for alias in item.iteration_aliases):
            raise ValueError("all iteration aliases belong to the final index")
        if any(item.kind == "repeated"
               and _iteration_aliases(self.index, item.loop)
               != item.iteration_aliases
               for item in self.invocations):
            raise ValueError(
                "iteration aliases are the exact reaching route for each loop")
        if any(not unshadowed_builtin(
                self.index, item.call.enclosing_callable, name)
               for item in self.invocations for name in item.iteration_builtins):
            raise ValueError("iteration builtin protocols are lexically proven")
        if any(construction.producer_call is not None
               and construction.producer_call not in self.index.calls_in(
                   construction.producer_call.enclosing_callable)
               for item in self.invocations
               for construction in item.constructions):
            raise ValueError("all child constructors belong to the final index")
        if any(self.index.class_by_symbol(candidate.symbol) is None
               for item in self.invocations for construction in item.constructions
               for candidate in construction.candidates):
            raise ValueError("all child candidates belong to the final index")
        for stage in self.stages:
            if sum(item.kind == "whole_callable_open" and item.parent == stage
                   for item in self.unresolved) != 1:
                raise ValueError(
                    "every stage candidate carries one open-callable disposition")


def _stage_occurrences(graph: UNetStageExecutionGraph) \
        -> tuple[StageClassOccurrence, ...]:
    rows: list[StageClassOccurrence] = []
    for parent in graph.repeated:
        for construction in parent.constructions:
            construction_span = (construction.producer_call.span
                                 if construction.producer_call is not None
                                 else construction.direct_site.span)
            for candidate in construction.candidates:
                ident = StageClassOccurrenceId(
                    graph.owner, parent.node_id.field, construction_span,
                    candidate.span, candidate.symbol)
                rows.append(StageClassOccurrence(
                    ident, (parent.node_id,), candidate, construction))
    direct_by_key: dict[tuple, tuple[DirectFieldConstruction, list]] = {}
    for parent in graph.direct:
        for construction in parent.construction.constructions:
            for candidate in construction.candidates:
                key = (construction.field_assign.span, candidate.span,
                       candidate.symbol, parent.node_id.field)
                if key not in direct_by_key:
                    direct_by_key[key] = (construction, [])
                direct_by_key[key][1].append(parent.node_id)
    for (_key, (construction, nodes)) in direct_by_key.items():
        for candidate in construction.candidates:
            if candidate.symbol != _key[2] or candidate.span != _key[1]:
                continue
            ident = StageClassOccurrenceId(
                graph.owner, _key[3], construction.field_assign.span,
                candidate.span, candidate.symbol)
            rows.append(StageClassOccurrence(
                ident, tuple(nodes), candidate, construction))
    return tuple(rows)


def _container_constructions(index, bundle, component, record):
    expanded = index
    rows = []
    for site in record.elements:
        call = _call_for_span(expanded, site.enclosing_callable, site.span)
        if call is None:
            issues = (StageConstructionIssue(
                "dynamic_constructor", "container element has no exact call", site.span),)
            rows.append(ChildConstructionEvidence(
                record.field, site, None, None, (), issues))
            continue
        candidates, issues, expanded = resolve_stage_constructor_candidates(
            expanded, bundle, component, call)
        rows.append(ChildConstructionEvidence(
            record.field, site, None, call, candidates, issues))
    return tuple(rows), expanded


def _direct_constructions(index, bundle, component, symbol, field):
    expanded = index
    init = SymbolId(symbol.source, f"{symbol.qualified_name}.__init__")
    rows = []
    for assignment in index.field_assigns_of(symbol):
        if assignment.field != field or assignment.enclosing_callable != init \
                or assignment.value.kind != "call":
            continue
        call = _call_for_span(expanded, init, assignment.value.span)
        if call is None:
            continue
        candidates, issues, expanded = resolve_stage_constructor_candidates(
            expanded, bundle, component, call)
        rows.append(ChildConstructionEvidence(
            field, None, assignment, call, candidates, issues))
    return tuple(rows), expanded


def read_unet_stage_cells(graph: UNetStageExecutionGraph,
                          bundle: SourceBundle) \
        -> ReaderResult[UNetStageCellInventory]:
    """Inventory every exact constructed child call under U11-C stages."""
    if not isinstance(graph, UNetStageExecutionGraph) \
            or not isinstance(bundle, SourceBundle):
        raise TypeError("U11-D1 requires U11-C graph + SourceBundle")
    stages = _stage_occurrences(graph)
    if not stages:
        return ReaderResult.failed(graph.owner, (ReaderFailure(
            "incomplete_graph", "U11-C exposes no exact stage-class candidate"),))
    expanded = graph.index
    invocations = []
    unresolved = []
    component = graph.owner.root.source.component_key
    for stage in stages:
        symbol = stage.occurrence_id.symbol
        forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
        forward_record = expanded.callable_by_symbol(forward)
        if forward_record is None:
            unresolved.append(UnresolvedStageChild(
                stage, "missing_forward", "stage candidate has no indexed forward",
                (stage.occurrence_id.candidate_span,)))
            continue
        matched_calls = set()
        records_by_field: dict[str, list[ContainerElementsRecord]] = {}
        for record in expanded.containers:
            if record.owner == symbol:
                records_by_field.setdefault(record.field, []).append(record)
        for loop in expanded.loops_in(forward):
            aliases = _iteration_aliases(expanded, loop)
            alias_map = ({_binding_target(item): item for item in aliases}
                         if aliases is not None else {})
            builtin_names = tuple(sorted(_iteration_builtin_names(
                loop.iterable, alias_map)))
            bindings = (_iteration_bindings(
                loop.iterable, loop.target, alias_map)
                if aliases is not None else None)
            if bindings is not None and any(not unshadowed_builtin(
                    expanded, forward, name) for name in builtin_names):
                bindings = None
            if bindings is None:
                unresolved.append(UnresolvedStageChild(
                    stage, "unsupported_iteration",
                    "loop target/iterable cannot be bound exactly to containers",
                    tuple(span for span in (loop.span,) if span is not None)))
                continue
            for field, target in bindings:
                records = tuple(records_by_field.get(field, ()))
                if not records:
                    unresolved.append(UnresolvedStageChild(
                        stage, "container_absent",
                        f"iterated field {field!r} has no exact container record",
                        tuple(span for span in (loop.span,) if span is not None)))
                    continue
                constructions = []
                for record in records:
                    rows, expanded = _container_constructions(
                        expanded, bundle, component, record)
                    constructions.extend(rows)
                if len(records) > 1:
                    unresolved.append(UnresolvedStageChild(
                        stage, "container_rival",
                        f"iterated field {field!r} has rival container constructions",
                        tuple(record.span for record in records
                              if record.span is not None)))
                for call in expanded.calls_in(forward):
                    if call.span is None or loop.body_span is None \
                            or not _within(call.span, loop.body_span) \
                            or call.callee.kind != "name" \
                            or call.callee.name != target:
                        continue
                    matched_calls.add(call)
                    reason = execution_taint_reason(expanded, forward, call)
                    if reason is not None:
                        unresolved.append(UnresolvedStageChild(
                            stage, "call_tainted", reason, (call.span,)))
                        continue
                    if constructions:
                        invocations.append(StageChildInvocation(
                            stage, "repeated", field, call, loop, target,
                            aliases, builtin_names, tuple(constructions)))
                    else:
                        unresolved.append(UnresolvedStageChild(
                            stage, "construction_incomplete",
                            f"iterated field {field!r} has no exact elements",
                            (call.span,)))
        for call in expanded.calls_in(forward):
            field = _self_field(call.callee)
            if call in matched_calls or field is None or call.span is None:
                continue
            constructions, expanded = _direct_constructions(
                expanded, bundle, component, symbol, field)
            if not constructions:
                continue  # self method / non-constructed field, not a child object
            reason = execution_taint_reason(expanded, forward, call)
            if reason is not None:
                unresolved.append(UnresolvedStageChild(
                    stage, "call_tainted", reason, (call.span,)))
                continue
            invocations.append(StageChildInvocation(
                stage, "direct", field, call, None, field, (), (), constructions))
        unresolved.append(UnresolvedStageChild(
            stage, "whole_callable_open",
            "positive child-call inventory; whole-callable CFG coverage is open",
            tuple(dict.fromkeys((
                *(span for span in (forward_record.span,) if span is not None),
                *(item.span for item in expanded.calls_in(forward)
                  if item.span is not None),
            )))))
    inventory = UNetStageCellInventory(
        graph, stages, tuple(invocations), tuple(unresolved), expanded, bundle)
    spans = tuple(dict.fromkeys(
        span for item in (*inventory.invocations, *inventory.unresolved)
        for span in ((item.call.span,) if isinstance(item, StageChildInvocation)
                     else item.spans)))
    return ReaderResult.incomplete(
        graph.owner, inventory,
        failures=(ReaderFailure(
            "incomplete_graph",
            "positive exact child invocations; mechanisms and CFG remain open"),),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact stage construction→container/direct child calls"),))


__all__ = [
    "ChildConstructionEvidence",
    "StageChildInvocation",
    "StageClassOccurrence",
    "StageClassOccurrenceId",
    "UNRESOLVED_CHILD_KINDS",
    "UNetStageCellInventory",
    "UnresolvedStageChild",
    "read_unet_stage_cells",
]
