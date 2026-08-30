"""U11-F2 — exact runtime external-source routes to U-Net attention lanes.

The reader joins F1's checkpoint-selected stage occurrence to D/E's exact
stage, cell, nested-block and attention-lane addresses.  At each invocation it
uses :mod:`invocation_source` to prove one formal-to-formal dataflow edge.  A
lane becomes ``external`` only when the complete contiguous route begins at a
required root formal and the selected stage source proves the root call's
instance guard.  Merely exposing a context formal is never sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_container_interface import default_attention_container_interface
from .attention_invocation_role import framework_attention_invocation_role
from .attention_lane import FrameworkAttentionLaneEvidence
from .component_owner import ComponentRootResolution
from .constructor_condition import resolve_constructor_guard
from .constructor_values import canonical_construction_target, constructor_frame
from .import_source import canonical_called_import_target
from .invocation_source import FormalSourceRoute, bind_formal_edge, compose_formal_route
from .program_index import CallObservation, ExprNode, SourceSpan, SymbolId
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_nested_mechanism import (
    AlternativeNestedOccurrenceId,
    NestedMechanismEvidence,
    UNetNestedMechanismInventory,
)
from .unet_stage_cells import ChildConstructionEvidence, StageClassOccurrence
from .unet_stage_selection import (
    SelectedStageOccurrence,
    UNetStageSelectionInventory,
)


ISSUE_KINDS = frozenset({
    "stage_occurrence_unmatched",
    "lane_route_unresolved",
    "root_call_guard_unresolved",
    "root_source_optional",
    "role_unresolved",
})


def _parent(row: NestedMechanismEvidence):
    oid = row.occurrence_id
    return oid.parent if hasattr(oid, "parent") else oid.alternative.parent


def _forward(index, symbol: SymbolId):
    return index.callable_by_symbol(SymbolId(
        symbol.source, f"{symbol.qualified_name}.forward"))


def _canonical_target(index, bundle, construction: ChildConstructionEvidence,
                      candidate):
    site = construction.site
    if site is None:
        return None
    imported = (canonical_called_import_target(
        bundle, candidate.import_chain[-1])
        if candidate.import_chain else None)
    return canonical_construction_target(
        index, site, candidate.symbol, canonical_import=imported)


def _lane_target(index, lane):
    if lane.child_symbol is None:
        return None
    return canonical_construction_target(
        index, lane.construction, lane.child_symbol,
        canonical_import=lane.canonical_import)


def _selected_stage(selected, nested):
    matches = tuple(item for item in nested.cells.cells.stages
                    if item.construction == selected.source.template
                    and item.candidate == selected.candidate)
    return matches[0] if len(matches) == 1 else None


def _cell_constructions(nested, cell_id):
    rows = {}
    for mechanism in nested.cells.mechanisms:
        if mechanism.occurrence_id != cell_id:
            continue
        for invocation in mechanism.invocations:
            for construction in invocation.constructions:
                for candidate in construction.candidates:
                    construction_span = (construction.site.span
                                         if construction.site is not None
                                         else construction.field_assign.value.span)
                    if candidate.symbol == cell_id.symbol \
                            and candidate.span == cell_id.candidate_span \
                            and construction_span == cell_id.construction_span:
                        rows[(
                            invocation.call.span,
                            construction_span,
                            candidate.symbol,
                            candidate.span,
                        )] = (invocation, construction, candidate)
    return tuple(rows.values())


def _name(expr: ExprNode | None, value: str) -> bool:
    return expr is not None and expr.kind == "name" and expr.name == value


def _selected_true_field(index, selected: SelectedStageOccurrence,
                         target_name: str, test: ExprNode) -> str | None:
    """Return the addressed instance field for one exact true guard pattern."""
    if test.kind != "boolop" or test.operator != "and" \
            or len(test.children) != 2:
        return None
    first, second = test.children
    if first.kind != "call" or len(first.children) != 3 \
            or not _name(first.children[0], "hasattr") \
            or not _name(first.children[1], target_name) \
            or first.children[2].kind != "constant" \
            or not isinstance(first.children[2].const_value, str):
        return None
    field = first.children[2].const_value
    if second.kind != "attribute" or second.name != field \
            or len(second.children) != 1 \
            or not _name(second.children[0], target_name):
        return None
    assignments = tuple(item for item in index.field_assigns_of(
        selected.candidate.symbol)
        if item.field == field and not item.guard
        and item.value.kind == "constant" and item.value.const_value is True)
    return field if len(assignments) == 1 else None


def _root_guard_spans(index, selected: SelectedStageOccurrence,
                      call: CallObservation) -> tuple[SourceSpan, ...] | None:
    source = selected.source
    spans = []
    for step in call.guard:
        if step.kind == "for" and step.span == source.loop.span:
            spans.append(step.span)
            continue
        if step.kind != "if" or step.test is None \
                or _selected_true_field(
                    index, selected, source.value_target, step.test) is None:
            return None
        spans.extend((step.span, step.test.span))
    return tuple(dict.fromkeys(item for item in spans
                               if isinstance(item, SourceSpan)))


@dataclass(frozen=True)
class RuntimeAttentionSource:
    selected_stage: SelectedStageOccurrence
    stage: StageClassOccurrence
    lane: FrameworkAttentionLaneEvidence
    nested: NestedMechanismEvidence
    route: FormalSourceRoute
    root_guard_spans: tuple[SourceSpan, ...]
    e2c_status: str
    e2c_kind: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.stage.candidate != self.selected_stage.candidate \
                or self.stage.construction != self.selected_stage.source.template:
            raise ValueError("runtime source belongs to the exact F1 stage occurrence")
        if self.nested.attention != self.lane \
                or _parent(self.nested).parent != self.stage.occurrence_id:
            raise ValueError("runtime source belongs to the exact nested lane")
        if self.route.owner != self.stage.occurrence_id.owner \
                or not self.route.non_none_external:
            raise ValueError("runtime external source is a required-root route")
        if not self.root_guard_spans:
            raise ValueError("selected-stage root call guard is positively proven")
        if self.e2c_status not in {"resolved", "incomplete", "failed"} \
                or self.e2c_kind not in {
                    "self", "context_slot", "conditional", "unresolved"}:
            raise ValueError("the row retains E2c's exact disposition")
        required = {
            *self.route.spans, *self.root_guard_spans,
            *self.lane.spans, *self.selected_stage.guard_spans,
        }
        if not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("runtime-source provenance closes selection+route+lane")

    @property
    def cross_attention(self) -> bool:
        return True


@dataclass(frozen=True)
class RuntimeAttentionSourceIssue:
    selected_stage: SelectedStageOccurrence
    kind: str
    detail: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.kind not in ISSUE_KINDS or not self.detail:
            raise ValueError("runtime-source issue has closed kind + detail")


@dataclass(frozen=True)
class UNetRuntimeAttentionSources:
    selection: UNetStageSelectionInventory
    nested_inventory: UNetNestedMechanismInventory
    sources: tuple[RuntimeAttentionSource, ...]
    issues: tuple[RuntimeAttentionSourceIssue, ...]

    def __post_init__(self) -> None:
        selected = self.selection.occurrences
        if any(item.selected_stage not in selected for item in self.sources) \
                or any(item.selected_stage not in selected for item in self.issues):
            raise ValueError("every F2 row belongs to an authoritative F1 occurrence")
        identities = tuple((item.selected_stage.source.template.topology_stage.field,
                            item.selected_stage.position,
                            item.lane.invocation.call.span,
                            item.route.spans)
                           for item in self.sources)
        if len(identities) != len(set(identities)):
            raise ValueError("runtime source rows retain exact occurrence identity")


def _lane_edge(index, owner, block_frame, lane):
    if block_frame is None \
            or lane.block_occurrence != block_frame.graph.root.occurrence:
        return None, "failed", "unresolved", "lane/block occurrence mismatch"
    target = _lane_target(index, lane)
    if target is None:
        return None, "failed", "unresolved", "lane target unavailable"
    try:
        child_frame = constructor_frame(index, target, block_frame)
    except ValueError:
        return None, "failed", "unresolved", "lane constructor frame rejected"
    if child_frame is None:
        return None, "failed", "unresolved", "lane constructor frame absent"
    interface_result = default_attention_container_interface(index, child_frame)
    if interface_result.status != "resolved":
        return None, "failed", "unresolved", "lane interface unresolved"
    interface = interface_result.require_value()
    role = framework_attention_invocation_role(index, block_frame, lane)
    role_kind = (role.value.kind if role.has_value else "unresolved")
    edge = bind_formal_edge(
        index, owner, lane.invocation.call, interface.forward,
        interface.context_formal.name)
    return (edge.value if edge.status == "resolved" else None,
            role.status, role_kind,
            "" if edge.status == "resolved" else "lane context binding unresolved")


def _frame_guard_resolver(index, frame, callable_symbol):
    """Resolve only constructor-decidable binding guards for one frame."""
    def resolve(binding):
        if not binding.guard:
            return True, ()
        result = resolve_constructor_guard(
            index, frame, callable_symbol, binding.guard, binding.span)
        if result.status != "resolved":
            return None, ()
        decision = result.require_value()
        return decision.decision, decision.spans
    return resolve


def _routes_for_lane(selected, stage, nested, row,
                     lane_cache):
    if not isinstance(row.occurrence_id, AlternativeNestedOccurrenceId) \
            or not isinstance(row.attention, FrameworkAttentionLaneEvidence):
        return (), "failed", "unresolved", "lane has no exact constructor alternative"
    alternative = row.occurrence_id.alternative
    cell_id = alternative.parent
    constructions = _cell_constructions(nested, cell_id)
    if not constructions:
        return (), "failed", "unresolved", "stage-to-cell construction is unavailable"
    index = nested.index
    bundle = nested.cells.cells.bundle
    routes = []
    role_status = "failed"
    role_kind = "unresolved"
    attempts = {"frame": 0, "lane": 0, "block": 0, "stage": 0,
                "root_guard": 0, "root": 0, "composed": 0}
    lane_failures = []
    for invocation, construction, candidate in constructions:
        transformer_target = _canonical_target(
            index, bundle, construction, candidate)
        if transformer_target is None:
            continue
        try:
            transformer_frame = constructor_frame(index, transformer_target)
            block_target = canonical_construction_target(
                index, alternative.site, alternative.candidate.symbol,
                canonical_import=(canonical_called_import_target(
                    bundle, alternative.candidate.import_chain[-1])
                    if alternative.candidate.import_chain else None))
            if block_target is None:
                continue
            block_frame = constructor_frame(index, block_target, transformer_frame)
        except ValueError:
            continue
        attempts["frame"] += 1
        cache_key = (alternative.site.span, row.attention.invocation.call.span)
        if cache_key not in lane_cache:
            lane_cache[cache_key] = _lane_edge(
                index, stage.occurrence_id.owner, block_frame, row.attention)
        lane_edge, role_status, role_kind, lane_detail = lane_cache[cache_key]
        if lane_edge is None:
            lane_failures.append(lane_detail)
            continue
        attempts["lane"] += 1
        block_forward = _forward(index, alternative.symbol)
        transformer_forward = _forward(index, cell_id.symbol)
        stage_forward = _forward(index, stage.occurrence_id.symbol)
        if None in {block_forward, transformer_forward, stage_forward}:
            continue
        block_edge_result = bind_formal_edge(
            index, stage.occurrence_id.owner, alternative.invocation,
            block_forward, lane_edge.caller_formal.name,
            binding_guard_resolver=_frame_guard_resolver(
                index, transformer_frame,
                alternative.invocation.enclosing_callable))
        if block_edge_result.status != "resolved":
            continue
        block_edge = block_edge_result.require_value()
        attempts["block"] += 1
        stage_edge_result = bind_formal_edge(
            index, stage.occurrence_id.owner, invocation.call,
            transformer_forward, block_edge.caller_formal.name)
        if stage_edge_result.status != "resolved":
            continue
        stage_edge = stage_edge_result.require_value()
        attempts["stage"] += 1
        for root_call in selected.source.template.topology_stage.calls:
            guard_spans = _root_guard_spans(index, selected, root_call)
            if guard_spans is None:
                continue
            attempts["root_guard"] += 1
            root_edge_result = bind_formal_edge(
                index, stage.occurrence_id.owner, root_call, stage_forward,
                stage_edge.caller_formal.name)
            if root_edge_result.status != "resolved":
                continue
            attempts["root"] += 1
            try:
                route = compose_formal_route((
                    root_edge_result.require_value(), stage_edge,
                    block_edge, lane_edge))
            except ValueError:
                continue
            routes.append((route, guard_spans))
            attempts["composed"] += 1
    unique = []
    for item in routes:
        if item not in unique:
            unique.append(item)
    detail = "" if unique else "formal-route attempts " + ", ".join(
        f"{name}={count}" for name, count in attempts.items()) + (
            f"; lane={tuple(dict.fromkeys(lane_failures))!r}"
            if lane_failures else "")
    return tuple(unique), role_status, role_kind, detail


def read_unet_runtime_attention_sources(
        selection: UNetStageSelectionInventory,
        nested: UNetNestedMechanismInventory,
        root_resolution: ComponentRootResolution,
) -> ReaderResult[UNetRuntimeAttentionSources]:
    if not isinstance(selection, UNetStageSelectionInventory) \
            or not isinstance(nested, UNetNestedMechanismInventory) \
            or not isinstance(root_resolution, ComponentRootResolution):
        raise TypeError("U11-F2 requires exact F1/E1/D0 evidence")
    owner = nested.cells.cells.graph.owner
    if root_resolution.status != "resolved" \
            or root_resolution.occurrence != owner:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "F1/E1/D0 evidence does not share one root"),))
    sources = []
    issues = []
    lane_cache = {}
    for selected in selection.occurrences:
        stage = _selected_stage(selected, nested)
        if stage is None:
            issues.append(RuntimeAttentionSourceIssue(
                selected, "stage_occurrence_unmatched",
                "selected F1 construction has no exact D1 stage occurrence"))
            continue
        rows = tuple(row for row in nested.mechanisms
                     if _parent(row).parent == stage.occurrence_id
                     and isinstance(row.attention, FrameworkAttentionLaneEvidence))
        for row in rows:
            routes, role_status, role_kind, detail = _routes_for_lane(
                selected, stage, nested, row, lane_cache)
            if not routes:
                issues.append(RuntimeAttentionSourceIssue(
                    selected, "lane_route_unresolved",
                    detail or "the exact lane has no contiguous external route",
                    row.attention.invocation.call.span))
                continue
            for route, guard_spans in routes:
                if not route.non_none_external:
                    issues.append(RuntimeAttentionSourceIssue(
                        selected, "root_source_optional",
                        "the exact root source formal permits None",
                        row.attention.invocation.call.span))
                    continue
                spans = tuple(dict.fromkeys((
                    *selected.guard_spans, *route.spans, *guard_spans,
                    *row.attention.spans,
                )))
                sources.append(RuntimeAttentionSource(
                    selected, stage, row.attention, row, route, guard_spans,
                    role_status, role_kind, spans))
    value = UNetRuntimeAttentionSources(
        selection, nested, tuple(sources), tuple(issues))
    spans = tuple(dict.fromkeys((
        *(span for item in sources for span in item.spans),
        *(span for issue in issues for span in (
            issue.span, *issue.selected_stage.guard_spans)
          if isinstance(span, SourceSpan)),
    )))
    provenance = ((ReaderProvenance(
        "source", spans=spans,
        detail="exact selected-stage -> cell -> block -> attention formal route"),)
        if spans else ())
    if issues:
        return ReaderResult.incomplete(
            owner, value,
            failures=(ReaderFailure(
                "incomplete_graph",
                "some selected-stage attention sources remain unresolved"),),
            provenance=provenance)
    return ReaderResult.resolved(owner, value, provenance=provenance)


__all__ = [
    "ISSUE_KINDS",
    "RuntimeAttentionSource",
    "RuntimeAttentionSourceIssue",
    "UNetRuntimeAttentionSources",
    "read_unet_runtime_attention_sources",
]
