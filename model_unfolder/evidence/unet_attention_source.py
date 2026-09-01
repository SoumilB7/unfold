"""U11-F2 — exact runtime external-source routes to U-Net attention lanes.

The reader joins F1's checkpoint-selected stage occurrence to D/E's exact
stage, cell, nested-block and attention-lane addresses.  At each invocation it
uses :mod:`invocation_source` to prove one formal-to-formal dataflow edge.  A
runtime context-source row exists only when the complete contiguous route begins
at a required non-optional root formal and the selected stage source proves the
root call's instance guard.  This boundary deliberately does not expose a
``cross_attention`` boolean: F4/G must still distinguish root conditioning from
the model's state carrier before projecting that architectural label.  Merely
exposing a context formal is never sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_container_interface import default_attention_container_interface
from .attention_invocation_role import framework_attention_invocation_role
from .attention_lane import FrameworkAttentionLaneEvidence
from .component_owner import ComponentRootResolution
from .constructor_condition import (
    resolve_constructor_guard,
    select_constructor_conditioned_call_argument,
)
from .constructor_values import canonical_construction_target, constructor_frame
from .import_source import canonical_called_import_target
from .invocation_source import FormalSourceRoute, bind_formal_edge, compose_formal_route
from .program_index import CallObservation, ExprNode, SourceSpan, SymbolId
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_selected_child_execution import (
    SelectedChildExecution,
    UNetSelectedChildExecution,
)
from .unet_selected_constructor import (
    selected_constructor_environments,
    selected_guard_evidence,
    selected_instance_guard_evidence,
)
from .unet_nested_mechanism import (
    AlternativeNestedOccurrenceId,
    NestedMechanismEvidence,
    UNetNestedMechanismInventory,
)
from .unet_stage_cells import (
    ChildConstructionEvidence,
    StageChildInvocation,
    StageClassOccurrence,
)
from .unet_stage_selection import (
    SelectedStageOccurrence,
    UNetStageSelectionInventory,
)


ISSUE_KINDS = frozenset({
    "stage_occurrence_unmatched",
    "lane_route_unresolved",
    "root_call_guard_unresolved",
    "root_preprocess_unresolved",
    "root_source_optional",
    "role_unresolved",
    "child_execution_unmatched",
    "constructor_route_unresolved",
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


def _construction_span(construction):
    return (construction.site.span if construction.site is not None
            else construction.field_assign.value.span)


def _call_address(call):
    return (call.enclosing_callable, call.span)


def _invocation_address(invocation):
    """Occurrence identity stable across a monotonic ProgramIndex expansion."""
    return (
        invocation.parent.occurrence_id,
        invocation.kind,
        invocation.field,
        _call_address(invocation.call),
        invocation.loop.span if invocation.loop is not None else None,
        invocation.target,
    )


def _construction_address(construction):
    return (
        construction.field,
        _construction_span(construction),
        tuple((item.symbol, item.span) for item in construction.candidates),
    )


def _execution_matches(selected, stage, nested_row, execution_inventory):
    parent = _parent(nested_row)
    rows = []
    for execution in execution_inventory.executions:
        population = execution.population
        if population.selected != selected or population.stage != stage \
                or population.field != parent.field:
            continue
        for construction in population.present_constructions:
            if len(construction.candidates) != 1:
                continue
            candidate = construction.candidates[0]
            if candidate.symbol == parent.symbol \
                    and candidate.span == parent.candidate_span \
                    and _construction_span(construction) \
                    == parent.construction_span:
                rows.append((execution, construction))
    return tuple(rows)


def _selected_alternative(environments, alternative):
    """Prove that one exact helper/site route is active for this occurrence."""
    matches = environments.for_callable(alternative.site.enclosing_callable)
    if not matches:
        return False, (), "nested constructor helper is not positively reached"
    selected = []
    unresolved = []
    # Comprehension/for steps prove symbolic population, not a conditional
    # constructor choice.  E1 already retains that exact template; F2b must
    # decide only its source-level branch guards.
    guard = tuple(step for step in alternative.site.guard
                  if step.kind not in {"for", "comprehension"})
    for environment in matches:
        evidence = selected_guard_evidence(
            environments, alternative.site.enclosing_callable, guard,
            alternative.site.span, helper_route=environment.helper_route)
        if evidence is None or type(evidence.value) is not bool:
            unresolved.append(environment)
        elif evidence.value:
            selected.append((environment, evidence))
    if unresolved:
        return None, (), "nested constructor branch is not exactly decidable"
    if len(selected) != 1:
        return (False if not selected else None), (), (
            "nested constructor route is inactive" if not selected else
            "several exact helper routes select one nested constructor")
    environment, evidence = selected[0]
    spans = tuple(dict.fromkeys((
        alternative.site.span, *environment.spans, *evidence.spans,
        *(step.span for step in guard),
    )))
    return True, spans, ""


def _name(expr: ExprNode | None, value: str) -> bool:
    return expr is not None and expr.kind == "name" and expr.name == value


def _selected_true_field(index, selected: SelectedStageOccurrence,
                         target_name: str, test: ExprNode):
    """Return the exact constructor write proving one true instance guard."""
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
    return assignments[0] if len(assignments) == 1 else None


def _root_guard_spans(index, selected: SelectedStageOccurrence,
                      call: CallObservation) -> tuple[SourceSpan, ...] | None:
    source = selected.source
    runtime_stage = source.template.topology_stage
    spans = []
    for step in call.guard:
        if step.kind == "for" and step.span == runtime_stage.loop.span:
            spans.append(step.span)
            continue
        assignment = (_selected_true_field(
            index, selected, runtime_stage.element_target, step.test)
                      if step.kind == "if" and step.test is not None else None)
        if assignment is None:
            return None
        spans.extend((step.span, step.test.span, assignment.span))
    return tuple(dict.fromkeys(item for item in spans
                               if isinstance(item, SourceSpan)))


@dataclass(frozen=True)
class RuntimeAttentionAlternativeRoute:
    """One exact runtime alternative's complete external-source route."""

    invocation: StageChildInvocation
    route: FormalSourceRoute
    root_guard_spans: tuple[SourceSpan, ...]
    constructor_guard_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, StageChildInvocation) \
                or not self.route.non_none_external \
                or sum(_call_address(edge.call)
                       == _call_address(self.invocation.call)
                       for edge in self.route.edges) != 1:
            raise ValueError(
                "an alternative route closes one exact runtime invocation")
        if any(not isinstance(item, SourceSpan) for item in (
                *self.root_guard_spans, *self.constructor_guard_spans)):
            raise ValueError("alternative-route guards retain exact source spans")


@dataclass(frozen=True)
class RuntimeAttentionSource:
    """A non-optional root-formal route to K/V, not yet a cross-attention fact."""
    selected_stage: SelectedStageOccurrence
    stage: StageClassOccurrence
    child_execution: SelectedChildExecution
    lane: FrameworkAttentionLaneEvidence
    nested: NestedMechanismEvidence
    routes: tuple[RuntimeAttentionAlternativeRoute, ...]
    e2c_status: str
    e2c_kind: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.stage.candidate != self.selected_stage.candidate \
                or self.stage.construction != self.selected_stage.source.template:
            raise ValueError("runtime source belongs to the exact F1 stage occurrence")
        if self.child_execution.population.selected != self.selected_stage \
                or self.child_execution.population.stage != self.stage:
            raise ValueError("runtime source belongs to one positive F3 execution")
        if self.nested.attention != self.lane \
                or _parent(self.nested).parent != self.stage.occurrence_id:
            raise ValueError("runtime source belongs to the exact nested lane")
        if not self.routes \
                or tuple(item.invocation for item in self.routes) \
                != self.child_execution.runtime_invocations \
                or any(item.route.owner != self.stage.occurrence_id.owner
                       for item in self.routes):
            raise ValueError(
                "runtime source carries one exact route per runtime alternative")
        if any(edge.argument_selection is not None
               and edge.argument_selection.frame.graph.root.occurrence
               != self.lane.block_occurrence
               for item in self.routes for edge in item.route.edges):
            raise ValueError(
                "a selected context argument belongs to this exact block occurrence")
        route_keys = tuple((item.route, item.invocation.call.span)
                           for item in self.routes)
        if len(route_keys) != len(set(route_keys)):
            raise ValueError("runtime alternative routes retain unique identity")
        endpoints = {
            (item.route.source_callable, item.route.source_formal,
             item.route.target_callable, item.route.target_formal,
             item.route.source_kind)
            for item in self.routes
        }
        if len(endpoints) != 1:
            raise ValueError("runtime alternatives prove one exact source endpoint")
        if self.e2c_status not in {"resolved", "incomplete", "failed"} \
                or self.e2c_kind not in {
                    "self", "context_slot", "conditional", "unresolved"}:
            raise ValueError("the row retains E2c's exact disposition")
        required = {
            *(span for item in self.routes for span in item.route.spans),
            *(span for item in self.routes for span in item.root_guard_spans),
            *(span for item in self.routes
              for span in item.constructor_guard_spans),
            *self.lane.spans, *self.selected_stage.guard_spans,
        }
        if not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("runtime-source provenance closes selection+route+lane")

@dataclass(frozen=True)
class RuntimeAttentionSourceIssue:
    selected_stage: SelectedStageOccurrence
    kind: str
    detail: str
    span: SourceSpan | None = None
    child_execution: SelectedChildExecution | None = None

    def __post_init__(self) -> None:
        if self.kind not in ISSUE_KINDS or not self.detail:
            raise ValueError("runtime-source issue has closed kind + detail")
        if self.child_execution is not None \
                and self.child_execution.population.selected != self.selected_stage:
            raise ValueError("runtime-source issue keeps its exact F3 execution")


@dataclass(frozen=True)
class UNetRuntimeAttentionSources:
    selection: UNetStageSelectionInventory
    nested_inventory: UNetNestedMechanismInventory
    child_execution: UNetSelectedChildExecution
    sources: tuple[RuntimeAttentionSource, ...] | None
    issues: tuple[RuntimeAttentionSourceIssue, ...] | None

    def __post_init__(self) -> None:
        base_index = self.child_execution.index
        expanded_index = self.nested_inventory.index
        if self.child_execution.children.operands.factory.selection \
                != self.selection \
                or base_index.bundle_source != expanded_index.bundle_source \
                or any(node not in expanded_index.source_nodes
                       for node in base_index.source_nodes) \
                or any(failure not in expanded_index.parse_failures
                       for failure in base_index.parse_failures) \
                or any(expanded_index.callable_by_symbol(
                           item.constructor.symbol) != item.constructor
                       for item in self.child_execution.operands) \
                or any(invocation.call not in expanded_index.calls_in(
                           invocation.call.enclosing_callable)
                       for execution in self.child_execution.executions
                       for invocation in execution.runtime_invocations):
            raise ValueError("F2b retains one exact F1/F3/E1 evidence universe")
        expected_sources, expected_issues = _derive(
            self.selection, self.nested_inventory, self.child_execution)
        if self.sources is None and self.issues is None:
            object.__setattr__(self, "sources", expected_sources)
            object.__setattr__(self, "issues", expected_issues)
        elif self.sources is None or self.issues is None \
                or (self.sources, self.issues) \
                != (expected_sources, expected_issues):
            raise ValueError("runtime attention sources recompute exactly")
        selected = self.selection.occurrences
        if any(item.selected_stage not in selected for item in self.sources) \
                or any(item.selected_stage not in selected for item in self.issues):
            raise ValueError("every F2 row belongs to an authoritative F1 occurrence")
        identities = tuple((item.selected_stage.source.template.topology_stage.field,
                            item.selected_stage.position,
                            item.lane.invocation.call.span,
                            tuple(route.invocation.call.span
                                  for route in item.routes),
                            tuple(route.route.spans for route in item.routes))
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
        detail = "; ".join(item.detail for item in interface_result.failures
                           if item.detail)
        return None, "failed", "unresolved", (
            "lane interface unresolved" + (f": {detail}" if detail else ""))
    interface = interface_result.require_value()
    role = framework_attention_invocation_role(index, block_frame, lane)
    role_kind = (role.value.kind if role.has_value else "unresolved")
    ordinary = tuple(item for item in interface.forward.params
                     if item.name != "self"
                     and item.kind not in {"vararg", "kwarg"})
    positional = tuple(item for item in ordinary
                       if item.kind in {"positional", "posonly"})
    target = next((item for item in ordinary
                   if item.name == interface.context_formal.name), None)
    original = None
    if target is not None and target in positional:
        position = positional.index(target)
        if position < len(lane.invocation.call.args):
            original = lane.invocation.call.args[position]
    keywords = tuple(value for name, value in lane.invocation.call.kwargs
                     if name == interface.context_formal.name)
    if len(keywords) == 1 and original is None:
        original = keywords[0]
    elif keywords:
        original = None
    selection = None
    if original is not None and original.kind == "ifexp":
        selected = select_constructor_conditioned_call_argument(
            index, block_frame, lane.invocation.call, original)
        if selected.status != "resolved":
            return None, role.status, role_kind, (
                "lane context branch is not constructor-decidable")
        selection = selected.require_value()
    edge = bind_formal_edge(
        index, owner, lane.invocation.call, interface.forward,
        interface.context_formal.name, argument_selection=selection)
    return (edge.value if edge.status == "resolved" else None,
            role.status, role_kind,
            "" if edge.status == "resolved" else "lane context binding unresolved")


def _frame_guard_resolver(index, frame, callable_symbol, environments=None):
    """Resolve only constructor-decidable binding guards for one frame."""
    def resolve(binding):
        if not binding.guard:
            return True, ()
        if environments is not None:
            evidence = selected_instance_guard_evidence(
                environments, callable_symbol, binding.guard, binding.span)
            if evidence is not None and type(evidence.value) is bool:
                return evidence.value, evidence.spans
        result = resolve_constructor_guard(
            index, frame, callable_symbol, binding.guard, binding.span)
        if result.status != "resolved":
            return None, ()
        decision = result.require_value()
        return decision.decision, decision.spans
    return resolve


def _routes_for_lane(selected, stage, nested, row, child_execution,
                     construction, environments, lane_cache):
    if not isinstance(row.occurrence_id, AlternativeNestedOccurrenceId) \
            or not isinstance(row.attention, FrameworkAttentionLaneEvidence):
        return (), "failed", "unresolved", "lane_route_unresolved", \
            "lane has no exact constructor alternative"
    alternative = row.occurrence_id.alternative
    cell_id = alternative.parent
    constructions = tuple(
        item for item in _cell_constructions(nested, cell_id)
        if _invocation_address(item[0]) in {
            _invocation_address(value)
            for value in child_execution.runtime_invocations
        }
        and _construction_address(item[1])
        == _construction_address(construction))
    if not constructions:
        return (), "failed", "unresolved", "lane_route_unresolved", (
            "the exact positive child execution does not construct this cell")
    alternative_state, alternative_spans, alternative_detail = \
        _selected_alternative(environments, alternative)
    if alternative_state is not True:
        return (), "failed", "unresolved", "lane_route_unresolved", \
            alternative_detail
    index = nested.index
    bundle = nested.cells.cells.bundle
    routes = []
    role_status = "failed"
    role_kind = "unresolved"
    attempts = {"frame": 0, "lane": 0, "block": 0, "stage": 0,
                "root_guard": 0, "root": 0, "composed": 0}
    lane_failures = []
    edge_failures = []
    for invocation, construction, candidate in constructions:
        base_invocations = tuple(
            item for item in child_execution.runtime_invocations
            if _invocation_address(item) == _invocation_address(invocation))
        if len(base_invocations) != 1:
            continue
        base_invocation = base_invocations[0]
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
        cache_key = (
            child_execution.population.selected.source.template.topology_stage.field,
            child_execution.population.selected.source.template.topology_order,
            child_execution.population.selected.position,
            _construction_span(construction), alternative.site.span,
            row.attention.invocation.call.span,
        )
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
                alternative.invocation.enclosing_callable,
                environments))
        if block_edge_result.status != "resolved":
            edge_failures.extend(
                f"block:{item.detail}" for item in block_edge_result.failures)
            continue
        block_edge = block_edge_result.require_value()
        attempts["block"] += 1
        stage_edge_result = bind_formal_edge(
            index, stage.occurrence_id.owner, invocation.call,
            transformer_forward, block_edge.caller_formal.name)
        if stage_edge_result.status != "resolved":
            edge_failures.extend(
                f"stage:{item.detail}" for item in stage_edge_result.failures)
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
                edge_failures.extend(
                    f"root:{item.detail}"
                    for item in root_edge_result.failures)
                continue
            attempts["root"] += 1
            try:
                route = compose_formal_route((
                    root_edge_result.require_value(), stage_edge,
                    block_edge, lane_edge))
            except ValueError:
                continue
            routes.append(RuntimeAttentionAlternativeRoute(
                base_invocation, route, guard_spans, alternative_spans))
            attempts["composed"] += 1
    by_invocation = []
    for invocation in child_execution.runtime_invocations:
        matches = tuple(item for item in routes
                        if _invocation_address(item.invocation)
                        == _invocation_address(invocation))
        if len(matches) != 1:
            detail = (
                "runtime alternative has no unique complete formal route: "
                f"call={invocation.call.span!r}, candidates={len(matches)}; "
                + ", ".join(
                    f"{name}={count}" for name, count in attempts.items())
                + (f"; lane={tuple(dict.fromkeys(lane_failures))!r}"
                   if lane_failures else "")
                + (f"; edges={tuple(dict.fromkeys(edge_failures))!r}"
                   if edge_failures else ""))
            issue_kind = ("root_preprocess_unresolved"
                          if attempts["root_guard"] and edge_failures
                          and any(item.startswith("root:")
                                  for item in edge_failures)
                          else "lane_route_unresolved")
            return (), role_status, role_kind, issue_kind, detail
        by_invocation.append(matches[0])
    if len(routes) != len(by_invocation):
        return (), role_status, role_kind, "lane_route_unresolved", (
            "formal route set contains an unattributed runtime alternative")
    detail = "" if by_invocation else "formal-route attempts " + ", ".join(
        f"{name}={count}" for name, count in attempts.items()) + (
            f"; lane={tuple(dict.fromkeys(lane_failures))!r}"
            if lane_failures else "") + (
            f"; edges={tuple(dict.fromkeys(edge_failures))!r}"
            if edge_failures else "")
    return (tuple(by_invocation), role_status, role_kind,
            "lane_route_unresolved", detail)


def _derive(selection, nested, child_execution):
    sources = []
    issues = []
    lane_cache = {}
    environment_cache = {}
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
            matches = _execution_matches(
                selected, stage, row, child_execution)
            if len(matches) != 1:
                issues.append(RuntimeAttentionSourceIssue(
                    selected, "child_execution_unmatched",
                    "the exact nested cell has no unique positive F3 execution",
                    row.attention.invocation.call.span))
                continue
            execution, construction = matches[0]
            operands = tuple(
                item for item in child_execution.operands
                if item.population == execution.population
                and item.construction == construction)
            if not operands:
                issues.append(RuntimeAttentionSourceIssue(
                    selected, "constructor_route_unresolved",
                    "the positive child has no exact selected constructor operands",
                    row.attention.invocation.call.span, execution))
                continue
            environment_key = (
                selected.source.template.topology_stage.field,
                selected.source.template.topology_order, selected.position,
                _construction_span(construction))
            if environment_key not in environment_cache:
                environment_cache[environment_key] = (
                    selected_constructor_environments(
                        nested.index, operands))
            environments = environment_cache[environment_key]
            routes, role_status, role_kind, issue_kind, detail = _routes_for_lane(
                selected, stage, nested, row, execution, construction,
                environments, lane_cache)
            if not routes:
                issues.append(RuntimeAttentionSourceIssue(
                    selected, issue_kind,
                    detail or "the exact lane has no contiguous external route",
                    row.attention.invocation.call.span, execution))
                continue
            if any(not item.route.non_none_external for item in routes):
                issues.append(RuntimeAttentionSourceIssue(
                    selected, "root_source_optional",
                    "one exact root source formal permits None",
                    row.attention.invocation.call.span, execution))
                continue
            spans = tuple(dict.fromkeys((
                *selected.guard_spans,
                *(span for item in routes for span in item.route.spans),
                *(span for item in routes for span in item.root_guard_spans),
                *(span for item in routes
                  for span in item.constructor_guard_spans),
                *row.attention.spans,
            )))
            sources.append(RuntimeAttentionSource(
                selected, stage, execution, row.attention, row, routes,
                role_status, role_kind, spans))
    return tuple(sources), tuple(issues)


def read_unet_runtime_attention_sources(
        selection: UNetStageSelectionInventory,
        nested: UNetNestedMechanismInventory,
        child_execution: UNetSelectedChildExecution,
        root_resolution: ComponentRootResolution,
) -> ReaderResult[UNetRuntimeAttentionSources]:
    if not isinstance(selection, UNetStageSelectionInventory) \
            or not isinstance(nested, UNetNestedMechanismInventory) \
            or not isinstance(child_execution, UNetSelectedChildExecution) \
            or not isinstance(root_resolution, ComponentRootResolution):
        raise TypeError("U11-F2b requires exact F1/E1/F3d/D0 evidence")
    owner = nested.cells.cells.graph.owner
    if root_resolution.status != "resolved" \
            or root_resolution.occurrence != owner \
            or child_execution.children.cells.graph.owner != owner:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "F1/E1/F3d/D0 evidence does not share one root"),))
    value = UNetRuntimeAttentionSources(
        selection, nested, child_execution, None, None)
    sources, issues = value.sources, value.issues
    spans = tuple(dict.fromkeys((
        *(span for item in selection.occurrences
          for span in item.guard_spans),
        *(span for item in sources for span in item.spans),
        *(span for issue in issues for span in (
            issue.span, *issue.selected_stage.guard_spans)
          if isinstance(span, SourceSpan)),
    )))
    provenance = ((ReaderProvenance(
        "source", spans=spans,
        detail="exact selected-stage -> cell -> block -> attention formal route"),)
        if spans else ())
    if issues or not sources:
        return ReaderResult.incomplete(
            owner, value,
            failures=(ReaderFailure(
                "incomplete_graph",
                ("some selected-stage attention sources remain unresolved"
                 if issues else
                 "no selected-stage attention source is positively proven")),),
            provenance=provenance)
    return ReaderResult.resolved(owner, value, provenance=provenance)


__all__ = [
    "ISSUE_KINDS",
    "RuntimeAttentionAlternativeRoute",
    "RuntimeAttentionSource",
    "RuntimeAttentionSourceIssue",
    "UNetRuntimeAttentionSources",
    "read_unet_runtime_attention_sources",
]
