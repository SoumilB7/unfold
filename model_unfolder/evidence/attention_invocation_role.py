"""Occurrence-qualified self vs contextual-input attention call evidence.

This join starts from an already-proven framework attention lane, its exact
parent constructor frame, and the source-proven container input interface.  It
binds Python call syntax to the interface formals, reduces only constructor-
decidable conditional arguments, and traces the selected expressions to the
parent callable's formals.

``context_slot`` deliberately does not yet mean that non-``None`` external
conditioning reaches the model at runtime.  It proves that this lane's K/V
interface consumes a distinct parent context formal when supplied.  U11-F owns
the later component-bound conditioning join.  This distinction prevents a
context-capable lane from being mislabeled as always-cross-attention.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_container_interface import (
    DefaultAttentionContainerInterface,
    default_attention_container_interface,
)
from .attention_lane import FrameworkAttentionLaneEvidence
from .constructor_condition import SelectedConstructorCallArgument
from .constructor_condition import (
    ConstructorGuardDecision,
    resolve_constructor_guard,
    select_constructor_conditioned_call_argument,
)
from .constructor_values import (
    ConstructorFrame,
    canonical_construction_target,
    constructor_frame,
)
from .diffusion_stream import local_lineage_at_callable
from .decoder_norm import norm_preserving_invocations_in_frame
from .program_index import (
    CallObservation,
    ExprNode,
    GuardStep,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


ROLE_KINDS = frozenset({"self", "context_slot", "conditional"})
ALTERNATIVE_ROLE_KINDS = frozenset({"self", "context_slot"})


@dataclass(frozen=True)
class AttentionInputRoleAlternative:
    """One exact unresolved-branch input role and its caller-formal lineage."""

    expression: ExprNode
    roots: tuple[str, ...]
    lineage_spans: tuple[SourceSpan, ...]
    kind: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.expression, ExprNode) \
                or self.kind not in ALTERNATIVE_ROLE_KINDS \
                or tuple(sorted(set(self.roots))) != self.roots \
                or len(self.roots) > 1:
            raise ValueError("a role alternative has one closed exact source")
        none_value = self.expression.kind == "constant" \
            and self.expression.const_value is None
        if self.kind == "context_slot" and (none_value or len(self.roots) != 1) \
                or self.kind == "self" and not (none_value or len(self.roots) == 1):
            raise ValueError("alternative kind agrees with its exact expression")
        if none_value and self.lineage_spans:
            raise ValueError("a None alternative carries no invented lineage")
        required = {self.expression.span, *self.lineage_spans}
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("alternative provenance closes expression + lineage")


@dataclass(frozen=True)
class ConstructedLanePresenceDecision:
    """An exact lane construction proves its own ``is not None`` guard true."""

    lane: FrameworkAttentionLaneEvidence
    step: GuardStep
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.lane, FrameworkAttentionLaneEvidence) \
                or not isinstance(self.step, GuardStep) \
                or self.step not in self.lane.invocation.call.guard \
                or not _is_exact_lane_presence_step(self.step, self.lane):
            raise ValueError("lane-presence proof closes its exact field guard")
        required = {
            self.step.span, self.lane.invocation.call.span,
            self.lane.construction.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("lane-presence proof retains call/site/guard spans")


@dataclass(frozen=True)
class FrameworkAttentionInvocationRole:
    """One exact lane call bound to self or a distinct context input slot."""

    lane: FrameworkAttentionLaneEvidence
    block_frame: ConstructorFrame
    interface: DefaultAttentionContainerInterface
    primary_expression: ExprNode
    context_expression: ExprNode | None
    context_selection: SelectedConstructorCallArgument | None
    primary_roots: tuple[str, ...]
    context_roots: tuple[str, ...]
    primary_lineage_spans: tuple[SourceSpan, ...]
    context_lineage_spans: tuple[SourceSpan, ...]
    presence_decisions: tuple[ConstructedLanePresenceDecision, ...]
    guard_decisions: tuple[ConstructorGuardDecision, ...]
    alternatives: tuple[AttentionInputRoleAlternative, ...]
    kind: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.lane, FrameworkAttentionLaneEvidence) \
                or not isinstance(self.block_frame, ConstructorFrame) \
                or not isinstance(
                    self.interface, DefaultAttentionContainerInterface):
            raise TypeError("an invocation role retains lane/frame/interface")
        if self.lane.block_occurrence \
                != self.block_frame.graph.root.occurrence \
                or self.interface.frame.parent != self.block_frame \
                or self.interface.frame.target.site != self.lane.construction \
                or self.interface.frame.target.symbol != self.lane.child_symbol:
            raise ValueError("lane, parent frame and container frame are identical")
        call = self.lane.invocation.call
        arguments = (*call.args, *(value for _name, value in call.kwargs))
        if not isinstance(self.primary_expression, ExprNode) \
                or self.primary_expression not in arguments \
                or self.context_expression is not None \
                and self.context_expression not in arguments:
            raise ValueError("role expressions are exact lane-call arguments")
        if self.context_selection is not None:
            if not isinstance(
                    self.context_selection, SelectedConstructorCallArgument) \
                    or self.context_selection.frame != self.block_frame \
                    or self.context_selection.call != call \
                    or self.context_selection.original \
                    != self.context_expression:
                raise ValueError("context selection closes this exact call branch")
            selected = self.context_selection.selected
        elif not self.alternatives:
            selected = self.context_expression
            if selected is not None and selected.kind == "ifexp":
                raise ValueError("conditional context needs exact branch evidence")
        else:
            selected = None
            if self.context_expression is None \
                    or self.context_expression.kind != "ifexp" \
                    or len(self.context_expression.children) != 3 \
                    or len(self.alternatives) != 2 \
                    or tuple(item.expression for item in self.alternatives) != (
                        self.context_expression.children[0],
                        self.context_expression.children[2]):
                raise ValueError("role alternatives close both exact if branches")
        if self.kind not in ROLE_KINDS \
                or tuple(sorted(set(self.primary_roots))) != self.primary_roots \
                or tuple(sorted(set(self.context_roots))) != self.context_roots \
                or len(self.primary_roots) != 1:
            raise ValueError("input role has closed kind and exact formal roots")
        if not self.primary_lineage_spans \
                or any(not isinstance(item, SourceSpan)
                       for item in (*self.primary_lineage_spans,
                                    *self.context_lineage_spans)):
            raise ValueError("input role retains exact lineage provenance")
        if len(self.presence_decisions) != len(_unique_values(
                self.presence_decisions)) \
                or any(not isinstance(item, ConstructedLanePresenceDecision)
                       or item.lane != self.lane
                       for item in self.presence_decisions):
            raise ValueError("presence decisions belong to this exact lane")
        if len(self.guard_decisions) != len(_unique_values(
                self.guard_decisions)) \
                or any(not isinstance(item, ConstructorGuardDecision)
                       or item.frame != self.block_frame
                       or item.callable_symbol != call.enclosing_callable
                       for item in self.guard_decisions):
            raise ValueError("role guard decisions belong to this exact block call")
        none_context = selected is None or (
            selected.kind == "constant" and selected.const_value is None)
        if not self.alternatives and not none_context \
                and not self.context_lineage_spans:
            raise ValueError("non-None context retains exact lineage provenance")
        if self.alternatives:
            for item in self.alternatives:
                if item.kind == "self" and item.roots \
                        and item.roots != self.primary_roots:
                    raise ValueError("self alternative uses the exact primary source")
                if item.kind == "context_slot" \
                        and item.roots == self.primary_roots:
                    raise ValueError("context alternative is distinct from primary")
            context_roots = tuple(sorted({
                root for item in self.alternatives
                if item.kind == "context_slot" for root in item.roots}))
            alternative_kinds = {item.kind for item in self.alternatives}
            expected_kind = (
                next(iter(alternative_kinds))
                if len(alternative_kinds) == 1 else "conditional")
            if self.kind != expected_kind \
                    or self.context_roots != context_roots \
                    or self.context_lineage_spans != tuple(dict.fromkeys(
                        span for item in self.alternatives
                        for span in item.lineage_spans)) \
                    or self.kind == "conditional" \
                    and alternative_kinds \
                    != {"self", "context_slot"}:
                raise ValueError("branch alternatives determine the exact role state")
        elif self.kind == "conditional":
            raise ValueError("a conditional role requires both exact alternatives")
        elif self.kind == "self":
            if not none_context \
                    and self.context_roots != self.primary_roots:
                raise ValueError("self role has None or the same exact source")
        elif none_context or len(self.context_roots) != 1 \
                or self.context_roots == self.primary_roots:
            raise ValueError("context slot has one distinct exact source formal")
        required = {
            call.span, self.primary_expression.span,
            *(self.interface.spans),
            *((self.context_expression.span,)
              if self.context_expression is not None else ()),
            *((*self.context_selection.spans,)
              if self.context_selection is not None else ()),
            *(span for item in self.alternatives for span in item.spans),
            *(span for item in self.presence_decisions for span in item.spans),
            *(span for item in self.guard_decisions for span in item.spans),
            *self.primary_lineage_spans, *self.context_lineage_spans,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("invocation-role provenance closes every join")


def framework_attention_invocation_role(
        index: ProgramIndex,
        block_frame: ConstructorFrame,
        lane: FrameworkAttentionLaneEvidence,
) -> ReaderResult[FrameworkAttentionInvocationRole]:
    """Prove one addressed framework lane's input role without name semantics."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(block_frame, ConstructorFrame) \
            or not isinstance(lane, FrameworkAttentionLaneEvidence):
        raise TypeError("attention role requires index/block frame/framework lane")
    owner = block_frame.graph.root.occurrence
    if lane.block_occurrence != owner or lane.child_symbol is None:
        return _failed(owner, "lane is not source-addressed at this block")
    target = canonical_construction_target(
        index, lane.construction, lane.child_symbol,
        canonical_import=lane.canonical_import)
    if target is None:
        return _failed(owner, "lane construction has no canonical source target")
    try:
        child_frame = constructor_frame(index, target, block_frame)
    except ValueError as exc:
        return _failed(owner, str(exc))
    interface_result = default_attention_container_interface(
        index, child_frame)
    if interface_result.status != "resolved":
        detail = "; ".join(
            item.detail for item in interface_result.failures
            if item.detail) or "unknown container-interface failure"
        return _failed(
            owner, f"attention container interface is not exact: {detail}")
    interface = interface_result.require_value()
    call = lane.invocation.call
    bound, expanded = _bind_call(call, interface.forward)
    primary = bound.get(interface.primary_formal.name)
    if primary is None:
        return _failed(owner, "lane call does not bind the primary interface")
    context = bound.get(interface.context_formal.name)
    if context is None and expanded:
        return _failed(owner, "expanded arguments may override omitted context")
    selection = None
    selection_failures = ()
    selected = context
    if context is not None and context.kind == "ifexp":
        selected_result = select_constructor_conditioned_call_argument(
            index, block_frame, call, context)
        if selected_result.status != "resolved":
            selection_failures = selected_result.failures
        else:
            selection = selected_result.require_value()
            selected = selection.selected
    record = index.callable_by_symbol(call.enclosing_callable)
    if record is None:
        return _failed(owner, "lane caller callable is absent")
    norm_result = norm_preserving_invocations_in_frame(index, block_frame)
    transparent_norm_calls = tuple(
        item.call for item in norm_result.require_value().candidates
    ) if norm_result.has_value else ()
    guard_decisions = {}
    presence_decisions = {}

    def binding_guard_state(binding):
        if not binding.guard:
            return True
        remaining = []
        for step in binding.guard:
            if not _is_exact_lane_presence_step(step, lane):
                remaining.append(step)
                continue
            spans = tuple(dict.fromkeys((
                step.span, lane.invocation.call.span,
                lane.construction.span,
            )))
            presence_decisions[step] = ConstructedLanePresenceDecision(
                lane, step, spans)
        if not remaining:
            return True
        result = resolve_constructor_guard(
            index, block_frame, record.symbol,
            tuple(remaining), binding.span)
        if result.status != "resolved":
            return None
        decision = result.require_value()
        guard_decisions[(binding.guard, binding.span)] = decision
        return decision.decision

    lineage = local_lineage_at_callable(
        index, record, transparent_calls=transparent_norm_calls,
        binding_guard_state=binding_guard_state)
    primary_trace = lineage.state_carriers(primary, call.span, call.guard)
    if primary_trace.unresolved or len(primary_trace.roots) != 1:
        definitions = (
            lineage.definitions(primary.name, call.span)
            if primary.kind == "name" and primary.name else ())
        definition_debug = tuple(
            (item.span.line, tuple(step.kind for step in item.guard), value.kind)
            for item, value in definitions)
        transparent_debug = tuple(
            item.span.line for item in transparent_norm_calls)
        norm_debug = tuple(item.detail for item in norm_result.failures)
        return _failed(
            owner,
            "primary input has no single caller-formal root: "
            f"roots={tuple(sorted(primary_trace.roots))!r}, "
            f"unresolved={primary_trace.unresolved!r}, "
            f"call={call.span.line}:{call.span.col}, "
            f"definitions={definition_debug!r}, "
            f"transparent_norms={transparent_debug!r}, "
            f"norm_evidence={norm_debug!r}")
    primary_roots = tuple(sorted(primary_trace.roots))
    context_roots = ()
    context_lineage_spans = ()
    alternatives = ()
    if selection_failures:
        alternatives = _conditional_alternatives(
            lineage, context, call, primary_roots)
        if alternatives is None:
            detail = "; ".join(
                item.detail for item in selection_failures
                if item.detail) or "unknown constructor-condition failure"
            return _failed(
                owner,
                f"context branch is not constructor-decidable: {detail}")
        alternative_kinds = {item.kind for item in alternatives}
        kind = next(iter(alternative_kinds)) \
            if len(alternative_kinds) == 1 else "conditional"
        context_roots = tuple(sorted({
            root for item in alternatives if item.kind == "context_slot"
            for root in item.roots}))
        context_lineage_spans = tuple(dict.fromkeys(
            span for item in alternatives for span in item.lineage_spans))
    elif selected is None or selected.kind == "constant" \
            and selected.const_value is None:
        kind = "self"
    else:
        context_trace = lineage.trace(selected, call.span, call.guard)
        if context_trace.unresolved or len(context_trace.roots) != 1:
            return _failed(owner, "context input has no single caller-formal root")
        context_roots = tuple(sorted(context_trace.roots))
        context_lineage_spans = tuple(context_trace.spans)
        kind = ("self" if context_roots == primary_roots
                else "context_slot")
    spans = tuple(dict.fromkeys(span for span in (
        call.span, primary.span,
        *((context.span,) if context is not None else ()),
        *((*selection.spans,) if selection is not None else ()),
        *(span for item in alternatives for span in item.spans),
        *(span for item in presence_decisions.values() for span in item.spans),
        *(span for item in guard_decisions.values() for span in item.spans),
        *primary_trace.spans, *context_lineage_spans,
        *interface.spans,
    ) if isinstance(span, SourceSpan)))
    value = FrameworkAttentionInvocationRole(
        lane, block_frame, interface, primary, context, selection,
        primary_roots, context_roots, tuple(primary_trace.spans),
        context_lineage_spans, _unique_values(presence_decisions.values()),
        _unique_values(guard_decisions.values()),
        alternatives, kind, spans)
    provenance = (ReaderProvenance(
        "source", spans=spans,
        detail="exact lane call + constructor branch + caller-formal lineage"),)
    if kind == "conditional":
        return ReaderResult.incomplete(
            owner, value, failures=selection_failures,
            provenance=provenance)
    return ReaderResult.resolved(owner, value, provenance=provenance)


def _conditional_alternatives(lineage, expression, call, primary_roots):
    if expression is None or expression.kind != "ifexp" \
            or len(expression.children) != 3:
        return None
    rows = []
    for branch in (expression.children[0], expression.children[2]):
        if branch.kind == "constant" and branch.const_value is None:
            rows.append(AttentionInputRoleAlternative(
                branch, (), (), "self", (branch.span,)))
            continue
        trace = lineage.trace(branch, call.span, call.guard)
        if trace.unresolved or len(trace.roots) != 1:
            return None
        roots = tuple(sorted(trace.roots))
        kind = "self" if roots == primary_roots else "context_slot"
        spans = tuple(dict.fromkeys((branch.span, *trace.spans)))
        rows.append(AttentionInputRoleAlternative(
            branch, roots, tuple(trace.spans), kind, spans))
    return tuple(rows)


def _unique_values(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return tuple(out)


def _is_exact_lane_presence_step(step, lane):
    if not isinstance(step, GuardStep) or step.kind not in {"if", "elif"} \
            or step.test is None or step.test.kind != "compare" \
            or step.test.operator != "is not" \
            or len(step.test.children) != 2:
        return False
    field = lane.construction.target
    if lane.construction.target_kind != "field" or not field:
        return False
    left, right = step.test.children

    def self_field(expression):
        return (
            expression.kind == "attribute" and expression.name == field
            and len(expression.children) == 1
            and expression.children[0].kind == "name"
            and expression.children[0].name == "self")

    def none(expression):
        return expression.kind == "constant" \
            and expression.const_value is None

    return self_field(left) and none(right) \
        or self_field(right) and none(left)


def _bind_call(call: CallObservation, callable_record):
    params = list(callable_record.params)
    if callable_record.kind == "method":
        if not params:
            return {}, True
        params = params[1:]
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    by_name = {item.name: item for item in params
               if item.kind not in {"vararg", "kwarg"}}
    if len(call.args) > len(positional):
        return {}, True
    bound = {param.name: actual
             for param, actual in zip(positional, call.args)}
    expanded = False
    for name, actual in call.kwargs:
        if name == "**":
            expanded = True
            continue
        if name not in by_name or name in bound:
            return {}, True
        bound[name] = actual
    return bound, expanded


def _failed(owner, detail):
    return ReaderResult.failed(owner, (ReaderFailure(
        "incomplete_graph", detail),))


__all__ = [
    "AttentionInputRoleAlternative",
    "FrameworkAttentionInvocationRole",
    "ROLE_KINDS",
    "framework_attention_invocation_role",
]
