"""Exact per-layer replacement cross-attention schedule.

This reader handles heterogeneous decoder stacks such as a source loop that
constructs one of two block implementations at each layer.  It joins four
independent proofs and no model/family vocabulary:

* the exact decoder-stage occurrence;
* the exact container invoked by that stage's ``forward``;
* the exact constructor selector for every requested layer index; and
* local Q/K/V lineage inside the selected block's positively-proven attention
  child.

``replacement_cross`` means Q descends from one callable formal while K and V
descend from the same other formal.  ``self`` means Q/K/V all descend from the
same formal.  Field, class, parameter, and config spellings never decide the
mechanism.  Rival sites, unresolved lineage, missing layers, and opaque
container execution remain typed failure rather than falling back to a config
list.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import (
    AttentionChildEvidence,
    attention_child_positive_census,
)
from .attention_storage import producer_sources_reaching_expressions
from .affine import construction_is_affine, site_is_affine
from .component_owner import OwnerOccurrenceId
from .construction_calls import resolve_construction_call
from .container_inventory import ContainerAddress, resolve_container_inventory
from .decoder_stage import DecoderStagePath, decoder_stage_for_config
from .execution_flow import UnresolvedInvocation, resolve_addressed_invocations
from .layer_selector import (
    ConfigSelectorOperand,
    LayerSelectorResolution,
    SelectedConstructionCandidate,
    resolve_layer_selector,
)
from .models import SourceBundle
from .program_index import ProgramIndex, SourceSpan
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_KINDS = frozenset({"self", "replacement_cross"})


@dataclass(frozen=True)
class AttentionInputLineage:
    """Q/K/V formal-source proof for one exact selected block occurrence."""

    kind: str
    block_occurrence: OwnerOccurrenceId
    attention: AttentionChildEvidence
    q_formal: str
    k_formal: str
    v_formal: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown attention input lineage {self.kind!r}")
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("attention input lineage names an exact block")
        if not isinstance(self.attention, AttentionChildEvidence) \
                or self.attention.block_occurrence != self.block_occurrence:
            raise ValueError("the attention proof belongs to the exact block")
        if any(not isinstance(item, str) or not item
               for item in (self.q_formal, self.k_formal, self.v_formal)):
            raise TypeError("Q/K/V retain exact callable-formal identities")
        if self.kind == "self" and len({
                self.q_formal, self.k_formal, self.v_formal}) != 1:
            raise ValueError("self attention has one exact Q/K/V source formal")
        if self.kind == "replacement_cross" and not (
                self.k_formal == self.v_formal
                and self.q_formal != self.k_formal):
            raise ValueError("replacement cross attention has Q != shared K/V source")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("attention lineage retains exact source spans")


@dataclass(frozen=True)
class ReplacementCrossAttentionSchedule:
    """Complete exact self/replacement-cross decision for every layer."""

    stage: DecoderStagePath
    container: ContainerAddress
    invocation: UnresolvedInvocation
    selector: LayerSelectorResolution
    layers: tuple[str, ...]
    lineages: tuple[AttentionInputLineage, ...]
    operands: tuple[ConfigSelectorOperand, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, DecoderStagePath):
            raise TypeError("a cross-attention schedule carries its decoder stage")
        if not isinstance(self.container, ContainerAddress) \
                or self.container.owner_occurrence != self.stage.stage_occurrence:
            raise ValueError("the schedule container belongs to the exact stage")
        if not isinstance(self.invocation, UnresolvedInvocation) \
                or self.invocation.caller_occurrence != self.stage.stage_occurrence:
            raise ValueError("the schedule carries the exact heterogeneous invocation")
        if self.invocation.reason != \
                "heterogeneous_or_unresolved_container_elements" \
                or tuple(self.invocation.evidence) != self.container.element_sites:
            raise ValueError("the invocation cites this exact heterogeneous container")
        if not isinstance(self.selector, LayerSelectorResolution) \
                or self.selector.status != "resolved" \
                or self.selector.owner != self.stage.stage_occurrence \
                or self.selector.target != self.container.field:
            raise ValueError("the exact selector completely resolves this container")
        if not self.layers or any(item not in _KINDS for item in self.layers):
            raise ValueError("the schedule contains only proven attention kinds")
        if set(self.layers) != _KINDS:
            raise ValueError(
                "a replacement-cross schedule proves both self and cross layers")
        if tuple(item.layer_index for item in self.selector.decisions) != \
                tuple(range(len(self.layers))):
            raise ValueError("the selector proves every schedule position exactly once")
        if len(self.lineages) != len(self.layers) \
                or any(lineage.kind != kind
                       for lineage, kind in zip(self.lineages, self.layers)):
            raise ValueError("each layer retains its exact matching lineage proof")
        if tuple(dict.fromkeys(
                operand for decision in self.selector.decisions
                for operand in decision.operands)) != self.operands:
            raise ValueError("the operand census is derived from the exact selector")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("the schedule retains exact source provenance")


def decoder_replacement_cross_attention_schedule_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    layer_count: int,
    *,
    allow_root_stage: bool,
    config_selector,
) -> ReaderResult[ReplacementCrossAttentionSchedule]:
    """Prove an exact heterogeneous self/replacement-cross layer schedule."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("replacement cross-attention needs ProgramIndex + SourceBundle")
    if isinstance(layer_count, bool) or not isinstance(layer_count, int) \
            or layer_count <= 0:
        raise ValueError("layer_count is a positive integer")
    stage_result = decoder_stage_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if stage_result.status != "resolved":
        return stage_result
    stage = stage_result.value
    root = stage.component_root
    owner = stage.stage_occurrence

    inventory = resolve_container_inventory(index, root, owner)
    if inventory.status != "resolved":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            f"decoder-stage container inventory is {inventory.status}: "
            f"{inventory.failure_kind}: {inventory.failure_detail}"),),
            provenance=stage_result.provenance)
    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    if invocations.status != "resolved":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            f"decoder-stage invocation census is {invocations.status}: "
            f"{invocations.failure_kind}: {invocations.failure_detail}"),),
            provenance=stage_result.provenance)
    matches = []
    for invocation in invocations.unresolved:
        if invocation.reason != \
                "heterogeneous_or_unresolved_container_elements":
            continue
        for container in inventory.containers:
            if tuple(invocation.evidence) == container.element_sites:
                matches.append((invocation, container))
    if len(matches) != 1:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the stage has no unique exact invoked heterogeneous container"),),
            provenance=stage_result.provenance)
    invocation, container = matches[0]

    selector = resolve_layer_selector(
        index, root, owner, container.record.enclosing_callable,
        container.field, tuple(range(layer_count)),
        _selector_index_name(index, container),
        config_selector=config_selector, config_prefix=config_path)
    if selector.status != "resolved":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            f"the heterogeneous block selector is {selector.status}: "
            f"{selector.failure_kind}: {selector.failure_detail}"),),
            provenance=stage_result.provenance)

    lineages = []
    for decision in selector.decisions:
        if decision.state != "selected" \
                or len(decision.selected_candidates) != 1:
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                f"layer {decision.layer_index} has no unique selected block"),),
                provenance=stage_result.provenance)
        selected = decision.selected_candidates[0]
        occurrence = _selected_occurrence(root.graph, owner, selected)
        if occurrence is None:
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                f"layer {decision.layer_index}'s selected block does not "
                "round-trip through the owner graph"),),
                provenance=stage_result.provenance)
        lineage = _attention_lineage(index, root, occurrence)
        if lineage.status != "resolved":
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                f"layer {decision.layer_index}'s selected block has no unique "
                "Q/K/V lineage proof"),), provenance=stage_result.provenance)
        lineages.append(lineage.value)

    if {item.kind for item in lineages} != _KINDS:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the selected stack does not prove both self and replacement-cross "
            "attention mechanisms"),), provenance=stage_result.provenance)

    operands = tuple(dict.fromkeys(
        operand for decision in selector.decisions
        for operand in decision.operands))
    spans = tuple(dict.fromkeys((
        invocation.call.span,
        container.record.span,
        *(operand.span for operand in operands),
        *(span for lineage in lineages for span in lineage.spans),
    )))
    spans = tuple(span for span in spans if isinstance(span, SourceSpan))
    evidence = ReplacementCrossAttentionSchedule(
        stage, container, invocation, selector,
        tuple(item.kind for item in lineages), tuple(lineages), operands, spans)
    config_paths = tuple(dict.fromkeys(item.path for item in operands))
    provenance = [*stage_result.provenance, ReaderProvenance(
        "source", spans=spans,
        detail="exact heterogeneous invocation, selection, and Q/K/V lineage")]
    if config_paths:
        provenance.append(ReaderProvenance(
            "code_and_config", spans=spans, config_paths=config_paths,
            detail="exact selector operands choose the per-layer block occurrences"))
    return ReaderResult.resolved(owner, evidence, provenance=tuple(provenance))


def _selector_index_name(index: ProgramIndex, container: ContainerAddress) -> str:
    """The one exact constructor-loop target guarding every element site."""
    for_spans = tuple(dict.fromkeys(
        step.span for site in container.element_sites for step in site.guard
        if step.kind == "for"))
    loops = tuple(
        loop for loop in index.loops_in(container.record.enclosing_callable)
        if loop.kind == "for" and loop.span in for_spans
        and loop.target is not None and loop.target.kind == "name"
        and loop.target.name)
    common = tuple(
        loop for loop in loops
        if all(any(step.kind == "for" and step.span == loop.span
                   for step in site.guard)
               for site in container.element_sites))
    if len(common) != 1:
        # This is deliberately not a name fallback.  A container built across
        # rival loops needs a separate, typed selector boundary.
        return "@unresolved_selector_loop"
    return common[0].target.name


def _selected_occurrence(graph, owner, selected: SelectedConstructionCandidate):
    matches = tuple(
        node.occurrence for node in graph.walk()
        if node.occurrence.sites[:-1] == owner.sites
        and node.via_site == selected.site_id
        and node.symbol == selected.candidate.symbol
        and graph.node_for(node.occurrence) is node)
    return matches[0] if len(matches) == 1 else None


def _attention_lineage(index, root, block_occurrence):
    census = attention_child_positive_census(index, root, block_occurrence)
    if census.status != "resolved" or len(census.value.candidates) != 1:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the selected block has no unique positively-proven attention child"),))
    return attention_input_lineage_for_child(
        index, root, block_occurrence, census.value.candidates[0])


def attention_input_lineage_for_child(
        index, root, block_occurrence, attention):
    """Classify one already-addressed attention child by exact Q/K/V lineage.

    This is positive mechanism evidence, not a child selector.  Callers must
    supply the exact child from the authoritative attention-child census.
    """
    if not isinstance(attention, AttentionChildEvidence) \
            or attention.block_occurrence != block_occurrence:
        raise ValueError("attention lineage consumes an exact child at this block")
    compute = attention.compute
    # ``entry_call`` may live inside a framework fallback function (for
    # Transformers' ALL_ATTENTION_FUNCTIONS dispatch).  ``input_calls`` are
    # the exact calls in the attention owner's own callable that supply Q/K/V
    # to that proven compute path.  Trace from that owner-side boundary so the
    # original hidden/external formal distinction is not erased by the
    # framework function's generic query/key/value formals.
    if len(compute.input_calls) != 1:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the attention compute path has no unique owner-side Q/K/V call"),))
    input_call = compute.input_calls[0]
    record = index.callable_by_symbol(input_call.enclosing_callable)
    if record is None or input_call.owner != compute.child_symbol:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the attention input call is not owner-bound"),))
    if len(input_call.args) >= 4 and input_call.args[0].kind == "name" \
            and input_call.args[0].name == "self":
        lane_expressions = input_call.args[1:4]
    elif len(input_call.args) >= 3:
        lane_expressions = input_call.args[:3]
    else:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the attention entry has no exact Q/K/V lanes"),))
    initial = {
        param.name: param.name
        for param in record.params if param.name != "self"
    }
    projections = {}
    for call in index.calls_in(input_call.enclosing_callable):
        if not _span_before(call.span, input_call.span) \
                or _self_field(call.callee) is None or not call.args:
            continue
        construction = resolve_construction_call(
            index, root, attention.compute_occurrence, call)
        if len(construction.alternatives) != 1:
            continue
        occurrence = construction.alternatives[0]
        if not (construction_is_affine(index, occurrence)
                or site_is_affine(index, occurrence.site)):
            continue
        if occurrence.occurrence in projections:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                "one affine occurrence is invoked more than once before attention"),))
        projections[occurrence.occurrence] = call
    if len(projections) != 3:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the exact attention owner does not invoke exactly three distinct "
            "affine producers before its proven compute call"),))

    projection_formals = {}
    for occurrence, call in projections.items():
        sources, _unpacks, _deps, uncertain = \
            producer_sources_reaching_expressions(
                index, input_call.enclosing_callable,
                ((call.span, (call.args[0],)),), {}, initial_sources=initial)
        if uncertain or len(sources) != 1:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                "an attention affine input has no unique callable-formal source"),))
        projection_formals[occurrence] = next(iter(sources))

    values = tuple(projection_formals.values())
    if len(set(values)) == 1:
        kind = "self"
        q = k = v = values[0]
    elif len(set(values)) == 2:
        counts = {
            value: values.count(value) for value in set(values)}
        singleton = tuple(value for value, count in counts.items() if count == 1)
        paired = tuple(value for value, count in counts.items() if count == 2)
        if len(singleton) != 1 or len(paired) != 1:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph", "the affine-input split is not one Q vs two K/V"),))
        singleton_occurrence = next(
            occurrence for occurrence, value in projection_formals.items()
            if value == singleton[0])
        q_sources, _unpacks, _deps, q_uncertain = \
            producer_sources_reaching_expressions(
                index, input_call.enclosing_callable,
                ((input_call.span, (lane_expressions[0],)),), projections)
        if q_uncertain or q_sources != frozenset((singleton_occurrence,)):
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                "the unique-input affine producer is not the proven Q lane"),))
        kind = "replacement_cross"
        q, k, v = singleton[0], paired[0], paired[0]
    else:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the three attention affine inputs do not prove self or Q-vs-K/V lineage"),))
    spans = tuple(dict.fromkeys((
        input_call.span, *compute.spans,
        *(step.call.span for step in attention.invocation_path),
    )))
    spans = tuple(span for span in spans if isinstance(span, SourceSpan))
    evidence = AttentionInputLineage(
        kind, block_occurrence, attention, str(q), str(k), str(v), spans)
    return ReaderResult.resolved(block_occurrence, evidence, provenance=(
        ReaderProvenance(
            "source", spans=spans,
            detail="exact local reaching definitions prove Q/K/V formal sources"),))


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    receiver = expression.children[0]
    return (expression.name if receiver.kind == "name"
            and receiver.name == "self" else None)


def _span_before(left, right):
    if not isinstance(left, SourceSpan) or not isinstance(right, SourceSpan) \
            or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) <= \
        (right.line, right.col)


__all__ = [
    "AttentionInputLineage",
    "ReplacementCrossAttentionSchedule",
    "attention_input_lineage_for_child",
    "decoder_replacement_cross_attention_schedule_for_path",
]
