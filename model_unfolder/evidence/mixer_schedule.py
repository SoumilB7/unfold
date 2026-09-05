"""Exact per-layer placement of already-proven decoder mixer mechanisms.

This boundary composes, but never replaces, the existing authorities:

* U3 proves the exact repeated decoder-block occurrence and construction index;
* U8-A proves which exact child construction is selected at layer ``i``;
* U3 execution evidence proves that exact child is invoked by the block forward;
* U6 proves the child's mechanism (ordinary softmax attention or gated delta).

Config values are therefore selector operands only.  A token, field spelling,
class name, or selected construction without a matching invocation and U6
mechanism proof cannot author a mixer kind.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention import (
    GatedDeltaGeometryBinding,
    exact_config_path_for_expression,
    gated_delta_geometry_at_occurrence,
)
from .attention_child import (
    AttentionChildEvidence,
    attention_child_positive_census,
)
from .component_owner import OwnerOccurrenceId
from .config_guard import ExactConfigGuardResolver
from .construction_arguments import (
    ConstructionArgumentBinding,
    bind_construction_site,
)
from .container_inventory import resolve_container_inventory
from .decoder_block import DecoderBlockCandidates, decoder_block_candidates_for_config
from .execution_flow import AddressedInvocation, resolve_addressed_invocations
from .framework_config import (
    framework_config_alias,
    framework_config_default_selector,
)
from .layer_selector import (
    ConfigSelectorOperand,
    LayerSelectionDecision,
    LayerSelectorResolution,
    resolve_layer_selector,
)
from .models import SourceBundle
from .program_index import (
    ComprehensionObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class BlockLayerIndexTransport:
    """Exact ``range(count) -> block __init__ formal`` transport."""

    binding: ConstructionArgumentBinding
    comprehension: ComprehensionObservation
    count_expression: ExprNode
    count_config_path: tuple[str, ...]
    count_source_kind: str
    layer_count: int
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ConstructionArgumentBinding):
            raise TypeError("block index transport carries an exact binding")
        if not isinstance(self.comprehension, ComprehensionObservation) \
                or len(self.comprehension.clauses) != 1:
            raise TypeError("block index transport carries one comprehension")
        clause = self.comprehension.clauses[0]
        if clause.async_flag or clause.filters or clause.target.kind != "name" \
                or self.binding.actual.kind != "name" \
                or self.binding.actual.name != clause.target.name:
            raise ValueError("block actual is the exact comprehension index")
        if self.count_expression != clause.iterable:
            raise ValueError("count expression is the exact comprehension iterable")
        if not self.count_config_path or any(
                not isinstance(part, str) or not part
                for part in self.count_config_path):
            raise TypeError("layer count carries an exact config path")
        if self.count_source_kind not in {"config_declared", "class_default"}:
            raise ValueError("layer count carries typed config provenance")
        if isinstance(self.layer_count, bool) \
                or not isinstance(self.layer_count, int) \
                or self.layer_count <= 0:
            raise ValueError("layer count is a positive integer")
        required = {
            self.binding.site.span,
            self.binding.actual.span,
            self.comprehension.span,
            self.count_expression.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("block index transport retains decisive spans")


@dataclass(frozen=True)
class MixerCandidateProof:
    """One exact construction/invocation carrying one U6 mechanism kind."""

    kind: str                         # ordinary_attention | gated_delta
    occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    target: str
    mechanism: AttentionChildEvidence | GatedDeltaGeometryBinding
    selector: LayerSelectorResolution
    invocation: AddressedInvocation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"ordinary_attention", "gated_delta"}:
            raise ValueError("mixer candidate kind is closed")
        if not isinstance(self.occurrence, OwnerOccurrenceId) \
                or not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("mixer candidate is exact-owner qualified")
        if not self.target or not isinstance(self.selector, LayerSelectorResolution) \
                or not isinstance(self.invocation, AddressedInvocation):
            raise TypeError("mixer candidate retains selector and invocation")
        if self.kind == "ordinary_attention":
            if not isinstance(self.mechanism, AttentionChildEvidence) \
                    or self.mechanism.child_occurrence != self.occurrence \
                    or self.mechanism.invocation != self.invocation:
                raise ValueError(
                    "ordinary placement carries its exact attention computation")
            mechanism_spans = self.mechanism.compute.spans
        else:
            if not isinstance(self.mechanism, GatedDeltaGeometryBinding) \
                    or self.mechanism.mixer_occurrence != self.occurrence:
                raise ValueError(
                    "gated-delta placement carries its exact mechanism geometry")
            mechanism_spans = self.mechanism.spans
        if self.selector.owner != self.invocation.caller_occurrence:
            # The selector is owned by the block, while the candidate occurrence
            # is the selected child.  No third owner is lawful.
            raise ValueError("mixer selector belongs to its exact block owner")
        if self.invocation.callee_owner_occurrence != self.occurrence:
            raise ValueError("mixer invocation addresses the exact candidate")
        if self.selector.target != self.target:
            raise ValueError("mixer selector target is the exact child field")
        required = {self.invocation.call.span, *mechanism_spans}
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("mixer candidate retains exact provenance")


@dataclass(frozen=True)
class MixerLayerDecision:
    """One complete construction-and-invocation decision for layer ``i``."""

    layer_index: int
    state: str                       # ordinary_attention | gated_delta | unresolved
    occurrence: OwnerOccurrenceId | None = None
    construction: LayerSelectionDecision | None = None
    invocation: AddressedInvocation | None = None
    operands: tuple[ConfigSelectorOperand, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, bool) \
                or not isinstance(self.layer_index, int) \
                or self.layer_index < 0:
            raise ValueError("mixer decision has a non-negative layer index")
        if self.state not in {"ordinary_attention", "gated_delta", "unresolved"}:
            raise ValueError("mixer decision state is closed")
        if any(not isinstance(item, ConfigSelectorOperand)
               for item in self.operands):
            raise TypeError("mixer operands are typed selector operands")
        if len(set(self.operands)) != len(self.operands):
            raise ValueError("mixer operands are occurrence-unique")
        if self.state == "unresolved":
            if self.occurrence is not None or self.construction is not None \
                    or self.invocation is not None or not self.reason:
                raise ValueError("an unresolved mixer carries reason only")
        elif not isinstance(self.occurrence, OwnerOccurrenceId) \
                or not isinstance(self.construction, LayerSelectionDecision) \
                or self.construction.state != "selected" \
                or not isinstance(self.invocation, AddressedInvocation) \
                or self.invocation.callee_owner_occurrence != self.occurrence \
                or self.reason:
            raise ValueError("a resolved mixer joins one construction and invocation")


@dataclass(frozen=True)
class DecoderMixerSchedule:
    """Complete exact per-layer mixer placement for one decoder block."""

    block_candidates: DecoderBlockCandidates
    block_occurrence: OwnerOccurrenceId
    transport: BlockLayerIndexTransport
    candidates: tuple[MixerCandidateProof, ...]
    decisions: tuple[MixerLayerDecision, ...]
    config_dependencies: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_candidates, DecoderBlockCandidates) \
                or not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or self.block_occurrence not in self.block_candidates.occurrences:
            raise ValueError("mixer schedule carries its exact decoder block")
        if not isinstance(self.transport, BlockLayerIndexTransport) \
                or self.transport.binding.child_occurrence != self.block_occurrence:
            raise ValueError("mixer schedule carries the block's index transport")
        if not self.candidates or any(
                not isinstance(item, MixerCandidateProof)
                or item.invocation.caller_occurrence != self.block_occurrence
                for item in self.candidates):
            raise ValueError("mixer schedule carries exact block-local candidates")
        identities = tuple(
            (item.kind, item.occurrence) for item in self.candidates)
        if len(set(identities)) != len(identities):
            raise ValueError("mixer candidates are mechanism-occurrence unique")
        expected = tuple(range(self.transport.layer_count))
        if tuple(item.layer_index for item in self.decisions) != expected \
                or any(item.state == "unresolved" for item in self.decisions):
            raise ValueError("a resolved mixer schedule covers every layer")
        candidate_by_identity = {
            (item.kind, item.occurrence): item for item in self.candidates}
        for decision in self.decisions:
            candidate = candidate_by_identity.get(
                (decision.state, decision.occurrence))
            if candidate is None:
                raise ValueError(
                    "every layer selects one carried mechanism candidate")
            if decision.invocation != candidate.invocation \
                    or decision.construction \
                    != candidate.selector.decisions[decision.layer_index]:
                raise ValueError(
                    "a layer decision round-trips through its candidate proof")
        paths = tuple(path for path, _kind in self.config_dependencies)
        if len(paths) != len(set(paths)) or any(
                kind not in {"config_declared", "class_default"}
                for _path, kind in self.config_dependencies):
            raise ValueError("mixer dependencies are exact, typed and unique")
        required_paths = {
            self.transport.count_config_path,
            *(operand.path for item in self.decisions
              for operand in item.operands),
        }
        if not required_paths <= set(paths):
            raise ValueError(
                "mixer dependencies include every decisive selector operand")
        required = {
            *self.transport.spans,
            *(span for item in self.candidates for span in item.spans),
        }
        if not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("mixer schedule provenance is closed")


def decoder_mixer_schedule_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[DecoderMixerSchedule]:
    """Resolve exact ordinary-attention/gated-delta placement at every layer."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("mixer schedule requires ProgramIndex and SourceBundle")
    candidates_result = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if candidates_result.status != "resolved":
        return candidates_result
    candidates = candidates_result.value
    if len(candidates.occurrences) != 1:
        return ReaderResult.failed(candidates.stage_occurrence, (ReaderFailure(
            "incomplete_graph",
            "mixer schedule requires one exact repeated-block occurrence"),),
            provenance=candidates_result.provenance)
    block_occurrence = candidates.occurrences[0]
    root = candidates.component_root
    block_node = root.graph.node_for(block_occurrence)
    if block_node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "decoder block is absent from its D0 graph"),))
    # One authority must see the same exact source-bound config defaults in
    # direct parser calls and when another schedule composes this reader.
    # Otherwise Qwen3's Q/K schedule can resolve its mixer while the parser's
    # separate mixer call stays unknown, leaving a proven operation trapped in
    # an opaque drill.
    effective_selector = config_selector
    stage_alias = framework_config_alias(
        index, root, candidates.stage_occurrence)
    if stage_alias.status == "resolved" and callable(effective_selector):
        effective_selector = framework_config_default_selector(
            index, stage_alias.value, effective_selector,
            config_prefix=config_path)
    transport = block_layer_index_transport(
        index, candidates, block_occurrence, effective_selector)
    if isinstance(transport, ReaderFailure):
        return ReaderResult.failed(
            block_occurrence, (transport,), provenance=candidates_result.provenance)

    inventory = resolve_container_inventory(index, root, block_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, block_occurrence, inventory)
    if invocations.status != "resolved":
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "block invocation census is unavailable"),))

    # Mechanism classification is occurrence-local.  A block-global mechanism
    # union would become ambiguous when the same implementation is constructed
    # twice and would let one sibling certify another.  The positive attention
    # census and gated-delta geometry boundary each certify their exact child.
    mechanism_results = []
    ordinary = attention_child_positive_census(
        index, root, block_occurrence)
    if ordinary.status == "resolved" and ordinary.value is not None:
        ordinary_candidates = ordinary.value.candidates
        ordinary_provenance = ordinary.provenance
        if len(ordinary_candidates) > 1:
            # An additive self+cross block legitimately contains two proven
            # attention computations.  ``mixer_schedule`` owns the primary
            # self-attention lane; cross-attention has its own independent
            # schedule/fact.  Reuse that boundary's exact invocation/dataflow
            # discriminator instead of selecting by field name, class name,
            # source order, or the mere presence of two children.
            from .cross_attention_schedule import \
                decoder_cross_attention_all_layers_for_path
            dual = decoder_cross_attention_all_layers_for_path(
                index, bundle, config_path,
                allow_root_stage=allow_root_stage)
            if dual.status == "resolved" and dual.value is not None \
                    and dual.value.block_occurrence == block_occurrence \
                    and dual.value.self_evidence in ordinary_candidates:
                ordinary_candidates = (dual.value.self_evidence,)
                ordinary_provenance = tuple(dict.fromkeys((
                    *ordinary.provenance, *dual.provenance)))
        mechanism_results.extend(
            ("ordinary_attention", item.child_occurrence, item,
             ordinary_provenance)
            for item in ordinary_candidates)
    for child in block_node.children:
        gated = gated_delta_geometry_at_occurrence(
            index, root, block_occurrence, child.occurrence)
        if gated.status == "resolved" and gated.value is not None:
            mechanism_results.append((
                "gated_delta", child.occurrence, gated.value,
                gated.provenance))
    proofs = []
    for kind, occurrence, mechanism, provenance in mechanism_results:
        if occurrence.sites[:-1] != block_occurrence.sites:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "out_of_owner", f"{kind} is not an immediate block child"),))
        node = root.graph.node_for(occurrence)
        if node is None or not node.via_field or node.via_site is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph", f"{kind} child address is incomplete"),))
        addressed = tuple(item for item in invocations.addressed
                          if item.callee_owner_occurrence == occurrence)
        if len(addressed) != 1:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                f"{kind} has {len(addressed)} exact block invocations"),))
        selector_callable = SymbolId(
            block_node.symbol.source, f"{block_node.symbol.qualified_name}.__init__")
        selector = resolve_layer_selector(
            index, root, block_occurrence, selector_callable, node.via_field,
            tuple(range(transport.layer_count)), transport.binding.formal.name,
            config_selector=effective_selector, config_prefix=config_path)
        if selector.status == "failed":
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                f"{kind} construction selector failed: {selector.failure_kind}"),))
        result_spans = tuple(
            span for item in provenance for span in item.spans)
        spans = tuple(dict.fromkeys((
            *result_spans,
            *addressed[0].provenance_spans,
            *(site.span for site in selector.candidates),
        )))
        proofs.append(MixerCandidateProof(
            kind, occurrence, node.symbol, node.via_field,
            mechanism, selector, addressed[0], spans))
    if not proofs:
        return ReaderResult.failed(
            block_occurrence, failures=(ReaderFailure(
                "unsupported_syntax",
                "no exact U6 mixer mechanism is available at this block"),),
            provenance=candidates_result.provenance)

    decisions = []
    dependency_kinds = {
        transport.count_config_path: transport.count_source_kind,
    }
    for layer_index in range(transport.layer_count):
        live = []
        uncertain = False
        operands = []
        for proof in proofs:
            decision = proof.selector.decisions[layer_index]
            for operand in decision.operands:
                if operand not in operands:
                    operands.append(operand)
                previous = dependency_kinds.get(operand.path)
                if previous is not None and previous != operand.source_kind:
                    uncertain = True
                dependency_kinds[operand.path] = operand.source_kind
            if decision.state in {"ambiguous", "unresolved"}:
                uncertain = True
                continue
            if decision.state != "selected":
                continue
            selected = decision.selected_candidates[0]
            if selected.site_id != proof.occurrence.sites[-1] \
                    or selected.candidate.symbol != proof.owner_symbol:
                uncertain = True
                continue
            guard = ExactConfigGuardResolver(
                index, block_node, effective_selector,
                config_prefix=config_path,
                parameter_values={transport.binding.formal.name: layer_index})
            invoked = guard.enabled(
                proof.invocation.guard, proof.invocation.call.enclosing_callable)
            for path, source_kind in guard.source_kinds:
                previous = dependency_kinds.get(path)
                if previous is not None and previous != source_kind:
                    uncertain = True
                dependency_kinds[path] = source_kind
            if invoked is True and guard.complete:
                live.append((proof, decision))
            elif invoked is None or not guard.complete:
                uncertain = True
        if uncertain or len(live) != 1:
            decisions.append(MixerLayerDecision(
                layer_index, "unresolved",
                operands=tuple(operands),
                reason=("construction/invocation selection is incomplete"
                        if uncertain else
                        f"expected one live mechanism, found {len(live)}")))
            continue
        proof, construction = live[0]
        decisions.append(MixerLayerDecision(
            layer_index, proof.kind, proof.occurrence,
            construction, proof.invocation, tuple(operands)))
    if any(item.state == "unresolved" for item in decisions):
        return ReaderResult.failed(
            block_occurrence, failures=(ReaderFailure(
                "incomplete_graph",
                "not every layer selects and invokes one proven mixer"),),
            provenance=candidates_result.provenance)
    spans = tuple(dict.fromkeys((
        *transport.spans,
        *(span for proof in proofs for span in proof.spans),
    )))
    value = DecoderMixerSchedule(
        candidates, block_occurrence, transport, tuple(proofs),
        tuple(decisions), tuple(dependency_kinds.items()), spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(*candidates_result.provenance, ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=tuple(dependency_kinds),
            detail=("exact construction index, candidate mechanism and block "
                    "invocation agree at every layer"))))


def block_layer_index_transport(index, candidates, block_occurrence, selector):
    """Bind the repeated-model-stage index to one decoder-block formal.

    This is shared address infrastructure for every per-layer schedule.  It
    proves only ``range(exact_count) -> construction actual -> block formal``;
    it assigns no attention/FFN role and interprets no schedule operand.
    """
    proofs = tuple(
        proof for proof in candidates.repeated_child.proofs
        if proof.child_occurrence == block_occurrence)
    if len(proofs) != 1:
        return ReaderFailure(
            "incomplete_graph", "one exact repeated-block proof is required")
    proof = proofs[0]
    site = proof.template.element_template
    bindings = bind_construction_site(
        index, candidates.component_root, proof.model_stage, site)
    if bindings.status not in {"resolved", "partial"}:
        return ReaderFailure(
            "incomplete_graph", "block constructor arguments are not exact")
    comprehensions = tuple(
        item for item in index.comprehensions_in(site.enclosing_callable)
        if item.span is not None and len(item.outputs) == 1
        and item.outputs[0].span == site.constructor.span
        and len(item.clauses) == 1)
    if len(comprehensions) != 1:
        return ReaderFailure(
            "incomplete_graph", "block construction has no unique comprehension")
    comprehension = comprehensions[0]
    clause = comprehension.clauses[0]
    if clause.async_flag or clause.filters or clause.target.kind != "name":
        return ReaderFailure(
            "unsupported_syntax", "block comprehension is not a direct range")
    index_bindings = tuple(
        item for item in bindings.bindings
        if item.actual.kind == "name" and item.actual.name == clause.target.name)
    if len(index_bindings) != 1:
        return ReaderFailure(
            "incomplete_graph", "comprehension index has no exact block formal")
    count = clause.iterable
    if count.kind != "call" or len(count.children) != 2 \
            or count.children[0].kind != "name" \
            or count.children[0].name != "range" \
            or _name_shadowed(index, site.enclosing_callable, "range"):
        return ReaderFailure(
            "unsupported_syntax", "layer comprehension is not exact builtin range")
    count_expression = count.children[1]
    stage_node = candidates.component_root.graph.node_for(proof.model_stage)
    path = exact_config_path_for_expression(
        index, stage_node, count_expression, config_prefix=candidates.config_path)
    selected = _select_config_value(selector, path)
    if selected is None or isinstance(selected[0], bool) \
            or not isinstance(selected[0], int) or selected[0] <= 0:
        return ReaderFailure(
            "incomplete_graph", "layer count is not exact positive config evidence")
    spans = tuple(dict.fromkeys((
        *index_bindings[0].spans, comprehension.span,
        clause.target.span, count.span, count_expression.span,
    )))
    return BlockLayerIndexTransport(
        index_bindings[0], comprehension, count, path,
        selected[1], selected[0], spans)


def _select_config_value(selector, path):
    if selector is None or path is None:
        return None
    selected = selector(path)
    from .framework_config import FrameworkConfigDefaultValue
    if isinstance(selected, FrameworkConfigDefaultValue):
        return selected.value, "class_default"
    source_kind = "config_declared"
    if isinstance(selected, tuple) and len(selected) in {2, 3} \
            and isinstance(selected[0], bool):
        present, value = selected[:2]
        if len(selected) == 3:
            source_kind = selected[2]
    else:
        present, value = selected is not None, selected
    if not present or source_kind not in {"config_declared", "class_default"}:
        return None
    return value, source_kind


def _name_shadowed(index, callable_symbol, name):
    return any(item.name == name and item.context in {
        "parameter", "store", "del"}
        for item in index.identifiers_in(callable_symbol)) or any(
        item.name == name
        for item in index.module_bindings_in(callable_symbol.source))


__all__ = [
    "BlockLayerIndexTransport",
    "DecoderMixerSchedule",
    "MixerCandidateProof",
    "MixerLayerDecision",
    "block_layer_index_transport",
    "decoder_mixer_schedule_for_path",
]
