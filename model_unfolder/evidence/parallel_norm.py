"""Exact input-normalization evidence for parallel decoder branches.

This reader answers one narrow question: do the exact attention and FFN
invocations in a selected decoder block consume the same normalization
construction occurrence or two distinct occurrences?

Names and field roles are never evidence.  Each norm is classified from its
exact construction/implementation, and each branch relationship is an exact
versioned def-use edge into the callee's first non-receiver positional formal.
Unknown inputs, inline FFNs, rival producers and unresolved relations abstain.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import (
    attention_child_evidence,
    attention_compute_proof_for_symbol,
)
from .component_owner import OwnerOccurrenceId, require_resolved_component_root
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_construction_call,
)
from .container_inventory import resolve_container_inventory
from .decoder_block import decoder_block_candidates_for_config
from .dispatch_selection import resolve_dispatch_candidates
from .execution_flow import (
    AddressedInvocation,
    ExecutionFlowResolution,
    HappensBeforeEdge,
    InvocationNodeId,
    InvocationResolution,
    resolve_addressed_invocations,
    resolve_execution_flow,
)
from .expert_storage import routed_expert_storage_at_block
from .ffn_mechanism import (
    EquivalentFFNMechanism,
    ffn_mechanism_at_block,
)
from .models import SourceBundle
from .primitive_semantics import classify_primitive_call
from .program_index import (
    CallObservation,
    CallSiteId,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


_NORMS = frozenset({"layernorm", "rmsnorm"})


@dataclass(frozen=True)
class ExactBranchInvocation:
    """One exact branch call, addressed directly or by complete dispatch."""

    caller_occurrence: OwnerOccurrenceId
    call: CallObservation
    node: InvocationNodeId
    candidate_symbols: tuple[SymbolId, ...]
    mechanism: str
    proof_kind: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.caller_occurrence, OwnerOccurrenceId):
            raise TypeError("a branch invocation is owner-qualified")
        if not isinstance(self.call, CallObservation) \
                or not isinstance(self.node, InvocationNodeId) \
                or self.node.call_site != CallSiteId.of(self.call):
            raise ValueError("a branch invocation retains its exact call node")
        if self.mechanism not in {
                "attention", "gated_delta_mixer",
                "ordinary_ffn", "routed_ffn"}:
            raise ValueError("a branch invocation has a closed mechanism kind")
        if self.proof_kind not in {
                "addressed_child", "dispatch_equivalent",
                "conditional_equivalent"}:
            raise ValueError("a branch invocation has a closed proof kind")
        if not self.candidate_symbols or any(
                not isinstance(item, SymbolId)
                for item in self.candidate_symbols):
            raise TypeError("a branch invocation carries exact candidate symbols")
        if len(set(self.candidate_symbols)) != len(self.candidate_symbols):
            raise ValueError("branch candidate symbols are unique")
        if self.proof_kind == "addressed_child" \
                and (self.node.kind != "addressed"
                     or len(self.candidate_symbols) != 1):
            raise ValueError("an addressed branch carries one graph child")
        if self.proof_kind == "dispatch_equivalent" \
                and self.node.kind != "observed":
            raise ValueError("a dispatch branch stays a neutral observed node")
        if self.proof_kind == "conditional_equivalent" \
                and self.node.kind != "observed":
            raise ValueError(
                "a conditional-equivalent branch stays a neutral observed node")
        if not self.spans or self.call.span not in self.spans \
                or any(not isinstance(span, SourceSpan)
                       for span in self.spans):
            raise ValueError("a branch invocation retains exact provenance")


@dataclass(frozen=True)
class BranchNormInput:
    """One exact norm occurrence feeding one exact child input."""

    branch: ExactBranchInvocation
    input_expression: ExprNode
    norm_occurrence: ConstructionOccurrenceId
    edge: HappensBeforeEdge
    primitive: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.branch, ExactBranchInvocation):
            raise TypeError("a branch input carries an exact branch invocation")
        if not isinstance(self.input_expression, ExprNode) \
                or self.input_expression.span is None:
            raise TypeError("a branch input carries an exact input expression")
        if not isinstance(self.norm_occurrence, ConstructionOccurrenceId):
            raise TypeError("a branch input carries an exact norm occurrence")
        if not isinstance(self.edge, HappensBeforeEdge) \
                or self.edge.target != self.branch.node \
                or self.edge.proof_kind != "versioned_def_use":
            raise ValueError("the norm has an exact local def-use edge to the branch")
        if self.edge.supporting_spans[-1] != self.input_expression.span:
            raise ValueError("the edge terminates at the exact branch input expression")
        if self.primitive not in _NORMS:
            raise ValueError("the input primitive is a proven normalization")
        required = {
            *self.branch.spans,
            self.input_expression.span,
            self.norm_occurrence.site.span,
            *self.edge.supporting_spans,
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("branch-normalization provenance is closed")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("branch-normalization spans are exact SourceSpan values")


@dataclass(frozen=True)
class ParallelNormEvidence:
    """Exact attention/FFN normalization occurrence comparison."""

    block_occurrence: OwnerOccurrenceId
    attention: BranchNormInput
    ffn_inputs: tuple[BranchNormInput, ...]
    norm_count: int
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("parallel-norm evidence names its exact decoder block")
        if not isinstance(self.attention, BranchNormInput) \
                or not self.ffn_inputs \
                or any(not isinstance(item, BranchNormInput)
                       for item in self.ffn_inputs):
            raise TypeError("parallel-norm evidence carries attention and FFN inputs")
        inputs = (self.attention, *self.ffn_inputs)
        if any(item.branch.caller_occurrence != self.block_occurrence
               or item.norm_occurrence.parent != self.block_occurrence
               for item in inputs):
            raise ValueError("all branch inputs and norms belong to the exact block")
        if len({item.branch.node.call_site for item in self.ffn_inputs}) \
                != len(self.ffn_inputs):
            raise ValueError("FFN branch input sites are unique")
        expected = len({item.norm_occurrence for item in inputs})
        if self.norm_count != expected or self.norm_count not in {1, 2}:
            raise ValueError("norm_count is the exact one/two-occurrence union")
        required = {
            span for item in inputs for span in item.spans
        }
        if not required or not required <= set(self.spans):
            raise ValueError("parallel-norm evidence retains every branch proof")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("parallel-norm spans are exact SourceSpan values")


@dataclass(frozen=True)
class ExactBranchCensus:
    """The exact attention and ordinary/routed FFN calls in one decoder block.

    This is deliberately mechanism/address evidence only.  It does not infer
    residual order or norm placement.  Cell-topology and parallel-norm readers
    consume the same census so those two facts cannot select different calls.
    """

    block_occurrence: OwnerOccurrenceId
    attention: ExactBranchInvocation
    ffn: tuple[ExactBranchInvocation, ...]
    invocations: InvocationResolution
    flow: ExecutionFlowResolution
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("an exact branch census names its decoder block")
        if not isinstance(self.attention, ExactBranchInvocation) \
                or not self.ffn \
                or any(not isinstance(item, ExactBranchInvocation)
                       for item in self.ffn):
            raise TypeError("the census carries exact attention and FFN calls")
        if self.attention.mechanism != "attention" \
                or any(item.mechanism not in {"ordinary_ffn", "routed_ffn"}
                       for item in self.ffn):
            raise ValueError("the census has one attention and exact FFN calls")
        if not isinstance(self.invocations, InvocationResolution) \
                or self.invocations.status != "resolved":
            raise TypeError("the census carries one resolved invocation census")
        if not isinstance(self.flow, ExecutionFlowResolution) \
                or self.flow.status != "partial":
            raise TypeError("the census carries one open execution-flow graph")
        if self.invocations.owner_occurrence != self.block_occurrence \
                or self.flow.owner_occurrence != self.block_occurrence:
            raise ValueError("the census substrates belong to the exact block")
        if self.invocations.callable_symbol != self.flow.callable_symbol:
            raise ValueError("invocation and flow substrates share one callable")
        branches = (self.attention, *self.ffn)
        if any(item.caller_occurrence != self.block_occurrence
               for item in branches):
            raise ValueError("every branch is invoked by the exact decoder block")
        if len({item.node.call_site for item in self.ffn}) != len(self.ffn):
            raise ValueError("FFN branch call sites are unique")
        sites = {item.node.call_site for item in branches}
        if len(sites) != len(branches):
            raise ValueError("attention and FFN branch sites are disjoint")
        if not sites <= set(self.invocations.call_sites) \
                or any(item.node not in self.flow.nodes for item in branches):
            raise ValueError(
                "every branch round-trips through the shared invocation/flow graph")
        required = {
            span for item in branches for span in item.spans
        }
        if not required or not required <= set(self.spans):
            raise ValueError("the census retains every branch proof span")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("branch-census spans are exact SourceSpan values")


def decoder_parallel_norm_count_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[ParallelNormEvidence]:
    """Compare exact branch-input norms for every selected block candidate."""
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "decoder_parallel_norm_count_for_path requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError(
            "decoder_parallel_norm_count_for_path requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")

    candidates = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return candidates
    results = tuple(
        _parallel_norm_at_block(
            index, candidates.value.component_root, occurrence)
        for occurrence in candidates.value.occurrences)
    ambiguous = tuple(item for item in results if item.status == "ambiguous")
    if ambiguous:
        return ReaderResult.ambiguous(
            candidates.value.stage_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                span for item in ambiguous
                for span in item.ambiguity.sites))),
            provenance=candidates.provenance)
    if any(item.status != "resolved" for item in results):
        failures = tuple(
            failure for item in results for failure in item.failures)
        return ReaderResult.failed(
            candidates.value.stage_occurrence,
            failures or (ReaderFailure(
                "incomplete_graph",
                "not every exact block candidate proves branch-input norms"),),
            provenance=candidates.provenance)
    counts = {item.value.norm_count for item in results}
    if len(counts) != 1:
        return ReaderResult.ambiguous(
            candidates.value.stage_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                item.value.spans[0] for item in results))),
            provenance=candidates.provenance)
    value = results[0].value
    return ReaderResult.resolved(
        candidates.value.stage_occurrence, value,
        provenance=(
            *candidates.provenance,
            *(origin for item in results for origin in item.provenance),
            ReaderProvenance(
                "derived",
                detail=(
                    "every exact repeated-child candidate proves the same "
                    "one/two normalization-occurrence count")),
        ))


def _parallel_norm_at_block(index, root, block_occurrence):
    root = require_resolved_component_root(
        root, caller="_parallel_norm_at_block")
    block = root.graph.node_for(block_occurrence)
    if block is None or index.class_by_symbol(block.symbol) is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner",
            "the exact decoder block does not round-trip through the index"),))

    census = exact_branch_census_at_block(
        index, root, block_occurrence)
    if census.status != "resolved":
        return census
    ffn_branches = census.value.ffn
    invocations = census.value.invocations
    flow = census.value.flow
    attention = census.value.attention

    norms = exact_norm_sources_at_block(
        index, root, block_occurrence, invocations)
    attention_input = _branch_norm_input(
        index, attention, norms, flow)
    if isinstance(attention_input, ReaderResult):
        return attention_input
    ffn_inputs = []
    for branch in ffn_branches:
        item = _branch_norm_input(
            index, branch, norms, flow)
        if isinstance(item, ReaderResult):
            return item
        ffn_inputs.append(item)

    inputs = (attention_input, *ffn_inputs)
    count = len({item.norm_occurrence for item in inputs})
    if count not in {1, 2}:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                item.norm_occurrence.site.span for item in inputs))))
    spans = tuple(dict.fromkeys(
        span for item in inputs for span in item.spans))
    value = ParallelNormEvidence(
        block_occurrence, attention_input, tuple(ffn_inputs), count, spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "exact normalization construction occurrences feed the exact "
                "attention and FFN input formals")),))


def exact_branch_census_at_block(index, root, block_occurrence):
    """Resolve the exact block-local attention/FFN call census.

    This composes already-closed mechanism readers with the one owner-qualified
    invocation/flow graph.  It never chooses by field/class name and never
    interprets lexical order as execution order.
    """
    root = require_resolved_component_root(
        root, caller="exact_branch_census_at_block")
    block = root.graph.node_for(block_occurrence)
    if block is None or index.class_by_symbol(block.symbol) is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner",
            "the exact decoder block does not round-trip through the index"),))

    inventory = resolve_container_inventory(index, root, block_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, block_occurrence, inventory)
    flow = resolve_execution_flow(
        index, root, block_occurrence, inventory)
    if invocations.status != "resolved" or flow.status != "partial":
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the exact block invocation/dataflow graph is unavailable"),))

    ffn = ffn_mechanism_at_block(index, root, block_occurrence)
    if ffn.status == "ambiguous":
        return ReaderResult.ambiguous(
            block_occurrence, ffn.ambiguity, provenance=ffn.provenance)
    if ffn.status == "resolved":
        if isinstance(ffn.value, EquivalentFFNMechanism):
            ffn_branches = (_conditional_equivalent_branch(ffn.value),)
        elif not ffn.value.invocations:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                "an inline FFN has no addressed branch-input call"),))
        else:
            ffn_branches = tuple(
                _addressed_branch(
                    root, invocation, "ordinary_ffn")
                for invocation in ffn.value.invocations)
    else:
        routed = _routed_ffn_branch(
            index, root, block_occurrence, invocations)
        if isinstance(routed, ExactBranchInvocation):
            ffn_branches = (routed,)
        else:
            return ReaderResult.failed(
                block_occurrence,
                ffn.failures or (ReaderFailure(
                    "incomplete_graph", "the exact FFN branch is unresolved"),),
                provenance=ffn.provenance)

    attention = _attention_branch(
        index, root, block_occurrence, invocations, flow)
    if isinstance(attention, ReaderResult):
        return attention
    spans = tuple(dict.fromkeys(
        span for item in (attention, *ffn_branches)
        for span in item.spans))
    value = ExactBranchCensus(
        block_occurrence, attention, tuple(ffn_branches),
        invocations, flow, spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "exact attention and FFN mechanisms join the same "
                "owner-qualified invocation/dataflow graph")),))


def _addressed_branch(root, invocation, mechanism):
    if not isinstance(invocation, AddressedInvocation):
        raise TypeError("_addressed_branch requires an AddressedInvocation")
    child = root.graph.node_for(invocation.callee_owner_occurrence)
    if child is None:
        raise ValueError("the addressed branch child round-trips through the graph")
    spans = tuple(dict.fromkeys(
        span for span in invocation.provenance_spans
        if isinstance(span, SourceSpan)))
    return ExactBranchInvocation(
        invocation.caller_occurrence,
        invocation.call,
        InvocationNodeId(invocation.call_site, "addressed"),
        (child.symbol,),
        mechanism,
        "addressed_child",
        spans,
    )


def _routed_ffn_branch(index, root, block_occurrence, invocations):
    """Address the block call owning a positive routed-expert storage path.

    This proves only that the exact child is the routed FFN cell stage.  Router
    policy and expert internals remain separate evidence; names never vote.
    """
    storage = routed_expert_storage_at_block(
        index, root, block_occurrence)
    if storage.status != "resolved" or not storage.value.construction_path:
        return None
    first_site = storage.value.construction_path[0]
    matches = tuple(
        item for item in invocations.addressed
        if item.callee_owner_occurrence.sites
        and item.callee_owner_occurrence.sites[-1] == first_site)
    if len(matches) != 1:
        return None
    branch = _addressed_branch(root, matches[0], "routed_ffn")
    return ExactBranchInvocation(
        branch.caller_occurrence, branch.call, branch.node,
        branch.candidate_symbols, branch.mechanism, branch.proof_kind,
        tuple(dict.fromkeys((*branch.spans, *storage.value.spans))),
    )


def _conditional_equivalent_branch(value):
    calls = {
        item.conditional_entry.call for item in value.variants
    }
    if len(calls) != 1:
        raise ValueError(
            "equivalent FFN alternatives share one exact block invocation")
    call = next(iter(calls))
    symbols = tuple(dict.fromkeys(
        item.owner_symbol for item in value.variants))
    spans = tuple(dict.fromkeys(
        span for span in (
            call.span,
            *(item.conditional_entry.site.span for item in value.variants),
            *value.spans,
        ) if isinstance(span, SourceSpan)))
    return ExactBranchInvocation(
        value.block_occurrence,
        call,
        InvocationNodeId(CallSiteId.of(call), "observed"),
        symbols,
        "ordinary_ffn",
        "conditional_equivalent",
        spans,
    )


def _attention_branch(index, root, block_occurrence, invocations, flow):
    direct = attention_child_evidence(index, root, block_occurrence)
    if direct.status == "ambiguous":
        return ReaderResult.ambiguous(
            block_occurrence, direct.ambiguity,
            provenance=direct.provenance)
    if direct.status == "resolved":
        return _addressed_branch(
            root, direct.value.invocation, "attention")

    dispatch = []
    for unresolved in invocations.unresolved:
        census = resolve_dispatch_candidates(
            index, root, block_occurrence, unresolved.call)
        if census.status != "resolved":
            continue
        proofs = tuple(
            attention_compute_proof_for_symbol(
                index, candidate.candidate.symbol)
            for candidate in census.value.candidates)
        if not proofs or any(proof is None for proof in proofs):
            continue
        node = InvocationNodeId(unresolved.call_site, "observed")
        if node not in flow.nodes:
            continue
        symbols = tuple(
            candidate.candidate.symbol
            for candidate in census.value.candidates)
        spans = tuple(dict.fromkeys(
            span for span in (
                unresolved.call.span,
                census.value.site.span,
                census.value.registry.span,
                *(candidate.candidate.reference.span
                  for candidate in census.value.candidates),
                *(span for proof in proofs for span in proof.spans),
            ) if isinstance(span, SourceSpan)))
        dispatch.append(ExactBranchInvocation(
            block_occurrence,
            unresolved.call,
            node,
            symbols,
            "attention",
            "dispatch_equivalent",
            spans,
        ))
    if len(dispatch) > 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(item.call.span for item in dispatch)))
    if len(dispatch) == 1:
        return dispatch[0]
    return ReaderResult.failed(
        block_occurrence,
        direct.failures or (ReaderFailure(
            "incomplete_graph",
            "no exact addressed or candidate-equivalent attention branch"),),
        provenance=direct.provenance)


def exact_norm_sources_at_block(index, root, owner, invocations):
    """Classify every exactly-addressed block-local norm invocation.

    The mapping key is the authoritative call site; the value retains the exact
    construction occurrence, primitive family, and all source spans.  Absence
    means unproved, never non-normalization.
    """
    sources = {}
    for invocation in (
            *invocations.addressed, *invocations.external_addressed):
        construction = resolve_construction_call(
            index, root, owner, invocation.call)
        primitive = classify_primitive_call(index, construction)
        if primitive.status != "resolved" or primitive.value not in _NORMS:
            continue
        sources[invocation.call_site] = (
            construction.selected.occurrence,
            primitive.value,
            tuple(dict.fromkeys(
                span for span in (
                    *invocation.provenance_spans,
                    construction.selected.site.span,
                    *(span for origin in primitive.provenance
                      for span in origin.spans),
                ) if isinstance(span, SourceSpan))),
        )
    return sources


def _branch_norm_input(index, branch, norms, flow):
    if not isinstance(branch, ExactBranchInvocation):
        raise TypeError("_branch_norm_input requires an ExactBranchInvocation")
    input_expression = _first_formal_actual(
        index, branch.candidate_symbols, branch.call)
    if input_expression is None or input_expression.span is None:
        return ReaderResult.failed(flow.owner_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the branch's first non-receiver positional input is unresolved"),))

    target = branch.node
    edges = tuple(
        edge for edge in (*flow.proven_edges, *flow.conditional_edges)
        if edge.target == target
        and edge.source.call_site in norms
        and edge.supporting_spans[-1] == input_expression.span)
    occurrences = {
        norms[edge.source.call_site][0] for edge in edges
    }
    if len(occurrences) > 1:
        return ReaderResult.ambiguous(
            flow.owner_occurrence,
            Ambiguity(sites=tuple(sorted(
                (item.site.span for item in occurrences),
                key=_span_key))))
    if len(occurrences) != 1 or len(edges) != 1:
        return ReaderResult.failed(flow.owner_occurrence, (ReaderFailure(
            "incomplete_graph",
            "one exact normalization occurrence does not uniquely feed "
            "the branch input"),))
    edge = edges[0]
    occurrence, primitive, norm_spans = norms[edge.source.call_site]
    spans = tuple(dict.fromkeys(
        span for span in (
            *branch.spans,
            input_expression.span,
            *norm_spans,
            *edge.supporting_spans,
        ) if isinstance(span, SourceSpan)))
    return BranchNormInput(
        branch, input_expression, occurrence, edge, primitive, spans)


def _first_formal_actual(
    index: ProgramIndex,
    candidate_symbols: tuple[SymbolId, ...],
    call: CallObservation,
) -> ExprNode | None:
    formals = []
    for child_symbol in candidate_symbols:
        forward = index.callable_by_symbol(SymbolId(
            child_symbol.source, f"{child_symbol.qualified_name}.forward"))
        if forward is None:
            return None
        positional = tuple(
            param for param in forward.params if param.kind == "positional")
        if forward.owner is not None:
            positional = positional[1:]
        if not positional:
            return None
        formals.append(positional[0].name)
    if len(set(formals)) != 1:
        return None
    if call.args:
        return call.args[0]
    matches = tuple(
        value for key, value in call.kwargs if key == formals[0])
    return matches[0] if len(matches) == 1 else None


def _span_key(span):
    return (
        span.source.component_key or "",
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


__all__ = [
    "ExactBranchInvocation",
    "ExactBranchCensus",
    "BranchNormInput",
    "ParallelNormEvidence",
    "exact_branch_census_at_block",
    "exact_norm_sources_at_block",
    "decoder_parallel_norm_count_for_path",
]
