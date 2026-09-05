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
    attention_compute_positive_proof_for_symbol,
)
from .component_owner import OwnerOccurrenceId, require_resolved_component_root
from .config_guard import ExactConfigGuardResolver
from .construction_calls import (
    resolve_construction_call,
)
from .container_inventory import resolve_container_inventory
from .dispatch_selection import resolve_dispatch_candidates
from .execution_flow import (
    AddressedInvocation,
    ExecutionFlowResolution,
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
from .primitive_semantics import (
    classify_primitive_call,
    primitive_kind_for_site,
)
from .program_index import (
    CallObservation,
    CallSiteId,
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
class ExactBranchCensus:
    """The exact attention and ordinary/routed FFN calls in one decoder block.

    This is deliberately mechanism/address evidence only.  It does not infer
    residual order or norm placement.  The cell-topology reader consumes this
    one census for every norm, residual, and parallel-input-count fact.
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
            attention_compute_positive_proof_for_symbol(
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


def exact_norm_sources_at_block(
        index, root, owner, invocations, *, config_selector=None):
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
    if config_selector is None:
        return sources
    node = root.graph.node_for(owner)
    if node is None:
        return sources
    resolver = ExactConfigGuardResolver(
        index, node, config_selector,
        config_prefix=tuple(getattr(root, "config_path", ()) or ()))
    for invocation in invocations.unresolved:
        callee = invocation.call.callee
        if callee.kind != "attribute" or len(callee.children) != 1 \
                or callee.children[0].kind != "name" \
                or callee.children[0].name != "self":
            continue
        construction = resolve_construction_call(
            index, root, owner, invocation.call)
        if construction.status != "ambiguous":
            continue
        selected = tuple(
            alternative for alternative in construction.alternatives
            if resolver.enabled(
                alternative.site.guard,
                alternative.site.enclosing_callable) is True)
        if len(selected) != 1:
            continue
        primitive = primitive_kind_for_site(index, selected[0].site)
        if primitive is None or primitive[0] not in _NORMS:
            continue
        alternative = selected[0]
        sources[invocation.call_site] = (
            alternative.occurrence,
            primitive[0],
            tuple(dict.fromkeys(
                span for span in (
                    invocation.call.span,
                    alternative.site.span,
                    *resolver.spans,
                    *primitive[1],
                ) if isinstance(span, SourceSpan))),
        )
    return sources


__all__ = [
    "ExactBranchInvocation",
    "ExactBranchCensus",
    "exact_branch_census_at_block",
    "exact_norm_sources_at_block",
]
