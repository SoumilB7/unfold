"""U3-F5a — positively prove an attention-compute child of one exact block.

This reader never searches a file for an ``Attention`` spelling and never
claims that an opaque sibling is not attention.  It starts from an exact block
occurrence, follows only graph-authoritative child invocations, and recognizes
attention computation through exact framework calls or indexed implementation
math:

* ``scaled_dot_product_attention``; or
* a dot-product operation and softmax in the same exact callable.

An indexed fallback function passed into a locally invoked dispatch result is
followed only through that exact binding.  This covers the Transformers
``get_interface(..., eager_attention_forward)`` protocol without treating the
registry or fallback function's spelling as semantic evidence.

The result proves one child invocation is attention.  It is deliberately not a
closed-world census of every possible attention child in the block.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .construction_calls import resolve_import_reference
from .container_inventory import resolve_container_inventory
from .execution_flow import AddressedInvocation, resolve_addressed_invocations
from .program_index import (
    CallObservation,
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


_SDPA_PROTOCOLS = frozenset({
    "torch.nn.functional.scaled_dot_product_attention",
})
_SOFTMAX_PROTOCOLS = frozenset({
    "torch.nn.functional.softmax",
    "torch.softmax",
})
_DOT_PROTOCOLS = frozenset({
    "torch.bmm",
    "torch.einsum",
    "torch.matmul",
})
_DISPATCH_FALLBACK_PROTOCOLS = {
    # Closed Transformers framework API: the second positional argument is the
    # eager implementation returned when the selected backend is unavailable.
    "...modeling_utils.ALL_ATTENTION_FUNCTIONS.get_interface": 1,
}


@dataclass(frozen=True)
class AttentionComputeProof:
    """One exact callable that positively proves attention computation."""

    child_symbol: SymbolId
    callable_symbol: SymbolId
    protocol: str               # scaled_dot_product_attention | dot_softmax
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.child_symbol, SymbolId):
            raise TypeError("an attention proof names its exact child symbol")
        if not isinstance(self.callable_symbol, SymbolId):
            raise TypeError("an attention proof names its exact callable")
        if self.protocol not in {
                "scaled_dot_product_attention", "dot_softmax"}:
            raise ValueError(f"unknown attention-compute protocol {self.protocol!r}")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("an attention proof carries exact source spans")
        if self.callable_symbol.source != self.child_symbol.source:
            raise ValueError("the compute callable and child share one source")
        if any(span.source != self.callable_symbol.source for span in self.spans):
            raise ValueError("attention-compute spans belong to the callable source")


@dataclass(frozen=True)
class AttentionChildEvidence:
    """A graph-authoritative block invocation plus code-compute proof."""

    block_occurrence: OwnerOccurrenceId
    child_occurrence: OwnerOccurrenceId
    invocation: AddressedInvocation
    compute: AttentionComputeProof

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("attention-child evidence names its exact block")
        if not isinstance(self.child_occurrence, OwnerOccurrenceId):
            raise TypeError("attention-child evidence names its exact child")
        if not isinstance(self.invocation, AddressedInvocation):
            raise TypeError("attention-child evidence carries its invocation")
        if not isinstance(self.compute, AttentionComputeProof):
            raise TypeError("attention-child evidence carries compute proof")
        if self.invocation.caller_occurrence != self.block_occurrence:
            raise ValueError("the invocation belongs to the exact block")
        if self.invocation.callee_owner_occurrence != self.child_occurrence:
            raise ValueError("the invocation addresses the exact child")
        if self.child_occurrence.sites[:-1] != self.block_occurrence.sites:
            raise ValueError("the attention child is an immediate block child")


def attention_child_evidence(
    index: ProgramIndex,
    root: ComponentRootResolution,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[AttentionChildEvidence]:
    """Prove one exact invoked child performs attention computation.

    Several positively proven children are ambiguity.  No positive proof is an
    incomplete graph, never evidence that attention is absent.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention_child_evidence requires a ProgramIndex")
    if not isinstance(root, ComponentRootResolution) or root.status != "resolved":
        raise ValueError(
            "attention_child_evidence requires a resolved component root")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError(
            "attention_child_evidence requires an exact block occurrence")
    graph = root.graph
    block_node = graph.node_for(block_occurrence)
    if block_node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner",
            "the requested block does not round-trip through the owner graph"),))
    if index.class_by_symbol(block_node.symbol) is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the requested block symbol is absent from this ProgramIndex"),))

    inventory = resolve_container_inventory(
        index, root, block_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, block_occurrence, inventory)
    if invocations.status == "failed":
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            f"child invocation resolution failed: "
            f"{invocations.failure_kind}: {invocations.failure_detail}"),))
    if invocations.status == "absent":
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact block forward is absent"),))

    candidates: list[AttentionChildEvidence] = []
    for invocation in invocations.addressed:
        child = graph.node_for(invocation.callee_owner_occurrence)
        if child is None or index.class_by_symbol(child.symbol) is None:
            continue
        proof = _attention_compute_proof(index, child.symbol)
        if proof is not None:
            candidates.append(AttentionChildEvidence(
                block_occurrence, child.occurrence, invocation, proof))

    unique = {
        (item.child_occurrence, item.invocation.call_site): item
        for item in candidates
    }
    if len(unique) > 1:
        sites = tuple(sorted(
            (item.invocation.call.span for item in unique.values()),
            key=_span_sort_key))
        return ReaderResult.ambiguous(
            block_occurrence, Ambiguity(sites=sites))
    if not unique:
        unresolved = tuple(
            item for item in invocations.unresolved
            if _self_field(item.call.callee) is not None)
        detail = (
            "no exact invoked child has a code-proven attention-compute protocol")
        if unresolved:
            detail += "; unresolved constructed-child calls: " + ", ".join(
                sorted({item.reason for item in unresolved}))
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", detail),))

    evidence = next(iter(unique.values()))
    spans = tuple(dict.fromkeys((
        evidence.invocation.call.span,
        *evidence.invocation.provenance_spans,
        *evidence.compute.spans,
    )))
    return ReaderResult.resolved(
        block_occurrence,
        evidence,
        provenance=(ReaderProvenance(
            "source",
            spans=tuple(span for span in spans
                        if isinstance(span, SourceSpan)),
            detail=(
                "exact block invocation plus code-proven attention "
                "computation")),),
    )


def _attention_compute_proof(
    index: ProgramIndex,
    child_symbol: SymbolId,
) -> AttentionComputeProof | None:
    for callable_symbol in _reachable_compute_callables(index, child_symbol):
        calls = index.calls_in(callable_symbol)
        sdpa_spans: list[SourceSpan] = []
        softmax_spans: list[SourceSpan] = []
        dot_spans: list[SourceSpan] = []
        for call in calls:
            # A mechanism present only behind a branch does not classify the
            # child unconditionally.  Branch-equivalence is a separate proof;
            # never infer it from one familiar branch.
            if call.guard:
                continue
            target, spans = _exact_call_target(index, call)
            if target in _SDPA_PROTOCOLS:
                sdpa_spans.extend(spans)
            elif target in _SOFTMAX_PROTOCOLS:
                softmax_spans.extend(spans)
            elif target in _DOT_PROTOCOLS:
                dot_spans.extend(spans)
        for expression, guard in _callable_expressions(
                index, callable_symbol):
            if not guard:
                dot_spans.extend(_matmul_spans(expression))
        if sdpa_spans:
            return AttentionComputeProof(
                child_symbol, callable_symbol,
                "scaled_dot_product_attention",
                tuple(dict.fromkeys(sdpa_spans)))
        if softmax_spans and dot_spans:
            return AttentionComputeProof(
                child_symbol, callable_symbol, "dot_softmax",
                tuple(dict.fromkeys((*softmax_spans, *dot_spans))))
    return None


def _reachable_compute_callables(
    index: ProgramIndex,
    child_symbol: SymbolId,
) -> tuple[SymbolId, ...]:
    forward = SymbolId(
        child_symbol.source, f"{child_symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return ()
    queue = [forward]
    seen: set[SymbolId] = set()
    out: list[SymbolId] = []
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        record = index.callable_by_symbol(current)
        if record is None:
            continue
        # The child forward may fold only its own self helpers.  Free functions
        # enter through the exact dispatch-binding relationship below.
        if record.owner not in {child_symbol, None}:
            continue
        seen.add(current)
        out.append(current)
        if record.owner == child_symbol:
            for name in record.self_method_calls:
                target = SymbolId(
                    child_symbol.source,
                    f"{child_symbol.qualified_name}.{name}")
                if target not in seen:
                    queue.append(target)
            # A directly called same-module free helper is an exact call edge,
            # not a whole-file search.  Follow it without interpreting its
            # spelling.
            for call in index.calls_in(current):
                if call.callee.kind != "name" or not call.callee.name:
                    continue
                target = SymbolId(child_symbol.source, call.callee.name)
                target_record = index.callable_by_symbol(target)
                if target_record is not None and target_record.owner is None \
                        and target not in seen:
                    queue.append(target)
            queue.extend(_bound_dispatch_fallbacks(index, current))
    return tuple(out)


def _bound_dispatch_fallbacks(
    index: ProgramIndex,
    callable_symbol: SymbolId,
) -> tuple[SymbolId, ...]:
    """Exact ``fn = resolver(..., fallback); fn(...)`` free-function links."""
    unguarded_called_names = {
        call.callee.name for call in index.calls_in(callable_symbol)
        if call.callee.kind == "name" and call.callee.name and not call.guard
    }
    if not unguarded_called_names:
        return ()
    out: list[SymbolId] = []
    for binding in index.bindings_in(callable_symbol):
        if binding.guard:
            continue
        target_names = {
            target.name for target in binding.targets
            if target.kind == "name" and target.name
        }
        if not target_names.intersection(unguarded_called_names) \
                or binding.value is None or binding.value.kind != "call":
            continue
        for argument in _proven_dispatch_fallbacks(
                index, callable_symbol, binding.value):
            if argument.kind != "name" or not argument.name:
                continue
            candidate = SymbolId(callable_symbol.source, argument.name)
            record = index.callable_by_symbol(candidate)
            if record is not None and record.owner is None:
                out.append(candidate)
    return tuple(dict.fromkeys(out))


def _proven_dispatch_fallbacks(
    index: ProgramIndex,
    caller: SymbolId,
    expression: ExprNode,
) -> tuple[ExprNode, ...]:
    """Return only arguments the exact selector proves it can return.

    Passing a function as an arbitrary argument is not a dataflow proof.  A
    selector is lawful here only through a closed external API protocol or an
    indexed same-module function whose every return is exactly one parameter.
    """
    if expression.kind != "call" or not expression.children:
        return ()
    callee = expression.children[0]
    positional = tuple(
        item for item in expression.children[1:]
        if isinstance(item, ExprNode))
    proof = resolve_import_reference(
        index, caller.source, caller, callee)
    if proof is not None:
        position = _DISPATCH_FALLBACK_PROTOCOLS.get(proof.qualified_target)
        return ((positional[position],)
                if position is not None and position < len(positional) else ())

    if callee.kind != "name" or not callee.name:
        return ()
    selector_symbol = SymbolId(caller.source, callee.name)
    selector = index.callable_by_symbol(selector_symbol)
    if selector is None or selector.owner is not None:
        return ()
    returns = index.return_observations_in(selector_symbol)
    if not returns or any(item.guard or item.value is None
                          or item.value.kind != "name"
                          for item in returns):
        return ()
    returned_names = {item.value.name for item in returns}
    if len(returned_names) != 1:
        return ()
    returned = next(iter(returned_names))
    params = [param for param in selector.params
              if param.kind in {"positional", "posonly"}]
    positions = [i for i, param in enumerate(params) if param.name == returned]
    if len(positions) != 1 or positions[0] >= len(positional):
        return ()
    return (positional[positions[0]],)


def _exact_call_target(index: ProgramIndex, call: CallObservation):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    if proof is None:
        return None, ()
    spans = tuple(dict.fromkeys(
        span for span in (call.span, proof.binding.span)
        if isinstance(span, SourceSpan)))
    return proof.qualified_target, spans


def _callable_expressions(index, callable_symbol):
    for binding in index.bindings_in(callable_symbol):
        if binding.value is not None:
            yield binding.value, binding.guard
    for returned in index.return_observations_in(callable_symbol):
        if returned.value is not None:
            yield returned.value, returned.guard


def _matmul_spans(expression: ExprNode):
    out: list[SourceSpan] = []
    if expression.kind == "binop" and expression.operator == "@" \
            and isinstance(expression.span, SourceSpan):
        out.append(expression.span)
    for child in expression.children:
        if isinstance(child, ExprNode):
            out.extend(_matmul_spans(child))
    for _, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            out.extend(_matmul_spans(child))
    return tuple(out)


def _self_field(expression):
    if expression.kind != "attribute" or not expression.children:
        return None
    root = expression.children[0]
    return (expression.name if root.kind == "name" and root.name == "self"
            else None)


def _span_sort_key(span):
    return (
        span.source.component_key or "",
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


__all__ = [
    "AttentionChildEvidence",
    "AttentionComputeProof",
    "attention_child_evidence",
]
