"""Exact, role-neutral attention score operand evidence.

This boundary derives the two score operands from an already-proven attention
compute protocol.  It deliberately does not ask how either operand was
projected or stored.  Projection storage and score participation are separate
architectural facts: latent, conditional, or sliced projections can still
produce the exact values used as query and key by attention.

For a direct framework attention call, the first two protocol operands are the
score operands.  For a positively-proven dot/softmax helper, the operands are
the two exact dot inputs whose result reaches the exact softmax call.  Their
formal origins are then bound back to the exact entry call.  Names are never
used as roles; formal names are only Python call-addresses after the callable
is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import AttentionChildEvidence
from .attention_storage import producer_sources_reaching_expressions
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .construction_calls import resolve_import_reference
from .program_index import (
    CallObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_SDPA_PROTOCOLS = frozenset({
    "torch.nn.functional.scaled_dot_product_attention",
    "...modeling_flash_attention_utils._flash_attention_forward",
    "transformers.modeling_flash_attention_utils._flash_attention_forward",
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


@dataclass(frozen=True)
class AttentionQKOperandsEvidence:
    """Two exact caller expressions that enter attention as score operands."""

    attention_occurrence: OwnerOccurrenceId
    attention_symbol: SymbolId
    compute_callable: SymbolId
    entry_call: CallObservation
    score_call: CallObservation
    query_operand: ExprNode
    key_operand: ExprNode
    protocol: str                 # direct_attention | dot_softmax
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("attention operands name an exact owner occurrence")
        if not isinstance(self.attention_symbol, SymbolId) \
                or not isinstance(self.compute_callable, SymbolId):
            raise TypeError("attention operands name exact owner/callable symbols")
        if not isinstance(self.entry_call, CallObservation) \
                or not isinstance(self.score_call, CallObservation):
            raise TypeError("attention operands carry exact entry and score calls")
        if self.protocol not in {"direct_attention", "dot_softmax"}:
            raise ValueError("attention operands carry a known proof protocol")
        if any(not isinstance(item, ExprNode) or item.span is None
               for item in (self.query_operand, self.key_operand)):
            raise TypeError("attention operands carry exact source expressions")
        source = self.attention_occurrence.root.source
        if self.attention_symbol.source != source \
                or self.entry_call.owner != self.attention_symbol \
                or self.entry_call.enclosing_callable.source != source \
                or self.compute_callable.source != source \
                or self.score_call.enclosing_callable != self.compute_callable:
            raise ValueError("attention operands belong to one exact owner/source")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 or span.source != source for span in self.spans):
            raise ValueError("attention operand provenance is source-qualified")
        required = {
            self.entry_call.span,
            self.score_call.span,
            self.query_operand.span,
            self.key_operand.span,
        }
        if None in required or not required.issubset(self.spans):
            raise ValueError("attention operand provenance cites every boundary")


def attention_qk_operands_evidence(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    attention: AttentionChildEvidence,
) -> ReaderResult[AttentionQKOperandsEvidence]:
    """Prove the exact query/key score operands of one attention occurrence."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention operand evidence requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_qk_operands_evidence")
    if not isinstance(attention, AttentionChildEvidence):
        raise TypeError("attention operand evidence requires attention evidence")
    owner = attention.compute_occurrence
    node = root.graph.node_for(owner)
    if node is None or node.symbol != attention.compute_owner_symbol \
            or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "attention occurrence does not round-trip"),))
    compute = attention.compute
    if compute.child_symbol != node.symbol:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "attention compute proof names another owner"),))

    candidates = []
    if compute.protocol == "scaled_dot_product_attention":
        candidates.extend(_direct_candidates(
            index, owner, node.symbol, compute))
    elif compute.protocol == "dot_softmax":
        candidate = _dot_softmax_candidate(
            index, owner, node.symbol, compute)
        if candidate is not None:
            candidates.append(candidate)
    if len(candidates) == 1:
        value = candidates[0]
        return ReaderResult.resolved(owner, value, provenance=(
            ReaderProvenance(
                "source", spans=value.spans,
                detail="exact expressions proven as attention score operands"),))
    if len(candidates) > 1:
        from .reader_result import Ambiguity
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(
                item.score_call.span for item in candidates)))
    return ReaderResult.failed(owner, (ReaderFailure(
        "incomplete_graph",
        "exact query/key score operands were not proven"),))


def _direct_candidates(index, owner, owner_symbol, compute):
    out = []
    for call in index.calls_in(compute.callable_symbol):
        target = _exact_target(index, compute.callable_symbol, call)
        if target not in _SDPA_PROTOCOLS or call.guard or len(call.args) < 2:
            continue
        q, k = call.args[:2]
        if call.span is None:
            continue
        entry = compute.entry_call
        # When the protocol is inside the child forward, its operands already
        # live in the entry callable.  A descended helper is mapped through its
        # exact Python argument binding below instead.
        mapped = ((q, k) if entry.enclosing_callable == compute.callable_symbol
                  else _map_formal_operands_to_entry(
                      index, compute.callable_symbol, entry, (q, k), call.span))
        if mapped is None:
            continue
        spans = _spans(entry, call, *mapped)
        out.append(AttentionQKOperandsEvidence(
            owner, owner_symbol, compute.callable_symbol, entry, call,
            mapped[0], mapped[1], "direct_attention", spans))
    return tuple(out)


def _dot_softmax_candidate(index, owner, owner_symbol, compute):
    callable_symbol = compute.callable_symbol
    dots = {}
    softmaxes = []
    for call in index.calls_in(callable_symbol):
        if call.guard:
            continue
        target = _exact_target(index, callable_symbol, call)
        if target in _DOT_PROTOCOLS:
            operands = _dot_operands(target, call)
            if operands is not None:
                dots[call] = operands
        elif target in _SOFTMAX_PROTOCOLS:
            softmaxes.append(call)
    reaching = set()
    for softmax in softmaxes:
        sources, _widths, _dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, callable_symbol,
                ((softmax.span, (*softmax.args,
                                 *(value for _key, value in softmax.kwargs))),),
                {call: call for call in dots})
        if uncertain:
            return None
        reaching.update(sources)
    if len(reaching) != 1:
        return None
    score_call = next(iter(reaching))
    q, k = dots[score_call]
    mapped = (
        (q, k)
        if compute.entry_call.enclosing_callable == callable_symbol
        else _map_formal_operands_to_entry(
            index, callable_symbol, compute.entry_call, (q, k), score_call.span))
    if mapped is None:
        return None
    spans = _spans(compute.entry_call, score_call, *mapped)
    return AttentionQKOperandsEvidence(
        owner, owner_symbol, callable_symbol, compute.entry_call, score_call,
        mapped[0], mapped[1], "dot_softmax", spans)


def _dot_operands(target, call):
    if target in {"torch.matmul", "torch.bmm"} and len(call.args) >= 2:
        return call.args[0], call.args[1]
    if target == "torch.einsum" and len(call.args) >= 3 \
            and call.args[0].kind == "constant" \
            and isinstance(call.args[0].const_value, str):
        return call.args[1], call.args[2]
    return None


def _map_formal_operands_to_entry(
        index, callable_symbol, entry, operands, consumer_span):
    """Map exact helper-formal origins back to exact entry actuals."""
    record = index.callable_by_symbol(callable_symbol)
    if record is None or entry.span is None:
        return None
    formals = tuple(item for item in record.params
                    if item.kind not in {"vararg", "kwarg"})
    if not formals:
        return None
    initial = {item.name: ("formal", number)
               for number, item in enumerate(formals)}
    formal_origins = []
    for operand in operands:
        sources, _widths, _dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, callable_symbol, ((consumer_span, (operand,)),), {},
                initial_sources=initial,
                preserve_local_tuple_lanes=True)
        if uncertain:
            return None
        formal_origins.append({item[1] for item in sources
                               if isinstance(item, tuple)
                               and len(item) == 2 and item[0] == "formal"})
    actuals = _bind_entry_actuals(entry, formals)
    # Exact owner-state actuals can participate in tensor reshaping (for
    # example a repetition count) but cannot be the score value itself.
    state_formals = {
        number for number, actual in actuals.items()
        if actual.kind == "name" and actual.name == "self"
    }
    resolved = []
    for origins in formal_origins:
        candidates = tuple(origins - state_formals)
        if len(candidates) != 1 or candidates[0] not in actuals:
            return None
        resolved.append(actuals[candidates[0]])
    return tuple(resolved)


def _bind_entry_actuals(call, formals):
    """Python's explicit positional/keyword binding, without defaults."""
    out = {}
    positional = tuple(number for number, item in enumerate(formals)
                       if item.kind in {"positional", "posonly"})
    for number, actual in enumerate(call.args):
        if number >= len(positional):
            break
        out[positional[number]] = actual
    by_name = {item.name: number for number, item in enumerate(formals)
               if item.kind != "posonly"}
    for name, actual in call.kwargs:
        if name == "**" or name not in by_name:
            continue
        number = by_name[name]
        if number in out:
            return {}
        out[number] = actual
    return out


def _exact_target(index, callable_symbol, call):
    proof = resolve_import_reference(
        index, callable_symbol.source, callable_symbol, call.callee)
    return proof.qualified_target if proof is not None else None


def _spans(entry, score, *operands):
    return tuple(dict.fromkeys(
        span for span in (
            entry.span, score.span, *(item.span for item in operands))
        if isinstance(span, SourceSpan)))


__all__ = [
    "AttentionQKOperandsEvidence",
    "attention_qk_operands_evidence",
]
