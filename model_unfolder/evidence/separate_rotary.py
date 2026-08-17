"""Exact separate-call Q/K rotary application evidence.

U8's original application reader covers helpers that rotate Q and K together.
Vision implementations also commonly invoke one identical rotation helper once
per lane.  This boundary proves that shape without using helper, variable,
class, field, or model names: the exact Q and K score operands each depend on
one call to the same indexed callable; the non-tensor arguments are identical;
and that callable's exact local call closure reaches the canonical algebraic
half-turn protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_operands import (
    AttentionQKOperandsEvidence,
    attention_qk_operands_evidence,
)
from .attention_storage import producer_sources_reaching_expressions
from .component_owner import OwnerOccurrenceId, require_resolved_component_root
from .position_application import half_turn_rotation_protocol
from .program_index import (
    CallObservation,
    CallSiteId,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class SeparateQKRotaryEvidence:
    attention_occurrence: OwnerOccurrenceId
    operands: AttentionQKOperandsEvidence
    query_call: CallObservation
    key_call: CallObservation
    helper_callable: SymbolId
    rotation_callable: SymbolId
    spans: tuple[SourceSpan, ...]
    kind: str = "rope"
    application: str = "qk_rotation"

    def __post_init__(self):
        if not isinstance(self.attention_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.operands, AttentionQKOperandsEvidence):
            raise TypeError("separate rotary evidence is attention-owner qualified")
        if self.operands.attention_occurrence != self.attention_occurrence:
            raise ValueError("rotary evidence and score operands share an owner")
        if any(not isinstance(call, CallObservation)
               for call in (self.query_call, self.key_call)):
            raise TypeError("separate rotary evidence carries exact calls")
        if self.query_call == self.key_call \
                or self.query_call.enclosing_callable \
                != self.key_call.enclosing_callable:
            raise ValueError("Q and K use distinct calls in one callable")
        if not isinstance(self.helper_callable, SymbolId) \
                or not isinstance(self.rotation_callable, SymbolId):
            raise TypeError("rotary helper closure carries exact callables")
        required = {
            self.query_call.span, self.key_call.span,
            *self.operands.spans,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("separate rotary provenance closes Q, K and operands")
        if self.kind != "rope" or self.application != "qk_rotation":
            raise ValueError("the separate-call protocol has a closed meaning")


def read_separate_qk_rotary(index, root, attention):
    if not isinstance(index, ProgramIndex):
        raise TypeError("separate rotary reading requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="read_separate_qk_rotary")
    operands_result = attention_qk_operands_evidence(index, root, attention)
    if operands_result.status != "resolved":
        if operands_result.status == "ambiguous":
            return ReaderResult.ambiguous(
                attention.compute_occurrence, operands_result.ambiguity)
        return ReaderResult.failed(
            attention.compute_occurrence,
            operands_result.failures or (ReaderFailure(
                "incomplete_graph", "exact Q/K operands are unavailable"),))
    operands = operands_result.value
    callable_symbol = operands.entry_call.enclosing_callable
    calls = tuple(index.calls_in(callable_symbol))
    rotary_candidates = _rotary_call_candidates(index, calls)
    # Seed only semantically-proven rotary calls as candidate producers.  If
    # every call were seeded, a later transparent tensor method (transpose,
    # view, contiguous) would replace the rotary call as the terminal producer
    # even though its receiver still descends from rotation.
    producers = {
        CallSiteId.of(call): call
        for call, _helper, _rotation in rotary_candidates
        if call.span is not None
    }
    query_sources, _, query_dependencies, query_uncertain = \
        producer_sources_reaching_expressions(
        index, callable_symbol,
        ((operands.query_operand.span, (operands.query_operand,)),), producers)
    key_sources, _, key_dependencies, key_uncertain = \
        producer_sources_reaching_expressions(
        index, callable_symbol,
        ((operands.key_operand.span, (operands.key_operand,)),), producers)
    if query_uncertain or key_uncertain:
        return ReaderResult.failed(attention.compute_occurrence, (ReaderFailure(
            "incomplete_graph", "Q/K producer lineage is not exact"),))

    query_ids = _dependency_closure(query_sources, query_dependencies)
    key_ids = _dependency_closure(key_sources, key_dependencies)
    query = tuple(item for item in rotary_candidates
                  if CallSiteId.of(item[0]) in query_ids)
    key = tuple(item for item in rotary_candidates
                if CallSiteId.of(item[0]) in key_ids)
    candidates = []
    for query_call, query_helper, query_rotation in query:
        for key_call, key_helper, key_rotation in key:
            if query_call == key_call or query_helper != key_helper \
                    or query_rotation != key_rotation:
                continue
            if _factor_signature(query_call) != _factor_signature(key_call):
                continue
            spans = tuple(dict.fromkeys(
                span for span in (
                    *operands.spans, query_call.span, key_call.span,
                    index.callable_by_symbol(query_helper).span,
                    index.callable_by_symbol(query_rotation).span,
                ) if isinstance(span, SourceSpan)))
            candidates.append(SeparateQKRotaryEvidence(
                attention.compute_occurrence, operands,
                query_call, key_call, query_helper, query_rotation, spans))
    unique = {
        (item.query_call.span, item.key_call.span,
         item.helper_callable, item.rotation_callable): item
        for item in candidates}
    if len(unique) > 1:
        return ReaderResult.ambiguous(
            attention.compute_occurrence,
            Ambiguity(sites=tuple(
                span for item in unique.values()
                for span in (item.query_call.span, item.key_call.span))))
    if not unique:
        return ReaderResult.absent(attention.compute_occurrence)
    value = next(iter(unique.values()))
    return ReaderResult.resolved(
        attention.compute_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=value.spans,
            detail=("exact Q and K operands independently reach one indexed "
                    "half-turn rotation protocol")),))


def _rotary_call_candidates(index, calls):
    out = []
    for call in calls:
        helper = _local_callable(index, call)
        if helper is None:
            continue
        rotation = _rotation_in_closure(index, helper)
        if rotation is not None:
            out.append((call, helper, rotation))
    return tuple(out)


def _rotation_in_closure(index, root):
    queue = [root]
    seen = set()
    found = []
    while queue:
        symbol = queue.pop(0)
        if symbol in seen:
            continue
        seen.add(symbol)
        if half_turn_rotation_protocol(index, symbol):
            found.append(symbol)
            continue
        for call in index.calls_in(symbol):
            child = _local_callable(index, call)
            if child is not None and child not in seen:
                queue.append(child)
    return found[0] if len(set(found)) == 1 else None


def _dependency_closure(sources, dependencies):
    found = set(sources)
    queue = list(sources)
    while queue:
        current = queue.pop()
        for upstream in dependencies.get(current, ()):
            if upstream not in found:
                found.add(upstream)
                queue.append(upstream)
    return frozenset(found)


def _local_callable(index, call):
    callee = call.callee
    if callee.kind == "name" and callee.name:
        # Exact same-module function binding.  A local store/parameter with the
        # same spelling blocks the address rather than being guessed through.
        if any(item.name == callee.name and item.context in {"parameter", "store"}
               for item in index.identifiers_in(call.enclosing_callable)):
            return None
        symbol = SymbolId(call.enclosing_callable.source, callee.name)
        record = index.callable_by_symbol(symbol)
        return symbol if record is not None and record.owner is None else None
    if callee.kind == "attribute" and len(callee.children) == 1 \
            and callee.children[0].kind == "name" \
            and callee.children[0].name == "self" and call.owner is not None:
        symbol = SymbolId(call.owner.source, f"{call.owner.qualified_name}.{callee.name}")
        return symbol if index.callable_by_symbol(symbol) is not None else None
    return None


def _factor_signature(call):
    # The tensor lane is argument 0.  Every other positional/keyword operand
    # must be structurally identical between Q and K; spans are provenance,
    # never semantic equality.
    return (
        tuple(_expr_shape(item) for item in call.args[1:]),
        tuple((name, _expr_shape(value)) for name, value in call.kwargs
              if name not in {"x", "query", "key"}),
    )


def _expr_shape(expression):
    if not isinstance(expression, ExprNode):
        return None
    return (
        expression.kind, expression.name, expression.const_value,
        expression.operator,
        tuple(_expr_shape(item) for item in expression.children),
        tuple((name, _expr_shape(item))
              for name, item in expression.keyword_children),
    )


__all__ = ["SeparateQKRotaryEvidence", "read_separate_qk_rotary"]
