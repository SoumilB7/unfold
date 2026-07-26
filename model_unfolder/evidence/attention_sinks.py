"""Exact-owner learned attention-sink evidence.

This reader proves only the positive mechanism:

1. the selected decoder block invokes one exact attention child;
2. that child constructs an exact ``torch.nn.Parameter`` field;
3. the exact attention-compute callable receives that child as an argument;
4. the parameter value and an exact attention-score producer meet in
   ``torch.cat``; and
5. the joined value reaches the exact softmax input.

Names such as ``sinks`` or ``s_aux`` are never evidence.  A parameter elsewhere
in the file, a parameter unused by the score path, or a concat after softmax
cannot satisfy the proof.  Source silence remains unknown and is represented by
a failed :class:`ReaderResult`, never by code-proven absence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import (
    AttentionChildEvidence,
    attention_child_evidence,
)
from .component_owner import OwnerOccurrenceId
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_import_reference,
)
from .decoder_block import decoder_block_path_for_config
from .models import SourceBundle
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


_PARAMETER_PROTOCOLS = frozenset({
    "torch.nn.Parameter",
    "torch.nn.parameter.Parameter",
})
_CAT_PROTOCOLS = frozenset({"torch.cat", "torch.concat", "torch.concatenate"})
_SOFTMAX_PROTOCOLS = frozenset({
    "torch.softmax",
    "torch.nn.functional.softmax",
})
_SCORE_PROTOCOLS = frozenset({
    "torch.bmm",
    "torch.einsum",
    "torch.matmul",
})

_SINK = "learned_parameter"
_SCORE = "attention_score"
_JOIN = "score_parameter_join"


@dataclass(frozen=True)
class AttentionSinkEvidence:
    """One exact learned parameter joining scores before softmax."""

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    attention_symbol: SymbolId
    parameter: ConstructionOccurrenceId
    compute_callable: SymbolId
    join_call: CallObservation
    softmax_call: CallObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("attention-sink evidence is occurrence-qualified")
        if not isinstance(self.attention_symbol, SymbolId) \
                or not isinstance(self.compute_callable, SymbolId):
            raise TypeError("attention-sink evidence carries exact symbols")
        if not isinstance(self.parameter, ConstructionOccurrenceId):
            raise TypeError("attention-sink evidence carries one construction")
        if not isinstance(self.join_call, CallObservation) \
                or not isinstance(self.softmax_call, CallObservation):
            raise TypeError("attention-sink evidence carries exact calls")
        if self.parameter.parent != self.attention_occurrence:
            raise ValueError("the learned parameter belongs to the attention owner")
        if self.attention_occurrence.sites[:-1] != self.block_occurrence.sites:
            raise ValueError("the attention owner is an immediate block child")
        if self.join_call.enclosing_callable != self.compute_callable \
                or self.softmax_call.enclosing_callable != self.compute_callable:
            raise ValueError("join and softmax belong to the compute callable")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("attention-sink evidence carries exact provenance")
        if self.parameter.site.span not in self.spans \
                or self.join_call.span not in self.spans \
                or self.softmax_call.span not in self.spans:
            raise ValueError("provenance includes construction, join and softmax")
        if any(span.source != self.attention_symbol.source
               for span in self.spans):
            raise ValueError("sink provenance belongs to the attention source")


def decoder_attention_sinks_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[AttentionSinkEvidence]:
    """Prove a learned score-sink mechanism for one selected decoder path."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("decoder_attention_sinks_for_path needs ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("decoder_attention_sinks_for_path needs SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    attention = attention_child_evidence(
        index, block.value.component_root, block.value.block_occurrence)
    if attention.status != "resolved":
        return attention
    result = _attention_sink_at_child(
        index, attention.value)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(
            *block.provenance,
            *attention.provenance,
            *result.provenance,
        ))


def _attention_sink_at_child(
    index: ProgramIndex,
    attention: AttentionChildEvidence,
) -> ReaderResult[AttentionSinkEvidence]:
    child = attention.compute.child_symbol
    owner = attention.child_occurrence
    compute = attention.compute.callable_symbol
    root_binding = _compute_receiver_binding(index, attention)
    if root_binding is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the compute callable is not exactly bound to the attention child"),))

    candidates = _parameter_candidates(index, owner, child)
    if not candidates:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the exact attention child has no unconditional learned Parameter"),))

    calls = tuple(index.calls_in(compute))
    targets = {call: _exact_call_target(index, call) for call in calls}
    # ``AttentionComputeProof.input_calls`` are the calls that enter the
    # compute boundary (the dispatch call for a followed free function).  The
    # score/softmax sites themselves are the exact calls indexed *inside* the
    # proven compute callable.
    score_calls = tuple(
        call for call in calls if targets.get(call) in _SCORE_PROTOCOLS)
    softmax_calls = tuple(
        call for call in calls if targets.get(call) in _SOFTMAX_PROTOCOLS)
    join_calls = tuple(
        call for call in calls if targets.get(call) in _CAT_PROTOCOLS)
    if not score_calls or not softmax_calls or not join_calls:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the exact attention proof has no score -> concat -> softmax chain"),))

    proven = []
    for field, occurrence, site in candidates:
        relation = _prove_join_before_softmax(
            index, compute, root_binding, field,
            score_calls, join_calls, softmax_calls)
        if relation is not None:
            join_call, softmax_call = relation
            spans = tuple(dict.fromkeys((
                site.span,
                attention.compute.entry_call.span,
                *(call.span for call in score_calls),
                join_call.span,
                softmax_call.span,
            )))
            proven.append(AttentionSinkEvidence(
                attention.block_occurrence,
                owner,
                child,
                occurrence,
                compute,
                join_call,
                softmax_call,
                tuple(span for span in spans
                      if isinstance(span, SourceSpan)),
            ))
    if not proven:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no exact learned Parameter joins scores before softmax"),))
    if len(proven) > 1:
        return ReaderResult.ambiguous(
            owner,
            Ambiguity(sites=tuple(item.parameter.site.span
                                  for item in proven)))
    evidence = proven[0]
    return ReaderResult.resolved(
        owner, evidence,
        provenance=(ReaderProvenance(
            "source",
            spans=evidence.spans,
            detail=(
                "an exact learned Parameter and exact score producer join "
                "before the exact attention softmax")),),
    )


def _parameter_candidates(index, owner, symbol):
    candidates = []
    for site in index.construction_sites_of(symbol):
        if site.owner != symbol or site.target_kind != "field" \
                or site.guard or not site.target:
            continue
        callee = _call_callee(site.constructor)
        if callee is None:
            continue
        proof = resolve_import_reference(
            index, symbol.source, site.enclosing_callable, callee)
        if proof is None or proof.qualified_target not in _PARAMETER_PROTOCOLS:
            continue
        candidates.append((
            site.target,
            ConstructionOccurrenceId(owner, site.site_id),
            site,
        ))
    return tuple(candidates)


def _compute_receiver_binding(index, attention):
    """Return the exact compute-callable parameter bound to child ``self``."""
    compute = attention.compute.callable_symbol
    record = index.callable_by_symbol(compute)
    if record is None:
        return None
    if record.owner == attention.compute.child_symbol:
        return "self"
    positional = tuple(
        param for param in record.params
        if param.kind in {"positional", "posonly"})
    bound = []
    for position, argument in enumerate(attention.compute.entry_call.args):
        if position >= len(positional):
            break
        if argument.kind == "name" and argument.name == "self":
            bound.append(positional[position].name)
    return bound[0] if len(bound) == 1 else None


def _prove_join_before_softmax(
    index,
    callable_symbol,
    receiver_binding,
    field,
    score_calls,
    join_calls,
    softmax_calls,
):
    score_spans = frozenset(call.span for call in score_calls)
    join_by_span = {call.span: call for call in join_calls}
    bindings = tuple(sorted(
        index.bindings_in(callable_symbol),
        key=lambda item: _span_key(item.span)))
    for softmax in sorted(softmax_calls, key=lambda item: _span_key(item.span)):
        env = {}
        uncertain = {}
        joined = {}
        for binding in bindings:
            if binding.span is None or not _span_before(
                    binding.span, softmax.span):
                continue
            tags = _expression_tags(
                binding.value, env, receiver_binding, field, score_spans)
            join = join_by_span.get(
                binding.value.span if binding.value is not None else None)
            if join is not None and {_SINK, _SCORE}.issubset(tags):
                tags = frozenset((*tags, _JOIN))
                joined[_JOIN] = join
            targets = tuple(
                name
                for target in binding.targets
                for name in _target_names(target))
            for name in targets:
                if binding.guard:
                    previous = env.get(name, frozenset())
                    env[name] = frozenset((*previous, *tags))
                    # A guarded transformation that preserves exactly the
                    # same provenance on both paths (for example optional
                    # mask addition to already-proven scores) is not a rival
                    # reaching definition.  A changed tag set remains
                    # uncertain and cannot certify the sink chain.
                    uncertain[name] = (
                        uncertain.get(name, False) or tags != previous)
                else:
                    env[name] = tags
                    uncertain[name] = any(
                        uncertain.get(dep, False)
                        for dep in _names_in(binding.value))
        softmax_tags = frozenset(
            tag
            for expression in (*softmax.args,
                               *(value for _, value in softmax.kwargs))
            for tag in _expression_tags(
                expression, env, receiver_binding, field, score_spans))
        if _JOIN not in softmax_tags:
            continue
        if any(uncertain.get(name, False)
               for expression in softmax.args
               for name in _names_in(expression)):
            continue
        join = joined.get(_JOIN)
        if join is not None:
            return join, softmax
    return None


def _expression_tags(expression, env, receiver_binding, field, score_spans):
    if expression is None:
        return frozenset()
    tags = set()
    if expression.kind == "name":
        tags.update(env.get(expression.name, ()))
    if _attribute_chain(expression) == (receiver_binding, field):
        tags.add(_SINK)
    if expression.kind == "call" and expression.span in score_spans:
        tags.add(_SCORE)
    for child in expression.children:
        tags.update(_expression_tags(
            child, env, receiver_binding, field, score_spans))
    for _, child in expression.keyword_children:
        tags.update(_expression_tags(
            child, env, receiver_binding, field, score_spans))
    return frozenset(tags)


def _attribute_chain(expression):
    parts = []
    current = expression
    while isinstance(current, ExprNode) and current.kind == "attribute":
        parts.append(current.name)
        current = current.children[0] if current.children else None
    if not isinstance(current, ExprNode) or current.kind != "name":
        return ()
    return tuple((current.name, *reversed(parts)))


def _names_in(expression):
    if expression is None:
        return frozenset()
    names = {expression.name} if expression.kind == "name" else set()
    for child in expression.children:
        names.update(_names_in(child))
    for _, child in expression.keyword_children:
        names.update(_names_in(child))
    return frozenset(names)


def _target_names(expression):
    if expression.kind == "name":
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(
            name for child in expression.children
            for name in _target_names(child))
    return ()


def _exact_call_target(index, call):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    return proof.qualified_target if proof is not None else ""


def _call_callee(expression):
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        return callee if isinstance(callee, ExprNode) else None
    return None


def _span_key(span):
    if span is None:
        return ("", 0, 0, 0, 0)
    return (
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (
        left.end_line or left.line,
        left.end_col or left.col,
    ) <= (right.line, right.col)


__all__ = [
    "AttentionSinkEvidence",
    "decoder_attention_sinks_for_path",
]
