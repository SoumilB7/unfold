"""Exact model-stage learned-absolute position application evidence.

This reader proves one positive relation and nothing broader::

    exact coordinate producer -> exact embedding primitive
        -> exact addition into the hidden stream -> exact repeated child

Field names, class names, model identity and config-field presence never select
the mechanism.  A fixed/sinusoidal producer, score-side bias and rotary
application are separate U8 evidence families.  Failure to prove this relation
therefore means ``unknown``, not ``no positional mechanism``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import producer_sources_reaching_expressions
from .construction_calls import resolve_construction_call
from .container_inventory import resolve_container_inventory
from .decoder_block import DecoderBlockPath, decoder_block_path_for_config
from .execution_flow import resolve_addressed_invocations
from .models import SourceBundle
from .primitive_semantics import (
    classify_primitive_alternative,
    classify_primitive_call,
)
from .position_coordinate import coordinate_origin
from .program_index import (
    BindingObservation,
    CallObservation,
    CallSiteId,
    ExprNode,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


@dataclass(frozen=True)
class LearnedAbsolutePositionEvidence:
    """One exact learned-position lookup and pre-stack addition."""

    owner: object
    embedding_call: CallObservation
    coordinate_spans: tuple[SourceSpan, ...]
    addition: BindingObservation
    repeated_call_sites: tuple[CallSiteId, ...]
    provenance_spans: tuple[SourceSpan, ...]
    kind: str = "learned_absolute"
    application: str = "embedding_add"

    def __post_init__(self) -> None:
        from .component_owner import OwnerOccurrenceId
        if not isinstance(self.owner, OwnerOccurrenceId):
            raise TypeError("absolute-position evidence has an exact owner")
        if not isinstance(self.embedding_call, CallObservation) \
                or self.embedding_call.span is None:
            raise TypeError("absolute-position evidence cites one exact call")
        if self.embedding_call.enclosing_callable.source \
                != self.owner.root.source:
            raise ValueError("the embedding call belongs to the owning source")
        if not self.coordinate_spans or any(
                not isinstance(span, SourceSpan)
                for span in self.coordinate_spans):
            raise ValueError("coordinate evidence retains exact source spans")
        if not isinstance(self.addition, BindingObservation) \
                or self.addition.value is None \
                or self.addition.value.kind != "binop" \
                or self.addition.value.operator != "+":
            raise TypeError("absolute-position application is an exact addition")
        if self.addition.guard:
            raise ValueError("a general position addition is not branch-guarded")
        if not self.repeated_call_sites \
                or len(set(self.repeated_call_sites)) \
                != len(self.repeated_call_sites):
            raise ValueError("repeated call sites are non-empty and unique")
        if any(site.enclosing_callable != self.embedding_call.enclosing_callable
               for site in self.repeated_call_sites):
            raise ValueError("position lookup and repeated calls share the stage")
        required = {
            self.embedding_call.span,
            self.addition.span,
            *self.coordinate_spans,
            *(site.span for site in self.repeated_call_sites),
        }
        if None in required or not required <= set(self.provenance_spans):
            raise ValueError("provenance closes lookup, coordinate, add and sink")
        if len(set(self.provenance_spans)) != len(self.provenance_spans):
            raise ValueError("absolute-position provenance is unique")
        if self.kind != "learned_absolute" \
                or self.application != "embedding_add":
            raise ValueError("this DTO expresses only learned embedding addition")


def decoder_learned_absolute_position_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[LearnedAbsolutePositionEvidence]:
    """Resolve an exact decoder path, then read only its model-stage forward."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("learned-absolute evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("learned-absolute evidence requires a SourceBundle")
    path = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if path.status != "resolved":
        return ReaderResult.failed(path.owner, (
            ReaderFailure(
                "incomplete_graph",
                "exact decoder path is not resolved: " + "; ".join(
                    item.detail for item in path.failures)),))
    return read_learned_absolute_position(index, path.value)


def read_learned_absolute_position(
    index: ProgramIndex,
    path: DecoderBlockPath,
) -> ReaderResult[LearnedAbsolutePositionEvidence]:
    """Prove coordinate lookup + addition + exact repeated-child reachability."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("learned-absolute reader requires a ProgramIndex")
    if not isinstance(path, DecoderBlockPath):
        raise TypeError("learned-absolute reader requires a DecoderBlockPath")
    root = path.component_root
    owner = path.stage_occurrence
    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    if invocations.status != "resolved":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            f"model-stage invocation census is {invocations.status}: "
            f"{invocations.failure_kind}: {invocations.failure_detail}"),))

    repeated_calls = tuple(dict.fromkeys(
        proof.template.call for proof in path.repeated_child.proofs))
    repeated_sites = tuple(CallSiteId.of(call) for call in repeated_calls)
    if not repeated_calls:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the decoder path retained no repeated call"),))
    callable_symbol = repeated_calls[0].enclosing_callable
    if any(call.enclosing_callable != callable_symbol
           for call in repeated_calls):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "repeated calls do not share one stage callable"),))

    primitive_calls = []
    for invocation in invocations.external_addressed:
        primitive = classify_primitive_alternative(
            index, invocation.construction)
        if primitive.status == "resolved" and primitive.value == "embedding":
            primitive_calls.append((
                invocation.call_site, invocation.call, primitive.provenance,
                None))
    for invocation in invocations.addressed:
        construction = resolve_construction_call(
            index, root, owner, invocation.call)
        primitive = classify_primitive_call(index, construction)
        if primitive.status == "resolved" and primitive.value == "embedding":
            primitive_calls.append((
                invocation.call_site, invocation.call, primitive.provenance,
                None))

    coordinate_calls = []
    for site, call, primitive_provenance, internal_coordinate in primitive_calls:
        if not call.args:
            continue
        coordinate = coordinate_origin(
            index, callable_symbol, call.args[0], call.span)
        if coordinate is not None:
            coordinate_calls.append((
                site, call, coordinate.spans, primitive_provenance))
    if not coordinate_calls:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no code-proven coordinate value feeds an embedding primitive"),))

    qualified = []
    for site, call, coordinate_spans, primitive_provenance in coordinate_calls:
        additions = pre_stack_additions_for_producer(
            index, callable_symbol, site, call, repeated_calls)
        for addition in additions:
            spans = [call.span, addition.span, *coordinate_spans]
            for item in primitive_provenance:
                spans.extend(item.spans)
            spans.extend(site.span for site in repeated_sites)
            exact_spans = tuple(dict.fromkeys(
                span for span in spans if isinstance(span, SourceSpan)))
            qualified.append(LearnedAbsolutePositionEvidence(
                owner, call, coordinate_spans, addition, repeated_sites,
                exact_spans))

    by_identity = {
        (CallSiteId.of(item.embedding_call), item.addition.span): item
        for item in qualified
    }
    if len(by_identity) > 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(sorted(
                (item.addition.span for item in by_identity.values()),
                key=_span_key))))
    if not by_identity:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "a coordinate embedding was not unconditionally added into the "
            "stream reaching the exact repeated child"),))
    value = next(iter(by_identity.values()))
    return ReaderResult.resolved(
        owner, value,
        provenance=(ReaderProvenance(
            "source", spans=value.provenance_spans,
            detail=(
                "exact coordinate protocol -> exact embedding primitive -> "
                "unconditional addition -> exact repeated-child invocation")),))


def pre_stack_additions_for_producer(
        index, callable_symbol, producer_site, producer_call, repeated_calls):
    """Exact producer -> addition -> repeated-child positive relation.

    This helper is mechanism-neutral.  The caller must independently prove
    what the producer is (learned embedding, fixed sinusoid, or another future
    mechanism); a call reaching an addition is never itself position evidence.
    """
    producer_calls = {producer_site: producer_call}
    out = []
    for binding in index.bindings_in(callable_symbol):
        if binding.guard or binding.value is None \
                or len(binding.targets) != 1 \
                or binding.targets[0].kind != "name" \
                or not binding.targets[0].name:
            continue
        for addition in _addition_nodes(binding.value):
            left, right = addition.children
            left_sources, _, _, left_uncertain = \
                producer_sources_reaching_expressions(
                    index, callable_symbol,
                    ((addition.span, (left,)),), producer_calls)
            right_sources, _, _, right_uncertain = \
                producer_sources_reaching_expressions(
                    index, callable_symbol,
                    ((addition.span, (right,)),), producer_calls)
            in_left = producer_site in left_sources
            in_right = producer_site in right_sources
            # Uncertainty on the OTHER operand is irrelevant to the positive
            # positional relation.  GPT-2 accepts either caller-provided token
            # embeddings or a guarded token lookup, while its independent
            # coordinate lookup and addition remain exact.  Only uncertainty
            # attached to the side carrying THIS producer can weaken the proof.
            if in_left == in_right \
                    or (in_left and left_uncertain) \
                    or (in_right and right_uncertain):
                continue
            marker = ("position_add", producer_site, binding.span)
            reached, _, _, uncertain = producer_sources_reaching_expressions(
                index, callable_symbol,
                tuple((call.span, call.args) for call in repeated_calls), {},
                initial_sources={binding.targets[0].name: marker},
                binding_predicate=lambda later, boundary=binding.span:
                    later.span is not None
                    and boundary is not None
                    and _span_before(boundary, later.span))
            if marker in reached and not uncertain:
                out.append(BindingObservation(
                    binding.owner, binding.enclosing_callable,
                    binding.statement, binding.targets, addition,
                    binding.assignment_kind, binding.guard, binding.span))
    return tuple(out)


def _addition_nodes(expression):
    out = []
    if expression.kind == "binop" and expression.operator == "+" \
            and len(expression.children) == 2:
        out.append(expression)
    for child in expression.children:
        if isinstance(child, ExprNode):
            out.extend(_addition_nodes(child))
    for _, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            out.extend(_addition_nodes(child))
    return tuple(out)


def _span_before(first, second):
    if first is None or second is None or first.source != second.source:
        return False
    return (
        first.end_line or first.line,
        first.end_col or first.col,
    ) <= (second.line, second.col)


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
    "LearnedAbsolutePositionEvidence",
    "decoder_learned_absolute_position_for_path",
    "read_learned_absolute_position",
    "pre_stack_additions_for_producer",
]
