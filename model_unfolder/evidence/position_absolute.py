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
from .construction_calls import (
    resolve_construction_call,
    resolve_import_reference,
)
from .container_inventory import resolve_container_inventory
from .decoder_block import DecoderBlockPath, decoder_block_path_for_config
from .execution_flow import resolve_addressed_invocations
from .models import SourceBundle
from .primitive_semantics import (
    classify_primitive_alternative,
    classify_primitive_call,
)
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


_COORDINATE_PROTOCOLS = frozenset({"torch.arange"})
_COORDINATE_WRAPPERS = frozenset({
    "clone", "contiguous", "expand", "expand_as", "long", "reshape",
    "to", "type_as", "unsqueeze", "view",
})


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
        internal_coordinate = _internal_coordinate_embedding_protocol(
            index, construction)
        if primitive.status == "resolved" and primitive.value == "embedding":
            primitive_calls.append((
                invocation.call_site, invocation.call, primitive.provenance,
                internal_coordinate))
        elif internal_coordinate is not None:
            # A subclass may override ``forward`` solely to construct exact
            # coordinates before delegating to its exact nn.Embedding base.
            # The generic primitive reader intentionally does not assume that
            # every override preserves base semantics; this narrower protocol
            # proves both the delegation and the coordinate construction.
            primitive_calls.append((
                invocation.call_site, invocation.call, (),
                internal_coordinate))

    coordinate_calls = []
    for site, call, primitive_provenance, internal_coordinate in primitive_calls:
        if not call.args:
            continue
        coordinate = internal_coordinate or _coordinate_origin(
            index, callable_symbol, call.args[0], call.span, frozenset())
        if coordinate is not None:
            coordinate_calls.append((
                site, call, tuple(coordinate), primitive_provenance))
    if not coordinate_calls:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no code-proven coordinate value feeds an embedding primitive"),))

    qualified = []
    for site, call, coordinate_spans, primitive_provenance in coordinate_calls:
        additions = _qualifying_additions(
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


def _qualifying_additions(index, callable_symbol, producer_site, producer_call,
                          repeated_calls):
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


def _coordinate_origin(index, callable_symbol, expression, before, seen):
    if expression is None or expression.span is None:
        return None
    key = (expression.kind, expression.span)
    if key in seen:
        return None
    seen = seen | {key}
    if expression.kind == "call" and expression.children:
        target = _resolved_call_target(index, callable_symbol, expression)
        if target in _COORDINATE_PROTOCOLS:
            return (expression.span,)
        callee = expression.children[0]
        if callee.kind == "attribute" and callee.name in _COORDINATE_WRAPPERS \
                and callee.children:
            inner = _coordinate_origin(
                index, callable_symbol, callee.children[0], expression.span, seen)
            return ((*(inner or ()), expression.span) if inner else None)
        return None
    if expression.kind == "binop" and expression.operator in {"+", "-"} \
            and len(expression.children) == 2:
        origins = tuple(
            _coordinate_origin(index, callable_symbol, child, before, seen)
            for child in expression.children)
        present = tuple(item for item in origins if item is not None)
        if len(present) == 1:
            return (*present[0], expression.span)
        return None
    if expression.kind == "name" and expression.name:
        bindings = tuple(
            item for item in index.bindings_in(callable_symbol)
            if item.span is not None and _span_before(item.span, before)
            and len(item.targets) == 1
            and item.targets[0].kind == "name"
            and item.targets[0].name == expression.name
            and item.value is not None)
        if not bindings:
            return None
        latest = max(bindings, key=lambda item: _span_key(item.span))
        if latest.guard and not _exact_none_default_guard(
                index, callable_symbol, expression.name, latest.guard):
            return None
        origin = _coordinate_origin(
            index, callable_symbol, latest.value, latest.span, seen)
        return ((*(origin or ()), latest.span) if origin else None)
    return None


def _exact_none_default_guard(index, callable_symbol, name, guard):
    record = index.callable_by_symbol(callable_symbol)
    param = next((item for item in (record.params if record else ())
                  if item.name == name), None)
    if param is None or not param.has_default or param.default is None \
            or param.default.kind != "constant" \
            or param.default.const_value is not None:
        return False
    step = guard[-1] if guard else None
    test = step.test if step is not None and step.kind in {"if", "elif"} else None
    if test is None or test.kind != "compare" or test.operator != "is" \
            or len(test.children) != 2:
        return False
    left, right = test.children
    return (
        left.kind == "name" and left.name == name
        and right.kind == "constant" and right.const_value is None
    ) or (
        right.kind == "name" and right.name == name
        and left.kind == "constant" and left.const_value is None
    )


def _internal_coordinate_embedding_protocol(index, construction):
    """Prove a coordinate-building override of one exact nn.Embedding base.

    This covers source shapes such as OPT without teaching the generic
    primitive classifier that every subclass override preserves its base.  The
    protocol requires an exact external embedding base, a None-defaulted
    coordinate formal, an exact cumsum branch for that formal, and one direct
    unguarded ``super().forward(coordinate_expr)`` return.
    """
    if construction.status != "resolved" \
            or construction.selected.kind != "internal":
        return None
    symbol = construction.selected.internal_symbol
    class_record = index.class_by_symbol(symbol)
    if class_record is None:
        return None
    embedding_bases = []
    for base in class_record.bases:
        proof = resolve_import_reference(index, symbol.source, None, base)
        if proof is not None and proof.qualified_target in {
                "torch.nn.Embedding",
                "torch.nn.modules.sparse.Embedding"}:
            embedding_bases.append((base, proof))
    if len(embedding_bases) != 1:
        return None
    forward = type(symbol)(symbol.source, f"{symbol.qualified_name}.forward")
    callable_record = index.callable_by_symbol(forward)
    if callable_record is None:
        return None
    coordinate_params = tuple(
        item for item in callable_record.params
        if item.name != "self" and item.has_default
        and item.default is not None
        and item.default.kind == "constant"
        and item.default.const_value is None)
    if len(coordinate_params) != 1:
        return None
    coordinate_name = coordinate_params[0].name
    bindings = tuple(
        item for item in index.bindings_in(forward)
        if len(item.targets) == 1
        and item.targets[0].kind == "name"
        and item.targets[0].name == coordinate_name
        and item.value is not None)
    if not bindings or any(
            not _exact_none_default_guard(
                index, forward, coordinate_name, item.guard)
            for item in bindings):
        return None
    cumsum_calls = tuple(
        call
        for item in bindings
        for call in _calls_in(item.value)
        if _resolved_call_target(index, forward, call) == "torch.cumsum"
        and len(call.children) >= 2
        and call.children[1].kind == "name"
        and call.children[1].name != coordinate_name)
    if len(cumsum_calls) != 1:
        return None
    returns = tuple(index.return_observations_in(forward))
    if len(returns) != 1 or returns[0].guard \
            or returns[0].value is None \
            or not _direct_super_forward_of(
                returns[0].value, coordinate_name):
        return None
    base, proof = embedding_bases[0]
    spans = tuple(dict.fromkeys(
        span for span in (
            class_record.span,
            base.span,
            proof.binding.span,
            *(item.span for item in bindings),
            cumsum_calls[0].span,
            returns[0].span,
        ) if isinstance(span, SourceSpan)))
    return spans or None


def _direct_super_forward_of(expression, coordinate_name):
    if expression.kind != "call" or len(expression.children) != 2:
        return False
    callee, argument = expression.children
    if callee.kind != "attribute" or callee.name != "forward" \
            or len(callee.children) != 1:
        return False
    receiver = callee.children[0]
    if receiver.kind != "call" or not receiver.children:
        return False
    super_callee = receiver.children[0]
    return (
        super_callee.kind == "name" and super_callee.name == "super"
        and _contains_name(argument, coordinate_name)
    )


def _contains_name(expression, name):
    return (
        expression.kind == "name" and expression.name == name
    ) or any(
        _contains_name(child, name)
        for child in expression.children
        if isinstance(child, ExprNode)
    ) or any(
        _contains_name(child, name)
        for _, child in expression.keyword_children
        if isinstance(child, ExprNode)
    )


def _calls_in(expression):
    out = [expression] if expression.kind == "call" else []
    for child in expression.children:
        if isinstance(child, ExprNode):
            out.extend(_calls_in(child))
    for _, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            out.extend(_calls_in(child))
    return tuple(out)


def _resolved_call_target(index, caller, expression):
    proof = resolve_import_reference(
        index, caller.source, caller, expression.children[0])
    return proof.qualified_target if proof is not None else None


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
]
