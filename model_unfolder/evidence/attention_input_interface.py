"""Code-proven primary/context input interface for one attention implementation.

This boundary classifies an implementation that has already been addressed by
its exact :class:`~model_unfolder.evidence.program_index.SymbolId`.  It does
not select a processor, attention lane, or model role.  The initial proof is
deliberately narrow:

* one exact SDPA call supplies the semantic Q/K/V operand positions;
* three distinct, source-local producer calls reach those positions;
* Q descends from one callable formal;
* K and V descend from the same other formal; and
* an exact ``context is None -> context = primary`` binding proves the fallback.

Class, method, field, and parameter spellings never enter the decision.  A
normalization or reshape may rewrite either input local only when its expression
retains that same formal and no rival input formal.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import (
    AttentionComputeProof,
    attention_compute_positive_proof_for_symbol,
)
from .attention_storage import producer_sources_reaching_expressions
from .component_owner import OwnerOccurrenceId
from .construction_calls import resolve_import_reference
from .program_index import (
    BindingObservation,
    CallObservation,
    CallableRecord,
    ClassRecord,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class ContextFallbackAttentionInterface:
    """One exact Q-primary / shared-KV-context callable interface."""

    owner_symbol: SymbolId
    callable: CallableRecord
    compute: AttentionComputeProof
    container_class: ClassRecord | None
    container_formal: ParamRecord
    primary_formal: ParamRecord
    context_formal: ParamRecord
    query_projection: CallObservation
    key_projection: CallObservation
    value_projection: CallObservation
    fallback: BindingObservation
    preserving_bindings: tuple[BindingObservation, ...]
    auxiliary_formals: tuple[ParamRecord, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_symbol, SymbolId) \
                or not isinstance(self.callable, CallableRecord) \
                or not isinstance(self.compute, AttentionComputeProof) \
                or self.compute.child_symbol != self.owner_symbol \
                or self.compute.callable_symbol != self.callable.symbol \
                or self.callable.owner != self.owner_symbol:
            raise ValueError("an attention interface closes its exact compute owner")
        if self.container_class is not None \
                and (not isinstance(self.container_class, ClassRecord)
                     or self.container_class.symbol.source
                     != self.owner_symbol.source):
            raise ValueError("optional container class is exact and same-source")
        formals = (self.container_formal, self.primary_formal,
                   self.context_formal)
        if any(not isinstance(item, ParamRecord) for item in formals) \
                or len({item.name for item in formals}) != 3 \
                or any(item not in self.callable.params for item in formals):
            raise ValueError("container, primary and context are distinct formals")
        if self.container_class is not None \
                and not _annotation_names_symbol(
                    self.container_formal, self.container_class.symbol):
            raise ValueError("container formal annotation names the exact class")
        if not self.context_formal.has_default \
                or self.context_formal.default.kind != "constant" \
                or self.context_formal.default.const_value is not None:
            raise ValueError("the context formal has one exact None default")
        projections = (
            self.query_projection, self.key_projection, self.value_projection)
        if any(not isinstance(item, CallObservation)
               or item.enclosing_callable != self.callable.symbol
               for item in projections) \
                or len({item.span for item in projections}) != 3:
            raise ValueError("Q/K/V retain three exact producer calls")
        expected_inputs = (
            self.primary_formal.name,
            self.context_formal.name,
            self.context_formal.name,
        )
        if any(len(call.args) != 1 or call.kwargs
               or call.args[0].kind != "name"
               or call.args[0].name != expected
               for call, expected in zip(projections, expected_inputs)):
            raise ValueError("Q uses primary while K/V share the context local")
        if not isinstance(self.fallback, BindingObservation) \
                or not _is_exact_none_fallback(
                    self.fallback, self.primary_formal.name,
                    self.context_formal.name):
            raise ValueError("the interface retains its exact None fallback")
        preserving = (*self.preserving_bindings, self.fallback)
        if len(set(preserving)) != len(preserving) \
                or any(not isinstance(item, BindingObservation)
                       or item.enclosing_callable != self.callable.symbol
                       for item in self.preserving_bindings):
            raise ValueError("input-preserving bindings are exact and unique")
        if tuple(sorted(self.auxiliary_formals,
                        key=lambda item: item.name)) != self.auxiliary_formals \
                or len({item.name for item in self.auxiliary_formals}) \
                != len(self.auxiliary_formals) \
                or any(item not in self.callable.params
                       or item in formals for item in self.auxiliary_formals):
            raise ValueError("auxiliary influences are explicit exact formals")
        required = {
            *(item.span for item in projections), self.fallback.span,
            *(item.span for item in self.preserving_bindings),
            *self.compute.spans,
            *((self.container_class.span,
               self.container_formal.annotation.span)
              if self.container_class is not None else ()),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan)
                       or item.source != self.owner_symbol.source
                       for item in self.spans):
            raise ValueError("attention-interface provenance closes every edge")


def context_fallback_attention_interface_at_symbol(
        index: ProgramIndex,
        symbol: SymbolId,
        container_symbol: SymbolId | None = None,
) -> ReaderResult[ContextFallbackAttentionInterface]:
    """Prove one exact implementation's Q-primary/KV-context interface."""
    if not isinstance(index, ProgramIndex) or not isinstance(symbol, SymbolId) \
            or container_symbol is not None \
            and not isinstance(container_symbol, SymbolId):
        raise TypeError("attention interface requires ProgramIndex + SymbolId")
    if index.class_by_symbol(symbol) is None:
        return _failed(symbol, "the implementation class is absent")
    strict = attention_compute_positive_proof_for_symbol(index, symbol)
    record = index.callable_by_symbol(SymbolId(
        symbol.source, f"{symbol.qualified_name}.__call__"))
    if record is None:
        record = index.callable_by_symbol(SymbolId(
            symbol.source, f"{symbol.qualified_name}.forward"))
    if record is None:
        return _failed(symbol, "the compute callable is absent")
    if strict is not None and strict.callable_symbol != record.symbol:
        # An instance invocation enters ``__call__`` when it exists.  A proof
        # found only in a separate ``forward`` cannot certify that interface
        # until an exact wrapper argument/return binding joins the two
        # callables.  Picking either signature here would turn address
        # proximity into mechanism evidence.
        return _failed(
            symbol,
            "the positive compute proof belongs to a different entry callable")
    formals = tuple(item for item in record.params
                    if item.name != "self"
                    and item.kind not in {"vararg", "kwarg"})
    by_name = {item.name: item for item in formals}

    container_class = None
    if container_symbol is not None:
        container_class = index.class_by_symbol(container_symbol)
        if container_class is None:
            return _failed(symbol, "the exact container class is absent")
    boundary = _qkv_boundary(
        index, symbol, record, strict, container_class)
    if boundary is None:
        return _failed(symbol, "no exact SDPA or dot-softmax Q/K/V boundary")
    compute, consumer, q_expression, k_expression, v_expression = boundary

    producers = _producer_calls(index, compute.callable_symbol, formals, consumer)
    lanes = []
    lane_names = ("query", "key", "value")
    for lane_name, expression in zip(
            lane_names, (q_expression, k_expression, v_expression)):
        sources, _unpacks, _dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, compute.callable_symbol,
                ((consumer.span, (expression,)),),
                {call: call for call in producers},
                metadata_only_attributes={"shape", "dtype", "device"})
        if uncertain or len(sources) != 1:
            return _failed(
                symbol,
                f"{lane_name} lane has {len(sources)} reaching producers"
                f" (uncertain={uncertain})")
        lanes.append(next(iter(sources)))
    query, key, value = lanes
    if len({query.span, key.span, value.span}) != 3:
        return _failed(symbol, "Q/K/V do not have three distinct producers")
    inputs = tuple(call.args[0] for call in (query, key, value))
    if any(item.kind != "name" or not item.name for item in inputs) \
            or inputs[1].name != inputs[2].name \
            or inputs[0].name == inputs[1].name:
        return _failed(symbol, "producer inputs are not primary vs shared context")
    primary = by_name.get(inputs[0].name)
    context = by_name.get(inputs[1].name)
    receivers = tuple(_root_name(call.callee) for call in (query, key, value))
    if primary is None or context is None \
            or len(set(receivers)) != 1 or receivers[0] not in by_name:
        return _failed(symbol, "producer sources are not exact callable formals")
    container = by_name[receivers[0]]
    if container_class is not None and not _annotation_names_symbol(
            container, container_class.symbol):
        return _failed(symbol, "container formal annotation is not exact")
    if not context.has_default or context.default.kind != "constant" \
            or context.default.const_value is not None:
        return _failed(symbol, "context has no exact None default")

    writes = tuple(
        item for item in index.bindings_in(compute.callable_symbol)
        if (
            any(_target_is_name(target, primary.name)
                for target in item.targets)
            and _span_before(item.span, query.span)
        ) or (
            any(_target_is_name(target, context.name)
                for target in item.targets)
            and _span_before(item.span, min(
                (key.span, value.span), key=_span_key))
        ))
    fallbacks = tuple(item for item in writes
                      if _is_exact_none_fallback(
                          item, primary.name, context.name))
    if len(fallbacks) != 1:
        return _failed(symbol, "context has no unique exact None fallback")
    fallback = fallbacks[0]
    preserving = tuple(
        item for item in writes if item != fallback
        and _binding_preserves_one_input(
            item, primary.name, context.name))
    if len(preserving) != len(writes) - 1:
        return _failed(symbol, "an input local has a rival reaching definition")
    auxiliary_names = set()
    for binding in preserving:
        targets = tuple(name for target in binding.targets
                        for name in _target_names(target))
        if len(targets) != 1:
            continue
        auxiliary_names.update(
            _names(binding.value) & set(by_name)
            - {targets[0], container.name, primary.name, context.name})
    auxiliary_formals = tuple(sorted(
        (by_name[name] for name in auxiliary_names), key=lambda item: item.name))
    spans = tuple(dict.fromkeys(span for span in (
        *compute.spans, query.span, key.span, value.span, fallback.span,
        *(item.span for item in preserving),
        *((container_class.span, container.annotation.span)
          if container_class is not None else ()),
    ) if isinstance(span, SourceSpan)))
    value_out = ContextFallbackAttentionInterface(
        symbol, record, compute, container_class,
        container, primary, context, query, key, value,
        fallback, preserving, auxiliary_formals, spans)
    return ReaderResult.resolved(
        OwnerOccurrenceId(value_out.owner_symbol), value_out,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact SDPA Q/K/V producers + context-None fallback"),))


def _qkv_boundary(index, symbol, record, strict, container_class):
    if strict is not None \
            and strict.protocol == "scaled_dot_product_attention" \
            and len(strict.input_calls) == 1 \
            and len(strict.input_calls[0].args) >= 3:
        call = strict.input_calls[0]
        return strict, call, call.args[0], call.args[1], call.args[2]
    if container_class is None:
        return None
    return _legacy_dot_softmax_boundary(
        index, symbol, record, container_class)


def _legacy_dot_softmax_boundary(index, symbol, record, container_class):
    bmm_rows = tuple(
        (call, proof) for call in index.calls_in(record.symbol)
        if not call.guard and len(call.args) == 2
        and (proof := resolve_import_reference(
            index, record.symbol.source, record.symbol, call.callee)) is not None
        and proof.qualified_target == "torch.bmm")
    if len(bmm_rows) != 1 or bmm_rows[0][0].args[0].kind != "name":
        return None
    bmm, bmm_proof = bmm_rows[0]
    score_bindings = tuple(
        item for item in index.bindings_in(record.symbol)
        if not item.guard and item.value is not None
        and item.value.kind == "call"
        and any(_target_is_name(target, bmm.args[0].name)
                for target in item.targets))
    if len(score_bindings) != 1:
        return None
    score_call = tuple(
        call for call in index.calls_in(record.symbol)
        if call.span == score_bindings[0].value.span)
    if len(score_call) != 1 or len(score_call[0].args) < 2 \
            or _root_name(score_call[0].callee) is None:
        return None
    score_call = score_call[0]
    receiver = _root_name(score_call.callee)
    formals = {item.name: item for item in record.params
               if item.name != "self" and item.kind not in {"vararg", "kwarg"}}
    container_formal = formals.get(receiver)
    if container_formal is None or not _annotation_names_symbol(
            container_formal, container_class.symbol):
        return None
    leaf = score_call.callee.name
    helper = index.callable_by_symbol(SymbolId(
        container_class.symbol.source,
        f"{container_class.symbol.qualified_name}.{leaf}"))
    if helper is None \
            or (helper_spans := _score_helper_evidence(index, helper)) is None:
        return None
    params = tuple(item for item in helper.params
                   if item.name != "self"
                   and item.kind not in {"vararg", "kwarg"})
    if len(params) < 2:
        return None
    # Python positional binding is the address proof.  The helper's own source
    # independently proves that its first two formals feed dot+softmax.
    spans = tuple(dict.fromkeys(span for span in (
        score_call.span, bmm.span, bmm_proof.binding.span, *helper_spans,
    ) if isinstance(span, SourceSpan)))
    compute = AttentionComputeProof(
        symbol, record.symbol, bmm, (score_call, bmm),
        "dot_softmax", spans)
    return compute, bmm, score_call.args[0], score_call.args[1], bmm.args[1]


def _score_helper_evidence(index, helper):
    params = tuple(item for item in helper.params
                   if item.name != "self"
                   and item.kind not in {"vararg", "kwarg"})
    if len(params) < 2:
        return None
    dots = tuple(
        (call, proof) for call in index.calls_in(helper.symbol)
        if (proof := resolve_import_reference(
            index, helper.symbol.source, helper.symbol, call.callee)) is not None
        and proof.qualified_target == "torch.baddbmm"
        and len(call.args) >= 3)
    if len(dots) != 1 \
            or not _expression_descends_from(
                dots[0][0].args[1], params[0].name) \
            or not _expression_descends_from(
                dots[0][0].args[2], params[1].name):
        return None
    producer, producer_proof = dots[0]
    softmax = tuple(
        call for call in index.calls_in(helper.symbol)
        if call.callee.kind == "attribute" and call.callee.name == "softmax")
    returns = tuple(item for item in index.return_observations_in(helper.symbol)
                    if item.value is not None)
    if len(softmax) != 1 or len(returns) != 1:
        return None
    sources, _unpacks, _deps, uncertain = \
        producer_sources_reaching_expressions(
            index, helper.symbol,
            ((softmax[0].span, (softmax[0].callee.children[0],)),),
            {producer: producer})
    if uncertain or sources != {producer}:
        return None
    output_sources, _unpacks, _deps, output_uncertain = \
        producer_sources_reaching_expressions(
            index, helper.symbol,
            ((returns[0].span, (returns[0].value,)),),
            {softmax[0]: softmax[0]})
    if output_uncertain or output_sources != {softmax[0]}:
        return None
    return tuple(dict.fromkeys(span for span in (
        helper.span, producer.span, producer_proof.binding.span,
        softmax[0].span, returns[0].span,
    ) if isinstance(span, SourceSpan)))


def _annotation_names_symbol(formal, symbol):
    annotation = formal.annotation
    return annotation is not None and annotation.kind == "name" \
        and annotation.name == symbol.qualified_name \
        and symbol.source == formal.annotation.span.source


def _expression_descends_from(expression, name):
    if expression.kind == "name":
        return expression.name == name
    # Exact tensor view/cast/transpose operations may wrap a formal.  Multiple
    # children would mix an additional operand and are refused.
    children = (*expression.children,
                *(child for _key, child in expression.keyword_children))
    candidates = tuple(child for child in children
                       if _expression_descends_from(child, name))
    return len(candidates) == 1 and all(
        child in candidates or not _contains_name(child)
        for child in children)


def _contains_name(expression):
    if expression.kind == "name":
        return True
    return any(_contains_name(child) for child in expression.children) \
        or any(_contains_name(child)
               for _key, child in expression.keyword_children)


def _producer_calls(index, callable_symbol, formals, sdpa):
    formal_names = {item.name for item in formals}
    rows = []
    for call in index.calls_in(callable_symbol):
        if not _span_before(call.span, sdpa.span) \
                or call.guard or len(call.args) != 1 or call.kwargs \
                or _root_name(call.callee) not in formal_names \
                or call.args[0].kind != "name" \
                or call.args[0].name not in formal_names:
            continue
        bindings = tuple(
            item for item in index.bindings_in(callable_symbol)
            if not item.guard and item.value is not None
            and item.value.span == call.span
            and len(item.targets) == 1
            and item.targets[0].kind == "name"
            and item.targets[0].name)
        if len(bindings) == 1:
            rows.append(call)
    return tuple(rows)


def _is_exact_none_fallback(binding, primary, context):
    if not isinstance(binding, BindingObservation) \
            or len(binding.targets) != 1 \
            or not _target_is_name(binding.targets[0], context) \
            or binding.value is None or binding.value.kind != "name" \
            or binding.value.name != primary or len(binding.guard) != 1:
        return False
    step = binding.guard[0]
    test = step.test
    if step.kind != "if" or test is None or test.kind != "compare" \
            or test.operator != "is" or len(test.children) != 2:
        return False
    left, right = test.children
    return (
        left.kind == "name" and left.name == context
        and right.kind == "constant" and right.const_value is None
    ) or (
        right.kind == "name" and right.name == context
        and left.kind == "constant" and left.const_value is None
    )


def _binding_preserves_one_input(binding, primary, context):
    targets = tuple(name for target in binding.targets
                    for name in _target_names(target))
    if len(targets) != 1 or targets[0] not in {primary, context} \
            or binding.value is None:
        return False
    referenced = _names(binding.value)
    own = targets[0]
    rival = context if own == primary else primary
    return own in referenced and rival not in referenced


def _root_name(expression):
    current = expression
    while current.kind == "attribute" and current.children:
        current = current.children[0]
    return current.name if current.kind == "name" else None


def _names(expression):
    rows = {expression.name} if expression.kind == "name" and expression.name else set()
    for child in expression.children:
        rows.update(_names(child))
    for _name, child in expression.keyword_children:
        rows.update(_names(child))
    return rows


def _target_names(target):
    if target.kind == "name" and target.name:
        return (target.name,)
    return tuple(name for child in target.children
                 for name in _target_names(child))


def _target_is_name(target, name):
    return target.kind == "name" and target.name == name


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) \
        < (right.line, right.col)


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


def _failed(symbol, detail):
    return ReaderResult.failed(OwnerOccurrenceId(symbol), (ReaderFailure(
        "incomplete_graph", detail),))


__all__ = [
    "ContextFallbackAttentionInterface",
    "context_fallback_attention_interface_at_symbol",
]
