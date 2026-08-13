"""Exact linear-coordinate attention-bias evidence (ALiBi mechanism).

The mechanism is established only when two independent proofs join:

* the neutral score-additive inventory proves an enabled exact ``baddbmm``
  receiver on the score-to-softmax lane; and
* that exact receiver traces, through exact owner-call argument bindings, to a
  producer returning ``head-dependent slopes * cumulative token coordinate``.

No class, field, local, formal, helper or model spelling is a role signal.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_score_additives import (
    BaddbmmReceiverApplication,
    EquivalentAttentionScoreAdditiveInventory,
    decoder_attention_score_additives_for_path,
)
from .call_arguments import (
    CallArgumentBinding,
    bind_addressed_invocation,
    bind_repeated_child_call,
)
from .component_owner import OwnerOccurrenceId, require_resolved_component_root
from .construction_calls import resolve_import_reference
from .container_inventory import resolve_container_inventory
from .decoder_block import decoder_block_candidates_for_config
from .execution_flow import resolve_addressed_invocations
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    CallObservation,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


_POW_PROTOCOLS = frozenset({"torch.pow"})
_ARANGE_PROTOCOLS = frozenset({"torch.arange"})
_CAT_PROTOCOLS = frozenset({"torch.cat"})


@dataclass(frozen=True)
class LinearCoordinateBiasProducer:
    """One exact slope-times-cumulative-coordinate producer callable."""

    callable_symbol: SymbolId
    coordinate_formal: ParamRecord
    head_count_formal: ParamRecord
    coordinate_binding: BindingObservation
    slope_bindings: tuple[BindingObservation, ...]
    product_binding: BindingObservation
    returned: ReturnObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.callable_symbol, SymbolId):
            raise TypeError("a linear-bias producer has an exact callable")
        if not isinstance(self.coordinate_formal, ParamRecord) \
                or not isinstance(self.head_count_formal, ParamRecord) \
                or self.coordinate_formal == self.head_count_formal:
            raise ValueError("coordinate and head-count formals are distinct")
        if not isinstance(self.coordinate_binding, BindingObservation) \
                or self.coordinate_binding.enclosing_callable \
                != self.callable_symbol or self.coordinate_binding.guard:
            raise ValueError("the coordinate binding is exact and unconditional")
        if not self.slope_bindings or any(
                not isinstance(item, BindingObservation)
                or item.enclosing_callable != self.callable_symbol
                for item in self.slope_bindings):
            raise ValueError("slope bindings belong to the producer callable")
        if not isinstance(self.product_binding, BindingObservation) \
                or self.product_binding.enclosing_callable != self.callable_symbol \
                or self.product_binding.guard:
            raise ValueError("the final slope-coordinate product is unconditional")
        if not isinstance(self.returned, ReturnObservation) \
                or self.returned.enclosing_callable != self.callable_symbol \
                or self.returned.guard or self.returned.value is None:
            raise ValueError("the producer has one exact unconditional return")
        if tuple(sorted(self.slope_bindings,
                        key=lambda item: _span_key(item.span))) \
                != self.slope_bindings \
                or any(_span_key(item.span)
                       >= _span_key(self.product_binding.span)
                       for item in (*self.slope_bindings,
                                    self.coordinate_binding)):
            raise ValueError(
                "producer inputs retain source order before their product")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("producer evidence carries exact source spans")
        required = {
            item.span for item in (*self.slope_bindings,
                                   self.coordinate_binding,
                                   self.product_binding)
        } | {self.returned.span}
        if None in required or not required <= set(self.spans):
            raise ValueError("producer provenance cites every decisive binding")


@dataclass(frozen=True)
class AlibiScoreBiasEvidence:
    """End-to-end exact producer-to-score ALiBi evidence."""

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    application: BaddbmmReceiverApplication
    attention_argument: CallArgumentBinding
    block_argument: CallArgumentBinding
    stage_definition: BindingObservation
    stage_producer_call: CallObservation
    wrapper_callable: SymbolId
    wrapper_return: ReturnObservation
    producer: LinearCoordinateBiasProducer
    transport_spans: tuple[SourceSpan, ...]
    kind: str = "alibi"
    application_kind: str = "score_bias"

    def __post_init__(self) -> None:
        if self.kind != "alibi" or self.application_kind != "score_bias":
            raise ValueError("the evidence kind is closed")
        if self.application.block_occurrence != self.block_occurrence \
                or self.application.attention_occurrence \
                != self.attention_occurrence:
            raise ValueError("application and evidence name one exact lane")
        if self.attention_argument.callee_occurrence \
                != self.attention_occurrence \
                or self.attention_argument.formal.name \
                != self.application.bias_operand.name:
            raise ValueError("the exact bias formal is bound at the attention call")
        if self.block_argument.callee_occurrence != self.block_occurrence:
            raise ValueError("the block argument belongs to the exact repeated child")
        if not isinstance(self.stage_definition, BindingObservation) \
                or self.stage_definition.value is None \
                or not _expr_contains_span(
                    self.stage_definition.value, self.stage_producer_call.span):
            raise ValueError("the stage definition is authored by the producer call")
        if not isinstance(self.wrapper_callable, SymbolId) \
                or not isinstance(self.wrapper_return, ReturnObservation) \
                or self.wrapper_return.enclosing_callable != self.wrapper_callable:
            raise ValueError("the transparent wrapper return is exact")
        if not self.transport_spans or any(not isinstance(span, SourceSpan)
                                           for span in self.transport_spans):
            raise ValueError("transport evidence carries exact source spans")
        required = {
            self.application.score_call.span,
            self.attention_argument.call.span,
            self.block_argument.call.span,
            self.stage_definition.span,
            self.stage_producer_call.span,
            self.wrapper_return.span,
        }
        if None in required or not required <= set(self.transport_spans):
            raise ValueError("transport provenance cites every owner hop")


def decoder_alibi_score_bias_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[AlibiScoreBiasEvidence]:
    """Prove ALiBi only when the exact producer reaches the exact score lane."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("ALiBi evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("ALiBi evidence requires a SourceBundle")
    candidates = decoder_block_candidates_for_config(
        index, bundle, tuple(config_path),
        allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return candidates
    additives = decoder_attention_score_additives_for_path(
        index, bundle, tuple(config_path),
        allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if additives.status != "resolved":
        return additives
    inventories = (
        additives.value.variants
        if isinstance(additives.value,
                      EquivalentAttentionScoreAdditiveInventory)
        else (additives.value,)
    )
    applications = tuple(
        (inventory, application)
        for inventory in inventories
        for application in inventory.applications
        if isinstance(application, BaddbmmReceiverApplication))
    if not applications:
        return ReaderResult.absent(
            candidates.value.stage_occurrence,
            provenance=additives.provenance)

    outcomes = tuple(_trace_application(
        index, candidates.value.component_root, candidates.value,
        inventory, application)
        for inventory, application in applications)
    if len(outcomes) == 1:
        return outcomes[0]
    if all(item.status == "resolved" for item in outcomes):
        identities = {
            (item.value.producer.callable_symbol,
             item.value.producer.product_binding.span)
            for item in outcomes}
        if len(identities) == 1:
            # More than one exact attention lane applying the same producer is
            # intentionally not collapsed into one lane-specific DTO.
            return ReaderResult.ambiguous(
                candidates.value.stage_occurrence,
                Ambiguity(sites=tuple(
                    item.value.application.score_call.span
                    for item in outcomes)),
                provenance=tuple(origin for item in outcomes
                                 for origin in item.provenance))
    return next((item for item in outcomes if item.status != "resolved"),
                ReaderResult.ambiguous(
                    candidates.value.stage_occurrence,
                    Ambiguity(sites=tuple(
                        app.score_call.span for _inventory, app in applications))))


def _trace_application(index, root, candidates, inventory, application):
    root = require_resolved_component_root(root, caller="_trace_application")
    block = application.block_occurrence
    block_inventory = resolve_container_inventory(index, root, block)
    invocations = resolve_addressed_invocations(
        index, root, block, block_inventory)
    if invocations.status != "resolved":
        return _failure(block, "incomplete_graph",
                        "block invocation census is unavailable")
    incoming = tuple(item for item in invocations.addressed
                     if item.callee_owner_occurrence
                     == application.attention_occurrence)
    if len(incoming) != 1:
        return _failure(block, "conflict",
                        "the exact attention occurrence has rival incoming calls")
    attention_call = bind_addressed_invocation(index, root, incoming[0])
    if attention_call.status not in {"resolved", "partial"} \
            or application.bias_operand.kind != "name":
        return _failure(block, "incomplete_graph",
                        "the exact baddbmm receiver formal is not bindable")
    attention_argument = attention_call.for_formal(
        application.bias_operand.name)
    if attention_argument is None:
        return _failure(block, "incomplete_graph",
                        "the baddbmm receiver has no exact incoming argument")
    block_formal = _formal_origin(
        index, attention_argument.call.enclosing_callable,
        attention_argument.actual, attention_argument.call.span)
    if block_formal is None:
        return _failure(block, "incomplete_graph",
                        "the attention bias actual has no unique block formal")

    proofs = tuple(proof for proof in candidates.repeated_child.proofs
                   if proof.child_occurrence == block)
    if len(proofs) != 1:
        return _failure(block, "conflict",
                        "the block occurrence has no unique repeated-call proof")
    block_call = bind_repeated_child_call(index, root, proofs[0])
    block_argument = block_call.for_formal(block_formal.name)
    if block_argument is None:
        return _failure(block, "incomplete_graph",
                        "the block bias formal has no exact stage argument")
    stage_definition, producer_call = _producer_definition(
        index, block_argument.call.enclosing_callable,
        block_argument.actual, block_argument.call.span)
    if stage_definition is None or producer_call is None:
        return _failure(block, "incomplete_graph",
                        "the stage bias actual has no unique producer definition")

    stage_node = root.graph.node_for(candidates.stage_occurrence)
    method = _self_field(producer_call.callee)
    # Deliberately direct-only.  The older first-base method helper is not an
    # exact Python-MRO proof; inherited wrappers remain unknown until the B1
    # precedence boundary exposes a reusable method-resolution contract.
    wrapper = (SymbolId(
        stage_node.symbol.source,
        f"{stage_node.symbol.qualified_name}.{method}")
        if stage_node is not None and method else None)
    if wrapper is not None and index.callable_by_symbol(wrapper) is None:
        wrapper = None
    if wrapper is None:
        return _failure(block, "unresolved_import",
                        "the stage producer method is not exactly indexed")
    wrapper_returns = tuple(item for item in index.return_observations_in(wrapper)
                            if not item.guard and item.value is not None)
    if len(wrapper_returns) != 1:
        return _failure(block, "conflict",
                        "the producer wrapper has no unique unconditional return")
    wrapper_return = wrapper_returns[0]
    helper_call = _call_expression_observation(
        index, wrapper, wrapper_return.value)
    if helper_call is None or helper_call.callee.kind != "name":
        return _failure(block, "unsupported_syntax",
                        "the producer wrapper is not one direct local helper call")
    helper = SymbolId(wrapper.source, helper_call.callee.name)
    helper_record = index.callable_by_symbol(helper)
    if helper_record is None or helper_record.owner is not None:
        return _failure(block, "unresolved_import",
                        "the bias producer helper is not an exact local function")
    producer = _classify_linear_coordinate_producer(index, helper)
    if producer is None:
        return _failure(block, "unsupported_syntax",
                        "the exact producer is not slope times cumulative coordinate")

    spans = tuple(dict.fromkeys((
        application.score_call.span,
        attention_argument.call.span,
        attention_argument.actual.span,
        block_argument.call.span,
        block_argument.actual.span,
        stage_definition.span,
        producer_call.span,
        wrapper_return.span,
        *producer.spans,
    )))
    value = AlibiScoreBiasEvidence(
        block, application.attention_occurrence, application,
        attention_argument, block_argument, stage_definition, producer_call,
        wrapper, wrapper_return, producer, spans)
    return ReaderResult.resolved(
        block, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "exact slope-times-cumulative-coordinate producer reaches "
                "the exact enabled baddbmm score receiver")),))


def _classify_linear_coordinate_producer(index, callable_symbol):
    record = index.callable_by_symbol(callable_symbol)
    if record is None or record.owner is not None:
        return None
    params = tuple(item for item in record.params
                   if item.kind not in {"vararg", "kwarg"})
    bindings = tuple(sorted(
        (item for item in index.bindings_in(callable_symbol)
         if item.span is not None and item.value is not None),
        key=lambda item: _span_key(item.span)))
    returns = tuple(item for item in index.return_observations_in(callable_symbol)
                    if not item.guard and item.value is not None)
    if len(returns) != 1:
        return None
    returned = returns[0]
    product_candidates = []
    for product in bindings:
        target = _single_target(product)
        if product.guard or target is None or product.value.kind != "binop" \
                or product.value.operator != "*" \
                or len(product.value.children) != 2 \
                or not _transparent_from_name(returned.value, target):
            continue
        left, right = product.value.children
        for slope_expr, coordinate_expr in ((left, right), (right, left)):
            slope_name = _base_name(slope_expr)
            coordinate_name = _base_name(coordinate_expr)
            if slope_name is None or coordinate_name is None:
                continue
            coordinate_writes = tuple(
                item for item in bindings if _single_target(item) == coordinate_name
                and _span_key(item.span) < _span_key(product.span))
            slope_writes = tuple(
                item for item in bindings if _single_target(item) == slope_name
                and _span_key(item.span) < _span_key(product.span))
            if len(coordinate_writes) != 1 or not slope_writes:
                continue
            coordinate_formal = _masked_cumulative_formal(
                index, callable_symbol, coordinate_writes[0].value, params)
            if coordinate_formal is None \
                    or not _slope_writes_are_closed(
                        index, callable_symbol, slope_name, slope_writes):
                continue
            dependencies = _arange_count_formals(
                index, callable_symbol, slope_writes[0].value,
                slope_writes[0].span, params, frozenset())
            head_formals = tuple(item for item in params
                                 if item != coordinate_formal
                                 and item.name in dependencies)
            if len(head_formals) != 1:
                continue
            product_candidates.append((
                coordinate_formal, head_formals[0], coordinate_writes[0],
                slope_writes, product))
    if len(product_candidates) != 1:
        return None
    coordinate_formal, head_formal, coordinate, slopes, product = \
        product_candidates[0]
    spans = tuple(dict.fromkeys((
        *(item.span for item in slopes), coordinate.span,
        product.span, returned.span)))
    return LinearCoordinateBiasProducer(
        callable_symbol, coordinate_formal, head_formal,
        coordinate, slopes, product, returned, spans)


def _masked_cumulative_formal(index, callable_symbol, expression, params):
    expression = _strip_subscripts(expression)
    if expression.kind != "binop" or expression.operator != "*" \
            or len(expression.children) != 2:
        return None
    for cumulative, mask in (
            (expression.children[0], expression.children[1]),
            (expression.children[1], expression.children[0])):
        if mask.kind != "name":
            continue
        formal = next((item for item in params if item.name == mask.name), None)
        if formal is None or cumulative.kind != "binop" \
                or cumulative.operator != "-" \
                or len(cumulative.children) != 2 \
                or cumulative.children[1].kind != "constant" \
                or cumulative.children[1].const_value != 1:
            continue
        call_expr = cumulative.children[0]
        call = _call_expression_observation(
            index, callable_symbol, call_expr)
        if call is None or call.callee.kind != "attribute" \
                or call.callee.name != "cumsum" \
                or len(call.callee.children) != 1 \
                or call.callee.children[0].kind != "name" \
                or call.callee.children[0].name != mask.name \
                or call.args or dict(call.kwargs).get("dim") is None:
            continue
        dim = dict(call.kwargs)["dim"]
        if dim.kind == "unaryop" and dim.operator == "-" \
                and len(dim.children) == 1 \
                and dim.children[0].kind == "constant" \
                and dim.children[0].const_value == 1:
            return formal
    return None


def _slope_writes_are_closed(index, callable_symbol, name, writes):
    first = writes[0]
    if first.guard or not _is_protocol_call(
            index, callable_symbol, first.value, _POW_PROTOCOLS) \
            or not _contains_protocol_through_bindings(
                index, callable_symbol, first.value, first.span,
                _ARANGE_PROTOCOLS, frozenset()):
        return False
    for item in writes[1:]:
        if not item.guard or not _is_protocol_call(
                index, callable_symbol, item.value, _CAT_PROTOCOLS) \
                or not _expr_contains_name(item.value, name) \
                or not _contains_protocol(
                    index, callable_symbol, item.value, _POW_PROTOCOLS):
            return False
    return True


def _contains_protocol_through_bindings(
        index, callable_symbol, expression, before, protocols, seen):
    if _contains_protocol(index, callable_symbol, expression, protocols):
        return True
    for name in _names_in(expression):
        key = (name, before)
        if key in seen:
            continue
        writes = tuple(item for item in index.bindings_in(callable_symbol)
                       if item.span is not None and item.value is not None
                       and _span_key(item.span) < _span_key(before)
                       and _single_target(item) == name and not item.guard)
        if len(writes) == 1 and _contains_protocol_through_bindings(
                index, callable_symbol, writes[0].value, writes[0].span,
                protocols, seen | {key}):
            return True
    return False


def _arange_count_formals(
        index, callable_symbol, expression, before, params, seen):
    """Formals controlling exact arange start/end/step, not device/dtype."""
    call = (_call_expression_observation(index, callable_symbol, expression)
            if expression.kind == "call" else None)
    proof = (resolve_import_reference(
        index, callable_symbol.source, callable_symbol, call.callee)
        if call is not None else None)
    if proof is not None and proof.qualified_target in _ARANGE_PROTOCOLS:
        count_expressions = tuple(call.args) + tuple(
            value for name, value in call.kwargs
            if name in {"start", "end", "step"})
        return frozenset(
            name for item in count_expressions
            for name in _formal_dependencies(
                index, callable_symbol, item, before, params, seen))
    found = set()
    for name in _names_in(expression):
        key = (name, before)
        if key in seen:
            continue
        writes = tuple(item for item in index.bindings_in(callable_symbol)
                       if item.span is not None and item.value is not None
                       and _span_key(item.span) < _span_key(before)
                       and _single_target(item) == name and not item.guard)
        if len(writes) == 1:
            found.update(_arange_count_formals(
                index, callable_symbol, writes[0].value, writes[0].span,
                params, seen | {key}))
    for child in expression.children:
        found.update(_arange_count_formals(
            index, callable_symbol, child, before, params, seen))
    for _name, child in expression.keyword_children:
        found.update(_arange_count_formals(
            index, callable_symbol, child, before, params, seen))
    return frozenset(found)


def _formal_dependencies(
        index, callable_symbol, expression, before, params, seen):
    param_names = {item.name for item in params}
    found = set(name for name in _names_in(expression) if name in param_names)
    for name in _names_in(expression):
        key = (name, before)
        if name in param_names or key in seen:
            continue
        writes = tuple(item for item in index.bindings_in(callable_symbol)
                       if item.span is not None and item.value is not None
                       and _span_key(item.span) < _span_key(before)
                       and _single_target(item) == name and not item.guard)
        if len(writes) == 1:
            found.update(_formal_dependencies(
                index, callable_symbol, writes[0].value, writes[0].span,
                params, seen | {key}))
    return frozenset(found)


def _formal_origin(index, callable_symbol, expression, before, seen=frozenset()):
    if expression.kind != "name" or not expression.name:
        return None
    record = index.callable_by_symbol(callable_symbol)
    if record is None:
        return None
    writes = tuple(item for item in index.bindings_in(callable_symbol)
                   if item.span is not None and item.value is not None
                   and _span_key(item.span) < _span_key(before)
                   and _single_target(item) == expression.name)
    if not writes:
        return next((item for item in record.params
                     if item.name == expression.name), None)
    latest = writes[-1]
    if latest.guard or expression.name in seen:
        return None
    return _formal_origin(
        index, callable_symbol, latest.value, latest.span,
        seen | {expression.name})


def _producer_definition(index, callable_symbol, expression, before):
    if expression.kind != "name":
        return None, None
    writes = tuple(item for item in index.bindings_in(callable_symbol)
                   if item.span is not None and item.value is not None
                   and _span_key(item.span) < _span_key(before)
                   and _single_target(item) == expression.name)
    if len(writes) != 1 or writes[0].guard:
        return None, None
    call = _call_expression_observation(
        index, callable_symbol, writes[0].value)
    return (writes[0], call) if call is not None else (None, None)


def _call_expression_observation(index, callable_symbol, expression):
    if expression is None or expression.kind != "call" \
            or expression.span is None:
        return None
    matches = tuple(item for item in index.calls_in(callable_symbol)
                    if item.span == expression.span)
    return matches[0] if len(matches) == 1 else None


def _is_protocol_call(index, callable_symbol, expression, protocols):
    call = _call_expression_observation(index, callable_symbol, expression)
    if call is None:
        return False
    proof = resolve_import_reference(
        index, callable_symbol.source, callable_symbol, call.callee)
    return proof is not None and proof.qualified_target in protocols


def _contains_protocol(index, callable_symbol, expression, protocols):
    if expression is None:
        return False
    if expression.kind == "call" and _is_protocol_call(
            index, callable_symbol, expression, protocols):
        return True
    return any(_contains_protocol(index, callable_symbol, child, protocols)
               for child in expression.children) or any(
        _contains_protocol(index, callable_symbol, child, protocols)
        for _name, child in expression.keyword_children)


def _strip_subscripts(expression):
    while expression.kind == "subscript" and expression.children:
        expression = expression.children[0]
    return expression


def _base_name(expression):
    expression = _strip_subscripts(expression)
    return expression.name if expression.kind == "name" else None


def _transparent_from_name(expression, name):
    if expression.kind == "name":
        return expression.name == name
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        return callee.kind == "attribute" and callee.children \
            and _transparent_from_name(callee.children[0], name)
    if expression.kind == "attribute" and expression.children:
        return _transparent_from_name(expression.children[0], name)
    return False


def _single_target(binding):
    names = tuple(name for target in binding.targets
                  for name in _target_names(target))
    return names[0] if len(names) == 1 else None


def _target_names(expression):
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     for name in _target_names(child))
    return ()


def _names_in(expression):
    if not isinstance(expression, ExprNode):
        return frozenset()
    out = {expression.name} if expression.kind == "name" \
        and expression.name else set()
    for child in expression.children:
        out.update(_names_in(child))
    for _name, child in expression.keyword_children:
        out.update(_names_in(child))
    return frozenset(out)


def _expr_contains_name(expression, name):
    return name in _names_in(expression)


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" \
        else None


def _expr_contains_span(expression, span):
    if not isinstance(expression, ExprNode) or span is None:
        return False
    if expression.span == span:
        return True
    return any(_expr_contains_span(child, span)
               for child in expression.children) or any(
        _expr_contains_span(child, span)
        for _name, child in expression.keyword_children)


def _span_key(span):
    if span is None:
        return ("", "", -1, -1, -1, -1)
    return (
        span.source.component_key, span.source.canonical_path,
        span.line, span.col, span.end_line or span.line,
        span.end_col or span.col)


def _failure(owner, kind, detail):
    return ReaderResult.failed(
        owner, (ReaderFailure(kind, detail),))


__all__ = [
    "LinearCoordinateBiasProducer",
    "AlibiScoreBiasEvidence",
    "decoder_alibi_score_bias_for_path",
]
