"""Exact learned relative-coordinate attention-bias evidence.

This reader proves two deliberately separate claims:

* an exact score-side additive operand is produced by
  ``relative coordinates -> bucket callable -> learned Embedding``; and
* an exact construction-index expression selects which repeated attention
  occurrence owns that Embedding table.

It does not infer a mechanism from class, method, field, local, model, or config
spellings.  It also does not claim that later layers reuse the first layer's
returned bias: loop-carried sharing is a separate execution-flow fact.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_score_additives import (
    EquivalentAttentionScoreAdditiveInventory,
    ExplicitAttentionScoreAdditiveApplication,
    decoder_attention_score_additives_for_path,
)
from .component_owner import OwnerOccurrenceId
from .construction_arguments import (
    ConstructionArgumentBinding,
    bind_construction_site,
)
from .construction_calls import ExternalReferenceProof, resolve_import_reference
from .decoder_block import decoder_block_candidates_for_config
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    CallObservation,
    ComprehensionObservation,
    ConstructionSite,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult


_EMBEDDING_PROTOCOLS = frozenset({
    "torch.nn.Embedding",
    "torch.nn.modules.sparse.Embedding",
})
_ARANGE_PROTOCOLS = frozenset({"torch.arange"})
_ABS_PROTOCOLS = frozenset({"torch.abs"})
_LOG_PROTOCOLS = frozenset({"torch.log"})
_WHERE_PROTOCOLS = frozenset({"torch.where"})


@dataclass(frozen=True)
class RelativeBucketProducer:
    """One exact relative-coordinate bucket and learned-table producer."""

    attention_occurrence: OwnerOccurrenceId
    forward_callable: SymbolId
    additive_application: ExplicitAttentionScoreAdditiveApplication
    additive_alias: BindingObservation
    compute_binding: BindingObservation
    compute_call: CallObservation
    compute_callable: SymbolId
    coordinate_calls: tuple[CallObservation, CallObservation]
    relative_binding: BindingObservation
    bucket_binding: BindingObservation
    bucket_call: CallObservation
    bucket_callable: SymbolId
    lookup_binding: BindingObservation
    lookup_call: CallObservation
    embedding_site: ConstructionSite
    embedding_primitive: ExternalReferenceProof
    returned: ReturnObservation
    flag_formal: ParamRecord
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.attention_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.forward_callable, SymbolId) \
                or not isinstance(self.compute_callable, SymbolId) \
                or not isinstance(self.bucket_callable, SymbolId):
            raise TypeError("relative-bias evidence carries exact addresses")
        if self.additive_application.application.enclosing_callable \
                != self.forward_callable:
            raise ValueError("relative-bias producer reaches the exact score lane")
        bindings = (
            self.additive_alias, self.compute_binding, self.relative_binding,
            self.bucket_binding, self.lookup_binding,
        )
        if any(not isinstance(item, BindingObservation) or item.value is None
               for item in bindings):
            raise TypeError("relative-bias evidence carries exact definitions")
        calls = (*self.coordinate_calls, self.compute_call,
                 self.bucket_call, self.lookup_call)
        if any(not isinstance(item, CallObservation) or item.span is None
               for item in calls):
            raise TypeError("relative-bias evidence carries exact calls")
        if self.compute_call.enclosing_callable != self.forward_callable \
                or self.compute_binding.enclosing_callable != self.forward_callable \
                or self.additive_alias.enclosing_callable != self.forward_callable:
            raise ValueError("score application and compute call share one forward")
        if any(item.enclosing_callable != self.compute_callable
               for item in (*self.coordinate_calls, self.relative_binding,
                            self.bucket_binding, self.bucket_call,
                            self.lookup_binding, self.lookup_call, self.returned)):
            raise ValueError("coordinate, bucket and lookup share one producer")
        attention_symbol = SymbolId(
            self.forward_callable.source,
            self.forward_callable.qualified_name.rsplit(".", 1)[0])
        if not isinstance(self.embedding_site, ConstructionSite) \
                or self.embedding_site.owner != attention_symbol:
            raise ValueError("the learned table is constructed by the attention class")
        if not isinstance(self.embedding_primitive, ExternalReferenceProof) \
                or self.embedding_primitive.qualified_target \
                not in _EMBEDDING_PROTOCOLS:
            raise ValueError("relative bias uses an exact learned embedding primitive")
        if not isinstance(self.returned, ReturnObservation) \
                or self.returned.value is None or self.returned.guard:
            raise ValueError("relative bias has one exact unconditional producer return")
        if not isinstance(self.flag_formal, ParamRecord) \
                or self.flag_formal.kind in {"vararg", "kwarg"}:
            raise TypeError("table ownership cites an exact constructor formal")
        required = {
            *(item.span for item in bindings),
            *(item.span for item in calls),
            self.embedding_site.span,
            self.embedding_primitive.binding.span,
            self.returned.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("relative-bias provenance cites every decisive boundary")


@dataclass(frozen=True)
class RelativeBiasOwnershipSchedule:
    attention_occurrence: OwnerOccurrenceId
    block_site: ConstructionSite
    comprehension: ComprehensionObservation
    transport: tuple[ConstructionArgumentBinding, ...]
    selector_expression: ExprNode
    index_parameter: str
    owner_index: int
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.attention_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.block_site, ConstructionSite) \
                or not isinstance(self.comprehension, ComprehensionObservation):
            raise TypeError("ownership schedule carries exact construction syntax")
        if not self.transport or any(
                not isinstance(item, ConstructionArgumentBinding)
                for item in self.transport):
            raise ValueError("ownership schedule carries every constructor hop")
        if not isinstance(self.selector_expression, ExprNode) \
                or not self.index_parameter:
            raise TypeError("ownership schedule carries an exact index predicate")
        if self.transport[0].site != self.block_site \
                or self.transport[0].actual != self.selector_expression \
                or self.transport[-1].child_occurrence \
                != self.attention_occurrence:
            raise ValueError("ownership transport closes block to exact attention")
        for previous, current in zip(self.transport, self.transport[1:]):
            if previous.child_occurrence != current.parent_occurrence \
                    or current.actual.kind != "name" \
                    or current.actual.name != previous.formal.name:
                raise ValueError("ownership flag crosses each constructor transparently")
        if isinstance(self.owner_index, bool) or self.owner_index != 0:
            raise ValueError("the proved ownership predicate selects index zero")
        required = {
            self.block_site.span, self.comprehension.span,
            self.selector_expression.span,
            *(span for item in self.transport for span in item.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("ownership provenance closes the index transport")


@dataclass(frozen=True)
class RelativePositionBiasEvidence:
    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    producer: RelativeBucketProducer
    ownership: RelativeBiasOwnershipSchedule
    spans: tuple[SourceSpan, ...]
    kind: str = "relative_bias"
    application: str = "attention_score_additive"

    def __post_init__(self) -> None:
        if self.kind != "relative_bias" \
                or self.application != "attention_score_additive":
            raise ValueError("relative-position evidence has a closed kind")
        if self.producer.attention_occurrence != self.attention_occurrence \
                or self.ownership.attention_occurrence \
                != self.attention_occurrence \
                or self.attention_occurrence.root != self.block_occurrence.root \
                or self.attention_occurrence.sites[:len(
                    self.block_occurrence.sites)] != self.block_occurrence.sites:
            raise ValueError("producer and score application share one exact lane")
        required = {*self.producer.spans, *self.ownership.spans}
        if not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("relative-position provenance closes both proofs")


def decoder_relative_position_bias_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[RelativePositionBiasEvidence]:
    """Prove an exact learned relative bucket producer and table-owner schedule."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("relative-position evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("relative-position evidence requires a SourceBundle")
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
    outcomes = []
    for inventory in inventories:
        for application in inventory.applications:
            if not isinstance(application,
                              ExplicitAttentionScoreAdditiveApplication):
                continue
            producer = _relative_producer(
                index, candidates.value, inventory.attention_occurrence,
                application)
            if producer is None:
                continue
            ownership = _ownership_schedule(
                index, candidates.value, inventory.block_occurrence,
                producer.attention_occurrence,
                producer.flag_formal)
            if isinstance(ownership, ReaderFailure):
                return ReaderResult.failed(
                    candidates.value.stage_occurrence, (ownership,))
            spans = tuple(dict.fromkeys((*producer.spans, *ownership.spans)))
            outcomes.append(RelativePositionBiasEvidence(
                inventory.block_occurrence,
                inventory.attention_occurrence,
                producer, ownership, spans))
    if not outcomes:
        return ReaderResult.absent(candidates.value.stage_occurrence)
    identities = {
        (item.attention_occurrence, item.producer.compute_call.span,
         item.producer.embedding_site.span,
         item.ownership.selector_expression.span): item
        for item in outcomes}
    if len(identities) != 1:
        return ReaderResult.ambiguous(
            candidates.value.stage_occurrence,
            Ambiguity(sites=tuple(sorted(
                (item.producer.compute_call.span for item in outcomes),
                key=_span_key))))
    value = next(iter(identities.values()))
    return ReaderResult.resolved(
        value.block_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=value.spans,
            detail=(
                "exact relative coordinates -> exact bucket callable -> "
                "learned embedding -> exact score additive; exact constructor "
                "index proves first-layer table ownership only")),))


def _relative_producer(index, candidates, attention_occurrence, application):
    root = candidates.component_root
    node = root.graph.node_for(attention_occurrence)
    if node is None:
        return None
    forward = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.forward")
    record = index.callable_by_symbol(forward)
    if record is None or application.additive_operand.kind != "name":
        return None
    bindings = tuple(index.bindings_in(forward))
    alias = tuple(item for item in bindings
                  if not item.guard and _single_target(item)
                  == application.additive_operand.name
                  and item.value is not None and item.value.kind == "name")
    if len(alias) != 1:
        return None
    source_name = alias[0].value.name
    if not any(item.name == source_name for item in record.params):
        return None
    compute = tuple(item for item in bindings
                    if _single_target(item) == source_name
                    and item.value is not None and item.value.kind == "call"
                    and _self_field(item.value.children[0]) is not None
                    and _guard_has_none_check(item.guard, source_name))
    if len(compute) != 1:
        return None
    compute_binding = compute[0]
    compute_call = _call_at(index, forward, compute_binding.value.span)
    method_name = _self_field(compute_binding.value.children[0])
    if compute_call is None or method_name is None:
        return None
    compute_callable = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.{method_name}")
    compute_record = index.callable_by_symbol(compute_callable)
    if compute_record is None or compute_record.owner != node.symbol:
        return None

    classified = _classify_compute(index, node.symbol, compute_callable)
    if classified is None:
        return None
    (coordinate_calls, relative_binding, bucket_binding, bucket_call,
     bucket_callable, lookup_binding, lookup_call, returned,
     embedding_field) = classified

    sites = tuple(item for item in index.construction_sites_of(node.symbol)
                  if item.target == embedding_field
                  and item.target_kind == "field")
    if len(sites) != 1:
        return None
    embedding_site = sites[0]
    primitive = resolve_import_reference(
        index, embedding_site.owner.source,
        embedding_site.enclosing_callable,
        embedding_site.constructor.children[0])
    if primitive is None or primitive.qualified_target \
            not in _EMBEDDING_PROTOCOLS:
        return None
    flag_field = _single_positive_self_guard(embedding_site.guard)
    if flag_field is None:
        return None
    init = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.__init__")
    flag_bindings = tuple(item for item in index.bindings_in(init)
                          if _self_target(item) == flag_field
                          and not item.guard and item.value is not None
                          and item.value.kind == "name")
    if len(flag_bindings) != 1:
        return None
    flag_formal = next((item for item in index.callable_by_symbol(init).params
                        if item.name == flag_bindings[0].value.name), None)
    if flag_formal is None:
        return None
    if not _compute_is_else_of_disabled_flag(
            bindings, compute_binding, source_name, flag_field):
        return None

    spans = tuple(dict.fromkeys((
        application.application.span, alias[0].span,
        compute_binding.span, compute_call.span,
        *(item.span for item in coordinate_calls),
        relative_binding.span, bucket_binding.span, bucket_call.span,
        lookup_binding.span, lookup_call.span,
        embedding_site.span, primitive.binding.span, returned.span,
        flag_bindings[0].span,
    )))
    return RelativeBucketProducer(
        attention_occurrence, forward, application, alias[0],
        compute_binding, compute_call, compute_callable, coordinate_calls,
        relative_binding, bucket_binding, bucket_call, bucket_callable,
        lookup_binding, lookup_call, embedding_site, primitive, returned,
        flag_formal, spans)


def _classify_compute(index, owner, callable_symbol):
    record = index.callable_by_symbol(callable_symbol)
    params = tuple(item for item in record.params
                   if item.kind not in {"vararg", "kwarg"})
    if len(params) < 3:
        return None
    query_name, key_name = params[1].name, params[2].name
    bindings = tuple(index.bindings_in(callable_symbol))
    calls = tuple(index.calls_in(callable_symbol))
    aranges = tuple(call for call in calls
                    if _external_call_target(index, call) in _ARANGE_PROTOCOLS
                    and len(call.args) == 1 and call.args[0].kind == "name"
                    and call.args[0].name in {query_name, key_name})
    if len(aranges) != 2 or {item.args[0].name for item in aranges} \
            != {query_name, key_name}:
        return None
    coordinate_names = {}
    for binding in bindings:
        target = _single_target(binding)
        if target is None or binding.value is None or binding.guard:
            continue
        for call in aranges:
            if _expr_contains_span(binding.value, call.span):
                coordinate_names[call.args[0].name] = target
    if set(coordinate_names) != {query_name, key_name}:
        return None
    if len(set(coordinate_names.values())) != 2:
        return None
    relative = tuple(item for item in bindings
                     if not item.guard and item.value is not None
                     and item.value.kind == "binop"
                     and item.value.operator == "-"
                     and tuple(_base_name(child)
                               for child in item.value.children)
                     == (coordinate_names[key_name], coordinate_names[query_name]))
    if len(relative) != 1 or _single_target(relative[0]) is None:
        return None
    relative_name = _single_target(relative[0])
    bucket_defs = []
    for binding in bindings:
        target = _single_target(binding)
        if target is None or binding.guard or binding.value is None \
                or binding.value.kind != "call":
            continue
        call = _call_at(index, callable_symbol, binding.value.span)
        method = _self_field(binding.value.children[0])
        if call is None or method is None or not call.args \
                or _base_name(call.args[0]) != relative_name:
            continue
        bucket_callable = SymbolId(
            owner.source, f"{owner.qualified_name}.{method}")
        if _classify_bucket(index, bucket_callable):
            bucket_defs.append((binding, call, bucket_callable))
    if len(bucket_defs) != 1:
        return None
    bucket_binding, bucket_call, bucket_callable = bucket_defs[0]
    bucket_name = _single_target(bucket_binding)
    lookups = []
    for binding in bindings:
        target = _single_target(binding)
        if target is None or binding.guard or binding.value is None \
                or binding.value.kind != "call":
            continue
        call = _call_at(index, callable_symbol, binding.value.span)
        field = _self_field(binding.value.children[0])
        if call is not None and field is not None and len(call.args) == 1 \
                and _base_name(call.args[0]) == bucket_name:
            lookups.append((binding, call, field))
    if len(lookups) != 1:
        return None
    lookup_binding, lookup_call, field = lookups[0]
    lookup_name = _single_target(lookup_binding)
    returns = tuple(item for item in index.return_observations_in(callable_symbol)
                    if not item.guard and item.value is not None
                    and _derived_from_name(
                        index, callable_symbol, item.value, lookup_name,
                        before=item.span))
    if len(returns) != 1:
        return None
    return (aranges, relative[0], bucket_binding, bucket_call,
            bucket_callable, lookup_binding, lookup_call, returns[0], field)


def _classify_bucket(index, callable_symbol):
    record = index.callable_by_symbol(callable_symbol)
    if record is None or not record.params:
        return False
    relative = record.params[0].name
    calls = tuple(index.calls_in(callable_symbol))
    protocols = tuple((_external_call_target(index, item), item)
                      for item in calls)
    abs_calls = tuple(item for target, item in protocols
                      if target in _ABS_PROTOCOLS)
    log_calls = tuple(item for target, item in protocols
                      if target in _LOG_PROTOCOLS)
    where_calls = tuple(item for target, item in protocols
                        if target in _WHERE_PROTOCOLS)
    if len(abs_calls) != 1 or len(log_calls) != 1 or len(where_calls) != 1:
        return False
    abs_call = abs_calls[0]
    if len(abs_call.args) != 1 or _base_name(abs_call.args[0]) != relative:
        return False
    where_call = where_calls[0]
    if len(where_call.args) != 3:
        return False
    bindings = tuple(index.bindings_in(callable_symbol))
    small = _base_name(where_call.args[0])
    exact_lane = _base_name(where_call.args[1])
    large_lane = _base_name(where_call.args[2])
    if small is None or not any(
            _single_target(item) == small and item.value is not None
            and item.value.kind == "compare" and item.value.operator == "<"
            and _base_name(item.value.children[0]) == relative
            for item in bindings):
        return False
    log_call = log_calls[0]
    if exact_lane != relative or large_lane is None or not any(
            _single_target(item) == large_lane and item.value is not None
            and _expr_contains_span(item.value, log_call.span)
            for item in bindings):
        return False
    returns = tuple(item for item in index.return_observations_in(callable_symbol)
                    if not item.guard and item.value is not None)
    where_binding = next((item for item in bindings
                          if item.value is not None
                          and _expr_contains_span(item.value, where_call.span)), None)
    return len(returns) == 1 and where_binding is not None \
        and _derived_from_targets(
            index, callable_symbol, returns[0].value,
            set(_target_names(where_binding)), before=returns[0].span)


def _ownership_schedule(index, candidates, block_occurrence,
                        attention_occurrence, flag_formal):
    root = candidates.component_root
    proofs = tuple(proof for proof in candidates.repeated_child.proofs
                   if proof.child_occurrence == block_occurrence)
    if len(proofs) != 1:
        return ReaderFailure("conflict", "one repeated block proof is required")
    block_site = proofs[0].template.element_template
    comprehensions = tuple(
        item for item in index.comprehensions_in(block_site.enclosing_callable)
        if item.span is not None and len(item.outputs) == 1
        and item.outputs[0].span == block_site.constructor.span
        and len(item.clauses) == 1)
    if len(comprehensions) != 1:
        return ReaderFailure(
            "incomplete_graph", "block construction has no unique comprehension")
    comprehension = comprehensions[0]
    clause = comprehension.clauses[0]
    if clause.async_flag or clause.filters or clause.target.kind != "name":
        return ReaderFailure(
            "unsupported_syntax", "repeated construction is not one direct index")

    site_lookup = {}
    for item in (*index.construction_sites,
                 *(element for record in index.containers
                   for element in record.elements)):
        site_lookup.setdefault(item.site_id, item)
    block_depth = len(block_occurrence.sites) - 1
    current = flag_formal.name
    transport = []
    selector_expr = None
    for depth in range(len(attention_occurrence.sites) - 1,
                       block_depth - 1, -1):
        site = site_lookup.get(attention_occurrence.sites[depth])
        if site is None:
            return ReaderFailure("incomplete_graph", "constructor site is absent")
        parent = OwnerOccurrenceId(
            attention_occurrence.root, attention_occurrence.sites[:depth])
        resolution = bind_construction_site(index, root, parent, site)
        if resolution.status not in {"resolved", "partial"}:
            return ReaderFailure(
                "incomplete_graph", "constructor arguments are not exact")
        binding = resolution.for_formal(current)
        if binding is None:
            return ReaderFailure(
                "incomplete_graph", "table-owner flag is not transported")
        transport.append(binding)
        if depth == block_depth:
            selector_expr = binding.actual
        elif binding.actual.kind == "name":
            current = binding.actual.name
        else:
            return ReaderFailure(
                "unsupported_syntax", "owner flag transport is not transparent")
    if selector_expr is None \
            or not _first_index_selector(
                index, block_site.enclosing_callable, selector_expr,
                clause.target.name):
        return ReaderFailure(
            "unsupported_syntax", "table ownership is not exact first-index selection")
    spans = tuple(dict.fromkeys((
        block_site.span, comprehension.span, selector_expr.span,
        *(span for binding in transport for span in binding.spans),
    )))
    return RelativeBiasOwnershipSchedule(
        attention_occurrence, block_site, comprehension,
        tuple(reversed(transport)), selector_expr, clause.target.name, 0, spans)


def _compute_is_else_of_disabled_flag(bindings, compute, source, flag):
    zeros = tuple(item for item in bindings
                  if _single_target(item) == source
                  and item.value is not None and item.value.kind == "call"
                  and _guard_has_none_check(item.guard, source)
                  and any(_is_not_self_field(step.test, flag)
                          for step in item.guard if step.test is not None))
    if len(zeros) != 1 or not compute.guard or not zeros[0].guard:
        return False
    return compute.guard[0] == zeros[0].guard[0] \
        and compute.guard[-1].kind == "else" \
        and compute.guard[-1].span == zeros[0].guard[-1].span


def _first_index_selector(index, callable_symbol, expression, index_name):
    if expression.kind != "call" or len(expression.children) != 2 \
            or expression.children[0].kind != "name" \
            or expression.children[0].name != "bool" \
            or _name_shadowed(index, callable_symbol, "bool"):
        return False
    compare = expression.children[1]
    return compare.kind == "compare" and compare.operator == "==" \
        and len(compare.children) == 2 \
        and compare.children[0].kind == "name" \
        and compare.children[0].name == index_name \
        and compare.children[1].kind == "constant" \
        and compare.children[1].const_value == 0 \
        and not isinstance(compare.children[1].const_value, bool)


def _external_call_target(index, call):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    return proof.qualified_target if proof is not None else None


def _call_at(index, callable_symbol, span):
    matches = tuple(item for item in index.calls_in(callable_symbol)
                    if item.span == span)
    return matches[0] if len(matches) == 1 else None


def _single_target(binding):
    if len(binding.targets) != 1 or binding.targets[0].kind != "name":
        return None
    return binding.targets[0].name


def _self_target(binding):
    if len(binding.targets) != 1:
        return None
    return _self_field(binding.targets[0])


def _target_names(binding):
    return tuple(name for target in binding.targets
                 for name in _expr_names(target))


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1 \
            or expression.children[0].kind != "name" \
            or expression.children[0].name != "self":
        return None
    return expression.name


def _single_positive_self_guard(guard):
    fields = tuple(_self_field(step.test) for step in guard
                   if step.kind == "if" and step.test is not None
                   and _self_field(step.test) is not None)
    return fields[0] if len(fields) == 1 else None


def _is_not_self_field(expression, field):
    return expression is not None and expression.kind == "unaryop" \
        and expression.operator == "not" and len(expression.children) == 1 \
        and _self_field(expression.children[0]) == field


def _guard_has_none_check(guard, name):
    return any(step.test is not None and step.test.kind == "compare"
               and step.test.operator == "is" and len(step.test.children) == 2
               and step.test.children[0].kind == "name"
               and step.test.children[0].name == name
               and step.test.children[1].kind == "constant"
               and step.test.children[1].const_value is None
               for step in guard)


def _base_name(expression):
    current = expression
    while current.kind in {"attribute", "subscript"} and current.children:
        current = current.children[0]
    if current.kind == "call" and len(current.children) == 1:
        current = current.children[0]
    return current.name if current.kind == "name" else None


def _derived_from_name(index, callable_symbol, expression, name, *, before):
    if name in _expr_names(expression):
        return True
    definitions = tuple(item for item in index.bindings_in(callable_symbol)
                        if item.span is not None and _span_key(item.span)
                        < _span_key(before) and _single_target(item) is not None
                        and item.value is not None)
    current = {name}
    for item in definitions:
        target = _single_target(item)
        if target in current:
            current.update(_expr_names(item.value))
    return bool(current & set(_expr_names(expression)))


def _derived_from_targets(index, callable_symbol, expression, targets, *, before):
    if set(_expr_names(expression)) & targets:
        return True
    definitions = tuple(item for item in index.bindings_in(callable_symbol)
                        if item.span is not None and _span_key(item.span)
                        < _span_key(before) and item.value is not None)
    current = set(targets)
    changed = True
    while changed:
        changed = False
        for item in definitions:
            names = set(_target_names(item))
            if names & current:
                added = set(_expr_names(item.value)) - current
                current.update(added)
                changed = changed or bool(added)
    return bool(current & set(_expr_names(expression)))


def _expr_names(expression):
    if not isinstance(expression, ExprNode):
        return ()
    names = ((expression.name,) if expression.kind == "name" else ())
    return (*names,
            *(name for child in expression.children for name in _expr_names(child)),
            *(name for _key, child in expression.keyword_children
              for name in _expr_names(child)))


def _expr_contains_span(expression, span):
    if not isinstance(expression, ExprNode):
        return False
    return expression.span == span or any(
        _expr_contains_span(child, span) for child in expression.children) or any(
        _expr_contains_span(child, span)
        for _name, child in expression.keyword_children)


def _name_shadowed(index, callable_symbol, name):
    return any(item.name == name and item.context in {"parameter", "store", "del"}
               for item in index.identifiers_in(callable_symbol)) \
        or any(item.name == name and item.kind != "import"
               for item in index.module_bindings_in(callable_symbol.source))


def _span_key(span):
    return (span.source.canonical_path, span.line, span.col,
            span.end_line, span.end_col)


__all__ = [
    "RelativeBiasOwnershipSchedule",
    "RelativeBucketProducer",
    "RelativePositionBiasEvidence",
    "decoder_relative_position_bias_for_path",
]
