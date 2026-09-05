"""Exact-owner attention head-dimension evidence.

U6 mechanism evidence already proves which source expression is the structural
factor shared by Q/K/V projection widths.  This reader evaluates that exact
factor through the exact owner construction and records every config operand it
actually used.  A plausible ``config.head_dim`` that the source never consumes
is therefore powerless.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention import (
    AttentionHeadBinding,
    decoder_attention_mechanism_for_path,
    exact_config_path_for_expression,
)
from .attention_child import AttentionChildEvidence, AttentionComputeProof
from .affine import site_is_affine
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .construction_calls import ConstructionOccurrenceId
from .decoder_block import decoder_block_path_for_config
from .expression_eval import (
    ConfigExpressionEvaluator,
    construction_guard_state,
    construction_site,
    constructor_argument_env,
    locals_before,
    qualify_premises,
    scoped_document,
    unique_premises,
)
from .models import SourceBundle
from .framework_config import (
    FrameworkConfigDefaultValue,
    framework_config_alias,
    framework_config_default_selector,
)
from .layer_selector import (
    ConfigSelectorOperand,
    LayerFieldSchedule,
    resolve_layer_field_schedule,
)
from .mixer_schedule import (
    DecoderMixerSchedule,
    MixerCandidateProof,
    decoder_mixer_schedule_for_path,
)
from .program_index import (
    BindingObservation,
    CallObservation,
    ExprNode,
    FieldAssignRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class AttentionHeadGeometry:
    owner_occurrence: object
    owner_symbol: SymbolId
    head_dim: int
    premises: tuple[tuple[tuple[str, ...], object], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        from .component_owner import OwnerOccurrenceId
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("attention geometry names an exact owner occurrence")
        if not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("attention geometry names an exact owner symbol")
        if self.owner_symbol.source.component_key != \
                self.owner_occurrence.root.source.component_key:
            raise ValueError("attention geometry owner belongs to its component")
        if not isinstance(self.head_dim, int) or isinstance(self.head_dim, bool) \
                or self.head_dim <= 0:
            raise ValueError("attention head dimension is a positive integer")
        if len({path for path, _value in self.premises}) != len(self.premises) \
                or any(not isinstance(path, tuple) or not path or any(
                    not isinstance(part, str) or not part for part in path)
                    for path, _value in self.premises):
            raise ValueError("attention geometry premises are exact and unique")
        if not self.spans or any(
                not isinstance(span, SourceSpan)
                or span.source.component_key
                != self.owner_symbol.source.component_key
                for span in self.spans):
            raise ValueError(
                "attention geometry carries exact component provenance")


@dataclass(frozen=True)
class AttentionGeometryApplication:
    """One exact ordinary-attention occurrence's geometry application.

    The field schedules alone are not architectural evidence.  ``reshape_calls``
    proves the selected per-layer dimension reaches the three Q/K/V lanes;
    ``repeat_calls`` proves the exact grouping quotient is applied to both K
    and V inside the positively-proven attention computation.
    """

    candidate: MixerCandidateProof
    query_heads_path: tuple[str, ...]
    group_assignment: FieldAssignRecord
    head_dim_schedule: LayerFieldSchedule
    group_schedule: LayerFieldSchedule
    reshape_calls: tuple[CallObservation, CallObservation, CallObservation]
    projection_variants: tuple[
        tuple[ConstructionOccurrenceId, ...],
        tuple[ConstructionOccurrenceId, ...],
        tuple[ConstructionOccurrenceId, ...],
    ]
    repeat_calls: tuple[CallObservation, CallObservation]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, MixerCandidateProof) \
                or self.candidate.kind != "ordinary_attention" \
                or not isinstance(self.candidate.mechanism, AttentionChildEvidence):
            raise TypeError("geometry application carries exact ordinary attention")
        if not self.query_heads_path or any(
                not isinstance(part, str) or not part
                for part in self.query_heads_path):
            raise TypeError("geometry application carries an exact query-head path")
        if not isinstance(self.group_assignment, FieldAssignRecord) \
                or self.group_assignment.field != self.group_schedule.field:
            raise TypeError("geometry grouping cites its exact field assignment")
        owner = self.candidate.mechanism.compute_occurrence
        for schedule in (self.head_dim_schedule, self.group_schedule):
            if not isinstance(schedule, LayerFieldSchedule) \
                    or schedule.status != "resolved" or schedule.owner != owner:
                raise ValueError("geometry carries complete exact-owner field schedules")
        if len(self.reshape_calls) != 3 or len(set(self.reshape_calls)) != 3 \
                or len(self.projection_variants) != 3 \
                or any(not variants for variants in self.projection_variants) \
                or any(not isinstance(item, ConstructionOccurrenceId)
                       or item.parent != owner
                       for variants in self.projection_variants
                       for item in variants) \
                or any(len(set(variants)) != len(variants)
                       for variants in self.projection_variants) \
                or len(self.repeat_calls) != 2 \
                or len(set(self.repeat_calls)) != 2:
            raise ValueError(
                "geometry application carries three projection/reshape lanes "
                "and two repeats")
        entry_callable = self.candidate.mechanism.compute.entry_call.enclosing_callable
        compute_callable = self.candidate.mechanism.compute.callable_symbol
        if any(not isinstance(call, CallObservation)
               or call.enclosing_callable != entry_callable
               for call in self.reshape_calls) \
                or any(not isinstance(call, CallObservation)
                       or call.enclosing_callable != compute_callable
                       for call in self.repeat_calls):
            raise ValueError("geometry calls belong to the exact entry/compute callables")
        required = {
            self.group_assignment.span,
            *(call.span for call in self.reshape_calls),
            *(item.site.span for variants in self.projection_variants
              for item in variants),
            *(call.span for call in self.repeat_calls),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan)
                       for span in self.spans):
            raise ValueError("geometry application retains every decisive span")


@dataclass(frozen=True)
class AttentionLayerGeometry:
    layer_index: int
    kind: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    application_occurrence: OwnerOccurrenceId
    operands: tuple[ConfigSelectorOperand, ...]

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, bool) \
                or not isinstance(self.layer_index, int) \
                or self.layer_index < 0:
            raise ValueError("layer geometry index is non-negative")
        if self.kind not in {"mha", "gqa", "mqa"}:
            raise ValueError("layer geometry kind is exact head sharing")
        values = (self.num_heads, self.num_kv_heads, self.head_dim)
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
               for item in values) or self.num_heads % self.num_kv_heads:
            raise ValueError("layer geometry has positive divisible head axes")
        expected = (
            "mha" if self.num_heads == self.num_kv_heads else
            "mqa" if self.num_kv_heads == 1 else "gqa")
        if self.kind != expected:
            raise ValueError("layer geometry kind agrees with its exact axes")
        if not isinstance(self.application_occurrence, OwnerOccurrenceId):
            raise TypeError("layer geometry names its exact attention occurrence")
        if any(not isinstance(item, ConfigSelectorOperand)
               for item in self.operands) \
                or len(set(self.operands)) != len(self.operands):
            raise ValueError("layer geometry operands are typed and unique")


@dataclass(frozen=True)
class DecoderAttentionGeometrySchedule:
    mixer_schedule: DecoderMixerSchedule
    applications: tuple[AttentionGeometryApplication, ...]
    decisions: tuple[AttentionLayerGeometry | None, ...]
    config_dependencies: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mixer_schedule, DecoderMixerSchedule) \
                or not self.applications:
            raise TypeError("geometry schedule composes one exact mixer schedule")
        if any(not isinstance(item, AttentionGeometryApplication)
               for item in self.applications):
            raise TypeError("geometry schedule applications are typed")
        identities = tuple(
            item.candidate.occurrence for item in self.applications)
        if len(set(identities)) != len(identities):
            raise ValueError("geometry applications are occurrence-unique")
        if len(self.decisions) != len(self.mixer_schedule.decisions):
            raise ValueError("geometry covers the exact layer schedule")
        by_occurrence = {
            item.candidate.occurrence: item for item in self.applications}
        for mixer, decision in zip(self.mixer_schedule.decisions, self.decisions):
            if mixer.state == "ordinary_attention":
                if not isinstance(decision, AttentionLayerGeometry) \
                        or decision.layer_index != mixer.layer_index \
                        or decision.application_occurrence != mixer.occurrence \
                        or mixer.occurrence not in by_occurrence:
                    raise ValueError("ordinary layers round-trip through geometry")
            elif decision is not None:
                raise ValueError("non-attention layers carry no attention geometry")
        paths = tuple(path for path, _kind in self.config_dependencies)
        if len(paths) != len(set(paths)) or any(
                kind not in {"config_declared", "class_default"}
                for _path, kind in self.config_dependencies):
            raise ValueError("geometry dependencies are exact, typed and unique")
        required = {
            *(span for item in self.applications for span in item.spans),
        }
        if not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("geometry schedule provenance is closed")


def decoder_attention_geometry_schedule_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[DecoderAttentionGeometrySchedule]:
    """Prove exact per-layer ordinary-attention head geometry.

    This is the conditional-construction counterpart of the U6 global head
    binding.  It is intentionally narrower: the exact mixer schedule must
    first prove the ordinary-attention occurrence; the occurrence's constructor
    must then prove a query/KV quotient and a per-layer dimension; finally those
    fields must reach Q/K/V reshape/projection and K/V repetition sites.
    """
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("attention geometry schedule requires index and bundle")
    mixer_result = decoder_mixer_schedule_for_path(
        index, bundle, tuple(config_path),
        allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if mixer_result.status != "resolved" or mixer_result.value is None:
        return mixer_result
    mixer = mixer_result.value
    root = mixer.block_candidates.component_root
    effective_selector = config_selector
    alias = framework_config_alias(
        index, root, mixer.block_candidates.stage_occurrence)
    if alias.status == "resolved" and callable(effective_selector):
        effective_selector = framework_config_default_selector(
            index, alias.value, effective_selector,
            config_prefix=tuple(config_path))

    applications = []
    dependencies = dict(mixer.config_dependencies)
    for candidate in mixer.candidates:
        if candidate.kind != "ordinary_attention":
            continue
        application = _geometry_application(
            index, root, mixer, candidate, effective_selector,
            tuple(config_path))
        if isinstance(application, ReaderFailure):
            return ReaderResult.failed(
                mixer.block_occurrence, (application,),
                provenance=mixer_result.provenance)
        applications.append(application)
        for schedule in (
                application.head_dim_schedule,
                application.group_schedule):
            for decision in schedule.decisions:
                for operand in decision.operands:
                    previous = dependencies.get(operand.path)
                    if previous is not None and previous != operand.source_kind:
                        return ReaderResult.failed(
                            mixer.block_occurrence, (ReaderFailure(
                                "conflicting_evidence",
                                "one geometry operand has conflicting provenance"),),
                            provenance=mixer_result.provenance)
                    dependencies[operand.path] = operand.source_kind
        present, _value, source_kind = _selected_value(
            effective_selector, application.query_heads_path)
        if not present:
            return ReaderResult.failed(
                mixer.block_occurrence, (ReaderFailure(
                    "incomplete_graph",
                    "the exact query-head operand is unavailable"),),
                provenance=mixer_result.provenance)
        dependencies[application.query_heads_path] = source_kind
    if not applications:
        return ReaderResult.absent(
            mixer.block_occurrence, provenance=mixer_result.provenance)

    by_occurrence = {
        item.candidate.occurrence: item for item in applications}
    decisions = []
    for mixer_decision in mixer.decisions:
        if mixer_decision.state != "ordinary_attention":
            decisions.append(None)
            continue
        application = by_occurrence.get(mixer_decision.occurrence)
        if application is None:
            return ReaderResult.failed(
                mixer.block_occurrence, (ReaderFailure(
                    "incomplete_graph",
                    "an ordinary layer lacks its exact geometry application"),),
                provenance=mixer_result.provenance)
        index_value = mixer_decision.layer_index
        head = application.head_dim_schedule.decisions[index_value]
        groups = application.group_schedule.decisions[index_value]
        present, query_heads, _source_kind = _selected_value(
            effective_selector, application.query_heads_path)
        if not present or any(
                item.state != "resolved" for item in (head, groups)) \
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0
                       for value in (query_heads, head.value, groups.value)) \
                or query_heads % groups.value:
            return ReaderResult.failed(
                mixer.block_occurrence, (ReaderFailure(
                    "unsupported_syntax",
                    "the exact per-layer head quotient is not evaluable"),),
                provenance=mixer_result.provenance)
        kv_heads = query_heads // groups.value
        kind = (
            "mha" if query_heads == kv_heads else
            "mqa" if kv_heads == 1 else "gqa")
        operands = tuple(dict.fromkeys((
            *mixer_decision.operands,
            *head.operands,
            *groups.operands,
        )))
        decisions.append(AttentionLayerGeometry(
            index_value, kind, query_heads, kv_heads, head.value,
            application.candidate.occurrence, operands))
    spans = tuple(dict.fromkeys((
        *mixer.spans,
        *(span for item in applications for span in item.spans),
    )))
    value = DecoderAttentionGeometrySchedule(
        mixer, tuple(applications), tuple(decisions),
        tuple(sorted(dependencies.items())), spans)
    return ReaderResult.resolved(
        mixer.block_occurrence, value,
        provenance=(*mixer_result.provenance, ReaderProvenance(
            "code_and_config" if dependencies else "source",
            spans=spans,
            config_paths=tuple(path for path, _kind in sorted(
                dependencies.items())),
            detail=("exact per-layer constructor fields reach Q/K/V reshape, "
                    "projection and K/V repetition sites"))))


def _geometry_application(
        index, root, mixer, candidate, config_selector, config_prefix):
    mechanism = candidate.mechanism
    if not isinstance(mechanism, AttentionChildEvidence):
        return ReaderFailure(
            "unsupported_syntax", "ordinary mixer lacks attention evidence")
    node = root.graph.node_for(mechanism.compute_occurrence)
    if node is None:
        return ReaderFailure(
            "out_of_owner", "attention occurrence is absent from its graph")
    constructor = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.__init__")
    layer_indices = tuple(range(mixer.transport.layer_count))
    index_parameter = candidate.selector.layer_index_parameter
    head_schedule = resolve_layer_field_schedule(
        index, root, mechanism.compute_occurrence, constructor, "head_dim",
        layer_indices, index_parameter, config_selector=config_selector,
        config_prefix=config_prefix)
    group_schedule = resolve_layer_field_schedule(
        index, root, mechanism.compute_occurrence, constructor,
        "num_key_value_groups", layer_indices, index_parameter,
        config_selector=config_selector, config_prefix=config_prefix)
    if head_schedule.status != "resolved" \
            or group_schedule.status != "resolved":
        return ReaderFailure(
            "incomplete_graph",
            "per-layer head dimension/group fields are not completely resolved")
    assignments = group_schedule.assignments
    if len(assignments) != 1:
        return ReaderFailure(
            "unsupported_syntax",
            "the grouping field has no unique exact quotient assignment")
    assignment = assignments[0]
    expression = assignment.value
    if expression.kind != "binop" or expression.operator != "//" \
            or len(expression.children) != 2:
        return ReaderFailure(
            "unsupported_syntax",
            "the grouping field is not an exact query/KV quotient")
    query_path = exact_config_path_for_expression(
        index, node, expression.children[0], config_prefix=config_prefix)
    if query_path is None:
        return ReaderFailure(
            "incomplete_graph", "query-head numerator has no exact config path")
    repeat_calls = _group_repeat_calls(index, mechanism.compute, group_schedule.field)
    if repeat_calls is None:
        return ReaderFailure(
            "incomplete_graph",
            "the exact grouping field does not reach both K/V repeat paths")
    projection_path = _head_reshape_calls(
        index, node, mechanism.compute, head_schedule, query_path)
    if projection_path is None:
        return ReaderFailure(
            "incomplete_graph",
            "the exact head dimension does not reach all Q/K/V shape paths")
    reshape_calls, projection_variants = projection_path
    spans = tuple(dict.fromkeys((
        *candidate.spans,
        assignment.span,
        *(item.span for item in head_schedule.assignments),
        *(item.span for item in group_schedule.assignments),
        *(call.span for call in reshape_calls),
        *(item.site.span for variants in projection_variants
          for item in variants),
        *(call.span for call in repeat_calls),
    )))
    return AttentionGeometryApplication(
        candidate, query_path, assignment, head_schedule, group_schedule,
        reshape_calls, projection_variants, repeat_calls, spans)


def _selected_value(selector, path):
    if not callable(selector):
        return False, None, ""
    selected = selector(path)
    if isinstance(selected, FrameworkConfigDefaultValue):
        return True, selected.value, "class_default"
    if isinstance(selected, tuple) and len(selected) in {2, 3} \
            and isinstance(selected[0], bool):
        present, value = selected[:2]
        kind = selected[2] if len(selected) == 3 else "config_declared"
        return bool(present), value, kind
    return selected is not None, selected, "config_declared"


def _group_repeat_calls(index, compute: AttentionComputeProof, field):
    callable_record = index.callable_by_symbol(compute.callable_symbol)
    if callable_record is None or not callable_record.params:
        return None
    module_name = callable_record.params[0].name
    matches = []
    targets = []
    for call in index.calls_in(compute.callable_symbol):
        if len(call.args) < 2 \
                or _attribute_chain(call.args[1]) != (module_name, field):
            continue
        target = _binding_target_for_call(
            index, compute.callable_symbol, call)
        if target is None:
            continue
        matches.append(call)
        targets.append(target)
    if len(matches) != 2 or len(set(targets)) != 2:
        return None
    decisive_calls = tuple(
        call for call in index.calls_in(compute.callable_symbol)
        if call.span in compute.spans)
    for call, target in zip(matches, targets):
        if not any(_span_before(call.span, consumer.span)
                   and any(_expr_contains_name(arg, target)
                           for arg in consumer.args)
                   for consumer in decisive_calls):
            return None
    return tuple(matches)


def _head_reshape_calls(index, node, compute, head_schedule, query_path):
    entry = compute.entry_call
    if len(entry.args) < 4 or _plain_name(entry.args[0]) != "self":
        return None
    lane_names = tuple(_plain_name(item) for item in entry.args[1:4])
    if any(name is None for name in lane_names) or len(set(lane_names)) != 3:
        return None
    head_paths = {
        operand.path for decision in head_schedule.decisions
        for operand in decision.operands
    }
    calls = index.calls_in(entry.enclosing_callable)
    reshapes = []
    projection_fields = []
    projection_variants = []
    for lane_index, lane in enumerate(lane_names):
        candidates = []
        for binding in index.bindings_in(entry.enclosing_callable):
            if binding.span is None or not _span_before(binding.span, entry.span) \
                    or lane not in _binding_plain_targets(binding) \
                    or binding.value is None:
                continue
            for call in calls:
                if call.span is None or not _expr_contains_span(
                        binding.value, call.span) \
                        or not _is_shape_call(call):
                    continue
                if not any(_expression_uses_dimension(
                        index, node, entry.enclosing_callable, arg,
                        "head_dim", head_paths, call.span, frozenset())
                        for arg in call.args):
                    continue
                projection_field = _projection_field_for_shape_call(call)
                if projection_field is None:
                    continue
                sites = tuple(
                    site for site in index.construction_sites_of(node.symbol)
                    if site.enclosing_callable.qualified_name.endswith(".__init__")
                    and site.target == projection_field
                    and site_is_affine(index, site)
                    and any(_expression_uses_dimension(
                        index, node, site.enclosing_callable, argument,
                        "head_dim", head_paths, site.span, frozenset())
                        for argument in (*site.args,
                                         *(value for _key, value in site.kwargs))))
                if not sites:
                    continue
                if lane_index == 0 and not any(
                        _expression_uses_config_path(
                            index, node, argument, query_path, site.span,
                            frozenset())
                        for site in sites
                        for argument in (*site.args,
                                         *(value for _key, value in site.kwargs))):
                    continue
                occurrences = tuple(
                    ConstructionOccurrenceId(
                        node.occurrence, site.site_id)
                    for site in sites)
                candidates.append((call, projection_field, occurrences))
        distinct = {
            (item[0].span, item[1], tuple(x.site for x in item[2])): item
            for item in candidates}
        if len(distinct) != 1:
            return None
        call, _projection_field, occurrences = next(iter(distinct.values()))
        reshapes.append(call)
        projection_fields.append(_projection_field)
        projection_variants.append(occurrences)
    if len(set(projection_fields)) != 3:
        return None
    return tuple(reshapes), tuple(projection_variants)


def _binding_target_for_call(index, callable_symbol, call):
    matches = []
    for binding in index.bindings_in(callable_symbol):
        if binding.value is None or not _expr_contains_span(
                binding.value, call.span):
            continue
        matches.extend(_binding_plain_targets(binding))
    return matches[0] if len(set(matches)) == 1 else None


def _binding_plain_targets(binding: BindingObservation):
    return tuple(
        name for target in binding.targets
        if (name := _plain_name(target)) is not None)


def _plain_name(expression):
    return expression.name if isinstance(expression, ExprNode) \
        and expression.kind == "name" and expression.name else None


def _attribute_chain(expression):
    if not isinstance(expression, ExprNode):
        return None
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind == "attribute" and len(expression.children) == 1:
        base = _attribute_chain(expression.children[0])
        return (*base, expression.name) if base is not None else None
    return None


def _is_shape_call(call):
    return call.callee.kind == "attribute" \
        and call.callee.name in {"view", "reshape"} \
        and len(call.callee.children) == 1


def _projection_field_for_shape_call(call):
    callee = call.callee
    if callee.kind != "attribute" or len(callee.children) != 1:
        return None
    receiver = callee.children[0]
    if receiver.kind != "call" or not receiver.children:
        return None
    chain = _attribute_chain(receiver.children[0])
    return chain[1] if chain is not None and len(chain) == 2 \
        and chain[0] == "self" else None


def _expression_uses_dimension(
        index, node, callable_symbol, expression, field, paths, before, seen):
    if not isinstance(expression, ExprNode):
        return False
    chain = _attribute_chain(expression)
    if chain == ("self", field):
        return True
    path = exact_config_path_for_expression(index, node, expression)
    if path in paths:
        return True
    if expression.kind == "name" and expression.name:
        key = (callable_symbol, expression.name)
        if key in seen:
            return False
        bindings = tuple(sorted(
            (item for item in index.bindings_in(callable_symbol)
             if item.span is not None and _span_before(item.span, before)
             and expression.name in _binding_plain_targets(item)
             and item.value is not None),
            key=lambda item: _span_key(item.span), reverse=True))
        if not bindings:
            return False
        latest = bindings[0]
        if latest.guard:
            return False
        return _expression_uses_dimension(
            index, node, callable_symbol, latest.value, field, paths,
            latest.span, seen | {key})
    return any(_expression_uses_dimension(
        index, node, callable_symbol, child, field, paths, before, seen)
        for child in (*expression.children,
                      *(value for _key, value in expression.keyword_children)))


def _expression_uses_config_path(
        index, node, expression, path, before, seen):
    if not isinstance(expression, ExprNode):
        return False
    if exact_config_path_for_expression(index, node, expression) == path:
        return True
    if expression.kind == "name" and expression.name:
        key = expression.name
        if key in seen:
            return False
        callable_symbol = _enclosing_callable_for_span(index, node, before)
        if callable_symbol is None:
            return False
        bindings = tuple(sorted(
            (item for item in index.bindings_in(callable_symbol)
             if item.span is not None and _span_before(item.span, before)
             and key in _binding_plain_targets(item) and item.value is not None),
            key=lambda item: _span_key(item.span), reverse=True))
        return bool(bindings) and _expression_uses_config_path(
            index, node, bindings[0].value, path, bindings[0].span,
            seen | {key})
    return any(_expression_uses_config_path(
        index, node, child, path, before, seen)
        for child in (*expression.children,
                      *(value for _key, value in expression.keyword_children)))


def _enclosing_callable_for_span(index, node, span):
    candidates = tuple(
        item.enclosing_callable for item in index.field_assigns_of(node.symbol)
        if item.span == span)
    if candidates:
        return candidates[0]
    sites = tuple(
        item.enclosing_callable for item in index.construction_sites_of(node.symbol)
        if item.span == span)
    return sites[0] if sites else None


def _expr_contains_name(expression, name):
    if not isinstance(expression, ExprNode):
        return False
    return expression.kind == "name" and expression.name == name or any(
        _expr_contains_name(child, name)
        for child in (*expression.children,
                      *(value for _key, value in expression.keyword_children)))


def _expr_contains_span(expression, span):
    if not isinstance(expression, ExprNode):
        return False
    return expression.span == span or any(
        _expr_contains_span(child, span)
        for child in (*expression.children,
                      *(value for _key, value in expression.keyword_children)))


def _span_before(left, right):
    return (left.line, left.col, left.end_line, left.end_col) < \
        (right.line, right.col, right.end_line, right.end_col)


def decoder_attention_head_geometry_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    config_document,
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[AttentionHeadGeometry]:
    """Evaluate the exact shared Q/K/V head-width factor."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention geometry requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("attention geometry requires a SourceBundle")
    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if block.status != "resolved":
        return block
    mechanism = decoder_attention_mechanism_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage,
        config_document=config_document, config_selector=config_selector)
    if mechanism.status != "resolved":
        return mechanism
    result = attention_head_geometry_at_block(
        index, block.value.component_root, block.value.block_occurrence,
        mechanism.value, config_document, tuple(config_path))
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *mechanism.provenance,
                    *result.provenance))


def attention_head_geometry_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
    binding: AttentionHeadBinding,
    config_document,
    config_path: tuple[str, ...],
) -> ReaderResult[AttentionHeadGeometry]:
    """Evaluate head width for one already-proven block-local binding.

    This is the occurrence-qualified primitive used by recursive towers.  It
    performs no block or mechanism selection: callers must supply the exact
    block and its exact :class:`AttentionHeadBinding`.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention geometry requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_head_geometry_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("attention geometry requires an exact block")
    if not isinstance(binding, AttentionHeadBinding) \
            or binding.block_occurrence != block_occurrence:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner",
            "the supplied head binding does not belong to this exact block"),))
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    graph = root.graph
    node = graph.node_for(binding.attention_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact attention owner does not round-trip"),))
    document = scoped_document(config_document, config_path)
    if document is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact component config is unavailable"),))
    env = constructor_argument_env(
        index, graph, binding.attention_occurrence, document)
    # A symbolic repeated-block selector can leave the attention constructor
    # call's ordinary arguments unresolved even though this exact attention
    # class reads its geometry directly from its own config binding.  Preserve
    # that direct evidence: an empty env can evaluate config paths/literals,
    # while any expression that genuinely depends on a missing constructor
    # formal still remains unknown.
    if env is None:
        env = {}
    sites = tuple(
        construction_site(index, node.symbol, item.site)
        for item in binding.projections)
    if any(site is None or construction_guard_state(
            index, graph, binding.attention_occurrence,
            site, document) is not True for site in sites):
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact attention projection site is unresolved"),))
    cutoff = min((site.span for site in sites), key=_span_key)
    evaluator = ConfigExpressionEvaluator(
        node.config_bindings, document, env)
    locals_before(index, sites[0].enclosing_callable, cutoff, evaluator)
    result = evaluator.expression(binding.common_factor)
    if result is None or not isinstance(result.value, int) \
            or isinstance(result.value, bool) or result.value <= 0:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "unsupported_syntax",
            "the exact shared Q/K/V factor is not evaluable"),))
    spans = tuple(dict.fromkeys((
        *binding.spans, *result.spans, *(site.span for site in sites))))
    premises = qualify_premises(unique_premises((
        *binding.selection_premises, *result.premises)), config_path)
    if premises is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "conflicting_evidence",
            "the exact head expression carries conflicting config premises"),))
    value = AttentionHeadGeometry(
        binding.attention_occurrence, node.symbol, result.value,
        premises, spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config" if premises else "source",
            spans=spans,
            config_paths=tuple(path for path, _ in premises),
            detail=("the exact source-selected Q/K/V common factor "
                    "evaluates through its exact owner construction")),))


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


__all__ = [
    "AttentionGeometryApplication", "AttentionHeadGeometry",
    "AttentionLayerGeometry", "DecoderAttentionGeometrySchedule",
    "attention_head_geometry_at_block",
    "decoder_attention_geometry_schedule_for_path",
    "decoder_attention_head_geometry_for_path",
]
