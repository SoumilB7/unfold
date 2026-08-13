"""U8-E exact cross-layer K/V reuse schedule.

The boundary first proves a real forward read and write through the same
shared-state mapping, then evaluates the constructor fields that select those
paths for each concrete repeated-block index.  Config counts and layer-type
lists are operands only; without the read/write/application path they author
nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import producer_sources_reaching_expressions
from .framework_config import (
    framework_config_alias,
    framework_config_default_selector,
)
from .layer_selector import (
    LayerFieldSchedule,
    resolve_layer_field_schedule,
)
from .mixer_schedule import (
    DecoderMixerSchedule,
    MixerCandidateProof,
    decoder_mixer_schedule_for_path,
)
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    ExprNode,
    GuardStep,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_UNKNOWN = object()


@dataclass(frozen=True)
class KVSharingApplication:
    """One exact attention forward reading and writing the same K/V store."""

    candidate: MixerCandidateProof
    forward: SymbolId
    read: BindingObservation
    write: BindingObservation
    state_parameter: str
    key_name: str
    value_name: str
    share_field: str
    read_key_field: str
    store_field: str
    write_key_field: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, MixerCandidateProof) \
                or self.candidate.kind != "ordinary_attention":
            raise ValueError("KV sharing belongs to one exact ordinary attention")
        if not isinstance(self.forward, SymbolId) \
                or self.forward \
                != self.candidate.mechanism.compute.entry_call.enclosing_callable:
            raise ValueError("KV sharing belongs to the exact caller forward")
        if not isinstance(self.read, BindingObservation) \
                or not isinstance(self.write, BindingObservation):
            raise TypeError("KV sharing retains exact read/write bindings")
        if any(not item for item in (
                self.state_parameter, self.key_name, self.value_name,
                self.share_field, self.read_key_field,
                self.store_field, self.write_key_field)):
            raise ValueError("KV sharing retains every exact address spelling")
        required = {self.read.span, self.write.span}
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("KV sharing retains exact source provenance")


@dataclass(frozen=True)
class DecoderKVSharingSchedule:
    """Exact source layer (or no reuse) for every decoder layer."""

    mixer_schedule: DecoderMixerSchedule
    application: KVSharingApplication
    field_schedules: tuple[LayerFieldSchedule, ...]
    decisions: tuple[int | None, ...]
    config_dependencies: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mixer_schedule, DecoderMixerSchedule) \
                or not isinstance(self.application, KVSharingApplication):
            raise TypeError("KV schedule carries mixer and application proof")
        if len(self.decisions) != self.mixer_schedule.transport.layer_count \
                or any(item is not None and (
                    isinstance(item, bool) or not isinstance(item, int)
                    or item < 0 or item >= index)
                    for index, item in enumerate(self.decisions)):
            raise ValueError("each KV reuse points to one earlier layer")
        if not self.field_schedules or any(
                item.status != "resolved" for item in self.field_schedules):
            raise ValueError("KV schedule carries complete constructor fields")
        paths = tuple(path for path, _kind in self.config_dependencies)
        if len(paths) != len(set(paths)) or any(
                kind not in {"config_declared", "class_default"}
                for _path, kind in self.config_dependencies):
            raise ValueError("KV dependencies are exact and uniquely typed")
        required = {
            *self.application.spans,
            *(span for schedule in self.field_schedules
              for decision in schedule.decisions for span in decision.spans),
        }
        if not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("KV schedule provenance is closed")


def decoder_kv_sharing_schedule_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[DecoderKVSharingSchedule]:
    """Resolve exact cross-layer K/V sources for one decoder stack."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("KV sharing requires ProgramIndex and SourceBundle")
    mixer_result = decoder_mixer_schedule_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if mixer_result.status != "resolved" or mixer_result.value is None:
        return mixer_result
    mixer = mixer_result.value
    root = mixer.block_candidates.component_root

    matches = []
    for candidate in mixer.candidates:
        if candidate.kind != "ordinary_attention":
            continue
        application = _sharing_application(index, candidate)
        if application is not None:
            matches.append(application)
    if not matches:
        return ReaderResult.absent(
            mixer.block_occurrence, provenance=mixer_result.provenance)
    if len(matches) != 1:
        return ReaderResult.failed(mixer.block_occurrence, (ReaderFailure(
            "conflict", "multiple ordinary-attention occurrences implement KV reuse"),),
            provenance=mixer_result.provenance)
    application = matches[0]
    occurrence = application.candidate.mechanism.compute_occurrence
    node = root.graph.node_for(occurrence)
    if node is None:
        return ReaderResult.failed(mixer.block_occurrence, (ReaderFailure(
            "out_of_owner", "KV compute occurrence is absent from owner graph"),))
    constructor = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.__init__")
    selector = config_selector
    stage_alias = framework_config_alias(
        index, root, mixer.block_candidates.stage_occurrence)
    if stage_alias.status == "resolved" and callable(selector):
        selector = framework_config_default_selector(
            index, stage_alias.value, selector, config_prefix=config_path)
    indices = tuple(range(mixer.transport.layer_count))
    fields = tuple(dict.fromkeys((
        application.share_field,
        application.read_key_field,
        application.store_field,
        application.write_key_field,
    )))
    schedules = tuple(
        resolve_layer_field_schedule(
            index, root, occurrence, constructor, field, indices,
            mixer.transport.binding.formal.name,
            config_selector=selector, config_prefix=config_path)
        for field in fields)
    if any(item.status != "resolved" for item in schedules):
        return ReaderResult.failed(occurrence, (ReaderFailure(
            "incomplete_graph",
            "KV read/write selector fields are not complete for every layer"),),
            provenance=mixer_result.provenance)
    by_field = {item.field: item for item in schedules}

    decisions = []
    for layer_index, mixer_decision in enumerate(mixer.decisions):
        if mixer_decision.state != "ordinary_attention" \
                or mixer_decision.occurrence != application.candidate.occurrence:
            decisions.append(None)
            continue
        shared = _field_value(by_field[application.share_field], layer_index)
        if not isinstance(shared, bool):
            return _failed_value(occurrence, "KV sharing gate is not boolean")
        if not shared:
            decisions.append(None)
            continue
        read_key = _field_value(
            by_field[application.read_key_field], layer_index)
        sources = []
        for source_index in range(layer_index):
            source_mixer = mixer.decisions[source_index]
            if source_mixer.state != "ordinary_attention" \
                    or source_mixer.occurrence \
                    != application.candidate.occurrence:
                continue
            stores = _field_value(
                by_field[application.store_field], source_index)
            write_key = _field_value(
                by_field[application.write_key_field], source_index)
            if stores is True and _safe_equal(write_key, read_key):
                sources.append(source_index)
        if len(sources) != 1 or not _shared_value_reaches_compute(
                index, application, by_field, layer_index):
            return _failed_value(
                occurrence,
                "KV read does not resolve to one earlier stored K/V producer")
        decisions.append(sources[0])

    dependencies = {}
    for schedule in schedules:
        for decision in schedule.decisions:
            for operand in decision.operands:
                previous = dependencies.get(operand.path)
                if previous is not None and previous != operand.source_kind:
                    return _failed_value(
                        occurrence, "one KV selector path has rival provenance")
                dependencies[operand.path] = operand.source_kind
    spans = tuple(dict.fromkeys((
        *application.spans,
        *(span for schedule in schedules
          for decision in schedule.decisions for span in decision.spans),
    )))
    value = DecoderKVSharingSchedule(
        mixer, application, schedules, tuple(decisions),
        tuple(dependencies.items()), spans)
    return ReaderResult.resolved(
        occurrence, value,
        provenance=(*mixer_result.provenance, ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=tuple(dependencies),
            detail=("exact constructor selectors and the exact forward K/V "
                    "mapping read/write agree for every repeated layer"))))


def _sharing_application(index, candidate):
    compute = candidate.mechanism.compute
    # The compute proof may descend into a free eager/SDPA helper.  K/V state
    # selection occurs in the exact owner callable that invokes that helper,
    # retained by the entry call; searching the helper itself would silently
    # miss the real application path.
    forward = compute.entry_call.enclosing_callable
    record = index.callable_by_symbol(forward)
    if record is None:
        return None
    parameters = {item.name for item in record.params if item.name != "self"}
    reads = []
    for binding in index.bindings_in(forward):
        names = _tuple_target_names(binding.targets)
        value = binding.value
        guard_field = _positive_self_guard(binding.guard)
        if len(names) != 2 or value is None or value.kind != "subscript" \
                or len(value.children) != 2 or guard_field is None:
            continue
        state = value.children[0]
        key_field = _self_field(value.children[1])
        if state.kind == "name" and state.name in parameters and key_field:
            reads.append((binding, state.name, names, guard_field, key_field))
    for read, state_name, names, share_field, read_key_field in reads:
        writes = []
        for binding in index.bindings_in(forward):
            if len(binding.targets) != 1 or binding.value is None \
                    or binding.value.kind not in {"tuple", "list"}:
                continue
            target = binding.targets[0]
            if target.kind != "subscript" or len(target.children) != 2 \
                    or target.children[0].kind != "name" \
                    or target.children[0].name != state_name:
                continue
            if tuple(_plain_name(item) for item in binding.value.children) != names:
                continue
            store_field = _positive_self_guard(binding.guard)
            write_key_field = _self_field(target.children[1])
            if store_field and write_key_field:
                writes.append((binding, store_field, write_key_field))
        if len(writes) != 1:
            continue
        write, store_field, write_key_field = writes[0]
        if not _compute_consumes_names(compute.input_calls, names):
            continue
        spans = tuple(dict.fromkeys((
            read.span, write.span,
            *(step.span for step in read.guard),
            *(step.span for step in write.guard),
            *compute.spans,
        )))
        return KVSharingApplication(
            candidate, forward, read, write, state_name,
            names[0], names[1], share_field, read_key_field,
            store_field, write_key_field, spans)
    return None


def _shared_value_reaches_compute(index, application, schedules, layer_index):
    source_k = ("shared_kv", "k", application.read.span)
    source_v = ("shared_kv", "v", application.read.span)
    consumers = tuple(
        (call.span, (*call.args, *(value for _name, value in call.kwargs)))
        for call in application.candidate.mechanism.compute.input_calls)

    def guard_state(guard, _cutoff):
        return _guard_value(guard, schedules, layer_index, index,
                            application.forward)

    sources, _widths, _dependencies, uncertain = \
        producer_sources_reaching_expressions(
            index, application.forward, consumers, {},
            initial_sources={
                application.key_name: source_k,
                application.value_name: source_v,
            },
            preserve_local_tuple_lanes=True,
            binding_predicate=lambda item: item != application.read,
            binding_guard_state=guard_state)
    return not uncertain and {source_k, source_v} <= set(sources)


def _guard_value(guard, schedules, layer_index, index, callable_symbol):
    result = True
    for step in guard:
        expression = step.test
        negate = False
        if step.kind == "else":
            controls = tuple(
                item for item in index.controls
                if item.enclosing_callable == callable_symbol
                and item.kind in {"if", "elif"}
                and item.span == step.span and item.controlling is not None)
            if len(controls) != 1:
                return None
            expression = controls[0].controlling
            negate = True
        value = _guard_expression(expression, schedules, layer_index)
        if value is _UNKNOWN:
            return None
        value = bool(value)
        if negate:
            value = not value
        result = result and value
        if not result:
            return False
    return result


def _guard_expression(expression, schedules, layer_index):
    if expression is None:
        return _UNKNOWN
    field = _self_field(expression)
    if field is not None and field in schedules:
        return _field_value(schedules[field], layer_index)
    if expression.kind == "constant":
        return expression.const_value
    if expression.kind == "unaryop" and expression.operator == "not" \
            and len(expression.children) == 1:
        child = _guard_expression(expression.children[0], schedules, layer_index)
        return _UNKNOWN if child is _UNKNOWN else not bool(child)
    if expression.kind == "boolop" and expression.operator in {"and", "or"}:
        values = tuple(_guard_expression(item, schedules, layer_index)
                       for item in expression.children)
        known = tuple(item for item in values if item is not _UNKNOWN)
        if expression.operator == "and":
            if any(not bool(item) for item in known):
                return False
            return True if len(known) == len(values) else _UNKNOWN
        if any(bool(item) for item in known):
            return True
        return False if len(known) == len(values) else _UNKNOWN
    return _UNKNOWN


def _field_value(schedule, layer_index):
    decision = schedule.decisions[layer_index]
    return decision.value if decision.state == "resolved" else _UNKNOWN


def _tuple_target_names(targets):
    if len(targets) != 1 or targets[0].kind not in {"tuple", "list"}:
        return ()
    names = tuple(_plain_name(item) for item in targets[0].children)
    return names if all(names) else ()


def _plain_name(expression):
    return expression.name if expression.kind == "name" and expression.name else None


def _self_field(expression):
    if expression.kind != "attribute" or not expression.name \
            or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" else None


def _positive_self_guard(guard: tuple[GuardStep, ...]):
    if len(guard) != 1 or guard[0].kind != "if":
        return None
    return _self_field(guard[0].test)


def _compute_consumes_names(calls, names):
    return any(
        all(any(_expression_contains_name(expression, name)
                for expression in (
                    *call.args,
                    *(value for _key, value in call.kwargs)))
            for name in names)
        for call in calls)


def _expression_contains_name(expression, name):
    return isinstance(expression, ExprNode) and (
        (expression.kind == "name" and expression.name == name)
        or any(_expression_contains_name(child, name)
               for child in expression.children if isinstance(child, ExprNode))
        or any(_expression_contains_name(child, name)
               for _key, child in expression.keyword_children
               if isinstance(child, ExprNode)))


def _safe_equal(left, right):
    try:
        return left == right
    except (TypeError, ValueError):
        return False


def _failed_value(owner, detail):
    return ReaderResult.failed(
        owner, (ReaderFailure("incomplete_graph", detail),))


__all__ = [
    "DecoderKVSharingSchedule",
    "KVSharingApplication",
    "decoder_kv_sharing_schedule_for_path",
]
