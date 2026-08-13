"""Exact initialization of the stored frequency base used by applied RoPE.

The position-application/factor readers already prove that one exact stored
owner state is multiplied by a coordinate and that its factors rotate the
exact Q/K score operands.  This boundary answers a narrower, independent
question: which exact source/config value initializes that stored state?

It recognizes an execution protocol, never a class/field/model spelling:

* the factor phase reaches one exact ``self.<state>`` field;
* the constructor registers that field from lane 0 of one exact local
  initializer call;
* straight-line/guard evaluation selects one exact same-class static helper,
  or one exact callable from an imported literal framework registry;
* the helper's first return is ``1 / (base ** exponent)``; and
* ``base`` is an exact config value, a framework-normalized value retaining
  its original operands, or an exact source literal default.

An external/dynamic initializer, unresolved guard, rival buffer, arbitrary
positive config value, or unused familiar field remains unknown.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .attention import exact_config_path_for_expression
from .component_owner import OwnerOccurrenceId
from .config_guard import ExactConfigGuardResolver
from .config_guard import NormalizedConfigValue
from .construction_calls import resolve_import_reference
from .framework_config import (
    FrameworkConfigChildRelay,
    framework_config_child_relay,
)
from .position_factors import (
    PositionComplexFactorEvidence,
    PositionTrigFactorEvidence,
)
from .position_schedule import (
    PositionApplicationScheduleEvidence,
    decoder_position_application_schedule_for_path,
)
from .program_index import (
    BindingObservation,
    CallObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class PositionFrequencyInitializationEvidence:
    """One exact applied factor's stored-state initialization proof."""

    schedule: PositionApplicationScheduleEvidence
    producer_occurrence: OwnerOccurrenceId
    producer_init: SymbolId
    stored_field: str
    buffer_call: CallObservation
    initialization_binding: BindingObservation
    initializer_call: CallObservation
    initializer_callable: SymbolId
    initializer_kind: str
    selector_config_path: tuple[str, ...] | None
    selector_value: object | None
    base_origin_kind: str
    base_expression: ExprNode
    base_config_path: tuple[str, ...]
    base_value: int | float
    base_dependencies: tuple[
        tuple[tuple[str, ...], str, object], ...]
    config_dependencies: tuple[
        tuple[tuple[str, ...], str, object], ...]
    config_address_relay: FrameworkConfigChildRelay | None
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        factor = self.schedule.factor
        if not isinstance(
                factor,
                (PositionTrigFactorEvidence, PositionComplexFactorEvidence)):
            raise TypeError("frequency initialization carries exact factor evidence")
        if self.producer_occurrence != factor.producer_occurrence:
            raise ValueError("frequency initialization belongs to the factor producer")
        if not isinstance(self.producer_init, SymbolId) \
                or not isinstance(self.initializer_callable, SymbolId):
            raise TypeError("frequency initialization carries exact callables")
        if self.producer_init.source != self.producer_occurrence.root.source \
                or self.initializer_callable.source.component_key \
                != self.producer_init.source.component_key:
            raise ValueError("frequency initialization stays in one component")
        if self.initializer_kind not in {"local_default", "imported_registry"}:
            raise ValueError("frequency initializer kind is closed")
        if self.initializer_kind == "local_default":
            if self.initializer_callable.source != self.producer_init.source \
                    or self.selector_config_path is not None \
                    or self.selector_value is not None:
                raise ValueError("local default initialization has no registry selector")
        elif not self.selector_config_path or self.selector_value is None \
                or not any(path == self.selector_config_path \
                           and value == self.selector_value
                           for path, _kind, value in self.config_dependencies):
            raise ValueError("imported initialization retains its exact selector")
        if self.base_origin_kind not in {
                "direct_config", "normalized_config", "code_default"}:
            raise ValueError("frequency-base origin kind is closed")
        if not self.stored_field:
            raise ValueError("frequency initialization names one stored field")
        if not isinstance(self.buffer_call, CallObservation) \
                or not isinstance(self.initializer_call, CallObservation) \
                or not isinstance(self.initialization_binding, BindingObservation):
            raise TypeError("frequency initialization carries exact calls/binding")
        if self.buffer_call.enclosing_callable != self.producer_init \
                or self.initializer_call.enclosing_callable != self.producer_init:
            raise ValueError("buffer and initializer calls belong to the constructor")
        if self.initialization_binding.enclosing_callable != self.producer_init \
                or self.initialization_binding.value is None \
                or self.initialization_binding.value.span \
                != self.initializer_call.span:
            raise ValueError("initializer binding owns the exact initializer call")
        buffer_field = _direct_self_field(self.buffer_call.callee)
        if buffer_field != "register_buffer" \
                or len(self.buffer_call.args) < 2 \
                or self.buffer_call.args[0].kind != "constant" \
                or self.buffer_call.args[0].const_value != self.stored_field:
            raise ValueError("frequency initialization carries the exact buffer write")
        buffer_value = self.buffer_call.args[1]
        if buffer_value.kind != "name" or not buffer_value.name \
                or _exact_lane_zero_name(
                    self.initialization_binding.targets) != buffer_value.name:
            raise ValueError("buffer state is exact lane zero of the initializer")
        if self.initializer_call.callee.kind != "name" \
                or not self.initializer_call.callee.name:
            raise ValueError("frequency initializer is one exact local call")
        if self.initializer_kind == "local_default":
            producer_class = self.producer_init.qualified_name.rsplit(
                ".__init__", 1)[0]
            initializer_class = self.initializer_callable.qualified_name.rsplit(
                ".", 1)[0]
            if producer_class == self.producer_init.qualified_name \
                    or initializer_class != producer_class:
                raise ValueError("frequency initializer is an exact same-class helper")
        if not isinstance(self.base_expression, ExprNode) \
                or self.base_expression.span is None \
                or self.base_expression.span.source != self.initializer_callable.source:
            raise TypeError("frequency base is an exact source expression")
        if not self.base_config_path or any(
                not isinstance(part, str) or not part
                for part in self.base_config_path):
            raise ValueError("frequency base cites one exact config path")
        if isinstance(self.base_value, bool) \
                or not isinstance(self.base_value, (int, float)) \
                or self.base_value <= 0:
            raise ValueError("frequency base is a positive numeric value")
        direct = any(path == self.base_config_path
                     and value == self.base_value
                     for path, _kind, value in self.base_dependencies)
        if self.base_origin_kind == "direct_config" and not direct:
            raise ValueError(
                "a direct frequency base cites its exact input dependencies")
        if self.base_origin_kind == "normalized_config" \
                and (direct or not self.base_dependencies):
            raise ValueError(
                "a normalized frequency base retains distinct source inputs")
        if self.base_origin_kind == "code_default" \
                and self.base_dependencies:
            raise ValueError("a code-default frequency base has no config input")
        if self.initializer_kind == "imported_registry" \
                and self.base_origin_kind == "code_default":
            raise ValueError("an external registry base cannot become a code default")
        if any(item not in self.config_dependencies
               for item in self.base_dependencies):
            raise ValueError("frequency base premises are part of all dependencies")
        if len({(path, kind) for path, kind, _value
                in self.config_dependencies}) \
                != len(self.config_dependencies) \
                or any(kind not in {"config_declared", "class_default"}
                       for _path, kind, _value in self.config_dependencies):
            raise ValueError("frequency dependencies are exact, typed and unique")
        if self.config_address_relay is not None:
            if not isinstance(
                    self.config_address_relay, FrameworkConfigChildRelay) \
                    or self.config_address_relay.child_occurrence \
                    != self.producer_occurrence:
                raise ValueError(
                    "frequency address relay belongs to the exact producer")
        required = {
            factor.producer_invocation.call.span,
            self.buffer_call.span,
            self.initialization_binding.span,
            self.initializer_call.span,
            self.base_expression.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("frequency initialization retains every exact boundary")


def decoder_position_frequency_initialization_for_path(
    index: ProgramIndex,
    bundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector,
) -> ReaderResult[PositionFrequencyInitializationEvidence]:
    """Prove the config base initializing one already-applied position factor."""
    schedule = decoder_position_application_schedule_for_path(
        index, bundle, tuple(config_path),
        allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if schedule.status != "resolved" or schedule.value is None:
        return ReaderResult.failed(schedule.owner, (ReaderFailure(
            "incomplete_graph",
            f"position schedule is {schedule.status}; initialization is unavailable"),))
    return position_frequency_initialization(
        index, schedule.value, config_selector=config_selector)


def position_frequency_initialization(
    index: ProgramIndex,
    schedule: PositionApplicationScheduleEvidence,
    *,
    config_selector,
) -> ReaderResult[PositionFrequencyInitializationEvidence]:
    if not isinstance(index, ProgramIndex):
        raise TypeError("frequency initialization requires a ProgramIndex")
    if not isinstance(schedule, PositionApplicationScheduleEvidence):
        raise TypeError("frequency initialization requires an exact schedule")
    factor = schedule.factor
    root = schedule.block_path.component_root
    owner = factor.producer_occurrence
    node = root.graph.node_for(owner)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return _failed(owner, "out_of_owner",
                       "the factor producer does not round-trip through its graph")
    init = SymbolId(node.symbol.source, f"{node.symbol.qualified_name}.__init__")
    if index.callable_by_symbol(init) is None:
        return _failed(owner, "incomplete_graph",
                       "the exact factor-producer constructor is unavailable")

    # Some framework models store the constructor config through an external
    # base and pass ``self.config`` into the rotary child.  Compose that already
    # typed framework proof for this exact child only; do not mutate the owner
    # graph or infer an address from the conventional field spelling.
    address_relay = None
    if not node.config_bindings:
        relay_result = framework_config_child_relay(index, root, owner)
        if relay_result.status == "resolved" and relay_result.value is not None:
            address_relay = relay_result.value
            node = replace(
                node, config_bindings=(address_relay.child_binding,),
                config_prefix_candidates=(
                    address_relay.child_binding.resolved_prefix,))

    stored_field = _stored_phase_field(index, factor)
    if stored_field is None:
        return _failed(owner, "incomplete_graph",
                       "the applied phase has no unique exact stored-state field")
    from .rope_config_normalization import rope_config_normalized_selector
    effective_selector = rope_config_normalized_selector(
        index, node, config_selector,
        config_prefix=tuple(getattr(root, "config_path", ()) or ()))
    selected = _selected_initializer(
        index, node, init, stored_field, effective_selector,
        tuple(getattr(root, "config_path", ()) or ()))
    if isinstance(selected, ReaderFailure):
        return ReaderResult.failed(owner, (selected,))
    (buffer_call, initialization_binding, initializer_call,
     initializer_callable, initializer_kind, selector_path, selector_value,
     guard_dependencies, guard_spans) = selected

    helper = index.callable_by_symbol(initializer_callable)
    if helper is None or (
            initializer_kind == "local_default" and not any(
                decorator.kind == "name" and decorator.name == "staticmethod"
                for decorator in helper.decorators)):
        return _failed(owner, "unsupported_syntax",
                       "the selected initializer callable is unavailable")
    helper_params = tuple(
        param for param in helper.params
        if param.kind in {"positional", "posonly", "keyword_only"})
    if not helper_params or not initializer_call.args:
        return _failed(owner, "incomplete_graph",
                       "the selected initializer has no explicit config argument")
    config_actual = initializer_call.args[0]
    config_field = _direct_self_field(config_actual)
    if config_field is None:
        return _failed(owner, "unsupported_syntax",
                       "the initializer config actual is not one exact self field")
    constructor_parameter = _stored_config_parameter(
        index, node, init, config_field)
    if constructor_parameter is None:
        return _failed(owner, "incomplete_graph",
                       "the initializer config actual has no exact constructor binding")
    if initializer_kind == "local_default" \
            and helper_params[0].name != constructor_parameter:
        # The current exact-path join is parameter-qualified.  A differently
        # named helper formal needs an explicit argument-binding node rather
        # than a spelling bridge.
        return _failed(owner, "unsupported_syntax",
                       "the helper config formal is not the bound constructor formal")

    base = (
        _default_frequency_base(index, initializer_callable)
        if initializer_kind == "local_default" else
        _registry_frequency_base(index, initializer_callable))
    if base is None:
        return _failed(owner, "incomplete_graph",
                       "the selected initializer has no exact inverse-power base")
    base_expression, base_path = base
    binding = next((item for item in node.config_bindings
                    if item.parameter == constructor_parameter), None)
    if binding is None:
        return _failed(owner, "incomplete_graph",
                       "the initializer config has no unique owner binding")
    exact_base_path = (
        binding.resolved_path(base_path)
        if initializer_kind == "local_default" else
        (*tuple(getattr(root, "config_path", ()) or ()), *base_path))
    if exact_base_path is None:
        return _failed(owner, "incomplete_graph",
                       "the inverse-power base has no unique owner config address")
    selected_base = _selected_value_with_dependencies(
        effective_selector, exact_base_path)
    if selected_base is None or isinstance(selected_base[0], bool) \
            or not isinstance(selected_base[0], (int, float)) \
            or selected_base[0] <= 0:
        return _failed(owner, "incomplete_graph",
                       "the exact inverse-power base has no positive typed value")
    base_value, base_dependencies, base_spans, base_origin_kind = selected_base

    # Both local and imported default initializers can read optional
    # ``config.rope_parameters`` operands that change the returned frequency
    # lane (most importantly partial_rotary_factor).  The base proof above is
    # deliberately narrow; enumerate the rest from the exact selected helper
    # so those architectural inputs are neither dropped nor inferred by name.
    registry_dependencies = _registry_config_dependencies(
        index, initializer_callable, effective_selector,
        tuple(getattr(root, "config_path", ()) or ()),
        require_mapping=initializer_kind == "imported_registry")
    if isinstance(registry_dependencies, ReaderFailure):
        return ReaderResult.failed(owner, (registry_dependencies,))
    dependencies = _unique_dependencies((
        *base_dependencies,
        *guard_dependencies,
        *registry_dependencies,
    ))
    if dependencies is None:
        return _failed(owner, "conflict",
                       "one config dependency has conflicting exact values")
    spans = tuple(dict.fromkeys(span for span in (
        *factor.spans,
        buffer_call.span,
        initialization_binding.span,
        initializer_call.span,
        helper.span,
        base_expression.span,
        *base_spans,
        *guard_spans,
    ) if isinstance(span, SourceSpan)))
    value = PositionFrequencyInitializationEvidence(
        schedule, owner, init, stored_field, buffer_call,
        initialization_binding, initializer_call, initializer_callable,
        initializer_kind, selector_path, selector_value,
        base_origin_kind, base_expression, exact_base_path, base_value,
        base_dependencies,
        dependencies,
        address_relay,
        spans)
    provenance_kind = "code_and_config" if dependencies else "source"
    return ReaderResult.resolved(owner, value, provenance=(
        ReaderProvenance(
            provenance_kind, spans=spans,
            config_paths=tuple(path for path, _kind, _value in dependencies),
            detail=("exact selected initializer returns an inverse-power "
                    "frequency base stored into the applied position factor")),))


def _selected_initializer(index, node, init, stored_field, selector, prefix):
    buffer_calls = tuple(
        call for call in index.calls_in(init)
        if _direct_self_field(call.callee) == "register_buffer"
        and not call.guard and len(call.args) >= 2
        and call.args[0].kind == "constant"
        and call.args[0].const_value == stored_field)
    if len(buffer_calls) != 1:
        return ReaderFailure(
            "incomplete_graph", "the stored state has no unique exact buffer write")
    buffer_call = buffer_calls[0]
    state = buffer_call.args[1]
    if state.kind != "name" or not state.name:
        return ReaderFailure(
            "unsupported_syntax", "the buffer value is not one exact local lane")
    init_bindings = tuple(
        item for item in index.bindings_in(init)
        if item.span is not None and _span_before(item.span, buffer_call.span)
        and item.value is not None and item.value.kind == "call"
        and _exact_lane_zero_name(item.targets) == state.name)
    if len(init_bindings) != 1:
        return ReaderFailure(
            "incomplete_graph", "the buffer lane has no unique initializer binding")
    initialization = init_bindings[0]
    calls = tuple(call for call in index.calls_in(init)
                  if call.span == initialization.value.span)
    if len(calls) != 1 or initialization.guard:
        return ReaderFailure(
            "unsupported_syntax", "the initializer call is not exact and unconditional")
    call = calls[0]
    if call.callee.kind != "name" or not call.callee.name:
        return ReaderFailure(
            "unsupported_syntax", "the initializer callee is not one local binding")
    alias = call.callee.name
    assignments = tuple(sorted((
        item for item in index.bindings_in(init)
        if item.span is not None and _span_before(item.span, call.span)
        and len(item.targets) == 1
        and item.targets[0].kind == "name"
        and item.targets[0].name == alias and item.value is not None),
        key=lambda item: _span_key(item.span)))
    if not assignments:
        return ReaderFailure(
            "incomplete_graph", "the local initializer binding is unavailable")
    chosen = None
    dependencies = []
    spans = []
    for assignment in assignments:
        enabled = True
        if assignment.guard:
            resolver = ExactConfigGuardResolver(
                index, node, selector, config_prefix=prefix)
            enabled = resolver.enabled(assignment.guard, init)
            dependencies.extend(
                (path, kind, _selected_value(selector, path, kind))
                for path, kind in resolver.source_kinds)
            spans.extend(resolver.spans)
        if enabled is None:
            return ReaderFailure(
                "unsupported_syntax", "initializer selection guard is unresolved")
        if enabled:
            chosen = assignment.value
    method = _direct_self_field(chosen) if chosen is not None else None
    selector_path = None
    selector_value = None
    if method is not None:
        helper = SymbolId(
            node.symbol.source, f"{node.symbol.qualified_name}.{method}")
        initializer_kind = "local_default"
        if index.callable_by_symbol(helper) is None:
            return ReaderFailure(
                "external_unavailable",
                "the selected initializer helper is not indexed")
    else:
        registry = _selected_imported_registry_callable(
            index, node, init, chosen, selector, prefix)
        if isinstance(registry, ReaderFailure):
            return registry
        helper, selector_path, selector_value, registry_span = registry
        initializer_kind = "imported_registry"
        spans.append(registry_span)
        selected_selector = selector(selector_path)
        dependencies.append((
            selector_path, selected_selector[2], selected_selector[1]))
    if any(value is _MISSING for _path, _kind, value in dependencies):
        return ReaderFailure(
            "incomplete_graph", "initializer selector value is unavailable")
    dependencies = _unique_dependencies(dependencies)
    if dependencies is None:
        return ReaderFailure(
            "conflict", "initializer selector has conflicting values")
    return (buffer_call, initialization, call, helper, initializer_kind,
            selector_path, selector_value,
            dependencies, tuple(dict.fromkeys(spans)))


def _selected_imported_registry_callable(
        index, node, init, expression, selector, prefix):
    """Resolve ``IMPORTED_REGISTRY[exact config selector]`` to one callable.

    This is an address/dataflow join only.  The registry key carries no RoPE
    semantics; the selected callable must independently prove its parameter
    reads and inverse-frequency return.
    """
    if not isinstance(expression, ExprNode) or expression.kind != "subscript" \
            or len(expression.children) != 2:
        return ReaderFailure(
            "external_unavailable",
            "the enacted initializer is neither local nor an exact registry entry")
    registry_expr, key_expr = expression.children
    if registry_expr.kind != "name" or not registry_expr.name:
        return ReaderFailure(
            "unsupported_syntax", "initializer registry is not one exact import")
    import_proof = resolve_import_reference(
        index, node.symbol.source, init, registry_expr)
    if import_proof is None:
        return ReaderFailure(
            "external_unavailable", "initializer registry import is unresolved")
    target_parts = import_proof.qualified_target.split(".")
    if len(target_parts) < 2:
        return ReaderFailure(
            "external_unavailable", "initializer registry target is incomplete")
    registry_name = target_parts[-1]
    module_leaf = target_parts[-2]
    registries = tuple(
        record for record in index.dispatch_registries
        if record.symbol.qualified_name == registry_name
        and record.symbol.source.component_key == node.symbol.source.component_key
        and record.symbol.source.canonical_path.rsplit("/", 1)[-1]
        == f"{module_leaf}.py")
    if len(registries) != 1:
        return ReaderFailure(
            "external_unavailable", "imported initializer registry is not indexed")
    selector_path = _exact_owner_config_path(
        index, node, key_expr, prefix)
    selected = selector(selector_path) if selector_path is not None else None
    if not isinstance(selected, tuple) or len(selected) != 3 \
            or not selected[0] \
            or selected[2] not in {"config_declared", "class_default"}:
        return ReaderFailure(
            "incomplete_graph", "initializer registry selector is unavailable")
    selector_value = selected[1]
    matches = tuple(
        value for key, value in registries[0].entries
        if key.kind == "constant" and key.const_value == selector_value)
    if len(matches) != 1 or matches[0].kind != "name" \
            or not matches[0].name:
        return ReaderFailure(
            "external_unavailable", "initializer registry has no unique selected entry")
    helper = SymbolId(registries[0].symbol.source, matches[0].name)
    if index.callable_by_symbol(helper) is None:
        return ReaderFailure(
            "external_unavailable", "selected registry callable is not indexed")
    return helper, selector_path, selector_value, matches[0].span


def _default_frequency_base(index, helper):
    returns = tuple(item for item in index.return_observations_in(helper)
                    if not item.guard and item.value is not None)
    if len(returns) != 1 or returns[0].value.kind not in {"tuple", "list"} \
            or not returns[0].value.children:
        return None
    expression = _resolve_local(
        index, helper, returns[0].value.children[0], returns[0].span,
        frozenset())
    if expression is None or expression.kind != "binop" \
            or expression.operator != "/" or len(expression.children) != 2 \
            or not _is_one(expression.children[0]):
        return None
    power = _resolve_local(
        index, helper, expression.children[1], expression.span, frozenset())
    if power is None or power.kind != "binop" or power.operator != "**" \
            or len(power.children) != 2:
        return None
    base = _resolve_local(
        index, helper, power.children[0], power.span, frozenset())
    record = index.callable_by_symbol(helper)
    params = tuple(param for param in record.params
                   if param.kind in {"positional", "posonly", "keyword_only"})
    if base is None or not params:
        return None
    relative = _config_segments(base, params[0].name)
    if relative is None or not relative:
        return None
    return base, relative


def _registry_frequency_base(index, helper):
    """Prove an imported initializer's returned lane depends on one base.

    The framework helper may be substantially more complex than the local
    default helper.  We do not reimplement it: one exact mapping alias must
    address ``config.rope_parameters``; one literal subscript is the base; and
    the first returned lane must reach that binding through an exponentiation.
    """
    record = index.callable_by_symbol(helper)
    if record is None:
        return None
    params = tuple(param for param in record.params
                   if param.kind in {"positional", "posonly", "keyword_only"})
    if not params:
        return None
    mapping_names = _registry_parameter_mapping_names(
        index, helper, params[0].name)
    if len(mapping_names) != 1:
        return None
    mapping_name = next(iter(mapping_names))
    bases = []
    for binding in index.bindings_in(helper):
        target = _single_name_target(binding)
        value = binding.value
        if target is None or binding.guard or value is None \
                or value.kind != "subscript" or len(value.children) != 2:
            continue
        receiver, key = value.children
        if receiver.kind == "name" and receiver.name == mapping_name \
                and key.kind == "constant" \
                and isinstance(key.const_value, str) and key.const_value:
            depends, powered = _return_lane_depends_on_name(
                index, helper, target)
            if depends and powered:
                bases.append((value, ("rope_parameters", key.const_value)))
    return bases[0] if len(bases) == 1 else None


def _registry_config_dependencies(
        index, helper, selector, prefix, *, require_mapping=True):
    """Present exact config operands read by the selected framework helper."""
    record = index.callable_by_symbol(helper)
    if record is None:
        return ReaderFailure(
            "external_unavailable", "selected registry callable is unavailable")
    params = tuple(param for param in record.params
                   if param.kind in {"positional", "posonly", "keyword_only"})
    mapping_names = (
        _registry_parameter_mapping_names(index, helper, params[0].name)
        if params else frozenset())
    if len(mapping_names) != 1:
        if require_mapping or len(mapping_names) > 1:
            return ReaderFailure(
                "unsupported_syntax",
                "selected registry callable has no exact config-parameter mapping")
    mapping_name = next(iter(mapping_names), None)
    paths: dict[tuple[str, ...], bool] = {}
    for binding in index.bindings_in(helper):
        value = binding.value
        if value is None or value.kind != "subscript" \
                or len(value.children) != 2:
            continue
        receiver, key = value.children
        if mapping_name is not None \
                and receiver.kind == "name" and receiver.name == mapping_name \
                and key.kind == "constant" \
                and isinstance(key.const_value, str) and key.const_value:
            paths[(*prefix, "rope_parameters", key.const_value)] = True
    for call in index.calls_in(helper):
        callee = call.callee
        if callee.kind != "attribute" or callee.name != "get" \
                or len(callee.children) != 1 or not call.args:
            continue
        receiver = callee.children[0]
        key = call.args[0]
        receiver_is_mapping = (
            mapping_name is not None
            and receiver.kind == "name" and receiver.name == mapping_name)
        receiver_is_config_mapping = (
            receiver.kind == "attribute" and receiver.name == "rope_parameters"
            and len(receiver.children) == 1
            and receiver.children[0].kind == "name"
            and receiver.children[0].name == params[0].name)
        if (receiver_is_mapping or receiver_is_config_mapping) \
                and key.kind == "constant" \
                and isinstance(key.const_value, str) and key.const_value:
            paths.setdefault(
                (*prefix, "rope_parameters", key.const_value), False)

    out = []
    for path, required in paths.items():
        selected = _optional_selected(selector, path)
        if not isinstance(selected, tuple) or len(selected) != 3 \
                or not selected[0]:
            if required:
                return ReaderFailure(
                    "incomplete_graph",
                    f"required initializer operand {'.'.join(path)} is unavailable")
            continue
        if selected[2] not in {"config_declared", "class_default"}:
            return ReaderFailure(
                "incomplete_graph",
                f"initializer operand {'.'.join(path)} has unknown provenance")
        out.append((path, selected[2], selected[1]))
    return tuple(out)


def _optional_selected(selector, path):
    """Query one optional operand without converting unrelated errors to absence."""
    try:
        return selector(path)
    except (KeyError, AttributeError, IndexError):
        return None


def _registry_parameter_mapping_names(index, helper, config_formal):
    """Names proven to be the non-layered ``config.rope_parameters`` map."""
    out = set()
    for binding in index.bindings_in(helper):
        target = _single_name_target(binding)
        value = binding.value
        if target is None or binding.guard or value is None:
            continue
        if _is_config_rope_parameters(value, config_formal):
            out.add(target)
            continue
        # Framework helpers support an optional per-layer map.  When that
        # formal has literal ``None`` as its default, the exact two-argument
        # constructor call selects the else lane below.
        if value.kind == "ifexp" and len(value.children) == 3 \
                and _is_config_rope_parameters(value.children[2], config_formal):
            condition = value.children[1]
            if condition.kind == "compare" and condition.operator == "is not" \
                    and len(condition.children) == 2 \
                    and condition.children[1].kind == "constant" \
                    and condition.children[1].const_value is None:
                formal = condition.children[0]
                record = index.callable_by_symbol(helper)
                param = next((item for item in record.params
                              if formal.kind == "name"
                              and item.name == formal.name), None)
                if param is not None and param.has_default \
                        and param.default is not None \
                        and param.default.kind == "constant" \
                        and param.default.const_value is None:
                    out.add(target)
    return frozenset(out)


def _is_config_rope_parameters(expression, formal):
    return expression.kind == "attribute" \
        and expression.name == "rope_parameters" \
        and len(expression.children) == 1 \
        and expression.children[0].kind == "name" \
        and expression.children[0].name == formal


def _single_name_target(binding):
    if len(binding.targets) != 1 or binding.targets[0].kind != "name" \
            or not binding.targets[0].name:
        return None
    return binding.targets[0].name


def _return_lane_depends_on_name(index, helper, target):
    returns = tuple(item for item in index.return_observations_in(helper)
                    if not item.guard and item.value is not None)
    if len(returns) != 1 or returns[0].value.kind not in {"tuple", "list"} \
            or not returns[0].value.children:
        return False, False
    return _expression_depends_on_name(
        index, helper, returns[0].value.children[0], returns[0].span,
        target, frozenset())


def _expression_depends_on_name(
        index, helper, expression, before, target, seen):
    if not isinstance(expression, ExprNode):
        return False, False
    powered_here = expression.kind == "binop" and expression.operator == "**"
    if expression.kind == "name" and expression.name:
        if expression.name == target:
            return True, powered_here
        key = (expression.name, before)
        if key in seen:
            return False, False
        definitions = tuple(
            item for item in index.bindings_in(helper)
            if not item.guard and item.value is not None
            and item.span is not None and _span_before(item.span, before)
            and _single_name_target(item) == expression.name)
        if len(definitions) != 1:
            return False, False
        return _expression_depends_on_name(
            index, helper, definitions[0].value, definitions[0].span,
            target, seen | {key})
    found = False
    powered = False
    for child in (*expression.children,
                  *(value for _name, value in expression.keyword_children)):
        child_found, child_powered = _expression_depends_on_name(
            index, helper, child, expression.span or before, target, seen)
        found = found or child_found
        powered = powered or child_powered
    return found, powered or (found and powered_here)


def _exact_owner_config_path(index, node, expression, prefix):
    return exact_config_path_for_expression(
        index, node, expression, config_prefix=tuple(prefix))


def _stored_phase_field(index, factor):
    fields = _self_fields(
        index, factor.producer_callable, factor.phase_expression,
        factor.phase_expression.span, frozenset())
    return next(iter(fields)) if len(fields) == 1 else None


def _self_fields(index, callable_symbol, expression, before, seen):
    expression = _resolve_local(index, callable_symbol, expression, before, seen)
    if expression is None:
        return set()
    field = _direct_self_field(expression)
    if field is not None:
        return {field}
    out = set()
    for child in expression.children:
        if isinstance(child, ExprNode):
            out |= _self_fields(
                index, callable_symbol, child, expression.span, seen)
    for _name, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            out |= _self_fields(
                index, callable_symbol, child, expression.span, seen)
    return out


def _resolve_local(index, callable_symbol, expression, before, seen):
    current = expression
    current_before = before
    local_seen = set(seen)
    while isinstance(current, ExprNode) and current.kind == "name":
        key = (current.name, current_before)
        if key in local_seen:
            return None
        local_seen.add(key)
        matches = tuple(
            item for item in index.bindings_in(callable_symbol)
            if not item.guard and item.value is not None
            and item.span is not None and _span_before(item.span, current_before)
            and len(item.targets) == 1
            and item.targets[0].kind == "name"
            and item.targets[0].name == current.name)
        if not matches:
            return current
        latest = max(matches, key=lambda item: _span_key(item.span))
        current = latest.value
        current_before = latest.span
    return current


def _stored_config_parameter(index, node, init, field):
    assignments = tuple(
        item for item in index.field_assigns_of(node.symbol)
        if item.enclosing_callable == init and item.field == field
        and not item.guard and item.value.kind == "name")
    if len(assignments) != 1:
        return None
    parameter = assignments[0].value.name
    matches = tuple(binding for binding in node.config_bindings
                    if binding.parameter == parameter
                    and binding.resolved_prefix is not None)
    return parameter if len(matches) == 1 else None


def _config_segments(expression, formal):
    if expression.kind == "name":
        return () if expression.name == formal else None
    if expression.kind == "attribute" and len(expression.children) == 1 \
            and expression.name:
        prefix = _config_segments(expression.children[0], formal)
        return (*prefix, expression.name) if prefix is not None else None
    if expression.kind == "subscript" and len(expression.children) == 2:
        base, key = expression.children
        prefix = _config_segments(base, formal)
        if prefix is None or key.kind != "constant" \
                or not isinstance(key.const_value, str) or not key.const_value:
            return None
        return (*prefix, key.const_value)
    return None


def _direct_self_field(expression):
    if not isinstance(expression, ExprNode) or expression.kind != "attribute" \
            or len(expression.children) != 1:
        return None
    receiver = expression.children[0]
    return (expression.name if receiver.kind == "name"
            and receiver.name == "self" else None)


def _exact_lane_zero_name(targets):
    """Return lane zero only for an explicit tuple/list destructure.

    A scalar assignment is not evidence that the helper's first return lane
    initializes the registered state; treating it as lane zero would silently
    strengthen a different return protocol into the one proved here.
    """
    if len(targets) != 1 or targets[0].kind not in {"tuple", "list"}:
        return None
    target = targets[0]
    if not target.children or target.children[0].kind != "name" \
            or not target.children[0].name:
        return None
    names = tuple(
        child.name for child in target.children
        if child.kind == "name" and child.name == target.children[0].name)
    return target.children[0].name if len(names) == 1 else None


def _selected_value(selector, path, kind):
    result = selector(path)
    if isinstance(result, tuple) and len(result) == 3 \
            and result[0] and result[2] == kind:
        return result[1]
    return _MISSING


def _selected_value_with_dependencies(selector, path):
    result = selector(path)
    if isinstance(result, NormalizedConfigValue):
        dependencies = tuple(
            (dependency, kind, _selected_value(selector, dependency, kind))
            for dependency, kind in result.dependencies)
        if any(value is _MISSING for _path, _kind, value in dependencies):
            return None
        return (result.value, dependencies, result.spans,
                "normalized_config" if dependencies else "code_default")
    if isinstance(result, tuple) and len(result) == 3 and result[0] \
            and result[2] in {"config_declared", "class_default"}:
        return (result[1], ((tuple(path), result[2], result[1]),), (),
                "direct_config")
    return None


def _unique_dependencies(values):
    out = []
    positions = {}
    for item in values:
        key = (item[0], item[1])
        if key in positions:
            if out[positions[key]][2] != item[2]:
                return None
            continue
        positions[key] = len(out)
        out.append(item)
    return tuple(out)


def _is_one(expression):
    return expression.kind == "constant" \
        and not isinstance(expression.const_value, bool) \
        and isinstance(expression.const_value, (int, float)) \
        and expression.const_value == 1


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) <= \
        (right.line, right.col)


def _span_key(span):
    return (span.source.canonical_path, span.line, span.col,
            span.end_line or span.line, span.end_col or span.col)


def _failed(owner, kind, detail):
    return ReaderResult.failed(owner, (ReaderFailure(kind, detail),))


_MISSING = object()


__all__ = [
    "PositionFrequencyInitializationEvidence",
    "decoder_position_frequency_initialization_for_path",
    "position_frequency_initialization",
]
