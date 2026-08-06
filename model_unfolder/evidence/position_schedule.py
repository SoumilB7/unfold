"""Exact per-layer selection of one already-proven position application.

The schedule is not read from a config list in isolation.  It is accepted only
after proving the construction-time index path:

``range(count) -> comprehension target -> repeated child __init__ formal ->
attention __init__ formal -> exact forward guard -> proved Q/K rotation``.

An inactive decision means only that this exact application call is disabled.
It is not a whole-position census and therefore never claims genuine NoPE.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention import exact_config_path_for_expression
from .attention_child import attention_child_evidence
from .component_owner import OwnerOccurrenceId
from .config_guard import ExactConfigGuardResolver, NormalizedConfigValue
from .construction_arguments import (
    ConstructionArgumentBinding,
    bind_construction_site,
)
from .decoder_block import DecoderBlockPath, decoder_block_path_for_config
from .models import SourceBundle
from .position_application import (
    QKHalfTurnApplicationEvidence,
    decoder_qk_half_turn_application_for_path,
)
from .position_factors import (
    PositionComplexFactorEvidence,
    PositionTrigFactorEvidence,
    decoder_position_complex_factors_for_path,
    decoder_position_trig_factors_for_path,
)
from .position_geometry import (
    PositionApplicationGeometryEvidence,
    decoder_position_application_geometry_for_path,
)
from .program_index import (
    ComprehensionObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class LayerIndexTransportEvidence:
    """Exact construction chain carrying the comprehension index to attention."""

    block_binding: ConstructionArgumentBinding
    attention_binding: ConstructionArgumentBinding
    comprehension: ComprehensionObservation
    count_expression: ExprNode
    count_config_path: tuple[str, ...]
    count_source_kind: str
    layer_count: int
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_binding, ConstructionArgumentBinding) \
                or not isinstance(
                    self.attention_binding, ConstructionArgumentBinding):
            raise TypeError("index transport carries exact constructor bindings")
        if self.attention_binding.parent_occurrence \
                != self.block_binding.child_occurrence:
            raise ValueError("attention construction descends from the exact block")
        if not isinstance(self.comprehension, ComprehensionObservation) \
                or len(self.comprehension.clauses) != 1:
            raise TypeError("index transport carries one exact comprehension")
        clause = self.comprehension.clauses[0]
        if clause.async_flag or clause.filters \
                or clause.target.kind != "name" \
                or self.block_binding.actual.kind != "name" \
                or self.block_binding.actual.name != clause.target.name:
            raise ValueError("block actual is the exact comprehension index binding")
        if self.attention_binding.actual.kind != "name" \
                or self.attention_binding.actual.name \
                != self.block_binding.formal.name:
            raise ValueError("attention actual is the exact block index formal")
        if not isinstance(self.count_expression, ExprNode) \
                or self.count_expression != clause.iterable:
            raise ValueError("count expression is the exact comprehension iterable")
        if not self.count_config_path or any(
                not isinstance(part, str) or not part
                for part in self.count_config_path):
            raise ValueError("layer count cites an exact config path")
        if self.count_source_kind not in {"config_declared", "class_default"}:
            raise ValueError("layer count has typed config provenance")
        if isinstance(self.layer_count, bool) \
                or not isinstance(self.layer_count, int) \
                or self.layer_count <= 0:
            raise ValueError("layer count is a positive integer")
        required = {
            self.block_binding.site.span,
            self.attention_binding.site.span,
            self.comprehension.span,
            self.count_expression.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("index transport retains every exact boundary span")


@dataclass(frozen=True)
class PositionApplicationLayerDecision:
    """Whether one exact application call executes at one proved layer index."""

    layer_index: int
    state: str                    # active | inactive

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, bool) \
                or not isinstance(self.layer_index, int) \
                or self.layer_index < 0:
            raise ValueError("position decision has a non-negative index")
        if self.state not in {"active", "inactive"}:
            raise ValueError("position decision has a closed state")


@dataclass(frozen=True)
class PositionApplicationScheduleEvidence:
    """A complete exact-index schedule for one positively-proven operation."""

    block_path: DecoderBlockPath
    attention_occurrence: OwnerOccurrenceId
    transport: LayerIndexTransportEvidence
    application: QKHalfTurnApplicationEvidence
    factor: PositionTrigFactorEvidence | PositionComplexFactorEvidence
    geometry: PositionApplicationGeometryEvidence
    decisions: tuple[PositionApplicationLayerDecision, ...]
    selector_config_paths: tuple[tuple[str, ...], ...]
    selector_config_values: tuple[
        tuple[tuple[str, ...], str, object], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_path, DecoderBlockPath) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("position schedule is exact-owner qualified")
        if self.attention_occurrence != self.application.attention_occurrence \
                or self.attention_occurrence != self.geometry.owner_occurrence:
            raise ValueError("application and geometry belong to one attention")
        if self.transport.block_binding.child_occurrence \
                != self.block_path.block_occurrence \
                or self.transport.attention_binding.child_occurrence \
                != self.attention_occurrence:
            raise ValueError("index transport joins the exact decoder path")
        if not isinstance(
                self.factor,
                (PositionTrigFactorEvidence, PositionComplexFactorEvidence)) \
                or self.factor.application != self.application \
                or self.geometry.application != self.application:
            raise ValueError("factor and geometry cite the canonical application")
        expected = tuple(range(self.transport.layer_count))
        if tuple(item.layer_index for item in self.decisions) != expected \
                or not self.decisions \
                or not any(item.state == "active" for item in self.decisions):
            raise ValueError("schedule covers every index and has a positive witness")
        if tuple(dict.fromkeys(self.selector_config_paths)) \
                != self.selector_config_paths \
                or any(not path for path in self.selector_config_paths):
            raise ValueError("selector paths are exact and unique")
        if tuple(dict.fromkeys(self.selector_config_values)) \
                != self.selector_config_values \
                or any(path not in self.selector_config_paths
                       or kind not in {"config_declared", "class_default"}
                       for path, kind, _value in self.selector_config_values):
            raise ValueError("selector values carry exact typed paths")
        required = {
            self.application.application_call.span,
            *self.transport.spans,
            *self.factor.spans,
            *self.geometry.spans,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("schedule provenance closes every joined boundary")


def decoder_position_application_schedule_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector,
) -> ReaderResult[PositionApplicationScheduleEvidence]:
    """Prove every index selecting one exact Q/K position application."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("position schedule requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("position schedule requires a SourceBundle")
    block_result = decoder_block_path_for_config(
        index, bundle, tuple(config_path), allow_root_stage=allow_root_stage)
    if block_result.status != "resolved":
        return _forward_failure(block_result, "decoder block address")
    block = block_result.value
    root = block.component_root
    attention_result = attention_child_evidence(
        index, root, block.block_occurrence)
    if attention_result.status != "resolved":
        return _forward_failure(attention_result, "attention child address")
    attention = attention_result.value
    transport = _index_transport(
        index, root, block, attention.compute_occurrence, config_selector)
    if isinstance(transport, ReaderFailure):
        return ReaderResult.failed(attention.compute_occurrence, (transport,))

    parameter = transport.attention_binding.formal.name
    applications = []
    for layer_index in range(transport.layer_count):
        result = decoder_qk_half_turn_application_for_path(
            index, bundle, tuple(config_path),
            allow_root_stage=allow_root_stage,
            config_selector=config_selector,
            constructor_parameter_values={parameter: layer_index})
        applications.append(result)
    resolved = tuple(item.value for item in applications
                     if item.status == "resolved")
    if not resolved:
        return ReaderResult.failed(attention.compute_occurrence, (ReaderFailure(
            "incomplete_graph", "no layer positively proves the application"),))
    canonical = resolved[0]
    identity = _application_identity(canonical)
    if any(_application_identity(item) != identity for item in resolved):
        return ReaderResult.failed(attention.compute_occurrence, (ReaderFailure(
            "conflict", "layer indices select rival position applications"),))

    node = root.graph.node_for(attention.compute_occurrence)
    decisions = []
    selector_paths = []
    selector_kinds = []
    selector_spans = []
    for layer_index, result in enumerate(applications):
        resolver = ExactConfigGuardResolver(
            index, node, config_selector,
            config_prefix=tuple(getattr(root, "config_path", ()) or ()),
            parameter_values={parameter: layer_index})
        enabled = resolver.enabled(
            canonical.application_call.guard,
            canonical.application_call.enclosing_callable)
        if enabled is None:
            return ReaderResult.failed(attention.compute_occurrence, (
                ReaderFailure(
                    "unsupported_syntax",
                    f"layer {layer_index} guard is not exactly evaluable"),))
        if enabled and (
                result.status != "resolved"
                or _application_identity(result.value) != identity):
            return ReaderResult.failed(attention.compute_occurrence, (
                ReaderFailure(
                    "incomplete_graph",
                    f"layer {layer_index} guard selects an unproved application"),))
        if not enabled and result.status == "resolved" \
                and _application_identity(result.value) == identity:
            return ReaderResult.failed(attention.compute_occurrence, (
                ReaderFailure(
                    "conflict",
                    f"layer {layer_index} resolves through an inactive guard"),))
        decisions.append(PositionApplicationLayerDecision(
            layer_index, "active" if enabled else "inactive"))
        selector_paths.extend(resolver.paths)
        selector_kinds.extend(resolver.source_kinds)
        selector_spans.extend(resolver.spans)

    representative = next(item.layer_index for item in decisions
                          if item.state == "active")
    parameter_values = {parameter: representative}
    if canonical.rotation_protocol == "complex_pair":
        factor_result = decoder_position_complex_factors_for_path(
            index, bundle, tuple(config_path),
            allow_root_stage=allow_root_stage,
            config_selector=config_selector,
            constructor_parameter_values=parameter_values)
    else:
        factor_result = decoder_position_trig_factors_for_path(
            index, bundle, tuple(config_path),
            allow_root_stage=allow_root_stage,
            config_selector=config_selector,
            constructor_parameter_values=parameter_values)
    geometry_result = decoder_position_application_geometry_for_path(
        index, bundle, tuple(config_path),
        allow_root_stage=allow_root_stage,
        config_selector=config_selector,
        constructor_parameter_values=parameter_values)
    if factor_result.status != "resolved":
        return _forward_failure(factor_result, "position factor provenance")
    if geometry_result.status != "resolved":
        return _forward_failure(geometry_result, "position geometry")
    # Reuse the representative reader result itself so all nested DTOs share
    # exact object equality in the schedule closure.
    canonical = applications[representative].value
    factor = factor_result.value
    geometry = geometry_result.value
    if factor.application != canonical or geometry.application != canonical:
        return ReaderResult.failed(attention.compute_occurrence, (ReaderFailure(
            "conflict", "representative application proofs do not coincide"),))
    values = []
    for path, kind in dict.fromkeys(selector_kinds):
        selected = _selected_value(config_selector, path)
        if selected is None or selected[0] != kind:
            return ReaderResult.failed(attention.compute_occurrence, (ReaderFailure(
                "conflict", "selector provenance changed during schedule proof"),))
        values.append((path, kind, selected[1]))
    if any(isinstance(value, tuple)
           and len(value) != transport.layer_count
           for _path, _kind, value in values):
        return ReaderResult.failed(attention.compute_occurrence, (ReaderFailure(
            "conflict",
            "an indexed selector sequence does not match the proved layer count"),))
    paths = tuple(dict.fromkeys(selector_paths))
    spans = tuple(dict.fromkeys((
        *transport.spans, *canonical.spans, *factor.spans, *geometry.spans,
        *selector_spans,
    )))
    value = PositionApplicationScheduleEvidence(
        block, attention.compute_occurrence, transport, canonical,
        factor, geometry, tuple(decisions), paths, tuple(values), spans)
    return ReaderResult.resolved(
        attention.compute_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=tuple(dict.fromkeys((
                transport.count_config_path, *paths))),
            detail=("exact construction index transport selects one proved "
                    "position application independently at every layer")),))


def _index_transport(index, root, block, attention_occurrence, selector):
    proofs = block.repeated_child.proofs
    if len(proofs) != 1:
        return ReaderFailure(
            "incomplete_graph", "one repeated-child proof is required")
    proof = proofs[0]
    block_site = proof.template.element_template
    block_bindings = bind_construction_site(
        index, root, proof.model_stage, block_site)
    if block_bindings.status not in {"resolved", "partial"}:
        return ReaderFailure(
            "incomplete_graph", "block constructor arguments are not exact")
    comprehensions = tuple(
        item for item in index.comprehensions_in(block_site.enclosing_callable)
        if item.span is not None
        and len(item.outputs) == 1
        and item.outputs[0].span == block_site.constructor.span
        and len(item.clauses) == 1)
    if len(comprehensions) != 1:
        return ReaderFailure(
            "incomplete_graph", "block construction has no unique comprehension")
    comprehension = comprehensions[0]
    clause = comprehension.clauses[0]
    if clause.async_flag or clause.filters or clause.target.kind != "name":
        return ReaderFailure(
            "unsupported_syntax", "layer comprehension is not a direct range")
    block_index = tuple(
        item for item in block_bindings.bindings
        if item.actual.kind == "name"
        and item.actual.name == clause.target.name)
    if len(block_index) != 1:
        return ReaderFailure(
            "incomplete_graph", "comprehension index has no exact block formal")
    if attention_occurrence.sites[:-1] != block.block_occurrence.sites:
        return ReaderFailure(
            "unsupported_syntax", "attention index crosses an unproved owner hop")
    attention_site = next(
        (item for item in index.construction_sites
         if item.site_id == attention_occurrence.sites[-1]), None)
    if attention_site is None:
        return ReaderFailure(
            "incomplete_graph", "attention construction site is unavailable")
    attention_bindings = bind_construction_site(
        index, root, block.block_occurrence, attention_site)
    if attention_bindings.status not in {"resolved", "partial"}:
        return ReaderFailure(
            "incomplete_graph", "attention constructor arguments are not exact")
    attention_index = tuple(
        item for item in attention_bindings.bindings
        if item.actual.kind == "name"
        and item.actual.name == block_index[0].formal.name)
    if len(attention_index) != 1:
        return ReaderFailure(
            "incomplete_graph", "block index has no exact attention formal")
    count = clause.iterable
    if count.kind != "call" or len(count.children) != 2 \
            or count.children[0].kind != "name" \
            or count.children[0].name != "range" \
            or _name_shadowed(
                index, block_site.enclosing_callable, "range"):
        return ReaderFailure(
            "unsupported_syntax", "layer comprehension is not exact builtin range")
    count_value = count.children[1]
    stage_node = root.graph.node_for(proof.model_stage)
    path = exact_config_path_for_expression(
        index, stage_node, count_value,
        config_prefix=tuple(getattr(root, "config_path", ()) or ()))
    selected = _selected_value(selector, path) if path is not None else None
    if selected is None or isinstance(selected[1], bool) \
            or not isinstance(selected[1], int) or selected[1] <= 0:
        return ReaderFailure(
            "incomplete_graph", "layer count is not exact positive config evidence")
    spans = tuple(dict.fromkeys((
        *block_index[0].spans, *attention_index[0].spans,
        comprehension.span, clause.target.span, count.span, count_value.span,
    )))
    return LayerIndexTransportEvidence(
        block_index[0], attention_index[0], comprehension, count,
        path, selected[0], selected[1], spans)


def _name_shadowed(index, callable_symbol, name):
    return any(item.name == name and item.context in {
        "parameter", "store", "del"}
        for item in index.identifiers_in(callable_symbol)) or any(
        item.name == name
        for item in index.module_bindings_in(callable_symbol.source))


def _selected_value(selector, path):
    if selector is None or path is None:
        return None
    selected = selector(path)
    if isinstance(selected, NormalizedConfigValue):
        return None
    kind = "config_declared"
    if isinstance(selected, tuple) and len(selected) in {2, 3} \
            and isinstance(selected[0], bool):
        present, value = selected[:2]
        if len(selected) == 3:
            kind = selected[2]
    else:
        present, value = selected is not None, selected
    if not present or kind not in {"config_declared", "class_default"}:
        return None
    return kind, _freeze(value)


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _application_identity(value):
    return (
        value.application_call.span, value.helper_callable,
        value.rotation_callable, value.rotation_protocol,
        value.factor_parameter_indices)


def _forward_failure(result, boundary):
    if result.status == "ambiguous":
        return ReaderResult.ambiguous(result.owner, result.ambiguity)
    failures = result.failures or (ReaderFailure(
        "incomplete_graph", f"{boundary} is {result.status}"),)
    return ReaderResult.failed(
        result.owner, failures, provenance=result.provenance)


__all__ = [
    "LayerIndexTransportEvidence",
    "PositionApplicationLayerDecision",
    "PositionApplicationScheduleEvidence",
    "decoder_position_application_schedule_for_path",
]
