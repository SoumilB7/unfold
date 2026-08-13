"""Exact framework-mask construction reaching an exact repeated block.

This U8 boundary is deliberately narrower than a mask *schedule*.  It proves
that an exact, import-resolved framework mask builder authored the value bound
to an exact repeated block formal.  If a source constructs several masks and
selects one dynamically (for example by a per-layer table), every rival is
retained and the result is incomplete.  A later schedule reader may resolve
that selector; this reader never assigns semantics to its raw token.

The protocol table below is framework API semantics, not model identity.  A
local function with the same spelling does not qualify, and model/class/field
names never participate in classification.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .attention_storage import producer_sources_reaching_expressions
from .attention_score_additives import (
    AttentionScoreAdditiveInventory,
    EquivalentAttentionScoreAdditiveInventory,
    ExplicitAttentionScoreAdditiveApplication,
    decoder_attention_score_additives_for_path,
)
from .attention_child import attention_child_positive_census
from .attention import exact_config_path_for_expression
from .call_arguments import (
    CallArgumentBinding,
    bind_addressed_invocation,
    bind_repeated_child_call,
)
from .component_owner import (
    ConfigBinding,
    OwnerOccurrenceId,
    resolve_child_config_bindings,
)
from .container_inventory import resolve_container_inventory
from .config_guard import ExactConfigGuardResolver, NormalizedConfigValue
from .framework_config import (
    FrameworkConfigAlias,
    FrameworkConfigDefaultValue,
    config_path_from_framework_alias,
    framework_config_alias,
    framework_config_default_selector,
)
from .construction_calls import ExternalReferenceProof, resolve_import_reference
from .cross_attention_replacement import attention_input_lineage_for_child
from .decoder_block import decoder_block_candidates_for_config
from .execution_flow import resolve_addressed_invocations
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    CallObservation,
    ConstructionSite,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .self_method_return import (
    SelfMethodReturnLane,
    SelfMethodReturnTransport,
    resolve_self_method_return_transport,
)


_MASK_PROTOCOLS = {
    "...masking_utils.create_causal_mask": "causal",
    "...masking_utils.create_bidirectional_mask": "bidirectional",
    "...masking_utils.create_sliding_window_causal_mask": "sliding_causal",
    "transformers.masking_utils.create_causal_mask": "causal",
    "transformers.masking_utils.create_bidirectional_mask": "bidirectional",
    "transformers.masking_utils.create_sliding_window_causal_mask":
        "sliding_causal",
    "...masking_utils.create_chunked_causal_mask": "chunked_causal",
    "transformers.masking_utils.create_chunked_causal_mask": "chunked_causal",
    "...masking_utils.create_bidirectional_sliding_window_mask":
        "sliding_bidirectional",
    "transformers.masking_utils.create_bidirectional_sliding_window_mask":
        "sliding_bidirectional",
}

_MASK_GEOMETRY_FIELDS = {
    "...masking_utils.create_sliding_window_causal_mask": "sliding_window",
    "transformers.masking_utils.create_sliding_window_causal_mask":
        "sliding_window",
    "...masking_utils.create_bidirectional_sliding_window_mask":
        "sliding_window",
    "transformers.masking_utils.create_bidirectional_sliding_window_mask":
        "sliding_window",
    "...masking_utils.create_chunked_causal_mask": "attention_chunk_size",
    "transformers.masking_utils.create_chunked_causal_mask":
        "attention_chunk_size",
}


@dataclass(frozen=True)
class MaskBuilderEvidence:
    """One exact import-resolved framework mask-builder call."""

    mechanism: str
    import_proof: ExternalReferenceProof
    call: CallObservation
    definition: BindingObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.mechanism not in set(_MASK_PROTOCOLS.values()):
            raise ValueError("mask builder has a closed mechanism vocabulary")
        if not isinstance(self.import_proof, ExternalReferenceProof) \
                or _MASK_PROTOCOLS.get(self.import_proof.qualified_target) \
                != self.mechanism:
            raise ValueError("mask semantics derive from an exact framework import")
        if not isinstance(self.call, CallObservation) \
                or not isinstance(self.definition, BindingObservation) \
                or self.definition.value is None:
            raise TypeError("mask evidence carries an exact call and definition")
        if self.call.enclosing_callable != self.definition.enclosing_callable \
                or not _expr_contains_span(self.definition.value, self.call.span):
            raise ValueError("the mask call is inside its exact definition")
        required = {self.call.span, self.definition.span}
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("mask-builder provenance retains call and definition")


@dataclass(frozen=True)
class MaskBlockApplication:
    """Mask producers reaching one exact formal of one exact block call."""

    block_occurrence: OwnerOccurrenceId
    binding: CallArgumentBinding
    builders: tuple[MaskBuilderEvidence, ...]
    source_uncertain: bool
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.binding, CallArgumentBinding):
            raise TypeError("a mask application is exact-block qualified")
        if self.binding.callee_occurrence != self.block_occurrence:
            raise ValueError("the mask binding targets the exact block occurrence")
        if not self.builders or any(
                not isinstance(item, MaskBuilderEvidence)
                for item in self.builders):
            raise ValueError("a mask application retains every reaching builder")
        if tuple(sorted(self.builders, key=lambda item: _span_key(item.call.span))) \
                != self.builders or len(set(self.builders)) != len(self.builders):
            raise ValueError("reaching mask builders are source-ordered and unique")
        required = {
            self.binding.call.span,
            self.binding.actual.span,
            *(span for item in self.builders for span in item.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("mask-application provenance closes the exact join")

    @property
    def mechanism(self) -> str | None:
        """The unique proven mechanism, or ``None`` for a retained selector."""
        mechanisms = {item.mechanism for item in self.builders}
        if not self.source_uncertain and len(self.builders) == 1 \
                and len(mechanisms) == 1:
            return next(iter(mechanisms))
        return None


@dataclass(frozen=True)
class AttentionMaskMechanismInventory:
    """Every exact framework mask value reaching repeated decoder blocks."""

    stage_occurrence: OwnerOccurrenceId
    stage_forward: SymbolId
    producer_occurrence: OwnerOccurrenceId
    producer_forward: SymbolId
    builders: tuple[MaskBuilderEvidence, ...]
    helper_transports: tuple[SelfMethodReturnTransport, ...]
    stage_input_bindings: tuple[CallArgumentBinding, ...]
    applications: tuple[MaskBlockApplication, ...]
    config_dependencies: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.stage_forward, SymbolId):
            raise TypeError("mask inventory is exact-stage qualified")
        if self.stage_forward.source != self.stage_occurrence.root.source \
                or not self.stage_forward.qualified_name.endswith(".forward"):
            raise ValueError("mask inventory names the exact stage forward")
        if not isinstance(self.producer_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.producer_forward, SymbolId) \
                or self.producer_forward.source \
                != self.producer_occurrence.root.source \
                or not self.producer_forward.qualified_name.endswith(".forward"):
            raise ValueError("mask inventory names the exact producer forward")
        if not self.builders or not self.applications:
            raise ValueError("mask inventory is a positive producer/application join")
        if any(not isinstance(item, SelfMethodReturnTransport)
               or item.owner_occurrence != self.producer_occurrence
               or item.caller.symbol != self.producer_forward
               for item in self.helper_transports):
            raise ValueError("mask helper transports belong to the exact producer")
        if self.producer_occurrence == self.stage_occurrence:
            if self.producer_forward != self.stage_forward \
                or self.stage_input_bindings:
                raise ValueError(
                    "the exact stage forward is also the local producer and "
                    "carries no parent-to-stage bindings")
        elif not self.stage_input_bindings:
            raise ValueError(
                "an upstream producer carries exact parent-to-stage bindings")
        if any(not isinstance(item, CallArgumentBinding)
               or item.call.enclosing_callable != self.producer_forward
               or item.callee_occurrence != self.stage_occurrence
               or item.callee_callable != self.stage_forward
               for item in self.stage_input_bindings):
            raise ValueError(
                "stage-input bindings close the exact producer-to-stage call")
        helper_symbols = {item.helper.symbol for item in self.helper_transports}
        if any(item.call.enclosing_callable not in {
                self.producer_forward, *helper_symbols} for item in self.builders):
            raise ValueError(
                "every mask builder belongs to the exact producer or helper")
        used_helpers = {
            item.call.enclosing_callable for item in self.builders
            if item.call.enclosing_callable != self.producer_forward}
        if used_helpers != helper_symbols:
            raise ValueError("every carried helper authors an inventoried builder")
        known = set(self.builders)
        if any(not set(item.builders) <= known for item in self.applications):
            raise ValueError("applications may cite only inventoried builders")
        if len({(item.block_occurrence, item.binding.formal.name)
                for item in self.applications}) != len(self.applications):
            raise ValueError("each exact block formal has one application record")
        if any(not isinstance(path, tuple) or not path or any(
                not isinstance(part, str) or not part for part in path)
                or kind not in {"config_declared", "class_default"}
                for path, kind in self.config_dependencies) \
                or tuple(dict.fromkeys(self.config_dependencies)) \
                != self.config_dependencies:
            raise ValueError("mask inventory dependencies are exact and typed")
        required = {
            *(span for item in self.builders for span in item.spans),
            *(span for item in self.helper_transports for span in item.spans),
            *(item.call.span for item in self.stage_input_bindings),
            *(item.actual.span for item in self.stage_input_bindings),
            *(span for item in self.applications for span in item.spans),
        }
        if not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("mask inventory retains all producer/application spans")


@dataclass(frozen=True)
class MaskScoreApplication:
    """One exact block-mask formal reaching one exact score-additive operand."""

    mask_application: MaskBlockApplication
    attention_occurrence: OwnerOccurrenceId
    owner_transport: tuple[CallArgumentBinding, ...]
    compute_callable: SymbolId
    compute_entry_call: CallObservation
    compute_formal: ParamRecord
    compute_actual: ExprNode | None
    score_application: ExplicitAttentionScoreAdditiveApplication
    conditional: bool
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mask_application, MaskBlockApplication) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId) \
                or not self.owner_transport or any(
                    not isinstance(item, CallArgumentBinding)
                    for item in self.owner_transport) \
                or not isinstance(self.compute_callable, SymbolId) \
                or not isinstance(self.compute_entry_call, CallObservation) \
                or not isinstance(self.compute_formal, ParamRecord) \
                or not isinstance(
                    self.score_application,
                    ExplicitAttentionScoreAdditiveApplication):
            raise TypeError("mask score evidence closes typed application boundaries")
        if self.owner_transport[0].call.enclosing_callable \
                != self.mask_application.binding.callee_callable \
                or self.owner_transport[-1].callee_occurrence \
                != self.attention_occurrence:
            raise ValueError("owner transport closes block forward to exact attention")
        for previous, current in zip(
                self.owner_transport, self.owner_transport[1:]):
            if previous.callee_callable != current.call.enclosing_callable:
                raise ValueError("owner transport is one exact invocation chain")
        if self.compute_entry_call.enclosing_callable \
                != self.owner_transport[-1].callee_callable:
            raise ValueError("compute entry belongs to the exact attention forward")
        if self.compute_actual is None:
            if self.compute_callable \
                    != self.owner_transport[-1].callee_callable \
                    or self.compute_formal \
                    != self.owner_transport[-1].formal:
                raise ValueError("direct compute reuses the exact attention formal")
        elif self.compute_callable \
                == self.owner_transport[-1].callee_callable \
                or self.compute_actual not in (
                    *self.compute_entry_call.args,
                    *(value for _name, value in self.compute_entry_call.kwargs)):
            raise ValueError("helper compute carries its exact entry actual")
        if self.score_application.application.enclosing_callable \
                != self.compute_callable:
            raise ValueError("the additive application belongs to exact attention forward")
        required = {
            self.mask_application.binding.call.span,
            *(item.call.span for item in self.owner_transport),
            *(item.actual.span for item in self.owner_transport),
            self.compute_entry_call.span,
            self.score_application.additive_operand.span,
            self.score_application.application.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("mask score provenance retains every exact join")


@dataclass(frozen=True)
class AttentionMaskScoreInventory:
    """Exact mask builders whose values reach exact attention score additions."""

    mechanisms: AttentionMaskMechanismInventory
    applications: tuple[MaskScoreApplication, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mechanisms, AttentionMaskMechanismInventory) \
                or not self.applications:
            raise ValueError("score inventory joins a positive mask inventory")
        known = set(self.mechanisms.applications)
        if any(item.mask_application not in known for item in self.applications):
            raise ValueError("score applications cite inventoried mask lanes only")
        identities = tuple((
            item.mask_application.block_occurrence,
            item.mask_application.binding.formal.name,
            item.attention_occurrence,
            item.score_application.application.span,
        ) for item in self.applications)
        if len(set(identities)) != len(identities):
            raise ValueError("score applications are exact-lane unique")
        required = {
            *(span for item in self.applications for span in item.spans),
            *self.mechanisms.spans,
        }
        if not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("score inventory retains every joined source span")


@dataclass(frozen=True)
class MaskStageConfigAddress:
    """Exact config-prefix transport from mask producer to repeated stage."""

    stage_occurrence: OwnerOccurrenceId
    producer_occurrence: OwnerOccurrenceId
    framework_config: FrameworkConfigAlias
    binding: ConfigBinding
    construction_site: ConstructionSite | None
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.producer_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.framework_config, FrameworkConfigAlias) \
                or not isinstance(self.binding, ConfigBinding):
            raise TypeError("a stage config address is exact and typed")
        if self.framework_config.owner_occurrence != self.producer_occurrence \
                or self.stage_occurrence.root != self.producer_occurrence.root:
            raise ValueError("the config transport stays in one exact owner graph")
        if self.stage_occurrence == self.producer_occurrence:
            if self.construction_site is not None \
                    or self.binding != self.framework_config.config_binding:
                raise ValueError("a local stage reuses its framework config binding")
        elif not isinstance(self.construction_site, ConstructionSite) \
                or not self.stage_occurrence.sites \
                or self.construction_site.site_id \
                != self.stage_occurrence.sites[-1] \
                or self.construction_site.owner \
                != self.framework_config.owner_symbol:
            raise ValueError(
                "an upstream config address cites the exact stage construction")
        required = {
            *self.framework_config.spans,
            *((self.construction_site.span,)
              if self.construction_site is not None else ()),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("stage config transport retains every address span")


@dataclass(frozen=True)
class MaskLayerDecision:
    """One config sequence element selecting one already-proven mask builder."""

    layer_index: int
    selector_value: object
    builder: MaskBuilderEvidence

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, bool) or not isinstance(
                self.layer_index, int) or self.layer_index < 0:
            raise ValueError("a mask-layer decision has a non-negative index")
        if not isinstance(self.builder, MaskBuilderEvidence):
            raise TypeError("a mask-layer decision selects exact builder evidence")


@dataclass(frozen=True)
class AttentionMaskLayerSchedule:
    """Exact per-layer selector over already-proven score-applied masks."""

    score_inventory: AttentionMaskScoreInventory
    application: MaskScoreApplication
    config_address: "MaskStageConfigAddress"
    selector_path: tuple[str, ...]
    count_path: tuple[str, ...]
    selector_source_kind: str
    count_source_kind: str
    decisions: tuple[MaskLayerDecision, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.score_inventory, AttentionMaskScoreInventory) \
                or not isinstance(self.application, MaskScoreApplication) \
                or self.application not in self.score_inventory.applications:
            raise ValueError("a mask schedule cites one exact score-applied lane")
        if not isinstance(self.config_address, MaskStageConfigAddress) \
                or self.config_address.stage_occurrence \
                != self.score_inventory.mechanisms.stage_occurrence:
            raise ValueError("a mask schedule carries the exact stage config address")
        for path in (self.selector_path, self.count_path):
            if not path or any(not isinstance(part, str) or not part for part in path):
                raise ValueError("mask schedule paths are exact tuple[str, ...]")
        if self.selector_source_kind not in {"config_declared", "class_default"} \
                or self.count_source_kind not in {
                    "config_declared", "class_default"}:
            raise ValueError("mask schedule operands have typed config provenance")
        if not self.decisions or tuple(
                item.layer_index for item in self.decisions) \
                != tuple(range(len(self.decisions))):
            raise ValueError("mask schedule decisions cover exact contiguous indices")
        known = set(self.application.mask_application.builders)
        if any(item.builder not in known for item in self.decisions):
            raise ValueError("mask schedule selects only reaching exact builders")
        required = {
            *self.config_address.spans,
            self.application.mask_application.binding.actual.span,
            *(item.builder.call.span for item in self.decisions),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("mask schedule retains selector/config/builder provenance")

    @property
    def framework_config(self) -> FrameworkConfigAlias:
        return self.config_address.framework_config


@dataclass(frozen=True)
class UniformAttentionMaskLayerSchedule:
    """One score-applied mask builder repeated for every exact layer slot."""

    score_inventory: AttentionMaskScoreInventory
    application: MaskScoreApplication
    config_address: "MaskStageConfigAddress"
    count_path: tuple[str, ...]
    count_source_kind: str
    decisions: tuple[MaskLayerDecision, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.score_inventory, AttentionMaskScoreInventory) \
                or not isinstance(self.application, MaskScoreApplication) \
                or self.application not in self.score_inventory.applications:
            raise ValueError("a uniform mask schedule cites one exact score lane")
        if not isinstance(self.config_address, MaskStageConfigAddress) \
                or self.config_address.stage_occurrence \
                != self.score_inventory.mechanisms.stage_occurrence:
            raise ValueError("a uniform schedule carries the exact stage config address")
        mask = self.application.mask_application
        if mask.mechanism is None or len(mask.builders) != 1:
            raise ValueError("a uniform schedule needs one unconditional builder")
        if not self.count_path or any(
                not isinstance(part, str) or not part
                for part in self.count_path):
            raise ValueError("a uniform mask count has one exact config path")
        if self.count_source_kind not in {
                "config_declared", "class_default"}:
            raise ValueError("a uniform mask count has typed config provenance")
        if not self.decisions or tuple(
                item.layer_index for item in self.decisions) \
                != tuple(range(len(self.decisions))):
            raise ValueError("uniform mask decisions cover contiguous layer slots")
        builder = mask.builders[0]
        if any(item.builder != builder or item.selector_value is not None
               for item in self.decisions):
            raise ValueError("every uniform decision cites the one exact builder")
        required = {
            *self.score_inventory.spans,
            *self.config_address.spans,
            self.application.mask_application.binding.actual.span,
            builder.call.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("uniform mask schedule retains exact provenance")

    @property
    def framework_config(self) -> FrameworkConfigAlias:
        return self.config_address.framework_config


@dataclass(frozen=True)
class AttentionMaskGeometry:
    """Exact framework mask geometry bound to the exact stage config object."""

    schedule: AttentionMaskLayerSchedule | UniformAttentionMaskLayerSchedule
    builder: MaskBuilderEvidence
    config_actual: ExprNode
    config_path: tuple[str, ...]
    source_kind: str
    value: int
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, (
                AttentionMaskLayerSchedule,
                UniformAttentionMaskLayerSchedule)) \
                or not isinstance(self.builder, MaskBuilderEvidence) \
                or self.builder not in {
                    item.builder for item in self.schedule.decisions}:
            raise ValueError("mask geometry belongs to an exact schedule builder")
        field = _MASK_GEOMETRY_FIELDS.get(
            self.builder.import_proof.qualified_target)
        if field is None or not self.config_path \
                or self.config_path[-1] != field:
            raise ValueError("mask geometry path comes from the framework protocol")
        if not isinstance(self.config_actual, ExprNode) \
                or _attribute_chain(self.config_actual) != (
                    "self", self.schedule.framework_config.stored_field):
            raise ValueError("mask geometry uses the exact framework config object")
        if self.source_kind not in {"config_declared", "class_default"}:
            raise ValueError("mask geometry retains typed value provenance")
        if isinstance(self.value, bool) or not isinstance(self.value, int) \
                or self.value < 1:
            raise ValueError("mask geometry is a positive integer")
        required = {
            *self.schedule.spans,
            self.builder.call.span,
            self.config_actual.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("mask geometry retains schedule/call/config provenance")


@dataclass(frozen=True)
class AttentionMaskExecution:
    """One exact enacted layer schedule plus every required geometry operand."""

    schedule: AttentionMaskLayerSchedule | UniformAttentionMaskLayerSchedule
    geometries: tuple[AttentionMaskGeometry, ...]
    config_dependencies: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, (
                AttentionMaskLayerSchedule,
                UniformAttentionMaskLayerSchedule)):
            raise TypeError("mask execution carries one typed exact schedule")
        if any(not isinstance(item, AttentionMaskGeometry)
               or item.schedule != self.schedule for item in self.geometries):
            raise ValueError("mask execution geometry belongs to this schedule")
        required_builders = {
            item.builder for item in self.schedule.decisions
            if item.builder.import_proof.qualified_target in _MASK_GEOMETRY_FIELDS}
        if {item.builder for item in self.geometries} != required_builders:
            raise ValueError("mask execution closes every enacted geometry protocol")
        if any(not isinstance(path, tuple) or not path or any(
                not isinstance(part, str) or not part for part in path)
                or kind not in {"config_declared", "class_default"}
                for path, kind in self.config_dependencies) \
                or tuple(dict.fromkeys(self.config_dependencies)) \
                != self.config_dependencies:
            raise ValueError("mask execution dependencies are exact and typed")
        by_path = {}
        for path, kind in self.config_dependencies:
            if path in by_path and by_path[path] != kind:
                raise ValueError("one mask dependency cannot carry rival provenance")
            by_path[path] = kind
        required = {
            *self.schedule.spans,
            *(span for item in self.geometries for span in item.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("mask execution retains schedule/geometry provenance")


def decoder_attention_mask_mechanisms_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[AttentionMaskMechanismInventory]:
    """Prove exact framework mask builders reaching exact repeated blocks."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("mask evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("mask evidence requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    candidates = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return candidates
    root = candidates.value.component_root
    stage = candidates.value.stage_occurrence
    node = root.graph.node_for(stage)
    if node is None:
        return _failed(stage, "out_of_owner", "stage is absent from its D0 graph")
    forwards = tuple(
        item for item in index.callables
        if item.owner == node.symbol
        and item.symbol.qualified_name.endswith(".forward"))
    if len(forwards) != 1:
        return _failed(
            stage, "callable_unavailable",
            f"expected one exact stage forward, found {len(forwards)}")
    forward = forwards[0].symbol
    producer_occurrence = stage
    producer_forward = forward
    stage_input_candidates = ()
    direct_builders, helper_transports, helper_builders = \
        _mask_evidence_in_scope(index, root, stage, forward)
    if not direct_builders and not helper_builders:
        upstream, upstream_failure = _exact_parent_stage_call(
            index, root, stage, forward)
        if upstream_failure:
            return _failed(stage, "incomplete_graph", upstream_failure)
        if upstream is not None:
            producer_occurrence, producer_forward, stage_input_candidates = upstream
            direct_builders, helper_transports, helper_builders = \
                _mask_evidence_in_scope(
                    index, root, producer_occurrence, producer_forward)
    builders = tuple(sorted(
        (*direct_builders, *helper_builders),
        key=lambda item: _span_key(item.call.span)))
    if not builders:
        return ReaderResult.absent(
            stage, provenance=(ReaderProvenance(
                "derived", detail=(
                    "no exact framework mask builder at the stage or its exact "
                    "immediate invocation owner")),))

    producer_node = root.graph.node_for(producer_occurrence)
    if producer_node is None:
        return _failed(
            stage, "out_of_owner", "mask producer is absent from its D0 graph")
    config_alias_result = framework_config_alias(
        index, root, producer_occurrence)
    config_alias = (config_alias_result.value
                    if config_alias_result.status == "resolved" else None)
    selector = config_selector if config_selector is not None else _missing_selector
    if config_alias is not None:
        selector = framework_config_default_selector(
            index, config_alias, selector, config_prefix=config_path)
    guard = ExactConfigGuardResolver(
        index, producer_node, selector,
        config_prefix=config_path, framework_config_alias=config_alias)
    applications = []
    failures = []
    producer_calls = {item: item.call for item in direct_builders}
    # Evaluate only the guards that decide the inventoried producers.  Feeding
    # every unrelated binding guard through the resolver would attach unrelated
    # config paths to this fact and let an irrelevant loop mark it incomplete.
    producer_guard_states = {
        item.definition.span: guard.enabled(
            item.definition.guard, item.call.enclosing_callable)
        for item in builders
    }
    lane_builders = {}
    lane_uncertainty = {}
    lane_transport = {}
    for transport in helper_transports:
        producers = {
            item: item.call for item in builders
            if item.call.enclosing_callable == transport.helper.symbol}
        for lane in transport.lanes:
            sources, _widths, _dependencies, uncertain = \
                producer_sources_reaching_expressions(
                    index, transport.helper.symbol,
                    ((transport.returned.span, (lane.returned_value,)),),
                    producers,
                    binding_guard_state=lambda steps, span: (
                        producer_guard_states[span]
                        if span in producer_guard_states
                        else (True if not steps else None)))
            if sources:
                lane_builders[lane] = sources
                lane_uncertainty[lane] = uncertain
                lane_transport[lane] = transport

    def helper_lane_states(binding, targets):
        matches = tuple(
            item for item in helper_transports
            if item.caller_definition == binding)
        if not matches:
            return None
        if len(matches) != 1 or len(matches[0].lanes) != len(targets):
            return tuple((frozenset(), True) for _target in targets)
        return tuple(
            (frozenset((lane,)), False) for lane in matches[0].lanes)

    # When the repeated stage receives its mask from its exact immediate
    # invocation owner, first close producer/helper lineage to each exact
    # parent->stage formal.  The frozen CallArgumentBinding itself is the
    # neutral source token installed at the stage entry; no spelling or role
    # participates in the join.
    stage_binding_builders = {}
    stage_binding_uncertainty = {}
    stage_binding_transports = {}
    producer_calls = {item: item.call for item in direct_builders}
    for stage_binding in stage_input_candidates:
        sources, _widths, _dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, producer_forward,
                ((stage_binding.call.span, (stage_binding.actual,)),),
                producer_calls,
                binding_lane_states=helper_lane_states,
                binding_guard_state=lambda steps, span: (
                    producer_guard_states[span]
                    if span in producer_guard_states
                    else (True if not steps else None)))
        resolved_sources = set()
        relevant_transports = set()
        for source in sources:
            if isinstance(source, MaskBuilderEvidence):
                resolved_sources.add(source)
            elif isinstance(source, SelfMethodReturnLane):
                resolved_sources.update(lane_builders.get(source, ()))
                uncertain = uncertain or lane_uncertainty.get(source, True)
                if source in lane_transport:
                    relevant_transports.add(lane_transport[source])
        if resolved_sources:
            stage_binding_builders[stage_binding] = frozenset(resolved_sources)
            stage_binding_uncertainty[stage_binding] = uncertain
            stage_binding_transports[stage_binding] = frozenset(
                relevant_transports)

    proofs = tuple(candidates.value.repeated_child.proofs)
    for proof in proofs:
        bound = bind_repeated_child_call(index, root, proof)
        if bound.status == "failed":
            failures.append(ReaderFailure(
                "incomplete_graph", bound.failure_detail, bound.call.span))
            continue
        for binding in bound.bindings:
            initial_sources = {
                item.formal.name: item for item in stage_binding_builders}
            sources, _widths, _dependencies, uncertain = \
                producer_sources_reaching_expressions(
                    index, forward,
                    ((binding.call.span, (binding.actual,)),),
                    ({item: item.call for item in direct_builders}
                     if producer_occurrence == stage else {}),
                    initial_sources=initial_sources,
                    binding_lane_states=(
                        helper_lane_states
                        if producer_occurrence == stage else None),
                    binding_guard_state=lambda steps, span: (
                        producer_guard_states[span]
                        if span in producer_guard_states
                        else (True if not steps else None)))
            if not sources:
                continue
            resolved_sources = set()
            relevant_transports = set()
            for source in sources:
                if isinstance(source, MaskBuilderEvidence):
                    resolved_sources.add(source)
                elif isinstance(source, SelfMethodReturnLane):
                    resolved_sources.update(lane_builders.get(source, ()))
                    uncertain = uncertain or lane_uncertainty.get(source, True)
                    if source in lane_transport:
                        relevant_transports.add(lane_transport[source])
                elif isinstance(source, CallArgumentBinding):
                    resolved_sources.update(
                        stage_binding_builders.get(source, ()))
                    uncertain = uncertain or stage_binding_uncertainty.get(
                        source, True)
                    relevant_transports.update(
                        stage_binding_transports.get(source, ()))
            sources = frozenset(resolved_sources)
            if not sources:
                continue
            if uncertain:
                # The neutral reaching-def helper intentionally refuses to
                # choose between guarded rewrites but retains only its latest
                # environment version.  For this inventory, preserve every
                # inventoried producer definition of the exact actual lane;
                # this is a rival census, never a selection.
                actual_names = _expression_names(binding.actual)
                sources = frozenset((
                    *sources,
                    *(item for item in builders
                      if actual_names & set(_binding_target_names(
                          item.definition))),
                ))
            ordered = tuple(sorted(sources, key=lambda item: _span_key(
                item.call.span)))
            spans = tuple(dict.fromkeys((
                binding.call.span, binding.actual.span,
                *(source.call.span for source in sources
                  if isinstance(source, CallArgumentBinding)),
                *(source.actual.span for source in sources
                  if isinstance(source, CallArgumentBinding)),
                *(span for transport in relevant_transports
                  for span in transport.spans),
                *(span for item in ordered for span in item.spans),
            )))
            applications.append(MaskBlockApplication(
                proof.child_occurrence, binding, ordered, uncertain, spans))

    if not applications:
        return _failed(
            stage, "incomplete_graph",
            "framework mask builders do not reach an exact repeated-block formal")
    applications = tuple(sorted(
        applications,
        key=lambda item: (
            tuple(_site_key(site) for site in item.block_occurrence.sites),
            item.binding.formal.name,
            _span_key(item.binding.call.span))))
    spans = tuple(dict.fromkeys((
        *(span for item in builders for span in item.spans),
        *(span for item in helper_transports for span in item.spans),
        *(item.call.span for item in stage_binding_builders),
        *(item.actual.span for item in stage_binding_builders),
        *(span for item in applications for span in item.spans),
        *guard.spans,
        *((config_alias.spans
           if config_alias is not None and guard.framework_alias_used else ())),
    )))
    dependencies = tuple(dict.fromkeys(guard.source_kinds))
    value = AttentionMaskMechanismInventory(
        stage, forward, producer_occurrence, producer_forward,
        builders, helper_transports, tuple(stage_binding_builders),
        applications, dependencies, spans)
    provenance = (ReaderProvenance(
        "code_and_config" if guard.paths else "source",
        spans=spans,
        config_paths=tuple(dict.fromkeys(guard.paths)),
        detail="exact framework mask builders reaching exact block formals"),)
    unresolved = tuple(item for item in applications
                       if item.mechanism is None)
    if failures or unresolved:
        details = list(failures)
        if unresolved:
            details.append(ReaderFailure(
                "incomplete_graph",
                "a block mask formal retains rival or conditional builders",
                unresolved[0].binding.actual.span))
        return ReaderResult.incomplete(
            stage, value, failures=tuple(details), provenance=provenance)
    return ReaderResult.resolved(stage, value, provenance=provenance)


def decoder_attention_mask_score_applications_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[AttentionMaskScoreInventory]:
    """Join the stage mask rail to the exact attention score consumer.

    Every owner hop comes from the positive attention-child invocation path;
    every actual is mapped to one exact callee formal.  An indexed helper
    compute is crossed only through its exact entry call and Python argument
    binding.  No descendant is selected from a class/name search.
    """
    mechanisms = decoder_attention_mask_mechanisms_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if mechanisms.status not in {"resolved", "incomplete"} \
            or mechanisms.value is None:
        return _propagate_without_value(
            mechanisms, "mask mechanism inventory is unavailable")
    additives = decoder_attention_score_additives_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if additives.status != "resolved":
        return _propagate_without_value(
            additives, "attention score-additive inventory is unavailable")
    inventories = (
        additives.value.variants
        if isinstance(additives.value, EquivalentAttentionScoreAdditiveInventory)
        else (additives.value,))
    root_candidates = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if root_candidates.status != "resolved":
        return _propagate_without_value(
            root_candidates, "decoder block address is unavailable")
    root = root_candidates.value.component_root

    joined = []
    failures = []
    for mask in mechanisms.value.applications:
        census = attention_child_positive_census(
            index, root, mask.block_occurrence)
        if census.status != "resolved":
            failures.append(ReaderFailure(
                "incomplete_graph",
                "the exact attention-child census is unavailable"))
            continue
        lane_matches = []
        reaches_selected_attention = False
        for score in inventories:
            if not isinstance(score, AttentionScoreAdditiveInventory) \
                    or score.block_occurrence != mask.block_occurrence:
                continue
            children = tuple(
                item for item in census.value.candidates
                if item.compute_occurrence == score.attention_occurrence)
            if len(children) != 1:
                continue
            child = children[0]
            if len(census.value.candidates) > 1:
                lineage = attention_input_lineage_for_child(
                    index, root, mask.block_occurrence, child)
                if lineage.status != "resolved" \
                        or lineage.value.kind != "self":
                    # Multiple exact attention children make lane identity a
                    # real choice.  Only a positively-proven self Q/K/V lane
                    # may enter this self-mask join; a replacement-cross lane
                    # (or an unresolved lineage) cannot certify or block it.
                    continue
            # With exactly one positively-proven attention child there is no
            # child-selection ambiguity to settle: the exact formal transport
            # below still has to reach that child's exact score operand.  This
            # preserves architectures whose fused projection path cannot be
            # classified by the stricter three-affine Q/K/V lineage reader.
            transported = _transport_mask_formal(
                index, root, mask, child.invocation_path)
            if transported is None:
                continue
            reaches_selected_attention = True
            owner_transport, current_formal, current_callable, token = transported
            compute = child.compute
            compute_actual = None
            compute_formal = current_formal
            if compute.callable_symbol != current_callable:
                bound_actuals = _bind_callable_actuals(
                    index, compute.callable_symbol, compute.entry_call)
                candidates_at_entry = []
                for formal, actual in bound_actuals:
                    sources, _widths, _dependencies, uncertain = \
                        producer_sources_reaching_expressions(
                            index, current_callable,
                            ((compute.entry_call.span, (actual,)),), {},
                            initial_sources={current_formal.name: token})
                    if sources == frozenset((token,)) and not uncertain:
                        candidates_at_entry.append((formal, actual))
                if len(candidates_at_entry) != 1:
                    continue
                compute_formal, compute_actual = candidates_at_entry[0]
            for score_application in score.applications:
                if not isinstance(
                        score_application,
                        ExplicitAttentionScoreAdditiveApplication):
                    continue
                consumer_sources, _widths, _dependencies, conditional = \
                    producer_sources_reaching_expressions(
                        index, compute.callable_symbol,
                        ((score_application.application.span,
                          (score_application.additive_operand,)),), {},
                        initial_sources={compute_formal.name: token})
                if consumer_sources != frozenset((token,)):
                    continue
                spans = tuple(dict.fromkeys((
                    *mask.spans,
                    *(binding.call.span for binding in owner_transport),
                    *(binding.actual.span for binding in owner_transport),
                    compute.entry_call.span,
                    *((compute_actual.span,) if compute_actual is not None else ()),
                    *score_application.spans,
                )))
                lane_matches.append(MaskScoreApplication(
                    mask, score.attention_occurrence, owner_transport,
                    compute.callable_symbol, compute.entry_call,
                    compute_formal, compute_actual,
                    score_application, conditional, spans))
        if not reaches_selected_attention:
            # A block can carry a distinct encoder/cross-attention mask formal.
            # Its producer is real, but it is outside this exact self-attention
            # score lane.  Absence of a transport path is positive lane
            # separation—not permission for that mask to certify or block the
            # selected self-attention mechanism.
            continue
        if len(lane_matches) != 1:
            failures.append(ReaderFailure(
                "incomplete_graph",
                ("the exact block mask formal has no unique "
                 "attention-score consumer"),
                mask.binding.actual.span))
        else:
            joined.extend(lane_matches)

    if not joined:
        if not failures:
            failures.append(ReaderFailure(
                "incomplete_graph",
                "no exact mask formal reaches the selected self-attention score lane"))
        return ReaderResult.failed(
            mechanisms.value.stage_occurrence, tuple(failures))
    spans = tuple(dict.fromkeys((
        *mechanisms.value.spans,
        *(span for item in joined for span in item.spans),
    )))
    value = AttentionMaskScoreInventory(
        mechanisms.value, tuple(joined), spans)
    provenance = tuple(dict.fromkeys((
        *mechanisms.provenance,
        *additives.provenance,
        ReaderProvenance(
            "source", spans=spans,
            detail="exact block mask formal reaches exact score addition"),
    )))
    relevant_incomplete = any(
        item.mask_application.mechanism is None for item in joined)
    if failures or relevant_incomplete:
        return ReaderResult.incomplete(
            mechanisms.value.stage_occurrence, value,
            failures=tuple(failures or (
                ReaderFailure(
                    "incomplete_graph",
                    "the selected self-attention mask lane is unresolved"),)),
            provenance=provenance)
    return ReaderResult.resolved(
        mechanisms.value.stage_occurrence, value, provenance=provenance)


def decoder_attention_mask_layer_schedule_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector,
) -> ReaderResult[AttentionMaskLayerSchedule]:
    """Resolve an exact enumerated config selector over proven mask builders.

    This admits only the concrete pattern the source proves: the exact repeated
    invocation selects from one literal builder dictionary using one config
    sequence indexed by the exact ``enumerate`` loop index.  Config tokens choose
    among builders whose semantics are already proven; they never name a mask.
    """
    if config_selector is None:
        raise TypeError("a mask schedule requires an owner-qualified config selector")
    scores = decoder_attention_mask_score_applications_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if scores.status not in {"resolved", "incomplete"} or scores.value is None:
        return _propagate_without_value(
            scores, "mask score-application inventory is unavailable")
    candidates = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return _propagate_without_value(
            candidates, "decoder block address is unavailable")
    root = candidates.value.component_root
    stage = candidates.value.stage_occurrence
    node = root.graph.node_for(stage)
    if node is None:
        return _failed(stage, "out_of_owner", "stage is absent from its D0 graph")
    unresolved_apps = tuple(
        item for item in scores.value.applications
        if item.mask_application.mechanism is None)
    if not unresolved_apps:
        return ReaderResult.absent(
            stage, provenance=(ReaderProvenance(
                "derived", detail="no per-layer mask selector is required"),))
    if len(unresolved_apps) != 1:
        return _failed(stage, "conflict",
                       "expected one exact unresolved mask selector lane")
    address_result = _mask_stage_config_address(
        index, root, scores.value.mechanisms)
    if address_result.status != "resolved":
        return _propagate_without_value(
            address_result, "stage config address is unavailable")
    config_address = address_result.value
    alias_result = config_address.framework_config
    if alias_result.owner_occurrence != stage:
        return _failed(
            stage, "incomplete_graph",
            "a per-layer selector requires the exact stage config alias")
    application = unresolved_apps[0]
    mask = application.mask_application
    proofs = tuple(
        proof for proof in candidates.value.repeated_child.proofs
        if proof.child_occurrence == mask.block_occurrence
        and proof.template.call == mask.binding.call)
    if len(proofs) != 1:
        return _failed(stage, "incomplete_graph",
                       "the mask call does not round-trip to one repeated template")
    proof = proofs[0]
    index_name = _enumeration_index_name(proof.template)
    if index_name is None:
        return _failed(stage, "unsupported_syntax",
                       "the mask selector is not indexed by an exact enumerate target")

    selected = _selector_shape(mask, alias_result, config_path, index_name)
    if selected is None:
        return _failed(stage, "unsupported_syntax",
                       "the exact mask actual is not a literal-map/config-sequence selector")
    selector_path, key_to_builder, selector_span = selected
    selector_value = _select_config_value(config_selector, selector_path)
    if selector_value is None:
        return _failed(stage, "incomplete_graph",
                       "the exact mask selector config sequence is unavailable")
    selector_items, selector_kind = selector_value
    if not isinstance(selector_items, (tuple, list)) or not selector_items:
        return _failed(stage, "unsupported_syntax",
                       "the exact mask selector value is not a non-empty sequence")

    count = _container_count_path(
        index, node, proof.template.container.count_expression,
        proof.template.container.record.enclosing_callable,
        config_prefix=config_path)
    if count is None:
        return _failed(stage, "unsupported_syntax",
                       "the repeated container count is not one exact config-bound range")
    count_path, count_span = count
    count_value = _select_config_value(config_selector, count_path)
    if count_value is None:
        return _failed(stage, "incomplete_graph",
                       "the repeated container count config value is unavailable")
    count_number, count_kind = count_value
    if isinstance(count_number, bool) or not isinstance(count_number, int) \
            or count_number < 1 or count_number != len(selector_items):
        return _failed(stage, "conflict",
                       "mask selector length disagrees with exact repeated-container count")

    decisions = []
    for layer_index, token in enumerate(selector_items):
        try:
            builder = key_to_builder[token]
        except (KeyError, TypeError):
            return _failed(
                stage, "conflict",
                f"selector element {layer_index} has no exact builder-map entry")
        decisions.append(MaskLayerDecision(layer_index, token, builder))
    spans = tuple(dict.fromkeys((
        *scores.value.spans,
        *config_address.spans,
        selector_span, count_span,
        *(item.builder.call.span for item in decisions),
    )))
    value = AttentionMaskLayerSchedule(
        scores.value, application, config_address,
        selector_path, count_path, selector_kind, count_kind,
        tuple(decisions), spans)
    return ReaderResult.resolved(
        stage, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=(selector_path, count_path),
            detail=("exact enumerate index selects already-proven mask builders "
                    "from an exact config sequence")),))


def decoder_uniform_attention_mask_layer_schedule_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector,
) -> ReaderResult[UniformAttentionMaskLayerSchedule]:
    """Repeat one exact score-applied mask over the exact container count."""
    if config_selector is None:
        raise TypeError("a uniform mask schedule requires a config selector")
    scores = decoder_attention_mask_score_applications_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if scores.status != "resolved" or scores.value is None:
        return _propagate_without_value(
            scores, "mask score-application inventory is unavailable")
    if len(scores.value.applications) != 1:
        return _failed(
            scores.owner, "conflict",
            "a uniform mask schedule requires one exact score-applied lane")
    application = scores.value.applications[0]
    mask = application.mask_application
    if mask.mechanism is None or len(mask.builders) != 1:
        return _failed(
            scores.owner, "conflict",
            "the exact mask lane is not one unconditional builder")

    candidates = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return _propagate_without_value(
            candidates, "decoder block address is unavailable")
    root = candidates.value.component_root
    stage = candidates.value.stage_occurrence
    node = root.graph.node_for(stage)
    if node is None:
        return _failed(stage, "out_of_owner", "stage is absent from its graph")
    address_result = _mask_stage_config_address(
        index, root, scores.value.mechanisms)
    if address_result.status != "resolved":
        return _propagate_without_value(
            address_result, "stage config address is unavailable")
    config_address = address_result.value
    proofs = tuple(
        proof for proof in candidates.value.repeated_child.proofs
        if proof.child_occurrence == mask.block_occurrence
        and proof.template.call == mask.binding.call)
    if len(proofs) != 1:
        return _failed(
            stage, "incomplete_graph",
            "the uniform mask call has no unique repeated template")
    proof = proofs[0]
    count = _container_count_path(
        index, replace(
            node, config_bindings=(config_address.binding,)),
        proof.template.container.count_expression,
        proof.template.container.record.enclosing_callable,
        config_prefix=config_path)
    if count is None:
        return _failed(
            stage, "unsupported_syntax",
            "the repeated container count is not one exact config-bound range")
    count_path, count_span = count
    selected = _select_config_value(config_selector, count_path)
    if selected is None:
        return _failed(
            stage, "incomplete_graph",
            "the repeated container count config value is unavailable")
    count_value, count_kind = selected
    if isinstance(count_value, bool) or not isinstance(count_value, int) \
            or count_value < 1:
        return _failed(
            stage, "conflict", "the exact repeated-container count is invalid")
    builder = mask.builders[0]
    decisions = tuple(
        MaskLayerDecision(layer_index, None, builder)
        for layer_index in range(count_value))
    spans = tuple(dict.fromkeys((
        *scores.value.spans, *config_address.spans,
        count_span, builder.call.span,
    )))
    value = UniformAttentionMaskLayerSchedule(
        scores.value, application, config_address,
        count_path, count_kind, decisions, spans)
    return ReaderResult.resolved(
        stage, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans, config_paths=(count_path,),
            detail=("one exact score-applied mask builder repeats over the "
                    "exact config-bound container count")),))


def decoder_attention_mask_geometry_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector,
) -> ReaderResult[tuple[AttentionMaskGeometry, ...]]:
    """Resolve geometry only from the exact enacted framework-mask calls.

    A config field is not geometry by itself.  The field becomes architectural
    evidence only when an exact imported framework protocol receives the exact
    stage config object and that builder is selected by an exact layer schedule.
    """
    execution = decoder_attention_mask_execution_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if execution.status != "resolved" or execution.value is None:
        return _propagate_without_value(
            execution, "mask execution is unavailable")
    if not execution.value.geometries:
        return ReaderResult.absent(
            execution.owner, provenance=(ReaderProvenance(
                "derived",
                detail="the exact enacted mask schedule has no geometry protocol"),))
    paths = tuple(dict.fromkeys(
        item.config_path for item in execution.value.geometries))
    return ReaderResult.resolved(
        execution.owner, execution.value.geometries,
        provenance=tuple(dict.fromkeys((
            *execution.provenance,
            ReaderProvenance(
                "code_and_config", spans=execution.value.spans,
                config_paths=paths,
                detail="exact enacted mask geometry projection"),
        ))))


def decoder_attention_mask_execution_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector,
) -> ReaderResult[AttentionMaskExecution]:
    """Resolve one authoritative schedule with all of its geometry operands."""
    if config_selector is None:
        raise TypeError("mask execution requires an owner-qualified config selector")
    schedule = decoder_attention_mask_layer_schedule_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if schedule.status == "incomplete":
        return schedule
    if schedule.status == "absent":
        schedule = decoder_uniform_attention_mask_layer_schedule_for_path(
            index, bundle, config_path, allow_root_stage=allow_root_stage,
            config_selector=config_selector)
    if schedule.status != "resolved" or schedule.value is None:
        return _propagate_without_value(
            schedule, "mask schedule is unavailable")

    selected_builders = tuple(dict.fromkeys(
        item.builder for item in schedule.value.decisions
        if item.builder.import_proof.qualified_target in _MASK_GEOMETRY_FIELDS))
    alias = schedule.value.framework_config
    selector = framework_config_default_selector(
        index, alias, config_selector, config_prefix=config_path)
    geometries = []
    failures = []
    for builder in selected_builders:
        actual = _builder_config_actual(index, builder)
        if actual is None:
            failures.append(ReaderFailure(
                "incomplete_graph",
                "the exact mask builder config actual is not uniquely proven",
                builder.call.span))
            continue
        if _attribute_chain(actual) != ("self", alias.stored_field):
            failures.append(ReaderFailure(
                "out_of_owner",
                "the exact mask builder does not receive the stage config object",
                actual.span))
            continue
        field = _MASK_GEOMETRY_FIELDS[builder.import_proof.qualified_target]
        resolved = alias.config_binding.resolved_path((field,))
        if resolved is None:
            failures.append(ReaderFailure(
                "incomplete_graph",
                f"the exact stage config cannot resolve {field!r}",
                actual.span))
            continue
        path = (*config_path, *resolved)
        selected = _select_config_value(selector, path)
        if selected is None:
            failures.append(ReaderFailure(
                "incomplete_graph",
                f"the exact geometry value at {'.'.join(path)} is unavailable",
                actual.span))
            continue
        value, source_kind = selected
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            failures.append(ReaderFailure(
                "conflict",
                f"the exact geometry value at {'.'.join(path)} is not positive",
                actual.span))
            continue
        spans = tuple(dict.fromkeys((
            *schedule.value.spans, builder.call.span, actual.span,
        )))
        geometries.append(AttentionMaskGeometry(
            schedule.value, builder, actual, path, source_kind, value, spans))

    if failures:
        return ReaderResult.failed(schedule.owner, tuple(failures))
    spans = tuple(dict.fromkeys((
        *schedule.value.spans,
        *(span for item in geometries for span in item.spans),
    )))
    schedule_paths = (
        (schedule.value.selector_path, schedule.value.count_path)
        if isinstance(schedule.value, AttentionMaskLayerSchedule)
        else (schedule.value.count_path,))
    paths = tuple(dict.fromkeys((
        *schedule_paths,
        *(item.config_path for item in geometries),
    )))
    dependencies = list(
        schedule.value.score_inventory.mechanisms.config_dependencies)
    if isinstance(schedule.value, AttentionMaskLayerSchedule):
        dependencies.extend((
            (schedule.value.selector_path, schedule.value.selector_source_kind),
            (schedule.value.count_path, schedule.value.count_source_kind),
        ))
    else:
        dependencies.append((
            schedule.value.count_path, schedule.value.count_source_kind))
    dependencies.extend(
        (item.config_path, item.source_kind) for item in geometries)
    dependencies = tuple(dict.fromkeys(dependencies))
    value = AttentionMaskExecution(
        schedule.value, tuple(geometries), dependencies, spans)
    return ReaderResult.resolved(
        schedule.owner, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans, config_paths=paths,
            detail=("exact enacted framework-mask schedule and every required "
                    "stage-config geometry value")),))


def _transport_mask_formal(index, root, mask, invocation_path):
    """Transport one exact formal through an authoritative owner-call path."""
    if not invocation_path:
        return None
    current_callable = mask.binding.callee_callable
    current_formal = mask.binding.formal
    token = ("block_mask_formal", current_formal.name)
    transport = []
    for invocation in invocation_path:
        if invocation.call.enclosing_callable != current_callable:
            return None
        bound = bind_addressed_invocation(index, root, invocation)
        if bound.status not in {"resolved", "partial"}:
            return None
        candidates = []
        for binding in bound.bindings:
            sources, _widths, _dependencies, uncertain = \
                producer_sources_reaching_expressions(
                    index, current_callable,
                    ((binding.call.span, (binding.actual,)),), {},
                    initial_sources={current_formal.name: token})
            if sources == frozenset((token,)) and not uncertain:
                candidates.append(binding)
        if len(candidates) != 1:
            return None
        selected = candidates[0]
        transport.append(selected)
        current_callable = selected.callee_callable
        current_formal = selected.formal
    return tuple(transport), current_formal, current_callable, token


def _builder_config_actual(index, builder):
    """Return the one exact ``config`` actual consumed by a mask protocol.

    Framework mask APIs accept config as the first positional or the named
    ``config`` argument.  ``**kwargs`` is opened only through one exact dict
    definition in the same callable, on the same guard path, before this call.
    A nearby or same-spelled config expression is never enough.
    """
    call = builder.call
    explicit = tuple(
        value for name, value in call.kwargs if name == "config")
    expansions = tuple(
        value for name, value in call.kwargs if name == "**")
    if len(explicit) == 1:
        return explicit[0]
    if explicit:
        return None
    if call.args:
        return call.args[0]
    if len(expansions) != 1:
        return None
    expansion = expansions[0]
    if expansion.kind != "name" or not expansion.name:
        return None
    mapping = _exact_kwargs_mapping(
        index, call.enclosing_callable, expansion.name, call.guard, call.span)
    if mapping is None:
        return None
    matches = tuple(
        value
        for key, (_ordinal, value) in zip(
            mapping.children, mapping.keyword_children)
        if key.kind == "constant" and key.const_value == "config")
    return matches[0] if len(matches) == 1 else None


def _exact_kwargs_mapping(
        index, callable_symbol, name, guard, before_span, seen=()):
    """Resolve one local kwargs dictionary through exact ``dict.copy()`` aliases.

    A copy is safe only when both bindings are unique, precede their use, and
    live on the exact same guard path.  Dict mutation, merging, competing
    definitions and arbitrary helper calls remain deliberately unsupported.
    """
    if not name or name in seen:
        return None
    definitions = tuple(
        item for item in index.bindings_in(callable_symbol)
        if item.value is not None
        and _binding_target_names(item) == (name,)
        and item.guard == guard
        and _span_precedes(item.span, before_span))
    if len(definitions) != 1:
        return None
    definition = definitions[0]
    mapping = definition.value
    if mapping.kind != "dict" \
            or len(mapping.children) != len(mapping.keyword_children):
        if mapping.kind != "call" or len(mapping.children) != 1 \
                or mapping.keyword_children:
            return None
        callee = mapping.children[0]
        if callee.kind != "attribute" or callee.name != "copy" \
                or len(callee.children) != 1:
            return None
        source = callee.children[0]
        if source.kind != "name" or not source.name:
            return None
        return _exact_kwargs_mapping(
            index, callable_symbol, source.name, guard,
            definition.span, (*seen, name))
    return mapping


def _bind_callable_actuals(index, callable_symbol, call):
    """Exact explicit Python argument binding for an indexed helper call."""
    record = index.callable_by_symbol(callable_symbol)
    if record is None:
        return ()
    params = list(record.params)
    if record.kind == "method":
        if not params or params[0].kind not in {"positional", "posonly"}:
            return ()
        params = params[1:]
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    by_name = {item.name: item for item in params
               if item.kind not in {"vararg", "kwarg", "posonly"}}
    accepts_kwargs = any(item.kind == "kwarg" for item in params)
    if len(call.args) > len(positional):
        return ()
    out = list(zip(positional, call.args))
    bound = {item.name for item, _actual in out}
    for name, actual in call.kwargs:
        if name == "**":
            if accepts_kwargs:
                continue
            return ()
        if name not in by_name:
            if accepts_kwargs:
                continue
            return ()
        if name in bound:
            return ()
        out.append((by_name[name], actual))
        bound.add(name)
    return tuple(out)


def _mask_stage_config_address(index, root, mechanisms):
    """Transport the exact framework config prefix to the repeated stage."""
    if not isinstance(mechanisms, AttentionMaskMechanismInventory):
        raise TypeError("stage config address consumes a mask inventory")
    stage = mechanisms.stage_occurrence
    producer = mechanisms.producer_occurrence
    alias = framework_config_alias(index, root, producer)
    if alias.status != "resolved":
        return alias
    if producer == stage:
        spans = alias.value.spans
        value = MaskStageConfigAddress(
            stage, producer, alias.value, alias.value.config_binding,
            None, spans)
        return ReaderResult.resolved(stage, value, provenance=alias.provenance)
    if not stage.sites or stage.sites[:-1] != producer.sites:
        return _failed(
            stage, "out_of_owner",
            "mask producer is not the repeated stage's exact immediate owner")
    producer_node = root.graph.node_for(producer)
    stage_node = root.graph.node_for(stage)
    if producer_node is None or stage_node is None:
        return _failed(stage, "out_of_owner", "config transport owner is absent")
    sites = tuple(
        item for item in index.construction_sites_of(producer_node.symbol)
        if item.site_id == stage.sites[-1])
    if len(sites) != 1:
        return _failed(
            stage, "incomplete_graph",
            "the repeated stage has no unique exact construction site")
    parent = replace(
        producer_node, config_bindings=(alias.value.config_binding,))
    bindings = tuple(
        item for item in resolve_child_config_bindings(
            index, parent, sites[0], stage_node.symbol)
        if item.resolved_prefix is not None)
    if len(bindings) != 1:
        return _failed(
            stage, "incomplete_graph",
            "the producer config has no unique exact stage-constructor binding")
    spans = tuple(dict.fromkeys((*alias.value.spans, sites[0].span)))
    value = MaskStageConfigAddress(
        stage, producer, alias.value, bindings[0], sites[0], spans)
    return ReaderResult.resolved(
        stage, value,
        provenance=(*alias.provenance, ReaderProvenance(
            "source", spans=spans,
            detail="exact producer config reaches exact repeated-stage constructor")))


def _mask_evidence_in_scope(index, root, occurrence, forward):
    """Collect direct + exact same-class-helper builders at one occurrence."""
    direct = _mask_builders(index, forward)
    transports = []
    helper_builders = []
    for call in index.calls_in(forward):
        transport = resolve_self_method_return_transport(
            index, root, occurrence, forward, call)
        if transport.status != "resolved":
            continue
        carried = _mask_builders(index, transport.value.helper.symbol)
        if not carried:
            continue
        transports.append(transport.value)
        helper_builders.extend(carried)
    return (
        direct,
        tuple(sorted(transports, key=lambda item: _span_key(item.call.span))),
        tuple(sorted(helper_builders,
                     key=lambda item: _span_key(item.call.span))),
    )


def _exact_parent_stage_call(index, root, stage, stage_forward):
    """Return the exact immediate-owner call that supplies ``stage``.

    This is an address/binding boundary only.  The parent is the exact prefix
    of the already-proven owner occurrence, and the call is selected solely by
    its addressed callee occurrence.  Multiple/no addressed calls are never
    ranked by spelling or by their arguments.
    """
    if not stage.sites:
        return None, ""
    parent = OwnerOccurrenceId(stage.root, stage.sites[:-1])
    parent_node = root.graph.node_for(parent)
    if parent_node is None or index.class_by_symbol(parent_node.symbol) is None:
        return None, "the stage's immediate owner does not round-trip"
    inventory = resolve_container_inventory(index, root, parent)
    if inventory.status == "failed":
        return None, inventory.failure_detail or inventory.failure_kind
    invocations = resolve_addressed_invocations(
        index, root, parent, inventory)
    if invocations.status == "absent":
        return None, ""
    if invocations.status != "resolved":
        return None, (
            invocations.failure_detail
            or "the immediate owner's invocation census is unavailable")
    matches = tuple(
        item for item in invocations.addressed
        if item.callee_owner_occurrence == stage)
    if not matches:
        return None, ""
    if len(matches) != 1:
        return None, (
            "multiple exact immediate-owner invocations target the repeated stage")
    bound = bind_addressed_invocation(index, root, matches[0])
    if bound.status == "failed" or bound.callee_callable != stage_forward:
        return None, (
            bound.failure_detail
            or "the parent-to-stage call arguments are not exactly bindable")
    return (parent, invocations.callable_symbol, bound.bindings), ""


def _mask_builders(index, forward):
    out = []
    bindings = tuple(item for item in index.bindings_in(forward)
                     if item.value is not None and item.span is not None)
    for call in index.calls_in(forward):
        proof = resolve_import_reference(
            index, forward.source, forward, call.callee)
        if proof is None or proof.qualified_target not in _MASK_PROTOCOLS:
            continue
        definitions = tuple(
            item for item in bindings
            if _expr_contains_span(item.value, call.span))
        if len(definitions) != 1:
            continue
        definition = definitions[0]
        spans = tuple(dict.fromkeys((call.span, definition.span)))
        out.append(MaskBuilderEvidence(
            _MASK_PROTOCOLS[proof.qualified_target], proof,
            call, definition, spans))
    return tuple(sorted(out, key=lambda item: _span_key(item.call.span)))


def _enumeration_index_name(template):
    if template.iteration_kind not in {"enumerated", "enumerated_sliced"}:
        return None
    target = template.loop.target
    if target.kind not in {"tuple", "list"}:
        return None
    indices = tuple(
        item.name for item in target.children
        if item != template.element_target
        and item.kind == "name" and item.name)
    return indices[0] if len(indices) == 1 else None


def _selector_shape(mask, alias, config_prefix, index_name):
    actual = mask.binding.actual
    if actual.kind != "subscript" or len(actual.children) != 2:
        return None
    mapping_expr, selected_expr = actual.children
    if mapping_expr.kind != "name" or not mapping_expr.name \
            or selected_expr.kind != "subscript" \
            or len(selected_expr.children) != 2:
        return None
    config_expr, index_expr = selected_expr.children
    if index_expr.kind != "name" or index_expr.name != index_name:
        return None
    selector_path = config_path_from_framework_alias(
        config_expr, alias, config_prefix=config_prefix)
    if selector_path is None:
        return None
    definitions = tuple(dict.fromkeys(item.definition for item in mask.builders))
    if len(definitions) != 1:
        return None
    definition = definitions[0]
    if _binding_target_names(definition) != (mapping_expr.name,) \
            or definition.value is None or definition.value.kind != "dict" \
            or len(definition.value.children) \
            != len(definition.value.keyword_children):
        return None
    by_span = {item.call.span: item for item in mask.builders}
    key_to_builder = {}
    for key_expr, (_ordinal, value_expr) in zip(
            definition.value.children, definition.value.keyword_children):
        if key_expr.kind != "constant":
            return None
        matching = tuple(
            builder for span, builder in by_span.items()
            if _expr_contains_span(value_expr, span))
        if len(matching) != 1:
            return None
        try:
            if key_expr.const_value in key_to_builder:
                return None
            key_to_builder[key_expr.const_value] = matching[0]
        except TypeError:
            return None
    if set(key_to_builder.values()) != set(mask.builders):
        return None
    return selector_path, key_to_builder, selected_expr.span


def _container_count_path(
        index, node, expression, callable_symbol, *, config_prefix):
    if expression is None or expression.kind != "call" \
            or len(expression.children) != 2 or expression.keyword_children:
        return None
    callee, argument = expression.children
    if callee.kind != "name" or callee.name != "range":
        return None
    # ProgramIndex intentionally keeps a comprehension's ``range`` inside the
    # neutral count expression rather than duplicating it into CallObservation.
    # Exact builtin identity is therefore proven from the expression plus the
    # complete module/callable binding census, not from a second AST walk.
    if any(item.name == "range" and item.kind != "import"
           for item in index.module_bindings_in(callable_symbol.source)) \
            or any(item.name == "range"
                   and item.context in {"parameter", "store", "del"}
                   for item in index.identifiers_in(callable_symbol)):
        return None
    path = exact_config_path_for_expression(
        index, node, argument, config_prefix=config_prefix)
    return (path, argument.span) if path is not None and argument.span is not None else None


def _select_config_value(selector, path):
    selected = selector(path)
    if isinstance(selected, FrameworkConfigDefaultValue):
        return selected.value, "class_default"
    if isinstance(selected, NormalizedConfigValue):
        kinds = {kind for _dependency, kind in selected.dependencies}
        if len(kinds) != 1:
            return None
        return selected.value, next(iter(kinds))
    source_kind = "config_declared"
    if isinstance(selected, tuple) and len(selected) in {2, 3} \
            and isinstance(selected[0], bool):
        present, value = selected[:2]
        if len(selected) == 3:
            source_kind = selected[2]
    else:
        present, value = selected is not None, selected
    if not present or source_kind not in {"config_declared", "class_default"}:
        return None
    return value, source_kind


def _expr_contains_span(expression, span):
    if not isinstance(expression, ExprNode) or span is None:
        return False
    if expression.span == span:
        return True
    return any(_expr_contains_span(child, span)
               for child in expression.children) or any(
        _expr_contains_span(child, span)
        for _name, child in expression.keyword_children)


def _binding_target_names(binding):
    return tuple(name for target in binding.targets
                 for name in _target_names(target))


def _target_names(expression):
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     for name in _target_names(child))
    return ()


def _attribute_chain(expression):
    parts = []
    current = expression
    while isinstance(current, ExprNode) and current.kind == "attribute" \
            and len(current.children) == 1 and current.name:
        parts.append(current.name)
        current = current.children[0]
    if not isinstance(current, ExprNode) or current.kind != "name" \
            or not current.name:
        return ()
    parts.append(current.name)
    parts.reverse()
    return tuple(parts)


def _span_precedes(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) <= (
        right.line, right.col)


def _expression_names(expression):
    if not isinstance(expression, ExprNode):
        return frozenset()
    names = ({expression.name} if expression.kind == "name"
             and expression.name else set())
    for child in expression.children:
        names.update(_expression_names(child))
    for _name, child in expression.keyword_children:
        names.update(_expression_names(child))
    return frozenset(names)


def _missing_selector(_path):
    return False, None, ""


def _failed(owner, kind, detail):
    return ReaderResult.failed(owner, (ReaderFailure(kind, detail),))


def _propagate_without_value(result, detail):
    """Preserve an upstream verdict without leaking its foreign DTO type.

    ReaderResult is generic only to the type checker. Returning an upstream
    result verbatim lets a downstream ``incomplete`` value masquerade as the
    downstream contract. Absence and ambiguity retain their exact meaning;
    every other non-value verdict becomes a typed downstream failure.
    """
    if result.status == "absent":
        return ReaderResult.absent(result.owner, provenance=result.provenance)
    if result.status == "ambiguous":
        return ReaderResult.ambiguous(
            result.owner, result.ambiguity, provenance=result.provenance)
    failures = tuple(result.failures) or (
        ReaderFailure("incomplete_graph", detail),)
    return ReaderResult.failed(
        result.owner, failures, provenance=result.provenance)


def _span_key(span):
    if span is None:
        return ("", "", -1, -1, -1, -1)
    return (
        span.source.component_key or "", span.source.canonical_path,
        span.line, span.col, span.end_line or span.line,
        span.end_col or span.col)


def _site_key(site):
    return (
        site.owner.source.component_key or "",
        site.owner.source.canonical_path,
        site.owner.qualified_name,
        site.enclosing_callable.qualified_name,
        *_span_key(site.span), site.ordinal)


__all__ = [
    "MaskBuilderEvidence",
    "MaskBlockApplication",
    "AttentionMaskMechanismInventory",
    "MaskScoreApplication",
    "AttentionMaskScoreInventory",
    "MaskLayerDecision",
    "AttentionMaskLayerSchedule",
    "UniformAttentionMaskLayerSchedule",
    "AttentionMaskGeometry",
    "AttentionMaskExecution",
    "decoder_attention_mask_mechanisms_for_path",
    "decoder_attention_mask_score_applications_for_path",
    "decoder_attention_mask_layer_schedule_for_path",
    "decoder_uniform_attention_mask_layer_schedule_for_path",
    "decoder_attention_mask_geometry_for_path",
    "decoder_attention_mask_execution_for_path",
]
