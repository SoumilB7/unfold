"""U10-F2 — exact checkpoint operands for the diffusion source projection.

F1 projects source evidence only.  F2 re-runs the canonical U10 readers against
the root :class:`PreparedDocument`, lets those readers select exact paths, and
then binds only paths retained by their typed results.  Candidate probes remain
inspections; a path becomes ``bound`` only after source evidence cites it.

This module never consumes an operand.  Consumption means a production fact
used the value and therefore belongs to the atomic F3 parser/renderer cutover.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .schema import DiffusionSourceProjection, project_diffusion_source
from ...evidence import config_access
from ...evidence.attention import (
    AttentionHeadBinding,
    AttentionScoreScalingBinding,
    GatedDeltaGeometryBinding,
)
from ...evidence.attention_lane import FrameworkAttentionGeometryEvidence
from ...evidence.attention_geometry import AttentionHeadGeometry
from ...evidence.cell_topology import DecoderCellTopologyEvidence
from ...evidence.component_owner import (
    ComponentRootResolution,
    OwnerOccurrenceId,
    resolve_owner_graph,
)
from ...evidence.config_registration import (
    RegisteredConstructorConfig,
    RegisteredConstructorDefaultValue,
    read_registered_constructor_config,
)
from ...evidence.diffusion_block import read_diffusion_block_facts
from ...evidence.diffusion_bookends import read_diffusion_bookends
from ...evidence.diffusion_companion import CompanionDenoiserInventory
from ...evidence.diffusion_conditioning import read_diffusion_conditioning_graph
from ...evidence.diffusion_root import DiffusionRootTopology
from ...evidence.diffusion_stack import (
    ConfigSelectedStackVariant,
    read_diffusion_stack_inventory,
)
from ...evidence.diffusion_stream import read_diffusion_stream_graph
from ...evidence.document import DocumentBinding
from ...evidence.ffn_mechanism import ConfigSelectedFFNMechanism
from ...evidence.program_index import ProgramIndex, SourceSpan
from ...evidence.qk_norm import QKNormCodeEvidence
from ...evidence.reader_result import ReaderFailure, ReaderProvenance, ReaderResult


def _span_key(span):
    return (
        span.source.component_key or "",
        span.source.canonical_path,
        span.source.content_fingerprint,
        span.line, span.col, span.end_line, span.end_col,
    )


def _owner_key(owner):
    return (
        owner.root.source.component_key or "",
        owner.root.source.canonical_path,
        owner.root.source.content_fingerprint,
        owner.root.qualified_name,
        tuple((
            site.owner.qualified_name,
            site.enclosing_callable.qualified_name,
            _span_key(site.span),
            site.ordinal,
        ) for site in owner.sites),
    )


@dataclass(frozen=True)
class _OperandSpec:
    path: tuple[str, ...]
    fact_owner: str
    fact_key: str
    reader: str
    source_owner: OwnerOccurrenceId
    source_spans: tuple[SourceSpan, ...]
    projection_slot: str = ""

    def __post_init__(self):
        if not self.path or any(not isinstance(part, str) or not part
                                for part in self.path):
            raise TypeError("a diffusion operand spec has one exact path")
        if not all(isinstance(item, str) and item
                   for item in (self.fact_owner, self.fact_key, self.reader)):
            raise TypeError("a diffusion operand spec names target and reader")
        if not isinstance(self.projection_slot, str):
            raise TypeError("an operand projection slot is typed text")
        if not isinstance(self.source_owner, OwnerOccurrenceId):
            raise TypeError("a diffusion operand spec names its exact source owner")
        if not self.source_spans or any(
                not isinstance(span, SourceSpan)
                or span.source.component_key
                != self.source_owner.root.source.component_key
                for span in self.source_spans):
            raise ValueError("a diffusion operand spec retains exact source spans")

    @property
    def identity(self):
        return (
            self.path, self.fact_owner, self.fact_key, self.reader,
            self.source_owner, self.source_spans,
            self.projection_slot,
        )


def _spec_key(spec):
    return (
        spec.path, spec.fact_owner, spec.fact_key, spec.reader,
        _owner_key(spec.source_owner),
        tuple(_span_key(span) for span in spec.source_spans),
        spec.projection_slot,
    )


def _projected_reader_results(projection):
    """Every block-level result whose value/status enters the F2 projection."""
    for block in projection.blocks:
        evidence = block.evidence
        yield evidence.attention_census_result
        yield evidence.ffn_result
        yield evidence.ffn_census_result
        yield evidence.norm_result
        yield evidence.operations_result
        yield evidence.cell_topology_result
        for lane in evidence.attention_lanes:
            yield lane.score_scaling_result
            yield lane.head_binding_result
            yield lane.head_geometry_result
            yield lane.projection_storage_result
            yield lane.qk_norm_result
            yield lane.position_application_result
            yield lane.separate_position_application_result


def _projected_config_paths(projection):
    return frozenset(
        path
        for result in _projected_reader_results(projection)
        for provenance in result.provenance
        for path in provenance.config_paths)


@dataclass(frozen=True)
class BoundDiffusionConfigOperand:
    """One code-named checkpoint occurrence, bound but not consumed."""

    component_root: OwnerOccurrenceId
    source_owner: OwnerOccurrenceId
    source_spans: tuple[SourceSpan, ...]
    document_path: tuple[str, ...]
    path: tuple[str, ...]
    resolution: config_access.ConfigResolution
    fact_owner: str
    fact_key: str
    reader: str
    projection_slot: str = ""

    def __post_init__(self):
        if not isinstance(self.component_root, OwnerOccurrenceId):
            raise TypeError("a bound operand names its exact source component")
        if not isinstance(self.source_owner, OwnerOccurrenceId) \
                or self.source_owner.root != self.component_root.root:
            raise ValueError("a bound operand names an exact owner in its component")
        if not self.source_spans or any(
                not isinstance(span, SourceSpan)
                or span.source.component_key
                != self.source_owner.root.source.component_key
                for span in self.source_spans):
            raise ValueError("a bound operand retains its decisive source spans")
        for path in (self.document_path, self.path):
            if not isinstance(path, tuple) or any(
                    not isinstance(part, str) or not part for part in path):
                raise TypeError("bound operand paths are exact tuple[str, ...]")
        if not isinstance(self.resolution, config_access.ConfigResolution) \
                or self.resolution.component != "root.denoiser" \
                or self.resolution.state != "present" \
                or self.resolution.selected_path != ".".join(self.path) \
                or self.resolution.selected_alias != self.path[-1]:
            raise ValueError("the bound resolution is the exact denoiser occurrence")
        if self.resolution.provenance != config_access.CHECKPOINT_DECLARED:
            raise ValueError("an F2 operand is declared by the checkpoint itself")
        if self.resolution.source_obj is None:
            raise ValueError("a bound operand retains its addressed source object")
        selected = tuple(
            item for item in self.resolution.present_aliases
            if item.dotted_path == self.resolution.selected_path
            and item.spelling == self.resolution.selected_alias)
        if len(selected) != 1 or selected[0].value is not self.resolution.value:
            raise ValueError("the bound value round-trips to its exact occurrence")
        if not all(isinstance(item, str) and item for item in (
                self.fact_owner, self.fact_key, self.reader)):
            raise TypeError("a bound operand names its intended target and reader")
        if not isinstance(self.projection_slot, str):
            raise TypeError("a bound operand projection slot is typed text")

    @property
    def component(self):
        return self.resolution.component

    @property
    def spelling(self):
        return self.resolution.selected_alias

    @property
    def value(self):
        return self.resolution.value

    @property
    def provenance(self):
        return self.resolution.provenance

    @property
    def identity(self):
        return (
            self.path, self.fact_owner, self.fact_key, self.reader,
            self.source_owner, self.source_spans,
            self.projection_slot,
        )


@dataclass(frozen=True)
class UnboundDiffusionConfigOperand:
    """One source-required operand with no exact checkpoint occurrence."""

    component_root: OwnerOccurrenceId
    source_owner: OwnerOccurrenceId
    source_spans: tuple[SourceSpan, ...]
    path: tuple[str, ...]
    fact_owner: str
    fact_key: str
    reader: str
    projection_slot: str
    reason: str

    def __post_init__(self):
        if not isinstance(self.component_root, OwnerOccurrenceId):
            raise TypeError("an unresolved operand names its exact component")
        if not isinstance(self.source_owner, OwnerOccurrenceId) \
                or self.source_owner.root != self.component_root.root:
            raise ValueError("an unresolved operand names its exact source owner")
        if not self.source_spans or any(
                not isinstance(span, SourceSpan)
                or span.source.component_key
                != self.source_owner.root.source.component_key
                for span in self.source_spans):
            raise ValueError("an unresolved operand retains exact source spans")
        if not self.path or any(not isinstance(part, str) or not part
                                for part in self.path):
            raise TypeError("an unresolved operand carries one exact path")
        if not all(isinstance(item, str) and item for item in (
                self.fact_owner, self.fact_key, self.reader, self.reason)):
            raise TypeError("an unresolved operand retains target, reader and reason")
        if not isinstance(self.projection_slot, str):
            raise TypeError("an unresolved operand projection slot is typed text")

    @property
    def identity(self):
        return (
            self.path, self.fact_owner, self.fact_key, self.reader,
            self.source_owner, self.source_spans,
            self.projection_slot,
        )


@dataclass(frozen=True)
class BoundDiffusionSourceProjection:
    """The F1 projection plus every exact operand its evidence retained."""

    source: DiffusionSourceProjection
    config_root: ComponentRootResolution
    registration_result: ReaderResult[RegisteredConstructorConfig]
    operand_specs: tuple[_OperandSpec, ...]
    operands: tuple[BoundDiffusionConfigOperand, ...]
    unresolved_operands: tuple[UnboundDiffusionConfigOperand, ...]

    def __post_init__(self):
        if not isinstance(self.source, DiffusionSourceProjection):
            raise TypeError("a bound projection retains the closed F1 projection")
        if not isinstance(self.config_root, ComponentRootResolution) \
                or not self.config_root.address_resolved \
                or self.config_root.graph.root.occurrence \
                != self.source.component_root:
            raise ValueError("the operand graph closes the projected component root")
        if not isinstance(self.registration_result, ReaderResult) \
                or self.registration_result.owner != self.source.component_root:
            raise ValueError("the registration result belongs to this component")
        if any(not isinstance(item, _OperandSpec)
               for item in self.operand_specs):
            raise TypeError("the bound projection retains typed operand evidence")
        if self.operand_specs != tuple(sorted(set(self.operand_specs),
                                             key=_spec_key)):
            raise ValueError("operand evidence is unique and canonical")
        if self.registration_result.status == "resolved":
            registration = self.registration_result.value
            if not isinstance(registration, RegisteredConstructorConfig) \
                    or registration.owner != self.source.component_root:
                raise ValueError("the registration proof closes this root")
            bindings = {
                item.parameter: item.resolved_prefix
                for item in self.config_root.graph.root.config_bindings}
            if bindings != registration.root_param_prefixes:
                raise ValueError(
                    "the config-bound owner graph derives from the registration")
        if any(not isinstance(item, BoundDiffusionConfigOperand)
               or item.component_root != self.source.component_root
               for item in self.operands):
            raise ValueError("every operand belongs to the projected component")
        if any(not isinstance(item, UnboundDiffusionConfigOperand)
               or item.component_root != self.source.component_root
               for item in self.unresolved_operands):
            raise ValueError("every unresolved operand belongs to the component")
        bound = tuple(item.identity for item in self.operands)
        unresolved = tuple(item.identity for item in self.unresolved_operands)
        bound_keys = tuple(_spec_key(_OperandSpec(*item)) for item in bound)
        unresolved_keys = tuple(
            _spec_key(_OperandSpec(*item)) for item in unresolved)
        if bound_keys != tuple(sorted(bound_keys)) \
                or unresolved_keys != tuple(sorted(unresolved_keys)) \
                or len(bound) != len(set(bound)) \
                or len(unresolved) != len(set(unresolved)) \
                or set(bound) & set(unresolved):
            raise ValueError("operand partitions are disjoint, unique and canonical")
        expected = tuple(item.identity for item in self.operand_specs)
        actual = tuple(sorted((*bound, *unresolved), key=lambda item: _spec_key(
            _OperandSpec(*item))))
        if actual != expected:
            raise ValueError("bound and unresolved rows exactly partition source paths")
        declared_paths = {item[0] for item in expected}
        hidden_paths = _projected_config_paths(self.source) - declared_paths
        if hidden_paths:
            raise ValueError(
                "every projected config dependency must enter the operand "
                f"partition; missing {sorted(hidden_paths)!r}")


class _ExactOperandSelector:
    """Resolve exact paths once; retain candidates until evidence selects them."""

    def __init__(self, binding: DocumentBinding, component_root: OwnerOccurrenceId,
                 registration: RegisteredConstructorConfig | None = None):
        if not isinstance(binding, DocumentBinding):
            raise TypeError("F2 requires the root's prepared DocumentBinding")
        if not isinstance(component_root, OwnerOccurrenceId):
            raise TypeError("F2 requires one exact component root")
        self.binding = binding
        self.component_root = component_root
        if registration is not None \
                and registration.owner != component_root:
            raise ValueError("registered config belongs to the selected root")
        self.registration = registration
        self.resolutions = {}

    def _resolve(self, path):
        path = tuple(path)
        if not path or any(not isinstance(part, str) or not part for part in path):
            return None
        if path in self.resolutions:
            return self.resolutions[path]
        container = self.binding.document
        for part in path[:-1]:
            if isinstance(container, dict):
                if part not in container:
                    self.resolutions[path] = None
                    return None
                container = container[part]
            elif hasattr(container, part):
                container = getattr(container, part)
            else:
                self.resolutions[path] = None
                return None
        resolution = config_access.resolve(
            container, path[-1], (), component="root.denoiser",
            path=path[:-1])
        expected = ".".join(path)
        if resolution.ambiguous or not resolution.present \
                or resolution.selected_path != expected \
                or resolution.provenance != config_access.CHECKPOINT_DECLARED:
            self.resolutions[path] = None
            return None
        self.resolutions[path] = resolution
        return resolution

    def value(self, path):
        resolution = self._resolve(path)
        return resolution.value if resolution is not None else None

    def guarded(self, path):
        resolution = self._resolve(path)
        if resolution is not None:
            return True, resolution.value, "config_declared"
        # A registered constructor default is code evidence, not a checkpoint
        # occurrence.  It may select an exact source guard only when the file
        # did not declare that path at all; a present, conflicting or
        # class-normalized value is never overwritten by the default.
        path = tuple(path)
        if self.registration is None or len(path) != 1 \
                or path[0] in self.binding.prepared.checkpoint:
            return False, None, ""
        matches = tuple(
            item for item in self.registration.parameters
            if item.name == path[0] and item.has_default
            and item.default is not None and item.default.kind == "constant")
        if len(matches) != 1:
            return False, None, ""
        parameter = matches[0]
        return RegisteredConstructorDefaultValue(
            parameter.default.const_value, path, parameter,
            self.registration)

    def bind(self, spec: _OperandSpec):
        resolution = self._resolve(spec.path)
        if resolution is None:
            return UnboundDiffusionConfigOperand(
                self.component_root, spec.source_owner, spec.source_spans,
                spec.path, spec.fact_owner, spec.fact_key, spec.reader,
                spec.projection_slot,
                "exact checkpoint occurrence is missing, ambiguous, or unaddressable")
        resolution.bind(
            reader=spec.reader, fact_owner=spec.fact_owner,
            fact_key=spec.fact_key)
        return BoundDiffusionConfigOperand(
            self.component_root, spec.source_owner, spec.source_spans,
            tuple(self.binding.document_path), spec.path, resolution,
            spec.fact_owner, spec.fact_key, spec.reader,
            spec.projection_slot)


def _spans(value, fallback):
    spans = tuple(dict.fromkeys(
        span for span in getattr(value, "spans", ())
        if isinstance(span, SourceSpan)))
    return spans or tuple(fallback)


def _spec(path, owner, key, reader, source_owner, source_spans,
          projection_slot=""):
    return _OperandSpec(
        tuple(path), owner, key, reader, source_owner,
        tuple(dict.fromkeys(source_spans)), projection_slot)


def _registered_self_config_path(index, root, occurrence, expression,
                                 registration):
    """Map ``self.config.<parameter>`` through an exact framework protocol.

    ``register_to_config`` is the authority for this address.  The familiar
    spelling alone proves nothing, and a model class that writes ``self.config``
    locally may have replaced the framework object, so that case is refused.
    """
    if registration is None or occurrence != registration.owner:
        return None
    if any(item.field == "config"
           for item in index.field_assigns_of(registration.owner.root)):
        return None
    segments = []
    current = expression
    while current.kind == "attribute" and len(current.children) == 1:
        segments.append(current.name)
        current = current.children[0]
    segments.reverse()
    if len(segments) < 2 or segments[0] != "config" \
            or current.kind != "name" or current.name != "self":
        return None
    parameter_paths = dict(registration.parameter_paths)
    prefix = parameter_paths.get(segments[1])
    if prefix is None:
        return None
    return tuple((*prefix, *segments[2:]))


def _bound_parameter_paths(index, root, occurrence, expression,
                           registration=None):
    """Return every exact constructor-parameter path used by ``expression``.

    A repeated-container count is retained as its whole normalized expression
    (for example ``range(num_layers)``), not merely as the argument inside the
    call.  Walking the frozen expression tree lets this boundary cite the exact
    parameter occurrence without interpreting the callee name or consulting
    diagnostic source text.  Names without one exact owner-graph binding are
    deliberately ignored.
    """
    if expression is None:
        return ()
    node = root.graph.node_for(occurrence)
    if node is None:
        return ()
    by_parameter = {}
    for binding in node.config_bindings:
        if binding.resolved_prefix is not None:
            by_parameter.setdefault(binding.parameter, []).append(
                binding.resolved_prefix)

    found = []

    def visit(item):
        if item is None:
            return
        framework_path = _registered_self_config_path(
            index, root, occurrence, item, registration)
        if framework_path is not None:
            found.append((framework_path, item.span))
            return
        if item.kind == "name" and item.name:
            paths = tuple(dict.fromkeys(by_parameter.get(item.name, ())))
            if len(paths) == 1:
                found.append((paths[0], item.span))
        for child in item.children:
            visit(child)
        for _keyword, child in item.keyword_children:
            visit(child)

    visit(expression)
    return tuple(dict.fromkeys(found))


def _block_operand_specs(
        projection: DiffusionSourceProjection,
        config_root: ComponentRootResolution,
        index: ProgramIndex,
        registration: RegisteredConstructorConfig | None = None,
):
    specs = []
    for block_index, block in enumerate(projection.blocks):
        # A source occurrence remains exact after it enters the config ledger.
        # The old broad owners (``denoiser.stack`` / ``denoiser.attention``)
        # collapsed two independently constructed stack templates that happened
        # to read the same checkpoint path.  F3 projects symbolic templates, so
        # its stable audit address is the canonical source-ordered stack slot.
        stack_owner = f"denoiser.stacks[{block_index}]"
        stack = block.evidence.stack
        if isinstance(stack.selection, ConfigSelectedStackVariant):
            specs.extend(_spec(
                path, stack_owner, "stack_variant",
                "read_diffusion_stack_inventory",
                stack.owner_occurrence, stack.selection.spans)
                for path, kind, _value in stack.selection.premises
                if kind == "config_declared")
        count_paths = (
            ((tuple(segment.name
                    for segment in stack.count_config_path.segments),
              stack.count_config_path.span),)
            if stack.count_config_path is not None else
            _bound_parameter_paths(
                index, config_root, stack.owner_occurrence,
                stack.count_expression, registration))
        for count_path, count_span in count_paths:
            specs.append(_spec(
                count_path,
                stack_owner, "num_layers",
                "read_diffusion_stack_inventory",
                stack.owner_occurrence,
                (count_span or stack.count_expression.span,)))
        for lane_index, lane in enumerate(block.evidence.attention_lanes):
            lane_owner = f"{stack_owner}.attention[{lane_index}]"
            framework_geometry = getattr(lane.child, "geometry", None)
            if isinstance(framework_geometry,
                          FrameworkAttentionGeometryEvidence):
                for fact_key, expression in (
                    ("framework_query_heads", framework_geometry.query_heads),
                    ("framework_key_value_heads",
                     framework_geometry.key_value_heads),
                    ("framework_head_dim", framework_geometry.head_dim),
                ):
                    for path, operand_span in _bound_parameter_paths(
                            index, config_root, stack.block_occurrence,
                            expression, registration):
                        specs.append(_spec(
                            path, lane_owner, fact_key,
                            "framework_attention_container_geometry",
                            stack.block_occurrence,
                            (operand_span, *framework_geometry.spans)))
            head = lane.head_binding_result
            if head.status == "resolved" and isinstance(
                    head.value, AttentionHeadBinding):
                specs.extend(_spec(
                    path, lane_owner, "head_protocol",
                    "attention_head_binding_at_block",
                    head.value.attention_occurrence,
                    _spans(head.value, lane.spans))
                    for path in dict.fromkeys((
                        head.value.query_heads_path,
                        head.value.key_value_heads_path,
                        *(path for path, _value
                          in head.value.selection_premises))))
            geometry = lane.head_geometry_result
            if geometry.status == "resolved" and isinstance(
                    geometry.value, AttentionHeadGeometry):
                specs.extend(_spec(
                    path, lane_owner, "head_dim",
                    "attention_head_geometry_at_block",
                    geometry.value.owner_occurrence,
                    _spans(geometry.value, lane.spans))
                    for path, _value in geometry.value.premises)
            scaling = lane.score_scaling_result
            if scaling.status == "resolved" and isinstance(
                    scaling.value, AttentionScoreScalingBinding):
                specs.extend(_spec(
                    path, lane_owner, "score_scaling",
                    "attention_score_scaling_for_child",
                    scaling.value.attention_occurrence,
                    _spans(scaling.value, lane.spans))
                    for path in scaling.value.config_paths)
            qk_norm = lane.qk_norm_result
            if qk_norm.status == "resolved" and isinstance(
                    qk_norm.value, QKNormCodeEvidence):
                specs.extend(_spec(
                    atom.config_path, lane_owner, "qk_norm",
                    "qk_norm_evidence_at_attention",
                    lane.child.compute_occurrence,
                    lane.spans)
                    for atom in qk_norm.value.gate)
            position = lane.position_application_result
            if position.status == "resolved":
                specs.extend(_spec(
                    path, lane_owner, "position_application",
                    "qk_half_turn_application_at_attention",
                    lane.child.compute_occurrence,
                    _spans(position.value, lane.spans))
                    for path in position.value.guard_config_paths)

        # A positively-proven non-softmax mixer is also a code-and-config
        # mechanism.  Its five operands were absent from F2 even though U6 had
        # already assigned their roles from split/reshape/repeat/Conv1d uses;
        # leaving them out would force F3 either to drop the proved mechanism or
        # reopen raw config.  Bind every role here, still without consuming it.
        for mixer_index, mixer in enumerate(block.evidence.non_softmax_mixers):
            if mixer.status != "resolved" or not isinstance(
                    mixer.value, GatedDeltaGeometryBinding):
                continue
            value = mixer.value
            mixer_owner = f"{stack_owner}.mixers[{mixer_index}]"
            for fact_key, path in (
                ("key_heads", value.key_heads_path),
                ("value_heads", value.value_heads_path),
                ("key_head_dim", value.key_head_dim_path),
                ("value_head_dim", value.value_head_dim_path),
                ("conv_kernel", value.conv_kernel_path),
            ):
                specs.append(_spec(
                    path, mixer_owner, fact_key,
                    "gated_delta_geometry_at_occurrence",
                    value.mixer_occurrence,
                    _spans(value, block.evidence.spans)))

        ffn = block.evidence.ffn_result
        if ffn.status == "resolved":
            value = ffn.value
            if isinstance(value, ConfigSelectedFFNMechanism):
                specs.append(_spec(
                    value.selector_config_path, f"{stack_owner}.ffn", "mechanism",
                    "ffn_mechanism_at_block",
                    value.wrapper_invocation.callee_owner_occurrence,
                    _spans(value, block.evidence.spans)))
            if value.activation_config_path:
                specs.append(_spec(
                    value.activation_config_path, f"{stack_owner}.ffn", "activation",
                    "ffn_mechanism_at_block", value.owner_occurrence,
                    _spans(value, block.evidence.spans)))
        cell = block.evidence.cell_topology_result
        if cell.status == "resolved" and isinstance(
                cell.value, DecoderCellTopologyEvidence):
            specs.extend(_spec(
                path, f"{stack_owner}.cell", "topology", "cell_topology_at_block",
                cell.value.block_occurrence,
                _spans(cell.value, block.evidence.spans))
                for path in dict.fromkeys((
                    *cell.value.norm_config_paths,
                    *cell.value.residual_config_paths,
                    *((cell.value.residual_scale_path,)
                      if cell.value.residual_scale_path is not None else ()))))

    # Root bookend dimensions are separate from the repeated-block facts above.
    # The U10-E reader has already proven an exact framework constructor role;
    # F2 only maps that constructor expression through the authoritative owner
    # graph to its exact checkpoint occurrence.
    for application in projection.bookends.applications:
        for dimension in application.dimension_operands:
            # A direct constructor parameter has one exact checkpoint value.
            # Arithmetic dimension expressions require a separate typed
            # evaluator; binding their leaves here would falsely project one
            # premise as the computed constructor width.
            if dimension.expression.kind != "name":
                continue
            for path, operand_span in _bound_parameter_paths(
                    index, config_root, application.owner_occurrence,
                    dimension.expression, registration):
                specs.append(_spec(
                    path, "denoiser", "bookend_geometry",
                    "diffusion_bookend_constructor_geometry",
                    application.owner_occurrence,
                    tuple(dict.fromkeys((
                        operand_span, dimension.construction_span,
                        application.call.span))),
                    dimension.slot))
    return tuple(sorted(set(specs), key=_spec_key))


def bind_diffusion_source_projection(
        index: ProgramIndex,
        root: ComponentRootResolution,
        binding: DocumentBinding,
        topology_result: ReaderResult[DiffusionRootTopology],
        companion_result: ReaderResult[CompanionDenoiserInventory],
) -> ReaderResult[BoundDiffusionSourceProjection]:
    """Build and bind the exact F2 projection; never consume an operand."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("F2 requires the call-local ProgramIndex")
    if not isinstance(root, ComponentRootResolution) or not root.address_resolved:
        raise ValueError("F2 requires a resolved D0 component root")
    if not isinstance(binding, DocumentBinding) or binding.owner != "root" \
            or binding.document_path:
        raise ValueError("F2 requires the prepared root document binding")
    # The binding is an input to this boundary, not an ambient precondition.
    # Entering it here makes every exactness/provenance decision reproducible
    # for direct callers and prevents an omitted outer scope from silently
    # downgrading real checkpoint occurrences to unestablished reads.
    with config_access.bound_document(binding), \
            config_access.owner_scope("root.denoiser"):
        return _bind_diffusion_source_projection_scoped(
            index, root, binding, topology_result, companion_result)


def _bind_diffusion_source_projection_scoped(
        index, root, binding, topology_result, companion_result):
    registration = read_registered_constructor_config(index, root)
    config_root = root
    if registration.status == "resolved":
        graph = resolve_owner_graph(
            index, root.graph.root.symbol,
            root_param_prefixes=registration.value.root_param_prefixes)
        config_root = replace(
            root, occurrence=graph.root.occurrence, graph=graph)
    registration_value = (
        registration.value if registration.status == "resolved" else None)
    selector = _ExactOperandSelector(
        binding, config_root.graph.root.occurrence, registration_value)
    stacks = read_diffusion_stack_inventory(
        index, config_root, config_guard_selector=selector.guarded)
    blocks = read_diffusion_block_facts(
        index, config_root, stacks, config_document=binding.document,
        config_value_selector=selector.value,
        config_guard_selector=selector.guarded)
    if not blocks.has_value:
        return ReaderResult.failed(
            root.graph.root.occurrence,
            blocks.failures or (ReaderFailure(
                "incomplete_graph", "U10-C config-bound facts unavailable"),),
            provenance=blocks.provenance)
    streams = read_diffusion_stream_graph(index, config_root, blocks)
    conditioning = read_diffusion_conditioning_graph(
        index, config_root, streams)
    bookends = read_diffusion_bookends(
        index, config_root, stacks, streams, conditioning)
    source = project_diffusion_source(
        topology_result, blocks, streams, conditioning,
        bookends, companion_result)
    if not source.has_value:
        return ReaderResult.failed(
            root.graph.root.occurrence,
            source.failures or (ReaderFailure(
                "incomplete_graph", "U10-F1 projection unavailable"),),
            provenance=source.provenance)
    operand_specs = _block_operand_specs(
        source.value, config_root, index, registration_value)
    selected = tuple(selector.bind(spec) for spec in operand_specs)
    operands = tuple(item for item in selected
                     if isinstance(item, BoundDiffusionConfigOperand))
    unresolved = tuple(item for item in selected
                       if isinstance(item, UnboundDiffusionConfigOperand))
    value = BoundDiffusionSourceProjection(
        source.value, config_root, registration, operand_specs,
        operands, unresolved)
    failures = tuple(dict.fromkeys((
        *source.failures,
        *registration.failures,
        ReaderFailure(
            "incomplete_graph",
            "U10-F2 binds exact operands but production consumption waits for F3"),
    )))
    return ReaderResult.incomplete(
        root.graph.root.occurrence, value, failures=failures,
        provenance=(ReaderProvenance(
            "derived",
            detail="canonical U10 projection plus exact bound checkpoint operands"),))


__all__ = [
    "BoundDiffusionConfigOperand",
    "UnboundDiffusionConfigOperand",
    "BoundDiffusionSourceProjection",
    "bind_diffusion_source_projection",
]
