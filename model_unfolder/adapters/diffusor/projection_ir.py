"""U10-F3 — the sole diffusion source/config projection into typed IR.

This module is intentionally a *projection*, not another reader.  It accepts
only :class:`BoundDiffusionSourceProjection`, consumes the exact operands F2
already bound, and translates the carried source facts into ``LayerSpec``.
It cannot reopen source, inspect a config object, use a model/class/field name
as semantics, or fill an unknown mechanism with a conventional DiT shape.

One source stack remains one symbolic template here.  The template is expanded
only when its exact, checkpoint-declared repetition operand was bound by F2.
An unbound count therefore produces no fabricated layer instances and remains
visible in ``unresolved``.  An exact count with incomplete cell evidence keeps
that repetition count.  A cell is wholly opaque only when its observed lanes
cannot be represented by the current IR.  Otherwise independently proven
attention/FFN mechanisms stay visible beside the shared honest
``wiring_unresolved`` block; one unresolved branch must never erase a
different, proven branch.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...ir import AttentionSpec, FFNSpec, LayerSpec
from ...evidence.attention import (
    AttentionHeadBinding,
    AttentionScoreScalingBinding,
    GatedDeltaGeometryBinding,
)
from ...evidence.attention_lane import FrameworkAttentionGeometryEvidence
from ...evidence.attention_geometry import AttentionHeadGeometry
from ...evidence.cell_topology import DecoderCellTopologyEvidence
from ...evidence.qk_norm import QKNormCodeEvidence
from ..transformer.assembly import decoder_layer, parallel_decoder_layer
from .config_binding import (
    BoundDiffusionConfigOperand,
    BoundDiffusionSourceProjection,
)
from .schema import DiffusionAttentionProjection, DiffusionBlockProjection


_SOFTMAX_PROTOCOLS = frozenset({
    "scaled_dot_product_attention",
    "dot_softmax",
})
_CONDITIONING_KINDS = frozenset({
    "norm_modulation", "bare_gate", "gate_in_norm",
})


def _stream_variant(kind: str | None, *, lane_count: int = 1) -> dict:
    rows = {
        "single_state": (
            "Self-attention", "single state",
            "One exact returned state enters this attention lane; its internal "
            "mechanism is shown only when separately proven."),
        "contextual_single_state": (
            "Cross-attention", "external context",
            "One returned state supplies queries while an exact external context "
            "supplies keys/values; its internal mechanism is independently proven."),
        "dual_state": (
            "Dual-state attention", "two returned states",
            "One exact attention lane updates at least two returned state roots."),
        "joined_inputs": (
            "Joined-input attention", "explicit join",
            "The lane consumes an exact source-proven input concatenation before "
            "the attention call."),
    }
    if kind in rows:
        title, tag, description = rows[kind]
        return {
            "short": title,
            "tag": tag,
            "label": [title],
            "title": title,
            "desc": description,
            # Machine conformance consumes this exact source-derived relation;
            # it never parses the human label/tag to rediscover semantics.
            "stream_relation": kind,
            "joined_sequence": kind == "joined_inputs",
        }
    return {
        "short": "Attention lanes" if lane_count != 1 else "Attention",
        "tag": "stream unresolved",
        "label": [
            f"{lane_count} attention lanes" if lane_count != 1 else "Attention",
            "stream unresolved",
        ],
        "title": "Attention stream relation unresolved",
        "stream_relation": None,
        "joined_sequence": False,
        "desc": (
            "The source proves the attention call occurrence, but the current "
            "evidence does not close its returned-state/context relation."
        ),
    }


def _positive_int(value):
    return (value if isinstance(value, int) and not isinstance(value, bool)
            and value > 0 else None)


@dataclass(frozen=True)
class DiffusionLayerTemplate:
    """One exact symbolic source block and its IR-safe projection."""

    source: DiffusionBlockProjection
    owner: str
    root_stage: bool
    stack_variant: dict | None
    count: int | None
    attention: AttentionSpec
    cross_attention: AttentionSpec | None
    ffn: FFNSpec
    norm_kind: str
    norm_placement: str
    residual_topology: str
    parallel_norm_count: int | None
    residual_scale: int | float | None
    hidden_size: int | None
    attention_conditioning: tuple[str, ...]
    cross_attention_conditioning: tuple[str, ...]
    ffn_conditioning: tuple[str, ...]
    materialization_blocked: bool
    unresolved: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.source, DiffusionBlockProjection) \
                or not self.owner.startswith("denoiser.stacks["):
            raise TypeError("a layer template retains one exact source stack")
        if not isinstance(self.root_stage, bool):
            raise TypeError("root-stage membership is an exact topology join")
        if self.stack_variant is not None and (
                not isinstance(self.stack_variant, dict)
                or set(self.stack_variant) != {
                    "selected_branch", "candidate_count"}
                or not isinstance(self.stack_variant["selected_branch"], int)
                or not isinstance(self.stack_variant["candidate_count"], int)
                or not 0 <= self.stack_variant["selected_branch"]
                < self.stack_variant["candidate_count"]):
            raise ValueError("a stack variant is one exact rival ordinal")
        if self.count is not None and _positive_int(self.count) is None:
            raise ValueError("a materialized stack count is a positive integer")
        if not isinstance(self.attention, AttentionSpec) \
                or not isinstance(self.ffn, FFNSpec) \
                or self.cross_attention is not None \
                and not isinstance(self.cross_attention, AttentionSpec):
            raise TypeError("a layer template contains typed IR facts")
        if self.norm_kind not in {"layernorm", "rmsnorm", "unknown"} \
                or self.norm_placement not in {"pre", "post", "double", "unknown"} \
                or self.residual_topology not in {
                    "sequential", "parallel", "unknown"}:
            raise ValueError("a layer template uses closed cell vocabularies")
        if self.parallel_norm_count is not None \
                and self.residual_topology != "parallel":
            raise ValueError("parallel norm count requires parallel topology")
        if self.hidden_size is not None and _positive_int(self.hidden_size) is None:
            raise ValueError("a projected hidden size is a positive integer")
        if any(kind not in _CONDITIONING_KINDS for kind in (
                *self.attention_conditioning,
                *self.cross_attention_conditioning,
                *self.ffn_conditioning)):
            raise ValueError("conditioning applications use the closed U10-D vocabulary")
        if not isinstance(self.materialization_blocked, bool):
            raise TypeError("cell materialization authority is an exact boolean")
        if tuple(dict.fromkeys(self.unresolved)) != self.unresolved:
            raise ValueError("unresolved projection reasons are unique and ordered")

    def layers(self, start: int) -> tuple[LayerSpec, ...]:
        if self.count is None or not self.root_stage:
            return ()
        out = []
        for index in range(start, start + self.count):
            if self.materialization_blocked:
                out.append(LayerSpec(
                    index=index,
                    attention=self.attention,
                    ffn=self.ffn,
                    norm_kind=self.norm_kind,
                    norm_placement=self.norm_placement,
                    residual_topology=self.residual_topology,
                    parallel_norm_count=self.parallel_norm_count,
                    residual_scale=self.residual_scale,
                    cross_attention=self.cross_attention,
                    blocks=[{
                        "id": "cell_structure_unresolved",
                        "role": "opaque",
                        "kind": "opaque",
                        "label": ["Repeated cell", "structure unresolved"],
                        "title": "Repeated cell structure unresolved",
                        "description": (
                            "The exact source and checkpoint prove this repeated "
                            "cell occurrence and its count, but do not close every "
                            "internal branch needed to draw the cell. No attention, "
                            "FFN, norm, or residual ordering is filled by convention."
                        ),
                        "resolved": False,
                        "static": True,
                    }],
                ))
                continue
            width = self.hidden_size or 0
            if self.residual_topology == "parallel":
                layer = parallel_decoder_layer(
                    index, self.attention, self.ffn, width,
                    norm_kind=self.norm_kind,
                    norm_placement=self.norm_placement,
                    norm_count=self.parallel_norm_count,
                    residual_scale=self.residual_scale)
                # The shared transformer helper has no additive cross lane.
                # Do not silently drop a proven one by using that helper.
                if self.cross_attention is not None:
                    layer = decoder_layer(
                        index, self.attention, self.ffn, width,
                        norm_kind=self.norm_kind,
                        norm_placement="unknown",
                        residual_topology="unknown",
                        cross_attention_spec=self.cross_attention)
            else:
                layer = decoder_layer(
                    index, self.attention, self.ffn, width,
                    norm_kind=self.norm_kind,
                    norm_placement=self.norm_placement,
                    residual_topology=self.residual_topology,
                    residual_scale=self.residual_scale,
                    cross_attention_spec=self.cross_attention)
            layer.blocks = _project_stream_blocks(
                _project_cross_context_block(
                    _project_conditioning_blocks(
                        layer.blocks,
                        attention=self.attention_conditioning,
                        cross_attention=self.cross_attention_conditioning,
                        ffn=self.ffn_conditioning),
                    self.attention, self.cross_attention),
                self.attention)
            out.append(layer)
        return tuple(out)


def _condition_block(branch: str, kind: str, number: int) -> dict:
    branch_label = {
        "attn": "attention", "cross_attn": "cross-attention", "ffn": "FFN",
    }[branch]
    if kind == "bare_gate":
        return {
            "id": f"{branch}_condition_gate_{number}",
            "role": "residual", "kind": "gate_mul", "label": "×",
            "title": f"Condition gate ({branch_label})",
            "description": (
                f"An exact condition-derived value multiplies the {branch_label} "
                "result before its residual update."),
        }
    if kind == "gate_in_norm":
        return {
            "id": f"{branch}_conditioned_norm_{number}",
            "role": "norm", "kind": "norm", "label": "Conditioned norm",
            "title": f"Condition-gated norm ({branch_label})",
            "description": (
                f"An exact normalization consumes both the {branch_label} result "
                "and a condition-derived gate."),
        }
    return {
        "id": f"{branch}_modulation_{number}",
        "role": "norm", "kind": "norm", "label": "Modulated norm",
        "title": f"Condition-modulated norm ({branch_label})",
        "description": (
            "An exact normalization consumes the returned state and a "
            f"condition input before the {branch_label} lane."),
    }


def _project_conditioning_blocks(blocks, *, attention, cross_attention, ffn):
    """Project only exact U10-D applications beside their exact branch."""
    by_branch = {
        "attn": tuple(attention),
        "cross_attn": tuple(cross_attention),
        "ffn": tuple(ffn),
    }
    out = []
    for block in blocks:
        branch = block.get("id")
        applications = by_branch.get(branch, ())
        # Norm modulation is an input application; the two gate dialects apply
        # to the branch output.  The renderer gets this ordering explicitly and
        # never reclassifies the textual label.
        out.extend(_condition_block(branch, kind, number)
                   for number, kind in enumerate(applications)
                   if kind == "norm_modulation")
        out.append(block)
        out.extend(_condition_block(branch, kind, number)
                   for number, kind in enumerate(applications)
                   if kind != "norm_modulation")
    return out


def _project_stream_blocks(blocks, attention: AttentionSpec):
    """Project a positively-proven input join as one explicit operation.

    The stream reader already proves that the attention invocation consumes an
    exact concatenated input.  Keeping that fact only in ``variant`` was enough
    to classify the conformance view but not enough to draw the operation, so
    code→diagram correctly reported a missing concat.  This projection is
    intentionally narrow: dual-state or unresolved streams gain no join node.
    """
    variant = attention.variant or {}
    if variant.get("stream_relation") != "joined_inputs":
        return blocks
    out = []
    inserted = False
    for block in blocks:
        if not inserted and block.get("id") == "attn":
            out.append({
                "id": "attention_input_join",
                "role": "residual",
                "kind": "concat",
                "label": "‖",
                "title": "Source-proven attention input join",
                "description": (
                    "The exact block execution concatenates the input streams "
                    "before this attention invocation."),
                "feeds": "attn",
            })
            inserted = True
        out.append(block)
    if not inserted:
        raise ValueError(
            "a joined-input attention relation requires its exact attention block")
    return out


def _project_cross_context_block(
        blocks, primary_attention: AttentionSpec,
        cross_attention: AttentionSpec | None):
    """Draw one exact external-state rail beside its addressed lane.

    A contextual lane may be the cell's primary/only attention (PRX) or a
    distinct additive cross-attention lane (LTX/Wan).  The typed specs already
    carry that occurrence-qualified distinction.  This helper projects it; it
    does not infer whether the source is text, image, or another modality.
    """
    candidates = tuple(
        (target, spec) for target, spec in (
            ("attn", primary_attention), ("cross_attn", cross_attention))
        if spec is not None and spec.cross_attention and spec.cross_kv_source)
    if not candidates:
        return blocks
    if len(candidates) != 1:
        raise ValueError(
            "one layer cannot project multiple external-context rails without "
            "distinct source identities")
    target, attention = candidates[0]
    if not any(block.get("id") == target for block in blocks):
        raise ValueError(
            "an external cross-attention source requires its exact lane")
    if any(block.get("id") == "cross_attention_states" for block in blocks):
        raise ValueError("the cross-attention source rail is occurrence-unique")
    source = attention.cross_kv_source
    rail = {
        "id": "cross_attention_states",
        "role": "conditioning",
        "kind": "block",
        "lane": "external-right",
        "feeds": target,
        "diffusion_stage": "cross_attention",
        "label": ["External", "context"],
        "title": "External cross-attention states",
        "description": (
            f"Exact source-proven {source} supplies keys and values to this "
            "cross-attention lane. The modality or encoder mechanism remains "
            "independent evidence."
        ),
    }
    return [rail, *blocks]


@dataclass(frozen=True)
class DiffusionIRProjection:
    """Complete F3 projection result, including every source block template."""

    bound: BoundDiffusionSourceProjection
    templates: tuple[DiffusionLayerTemplate, ...]
    layers: tuple[LayerSpec, ...]
    consumed_operands: tuple[BoundDiffusionConfigOperand, ...]
    bookend_geometry: tuple[dict, ...]
    unresolved: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.bound, BoundDiffusionSourceProjection):
            raise TypeError("the IR projection retains its F2 authority")
        if tuple(item.source for item in self.templates) \
                != self.bound.source.blocks:
            raise ValueError("IR templates exactly cover source block projections")
        if self.consumed_operands != self.bound.operands:
            raise ValueError("F3 consumes every and only F2-bound operand")
        if self.bookend_geometry != diffusion_bookend_geometry_fact_value(
                self.bound):
            raise ValueError("bookend geometry derives solely from F2 operands")
        expected_rows = []
        start = 0
        for template in self.templates:
            rows = template.layers(start)
            expected_rows.extend(rows)
            start += len(rows)
        expected = tuple(expected_rows)
        # Signature equality is intentionally insufficient here: signatures
        # group visual layer variants and omit some block/card detail.  The
        # projection closure must reject a caller that keeps the same signature
        # while altering an actual structural block.
        if self.layers != expected:
            raise ValueError("materialized layers derive solely from templates")


def diffusion_bookend_fact_value(projection: DiffusionIRProjection | object) -> dict:
    """Canonical source-only value consumed by the denoiser bookend blocks.

    The helper accepts the completed IR projection or its bound projection so
    the fact author and the actual block consumer hash the same raw operation
    identities.  Converting underscores to spaces is presentation only and
    remains downstream.
    """
    bound = (projection.bound if isinstance(projection, DiffusionIRProjection)
             else projection)
    source = bound.source
    by_role = {}
    for application in source.bookends.applications:
        kinds = by_role.setdefault(application.role, [])
        for operation in application.operations:
            if operation.kind not in kinds:
                kinds.append(operation.kind)
    return {
        "state_input": tuple(by_role.get("state_input", ())),
        "conditioning_input": tuple(by_role.get("conditioning_input", ())),
        "state_output": tuple(by_role.get("state_output", ())),
        "temporal_operations": tuple(source.temporal_operation_kinds),
    }


def diffusion_bookend_geometry_fact_value(
        projection: DiffusionIRProjection | object) -> tuple[dict, ...]:
    """Exact constructor dimensions whose direct config operands F2 bound."""
    if isinstance(projection, DiffusionIRProjection):
        return projection.bookend_geometry
    bound = projection
    dimensions = {
        item.slot: (application.role, item.operation_kind,
                    item.dimension_role)
        for application in bound.source.bookends.applications
        for item in application.dimension_operands
    }
    rows = []
    seen_slots = set()
    for operand in bound.operands:
        if operand.fact_owner != "denoiser" \
                or operand.fact_key != "bookend_geometry" \
                or not operand.projection_slot:
            continue
        metadata = dimensions.get(operand.projection_slot)
        if metadata is None:
            raise ValueError(
                "a bound bookend dimension must round-trip to U10-E evidence")
        application_role, operation_kind, dimension_role = metadata
        if operand.projection_slot in seen_slots:
            raise ValueError("bookend geometry slots are occurrence-unique")
        seen_slots.add(operand.projection_slot)
        rows.append({
            "_sort_slot": operand.projection_slot,
            "application_role": application_role,
            "operation_kind": operation_kind,
            "dimension_role": dimension_role,
            "value": operand.value,
        })
    rows.sort(key=lambda item: item["_sort_slot"])
    for item in rows:
        item.pop("_sort_slot")
    return tuple(rows)


class _OperandConsumer:
    def __init__(self, bound: BoundDiffusionSourceProjection):
        self.bound = bound
        self.used: set[int] = set()
        self.expected: dict[tuple[str, str, str], tuple[object, list]] = {}
        self.projected: dict[tuple[str, str, str], object] = {}
        self.code_projected: dict[tuple[str, str], object] = {}

    @staticmethod
    def qualified_owner(owner: str) -> str:
        return owner if owner == "root" or owner.startswith("root.") \
            else f"root.{owner}"

    @staticmethod
    def target_key(mechanism: str) -> str:
        keys = {
            "diffusion_root_stack_depth": "diffusion_stack_depth",
            "diffusion_nested_stack_depth": "diffusion_stack_depth",
            "diffusion_stack_variant": "diffusion_stack_variant",
            "diffusion_attention_head_protocol": "diffusion_attention_head_protocol",
            "diffusion_attention_head_geometry": "diffusion_attention_head_dim",
            "diffusion_attention_score_scaling": "diffusion_attention_score_scaling",
            "diffusion_attention_qk_norm": "diffusion_attention_qk_norm",
            "diffusion_attention_position_application": "diffusion_attention_position_application",
            "diffusion_gated_delta_geometry": "diffusion_gated_delta_geometry",
            "diffusion_ffn_mechanism": "diffusion_ffn_mechanism",
            "diffusion_ffn_activation": "diffusion_ffn_mechanism",
            "diffusion_cell_topology": "diffusion_cell_topology",
            "diffusion_bookend_geometry": "diffusion_bookend_geometry",
        }
        try:
            return keys[mechanism]
        except KeyError as exc:
            raise ValueError(
                f"unregistered diffusion projection mechanism {mechanism!r}") from exc

    def rows(self, owner: str, key: str):
        return tuple((number, item) for number, item in enumerate(
            self.bound.operands) if item.fact_owner == owner
            and item.fact_key == key)

    def bind_operand_fact(self, number, operand, *, mechanism, expected_value):
        if number in self.used:
            raise ValueError("one F2 operand cannot be consumed twice")
        fact_key = self.target_key(mechanism)
        fact_owner = self.qualified_owner(operand.fact_owner)
        operand.resolution.consume_decision(
            mechanism=mechanism,
            fact_owner=fact_owner,
            fact_key=fact_key,
            reader=operand.reader,
            status="code_and_config",
            expected_value=expected_value)
        target = (fact_owner, fact_key, mechanism)
        previous = self.expected.get(target)
        if previous is not None and previous[0] != expected_value:
            raise ValueError(
                "one diffusion fact target received conflicting expected values")
        if previous is None:
            self.expected[target] = (expected_value, [operand])
        else:
            previous[1].append(operand)
        self.used.add(number)
        return operand.value

    def project(self, owner: str, *, mechanism: str, actual_value) -> None:
        """Record what the typed-spec projector actually wrote.

        This is deliberately separate from :meth:`bind_operand_fact`: the consumption
        authors the expected fact from source+config evidence; this call reads
        the completed spec field.  Their hashes meet only in the receipt net.
        """
        target = (self.qualified_owner(owner), self.target_key(mechanism), mechanism)
        if target not in self.expected:
            return
        if target in self.projected and self.projected[target] != actual_value:
            raise ValueError("one diffusion fact target projected conflicting values")
        self.projected[target] = actual_value

    def project_fact(self, owner: str, *, fact_key: str, mechanism: str,
                     value, source_owner, source_spans, receipts=None) -> None:
        """Project one value through its strongest exact evidence channel.

        A source reader can resolve a mechanism without reading checkpoint
        config (unconditional Q/K norm, literal score scaling, a fixed FFN,
        and so on).  F3 used to write those values into ``LayerSpec`` but emit
        neither a fact nor a receipt because :meth:`project` only sees F2
        consumption targets.  That made a source-only drawing less auditable
        than a config-bound one.

        If any exact config operand contributes to this fact, the fact is
        finalized once in :meth:`finish` as ``code_and_config`` and every
        contributing mechanism gets its own receipt.  Otherwise this method
        records one ``code_proven`` fact and one real-consumer receipt.  Two
        source projections of the same fact must agree exactly.
        """
        qualified = self.qualified_owner(owner)
        target = (qualified, fact_key, mechanism)
        fact_targets = tuple(
            item for item in self.expected
            if item[:2] == (qualified, fact_key))
        if fact_targets:
            for item in fact_targets:
                if self.expected[item][0] != value:
                    raise ValueError(
                        "source projection disagrees with its exact config-bound fact")
            if target in self.expected:
                self.project(owner, mechanism=mechanism, actual_value=value)
            return
        identity = (qualified, fact_key)
        if identity in self.code_projected:
            previous = self.code_projected[identity]
            if previous != value:
                raise ValueError("one source-only diffusion fact projected conflicting values")
            return
        self.code_projected[identity] = value
        self.project_code_fact(
            owner, fact_key=fact_key, mechanism=mechanism, value=value,
            source_owner=source_owner, source_spans=source_spans,
            receipts=(receipts if receipts is not None
                      else ((fact_key, (fact_key,)),)))

    def project_code_fact(self, owner: str, *, fact_key: str, mechanism: str,
                          value, source_owner, source_spans, receipts=(),
                          status="code_proven", config_paths=()) -> None:
        """Ledger and receipt one source-only typed projection.

        Unlike F2 operands this fact has no config consumption obligation.  It
        is still a first-class owner-qualified fact, and every structural
        surface it authors emits its own route-validated receipt.
        """
        from ...evidence.context import active_facts, active_parse_context
        from ...evidence.facts import EvidenceFact, SourceSpan
        from ...evidence.receipts import ProjectionReceipt, value_status_hash

        qualified = self.qualified_owner(owner)
        if status not in {"code_proven", "class_default"}:
            raise ValueError("source projection uses a closed evidence tier")
        facts = active_facts()
        context = active_parse_context.get()
        if facts is None:
            return
        if any(span.source.component_key
               != source_owner.root.source.component_key
               for span in source_spans):
            raise ValueError(
                "a diffusion fact's occurrence and spans share one component")
        spans = tuple(dict.fromkeys(SourceSpan(
            component=span.source.component_key,
            # ``OwnerOccurrenceId.root`` names the component root, not the
            # nested class that owns every carried span.  Stamping that root
            # class here would be fabricated provenance for block/attention/
            # FFN facts.  File+line remain exact; class stays unset until the
            # typed fact channel carries one enclosing symbol per span.
            class_name=None,
            file=span.source.canonical_path,
            line=span.line,
        ) for span in source_spans))
        facts.record_typed(EvidenceFact(
            key=fact_key, owner=qualified, value=value,
            status=status, completeness="complete",
            source_spans=spans,
            config_paths=tuple(config_paths),
            legacy_source=(
                "exact diffusion source occurrence"
                if status == "code_proven" else
                "exact registered constructor default + diffusion source"),
            reason=("U10 exact source evidence authors one typed diffusion "
                    "projection")))
        if context is not None:
            for structural_target, node_ids in receipts:
                context.projection_receipts.append(ProjectionReceipt(
                    fact_id=f"{qualified}.{fact_key}", owner=qualified,
                    fact_key=fact_key, mechanism=mechanism,
                    fact_value_status_hash=value_status_hash(
                        value, status),
                    surface="spec", structural_target=structural_target,
                    projector_symbol=(
                        "adapters.diffusor.projection_ir.project_diffusion_ir"),
                    node_ids=tuple(node_ids), projection_kind="field"))

    def finish(self):
        missing = tuple(item for number, item in enumerate(self.bound.operands)
                        if number not in self.used)
        if missing:
            raise ValueError(
                "every F2-bound operand must feed exactly one F3 fact; "
                f"unconsumed targets: {[(x.fact_owner, x.fact_key) for x in missing]!r}")
        if set(self.projected) != set(self.expected):
            absent = sorted(set(self.expected) - set(self.projected))
            extra = sorted(set(self.projected) - set(self.expected))
            raise ValueError(
                "diffusion spec receipt partition is incomplete: "
                f"missing={absent!r}, extra={extra!r}")

        from ...evidence.context import active_facts, active_parse_context
        from ...evidence.facts import EvidenceFact, SourceSpan
        from ...evidence.receipts import ProjectionReceipt, value_status_hash
        facts = active_facts()
        context = active_parse_context.get()
        if facts is not None:
            grouped = {}
            for (owner, fact_key, mechanism), (expected, operands) in sorted(
                    self.expected.items()):
                identity = (owner, fact_key)
                row = grouped.setdefault(identity, {
                    "value": expected, "operands": [], "mechanisms": []})
                if row["value"] != expected:
                    raise ValueError(
                        "one diffusion fact received conflicting mechanism values")
                row["operands"].extend(operands)
                row["mechanisms"].append(mechanism)
            for (owner, fact_key), row in sorted(grouped.items()):
                expected = row["value"]
                operands = row["operands"]
                spans = []
                config_paths = []
                for operand in operands:
                    if any(span.source.component_key
                           != operand.source_owner.root.source.component_key
                           for span in operand.source_spans):
                        raise ValueError(
                            "a diffusion operand occurrence and spans share one component")
                    config_path = ".".join(operand.path)
                    if config_path not in config_paths:
                        config_paths.append(config_path)
                    for span in operand.source_spans:
                        converted = SourceSpan(
                            component=span.source.component_key,
                            # The operand retains an occurrence chain, but its
                            # ``root`` is still the component root.  Never
                            # mislabel a nested decisive span with that class.
                            class_name=None,
                            file=span.source.canonical_path,
                            line=span.line)
                        if converted not in spans:
                            spans.append(converted)
                facts.record_typed(EvidenceFact(
                    key=fact_key, owner=owner, value=expected,
                    status="code_and_config", completeness="complete",
                    source_spans=tuple(spans),
                    config_paths=tuple(config_paths),
                    legacy_source=(" + ".join(config_paths)
                                   + " + exact diffusion source"),
                    reason=("U10 source occurrence and exact checkpoint operand "
                            "author one typed diffusion spec field")))
                if context is not None:
                    for mechanism in row["mechanisms"]:
                        actual = self.projected[(owner, fact_key, mechanism)]
                        context.projection_receipts.append(ProjectionReceipt(
                            fact_id=f"{owner}.{fact_key}", owner=owner,
                            fact_key=fact_key, mechanism=mechanism,
                            fact_value_status_hash=value_status_hash(
                                actual, "code_and_config"),
                            surface="spec", structural_target=fact_key,
                            projector_symbol=(
                                "adapters.diffusor.projection_ir.project_diffusion_ir"),
                            node_ids=(fact_key,), projection_kind="field"))
        return self.bound.operands


def _head_shape(lane, owner, consumer):
    result = lane.evidence.head_binding_result
    if result.status != "resolved" or not isinstance(
            result.value, AttentionHeadBinding):
        return None, None, None
    binding = result.value
    rows = consumer.rows(owner, "head_protocol")
    by_path = {}
    for number, operand in rows:
        by_path.setdefault(operand.path, []).append((number, operand))
    required = tuple(dict.fromkeys((
        binding.query_heads_path, binding.key_value_heads_path,
        *(path for path, _value in binding.selection_premises))))
    # A missing rival operand makes the *classification* incomplete; it does
    # not erase the exact role of an independently bound operand.  Consume the
    # present rows into one explicitly-unresolved head fact so F3 neither leaks
    # a bound row nor turns a partial grouped-KV declaration into MHA.
    exact = {
        path: by_path[path][0]
        for path in required if len(by_path.get(path, ())) == 1
    }
    raw = {path: item[1].value for path, item in exact.items()}
    query = _positive_int(raw.get(binding.query_heads_path))
    kv = _positive_int(raw.get(binding.key_value_heads_path))
    if query is None or kv is None:
        final = None
    elif binding.protocol == "equal_heads" and query == kv:
        final = "mha"
    elif binding.protocol == "grouped_kv" and kv == 1 and query > 1:
        final = "mqa"
    elif binding.protocol == "grouped_kv" and 1 < kv < query \
            and query % kv == 0:
        final = "gqa"
    else:
        final = None
    # A head-sharing protocol cannot certify MHA/GQA/MQA when the exact
    # compute occurrence is a dispatch/unknown mixer.  Apply this before the
    # consumption expectation is authored so the fact and the final spec can
    # never disagree about the value they project.
    if lane.compute_protocol not in _SOFTMAX_PROTOCOLS:
        final = None
    expected = {
        "kind": final, "num_heads": query, "num_kv_heads": kv,
        "projection_mode": (
            "split_qkv" if lane.projection_storage == "split"
            else lane.projection_storage),
        "output_gate": (binding.output_gate.activation
                        if binding.output_gate is not None else None),
    }
    for path in required:
        if path not in exact:
            continue
        number, operand = exact[path]
        consumer.bind_operand_fact(
            number, operand, mechanism="diffusion_attention_head_protocol",
            expected_value=expected)
    return final, query, kv


def _framework_head_shape(lane, owner, consumer):
    """Project only geometry named by the closed Diffusers Attention API.

    An omitted ``kv_heads`` remains unknown: this boundary does not import or
    recreate the external class's defaulting behavior.  An explicit heads/KV
    pair can classify the head-sharing shape while score math, projection
    storage, masks, and all other mechanism details remain unknown.
    """
    geometry = getattr(lane.evidence.child, "geometry", None)
    if not isinstance(geometry, FrameworkAttentionGeometryEvidence):
        return None, None, None, None

    def one(key):
        rows = consumer.rows(owner, key)
        return rows[0] if len(rows) == 1 else None

    query_row = one("framework_query_heads")
    kv_row = one("framework_key_value_heads")
    dim_row = one("framework_head_dim")
    query = _positive_int(query_row[1].value) if query_row else None
    kv = _positive_int(kv_row[1].value) if kv_row else None
    head_dim = _positive_int(dim_row[1].value) if dim_row else None
    if query is not None and kv is not None and query == kv:
        kind = "mha"
    elif query is not None and kv == 1 and query > 1:
        kind = "mqa"
    elif query is not None and kv is not None and 1 < kv < query \
            and query % kv == 0:
        kind = "gqa"
    else:
        kind = None
    expected = {
        "kind": kind, "num_heads": query, "num_kv_heads": kv,
        "projection_mode": None, "output_gate": None,
    }
    for row in (query_row, kv_row):
        if row is not None:
            consumer.bind_operand_fact(
                *row, mechanism="diffusion_attention_head_protocol",
                expected_value=expected)
    if dim_row is not None:
        consumer.bind_operand_fact(
            *dim_row, mechanism="diffusion_attention_head_geometry",
            expected_value=head_dim)
    return kind, query, kv, head_dim


def _single_operand_value(consumer, owner, key, *, mechanism,
                          expected_value=None, transform=lambda value: value):
    rows = consumer.rows(owner, key)
    if len(rows) != 1:
        return None
    number, operand = rows[0]
    value = transform(operand.value)
    consumer.bind_operand_fact(
        number, operand, mechanism=mechanism,
        expected_value=value if expected_value is None else expected_value)
    return value


def _head_dimension(lane: DiffusionAttentionProjection, owner: str,
                    consumer: _OperandConsumer) -> int | None:
    """Consume every premise of one code-derived head dimension.

    Head geometry is commonly an expression such as
    ``hidden_size // num_attention_heads``.  F2 therefore binds more than one
    checkpoint occurrence to the single derived dimension.  Treating it like
    a one-row scalar silently left those operands unconsumed and made the real
    parser fail as soon as a diffusion source exposed derived geometry.
    """
    result = lane.evidence.head_geometry_result
    if result.status != "resolved" or not isinstance(
            result.value, AttentionHeadGeometry):
        return None
    geometry = result.value
    rows = consumer.rows(owner, "head_dim")
    by_path = {}
    for number, operand in rows:
        by_path.setdefault(operand.path, []).append((number, operand))
    required = tuple(path for path, _value in geometry.premises)
    if any(len(by_path.get(path, ())) != 1 for path in required):
        return None
    for path in required:
        number, operand = by_path[path][0]
        consumer.bind_operand_fact(
            number, operand,
            mechanism="diffusion_attention_head_geometry",
            expected_value=geometry.head_dim)
    return geometry.head_dim


def _lane_source_owner(lane: DiffusionAttentionProjection):
    """The exact owner of a direct or framework-dispatched attention lane."""
    return (getattr(lane.evidence.child, "compute_occurrence", None)
            or lane.evidence.block_occurrence)


def _attention_spec(lane: DiffusionAttentionProjection, owner: str,
                    consumer: _OperandConsumer) -> AttentionSpec:
    kind, query_heads, kv_heads = _head_shape(lane, owner, consumer)
    head_dim = _head_dimension(lane, owner, consumer)
    framework_geometry = getattr(lane.evidence.child, "geometry", None)
    if isinstance(framework_geometry, FrameworkAttentionGeometryEvidence):
        framework_kind, framework_query, framework_kv, framework_dim = \
            _framework_head_shape(lane, owner, consumer)
        if query_heads is None:
            kind, query_heads, kv_heads = (
                framework_kind, framework_query, framework_kv)
        if head_dim is None:
            head_dim = framework_dim
    scaling = None
    scaling_result = lane.evidence.score_scaling_result
    if scaling_result.status == "resolved" and isinstance(
            scaling_result.value, AttentionScoreScalingBinding):
        scaling = scaling_result.value.scaled
        for number, operand in consumer.rows(owner, "score_scaling"):
            consumer.bind_operand_fact(
                number, operand,
                mechanism="diffusion_attention_score_scaling",
                expected_value=scaling)
    qk_norm = None
    qk_result = lane.evidence.qk_norm_result
    if qk_result.status == "resolved" and isinstance(
            qk_result.value, QKNormCodeEvidence):
        qk_norm = qk_result.value.present
        rows = consumer.rows(owner, "qk_norm")
        if qk_norm is None and rows:
            qk_norm = all(bool(operand.value) for _number, operand in rows)
            for number, operand in rows:
                consumer.bind_operand_fact(
                    number, operand, mechanism="diffusion_attention_qk_norm",
                    expected_value=qk_norm)
    position = bool(lane.position_protocols)
    for number, operand in consumer.rows(owner, "position_application"):
        enabled = bool(operand.value)
        consumer.bind_operand_fact(
            number, operand,
            mechanism="diffusion_attention_position_application",
            expected_value={"kind": "rope" if enabled and position else "unknown",
                            "application": "qk_rotation" if enabled and position
                            else "unknown"})
        position = position and enabled
    head = lane.evidence.head_binding_result.value \
        if lane.evidence.head_binding_result.status == "resolved" else None
    storage = lane.projection_storage
    projection_mode = "split_qkv" if storage == "split" else storage
    spec = AttentionSpec(
        kind=kind,
        num_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        # The compute protocol proves an attention operation, not whether its
        # score domain is full, causal, windowed, or otherwise masked.  U10 has
        # no mask reader yet, so the only lawful projection is unknown.
        mask="unknown",
        qk_norm=qk_norm,
        rope=True if position else None,
        position_kind="rope" if position else "unknown",
        position_application="qk_rotation" if position else "unknown",
        no_rope=False,
        cached=None,
        output_projection=None,
        scores_scaled=scaling,
        projection_mode=projection_mode,
        output_gate=(head.output_gate.activation
                     if isinstance(head, AttentionHeadBinding)
                     and head.output_gate is not None else None),
        cross_attention=lane.stream_kind == "contextual_single_state",
        cross_kv_source=("external context" if lane.stream_kind
                         == "contextual_single_state" else None),
        variant=_stream_variant(lane.stream_kind),
    )
    if isinstance(head, AttentionHeadBinding):
        consumer.project_fact(
            owner, fact_key="diffusion_attention_head_protocol",
            mechanism="diffusion_attention_head_protocol",
            value={"kind": spec.kind, "num_heads": spec.num_heads,
                   "num_kv_heads": spec.num_kv_heads,
                   "projection_mode": spec.projection_mode,
                   "output_gate": spec.output_gate},
            source_owner=head.attention_occurrence,
            source_spans=lane.evidence.spans)
    elif isinstance(framework_geometry, FrameworkAttentionGeometryEvidence) \
            and query_heads is not None:
        consumer.project_fact(
            owner, fact_key="diffusion_attention_head_protocol",
            mechanism="diffusion_attention_head_protocol",
            value={"kind": spec.kind, "num_heads": spec.num_heads,
                   "num_kv_heads": spec.num_kv_heads,
                   "projection_mode": spec.projection_mode,
                   "output_gate": spec.output_gate},
            source_owner=framework_geometry.block_occurrence,
            source_spans=framework_geometry.spans)
    geometry_result = lane.evidence.head_geometry_result
    if geometry_result.status == "resolved":
        consumer.project_fact(
            owner, fact_key="diffusion_attention_head_dim",
            mechanism="diffusion_attention_head_geometry",
            value=spec.head_dim,
            source_owner=geometry_result.value.owner_occurrence,
            source_spans=lane.evidence.spans)
    elif isinstance(framework_geometry, FrameworkAttentionGeometryEvidence) \
            and head_dim is not None:
        consumer.project_fact(
            owner, fact_key="diffusion_attention_head_dim",
            mechanism="diffusion_attention_head_geometry",
            value=spec.head_dim,
            source_owner=framework_geometry.block_occurrence,
            source_spans=framework_geometry.spans)
    if scaling_result.status == "resolved":
        consumer.project_fact(
            owner, fact_key="diffusion_attention_score_scaling",
            mechanism="diffusion_attention_score_scaling",
            value=spec.scores_scaled,
            source_owner=scaling_result.value.attention_occurrence,
            source_spans=lane.evidence.spans)
    if qk_result.status == "resolved":
        consumer.project_fact(
            owner, fact_key="diffusion_attention_qk_norm",
            mechanism="diffusion_attention_qk_norm",
            value=spec.qk_norm,
            source_owner=_lane_source_owner(lane),
            source_spans=lane.evidence.spans)
    if lane.position_protocols:
        consumer.project_fact(
            owner, fact_key="diffusion_attention_position_application",
            mechanism="diffusion_attention_position_application",
            value={"kind": spec.position_kind,
                   "application": spec.position_application},
            source_owner=_lane_source_owner(lane),
            source_spans=lane.evidence.spans)
    if lane.stream_relation is not None:
        receipt_rows = [("variant", ("stream_relation",))]
        if lane.stream_relation.kind == "joined_inputs":
            receipt_rows.append(("blocks", ("attention_input_join",)))
        consumer.project_code_fact(
            owner, fact_key="diffusion_stream_relation",
            mechanism="diffusion_stream_relation",
            value=lane.stream_relation.kind,
            source_owner=lane.stream_relation.block_occurrence,
            source_spans=lane.stream_relation.spans,
            receipts=tuple(receipt_rows))
    return spec


def _gated_delta_spec(block, owner, consumer):
    rows = []
    for mixer_index, result in enumerate(block.evidence.non_softmax_mixers):
        if result.status != "resolved" or not isinstance(
                result.value, GatedDeltaGeometryBinding):
            continue
        mixer_owner = f"{owner}.mixers[{mixer_index}]"
        by_key = {}
        for key in ("key_heads", "value_heads", "key_head_dim",
                    "value_head_dim", "conv_kernel"):
            selected = consumer.rows(mixer_owner, key)
            if len(selected) == 1:
                by_key[key] = selected[0]
        values = {
            key: (_positive_int(by_key[key][1].value)
                  if key in by_key else None)
            for key in ("key_heads", "value_heads", "key_head_dim",
                        "value_head_dim", "conv_kernel")
        }
        geometry_valid = (
            all(value is not None for value in values.values())
            and values["value_heads"] >= values["key_heads"]
            and values["value_heads"] % values["key_heads"] == 0)
        projected_geometry = {
            "kind": "gated_delta",
            "geometry_valid": geometry_valid,
            "key_heads": values["key_heads"] if geometry_valid else None,
            "value_heads": values["value_heads"] if geometry_valid else None,
            "key_head_dim": values["key_head_dim"] if geometry_valid else None,
            "value_head_dim": values["value_head_dim"] if geometry_valid else None,
            "conv_kernel": values["conv_kernel"] if geometry_valid else None,
        }
        for number, operand in by_key.values():
            consumer.bind_operand_fact(
                number, operand,
                mechanism="diffusion_gated_delta_geometry",
                expected_value=projected_geometry)
        # The source proves the gated-delta mechanism independently of whether
        # all five checkpoint geometry operands are present and valid.  Keep
        # the mechanism, consume every exact operand into a partial geometry,
        # and leave unproved dimensions unknown.  Dropping the whole lane (or
        # raising at ``finish``) would make unknown config erase code evidence.
        spec = AttentionSpec(
            kind="gated_delta",
            mixer_state="gated_delta",
            num_heads=values["value_heads"] if geometry_valid else None,
            num_kv_heads=values["key_heads"] if geometry_valid else None,
            head_dim=values["key_head_dim"] if geometry_valid else None,
            v_head_dim=values["value_head_dim"] if geometry_valid else None,
            conv_kernel_size=(values["conv_kernel"]
                              if geometry_valid else None),
            mask="unknown",
            position_kind="unknown",
            position_application="unknown",
            cached=None,
        )
        consumer.project_fact(
            mixer_owner, fact_key="diffusion_gated_delta_geometry",
            mechanism="diffusion_gated_delta_geometry",
            value={
                "kind": spec.kind,
                "geometry_valid": geometry_valid,
                "key_heads": spec.num_kv_heads,
                "value_heads": spec.num_heads,
                "key_head_dim": spec.head_dim,
                "value_head_dim": spec.v_head_dim,
                "conv_kernel": spec.conv_kernel_size,
            }, source_owner=result.value.mixer_occurrence,
            source_spans=block.evidence.spans)
        rows.append(spec)
    return tuple(rows)


def _ffn_spec(block: DiffusionBlockProjection, owner: str,
              consumer: _OperandConsumer) -> FFNSpec:
    if block.ffn is None:
        return FFNSpec(kind=None, gated=None)
    projected = {
        "kind": "dense",
        "gated": block.ffn.gated,
        "projection_mode": block.ffn.projection_mode,
        "activation": block.ffn.activation,
    }
    for key, mechanism in (
        ("mechanism", "diffusion_ffn_mechanism"),
        ("activation", "diffusion_ffn_activation"),
    ):
        for number, operand in consumer.rows(f"{owner}.ffn", key):
            consumer.bind_operand_fact(
                number, operand, mechanism=mechanism,
                expected_value=projected)
    spec = FFNSpec(
        kind="dense",
        gated=block.ffn.gated,
        activation=block.ffn.activation,
        # A config-selected activation is code-and-config evidence, not a class
        # fixed default.  Keep the old compatibility flag truthful until F4
        # replaces it with the typed fact status everywhere.
        activation_from_class=(
            block.ffn.activation is not None
            and not block.ffn.activation_config_path),
        projection_mode=block.ffn.projection_mode,
    )
    value = {"kind": spec.kind, "gated": spec.gated,
             "projection_mode": spec.projection_mode,
             "activation": spec.activation}
    for mechanism in (
            "diffusion_ffn_mechanism", "diffusion_ffn_activation"):
        consumer.project_fact(
            f"{owner}.ffn", fact_key="diffusion_ffn_mechanism",
            mechanism=mechanism, value=value,
            source_owner=block.ffn.evidence.owner_occurrence,
            source_spans=block.evidence.spans)
    return spec


def _cell(block, owner, consumer):
    norm_kind = block.norm_kind or "unknown"
    placement = block.norm_placement or "unknown"
    topology = block.residual_topology or "unknown"
    parallel_count = None
    residual_scale = None
    if block.norm_kind is not None:
        consumer.project_fact(
            f"{owner}.cell", fact_key="diffusion_norm_mechanism",
            mechanism="diffusion_norm_mechanism", value=norm_kind,
            source_owner=block.evidence.stack.block_occurrence,
            source_spans=block.evidence.spans)
    result = block.evidence.cell_topology_result
    if result.status == "resolved" and isinstance(
            result.value, DecoderCellTopologyEvidence):
        cell = result.value
        parallel_count = cell.parallel_input_norm_count
        residual_scale = cell.residual_scale_value
        projected = {
            "norm_placement": placement,
            "residual_topology": topology,
            "parallel_norm_count": parallel_count,
            "residual_scale": residual_scale,
        }
        rows = consumer.rows(f"{owner}.cell", "topology")
        for number, operand in rows:
            # A config-bound scale supplies its exact numeric result. Other
            # selector operands decide the already-projected topology.
            if cell.residual_scale_path == operand.path:
                residual_scale = operand.value
                projected["residual_scale"] = residual_scale
            consumer.bind_operand_fact(
                number, operand, mechanism="diffusion_cell_topology",
                expected_value=projected)
        consumer.project_fact(
            f"{owner}.cell", fact_key="diffusion_cell_topology",
            mechanism="diffusion_cell_topology",
            value={
                "norm_placement": placement,
                "residual_topology": topology,
                "parallel_norm_count": parallel_count,
                "residual_scale": residual_scale,
            }, source_owner=cell.block_occurrence,
            source_spans=block.evidence.spans)
    return norm_kind, placement, topology, parallel_count, residual_scale


def project_diffusion_ir(
        bound: BoundDiffusionSourceProjection) -> DiffusionIRProjection:
    """Consume F2 operands once and produce the only U10-F3 layer templates."""
    if not isinstance(bound, BoundDiffusionSourceProjection):
        raise TypeError("F3 requires a BoundDiffusionSourceProjection")
    consumer = _OperandConsumer(bound)
    templates = []
    unresolved = []
    topology = (bound.source.topology_result.value
                if bound.source.topology_result.has_value else None)
    if topology is not None:
        topology_spans = tuple(dict.fromkeys((
            *(stage.loop.span for stage in topology.stages),
            *((topology.skip_route.spans)
              if topology.skip_route is not None else ()),
        )))
        consumer.project_fact(
            "denoiser", fact_key="diffusion_root_topology",
            mechanism="diffusion_root_topology", value=topology.kind,
            source_owner=topology.owner, source_spans=topology_spans)
    bookend_spans = tuple(dict.fromkeys(
        span for application in bound.source.bookends.applications
        for span in (*application.operation_spans, *application.route_spans)))
    if bookend_spans:
        consumer.project_code_fact(
            "denoiser", fact_key="diffusion_bookend_operations",
            mechanism="diffusion_bookend_operations",
            value=diffusion_bookend_fact_value(bound),
            source_owner=bound.source.component_root,
            source_spans=bookend_spans, receipts=())
    geometry_value = diffusion_bookend_geometry_fact_value(bound)
    geometry_rows = consumer.rows("denoiser", "bookend_geometry")
    if geometry_rows:
        for number, operand in geometry_rows:
            consumer.bind_operand_fact(
                number, operand, mechanism="diffusion_bookend_geometry",
                expected_value=geometry_value)
        consumer.project(
            "denoiser", mechanism="diffusion_bookend_geometry",
            actual_value=geometry_value)
    root_container_records = set()
    if topology is not None and topology.kind == "repeated_stack":
        for stage in topology.stages:
            for address in stage.container_records:
                record = getattr(address, "record", None)
                if record is not None:
                    root_container_records.add(record)
                root_container_records.update(getattr(address, "records", ()))
    for stack_index, block in enumerate(bound.source.blocks):
        owner = f"denoiser.stacks[{stack_index}]"
        stack = block.evidence.stack
        root_stage = (
            stack.owner_occurrence == bound.source.component_root
            and stack.container.record in root_container_records)
        stack_variant = None
        if stack.selection is not None:
            stack_variant = {
                "selected_branch": stack.selection.selected_branch,
                "candidate_count": len(stack.selection.rival.records),
            }
            variant_rows = consumer.rows(owner, "stack_variant")
            if variant_rows:
                for number, operand in variant_rows:
                    consumer.bind_operand_fact(
                        number, operand, mechanism="diffusion_stack_variant",
                        expected_value=stack_variant)
                consumer.project_fact(
                    owner, fact_key="diffusion_stack_variant",
                    mechanism="diffusion_stack_variant", value=stack_variant,
                    source_owner=stack.owner_occurrence,
                    source_spans=stack.selection.spans)
            elif stack.selection.premises and all(
                    kind == "class_default"
                    for _path, kind, _value in stack.selection.premises):
                consumer.project_code_fact(
                    owner, fact_key="diffusion_stack_variant",
                    mechanism="diffusion_stack_variant", value=stack_variant,
                    source_owner=stack.owner_occurrence,
                    source_spans=stack.selection.spans,
                    status="class_default",
                    config_paths=tuple(".".join(path) for path, _kind, _value
                                       in stack.selection.premises))
            elif not stack.selection.premises:
                consumer.project_code_fact(
                    owner, fact_key="diffusion_stack_variant",
                    mechanism="diffusion_stack_variant", value=stack_variant,
                    source_owner=stack.owner_occurrence,
                    source_spans=stack.selection.spans)
            else:
                raise ValueError(
                    "a selected stack variant has no auditable evidence route")
        count = _single_operand_value(
            consumer, owner, "num_layers",
            mechanism=("diffusion_root_stack_depth" if root_stage
                       else "diffusion_nested_stack_depth"),
            transform=_positive_int)
        consumer.project(
            owner,
            mechanism=("diffusion_root_stack_depth" if root_stage
                       else "diffusion_nested_stack_depth"),
            actual_value=count)
        reasons = []
        if count is None:
            reasons.append("exact repetition count is not checkpoint-bound")
        if not root_stage:
            reasons.append(
                "positive stack is not an exact root-topology stage")

        # Source classes may contain rival guarded calls while one exact
        # construction selects only one branch.  A positively inactive call is
        # not a runtime lane; an otherwise unresolved call remains in the
        # cardinality and still blocks materialization.
        runtime_attention = tuple(
            lane for lane in block.attention if lane.runtime_active is not False)
        ordinary = tuple(
            _attention_spec(lane, f"{owner}.attention[{lane_index}]", consumer)
            for lane_index, lane in enumerate(runtime_attention))
        mixers = _gated_delta_spec(block, owner, consumer)
        lanes = (*ordinary, *mixers)
        primary = None
        primary_lane = None
        cross = None
        cross_lane = None
        materialization_blocked = False
        if len(lanes) == 1:
            primary = lanes[0]
            primary_lane = runtime_attention[0] if ordinary else None
        elif len(lanes) == 2:
            paired = tuple(zip(ordinary, runtime_attention))
            ordinary_primary = tuple(item for item in paired
                                     if not item[0].cross_attention)
            cross_rows = tuple(item for item in paired
                               if item[0].cross_attention)
            if len(ordinary_primary) == len(cross_rows) == 1:
                primary, primary_lane = ordinary_primary[0]
                cross, cross_lane = cross_rows[0]
        if primary is None:
            reasons.append(
                "attention lanes do not fit the current primary-plus-cross IR")
            materialization_blocked = True
            primary = AttentionSpec(
                kind=None, num_heads=None, num_kv_heads=None, head_dim=None,
                mask="unknown", position_kind="unknown",
                position_application="unknown", cached=None,
                variant=_stream_variant(None, lane_count=len(lanes)))

        ffn = _ffn_spec(block, owner, consumer)
        if block.ffn is None:
            reasons.append("FFN mechanism is not uniquely source-resolved")
        norm, placement, topology, parallel_count, residual_scale = _cell(
            block, owner, consumer)
        if topology == "parallel" and cross is not None:
            reasons.append(
                "parallel cell plus additive cross lane lacks one exact IR shape")
            topology = placement = "unknown"
            parallel_count = None
        hidden_size = (
            primary.num_heads * primary.head_dim
            if primary.num_heads and primary.head_dim else None)
        primary_conditioning = tuple(
            item.kind for item in (primary_lane.conditioning
                                   if primary_lane is not None else ()))
        cross_conditioning = tuple(
            item.kind for item in (cross_lane.conditioning
                                   if cross_lane is not None else ()))
        ffn_conditioning = tuple(
            item.kind for item in (block.ffn.conditioning
                                   if block.ffn is not None else ()))
        conditioning_value = {
            "attention": primary_conditioning,
            "cross_attention": cross_conditioning,
            "ffn": ffn_conditioning,
        }
        conditioning_rows = tuple((
            *(primary_lane.conditioning if primary_lane is not None else ()),
            *(cross_lane.conditioning if cross_lane is not None else ()),
            *(block.ffn.conditioning if block.ffn is not None else ()),
        ))
        # A missing FFN mechanism or unresolved norm wiring does not erase a
        # separately proven attention/conditioning application.  Receipts are
        # emitted only when this exact repeated root cell is actually drawn.
        if conditioning_rows and count is not None and root_stage \
                and not materialization_blocked:
            consumer.project_fact(
                f"{owner}.cell",
                fact_key="diffusion_conditioning_applications",
                mechanism="diffusion_conditioning_applications",
                value=conditioning_value,
                source_owner=block.evidence.stack.block_occurrence,
                source_spans=tuple(dict.fromkeys(
                    span for item in conditioning_rows for span in item.spans)))
        template = DiffusionLayerTemplate(
            source=block, owner=owner, root_stage=root_stage,
            stack_variant=stack_variant, count=count,
            attention=primary, cross_attention=cross, ffn=ffn,
            norm_kind=norm, norm_placement=placement,
            residual_topology=topology,
            parallel_norm_count=parallel_count,
            residual_scale=residual_scale, hidden_size=hidden_size,
            attention_conditioning=primary_conditioning,
            cross_attention_conditioning=cross_conditioning,
            ffn_conditioning=ffn_conditioning,
            materialization_blocked=materialization_blocked,
            unresolved=tuple(dict.fromkeys(reasons)))
        templates.append(template)
        unresolved.extend(f"{owner}: {reason}" for reason in reasons)

    layers = []
    for template in templates:
        layers.extend(template.layers(len(layers)))
    consumed = consumer.finish()
    return DiffusionIRProjection(
        bound, tuple(templates), tuple(layers), consumed, geometry_value,
        tuple(unresolved))


__all__ = [
    "DiffusionLayerTemplate",
    "DiffusionIRProjection",
    "diffusion_bookend_fact_value",
    "diffusion_bookend_geometry_fact_value",
    "project_diffusion_ir",
]
