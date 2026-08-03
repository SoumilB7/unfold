"""U2-R6 — the ONE structural-debt register (§R6).

Replaces the four separate registers (``LEGACY_EXTRAS``,
``DRAWN_UNLEDGERED_DEBT``, ``PENDING_PROJECTION_DEBT``,
``PENDING_CONFIG_CLASSIFICATION``) with a single exact schema.  A debt row is
not an allowlist entry: it names the WRITER (module + symbol + sink + target),
the CONSUMER that still needs the raw form, the U3–U14 unit that migrates it,
and a deletion condition a machine can evaluate — so "stale" and "growth" are
blocking gates, not judgment calls, and deleting/migrating a writer must
shrink the register in the same commit.

Deletion conditions are a closed predicate DSL, never prose::

    fact_registered:<fact_key>       a FactDefinition with this key exists
    fact_routed:<fact_key>           ... and it carries >=1 ProjectionRoute
    status_retired:<fact_key>:<st>   <st> no longer in allowed_statuses
    unknown_policy_retired:<key>     unknown_policy != 'legacy_convention'
    no_writer:<sink>:<target>        NO census writer of this (sink, target)
    writer_gone:<mod>::<sym>:<sink>:<target>  this exact census key is gone
    classified:<config_path>         the OWNER-EXACT classification table
                                     contains (row.owner, path)
    symbol_deleted:<module>::<sym>   the named production symbol is gone

Every verb evaluates against the live registry/census — a condition that is
already true means the row (and its writer) should have been deleted in the
same commit, and the ``satisfied`` gate blocks.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------- #
# Closed vocabularies
# --------------------------------------------------------------------------- #
# U3–U14 only (§R6: "Every row must be assigned to U3–U14 or deleted now").
MIGRATION_UNITS = frozenset(f"U{i}" for i in range(3, 15))

# The census sink kinds, plus the two debt families that are not raw-write
# sinks: a leaf the renderer draws without a ledger fact ("drawn_leaf" — the
# former DRAWN_UNLEDGERED_DEBT and legacy_convention statuses), and a config
# occurrence still awaiting its consumer or classification ("config_read" —
# the former PENDING_PROJECTION_DEBT / PENDING_CONFIG_CLASSIFICATION).
SINK_KINDS = frozenset({
    "ledger", "spec", "spec_field", "extras", "opgraph", "card", "params",
    "drawn_leaf", "config_read", "consumer_read",
})

_CONDITION_VERBS = frozenset({
    "fact_registered", "fact_routed", "status_retired",
    "unknown_policy_retired", "no_writer", "writer_gone", "classified",
    "symbol_deleted",
})


def _parse_condition(condition: str) -> tuple[str, tuple[str, ...]]:
    """Split a deletion condition into (verb, args) and validate its shape.

    Raises ``ValueError`` on prose, an unknown verb, or a wrong arity — a row
    with an uncheckable deletion condition cannot be constructed at all.
    """
    if not condition or ":" not in condition:
        raise ValueError(
            f"deletion_condition {condition!r} is not a checkable predicate "
            f"(expected '<verb>:<args>' with verb in {sorted(_CONDITION_VERBS)})")
    verb, _, rest = condition.partition(":")
    if verb not in _CONDITION_VERBS:
        raise ValueError(
            f"deletion_condition verb {verb!r} not in the closed set "
            f"{sorted(_CONDITION_VERBS)} — prose conditions are not checkable")
    if verb == "symbol_deleted":
        module, sep, symbol = rest.partition("::")
        if not sep or not module or not symbol:
            raise ValueError(
                f"symbol_deleted needs '<module>::<symbol>', got {rest!r}")
        return verb, (module, symbol)
    if verb == "writer_gone":
        module, sep, tail = rest.partition("::")
        parts = tail.split(":") if sep else []
        if not module or len(parts) != 3 or not all(parts):
            raise ValueError(
                f"writer_gone needs '<module>::<symbol>:<sink>:<target>', "
                f"got {rest!r}")
        return verb, (module, *parts)
    args = tuple(a for a in rest.split(":") if a)
    arity = {"fact_registered": 1, "fact_routed": 1, "status_retired": 2,
             "unknown_policy_retired": 1, "no_writer": 2, "classified": 1}[verb]
    if len(args) != arity:
        raise ValueError(
            f"deletion_condition {condition!r}: {verb} takes {arity} "
            f"arg(s), got {len(args)}")
    return verb, args


@dataclass(frozen=True)
class StructuralDebt:
    """One exact structural-debt row (§R6 schema, all ten fields)."""

    owner: str                     # owner path whose structure the write claims
    source_occurrence: str | None  # config path/occurrence backing it, if any
    writer_module: str             # production module of the raw writer
    writer_symbol: str             # enclosing function/method of the writer
    sink_kind: str                 # SINK_KINDS member
    structural_target: str         # EXACT target — never a family prefix
    reason: str                    # why the raw form still exists
    last_consumer: str             # '<module>::<symbol>' that still reads it
    migration_unit: str            # U3–U14 (H7/H8/UNASSIGNED are rejected)
    deletion_condition: str        # closed-DSL predicate (see module docstring)
    census_target: str | None = None  # census key when the write site is
    #   dynamically keyed ('<dynamic>'/'<update>') — the row stays EXACT about
    #   what is written while joining the writer census on the real site.

    def __post_init__(self) -> None:
        for field in ("owner", "writer_module", "writer_symbol",
                      "structural_target", "reason", "last_consumer"):
            if not getattr(self, field):
                raise ValueError(f"StructuralDebt.{field} may not be empty "
                                 f"({self.sink_kind}:{self.structural_target})")
        if self.sink_kind not in SINK_KINDS:
            raise ValueError(f"sink_kind {self.sink_kind!r} not in "
                             f"{sorted(SINK_KINDS)}")
        if self.migration_unit not in MIGRATION_UNITS:
            raise ValueError(
                f"{self.sink_kind}:{self.structural_target}: migration_unit "
                f"{self.migration_unit!r} is not U3–U14 — H-era units, "
                f"'scoped' and UNASSIGNED block U2 (§R6)")
        if "::" not in self.last_consumer:
            raise ValueError(
                f"{self.sink_kind}:{self.structural_target}: last_consumer "
                f"{self.last_consumer!r} must be '<module>::<symbol>' so the "
                f"consumer join is checkable")
        _parse_condition(self.deletion_condition)

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        """Row identity: one row per (sink, target, WRITER, owner, occurrence)
        — a second author of one target is a second debt, and two owners'
        reads of one path (or one owner's two exact paths) are two debts,
        never hidden under the first row."""
        return (self.sink_kind, self.structural_target,
                self.writer_module, self.writer_symbol,
                self.owner, self.source_occurrence or "")

    @property
    def writer_key(self) -> tuple[str, str, str, str]:
        return (self.writer_module, self.writer_symbol, self.sink_kind,
                self.census_target or self.structural_target)


# --------------------------------------------------------------------------- #
# The register — every row EXACT (owner + writer + sink + target + consumer +
# U3–U14 unit + checkable deletion condition).  Replaces LEGACY_EXTRAS,
# DRAWN_UNLEDGERED_DEBT, PENDING_PROJECTION_DEBT and
# PENDING_CONFIG_CLASSIFICATION (U2-R6; row content from the reviewed
# writer/consumer audit in z-docs/07-current-state/r6-debt-rows.md).
# --------------------------------------------------------------------------- #
_TP = "model_unfolder/adapters/transformer/parser.py"
_DP = "model_unfolder/adapters/diffusor/parser.py"
_TA = "model_unfolder/adapters/transformer/assembly.py"
_MB = ("model_unfolder/adapters/transformer/special_parts/modalities/"
       "builder.py")
_OG = "model_unfolder/opgraph.py"
_AT = "model_unfolder/adapters/transformer/blocks/attention.py"
# "JSON document only": no renderer reads the raw key — it reaches the product
# solely through ModelIR.to_dict serialization (the U14 JSON-contract surface).
_JSON_DOC = "model_unfolder/ir.py::to_dict"
_EXCUSAL = "model_unfolder/parser.py::config_to_ir"


def _extras(target, owner, reason, unit, condition, *, occurrence=None,
            consumer=_JSON_DOC, module=_TP, symbol="parse",
            census=None) -> StructuralDebt:
    return StructuralDebt(owner=owner, source_occurrence=occurrence,
                          writer_module=module, writer_symbol=symbol,
                          sink_kind="extras", structural_target=target,
                          reason=reason, last_consumer=consumer,
                          migration_unit=unit, deletion_condition=condition,
                          census_target=census)


def _drawn(target, owner, reason, unit, condition, *, occurrence,
           consumer, module, symbol) -> StructuralDebt:
    return StructuralDebt(owner=owner, source_occurrence=occurrence,
                          writer_module=module, writer_symbol=symbol,
                          sink_kind="drawn_leaf", structural_target=target,
                          reason=reason, last_consumer=consumer,
                          migration_unit=unit, deletion_condition=condition)


def _config(target, owner, path, reason, unit, condition) -> StructuralDebt:
    return StructuralDebt(owner=owner, source_occurrence=path,
                          writer_module="model_unfolder/parser.py",
                          writer_symbol="config_to_ir",
                          sink_kind="config_read", structural_target=target,
                          reason=reason, last_consumer=_EXCUSAL,
                          migration_unit=unit, deletion_condition=condition)


STRUCTURAL_DEBT: tuple[StructuralDebt, ...] = (
    # ---- raw ``ir.extras`` writes (former LEGACY_EXTRAS, per-target EXACT:
    # ---- a top-level row excuses nothing below it) ------------------------ #
    _extras("moe", "root.decoder.layer[i].ffn",
            "per-layer MoE schedule (experts/shared/top-k) as raw extras",
            "U7", "fact_registered:moe_schedule",
            occurrence="num_experts + num_experts_per_tok + "
                       "num_shared_experts (text_cfg)"),
    _extras("moe.every_layer", "root.decoder.layer[i].ffn",
            "which layers are MoE vs dense, as a raw extras leaf",
            "U7", "fact_registered:moe_schedule"),
    _extras("moe.num_experts", "root.decoder.layer[i].ffn",
            "routed expert count as a raw extras leaf",
            "U7", "fact_registered:moe_schedule",
            occurrence="text_cfg num_experts"),
    _extras("moe.num_experts_per_tok", "root.decoder.layer[i].ffn",
            "per-token expert fan-out as a raw extras leaf",
            "U7", "fact_registered:moe_schedule",
            occurrence="text_cfg num_experts_per_tok"),
    _extras("moe.num_shared_experts", "root.decoder.layer[i].ffn",
            "shared-expert count as a raw extras leaf",
            "U7", "fact_registered:moe_schedule",
            occurrence="text_cfg num_shared_experts"),
    _extras("mtp", "root",
            "multi-token-prediction module structure as raw extras",
            "U7", "fact_registered:mtp",
            occurrence="text_cfg num_nextn_predict_layers | num_mtp_layers",
            consumer="model_unfolder/renderers/html/views.py::"
                     "_build_architecture_view"),
    _extras("mtp.num_modules", "root",
            "MTP module count as a raw extras leaf", "U7",
            "fact_registered:mtp"),
    _extras("mtp.predicts_extra_tokens", "root",
            "MTP extra-token count as a raw extras leaf", "U7",
            "fact_registered:mtp"),
    _extras("mtp.shares_embedding", "root",
            "MTP embedding-sharing flag as a raw extras leaf", "U7",
            "fact_registered:mtp"),
    _extras("mtp.shares_output_head", "root",
            "MTP head-sharing flag as a raw extras leaf", "U7",
            "fact_registered:mtp"),
    _extras("sliding_window", "root.decoder.attention",
            "sliding-window schedule (window + first-full layers)",
            "U8", "fact_registered:window_schedule",
            occurrence="text_cfg sliding_window/use_sliding_window/"
                       "max_window_layers"),
    _extras("sliding_window.window", "root.decoder.attention",
            "window size as a raw extras leaf", "U8",
            "fact_registered:window_schedule"),
    _extras("sliding_window.first_full_layers", "root.decoder.attention",
            "first-full-attention layer count as a raw extras leaf", "U8",
            "fact_registered:window_schedule"),
    _extras("dual_kv", "root.decoder.attention",
            "dual global/sliding KV schedule as raw extras",
            "U8", "fact_registered:kv_schedule",
            occurrence="text_cfg num_global_key_value_heads | "
                       "global_head_dim"),
    _extras("dual_kv.global", "root.decoder.attention",
            "global-lane KV geometry as a raw extras leaf", "U8",
            "fact_registered:kv_schedule"),
    _extras("dual_kv.sliding", "root.decoder.attention",
            "sliding-lane KV geometry as a raw extras leaf", "U8",
            "fact_registered:kv_schedule"),
    _extras("irope", "root.decoder.attention",
            "interleaved-RoPE (NoPE-interval) schedule as raw extras",
            "U8", "fact_routed:position_kind",
            occurrence="text_cfg no_rope_layer_interval"),
    _extras("irope.no_rope_interval", "root.decoder.attention",
            "NoPE interval as a raw extras leaf", "U8",
            "fact_routed:position_kind"),
    _extras("num_kv_shared_layers", "root.decoder.attention",
            "cross-layer KV-sharing count as raw extras (drawn sharing "
            "comes from CrossLayerEdge spec fields)",
            "U8", "fact_registered:kv_shared_layers",
            occurrence="text_cfg num_kv_shared_layers"),
    _extras("partial_rotary_factor", "root.decoder.attention",
            "partial-rotary fraction as raw extras (drawn partial rotary "
            "comes from AttentionSpec.rope_dim)",
            "U8", "fact_registered:partial_rotary",
            occurrence="text_cfg partial_rotary_factor | "
                       "rope_scaling.partial_rotary_factor"),
    _extras("position_encoding", "root.decoder.attention",
            "code-derived position-mechanism descriptor as raw extras",
            "U8", "fact_routed:position_kind",
            occurrence="_code_position_evidence (modeling source)",
            consumer="model_unfolder/evidence/conformance.py::"
                     "_drawn_position_kinds"),
    _extras("rope", "root.decoder.attention",
            "RoPE theta/scaling descriptor as raw extras (drawn rope reads "
            "AttentionSpec.rope via attention_detail)",
            "U8", "fact_registered:rope_theta",
            occurrence="text_cfg rope_theta + rope_parameters|rope_scaling"),
    _extras("softcap", "model",
            "final vocabulary-logit softcap descriptor as raw extras; the "
            "attention cap and query-score operand have left this legacy lane",
            "U14", "fact_routed:final_logit_softcap",
            occurrence="text_cfg final_logit_softcapping"),
    _extras("block_diffusion", "root",
            "block-diffusion canvas descriptor as raw extras",
            "U10", "fact_registered:block_diffusion_canvas",
            occurrence="cfg canvas_length",
            consumer="model_unfolder/renderers/html/views_diffusion.py::"
                     "_build_block_diffusion_view"),
    _extras("block_diffusion.canvas_length", "root",
            "canvas length as a raw extras leaf", "U10",
            "fact_registered:block_diffusion_canvas",
            consumer="model_unfolder/renderers/html/views_diffusion.py::"
                     "_build_block_diffusion_view"),
    _extras("codebooks", "root.decoder",
            "audio K-codebook structure as raw extras (render blocks consume "
            "the codebooks ARG, not this key)",
            "U7", "fact_registered:codebooks",
            occurrence="text_cfg num_codebooks + audio_channels + "
                       "decoder_codebook_streams_for_path",
            module=_TA, symbol="decoder_extras"),
    _extras("render", "root",
            "PRESENTATION render-spec (theme/layout/blocks) — renderer "
            "semantics, not architecture; retires by reclassification to a "
            "scoped presentation register, not by becoming a fact",
            "U5", "no_writer:extras:render",
            module=_TA, symbol="decoder_extras",
            consumer="model_unfolder/renderers/html/sections.py::"
                     "_stats_banner"),
    _extras("render", "root",
            "UNet-side render-spec author (same PRESENTATION debt)",
            "U5", "no_writer:extras:render",
            module=_DP, symbol="_parse_unet_model",
            consumer="model_unfolder/renderers/html/sections.py::"
                     "_stats_banner"),
    _extras("render", "root",
            "DiT-side render-spec author (same PRESENTATION debt)",
            "U5", "no_writer:extras:render",
            module=_DP, symbol="parse",
            consumer="model_unfolder/renderers/html/sections.py::"
                     "_stats_banner"),
    _extras("modalities", "root",
            "multimodal tower descriptors merged into extras via the "
            "dynamic-keyed _merge_extras loop",
            "U9", f"writer_gone:{_TA}::_merge_extras:extras:<dynamic>",
            occurrence="vision/audio/video sub-configs + source-bundle "
                       "tower/projector/fusion evidence",
            module=_TA, symbol="_merge_extras", census="<dynamic>",
            consumer="model_unfolder/evidence/conformance.py::"
                     "_check_vision_facts"),
    _extras("modalities.inputs", "root",
            "modality input-path payload authored by the multimodal builder",
            "U9", f"writer_gone:{_MB}::multimodal_extras:extras:<dynamic>",
            occurrence="vision/audio/video sub-configs",
            module=_MB, symbol="multimodal_extras", census="<dynamic>",
            consumer="model_unfolder/evidence/conformance.py::"
                     "_check_vision_facts"),
    _extras("diffusion", "root",
            "UNet meta descriptor (channels/text-encoders/scheduler) as raw "
            "extras", "U11", "fact_registered:diffusion_meta",
            occurrence="unet config boc/in_channels/cross_attention_dim + "
                       "pipeline text_encoders/scheduler",
            module=_DP, symbol="_parse_unet_model",
            consumer="model_unfolder/renderers/html/sections.py::"
                     "_diffusion_stats"),
    _extras("diffusion", "root",
            "DiT/MMDiT meta descriptor (stream counts/dims/text-encoders/"
            "scheduler) as raw extras", "U10",
            "fact_registered:diffusion_meta",
            occurrence="denoiser config stream/dim fields + pipeline "
                       "text_encoders/scheduler",
            module=_DP, symbol="parse",
            consumer="model_unfolder/renderers/html/sections.py::"
                     "_diffusion_stats"),
    _extras("unet", "root",
            "full UNet block-structure descriptor as raw extras",
            "U11", "fact_registered:unet_blocks",
            occurrence="UNet config geometry assembled by diffusor/unet.py",
            module=_DP, symbol="_parse_unet_model",
            consumer="model_unfolder/renderers/html/block_views/unet.py::"
                     "build_unet_view"),
    # ---- drawn-but-unledgered leaves (former DRAWN_UNLEDGERED_DEBT) ------- #
    _drawn("position_kind", "root.decoder.attention",
           "positional scheme (rope/alibi/learned/none) drawn from the "
           "AttentionSpec tri-state; no ledger writer yet",
           "U8", "fact_routed:position_kind",
           occurrence="AttentionSpec.position_kind (tri-state)",
           module=_OG, symbol="_sdpa_region",
           consumer="model_unfolder/renderers/html/fact_projection.py::"
                    "attention_facts"),
    _drawn("qk_norm", "root.decoder.attention",
           "the uniform owner-level Q/K-norm fact is routed in U6, but a "
           "heterogeneous layer schedule still projects from per-layer specs "
           "without occurrence-qualified facts; U8 must close that schedule",
           "U8", "fact_registered:qk_norm_schedule",
           occurrence="AttentionSpec.qk_norm on a mixed layer schedule",
           module=_AT, symbol="_sdpa_detailed_child_blocks",
           consumer="model_unfolder/renderers/html/fact_projection.py::"
                    "attention_facts"),
    # ---- config occurrences awaiting their consumer (former
    # ---- PENDING_PROJECTION_DEBT; owner + EXACT dotted path) -------------- #
    _config("the conditioning card on the denoiser view",
            "root.denoiser", "max_sequence_length",
            "max text-token sequence the denoiser conditions on (Mochi) — "
            "a declared conditioning limit",
            "U10", "fact_registered:denoiser_conditioning_limit"),
    # U4-C exposes these declarations instead of consuming them as FFN
    # architecture on sight. They remain exact, owner-qualified input debt
    # until U7/U10 proves the corresponding constructor dispatch. A row for
    # one encoder slot or spelling excuses no sibling occurrence.
    _config("the decoder FFN activation dispatch",
            "root", "hidden_act",
            "activation declaration is visible but the exact decoder FFN "
            "reader has not yet bound every supported occurrence that reads it",
            "U7", "fact_registered:ffn_activation_binding"),
    _config("the decoder routed-FFN layer schedule",
            "root", "first_k_dense_replace",
            "the checkpoint supplies the dense-prefix value, but U7 must bind "
            "that path to the exact source schedule before consuming it",
            "U7", "fact_registered:moe_schedule_binding"),
    _config("the decoder routed-FFN layer schedule",
            "root", "moe_layer_freq",
            "the checkpoint supplies the routed-layer frequency, but U7 must "
            "bind that path to the exact source schedule before consuming it",
            "U7", "fact_registered:moe_schedule_binding"),
    _config("the decoder routed-FFN layer schedule",
            "root", "interleave_moe_layer_step",
            "the checkpoint supplies an interleave interval, but U7 must bind "
            "that path to the exact source schedule before consuming it",
            "U7", "fact_registered:moe_schedule_binding"),
    _config("the decoder routed-FFN layer schedule",
            "root", "decoder_sparse_step",
            "the checkpoint supplies a sparse-layer step, but U7 must bind "
            "that path to the exact source schedule before consuming it",
            "U7", "fact_registered:moe_schedule_binding"),
    _config("the decoder routed-FFN layer schedule",
            "root", "mlp_only_layers",
            "the checkpoint supplies an ordinary-MLP membership list, but U7 "
            "must bind it to the exact source schedule before consuming it",
            "U7", "fact_registered:moe_schedule_binding"),
    _config("the decoder routed-FFN layer schedule",
            "root", "moe_layers",
            "the checkpoint supplies routed-layer membership, but U7 must bind "
            "that path to the exact source schedule before consuming it",
            "U7", "fact_registered:moe_schedule_binding"),
    _config("the decoder routed-FFN layer schedule",
            "root", "moe_layers_enum",
            "the checkpoint supplies a serialized membership set, but U7 must "
            "bind that path to the exact source schedule before consuming it",
            "U7", "fact_registered:moe_schedule_binding"),
    _config("the routed-expert activation clipping operand",
            "root", "swiglu_limit",
            "the clipping value is visible, but U7 must bind it to the exact "
            "expert activation callable before it may enter a fact or view",
            "U7", "fact_registered:expert_activation_clip"),
    _config("the text-encoder FFN activation dispatch",
            "root.text_encoder", "hidden_act",
            "activation declaration is visible but the exact encoder FFN "
            "reader does not yet prove this occurrence consumes it",
            "U7", "fact_registered:ffn_activation_binding"),
    _config("the text-encoder FFN activation dispatch",
            "root.text_encoder", "dense_act_fn",
            "T5-style activation declaration awaits exact dataflow through "
            "the encoder FFN owner",
            "U7", "fact_registered:ffn_activation_binding"),
    _config("the text-encoder FFN mechanism dispatch",
            "root.text_encoder", "feed_forward_proj",
            "combined activation/gate declaration awaits exact source binding",
            "U7", "fact_registered:ffn_mechanism_binding"),
    _config("the text-encoder FFN gate dispatch",
            "root.text_encoder", "is_gated_act",
            "gate declaration awaits exact source binding",
            "U7", "fact_registered:ffn_gate_binding"),
    _config("the second text-encoder FFN activation dispatch",
            "root.text_encoder_2", "hidden_act",
            "activation declaration is visible but the exact encoder FFN "
            "reader does not yet prove this occurrence consumes it",
            "U7", "fact_registered:ffn_activation_binding"),
    _config("the second text-encoder FFN activation dispatch",
            "root.text_encoder_2", "dense_act_fn",
            "T5-style activation declaration awaits exact dataflow through "
            "the encoder FFN owner",
            "U7", "fact_registered:ffn_activation_binding"),
    _config("the second text-encoder FFN mechanism dispatch",
            "root.text_encoder_2", "feed_forward_proj",
            "combined activation/gate declaration awaits exact source binding",
            "U7", "fact_registered:ffn_mechanism_binding"),
    _config("the second text-encoder FFN gate dispatch",
            "root.text_encoder_2", "is_gated_act",
            "gate declaration awaits exact source binding",
            "U7", "fact_registered:ffn_gate_binding"),
    _config("the third text-encoder FFN activation dispatch",
            "root.text_encoder_3", "dense_act_fn",
            "T5-style activation declaration awaits exact dataflow through "
            "the encoder FFN owner",
            "U7", "fact_registered:ffn_activation_binding"),
    _config("the third text-encoder FFN mechanism dispatch",
            "root.text_encoder_3", "feed_forward_proj",
            "combined activation/gate declaration awaits exact source binding",
            "U7", "fact_registered:ffn_mechanism_binding"),
    _config("the third text-encoder FFN gate dispatch",
            "root.text_encoder_3", "is_gated_act",
            "gate declaration awaits exact source binding",
            "U7", "fact_registered:ffn_gate_binding"),
    _config("the conditioning-encoder FFN activation dispatch",
            "root.conditioning", "dense_act_fn",
            "T5-style activation declaration awaits exact dataflow through "
            "the conditioning encoder FFN owner",
            "U7", "fact_registered:ffn_activation_binding"),
    _config("the conditioning-encoder FFN mechanism dispatch",
            "root.conditioning", "feed_forward_proj",
            "combined activation/gate declaration awaits exact source binding",
            "U7", "fact_registered:ffn_mechanism_binding"),
    _config("the conditioning-encoder FFN gate dispatch",
            "root.conditioning", "is_gated_act",
            "gate declaration awaits exact source binding",
            "U7", "fact_registered:ffn_gate_binding"),
    _config("the denoiser FFN activation dispatch",
            "root.denoiser", "activation_fn",
            "the activation operand is visible, but only an exact denoiser "
            "constructor dispatch may project it as FFN mechanism evidence",
            "U10", "fact_registered:dit_ffn_activation_binding"),
    _config("the VAE-decoder ResNet cells' activation chip",
            "root.vae", "_vae_config.act_fn",
            "the VAE decoder's convolution activation (video VAEs) — a "
            "constructor record",
            "U12", "fact_registered:vae_act_fn"),
    _config("the VAE latent-depth / temporal-axis chip",
            "root.vae", "_vae_config.temporal_compression_ratio",
            "the VAE's own temporal compression (HunyuanVideo/Wan) — "
            "distinct from the denoiser-level ratio",
            "U12", "fact_registered:vae_temporal_compression"),
    _config("the VAE encoder intake chip",
            "root.vae", "_vae_config.in_channels",
            "VAE encoder input channels — awaits the source-derived VAE "
            "component graph (V-02)",
            "U12", "fact_registered:vae_in_channels"),
    _config("the VAE sample-geometry chip (height)",
            "root.vae", "_vae_config.sample_height",
            "VAE declared sample height (CogVideoX) — V-02",
            "U12", "fact_registered:vae_sample_geometry"),
    _config("the VAE sample-geometry chip (width)",
            "root.vae", "_vae_config.sample_width",
            "VAE declared sample width (CogVideoX) — V-02",
            "U12", "fact_registered:vae_sample_geometry"),
    _config("the VAE patchify chip",
            "root.vae", "_vae_config.patch_size",
            "patchified VAE spatial patch (FLUX.2/LTX) — V-03",
            "U12", "fact_registered:vae_patchify"),
    _config("the VAE temporal-patchify chip",
            "root.vae", "_vae_config.patch_size_t",
            "patchified VAE temporal patch (LTX) — V-03",
            "U12", "fact_registered:vae_patchify"),
    _config("the VAE decoder attention chip",
            "root.vae", "_vae_config.attention_head_dim",
            "DC-AE decoder attention head width (Sana) — V-05",
            "U12", "fact_registered:vae_decoder_attention"),
    # Qwen3.5's position declaration remains exact U8 work.  Its five
    # recurrent-mixer geometry paths left this register in U6 once the exact
    # split/reshape/repeat/Conv1d/recurrent protocol bound them.
    _config("the interleaved multimodal rotary application", "root",
            "rope_parameters.mrope_interleaved",
            "the checkpoint declaration is visible, but U8 must bind it to "
            "the exact rotary application before projecting the schedule",
            "U8", "fact_registered:mrope_interleaving"),
    _config("the denoiser projection-bias fact", "root.denoiser",
            "attention_bias",
            "declaration awaits exact denoiser projection-construction binding",
            "U10", "fact_registered:denoiser_attention_bias"),
    _config("the denoiser Q/K-normalization fact", "root.denoiser",
            "qk_norm",
            "declaration awaits exact denoiser Q/K norm execution binding",
            "U10", "fact_registered:qk_norm"),
    _config("the denoiser rotary-application fact", "root.denoiser",
            "use_rotary_positional_embeddings",
            "declaration awaits exact denoiser positional call binding; "
            "it cannot independently author Q/K rotation",
            "U10", "fact_registered:position_kind"),
    _config("the denoiser rotary-geometry fact", "root.denoiser",
            "rope_theta",
            "numeric rotary operand awaits its U8 owner-bound position fact",
            "U8", "fact_registered:rope_theta"),
    _config("the UNet ResNet-cell norm chip",
            "root.denoiser", "norm_num_groups",
            "GroupNorm group count declared on UNet/legacy-DiT denoisers — "
            "U-06 derives the cell norm from source",
            "U11", "fact_registered:unet_cell_norm"),
    _config("the projector output-width chip",
            "root.vision", "vision_config.hidden_size",
            "the vision config's declared merger/output width (qwen2-vl: "
            "3584 beside internal embed_dim=1280) — a config comparison may "
            "NOT infer its meaning.  COR-4: where construction evidence "
            "exists the source-bound projector CONSUMES it exactly (fact "
            "projector_out_features), discharging the occurrence; it stays "
            "visible debt only for source-less grid towers",
            "U9", "fact_registered:hidden_size"),
    _config("the latent scale chip (VAE-owned)",
            "root.denoiser", "scaling_factor",
            "pipeline-level latent-scale duplicate on the denoiser config "
            "(Lumina); the VAE's own scaling_factor is the drawn read",
            "U12", "fact_registered:latent_io"),
    # ---- explicit-null occurrences awaiting classification (former
    # ---- PENDING_CONFIG_CLASSIFICATION) ----------------------------------- #
    _config("feature-absence classification (VQ-embedding absent)",
            "root.denoiser", "num_vector_embeds",
            "explicit-null VQ-embedding declaration on the legacy DiT "
            "config", "U11", "classified:num_vector_embeds"),
    _config("pipeline watermark flag — no structural target; candidate for "
            "scoped-ignore deletion",
            "root.denoiser", "add_watermarker",
            "explicit-null pipeline watermark flag on the UNet config",
            "U11", "classified:add_watermarker"),
    _config("class-conditioning absence classification",
            "root.denoiser", "class_embed_type",
            "explicit-null class-conditioning declaration (feature absent)",
            "U11", "classified:class_embed_type"),
    _config("cross-attention norm absence classification",
            "root.denoiser", "cross_attention_norm",
            "explicit-null cross-attention norm declaration",
            "U11", "classified:cross_attention_norm"),
    _config("mid-block attention-mode absence classification",
            "root.denoiser", "mid_block_only_cross_attention",
            "explicit-null mid-block attention-mode declaration",
            "U11", "classified:mid_block_only_cross_attention"),
    _config("class-embedding-count absence classification",
            "root.denoiser", "num_class_embeds",
            "explicit-null class-embedding count declaration",
            "U11", "classified:num_class_embeds"),
    _config("timestep-embedding width — binds to the source-derived "
            "time-embed cell when non-null",
            "root.denoiser", "time_embedding_dim",
            "explicit-null timestep-embedding width declaration",
            "U11", "classified:time_embedding_dim"),
    # is_encoder_decoder rows DELETED (Soumil's final vet): the field is
    # ARCHITECTURE, consumed into decoder.attention.mask at its deciding
    # reads — debt shrinks in the same commit as the consumption.
    # ---- U2-R7 dispositions: standing occurrences classified PENDING (one
    # ---- exact row per owner x path; readers stay visible pending debt) ---- #
    _config("the residual-tap topology annotation (post-LN tap on the layer "
            "card)", "root", "apply_residual_connection_post_layernorm",
            "declaration is inspected but cannot author the residual tap; U7 "
            "binds it to the exact owner dataflow",
            "U7", "fact_registered:residual_topology"),
    # RoPE scaling-descriptor subkeys: feed only raw extras.rope + the
    # declared-tier chip; the rope_theta fact retires the whole family.
    _config("the rope card's scaling-factor line", "root",
            "rope_parameters.factor",
            "scaling descriptor subkey feeds only raw extras.rope",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's original-context line", "root",
            "rope_parameters.original_max_position_embeddings",
            "scaling descriptor subkey feeds only raw extras.rope",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's theta line", "root",
            "rope_parameters.rope_theta",
            "feeds extras.rope + the declared-tier chip only",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's theta line (nested attention config)", "root",
            "attn_config.rope_theta",
            "feeds extras.rope + the declared-tier chip only; the exact "
            "rotary application remains U8 work",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's scaling-type line", "root",
            "rope_parameters.rope_type",
            "feeds extras.rope + the declared-tier chip only",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's scaling-type line (legacy spelling)", "root",
            "rope_parameters.type",
            "feeds extras.rope + the declared-tier chip only",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's theta line (wrapper path)", "root",
            "text_config.rope_parameters.rope_theta",
            "feeds extras.rope + the declared-tier chip only",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's scaling-type line (wrapper path)", "root",
            "text_config.rope_parameters.rope_type",
            "feeds extras.rope + the declared-tier chip only",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's scaling-type line (wrapper path, legacy "
            "spelling)", "root", "text_config.rope_parameters.type",
            "feeds extras.rope + the declared-tier chip only",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's theta line (text-encoder slot)",
            "root.text_encoder", "rope_parameters.rope_theta",
            "feeds extras.rope + the declared-tier chip only",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's scaling-type line (text-encoder slot)",
            "root.text_encoder", "rope_parameters.rope_type",
            "feeds extras.rope + the declared-tier chip only",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's theta line (text-encoder slot, wrapper path)",
            "root.text_encoder", "text_config.rope_parameters.rope_theta",
            "feeds extras.rope + the declared-tier chip only",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's scaling-type line (text-encoder slot, wrapper "
            "path)", "root.text_encoder",
            "text_config.rope_parameters.rope_type",
            "feeds extras.rope + the declared-tier chip only",
            "U8", "fact_registered:rope_theta"),
    # UNet structure reaches render only through raw extras.unet — these
    # declared fields await the U11 source-derived component graph (the
    # register's extras:unet row is the write-side twin).
    _config("the drawn UNet stage-channel ladder", "root.denoiser",
            "block_out_channels",
            "UNet structure reaches render only through raw extras.unet",
            "U11", "fact_registered:unet_blocks"),
    _config("per-stage attention placement of the drawn U (down path)",
            "root.denoiser", "down_block_types",
            "raw extras.unet only", "U11", "fact_registered:unet_blocks"),
    _config("per-stage attention placement of the drawn U (up path)",
            "root.denoiser", "up_block_types",
            "raw extras.unet only", "U11", "fact_registered:unet_blocks"),
    _config("the drawn bottleneck stage + its declared provenance",
            "root.denoiser", "mid_block_type",
            "raw extras.unet only", "U11", "fact_registered:unet_blocks"),
    _config("per-stage ResNet counts of the drawn U", "root.denoiser",
            "layers_per_block",
            "raw extras.unet only", "U11", "fact_registered:unet_blocks"),
    _config("per-stage Transformer2D depth (SDXL mid=10)", "root.denoiser",
            "transformer_layers_per_block",
            "raw extras.unet only", "U11", "fact_registered:unet_blocks"),
    _config("the drawn encoder-to-cross-attn width bridge (text_proj card)",
            "root.denoiser", "encoder_hid_dim",
            "builds the bridge card; raw extras.unet only",
            "U11", "fact_registered:unet_blocks"),
    _config("the conditioning-modality resolution (kv_label / projector / "
            "kv_text)", "root.denoiser", "encoder_hid_dim_type",
            "declared enum decides the drawn K/V modality story; lands in "
            "raw extras.diffusion.conditioning",
            "U11", "fact_registered:diffusion_meta"),
    _config("the SDXL text_time micro-conditioning card (added-cond)",
            "root.denoiser", "addition_embed_type",
            "render-only added-conditioning block today",
            "U11", "fact_registered:unet_added_conditioning"),
    _config("added-cond time-embed width (SDXL/SVD)", "root.denoiser",
            "addition_time_embed_dim",
            "render-only added-conditioning block today",
            "U11", "fact_registered:unet_added_conditioning"),
    _config("added-cond projection-in width", "root.denoiser",
            "projection_class_embeddings_input_dim",
            "render-only added-conditioning block today",
            "U11", "fact_registered:unet_added_conditioning"),
    _config("the conditioning card's max-text-tokens chip (CogVideoX "
            "spelling)", "root.denoiser", "max_text_seq_length",
            "declared conditioning limit — the same mechanism as the "
            "max_sequence_length row; chip-only today",
            "U10", "fact_registered:denoiser_conditioning_limit"),
    _config("the DiT FFN inner-width derivation", "root.denoiser",
            "ffn_dim_multiplier",
            "Lumina-family inner width = round(2/3*4h; multiple_of, "
            "ffn_dim_multiplier); chip-only today while the drawn width "
            "stays unknown until the derivation is registered",
            "U10", "fact_registered:ffn_width_derived"),
    _config("the DiT FFN inner-width derivation (rounding quantum)",
            "root.denoiser", "multiple_of",
            "pairs with ffn_dim_multiplier",
            "U10", "fact_registered:ffn_width_derived"),
    _config("the cross-attention sublayer's own head geometry",
            "root.denoiser", "num_cross_attention_heads",
            "Sana declares distinct cross heads; the drawn cross-attn "
            "sublayer reuses the SELF spec today (chip only)",
            "U10", "fact_registered:cross_attention_geometry"),
    _config("the cross-attention sublayer's own head geometry (head width)",
            "root.denoiser", "cross_attention_head_dim",
            "pairs with num_cross_attention_heads",
            "U10", "fact_registered:cross_attention_geometry"),
    # ---- raw-spelling twins the boundary fix surfaced (checkpoints that
    # ---- declare rope_scaling/rope_theta literally; same U8 family) ------- #
    _config("the rope card's theta line (raw spelling)",
            "root.text_encoder", "rope_scaling.rope_theta",
            "raw rope_scaling twin of the rope_parameters row",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's scaling-type line (raw spelling)",
            "root.text_encoder", "rope_scaling.rope_type",
            "raw rope_scaling twin", "U8", "fact_registered:rope_theta"),
    _config("the rope card's theta line (bare sibling declaration)",
            "root.text_encoder", "rope_theta",
            "top-level theta beside a scaling dict",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's theta line (wrapper, raw spelling)",
            "root.text_encoder", "text_config.rope_scaling.rope_theta",
            "raw rope_scaling twin under the text wrapper",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's scaling-type line (wrapper, raw spelling)",
            "root.text_encoder", "text_config.rope_scaling.rope_type",
            "raw rope_scaling twin under the text wrapper",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's scaling-type line (wrapper, raw legacy "
            "spelling)", "root.text_encoder", "text_config.rope_scaling.type",
            "raw rope_scaling twin under the text wrapper",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's theta line (wrapper, bare sibling)",
            "root.text_encoder", "text_config.rope_theta",
            "top-level theta beside a scaling dict, under the text wrapper",
            "U8", "fact_registered:rope_theta"),
    _config("the rope card's partial-rotary line (redundant-equal rival "
            "occurrence)", "root", "rope_parameters.partial_rotary_factor",
            "the supplying occurrence is consumed; a witness declaring BOTH "
            "spellings equal leaves this rival as a noted inspection",
            "U8", "fact_registered:partial_rotary"),
    # ---- vision residue: reads whose consuming form awaits the named
    # ---- vision facts (per-occurrence, exact paths) ----------------------- #
    _config("the vision tower position table (raw spelling probe)",
            "root.text_encoder.vision", "rope_scaling",
            "bare-theta towers signal rope without a table; the table "
            "spelling probe stays an inspection on them",
            "U9", "fact_registered:vision_position"),
    _config("the vision tower depth (host-level spelling)",
            "root.text_encoder.vision", "num_hidden_layers",
            "host-level depth read beside the vision_config chain",
            "U9", "fact_registered:vision_tower_geometry"),
    _config("the cross-attention schedule derivation (text depth)",
            "root.text_encoder.vision", "text_config.num_hidden_layers",
            "freq x text-depth authors the schedule only on cross-attn "
            "models; plain VL models keep an inspected read",
            "U9", "fact_registered:cross_attention_schedule"),
    _config("the cross-attention schedule derivation (text depth)",
            "root.vision", "text_config.num_hidden_layers",
            "freq x text-depth authors the schedule only on cross-attn "
            "models; plain VL models keep an inspected read",
            "U9", "fact_registered:cross_attention_schedule"),
    _config("the patch-embedding input channels (losing rival spelling)",
            "root.text_encoder.vision", "vision_config.in_channels",
            "the winning spelling is consumed; a config declaring the rival "
            "too leaves it as an input-stage inspection",
            "U9", "fact_registered:vision_patch_geometry"),
    _config("the patch-embedding input channels (losing rival spelling)",
            "root.vision", "vision_config.in_channels",
            "the winning spelling is consumed; a config declaring the rival "
            "too leaves it as an input-stage inspection",
            "U9", "fact_registered:vision_patch_geometry"),
    # Final vet follow-through: the ROOT-level is_encoder_decoder is CONSUMED
    # (decoder.attention.mask); the class-stamped copies inside sub-configs
    # (transformers serializes is_encoder_decoder=False into every
    # sub-config) are unread boilerplate awaiting CLASSIFICATION — exact
    # per-owner pending rows, never a return to any global ignore.
    _config("class-stamped decoderness boilerplate on the text sub-config",
            "root", "text_config.is_encoder_decoder",
            "class-serialized copy the parse never reads (the root-level "
            "occurrence is the consumed decoderness read)",
            "U8", "classified:text_config.is_encoder_decoder"),
    _config("class-stamped decoderness boilerplate on the vision sub-config",
            "root.vision", "vision_config.is_encoder_decoder",
            "class-serialized copy the parse never reads (the root-level "
            "occurrence is the consumed decoderness read)",
            "U8", "classified:vision_config.is_encoder_decoder"),
)


# U5 consumer firewall: one exact, content-fingerprinted row per terminal
# consumer symbol/kind.  This is NOT a second allowlist.  Rows are ordinary
# StructuralDebt and participate in the same live writer/consumer/shrink gates.
# The fingerprint is over the complete normalized target set for that symbol;
# any added/removed path changes the live census key and fails in both
# directions.  The human-readable targets are always available from
# ``group_consumer_accesses()``.
_CONSUMER_DEBT_BASELINE = (
    ('model_unfolder/evidence/conformance.py', '<module>', 'conformance', 'backward_import', '58ce4a82144f8b64'),
    ('model_unfolder/evidence/conformance.py', '_as_mapping', 'conformance', 'raw_config', '06bfc54c63be216f'),
    ('model_unfolder/evidence/conformance.py', '_branch_inactive', 'conformance', 'raw_config', 'd0391b30cd356cd9'),
    ('model_unfolder/evidence/conformance.py', '_check_audio_facts', 'conformance', 'raw_extras', '2dc8a1f8de49ea25'),
    ('model_unfolder/evidence/conformance.py', '_check_component_storage_facts', 'conformance', 'raw_extras', '261bebf740a7e89d'),
    ('model_unfolder/evidence/conformance.py', '_check_fusion_facts', 'conformance', 'raw_extras', '007defd60449a154'),
    ('model_unfolder/evidence/conformance.py', '_check_projector_facts', 'conformance', 'raw_extras', 'bfb96c435232e9b2'),
    ('model_unfolder/evidence/conformance.py', '_check_storage_facts', 'conformance', 'backward_import', 'b5c1d62f90035b94'),
    ('model_unfolder/evidence/conformance.py', '_check_storage_facts', 'conformance', 'raw_extras', '8309e956d0a3abbf'),
    ('model_unfolder/evidence/conformance.py', '_check_vision_facts', 'conformance', 'raw_extras', 'abeededaa3def2a3'),
    ('model_unfolder/evidence/conformance.py', '_config_field_value', 'conformance', 'raw_config', 'f9c05d8e8878f612'),
    ('model_unfolder/evidence/conformance.py', '_constructor_envs', 'conformance', 'source_reopen', 'a821dd49ada1014e'),
    ('model_unfolder/evidence/conformance.py', '_drawn_fusion_routes', 'conformance', 'raw_extras', '342c661d01d15158'),
    ('model_unfolder/evidence/conformance.py', '_drawn_position_kinds', 'conformance', 'raw_extras', '2e884d5ab0b3db2d'),
    ('model_unfolder/evidence/conformance.py', '_family', 'conformance', 'raw_config', '6aed1a33205d116b'),
    ('model_unfolder/evidence/conformance.py', '_imported_model_files', 'conformance', 'source_reopen', '7fef0151ecf6fcd9'),
    ('model_unfolder/evidence/conformance.py', '_init_helper_block_classes', 'conformance', 'source_reopen', 'a821dd49ada1014e'),
    ('model_unfolder/evidence/conformance.py', '_is_block_class', 'conformance', 'backward_import', 'd0e50b755ed393d8'),
    ('model_unfolder/evidence/conformance.py', '_op_is_dormant', 'conformance', 'raw_config', '6aff71a545173320'),
    ('model_unfolder/evidence/conformance.py', '_reachable_forward_ops', 'conformance', 'backward_import', 'd3dd4554c4cd31a7'),
    ('model_unfolder/evidence/conformance.py', '_selected_init_refs', 'conformance', 'source_reopen', 'a821dd49ada1014e'),
    ('model_unfolder/evidence/conformance.py', 'check_fact_conformance', 'conformance', 'backward_import', 'd0e50b755ed393d8'),
    ('model_unfolder/evidence/conformance.py', 'check_fact_conformance', 'conformance', 'raw_config', 'e9e2a14f8eec059c'),
    ('model_unfolder/evidence/conformance.py', 'check_fact_conformance', 'conformance', 'source_reopen', 'ffcbd4ee3efa2a75'),
    ('model_unfolder/evidence/conformance.py', 'check_model_conformance', 'conformance', 'source_reopen', 'ffcbd4ee3efa2a75'),
    ('model_unfolder/evidence/conformance.py', 'check_nested_conformance', 'conformance', 'source_reopen', 'ffcbd4ee3efa2a75'),
    ('model_unfolder/evidence/conformance.py', 'check_wiring_conformance', 'conformance', 'raw_config', 'd0391b30cd356cd9'),
    ('model_unfolder/evidence/conformance.py', 'check_wiring_conformance', 'conformance', 'source_reopen', 'ffcbd4ee3efa2a75'),
    ('model_unfolder/evidence/conformance.py', 'diff_conformance', 'conformance', 'raw_config', 'b03beb21e8e55c5b'),
    ('model_unfolder/expanded/__init__.py', 'build_expanded', 'json', 'raw_extras', '6f5f5a85d05778f7'),
    ('model_unfolder/expanded/attention.py', 'build_attention', 'json', 'truthy_cleanup', 'ba941c7678d03b43'),
    ('model_unfolder/expanded/block_graph.py', '_block_node', 'json', 'truthy_cleanup', '10910087de24bfca'),
    ('model_unfolder/expanded/ffn.py', 'build_ffn', 'json', 'truthy_cleanup', '2355e1a94c38a944'),
    ('model_unfolder/expanded/loop.py', 'build_sampling_loop', 'json', 'raw_extras', '6857ba8217a73268'),
    ('model_unfolder/expanded/loop.py', 'build_sampling_loop', 'json', 'truthy_cleanup', '66f649a8fc8501f7'),
    ('model_unfolder/expanded/modalities.py', 'build_modalities', 'json', 'raw_extras', 'ff158202af354f0a'),
    ('model_unfolder/expanded/modalities.py', '_normalise_fusion', 'json', 'raw_extras', 'dc4c7690e4e88477'),
    ('model_unfolder/expanded/pathways.py', 'build_external_pathways', 'json', 'raw_extras', '8b89b379fd823424'),
    ('model_unfolder/expanded/sections.py', '_diffusion_io', 'json', 'raw_extras', '720bc5b7888319bc'),
    ('model_unfolder/expanded/sections.py', '_is_diffusion', 'json', 'raw_extras', '538efc70e7a2bf65'),
    ('model_unfolder/expanded/sections.py', 'build_dimensions', 'json', 'raw_extras', 'ab038e67ecaac4d6'),
    ('model_unfolder/expanded/sections.py', 'build_io', 'json', 'raw_extras', '5a84fc171da4bbf5'),
    ('model_unfolder/params.py', '_attn_params', 'params', 'spec_default', 'afaeb4c00bd9dee4'),
    ('model_unfolder/params.py', '_ffn_params', 'params', 'spec_default', '949c0c547ad432a2'),
    ('model_unfolder/renderers/html/block_views/declared_ops.py', 'build_declared_ops_view', 'renderer', 'raw_extras', 'ba8f51d3eafee8db'),
    ('model_unfolder/renderers/html/block_views/modality_views/common.py', 'audio_input', 'renderer', 'raw_extras', '680261edc5f7acee'),
    ('model_unfolder/renderers/html/block_views/modality_views/common.py', 'fusion_spec', 'renderer', 'raw_extras', 'dad11fdde8a702b6'),
    ('model_unfolder/renderers/html/block_views/modality_views/common.py', 'video_input', 'renderer', 'raw_extras', '658b78f2c8b074a7'),
    ('model_unfolder/renderers/html/block_views/modality_views/common.py', 'vision_input', 'renderer', 'raw_extras', 'aec51c4910a259ec'),
    ('model_unfolder/renderers/html/block_views/modality_views/conditioning.py', 'conditioning_input', 'renderer', 'raw_extras', '3fbbeba7773755e3'),
    ('model_unfolder/renderers/html/block_views/modality_views/video.py', 'build_video_path_view', 'renderer', 'raw_extras', '9f0b39c9d9150f2d'),
    ('model_unfolder/renderers/html/block_views/registry.py', 'render_view', 'renderer', 'raw_extras', 'f6568ea08cd6042b'),
    ('model_unfolder/renderers/html/block_views/unet.py', '_text_source_label', 'renderer', 'raw_extras', '94cf4b192dfe8dc8'),
    ('model_unfolder/renderers/html/block_views/unet.py', '_draw_text_conditioning', 'renderer', 'raw_extras', '4271fc2560fe3393'),
    ('model_unfolder/renderers/html/block_views/unet.py', '_stage_title', 'renderer', 'raw_extras', '7f1d4831e31adb5b'),
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_view', 'renderer', 'raw_extras', 'f337a110eca6bd73'),
    ('model_unfolder/renderers/html/document.py', '_render_fragment_body', 'renderer', 'raw_extras', '4e0f5a0ff4503742'),
    ('model_unfolder/renderers/html/document.py', 'render_fragment', 'renderer', 'raw_extras', '4592b54801fb2c20'),
    ('model_unfolder/renderers/html/evidence.py', '_code_evidence_section', 'renderer', 'raw_extras', 'ce0f5f767cc3d150'),
    ('model_unfolder/renderers/html/fact_projection.py', 'fact_provenance', 'renderer', 'raw_extras', 'a9089eb1467f73d3'),
    ('model_unfolder/renderers/html/metadata.py', '_arch_badges', 'renderer', 'raw_extras', '3b120c902b366719'),
    ('model_unfolder/renderers/html/metadata.py', '_block_lookup', 'renderer', 'raw_extras', '2b00ebf355b64033'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_audio_cell_cards', 'renderer', 'backward_import', '962109a2344e67b0'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_conditioning_children', 'renderer', 'backward_import', '3dfe0c4d151975ba'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_modality_badges', 'renderer', 'raw_extras', '72aee0d2fbeff24f'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_fusion_children', 'renderer', 'raw_extras', 'b5786f19ba04bd6a'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_fusion_description', 'renderer', 'raw_extras', '3649e8c389d841d0'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_is_conditioning_fusion', 'renderer', 'raw_extras', '944dba77202a5d65'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_multimodal_block_lookup', 'renderer', 'raw_extras', '09b28173badb0c5f'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_unified_fusion_children', 'renderer', 'raw_extras', '6d2771bda17006b0'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_vision_cell_cards', 'renderer', 'backward_import', '962109a2344e67b0'),
    ('model_unfolder/renderers/html/sections.py', '_diffusion_stats', 'renderer', 'raw_extras', 'da27bb4cf97df230'),
    ('model_unfolder/renderers/html/sections.py', '_stats_banner', 'renderer', 'raw_extras', '538efc70e7a2bf65'),
    ('model_unfolder/renderers/html/views.py', '_build_architecture_view', 'renderer', 'raw_extras', 'c2f7729c70d3b7ef'),
    ('model_unfolder/renderers/html/views.py', '_draw_mtp_head', 'renderer', 'raw_extras', '9254d402d9451cd0'),
    ('model_unfolder/renderers/html/views.py', '_is_diffusion_architecture', 'renderer', 'raw_extras', '538efc70e7a2bf65'),
    ('model_unfolder/renderers/html/views_diffusion.py', '_build_block_diffusion_loop_cards', 'renderer', 'raw_extras', '261bebf740a7e89d'),
    ('model_unfolder/renderers/html/views_diffusion.py', '_build_block_diffusion_view', 'renderer', 'raw_extras', '1c3216f252696602'),
    ('model_unfolder/renderers/html/views_diffusion.py', '_build_loop_cards', 'renderer', 'raw_extras', '1daab3bea1a4e581'),
    ('model_unfolder/renderers/html/views_diffusion.py', '_build_loop_descendant_levels', 'renderer', 'raw_extras', '261bebf740a7e89d'),
    ('model_unfolder/renderers/html/views_diffusion.py', '_build_loop_view', 'renderer', 'raw_extras', 'b6d427b43e3e45aa'),
    ('model_unfolder/renderers/html/views_diffusion.py', 'render_diffusion_fragment', 'renderer', 'raw_extras', '2a9f26e37a501e39'),
)

_CONSUMER_REASONS = {
    "backward_import": "terminal consumer still imports an upstream reader/vocabulary",
    "raw_config": "terminal consumer still reinterprets raw config instead of an owner-bound fact",
    "raw_extras": "terminal consumer still projects transitional raw extras",
    "semantic_bucket": "JSON trace still matches evidence by global semantic bucket",
    "source_reopen": "conformance still reopens/reparses source instead of receiving ProgramIndex",
    "spec_default": "parameter formula still selects a conventional branch from an unresolved spec",
    "truthy_cleanup": "JSON truthiness cleanup still conflates explicit false/empty with absence",
}


def _consumer_debt(module: str, symbol: str, consumer: str,
                   kind: str, fingerprint: str) -> StructuralDebt:
    target = f"{consumer}.{kind}.{fingerprint}"
    return StructuralDebt(
        owner=f"consumer.{consumer}",
        source_occurrence=f"normalized-target-set:{fingerprint}",
        writer_module=module,
        writer_symbol=symbol,
        sink_kind="consumer_read",
        structural_target=target,
        reason=_CONSUMER_REASONS[kind],
        last_consumer=f"{module}::{symbol}",
        migration_unit="U14",
        deletion_condition=(
            f"writer_gone:{module}::{symbol}:consumer_read:{target}"
        ),
        census_target=target,
    )


STRUCTURAL_DEBT += tuple(_consumer_debt(*spec)
                         for spec in _CONSUMER_DEBT_BASELINE)


# ---- derived join surfaces (the parser/sable joins the old registers fed) -- #

def _is_classification(row: StructuralDebt) -> bool:
    return (row.sink_kind == "config_read"
            and row.deletion_condition.startswith("classified:"))


def pending_projection_paths(rows=None) -> frozenset:
    """(owner, exact dotted path) for config occurrences awaiting a
    PROJECTION — the parser's exact-only excusal join (COR-2)."""
    rows = STRUCTURAL_DEBT if rows is None else rows
    return frozenset((r.owner, r.source_occurrence) for r in rows
                     if r.sink_kind == "config_read"
                     and not _is_classification(r))


def pending_classification_paths(rows=None) -> frozenset:
    """(owner, exact dotted path) for occurrences awaiting CLASSIFICATION
    (explicit-null declarations) — parser excusal + claims-audit join."""
    rows = STRUCTURAL_DEBT if rows is None else rows
    return frozenset((r.owner, r.source_occurrence) for r in rows
                     if _is_classification(r))


# fabrication_debt_keys DELETED (Soumil's final vet): config_read debt
# rows are INPUT classifications and may never authorize a drawn output
# receipt — the reverse-fabrication net accepts only typed facts and
# exact migration-claim targets.
def drawn_unledgered_pairs(rows=None) -> frozenset:
    """(owner, leaf) pairs lawfully drawn WITHOUT a registered fact —
    Soumil's final vet: drawn debt keeps its OWNER identity, so one owner's
    unledgered leaf never excuses another owner drawing the same name.
    Owners are normalized to the registry's pattern space (the parse-root
    ``root.`` prefix stripped) — the JOIN key every drawn gate uses."""
    from .registry import REGISTRY
    rows = STRUCTURAL_DEBT if rows is None else rows
    return frozenset(
        ((r.owner[5:] if r.owner.startswith("root.") else r.owner),
         r.structural_target)
        for r in rows
        if r.sink_kind == "drawn_leaf"
        and r.structural_target not in REGISTRY)


def drawn_leaf_is_lawful(owner: str, leaf: str, rows=None) -> bool:
    """THE reverse-fabrication join for one drawn (owner, leaf): lawful only
    when a registered FactDefinition covers THIS owner, or an unledgered
    drawn_leaf debt row carries THIS owner — a sibling owner's registration
    or debt authorizes nothing (never a bare leaf-name join)."""
    from .registry import REGISTRY, owner_matches_pattern
    defn = REGISTRY.get(leaf)
    if defn is not None and any(
            owner == pattern or owner_matches_pattern(owner, pattern)
            for pattern in defn.owner_patterns):
        return True
    return (owner, leaf) in drawn_unledgered_pairs(rows)


def drawn_unledgered_names(rows=None) -> frozenset:
    """DISPLAY/COMPATIBILITY view only (leaf names, owners collapsed).
    NEVER a gate input — every gate joins :func:`drawn_leaf_is_lawful` /
    :func:`drawn_unledgered_pairs` (Soumil's final vet, round 2: the
    name-collapsed join let one owner's debt authorize a sibling owner's
    drawing of the same leaf name)."""
    return frozenset(name for _, name in drawn_unledgered_pairs(rows))


def debt_keys(rows: tuple[StructuralDebt, ...] | None = None) -> frozenset:
    """(sink_kind, structural_target) pairs — the join surface parser/sable
    already use for excusal and fabrication debt."""
    rows = STRUCTURAL_DEBT if rows is None else rows
    return frozenset((r.sink_kind, r.structural_target) for r in rows)


def debt_targets(sink: str, rows=None) -> frozenset:
    rows = STRUCTURAL_DEBT if rows is None else rows
    return frozenset(r.structural_target for r in rows if r.sink_kind == sink)


# --------------------------------------------------------------------------- #
# Live-world lookups (injectable for hermetic poisons)
# --------------------------------------------------------------------------- #
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=256)
def _module_symbols(module: str) -> frozenset[str] | None:
    """Bare compatibility names plus exact qualified symbol paths."""
    path = _PKG_ROOT / module
    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return None
    names = {"<module>"}

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        next_scope = scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)  # compatibility for pre-U5 debt rows
            next_scope = scope + (node.name,)
            names.add(".".join(next_scope))
        for child in ast.iter_child_nodes(node):
            visit(child, next_scope)

    visit(tree, ())
    return frozenset(names)


def _symbol_exists(module: str, symbol: str) -> bool:
    symbols = _module_symbols(module)
    return symbols is not None and symbol in symbols


def _live_census_keys():
    from .structural_writes import scan_structural_write_multiset
    keys = {(k.module, k.enclosing_symbol, k.sink_kind, k.normalized_target)
            for k in scan_structural_write_multiset()}
    from .consumer_firewall import group_consumer_accesses
    keys.update(group.key for group in group_consumer_accesses())
    return keys


def _fact_definition(fact_key: str):
    from .registry import fact_definition
    return fact_definition(fact_key)


def _classification_table() -> frozenset:
    """The typed config-classification table (lands with U11) as OWNER-EXACT
    ``(owner, exact path)`` pairs; absent → empty, so every ``classified:``
    condition mechanically evaluates False until it exists.  U11 must publish
    the table in that pair shape — a bare-path table cannot satisfy any row
    (Soumil's final vet, round 2: no deletion condition is bare-path)."""
    try:
        from . import registry
        table = getattr(registry, "CONFIG_CLASSIFICATION", None)
    except ImportError:
        table = None
    if table is None:
        return frozenset()
    return frozenset(table)


def _defn_covers_owner(defn, owner: str) -> bool:
    """Soumil's final vet: a fact-side condition is bound to the ROW'S OWNER —
    a fact registered for a different owner satisfies nothing here."""
    from .registry import owner_matches_pattern
    return any(owner == pattern or owner_matches_pattern(owner, pattern)
               for pattern in defn.owner_patterns)


def deletion_condition_met(row: StructuralDebt, *,
                           census_keys=None) -> bool:
    """Evaluate the row's deletion predicate against the live world.

    The fact-side verbs (fact_registered / fact_routed / status_retired /
    unknown_policy_retired) additionally require the registered definition's
    owner_patterns to COVER this row's owner — registration under a foreign
    owner never retires another owner's debt."""
    verb, args = _parse_condition(row.deletion_condition)
    if verb == "fact_registered":
        defn = _fact_definition(args[0])
        return defn is not None and _defn_covers_owner(defn, row.owner)
    if verb == "fact_routed":
        defn = _fact_definition(args[0])
        return (defn is not None and bool(defn.projection_routes)
                and _defn_covers_owner(defn, row.owner))
    if verb == "status_retired":
        defn = _fact_definition(args[0])
        return (defn is not None and args[1] not in defn.allowed_statuses
                and _defn_covers_owner(defn, row.owner))
    if verb == "unknown_policy_retired":
        defn = _fact_definition(args[0])
        return (defn is not None
                and defn.unknown_policy != "legacy_convention"
                and _defn_covers_owner(defn, row.owner))
    if verb == "no_writer":
        keys = _live_census_keys() if census_keys is None else census_keys
        sink, target = args
        return not any(k[2] == sink and k[3] == target for k in keys)
    if verb == "writer_gone":
        keys = _live_census_keys() if census_keys is None else census_keys
        return tuple(args) not in keys
    if verb == "classified":
        return (row.owner, args[0]) in _classification_table()
    if verb == "symbol_deleted":
        return not _symbol_exists(args[0], args[1])
    raise AssertionError(verb)  # unreachable: _parse_condition is closed


# --------------------------------------------------------------------------- #
# The blocking gates (§R6 "Complete when")
# --------------------------------------------------------------------------- #
def duplicate_debt_rows(rows=None) -> list[tuple[str, str, str, str]]:
    """Two rows for one (sink, target, writer) — one exact row per debt,
    never a family; duplicates mean an excuse is being laundered twice.
    Distinct WRITERS of one target are distinct debts and lawful."""
    rows = STRUCTURAL_DEBT if rows is None else rows
    seen: set[tuple[str, str, str, str]] = set()
    dupes: list[tuple[str, str, str, str]] = []
    for r in rows:
        if r.key in seen:
            dupes.append(r.key)
        seen.add(r.key)
    return dupes


def unbacked_debt_rows(rows=None, *, census_keys=None) -> list[StructuralDebt]:
    """Rows whose writer is DEAD (§R6: every debt row points to a live
    writer).  Census-sink rows must join the writer census on full identity;
    non-census rows (projection/config/status) must name a writer symbol that
    still exists in its module."""
    rows = STRUCTURAL_DEBT if rows is None else rows
    census_sinks = {"ledger", "spec", "spec_field", "extras", "opgraph",
                    "card", "params", "consumer_read"}
    keys = None
    dead: list[StructuralDebt] = []
    for r in rows:
        if r.sink_kind in census_sinks:
            if keys is None:
                keys = _live_census_keys() if census_keys is None \
                    else census_keys
            if r.writer_key not in keys:
                dead.append(r)
        elif not _symbol_exists(r.writer_module, r.writer_symbol):
            dead.append(r)
    return dead


def unconsumed_debt_rows(rows=None) -> list[StructuralDebt]:
    """Rows whose last_consumer no longer exists (§R6: ... and consumer).
    A row nobody reads is not debt, it is a fossil — delete it."""
    rows = STRUCTURAL_DEBT if rows is None else rows
    dead = []
    for r in rows:
        module, _, symbol = r.last_consumer.partition("::")
        if not _symbol_exists(module, symbol):
            dead.append(r)
    return dead


def satisfied_debt_rows(rows=None, *, census_keys=None) -> list[StructuralDebt]:
    """Rows whose deletion condition is ALREADY TRUE — the migration landed
    but the row survived.  Blocking: deleting/migrating a writer must shrink
    the register in the same commit (§R6)."""
    rows = STRUCTURAL_DEBT if rows is None else rows
    return [r for r in rows
            if deletion_condition_met(r, census_keys=census_keys)]


def unrowed_extras_writes(rows=None, *, census_keys=None) -> list[str]:
    """Growth gate, WRITER-EXACT (Soumil's final vet): every raw extras
    census WRITER KEY (module, symbol, sink, target) needs a debt row whose
    writer_key matches exactly — a row for one author of a target excuses
    neither a second author nor a nested leaf (§R6: family-wide excuses
    block U2; coverage joins on the same writer identity the baseline gate
    ratchets)."""
    from .structural_writes import _INFRA_EXTRAS
    rows = STRUCTURAL_DEBT if rows is None else rows
    keys = _live_census_keys() if census_keys is None else census_keys
    rowed = frozenset(r.writer_key for r in rows if r.sink_kind == "extras")
    missing = []
    for key in keys:
        module, symbol, sink, target = key
        if sink != "extras":
            continue
        if target.split(".", 1)[0] in _INFRA_EXTRAS:
            continue
        if key not in rowed:
            missing.append(f"{module}::{symbol} -> extras:{target}")
    return sorted(missing)


def unrowed_consumer_reads(rows=None, *, census_keys=None) -> list[str]:
    """Every terminal-consumer dependency/read group needs one exact row.

    The group's target contains a fingerprint of its complete normalized path
    set.  Adding or removing a read therefore strands the old row *and* creates
    an unrowed new group; an exact-symbol quarantine cannot silently grow.
    """
    rows = STRUCTURAL_DEBT if rows is None else rows
    keys = _live_census_keys() if census_keys is None else census_keys
    rowed = frozenset(r.writer_key for r in rows
                      if r.sink_kind == "consumer_read")
    return sorted(
        f"{module}::{symbol} -> {target}"
        for module, symbol, sink, target in keys
        if sink == "consumer_read"
        and (module, symbol, sink, target) not in rowed
    )


def debt_problems(rows=None, *, census_keys=None) -> list[str]:
    """Every §R6 gate in one blocking report (empty == lawful)."""
    rows = STRUCTURAL_DEBT if rows is None else rows
    # One immutable census per audit.  Recomputing it inside every
    # writer_gone condition made the exact U5 consumer rows multiply a full
    # AST scan by the number of rows and could turn a blocking gate into a
    # minutes-long accidental denial of service.
    keys = _live_census_keys() if census_keys is None else census_keys
    problems = [f"duplicate debt row {k}" for k in duplicate_debt_rows(rows)]
    problems += [f"dead writer: {r.writer_module}::{r.writer_symbol} for "
                 f"{r.sink_kind}:{r.structural_target}"
                 for r in unbacked_debt_rows(rows, census_keys=keys)]
    problems += [f"dead consumer: {r.last_consumer} for "
                 f"{r.sink_kind}:{r.structural_target}"
                 for r in unconsumed_debt_rows(rows)]
    problems += [f"deletion condition already met "
                 f"({r.deletion_condition}) — delete the row: "
                 f"{r.sink_kind}:{r.structural_target}"
                 for r in satisfied_debt_rows(rows, census_keys=keys)]
    problems += [f"raw extras write without a writer-exact debt row: {t}"
                 for t in unrowed_extras_writes(rows, census_keys=keys)]
    problems += [f"terminal consumer read without an exact debt row: {t}"
                 for t in unrowed_consumer_reads(rows, census_keys=keys)]
    return problems
