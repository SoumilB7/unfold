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
    classified:<config_path>         the config-classification table has it
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
    "drawn_leaf", "config_read",
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
    def key(self) -> tuple[str, str, str, str]:
        """Row identity: one row per (sink, target, WRITER) — a second author
        of one target is a second debt, never hidden under the first row."""
        return (self.sink_kind, self.structural_target,
                self.writer_module, self.writer_symbol)

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
    _extras("qk_norm", "root.decoder.attention",
            "model-level QK-norm bool as raw extras (drawn QK-norm reads "
            "AttentionSpec.qk_norm, never this key)",
            "U6", "fact_routed:qk_norm",
            occurrence="text_cfg use_qk_norm | qk_norm | qk_layernorm"),
    _extras("dual_kv", "root.decoder.attention",
            "dual global/sliding KV schedule as raw extras",
            "U6", "fact_registered:kv_schedule",
            occurrence="text_cfg num_global_key_value_heads | "
                       "global_head_dim"),
    _extras("dual_kv.global", "root.decoder.attention",
            "global-lane KV geometry as a raw extras leaf", "U6",
            "fact_registered:kv_schedule"),
    _extras("dual_kv.sliding", "root.decoder.attention",
            "sliding-lane KV geometry as a raw extras leaf", "U6",
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
            "U6", "fact_registered:kv_shared_layers",
            occurrence="text_cfg num_kv_shared_layers"),
    _extras("parallel_residual", "root.decoder.layer",
            "parallel-residual topology flag as raw extras (drawn topology "
            "comes from parallel_decoder_layer blocks)",
            "U7", "fact_registered:residual_topology",
            occurrence="text_cfg use_parallel_residual + "
                       "everchanging layer_topology.yaml"),
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
    _extras("softcap", "root.decoder.attention",
            "attention/final softcap descriptor as raw extras (drawn softcap "
            "is AttentionSpec.logit_softcap -> Op:attn_softcap)",
            "U6", "fact_routed:logit_softcap",
            occurrence="text_cfg attn_logit_softcapping | "
                       "final_logit_softcapping | query_pre_attn_scalar"),
    _extras("attention", "root.decoder.attention",
            "clip-QKV info-only attention annotation as raw extras",
            "U6", "fact_registered:clip_qkv",
            occurrence="text_cfg clip_qkv | attn_cfg clip_qkv"),
    _extras("attention_k_eq_v + use_double_wide_mlp pass-through flags",
            "root",
            "variable-keyed pass-through flag loop (extras[flag] = val)",
            "U6", "classified:attention_k_eq_v",
            occurrence="text_cfg attention_k_eq_v | use_double_wide_mlp",
            census="<dynamic>"),
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
                       "decoder_codebook_streams_from_files",
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
           "per-head Q/K normalisation drawn from spec.qk_norm",
           "U6", "fact_routed:qk_norm",
           occurrence="AttentionSpec.qk_norm",
           module=_AT, symbol="_sdpa_detailed_child_blocks",
           consumer="model_unfolder/renderers/html/fact_projection.py::"
                    "attention_facts"),
    _drawn("q_norm", "root.decoder.attention",
           "separate Q-norm variant drawn from spec (folds into qk_norm "
           "when the registered fact declares per-lane leaves)",
           "U6", "fact_registered:q_norm",
           occurrence="attention_detail derived leaf bool(qk_norm)",
           module=_AT, symbol="attention_detail",
           consumer="model_unfolder/renderers/html/fact_projection.py::"
                    "attention_facts"),
    _drawn("k_norm", "root.decoder.attention",
           "separate K-norm variant drawn from spec (folds into qk_norm "
           "when the registered fact declares per-lane leaves)",
           "U6", "fact_registered:k_norm",
           occurrence="attention_detail derived leaf bool(qk_norm)",
           module=_AT, symbol="attention_detail",
           consumer="model_unfolder/renderers/html/fact_projection.py::"
                    "attention_facts"),
    _drawn("logit_softcap", "root.decoder.attention",
           "logit soft-cap op drawn from spec.logit_softcap",
           "U6", "fact_routed:logit_softcap",
           occurrence="AttentionSpec.logit_softcap",
           module=_OG, symbol="_sdpa_core_ops",
           consumer="model_unfolder/renderers/html/fact_projection.py::"
                    "attention_facts"),
    # ---- legacy_convention drawings (registered facts whose UNKNOWN state
    # ---- is drawn as a fabricated convention) ----------------------------- #
    _drawn("norm_placement", "root.decoder.layer",
           "reader-abstain still draws pre-norm + banner; honest-unknown "
           "cell replaces the convention",
           "U7", "unknown_policy_retired:norm_placement",
           occurrence="REGISTRY['norm_placement'] "
                      "unknown_policy='legacy_convention'",
           module=_TP, symbol="parse",
           consumer="model_unfolder/renderers/html/views.py::"
                    "_build_architecture_view"),
    _drawn("scores_scale", "root.decoder.attention",
           "silent input drawn as sqrt(d)-scaled scores denominator",
           "U6", "unknown_policy_retired:scores_scale",
           occurrence="REGISTRY['scores_scale'] "
                      "unknown_policy='legacy_convention'",
           module=_OG, symbol="_sdpa_core_ops",
           consumer="model_unfolder/renderers/html/fact_projection.py::"
                    "attention_facts"),
    _drawn("projection_mode", "root.decoder.attention",
           "absent mode drawn as solid split projections; asserted rows "
           "are pinned census debt",
           "U6", "status_retired:projection_mode:legacy_asserted",
           occurrence="REGISTRY['projection_mode'] statuses "
                      "asserted/legacy_asserted",
           module=_OG, symbol="_sdpa_region",
           consumer="model_unfolder/renderers/html/fact_projection.py::"
                    "attention_facts"),
    _drawn("attention_kind", "layers[i].attention",
           "diffusion per-layer joint-attention kind asserted with None "
           "value (json-only; 543 records / 13 fixtures)",
           "U10", "status_retired:attention_kind:legacy_asserted",
           occurrence="REGISTRY['attention_kind'] value_types NoneType",
           module=_DP, symbol="parse", consumer=_JSON_DOC),
    _drawn("ffn_storage", "layers[i].ffn",
           "per-layer storage tag asserted with None value (json-only; "
           "94 records / 2 fixtures)",
           "U7", "status_retired:ffn_storage:legacy_asserted",
           occurrence="REGISTRY['ffn_storage'] value_types NoneType",
           module=_TP, symbol="parse", consumer=_JSON_DOC),
    # ---- config occurrences awaiting their consumer (former
    # ---- PENDING_PROJECTION_DEBT; owner + EXACT dotted path) -------------- #
    _config("the conditioning card on the denoiser view",
            "root.denoiser", "max_sequence_length",
            "max text-token sequence the denoiser conditions on (Mochi) — "
            "a declared conditioning limit",
            "U10", "fact_registered:denoiser_conditioning_limit"),
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
)


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


def fabrication_debt_keys(rows=None) -> set:
    """(owner, canonical leaf) for every config_read row — the sable
    fabrication-net debt join (leaf of the exact path)."""
    rows = STRUCTURAL_DEBT if rows is None else rows
    return {(r.owner, r.source_occurrence.rsplit(".", 1)[-1]) for r in rows
            if r.sink_kind == "config_read"}


def drawn_unledgered_names(rows=None) -> frozenset:
    """Leaf names lawfully drawn WITHOUT a registered fact — drawn_leaf rows
    whose fact is not (yet) in the REGISTRY.  A drawn leaf that is neither
    registered nor here is fabrication (exclusive-or preserved: a registered
    fact's convention row never re-excuses its leaf)."""
    from .registry import REGISTRY
    rows = STRUCTURAL_DEBT if rows is None else rows
    return frozenset(r.structural_target for r in rows
                     if r.sink_kind == "drawn_leaf"
                     and r.structural_target not in REGISTRY)


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
    """Top-level and nested def/class names in a production module, or None
    if the module file does not exist."""
    path = _PKG_ROOT / module
    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return None
    names = {"<module>"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
    return frozenset(names)


def _symbol_exists(module: str, symbol: str) -> bool:
    symbols = _module_symbols(module)
    return symbols is not None and symbol in symbols


def _live_census_keys():
    from .structural_writes import scan_structural_write_multiset
    return {(k.module, k.enclosing_symbol, k.sink_kind, k.normalized_target)
            for k in scan_structural_write_multiset()}


def _fact_definition(fact_key: str):
    from .registry import fact_definition
    return fact_definition(fact_key)


def _classification_table() -> frozenset[str]:
    """The typed config-classification table (lands with U11); absent → empty,
    so every ``classified:`` condition mechanically evaluates False until it
    exists."""
    try:
        from . import registry
        table = getattr(registry, "CONFIG_CLASSIFICATION", None)
    except ImportError:
        table = None
    if table is None:
        return frozenset()
    return frozenset(table)


def deletion_condition_met(row: StructuralDebt, *,
                           census_keys=None) -> bool:
    """Evaluate the row's deletion predicate against the live world."""
    verb, args = _parse_condition(row.deletion_condition)
    if verb == "fact_registered":
        return _fact_definition(args[0]) is not None
    if verb == "fact_routed":
        defn = _fact_definition(args[0])
        return defn is not None and bool(defn.projection_routes)
    if verb == "status_retired":
        defn = _fact_definition(args[0])
        return defn is not None and args[1] not in defn.allowed_statuses
    if verb == "unknown_policy_retired":
        defn = _fact_definition(args[0])
        return defn is not None and defn.unknown_policy != "legacy_convention"
    if verb == "no_writer":
        keys = _live_census_keys() if census_keys is None else census_keys
        sink, target = args
        return not any(k[2] == sink and k[3] == target for k in keys)
    if verb == "writer_gone":
        keys = _live_census_keys() if census_keys is None else census_keys
        return tuple(args) not in keys
    if verb == "classified":
        return args[0] in _classification_table()
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
                    "card", "params"}
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


def unrowed_extras_writes(rows=None, *, surface=None) -> list[str]:
    """Growth gate: every raw top-level-or-nested extras write in the pinned
    census surface needs its own EXACT debt row — a top-level key excuses
    nothing below it (§R6: family-wide excuses block U2)."""
    from .structural_writes import STRUCTURAL_SURFACE, _INFRA_EXTRAS
    rows = STRUCTURAL_DEBT if rows is None else rows
    surface = STRUCTURAL_SURFACE if surface is None else surface
    rowed = debt_targets("extras", rows) | frozenset(
        r.census_target for r in rows
        if r.sink_kind == "extras" and r.census_target)
    missing = []
    for sink, target in surface:
        if sink != "extras":
            continue
        top = target.split(".", 1)[0]
        if top in _INFRA_EXTRAS:
            continue
        if target not in rowed:
            missing.append(target)
    return sorted(missing)


def debt_problems(rows=None, *, census_keys=None, surface=None) -> list[str]:
    """Every §R6 gate in one blocking report (empty == lawful)."""
    rows = STRUCTURAL_DEBT if rows is None else rows
    problems = [f"duplicate debt row {k}" for k in duplicate_debt_rows(rows)]
    problems += [f"dead writer: {r.writer_module}::{r.writer_symbol} for "
                 f"{r.sink_kind}:{r.structural_target}"
                 for r in unbacked_debt_rows(rows, census_keys=census_keys)]
    problems += [f"dead consumer: {r.last_consumer} for "
                 f"{r.sink_kind}:{r.structural_target}"
                 for r in unconsumed_debt_rows(rows)]
    problems += [f"deletion condition already met "
                 f"({r.deletion_condition}) — delete the row: "
                 f"{r.sink_kind}:{r.structural_target}"
                 for r in satisfied_debt_rows(rows, census_keys=census_keys)]
    problems += [f"raw extras write without an exact debt row: {t}"
                 for t in unrowed_extras_writes(rows, surface=surface)]
    return problems
