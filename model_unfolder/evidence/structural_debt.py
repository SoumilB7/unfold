"""U2-R6 — the ONE structural-debt register (§R6).

Replaces the four separate registers (``LEGACY_EXTRAS``,
``DRAWN_UNLEDGERED_DEBT``, ``PENDING_PROJECTION_DEBT``,
``PENDING_CONFIG_CLASSIFICATION``) with a single exact schema.  A debt row is
not an allowlist entry: it names the WRITER (module + symbol + sink + target),
the CONSUMER that still needs the raw form, the U3–U15 unit that migrates it,
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
# The current master plan runs through U15.  The original U2 register ended at
# U14; extending this closed vocabulary is required so U15's final semantic-
# config deletion work is named honestly instead of being mislabeled as U14.
MIGRATION_UNITS = frozenset(f"U{i}" for i in range(3, 16))

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
    migration_unit: str            # U3–U15 (H7/H8/UNASSIGNED are rejected)
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
                f"{self.migration_unit!r} is not U3–U15 — H-era units, "
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
# U3–U15 unit + checkable deletion condition).  Replaces LEGACY_EXTRAS,
# DRAWN_UNLEDGERED_DEBT, PENDING_PROJECTION_DEBT and
# PENDING_CONFIG_CLASSIFICATION (U2-R6; row content from the reviewed
# writer/consumer audit in z-docs/07-current-state/r6-debt-rows.md).
# --------------------------------------------------------------------------- #
_TP = "model_unfolder/adapters/transformer/parser.py"
_DP = "model_unfolder/adapters/diffusor/parser.py"
_TA = "model_unfolder/adapters/transformer/assembly.py"
_MB = ("model_unfolder/adapters/transformer/special_parts/modalities/"
       "builder.py")
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
    _extras("softcap", "model",
            "final vocabulary-logit softcap descriptor as raw extras; the "
            "attention cap and query-score operand have left this legacy lane",
            "U14", "fact_routed:final_logit_softcap",
            occurrence="text_cfg final_logit_softcapping"),
    _extras("block_diffusion", "root",
            "block-diffusion canvas descriptor as raw extras",
            "U14", "fact_registered:block_diffusion_canvas",
            occurrence="cfg canvas_length",
            consumer="model_unfolder/renderers/html/views_diffusion.py::"
                     "_build_block_diffusion_view"),
    _extras("block_diffusion.canvas_length", "root",
            "canvas length as a raw extras leaf", "U14",
            "fact_registered:block_diffusion_canvas",
            consumer="model_unfolder/renderers/html/views_diffusion.py::"
                     "_build_block_diffusion_view"),
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
    _extras("render", "root.denoiser",
            "U10-F3 source-projected render DTO; presentation remains a raw "
            "extras transport until the U14 renderer boundary is typed",
            "U14", "no_writer:extras:render",
            occurrence="U10 source projection + U11/U12/U13 handoffs",
            module=_DP, symbol="_parse_projected_denoiser",
            consumer="model_unfolder/renderers/html/metadata.py::_block_lookup"),
    _extras("modalities", "root",
            "multimodal tower descriptors merged into extras via the "
            "dynamic-keyed _merge_extras loop",
            "U14", f"writer_gone:{_TA}::_merge_extras:extras:<dynamic>",
            occurrence="vision/audio/video sub-configs + source-bundle "
                       "tower/projector/fusion evidence",
            module=_TA, symbol="_merge_extras", census="<dynamic>",
            consumer="model_unfolder/evidence/conformance.py::"
                     "_check_recursive_component_facts"),
    _extras("modalities.inputs", "root",
            "modality input-path payload authored by the multimodal builder",
            "U14", f"writer_gone:{_MB}::multimodal_extras:extras:<dynamic>",
            occurrence="vision/audio/video sub-configs",
            module=_MB, symbol="multimodal_extras", census="<dynamic>",
            consumer="model_unfolder/evidence/conformance.py::"
                     "_check_recursive_component_facts"),
    _extras("diffusion", "root",
            "UNet meta descriptor (channels/text-encoders/scheduler) as raw "
            "extras", "U11", "fact_registered:diffusion_meta",
            occurrence="unet config boc/in_channels/cross_attention_dim + "
                       "pipeline text_encoders/scheduler",
            module=_DP, symbol="_parse_unet_model",
            consumer="model_unfolder/renderers/html/sections.py::"
                     "_diffusion_stats"),
    _extras("unet", "root",
            "full UNet block-structure descriptor as raw extras",
            "U11", "fact_registered:unet_blocks",
            occurrence="UNet config geometry assembled by diffusor/unet.py",
            module=_DP, symbol="_parse_unet_model",
            consumer="model_unfolder/renderers/html/block_views/unet.py::"
                     "build_unet_view"),
    # U8 retired the former position_kind/qk_norm drawn-leaf debts: the exact
    # per-layer position_schedule and qk_norm_schedule facts now authorize the
    # canonical layer-map projection and are receipted by that real consumer.
    # ---- config occurrences awaiting their consumer (former
    # ---- PENDING_PROJECTION_DEBT; owner + EXACT dotted path) -------------- #
    _config("the conditioning card on the denoiser view",
            "root.denoiser", "max_sequence_length",
            "max text-token sequence the denoiser conditions on (Mochi) — "
            "a declared conditioning limit",
            "U15", "classified:max_sequence_length"),
    _config("the composite conditioning encoder's attention geometry",
            "root.conditioning", "text_encoder.num_heads",
            "U9 now proves the exact task-specific text-encoder occurrence "
            "and projects its code-bound head geometry through the typed "
            "component result.  The remaining debt is U14's FactLedger + "
            "projection-receipt cutover for the modality extras surface; "
            "until then this raw checkpoint operand remains visibly unread "
            "rather than being falsely marked consumed by the old parser",
            "U14", "fact_registered:conditioning_attention_geometry"),
    # U9 proves the exact task-specific conditioning owner and source
    # mechanisms.  These remaining checkpoint declarations are deliberately
    # not copied into raw modality extras until U14 gives them owner-qualified
    # facts and receipts.  Keep them visible as exact debt rather than either
    # pretending U9 consumed them or failing the recursive unread audit.
    _config("the conditioning encoder FFN width operand",
            "root.conditioning", "text_encoder.d_ff",
            "the exact conditioning FFN occurrence is source-qualified, but "
            "U14 must bind and receipt its affine width before projection",
            "U14", "fact_registered:conditioning_ffn_geometry"),
    _config("the conditioning encoder activation alias",
            "root.conditioning", "text_encoder.dense_act_fn",
            "the checkpoint activation alias awaits U14's exact winning-path "
            "fact and projection receipt",
            "U14", "fact_registered:conditioning_ffn_activation"),
    _config("the conditioning encoder FFN protocol declaration",
            "root.conditioning", "text_encoder.feed_forward_proj",
            "the source-qualified FFN protocol remains unprojected until U14 "
            "owns its config operand as a typed fact",
            "U14", "fact_registered:conditioning_ffn_protocol"),
    _config("the conditioning encoder gated-activation declaration",
            "root.conditioning", "text_encoder.is_gated_act",
            "the checkpoint gate declaration cannot author a mechanism; U14 "
            "must join it to the exact FFN occurrence and receipt the result",
            "U14", "fact_registered:conditioning_ffn_gating"),
    _config("the conditioning encoder repetition operand",
            "root.conditioning", "text_encoder.num_layers",
            "U9 proves the exact repeated stage; U14 must ledger and receipt "
            "the checkpoint count operand on the modality projection",
            "U14", "fact_registered:conditioning_tower_depth"),
    _config("the conditioning token-embedding vocabulary operand",
            "root.conditioning", "text_encoder.vocab_size",
            "the declared vocabulary size awaits an exact embedding-owner "
            "fact and U14 projection receipt",
            "U14", "fact_registered:conditioning_vocab_size"),
    _config("the denoiser FFN activation dispatch",
            "root.denoiser", "activation_fn",
            "the activation operand is visible, but only an exact denoiser "
            "constructor dispatch may project it as FFN mechanism evidence",
            "U15", "classified:activation_fn"),
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
    # Multimodal position coordinates are U9 work.  The text attention reader
    # can prove the Q/K rotation algebra, but it must not invent the origin of
    # cos/sin supplied by the wrapper's T/H/W coordinate pipeline.  Each exact
    # checkpoint occurrence stays visible here until that wrapper→phase→Q/K
    # chain is proved.  This is mechanism debt, never a family exemption.
    _config("the interleaved multimodal rotary application", "root",
            "rope_parameters.mrope_interleaved",
            "the checkpoint declaration is visible, but the modality-owned "
            "T/H/W coordinate split and its exact rotary application belong "
            "to U9, not the text-layer position schedule",
            "U9", "fact_registered:mrope_interleaving"),
    _config("the multimodal rotary coordinate partition", "root",
            "rope_parameters.mrope_section",
            "the exact checkpoint operand is visible, but only the U9 "
            "wrapper coordinate construction can prove how its T/H/W "
            "sections reach the text attention's supplied phase",
            "U9", "fact_registered:mrope_coordinates"),
    _config("the multimodal rotary frequency base", "root",
            "rope_parameters.rope_theta",
            "the text attention proves rotation but not the wrapper-owned "
            "phase producer that consumes this frequency base",
            "U9", "fact_registered:mrope_coordinates"),
    _config("the multimodal rotary initializer selector", "root",
            "rope_parameters.rope_type",
            "the selector cannot author a mechanism; U9 must bind the exact "
            "selected wrapper phase initializer",
            "U9", "fact_registered:mrope_coordinates"),
    _config("the multimodal rotary width operand", "root",
            "rope_parameters.partial_rotary_factor",
            "the operand changes wrapper-produced phase width, but its exact "
            "coordinate/phase chain is outside the U8 text-layer boundary",
            "U9", "fact_registered:mrope_coordinates"),
    _config("the duplicate multimodal rotary width declaration", "root",
            "partial_rotary_factor",
            "the root duplicate remains visible until U9 proves which exact "
            "wrapper normalization/initializer spelling is enacted",
            "U9", "fact_registered:mrope_coordinates"),
    _config("the multimodal rotary coordinate partition", "root",
            "text_config.rope_parameters.mrope_section",
            "the exact checkpoint operand is visible, but only the U9 "
            "wrapper/modality coordinate construction can prove how its "
            "T/H/W sections reach Q/K rotation",
            "U9", "fact_registered:mrope_coordinates"),
    _config("the multimodal rotary frequency base", "root",
            "text_config.rope_parameters.rope_theta",
            "the numeric operand cannot author rotation; U9 must bind it to "
            "the exact multimodal frequency initializer used by the wrapper",
            "U9", "fact_registered:mrope_coordinates"),
    _config("the multimodal rotary initializer selector", "root",
            "text_config.rope_parameters.rope_type",
            "selector syntax is not mechanism evidence; U9 must resolve the "
            "selected callable and its multimodal application",
            "U9", "fact_registered:mrope_coordinates"),
    _config("the legacy multimodal rotary initializer selector", "root",
            "text_config.rope_parameters.type",
            "the legacy selector spelling remains exact visible debt until "
            "U9 proves and classifies the selected multimodal callable",
            "U9", "fact_registered:mrope_coordinates"),
    _config("the denoiser projection-bias fact", "root.denoiser",
            "attention_bias",
            "declaration awaits exact denoiser projection-construction binding",
            "U15", "classified:attention_bias"),
    _config("the denoiser Q/K-normalization fact", "root.denoiser",
            "qk_norm",
            "declaration awaits exact denoiser Q/K norm execution binding",
            "U15", "classified:qk_norm"),
    _config("the denoiser rotary-application fact", "root.denoiser",
            "use_rotary_positional_embeddings",
            "declaration awaits exact denoiser positional call binding; "
            "it cannot independently author Q/K rotation",
            "U15", "classified:use_rotary_positional_embeddings"),
    _config("the denoiser rotary-geometry fact", "root.denoiser",
            "rope_theta",
            "numeric rotary operand awaits the denoiser-owned positional "
            "application and geometry proof in U10",
            "U15", "classified:rope_theta"),
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
    # U9 moves recursive vision architecture from the config-authored shell to
    # exact source readers.  These checkpoint values now enter only as operands
    # of those readers, but the raw ``extras.modalities`` projection still has
    # no FactLedger/ProjectionReceipt route.  Keep every occurrence visible and
    # owner-exact until U14 performs that representation cutover; marking them
    # merely "accessed" here would recreate the inspected-vs-consumed lie.
    _config("the source-bound vision tower repetition operand",
            "root.vision", "vision_config.depth",
            "U9 proves the exact repeated vision stage and binds this depth "
            "operand; U14 must ledger and receipt the modality projection",
            "U14", "fact_registered:vision_tower_depth"),
    _config("the source-bound vision hidden-width operands",
            "root.vision", "vision_config.embed_dim",
            "U9 binds this width to exact vision constructors; U14 must "
            "ledger its owner-qualified projections",
            "U14", "fact_registered:vision_hidden_geometry"),
    _config("the source-bound vision FFN activation",
            "root.vision", "vision_config.hidden_act",
            "U9 proves the activation dispatch at the exact vision FFN; U14 "
            "must ledger and receipt that mechanism operand",
            "U14", "fact_registered:vision_ffn_activation"),
    _config("the source-bound patch input-channel operand",
            "root.vision", "vision_config.in_chans",
            "U9 binds this spelling to the exact patch constructor; U14 must "
            "ledger the patch-geometry projection",
            "U14", "fact_registered:vision_patch_input_channels"),
    _config("the source-bound vision FFN expansion operand",
            "root.vision", "vision_config.mlp_ratio",
            "U9 binds this ratio to the exact vision FFN construction; U14 "
            "must ledger and receipt the derived width",
            "U14", "fact_registered:vision_ffn_ratio"),
    _config("the source-bound vision attention head count",
            "root.vision", "vision_config.num_heads",
            "U9 binds this count to the exact vision attention occurrence; "
            "U14 must ledger and receipt its geometry",
            "U14", "fact_registered:vision_attention_geometry"),
    _config("the source-bound spatial patch-size operand",
            "root.vision", "vision_config.patch_size",
            "U9 binds this size to the exact patch constructor; U14 must "
            "ledger and receipt the patch-geometry projection",
            "U14", "fact_registered:vision_patch_geometry"),
    _config("the source-bound spatial merge-size operand",
            "root.vision", "vision_config.spatial_merge_size",
            "U9 binds this size to the exact merger/position route; U14 must "
            "ledger and receipt the owner-qualified projection",
            "U14", "fact_registered:vision_projector_geometry"),
    _config("the source-bound temporal patch-size operand",
            "root.vision", "vision_config.temporal_patch_size",
            "U9 binds this size to the exact 3D patch constructor; U14 must "
            "ledger and receipt the temporal patch geometry",
            "U14", "fact_registered:vision_temporal_patch_geometry"),
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
    _config("the model-stage embedding-input scale", "root",
            "embedding_multiplier",
            "the checkpoint value affects the root embedding path, whose "
            "exact source-bound input projection is U14 work",
            "U14", "fact_registered:embedding_scale"),
    _config("the language-head logits divisor", "root", "logits_scaling",
            "the checkpoint value affects the root output head, whose exact "
            "source-bound output projection is U14 work",
            "U14", "fact_registered:logits_scale"),
    _config(
        "classification of a config value copied into attention state but "
        "excluded from the proved position initializer",
        "root", "attn_config.rope_theta",
        "the exact attention constructor stores this checkpoint value, while "
        "the exact applied Q/K rotation is initialized independently from a "
        "framework-normalized code default; because the attention object is "
        "also passed to a selectable backend, absence of a direct field read "
        "is not yet a complete global deadness proof",
        "U11", "classified:attn_config.rope_theta"),
    # U8 rotary debt is closed: exact applied-position and initializer readers
    # consume only source-selected operands into typed schedule/theta facts.
    # Unused scaling metadata is scoped-ignored at its read site; completed
    # receipt obligations are never kept artificially pending by debt rows.
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
    # U10-F4 deleted the generic present-field -> chip author.  These UNet
    # declarations were previously made to look "handled" only because that
    # author read them; none currently has a source-bound structural consumer.
    # Keep them as exact U11 occurrences instead of restoring the read or
    # silently treating the field spelling as architecture.
    _config("the added-conditioning attention head count", "root.denoiser",
            "addition_embed_type_num_heads",
            "the checkpoint declaration needs the exact U11 added-conditioning "
            "constructor before it can parameterize a drawn mechanism",
            "U11", "fact_registered:unet_added_conditioning"),
    _config("the denoiser input-centering operation", "root.denoiser",
            "center_input_sample",
            "field presence cannot create the input arithmetic; U11 must bind "
            "the exact root-forward branch",
            "U11", "fact_registered:unet_input_centering"),
    _config("the class-conditioning concatenation", "root.denoiser",
            "class_embeddings_concat",
            "U11 must prove where the class embedding joins the state",
            "U11", "fact_registered:unet_class_conditioning"),
    _config("the input-convolution kernel", "root.denoiser",
            "conv_in_kernel",
            "numeric declaration awaits the exact U11 input-convolution owner",
            "U11", "fact_registered:unet_input_projection"),
    _config("the output-convolution kernel", "root.denoiser",
            "conv_out_kernel",
            "numeric declaration awaits the exact U11 output-convolution owner",
            "U11", "fact_registered:unet_output_projection"),
    _config("the downsampling padding", "root.denoiser",
            "downsample_padding",
            "numeric declaration awaits the exact U11 downsampler occurrence",
            "U11", "fact_registered:unet_downsampling"),
    _config("the dual-cross-attention selector", "root.denoiser",
            "dual_cross_attention",
            "a boolean declaration cannot prove the constructed attention path",
            "U11", "fact_registered:unet_attention"),
    _config("the mid-block residual scale", "root.denoiser",
            "mid_block_scale_factor",
            "the scale must bind to the exact U11 mid-block residual expression",
            "U11", "fact_registered:unet_residual_topology"),
    _config("the ResNet output residual scale", "root.denoiser",
            "resnet_out_scale_factor",
            "the scale must bind to the exact U11 ResNet residual expression",
            "U11", "fact_registered:unet_residual_topology"),
    _config("the time-activation skip branch", "root.denoiser",
            "resnet_skip_time_act",
            "U11 must prove the exact ResNet timestep-conditioning dataflow",
            "U11", "fact_registered:unet_time_conditioning"),
    _config("the ResNet time scale/shift mechanism", "root.denoiser",
            "resnet_time_scale_shift",
            "the enum cannot select modulation until U11 binds its source branch",
            "U11", "fact_registered:unet_time_conditioning"),
    _config("the timestep-condition projection width", "root.denoiser",
            "time_cond_proj_dim",
            "numeric declaration awaits the exact U11 condition projector",
            "U11", "fact_registered:unet_time_conditioning"),
    _config("the timestep-embedding activation", "root.denoiser",
            "time_embedding_act_fn",
            "activation spelling awaits the exact U11 embedding operation",
            "U11", "fact_registered:unet_time_embedding"),
    _config("the timestep-embedding mechanism", "root.denoiser",
            "time_embedding_type",
            "the enum cannot create an embedding graph without U11 source proof",
            "U11", "fact_registered:unet_time_embedding"),
    _config("the timestep post-activation", "root.denoiser",
            "timestep_post_act",
            "activation spelling awaits the exact U11 post-embedding operation",
            "U11", "fact_registered:unet_time_embedding"),
    _config("the conditioning card's max-text-tokens chip (CogVideoX "
            "spelling)", "root.denoiser", "max_text_seq_length",
            "declared conditioning limit — the same mechanism as the "
            "max_sequence_length row; chip-only today",
            "U15", "classified:max_text_seq_length"),
    _config("the DiT FFN inner-width derivation", "root.denoiser",
            "ffn_dim_multiplier",
            "Lumina-family inner width = round(2/3*4h; multiple_of, "
            "ffn_dim_multiplier); chip-only today while the drawn width "
            "stays unknown until the derivation is registered",
            "U15", "classified:ffn_dim_multiplier"),
    _config("the DiT FFN inner-width derivation (rounding quantum)",
            "root.denoiser", "multiple_of",
            "pairs with ffn_dim_multiplier",
            "U15", "classified:multiple_of"),
    _config("the cross-attention sublayer's own head geometry",
            "root.denoiser", "num_cross_attention_heads",
            "Sana declares distinct cross heads; the drawn cross-attn "
            "sublayer reuses the SELF spec today (chip only)",
            "U15", "classified:num_cross_attention_heads"),
    _config("the cross-attention sublayer's own head geometry (head width)",
            "root.denoiser", "cross_attention_head_dim",
            "pairs with num_cross_attention_heads",
            "U15", "classified:cross_attention_head_dim"),
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
            "U9", "classified:text_config.is_encoder_decoder"),
    _config("class-stamped decoderness boilerplate on the vision sub-config",
            "root.vision", "vision_config.is_encoder_decoder",
            "class-serialized copy the parse never reads (the root-level "
            "occurrence is the consumed decoderness read)",
            "U9", "classified:vision_config.is_encoder_decoder"),
)

# U10-F4 made the source/config projector the sole denoiser author.  The old
# config-driven cards had silently counted every familiar declaration as
# "handled" even when no source occurrence established what it meant.  These
# exact checkpoint paths are the honest carry-forward exposed by deleting that
# author.  They create no architecture and excuse no sibling/path: U15 must
# classify each exact owner/path occurrence after its source-bound use is
# known, or remove the read.  The
# VAE paths are independently owned U12 component work.
_DIFFUSION_SOURCE_CLOSURE_PATHS = (
    "added_kv_proj_dim", "attention_head_dim", "attention_out_bias",
    "attention_type", "axes_dim", "axes_dim_rope", "axes_dims_rope",
    "axes_lens", "bottleneck_size", "boundary_ratio", "cap_feat_dim",
    "caption_channels", "caption_projection_dim", "context_in_dim",
    "cross_attn_norm", "cross_attention_dim", "default_sample_size",
    "double_self_attention", "eps", "ffn_dim", "flip_sin_to_cos",
    "freq_dim", "freq_shift", "guidance_embeds", "hidden_size",
    "image_dim", "in_channels", "interpolation_scale",
    "joint_attention_dim", "mlp_ratio", "norm_elementwise_affine",
    "norm_type", "num_attention_heads", "num_embeds_ada_norm",
    "num_heads", "num_kv_heads", "num_layers", "num_mmdit_layers",
    "num_single_dit_layers", "num_single_layers", "only_cross_attention",
    "out_channels", "patch_size", "patch_size_t", "pooled_projection_dim",
    "pos_embed_max_size", "pos_embed_seq_len", "resolution_embeds",
    "rope_axes_dim", "rope_max_seq_len", "sample_frames", "sample_height",
    "sample_size", "sample_width", "scheduler",
    "spatial_interpolation_scale", "temporal_compression_ratio",
    "temporal_interpolation_scale", "text_dim", "text_embed_dim", "theta",
    "time_embed_dim", "time_factor", "time_max_period",
    "timestep_activation_fn", "timestep_guidance_channels",
    "upcast_attention", "use_additional_conditions",
    "use_linear_projection",
)
_VAE_SOURCE_CLOSURE_PATHS = (
    "_vae_config.add_attention_block", "_vae_config.attn_scales",
    "_vae_config.batch_norm_eps", "_vae_config.batch_norm_momentum",
    "_vae_config.decoder_act_fns", "_vae_config.decoder_block_types",
    "_vae_config.decoder_causal", "_vae_config.decoder_layers_per_block",
    "_vae_config.decoder_norm_types", "_vae_config.decoder_qkv_multiscales",
    "_vae_config.downsample_block_type",
    "_vae_config.encoder_block_out_channels",
    "_vae_config.encoder_block_types", "_vae_config.encoder_causal",
    "_vae_config.encoder_layers_per_block",
    "_vae_config.encoder_qkv_multiscales",
    "_vae_config.invert_scale_latents", "_vae_config.resnet_norm_eps",
    "_vae_config.sample_size", "_vae_config.spatial_compression_ratio",
    "_vae_config.spatial_expansions", "_vae_config.spatio_temporal_scaling",
    "_vae_config.temperal_downsample", "_vae_config.temporal_expansions",
    "_vae_config.upsample_block_type",
)

_already_registered_config_paths = frozenset(
    (row.owner, row.source_occurrence) for row in STRUCTURAL_DEBT
    if row.sink_kind == "config_read")
STRUCTURAL_DEBT += tuple(
    _config(
        "an exact denoiser source/config operand", "root.denoiser", path,
        "the config-driven denoiser author is deleted; this checkpoint value "
        "cannot project until an exact U10 source occurrence retains its path",
        "U15", f"classified:{path}")
    for path in _DIFFUSION_SOURCE_CLOSURE_PATHS
    if ("root.denoiser", path) not in _already_registered_config_paths
)
STRUCTURAL_DEBT += tuple(
    _config(
        "an exact VAE component operand", "root.vae", path,
        "the generic config-fact chip is not denoiser authority; U12 must bind "
        "this exact VAE occurrence to its component source graph",
        "U12", f"classified:{path}")
    for path in _VAE_SOURCE_CLOSURE_PATHS
    if ("root.vae", path) not in _already_registered_config_paths
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
    ('model_unfolder/evidence/conformance.py', '_check_component_storage_facts', 'conformance', 'raw_extras', '261bebf740a7e89d'),
    ('model_unfolder/evidence/conformance.py', '_check_fusion_facts', 'conformance', 'raw_extras', '007defd60449a154'),
    ('model_unfolder/evidence/conformance.py', '_check_fusion_facts', 'conformance', 'backward_import', 'bc17002606a6c102'),
    ('model_unfolder/evidence/conformance.py', '_check_projector_facts', 'conformance', 'raw_extras', '3382f12b04c4a08e'),
    ('model_unfolder/evidence/conformance.py', '_check_recursive_component_facts', 'conformance', 'backward_import', 'bc17002606a6c102'),
    ('model_unfolder/evidence/conformance.py', '_check_recursive_component_facts', 'conformance', 'raw_extras', 'abeededaa3def2a3'),
    ('model_unfolder/evidence/conformance.py', '_check_storage_facts', 'conformance', 'backward_import', 'b5c1d62f90035b94'),
    ('model_unfolder/evidence/conformance.py', '_check_storage_facts', 'conformance', 'raw_extras', '8309e956d0a3abbf'),
    ('model_unfolder/evidence/conformance.py', '_config_field_value', 'conformance', 'raw_config', 'f9c05d8e8878f612'),
    ('model_unfolder/evidence/conformance.py', '_constructor_envs', 'conformance', 'source_reopen', 'a821dd49ada1014e'),
    ('model_unfolder/evidence/conformance.py', '_drawn_fusion_routes', 'conformance', 'raw_extras', '342c661d01d15158'),
    ('model_unfolder/evidence/conformance.py', '_family', 'conformance', 'raw_config', '6aed1a33205d116b'),
    ('model_unfolder/evidence/conformance.py', '_imported_model_files', 'conformance', 'source_reopen', '7fef0151ecf6fcd9'),
    ('model_unfolder/evidence/conformance.py', '_init_helper_block_classes', 'conformance', 'source_reopen', 'a821dd49ada1014e'),
    ('model_unfolder/evidence/conformance.py', '_is_block_class', 'conformance', 'backward_import', 'd0e50b755ed393d8'),
    ('model_unfolder/evidence/conformance.py', '_op_is_dormant', 'conformance', 'raw_config', '6aff71a545173320'),
    ('model_unfolder/evidence/conformance.py', '_reachable_forward_ops', 'conformance', 'backward_import', 'd3dd4554c4cd31a7'),
    ('model_unfolder/evidence/conformance.py', '_selected_init_refs', 'conformance', 'source_reopen', 'a821dd49ada1014e'),
    ('model_unfolder/evidence/conformance.py', 'check_fact_conformance', 'conformance', 'backward_import', 'd0e50b755ed393d8'),
    ('model_unfolder/evidence/conformance.py', 'check_fact_conformance', 'conformance', 'raw_config', 'acbeae587f3e1678'),
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
    ('model_unfolder/expanded/sections.py', '_is_projected_diffusion', 'json', 'raw_extras', '8048b69bd5b33a4e'),
    ('model_unfolder/expanded/sections.py', '_projected_diffusion_io', 'json', 'raw_extras', '8309e956d0a3abbf'),
    ('model_unfolder/expanded/sections.py', '_is_diffusion', 'json', 'raw_extras', '538efc70e7a2bf65'),
    ('model_unfolder/expanded/sections.py', 'build_dimensions', 'json', 'raw_extras', 'ab038e67ecaac4d6'),
    ('model_unfolder/expanded/sections.py', 'build_io', 'json', 'raw_extras', 'b99c4724e4ce0076'),
    ('model_unfolder/params.py', '_attn_params', 'params', 'spec_default', 'ee4e0ed83a8c6b3f'),
    ('model_unfolder/params.py', '_ffn_params', 'params', 'spec_default', '949c0c547ad432a2'),
    ('model_unfolder/renderers/html/block_views/declared_ops.py', 'build_declared_ops_view', 'renderer', 'raw_extras', 'ba8f51d3eafee8db'),
    ('model_unfolder/renderers/html/block_views/modality_views/common.py', 'audio_input', 'renderer', 'raw_extras', '680261edc5f7acee'),
    ('model_unfolder/renderers/html/block_views/modality_views/common.py', 'fusion_spec', 'renderer', 'raw_extras', 'dad11fdde8a702b6'),
    ('model_unfolder/renderers/html/block_views/modality_views/common.py', 'video_input', 'renderer', 'raw_extras', '658b78f2c8b074a7'),
    ('model_unfolder/renderers/html/block_views/modality_views/common.py', 'vision_input', 'renderer', 'raw_extras', 'aec51c4910a259ec'),
    ('model_unfolder/renderers/html/block_views/modality_views/conditioning.py', 'conditioning_input', 'renderer', 'raw_extras', '3fbbeba7773755e3'),
    ('model_unfolder/renderers/html/block_views/modality_views/video.py', 'build_video_path_view', 'renderer', 'raw_extras', 'de8ef3ebd9f7f11f'),
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
    ('model_unfolder/renderers/html/metadata_modalities.py', '_multimodal_block_lookup', 'renderer', 'raw_extras', '3e2116805ecafa81'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_unified_fusion_children', 'renderer', 'raw_extras', '6d2771bda17006b0'),
    ('model_unfolder/renderers/html/metadata_modalities.py', '_vision_cell_cards', 'renderer', 'backward_import', '962109a2344e67b0'),
    ('model_unfolder/renderers/html/sections.py', '_diffusion_stats', 'renderer', 'raw_extras', 'da27bb4cf97df230'),
    ('model_unfolder/renderers/html/sections.py', '_stats_banner', 'renderer', 'raw_extras', '538efc70e7a2bf65'),
    ('model_unfolder/renderers/html/views.py', '_build_architecture_view', 'renderer', 'raw_extras', '933be75c6d3a7af4'),
    ('model_unfolder/renderers/html/views.py', '_is_diffusion_architecture', 'renderer', 'raw_extras', '538efc70e7a2bf65'),
    ('model_unfolder/renderers/html/views_diffusion.py', '_build_block_diffusion_loop_cards', 'renderer', 'raw_extras', '261bebf740a7e89d'),
    ('model_unfolder/renderers/html/views_diffusion.py', '_build_block_diffusion_view', 'renderer', 'raw_extras', '1c3216f252696602'),
    ('model_unfolder/renderers/html/views_diffusion.py', '_build_loop_cards', 'renderer', 'raw_extras', '1daab3bea1a4e581'),
    ('model_unfolder/renderers/html/views_diffusion.py', '_build_loop_descendant_levels', 'renderer', 'raw_extras', '261bebf740a7e89d'),
    ('model_unfolder/renderers/html/views_diffusion.py', '_build_loop_view', 'renderer', 'raw_extras', '07d25d5749cfe49b'),
    ('model_unfolder/renderers/html/views_diffusion.py', 'render_diffusion_fragment', 'renderer', 'raw_extras', '45077a51a50d7e69'),
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
