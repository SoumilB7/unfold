"""The transformer-LLM parser — the only adapter.

There are no per-family adapters and no "supported model" gate.  Every
transformer-LLM config flows through ``parse()``.

Architectural mechanisms are source- and owner-driven. Config values are
operands only after an exact reader proves where the modeling code consumes
them. A count or activation spelling may remain useful geometry while the
mechanism that uses it stays unknown; it cannot select attention, FFN, MoE,
position, or layer topology by itself. Transitional config-authored facts are
named in the structural-debt register and migrate in U7/U8 rather than being
extended here.

Warnings policy: warn only for *specific* config problems (missing
critical field, unrecognized layer_type value, …).  Never warn just
because no family-specific code path matched — there are none.
"""
from __future__ import annotations

from typing import Any

from . import debug
from ...everchanging import (load_aliases, load_layer_type_labels,
                            load_layer_schedules)
from ...ir import AttentionSpec, CrossLayerEdge, FFNSpec, ModelIR
from .assembly import decoder_extras, decoder_layer, parallel_decoder_layer
from .blocks import mtp_head_block
from .common import architecture_name, get_config_value as _g, model_name
from .special_parts.per_layer_embedding import (
    per_layer_embedding_blocks,
    per_layer_embedding_extras,
)
from .special_parts.modalities import multimodal_extras
from ...evidence.identity_roles import identity_address
from ...evidence import config_access as _config_access
from .special_parts.modalities.fusion import apply_fusion_evidence
from .special_parts.modalities.vision import apply_projector_evidence, apply_vision_evidence
from .special_parts.modalities.audio import apply_audio_evidence
from .special_parts.modalities.detect import cross_attention_layers as _cross_attention_layers


# ---------------------------------------------------------------------------
# Field aliases: every canonical field has a list of names we look up in order.
# The table itself is *data*, loaded from ``everchanging/aliases.json`` so a new
# config dialect is supported by editing JSON — no code change here.  Adding a
# new alias is the only kind of per-family handling that exists.
# ---------------------------------------------------------------------------

_ALIASES: dict[str, list[str]] = load_aliases()


# Per-layer attention-type label vocabulary — data, not code (everchanging/
# transformer/layer_types.yaml).  Add a new spelling there, not here.
_LAYER_TYPE_LABELS = load_layer_type_labels()
_SLIDING_LABELS = set(_LAYER_TYPE_LABELS["sliding"])
_FULL_LABELS    = set(_LAYER_TYPE_LABELS["full"])
_COMPRESSED_SPARSE_LABELS = set(_LAYER_TYPE_LABELS["compressed_sparse"])
_HEAVILY_COMPRESSED_LABELS = set(_LAYER_TYPE_LABELS["heavily_compressed"])

# Per-layer TYPE-SCHEDULE vocabulary — the six config spellings of "which type
# is layer i" normalized into the canonical layer_types / MoE-membership readers
# (everchanging/transformer/layer_schedules.yaml).  Data, not code — a new
# dialect is a row there.  U3.
_LAYER_SCHEDULES = load_layer_schedules()
# canonical layer_types token -> the token-MIXER cell it draws (non-softmax
# mixer that replaces attention on that layer).
_MIXER_KINDS: dict[str, str] = _LAYER_SCHEDULES["mixer_kinds"]

def _declared_scores_scale(multiplier, query_pre_attn_scalar, head_dim):
    """The EFFECTIVE config-declared QK^T scale, or None when it equals the
    default 1/sqrt(head_dim) (drawing sqrt(dim) is then exactly true).

    Two declaration dialects, each with its own semantics:
    * ``attention_multiplier`` (Granite family) — the scale directly;
    * ``query_pre_attn_scalar`` (Gemma-2/3) — scale = value ** -0.5.
    """
    scale = None
    if multiplier is not None:
        scale = float(multiplier)
    elif query_pre_attn_scalar:
        scale = float(query_pre_attn_scalar) ** -0.5
    if scale is None or not head_dim:
        return None
    default = float(head_dim) ** -0.5
    return None if abs(scale - default) <= 1e-6 * default else scale


def _spellings(canonical: str) -> list[str]:
    """The declared true-synonym spellings for one canonical field."""
    return list(dict.fromkeys([canonical, *_ALIASES.get(canonical, ())]))


def _carries(cfg: Any, canonical: str) -> bool:
    """Pure OCCURRENCE-membership probe (no value read, no event) — used for
    adapter-shape dispatch, never for a value decision (REC-3 §9.2: the
    first-hit value resolver is DELETED, not wrapped)."""
    return any((s in cfg) if isinstance(cfg, dict) else hasattr(cfg, s)
               for s in _spellings(canonical))


def _source_files(cfg: Any, context=None):
    """Return this parse's already-resolved source files.

    Direct adapter callers still get a complete parse: ``parse`` creates the
    context once before any detector runs.  The fallback is retained only for
    isolated helper tests and third-party calls to these private helpers.
    """
    if context is not None:
        return context.source_bundle.files
    from ...evidence.sources import resolve_source_files
    return resolve_source_files(cfg, source="local").files


def _cell_topology_result(
        context=None, *, config_path=(), config_root=None):
    """One call-local exact decoder-cell topology result.

    The reader starts at the selected config path and exact repeated-block
    occurrence, then joins the canonical attention/FFN/norm calls to positive
    residual equations.  It has no whole-file candidate selection and no
    conventional pre/sequential fallback.
    """
    if context is None:
        return None
    from ...evidence.cell_topology import decoder_cell_topology_for_path
    from ...evidence import config_access as _config_access
    path = tuple(config_path)
    resolutions = {}

    def _select(exact_path):
        if config_root is None or not exact_path:
            return None
        parent = config_root
        for part in exact_path[:-1]:
            parent = (parent.get(part) if isinstance(parent, dict)
                      else getattr(parent, part, None))
            if parent is None:
                return None
        resolution = _config_access.resolve(
            parent, exact_path[-1], path=tuple(exact_path[:-1]))
        expected = ".".join(exact_path)
        if resolution.ambiguous or not resolution.present \
                or resolution.selected_path != expected:
            return None
        resolutions[tuple(exact_path)] = resolution
        return resolution.value

    def _consume_dependency(exact_path, fact_key):
        resolution = resolutions.get(tuple(exact_path))
        if resolution is None:
            raise ValueError(
                "cell topology cited a config path it did not resolve")
        resolution.bind(
            reader="decoder_cell_topology_for_path",
            fact_owner="decoder.layer", fact_key=fact_key)
        resolution.consume_decision(
            mechanism="cell_topology",
            fact_owner="decoder.layer", fact_key=fact_key,
            reader="decoder_cell_topology_for_path",
            status="code_and_config")

    def _read():
        result = decoder_cell_topology_for_path(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True,
            config_selector=_select,
        )
        if result.status == "resolved":
            for fact_key, dependencies in (
                    ("norm_placement", result.value.norm_config_paths),
                    ("residual_topology",
                     result.value.residual_config_paths)):
                for dependency in dependencies:
                    _consume_dependency(dependency, fact_key)
        return result

    return context.cached_reader_result(
        "decoder.layer.cell_topology",
        path,
        _read,
    )


def _decoder_norm_result(context=None, *, config_path=()):
    """One call-local exact decoder-block normalization result."""
    if context is None:
        return None
    from ...evidence.decoder_norm import decoder_norm_kind_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.layer.norm_kind",
        config_path,
        lambda: decoder_norm_kind_for_path(
            context.program_index(),
            context.source_bundle,
            config_path,
            allow_root_stage=True,
        ),
    )


def _ffn_mechanism_result(
    context=None, *, config_path=(), config_root=None,
):
    """One call-local exact-owner ordinary FFN result."""
    if context is None:
        return None
    from ...evidence.ffn_mechanism import decoder_ffn_mechanism_for_path
    from ...evidence import config_access as _config_access
    config_path = tuple(config_path)

    def _select(exact_path):
        """Supply one exact boolean only after the source reader names it."""
        if config_root is None or not exact_path:
            return None
        parent = config_root
        for part in exact_path[:-1]:
            parent = (parent.get(part) if isinstance(parent, dict)
                      else getattr(parent, part, None))
            if parent is None:
                return None
        resolution = _config_access.resolve(
            parent, exact_path[-1], path=tuple(exact_path[:-1]))
        expected = ".".join(exact_path)
        if resolution.ambiguous or not resolution.present \
                or resolution.selected_path != expected \
                or not isinstance(resolution.value, bool):
            return None
        resolution.bind(
            reader="decoder_ffn_mechanism_for_path",
            fact_owner="decoder.ffn", fact_key="gated")
        decision = resolution.consume_decision(
            mechanism="ffn_mechanism",
            fact_owner="decoder.ffn", fact_key="gated",
            reader="decoder_ffn_mechanism_for_path",
            status="code_and_config")
        return decision.value

    return context.cached_reader_result(
        "decoder.ffn.mechanism",
        config_path,
        lambda: decoder_ffn_mechanism_for_path(
            context.program_index(),
            context.source_bundle,
            config_path,
            allow_root_stage=True,
            config_selector=_select,
        ),
    )


def _code_embedding_norm(cfg: Any, context=None) -> str | None:
    """An unconditional norm whose output feeds the exact repeated block.

    This is the U3 owner-qualified reader, not a whole-file role/name scan.
    Unresolved evidence stays ``None`` and therefore cannot fabricate a block.
    """
    if context is None:
        return None
    from ...evidence.embedding_bookend import embedding_stage_norm_evidence
    evidence = embedding_stage_norm_evidence(
        context.program_index(), context.source_bundle,
        allow_root_stage=True)
    return evidence.value if evidence.status == "resolved" else None


def _attention_storage_result(context, config_path):
    if context is None:
        return None
    from ...evidence.attention_storage import (
        decoder_attention_projection_storage_for_path,
    )
    return context.cached_reader_result(
        "decoder.attention.projection_storage",
        config_path,
        lambda: decoder_attention_projection_storage_for_path(
            context.program_index(),
            context.source_bundle,
            tuple(config_path),
            allow_root_stage=True,
        ),
    )


def _attention_mechanism_result(context=None, *, config_path=()):
    """One call-local exact-owner attention mechanism binding."""
    if context is None:
        return None
    from ...evidence.attention import decoder_attention_mechanism_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.mechanism",
        config_path,
        lambda: decoder_attention_mechanism_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _gated_delta_geometry_result(context=None, *, config_path=()):
    """One call-local exact recurrent-mixer geometry binding."""
    if context is None:
        return None
    from ...evidence.attention import decoder_gated_delta_geometry_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.gated_delta_geometry",
        config_path,
        lambda: decoder_gated_delta_geometry_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _code_attention_storage_mode(
        cfg: Any, context=None, *, config_path=()) -> str | None:
    """Owner-qualified Q/K/V storage; uncertainty never becomes split/fused."""
    result = _attention_storage_result(context, config_path)
    if result is None or result.status != "resolved":
        return None
    return "split_qkv" if result.value == "split" else result.value


def _qk_norm_result(context=None, *, config_path=()):
    """One call-local exact-owner Q/K-normalization result."""
    if context is None:
        return None
    from ...evidence.qk_norm import decoder_qk_norm_evidence_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.qk_norm",
        config_path,
        lambda: decoder_qk_norm_evidence_for_path(
            context.program_index(),
            context.source_bundle,
            config_path,
            allow_root_stage=True,
        ),
    )


def _resolve_qk_norm_layers(
        code_ev, cfg, num_layers: int, *,
        context=None, config_path=()) -> list[bool | None]:
    """Per-layer Q/K-norm facts from the code evidence.

    Code-first resolution: unconditional construction → True everywhere (the
    config is not consulted); gated → AND of the values of the exact config
    fields the code itself names, read from THIS checkpoint (per-layer when the
    code indexes by layer index); no positive source proof / unresolvable gate
    → unknown. A declaration without an exact owner-bound use is not a
    mechanism fact. Source silence never proves absence."""
    n = max(int(num_layers or 0), 0)
    if code_ev is None:
        return [None] * n
    if code_ev.present is True:
        return [True] * n
    per_layer = [True] * n
    for atom in code_ev.gate:
        if tuple(atom.config_path[:-1]) != tuple(config_path):
            return [None] * n
        if context is None:
            missing = object()
            raw = _g(cfg, atom.field, missing)
            if raw is missing:
                return [None] * n
        else:
            resolution = _config_access.resolve(
                cfg, atom.field, (), path=tuple(config_path))
            if resolution.ambiguous or not resolution.present:
                return [None] * n
            raw = resolution.consume_decision(
                mechanism="qk_norm_gate",
                fact_owner="decoder.attention",
                fact_key="qk_norm",
                reader="adapters.transformer.parser._resolve_qk_norm_layers",
                status="code_and_config",
            ).value
        if atom.per_layer:
            if not isinstance(raw, (list, tuple)) or len(raw) < n:
                return [None] * n
            per_layer = [p and bool(raw[i]) for i, p in enumerate(per_layer)]
        else:
            if raw is None:
                return [None] * n
            per_layer = [p and bool(raw) for p in per_layer]
    return per_layer


def _code_parallel_norm_count(
    cfg: Any, context=None, *, config_path=(),
):
    """Exact norm occurrences feeding the selected attention and FFN inputs."""
    if context is None:
        return None
    from ...evidence.parallel_norm import (
        decoder_parallel_norm_count_for_path,
    )
    result = decoder_parallel_norm_count_for_path(
        context.program_index(),
        context.source_bundle,
        tuple(config_path),
        allow_root_stage=True,
    )
    return (
        result.value.norm_count
        if result.status == "resolved" else None
    )


def _projection_bias_result(
    context, mechanism, config_path, *, ffn_mechanism_result=None,
):
    """Call-local exact-owner projection bias evidence."""
    if context is None:
        return None
    from ...evidence.projection_bias import (
        decoder_attention_bias_for_path,
        decoder_ffn_bias_for_path,
    )
    readers = {
        "attention": decoder_attention_bias_for_path,
        "ordinary_ffn": decoder_ffn_bias_for_path,
    }
    reader = readers[mechanism]
    path = tuple(config_path)
    return context.cached_reader_result(
        f"decoder.{mechanism}.projection_bias",
        path,
        lambda: reader(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True,
            **({"mechanism_result": ffn_mechanism_result}
               if mechanism == "ordinary_ffn" else {})),
    )


def _code_attention_bias(cfg: Any, context=None, *, config_path=()):
    """Uniform exact Q/K/V/O bias or their one exact bound config path."""
    result = _projection_bias_result(context, "attention", config_path)
    return result if result is not None and result.status == "resolved" else None


def _attention_output_projection_result(context=None, *, config_path=()):
    """One call-local exact attention-result -> output-Linear proof."""
    if context is None:
        return None
    from ...evidence.attention_output import (
        decoder_attention_output_projection_for_path,
    )
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.output_projection",
        path,
        lambda: decoder_attention_output_projection_for_path(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True),
    )


def _code_moe_schedule(cfg: Any, context=None, num_layers: int = 0):
    """Per-layer MoE?/dense schedule READ FROM THE DECODER LAYER's CONSTRUCTION —
    the code-authoritative replacement for the config schedule flags: which
    layers build an experts class (name-independent) as their FFN field, gated
    per layer.  Returns ``list[bool]`` (len num_layers) or None on any doubt
    (hybrid SSM-MoE, exotic gate, no source) → caller stays unknown.
    Best-effort, never raises into the parse."""
    try:
        from ...evidence.patterns import decoder_moe_schedule_from_files
        sched = decoder_moe_schedule_from_files(_source_files(cfg, context), cfg)
        if sched is not None and num_layers and len(sched) == num_layers:
            return sched
        return None
    except Exception:
        return None


def _code_router(cfg: Any, context=None):
    """MoE routing behaviour READ FROM THE MODELING SOURCE — the code channel
    for the score transform / aux-loss-free bias / sparsemixer that modern
    checkpoints leave out of config (GLM-4.5 copied DeepSeek-V3's routing code
    but not its ``scoring_func``/``topk_method`` strings).  Best-effort, never
    raises into the parse."""
    try:
        from ...evidence.patterns import decoder_router_evidence_from_files
        return decoder_router_evidence_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _score_scaling_result(context=None, *, config_path=()):
    """One call-local exact score-product-to-softmax result."""
    if context is None:
        return None
    from ...evidence.attention import (
        decoder_attention_score_scaling_for_path,
    )
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.score_scaling",
        config_path,
        lambda: decoder_attention_score_scaling_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _attention_logit_softcap_result(context=None, *, config_path=()):
    """Call-local exact score-softcap source/config binding."""
    if context is None:
        return None
    from ...evidence.attention import (
        decoder_attention_logit_softcap_for_path,
    )
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.logit_softcap",
        config_path,
        lambda: decoder_attention_logit_softcap_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _attention_qkv_clip_result(context=None, *, config_path=()):
    """Call-local exact fused-QKV projection/clamp result."""
    if context is None:
        return None
    from ...evidence.attention import decoder_attention_qkv_clip_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.qkv_clip",
        config_path,
        lambda: decoder_attention_qkv_clip_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _attention_cache_result(context=None, *, config_path=()):
    """Call-local exact projected K/V -> cache update -> compute result."""
    if context is None:
        return None
    from ...evidence.attention import decoder_attention_cache_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.cached",
        config_path,
        lambda: decoder_attention_cache_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _code_scores_scaled(
        cfg: Any, context=None, *, config_path=()) -> bool | None:
    """Score scaling from the exact selected attention occurrence."""
    result = _score_scaling_result(context, config_path=config_path)
    return (
        result.value.scaled
        if result is not None and result.status == "resolved" else None)


def _code_mlp_bias(
    cfg: Any, context=None, *, config_path=(), ffn_mechanism_result=None,
) -> bool | None:
    """Source-only bias of the exact ordinary-FFN projection occurrences."""
    result = _projection_bias_result(
        context, "ordinary_ffn", config_path,
        ffn_mechanism_result=ffn_mechanism_result)
    return (
        result.value.value
        if result is not None and result.status == "resolved" else None
    )


def _attention_sinks_result(context=None, *, config_path=()):
    """Call-local learned-sink proof for one exact attention occurrence."""
    if context is None:
        return None
    from ...evidence.attention_sinks import decoder_attention_sinks_for_path
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.sinks",
        path,
        lambda: decoder_attention_sinks_for_path(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True),
    )


def _code_attention_sinks(cfg: Any, context=None, *, config_path=()) -> bool:
    """Whether an exact learned Parameter joins scores before exact softmax.

    This is positive-only source evidence.  Incomplete source or an unproven
    chain omits the sink mechanism; neither field spellings nor file-wide
    markers can manufacture it.
    """
    result = _attention_sinks_result(
        context, config_path=config_path)
    return result is not None and result.status == "resolved"


def _cross_attention_schedule_result(context=None, *, config_path=()):
    """Call-local additive cross-attention proof for one exact decoder block."""
    if context is None:
        return None
    from ...evidence.cross_attention_schedule import (
        decoder_cross_attention_all_layers_for_path,
    )
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.cross_all_layers",
        path,
        lambda: decoder_cross_attention_all_layers_for_path(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True),
    )


def _code_cross_attention_all_layers(context=None, *, config_path=()):
    """Positive-only exact dual-attention construction evidence."""
    result = _cross_attention_schedule_result(
        context, config_path=config_path)
    return result is not None and result.status == "resolved"


def _code_intermediate_size(cfg: Any, context=None, *, config_path=()):
    """Exact-owner FFN width from its output-projection input expression."""
    if context is None:
        return None
    from ...evidence.ffn_width import \
        decoder_ffn_intermediate_width_for_path
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.ffn.intermediate_width", path,
        lambda: decoder_ffn_intermediate_width_for_path(
            context.program_index(), context.source_bundle, path, cfg,
            allow_root_stage=True),
    )


def _code_rope_dim(cfg: Any, context=None) -> int | None:
    """The rotated head-width read from an explicit-dim rotary CONSTRUCTION
    (ChatGLM's ``RotaryEmbedding(rotary_dim // 2)``) — only consulted when
    every config spelling of the fraction is silent.  Best-effort, never
    raises into the parse."""
    try:
        from ...evidence.patterns import decoder_rope_dim_from_files
        return decoder_rope_dim_from_files(_source_files(cfg, context), cfg=cfg)
    except Exception:
        return None


def _expert_storage_result(context=None, *, config_path=()):
    """One call-local exact-address routed-expert storage result."""
    if context is None:
        return None
    from ...evidence.expert_storage import \
        decoder_routed_expert_storage_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.ffn.expert_storage",
        config_path,
        lambda: decoder_routed_expert_storage_for_path(
            context.program_index(),
            context.source_bundle,
            config_path,
            allow_root_stage=True,
        ),
    )


def _code_expert_storage(
        cfg: Any, context=None, *, config_path=()) -> str | None:
    """Positive-only routed-expert storage from its exact construction path."""
    result = _expert_storage_result(
        context, config_path=config_path)
    return result.value.projection_mode \
        if result is not None and result.status == "resolved" else None


def _code_lm_head_tying(
    cfg: Any, context=None, *, config_path=(),
) -> bool | None:
    """Exact returned-head ↔ stack-feeding-embedding assignment proof.

    ``_tied_weights_keys`` remains capability only.  Missing/ambiguous source
    evidence stays unknown and falls through to the class-default tier.
    """
    if context is None:
        return None
    from ...evidence.weight_tying import manual_weight_tying_for_path
    result = manual_weight_tying_for_path(
        context.program_index(), context.source_bundle, tuple(config_path))
    return True if result.status == "resolved" else None


def _code_attention_causality(cfg: Any, context=None) -> str | None:
    """The mask DIRECTION read from mask-machinery calls / ``self.is_causal``
    literals in the modeling source (U2 P2d) — ``"causal"`` /
    ``"bidirectional"`` / ``None``.  ``if …is_decoder…`` gates around the
    calls are resolved from the checkpoint config, so one source file
    honestly yields either direction (BERT / T5-encoder)."""
    try:
        from ...evidence.patterns import attention_causality_from_files
        return attention_causality_from_files(_source_files(cfg, context), cfg)
    except Exception:
        return None


def _code_position(cfg: Any, context=None):
    """Typed positional evidence shared verbatim with fact-conformance."""
    try:
        from ...evidence.position import decoder_positional_evidence
        bundle = context.source_bundle if context is not None else None
        return decoder_positional_evidence(cfg, source="local", bundle=bundle)
    except Exception:
        # A detector failure is not permission to assert a source-derived fact.
        # Treat it as present-but-unresolved; fact-conformance will expose the
        # same state when the source oracle is available.
        from ...evidence.models import PositionalEvidence
        return PositionalEvidence("ambiguous", reason="positional detector raised")


from .common import TEXT_WRAPPER_KEYS as _TEXT_WRAPPER_KEYS
from .common import wrapper_path as _wrapper_path
from ...everchanging import load_composite_slots as _load_composite_slots


def _unwrap_text_with_path(
    cfg: Any, _depth: int = 0, *,
    class_defaults_by_path=None, _base_path=(),
) -> tuple[Any, tuple[str, ...]]:
    """If a multimodal wrapper hides the LM config under a sub-key, unwrap it.

    Handles one further level of nesting (e.g. Qwen3-Omni's
    ``thinker_config.text_config``) by recursing into a wrapper that doesn't
    itself carry transformer shape.  The address travels with the selected
    object: completing a sparse child through its config class creates a new
    mapping object, so an identity walk performed afterward cannot recover the
    checkpoint path.
    """
    if _depth > 3:
        return cfg, ()
    defaults_by_path = class_defaults_by_path or {}
    for key in _TEXT_WRAPPER_KEYS:
        sub = _g(cfg, key)
        if sub is None:
            continue
        # A composite AutoConfig nests sub-configs as OBJECTS.  Keep that exact
        # object as the read carrier: converting it to a dict would make a real
        # path point at one object while the read cites an unrelated copy.
        if _has_transformer_shape(sub):
            return sub, (key,)
        sub_mapping = (
            sub if isinstance(sub, dict)
            else sub.to_dict() if hasattr(sub, "to_dict") else None)
        if isinstance(sub_mapping, dict):
            absolute_path = (*_base_path, key)
            completed = defaults_by_path.get(absolute_path)
            if not isinstance(completed, dict):
                completed = _complete_config_from_transformers_registry(
                    sub_mapping)
            if _has_transformer_shape(completed):
                # Completion proves that this declared child is a transformer
                # config, but the parser must keep reading the checkpoint's
                # original object.  Class-supplied values travel separately as
                # typed class-default premises; replacing the object would
                # falsely report those values as checkpoint declarations.
                return sub, (key,)
            # Wrapper that itself nests the LM deeper (Omni thinker_config).
            nested, nested_path = _unwrap_text_with_path(
                sub, _depth + 1,
                class_defaults_by_path=defaults_by_path,
                _base_path=absolute_path)
            if nested_path and _has_transformer_shape(nested):
                return nested, (key, *nested_path)
    # Composite/seq2seq wrapper (MusicGen): the MAIN stack is a declared BARE
    # slot (``decoder`` — composite_slots vocabulary), not a ``*_config`` key.
    # A slot only counts when its child declares its own model_type; sparse
    # dicts are completed through HF's config registry like any nested LM.
    from ...everchanging import load_composite_slots
    for key, role in (load_composite_slots().get("slots") or {}).items():
        if role != "main":
            continue
        sub = _g(cfg, key)
        sub_mapping = (
            sub if isinstance(sub, dict)
            else sub.to_dict() if hasattr(sub, "to_dict") else None)
        if not isinstance(sub_mapping, dict) \
                or not sub_mapping.get("model_type"):
            continue
        if _has_transformer_shape(sub):
            return sub, (key,)
        absolute_path = (*_base_path, key)
        completed = defaults_by_path.get(absolute_path)
        if not isinstance(completed, dict):
            completed = _complete_config_from_transformers_registry(
                sub_mapping)
        if _has_transformer_shape(completed):
            return sub, (key,)
    return cfg, ()


def _unwrap_text(cfg: Any, _depth: int = 0) -> Any:
    """Compatibility value view; parsing consumes the address-carrying form."""
    return _unwrap_text_with_path(cfg, _depth)[0]


def _composite_encoder_model_type(cfg: Any) -> str | None:
    """The declared encoder-role slot's own model_type string, or None.

    Evidence chain for seq2seq composites: the slot NAME comes from the
    composite_slots vocabulary and only counts when the child itself declares
    a ``model_type`` (MusicGen's ``text_encoder: {model_type: t5, ...}``).
    The returned string is the config's own declaration — used for side-state
    wording, never for structural decisions."""
    from ...everchanging import load_composite_slots
    for key, role in (load_composite_slots().get("slots") or {}).items():
        if role != "encoder":
            continue
        sub = _g(cfg, key)
        if not isinstance(sub, dict) and hasattr(sub, "to_dict"):
            sub = sub.to_dict()
        if isinstance(sub, dict) and sub.get("model_type"):
            return str(sub.get("model_type"))
    return None


@identity_address
def _complete_config_from_transformers_registry(text_cfg: dict) -> dict:
    """Materialize sparse nested configs through HF's generic config registry."""
    model_type = str(text_cfg.get("model_type") or "").lower()
    if not model_type:
        return text_cfg

    try:
        from transformers import CONFIG_MAPPING
    except Exception:
        return text_cfg

    try:
        config_cls = CONFIG_MAPPING[model_type]
        completed = config_cls(**text_cfg)
    except Exception:
        return text_cfg

    if hasattr(completed, "to_dict"):
        return completed.to_dict()
    return text_cfg


def _has_transformer_shape(cfg: Any) -> bool:
    return any(
        _carries(cfg, field)
        for field in ("num_hidden_layers", "hidden_size", "num_attention_heads")
    )


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

def matches(_cfg: Any) -> bool:
    return True  # the only adapter — must be registered last in the global list


# REC-6 (§12.3): this adapter's parse root, DECLARED (never guessed from a
# module name).
ROOT_COMPONENT = "root"


def parse(cfg: Any, context=None) -> ModelIR:
    if context is None:
        from ...evidence.context import ParseContext
        context = ParseContext.build(cfg, source="local")
    debug.reset()  # start a fresh field-access record for this parse
    warnings: list[str] = []

    # ---- U2 P1: per-fact provenance (the FactLedger) ----
    # The first high-risk structural families record WHICH channel decided
    # them at their decision point.  This ledger is deliberately incremental;
    # facts not registered here are not silently claimed as covered.
    # ``oracle_missing`` vs ``ambiguous`` says WHY a registered fact is unknown.
    _facts = getattr(context, "facts", None)
    _source_present = bool(getattr(getattr(context, "source_bundle", None),
                                   "files", ()) or ())

    def _note_fact(owner: str, name: str, value, status: str, source=None):
        if _facts is not None:
            _facts.record(owner, name, value, status, source)

    def _note_bound_attention_fact(
            bound, reader_result, actual_config_paths):
        """Publish the U6 mechanism with its exact typed evidence channels."""
        if _facts is None:
            return
        from ...evidence.facts import EvidenceFact, SourceSpan as FactSourceSpan

        spans = tuple(dict.fromkeys(
            span for provenance in reader_result.provenance
            for span in provenance.spans))
        fact_spans = tuple(FactSourceSpan(
            component=span.source.component_key or "root",
            file=span.source.canonical_path,
            line=span.line,
        ) for span in spans)
        config_paths = tuple(
            ".".join(actual)
            for path, _value in bound.premises
            if (actual := actual_config_paths.get(path)) is not None)
        _facts.record_typed(EvidenceFact(
            key="mechanism",
            owner="decoder.attention",
            value=bound.kind,
            status="code_and_config",
            completeness="presence_only",
            source_spans=fact_spans,
            config_paths=config_paths,
            legacy_source="decoder_attention_mechanism_for_path",
            reason=(
                "exact owner source protocol joined to the exact selected "
                "checkpoint occurrences"),
        ))

    def _note_typed_fact(
            *, key, owner, value, status, reader_result, config_paths, reader,
            reason, completeness="complete"):
        """One native typed-fact writer shared by exact U6 evidence joins.

        Keeping the registry-validated write here prevents every migrated
        attention leaf from becoming a new unreviewed structural writer.  The
        caller still supplies the exact reader result and selected config
        occurrence; this helper performs no interpretation or fallback.
        """
        if _facts is None:
            return
        from ...evidence.facts import EvidenceFact, SourceSpan as FactSourceSpan

        spans = tuple(dict.fromkeys(
            span for provenance in reader_result.provenance
            for span in provenance.spans))
        _facts.record_typed(EvidenceFact(
            key=key,
            owner=owner,
            value=value,
            status=status,
            completeness=completeness,
            source_spans=tuple(FactSourceSpan(
                component=span.source.component_key or "root",
                file=span.source.canonical_path,
                line=span.line,
            ) for span in spans),
            config_paths=tuple(".".join(path) for path in config_paths),
            legacy_source=reader,
            reason=reason,
        ))

    _unknown_status = "ambiguous" if _source_present else "oracle_missing"
    model_type = (_g(cfg, "model_type") or "unknown").lower()
    arch_name  = architecture_name(cfg, model_type)

    _defaults_by_path = (
        getattr(context, "class_defaults_by_path", None) or {})
    text_cfg, _text_path = _unwrap_text_with_path(
        cfg, class_defaults_by_path=_defaults_by_path)
    _selected_class_defaults = _defaults_by_path.get(_text_path)
    if not isinstance(_selected_class_defaults, dict):
        _selected_class_defaults = (
            _complete_config_from_transformers_registry(text_cfg)
            if isinstance(text_cfg, dict) else {})
    if _selected_class_defaults is text_cfg:
        _selected_class_defaults = {}
    # Preserve U2 shadow mode for an already-shaped checkpoint: class defaults
    # may support individually-tiered facts below, but may not silently author
    # arbitrary structure (for example Gemma-2's layer schedule).  The broad
    # completion tier is retained only for a declared sparse child that could
    # not otherwise be parsed as a transformer at all.
    _shape_completion_defaults = (
        _selected_class_defaults
        if not _has_transformer_shape(text_cfg) else {})
    _fact_class_defaults = (
        _selected_class_defaults
        if _text_path else (getattr(context, "class_defaults", None) or {}))
    # Nested text_config (multimodal wrapper) is fully supported — no warning needed.

    from ...evidence import config_access as _config_access

    # DBRX-style nested config dictionaries are address containers, not
    # architectural claims. Resolve their exact checkpoint locations and mark
    # the containers themselves as syntax-only; their child fields keep their
    # own independent evidence/consumption decisions below.
    def _nested_scope(key: str) -> dict:
        carries = (
            key in text_cfg if isinstance(text_cfg, dict)
            else hasattr(text_cfg, key))
        if not carries:
            # Do not manufacture an absent read on every ordinary model merely
            # because DBRX supports this optional namespace.
            return {}
        resolved = _config_access.resolve(
            text_cfg, key, (), path=_text_path)
        value = resolved.value if resolved.present else None
        if resolved.present:
            resolved.ignore(
                reason=(f"{key} is a config namespace; only independently "
                        "resolved child occurrences can author facts"))
        return value if isinstance(value, dict) else {}

    attn_cfg = _nested_scope("attn_config")
    ffn_cfg = _nested_scope("ffn_config")

    # The selector carries this address through sparse config completion.
    # Identity lookup remains a consistency check for unmodified child objects,
    # never the sole way to rediscover a copied child's checkpoint location.
    _identity_path = _wrapper_path(cfg, text_cfg)
    if _identity_path and _identity_path != _text_path:
        raise ValueError(
            "text-config selection and identity address disagree")
    context.selected_config_paths["transformer.main"] = tuple(_text_path)
    _TIERS = (
        (text_cfg, _text_path),
        (attn_cfg, (*_text_path, "attn_config")),
        (ffn_cfg, (*_text_path, "ffn_config")),
    )

    def _scoped(field):
        """REC-3 (§9.2/§9.3): text_config / attn_config / ffn_config are
        STRUCTURAL SCOPES, not aliases of one unordered object — the FIRST
        scope carrying any spelling resolves the field with its EXACT
        container path; a conflicted scope stops the search (typed ambiguity,
        never a sibling-scope coin flip); all-absent premises once against
        the text scope."""
        names = _spellings(field)
        for tier_cfg, tier_path in _TIERS:
            if tier_cfg is None:
                continue
            if any((s in tier_cfg) if isinstance(tier_cfg, dict)
                   else hasattr(tier_cfg, s) for s in names):
                return _config_access.resolve(
                    tier_cfg, field, _ALIASES.get(field, ()), path=tier_path)
        return _config_access.resolve(
            text_cfg if text_cfg is not None else {}, field,
            _ALIASES.get(field, ()), path=_text_path,
            class_defaults=_shape_completion_defaults)

    def get(field, default=None):
        """Inspect a scoped value (a branch may read and discard it).  An
        ambiguous or absent or explicit-null occurrence yields ``default`` —
        lawful ONLY because the typed ambiguity event is recorded and the
        blocking ``config_ambiguity`` net fails the model outright."""
        res = _scoped(field)
        if (res.state == "present" or res.source_kind == "class_default") \
                and res.value is not None:
            return res.value
        return default

    def consume(field, default=None, *, fact_owner="model", fact_key=None):
        """A value that FLOWS INTO a fact/geometry — consumed under the exact
        supplying occurrence with its fact owner/key (§9.3).  AMBIGUOUS stays
        unchosen: ``None`` back to the caller, the typed event recorded, the
        model blocked by the ambiguity net — never a defaulted structural
        claim (Law E).  Absent consumes are typed premises."""
        res = _scoped(field)
        if res.ambiguous:
            return None
        value = res.consume(fact_owner=fact_owner, fact_key=fact_key or field)
        return default if value is None else value

    _attention_actual_config_paths = {}

    def _consume_code_bound_path(field, exact_path, *, fact_key=None):
        """Consume only when U1 selected the exact path proven by source code."""
        res = _scoped(field)
        exact = tuple(exact_path)
        # A complete, exact source reader may name a config-class property that
        # the checkpoint omits.  In that one case the installed class default
        # is a lawful operand of the code proof (Falcon's alternate dispatch
        # selector), but it is not laundered into a checkpoint occurrence.
        if res.state == "absent" and exact and exact[-1] == field \
                and _fact_class_defaults.get(field) is not None:
            res = _config_access.resolve(
                text_cfg, field, _ALIASES.get(field, ()), path=_text_path,
                class_defaults={field: _fact_class_defaults[field]})
        selected = (
            tuple(res.selected_path.split("."))
            if isinstance(res.selected_path, str) and res.selected_path else ())
        # Modeling code reads the config CLASS's canonical property while a
        # checkpoint may use one of that property's audited input spellings
        # (GPT-BigCode: ``num_attention_heads`` versus ``n_head``).  The alias
        # resolver has already proven which spelling supplied this canonical
        # field.  Accept only that same-owner bridge: the exact source leaf
        # must be this canonical field and the parent path must be identical.
        same_property = (
            bool(exact) and exact[-1] == field
            and selected[:-1] == exact[:-1]
            and bool(selected))
        class_default = (
            res.state == "absent" and res.source_kind == "class_default"
            and res.value is not None and exact[-1] == field)
        if res.ambiguous or (
                not class_default and selected != exact and not same_property):
            return None
        decision = res.consume_decision(
            reader="adapters.transformer.parser.parse",
            fact_owner="decoder.attention",
            fact_key=fact_key or field,
            mechanism="attention_mechanism",
        )
        _attention_actual_config_paths[exact] = (
            None if class_default else selected)
        return decision.value

    def _resolve_exact_config_path(exact_path):
        """Resolve one source-proven spelling without alias search.

        Width formulas may cite several independently meaningful operands
        (for example explicit ``n_inner=None`` plus ``n_embd``).  Each must
        round-trip through the exact document occurrence before the derived
        geometry is accepted.  Resolution and consumption are deliberately
        separate: the weakest deciding origin sets ONE honest fact status
        before any obligation is emitted.
        """
        exact = tuple(exact_path)
        if not exact:
            return None
        container = cfg
        for segment in exact[:-1]:
            if isinstance(container, dict):
                if segment not in container:
                    return None
                container = container[segment]
            elif hasattr(container, segment):
                container = getattr(container, segment)
            else:
                return None
        resolution = _config_access.resolve(
            container, exact[-1], (), path=exact[:-1])
        return resolution if resolution.state == "present" else None

    num_layers   = consume("num_hidden_layers", fact_owner="model", fact_key="num_layers")
    hidden_size  = consume("hidden_size", fact_owner="model", fact_key="hidden_size")
    num_heads    = consume("num_attention_heads", fact_owner="decoder.attention", fact_key="num_heads")
    num_kv_heads = consume("num_key_value_heads", fact_owner="decoder.attention", fact_key="num_kv_heads") or num_heads
    head_dim     = consume("head_dim", fact_owner="decoder.attention", fact_key="head_dim") or (hidden_size // num_heads if (num_heads and hidden_size) else None)
    intermediate_size = consume("intermediate_size", fact_owner="decoder.ffn", fact_key="intermediate_size")
    # OLMo-style: intermediate_size derived from mlp_ratio * hidden_size.
    if not intermediate_size:
        mlp_ratio = consume("mlp_ratio", fact_owner="decoder.ffn", fact_key="mlp_ratio")
        if mlp_ratio and hidden_size:
            intermediate_size = int(hidden_size * float(mlp_ratio))
    # GPT-J/GPT-2/CodeGen: config's ``n_inner`` is None and the layer computes
    # ``4 * n_embd`` itself — read that default EXPRESSION from the source so the
    # FFN width (and thus the param count) isn't undercounted to zero.
    if not intermediate_size:
        _width_result = _code_intermediate_size(
            cfg, context, config_path=_text_path)
        if _width_result is not None and _width_result.status == "resolved":
            _width_resolutions = []
            for _path, _expected in _width_result.value.premises:
                _resolution = _resolve_exact_config_path(_path)
                # ``""`` is legal only for a caller-supplied runtime
                # PretrainedConfig: it proves the value the current model code
                # will consume, but the access event keeps origin unestablished
                # and is never promoted to checkpoint_declared.  Loader stamps
                # remain powerless.
                if _resolution is None or _resolution.value != _expected \
                        or _resolution.provenance not in {
                            "", "checkpoint_declared", "class_default"}:
                    _width_resolutions = []
                    break
                _width_resolutions.append(_resolution)
            if len(_width_resolutions) == len(_width_result.value.premises):
                _width_status = (
                    "class_default"
                    if any(item.provenance == "class_default"
                           for item in _width_resolutions)
                    else "code_and_config")
                for _resolution in _width_resolutions:
                    _resolution.consume_decision(
                        reader="decoder_ffn_intermediate_width_for_path",
                        fact_owner="decoder.ffn",
                        fact_key="intermediate_size",
                        mechanism="ffn_intermediate_width",
                        status=_width_status)
                intermediate_size = _width_result.value.value
                _note_typed_fact(
                    key="intermediate_size", owner="decoder.ffn",
                    value=intermediate_size, status=_width_status,
                    reader_result=_width_result,
                    config_paths=tuple(
                        path for path, _value
                        in _width_result.value.premises),
                    reader="decoder_ffn_intermediate_width_for_path",
                    reason=("the exact FFN output-projection input expression "
                            "evaluates from the cited config operands"),
                )
    # DBRX-style: activation lives in a nested dict like ``ffn_act_fn = {"name": "silu"}``.
    # Read the declared value for the config ledger, but project it only when
    # the exact-owner mechanism reader proves either a literal activation or
    # the exact config-dispatch path that selects this value.
    _ffn_mechanism = _ffn_mechanism_result(
        context, config_path=_text_path, config_root=cfg)
    _ffn_mechanism_value = (
        _ffn_mechanism.value
        if _ffn_mechanism is not None
        and _ffn_mechanism.status == "resolved"
        else None
    )
    _activation_res = _scoped("hidden_act")
    _activation_decision_res = _activation_res
    # Inspect the alternate declaration regardless of which spelling wins.
    # Inspection is not consumption: the value is projected only if the exact
    # source mechanism below proves that it dispatches through this path.
    _ffp_res_for_act = _scoped("feed_forward_proj")
    activation_raw = (
        None if _activation_res.ambiguous
        else _activation_res.value
    )
    if isinstance(activation_raw, dict):
        activation_raw = activation_raw.get("name")
    _nested_activation_res = None
    if activation_raw is None and ffn_cfg:
        _ffn_path = (*_text_path, "ffn_config")
        _nested_act_res = _config_access.resolve(
            ffn_cfg, "ffn_act_fn", (), path=_ffn_path)
        nested_act = (
            _nested_act_res.value if _nested_act_res.present else None)
        if isinstance(nested_act, dict):
            # The object is another namespace; the actual declared operand is
            # its exact ``name`` occurrence.
            _nested_act_res.ignore(
                reason=("ffn_act_fn is a declaration container; its exact "
                        "name child is arbitrated against source evidence"))
            _nested_activation_res = _config_access.resolve(
                nested_act, "name", (),
                path=(*_ffn_path, "ffn_act_fn"))
            if _nested_activation_res.present:
                activation_raw = _nested_activation_res.value
                _activation_decision_res = _nested_activation_res
    _activation_status, _activation_src = "config_declared", "hidden_act"
    if activation_raw is None:
        # U2 P3b: the T5-family declaration — ``feed_forward_proj`` names the
        # activation (with an optional "gated-" prefix owned by the gate
        # decision below).  Un-ignored: a positive declaration, never noise.
        # U2-R7: ONE consumption for the occurrence, into its PRIMARY decision
        # (activation); the gate decision below re-inspects the same value.
        _ffp_for_act = (
            None if _ffp_res_for_act.ambiguous
            else _ffp_res_for_act.value
        )
        if isinstance(_ffp_for_act, str) and _ffp_for_act:
            _activation_decision_res = _ffp_res_for_act
            activation_raw = _ffp_for_act.lower()
            if activation_raw.startswith("gated-"):
                activation_raw = activation_raw[len("gated-"):]
            _activation_status, _activation_src = (
                "config_declared", "feed_forward_proj")
    if activation_raw is None:
        # A hydrated class default may supply an operand, but it cannot prove
        # this exact FFN consumes that operand. The mechanism join below is
        # still mandatory.
        _cd_for_act = _fact_class_defaults
        _cd_act = next((_cd_for_act.get(s) for s in _spellings("hidden_act")
                        if _cd_for_act.get(s) is not None), None)
        _cd_src = "hidden_act"
        if not isinstance(_cd_act, str):
            _cd_ffp = _cd_for_act.get("feed_forward_proj")
            if isinstance(_cd_ffp, str) and _cd_ffp:
                _cd_act = _cd_ffp.lower()
                if _cd_act.startswith("gated-"):
                    _cd_act = _cd_act[len("gated-"):]
                _cd_src = "feed_forward_proj"
        if isinstance(_cd_act, str) and _cd_act:
            activation_raw = _cd_act
            _activation_status, _activation_src = (
                "class_default",
                f"installed config-class default ({_cd_src})")
    # U4-C: declaration is not application.  The exact mechanism must prove a
    # literal activation or the exact config path it dispatches through.
    if _ffn_mechanism_value is None:
        activation_raw = None
        _activation_status, _activation_src = _unknown_status, None
    elif _ffn_mechanism_value.activation is not None:
        activation_raw = _ffn_mechanism_value.activation
        _activation_status, _activation_src = (
            "code_proven", "decoder_ffn_mechanism_for_path")
    elif _ffn_mechanism_value.activation_config_path:
        dispatch_path = tuple(
            _ffn_mechanism_value.activation_config_path)
        # Source has already proved the exact config spelling used by this
        # activation dispatch.  Resolve THAT occurrence directly; canonical
        # alias arbitration must not choose an equal sibling spelling and then
        # reject the path the callable actually reads (Gemma configs commonly
        # carry both hidden_act and hidden_activation).
        _dispatch_parent = cfg
        for _part in dispatch_path[:-1]:
            _dispatch_parent = (
                _dispatch_parent.get(_part)
                if isinstance(_dispatch_parent, dict)
                else getattr(_dispatch_parent, _part, None))
            if _dispatch_parent is None:
                break
        _dispatch_res = (
            _config_access.resolve(
                _dispatch_parent, dispatch_path[-1], (),
                path=dispatch_path[:-1])
            if _dispatch_parent is not None else None)
        _dispatch_value = (
            _dispatch_res.value
            if _dispatch_res is not None and _dispatch_res.present
            and not _dispatch_res.ambiguous else None)
        if isinstance(_dispatch_value, dict):
            _dispatch_value = _dispatch_value.get("name")
        if isinstance(_dispatch_value, str):
            _dispatch_value = _dispatch_value.lower()
            if _dispatch_value.startswith("gated-"):
                _dispatch_value = _dispatch_value[len("gated-"):]
        if not isinstance(_dispatch_value, str) or not _dispatch_value:
            activation_raw = None
            _activation_status, _activation_src = _unknown_status, None
        else:
            activation_raw = _dispatch_value
            _activation_decision_res = _dispatch_res
            _activation_status = (
                "class_default"
                if _dispatch_res.provenance == "class_default"
                else "config_declared")
            # Consume only the exact occurrence the source dispatch names.
            # Equal aliases remain visible inspections, never alternate proof.
            _dispatch_res.consume_decision(
                mechanism="ffn_activation",
                fact_owner="decoder.ffn",
                fact_key="activation",
                reader="adapters.transformer.parser.parse",
                status=_activation_status,
            )
            if _activation_status == "config_declared":
                _activation_status = "code_and_config"
            _activation_src = (
                "decoder_ffn_mechanism_for_path:"
                + ".".join(dispatch_path))
    else:
        activation_raw = None
        _activation_status, _activation_src = _unknown_status, None

    activation_defaulted = activation_raw is None
    activation = activation_raw.lower() if isinstance(activation_raw, str) else None
    if activation_defaulted:
        _activation_status, _activation_src = _unknown_status, None
    _note_fact("decoder.ffn", "activation", activation,
               _activation_status, _activation_src)
    if _nested_activation_res is not None \
            and _nested_activation_res.present \
            and not (
                _activation_status == "code_and_config"
                and _activation_decision_res is _nested_activation_res):
        _nested_activation_res.ignore(
            reason=("the checkpoint declares an activation name, but exact "
                    "FFN source did not prove this occurrence selects the "
                    "executed activation"))
    sliding_window = consume("sliding_window", fact_owner="decoder.attention", fact_key="sliding_window")
    # ---- Sliding-window enable toggle (Qwen2/2.5/3) ----
    # A config may declare a window size but turn SWA *off* (use_sliding_window
    # = False); honor that, otherwise we'd draw sliding attention on what is
    # really a full-attention model.  When absent (Mistral), the window applies.
    # U2-R7: these reads touch the (possibly nested) text config — NAME the
    # object so the ledger records an exact located path instead of an honest
    # bare leaf (wrapper_path is () when text IS the root; the container is
    # obj-qualified either way, so host reads are never mislabeled).
    with _config_access.config_container(_text_path, obj=text_cfg):
        # U2-R7: both flow into the sliding-window schedule (the enable toggle
        # and the bottom-full split) — consumed like ``sliding_window`` above.
        use_sliding_window = consume("use_sliding_window",
                                     fact_owner="decoder.attention",
                                     fact_key="use_sliding_window")
        max_window_layers  = consume("max_window_layers",
                                     fact_owner="decoder.attention",
                                     fact_key="first_full_layers")
    if use_sliding_window is False:
        sliding_window = None
        max_window_layers = None
    with _config_access.config_container(_text_path, obj=text_cfg):
        # U2-R7: the canonical per-layer mask/kind schedule — consumed into the
        # attention schedule fact it becomes.
        layer_types  = consume("layer_types", fact_owner="decoder.attention",
                               fact_key="layer_types") or []
        full_attention_interval = _g(text_cfg, "full_attention_interval") or 0
    if not layer_types and full_attention_interval and num_layers:
        layer_types = [
            "linear_attention" if (i + 1) % int(full_attention_interval) else "full_attention"
            for i in range(num_layers or 0)
        ]
    # Resolved through aliases so dialect spellings (DeepSeek-V4 ``compress_rates``)
    # are picked up — see everchanging/aliases.yaml.
    compress_ratios = get("compress_ratios") or []
    if not layer_types and compress_ratios:
        layer_types = _layer_types_from_compress_ratios(compress_ratios, num_layers)
    # U3: the six per-layer TYPE-SCHEDULE spellings (attn_type_list / block_types
    # / attention_types / dense_attention_every_n_layers) normalize into the SAME
    # layer_types list the working schedules flow through — the config channel,
    # consulted ONLY when no canonical layer_types list exists.  This proves
    # placement of declared opaque mixers and full-attention routes; it does
    # NOT classify the full route as MHA/GQA when modeling source is missing.
    _schedule_source = None
    if not layer_types:
        _sched, _schedule_source = _normalize_layer_schedule(
            text_cfg, num_layers, sliding_window)
        layer_types = _sched or []
        if _schedule_source:
            _note_fact("decoder.layer", "layer_schedule",
                       f"{len(set(layer_types))} layer types over {num_layers}",
                       "config_declared", _schedule_source)
    # Granite-style declared SCALE family: a constant multiplier on each
    # sublayer's residual contribution (drawn as a × connector with its
    # constant operand), plus embedding/attention/logits scales (card facts).
    # An undrawn SPEECH stack (Qwen-Omni talker + token2wav) is a stated
    # omission, never a silent one.
    if _g(cfg, "talker_config") is not None:
        warnings.append("Speech-generation stack (talker + token2wav vocoder) not "
                        "drawn — the diagram shows the thinker (LM).")
    # Declared-but-undrawn speech components (VITS flows / duration predictor /
    # HiFiGAN ladder; SpeechT5 pre/post-nets) — STATED omissions, never silent.
    _undrawn = _load_composite_slots().get("undrawn_component_fields") or {}
    _undrawn_labels = sorted({label for field, label in _undrawn.items()
                              if _g(cfg, field) not in (None, False)})
    if _undrawn_labels:
        warnings.append(
            "Config declares speech-synthesis components not drawn yet — "
            + ", ".join(_undrawn_labels)
            + " — only the main transformer stack is drawn (audio plan U-E).")
    # A FLAT seq2seq config (SpeechT5: encoder_layers + decoder_layers, no
    # composite slots) is drawn as ONE stack today — say which half.
    # Soumil's final vet: is_encoder_decoder is ARCHITECTURE (drawn seq2seq
    # half, mask causality, cross-attn schedule) — consumed ONCE here into
    # the mask fact; the two later deciding sites reuse this value.
    _ied_res = _config_access.resolve(cfg, "is_encoder_decoder", ())
    _is_enc_dec = bool(
        None if _ied_res.ambiguous else _ied_res.consume_decision(
            mechanism="decoderness", fact_owner="decoder.attention",
            fact_key="mask",
            reader="adapters.transformer.parser.parse").value)
    if (_is_enc_dec
            and _composite_encoder_model_type(cfg) is None
            and _g(cfg, "encoder_layers") and _g(cfg, "decoder_layers")):
        warnings.append(
            f"Flat seq2seq config: the drawn stack is the encoder half; the "
            f"{_g(cfg, 'decoder_layers')}-layer decoder (and any task "
            "pre/post-nets) is not drawn (audio plan U-E).")
    # A codec-role composite slot (MusicGen's audio_encoder/EnCodec) is a STATED
    # omission until the codec tower lands (audio plan U-C) — never silent.
    for _slot_key, _slot_role in (_load_composite_slots().get("slots") or {}).items():
        if _slot_role != "codec":
            continue
        _codec_sub = _g(cfg, _slot_key)
        if not isinstance(_codec_sub, dict) and hasattr(_codec_sub, "to_dict"):
            _codec_sub = _codec_sub.to_dict()
        if isinstance(_codec_sub, dict) and _codec_sub.get("model_type"):
            warnings.append(
                f"Audio codec ({_codec_sub.get('model_type')}) not drawn — it "
                "tokenizes/decodes the audio-token streams this decoder "
                "generates (waveform ↔ codebook tokens).")
    # A residual multiplier is an operand, not proof that this exact layer
    # forward applies it.  U7 may promote it after binding the config path to
    # the resolved residual operation; U4-D keeps the declaration inspected.
    _scoped("residual_multiplier")
    residual_multiplier = None
    get("embedding_multiplier")
    attention_multiplier = get("attention_multiplier")
    query_pre_attn_scalar = consume("query_pre_attn_scalar",
                                    fact_owner="decoder.attention",
                                    fact_key="scores_scale")
    get("logits_scaling")
    # U2-R7: the helper reads eps spellings off the (possibly nested) text
    # config and has no path of its own — the CALLER names the object, and
    # names the fact target the eps evidence flows into (the norm-kind
    # decision recorded under decoder.layer below); the encoder panel's
    # post-parse advisory call passes no target and keeps inspected reads,
    # so one occurrence is never consumed twice.
    with _config_access.config_container(_text_path, obj=text_cfg):
        _norm_kind_ev, _norm_kind_prov = _norm_kind_evidence_src(
            text_cfg, get("norm_type"), context,
            eps_fact=("decoder.layer", "norm_kind_eps"),
            config_path=_text_path)
    # U2 default-kill: no channel → typed "unknown" (generic Normalization
    # label + honest card prose), never a silent modern-LM rmsnorm.
    norm_kind    = _norm_kind_ev or "unknown"
    _note_fact("decoder.layer", "norm_kind", norm_kind,
               *( _norm_kind_prov if _norm_kind_prov else (_unknown_status, None)))
    # Norm placement (pre / post / double-sandwich) is STRUCTURE and carries no
    # config flag — so it is READ FROM THE LAYER'S forward() dataflow (code ->
    # structure), the general replacement for the model_type identity table.
    _cell_topology_result_value = _cell_topology_result(
        context, config_path=_text_path, config_root=cfg)
    _cell_topology = (
        _cell_topology_result_value.value
        if _cell_topology_result_value is not None
        and _cell_topology_result_value.status == "resolved" else None
    )
    # FFN gating READ FROM THE MLP's forward() (gate_mul present?) — code wins;
    # a gate-family activation string is the config-derived second channel; NO
    # channel ⇒ typed unknown (None) drawn as the honest undeclared-FFN block,
    # never derived from norm_kind (the census cascade, killed in U2).
    _code_gated = (
        _ffn_mechanism_value.gated
        if _ffn_mechanism_value is not None else None
    )
    # Gate declarations remain visible to the config ledger, but they do not
    # prove that this exact FFN has a third projection.  The old path consumed
    # ``is_gated_act`` / ``feed_forward_proj`` as architecture on sight.  U4-C
    # deliberately leaves the declaration inspected until the exact source
    # mechanism proves the projection topology.
    from ...evidence.ffn_mechanism import ConfigSelectedFFNMechanism
    _ffn_selected_by_config = isinstance(
        _ffn_mechanism_value, ConfigSelectedFFNMechanism)
    if not _ffn_selected_by_config:
        _scoped("is_gated_act")
    # U4-C: a config/class declaration or activation spelling may be useful
    # evidence only after exact source binds it to this FFN.  The ordinary
    # mechanism reader is that binding.  Without it, gate topology is unknown.
    ffn_gated = bool(_code_gated) if _code_gated is not None else None
    # ``decoder.ffn.gated`` names the ordinary/shared FFN mechanism.  A routed
    # expert proof must never be laundered into this owner, and a routed-only
    # block has no ordinary gate fact to record.  The exact reader result
    # remains in ``context.reader_results`` as the typed abstention.
    if ffn_gated is not None:
        _note_fact("decoder.ffn", "gated", ffn_gated,
                   ("code_and_config" if _ffn_selected_by_config
                    else "code_proven"),
                   source=(
                       "decoder_ffn_mechanism_for_path:"
                       + ".".join(
                           _ffn_mechanism_value.selector_config_path)
                       if _ffn_selected_by_config else
                       "decoder_ffn_mechanism_for_path"))
    _code_storage_mode = (
        _ffn_mechanism_value.projection_mode
        if _ffn_mechanism_value is not None else None
    )
    _code_attention_storage = _code_attention_storage_mode(
        text_cfg, context, config_path=_text_path)
    _code_position_evidence = _code_position(cfg, context)
    # Projection storage is mechanism-scoped too.  Do not manufacture an
    # ordinary ``split`` fact when only a routed-expert mechanism exists.
    if _code_storage_mode is not None:
        _note_fact("decoder.ffn", "projection_mode",
                   _code_storage_mode,
                   ("code_and_config" if _ffn_selected_by_config
                    else "code_proven"),
                   source=(
                       "decoder_ffn_mechanism_for_path:"
                       + ".".join(
                           _ffn_mechanism_value.selector_config_path)
                       if _ffn_selected_by_config else
                       "decoder_ffn_mechanism_for_path"))
    _note_fact(
        "decoder.attention", "projection_mode",
        _code_attention_storage,
        "code_proven" if _code_attention_storage is not None else _unknown_status,
        source=("decoder_attention_projection_storage_for_path"
                if _code_attention_storage is not None else None),
    )
    _output_projection_result = _attention_output_projection_result(
        context, config_path=_text_path)
    attention_output_projection = (
        True if _output_projection_result is not None
        and _output_projection_result.status == "resolved" else None)
    if attention_output_projection is True:
        _note_typed_fact(
            key="output_projection",
            owner="decoder.attention",
            value=True,
            status="code_proven",
            reader_result=_output_projection_result,
            config_paths=(),
            reader="decoder_attention_output_projection_for_path",
            reason=(
                "the selected attention-value terminal reaches one unique "
                "exact Linear construction and call"),
        )
    # Placement is an owner-bound wiring fact.  Source presence is not proof
    # of pre-norm: an abstaining reader stays unknown on every surface.
    norm_placement = (
        _cell_topology.norm_placement if _cell_topology is not None else None)
    _norm_placement_config_paths = (
        _cell_topology.norm_config_paths
        if _cell_topology is not None else ())
    _norm_placement_status = (
        "code_and_config"
        if _norm_placement_config_paths else "code_proven")
    if norm_placement is not None:
        _note_typed_fact(
            key="norm_placement", owner="decoder.layer",
            value=norm_placement, status=_norm_placement_status,
            reader_result=_cell_topology_result_value,
            config_paths=_norm_placement_config_paths,
            reader="decoder_cell_topology_for_path",
            reason=(
                "exact attention/FFN norm boundaries and residual equations "
                "prove the cell placement"),
        )
    else:
        _note_fact(
            "decoder.layer", "norm_placement", "unknown", _unknown_status)
    if not norm_placement:
        norm_placement = "unknown"
    # Position application is a mechanism fact. A declared theta/scaling value
    # remains visible to the config ledger, but cannot create Q/K rotation.
    _position_mechanisms = list(_code_position_evidence.mechanisms)
    # U2-R7: the rope container DICT occurrence itself feeds the uses_rope
    # decision — consumed once here; every later reader reuses this value.
    _declared_scaling = consume("rope_scaling",
                                fact_owner="decoder.attention", fact_key="rope")
    if _code_position_evidence.status == "proven":
        uses_rope = "rope" in _code_position_evidence.kinds
    else:
        uses_rope = None
        if _code_position_evidence.status == "oracle_missing":
            warnings.append(
                "Modeling source is unavailable; the positional scheme remains unknown."
            )
        else:
            warnings.append(
                "Modeling source is present but the configured positional scheme is unresolved."
            )

    # REC-3 (§9.6, Law D): a CONFLICTED field is not a MISSING field — the
    # warning names the true condition; the structured record + blocking
    # ``config_ambiguity`` net carry the exact rival occurrences.
    def _ambiguous_here(field: str) -> bool:
        led = _config_access.active_ledger()
        return led is not None and any(
            e.intent == "ambiguous" and e.canonical == field for e in led.events)

    if not num_layers:
        warnings.append(
            "Config declares conflicting num_hidden_layers values — layer list "
            "withheld until the checkpoint is unambiguous."
            if _ambiguous_here("num_hidden_layers") else
            "Config missing num_hidden_layers (and aliases) — layer list will be empty.")
    if not hidden_size:
        warnings.append(
            "Config declares conflicting hidden_size values — geometry withheld "
            "until the checkpoint is unambiguous."
            if _ambiguous_here("hidden_size") else
            "Config missing hidden_size (and aliases) — geometry will be incomplete.")

    # ---- Per-layer dual KV (Gemma 4 sliding vs global; might appear elsewhere) ----
    num_kv_global   = _g(text_cfg, "num_global_key_value_heads") or num_kv_heads
    head_dim_global = _g(text_cfg, "global_head_dim") or head_dim

    # ---- Attention shape ----
    # U2-R7: the five MLA geometry fields flow straight into the attention
    # spec/param math — consumed under their canonical names.
    q_lora_rank  = consume("q_lora_rank", fact_owner="decoder.attention",
                           fact_key="q_lora_rank")
    kv_lora_rank = consume("kv_lora_rank", fact_owner="decoder.attention",
                           fact_key="kv_lora_rank")
    is_mla       = bool(kv_lora_rank)
    # MLA decoupled head geometry — Q/K split into nope + rope, V its own width
    # (DeepSeek/Kimi). Needed for an accurate MLA parameter count.
    qk_nope_head_dim = consume("qk_nope_head_dim",
                               fact_owner="decoder.attention",
                               fact_key="qk_nope_head_dim")
    qk_rope_head_dim = consume("qk_rope_head_dim",
                               fact_owner="decoder.attention",
                               fact_key="qk_rope_head_dim")
    v_head_dim_cfg   = consume("v_head_dim", fact_owner="decoder.attention",
                               fact_key="v_head_dim")
    _attention_mechanism_evidence = _attention_mechanism_result(
        context, config_path=_text_path)
    _bound_attention = None
    if _attention_mechanism_evidence is not None \
            and _attention_mechanism_evidence.status == "resolved":
        from ...evidence.attention import (
            AttentionHeadBinding,
            EquivalentDispatchMultiQueryBinding,
            LatentAttentionBinding,
            MultiQueryAttentionBinding,
            bind_attention_mechanism,
        )
        _binding = _attention_mechanism_evidence.value
        _bound_values = {}
        if isinstance(_binding, AttentionHeadBinding):
            _bound_values[_binding.query_heads_path] = \
                _consume_code_bound_path(
                    "num_attention_heads", _binding.query_heads_path,
                    fact_key="num_heads")
            if _binding.protocol == "grouped_kv":
                _bound_values[_binding.key_value_heads_path] = \
                    _consume_code_bound_path(
                        "num_key_value_heads",
                        _binding.key_value_heads_path,
                        fact_key="num_kv_heads")
        elif isinstance(_binding, LatentAttentionBinding):
            for _field, _path, _fact_key in (
                ("num_attention_heads", _binding.num_heads_path, "num_heads"),
                ("kv_lora_rank", _binding.kv_lora_rank_path, "kv_lora_rank"),
                ("qk_rope_head_dim", _binding.qk_rope_head_dim_path,
                 "qk_rope_head_dim"),
                ("qk_nope_head_dim", _binding.qk_nope_head_dim_path,
                 "qk_nope_head_dim"),
                ("v_head_dim", _binding.value_head_dim_path, "v_head_dim"),
            ):
                _bound_values[_path] = _consume_code_bound_path(
                    _field, _path, fact_key=_fact_key)
        elif isinstance(
                _binding,
                (MultiQueryAttentionBinding,
                 EquivalentDispatchMultiQueryBinding)):
            _bound_values[_binding.num_heads_path] = \
                _consume_code_bound_path(
                    "num_attention_heads", _binding.num_heads_path,
                    fact_key="num_heads")
            _bound_values[_binding.selector_path] = \
                _consume_code_bound_path(
                    "multi_query", _binding.selector_path,
                    fact_key="mechanism")
            if isinstance(_binding, EquivalentDispatchMultiQueryBinding):
                _bound_values[_binding.alternate_architecture_path] = \
                    _consume_code_bound_path(
                        "new_decoder_architecture",
                        _binding.alternate_architecture_path,
                        fact_key="mechanism")
        _bound_attention = bind_attention_mechanism(
            _binding, _bound_values)
    if _bound_attention is not None:
        num_heads = _bound_attention.num_heads
        num_kv_heads = _bound_attention.num_kv_heads
        is_mla = _bound_attention.kind == "mla"
        if not is_mla:
            q_lora_rank = kv_lora_rank = None
            qk_nope_head_dim = qk_rope_head_dim = v_head_dim_cfg = None
        _note_bound_attention_fact(
            _bound_attention, _attention_mechanism_evidence,
            _attention_actual_config_paths)
        _output_gate = getattr(_binding, "output_gate", None)
        if _output_gate is not None:
            _note_typed_fact(
                key="output_gate",
                owner="decoder.attention",
                value=_output_gate.activation,
                status="code_proven",
                reader_result=_attention_mechanism_evidence,
                config_paths=(),
                reader="decoder_attention_mechanism_for_path",
                reason=(
                    "the exact query projection is split into query and gate "
                    "lanes; the sibling lane passes through sigmoid and "
                    "multiplies the attention result before the exact output "
                    "projection"),
            )
            attn_output_gate = _output_gate.activation
        else:
            attn_output_gate = None
    else:
        is_mla = False
        attn_output_gate = None
        _note_fact(
            "decoder.attention", "mechanism", None,
            _unknown_status, None)
    # Hybrid linear-recurrent geometry is a code-and-config join.  The reader
    # assigns all five roles from split/reshape/repeat/Conv1d/recurrent uses;
    # the familiar config spellings cannot populate a detailed mixer alone.
    _gated_delta_evidence = _gated_delta_geometry_result(
        context, config_path=_text_path)
    _bound_gated_delta_geometry = None
    if _gated_delta_evidence is not None \
            and _gated_delta_evidence.status == "resolved":
        _geometry = _gated_delta_evidence.value
        _role_paths = (
            ("linear_num_key_heads", _geometry.key_heads_path),
            ("linear_num_value_heads", _geometry.value_heads_path),
            ("linear_key_head_dim", _geometry.key_head_dim_path),
            ("linear_value_head_dim", _geometry.value_head_dim_path),
            ("linear_conv_kernel_dim", _geometry.conv_kernel_path),
        )
        _role_values = tuple(
            _consume_code_bound_path(field, path,
                                     fact_key="gated_delta_geometry")
            for field, path in _role_paths)
        if all(isinstance(value, int) and not isinstance(value, bool)
               and value > 0 for value in _role_values):
            (_linear_k_heads, _linear_v_heads,
             _linear_k_dim, _linear_v_dim,
             _linear_kernel) = _role_values
            if _linear_v_heads >= _linear_k_heads \
                    and _linear_v_heads % _linear_k_heads == 0:
                _bound_gated_delta_geometry = _role_values
                _note_typed_fact(
                    key="gated_delta_geometry",
                    owner="decoder.attention",
                    value=_role_values,
                    status="code_and_config",
                    reader_result=_gated_delta_evidence,
                    config_paths=tuple(
                        selected for _field, path in _role_paths
                        if (selected := _attention_actual_config_paths.get(path))
                        is not None),
                    reader="decoder_gated_delta_geometry_for_path",
                    reason=(
                        "exact split widths, reshape widths, Q/K repeat ratio, "
                        "Conv1d kernel and two recurrent terminals bind all "
                        "five geometry values"),
                    completeness="presence_only",
                )
    if _bound_gated_delta_geometry is None:
        # Inspected-only compatibility views keep declarations visible to the
        # access audit while withholding architecture on reader abstention.
        for _field in (
                "linear_num_key_heads", "linear_num_value_heads",
                "linear_key_head_dim", "linear_value_head_dim",
                "linear_conv_kernel_dim"):
            _g(text_cfg, _field)
        linear_num_k_heads = linear_num_v_heads = None
        linear_k_head_dim = linear_v_head_dim = None
        linear_conv_kernel = None
    else:
        (linear_num_k_heads, linear_num_v_heads,
         linear_k_head_dim, linear_v_head_dim,
         linear_conv_kernel) = _bound_gated_delta_geometry
    # These declarations remain visible to the config-access audit, but they
    # cannot author the mechanism.  Some implementations apply the proven
    # sigmoid gate even when a familiar flag is false; the exact forward chain
    # above is the authority.
    _g(text_cfg, "attn_output_gate")
    _g(text_cfg, "output_gate_type")
    # Determine if the stack mixes sliding + full layers — affects mask labeling
    # (a full layer in a sliding stack is labeled "global", not "causal").
    sliding_window_pattern = _g(text_cfg, "sliding_window_pattern") or 0
    # Qwen splits the stack: the bottom ``max_window_layers`` use full attention
    # and the rest slide — so a partial split also makes this a mixed stack.
    has_max_window_split = bool(
        sliding_window and max_window_layers and 0 < max_window_layers < num_layers
    )
    has_sliding_in_stack = (
        any(_is_sliding_label(lt) for lt in layer_types)
        or bool(sliding_window_pattern and sliding_window)
        or has_max_window_split
    )

    # ---- Position encoding ----
    no_rope_interval     = _g(text_cfg, "no_rope_layer_interval") or 0
    no_rope_list_declared = _g(text_cfg, "no_rope_layers")
    # These values announce a per-layer selector but do not prove its source
    # semantics. Until U8 binds the selector, projecting model-wide RoPE onto
    # every layer would be as wrong as fabricating NoPE on selected layers.
    position_schedule_unresolved = bool(
        no_rope_interval or isinstance(no_rope_list_declared, (list, tuple))
    )
    _g(text_cfg, "alibi")  # config ownership; source proves how the switch is used
    rotary_pct           = _g(text_cfg, "rotary_pct")
    rotary_dim           = _g(text_cfg, "rotary_dim")
    # U2-R7: consumed into the derived rope_dim (decoder.attention.rope_dim).
    partial_rotary_fac   = consume("partial_rotary_factor",
                                   fact_owner="decoder.attention",
                                   fact_key="rope_dim")
    # Multimodal RoPE (Qwen2-VL / Qwen3-VL): rope_scaling.mrope_section splits the
    # rotary dims across (temporal, height, width) position axes — a Tier-3 property.
    # U2.2a: the container names the spelling the DOCUMENT supplies — chosen by
    # literal presence, never by an alias-resolving read (``_g`` answers a
    # request for ``rope_parameters`` with a document's ``rope_scaling`` value,
    # which would assert a path that exists nowhere).  Absent both, no container
    # is declared and the enclosed reads stay honestly inexact.
    _rope_container      = _config_access.present_spelling(
        text_cfg, ("rope_parameters", "rope_scaling"))
    # U2-R7: the rope dict occurrence was already resolved and CONSUMED once at
    # the position-scheme decision above — REUSE that value (one resolve per
    # occurrence; a second located read would re-record the same occurrence).
    _rope_scaling        = _declared_scaling or {}
    _rope_path           = (*_text_path, _rope_container) if _rope_container else ()
    # The modern transformers rope dialect NESTS the partial factor inside the
    # rope-parameters dict (GPT-NeoX's legacy top-level ``rotary_pct`` no longer
    # exists on the config class) — the SAME fact in two spellings.
    #
    # COR-4 (§9) alias law: rival spellings are READ and COMPARED — equal ones
    # are redundant evidence, disagreeing ones are structured ambiguity that
    # authors nothing.  Never a silent first-match: guarding the nested read
    # behind ``top-level is None`` left the rival spelling unread whenever both
    # were declared, so whichever the parser happened to reach won unexamined
    # and a real disagreement could not be seen.
    _nested_prf = (_rope_scaling.get("partial_rotary_factor")
                   if isinstance(_rope_scaling, dict) else None)
    if _nested_prf is not None:
        if partial_rotary_fac is None:
            # U2-R7: the nested spelling is the SUPPLYING occurrence — consumed
            # at its exact nested path into the same derived rope_dim target.
            # The rival-spelling comparison below is untouched: this branch is
            # exactly the old ``top-level is None`` assignment.
            with _config_access.config_container(_rope_path, obj=_rope_scaling):
                partial_rotary_fac = _config_access.resolve(
                    _rope_scaling, "partial_rotary_factor", (),
                    path=_rope_path).consume_decision(
                        mechanism="rope_dim",
                        fact_owner="decoder.attention", fact_key="rope_dim",
                        reader="adapters.transformer.parser.parse").value
        elif partial_rotary_fac == _nested_prf:
            # Redundant equal rival — recorded as the inspection it is.
            with _config_access.config_container(_rope_path, obj=_rope_scaling):
                debug.note_access("partial_rotary_factor",
                                  source_obj=_rope_scaling)
        else:
            with _config_access.config_container(_rope_path, obj=_rope_scaling):
                debug.note_access("partial_rotary_factor",
                                  source_obj=_rope_scaling)
            _config_access.emit(
                "partial_rotary_factor", intent="ambiguous", present=True,
                config_path=".".join((*_rope_path, "partial_rotary_factor")),
                reason=(
                    "rival spellings of one fact disagree: "
                    f"{'.'.join((*_text_path, 'partial_rotary_factor'))}="
                    f"{partial_rotary_fac!r} vs "
                    f"{'.'.join((*_rope_path, 'partial_rotary_factor'))}="
                    f"{_nested_prf!r}"))
            partial_rotary_fac = None
    rope_dim_value       = _rope_dim(rotary_pct, rotary_dim, partial_rotary_fac, head_dim)
    if rope_dim_value is None:
        # Config silent on the fraction — the CODE may still state it (ChatGLM
        # constructs ``RotaryEmbedding(rotary_dim // 2)``; the halving exists
        # nowhere in config).  A full-width value is not "partial" — drop it.
        _code_rd = _code_rope_dim(text_cfg, context)
        if _code_rd and head_dim and 0 < _code_rd < head_dim:
            rope_dim_value = _code_rd
    mrope_section        = _rope_scaling.get("mrope_section") if isinstance(_rope_scaling, dict) else None
    if mrope_section is not None:
        # U2-R7: the nested mrope split flows into the attention position fact
        # — consumed at its exact nested occurrence (ONE resolve; the guard
        # above is a raw membership probe, not a ledger read).
        with _config_access.config_container(_rope_path, obj=_rope_scaling):
            mrope_section = _config_access.resolve(
                _rope_scaling, "mrope_section", (),
                path=_rope_path).consume_decision(
                    mechanism="mrope",
                    fact_owner="decoder.attention", fact_key="mrope_section",
                    reader="adapters.transformer.parser.parse").value

    # ---- QK-Norm ----
    # Config is consulted only for the exact gate paths named by the source
    # reader below.  A familiar qk_norm/use_qk_norm spelling on its own is not
    # an architectural input and must not create an audit occurrence.
    use_qk_norm = None
    _qk_code_result = _qk_norm_result(context, config_path=_text_path)
    _qk_code = (
        _qk_code_result.value
        if _qk_code_result is not None
        and _qk_code_result.status == "resolved" else None)
    # A failed mechanism reader may still prove that one exact config
    # occurrence controls distinct Q/K transformations.  Bind that occurrence
    # to the unresolved claim, but do not consume its value and do not assert
    # Q/K normalization until the child primitive is itself proven.  This is
    # the honest boundary for composite/repeated transforms whose execution is
    # outside the current ProgramIndex contract.
    if _qk_code_result is not None \
            and _qk_code_result.status != "resolved":
        _unresolved_qk_paths = tuple(dict.fromkeys(
            path
            for provenance in _qk_code_result.provenance
            for path in provenance.config_paths))
        for _bound_path in _unresolved_qk_paths:
            _prefix = tuple(_text_path)
            if len(_bound_path) != len(_prefix) + 1 \
                    or _bound_path[:len(_prefix)] != _prefix:
                continue
            _resolution = _config_access.resolve(
                text_cfg, _bound_path[-1], (), path=_text_path)
            if _resolution.ambiguous or not _resolution.present \
                    or _resolution.selected_path != ".".join(_bound_path):
                continue
            _resolution.bind(
                "decoder_qk_norm_evidence_for_path:unclassified_qk_transform",
                fact_owner="decoder.attention", fact_key="qk_norm")
    qk_norm_layers = _resolve_qk_norm_layers(
        _qk_code,
        text_cfg, num_layers,
        context=context, config_path=_text_path)
    # U6 owns the uniform attention fact.  A heterogeneous per-layer schedule
    # is deliberately not collapsed into one model-wide boolean; U8 will emit
    # occurrence-qualified schedule facts.  This preserves the exact source
    # reader's strength instead of laundering a mixed stack through ``any``.
    if _qk_code_result is not None and _qk_code_result.status == "resolved" \
            and qk_norm_layers \
            and all(value is not None for value in qk_norm_layers) \
            and len(set(qk_norm_layers)) == 1:
        _qk_value = bool(qk_norm_layers[0])
        _qk_status = (
            "code_proven" if _qk_code.present is True
            else "code_and_config")
        _note_typed_fact(
            key="qk_norm",
            owner="decoder.attention",
            value=_qk_value,
            status=_qk_status,
            reader_result=_qk_code_result,
            config_paths=tuple(atom.config_path for atom in _qk_code.gate),
            reader="decoder_qk_norm_evidence_for_path",
            reason=(
                "two exact normalization applications independently feed the "
                "selected attention score's Q and K lanes"
                + (" under the exact selected config gate"
                   if _qk_code.gate else "")),
        )

    # ---- Bias terms on the Q/K/V/O projections (Qwen2, GPT-2, Phi, ...) ----
    # CODE construction is authoritative. QKV alone cannot certify an output
    # projection. Exact disagreeing construction expressions are retained as
    # a mixed pattern; config-bound terms are evaluated only after the reader
    # names their exact paths. U2 default-kill:
    # when BOTH are silent the bias is a typed UNKNOWN (None) — never a
    # silent False indistinguishable from proven-False.
    # U2-R7: consumed into the bias fact/spec — every alias spelling
    # (use_qkv_bias, add_qkv_bias, ...) resolves through this ONE read.
    # A config occurrence is read only after the source reader binds it to an
    # exact ``Linear(..., bias=config.<field>)`` expression.  An unrelated
    # declaration is not even structural input and cannot create audit debt.
    _declared_bias_resolution = None
    _code_bias = _code_attention_bias(
        text_cfg, context, config_path=_text_path)
    from ...evidence.projection_bias import ProjectionBiasPatternEvidence
    if _code_bias is not None and isinstance(
            _code_bias.value, ProjectionBiasPatternEvidence):
        # A latent path may deliberately mix literal bias=False stages with
        # stages bound to one exact config occurrence.  Resolve the expressions
        # only after source has identified every exact construction; the raw
        # config value never authors the projection layout by itself.
        decisions = {}
        for _bound_path in _code_bias.value.config_paths:
            _prefix = tuple(_text_path)
            if len(_bound_path) != len(_prefix) + 1 \
                    or _bound_path[:len(_prefix)] != _prefix:
                continue
            _resolution = _config_access.resolve(
                text_cfg, _bound_path[-1], (), path=_text_path)
            if _resolution.ambiguous or not _resolution.present \
                    or _resolution.selected_path != ".".join(_bound_path) \
                    or not isinstance(_resolution.value, bool):
                continue
            _resolution.bind(
                "decoder_attention_bias_for_path",
                fact_owner="decoder.attention", fact_key="bias")
            decisions[_bound_path] = _resolution.consume_decision(
                mechanism="projection_bias",
                fact_owner="decoder.attention", fact_key="bias",
                reader="decoder_attention_bias_for_path",
                status="code_and_config")
        if len(decisions) == len(_code_bias.value.config_paths):
            _bias_values = {
                term.value if term.config_path is None
                else decisions[term.config_path].value
                for term in _code_bias.value.terms
            }
            use_attention_bias = (
                next(iter(_bias_values))
                if len(_bias_values) == 1 else "mixed")
            _bias_status = (
                "code_and_config"
                if _code_bias.value.config_paths else "code_proven")
            _note_typed_fact(
                key="bias",
                owner="decoder.attention",
                value=use_attention_bias,
                status=_bias_status,
                reader_result=_code_bias,
                config_paths=_code_bias.value.config_paths,
                reader="decoder_attention_bias_for_path",
                reason=(
                    "every exact attention affine construction contributes its "
                    "literal or config-bound bias expression"
                    + ("; the selected checkpoint makes them uniform"
                       if len(_bias_values) == 1
                       else "; the selected checkpoint produces a mixed layout")),
            )
        else:
            use_attention_bias = None
            _note_fact("decoder.attention", "bias", None,
                       _unknown_status, None)
    elif _code_bias is not None and _code_bias.value.value is not None:
        use_attention_bias = _code_bias.value.value
        _note_fact("decoder.attention", "bias", use_attention_bias,
                   "code_proven", "decoder_attention_bias_for_path")
    elif _code_bias is not None and _code_bias.value.config_path is not None:
        _bound_path = _code_bias.value.config_path
        _prefix = tuple(_text_path)
        if len(_bound_path) == len(_prefix) + 1 \
                and _bound_path[:len(_prefix)] == _prefix:
            _bound_leaf = _bound_path[-1]
            _declared_bias_resolution = _config_access.resolve(
                text_cfg, _bound_leaf, (), path=_text_path)
        if _declared_bias_resolution is not None \
                and not _declared_bias_resolution.ambiguous \
                and _declared_bias_resolution.present \
                and _declared_bias_resolution.selected_path \
                == ".".join(_bound_path) \
                and isinstance(_declared_bias_resolution.value, bool):
            _declared_bias_resolution.bind(
                "decoder_attention_bias_for_path",
                fact_owner="decoder.attention", fact_key="bias")
            _bias_decision = _declared_bias_resolution.consume_decision(
                mechanism="projection_bias",
                fact_owner="decoder.attention", fact_key="bias",
                reader="decoder_attention_bias_for_path",
                status="code_and_config")
            use_attention_bias = _bias_decision.value
            _note_typed_fact(
                key="bias",
                owner="decoder.attention",
                value=use_attention_bias,
                status="code_and_config",
                reader_result=_code_bias,
                config_paths=(_bound_path,),
                reader="decoder_attention_bias_for_path",
                reason=(
                    "exact Q/K/V/O projection constructors bind their bias "
                    "operand to this exact config occurrence"),
            )
        else:
            use_attention_bias = None
            _note_fact("decoder.attention", "bias", None,
                       _unknown_status, None)
    else:
        # Reader failure/absence stays unknown.  Looking up a conventional
        # alias here would turn config presence back into architectural input.
        use_attention_bias = None
        _note_fact("decoder.attention", "bias", None, _unknown_status, None)
    # Code-proven scores-scaling verdict (False ⇒ raw QK^T, T5 family).
    # A declared constant supplies the OPERAND only after code has proved that
    # this exact attention path applies a scale.  A number in config cannot,
    # by itself, manufacture an operation.
    code_scores_scaled = _code_scores_scaled(
        text_cfg, context, config_path=_text_path)
    _declared_score_scale = (
        attention_multiplier is not None or bool(query_pre_attn_scalar)
    )
    _applied_declared_scale = bool(
        _declared_score_scale and code_scores_scaled is True
    )
    _note_fact("decoder.attention", "scores_scale",
               "declared" if _applied_declared_scale else
               "unscaled (raw QK^T)" if code_scores_scaled is False else
               "sqrt(head_dim)" if code_scores_scaled is True else None,
               "code_and_config" if _applied_declared_scale
               else "code_proven" if code_scores_scaled is not None
               else _unknown_status,
               source=("attention_multiplier/query_pre_attn_scalar"
                       if _applied_declared_scale
                       else "decoder_attention_score_scaling_for_path"
                       if code_scores_scaled is not None
                       else None))
    # Learned sink logits in the softmax — config-silent, code-only.
    code_attention_sinks = _code_attention_sinks(
        text_cfg, context, config_path=_text_path)
    # H8 (§16.6) — migrate ``sinks`` from drawn-but-unledgered to a REGISTERED
    # code-proven fact.  Presence-proven from the attention forward, so recorded
    # only when True (no negative-proof obligation); its drawn witness is the
    # attention drill's sink column, so the projection-audit is satisfied.
    if code_attention_sinks:
        _note_fact("decoder.attention", "sinks", True, "code_proven",
                   "decoder_attention_sinks_for_path")
    # U6: a softcap node is authored only when the exact selected attention
    # callable proves scores/cap -> tanh -> *cap before softmax AND binds that
    # same cap operand to this exact checkpoint occurrence.  A familiar key or
    # a positive number alone is not mechanism evidence.
    attn_logit_softcap = None
    _softcap_code = _attention_logit_softcap_result(
        context, config_path=_text_path)
    _softcap_resolution = None
    if _softcap_code is not None and _softcap_code.status == "resolved":
        _bound_path = _softcap_code.value.config_path
        _prefix = tuple(_text_path)
        if len(_bound_path) == len(_prefix) + 1 \
                and _bound_path[:len(_prefix)] == _prefix:
            _softcap_resolution = _config_access.resolve(
                text_cfg, _bound_path[-1], (), path=_text_path)
        if _softcap_resolution is not None \
                and not _softcap_resolution.ambiguous \
                and _softcap_resolution.present \
                and _softcap_resolution.selected_path == ".".join(_bound_path) \
                and isinstance(_softcap_resolution.value, (int, float)) \
                and not isinstance(_softcap_resolution.value, bool):
            _softcap_resolution.bind(
                "decoder_attention_logit_softcap_for_path",
                fact_owner="decoder.attention", fact_key="logit_softcap")
            _softcap_decision = _softcap_resolution.consume_decision(
                mechanism="attention_logit_softcap",
                fact_owner="decoder.attention", fact_key="logit_softcap",
                reader="decoder_attention_logit_softcap_for_path",
                status="code_and_config")
            attn_logit_softcap = _softcap_decision.value
            _note_typed_fact(
                key="logit_softcap",
                owner="decoder.attention",
                value=attn_logit_softcap,
                status="code_and_config",
                reader_result=_softcap_code,
                config_paths=(_bound_path,),
                reader="decoder_attention_logit_softcap_for_path",
                reason=(
                    "the exact attention score path divides by, tanh-clamps, "
                    "and multiplies by this exact config-bound operand"),
            )
    else:
        # Keep an unproven declaration visible to the scoped config ledger; it
        # cannot reach AttentionSpec.
        _softcap_resolution = _config_access.resolve(
            text_cfg, "attn_logit_softcapping", (), path=_text_path)
    # Fused-QKV clipping is a separate numerical operation.  The source must
    # prove projection -> clamp -> live attention compute before the exact
    # config value is consumed.  Merely declaring ``clip_qkv`` is powerless.
    qkv_clip = None
    _qkv_clip_code = _attention_qkv_clip_result(
        context, config_path=_text_path)
    if _qkv_clip_code is not None and _qkv_clip_code.status == "resolved":
        _clip_path = _qkv_clip_code.value.config_path
        _clip_value = _consume_code_bound_path(
            "clip_qkv", _clip_path, fact_key="qkv_clip")
        if isinstance(_clip_value, (int, float)) \
                and not isinstance(_clip_value, bool):
            qkv_clip = _clip_value
            _note_typed_fact(
                key="qkv_clip",
                owner="decoder.attention",
                value=qkv_clip,
                status="code_and_config",
                reader_result=_qkv_clip_code,
                config_paths=(_clip_path,),
                reader="decoder_attention_qkv_clip_for_path",
                reason=(
                    "the exact fused QKV projection reaches a clamp bound by "
                    "this exact config operand and the clamped lane reaches "
                    "the selected attention compute"),
            )
    else:
        _clip_resolution = _scoped("clip_qkv")
        if _clip_resolution.present and not _clip_resolution.ambiguous:
            _clip_resolution.ignore(
                reason=(
                    "clip_qkv is a declaration only; the selected attention "
                    "source does not prove a live projection/clamp path"))
    # Cache capability is an independent code fact.  A use_cache declaration,
    # decoder-ness, or a cache-looking parameter cannot author it.  The exact
    # source must prove projected K/V -> parameter update -> two replacement
    # lanes reaching the selected attention compute.  A failed proof remains
    # None (unknown), never False.
    attention_cached = None
    _cache_code = _attention_cache_result(context, config_path=_text_path)
    if _cache_code is not None and _cache_code.status == "resolved":
        attention_cached = True
        _note_typed_fact(
            key="cached",
            owner="decoder.attention",
            value=True,
            status="code_proven",
            reader_result=_cache_code,
            config_paths=(),
            reader="decoder_attention_cache_for_path",
            reason=(
                "two exact projected lanes update a callable parameter and "
                "both returned replacements reach the selected attention "
                "compute"),
        )
    # MLP projection bias — the FFN twin of attention_bias (a Tier-3 chip when
    # True; None keeps "config does not declare it").  Code-authoritative like
    # its twin: Bloom's MLP Linears default to bias=True with a silent config;
    # `bias=config.mlp_bias` families still honor the checkpoint value through
    # the reader; Conv1D layouts abstain → the config spelling stands.
    _mlp_bias = consume("mlp_bias", fact_owner="decoder.ffn", fact_key="bias")
    # class_default tier (the attention twin's rule): an absent mlp_bias key
    # resolves to the installed config class's default at runtime.
    if _mlp_bias is None:
        _mlp_bias = _fact_class_defaults.get("mlp_bias")
    use_mlp_bias = bool(_mlp_bias) if _mlp_bias is not None else None
    _code_mlp = _code_mlp_bias(
        text_cfg, context, config_path=_text_path,
        ffn_mechanism_result=_ffn_mechanism)
    if _code_mlp is not None:
        use_mlp_bias = _code_mlp
    # An encoder-reuse switch (Gemma-2 5.x spelling): when TRUE the stack runs
    # bidirectional attention — a MASK fact, consumed here so a positive value
    # can never be silently dropped.
    forced_bidirectional = bool(consume("use_bidirectional_attention",
                                        fact_owner="decoder.attention",
                                        fact_key="mask"))
    # U2 mask default-kill: "causal" may only be DRAWN when the config
    # declares decoder-ness (is_decoder / causal-LM architecture suffix /
    # decoder-only wrapper / composite decoder slot — general vocabulary in
    # everchanging/transformer/decoderness.yaml). Otherwise the mask is a
    # typed "unknown" (pale chip) — BERT/T5 encoders stop being drawn causal.
    # The declaration is RESOLVED ON THE CONTEXT (ParseContext.build), so the
    # name-blind guard's scrubbed parse consumes the identical declaration;
    # a nested text-scope is_decoder flag (never scrubbed) still counts.
    _decoderness_src = (getattr(context, "declared_decoderness", None)
                        or ("is_decoder"
                            if consume("is_decoder",
                                       fact_owner="decoder.attention",
                                       fact_key="mask") is True else None))
    # U2 P2d — the CODE channel, wired ABOVE config-decoderness: mask-machinery
    # calls (is_decoder gates resolved from the checkpoint) / is_causal
    # literals.  BERT/T5-encoder become code-proven BIDIRECTIONAL; plain
    # decoders code-proven causal.  One honest discard: on a flat enc-dec
    # config with no declared decoder slot the drawn stack is the ENCODER
    # half, while the file provably also contains the (undrawn) decoder —
    # a causal-only verdict there is the other half's machinery (Whisper),
    # never this stack's fact.
    _code_causality = _code_attention_causality(text_cfg, context)
    if (_code_causality == "causal" and _is_enc_dec
            and not _decoderness_src):
        _code_causality = None
    if forced_bidirectional:
        _note_fact("decoder.attention", "mask", "bidirectional",
                   "config_declared", "use_bidirectional_attention")
    elif _code_causality is not None:
        _note_fact("decoder.attention", "mask", _code_causality,
                   "code_proven", "attention_causality_from_files")
    elif _decoderness_src:
        _note_fact("decoder.attention", "mask", "causal",
                   "config_declared", _decoderness_src)
    elif layer_types or sliding_window:
        # U8 owns per-layer mask/schedule SEMANTICS (including deriving the fact's
        # status from the deciding read's origin).  U2 keeps the existing
        # behaviour-preserving fact and does not arbitrate the schedule.
        _note_fact("decoder.attention", "mask", "windowed schedule",
                   "config_declared", "layer_types/sliding_window")
    else:
        _note_fact("decoder.attention", "mask", "unknown", _unknown_status, None)
    # Declaration-only until U7 proves the exact residual tap in the owner.
    # Inspect it for ownership, but do not let the flag author a drawing.
    _scoped("apply_residual_connection_post_layernorm")

    # ---- Layer topology ----
    # Parallel residual is projected only after the exact cell reader proves its
    # residual equations.  A config selector can choose guarded alternatives
    # only when the exact block assignment binds that guard operand to the exact
    # config path; an unbound declaration remains unknown.
    # Distinct INPUT norms a parallel-residual layer applies, read from the code
    # dataflow: 1 = SHARED (GPT-J), 2 = SEPARATE (GPT-NeoX input+post norms) —
    # fixes the "two-norms-drawn-as-one" bug; None (Falcon conditional) → 1.
    parallel_norm_count = None
    # A selector is consumed only inside ``_cell_topology_result`` after the
    # exact block assignment and guard bind that occurrence to one derived fact.
    # Unbound declarations remain visible audit debt; they are never globally
    # ignored and never author a conventional topology.
    residual_topology = (
        _cell_topology.residual_topology
        if _cell_topology is not None else "unknown")
    _residual_topology_config_paths = (
        _cell_topology.residual_config_paths
        if _cell_topology is not None else ())
    _residual_topology_status = (
        "code_and_config"
        if _residual_topology_config_paths else "code_proven")
    use_parallel_residual = residual_topology == "parallel"
    if use_parallel_residual:
        parallel_norm_count = _code_parallel_norm_count(
            text_cfg, context, config_path=_text_path)
    if residual_topology != "unknown":
        _note_typed_fact(
            key="residual_topology", owner="decoder.layer",
            value=residual_topology, status=_residual_topology_status,
            reader_result=_cell_topology_result_value,
            config_paths=_residual_topology_config_paths,
            reader="decoder_cell_topology_for_path",
            reason=(
                "exact residual equations prove sequential or parallel cell "
                "topology"),
        )
    else:
        _note_fact(
            "decoder.layer", "residual_topology", "unknown", _unknown_status)
    _note_fact(
        "decoder.layer", "parallel_norm_count", parallel_norm_count,
        "code_proven" if parallel_norm_count is not None else _unknown_status,
        source=(
            "decoder_parallel_norm_count_for_path"
            if parallel_norm_count is not None else None
        ),
    )

    # ---- MoE ----
    num_experts = consume(
        "num_experts", fact_owner="decoder.ffn", fact_key="num_experts")
    num_experts_per_tok = consume(
        "num_experts_per_tok", fact_owner="decoder.ffn",
        fact_key="num_experts_per_tok")
    num_shared_experts = consume(
        "num_shared_experts", fact_owner="decoder.ffn",
        fact_key="num_shared_experts")
    moe_intermediate_size = consume(
        "moe_intermediate_size", fact_owner="decoder.ffn",
        fact_key="moe_intermediate_size")
    enable_moe_block    = _g(text_cfg, "enable_moe_block")
    moe_active          = bool(num_experts) and (enable_moe_block is not False)
    _code_expert_fused = (
        _code_expert_storage(
            text_cfg, context, config_path=_text_path)
        if moe_active else None)
    if _code_expert_fused is not None:
        _note_fact("decoder.ffn.expert", "expert_projection_mode",
                   _code_expert_fused, "code_proven",
                   source="decoder_routed_expert_storage_for_path")
    # Schedule declarations stay visible as inputs to the legacy source reader,
    # but are not consumed as a diagram fact merely because they exist.
    # U7 owns their exact source/config binding.  Until then only the
    # source-resolved schedule below may author per-layer kind or the
    # ``every_layer`` summary.
    _g(text_cfg, "first_k_dense_replace")
    _g(text_cfg, "moe_layer_freq")
    _g(text_cfg, "interleave_moe_layer_step")
    _g(text_cfg, "decoder_sparse_step")
    _g(text_cfg, "mlp_only_layers")
    _g(text_cfg, "moe_layers")
    for _field in _LAYER_SCHEDULES["moe_comma_string_fields"]:
        _g(text_cfg, _field)
    # CODE-AUTHORITATIVE per-layer MoE schedule (which layers build an experts
    # class as their FFN field) — read from the decoder layer's construction,
    # name-independent.  Supplies the per-layer SHAPE; ``moe_active``
    # (num_experts>0) stays the is-this-MoE-at-all geometry gate.  None when the
    # code can't resolve it (hybrid SSM-MoE / exotic gate / no source) → the
    # schedule stays unknown. Config schedule fields remain visible evidence
    # for U7, but never select layer architecture here.
    code_moe_schedule = (
        _code_moe_schedule(text_cfg, context, num_layers)
        if moe_active else None
    )
    moe_every_layer = (
        all(code_moe_schedule)
        if code_moe_schedule is not None else None
    )
    # Router behaviour: gating fn, grouped/node-limited routing, top-k renorm,
    # routed-output scale (DeepSeek-V3, Kimi-K2, GLM, Qwen3-MoE).
    moe_routing = (_moe_routing(text_cfg, context, path=_text_path)
                   if moe_active else None)
    # A clip declaration is not enough to prove which exact activation consumes
    # it (GPT-OSS applies this inside routed experts, not the ordinary/shared
    # mechanism read above). Keep it inspected until U7 binds the exact expert
    # callable and dispatch.
    _scoped("swiglu_limit")
    activation_clip = None

    # ---- Cross-layer KV
    #  sharing (the last N layers reuse K/V from earlier) ----
    num_kv_shared_layers   = _g(text_cfg, "num_kv_shared_layers") or 0
    first_shared_layer     = (num_layers - num_kv_shared_layers) if num_kv_shared_layers else num_layers

    # ---- Per-Layer Embedding side pathway ----
    ple_dim   = _g(text_cfg, "hidden_size_per_layer_input") or 0
    ple_vocab = _g(text_cfg, "vocab_size_per_layer_input") or get("vocab_size", 0)

    # ---- Decoder layers that read external modality states through cross-attention ----
    cross_attn_layer_set = set(_cross_attention_layers(cfg, text_cfg) or [])
    # Seq2seq composite (MusicGen): the schedule is CONSTRUCTION evidence —
    # the decoder-layer class builds its cross-attention module
    # unconditionally in __init__, so EVERY layer cross-attends the declared
    # encoder's states.  The config alone cannot say (MusicGen's decoder
    # sub-config even carries add_cross_attention: false); the declared
    # is_encoder_decoder + encoder-role slot scope the source read.
    composite_encoder_type = _composite_encoder_model_type(cfg)
    # ADDITIVE cross-attention: the construction reader requires BOTH a
    # self-attention and a cross-attention field on the layer class, so the
    # proven shape keeps self-attention AND gains a cross sublayer — unlike a
    # declared mllama schedule, whose cross layers REPLACE self-attention.
    cross_attention_additive = False
    if (not cross_attn_layer_set and num_layers
            and _is_enc_dec and composite_encoder_type):
        if _code_cross_attention_all_layers(
                context, config_path=_text_path):
            cross_attn_layer_set = set(range(num_layers or 0))
            cross_attention_additive = True
        else:
            # Declared enc-dec composite whose decoder SOURCE we can't read
            # (custom package not installed — Parler-TTS): the schedule stays
            # unproven and nothing is drawn, but never silently.
            warnings.append(
                "Cross-attention schedule unproven (the decoder's modeling "
                "source is not installed) — the declared encoder conditioning "
                "is shown, but no per-layer cross-attention is drawn.")
    has_vision_side_state = (_g(cfg, "vision_config") is not None
                             or _g(cfg, "vision_model_config") is not None)
    has_cross_attention_side_state = bool(
        cross_attn_layer_set and (has_vision_side_state or composite_encoder_type)
    )

    # Declared decoder-scope flags (Parler/MusicGen lineage): read them so the
    # ownership audit sees them; each is folded only where it is a proven fact.
    # scale_embedding AUTHORS a drawn embed-card fact ("scaled × sqrt(d)") —
    # a present declaration is consumed into the embedding-scale fact.
    with _config_access.config_container(_text_path, obj=text_cfg):
        _se_res = _config_access.resolve(text_cfg, "scale_embedding", ())
        _scale_embedding = (
            None if _se_res.ambiguous or _se_res.state != "present"
            else _se_res.consume_decision(
                mechanism="embedding_scale", fact_owner="model",
                fact_key="embedding_scale",
                reader="adapters.transformer.parser.parse").value)
    _declared_rope = _g(text_cfg, "rope_embeddings")       # declared positional flag
    _cross_kv_heads_declared = _g(text_cfg, "num_cross_attention_key_value_heads")
    if _declared_rope and _code_position_evidence.status != "proven":
        warnings.append(
            "Config declares rope_embeddings but the positional drawing is "
            "unresolved from source — the declared flag is not yet folded in.")

    # ---- Walk the layer stack ----
    unknown_layer_types: set[str] = set()
    cross_layer_edges: list[CrossLayerEdge] = []

    layers = []
    for i in range(num_layers or 0):
        mask, window, is_full_in_sliding_stack = _layer_mask(
            i, layer_types, sliding_window, sliding_window_pattern,
            has_sliding_in_stack, unknown_layer_types, max_window_layers,
        )
        if forced_bidirectional and mask in ("causal", "global"):
            mask = "bidirectional"       # encoder-reuse switch: a MASK fact
        elif _code_causality == "bidirectional" and mask in ("causal", "global"):
            # U2 P2d: the source PROVES this stack's self-attention is
            # bidirectional (BERT/T5-encoder machinery, is_decoder resolved
            # from the checkpoint) — same flip as the declared switch above.
            mask = "bidirectional"
        elif mask == "causal" and not _decoderness_src and _code_causality != "causal":
            # U2 default-kill: nothing declared OR code-proved this stack a
            # decoder — the causal fall-through is a typed unknown, not an
            # asserted fact (the census's headline: T5/BERT drawn causal).
            mask = "unknown"
        compress_ratio = _compress_ratio_for_layer(i, compress_ratios, layer_types)

        # Per-layer dual KV: full layers in a sliding stack use the global counts.
        if is_full_in_sliding_stack:
            layer_kv_heads = num_kv_global
            layer_head_dim = head_dim_global
        else:
            layer_kv_heads = num_kv_heads
            layer_head_dim = head_dim

        layer_type = layer_types[i] if i < len(layer_types) else None
        # U3: a canonical layer_types token may name a NON-softmax token mixer
        # (Qwen3-Next linear_attention -> gated_delta; MiniMax lightning ->
        # linear; recurrentgemma recurrent -> recurrent).  is_gated_delta keeps
        # its exact meaning for byte-stability; is_mixer covers all three.
        mixer_kind = _mixer_kind_for(layer_type)
        is_gated_delta = mixer_kind == "gated_delta"
        is_mixer = mixer_kind is not None
        if mixer_kind:
            attn_kind = mixer_kind
        elif _bound_attention is not None \
                and layer_kv_heads == _bound_attention.num_kv_heads:
            attn_kind = _bound_attention.kind
        else:
            attn_kind = None
        # The layer TYPE follows the declared schedule alone: a bare component
        # config (mllama_text_model) declares cross_attention_layers without a
        # vision_config sibling, and its cross layers ARE cross-attention
        # layers — suppressing them drew 8 wrong layer types on the standalone
        # parse (caught by the U4 schedule lock).  The vision gate only decides
        # what the side-state SOURCE honestly says.
        is_cross_attn_layer = i in cross_attn_layer_set

        kv_source: int | None = None
        if i >= first_shared_layer:
            kv_source = _last_matching_layer(layer_types, i, first_shared_layer)
            if kv_source is not None:
                cross_layer_edges.append(
                    CrossLayerEdge(kind="kv_share", from_layer=kv_source, to_layer=i, shared=["K", "V"])
                )

        position_mechanism = (
            ("unknown", "unknown")
            if position_schedule_unresolved
            else _position_for_layer(
                _code_position_evidence, mixer_kind=mixer_kind,
            )
        )
        layer_uses_rope = None if position_schedule_unresolved else uses_rope
        # NoPE is a mechanism claim. U8 will restore interleaved schedules after
        # proving the exact source selector; raw list/interval values cannot
        # author it in U4.
        is_nope = position_mechanism == ("none", "none")
        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=(linear_num_v_heads or num_heads) if is_gated_delta else num_heads,
            num_kv_heads=(linear_num_k_heads or layer_kv_heads) if is_gated_delta else layer_kv_heads,
            head_dim=(linear_k_head_dim or layer_head_dim) if is_gated_delta else layer_head_dim,
            kv_lora_rank=kv_lora_rank if is_mla else None,
            q_lora_rank=q_lora_rank if is_mla else None,
            qk_nope_head_dim=qk_nope_head_dim if is_mla else None,
            qk_rope_head_dim=qk_rope_head_dim if is_mla else None,
            v_head_dim=(linear_v_head_dim if is_gated_delta else v_head_dim_cfg if is_mla else None),
            rope_dim=rope_dim_value,
            mask=mask,
            window_size=window,
            kv_source_layer=kv_source,
            qk_norm=qk_norm_layers[i] if i < len(qk_norm_layers) else use_qk_norm,
            rope=(layer_uses_rope and not is_mixer),
            position_kind=position_mechanism[0],
            position_application=position_mechanism[1],
            position_declared=False,
            rope_theta_declared=None,
            bias=use_attention_bias,
            no_rope=is_nope,
            cross_attention=is_cross_attn_layer and not cross_attention_additive,
            cross_kv_source=(("projected image states"
                              if has_vision_side_state and has_cross_attention_side_state
                              else "encoded prompt states (the "
                                   "conditioning encoder tower)"
                              if has_cross_attention_side_state else
                              "external encoder states (encoder not in this config)")
                             if is_cross_attn_layer and not cross_attention_additive
                             else None),
            compress_ratio=compress_ratio,
            # Sparse-attention indexer fan-in. CSA declares it alongside a
            # compress_ratio; DeepSeek-V3.2 DSA declares its own indexer geometry
            # (index_n_heads/index_head_dim) — read both so neither is dropped.
            index_topk=_g(text_cfg, "index_topk"),
            index_n_heads=_g(text_cfg, "index_n_heads"),
            index_head_dim=_g(text_cfg, "index_head_dim"),
            mrope_section=mrope_section,
            conv_kernel_size=linear_conv_kernel if is_gated_delta else None,
            output_gate=(attn_output_gate if not is_gated_delta else None),
            scores_scale=(
                _declared_scores_scale(
                    attention_multiplier, query_pre_attn_scalar,
                    layer_head_dim or (hidden_size // num_heads
                                       if hidden_size and num_heads else None)
                )
                if _applied_declared_scale else None
            ),
            scores_scaled=code_scores_scaled,
            sinks=(code_attention_sinks and attn_kind in ("mha", "gqa", "mqa")),
            logit_softcap=attn_logit_softcap,
            qkv_clip=qkv_clip,
            cached=attention_cached,
            output_projection=attention_output_projection,
            asserted=(),
            projection_mode=(
                _code_attention_storage
                if attn_kind in ("mha", "gqa", "mqa")
                and not is_gated_delta
                else None
            ),
            variant=_mixer_variant(mixer_kind),
        )

        # Code decides per-layer kind only when exact construction resolution
        # succeeds. A config schedule remains visible evidence for U8 but is
        # not a fallback architecture selector.
        # U4-C: per-layer kind is projected only from source-resolved
        # construction. Config schedules remain visible evidence/debt for U7;
        # they no longer manufacture dense/MoE variants on reader abstention.
        code_layer_is_moe = (
            bool(code_moe_schedule[i])
            if code_moe_schedule is not None else None
        )
        ffn_kind = (
            "moe" if code_layer_is_moe is True
            # The exact construction schedule proves the layer selects its
            # ordinary (non-routed) FFN occurrence. That is sufficient for the
            # outer kind "dense"; gate/storage/activation remain independent
            # and may still be unknown.
            else "dense" if code_layer_is_moe is False
            else "dense"
            if not moe_active and _ffn_mechanism_value is not None
            else None
        )

        if ffn_kind == "moe":
            ffn = FFNSpec(
                kind="moe",
                # The resolved activation belongs to the ordinary/shared FFN
                # owner. Routed experts require their own activation proof.
                activation=activation,
                # The declared ordinary/shared width remains its own lane.
                # Routed experts never borrow it; DBRX is the control.
                intermediate_size=intermediate_size,
                gated=ffn_gated,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_shared_experts=num_shared_experts,
                # Routed-expert width is an independent lane. An absent expert
                # width cannot borrow the ordinary/shared FFN width.
                expert_intermediate_size=moe_intermediate_size or None,
                routing=moe_routing,
                activation_clip=activation_clip,
                bias=use_mlp_bias,
                projection_mode=_code_storage_mode,
                expert_projection_mode=_code_expert_fused,
                # U4-D: unresolved cell topology is represented by the layer's
                # typed unknown fields, never carried as an asserted FFN fact.
                asserted=(),
            )
        else:
            ffn = FFNSpec(
                kind=ffn_kind,
                activation=activation,
                intermediate_size=intermediate_size,
                gated=ffn_gated,
                activation_clip=activation_clip,
                bias=use_mlp_bias,
                projection_mode=_code_storage_mode,
                # B5/U2: see the MoE branch — unknown FFN storage is represented
                # by projection_mode=None and an opaque region, never a second
                # asserted tag.
                asserted=(),
            )

        # The ADDITIVE cross sublayer's own spec: same construction-declared
        # geometry, K/V from the encoder's states (full over the prompt, no
        # positional claim asserted).
        cross_spec = None
        if is_cross_attn_layer and cross_attention_additive:
            # Parler declares a separate KV-head count for the CROSS sublayer
            # (num_cross_attention_key_value_heads) — its own GQA geometry.
            cross_spec = AttentionSpec(
                kind=attn_kind,
                num_heads=num_heads,
                num_kv_heads=(int(_cross_kv_heads_declared)
                              if _cross_kv_heads_declared else layer_kv_heads),
                head_dim=layer_head_dim,
                mask="full",
                rope=None,
                position_kind="unknown",
                position_application="unknown",
                bias=use_attention_bias,
                cross_attention=True,
                # U2-R9: structural prose, identity-free — the slot's declared
                # type is a display LABEL on the tower card, never in the
                # wiring description (name-blind law).
                cross_kv_source=("encoded prompt states (the "
                                 "conditioning encoder tower)"),
            )

        extra_blocks = list(per_layer_embedding_blocks(hidden_size, ple_dim, activation="gelu")) if ple_dim else []
        if is_cross_attn_layer:
            extra_blocks.append(_cross_attention_states_side_block(
                "conditioning" if (composite_encoder_type and not has_vision_side_state)
                else "vision",
                encoder_type=composite_encoder_type,
                feeds="cross_attn" if cross_spec is not None else "attn",
            ))

        if use_parallel_residual:
            layers.append(parallel_decoder_layer(
                i, attn, ffn, hidden_size, norm_kind=norm_kind,
                norm_count=parallel_norm_count))
        else:
            layers.append(decoder_layer(
                i, attn, ffn, hidden_size,
                norm_kind=norm_kind,
                norm_placement=norm_placement,
                residual_topology=residual_topology,
                extra_blocks=extra_blocks,
                residual_scale=residual_multiplier,
                cross_attention_spec=cross_spec,
            ))

    for lt in sorted(unknown_layer_types):
        warnings.append(f"Config layer_types contains unrecognized value {lt!r} — treated as causal.")

    vocab_size = consume("vocab_size", fact_owner="model", fact_key="vocab_size") or 0  # embed-table count; ambiguity already blocks
    # U2 default-kill (the live wrong-value fix): absence of the tie flag is
    # NOT "untied" — the installed config CLASS default decides (gpt2 / t5 /
    # bert / bloom / falcon all omit the key and tie by class default). Tiers
    # (U2 P2b): declared flag → CODE (unconditional manual tying idiom, the
    # remote-code channel; ``_tied_weights_keys`` is capability, never proof)
    # → class default (context.class_defaults, resolved once at context
    # build) → typed unknown (param count annotates, never picks).
    _tie_raw = consume("tie_word_embeddings", fact_owner="model", fact_key="tie_word_embeddings")
    if _tie_raw is None and text_cfg is not cfg:
        # U1 (unmasked by the exact resolver): a multimodal WRAPPER declares the
        # text head's tying at WRAPPER level (qwen2-vl: top-level
        # ``tie_word_embeddings: false`` while text_config lacks the field).
        # The old absent-union hid this unread declaration; the checkpoint's
        # explicit value belongs to rung 1 of the ladder, above code/class
        # rungs.  Owner stays root — it is the wrapper's declaration.
        _wrap_tie = _config_access.resolve(
            cfg, "tie_word_embeddings", _ALIASES.get("tie_word_embeddings", ()))
        if _wrap_tie.state == "present":
            _tie_raw = _wrap_tie.consume(fact_key="tie_word_embeddings")
    if _tie_raw is not None:
        tie_word_embeddings = bool(_tie_raw)
        _note_fact("model", "tie_word_embeddings", tie_word_embeddings,
                   "config_declared", "tie_word_embeddings")
    elif _code_lm_head_tying(
            text_cfg, context, config_path=_text_path):
        tie_word_embeddings = True
        _note_fact("model", "tie_word_embeddings", True,
                   "code_proven", "manual_weight_tying_for_path")
    else:
        _tie_cls = _fact_class_defaults.get("tie_word_embeddings")
        if _tie_cls is not None:
            tie_word_embeddings = bool(_tie_cls)
            _note_fact("model", "tie_word_embeddings", tie_word_embeddings,
                       "class_default",
                       "installed config-class default (AutoConfig.for_model)")
        else:
            tie_word_embeddings = None
            _note_fact("model", "tie_word_embeddings", None, _unknown_status, None)

    _owner_ns = getattr(context, "component_namespace", "root")
    modality_extras = multimodal_extras(cfg, text_cfg, hidden_size, namespace=_owner_ns)
    if modality_extras:
        try:
            from ...evidence.audio import audio_tower_evidence
            modality_extras = apply_audio_evidence(
                modality_extras,
                audio_tower_evidence(cfg, bundle=context.source_bundle),
            )
        except Exception:
            # Missing/ambiguous source keeps one honest opaque audio tower and
            # connector.  It never revives the former family-labelled sketch.
            pass
        try:
            from ...evidence.vision import vision_tower_evidence
            modality_extras = apply_vision_evidence(
                modality_extras,
                vision_tower_evidence(
                    cfg, bundle=context.source_bundle,
                    index=context.program_index()),
            )
        except Exception:
            # A failed source extractor must leave the path honestly generic;
            # it is never permission to restore a family-derived structure.
            pass
        try:
            from ...evidence.fusion import fusion_evidence
            modality_extras = apply_fusion_evidence(
                modality_extras,
                fusion_evidence(cfg, bundle=context.source_bundle),
                cfg,
                text_cfg,
            )
        except Exception:
            # Fusion structure is wrapper-code evidence.  Failure must leave the
            # base payload opaque, never revive a family/config guess.
            pass
        try:
            from ...evidence.projector import projector_evidence
            modality_extras = apply_projector_evidence(
                modality_extras,
                projector_evidence(cfg, bundle=context.source_bundle),
                cfg, owner_namespace=_owner_ns,
            )
        except Exception:
            # As with the tower extractor, failure leaves one honest generic
            # connector.  Config dimensions survive; callable structure does not.
            pass
    # Multi-codebook token streams (MusicGen family): K is the decoder
    # config's OWN num_codebooks; the summed-embeddings / stacked-heads SHAPE
    # comes from construction+forward evidence — only-when-present so every
    # single-stream decoder stays byte-stable.
    codebooks = None
    # U2-R9: name the (possibly slot-nested) text config for these reads —
    # a MusicGen decoder document's codebook fields must ledger located.
    with _config_access.config_container(_text_path, obj=text_cfg):
        _cb_res = _config_access.resolve(text_cfg, "num_codebooks", ())
        _num_codebooks = (None if _cb_res.ambiguous else
                          (_cb_res.value if _cb_res.state == "present"
                           else None))
    if isinstance(_num_codebooks, int) and _num_codebooks > 1:
        # These declarations AUTHOR the drawn K-codebook structure — consumed
        # (typed decision), matching the register's extras:codebooks row.
        with _config_access.config_container(_text_path, obj=text_cfg):
            _cb_res.consume_decision(
                mechanism="codebooks", fact_owner="decoder",
                fact_key="num_codebooks",
                reader="adapters.transformer.parser.parse")
            _ac_res = _config_access.resolve(text_cfg, "audio_channels", ())
            _located_audio_channels = (
                None if _ac_res.ambiguous or _ac_res.state != "present"
                else _ac_res.consume_decision(
                    mechanism="codebooks", fact_owner="decoder",
                    fact_key="audio_channels",
                    reader="adapters.transformer.parser.parse").value)
        from ...evidence.codebook_streams import \
            decoder_codebook_streams_for_path
        _cb_result = decoder_codebook_streams_for_path(
            context.program_index(), context.source_bundle, _text_path)
        streams = {
            "embeddings_summed": None,
            "heads_stacked": None,
        }
        if _cb_result.has_value:
            streams = {
                "embeddings_summed":
                    _cb_result.value.embeddings_summed,
                "heads_stacked": _cb_result.value.heads_stacked,
            }
        codebooks = {
            "num": _num_codebooks,
            "vocab_per_book": vocab_size,
            "audio_channels": _located_audio_channels,
            **streams,
        }
    # U2-R7: ONE consumption for the occurrence — the LM-head card here and the
    # block-diffusion canvas path below share this value (never two consumes
    # of one occurrence in a single parse).
    final_logit_softcap = consume("final_logit_softcapping",
                                  fact_owner="model",
                                  fact_key="final_logit_softcapping")
    from ...ir import canonical_norm_kind
    embedding_norm_kind = canonical_norm_kind(
        _code_embedding_norm(text_cfg, context))
    if embedding_norm_kind is not None:
        _note_fact(
            "model", "embedding_norm_kind", embedding_norm_kind,
            "code_proven", source="embedding_stage_norm_evidence",
        )
    # U7 owns the root pre-head reader. A repeated layer norm cannot certify
    # the distinct model-stage final norm.
    final_norm_kind = None
    _note_fact(
        "model", "final_norm_kind", None, _unknown_status, source=None,
    )
    extras = decoder_extras(
        vocab_size,
        hidden_size,
        tie_word_embeddings,
        per_layer_embedding_extras(hidden_size, ple_dim, ple_vocab, num_layers) if ple_dim else None,
        modality_extras,
        embed_norm=embedding_norm_kind,
        final_norm=final_norm_kind,
        # Gemma-2's final_logit_softcapping is a REAL pre-sampling op — the LM
        # head card states it (only-when-present; everyone else byte-stable).
        final_logit_softcap=final_logit_softcap,
        codebooks=codebooks,
    )
    if _scale_embedding:
        for block in extras["render"]["model_blocks"]:
            if block.get("id") == "embed":
                block["facts"] = (block.get("facts") or []) + [
                    "scaled × √d (scale_embedding)"]
    extras["position_encoding"] = {
        **_code_position_evidence.to_dict(),
    }
    absolute_item = next((item for item in _position_mechanisms
                          if item.kind in {"learned_absolute", "fixed_absolute"}), None)
    if absolute_item is not None:
        learned = absolute_item.kind == "learned_absolute"
        position_label = "Learned Position Embedding" if learned else "Fixed Position Encoding"
        extras["render"]["model_blocks"].extend([
            {
                "id": "position_ids", "role": "input", "kind": "source",
                "label": "Position IDs", "title": "Position indices",
                "description": "Sequence-position indices used to look up learned positional vectors.",
            },
            {
                "id": "position_embed", "role": "embedding", "kind": "embedding",
                "label": position_label, "title": position_label,
                "description": (
                    "Looks up one learned positional vector for each sequence position."
                    if learned else
                    "Selects a deterministic sinusoidal vector for each sequence position."
                ),
            },
            {
                "id": "position_add", "role": "residual", "kind": "residual_add",
                "label": "+", "title": "Token + position embedding",
                "description": (
                    "Adds the learned positional vector to the token embedding before the decoder stack."
                    if learned else
                    "Adds the fixed positional vector to the token embedding before the decoder stack."
                ),
            },
        ])

    # ---- Block diffusion (masked/canvas-denoising text LMs) ----------------------
    # Detected by EVIDENCE, not one exact model_type string: a block-diffusion LM
    # declares a denoising CANVAS (``canvas_length``) and/or sits in the diffusion
    # architecture family — so a sibling block-diffusion model (not just
    # diffusion_gemma) routes here too.  The inner text_config is parsed as a
    # normal transformer for the per-layer IR; we then override:
    #   1. The render layout (block_diffusion loop view).
    #   2. Per-layer blocks: this family has post-attention norm, parallel
    #      dense-MLP + MoE, post-FFN norm, and a per-layer learned scalar —
    #      none of which the generic decoder_layer topology expresses (the block
    #      builder is the opaque-source fallback for these research models).
    # Block-diffusion layout is a CONFIG fact (canvas_length declares the
    # denoising canvas) — never a model_type spelling.  A block-diffusion
    # config without canvas_length renders as the plain decoder its config
    # declares; identity must not fill the gap (eradication plan I-07).
    if _g(cfg, "canvas_length") is not None:
        from .blocks.model import block_diffusion_loop_blocks
        canvas_length = int(_g(cfg, "canvas_length") or 256)
        # U2-R7: the occurrence was consumed ONCE above (the LM-head card
        # read) — reuse that value here.
        final_softcap = final_logit_softcap
        extras["render"]["layout"] = "block_diffusion"
        extras["render"]["loop_blocks"] = block_diffusion_loop_blocks(
            n_layers=num_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            canvas_length=canvas_length,
            final_logit_softcap=final_softcap,
            ffn_intermediate_size=intermediate_size,
        )
        extras["block_diffusion"] = {"canvas_length": canvas_length}
        # `hidden_states * self.layer_scalar` is a Tier-3 layer property (one
        # learned scalar) — a block would be wrong (Gate C), and the frame caption
        # wasn't worth the space, so it is intentionally not surfaced here.
        # This single stack is run two ways with TIED weights (HF:
        # encoder.language_model.layers ↔ decoder.layers): the encoder is causal,
        # the decoder bidirectional.  Caption the × N frame so the shared dual
        # role is clear when landing on this panel from either loop block.
        extras["render"]["repeat_note"] = [
            "shared by encoder (causal)",
            "& decoder (bidirectional)",
        ]

    # ---- Multi-Token Prediction heads (DeepSeek-V3 style next-token modules) ----
    # U2-R7: consumed into the MTP-modules fact (extras["mtp"]).
    mtp_modules = (consume("num_nextn_predict_layers", fact_owner="model",
                           fact_key="mtp_modules")
                   or _g(text_cfg, "num_mtp_layers"))
    try:
        mtp_modules = int(mtp_modules) if mtp_modules else 0
    except (TypeError, ValueError):
        mtp_modules = 0
    if mtp_modules > 0:
        extras["mtp"] = {
            "num_modules": mtp_modules,
            "predicts_extra_tokens": mtp_modules,
            "shares_embedding": True,
            "shares_output_head": True,
        }
        # The MTP transformer block is a decoder layer, so hand it a
        # representative layer's own blocks; the router renders each (attention,
        # FFN/MoE, …) wherever it appears — no MTP-specific plumbing.
        rep_blocks = layers[-1].blocks if layers else []
        extras["render"]["model_blocks"].append(
            mtp_head_block(
                mtp_modules, hidden_size, vocab_size, tie_word_embeddings,
                block_children=rep_blocks,
            )
        )

    if sliding_window:
        extras["sliding_window"] = {
            "window": sliding_window,
            "first_full_layers": max_window_layers or 0,
        }
    if moe_active:
        extras["moe"] = {
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "num_shared_experts": num_shared_experts,
            "every_layer": moe_every_layer,
        }
        if moe_routing:
            extras["moe"]["routing"] = moe_routing
    if no_rope_interval:
        extras["irope"] = {"no_rope_interval": no_rope_interval}
    if num_kv_shared_layers:
        extras["num_kv_shared_layers"] = num_kv_shared_layers
    if rope_dim_value and head_dim:
        extras.setdefault("rope", {})["partial_pct"] = round(rope_dim_value / head_dim, 3)

    # RoPE scaling (YaRN, linear, dynamic, ntk, ...) reported as info-only.
    # U2-R7: name the text config for these reads (located, not bare leaves).
    with _config_access.config_container(_text_path, obj=text_cfg):
        rope_params = _g(text_cfg, "rope_parameters") or _g(text_cfg, "rope_scaling")
    if isinstance(rope_params, dict):
        rope_type = rope_params.get("rope_type") or rope_params.get("type")
        with _config_access.config_container(_text_path, obj=text_cfg):
            _theta_fallback = _g(text_cfg, "rope_theta")
        scaling = {
            "type": rope_type,
            "factor": rope_params.get("factor"),
            "original_max_position_embeddings": rope_params.get("original_max_position_embeddings"),
            "rope_theta": rope_params.get("rope_theta") or _theta_fallback,
        }
        extras.setdefault("rope", {}).update({k: v for k, v in scaling.items() if v is not None})
        # These SUBKEYS are inspected above via plain dict reads. Record the
        # inspection so ownership is visible; do not label it consumption until
        # the corresponding rendered-fact receipts exist.
        with _config_access.config_container(_rope_path, obj=rope_params):
            for inspected in ("rope_type", "type", "factor",
                              "original_max_position_embeddings", "rope_theta"):
                if inspected in rope_params:
                    debug.note_access(inspected, source_obj=rope_params)

    # RoPE base frequency — present on most rotary models even without a scaling
    # dict (the block above only fires when one is declared); surface it always.
    with _config_access.config_container(_text_path, obj=text_cfg):
        rope_theta = _g(text_cfg, "rope_theta")
    if rope_theta is None and attn_cfg:
        with _config_access.config_container(
                (*_text_path, "attn_config"), obj=attn_cfg):
            rope_theta = _g(attn_cfg, "rope_theta")
    if rope_theta is not None:
        extras.setdefault("rope", {}).setdefault("rope_theta", rope_theta)

    # Logit / query softcap (Gemma 2/3 style) — info-only annotation.
    # The attention cap and query-score operand now live on their typed U6
    # facts/specs.  Keep only the distinct model-head cap in this legacy extras
    # family until its own exact pre-sampling projection is migrated.
    for cap_key in ("final_logit_softcapping",):
        val = _g(text_cfg, cap_key)
        if val is not None:
            extras.setdefault("softcap", {})[cap_key] = val

    # Surface the raw partial-rotary fraction the config declared, when present.
    if partial_rotary_fac is not None:
        extras["partial_rotary_factor"] = partial_rotary_fac

    # Per-layer dual-KV info, when both sides differ.
    if _g(text_cfg, "num_global_key_value_heads") or _g(text_cfg, "global_head_dim"):
        extras["dual_kv"] = {
            "sliding": {"num_kv_heads": num_kv_heads, "head_dim": head_dim},
            "global":  {"num_kv_heads": num_kv_global, "head_dim": head_dim_global},
        }

    # U2: ONE consolidated banner line for every fact the ledger left
    # unresolved (position warns separately, unchanged) — the render tier
    # draws these pale; the banner says why.
    if _facts is not None:
        _pale_facts = sorted({
            key.rsplit(".", 1)[1].replace("_", " ")
            for key, rec in _facts.records.items()
            if rec.status in ("unknown", "ambiguous", "oracle_missing")
        })
        if _pale_facts:
            warnings.append(
                "Unresolved code-defined facts (drawn honestly, never asserted): "
                + ", ".join(_pale_facts)
                + (" — modeling source is present but unresolved."
                   if _source_present else " — modeling source is unavailable."))

    ir = ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=consume("max_position_embeddings", fact_owner="model", fact_key="max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        embedding_norm_kind=embedding_norm_kind,
        final_norm_kind=final_norm_kind,
        cross_layer_edges=cross_layer_edges,
        extras=extras,
        warnings=warnings,
    )

    # Centralized diagnostics (toggle in adapters/transformer/debug.py), emitted
    # after every field access so the unparsed report is accurate:
    #   * config fields the parser never read, and
    #   * the reasons this config came out partial.
    debug.report_unparsed([cfg, text_cfg, attn_cfg, ffn_cfg], model=ir.name)
    debug.report_partial(warnings, model=ir.name)

    return ir


# ---------------------------------------------------------------------------
# Per-layer helpers
# ---------------------------------------------------------------------------


def _is_sliding_label(lt: str) -> bool:
    return lt in _SLIDING_LABELS or "sliding" in lt


def _is_full_label(lt: str) -> bool:
    return lt in _FULL_LABELS


def _is_compressed_sparse_label(lt: str) -> bool:
    return lt in _COMPRESSED_SPARSE_LABELS


def _is_heavily_compressed_label(lt: str) -> bool:
    return lt in _HEAVILY_COMPRESSED_LABELS


def _layer_mask(i, layer_types, sliding_window, sliding_window_pattern, has_sliding_in_stack, unknown, max_window_layers=None):
    """Resolve (mask, window, is_full_in_sliding_stack) for a single layer."""
    if layer_types and i < len(layer_types):
        lt = layer_types[i]
        if _is_sliding_label(lt):
            return "sliding", sliding_window, False
        if _is_compressed_sparse_label(lt):
            return "compressed_sparse", None, False
        if _is_heavily_compressed_label(lt):
            return "heavily_compressed", None, False
        if _is_full_label(lt):
            mask = "global" if has_sliding_in_stack else "causal"
            return mask, None, has_sliding_in_stack
        if _mixer_kind_for(lt):
            # A token-MIXER (linear_attention / lightning / recurrent) replaces
            # attention — it carries no attention mask (drawn as its own cell).
            return "causal", None, False
        unknown.add(lt)
        return "causal", None, False
    if sliding_window_pattern and sliding_window:
        # Every Nth layer is full; rest are sliding.
        if (i + 1) % sliding_window_pattern == 0:
            return "global", None, True
        return "sliding", sliding_window, False
    if sliding_window:
        # Qwen: the bottom ``max_window_layers`` layers use full attention; the
        # rest slide.  (HF: SWA applies only where layer_idx >= max_window_layers.)
        if max_window_layers and i < max_window_layers:
            return ("global" if has_sliding_in_stack else "causal"), None, has_sliding_in_stack
        return "sliding", sliding_window, False
    return "causal", None, False


def _layer_types_from_compress_ratios(compress_ratios: Any, num_layers: int) -> list[str]:
    """DeepSeek-V4 style compress ratios are structural layer-type data.

    Public configs declare ``compress_ratio=0`` for SWA, ``4`` for compressed
    sparse attention (CSA), and ``128`` for hierarchical compressed attention
    (HCA).  Preserve unknown positive ratios as compressed sparse variants
    rather than warning as an unknown mask.
    """
    if not isinstance(compress_ratios, (list, tuple)):
        return []
    values = list(compress_ratios)
    if num_layers and len(values) > num_layers:
        values = values[:num_layers]
    out: list[str] = []
    for raw in values:
        try:
            ratio = int(raw)
        except (TypeError, ValueError):
            out.append(str(raw))
            continue
        if ratio == 0:
            out.append("sliding_attention")
        elif ratio == 128:
            out.append("heavily_compressed_attention")
        else:
            out.append("compressed_sparse_attention")
    return out


def _compress_ratio_for_layer(i: int, compress_ratios: Any, layer_types: list[str]) -> int | None:
    if isinstance(compress_ratios, (list, tuple)) and i < len(compress_ratios):
        try:
            ratio = int(compress_ratios[i])
        except (TypeError, ValueError):
            ratio = 0
        return ratio or None
    if layer_types and i < len(layer_types):
        lt = layer_types[i]
        if _is_compressed_sparse_label(lt):
            return 4
        if _is_heavily_compressed_label(lt):
            return 128
    return None


# ---------------------------------------------------------------------------
# Per-layer TYPE-SCHEDULE normalizers (U3) — one engine, six spellings.
# Each config dialect below answers the SAME question ("which type is layer i?")
# in a different vocabulary; all normalize into the canonical layer_types list
# (mixer/mask dimension) or the MoE membership list the per-layer walk already
# consumes.  The vocabulary (fields, forms, value maps) lives in
# everchanging/transformer/layer_schedules.yaml — data, not code.
# ---------------------------------------------------------------------------

def _mixer_kind_for(layer_type) -> str | None:
    """Return the safest IR kind justified by a scheduled mixer token.

    A schedule proves placement and its declared type name, not the internal
    operation graph. Source-unresolved types therefore resolve to an opaque
    ``declared_*`` kind. ``gated_delta`` is the pre-existing legacy path.
    """
    if not isinstance(layer_type, str):
        return None
    return _MIXER_KINDS.get(layer_type.lower())


def _mixer_variant(mixer_kind: str | None) -> dict | None:
    """The variant card for a scheduled token-mixer layer."""
    if mixer_kind == "gated_delta":
        return {
            "short": "Gated DeltaNet",
            "tag": "linear recurrent mixer",
            "label": ["Gated DeltaNet", "Token Mixer"],
            "title": "Gated DeltaNet token mixer",
            "desc": (
                "Causal depthwise convolution feeds a gated delta-rule recurrence; "
                "cached decoding switches to the recurrent update path."
            ),
        }
    if mixer_kind == "declared_lightning_mixer":
        return {
            "short": "Lightning Attention",
            "tag": "declared mixer; internals unresolved",
            "label": ["Lightning Attention", "Token Mixer"],
            "title": "Lightning attention (config-declared)",
            "desc": (
                "The per-layer config schedule declares a Lightning-attention "
                "mixer here. Its operation graph was not resolved from source, "
                "so no kernel, state, mask, or positional mechanism is invented."
            ),
        }
    if mixer_kind == "declared_recurrent_mixer":
        return {
            "short": "Recurrent mixer",
            "tag": "declared mixer; internals unresolved",
            "label": ["Recurrent block", "Token Mixer"],
            "title": "Recurrent mixer (config-declared)",
            "desc": (
                "The per-layer config schedule declares a recurrent mixer here. "
                "Its recurrence, convolution, state, and gating graph were not "
                "resolved from source and are intentionally not drawn."
            ),
        }
    return None


def _schedule_alias(token) -> str:
    """Normalize one raw per-layer schedule token (int / string) to its canonical
    layer_types value via layer_schedules.yaml value_aliases (identity when no
    alias is declared)."""
    key = str(token).strip().lower()
    return _LAYER_SCHEDULES["value_aliases"].get(key, key)


def _expand_nested_tile(raw) -> list[str] | None:
    """gpt-neo ``attention_types = [[["global","local"], N], …]`` -> the flat
    per-layer token list (the pattern repeated N times, concatenated).  None when
    the value isn't the nested ``[[pattern, repeat], …]`` form (None-on-doubt)."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    out: list[str] = []
    for item in raw:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            return None
        pattern, repeat = item
        if not isinstance(pattern, (list, tuple)) or not isinstance(repeat, int):
            return None
        for _ in range(int(repeat)):
            out.extend(_schedule_alias(t) for t in pattern)
    return out or None


def _fit_schedule(tokens, num_layers: int) -> list[str] | None:
    """Fit a per-layer token list to ``num_layers`` — exact, or truncate a longer
    list.  None when shorter than the stack (a partial schedule is doubt)."""
    if not num_layers or not tokens or len(tokens) < num_layers:
        return None
    return list(tokens[:num_layers])


def _normalize_layer_schedule(text_cfg, num_layers: int, sliding_window):
    """Normalize the per-layer TYPE-SCHEDULE spellings into the canonical
    ``layer_types`` list — the CONFIG channel, consulted ONLY when the canonical
    ``layer_types`` list is absent (so families that already build it stay
    byte-identical).  Returns ``(layer_types, source_field)`` or ``(None, None)``;
    None-on-doubt so an unresolvable shape never fabricates a schedule."""
    sched = _LAYER_SCHEDULES
    # value_list (attn_type_list): the field IS the per-layer list.
    for field in sched["value_list_fields"]:
        raw = _g(text_cfg, field)
        if isinstance(raw, (list, tuple)) and raw:
            fitted = _fit_schedule([_schedule_alias(t) for t in raw], num_layers)
            if fitted is not None:
                return fitted, field
    # pattern_tile (block_types): a short pattern tiled to num_layers.
    for field in sched["pattern_tile_fields"]:
        raw = _g(text_cfg, field)
        if isinstance(raw, (list, tuple)) and raw and num_layers:
            pat = [_schedule_alias(t) for t in raw]
            tiled = (pat * (num_layers // len(pat) + 1))[:num_layers]
            # Griffin/Hawk hybrid: the "attention" blocks are LOCAL (windowed)
            # when the config declares a window (recurrentgemma
            # attention_window_size) — only in a stack that also has recurrent
            # mixer layers, so the plain full-attention "attention" label of a
            # non-hybrid stack is never touched.
            if sliding_window and any(_mixer_kind_for(t) for t in tiled):
                tiled = ["sliding_attention" if t in ("attention", "full_attention")
                         else t for t in tiled]
            return list(tiled), field
    # nested_tile (attention_types): [[pattern, repeat], …].
    for field in sched["nested_tile_fields"]:
        expanded = _expand_nested_tile(_g(text_cfg, field))
        fitted = _fit_schedule(expanded, num_layers) if expanded else None
        if fitted is not None:
            return fitted, field
    # dense_interval (dense_attention_every_n_layers): i % N == 0 -> dense.
    for field in sched["dense_interval_fields"]:
        n = _g(text_cfg, field)
        if isinstance(n, int) and n > 0 and num_layers:
            on, off = sched["dense_interval_on"], sched["dense_interval_off"]
            return [on if i % n == 0 else off for i in range(num_layers or 0)], field
    return None, None


def _moe_routing(cfg: Any, context=None, path: tuple = ()) -> dict | None:
    """Collect the MoE router knobs that decide *how* experts get picked.

    Config strings first (DeepSeek/Kimi declare the full set), then the CODE
    channel fills/overrides the two facts modern checkpoints omit: GLM-4.5
    copied DeepSeek-V3's routing CODE (``.sigmoid()`` + ``e_score_correction_bias``)
    but not its ``scoring_func``/``topk_method`` STRINGS, so the string reader
    drew softmax and dropped the bias.  Code is the enacted truth: it decides the
    score transform and the aux-loss-free bias; a declared string that agrees is
    confirmation, one that disagrees loses to the code (with a recorded note).
    ``None`` when neither channel declares anything.

    U2-R7: each declared knob is CONSUMED into its routing fact
    (``decoder.ffn.routing_<field>`` — the dict the render reads); ``path``
    locates ``cfg`` in its document (the caller's text-scope path)."""
    def _knob(field):
        res = _config_access.resolve(
            cfg, field, _ALIASES.get(field, ()), path=tuple(path))
        if res.ambiguous:
            return None
        return res.consume_decision(
            mechanism="moe_routing", fact_owner="decoder.ffn",
            fact_key=f"routing_{field}",
            reader="adapters.transformer.parser._moe_routing").value
    routing = {
        "scoring_func":          _knob("scoring_func"),          # sigmoid | softmax
        "topk_method":           _knob("topk_method"),           # noaux_tc, group_limited_greedy, ...
        "n_group":               _knob("n_group"),               # expert groups (node-limited routing)
        "topk_group":            _knob("topk_group"),            # groups kept per token
        "norm_topk_prob":        _knob("norm_topk_prob"),        # renormalize the top-k gate weights
        "routed_scaling_factor": _knob("routed_scaling_factor"),  # scale on routed-expert output
    }
    routing = {k: v for k, v in routing.items() if v is not None}

    code = _code_router(cfg, context)
    if code is not None:
        # Score transform: code decides; a declared string only confirms.
        if code.scoring_fn:
            declared = routing.get("scoring_func")
            if declared and str(declared).lower() != code.scoring_fn:
                routing["_scoring_declared"] = declared      # disagreement note (audit)
            routing["scoring_func"] = code.scoring_fn
        # Aux-loss-free bias correction: code proof (e_score_correction_bias) OR
        # a declared ``noaux_tc``.  A resolved boolean the render reads directly,
        # so a checkpoint that enacts the bias without the string still draws it.
        if code.bias_correction or routing.get("topk_method") == "noaux_tc":
            routing["bias_correction"] = True
        if code.sparsemixer:
            routing["sparsemixer"] = True
        # The score transform is drawn as its OWN node only when the code runs it
        # BEFORE the top-k (Mixtral/GLM/DSV3) — a node before top-k would misdraw
        # gpt-oss/Granite, which top-k the raw logits and softmax the winners.
        if code.scoring_fn and code.scoring_before_topk:
            routing["scoring_before_topk"] = True

    # BOTH channels silent on the score transform while the modeling source IS
    # installed ⇒ an extractor/vocabulary gap, not an honest absence — stamp
    # the ambiguity envelope the BLOCKING evidence_ambiguity net reads (the
    # honest replacement for the old render-side `or "softmax"` assertion).
    # No source at all stays exempt (oracle_missing discipline).
    if "scoring_func" not in routing and code is None:
        try:
            has_source = bool(_source_files(cfg, context))
        except Exception:
            has_source = False
        if has_source:
            routing["evidence"] = {
                "status": "ambiguous", "component": "router",
                "reason": "score transform not declared in config and no "
                          "router class resolved from the installed source",
            }

    return routing or None


def _norm_kind_evidence(
        cfg: Any, explicit_norm_type: Any = None, context=None) -> str | None:
    """The norm kind from EVIDENCE only — None when nothing states it (the
    caller chooses its default and KNOWS it is a default).  Channel order:

    1. an explicit ``rmsnorm`` bool / ``norm_type`` config declaration;
    2. exact constructed normalization primitives on the selected decoder
       block, classified from an exact framework protocol or implementation
       MATH — the only channel that never lies, so it outranks BOTH eps spellings:
       PhiMoE/Persimmon construct ``nn.LayerNorm`` while carrying
       ``rms_norm_eps`` (the RMS spelling lies about the kind), and T5 carries
       ``layer_norm_epsilon`` while ``T5LayerNorm`` computes a variance-only
       rescale (RMS).  The primitive classifier also maps exact torch import
       targets (``nn.LayerNorm``/``nn.RMSNorm``) as fixed library math —
       reading the library protocol, not a model class spelling;
    3. ``rms_norm_eps`` spelling — RMS when no source math is readable;
    4. the ``layer_norm_eps*`` spelling hint.
    """
    path = (
        tuple(context.selected_config_paths.get("transformer.main", ()))
        if context is not None else ())
    return _norm_kind_evidence_src(
        cfg, explicit_norm_type, context, config_path=path)[0]


def _norm_kind_evidence_src(cfg: Any, explicit_norm_type: Any = None,
                            context=None, eps_fact: tuple | None = None,
                            config_path=()) -> tuple:
    """``(kind|None, (status, source)|None)`` — the kind PLUS which channel
    decided it, for the FactLedger (U2). Channel order as documented above.

    U2-R7: when the caller names ``eps_fact`` (the ``(fact_owner, fact_key)``
    its norm-kind fact records under — the main transformer parse), the eps
    spellings are CONSUMED into that target; with no target (the encoder
    panel's post-parse advisory call, whose parse already consumed the same
    occurrences once) they stay plain inspected reads — one occurrence is
    never consumed twice in a single parse."""
    # Both eps spellings are read UP FRONT: they are real config facts (the
    # epsilon in use) and must record their access for the ownership audit even
    # when a higher channel (math) decides the KIND before the spelling hint.
    if eps_fact is not None:
        _eps_owner, _eps_key = eps_fact
        def _eps(canonical, aliases=()):
            res = _config_access.resolve(cfg, canonical, aliases)
            if res.ambiguous:
                return None
            return res.consume_decision(
                mechanism="norm_kind_eps", fact_owner=_eps_owner,
                fact_key=_eps_key,
                reader="adapters.transformer.parser."
                       "_norm_kind_evidence_src").value
        rms_eps = _eps("rms_norm_eps")
        ln_eps = _eps("layer_norm_epsilon")
        ln_eps2 = _eps("layer_norm_eps", ("layernorm_epsilon",))  # chatglm spelling
    else:
        rms_eps = _g(cfg, "rms_norm_eps")
        ln_eps = _g(cfg, "layer_norm_epsilon")
        ln_eps2 = _g(cfg, "layer_norm_eps") or _g(cfg, "layernorm_epsilon")  # chatglm spelling
    # Keep declarations visible to the access ledger, but do not let a
    # spelling select a primitive.  The exact block's constructed/called norm
    # is the authority.
    _g(cfg, "rmsnorm")
    _ = explicit_norm_type
    norm_result = _decoder_norm_result(
        context, config_path=tuple(config_path))
    if norm_result is not None and norm_result.status == "resolved":
        return norm_result.value, (
            "code_proven", "decoder_norm_kind_for_path")
    # Readable source plus a typed ambiguous/incomplete exact-owner result is
    # evidence AGAINST trusting a field-name heuristic.  Keep the kind unknown;
    # only a genuinely missing source may fall through to the legacy spelling
    # hint while U3-G finishes the remaining address boundary.
    if context is not None and context.source_bundle.files \
            and norm_result is not None:
        return None, (
            "ambiguous",
            f"decoder_norm_kind_for_path:{norm_result.status}")
    # Source absence does not make field spelling architectural evidence.
    # The reads remain classified migration debt for U7.
    _ = (rms_eps, ln_eps, ln_eps2)
    return None, None


def _rope_dim(rotary_pct, rotary_dim, partial_rotary_factor, head_dim) -> int | None:
    """Compute the actual rotary dim from any of the config flavours."""
    if rotary_dim:
        return int(rotary_dim)
    if rotary_pct and head_dim:
        return int(head_dim * float(rotary_pct))
    if partial_rotary_factor and head_dim:
        return int(head_dim * float(partial_rotary_factor))
    return None


def _position_for_layer(evidence, *, mixer_kind: str | None) -> tuple[str, str]:
    """Project model evidence onto one concrete layer without moving altitudes.
    An opaque scheduled mixer has unknown internals, so model-level attention
    evidence must not be projected into it. The legacy gated-delta path may
    select a proven ``none`` mechanism as before."""
    if mixer_kind and mixer_kind != "gated_delta":
        return "unknown", "none"
    if evidence.status == "proven":
        mechanisms = list(evidence.mechanisms)
        if mixer_kind == "gated_delta":
            selected = next((item for item in mechanisms if item.kind == "none"), None)
            if selected is not None:
                return selected.kind, selected.application
        # Attention-stage mechanisms take precedence on the attention card.  A
        # model may independently add an absolute position vector before the
        # stack; that operation remains represented by the model-level blocks.
        selected = next((item for item in mechanisms
                         if item.kind in {"rope", "alibi", "none"}), None)
        if selected is None and mechanisms:
            selected = mechanisms[0]
        if selected is not None:
            return selected.kind, selected.application
    if evidence.status == "ambiguous":
        return "unknown", "none"
    return "unknown", "none"


def _last_matching_layer(layer_types, i: int, first_shared: int) -> int | None:
    """For cross-layer KV sharing: most recent non-shared layer of the same type."""
    if not layer_types or i >= len(layer_types):
        return None
    target_type = layer_types[i]
    for j in range(min(first_shared, len(layer_types)) - 1, -1, -1):
        if layer_types[j] == target_type:
            return j
    return None


def _cross_attention_states_side_block(source_kind: str = "vision",
                                       encoder_type: str | None = None,
                                       feeds: str = "attn") -> dict:
    """Layer-local external states read by cross-attention layers.

    ``source_kind`` follows the SAME evidence that words ``cross_kv_source``:
    a vision sibling ⇒ projected image states; a declared encoder-role
    composite slot (MusicGen's t5) ⇒ the encoder's prompt states.  ``feeds``
    targets the ADDITIVE cross sublayer's own block when one exists."""
    if source_kind == "conditioning":
        return {
            "id": "cross_attention_states",
            "role": "conditioning",
            "kind": "conditioning",
            "diffusion_stage": "cross_attention",
            "lane": "external_left",
            "feeds": feeds,
            "offset_y": 0,
            "label": ["Encoded prompt", "states"],
            "title": "Encoded prompt states",
            # U2-R9: identity-free structural prose (name-blind law) — the
            # slot's declared type is a display label on the encoder panel.
            "description": (
                "encoder_outputs: the conditioning encoder's "
                "output states (see the prompt-encoder panel); this tensor "
                "supplies K/V to the decoder's cross-attention layers."
            ),
            "view": "conditioning_path",
            "w": 250,
            "h": 50,
            "font": 15,
        }
    return {
        "id": "cross_attention_states",
        "role": "vision",
        "kind": "vision",
        "lane": "external_left",
        "feeds": "attn",
        "offset_y": 0,
        "label": ["Projected image", "states"],
        "title": "Projected image states",
        "description": (
            "cross_attention_states: vision_model(pixel_values) -> multi_modal_projector; this tensor supplies K/V to the selected decoder cross-attention layer."
        ),
        "view": "vision_path",
        "w": 250,
        "h": 50,
        "font": 15,
    }
