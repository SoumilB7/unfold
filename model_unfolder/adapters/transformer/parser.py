"""The transformer-LLM parser — the only adapter.

There are no per-family adapters and no "supported model" gate.  Every
transformer-LLM config flows through ``parse()``; the IR is derived from
the config fields actually present.

Detection is config-driven:

* ``use_parallel_residual`` / ``parallel_attn``  → parallel-residual layer
* ``q_lora_rank`` / ``kv_lora_rank``             → MLA attention
* ``multi_query``                                → explicit MQA
* ``layer_types`` list                           → per-layer mask interleave
* ``sliding_window`` / ``sliding_window_pattern``→ sliding-window mask
* ``no_rope_layer_interval``                     → NoPE every Nth layer
* ``first_k_dense_replace`` / ``moe_layer_freq`` → MoE / dense interleave
* ``interleave_moe_layer_step``                  → Llama 4-style MoE interleave
* ``enable_moe_block``                           → explicit MoE gate
* ``attn_logit_softcapping`` / ``final_logit_softcapping`` → softcap extras
* ``query_pre_attn_scalar``                      → Q pre-scale extra
* ``use_qk_norm`` / ``qk_norm``                  → QK-norm flag
* ``rope_parameters`` / ``rope_scaling``         → RoPE scaling extras
* ``rotary_pct`` / ``rotary_dim``                → partial RoPE
* ``num_kv_shared_layers``                       → cross-layer KV-share edges
* ``num_global_key_value_heads`` / ``global_head_dim`` → per-layer dual KV
* ``hidden_size_per_layer_input``                → Per-Layer Embedding side blocks
* ``num_local_experts`` / ``n_routed_experts``   → MoE FFN
* ``n_shared_experts`` / ``num_shared_experts``  → shared experts
* ``clip_qkv``                                   → attention extras
* ``cross_attention_layers``                     → vision side-attention layers
* ``num_nextn_predict_layers`` / ``num_mtp_layers`` → Multi-Token Prediction head stack

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


def _code_layer_topology(cfg: Any, context=None) -> dict | None:
    """The decoder layer's macro-topology (norm placement + parallel residual)
    READ FROM THE MODELING SOURCE — the code-based replacement for the
    ``layer_topology.yaml`` model_type table.  "code -> structure": where the
    norms sit and whether attention ∥ FFN is wiring the forward() states, not a
    per-family lookup.  Returns ``{"norm_placement", "parallel_residual"}`` or
    None (no source / no layer class found → caller falls back to the table
    cache, then the safe pre/sequential default).  Best-effort, never raises into
    the parse."""
    try:
        from ...evidence.patterns import decoder_layer_topology_from_files
        files = _source_files(cfg, context)
        return decoder_layer_topology_from_files(files)
    except Exception:
        return None


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


def _ffn_mechanism_result(context=None, *, config_path=()):
    """One call-local exact-owner ordinary FFN result."""
    if context is None:
        return None
    from ...evidence.ffn_mechanism import decoder_ffn_mechanism_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.ffn.mechanism",
        config_path,
        lambda: decoder_ffn_mechanism_for_path(
            context.program_index(),
            context.source_bundle,
            config_path,
            allow_root_stage=True,
        ),
    )


def _code_ffn_gated(
        cfg: Any, context=None, *, config_path=()) -> bool | None:
    """Gate structure from the exact selected decoder FFN occurrence."""
    result = _ffn_mechanism_result(
        context, config_path=config_path)
    return result.value.gated \
        if result is not None and result.status == "resolved" else None


def _code_ffn_storage_mode(
        cfg: Any, context=None, *, expected_gated=None,
        config_path=()) -> str | None:
    """FFN projection STORAGE read from the modeling source: split gate/up/down
    modules vs a fused ``gate_up_proj`` that is chunked in forward (Mixtral /
    DeepSeek-V3 naive MoE / gpt-oss experts).  A whiteboard-equivalent split
    drawing is NOT code-faithful when the source stores fused — the drill must
    show the fused projection + split (anti-overlook rule).  None keeps the
    default split rendering (unproven storage stays the conventional shape only
    for the ALREADY-drawn gated structure; the gate-or-not fact itself remains
    evidence/tri-state via ``_code_ffn_gated``)."""
    result = _ffn_mechanism_result(
        context, config_path=config_path)
    if result is None or result.status != "resolved":
        return None
    if expected_gated is not None \
            and result.value.gated != expected_gated:
        return None
    return result.value.projection_mode


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


def _code_attention_fused_qkv(
        cfg: Any, context=None, *, config_path=()) -> bool | None:
    """Owner-qualified Q/K/V storage; uncertainty never becomes split/fused."""
    result = _attention_storage_result(context, config_path)
    if result is None or result.status != "resolved":
        return None
    return result.value == "fused_qkv"


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


def _code_qk_norm(cfg: Any, context=None, *, config_path=()):
    """Positive Q/K normalization from the exact selected decoder owner.

    Missing or incomplete source evidence returns ``None`` so the checkpoint's
    declaration remains the fallback.  It is never converted to code-proven
    absence.
    """
    result = _qk_norm_result(context, config_path=config_path)
    return result.value if result is not None and result.status == "resolved" \
        else None


def _resolve_qk_norm_layers(
        code_ev, cfg, declared: bool, num_layers: int, *,
        context=None, config_path=()) -> list[bool]:
    """Per-layer Q/K-norm facts from the code evidence.

    Code-first resolution: unconditional construction → True everywhere (the
    config is not consulted); gated → AND of the values of the exact config
    fields the code itself names, read from THIS checkpoint (per-layer when the
    code indexes by layer index); no positive source proof / unresolvable gate
    → the declared spelling.  Source silence never proves absence."""
    n = max(int(num_layers or 0), 0)
    if code_ev is None:
        return [declared] * n
    if code_ev.present is True:
        return [True] * n
    per_layer = [True] * n
    for atom in code_ev.gate:
        if tuple(atom.config_path[:-1]) != tuple(config_path):
            return [declared] * n
        if context is None:
            missing = object()
            raw = _g(cfg, atom.field, missing)
            if raw is missing:
                return [declared] * n
        else:
            resolution = _config_access.resolve(
                cfg, atom.field, (), path=tuple(config_path))
            if resolution.ambiguous or not resolution.present:
                return [declared] * n
            raw = resolution.consume_decision(
                mechanism="qk_norm_gate",
                fact_owner="decoder.attention",
                fact_key="qk_norm",
                reader="adapters.transformer.parser._resolve_qk_norm_layers",
            ).value
        if atom.per_layer:
            if not isinstance(raw, (list, tuple)) or len(raw) < n:
                return [declared] * n
            per_layer = [p and bool(raw[i]) for i, p in enumerate(per_layer)]
        else:
            if raw is None:
                return [declared] * n
            per_layer = [p and bool(raw) for p in per_layer]
    return per_layer


def _code_parallel_norm_count(cfg: Any, context=None):
    """Distinct INPUT norms a parallel-residual decoder layer applies (1 shared
    / 2 separate), READ FROM THE CODE dataflow — fixes GPT-NeoX's two norms
    drawn as one shared.  None when unresolvable (Falcon conditional) → the
    caller defaults to 1.  Best-effort, never raises."""
    try:
        from ...evidence.patterns import decoder_parallel_norm_count_from_files
        return decoder_parallel_norm_count_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _projection_bias_result(context, mechanism, config_path):
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
            allow_root_stage=True),
    )


def _code_attention_bias(cfg: Any, context=None, *, config_path=()):
    """Source-only bias of the exact Q/K/V projection occurrences.

    Config-dependent bias expressions intentionally abstain here and stay on
    the parser's owner-scoped config channel.
    """
    result = _projection_bias_result(context, "attention", config_path)
    return (
        result.value.value
        if result is not None and result.status == "resolved" else None
    )


def _code_moe_schedule(cfg: Any, context=None, num_layers: int = 0):
    """Per-layer MoE?/dense schedule READ FROM THE DECODER LAYER's CONSTRUCTION —
    the code-authoritative replacement for the config schedule flags: which
    layers build an experts class (name-independent) as their FFN field, gated
    per layer.  Returns ``list[bool]`` (len num_layers) or None on any doubt
    (hybrid SSM-MoE, exotic gate, no source) → caller falls back to config.
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


def _code_scores_scaled(cfg: Any, context=None) -> bool | None:
    """Whether the attention forward SCALES its scores, READ FROM THE MODELING
    SOURCE (``attention_score_scaling_from_files``) — the same verdict the
    encoder-tower path already draws (T5's raw ``QK^T``).  False ⇒ the forward
    performs no scale (folded into init); True/None keep the sqrt(dim)
    rendering.  Wiring this on the MAIN path removes the one asserted default
    that had a live, unread oracle.  Best-effort, never raises into the parse."""
    try:
        from ...evidence.patterns import attention_score_scaling_from_files
        return attention_score_scaling_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _code_mlp_bias(cfg: Any, context=None, *, config_path=()) -> bool | None:
    """Source-only bias of the exact ordinary-FFN projection occurrences."""
    result = _projection_bias_result(context, "ordinary_ffn", config_path)
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


def _code_intermediate_size(cfg: Any, context=None) -> int | None:
    """FFN intermediate width from the modeling source's OWN default expression
    when the config field is absent (GPT-J/GPT-2/CodeGen ``n_inner=None →
    4×n_embd``) — fixes the param undercount without a per-model table.
    Best-effort, never raises into the parse."""
    try:
        from ...evidence.patterns import decoder_intermediate_size_from_files
        return decoder_intermediate_size_from_files(
            _source_files(cfg, context), cfg, _ALIASES.get("intermediate_size", ()))
    except Exception:
        return None


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


def _code_ffn_activation(
        cfg: Any, context=None, *, config_path=()) -> str | None:
    """Config-silent FFN activation read from the exact modeling source."""
    result = _ffn_mechanism_result(
        context, config_path=config_path)
    return result.value.activation \
        if result is not None and result.status == "resolved" else None


def _code_lm_head_tying(cfg: Any, context=None) -> bool | None:
    """Unconditional manual lm_head↔embedding tying read from the source
    (legacy/remote-code idiom).  ``_tied_weights_keys`` is capability, never
    proof — the reader ignores it by design (U2 P2b)."""
    try:
        from ...evidence.patterns import lm_head_tying_from_files
        return lm_head_tying_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _code_ffn_activation_dispatch(
        cfg: Any, context=None, *, config_path=()) -> str | None:
    """The config field the FFN's ``ACT2FN[...]``/``get_activation(...)``
    dispatch reads (U2 P2c) — names the deciding field so a config-supplied
    activation can be recorded ``code_and_config``, not merely declared."""
    result = _ffn_mechanism_result(
        context, config_path=config_path)
    if result is None or result.status != "resolved":
        return None
    path = result.value.activation_config_path
    if not path or tuple(path[:-1]) != tuple(config_path):
        return None
    return path[-1]


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


def _nested(cfg: Any, key: str) -> Any:
    """Some configs (DBRX) nest fields under sub-dicts like ``attn_config``."""
    val = _g(cfg, key)
    return val if isinstance(val, dict) else {}


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

    # DBRX-style nested config dicts: pull through transparently.
    attn_cfg = _nested(text_cfg, "attn_config")
    ffn_cfg  = _nested(text_cfg, "ffn_config")

    from ...evidence import config_access as _config_access

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
        code_inter = _code_intermediate_size(text_cfg, context)
        if code_inter:
            intermediate_size = code_inter
    # DBRX-style: activation lives in a nested dict like ``ffn_act_fn = {"name": "silu"}``.
    activation_raw = consume("hidden_act", fact_owner="decoder.ffn", fact_key="activation")
    if isinstance(activation_raw, dict):
        activation_raw = activation_raw.get("name")
    if activation_raw is None:
        nested_act = _g(ffn_cfg, "ffn_act_fn")
        if isinstance(nested_act, dict):
            activation_raw = nested_act.get("name")
    _activation_status, _activation_src = "config_declared", "hidden_act"
    if activation_raw is None:
        # U2 P3b: the T5-family declaration — ``feed_forward_proj`` names the
        # activation (with an optional "gated-" prefix owned by the gate
        # decision below).  Un-ignored: a positive declaration, never noise.
        # U2-R7: ONE consumption for the occurrence, into its PRIMARY decision
        # (activation); the gate decision below re-inspects the same value.
        _ffp_for_act = consume("feed_forward_proj",
                               fact_owner="decoder.ffn", fact_key="activation")
        if isinstance(_ffp_for_act, str) and _ffp_for_act:
            activation_raw = _ffp_for_act.lower()
            if activation_raw.startswith("gated-"):
                activation_raw = activation_raw[len("gated-"):]
            _activation_status, _activation_src = (
                "config_declared", "feed_forward_proj")
    if activation_raw is None:
        activation_raw = _code_ffn_activation(
            text_cfg, context, config_path=_text_path)
        _activation_status, _activation_src = (
            "code_proven", "decoder_ffn_mechanism_for_path")
        if activation_raw is None:
            # U2 P3b class_default tier (the hydration channel): the installed
            # config CLASS's own activation default decides before the typed
            # unknown (t5-base: dense_act_fn='relu' — the dense-relu truth).
            _cd_for_act = _fact_class_defaults
            _cd_act = next((_cd_for_act.get(s) for s in _spellings("hidden_act")
                            if _cd_for_act.get(s) is not None), None)
            _cd_src = "hidden_act family"
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
    elif _activation_status == "config_declared" and _activation_src == "hidden_act":
        # U2 P2c: the config supplied the value — when the source also proves
        # the ACT2FN/get_activation dispatch, the fact is CODE-AND-CONFIG:
        # code proves an activation applies and NAMES the deciding field,
        # config supplies which (the endorsed envelope's fourth tier).
        _dispatch_field = _code_ffn_activation_dispatch(
            text_cfg, context, config_path=_text_path)
        if _dispatch_field:
            _activation_status = "code_and_config"
            _activation_src = f"ACT2FN[config.{_dispatch_field}]"
    # U2 default-kill: when neither config nor code names the activation it is
    # a typed UNKNOWN (None), never a silent ``silu`` — the render tier shows
    # an unresolved activation instead of a confident label.
    activation_defaulted = activation_raw is None
    activation = activation_raw.lower() if isinstance(activation_raw, str) else None
    if activation_defaulted:
        _activation_status, _activation_src = _unknown_status, None
    _note_fact("decoder.ffn", "activation", activation,
               _activation_status, _activation_src)
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
    # consulted ONLY when no canonical layer_types list exists (so families that
    # build it stay byte-identical).  A remote-code model whose modeling oracle
    # is missing (MiniMax-Text-01 / Phi-3-small) is served this config-declared
    # schedule rather than a fabricated uniform tower.
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
    residual_multiplier = get("residual_multiplier")
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
    _code_topo = _code_layer_topology(text_cfg, context)
    # FFN gating READ FROM THE MLP's forward() (gate_mul present?) — code wins;
    # a gate-family activation string is the config-derived second channel; NO
    # channel ⇒ typed unknown (None) drawn as the honest undeclared-FFN block,
    # never derived from norm_kind (the census cascade, killed in U2).
    _code_gated = _code_ffn_gated(
        text_cfg, context, config_path=_text_path)
    # Some configs declare gate-ness directly (T5's is_gated_act /
    # feed_forward_proj).  These are semantic declarations, unlike a plain
    # activation such as ``silu`` which does NOT by itself prove a third gate
    # projection.  Source remains authoritative when present.
    # U2-R7: ``is_gated_act`` is consumed into the gate decision.  When it is
    # absent, ``feed_forward_proj`` DECIDES gate-ness and that deciding read
    # is consumed too — one declaration lawfully feeding two facts
    # (activation above, gated here); when is_gated_act is present the
    # feed_forward_proj glance stays an inspection (redundant, not deciding).
    _declared_gated = consume("is_gated_act",
                              fact_owner="decoder.ffn", fact_key="gated")
    if _declared_gated is not None:
        # is_gated_act decided; a feed_forward_proj beside it is a redundant
        # co-declaration of the same mechanism — a conscious scoped ignore,
        # never accessed-but-unconsumed debt.
        with _config_access.config_container(_text_path, obj=text_cfg):
            _ffp_res = _config_access.resolve(text_cfg, "feed_forward_proj", ())
        if _ffp_res.state == "present":
            _ffp_res.ignore(reason="redundant co-declaration beside "
                                   "is_gated_act — the consumed occurrence "
                                   "decided the gate fact")
        _feed_forward_proj = _ffp_res.value if _ffp_res.state == "present" \
            else None
        _config_gated = bool(_declared_gated)
        _config_gated_source = "is_gated_act"
    else:
        _feed_forward_proj = consume("feed_forward_proj",
                                     fact_owner="decoder.ffn",
                                     fact_key="gated")
        if isinstance(_feed_forward_proj, str):
            _config_gated = "gated" in _feed_forward_proj.lower()
            _config_gated_source = "feed_forward_proj"
        else:
            _config_gated = None
            _config_gated_source = None
    # U2 P3b class_default tier: the installed config CLASS's own gate
    # declaration (T5Config: is_gated_act=False / feed_forward_proj='relu' —
    # t5-base omits both keys yet IS dense-relu).  Sits below the checkpoint
    # declaration, above the derived activation-family signal.
    _clsdef_gated = None
    _clsdef_gated_source = None
    if _config_gated is None:
        _cd_for_gate = _fact_class_defaults
        _cd_gate_flag = _cd_for_gate.get("is_gated_act")
        _cd_gate_ffp = _cd_for_gate.get("feed_forward_proj")
        if _cd_gate_flag is not None:
            _clsdef_gated = bool(_cd_gate_flag)
            _clsdef_gated_source = ("installed config-class default "
                                    "(is_gated_act)")
        elif isinstance(_cd_gate_ffp, str) and _cd_gate_ffp:
            _clsdef_gated = "gated" in _cd_gate_ffp.lower()
            _clsdef_gated_source = ("installed config-class default "
                                    "(feed_forward_proj)")
    if _code_gated is not None:
        ffn_gated = bool(_code_gated)
    elif _config_gated is not None:
        ffn_gated = _config_gated
    elif _clsdef_gated is not None:
        ffn_gated = _clsdef_gated
    else:
        ffn_gated = _is_gated(activation, norm_kind, None)
    # ``decoder.ffn.gated`` names the ordinary/shared FFN mechanism.  A routed
    # expert proof must never be laundered into this owner, and a routed-only
    # block has no ordinary gate fact to record.  The exact reader result
    # remains in ``context.reader_results`` as the typed abstention.
    if ffn_gated is not None:
        _note_fact("decoder.ffn", "gated", ffn_gated,
                   "code_proven" if _code_gated is not None
                   else "config_declared" if _config_gated is not None
                   else "class_default" if _clsdef_gated is not None
                   else "derived",
                   source=("decoder_ffn_mechanism_for_path"
                           if _code_gated is not None
                           else _config_gated_source
                           if _config_gated is not None
                           else _clsdef_gated_source
                           if _clsdef_gated is not None
                           else "explicit GLU activation name (hidden_act)"))
    _code_storage_mode = _code_ffn_storage_mode(
        text_cfg, context, expected_gated=_code_gated,
        config_path=_text_path)
    _code_fused_qkv = _code_attention_fused_qkv(
        text_cfg, context, config_path=_text_path)
    _code_position_evidence = _code_position(cfg, context)
    # Projection storage is mechanism-scoped too.  Do not manufacture an
    # ordinary ``split`` fact when only a routed-expert mechanism exists.
    if _code_storage_mode is not None:
        _note_fact("decoder.ffn", "projection_mode",
                   _code_storage_mode, "code_proven",
                   source="decoder_ffn_mechanism_for_path")
    _note_fact("decoder.attention", "projection_mode",
               "fused_qkv" if _code_fused_qkv else "split",
               "code_proven" if _code_fused_qkv is not None else "asserted",
               source=("decoder_attention_projection_storage_for_path"
                       if _code_fused_qkv is not None
                       else "split convention kept (storage unproven)"))
    # Placement (B2, U2 default-kill), two unknown tiers:
    # * source ABSENT (oracle_missing) → typed "unknown": the layer draws its
    #   declared sublayers plus ONE pale "code-defined wiring" block (the
    #   tower_cell honest-unknown primitive ported to the main path) — never
    #   an asserted pre-norm shape at zero evidence.
    # * source PRESENT but the reader abstained on the idiom (T5's ModuleList
    #   sublayers, MusicGen's interleaved cross-attn) → AMBIGUOUS: the
    #   conventional pre cell stays DRAWN — the op/nested-conformance oracle
    #   still checks its norms/residual ops against the readable forward()
    #   (dropping them here made the net flag code-proven residual_adds as
    #   omitted) — recorded ambiguous, tagged asserted, and bannered.
    norm_placement = (_code_topo or {}).get("norm_placement")
    norm_placement_defaulted = not norm_placement
    _note_fact("decoder.layer", "norm_placement",
               norm_placement or ("pre" if _source_present else "unknown"),
               "code_proven" if norm_placement else _unknown_status,
               source=("decoder_layer_topology_from_files" if norm_placement
                       else "pre convention kept (reader abstained on the idiom)"
                       if _source_present else None))
    norm_placement_conventional = norm_placement_defaulted and _source_present
    if not norm_placement:
        norm_placement = "pre" if _source_present else "unknown"
    # Position scheme: configured source evidence asserts a mechanism when it
    # can (code_proven).  U2 P3a — the CONFIG-DECLARED fallback: when the code
    # channel is oracle_missing/ambiguous but the config itself declares RoPE
    # (a θ / a scaling dict, alias spellings in aliases.yaml), RoPE is drawn
    # as CONFIG-DECLARED — a "θ=… (config-declared)" chip states the tier,
    # replacing the honest-unknown banner.  A family convention is still not
    # evidence: no declaration ⇒ typed unknown, exactly as before.
    _position_mechanisms = list(_code_position_evidence.mechanisms)
    _declared_theta = get("rope_theta")
    # U2-R7: the rope container DICT occurrence itself feeds the uses_rope
    # decision — consumed once here; every later reader reuses this value.
    _declared_scaling = consume("rope_scaling",
                                fact_owner="decoder.attention", fact_key="rope")
    _rope_config_declared = False
    if _code_position_evidence.status == "proven":
        uses_rope = "rope" in _code_position_evidence.kinds
    elif _declared_theta is not None or isinstance(_declared_scaling, dict):
        uses_rope = True
        _rope_config_declared = True
        _note_fact("decoder.attention", "position", "rope", "config_declared",
                   "rope_theta" if _declared_theta is not None else "rope_scaling")
    else:
        uses_rope = False
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
    has_multi_query_flag = bool(_g(text_cfg, "multi_query")
                                or _g(text_cfg, "multi_query_attention"))  # chatglm spelling
    # The flag means "KV heads are shared", NOT "exactly one": ChatGLM declares
    # multi_query_attention: true AND multi_query_group_num: 2 (a 2-group GQA).
    # Only when NO explicit group count is declared does the flag default the
    # KV count to 1 (Falcon-7B / GPT-BigCode true MQA).
    # U6 owns head-sharing SEMANTICS and the evidence-strength migration; U2
    # keeps this behaviour-preserving precedence.
    if has_multi_query_flag and not get("num_key_value_heads"):
        num_kv_heads = 1
    # Hybrid linear-recurrent token mixers (for example a gated delta network)
    # carry geometry separate from the full-attention head fields.
    linear_num_k_heads = _g(text_cfg, "linear_num_key_heads")
    linear_num_v_heads = _g(text_cfg, "linear_num_value_heads")
    linear_k_head_dim = _g(text_cfg, "linear_key_head_dim")
    linear_v_head_dim = _g(text_cfg, "linear_value_head_dim")
    linear_conv_kernel = _g(text_cfg, "linear_conv_kernel_dim")
    attn_output_gate = _g(text_cfg, "attn_output_gate")
    _g(text_cfg, "output_gate_type")  # ownership acknowledged; source applies sigmoid
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
    no_rope_list         = _g(text_cfg, "no_rope_layers")   # materialized per-layer use_rope flags
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
    # Spelling read stays FIRST for config-ownership (all three aliases are
    # audited) and as the no-oracle fallback; the code evidence then decides.
    # U2-R7: the three rival spellings are ONE fact — resolved together (alias
    # law: equal = redundant, unequal = typed ambiguity that authors nothing)
    # and consumed into the qk_norm decision.
    _qk_res = _config_access.resolve(
        text_cfg, "use_qk_norm", ("qk_norm", "qk_layernorm"), path=_text_path)
    _qk_declared = None if _qk_res.ambiguous else _qk_res.consume_decision(
        mechanism="qk_norm", fact_owner="decoder.attention",
        fact_key="qk_norm",
        reader="adapters.transformer.parser.parse").value
    use_qk_norm = bool(_qk_declared)
    qk_norm_layers = _resolve_qk_norm_layers(
        _code_qk_norm(text_cfg, context, config_path=_text_path),
        text_cfg, use_qk_norm, num_layers,
        context=context, config_path=_text_path)

    # ---- Bias terms on the Q/K/V/O projections (Qwen2, GPT-2, Phi, ...) ----
    # CODE construction is authoritative (Bloom/Qwen2 hardcode
    # `nn.Linear(..., bias=True)` on QKV while declaring no `attention_bias`);
    # the config spelling is the declared channel behind it. U2 default-kill:
    # when BOTH are silent the bias is a typed UNKNOWN (None) — never a
    # silent False indistinguishable from proven-False.
    # U2-R7: consumed into the bias fact/spec — every alias spelling
    # (use_qkv_bias, add_qkv_bias, ...) resolves through this ONE read.
    _declared_attn_bias = consume("attention_bias",
                                  fact_owner="decoder.attention",
                                  fact_key="bias")
    if _declared_attn_bias is None:
        _declared_attn_bias = _g(attn_cfg, "attention_bias")
    _code_bias = _code_attention_bias(
        text_cfg, context, config_path=_text_path)
    # class_default tier: ``Linear(bias=config.attention_bias)`` reads the
    # config ATTRIBUTE at runtime — an absent key resolves to the installed
    # config class's default, real evidence (the hydration channel) and the
    # same value the hydrated embedded/sub-model rail sees (parity net).
    _cls_defaults = _fact_class_defaults
    if _code_bias is not None:
        use_attention_bias = bool(_code_bias)
        _note_fact("decoder.attention", "bias", use_attention_bias,
                   "code_proven", "decoder_attention_bias_for_path")
    elif _declared_attn_bias is not None:
        use_attention_bias = bool(_declared_attn_bias)
        _note_fact("decoder.attention", "bias", use_attention_bias,
                   "config_declared", "attention_bias")
    elif _cls_defaults.get("attention_bias") is not None:
        use_attention_bias = bool(_cls_defaults["attention_bias"])
        _note_fact("decoder.attention", "bias", use_attention_bias,
                   "class_default",
                   "installed config-class default (AutoConfig.for_model)")
    else:
        use_attention_bias = None
        _note_fact("decoder.attention", "bias", None, _unknown_status, None)
    # Code-proven scores-scaling verdict (False ⇒ raw QK^T, T5 family).  A
    # config-DECLARED scale is a stronger, load-bearing statement: when one
    # exists the declared float wins and the silent-scale read is ignored.
    code_scores_scaled = _code_scores_scaled(text_cfg, context)
    _note_fact("decoder.attention", "scores_scale",
               "declared" if (attention_multiplier is not None
                              or query_pre_attn_scalar) else
               ("unscaled (raw QK^T)" if code_scores_scaled is False
                else "sqrt(head_dim)"),
               "config_declared" if (attention_multiplier is not None
                                     or query_pre_attn_scalar)
               else "code_proven" if code_scores_scaled is not None
               else "asserted",
               source=("attention_multiplier/query_pre_attn_scalar"
                       if (attention_multiplier is not None or query_pre_attn_scalar)
                       else "attention_score_scaling_from_files"
                       if code_scores_scaled is not None
                       else "sqrt(dim) convention kept"))
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
    # Gemma-2's attention-logit softcap is a REAL op between QK^T and the
    # softmax (scores/cap → tanh → ×cap) — drawn as a node, not extras-only.
    attn_logit_softcap = consume("attn_logit_softcapping",
                                 fact_owner="decoder.attention",
                                 fact_key="logit_softcap")
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
        text_cfg, context, config_path=_text_path)
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
    # BLOOM-family topology switch: when TRUE the residual taps the POST-LN
    # output instead of the raw hidden state — surfaced as a layer annotation
    # (the tap is a wiring fact; False is the drawn default).
    residual_post_layernorm = bool(_g(text_cfg, "apply_residual_connection_post_layernorm"))

    # ---- Layer topology ----
    # Parallel residual: a config flag when the family TOGGLES it (Falcon
    # new_decoder_architecture / GPT-NeoX use_parallel_residual — gated inside an
    # `if`, so the config decides); else READ FROM the forward() when it is
    # UNCONDITIONAL structure with no flag (Cohere, GPT-J, Phi — all flagless,
    # all missed by the old model_type table, so all silently drawn sequential).
    # Distinct INPUT norms a parallel-residual layer applies, read from the code
    # dataflow: 1 = SHARED (GPT-J), 2 = SEPARATE (GPT-NeoX input+post norms) —
    # fixes the "two-norms-drawn-as-one" bug; None (Falcon conditional) → 1.
    parallel_norm_count = _code_parallel_norm_count(text_cfg, context) or 1
    # U2-R7: the two rival spellings (GPT-NeoX use_parallel_residual / Falcon
    # parallel_attn) are ONE declared fact — resolved together and consumed
    # into the layer-topology fact; disagreement is typed ambiguity that
    # authors nothing (the code channel then decides).
    _par_res = _config_access.resolve(
        text_cfg, "use_parallel_residual", ("parallel_attn",), path=_text_path)
    _par_declared = None if _par_res.ambiguous else _par_res.consume_decision(
        mechanism="residual_topology", fact_owner="decoder.layer",
        fact_key="parallel_residual",
        reader="adapters.transformer.parser.parse").value
    use_parallel_residual = bool(
        _par_declared or (_code_topo or {}).get("parallel_residual")
    )

    # ---- MoE ----
    num_experts         = consume("num_experts", fact_owner="decoder.ffn", fact_key="num_experts") or 0  # iteration count only — ambiguity already blocks
    num_experts_per_tok = consume("num_experts_per_tok", 0, fact_owner="decoder.ffn", fact_key="num_experts_per_tok") or 0
    num_shared_experts  = consume("num_shared_experts", 0, fact_owner="decoder.ffn", fact_key="num_shared_experts") or 0
    moe_intermediate_size = consume("moe_intermediate_size", 0, fact_owner="decoder.ffn", fact_key="moe_intermediate_size") or 0
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
    # U2-R7: both feed the per-layer dense/MoE schedule (FFNSpec.kind).
    first_k_dense       = consume("first_k_dense_replace",
                                  fact_owner="decoder.ffn",
                                  fact_key="moe_schedule") or 0
    moe_layer_freq      = consume("moe_layer_freq",
                                  fact_owner="decoder.ffn",
                                  fact_key="moe_schedule") or 1
    interleave_moe_step = _g(text_cfg, "interleave_moe_layer_step") or 0
    # Qwen3-MoE: MoE applies every ``decoder_sparse_step`` layers, except the
    # explicit ``mlp_only_layers`` which stay dense.
    decoder_sparse_step = _g(text_cfg, "decoder_sparse_step") or 0
    mlp_only_layers     = set(_g(text_cfg, "mlp_only_layers") or [])
    # Llama-4's explicit MoE-layer LIST — the code schedule reads its VALUE, so
    # record ownership here (the audit tracks _g access); also the direct config
    # fallback (layer i is MoE iff i in moe_layers) for a source-less parse.
    moe_layers_list     = _g(text_cfg, "moe_layers")
    # U3: step3 spells the same MoE-layer membership as a comma-STRING
    # (``moe_layers_enum = "4,5,…,60"``) — parse it into the SAME list the
    # ``moe_layers`` reader consumes so layers 0-3 stay dense instead of being
    # fabricated as MoE.  Only when the explicit list is absent.
    if moe_layers_list is None:
        _enum_moe = _moe_layers_from_enum(text_cfg)
        if _enum_moe is not None:
            moe_layers_list = _enum_moe
            _note_fact("decoder.ffn", "moe_schedule",
                       f"{len(_enum_moe)} MoE layers over {num_layers}",
                       "config_declared", "moe_layers_enum")
    moe_every_layer     = moe_active and not any([
        first_k_dense,
        interleave_moe_step,
        decoder_sparse_step and decoder_sparse_step > 1,
        mlp_only_layers,
    ])
    # CODE-AUTHORITATIVE per-layer MoE schedule (which layers build an experts
    # class as their FFN field) — read from the decoder layer's construction,
    # name-independent.  Supplies the per-layer SHAPE; ``moe_active``
    # (num_experts>0) stays the is-this-MoE-at-all geometry gate.  None when the
    # code can't resolve it (hybrid SSM-MoE / exotic gate / no source) → the
    # config ``_is_dense_at_layer`` path is used unchanged (the 12 agreeing
    # families stay byte-identical; llama-4 + ernie are the code-right fixes).
    code_moe_schedule   = _code_moe_schedule(text_cfg, context, num_layers) if moe_active else None
    # Router behaviour: gating fn, grouped/node-limited routing, top-k renorm,
    # routed-output scale (DeepSeek-V3, Kimi-K2, GLM, Qwen3-MoE).
    moe_routing = (_moe_routing(text_cfg, context, path=_text_path)
                   if moe_active else None)
    # gpt-oss clamps its SwiGLU activation to ±swiglu_limit — a Tier-3 property.
    activation_clip = consume("swiglu_limit", fact_owner="decoder.ffn",
                              fact_key="activation_clip")

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
        from ...evidence.patterns import decoder_cross_attention_all_layers_from_files
        _bundle = getattr(context, "source_bundle", None)
        _comp_files = getattr(_bundle, "component_files", {}) or {}
        _main_files = (_comp_files.get("decoder") or _comp_files.get("root")
                       or getattr(_bundle, "files", None))
        if decoder_cross_attention_all_layers_from_files(_main_files):
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
        else:
            attn_kind = _attention_kind(is_mla, num_heads, layer_kv_heads,
                                        has_multi_query_flag)
        # NoPE placement comes from the field the code actually indexes
        # (``self.use_rope = config.no_rope_layers[layer_idx]`` — truthy means
        # the layer USES rope), falling back to the interval rule with the
        # config class's own phase: ``use_rope = (i + 1) % interval != 0`` puts
        # NoPE at layers 3, 7, 11…, not 0, 4, 8….
        if isinstance(no_rope_list, (list, tuple)) and i < len(no_rope_list):
            is_nope = not bool(no_rope_list[i])
        else:
            is_nope = bool(no_rope_interval > 1 and (i + 1) % no_rope_interval == 0)
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

        position_mechanism = _position_for_layer(
            _code_position_evidence, mixer_kind=mixer_kind,
        )
        if (_rope_config_declared and position_mechanism[0] == "unknown"
                and not is_mixer):
            # U2 P3a: the config-declared RoPE fallback projects onto the
            # layer (chip carries the declared tier), replacing the
            # position-unresolved chip.
            position_mechanism = ("rope", "qk_rotation")
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
            rope=uses_rope and not is_mixer,
            position_kind=position_mechanism[0],
            position_application=position_mechanism[1],
            # U2 P3a: the declared-tier marker + θ ride the spec only on the
            # config-declared fallback (chip says "(config-declared)").
            position_declared=(_rope_config_declared
                               and position_mechanism[0] == "rope"),
            rope_theta_declared=(
                float(_declared_theta)
                if (_rope_config_declared and position_mechanism[0] == "rope"
                    and isinstance(_declared_theta, (int, float)))
                else None),
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
            output_gate=("sigmoid" if attn_output_gate and not is_gated_delta else None),
            scores_scale=_declared_scores_scale(
                attention_multiplier, query_pre_attn_scalar,
                layer_head_dim or (hidden_size // num_heads
                                   if hidden_size and num_heads else None)),
            # only a False verdict changes rendering, and only when no config-
            # declared scale contradicts it (declared float > silent-scale read)
            scores_scaled=(False if (code_scores_scaled is False
                                     and attention_multiplier is None
                                     and not query_pre_attn_scalar) else None),
            sinks=(code_attention_sinks and attn_kind in ("mha", "gqa", "mqa")),
            logit_softcap=attn_logit_softcap,
            # B5/U2: mask is no longer taggable-asserted — it is either
            # evidence-backed (decoder-ness / schedule / bidirectional flag)
            # or a typed "unknown" drawn honestly. The sqrt(dim) scores
            # denominator remains asserted when neither a declared scale nor
            # the code verdict backs it (JSON-only; SVG-hash stable).
            asserted=tuple(
                ["scores_scale"] if (code_scores_scaled is None
                                     and attention_multiplier is None
                                     and not query_pre_attn_scalar) else []),
            projection_mode=("fused_qkv" if (_code_fused_qkv and attn_kind in ("mha", "gqa", "mqa")
                                             and not is_gated_delta) else None),
            variant=_mixer_variant(mixer_kind),
        )

        # Code decides the per-layer shape when it resolved the construction;
        # else the config schedule path (unchanged).  moe_active is the shared
        # is-this-MoE gate in both branches.
        if code_moe_schedule is not None:
            is_dense_at_layer = not code_moe_schedule[i]
        elif isinstance(moe_layers_list, (list, tuple, set)):
            # config fallback for the explicit MoE-layer list (Llama-4 w/o source)
            is_dense_at_layer = i not in moe_layers_list
        else:
            is_dense_at_layer = _is_dense_at_layer(
                i,
                moe_active=moe_active,
                first_k_dense=first_k_dense,
                interleave_moe_step=interleave_moe_step,
                moe_layer_freq=moe_layer_freq,
                decoder_sparse_step=decoder_sparse_step,
                mlp_only_layers=mlp_only_layers,
            )

        if moe_active and not is_dense_at_layer:
            ffn = FFNSpec(
                kind="moe",
                activation=activation,
                intermediate_size=intermediate_size or moe_intermediate_size,
                gated=ffn_gated,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_shared_experts=num_shared_experts,
                expert_intermediate_size=moe_intermediate_size or intermediate_size,
                routing=moe_routing,
                activation_clip=activation_clip,
                bias=use_mlp_bias,
                projection_mode=_code_storage_mode,
                expert_projection_mode=_code_expert_fused,
                # B5/U2: unbacked activation/norm-kind facts are typed
                # unknowns (drawn honestly), never "asserted". Two DRAWN
                # conventions remain tagged: split storage (layout unproven)
                # and the pre cell kept when source is present but the
                # topology reader abstained on the idiom.
                asserted=tuple(
                    (["norm_placement"] if norm_placement_conventional else [])
                    + (["ffn_storage"]
                       if (_code_expert_fused is None
                           and _code_storage_mode is None) else [])),
            )
        else:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=ffn_gated,
                activation_clip=activation_clip,
                bias=use_mlp_bias,
                projection_mode=_code_storage_mode,
                # B5/U2: see the MoE branch — only the two DRAWN conventions
                # (kept-pre placement, split storage) remain asserted.
                asserted=tuple(
                    (["norm_placement"] if norm_placement_conventional else [])
                    + (["ffn_storage"] if _code_storage_mode is None else [])),
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
                rope=False,
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
    elif _code_lm_head_tying(text_cfg, context):
        tie_word_embeddings = True
        _note_fact("model", "tie_word_embeddings", True,
                   "code_proven", "lm_head_tying_from_files")
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
                vision_tower_evidence(cfg, bundle=context.source_bundle),
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
        from ...evidence.patterns import decoder_codebook_streams_from_files
        _cb_bundle = getattr(context, "source_bundle", None)
        _cb_comp = getattr(_cb_bundle, "component_files", {}) or {}
        _cb_files = (_cb_comp.get("decoder") or _cb_comp.get("root")
                     or getattr(_cb_bundle, "files", None))
        streams = decoder_codebook_streams_from_files(_cb_files)
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
    extras = decoder_extras(
        vocab_size,
        hidden_size,
        tie_word_embeddings,
        per_layer_embedding_extras(hidden_size, ple_dim, ple_vocab, num_layers) if ple_dim else None,
        modality_extras,
        embed_norm=_code_embedding_norm(text_cfg, context),
        # Gemma-2's final_logit_softcapping is a REAL pre-sampling op — the LM
        # head card states it (only-when-present; everyone else byte-stable).
        final_logit_softcap=final_logit_softcap,
        codebooks=codebooks,
    )
    # U2: an unknown norm kind must not print "Final RMSNorm" — the short
    # generic label keeps the pill width; the card prose says why it is
    # generic (metadata's kind-aware tooltip).
    final_norm_name = {"layernorm": "LayerNorm",
                       "rmsnorm": "RMSNorm"}.get(norm_kind, "Norm")
    if residual_post_layernorm:
        extras["render"].setdefault("layer_annotations", []).append(
            "residual taps the post-LayerNorm output")
    if _scale_embedding:
        for block in extras["render"]["model_blocks"]:
            if block.get("id") == "embed":
                block["facts"] = (block.get("facts") or []) + [
                    "scaled × √d (scale_embedding)"]
    for block in extras["render"]["model_blocks"]:
        if block.get("id") == "final_rms":
            block.update({
                "label": f"Final {final_norm_name}",
                "description": (
                    f"{final_norm_name} over the last hidden state before the output head."
                ),
            })
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
    #   3. qk_norm: Q/K/V norms are unconditional in __init__ (not a config flag).
    # Block-diffusion layout is a CONFIG fact (canvas_length declares the
    # denoising canvas) — never a model_type spelling.  A block-diffusion
    # config without canvas_length renders as the plain decoder its config
    # declares; identity must not fill the gap (eradication plan I-07).
    if _g(cfg, "canvas_length") is not None:
        from .blocks.model import block_diffusion_loop_blocks
        from .blocks.layers import diffusion_gemma_layer_blocks
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
        for layer in layers:
            layer.attention.qk_norm = True
            layer.blocks = diffusion_gemma_layer_blocks(
                layer.attention,
                layer.ffn,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                norm_kind=norm_kind,
            )

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
    if use_parallel_residual:
        extras["parallel_residual"] = True
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
    rope_theta = rope_theta or _g(attn_cfg, "rope_theta")
    if rope_theta is not None:
        extras.setdefault("rope", {}).setdefault("rope_theta", rope_theta)

    # Logit / query softcap (Gemma 2/3 style) — info-only annotation.
    for cap_key in ("attn_logit_softcapping", "final_logit_softcapping", "query_pre_attn_scalar"):
        val = _g(text_cfg, cap_key)
        if val is not None:
            extras.setdefault("softcap", {})[cap_key] = val

    if any(qk_norm_layers) if qk_norm_layers else use_qk_norm:
        extras["qk_norm"] = True

    # Generic attention-side knobs surfaced as info-only annotations.
    clip_qkv = _g(text_cfg, "clip_qkv") or _g(attn_cfg, "clip_qkv")
    if clip_qkv is not None:
        extras.setdefault("attention", {})["clip_qkv"] = clip_qkv

    # Surface the raw partial-rotary fraction the config declared, when present.
    if partial_rotary_fac is not None:
        extras["partial_rotary_factor"] = partial_rotary_fac

    # Per-layer dual-KV info, when both sides differ.
    if _g(text_cfg, "num_global_key_value_heads") or _g(text_cfg, "global_head_dim"):
        extras["dual_kv"] = {
            "sliding": {"num_kv_heads": num_kv_heads, "head_dim": head_dim},
            "global":  {"num_kv_heads": num_kv_global, "head_dim": head_dim_global},
        }

    # Pass-through flags that show up on a few families and are interesting at the model card level.
    for flag in ("attention_k_eq_v", "use_double_wide_mlp"):
        val = _g(text_cfg, flag)
        if val:
            extras[flag] = val

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


def _moe_layers_from_enum(text_cfg) -> list[int] | None:
    """step3 ``moe_layers_enum = "4,5,…,60"`` -> the MoE-layer membership list the
    ``moe_layers`` reader already consumes.  None when absent/unparseable."""
    for field in _LAYER_SCHEDULES["moe_comma_string_fields"]:
        raw = _g(text_cfg, field)
        if isinstance(raw, str) and raw.strip():
            try:
                return [int(x) for x in raw.strip().split(",") if x.strip()]
            except ValueError:
                return None
    return None


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


def _is_dense_at_layer(i: int, *, moe_active: bool, first_k_dense: int, interleave_moe_step: int, moe_layer_freq: int, decoder_sparse_step: int = 0, mlp_only_layers=()) -> bool:
    """Is layer ``i`` a dense FFN (vs MoE) given the config flags?"""
    if not moe_active:
        return True
    if first_k_dense and i < first_k_dense:
        return True
    if interleave_moe_step and (i % interleave_moe_step == 0):
        return True
    if moe_layer_freq and moe_layer_freq > 1 and (i % moe_layer_freq != 0):
        return True
    # Qwen3-MoE: dense when explicitly listed, or when this layer isn't on the
    # sparse step (HF: MoE iff (i + 1) % decoder_sparse_step == 0).
    if mlp_only_layers and i in mlp_only_layers:
        return True
    if decoder_sparse_step and decoder_sparse_step > 1 and ((i + 1) % decoder_sparse_step != 0):
        return True
    return False


def _attention_kind(is_mla: bool, num_q: int, num_kv: int, has_multi_query_flag: bool) -> str:
    """Classify the attention head pattern.

    Note: ``num_kv == 1`` alone is *not* enough to label a layer as MQA.  Many
    GQA models (e.g. Gemma 4 global layers) reach 1 KV head as an extreme of
    grouping; their designers still call it GQA.  Only when the config carries
    an explicit ``multi_query`` flag (Falcon 7B, GPT-BigCode) do we tag MQA.
    """
    if is_mla:
        return "mla"
    if not num_q:
        return "mha"
    if num_kv == num_q:
        return "mha"
    if has_multi_query_flag and num_kv == 1:
        return "mqa"
    return "gqa"


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
    declared_rms = _g(cfg, "rmsnorm")            # chatglm boolean declaration
    if declared_rms is True:
        return "rmsnorm", ("config_declared", "rmsnorm flag")
    if declared_rms is False:
        return "layernorm", ("config_declared", "rmsnorm flag")
    if explicit_norm_type:
        nt = str(explicit_norm_type).lower()
        if "rms" in nt:
            return "rmsnorm", ("config_declared", "norm_type")
        if "layer" in nt:
            return "layernorm", ("config_declared", "norm_type")
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
    # The eps SPELLING is a hint derived from a field NAME, not a declaration
    # of the kind (PhiMoE carries rms_norm_eps but constructs nn.LayerNorm) —
    # recorded as ``derived`` so the ledger never upgrades a spelling to fact.
    if rms_eps is not None:
        return "rmsnorm", ("derived", "rms_norm_eps spelling")
    if ln_eps is not None or ln_eps2 is not None:
        return "layernorm", ("derived", "layer_norm_eps spelling")
    return None, None


def _is_gated(activation: str | None, norm_kind: str | None = None,
              code_gated: bool | None = None) -> bool | None:
    """Whether the FFN has a separate gate projection (SwiGLU/GeGLU style).

    Evidence order (code > config-signal > ABSTAIN):

    * ``code_gated`` — the MLP class's ``forward()`` either does a ``gate_mul`` or
      not (read from the modeling source). The ground truth: it wins whenever
      available, so a dense RMSNorm decoder (Phi) is no longer mis-gated.
      This is the SAME evidence the nested-conformance net reads, so the
      drawing and the net cannot diverge.
    * An activation whose NAME explicitly declares a GLU (``swiglu``,
      ``geglu``). A plain elementwise activation such as ``silu``/``gelu`` is
      deliberately insufficient: both dense and gated FFNs use those
      nonlinearities. Direct config declarations (``is_gated_act`` /
      ``feed_forward_proj``) are handled at the caller before this helper.
    * Otherwise ABSTAIN (``None``). Gate-or-not is a CODE fact; the old
      ``norm_kind == "rmsnorm"`` terminal fabricated a gate for dense RMSNorm
      decoders at zero evidence — the census's cascade (a defaulted norm_kind
      feeding a structural guess). ``None`` draws the honest undeclared-FFN
      block instead (U2 default-kill).  ``norm_kind`` stays in the signature
      only so call sites read naturally; it no longer decides anything.
    """
    if code_gated is not None:
        return code_gated
    a = (activation or "").lower()
    if "glu" in a:
        return True
    # U2 P2c/P3b RETIRED the gate-family activation tier (silu/swish/tanh-GELU
    # → True): plain elementwise spellings are used by dense AND gated FFNs,
    # so they were never proof.  The tier existed only to keep the blessed
    # MoE fixtures' expert drills drawn — the MoE expert-hop in
    # the exact-owner FFN mechanism reader code-proves ordinary FFNs; routed
    # experts remain on their separately owned U3-F reader. The corpus audit
    # shows ZERO derived-tier
    # reliance, so retirement is drift-free.  Strict rule: no evidence ⇒
    # abstain (the honest undeclared-FFN block).
    return None


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
