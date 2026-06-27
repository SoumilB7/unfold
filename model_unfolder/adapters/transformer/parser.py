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
from ...everchanging import load_aliases, load_layer_type_labels, load_layer_topology
from ...ir import AttentionSpec, CrossLayerEdge, FFNSpec, ModelIR
from .assembly import decoder_extras, decoder_layer, parallel_decoder_layer
from .blocks import mtp_head_block
from .common import architecture_name, get_config_value as _g, model_name
from .special_parts.per_layer_embedding import (
    per_layer_embedding_blocks,
    per_layer_embedding_extras,
)
from .special_parts.modalities import multimodal_extras
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

# Per-family macro-topology (post/sandwich norm, flag-less parallel residual) —
# data, not code (everchanging/transformer/layer_topology.yaml).
_LAYER_TOPOLOGY = load_layer_topology()


def _resolve(cfg: Any, canonical: str, default=None):
    """Try every known alias for a field, return the first hit."""
    aliases = _ALIASES.get(canonical, [canonical])
    for alias in aliases:
        val = _g(cfg, alias)
        if val is not None:
            # The field is handled — treat all its spellings as parsed so a
            # redundant sibling alias also present in the config (e.g. both
            # num_experts and n_routed_experts) isn't flagged as unparsed.
            for a in aliases:
                debug.note_access(a)
            return val
    return default


def _code_layer_topology(cfg: Any) -> dict | None:
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
        from ...evidence.sources import resolve_source_files
        files = resolve_source_files(cfg, source="local").files
        return decoder_layer_topology_from_files(files)
    except Exception:
        return None


def _code_norm_kind(cfg: Any) -> str | None:
    """The decoder's norm KIND (rmsnorm/layernorm) READ FROM THE MODELING SOURCE
    — used only as a config-silent fallback (no eps field), replacing the legacy
    model_type family-set.  Best-effort, never raises into the parse."""
    try:
        from ...evidence.patterns import decoder_norm_kind_from_files
        from ...evidence.sources import resolve_source_files
        return decoder_norm_kind_from_files(resolve_source_files(cfg, source="local").files)
    except Exception:
        return None


_TEXT_WRAPPER_KEYS = (
    "text_config", "language_config", "llm_config", "text_model_config",
    "thinker_config",  # Qwen3-Omni nests the LM under thinker_config.text_config
)


def _unwrap_text(cfg: Any, _depth: int = 0) -> Any:
    """If a multimodal wrapper hides the LM config under a sub-key, unwrap it.

    Handles one further level of nesting (e.g. Qwen3-Omni's
    ``thinker_config.text_config``) by recursing into a wrapper that doesn't
    itself carry transformer shape.
    """
    if _depth > 3:
        return cfg
    for key in _TEXT_WRAPPER_KEYS:
        sub = _g(cfg, key)
        if sub is None:
            continue
        if isinstance(sub, dict):
            if _has_transformer_shape(sub):
                return sub
            completed = _complete_config_from_transformers_registry(sub)
            if _has_transformer_shape(completed):
                return completed
            # Wrapper that itself nests the LM deeper (Omni thinker_config).
            nested = _unwrap_text(sub, _depth + 1)
            if nested is not sub and _has_transformer_shape(nested):
                return nested
        elif _has_transformer_shape(sub):
            return sub
    return cfg


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
        _resolve(cfg, field) is not None
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


def parse(cfg: Any) -> ModelIR:
    debug.reset()  # start a fresh field-access record for this parse
    warnings: list[str] = []
    model_type = (_g(cfg, "model_type") or "unknown").lower()
    arch_name  = architecture_name(cfg, model_type)

    text_cfg = _unwrap_text(cfg)
    # Nested text_config (multimodal wrapper) is fully supported — no warning needed.

    # DBRX-style nested config dicts: pull through transparently.
    attn_cfg = _nested(text_cfg, "attn_config")
    ffn_cfg  = _nested(text_cfg, "ffn_config")

    def get(field, default=None):
        """Resolve from text_cfg, falling back to nested sub-configs."""
        val = _resolve(text_cfg, field)
        if val is None:
            val = _resolve(attn_cfg, field)
        if val is None:
            val = _resolve(ffn_cfg, field)
        return default if val is None else val

    num_layers   = get("num_hidden_layers", 0)
    hidden_size  = get("hidden_size", 0)
    num_heads    = get("num_attention_heads", 0)
    num_kv_heads = get("num_key_value_heads") or num_heads
    head_dim     = get("head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = get("intermediate_size", 0)
    # OLMo-style: intermediate_size derived from mlp_ratio * hidden_size.
    if not intermediate_size:
        mlp_ratio = get("mlp_ratio")
        if mlp_ratio and hidden_size:
            intermediate_size = int(hidden_size * float(mlp_ratio))
    # DBRX-style: activation lives in a nested dict like ``ffn_act_fn = {"name": "silu"}``.
    activation_raw = get("hidden_act")
    if isinstance(activation_raw, dict):
        activation_raw = activation_raw.get("name")
    if activation_raw is None:
        nested_act = _g(ffn_cfg, "ffn_act_fn")
        if isinstance(nested_act, dict):
            activation_raw = nested_act.get("name")
    activation   = (activation_raw or "silu").lower()
    sliding_window = get("sliding_window")
    # ---- Sliding-window enable toggle (Qwen2/2.5/3) ----
    # A config may declare a window size but turn SWA *off* (use_sliding_window
    # = False); honor that, otherwise we'd draw sliding attention on what is
    # really a full-attention model.  When absent (Mistral), the window applies.
    use_sliding_window = _g(text_cfg, "use_sliding_window")
    max_window_layers  = _g(text_cfg, "max_window_layers")
    if use_sliding_window is False:
        sliding_window = None
        max_window_layers = None
    layer_types  = _g(text_cfg, "layer_types") or []
    # Resolved through aliases so dialect spellings (DeepSeek-V4 ``compress_rates``)
    # are picked up — see everchanging/aliases.yaml.
    compress_ratios = _resolve(text_cfg, "compress_ratios") or []
    if not layer_types and compress_ratios:
        layer_types = _layer_types_from_compress_ratios(compress_ratios, num_layers)
    norm_kind    = _norm_kind(text_cfg, get("norm_type"))
    _mt_candidates = {model_type, str(_g(text_cfg, "model_type") or "").lower()}
    # Norm placement (pre / post / double-sandwich) is STRUCTURE and carries no
    # config flag — so it is READ FROM THE LAYER'S forward() dataflow (code ->
    # structure), the general replacement for the model_type identity table.
    # The table is now only an offline fallback cache when source can't be read.
    _code_topo = _code_layer_topology(text_cfg)
    norm_placement = (
        (_code_topo or {}).get("norm_placement")
        or next((_LAYER_TOPOLOGY["norm_placement"][mt] for mt in _mt_candidates
                 if mt in _LAYER_TOPOLOGY["norm_placement"]), "pre")
    )
    # RoPE is the default for decoder LLMs; ALiBi / learned-absolute families
    # (BLOOM/MPT/GPT-2/OPT) don't apply it — drawing a RoPE step there would be
    # fabricated wiring.
    uses_rope = not (_mt_candidates & set(_LAYER_TOPOLOGY["no_rope"]))

    if not num_layers:
        warnings.append("Config missing num_hidden_layers (and aliases) — layer list will be empty.")
    if not hidden_size:
        warnings.append("Config missing hidden_size (and aliases) — geometry will be incomplete.")

    # ---- Per-layer dual KV (Gemma 4 sliding vs global; might appear elsewhere) ----
    num_kv_global   = _g(text_cfg, "num_global_key_value_heads") or num_kv_heads
    head_dim_global = _g(text_cfg, "global_head_dim") or head_dim

    # ---- Attention shape ----
    q_lora_rank  = _g(text_cfg, "q_lora_rank")
    kv_lora_rank = _g(text_cfg, "kv_lora_rank")
    is_mla       = bool(kv_lora_rank)
    # MLA decoupled head geometry — Q/K split into nope + rope, V its own width
    # (DeepSeek/Kimi). Needed for an accurate MLA parameter count.
    qk_nope_head_dim = _g(text_cfg, "qk_nope_head_dim")
    qk_rope_head_dim = _g(text_cfg, "qk_rope_head_dim")
    v_head_dim_cfg   = _g(text_cfg, "v_head_dim")
    has_multi_query_flag = bool(_g(text_cfg, "multi_query"))
    if has_multi_query_flag:
        num_kv_heads = 1
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
    rotary_pct           = _g(text_cfg, "rotary_pct")
    rotary_dim           = _g(text_cfg, "rotary_dim")
    partial_rotary_fac   = _g(text_cfg, "partial_rotary_factor")
    rope_dim_value       = _rope_dim(rotary_pct, rotary_dim, partial_rotary_fac, head_dim)
    # Multimodal RoPE (Qwen2-VL / Qwen3-VL): rope_scaling.mrope_section splits the
    # rotary dims across (temporal, height, width) position axes — a Tier-3 property.
    _rope_scaling        = _g(text_cfg, "rope_parameters") or _g(text_cfg, "rope_scaling") or {}
    mrope_section        = _rope_scaling.get("mrope_section") if isinstance(_rope_scaling, dict) else None

    # ---- QK-Norm ----
    use_qk_norm = bool(_g(text_cfg, "use_qk_norm") or _g(text_cfg, "qk_norm") or _g(text_cfg, "qk_layernorm"))

    # ---- Bias terms on the Q/K/V/O projections (Qwen2, GPT-2, Phi, ...) ----
    use_attention_bias = bool(_g(text_cfg, "attention_bias") or _g(attn_cfg, "attention_bias"))

    # ---- Layer topology ----
    # Parallel residual: a config flag when the family TOGGLES it (Falcon
    # new_decoder_architecture / GPT-NeoX use_parallel_residual — gated inside an
    # `if`, so the config decides); else READ FROM the forward() when it is
    # UNCONDITIONAL structure with no flag (Cohere, GPT-J, Phi — all flagless,
    # all missed by the old model_type table, so all silently drawn sequential).
    # The table stays only as an offline fallback cache.
    use_parallel_residual = bool(
        _g(text_cfg, "use_parallel_residual") or _g(text_cfg, "parallel_attn")
        or (_code_topo or {}).get("parallel_residual")
        or _mt_candidates & set(_LAYER_TOPOLOGY["parallel_residual"])
    )

    # ---- MoE ----
    num_experts         = get("num_experts", 0)
    num_experts_per_tok = get("num_experts_per_tok", 0)
    num_shared_experts  = get("num_shared_experts", 0)
    moe_intermediate_size = get("moe_intermediate_size", 0)
    enable_moe_block    = _g(text_cfg, "enable_moe_block")
    moe_active          = bool(num_experts) and (enable_moe_block is not False)
    first_k_dense       = _g(text_cfg, "first_k_dense_replace") or 0
    moe_layer_freq      = _g(text_cfg, "moe_layer_freq") or 1
    interleave_moe_step = _g(text_cfg, "interleave_moe_layer_step") or 0
    # Qwen3-MoE: MoE applies every ``decoder_sparse_step`` layers, except the
    # explicit ``mlp_only_layers`` which stay dense.
    decoder_sparse_step = _g(text_cfg, "decoder_sparse_step") or 0
    mlp_only_layers     = set(_g(text_cfg, "mlp_only_layers") or [])
    moe_every_layer     = moe_active and not any([
        first_k_dense,
        interleave_moe_step,
        decoder_sparse_step and decoder_sparse_step > 1,
        mlp_only_layers,
    ])
    # Router behaviour: gating fn, grouped/node-limited routing, top-k renorm,
    # routed-output scale (DeepSeek-V3, Kimi-K2, GLM, Qwen3-MoE).
    moe_routing = _moe_routing(text_cfg) if moe_active else None
    # gpt-oss clamps its SwiGLU activation to ±swiglu_limit — a Tier-3 property.
    activation_clip = _g(text_cfg, "swiglu_limit")

    # ---- Cross-layer KV
    #  sharing (the last N layers reuse K/V from earlier) ----
    num_kv_shared_layers   = _g(text_cfg, "num_kv_shared_layers") or 0
    first_shared_layer     = (num_layers - num_kv_shared_layers) if num_kv_shared_layers else num_layers

    # ---- Per-Layer Embedding side pathway ----
    ple_dim   = _g(text_cfg, "hidden_size_per_layer_input") or 0
    ple_vocab = _g(text_cfg, "vocab_size_per_layer_input") or get("vocab_size", 0)

    # ---- Decoder layers that read external modality states through cross-attention ----
    cross_attn_layer_set = set(_cross_attention_layers(cfg, text_cfg) or [])
    has_cross_attention_side_state = bool(
        cross_attn_layer_set
        and (_g(cfg, "vision_config") is not None or _g(cfg, "vision_model_config") is not None)
    )

    # ---- Walk the layer stack ----
    unknown_layer_types: set[str] = set()
    cross_layer_edges: list[CrossLayerEdge] = []

    layers = []
    for i in range(num_layers):
        mask, window, is_full_in_sliding_stack = _layer_mask(
            i, layer_types, sliding_window, sliding_window_pattern,
            has_sliding_in_stack, unknown_layer_types, max_window_layers,
        )
        compress_ratio = _compress_ratio_for_layer(i, compress_ratios, layer_types)

        # Per-layer dual KV: full layers in a sliding stack use the global counts.
        if is_full_in_sliding_stack:
            layer_kv_heads = num_kv_global
            layer_head_dim = head_dim_global
        else:
            layer_kv_heads = num_kv_heads
            layer_head_dim = head_dim

        attn_kind = _attention_kind(is_mla, num_heads, layer_kv_heads, has_multi_query_flag)
        is_nope   = bool(no_rope_interval > 1 and i % no_rope_interval == 0)
        is_cross_attn_layer = has_cross_attention_side_state and i in cross_attn_layer_set

        kv_source: int | None = None
        if i >= first_shared_layer:
            kv_source = _last_matching_layer(layer_types, i, first_shared_layer)
            if kv_source is not None:
                cross_layer_edges.append(
                    CrossLayerEdge(kind="kv_share", from_layer=kv_source, to_layer=i, shared=["K", "V"])
                )

        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_heads,
            num_kv_heads=layer_kv_heads,
            head_dim=layer_head_dim,
            kv_lora_rank=kv_lora_rank if is_mla else None,
            q_lora_rank=q_lora_rank if is_mla else None,
            qk_nope_head_dim=qk_nope_head_dim if is_mla else None,
            qk_rope_head_dim=qk_rope_head_dim if is_mla else None,
            v_head_dim=v_head_dim_cfg if is_mla else None,
            rope_dim=rope_dim_value,
            mask=mask,
            window_size=window,
            kv_source_layer=kv_source,
            qk_norm=use_qk_norm,
            rope=uses_rope,
            bias=use_attention_bias,
            no_rope=is_nope,
            cross_attention=is_cross_attn_layer,
            cross_kv_source="projected image states" if is_cross_attn_layer else None,
            compress_ratio=compress_ratio,
            # Sparse-attention indexer fan-in. CSA declares it alongside a
            # compress_ratio; DeepSeek-V3.2 DSA declares its own indexer geometry
            # (index_n_heads/index_head_dim) — read both so neither is dropped.
            index_topk=_g(text_cfg, "index_topk"),
            index_n_heads=_g(text_cfg, "index_n_heads"),
            index_head_dim=_g(text_cfg, "index_head_dim"),
            mrope_section=mrope_section,
        )

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
                gated=_is_gated(activation, norm_kind),
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_shared_experts=num_shared_experts,
                expert_intermediate_size=moe_intermediate_size or intermediate_size,
                routing=moe_routing,
                activation_clip=activation_clip,
            )
        else:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=_is_gated(activation, norm_kind),
                activation_clip=activation_clip,
            )

        extra_blocks = list(per_layer_embedding_blocks(hidden_size, ple_dim, activation="gelu")) if ple_dim else []
        if is_cross_attn_layer:
            extra_blocks.append(_cross_attention_states_side_block())

        if use_parallel_residual:
            layers.append(parallel_decoder_layer(i, attn, ffn, hidden_size, norm_kind=norm_kind))
        else:
            layers.append(decoder_layer(
                i, attn, ffn, hidden_size,
                norm_kind=norm_kind,
                norm_placement=norm_placement,
                extra_blocks=extra_blocks,
            ))

    for lt in sorted(unknown_layer_types):
        warnings.append(f"Config layer_types contains unrecognized value {lt!r} — treated as causal.")

    vocab_size = get("vocab_size", 0)
    tie_word_embeddings = bool(get("tie_word_embeddings", False))

    extras = decoder_extras(
        vocab_size,
        hidden_size,
        tie_word_embeddings,
        per_layer_embedding_extras(hidden_size, ple_dim, ple_vocab, num_layers) if ple_dim else None,
        multimodal_extras(cfg, text_cfg, hidden_size),
    )

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
    if _g(cfg, "canvas_length") is not None or "diffusion" in model_type:
        from .blocks.model import block_diffusion_loop_blocks
        from .blocks.layers import diffusion_gemma_layer_blocks
        canvas_length = int(_g(cfg, "canvas_length") or 256)
        final_softcap = get("final_logit_softcapping")
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
    mtp_modules = _g(text_cfg, "num_nextn_predict_layers") or _g(text_cfg, "num_mtp_layers")
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
    rope_params = _g(text_cfg, "rope_parameters") or _g(text_cfg, "rope_scaling")
    if isinstance(rope_params, dict):
        rope_type = rope_params.get("rope_type") or rope_params.get("type")
        scaling = {
            "type": rope_type,
            "factor": rope_params.get("factor"),
            "original_max_position_embeddings": rope_params.get("original_max_position_embeddings"),
            "rope_theta": rope_params.get("rope_theta") or _g(text_cfg, "rope_theta"),
        }
        extras.setdefault("rope", {}).update({k: v for k, v in scaling.items() if v is not None})

    # RoPE base frequency — present on most rotary models even without a scaling
    # dict (the block above only fires when one is declared); surface it always.
    rope_theta = _g(text_cfg, "rope_theta") or _g(attn_cfg, "rope_theta")
    if rope_theta is not None:
        extras.setdefault("rope", {}).setdefault("rope_theta", rope_theta)

    # Logit / query softcap (Gemma 2/3 style) — info-only annotation.
    for cap_key in ("attn_logit_softcapping", "final_logit_softcapping", "query_pre_attn_scalar"):
        val = _g(text_cfg, cap_key)
        if val is not None:
            extras.setdefault("softcap", {})[cap_key] = val

    if use_qk_norm:
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

    ir = ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=get("max_position_embeddings"),
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
        if lt == "linear_attention":
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


def _moe_routing(cfg: Any) -> dict | None:
    """Collect the MoE router knobs that decide *how* experts get picked.

    Returns only the fields the config actually declares (DeepSeek/Kimi/GLM use
    the full set; Qwen3-MoE just ``norm_topk_prob``), or ``None`` when none are.
    """
    routing = {
        "scoring_func":          _g(cfg, "scoring_func"),          # sigmoid | softmax
        "topk_method":           _g(cfg, "topk_method"),           # noaux_tc, group_limited_greedy, ...
        "n_group":               _g(cfg, "n_group"),               # expert groups (node-limited routing)
        "topk_group":            _g(cfg, "topk_group"),            # groups kept per token
        "norm_topk_prob":        _g(cfg, "norm_topk_prob"),        # renormalize the top-k gate weights
        "routed_scaling_factor": _g(cfg, "routed_scaling_factor"),  # scale on routed-expert output
    }
    routing = {k: v for k, v in routing.items() if v is not None}
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


def _norm_kind(cfg: Any, explicit_norm_type: Any = None) -> str:
    """Pick LayerNorm vs RMSNorm from config — first explicit field, then eps hints."""
    if explicit_norm_type:
        nt = str(explicit_norm_type).lower()
        if "rms" in nt:
            return "rmsnorm"
        if "layer" in nt:
            return "layernorm"
    if _g(cfg, "rms_norm_eps") is not None:
        return "rmsnorm"
    if _g(cfg, "layer_norm_epsilon") is not None or _g(cfg, "layer_norm_eps") is not None:
        return "layernorm"
    # Config carries no eps field — read the norm KIND from the layer's norm
    # submodule class in the modeling source (code -> fact), the general
    # replacement for the old legacy-family model_type set (gpt2/neox/opt/…).
    code_kind = _code_norm_kind(cfg)
    if code_kind:
        return code_kind
    return "rmsnorm"


def _is_gated(activation: str, norm_kind: str | None = None) -> bool:
    """Whether the FFN has a separate gate projection (SwiGLU/GeGLU style).

    Heuristics, in order:

    * Activation explicitly says so: ``silu``/``swish`` (Llama/Mistral/Qwen/
      DeepSeek/GPT-OSS style), anything with ``glu`` in the name, or
      ``gelu_pytorch_tanh`` (Gemma family — uses gate/up/down despite the
      GELU name).
    * Otherwise, default to gated when the model uses ``RMSNorm`` (a strong
      "modern decoder" signal — modern LMs almost universally use
      gate/up/down).  Legacy LayerNorm models (GPT-2 / NeoX / J / BLOOM /
      MPT / OPT / Falcon / Phi-1/2) end up here and are correctly
      classified as non-gated.
    """
    a = (activation or "").lower()
    if "glu" in a or a in {"silu", "swish", "gelu_pytorch_tanh"}:
        return True
    return norm_kind == "rmsnorm"


def _rope_dim(rotary_pct, rotary_dim, partial_rotary_factor, head_dim) -> int | None:
    """Compute the actual rotary dim from any of the config flavours."""
    if rotary_dim:
        return int(rotary_dim)
    if rotary_pct and head_dim:
        return int(head_dim * float(rotary_pct))
    if partial_rotary_factor and head_dim:
        return int(head_dim * float(partial_rotary_factor))
    return None


def _last_matching_layer(layer_types, i: int, first_shared: int) -> int | None:
    """For cross-layer KV sharing: most recent non-shared layer of the same type."""
    if not layer_types or i >= len(layer_types):
        return None
    target_type = layer_types[i]
    for j in range(min(first_shared, len(layer_types)) - 1, -1, -1):
        if layer_types[j] == target_type:
            return j
    return None


def _cross_attention_states_side_block() -> dict:
    """Layer-local projected image states read by cross-attention layers."""
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
