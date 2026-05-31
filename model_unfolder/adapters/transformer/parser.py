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
# Add a new alias whenever you encounter a new config dialect — that is the
# only kind of per-family handling that exists.
# ---------------------------------------------------------------------------

_ALIASES: dict[str, list[str]] = {
    "num_hidden_layers":       ["num_hidden_layers", "n_layers", "num_layers", "n_layer", "num_blocks", "n_blocks"],
    "num_attention_heads":     ["num_attention_heads", "n_heads", "num_heads", "n_head", "num_q_heads"],
    "num_key_value_heads":     ["num_key_value_heads", "n_kv_heads", "num_kv_heads", "num_key_heads", "kv_n_heads"],
    "hidden_size":             ["hidden_size", "d_model", "n_embd", "model_dim", "embed_dim", "dim"],
    "intermediate_size":       ["intermediate_size", "ffn_dim", "mlp_dim", "inner_dim", "ffn_hidden_size", "feed_forward_proj_dim", "n_inner"],
    "hidden_act":              ["hidden_act", "activation_function", "hidden_activation", "act_fn", "activation", "activation_type"],
    "vocab_size":              ["vocab_size", "n_vocab", "padded_vocab_size"],
    "max_position_embeddings": ["max_position_embeddings", "max_seq_len", "n_positions", "context_length", "max_seq_length", "seq_length", "max_sequence_length"],
    "sliding_window":          ["sliding_window", "attention_window", "window_size"],
    "num_experts":             ["num_local_experts", "num_experts", "n_routed_experts", "n_experts", "moe_num_experts"],
    "num_experts_per_tok":     ["num_experts_per_tok", "top_k_experts", "top_k", "num_selected_experts", "num_experts_per_token", "moe_top_k"],
    "num_shared_experts":      ["num_shared_experts", "n_shared_experts"],
    "moe_intermediate_size":   ["moe_intermediate_size", "expert_intermediate_size", "expert_hidden_size"],
    "head_dim":                ["head_dim", "d_head", "head_size", "kv_channels"],
    "tie_word_embeddings":     ["tie_word_embeddings", "tie_embeddings", "tie_word_embedding_weights"],
    "norm_type":               ["norm_type"],   # OLMo-style ("layer_norm" | "rms_norm")
    "mlp_ratio":               ["mlp_ratio"],   # OLMo-style: intermediate = hidden_size * mlp_ratio
}


_SLIDING_LABELS = {"sliding_attention", "sliding"}
_FULL_LABELS    = {"full_attention", "full", "global", "global_attention", "causal", ""}


def _resolve(cfg: Any, canonical: str, default=None):
    """Try every known alias for a field, return the first hit."""
    for alias in _ALIASES.get(canonical, [canonical]):
        val = _g(cfg, alias)
        if val is not None:
            return val
    return default


def _unwrap_text(cfg: Any) -> Any:
    """If a multimodal wrapper hides the LM config under a sub-key, unwrap it."""
    for key in ("text_config", "language_config", "llm_config", "text_model_config"):
        sub = _g(cfg, key)
        if sub is None:
            continue
        if isinstance(sub, dict):
            if _has_transformer_shape(sub):
                return sub
            completed = _complete_config_from_transformers_registry(sub)
            if _has_transformer_shape(completed):
                return completed
        if not isinstance(sub, dict):
            if _has_transformer_shape(sub):
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
    layer_types  = _g(text_cfg, "layer_types") or []
    norm_kind    = _norm_kind(text_cfg, get("norm_type"))
    norm_placement = "pre"

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
    is_mla       = bool(q_lora_rank or kv_lora_rank)
    has_multi_query_flag = bool(_g(text_cfg, "multi_query"))
    if has_multi_query_flag:
        num_kv_heads = 1
    # Determine if the stack mixes sliding + full layers — affects mask labeling
    # (a full layer in a sliding stack is labeled "global", not "causal").
    sliding_window_pattern = _g(text_cfg, "sliding_window_pattern") or 0
    has_sliding_in_stack = (
        any(_is_sliding_label(lt) for lt in layer_types)
        or bool(sliding_window_pattern and sliding_window)
    )

    # ---- Position encoding ----
    no_rope_interval     = _g(text_cfg, "no_rope_layer_interval") or 0
    rotary_pct           = _g(text_cfg, "rotary_pct")
    rotary_dim           = _g(text_cfg, "rotary_dim")
    partial_rotary_fac   = _g(text_cfg, "partial_rotary_factor")
    rope_dim_value       = _rope_dim(rotary_pct, rotary_dim, partial_rotary_fac, head_dim)

    # ---- QK-Norm ----
    use_qk_norm = bool(_g(text_cfg, "use_qk_norm") or _g(text_cfg, "qk_norm") or _g(text_cfg, "qk_layernorm"))

    # ---- Layer topology ----
    use_parallel_residual = bool(_g(text_cfg, "use_parallel_residual") or _g(text_cfg, "parallel_attn"))

    # ---- MoE ----
    num_experts         = get("num_experts", 0)
    num_experts_per_tok = get("num_experts_per_tok", 0)
    num_shared_experts  = get("num_shared_experts", 0)
    moe_intermediate_size = get("moe_intermediate_size", 0)
    enable_moe_block    = _g(text_cfg, "enable_moe_block")
    moe_active          = bool(num_experts) and (enable_moe_block is not False)
    moe_every_layer     = moe_active and not any([
        _g(text_cfg, "first_k_dense_replace"),
        _g(text_cfg, "interleave_moe_layer_step"),
    ])
    first_k_dense       = _g(text_cfg, "first_k_dense_replace") or 0
    moe_layer_freq      = _g(text_cfg, "moe_layer_freq") or 1
    interleave_moe_step = _g(text_cfg, "interleave_moe_layer_step") or 0

    # ---- Cross-layer KV sharing (the last N layers reuse K/V from earlier) ----
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
            has_sliding_in_stack, unknown_layer_types,
        )

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
            rope_dim=rope_dim_value,
            mask=mask,
            window_size=window,
            kv_source_layer=kv_source,
            qk_norm=use_qk_norm,
            no_rope=is_nope,
            cross_attention=is_cross_attn_layer,
        )

        is_dense_at_layer = _is_dense_at_layer(
            i,
            moe_active=moe_active,
            first_k_dense=first_k_dense,
            interleave_moe_step=interleave_moe_step,
            moe_layer_freq=moe_layer_freq,
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
            )
        else:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=_is_gated(activation, norm_kind),
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
        # The MTP block's attention + FFN are the same as a main decoder layer,
        # so they reuse those drill-downs (no separate views). Pass a
        # representative layer's attention/FFN children for the deeper levels.
        rep_blocks = layers[-1].blocks if layers else []
        rep_attn = next((b for b in rep_blocks if b.get("id") == "attn"), None)
        rep_ffn = next((b for b in rep_blocks if b.get("id") == "ffn"), None)
        extras["render"]["model_blocks"].append(
            mtp_head_block(
                mtp_modules, hidden_size, vocab_size, tie_word_embeddings,
                attn_children=(rep_attn or {}).get("children"),
                ffn_children=(rep_ffn or {}).get("children"),
            )
        )

    if use_parallel_residual:
        extras["parallel_residual"] = True
    if moe_active:
        extras["moe"] = {
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "num_shared_experts": num_shared_experts,
            "every_layer": moe_every_layer,
        }
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

    return ModelIR(
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


# ---------------------------------------------------------------------------
# Per-layer helpers
# ---------------------------------------------------------------------------


def _is_sliding_label(lt: str) -> bool:
    return lt in _SLIDING_LABELS or "sliding" in lt


def _is_full_label(lt: str) -> bool:
    return lt in _FULL_LABELS


def _layer_mask(i, layer_types, sliding_window, sliding_window_pattern, has_sliding_in_stack, unknown):
    """Resolve (mask, window, is_full_in_sliding_stack) for a single layer."""
    if layer_types and i < len(layer_types):
        lt = layer_types[i]
        if _is_sliding_label(lt):
            return "sliding", sliding_window, False
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
        return "sliding", sliding_window, False
    return "causal", None, False


def _is_dense_at_layer(i: int, *, moe_active: bool, first_k_dense: int, interleave_moe_step: int, moe_layer_freq: int) -> bool:
    """Is layer ``i`` a dense FFN (vs MoE) given the config flags?"""
    if not moe_active:
        return True
    if first_k_dense and i < first_k_dense:
        return True
    if interleave_moe_step and (i % interleave_moe_step == 0):
        return True
    if moe_layer_freq and moe_layer_freq > 1 and (i % moe_layer_freq != 0):
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
    # Legacy decoder-only families predate RMSNorm — they universally use LayerNorm.
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in {"gpt_neox", "gptj", "gpt2", "bloom", "mpt", "falcon", "opt", "phi"}:
        return "layernorm"
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
        "detail_view": "vision_path",
        "w": 250,
        "h": 50,
        "font": 15,
    }
