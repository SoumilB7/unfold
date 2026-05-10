"""Adapter for the Qwen model family.

Covers:
* Qwen2 / Qwen2.5 (dense)     — model_type: "qwen2"
* Qwen2-MoE                   — model_type: "qwen2_moe"
* Qwen3 (dense)               — model_type: "qwen3"
* Qwen3-MoE / Qwen2.5-Max     — model_type: "qwen3_moe"
* Qwen3.5 / Qwen3.6 (hybrid)  — model_type: "qwen3_5_moe", text nested under text_config.
                                 Alternates linear (SSM-style) and full-context attention layers.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_FLAT_TYPES    = {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}
_WRAPPED_TYPES = {"qwen3_5_moe", "qwen3_5_moe_text"}
_MODEL_TYPES   = _FLAT_TYPES | _WRAPPED_TYPES


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("qwen" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    text_cfg  = _text_config(cfg)
    arch_name = architecture_name(cfg, "qwen")
    model_type = (_g(text_cfg, "model_type") or "").lower()
    is_moe = "moe" in model_type or bool(_g(text_cfg, "num_experts"))

    num_layers  = _g(text_cfg, "num_hidden_layers", 0)
    num_heads   = _g(text_cfg, "num_attention_heads", 0)
    num_kv_heads = _g(text_cfg, "num_key_value_heads", num_heads)
    hidden_size  = _g(text_cfg, "hidden_size", 0)
    head_dim     = _g(text_cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    activation   = (_g(text_cfg, "hidden_act") or "silu").lower()

    # Standard (non-hybrid) attention kind
    if num_kv_heads == num_heads:
        full_attn_kind = "mha"
    elif num_kv_heads == 1:
        full_attn_kind = "mqa"
    else:
        full_attn_kind = "gqa"

    sliding_window  = _g(text_cfg, "sliding_window")
    sliding_pattern = _g(text_cfg, "sliding_window_pattern")
    layer_types     = _g(text_cfg, "layer_types") or []

    # Hybrid linear-attention fields (Qwen3.5/3.6)
    linear_num_kv_heads = _g(text_cfg, "linear_num_key_heads") or 0
    linear_head_dim     = _g(text_cfg, "linear_key_head_dim") or head_dim

    # MoE FFN fields
    num_experts          = _g(text_cfg, "num_experts") or 0
    num_experts_per_tok  = _g(text_cfg, "num_experts_per_tok") or _g(text_cfg, "top_k") or 0
    num_shared_experts   = _g(text_cfg, "num_shared_experts") or 0
    moe_intermediate_size = _g(text_cfg, "moe_intermediate_size") or 0
    dense_intermediate_size = _g(text_cfg, "intermediate_size") or 0

    layers = []
    for i in range(num_layers):
        layer_type = layer_types[i] if i < len(layer_types) else "full_attention"
        is_linear = layer_type == "linear_attention"

        if is_linear:
            # SSM / recurrent linear-attention layer — no positional mask, compact heads
            attn = AttentionSpec(
                kind="linear",
                num_heads=num_heads,
                num_kv_heads=linear_num_kv_heads or num_kv_heads,
                head_dim=linear_head_dim,
                mask="causal",
            )
        else:
            if sliding_pattern and sliding_window:
                mask = "sliding" if (i % sliding_pattern) != (sliding_pattern - 1) else "causal"
                win  = sliding_window if mask == "sliding" else None
            elif sliding_window:
                mask, win = "sliding", sliding_window
            else:
                mask, win = "causal", None

            attn = AttentionSpec(
                kind=full_attn_kind,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                mask=mask,
                window_size=win,
            )

        if is_moe and num_experts:
            ffn = FFNSpec(
                kind="moe",
                activation=activation,
                intermediate_size=dense_intermediate_size or moe_intermediate_size,
                gated=True,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_shared_experts=num_shared_experts,
                expert_intermediate_size=moe_intermediate_size,
            )
        else:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
                intermediate_size=dense_intermediate_size,
                gated=True,
            )

        layers.append(decoder_layer(i, attn, ffn, hidden_size))

    vocab_size = _g(text_cfg, "vocab_size", 0) or _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(
        _g(text_cfg, "tie_word_embeddings", _g(cfg, "tie_word_embeddings", False))
    )
    mtp_layers = _g(text_cfg, "mtp_num_hidden_layers") or 0
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)
    if mtp_layers:
        extras["mtp"] = {"num_layers": mtp_layers}
    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_g(text_cfg, "max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=extras,
    )


def _text_config(cfg: Any) -> Any:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _WRAPPED_TYPES:
        sub = _g(cfg, "text_config")
        if sub is not None:
            return sub
    return cfg
