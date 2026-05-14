"""Adapter for Microsoft Phi-family decoder configs.

Phi-2 is not Llama-shaped enough to hide inside the Llama adapter: it uses the
``phi`` model type, GELU MLPs, and partial RoPE metadata.  Phi-3/Phi-4 configs
are closer to Llama, but keeping them here makes the supported family boundary
explicit and gives us one place to handle Phi-MoE and multimodal wrappers later.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"phi", "phi3", "phi4"}


def matches(cfg: Any) -> bool:
    text_cfg = _text_config(cfg)
    model_type = (_g(text_cfg, "model_type") or _g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = [a.lower() for a in (_g(cfg, "architectures") or [])]
    return any("phiforcausallm" in a or "phi3forcausallm" in a for a in arches)


def parse(cfg: Any) -> ModelIR:
    text_cfg = _text_config(cfg)
    arch_name = architecture_name(cfg, "phi")
    model_type = (_g(text_cfg, "model_type") or _g(cfg, "model_type") or "phi").lower()

    num_layers = _g(text_cfg, "num_hidden_layers") or _g(text_cfg, "n_layer") or 0
    hidden_size = _g(text_cfg, "hidden_size") or _g(text_cfg, "n_embd") or 0
    num_heads = _g(text_cfg, "num_attention_heads") or _g(text_cfg, "n_head") or 0
    num_kv_heads = _g(text_cfg, "num_key_value_heads", num_heads)
    head_dim = _g(text_cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = (
        _g(text_cfg, "intermediate_size")
        or _g(text_cfg, "n_inner")
        or (4 * hidden_size if hidden_size else 0)
    )
    activation = (
        _g(text_cfg, "hidden_act")
        or _g(text_cfg, "activation_function")
        or ("gelu_new" if model_type == "phi" else "silu")
    ).lower()
    gated = model_type != "phi"

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    sliding_window = _g(text_cfg, "sliding_window")
    sliding_pattern = _g(text_cfg, "sliding_window_pattern")
    use_qk_norm = bool(_g(text_cfg, "qk_layernorm", False) or _g(text_cfg, "use_qk_norm", False))

    num_experts = _g(text_cfg, "num_local_experts") or _g(text_cfg, "num_experts") or 0
    num_experts_per_tok = _g(text_cfg, "num_experts_per_tok") or _g(text_cfg, "num_experts_per_token") or 0
    moe_intermediate_size = _g(text_cfg, "moe_intermediate_size") or _g(text_cfg, "expert_intermediate_size") or 0

    layers = []
    for i in range(num_layers):
        if sliding_pattern and sliding_window:
            mask = "sliding" if (i % sliding_pattern) != (sliding_pattern - 1) else "causal"
            win = sliding_window if mask == "sliding" else None
        elif sliding_window:
            mask, win = "sliding", sliding_window
        else:
            mask, win = "causal", None

        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask=mask,
            window_size=win,
            qk_norm=use_qk_norm,
        )

        if num_experts:
            ffn = FFNSpec(
                kind="moe",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=True,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                expert_intermediate_size=moe_intermediate_size or intermediate_size,
            )
        else:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=gated,
            )
        layers.append(decoder_layer(i, attn, ffn, hidden_size, norm_kind=_norm_kind(text_cfg)))

    vocab_size = _g(text_cfg, "vocab_size", 0) or _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(_g(text_cfg, "tie_word_embeddings", _g(cfg, "tie_word_embeddings", False)))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)

    partial_rotary_factor = _g(text_cfg, "partial_rotary_factor")
    if partial_rotary_factor is not None:
        extras["partial_rotary_factor"] = partial_rotary_factor
    if num_experts:
        extras["moe"] = {
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
        }

    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_g(text_cfg, "max_position_embeddings") or _g(text_cfg, "n_positions"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=extras,
    )


def _text_config(cfg: Any) -> Any:
    for key in ("text_config", "language_config", "llm_config"):
        sub = _g(cfg, key)
        if sub is not None:
            return sub
    return cfg


def _norm_kind(cfg: Any) -> str:
    norm = (_g(cfg, "norm_type") or _g(cfg, "normalization") or "").lower()
    if "rms" in norm:
        return "rmsnorm"
    if "layer" in norm:
        return "layernorm"
    return "layernorm" if (_g(cfg, "model_type") or "").lower() == "phi" else "rmsnorm"
