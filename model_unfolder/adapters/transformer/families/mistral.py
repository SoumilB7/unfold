"""Adapter for the Mistral model family.

Covers:
* Mistral 7B / Mistral Small / Ministral  — flat GQA config (model_type: "mistral")
* Mixtral 8x7B / 8x22B                   — sparse MoE (model_type: "mixtral")
* Mistral Medium 3.5 / Pixtral-class      — multimodal wrapper with text nested
                                            under text_config (model_type: "mistral3")
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_FLAT_TYPES  = {"mistral", "mixtral"}
_WRAPPED_TYPES = {"mistral3", "ministral3"}
_ALL_TYPES = _FLAT_TYPES | _WRAPPED_TYPES


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _ALL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("mistral" in a.lower() or "mixtral" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    text_cfg = _text_config(cfg)
    arch_name = architecture_name(cfg, "mistral")

    num_layers    = _g(text_cfg, "num_hidden_layers", 0)
    num_heads     = _g(text_cfg, "num_attention_heads", 0)
    num_kv_heads  = _g(text_cfg, "num_key_value_heads", num_heads)
    hidden_size   = _g(text_cfg, "hidden_size", 0)
    head_dim      = _g(text_cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    activation    = (_g(text_cfg, "hidden_act", "silu") or "silu").lower()

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    sliding_window  = _g(text_cfg, "sliding_window")
    sliding_pattern = _g(text_cfg, "sliding_window_pattern")

    # MoE fields (Mixtral uses num_local_experts; generic fallback to num_experts)
    num_experts        = _g(text_cfg, "num_local_experts") or _g(text_cfg, "num_experts") or 0
    num_experts_per_tok = _g(text_cfg, "num_experts_per_tok") or 0
    intermediate_size  = _g(text_cfg, "intermediate_size", 0)
    is_moe = bool(num_experts)

    layers = []
    for i in range(num_layers):
        if sliding_pattern and sliding_window:
            mask = "sliding" if (i % sliding_pattern) != (sliding_pattern - 1) else "causal"
            win  = sliding_window if mask == "sliding" else None
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
        )

        if is_moe:
            ffn = FFNSpec(
                kind="moe",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=True,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                expert_intermediate_size=intermediate_size,
            )
        else:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=True,
            )

        layers.append(decoder_layer(i, attn, ffn, hidden_size))

    vocab_size = _g(text_cfg, "vocab_size", 0) or _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(
        _g(text_cfg, "tie_word_embeddings", _g(cfg, "tie_word_embeddings", False))
    )
    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_g(text_cfg, "max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=decoder_extras(vocab_size, hidden_size, tie_word_embeddings),
    )


def _text_config(cfg: Any) -> Any:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _WRAPPED_TYPES:
        sub = _g(cfg, "text_config")
        if sub is not None:
            return sub
    return cfg
