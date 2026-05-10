"""Adapter for Llama, Phi, and similar GQA/MHA dense models."""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_FAMILIES = {"llama", "phi3", "gemma"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _FAMILIES:
        return True
    arches = _g(cfg, "architectures") or []
    return any(
        any(fam in a.lower() for fam in ("llama", "phi3"))
        for a in arches
    )


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "llama")

    num_layers = _g(cfg, "num_hidden_layers", 0)
    num_heads = _g(cfg, "num_attention_heads", 0)
    num_kv_heads = _g(cfg, "num_key_value_heads", num_heads)
    hidden_size = _g(cfg, "hidden_size", 0)
    head_dim = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = _g(cfg, "intermediate_size", 0)
    activation = (_g(cfg, "hidden_act", "silu") or "silu").lower()

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    sliding_window = _g(cfg, "sliding_window")
    sliding_pattern = _g(cfg, "sliding_window_pattern")
    layer_types = _g(cfg, "layer_types")

    layers = []
    for i in range(num_layers):
        if layer_types and i < len(layer_types):
            layer_type = layer_types[i]
            if "sliding" in layer_type:
                mask, win = "sliding", sliding_window
            else:
                mask, win = "causal", None
        elif sliding_pattern and sliding_window:
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
        )
        ffn = FFNSpec(
            kind="dense",
            activation=activation,
            intermediate_size=intermediate_size,
            gated=True,
        )
        layers.append(decoder_layer(i, attn, ffn, hidden_size))

    vocab_size = _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))
    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_g(cfg, "max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=decoder_extras(vocab_size, hidden_size, tie_word_embeddings),
    )
