"""Adapter for Llama, Mistral, Qwen, and similar GQA/MHA dense models."""
from __future__ import annotations
from typing import Any
from ..ir import ModelIR, LayerSpec, AttentionSpec, FFNSpec


_FAMILIES = {"llama", "mistral", "qwen2", "qwen3", "phi3", "gemma"}


def _g(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def matches(cfg: Any) -> bool:
    arches = _g(cfg, "architectures") or []
    model_type = _g(cfg, "model_type", "")
    for a in arches:
        if any(fam in a.lower() for fam in ("llama", "mistral", "qwen", "phi3")):
            return True
    if model_type in _FAMILIES:
        return True
    return False


def parse(cfg: Any) -> ModelIR:
    num_layers = _g(cfg, "num_hidden_layers", 0)
    num_heads = _g(cfg, "num_attention_heads", 0)
    num_kv_heads = _g(cfg, "num_key_value_heads", num_heads)
    hidden_size = _g(cfg, "hidden_size", 0)
    head_dim = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    sliding_window = _g(cfg, "sliding_window")
    sliding_pattern = _g(cfg, "sliding_window_pattern")
    layer_types = _g(cfg, "layer_types")

    intermediate_size = _g(cfg, "intermediate_size", 0)
    activation = (_g(cfg, "hidden_act", "silu") or "silu").lower()

    architectures = _g(cfg, "architectures") or []
    arch_name = architectures[0] if architectures else _g(cfg, "model_type", "llama")
    name = _g(cfg, "_name_or_path") or _g(cfg, "name_or_path") or arch_name

    layers = []
    for i in range(num_layers):
        if layer_types and i < len(layer_types):
            t = layer_types[i]
            if "sliding" in t:
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
        layers.append(LayerSpec(index=i, attention=attn, ffn=ffn))

    return ModelIR(
        name=str(name).split("/")[-1] if name else arch_name,
        architecture=arch_name,
        vocab_size=_g(cfg, "vocab_size", 0),
        hidden_size=hidden_size,
        max_position_embeddings=_g(cfg, "max_position_embeddings"),
        tie_word_embeddings=bool(_g(cfg, "tie_word_embeddings", False)),
        layers=layers,
    )
