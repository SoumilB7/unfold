"""Adapter for Cohere's Command R family.

Covers:
* Command R / Command R+  — model_type: "cohere"
  Dense GQA decoder with per-head Q/K normalisation (``use_qk_norm: true``).
  No sliding window — all layers use full causal attention.

* Command R7B (Cohere2)   — model_type: "cohere2"
  Same as above but adds a sliding-window / global interleaving pattern
  (``sliding_window`` + ``sliding_window_pattern``).

The ``qk_norm`` flag on ``AttentionSpec`` marks layers that apply separate
layer norms to Q and K projections before the dot product, which is a
distinguishing visual feature of this family.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"cohere", "cohere2"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("cohere" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "cohere")
    model_type = (_g(cfg, "model_type") or "cohere").lower()

    num_layers    = _g(cfg, "num_hidden_layers", 0)
    hidden_size   = _g(cfg, "hidden_size", 0)
    num_heads     = _g(cfg, "num_attention_heads", 0)
    num_kv_heads  = _g(cfg, "num_key_value_heads", num_heads)
    head_dim      = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = _g(cfg, "intermediate_size", 0)
    activation    = (_g(cfg, "hidden_act") or "silu").lower()
    use_qk_norm   = bool(_g(cfg, "use_qk_norm", False))

    # Cohere2 adds sliding-window attention (Command R7B)
    sliding_window  = _g(cfg, "sliding_window")
    sliding_pattern = _g(cfg, "sliding_window_pattern")

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    layers = []
    for i in range(num_layers):
        if model_type == "cohere2" and sliding_pattern and sliding_window:
            # Cohere2: last layer in each group is global, rest are sliding
            is_full = (i % sliding_pattern) == (sliding_pattern - 1)
            mask = "global" if is_full else "sliding"
            win  = None if is_full else sliding_window
        elif model_type == "cohere2" and sliding_window:
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
        ffn = FFNSpec(
            kind="dense",
            activation=activation,
            intermediate_size=intermediate_size,
            gated=True,
        )
        layers.append(decoder_layer(i, attn, ffn, hidden_size))

    vocab_size = _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)

    logit_scale = _g(cfg, "logit_scale")
    if logit_scale:
        extras["logit_scale"] = logit_scale

    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_g(cfg, "max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=extras,
    )
