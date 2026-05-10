"""Adapter for pure-SSM Mamba and Mamba-2 models (state-spaces/).

Unlike transformer hybrids (Jamba, Falcon-H1), these models have **no
attention layers at all** — every layer is a selective state-space block.

Mamba (v1):  model_type "mamba"   — selective SSM, S4-style gating
Mamba-2:     model_type "mamba2"  — state-space duality, chunk-parallel

Config uses non-standard field names:
  ``n_layer``   → number of layers   (fallback: ``num_hidden_layers``)
  ``d_model``   → hidden size        (fallback: ``hidden_size``)
  ``d_state``   → SSM state size     (Mamba-1: 16; Mamba-2: 128)
  ``d_conv``    → convolution width  (typically 4)
  ``expand``    → inner-dim factor   (typically 2)
  ``headdim``   → head dim (Mamba-2 only)

IR mapping:
  AttentionSpec(kind="ssm")  — the selective SSM block
  FFNSpec(kind="dense", intermediate_size=d_model*expand) — inner expansion
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"mamba", "mamba2"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    if any(a.lower().startswith("mamba") or "mambafor" in a.lower() for a in arches):
        return True
    # Old state-spaces configs have no model_type or architectures.
    # The combination n_layer + d_model (without num_hidden_layers / hidden_size)
    # is unique to Mamba among configs that reach us without a model_type.
    if (not model_type
            and _g(cfg, "n_layer") is not None
            and _g(cfg, "d_model") is not None
            and _g(cfg, "num_hidden_layers") is None):
        return True
    return False


def parse(cfg: Any) -> ModelIR:
    model_type = (_g(cfg, "model_type") or "mamba").lower()
    arch_name  = architecture_name(cfg, model_type)

    # Mamba uses d_model / n_layer; allow transformer-style names as fallback
    num_layers   = _g(cfg, "n_layer") or _g(cfg, "num_hidden_layers", 0)
    hidden_size  = _g(cfg, "d_model") or _g(cfg, "hidden_size", 0)
    d_state      = _g(cfg, "d_state") or _g(cfg, "state_size") or 16
    d_conv       = _g(cfg, "d_conv") or 4
    expand       = _g(cfg, "expand") or 2
    vocab_size   = _g(cfg, "vocab_size", 0)

    # Mamba-2 introduces explicit heads
    headdim      = _g(cfg, "headdim") or _g(cfg, "head_dim") or d_state
    num_heads    = (hidden_size * expand // headdim) if headdim else 0

    layers = []
    for i in range(num_layers):
        attn = AttentionSpec(
            kind="ssm",
            num_heads=num_heads if model_type == "mamba2" else 0,
            num_kv_heads=0,
            head_dim=headdim if model_type == "mamba2" else d_state,
            mask="causal",
        )
        # The SSM block includes an inner expansion projection; surface its width
        ffn = FFNSpec(
            kind="dense",
            activation="silu",
            intermediate_size=hidden_size * expand,
            gated=False,
        )
        layers.append(decoder_layer(i, attn, ffn, hidden_size))

    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)
    extras["mamba"] = {
        "version": 2 if model_type == "mamba2" else 1,
        "d_state": d_state,
        "d_conv": d_conv,
        "expand": expand,
        "pure": True,
    }

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
