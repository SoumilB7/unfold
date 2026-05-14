"""Adapter for Databricks DBRX configs.

DBRX is decoder-only, but its public HuggingFace config does not use the
standard flat transformer field names for the important parts:

* attention lives under ``attn_config`` with ``kv_n_heads``
* MoE lives under ``ffn_config`` with ``moe_num_experts`` / ``moe_top_k``

That nested shape is why DBRX deserves a dedicated adapter instead of generic
fallback parsing.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"dbrx"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = [a.lower() for a in (_g(cfg, "architectures") or [])]
    return any("dbrx" in a for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "dbrx")
    attn_cfg = _g(cfg, "attn_config") or {}
    ffn_cfg = _g(cfg, "ffn_config") or {}

    hidden_size = _g(cfg, "d_model") or _g(cfg, "hidden_size") or 0
    num_layers = _g(cfg, "n_layers") or _g(cfg, "num_hidden_layers") or 0
    num_heads = _g(cfg, "n_heads") or _g(cfg, "num_attention_heads") or 0
    num_kv_heads = _g(attn_cfg, "kv_n_heads") or _g(cfg, "num_key_value_heads") or num_heads
    head_dim = _g(attn_cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    activation = _activation_name(_g(ffn_cfg, "ffn_act_fn")) or "silu"
    expert_hidden = _g(ffn_cfg, "ffn_hidden_size") or _g(cfg, "intermediate_size") or 0
    num_experts = _g(ffn_cfg, "moe_num_experts") or _g(ffn_cfg, "num_experts") or 4
    num_experts_per_tok = _g(ffn_cfg, "moe_top_k") or _g(ffn_cfg, "num_experts_per_tok") or 1

    layers = []
    for i in range(num_layers):
        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask="causal",
        )
        ffn = FFNSpec(
            kind="moe",
            activation=activation,
            intermediate_size=expert_hidden,
            gated=True,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            expert_intermediate_size=expert_hidden,
        )
        layers.append(decoder_layer(i, attn, ffn, hidden_size, norm_kind="rmsnorm"))

    vocab_size = _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)
    extras["moe"] = {
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
    }
    extras["dbrx"] = {
        "clip_qkv": _g(attn_cfg, "clip_qkv"),
        "rope_theta": _g(attn_cfg, "rope_theta"),
        "router_aux_loss_coef": _g(cfg, "router_aux_loss_coef"),
    }

    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_g(cfg, "max_seq_len") or _g(cfg, "max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=extras,
    )


def _activation_name(value: Any) -> str | None:
    if isinstance(value, dict):
        name = value.get("name")
        return name.lower() if isinstance(name, str) else name
    if isinstance(value, str):
        return value.lower()
    return None
