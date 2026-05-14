"""Adapter for AllenAI OLMo / OLMoE decoder configs.

OLMo-2 used to be routed through the Llama-like adapter.  Keeping OLMo here
lets OLMo 1, OLMo-2, and OLMoE share one family boundary while preserving the
pieces that actually matter for rendering: QK norm, dense vs MoE FFNs, and
LayerNorm/RMSNorm defaults.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"olmo", "olmo2", "olmoe"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = [a.lower() for a in (_g(cfg, "architectures") or [])]
    return any("olmo" in a for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "olmo")
    model_type = (_g(cfg, "model_type") or "olmo").lower()

    num_layers = _first(cfg, "num_hidden_layers", "n_layers", "num_layers", default=0)
    hidden_size = _first(cfg, "hidden_size", "d_model", "dim", default=0)
    num_heads = _first(cfg, "num_attention_heads", "n_heads", default=0)
    num_kv_heads = _first(cfg, "num_key_value_heads", "n_kv_heads", default=num_heads)
    head_dim = _first(cfg, "head_dim", "d_head", default=None) or (
        hidden_size // num_heads if num_heads else None
    )
    intermediate_size = (
        _first(cfg, "intermediate_size", "mlp_hidden_size", "ffn_hidden_size", default=0)
        or _mlp_ratio_size(cfg, hidden_size)
    )
    activation = (_first(cfg, "hidden_act", "activation_type", "activation", default="silu") or "silu").lower()

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    use_qk_norm = bool(_g(cfg, "use_qk_norm", False) or _g(cfg, "qk_norm", False))
    norm_kind = _norm_kind(cfg)

    num_experts = _first(cfg, "num_experts", "n_experts", default=0) or 0
    num_experts_per_tok = _first(
        cfg,
        "num_experts_per_tok",
        "top_k",
        "top_k_experts",
        default=0,
    ) or 0
    expert_intermediate = _first(
        cfg,
        "expert_intermediate_size",
        "moe_intermediate_size",
        "expert_hidden_size",
        default=0,
    ) or intermediate_size

    layers = []
    for i in range(num_layers):
        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask="causal",
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
                expert_intermediate_size=expert_intermediate,
            )
        else:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=_is_gated(activation),
            )
        layers.append(decoder_layer(i, attn, ffn, hidden_size, norm_kind=norm_kind))

    vocab_size = _first(cfg, "vocab_size", "padded_vocab_size", default=0)
    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)
    if model_type == "olmoe" or num_experts:
        extras["moe"] = {
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
        }

    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_first(cfg, "max_position_embeddings", "max_sequence_length", "n_positions"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=extras,
    )


def _first(cfg: Any, *names: str, default=None):
    for name in names:
        value = _g(cfg, name)
        if value is not None:
            return value
    return default


def _mlp_ratio_size(cfg: Any, hidden_size: int) -> int:
    ratio = _g(cfg, "mlp_ratio")
    if ratio and hidden_size:
        try:
            return int(hidden_size * float(ratio))
        except (TypeError, ValueError):
            return 0
    return 0


def _norm_kind(cfg: Any) -> str:
    norm = (_g(cfg, "norm_type") or _g(cfg, "layer_norm_type") or "").lower()
    if "rms" in norm:
        return "rmsnorm"
    if "layer" in norm:
        return "layernorm"
    return "rmsnorm" if (_g(cfg, "model_type") or "").lower() == "olmo2" else "layernorm"


def _is_gated(activation: str) -> bool:
    return activation.replace("-", "_") in {"swiglu", "geglu", "silu"}
