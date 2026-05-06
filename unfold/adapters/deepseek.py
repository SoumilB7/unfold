"""Adapter for DeepSeek-V3 and Kimi K2 (which shares the architecture)."""
from __future__ import annotations
from typing import Any
from ..ir import ModelIR, LayerSpec, AttentionSpec, FFNSpec


def _g(cfg: Any, name: str, default=None):
    """Get a config value whether cfg is a HF PretrainedConfig or a dict."""
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def matches(cfg: Any) -> bool:
    arches = _g(cfg, "architectures") or []
    model_type = _g(cfg, "model_type", "")
    if any("DeepseekV3" in a or "DeepseekV2" in a or "Kimi" in a for a in arches):
        return True
    if model_type in {"deepseek_v3", "deepseek_v2", "kimi"}:
        return True
    return False


def parse(cfg: Any) -> ModelIR:
    num_layers = _g(cfg, "num_hidden_layers", 0)
    hidden_size = _g(cfg, "hidden_size", 0)
    num_heads = _g(cfg, "num_attention_heads", 0)
    num_kv_heads = _g(cfg, "num_key_value_heads", num_heads)

    kv_lora_rank = _g(cfg, "kv_lora_rank")
    q_lora_rank = _g(cfg, "q_lora_rank")
    qk_rope_head_dim = _g(cfg, "qk_rope_head_dim")
    qk_nope_head_dim = _g(cfg, "qk_nope_head_dim")
    v_head_dim = _g(cfg, "v_head_dim")

    attn_kind = "mla" if kv_lora_rank is not None else "gqa"
    head_dim = (qk_nope_head_dim or 0) + (qk_rope_head_dim or 0) or _g(cfg, "head_dim")

    intermediate_size = _g(cfg, "intermediate_size", 0)
    moe_intermediate_size = _g(cfg, "moe_intermediate_size", intermediate_size)
    n_routed_experts = _g(cfg, "n_routed_experts") or _g(cfg, "num_experts", 0) or 0
    n_shared_experts = _g(cfg, "n_shared_experts") or _g(cfg, "num_shared_experts", 0) or 0
    num_experts_per_tok = _g(cfg, "num_experts_per_tok", 0)
    first_k_dense_replace = _g(cfg, "first_k_dense_replace", 0)
    moe_layer_freq = _g(cfg, "moe_layer_freq", 1)
    activation = (_g(cfg, "hidden_act", "silu") or "silu").lower()

    architectures = _g(cfg, "architectures") or []
    arch_name = architectures[0] if architectures else _g(cfg, "model_type", "deepseek_v3")
    name = _g(cfg, "_name_or_path") or _g(cfg, "name_or_path") or arch_name

    layers = []
    for i in range(num_layers):
        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            rope_dim=qk_rope_head_dim,
            mask="causal",
        )
        is_moe_layer = (
            n_routed_experts > 0
            and i >= first_k_dense_replace
            and (i - first_k_dense_replace) % max(moe_layer_freq, 1) == 0
        )
        if is_moe_layer:
            ffn = FFNSpec(
                kind="moe",
                activation=activation,
                intermediate_size=moe_intermediate_size,
                gated=True,
                num_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_shared_experts=n_shared_experts,
                expert_intermediate_size=moe_intermediate_size,
            )
        else:
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
        extras={
            "v_head_dim": v_head_dim,
            "first_k_dense_replace": first_k_dense_replace,
            "moe_layer_freq": moe_layer_freq,
        },
    )
