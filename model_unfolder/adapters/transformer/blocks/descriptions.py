"""Labels, titles, and descriptions for transformer block specs."""
from __future__ import annotations

from ....ir import AttentionSpec, FFNSpec
from ....labels import activation_label
from ..common import format_dim as _fmt


def attention_label(attention: AttentionSpec) -> list[str]:
    kind = attention.kind
    if kind == "mla":
        return ["Multi-Head Latent", "Attention"]
    if kind == "mqa":
        return ["Multi-Query", "Attention"]
    if kind == "gqa":
        tag = "(QK-Norm)" if attention.qk_norm else "Attention"
        return ["Grouped-Query", tag]
    if kind == "ssm":
        shared_tag = "(Shared)" if attention.shared else "Block"
        return ["Selective SSM", shared_tag]
    if kind == "recurrent":
        return ["Linear Recurrent", "Unit (LRU)"]
    if kind == "rwkv":
        return ["RWKV", "Token-Mixing"]
    if kind == "linear":
        return ["Linear", "Attention"]

    tags = []
    if attention.qk_norm:
        tags.append("QK-Norm")
    if attention.no_rope:
        tags.append("NoPE")
    if tags:
        return ["Multi-Head Attn", f"({', '.join(tags)})"]
    return ["Multi-Head", "Attention"]


def attention_title(attention: AttentionSpec) -> str:
    if attention.kind == "mqa":
        base = "Multi-query attention"
    else:
        base = {
            "mla": "Multi-head latent attention",
            "gqa": "Grouped-query attention",
            "ssm": "Selective state-space model (Mamba)",
            "recurrent": "Linear Recurrent Unit (LRU)",
            "rwkv": "RWKV token-mixing",
            "linear": "Linear attention",
        }.get(attention.kind, "Attention")
    extras = []
    if attention.qk_norm:
        extras.append("QK-Norm")
    if attention.shared:
        extras.append("weight-shared")
    if attention.no_rope:
        extras.append("NoPE")
    return f"{base} ({', '.join(extras)})" if extras else base


def describe_attention(attention: AttentionSpec) -> str:
    if attention.kind == "mla":
        text = (
            f"Multi-head latent attention; {attention.num_heads} heads; "
            f"KV LoRA {_fmt(attention.kv_lora_rank)}"
        )
        if attention.q_lora_rank:
            text += f"; Q LoRA {_fmt(attention.q_lora_rank)}"
        return text
    if attention.kind == "mqa":
        return f"Multi-query; {attention.num_heads} Q / 1 KV head"
    if attention.kind == "gqa":
        return (
            f"Grouped-query; {attention.num_heads} Q / {attention.num_kv_heads} KV heads; "
            f"head dim {_fmt(attention.head_dim)}"
        )
    if attention.kind == "ssm":
        shared = "; weight-shared across positions" if attention.shared else ""
        return f"Selective SSM; state dim {_fmt(attention.head_dim)}{shared}"
    if attention.kind == "recurrent":
        return f"Linear Recurrent Unit; LRU width {_fmt(attention.head_dim)}"
    if attention.kind == "rwkv":
        return f"RWKV token-mixing; {attention.num_heads} heads"
    if attention.kind == "linear":
        return (
            f"Linear attention; {attention.num_heads} Q / {attention.num_kv_heads} KV; "
            f"head dim {_fmt(attention.head_dim)}"
        )

    extras = []
    if attention.qk_norm:
        extras.append("QK-Norm")
    if attention.no_rope:
        extras.append("NoPE")
    suffix = f"; {', '.join(extras)}" if extras else ""
    return f"Multi-head; {attention.num_heads} heads; head dim {_fmt(attention.head_dim)}{suffix}"


def describe_ffn(ffn: FFNSpec) -> str:
    if ffn.kind == "moe":
        text = f"MoE; {_fmt(ffn.num_experts)} experts; top-{ffn.num_experts_per_tok}"
        if ffn.num_shared_experts:
            text += f" + {ffn.num_shared_experts} shared"
        if ffn.num_experts and ffn.num_experts_per_tok:
            text += f"; {100 * ffn.num_experts_per_tok / ffn.num_experts:.1f}% active"
        text += f"; expert hidden {_fmt(ffn.expert_intermediate_size or ffn.intermediate_size)}"
        return text
    gated = "gated " if ffn.gated else ""
    return f"{gated}FFN; {activation_label(ffn.activation)}; hidden {_fmt(ffn.intermediate_size)}"
