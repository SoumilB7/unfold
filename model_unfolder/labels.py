"""Renderer-agnostic vocabulary for talking about transformer specs.

Anything that needs to refer to attention masks ("SWA", "full"), attention
kinds ("GQA"), or build a human description of an attention / FFN block goes
through this module.  Keeping it in one place means changing the wording
(swap "SWA" for "Sliding", say) only requires editing this file — no scavenger
hunt across the renderer.

The functions are pure and operate on plain ``dict``-shaped specs (i.e. what
``ModelIR.to_dict()`` produces), so they're equally useful inside the HTML
renderer, a future text/markdown renderer, or notebook utilities.
"""
from __future__ import annotations

_MASK_SHORT = {
    "sliding": "SWA",
    "global": "full",
    "causal": "causal",
    "chunked": "chunked",
}
_MASK_LONG = {
    "sliding": "Sliding-window",
    "global": "Full / global",
    "causal": "Causal",
    "chunked": "Chunked",
}
_MASK_TITLE = {
    "sliding": "Sliding-window attention",
    "global": "Full-context attention",
    "causal": "Causal attention",
    "chunked": "Chunked attention",
}

_KIND_SHORT = {
    "mla": "MLA",
    "gqa": "GQA",
    "mqa": "MQA",
    "mha": "MHA",
    "ssm": "SSM",
    "recurrent": "LRU",
    "linear": "Lin-Attn",
    "rwkv": "RWKV",
}
_KIND_LONG = {
    "mla": "Multi-head latent attention",
    "gqa": "Grouped-query attention",
    "mqa": "Multi-query attention",
    "mha": "Multi-head attention",
    "ssm": "Selective state-space model (Mamba)",
    "recurrent": "Linear Recurrent Unit (LRU)",
    "linear": "Linear attention",
    "rwkv": "RWKV token-mixing",
}
_ACTIVATION_LABELS = {
    "gelu": "GELU",
    "gelu_new": "GELU",
    "gelu_fast": "GELU",
    "gelu_pytorch_tanh": "GELU",
    "relu": "ReLU",
    "silu": "SiLU",
    "swish": "SiLU",
    "geglu": "GEGLU",
    "swiglu": "SwiGLU",
}


def mask_short(attention: dict) -> str:
    """Compact mask tag — ``"SWA"`` / ``"full"`` / ``"causal"``."""
    return _MASK_SHORT.get(attention.get("mask", "causal"), "causal")


def mask_long(attention: dict) -> str:
    """Human-readable mask label — ``"Sliding-window"`` / ``"Full / global"``."""
    return _MASK_LONG.get(attention.get("mask", "causal"), "Causal")


def mask_title(attention: dict) -> str:
    """Tooltip-style mask description."""
    return _MASK_TITLE.get(attention.get("mask", "causal"), "Causal attention")


def mask_chip(attention: dict) -> str:
    """One-line chip; for sliding includes the window size: ``"SWA 1,024"``."""
    label = mask_short(attention)
    window = attention.get("window_size")
    if window and is_sliding(attention):
        return f"{label} {_fmt_int(window)}"
    return label


def kind_short(attention: dict) -> str:
    short = _KIND_SHORT.get(attention.get("kind", ""), "MHA")
    tags = []
    if attention.get("qk_norm"):
        tags.append("QK-Norm")
    if attention.get("shared"):
        tags.append("Shared")
    if attention.get("no_rope"):
        tags.append("NoPE")
    return f"{short} ({', '.join(tags)})" if tags else short


def kind_long(attention: dict) -> str:
    base = _KIND_LONG.get(attention.get("kind", ""), "Multi-head attention")
    extras = []
    if attention.get("qk_norm"):
        extras.append("per-head Q/K normalisation")
    if attention.get("shared"):
        extras.append("weight-shared across positions")
    if attention.get("no_rope"):
        extras.append("no positional encoding (NoPE)")
    return f"{base}; {'; '.join(extras)}" if extras else base


def is_sliding(attention: dict) -> bool:
    return attention.get("mask") == "sliding"


def is_global(attention: dict) -> bool:
    return attention.get("mask") == "global"


def kv_shared(attention: dict) -> bool:
    """True when this layer reuses K/V from an earlier layer (Gemma 4 small)."""
    return attention.get("kv_source_layer") is not None


def activation_label(name: str | None) -> str:
    """Display label for activation names stored in configs.

    Configs often expose backend-specific names such as
    ``gelu_pytorch_tanh``.  Diagrams should show the mathematical operation,
    not the implementation detail.
    """
    key = (name or "").strip().lower().replace("-", "_")
    if key in _ACTIVATION_LABELS:
        return _ACTIVATION_LABELS[key]
    if key.startswith("gelu"):
        return "GELU"
    return key.replace("_", " ").title() if key else "Activation"


def describe_attention(attention: dict) -> str:
    """Multi-clause human description suitable for tooltips and cards."""
    kind = attention.get("kind")
    if kind == "mla":
        text = (
            f"Multi-head latent attention; {attention.get('num_heads')} heads; "
            f"KV LoRA {_fmt_int(attention.get('kv_lora_rank'))}"
        )
        if attention.get("q_lora_rank"):
            text += f"; Q LoRA {_fmt_int(attention.get('q_lora_rank'))}"
        text += "; cache ports mark latent write/read state"
    elif kind == "mqa":
        text = f"Multi-query; {attention.get('num_heads')} Q / 1 KV head; cache ports mark K/V write/read state"
    elif kind == "gqa":
        text = (
            f"Grouped-query; {attention.get('num_heads')} Q / "
            f"{attention.get('num_kv_heads')} KV heads; "
            f"head dim {_fmt_int(attention.get('head_dim'))}; cache ports mark K/V write/read state"
        )
    elif kind == "ssm":
        text = f"Selective SSM (Mamba); state dim {_fmt_int(attention.get('head_dim'))}"
    elif kind == "recurrent":
        text = f"Linear Recurrent Unit; LRU width {_fmt_int(attention.get('head_dim'))}"
    elif kind == "rwkv":
        text = f"RWKV token-mixing; {attention.get('num_heads')} heads"
    elif kind == "linear":
        text = (
            f"Linear attention; {attention.get('num_heads')} Q / "
            f"{attention.get('num_kv_heads')} KV heads; "
            f"head dim {_fmt_int(attention.get('head_dim'))}"
        )
    else:
        text = (
            f"Multi-head; {attention.get('num_heads')} heads; "
            f"head dim {_fmt_int(attention.get('head_dim'))}; cache ports mark K/V write/read state"
        )
    if is_sliding(attention) and attention.get("window_size"):
        text += f"; sliding window {_fmt_int(attention.get('window_size'))}"
    # Surface structural flags as annotations
    extras = []
    if attention.get("qk_norm"):
        extras.append("QK-Norm")
    if attention.get("shared"):
        extras.append("weight-shared")
    if attention.get("no_rope"):
        extras.append("NoPE")
    if extras:
        text += "; " + ", ".join(extras)
    return text


def describe_ffn(ffn: dict) -> str:
    """Multi-clause human description of an FFN / MoE block."""
    if ffn.get("kind") == "moe":
        text = f"MoE; {_fmt_int(ffn.get('num_experts'))} experts; top-{ffn.get('num_experts_per_tok')}"
        if ffn.get("num_shared_experts"):
            text += f" + {ffn.get('num_shared_experts')} shared"
        if ffn.get("num_experts") and ffn.get("num_experts_per_tok"):
            text += f"; {100 * ffn['num_experts_per_tok'] / ffn['num_experts']:.1f}% active"
        text += f"; expert hidden {_fmt_int(ffn.get('expert_intermediate_size') or ffn.get('intermediate_size'))}"
        return text
    gated = "gated " if ffn.get("gated") else ""
    return f"{gated}FFN; {activation_label(ffn.get('activation'))}; hidden {_fmt_int(ffn.get('intermediate_size'))}"


def _fmt_int(value) -> str:
    if value is None:
        return "?"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)
