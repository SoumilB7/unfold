"""Reusable attention-family child block declarations."""
from __future__ import annotations

from ....ir import AttentionSpec
from ..common import format_dim as _fmt


def attention_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[dict]:
    builders = {
        "mla": _mla_child_blocks,
        "ssm": _ssm_child_blocks,
        "recurrent": _recurrent_child_blocks,
        "rwkv": _rwkv_child_blocks,
        "linear": _linear_attention_child_blocks,
    }
    builder = builders.get(attention.kind, _sdpa_child_blocks)
    return builder(attention, hidden_size)


def _sdpa_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[dict]:
    hidden = _fmt(hidden_size)
    num_heads = attention.num_heads or 0
    num_kv_heads = attention.num_kv_heads or num_heads
    head_dim = attention.head_dim or 0
    q_per_group = num_heads // num_kv_heads if (num_heads and num_kv_heads and num_heads % num_kv_heads == 0) else None
    q_out = _fmt(num_heads * head_dim) if (num_heads and head_dim) else hidden
    kv_out = _fmt(num_kv_heads * head_dim) if (num_kv_heads and head_dim) else hidden
    d_k = _fmt(head_dim) if head_dim else "d_k"
    attention_title, attention_desc = _sdpa_operation_meta(attention, num_heads, num_kv_heads, d_k, q_per_group)
    return [
        {
            "id": "q_proj",
            "title": "Query projection",
            "description": f"Linear; {hidden} -> {q_out}  ({num_heads} heads x {d_k} dims)",
        },
        {
            "id": "k_proj",
            "title": "Key projection",
            "description": f"Linear; {hidden} -> {kv_out}  ({num_kv_heads} KV-heads x {d_k} dims)",
        },
        {
            "id": "v_proj",
            "title": "Value projection",
            "description": f"Linear; {hidden} -> {kv_out}  ({num_kv_heads} KV-heads x {d_k} dims)",
        },
        {
            "id": "qkv_dot",
            "title": attention_title,
            "description": attention_desc,
        },
        {
            "id": "o_proj",
            "title": "Output projection",
            "description": f"Linear; {q_out} -> {hidden}  (recombines all {num_heads} heads)",
        },
    ]


def _sdpa_operation_meta(
    attention: AttentionSpec,
    num_heads: int,
    num_kv_heads: int,
    d_k: str,
    q_per_group: int | None,
) -> tuple[str, str]:
    if attention.kind == "mqa":
        return (
            "Multi-query scaled dot-product attention",
            (
                f"scores = softmax(QK^T / sqrt({d_k})); "
                f"{num_heads} query heads share one K/V head"
            ),
        )
    if attention.kind == "gqa":
        group = (
            f"; each KV head serves {q_per_group} query heads"
            if q_per_group
            else ""
        )
        return (
            "Grouped scaled dot-product attention",
            (
                f"scores = softmax(QK^T / sqrt({d_k})); "
                f"{num_heads} query heads attend through {num_kv_heads} shared KV heads{group}"
            ),
        )
    return (
        "Scaled dot-product attention",
        (
            f"scores = softmax(QK^T / sqrt({d_k})); "
            "context = scores * V; "
            f"output shape [batch, {num_heads}, seq, {d_k}]"
        ),
    )


def _mla_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[dict]:
    hidden = _fmt(hidden_size)
    q_rank = _fmt(attention.q_lora_rank) if attention.q_lora_rank else "direct"
    kv_rank = _fmt(attention.kv_lora_rank)
    rope = _fmt(attention.rope_dim)
    num_heads = attention.num_heads or 0
    head_dim = attention.head_dim or 0
    q_out = _fmt(num_heads * head_dim) if (num_heads and head_dim) else hidden
    return [
        {
            "id": "mla_q",
            "label": "Q path",
            "title": "Query path",
            "description": (
                f"Q projection with LoRA rank {q_rank}"
                if attention.q_lora_rank
                else f"Q projection; {hidden} -> {q_out}"
            ),
        },
        {
            "id": "mla_kv_down",
            "label": "KV down",
            "title": "KV latent down-projection",
            "description": f"Compress K/V context; {hidden} -> latent rank {kv_rank} plus RoPE {rope}",
        },
        {
            "id": "mla_kv_up",
            "label": "KV up",
            "title": "KV latent up-projection",
            "description": f"Expand latent K/V for {num_heads} query heads",
        },
        {
            "id": "mla_rope",
            "label": "RoPE split",
            "title": "RoPE side channel",
            "description": f"Rotary positional slice; dim {rope}",
        },
        {
            "id": "mla_attn",
            "label": "Latent attention",
            "title": "Multi-head latent attention",
            "description": "Attention over decompressed latent K/V plus the RoPE side channel",
        },
        {
            "id": "o_proj",
            "label": "Linear (out)",
            "title": "Output projection",
            "description": f"Linear; {q_out} -> {hidden}",
        },
    ]


def _ssm_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[dict]:
    hidden = _fmt(hidden_size)
    state = _fmt(attention.head_dim)
    return [
        {
            "id": "ssm_in_proj",
            "label": "Input projection",
            "title": "SSM input projection",
            "description": f"Project hidden activations into SSM channels; hidden {hidden}",
        },
        {
            "id": "ssm_conv",
            "label": "Local conv",
            "title": "Short convolution",
            "description": "Depthwise local mixing before the state-space recurrence",
        },
        {
            "id": "ssm_scan",
            "label": "Selective scan",
            "title": "Selective state-space scan",
            "description": f"Token recurrence with state dimension {state}",
        },
        {
            "id": "ssm_gate",
            "label": "Gate",
            "title": "SSM gate",
            "description": "Element-wise gate controlling the recurrent output",
        },
        {
            "id": "ssm_out_proj",
            "label": "Output projection",
            "title": "SSM output projection",
            "description": f"Project SSM channels back to hidden dim {hidden}",
        },
    ]


def _recurrent_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[dict]:
    hidden = _fmt(hidden_size)
    width = _fmt(attention.head_dim)
    return [
        {
            "id": "lru_in_proj",
            "label": "Input projection",
            "title": "LRU input projection",
            "description": f"Linear; hidden {hidden} -> recurrent width {width}",
        },
        {
            "id": "lru_state",
            "label": "Recurrent state",
            "title": "Linear recurrent state",
            "description": f"State update over sequence positions; width {width}",
        },
        {
            "id": "lru_gate",
            "label": "Gate",
            "title": "Recurrent gate",
            "description": "Element-wise gate controlling recurrent features",
        },
        {
            "id": "lru_out_proj",
            "label": "Output projection",
            "title": "LRU output projection",
            "description": f"Linear; recurrent width {width} -> hidden {hidden}",
        },
    ]


def _rwkv_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[dict]:
    hidden = _fmt(hidden_size)
    heads = attention.num_heads or 0
    return [
        {
            "id": "rwkv_receptance",
            "label": "Receptance",
            "title": "Receptance gate",
            "description": f"Token-wise gate over hidden dim {hidden}",
        },
        {
            "id": "rwkv_key",
            "label": "Key",
            "title": "RWKV key projection",
            "description": f"Key-like channel mixing over {heads} recurrent heads",
        },
        {
            "id": "rwkv_value",
            "label": "Value",
            "title": "RWKV value projection",
            "description": "Value channel sent through time-mixing recurrence",
        },
        {
            "id": "rwkv_time_mix",
            "label": "Time-mix",
            "title": "Time-decay recurrence",
            "description": "Linear-time weighted recurrence replacing self-attention",
        },
        {
            "id": "rwkv_out",
            "label": "Output projection",
            "title": "RWKV output projection",
            "description": f"Project mixed channels back to hidden dim {hidden}",
        },
    ]


def _linear_attention_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[dict]:
    hidden = _fmt(hidden_size)
    num_heads = attention.num_heads or 0
    num_kv_heads = attention.num_kv_heads or num_heads
    head_dim = attention.head_dim or 0
    q_out = _fmt(num_heads * head_dim) if (num_heads and head_dim) else hidden
    kv_out = _fmt(num_kv_heads * head_dim) if (num_kv_heads and head_dim) else hidden
    return [
        {
            "id": "q_proj",
            "label": "Linear (Q)",
            "title": "Query projection",
            "description": f"Linear; {hidden} -> {q_out}",
        },
        {
            "id": "k_proj",
            "label": "Linear (K)",
            "title": "Key projection",
            "description": f"Linear; {hidden} -> {kv_out}",
        },
        {
            "id": "v_proj",
            "label": "Linear (V)",
            "title": "Value projection",
            "description": f"Linear; {hidden} -> {kv_out}",
        },
        {
            "id": "kernel_map",
            "label": "Kernel map",
            "title": "Feature map",
            "description": "Apply kernel feature map so attention can be accumulated linearly",
        },
        {
            "id": "linear_mix",
            "label": "Linear mix",
            "title": "Linear attention mix",
            "description": "Prefix/state accumulation computes attention in linear time",
        },
        {
            "id": "o_proj",
            "label": "Linear (out)",
            "title": "Output projection",
            "description": f"Linear; {q_out} -> {hidden}",
        },
    ]
