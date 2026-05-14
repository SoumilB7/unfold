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
    if attention.kind in {"mha", "gqa", "mqa"}:
        return _sdpa_detailed_child_blocks(
            attention.kind,
            hidden,
            q_out,
            kv_out,
            num_heads,
            num_kv_heads,
            d_k,
            q_per_group,
        )

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
            "description": (
                f"Linear; {hidden} -> {kv_out}  ({num_kv_heads} KV-heads x {d_k} dims). "
                "Cache ports show K/V write/read during generation: arrowhead for write, blunt tail for read."
            ),
        },
        {
            "id": "v_proj",
            "title": "Value projection",
            "description": (
                f"Linear; {hidden} -> {kv_out}  ({num_kv_heads} KV-heads x {d_k} dims). "
                "Cache ports show K/V write/read during generation: arrowhead for write, blunt tail for read."
            ),
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


def _sdpa_detailed_child_blocks(
    kind: str,
    hidden: str,
    q_out: str,
    kv_out: str,
    num_heads: int,
    num_kv_heads: int,
    d_k: str,
    q_per_group: int | None,
) -> list[dict]:
    kv_label = "1 shared K/V head" if kind == "mqa" else f"{num_kv_heads} KV-heads"
    scaled_title = "Scaled attention scores"
    scaled_desc = "Per head: QK^T / sqrt(dim); dot-product scores scaled for numerical stability"
    if kind == "gqa":
        scaled_title = "Grouped scaled dot-product attention"
        group = f"; each KV head serves {q_per_group} query heads" if q_per_group else ""
        scaled_desc = (
            f"Grouped SDPA scores: {num_heads} query heads attend through "
            f"{num_kv_heads} shared K/V heads{group}; scores use QK^T / sqrt(dim)"
        )
    elif kind == "mqa":
        scaled_title = "Multi-query scaled dot-product attention"
        scaled_desc = (
            f"Multi-Query SDPA scores: {num_heads} query heads share one K/V stream; "
            "scores use QK^T / sqrt(dim)"
        )

    return [
        {
            "id": "q_proj",
            "title": "Query projection",
            "description": f"Linear; {hidden} -> {q_out}  ({num_heads} heads x {d_k} dims)",
        },
        {
            "id": "k_proj",
            "title": "Key projection",
            "description": (
                f"Linear; {hidden} -> {kv_out}  ({kv_label} x {d_k} dims). "
                "Cache ports show K/V write/read during generation: arrowhead for write, blunt tail for read."
            ),
        },
        {
            "id": "v_proj",
            "title": "Value projection",
            "description": (
                f"Linear; {hidden} -> {kv_out}  ({kv_label} x {d_k} dims). "
                "Cache ports show K/V write/read during generation: arrowhead for write, blunt tail for read."
            ),
        },
        {
            "id": "scaled_scores",
            "title": scaled_title,
            "description": scaled_desc,
        },
        {
            "id": "attn_softmax",
            "title": "Softmax weights",
            "description": "Normalize each query row into attention weights over source tokens",
        },
        {
            "id": "attn_apply_v",
            "title": "Apply values",
            "description": "Multiply attention weights by V to produce one context vector per head",
        },
        {
            "id": "concat_heads",
            "title": "Concatenate heads",
            "description": f"Stack all {num_heads} per-head context vectors back into width {q_out}",
        },
        {
            "id": "o_proj",
            "title": "Output projection",
            "description": f"Linear; {q_out} -> {hidden}  (mixes information across heads)",
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
    query_children = [
        {
            "id": "mla_q",
            "label": "Q projection",
            "title": "Query projection",
            "description": (
                f"Projects hidden states into query latent space through LoRA rank {q_rank}"
                if attention.q_lora_rank
                else f"Projects hidden states directly into query heads; {hidden} -> {q_out}"
            ),
        },
        {
            "id": "mla_q_nope",
            "label": "Q noPE",
            "title": "Query content slice",
            "description": "Query content component that does not receive rotary position encoding",
        },
        {
            "id": "mla_q_rope",
            "label": "Q RoPE",
            "title": "Query positional slice",
            "description": f"Query positional component prepared for rotary position encoding; dim {rope}",
        },
        {
            "id": "mla_q_rope_apply",
            "label": "Apply RoPE",
            "title": "Apply RoPE to query",
            "description": "Applies rotary position encoding to the query positional slice",
        },
        {
            "id": "mla_q_concat",
            "label": "Q concat",
            "title": "Final MLA query",
            "description": "Concatenates Q noPE with RoPE-encoded Q RoPE before score computation",
        },
    ]
    kv_children = [
        {
            "id": "mla_kv_down",
            "label": "KV compress",
            "title": "K/V latent compression",
            "description": f"Compresses the token state into the shared latent K/V cache; {hidden} -> rank {kv_rank}",
        },
        {
            "id": "mla_cache",
            "label": "latent cache c_t",
            "title": "Stored latent cache",
            "description": f"Compressed K/V latent stored in the cache instead of full K and V heads; rank {kv_rank}",
        },
        {
            "id": "mla_kv_up",
            "label": "KV expand",
            "title": "K/V head expansion",
            "description": f"Expands cached latent c_t into K noPE content and V values for {num_heads} query heads",
        },
        {
            "id": "mla_k_nope",
            "label": "K noPE",
            "title": "Latent key content",
            "description": "Key content expanded from the compressed K/V latent; concatenated with the RoPE key before scoring",
        },
        {
            "id": "mla_k_rope",
            "label": "K RoPE",
            "title": "Key positional slice",
            "description": f"Key positional component produced alongside the latent cache; dim {rope}",
        },
        {
            "id": "mla_k_rope_apply",
            "label": "Apply RoPE",
            "title": "Apply RoPE to key",
            "description": "Applies rotary position encoding to the key positional slice",
        },
        {
            "id": "mla_k_merge",
            "label": "K concat",
            "title": "Composed MLA key",
            "description": "Concatenates K noPE with the RoPE key side-channel before QK^T score computation",
        },
        {
            "id": "mla_v",
            "label": "V values",
            "title": "Latent value heads",
            "description": "Value heads expanded from the compressed K/V latent; consumed after softmax",
        },
    ]
    return [
        {
            "id": "mla_query_path",
            "label": "Query path",
            "title": "MLA query path",
            "description": (
                "Builds Q by projecting the hidden state, splitting content and positional slices, "
                "applying RoPE to the positional slice, then concatenating them"
            ),
            "detail_view": "mla_query_path",
            "children": query_children,
        },
        {
            "id": "mla_kv_path",
            "label": "KV cache path",
            "title": "MLA K/V cache path",
            "description": (
                f"Compresses hidden state into rank {kv_rank} latent cache, expands K/V content, "
                "and combines K noPE with a RoPE key side-channel"
            ),
            "detail_view": "mla_kv_cache_path",
            "children": kv_children,
        },
        {
            "id": "scaled_scores",
            "label": "Latent scores",
            "title": "Multi-Head Latent scores",
            "description": "Q attends to expanded latent K plus the RoPE key side-channel; scores use QK^T / sqrt(dim)",
        },
        {
            "id": "attn_softmax",
            "label": "Softmax",
            "title": "Softmax weights",
            "description": "Normalize latent attention scores over source positions",
        },
        {
            "id": "attn_apply_v",
            "label": "Apply V",
            "title": "Apply latent values",
            "description": "Multiply softmax weights by V expanded from the compressed K/V latent",
        },
        {
            "id": "concat_heads",
            "label": "Concat heads",
            "title": "Concatenate latent heads",
            "description": f"Stack all {num_heads} context heads back into width {q_out}",
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
