"""Reusable attention-family child block declarations."""
from __future__ import annotations

from ....block_schema import Block

from ....ir import AttentionSpec
from ....labels import CACHE_PORT_FACT
from ..common import format_dim as _fmt


def attention_detail(attention: AttentionSpec) -> dict:
    """Serializable attention facts for block-local detail rendering."""
    return {
        "kind": attention.kind,
        "num_heads": attention.num_heads,
        "num_kv_heads": attention.num_kv_heads,
        "head_dim": attention.head_dim,
        "kv_lora_rank": attention.kv_lora_rank,
        "q_lora_rank": attention.q_lora_rank,
        "rope_dim": attention.rope_dim,
        "qk_nope_head_dim": attention.qk_nope_head_dim,
        "qk_rope_head_dim": attention.qk_rope_head_dim,
        "v_head_dim": attention.v_head_dim,
        "mask": attention.mask,
        "window_size": attention.window_size,
        "kv_source_layer": attention.kv_source_layer,
        "qk_norm": attention.qk_norm,
        "rope": attention.rope,
        "bias": attention.bias,
        "shared": attention.shared,
        "no_rope": attention.no_rope,
        "cross_attention": attention.cross_attention,
        "cross_kv_source": attention.cross_kv_source,
        "compress_ratio": attention.compress_ratio,
        "index_topk": attention.index_topk,
        "index_n_heads": attention.index_n_heads,
        "index_head_dim": attention.index_head_dim,
        "mrope_section": attention.mrope_section,
        "variant": attention.variant,
    }


def attention_child_blocks(attention: AttentionSpec, hidden_size: int, *,
                           generic: bool = False) -> list[Block]:
    """Per-op description cards for an attention block.

    ``generic=True`` produces source-neutral SDPA cards ("over the input" rather
    than "over the hidden state" / "over the encoded text").  Use it when the SAME
    op cards are shared by sibling self- and cross-attention blocks (UNet
    Transformer2D): they share op ids, so a source-specific card on one would be
    wrong on the other — the K/V source is carried by the cross-attention's own
    ``cross_attention_states`` node instead."""
    builders = {
        "mla": _mla_child_blocks,
        "ssm": _ssm_child_blocks,
        "recurrent": _recurrent_child_blocks,
        "rwkv": _rwkv_child_blocks,
        "linear": _linear_attention_child_blocks,
    }
    builder = builders.get(attention.kind, _sdpa_child_blocks)
    if builder is _sdpa_child_blocks:
        return _sdpa_child_blocks(attention, hidden_size, generic=generic)
    return builder(attention, hidden_size)


def _sdpa_child_blocks(attention: AttentionSpec, hidden_size: int, *, generic: bool = False) -> list[Block]:
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
            attention,
            hidden,
            q_out,
            kv_out,
            num_heads,
            num_kv_heads,
            d_k,
            q_per_group,
            generic=generic,
        )

    attention_title, attention_desc = _sdpa_operation_meta(attention, num_heads, num_kv_heads, d_k, q_per_group)
    return [
        {
            "id": "q_proj",
            "title": "Query projection",
            "description": "Linear projection producing the per-head queries.",
            "facts": [f"{hidden} → {q_out}", f"{num_heads} Q heads", f"head dim {d_k}"],
        },
        {
            "id": "k_proj",
            "title": "Key projection",
            "description": "Linear projection producing the keys.",
            "facts": [f"{hidden} → {kv_out}", f"{num_kv_heads} KV heads", CACHE_PORT_FACT],
        },
        {
            "id": "v_proj",
            "title": "Value projection",
            "description": "Linear projection producing the values.",
            "facts": [f"{hidden} → {kv_out}", f"{num_kv_heads} KV heads", CACHE_PORT_FACT],
        },
        {
            "id": "qkv_dot",
            "title": attention_title,
            "description": attention_desc,
        },
        {
            "id": "o_proj",
            "title": "Output projection",
            "description": "Linear recombining every head back into the residual width.",
            "facts": [f"{q_out} → {hidden}"],
        },
    ]


def _sdpa_detailed_child_blocks(
    attention: AttentionSpec,
    hidden: str,
    q_out: str,
    kv_out: str,
    num_heads: int,
    num_kv_heads: int,
    d_k: str,
    q_per_group: int | None,
    *,
    generic: bool = False,
) -> list[Block]:
    kind = attention.kind
    # generic mode: source-neutral wording shared by sibling self/cross blocks.
    cross_attention = attention.cross_attention and not generic
    cross_src = attention.cross_kv_source or "external encoder states"
    is_text_cross = cross_attention and any(
        w in cross_src.lower() for w in ("text", "prompt", "encoder", "caption"))
    kv_chip = "1 shared KV head" if kind == "mqa" else f"{num_kv_heads} KV heads"
    group_fact = [f"{q_per_group} Q per KV head"] if (q_per_group and num_kv_heads > 1) else []
    scaled_title = "Scaled attention scores"
    scaled_desc = "Per head: QK^T / sqrt(dim) — dot-product scores scaled for numerical stability."
    if cross_attention:
        scaled_title = "Cross-attention scores"
        scaled_desc = (
            f"Query heads attend over the {cross_src} (its K/V), not the latent "
            "itself; scores use QK^T / sqrt(dim)."
        )
    elif kind == "gqa":
        scaled_title = "Grouped scaled dot-product attention"
        scaled_desc = (
            "Query heads attend through a smaller set of shared K/V heads; "
            "scores use QK^T / sqrt(dim)."
        )
    elif kind == "mqa":
        scaled_title = "Multi-query scaled dot-product attention"
        scaled_desc = (
            "All query heads share one K/V stream; scores use QK^T / sqrt(dim)."
        )

    if generic:
        # shared self/cross cards: name no specific source, and no causal cache
        # (these ops are also used by full bidirectional image attention).
        q_src = kv_src = "the input"
        cache_facts: list[str] = []
    else:
        q_src = (("the latent query tokens" if is_text_cross else "decoder hidden states")
                 if cross_attention else "the hidden state")
        kv_src = f"the {cross_src}" if cross_attention else "the hidden state"
        cache_facts = [] if cross_attention else [CACHE_PORT_FACT]
    cross_chip = ["from cross-attention source"] if cross_attention else []
    cards = [
        {
            "id": "q_proj",
            "title": "Query projection",
            "description": f"Linear over {q_src} producing the per-head queries.",
            "facts": [f"{hidden} → {q_out}", f"{num_heads} Q heads", f"head dim {d_k}"],
        },
        {
            "id": "k_proj",
            "title": "Key projection",
            "description": f"Linear over {kv_src} producing the keys.",
            "facts": [f"{hidden} → {kv_out}", kv_chip, *cross_chip, *cache_facts],
        },
        {
            "id": "v_proj",
            "title": "Value projection",
            "description": f"Linear over {kv_src} producing the values.",
            "facts": [f"{hidden} → {kv_out}", kv_chip, *cross_chip, *cache_facts],
        },
        {
            "id": "scaled_scores",
            "title": scaled_title,
            "description": scaled_desc,
            "facts": [f"{num_heads} Q heads", kv_chip, *group_fact],
        },
        {
            "id": "attn_softmax",
            "title": "Softmax weights",
            "description": "Normalize each query row into attention weights over source tokens.",
        },
        {
            "id": "attn_apply_v",
            "title": "Apply values",
            "description": "Multiply attention weights by V to produce one context vector per head.",
        },
        {
            "id": "concat_heads",
            "title": "Concatenate heads",
            "description": "Stack the per-head context vectors back into one width.",
            "facts": [f"{num_heads} × {d_k}", f"→ {q_out}"],
        },
        {
            "id": "o_proj",
            "title": "Output projection",
            "description": "Linear mixing information across heads, back to the residual width.",
            "facts": [f"{q_out} → {hidden}"],
        },
    ]
    if attention.rope and not attention.no_rope and not cross_attention:
        # RoPE rotates Q and K before the scores (apply_rotary_pos_emb) — cards for
        # the two rope nodes the SDPA region now draws on the Q/K lanes.
        cards += [
            {"id": "q_rope", "title": "Apply RoPE (Q)",
             "description": "Rotary position embedding applied to the query heads before the scores."},
            {"id": "k_rope", "title": "Apply RoPE (K)",
             "description": "Rotary position embedding applied to the key heads before the scores."},
        ]
    if generic:
        # These cards are SHARED across stages of different width (the panel dedups
        # by id). Per-stage dims would be wrong on the shared card, so drop them —
        # the widths/heads live on the (per-stage) attention block card + diagram.
        for c in cards:
            c.pop("facts", None)
    return cards


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


def _mla_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[Block]:
    hidden = _fmt(hidden_size)
    q_rank = _fmt(attention.q_lora_rank) if attention.q_lora_rank else "direct"
    kv_rank = _fmt(attention.kv_lora_rank)
    num_heads = attention.num_heads or 0
    head_dim = attention.head_dim or 0
    # Per-head slice widths — straight from the config (DeepSeek/Kimi declare
    # them); the noPE width falls back to head_dim minus the RoPE width.
    rope_v = attention.rope_dim or attention.qk_rope_head_dim or 0
    rope = _fmt(rope_v) if rope_v else "?"
    nope_v = attention.qk_nope_head_dim or ((head_dim - rope_v) if (head_dim and rope_v) else None)
    v_v = attention.v_head_dim or nope_v
    nope_fact = [f"{_fmt(nope_v)} per head"] if nope_v else []
    concat_fact = ([f"head dim {_fmt(nope_v + rope_v)} = {_fmt(nope_v)} + {_fmt(rope_v)}"]
                   if (nope_v and rope_v) else [])
    q_out = _fmt(num_heads * head_dim) if (num_heads and head_dim) else hidden
    query_children = [
        {
            "id": "mla_q",
            "label": "Q projection",
            "title": "Query projection",
            "description": (
                "Projects hidden states into query latent space through a low-rank bottleneck."
                if attention.q_lora_rank
                else "Projects hidden states directly into query heads."
            ),
            "facts": ([f"rank {q_rank}"] if attention.q_lora_rank else []) + [f"{hidden} → {q_out}"],
        },
        {
            "id": "mla_q_nope",
            "label": "Q noPE",
            "title": "Query content slice",
            "description": "Query content component that does not receive rotary position encoding.",
            "facts": nope_fact,
        },
        {
            "id": "mla_q_rope",
            "label": "Q RoPE",
            "title": "Query positional slice",
            "description": "Query positional component prepared for rotary position encoding.",
            "facts": [f"{rope} per head"],
        },
        {
            "id": "mla_q_rope_apply",
            "label": "Apply RoPE",
            "title": "Apply RoPE to query",
            "description": "Applies rotary position encoding to the query positional slice.",
            "facts": [f"{rope} per head"] if rope_v else [],
        },
        {
            "id": "mla_q_concat",
            "label": "Q concat",
            "title": "Final MLA query",
            "description": "Concatenates Q noPE with RoPE-encoded Q RoPE before score computation.",
            "facts": concat_fact,
        },
    ]
    kv_children = [
        {
            "id": "mla_kv_down",
            "label": "KV compress",
            "title": "K/V latent compression",
            "description": "Compresses the token state into the shared latent K/V cache.",
            "facts": [f"{hidden} → rank {kv_rank}"],
        },
        {
            "id": "mla_cache",
            "label": "latent cache c_t",
            "title": "Stored latent cache",
            "description": "Compressed K/V latent stored in the cache instead of full K and V heads.",
            "facts": [f"rank {kv_rank}", CACHE_PORT_FACT],
        },
        {
            "id": "mla_kv_up",
            "label": "KV expand",
            "title": "K/V head expansion",
            "description": "Expands the cached latent c_t into K noPE content and V values.",
            "facts": [f"{num_heads} heads"]
                + ([f"{_fmt(nope_v)} + {_fmt(v_v)} per head"] if (nope_v and v_v) else []),
        },
        {
            "id": "mla_k_nope",
            "label": "K noPE",
            "title": "Latent key content",
            "description": "Key content expanded from the compressed K/V latent; concatenated with the RoPE key before scoring.",
            "facts": nope_fact,
        },
        {
            "id": "mla_k_rope",
            "label": "K RoPE",
            "title": "Key positional slice",
            "description": "Key positional component produced alongside the latent cache.",
            "facts": [f"{rope} shared across heads"],
        },
        {
            "id": "mla_k_rope_apply",
            "label": "Apply RoPE",
            "title": "Apply RoPE to key",
            "description": "Applies rotary position encoding to the key positional slice.",
            "facts": [f"{rope} per head"] if rope_v else [],
        },
        {
            "id": "mla_k_merge",
            "label": "K concat",
            "title": "Composed MLA key",
            "description": "Concatenates K noPE with the RoPE key side-channel before QK^T score computation.",
            "facts": concat_fact,
        },
        {
            "id": "mla_v",
            "label": "V values",
            "title": "Latent value heads",
            "description": "Value heads expanded from the compressed K/V latent; consumed after softmax.",
            "facts": ([f"{_fmt(v_v)} per head"] if v_v else []) + [f"{num_heads} heads"],
        },
    ]
    indexer_block = []
    if attention.index_n_heads:
        # DeepSeek-V3.2 DSA: the lightning indexer is a real sub-module (its own
        # heads/dim) — a Tier-1 drill-down, not just a chip.
        indexer_block = [{
            "id": "mla_indexer",
            "label": ["Sparse indexer", "(DSA)"],
            "title": "DeepSeek Sparse Attention indexer",
            "description": (
                "A lightweight scorer with its own small heads that scores every key "
                "against the query and keeps only the top-k per query; the latent "
                "attention then runs over that sparse subset of the context."
            ),
            "facts": [f"{_fmt(attention.index_n_heads)} indexer heads",
                      f"head dim {_fmt(attention.index_head_dim)}",
                      f"top-{_fmt(attention.index_topk)} keys"],
            "view": "dsa_indexer",
            "children": [
                {"id": "dsa_proj", "title": "Indexer projections",
                 "description": f"The indexer's own lightweight query/key projections "
                                f"({_fmt(attention.index_n_heads)} heads × {_fmt(attention.index_head_dim)})."},
                {"id": "dsa_score", "title": "Index scores",
                 "description": "Scores every key against the query with the indexer heads (cheap, separate from the main attention)."},
                {"id": "dsa_topk", "title": f"Keep top-{_fmt(attention.index_topk)}",
                 "description": f"Selects the top-{_fmt(attention.index_topk)} keys per query; the latent attention runs only over those."},
            ],
        }]
    return indexer_block + [
        {
            "id": "mla_query_path",
            "label": "Query path",
            "title": "MLA query path",
            "description": (
                "Builds Q by projecting the hidden state, splitting content and positional slices, "
                "applying RoPE to the positional slice, then concatenating them."
            ),
            "facts": ([f"rank {q_rank}"] if attention.q_lora_rank else []),
            "view": "mla_query_path",
            "children": query_children,
        },
        {
            "id": "mla_kv_path",
            "label": "KV cache path",
            "title": "MLA K/V cache path",
            "description": (
                "Compresses the hidden state into the latent cache, expands K/V content, "
                "and combines K noPE with a RoPE key side-channel."
            ),
            "facts": [f"cache rank {kv_rank}", CACHE_PORT_FACT],
            "view": "mla_kv_cache_path",
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
            "description": "Normalize latent attention scores over source positions.",
        },
        {
            "id": "attn_apply_v",
            "label": "Apply V",
            "title": "Apply latent values",
            "description": "Multiply softmax weights by V expanded from the compressed K/V latent.",
        },
        {
            "id": "concat_heads",
            "label": "Concat heads",
            "title": "Concatenate latent heads",
            "description": "Stack the per-head context vectors back into one width.",
            "facts": [f"{num_heads} heads", f"→ {q_out}"],
        },
        {
            "id": "o_proj",
            "label": "Linear (out)",
            "title": "Output projection",
            "description": "Linear back to the residual width.",
            "facts": [f"{q_out} → {hidden}"],
        },
    ]


def _ssm_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[Block]:
    hidden = _fmt(hidden_size)
    state = _fmt(attention.head_dim)
    return [
        {
            "id": "ssm_in_proj",
            "label": "Input projection",
            "title": "SSM input projection",
            "description": "Project hidden activations into SSM channels.",
            "facts": [f"hidden {hidden}"],
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
            "description": "Token recurrence over the sequence.",
            "facts": [f"state dim {state}"],
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
            "description": "Project SSM channels back to the residual width.",
            "facts": [f"→ {hidden}"],
        },
    ]


def _recurrent_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[Block]:
    hidden = _fmt(hidden_size)
    width = _fmt(attention.head_dim)
    return [
        {
            "id": "lru_in_proj",
            "label": "Input projection",
            "title": "LRU input projection",
            "description": "Linear into the recurrent width.",
            "facts": [f"{hidden} → {width}"],
        },
        {
            "id": "lru_state",
            "label": "Recurrent state",
            "title": "Linear recurrent state",
            "description": "State update over sequence positions.",
            "facts": [f"width {width}"],
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
            "description": "Linear back to the residual width.",
            "facts": [f"{width} → {hidden}"],
        },
    ]


def _rwkv_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[Block]:
    hidden = _fmt(hidden_size)
    heads = attention.num_heads or 0
    return [
        {
            "id": "rwkv_receptance",
            "label": "Receptance",
            "title": "Receptance gate",
            "description": "Token-wise gate over the hidden state.",
            "facts": [f"dim {hidden}"],
        },
        {
            "id": "rwkv_key",
            "label": "Key",
            "title": "RWKV key projection",
            "description": "Key-like channel mixing over the recurrent heads.",
            "facts": [f"{heads} heads"],
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
            "description": "Project mixed channels back to the residual width.",
            "facts": [f"→ {hidden}"],
        },
    ]


def _linear_attention_child_blocks(attention: AttentionSpec, hidden_size: int) -> list[Block]:
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
            "description": "Linear projection producing the queries.",
            "facts": [f"{hidden} → {q_out}"],
        },
        {
            "id": "k_proj",
            "label": "Linear (K)",
            "title": "Key projection",
            "description": "Linear projection producing the keys.",
            "facts": [f"{hidden} → {kv_out}"],
        },
        {
            "id": "v_proj",
            "label": "Linear (V)",
            "title": "Value projection",
            "description": "Linear projection producing the values.",
            "facts": [f"{hidden} → {kv_out}"],
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
            "description": "Linear back to the residual width.",
            "facts": [f"{q_out} → {hidden}"],
        },
    ]
