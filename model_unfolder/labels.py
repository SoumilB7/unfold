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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ir import AttentionSpec

_MASK_SHORT = {
    "sliding": "SWA",
    "global": "full",
    "full": "full",
    "causal": "causal",
    "bidirectional": "bidirectional",
    "chunked": "chunked",
    "compressed_sparse": "CSA",
    "heavily_compressed": "HCA",
}
_MASK_LONG = {
    "sliding": "Sliding-window",
    "global": "Full / global",
    "full": "Full (bidirectional)",
    "causal": "Causal",
    "bidirectional": "Bidirectional",
    "chunked": "Chunked",
    "compressed_sparse": "Compressed sparse",
    "heavily_compressed": "Hierarchical compressed",
}
_MASK_TITLE = {
    "sliding": "Sliding-window attention",
    "global": "Full-context attention",
    "full": "Full bidirectional attention (no causal mask)",
    "causal": "Causal attention",
    "bidirectional": "Bidirectional attention (no causal mask — every token "
                     "attends to every token)",
    "chunked": "Chunked attention",
    "compressed_sparse": "Compressed sparse attention",
    "heavily_compressed": "Hierarchical compressed attention",
}

_KIND_SHORT = {
    "mla": "MLA",
    "gqa": "GQA",
    "mqa": "MQA",
    "mha": "MHA",
    "ssm": "SSM",
    "recurrent": "LRU",
    "linear": "Lin-Attn",
    "gated_delta": "Gated-Delta",
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
    "gated_delta": "Gated DeltaNet recurrent mixer",
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
    "glu": "GLU",
}


def mask_short(attention: dict) -> str:
    """Compact mask tag — ``"SWA"`` / ``"full"`` / ``"causal"``.

    U2: an unresolved mask (``None``/``"unknown"``) says so instead of
    re-asserting the causal default the parser just refused to fabricate."""
    mask = attention.get("mask")
    if mask not in _MASK_SHORT:
        return "unresolved"
    return _MASK_SHORT[mask]


def mask_long(attention: dict) -> str:
    """Human-readable mask label — ``"Sliding-window"`` / ``"Full / global"``."""
    mask = attention.get("mask")
    if mask not in _MASK_LONG:
        return "Mask unresolved"
    return _MASK_LONG[mask]


def mask_title(attention: dict) -> str:
    """Tooltip-style mask description."""
    mask = attention.get("mask")
    if mask not in _MASK_TITLE:
        return ("Attention mask unresolved — the config does not declare "
                "whether this stack is a causal decoder, and the mask is not "
                "read from code yet; nothing is asserted")
    return _MASK_TITLE[mask]


def mask_chip(attention: dict) -> str:
    """One-line chip; for sliding includes the window size: ``"SWA 1,024"``."""
    label = mask_short(attention)
    window = attention.get("window_size")
    ratio = attention.get("compress_ratio")
    if ratio and attention.get("mask") in {"compressed_sparse", "heavily_compressed"}:
        return f"{label} r{_fmt_int(ratio)}"
    if window and is_sliding(attention):
        return f"{label} {_fmt_int(window)}"
    return label


def kind_short(attention: dict) -> str:
    kind = attention.get("kind")
    variant = attention.get("variant")
    if kind in (None, "", "unknown"):
        if variant and variant.get("short"):
            base = (f"{variant['short']} · {variant['tag']}"
                    if variant.get("tag") else variant["short"])
            return f"{base} · unresolved"
        return ("XAttn unresolved" if attention.get("cross_attention")
                else "Attn unresolved")
    if variant and variant.get("short"):
        # Variant is self-describing (e.g. "Joint Attn · MM-DiT"); it already
        # encodes everything, so don't also append the auto QK/bias/NoPE tags.
        return f"{variant['short']} · {variant['tag']}" if variant.get("tag") else variant["short"]
    short = _KIND_SHORT.get(kind)
    if short is None:
        return ("XAttn unresolved" if attention.get("cross_attention")
                else "Attn unresolved")
    if attention.get("cross_attention"):
        short = f"{short} XAttn"
    tags = []
    if attention.get("qk_norm"):
        tags.append("QK-Norm")
    if attention.get("bias"):
        tags.append("+bias")
    if attention.get("shared"):
        tags.append("Shared")
    if attention.get("no_rope"):
        tags.append("NoPE")
    if _partial_rope_dims(attention):
        tags.append("Partial RoPE")
    return f"{short} ({', '.join(tags)})" if tags else short


def _partial_rope_dims(attention: dict) -> tuple[int, int] | None:
    """(rotated, head_dim) when the code rotates only a slice of each head
    (GPT-NeoX/GPT-J/StableLM/Persimmon partial rotary).  MLA is excluded —
    its rope/nope split is a different mechanism with its own drawing."""
    rd, hd = attention.get("rope_dim"), attention.get("head_dim")
    if (attention.get("kind") != "mla" and isinstance(rd, int)
            and isinstance(hd, int) and 0 < rd < hd):
        return rd, hd
    return None


def kind_long(attention: dict) -> str:
    kind = attention.get("kind")
    variant = attention.get("variant")
    if kind in (None, "", "unknown"):
        if variant and (variant.get("title") or variant.get("short")):
            base = variant.get("title") or variant["short"]
            return f"{base} — attention mechanism unresolved"
        return ("Cross-attention mechanism unresolved"
                if attention.get("cross_attention")
                else "Attention mechanism unresolved")
    if variant and (variant.get("title") or variant.get("short")):
        return variant.get("title") or variant["short"]
    base = _KIND_LONG.get(kind)
    if base is None:
        return ("Cross-attention mechanism unresolved"
                if attention.get("cross_attention")
                else "Attention mechanism unresolved")
    if attention.get("cross_attention"):
        base = {
            "gqa": "Grouped-query cross-attention",
            "mqa": "Multi-query cross-attention",
            "mha": "Multi-head cross-attention",
        }.get(attention.get("kind", ""), "Cross-attention")
    extras = []
    if attention.get("qk_norm"):
        extras.append("per-head Q/K normalisation")
    if attention.get("bias"):
        extras.append("bias on Q/K/V/O projections")
    if attention.get("shared"):
        extras.append("weight-shared across positions")
    _pr = _partial_rope_dims(attention)
    if _pr:
        extras.append(f"rotary on {_pr[0]} of {_pr[1]} head dims")
    if attention.get("no_rope"):
        extras.append("no positional encoding (NoPE)")
    return f"{base}; {'; '.join(extras)}" if extras else base


def attention_tower_label(attention: dict) -> str | list[str]:
    """Compact repeated-cell label without strengthening an unknown kind."""
    kind = attention.get("kind")
    if kind in (None, "", "unknown"):
        return [
            "Cross-attention" if attention.get("cross_attention") else "Self-attention",
            "mechanism unresolved",
        ]
    return kind_short(attention)

def moe_router_detail(ffn: dict) -> str:
    """Longer router tooltip describing the gating and selection behaviour."""
    r = ffn.get("routing") or {}
    bits = []
    if r.get("scoring_func"):
        bits.append(f"{r['scoring_func']} gating")
    if r.get("topk_method"):
        bits.append(f"{r['topk_method']} selection")
    if (r.get("n_group") or 0) > 1 and r.get("topk_group"):
        bits.append(f"group-limited: top-{r['topk_group']} of {r['n_group']} groups")
    if r.get("norm_topk_prob"):
        bits.append("normalized top-k weights")
    if r.get("routed_scaling_factor"):
        bits.append(f"routed output ×{r['routed_scaling_factor']}")
    return "; ".join(bits)


def is_sliding(attention: dict) -> bool:
    return attention.get("mask") == "sliding"

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


#: One chip, used verbatim by every card whose block draws cache ports.
CACHE_PORT_FACT = "cache ports: ⌃ write · ⊥ read"


def describe_attention(attention: dict) -> str:
    """Multi-clause human description suitable for tooltips and cards."""
    variant = attention.get("variant")
    if variant and variant.get("desc"):
        return variant["desc"]
    kind = attention.get("kind")
    if attention.get("cross_attention"):
        kv_heads = attention.get("num_kv_heads") or attention.get("num_heads")
        text = (
            "Cross-attention; decoder hidden states produce Q; "
            "cross_attention_states produce K/V; "
            f"{attention.get('num_heads')} Q / {kv_heads} KV heads; "
            f"head dim {_fmt_int(attention.get('head_dim'))}"
        )
    elif kind == "mla":
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
    elif kind == "gated_delta":
        text = (
            "Gated delta-rule recurrence; causal depthwise convolution; "
            f"{attention.get('num_kv_heads')} K / {attention.get('num_heads')} V heads"
        )
    else:
        text = (
            f"Multi-head; {attention.get('num_heads')} heads; "
            f"head dim {_fmt_int(attention.get('head_dim'))}; cache ports mark K/V write/read state"
        )
    if is_sliding(attention) and attention.get("window_size"):
        text += f"; sliding window {_fmt_int(attention.get('window_size'))}"
    if attention.get("mask") == "compressed_sparse":
        text += f"; CSA compress ratio {_fmt_int(attention.get('compress_ratio'))}"
        if attention.get("index_topk"):
            text += f"; index top-k {_fmt_int(attention.get('index_topk'))}"
    if attention.get("mask") == "heavily_compressed":
        text += f"; HCA compress ratio {_fmt_int(attention.get('compress_ratio'))}"
    if attention.get("index_n_heads"):
        # DeepSeek-V3.2 DSA: a separate lightweight indexer scores all keys and
        # keeps the top-k per query for the (MLA) attention to attend over.
        text += (
            f"; DeepSeek Sparse Attention — a lightweight indexer "
            f"({_fmt_int(attention.get('index_n_heads'))} heads × "
            f"{_fmt_int(attention.get('index_head_dim'))}) keeps the top-"
            f"{_fmt_int(attention.get('index_topk'))} keys per query"
        )
    # Surface structural flags as annotations
    extras = []
    if attention.get("qk_norm"):
        extras.append("QK-Norm")
    if attention.get("bias"):
        extras.append("+bias")
    if attention.get("shared"):
        extras.append("weight-shared")
    if attention.get("no_rope"):
        extras.append("NoPE")
    if extras:
        text += "; " + ", ".join(extras)
    return text


# ---------------------------------------------------------------------------
# Card summaries — explanation prose + fact chips, kept separate.
# The diagram shows structure; the numbers live here as data.
# ---------------------------------------------------------------------------


def attention_summary(attention: dict) -> tuple[str, list[str]]:
    """(explanation sentence, fact chips) for an attention/token-mixing block."""
    kind = attention.get("kind")
    facts: list[str] = []
    variant = attention.get("variant")
    if kind in (None, "", "unknown"):
        role = "Cross-attention" if attention.get("cross_attention") else "Attention"
        desc = (
            f"{role} exists, but its internal mechanism is unresolved from "
            "the available source evidence."
        )
        _head_facts(attention, facts)
    elif variant and variant.get("desc"):
        desc = variant["desc"]
        _head_facts(attention, facts)
    elif attention.get("cross_attention"):
        desc = ("Cross-attention — decoder hidden states produce the queries; "
                "the projected image states produce K and V.")
        kv = attention.get("num_kv_heads") or attention.get("num_heads")
        facts += [f"{attention.get('num_heads')} Q heads", f"{kv} KV heads"]
        _dim_fact(attention, facts)
    elif kind == "mla":
        desc = ("Multi-head latent attention — K/V are stored as one compressed "
                "latent and expanded per head at read time.")
        facts.append(f"{attention.get('num_heads')} heads")
        if attention.get("kv_lora_rank"):
            facts.append(f"KV rank {_fmt_int(attention.get('kv_lora_rank'))}")
        if attention.get("q_lora_rank"):
            facts.append(f"Q rank {_fmt_int(attention.get('q_lora_rank'))}")
    elif kind == "mqa":
        desc = "Multi-query attention — every query head reads one shared K/V head."
        facts += [f"{attention.get('num_heads')} Q heads", "1 KV head"]
        _dim_fact(attention, facts)
    elif kind == "gqa":
        desc = "Grouped-query attention — query heads share a smaller set of K/V heads."
        facts += [f"{attention.get('num_heads')} Q heads",
                  f"{attention.get('num_kv_heads')} KV heads"]
        _dim_fact(attention, facts)
    elif kind == "ssm":
        desc = "Selective state-space mixer (Mamba) — a recurrence replaces attention."
        if attention.get("head_dim"):
            facts.append(f"state dim {_fmt_int(attention.get('head_dim'))}")
    elif kind == "recurrent":
        desc = "Linear recurrent unit — a gated recurrence replaces attention."
        if attention.get("head_dim"):
            facts.append(f"width {_fmt_int(attention.get('head_dim'))}")
    elif kind == "rwkv":
        desc = "RWKV token-mixing — a time-decay recurrence replaces attention."
        if attention.get("num_heads"):
            facts.append(f"{attention.get('num_heads')} heads")
    elif kind == "linear":
        desc = "Linear attention — kernelized scores accumulate in linear time."
        _head_facts(attention, facts)
    elif kind == "gated_delta":
        desc = ("Gated DeltaNet — causal depthwise convolution feeds a gated "
                "delta-rule recurrence with cached recurrent state.")
        if attention.get("conv_kernel_size"):
            facts.append(f"conv kernel {attention.get('conv_kernel_size')}")
        _head_facts(attention, facts)
    else:
        desc = "Multi-head self-attention — every head attends over the sequence."
        facts.append(f"{attention.get('num_heads')} heads")
        _dim_fact(attention, facts)

    if attention.get("index_n_heads"):
        # DeepSeek-V3.2 DSA: a separate lightweight indexer scores all keys and
        # keeps only the top-k per query, so the (MLA) attention runs sparsely.
        desc += (" DeepSeek Sparse Attention adds a lightweight indexer that scores "
                 "all keys and keeps only the top-k per query, so this attention runs "
                 "over a sparse subset of the context.")

    if attention.get("mrope_section"):
        desc += (" Multimodal RoPE splits the rotary dimensions across temporal, "
                 "height and width position axes (for interleaved text + image/video tokens).")

    if is_sliding(attention) and attention.get("window_size"):
        facts.append(f"window {_fmt_int(attention.get('window_size'))}")
    if attention.get("mask") == "compressed_sparse":
        facts.append(f"CSA ratio {_fmt_int(attention.get('compress_ratio'))}")
        if attention.get("index_topk"):
            facts.append(f"index top-{_fmt_int(attention.get('index_topk'))}")
    if attention.get("mask") == "heavily_compressed":
        facts.append(f"HCA ratio {_fmt_int(attention.get('compress_ratio'))}")
    if attention.get("index_n_heads"):
        facts.append(f"DSA top-{_fmt_int(attention.get('index_topk'))} keys")
        facts.append(f"indexer {_fmt_int(attention.get('index_n_heads'))}×{_fmt_int(attention.get('index_head_dim'))}")
    if attention.get("mrope_section"):
        facts.append("M-RoPE " + "/".join(str(s) for s in attention["mrope_section"]))
    if attention.get("rope_3d") and not attention.get("no_rope"):
        # Video DiTs: the temporal axis lives in the positional encoding. Surface it
        # as a chip so the block reads as VIDEO (a 3rd, time, dimension) without
        # drilling into the attention's "apply RoPE" leaves.
        facts.append("3D RoPE · T·H·W")
    if attention.get("output_gate"):
        facts.append(f"{attention['output_gate']} output gate")
    position_kind = attention.get("position_kind")
    if position_kind == "alibi":
        facts.append("ALiBi")
    # Learned/fixed absolute positions are model-input operations.  Their
    # embedding + add blocks live before the repeated decoder stack; repeating
    # the fact on an attention card assigns it to the wrong computation stage.
    elif position_kind == "none" and not attention.get("no_rope"):
        facts.append("no positional transform")
    elif position_kind == "unknown":
        facts.append("position scheme unresolved")
    # U2 P3a: the config-declared RoPE fallback states its TIER — the θ chip
    # replaces the honest-unknown banner, never a silent assertion.
    if attention.get("position_declared"):
        theta = attention.get("rope_theta_declared")
        facts.append(f"RoPE θ={_fmt_int(theta)} (config-declared)" if theta
                     else "RoPE (config-declared)")
    # U2 honest-unknown chips (the position-chip discipline, generalized):
    # an unresolved mask/bias states itself instead of silently rendering
    # like the evidence-backed causal/bias-less case.
    if attention.get("mask") in (None, "unknown"):
        facts.append("mask unresolved")
    for flag, chip in (("qk_norm", "QK-Norm"), ("bias", "+bias"),
                       ("shared", "weight-shared"), ("no_rope", "NoPE")):
        if attention.get(flag):
            facts.append(chip)
    if attention.get("bias") is None and "bias" in attention:
        facts.append("bias unresolved")
    elif attention.get("bias") is False:
        facts.append("bias-free projections")
    if kind in {"mha", "gqa", "mqa"}:
        if attention.get("qk_norm") is None:
            facts.append("QK norm unresolved")
        elif attention.get("qk_norm") is False:
            facts.append("no QK norm")
        if attention.get("cached") is None:
            facts.append("cache unresolved")
        elif attention.get("cached") is False:
            facts.append("no KV cache")
        if attention.get("projection_mode") is None:
            facts.append("QKV storage unresolved")
        if (attention.get("scores_scaled") is None
                and attention.get("scores_scale") is None):
            facts.append("score scaling unresolved")
        if attention.get("position_kind") in {None, "unknown"} \
                or attention.get("position_application") in {None, "unknown"}:
            facts.append("position application unresolved")
        elif (attention.get("position_kind"),
              attention.get("position_application")) == ("none", "none"):
            facts.append("no attention-stage position op")
    _pr = _partial_rope_dims(attention)
    if _pr:
        facts.append(f"partial rotary — {_pr[0]} of {_pr[1]} head dims")
    return desc, facts


def _head_facts(attention: dict, facts: list[str]) -> None:
    q, kv = attention.get("num_heads"), attention.get("num_kv_heads")
    if q and (not kv or kv == q):
        facts.append(f"{q} heads")
    elif q:
        facts.append(f"{q} Q heads")
        facts.append(f"{kv} KV heads")
    _dim_fact(attention, facts)


def _dim_fact(attention: dict, facts: list[str]) -> None:
    if attention.get("head_dim"):
        facts.append(f"head dim {_fmt_int(attention.get('head_dim'))}")


def ffn_summary(ffn: dict) -> tuple[str, list[str]]:
    """(explanation sentence, fact chips) for an FFN / MoE block."""
    from .opgraph import ffn_structure_state

    state = ffn_structure_state(ffn)
    if state == "moe":
        desc = ("Mixture of experts — the router sends each token through a few "
                "expert FFNs instead of one dense MLP.")
        facts = []
        if ffn.get("num_experts") is not None:
            facts.append(f"{_fmt_int(ffn.get('num_experts'))} experts")
        if ffn.get("num_experts_per_tok"):
            chip = f"top-{ffn.get('num_experts_per_tok')}"
            if ffn.get("num_shared_experts"):
                chip += f" + {ffn.get('num_shared_experts')} shared"
            facts.append(chip)
        if ffn.get("num_experts") and ffn.get("num_experts_per_tok"):
            facts.append(f"{100 * ffn['num_experts_per_tok'] / ffn['num_experts']:.1f}% active")
        if ffn.get("expert_intermediate_size") is not None:
            facts.append(
                f"expert hidden {_fmt_int(ffn.get('expert_intermediate_size'))}")
        expert_storage = ffn.get("expert_projection_mode")
        if expert_storage is None:
            facts.append("expert storage unresolved")
        elif expert_storage == "fused_gate_up":
            facts.append("fused expert gate+up")
        elif expert_storage == "split":
            facts.append("split expert gate/up")
        elif expert_storage == "dense":
            facts.append("plain expert MLP")
        if ffn.get("num_shared_experts") and ffn.get("activation") is not None:
            facts.append(
                "shared FFN activation "
                + activation_label(ffn.get("activation")))
        if ffn.get("activation_clip"):
            facts.append(f"clamped ±{ffn['activation_clip']:g}")
        if ffn.get("bias"):
            facts.append("learned bias")
        return desc, facts
    if state == "mechanism_unresolved":
        desc = (
            "Feed-forward mechanism unresolved — known geometry is retained, "
            "but no dense, gated, projection-storage, or activation path is "
            "invented."
        )
        facts = ["mechanism unresolved"]
        if ffn.get("intermediate_size") is not None:
            facts.append(f"hidden {_fmt_int(ffn.get('intermediate_size'))}")
        if ffn.get("activation") is not None:
            facts.append(activation_label(ffn.get("activation")))
        return desc, facts
    if state == "unsupported":
        name = str(ffn.get("class_name") or ffn.get("kind") or "custom FFN")
        return (
            f"Unsupported feed-forward mechanism ({name}) — retained as one "
            "opaque block rather than mapped to a familiar MLP.",
            ["unsupported mechanism"],
        )
    if state == "gating_unresolved":
        desc = (
            "Feed-forward — its gate topology is unresolved from source "
            "evidence, so no dense or gated inner graph is drawn."
        )
        facts = ["gating unresolved"]
        if ffn.get("intermediate_size") is not None:
            facts.append(f"hidden {_fmt_int(ffn.get('intermediate_size'))}")
        if ffn.get("activation") is not None:
            facts.append(activation_label(ffn.get("activation")))
        return desc, facts
    if state == "storage_unresolved":
        gated = ffn.get("gated")
        shape = "gated" if gated is True else "plain" if gated is False else "feed-forward"
        desc = (
            f"The {shape} FFN mechanism is known, but its projection storage is "
            "unresolved; no split, fused, or dense module layout is invented."
        )
        facts = ["projection storage unresolved"]
        if ffn.get("intermediate_size") is not None:
            facts.append(f"hidden {_fmt_int(ffn.get('intermediate_size'))}")
        if ffn.get("activation") is not None:
            facts.append(activation_label(ffn.get("activation")))
        return desc, facts
    if state == "conv_glu":
        desc = (
            "Convolutional gated linear unit — pointwise expansion, local "
            "depthwise mixing, a gated activation, then pointwise projection."
        )
    elif state == "gated":
        desc = ("Gated MLP — a gate path modulates the up projection before "
                "projecting back down.")
    else:
        desc = "Two-layer MLP — expand, apply the non-linearity, project back."
    if ffn.get("activation_from_class"):
        # Activation provenance is independent of gate/storage topology.
        desc += (" The activation is fixed in the model class, not the config "
                 "(surfaced as a code-derived fact).")
    facts = []
    if ffn.get("activation") is not None:
        facts.append(activation_label(ffn.get("activation")))
    else:
        facts.append("activation unresolved")
    if ffn.get("intermediate_size") is not None:
        facts.append(f"hidden {_fmt_int(ffn.get('intermediate_size'))}")
    storage = ffn.get("projection_mode")
    if storage is None:
        facts.append("projection storage unresolved")
    elif storage == "fused_gate_up":
        facts.append("fused gate+up")
    elif storage == "split":
        facts.append("split gate/up")
    if ffn.get("activation_clip"):
        facts.append(f"clamped ±{ffn['activation_clip']:g}")
    return desc, facts


def ffn_label(ffn: dict) -> str | list[str]:
    """Compact block label derived from the one canonical FFN state."""
    from .opgraph import ffn_structure_state

    state = ffn_structure_state(ffn)
    if state == "moe":
        return ["Mixture of Experts", "(MoE)"]
    if state == "conv_glu":
        return ["Conv-GLU", "Feed-forward"]
    if state == "mechanism_unresolved":
        return ["Feed-forward", "mechanism unresolved"]
    if state == "unsupported":
        return ["Feed-forward", "unsupported mechanism"]
    if state == "gating_unresolved":
        return ["Feed-forward", "gating unresolved"]
    if state == "storage_unresolved":
        return [
            "Gated FFN" if ffn.get("gated") is True else "Feed-forward",
            "storage unresolved",
        ]
    return "Gated FFN" if state == "gated" else "Feed-forward (FFN)"


def ffn_title(ffn: dict) -> str:
    """Card title derived from the one canonical FFN state."""
    from .opgraph import ffn_structure_state

    state = ffn_structure_state(ffn)
    return {
        "moe": "Mixture of experts",
        "conv_glu": "Convolutional gated feed-forward",
        "mechanism_unresolved": "Feed-forward mechanism unresolved",
        "unsupported": "Unsupported feed-forward mechanism",
        "gating_unresolved": "Feed-forward gating unresolved",
        "storage_unresolved": "Feed-forward projection storage unresolved",
        "gated": "Gated feed-forward",
        "dense": "Feed-forward",
    }[state]


def ffn_short(ffn: dict) -> str:
    """One-line group/badge vocabulary from the canonical FFN state."""
    from .opgraph import ffn_structure_state

    return {
        "moe": "MoE",
        "conv_glu": "Conv-GLU",
        "mechanism_unresolved": "FFN unresolved",
        "unsupported": "FFN unsupported",
        "gating_unresolved": "Gating unresolved",
        "storage_unresolved": "Storage unresolved",
        "gated": "Gated FFN",
        "dense": "Dense FFN",
    }[ffn_structure_state(ffn)]


def router_facts(ffn: dict) -> list[str]:
    """Fact chips for an MoE router (selection knobs the config declares)."""
    facts = []
    if ffn.get("num_experts"):
        facts.append(f"{_fmt_int(ffn.get('num_experts'))} experts")
    if ffn.get("num_experts_per_tok"):
        facts.append(f"top-{ffn.get('num_experts_per_tok')}")
    routing = ffn.get("routing") or {}
    if routing.get("scoring_func"):
        facts.append(str(routing["scoring_func"]))
    if (routing.get("n_group") or 0) > 1 and routing.get("topk_group"):
        facts.append(f"keep {routing['topk_group']}/{routing['n_group']} groups")
    if routing.get("norm_topk_prob"):
        facts.append("renormalized")
    if routing.get("routed_scaling_factor"):
        facts.append(f"scale {routing['routed_scaling_factor']}")
    return facts


def _fmt_int(value) -> str:
    if value is None:
        return "?"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)

# ---------------------------------------------------------------------------
# Spec-typed block vocabulary (operates on AttentionSpec attributes — the
# adapter side authors labels/titles straight from the typed spec).
# ---------------------------------------------------------------------------


def attention_label(attention: AttentionSpec) -> list[str]:
    kind = attention.kind
    prefix = _attention_mask_prefix(attention)
    if kind in (None, "", "unknown"):
        if attention.variant and attention.variant.get("label"):
            label = list(attention.variant["label"])
            if label:
                # Hero blocks have a fixed two-line compact label.  Keep the
                # variant's primary name here and put its full topology plus
                # the mechanism explanation in the title/inspect card; adding
                # a third line overflows the block.
                return [label[0], "(unresolved)"]
        if attention.cross_attention:
            # Hero blocks have a constrained two-line width.  The title,
            # description, fact chip and drill carry the full "mechanism
            # unresolved" wording; keep the visible box truthful and legible.
            return _prefixed_label(prefix, "Cross-Attention", "(unresolved)")
        return _prefixed_label(prefix, "Attention", "(mechanism unresolved)")
    if attention.variant and attention.variant.get("label"):
        return list(attention.variant["label"])
    if attention.cross_attention:
        # Name the actual side source (from the spec's own cross_kv_source):
        # a seq2seq composite's K/V are encoded PROMPT states, not vision.
        side = ("Prompt" if "prompt states" in str(attention.cross_kv_source or "")
                else "Vision")
        return _prefixed_label(prefix, side, "Cross-Attention")
    if kind == "mla":
        return _prefixed_label(prefix, "Multi-Head Latent", "Attention")
    if kind == "mqa":
        return _prefixed_label(prefix, "Multi-Query", "Attention")
    if kind == "gqa":
        gqa_tags = []
        if attention.qk_norm:
            gqa_tags.append("QK-Norm")
        if _spec_partial_rope(attention):
            gqa_tags.append("Partial RoPE")
        tag = f"({', '.join(gqa_tags)})" if gqa_tags else "Attention"
        return _prefixed_label(prefix, "Grouped-Query", tag)
    if kind == "ssm":
        shared_tag = "(Shared)" if attention.shared else "Block"
        return ["Selective SSM", shared_tag]
    if kind == "recurrent":
        return ["Linear Recurrent", "Unit (LRU)"]
    if kind == "rwkv":
        return ["RWKV", "Token-Mixing"]
    if kind == "linear":
        return ["Linear", "Attention"]
    if kind == "gated_delta":
        return ["Gated DeltaNet", "Token Mixer"]

    tags = []
    if attention.qk_norm:
        tags.append("QK-Norm")
    if attention.no_rope:
        tags.append("NoPE")
    if _spec_partial_rope(attention):
        tags.append("Partial RoPE")
    if tags:
        return ["Multi-Head Attn", f"({', '.join(tags)})"]
    return ["Multi-Head", "Attention"]


def _spec_partial_rope(attention: AttentionSpec) -> bool:
    """True when the spec proves a partial head rotation (rope_dim strictly
    inside head_dim; MLA's rope/nope split is its own mechanism)."""
    return (attention.kind != "mla" and isinstance(attention.rope_dim, int)
            and isinstance(attention.head_dim, int)
            and 0 < attention.rope_dim < attention.head_dim)


def attention_title(attention: AttentionSpec) -> str:
    if attention.kind in (None, "", "unknown"):
        if attention.variant and (
                attention.variant.get("title") or attention.variant.get("short")):
            base = attention.variant.get("title") or attention.variant["short"]
            return f"{base} — attention mechanism unresolved"
        base = ("Cross-attention mechanism unresolved"
                if attention.cross_attention
                else "Attention mechanism unresolved")
        return _prefixed_title(_attention_mask_title_prefix(attention), base)
    if attention.variant and attention.variant.get("title"):
        return attention.variant["title"]
    if attention.cross_attention:
        base = {
            "gqa": "Grouped-query cross-attention",
            "mqa": "Multi-query cross-attention",
            "mha": "Multi-head cross-attention",
        }.get(attention.kind, "Cross-attention")
        return _prefixed_title(_attention_mask_title_prefix(attention), base)
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
            "gated_delta": "Gated DeltaNet token mixer",
        }.get(attention.kind, "Attention")
    base = _prefixed_title(_attention_mask_title_prefix(attention), base)
    extras = []
    if attention.qk_norm:
        extras.append("QK-Norm")
    if attention.shared:
        extras.append("weight-shared")
    if attention.no_rope:
        extras.append("NoPE")
    return f"{base} ({', '.join(extras)})" if extras else base


def _attention_mask_prefix(attention: AttentionSpec) -> str:
    return "SW" if attention.mask == "sliding" else ""


def _attention_mask_title_prefix(attention: AttentionSpec) -> str:
    return "Sliding-window" if attention.mask == "sliding" else ""


def _prefixed_label(prefix: str, first: str, second: str) -> list[str]:
    return [f"{prefix} · {first}", second] if prefix else [first, second]


def _prefixed_title(prefix: str, title: str) -> str:
    return f"{prefix} {title}" if prefix else title


# ---------------------------------------------------------------------------
# Op-card vocabulary — cards as the THIRD projection of a canonical region.
# The SVG and the JSON already project from the op graph; this derives the
# inspect card for any op, so no view ever hand-writes per-node descriptions
# again.  Authors may still override per card (their dicts win by id).
# ---------------------------------------------------------------------------

_OP_TITLES = {
    "linear": "Linear",
    "activation": "Activation",
    "elementwise": "Element-wise op",
    "norm": "Normalization",
    "route": "Router",
    "attention_core": "Attention core",
    "conv": "Convolution",
    "concat": "Concatenate",
    "reshape": "Reshape",
    "slice": "Split",
    "rope": "Rotary embedding",
    "cache": "Cache",
    "subgraph": "Sub-block",
    "opaque": "Custom block",
    "input": "Input",
}

_OP_SENTENCES = {
    "linear": "Linear projection (a learned weight matrix applied to every position).",
    "activation": "Element-wise non-linearity.",
    "elementwise": "Combines its input lanes element by element.",
    "norm": "Keeps activation scales stable for the next op.",
    "route": "Scores the options per token and keeps the top-k.",
    "attention_core": "The token-mixing kernel — positions exchange information here.",
    "conv": "Convolution over the spatial/temporal grid.",
    "concat": "Joins tensor lanes along the feature dimension.",
    "reshape": "Regroups one tensor into a new shape (no learned weights).",
    "slice": "Splits the tensor into named lanes.",
    "rope": "Applies rotary position encoding to this lane.",
    "position": "Constructs or applies positional information at this stage.",
    "cache": "Stored tensor reused across steps (write on entry, read on reuse).",
    "subgraph": "Compound block with its own internal structure.",
    "opaque": "Internals not declared by the config — drawn as one honest block.",
    "input": "External input feeding this block.",
}

_ELEMENTWISE_TITLES = {"mul": "Gate product", "add": "Add", "sum": "Weighted sum"}


def op_card(op) -> dict:
    """Inspect-card dict derived from one canonical op.

    Duck-typed over :class:`model_unfolder.opgraph.Op` (``id``/``kind``/
    ``label``/``in_features``/``out_features``/``fn``/``meta``).  ``meta`` may
    carry ``desc`` to override the kind sentence.
    """
    kind = op.kind
    raw_title = op.label or _OP_TITLES.get(kind, kind)
    title = " — ".join(str(line) for line in raw_title) \
        if isinstance(raw_title, (list, tuple)) else raw_title
    if kind == "activation" and op.fn and not op.label:
        title = activation_label(op.fn)
    elif kind == "elementwise" and op.fn and not op.label:
        title = _ELEMENTWISE_TITLES.get(op.fn, title)
    elif kind == "opaque":
        opaque_title = op.label or (op.meta or {}).get("class_name") or title
        title = " — ".join(str(line) for line in opaque_title) \
            if isinstance(opaque_title, (list, tuple)) else opaque_title
    facts = []
    if op.in_features and op.out_features:
        facts.append(f"{_fmt_int(op.in_features)} → {_fmt_int(op.out_features)}")
    if op.fn and kind in ("activation", "attention_core"):
        facts.append(str(op.fn))
    return {
        "id": op.id,
        "title": title,
        "description": (op.meta or {}).get("desc") or _OP_SENTENCES.get(kind, ""),
        "facts": facts,
    }


def cards_from_region(region) -> list[dict]:
    """Derive the inspect cards for every drawable op in a region — the
    automatic companion to the rendered SVG, so click targets always have a
    card without any per-view authoring."""
    return [op_card(o) for o in region.ops
            if o.kind != "output" and not (o.kind == "input" and o.id == "hidden")]
