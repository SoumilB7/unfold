"""Rough parameter-count estimation from an IR.

These are *estimates*. We don't try to model every implementation detail
(bias terms, MLA's exact projection layout, expert grouping, etc.) — just
enough to give the right order of magnitude and a useful active/total split
for MoE models.
"""
from __future__ import annotations
from .ir import ModelIR, AttentionSpec, FFNSpec


def _attn_params(a: AttentionSpec, hidden: int) -> int:
    h = hidden
    head_dim = a.head_dim or (h // max(a.num_heads, 1))
    nq = a.num_heads
    nkv = a.num_kv_heads or nq

    if a.kind == "mla":
        # Q path: optional LoRA down then up to (nope+rope)*nq
        q_out = nq * head_dim
        if a.q_lora_rank:
            q = h * a.q_lora_rank + a.q_lora_rank * q_out
        else:
            q = h * q_out
        # KV path: hidden -> (kv_lora_rank + rope_dim), then up to nq*(nope+v)
        kv_lora = a.kv_lora_rank or 0
        rope = a.rope_dim or 0
        kv_down = h * (kv_lora + rope)
        kv_up = kv_lora * (nq * head_dim * 2)  # K nope + V — rough
        kv = kv_down + kv_up
        out = nq * head_dim * h
        return q + kv + out

    qkv = h * (nq + 2 * nkv) * head_dim
    out = nq * head_dim * h
    return qkv + out


def _ffn_params(f: FFNSpec, hidden: int) -> tuple:
    """Returns (total_params, active_params_per_token)."""
    g = 3 if f.gated else 2
    if f.kind == "moe":
        per_expert = g * hidden * (f.expert_intermediate_size or f.intermediate_size)
        n_routed = f.num_experts or 0
        n_shared = f.num_shared_experts or 0
        n_active = f.num_experts_per_tok or 0
        router = hidden * n_routed
        total = per_expert * (n_routed + n_shared) + router
        active = per_expert * (n_active + n_shared) + router
        return total, active
    p = g * hidden * f.intermediate_size
    return p, p


def estimate_params(ir: ModelIR) -> dict:
    """Estimate parameter counts for a model.

    Returns a dict::

        {
            "total":     int,   # all parameters
            "active":    int,   # active per token (== total for non-MoE)
            "embed":     int,
            "output":    int,
            "per_layer": [{"total": int, "active": int}, ...],
            "is_sparse": bool,
        }
    """
    h = ir.hidden_size
    v = ir.vocab_size

    embed = v * h
    output = 0 if ir.tie_word_embeddings else v * h
    final_norm = h

    per_layer = []
    layers_total = 0
    layers_active = 0
    is_sparse = False

    for layer in ir.layers:
        a_p = _attn_params(layer.attention, h)
        f_total, f_active = _ffn_params(layer.ffn, h)
        if layer.ffn.kind == "moe":
            is_sparse = True
        norm_p = 2 * h
        t = a_p + f_total + norm_p
        ac = a_p + f_active + norm_p
        per_layer.append({"total": t, "active": ac, "attn": a_p, "ffn": f_total})
        layers_total += t
        layers_active += ac

    total = embed + output + final_norm + layers_total
    active = embed + output + final_norm + layers_active

    return {
        "total": total,
        "active": active,
        "embed": embed,
        "output": output,
        "per_layer": per_layer,
        "is_sparse": is_sparse,
    }


def humanize(n: int) -> str:
    """Format a parameter count as e.g. '671B', '37.4B', '8.2M'."""
    if n is None:
        return "?"
    n = float(n)
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            v = n / scale
            if v >= 100:
                return f"{v:.0f}{unit}"
            if v >= 10:
                return f"{v:.1f}{unit}"
            return f"{v:.2f}{unit}"
    return f"{int(n)}"
