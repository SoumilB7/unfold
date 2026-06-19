"""Rough parameter-count estimation from an IR.

These are *estimates*. We don't try to model every implementation detail
(bias terms, MLA's exact projection layout, expert grouping, etc.) — just
enough to give the right order of magnitude and a useful active/total split
for MoE models.
"""
from __future__ import annotations
from .ir import ModelIR, AttentionSpec, FFNSpec


def _attn_params(a: AttentionSpec, hidden: int) -> int:
    h = _as_count(hidden)
    nq = _as_count(a.num_heads)
    head_dim = _as_count(a.head_dim) or (h // max(nq, 1))
    nkv = _as_count(a.num_kv_heads) or nq

    if a.kind == "mla":
        # MLA splits each head into a non-positional (nope) part and a rotary
        # (rope) part for Q/K, with V its own width.  These dims are what make
        # the count correct — falling back to head_dim (hidden/num_heads) badly
        # undercounts (DeepSeek heads are 192/128 wide, not hidden/num_heads).
        qk_rope = a.qk_rope_head_dim or a.rope_dim or 0
        qk_nope = a.qk_nope_head_dim or max(head_dim - qk_rope, 0) or head_dim
        qk_head = qk_nope + qk_rope          # Q/K per-head width
        v_head = a.v_head_dim or head_dim    # V per-head width
        # Q path: hidden -> [q_lora] -> nq*qk_head  (LoRA down/up when present)
        if a.q_lora_rank:
            q = h * a.q_lora_rank + a.q_lora_rank * (nq * qk_head)
        else:
            q = h * (nq * qk_head)
        # KV path: hidden -> (kv_lora + rope), then kv_lora -> nq*(nope + v)
        kv_lora = a.kv_lora_rank or 0
        kv_a = h * (kv_lora + qk_rope)
        kv_b = kv_lora * (nq * (qk_nope + v_head))
        out = (nq * v_head) * h
        return q + kv_a + kv_b + out

    qkv = h * (nq + 2 * nkv) * head_dim
    out = nq * head_dim * h
    return qkv + out


def _as_count(v, default: int = 0) -> int:
    """Coerce a config count/size to an int for parameter estimation.

    Some configs declare a per-layer/per-block LIST (or None) where a scalar is
    expected (e.g. a heterogeneous MoE schedule).  Parameter counts are estimates,
    so fall back to the first numeric element / the default rather than crashing —
    the diagram still renders, with an approximate count, instead of failing.
    """
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, (list, tuple)):
        for x in v:
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return int(x)
    return default


def _ffn_params(f: FFNSpec, hidden: int) -> tuple:
    """Returns (total_params, active_params_per_token)."""
    g = 3 if f.gated else 2
    hidden = _as_count(hidden)
    if f.kind == "moe":
        per_expert = g * hidden * _as_count(f.expert_intermediate_size or f.intermediate_size)
        n_routed = _as_count(f.num_experts)
        n_shared = _as_count(f.num_shared_experts)
        n_active = _as_count(f.num_experts_per_tok)
        router = hidden * n_routed
        total = per_expert * (n_routed + n_shared) + router
        active = per_expert * (n_active + n_shared) + router
        return total, active
    p = g * hidden * _as_count(f.intermediate_size)
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
