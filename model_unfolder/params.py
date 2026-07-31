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
    """Returns (total_params, active_params_per_token).

    ``gated is None`` (U2 typed unknown) counts the 2-projection floor — the
    caller (``estimate_params``) ANNOTATES the estimate so an unknown never
    silently picks a branch (blast_radius found a −33% FFN hazard here)."""
    hidden = _as_count(hidden)
    if f.kind == "moe":
        # Routed experts and the ordinary/shared FFN are separate mechanisms.
        # Exact fused gate+up expert storage proves three matrix-widths even
        # when the ordinary FFN gate verdict is unknown; never let that proof
        # silently certify a shared expert in the opposite direction.
        expert_g = (
            3 if f.expert_projection_mode in {"fused_gate_up", "split"}
            else 2
        )
        shared_g = 3 if f.gated else 2
        expert_width = _as_count(f.expert_intermediate_size)
        # Shared experts are instances of the expert-width lane; the ordinary
        # dense-layer width is a separate layer-variant fact and must not be
        # substituted here.
        shared_width = expert_width
        per_expert = expert_g * hidden * expert_width
        per_shared = shared_g * hidden * shared_width
        n_routed = _as_count(f.num_experts)
        n_shared = _as_count(f.num_shared_experts)
        n_active = _as_count(f.num_experts_per_tok)
        router = hidden * n_routed
        total = (
            per_expert * n_routed + per_shared * n_shared + router)
        active = (
            per_expert * n_active + per_shared * n_shared + router)
        return total, active
    g = 3 if f.gated else 2
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
    # COR-3 (§8.A): unresolved width -> an explicitly INCOMPLETE estimate.
    # Computing with zero would present "0 params" as a known claim.
    if not h:
        return {"total": None, "active": None, "embed": None, "output": None,
                "per_layer": [], "is_sparse": False,
                "incomplete": "hidden_size unresolved — parameter formulas "
                              "need the model width"}

    # U2 unknown policy: an unknown never silently picks a branch — the
    # estimate keeps a deterministic convention AND says so (``assumptions``
    # rides into the count card). tie=None counts an untied head (upper
    # bound); gated=None counts the 2-projection floor.
    assumptions: list[str] = []

    embed = v * h
    output = 0 if ir.tie_word_embeddings else v * h
    if ir.tie_word_embeddings is None:
        assumptions.append(
            "embedding/head tying unknown — output head counted untied")
    embedding_norm = (
        h if ir.embedding_norm_kind == "rmsnorm"
        else 2 * h if ir.embedding_norm_kind == "layernorm"
        else 0
    )
    if ir.embedding_norm_kind not in {"rmsnorm", "layernorm"}:
        assumptions.append(
            "embedding-stage normalization not proven — its parameters omitted")
    final_norm = (
        h if ir.final_norm_kind == "rmsnorm"
        else 2 * h if ir.final_norm_kind == "layernorm"
        else 0
    )
    if ir.final_norm_kind not in {"rmsnorm", "layernorm"}:
        assumptions.append(
            "final-stage normalization unresolved — its parameters omitted")

    per_layer = []
    layers_total = 0
    layers_active = 0
    is_sparse = False

    for layer in ir.layers:
        a_p = _attn_params(layer.attention, h)
        f_total, f_active = _ffn_params(layer.ffn, h)
        if layer.attention.kind in {None, "", "unknown"} \
                and _ATTENTION_UNKNOWN_NOTE not in assumptions:
            assumptions.append(_ATTENTION_UNKNOWN_NOTE)
        if layer.ffn.kind == "moe":
            is_sparse = True
        pending_notes = (
            (_EXPERT_GATED_NOTE
             if layer.ffn.kind == "moe"
             and layer.ffn.expert_projection_mode is None
             and layer.ffn.expert_intermediate_size is not None else None),
            # ``gated`` describes the ordinary/shared FFN lane.  A routed-only
            # MoE with zero shared experts has no such parameters, so its
            # unknown value cannot affect this formula and must not manufacture
            # a misleading assumption (GPT-OSS is the real control).
            (_GATED_NOTE
             if layer.ffn.gated is None
             and (
                 (
                     layer.ffn.kind == "moe"
                     and _as_count(layer.ffn.num_shared_experts) > 0
                     and layer.ffn.expert_intermediate_size is not None
                 )
                 or (
                     layer.ffn.kind != "moe"
                     and layer.ffn.intermediate_size is not None
                 )
             )
             else None),
            (_EXPERT_WIDTH_NOTE
             if layer.ffn.kind == "moe"
             and layer.ffn.expert_intermediate_size is None
             else None),
            (_ORDINARY_WIDTH_NOTE
             if layer.ffn.intermediate_size is None
             and layer.ffn.kind != "moe"
             else None),
        )
        for note in pending_notes:
            if note is not None and note not in assumptions:
                assumptions.append(note)
        per_norm = (
            h if layer.norm_kind == "rmsnorm"
            else 2 * h if layer.norm_kind == "layernorm"
            else 0
        )
        norm_count = (
            4 if layer.norm_placement == "double"
            and layer.residual_topology == "sequential"
            else 2 if layer.norm_placement in {"pre", "post"}
            and layer.residual_topology == "sequential"
            else layer.parallel_norm_count
            if layer.residual_topology == "parallel"
            else None
        )
        norm_p = per_norm * norm_count if norm_count is not None else 0
        if per_norm == 0 or norm_count is None:
            note = (
                "layer normalization parameters unresolved — omitted until "
                "kind, placement, and residual topology are owner-proven"
            )
            if note not in assumptions:
                assumptions.append(note)
        t = a_p + f_total + norm_p
        ac = a_p + f_active + norm_p
        per_layer.append({"total": t, "active": ac, "attn": a_p, "ffn": f_total})
        layers_total += t
        layers_active += ac

    total = embed + output + embedding_norm + final_norm + layers_total
    active = embed + output + embedding_norm + final_norm + layers_active

    return {
        "total": total,
        "active": active,
        "embed": embed,
        "output": output,
        "per_layer": per_layer,
        "is_sparse": is_sparse,
        # Only-when-present so every fully-resolved model stays byte-stable.
        **({"assumptions": assumptions} if assumptions else {}),
    }


#: U2: the one-line annotation for an unknown FFN gate structure.
_ATTENTION_UNKNOWN_NOTE = (
    "attention mechanism unknown — Q/K/V/O parameter estimate is a "
    "temporary estimation convention, not code-proven architecture"
)
_GATED_NOTE = ("FFN structure unknown — counted as 2 projections "
               "(a gated FFN would add hidden x inner per layer)")
_EXPERT_GATED_NOTE = (
    "routed-expert structure unknown — counted as 2 projections "
    "(a gated expert would add hidden x expert-inner per expert)")
_EXPERT_WIDTH_NOTE = (
    "routed-expert inner width unknown — expert matrix terms omitted")
_ORDINARY_WIDTH_NOTE = (
    "ordinary/shared FFN inner width unknown — its matrix terms omitted")


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
