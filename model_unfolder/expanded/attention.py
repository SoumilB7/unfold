"""Attention spec + operation graph.

The operation graph is **not** authored here — it is projected from the one
canonical :func:`...opgraph.attention_region`, the same region the HTML
renderer draws (MLA's query/KV drill regions are embedded as nested
``subgraph`` graphs).  The schema keeps its published node names
(``scores``/``softmax``/``context``) via an explicit rename of the region's
ids; cached SDPA kinds project the canonical region's own K/V-cache node.

Schema (per the test contract):

* ``kind``                — gqa / mha / mqa / mla / ssm / recurrent / rwkv / linear
* ``heads``               — query / key_value / kv_groups / head_dim
                            + query_width / key_value_width / residual_width
                            (and ``expanded_attention_width`` when q*hd != hidden)
* ``mask``                — type + optional window_size
* ``projections``         — named linear specs (query, key, value, output | MLA: q_lora_*, kv_lora_*, output)
* ``operation_graph``     — DAG of {id, operation, inputs, outputs, parameters?, formula?}
* ``cache``               — kv-cache descriptor
* ``trace``               — exact IR path. Finding ids remain empty until an
                            owner-qualified fact receipt exists; global
                            semantic buckets are not provenance.
"""
from __future__ import annotations

from typing import Any

from ..opgraph import attention_region, mla_kv_region, mla_query_region
from .ops import linear
from .region import region_to_json
from .utils import drop_none


def build_attention(attn: dict, hidden: int | None,
                    group_path: str, evidence: dict | None = None) -> dict[str, Any]:
    """Project canonical attention; ``evidence`` is ignored compatibility input.

    The global diagnostic document has no owner identity, so consuming it here
    would recreate the sibling-provenance bug. U14 may replace this argument
    with a typed owner-qualified receipt; U5 deliberately does not reinterpret
    it.
    """
    kind = attn.get("kind")
    heads = _heads(attn, hidden)
    out: dict[str, Any] = {
        "kind":            kind,
        "heads":           heads,
        "mask":            drop_none({"type": attn.get("mask"), "window_size": attn.get("window_size")}),
        "projections":     _projections(attn, hidden, heads),
        "operation_graph": _operation_graph(attn, hidden, heads),
        "cache":           _cache(attn),
        "qk_norm":         attn.get("qk_norm"),
        "bias":            attn.get("bias"),
        "rope":            attn.get("rope"),
        "position_kind":   attn.get("position_kind"),
        "position_application": attn.get("position_application"),
        "projection_mode": attn.get("projection_mode"),
        "scores_scaled":   attn.get("scores_scaled"),
        "trace": {
            "ir_path":          f"{group_path}.attention",
            "code_finding_ids": [],
        },
    }
    out.update(drop_none({
        "qkv_clip":        attn.get("qkv_clip"),
        "shared":          attn.get("shared") or None,
        "no_rope":         attn.get("no_rope") or None,
        "kv_source_layer": attn.get("kv_source_layer"),
        "kv_lora_rank":    attn.get("kv_lora_rank"),
        "q_lora_rank":     attn.get("q_lora_rank"),
        "rope_dim":        attn.get("rope_dim"),
    }))
    return out


# ---------- spec slices ----------


def _heads(attn: dict, hidden: int | None) -> dict[str, Any]:
    q  = attn.get("num_heads")
    kv = attn.get("num_kv_heads")
    hd = attn.get("head_dim")
    q_width  = q * hd if q and hd else None
    kv_width = kv * hd if kv and hd else None
    return drop_none({
        "query":                   q,
        "key_value":               kv,
        "kv_groups":               (q // kv) if q and kv else None,
        "head_dim":                hd,
        "query_width":             q_width,
        "key_value_width":         kv_width,
        "residual_width":          hidden,
        "expanded_attention_width": q_width if (q_width is not None and hidden is not None and q_width != hidden) else None,
    })


def _projections(attn: dict, hidden: int | None, heads: dict) -> dict[str, Any]:
    q_w  = heads.get("query_width")
    kv_w = heads.get("key_value_width")
    residual_w = heads.get("residual_width")
    if attn.get("kind") == "mla":
        return drop_none({
            "query_lora_a": linear(hidden, attn.get("q_lora_rank")),
            "query_lora_b": linear(attn.get("q_lora_rank"), q_w),
            "kv_lora_a":    linear(hidden, attn.get("kv_lora_rank")),
            "kv_lora_b":    linear(attn.get("kv_lora_rank"), q_w),
            "output":       linear(q_w, residual_w),
        })
    if attn.get("kind") not in {"mha", "gqa", "mqa", "linear"}:
        # Geometry alone does not prove Q/K/V storage.  Unknown and non-SDPA
        # mixers project their canonical region below; expanded JSON must not
        # independently manufacture the conventional four Linears.  Linear
        # attention is included because its canonical region explicitly
        # contains Q/K/V/out projections (Sana's code-proven processor).
        return {}
    if attn.get("kind") in {"mha", "gqa", "mqa"} \
            and attn.get("projection_mode") != "split_qkv":
        if attn.get("projection_mode") == "fused_qkv":
            return drop_none({
                "fused_qkv": linear(hidden, q_w + 2 * kv_w
                                    if q_w is not None and kv_w is not None
                                    else None),
                "output": linear(q_w, residual_w),
            })
        return {}
    return drop_none({
        "query":  linear(hidden, q_w),
        "key":    linear(hidden, kv_w),
        "value":  linear(hidden, kv_w),
        "output": linear(q_w, residual_w),
    })


def _cache(attn: dict) -> dict[str, Any]:
    kind = attn.get("kind")
    cached = attn.get("cached")
    if cached is None and kind in {"mha", "gqa", "mqa", "mla"}:
        return {"enabled": None, "status": "unresolved"}
    if cached is False:
        return {"enabled": False, "kind": "none"}
    if kind == "mla" and cached is True:
        return {
            "enabled": True,
            "kind":    "latent_kv",
            "stores":  ["kv_latent"],
            "rank":    attn.get("kv_lora_rank"),
        }
    if kind in {"mha", "gqa", "mqa"} and cached is True:
        return drop_none({
            "enabled":  True,
            "kind":     "kv",
            "stores":   ["key", "value"],
            "kv_heads": attn.get("num_kv_heads"),
            "head_dim": attn.get("head_dim"),
        })
    # An unresolved or non-KV attention mechanism cannot carry a KV-cache
    # verdict.  ``False`` is reserved for a code/config-proven uncached
    # cache-capable mechanism; do not collapse not-applicable into that fact.
    return {"enabled": None, "status": "not_applicable"}


# ---------- operation graph (projected from the canonical region) ----------

#: region op ids -> the schema's published node names.  The region keeps ONE id
#: set (also the render/card coupling); the schema contract keeps its names.
_PUBLIC_IDS = {
    "scaled_scores": "scores",
    "attn_softmax": "softmax",
    "attn_apply_v": "context",
}

def _operation_graph(attn: dict, hidden: int | None, heads: dict) -> dict[str, Any]:
    region = attention_region(attn, hidden)
    kind = attn.get("kind")
    if kind == "mla":
        # Embed the canonical query/KV drill regions in their subgraph ops —
        # the same regions the renderer's drill-down views draw.
        nested = {"mla_query_path": mla_query_region(attn, hidden),
                  "mla_kv_path": mla_kv_region(attn, hidden)}
        for op in region.ops:
            if op.id in nested:
                op.meta["region"] = nested[op.id]
    graph = region_to_json(region, rename=_PUBLIC_IDS)
    return graph
