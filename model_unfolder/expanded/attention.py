"""Attention spec + operation graph.

The operation graph is **not** authored here — it is projected from the one
canonical :func:`...opgraph.attention_region`, the same region the HTML
renderer draws (MLA's query/KV drill regions are embedded as nested
``subgraph`` graphs).  The schema keeps its published node names
(``scores``/``softmax``/``context``) via an explicit rename of the region's
ids, and the kv-cache node is spliced into the dataflow for cached SDPA kinds.

Schema (per the test contract):

* ``kind``                — gqa / mha / mqa / mla / ssm / recurrent / rwkv / linear
* ``heads``               — query / key_value / kv_groups / head_dim
                            + query_width / key_value_width / residual_width
                            (and ``expanded_attention_width`` when q*hd != hidden)
* ``mask``                — type + optional window_size
* ``projections``         — named linear specs (query, key, value, output | MLA: q_lora_*, kv_lora_*, output)
* ``operation_graph``     — DAG of {id, operation, inputs, outputs, parameters?, formula?}
* ``cache``               — kv-cache descriptor
* ``trace``               — ir_path + code_finding_ids
"""
from __future__ import annotations

from typing import Any

from ..opgraph import attention_region, mla_kv_region, mla_query_region
from .ops import linear, node
from .region import region_to_json
from .utils import drop_none


def build_attention(attn: dict, hidden: int | None, group_path: str, evidence: dict | None) -> dict[str, Any]:
    kind = attn.get("kind")
    heads = _heads(attn, hidden)
    out: dict[str, Any] = {
        "kind":            kind,
        "heads":           heads,
        "mask":            drop_none({"type": attn.get("mask"), "window_size": attn.get("window_size")}),
        "projections":     _projections(attn, hidden, heads),
        "operation_graph": _operation_graph(attn, hidden, heads),
        "cache":           _cache(attn),
        "trace": {
            "ir_path":          f"{group_path}.attention",
            "code_finding_ids": _evidence_ids(evidence, "attention", _evidence_values(kind)),
        },
    }
    out.update(drop_none({
        "qk_norm":         attn.get("qk_norm") or None,
        "bias":            attn.get("bias") or None,
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
    return drop_none({
        "query":  linear(hidden, q_w),
        "key":    linear(hidden, kv_w),
        "value":  linear(hidden, kv_w),
        "output": linear(q_w, residual_w),
    })


def _cache(attn: dict) -> dict[str, Any]:
    kind = attn.get("kind")
    if kind == "mla":
        return {
            "enabled": True,
            "kind":    "latent_kv",
            "stores":  ["kv_latent"],
            "rank":    attn.get("kv_lora_rank"),
        }
    if kind in {"mha", "gqa", "mqa"}:
        return drop_none({
            "enabled":  True,
            "kind":     "kv",
            "stores":   ["key", "value"],
            "kv_heads": attn.get("num_kv_heads"),
            "head_dim": attn.get("head_dim"),
        })
    return {"enabled": False}


# ---------- operation graph (projected from the canonical region) ----------

#: region op ids -> the schema's published node names.  The region keeps ONE id
#: set (also the render/card coupling); the schema contract keeps its names.
_PUBLIC_IDS = {
    "scaled_scores": "scores",
    "attn_softmax": "softmax",
    "attn_apply_v": "context",
}

_CACHED_SDPA_KINDS = {"mha", "gqa", "mqa", None}


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
    if kind in _CACHED_SDPA_KINDS:
        _splice_kv_cache(graph, attn)
    return graph


def _splice_kv_cache(graph: dict[str, Any], attn: dict) -> None:
    """Insert the kv-cache node into the K/V dataflow (write after the
    projections, read by scores and context) — a cache-semantics enrichment of
    the projected structure, not a second authoring of it."""
    nodes = graph["nodes"]
    ids = {n["id"] for n in nodes}
    if not {"k_proj", "v_proj", "scores", "context"} <= ids:
        return
    cache = node("kv_cache", "cache", inputs=["k_proj", "v_proj"],
                 outputs=["kv_cache"], stores=["key", "value"],
                 kv_heads=attn.get("num_kv_heads"))
    at = next(i for i, n in enumerate(nodes) if n["id"] == "v_proj") + 1
    nodes.insert(at, cache)
    for n in nodes:
        if n["id"] in {"scores", "context"} and n.get("inputs"):
            n["inputs"] = ["kv_cache" if i in {"k_proj", "v_proj"} else i
                           for i in n["inputs"]]
    graph["edges"] = [
        e for e in graph["edges"]
        if not (e["from"] in {"k_proj", "v_proj"} and e["to"] in {"scores", "context"})
    ] + [
        {"from": "k_proj", "to": "kv_cache"}, {"from": "v_proj", "to": "kv_cache"},
        {"from": "kv_cache", "to": "scores"}, {"from": "kv_cache", "to": "context"},
    ]


# ---------- evidence linking ----------


def _evidence_values(kind: str | None) -> list[str]:
    if kind == "mla":
        return ["mla", "grouped_kv_attention"]
    if kind == "mqa":
        return ["multi_query_attention", "grouped_kv_attention", "split_qkv_attention", "fused_qkv_attention"]
    if kind == "gqa":
        return ["grouped_kv_attention", "split_qkv_attention", "fused_qkv_attention"]
    if kind == "mha":
        return ["split_qkv_attention", "fused_qkv_attention"]
    return [kind] if kind else []


def _evidence_ids(evidence: dict | None, kind: str, values: list[str]) -> list[str]:
    if not evidence:
        return []
    detections = evidence.get("detections") or {}
    out: list[str] = []
    for v in values:
        out.extend(((detections.get(kind) or {}).get(v) or {}).get("finding_ids") or [])
    return out
