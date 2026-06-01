"""Attention spec + operation graph.

Schema (per the test contract):

* ``kind``                — gqa / mha / mqa / mla
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

from .ops import linear, node, edges_from_nodes
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


# ---------- operation graph ----------


def _operation_graph(attn: dict, hidden: int | None, heads: dict) -> dict[str, Any]:
    if attn.get("kind") == "mla":
        nodes = _mla_nodes(attn, hidden, heads)
    else:
        nodes = _sdpa_nodes(attn, hidden, heads)
    return {"nodes": nodes, "edges": edges_from_nodes(nodes)}


def _sdpa_nodes(attn: dict, hidden: int | None, heads: dict) -> list[dict[str, Any]]:
    q_w  = heads.get("query_width")
    kv_w = heads.get("key_value_width")
    return [
        node("hidden",    "input",              width=hidden),
        node("q_proj",    "linear",             inputs=["hidden"], outputs=["q"],     parameters=linear(hidden, q_w)),
        node("k_proj",    "linear",             inputs=["hidden"], outputs=["key"],   parameters=linear(hidden, kv_w)),
        node("v_proj",    "linear",             inputs=["hidden"], outputs=["value"], parameters=linear(hidden, kv_w)),
        node("kv_cache",  "cache",              inputs=["key", "value"], outputs=["key_cached", "value_cached"], stores=["key", "value"]),
        node("scores",    "scaled_dot_product", inputs=["q", "key_cached"], outputs=["scores"], formula="QK^T/sqrt(dim)"),
        node("softmax",   "softmax",            inputs=["scores"], outputs=["weights"]),
        node("context",   "matmul",             inputs=["weights", "value_cached"], outputs=["context"]),
        node("concat_heads", "reshape_concat",  inputs=["context"], outputs=["attention_out"], width=q_w),
        node("o_proj",    "linear",             inputs=["attention_out"], outputs=["residual_delta"], parameters=linear(q_w, hidden)),
    ]


def _mla_nodes(attn: dict, hidden: int | None, heads: dict) -> list[dict[str, Any]]:
    q_w     = heads.get("query_width")
    q_rank  = attn.get("q_lora_rank")
    kv_rank = attn.get("kv_lora_rank")
    rope    = attn.get("rope_dim")
    return [
        node("hidden",     "input",              width=hidden),
        node("q_lora_down","linear",             inputs=["hidden"],         outputs=["q_latent"],         parameters=linear(hidden, q_rank)),
        node("q_lora_up",  "linear",             inputs=["q_latent"],       outputs=["q_nope", "q_rope"], parameters=linear(q_rank, q_w)),
        node("q_rope",     "rope",               inputs=["q_rope"],         outputs=["q_rope_encoded"],   width=rope),
        node("q_concat",   "concat",             inputs=["q_nope", "q_rope_encoded"], outputs=["q"]),
        node("kv_compress","linear",             inputs=["hidden"],         outputs=["kv_latent"],        parameters=linear(hidden, kv_rank)),
        node("kv_cache",   "cache",              inputs=["kv_latent"],      outputs=["kv_latent_cached"], stores=["kv_latent"]),
        node("kv_expand",  "linear",             inputs=["kv_latent_cached"], outputs=["k_nope", "value"], parameters=linear(kv_rank, q_w)),
        node("k_rope",     "rope",               inputs=["kv_latent_cached"], outputs=["k_rope_encoded"], width=rope),
        node("k_concat",   "concat",             inputs=["k_nope", "k_rope_encoded"], outputs=["key"]),
        node("scores",     "scaled_dot_product", inputs=["q", "key"],       outputs=["scores"], formula="QK^T/sqrt(dim)"),
        node("softmax",    "softmax",            inputs=["scores"],         outputs=["weights"]),
        node("context",    "matmul",             inputs=["weights", "value"], outputs=["context"]),
        node("concat_heads", "reshape_concat",   inputs=["context"],        outputs=["attention_out"], width=q_w),
        node("output_projection", "linear",      inputs=["attention_out"],  outputs=["residual_delta"], parameters=linear(q_w, hidden)),
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
