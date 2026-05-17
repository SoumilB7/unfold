"""FFN spec + operation graph (dense / gated / MoE)."""
from __future__ import annotations

from typing import Any

from .ops import linear, node, edges_from_nodes
from .utils import drop_none


def build_ffn(ffn: dict, hidden: int | None, group_path: str, evidence: dict | None) -> dict[str, Any]:
    kind = ffn.get("kind")
    out: dict[str, Any] = {
        "kind":              kind,
        "activation":        ffn.get("activation"),
        "intermediate_size": ffn.get("intermediate_size"),
        "gated":             bool(ffn.get("gated")),
        "operation_graph":   _operation_graph(ffn, hidden),
        "trace": {
            "ir_path":          f"{group_path}.ffn",
            "code_finding_ids": _evidence_ids(evidence, "ffn", _evidence_values(ffn)),
        },
    }
    if kind == "moe":
        n = ffn.get("num_experts")
        k = ffn.get("num_experts_per_tok")
        out["router"]  = drop_none({"num_experts": n, "top_k": k,
                                    "active_fraction": (k / n) if n and k else None})
        out["experts"] = drop_none({"count": n,
                                    "shared": ffn.get("num_shared_experts") or 0,
                                    "expert_intermediate_size": ffn.get("expert_intermediate_size") or ffn.get("intermediate_size")})
    return drop_none(out)


# ---------- operation graph ----------


def _operation_graph(ffn: dict, hidden: int | None) -> dict[str, Any]:
    intermediate = ffn.get("expert_intermediate_size") or ffn.get("intermediate_size")
    if ffn.get("kind") == "moe":
        nodes = [
            node("hidden",          "input",           width=hidden),
            node("router",          "top_k_router",    inputs=["hidden"], outputs=["expert_indices", "expert_weights"], top_k=ffn.get("num_experts_per_tok")),
            node("expert_template", "ffn_template",    inputs=["hidden"], outputs=["expert_output"], graph=_dense_nodes(ffn, hidden, intermediate)),
            node("weighted_sum",    "weighted_sum",    inputs=["expert_output", "expert_weights"], outputs=["residual_delta"]),
        ]
    else:
        nodes = _dense_nodes(ffn, hidden, intermediate)
    return {"nodes": nodes, "edges": edges_from_nodes(nodes)}


def _dense_nodes(ffn: dict, hidden: int | None, intermediate: int | None) -> list[dict[str, Any]]:
    if ffn.get("gated"):
        return [
            node("hidden",     "input",                  width=hidden),
            node("gate_proj",  "linear",                 inputs=["hidden"], outputs=["gate"], parameters=linear(hidden, intermediate)),
            node("up_proj",    "linear",                 inputs=["hidden"], outputs=["up"],   parameters=linear(hidden, intermediate)),
            node("activation", "activation",             inputs=["gate"],   outputs=["gate_act"], function=ffn.get("activation")),
            node("multiply",   "elementwise_multiply",   inputs=["gate_act", "up"], outputs=["intermediate"], width=intermediate),
            node("down_proj",  "linear",                 inputs=["intermediate"], outputs=["residual_delta"], parameters=linear(intermediate, hidden)),
        ]
    return [
        node("hidden",     "input",      width=hidden),
        node("up_proj",    "linear",     inputs=["hidden"],          outputs=["intermediate"],    parameters=linear(hidden, intermediate)),
        node("activation", "activation", inputs=["intermediate"],    outputs=["intermediate_act"], function=ffn.get("activation")),
        node("down_proj",  "linear",     inputs=["intermediate_act"], outputs=["residual_delta"], parameters=linear(intermediate, hidden)),
    ]


# ---------- evidence linking ----------


def _evidence_values(ffn: dict) -> list[str]:
    if ffn.get("kind") == "moe":
        return ["mixture_of_experts"]
    return ["gated_dense_ffn" if ffn.get("gated") else "plain_dense_ffn"]


def _evidence_ids(evidence: dict | None, kind: str, values: list[str]) -> list[str]:
    if not evidence:
        return []
    detections = evidence.get("detections") or {}
    out: list[str] = []
    for v in values:
        out.extend(((detections.get(kind) or {}).get(v) or {}).get("finding_ids") or [])
    return out
