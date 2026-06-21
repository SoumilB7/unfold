"""FFN spec + operation graph (dense / gated / MoE).

The operation graph is **not** authored here — it is projected from the one
canonical :func:`...opgraph.ffn_region`, the same region the HTML renderer draws.
This module only maps that region onto the JSON node schema.
"""
from __future__ import annotations

from typing import Any

from ..opgraph import ffn_region
from .ops import edges_from_nodes, node
from .region import region_to_json as _region_to_json
from .utils import drop_none


def build_ffn(ffn: dict, hidden: int | None, group_path: str, evidence: dict | None) -> dict[str, Any]:
    kind = ffn.get("kind")
    # gated is tri-state: True / False are declared facts; None means the config
    # does not declare the inner structure.  Keep that distinction in the JSON
    # (gated=null + structure_declared=false) rather than collapsing None to false.
    declared = ffn.get("gated") is not None or kind == "moe"
    out: dict[str, Any] = {
        "kind":              kind,
        "activation":        ffn.get("activation"),
        "activation_assumed": ffn.get("activation_assumed") or None,
        "activation_from_class": ffn.get("activation_from_class") or None,
        "intermediate_size": ffn.get("intermediate_size"),
        "gated":             bool(ffn.get("gated")) if declared else None,
        "structure_declared": None if declared else False,
        "operation_graph":   _operation_graph(ffn, hidden),
        "trace": {
            "ir_path":          f"{group_path}.ffn",
            "code_finding_ids": _evidence_ids(evidence, "ffn", _evidence_values(ffn)),
        },
    }
    if kind == "moe":
        n = ffn.get("num_experts")
        k = ffn.get("num_experts_per_tok")
        routing = ffn.get("routing") or {}
        out["router"]  = drop_none({"num_experts": n, "top_k": k,
                                    "active_fraction": (k / n) if n and k else None,
                                    **routing})
        out["experts"] = drop_none({"count": n,
                                    "shared": ffn.get("num_shared_experts") or 0,
                                    "expert_intermediate_size": ffn.get("expert_intermediate_size") or ffn.get("intermediate_size")})
    return drop_none(out)


# ---------- operation graph ----------


def _operation_graph(ffn: dict, hidden: int | None) -> dict[str, Any]:
    intermediate = ffn.get("expert_intermediate_size") or ffn.get("intermediate_size")
    if ffn.get("kind") == "moe":
        # MoE keeps its router/template framing, but the expert's internals are
        # the same canonical region the renderer draws.
        expert = ffn_region(
            {"kind": "dense", "gated": bool(ffn.get("gated", True)),
             "activation": ffn.get("activation"), "intermediate_size": intermediate},
            hidden,
        )
        nodes = [
            node("hidden",          "input",        width=hidden),
            node("router",          "top_k_router", inputs=["hidden"], outputs=["expert_indices", "expert_weights"], top_k=ffn.get("num_experts_per_tok")),
            node("expert_template", "ffn_template", inputs=["hidden"], outputs=["expert_output"], graph=_region_to_json(expert)),
            node("weighted_sum",    "weighted_sum", inputs=["expert_output", "expert_weights"], outputs=["residual_delta"]),
        ]
        return {"nodes": nodes, "edges": edges_from_nodes(nodes)}
    return _region_to_json(ffn_region(ffn, hidden))


# ---------- evidence linking ----------


def _evidence_values(ffn: dict) -> list[str]:
    if ffn.get("kind") == "moe":
        return ["mixture_of_experts"]
    if ffn.get("gated") is None:
        return []   # inner structure undeclared — claim no specific FFN evidence
    return ["gated_dense_ffn" if ffn.get("gated") else "plain_dense_ffn"]


def _evidence_ids(evidence: dict | None, kind: str, values: list[str]) -> list[str]:
    if not evidence:
        return []
    detections = evidence.get("detections") or {}
    out: list[str] = []
    for v in values:
        out.extend(((detections.get(kind) or {}).get(v) or {}).get("finding_ids") or [])
    return out
