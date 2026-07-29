"""FFN spec + operation graph (dense / gated / MoE).

The operation graph is **not** authored here — it is projected from the one
canonical :func:`...opgraph.ffn_region`, the same region the HTML renderer draws.
This module only maps that region onto the JSON node schema.
"""
from __future__ import annotations

from typing import Any

from ..opgraph import ffn_region, ffn_structure_declared, ffn_structure_state
from .ops import edges_from_nodes, node
from .region import region_to_json as _region_to_json
from .utils import drop_none


def build_ffn(ffn: dict, hidden: int | None, group_path: str, evidence: dict | None) -> dict[str, Any]:
    kind = ffn.get("kind")
    structure_state = ffn_structure_state(ffn)
    out: dict[str, Any] = {
        "kind":              kind,
        "activation":        ffn.get("activation"),
        "activation_assumed": ffn.get("activation_assumed") or None,
        "activation_from_class": ffn.get("activation_from_class") or None,
        "intermediate_size": ffn.get("intermediate_size"),
        "gated":             ffn.get("gated"),
        "projection_mode":   ffn.get("projection_mode"),
        "structure_state":    structure_state,
        "structure_declared": ffn_structure_declared(ffn),
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
        # These are independent expert facts. Preserve explicit unknowns so a
        # consumer cannot borrow the ordinary FFN's width/storage when an
        # expert-local value is absent.
        out["experts"] = {
            "count": n,
            "shared": ffn.get("num_shared_experts"),
            "expert_intermediate_size": ffn.get("expert_intermediate_size"),
            "projection_mode": ffn.get("expert_projection_mode"),
        }
    # U4-C: unknown is data, not an omitted key that a downstream default may
    # refill. Keep the top-level tri-state fields in the expanded contract.
    return out


# ---------- operation graph ----------


def _operation_graph(ffn: dict, hidden: int | None) -> dict[str, Any]:
    if ffn.get("kind") == "moe":
        # MoE keeps its router/template framing, but the expert's internals are
        # the same canonical region the renderer draws.
        expert_mode = ffn.get("expert_projection_mode")
        expert_gated = (
            True if expert_mode in {"split", "fused_gate_up"}
            else False if expert_mode == "dense"
            else None
        )
        expert = ffn_region(
            {
                "kind": "dense",
                "gated": expert_gated,
                # The top-level activation belongs to the ordinary/shared FFN.
                # Expert activation remains unknown until U7 proves it locally.
                "activation": None,
                "intermediate_size": ffn.get("expert_intermediate_size"),
                "projection_mode": expert_mode,
            },
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
