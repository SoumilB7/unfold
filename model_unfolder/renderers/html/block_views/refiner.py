"""Secondary-stack (refiner) tower — THE one tower projection.

A refiner detected by the general secondary-stack evidence is just another
transformer tower: its drill renders the ``sub_model`` groups through the ONE
cell projector (placement, gates, labels can never diverge from any other
tower), inside the ONE frame.  Its attention/FFN drills are the registered
canonical views, namespaced per group by the shared card builder.
"""
from __future__ import annotations

from ....labels import attention_tower_label
from ..tower import tower_cell, tower_graph
from ..graph_engine import render_graph


def build_refiner_tower_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    detail = block.get("detail") if isinstance(block.get("detail"), dict) else {}
    sub_model = detail.get("sub_model") if isinstance(detail.get("sub_model"), dict) else {}
    groups = sub_model.get("groups") or []
    prefix = str(block.get("id") or "refiner")

    spec: dict = {
        "source": {"id": f"{prefix}_in", "label": detail.get("entry_label")},
        "output": {"id": f"{prefix}_out", "static": True},
    }
    if len(groups) > 1:
        spec["cells"] = [{"cell": _cell(group, f"{prefix}_g{k}"),
                          "repeat": group.get("count")}
                         for k, group in enumerate(groups)]
    elif groups:
        spec["cell"] = _cell(groups[0], prefix)
        spec["repeat"] = groups[0].get("count") or sub_model.get("layers")
    else:
        spec["cell"] = [{"id": f"{prefix}_op_unknown", "kind": "norm",
                         "label": "Code-defined block", "resolved": False}]
        spec["repeat"] = sub_model.get("layers")
    graph = tower_graph(spec)
    return render_graph(graph, info, mount_id, "refiner-tower",
                        f"{ir.get('name', 'model')} {str(block.get('title') or 'refiner')}")


def _cell(group: dict, prefix: str) -> list[dict]:
    placement = group.get("norm_placement")
    gate = group.get("residual_gate")
    return tower_cell(
        prefix,
        attn_label=attention_tower_label(group.get("attention") or {}),
        norm_label=group.get("norm") or "Norm",
        placement=placement if placement in ("pre", "post", "double") else "unknown",
        ffn_fact=group.get("ffn") or {},
        attn_gate=gate,
        ffn_gate=gate,
        unknown_label="Code-defined block",
    )
