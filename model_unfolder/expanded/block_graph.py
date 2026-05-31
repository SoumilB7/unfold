"""Block-level DAG: one node per IR block, sequential edges.

This is the *coarse* view (norm → attn → add → norm → ffn → add).  The
fine-grained tensor-level graphs live on each block under
``operation_graph``.
"""
from __future__ import annotations

from typing import Any

from .utils import drop_none


def build_block_graph(blocks: list[dict], group_path: str) -> dict[str, Any]:
    nodes = [
        _block_node(b, f"{group_path}.blocks[{i}]")
        for i, b in enumerate(blocks)
        if isinstance(b, dict)
    ]
    edges = []
    prev = "layer_input"
    for n in nodes:
        edges.append({"from": prev, "to": n["id"], "kind": "sequence"})
        prev = n["id"]
    edges.append({"from": prev, "to": "layer_output", "kind": "sequence"})
    return {"nodes": nodes, "edges": edges}


def _block_node(block: dict, path: str) -> dict[str, Any]:
    return drop_none({
        "id":          block.get("id"),
        "role":        block.get("role"),
        "kind":        block.get("kind"),
        "view": block.get("view"),
        "children": [
            _block_node(c, f"{path}.children[{i}]")
            for i, c in enumerate(block.get("children") or [])
            if isinstance(c, dict)
        ] or None,
        "trace": {"ir_path": path},
    })
