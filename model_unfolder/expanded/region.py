"""Project a canonical :class:`~..opgraph.Region` onto the JSON node/edge schema.

The one structural op-graph (``opgraph``) is the only author of a block's
internals; this module is the *JSON* projection of it, shared by ``ffn.py`` and
``attention.py``.  ``rename`` lets a projection keep its published node names
(the schema's ``scores``/``softmax``/``context``) while the region keeps the
single id set that also couples render nodes to inspect cards.
"""
from __future__ import annotations

from typing import Any

from ..opgraph import Region
from .ops import linear, node

#: op kind -> JSON ``operation`` name (kinds whose name isn't the kind itself).
_JSON_OP = {
    "route": "top_k_router",
    "elementwise": "elementwise",
}

#: attention_core ``fn`` values pass through as the operation name directly.


def region_to_json(region: Region, *, rename: dict[str, str] | None = None) -> dict[str, Any]:
    """Project the canonical op-graph onto the JSON node/edge schema."""
    rename = rename or {}

    def rid(op_id: str) -> str:
        return rename.get(op_id, op_id)

    nodes: list[dict[str, Any]] = []
    nested_done = set()
    for op in region.ops:
        operation = _operation_name(op)
        fields: dict[str, Any] = {}
        if op.kind == "linear":
            fields["parameters"] = linear(op.in_features, op.out_features)
        elif op.kind == "activation":
            fields["function"] = op.fn
        elif op.kind == "input":
            fields["width"] = op.out_features
        elif op.kind == "elementwise":
            operation = ("elementwise_multiply" if op.fn == "mul"
                         else "weighted_sum" if op.fn == "add"
                         else "matmul" if op.fn == "matmul"
                         else "elementwise")
        elif op.kind == "attention_core":
            if op.meta.get("formula"):
                fields["formula"] = op.meta["formula"]
        elif op.kind == "cache":
            fields["stores"] = op.meta.get("stores")
        elif op.kind == "subgraph":
            graph = op.meta.get("region")
            if isinstance(graph, Region):
                fields["graph"] = region_to_json(graph)
                nested_done.add(op.id)
        elif op.kind == "opaque":
            fields["class_name"] = op.meta.get("class_name")
        inputs = [rid(i) for i in region.inputs_of(op.id)]
        nodes.append(node(rid(op.id), operation, inputs=inputs or None,
                          outputs=[rid(op.id)], **fields))
    edges = [{"from": rid(e.src), "to": rid(e.dst)} for e in region.edges]
    return {"nodes": nodes, "edges": edges}


def _operation_name(op) -> str:
    if op.kind == "attention_core":
        return op.fn or "attention_core"
    return _JSON_OP.get(op.kind, op.kind)
