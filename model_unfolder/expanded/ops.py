"""Shared building blocks for the operation graphs.

These three helpers are the *only* primitives both ``attention.py`` and
``ffn.py`` use to emit nodes/edges/linear-parameter dicts.  Keeping them
here makes the schema for a graph node uniform across blocks.
"""
from __future__ import annotations

from typing import Any

from .utils import drop_none, shape


def node(node_id: str, operation: str, **fields) -> dict[str, Any]:
    """One node in an operation graph: ``{id, operation, …optional fields}``."""
    return drop_none({"id": node_id, "operation": operation, **fields})


def linear(in_features: int | None, out_features: int | None) -> dict[str, Any] | None:
    """Linear-layer parameter dict: ``{in, out, weight_shape}``."""
    if in_features is None and out_features is None:
        return None
    return drop_none({
        "in_features":  in_features,
        "out_features": out_features,
        "weight_shape": shape(out_features, in_features),
    })


def edges_from_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Derive edges purely from node ``inputs``/``outputs`` tensor names."""
    producers: dict[str, str] = {}
    for n in nodes:
        for output in n.get("outputs") or []:
            producers[output] = n["id"]
    edges: list[dict[str, str]] = []
    for n in nodes:
        for input_name in n.get("inputs") or []:
            if input_name in producers:
                edges.append({"from": producers[input_name], "to": n["id"], "tensor": input_name})
    return edges
