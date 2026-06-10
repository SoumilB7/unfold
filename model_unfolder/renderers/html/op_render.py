"""Project a canonical :class:`~...opgraph.Region` onto a render :class:`~.graph.Graph`.

This is the *render* projection of the one structural op-graph: it maps each
primitive op to a glyph and derives the column + branch/merge layout **from the
region's edges** — one rule (a node with ≥2 inputs is a merge; its input chains
become parallel lanes), so gated MLP, dense MLP, and an opaque custom block all
flow through unchanged with no per-family branching.
"""
from __future__ import annotations

from ...labels import activation_label
from ...opgraph import Op, Region
from .graph import Graph, Node, Parallel


def region_to_graph(region: Region) -> Graph:
    by_id = region.by_id()
    nodes = [_node_for(o, region) for o in region.ops]

    merges = region.merges()
    src = next((o.id for o in region.ops if o.kind == "input"), region.ops[0].id)

    if not merges:
        flow = _topo_chain(region, src)
        parallels: list[Parallel] = []
    else:
        merge = merges[0]                       # FFN-family regions have one merge
        lanes = [_lane(region, inp, src) for inp in region.inputs_of(merge)]
        flow = [src, merge, *_forward_chain(region, merge)]
        parallels = [Parallel(src, merge, lanes)]

    # A presentation-only output bookend so the block reads as in→…→out.
    nodes.append(Node("region_out", "output", "→ residual", static=True))
    flow = [*flow, "region_out"]

    return Graph(nodes=nodes, flow=flow, parallels=parallels)


def _node_for(op: Op, region: Region) -> Node:
    dims = (f"{op.in_features:,} → {op.out_features:,}"
            if (op.in_features and op.out_features) else None)
    if op.kind == "input":
        return Node(op.id, "source", "Hidden states",
                    sub=(f"{op.out_features:,}-d" if op.out_features else None), static=True)
    if op.kind == "linear":
        return Node(op.id, "linear", op.label or "Linear", sub=dims, static=True)
    if op.kind == "activation":
        return Node(op.id, "activation", activation_label(op.fn or "silu"), static=True)
    if op.kind == "elementwise":
        return Node(op.id, "residual_add" if op.fn == "add" else "gate_mul", static=True)
    if op.kind == "route":
        return Node(op.id, "router", op.label or "Router")
    if op.kind == "opaque":
        return Node(op.id, "opaque", op.label or "Custom block", sub=dims,
                    resolved=region.resolved, static=region.resolved is False)
    return Node(op.id, "norm", op.label or op.kind, static=True)


def _lane(region: Region, start: str, src: str) -> list[str]:
    """Trace one merge input back toward ``src``; return it bottom→top."""
    chain = [start]
    cur = start
    while True:
        preds = region.inputs_of(cur)
        if len(preds) == 1 and preds[0] != src:
            cur = preds[0]
            chain.append(cur)
        else:
            break
    return list(reversed(chain))


def _forward_chain(region: Region, start: str) -> list[str]:
    """Single-successor chain after ``start`` (e.g. merge → down_proj)."""
    out: list[str] = []
    cur = start
    while True:
        succ = [e.dst for e in region.edges if e.src == cur]
        if len(succ) == 1:
            cur = succ[0]
            out.append(cur)
        else:
            break
    return out


def _topo_chain(region: Region, src: str) -> list[str]:
    """The full single-path chain from ``src`` (used when there is no merge)."""
    return [src, *_forward_chain(region, src)]
