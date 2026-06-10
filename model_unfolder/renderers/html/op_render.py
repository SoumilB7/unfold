"""Project a canonical :class:`~...opgraph.Region` onto a render :class:`~.graph.Graph`.

This is the *render* projection of the one structural op-graph: it maps each
primitive op to a glyph and derives the column + branch/merge layout **from the
region's edges** — one set of rules, no per-family branching:

* the **spine** is the chain from the primary input; at a fan-out, branches that
  are pure chains become parallel *lanes* and the spine jumps to their join (the
  first node ≥2 lanes feed); a branch with internal structure *is* the spine;
* a lane's merge target(s) are wherever its top node's edges point — the common
  join (FFN's ⊗), a node further up the spine (attention's V into ⊙), several
  nodes (MLA's KV path into scores *and* ⊙), or nothing (a labelled output);
* a lane fed by a secondary ``input`` op gets that op as a side source block
  (cross-attention's projected image states), and a lane that taps a lower spine
  node branches from a dot on that node's stem (MLA's RoPE key side-channel).

So a gated MLP, a custom opaque block, and every attention family flow through
unchanged — the variety lives in the region, not here.
"""
from __future__ import annotations

from ...labels import activation_label
from ...opgraph import Op, Region
from .graph import Graph, Lane, Node, Parallel


def region_to_graph(
    region: Region,
    *,
    clickable: bool = False,
    out_label: str | None = "→ residual",
    ports: bool = False,
) -> Graph:
    """Lay a region out as flow + parallels.

    ``clickable=True`` makes op nodes click-drill targets (used when the region's
    op ids are also inspect-card ids, as in the attention family); the default
    keeps ops static for leaf views like the FFN.

    ``ports=True`` renders the in/out anchors as bare mono captions on the flow
    stem ("in (4,608)" / "→ residual") instead of full source/output blocks —
    for drill views where those blocks would only restate the obvious.
    """
    by_op = region.by_id()
    succ: dict[str, list[str]] = {}
    pred: dict[str, list[str]] = {}
    for e in region.edges:
        succ.setdefault(e.src, []).append(e.dst)
        pred.setdefault(e.dst, []).append(e.src)

    inputs = [o.id for o in region.ops if o.kind == "input"]
    primary = "hidden" if "hidden" in by_op else (inputs[0] if inputs else region.ops[0].id)
    op_order = {o.id: i for i, o in enumerate(region.ops)}

    def is_merge(n: str) -> bool:
        return len(pred.get(n, [])) >= 2

    def chase(start: str) -> tuple[list[str], list[str], bool]:
        """Follow a pure chain from ``start``: (ids, merge targets, pure?).

        ``pure=False`` means the branch fans out into non-merge nodes — it is
        spine material, and ``ids`` end at the fan-out node.
        """
        ids = [start]
        cur = start
        while True:
            nxt = succ.get(cur, [])
            if not nxt:
                return ids, [], True                       # output lane
            if len(nxt) == 1:
                t = nxt[0]
                if is_merge(t) or len(pred.get(t, [])) > 1:
                    return ids, [t], True
                ids.append(t)
                cur = t
                continue
            if all(is_merge(t) for t in nxt):              # multi-target lane
                return ids, sorted(nxt, key=lambda n: op_order.get(n, 0)), True
            return ids, nxt, False

    # Lanes fed by secondary inputs (side sources) wait at the first branch point.
    pending: list[Lane] = []
    for ext in inputs:
        if ext == primary:
            continue
        for s in succ.get(ext, []):
            ids, dsts, _pure = chase(s)
            pending.append(Lane(ids, dst=dsts, src=ext))

    flow = [primary]
    parallels: list[Parallel] = []
    cur = primary

    while True:
        branches = succ.get(cur, [])
        if not branches:
            break
        if len(branches) == 1 and not pending:
            flow.append(branches[0])
            cur = branches[0]
            continue

        # A branch point (or pending lanes that must join): classify branches.
        new_lanes: list[Lane] = []
        spine_ids: list[str] | None = None
        for b in branches:
            ids, dsts, pure = chase(b)
            if pure:
                new_lanes.append(Lane(ids, dst=dsts))
            elif spine_ids is None:
                spine_ids = ids
            else:                                          # defensive: extra structure
                new_lanes.append(Lane(ids, dst=[t for t in dsts if is_merge(t)]))

        if spine_ids is not None:
            # Structured branch is the spine; pure branches wait for their join.
            for lane in new_lanes:
                lane.src = cur
            pending.extend(new_lanes)
            flow.extend(spine_ids)
            cur = spine_ids[-1]
            continue

        # All branches are pure chains: resolve the join across new + pending.
        lanes = new_lanes + pending
        pending = []
        counts: dict[str, int] = {}
        for lane in lanes:
            for d in lane.dst or []:
                counts[d] = counts.get(d, 0) + 1
        join_candidates = sorted(
            (d for d, c in counts.items() if c >= 2),
            key=lambda n: op_order.get(n, 0),
        )
        join = join_candidates[0] if join_candidates else next(
            (d for lane in lanes for d in (lane.dst or [])), None)
        if join is None:
            break                                          # nothing converges
        for lane in lanes:
            if lane.dst == [join]:
                lane.dst = None                            # the common case
            if lane.dst == []:
                lane.out_label = _lane_out_label(lane, by_op)
        # Slot order decides left→right placement: spine-tap lanes take the
        # left edge (their elbow climbs from a lower spine node and would cut
        # straight through the spine if placed centre), shared/side-source
        # lanes fill the middle, output lanes exit on the right.
        input_ids = {o.id for o in region.ops if o.kind == "input"}
        def _slot(lane: Lane) -> int:
            if lane.dst is not None and not lane.dst:
                return 2                                   # output lane → right
            if lane.src is not None and lane.src not in input_ids:
                return 0                                   # spine tap → left
            return 1
        lanes.sort(key=_slot)
        parallels.append(Parallel(cur, join, lanes))
        flow.append(join)
        cur = join

    nodes = [_node_for(o, region, clickable, primary, ports) for o in region.ops]

    # A presentation-only output anchor so the block reads as in→…→out.
    # With ports, ``out_label=None`` leaves a bare exit arrow (the port node
    # keeps the geometry the arrow needs but paints no caption).
    nodes.append(Node("region_out", "port" if ports else "output",
                      out_label if ports else (out_label or "Output"), static=True))
    flow = [*flow, "region_out"]

    return Graph(nodes=nodes, flow=flow, parallels=parallels)


def _lane_out_label(lane: Lane, by_op: dict[str, Op]) -> str | None:
    top = by_op.get(lane.ids[-1]) if lane.ids else None
    return (top.meta or {}).get("out_label") if top else None


def _node_for(op: Op, region: Region, clickable: bool, primary: str, ports: bool) -> Node:
    dims = (f"{op.in_features:,} → {op.out_features:,}"
            if (op.in_features and op.out_features) else None)
    static = not clickable
    if op.kind == "input":
        if op.id == primary:
            if ports:
                label = f"in ({op.out_features:,})" if op.out_features else "in"
                return Node(op.id, "port", label, static=True)
            return Node(op.id, "source", op.label or "Hidden states",
                        sub=(f"{op.out_features:,}-d" if op.out_features else None),
                        static=True)
        return Node(op.id, "source", op.label, w=250, h=46, static=static)
    if op.kind == "linear":
        return Node(op.id, "linear", op.label or "Linear", sub=dims, static=static,
                    cache_ports=bool(op.meta.get("cached")))
    if op.kind == "activation":
        label = op.label if op.label is not None else activation_label(op.fn or "silu")
        return Node(op.id, "activation", label, static=static)
    if op.kind == "elementwise":
        kind = {"mul": "gate_mul", "add": "residual_add"}.get(op.fn or "", "dot_product")
        return Node(op.id, kind, static=static)
    if op.kind == "attention_core":
        if op.fn == "scaled_dot_product":
            return Node(op.id, "formula", static=static, meta=dict(op.meta))
        return Node(op.id, "attention", op.label, static=static)
    if op.kind in ("concat", "slice", "rope", "cache", "conv", "subgraph"):
        return Node(op.id, op.kind, op.label, static=static, meta=dict(op.meta))
    if op.kind == "route":
        return Node(op.id, "router", op.label or "Router")
    if op.kind == "opaque":
        return Node(op.id, "opaque", op.label or "Custom block", sub=dims,
                    resolved=region.resolved, static=region.resolved is False)
    return Node(op.id, "norm", op.label or op.kind, static=static)
