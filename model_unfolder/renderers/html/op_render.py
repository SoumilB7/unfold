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
    out_label: str | None = None,
) -> Graph:
    """Lay a region out as flow + parallels.

    ``clickable=True`` makes op nodes click-drill targets (used when the region's
    op ids are also inspect-card ids, as in the attention family); the default
    keeps ops static for leaf views like the FFN.

    The in/out anchors are always bare ports: a small mono ``in (4,608)``
    caption below, and a headless exit arrow above (captioned only when
    ``out_label`` says something, e.g. the MLA drills' "→ scores (Q)").
    """
    by_op = region.by_id()
    succ: dict[str, list[str]] = {}
    pred: dict[str, list[str]] = {}
    for e in region.edges:
        succ.setdefault(e.src, []).append(e.dst)
        pred.setdefault(e.dst, []).append(e.src)

    inputs = [o.id for o in region.ops if o.kind == "input"]
    # The primary input is identified by CANONICAL identity, not the raw id —
    # a namespaced region instance (``<ns>hidden``) keeps its original id in
    # meta["canonical_id"], so rename depth can never detach the spine root.
    primary = next(
        (o.id for o in region.ops
         if (o.meta or {}).get("canonical_id", o.id) == "hidden"),
        inputs[0] if inputs else region.ops[0].id,
    )
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
            # A direct edge from the branch point to a later merge is a real
            # skip lane with no operation node of its own (residual identity).
            # Treating the merge as a one-node output lane loses the arriving
            # edge and leaves a visually dangling ⊕.
            if is_merge(b):
                new_lanes.append(Lane([], dst=[b], src=cur))
                continue
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

    nodes = [_node_for(o, region, clickable, primary) for o in region.ops]

    # A presentation-only exit anchor: ``out_label=None`` leaves a bare arrow
    # (the port node keeps the geometry the arrow needs but paints no caption).
    nodes.append(Node("region_out", "port", out_label, static=True))
    flow = [*flow, "region_out"]

    return Graph(nodes=nodes, flow=flow, parallels=parallels)


def _lane_out_label(lane: Lane, by_op: dict[str, Op]) -> str | None:
    top = by_op.get(lane.ids[-1]) if lane.ids else None
    return (top.meta or {}).get("out_label") if top else None


def _node_for(op: Op, region: Region, clickable: bool, primary: str) -> Node:
    static = not clickable
    if op.kind == "input":
        if op.id == primary:
            label = f"in ({op.out_features:,})" if op.out_features else "in"
            return Node(op.id, "port", label, static=True, meta=dict(op.meta))
        # A secondary input (cross-attention's text / image states) is drawn as a
        # solid block like everything else — not the light accent bookend.
        return Node(op.id, "embedding", op.label, w=250, h=46, static=static)
    if op.kind == "linear":
        return Node(op.id, "linear", op.label or "Linear", static=static,
                    cache_ports=bool(op.meta.get("cached")))
    if op.kind == "activation":
        label = op.label if op.label is not None else activation_label(op.fn or "silu")
        return Node(op.id, "activation", label, static=static)
    if op.kind == "position":
        return Node(op.id, "embedding", op.label or "Position encoding", static=static)
    if op.kind == "elementwise":
        if op.fn not in {"mul", "add", "matmul"} and op.label:
            return Node(op.id, "activation", op.label, static=static)
        kind = {"mul": "gate_mul", "add": "residual_add"}.get(op.fn or "", "dot_product")
        return Node(op.id, kind, static=static)
    if op.kind == "attention_core":
        if op.fn == "scaled_dot_product":
            return Node(op.id, "formula", static=static, meta=dict(op.meta))
        return Node(op.id, "attention", op.label, static=static)
    if op.kind in ("concat", "reshape", "slice", "rope", "cache", "conv", "subgraph"):
        return Node(op.id, op.kind, op.label, static=static, meta=dict(op.meta))
    if op.kind == "route":
        return Node(op.id, "router", op.label or "Router")
    if op.kind == "opaque":
        # An unresolved opaque is the honest-unknown signal: pale + static (the config
        # does not declare the structure — nothing to drill or click). A resolved opaque
        # (a named custom block) follows the view's clickable flag.
        return Node(op.id, "opaque", op.label or "Custom block",
                    resolved=region.resolved,
                    static=True if region.resolved is False else static)
    return Node(op.id, "norm", op.label or op.kind, static=static)
