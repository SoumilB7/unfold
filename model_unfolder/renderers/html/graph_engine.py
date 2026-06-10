"""Lay out a :class:`~.graph.Graph` to SVG — the one engine every view uses.

Given a declarative graph (nodes + bottom→top flow + residual edges + repeat
groups + branch/merge parallels), this places the column, draws the flow arrows,
the residual side-loops, the dashed repeat-frame with its ``× N`` badge, and the
parallel branch-and-merge sections, then self-sizes the canvas.

Crucially the *concepts* live here once: a residual is always
:func:`~.svg._residual_loop_right`; a repeat is always the dashed frame + badge;
a gate∥up / Q∥K∥V branch is always a :class:`~.graph.Parallel`.  Views stop
re-implementing them (and stop mushing several ops into one labelled box, because
they declare typed nodes instead of drawing rectangles).
"""
from __future__ import annotations

from .graph import Graph, Parallel
from .stack_view import fit_svg, point
from .svg import (
    _branch_dot,
    _elbow_hv,
    _elbow_vh,
    _ids,
    _plus_block,
    _rect_block,
    _residual_loop_right,
    _svg_tag,
    _svg_text,
    _v_line,
)
from .theme import C, FONT_HEAD, FONT_MONO, GAP

_FLOW_GAP = 30.0          # vertical gap between consecutive flow nodes
_GROUP_PAD = 20.0         # padding between a repeat-frame and its members
_GROUP_HEADER = 30.0      # extra room above a group's top member for its badge
_LANE_GAP = 46.0          # offset of the residual lane past the widest node
_BRANCH_GAP = 36.0        # horizontal gap between parallel lanes
_INTRA_GAP = 24.0         # gap between stacked nodes inside one lane
_BRANCH_STUB = 34.0       # split-dot → lane-bottom rise
_MERGE_STUB = 34.0        # lane-top → merge-node rise


def render_graph(
    graph: Graph,
    info: dict,
    mount_id: str,
    view_key: str,
    title: str,
    *,
    min_width: int = 560,
    pad: int = 46,
) -> str:
    by_id = graph.by_id()
    arrow_id, shadow_id = _ids(mount_id, view_key)
    parts: list[str] = []
    regions: list[dict] = []
    cx = 0.0

    # --- 1. positions: walk the flow top -> bottom, assigning each node a top y.
    # A group's top member needs header room above it (for the badge); a parallel
    # section reserves the full height of its tallest lane between src and dst.
    order_bottom_up = [by_id[i] for i in graph.flow if i in by_id]
    top_members = {g.members[-1] for g in graph.groups if g.members}
    bottom_members = {g.members[0] for g in graph.groups if g.members}
    par_by_pair = {(p.src, p.dst): p for p in graph.parallels}
    par_height = {(p.src, p.dst): _parallel_height(p, by_id) for p in graph.parallels}

    order_top_down = list(reversed(order_bottom_up))
    geom: dict[str, dict] = {}
    y = 0.0
    for idx, node in enumerate(order_top_down):
        geom[node.id] = _geom(cx, y, node.width(), node.height())
        y += node.height()
        if idx + 1 < len(order_top_down):
            nxt = order_top_down[idx + 1]                  # the node below
            gap = _FLOW_GAP
            if (nxt.id, node.id) in par_height:            # node=dst (above), nxt=src (below)
                gap = par_height[(nxt.id, node.id)]
            else:
                if nxt.id in top_members:
                    gap += _GROUP_HEADER
                if node.id in bottom_members:
                    gap += 8
            y += gap

    max_right = max((g["right"] for g in geom.values()), default=cx)
    lane = max_right + _LANE_GAP

    # --- 2. repeat-frames (behind the nodes) ---
    for group in graph.groups:
        members = [geom[m] for m in group.members if m in geom]
        if not members:
            continue
        gx0 = min(m["left"] for m in members) - _GROUP_PAD
        gx1 = max(lane, max(m["right"] for m in members)) + _GROUP_PAD
        gy0 = min(m["top"] for m in members) - _GROUP_PAD - _GROUP_HEADER
        gy1 = max(m["bottom"] for m in members) + _GROUP_PAD
        parts.append(_svg_tag("rect", {
            "x": gx0, "y": gy0, "width": gx1 - gx0, "height": gy1 - gy0,
            "rx": 18, "ry": 18, "fill": C["bg_inner"], "opacity": 0.5,
            "stroke": C["block"], "stroke-width": 1.0, "stroke-dasharray": "5 4"}))
        regions += [point(gx0, gy0), point(gx1, gy1)]
        _badge(parts, gx0 + 12, gy0 + 5, group.badge())

    # --- 3. flow nodes ---
    for node in order_bottom_up:
        _draw_node(parts, info, shadow_id, node, geom[node.id])
        regions.append(geom[node.id])

    # --- 4. flow arrows (skip pairs a parallel section wires itself) ---
    for lower, upper in zip(order_bottom_up, order_bottom_up[1:]):
        if (lower.id, upper.id) in par_by_pair:
            continue
        parts.append(_v_line(geom[lower.id], geom[upper.id], arrow_id))

    # --- 5. parallel branch/merge sections ---
    for par in graph.parallels:
        _draw_parallel(parts, regions, info, shadow_id, arrow_id, par, by_id, geom, cx)

    # --- 6. residual side-loops ---
    for edge in graph.residuals():
        if edge.src in geom and edge.dst in geom:
            parts.append(_residual_loop_right(geom[edge.src], geom[edge.dst], lane, arrow_id))

    # --- 7. downstream note ---
    if graph.note:
        top = geom[order_bottom_up[-1].id]
        ny = top["top"] - 20
        parts.append(_svg_text(cx, ny, graph.note, {
            "text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11}))
        regions.append(point(cx, ny - 6))

    return fit_svg(arrow_id, shadow_id, parts, regions, title, min_width=min_width, pad=pad)


# ---------------------------------------------------------------------------

def _draw_node(parts, info, shadow_id, node, g) -> None:
    if node.glyph().shape == "circle":
        _plus_block(parts, info, shadow_id, node.data_id(), g["cx"], g["cy"],
                    sym=node.glyph().sym, clickable=not node.static)
    else:
        _rect_block(parts, info, shadow_id, node.data_id(), g["left"], g["top"],
                    g["w"], g["h"], node.heading(), font_size=node.font_size(),
                    resolved=node.resolved, sub=node.sub, accent=node.glyph().accent,
                    clickable=not node.static)


def _lane_height(lane_ids: list[str], by_id: dict) -> float:
    nodes = [by_id[i] for i in lane_ids if i in by_id]
    if not nodes:
        return 0.0
    return sum(n.height() for n in nodes) + _INTRA_GAP * (len(nodes) - 1)


def _parallel_height(par: Parallel, by_id: dict) -> float:
    tallest = max((_lane_height(l, by_id) for l in par.lanes), default=0.0)
    return tallest + _BRANCH_STUB + _MERGE_STUB


def _draw_parallel(parts, regions, info, shadow_id, arrow_id, par, by_id, geom, cx) -> None:
    """Split below ``dst`` into lanes that climb and merge back into ``dst``."""
    if par.src not in geom or par.dst not in geom:
        return
    src_g, dst_g = geom[par.src], geom[par.dst]
    split_y = src_g["top"] - 16
    parts.append(_branch_dot(cx, split_y))

    # Width-aware horizontal spread: lay lanes side by side (centred on cx) using
    # each lane's widest node, so 2 wide FFN columns and 4 narrow experts both fit.
    lane_w = [max((by_id[i].width() for i in ids if i in by_id), default=120.0)
              for ids in par.lanes]
    total = sum(lane_w) + _BRANCH_GAP * (len(lane_w) - 1)
    edge = cx - total / 2
    xs = []
    for w in lane_w:
        xs.append(edge + w / 2)
        edge += w + _BRANCH_GAP
    lane_bottom = src_g["top"] - _BRANCH_STUB        # bottom edge of each lane's first node

    for lane_x, lane_ids in zip(xs, par.lanes):
        nodes = [by_id[i] for i in lane_ids if i in by_id]
        if not nodes:
            continue
        # stack bottom -> top
        lane_geom = []
        edge_bottom = lane_bottom
        for node in nodes:
            g = _geom(lane_x, edge_bottom - node.height(), node.width(), node.height())
            lane_geom.append(g)
            _draw_node(parts, info, shadow_id, node, g)
            regions.append(g)
            edge_bottom = g["top"] - _INTRA_GAP
        # branch: split dot -> first (bottom) node
        parts.append(_elbow_hv(cx, split_y, lane_x, lane_geom[0]["bottom"] + GAP, arrow_id))
        # internal flow
        for lower, upper in zip(lane_geom, lane_geom[1:]):
            parts.append(_v_line(lower, upper, arrow_id))
        # merge: top node -> dst (into its near side)
        top_g = lane_geom[-1]
        if lane_x < cx:
            target_x = dst_g["cx"] - dst_g["w"] / 2 - GAP
        elif lane_x > cx:
            target_x = dst_g["cx"] + dst_g["w"] / 2 + GAP
        else:
            target_x = dst_g["cx"]
        parts.append(_elbow_vh(lane_x, top_g["top"], target_x, dst_g["cy"], arrow_id))


def _geom(cx: float, top: float, w: float, h: float) -> dict:
    x = cx - w / 2
    return {"left": x, "right": x + w, "top": top, "bottom": top + h,
            "cx": cx, "cy": top + h / 2, "w": w, "h": h}


def _badge(parts: list[str], x: float, y: float, text: str) -> None:
    w = max(120.0, 12.0 + 7.4 * len(text))
    parts.append(_svg_tag("rect", {
        "x": x, "y": y, "width": w, "height": 26, "rx": 13, "ry": 13,
        "fill": C["bg_outer"], "stroke": C["border"], "stroke-width": 0.7}))
    parts.append(_svg_text(x + w / 2, y + 13, text, {
        "text-anchor": "middle", "dominant-baseline": "central",
        "fill": C["text"], "font-family": FONT_HEAD, "font-size": 15}))
