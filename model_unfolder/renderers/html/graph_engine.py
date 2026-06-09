"""Lay out a :class:`~.graph.Graph` to SVG — the one engine every view uses.

Given a declarative graph (nodes + bottom→top flow + residual edges + repeat
groups), this places the column, draws the flow arrows, the residual side-loops,
and the dashed repeat-frame with its ``× N`` badge, then self-sizes the canvas.

Crucially the *concepts* live here once: a residual is always
:func:`~.svg._residual_loop_right`; a repeat is always the dashed frame + badge
below.  Views stop re-implementing them (and stop mushing several ops into one
labelled box, because they declare typed nodes instead of drawing rectangles).
"""
from __future__ import annotations

from .graph import Graph, Node
from .stack_view import fit_svg, point
from .svg import _ids, _plus_block, _rect_block, _residual_loop_right, _svg_tag, _svg_text, _v_line
from .theme import C, FONT_HEAD, FONT_MONO

_FLOW_GAP = 30.0          # vertical gap between consecutive flow nodes
_GROUP_PAD = 20.0         # padding between a repeat-frame and its members
_GROUP_HEADER = 30.0      # extra room above a group's top member for its badge
_LANE_GAP = 46.0          # offset of the residual lane past the widest node


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
    # A group's top member needs header room above it (for the badge) and its
    # bottom member a little pad below, so the dashed frame never collides with
    # neighbouring nodes.  Top/bottom members are the last/first member in flow.
    order_bottom_up = [by_id[i] for i in graph.flow if i in by_id]
    top_members = {g.members[-1] for g in graph.groups if g.members}
    bottom_members = {g.members[0] for g in graph.groups if g.members}
    order_top_down = list(reversed(order_bottom_up))
    geom: dict[str, dict] = {}
    y = 0.0
    for idx, node in enumerate(order_top_down):
        geom[node.id] = _geom(cx, y, node.width(), node.height())
        y += node.height()
        if idx + 1 < len(order_top_down):
            nxt = order_top_down[idx + 1]
            gap = _FLOW_GAP
            if nxt.id in top_members:
                gap += _GROUP_HEADER
            if node.id in bottom_members:
                gap += 8
            y += gap

    # residual lane sits just past the widest node on the right
    max_right = max((g["right"] for g in geom.values()), default=cx)
    lane = max_right + _LANE_GAP

    # --- 2. repeat-frames (drawn first, behind the nodes) ---
    for group in graph.groups:
        members = [geom[m] for m in group.members if m in geom]
        if not members:
            continue
        gx0 = min(m["left"] for m in members) - _GROUP_PAD
        gx1 = max(lane, max(m["right"] for m in members)) + _GROUP_PAD
        gy0 = min(m["top"] for m in members) - _GROUP_PAD - _GROUP_HEADER   # header band for badge
        gy1 = max(m["bottom"] for m in members) + _GROUP_PAD
        parts.append(_svg_tag("rect", {
            "x": gx0, "y": gy0, "width": gx1 - gx0, "height": gy1 - gy0,
            "rx": 18, "ry": 18, "fill": C["bg_inner"], "opacity": 0.5,
            "stroke": C["block"], "stroke-width": 1.0, "stroke-dasharray": "5 4"}))
        regions += [point(gx0, gy0), point(gx1, gy1)]
        _badge(parts, gx0 + 12, gy0 + 5, group.badge())

    # --- 3. nodes ---
    for node in order_bottom_up:
        g = geom[node.id]
        if node.glyph().shape == "circle":
            _plus_block(parts, info, shadow_id, node.data_id(), g["cx"], g["cy"], sym=node.glyph().sym)
        else:
            _rect_block(parts, info, shadow_id, node.data_id(), g["left"], g["top"],
                        g["w"], g["h"], node.heading(), font_size=node.glyph().font,
                        resolved=node.resolved, sub=node.sub, accent=node.glyph().accent,
                        clickable=not node.static)
        regions.append(g)

    # --- 4. flow arrows (consecutive nodes in flow order) ---
    for lower, upper in zip(order_bottom_up, order_bottom_up[1:]):
        parts.append(_v_line(geom[lower.id], geom[upper.id], arrow_id))

    # --- 5. residual side-loops: tap below ``src`` (raw x), add at ``dst`` ⊕ ---
    for edge in graph.residuals():
        if edge.src in geom and edge.dst in geom:
            parts.append(_residual_loop_right(geom[edge.src], geom[edge.dst], lane, arrow_id))

    # --- 6. downstream note above the top node ---
    if graph.note:
        top = geom[order_bottom_up[-1].id]
        ny = top["top"] - 20
        parts.append(_svg_text(cx, ny, graph.note, {
            "text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11}))
        regions.append(point(cx, ny - 6))

    return fit_svg(arrow_id, shadow_id, parts, regions, title, min_width=min_width, pad=pad)


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
