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
    _cache_io_ports,
    _elbow_hv,
    _elbow_vh,
    _formula_block,
    _ids,
    _merge_up_route,
    _plus_block,
    _rect_block,
    _residual_loop_right,
    _svg_tag,
    _svg_text,
    _v_line,
    _v_seg,
    _window_strip,
)
from .theme import C, FONT_HEAD, FONT_MONO, GAP

_FLOW_GAP = 30.0          # vertical gap between consecutive flow nodes
_GROUP_PAD = 26.0         # padding between a repeat-frame and its members
_GROUP_HEADER = 44.0      # extra room above a group's top member for its badge
_LANE_GAP = 46.0          # offset of the residual lane past the widest node
_BRANCH_GAP = 36.0        # horizontal gap between parallel lanes
_INTRA_GAP = 24.0         # gap between stacked nodes inside one lane
_BRANCH_STUB = 48.0       # split-dot → lane-bottom rise
_MERGE_STUB = 46.0        # lane-top → merge-node rise


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
    flow_set = set(graph.flow)
    par_height = {(p.src, p.dst): _parallel_height(p, by_id, flow_set) for p in graph.parallels}
    # nodes whose outgoing stem carries a side-lane tap dot need a longer stem,
    # or the dot crowds the arrowhead entering the node above
    tap_srcs = {lane.src for p in graph.parallels for lane in p.norm_lanes()
                if lane.src is not None and lane.src in flow_set}

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
                    gap += 30                              # air between frame edge and the block below
                if nxt.id in tap_srcs:
                    gap += 20                              # room for the tap dot on the stem
            y += gap

    max_right = max((g["right"] for g in geom.values()), default=cx)
    lane = max_right + _LANE_GAP

    # Residual loops get a *tight* lane each: just past the widest node in the
    # span they bypass, not past the whole graph — so a wide source/output
    # block elsewhere can't push every loop (and the repeat-frame with it)
    # far out to the right.
    flow_idx = {nid: i for i, nid in enumerate(graph.flow)}
    res_lane: dict[tuple[str, str], float] = {}
    for e in graph.residuals():
        if e.src in geom and e.dst in geom:
            i0, i1 = sorted((flow_idx.get(e.src, 0), flow_idx.get(e.dst, 0)))
            span = [geom[n] for n in graph.flow[i0:i1 + 1] if n in geom]
            res_lane[(e.src, e.dst)] = max(g["right"] for g in span) + _LANE_GAP

    # --- 2. repeat-frames (behind the nodes) ---
    for group in graph.groups:
        members = [geom[m] for m in group.members if m in geom]
        if not members:
            continue
        member_set = set(group.members)
        loop_lanes = [x for (s, d), x in res_lane.items()
                      if s in member_set and d in member_set]
        gx0 = min(m["left"] for m in members) - _GROUP_PAD
        gx1 = max([*loop_lanes, max(m["right"] for m in members)]) + _GROUP_PAD
        # symmetric about the column so the cell reads centred, not leaning
        half = max(cx - gx0, gx1 - cx)
        gx0, gx1 = cx - half, cx + half
        gy0 = min(m["top"] for m in members) - _GROUP_PAD - _GROUP_HEADER
        gy1 = max(m["bottom"] for m in members) + _GROUP_PAD
        # a residual that taps the bottom member starts on its input stem,
        # below the block — the frame must cover the tap, not clip the loop
        for (s, _d) in res_lane:
            if s in member_set and s in geom:
                gy1 = max(gy1, geom[s]["bottom"] + GAP + 16 + 22)
        # the same solid cell frame the main architecture view draws
        parts.append(_svg_tag("rect", {
            "x": gx0, "y": gy0, "width": gx1 - gx0, "height": gy1 - gy0,
            "rx": 18, "ry": 18, "fill": C["bg_inner"], "stroke": "none"}))
        regions += [point(gx0, gy0), point(gx1, gy1)]
        _badge(parts, gx1, gy0 + 12, group.badge())

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

    # --- 6. residual side-loops (each on its own tight lane) ---
    for edge in graph.residuals():
        if edge.src in geom and edge.dst in geom:
            loop_lane = res_lane.get((edge.src, edge.dst), lane)
            parts.append(_residual_loop_right(geom[edge.src], geom[edge.dst], loop_lane, arrow_id))

    # --- 7. downstream note ---
    if graph.note:
        top = geom[order_bottom_up[-1].id]
        ny = top["top"] - 20
        parts.append(_svg_text(cx, ny, graph.note, {
            "text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11}))
        regions.append(point(cx, ny - 6))

    # --- 8. aside fact panel (heading + mono rows) ---
    # Tucked beside the rows it actually overlaps (the upper spine is narrow),
    # not past the graph's full width — so it fills the empty corner instead of
    # widening the canvas and shrinking the diagram.
    if graph.aside:
        aside_w, aside_h = _aside_size(graph.aside)
        aside_top = min((g["top"] for g in geom.values()), default=0.0)
        band_right = max(
            (r["right"] for r in regions
             if r.get("bottom", 0) > aside_top and r.get("top", 0) < aside_top + aside_h),
            default=cx,
        )
        if graph.residuals():
            band_right = max(band_right, lane)
        regions.append(_draw_aside(parts, band_right + 18, aside_top, graph.aside, aside_w, aside_h))

    return fit_svg(arrow_id, shadow_id, parts, regions, title, min_width=min_width, pad=pad)


# ---------------------------------------------------------------------------

def _draw_node(parts, info, shadow_id, node, g) -> None:
    shape = node.glyph().shape
    if shape == "circle":
        _plus_block(parts, info, shadow_id, node.data_id(), g["cx"], g["cy"],
                    sym=node.glyph().sym, clickable=not node.static)
    elif shape == "formula":
        _formula_block(parts, info, shadow_id, node.data_id(), g["left"], g["top"],
                       g["w"], g["h"],
                       numerator=node.meta.get("numerator", "Q K^T"),
                       denominator=node.meta.get("denominator", "sqrt(dim)"),
                       clickable=not node.static)
    elif shape == "window":
        _window_strip(parts, g["left"], g["top"], g["w"], g["h"],
                      node.meta.get("window_size"))
    elif shape == "port":
        heading = node.heading()
        text = heading if isinstance(heading, str) else " ".join(heading)
        if text:                       # an unlabelled port is a bare exit arrow
            parts.append(_svg_text(g["cx"], g["cy"], text, {
                "text-anchor": "middle", "dominant-baseline": "central",
                "fill": C["muted"], "font-family": FONT_MONO,
                "font-size": node.font_size()}))
    else:
        _rect_block(parts, info, shadow_id, node.data_id(), g["left"], g["top"],
                    g["w"], g["h"], node.heading(), font_size=node.font_size(),
                    resolved=node.resolved, sub=node.sub, accent=node.glyph().accent,
                    clickable=not node.static)
        if node.kind == "cache" or node.cache_ports:
            # one convention everywhere: the port pair sits bottom-right
            _cache_io_ports(parts, g["left"], g["top"], g["w"], g["h"])


def _lane_height(lane_ids: list[str], by_id: dict) -> float:
    nodes = [by_id[i] for i in lane_ids if i in by_id]
    if not nodes:
        return 0.0
    return sum(n.height() for n in nodes) + _INTRA_GAP * (len(nodes) - 1)


def _parallel_height(par: Parallel, by_id: dict, flow: set | None = None) -> float:
    tallest = max((_lane_height(l.ids, by_id) for l in par.norm_lanes()), default=0.0)
    return tallest + _BRANCH_STUB + _MERGE_STUB + _ext_source_extra(par, by_id, flow or set())


def _ext_source_extra(par: Parallel, by_id: dict, flow: set) -> float:
    """Extra vertical band for off-flow side sources drawn below the lanes."""
    ext = [by_id[l.src] for l in par.norm_lanes()
           if l.src is not None and l.src not in flow and l.src in by_id]
    return (max(n.height() for n in ext) + 40.0) if ext else 0.0


def _draw_parallel(parts, regions, info, shadow_id, arrow_id, par, by_id, geom, cx) -> None:
    """Split below ``dst`` into lanes that climb and merge back into ``dst``.

    Lanes may also tap a *named* source (a lower spine node, or an off-flow side
    block like cross-attention's image states), merge into spine nodes *above*
    ``dst`` (attention's V joining at ⊙), or exit upward as labelled outputs.
    """
    if par.src not in geom or par.dst not in geom:
        return
    src_g, dst_g = geom[par.src], geom[par.dst]
    lanes = par.norm_lanes()
    split_y = src_g["top"] - 16
    if any(lane.src is None for lane in lanes):
        # the stem that carries the source up into the split dot — without it
        # the fan-out floats disconnected above its source
        parts.append(_svg_tag("line", {
            "x1": cx, "y1": src_g["top"], "x2": cx, "y2": split_y,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "fill": "none"}))
        parts.append(_branch_dot(cx, split_y))

    # Width-aware horizontal spread: lay lanes side by side (centred on cx) using
    # each lane's widest node, so 2 wide FFN columns and 4 narrow experts both fit.
    lane_w = [max((by_id[i].width() for i in lane.ids if i in by_id), default=120.0)
              for lane in lanes]
    total = sum(lane_w) + _BRANCH_GAP * (len(lane_w) - 1)
    edge = cx - total / 2
    xs = []
    for w in lane_w:
        xs.append(edge + w / 2)
        edge += w + _BRANCH_GAP
    # Off-flow side sources get their own band between ``src`` and the lanes.
    ext_extra = _ext_source_extra(par, by_id, set(geom))
    lane_bottom = src_g["top"] - _BRANCH_STUB - ext_extra   # lanes' first-node bottom edge

    lane_geoms: list[list[dict]] = []
    for lane_x, lane in zip(xs, lanes):
        nodes = [by_id[i] for i in lane.ids if i in by_id]
        if not nodes:
            lane_geoms.append([])
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
        lane_geoms.append(lane_geom)

        # branch: this lane's source -> first (bottom) node
        if lane.src is None:
            parts.append(_elbow_hv(cx, split_y, lane_x, lane_geom[0]["bottom"] + GAP, arrow_id))
        elif lane.src in geom:
            # tap a lower spine node: dot on its outgoing stem, elbow to the lane
            tap_g = geom[lane.src]
            tap_y = tap_g["top"] - 16
            parts.append(_branch_dot(tap_g["cx"], tap_y))
            parts.append(_elbow_hv(tap_g["cx"], tap_y, lane_x, lane_geom[0]["bottom"] + GAP, arrow_id))
        # off-flow side sources are drawn once for all their lanes below

        # internal flow
        for lower, upper in zip(lane_geom, lane_geom[1:]):
            parts.append(_v_line(lower, upper, arrow_id))

        # merge / output: top node -> each target (or out of the diagram)
        top_g = lane_geom[-1]
        if lane.dst is not None and not lane.dst:
            parts.append(_v_seg(lane_x, top_g["top"], top_g["top"] - 30, arrow_id))
            if lane.out_label:
                parts.append(_svg_text(lane_x, top_g["top"] - 42, lane.out_label, {
                    "text-anchor": "middle", "fill": C["muted"],
                    "font-family": FONT_MONO, "font-size": 11}))
            regions.append(point(lane_x, top_g["top"] - 50))
            continue
        for dst_id in (lane.dst or [par.dst]):
            d_g = geom.get(dst_id)
            if d_g is None:
                continue
            d_node = by_id.get(dst_id)
            rect_dst = d_node is not None and d_node.glyph().shape != "circle"
            if dst_id == par.dst and rect_dst:
                # Merge into the bottom edge at an inset entry point — never
                # poke a rect's side at mid-height.  The horizontal run stays in
                # the merge-stub band just below ``dst``, ABOVE every lane's top,
                # so it can't cut through a taller neighbouring lane.
                entry_x = (d_g["cx"] if d_g["w"] < 60
                           else min(max(lane_x, d_g["left"] + 26), d_g["right"] - 26))
                entry_y = d_g["bottom"] + GAP
                lane_y = min(d_g["bottom"] + 26, (top_g["top"] + entry_y) / 2)
                parts.append(_merge_up_route(lane_x, top_g["top"], entry_x, entry_y, lane_y, arrow_id))
            else:
                # A circle (⊕/⊗/⊙) or a target further up the spine: enter the
                # nearest side at its centre height.
                if lane_x < d_g["left"]:
                    target_x = d_g["left"] - GAP
                elif lane_x > d_g["right"]:
                    target_x = d_g["right"] + GAP
                else:
                    target_x = d_g["cx"]
                parts.append(_elbow_vh(lane_x, top_g["top"], target_x, d_g["cy"], arrow_id))

    _draw_side_sources(parts, regions, info, shadow_id, arrow_id,
                       lanes, xs, lane_geoms, by_id, geom, lane_bottom)


def _draw_side_sources(parts, regions, info, shadow_id, arrow_id,
                       lanes, xs, lane_geoms, by_id, geom, lane_bottom) -> None:
    """Draw each off-flow lane source once, centred under the lanes it feeds,
    in the reserved band between the spine source and the lane bottoms."""
    by_src: dict[str, list[int]] = {}
    for i, lane in enumerate(lanes):
        if lane.src is not None and lane.src not in geom and lane.src in by_id:
            by_src.setdefault(lane.src, []).append(i)
    for src_id, lane_idx in by_src.items():
        node = by_id[src_id]
        block_cx = sum(xs[i] for i in lane_idx) / len(lane_idx)
        g = _geom(block_cx, lane_bottom + 34, node.width(), node.height())
        _draw_node(parts, info, shadow_id, node, g)
        regions.append(g)
        dot_y = g["top"] - 12
        parts.append(_svg_tag("line", {
            "x1": g["cx"], "y1": g["top"], "x2": g["cx"], "y2": dot_y,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "fill": "none"}))
        parts.append(_branch_dot(g["cx"], dot_y))
        for i in lane_idx:
            if lane_geoms[i]:
                parts.append(_elbow_hv(g["cx"], dot_y, xs[i],
                                       lane_geoms[i][0]["bottom"] + GAP, arrow_id))


_ASIDE_PAD = 16.0
_ASIDE_CHIP_H = 30.0
_ASIDE_CHIP_GAP = 7.0
_ASIDE_DOTS_H = 14.0


def _aside_size(aside: dict) -> tuple[float, float]:
    rows = aside.get("rows") or []
    footer = aside.get("footer") or []
    widths = [10.0 + 7.4 * len(aside.get("title", ""))]
    widths += [44.0 + 6.4 * (len(r[0]) + len(r[1])) for r in rows if r != "..."]
    widths += [6.4 * len(t) for t in footer]
    width = max(176.0, max(widths, default=0.0) + 2 * _ASIDE_PAD)
    rows_h = sum(_ASIDE_DOTS_H if r == "..." else _ASIDE_CHIP_H + _ASIDE_CHIP_GAP for r in rows)
    footer_h = (10.0 + 16.0 * len(footer)) if footer else 0.0
    return width, _ASIDE_PAD + 24.0 + rows_h + footer_h + _ASIDE_PAD - (
        _ASIDE_CHIP_GAP if rows and rows[-1] != "..." else 0.0)


def _draw_aside(parts: list[str], x: float, y: float, aside: dict,
                width: float, height: float) -> dict:
    """The side fact card: a heading, one badge chip per mapping row
    (strong text left, detail right), and a divided footer for the takeaway."""
    parts.append(_svg_tag("rect", {
        "x": x, "y": y, "width": width, "height": height, "rx": 14, "ry": 14,
        "fill": C["bg_card"], "stroke": C["border"], "stroke-width": 0.7}))
    parts.append(_svg_text(x + _ASIDE_PAD, y + _ASIDE_PAD + 6, aside.get("title", ""), {
        "fill": C["text"], "font-family": FONT_MONO, "font-size": 10,
        "font-weight": 700, "letter-spacing": "0.08em"}))

    row_y = y + _ASIDE_PAD + 22.0
    chip_w = width - 2 * _ASIDE_PAD
    for row in aside.get("rows") or []:
        if row == "...":
            parts.append(_svg_text(x + width / 2, row_y + 4, "···", {
                "text-anchor": "middle", "fill": C["muted"],
                "font-family": FONT_MONO, "font-size": 10}))
            row_y += _ASIDE_DOTS_H
            continue
        strong, sub = row
        parts.append(_svg_tag("rect", {
            "x": x + _ASIDE_PAD, "y": row_y, "width": chip_w, "height": _ASIDE_CHIP_H,
            "rx": 9, "ry": 9, "fill": C["badge_bg"],
            "stroke": C["border"], "stroke-width": 0.7}))
        parts.append(_svg_text(x + _ASIDE_PAD + 12, row_y + _ASIDE_CHIP_H / 2, strong, {
            "dominant-baseline": "central", "fill": C["text"],
            "font-family": FONT_MONO, "font-size": 10, "font-weight": 700}))
        parts.append(_svg_text(x + _ASIDE_PAD + chip_w - 12, row_y + _ASIDE_CHIP_H / 2, sub, {
            "text-anchor": "end", "dominant-baseline": "central", "fill": C["muted"],
            "font-family": FONT_MONO, "font-size": 9.5}))
        row_y += _ASIDE_CHIP_H + _ASIDE_CHIP_GAP

    footer = aside.get("footer") or []
    if footer:
        row_y += 2.0
        parts.append(_svg_tag("line", {
            "x1": x + _ASIDE_PAD, "y1": row_y, "x2": x + width - _ASIDE_PAD, "y2": row_y,
            "stroke": C["border"], "stroke-width": 0.7}))
        row_y += 8.0
        for i, text in enumerate(footer):
            parts.append(_svg_text(x + _ASIDE_PAD, row_y + 8, text, {
                "fill": C["text"] if i == 0 else C["muted"],
                "font-family": FONT_MONO, "font-size": 10,
                "font-weight": 700 if i == 0 else None}))
            row_y += 16.0
    return {"left": x, "right": x + width, "top": y, "bottom": y + height,
            "cx": x + width / 2, "cy": y + height / 2, "w": width, "h": height}


def _geom(cx: float, top: float, w: float, h: float) -> dict:
    x = cx - w / 2
    return {"left": x, "right": x + w, "top": top, "bottom": top + h,
            "cx": cx, "cy": top + h / 2, "w": w, "h": h}


def _badge(parts: list[str], right: float, y: float, text: str) -> None:
    """The white repeat pill, top-right of the cell frame — same styling as the
    main architecture view's ``× N`` badge."""
    w = max(66.0, 18.0 + 8.2 * len(text))
    x = right - w - 12
    parts.append(_svg_tag("rect", {
        "x": x, "y": y, "width": w, "height": 26, "rx": 13, "ry": 13,
        "fill": "rgba(255,255,255,0.65)", "stroke": C["border"], "stroke-width": 0.5}))
    parts.append(_svg_text(x + w / 2, y + 13, text, {
        "text-anchor": "middle", "dominant-baseline": "central",
        "fill": C["text"], "font-family": FONT_HEAD, "font-size": 17}))
