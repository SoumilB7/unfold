"""Low-level SVG primitives and routing helpers."""
from __future__ import annotations

from typing import Any

from .theme import BLOCK_LABEL_FONT_SIZE, C, FONT_HEAD, FONT_MONO, GAP
from .utils import _attr, _html, _num

SVG_FONT_BOOST = 2
BLOCK_LABEL_FONT_BOOST = 3

# Every diagram bakes its own <marker>/filter <defs> referenced by url(#id). Many
# panels share a view_key (self- AND cross-attention both render as "attn"; repeated
# layer-groups bake identical copies), so keying ids only by (mount, view) collided —
# and in the browser url(#dup) binds to the FIRST match, which is a hidden drill panel,
# so the arrowheads silently vanished (rsvg renders each svg isolated, hiding it). A
def _ids(mount_id: str, view: str) -> tuple[str, str]:
    from .render_context import ensure_render_context
    n = ensure_render_context().next_id()
    return f"{mount_id}-{view}-{n}-arrow", f"{mount_id}-{view}-{n}-shadow"


def _defs(arrow_id: str, shadow_id: str) -> str:
    marker = _svg_tag(
        "marker",
        {
            "id": arrow_id,
            "viewBox": "0 0 10 10",
            "refX": 8,
            "refY": 5,
            "markerWidth": 7,
            "markerHeight": 7,
            "orient": "auto-start-reverse",
        },
        _svg_tag(
            "path",
            {
                "d": "M2 1L8 5L2 9",
                "fill": "none",
                "stroke": C["arrow"],
                "stroke-width": 1.7,
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        ),
    )
    shadow = _svg_tag(
        "filter",
        {"id": shadow_id, "x": "-20%", "y": "-20%", "width": "140%", "height": "140%"},
        "".join(
            [
                _svg_tag("feGaussianBlur", {"in": "SourceAlpha", "stdDeviation": 1}),
                _svg_tag("feOffset", {"dx": 0, "dy": 1, "result": "off"}),
                _svg_tag(
                    "feComponentTransfer",
                    {},
                    _svg_tag("feFuncA", {"type": "linear", "slope": "0.16"}),
                ),
                _svg_tag(
                    "feMerge",
                    {},
                    _svg_tag("feMergeNode", {})
                    + _svg_tag("feMergeNode", {"in": "SourceGraphic"}),
                ),
            ]
        ),
    )
    return _svg_tag("defs", {}, marker + shadow)


def _region_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", stroke_width: float = 0) -> str:
    return _svg_tag(
        "rect",
        {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rx": 18,
            "ry": 18,
            "fill": fill,
            "stroke": stroke,
            "stroke-width": stroke_width,
        },
    )


def _rect_block(
    parts: list[str],
    info: dict,
    shadow_id: str,
    node_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str | list[str],
    font_size: int | None = None,
    *,
    resolved: bool = True,
    sub: str | None = None,
    accent: bool = False,
    clickable: bool = True,
) -> dict:
    """A rounded, clickable block.

    ``sub`` adds a second, smaller mono line beneath the heading (dims like
    ``768 → 3,072``); ``accent`` paints it as an input/output bookend (lighter
    fill).  Both are optional so this one primitive replaces the per-view
    ``_box`` re-implementations.
    """
    lines = label if isinstance(label, list) else [label]
    label_font_size = BLOCK_LABEL_FONT_SIZE if font_size is None else font_size + BLOCK_LABEL_FONT_BOOST
    line_h = label_font_size + SVG_FONT_BOOST + 2
    # Centre the WHOLE text stack (all heading lines + the optional sub line),
    # so a two-line heading and a dims line never overlap.
    sub_h = 16 if sub else 0
    stack_top = y + h / 2 - (len(lines) * line_h + sub_h) / 2
    start_y = stack_top + line_h / 2
    sub_cy = stack_top + len(lines) * line_h + sub_h / 2 + 1
    if accent:
        fill, stroke = C["bg_inner"], C["block_alt"]
        text_fill = C["text"]
        stroke_width = 0.6
    elif resolved:
        fill, stroke = C["block"], C["block_alt"]
        text_fill = C["text_block"]
        stroke_width = 0.6
    else:
        fill, stroke = C["badge_bg"], C["border"]
        text_fill = C["text"]
        stroke_width = 1.0

    children = [_node_title(info, node_id)]
    children.append(
        _svg_tag(
            "rect",
            {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "rx": 11,
                "ry": 11,
                "fill": fill,
                "stroke": stroke,
                "stroke-width": stroke_width,
                "filter": f"url(#{shadow_id})",
            },
        )
    )
    for i, line in enumerate(lines):
        children.append(
            _svg_text(
                x + w / 2,
                start_y + i * line_h,
                line,
                {
                    "text-anchor": "middle",
                    "dominant-baseline": "central",
                    "fill": text_fill,
                    "font-family": FONT_HEAD,
                    "font-size": label_font_size,
                    "pointer-events": "none",
                },
            )
        )
    if sub:
        children.append(
            _svg_text(
                x + w / 2,
                sub_cy,
                sub,
                {
                    "text-anchor": "middle",
                    "dominant-baseline": "central",
                    "fill": C["muted"] if accent else "rgba(255,255,255,0.84)",
                    "font-family": FONT_MONO,
                    "font-size": 11,
                    "pointer-events": "none",
                },
            )
        )
    if clickable:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)
    return {
        "left": x,
        "right": x + w,
        "top": y,
        "bottom": y + h,
        "cx": x + w / 2,
        "cy": y + h / 2,
        "w": w,
        "h": h,
    }


def _plus_block(parts: list[str], info: dict, shadow_id: str, node_id: str, cx: float, cy: float, sym: str = "+", *, clickable: bool = True) -> dict:
    r = 14
    children = [
        _node_title(info, node_id),
        _svg_tag(
            "circle",
            {
                "cx": cx,
                "cy": cy,
                "r": r,
                "fill": C["block"],
                "stroke": C["block_alt"],
                "stroke-width": 0.6,
                "filter": f"url(#{shadow_id})",
            },
        ),
    ]
    # Draw "+" / "×" as crossed strokes — Caveat's glyphs have uneven bearings,
    # so text-based rendering drifts off-centre. Lines are guaranteed symmetric.
    arm = 5
    stroke_attrs = {
        "stroke": C["text_block"],
        "stroke-width": 2.2,
        "stroke-linecap": "round",
        "pointer-events": "none",
    }
    if sym == "+":
        children.append(_svg_tag("line", {"x1": cx - arm, "y1": cy, "x2": cx + arm, "y2": cy, **stroke_attrs}))
        children.append(_svg_tag("line", {"x1": cx, "y1": cy - arm, "x2": cx, "y2": cy + arm, **stroke_attrs}))
    elif sym in ("×", "x", "*"):
        children.append(_svg_tag("line", {"x1": cx - arm, "y1": cy - arm, "x2": cx + arm, "y2": cy + arm, **stroke_attrs}))
        children.append(_svg_tag("line", {"x1": cx - arm, "y1": cy + arm, "x2": cx + arm, "y2": cy - arm, **stroke_attrs}))
    elif sym in ("dot", "⊙"):
        children.append(_svg_tag("circle", {
            "cx": cx, "cy": cy, "r": 5, "fill": "none",
            "stroke": C["text_block"], "stroke-width": 2, "pointer-events": "none",
        }))
    elif sym in ("‖", "||"):
        # Concat: two vertical bars, stroke-drawn (font glyphs drift off-centre).
        gap = 2.6
        children.append(_svg_tag("line", {"x1": cx - gap, "y1": cy - arm, "x2": cx - gap, "y2": cy + arm, **stroke_attrs}))
        children.append(_svg_tag("line", {"x1": cx + gap, "y1": cy - arm, "x2": cx + gap, "y2": cy + arm, **stroke_attrs}))
    else:
        children.append(_svg_text(
            cx,
            cy + 1,
            sym,
            {
                "text-anchor": "middle",
                "dominant-baseline": "central",
                "fill": C["text_block"],
                "font-family": FONT_HEAD,
                "font-size": 22,
                "pointer-events": "none",
            },
        ))
    if clickable:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)
    return {"left": cx - r, "right": cx + r, "top": cy - r, "bottom": cy + r, "cx": cx, "cy": cy, "r": r}


def _node_title(info: dict, node_id: str) -> str:
    # Tooltips intentionally disabled — block details live in the inspect panel.
    return ""


def _formula_block(
    parts: list[str],
    info: dict,
    shadow_id: str,
    node_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    numerator: str = "Q K^T",
    denominator: str | None = "sqrt(dim)",
    clickable: bool = True,
) -> dict:
    """Green fraction block (numerator over a rule over denominator) — the
    scaled-score step of SDPA-style attention.  A falsy ``denominator`` draws
    the numerator alone, centred (a code-proven UNscaled score — T5-family —
    where painting the sqrt would fabricate an op the forward never performs)."""
    children = [
        _node_title(info, node_id),
        _svg_tag("rect", {
            "x": x, "y": y, "width": w, "height": h, "rx": 11, "ry": 11,
            "fill": C["block"], "stroke": C["block_alt"], "stroke-width": 0.6,
            "filter": f"url(#{shadow_id})",
        }),
    ]
    if denominator:
        children += [
            _svg_text(x + w / 2, y + h * 0.32, numerator, {
                "text-anchor": "middle", "dominant-baseline": "central",
                "fill": C["text_block"], "font-family": FONT_HEAD, "font-size": 22,
                "pointer-events": "none",
            }),
            _svg_tag("line", {
                "x1": x + 72, "y1": y + h * 0.52, "x2": x + w - 72, "y2": y + h * 0.52,
                "stroke": C["text_block"], "stroke-width": 1.7,
                "stroke-linecap": "round", "pointer-events": "none",
            }),
            _svg_text(x + w / 2, y + h * 0.73, denominator, {
                "text-anchor": "middle", "dominant-baseline": "central",
                "fill": C["text_block"], "font-family": FONT_HEAD, "font-size": 19,
                "pointer-events": "none",
            }),
        ]
    else:
        children.append(_svg_text(x + w / 2, y + h / 2, numerator, {
            "text-anchor": "middle", "dominant-baseline": "central",
            "fill": C["text_block"], "font-family": FONT_HEAD, "font-size": 22,
            "pointer-events": "none",
        }))
    if clickable:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)
    return {"left": x, "right": x + w, "top": y, "bottom": y + h,
            "cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h}


def _window_strip(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    window_size: int | None,
) -> dict:
    """Token-context strip with the local sliding window highlighted — drawn as
    the *input* node of a sliding-window attention graph."""
    cell_w, cell_h, gap = 22, 18, 4
    n_cells, active_start, active_count = 15, 8, 5
    strip_w = n_cells * cell_w + (n_cells - 1) * gap
    active_w = active_count * cell_w + (active_count - 1) * gap
    strip_x = x + (w - strip_w) / 2
    strip_y = y + 4
    active_x = strip_x + active_start * (cell_w + gap)

    for idx in range(n_cells):
        is_active = active_start <= idx < active_start + active_count
        parts.append(_svg_tag("rect", {
            "x": strip_x + idx * (cell_w + gap), "y": strip_y,
            "width": cell_w, "height": cell_h, "rx": 4, "ry": 4,
            "fill": C["badge_bg"] if is_active else C["bg_card"],
            "stroke": C["block"] if is_active else C["border"],
            "stroke-width": 1 if is_active else 0.8,
            "opacity": 1 if is_active else 0.82,
        }))
    parts.append(_svg_tag("rect", {
        "x": active_x - 4, "y": strip_y - 4,
        "width": active_w + 8, "height": cell_h + 8, "rx": 7, "ry": 7,
        "fill": "none", "stroke": C["block"], "stroke-width": 1,
    }))
    parts.append(_svg_text(
        x + w / 2, strip_y + cell_h + 18,
        f"local window {window_size:,}" if window_size else "local window",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 9},
    ))
    return {"left": x, "right": x + w, "top": y, "bottom": y + h,
            "cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h}

def _cache_io_ports(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    compact: bool = False,
) -> None:
    """Write/read port pair in a block's lower-right corner — marks a tensor the
    KV cache stores (arrowhead = write, blunt tail = read)."""
    radius = 3.7 if compact else 5.2
    gap = 9 if compact else 13
    pad_right = 11 if compact else 17
    cy = y + h - (8 if compact else 10)
    head_cx = x + w - pad_right - gap
    tail_cx = x + w - pad_right
    stroke_w = 1.25 if compact else 1.45
    ports = [
        _cache_port(head_cx, cy, radius, "head", stroke_width=stroke_w),
        _cache_port(tail_cx, cy, radius, "tail", stroke_width=stroke_w),
    ]
    parts.append(_svg_tag("g", {
        "class": "uf-cache-ports", "pointer-events": "none", "aria-hidden": "true",
    }, "".join(ports)))

def _cache_port(
    cx: float,
    cy: float,
    radius: float,
    kind: str,
    *,
    stroke_width: float = 1.45,
) -> str:
    port = [_svg_tag("circle", {
        "cx": cx, "cy": cy, "r": radius,
        "fill": C["bg_outer"], "stroke": C["border"], "stroke-width": 0.7,
        "pointer-events": "none",
    })]
    if kind == "head":
        port.append(_svg_tag("path", {
            "d": f"M {cx - radius * 0.52:g} {cy + radius * 0.22:g} L {cx:g} {cy - radius * 0.48:g} L {cx + radius * 0.52:g} {cy + radius * 0.22:g}",
            "fill": "none", "stroke": C["arrow"], "stroke-width": stroke_width,
            "stroke-linecap": "round", "stroke-linejoin": "round",
            "pointer-events": "none",
        }))
    else:
        port.append(_svg_tag("line", {
            "x1": cx, "y1": cy - radius * 0.5, "x2": cx, "y2": cy + radius * 0.42,
            "stroke": C["arrow"], "stroke-width": stroke_width,
            "stroke-linecap": "round", "pointer-events": "none",
        }))
        port.append(_svg_tag("line", {
            "x1": cx - radius * 0.45, "y1": cy + radius * 0.42,
            "x2": cx + radius * 0.45, "y2": cy + radius * 0.42,
            "stroke": C["arrow"], "stroke-width": stroke_width,
            "stroke-linecap": "round", "pointer-events": "none",
        }))
    return "".join(port)


def _v_line(src: dict, dst: dict, arrow_id: str) -> str:
    # Arrows originate on the source edge and stop with breathing room before
    # the destination.  That keeps flow lines visually attached to the block
    # they leave, while the arrowhead has enough room before the next block.
    if src["cy"] > dst["cy"]:
        y1 = src["top"]
        y2 = dst["bottom"] + GAP
    else:
        y1 = src["bottom"]
        y2 = dst["top"] - GAP
    return _svg_tag(
        "line",
        {
            "x1": src["cx"],
            "y1": y1,
            "x2": src["cx"],
            "y2": y2,
            "stroke": C["arrow"],
            "stroke-width": 1.6,
            "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})",
            "fill": "none",
        },
    )


def _v_seg(x: float, y1: float, y2: float, arrow_id: str) -> str:
    return _svg_tag(
        "line",
        {
            "x1": x,
            "y1": y1,
            "x2": x,
            "y2": y2,
            "stroke": C["arrow"],
            "stroke-width": 1.6,
            "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})",
            "fill": "none",
        },
    )


def _elbow_vh(x1: float, y1: float, x2: float, y2: float, arrow_id: str | None) -> str:
    if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
        d = f"M {_num(x1)} {_num(y1)} L {_num(x2)} {_num(y2)}"
    else:
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        r = min(10, abs(x2 - x1) / 2, abs(y2 - y1) / 2)
        d = (
            f"M {_num(x1)} {_num(y1)} "
            f"L {_num(x1)} {_num(y2 - sy * r)} "
            f"Q {_num(x1)} {_num(y2)} {_num(x1 + sx * r)} {_num(y2)} "
            f"L {_num(x2)} {_num(y2)}"
        )
    return _path(d, arrow_id)


def _elbow_hv(x1: float, y1: float, x2: float, y2: float, arrow_id: str) -> str:
    if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
        d = f"M {_num(x1)} {_num(y1)} L {_num(x2)} {_num(y2)}"
    else:
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        r = min(10, abs(x2 - x1) / 2, abs(y2 - y1) / 2)
        d = (
            f"M {_num(x1)} {_num(y1)} "
            f"L {_num(x2 - sx * r)} {_num(y1)} "
            f"Q {_num(x2)} {_num(y1)} {_num(x2)} {_num(y1 + sy * r)} "
            f"L {_num(x2)} {_num(y2)}"
        )
    return _path(d, arrow_id)


def _merge_up_route(x1: float, y1: float, x2: float, entry_y: float, lane_y: float, arrow_id: str) -> str:
    """Route a lane's top up into the *bottom edge* of a merge block: rise from
    the lane, run horizontally just below the block, then rise into the bottom
    at an inset entry point.  Radii self-clamp to the available room, so the
    route stays valid however tight the layout gets."""
    if abs(x2 - x1) < 1:
        return _v_seg(x1, y1, entry_y, arrow_id)
    sx = 1 if x2 > x1 else -1
    r = max(2.0, min(10.0, (y1 - lane_y) / 2, (lane_y - entry_y) / 2, abs(x2 - x1) / 2))
    d = (
        f"M {_num(x1)} {_num(y1)} "
        f"L {_num(x1)} {_num(lane_y + r)} "
        f"Q {_num(x1)} {_num(lane_y)} {_num(x1 + sx * r)} {_num(lane_y)} "
        f"L {_num(x2 - sx * r)} {_num(lane_y)} "
        f"Q {_num(x2)} {_num(lane_y)} {_num(x2)} {_num(lane_y - r)} "
        f"L {_num(x2)} {_num(entry_y)}"
    )
    return _path(d, arrow_id)


def _block_top_to_block_bottom(x1: float, y1: float, x2: float, y2: float, arrow_id: str) -> str:
    """Route a block output into a block above it.

    Use this when the source point is the top edge of a block.  The line leaves
    vertically first, runs across under the target, then enters the target from
    below on a vertical final segment.  That avoids the recurring "side-bottom"
    arrowhead that happens when a vertical-horizontal elbow terminates at a
    block's bottom edge.  Split-dot fan-outs should keep using ``_elbow_hv``.
    """
    if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
        d = f"M {_num(x1)} {_num(y1)} L {_num(x2)} {_num(y2)}"
    else:
        clear = min(32, max(14, abs(y2 - y1) / 2))
        lane_y = y2 + clear if y1 > y2 else y2 - clear
        d = (
            f"M {_num(x1)} {_num(y1)} "
            f"L {_num(x1)} {_num(lane_y)} "
            f"L {_num(x2)} {_num(lane_y)} "
            f"L {_num(x2)} {_num(y2)}"
        )
    return _path(d, arrow_id)


def _residual_loop_right(src: dict, dst: dict, lane: float, arrow_id: str) -> str:
    """Residual bypass that taps the *input arrow* of ``src`` rather than
    leaving from ``src``'s right side.

    The visual is "the same x flowing into rms1 also feeds add1" — so the
    branch peels off the central column just below ``src``, runs to the lane
    on the right, climbs to ``dst`` level, and arrives from the right.
    """
    r = 12
    start_x, start_y = _input_tap(src)
    end_x, end_y = dst["right"], dst["cy"]
    d = (
        f"M {_num(start_x)} {_num(start_y)} "
        f"L {_num(lane - r)} {_num(start_y)} "
        f"Q {_num(lane)} {_num(start_y)} {_num(lane)} {_num(start_y - r)} "
        f"L {_num(lane)} {_num(end_y + r)} "
        f"Q {_num(lane)} {_num(end_y)} {_num(lane - r)} {_num(end_y)} "
        f"L {_num(end_x + GAP)} {_num(end_y)}"
    )
    return _path(d, arrow_id)


def _input_tap(node: dict) -> tuple[float, float]:
    """Tap point on the input stem below a block."""
    return node["cx"], node["bottom"] + GAP + 16


def _branch_dot(cx: float, cy: float) -> str:
    """Small filled circle marking a tap point on an arrow stem."""
    return _svg_tag("circle", {"cx": cx, "cy": cy, "r": 3.2, "fill": C["arrow"]})


def _path(d: str, arrow_id: str | None) -> str:
    return _svg_tag(
        "path",
        {
            "d": d,
            "fill": "none",
            "stroke": C["arrow"],
            "stroke-width": 1.6,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "marker-end": f"url(#{arrow_id})" if arrow_id else None,
        },
    )


def _svg(w: int, h: int, title: str, parts: list[str]) -> str:
    # aria-label keeps the diagram accessible to screen readers without
    # producing a native hover tooltip the way <title> does.
    return _svg_tag(
        "svg",
        {
            "width": "100%",
            "viewBox": f"0 0 {w} {h}",
            "role": "img",
            "aria-label": _attr(title),
            "xmlns": "http://www.w3.org/2000/svg",
        },
        "".join(parts),
    )


def _svg_text(x: float, y: float, text: Any, attrs: dict[str, Any] | None = None) -> str:
    attrs = dict(attrs or {})
    if isinstance(attrs.get("font-size"), (int, float)):
        attrs["font-size"] += SVG_FONT_BOOST
    attrs.update({"x": x, "y": y})
    return _svg_tag("text", attrs, _html(text))


def _svg_tag(name: str, attrs: dict[str, Any] | None = None, content: str | None = None) -> str:
    attr_text = "".join(
        f' {key}="{_attr(_num(value))}"'
        for key, value in (attrs or {}).items()
        if value is not None
    )
    if content is None:
        return f"<{name}{attr_text}/>"
    return f"<{name}{attr_text}>{content}</{name}>"
