"""Shared SVG pieces for attention detail views."""
from __future__ import annotations

from ...svg import (
    _branch_dot,
    _elbow_hv,
    _ids,
    _rect_block,
    _svg_tag,
    _svg_text,
    _v_line,
)
from ...stack_view import fit_svg, point
from ...theme import C, FONT_HEAD, FONT_MONO, GAP
from ...utils import _fmt_int


def queries_per_kv_group(num_heads: int, num_kv_heads: int) -> int | None:
    if not num_heads or not num_kv_heads:
        return None
    if num_heads % num_kv_heads:
        return None
    return num_heads // num_kv_heads


def has_cross_attention_context(info: dict) -> bool:
    """Whether this layer variant reads external vision states in attention."""
    if (info.get("dominant", {}).get("spec", {}).get("attention") or {}).get("cross_attention"):
        return True
    return any(
        block.get("id") in {"cross_attention_adapter", "cross_attention_states"}
        for block in (info.get("dominant", {}).get("spec", {}).get("blocks") or [])
    )


def cross_context_to_kv_inputs(
    parts: list[str],
    info: dict,
    shadow_id: str,
    arrow_id: str,
    q_proj: dict,
    k_proj: dict,
    v_proj: dict,
    branch_y: float,
) -> None:
    """Show cross-attention input: Q from decoder, K/V from vision states."""
    q_input_y = branch_y + 82
    parts.append(_svg_tag("line", {
        "x1": q_proj["cx"], "y1": q_input_y,
        "x2": q_proj["cx"], "y2": q_proj["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    vision_w = 250
    vision_cx = (k_proj["cx"] + v_proj["cx"]) / 2
    vision = _rect_block(
        parts,
        info,
        shadow_id,
        "cross_attention_states",
        vision_cx - vision_w / 2,
        branch_y + 30,
        vision_w,
        46,
        ["Projected image", "states"],
        font_size=15,
    )
    vision_branch_x = vision["cx"]
    vision_branch_y = branch_y
    parts.append(_svg_tag("line", {
        "x1": vision["cx"], "y1": vision["top"],
        "x2": vision_branch_x, "y2": vision_branch_y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "fill": "none",
    }))
    parts.append(_elbow_hv(vision_branch_x, vision_branch_y, k_proj["cx"], k_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(vision_branch_x, vision_branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))
    parts.append(_branch_dot(vision_branch_x, vision_branch_y))


def gqa_grouping_panel(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    num_heads: int,
    num_kv_heads: int,
    q_per_group: int | None,
) -> dict:
    if w <= 300:
        return _gqa_compact_legend(parts, x, y, w, h, num_heads, num_kv_heads, q_per_group)

    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "rx": 14,
        "ry": 14,
        "fill": C["bg_card"],
        "stroke": C["border"],
        "stroke-width": 0.7,
    }))
    parts.append(_svg_text(
        x + 18,
        y + 24,
        "KV sharing pattern",
        {
            "fill": C["text"],
            "font-family": FONT_MONO,
            "font-size": 11,
            "font-weight": 700,
            "letter-spacing": "0.08em",
        },
    ))

    card_y = y + 46
    card_w = 112
    gap = 16
    start_x = x + 38
    for i, (top, bottom) in enumerate(_gqa_card_specs(num_heads, num_kv_heads, q_per_group)):
        _gqa_group_card(parts, start_x + i * (card_w + gap), card_y, card_w, 48, top, bottom)

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


def _gqa_compact_legend(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    num_heads: int,
    num_kv_heads: int,
    q_per_group: int | None,
) -> dict:
    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "rx": 14,
        "ry": 14,
        "fill": C["bg_card"],
        "stroke": C["border"],
        "stroke-width": 0.7,
    }))
    parts.append(_svg_text(
        x + 18,
        y + 24,
        "KV sharing pattern",
        {
            "fill": C["text"],
            "font-family": FONT_MONO,
            "font-size": 10,
            "font-weight": 700,
            "letter-spacing": "0.08em",
        },
    ))
    subtitle = (
        f"{q_per_group} Q heads per KV"
        if q_per_group
        else f"{num_heads} Q / {num_kv_heads} KV"
    )
    parts.append(_svg_text(
        x + 18,
        y + 42,
        subtitle,
        {"fill": C["muted"], "font-family": FONT_MONO, "font-size": 9},
    ))

    card_y = y + 58
    card_h = 34
    gap = 7
    card_w = w - 36
    for i, (top, bottom) in enumerate(_gqa_card_specs(num_heads, num_kv_heads, q_per_group)):
        cy = card_y + i * (card_h + gap)
        parts.append(_svg_tag("rect", {
            "x": x + 18,
            "y": cy,
            "width": card_w,
            "height": card_h,
            "rx": 9,
            "ry": 9,
            "fill": C["badge_bg"],
            "stroke": C["border"],
            "stroke-width": 0.7,
        }))
        parts.append(_svg_text(
            x + 36,
            cy + 14,
            top,
            {"fill": C["text"], "font-family": FONT_MONO, "font-size": 10, "font-weight": 700},
        ))
        parts.append(_svg_text(
            x + 36,
            cy + 28,
            bottom,
            {"fill": C["muted"], "font-family": FONT_MONO, "font-size": 8.5},
        ))
        _cache_io_ports(parts, x + 18, cy, card_w, card_h, compact=True)

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


def mqa_shared_kv_node(parts: list[str], x: float, y: float, w: float, h: float, num_heads: int) -> dict:
    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "rx": 12,
        "ry": 12,
        "fill": C["badge_bg"],
        "stroke": C["border"],
        "stroke-width": 0.7,
    }))
    parts.append(_svg_text(
        x + w / 2,
        y + 18,
        "Shared K/V cache",
        {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_MONO, "font-size": 12, "font-weight": 700},
    ))
    parts.append(_svg_text(
        x + w / 2,
        y + 35,
        f"1 K + 1 V reused by {num_heads} Q" if num_heads else "1 K + 1 V reused",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
    ))
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


def _gqa_card_specs(num_heads: int, num_kv_heads: int, q_per_group: int | None) -> list[tuple[str, str]]:
    if not num_heads or not num_kv_heads or not q_per_group:
        return [("Q groups", "share K/V"), ("...", "..."), ("KV heads", "reused")]

    def q_range(group_idx: int) -> str:
        start = group_idx * q_per_group
        end = min(start + q_per_group - 1, num_heads - 1)
        return f"Q{start}" if start == end else f"Q{start}-Q{end}"

    if num_kv_heads == 1:
        return [(f"{q_range(0)}", "use KV0")]
    if num_kv_heads == 2:
        return [(q_range(0), "use KV0"), (q_range(1), "use KV1")]
    return [
        (q_range(0), "use KV0"),
        (q_range(1), "use KV1"),
        ("...", "..."),
        (q_range(num_kv_heads - 1), f"use KV{num_kv_heads - 1}"),
    ]


def _gqa_group_card(parts: list[str], x: float, y: float, w: float, h: float, top: str, bottom: str) -> None:
    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "rx": 10,
        "ry": 10,
        "fill": C["badge_bg"],
        "stroke": C["border"],
        "stroke-width": 0.7,
    }))
    parts.append(_svg_text(
        x + w / 2,
        y + 18,
        top,
        {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_MONO, "font-size": 12, "font-weight": 700},
    ))
    parts.append(_svg_text(
        x + w / 2,
        y + 36,
        bottom,
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
    ))
    _cache_io_ports(parts, x, y, w, h, compact=True)


def kv_cache_badge(parts: list[str], x: float, y: float, title: str, subtitle: str) -> dict:
    w, h = 162, 52
    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "rx": 12,
        "ry": 12,
        "fill": C["bg_card"],
        "stroke": C["border"],
        "stroke-width": 0.7,
    }))
    parts.append(_svg_text(
        x + 81,
        y + 20,
        title,
        {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_MONO, "font-size": 11, "font-weight": 700},
    ))
    parts.append(_svg_text(
        x + 81,
        y + 37,
        subtitle,
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 9},
    ))
    return {"left": x, "right": x + w, "top": y, "bottom": y + h, "cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h}


def kv_cache_port_hint(parts: list[str], kv_nodes: list[dict]) -> None:
    """Two punched ports on K/V blocks: arrowhead for write, tail for read."""
    for node in kv_nodes:
        _cache_io_ports(parts, node["left"], node["top"], node["w"], node["h"])


def cache_read_write_ports(
    parts: list[str],
    node: dict,
    *,
    write_side: str = "bottom",
    read_side: str = "top",
) -> None:
    """Punched ports on a cache node, placed where cache write/read happen."""
    write = _edge_port_center(node, write_side, "write")
    read = _edge_port_center(node, read_side, "read")
    ports = [
        _cache_port(write[0], write[1], 5.2, "head"),
        _cache_port(read[0], read[1], 5.2, "tail"),
    ]
    parts.append(_svg_tag("g", {
        "class": "uf-cache-ports",
        "pointer-events": "none",
        "aria-hidden": "true",
    }, "".join(ports)))


def _cache_io_ports(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    compact: bool = False,
) -> None:
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
        "class": "uf-cache-ports",
        "pointer-events": "none",
        "aria-hidden": "true",
    }, "".join(ports)))


def _edge_port_center(node: dict, side: str, role: str) -> tuple[float, float]:
    x, y, w, h = node["left"], node["top"], node["w"], node["h"]
    inset = 17
    x_pos = x + w - inset
    if side == "top":
        return x_pos, y + 10
    if side == "bottom":
        return x_pos, y + h - 10
    y_pos = y + h - (18 if role == "write" else 34)
    if side == "left":
        return x + 10, y_pos
    return x + w - 10, y_pos


def _cache_port(
    cx: float,
    cy: float,
    radius: float,
    kind: str,
    *,
    stroke_width: float = 1.45,
) -> str:
    port = [_svg_tag("circle", {
        "cx": cx,
        "cy": cy,
        "r": radius,
        "fill": C["bg_outer"],
        "stroke": C["border"],
        "stroke-width": 0.7,
        "pointer-events": "none",
    })]
    if kind == "head":
        port.append(_svg_tag("path", {
            "d": f"M {cx - radius * 0.52:g} {cy + radius * 0.22:g} L {cx:g} {cy - radius * 0.48:g} L {cx + radius * 0.52:g} {cy + radius * 0.22:g}",
            "fill": "none",
            "stroke": C["arrow"],
            "stroke-width": stroke_width,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "pointer-events": "none",
        }))
    else:
        port.append(_svg_tag("line", {
            "x1": cx,
            "y1": cy - radius * 0.5,
            "x2": cx,
            "y2": cy + radius * 0.42,
            "stroke": C["arrow"],
            "stroke-width": stroke_width,
            "stroke-linecap": "round",
            "pointer-events": "none",
        }))
        port.append(_svg_tag("line", {
            "x1": cx - radius * 0.45,
            "y1": cy + radius * 0.42,
            "x2": cx + radius * 0.45,
            "y2": cy + radius * 0.42,
            "stroke": C["arrow"],
            "stroke-width": stroke_width,
            "stroke-linecap": "round",
            "pointer-events": "none",
        }))
    return "".join(port)


def vertical_attention_stack(
    ir: dict,
    info: dict,
    mount_id: str,
    view: str,
    title: str,
    nodes: list[tuple[str, str | list[str], int | None]],
    *,
    h: int = 590,
) -> str:
    w = 560  # internal layout grid (block width / step spacing); canvas auto-fits
    arrow_id, shadow_id = _ids(mount_id, view)
    parts: list[str] = []

    hidden = _fmt_int(ir.get("hidden_size"))
    cx = w / 2
    geoms = []
    top_y = 86
    step = (h - 190) / max(len(nodes) - 1, 1)
    for i, (node_id, label, font_size) in enumerate(nodes):
        block_w = 240 if isinstance(label, list) else 210
        y = top_y + i * step
        geoms.append(
            _rect_block(
                parts, info, shadow_id, node_id,
                cx - block_w / 2, y, block_w, 50,
                label,
                font_size=font_size or 16,
            )
        )

    for src, dst in zip(reversed(geoms), reversed(geoms[:-1])):
        parts.append(_v_line(src, dst, arrow_id))

    input_node = geoms[-1]
    in_label_y = input_node["bottom"] + 20
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": input_node["bottom"] + 38,
        "x2": cx, "y2": input_node["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, in_label_y,
        f"in ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))
    output_stem(parts, cx, geoms[0], arrow_id, hidden)

    regions = [*geoms, point(cx, geoms[0]["top"] - 50), point(cx, input_node["bottom"] + 44)]
    return fit_svg(arrow_id, shadow_id, parts, regions, f"{ir.get('name', 'model')} {title}", min_width=w)


def output_stem(
    parts: list[str],
    cx: float,
    top_node: dict,
    arrow_id: str,
    hidden: str,
    *,
    show_label: bool = True,
) -> None:
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": top_node["top"],
        "x2": cx, "y2": top_node["top"] - 34,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    if not show_label:
        return
    parts.append(_svg_text(
        cx, top_node["top"] - 44,
        f"out ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))


def attn_dim_label(parts: list[str], x: float, y: float, text: str, *, anchor: str = "start") -> None:
    parts.append(_svg_text(
        x, y, text,
        {
            "text-anchor": anchor,
            "dominant-baseline": "central",
            "fill": C["muted"],
            "font-family": FONT_MONO,
            "font-size": 10,
        },
    ))


def sdpa_fraction_block(
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
    denominator: str = "sqrt(dim)",
) -> dict:
    """Green formula block for the scaled score step in SDPA-style attention."""
    children = [
        _svg_tag("rect", {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rx": 11,
            "ry": 11,
            "fill": C["block"],
            "stroke": C["block_alt"],
            "stroke-width": 0.6,
            "filter": f"url(#{shadow_id})",
        }),
        _svg_text(
            x + w / 2,
            y + h * 0.32,
            numerator,
            {
                "text-anchor": "middle",
                "dominant-baseline": "central",
                "fill": C["text_block"],
                "font-family": FONT_HEAD,
                "font-size": 22,
                "pointer-events": "none",
            },
        ),
        _svg_tag("line", {
            "x1": x + 72,
            "y1": y + h * 0.52,
            "x2": x + w - 72,
            "y2": y + h * 0.52,
            "stroke": C["text_block"],
            "stroke-width": 1.7,
            "stroke-linecap": "round",
            "pointer-events": "none",
        }),
        _svg_text(
            x + w / 2,
            y + h * 0.73,
            denominator,
            {
                "text-anchor": "middle",
                "dominant-baseline": "central",
                "fill": C["text_block"],
                "font-family": FONT_HEAD,
                "font-size": 19,
                "pointer-events": "none",
            },
        ),
    ]
    parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    return {"left": x, "right": x + w, "top": y, "bottom": y + h, "cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h}


def sdpa_dot_operator(parts: list[str], info: dict, shadow_id: str, node_id: str, cx: float, cy: float) -> dict:
    """Small green dot-product operator used for attention weights × V."""
    r = 16
    children = [
        _svg_tag("circle", {
            "cx": cx,
            "cy": cy,
            "r": r,
            "fill": C["block"],
            "stroke": C["block_alt"],
            "stroke-width": 0.6,
            "filter": f"url(#{shadow_id})",
        }),
        _svg_tag("circle", {
            "cx": cx,
            "cy": cy,
            "r": 5,
            "fill": "none",
            "stroke": C["text_block"],
            "stroke-width": 2,
            "pointer-events": "none",
        }),
    ]
    parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    return {"left": cx - r, "right": cx + r, "top": cy - r, "bottom": cy + r, "cx": cx, "cy": cy, "r": r}


def input_to_block(x1: float, y1: float, x2: float, y2: float, arrow_id: str, *, lane_offset: float = 26) -> str:
    """Route an upward input into the lower edge of a block without grazing it."""
    lane_y = y2 + lane_offset
    r = 10
    d = (
        f"M {x1:g} {y1:g} "
        f"L {x1:g} {lane_y + r:g} "
        f"Q {x1:g} {lane_y:g} {x1 + r:g} {lane_y:g} "
        f"L {x2 - r:g} {lane_y:g} "
        f"Q {x2:g} {lane_y:g} {x2:g} {lane_y - r:g} "
        f"L {x2:g} {y2:g}"
    )
    return _svg_tag("path", {
        "d": d,
        "fill": "none",
        "stroke": C["arrow"],
        "stroke-width": 1.6,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})",
    })
