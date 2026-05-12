"""Plain multi-head scaled dot-product attention detail view."""
from __future__ import annotations

from ...svg import (
    _branch_dot,
    _defs,
    _elbow_vh,
    _elbow_hv,
    _ids,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
    _v_seg,
)
from ...theme import C, FONT_HEAD, GAP
from ...utils import _fmt_int
from .common import output_stem


def build(ir: dict, info: dict, mount_id: str) -> str:
    """Rich SVG detail view for standard MHA / SDPA attention blocks."""
    w, h = 820, 880
    arrow_id, shadow_id = _ids(mount_id, "attn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden_sz = ir.get("hidden_size") or 0
    num_heads = attn.get("num_heads") or 0
    head_dim = attn.get("head_dim") or (hidden_sz // num_heads if num_heads else 0)
    d_k = str(head_dim) if head_dim else "d_k"

    cx = w / 2
    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 100, 72, 200, 52, "Linear (out)")
    concat = _rect_block(
        parts,
        info,
        shadow_id,
        "concat_heads",
        cx - 112,
        164,
        224,
        54,
        ["Concat heads", f"{num_heads} x {d_k}" if num_heads else "per head"],
        font_size=16,
    )
    value_dot = _dot_operator(parts, info, shadow_id, "attn_apply_v", cx, 276)
    softmax = _rect_block(parts, info, shadow_id, "attn_softmax", cx - 96, 344, 192, 52, "Softmax")
    scaled_scores = _fraction_block(
        parts,
        info,
        shadow_id,
        "scaled_scores",
        cx - 140,
        452,
        280,
        82,
        numerator="Q K^T",
        denominator="sqrt(dim)",
    )

    parts.append(_v_line(scaled_scores, softmax, arrow_id))
    parts.append(_v_line(softmax, value_dot, arrow_id))
    parts.append(_v_line(value_dot, concat, arrow_id))
    parts.append(_v_line(concat, o_proj, arrow_id))

    proj_w, proj_h, proj_y = 185, 52, 704
    q_proj = _rect_block(parts, info, shadow_id, "q_proj", 78, proj_y, proj_w, proj_h, "Linear (Q)")
    k_proj = _rect_block(parts, info, shadow_id, "k_proj", cx - proj_w / 2, proj_y, proj_w, proj_h, "Linear (K)")
    v_proj = _rect_block(parts, info, shadow_id, "v_proj", w - 78 - proj_w, proj_y, proj_w, proj_h, "Linear (V)")

    branch_x, branch_y = cx, 792
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 42, "x2": branch_x, "y2": branch_y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "fill": "none",
    }))
    parts.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(branch_x, branch_y, k_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))
    parts.append(_branch_dot(branch_x, branch_y))

    parts.append(_input_to_block(q_proj["cx"], q_proj["top"], scaled_scores["left"] + 92, scaled_scores["bottom"], arrow_id))
    parts.append(_v_seg(k_proj["cx"], k_proj["top"], scaled_scores["bottom"], arrow_id))
    parts.append(_elbow_vh(v_proj["cx"], v_proj["top"], value_dot["right"] + GAP, value_dot["cy"], arrow_id))
    output_stem(parts, cx, o_proj, arrow_id, _fmt_int(hidden_sz), show_label=False)

    return _svg(w, h, f"{ir.get('name', 'model')} attention", parts)


def _fraction_block(
    parts: list[str],
    info: dict,
    shadow_id: str,
    node_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    numerator: str,
    denominator: str,
) -> dict:
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
            y + 27,
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
            "y1": y + 42,
            "x2": x + w - 72,
            "y2": y + 42,
            "stroke": C["text_block"],
            "stroke-width": 1.7,
            "stroke-linecap": "round",
            "pointer-events": "none",
        }),
        _svg_text(
            x + w / 2,
            y + 58,
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


def _input_to_block(x1: float, y1: float, x2: float, y2: float, arrow_id: str) -> str:
    lane_y = y2 + 26
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


def _dot_operator(
    parts: list[str],
    info: dict,
    shadow_id: str,
    node_id: str,
    cx: float,
    cy: float,
) -> dict:
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
