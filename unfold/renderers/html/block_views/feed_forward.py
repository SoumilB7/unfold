"""Detail SVGs for feed-forward blocks."""
from __future__ import annotations

from ..svg import (
    _defs,
    _elbow_hv,
    _elbow_vh,
    _ids,
    _plus_block,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
    _v_seg,
)
from ..theme import C, FONT_BODY, FONT_HEAD, FONT_MONO, GAP


def build_moe_view(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 720, 620
    arrow_id, shadow_id = _ids(mount_id, "moe")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    router_w = 540
    router = _rect_block(parts, info, shadow_id, "router", (w - router_w) / 2, h - 130, router_w, 50, "Router")
    sum_node = _plus_block(parts, info, shadow_id, "add_moe", cx, 100)

    expert_w, expert_h = 116, 54
    expert_y = 235
    n_total = ffn.get("num_experts")
    last_label = str(n_total) if n_total else "N"
    side_pad = 60
    gap = (w - 2 * side_pad - 4 * expert_w) / 3
    slots = [
        (side_pad + 0 * (expert_w + gap), "Expert 1", "expert_1"),
        (side_pad + 1 * (expert_w + gap), "Expert k", "expert_k"),
        (side_pad + 2 * (expert_w + gap), "Expert k+1", "expert_kp1"),
        (side_pad + 3 * (expert_w + gap), f"Expert {last_label}", "expert_n"),
    ]
    experts = [
        _rect_block(parts, info, shadow_id, node_id, x, expert_y, expert_w, expert_h, label, font_size=15)
        for x, label, node_id in slots
    ]

    dots_x = (experts[1]["right"] + experts[2]["left"]) / 2
    dots_y = expert_y + expert_h / 2
    for i in range(-2, 3):
        parts.append(_svg_tag("circle", {"cx": dots_x + i * 7, "cy": dots_y, "r": 2.5, "fill": C["muted"]}))

    for expert in experts:
        parts.append(_v_seg(expert["cx"], router["top"], expert["bottom"] + GAP, arrow_id))

    for expert in experts:
        target_x = sum_node["cx"] + (-sum_node["r"] - GAP if expert["cx"] < sum_node["cx"] else sum_node["r"] + GAP)
        parts.append(_elbow_vh(expert["cx"], expert["top"], target_x, sum_node["cy"], arrow_id))

    if ffn.get("num_experts") and ffn.get("num_experts_per_tok"):
        sparsity = 100 * ffn["num_experts_per_tok"] / ffn["num_experts"]
        cg_x, cg_y, cg_w, cg_h = w - 244, 58, 188, 58
        parts.append(
            _svg_tag(
                "rect",
                {
                    "x": cg_x,
                    "y": cg_y,
                    "width": cg_w,
                    "height": cg_h,
                    "rx": 10,
                    "ry": 10,
                    "fill": C["bg_card"],
                    "stroke": C["border"],
                    "stroke-width": 0.5,
                },
            )
        )
        parts.append(
            _svg_text(
                cg_x + 12,
                cg_y + 18,
                "ACTIVE PER TOKEN",
                {
                    "fill": C["muted"],
                    "font-family": FONT_BODY,
                    "font-size": 10,
                    "letter-spacing": "0.12em",
                    "font-weight": 600,
                },
            )
        )
        parts.append(
            _svg_text(
                cg_x + 12,
                cg_y + 44,
                f"{ffn['num_experts_per_tok']} / {ffn['num_experts']}  -  {sparsity:.1f}%",
                {"fill": C["text"], "font-family": FONT_HEAD, "font-size": 22},
            )
        )

    parts.append(_svg_tag("line", {
        "x1": sum_node["cx"], "y1": sum_node["top"],
        "x2": sum_node["cx"], "y2": sum_node["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        sum_node["cx"], sum_node["top"] - 46,
        "out",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": router["bottom"] + 36,
        "x2": cx, "y2": router["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, router["bottom"] + 50,
        "in",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    return _svg(w, h, f"{ir.get('name', 'model')} mixture of experts", parts)


def build_ffn_view(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 720, 660
    arrow_id, shadow_id = _ids(mount_id, "ffn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    act_name = (ffn.get("activation") or "silu").upper()

    down_proj = _rect_block(parts, info, shadow_id, "down_proj", cx - 90, 110, 180, 50, "Linear (down)")
    mul_node = _plus_block(parts, info, shadow_id, "mul", cx, 230, "×")
    silu = _rect_block(parts, info, shadow_id, "silu", cx - 270, 330, 180, 50, act_name)
    up_proj = _rect_block(parts, info, shadow_id, "up_proj", cx + 90, 330, 180, 50, "Linear (up)")
    gate_proj = _rect_block(parts, info, shadow_id, "gate_proj", cx - 270, 460, 180, 50, "Linear (gate)")

    branch_y = h - 110
    parts.append(_svg_tag("circle", {"cx": cx, "cy": branch_y, "r": 4, "fill": C["arrow"]}))
    parts.append(_elbow_hv(cx, branch_y, gate_proj["cx"], gate_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(cx, branch_y, up_proj["cx"], up_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_line(gate_proj, silu, arrow_id))
    parts.append(_elbow_vh(silu["cx"], silu["top"], mul_node["cx"] - mul_node["r"] - GAP, mul_node["cy"], arrow_id))
    parts.append(_elbow_vh(up_proj["cx"], up_proj["top"], mul_node["cx"] + mul_node["r"] + GAP, mul_node["cy"], arrow_id))
    parts.append(_v_line(mul_node, down_proj, arrow_id))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": down_proj["top"],
        "x2": cx, "y2": down_proj["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, down_proj["top"] - 46,
        "out",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": branch_y + 38,
        "x2": cx, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(
        _svg_text(
            cx,
            h - 48,
            "in  ·  x",
            {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
        )
    )

    return _svg(w, h, f"{ir.get('name', 'model')} feed-forward block", parts)
