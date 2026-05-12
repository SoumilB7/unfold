"""Detail SVGs for mixture-of-experts blocks."""
from __future__ import annotations

from ....labels import activation_label
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
    _v_line,
    _v_seg,
)
from ..theme import C, GAP


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

    parts.append(_svg_tag("line", {
        "x1": sum_node["cx"], "y1": sum_node["top"],
        "x2": sum_node["cx"], "y2": sum_node["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": router["bottom"] + 36,
        "x2": cx, "y2": router["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    return _svg(w, h, f"{ir.get('name', 'model')} mixture of experts", parts)


def build_moe_expert_view(ir: dict, info: dict, mount_id: str, child: dict) -> str:
    """Third-level view for the FFN that lives inside one MoE expert."""
    w, h = 720, 520
    arrow_id, shadow_id = _ids(mount_id, child.get("id", "expert"))
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    act_name = activation_label(ffn.get("activation") or "silu")

    down_proj = _rect_block(parts, info, shadow_id, "expert_down_proj", cx - 92, 78, 184, 50, "Linear (down)")
    mul_node = _plus_block(parts, info, shadow_id, "expert_mul", cx, 180, "x")
    gate_proj = _rect_block(parts, info, shadow_id, "expert_gate_proj", 96, 360, 184, 50, "Linear (gate)", font_size=16)
    act = _rect_block(parts, info, shadow_id, "expert_act", 96, 252, 184, 50, act_name, font_size=16)
    up_proj = _rect_block(parts, info, shadow_id, "expert_up_proj", 440, 360, 184, 50, "Linear (up)", font_size=16)

    branch_y = h - 66
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": branch_y + 36,
        "x2": cx, "y2": branch_y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "fill": "none",
    }))
    parts.append(_svg_tag("circle", {"cx": cx, "cy": branch_y, "r": 3.8, "fill": C["arrow"]}))
    parts.append(_elbow_hv(cx, branch_y, gate_proj["cx"], gate_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(cx, branch_y, up_proj["cx"], up_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_line(gate_proj, act, arrow_id))
    parts.append(_elbow_vh(act["cx"], act["top"], mul_node["cx"] - mul_node["r"] - GAP, mul_node["cy"], arrow_id))
    parts.append(_elbow_vh(up_proj["cx"], up_proj["top"], mul_node["cx"] + mul_node["r"] + GAP, mul_node["cy"], arrow_id))
    parts.append(_v_line(mul_node, down_proj, arrow_id))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": down_proj["top"],
        "x2": cx, "y2": down_proj["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    return _svg(w, h, f"{ir.get('name', 'model')} MoE expert feed-forward", parts)
