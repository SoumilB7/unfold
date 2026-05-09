"""SVG views for architecture, FFN, MoE, and layer maps."""
from __future__ import annotations

from .metadata import _block_label, _signature
from .svg import (
    _defs,
    _elbow_hv,
    _elbow_vh,
    _ids,
    _path,
    _plus_block,
    _rect_block,
    _region_rect,
    _residual_loop_right,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
    _v_seg,
)
from .theme import C, FONT_BODY, FONT_HEAD, FONT_MONO, GAP


def _build_architecture_view(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 720, 920
    arrow_id, shadow_id = _ids(mount_id, "arch")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 26, w - 80, h - 52, C["bg_outer"]))

    cx = w / 2
    inner_x, inner_y, inner_w, inner_h = 110, 200, w - 220, 490
    parts.append(_region_rect(inner_x, inner_y, inner_w, inner_h, C["bg_inner"]))

    # Layout (top → bottom, smaller y = higher on screen):
    #   lm_head → final_rms → [inner block: add2 → ffn → rms2 → add1 → attn → rms1] → embed → tok_text
    tok_text  = _rect_block(parts, info, shadow_id, "tok_text",  cx - 110, h - 100, 220, 44, _block_label(info, "tok_text", "Tokenized text"), font_size=17)
    embed     = _rect_block(parts, info, shadow_id, "embed",     cx - 130, h - 168, 260, 44, _block_label(info, "embed", "Token Embedding layer"), font_size=17)
    rms1      = _rect_block(parts, info, shadow_id, "rms1",      cx - 80,  inner_y + 400, 160, 36, _block_label(info, "rms1", "RMSNorm"), font_size=16)
    attn      = _rect_block(parts, info, shadow_id, "attn",      cx - 115, inner_y + 305, 230, 60, _block_label(info, "attn", ["Multi-Head", "Attention"]), font_size=17)
    add1      = _plus_block(parts, info, shadow_id, "add1",      cx,       inner_y + 270)
    rms2      = _rect_block(parts, info, shadow_id, "rms2",      cx - 80,  inner_y + 195, 160, 36, _block_label(info, "rms2", "RMSNorm"), font_size=16)
    ffn_node  = _rect_block(parts, info, shadow_id, "ffn",       cx - 80,  inner_y + 110, 160, 44, _block_label(info, "ffn", "Feed-Forward"), font_size=17)
    add2      = _plus_block(parts, info, shadow_id, "add2",      cx,       inner_y + 75)
    final_rms = _rect_block(parts, info, shadow_id, "final_rms", cx - 90,  140, 180, 36, _block_label(info, "final_rms", "Final RMSNorm"), font_size=16)
    lm_head   = _rect_block(parts, info, shadow_id, "lm_head",   cx - 130, 70,  260, 44, _block_label(info, "lm_head", "Linear output layer"), font_size=17)

    for src, dst in (
        (tok_text, embed),
        (embed, rms1),
        (rms1, attn),
        (attn, add1),
        (add1, rms2),
        (rms2, ffn_node),
        (ffn_node, add2),
        (add2, final_rms),
        (final_rms, lm_head),
    ):
        parts.append(_v_line(src, dst, arrow_id))

    parts.append(
        _svg_tag(
            "line",
            {
                "x1": cx,
                "y1": lm_head["top"],
                "x2": cx,
                "y2": lm_head["top"] - 32,
                "stroke": C["arrow"],
                "stroke-width": 1.6,
                "stroke-linecap": "round",
                "marker-end": f"url(#{arrow_id})",
                "fill": "none",
            },
        )
    )

    lane = inner_x + inner_w - 28
    parts.append(_residual_loop_right(rms1, add1, lane, arrow_id))
    parts.append(_residual_loop_right(rms2, add2, lane, arrow_id))

    parts.append(
        _svg_tag(
            "rect",
            {
                "x": inner_x + inner_w - 78,
                "y": inner_y + 12,
                "width": 66,
                "height": 26,
                "rx": 13,
                "ry": 13,
                "fill": "rgba(255,255,255,0.65)",
                "stroke": C["border"],
                "stroke-width": 0.5,
            },
        )
    )
    parts.append(
        _svg_text(
            inner_x + inner_w - 45,
            inner_y + 25,
            f"x {len(ir.get('layers', []))}",
            {
                "text-anchor": "middle",
                "dominant-baseline": "central",
                "fill": C["text"],
                "font-family": FONT_HEAD,
                "font-size": 20,
            },
        )
    )

    return _svg(w, h, f"{ir.get('name', 'model')} architecture", parts)


def _build_moe_view(ir: dict, info: dict, mount_id: str) -> str:
    # Slightly taller (h=620) to leave room for incoming arrow below router
    # and outgoing arrow above the sum node.
    w, h = 720, 620
    arrow_id, shadow_id = _ids(mount_id, "moe")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    # Router spans wide enough that all four expert arrows visibly emerge from
    # within the router rectangle (otherwise the leftmost/rightmost arrows
    # appear to start in empty space, far outside the router block).
    router_w = 540
    router = _rect_block(parts, info, shadow_id, "router",
                         (w - router_w) / 2, h - 130, router_w, 50, "Router")
    sum_node = _plus_block(parts, info, shadow_id, "add_moe", cx, 100)

    # Expert layout: 4 evenly-spaced columns symmetric around the centre.
    # Previous version overlapped slot 3 and slot 4 (wrong x for slot 4).
    expert_w, expert_h = 116, 54
    expert_y = 235
    n_total = ffn.get("num_experts")
    last_label = str(n_total) if n_total else "N"
    # Spacing computed so columns are: 60, 220, 380, 540 with w=720.
    side_pad = 60
    gap = (w - 2 * side_pad - 4 * expert_w) / 3
    slots = [
        (side_pad + 0 * (expert_w + gap), "Expert 1",        "expert_1"),
        (side_pad + 1 * (expert_w + gap), "Expert k",        "expert_k"),
        (side_pad + 2 * (expert_w + gap), "Expert k+1",      "expert_kp1"),
        (side_pad + 3 * (expert_w + gap), f"Expert {last_label}", "expert_n"),
    ]
    experts = [
        _rect_block(parts, info, shadow_id, node_id, x, expert_y, expert_w, expert_h, label, font_size=15)
        for x, label, node_id in slots
    ]

    # Ellipsis lives in the gap between expert 2 (k) and expert 3 (k+1).
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

    # Outgoing arrow at the top — leaves the weighted-sum node going up.
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

    # Incoming arrow at the bottom — enters the router from below.
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


def _build_ffn_view(ir: dict, info: dict, mount_id: str) -> str:
    # Tall enough that both the outgoing "out" label and the incoming "in·x"
    # label sit inside the outer region boundary (region = y 30 → h-30).
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

    # Outgoing arrow at top — leaves down_proj going up.
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

    # Incoming arrow at the bottom — points up into the input branch dot.
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


def _build_layer_map(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 720, 240
    arrow_id, shadow_id = _ids(mount_id, "map")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_card"], stroke=C["border"], stroke_width=0.5))

    palette = ["#0F6E56", "#1D9E75", "#0E7C8C", "#3C3489", "#993C1D", "#185FA5", "#65A30D"]
    sig_to_color = {group["sig"]: palette[i % len(palette)] for i, group in enumerate(info["groups"])}

    strip_x, strip_y, strip_w, strip_h = 80, 90, w - 160, 36
    layers = ir.get("layers", [])
    n = len(layers)
    col_w = strip_w / max(n, 1)

    for i, layer in enumerate(layers):
        sig = _signature(layer)
        parts.append(
            _svg_tag(
                "rect",
                {
                    "x": strip_x + i * col_w,
                    "y": strip_y,
                    "width": max(col_w - 0.5, 1),
                    "height": strip_h,
                    "fill": sig_to_color.get(sig, palette[0]),
                    "opacity": 0.95,
                },
            )
        )

    parts.append(
        _svg_tag(
            "rect",
            {
                "x": strip_x,
                "y": strip_y,
                "width": strip_w,
                "height": strip_h,
                "fill": "none",
                "stroke": C["text"],
                "stroke-width": 0.4,
                "rx": 4,
                "ry": 4,
            },
        )
    )

    if n:
        for idx in (0, n - 1):
            x = strip_x + (idx + 0.5) * col_w
            parts.append(
                _svg_text(
                    x,
                    strip_y + strip_h + 16,
                    f"L{idx}",
                    {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
                )
            )

    type_word = "type" if len(info["groups"]) == 1 else "types"
    parts.append(
        _svg_text(
            strip_x,
            70,
            f"{n} layers - {len(info['groups'])} {type_word}",
            {"fill": C["text"], "font-family": FONT_BODY, "font-size": 12, "font-weight": 600},
        )
    )

    lx, ly = strip_x, strip_y + strip_h + 44
    for group in info["groups"]:
        spec = group["spec"]
        ffn_kind = "MoE" if spec["ffn"].get("kind") == "moe" else "Dense"
        mask = "SWA" if spec["attention"].get("mask") == "sliding" else "full"
        first, last = group["indices"][0], group["indices"][-1]
        label = (
            f"{spec['attention'].get('kind', '').upper()} + {ffn_kind} ({mask}) - "
            f"L{first}-L{last} - {len(group['indices'])}x"
        )
        color = sig_to_color[group["sig"]]
        parts.append(_svg_tag("rect", {"x": lx, "y": ly - 9, "width": 12, "height": 12, "fill": color, "rx": 2}))
        parts.append(
            _svg_text(
                lx + 18,
                ly,
                label,
                {"dominant-baseline": "central", "fill": C["text"], "font-family": FONT_BODY, "font-size": 12},
            )
        )
        ly += 20

    return _svg(w, h, f"{ir.get('name', 'model')} layer map", parts)
