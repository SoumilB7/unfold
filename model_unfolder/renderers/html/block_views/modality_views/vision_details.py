"""Drill-down SVGs for vision pathway internals."""
from __future__ import annotations

from ...svg import _defs, _ids, _rect_block, _region_rect, _svg, _svg_tag, _svg_text
from ...theme import C
from ...utils import _fmt_int
from .common import vision_input


def build_patch_embedding_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """Show pixels being split into patches and projected into patch tokens."""
    vision = vision_input(ir)
    input_spec = vision.get("input") or {}
    embedding = vision.get("embedding") or {}
    patch = input_spec.get("patch_size") or embedding.get("patch_size")
    image_size = input_spec.get("image_size")
    out = embedding.get("out_features")

    w, h = 760, 620
    arrow_id, shadow_id = _ids(mount_id, "vision-patch-embedding")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    pixels = _rect_block(parts, info, shadow_id, "vision_pixels", cx - 115, 536, 230, 46, "Image pixels")
    grid = _patch_grid(parts, cx, 346, patch, image_size)
    flatten = _rect_block(parts, info, shadow_id, "vision_patch_flatten", cx - 125, 270, 250, 46, "Flatten patches")
    project = _rect_block(parts, info, shadow_id, "vision_patch_project", cx - 150, 180, 300, 52, _projection_label(out))
    tokens = _rect_block(parts, info, shadow_id, "vision_patch_tokens", cx - 130, 88, 260, 46, "Patch tokens")

    _up_arrow(parts, pixels["cx"], pixels["top"], grid["bottom"] + 16)
    _up_arrow(parts, grid["cx"], grid["top"], flatten["bottom"] + 12)
    _up_arrow(parts, flatten["cx"], flatten["top"], project["bottom"] + 12)
    _up_arrow(parts, project["cx"], project["top"], tokens["bottom"] + 12)
    _up_arrow(parts, tokens["cx"], tokens["top"], tokens["top"] - 36)

    return _svg(w, h, f"{ir.get('name', 'model')} patch embedding", parts)


def build_vision_encoder_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """Show the ViT-style encoder stack inside the vision pathway."""
    vision = vision_input(ir)
    encoder = vision.get("encoder") or {}
    layers = encoder.get("num_layers")
    heads = encoder.get("num_attention_heads")
    hidden = encoder.get("hidden_size")
    pos = (encoder.get("position_encoding") or {}).get("kind")

    w, h = 820, 900
    arrow_id, shadow_id = _ids(mount_id, "vision-encoder")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    _tower_badge(parts, 604, 50)
    patch_tokens = _rect_block(parts, info, shadow_id, "vision_patch_tokens", cx - 140, 800, 280, 48, "Patch tokens")
    pos_block = _rect_block(parts, info, shadow_id, "vision_position", cx - 160, 690, 320, 52, _pos_label(pos))
    stack = _encoder_stack(parts, info, shadow_id, cx, 170, layers, heads, hidden)
    encoded = _rect_block(parts, info, shadow_id, "vision_encoded_states", cx - 175, 72, 350, 54, "Encoded image states")

    _up_arrow(parts, patch_tokens["cx"], patch_tokens["top"], pos_block["bottom"] + 12)
    _v_stem(parts, pos_block["cx"], pos_block["top"], stack["bottom"])
    _up_arrow(parts, stack["cx"], stack["top"], encoded["bottom"] + 12)
    _up_arrow(parts, encoded["cx"], encoded["top"], encoded["top"] - 34)

    return _svg(w, h, f"{ir.get('name', 'model')} vision encoder", parts)


def build_vision_self_attention_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """Show the self-attention sublayer inside one vision encoder block."""
    encoder = (vision_input(ir).get("encoder") or {})
    heads = encoder.get("num_attention_heads")
    hidden = encoder.get("hidden_size")
    head_dim = _head_dim(heads, hidden)

    w, h = 820, 760
    arrow_id, shadow_id = _ids(mount_id, "vision-self-attention")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    out = _rect_block(parts, info, shadow_id, "vision_attn_out", cx - 105, 70, 210, 50, "Linear (out)")
    concat = _rect_block(parts, info, shadow_id, "vision_attn_concat", cx - 120, 154, 240, 56, _concat_label(heads, head_dim), font_size=16)
    apply_v = _dot_node(parts, "vision_attn_values", cx, 270, shadow_id)
    softmax = _rect_block(parts, info, shadow_id, "vision_attn_softmax", cx - 100, 338, 200, 52, "Softmax")
    scaled = _fraction_block(parts, info, shadow_id, "vision_attn_scaled", cx - 150, 446, 300, 84)

    q_proj = _rect_block(parts, info, shadow_id, "vision_attn_q", 80, 606, 190, 52, "Linear (Q)")
    k_proj = _rect_block(parts, info, shadow_id, "vision_attn_k", 315, 606, 190, 52, "Linear (K)")
    v_proj = _rect_block(parts, info, shadow_id, "vision_attn_v", 550, 606, 190, 52, "Linear (V)")

    _up_arrow(parts, cx, 724, 692)
    _branch_to_three(parts, cx, 692, [q_proj, k_proj, v_proj], arrow_id)
    _marker_path(parts, f"M {q_proj['cx']} {q_proj['top']} L {q_proj['cx']} 548 Q {q_proj['cx']} 538 {q_proj['cx'] + 10} 538 L {scaled['left'] + 48} 538 Q {scaled['left'] + 58} 538 {scaled['left'] + 58} 548 L {scaled['left'] + 58} {scaled['bottom'] + 8}", arrow_id)
    _marker_path(parts, f"M {k_proj['cx']} {k_proj['top']} L {k_proj['cx']} {scaled['bottom'] + 8}", arrow_id)
    _marker_path(parts, f"M {v_proj['cx']} {v_proj['top']} L {v_proj['cx']} 280 Q {v_proj['cx']} 270 {v_proj['cx'] - 10} 270 L {apply_v['right'] + 8} 270", arrow_id)
    _up_arrow(parts, scaled["cx"], scaled["top"], softmax["bottom"] + 12)
    _up_arrow(parts, softmax["cx"], softmax["top"], apply_v["bottom"] + 12)
    _up_arrow(parts, apply_v["cx"], apply_v["top"], concat["bottom"] + 12)
    _up_arrow(parts, concat["cx"], concat["top"], out["bottom"] + 12)
    _up_arrow(parts, out["cx"], out["top"], out["top"] - 34)

    return _svg(w, h, f"{ir.get('name', 'model')} vision self-attention", parts)


def build_vision_mlp_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """Show the feed-forward sublayer inside one vision encoder block."""
    encoder = (vision_input(ir).get("encoder") or {})
    hidden = encoder.get("hidden_size")
    intermediate = encoder.get("intermediate_size")

    w, h = 720, 500
    arrow_id, shadow_id = _ids(mount_id, "vision-mlp")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    in_block = _rect_block(parts, info, shadow_id, "vision_mlp_input", cx - 125, 390, 250, 46, "Patch states")
    fc1 = _rect_block(parts, info, shadow_id, "vision_mlp_fc1", cx - 150, 282, 300, 52, _mlp_linear_label("Linear (in)", hidden, intermediate))
    act = _rect_block(parts, info, shadow_id, "vision_mlp_activation", cx - 105, 190, 210, 46, "Activation")
    fc2 = _rect_block(parts, info, shadow_id, "vision_mlp_fc2", cx - 150, 98, 300, 52, _mlp_linear_label("Linear (out)", intermediate, hidden))

    _up_arrow(parts, in_block["cx"], in_block["top"], fc1["bottom"] + 12)
    _up_arrow(parts, fc1["cx"], fc1["top"], act["bottom"] + 12)
    _up_arrow(parts, act["cx"], act["top"], fc2["bottom"] + 12)
    _up_arrow(parts, fc2["cx"], fc2["top"], fc2["top"] - 34)

    return _svg(w, h, f"{ir.get('name', 'model')} vision MLP", parts)


def _tower_badge(parts: list[str], x: float, y: float) -> None:
    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": 158,
        "height": 28,
        "rx": 14,
        "ry": 14,
        "fill": "rgba(255,255,255,0.72)",
        "stroke": C["border"],
        "stroke-width": 0.6,
    }))
    parts.append(_svg_text(x + 79, y + 14, "separate tower", {
        "text-anchor": "middle",
        "dominant-baseline": "central",
        "fill": C["text"],
        "font-family": "ui-monospace, \"JetBrains Mono\", \"SF Mono\", Menlo, monospace",
        "font-size": 11,
        "font-weight": 700,
    }))


def _dot_node(parts: list[str], node_id: str, cx: float, cy: float, shadow_id: str) -> dict:
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


def _fraction_block(parts: list[str], info: dict, shadow_id: str, node_id: str, x: float, y: float, w: float, h: float) -> dict:
    block = _rect_block(parts, info, shadow_id, node_id, x, y, w, h, ["Q K^T", "sqrt(dim)"], font_size=18)
    parts.append(_svg_tag("line", {
        "x1": x + 72,
        "y1": y + h / 2 + 1,
        "x2": x + w - 72,
        "y2": y + h / 2 + 1,
        "stroke": C["text_block"],
        "stroke-width": 1.7,
        "stroke-linecap": "round",
        "pointer-events": "none",
    }))
    return block


def _branch_to_three(parts: list[str], x: float, y: float, blocks: list[dict], arrow_id: str) -> None:
    parts.append(_svg_tag("circle", {"cx": x, "cy": y, "r": 3.2, "fill": C["arrow"]}))
    for block in blocks:
        if block["cx"] == x:
            _marker_path(parts, f"M {x} {y} L {block['cx']} {block['top'] - 8}", arrow_id)
        else:
            turn_y = y
            _marker_path(
                parts,
                f"M {x} {turn_y} L {block['cx']} {turn_y} Q {block['cx']} {turn_y} {block['cx']} {turn_y - 10} "
                f"L {block['cx']} {block['top'] - 8}",
                arrow_id,
            )


def _marker_path(parts: list[str], d: str, arrow_id: str) -> None:
    parts.append(_svg_tag("path", {
        "d": d,
        "fill": "none",
        "stroke": C["arrow"],
        "stroke-width": 1.6,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})",
    }))


def _concat_label(heads: int | None, head_dim: int | None):
    if heads and head_dim:
        return ["Concat heads", f"{_fmt_int(heads)} x {_fmt_int(head_dim)}"]
    if heads:
        return ["Concat heads", f"{_fmt_int(heads)} heads"]
    return "Concat heads"


def _mlp_linear_label(name: str, in_dim: int | None, out_dim: int | None):
    if in_dim and out_dim:
        return [name, f"{_fmt_int(in_dim)} -> {_fmt_int(out_dim)}"]
    return name


def _head_dim(heads: int | None, hidden: int | None) -> int | None:
    if heads and hidden and hidden % heads == 0:
        return hidden // heads
    return None


def _patch_grid(parts: list[str], cx: float, y: float, patch: int | None, image_size: int | None) -> dict:
    """Draw a compact patch grid without turning every tile into a node."""
    cell = 24
    gap = 6
    cols = 5
    rows = 3
    grid_w = cols * cell + (cols - 1) * gap
    grid_h = rows * cell + (rows - 1) * gap
    panel_w = 304
    panel_h = 150
    x = cx - panel_w / 2
    x0 = cx - grid_w / 2
    tile_y = y + 38
    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": panel_w,
        "height": panel_h,
        "rx": 12,
        "ry": 12,
        "fill": "#FFFFFF",
        "stroke": C["border"],
        "stroke-width": 0.7,
    }))
    parts.append(_svg_text(cx, y + 22, _patch_grid_title(patch, image_size), {
        "text-anchor": "middle",
        "fill": C["text"],
        "font-family": "ui-monospace, \"JetBrains Mono\", \"SF Mono\", Menlo, monospace",
        "font-size": 11,
        "font-weight": 700,
    }))
    for row in range(rows):
        for col in range(cols):
            emphasis = row == rows - 1 and col == cols - 1
            parts.append(_svg_tag("rect", {
                "x": x0 + col * (cell + gap),
                "y": tile_y + row * (cell + gap),
                "width": cell,
                "height": cell,
                "rx": 5,
                "ry": 5,
                "fill": C["badge_bg"] if emphasis else "#F4FBF8",
                "stroke": "#1F9E78" if emphasis else C["border"],
                "stroke-width": 0.8,
            }))
    parts.append(_svg_text(cx, y + panel_h - 17, _patch_grid_subtitle(patch, image_size), {
        "text-anchor": "middle",
        "fill": C["muted"],
        "font-family": "ui-monospace, \"JetBrains Mono\", \"SF Mono\", Menlo, monospace",
        "font-size": 10,
    }))
    return {
        "left": x,
        "right": x + panel_w,
        "top": y,
        "bottom": y + panel_h,
        "cx": cx,
        "cy": y + panel_h / 2,
        "w": panel_w,
        "h": panel_h,
    }


def _encoder_stack(
    parts: list[str],
    info: dict,
    shadow_id: str,
    cx: float,
    y: float,
    layers: int | None,
    heads: int | None,
    hidden: int | None,
) -> dict:
    """Draw one repeated pre-norm ViT encoder block."""
    label = f"x {_fmt_int(layers)}" if layers else "repeat"
    region = {
        "left": cx - 280,
        "right": cx + 280,
        "top": y,
        "bottom": y + 470,
        "cx": cx,
        "cy": y + 235,
        "w": 560,
        "h": 470,
    }
    parts.append(_svg_tag("rect", {
        "x": region["left"],
        "y": region["top"],
        "width": region["w"],
        "height": region["h"],
        "rx": 18,
        "ry": 18,
        "fill": "#9FE1CB",
        "stroke": "none",
    }))
    parts.append(_svg_tag("rect", {
        "x": region["right"] - 70,
        "y": region["top"] + 18,
        "width": 52,
        "height": 24,
        "rx": 12,
        "ry": 12,
        "fill": "rgba(255,255,255,0.72)",
        "stroke": C["border"],
        "stroke-width": 0.5,
    }))
    parts.append(_svg_text(region["right"] - 44, region["top"] + 30, label, {
        "text-anchor": "middle",
        "dominant-baseline": "central",
        "fill": C["text"],
        "font-family": "\"Caveat\",\"Patrick Hand\",\"Comic Sans MS\",cursive",
        "font-size": 20,
    }))

    norm1 = _rect_block(parts, info, shadow_id, "vision_encoder_norm1", cx - 105, y + 390, 210, 42, "LayerNorm", font_size=16)
    attn = _rect_block(parts, info, shadow_id, "vision_encoder_attn", cx - 170, y + 300, 340, 60, _attention_label(heads, hidden), font_size=16)
    add1 = _plain_plus(parts, cx, y + 252)
    norm2 = _rect_block(parts, info, shadow_id, "vision_encoder_norm2", cx - 105, y + 182, 210, 42, "LayerNorm", font_size=16)
    mlp = _rect_block(parts, info, shadow_id, "vision_encoder_mlp", cx - 120, y + 104, 240, 52, "MLP", font_size=16)
    add2 = _plain_plus(parts, cx, y + 56)

    _up_arrow(parts, cx, region["bottom"], norm1["bottom"] + 12)
    _up_arrow(parts, norm1["cx"], norm1["top"], attn["bottom"] + 12)
    _up_arrow(parts, attn["cx"], attn["top"], add1["bottom"] + 12)
    _up_arrow(parts, add1["cx"], add1["top"], norm2["bottom"] + 12)
    _up_arrow(parts, norm2["cx"], norm2["top"], mlp["bottom"] + 12)
    _up_arrow(parts, mlp["cx"], mlp["top"], add2["bottom"] + 12)
    _v_stem(parts, add2["cx"], add2["top"], region["top"])

    _residual_to_plus(parts, cx, norm1["bottom"] + 16, region["right"] - 58, add1)
    _residual_to_plus(parts, cx, add1["top"] - 18, region["right"] - 90, add2)

    return region


def _projection_label(out_features: int | None):
    if out_features:
        return ["Linear / Conv2d", f"to {_fmt_int(out_features)}d"]
    return ["Linear / Conv2d", "projection"]


def _patch_grid_title(patch: int | None, image_size: int | None) -> str:
    if patch and image_size and image_size % patch == 0:
        side = image_size // patch
        return f"{_fmt_int(side)}x{_fmt_int(side)} patch grid"
    return "patch grid"


def _patch_grid_subtitle(patch: int | None, image_size: int | None) -> str:
    bits = []
    if patch:
        bits.append(f"{_fmt_int(patch)}px patch")
    if image_size:
        bits.append(f"{_fmt_int(image_size)}px image")
    return " from ".join(bits) if bits else "image split into patch tiles"


def _attention_label(heads: int | None, hidden: int | None):
    if heads and hidden:
        return ["Self-attention", f"{_fmt_int(heads)} heads · {_fmt_int(hidden)}d"]
    if heads:
        return ["Self-attention", f"{_fmt_int(heads)} heads"]
    return "Self-attention"


def _pos_label(pos: str | None):
    if pos:
        return ["Add positions", str(pos).replace("_", " ")]
    return "Add position embeddings"


def _plain_plus(parts: list[str], cx: float, cy: float) -> dict:
    r = 14
    parts.append(_svg_tag("circle", {
        "cx": cx,
        "cy": cy,
        "r": r,
        "fill": C["block"],
        "stroke": C["block_alt"],
        "stroke-width": 0.6,
    }))
    attrs = {
        "stroke": C["text_block"],
        "stroke-width": 2.2,
        "stroke-linecap": "round",
        "pointer-events": "none",
    }
    parts.append(_svg_tag("line", {"x1": cx - 5, "y1": cy, "x2": cx + 5, "y2": cy, **attrs}))
    parts.append(_svg_tag("line", {"x1": cx, "y1": cy - 5, "x2": cx, "y2": cy + 5, **attrs}))
    return {"left": cx - r, "right": cx + r, "top": cy - r, "bottom": cy + r, "cx": cx, "cy": cy, "r": r}


def _residual_to_plus(parts: list[str], start_x: float, start_y: float, lane_x: float, plus: dict) -> None:
    r = 10
    end_x = plus["right"] + 7
    end_y = plus["cy"]
    d = (
        f"M {start_x} {start_y} "
        f"L {lane_x - r} {start_y} "
        f"Q {lane_x} {start_y} {lane_x} {start_y - r} "
        f"L {lane_x} {end_y + r} "
        f"Q {lane_x} {end_y} {lane_x - r} {end_y} "
        f"L {end_x} {end_y}"
    )
    parts.append(_svg_tag("path", {
        "d": d,
        "fill": "none",
        "stroke": C["arrow"],
        "stroke-width": 1.4,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }))
    parts.append(_svg_tag("path", {
        "d": f"M {end_x + 6} {end_y - 5.5} L {end_x} {end_y} L {end_x + 6} {end_y + 5.5}",
        "fill": "none",
        "stroke": C["arrow"],
        "stroke-width": 1.6,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }))


def _v_stem(parts: list[str], x: float, y1: float, y2: float) -> None:
    parts.append(_svg_tag("line", {
        "x1": x,
        "y1": y1,
        "x2": x,
        "y2": y2,
        "stroke": C["arrow"],
        "stroke-width": 1.8,
        "stroke-linecap": "round",
        "fill": "none",
    }))


def _up_arrow(parts: list[str], x: float, y1: float, y2: float) -> None:
    parts.append(_svg_tag("line", {
        "x1": x,
        "y1": y1,
        "x2": x,
        "y2": y2,
        "stroke": C["arrow"],
        "stroke-width": 1.8,
        "stroke-linecap": "round",
        "fill": "none",
    }))
    parts.append(_svg_tag("path", {
        "d": f"M {x - 5.5} {y2 + 7} L {x} {y2} L {x + 5.5} {y2 + 7}",
        "fill": "none",
        "stroke": C["arrow"],
        "stroke-width": 1.8,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }))
