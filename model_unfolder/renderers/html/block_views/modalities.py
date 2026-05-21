"""Detail SVGs for model-level multimodal input pathways."""
from __future__ import annotations

from ..svg import (
    _defs,
    _elbow_vh,
    _ids,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
)
from ..theme import C, FONT_HEAD, FONT_MONO, GAP
from ..utils import _fmt_int


def build_vision_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Vision encoder -> projector -> soft visual tokens."""
    w, h = 720, 560
    arrow_id, shadow_id = _ids(mount_id, "vision-path")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    vision = _vision(ir)
    encoder = vision.get("encoder") or {}
    projector = vision.get("projector") or {}
    tokens = vision.get("tokens") or {}
    input_spec = vision.get("input") or {}
    cx = w / 2

    pixels = _rect_block(parts, info, shadow_id, "vision_pixels", cx - 105, 455, 210, 44, "Image pixels")
    patches = _rect_block(
        parts, info, shadow_id, "vision_patches", cx - 115, 365, 230, 44,
        _patch_label(input_spec),
    )
    enc = _rect_block(
        parts, info, shadow_id, "vision_encoder", cx - 150, 260, 300, 54,
        _encoder_label(encoder),
    )
    proj = _rect_block(
        parts, info, shadow_id, "vision_projector", cx - 135, 155, 270, 48,
        _projector_label(projector),
    )
    soft = _rect_block(
        parts, info, shadow_id, "visual_tokens", cx - 145, 70, 290, 48,
        _visual_token_label(tokens),
    )

    for src, dst in ((pixels, patches), (patches, enc), (enc, proj), (proj, soft)):
        parts.append(_v_line(src, dst, arrow_id))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": soft["top"],
        "x2": cx, "y2": soft["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    return _svg(w, h, f"{ir.get('name', 'model')} vision pathway", parts)


def build_multimodal_fusion_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Show visual soft tokens replacing placeholder slots in the text stream."""
    w, h = 760, 600
    arrow_id, shadow_id = _ids(mount_id, "multimodal-fusion")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    fusion = _fusion(ir)
    vision_spec = _vision(ir)
    tokens = vision_spec.get("tokens") or {}
    token_count = tokens.get("count")
    token_width = tokens.get("width")
    visual_shape = (
        f"{_fmt_int(token_count)} x {_fmt_int(token_width)}"
        if token_count and token_width
        else (f"{_fmt_int(token_count)} tokens" if token_count else "visual tokens")
    )

    cx = w / 2
    surface = {"left": 70, "top": 160, "right": 690, "bottom": 385, "cx": cx, "cy": 272}
    text = _rect_block(
        parts, info, shadow_id, "embed",
        105, 465, 250, 50,
        "Text embeddings",
        font_size=20,
    )
    vision = _rect_block(
        parts, info, shadow_id, "vision_path",
        405, 465, 250, 50,
        "Visual tokens",
        font_size=20,
    )
    stack = _rect_block(
        parts, info, shadow_id, "stack_input",
        cx - 150, 70, 300, 50,
        "Decoder input",
        font_size=20,
    )

    _fusion_surface(parts, fusion, surface, token_count)
    parts.append(_svg_tag("line", {
        "x1": text["cx"], "y1": text["top"],
        "x2": text["cx"], "y2": surface["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_tag("line", {
        "x1": vision["cx"], "y1": vision["top"],
        "x2": vision["cx"], "y2": surface["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": surface["top"],
        "x2": cx, "y2": stack["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    return _svg(w, h, f"{ir.get('name', 'model')} multimodal fusion", parts)


def _vision(ir: dict) -> dict:
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    return ((modalities.get("inputs") or {}).get("vision") or {})


def _fusion(ir: dict) -> dict:
    return (((ir.get("extras") or {}).get("modalities") or {}).get("fusion") or {})


def _patch_label(input_spec: dict) -> str | list[str]:
    patch = input_spec.get("patch_size")
    image = input_spec.get("image_size")
    if patch and image:
        return [f"Patchify {patch}x{patch}", f"{_fmt_int(image)} px image"]
    if patch:
        return f"Patchify {patch}x{patch}"
    return "Patchify image"


def _encoder_label(encoder: dict) -> list[str]:
    kind = str(encoder.get("kind") or "vision transformer").replace("_", " ")
    layers = encoder.get("num_layers")
    heads = encoder.get("num_attention_heads")
    line2 = []
    if layers:
        line2.append(f"{_fmt_int(layers)} layers")
    if heads:
        line2.append(f"{_fmt_int(heads)} heads")
    return [kind, " · ".join(line2)] if line2 else [kind]


def _projector_label(projector: dict) -> str | list[str]:
    in_features = projector.get("in_features")
    out_features = projector.get("out_features")
    if in_features and out_features:
        return ["Projector", f"{_fmt_int(in_features)} -> {_fmt_int(out_features)}"]
    return "Projector"


def _visual_token_label(tokens: dict) -> str | list[str]:
    count = tokens.get("count")
    width = tokens.get("width")
    if count and width:
        return ["Soft visual tokens", f"{_fmt_int(count)} x {_fmt_int(width)}"]
    if count:
        return ["Soft visual tokens", f"{_fmt_int(count)} tokens"]
    return "Soft visual tokens"


def _fusion_label(fusion: dict) -> str | list[str]:
    kind = str(fusion.get("kind") or "fusion").replace("_", " ")
    placeholder = fusion.get("placeholder") or {}
    token_id = placeholder.get("token_id")
    if token_id is not None:
        return [kind, f"replace image token {_fmt_int(token_id)}"]
    return kind


def _fusion_surface(parts: list[str], fusion: dict, box: dict, token_count: int | None) -> None:
    """Draw Gemma-style BOI/image-token/EOI placeholder replacement."""
    parts.append(_svg_tag("rect", {
        "x": box["left"], "y": box["top"],
        "width": box["right"] - box["left"],
        "height": box["bottom"] - box["top"],
        "rx": 15, "ry": 15,
        "fill": C["bg_card"],
        "stroke": C["border"],
        "stroke-width": 0.8,
    }))
    kind = str(fusion.get("kind") or "modality_token_fusion")
    title = {
        "placeholder_replace": "scatter vision features into image-token slots",
        "prefix_soft_tokens": "prepend visual soft tokens",
        "unified_multimodal_stream": "interleave multimodal tokens",
        "cross_attention": "condition text through visual context",
    }.get(kind, "compose multimodal token stream")
    parts.append(_svg_text(
        box["cx"], box["top"] + 28,
        title,
        {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_HEAD, "font-size": 20},
    ))

    row_x = box["left"] + 88
    text_y = box["top"] + 62
    vision_y = box["top"] + 118
    mixed_y = box["top"] + 174
    gap = 8
    tok_w = 46
    guard_w = 46
    visual_span_w = 220
    last_visual = f"v{int(token_count) - 1}" if token_count else "vN"

    tok1_x = row_x
    tok2_x = tok1_x + tok_w + gap
    boi_x = tok2_x + tok_w + gap
    img_x = boi_x + guard_w + gap
    eoi_x = img_x + visual_span_w + gap
    tok3_x = eoi_x + guard_w + gap

    _row_label(parts, box["left"] + 34, text_y + 14, "text")
    _slot(parts, tok1_x, text_y, tok_w, "tok", node_id="fusion_text_tokens")
    _slot(parts, tok2_x, text_y, tok_w, "tok", node_id="fusion_text_tokens")
    _slot(parts, boi_x, text_y, guard_w, "BOI", node_id="fusion_boi")
    _slot(
        parts,
        img_x,
        text_y,
        visual_span_w,
        f"IMG x {_fmt_int(token_count) if token_count else 'n'}",
        emphasis=True,
        node_id="fusion_image_slots",
    )
    _slot(parts, eoi_x, text_y, guard_w, "EOI", node_id="fusion_eoi")
    _slot(parts, tok3_x, text_y, tok_w, "tok", node_id="fusion_text_tokens")

    _row_label(parts, box["left"] + 34, vision_y + 14, "vision")
    _slot(parts, img_x, vision_y, 44, "v0", emphasis=True, node_id="fusion_vision_tokens")
    _slot(parts, img_x + 52, vision_y, 44, "v1", emphasis=True, node_id="fusion_vision_tokens")
    _slot(parts, img_x + 104, vision_y, 44, "...", node_id="fusion_vision_tokens")
    _slot(parts, img_x + 156, vision_y, 64, last_visual, emphasis=True, node_id="fusion_vision_tokens")

    _row_label(parts, box["left"] + 34, mixed_y + 14, "mixed")
    _slot(parts, tok1_x, mixed_y, tok_w, "tok", node_id="fusion_mixed_stream")
    _slot(parts, tok2_x, mixed_y, tok_w, "tok", node_id="fusion_mixed_stream")
    _slot(parts, boi_x, mixed_y, guard_w, "BOI", node_id="fusion_boi")
    _slot(parts, img_x, mixed_y, 44, "v0", emphasis=True, node_id="fusion_mixed_stream")
    _slot(parts, img_x + 52, mixed_y, 44, "v1", emphasis=True, node_id="fusion_mixed_stream")
    _slot(parts, img_x + 104, mixed_y, 44, "...", node_id="fusion_mixed_stream")
    _slot(parts, img_x + 156, mixed_y, 64, last_visual, emphasis=True, node_id="fusion_mixed_stream")
    _slot(parts, eoi_x, mixed_y, guard_w, "EOI", node_id="fusion_eoi")
    _slot(parts, tok3_x, mixed_y, tok_w, "tok", node_id="fusion_mixed_stream")


def _row_label(parts: list[str], x: float, y: float, label: str) -> None:
    parts.append(_svg_text(
        x, y, label,
        {
            "dominant-baseline": "central",
            "fill": C["muted"],
            "font-family": FONT_MONO,
            "font-size": 10,
            "font-weight": 700,
            "letter-spacing": "0.08em",
        },
    ))


def _slot(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    label: str,
    emphasis: bool = False,
    node_id: str | None = None,
) -> None:
    fill = C["badge_bg"] if emphasis else "#F4FBF8"
    stroke = "#1F9E78" if emphasis else C["border"]
    children = [_svg_tag("rect", {
        "x": x, "y": y, "width": w, "height": 28,
        "rx": 7, "ry": 7,
        "fill": fill,
        "stroke": stroke,
        "stroke-width": 0.8,
    })]
    children.append(_svg_text(
        x + w / 2, y + 14, label,
        {
            "text-anchor": "middle",
            "dominant-baseline": "central",
            "fill": C["text"] if emphasis else C["muted"],
            "font-family": FONT_MONO,
            "font-size": 9,
            "font-weight": 700 if emphasis else 500,
        },
    ))
    if node_id:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)
