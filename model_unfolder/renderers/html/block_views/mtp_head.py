"""Detail SVG for the Multi-Token Prediction (MTP) head stack."""
from __future__ import annotations

from ..metadata import _block_label
from ..stack_view import fit_svg, point
from ..svg import _elbow_vh, _ids, _rect_block, _svg_tag, _svg_text, _v_line
from ..theme import C, FONT_MONO
from .modality_views.vision_details import (
    _plain_plus,
    _residual_to_plus,
    _tower_badge,
    _up_arrow,
    _v_stem,
)


def build_mtp_head_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One MTP module, drawn bottom-to-top with its two inputs merging.

    Two inputs — the trunk's previous hidden state (left) and the next-token
    embedding (right) — are each RMSNorm'd, concatenated, projected ``2d -> d``
    (``eh_proj``), passed through one transformer block, then the shared output
    head. Every box is a clickable child of the MTP block.
    """
    detail = block.get("detail") or {}
    n = detail.get("num_modules") or 1
    arrow_id, shadow_id = _ids(mount_id, "mtp-head")
    parts: list[str] = []

    def lbl(node_id: str, default):
        return _block_label(info, node_id, default)

    # Centre column (merge -> output) and two input branches either side.
    head   = _rect_block(parts, info, shadow_id, "mtp_head",   -120, -360, 240, 46, lbl("mtp_head", "Shared output head"))
    tblock = _rect_block(parts, info, shadow_id, "mtp_block",  -120, -268, 240, 54, lbl("mtp_block", "Transformer block"))
    proj   = _rect_block(parts, info, shadow_id, "mtp_proj",   -110, -176, 220, 46, lbl("mtp_proj", "Linear  2d -> d"))
    concat = _rect_block(parts, info, shadow_id, "mtp_concat", -130,  -86, 260, 50, lbl("mtp_concat", "Concat"))
    hnorm  = _rect_block(parts, info, shadow_id, "mtp_hnorm",  -310,    0, 160, 46, lbl("mtp_hnorm", ["RMSNorm", "(hidden)"]))
    enorm  = _rect_block(parts, info, shadow_id, "mtp_enorm",   150,    0, 160, 46, lbl("mtp_enorm", ["RMSNorm", "(embedding)"]))
    emb    = _rect_block(parts, info, shadow_id, "mtp_emb",     135,   86, 190, 46, lbl("mtp_emb", ["Next-token", "embedding"]))

    # Centre flow (bottom -> top) and the right embedding branch.
    parts.append(_v_line(concat, proj, arrow_id))
    parts.append(_v_line(proj, tblock, arrow_id))
    parts.append(_v_line(tblock, head, arrow_id))
    parts.append(_v_line(emb, enorm, arrow_id))

    # Both norms elbow into the concat (left + right edges).
    parts.append(_elbow_vh(hnorm["cx"], hnorm["top"], concat["left"], concat["cy"], arrow_id))
    parts.append(_elbow_vh(enorm["cx"], enorm["top"], concat["right"], concat["cy"], arrow_id))

    # Output arrow + label above the shared head.
    parts.append(_svg_tag("line", {
        "x1": head["cx"], "y1": head["top"], "x2": head["cx"], "y2": head["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        head["cx"], head["top"] - 42, "logits  ->  token t+k+1",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Input caption under the hidden-state branch.
    parts.append(_svg_text(
        hnorm["cx"], hnorm["bottom"] + 16, "prev hidden state",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
    ))

    if n > 1:
        parts.append(_svg_text(
            head["right"] + 14, head["cy"], f"x{n} modules",
            {"dominant-baseline": "central", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
        ))

    regions = [
        head, tblock, proj, concat, hnorm, enorm, emb,
        point(head["cx"], head["top"] - 48),       # output label
        point(hnorm["cx"], hnorm["bottom"] + 22),  # input caption
    ]
    return fit_svg(arrow_id, shadow_id, parts, regions, f"{ir.get('name', 'model')} MTP head")


def build_mtp_transformer_block_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The MTP module's transformer block, drawn as its own tower.

    It is one decoder block of the same shape as the main stack, so the
    attention and FFN labels are pulled from the dominant layer's blocks
    (e.g. Multi-Head Latent Attention + MoE for DeepSeek-V3)."""
    spec = (info.get("dominant") or {}).get("spec") or {}
    layer_blocks = spec.get("blocks") or []
    attn_label = next((b.get("label") for b in layer_blocks if b.get("id") == "attn"), None) or "Attention"
    ffn_label = next((b.get("label") for b in layer_blocks if b.get("id") == "ffn"), None) or "Feed-Forward"

    arrow_id, shadow_id = _ids(mount_id, "mtp-transformer-block")
    parts: list[str] = []
    cx = 0

    region = {"left": cx - 230, "right": cx + 230, "top": 40, "bottom": 514,
              "w": 460, "h": 474, "cx": cx, "cy": 277}
    parts.append(_svg_tag("rect", {
        "x": region["left"], "y": region["top"], "width": region["w"], "height": region["h"],
        "rx": 18, "ry": 18, "fill": "#9FE1CB", "stroke": "none",
    }))
    _tower_badge(parts, region["right"] - 168, region["top"] + 16)

    # Uniform 30px gaps between sublayers, balanced 34px margins inside the tower.
    norm1 = _rect_block(parts, info, shadow_id, "mtp_block_norm1", cx - 105, 438, 210, 42, "RMSNorm", font_size=16)
    attn  = _rect_block(parts, info, shadow_id, "mtp_block_attn", cx - 175, 348, 350, 60, attn_label, font_size=16)
    add1  = _plain_plus(parts, cx, 304)
    norm2 = _rect_block(parts, info, shadow_id, "mtp_block_norm2", cx - 105, 218, 210, 42, "RMSNorm", font_size=16)
    ffn   = _rect_block(parts, info, shadow_id, "mtp_block_ffn", cx - 120, 132, 240, 56, ffn_label, font_size=16)
    add2  = _plain_plus(parts, cx, 88)

    _up_arrow(parts, cx, region["bottom"], norm1["bottom"] + 12)
    _up_arrow(parts, norm1["cx"], norm1["top"], attn["bottom"] + 12)
    _up_arrow(parts, attn["cx"], attn["top"], add1["bottom"] + 12)
    _up_arrow(parts, add1["cx"], add1["top"], norm2["bottom"] + 12)
    _up_arrow(parts, norm2["cx"], norm2["top"], ffn["bottom"] + 12)
    _up_arrow(parts, ffn["cx"], ffn["top"], add2["bottom"] + 12)
    _v_stem(parts, add2["cx"], add2["top"], region["top"])
    _up_arrow(parts, cx, region["top"], region["top"] - 34)

    _residual_to_plus(parts, cx, norm1["bottom"] + 16, region["right"] - 40, add1)
    _residual_to_plus(parts, cx, add1["top"] - 16, region["right"] - 72, add2)

    parts.append(_svg_text(cx, region["bottom"] + 22, "from eh_proj  (d)",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10}))
    parts.append(_svg_text(cx, region["top"] - 44, "to shared output head",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10}))

    regions = [
        region, norm1, attn, add1, norm2, ffn, add2,
        point(cx, region["top"] - 48),
        point(cx, region["bottom"] + 30),
    ]
    return fit_svg(arrow_id, shadow_id, parts, regions, f"{ir.get('name', 'model')} MTP transformer block")
