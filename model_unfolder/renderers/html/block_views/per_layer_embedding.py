"""Detail SVG for reusable Per-Layer Embedding blocks."""
from __future__ import annotations

from ..metadata import _block_label
from ..stack_view import fit_svg, point
from ..svg import (
    _ids,
    _plus_block,
    _rect_block,
    _svg_tag,
    _svg_text,
    _v_line,
)
from ..theme import C, FONT_MONO, GAP
from ..utils import _fmt_int


def build_per_layer_embedding_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """Render the canonical PLE gate -> multiply -> projection detail view.

    The adapter declares the child node ids and dimensions in ``block['detail']``.
    Any model family with the same part emits the same block contract.
    """
    detail = block.get("detail") or {}
    view_id = detail.get("view_id") or block.get("id") or "ple"
    arrow_id, shadow_id = _ids(mount_id, view_id)
    parts: list[str] = []

    ids = _node_ids(block)
    hidden_size = detail.get("hidden_size") or ir.get("hidden_size")
    embedding_dim = detail.get("embedding_dim") or (
        ((ir.get("extras") or {}).get("per_layer_embeddings") or {}).get("hidden")
    )
    cx = 0  # fit_svg translates + centres content; absolute centre is irrelevant
    base = 660  # vertical layout origin (canvas auto-fits, not this value)
    y_shift = -24

    gate = _rect_block(
        parts, info, shadow_id, ids["gate"],
        cx - 110, base - 160 + y_shift, 220, 50,
        _label(info, block, ids["gate"], "Linear (gate)"),
    )
    act = _rect_block(
        parts, info, shadow_id, ids["activation"],
        cx - 90, base - 250 + y_shift, 180, 44,
        _label(info, block, ids["activation"], "Activation"),
    )
    mul = _plus_block(parts, info, shadow_id, ids["multiply"], cx, base - 320 + y_shift, "×")
    proj = _rect_block(
        parts, info, shadow_id, ids["projection"],
        cx - 110, base - 410 + y_shift, 220, 50,
        _label(info, block, ids["projection"], "Linear (up)"),
    )
    norm = _rect_block(
        parts, info, shadow_id, ids["norm"],
        cx - 90, base - 500 + y_shift, 180, 44,
        _label(info, block, ids["norm"], "RMSNorm"),
    )

    for src, dst in ((gate, act), (act, mul), (mul, proj), (proj, norm)):
        parts.append(_v_line(src, dst, arrow_id))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": norm["top"],
        "x2": cx, "y2": norm["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, norm["top"] - 46,
        detail.get("output_label") or "out  -> add (residual)",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": gate["bottom"] + 38,
        "x2": cx, "y2": gate["bottom"] + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, gate["bottom"] + 56,
        detail.get("input_label") or "in  (hidden)",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    external = _external_tensor_block(
        parts,
        shadow_id,
        detail.get("pathway_id") or "per_layer_input",
        cx + 82,
        mul["cy"] - 21,
        178,
        42,
        detail.get("external_label") or "per_layer_input[L]",
    )
    parts.append(_svg_tag("line", {
        "x1": external["left"] - GAP, "y1": external["cy"],
        "x2": mul["cx"] + mul["r"] + GAP, "y2": mul["cy"],
        "stroke": "#1F9E78", "stroke-width": 1.6, "stroke-linecap": "round",
        "stroke-dasharray": "5 4",
        "marker-end": f"url(#{arrow_id})",
    }))
    parts.append(_svg_text(
        external["cx"], external["bottom"] + 16,
        "(outside stack)",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 9},
    ))

    parts.append(_svg_text(
        gate["right"] + 14, gate["cy"],
        f"{_fmt_int(hidden_size)}  ->  {_fmt_int(embedding_dim)}" if embedding_dim else "",
        {"dominant-baseline": "central", "fill": C["muted"],
         "font-family": FONT_MONO, "font-size": 10},
    ))
    parts.append(_svg_text(
        proj["right"] + 14, proj["cy"],
        f"{_fmt_int(embedding_dim)}  ->  {_fmt_int(hidden_size)}" if embedding_dim else "",
        {"dominant-baseline": "central", "fill": C["muted"],
         "font-family": FONT_MONO, "font-size": 10},
    ))

    regions = [
        gate, act, mul, proj, norm, external,
        point(cx, norm["top"] - 58),          # out-label above norm
        point(cx, gate["bottom"] + 64),       # in-label below gate
        point(external["cx"], external["bottom"] + 22),  # "(outside stack)" label
    ]
    return fit_svg(
        arrow_id, shadow_id, parts, regions,
        f"{ir.get('name', 'model')} per-layer embeddings block",
    )


def _node_ids(block: dict) -> dict[str, str]:
    ids = ((block.get("detail") or {}).get("nodes") or {}).copy()
    block_id = block.get("id") or "ple"
    ids.setdefault("gate", f"{block_id}_gate")
    ids.setdefault("activation", f"{block_id}_act")
    ids.setdefault("multiply", f"{block_id}_mul")
    ids.setdefault("projection", f"{block_id}_proj")
    ids.setdefault("norm", f"{block_id}_norm")
    return ids


def _external_tensor_block(
    parts: list[str],
    shadow_id: str,
    node_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
) -> dict:
    children = [
        _svg_tag("rect", {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rx": 10,
            "ry": 10,
            "fill": C["badge_bg"],
            "stroke": "#1F9E78",
            "stroke-width": 1,
            "stroke-dasharray": "4 3",
            "filter": f"url(#{shadow_id})",
        }),
        _svg_text(
            x + w / 2,
            y + h / 2,
            label,
            {
                "text-anchor": "middle",
                "dominant-baseline": "central",
                "fill": "#0F6E56",
                "font-family": FONT_MONO,
                "font-size": 12,
                "font-weight": 700,
                "pointer-events": "none",
            },
        ),
    ]
    parts.append(_svg_tag("g", {"class": "uf-node uf-external-tensor", "data-id": node_id}, "".join(children)))
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


def _label(info: dict, block: dict, node_id: str, default: str) -> str:
    label = _block_label(info, node_id, None)
    if label:
        return label
    for child in block.get("children") or []:
        if child.get("id") == node_id:
            return child.get("label") or default
    return default
