"""Detail SVG for reusable Per-Layer Embedding blocks."""
from __future__ import annotations

from ..metadata import _block_label
from ..svg import (
    _defs,
    _ids,
    _plus_block,
    _rect_block,
    _region_rect,
    _svg,
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
    w, h = 720, 660
    detail = block.get("detail") or {}
    view_id = detail.get("view_id") or block.get("id") or "ple"
    arrow_id, shadow_id = _ids(mount_id, view_id)
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ids = _node_ids(block)
    hidden_size = detail.get("hidden_size") or ir.get("hidden_size")
    embedding_dim = detail.get("embedding_dim") or (
        ((ir.get("extras") or {}).get("per_layer_embeddings") or {}).get("hidden")
    )
    cx = w / 2

    gate = _rect_block(
        parts, info, shadow_id, ids["gate"],
        cx - 110, h - 160, 220, 50,
        _label(info, block, ids["gate"], "Linear (gate)"),
    )
    act = _rect_block(
        parts, info, shadow_id, ids["activation"],
        cx - 90, h - 250, 180, 44,
        _label(info, block, ids["activation"], "Activation"),
    )
    mul = _plus_block(parts, info, shadow_id, ids["multiply"], cx, h - 320, "×")
    proj = _rect_block(
        parts, info, shadow_id, ids["projection"],
        cx - 110, h - 410, 220, 50,
        _label(info, block, ids["projection"], "Linear (up)"),
    )
    norm = _rect_block(
        parts, info, shadow_id, ids["norm"],
        cx - 90, h - 500, 180, 44,
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

    feed_x = mul["cx"] + mul["r"] + 24
    parts.append(_svg_tag("line", {
        "x1": feed_x + 110, "y1": mul["cy"],
        "x2": mul["cx"] + mul["r"] + GAP, "y2": mul["cy"],
        "stroke": "#1F9E78", "stroke-width": 1.6, "stroke-linecap": "round",
        "stroke-dasharray": "5 4",
        "marker-end": f"url(#{arrow_id})",
    }))
    parts.append(_svg_text(
        feed_x + 116, mul["cy"] - 10,
        detail.get("external_label") or "per_layer_input[L]",
        {"fill": "#1F9E78", "font-family": FONT_MONO, "font-size": 11, "font-weight": 700},
    ))
    parts.append(_svg_text(
        feed_x + 116, mul["cy"] + 6,
        detail.get("external_description") or f"({_fmt_int(embedding_dim)}-d, built outside layers)",
        {"fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
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

    return _svg(w, h, f"{ir.get('name', 'model')} per-layer embeddings block", parts)


def _node_ids(block: dict) -> dict[str, str]:
    ids = ((block.get("detail") or {}).get("nodes") or {}).copy()
    block_id = block.get("id") or "ple"
    ids.setdefault("gate", f"{block_id}_gate")
    ids.setdefault("activation", f"{block_id}_act")
    ids.setdefault("multiply", f"{block_id}_mul")
    ids.setdefault("projection", f"{block_id}_proj")
    ids.setdefault("norm", f"{block_id}_norm")
    return ids


def _label(info: dict, block: dict, node_id: str, default: str) -> str:
    label = _block_label(info, node_id, None)
    if label:
        return label
    for child in block.get("children") or []:
        if child.get("id") == node_id:
            return child.get("label") or default
    return default
