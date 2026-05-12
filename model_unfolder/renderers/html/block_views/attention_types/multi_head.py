"""Plain multi-head scaled dot-product attention detail view."""
from __future__ import annotations

from ...svg import (
    _branch_dot,
    _defs,
    _elbow_hv,
    _ids,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _v_line,
    _v_seg,
)
from ...theme import C, GAP
from ...utils import _fmt_int
from .common import output_stem


def build(ir: dict, info: dict, mount_id: str) -> str:
    """Rich SVG detail view for standard MHA / SDPA attention blocks."""
    w, h = 720, 590
    arrow_id, shadow_id = _ids(mount_id, "attn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden_sz = ir.get("hidden_size") or 0
    num_heads = attn.get("num_heads") or 0
    num_kv_heads = attn.get("num_kv_heads") or num_heads
    head_dim = attn.get("head_dim") or (hidden_sz // num_heads if num_heads else 0)
    d_k = str(head_dim) if head_dim else "d_k"

    cx = w / 2
    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 90, 100, 180, 50, "Linear (out)")

    if num_heads and head_dim:
        attn_kind = attn.get("kind", "mha")
        if attn_kind == "gqa" and num_kv_heads and num_kv_heads < num_heads:
            subtitle = f"{num_heads}Q / {num_kv_heads} KV heads  ·  d_k = {d_k}"
        elif attn_kind == "mqa":
            subtitle = f"{num_heads}Q / 1 KV head (MQA)  ·  d_k = {d_k}"
        else:
            subtitle = f"{num_heads} heads  ·  d_k = {d_k}"
        sdpa_label: str | list = ["Scaled Dot-Product Attention", subtitle]
    else:
        sdpa_label = "Scaled Dot-Product Attention"
    sdpa = _rect_block(parts, info, shadow_id, "qkv_dot", 80, 200, 560, 54, sdpa_label, font_size=15)

    proj_w, proj_h, proj_y = 165, 50, 315
    q_proj = _rect_block(parts, info, shadow_id, "q_proj", 70, proj_y, proj_w, proj_h, "Linear (Q)")
    k_proj = _rect_block(parts, info, shadow_id, "k_proj", 278, proj_y, proj_w, proj_h, "Linear (K)")
    v_proj = _rect_block(parts, info, shadow_id, "v_proj", 486, proj_y, proj_w, proj_h, "Linear (V)")

    branch_x, branch_y = cx, 475
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 38, "x2": branch_x, "y2": branch_y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "fill": "none",
    }))
    parts.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(branch_x, branch_y, k_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))
    parts.append(_branch_dot(branch_x, branch_y))

    sdpa_entry_gap = sdpa["bottom"] + GAP
    parts.append(_v_seg(q_proj["cx"], q_proj["top"], sdpa_entry_gap, arrow_id))
    parts.append(_v_seg(k_proj["cx"], k_proj["top"], sdpa_entry_gap, arrow_id))
    parts.append(_v_seg(v_proj["cx"], v_proj["top"], sdpa_entry_gap, arrow_id))
    parts.append(_v_line(sdpa, o_proj, arrow_id))
    output_stem(parts, cx, o_proj, arrow_id, _fmt_int(hidden_sz), show_label=False)

    return _svg(w, h, f"{ir.get('name', 'model')} attention", parts)
