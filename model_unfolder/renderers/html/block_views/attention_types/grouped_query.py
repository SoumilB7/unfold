"""Grouped-query attention detail view."""
from __future__ import annotations

from ...svg import (
    _branch_dot,
    _defs,
    _elbow_hv,
    _elbow_vh,
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
from .common import (
    gqa_grouping_panel,
    input_to_block,
    kv_cache_badge,
    output_stem,
    placed_figure,
    queries_per_kv_group,
    sdpa_dot_operator,
    sdpa_fraction_block,
)


def build(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for grouped-query attention."""
    w, h = 820, 880
    arrow_id, shadow_id = _ids(mount_id, "gqa-attn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))
    body: list[str] = []

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden_sz = ir.get("hidden_size") or 0
    hidden = _fmt_int(hidden_sz)
    num_heads = attn.get("num_heads") or 0
    num_kv_heads = attn.get("num_kv_heads") or num_heads
    head_dim = attn.get("head_dim") or (hidden_sz // num_heads if num_heads else 0)
    d_k = str(head_dim) if head_dim else "d_k"
    q_per_group = queries_per_kv_group(num_heads, num_kv_heads)

    cx = w / 2
    o_proj = _rect_block(body, info, shadow_id, "o_proj", cx - 100, 72, 200, 52, "Linear (out)")
    concat = _rect_block(
        body,
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
    value_dot = sdpa_dot_operator(body, info, shadow_id, "attn_apply_v", cx, 276)
    softmax = _rect_block(body, info, shadow_id, "attn_softmax", cx - 96, 344, 192, 52, "Softmax")
    scaled_scores = sdpa_fraction_block(body, info, shadow_id, "scaled_scores", cx - 140, 452, 280, 82)
    gqa_grouping_panel(body, 64, 92, 220, 218, num_heads, num_kv_heads, q_per_group)

    body.append(_v_line(scaled_scores, softmax, arrow_id))
    body.append(_v_line(softmax, value_dot, arrow_id))
    body.append(_v_line(value_dot, concat, arrow_id))
    body.append(_v_line(concat, o_proj, arrow_id))

    proj_w, proj_h, proj_y = 185, 52, 704
    q_proj = _rect_block(body, info, shadow_id, "q_proj", 78, proj_y, proj_w, proj_h, ["Linear (Q)", f"{num_heads} heads"], font_size=15)
    kv_head_label = f"{num_kv_heads} head" if num_kv_heads == 1 else f"{num_kv_heads} heads"
    k_proj = _rect_block(body, info, shadow_id, "k_proj", cx - proj_w / 2, proj_y, proj_w, proj_h, ["Linear (K)", kv_head_label], font_size=15)
    v_proj = _rect_block(body, info, shadow_id, "v_proj", w - 78 - proj_w, proj_y, proj_w, proj_h, ["Linear (V)", kv_head_label], font_size=15)

    branch_x, branch_y = cx, 792
    body.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 34, "x2": branch_x, "y2": branch_y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "fill": "none",
    }))
    body.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    body.append(_elbow_hv(branch_x, branch_y, k_proj["cx"], k_proj["bottom"] + GAP, arrow_id))
    body.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))
    body.append(_branch_dot(branch_x, branch_y))

    body.append(input_to_block(q_proj["cx"], q_proj["top"], scaled_scores["left"] + 92, scaled_scores["bottom"], arrow_id))
    body.append(_v_seg(k_proj["cx"], k_proj["top"], scaled_scores["bottom"], arrow_id))
    body.append(_elbow_vh(v_proj["cx"], v_proj["top"], value_dot["right"] + GAP, value_dot["cy"], arrow_id))
    output_stem(body, cx, o_proj, arrow_id, hidden, show_label=False)

    if q_per_group and q_per_group > 1:
        kv_cache_badge(body, w - 218, 58, f"KV cache {q_per_group}x smaller", "than full MHA")

    parts.append(placed_figure(body, "gqa"))
    return _svg(w, h, f"{ir.get('name', 'model')} grouped-query attention", parts)
