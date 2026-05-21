"""Multi-query attention detail view."""
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
    input_to_block,
    kv_cache_badge,
    kv_cache_port_hint,
    mqa_shared_kv_node,
    output_stem,
    sdpa_dot_operator,
    sdpa_fraction_block,
)
from .sliding_window import canvas_height, is_sliding_window, sliding_window_input


def build(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for multi-query attention / single shared K/V attention."""
    attn = info["dominant"]["spec"].get("attention") or {}
    is_sliding = is_sliding_window(attn)
    w, h = 820, canvas_height(attn, 920, extra_height=60)
    arrow_id, shadow_id = _ids(mount_id, "mqa-attn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))
    body: list[str] = []

    hidden_sz = ir.get("hidden_size") or 0
    hidden = _fmt_int(hidden_sz)
    num_heads = attn.get("num_heads") or 0
    head_dim = attn.get("head_dim") or (hidden_sz // num_heads if num_heads else 0)
    d_k = str(head_dim) if head_dim else "d_k"
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
    shared_kv = mqa_shared_kv_node(body, cx - 125, 602, 250, 56, num_heads)

    body.append(_v_line(shared_kv, scaled_scores, arrow_id))
    body.append(_elbow_vh(shared_kv["right"] - 42, shared_kv["top"], value_dot["right"] + GAP, value_dot["cy"], arrow_id))
    body.append(_v_line(scaled_scores, softmax, arrow_id))
    body.append(_v_line(softmax, value_dot, arrow_id))
    body.append(_v_line(value_dot, concat, arrow_id))
    body.append(_v_line(concat, o_proj, arrow_id))

    proj_w, proj_h, proj_y = 185, 52, 742
    q_proj = _rect_block(body, info, shadow_id, "q_proj", 78, proj_y, proj_w, proj_h, ["Linear (Q)", f"{num_heads} heads"], font_size=15)
    k_proj = _rect_block(body, info, shadow_id, "k_proj", cx - proj_w / 2, proj_y, proj_w, proj_h, ["Linear (K)", "1 head"], font_size=15)
    v_proj = _rect_block(body, info, shadow_id, "v_proj", w - 78 - proj_w, proj_y, proj_w, proj_h, ["Linear (V)", "1 head"], font_size=15)
    kv_cache_port_hint(body, [k_proj, v_proj])

    branch_x, branch_y = cx, 834
    body.append(_branch_dot(branch_x, branch_y))
    if is_sliding:
        sliding_window_input(body, arrow_id, branch_x, branch_y, attn.get("window_size"))
    else:
        body.append(_svg_tag("line", {
            "x1": branch_x, "y1": branch_y + 34, "x2": branch_x, "y2": branch_y + 8,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none",
        }))

    body.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    body.append(_elbow_hv(branch_x, branch_y, k_proj["cx"], k_proj["bottom"] + GAP, arrow_id))
    body.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))

    body.append(input_to_block(q_proj["cx"], q_proj["top"], scaled_scores["left"] + 92, scaled_scores["bottom"], arrow_id))
    body.append(_v_seg(k_proj["cx"], k_proj["top"], shared_kv["bottom"], arrow_id))
    body.append(input_to_block(v_proj["cx"], v_proj["top"], shared_kv["right"] - 62, shared_kv["bottom"], arrow_id, lane_offset=22))
    output_stem(body, cx, o_proj, arrow_id, hidden, show_label=False)

    if num_heads and num_heads > 1:
        kv_cache_badge(body, w - 218, 58, f"KV cache {num_heads}x smaller", "than full MHA")

    parts.extend(body)
    return _svg(w, h, f"{ir.get('name', 'model')} multi-query attention", parts)
