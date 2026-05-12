"""Multi-query attention detail view."""
from __future__ import annotations

from ...svg import (
    _branch_dot,
    _defs,
    _elbow_hv,
    _elbow_vh,
    _ids,
    _rect_block,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
)
from ...theme import C, FONT_MONO, GAP
from ...utils import _fmt_int
from .common import attn_dim_label, dynamic_region_rect, kv_cache_badge, mqa_shared_kv_node, output_stem


def build(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for multi-query attention / single shared K/V attention."""
    w, h = 780, 604
    dx = (w - 720) / 2
    arrow_id, shadow_id = _ids(mount_id, "mqa-attn")
    parts = [_defs(arrow_id, shadow_id)]
    body: list[str] = []

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden_sz = ir.get("hidden_size") or 0
    hidden = _fmt_int(hidden_sz)
    num_heads = attn.get("num_heads") or 0
    head_dim = attn.get("head_dim") or (hidden_sz // num_heads if num_heads else 0)
    q_out = _fmt_int(num_heads * head_dim) if (num_heads and head_dim) else hidden
    kv_out = _fmt_int(head_dim) if head_dim else hidden
    d_k = str(head_dim) if head_dim else "d_k"
    cx = w / 2

    o_proj = _rect_block(body, info, shadow_id, "o_proj", cx - 92, 82, 184, 50, "Linear (out)")
    sdpa = _rect_block(
        body,
        info,
        shadow_id,
        "qkv_dot",
        132 + dx,
        180,
        456,
        58,
        ["Multi-Query SDPA", f"{num_heads} Q heads + 1 shared K/V  ·  d_k = {d_k}"],
        font_size=15,
    )

    shared_kv = mqa_shared_kv_node(body, 260 + dx, 310, 200, 48, num_heads)

    q_proj = _rect_block(body, info, shadow_id, "q_proj", 92 + dx, 418, 184, 50, ["Linear (Q)", f"{num_heads} heads"], font_size=15)
    k_proj = _rect_block(body, info, shadow_id, "k_proj", 330 + dx, 422, 140, 48, ["Linear (K)", "1 head"], font_size=15)
    v_proj = _rect_block(body, info, shadow_id, "v_proj", 500 + dx, 422, 140, 48, ["Linear (V)", "1 head"], font_size=15)

    branch_x, branch_y = cx, 526
    body.append(_branch_dot(branch_x, branch_y))
    body.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 34, "x2": branch_x, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    body.append(_svg_text(
        branch_x, h - 24,
        f"in ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    body.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    body.append(_elbow_hv(branch_x, branch_y, k_proj["cx"], k_proj["bottom"] + GAP, arrow_id))
    body.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))

    sdpa_entry_y = sdpa["bottom"] + GAP
    body.append(_elbow_hv(q_proj["cx"], q_proj["top"], sdpa["left"] + 120, sdpa_entry_y, arrow_id))
    body.append(_elbow_vh(k_proj["cx"], k_proj["top"], shared_kv["left"] + 62, shared_kv["bottom"] + GAP, arrow_id))
    body.append(_elbow_vh(v_proj["cx"], v_proj["top"], shared_kv["right"] - 62, shared_kv["bottom"] + GAP, arrow_id))
    body.append(_v_line(shared_kv, sdpa, arrow_id))
    body.append(_v_line(sdpa, o_proj, arrow_id))
    output_stem(body, cx, o_proj, arrow_id, hidden)

    attn_dim_label(body, q_proj["left"] + 4, q_proj["bottom"] + 23, f"Q: {hidden} -> {q_out}")
    attn_dim_label(body, k_proj["left"] + 4, k_proj["bottom"] + 23, f"K: {hidden} -> {kv_out}")
    attn_dim_label(body, v_proj["left"] + 4, v_proj["bottom"] + 23, f"V: {hidden} -> {kv_out}")
    attn_dim_label(body, o_proj["left"] - 16, o_proj["cy"], f"{q_out} -> {hidden}", anchor="end")

    badge = None
    if num_heads and num_heads > 1:
        badge = kv_cache_badge(body, w - 218, 58, f"KV cache {num_heads}x smaller", "than full MHA")

    body.append(_svg_text(
        cx,
        shared_kv["top"] - 18,
        f"1 shared K/V head feeds all {num_heads} Q heads" if num_heads else "single shared K/V head",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    region = dynamic_region_rect(
        [
            o_proj,
            sdpa,
            shared_kv,
            q_proj,
            k_proj,
            v_proj,
            badge,
            {"left": cx - 86, "right": cx + 86, "top": o_proj["top"] - 56, "bottom": branch_y + 56},
        ],
        w,
        h,
        pad_x=86,
        pad_y=36,
    )
    parts.append(region)
    parts.extend(body)
    return _svg(w, h, f"{ir.get('name', 'model')} multi-query attention", parts)
