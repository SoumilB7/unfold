"""Latent-attention detail view."""
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
    _svg_text,
    _v_line,
    _v_seg,
)
from ...theme import C, FONT_MONO, GAP
from ...utils import _fmt_int


def build(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for Multi-head Latent Attention (DeepSeek / Kimi style)."""
    w, h = 720, 620
    arrow_id, shadow_id = _ids(mount_id, "mla")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden = _fmt_int(ir.get("hidden_size"))
    kv_rank = _fmt_int(attn.get("kv_lora_rank"))
    q_rank = _fmt_int(attn.get("q_lora_rank")) if attn.get("q_lora_rank") else "direct"
    rope_dim = _fmt_int(attn.get("rope_dim"))
    cx = w / 2

    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 90, 82, 180, 50, "Linear (out)")
    mla = _rect_block(parts, info, shadow_id, "mla_attn", 142, 182, 436, 58, ["Multi-Head Latent", "Attention"], font_size=16)
    q_path = _rect_block(parts, info, shadow_id, "mla_q", 76, 350, 170, 50, ["Q path", f"rank {q_rank}"], font_size=15)
    kv_down = _rect_block(parts, info, shadow_id, "mla_kv_down", 276, 418, 170, 50, ["KV down", f"rank {kv_rank}"], font_size=15)
    kv_up = _rect_block(parts, info, shadow_id, "mla_kv_up", 276, 302, 170, 50, "KV up", font_size=16)
    rope = _rect_block(parts, info, shadow_id, "mla_rope", 476, 350, 170, 50, ["RoPE split", f"dim {rope_dim}"], font_size=15)

    branch_x, branch_y = cx, 540
    parts.append(_branch_dot(branch_x, branch_y))
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 34, "x2": branch_x, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        branch_x, h - 28,
        f"in ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    parts.append(_elbow_hv(branch_x, branch_y, q_path["cx"], q_path["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, kv_down["cx"], kv_down["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, rope["cx"], rope["bottom"] + GAP, arrow_id))
    parts.append(_v_line(kv_down, kv_up, arrow_id))
    parts.append(_elbow_hv(q_path["cx"], q_path["top"], 210, mla["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(kv_up["cx"], kv_up["top"], mla["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(rope["cx"], rope["top"], 510, mla["bottom"] + GAP, arrow_id))
    parts.append(_v_line(mla, o_proj, arrow_id))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": o_proj["top"],
        "x2": cx, "y2": o_proj["top"] - 34,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, o_proj["top"] - 44,
        f"out ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    return _svg(w, h, f"{ir.get('name', 'model')} multi-head latent attention", parts)
