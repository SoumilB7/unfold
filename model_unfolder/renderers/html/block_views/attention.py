"""Inspect-card content and detail SVG for attention blocks."""
from __future__ import annotations

from ....labels import describe_attention, kv_shared, mask_long
from ..svg import (
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
from ..theme import C, FONT_MONO, GAP
from ..utils import _fmt_int, _html


def build_attention_view(ir: dict, info: dict, mount_id: str) -> str:
    """Rich SVG detail view for MHA / GQA / MQA attention blocks.

    Layout (bottom → top, matching the feed-forward convention):
      input branch → Q / K / V projections → scaled dot-product → output projection
    """
    w, h = 720, 590
    arrow_id, shadow_id = _ids(mount_id, "attn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden_sz = ir.get("hidden_size") or 0
    hidden = _fmt_int(hidden_sz)
    num_heads = attn.get("num_heads") or 0
    num_kv_heads = attn.get("num_kv_heads") or num_heads
    head_dim = attn.get("head_dim") or (hidden_sz // num_heads if num_heads else 0)
    q_out = _fmt_int(num_heads * head_dim) if (num_heads and head_dim) else hidden
    kv_out = _fmt_int(num_kv_heads * head_dim) if (num_kv_heads and head_dim) else hidden
    d_k = str(head_dim) if head_dim else "d_k"

    cx = w / 2

    # --- Output projection (top) ---
    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 90, 100, 180, 50, "Linear (out)")

    # --- Scaled dot-product attention (center, wide) ---
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

    # --- Q / K / V projections (symmetric trio) ---
    proj_w, proj_h, proj_y = 165, 50, 315
    q_proj = _rect_block(parts, info, shadow_id, "q_proj",  70, proj_y, proj_w, proj_h, "Linear (Q)")
    k_proj = _rect_block(parts, info, shadow_id, "k_proj", 278, proj_y, proj_w, proj_h, "Linear (K)")
    v_proj = _rect_block(parts, info, shadow_id, "v_proj", 486, proj_y, proj_w, proj_h, "Linear (V)")

    # --- Branch point ---
    branch_x, branch_y = cx, 475
    parts.append(_branch_dot(branch_x, branch_y))

    # Input stem
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 38, "x2": branch_x, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        branch_x, h - 32,
        f"in ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Branch → Q, K, V
    parts.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(branch_x, branch_y, k_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))

    # Q → SDPA left entry, K → center, V → right entry
    sdpa_entry_gap = sdpa["bottom"] + GAP
    parts.append(_elbow_hv(q_proj["cx"], q_proj["top"], 200, sdpa_entry_gap, arrow_id))
    parts.append(_v_seg(k_proj["cx"], k_proj["top"], sdpa_entry_gap, arrow_id))
    parts.append(_elbow_hv(v_proj["cx"], v_proj["top"], 520, sdpa_entry_gap, arrow_id))

    # SDPA → O projection
    parts.append(_v_line(sdpa, o_proj, arrow_id))

    # O projection → output label
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": o_proj["top"], "x2": cx, "y2": o_proj["top"] - 34,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, o_proj["top"] - 44,
        f"out ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Dimension annotations
    _attn_dim_label(parts, q_proj["right"] + 10, q_proj["cy"], f"{hidden} → {q_out}")
    _attn_dim_label(parts, k_proj["right"] + 10, k_proj["cy"], f"{hidden} → {kv_out}")
    _attn_dim_label(parts, v_proj["right"] + 10, v_proj["cy"], f"{hidden} → {kv_out}")
    _attn_dim_label(parts, o_proj["right"] + 10, o_proj["cy"], f"{q_out} → {hidden}")

    return _svg(w, h, f"{ir.get('name', 'model')} attention", parts)


def _attn_dim_label(parts: list[str], x: float, y: float, text: str) -> None:
    parts.append(_svg_text(
        x, y, text,
        {
            "text-anchor": "start",
            "dominant-baseline": "central",
            "fill": C["muted"],
            "font-family": FONT_MONO,
            "font-size": 10,
        },
    ))


def attention_card(ir: dict, info: dict, meta_for: callable) -> str:
    """Inspect card for the attention block."""
    attn_groups = [
        g for g in info.get("groups", []) if g.get("spec", {}).get("attention")
    ]
    if len(attn_groups) <= 1:
        title, desc = meta_for("attn")
        return (
            '<div class="uf-card-detail uf-card-attn">'
            f'<div class="uf-card-title">{_html(title)}</div>'
            f'<div class="uf-card-desc">{_html(desc)}</div>'
            "</div>"
        )

    rows = "".join(_attention_row_for_group(group, ir) for group in attn_groups)
    return (
        '<div class="uf-card-detail uf-card-attn">'
        '<div class="uf-card-title">Attention layers</div>'
        '<div class="uf-card-desc">'
        f"{len(attn_groups)} attention variants in this model — each row is one variant."
        "</div>"
        f'<div class="uf-attn-rows">{rows}</div>'
        "</div>"
    )


def _attention_row_for_group(group: dict, ir: dict) -> str:
    attn = group["spec"]["attention"]
    indices = group["indices"]
    n_layers = len(indices)
    layers = ir.get("layers", [])
    n_shared = sum(
        1 for i in indices
        if 0 <= i < len(layers) and kv_shared(layers[i].get("attention") or {})
    )
    return _attention_row(attn, n_layers, n_shared)


def _attention_row(attn: dict, n_layers: int, n_shared: int) -> str:
    title = f"{mask_long(attn)} · {describe_attention(attn)}"
    bits: list[str] = []
    if attn.get("window_size"):
        bits.append(f"window {attn['window_size']}")
    if n_shared:
        bits.append(f"{n_shared} of {n_layers} reuse K/V from earlier layers")
    else:
        bits.append(f"{n_layers} layers")
    detail = "  ·  ".join(bits)
    return (
        '<div class="uf-attn-row">'
        f'<div class="uf-attn-row-title">{_html(title)}</div>'
        f'<div class="uf-attn-row-detail">{_html(detail)}</div>'
        "</div>"
    )


def attention_card_css(mount_id: str, theme: dict) -> str:
    return f"""
#{mount_id} .uf-attn-rows {{
  margin-top:10px;
  display:flex;
  flex-direction:column;
  gap:8px;
}}
#{mount_id} .uf-attn-row {{
  padding:9px 12px;
  background:{theme['bg_card']};
  border:0.5px solid {theme['border']};
  border-left:3px solid {theme['block']};
  border-radius:8px;
}}
#{mount_id} .uf-attn-row-title {{
  font-family:{theme['font_head']};
  font-size:16px;
  color:{theme['text']};
  line-height:1.15;
}}
#{mount_id} .uf-attn-row-detail {{
  margin-top:3px;
  font-size:12px;
  color:{theme['muted']};
  font-family:{theme['font_mono']};
}}
"""
