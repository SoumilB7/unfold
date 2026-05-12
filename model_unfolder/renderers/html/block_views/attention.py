"""Attention detail-card dispatcher and grouped attention summary cards."""
from __future__ import annotations

from ....labels import describe_attention, kv_shared, mask_long
from ..utils import _html
from .attention_types import (
    build_gqa_attention_view,
    build_linear_attention_view,
    build_mla_attention_view,
    build_mqa_attention_view,
    build_recurrent_view,
    build_rwkv_view,
    build_sdpa_attention_view,
    build_ssm_view,
)


def build_attention_view(ir: dict, info: dict, mount_id: str) -> str:
    """Rich SVG detail view for the active attention-like block."""
    attn = info["dominant"]["spec"].get("attention") or {}
    kind = attn.get("kind")
    if kind == "mla":
        return build_mla_attention_view(ir, info, mount_id)
    if kind == "mqa":
        return build_mqa_attention_view(ir, info, mount_id)
    if kind == "gqa":
        return build_gqa_attention_view(ir, info, mount_id)
    if kind == "ssm":
        return build_ssm_view(ir, info, mount_id)
    if kind == "recurrent":
        return build_recurrent_view(ir, info, mount_id)
    if kind == "rwkv":
        return build_rwkv_view(ir, info, mount_id)
    if kind == "linear":
        return build_linear_attention_view(ir, info, mount_id)
    return build_sdpa_attention_view(ir, info, mount_id)


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
