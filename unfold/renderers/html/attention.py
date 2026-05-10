"""HTML renderer pieces specific to the attention block.

Anything that turns an attention spec into HTML (the multi-variant inspect
card, the per-row description) lives here so attention semantics aren't
scattered across the larger ``components`` module.

Pure label / predicate logic (``mask_short``, ``is_sliding``, ``kv_shared``,
``describe_attention`` …) lives in :mod:`unfold.labels` — this module only
deals with HTML.
"""
from __future__ import annotations

from ...labels import describe_attention, kv_shared, mask_long
from .utils import _attr, _html


def attention_card(ir: dict, info: dict, meta_for: callable) -> str:
    """Inspect card for the attention block.

    With a single attention variant this falls through to a normal card.  With
    multiple variants (Gemma 4: sliding + full) the card lists every variant
    as its own row so the bifurcation is explicit no matter which variant
    pill is currently active.  Each row also notes how many of that variant
    reuse K/V from earlier layers.
    """
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
    """One row in the multi-variant attention card."""
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


# Scoped CSS for the attention card.  Lives next to the attention rendering
# so updating row styling doesn't require digging through the central style
# block.
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
