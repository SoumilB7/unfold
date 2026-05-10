"""Scoped CSS for rendered HTML fragments."""
from __future__ import annotations

from .block_views import attention_card_css
from .theme import C, FONT_BODY, FONT_HEAD, FONT_MONO


def _style(mount_id: str) -> str:
    return f"""
#{mount_id} {{
  font-family:{FONT_BODY};
  color:{C['text']};
  max-width:720px;
}}
#{mount_id} .uf-card {{
  background:{C['bg_card']};
  border:1.5px solid {C['block']};
  border-radius:14px;
  padding:20px 22px 18px;
  box-shadow:0 1px 3px rgba(15,23,42,0.05);
}}
#{mount_id} .uf-header {{
  margin-bottom:14px;
}}
#{mount_id} .uf-name {{
  font-family:{FONT_HEAD};
  font-size:26px;
  line-height:1;
  color:{C['text']};
}}
#{mount_id} .uf-arch {{
  color:{C['muted']};
  font-size:11px;
  margin-top:3px;
  font-family:{FONT_MONO};
}}
#{mount_id} .uf-badges {{
  display:flex;
  gap:6px;
  flex-wrap:wrap;
  margin-top:6px;
}}
#{mount_id} .uf-badge {{
  display:inline-flex;
  align-items:center;
  height:22px;
  padding:0 9px;
  background:{C['badge_bg']};
  color:{C['badge_text']};
  border-radius:11px;
  font-size:11px;
  font-weight:600;
  letter-spacing:0.02em;
}}
#{mount_id} .uf-stats {{
  display:grid;
  grid-template-columns:repeat(5,minmax(0,1fr));
  gap:1px;
  background:{C['border']};
  border:0.5px solid {C['border']};
  border-radius:8px;
  overflow:hidden;
  margin-bottom:18px;
}}
#{mount_id} .uf-stat {{
  padding:8px 12px;
  background:{C['bg_card']};
}}
#{mount_id} .uf-stat-key {{
  font-size:9.5px;
  letter-spacing:0.12em;
  color:{C['muted']};
  font-weight:600;
}}
#{mount_id} .uf-stat-val {{
  font-family:{FONT_HEAD};
  font-size:19px;
  color:{C['text']};
  margin-top:2px;
  line-height:1.05;
}}
#{mount_id} .uf-section {{
  margin-top:14px;
}}
#{mount_id} .uf-section-head {{
  display:flex;
  align-items:center;
  gap:10px;
  padding:9px 12px;
  background:{C['badge_bg']};
  border:0.5px solid {C['border']};
  border-radius:8px;
  cursor:default;
  user-select:none;
  list-style:none;
}}
#{mount_id} .uf-section-collapsible > summary.uf-section-head {{
  cursor:pointer;
  transition:background .15s;
}}
#{mount_id} .uf-section-collapsible > summary.uf-section-head:hover {{
  background:#C8E9D7;
}}
#{mount_id} .uf-section-head::-webkit-details-marker,
#{mount_id} summary.uf-section-head::-webkit-details-marker {{
  display:none;
}}
#{mount_id} .uf-section-label {{
  font-size:10.5px;
  letter-spacing:0.14em;
  font-weight:700;
  color:{C['text']};
}}
#{mount_id} .uf-section-sub {{
  font-size:11px;
  color:{C['muted']};
}}
#{mount_id} .uf-chevron {{
  margin-left:auto;
  font-size:18px;
  color:{C['muted']};
  transition:transform .15s ease;
  transform:rotate(0deg);
}}
#{mount_id} details.uf-section[open] .uf-chevron {{
  transform:rotate(90deg);
}}
#{mount_id} .uf-section-body {{
  background:{C['canvas']};
  border:0.5px solid {C['border']};
  border-radius:10px;
  padding:6px;
  margin-top:6px;
  animation:uf-fade .25s ease-out;
}}
#{mount_id} .uf-section-body svg {{
  display:block;
  max-width:100%;
  height:auto;
}}
#{mount_id} .uf-node rect,
#{mount_id} .uf-node circle {{
  transition:filter .15s, stroke .15s, stroke-width .15s;
}}
#{mount_id} .uf-section-arch .uf-node {{
  cursor:pointer;
}}
#{mount_id} .uf-section-arch .uf-node:hover rect,
#{mount_id} .uf-section-arch .uf-node:hover circle {{
  filter:brightness(1.1) drop-shadow(0 2px 4px rgba(0,0,0,.18));
}}
#{mount_id} .uf-node.uf-selected rect,
#{mount_id} .uf-node.uf-selected circle {{
  stroke:#FACC15;
  stroke-width:2.5;
}}
#{mount_id} .uf-group-radio {{
  position:absolute;
  width:0;height:0;
  opacity:0;
  pointer-events:none;
}}
#{mount_id} .uf-group-toggle {{
  display:inline-flex;
  gap:3px;
  padding:3px;
  margin-bottom:8px;
  background:{C['bg_card']};
  border:0.5px solid {C['border']};
  border-radius:8px;
  flex-wrap:wrap;
}}
#{mount_id} .uf-group-pill {{
  padding:5px 11px;
  font-size:11px;
  font-weight:600;
  color:{C['muted']};
  cursor:pointer;
  border-radius:5px;
  transition:background .15s,color .15s;
  white-space:nowrap;
  user-select:none;
}}
#{mount_id} .uf-group-pill:hover {{
  background:{C['badge_bg']};
  color:{C['text']};
}}
#{mount_id} .uf-arch-variant,
#{mount_id} .uf-l2-variant,
#{mount_id} .uf-l3-variant {{ display:none; }}
#{mount_id} .uf-inspect {{
  display:none;
  margin-top:10px;
  padding:10px 12px;
  background:rgba(244,251,248,0.55);
  border:0.5px solid {C['border']};
  border-radius:9px;
  min-height:60px;
  animation:uf-fade .2s ease-out;
}}
#{mount_id} .uf-section-arch[open] ~ .uf-inspect {{
  display:flex;
  flex-direction:column;
  justify-content:center;
}}
#{mount_id} .uf-card-detail {{
  display:none;
  width:100%;
  animation:uf-fade .2s ease-out;
}}
#{mount_id} .uf-card-default {{
  display:block;
}}
#{mount_id} .uf-card-hint {{
  font-size:11px;
  color:{C['muted']};
  font-style:italic;
  text-align:center;
  padding:0;
}}
#{mount_id} .uf-sub-inspect {{
  display:none;
  margin-top:8px;
  padding:10px 14px;
  background:{C['bg_card']};
  border:0.5px solid {C['border']};
  border-left:3px solid {C['block']};
  border-radius:9px;
  animation:uf-fade .2s ease-out;
}}
#{mount_id} .uf-section-arch[open] ~ .uf-sub-inspect.uf-sub-active {{
  display:block;
}}
#{mount_id} .uf-node.uf-sub-selected rect,
#{mount_id} .uf-node.uf-sub-selected circle {{
  stroke:#FACC15;
  stroke-width:2.5;
}}
#{mount_id} .uf-card-title {{
  font-family:{FONT_HEAD};
  font-size:20px;
  color:{C['text']};
  line-height:1.1;
}}
#{mount_id} .uf-card-desc {{
  font-size:13px;
  color:{C['muted']};
  margin-top:4px;
  line-height:1.45;
}}
#{mount_id} .uf-card-svg {{
  margin-top:10px;
  background:{C['bg_card']};
  border:0.5px solid {C['border']};
  border-radius:8px;
  padding:6px;
}}
#{mount_id} .uf-card-svg svg {{
  display:block;
  max-width:100%;
  height:auto;
}}
{_attention_styles(mount_id)}
@keyframes uf-fade {{
  from {{ opacity:0; transform:translateY(2px); }}
  to   {{ opacity:1; transform:none; }}
}}
"""


def _attention_styles(mount_id: str) -> str:
    return attention_card_css(
        mount_id,
        {
            "bg_card": C["bg_card"],
            "border": C["border"],
            "block": C["block"],
            "text": C["text"],
            "muted": C["muted"],
            "font_head": FONT_HEAD,
            "font_mono": FONT_MONO,
        },
    )
