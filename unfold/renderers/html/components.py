"""HTML chrome, inspect cards, scoped CSS, and interaction scripts."""
from __future__ import annotations

from .metadata import _arch_badges, _describe_attention, _describe_ffn
from .theme import C, FONT_BODY, FONT_HEAD, FONT_MONO
from .utils import _attr, _fmt_int, _html
from .views import _build_ffn_view, _build_moe_view


def _build_inspect_cards(ir: dict, info: dict, mount_id: str) -> str:
    """Cards-only HTML for the L2 inspect panel.  The caller wraps in a
    .uf-l2-variant container so multiple variants can coexist."""
    panels: list[str] = []
    ffn = info["dominant"]["spec"]["ffn"]

    panels.append(_hint_card(
        "default",
        "Click a block above to inspect it",
    ))
    for node_id in ("tok_text", "embed", "rms1", "attn", "add1", "rms2"):
        panels.append(_simple_card(node_id, *_meta(info, node_id)))

    # The interesting one: clicking the FFN block in the architecture reveals
    # the FULL internal FFN diagram (or the MoE diagram for sparse models).
    ffn_title, ffn_desc = _meta(info, "ffn")
    if ffn.get("kind") == "moe":
        panels.append(_rich_card(
            "ffn",
            ffn_title,
            ffn_desc,
            _build_moe_view(ir, info, mount_id),
        ))
    else:
        panels.append(_rich_card(
            "ffn",
            ffn_title,
            ffn_desc,
            _build_ffn_view(ir, info, mount_id),
        ))

    for node_id in ("add2", "final_rms", "lm_head"):
        panels.append(_simple_card(node_id, *_meta(info, node_id)))

    return "".join(panels)


def _attention_title(attn: dict) -> str:
    kinds = {"mla": "Multi-head latent attention", "gqa": "Grouped-query attention",
             "mqa": "Multi-query attention"}
    return kinds.get(attn.get("kind", ""), "Attention")


def _meta(info: dict, node_id: str) -> tuple[str, str]:
    return info.get("meta", {}).get(node_id, (node_id, ""))


def _simple_card(node_id: str, title: str, desc: str) -> str:
    return (
        f'<div class="uf-card-detail uf-card-{_attr(node_id)}">'
        f'<div class="uf-card-title">{_html(title)}</div>'
        f'<div class="uf-card-desc">{_html(desc)}</div>'
        '</div>'
    )


def _hint_card(node_id: str, hint: str) -> str:
    """Subtle placeholder shown when no block is selected — just a one-line
    hint inside the same panel box."""
    return (
        f'<div class="uf-card-detail uf-card-hint uf-card-{_attr(node_id)}">'
        f'{_html(hint)}'
        '</div>'
    )


def _l3_card(node_id: str, title: str, desc: str) -> str:
    """L3 sub-inspect detail card. Same layout as a simple card, but lives in
    the .uf-sub-inspect container so the L3 toggle CSS applies."""
    return (
        f'<div class="uf-card-detail uf-l3-{_attr(node_id)}">'
        f'<div class="uf-card-title">{_html(title)}</div>'
        f'<div class="uf-card-desc">{_html(desc)}</div>'
        '</div>'
    )


def _build_sub_inspect_cards(ir: dict, info: dict, mount_id: str) -> str:
    """Cards-only HTML for the L3 sub-inspect panel.  Only one card
    is visible at a time; the JS click handler toggles them."""
    panels: list[str] = []
    ffn = info["dominant"]["spec"]["ffn"]
    ffn_block = info.get("blocks", {}).get("ffn", {})
    children = ffn_block.get("children", [])

    # Default L3 card is never actually visible (the whole L3 box is hidden
    # until uf-sub-active is set), but kept as a safe target for showL3('default').
    panels.append(_l3_card("default", "", ""))

    if children:
        for child in children:
            panels.append(_l3_card(child["id"], child.get("title", child["id"]), child.get("description", "")))
    else:
        panels.extend(_fallback_sub_inspect_cards(ir, ffn))

    return "".join(panels)


def _fallback_sub_inspect_cards(ir: dict, ffn: dict) -> list[str]:
    h = _fmt_int(ir.get("hidden_size"))
    inter = _fmt_int(ffn.get("expert_intermediate_size") or ffn.get("intermediate_size"))
    activation = (ffn.get("activation") or "silu").upper()
    panels = [
        _l3_card("gate_proj", "Gate projection", f"Linear · {h} → {inter} (gated path through {activation})"),
        _l3_card("up_proj", "Up projection", f"Linear · {h} → {inter}"),
        _l3_card("silu", f"{activation} activation", "Element-wise non-linearity applied to the gate path"),
        _l3_card("mul", "Element-wise multiply", f"{activation}(gate) × up — combines the gated and ungated paths"),
        _l3_card("down_proj", "Down projection", f"Linear · {inter} → {h}"),
    ]
    if ffn.get("kind") == "moe":
        n_experts = _fmt_int(ffn.get("num_experts")) if ffn.get("num_experts") else "N"
        n_active = ffn.get("num_experts_per_tok") or "k"
        n_shared = ffn.get("num_shared_experts") or 0
        panels.append(_l3_card("router", "Router", f"Linear · {h} → {n_experts} (selects top-{n_active} experts per token)"))
        expert_desc = (
            f"Dense FFN with same shape as above · {h} → {inter} → {h} · "
            f"only top-{n_active} of {n_experts} active per token"
            + (f" · plus {n_shared} shared expert(s) always active" if n_shared else "")
        )
        for eid in ("expert_1", "expert_k", "expert_kp1", "expert_n"):
            panels.append(_l3_card(eid, "Expert FFN", expert_desc))
        panels.append(_l3_card("add_moe", "Weighted sum", f"Combines top-{n_active} expert outputs weighted by router probabilities"))
    return panels


def _rich_card(node_id: str, title: str, desc: str, svg: str) -> str:
    return (
        f'<div class="uf-card-detail uf-card-{_attr(node_id)}">'
        f'<div class="uf-card-title">{_html(title)}</div>'
        f'<div class="uf-card-desc">{_html(desc)}</div>'
        f'<div class="uf-card-svg">{svg}</div>'
        '</div>'
    )


def _click_script(mount_id: str) -> str:
    """Inline JS for click-to-inspect.

    Three levels of progressive disclosure:
      L1  the architecture diagram (always rendered)
      L2  inspect panel — opened by clicking an L1 block
      L3  sub-inspect panel — opened by clicking a sub-block inside L2's SVG
          (e.g. SiLU / Linear (gate) / Router / Expert)

    Re-clicking the currently selected block returns to the default state.
    Switching the L2 selection automatically resets L3.
    """
    return f"""
<script>
(function() {{
  var root = document.getElementById('{mount_id}');
  if (!root) return;

  // L1 (architecture) blocks — direct children of the architecture details.
  var l1 = root.querySelectorAll('.uf-section-arch .uf-node');
  if (!l1.length) l1 = root.querySelectorAll('.uf-section-body .uf-node');
  // L2 detail cards (only the .uf-inspect container, not the L3 sub-inspect).
  var l2cards = root.querySelectorAll('.uf-inspect .uf-card-detail');
  // L3 sub-block click targets — they live INSIDE the L2 SVG.
  var l3 = root.querySelectorAll('.uf-inspect .uf-card-svg .uf-node');
  var l3cards = root.querySelectorAll('.uf-sub-inspect .uf-card-detail');
  var l3box = root.querySelector('.uf-sub-inspect');

  function showL2(id) {{
    l2cards.forEach(function(p) {{
      p.style.display = p.classList.contains('uf-card-' + id) ? 'block' : 'none';
    }});
    l1.forEach(function(n) {{
      if (n.getAttribute('data-id') === id) n.classList.add('uf-selected');
      else n.classList.remove('uf-selected');
    }});
    // Switching L2 always resets L3 (the sub-block context is gone).
    showL3('default');
  }}

  function showL3(id) {{
    l3cards.forEach(function(p) {{
      p.style.display = p.classList.contains('uf-l3-' + id) ? 'block' : 'none';
    }});
    l3.forEach(function(n) {{
      if (n.getAttribute('data-id') === id) n.classList.add('uf-sub-selected');
      else n.classList.remove('uf-sub-selected');
    }});
    if (l3box) {{
      if (id === 'default') l3box.classList.remove('uf-sub-active');
      else l3box.classList.add('uf-sub-active');
    }}
  }}

  l1.forEach(function(n) {{
    n.style.cursor = 'pointer';
    n.addEventListener('click', function(e) {{
      e.stopPropagation();
      if (n.classList.contains('uf-selected')) {{ showL2('default'); }}
      else {{ showL2(n.getAttribute('data-id')); }}
    }});
  }});

  l3.forEach(function(n) {{
    n.style.cursor = 'pointer';
    n.addEventListener('click', function(e) {{
      e.stopPropagation();
      if (n.classList.contains('uf-sub-selected')) {{ showL3('default'); }}
      else {{ showL3(n.getAttribute('data-id')); }}
    }});
  }});
}})();
</script>
"""


def _details_section(label: str, sub: str, svg: str) -> str:
    """Collapsible section using <details>; closed by default."""
    return (
        '<details class="uf-section uf-section-collapsible">'
        '<summary class="uf-section-head">'
        f'<span class="uf-section-label">{_html(label)}</span>'
        f'<span class="uf-section-sub">{_html(sub)}</span>'
        '<span class="uf-chevron" aria-hidden="true">›</span>'
        '</summary>'
        f'<div class="uf-section-body">{svg}</div>'
        '</details>'
    )


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
  align-items:baseline;
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
/* Layer-type variant toggle (only rendered when 2+ groups exist) */
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

/* L2 inspect panel — visible only while the architecture is expanded */
#{mount_id} .uf-inspect {{
  display:none;
  margin-top:10px;
  padding:10px 12px;
  background:{C['canvas']};
  border:0.5px solid {C['border']};
  border-radius:9px;
  min-height:32px;
  animation:uf-fade .2s ease-out;
}}
#{mount_id} .uf-section-arch[open] ~ .uf-inspect {{
  display:block;
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
/* L3 sub-inspect — visible only when arch open AND a sub-block is selected */
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
@keyframes uf-fade {{
  from {{ opacity:0; transform:translateY(2px); }}
  to   {{ opacity:1; transform:none; }}
}}
"""


def _header(ir: dict, info: dict) -> str:
    badges = []
    for badge in _arch_badges(ir, info):
        title = badge.get("title") or ""
        badges.append(
            f'<span class="uf-badge" title="{_attr(title)}">{_html(badge["text"])}</span>'
        )
    return f"""
<div class="uf-header">
  <div class="uf-name">{_html(ir.get("name", "model"))}</div>
  <div class="uf-arch">{_html(ir.get("architecture", ""))}</div>
  <div class="uf-badges">{''.join(badges)}</div>
</div>
"""


def _stats_banner(ir: dict) -> str:
    params = ir.get("params") or {}
    param_text = (
        f"{params.get('total_h')} ({params.get('active_h')} act.)"
        if params.get("is_sparse")
        else params.get("total_h", "?")
    )
    items = [
        ("Layers", str(len(ir.get("layers", [])))),
        ("Hidden", _fmt_int(ir.get("hidden_size"))),
        ("Vocab", _fmt_int(ir.get("vocab_size"))),
        ("Context", _fmt_int(ir.get("max_position_embeddings")) if ir.get("max_position_embeddings") else "-"),
        ("Params", param_text or "?"),
    ]
    cells = []
    for key, value in items:
        cells.append(
            '<div class="uf-stat">'
            f'<div class="uf-stat-key">{_html(key.upper())}</div>'
            f'<div class="uf-stat-val">{_html(value)}</div>'
            '</div>'
        )
    return f'<div class="uf-stats">{"".join(cells)}</div>'
