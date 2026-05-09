"""Pure-Python HTML/SVG renderer for unfold diagrams."""
from __future__ import annotations

from html import escape
from typing import Any


FONT_HEAD = '"Caveat","Patrick Hand","Comic Sans MS",cursive'
FONT_BODY = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_MONO = 'ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace'
GAP = 6

C = {
    "bg_outer": "#E1F5EE",
    "bg_inner": "#9FE1CB",
    "bg_card": "#FFFFFF",
    "canvas": "#F4FBF8",
    "block": "#0F6E56",
    "block_alt": "#0E5C48",
    "text_block": "#FFFFFF",
    "arrow": "#0F6E56",
    "text": "#04342C",
    "muted": "#5F7C73",
    "border": "#B6DDCB",
    "badge_bg": "#D6F1E4",
    "badge_text": "#0E5C48",
}


def render_fragment(ir: dict, mount_id: str) -> str:
    """Render a complete HTML fragment.

    For heterogeneous models (multiple layer-type groups, e.g. DeepSeek-V3's
    dense → MoE phase change), one architecture / L2 / L3 panel is emitted
    per group, all sharing the same DOM tree.  Hidden via CSS, toggled by
    radio buttons in a pill bar above the SVG.  Pure-CSS toggle, no JS.
    """
    info = _make_info(ir)
    groups = info["groups"]
    dominant_idx = groups.index(info["dominant"]) if groups else 0

    radios: list[str] = []
    pills: list[str] = []
    arch_variants: list[str] = []
    l2_variants: list[str] = []
    l3_variants: list[str] = []
    variant_css: list[str] = []
    radio_name = f"{mount_id}-g"

    for i, group in enumerate(groups):
        variant_info = {
            "groups": groups,
            "dominant": group,
            "meta": _meta_for(ir, group["spec"]),
        }
        suffix = f"-g{i}"
        radio_id = f"{mount_id}-g{i}"
        checked = " checked" if i == dominant_idx else ""

        radios.append(
            f'<input type="radio" name="{_attr(radio_name)}" '
            f'id="{_attr(radio_id)}" class="uf-group-radio"{checked}>'
        )
        pills.append(
            f'<label for="{_attr(radio_id)}" class="uf-group-pill">'
            f'{_html(_group_label(group))}</label>'
        )

        arch_svg = _build_architecture_view(ir, variant_info, mount_id + suffix)
        arch_variants.append(
            f'<div class="uf-arch-variant uf-arch-variant-{i}">{arch_svg}</div>'
        )
        l2_inner = _build_inspect_cards(ir, variant_info, mount_id + suffix)
        l2_variants.append(
            f'<div class="uf-l2-variant uf-l2-variant-{i}">{l2_inner}</div>'
        )
        l3_inner = _build_sub_inspect_cards(ir, variant_info, mount_id + suffix)
        l3_variants.append(
            f'<div class="uf-l3-variant uf-l3-variant-{i}">{l3_inner}</div>'
        )

        # Per-variant visibility + active-pill styling
        variant_css.append(
            f"#{mount_id} #{radio_id}:checked ~ .uf-card .uf-arch-variant-{i},"
            f"#{mount_id} #{radio_id}:checked ~ .uf-card .uf-l2-variant-{i},"
            f"#{mount_id} #{radio_id}:checked ~ .uf-card .uf-l3-variant-{i} "
            f"{{ display:block; }}"
        )
        variant_css.append(
            f"#{mount_id} #{radio_id}:checked ~ .uf-card .uf-group-pill[for='{radio_id}'] "
            f"{{ background:{C['block']}; color:#FFFFFF; }}"
        )

    style = _style(mount_id) + "\n" + "\n".join(variant_css)

    toggle_html = (
        f'<div class="uf-group-toggle" role="tablist">{"".join(pills)}</div>'
        if len(groups) > 1 else ""
    )

    arch_section = (
        '<details class="uf-section uf-section-arch uf-section-collapsible" open>'
        '<summary class="uf-section-head">'
        '<span class="uf-section-label">ARCHITECTURE</span>'
        f'<span class="uf-section-sub">Per-layer block · repeats × {len(ir.get("layers", []))}</span>'
        '<span class="uf-chevron" aria-hidden="true">›</span>'
        '</summary>'
        f'<div class="uf-section-body">{toggle_html}{"".join(arch_variants)}</div>'
        '</details>'
    )
    inspect_panel = f'<div class="uf-inspect">{"".join(l2_variants)}</div>'
    sub_inspect_panel = f'<div class="uf-sub-inspect">{"".join(l3_variants)}</div>'

    map_svg = _build_layer_map(ir, info, mount_id)
    n_groups = len(groups)
    map_sub = (
        "All layers structurally identical"
        if n_groups <= 1
        else f"{n_groups} layer types across {len(ir.get('layers', []))} layers"
    )
    layer_map_section = _details_section("LAYER MAP", map_sub, map_svg)

    return f"""
<div id="{_attr(mount_id)}" class="uf-root">
<style>
{style}
</style>
{''.join(radios)}
<div class="uf-card">
{_header(ir, info)}
{_stats_banner(ir)}
{arch_section}
{inspect_panel}
{sub_inspect_panel}
{layer_map_section}
</div>
{_click_script(mount_id)}
</div>
"""


def _build_inspect_cards(ir: dict, info: dict, mount_id: str) -> str:
    """Cards-only HTML for the L2 inspect panel.  The caller wraps in a
    .uf-l2-variant container so multiple variants can coexist."""
    panels: list[str] = []
    h = _fmt_int(ir.get("hidden_size"))
    v = _fmt_int(ir.get("vocab_size"))
    tied = ir.get("tie_word_embeddings")
    attention = info["dominant"]["spec"]["attention"]
    ffn = info["dominant"]["spec"]["ffn"]

    panels.append(_hint_card(
        "default",
        "Click a block above to inspect it",
    ))
    panels.append(_simple_card(
        "tok_text",
        "Tokenized text",
        f"Token IDs after tokenization · shape [batch, seq_len]",
    ))
    panels.append(_simple_card(
        "embed",
        "Token embedding",
        f"Lookup table · vocab × hidden = {v} × {h}"
        + (" · weights tied with output" if tied else ""),
    ))
    panels.append(_simple_card(
        "rms1",
        "Pre-attention RMSNorm",
        f"RMSNorm · dim {h}",
    ))
    panels.append(_simple_card(
        "attn",
        _attention_title(attention),
        _describe_attention(attention),
    ))
    panels.append(_simple_card(
        "add1",
        "Residual add",
        "Adds block input + attention output",
    ))
    panels.append(_simple_card(
        "rms2",
        "Pre-FFN RMSNorm",
        f"RMSNorm · dim {h}",
    ))

    # The interesting one: clicking the FFN block in the architecture reveals
    # the FULL internal FFN diagram (or the MoE diagram for sparse models).
    if ffn.get("kind") == "moe":
        panels.append(_rich_card(
            "ffn",
            "Mixture of experts",
            _describe_ffn(ffn),
            _build_moe_view(ir, info, mount_id),
        ))
    else:
        panels.append(_rich_card(
            "ffn",
            "Feed-forward block",
            _describe_ffn(ffn),
            _build_ffn_view(ir, info, mount_id),
        ))

    panels.append(_simple_card(
        "add2",
        "Residual add",
        "Adds post-attention + FFN output",
    ))
    panels.append(_simple_card(
        "final_rms",
        "Final RMSNorm",
        f"RMSNorm · dim {h}",
    ))
    panels.append(_simple_card(
        "lm_head",
        "Linear output (LM head)",
        f"Linear · {h} → {v}"
        + (" · weights tied with input embedding" if tied else ""),
    ))

    return "".join(panels)


def _attention_title(attn: dict) -> str:
    kinds = {"mla": "Multi-head latent attention", "gqa": "Grouped-query attention",
             "mqa": "Multi-query attention"}
    return kinds.get(attn.get("kind", ""), "Attention")


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
    h = _fmt_int(ir.get("hidden_size"))
    ffn = info["dominant"]["spec"]["ffn"]
    inter = _fmt_int(ffn.get("expert_intermediate_size") or ffn.get("intermediate_size"))
    activation = (ffn.get("activation") or "silu").upper()

    # Default L3 card is never actually visible (the whole L3 box is hidden
    # until uf-sub-active is set), but kept as a safe target for showL3('default').
    panels.append(_l3_card("default", "", ""))

    # Dense FFN sub-blocks
    panels.append(_l3_card("gate_proj", "Gate projection",
        f"Linear · {h} → {inter} (gated path through {activation})"))
    panels.append(_l3_card("up_proj", "Up projection",
        f"Linear · {h} → {inter}"))
    panels.append(_l3_card("silu", f"{activation} activation",
        "Element-wise non-linearity applied to the gate path"))
    panels.append(_l3_card("mul", "Element-wise multiply",
        f"{activation}(gate) × up — combines the gated and ungated paths"))
    panels.append(_l3_card("down_proj", "Down projection",
        f"Linear · {inter} → {h}"))

    # MoE sub-blocks (only meaningful for sparse models)
    if ffn.get("kind") == "moe":
        n_experts = _fmt_int(ffn.get("num_experts")) if ffn.get("num_experts") else "N"
        n_active = ffn.get("num_experts_per_tok") or "k"
        n_shared = ffn.get("num_shared_experts") or 0
        panels.append(_l3_card("router", "Router",
            f"Linear · {h} → {n_experts} (selects top-{n_active} experts per token)"))
        expert_desc = (
            f"Dense FFN with same shape as above · {h} → {inter} → {h} · "
            f"only top-{n_active} of {n_experts} active per token"
            + (f" · plus {n_shared} shared expert(s) always active" if n_shared else "")
        )
        for eid in ("expert_1", "expert_k", "expert_kp1", "expert_n"):
            panels.append(_l3_card(eid, "Expert FFN", expert_desc))
        panels.append(_l3_card("add_moe", "Weighted sum",
            f"Combines top-{n_active} expert outputs weighted by router probabilities"))

    return "".join(panels)


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


def render_document(ir: dict, mount_id: str) -> str:
    """Render a standalone HTML document."""
    title = f"{ir.get('name', 'model')} - unfolded"
    body = render_fragment(ir, mount_id)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="generator" content="Unfold">
<title>{_html(title)}</title>
<style>
  body {{ margin: 0; padding: 32px 24px; background: #F8FAFC; min-height: 100vh; }}
  .uf-frame {{ max-width: 820px; margin: 0 auto; }}
</style>
</head>
<body>
  <div class="uf-frame">{body}</div>
</body>
</html>
"""


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


def _build_architecture_view(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 720, 920
    arrow_id, shadow_id = _ids(mount_id, "arch")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 26, w - 80, h - 52, C["bg_outer"]))

    cx = w / 2
    inner_x, inner_y, inner_w, inner_h = 110, 200, w - 220, 490
    parts.append(_region_rect(inner_x, inner_y, inner_w, inner_h, C["bg_inner"]))

    attention = info["dominant"]["spec"]["attention"]
    ffn = info["dominant"]["spec"]["ffn"]

    if attention.get("kind") == "mla":
        attn_label = ["Multi-Head Latent", "Attention"]
    elif attention.get("kind") == "gqa":
        attn_label = ["Grouped-Query", "Attention"]
    elif attention.get("kind") == "mqa":
        attn_label = ["Multi-Query", "Attention"]
    else:
        attn_label = ["Multi-Head", "Attention"]

    ffn_label = "MoE" if ffn.get("kind") == "moe" else "Feed-Forward"

    # Layout (top → bottom, smaller y = higher on screen):
    #   lm_head → final_rms → [inner block: add2 → ffn → rms2 → add1 → attn → rms1] → embed → tok_text
    tok_text  = _rect_block(parts, info, shadow_id, "tok_text",  cx - 110, h - 100, 220, 44, "Tokenized text", font_size=17)
    embed     = _rect_block(parts, info, shadow_id, "embed",     cx - 130, h - 168, 260, 44, "Token Embedding layer", font_size=17)
    rms1      = _rect_block(parts, info, shadow_id, "rms1",      cx - 80,  inner_y + 400, 160, 36, "RMSNorm", font_size=16)
    attn      = _rect_block(parts, info, shadow_id, "attn",      cx - 115, inner_y + 305, 230, 60, attn_label, font_size=17)
    add1      = _plus_block(parts, info, shadow_id, "add1",      cx,       inner_y + 270)
    rms2      = _rect_block(parts, info, shadow_id, "rms2",      cx - 80,  inner_y + 195, 160, 36, "RMSNorm", font_size=16)
    ffn_node  = _rect_block(parts, info, shadow_id, "ffn",       cx - 80,  inner_y + 110, 160, 44, ffn_label, font_size=17)
    add2      = _plus_block(parts, info, shadow_id, "add2",      cx,       inner_y + 75)
    final_rms = _rect_block(parts, info, shadow_id, "final_rms", cx - 90,  140, 180, 36, "Final RMSNorm", font_size=16)
    lm_head   = _rect_block(parts, info, shadow_id, "lm_head",   cx - 130, 70,  260, 44, "Linear output layer", font_size=17)

    for src, dst in (
        (tok_text, embed),
        (embed, rms1),
        (rms1, attn),
        (attn, add1),
        (add1, rms2),
        (rms2, ffn_node),
        (ffn_node, add2),
        (add2, final_rms),
        (final_rms, lm_head),
    ):
        parts.append(_v_line(src, dst, arrow_id))

    parts.append(
        _svg_tag(
            "line",
            {
                "x1": cx,
                "y1": lm_head["top"],
                "x2": cx,
                "y2": lm_head["top"] - 32,
                "stroke": C["arrow"],
                "stroke-width": 1.6,
                "stroke-linecap": "round",
                "marker-end": f"url(#{arrow_id})",
                "fill": "none",
            },
        )
    )

    lane = inner_x + inner_w - 28
    parts.append(_residual_loop_right(rms1, add1, lane, arrow_id))
    parts.append(_residual_loop_right(rms2, add2, lane, arrow_id))

    parts.append(
        _svg_tag(
            "rect",
            {
                "x": inner_x + inner_w - 78,
                "y": inner_y + 12,
                "width": 66,
                "height": 26,
                "rx": 13,
                "ry": 13,
                "fill": "rgba(255,255,255,0.65)",
                "stroke": C["border"],
                "stroke-width": 0.5,
            },
        )
    )
    parts.append(
        _svg_text(
            inner_x + inner_w - 45,
            inner_y + 25,
            f"x {len(ir.get('layers', []))}",
            {
                "text-anchor": "middle",
                "dominant-baseline": "central",
                "fill": C["text"],
                "font-family": FONT_HEAD,
                "font-size": 20,
            },
        )
    )

    return _svg(w, h, f"{ir.get('name', 'model')} architecture", parts)


def _build_moe_view(ir: dict, info: dict, mount_id: str) -> str:
    # Slightly taller (h=620) to leave room for incoming arrow below router
    # and outgoing arrow above the sum node.
    w, h = 720, 620
    arrow_id, shadow_id = _ids(mount_id, "moe")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    # Router spans wide enough that all four expert arrows visibly emerge from
    # within the router rectangle (otherwise the leftmost/rightmost arrows
    # appear to start in empty space, far outside the router block).
    router_w = 540
    router = _rect_block(parts, info, shadow_id, "router",
                         (w - router_w) / 2, h - 130, router_w, 50, "Router")
    sum_node = _plus_block(parts, info, shadow_id, "add_moe", cx, 100)

    # Expert layout: 4 evenly-spaced columns symmetric around the centre.
    # Previous version overlapped slot 3 and slot 4 (wrong x for slot 4).
    expert_w, expert_h = 116, 54
    expert_y = 235
    n_total = ffn.get("num_experts")
    last_label = str(n_total) if n_total else "N"
    # Spacing computed so columns are: 60, 220, 380, 540 with w=720.
    side_pad = 60
    gap = (w - 2 * side_pad - 4 * expert_w) / 3
    slots = [
        (side_pad + 0 * (expert_w + gap), "Expert 1",        "expert_1"),
        (side_pad + 1 * (expert_w + gap), "Expert k",        "expert_k"),
        (side_pad + 2 * (expert_w + gap), "Expert k+1",      "expert_kp1"),
        (side_pad + 3 * (expert_w + gap), f"Expert {last_label}", "expert_n"),
    ]
    experts = [
        _rect_block(parts, info, shadow_id, node_id, x, expert_y, expert_w, expert_h, label, font_size=15)
        for x, label, node_id in slots
    ]

    # Ellipsis lives in the gap between expert 2 (k) and expert 3 (k+1).
    dots_x = (experts[1]["right"] + experts[2]["left"]) / 2
    dots_y = expert_y + expert_h / 2
    for i in range(-2, 3):
        parts.append(_svg_tag("circle", {"cx": dots_x + i * 7, "cy": dots_y, "r": 2.5, "fill": C["muted"]}))

    for expert in experts:
        parts.append(_v_seg(expert["cx"], router["top"], expert["bottom"] + GAP, arrow_id))

    for expert in experts:
        target_x = sum_node["cx"] + (-sum_node["r"] - GAP if expert["cx"] < sum_node["cx"] else sum_node["r"] + GAP)
        parts.append(_elbow_vh(expert["cx"], expert["top"], target_x, sum_node["cy"], arrow_id))

    if ffn.get("num_experts") and ffn.get("num_experts_per_tok"):
        sparsity = 100 * ffn["num_experts_per_tok"] / ffn["num_experts"]
        cg_x, cg_y, cg_w, cg_h = w - 224, 56, 184, 56
        parts.append(
            _svg_tag(
                "rect",
                {
                    "x": cg_x,
                    "y": cg_y,
                    "width": cg_w,
                    "height": cg_h,
                    "rx": 10,
                    "ry": 10,
                    "fill": C["bg_card"],
                    "stroke": C["border"],
                    "stroke-width": 0.5,
                },
            )
        )
        parts.append(
            _svg_text(
                cg_x + 12,
                cg_y + 18,
                "ACTIVE PER TOKEN",
                {
                    "fill": C["muted"],
                    "font-family": FONT_BODY,
                    "font-size": 10,
                    "letter-spacing": "0.12em",
                    "font-weight": 600,
                },
            )
        )
        parts.append(
            _svg_text(
                cg_x + 12,
                cg_y + 44,
                f"{ffn['num_experts_per_tok']} / {ffn['num_experts']}  -  {sparsity:.1f}%",
                {"fill": C["text"], "font-family": FONT_HEAD, "font-size": 22},
            )
        )

    # Outgoing arrow at the top — leaves the weighted-sum node going up.
    parts.append(_svg_tag("line", {
        "x1": sum_node["cx"], "y1": sum_node["top"],
        "x2": sum_node["cx"], "y2": sum_node["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        sum_node["cx"], sum_node["top"] - 46,
        "out",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Incoming arrow at the bottom — enters the router from below.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": router["bottom"] + 36,
        "x2": cx, "y2": router["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, router["bottom"] + 50,
        "in",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    return _svg(w, h, f"{ir.get('name', 'model')} mixture of experts", parts)


def _build_ffn_view(ir: dict, info: dict, mount_id: str) -> str:
    # h=640 (was 600) to fit incoming arrow + label under the input branch.
    w, h = 720, 640
    arrow_id, shadow_id = _ids(mount_id, "ffn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    act_name = (ffn.get("activation") or "silu").upper()

    down_proj = _rect_block(parts, info, shadow_id, "down_proj", cx - 90, 90, 180, 50, "Linear (down)")
    mul_node = _plus_block(parts, info, shadow_id, "mul", cx, 210, "x")
    silu = _rect_block(parts, info, shadow_id, "silu", cx - 270, 310, 180, 50, act_name)
    up_proj = _rect_block(parts, info, shadow_id, "up_proj", cx + 90, 310, 180, 50, "Linear (up)")
    gate_proj = _rect_block(parts, info, shadow_id, "gate_proj", cx - 270, 440, 180, 50, "Linear (gate)")

    branch_y = h - 110
    parts.append(_svg_tag("circle", {"cx": cx, "cy": branch_y, "r": 4, "fill": C["arrow"]}))
    parts.append(_elbow_hv(cx, branch_y, gate_proj["cx"], gate_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(cx, branch_y, up_proj["cx"], up_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_line(gate_proj, silu, arrow_id))
    parts.append(_elbow_vh(silu["cx"], silu["top"], mul_node["cx"] - mul_node["r"] - GAP, mul_node["cy"], arrow_id))
    parts.append(_elbow_vh(up_proj["cx"], up_proj["top"], mul_node["cx"] + mul_node["r"] + GAP, mul_node["cy"], arrow_id))
    parts.append(_v_line(mul_node, down_proj, arrow_id))

    # Outgoing arrow at top — leaves down_proj going up.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": down_proj["top"],
        "x2": cx, "y2": down_proj["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, down_proj["top"] - 46,
        "out",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Incoming arrow at the bottom — points up into the input branch dot.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": branch_y + 38,
        "x2": cx, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(
        _svg_text(
            cx,
            h - 24,
            "in  ·  x",
            {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
        )
    )
    parts.append(
        _svg_text(
            cx,
            50,
            f"intermediate hidden = {_fmt_int(ffn.get('expert_intermediate_size') or ffn.get('intermediate_size'))}",
            {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
        )
    )

    return _svg(w, h, f"{ir.get('name', 'model')} feed-forward block", parts)


def _build_layer_map(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 720, 240
    arrow_id, shadow_id = _ids(mount_id, "map")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_card"], stroke=C["border"], stroke_width=0.5))

    palette = ["#0F6E56", "#1D9E75", "#0E7C8C", "#3C3489", "#993C1D", "#185FA5", "#65A30D"]
    sig_to_color = {group["sig"]: palette[i % len(palette)] for i, group in enumerate(info["groups"])}

    strip_x, strip_y, strip_w, strip_h = 80, 90, w - 160, 36
    layers = ir.get("layers", [])
    n = len(layers)
    col_w = strip_w / max(n, 1)

    for i, layer in enumerate(layers):
        sig = _signature(layer)
        parts.append(
            _svg_tag(
                "rect",
                {
                    "x": strip_x + i * col_w,
                    "y": strip_y,
                    "width": max(col_w - 0.5, 1),
                    "height": strip_h,
                    "fill": sig_to_color.get(sig, palette[0]),
                    "opacity": 0.95,
                },
            )
        )

    parts.append(
        _svg_tag(
            "rect",
            {
                "x": strip_x,
                "y": strip_y,
                "width": strip_w,
                "height": strip_h,
                "fill": "none",
                "stroke": C["text"],
                "stroke-width": 0.4,
                "rx": 4,
                "ry": 4,
            },
        )
    )

    if n:
        for idx in (0, n - 1):
            x = strip_x + (idx + 0.5) * col_w
            parts.append(
                _svg_text(
                    x,
                    strip_y + strip_h + 16,
                    f"L{idx}",
                    {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
                )
            )

    type_word = "type" if len(info["groups"]) == 1 else "types"
    parts.append(
        _svg_text(
            strip_x,
            70,
            f"{n} layers - {len(info['groups'])} {type_word}",
            {"fill": C["text"], "font-family": FONT_BODY, "font-size": 12, "font-weight": 600},
        )
    )

    lx, ly = strip_x, strip_y + strip_h + 44
    for group in info["groups"]:
        spec = group["spec"]
        ffn_kind = "MoE" if spec["ffn"].get("kind") == "moe" else "Dense"
        mask = "SWA" if spec["attention"].get("mask") == "sliding" else "full"
        first, last = group["indices"][0], group["indices"][-1]
        label = (
            f"{spec['attention'].get('kind', '').upper()} + {ffn_kind} ({mask}) - "
            f"L{first}-L{last} - {len(group['indices'])}x"
        )
        color = sig_to_color[group["sig"]]
        parts.append(_svg_tag("rect", {"x": lx, "y": ly - 9, "width": 12, "height": 12, "fill": color, "rx": 2}))
        parts.append(
            _svg_text(
                lx + 18,
                ly,
                label,
                {"dominant-baseline": "central", "fill": C["text"], "font-family": FONT_BODY, "font-size": 12},
            )
        )
        ly += 20

    return _svg(w, h, f"{ir.get('name', 'model')} layer map", parts)


def _make_info(ir: dict) -> dict:
    groups = []
    cur = None
    for layer in ir.get("layers", []):
        sig = _signature(layer)
        if cur and cur["sig"] == sig:
            cur["indices"].append(layer.get("index", len(cur["indices"])))
        else:
            cur = {"sig": sig, "indices": [layer.get("index", 0)], "spec": layer}
            groups.append(cur)

    if groups:
        dominant = max(groups, key=lambda group: len(group["indices"]))
    else:
        dominant = {
            "sig": "",
            "indices": [],
            "spec": {
                "attention": {"kind": "mha", "num_heads": 0, "num_kv_heads": 0},
                "ffn": {"kind": "dense", "activation": "silu", "intermediate_size": 0, "gated": True},
            },
        }

    return {"groups": groups, "dominant": dominant, "meta": _meta_for(ir, dominant["spec"])}


def _meta_for(ir: dict, spec: dict) -> dict:
    """Tooltip / detail-card text for one layer-type's spec.  Re-computed per
    variant so a heterogeneous model (e.g. DeepSeek-V3 dense + MoE) gets
    correct tooltips for whichever layer type is currently displayed."""
    attention = spec.get("attention", {})
    ffn = spec.get("ffn", {})
    hidden = _fmt_int(ir.get("hidden_size"))
    vocab = _fmt_int(ir.get("vocab_size"))
    return {
        "tok_text": ("Tokenized text", "Input token IDs; shape [batch, seq_len]"),
        "embed": (
            "Token embedding",
            f"{vocab} x {hidden}" + (" (tied with output)" if ir.get("tie_word_embeddings") else ""),
        ),
        "rms1": ("Pre-attention norm", f"RMSNorm; dim {hidden}"),
        "attn": ("Attention", _describe_attention(attention)),
        "add1": ("Residual add", "block input + attention output"),
        "rms2": ("Pre-FFN norm", f"RMSNorm; dim {hidden}"),
        "ffn": ("Mixture of experts" if ffn.get("kind") == "moe" else "Feed-forward", _describe_ffn(ffn)),
        "add2": ("Residual add", "post-attention + FFN output"),
        "final_rms": ("Final norm", f"RMSNorm; dim {hidden}"),
        "lm_head": (
            "LM head",
            f"{hidden} -> {vocab}" + (" (tied)" if ir.get("tie_word_embeddings") else ""),
        ),
        "router": ("Router", f"Routes tokens to top-{ffn.get('num_experts_per_tok') or 'k'} experts"),
        "add_moe": ("Weighted sum", "Combines selected expert outputs"),
        "expert_1": ("Expert", _describe_ffn(ffn)),
        "expert_k": ("Expert", _describe_ffn(ffn)),
        "expert_kp1": ("Expert", _describe_ffn(ffn)),
        "expert_n": ("Expert", _describe_ffn(ffn)),
        "down_proj": ("Down projection", f"intermediate -> hidden ({hidden})"),
        "mul": ("Gate product", "activation(gate) x up projection"),
        "silu": ("Activation", (ffn.get("activation") or "silu").upper()),
        "up_proj": ("Up projection", f"hidden -> {_fmt_int(ffn.get('expert_intermediate_size') or ffn.get('intermediate_size'))}"),
        "gate_proj": ("Gate projection", f"hidden -> {_fmt_int(ffn.get('expert_intermediate_size') or ffn.get('intermediate_size'))}"),
    }


def _group_label(group: dict) -> str:
    """Short human label for a layer-type group, used on the toggle pill."""
    spec = group["spec"]
    indices = group["indices"]
    bits = []
    if spec.get("attention", {}).get("mask") == "sliding":
        bits.append("SWA")
    bits.append(spec.get("attention", {}).get("kind", "?").upper())
    bits.append("MoE" if spec.get("ffn", {}).get("kind") == "moe" else "Dense")
    label = " · ".join(bits)
    span = f"L{indices[0]}" if len(indices) == 1 else f"L{indices[0]}–L{indices[-1]}"
    return f"{label}  ({span} · {len(indices)}×)"


def _signature(layer: dict) -> str:
    attention = layer.get("attention", {})
    ffn = layer.get("ffn", {})
    return "|".join(
        str(value)
        for value in (
            attention.get("kind"),
            attention.get("mask"),
            attention.get("window_size"),
            ffn.get("kind"),
            ffn.get("num_experts"),
            layer.get("norm_kind"),
            layer.get("norm_placement"),
        )
    )


def _describe_attention(attention: dict) -> str:
    kind = attention.get("kind")
    if kind == "mla":
        text = (
            f"Multi-head latent attention; {attention.get('num_heads')} heads; "
            f"KV LoRA {_fmt_int(attention.get('kv_lora_rank'))}"
        )
        if attention.get("q_lora_rank"):
            text += f"; Q LoRA {_fmt_int(attention.get('q_lora_rank'))}"
        return text
    if kind == "gqa":
        return (
            f"Grouped-query; {attention.get('num_heads')} Q / "
            f"{attention.get('num_kv_heads')} KV heads; head dim {_fmt_int(attention.get('head_dim'))}"
        )
    if kind == "mqa":
        return f"Multi-query; {attention.get('num_heads')} Q / 1 KV head"
    return f"Multi-head; {attention.get('num_heads')} heads; head dim {_fmt_int(attention.get('head_dim'))}"


def _describe_ffn(ffn: dict) -> str:
    if ffn.get("kind") == "moe":
        text = f"MoE; {_fmt_int(ffn.get('num_experts'))} experts; top-{ffn.get('num_experts_per_tok')}"
        if ffn.get("num_shared_experts"):
            text += f" + {ffn.get('num_shared_experts')} shared"
        if ffn.get("num_experts") and ffn.get("num_experts_per_tok"):
            text += f"; {100 * ffn['num_experts_per_tok'] / ffn['num_experts']:.1f}% active"
        text += f"; expert hidden {_fmt_int(ffn.get('expert_intermediate_size') or ffn.get('intermediate_size'))}"
        return text
    gated = "gated " if ffn.get("gated") else ""
    return f"{gated}FFN; {ffn.get('activation')}; hidden {_fmt_int(ffn.get('intermediate_size'))}"


def _arch_badges(ir: dict, info: dict) -> list[dict[str, str]]:
    badges = []
    attention = info["dominant"]["spec"]["attention"]
    ffn = info["dominant"]["spec"]["ffn"]
    kind = attention.get("kind")

    if kind == "mla":
        badges.append({"text": "MLA", "title": "Multi-head latent attention"})
    elif kind == "gqa":
        badges.append({"text": f"GQA {attention.get('num_heads')}/{attention.get('num_kv_heads')}", "title": "Grouped-query attention"})
    elif kind == "mqa":
        badges.append({"text": "MQA", "title": "Multi-query attention"})
    else:
        badges.append({"text": "MHA", "title": "Multi-head attention"})

    if ffn.get("kind") == "moe":
        badges.append(
            {
                "text": f"MoE {ffn.get('num_experts_per_tok')}/{ffn.get('num_experts')}",
                "title": f"Mixture of experts; top-{ffn.get('num_experts_per_tok')} of {ffn.get('num_experts')}",
            }
        )
    else:
        badges.append({"text": "Dense FFN", "title": "Dense feed-forward"})

    if len(info["groups"]) > 1:
        badges.append({"text": f"{len(info['groups'])} layer types", "title": ""})
    if attention.get("mask") == "sliding":
        badges.append({"text": f"SWA {_fmt_int(attention.get('window_size'))}", "title": "Sliding-window attention"})
    return badges


def _ids(mount_id: str, view: str) -> tuple[str, str]:
    return f"{mount_id}-{view}-arrow", f"{mount_id}-{view}-shadow"


def _defs(arrow_id: str, shadow_id: str) -> str:
    marker = _svg_tag(
        "marker",
        {
            "id": arrow_id,
            "viewBox": "0 0 10 10",
            "refX": 8,
            "refY": 5,
            "markerWidth": 6,
            "markerHeight": 6,
            "orient": "auto-start-reverse",
        },
        _svg_tag(
            "path",
            {
                "d": "M2 1L8 5L2 9",
                "fill": "none",
                "stroke": "context-stroke",
                "stroke-width": 1.5,
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        ),
    )
    shadow = _svg_tag(
        "filter",
        {"id": shadow_id, "x": "-20%", "y": "-20%", "width": "140%", "height": "140%"},
        "".join(
            [
                _svg_tag("feGaussianBlur", {"in": "SourceAlpha", "stdDeviation": 1}),
                _svg_tag("feOffset", {"dx": 0, "dy": 1, "result": "off"}),
                _svg_tag(
                    "feComponentTransfer",
                    {},
                    _svg_tag("feFuncA", {"type": "linear", "slope": "0.16"}),
                ),
                _svg_tag(
                    "feMerge",
                    {},
                    _svg_tag("feMergeNode", {})
                    + _svg_tag("feMergeNode", {"in": "SourceGraphic"}),
                ),
            ]
        ),
    )
    return _svg_tag("defs", {}, marker + shadow)


def _region_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", stroke_width: float = 0) -> str:
    return _svg_tag(
        "rect",
        {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rx": 18,
            "ry": 18,
            "fill": fill,
            "stroke": stroke,
            "stroke-width": stroke_width,
        },
    )


def _rect_block(
    parts: list[str],
    info: dict,
    shadow_id: str,
    node_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str | list[str],
    font_size: int = 18,
) -> dict:
    lines = label if isinstance(label, list) else [label]
    line_h = font_size + 3
    start_y = y + h / 2 - ((len(lines) - 1) * line_h) / 2

    children = [_node_title(info, node_id)]
    children.append(
        _svg_tag(
            "rect",
            {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "rx": 11,
                "ry": 11,
                "fill": C["block"],
                "stroke": C["block_alt"],
                "stroke-width": 0.6,
                "filter": f"url(#{shadow_id})",
            },
        )
    )
    for i, line in enumerate(lines):
        children.append(
            _svg_text(
                x + w / 2,
                start_y + i * line_h,
                line,
                {
                    "text-anchor": "middle",
                    "dominant-baseline": "central",
                    "fill": C["text_block"],
                    "font-family": FONT_HEAD,
                    "font-size": font_size,
                    "pointer-events": "none",
                },
            )
        )
    parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    return {
        "left": x,
        "right": x + w,
        "top": y,
        "bottom": y + h,
        "cx": x + w / 2,
        "cy": y + h / 2,
        "w": w,
        "h": h,
    }


def _plus_block(parts: list[str], info: dict, shadow_id: str, node_id: str, cx: float, cy: float, sym: str = "+") -> dict:
    r = 14
    children = [
        _node_title(info, node_id),
        _svg_tag(
            "circle",
            {
                "cx": cx,
                "cy": cy,
                "r": r,
                "fill": C["block"],
                "stroke": C["block_alt"],
                "stroke-width": 0.6,
                "filter": f"url(#{shadow_id})",
            },
        ),
        _svg_text(
            cx,
            cy + 1,
            sym,
            {
                "text-anchor": "middle",
                "dominant-baseline": "central",
                "fill": C["text_block"],
                "font-family": FONT_HEAD,
                "font-size": 22,
                "pointer-events": "none",
            },
        ),
    ]
    parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    return {"left": cx - r, "right": cx + r, "top": cy - r, "bottom": cy + r, "cx": cx, "cy": cy, "r": r}


def _node_title(info: dict, node_id: str) -> str:
    meta = info["meta"].get(node_id)
    if not meta:
        return ""
    return _svg_tag("title", {}, f"{_html(meta[0])}: {_html(meta[1])}")


def _v_line(src: dict, dst: dict, arrow_id: str) -> str:
    if src["cy"] > dst["cy"]:
        y1 = src["top"]
        y2 = dst["bottom"] + GAP
    else:
        y1 = src["bottom"]
        y2 = dst["top"] - GAP
    return _svg_tag(
        "line",
        {
            "x1": src["cx"],
            "y1": y1,
            "x2": src["cx"],
            "y2": y2,
            "stroke": C["arrow"],
            "stroke-width": 1.6,
            "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})",
            "fill": "none",
        },
    )


def _v_seg(x: float, y1: float, y2: float, arrow_id: str) -> str:
    return _svg_tag(
        "line",
        {
            "x1": x,
            "y1": y1,
            "x2": x,
            "y2": y2,
            "stroke": C["arrow"],
            "stroke-width": 1.6,
            "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})",
            "fill": "none",
        },
    )


def _elbow_vh(x1: float, y1: float, x2: float, y2: float, arrow_id: str) -> str:
    if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
        d = f"M {_num(x1)} {_num(y1)} L {_num(x2)} {_num(y2)}"
    else:
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        r = min(10, abs(x2 - x1) / 2, abs(y2 - y1) / 2)
        d = (
            f"M {_num(x1)} {_num(y1)} "
            f"L {_num(x1)} {_num(y2 - sy * r)} "
            f"Q {_num(x1)} {_num(y2)} {_num(x1 + sx * r)} {_num(y2)} "
            f"L {_num(x2)} {_num(y2)}"
        )
    return _path(d, arrow_id)


def _elbow_hv(x1: float, y1: float, x2: float, y2: float, arrow_id: str) -> str:
    if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
        d = f"M {_num(x1)} {_num(y1)} L {_num(x2)} {_num(y2)}"
    else:
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        r = min(10, abs(x2 - x1) / 2, abs(y2 - y1) / 2)
        d = (
            f"M {_num(x1)} {_num(y1)} "
            f"L {_num(x2 - sx * r)} {_num(y1)} "
            f"Q {_num(x2)} {_num(y1)} {_num(x2)} {_num(y1 + sy * r)} "
            f"L {_num(x2)} {_num(y2)}"
        )
    return _path(d, arrow_id)


def _residual_loop_right(src: dict, dst: dict, lane: float, arrow_id: str) -> str:
    r = 12
    start_x, start_y = src["right"], src["cy"]
    end_x, end_y = dst["right"], dst["cy"]
    d = (
        f"M {_num(start_x)} {_num(start_y)} "
        f"L {_num(lane - r)} {_num(start_y)} "
        f"Q {_num(lane)} {_num(start_y)} {_num(lane)} {_num(start_y - r)} "
        f"L {_num(lane)} {_num(end_y + r)} "
        f"Q {_num(lane)} {_num(end_y)} {_num(lane - r)} {_num(end_y)} "
        f"L {_num(end_x + GAP)} {_num(end_y)}"
    )
    return _path(d, arrow_id)


def _path(d: str, arrow_id: str) -> str:
    return _svg_tag(
        "path",
        {
            "d": d,
            "fill": "none",
            "stroke": C["arrow"],
            "stroke-width": 1.6,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "marker-end": f"url(#{arrow_id})",
        },
    )


def _svg(w: int, h: int, title: str, parts: list[str]) -> str:
    return _svg_tag(
        "svg",
        {"width": "100%", "viewBox": f"0 0 {w} {h}", "role": "img", "xmlns": "http://www.w3.org/2000/svg"},
        _svg_tag("title", {}, _html(title)) + "".join(parts),
    )


def _svg_text(x: float, y: float, text: Any, attrs: dict[str, Any] | None = None) -> str:
    attrs = dict(attrs or {})
    attrs.update({"x": x, "y": y})
    return _svg_tag("text", attrs, _html(text))


def _svg_tag(name: str, attrs: dict[str, Any] | None = None, content: str | None = None) -> str:
    attr_text = "".join(
        f' {key}="{_attr(_num(value))}"'
        for key, value in (attrs or {}).items()
        if value is not None
    )
    if content is None:
        return f"<{name}{attr_text}/>"
    return f"<{name}{attr_text}>{content}</{name}>"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "?"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _num(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _html(value: Any) -> str:
    return escape(str(value), quote=False)


def _attr(value: Any) -> str:
    return escape(str(value), quote=True)
