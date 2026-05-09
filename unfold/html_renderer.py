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
    """Render a complete, script-free HTML fragment."""
    info = _make_info(ir)
    views = _views(ir, info, mount_id)
    style = _style(mount_id, views)

    inputs = []
    labels = []
    sections = []
    for i, view in enumerate(views):
        view_id, label, svg = view
        input_id = f"{mount_id}-{view_id}"
        checked = " checked" if i == 0 else ""
        inputs.append(
            f'<input class="uf-toggle" type="radio" name="{_attr(mount_id)}-tabs" '
            f'id="{_attr(input_id)}"{checked}>'
        )
        labels.append(f'<label class="uf-tab" for="{_attr(input_id)}">{_html(label)}</label>')
        sections.append(f'<section class="uf-view uf-view-{_attr(view_id)}">{svg}</section>')

    return f"""
<div id="{_attr(mount_id)}" class="uf-root">
<style>
{style}
</style>
{_header(ir, info)}
{_stats_banner(ir)}
{''.join(inputs)}
<div class="uf-tabs" role="tablist">{''.join(labels)}</div>
<div class="uf-canvas">{''.join(sections)}</div>
{_details_panel(info)}
</div>
"""


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


def _views(ir: dict, info: dict, mount_id: str) -> list[tuple[str, str, str]]:
    ffn = info["dominant"]["spec"]["ffn"]
    views = [
        ("arch", "Architecture", _build_architecture_view(ir, info, mount_id)),
    ]
    if ffn.get("kind") == "moe":
        views.append(("moe", "MoE", _build_moe_view(ir, info, mount_id)))
    views.extend(
        [
            ("ffn", "Expert" if ffn.get("kind") == "moe" else "FFN", _build_ffn_view(ir, info, mount_id)),
            ("map", "Layer map", _build_layer_map(ir, info, mount_id)),
        ]
    )
    return views


def _style(mount_id: str, views: list[tuple[str, str, str]]) -> str:
    tab_rules = []
    for view_id, _, _ in views:
        input_id = f"{mount_id}-{view_id}"
        tab_rules.append(
            f"#{mount_id} #{input_id}:checked ~ .uf-canvas .uf-view-{view_id} {{ display:block; }}"
        )
        tab_rules.append(
            f"#{mount_id} #{input_id}:checked ~ .uf-tabs label[for='{input_id}'] "
            f"{{ background:{C['block']}; color:#FFFFFF; }}"
        )

    return f"""
#{mount_id} {{
  font-family:{FONT_BODY};
  color:{C['text']};
}}
#{mount_id} .uf-header {{
  margin-bottom:12px;
}}
#{mount_id} .uf-name {{
  font-family:{FONT_HEAD};
  font-size:30px;
  line-height:1;
  color:{C['text']};
}}
#{mount_id} .uf-arch {{
  color:{C['muted']};
  font-size:12px;
  margin-top:4px;
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
  border-radius:10px;
  overflow:hidden;
  margin-bottom:14px;
}}
#{mount_id} .uf-stat {{
  padding:10px 14px;
  background:{C['bg_card']};
}}
#{mount_id} .uf-stat-key {{
  font-size:10px;
  letter-spacing:0.12em;
  color:{C['muted']};
  font-weight:600;
}}
#{mount_id} .uf-stat-val {{
  font-family:{FONT_HEAD};
  font-size:22px;
  color:{C['text']};
  margin-top:2px;
  line-height:1.05;
}}
#{mount_id} .uf-toggle {{
  position:absolute;
  opacity:0;
  pointer-events:none;
}}
#{mount_id} .uf-tabs {{
  display:flex;
  gap:4px;
  margin-bottom:10px;
  padding:4px;
  background:{C['bg_card']};
  border:0.5px solid {C['border']};
  border-radius:9px;
  width:fit-content;
}}
#{mount_id} .uf-tab {{
  appearance:none;
  border:0;
  background:transparent;
  color:{C['muted']};
  padding:7px 14px;
  border-radius:6px;
  font:600 12px {FONT_BODY};
  cursor:pointer;
  transition:background .15s,color .15s;
}}
#{mount_id} .uf-tab:hover {{
  background:{C['badge_bg']};
  color:{C['text']};
}}
#{mount_id} .uf-canvas {{
  background:{C['canvas']};
  border:0.5px solid {C['border']};
  border-radius:12px;
  padding:8px;
}}
#{mount_id} .uf-view {{
  display:none;
}}
#{mount_id} .uf-view svg {{
  display:block;
  max-width:100%;
  height:auto;
  animation:uf-fade .35s ease-out;
}}
#{mount_id} .uf-node rect,
#{mount_id} .uf-node circle {{
  transition:filter .15s;
}}
#{mount_id} .uf-node:hover rect,
#{mount_id} .uf-node:hover circle {{
  filter:brightness(1.08) drop-shadow(0 2px 4px rgba(0,0,0,.18));
}}
#{mount_id} .uf-panel {{
  margin-top:10px;
  padding:12px 14px;
  background:{C['bg_card']};
  border:0.5px solid {C['border']};
  border-radius:10px;
  font-size:13px;
  color:{C['muted']};
  line-height:1.45;
}}
#{mount_id} .uf-panel summary {{
  cursor:pointer;
  color:{C['text']};
  font-weight:600;
}}
#{mount_id} .uf-detail-grid {{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:8px;
  margin-top:10px;
}}
#{mount_id} .uf-detail {{
  border-top:1px solid {C['border']};
  padding-top:8px;
}}
#{mount_id} .uf-detail-title {{
  color:{C['text']};
  font-weight:600;
}}
@keyframes uf-fade {{ from {{ opacity:0; transform:translateY(4px); }} to {{ opacity:1; transform:none; }} }}
{chr(10).join(tab_rules)}
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


def _details_panel(info: dict) -> str:
    details = []
    for key in ("embed", "rms1", "attn", "ffn", "final_rms", "lm_head"):
        title, desc = info["meta"][key]
        details.append(
            '<div class="uf-detail">'
            f'<div class="uf-detail-title">{_html(title)}</div>'
            f'<div>{_html(desc)}</div>'
            '</div>'
        )
    return (
        '<details class="uf-panel" open>'
        '<summary>Block details</summary>'
        '<div style="margin-top:6px">Hover blocks in the diagram for quick tooltips; '
        'the key dimensions are listed below.</div>'
        f'<div class="uf-detail-grid">{"".join(details)}</div>'
        '</details>'
    )


def _build_architecture_view(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 720, 1080
    arrow_id, shadow_id = _ids(mount_id, "arch")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    inner_x, inner_y, inner_w, inner_h = 110, 240, w - 220, 580
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

    tok_text = _rect_block(parts, info, shadow_id, "tok_text", cx - 110, h - 120, 220, 48, "Tokenized text", font_size=18)
    embed = _rect_block(parts, info, shadow_id, "embed", cx - 130, h - 200, 260, 48, "Token Embedding layer")
    rms1 = _rect_block(parts, info, shadow_id, "rms1", cx - 80, inner_y + 470, 160, 40, "RMSNorm", font_size=18)
    attn = _rect_block(parts, info, shadow_id, "attn", cx - 115, inner_y + 360, 230, 70, attn_label, font_size=18)
    add1 = _plus_block(parts, info, shadow_id, "add1", cx, inner_y + 320)
    rms2 = _rect_block(parts, info, shadow_id, "rms2", cx - 80, inner_y + 230, 160, 40, "RMSNorm", font_size=18)
    ffn_node = _rect_block(parts, info, shadow_id, "ffn", cx - 80, inner_y + 130, 160, 50, ffn_label)
    add2 = _plus_block(parts, info, shadow_id, "add2", cx, inner_y + 90)
    final_rms = _rect_block(parts, info, shadow_id, "final_rms", cx - 90, 170, 180, 40, "Final RMSNorm", font_size=18)
    lm_head = _rect_block(parts, info, shadow_id, "lm_head", cx - 130, 90, 260, 48, "Linear output layer")

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
    w, h = 720, 580
    arrow_id, shadow_id = _ids(mount_id, "moe")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    router = _rect_block(parts, info, shadow_id, "router", cx - 80, h - 110, 160, 50, "Router")
    sum_node = _plus_block(parts, info, shadow_id, "add_moe", cx, 80)

    expert_y, expert_w, expert_h = 240, 130, 60
    slots = [
        (110, ["Feed forward", "(expert 1)"], "expert_1"),
        (270, ["Feed forward", "(expert k)"], "expert_k"),
        (430, ["Feed forward", "(expert k+1)"], "expert_kp1"),
        (w - 110 - expert_w, ["Feed forward", f"(expert {ffn.get('num_experts') or 'N'})"], "expert_n"),
    ]
    experts = [
        _rect_block(parts, info, shadow_id, node_id, x, expert_y, expert_w, expert_h, label, font_size=14)
        for x, label, node_id in slots
    ]

    dots_x = (experts[1]["right"] + experts[2]["left"]) / 2
    dots_y = expert_y + expert_h / 2
    for i in range(-2, 3):
        parts.append(_svg_tag("circle", {"cx": dots_x + i * 7, "cy": dots_y, "r": 2.5, "fill": C["muted"]}))

    parts.append(
        _svg_text(
            experts[3]["right"],
            experts[3]["bottom"] + 22,
            f"({ffn.get('num_experts') or 'N'})",
            {
                "text-anchor": "end",
                "fill": C["text"],
                "font-family": FONT_HEAD,
                "font-size": 18,
                "font-style": "italic",
            },
        )
    )

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

    parts.append(
        _svg_text(
            cx,
            h - 22,
            f"top-{ffn.get('num_experts_per_tok') or 'k'} of {ffn.get('num_experts') or 'N'} experts active per token",
            {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_HEAD, "font-size": 18},
        )
    )

    return _svg(w, h, f"{ir.get('name', 'model')} mixture of experts", parts)


def _build_ffn_view(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 720, 600
    arrow_id, shadow_id = _ids(mount_id, "ffn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    act_name = (ffn.get("activation") or "silu").upper()

    down_proj = _rect_block(parts, info, shadow_id, "down_proj", cx - 90, 80, 180, 50, "Linear (down)")
    mul_node = _plus_block(parts, info, shadow_id, "mul", cx, 200, "x")
    silu = _rect_block(parts, info, shadow_id, "silu", cx - 270, 300, 180, 50, act_name)
    up_proj = _rect_block(parts, info, shadow_id, "up_proj", cx + 90, 300, 180, 50, "Linear (up)")
    gate_proj = _rect_block(parts, info, shadow_id, "gate_proj", cx - 270, 430, 180, 50, "Linear (gate)")

    branch_y = h - 70
    parts.append(_svg_tag("circle", {"cx": cx, "cy": branch_y, "r": 4, "fill": C["arrow"]}))
    parts.append(_elbow_hv(cx, branch_y, gate_proj["cx"], gate_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(cx, branch_y, up_proj["cx"], up_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_line(gate_proj, silu, arrow_id))
    parts.append(_elbow_vh(silu["cx"], silu["top"], mul_node["cx"] - mul_node["r"] - GAP, mul_node["cy"], arrow_id))
    parts.append(_elbow_vh(up_proj["cx"], up_proj["top"], mul_node["cx"] + mul_node["r"] + GAP, mul_node["cy"], arrow_id))
    parts.append(_v_line(mul_node, down_proj, arrow_id))
    parts.append(
        _svg_tag(
            "line",
            {
                "x1": cx,
                "y1": down_proj["top"],
                "x2": cx,
                "y2": down_proj["top"] - 32,
                "stroke": C["arrow"],
                "stroke-width": 1.6,
                "stroke-linecap": "round",
                "marker-end": f"url(#{arrow_id})",
                "fill": "none",
            },
        )
    )

    parts.append(
        _svg_text(
            cx,
            h - 22,
            "x (input)",
            {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_HEAD, "font-size": 18},
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

    attention = dominant["spec"]["attention"]
    ffn = dominant["spec"]["ffn"]
    hidden = _fmt_int(ir.get("hidden_size"))
    vocab = _fmt_int(ir.get("vocab_size"))

    meta = {
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

    return {"groups": groups, "dominant": dominant, "meta": meta}


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
