"""Top-level HTML document and fragment rendering."""
from __future__ import annotations

from .components import (
    _build_inspect_cards,
    _build_sub_inspect_cards,
    _click_script,
    _details_section,
    _header,
    _stats_banner,
    _style,
)
from .metadata import _block_lookup, _group_label, _make_info, _meta_for
from .theme import C, FONT_IMPORT, FONT_LINK
from .utils import _attr, _html
from .views import _build_architecture_view, _build_layer_map


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
            "blocks": _block_lookup(ir, group["spec"]),
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
            f'{_html(_group_label(group, info))}</label>'
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
    n_layers = len(ir.get("layers", []))
    if n_groups <= 1:
        map_sub = "All layers structurally identical"
    elif info.get("period") and info["period"] < n_layers:
        cycles = n_layers // info["period"]
        map_sub = (
            f"{n_groups} layer types  ·  {info['period']}-layer cycle ×{cycles}"
        )
    else:
        map_sub = f"{n_groups} layer types across {n_layers} layers"
    layer_map_section = _details_section("LAYER MAP", map_sub, map_svg)

    return f"""
<div id="{_attr(mount_id)}" class="uf-root">
<style>
{FONT_IMPORT}
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{_attr(FONT_LINK)}">
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
