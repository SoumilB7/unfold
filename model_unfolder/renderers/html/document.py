"""Top-level HTML document and fragment rendering."""
from __future__ import annotations
from .cards import _build_inspect_cards, _build_nested_inspect_panels
from .evidence import _code_evidence_section
from .interactions import _click_script
from .metadata import _block_lookup, _group_label, _make_info, _meta_for
from .sections import _details_section, _header, _stats_banner
from .styles import _style
from .theme import C, FONT_IMPORT, FONT_LINK, use_theme
from .utils import _attr, _html
from .views import _build_architecture_view, _build_layer_map


def render_fragment(ir: dict, mount_id: str, include_font_import: bool = True) -> str:
    """Render a complete HTML fragment.

    For heterogeneous models (multiple layer-type groups, e.g. DeepSeek-V3's
    dense → MoE phase change), one architecture / L2 / L3 panel is emitted
    per group.  When ``view_variants`` are present in extras (e.g. GPT-NeoX
    parallel vs sequential), one arch panel is emitted per (group × variant)
    combination.  All panels share the same DOM tree, hidden via CSS and
    toggled by radio buttons in a pill bar.  Pure-CSS toggle, no JS.

    The diagram is themed by ``extras["render"]["theme"]`` (e.g. ``"blue"`` for
    diffusion); the whole fragment is built under that palette and restored after.
    """
    render = (ir.get("extras") or {}).get("render") or {}
    with use_theme(render.get("theme")):
        # Diffusion models render as a sampling loop (the hero) with the DiT
        # denoiser drilling out of it — a different top-level topology than the
        # transformer layer stack, so it has its own fragment builder.
        if render.get("family") == "diffusion":
            from .views_diffusion import render_diffusion_fragment
            return render_diffusion_fragment(ir, mount_id, include_font_import)
        if render.get("layout") == "block_diffusion":
            from .views_diffusion import render_block_diffusion_fragment
            return render_block_diffusion_fragment(ir, mount_id, include_font_import)
        return _render_fragment_body(ir, mount_id, include_font_import)


def _render_fragment_body(ir: dict, mount_id: str, include_font_import: bool) -> str:
    info = _make_info(ir)
    groups = info["groups"]
    dominant_idx = groups.index(info["dominant"]) if groups else 0

    # view_variants allow a single layer type to present multiple topology views
    # (e.g. "Parallel (actual)" vs "Sequential view" for GPT-NeoX).
    view_variants: list[dict | None] = (ir.get("extras") or {}).get("view_variants") or [None]

    radios: list[str] = []
    pills: list[str] = []
    arch_variants: list[str] = []
    l2_variants: list[str] = []
    nested_variants_by_depth: list[list[str]] = []
    variant_css: list[str] = []
    radio_name = f"{mount_id}-g"

    variant_idx = 0
    for i, group in enumerate(groups):
        for j, view_var in enumerate(view_variants):
            # --- Determine display spec for the arch view ---
            if view_var is not None:
                display_spec = {**group["spec"], "blocks": view_var["blocks"]}
                display_group = {**group, "spec": display_spec}
                # Pill label: use view_variant label; prefix with group label when
                # there are multiple layer types so each pill is unambiguous.
                if len(groups) > 1:
                    pill_label = f"{_group_label(group, info)} · {view_var['label']}"
                else:
                    pill_label = view_var["label"]
            else:
                display_group = group
                pill_label = _group_label(group, info)

            arch_blocks = _block_lookup(ir, display_group["spec"])
            arch_info = {
                "groups": groups,
                "dominant": display_group,
                "blocks": arch_blocks,
                "meta": _meta_for(ir, display_group["spec"], arch_blocks),
            }
            # L2/L3 inspect cards describe block semantics, not topology, so they
            # always use the original group spec regardless of the topology view.
            l2_blocks = _block_lookup(ir, group["spec"])
            l2_info = {
                "groups": groups,
                "dominant": group,
                "blocks": l2_blocks,
                "meta": _meta_for(ir, group["spec"], l2_blocks),
            }

            suffix = f"-g{variant_idx}"
            radio_id = f"{mount_id}-g{variant_idx}"
            checked = " checked" if (i == dominant_idx and j == 0) else ""

            radios.append(
                f'<input type="radio" name="{_attr(radio_name)}" '
                f'id="{_attr(radio_id)}" class="uf-group-radio"{checked}>'
            )
            pills.append(
                f'<label for="{_attr(radio_id)}" class="uf-group-pill">'
                f'{_html(pill_label)}</label>'
            )

            arch_svg = _build_architecture_view(ir, arch_info, mount_id + suffix)
            arch_variants.append(
                f'<div class="uf-arch-variant uf-arch-variant-{variant_idx}">{arch_svg}</div>'
            )
            l2_inner = _build_inspect_cards(ir, l2_info, mount_id + suffix)
            l2_variants.append(
                f'<div class="uf-l2-variant uf-l2-variant-{variant_idx}">{l2_inner}</div>'
            )
            nested_panels = _build_nested_inspect_panels(ir, l2_info, mount_id + suffix)
            for depth_idx, nested_inner in enumerate(nested_panels):
                while len(nested_variants_by_depth) <= depth_idx:
                    nested_variants_by_depth.append([])
                nested_variants_by_depth[depth_idx].append(
                    f'<div class="uf-nested-variant uf-nested-variant-{variant_idx}">{nested_inner}</div>'
                )

            # Per-variant visibility + active-pill styling
            variant_css.append(
                f"#{mount_id} #{radio_id}:checked ~ .uf-card .uf-arch-variant-{variant_idx},"
                f"#{mount_id} #{radio_id}:checked ~ .uf-card .uf-l2-variant-{variant_idx},"
                f"#{mount_id} #{radio_id}:checked ~ .uf-card .uf-nested-variant-{variant_idx} "
                f"{{ display:block; }}"
            )
            variant_css.append(
                f"#{mount_id} #{radio_id}:checked ~ .uf-card .uf-group-pill[for='{radio_id}'] "
                f"{{ background:{C['block']}; color:#FFFFFF; }}"
            )

            variant_idx += 1

    style = _style(mount_id) + "\n" + "\n".join(variant_css)

    toggle_html = (
        f'<div class="uf-group-toggle" role="tablist">{"".join(pills)}</div>'
        if variant_idx > 1 else ""
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
    inspect_panel = (
        f'<div class="uf-inspect uf-inspect-panel uf-panel-hint" data-depth="2">'
        f'{"".join(l2_variants)}</div>'
    )
    nested_inspect_panels = "".join(
        f'<div class="uf-nested-inspect uf-inspect-panel uf-panel-compact" '
        f'data-depth="{depth_idx + 3}">{"".join(variants)}</div>'
        for depth_idx, variants in enumerate(nested_variants_by_depth)
    )

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
    evidence_section = _code_evidence_section(ir)

    return f"""
<div id="{_attr(mount_id)}" class="uf-root">
<style>
{FONT_IMPORT if include_font_import else ""}
{style}
</style>
{''.join(radios)}
<div class="uf-card">
{_header(ir, info, mount_id)}
{_stats_banner(ir)}
{arch_section}
{inspect_panel}
{nested_inspect_panels}
{layer_map_section}
{evidence_section}
</div>
{_click_script(mount_id)}
</div>
"""


def render_document(ir: dict, mount_id: str) -> str:
    """Render a standalone HTML document."""
    title = f"{ir.get('name', 'model')} - unfolded"
    body = render_fragment(ir, mount_id, include_font_import=False)
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
