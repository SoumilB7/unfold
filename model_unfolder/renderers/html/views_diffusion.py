"""Diffusion rendering: the sampling loop is the hero; the DiT denoiser drills out of it.

A diffusion model isn't a feed-forward stack — it's an iterative sampler that
applies the *same* denoiser network for T steps, a scheduler nudging the latent
toward clean each step.  So the main image here is that recursion:

    Noise z_T -> [ Denoiser (DiT) <-> Scheduler ] x T -> VAE decode -> Image

The ``denoiser`` node opens the transformer architecture (the DiT stack) as its
drill-down — which is exactly the transformer rendering, one panel deeper.  This
module owns only the loop view + the panel wiring that shifts the transformer
hierarchy down by one; everything below the denoiser reuses the existing engine.
"""
from __future__ import annotations

from .cards import (
    _build_inspect_cards,
    _build_nested_inspect_panels,
    _hint_card,
    _rich_card,
    _simple_card,
)
from .evidence import _code_evidence_section
from .interactions import _click_script
from .metadata import _make_info
from .sections import _details_section, _header, _stats_banner
from .styles import _style
from .svg import (
    _block_top_to_block_bottom,
    _defs,
    _ids,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
)
from .theme import C, FONT_BODY, FONT_HEAD, FONT_MONO, GAP
from .utils import _attr, _html
from .views import _build_architecture_view, _build_layer_map


# ---------------------------------------------------------------------------
# Fragment assembly — loop on top, the transformer hierarchy shifted one deeper
# ---------------------------------------------------------------------------

def render_diffusion_fragment(ir: dict, mount_id: str, include_font_import: bool) -> str:
    from .theme import FONT_IMPORT  # local: keep import surface with the others

    info = _make_info(ir)

    loop_svg = _build_loop_view(ir, info, mount_id)
    loop_cards = _build_loop_cards(ir, info, mount_id)          # panel[0]  (L2)
    dit_cards = _build_inspect_cards(ir, info, mount_id)        # panel[1]  (L3) — DiT blocks
    dit_nested = _build_nested_inspect_panels(ir, info, mount_id)  # panel[2+] (L4+)

    style = _style(mount_id)

    arch_section = (
        '<details class="uf-section uf-section-arch uf-section-collapsible" open>'
        '<summary class="uf-section-head">'
        '<span class="uf-section-label">SAMPLING LOOP</span>'
        '<span class="uf-section-sub">Denoiser applied iteratively · click it to open its architecture</span>'
        '<span class="uf-chevron" aria-hidden="true">›</span>'
        '</summary>'
        f'<div class="uf-section-body">{loop_svg}</div>'
        '</details>'
    )

    inspect_panel = (
        '<div class="uf-inspect uf-inspect-panel uf-panel-hint" data-depth="2">'
        f'{loop_cards}</div>'
    )
    nested_levels = [dit_cards] + dit_nested
    nested_panels = "".join(
        f'<div class="uf-nested-inspect uf-inspect-panel uf-panel-compact" '
        f'data-depth="{i + 3}">{level}</div>'
        for i, level in enumerate(nested_levels)
    )

    # The layer map describes the DENOISER's own stack (now split into double- vs
    # single-stream groups), so label it as such.
    map_svg = _build_layer_map(ir, info, mount_id)
    n_layers = len(ir.get("layers", []))
    n_groups = len(info["groups"])
    if n_groups <= 1:
        map_sub = "Denoiser layers · all structurally identical"
    else:
        map_sub = f"Denoiser · {n_groups} block types across {n_layers} layers"
    layer_map_section = _details_section("DENOISER LAYER MAP", map_sub, map_svg)
    evidence_section = _code_evidence_section(ir)

    return f"""
<div id="{_attr(mount_id)}" class="uf-root">
<style>
{FONT_IMPORT if include_font_import else ""}
{style}
</style>
<div class="uf-card">
{_header(ir, info)}
{_stats_banner(ir)}
{arch_section}
{inspect_panel}
{nested_panels}
{layer_map_section}
{evidence_section}
</div>
{_click_script(mount_id)}
</div>
"""


def _build_loop_cards(ir: dict, info: dict, mount_id: str) -> str:
    """L2 cards for the loop nodes; the denoiser card embeds the DiT architecture."""
    loop_blocks = ((ir.get("extras") or {}).get("render") or {}).get("loop_blocks") or []
    cards = [_hint_card("default", "Click a block — open the Denoiser to see its architecture")]
    dit_svg = _build_architecture_view(ir, info, mount_id)
    for block in loop_blocks:
        bid = block.get("id")
        if not bid:
            continue
        title = block.get("title") or bid
        desc = block.get("description", "")
        if bid == "denoiser":
            cards.append(_rich_card(bid, title, desc, dit_svg))
        else:
            cards.append(_simple_card(bid, title, desc))
    return "".join(cards)


# ---------------------------------------------------------------------------
# The sampling-loop SVG (vertical: noise at bottom -> image at top)
# ---------------------------------------------------------------------------

def _build_loop_view(ir: dict, info: dict, mount_id: str) -> str:
    blocks = {
        b["id"]: b
        for b in (((ir.get("extras") or {}).get("render") or {}).get("loop_blocks") or [])
    }

    def label(bid: str, default: str):
        lab = (blocks.get(bid) or {}).get("label", default)
        if isinstance(lab, list):
            lab = [ln for ln in lab if ln]  # drop empty stacked lines
        return lab or default

    diffusion = (ir.get("extras") or {}).get("diffusion") or {}
    scheduler = diffusion.get("scheduler")

    w, h = 720, 660
    cx = 285          # main latent column
    sched_cx = 588    # scheduler column (to the right)

    arrow_id, shadow_id = _ids(mount_id, "loop")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(28, 22, w - 56, h - 44, C["bg_outer"]))

    # --- Recursion region behind the denoiser + scheduler ---
    loop_x, loop_y, loop_w, loop_h = 112, 196, 556, 182
    parts.append(_svg_tag("rect", {
        "x": loop_x, "y": loop_y, "width": loop_w, "height": loop_h,
        "rx": 20, "ry": 20, "fill": C["bg_inner"], "opacity": 0.55,
        "stroke": C["block"], "stroke-width": 1.1, "stroke-dasharray": "6 5",
    }))

    # --- Nodes ---
    image = _rect_block(parts, info, shadow_id, "image",
                        cx - 78, 40, 156, 44, label("image", "Image"), font_size=17)
    vae = _rect_block(parts, info, shadow_id, "vae_decode",
                      cx - 96, 116, 192, 46, label("vae_decode", "VAE decode"), font_size=16)

    # Denoiser — the hero. Faint offset cards behind imply an expandable stack.
    # Pushed down a touch so the badge band above it stays clear.
    den_x, den_y, den_w, den_h = cx - 152, 240, 304, 100
    for off in (12, 6):
        parts.append(_svg_tag("rect", {
            "x": den_x + off, "y": den_y - off, "width": den_w, "height": den_h,
            "rx": 13, "ry": 13, "fill": C["block"], "opacity": 0.4,
            "stroke": C["block_alt"], "stroke-width": 0.6,
        }))
    denoiser = _rect_block(parts, info, shadow_id, "denoiser",
                           den_x, den_y, den_w, den_h,
                           "DiT Denoiser", font_size=20)

    scheduler = _rect_block(parts, info, shadow_id, "scheduler",
                            sched_cx - 78, den_y + 6, 156, den_h - 12,
                            label("scheduler", ["Scheduler", "step"]), font_size=15)

    timestep = _rect_block(parts, info, shadow_id, "timestep",
                           150 - 86, 432, 172, 52, label("timestep", "Timestep t"), font_size=15)
    text_encoder = _rect_block(parts, info, shadow_id, "text_encoder",
                               438 - 112, 432, 224, 52,
                               label("text_encoder", ["Text prompt", "-> encoder"]), font_size=15)

    noise = _rect_block(parts, info, shadow_id, "noise",
                        cx - 90, 540, 180, 60, label("noise", ["Noise", "z_T"]), font_size=17)
    _latent_grid(parts, noise["left"] - 84, noise["cy"] - 32)

    # --- Flow arrows ---
    # Inputs converge into the denoiser bottom.
    parts.append(_v_line(noise, denoiser, arrow_id))
    parts.append(_block_top_to_block_bottom(
        timestep["cx"], timestep["top"], denoiser["cx"] - 96, denoiser["bottom"] + GAP, arrow_id))
    parts.append(_block_top_to_block_bottom(
        text_encoder["cx"], text_encoder["top"], denoiser["cx"] + 96, denoiser["bottom"] + GAP, arrow_id))

    # Denoiser <-> Scheduler cycle (the recursion).
    cycle_top_y = denoiser["cy"] - 17
    cycle_bot_y = denoiser["cy"] + 17
    parts.append(_h_arrow(denoiser["right"] + 2, scheduler["left"] - GAP, cycle_top_y, arrow_id))
    parts.append(_svg_text(
        (denoiser["right"] + scheduler["left"]) / 2, cycle_top_y - 12, "ε̂",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 12}))
    parts.append(_h_arrow(scheduler["left"] - 2, denoiser["right"] + GAP, cycle_bot_y, arrow_id))
    parts.append(_svg_text(
        (denoiser["right"] + scheduler["left"]) / 2, cycle_bot_y + 13, "z₋₁",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11}))

    # Out of the loop: clean latent -> VAE -> image.
    parts.append(_v_line(denoiser, vae, arrow_id))
    parts.append(_svg_text(
        denoiser["cx"] + 14, (denoiser["top"] + vae["bottom"]) / 2, "z₀ (clean)",
        {"dominant-baseline": "central", "fill": C["muted"],
         "font-family": FONT_MONO, "font-size": 11}))
    parts.append(_v_line(vae, image, arrow_id))

    return _svg(w, h, f"{ir.get('name', 'model')} sampling loop", parts)


def _h_arrow(x1: float, x2: float, y: float, arrow_id: str) -> str:
    return _svg_tag("line", {
        "x1": x1, "y1": y, "x2": x2, "y2": y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    })


def _latent_grid(parts: list[str], x0: float, y0: float, n: int = 4, cell: int = 15) -> None:
    """A small NxN checker glyph suggesting the noisy latent tensor."""
    for r in range(n):
        for c in range(n):
            # Alternating opacity gives a noisy speckle without inventing values.
            op = 0.85 if (r + c) % 2 == 0 else 0.4
            parts.append(_svg_tag("rect", {
                "x": x0 + c * cell, "y": y0 + r * cell,
                "width": cell - 1.5, "height": cell - 1.5,
                "rx": 2, "ry": 2, "fill": C["block"], "opacity": op,
            }))
    parts.append(_svg_tag("rect", {
        "x": x0 - 1.5, "y": y0 - 1.5, "width": n * cell + 1, "height": n * cell + 1,
        "rx": 3, "ry": 3, "fill": "none", "stroke": C["border"], "stroke-width": 0.8,
    }))
