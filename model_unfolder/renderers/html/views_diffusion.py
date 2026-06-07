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

from .block_views import block_detail_svg
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
from .views import _build_architecture_view, _build_layer_map, _is_resolved_diffusion_block


# ---------------------------------------------------------------------------
# Fragment assembly — loop on top, the transformer hierarchy shifted one deeper
# ---------------------------------------------------------------------------

def render_diffusion_fragment(ir: dict, mount_id: str, include_font_import: bool) -> str:
    from .theme import FONT_IMPORT  # local: keep import surface with the others

    info = _make_info(ir)

    loop_svg = _build_loop_view(ir, info, mount_id)
    loop_cards = _build_loop_cards(ir, info, mount_id)          # panel[0]  (L2)
    dit_cards = _build_inspect_cards(ir, info, mount_id)        # panel[1]  (L3) — DiT blocks
    loop_child_cards = _build_loop_child_cards(ir, info, mount_id)
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
    nested_levels = [dit_cards + loop_child_cards] + dit_nested
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
        elif block.get("view"):
            # e.g. the VAE decoder — render its own drill-down view as the card.
            svg = block_detail_svg(ir, info, mount_id, block)
            cards.append(_rich_card(bid, title, desc, svg) if svg
                         else _simple_card(bid, title, desc))
        else:
            cards.append(_simple_card(bid, title, desc))
    return "".join(cards)


def _build_loop_child_cards(ir: dict, info: dict, mount_id: str) -> str:
    loop_blocks = ((ir.get("extras") or {}).get("render") or {}).get("loop_blocks") or []
    cards: list[str] = []
    seen: set[str] = set()
    for block in loop_blocks:
        for child in block.get("children") or []:
            cid = child.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            title = child.get("title") or child.get("label") or cid
            desc = child.get("description", "")
            svg = block_detail_svg(ir, info, mount_id, child)
            cards.append(_rich_card(cid, title, desc, svg) if svg else _simple_card(cid, title, desc))
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

    def resolved(bid: str) -> bool:
        # Approved diffusion stages render solid; anything else renders pale to
        # flag that its place isn't decided yet (block_schema.DIFFUSION_STAGES).
        return _is_resolved_diffusion_block(True, info, bid, blocks.get(bid))

    diffusion = (ir.get("extras") or {}).get("diffusion") or {}
    scheduler = diffusion.get("scheduler")

    w, h = 720, 700
    cx = 292          # main latent column
    sched_cx = 594    # scheduler column (to the right)

    arrow_id, shadow_id = _ids(mount_id, "loop")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(28, 22, w - 56, h - 44, C["bg_outer"]))

    # --- Recursion region behind the denoiser + scheduler ---
    loop_x, loop_y, loop_w, loop_h = 108, 190, 570, 206
    parts.append(_svg_tag("rect", {
        "x": loop_x, "y": loop_y, "width": loop_w, "height": loop_h,
        "rx": 20, "ry": 20, "fill": C["bg_inner"], "opacity": 0.55,
        "stroke": C["block"], "stroke-width": 1.1, "stroke-dasharray": "6 5",
    }))

    # --- Nodes ---
    image = _rect_block(parts, info, shadow_id, "image",
                        cx - 78, 40, 156, 44, label("image", "Image"), font_size=17,
                        resolved=resolved("image"))
    vae = _rect_block(parts, info, shadow_id, "vae_decode",
                      cx - 96, 116, 192, 46, label("vae_decode", "VAE decode"), font_size=16,
                      resolved=resolved("vae_decode"))

    # Denoiser — the hero. Two subtle offset cards imply "same network repeated"
    # without crowding the scheduler lane.
    den_x, den_y, den_w, den_h = cx - 148, 246, 296, 96
    for off in (9, 4):
        parts.append(_svg_tag("rect", {
            "x": den_x + off, "y": den_y - off, "width": den_w, "height": den_h,
            "rx": 13, "ry": 13, "fill": C["block"], "opacity": 0.4,
            "stroke": C["block_alt"], "stroke-width": 0.6,
        }))
    denoiser = _rect_block(parts, info, shadow_id, "denoiser",
                           den_x, den_y, den_w, den_h,
                           "DiT Denoiser", font_size=20, resolved=resolved("denoiser"))

    scheduler = _rect_block(parts, info, shadow_id, "scheduler",
                            sched_cx - 78, den_y + 6, 156, den_h - 12,
                            label("scheduler", ["Scheduler", "step"]), font_size=15,
                            resolved=resolved("scheduler"))

    timestep = _rect_block(parts, info, shadow_id, "timestep",
                           34, 456, 168, 52, label("timestep", "Timestep t"), font_size=15,
                           resolved=resolved("timestep"))

    noise = _rect_block(parts, info, shadow_id, "noise",
                        cx - 84, 572, 168, 58, label("noise", "Noise"), font_size=18,
                        resolved=resolved("noise"))
    _latent_grid(parts, noise["left"] - 88, noise["cy"] - 33)

    # Text conditioning: one block per real encoder (+ a shared prompt source),
    # drawn on the right and feeding the denoiser.
    _draw_text_conditioning(parts, info, shadow_id, arrow_id, label, blocks, denoiser)

    # --- Flow arrows: latent + timestep converge into the denoiser bottom ---
    parts.append(_v_line(noise, denoiser, arrow_id))
    parts.append(_clean_elbow_to_bottom(
        timestep["cx"], timestep["top"], denoiser["cx"] - 112, denoiser["bottom"] + GAP, arrow_id))

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
    parts.append(_v_line(vae, image, arrow_id))

    return _svg(w, h, f"{ir.get('name', 'model')} sampling loop", parts)


def _draw_text_conditioning(parts, info, shadow_id, arrow_id, label, blocks, denoiser):
    """Draw the prompt source + one block per text encoder, feeding the denoiser.

    Encoders are the loop blocks with ids ``encoder_0..N``; ``prompt`` is their
    shared source.  Falls back to a single ``text_encoder`` block when no encoders
    were detected (a bare transformer config without the pipeline index)."""
    def _resolved(bid):
        return _is_resolved_diffusion_block(True, info, bid, blocks.get(bid))

    enc_ids = sorted(bid for bid in blocks if bid.startswith("encoder_"))
    if not enc_ids:
        te = _rect_block(parts, info, shadow_id, "text_encoder",
                         326, 456, 224, 52,
                         label("text_encoder", ["Text prompt", "-> encoder"]), font_size=15,
                         resolved=_resolved("text_encoder"))
        parts.append(_clean_elbow_to_bottom(
            te["cx"], te["top"], denoiser["cx"] + 104, denoiser["bottom"] + GAP, arrow_id))
        return

    n = len(enc_ids)
    row_x0, row_x1, row_y = 300, 692, 456
    ew = min(142, int((row_x1 - row_x0 - (n - 1) * 18) / n))
    centers = [row_x0 + i * (ew + 18) + ew / 2 for i in range(n)]

    encoders = []
    for bid, ccx in zip(enc_ids, centers):
        encoders.append(_rect_block(parts, info, shadow_id, bid,
                                    ccx - ew / 2, row_y, ew, 52, label(bid, "Encoder"), font_size=15,
                                    resolved=_resolved(bid)))

    # Shared prompt source below the encoder row, centered under them.
    prompt_cx = sum(centers) / n
    prompt = _rect_block(parts, info, shadow_id, "prompt",
                         prompt_cx - 70, 572, 140, 48, label("prompt", "Text prompt"), font_size=15,
                         resolved=_resolved("prompt"))

    # Prompt fans up into each encoder through one bus with an explicit split dot.
    parts.append(_split_up_to_targets(
        prompt["cx"], prompt["top"], [e["cx"] for e in encoders], encoders[0]["bottom"] + GAP, arrow_id
    ))

    # Each encoder keeps its OWN arrow into the denoiser (they are independent).
    # To avoid crossings, the feed points are assigned left→right with each feed
    # placed just right of the previous encoder's column, so the [feed, source]
    # routes stay disjoint.
    feeds = _ordered_feeds([e["cx"] for e in encoders], denoiser["cx"] + 28, denoiser["right"] - 14)
    for enc, feed_x in zip(encoders, feeds):
        parts.append(_clean_elbow_to_bottom(
            enc["cx"], enc["top"], feed_x, denoiser["bottom"] + GAP, arrow_id))


def _h_arrow(x1: float, x2: float, y: float, arrow_id: str) -> str:
    return _svg_tag("line", {
        "x1": x1, "y1": y, "x2": x2, "y2": y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    })


def _clean_elbow_to_bottom(x1: float, y1: float, x2: float, y2: float, arrow_id: str) -> str:
    """Crisp 90-degree route into a block bottom, with the arrowhead vertical."""
    mid_y = min(y1 - 18, y2 + 38)
    d = (
        f"M {x1:.2f} {y1:.2f} "
        f"L {x1:.2f} {mid_y:.2f} "
        f"L {x2:.2f} {mid_y:.2f} "
        f"L {x2:.2f} {y2:.2f}"
    )
    return _svg_tag("path", {
        "d": d,
        "fill": "none",
        "stroke": C["arrow"],
        "stroke-width": 1.6,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})",
    })


def _split_up_to_targets(src_x: float, src_top: float, target_xs: list[float], target_y: float, arrow_id: str) -> str:
    if not target_xs:
        return ""
    if len(target_xs) == 1:
        return _clean_elbow_to_bottom(src_x, src_top, target_xs[0], target_y, arrow_id)

    split_y = min(src_top - 26, target_y + 34)
    x0, x1 = min(target_xs), max(target_xs)
    parts = [
        _svg_tag("line", {
            "x1": src_x, "y1": src_top, "x2": src_x, "y2": split_y,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round", "fill": "none",
        }),
        _svg_tag("line", {
            "x1": x0, "y1": split_y, "x2": x1, "y2": split_y,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round", "fill": "none",
        }),
        _junction_dot(src_x, split_y),
    ]
    for tx in target_xs:
        parts.append(_svg_tag("line", {
            "x1": tx, "y1": split_y, "x2": tx, "y2": target_y,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none",
        }))
    return "".join(parts)


def _ordered_feeds(sources: list[float], lo: float, hi: float) -> list[float]:
    """Feed x's (left→right) for individual encoder arrows into the denoiser bottom.

    Each feed is placed just right of the previous encoder's column so the
    per-encoder routes are disjoint and never cross: ``feed[i] > source[i-1]``.
    Clamped to ``[lo, hi]``."""
    feeds: list[float] = []
    prev_feed = lo - 56
    for i, _ in enumerate(sources):
        f = max(lo, prev_feed + 56)
        if i > 0:
            f = max(f, sources[i - 1] + 22)   # clear the previous encoder's column
        f = min(f, hi)
        feeds.append(f)
        prev_feed = f
    return feeds


def _junction_dot(x: float, y: float) -> str:
    return _svg_tag("circle", {"cx": x, "cy": y, "r": 3.2, "fill": C["arrow"]})


def _latent_grid(parts: list[str], x0: float, y0: float, n: int = 5, cell: int = 12) -> None:
    """A small 5x5 Gaussian-ish latent glyph."""
    vals = [
        [0.20, 0.35, 0.45, 0.34, 0.22],
        [0.34, 0.58, 0.72, 0.56, 0.36],
        [0.44, 0.75, 0.95, 0.78, 0.46],
        [0.32, 0.55, 0.74, 0.59, 0.38],
        [0.22, 0.36, 0.48, 0.37, 0.24],
    ]
    for r in range(n):
        for c in range(n):
            op = vals[r][c]
            parts.append(_svg_tag("rect", {
                "x": x0 + c * cell, "y": y0 + r * cell,
                "width": cell - 2, "height": cell - 2,
                "rx": 2.2, "ry": 2.2, "fill": C["block"], "opacity": op,
                "stroke": C["block"], "stroke-width": 0.15,
            }))
    parts.append(_svg_tag("rect", {
        "x": x0 - 2, "y": y0 - 2, "width": n * cell, "height": n * cell,
        "rx": 4, "ry": 4, "fill": "none", "stroke": C["border"], "stroke-width": 0.9,
    }))
