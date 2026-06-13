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
    _unique_children,
)
from .evidence import _code_evidence_section
from .graph_engine import _badge
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

    # A UNet denoiser has no flat transformer-layer stack — its architecture is
    # the U-shape drawn in the denoiser card, so the DiT layer-map / per-layer
    # card machinery is skipped.
    is_unet = bool((ir.get("extras") or {}).get("unet"))
    info = _make_info(ir) if ir.get("layers") else _stub_info()

    loop_svg = _build_loop_view(ir, info, mount_id)
    loop_cards = _build_loop_cards(ir, info, mount_id)          # panel[0]  (L2)
    # Descendant levels below the loop blocks: [0] = VAE decoder stages, [1] =
    # the ResNet ops inside a stage, ... — one nested panel per level.
    loop_levels = _build_loop_descendant_levels(ir, info, mount_id)

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

    if is_unet:
        # UNet: the denoiser view is a leaf; only the VAE-decoder descendants need
        # deeper panels.  No DiT cards, no per-layer map.
        nested_levels = list(loop_levels)
        layer_map_section = ""
    else:
        dit_cards = _build_inspect_cards(ir, info, mount_id)        # panel[1] (L3)
        dit_nested = _build_nested_inspect_panels(ir, info, mount_id)  # panel[2+] (L4+)
        # Merge level-wise: the DiT layer drill-downs and the loop's VAE
        # descendants share panels at matching depths.
        nested_levels = _merge_levels([dit_cards] + dit_nested, loop_levels)
        map_svg = _build_layer_map(ir, info, mount_id)
        n_layers = len(ir.get("layers", []))
        n_groups = len(info["groups"])
        map_sub = ("Denoiser layers · all structurally identical" if n_groups <= 1
                   else f"Denoiser · {n_groups} block types across {n_layers} layers")
        layer_map_section = _details_section("DENOISER LAYER MAP", map_sub, map_svg)

    nested_panels = "".join(
        f'<div class="uf-nested-inspect uf-inspect-panel uf-panel-compact" '
        f'data-depth="{i + 3}">{level}</div>'
        for i, level in enumerate(nested_levels)
    )
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


def _stub_info() -> dict:
    """Minimal info for diffusion models with no flat layer stack (UNet)."""
    return {"groups": [], "blocks": {}, "meta": {}, "dominant": None, "layer_sigs": []}


def _build_loop_cards(ir: dict, info: dict, mount_id: str) -> str:
    """L2 cards for the loop nodes; the denoiser card embeds its architecture —
    the DiT stack, or the UNet U-shape (per ``render.denoiser_view``)."""
    render = (ir.get("extras") or {}).get("render") or {}
    loop_blocks = render.get("loop_blocks") or []
    denoiser_view = render.get("denoiser_view", "dit")
    cards = [_hint_card("default", "Click a block — open the Denoiser to see its architecture")]
    for block in loop_blocks:
        bid = block.get("id")
        if not bid:
            continue
        title = block.get("title") or bid
        desc = block.get("description", "")
        facts = block.get("facts")
        if bid == "denoiser":
            if denoiser_view == "unet":
                svg = block_detail_svg(ir, info, mount_id, {"id": "denoiser", "view": "unet"})
            else:
                svg = _build_architecture_view(ir, info, mount_id)
            cards.append(_rich_card(bid, title, desc, svg, facts) if svg
                         else _simple_card(bid, title, desc, facts))
        elif block.get("view"):
            # e.g. the VAE decoder — render its own drill-down view as the card.
            svg = block_detail_svg(ir, info, mount_id, block)
            cards.append(_rich_card(bid, title, desc, svg, facts) if svg
                         else _simple_card(bid, title, desc, facts))
        else:
            cards.append(_simple_card(bid, title, desc, facts))
    return "".join(cards)


def _build_loop_descendant_levels(ir: dict, info: dict, mount_id: str) -> list[str]:
    """One cards-string per depth below the loop blocks.

    Level 0 is the loop blocks' direct children (e.g. the VAE decoder's up
    stages); level 1 is *their* children (e.g. the ResNet ops inside a stage);
    and so on.  Each level becomes one nested inspect panel, so a node clicked in
    level *k*'s card opens level *k+1*."""
    loop_blocks = ((ir.get("extras") or {}).get("render") or {}).get("loop_blocks") or []
    current: list[dict] = []
    for block in loop_blocks:
        current.extend(block.get("children") or [])

    levels: list[str] = []
    while current:
        current = _unique_children(current)
        levels.append(_cards_for_children(ir, info, mount_id, current))
        nxt: list[dict] = []
        for child in current:
            nxt.extend(child.get("children") or [])
        current = nxt
    return levels


def _cards_for_children(ir: dict, info: dict, mount_id: str, children: list[dict]) -> str:
    cards: list[str] = []
    for child in children:
        cid = child.get("id")
        if not cid:
            continue
        title = child.get("title") or child.get("label") or cid
        desc = child.get("description", "")
        facts = child.get("facts")
        svg = block_detail_svg(ir, info, mount_id, child)
        cards.append(_rich_card(cid, title, desc, svg, facts) if svg
                     else _simple_card(cid, title, desc, facts))
    return "".join(cards)


def _merge_levels(a: list[str], b: list[str]) -> list[str]:
    """Concatenate two lists of per-level cards-strings level-by-level, so cards
    that belong at the same drill depth land in the same panel."""
    return [
        (a[i] if i < len(a) else "") + (b[i] if i < len(b) else "")
        for i in range(max(len(a), len(b)))
    ]


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

    # ------------------------------------------------------------------
    # Layout: ONE latent spine (Noise -> junction -> Denoiser -> VAE ->
    # Image), the recursion drawn as a literal circuit (Denoiser -ε̂->
    # Scheduler -> return rail -z_t-1-> junction -> Denoiser), and all
    # conditioning entering the denoiser's LEFT edge from a source stack —
    # latent flows vertically, conditioning enters laterally.
    # ------------------------------------------------------------------
    w, h = 760, 640
    cx = 380                            # the latent spine column
    sched_cx = 615                      # scheduler column, right of the denoiser

    arrow_id, shadow_id = _ids(mount_id, "loop")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(28, 22, w - 56, h - 44, C["bg_outer"]))

    den_x, den_y, den_w, den_h = cx - 130, 246, 260, 96
    # The latent lives in an explicit state cell on the denoiser's input. It has
    # two writers at *different times* — Noise seeds it once, the scheduler
    # overwrites it every step — and one reader, the denoiser. Drawing it as a
    # named slot (not a bare vertex where lines meet) is what stops "two arrows
    # in" from reading as a sum: a register with two writers and one reader is
    # unambiguous; a junction is not.
    buf_w, buf_h = 116, 40
    buf_x, buf_y = cx - buf_w / 2, den_y + den_h + 24
    buf_cy, buf_bottom = buf_y + buf_h / 2, buf_y + buf_h
    rail_y = buf_cy                     # the z_{t-1} return rail meets the cell

    # --- The loop frame: the SAME solid cell frame + white repeat pill the
    # engine draws for "× N layers" — one visual language for "this part runs
    # repeatedly".  The step count is a runtime choice (never in the config),
    # so the pill states the loop's honest terminating fact: t reaches 0.
    loop_x, loop_y = den_x - 36, 190
    loop_w, loop_h = (sched_cx + 75 + 26) - loop_x, (buf_bottom + 14) - loop_y
    parts.append(_svg_tag("rect", {
        "x": loop_x, "y": loop_y, "width": loop_w, "height": loop_h,
        "rx": 18, "ry": 18, "fill": C["bg_inner"], "stroke": "none",
    }))
    _badge(parts, loop_x + loop_w, loop_y + 12, "↺ t → 0")

    # --- Spine nodes (bottom -> top: Noise, Denoiser, VAE, Image) ---
    image = _rect_block(parts, info, shadow_id, "image",
                        cx - 78, 40, 156, 44, label("image", "Image"), font_size=17,
                        resolved=resolved("image"))
    vae = _rect_block(parts, info, shadow_id, "vae_decode",
                      cx - 96, 116, 192, 46, label("vae_decode", "VAE decode"), font_size=16,
                      resolved=resolved("vae_decode"))
    denoiser = _rect_block(parts, info, shadow_id, "denoiser",
                           den_x, den_y, den_w, den_h,
                           "DiT Denoiser", font_size=20, resolved=resolved("denoiser"))
    scheduler = _rect_block(parts, info, shadow_id, "scheduler",
                            sched_cx - 75, den_y + 6, 150, den_h - 12,
                            label("scheduler", ["Scheduler", "step"]), font_size=15,
                            resolved=resolved("scheduler"))
    noise = _rect_block(parts, info, shadow_id, "noise",
                        cx - 84, 488, 168, 58, label("noise", "Noise"), font_size=18,
                        resolved=resolved("noise"))
    _latent_grid(parts, noise["left"] - 88, noise["cy"] - 33)
    # The latent state cell — the slot every step reads from and writes back to.
    # Non-clickable: it's a wire/register, not a drillable compute step.
    latent = _rect_block(parts, info, shadow_id, "latent",
                         buf_x, buf_y, buf_w, buf_h, "latent", font_size=15,
                         resolved=True, clickable=False)

    # --- Conditioning stack on the left (text encoders + timestep), each
    # entering the denoiser's left edge — every step receives them. ---
    _draw_conditioning_stack(parts, info, shadow_id, arrow_id, label, blocks, denoiser)

    # --- The latent circuit: the cell is written twice (seeded once, then
    # overwritten each step) and read once per step by the denoiser. ---
    mono = {"fill": C["muted"], "font-family": FONT_MONO, "font-size": 12}
    # Seed write: Noise rises into the cell's BOTTOM edge — once, from outside
    # the loop frame (so the seed visibly enters the recursion a single time).
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": noise["top"], "x2": cx, "y2": latent["bottom"] + 4,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none"}))
    parts.append(_svg_text(cx + 12, (noise["top"] + latent["bottom"]) / 2, "z_T · once",
                           {**mono, "text-anchor": "start"}))
    # ε̂ forward: denoiser -> scheduler.
    eps_y = den_y + 26
    parts.append(_h_arrow(denoiser["right"] + 2, scheduler["left"] - 8, eps_y, arrow_id))
    parts.append(_svg_text((denoiser["right"] + scheduler["left"]) / 2, eps_y - 12, "ε̂",
                           {**mono, "text-anchor": "middle"}))
    # Each-step write: scheduler output returns and overwrites the cell's RIGHT
    # edge — a different writer, a different edge, a different time than the seed.
    parts.append(_svg_tag("path", {
        "d": (f"M {sched_cx} {scheduler['bottom']} L {sched_cx} {rail_y} "
              f"L {latent['right'] + 4} {rail_y}"),
        "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
        "stroke-linecap": "round", "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})"}))
    parts.append(_svg_text((latent["right"] + sched_cx) / 2, rail_y - 9, "z_t-1 · each step",
                           {**mono, "text-anchor": "middle", "font-size": 11}))
    # Read: the denoiser reads the current latent off the cell's TOP edge.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": latent["top"], "x2": cx, "y2": denoiser["bottom"] + 4,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none"}))
    parts.append(_svg_text(cx + 12, (latent["top"] + denoiser["bottom"]) / 2, "z_t",
                           {**mono, "text-anchor": "start", "font-size": 11}))

    # --- Out of the loop, once: clean latent -> VAE -> image. ---
    parts.append(_v_line(denoiser, vae, arrow_id))
    parts.append(_svg_text(cx + 12, (loop_y + vae["bottom"]) / 2 + 4, "z_0",
                           {**mono, "text-anchor": "start", "font-size": 11}))
    parts.append(_v_line(vae, image, arrow_id))

    return _svg(w, h, f"{ir.get('name', 'model')} sampling loop", parts)


def _draw_conditioning_stack(parts, info, shadow_id, arrow_id, label, blocks, den):
    """The conditioning sources as ONE left-hand stack — the timestep on top,
    text encoder(s) beneath it, the prompt source directly under its
    encoder(s).  The stack is vertically centred on the denoiser so each
    source lines up with its entry point and the arrow runs STRAIGHT into the
    denoiser's left edge; only oversized stacks fall back to nested elbows.

    Latent flows vertically through the loop; conditioning enters laterally —
    the two axes keep their meanings."""
    def _resolved(bid):
        return _is_resolved_diffusion_block(True, info, bid, blocks.get(bid))

    enc_ids = sorted(bid for bid in blocks if bid.startswith("encoder_"))
    entries: list[tuple[str, object, bool]] = [
        ("timestep", label("timestep", "Timestep t"), _resolved("timestep"))]
    if enc_ids:
        entries += [(bid, label(bid, "Encoder"), _resolved(bid)) for bid in enc_ids]
    else:
        entries += [("text_encoder", label("text_encoder", ["Text prompt", "→ encoder"]),
                     _resolved("text_encoder"))]

    # Stack centred on the denoiser; sources whose centre fits the denoiser's
    # left edge get a straight arrow, the rest a nested elbow.
    col_cx, bw, bh, gap = 130, 150, 44, 14
    n = len(entries)
    stack_h = n * bh + (n - 1) * gap
    stack_top = max(den["cy"] - stack_h / 2, 206)
    srcs = []
    for i, (bid, lab, res) in enumerate(entries):
        y = stack_top + i * (bh + gap)
        srcs.append(_rect_block(parts, info, shadow_id, bid,
                                col_cx - bw / 2, y, bw, bh, lab, font_size=14,
                                resolved=res))

    lo, hi = den["top"] + 18, den["bottom"] - 18
    entry_ys = [min(max(src["cy"], lo), hi) for src in srcs]
    for i, (src, ey) in enumerate(zip(srcs, entry_ys)):
        if abs(ey - src["cy"]) < 0.5:
            parts.append(_h_arrow(src["right"] + 2, den["left"] - 4, ey, arrow_id))
        else:
            lane = den["left"] - 30 + i * 7
            parts.append(_svg_tag("path", {
                "d": (f"M {src['right']} {src['cy']} L {lane} {src['cy']} "
                      f"L {lane} {ey} L {den['left'] - 4} {ey}"),
                "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
                "stroke-linecap": "round", "stroke-linejoin": "round",
                "marker-end": f"url(#{arrow_id})"}))

    # Prompt source directly under its encoder(s) — nothing sits between them.
    if not enc_ids:
        return
    enc_srcs = srcs[1:]
    prompt_y = stack_top + n * (bh + gap) + 10
    prompt = _rect_block(parts, info, shadow_id, "prompt",
                         col_cx - 68, prompt_y, 136, 46, label("prompt", "Text prompt"),
                         font_size=14, resolved=_resolved("prompt"))
    if len(enc_srcs) == 1:
        parts.append(_svg_tag("line", {
            "x1": prompt["cx"], "y1": prompt["top"], "x2": prompt["cx"],
            "y2": enc_srcs[-1]["bottom"] + 4,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none"}))
        return
    bus_x = col_cx - bw / 2 - 24
    split_y = prompt["top"] - 12
    parts.append(_svg_tag("path", {
        "d": (f"M {prompt['cx']} {prompt['top']} L {prompt['cx']} {split_y} "
              f"L {bus_x} {split_y} L {bus_x} {enc_srcs[0]['cy']}"),
        "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
        "stroke-linecap": "round", "stroke-linejoin": "round"}))
    parts.append(_junction_dot(prompt["cx"], split_y))
    for enc in enc_srcs:
        parts.append(_svg_tag("line", {
            "x1": bus_x, "y1": enc["cy"], "x2": enc["left"] - 4, "y2": enc["cy"],
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none"}))


def _h_arrow(x1: float, x2: float, y: float, arrow_id: str) -> str:
    return _svg_tag("line", {
        "x1": x1, "y1": y, "x2": x2, "y2": y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    })


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
