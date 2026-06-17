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
    _elbow_hv,
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
    render = (ir.get("extras") or {}).get("render") or {}
    blocks = {b["id"]: b for b in (render.get("loop_blocks") or [])}
    loop_edges = render.get("loop_edges") or []

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

    # --- Nodes: positions are hand-authored (the latent spine vertical, the
    # scheduler to its right, conditioning on the left); their boxes join `pos`
    # so the declared edges can be wired to real geometry. ---
    pos: dict[str, dict] = {}
    pos["image"] = _rect_block(parts, info, shadow_id, "image",
                               cx - 78, 40, 156, 44, label("image", "Image"), font_size=17,
                               resolved=resolved("image"))
    pos["vae_decode"] = _rect_block(parts, info, shadow_id, "vae_decode",
                                    cx - 96, 116, 192, 46, label("vae_decode", "VAE decode"),
                                    font_size=16, resolved=resolved("vae_decode"))
    pos["denoiser"] = _rect_block(parts, info, shadow_id, "denoiser",
                                  den_x, den_y, den_w, den_h,
                                  label("denoiser", "DiT Denoiser"), font_size=20,
                                  resolved=resolved("denoiser"))
    pos["scheduler"] = _rect_block(parts, info, shadow_id, "scheduler",
                                   sched_cx - 75, den_y + 6, 150, den_h - 12,
                                   label("scheduler", ["Scheduler", "step"]), font_size=15,
                                   resolved=resolved("scheduler"))
    pos["noise"] = _rect_block(parts, info, shadow_id, "noise",
                               cx - 84, 488, 168, 58, label("noise", "Noise"), font_size=18,
                               resolved=resolved("noise"))
    _latent_grid(parts, pos["noise"]["left"] - 88, pos["noise"]["cy"] - 33)
    # The latent state cell — the slot every step reads from and writes back to.
    pos["latent"] = _rect_block(parts, info, shadow_id, "latent",
                                buf_x, buf_y, buf_w, buf_h, label("latent", "latent"),
                                font_size=15, resolved=True)
    _place_conditioning(parts, info, shadow_id, label, resolved, blocks, pos)

    # --- Arrows: every connection is drawn from the DECLARED loop edges — the
    # same list the JSON `sampling_loop` projects — so the diagram and the data
    # cannot drift.  Routing is per-edge; the topology lives in the declaration.
    _draw_loop_edges(parts, loop_edges, pos, cx, loop_y, arrow_id)

    return _svg(w, h, f"{ir.get('name', 'model')} sampling loop", parts)


def _place_conditioning(parts, info, shadow_id, label, resolved, blocks, pos):
    """Position the conditioning sources — timestep on top, text encoder(s)
    beneath it, the prompt directly under them — and add their boxes to ``pos``.

    Draws ONLY the node rects.  Their arrows (timestep/encoder → denoiser, and
    the prompt's fan-out to the encoders) are declared loop edges drawn by
    :func:`_draw_loop_edges`, so the wiring has a single author.  The stack is
    centred on the denoiser so aligned sources get a straight arrow; latent
    flows vertically, conditioning enters laterally — the two axes keep meaning."""
    enc_ids = sorted(bid for bid in blocks if bid.startswith("encoder_"))
    entries: list[tuple[str, object]] = [("timestep", label("timestep", "Timestep t"))]
    if enc_ids:
        entries += [(bid, label(bid, "Encoder")) for bid in enc_ids]
    else:
        entries += [("text_encoder", label("text_encoder", ["Text prompt", "→ encoder"]))]

    den = pos["denoiser"]
    col_cx, bw, bh, gap = 130, 150, 44, 14
    n = len(entries)
    stack_h = n * bh + (n - 1) * gap
    stack_top = max(den["cy"] - stack_h / 2, 206)
    for i, (bid, lab) in enumerate(entries):
        y = stack_top + i * (bh + gap)
        pos[bid] = _rect_block(parts, info, shadow_id, bid,
                               col_cx - bw / 2, y, bw, bh, lab, font_size=14,
                               resolved=resolved(bid))
    if not enc_ids:
        return
    prompt_y = stack_top + n * (bh + gap) + 10
    pos["prompt"] = _rect_block(parts, info, shadow_id, "prompt",
                                col_cx - 68, prompt_y, 136, 46, label("prompt", "Text prompt"),
                                font_size=14, resolved=resolved("prompt"))


def _draw_loop_edges(parts, loop_edges, pos, cx, loop_y, arrow_id):
    """Draw each declared loop edge with its per-edge routing.  This is the ONLY
    place loop arrows are drawn — the edge list (authored in the diffusor
    adapter, also projected to JSON) is the single source of the topology."""
    mono = {"fill": C["muted"], "font-family": FONT_MONO, "font-size": 12}
    prompt_targets: list[dict] = []
    for e in loop_edges:
        frm, to = pos.get(e.get("from")), pos.get(e.get("to"))
        if frm is None or to is None:
            continue
        route = e.get("route")
        if route == "spine":
            _edge_spine(parts, e, frm, to, cx, loop_y, arrow_id, mono)
        elif route == "eps":
            _edge_eps(parts, frm, to, arrow_id, mono)
        elif route == "rail":
            _edge_rail(parts, e, frm, to, arrow_id, mono)
        elif route == "lateral":
            _edge_lateral(parts, e, frm, to, arrow_id)
        elif route == "prompt":
            prompt_targets.append(to)
    if prompt_targets:
        _draw_prompt_fanout(parts, pos["prompt"], prompt_targets, arrow_id)


def _edge_spine(parts, e, frm, to, cx, loop_y, arrow_id, mono):
    """A vertical latent-column edge: the source always sits below the target,
    so it rises from the source's top into the target's bottom edge."""
    gap = e.get("gap", 6)
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": frm["top"], "x2": cx, "y2": to["bottom"] + gap,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none"}))
    lab = e.get("label")
    if lab:
        # z_0 sits up near the loop-frame top, not at the segment midpoint.
        ly = ((loop_y + to["bottom"]) / 2 + 4 if e.get("label_at") == "frame_top"
              else (frm["top"] + to["bottom"]) / 2)
        parts.append(_svg_text(cx + 12, ly, lab,
                               {**mono, "text-anchor": "start",
                                "font-size": e.get("label_size", 12)}))


def _edge_eps(parts, frm, to, arrow_id, mono):
    """Denoiser → scheduler: the predicted noise/velocity ε̂, an inset horizontal."""
    eps_y = frm["top"] + 26
    parts.append(_h_arrow(frm["right"] + 2, to["left"] - 8, eps_y, arrow_id))
    parts.append(_svg_text((frm["right"] + to["left"]) / 2, eps_y - 12, "ε̂",
                           {**mono, "text-anchor": "middle"}))


def _edge_rail(parts, e, frm, to, arrow_id, mono):
    """The loop-carried back-edge: scheduler output drops down and re-enters the
    latent cell's right edge (z_{t-1} becoming the next step's z_t)."""
    ry = to["cy"]
    parts.append(_svg_tag("path", {
        "d": (f"M {frm['cx']} {frm['bottom']} L {frm['cx']} {ry} "
              f"L {to['right'] + 4} {ry}"),
        "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
        "stroke-linecap": "round", "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})"}))
    lab = e.get("label")
    if lab:
        parts.append(_svg_text((to["right"] + frm["cx"]) / 2, ry - 9, lab,
                               {**mono, "text-anchor": "middle", "font-size": 11}))


def _edge_lateral(parts, e, frm, to, arrow_id):
    """A conditioning source entering the denoiser's left edge: straight when the
    source's centre lines up, else a nested elbow on its own lane."""
    lo, hi = to["top"] + 18, to["bottom"] - 18
    ey = min(max(frm["cy"], lo), hi)
    if abs(ey - frm["cy"]) < 0.5:
        parts.append(_h_arrow(frm["right"] + 2, to["left"] - 4, ey, arrow_id))
    else:
        lane = to["left"] - 30 + e.get("lane_index", 0) * 7
        parts.append(_svg_tag("path", {
            "d": (f"M {frm['right']} {frm['cy']} L {lane} {frm['cy']} "
                  f"L {lane} {ey} L {to['left'] - 4} {ey}"),
            "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
            "stroke-linecap": "round", "stroke-linejoin": "round",
            "marker-end": f"url(#{arrow_id})"}))


def _draw_prompt_fanout(parts, prompt, encs, arrow_id):
    """The prompt → encoder(s) splitter: a straight feed to a single encoder,
    or a shared bus fanning out to several."""
    if len(encs) == 1:
        enc = encs[0]
        parts.append(_svg_tag("line", {
            "x1": prompt["cx"], "y1": prompt["top"], "x2": prompt["cx"],
            "y2": enc["bottom"] + 4,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none"}))
        return
    bus_x = encs[0]["left"] - 24
    split_y = prompt["top"] - 12
    parts.append(_svg_tag("path", {
        "d": (f"M {prompt['cx']} {prompt['top']} L {prompt['cx']} {split_y} "
              f"L {bus_x} {split_y} L {bus_x} {encs[0]['cy']}"),
        "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
        "stroke-linecap": "round", "stroke-linejoin": "round"}))
    parts.append(_junction_dot(prompt["cx"], split_y))
    for enc in encs:
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


# ---------------------------------------------------------------------------
# Block diffusion fragment — DiffusionGemma generation loop
#
# Architecture: encoder (causal, one pass per canvas) → KV cache →
#   denoising loop (bidirectional decoder × ≤48 steps, with entropy-bound
#   accept/renoise and self-conditioning from prev step's logits).
# ---------------------------------------------------------------------------

def render_block_diffusion_fragment(ir: dict, mount_id: str, include_font_import: bool) -> str:
    from .theme import FONT_IMPORT
    info = _make_info(ir) if ir.get("layers") else _stub_info()

    loop_svg = _build_block_diffusion_view(ir, info, mount_id)
    loop_cards = _build_block_diffusion_loop_cards(ir, info, mount_id)

    dit_cards = _build_inspect_cards(ir, info, mount_id)
    dit_nested = _build_nested_inspect_panels(ir, info, mount_id)
    loop_levels = _build_loop_descendant_levels(ir, info, mount_id)
    nested_levels = _merge_levels([dit_cards] + dit_nested, loop_levels)

    style = _style(mount_id)

    arch_section = (
        '<details class="uf-section uf-section-arch uf-section-collapsible" open>'
        '<summary class="uf-section-head">'
        '<span class="uf-section-label">BLOCK DIFFUSION LOOP</span>'
        '<span class="uf-section-sub">'
        'Encoder fills KV cache · Decoder denoises canvas bidirectionally · '
        'open Encoder or Decoder to see the shared transformer stack'
        '</span>'
        '<span class="uf-chevron" aria-hidden="true">›</span>'
        '</summary>'
        f'<div class="uf-section-body">{loop_svg}</div>'
        '</details>'
    )
    inspect_panel = (
        '<div class="uf-inspect uf-inspect-panel uf-panel-hint" data-depth="2">'
        f'{loop_cards}</div>'
    )
    nested_panels = "".join(
        f'<div class="uf-nested-inspect uf-inspect-panel uf-panel-compact" '
        f'data-depth="{i + 3}">{level}</div>'
        for i, level in enumerate(nested_levels)
    )

    map_svg = _build_layer_map(ir, info, mount_id)
    n_layers = len(ir.get("layers", []))
    n_groups = len(info["groups"])
    map_sub = (
        f"Shared encoder/decoder stack · {n_groups} layer types across {n_layers} layers"
        if n_groups > 1
        else f"Shared encoder/decoder stack · {n_layers} layers, all identical"
    )
    layer_map_section = _details_section("SHARED ENCODER/DECODER LAYER MAP", map_sub, map_svg)
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


def _build_block_diffusion_loop_cards(ir: dict, info: dict, mount_id: str) -> str:
    """L2 cards for the block diffusion loop nodes.

    Both the encoder and decoder cards embed the shared transformer architecture
    view, since those two blocks run the same weights and the same per-layer
    structure — clicking either lets the user drill into the 30-layer stack.
    """
    render = (ir.get("extras") or {}).get("render") or {}
    loop_blocks = render.get("loop_blocks") or []
    arch_embed_ids = {"bd_encoder", "bd_decoder"}

    cards = [_hint_card(
        "default",
        "Click a block · open Encoder or Decoder to explore the shared transformer stack",
    )]
    for block in loop_blocks:
        bid = block.get("id")
        if not bid:
            continue
        title = block.get("title") or bid
        desc = block.get("description", "")
        facts = block.get("facts")
        if bid in arch_embed_ids:
            svg = _build_architecture_view(ir, info, mount_id)
            cards.append(
                _rich_card(bid, title, desc, svg, facts) if svg
                else _simple_card(bid, title, desc, facts)
            )
        elif block.get("view"):
            svg = block_detail_svg(ir, info, mount_id, block)
            cards.append(
                _rich_card(bid, title, desc, svg, facts) if svg
                else _simple_card(bid, title, desc, facts)
            )
        else:
            cards.append(_simple_card(bid, title, desc, facts))
    return "".join(cards)


def _kv_store_glyph(parts, info, shadow_id, node_id, x, y, w, h, label, font_size=14):
    """A KV cache drawn as a STORE, not a compute block.

    Two offset cards behind the front rect read as a stack of stored per-layer
    entries — that distinct shape (plus its place off the denoising chain) is why
    it doesn't look like a pipeline stage.  It's storage shared between the
    encoder (writer, once) and the decoder (reader, every step); the directional
    write arrow in and read arrow out carry that asymmetry.
    """
    for off in (10, 5):
        parts.append(_svg_tag("rect", {
            "x": x + off, "y": y - off, "width": w, "height": h,
            "rx": 10, "ry": 10, "fill": C["block"], "opacity": 0.42,
            "stroke": C["block_alt"], "stroke-width": 0.6,
        }))
    return _rect_block(parts, info, shadow_id, node_id, x, y, w, h, label, font_size=font_size)


def _build_block_diffusion_view(ir: dict, info: dict, mount_id: str) -> str:
    """Generation-loop SVG for DiffusionGemma.

    What the arrows explain — two processes sharing one store:
      * SETUP (once, outside the loop): Prompt → Encoder → writes the KV store.
      * LOOP  (≤48 steps): Canvas → Self-cond → Decoder → LM head → Sampler,
        with the Decoder READING the KV store each step, and the Sampler feeding
        its result back to the Canvas (renoise) and Self-cond (prev logits).

    The KV cache is a store glyph (stack + write/read ports), not a pipeline
    block: the encoder writes it once, the decoder reads it every step — that
    write-once / read-each-step asymmetry is the whole point.  The denoising
    chain is laid out with a UNIFORM gap so every flow arrow is the same length.
    No arrow carries a text label; the topology (and the cards) carry the meaning.
    """
    n_layers = len(ir.get("layers", []))
    bd = ((ir.get("extras") or {}).get("block_diffusion")) or {}
    canvas_len = bd.get("canvas_length", 256)

    w, h = 760, 620
    enc_cx = 152    # encoder column centre-x (left, outside the loop)
    den_cx = 478    # denoising chain centre-x (inside the loop)
    gap = 36        # the ONE vertical gap → every chain arrow is identical

    arrow_id, shadow_id = _ids(mount_id, "bdloop")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(28, 22, w - 56, h - 44, C["bg_outer"]))

    # ── Loop frame ────────────────────────────────────────────────────
    loop_x, loop_y, loop_w, loop_h = 300, 96, 360, 492
    parts.append(_svg_tag("rect", {
        "x": loop_x, "y": loop_y, "width": loop_w, "height": loop_h,
        "rx": 18, "ry": 18, "fill": C["bg_inner"], "stroke": "none",
    }))
    _badge(parts, loop_x + loop_w, loop_y + 14, "↺ up to 48 steps")

    pos: dict[str, dict] = {}

    # ── Denoising chain: stacked bottom→top with a uniform gap, so the five
    # flow arrows between them are all exactly `gap` long. ──
    chain = [
        ("bd_canvas", 176, 52, [f"Canvas · {canvas_len} tokens", "init U(V)"], 13),
        ("bd_self_cond", 172, 46, "Self-conditioning", 14),
        ("bd_decoder", 228, 74, [f"Decoder  ×{n_layers}", "bidirectional layers"], 15),
        ("bd_lm_head", 196, 50, "LM head · softcap", 14),
        ("bd_sampler", 204, 58, ["Accept / renoise", "(entropy bound)"], 13),
    ]
    bottom = loop_y + loop_h - 20   # canvas bottom edge
    for bid, bw, bh, label, fs in chain:
        top = bottom - bh
        pos[bid] = _rect_block(parts, info, shadow_id, bid,
                               den_cx - bw / 2, top, bw, bh, label, font_size=fs)
        bottom = top - gap          # next block sits one uniform gap higher

    dec = pos["bd_decoder"]

    # ── Encoder column (left, outside the loop) — KV store level with decoder ──
    pos["bd_kv_cache"] = _kv_store_glyph(
        parts, info, shadow_id, "bd_kv_cache",
        enc_cx - 62, dec["cy"] - 23, 124, 46, "KV Cache")
    kv = pos["bd_kv_cache"]
    pos["bd_encoder"] = _rect_block(
        parts, info, shadow_id, "bd_encoder",
        enc_cx - 92, kv["bottom"] + gap, 184, 72,
        [f"Encoder  ×{n_layers}", "causal layers"], font_size=15)
    enc = pos["bd_encoder"]
    pos["bd_prompt"] = _rect_block(
        parts, info, shadow_id, "bd_prompt",
        enc_cx - 78, enc["bottom"] + gap, 156, 42, "Prompt tokens", font_size=14)

    # ── Setup arrows: Prompt → Encoder → (writes) KV store ──
    parts.append(_v_line(pos["bd_prompt"], enc, arrow_id))
    parts.append(_v_line(enc, kv, arrow_id))

    # ── KV store → Decoder: a level read into the loop (decoder.cy == kv.cy) ──
    parts.append(_svg_tag("line", {
        "x1": kv["right"] + 5, "y1": kv["cy"],
        "x2": dec["left"] - GAP, "y2": dec["cy"],
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    # ── Denoising chain flow arrows (all `gap` long) ──
    for lo, hi in zip(chain, chain[1:]):
        parts.append(_v_line(pos[lo[0]], pos[hi[0]], arrow_id))

    # ── Output: the loop's committed result exits the top of the frame ──
    sam = pos["bd_sampler"]
    parts.append(_svg_tag("line", {
        "x1": den_cx, "y1": sam["top"], "x2": den_cx, "y2": loop_y - 22,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    # ── Feedback: the Sampler's result seeds the next step.  Two nested arcs
    # (no labels) both ORIGINATE at the Sampler and run the right rails down to
    # the Canvas (renoise) and Self-cond (prev logits) — nested so they never
    # cross, identical weight so the loop reads as one feedback mechanism. ──
    sc, can = pos["bd_self_cond"], pos["bd_canvas"]
    rail_inner, rail_outer = loop_x + loop_w - 46, loop_x + loop_w - 22
    r = 10

    def _feedback(y_exit, rail, dst):
        return (
            f"M {sam['right'] + 5} {y_exit} "
            f"L {rail - r} {y_exit} Q {rail} {y_exit} {rail} {y_exit + r} "
            f"L {rail} {dst['cy'] - r} Q {rail} {dst['cy']} {rail - r} {dst['cy']} "
            f"L {dst['right'] + GAP} {dst['cy']}"
        )

    for y_exit, rail, dst in (
        (sam["cy"] - 7, rail_inner, sc),    # → Self-conditioning (shorter, inner)
        (sam["cy"] + 7, rail_outer, can),   # → Canvas (longer, outer)
    ):
        parts.append(_svg_tag("path", {
            "d": _feedback(y_exit, rail, dst),
            "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
            "stroke-linecap": "round", "stroke-linejoin": "round",
            "marker-end": f"url(#{arrow_id})",
        }))

    return _svg(w, h, f"{ir.get('name', 'model')} block diffusion loop", parts)
