"""Detail SVG for the diffusion VAE decoder (AutoencoderKL).

Opened from the loop's ``vae_decode`` node.  Two levels:

* :func:`build_vae_decoder_view` — the decoder as a straight stage pipeline,
  latent → up stages → output head → image.  Boxes are uniform width; channel
  counts live in each stage's sub-label.  Each decoder stage is clickable and
  drills into...
* :func:`build_vae_decoder_block_view` — ...one decoder block's ResNet stack: the
  real residual structure (GroupNorm → SiLU → Conv 3×3, twice, + skip) with a
  ``×N`` depth-stack and the optional upsample op.

Everything (channels, ResNet count) is read from the VAE config the loader
fetched — nothing invented.  Internal op rects are non-interactive
(no ``data-id``) so a leaf view has no dangling click targets.
"""
from __future__ import annotations

from ....block_schema import DIFFUSION_PART_KINDS
from ..graph_engine import render_graph
from ..stack_view import fit_svg, point
from ..tower import tower_graph
from ..svg import _ids, _svg_tag, _svg_text, _v_line
from ..theme import C, FONT_HEAD, FONT_MONO


# ---------------------------------------------------------------------------
# Level 1 — the decoder as a straight stage pipeline
# ---------------------------------------------------------------------------

def build_vae_decoder_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """The decoder as a stage pipeline on the ONE tower backbone: in-port →
    up stages → output head → bare exit arrow.  Stage names only on blocks —
    channels / ResNet counts are fact chips on the stage cards."""
    d = block.get("detail") or {}
    channels = [c for c in (d.get("block_out_channels") or []) if isinstance(c, int)]
    children = {c.get("id"): c for c in (block.get("children") or [])}
    latent = d.get("latent_channels")

    pre: list[dict] = []
    for idx, _c in enumerate(reversed(channels), start=1):
        block_no = len(channels) - idx + 1
        node_id = f"vae_decoder_block_{block_no}"
        child = children.get(node_id, {})
        part_kind = child.get("diffusion_part_kind") or (child.get("detail") or {}).get("diffusion_part_kind")
        pre.append({
            "id": node_id, "kind": "embedding",
            "label": child.get("title") or f"Up stage {block_no}",
            "resolved": part_kind in DIFFUSION_PART_KINDS,
            "w": 240, "h": 58,
        })
    if channels:
        # Same condition as the card author — a node must never be clickable
        # without a card behind it.
        pre.append({"id": "vae_output_head", "kind": "embedding", "label": "Output head",
                    "w": 240, "h": 58})

    graph = tower_graph({
        "source": {"id": "vae_clean_latent",
                   "label": (f"in ({latent} ch · latent res)" if latent else "in (latent)")},
        "pre": pre,
        "output": {"id": "vae_image"},
    })
    return render_graph(graph, info, mount_id, "vae",
                        f"{ir.get('name', 'model')} VAE decoder", min_width=600)


# ---------------------------------------------------------------------------
# Level 2 — one decoder block: the ResNet stack
# ---------------------------------------------------------------------------

def build_vae_decoder_block_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One decoder block's residual cell: input → (GroupNorm+SiLU → Conv 3×3) ×2
    → ⊕, with a skip bypass; the cell repeats ``N`` times, then optional upsample.

    Main path runs dead-centre; the skip arcs up the right; spacing is uniform.
    """
    d = block.get("detail") or {}
    channels = d.get("channels")
    resnets = d.get("resnets") or 1
    upsamples = bool(d.get("upsamples"))
    ch = f"{channels} ch" if channels else None

    arrow_id, shadow_id = _ids(mount_id, block.get("id") or "vae-block")
    parts: list[str] = []
    regions: list[dict] = []
    cx = 0.0
    opw, oph = 196.0, 44.0          # inner op boxes
    accw, acch = 216.0, 48.0        # input / output bookends
    upw, uph = 204.0, 46.0          # upsample
    cell_gap, edge_gap = 24.0, 34.0
    skip_lane = cx + 142.0

    # --- vertical layout (top -> bottom), each value is a box TOP y ---
    y = 0.0
    out_y = y; y += acch + edge_gap
    up_y = None
    if upsamples:
        up_y = y; y += uph + edge_gap
    plus_top = y; plus_cy = y + 15; y += 30 + cell_gap
    conv2_y = y; y += oph + cell_gap
    n2_y = y; y += oph + cell_gap
    conv1_y = y; y += oph + cell_gap
    n1_y = y; y += oph + edge_gap
    inp_y = y; y += acch

    # --- "repeat × N" container around the residual cell (⊕ down to n1) ---
    grp_top = plus_top - 18
    grp_bot = n1_y + oph + 18
    grp_left = cx - opw / 2 - 30
    grp_right = skip_lane + 96
    parts.append(_svg_tag("rect", {
        "x": grp_left, "y": grp_top, "width": grp_right - grp_left, "height": grp_bot - grp_top,
        "rx": 18, "ry": 18, "fill": C["bg_inner"], "opacity": 0.5,
        "stroke": C["block"], "stroke-width": 1.0, "stroke-dasharray": "5 4"}))
    regions += [point(grp_left, grp_top), point(grp_right, grp_bot)]
    badge_cx = grp_left + 64
    parts.append(_svg_tag("rect", {
        "x": badge_cx - 52, "y": grp_top + 12, "width": 104, "height": 26,
        "rx": 13, "ry": 13, "fill": C["bg_outer"], "stroke": C["border"], "stroke-width": 0.7}))
    parts.append(_svg_text(badge_cx, grp_top + 25, f"repeat × {resnets}",
                           {"text-anchor": "middle", "dominant-baseline": "central",
                            "fill": C["text"], "font-family": FONT_HEAD, "font-size": 15}))

    # --- boxes (all centred on cx) ---
    out = _stage_block(parts, cx, out_y, accw, acch, "Block output", shadow_id,
                       sub=ch, accent=True)
    regions.append(out)
    up = None
    if upsamples:
        up = _stage_block(parts, cx, up_y, upw, uph, "Upsample", shadow_id,
                          sub="nearest 2× → conv", node_id="vae_op_upsample")
        regions.append(up)
    plus = _plain_plus(parts, cx, plus_cy, node_id="vae_op_residual")
    conv2 = _stage_block(parts, cx, conv2_y, opw, oph, "Conv 3×3", shadow_id, node_id="vae_op_conv")
    n2 = _stage_block(parts, cx, n2_y, opw, oph, "GroupNorm + SiLU", shadow_id, node_id="vae_op_norm")
    conv1 = _stage_block(parts, cx, conv1_y, opw, oph, "Conv 3×3", shadow_id, node_id="vae_op_conv")
    n1 = _stage_block(parts, cx, n1_y, opw, oph, "GroupNorm + SiLU", shadow_id, node_id="vae_op_norm")
    inp = _stage_block(parts, cx, inp_y, accw, acch, "Block input", shadow_id, sub=ch, accent=True)
    regions += [plus, conv2, n2, conv1, n1, inp]

    # --- flow: input → split → main path ↑ → ⊕ ; skip arcs up the right ---
    split_cy = (grp_bot + inp_y) / 2
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": inp["top"], "x2": cx, "y2": n1["bottom"] + 6,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none"}))
    parts.append(_svg_tag("circle", {"cx": cx, "cy": split_cy, "r": 3.6, "fill": C["arrow"]}))
    for src, dst in ((n1, conv1), (conv1, n2), (n2, conv2)):
        parts.append(_v_line(src, dst, arrow_id))
    # conv2 → ⊕ (straight up, centred)
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": conv2["top"], "x2": cx, "y2": plus["bottom"] + 4,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none"}))
    # skip: split → right → up → into ⊕ from the right
    parts.append(_svg_tag("path", {
        "d": (f"M {cx} {split_cy} L {skip_lane} {split_cy} "
              f"L {skip_lane} {plus_cy} L {plus['right'] + 5} {plus_cy}"),
        "fill": "none", "stroke": C["arrow"], "stroke-width": 1.5,
        "stroke-linecap": "round", "stroke-linejoin": "round", "stroke-dasharray": "5 4",
        "marker-end": f"url(#{arrow_id})"}))
    parts.append(_svg_text(skip_lane + 10, (split_cy + plus_cy) / 2 - 7, "skip",
                           {"fill": C["muted"], "font-family": FONT_MONO, "font-size": 11}))
    parts.append(_svg_text(skip_lane + 10, (split_cy + plus_cy) / 2 + 9, "identity / 1×1",
                           {"fill": C["muted"], "font-family": FONT_MONO, "font-size": 10}))

    # --- ⊕ → (upsample) → output ---
    if up is not None:
        parts.append(_v_line(plus, up, arrow_id))
        parts.append(_v_line(up, out, arrow_id))
    else:
        parts.append(_v_line(plus, out, arrow_id))

    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{ir.get('name', 'model')} {block.get('title') or 'VAE decoder block'}",
                   min_width=560, pad=44)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def _plain_plus(parts: list[str], cx: float, cy: float, r: float = 15.0,
                *, node_id: str | None = None) -> dict:
    glyph = [_svg_tag("circle", {
        "cx": cx, "cy": cy, "r": r, "fill": C["block"],
        "stroke": C["block_alt"], "stroke-width": 0.6})]
    arm = 5
    for x1, y1, x2, y2 in ((cx - arm, cy, cx + arm, cy), (cx, cy - arm, cx, cy + arm)):
        glyph.append(_svg_tag("line", {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "stroke": C["text_block"],
            "stroke-width": 2.2, "stroke-linecap": "round", "pointer-events": "none"}))
    if node_id:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(glyph)))
    else:
        parts.extend(glyph)
    return {"left": cx - r, "right": cx + r, "top": cy - r, "bottom": cy + r,
            "cx": cx, "cy": cy, "w": 2 * r, "h": 2 * r, "r": r}


def _stage_block(
    parts,
    cx,
    y,
    w,
    h,
    main,
    shadow_id,
    *,
    node_id: str | None = None,
    sub: str | None = None,
    accent=False,
    resolved=True,
) -> dict:
    fill = C["bg_inner"] if accent else C["block"] if resolved else C["badge_bg"]
    main_color = C["text"] if accent or not resolved else C["text_block"]
    sub_color = C["muted"] if accent or not resolved else "rgba(255,255,255,0.84)"
    stroke = C["block_alt"] if resolved else C["border"]
    stroke_width = 0.6 if resolved else 1.0
    x = cx - w / 2
    children = [_svg_tag("rect", {
        "x": x, "y": y, "width": w, "height": h, "rx": 11, "ry": 11,
        "fill": fill, "stroke": stroke, "stroke-width": stroke_width,
        "filter": f"url(#{shadow_id})",
    })]
    children.append(_svg_text(
        cx, y + (19 if sub else h / 2), main,
        {
            "text-anchor": "middle",
            "dominant-baseline": "central",
            "fill": main_color,
            "font-family": FONT_HEAD,
            "font-size": 17,
            "pointer-events": "none",
        },
    ))
    if sub:
        children.append(_svg_text(
            cx, y + 38, sub,
            {
                "text-anchor": "middle",
                "dominant-baseline": "central",
                "fill": sub_color,
                "font-family": FONT_MONO,
                "font-size": 11,
                "pointer-events": "none",
            },
        ))
    if node_id:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)
    return {
        "left": x,
        "right": x + w,
        "top": y,
        "bottom": y + h,
        "cx": cx,
        "cy": y + h / 2,
        "w": w,
        "h": h,
    }


