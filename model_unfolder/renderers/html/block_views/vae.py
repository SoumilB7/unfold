"""Detail SVG for the diffusion VAE decoder (AutoencoderKL).

Opened from the loop's ``vae_decode`` node.  Two levels:

* :func:`build_vae_decoder_view` — the decoder as a *resolution funnel*: each
  stage widens toward the image (spatial doubles on every upsample) while the
  channel count drops, so the upscaling is legible at a glance.  Each decoder
  stage is clickable and drills into...
* :func:`build_vae_decoder_block_view` — ...one decoder block's ResNet stack: the
  real residual structure (GroupNorm → SiLU → Conv 3×3, twice, + skip) with a
  ``×N`` depth-stack and the optional ×2 upsample.

Everything (channels, ResNet count, upsample factor) is read from the VAE config
the loader fetched — nothing invented.  Internal op rects are non-interactive
(no ``data-id``) so a leaf view has no dangling click targets.
"""
from __future__ import annotations

from ..stack_view import fit_svg, point
from ..svg import _ids, _svg_tag, _svg_text, _v_line
from ..theme import C, FONT_HEAD, FONT_MONO


# ---------------------------------------------------------------------------
# Level 1 — the decoder as a resolution funnel
# ---------------------------------------------------------------------------

def build_vae_decoder_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    d = block.get("detail") or {}
    channels = [c for c in (d.get("block_out_channels") or []) if isinstance(c, int)]
    children = {c.get("id"): c for c in (block.get("children") or [])}
    resnets = (d.get("layers_per_block") or 1) + 1
    out_ch = d.get("out_channels") or 3
    latent = d.get("latent_channels")
    scale = 2 ** (len(channels) - 1) if channels else None

    arrow_id, shadow_id = _ids(mount_id, "vae")
    parts: list[str] = []
    regions: list[dict] = []
    cx, h, gap = 0.0, 58.0, 40.0

    # Stage descriptors, bottom (latent) -> top (image).  ``level`` is the spatial
    # scale (each upsample +1); width grows with it to form the funnel.
    stages: list[dict] = [{
        "id": "vae_clean_latent", "title": "Clean latent",
        "sub": (f"{latent} ch · latent res" if latent else "latent resolution"),
        "level": 0, "accent": True, "up": False,
    }]
    level = 0
    for idx, c in enumerate(reversed(channels), start=1):
        up = idx > 1
        if up:
            level += 1
        block_no = len(channels) - idx + 1
        stages.append({
            "id": f"vae_decoder_block_{block_no}",
            "title": f"Decoder block {block_no}",
            "sub": None,
            "level": level, "accent": False, "up": up,
        })
    top = level
    stages.append({
        "id": "vae_output_head", "title": "Output head",
        "sub": f"conv 3×3 → {out_ch} ch", "level": top, "accent": False, "up": False,
    })
    stages.append({
        "id": "vae_image", "title": "Image",
        "sub": (f"{out_ch}ch RGB · {scale}× upscaled" if scale else f"{out_ch} ch"),
        "level": top, "accent": True, "up": False,
    })

    def width(level: int, accent: bool) -> float:
        return 150.0 + level * 42.0 + (14.0 if accent else 0.0)

    # Draw top -> bottom (image first), then connect bottom -> top.
    geoms: dict[str, dict] = {}
    y = 0.0
    for st in reversed(stages):
        title = children.get(st["id"], {}).get("title") or st["title"]
        g = _stage_block(parts, cx, y, width(st["level"], st["accent"]), h,
                         title, shadow_id, sub=st["sub"],
                         accent=st["accent"], node_id=st["id"])
        geoms[st["id"]] = g
        regions.append(g)
        y += h + gap

    chain = [geoms[st["id"]] for st in stages]
    for src, dst in zip(chain, chain[1:]):
        parts.append(_v_line(src, dst, arrow_id))

    # ×2 markers on the arrow feeding each upsampling stage.
    for st in stages:
        if st["up"]:
            g = geoms[st["id"]]
            ly = g["bottom"] + gap / 2
            parts.append(_svg_text(
                g["right"] + 18, ly, "↑ 2×",
                {"dominant-baseline": "central", "fill": C["muted"],
                 "font-family": FONT_MONO, "font-size": 12}))
            regions.append(point(g["right"] + 56, ly))

    # A faint side axis cueing the funnel direction.
    regions.append(point(0, -8))
    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{ir.get('name', 'model')} VAE decoder", min_width=600, pad=44)


# ---------------------------------------------------------------------------
# Level 2 — one decoder block: the ResNet stack
# ---------------------------------------------------------------------------

def build_vae_decoder_block_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    d = block.get("detail") or {}
    channels = d.get("channels")
    resnets = d.get("resnets") or 1
    upsamples = bool(d.get("upsamples"))
    ch = f"{channels} ch" if channels else None

    arrow_id, shadow_id = _ids(mount_id, block.get("id") or "vae-block")
    parts: list[str] = []
    regions: list[dict] = []
    cx = 0.0
    h_acc, h_op, gap = 50.0, 42.0, 30.0
    h_up = 54.0
    grp_w = 470.0
    grp_h = 318.0
    main_x = cx - 70.0
    skip_x = cx + 170.0

    y = 0.0
    y_out = y; y += h_acc + gap
    if upsamples:
        y_up = y; y += h_up + gap
    grp_top = y; y += grp_h + gap
    y_in = y; y += h_acc

    out = _stage_block(parts, cx, y_out, 230, h_acc, "Block output", shadow_id,
                       sub=(f"{ch} · 2× spatial" if (ch and upsamples) else ch), accent=True)
    regions.append(out)
    if upsamples:
        up = _stage_block(parts, cx, y_up, 224, h_up, "Upsample", shadow_id,
                          sub="nearest 2×, then conv")
        regions.append(up)

    # Flat repeat container: the VAE decoder block contains several ordinary
    # residual cells, but it is not a third spatial dimension.
    parts.append(_svg_tag("rect", {
        "x": cx - grp_w / 2, "y": grp_top, "width": grp_w, "height": grp_h,
        "rx": 18, "ry": 18, "fill": C["bg_inner"], "opacity": 0.55,
        "stroke": C["block"], "stroke-width": 1.0, "stroke-dasharray": "5 4"}))
    # The container is wider than the inner op boxes — register its corners so
    # fit_svg's bounding box includes it (otherwise its edges get clipped).
    regions.append(point(cx - grp_w / 2, grp_top))
    regions.append(point(cx + grp_w / 2, grp_top + grp_h))
    parts.append(_svg_tag("rect", {
        "x": cx + grp_w / 2 - 118, "y": grp_top + 14, "width": 96, "height": 28,
        "rx": 14, "ry": 14, "fill": C["bg_outer"], "stroke": C["border"],
        "stroke-width": 0.7}))
    parts.append(_svg_text(
        cx + grp_w / 2 - 70, grp_top + 28, f"repeat × {resnets}",
        {"text-anchor": "middle", "dominant-baseline": "central", "fill": C["text"],
         "font-family": FONT_HEAD, "font-size": 17}))

    split = {"cx": cx, "cy": grp_top + grp_h - 30, "left": cx - 3.2, "right": cx + 3.2,
             "top": grp_top + grp_h - 33.2, "bottom": grp_top + grp_h - 26.8}
    # The merge sits ABOVE the top Conv box (which starts at grp_top+36), so it
    # isn't hidden behind it.
    plus = _plain_plus(parts, cx, grp_top + 18)
    regions.append(plus)

    n1 = _stage_block(parts, main_x, grp_top + 220, 178, h_op, "GroupNorm + SiLU", shadow_id)
    conv1 = _stage_block(parts, main_x, grp_top + 156, 178, h_op, "Conv 3×3", shadow_id)
    n2 = _stage_block(parts, main_x, grp_top + 96, 178, h_op, "GroupNorm + SiLU", shadow_id)
    conv2 = _stage_block(parts, main_x, grp_top + 36, 178, h_op, "Conv 3×3", shadow_id)
    skip = _stage_block(parts, skip_x, grp_top + 142, 142, 46, "Skip path", shadow_id,
                        sub="identity / 1×1")
    regions += [n1, conv1, n2, conv2, skip]

    inp = _stage_block(parts, cx, y_in, 230, h_acc, "Block input", shadow_id, sub=ch, accent=True)
    regions.append(inp)

    # Flow: input enters the repeated residual stack, splits into main and skip
    # branches, rejoins at +, then optional upsample and output.
    parts.append(_v_line(inp, split, arrow_id))
    parts.append(_svg_tag("circle", {
        "cx": split["cx"], "cy": split["cy"], "r": 3.4, "fill": C["arrow"]}))
    for dst in (n1, skip):
        parts.append(_svg_tag("path", {
            "d": (f"M {split['cx']} {split['cy']} L {dst['cx']} {split['cy']} "
                  f"L {dst['cx']} {dst['bottom'] + 6}"),
            "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
            "stroke-linecap": "round", "stroke-linejoin": "round",
            "marker-end": f"url(#{arrow_id})"}))
    for src, dst in ((n1, conv1), (conv1, n2), (n2, conv2)):
        parts.append(_v_line(src, dst, arrow_id))
    parts.append(_svg_tag("path", {
        "d": (f"M {conv2['cx']} {conv2['top']} L {conv2['cx']} {plus['cy']} "
              f"L {plus['left'] - 5} {plus['cy']}"),
        "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
        "stroke-linecap": "round", "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})"}))
    parts.append(_svg_tag("path", {
        "d": (f"M {skip['left']} {skip['cy']} L {plus['right'] + 42} {skip['cy']} "
              f"L {plus['right'] + 42} {plus['cy']} L {plus['right'] + 5} {plus['cy']}"),
        "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
        "stroke-linecap": "round", "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})"}))

    if upsamples:
        parts.append(_v_line(plus, up, arrow_id))
        parts.append(_v_line(up, out, arrow_id))
    else:
        parts.append(_v_line(plus, out, arrow_id))

    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{ir.get('name', 'model')} {block.get('title') or 'VAE decoder block'}",
                   min_width=640, pad=44)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def _plain_plus(parts: list[str], cx: float, cy: float, r: float = 15.0) -> dict:
    parts.append(_svg_tag("circle", {
        "cx": cx, "cy": cy, "r": r, "fill": C["block"],
        "stroke": C["block_alt"], "stroke-width": 0.6}))
    arm = 5
    for x1, y1, x2, y2 in ((cx - arm, cy, cx + arm, cy), (cx, cy - arm, cx, cy + arm)):
        parts.append(_svg_tag("line", {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "stroke": C["text_block"],
            "stroke-width": 2.2, "stroke-linecap": "round"}))
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
) -> dict:
    fill = C["bg_inner"] if accent else C["block"]
    main_color = C["text"] if accent else C["text_block"]
    sub_color = C["muted"] if accent else "rgba(255,255,255,0.84)"
    x = cx - w / 2
    children = [_svg_tag("rect", {
        "x": x, "y": y, "width": w, "height": h, "rx": 11, "ry": 11,
        "fill": fill, "stroke": C["block_alt"], "stroke-width": 0.6,
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
