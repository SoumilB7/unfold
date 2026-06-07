"""Detail SVG for a UNet diffusion denoiser (UNet2DConditionModel).

Opened from the loop's ``denoiser`` node for UNet pipelines.  Drawn as a U: the
down path (encoder) on the left going down, a mid block at the bottom, the up
path (decoder) on the right coming back up, and dashed **skip connections**
linking equal-resolution stages across the U.  Each stage shows its channel
width, ResNet count, and whether it has cross-attention to text.

Read from ``extras["unet"]`` (see ``adapters/diffusor/unet.py``).  Stage rects are
non-interactive (a leaf view) — no dangling click targets.
"""
from __future__ import annotations

from ..stack_view import fit_svg, point
from ..svg import _ids, _path, _svg_tag, _svg_text, _v_seg
from ..theme import C, FONT_HEAD, FONT_MONO


def build_unet_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    u = (ir.get("extras") or {}).get("unet") or {}
    down = u.get("down") or []
    up = u.get("up") or []
    mid = u.get("mid") or {}
    n = len(down)
    boc = u.get("block_out_channels") or []
    in_ch, out_ch = u.get("in_channels"), u.get("out_channels")
    cad = u.get("cross_attention_dim")
    scale = u.get("downscale")

    arrow_id, shadow_id = _ids(mount_id, "unet")
    parts: list[str] = []
    regions: list[dict] = []
    LX, RX = 178.0, 548.0
    sw, sh, row_gap, y0 = 162.0, 50.0, 80.0, 98.0

    def stage(cx: float, y: float, st: dict) -> dict:
        ch = st.get("channels")
        sub = f"{st.get('resnets')}× ResNet"
        if st.get("attn"):
            t = st.get("transformers") or 1
            sub += f"  ·  XAttn×{t}" if t > 1 else "  ·  +XAttn"
        return _box(parts, cx, y, sw, sh, f"{ch} ch" if ch else "stage", sub, shadow_id)

    conv_in = _box(parts, LX, 22, sw, 44, "Conv in",
                   (f"{in_ch} → {boc[0]} ch" if (in_ch and boc) else None), shadow_id, accent=True)
    conv_out = _box(parts, RX, 22, sw, 44, "Conv out",
                    (f"→ {out_ch} ch" if out_ch else None), shadow_id, accent=True)
    regions += [conv_in, conv_out]

    # Left column = down (top->bottom); right column row i = the matching up stage
    # (up[n-1-i]) so equal-resolution stages share a row for a horizontal skip.
    down_g, up_g = [], []
    for i in range(n):
        y = y0 + i * row_gap
        down_g.append(stage(LX, y, down[i]))
        up_g.append(stage(RX, y, up[n - 1 - i]))
    regions += down_g + up_g

    mid_cx, mid_y = (LX + RX) / 2, y0 + n * row_gap
    mid_g = _box(parts, mid_cx, mid_y, sw + 40, sh, "Mid block",
                 (f"{mid.get('channels')} ch · ResNet+Attn" if mid.get("channels") else "ResNet+Attn"),
                 shadow_id)
    regions.append(mid_g)

    # --- skip connections (equal-resolution stages across the U) ---
    for i in range(n):
        a, b = down_g[i], up_g[i]
        parts.append(_svg_tag("path", {
            "d": f"M {a['right']} {a['cy']} L {b['left']} {b['cy']}",
            "fill": "none", "stroke": C["arrow"], "stroke-width": 1.4,
            "stroke-linecap": "round", "stroke-dasharray": "5 4",
            "marker-end": f"url(#{arrow_id})"}))
    parts.append(_svg_text(
        (down_g[0]["right"] + up_g[0]["left"]) / 2, down_g[0]["cy"] - 12, "skip connections",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11}))

    # --- main flow: conv_in -> down -> mid -> up -> conv_out ---
    parts.append(_v_seg(LX, conv_in["bottom"], down_g[0]["top"] - 6, arrow_id))
    for i in range(n - 1):
        parts.append(_v_seg(LX, down_g[i]["bottom"], down_g[i + 1]["top"] - 6, arrow_id))
        if down[i].get("sample"):
            parts.append(_svg_text(LX - sw / 2 - 14, (down_g[i]["bottom"] + down_g[i + 1]["top"]) / 2,
                                   "↓2", {"text-anchor": "middle", "fill": C["muted"],
                                          "font-family": FONT_MONO, "font-size": 11}))
            regions.append(point(LX - sw / 2 - 30, down_g[i]["bottom"]))
    parts.append(_path(f"M {LX} {down_g[n-1]['bottom']} L {LX} {mid_g['cy']} L {mid_g['left'] - 6} {mid_g['cy']}", arrow_id))
    parts.append(_path(f"M {mid_g['right']} {mid_g['cy']} L {RX} {mid_g['cy']} L {RX} {up_g[n-1]['bottom'] + 6}", arrow_id))
    for i in range(n - 1, 0, -1):
        parts.append(_v_seg(RX, up_g[i]["top"], up_g[i - 1]["bottom"] + 6, arrow_id))
        if up[n - 1 - i].get("sample"):
            parts.append(_svg_text(RX + sw / 2 + 14, (up_g[i]["top"] + up_g[i - 1]["bottom"]) / 2,
                                   "↑2", {"text-anchor": "middle", "fill": C["muted"],
                                          "font-family": FONT_MONO, "font-size": 11}))
            regions.append(point(RX + sw / 2 + 30, up_g[i]["top"]))
    parts.append(_v_seg(RX, up_g[0]["top"], conv_out["bottom"] + 6, arrow_id))

    # --- conditioning caption (two lines so it doesn't overrun the canvas) ---
    line1 = "timestep → every ResNet" + (f"   ·   latent {scale}× downscaled" if scale else "")
    lines = [line1]
    if cad:
        lines.append(f"text conditioning (dim {cad}) → cross-attention stages")
    for k, cap in enumerate(lines):
        cy = mid_g["bottom"] + 26 + k * 18
        parts.append(_svg_text(mid_cx, cy, cap,
                               {"text-anchor": "middle", "fill": C["muted"],
                                "font-family": FONT_MONO, "font-size": 11}))
        half = len(cap) * 3.4  # approx half-width so fit_svg includes the caption
        regions.append(point(mid_cx - half, cy))
        regions.append(point(mid_cx + half, cy))

    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{ir.get('name', 'model')} U-net denoiser", min_width=720, pad=44)


def _box(parts, cx, y, w, h, main, sub, shadow_id, *, accent=False) -> dict:
    fill = C["bg_inner"] if accent else C["block"]
    main_color = C["text"] if accent else C["text_block"]
    sub_color = C["muted"] if accent else "rgba(255,255,255,0.84)"
    x = cx - w / 2
    parts.append(_svg_tag("rect", {
        "x": x, "y": y, "width": w, "height": h, "rx": 11, "ry": 11,
        "fill": fill, "stroke": C["block_alt"], "stroke-width": 0.6,
        "filter": f"url(#{shadow_id})"}))
    parts.append(_svg_text(cx, y + (18 if sub else h / 2), main,
                           {"text-anchor": "middle", "dominant-baseline": "central",
                            "fill": main_color, "font-family": FONT_HEAD, "font-size": 17,
                            "pointer-events": "none"}))
    if sub:
        parts.append(_svg_text(cx, y + 36, sub,
                               {"text-anchor": "middle", "dominant-baseline": "central",
                                "fill": sub_color, "font-family": FONT_MONO, "font-size": 10.5,
                                "pointer-events": "none"}))
    return {"left": x, "right": x + w, "top": y, "bottom": y + h,
            "cx": cx, "cy": y + h / 2, "w": w, "h": h}
