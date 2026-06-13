"""Detail SVG for a UNet diffusion denoiser (UNet2DConditionModel).

Opened from the loop's ``denoiser`` node for UNet pipelines.  Drawn as a U: the
down path (encoder) on the left going down, a mid block at the bottom, the up
path (decoder) on the right coming back up, and dashed **skip connections**
linking equal-resolution stages across the U.  Each stage shows its channel
width, ResNet count, and whether it has cross-attention to text.

Read from ``extras["unet"]`` (see ``adapters/diffusor/unet.py``).  Every stage box
is a clickable node (``unet_down_i`` / ``unet_mid`` / ``unet_up_j`` / ``unet_conv_in``
/ ``unet_conv_out``) coupled to a card declared in ``unet_denoiser_children`` —
the box shows only the stage name; channels / ResNet / attention counts are chips
on the card.
"""
from __future__ import annotations

from ....block_schema import DIFFUSION_PART_KINDS
from ..graph_engine import render_graph
from ..stack_view import fit_svg
from ..svg import _ids, _path, _svg_tag, _svg_text, _v_seg
from ..theme import C, FONT_HEAD
from ..tower import tower_graph


def build_unet_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    u = (ir.get("extras") or {}).get("unet") or {}
    down = u.get("down") or []
    up = u.get("up") or []
    mid = u.get("mid") or {}
    n = len(down)

    arrow_id, shadow_id = _ids(mount_id, "unet")
    parts: list[str] = []
    regions: list[dict] = []
    LX, RX = 188.0, 588.0
    sw, sh, row_gap, y0 = 200.0, 58.0, 120.0, 92.0
    MERGE_DROP = 26.0                 # the concat connector sits this far below an up stage
    DOWN_DROP = sh / 2 + MERGE_DROP   # each down stage sits at its skip's entry level, so
                                      # the skip runs STRAIGHT into the connector (no elbow)

    def stage(cx: float, y: float, st: dict, default_kind: str) -> dict:
        # Stage NAME only on the box; channels / ResNet / attention counts are
        # fact chips on the stage's card (numbers never ride the diagram block).
        resolved = st.get("diffusion_part_kind") in DIFFUSION_PART_KINDS
        return _box(parts, cx, y, sw, sh, _stage_title(st, default_kind), shadow_id,
                    resolved=resolved, node_id=st.get("id"))

    conv_in = _box(parts, LX, 22, sw, 44, "Conv in", shadow_id, node_id="unet_conv_in")
    conv_out = _box(parts, RX, 22, sw, 44, "Conv out", shadow_id, node_id="unet_conv_out")
    regions += [conv_in, conv_out]

    # Right column = up stages; left column row i = the matching down stage, sat
    # DOWN_DROP lower so its right-edge centre is level with the up stage's concat
    # connector — that makes every skip a straight horizontal arrow.
    down_g, up_g = [], []
    for i in range(n):
        y = y0 + i * row_gap
        up_g.append(stage(RX, y, up[n - 1 - i], "up_stage"))
        down_g.append(stage(LX, y + DOWN_DROP, down[i], "down_stage"))
    regions += down_g + up_g

    mid_cx = (LX + RX) / 2
    mid_y = down_g[n - 1]["bottom"] + 26
    mid_g = _box(parts, mid_cx, mid_y, sw + 44, sh, _stage_title(mid, "mid_stage"), shadow_id,
                 resolved=mid.get("diffusion_part_kind") in DIFFUSION_PART_KINDS,
                 node_id="unet_mid")
    regions.append(mid_g)

    # --- DOWN path: conv_in -> down stages -> mid (the resolution change per
    # stage lives on the cards + the denoiser description, not on the diagram). ---
    parts.append(_v_seg(LX, conv_in["bottom"], down_g[0]["top"] - 6, arrow_id))
    for i in range(n - 1):
        parts.append(_v_seg(LX, down_g[i]["bottom"], down_g[i + 1]["top"] - 6, arrow_id))
    parts.append(_path(f"M {LX} {down_g[n-1]['bottom']} L {LX} {mid_g['cy']} L {mid_g['left'] - 6} {mid_g['cy']}", arrow_id))

    # --- UP path: at EACH up stage, the skip from the matching down stage and the
    # features coming up from below MERGE at a concat connector (‖) just below the
    # stage, then go on up into it.  UNet skips concatenate along the channel axis
    # — two arrows in, one out; solid lines, ⊕ reserved for addition. ---
    parts.append(_path(  # mid -> below the bottom up stage (into its concat)
        f"M {mid_g['right']} {mid_g['cy']} L {RX} {mid_g['cy']} "
        f"L {RX} {up_g[n-1]['bottom'] + MERGE_DROP + 11}", arrow_id))
    for i in range(n):
        b = up_g[i]
        my = b["bottom"] + MERGE_DROP              # == down_g[i] centre → straight skip
        if i < n - 1:                              # features coming up from below (straight)
            parts.append(_v_seg(RX, up_g[i + 1]["top"], my + 11, arrow_id))
        parts.append(_svg_tag("line", {           # the skip, straight in from the left
            "x1": down_g[i]["right"], "y1": my, "x2": RX - 11, "y2": my,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none"}))
        _concat_node(parts, RX, my)
        parts.append(_v_seg(RX, my - 11, b["bottom"] + 4, arrow_id))   # ‖ -> up stage (straight)
    parts.append(_v_seg(RX, up_g[0]["top"], conv_out["bottom"] + 6, arrow_id))

    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{ir.get('name', 'model')} U-net denoiser", min_width=720, pad=44)


def build_unet_stage_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One resolution stage's internals on the ONE tower backbone — the same
    residual-cell pattern the VAE decoder block view uses: in → [GroupNorm+SiLU →
    Conv 3×3, twice → ⊕ residual (+ a Transformer: self-attn → text cross-attn →
    FF when the stage has cross-attention)] × N → optional 2× resample → out.

    Every op is a REAL clickable node whose id matches a described card declared
    in ``unet_denoiser_children`` (``_unet_stage_ops``) — not a static text blob.
    The structural shape (kinds, the residual lane) lives here; the prose lives on
    the cards.  Read from the stage's ``detail``; nothing invented."""
    d = block.get("detail") or {}
    ch = d.get("channels")
    resnets = int(d.get("resnets") or 1)
    direction = d.get("direction")
    op = {"w": 216, "h": 44}

    cell = [
        {"id": "unet_op_norm1", "kind": "norm", "label": "GroupNorm + SiLU", **op},
        {"id": "unet_op_conv1", "kind": "embedding", "label": "Conv 3×3", **op},
        {"id": "unet_op_norm2", "kind": "norm", "label": "GroupNorm + SiLU", **op},
        {"id": "unet_op_conv2", "kind": "embedding", "label": "Conv 3×3", **op},
        {"id": "unet_op_residual", "kind": "residual_add", "residual_from": "unet_op_norm1"},
    ]
    if d.get("attn"):
        cell += [
            {"id": "unet_op_selfattn", "kind": "attention", "label": "Self-attention", **op},
            {"id": "unet_op_crossattn", "kind": "attention", "label": "Cross-attention (text)", **op},
            {"id": "unet_op_ff", "kind": "ffn", "label": "Feed-forward", **op},
        ]
    post = ([{"id": f"unet_op_{'down' if direction == 'down' else 'up'}sample",
              "kind": "embedding",
              "label": "Downsample" if direction == "down" else "Upsample",
              "sub": "stride-2 conv" if direction == "down" else "nearest 2× → conv", **op}]
            if d.get("sample") else [])

    graph = tower_graph({
        "source": {"id": "unet_stage_in", "label": f"in ({ch:,} ch)" if ch else "in"},
        "cell": cell,
        "repeat": resnets,
        "post": post,
        "output": {"id": "unet_stage_out"},
    })
    return render_graph(graph, info, mount_id, f"unetstage_{block.get('id') or 'x'}",
                        f"{ir.get('name', 'model')} {block.get('title') or 'U-net stage'}",
                        min_width=560)


def _stage_title(st: dict, default_kind: str) -> str:
    if st.get("custom_label"):
        return str(st["custom_label"])
    kind = st.get("diffusion_part_kind") or default_kind
    return {
        "down_stage": "Down stage",
        "mid_stage": "Mid stage",
        "up_stage": "Up stage",
    }.get(kind, "Stage")


def _concat_node(parts: list[str], cx: float, cy: float) -> None:
    """A concatenation connector (‖) where a skip merges into an up stage.  Two
    parallel bars — deliberately NOT a '+': UNet skips concatenate along the
    channel axis, and ⊕ is reserved across the package for addition."""
    parts.append(_svg_tag("circle", {
        "cx": cx, "cy": cy, "r": 11, "fill": C["block"],
        "stroke": C["block_alt"], "stroke-width": 0.6}))
    for dx in (-3.0, 3.0):
        parts.append(_svg_tag("line", {
            "x1": cx + dx, "y1": cy - 5, "x2": cx + dx, "y2": cy + 5,
            "stroke": C["text_block"], "stroke-width": 2,
            "stroke-linecap": "round", "pointer-events": "none"}))


def _box(parts, cx, y, w, h, main, shadow_id, *, resolved=True, node_id=None) -> dict:
    """A solid, clickable U-net node (name only).  No light input/output accent —
    conv-in/out are real conv ops, drawn solid like every other stage; their
    dims live on the card.  ``node_id`` wraps it as a click target coupled to its
    card."""
    fill = C["block"] if resolved else C["badge_bg"]
    main_color = C["text_block"] if resolved else C["text"]
    stroke = C["block_alt"] if resolved else C["border"]
    stroke_width = 0.6 if resolved else 1.0
    x = cx - w / 2
    children = [
        _svg_tag("rect", {
            "x": x, "y": y, "width": w, "height": h, "rx": 11, "ry": 11,
            "fill": fill, "stroke": stroke, "stroke-width": stroke_width,
            "filter": f"url(#{shadow_id})"}),
        _svg_text(cx, y + h / 2, main,
                  {"text-anchor": "middle", "dominant-baseline": "central",
                   "fill": main_color, "font-family": FONT_HEAD, "font-size": 17,
                   "pointer-events": "none"}),
    ]
    if node_id:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)
    return {"left": x, "right": x + w, "top": y, "bottom": y + h,
            "cx": cx, "cy": y + h / 2, "w": w, "h": h}
