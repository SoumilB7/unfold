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
from ..stack_view import fit_svg, point
from ..svg import _ids, _path, _svg_tag, _svg_text, _v_seg
from ..theme import C, FONT_HEAD, FONT_MONO
from ..tower import tower_graph


def _text_source_label(ir: dict):
    """Label for the 'encoded text' source — makes the two-CLIP origin visible.

    The text the U-net cross-attends is a SINGLE tensor: the text encoders' token
    features concatenated along the feature axis (SDXL: 768 + 1,280 = 2,048).  So
    one box is correct, but the label shows it is the concatenation of the N
    encoders, answering 'where did the second CLIP go?'."""
    extras = ir.get("extras") or {}
    encs = (extras.get("diffusion") or {}).get("text_encoders") or []
    cad = (extras.get("unet") or {}).get("cross_attention_dim")
    tail = f" → {cad:,}" if cad else ""
    if len(encs) >= 2:
        fam = "CLIP" if all("CLIP" in str(e) for e in encs) else "encoders"
        return ["Encoded text", f"{len(encs)}× {fam}{tail}"]
    if cad:
        return ["Encoded text", f"{cad:,}-d"]
    return "Encoded text"


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

    # --- TEXT CONDITIONING: the encoded prompt (the text encoders' K/V) enters
    # EVERY cross-attention stage. Drawn as a source at the bottom that broadcasts
    # laterally into each CrossAttn stage from its outer edge — latent flows
    # vertically through the U, conditioning enters from the side. ---
    _draw_text_conditioning(parts, regions, shadow_id, arrow_id, u,
                            down, up, mid, down_g, up_g, mid_g, LX, RX, sw,
                            text_label=_text_source_label(ir))

    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{ir.get('name', 'model')} U-net denoiser", min_width=720, pad=44)


def _draw_text_conditioning(parts, regions, shadow_id, arrow_id, u, down, up,
                            mid, down_g, up_g, mid_g, LX, RX, sw,
                            text_label="Encoded text") -> None:
    """Show the encoded text feeding the cross-attention stages.

    Only the stages whose block type carries cross-attention receive it (down[i]
    / up[n-1-i] with ``attn``, and the mid block).  A source box at the bottom
    fans out: straight up into the bottleneck, and out to a left/right lateral bus
    that taps each cross-attn stage on its OUTER edge.  Skipped entirely for an
    unconditional U-net (no ``cross_attention_dim`` / no attention stages)."""
    cad = u.get("cross_attention_dim")
    n = len(down)
    down_attn = [g for i, g in enumerate(down_g) if down[i].get("attn")]
    up_attn = [g for i, g in enumerate(up_g) if up[n - 1 - i].get("attn")]
    mid_attn = bool(mid.get("attn"))
    if not cad or not (down_attn or up_attn or mid_attn):
        return

    mid_cx = (LX + RX) / 2
    tb_h = 56.0 if isinstance(text_label, (list, tuple)) else 48.0
    tb = _box(parts, mid_cx, mid_g["bottom"] + 44, 252.0, tb_h, text_label,
              shadow_id, node_id="unet_text_cond")
    regions.append(tb)
    cy = tb["cy"]
    x_L = (LX - sw / 2) - 34          # lane left of the down column
    x_R = (RX + sw / 2) + 34          # lane right of the up column

    def tap(x_from: float, into_left_edge: bool, g: dict) -> None:
        edge = (g["left"] - 4) if into_left_edge else (g["right"] + 4)
        parts.append(_svg_tag("line", {
            "x1": x_from, "y1": g["cy"], "x2": edge, "y2": g["cy"],
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none"}))

    if mid_attn:                                   # straight up into the bottleneck
        parts.append(_svg_tag("line", {
            "x1": mid_cx, "y1": tb["top"], "x2": mid_cx, "y2": mid_g["bottom"] + 5,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none"}))
    if down_attn:                                  # left bus + taps into down stages
        top_y = min(g["cy"] for g in down_attn)
        parts.append(_svg_tag("path", {
            "d": f"M {tb['left']} {cy} L {x_L} {cy} L {x_L} {top_y}",
            "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
            "stroke-linecap": "round", "stroke-linejoin": "round"}))
        for g in down_attn:
            tap(x_L, True, g)
        regions.append(point(x_L - 6, cy))
    if up_attn:                                    # right bus + taps into up stages
        top_y = min(g["cy"] for g in up_attn)
        parts.append(_svg_tag("path", {
            "d": f"M {tb['right']} {cy} L {x_R} {cy} L {x_R} {top_y}",
            "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
            "stroke-linecap": "round", "stroke-linejoin": "round"}))
        for g in up_attn:
            tap(x_R, False, g)
        regions.append(point(x_R + 6, cy))


def build_unet_stage_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """A resolution stage: in → [ResNet block (+ Transformer block when the stage
    has cross-attention)] × layers_per_block → optional 2× resample → out.  The
    ResNet block and Transformer block are clickable and DRILL FURTHER (into the
    residual cell, and into self→cross→FF × depth respectively)."""
    d = block.get("detail") or {}
    ch = d.get("channels")
    resnets = int(d.get("resnets") or 1)
    direction = d.get("direction")
    op = {"w": 224, "h": 48}

    # Block NAME only — the per-block facts (transformer depth, self/cross/FF,
    # stride-2 conv) are chips/prose on each block's card, never a sub-caption on
    # the diagram block.
    cell = [{"id": "unet_resnet", "kind": "norm", "label": "ResNet block", **op}]
    if d.get("attn"):
        cell.append({"id": "unet_transformer", "kind": "attention",
                     "label": "Transformer block", **op})
    post = ([{"id": f"unet_{'down' if direction == 'down' else 'up'}sample",
              "kind": "embedding",
              "label": "Downsample" if direction == "down" else "Upsample",
              "w": 224, "h": 46}]
            if d.get("sample") else [])

    spec = {
        "source": {"id": "unet_stage_in", "label": f"in ({ch:,} ch)" if ch else "in"},
        "cell": cell,
        "repeat": resnets,
        "post": post,
        "output": {"id": "unet_stage_out"},
    }
    if d.get("attn"):
        # The encoded text enters the Transformer block (its cross-attention) — show
        # it feeding in from the side, the same conditioning seen one level up.
        spec["side_inputs"] = [{
            "node": {"id": "unet_stage_text", "kind": "embedding",
                     "label": _text_source_label(ir), "w": 210},
            "target": "unet_transformer",
        }]
    graph = tower_graph(spec)
    return render_graph(graph, info, mount_id, f"unetstage_{block.get('id') or 'x'}",
                        f"{ir.get('name', 'model')} {block.get('title') or 'U-net stage'}",
                        min_width=560)


def build_unet_resnet_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One ResNet block — the residual cell, the SAME shape as the VAE decoder
    block: in → GroupNorm+SiLU → Conv 3×3 → GroupNorm+SiLU → Conv 3×3 → ⊕ → out."""
    ch = (block.get("detail") or {}).get("channels")
    op = {"w": 216, "h": 44}
    graph = tower_graph({
        "source": {"id": "unet_res_in", "label": f"in ({ch:,} ch)" if ch else "in"},
        "cell": [
            {"id": "unet_op_norm1", "kind": "norm", "label": "GroupNorm + SiLU", **op},
            {"id": "unet_op_conv1", "kind": "embedding", "label": "Conv 3×3", **op},
            {"id": "unet_op_norm2", "kind": "norm", "label": "GroupNorm + SiLU", **op},
            {"id": "unet_op_conv2", "kind": "embedding", "label": "Conv 3×3", **op},
            {"id": "unet_op_residual", "kind": "residual_add", "residual_from": "unet_op_norm1"},
        ],
        "output": {"id": "unet_res_out"},
    })
    return render_graph(graph, info, mount_id, f"unetres_{block.get('id') or 'x'}",
                        f"{ir.get('name', 'model')} ResNet block", min_width=520)


def build_unet_transformer_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One Transformer2D block: in → [Self-attention → Cross-attention (text) →
    Feed-forward] × transformer_layers → out.  Each sub-block opens the canonical
    attention / feed-forward view (the same opener a transformer layer uses)."""
    t = int((block.get("detail") or {}).get("transformers") or 1)
    op = {"w": 232, "h": 46}
    graph = tower_graph({
        "source": {"id": "unet_tf_in", "label": "in (latent tokens)"},
        "cell": [
            {"id": "unet_selfattn", "kind": "attention", "label": "Self-attention", **op},
            {"id": "unet_crossattn", "kind": "attention", "label": "Cross-attention (text)", **op},
            {"id": "unet_ff", "kind": "ffn", "label": "Feed-forward", **op},
        ],
        "repeat": t,
        "output": {"id": "unet_tf_out"},
        # The encoded text enters beside the cross-attention sub-block — it supplies
        # that sublayer's K/V (self-attention and FF stay on the latent).
        "side_inputs": [{
            "node": {"id": "unet_tf_text", "kind": "embedding",
                     "label": _text_source_label(ir), "w": 210},
            "target": "unet_crossattn",
        }],
    })
    return render_graph(graph, info, mount_id, f"unettf_{block.get('id') or 'x'}",
                        f"{ir.get('name', 'model')} Transformer block", min_width=540)


def build_encoded_text_concat_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """How the text encoders combine into the cross-attention K/V.

    Each encoder produces a token sequence at its own width; SDXL concatenates the
    two CLIP penultimate hidden states along the feature axis (768 + 1,280 =
    2,048) — the single tensor the U-net cross-attends.  Drawn as N encoder boxes
    feeding one ``‖`` concat connector (the SAME op as the U-net skips) → K/V.  A
    single-encoder model (SD1.5) draws a straight pass-through (no concat)."""
    d = block.get("detail") or {}
    encoders = d.get("encoders") or []
    cad = d.get("cross_attention_dim")
    arrow_id, shadow_id = _ids(mount_id, "txtconcat")
    parts: list[str] = []
    regions: list[dict] = []

    bw, bh, gap = 168.0, 58.0, 52.0
    n = max(len(encoders), 1)
    y_enc, y_concat, y_out = 196.0, 96.0, 36.0
    total = n * bw + (n - 1) * gap
    x0 = -total / 2 + bw / 2

    enc_g: list[dict] = []
    for i in range(n):
        e = encoders[i] if i < len(encoders) else {"name": "Text encoder"}
        cxn = x0 + i * (bw + gap)
        hid = e.get("hidden")
        name = str(e.get("name") or "Text encoder")
        label = [name, f"{hid:,}-d"] if hid else name
        g = _box(parts, cxn, y_enc, bw, bh, label, shadow_id)
        enc_g.append(g)
        regions.append(g)

    if len(encoders) >= 2:
        # the ‖ concat connector is clickable — opens a card explaining the op
        _concat_node(parts, 0.0, y_concat, node_id="text_concat_op")
        regions.append(point(0.0, y_concat))
        for g in enc_g:                              # each encoder → the ‖ connector
            if abs(g["cx"]) < 1:
                parts.append(_v_seg(0.0, g["top"], y_concat + 11, arrow_id))
            else:
                side = -11 if g["cx"] < 0 else 11
                parts.append(_path(
                    f"M {g['cx']} {g['top']} L {g['cx']} {y_concat} L {side} {y_concat}",
                    arrow_id))
        parts.append(_v_seg(0.0, y_concat - 11, y_out, arrow_id))
    else:                                            # single encoder: straight to K/V
        parts.append(_v_seg(0.0, enc_g[0]["top"], y_out, arrow_id))

    out_label = f"K/V ({cad:,})" if cad else "cross-attention K/V"
    parts.append(_svg_text(0.0, y_out - 10, out_label, {
        "text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO,
        "font-size": 12}))
    regions.append(point(0.0, y_out - 18))

    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{ir.get('name', 'model')} encoded text → K/V", min_width=520, pad=44)


def _stage_title(st: dict, default_kind: str) -> str:
    if st.get("custom_label"):
        return str(st["custom_label"])
    kind = st.get("diffusion_part_kind") or default_kind
    return {
        "down_stage": "Down stage",
        "mid_stage": "Mid stage",
        "up_stage": "Up stage",
    }.get(kind, "Stage")


def _concat_node(parts: list[str], cx: float, cy: float, node_id: str | None = None) -> None:
    """A concatenation connector (‖) where lanes join along the feature axis.  Two
    parallel bars — deliberately NOT a '+': concatenation, and ⊕ is reserved
    across the package for addition.  ``node_id`` makes it a clickable drill
    target coupled to a card explaining the op."""
    children = [_svg_tag("circle", {
        "cx": cx, "cy": cy, "r": 11, "fill": C["block"],
        "stroke": C["block_alt"], "stroke-width": 0.6})]
    for dx in (-3.0, 3.0):
        children.append(_svg_tag("line", {
            "x1": cx + dx, "y1": cy - 5, "x2": cx + dx, "y2": cy + 5,
            "stroke": C["text_block"], "stroke-width": 2,
            "stroke-linecap": "round", "pointer-events": "none"}))
    if node_id:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)


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
    ]
    # A list label stacks: a heading line plus a smaller second line (e.g. the
    # "2× CLIP → 2,048" origin under "Encoded text").
    lines = [s for s in main if s] if isinstance(main, (list, tuple)) else [main]
    if len(lines) == 1:
        children.append(_svg_text(cx, y + h / 2, lines[0],
            {"text-anchor": "middle", "dominant-baseline": "central",
             "fill": main_color, "font-family": FONT_HEAD, "font-size": 17,
             "pointer-events": "none"}))
    else:
        children.append(_svg_text(cx, y + h / 2 - 9, lines[0],
            {"text-anchor": "middle", "dominant-baseline": "central",
             "fill": main_color, "font-family": FONT_HEAD, "font-size": 16,
             "pointer-events": "none"}))
        children.append(_svg_text(cx, y + h / 2 + 11, lines[1],
            {"text-anchor": "middle", "dominant-baseline": "central",
             "fill": main_color, "font-family": FONT_MONO, "font-size": 12,
             "pointer-events": "none"}))
    if node_id:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)
    return {"left": x, "right": x + w, "top": y, "bottom": y + h,
            "cx": cx, "cy": y + h / 2, "w": w, "h": h}
