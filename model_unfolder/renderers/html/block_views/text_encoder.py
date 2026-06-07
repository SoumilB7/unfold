"""Detail SVG for a diffusion text encoder (CLIP / T5 / …).

Opened from the sampling loop's per-encoder node (``encoder_0``, ``encoder_1``,
…).  A diffusion pipeline's text encoders are *separate* transformer models; the
loader fetches each one's ``config.json`` when it can, so the view shows the real
depth/width/heads (``× 12 layers``, ``12 heads``, ``768 → 3072``) and falls back
to a schematic ``× N`` only when no encoder config was available — never an
invented number.

The layer cell is the classic *pre-norm* Transformer block, drawn with the same
residual convention as the LLM decoder-layer view: ``Norm → sublayer → ⊕``,
where the ⊕ (``_plus_block``) adds back the block input via a solid residual loop
(``_residual_loop_right``) that taps the stem *below* the norm.  Both CLIP and T5
are pre-norm.

The op boxes/nodes (embedding, norm, self-attention, FFN, residual add) are
clickable ``.uf-node``s that drill into a description card — see
``_text_encoder_ops`` in ``adapters/diffusor/blocks.py`` for the matching cards.
"""
from __future__ import annotations

from ..stack_view import fit_svg, point
from ..svg import _ids, _plus_block, _residual_loop_right, _svg_tag, _svg_text, _v_line
from ..theme import C, FONT_HEAD, FONT_MONO


def build_text_encoder_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    d = block.get("detail") or {}
    name = str(d.get("name") or block.get("title") or "Text encoder")
    text_dim = d.get("text_dim")
    pooled = d.get("pooled")
    layers, hidden, heads, ffn = d.get("layers"), d.get("hidden"), d.get("heads"), d.get("ffn")
    vocab = d.get("vocab")
    pfx = d.get("node_prefix") or block.get("id") or "encoder"
    upper = name.upper()
    is_clip = "CLIP" in upper
    is_t5 = "T5" in upper

    arrow_id, shadow_id = _ids(mount_id, "txtenc")
    parts: list[str] = []
    regions: list[dict] = []
    cx = 0.0
    sw, sh = 264.0, 52.0           # sublayer boxes (attention / FFN)
    nw, nh = 224.0, 42.0           # pre-norm strips
    ew, eh = 280.0, 52.0           # embedding
    aw, ah = 296.0, 56.0           # input / output bookends
    r = 14.0                       # residual-add circle radius
    g_io, g_sm, g_res = 40.0, 26.0, 42.0     # io gap / tight gap / residual-tap gap
    lane = cx + sw / 2 + 50        # residual bypass lane (right side)

    norm = "RMSNorm" if is_t5 else "LayerNorm"
    embed_main = "Token embedding" if is_t5 else "Token + positional embedding"
    embed_sub = " · ".join(s for s in (
        f"{_n(vocab)} vocab" if vocab else "",
        f"{_n(hidden)}-d" if hidden else "",
    ) if s) or None
    attn_sub = (f"{heads} heads" if heads else None)
    ffn_sub = (f"{_n(hidden)} → {_n(ffn)}" if (hidden and ffn) else None)
    if is_clip:
        out_main = "Pooled embedding"
        out_sub = f"1 × {_n(pooled)}-d global vector" if pooled else "global prompt vector"
        out_note = "→ global AdaLN conditioning"
    elif is_t5:
        out_main = "Token sequence"
        out_sub = f"tokens × {_n(text_dim)}-d" if text_dim else "per-token embeddings"
        out_note = "→ joint / cross attention"
    else:
        out_main = "Prompt embedding"
        out_sub = f"width {_n(text_dim)}" if text_dim else "conditioning embedding"
        out_note = "→ denoiser conditioning"
    badge = f"× {layers} layers" if layers else "× N layers"

    # --- vertical layout (top -> bottom); each value is a box TOP y (circles: cy) ---
    header = 38.0
    y = 0.0
    out_y = y; y += ah + g_io
    grp_top = y; y += header
    add2_cy = y + r; y += 2 * r + g_sm
    ffn_y = y; y += sh + g_sm
    rms2_y = y; y += nh + g_res          # gap below rms2 hosts the FFN residual tap
    add1_cy = y + r; y += 2 * r + g_sm
    attn_y = y; y += sh + g_sm
    rms1_y = y; y += nh
    grp_bot = y + 26
    y += g_res                            # gap below rms1 hosts the attention residual tap
    emb_y = y; y += eh + g_io
    inp_y = y; y += ah

    # --- "× N layers" container around the transformer layer cell ---
    grp_left = cx - sw / 2 - 30
    grp_right = lane + 30
    parts.append(_svg_tag("rect", {
        "x": grp_left, "y": grp_top, "width": grp_right - grp_left, "height": grp_bot - grp_top,
        "rx": 18, "ry": 18, "fill": C["bg_inner"], "opacity": 0.5,
        "stroke": C["block"], "stroke-width": 1.0, "stroke-dasharray": "5 4"}))
    regions += [point(grp_left, grp_top), point(grp_right, grp_bot)]
    badge_w = max(120.0, 12.0 + 7.4 * len(badge))
    badge_cx = grp_left + badge_w / 2 + 12
    parts.append(_svg_tag("rect", {
        "x": badge_cx - badge_w / 2, "y": grp_top + 6, "width": badge_w, "height": 26,
        "rx": 13, "ry": 13, "fill": C["bg_outer"], "stroke": C["border"], "stroke-width": 0.7}))
    parts.append(_svg_text(badge_cx, grp_top + 19, badge,
                           {"text-anchor": "middle", "dominant-baseline": "central",
                            "fill": C["text"], "font-family": FONT_HEAD, "font-size": 15}))
    regions.append(point(badge_cx + badge_w / 2, grp_top))

    # --- boxes + residual-add circles (bottom -> top: emb, rms1, attn, ⊕, rms2, ffn, ⊕, out) ---
    out = _box(parts, cx, out_y, aw, ah, out_main, out_sub, shadow_id, accent=True)
    add2 = _plus_block(parts, info, shadow_id, f"{pfx}_op_add", cx, add2_cy)
    ffn = _box(parts, cx, ffn_y, sw, sh, "Feed-forward (FFN)", ffn_sub, shadow_id, node_id=f"{pfx}_op_ffn")
    rms2 = _box(parts, cx, rms2_y, nw, nh, norm, None, shadow_id, node_id=f"{pfx}_op_norm")
    add1 = _plus_block(parts, info, shadow_id, f"{pfx}_op_add", cx, add1_cy)
    attn = _box(parts, cx, attn_y, sw, sh, "Multi-head self-attention", attn_sub, shadow_id, node_id=f"{pfx}_op_selfattn")
    rms1 = _box(parts, cx, rms1_y, nw, nh, norm, None, shadow_id, node_id=f"{pfx}_op_norm")
    emb = _box(parts, cx, emb_y, ew, eh, embed_main, embed_sub, shadow_id, node_id=f"{pfx}_op_embed")
    inp = _box(parts, cx, inp_y, aw, ah, "Prompt tokens", "tokenized text", shadow_id, accent=True)
    regions += [out, ffn, rms2, attn, rms1, emb, inp,
                point(add1["cx"], add1["top"]), point(add2["cx"], add2["top"])]

    # --- main flow (bottom -> top) ---
    for src, dst in ((inp, emb), (emb, rms1), (rms1, attn), (attn, add1),
                     (add1, rms2), (rms2, ffn), (ffn, add2), (add2, out)):
        parts.append(_v_line(src, dst, arrow_id))

    # --- residuals: x (below the pre-norm) added back at the ⊕ — solid side loop ---
    parts.append(_residual_loop_right(rms1, add1, lane, arrow_id))
    parts.append(_residual_loop_right(rms2, add2, lane, arrow_id))

    # --- how the output is consumed downstream ---
    ny = out["top"] - 20
    parts.append(_svg_text(cx, ny, out_note, {
        "text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11}))
    regions.append(point(cx, ny - 6))

    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{name} text encoder", min_width=560, pad=46)


def _n(v) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)


def _box(parts, cx, y, w, h, main, sub, shadow_id, *, node_id=None, accent=False) -> dict:
    fill = C["bg_inner"] if accent else C["block"]
    main_color = C["text"] if accent else C["text_block"]
    sub_color = C["muted"] if accent else "rgba(255,255,255,0.84)"
    x = cx - w / 2
    children = [_svg_tag("rect", {
        "x": x, "y": y, "width": w, "height": h, "rx": 11, "ry": 11,
        "fill": fill, "stroke": C["block_alt"], "stroke-width": 0.6,
        "filter": f"url(#{shadow_id})"})]
    children.append(_svg_text(cx, y + (18 if sub else h / 2), main, {
        "text-anchor": "middle", "dominant-baseline": "central",
        "fill": main_color, "font-family": FONT_HEAD, "font-size": 16, "pointer-events": "none"}))
    if sub:
        children.append(_svg_text(cx, y + 37, sub, {
            "text-anchor": "middle", "dominant-baseline": "central",
            "fill": sub_color, "font-family": FONT_MONO, "font-size": 11, "pointer-events": "none"}))
    if node_id:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)
    return {"left": x, "right": x + w, "top": y, "bottom": y + h,
            "cx": cx, "cy": y + h / 2, "w": w, "h": h}
