"""Top-level SVG views for architecture and layer maps."""
from __future__ import annotations

from ...labels import kind_short, mask_short
from .metadata import _block_label, _indices_summary, _signature
from .svg import (
    _block_top_to_block_bottom,
    _branch_dot,
    _defs,
    _elbow_hv,
    _elbow_vh,
    _input_tap,
    _ids,
    _plus_block,
    _rect_block,
    _region_rect,
    _residual_loop_right,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
)
from .theme import C, FONT_BODY, FONT_HEAD, FONT_MONO, GAP


# --- Layout vocabulary for the data-driven architecture view ----------------
# Each ``kind`` declares its glyph (rect / circle), nominal size, and font.
# A new architectural feature gets rendered by adding a kind here and tagging
# the relevant blocks in the adapter — no edits to the layout engine itself.
_KIND_LAYOUT = {
    "norm":         {"shape": "rect",   "w": 160, "h": 36, "font": 16},
    "linear":       {"shape": "rect",   "w": 200, "h": 38, "font": 15},
    "activation":   {"shape": "rect",   "w": 150, "h": 36, "font": 15},
    "attention":    {"shape": "rect",   "w": 230, "h": 60, "font": 17},
    "ffn":          {"shape": "rect",   "w": 160, "h": 44, "font": 17},
    "ple":          {"shape": "rect",   "w": 160, "h": 44, "font": 17},
    "vision":       {"shape": "rect",   "w": 210, "h": 46, "font": 15},
    "fusion":       {"shape": "rect",   "w": 230, "h": 50, "font": 15},
    "residual_add": {"shape": "circle", "w": 28,  "h": 28, "sym": "+"},
    "gate_mul":     {"shape": "circle", "w": 28,  "h": 28, "sym": "×"},
}
_BLOCK_GAP = 32  # vertical gap between consecutive layer-body blocks
# Larger than the arrow padding (`GAP` ×2) so the chain arrow has a visible
# stem between blocks rather than collapsing to just an arrowhead.


def _build_architecture_view(ir: dict, info: dict, mount_id: str) -> str:
    """Data-driven decoder architecture view.

    The layer body comes from ``info['dominant']['spec']['blocks']`` — an
    ordered list of typed blocks emitted by the adapter.  Each block's
    ``kind`` selects a glyph in :data:`_KIND_LAYOUT`; ``residual_from`` adds
    a residual loop when the bypass is not already represented by the central
    chain; side-lane blocks render as parallel rails to the left or right of
    the inner region.

    The view grows from the block list, so side-path models get extra vertical
    space while compact decoder-only models keep the same vocabulary.
    """
    spec = info["dominant"]["spec"]
    layer_blocks = list(spec.get("blocks") or [])

    # Side blocks live OFF the central column.  They share a row with the block
    # they feed but get their own offset x-position and explicit connections.
    chain_blocks = [b for b in layer_blocks if not b.get("lane")]
    side_blocks = [b for b in layer_blocks if b.get("lane")]

    inner_x, inner_w = 110, 500

    # Default chain center.  Auto-shift right when a side_align="tap" block on
    # the left lane would overlap the widest chain block at the default cx.
    # This handles parallel-residual architectures (e.g. GPT-NeoX / GPT-J) where
    # FFN and Attention share the same y-row without any renderer special-casing.
    cx = 360
    _tap_left = [
        b for b in side_blocks
        if b.get("side_align") == "tap" and b.get("lane") == "left"
    ]
    if _tap_left:
        side_right = max(
            inner_x + 30 + (_KIND_LAYOUT.get(b.get("kind"), _KIND_LAYOUT["norm"])["w"])
            for b in _tap_left
        )
        chain_half_w = max(
            (_KIND_LAYOUT.get(b.get("kind"), _KIND_LAYOUT["norm"])["w"]) // 2
            for b in chain_blocks
        ) if chain_blocks else 115
        cx = max(cx, side_right + 20 + chain_half_w)

    # --- 1. Compute heights from the chain block list ---
    inner_padding = 60
    stack_h = _layer_stack_height(chain_blocks)
    inner_h = max(490, stack_h + 2 * inner_padding)

    inner_y = 200
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    has_vision_fusion = "vision" in (modalities.get("inputs") or {}) and modalities.get("fusion")
    h = inner_y + inner_h + (292 if has_vision_fusion else 232)
    w = 720

    arrow_id, shadow_id = _ids(mount_id, "arch")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 26, w - 80, h - 52, C["bg_outer"]))
    parts.append(_region_rect(inner_x, inner_y, inner_w, inner_h, C["bg_inner"]))

    # --- 2. Model-level scaffold (positions tracked by total height h) ---
    if has_vision_fusion:
        tok_text, embed, stack_input = _draw_multimodal_input_scaffold(
            parts, info, shadow_id, arrow_id, cx, inner_y, inner_h, h,
        )
    else:
        tok_text = _rect_block(parts, info, shadow_id, "tok_text",
                               cx - 110, h - 100, 220, 44,
                               _block_label(info, "tok_text", "Tokenized text"), font_size=17)
        embed = _rect_block(parts, info, shadow_id, "embed",
                            cx - 130, h - 168, 260, 44,
                            _block_label(info, "embed", "Token Embedding layer"), font_size=17)
        stack_input = embed
    final_rms = _rect_block(parts, info, shadow_id, "final_rms",
                            cx - 90, 140, 180, 36,
                            _block_label(info, "final_rms", "Final RMSNorm"), font_size=16)
    lm_head = _rect_block(parts, info, shadow_id, "lm_head",
                          cx - 130, 70, 260, 44,
                          _block_label(info, "lm_head", "Linear output layer"), font_size=17)

    # --- 3. Layer body (data-driven, stacked bottom-up) ---
    block_pos: dict[str, dict] = {}
    free = inner_h - stack_h
    y_cursor = inner_y + inner_h - free / 2
    for block in chain_blocks:
        layout = _KIND_LAYOUT.get(block["kind"]) or _KIND_LAYOUT["norm"]
        block_h = layout["h"]
        top = y_cursor - block_h
        if layout["shape"] == "rect":
            geom = _rect_block(
                parts, info, shadow_id, block["id"],
                cx - layout["w"] / 2, top, layout["w"], block_h,
                _block_label(info, block["id"], block.get("label")),
                font_size=layout["font"],
            )
        else:
            geom = _plus_block(
                parts, info, shadow_id, block["id"],
                cx, top + block_h / 2, sym=layout.get("sym", "+"),
            )
        block_pos[block["id"]] = geom
        y_cursor = top - _BLOCK_GAP

    # --- 4. Linear chain arrows ---
    chain = [stack_input] + [block_pos[b["id"]] for b in chain_blocks] + [final_rms, lm_head]
    if not has_vision_fusion:
        chain.insert(0, tok_text)
    for src, dst in zip(chain, chain[1:]):
        parts.append(_v_line(src, dst, arrow_id))

    # Output arrow above lm_head.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": lm_head["top"], "x2": cx, "y2": lm_head["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    # --- 5. Residual loops (declared via residual_from) ---
    chain_ids = [b["id"] for b in chain_blocks]
    chain_prev = {block_id: chain_ids[i - 1] for i, block_id in enumerate(chain_ids[1:], start=1)}
    branch_taps: set[tuple[float, float]] = set()
    lane = inner_x + inner_w - 28
    for block in layer_blocks:
        src_id = block.get("residual_from")
        if src_id and src_id in block_pos and block["id"] in block_pos:
            if chain_prev.get(block["id"]) == src_id:
                continue
            src_geom = block_pos[src_id]
            dst_geom = block_pos[block["id"]]
            parts.append(_residual_loop_right(src_geom, dst_geom, lane, arrow_id))
            # Junction dot at the tap point on the input-arrow stem so the
            # bypass visually originates from the arrow, not from the block.
            _mark_branch_tap(parts, branch_taps, _input_tap(src_geom))

    # --- 6. Side blocks — placed off the central column ---
    for block in side_blocks:
        _draw_side_block(
            parts, info, shadow_id,
            block, block_pos,
            inner_x, inner_w, arrow_id, branch_taps,
        )

    # --- 7. × N badge over the inner region ---
    parts.append(_svg_tag("rect", {
        "x": inner_x + inner_w - 78, "y": inner_y + 12,
        "width": 66, "height": 26, "rx": 13, "ry": 13,
        "fill": "rgba(255,255,255,0.65)", "stroke": C["border"], "stroke-width": 0.5,
    }))
    parts.append(_svg_text(
        inner_x + inner_w - 45, inner_y + 25,
        f"x {len(ir.get('layers', []))}",
        {"text-anchor": "middle", "dominant-baseline": "central",
         "fill": C["text"], "font-family": FONT_HEAD, "font-size": 20},
    ))

    return _svg(w, h, f"{ir.get('name', 'model')} architecture", parts)


def _draw_multimodal_input_scaffold(
    parts: list[str],
    info: dict,
    shadow_id: str,
    arrow_id: str,
    cx: float,
    inner_y: float,
    inner_h: float,
    h: float,
) -> tuple[dict, dict, dict]:
    """Draw text + vision inputs entering a model-level fusion node."""
    fusion_y = inner_y + inner_h + 34
    embed_y = fusion_y + 88
    tok_y = embed_y + 66
    text_x = cx - 145
    vision_x = cx + 160

    tok_text = _rect_block(
        parts, info, shadow_id, "tok_text",
        text_x - 105, tok_y, 210, 42,
        _block_label(info, "tok_text", "Tokenized text"), font_size=16,
    )
    embed = _rect_block(
        parts, info, shadow_id, "embed",
        text_x - 125, embed_y, 250, 44,
        _block_label(info, "embed", "Token Embedding"), font_size=16,
    )
    vision = _rect_block(
        parts, info, shadow_id, "vision_path",
        vision_x - 105, embed_y, 210, 44,
        _block_label(info, "vision_path", "Vision pathway"), font_size=16,
    )
    fusion = _rect_block(
        parts, info, shadow_id, "fusion",
        cx - 125, fusion_y, 250, 50,
        _block_label(info, "fusion", "Multimodal fusion"), font_size=16,
    )

    parts.append(_v_line(tok_text, embed, arrow_id))
    parts.append(_block_top_to_block_bottom(
        embed["cx"], embed["top"], fusion["cx"] - 56, fusion["bottom"] + GAP, arrow_id,
    ))
    parts.append(_block_top_to_block_bottom(
        vision["cx"], vision["top"], fusion["cx"] + 56, fusion["bottom"] + GAP, arrow_id,
    ))
    return tok_text, embed, fusion


def _layer_stack_height(layer_blocks: list[dict]) -> int:
    if not layer_blocks:
        return 0
    total = sum(_KIND_LAYOUT.get(b["kind"], _KIND_LAYOUT["norm"])["h"] for b in layer_blocks)
    total += _BLOCK_GAP * (len(layer_blocks) - 1)
    return total


def _draw_side_block(
    parts: list[str],
    info: dict,
    shadow_id: str,
    block: dict,
    block_pos: dict,
    inner_x: float,
    inner_w: float,
    arrow_id: str,
    branch_taps: set[tuple[float, float]],
) -> None:
    """Render a block that lives OFF the central chain.

    The block is drawn at the y-row of whatever it ``feeds``, offset to the
    declared ``lane`` (left/right).  Its input is a long arrow tapping the
    chain at the bottom of the ``tap_from`` block; its output is a short
    horizontal arrow into the ``feeds`` target.
    """
    layout = _KIND_LAYOUT.get(block["kind"]) or _KIND_LAYOUT["norm"]
    block_w = layout["w"]
    block_h = layout["h"]
    lane = block.get("lane", "left")
    feeds_id = block.get("feeds")
    tap_id = block.get("tap_from")

    feeds_geom = block_pos.get(feeds_id) if feeds_id else None
    tap_geom = block_pos.get(tap_id) if tap_id else None
    if not feeds_geom or not tap_geom:
        return  # mis-declared; nothing to anchor to

    # ``side_align="tap"`` places the block at the same y as the tap source
    # (e.g. parallel FFN beside Attention) instead of the default feeds target.
    if block.get("side_align") == "tap":
        cy = tap_geom["cy"]
    else:
        cy = feeds_geom["cy"]

    if lane == "left":
        block_x = inner_x + 30
    else:
        block_x = inner_x + inner_w - 30 - block_w
    top = cy - block_h / 2

    geom = _rect_block(
        parts, info, shadow_id, block["id"],
        block_x, top, block_w, block_h,
        _block_label(info, block["id"], block.get("label")),
        font_size=layout["font"],
    )
    block_pos[block["id"]] = geom

    # --- Input: long arrow up the side, tapping the chain at tap_from's input
    #     stem (so the visual reads "the same x flowing into the layer also
    #     feeds this side block").  Routed as a rounded L-bend.
    rail_x = geom["cx"]
    tap_x, tap_y = _input_tap(tap_geom)
    parts.append(_elbow_hv(tap_x, tap_y, rail_x, geom["bottom"] + GAP, arrow_id))
    _mark_branch_tap(parts, branch_taps, (tap_x, tap_y))

    # --- Output: arrow into feeds target.
    #     When the side block is at a different y from its feeds target (e.g.
    #     parallel FFN at Attn level feeding into the higher add node), use an
    #     elbow so the arrow arrives squarely at the target's edge.
    feeds_cy = feeds_geom["cy"]
    if lane == "left":
        x2 = feeds_geom["left"] - GAP
    else:
        x2 = feeds_geom["right"] + GAP

    if abs(cy - feeds_cy) > 4:
        # The side block sits at a different y than its feeds target (e.g. parallel
        # FFN at Attn level feeding up into the add node).  Route from the block's
        # leading edge (top if climbing, bottom if descending) as a vertical-first
        # elbow: go straight up/down from the block center, then turn horizontally
        # into the feeds target.  This reads as "FFN output rises to meet the add".
        if cy > feeds_cy:
            y_start = geom["top"] - GAP   # arrow leaves from top of block
        else:
            y_start = geom["bottom"] + GAP  # arrow leaves from bottom of block
        parts.append(_elbow_vh(geom["cx"], y_start, x2, feeds_cy, arrow_id))
    else:
        if lane == "left":
            x1 = geom["right"]
        else:
            x1 = geom["left"]
        parts.append(_svg_tag("line", {
            "x1": x1, "y1": cy, "x2": x2, "y2": cy,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none",
        }))


def _mark_branch_tap(
    parts: list[str],
    branch_taps: set[tuple[float, float]],
    tap: tuple[float, float],
) -> None:
    key = (round(tap[0], 3), round(tap[1], 3))
    if key in branch_taps:
        return
    branch_taps.add(key)
    parts.append(_branch_dot(*tap))


def _build_layer_map(ir: dict, info: dict, mount_id: str) -> str:
    w = 720
    layers = ir.get("layers", [])
    kv_shared_indices = [
        i for i, layer in enumerate(layers)
        if (layer.get("attention") or {}).get("kv_source_layer") is not None
    ]
    has_kv_share = bool(kv_shared_indices)
    n_legend_rows = len(info["groups"]) + (1 if has_kv_share else 0)
    # Reserve extra room for the optional "KV CACHE" sub-strip and its annotation.
    extra = 56 if has_kv_share else 0
    h = max(240, 160 + extra + 22 * n_legend_rows)
    arrow_id, shadow_id = _ids(mount_id, "map")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_hatch_pattern(mount_id))
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_card"], stroke=C["border"], stroke_width=0.5))

    # Green-family palette so the layer map shares the diagram's theme.
    # Ordered dark → light so consecutive groups read like a gradient step.
    palette = ["#0F6E56", "#1F9E78", "#5BB89A", "#0A4F3F", "#7FCFB4", "#0E5C48", "#A0E3CD"]
    sig_to_color = {group["sig"]: palette[i % len(palette)] for i, group in enumerate(info["groups"])}

    strip_x, strip_y, strip_w, strip_h = 80, 90, w - 160, 36
    n = len(layers)
    col_w = strip_w / max(n, 1)

    layer_sigs = info.get("layer_sigs") or [_signature(layer) for layer in layers]
    for i, sig in enumerate(layer_sigs):
        parts.append(
            _svg_tag(
                "rect",
                {
                    "x": strip_x + i * col_w,
                    "y": strip_y,
                    "width": max(col_w - 0.5, 1),
                    "height": strip_h,
                    "fill": sig_to_color.get(sig, palette[0]),
                    "opacity": 0.95,
                },
            )
        )

    # KV-share overlay — diagonal hatch on layers that don't compute their own K/V.
    for i in kv_shared_indices:
        parts.append(
            _svg_tag(
                "rect",
                {
                    "x": strip_x + i * col_w,
                    "y": strip_y,
                    "width": max(col_w - 0.5, 1),
                    "height": strip_h,
                    "fill": f"url(#uf-{mount_id}-hatch)",
                    "pointer-events": "none",
                },
            )
        )

    parts.append(
        _svg_tag(
            "rect",
            {
                "x": strip_x,
                "y": strip_y,
                "width": strip_w,
                "height": strip_h,
                "fill": "none",
                "stroke": C["text"],
                "stroke-width": 0.4,
                "rx": 4,
                "ry": 4,
            },
        )
    )

    if n:
        for idx in (0, n - 1):
            x = strip_x + (idx + 0.5) * col_w
            parts.append(
                _svg_text(
                    x,
                    strip_y + strip_h + 16,
                    f"L{idx}",
                    {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
                )
            )

    type_word = "type" if len(info["groups"]) == 1 else "types"
    parts.append(
        _svg_text(
            strip_x,
            70,
            f"{n} layers - {len(info['groups'])} {type_word}",
            {"fill": C["text"], "font-family": FONT_BODY, "font-size": 12, "font-weight": 600},
        )
    )

    legend_y = strip_y + strip_h + 44

    if has_kv_share:
        first = kv_shared_indices[0]
        last = kv_shared_indices[-1]
        # Bracket above the strip marking where KV reuse kicks in.
        bracket_y = strip_y - 8
        x_start = strip_x + first * col_w
        x_end = strip_x + (last + 1) * col_w - 0.5
        parts.append(
            _svg_tag(
                "path",
                {
                    "d": f"M {x_start} {bracket_y - 6} L {x_start} {bracket_y} L {x_end} {bracket_y} L {x_end} {bracket_y - 6}",
                    "fill": "none",
                    "stroke": C["muted"],
                    "stroke-width": 1.0,
                    "stroke-linecap": "round",
                },
            )
        )
        # Sources of the K/V tensors — collected from cross-layer edges.
        edges = ir.get("cross_layer_edges") or []
        kv_sources = sorted({e.get("from_layer") for e in edges if e.get("kind") == "kv_share"})
        src_summary = (
            f"L{kv_sources[0]}–L{kv_sources[-1]}" if len(kv_sources) > 1
            else (f"L{kv_sources[0]}" if kv_sources else "earlier layer")
        )
        share_label = (
            f"K/V reused: L{first}–L{last} ({len(kv_shared_indices)} layers)  ←  {src_summary}"
        )
        parts.append(
            _svg_text(
                (x_start + x_end) / 2,
                bracket_y - 12,
                share_label,
                {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
            )
        )
        legend_y += 8

    lx, ly = strip_x, legend_y
    for group in info["groups"]:
        spec = group["spec"]
        ffn_kind = "MoE" if spec["ffn"].get("kind") == "moe" else "Dense"
        attn = spec.get("attention", {})
        label = (
            f"{kind_short(attn)} + {ffn_kind} ({mask_short(attn)})"
            f"  ·  {_indices_summary(group, info)}"
        )
        color = sig_to_color[group["sig"]]
        parts.append(_svg_tag("rect", {"x": lx, "y": ly - 9, "width": 12, "height": 12, "fill": color, "rx": 2}))
        parts.append(
            _svg_text(
                lx + 18,
                ly,
                label,
                {"dominant-baseline": "central", "fill": C["text"], "font-family": FONT_BODY, "font-size": 12},
            )
        )
        ly += 20

    if has_kv_share:
        # Hatched chip in the legend.
        parts.append(
            _svg_tag(
                "rect",
                {"x": lx, "y": ly - 9, "width": 12, "height": 12, "fill": palette[0], "rx": 2},
            )
        )
        parts.append(
            _svg_tag(
                "rect",
                {
                    "x": lx,
                    "y": ly - 9,
                    "width": 12,
                    "height": 12,
                    "fill": f"url(#uf-{mount_id}-hatch)",
                    "rx": 2,
                },
            )
        )
        parts.append(
            _svg_text(
                lx + 18,
                ly,
                f"K/V reused (no own K/V projections)  ·  {len(kv_shared_indices)} layers",
                {"dominant-baseline": "central", "fill": C["text"], "font-family": FONT_BODY, "font-size": 12},
            )
        )

    return _svg(w, h, f"{ir.get('name', 'model')} layer map", parts)


def _hatch_pattern(mount_id: str) -> str:
    """Diagonal-stripe pattern used to mark KV-shared layers."""
    pid = f"uf-{mount_id}-hatch"
    return (
        '<defs>'
        f'<pattern id="{pid}" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">'
        '<rect width="6" height="6" fill="none"/>'
        '<line x1="0" y1="0" x2="0" y2="6" stroke="rgba(255,255,255,0.55)" stroke-width="2"/>'
        '</pattern>'
        '</defs>'
    )
