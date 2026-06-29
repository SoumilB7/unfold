"""Top-level SVG views for architecture and layer maps."""
from __future__ import annotations

from ...block_schema import DIFFUSION_BLOCK_IDS, DIFFUSION_STAGES
from ...labels import kind_short, mask_short
from .metadata import _block_label, _indices_summary, _signature
from .svg import (
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
from .views_modalities import draw_multimodal_input_scaffold


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
    "concat":       {"shape": "circle", "w": 30,  "h": 30, "sym": "‖"},
}
_BLOCK_GAP = 32  # vertical gap between consecutive layer-body blocks
# Larger than the arrow padding (`GAP` ×2) so the chain arrow has a visible
# stem between blocks rather than collapsing to just an arrowhead.

def _is_diffusion_architecture(ir: dict) -> bool:
    return ((ir.get("extras") or {}).get("render") or {}).get("family") == "diffusion"


def _is_resolved_diffusion_block(is_diffusion: bool, info: dict, node_id: str, block: dict | None = None) -> bool:
    """Only approved diffusion slots render as solid architecture blocks.

    Unknown diffusion nodes are still drawn and clickable, but pale, so a new
    adapter fact cannot quietly become a first-class block until we bless its
    stage or slot here.
    """
    if not is_diffusion:
        return True
    data = dict(info.get("blocks", {}).get(node_id, {}))
    if block:
        data.update(block)
    stage = data.get("diffusion_stage")
    if stage is not None:
        return stage in DIFFUSION_STAGES
    return node_id in DIFFUSION_BLOCK_IDS


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
    is_diffusion = _is_diffusion_architecture(ir)
    spec = info["dominant"]["spec"]
    layer_blocks = list(spec.get("blocks") or [])
    group_indices = (info.get("dominant") or {}).get("indices")
    repeat_n = len(group_indices) if group_indices else len(ir.get("layers", []))
    repeated_region = repeat_n != 1

    # Side blocks live OFF the central column.  They share a row with the block
    # they feed but get their own offset x-position and explicit connections.
    # Branch blocks (``branch_side``) are a symmetric parallel split: two equal
    # branches that fan out from one chain block and converge into a ⊕ merge.
    chain_blocks = [b for b in layer_blocks if not b.get("lane") and not b.get("branch_side")]
    side_blocks = [b for b in layer_blocks if b.get("lane")]
    branch_blocks = [b for b in layer_blocks if b.get("branch_side")]

    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    position_evidence = ((ir.get("extras") or {}).get("position_encoding") or {})
    position_mechanisms = position_evidence.get("mechanisms") or [] \
        if isinstance(position_evidence, dict) else []
    has_absolute_position = any(
        item.get("kind") in {"learned_absolute", "fixed_absolute"}
        and item.get("application") == "embedding_add"
        for item in position_mechanisms if isinstance(item, dict)
    )
    modality_inputs = modalities.get("inputs") or {}
    fusion_spec = modalities.get("fusion") or {}
    has_modality_fusion = bool(modality_inputs) and bool(fusion_spec)
    has_cross_attention_fusion = has_modality_fusion and fusion_spec.get("kind") == "cross_attention"
    has_external_side_stream = any(str(block.get("lane", "")).startswith("external") for block in side_blocks)

    modality_count = len(modality_inputs)
    has_wide_modality_scaffold = has_modality_fusion and modality_count >= 3
    needs_wide_arch = has_external_side_stream or has_wide_modality_scaffold
    inner_x, inner_w = (230, 500) if needs_wide_arch else (110, 500)

    # Default chain center.  Auto-shift right when a side_align="tap" block on
    # the left lane would overlap the widest chain block at the default cx.
    # This handles parallel-residual architectures (e.g. GPT-NeoX / GPT-J) where
    # FFN and Attention share the same y-row without any renderer special-casing.
    cx = 480 if needs_wide_arch else 360
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
    # Size the shaded region to the content (chain height + padding).  The floor
    # only matters for short stacks (parallel-residual layers — e.g. GPT-NeoX, the
    # DiT single-stream block — have ~3 chain blocks); keep it modest so the box
    # hugs the content instead of leaving a large empty lower half.  Sequential
    # decoder layers (~6 blocks) are content-driven and unaffected.
    inner_padding = 60
    # A parallel branch split needs a reserved row (between its split source and
    # its ⊕ merge) tall enough for the branches; the merge carries that as extra
    # gap below it, so it counts toward both the stack height and the y-cursor.
    merge_id = branch_blocks[0].get("feeds") if branch_blocks else None
    branch_row_h = 0
    if branch_blocks:
        branch_h = max(
            b.get("h") or _KIND_LAYOUT.get(b["kind"], _KIND_LAYOUT["norm"])["h"]
            for b in branch_blocks
        )
        branch_row_h = branch_h + 2 * _BLOCK_GAP
    stack_h = _layer_stack_height(chain_blocks) + branch_row_h
    inner_h = max(340, stack_h + 2 * inner_padding)

    # Multi-Token Prediction heads draw as a stack above lm_head; reserve top
    # headroom by pushing the fixed top anchors (and total height) down.
    mtp = (ir.get("extras") or {}).get("mtp")
    mtp_pad = 108 if mtp else 0

    inner_y = 200 + mtp_pad
    has_audio_fusion = has_modality_fusion and "audio" in modality_inputs
    position_pad = 56 if has_absolute_position and not has_modality_fusion else 0
    if has_modality_fusion and not has_cross_attention_fusion:
        h = inner_y + inner_h + (360 if has_audio_fusion else 292)
    else:
        h = inner_y + inner_h + 232 + position_pad
    w = 960 if needs_wide_arch else 720

    arrow_id, shadow_id = _ids(mount_id, "arch")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 26, w - 80, h - 52, C["bg_outer"]))
    if repeated_region:
        parts.append(_region_rect(inner_x, inner_y, inner_w, inner_h, C["bg_inner"]))

    # --- 2. Model-level scaffold (positions tracked by total height h) ---
    if has_modality_fusion:
        tok_text, embed, stack_input = draw_multimodal_input_scaffold(
            parts, info, shadow_id, arrow_id, cx, inner_y, inner_h, h, modalities,
        )
    elif has_absolute_position:
        tok_text = _rect_block(parts, info, shadow_id, "tok_text",
                               cx - 280, h - 100, 220, 44,
                               _block_label(info, "tok_text", "Tokenized text"), font_size=17,
                               resolved=True)
        position_ids = _rect_block(parts, info, shadow_id, "position_ids",
                                   cx + 60, h - 100, 220, 44,
                                   _block_label(info, "position_ids", "Position IDs"), font_size=17,
                                   resolved=True)
        embed = _rect_block(parts, info, shadow_id, "embed",
                            cx - 300, h - 174, 260, 44,
                            _block_label(info, "embed", "Token Embedding layer"), font_size=17,
                            resolved=True)
        position_embed = _rect_block(parts, info, shadow_id, "position_embed",
                                     cx + 40, h - 174, 260, 44,
                                     _block_label(info, "position_embed", "Learned Position Embedding"),
                                     font_size=15, resolved=True)
        position_add = _plus_block(parts, info, shadow_id, "position_add",
                                   cx, h - 230, sym="+", clickable=True)
        parts.append(_v_line(tok_text, embed, arrow_id))
        parts.append(_v_line(position_ids, position_embed, arrow_id))
        parts.append(_elbow_vh(embed["cx"], embed["top"],
                               position_add["left"] - GAP, position_add["cy"], arrow_id))
        parts.append(_elbow_vh(position_embed["cx"], position_embed["top"],
                               position_add["right"] + GAP, position_add["cy"], arrow_id))
        stack_input = position_add
    else:
        tok_text = _rect_block(parts, info, shadow_id, "tok_text",
                               cx - 110, h - 100, 220, 44,
                               _block_label(info, "tok_text", "Tokenized text"), font_size=17,
                               resolved=_is_resolved_diffusion_block(is_diffusion, info, "tok_text"))
        embed = _rect_block(parts, info, shadow_id, "embed",
                            cx - 130, h - 168, 260, 44,
                            _block_label(info, "embed", "Token Embedding layer"), font_size=17,
                            resolved=_is_resolved_diffusion_block(is_diffusion, info, "embed"))
        stack_input = embed
    final_rms = _rect_block(parts, info, shadow_id, "final_rms",
                            cx - 90, 140 + mtp_pad, 180, 36,
                            _block_label(info, "final_rms", "Final RMSNorm"), font_size=16,
                            resolved=_is_resolved_diffusion_block(is_diffusion, info, "final_rms"))
    lm_head = _rect_block(parts, info, shadow_id, "lm_head",
                          cx - 130, 70 + mtp_pad, 260, 44,
                          _block_label(info, "lm_head", "Linear output layer"), font_size=17,
                          resolved=_is_resolved_diffusion_block(is_diffusion, info, "lm_head"))

    # --- 3. Layer body (data-driven, stacked bottom-up) ---
    block_pos: dict[str, dict] = {}
    free = inner_h - stack_h
    y_cursor = inner_y + inner_h - free / 2
    for block in chain_blocks:
        layout = _KIND_LAYOUT.get(block["kind"]) or _KIND_LAYOUT["norm"]
        block_w = block.get("w") or layout["w"]
        block_h = block.get("h") or layout["h"]
        font_size = block.get("font") or layout.get("font", 16)
        # Tier-2 connectors (residual ⊕, gate ×) are drawn as glyphs on the
        # topology, not first-class blocks: `static` makes them non-clickable
        # with no card.  The block-tier paradigm lives in the adapter (which
        # tags the block); the engine just honours the flag.
        clickable = not block.get("static")
        # Reserve the parallel-branch row below the ⊕ merge (between it and the
        # split source just under it), so the two branches have a clear band.
        if branch_blocks and block["id"] == merge_id:
            y_cursor -= branch_row_h
        top = y_cursor - block_h
        if layout["shape"] == "rect":
            geom = _rect_block(
                parts, info, shadow_id, block["id"],
                cx - block_w / 2, top, block_w, block_h,
                _block_label(info, block["id"], block.get("label")),
                font_size=font_size,
                resolved=_is_resolved_diffusion_block(is_diffusion, info, block["id"], block),
                clickable=clickable,
            )
        else:
            geom = _plus_block(
                parts, info, shadow_id, block["id"],
                cx, top + block_h / 2, sym=layout.get("sym", "+"),
                clickable=clickable,
            )
        block_pos[block["id"]] = geom
        y_cursor = top - _BLOCK_GAP

    # --- 4. Linear chain arrows ---
    chain = [stack_input] + [block_pos[b["id"]] for b in chain_blocks] + [final_rms, lm_head]
    if not has_modality_fusion and not has_absolute_position:
        chain.insert(0, tok_text)
    merge_geom = block_pos.get(merge_id) if merge_id else None
    for src, dst in zip(chain, chain[1:]):
        # The split source → ⊕ merge segment is drawn by the branch fan-out/merge
        # routing below (the flow goes THROUGH the two branches), not a direct line.
        if merge_geom is not None and dst is merge_geom:
            continue
        parts.append(_v_line(src, dst, arrow_id))

    # --- 4b. Parallel branch split: source fans out to side-by-side branches
    #     that converge into the ⊕ merge (DiffusionGemma's dense MLP ∥ MoE). ---
    if branch_blocks and merge_geom is not None:
        _draw_branch_split(
            parts, info, shadow_id, arrow_id, cx,
            chain_blocks, merge_id, merge_geom, branch_blocks,
            is_diffusion, block_pos=block_pos,
        )

    # Output arrow above lm_head — or the MTP head stack when present.
    if mtp:
        _draw_mtp_head(parts, info, shadow_id, arrow_id, lm_head, mtp)
    else:
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
    # Bottom-origin side blocks sit on the input row (aligned with the embed/
    # Patchify block) so the inputs read as one tier.
    input_cy = embed["cy"] if isinstance(embed, dict) else (inner_y + inner_h + 86)
    for block in side_blocks:
        _draw_side_block(
            parts, info, shadow_id,
            block, block_pos,
            inner_x, inner_w, inner_y, inner_h, input_cy, stack_input, arrow_id, branch_taps,
            is_diffusion,
        )

    # --- 7. × N badge over the inner region ---
    # The badge counts how many times THIS layer body repeats. For a
    # heterogeneous model the view is rendered once per layer-type variant
    # (DeepSeek-V3: 3 dense + 58 MoE), so the badge must reflect the displayed
    # group's layer count — not the global total — to stay consistent with its
    # own toggle pill ("L3–L60 · 58×"). Falls back to the total when no group
    # indices are available (single homogeneous stack ⇒ identical anyway).
    if repeated_region:
        parts.append(_svg_tag("rect", {
            "x": inner_x + inner_w - 78, "y": inner_y + 12,
            "width": 66, "height": 26, "rx": 13, "ry": 13,
            "fill": "rgba(255,255,255,0.65)", "stroke": C["border"], "stroke-width": 0.5,
        }))
        parts.append(_svg_text(
            inner_x + inner_w - 45, inner_y + 25,
            f"x {repeat_n}",
            {"text-anchor": "middle", "dominant-baseline": "central",
             "fill": C["text"], "font-family": FONT_HEAD, "font-size": 20},
        ))
    # Optional caption under the × N badge — clarifies what the repeat means when
    # one stack plays several roles (e.g. a shared encoder/decoder), right-aligned
    # to the badge so it reads as a footnote to the repeat count.
    repeat_note = ((ir.get("extras") or {}).get("render") or {}).get("repeat_note")
    if repeat_note and repeated_region:
        note_y = inner_y + 50
        for line in (repeat_note if isinstance(repeat_note, list) else [repeat_note]):
            parts.append(_svg_text(
                inner_x + inner_w - 12, note_y, line,
                {"text-anchor": "end", "fill": C["muted"],
                 "font-family": FONT_MONO, "font-size": 10.5},
            ))
            note_y += 14

    # --- 7b. Layer-level annotations (Tier-3 properties of the layer) ---
    # A property that applies to the whole layer but isn't its own computation
    # (e.g. a learned output scalar) is a small caption inside the frame — never
    # a box.  Adapters declare them in render.layer_annotations; the engine just
    # places them top-left, opposite the × N badge.
    annotations = ((ir.get("extras") or {}).get("render") or {}).get("layer_annotations") or []
    ann_y = inner_y + 13
    ann_font = 11.5
    for ann in annotations:
        # Pill must comfortably fit the mono text across fonts, so size it from a
        # generous per-char width (≈0.62·font) + padding — never let text overflow.
        chip_w = 24 + len(ann) * ann_font * 0.62
        parts.append(_svg_tag("rect", {
            "x": inner_x + 14, "y": ann_y,
            "width": chip_w, "height": 24, "rx": 12, "ry": 12,
            "fill": "rgba(255,255,255,0.55)", "stroke": C["border"], "stroke-width": 0.5,
        }))
        parts.append(_svg_text(
            inner_x + 14 + chip_w / 2, ann_y + 12, ann,
            {"text-anchor": "middle", "dominant-baseline": "central",
             "fill": C["muted"], "font-family": FONT_MONO, "font-size": ann_font},
        ))
        ann_y += 30

    # --- 7c. Per-variant stack caption (Tier-3) ---
    # A property of THIS block-type's stack, not the whole model — e.g. Flux's
    # single-stream stack joins text+image into one sequence once before the
    # stack.  Read off the dominant group's attention variant (not the global
    # render extras), so it shows on the single-stream variant ONLY, never the
    # dual.  Anchored bottom-left (the input side — it reads as "before the
    # stack"), short lines kept clear of the centre spine and the × N badge.
    stack_note = ((spec.get("attention") or {}).get("variant") or {}).get("stack_note")
    if stack_note:
        note_lines = stack_note if isinstance(stack_note, list) else [stack_note]
        note_y = inner_y + inner_h - 14 * len(note_lines) - 6
        for line in note_lines:
            parts.append(_svg_text(
                inner_x + 16, note_y, line,
                {"text-anchor": "start", "fill": C["muted"],
                 "font-family": FONT_MONO, "font-size": 11},
            ))
            note_y += 15

    return _svg(w, h, f"{ir.get('name', 'model')} architecture", parts)


def _draw_branch_split(
    parts: list[str],
    info: dict,
    shadow_id: str,
    arrow_id: str,
    cx: float,
    chain_blocks: list[dict],
    merge_id: str,
    merge_geom: dict,
    branch_blocks: list[dict],
    is_diffusion: bool,
    *,
    block_pos: dict,
) -> None:
    """A symmetric parallel split: the chain block below the ⊕ merge fans out to
    two side-by-side branches that converge back into the merge.

    The split source is the merge's chain predecessor (e.g. ``rms2``).  One
    split dot on its output stem feeds both branch bottoms; each branch top rises
    into the merge.  Branches sit symmetrically left/right of the spine in the
    band reserved between the source and the merge.
    """
    chain_ids = [b["id"] for b in chain_blocks]
    src_idx = chain_ids.index(merge_id) - 1
    src_geom = block_pos[chain_ids[src_idx]]

    row_cy = (src_geom["top"] + merge_geom["bottom"]) / 2
    split_y = src_geom["top"] - GAP                  # split dot sits on the source's output stem
    left = [b for b in branch_blocks if b.get("branch_side") == "left"]
    right = [b for b in branch_blocks if b.get("branch_side") != "left"]
    ordered = left + right
    n = len(ordered)
    # Centre the branch row on the spine; widest branch sets the column pitch.
    col_w = max(b.get("w") or _KIND_LAYOUT.get(b["kind"], _KIND_LAYOUT["norm"])["w"] for b in ordered)
    pitch = col_w + 44
    start_x = cx - pitch * (n - 1) / 2

    # Source output stem up to the split dot, then the dot.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": src_geom["top"], "x2": cx, "y2": split_y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round", "fill": "none",
    }))
    parts.append(_branch_dot(cx, split_y))

    branch_geoms: list[dict] = []
    for i, block in enumerate(ordered):
        b_w = block.get("w") or _KIND_LAYOUT.get(block["kind"], _KIND_LAYOUT["norm"])["w"]
        b_h = block.get("h") or _KIND_LAYOUT.get(block["kind"], _KIND_LAYOUT["norm"])["h"]
        b_cx = start_x + i * pitch
        geom = _rect_block(
            parts, info, shadow_id, block["id"],
            b_cx - b_w / 2, row_cy - b_h / 2, b_w, b_h,
            _block_label(info, block["id"], block.get("label")),
            font_size=block.get("font") or _KIND_LAYOUT.get(block["kind"], _KIND_LAYOUT["norm"]).get("font", 16),
            resolved=_is_resolved_diffusion_block(is_diffusion, info, block["id"], block),
        )
        block_pos[block["id"]] = geom
        branch_geoms.append(geom)
        # Fan out: split dot → horizontal → up into the branch bottom (arrowhead up).
        parts.append(_elbow_hv(cx, split_y, geom["cx"], geom["bottom"] + GAP, arrow_id))

    # Converge (mirror of the fan-out): each branch top rises and turns in to a
    # shared rail just below the ⊕; one arrow then enters the merge.  The risers
    # keep every branch CONNECTED to the merge — no floating rail.
    r = 8
    rail_y = merge_geom["bottom"] + 18
    for geom in branch_geoms:
        sx = 1 if merge_geom["cx"] >= geom["cx"] else -1
        parts.append(_svg_tag("path", {
            "d": (
                f"M {geom['cx']} {geom['top']} "
                f"L {geom['cx']} {rail_y + r} "
                f"Q {geom['cx']} {rail_y} {geom['cx'] + sx * r} {rail_y} "
                f"L {merge_geom['cx']} {rail_y}"
            ),
            "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
            "stroke-linecap": "round", "stroke-linejoin": "round",
        }))
    parts.append(_svg_tag("line", {
        "x1": merge_geom["cx"], "y1": rail_y,
        "x2": merge_geom["cx"], "y2": merge_geom["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))


def _draw_mtp_head(
    parts: list[str],
    info: dict,
    shadow_id: str,
    arrow_id: str,
    lm_head: dict,
    mtp: dict,
) -> None:
    """Draw the Multi-Token Prediction head as a small stacked-card glyph above
    lm_head, fed from the shared trunk output and emitting the final logits."""
    n = mtp.get("num_modules") or 1
    cx = lm_head["cx"]
    w, h = 224, 46
    bottom = lm_head["top"] - 38
    top = bottom - h

    # Decorative offset cards behind the front one imply a stack of N modules.
    for off in (12, 6):
        parts.append(_svg_tag("rect", {
            "x": cx - w / 2 + off, "y": top - off, "width": w, "height": h,
            "rx": 11, "ry": 11, "fill": C["block"], "opacity": 0.45,
            "stroke": C["block_alt"], "stroke-width": 0.6,
        }))

    label = _block_label(info, "mtp", f"MTP head x{n}" if n > 1 else "MTP head")
    geom = _rect_block(parts, info, shadow_id, "mtp", cx - w / 2, top, w, h, label, font_size=15)

    # Shared trunk output -> MTP, then MTP -> logits.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": lm_head["top"], "x2": cx, "y2": geom["bottom"] + 4,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": geom["top"], "x2": cx, "y2": geom["top"] - 30,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        geom["right"] + 12, geom["cy"],
        f"+{n} future token{'s' if n != 1 else ''}",
        {"dominant-baseline": "central", "fill": C["muted"],
         "font-family": FONT_MONO, "font-size": 10},
    ))


def _layer_stack_height(layer_blocks: list[dict]) -> int:
    if not layer_blocks:
        return 0
    total = sum(
        b.get("h") or _KIND_LAYOUT.get(b["kind"], _KIND_LAYOUT["norm"])["h"]
        for b in layer_blocks
    )
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
    inner_y: float,
    inner_h: float,
    input_cy: float,
    input_geom: dict,
    arrow_id: str,
    branch_taps: set[tuple[float, float]],
    is_diffusion: bool = False,
) -> None:
    """Render a block that lives OFF the central chain.

    The block is drawn at the y-row of whatever it ``feeds``, offset to the
    declared ``lane`` (left/right).  Its input is a long arrow tapping the
    chain at the bottom of the ``tap_from`` block; its output is a short
    horizontal arrow into the ``feeds`` target.
    """
    layout = _KIND_LAYOUT.get(block["kind"]) or _KIND_LAYOUT["norm"]
    block_w = block.get("w") or layout["w"]
    block_h = block.get("h") or layout["h"]
    font_size = block.get("font") or layout.get("font", 16)
    lane = block.get("lane", "left")
    feeds_id = block.get("feeds")
    tap_id = block.get("tap_from")

    feeds_geom = block_pos.get(feeds_id) if feeds_id else None
    tap_geom = block_pos.get(tap_id) if tap_id else None
    # Bottom-origin side block (e.g. diffusion conditioning): drawn at the bottom
    # like the other inputs, with the arrow bending up into its target.
    if feeds_geom and str(lane).startswith("external_bottom"):
        _draw_bottom_side_block(
            parts, info, shadow_id, block, feeds_geom,
            inner_x, inner_w, input_cy, input_geom, arrow_id,
            block_w, block_h, font_size, is_diffusion, block_pos,
        )
        return
    if feeds_geom and str(block.get("lane", "")).startswith("external"):
        _draw_external_side_block(
            parts, info, shadow_id, block, feeds_geom,
            inner_x, inner_w, arrow_id, block_pos,
            block_w, block_h, font_size, is_diffusion,
        )
        return
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
        font_size=font_size,
        resolved=_is_resolved_diffusion_block(is_diffusion, info, block["id"], block),
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


def _draw_bottom_side_block(
    parts: list[str],
    info: dict,
    shadow_id: str,
    block: dict,
    feeds_geom: dict,
    inner_x: float,
    inner_w: float,
    input_cy: float,
    input_geom: dict,
    arrow_id: str,
    block_w: float,
    block_h: float,
    font_size: int,
    is_diffusion: bool = False,
    block_pos: dict | None = None,
) -> None:
    """A side input drawn at the BOTTOM (aligned with the main input row), routed
    up the outside of the inner region and bent into the target's near edge.

    Used for conditioning rails (timestep -> AdaLN, text -> attention): they read
    as inputs entering from below, not as states floating in from the sides.
    ``also_feeds`` targets (e.g. the AdaLN gate × nodes this conditioning drives)
    each get their own elbow on a staggered left lane, so a gate × visibly shows
    the conditioning it multiplies by instead of a dangling input.
    """
    lane = str(block.get("lane", "external_bottom_left"))
    left = lane.endswith("left")
    block_y = input_cy - block_h / 2
    # Wide DiT views reserve exactly one side lane on each side of the inner
    # stack.  A pure "outside the inner region" placement leaves 190px-wide
    # conditioning blocks kissing or exceeding the outer frame, so allow a small
    # overlap into the inner tint while preserving a visible margin to the card.
    outer_left = 40
    outer_right = inner_x + inner_w + (inner_x - outer_left)
    side_overlap = 12
    side_margin = 52
    clear_main = 22
    if left:
        block_x = max(outer_left + side_margin, inner_x - block_w + side_overlap)
        target_x = feeds_geom["left"] - GAP
    else:
        right_margin = 18
        ideal_x = max(inner_x + inner_w - side_overlap, input_geom["right"] + clear_main)
        block_x = min(ideal_x, outer_right - block_w - right_margin)
        target_x = feeds_geom["right"] + GAP

    geom = _rect_block(
        parts, info, shadow_id, block["id"],
        block_x, block_y, block_w, block_h,
        _block_label(info, block["id"], block.get("label")),
        font_size=font_size,
        resolved=_is_resolved_diffusion_block(is_diffusion, info, block["id"], block),
    )
    # Up the outside, then a single bend horizontally into the target edge.
    parts.append(_elbow_vh(geom["cx"], geom["top"], target_x, feeds_geom["cy"], arrow_id))
    # Fan into the gate × nodes this conditioning drives (AdaLN gate_msa/gate_mlp):
    # a shared trunk up the block's left, bending into each gate's left edge at the
    # gate's height (mirror of the right-side residual loops) — so each × shows the
    # timestep gate entering it instead of a dangling input.
    also = [t for t in (block.get("also_feeds") or []) if block_pos and t in block_pos]
    for tid in also:
        g = block_pos[tid]
        parts.append(_elbow_vh(geom["cx"], geom["top"], g["left"] - GAP, g["cy"], arrow_id))


def _draw_external_side_block(
    parts: list[str],
    info: dict,
    shadow_id: str,
    block: dict,
    feeds_geom: dict,
    inner_x: float,
    _inner_w: float,
    arrow_id: str,
    block_pos: dict,
    block_w: float,
    block_h: float,
    font_size: int,
    is_diffusion: bool = False,
) -> None:
    """Draw a layer-local side stream, e.g. vision states into cross-attention."""
    lane = block.get("lane", "external_left")
    if lane.endswith("left"):
        block_x = max(56, inner_x - block_w - 34)
        target_x = feeds_geom["left"] - GAP
    else:
        block_x = inner_x + _inner_w + 34
        target_x = feeds_geom["right"] + GAP

    cy = feeds_geom["cy"] + float(block.get("offset_y", 28))
    top = cy - block_h / 2
    geom = _rect_block(
        parts, info, shadow_id, block["id"],
        block_x, top, block_w, block_h,
        _block_label(info, block["id"], block.get("label")),
        font_size=font_size,
        resolved=_is_resolved_diffusion_block(is_diffusion, info, block["id"], block),
    )
    block_pos[block["id"]] = geom
    if lane.endswith("left"):
        route_x = (geom["right"] + target_x) / 2 if geom["right"] < target_x else target_x - 44
    else:
        route_x = (geom["left"] + target_x) / 2 if geom["left"] > target_x else target_x + 44

    source_id = block.get("source_id")
    if source_id:
        source_w = block.get("source_w") or 230
        source_h = block.get("source_h") or 46
        source_gap = block.get("source_gap") or 56
        source_x = geom["cx"] - source_w / 2
        source_top = geom["bottom"] + source_gap
        source = _rect_block(
            parts, info, shadow_id, source_id,
            source_x, source_top, source_w, source_h,
            _block_label(info, source_id, block.get("source_label", source_id)),
            font_size=font_size,
            resolved=_is_resolved_diffusion_block(is_diffusion, info, source_id, {"id": source_id}),
        )
        block_pos[source_id] = source
        parts.append(_v_line(source, geom, arrow_id))

    # Route out with a visible 90-degree turn so the adapter reads as an
    # external conditioning path, not another central-chain block.
    x_start = geom["right"] if lane.endswith("left") else geom["left"]
    parts.append(_svg_tag("path", {
        "d": (
            f"M {x_start} {geom['cy']} "
            f"L {route_x} {geom['cy']} "
            f"L {route_x} {feeds_geom['cy']} "
            f"L {target_x} {feeds_geom['cy']}"
        ),
        "stroke": C["arrow"],
        "stroke-width": 1.6,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})",
        "fill": "none",
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
    # Legend rows start below the strip (strip_y 90 + strip_h 36 + 44 = 170) and
    # step 20px each (kv-share shifts them +8).  Size the card so the last row and
    # its text descender stay inside the rounded border (bottom edge is h - 30).
    legend_bottom = 170 + 20 * max(n_legend_rows - 1, 0) + (8 if has_kv_share else 0)
    h = max(240, legend_bottom + 50)
    arrow_id, shadow_id = _ids(mount_id, "map")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_hatch_pattern(mount_id))
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_card"], stroke=C["border"], stroke_width=0.5))

    # Theme gradient (dark → light) so the layer map shares the diagram's colour
    # family — teal for transformers, blue for diffusion. Driven by the active
    # palette, not hardcoded, so a new theme recolours the map for free.
    palette = C.get("map_palette") or [C["block"], C["block_alt"]]
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
