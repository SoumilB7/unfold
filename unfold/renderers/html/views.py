"""SVG views for architecture, FFN, MoE, and layer maps."""
from __future__ import annotations

from ...labels import kind_short, mask_short
from .metadata import _block_label, _indices_summary, _signature
from .svg import (
    _branch_dot,
    _defs,
    _elbow_hv,
    _elbow_vh,
    _ids,
    _path,
    _plus_block,
    _rect_block,
    _region_rect,
    _residual_loop_right,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
    _v_seg,
)
from .theme import C, FONT_BODY, FONT_HEAD, FONT_MONO, GAP
from .utils import _fmt_int


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
    "residual_add": {"shape": "circle", "w": 28,  "h": 28, "sym": "+"},
    "gate_mul":     {"shape": "circle", "w": 28,  "h": 28, "sym": "×"},
}
_BLOCK_GAP = 28  # vertical gap between consecutive layer-body blocks
# Larger than the arrow padding (`GAP` ×2) so the chain arrow has a visible
# stem between blocks rather than collapsing to just an arrowhead.


def _build_architecture_view(ir: dict, info: dict, mount_id: str) -> str:
    """Data-driven decoder architecture view.

    The layer body comes from ``info['dominant']['spec']['blocks']`` — an
    ordered list of typed blocks emitted by the adapter.  Each block's
    ``kind`` selects a glyph in :data:`_KIND_LAYOUT`; ``residual_from`` adds
    a residual loop on the right-hand lane; ``external_from`` ties the block
    to an entry in ``ir.extras['external_pathways']`` and renders a parallel
    rail to the left of the inner region.

    Standard 6-block llama models render exactly as before; PLE-bearing
    Gemma 4 models extend the inner region downward to fit 12 blocks plus
    a parallel construction inset above the embed.
    """
    spec = info["dominant"]["spec"]
    layer_blocks = list(spec.get("blocks") or [])

    # Side blocks (e.g. PLE) live OFF the central column.  They share a row
    # with the block they feed but get their own offset x-position and
    # explicit input/output connections.
    chain_blocks = [b for b in layer_blocks if not b.get("lane")]
    side_blocks = [b for b in layer_blocks if b.get("lane")]

    cx = 360
    inner_x, inner_w = 110, 500

    # --- 1. Compute heights from the chain block list ---
    inner_padding = 60
    stack_h = _layer_stack_height(chain_blocks)
    inner_h = max(490, stack_h + 2 * inner_padding)

    inner_y = 200
    h = inner_y + inner_h + 232  # 232 = embed + tok_text + bottom padding
    w = 720

    arrow_id, shadow_id = _ids(mount_id, "arch")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 26, w - 80, h - 52, C["bg_outer"]))
    parts.append(_region_rect(inner_x, inner_y, inner_w, inner_h, C["bg_inner"]))

    # --- 2. Model-level scaffold (positions tracked by total height h) ---
    tok_text = _rect_block(parts, info, shadow_id, "tok_text",
                           cx - 110, h - 100, 220, 44,
                           _block_label(info, "tok_text", "Tokenized text"), font_size=17)
    embed = _rect_block(parts, info, shadow_id, "embed",
                        cx - 130, h - 168, 260, 44,
                        _block_label(info, "embed", "Token Embedding layer"), font_size=17)
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
    chain = [tok_text, embed] + [block_pos[b["id"]] for b in chain_blocks] + [final_rms, lm_head]
    for src, dst in zip(chain, chain[1:]):
        parts.append(_v_line(src, dst, arrow_id))

    # Output arrow above lm_head.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": lm_head["top"], "x2": cx, "y2": lm_head["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    # --- 5. Residual loops (declared via residual_from) ---
    lane = inner_x + inner_w - 28
    for block in layer_blocks:
        src_id = block.get("residual_from")
        if src_id and src_id in block_pos and block["id"] in block_pos:
            src_geom = block_pos[src_id]
            dst_geom = block_pos[block["id"]]
            parts.append(_residual_loop_right(src_geom, dst_geom, lane, arrow_id))
            # Junction dot at the tap point on the input-arrow stem so the
            # bypass visually originates from the arrow, not from the block.
            parts.append(_branch_dot(src_geom["cx"], src_geom["bottom"] + GAP + 8))

    # --- 6. Side blocks (e.g. PLE) — placed off the central column ---
    for block in side_blocks:
        _draw_side_block(
            parts, info, shadow_id,
            block, block_pos,
            inner_x, inner_w, cx, arrow_id,
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
    cx: float,
    arrow_id: str,
) -> None:
    """Render a block that lives OFF the central chain (e.g. Gemma 4 PLE).

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

    # Side block sits at the same y as the block it feeds, shifted left/right.
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
    #     feeds this side block").  Routed as an L-bend.
    rail_x = geom["cx"]
    tap_y = tap_geom["bottom"] + GAP + 8
    parts.append(_branch_dot(tap_geom["cx"], tap_y))
    parts.append(_svg_tag("path", {
        "d": (
            f"M {tap_geom['cx']} {tap_y} "
            f"L {rail_x} {tap_y} "
            f"L {rail_x} {geom['bottom'] + GAP}"
        ),
        "fill": "none", "stroke": C["arrow"], "stroke-width": 1.6,
        "stroke-linecap": "round", "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})",
    }))

    # --- Output: short horizontal arrow into feeds target.
    if lane == "left":
        x1 = geom["right"]
        x2 = feeds_geom["left"] - GAP
    else:
        x1 = geom["left"]
        x2 = feeds_geom["right"] + GAP
    parts.append(_svg_tag("line", {
        "x1": x1, "y1": cy, "x2": x2, "y2": cy,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))


def _build_moe_view(ir: dict, info: dict, mount_id: str) -> str:
    # Slightly taller (h=620) to leave room for incoming arrow below router
    # and outgoing arrow above the sum node.
    w, h = 720, 620
    arrow_id, shadow_id = _ids(mount_id, "moe")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    # Router spans wide enough that all four expert arrows visibly emerge from
    # within the router rectangle (otherwise the leftmost/rightmost arrows
    # appear to start in empty space, far outside the router block).
    router_w = 540
    router = _rect_block(parts, info, shadow_id, "router",
                         (w - router_w) / 2, h - 130, router_w, 50, "Router")
    sum_node = _plus_block(parts, info, shadow_id, "add_moe", cx, 100)

    # Expert layout: 4 evenly-spaced columns symmetric around the centre.
    # Previous version overlapped slot 3 and slot 4 (wrong x for slot 4).
    expert_w, expert_h = 116, 54
    expert_y = 235
    n_total = ffn.get("num_experts")
    last_label = str(n_total) if n_total else "N"
    # Spacing computed so columns are: 60, 220, 380, 540 with w=720.
    side_pad = 60
    gap = (w - 2 * side_pad - 4 * expert_w) / 3
    slots = [
        (side_pad + 0 * (expert_w + gap), "Expert 1",        "expert_1"),
        (side_pad + 1 * (expert_w + gap), "Expert k",        "expert_k"),
        (side_pad + 2 * (expert_w + gap), "Expert k+1",      "expert_kp1"),
        (side_pad + 3 * (expert_w + gap), f"Expert {last_label}", "expert_n"),
    ]
    experts = [
        _rect_block(parts, info, shadow_id, node_id, x, expert_y, expert_w, expert_h, label, font_size=15)
        for x, label, node_id in slots
    ]

    # Ellipsis lives in the gap between expert 2 (k) and expert 3 (k+1).
    dots_x = (experts[1]["right"] + experts[2]["left"]) / 2
    dots_y = expert_y + expert_h / 2
    for i in range(-2, 3):
        parts.append(_svg_tag("circle", {"cx": dots_x + i * 7, "cy": dots_y, "r": 2.5, "fill": C["muted"]}))

    for expert in experts:
        parts.append(_v_seg(expert["cx"], router["top"], expert["bottom"] + GAP, arrow_id))

    for expert in experts:
        target_x = sum_node["cx"] + (-sum_node["r"] - GAP if expert["cx"] < sum_node["cx"] else sum_node["r"] + GAP)
        parts.append(_elbow_vh(expert["cx"], expert["top"], target_x, sum_node["cy"], arrow_id))

    if ffn.get("num_experts") and ffn.get("num_experts_per_tok"):
        sparsity = 100 * ffn["num_experts_per_tok"] / ffn["num_experts"]
        cg_x, cg_y, cg_w, cg_h = w - 244, 58, 188, 58
        parts.append(
            _svg_tag(
                "rect",
                {
                    "x": cg_x,
                    "y": cg_y,
                    "width": cg_w,
                    "height": cg_h,
                    "rx": 10,
                    "ry": 10,
                    "fill": C["bg_card"],
                    "stroke": C["border"],
                    "stroke-width": 0.5,
                },
            )
        )
        parts.append(
            _svg_text(
                cg_x + 12,
                cg_y + 18,
                "ACTIVE PER TOKEN",
                {
                    "fill": C["muted"],
                    "font-family": FONT_BODY,
                    "font-size": 10,
                    "letter-spacing": "0.12em",
                    "font-weight": 600,
                },
            )
        )
        parts.append(
            _svg_text(
                cg_x + 12,
                cg_y + 44,
                f"{ffn['num_experts_per_tok']} / {ffn['num_experts']}  -  {sparsity:.1f}%",
                {"fill": C["text"], "font-family": FONT_HEAD, "font-size": 22},
            )
        )

    # Outgoing arrow at the top — leaves the weighted-sum node going up.
    parts.append(_svg_tag("line", {
        "x1": sum_node["cx"], "y1": sum_node["top"],
        "x2": sum_node["cx"], "y2": sum_node["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        sum_node["cx"], sum_node["top"] - 46,
        "out",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Incoming arrow at the bottom — enters the router from below.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": router["bottom"] + 36,
        "x2": cx, "y2": router["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, router["bottom"] + 50,
        "in",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    return _svg(w, h, f"{ir.get('name', 'model')} mixture of experts", parts)


def _build_ffn_view(ir: dict, info: dict, mount_id: str) -> str:
    # Tall enough that both the outgoing "out" label and the incoming "in·x"
    # label sit inside the outer region boundary (region = y 30 → h-30).
    w, h = 720, 660
    arrow_id, shadow_id = _ids(mount_id, "ffn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    act_name = (ffn.get("activation") or "silu").upper()

    down_proj = _rect_block(parts, info, shadow_id, "down_proj", cx - 90, 110, 180, 50, "Linear (down)")
    mul_node = _plus_block(parts, info, shadow_id, "mul", cx, 230, "×")
    silu = _rect_block(parts, info, shadow_id, "silu", cx - 270, 330, 180, 50, act_name)
    up_proj = _rect_block(parts, info, shadow_id, "up_proj", cx + 90, 330, 180, 50, "Linear (up)")
    gate_proj = _rect_block(parts, info, shadow_id, "gate_proj", cx - 270, 460, 180, 50, "Linear (gate)")

    branch_y = h - 110
    parts.append(_svg_tag("circle", {"cx": cx, "cy": branch_y, "r": 4, "fill": C["arrow"]}))
    parts.append(_elbow_hv(cx, branch_y, gate_proj["cx"], gate_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(cx, branch_y, up_proj["cx"], up_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_line(gate_proj, silu, arrow_id))
    parts.append(_elbow_vh(silu["cx"], silu["top"], mul_node["cx"] - mul_node["r"] - GAP, mul_node["cy"], arrow_id))
    parts.append(_elbow_vh(up_proj["cx"], up_proj["top"], mul_node["cx"] + mul_node["r"] + GAP, mul_node["cy"], arrow_id))
    parts.append(_v_line(mul_node, down_proj, arrow_id))

    # Outgoing arrow at top — leaves down_proj going up.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": down_proj["top"],
        "x2": cx, "y2": down_proj["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, down_proj["top"] - 46,
        "out",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Incoming arrow at the bottom — points up into the input branch dot.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": branch_y + 38,
        "x2": cx, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(
        _svg_text(
            cx,
            h - 48,
            "in  ·  x",
            {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
        )
    )

    return _svg(w, h, f"{ir.get('name', 'model')} feed-forward block", parts)


def _build_ple_view(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for the per-layer-embeddings block.

    The forward pass is a 5-stage chain: ``gate → act → × per_layer_input →
    proj → norm``.  We draw it as a vertical chain (so it fits the same
    inspect panel as the FFN view), with the multiplication step annotated
    as receiving the cross-layer ``per_layer_input`` vector — that external
    feed isn't drawn yet (TODO: visualise the parallel pathway).
    """
    w, h = 720, 660
    arrow_id, shadow_id = _ids(mount_id, "ple")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    spec = info["dominant"]["spec"]
    ple_dim = ((ir.get("extras") or {}).get("per_layer_embeddings") or {}).get("hidden")
    hidden_size = ir.get("hidden_size")
    cx = w / 2

    # Stack bottom → top in execution order.
    gate = _rect_block(parts, info, shadow_id, "ple_gate",  cx - 110, h - 160, 220, 50, "Linear (gate)")
    act_label = _ple_activation(spec)
    act = _rect_block(parts, info, shadow_id, "ple_act",   cx - 90,  h - 250, 180, 44, act_label)
    mul = _plus_block(parts, info, shadow_id, "ple_mul",   cx,        h - 320, "×")
    proj = _rect_block(parts, info, shadow_id, "ple_proj", cx - 110, h - 410, 220, 50, "Linear (up)")
    norm = _rect_block(parts, info, shadow_id, "ple_norm", cx - 90,  h - 500, 180, 44, "RMSNorm")

    # Linear chain.
    for src, dst in ((gate, act), (act, mul), (mul, proj), (proj, norm)):
        parts.append(_v_line(src, dst, arrow_id))

    # Outgoing arrow above the norm.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": norm["top"],
        "x2": cx, "y2": norm["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, norm["top"] - 46, "out  →  add (residual)",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Incoming arrow below the gate.
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": gate["bottom"] + 38,
        "x2": cx, "y2": gate["bottom"] + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, gate["bottom"] + 56, "in  (hidden)",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # External feed annotation — points at × from the right.
    feed_x = mul["cx"] + mul["r"] + 24
    parts.append(_svg_tag("line", {
        "x1": feed_x + 110, "y1": mul["cy"],
        "x2": mul["cx"] + mul["r"] + GAP, "y2": mul["cy"],
        "stroke": "#1F9E78", "stroke-width": 1.6, "stroke-linecap": "round",
        "stroke-dasharray": "5 4",
        "marker-end": f"url(#{arrow_id})",
    }))
    parts.append(_svg_text(
        feed_x + 116, mul["cy"] - 10,
        "per_layer_input[L]",
        {"fill": "#1F9E78", "font-family": FONT_MONO, "font-size": 11, "font-weight": 700},
    ))
    parts.append(_svg_text(
        feed_x + 116, mul["cy"] + 6,
        f"({_fmt_int(ple_dim) if ple_dim else '?'}-d, built outside layers)",
        {"fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
    ))

    # Dim annotations alongside the chain.
    parts.append(_svg_text(
        gate["right"] + 14, gate["cy"],
        f"{_fmt_int(hidden_size)}  →  {_fmt_int(ple_dim)}" if ple_dim else "",
        {"dominant-baseline": "central", "fill": C["muted"],
         "font-family": FONT_MONO, "font-size": 10},
    ))
    parts.append(_svg_text(
        proj["right"] + 14, proj["cy"],
        f"{_fmt_int(ple_dim)}  →  {_fmt_int(hidden_size)}" if ple_dim else "",
        {"dominant-baseline": "central", "fill": C["muted"],
         "font-family": FONT_MONO, "font-size": 10},
    ))

    return _svg(w, h, f"{ir.get('name', 'model')} per-layer embeddings block", parts)


def _ple_activation(spec: dict) -> str:
    # Gemma 4 uses gelu_pytorch_tanh for PLE; fall back to whatever the
    # adapter declared for the FFN if it's set, otherwise plain GELU.
    raw = (spec.get("ffn") or {}).get("activation") or "gelu"
    return raw.split("_")[0].upper()


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

    for i, layer in enumerate(layers):
        sig = _signature(layer)
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
