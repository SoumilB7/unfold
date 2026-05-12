"""Inspect-card content and detail SVG for attention blocks."""
from __future__ import annotations

from ....labels import describe_attention, kv_shared, mask_long
from ..svg import (
    _branch_dot,
    _defs,
    _elbow_hv,
    _elbow_vh,
    _ids,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
    _v_seg,
)
from ..theme import C, FONT_MONO, GAP
from ..utils import _fmt_int, _html


_DETAIL_PLACEMENT = {
    "gqa": {"dx": 0, "dy": 8},
}


def build_attention_view(ir: dict, info: dict, mount_id: str) -> str:
    """Rich SVG detail view for the active attention-like block."""
    attn = info["dominant"]["spec"].get("attention") or {}
    kind = attn.get("kind")
    if kind == "mla":
        return _build_mla_attention_view(ir, info, mount_id)
    if kind == "mqa":
        return _build_mqa_attention_view(ir, info, mount_id)
    if kind == "gqa":
        return _build_gqa_attention_view(ir, info, mount_id)
    if kind == "ssm":
        return _build_ssm_view(ir, info, mount_id)
    if kind == "recurrent":
        return _build_recurrent_view(ir, info, mount_id)
    if kind == "rwkv":
        return _build_rwkv_view(ir, info, mount_id)
    if kind == "linear":
        return _build_linear_attention_view(ir, info, mount_id)
    return _build_sdpa_attention_view(ir, info, mount_id)


def _build_gqa_attention_view(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for grouped-query attention.

    GQA keeps one query head per query stream, but shares each K/V head across a
    small group of query heads. The diagram makes that sharing the focal point
    instead of presenting it as ordinary three-way Q/K/V attention.
    """
    w, h = 720, 730
    arrow_id, shadow_id = _ids(mount_id, "gqa-attn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))
    body: list[str] = []

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden_sz = ir.get("hidden_size") or 0
    hidden = _fmt_int(hidden_sz)
    num_heads = attn.get("num_heads") or 0
    num_kv_heads = attn.get("num_kv_heads") or num_heads
    head_dim = attn.get("head_dim") or (hidden_sz // num_heads if num_heads else 0)
    q_out = _fmt_int(num_heads * head_dim) if (num_heads and head_dim) else hidden
    kv_out = _fmt_int(num_kv_heads * head_dim) if (num_kv_heads and head_dim) else hidden
    d_k = str(head_dim) if head_dim else "d_k"
    q_per_group = _queries_per_kv_group(num_heads, num_kv_heads)

    cx = w / 2

    o_proj = _rect_block(body, info, shadow_id, "o_proj", cx - 92, 78, 184, 50, "Linear (out)")
    sdpa = _rect_block(
        body,
        info,
        shadow_id,
        "qkv_dot",
        126,
        165,
        468,
        58,
        ["Grouped SDPA", f"{num_heads} Q / {num_kv_heads} KV heads  ·  d_k = {d_k}"],
        font_size=15,
    )

    panel = _gqa_grouping_panel(body, 76, 280, 568, 114, num_heads, num_kv_heads, q_per_group)

    proj_w, proj_h, proj_y = 168, 50, 474
    q_proj = _rect_block(body, info, shadow_id, "q_proj", 70, proj_y, proj_w, proj_h, ["Linear (Q)", f"{num_heads} heads"], font_size=15)
    kv_head_label = f"{num_kv_heads} head" if num_kv_heads == 1 else f"{num_kv_heads} heads"
    k_proj = _rect_block(body, info, shadow_id, "k_proj", 276, proj_y, proj_w, proj_h, ["Linear (K)", kv_head_label], font_size=15)
    v_proj = _rect_block(body, info, shadow_id, "v_proj", 482, proj_y, proj_w, proj_h, ["Linear (V)", kv_head_label], font_size=15)

    branch_x, branch_y = cx, 582
    body.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 36, "x2": branch_x, "y2": branch_y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "fill": "none",
    }))
    body.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    body.append(_elbow_hv(branch_x, branch_y, k_proj["cx"], k_proj["bottom"] + GAP, arrow_id))
    body.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))
    body.append(_branch_dot(branch_x, branch_y))

    panel_entry_y = panel["bottom"] + GAP
    body.append(_v_seg(q_proj["cx"], q_proj["top"], panel_entry_y, arrow_id))
    body.append(_v_seg(k_proj["cx"], k_proj["top"], panel_entry_y, arrow_id))
    body.append(_v_seg(v_proj["cx"], v_proj["top"], panel_entry_y, arrow_id))
    body.append(_v_line(panel, sdpa, arrow_id))
    body.append(_v_line(sdpa, o_proj, arrow_id))
    _output_stem(body, cx, o_proj, arrow_id, hidden, show_label=False)

    if q_per_group and q_per_group > 1:
        _kv_cache_badge(body, w - 218, 58, f"KV cache {q_per_group}x smaller", "than full MHA")

    parts.append(_placed_figure(body, "gqa"))
    return _svg(w, h, f"{ir.get('name', 'model')} grouped-query attention", parts)


def _build_mqa_attention_view(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for multi-query attention / single shared K/V attention."""
    w, h = 780, 604
    dx = (w - 720) / 2
    arrow_id, shadow_id = _ids(mount_id, "mqa-attn")
    parts = [_defs(arrow_id, shadow_id)]
    body: list[str] = []

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden_sz = ir.get("hidden_size") or 0
    hidden = _fmt_int(hidden_sz)
    num_heads = attn.get("num_heads") or 0
    head_dim = attn.get("head_dim") or (hidden_sz // num_heads if num_heads else 0)
    q_out = _fmt_int(num_heads * head_dim) if (num_heads and head_dim) else hidden
    kv_out = _fmt_int(head_dim) if head_dim else hidden
    d_k = str(head_dim) if head_dim else "d_k"
    cx = w / 2

    o_proj = _rect_block(body, info, shadow_id, "o_proj", cx - 92, 82, 184, 50, "Linear (out)")
    sdpa = _rect_block(
        body,
        info,
        shadow_id,
        "qkv_dot",
        132 + dx,
        180,
        456,
        58,
        ["Multi-Query SDPA", f"{num_heads} Q heads + 1 shared K/V  ·  d_k = {d_k}"],
        font_size=15,
    )

    shared_kv = _mqa_shared_kv_node(body, 260 + dx, 310, 200, 48, num_heads)

    q_proj = _rect_block(body, info, shadow_id, "q_proj", 92 + dx, 418, 184, 50, ["Linear (Q)", f"{num_heads} heads"], font_size=15)
    k_proj = _rect_block(body, info, shadow_id, "k_proj", 330 + dx, 422, 140, 48, ["Linear (K)", "1 head"], font_size=15)
    v_proj = _rect_block(body, info, shadow_id, "v_proj", 500 + dx, 422, 140, 48, ["Linear (V)", "1 head"], font_size=15)

    branch_x, branch_y = cx, 526
    body.append(_branch_dot(branch_x, branch_y))
    body.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 34, "x2": branch_x, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    body.append(_svg_text(
        branch_x, h - 24,
        f"in ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    body.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    body.append(_elbow_hv(branch_x, branch_y, k_proj["cx"], k_proj["bottom"] + GAP, arrow_id))
    body.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))

    sdpa_entry_y = sdpa["bottom"] + GAP
    body.append(_elbow_hv(q_proj["cx"], q_proj["top"], sdpa["left"] + 120, sdpa_entry_y, arrow_id))
    body.append(_elbow_vh(k_proj["cx"], k_proj["top"], shared_kv["left"] + 62, shared_kv["bottom"] + GAP, arrow_id))
    body.append(_elbow_vh(v_proj["cx"], v_proj["top"], shared_kv["right"] - 62, shared_kv["bottom"] + GAP, arrow_id))
    body.append(_v_line(shared_kv, sdpa, arrow_id))
    body.append(_v_line(sdpa, o_proj, arrow_id))
    _output_stem(body, cx, o_proj, arrow_id, hidden)

    _attn_dim_label(body, q_proj["left"] + 4, q_proj["bottom"] + 23, f"Q: {hidden} -> {q_out}")
    _attn_dim_label(body, k_proj["left"] + 4, k_proj["bottom"] + 23, f"K: {hidden} -> {kv_out}")
    _attn_dim_label(body, v_proj["left"] + 4, v_proj["bottom"] + 23, f"V: {hidden} -> {kv_out}")
    _attn_dim_label(body, o_proj["left"] - 16, o_proj["cy"], f"{q_out} -> {hidden}", anchor="end")

    badge = None
    if num_heads and num_heads > 1:
        badge = _kv_cache_badge(body, w - 218, 58, f"KV cache {num_heads}x smaller", "than full MHA")

    body.append(_svg_text(
        cx,
        shared_kv["top"] - 18,
        f"1 shared K/V head feeds all {num_heads} Q heads" if num_heads else "single shared K/V head",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    region = _dynamic_region_rect(
        [
            o_proj,
            sdpa,
            shared_kv,
            q_proj,
            k_proj,
            v_proj,
            badge,
            {"left": cx - 86, "right": cx + 86, "top": o_proj["top"] - 56, "bottom": branch_y + 56},
        ],
        w,
        h,
        pad_x=86,
        pad_y=36,
    )
    parts.append(region)
    parts.extend(body)
    return _svg(w, h, f"{ir.get('name', 'model')} multi-query attention", parts)


def _build_sdpa_attention_view(ir: dict, info: dict, mount_id: str) -> str:
    """Rich SVG detail view for MHA / GQA / MQA attention blocks.

    Layout (bottom → top, matching the feed-forward convention):
      input branch → Q / K / V projections → scaled dot-product → output projection
    """
    w, h = 720, 590
    arrow_id, shadow_id = _ids(mount_id, "attn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden_sz = ir.get("hidden_size") or 0
    hidden = _fmt_int(hidden_sz)
    num_heads = attn.get("num_heads") or 0
    num_kv_heads = attn.get("num_kv_heads") or num_heads
    head_dim = attn.get("head_dim") or (hidden_sz // num_heads if num_heads else 0)
    q_out = _fmt_int(num_heads * head_dim) if (num_heads and head_dim) else hidden
    kv_out = _fmt_int(num_kv_heads * head_dim) if (num_kv_heads and head_dim) else hidden
    d_k = str(head_dim) if head_dim else "d_k"

    cx = w / 2

    # --- Output projection (top) ---
    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 90, 100, 180, 50, "Linear (out)")

    # --- Scaled dot-product attention (center, wide) ---
    if num_heads and head_dim:
        attn_kind = attn.get("kind", "mha")
        if attn_kind == "gqa" and num_kv_heads and num_kv_heads < num_heads:
            subtitle = f"{num_heads}Q / {num_kv_heads} KV heads  ·  d_k = {d_k}"
        elif attn_kind == "mqa":
            subtitle = f"{num_heads}Q / 1 KV head (MQA)  ·  d_k = {d_k}"
        else:
            subtitle = f"{num_heads} heads  ·  d_k = {d_k}"
        sdpa_label: str | list = ["Scaled Dot-Product Attention", subtitle]
    else:
        sdpa_label = "Scaled Dot-Product Attention"

    sdpa = _rect_block(parts, info, shadow_id, "qkv_dot", 80, 200, 560, 54, sdpa_label, font_size=15)

    # --- Q / K / V projections (symmetric trio) ---
    proj_w, proj_h, proj_y = 165, 50, 315
    q_proj = _rect_block(parts, info, shadow_id, "q_proj",  70, proj_y, proj_w, proj_h, "Linear (Q)")
    k_proj = _rect_block(parts, info, shadow_id, "k_proj", 278, proj_y, proj_w, proj_h, "Linear (K)")
    v_proj = _rect_block(parts, info, shadow_id, "v_proj", 486, proj_y, proj_w, proj_h, "Linear (V)")

    # --- Branch point ---
    branch_x, branch_y = cx, 475
    parts.append(_branch_dot(branch_x, branch_y))

    # Input stem
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 38, "x2": branch_x, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        branch_x, h - 32,
        f"in ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Branch → Q, K, V
    parts.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(branch_x, branch_y, k_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))

    # Q → SDPA left entry, K → center, V → right entry
    sdpa_entry_gap = sdpa["bottom"] + GAP
    parts.append(_elbow_hv(q_proj["cx"], q_proj["top"], 200, sdpa_entry_gap, arrow_id))
    parts.append(_v_seg(k_proj["cx"], k_proj["top"], sdpa_entry_gap, arrow_id))
    parts.append(_elbow_hv(v_proj["cx"], v_proj["top"], 520, sdpa_entry_gap, arrow_id))

    # SDPA → O projection
    parts.append(_v_line(sdpa, o_proj, arrow_id))

    # O projection → output label
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": o_proj["top"], "x2": cx, "y2": o_proj["top"] - 34,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, o_proj["top"] - 44,
        f"out ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Dimension annotations
    _attn_dim_label(parts, q_proj["right"] + 10, q_proj["cy"], f"{hidden} → {q_out}")
    _attn_dim_label(parts, k_proj["right"] + 10, k_proj["cy"], f"{hidden} → {kv_out}")
    _attn_dim_label(parts, v_proj["right"] + 10, v_proj["cy"], f"{hidden} → {kv_out}")
    _attn_dim_label(parts, o_proj["right"] + 10, o_proj["cy"], f"{q_out} → {hidden}")

    return _svg(w, h, f"{ir.get('name', 'model')} attention", parts)


def _queries_per_kv_group(num_heads: int, num_kv_heads: int) -> int | None:
    if not num_heads or not num_kv_heads:
        return None
    if num_heads % num_kv_heads:
        return None
    return num_heads // num_kv_heads


def _gqa_grouping_panel(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    num_heads: int,
    num_kv_heads: int,
    q_per_group: int | None,
) -> dict:
    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "rx": 14,
        "ry": 14,
        "fill": C["bg_card"],
        "stroke": C["border"],
        "stroke-width": 0.7,
    }))
    parts.append(_svg_text(
        x + 18,
        y + 24,
        "KV sharing pattern",
        {
            "fill": C["text"],
            "font-family": FONT_MONO,
            "font-size": 11,
            "font-weight": 700,
            "letter-spacing": "0.08em",
        },
    ))

    card_y = y + 46
    card_w = 112
    gap = 16
    start_x = x + 38
    cards = _gqa_card_specs(num_heads, num_kv_heads, q_per_group)
    for i, (top, bottom) in enumerate(cards):
        _gqa_group_card(parts, start_x + i * (card_w + gap), card_y, card_w, 48, top, bottom)

    return {
        "left": x,
        "right": x + w,
        "top": y,
        "bottom": y + h,
        "cx": x + w / 2,
        "cy": y + h / 2,
        "w": w,
        "h": h,
    }


def _mqa_shared_kv_node(parts: list[str], x: float, y: float, w: float, h: float, num_heads: int) -> dict:
    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "rx": 12,
        "ry": 12,
        "fill": C["badge_bg"],
        "stroke": C["border"],
        "stroke-width": 0.7,
    }))
    parts.append(_svg_text(
        x + w / 2,
        y + 18,
        "Shared K/V cache",
        {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_MONO, "font-size": 12, "font-weight": 700},
    ))
    parts.append(_svg_text(
        x + w / 2,
        y + 35,
        f"1 K + 1 V reused by {num_heads} Q" if num_heads else "1 K + 1 V reused",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
    ))
    return {
        "left": x,
        "right": x + w,
        "top": y,
        "bottom": y + h,
        "cx": x + w / 2,
        "cy": y + h / 2,
        "w": w,
        "h": h,
    }


def _gqa_card_specs(num_heads: int, num_kv_heads: int, q_per_group: int | None) -> list[tuple[str, str]]:
    if not num_heads or not num_kv_heads or not q_per_group:
        return [("Q groups", "share K/V"), ("...", "..."), ("KV heads", "reused")]

    def q_range(group_idx: int) -> str:
        start = group_idx * q_per_group
        end = min(start + q_per_group - 1, num_heads - 1)
        return f"Q{start}" if start == end else f"Q{start}-Q{end}"

    if num_kv_heads == 1:
        return [(f"{q_range(0)}", "use KV0")]
    if num_kv_heads == 2:
        return [(q_range(0), "use KV0"), (q_range(1), "use KV1")]
    return [
        (q_range(0), "use KV0"),
        (q_range(1), "use KV1"),
        ("...", "..."),
        (q_range(num_kv_heads - 1), f"use KV{num_kv_heads - 1}"),
    ]


def _gqa_group_card(parts: list[str], x: float, y: float, w: float, h: float, top: str, bottom: str) -> None:
    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "rx": 10,
        "ry": 10,
        "fill": C["badge_bg"],
        "stroke": C["border"],
        "stroke-width": 0.7,
    }))
    parts.append(_svg_text(
        x + w / 2,
        y + 18,
        top,
        {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_MONO, "font-size": 12, "font-weight": 700},
    ))
    parts.append(_svg_text(
        x + w / 2,
        y + 36,
        bottom,
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
    ))


def _kv_cache_badge(parts: list[str], x: float, y: float, title: str, subtitle: str) -> dict:
    w, h = 162, 52
    parts.append(_svg_tag("rect", {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "rx": 12,
        "ry": 12,
        "fill": C["bg_card"],
        "stroke": C["border"],
        "stroke-width": 0.7,
    }))
    parts.append(_svg_text(
        x + 81,
        y + 20,
        title,
        {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_MONO, "font-size": 11, "font-weight": 700},
    ))
    parts.append(_svg_text(
        x + 81,
        y + 37,
        subtitle,
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 9},
    ))
    return {"left": x, "right": x + w, "top": y, "bottom": y + h, "cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h}


def _dynamic_region_rect(
    geoms: list[dict | None],
    svg_w: int,
    svg_h: int,
    *,
    pad_x: float,
    pad_y: float,
) -> str:
    real_geoms = [g for g in geoms if g]
    if not real_geoms:
        return _region_rect(40, 30, svg_w - 80, svg_h - 60, C["bg_outer"])
    left = max(10, min(g["left"] for g in real_geoms) - pad_x)
    right = min(svg_w - 10, max(g["right"] for g in real_geoms) + pad_x)
    top = max(18, min(g["top"] for g in real_geoms) - pad_y)
    bottom = min(svg_h - 18, max(g["bottom"] for g in real_geoms) + pad_y)
    return _region_rect(left, top, right - left, bottom - top, C["bg_outer"])


def _placed_figure(parts: list[str], key: str) -> str:
    """Wrap a detail drawing in a tiny placement shim.

    This is the deliberate adjustment trigger for hand-drawn block detail
    views.  Keep node coordinates stable inside each view, then nudge the whole
    figure with ``_DETAIL_PLACEMENT`` when it looks visually off in the card.
    """
    placement = _DETAIL_PLACEMENT.get(key, {})
    dx = placement.get("dx", 0)
    dy = placement.get("dy", 0)
    body = "".join(parts)
    if not dx and not dy:
        return body
    return _svg_tag("g", {"transform": f"translate({dx} {dy})"}, body)


def _build_mla_attention_view(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for Multi-head Latent Attention (DeepSeek / Kimi style)."""
    w, h = 720, 620
    arrow_id, shadow_id = _ids(mount_id, "mla")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden = _fmt_int(ir.get("hidden_size"))
    kv_rank = _fmt_int(attn.get("kv_lora_rank"))
    q_rank = _fmt_int(attn.get("q_lora_rank")) if attn.get("q_lora_rank") else "direct"
    rope_dim = _fmt_int(attn.get("rope_dim"))
    cx = w / 2

    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 90, 82, 180, 50, "Linear (out)")
    mla = _rect_block(parts, info, shadow_id, "mla_attn", 142, 182, 436, 58, ["Multi-Head Latent", "Attention"], font_size=16)
    q_path = _rect_block(parts, info, shadow_id, "mla_q", 76, 350, 170, 50, ["Q path", f"rank {q_rank}"], font_size=15)
    kv_down = _rect_block(parts, info, shadow_id, "mla_kv_down", 276, 418, 170, 50, ["KV down", f"rank {kv_rank}"], font_size=15)
    kv_up = _rect_block(parts, info, shadow_id, "mla_kv_up", 276, 302, 170, 50, "KV up", font_size=16)
    rope = _rect_block(parts, info, shadow_id, "mla_rope", 476, 350, 170, 50, ["RoPE split", f"dim {rope_dim}"], font_size=15)

    branch_x, branch_y = cx, 540
    parts.append(_branch_dot(branch_x, branch_y))
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 34, "x2": branch_x, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        branch_x, h - 28,
        f"in ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    parts.append(_elbow_hv(branch_x, branch_y, q_path["cx"], q_path["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, kv_down["cx"], kv_down["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, rope["cx"], rope["bottom"] + GAP, arrow_id))
    parts.append(_v_line(kv_down, kv_up, arrow_id))
    parts.append(_elbow_hv(q_path["cx"], q_path["top"], 210, mla["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(kv_up["cx"], kv_up["top"], mla["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(rope["cx"], rope["top"], 510, mla["bottom"] + GAP, arrow_id))
    parts.append(_v_line(mla, o_proj, arrow_id))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": o_proj["top"],
        "x2": cx, "y2": o_proj["top"] - 34,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, o_proj["top"] - 44,
        f"out ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    return _svg(w, h, f"{ir.get('name', 'model')} multi-head latent attention", parts)


def _build_ssm_view(ir: dict, info: dict, mount_id: str) -> str:
    attn = info["dominant"]["spec"].get("attention") or {}
    state = _fmt_int(attn.get("head_dim"))
    subtitle = f"state dim {state}" if state != "?" else "selective recurrence"
    return _vertical_attention_stack(
        ir,
        info,
        mount_id,
        "ssm",
        "selective state-space block",
        [
            ("ssm_out_proj", "Output projection", None),
            ("ssm_gate", "Gate", None),
            ("ssm_scan", ["Selective Scan", subtitle], 16),
            ("ssm_conv", "Local Conv", None),
            ("ssm_in_proj", "Input projection", None),
        ],
    )


def _build_recurrent_view(ir: dict, info: dict, mount_id: str) -> str:
    attn = info["dominant"]["spec"].get("attention") or {}
    width = _fmt_int(attn.get("head_dim"))
    return _vertical_attention_stack(
        ir,
        info,
        mount_id,
        "recurrent",
        "linear recurrent unit",
        [
            ("lru_out_proj", "Output projection", None),
            ("lru_gate", "Gate", None),
            ("lru_state", ["Recurrent State", f"width {width}"], 16),
            ("lru_in_proj", "Input projection", None),
        ],
        h=520,
    )


def _build_linear_attention_view(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 720, 610
    arrow_id, shadow_id = _ids(mount_id, "linear-attn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    hidden = _fmt_int(ir.get("hidden_size"))
    cx = w / 2
    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 90, 78, 180, 50, "Linear (out)")
    mix = _rect_block(parts, info, shadow_id, "linear_mix", 165, 178, 390, 54, ["Linear Attention Mix", "prefix/state accumulation"], font_size=15)
    kernel = _rect_block(parts, info, shadow_id, "kernel_map", 215, 286, 290, 48, "Kernel feature map", font_size=15)
    q_proj = _rect_block(parts, info, shadow_id, "q_proj", 70, 400, 165, 50, "Linear (Q)")
    k_proj = _rect_block(parts, info, shadow_id, "k_proj", 278, 400, 165, 50, "Linear (K)")
    v_proj = _rect_block(parts, info, shadow_id, "v_proj", 486, 400, 165, 50, "Linear (V)")

    branch_x, branch_y = cx, 540
    parts.append(_branch_dot(branch_x, branch_y))
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 34, "x2": branch_x, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(branch_x, h - 28, f"in ({hidden})", {
        "text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11,
    }))

    parts.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(branch_x, branch_y, k_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(q_proj["cx"], q_proj["top"], 250, kernel["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(k_proj["cx"], k_proj["top"], kernel["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(v_proj["cx"], v_proj["top"], 470, mix["bottom"] + GAP, arrow_id))
    parts.append(_v_line(kernel, mix, arrow_id))
    parts.append(_v_line(mix, o_proj, arrow_id))
    _output_stem(parts, cx, o_proj, arrow_id, hidden)

    return _svg(w, h, f"{ir.get('name', 'model')} linear attention", parts)


def _build_rwkv_view(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 600, 560
    arrow_id, shadow_id = _ids(mount_id, "rwkv")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    hidden = _fmt_int(ir.get("hidden_size"))
    cx = w / 2
    out = _rect_block(parts, info, shadow_id, "rwkv_out", cx - 100, 80, 200, 50, "Output projection")
    time_mix = _rect_block(parts, info, shadow_id, "rwkv_time_mix", cx - 135, 190, 270, 54, ["Time-Mix", "linear recurrence"], font_size=16)
    receptance = _rect_block(parts, info, shadow_id, "rwkv_receptance", 55, 335, 150, 48, "Receptance", font_size=15)
    key = _rect_block(parts, info, shadow_id, "rwkv_key", 225, 335, 150, 48, "Key", font_size=15)
    value = _rect_block(parts, info, shadow_id, "rwkv_value", 395, 335, 150, 48, "Value", font_size=15)

    branch_x, branch_y = cx, 480
    parts.append(_branch_dot(branch_x, branch_y))
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 32, "x2": branch_x, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(branch_x, h - 26, f"in ({hidden})", {
        "text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11,
    }))
    for node in (receptance, key, value):
        parts.append(_elbow_hv(branch_x, branch_y, node["cx"], node["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(receptance["cx"], receptance["top"], 210, time_mix["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(key["cx"], key["top"], time_mix["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(value["cx"], value["top"], 390, time_mix["bottom"] + GAP, arrow_id))
    parts.append(_v_line(time_mix, out, arrow_id))
    _output_stem(parts, cx, out, arrow_id, hidden)

    return _svg(w, h, f"{ir.get('name', 'model')} RWKV token mixing", parts)


def _vertical_attention_stack(
    ir: dict,
    info: dict,
    mount_id: str,
    view: str,
    title: str,
    nodes: list[tuple[str, str | list[str], int | None]],
    *,
    h: int = 590,
) -> str:
    w = 560
    arrow_id, shadow_id = _ids(mount_id, view)
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    hidden = _fmt_int(ir.get("hidden_size"))
    cx = w / 2
    geoms = []
    top_y = 86
    step = (h - 190) / max(len(nodes) - 1, 1)
    for i, (node_id, label, font_size) in enumerate(nodes):
        block_w = 240 if isinstance(label, list) else 210
        y = top_y + i * step
        geoms.append(
            _rect_block(
                parts, info, shadow_id, node_id,
                cx - block_w / 2, y, block_w, 50,
                label,
                font_size=font_size or 16,
            )
        )

    for src, dst in zip(reversed(geoms), reversed(geoms[:-1])):
        parts.append(_v_line(src, dst, arrow_id))

    input_node = geoms[-1]
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": input_node["bottom"] + 38,
        "x2": cx, "y2": input_node["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        cx, h - 34,
        f"in ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))
    _output_stem(parts, cx, geoms[0], arrow_id, hidden)

    return _svg(w, h, f"{ir.get('name', 'model')} {title}", parts)


def _output_stem(
    parts: list[str],
    cx: float,
    top_node: dict,
    arrow_id: str,
    hidden: str,
    *,
    show_label: bool = True,
) -> None:
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": top_node["top"],
        "x2": cx, "y2": top_node["top"] - 34,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    if not show_label:
        return
    parts.append(_svg_text(
        cx, top_node["top"] - 44,
        f"out ({hidden})",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))


def _attn_dim_label(parts: list[str], x: float, y: float, text: str, *, anchor: str = "start") -> None:
    parts.append(_svg_text(
        x, y, text,
        {
            "text-anchor": anchor,
            "dominant-baseline": "central",
            "fill": C["muted"],
            "font-family": FONT_MONO,
            "font-size": 10,
        },
    ))


def attention_card(ir: dict, info: dict, meta_for: callable) -> str:
    """Inspect card for the attention block."""
    attn_groups = [
        g for g in info.get("groups", []) if g.get("spec", {}).get("attention")
    ]
    if len(attn_groups) <= 1:
        title, desc = meta_for("attn")
        return (
            '<div class="uf-card-detail uf-card-attn">'
            f'<div class="uf-card-title">{_html(title)}</div>'
            f'<div class="uf-card-desc">{_html(desc)}</div>'
            "</div>"
        )

    rows = "".join(_attention_row_for_group(group, ir) for group in attn_groups)
    return (
        '<div class="uf-card-detail uf-card-attn">'
        '<div class="uf-card-title">Attention layers</div>'
        '<div class="uf-card-desc">'
        f"{len(attn_groups)} attention variants in this model — each row is one variant."
        "</div>"
        f'<div class="uf-attn-rows">{rows}</div>'
        "</div>"
    )


def _attention_row_for_group(group: dict, ir: dict) -> str:
    attn = group["spec"]["attention"]
    indices = group["indices"]
    n_layers = len(indices)
    layers = ir.get("layers", [])
    n_shared = sum(
        1 for i in indices
        if 0 <= i < len(layers) and kv_shared(layers[i].get("attention") or {})
    )
    return _attention_row(attn, n_layers, n_shared)


def _attention_row(attn: dict, n_layers: int, n_shared: int) -> str:
    title = f"{mask_long(attn)} · {describe_attention(attn)}"
    bits: list[str] = []
    if attn.get("window_size"):
        bits.append(f"window {attn['window_size']}")
    if n_shared:
        bits.append(f"{n_shared} of {n_layers} reuse K/V from earlier layers")
    else:
        bits.append(f"{n_layers} layers")
    detail = "  ·  ".join(bits)
    return (
        '<div class="uf-attn-row">'
        f'<div class="uf-attn-row-title">{_html(title)}</div>'
        f'<div class="uf-attn-row-detail">{_html(detail)}</div>'
        "</div>"
    )


def attention_card_css(mount_id: str, theme: dict) -> str:
    return f"""
#{mount_id} .uf-attn-rows {{
  margin-top:10px;
  display:flex;
  flex-direction:column;
  gap:8px;
}}
#{mount_id} .uf-attn-row {{
  padding:9px 12px;
  background:{theme['bg_card']};
  border:0.5px solid {theme['border']};
  border-left:3px solid {theme['block']};
  border-radius:8px;
}}
#{mount_id} .uf-attn-row-title {{
  font-family:{theme['font_head']};
  font-size:16px;
  color:{theme['text']};
  line-height:1.15;
}}
#{mount_id} .uf-attn-row-detail {{
  margin-top:3px;
  font-size:12px;
  color:{theme['muted']};
  font-family:{theme['font_mono']};
}}
"""
