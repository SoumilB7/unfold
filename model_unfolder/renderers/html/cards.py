"""Inspect-card HTML for architecture block clicks."""
from __future__ import annotations

import re

from ...labels import activation_label
from .block_views import attention_card, block_detail_svg, sub_block_detail_svg
from .utils import _attr, _fmt_int, _html, facts_html

_VIEWBOX_RE = re.compile(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"')


def _build_inspect_cards(ir: dict, info: dict, mount_id: str) -> str:
    """Cards-only HTML for the L2 inspect panel."""
    panels: list[str] = [_hint_card("default", "Click a block above to inspect it")]

    spec = info["dominant"]["spec"]
    layer_blocks = spec.get("blocks") or []

    for node_id in ("tok_text", "embed"):
        panels.append(_simple_card(node_id, *_meta(info, node_id)))

    for node_id in ("vision_path", "video_path", "audio_path", "fusion"):
        block = info.get("blocks", {}).get(node_id)
        if not block:
            continue
        svg = block_detail_svg(ir, info, mount_id, block)
        title, desc, facts = _meta(info, node_id)
        if svg:
            panels.append(_rich_card(node_id, title, desc, svg, facts))
        else:
            panels.append(_simple_card(node_id, title, desc, facts))

    for block in layer_blocks:
        kind = block.get("kind")
        node_id = block["id"]
        # Tier-2 connectors (static) are glyphs on the topology, not clickable
        # blocks — they get no inspect card (mirrors their non-clickable render).
        if block.get("static"):
            continue
        if kind == "attention":
            svg = block_detail_svg(ir, info, mount_id, block)
            if svg:
                title, desc, facts = _meta(info, node_id)
                panels.append(_rich_card(node_id, title, desc, svg, facts + _io_dim_fact(ir)))
            else:
                panels.append(attention_card(ir, info, lambda nid: _meta(info, nid)))
            continue

        svg = block_detail_svg(ir, info, mount_id, block)
        if svg:
            title, desc, facts = _meta(info, node_id)
            panels.append(_rich_card(node_id, title, desc, svg, facts + _io_dim_fact(ir)))
        else:
            panels.append(_simple_card(node_id, *_meta(info, node_id)))

    for node_id in ("final_rms", "lm_head"):
        panels.append(_simple_card(node_id, *_meta(info, node_id)))

    mtp_block = info.get("blocks", {}).get("mtp")
    if mtp_block:
        svg = block_detail_svg(ir, info, mount_id, mtp_block)
        title, desc, facts = _meta(info, "mtp")
        if svg:
            panels.append(_rich_card("mtp", title, desc, svg, facts))
        else:
            panels.append(_simple_card("mtp", title, desc, facts))

    return "".join(panels)


def _build_nested_inspect_panels(ir: dict, info: dict, mount_id: str) -> list[str]:
    """Cards-only HTML for recursive nested inspect panels."""
    levels = _nested_child_levels(info)
    if not levels:
        levels = [_fallback_sub_inspect_children(ir, info["dominant"]["spec"]["ffn"])]
    return [_nested_panel(ir, info, mount_id, children) for children in levels if children]


def _meta(info: dict, node_id: str) -> tuple[str, str, list[str]]:
    """Card meta normalized to (title, desc, facts) — older 2-tuples get []"""
    entry = info.get("meta", {}).get(node_id, (node_id, ""))
    if len(entry) >= 3:
        return entry[0], entry[1], list(entry[2] or [])
    return entry[0], entry[1], []


def _io_dim_fact(ir: dict) -> list[str]:
    hidden = _fmt_int(ir.get("hidden_size"))
    return [f"in/out {hidden}"] if hidden else []


def _simple_card(node_id: str, title: str, desc: str, facts: list[str] | None = None) -> str:
    return (
        f'<div class="uf-card-detail uf-card-{_attr(node_id)}" '
        f'data-card-id="{_attr(node_id)}" data-card-size="compact">'
        f'<div class="uf-card-title">{_html(title)}</div>'
        f'<div class="uf-card-desc">{_html(desc)}</div>'
        f"{facts_html(facts)}"
        "</div>"
    )


def _hint_card(node_id: str, hint: str) -> str:
    return (
        f'<div class="uf-card-detail uf-card-hint uf-card-{_attr(node_id)}" '
        f'data-card-id="{_attr(node_id)}" data-card-size="hint">'
        f"{_html(hint)}"
        "</div>"
    )


def _nested_panel(ir: dict, info: dict, mount_id: str, children: list[dict]) -> str:
    panels: list[str] = [_nested_card("default", "", "")]
    for child in _unique_children(children):
        child_id = child.get("id")
        if not child_id or child.get("static"):
            continue
        svg = sub_block_detail_svg(ir, info, mount_id, child)
        title = child.get("title") or child.get("label") or child_id
        panels.append(_nested_card(child_id, title, child.get("description", ""), svg,
                                   child.get("facts")))
    return "".join(panels)


def _nested_card(node_id: str, title: str, desc: str, svg: str | None = None,
                 facts: list[str] | None = None) -> str:
    svg_html = f'<div class="uf-card-svg">{svg}</div>' if svg else ""
    size_attrs = _size_attrs(svg)
    return (
        f'<div class="uf-card-detail" data-card-id="{_attr(node_id)}"{size_attrs}>'
        f'<div class="uf-card-title">{_html(title)}</div>'
        f'<div class="uf-card-desc">{_html(desc)}</div>'
        f"{facts_html(facts)}"
        f"{svg_html}"
        "</div>"
    )


def _rich_card(node_id: str, title: str, desc: str, svg: str,
               facts: list[str] | None = None) -> str:
    size_attrs = _size_attrs(svg)
    return (
        f'<div class="uf-card-detail uf-card-{_attr(node_id)}" '
        f'data-card-id="{_attr(node_id)}"{size_attrs}>'
        f'<div class="uf-card-title">{_html(title)}</div>'
        f'<div class="uf-card-desc">{_html(desc)}</div>'
        f"{facts_html(facts)}"
        f'<div class="uf-card-svg">{svg}</div>'
        "</div>"
    )


def _size_attrs(svg: str | None) -> str:
    size, width, height = _card_size(svg)
    attrs = f' data-card-size="{_attr(size)}"'
    if width is not None and height is not None:
        attrs += f' data-svg-width="{_attr(width)}" data-svg-height="{_attr(height)}"'
    return attrs


def _card_size(svg: str | None) -> tuple[str, int | None, int | None]:
    if not svg:
        return "compact", None, None
    match = _VIEWBOX_RE.search(svg)
    if not match:
        return "diagram", None, None
    width = int(float(match.group(1)))
    height = int(float(match.group(2)))
    if height >= 640:
        return "diagram-tall", width, height
    if height >= 540:
        return "diagram", width, height
    return "diagram-compact", width, height


def _sub_inspect_children(info: dict) -> list[dict]:
    children: list[dict] = []
    for block in info.get("blocks", {}).values():
        if block.get("role") in {"modality_input", "fusion", "mtp"}:
            children.extend(block.get("children") or [])
    for block in (info["dominant"]["spec"].get("blocks") or []):
        children.extend(block.get("children") or [])
    return children


def _nested_child_levels(info: dict) -> list[list[dict]]:
    levels: list[list[dict]] = []
    current = _sub_inspect_children(info)
    while current:
        current = _unique_children(current)
        levels.append(current)
        next_level: list[dict] = []
        for child in current:
            next_level.extend(child.get("children") or [])
        current = next_level
    return levels


def _unique_children(children: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for child in children:
        child_id = child.get("id")
        if not child_id or child_id in seen:
            continue
        seen.add(child_id)
        unique.append(child)
    return unique


def _fallback_sub_inspect_children(ir: dict, ffn: dict) -> list[dict]:
    h = _fmt_int(ir.get("hidden_size"))
    inter = _fmt_int(ffn.get("expert_intermediate_size") or ffn.get("intermediate_size"))
    activation = activation_label(ffn.get("activation") or "silu")
    if ffn.get("kind") != "moe" and not ffn.get("gated", True):
        return [
            {"id": "up_proj", "title": "Input projection", "description": f"Linear · {h} → {inter}"},
            {
                "id": "silu",
                "title": activation,
                "description": "Element-wise non-linearity after the input projection",
            },
            {"id": "down_proj", "title": "Output projection", "description": f"Linear · {inter} → {h}"},
        ]

    panels = [
        {
            "id": "gate_proj",
            "title": "Gate projection",
            "description": f"Linear · {h} → {inter} (gated path through {activation})",
        },
        {"id": "up_proj", "title": "Up projection", "description": f"Linear · {h} → {inter}"},
        {
            "id": "silu",
            "title": activation,
            "description": "Element-wise non-linearity applied to the gate path.",
        },
        {
            "id": "mul",
            "title": "Element-wise multiply",
            "description": f"{activation}(gate) × up — combines the gated and ungated paths",
        },
        {"id": "down_proj", "title": "Down projection", "description": f"Linear · {inter} → {h}"},
    ]
    if ffn.get("kind") == "moe":
        n_experts = _fmt_int(ffn.get("num_experts")) if ffn.get("num_experts") else "N"
        n_active = ffn.get("num_experts_per_tok") or "k"
        n_shared = ffn.get("num_shared_experts") or 0
        panels.append({
            "id": "router",
            "title": "Router",
            "description": f"Linear · {h} → {n_experts} (selects top-{n_active} experts per token)",
        })
        expert_desc = (
            f"Dense FFN with same shape as above · {h} → {inter} → {h} · "
            f"only top-{n_active} of {n_experts} active per token"
            + (f" · plus {n_shared} shared expert(s) always active" if n_shared else "")
        )
        for eid in ("expert_1", "expert_k", "expert_kp1", "expert_n"):
            panels.append({"id": eid, "title": "Expert FFN", "description": expert_desc})
        panels.append({
            "id": "add_moe",
            "title": "Weighted sum",
            "description": f"Combines top-{n_active} expert outputs weighted by router probabilities",
        })
    return panels
