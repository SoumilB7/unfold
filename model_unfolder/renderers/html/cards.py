"""Inspect-card HTML for architecture block clicks."""
from __future__ import annotations

from ...labels import activation_label
from .block_views import attention_card, block_detail_svg
from .utils import _attr, _fmt_int, _html


def _build_inspect_cards(ir: dict, info: dict, mount_id: str) -> str:
    """Cards-only HTML for the L2 inspect panel."""
    panels: list[str] = [_hint_card("default", "Click a block above to inspect it")]

    spec = info["dominant"]["spec"]
    layer_blocks = spec.get("blocks") or []

    for node_id in ("tok_text", "embed"):
        panels.append(_simple_card(node_id, *_meta(info, node_id)))

    for block in layer_blocks:
        kind = block.get("kind")
        node_id = block["id"]
        if kind == "attention":
            svg = block_detail_svg(ir, info, mount_id, block)
            if svg:
                title, desc = _meta(info, node_id)
                desc = _with_io_dim(ir, desc)
                panels.append(_rich_card(node_id, title, desc, svg))
            else:
                panels.append(attention_card(ir, info, lambda nid: _meta(info, nid)))
            continue

        svg = block_detail_svg(ir, info, mount_id, block)
        if svg:
            title, desc = _meta(info, node_id)
            desc = _with_io_dim(ir, desc)
            panels.append(_rich_card(node_id, title, desc, svg))
        else:
            panels.append(_simple_card(node_id, *_meta(info, node_id)))

    for node_id in ("final_rms", "lm_head"):
        panels.append(_simple_card(node_id, *_meta(info, node_id)))

    return "".join(panels)


def _build_sub_inspect_cards(ir: dict, info: dict, mount_id: str) -> str:
    """Cards-only HTML for the L3 sub-inspect panel."""
    panels: list[str] = [_l3_card("default", "", "")]
    ffn = info["dominant"]["spec"]["ffn"]
    children = _sub_inspect_children(info)

    if children:
        seen: set[str] = set()
        for child in children:
            child_id = child.get("id")
            if not child_id or child_id in seen:
                continue
            seen.add(child_id)
            panels.append(_l3_card(child_id, child.get("title", child_id), child.get("description", "")))
    else:
        panels.extend(_fallback_sub_inspect_cards(ir, ffn))

    return "".join(panels)


def _meta(info: dict, node_id: str) -> tuple[str, str]:
    return info.get("meta", {}).get(node_id, (node_id, ""))


def _with_io_dim(ir: dict, desc: str) -> str:
    hidden = _fmt_int(ir.get("hidden_size"))
    if not hidden:
        return desc
    suffix = f"input/output dim {hidden}"
    return f"{desc}; {suffix}" if desc else suffix


def _simple_card(node_id: str, title: str, desc: str) -> str:
    return (
        f'<div class="uf-card-detail uf-card-{_attr(node_id)}">'
        f'<div class="uf-card-title">{_html(title)}</div>'
        f'<div class="uf-card-desc">{_html(desc)}</div>'
        "</div>"
    )


def _hint_card(node_id: str, hint: str) -> str:
    return (
        f'<div class="uf-card-detail uf-card-hint uf-card-{_attr(node_id)}">'
        f"{_html(hint)}"
        "</div>"
    )


def _l3_card(node_id: str, title: str, desc: str) -> str:
    return (
        f'<div class="uf-card-detail uf-l3-{_attr(node_id)}">'
        f'<div class="uf-card-title">{_html(title)}</div>'
        f'<div class="uf-card-desc">{_html(desc)}</div>'
        "</div>"
    )


def _rich_card(node_id: str, title: str, desc: str, svg: str) -> str:
    return (
        f'<div class="uf-card-detail uf-card-{_attr(node_id)}">'
        f'<div class="uf-card-title">{_html(title)}</div>'
        f'<div class="uf-card-desc">{_html(desc)}</div>'
        f'<div class="uf-card-svg">{svg}</div>'
        "</div>"
    )


def _sub_inspect_children(info: dict) -> list[dict]:
    children: list[dict] = []
    for block in (info["dominant"]["spec"].get("blocks") or []):
        children.extend(block.get("children") or [])
    return children


def _fallback_sub_inspect_cards(ir: dict, ffn: dict) -> list[str]:
    h = _fmt_int(ir.get("hidden_size"))
    inter = _fmt_int(ffn.get("expert_intermediate_size") or ffn.get("intermediate_size"))
    activation = activation_label(ffn.get("activation") or "silu")
    if ffn.get("kind") != "moe" and not ffn.get("gated", True):
        return [
            _l3_card("up_proj", "Input projection", f"Linear · {h} → {inter}"),
            _l3_card("silu", f"{activation} activation", "Element-wise non-linearity after the input projection"),
            _l3_card("down_proj", "Output projection", f"Linear · {inter} → {h}"),
        ]

    panels = [
        _l3_card("gate_proj", "Gate projection", f"Linear · {h} → {inter} (gated path through {activation})"),
        _l3_card("up_proj", "Up projection", f"Linear · {h} → {inter}"),
        _l3_card("silu", f"{activation} activation", "Element-wise non-linearity applied to the gate path"),
        _l3_card("mul", "Element-wise multiply", f"{activation}(gate) × up — combines the gated and ungated paths"),
        _l3_card("down_proj", "Down projection", f"Linear · {inter} → {h}"),
    ]
    if ffn.get("kind") == "moe":
        n_experts = _fmt_int(ffn.get("num_experts")) if ffn.get("num_experts") else "N"
        n_active = ffn.get("num_experts_per_tok") or "k"
        n_shared = ffn.get("num_shared_experts") or 0
        panels.append(_l3_card("router", "Router", f"Linear · {h} → {n_experts} (selects top-{n_active} experts per token)"))
        expert_desc = (
            f"Dense FFN with same shape as above · {h} → {inter} → {h} · "
            f"only top-{n_active} of {n_experts} active per token"
            + (f" · plus {n_shared} shared expert(s) always active" if n_shared else "")
        )
        for eid in ("expert_1", "expert_k", "expert_kp1", "expert_n"):
            panels.append(_l3_card(eid, "Expert FFN", expert_desc))
        panels.append(_l3_card("add_moe", "Weighted sum", f"Combines top-{n_active} expert outputs weighted by router probabilities"))
    return panels
