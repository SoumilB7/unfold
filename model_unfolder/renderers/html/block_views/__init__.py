"""Reusable rich block detail views for the HTML renderer."""
from __future__ import annotations

from .attention import attention_card, attention_card_css, build_attention_view
from .feed_forward import build_ffn_view, build_moe_view
from .per_layer_embedding import build_per_layer_embedding_view


def block_detail_svg(ir: dict, info: dict, mount_id: str, block: dict) -> str | None:
    """Return a rich SVG for a clicked architecture block, when one exists."""
    if block.get("kind") == "attention":
        return build_attention_view(ir, info, mount_id)

    if block.get("kind") == "ffn":
        ffn = info["dominant"]["spec"]["ffn"]
        if ffn.get("kind") == "moe":
            return build_moe_view(ir, info, mount_id)
        return build_ffn_view(ir, info, mount_id)

    if block.get("detail_view") == "per_layer_embedding":
        return build_per_layer_embedding_view(ir, info, mount_id, block)

    return None
