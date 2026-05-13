"""Reusable rich block detail views for the HTML renderer."""
from __future__ import annotations

from .attention import attention_card, attention_card_css, build_attention_view
from .attention_types import build_mla_kv_cache_view, build_mla_query_path_view
from .feed_forward import build_dense_ffn_view, build_ffn_view
from .mixture_of_experts import build_moe_expert_view, build_moe_view
from .per_layer_embedding import build_per_layer_embedding_view


def block_detail_svg(ir: dict, info: dict, mount_id: str, block: dict) -> str | None:
    """Return a rich SVG for a clicked architecture block, when one exists."""
    if block.get("kind") == "attention":
        return build_attention_view(ir, info, mount_id)

    if block.get("kind") == "ffn":
        ffn = info["dominant"]["spec"]["ffn"]
        if ffn.get("kind") == "moe":
            return build_moe_view(ir, info, mount_id)
        if block.get("detail_view") == "dense_ffn" or not ffn.get("gated", True):
            return build_dense_ffn_view(ir, info, mount_id)
        return build_ffn_view(ir, info, mount_id)

    if block.get("detail_view") == "per_layer_embedding":
        return build_per_layer_embedding_view(ir, info, mount_id, block)

    return None


def sub_block_detail_svg(ir: dict, info: dict, mount_id: str, child: dict) -> str | None:
    """Return a rich SVG for a clicked node inside a detail view."""
    if child.get("detail_view") == "mla_query_path":
        return build_mla_query_path_view(ir, info, mount_id, child)
    if child.get("detail_view") == "mla_kv_cache_path":
        return build_mla_kv_cache_view(ir, info, mount_id, child)
    if child.get("detail_view") == "moe_expert":
        return build_moe_expert_view(ir, info, mount_id, child)
    return None
