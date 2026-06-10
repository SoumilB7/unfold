"""Reusable rich block detail views for the HTML renderer."""
from __future__ import annotations

from .attention import (
    attention_card,
    attention_card_css,
    build_attention_view,
    build_mla_kv_cache_view,
    build_mla_query_path_view,
)
from .feed_forward import build_dense_ffn_view, build_ffn_view
from .mixture_of_experts import build_moe_expert_view, build_moe_view
from .modalities import build_audio_path_view, build_multimodal_fusion_view, build_video_path_view, build_vision_path_view
from .per_layer_embedding import build_per_layer_embedding_view
from .registry import render_block_detail, render_sub_block_detail


def block_detail_svg(ir: dict, info: dict, mount_id: str, block: dict) -> str | None:
    """Return a rich SVG for a clicked architecture block, when one exists."""
    return render_block_detail(ir, info, mount_id, block)


def sub_block_detail_svg(ir: dict, info: dict, mount_id: str, child: dict) -> str | None:
    """Return a rich SVG for a clicked node inside a detail view."""
    return render_sub_block_detail(ir, info, mount_id, child)
