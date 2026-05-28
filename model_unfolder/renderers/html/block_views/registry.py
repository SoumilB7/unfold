"""Registry-backed dispatch for rich block detail views."""
from __future__ import annotations

from collections.abc import Callable

from .attention import build_attention_view
from .attention_types import build_mla_kv_cache_view, build_mla_query_path_view
from .feed_forward import build_dense_ffn_view, build_ffn_view
from .mixture_of_experts import build_moe_expert_view, build_moe_view
from .modalities import build_audio_path_view, build_multimodal_fusion_view, build_video_path_view, build_vision_path_view
from .modality_views.vision_details import (
    build_patch_embedding_view,
    build_vision_encoder_view,
    build_vision_mlp_view,
    build_vision_self_attention_view,
)
from .per_layer_embedding import build_per_layer_embedding_view

BlockRenderer = Callable[[dict, dict, str, dict], str | None]


def render_block_detail(ir: dict, info: dict, mount_id: str, block: dict) -> str | None:
    """Return a rich SVG for a clicked architecture block, when one exists."""
    if block.get("kind") == "attention":
        return build_attention_view(ir, info, mount_id)
    if block.get("kind") == "ffn":
        return _render_ffn_detail(ir, info, mount_id, block)

    renderer = BLOCK_DETAIL_VIEWS.get(block.get("detail_view"))
    if renderer:
        return renderer(ir, info, mount_id, block)
    return None


def render_sub_block_detail(ir: dict, info: dict, mount_id: str, child: dict) -> str | None:
    """Return a rich SVG for a clicked node inside a detail view."""
    renderer = SUB_BLOCK_DETAIL_VIEWS.get(child.get("detail_view"))
    if renderer:
        return renderer(ir, info, mount_id, child)
    return None


def _render_ffn_detail(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """Render the right FFN detail variant for dense/MoE blocks."""
    ffn = info["dominant"]["spec"]["ffn"]
    if ffn.get("kind") == "moe":
        return build_moe_view(ir, info, mount_id)
    if block.get("detail_view") == "dense_ffn" or not ffn.get("gated", True):
        return build_dense_ffn_view(ir, info, mount_id)
    return build_ffn_view(ir, info, mount_id)


BLOCK_DETAIL_VIEWS: dict[str | None, BlockRenderer] = {
    "per_layer_embedding": build_per_layer_embedding_view,
    "vision_path": build_vision_path_view,
    "audio_path": build_audio_path_view,
    "video_path": build_video_path_view,
    "multimodal_fusion": build_multimodal_fusion_view,
}

SUB_BLOCK_DETAIL_VIEWS: dict[str | None, BlockRenderer] = {
    "mla_query_path": build_mla_query_path_view,
    "mla_kv_cache_path": build_mla_kv_cache_view,
    "moe_expert": build_moe_expert_view,
    "vision_patch_embedding": build_patch_embedding_view,
    "vision_encoder": build_vision_encoder_view,
    "vision_self_attention": build_vision_self_attention_view,
    "vision_mlp": build_vision_mlp_view,
}
