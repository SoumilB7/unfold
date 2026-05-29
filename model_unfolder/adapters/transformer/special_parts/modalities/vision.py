"""Vision and video modality path extraction."""
from __future__ import annotations

from typing import Any

from .accessors import architecture, as_int, drop_none, first, present_paths
from .detect import (
    has_cross_attention_adapter,
    has_video_input,
    is_unified_grid_stream,
    position_encoding_hint,
    vision_family_hint,
)
from .schema import pipeline_step


def vision_path(cfg: Any, text_cfg: Any, vision_cfg: Any, text_hidden_size: int) -> dict:
    """Return image/vision intake as semantic facts."""
    cross_attn = has_cross_attention_adapter(cfg, text_cfg)
    unified_grid = is_unified_grid_stream(cfg, vision_cfg)
    image_size = first(vision_cfg, "image_size", "input_size")
    patch_size = first(vision_cfg, "patch_size", "patch_size_h")
    hidden_size = vision_encoder_hidden_size(cfg, vision_cfg, unified_grid)
    projector_out = vision_projector_out(cfg, vision_cfg, text_hidden_size, cross_attn, unified_grid)
    projector_in = vision_projector_in(vision_cfg, hidden_size, cross_attn, unified_grid)
    num_layers = first(vision_cfg, "num_hidden_layers", "num_layers", "depth")
    num_heads = first(vision_cfg, "num_attention_heads", "num_heads", "attention_heads")
    token_count = visual_token_count(cfg, vision_cfg, cross_attn)
    encoder_kind = vision_encoder_kind(cfg, vision_cfg)
    projector_kind_value = projector_kind(cfg)
    projector_activation = first(cfg, "projector_hidden_act", "mm_projector_act")
    token_kind = (
        "vision_cross_attention_states" if cross_attn
        else "grid_visual_tokens" if unified_grid
        else "soft_visual_tokens"
    )
    token_node_id = "vision_context" if cross_attn else "soft_visual_tokens"
    final_operation = (
        "emit_cross_attention_states" if cross_attn
        else "emit_grid_token_stream" if unified_grid
        else "emit_soft_token_stream"
    )
    projection_operation = (
        "project_to_decoder_width" if cross_attn
        else "merge_patches_to_text_width" if unified_grid
        else "project_to_text_width"
    )

    return drop_none({
        "kind": (
            "image_to_cross_attention_states" if cross_attn
            else "image_to_grid_tokens" if unified_grid
            else "image_to_soft_visual_tokens"
        ),
        "input": drop_none({
            "kind": "image_pixels",
            "shape": ["batch", "images", "channels", "height", "width"],
            "image_size": image_size,
            "patch_size": patch_size,
        }),
        "embedding": drop_none({
            "kind": "patch_embedding",
            "patch_size": patch_size,
            "out_features": hidden_size,
            "grid": patch_grid_geometry(vision_cfg),
        }),
        "encoder": drop_none({
            "kind": encoder_kind,
            "architecture": architecture(vision_cfg),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_heads,
            "patch_size": patch_size,
            "position_encoding": vision_position_encoding(cfg, vision_cfg),
        }),
        "projector": drop_none({
            "kind": projector_kind_value,
            "in_features": projector_in,
            "out_features": projector_out,
            "activation": projector_activation,
        }),
        "tokens": drop_none({
            "kind": token_kind,
            "count": token_count,
            "count_options": token_count_options(cfg),
            "width": text_hidden_size or None,
            "grid": grid_spec(cfg, vision_cfg, "image") if unified_grid else None,
        }),
        "pipeline": drop_none([
            pipeline_step(
                "image_pixels",
                "input",
                "image_pixels",
                shape=["batch", "images", "channels", "height", "width"],
            ),
            pipeline_step(
                "patch_embedding",
                "patch_embedding",
                "patch_embedding",
                patch_size=patch_size,
                out_features=hidden_size,
            ),
            pipeline_step(
                "vision_encoder",
                "encode",
                encoder_kind,
                hidden_size=hidden_size,
                num_layers=num_layers,
            ),
            pipeline_step(
                "projector",
                projection_operation,
                projector_kind_value,
                in_features=projector_in,
                out_features=projector_out,
                activation=projector_activation,
            ),
            pipeline_step(
                token_node_id,
                final_operation,
                token_kind,
                count=token_count,
                width=text_hidden_size or None,
                grid=grid_spec(cfg, vision_cfg, "image") if unified_grid else None,
            ),
        ]),
        "trace": {
            "config_paths": present_paths(cfg, vision_cfg, [
                ("vision_config", vision_cfg),
                ("image_seq_length", cfg),
                ("image_token_id", cfg),
                ("image_token_index", cfg),
                ("vision_start_token_id", cfg),
                ("vision_end_token_id", cfg),
                ("mm_projector_type", cfg),
            ]),
        },
    })


def video_path(cfg: Any, vision_cfg: Any, text_hidden_size: int) -> dict:
    """Return video intake when a model reuses its visual tower for frames."""
    patch_size = first(vision_cfg, "patch_size", "patch_size_h")
    hidden_size = vision_encoder_hidden_size(cfg, vision_cfg, unified_grid=True)
    projector_in = vision_projector_in(vision_cfg, hidden_size, cross_attn=False, unified_grid=True)
    num_layers = first(vision_cfg, "num_hidden_layers", "num_layers", "depth")
    num_heads = first(vision_cfg, "num_attention_heads", "num_heads", "attention_heads")
    projector_out = vision_projector_out(cfg, vision_cfg, text_hidden_size, cross_attn=False, unified_grid=True)
    encoder_kind = vision_encoder_kind(cfg, vision_cfg)
    projector_kind_value = projector_kind(cfg)
    return drop_none({
        "kind": "video_to_grid_tokens",
        "input": drop_none({
            "kind": "video_frames",
            "shape": ["batch", "videos", "frames", "channels", "height", "width"],
            "patch_size": patch_size,
            "temporal_patch_size": first(vision_cfg, "temporal_patch_size"),
        }),
        "embedding": drop_none({
            "kind": "temporal_patch_embedding",
            "patch_size": patch_size,
            "temporal_patch_size": first(vision_cfg, "temporal_patch_size"),
            "out_features": hidden_size,
            "grid": patch_grid_geometry(vision_cfg),
        }),
        "encoder": drop_none({
            "kind": encoder_kind,
            "architecture": architecture(vision_cfg),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_heads,
            "position_encoding": vision_position_encoding(cfg, vision_cfg),
        }),
        "projector": drop_none({
            "kind": projector_kind_value,
            "in_features": projector_in,
            "out_features": projector_out,
        }),
        "tokens": drop_none({
            "kind": "grid_video_tokens",
            "width": text_hidden_size or None,
            "grid": grid_spec(cfg, vision_cfg, "video"),
        }),
        "pipeline": drop_none([
            pipeline_step(
                "video_frames",
                "input",
                "video_frames",
                shape=["batch", "videos", "frames", "channels", "height", "width"],
            ),
            pipeline_step(
                "video_patch_embedding",
                "temporal_patch_embedding",
                "temporal_patch_embedding",
                patch_size=patch_size,
                temporal_patch_size=first(vision_cfg, "temporal_patch_size"),
                out_features=hidden_size,
            ),
            pipeline_step(
                "video_encoder",
                "encode",
                encoder_kind,
                hidden_size=hidden_size,
                num_layers=num_layers,
            ),
            pipeline_step(
                "video_projector",
                "merge_patches_to_text_width",
                projector_kind_value,
                in_features=projector_in,
                out_features=projector_out,
            ),
            pipeline_step(
                "video_tokens",
                "emit_grid_token_stream",
                "grid_video_tokens",
                width=text_hidden_size or None,
                grid=grid_spec(cfg, vision_cfg, "video"),
            ),
        ]),
        "trace": {
            "config_paths": present_paths(cfg, vision_cfg, [
                ("vision_config", vision_cfg),
                ("video_token_id", cfg),
                ("video_token_index", cfg),
                ("vision_start_token_id", cfg),
                ("vision_end_token_id", cfg),
            ]),
        },
    })


def vision_encoder_hidden_size(cfg: Any, vision_cfg: Any, unified_grid: bool) -> Any:
    """Return the width used inside the visual encoder itself."""
    if unified_grid:
        return first(vision_cfg, "embed_dim", "vision_hidden_size", "width", "hidden_size")
    return first(vision_cfg, "hidden_size", "vision_hidden_size", "width", "embed_dim")


def vision_projector_out(
    cfg: Any,
    vision_cfg: Any,
    text_hidden_size: int,
    cross_attn: bool,
    unified_grid: bool,
) -> Any:
    """Return output width of the vision projector/merger."""
    if cross_attn:
        return text_hidden_size or first(cfg, "projection_dim", "text_hidden_size")
    if unified_grid:
        return text_hidden_size or first(vision_cfg, "hidden_size", "out_hidden_size", "output_dim")
    return text_hidden_size or first(cfg, "projection_dim", "text_hidden_size")


def vision_projector_in(vision_cfg: Any, encoder_hidden_size: Any, cross_attn: bool, unified_grid: bool) -> Any:
    """Return input width of the vision projector/merger."""
    if cross_attn:
        return first(vision_cfg, "vision_output_dim", "output_dim", "projection_dim") or encoder_hidden_size
    if unified_grid:
        return merged_patch_features(vision_cfg, encoder_hidden_size) or encoder_hidden_size
    return encoder_hidden_size


def merged_patch_features(vision_cfg: Any, encoder_hidden_size: Any) -> int | None:
    """Return flattened merged patch width for grid-token mergers."""
    hidden = as_int(encoder_hidden_size)
    merge = as_int(first(vision_cfg, "spatial_merge_size"))
    if hidden is None or merge is None:
        return None
    return hidden * (merge ** 2)


def vision_encoder_kind(cfg: Any, vision_cfg: Any) -> str:
    """Return a semantic kind for the vision tower."""
    return vision_family_hint(cfg, vision_cfg) or "vision_transformer"


def vision_position_encoding(cfg: Any, vision_cfg: Any) -> dict | None:
    """Return position encoding facts for the vision tower."""
    hint = position_encoding_hint(cfg)
    if hint:
        return hint
    if first(vision_cfg, "use_absolute_position_embeddings") is not None:
        return {"kind": "learned_absolute"}
    return None


def projector_kind(cfg: Any) -> str:
    """Return projector kind from structural config fields."""
    vision_cfg = first(cfg, "vision_config", "vision_model_config")
    if is_unified_grid_stream(cfg, vision_cfg):
        return "patch_merger"
    raw = first(cfg, "mm_projector_type", "projector_type", "multi_modal_projector_type")
    if raw:
        return str(raw)
    if first(cfg, "projector_hidden_act", "mm_projector_act"):
        return "mlp_projector"
    return "linear_projector"


def visual_token_count(cfg: Any, vision_cfg: Any, cross_attn: bool = False) -> int | None:
    """Return fixed per-image token count when the config declares one."""
    if cross_attn or has_cross_attention_adapter(cfg):
        count = mllama_tile_token_count(vision_cfg)
        if count is not None:
            return count
    direct = first(
        cfg,
        "image_seq_length",
        "num_image_tokens",
        "mm_tokens_per_image",
        "vision_soft_tokens_per_image",
        "tokens_per_image",
    )
    if direct is not None:
        return direct
    image_size = first(vision_cfg, "image_size")
    patch_size = first(vision_cfg, "patch_size")
    if image_size and patch_size:
        try:
            return int((int(image_size) // int(patch_size)) ** 2)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def mllama_tile_token_count(vision_cfg: Any) -> int | None:
    """Return tile token count for cross-attention vision towers."""
    image_size = first(vision_cfg, "image_size")
    patch_size = first(vision_cfg, "patch_size")
    if not (image_size and patch_size):
        return None
    try:
        patches = int((int(image_size) // int(patch_size)) ** 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return patches + 1


def token_count_options(cfg: Any) -> list[int] | None:
    """Return optional per-image token count choices."""
    for key in ("image_token_count_options", "soft_token_count_options", "tokens_per_image_options"):
        value = first(cfg, key)
        if isinstance(value, (list, tuple)) and value:
            return [int(v) for v in value if v is not None]
    return None


def patch_grid_geometry(vision_cfg: Any) -> dict | None:
    """Normalized patch-grid geometry as a single object.

    Models the patch layout as dims, not scalars, so square, non-square
    (``patch_size_h`` != ``patch_size_w``), dynamic-resolution (no fixed
    ``image_size``), temporal (video), and patch-merged towers all flow
    through one shape that the renderer formats without per-model branches::

        {
          "kind": "static_patch_grid" | "dynamic_patch_grid",
          "patch": {"h": 14, "w": 14, "t": 2?},   # t only for temporal
          "input": {"h": 448, "w": 448} | absent,  # absent => dynamic
          "tiles": {"h": 32, "w": 32} | absent,    # floor-div, when computable
          "spatial_merge_size": 2?,
        }
    """
    img_h, img_w = _hw(first(vision_cfg, "image_size", "input_size"))
    patch_h = as_int(first(vision_cfg, "patch_size", "patch_size_h"))
    patch_w = as_int(first(vision_cfg, "patch_size_w")) or patch_h
    temporal = as_int(first(vision_cfg, "temporal_patch_size"))
    merge = as_int(first(vision_cfg, "spatial_merge_size"))

    if patch_h is None and patch_w is None and img_h is None:
        return None

    dynamic = img_h is None
    tiles = None
    if not dynamic and patch_h and patch_w and img_w is not None:
        if img_h % patch_h == 0 and img_w % patch_w == 0:
            tiles = {"h": img_h // patch_h, "w": img_w // patch_w}

    return drop_none({
        "kind": "dynamic_patch_grid" if dynamic else "static_patch_grid",
        "patch": drop_none({"h": patch_h, "w": patch_w, "t": temporal}),
        "input": None if dynamic else drop_none({"h": img_h, "w": img_w}),
        "tiles": tiles,
        "spatial_merge_size": merge,
    })


def _hw(value: Any) -> tuple[int | None, int | None]:
    """Split a size config value into (height, width); scalars mean square."""
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return as_int(value[0]), as_int(value[1])
        if len(value) == 1:
            v = as_int(value[0])
            return v, v
        return None, None
    v = as_int(value)
    return v, v


def grid_spec(cfg: Any, vision_cfg: Any, modality: str) -> dict | None:
    """Return dynamic THW grid metadata for image/video streams."""
    runtime_name = "video_grid_thw" if modality == "video" else "image_grid_thw"
    return drop_none({
        "kind": "dynamic_thw_grid",
        "runtime_input": runtime_name,
        "axes": ["time", "height", "width"],
        "patch_size": first(vision_cfg, "patch_size", "patch_size_h"),
        "temporal_patch_size": first(vision_cfg, "temporal_patch_size"),
        "spatial_merge_size": first(vision_cfg, "spatial_merge_size"),
        "position_encoding": "multimodal_rope",
    })


def grid_runtime_inputs(modalities: dict[str, Any]) -> list[str] | None:
    """Return runtime grid tensors consumed by unified multimodal streams."""
    inputs: list[str] = []
    if "vision" in modalities:
        grid = ((modalities["vision"].get("tokens") or {}).get("grid") or {})
        if grid.get("runtime_input"):
            inputs.append(grid["runtime_input"])
    if "video" in modalities:
        grid = ((modalities["video"].get("tokens") or {}).get("grid") or {})
        if grid.get("runtime_input"):
            inputs.append(grid["runtime_input"])
    return inputs or None


__all__ = [
    "grid_runtime_inputs",
    "has_video_input",
    "video_path",
    "vision_path",
]
