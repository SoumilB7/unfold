"""Vision and video modality path extraction."""
from __future__ import annotations

from typing import Any

from .accessors import architecture, as_int, drop_none, first, nested, present_paths
from .detect import (
    has_cross_attention_adapter,
    has_video_input,
    is_unified_grid_stream,
    model_family_hint,
    vision_family_hint,
)
from .schema import Stage, assemble_path


def vision_path(cfg: Any, text_cfg: Any, vision_cfg: Any, text_hidden_size: int) -> dict:
    """Return image/vision intake as semantic facts."""
    cross_attn = has_cross_attention_adapter(cfg, text_cfg)
    unified_grid = is_unified_grid_stream(cfg, vision_cfg)
    image_size = first(vision_cfg, "image_size", "input_size")
    patch_size = first(vision_cfg, "patch_size", "patch_size_h")
    input_channels = first(vision_cfg, "in_channels", "num_channels")
    hidden_size = vision_encoder_hidden_size(cfg, vision_cfg, unified_grid)
    projector_out = vision_projector_out(cfg, vision_cfg, text_hidden_size, cross_attn, unified_grid)
    projector_in = vision_projector_in(vision_cfg, hidden_size, cross_attn, unified_grid)
    num_layers = first(vision_cfg, "num_hidden_layers", "num_layers", "depth")
    num_heads = first(vision_cfg, "num_attention_heads", "num_heads", "attention_heads")
    # A perceiver resampler emits a fixed number of latent tokens regardless of
    # the patch count, so that count wins when present.
    token_count = perceiver_latents(cfg) or visual_token_count(cfg, vision_cfg, cross_attn)
    encoder_kind = vision_encoder_kind(cfg, vision_cfg)
    vision_family = vision_family_hint(cfg, vision_cfg)
    projector_kind_value = projector_kind(cfg)
    projector_activation = first(cfg, "projector_hidden_act", "mm_projector_act")
    embedding_ops = vision_patch_embedding_ops(cfg, vision_cfg, hidden_size)
    projector_profile = vision_projector_ops(
        cfg, vision_cfg, hidden_size, projector_in, projector_out,
        projector_activation, unified_grid,
    )
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

    image_shape = ["batch", "images", "channels", "height", "width"]
    grid = grid_spec(cfg, vision_cfg, "image") if unified_grid else None
    path_kind = (
        "image_to_cross_attention_states" if cross_attn
        else "image_to_grid_tokens" if unified_grid
        else "image_to_soft_visual_tokens"
    )

    # --- Structural vision features, all config-driven (no family names) ---
    tiling = image_tiling(cfg, vision_cfg)
    reduction = token_reduction(cfg, vision_cfg)
    multilayer = multilayer_features(vision_cfg)
    features = feature_selection(cfg)
    encoder_fields = drop_none({
        "architecture": architecture(vision_cfg),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        # A vision tower that declares global layers runs a local+global stack.
        "num_global_layers": first(vision_cfg, "num_global_layers"),
        "num_channels": first(vision_cfg, "num_channels"),
        "global_head_dim": first(vision_cfg, "global_head_dim"),
        "intermediate_size": first(vision_cfg, "intermediate_size", "mlp_dim"),
        "activation": first(vision_cfg, "hidden_act", "hidden_activation"),
        "patch_size": patch_size,
        # Which encoder output the connector reads (single layer + CLS policy).
        "feature_layer": (features or {}).get("layer"),
        "feature_select_strategy": (features or {}).get("select_strategy"),
        # Concatenated multi-layer features (e.g. mllama) widen the output.
        "intermediate_layers_indices": (multilayer or {}).get("layers"),
        "output_dim": (multilayer or {}).get("output_dim")
                      or first(vision_cfg, "vision_output_dim", "output_dim"),
        "position_encoding": vision_position_encoding(cfg, vision_cfg),
        "norm_kind": "RMSNorm" if vision_family == "pixtral_vision_transformer" else "LayerNorm",
        "ffn_gated": vision_family == "pixtral_vision_transformer",
    })

    stages = [
        Stage("input", "image_pixels", "input", "image_pixels",
              {"shape": image_shape, "image_size": image_size, "patch_size": patch_size,
               "channels": input_channels},
              step_fields={"shape": image_shape, "channels": input_channels}),
    ]
    if tiling:
        stages.append(
            Stage("tiling", "vision_tiles", "tile_image", "image_tiling",
                  {"mode": tiling.get("mode"), "max_tiles": tiling.get("max_tiles"),
                   "aspect_ratios": tiling.get("aspect_ratios"),
                   "num_layouts": tiling.get("num_layouts"),
                   "aspect_ratio_policy": tiling.get("aspect_ratio_policy")},
                  step_fields=drop_none({"mode": tiling.get("mode"),
                                         "max_tiles": tiling.get("max_tiles"),
                                         "num_layouts": tiling.get("num_layouts")}))
        )
    stages.append(
        Stage("embedding", "patch_embedding", "patch_embedding", "patch_embedding",
              drop_none({"patch_size": patch_size, "out_features": hidden_size,
                         "grid": patch_grid_geometry(vision_cfg),
                         "ops": embedding_ops or None}),
              step_fields={"patch_size": patch_size, "out_features": hidden_size})
    )
    stages.append(
        Stage("encoder", "vision_encoder", "encode", encoder_kind,
              encoder_fields,
              step_fields={"hidden_size": hidden_size, "num_layers": num_layers})
    )
    if reduction:
        stages.append(
            Stage("reduction", "vision_token_reduce", reduction["operation"], reduction["kind"],
                  reduction["fields"])
        )
    stages.append(
        Stage("projector", "projector", projection_operation, projector_kind_value,
              drop_none({"in_features": projector_in, "out_features": projector_out,
                         "activation": projector_activation, "num_latents": perceiver_latents(cfg),
                         "profile": (projector_profile or {}).get("profile"),
                         "ops": (projector_profile or {}).get("ops")}))
    )
    stages.append(
        Stage("tokens", token_node_id, final_operation, token_kind,
              {"count": token_count, "count_options": token_count_options(cfg),
               "width": text_hidden_size or None, "grid": grid},
              step_fields={"count": token_count, "width": text_hidden_size or None, "grid": grid})
    )
    return assemble_path(
        path_kind,
        stages,
        present_paths(cfg, vision_cfg, [
            ("vision_config", vision_cfg),
            ("image_seq_length", cfg),
            ("image_token_id", cfg),
            ("image_token_index", cfg),
            ("vision_start_token_id", cfg),
            ("vision_end_token_id", cfg),
            ("mm_projector_type", cfg),
        ]),
    )


def vision_patch_embedding_ops(cfg: Any, vision_cfg: Any, hidden_size: Any) -> list[dict]:
    """Return an exact patch-embedding chain only for established signatures.

    These are class/config-family signatures, never model ids. Unknown towers
    keep the honest generic patch view instead of receiving a guessed backend.
    """
    family = vision_family_hint(cfg, vision_cfg)
    patch = first(vision_cfg, "patch_size", "patch_size_h")
    temporal = first(vision_cfg, "temporal_patch_size")
    channels = first(vision_cfg, "in_channels", "num_channels")
    if temporal is not None:
        kernel = f"{temporal}×{patch}×{patch}" if patch is not None else None
        return [
            {"kind": "reshape", "label": "Reshape patches"},
            {"kind": "conv", "label": "Conv3d", "in": channels, "out": hidden_size,
             "meta": {"desc": "3D patch convolution over time, height, and width.",
                      "kernel": kernel}},
            {"kind": "reshape", "label": "Flatten tokens"},
        ]
    if family in {"siglip_vision_transformer", "pixtral_vision_transformer"}:
        ops = [
            {"kind": "conv", "label": "Conv2d", "in": channels, "out": hidden_size,
             "meta": {"desc": "2D patch convolution over the image grid.",
                      "kernel": f"{patch}×{patch}" if patch is not None else None}},
            {"kind": "reshape", "label": "Flatten spatial grid"},
            {"kind": "reshape", "label": "Transpose to tokens"},
        ]
        if family == "pixtral_vision_transformer":
            ops.append({"kind": "norm", "label": "RMSNorm"})
        return ops
    return []


def vision_projector_ops(
    cfg: Any,
    vision_cfg: Any,
    vision_hidden: Any,
    projector_in: Any,
    projector_out: Any,
    activation: Any,
    unified_grid: bool,
) -> dict | None:
    """Return source-established projector/merger chains for shared families."""
    merge = as_int(first(vision_cfg, "spatial_merge_size") or first(cfg, "spatial_merge_size"))
    hidden = as_int(vision_hidden)
    merged = hidden * merge**2 if hidden is not None and merge is not None else projector_in

    # Qwen-VL PatchMerger: norm each patch, regroup the spatial merge unit,
    # then Linear -> GELU -> Linear. Temporal patching is the structural
    # signature shared by its image/video-capable vision families.
    if unified_grid and first(vision_cfg, "temporal_patch_size") is not None:
        return {
            "profile": "qwen_vl_patch_merger",
            "ops": [
                {"kind": "norm", "label": "LayerNorm"},
                {"kind": "reshape", "label": "Merge neighbouring patches"},
                {"kind": "linear", "label": "Linear", "in": merged, "out": merged},
                {"kind": "activation", "fn": "gelu"},
                {"kind": "linear", "label": "Linear", "in": merged, "out": projector_out},
            ],
        }

    # Mistral3 wraps Pixtral with RMSNorm + a learned patch merger before its
    # two-linear projector. Plain Pixtral/LLaVA wrappers are deliberately not
    # swept into this profile because their connector code differs.
    root_arch = str(architecture(cfg) or "").lower()
    if model_family_hint(cfg) == "mistral3" or "mistral3" in root_arch:
        ops = [
            {"kind": "norm", "label": "RMSNorm"},
            {"kind": "reshape", "label": "Group neighbouring patches"},
            {"kind": "linear", "label": "Patch merge", "in": merged, "out": hidden},
            {"kind": "linear", "label": "Linear", "in": hidden, "out": projector_out},
        ]
        if activation:
            ops.append({"kind": "activation", "fn": str(activation)})
        ops.append({"kind": "linear", "label": "Linear",
                    "in": projector_out, "out": projector_out})
        return {"profile": "mistral3_multimodal_projector", "ops": ops}
    return None


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
    temporal_patch_size = first(vision_cfg, "temporal_patch_size")
    input_channels = first(vision_cfg, "in_channels", "num_channels")
    video_shape = ["batch", "videos", "frames", "channels", "height", "width"]
    grid = grid_spec(cfg, vision_cfg, "video")
    embedding_ops = vision_patch_embedding_ops(cfg, vision_cfg, hidden_size)
    projector_profile = vision_projector_ops(
        cfg, vision_cfg, hidden_size, projector_in, projector_out,
        first(cfg, "projector_hidden_act", "mm_projector_act"), True,
    )

    stages = [
        Stage("input", "video_frames", "input", "video_frames",
              {"shape": video_shape, "patch_size": patch_size,
               "temporal_patch_size": temporal_patch_size, "channels": input_channels},
              step_fields={"shape": video_shape, "channels": input_channels}),
        Stage("embedding", "video_patch_embedding", "temporal_patch_embedding", "temporal_patch_embedding",
              drop_none({"patch_size": patch_size, "temporal_patch_size": temporal_patch_size,
                         "out_features": hidden_size, "grid": patch_grid_geometry(vision_cfg),
                         "ops": embedding_ops or None}),
              step_fields={"patch_size": patch_size, "temporal_patch_size": temporal_patch_size,
                           "out_features": hidden_size}),
        Stage("encoder", "video_encoder", "encode", encoder_kind,
              {"architecture": architecture(vision_cfg), "hidden_size": hidden_size,
               "num_layers": num_layers, "num_attention_heads": num_heads,
               "position_encoding": vision_position_encoding(cfg, vision_cfg)},
              step_fields={"hidden_size": hidden_size, "num_layers": num_layers}),
        Stage("projector", "video_projector", "merge_patches_to_text_width", projector_kind_value,
              drop_none({"in_features": projector_in, "out_features": projector_out,
                         "profile": (projector_profile or {}).get("profile"),
                         "ops": (projector_profile or {}).get("ops")})),
        Stage("tokens", "video_tokens", "emit_grid_token_stream", "grid_video_tokens",
              {"width": text_hidden_size or None, "grid": grid}),
    ]
    return assemble_path(
        "video_to_grid_tokens",
        stages,
        present_paths(cfg, vision_cfg, [
            ("vision_config", vision_cfg),
            ("video_token_id", cfg),
            ("video_token_index", cfg),
            ("vision_start_token_id", cfg),
            ("vision_end_token_id", cfg),
        ]),
    )


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
    """Derive the vision tower's position encoding from config fields only.

    A vision tower positions patches on a 2D grid, so a learned table or RoPE
    here is inherently 2D/axial.  We read the structural signals directly —
    no model-family lookup:

    * ``position_embedding_size`` / ``use_absolute_position_embeddings`` -> learned 2D table
    * ``rope_parameters`` / ``rope_theta`` in the vision config            -> 2D RoPE

    A tower that declares both (e.g. Gemma 4 vision) reports
    ``"learned_2d_plus_rope_2d"``; mllama, which uses neither standard field,
    reports nothing rather than a guessed label.
    """
    # A unified grid stream (Qwen-VL etc.) positions patches with model-level
    # multimodal RoPE — itself a structural property (mrope rope-scaling,
    # spatial-merge, or runtime grid_thw), not a family name.
    if is_unified_grid_stream(cfg, vision_cfg):
        return {"kind": "multimodal_rope"}

    methods: list[str] = []
    learned = (
        first(vision_cfg, "position_embedding_size", "num_positions") is not None
        or bool(first(vision_cfg, "use_absolute_position_embeddings"))
    )
    if learned:
        methods.append("learned_2d")
    if first(vision_cfg, "rope_parameters", "rope_theta", "rope_scaling") is not None:
        methods.append("rope_2d")
    if not methods:
        return None
    rope = first(vision_cfg, "rope_parameters", "rope_scaling")
    return drop_none({
        "kind": "_plus_".join(methods),
        "rope": rope if isinstance(rope, dict) else None,
    })


def image_tiling(cfg: Any, vision_cfg: Any) -> dict | None:
    """Detect how a high-res image is split before encoding.

    Two modes, both config-driven:

    * ``fixed_tiles`` — ``max_num_tiles`` (mllama): up to N fixed-size tiles,
      each encoded independently; ``supported_aspect_ratios`` lists the layouts.
    * ``anyres`` — ``image_grid_pinpoints`` (LLaVA-NeXT / OneVision): the image
      is resized to the best-fitting grid from a list of candidate resolutions.

    Absent -> the model encodes the image as a single field.
    """
    max_tiles = as_int(first(vision_cfg, "max_num_tiles", "max_image_tiles"))
    if max_tiles:
        return drop_none({
            "mode": "fixed_tiles",
            "max_tiles": max_tiles,
            "aspect_ratios": first(vision_cfg, "supported_aspect_ratios"),
        })
    pinpoints = first(cfg, "image_grid_pinpoints")
    if pinpoints:
        return drop_none({
            "mode": "anyres",
            "num_layouts": len(pinpoints) if isinstance(pinpoints, (list, tuple)) else None,
            "aspect_ratio_policy": first(cfg, "vision_aspect_ratio"),
        })
    return None


def token_reduction(cfg: Any, vision_cfg: Any) -> dict | None:
    """Detect a post-encoder token-reduction step from config.

    Two reduction families, both before the projector:

    * ``pooling_kernel_size`` -> k×k average pool (Gemma 4), cuts tokens by k²
    * ``downsample_ratio`` (InternVL, e.g. 0.5) or ``scale_factor`` (Idefics3 /
      SmolVLM, e.g. 3) -> pixel-shuffle / space-to-depth, cuts tokens by 1/r²

    ``spatial_merge_size`` is *not* reported here — for grid streams it is
    already surfaced on the grid and the patch-merger projector.
    """
    pool = as_int(first(vision_cfg, "pooling_kernel_size", "pool_kernel_size"))
    if pool:
        return {
            "operation": "pool_tokens",
            "kind": "token_pooling",
            "fields": {"kernel_size": pool, "reduces_tokens_by": pool * pool},
        }
    ratio = first(cfg, "downsample_ratio")
    if ratio:
        try:
            factor = round((1.0 / float(ratio)) ** 2)
        except (TypeError, ValueError, ZeroDivisionError):
            factor = None
        return {
            "operation": "pixel_shuffle",
            "kind": "pixel_shuffle",
            "fields": drop_none({"downsample_ratio": ratio, "reduces_tokens_by": factor}),
        }
    scale = as_int(first(cfg, "scale_factor", "pixel_shuffle_factor"))
    if scale:
        return {
            "operation": "pixel_shuffle",
            "kind": "pixel_shuffle",
            "fields": {"scale_factor": scale, "reduces_tokens_by": scale * scale},
        }
    return None


def multilayer_features(vision_cfg: Any) -> dict | None:
    """Detect *multi-layer* feature concatenation from config.

    Signal: ``intermediate_layers_indices`` or a list-valued
    ``vision_feature_layers`` — the encoder concatenates hidden states from
    several layers (mllama joins [3,7,15,23,30]+final, hence
    ``vision_output_dim`` >> hidden).  A single ``vision_feature_layer`` (LLaVA)
    is *not* a concat — that is handled by :func:`feature_selection`.
    """
    layers = first(vision_cfg, "intermediate_layers_indices", "vision_feature_layers")
    if not isinstance(layers, (list, tuple)) or len(layers) <= 1:
        return None
    return drop_none({
        "layers": list(layers),
        "output_dim": first(vision_cfg, "vision_output_dim", "output_dim"),
    })


def feature_selection(cfg: Any) -> dict | None:
    """Which encoder output the connector consumes (LLaVA-style).

    * ``vision_feature_layer`` — single hidden-state layer (e.g. -2 = penultimate)
    * ``vision_feature_select_strategy`` — ``"default"`` drops the CLS token,
      ``"full"`` keeps every patch token
    """
    layer = first(cfg, "vision_feature_layer")
    strategy = first(cfg, "vision_feature_select_strategy")
    if layer is None and strategy is None:
        return None
    if isinstance(layer, (list, tuple)):  # a list is multi-layer concat, not single select
        layer = None
    return drop_none({"layer": layer, "select_strategy": strategy})


def projector_kind(cfg: Any) -> str:
    """Classify the vision connector from structural config fields.

    Priority reflects how distinctive each mechanism is:
    perceiver resampler -> patch merger (grid) -> declared projector type ->
    MLP (has an activation) -> plain linear.
    """
    vision_cfg = first(cfg, "vision_config", "vision_model_config")
    if first(cfg, "perceiver_config", "use_resampler", "resampler_config") is not None:
        return "perceiver_resampler"
    if is_unified_grid_stream(cfg, vision_cfg):
        return "patch_merger"
    raw = first(cfg, "mm_projector_type", "projector_type", "multi_modal_projector_type")
    if raw:
        return str(raw)
    if first(cfg, "projector_hidden_act", "mm_projector_act"):
        return "mlp_projector"
    return "linear_projector"


def perceiver_latents(cfg: Any) -> int | None:
    """Fixed query/token count for a perceiver-resampler connector (Idefics2)."""
    pc = nested(cfg, "perceiver_config") or nested(cfg, "resampler_config") or {}
    return as_int(first(pc, "resampler_n_latents", "n_latents", "num_latents", "num_queries"))


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
