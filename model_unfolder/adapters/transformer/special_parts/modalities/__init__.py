"""Model-level multimodal pathway extraction.

This module keeps modality structure out of the transformer layer parser.  The
decoder stack is still parsed from the text config; multimodal inputs are
recorded as top-level pathways that eventually feed the stack input.
"""
from __future__ import annotations

from typing import Any

from ...common import get_config_value as _g


def multimodal_extras(cfg: Any, text_cfg: Any, text_hidden_size: int) -> dict | None:
    """Return structured multimodal extras, if the config declares them."""
    vision_cfg = _nested(cfg, "vision_config") or _nested(cfg, "vision_model_config")
    audio_cfg = _nested(cfg, "audio_config") or _nested(cfg, "audio_model_config")

    modalities: dict[str, Any] = {}
    if vision_cfg is not None:
        modalities["vision"] = _vision_path(cfg, vision_cfg, text_hidden_size)
    if audio_cfg is not None:
        modalities["audio"] = _audio_path(cfg, audio_cfg, text_hidden_size)

    if not modalities:
        return None

    fusion = _fusion(cfg, text_cfg, modalities, text_hidden_size)
    return {"modalities": {"inputs": modalities, "fusion": fusion}}


def _vision_path(cfg: Any, vision_cfg: Any, text_hidden_size: int) -> dict:
    image_size = _first(vision_cfg, "image_size", "input_size")
    patch_size = _first(vision_cfg, "patch_size", "patch_size_h")
    hidden_size = _first(vision_cfg, "hidden_size", "vision_hidden_size", "width")
    num_layers = _first(vision_cfg, "num_hidden_layers", "num_layers", "depth")
    num_heads = _first(vision_cfg, "num_attention_heads", "num_heads")
    token_count = _visual_token_count(cfg, vision_cfg)
    encoder_kind = _vision_encoder_kind(cfg, vision_cfg)
    projector_kind = _projector_kind(cfg)
    projector_activation = _first(cfg, "projector_hidden_act", "mm_projector_act")
    projector_out = text_hidden_size or _first(cfg, "projection_dim", "text_hidden_size")

    return _drop_none({
        "kind": "image_to_soft_visual_tokens",
        "input": _drop_none({
            "kind": "image_pixels",
            "shape": ["batch", "images", "channels", "height", "width"],
            "image_size": image_size,
            "patch_size": patch_size,
        }),
        "embedding": _drop_none({
            "kind": "patch_embedding",
            "patch_size": patch_size,
            "out_features": hidden_size,
        }),
        "encoder": _drop_none({
            "kind": encoder_kind,
            "architecture": _architecture(vision_cfg),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_heads,
            "patch_size": patch_size,
            "position_encoding": _vision_position_encoding(cfg, vision_cfg),
        }),
        "projector": _drop_none({
            "kind": projector_kind,
            "in_features": hidden_size,
            "out_features": projector_out,
            "activation": projector_activation,
        }),
        "tokens": _drop_none({
            "kind": "soft_visual_tokens",
            "count": token_count,
            "count_options": _token_count_options(cfg),
            "width": text_hidden_size or None,
        }),
        "pipeline": _drop_none([
            {
                "id": "image_pixels",
                "operation": "input",
                "kind": "image_pixels",
                "shape": ["batch", "images", "channels", "height", "width"],
            },
            {
                "id": "patch_embedding",
                "operation": "patch_embedding",
                "patch_size": patch_size,
                "out_features": hidden_size,
            },
            {
                "id": "vision_encoder",
                "operation": "encode",
                "kind": encoder_kind,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
            },
            {
                "id": "projector",
                "operation": "project_to_text_width",
                "kind": projector_kind,
                "in_features": hidden_size,
                "out_features": projector_out,
                "activation": projector_activation,
            },
            {
                "id": "soft_visual_tokens",
                "operation": "emit_soft_token_stream",
                "kind": "soft_visual_tokens",
                "count": token_count,
                "width": text_hidden_size or None,
            },
        ]),
        "trace": {
            "config_paths": _present_paths(cfg, vision_cfg, [
                ("vision_config", vision_cfg),
                ("image_seq_length", cfg),
                ("image_token_id", cfg),
                ("image_token_index", cfg),
                ("mm_projector_type", cfg),
            ]),
        },
    })


def _audio_path(cfg: Any, audio_cfg: Any, text_hidden_size: int) -> dict:
    hidden_size = _first(audio_cfg, "hidden_size", "conf_hidden_size", "d_model", "encoder_dim")
    num_layers = _first(audio_cfg, "num_hidden_layers", "conf_num_hidden_layers", "encoder_layers", "num_layers", "depth")
    num_heads = _first(audio_cfg, "num_attention_heads", "conf_num_attention_heads", "encoder_attention_heads", "num_heads")
    feature_size = _first(audio_cfg, "input_feat_size", "feature_size", "num_mel_bins")
    projector_out = text_hidden_size or _first(audio_cfg, "output_proj_dims", "projection_dim")
    projector_kind = _audio_projector_kind(audio_cfg)
    token_count = _audio_token_count(cfg, audio_cfg)
    ms_per_token = _first(cfg, "audio_ms_per_token", "audio_token_ms")
    encoder_kind = _audio_encoder_kind(cfg, audio_cfg)
    return _drop_none({
        "kind": "audio_to_soft_tokens",
        "input": _drop_none({
            "kind": "audio_features",
            "shape": ["batch", "segments", "frames", "features"],
            "feature_size": feature_size,
        }),
        "encoder": _drop_none({
            "kind": encoder_kind,
            "architecture": _architecture(audio_cfg),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_heads,
        }),
        "projector": _drop_none({
            "kind": projector_kind,
            "in_features": hidden_size,
            "out_features": projector_out,
        }),
        "tokens": _drop_none({
            "kind": "soft_audio_tokens",
            "count": token_count,
            "ms_per_token": ms_per_token,
            "width": text_hidden_size or None,
        }),
        "pipeline": _drop_none([
            {
                "id": "audio_features",
                "operation": "input",
                "kind": "audio_features",
                "shape": ["batch", "segments", "frames", "features"],
                "feature_size": feature_size,
            },
            {
                "id": "audio_encoder",
                "operation": "encode",
                "kind": encoder_kind,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
            },
            {
                "id": "audio_projector",
                "operation": "project_to_text_width",
                "kind": projector_kind,
                "in_features": hidden_size,
                "out_features": projector_out,
            },
            {
                "id": "soft_audio_tokens",
                "operation": "emit_soft_token_stream",
                "kind": "soft_audio_tokens",
                "count": token_count,
                "ms_per_token": ms_per_token,
                "width": text_hidden_size or None,
            },
        ]),
        "trace": {
            "config_paths": _present_paths(cfg, audio_cfg, [
                ("audio_config", audio_cfg),
                ("audio_token_id", cfg),
                ("audio_token_index", cfg),
                ("audio_soft_tokens_per_image", cfg),
                ("audio_ms_per_token", cfg),
                ("boa_token_id", cfg),
                ("eoa_token_id", cfg),
            ]),
        },
    })


def _fusion(cfg: Any, text_cfg: Any, modalities: dict[str, Any], text_hidden_size: int) -> dict:
    kind = _fusion_kind(cfg)
    placeholders = _placeholders(cfg)
    placeholder = placeholders.get("image") or placeholders.get("audio")
    source_ids = ["io.token_embedding"]
    if "vision" in modalities:
        source_ids.append("modalities.inputs.vision.tokens")
    if "audio" in modalities:
        source_ids.append("modalities.inputs.audio.tokens")

    return _drop_none({
        "kind": kind,
        "operation": _fusion_operation(kind),
        "sources": source_ids,
        "target": "stack.input_embeddings",
        "placeholder": placeholder,
        "placeholders": placeholders or None,
        "mechanism": _fusion_mechanism(kind, placeholders, modalities),
        "output": _drop_none({
            "kind": "mixed_embeddings" if kind != "cross_attention" else "decoder_hidden_states",
            "width": text_hidden_size or _first(text_cfg, "hidden_size"),
        }),
        "trace": {
            "config_paths": _present_paths(cfg, text_cfg, [
                ("image_token_id", cfg),
                ("image_token_index", cfg),
                ("image_seq_length", cfg),
                ("audio_token_id", cfg),
                ("audio_token_index", cfg),
                ("audio_soft_tokens_per_image", cfg),
                ("audio_ms_per_token", cfg),
                ("boa_token_id", cfg),
                ("eoa_token_id", cfg),
                ("cross_attention_layers", cfg),
                ("cross_attention_frequency", cfg),
            ]),
        },
    })


def _fusion_operation(kind: str) -> str:
    return {
        "placeholder_replace": "scatter_soft_tokens_into_placeholder_slots",
        "prefix_soft_tokens": "prepend_soft_tokens",
        "unified_multimodal_stream": "interleave_modal_tokens",
        "cross_attention": "condition_decoder_hidden_states",
    }.get(kind, "fuse_modal_token_streams")


def _fusion_mechanism(kind: str, placeholders: dict[str, dict], modalities: dict[str, Any]) -> dict | None:
    if kind == "placeholder_replace":
        routes = []
        if "vision" in modalities:
            routes.append(_drop_none({
                "kind": "scatter",
                "source": "modalities.inputs.vision.tokens",
                "into": "io.token_embedding",
                "at": placeholders.get("image"),
            }))
        if "audio" in modalities:
            routes.append(_drop_none({
                "kind": "scatter",
                "source": "modalities.inputs.audio.tokens",
                "into": "io.token_embedding",
                "at": placeholders.get("audio"),
            }))
        if len(routes) == 1:
            return routes[0]
        if routes:
            return {"kind": "scatter_many", "routes": routes}
    if kind == "prefix_soft_tokens" and "vision" in modalities:
        return {"kind": "prefix", "source": "modalities.inputs.vision.tokens", "before": "io.token_embedding"}
    if kind == "unified_multimodal_stream":
        return {"kind": "interleave", "sources": list(modalities.keys())}
    if kind == "cross_attention":
        return {"kind": "cross_attention", "sources": list(modalities.keys())}
    return None


def _fusion_kind(cfg: Any) -> str:
    model_type = str(_g(cfg, "model_type", "") or "").lower()
    arch = " ".join(_g(cfg, "architectures", []) or []).lower()
    if model_type in {"mllama", "llama4"} or "mllama" in arch:
        return "cross_attention"
    if model_type.startswith(("qwen2_vl", "qwen2_5_vl", "qwen3_vl")):
        return "unified_multimodal_stream"
    if model_type == "paligemma":
        return "prefix_soft_tokens"
    if _placeholders(cfg):
        return "placeholder_replace"
    return "modality_token_fusion"


def _placeholder(cfg: Any) -> dict | None:
    token_id = _first(cfg, "image_token_id", "image_token_index")
    token = _first(cfg, "image_token", "image_token_string")
    if token_id is None and token is None:
        return None
    return _drop_none({"kind": "image_placeholder", "token_id": token_id, "token": token})


def _placeholders(cfg: Any) -> dict[str, dict]:
    placeholders: dict[str, dict] = {}
    image = _placeholder(cfg)
    if image:
        image.update(_drop_none({
            "begin_token_id": _first(cfg, "boi_token_id", "image_boi_token_id"),
            "end_token_id": _first(cfg, "eoi_token_id", "image_eoi_token_id"),
        }))
        placeholders["image"] = image

    audio_token_id = _first(cfg, "audio_token_id", "audio_token_index")
    audio_token = _first(cfg, "audio_token", "audio_token_string")
    if audio_token_id is not None or audio_token is not None:
        placeholders["audio"] = _drop_none({
            "kind": "audio_placeholder",
            "token_id": audio_token_id,
            "token": audio_token,
            "begin_token_id": _first(cfg, "boa_token_id", "audio_boa_token_id"),
            "end_token_id": _first(cfg, "eoa_token_id", "audio_eoa_token_id"),
        })
    return placeholders


def _vision_encoder_kind(cfg: Any, vision_cfg: Any) -> str:
    model_type = str(_g(cfg, "model_type", "") or "").lower()
    vision_type = str(_g(vision_cfg, "model_type", "") or "").lower()
    arch = " ".join(_g(vision_cfg, "architectures", []) or []).lower()
    if model_type == "gemma4" or "gemma4" in arch:
        return "gemma4_vision"
    if model_type.startswith(("qwen2_vl", "qwen2_5_vl", "qwen3_vl")):
        return "qwen_vl_vision_transformer"
    if model_type == "pixtral" or "pixtral" in arch:
        return "pixtral_vision_transformer"
    if "siglip" in vision_type or "siglip" in arch:
        return "siglip_vision_transformer"
    return "vision_transformer"


def _audio_encoder_kind(cfg: Any, audio_cfg: Any) -> str:
    model_type = str(_g(cfg, "model_type", "") or "").lower()
    audio_type = str(_g(audio_cfg, "model_type", "") or "").lower()
    arch = " ".join(_g(audio_cfg, "architectures", []) or []).lower()
    if model_type == "gemma4" or "gemma4" in audio_type or "gemma4" in arch:
        return "gemma4_audio"
    return "audio_encoder"


def _vision_position_encoding(cfg: Any, vision_cfg: Any) -> dict | None:
    model_type = str(_g(cfg, "model_type", "") or "").lower()
    if model_type == "gemma4":
        return {"kind": "learned_2d_plus_rope_2d"}
    if model_type.startswith(("qwen2_vl", "qwen2_5_vl", "qwen3_vl")):
        return {"kind": "multimodal_rope"}
    if _g(vision_cfg, "use_absolute_position_embeddings") is not None:
        return {"kind": "learned_absolute"}
    return None


def _projector_kind(cfg: Any) -> str:
    raw = _first(cfg, "mm_projector_type", "projector_type", "multi_modal_projector_type")
    if raw:
        return str(raw)
    if _first(cfg, "projector_hidden_act", "mm_projector_act"):
        return "mlp_projector"
    return "linear_projector"


def _audio_projector_kind(audio_cfg: Any) -> str:
    raw = _first(audio_cfg, "projector_type", "multi_modal_projector_type")
    return str(raw) if raw else "linear_projector"


def _visual_token_count(cfg: Any, vision_cfg: Any) -> int | None:
    direct = _first(
        cfg,
        "image_seq_length",
        "num_image_tokens",
        "mm_tokens_per_image",
        "vision_soft_tokens_per_image",
        "tokens_per_image",
    )
    if direct is not None:
        return direct
    image_size = _first(vision_cfg, "image_size")
    patch_size = _first(vision_cfg, "patch_size")
    if image_size and patch_size:
        try:
            return int((int(image_size) // int(patch_size)) ** 2)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def _token_count_options(cfg: Any) -> list[int] | None:
    for key in ("image_token_count_options", "soft_token_count_options", "tokens_per_image_options"):
        value = _g(cfg, key)
        if isinstance(value, (list, tuple)) and value:
            return [int(v) for v in value if v is not None]
    return None


def _audio_token_count(cfg: Any, audio_cfg: Any) -> int | None:
    return _first(
        cfg,
        "audio_seq_length",
        "audio_soft_tokens_per_image",
        "audio_tokens_per_audio",
        "num_audio_tokens",
        "audio_token_count",
        "soft_audio_token_count",
    ) or _first(audio_cfg, "audio_seq_length", "num_audio_tokens")


def _architecture(cfg: Any) -> str | None:
    architectures = _g(cfg, "architectures") or []
    if architectures:
        return architectures[0]
    model_type = _g(cfg, "model_type")
    return str(model_type) if model_type else None


def _nested(cfg: Any, key: str) -> Any:
    value = _g(cfg, key)
    return value if isinstance(value, dict) or value is not None else None


def _first(cfg: Any, *keys: str) -> Any:
    for key in keys:
        value = _g(cfg, key)
        if value is not None:
            return value
    return None


def _present_paths(_root_cfg: Any, nested_cfg: Any, entries: list[tuple[str, Any]]) -> list[str]:
    paths: list[str] = []
    for key, cfg in entries:
        if key in {"vision_config", "audio_config"} and nested_cfg is not None:
            paths.append(key)
        elif _g(cfg, key) is not None:
            paths.append(key)
    return paths


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value if v is not None]
    return value


__all__ = ["multimodal_extras"]
