"""Model-level modality fusion extraction."""
from __future__ import annotations

from typing import Any

from .accessors import drop_none, first, present_paths
from .detect import cross_attention_layers, fusion_kind, placeholders
from .vision import grid_runtime_inputs


def fusion_path(cfg: Any, text_cfg: Any, modalities: dict[str, Any], text_hidden_size: int) -> dict:
    """Return semantic facts for how modality tokens/states meet the decoder."""
    vision_cfg = first(cfg, "vision_config", "vision_model_config")
    kind = fusion_kind(cfg, text_cfg, vision_cfg)
    placeholder_map = placeholders(cfg)
    placeholder = placeholder_map.get("image") or placeholder_map.get("audio")
    cross_layers = cross_attention_layers(cfg, text_cfg)
    source_ids = ["io.token_embedding"]
    if "vision" in modalities:
        source_ids.append("modalities.inputs.vision.tokens")
    if "video" in modalities:
        source_ids.append("modalities.inputs.video.tokens")
    if "audio" in modalities:
        source_ids.append("modalities.inputs.audio.tokens")

    return drop_none({
        "kind": kind,
        "operation": fusion_operation(kind),
        "sources": source_ids,
        "target": "decoder.cross_attention_layers" if kind == "cross_attention" else "stack.input_embeddings",
        "placeholder": placeholder,
        "placeholders": placeholder_map or None,
        "mechanism": fusion_mechanism(kind, placeholder_map, modalities, cross_layers),
        "output": drop_none({
            "kind": "mixed_embeddings" if kind != "cross_attention" else "decoder_hidden_states",
            "width": text_hidden_size or first(text_cfg, "hidden_size"),
        }),
        "trace": {
            "config_paths": present_paths(cfg, text_cfg, [
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
                ("cross_attention_layers", text_cfg),
                ("cross_attention_frequency", cfg),
                ("cross_attention_frequency", text_cfg),
            ]),
        },
    })


def fusion_operation(kind: str) -> str:
    """Return the low-level operation name for a fusion kind."""
    return {
        "placeholder_replace": "scatter_soft_tokens_into_placeholder_slots",
        "prefix_soft_tokens": "prepend_soft_tokens",
        "unified_multimodal_stream": "interleave_modal_tokens",
        "cross_attention": "condition_decoder_hidden_states",
    }.get(kind, "fuse_modal_token_streams")


def fusion_mechanism(
    kind: str,
    placeholder_map: dict[str, dict],
    modalities: dict[str, Any],
    cross_layers: list[int] | None = None,
) -> dict | None:
    """Return modality-specific fusion mechanism facts."""
    if kind == "placeholder_replace":
        routes = []
        if "vision" in modalities:
            routes.append(drop_none({
                "kind": "scatter",
                "source": "modalities.inputs.vision.tokens",
                "into": "io.token_embedding",
                "at": placeholder_map.get("image"),
            }))
        if "audio" in modalities:
            routes.append(drop_none({
                "kind": "scatter",
                "source": "modalities.inputs.audio.tokens",
                "into": "io.token_embedding",
                "at": placeholder_map.get("audio"),
            }))
        if len(routes) == 1:
            return routes[0]
        if routes:
            return {"kind": "scatter_many", "routes": routes}
    if kind == "prefix_soft_tokens" and "vision" in modalities:
        return {"kind": "prefix", "source": "modalities.inputs.vision.tokens", "before": "io.token_embedding"}
    if kind == "unified_multimodal_stream":
        return drop_none({
            "kind": "interleave_grid_streams",
            "sources": list(modalities.keys()),
            "position_encoding": "multimodal_rope",
            "runtime_grid_inputs": grid_runtime_inputs(modalities),
        })
    if kind == "cross_attention":
        return drop_none({
            "kind": "cross_attention",
            "sources": list(modalities.keys()),
            "layers": cross_layers,
            "num_layers": len(cross_layers) if cross_layers else None,
        })
    return None


__all__ = ["fusion_path"]

