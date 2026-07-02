"""Model-level modality fusion extraction."""
from __future__ import annotations

from typing import Any

from .accessors import drop_none, first, present_paths
from .detect import cross_attention_layers, placeholders
from .vision import grid_runtime_inputs


def fusion_path(cfg: Any, text_cfg: Any, modalities: dict[str, Any], text_hidden_size: int) -> dict:
    """Return semantic facts for how modality tokens/states meet the decoder."""
    kind = "code_defined_fusion"
    placeholder_map = placeholders(cfg)
    placeholder = placeholder_map.get("image") or placeholder_map.get("audio")
    source_ids = ["io.token_embedding"]
    if "vision" in modalities:
        source_ids.append("modalities.inputs.vision.tokens")
    if "video" in modalities:
        source_ids.append("modalities.inputs.video.tokens")
    if "audio" in modalities:
        source_ids.append("modalities.inputs.audio.tokens")

    return drop_none({
        "kind": kind,
        "operation": "unknown",
        "sources": source_ids,
        "target": "unknown",
        "placeholder": placeholder,
        "placeholders": placeholder_map or None,
        "mechanism": None,
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


def apply_fusion_evidence(payload: dict | None, evidence, cfg: Any,
                          text_cfg: Any) -> dict | None:
    """Make wrapper evidence authoritative for every fusion projection."""
    if not payload or evidence is None:
        return payload
    modalities_root = payload.get("modalities") or {}
    inputs = modalities_root.get("inputs") or {}
    fusion = modalities_root.get("fusion") or {}
    fusion["source_evidence"] = evidence.to_dict()
    fusion["source_owner"] = evidence.owner_class
    fusion["source_component"] = evidence.component
    if evidence.status != "proven":
        fusion.update({
            "kind": "code_defined_fusion", "operation": "unknown",
            "target": "unknown", "mechanism": None,
        })
        return payload

    route_modalities = [route.modality for route in evidence.routes
                        if route.modality in inputs]
    source_ids = ["io.token_embedding", *(
        f"modalities.inputs.{name}.tokens" for name in route_modalities
    )]
    cross_layers = cross_attention_layers(cfg, text_cfg)
    fusion.update({
        "kind": evidence.kind,
        "operation": evidence.operation,
        "sources": source_ids,
        "target": ("decoder.cross_attention_layers"
                   if evidence.kind == "cross_attention" else "stack.input_embeddings"),
        "mechanism": _evidence_mechanism(
            evidence, fusion.get("placeholders") or {}, inputs, cross_layers,
        ),
    })
    output = fusion.get("output") or {}
    output["kind"] = ("decoder_hidden_states" if evidence.kind == "cross_attention"
                      else "mixed_embeddings")
    fusion["output"] = output
    return payload


def _evidence_mechanism(evidence, placeholder_map: dict[str, dict],
                        modalities: dict[str, Any],
                        cross_layers: list[int] | None) -> dict | None:
    names = [route.modality for route in evidence.routes if route.modality in modalities]
    if evidence.kind == "placeholder_replace":
        routes = [drop_none({
            "kind": "scatter", "operation": route.operation,
            "source": f"modalities.inputs.{route.modality}.tokens",
            "into": "io.token_embedding",
            "at": placeholder_map.get("image" if route.modality == "vision" else route.modality),
        }) for route in evidence.routes if route.modality in modalities]
        if len(routes) == 1:
            return routes[0]
        return {"kind": "scatter_many", "routes": routes} if routes else None
    if evidence.kind == "unified_multimodal_stream":
        return drop_none({
            "kind": "grid_placeholder_replace",
            "operation": evidence.operation,
            "sources": names,
            "position_encoding": "multimodal_rope" if evidence.grid_positions else None,
            "runtime_grid_inputs": grid_runtime_inputs(modalities),
        })
    if evidence.kind == "cross_attention":
        return drop_none({
            "kind": "cross_attention", "operation": "cross_attention_states",
            "sources": names, "layers": cross_layers,
            "num_layers": len(cross_layers) if cross_layers else None,
        })
    if evidence.kind == "prefix_soft_tokens":
        name = names[0] if names else None
        return ({"kind": "prefix", "operation": "prefix_concat",
                 "source": f"modalities.inputs.{name}.tokens",
                 "before": "io.token_embedding"} if name else None)
    return None


__all__ = ["apply_fusion_evidence", "fusion_path"]
