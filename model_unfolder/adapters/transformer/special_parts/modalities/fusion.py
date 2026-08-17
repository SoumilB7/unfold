"""Model-level modality fusion extraction."""
from __future__ import annotations

from typing import Any

from .accessors import drop_none, present_paths
from .detect import placeholders


def fusion_path(cfg: Any, text_cfg: Any, modalities: dict[str, Any], text_hidden_size: int) -> dict:
    """Return semantic facts for how modality tokens/states meet the decoder."""
    kind = "code_defined_fusion"
    target = "unknown"
    placeholder_map = placeholders(cfg)
    placeholder = placeholder_map.get("image") or placeholder_map.get("audio")
    return drop_none({
        "kind": kind,
        "operation": "unknown",
        # Declared lanes are addresses, not proof that their values participate
        # in the model's fusion route.  Exact wrapper evidence installs sources.
        "sources": [],
        "target": target,
        "placeholder": placeholder,
        "placeholders": placeholder_map or None,
        "mechanism": None,
        "output": {"kind": "code_defined_fusion_output"},
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


def apply_fusion_evidence(
        payload: dict | None, evidence, *, cross_layers=None,
        multiaxis_result=None) -> dict | None:
    """Make wrapper evidence authoritative for every fusion projection."""
    if not payload:
        return payload
    modalities_root = payload.get("modalities") or {}
    inputs = modalities_root.get("inputs") or {}
    fusion = modalities_root.get("fusion") or {}
    # Fusion vocabulary in the config shell is context/geometry only.  The
    # source route is the sole mechanism authority, including the negative
    # case: no evidence means an opaque fusion node, never a retained config
    # guess.
    fusion.update({
        "kind": "code_defined_fusion", "operation": "unknown",
        "target": "unknown", "mechanism": None, "sources": [],
    })
    fusion["output"] = {"kind": "code_defined_fusion_output"}
    _clear_route_claims(inputs)
    for key in ("source_component", "source_evidence", "source_owner"):
        fusion.pop(key, None)
    if evidence is None:
        return payload
    fusion["source_evidence"] = evidence.to_dict()
    fusion["source_owner"] = evidence.owner_class
    fusion["source_component"] = evidence.component
    if evidence.status != "proven":
        return payload

    route_modalities = [route.modality for route in evidence.routes
                        if route.modality in inputs]
    source_ids = ["io.token_embedding", *(
        f"modalities.inputs.{name}.tokens" for name in route_modalities
    )]
    multiaxis = bool(
        getattr(multiaxis_result, "status", "") == "resolved"
        and getattr(multiaxis_result, "value", ()))
    kind = (
        "unified_multimodal_stream"
        if evidence.kind == "placeholder_replace" and multiaxis
        else evidence.kind)
    operation = (
        "scatter_grid_tokens_into_placeholder_slots"
        if kind == "unified_multimodal_stream"
        else evidence.operation)
    fusion.update({
        "kind": kind,
        "operation": operation,
        "sources": source_ids,
        "target": ("decoder.cross_attention_layers"
                   if kind == "cross_attention" else "stack.input_embeddings"),
        "mechanism": _evidence_mechanism(
            evidence, fusion.get("placeholders") or {}, inputs, cross_layers,
            multiaxis=multiaxis,
        ),
    })
    output = fusion.get("output") or {}
    output["kind"] = ("decoder_hidden_states" if kind == "cross_attention"
                      else "mixed_embeddings")
    fusion["output"] = output
    _project_route_claims(inputs, evidence, multiaxis=multiaxis)
    return payload


def _clear_route_claims(inputs: dict[str, Any]) -> None:
    """Remove route vocabulary until the wrapper source proves a route."""
    for path in inputs.values():
        if not isinstance(path, dict):
            continue
        path["kind"] = "code_defined_modality_path"
        tokens = path.get("tokens")
        if isinstance(tokens, dict):
            tokens["kind"] = "code_defined_tokens"
        pipeline = path.get("pipeline") or ()
        if pipeline:
            pipeline[-1]["kind"] = "code_defined_tokens"
            pipeline[-1]["operation"] = "unknown"


def _project_route_claims(inputs: dict[str, Any], evidence, *,
                          multiaxis: bool) -> None:
    """Project wrapper-proven route semantics into each exact modality lane.

    These labels are presentation of the already-proven fusion relation.  They
    never inspect model/config/class vocabulary.  Multi-axis position evidence
    can refine a proven placeholder route for visual lanes, but cannot create
    a fusion route by itself.
    """
    for route in evidence.routes:
        path = inputs.get(route.modality)
        if not isinstance(path, dict):
            continue
        if evidence.kind == "cross_attention":
            path_kind = (
                "image_to_cross_attention_states"
                if route.modality == "vision" else
                "prompt_to_cross_attention_states"
                if route.modality == "conditioning" else
                f"{route.modality}_to_cross_attention_states")
            token_kind = ("vision_cross_attention_states"
                          if route.modality == "vision"
                          else "cross_attention_states")
            operation = "emit_cross_attention_states"
        elif evidence.kind == "placeholder_replace" and multiaxis \
                and route.modality in {"vision", "video"}:
            path_kind = ("image_to_grid_tokens" if route.modality == "vision"
                         else "video_to_grid_tokens")
            token_kind = ("grid_visual_tokens" if route.modality == "vision"
                          else "grid_video_tokens")
            operation = "emit_grid_token_stream"
        elif evidence.kind in {"placeholder_replace", "prefix_soft_tokens"}:
            path_kind = (
                "image_to_soft_visual_tokens"
                if route.modality == "vision" else
                f"{route.modality}_to_soft_tokens")
            token_kind = ("soft_visual_tokens" if route.modality == "vision"
                          else f"soft_{route.modality}_tokens")
            operation = "emit_soft_token_stream"
        else:
            continue
        path["kind"] = path_kind
        tokens = path.get("tokens")
        if isinstance(tokens, dict):
            tokens["kind"] = token_kind
        pipeline = path.get("pipeline") or ()
        if pipeline:
            pipeline[-1]["kind"] = token_kind
            pipeline[-1]["operation"] = operation


def _evidence_mechanism(evidence, placeholder_map: dict[str, dict],
                        modalities: dict[str, Any],
                        cross_layers: list[int] | None, *,
                        multiaxis: bool = False) -> dict | None:
    names = [route.modality for route in evidence.routes if route.modality in modalities]
    if evidence.kind == "placeholder_replace" and multiaxis:
        return drop_none({
            "kind": "grid_placeholder_replace",
            "operation": "scatter_grid_tokens_into_placeholder_slots",
            "sources": names,
            "position_encoding": "multimodal_rope",
        })
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
