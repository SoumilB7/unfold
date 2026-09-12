"""Pure U9 typed-evidence projection into modality path specifications.

This module performs no source/config discovery.  It receives the closed U9
component result and copies only proven mechanism facts into the pre-existing
geometry shell.  Missing evidence removes old structural claims instead of
reconstructing them from modality, class, or config vocabulary.
"""
from __future__ import annotations

from copy import deepcopy

from .....evidence.component_tower import (
    ComponentTowerMechanisms,
    RecursiveComponentMechanisms,
)
from .....evidence.wrapper_features import WrapperFeatureRoute
from .registry import MODALITY_REGISTRY
from .schema import tower_submodel_spec


_COMPONENT_TO_MODALITY = {
    key: spec.name
    for spec in MODALITY_REGISTRY
    for key in spec.config_keys
}


def apply_recursive_component_evidence(
    payload: dict | None,
    evidence: RecursiveComponentMechanisms,
) -> dict | None:
    """Overlay exact component facts; never select a mechanism by address."""
    if not payload:
        return payload
    inputs = (payload.get("modalities") or {}).get("inputs") or {}
    # ``multimodal_extras`` still builds a geometry shell from checkpoint
    # values.  Before any source projection, remove every architectural claim
    # that the legacy shell used to infer from config vocabulary.  A later
    # evidence projector may add the claim back, but source-missing components
    # now stay visibly unknown instead of retaining a plausible old answer.
    for modality, path in inputs.items():
        if isinstance(path, dict):
            _clear_legacy_component_claims(path, modality=modality)
    if not isinstance(evidence, RecursiveComponentMechanisms):
        return payload
    by_component = {}
    for tower in evidence.towers:
        by_component.setdefault(tower.component.component_key, []).append(tower)
    for towers in by_component.values():
        key = towers[0].component.config_path[-1]
        modality = _COMPONENT_TO_MODALITY.get(key)
        if modality is None:
            continue
        names = ("vision", "video") if modality == "vision" else (modality,)
        for name in names:
            path = inputs.get(name)
            if isinstance(path, dict):
                _apply_towers(path, tuple(towers), modality=name)
    return payload


_SOURCE_ONLY_ENCODER_KEYS = frozenset({
    "activation",
    "architecture",
    "attention_class",
    "attention_kind",
    "attention_position_kind",
    "attention_window",
    "embedding_scaled",
    "feature_operations",
    "ffn_class",
    "ffn_gated",
    "ffn_projection_mode",
    "final_norm_kind",
    "frontend_ops",
    "full_attention_block_indexes",
    "global_head_dim",
    "head_dim",
    "hidden_size",
    "input_position_kind",
    "intermediate_layers_indices",
    "intermediate_size",
    "k_norm",
    "norm_kind",
    "norm_placement",
    "num_attention_heads",
    "num_channels",
    "num_global_layers",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_layers",
    "output_dim",
    "patch_size",
    "position_encoding",
    "post_ops",
    "post_rope_scale",
    "projection_mode",
    "q_norm",
    "residual_gated",
    "sub_model",
    "v_norm",
    "variants",
    "source_owners",
    "video_tokens_per_second",
    "max_source_positions",
})


def _clear_legacy_component_claims(path: dict, *, modality: str) -> None:
    """Demote the old config-derived mechanism shell to geometry only.

    ``multimodal_extras`` still needs checkpoint dimensions to create a stable
    navigable lane.  It does *not* get to name the lane's execution mechanism:
    path, embedding, encoder, output-token, and pipeline kinds are reset here
    before any exact source projector runs.  ``modality`` is presentation
    context only and deliberately does not select a mechanism label.
    """
    path["kind"] = "code_defined_modality_path"
    encoder = path.get("encoder")
    if isinstance(encoder, dict):
        for key in _SOURCE_ONLY_ENCODER_KEYS:
            encoder.pop(key, None)
        encoder["kind"] = "code_defined_encoder"
    embedding = path.get("embedding")
    if isinstance(embedding, dict):
        for key in tuple(embedding):
            if key == "kind":
                continue
            embedding.pop(key, None)
        embedding["kind"] = "code_defined_embedding"
    tokens = path.get("tokens")
    if isinstance(tokens, dict):
        for key in tuple(tokens):
            if key == "kind":
                continue
            tokens.pop(key, None)
        tokens["kind"] = "code_defined_tokens"
    input_spec = path.get("input")
    if isinstance(input_spec, dict):
        # The declared lane and its symbolic tensor axes are presentation
        # context.  Numeric patch/image/channel/rate geometry is an
        # architectural operand and may return only through an exact source
        # reader that proves which operation consumes it.
        for key in tuple(input_spec):
            if key not in {"kind", "shape"}:
                input_spec.pop(key, None)
    path.pop("reduction", None)
    path.pop("tiling", None)
    path.pop("source_evidence", None)
    # Tiling and reduction stages were authored solely by config predicates.
    # Exact component operations can reuse/add a reduction stage later; an
    # unproven stage must not survive merely because the shell pre-created it.
    path["pipeline"] = [
        step for step in (path.get("pipeline") or ())
        if step.get("id") not in {"vision_tiles", "vision_token_reduce"}
    ]
    # The first stage is the externally-declared input lane (image/audio/video
    # or prompt data).  Every later pipeline kind/operation used to be a second
    # copy of config-authored architecture, so keep its geometry fields but
    # erase the mechanism until a source projector restores it.
    for position, step in enumerate(path.get("pipeline") or ()):
        if position == 0:
            for key in tuple(step):
                if key not in {"id", "kind", "operation", "shape"}:
                    step.pop(key, None)
            continue
        step_id = step.get("id")
        for key in tuple(step):
            if key not in {"id", "kind", "operation"}:
                step.pop(key, None)
        step["id"] = step_id
        step["kind"] = "code_defined_stage"
        step["operation"] = "unknown"


def _apply_tower(path: dict, tower: ComponentTowerMechanisms, *,
                 modality: str) -> None:
    encoder = path.setdefault("encoder", {})
    # The exact active component proves that an encoder tower exists.  Its
    # modality is presentation/address context, not a mechanism classifier.
    encoder["kind"] = ("vision_encoder" if modality in {"vision", "video"}
                       else f"{modality}_encoder")
    encoder["source_component"] = tower.component.component_key
    encoder["source_owner"] = tower.stage_symbol.qualified_name

    variants = []
    for variant_index, item in enumerate(tower.variants):
        bound = item.bound_attention
        geometry = item.attention_geometry
        storage = item.attention_storage
        ffn = item.ffn
        item_topology = (
            item.cell_topology_result.value
            if item.cell_topology_result.status == "resolved" else None)
        gates = (
            tuple(
                branch.residual_gate
                for branch in (*item_topology.mixers, *item_topology.ffns)
                if branch.residual_gate is not None)
            if item_topology is not None else ())
        gate_activations = {gate.activation for gate in gates}
        gate_sources = {gate.source for gate in gates}
        operation_value = (
            item.operations_result.value
            if item.operations_result.status in {"resolved", "incomplete"}
            else None)
        variants.append({
            "repeat": (tower.repeat_value
                       if len(tower.variants) == 1 else None),
            "block_class": item.block_symbol.qualified_name,
            "source_file": item.block_symbol.source.canonical_path,
            "attention_kind": bound.kind if bound is not None else None,
            "num_attention_heads": (
                bound.num_heads if bound is not None else None),
            "num_key_value_heads": (
                bound.num_kv_heads if bound is not None else None),
            "head_dim": geometry.head_dim if geometry is not None else None,
            "hidden_size": (
                bound.num_heads * geometry.head_dim
                if bound is not None and geometry is not None
                and bound.num_heads and geometry.head_dim else None),
            "projection_mode": (
                "split_qkv" if storage == "split" else storage),
            "ffn_gated": ffn.gated if ffn is not None else None,
            "ffn_projection_mode": (
                ffn.projection_mode if ffn is not None else None),
            "activation": ffn.activation if ffn is not None else None,
            "norm_kind": item.norm_kind,
            "norm_placement": (
                item_topology.norm_placement
                if item_topology is not None else None),
            "position_kind": tower.position.kind_for(variant_index),
            "position_application": tower.position.application_for(
                variant_index),
            "residual_gated": (
                bool(gates) if item_topology is not None else None),
            "gate_activation": (
                next(iter(gate_activations))
                if len(gate_activations) == 1 else None),
            "gate_source": (
                next(iter(gate_sources))
                if len(gate_sources) == 1 else None),
            "standard_cell": (
                item_topology is not None
                and item.ffn_census_result.status == "resolved"
                and len(item.ffn_census_result.value.candidates) == 1),
            "ops": ([operation.to_dict()
                     for operation in operation_value.operations]
                    if operation_value is not None else []),
        })
    encoder["variants"] = variants
    _retain_variant_attention_consensus(encoder, variants)
    if tower.repeat_value is not None:
        encoder["num_layers"] = tower.repeat_value
    if variants and all(item["standard_cell"] for item in variants):
        encoder["sub_model"] = tower_submodel_spec(
            encoder, variants, component=tower.component.component_key)
    else:
        encoder.pop("sub_model", None)
    if tower.final_norm_result.status == "resolved":
        encoder["final_norm_kind"] = tower.final_norm_result.value
    else:
        encoder.pop("final_norm_kind", None)

    position, application = _position_fact(tower)
    if position is None:
        encoder.pop("position_encoding", None)
        encoder.pop("input_position_kind", None)
    else:
        encoder["position_encoding"] = {
            "kind": position,
            **({"application": application} if application else {}),
        }
        encoder["input_position_kind"] = position

    boundary = tower.boundary_operations_result
    component_boundary = tower.component_boundary_operations_result
    frontend = tower.frontend_operations_result
    existing_embedding = path.get("embedding")
    embedding = (existing_embedding
                 if isinstance(existing_embedding, dict) else {})
    if component_boundary.status in {"resolved", "incomplete"}:
        front_ops = tuple(component_boundary.value.frontend)
        inner_post = (tuple(boundary.value.post)
                      if boundary.status in {"resolved", "incomplete"}
                      else ())
        post_ops = (*inner_post, *component_boundary.value.post)
    elif boundary.status in {"resolved", "incomplete"}:
        front_ops = tuple(boundary.value.frontend)
        post_ops = tuple(boundary.value.post)
    elif frontend.status in {"resolved", "incomplete"}:
        front_ops = tuple(
            operation for route in frontend.value.routes
            for operation in route.operations)
        post_ops = ()
    else:
        front_ops = post_ops = ()
    if front_ops or post_ops:
        encoder["frontend_ops"] = [item.to_dict() for item in front_ops]
        if post_ops:
            encoder["post_ops"] = [item.to_dict() for item in post_ops]
        else:
            encoder.pop("post_ops", None)
        if front_ops:
            # The repeated stage is not necessarily the class that implements
            # input embedding.  Cite the exact producer retained by the first
            # proven frontend operation (for example a nested patch embedder),
            # rather than laundering that operation through the stage owner.
            embedding["source_owner"] = front_ops[0].class_name
            embedding["ops"] = [operation.to_dict() for operation in front_ops]
            embedding["kind"] = "code_defined_embedding"
            path["embedding"] = embedding
        else:
            embedding.pop("ops", None)
        ops = (*front_ops, *post_ops)
        reduction_ops = tuple(
            operation for operation in ops
            if operation.kind in {"pooling", "pixel_shuffle", "pixel_unshuffle"})
        if reduction_ops:
            path["reduction"] = {
                "kind": reduction_ops[-1].kind,
                "operation": reduction_ops[-1].kind,
                "source_evidence": [item.to_dict() for item in reduction_ops],
            }
            _install_reduction_step(path, reduction_ops[-1].kind)
        else:
            path.pop("reduction", None)
    else:
        encoder.pop("frontend_ops", None)
        encoder.pop("post_ops", None)
        embedding.pop("ops", None)
        if existing_embedding is None and not embedding:
            path.pop("embedding", None)
        path.pop("reduction", None)
    # A checkpoint declaration may describe a processor policy, but the model
    # bundle did not prove a tiling operation.  Keep the declaration out of the
    # model architecture until external preprocessing source has its own typed
    # evidence boundary.
    path.pop("tiling", None)
    for step in path.get("pipeline") or ():
        if step.get("id") in {"patch_embedding", "video_patch_embedding"}:
            if embedding.get("ops"):
                step["ops"] = embedding["ops"]
            else:
                step.pop("ops", None)
        elif step.get("id") in {
                "vision_encoder", "video_encoder", "audio_encoder",
                "conditioning_encoder"}:
            step["kind"] = encoder["kind"]
            step["operation"] = "encode"
            step["source_component"] = tower.component.component_key
            step["source_owner"] = tower.stage_symbol.qualified_name
            if tower.repeat_value is not None:
                step["num_layers"] = tower.repeat_value
            if encoder.get("hidden_size") is not None:
                step["hidden_size"] = encoder["hidden_size"]


def _apply_towers(path: dict, towers: tuple[ComponentTowerMechanisms, ...], *,
                  modality: str) -> None:
    """Project every exact repeated stage of one component without picking one."""
    if not towers:
        return
    accumulated = []
    attention_snapshots = []
    stage_snapshots = []
    owners = []
    for tower in towers:
        _apply_tower(path, tower, modality=modality)
        encoder = path.setdefault("encoder", {})
        accumulated.extend(encoder.get("variants") or ())
        owners.append(tower.stage_symbol.qualified_name)
        attention_snapshots.append(tuple(
            encoder.get(key) for key in (
                "attention_kind", "num_attention_heads",
                "num_key_value_heads", "projection_mode", "head_dim")))
        # A component can contain several exact repeated stages.  Keep a
        # stage's component-level summary only when every stage independently
        # projects the same value.  Otherwise the last stage would silently
        # become the whole component (the same bug that once reported Mllama's
        # 8 global blocks as its complete 40-block vision tower).
        stage_snapshots.append({
            "final_norm_kind": deepcopy(encoder.get("final_norm_kind")),
            "position_encoding": deepcopy(encoder.get("position_encoding")),
            "input_position_kind": deepcopy(encoder.get("input_position_kind")),
            "frontend_ops": deepcopy(encoder.get("frontend_ops")),
            "post_ops": deepcopy(encoder.get("post_ops")),
            "embedding": deepcopy(path.get("embedding")),
            "reduction": deepcopy(path.get("reduction")),
        })
    encoder = path.setdefault("encoder", {})
    encoder["variants"] = accumulated
    if len(towers) == 1:
        encoder["source_owner"] = owners[0]
        encoder.pop("source_owners", None)
    else:
        encoder.pop("source_owner", None)
        encoder["source_owners"] = list(dict.fromkeys(owners))
    # A component-level attention summary is legal only when every exact stage
    # independently projects the same values.  Stage variants retain their own
    # facts either way.
    if len(set(attention_snapshots)) != 1:
        for key in (
                "attention_kind", "num_attention_heads",
                "num_key_value_heads", "projection_mode", "head_dim"):
            encoder.pop(key, None)
    for key in (
            "final_norm_kind", "position_encoding", "input_position_kind",
            "frontend_ops", "post_ops"):
        _retain_stage_consensus(encoder, key, stage_snapshots)
    _retain_stage_consensus(path, "embedding", stage_snapshots)
    _retain_stage_consensus(path, "reduction", stage_snapshots)
    if path.get("reduction"):
        _install_reduction_step(path, path["reduction"]["kind"])
    else:
        path["pipeline"] = [
            item for item in (path.get("pipeline") or ())
            if item.get("id") != "vision_token_reduce"
        ]
    # ``_apply_tower`` projects one exact repeated stage.  A component may
    # execute several such stages (Mllama has a 32-block local encoder followed
    # by an 8-block gated global encoder).  Letting the last stage overwrite
    # ``num_layers`` falsely reports eight layers for the whole component.
    # Every count below is already bound to the exact stage constructor; only
    # when all stages carry that proof may the component expose their total.
    repeat_values = tuple(item.repeat_value for item in towers)
    if repeat_values and all(value is not None for value in repeat_values):
        encoder["num_layers"] = sum(repeat_values)
    else:
        encoder.pop("num_layers", None)
    for step in path.get("pipeline") or ():
        if step.get("id") in {"patch_embedding", "video_patch_embedding"}:
            embedding = path.get("embedding") or {}
            if embedding.get("ops"):
                step["ops"] = embedding["ops"]
            else:
                step.pop("ops", None)
        if step.get("id") in {
                "vision_encoder", "video_encoder", "audio_encoder",
                "conditioning_encoder"}:
            if encoder.get("num_layers") is not None:
                step["num_layers"] = encoder["num_layers"]
            else:
                step.pop("num_layers", None)
    if accumulated and all(item.get("standard_cell") for item in accumulated):
        encoder["sub_model"] = tower_submodel_spec(
            encoder, accumulated,
            component=towers[0].component.component_key)
    else:
        encoder.pop("sub_model", None)


def _retain_variant_attention_consensus(encoder: dict,
                                        variants: list[dict]) -> None:
    """Project a stage summary only when all exact block variants agree."""
    keys = (
        "attention_kind", "num_attention_heads", "num_key_value_heads",
        "projection_mode", "head_dim", "hidden_size",
    )
    snapshots = [tuple(item.get(key) for key in keys) for item in variants]
    if snapshots and len(set(snapshots)) == 1:
        for key, value in zip(keys, snapshots[0]):
            if value is None:
                encoder.pop(key, None)
            else:
                encoder[key] = value
        return
    for key in keys:
        encoder.pop(key, None)


_NO_CONSENSUS = object()


def _retain_stage_consensus(target: dict, key: str, snapshots: list[dict]) -> None:
    """Retain a component summary only when every exact stage agrees."""
    values = [snapshot.get(key, _NO_CONSENSUS) for snapshot in snapshots]
    first = values[0] if values else _NO_CONSENSUS
    if first is not _NO_CONSENSUS and all(value == first for value in values[1:]):
        target[key] = first
    else:
        target.pop(key, None)


def _install_reduction_step(path: dict, kind: str) -> None:
    """Insert the exact reduction between encoder and connector once."""
    pipeline = [
        item for item in (path.get("pipeline") or ())
        if item.get("id") != "vision_token_reduce"
    ]
    step = {
        "id": "vision_token_reduce",
        "operation": kind,
        "kind": kind,
    }
    connector_ids = {
        "projector", "video_projector", "audio_projector",
        "conditioning_projector",
    }
    position = next((number for number, item in enumerate(pipeline)
                     if item.get("id") in connector_ids), len(pipeline))
    pipeline.insert(position, step)
    path["pipeline"] = pipeline


def apply_wrapper_feature_evidence(payload: dict | None, result) -> dict | None:
    """Replace config-authored feature claims with exact wrapper operations."""
    if not payload:
        return payload
    inputs = (payload.get("modalities") or {}).get("inputs") or {}
    routes = result.value if getattr(result, "status", "") == "resolved" else ()
    operations = tuple(
        operation for route in routes
        if isinstance(route, WrapperFeatureRoute)
        for operation in route.operations)
    for name in ("vision", "video"):
        path = inputs.get(name)
        if not isinstance(path, dict):
            continue
        encoder = path.setdefault("encoder", {})
        if operations:
            kinds = {item.kind for item in operations}
            if "single_layer_select" not in kinds:
                encoder.pop("feature_layer", None)
            if "multi_layer_select" not in kinds:
                encoder.pop("intermediate_layers_indices", None)
                encoder.pop("output_dim", None)
            if "drop_first_token" not in kinds:
                encoder.pop("feature_select_strategy", None)
            encoder["feature_operations"] = [
                {"kind": item.kind, "conditional": bool(item.guard)}
                for item in operations]
        else:
            for key in (
                    "feature_layer", "intermediate_layers_indices",
                    "feature_select_strategy", "output_dim"):
                encoder.pop(key, None)
            encoder.pop("feature_operations", None)
    return payload


def _position_fact(tower):
    return tower.position.kind, tower.position.application


__all__ = [
    "apply_recursive_component_evidence", "apply_wrapper_feature_evidence",
]
