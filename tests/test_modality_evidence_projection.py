"""U9-F source-authority cutover controls for modality projections."""
from __future__ import annotations

from copy import deepcopy

from model_unfolder.adapters.transformer.special_parts.modalities.evidence_projection import (
    apply_recursive_component_evidence,
)
from model_unfolder.adapters.transformer.special_parts.modalities.fusion import (
    apply_fusion_evidence,
)
from model_unfolder.adapters.transformer.special_parts.modalities.vision import (
    apply_projector_evidence,
)
from model_unfolder.adapters.transformer.special_parts.modalities.schema import (
    tower_submodel_spec,
)
from model_unfolder.evidence.models import FusionEvidence, FusionRouteEvidence
from model_unfolder.renderers.html.block_views.modality_views.audio import (
    encoder_tower_spec,
)
from model_unfolder.renderers.html.metadata_modalities import (
    _multimodal_block_lookup,
)
from model_unfolder.renderers.html.block_views.modality_views.vision import (
    _connector_label,
)


_LEGACY = {
    "modalities": {
        "inputs": {
            "vision": {
                "kind": "image_to_soft_visual_tokens",
                "input": {
                    "kind": "image_pixels",
                    "shape": ["batch", "channels", "height", "width"],
                    "image_size": 224,
                    "patch_size": 14,
                },
                "embedding": {
                    "kind": "patch_embedding", "patch_size": 14,
                    "ops": [{"kind": "conv2d"}],
                    "source_owner": "ConfigGuessedTower",
                },
                "encoder": {
                    "kind": "vision_encoder", "hidden_size": 1280,
                    "num_layers": 24, "attention_kind": "mha",
                    "ffn_gated": False, "norm_kind": "LayerNorm",
                    "activation": "gelu", "variants": [{"kind": "dense"}],
                    "sub_model": {"groups": []},
                    "position_encoding": {"kind": "learned_absolute"},
                },
                "tiling": {"kind": "image_tiling", "max_tiles": 4},
                "reduction": {"kind": "pixel_shuffle"},
                "projector": {
                    "kind": "linear_projector", "in_features": 1280,
                    "out_features": 4096, "activation": "gelu",
                    "ops": [{"kind": "linear"}],
                    "source_class": "ConfigGuessedProjector",
                },
                "tokens": {
                    "kind": "soft_visual_tokens", "count": 256,
                    "width": 4096,
                },
                "pipeline": [
                    {"id": "image_pixels", "kind": "image_pixels"},
                    {"id": "patch_embedding", "kind": "patch_embedding",
                     "operation": "patch_embedding"},
                    {"id": "vision_tiles", "kind": "image_tiling"},
                    {"id": "vision_encoder", "kind": "vision_encoder",
                     "operation": "encode"},
                    {"id": "vision_token_reduce", "kind": "pixel_shuffle"},
                    {"id": "projector", "kind": "linear_projector",
                     "out_features": 4096, "ops": [{"kind": "linear"}]},
                    {"id": "soft_visual_tokens", "kind": "soft_visual_tokens",
                     "operation": "emit_soft_token_stream"},
                ],
            },
        },
        "fusion": {
            "kind": "cross_attention", "operation": "cross_attention_states",
            "target": "decoder.cross_attention_layers",
            "mechanism": {"kind": "cross_attention"},
            "source_owner": "ConfigGuessedWrapper",
        },
    },
}


def test_missing_source_evidence_cannot_leave_config_mechanisms_behind():
    payload = deepcopy(_LEGACY)
    apply_recursive_component_evidence(payload, None)
    apply_projector_evidence(payload, None)
    apply_fusion_evidence(payload, None)

    vision = payload["modalities"]["inputs"]["vision"]
    # A declaration can keep the lane visible, but config alone cannot author
    # its layer/head/width/token geometry or mechanism.
    assert vision["kind"] == "code_defined_modality_path"
    assert vision["embedding"]["kind"] == "code_defined_embedding"
    assert vision["encoder"]["kind"] == "code_defined_encoder"
    assert vision["tokens"]["kind"] == "code_defined_tokens"
    assert vision["embedding"] == {"kind": "code_defined_embedding"}
    assert vision["tokens"] == {"kind": "code_defined_tokens"}
    assert vision["input"] == {
        "kind": "image_pixels",
        "shape": ["batch", "channels", "height", "width"],
    }
    assert "ops" not in vision["embedding"]
    for key in (
            "attention_kind", "hidden_size", "num_layers", "ffn_gated",
            "norm_kind", "activation",
            "variants", "sub_model", "position_encoding"):
        assert key not in vision["encoder"]
    assert "tiling" not in vision and "reduction" not in vision
    assert {step["id"] for step in vision["pipeline"]} == {
        "image_pixels", "patch_embedding", "vision_encoder", "projector",
        "soft_visual_tokens"}
    for step in vision["pipeline"][1:]:
        assert step["operation"] == "unknown"

    projector = vision["projector"]
    assert projector == {"kind": "code_defined_projector"}
    projector_step = next(
        step for step in vision["pipeline"] if step["id"] == "projector")
    assert projector_step == {
        "id": "projector", "kind": "code_defined_projector",
        "operation": "unknown"}

    fusion = payload["modalities"]["fusion"]
    assert fusion["kind"] == "code_defined_fusion"
    assert fusion["operation"] == fusion["target"] == "unknown"
    assert fusion["mechanism"] is None
    assert fusion["sources"] == []
    assert fusion["output"] == {"kind": "code_defined_fusion_output"}
    assert "source_owner" not in fusion

    blocks = _multimodal_block_lookup({"extras": payload})
    vision_block = blocks["vision_path"]
    assert vision_block["kind"] == "code_defined_modality_path"
    assert vision_block["title"] == "Code-defined vision pathway"
    assert "soft visual tokens" not in str(vision_block).lower()


def test_audio_geometry_cannot_fabricate_a_transformer_cell():
    tower = encoder_tower_spec({
        "hidden_size": 1024,
        "num_layers": 12,
        "num_attention_heads": 8,
    }, prefix="audio")
    assert tower["cell"] == [{
        "id": "audio_opaque", "kind": "opaque",
        "label": "Code-defined audio cell", "static": True,
        "resolved": False,
    }]


def test_tower_groups_carry_per_occurrence_geometry_without_renderer_math():
    common = {
        "repeat": 1, "block_class": "Block", "source_file": "model.py",
        "attention_kind": "gqa", "projection_mode": "split_qkv",
        "ffn_gated": False, "ffn_projection_mode": "dense",
        "intermediate_size": 4096,
        "activation": "gelu", "norm_kind": "layernorm",
        "norm_placement": "pre", "residual_gated": False,
        "position_kind": "rope", "position_application": "qk_rotation",
        "standard_cell": True,
    }
    first = {**common, "num_attention_heads": 8, "num_key_value_heads": 2,
             "head_dim": 64, "hidden_size": 512}
    # Hidden/head counts could arithmetically imply 64, but the exact head-dim
    # fact is deliberately absent for this second occurrence.
    second = {**common, "num_attention_heads": 16, "num_key_value_heads": 4,
              "head_dim": None, "hidden_size": 1024}

    groups = tower_submodel_spec(
        {"num_layers": 2}, [first, second], component="vision_config")["groups"]

    assert groups[0]["attention"]["head_dim"] == 64
    assert groups[0]["attention"]["num_kv_heads"] == 2
    assert groups[1]["attention"]["head_dim"] is None
    assert groups[1]["attention"]["num_heads"] == 16
    assert groups[1]["attention"]["num_kv_heads"] == 4
    assert all(group["ffn"]["intermediate_size"] is None for group in groups)


def test_exact_fusion_route_is_the_only_authority_that_restores_token_kind():
    payload = deepcopy(_LEGACY)
    apply_recursive_component_evidence(payload, None)
    evidence = FusionEvidence(
        "proven", owner_class="Root", source_file="model.py", line=9,
        kind="placeholder_replace",
        operation="scatter_soft_tokens_into_placeholder_slots",
        routes=(FusionRouteEvidence(
            "vision", "masked_scatter", "model.py", 9),),
    )
    apply_fusion_evidence(payload, evidence)
    vision = payload["modalities"]["inputs"]["vision"]
    assert vision["kind"] == "image_to_soft_visual_tokens"
    assert vision["tokens"]["kind"] == "soft_visual_tokens"
    assert vision["pipeline"][-1]["kind"] == "soft_visual_tokens"
    assert vision["pipeline"][-1]["operation"] == "emit_soft_token_stream"


def test_multiaxis_cannot_create_grid_tokens_without_a_proven_fusion_route():
    class _Resolved:
        status = "resolved"
        value = (object(),)

    payload = deepcopy(_LEGACY)
    apply_recursive_component_evidence(payload, None)
    apply_fusion_evidence(payload, None, multiaxis_result=_Resolved())
    vision = payload["modalities"]["inputs"]["vision"]
    assert vision["kind"] == "code_defined_modality_path"
    assert vision["tokens"]["kind"] == "code_defined_tokens"


def test_grid_route_cannot_relabel_an_unknown_projector_as_a_merger():
    # Token routing and connector implementation are independent facts.
    assert _connector_label({"kind": "code_defined_projector"}) == "Projector"
