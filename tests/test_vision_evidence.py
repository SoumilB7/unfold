"""U9 visual controls through recursive component and fusion authorities."""
from __future__ import annotations

from copy import deepcopy

from model_unfolder import unfold
from model_unfolder.evidence.conformance import check_fact_conformance
from test_support import MLLAMA_VISION_TINY_CONFIG, QWEN2VL_STYLE


def test_qwen2vl_visual_lane_joins_exact_tower_fusion_and_projector_facts():
    diagram = unfold(QWEN2VL_STYLE)
    ir = diagram.to_ir()
    vision = ir["extras"]["modalities"]["inputs"]["vision"]
    assert vision["kind"] == "image_to_grid_tokens"
    assert vision["tokens"]["kind"] == "grid_visual_tokens"
    assert vision["embedding"]["kind"] == "code_defined_embedding"
    assert vision["embedding"]["ops"]
    assert vision["encoder"]["kind"] == "vision_encoder"
    assert vision["encoder"]["source_owner"] == "Qwen2VisionTransformerPretrainedModel"
    assert vision["encoder"]["variants"][0]["ffn_gated"] is False
    assert not [problem for problem in check_fact_conformance(QWEN2VL_STYLE, ir)
                if problem.kind in {"wrong_vision_fact", "wrong_projector_fact",
                                    "wrong_fusion_fact"}]


def test_mllama_keeps_two_exact_vision_stages_and_cross_attention_route():
    ir = unfold(MLLAMA_VISION_TINY_CONFIG).to_ir()
    vision = ir["extras"]["modalities"]["inputs"]["vision"]
    assert vision["kind"] == "image_to_cross_attention_states"
    assert vision["tokens"]["kind"] == "vision_cross_attention_states"
    variants = vision["encoder"]["variants"]
    assert [item["repeat"] for item in variants] == [32, 8]
    assert [item["residual_gated"] for item in variants] == [False, True]
    assert not [
        problem for problem in check_fact_conformance(
            MLLAMA_VISION_TINY_CONFIG, ir)
        if problem.kind in {
            "wrong_vision_fact", "wrong_projector_fact", "wrong_fusion_fact"}
    ]


def test_source_missing_visual_shell_keeps_address_but_no_architecture():
    from model_unfolder.adapters.transformer.special_parts.modalities.evidence_projection import (
        apply_recursive_component_evidence,
    )
    from model_unfolder.adapters.transformer.special_parts.modalities.vision import vision_path

    cfg = deepcopy(QWEN2VL_STYLE)
    path = vision_path(cfg, cfg["text_config"], cfg["vision_config"], 64)
    payload = {"modalities": {"inputs": {"vision": path}}}
    apply_recursive_component_evidence(payload, None)
    vision = payload["modalities"]["inputs"]["vision"]
    assert vision["kind"] == "code_defined_modality_path"
    assert vision["embedding"]["kind"] == "code_defined_embedding"
    assert vision["encoder"]["kind"] == "code_defined_encoder"
    assert vision["tokens"]["kind"] == "code_defined_tokens"
    assert "num_layers" not in vision["encoder"]
    assert "hidden_size" not in vision["encoder"]
    assert "num_attention_heads" not in vision["encoder"]
    assert "sub_model" not in vision["encoder"]


def test_retired_vision_reader_is_not_a_second_source_authority():
    import model_unfolder.evidence.vision as vision

    assert not hasattr(vision, "vision_tower_evidence")
