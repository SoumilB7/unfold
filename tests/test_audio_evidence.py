"""U9 audio controls through the shared recursive component authority."""
from __future__ import annotations

from copy import deepcopy

from model_unfolder import unfold
from model_unfolder.evidence.conformance import check_fact_conformance
from test_support import GEMMA4_VISION_TINY_CONFIG


def _gemma_audio_config():
    cfg = deepcopy(GEMMA4_VISION_TINY_CONFIG)
    cfg.update({
        "audio_token_id": 258881,
        "boa_token_id": 256000,
        "eoa_token_id": 258883,
        "audio_seq_length": 750,
        "audio_ms_per_token": 40,
        "audio_config": {
            "architectures": ["Gemma4AudioModel"],
            "model_type": "gemma4_audio",
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "output_proj_dims": 64,
            "feature_size": 128,
        },
    })
    return cfg


def test_audio_ir_uses_recursive_component_projector_and_fusion_results():
    cfg = _gemma_audio_config()
    diagram = unfold(cfg)
    ir = diagram.to_ir()
    audio = ir["extras"]["modalities"]["inputs"]["audio"]
    assert audio["encoder"]["source_owner"] == "Gemma4AudioModel"
    assert audio["encoder"]["kind"] == "audio_encoder"
    assert audio["projector"]["source_class"] == "Gemma4MultimodalEmbedder"
    assert [item["kind"] for item in audio["projector"]["ops"]] == [
        "norm", "linear"]
    assert audio["kind"] == "audio_to_soft_tokens"
    assert audio["tokens"]["kind"] == "soft_audio_tokens"
    assert not [problem for problem in check_fact_conformance(cfg, ir)
                if problem.kind in {"wrong_audio_fact", "wrong_projector_fact",
                                    "wrong_fusion_fact"}]
    assert diagram.wiring_problems() == []


def test_audio_geometry_without_source_stays_an_opaque_cell():
    from model_unfolder.adapters.transformer.special_parts.modalities.audio import audio_path
    from model_unfolder.adapters.transformer.special_parts.modalities.evidence_projection import (
        apply_recursive_component_evidence,
    )
    from model_unfolder.renderers.html.block_views.modality_views.audio import (
        encoder_tower_spec,
    )

    payload = {"modalities": {"inputs": {
        "audio": audio_path(
            {"audio_token_id": 1},
            {"num_hidden_layers": 2, "num_attention_heads": 4}, 64),
    }}}
    apply_recursive_component_evidence(payload, None)
    encoder = payload["modalities"]["inputs"]["audio"]["encoder"]
    assert encoder["kind"] == "code_defined_encoder"
    tower = encoder_tower_spec(encoder, prefix="audio")
    assert [item["kind"] for item in tower["cell"]] == ["opaque"]


def test_retired_audio_reader_is_not_a_second_source_authority():
    import model_unfolder.evidence as evidence

    assert not hasattr(evidence, "audio_tower_evidence")
