from __future__ import annotations

from dataclasses import replace

from model_unfolder import unfold
from model_unfolder.adapters.transformer.special_parts.modalities.audio import (
    apply_audio_evidence,
    audio_path,
)
from model_unfolder.evidence.audio import audio_tower_evidence
from model_unfolder.evidence.conformance import check_fact_conformance
from model_unfolder.evidence.models import AudioTowerEvidence, SourceBundle


GEMMA_AUDIO = {
    "model_type": "gemma4",
    "architectures": ["Gemma4ForConditionalGeneration"],
    "audio_token_id": 1,
    "audio_config": {
        "model_type": "gemma4_audio", "num_hidden_layers": 3,
        "hidden_size": 64, "num_attention_heads": 4,
        "output_proj_dims": 128,
    },
    "text_config": {
        "model_type": "gemma4_text", "num_hidden_layers": 2,
        "hidden_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "intermediate_size": 256,
        "vocab_size": 100,
    },
}

QWEN_AUDIO = {
    "model_type": "qwen2_audio",
    "architectures": ["Qwen2AudioForConditionalGeneration"],
    "audio_token_index": 1,
    "audio_config": {
        "model_type": "qwen2_audio_encoder", "encoder_layers": 3,
        "d_model": 64, "encoder_attention_heads": 4,
        "encoder_ffn_dim": 128, "num_mel_bins": 80,
    },
    "text_config": {
        "model_type": "qwen2", "num_hidden_layers": 2,
        "hidden_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "intermediate_size": 256,
        "vocab_size": 100,
    },
}


def test_real_audio_towers_resolve_distinct_source_graphs():
    gemma = audio_tower_evidence(GEMMA_AUDIO)
    qwen = audio_tower_evidence(QWEN_AUDIO)
    assert (gemma.status, gemma.owner_class) == ("proven", "Gemma4AudioModel")
    assert (qwen.status, qwen.owner_class) == ("proven", "Qwen2AudioEncoder")
    assert (gemma.position_kind, gemma.position_application) == (
        "relative", "attention_side_input",
    )
    assert (qwen.position_kind, qwen.position_application) == (
        "fixed_absolute", "embedding_add",
    )
    assert gemma.variants[0].repeat_field == "num_hidden_layers"
    assert qwen.variants[0].repeat_field == "encoder_layers"

    gemma_labels = [op.label for op in gemma.variants[0].ops]
    assert gemma_labels == [
        "Feed-forward 1", "Clamp", "RMSNorm", "Self-attention", "Clamp",
        "RMSNorm", "Residual add", "LightConv1d", "Feed-forward 2",
        "Clamp", "RMSNorm",
    ]
    qwen_kinds = [op.kind for op in qwen.variants[0].ops]
    assert qwen_kinds.count("elementwise") == 2
    assert [op.label for op in qwen.frontend_ops][:4] == [
        "Conv1d", "GELU", "Conv1d", "GELU",
    ]
    assert [op.label for op in gemma.frontend_ops].count("Conv2d") == 2
    assert gemma.frontend_ops[0].label == "Add channel axis"
    assert gemma.frontend_ops[-2].label == "Flatten subsampled features"
    light_conv = next(item for item in gemma.variants[0].callables
                      if item.class_name == "Gemma4AudioLightConv1d")
    assert any(op.kind == "conv" and "Depthwise" in op.label for op in light_conv.ops)
    assert [op.label for op in qwen.post_ops] == [
        "Reorder tensor axes", "Temporal average pool",
        "Reorder tensor axes", "LayerNorm",
    ]


def test_audio_ir_and_fact_net_consume_the_same_record():
    diagram = unfold(QWEN_AUDIO)
    ir = diagram.to_ir()
    audio = ir["extras"]["modalities"]["inputs"]["audio"]
    assert audio["encoder"]["source_owner"] == "Qwen2AudioEncoder"
    assert audio["projector"]["source_class"] == "Qwen2AudioMultiModalProjector"
    assert not [problem for problem in check_fact_conformance(QWEN_AUDIO, ir)
                if problem.kind == "wrong_audio_fact"]

    audio["encoder"]["position_encoding"]["kind"] = "relative"
    problems = check_fact_conformance(QWEN_AUDIO, ir)
    assert any(problem.kind == "wrong_audio_fact" and "position.kind" in problem.op
               for problem in problems)


def test_audio_rendering_is_source_shaped_and_wiring_clean():
    for cfg, required in (
        (GEMMA_AUDIO, ("Feed-forward 1", "LightConv1d", "Relative positions", "GLU")),
        (QWEN_AUDIO, ("Conv1d", "Fixed positions", "Temporal average pool")),
    ):
        diagram = unfold(cfg)
        html = diagram.to_html(standalone=True)
        assert all(label in html for label in required)
        assert "gemma4 audio" not in html.lower()
        assert diagram.wiring_problems() == []


def test_unresolved_audio_is_opaque_not_a_conventional_attention_ffn_cell():
    base = {"modalities": {"inputs": {
        "audio": audio_path({"audio_token_id": 1}, {"num_hidden_layers": 2}, 64),
    }}}
    evidence = AudioTowerEvidence(
        "ambiguous", component="audio_config", owner_class="CustomAudio",
        reason="no exact repeated audio block resolved",
    )
    payload = apply_audio_evidence(base, evidence)
    encoder = payload["modalities"]["inputs"]["audio"]["encoder"]
    assert encoder["evidence_status"] == "ambiguous"
    assert encoder["variants"] == []
    assert payload["modalities"]["inputs"]["audio"]["projector"]["kind"] == \
        "code_defined_projector"


def test_audio_extractor_does_not_guess_from_model_identity(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text("class Wrapper:\n    def forward(self, audio):\n        return audio\n")
    bundle = SourceBundle(
        source="path", files=(str(source),), architecture="Wrapper",
        component_files={"root": (str(source),)},
        component_architectures={"root": "Wrapper"},
    )
    first = audio_tower_evidence({"model_type": "gemma4", "audio_config": {}}, bundle=bundle)
    second = audio_tower_evidence({"model_type": "qwen2_audio", "audio_config": {}}, bundle=bundle)
    assert replace(first, reason="") == replace(second, reason="")
    assert first.status == "ambiguous"
