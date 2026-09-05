"""Expanded architecture JSON tests."""
from copy import deepcopy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_unfolder import config_to_ir, unfold


LLAMA_TINY_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "_name_or_path": "meta-llama/Meta-Llama-3-8B",
    "vocab_size": 32000,
    "hidden_size": 64,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}

# Shared vision fixtures live in the importable test_support package (§16.1).
from test_support import GEMMA4_VISION_TINY_CONFIG, MLLAMA_VISION_TINY_CONFIG


QWEN2_AUDIO_SPARSE_CONFIG = {
    "architectures": ["Qwen2AudioForConditionalGeneration"],
    "model_type": "qwen2_audio",
    "_name_or_path": "Qwen/Qwen2-Audio-7B",
    "audio_token_index": 151646,
    "vocab_size": 156032,
    "audio_config": {
        "model_type": "qwen2_audio_encoder",
        "num_mel_bins": 128,
        "encoder_layers": 32,
        "encoder_attention_heads": 20,
        "encoder_ffn_dim": 5120,
        "d_model": 1280,
    },
    "text_config": {
        "intermediate_size": 11008,
        "max_position_embeddings": 8192,
        "model_type": "qwen2",
        "rope_theta": 10000,
        "rms_norm_eps": 1e-5,
        "sliding_window": 32768,
        "vocab_size": 156032,
    },
}


QWEN2_VL_TINY_CONFIG = {
    "architectures": ["Qwen2VLForConditionalGeneration"],
    "model_type": "qwen2_vl",
    "_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "text_config": {
        "architectures": ["Qwen2VLForCausalLM"],
        "model_type": "qwen2_vl_text",
        "vocab_size": 152064,
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32768,
        "hidden_act": "silu",
        "rope_scaling": {"type": "mrope"},
    },
    "vision_config": {
        "architectures": ["Qwen2VisionTransformerPretrainedModel"],
        # Match the actual nested config class.  Using the wrapper's
        # ``qwen2_vl`` discriminator here makes source resolution select the
        # wrong config boundary and is not a valid Qwen2-VL counterexample.
        "model_type": "qwen2_vl_vision",
        "embed_dim": 32,
        "hidden_size": 64,
        "depth": 3,
        "num_heads": 4,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
    },
}


def test_expanded_json_is_structural_not_renderer_copy():
    data = unfold(LLAMA_TINY_CONFIG, return_json=True)
    encoded = json.dumps(data)

    assert data["schema_version"] == "3.2"
    assert data["format"] == "model_unfolder.expanded"
    assert data["model"] == {
        "name": "Meta-Llama-3-8B",
        "architecture": "LlamaForCausalLM",
    }
    assert data["dimensions"]["hidden_size"] == 64
    assert data["stack"]["num_layers"] == 2

    assert "summary" not in data
    assert "features" not in data
    assert "description" not in encoded
    assert "label" not in encoded
    assert "title" not in encoded


def test_expanded_json_has_traceable_attention_and_ffn_graphs():
    data = unfold(LLAMA_TINY_CONFIG, return_json=True)
    group = data["layer_groups"][0]

    assert group["layers"]["ranges"] == [{"start": 0, "end": 1, "step": 1, "count": 2}]
    assert group["attention"]["kind"] == "gqa"
    assert group["attention"]["heads"] == {
        "query": 4,
        "key_value": 2,
        "kv_groups": 2,
        "head_dim": 16,
        "query_width": 64,
        "key_value_width": 32,
        "residual_width": 64,
    }
    assert group["attention"]["projections"]["key"]["out_features"] == 32
    assert group["attention"]["cache"] == {
        "enabled": True,
        "kind": "kv",
        "stores": ["key", "value"],
        "kv_heads": 2,
        "head_dim": 16,
    }
    assert group["attention"]["trace"]["ir_path"] == "layers[0].attention"

    attention_nodes = {node["id"]: node for node in group["attention"]["operation_graph"]["nodes"]}
    assert attention_nodes["scores"]["operation"] == "scaled_dot_product"
    assert attention_nodes["scores"]["formula"] == "QK^T/sqrt(dim)"
    assert attention_nodes["k_proj"]["parameters"]["weight_shape"] == [32, 64]

    ffn_nodes = {node["id"]: node for node in group["ffn"]["operation_graph"]["nodes"]}
    assert group["ffn"]["kind"] == "dense"
    assert ffn_nodes["gate_proj"]["parameters"]["out_features"] == 256
    assert ffn_nodes["multiply"]["operation"] == "elementwise_multiply"


def test_expanded_json_carries_structured_code_evidence(tmp_path):
    path = tmp_path / "modeling_fake.py"
    path.write_text(
        """
class FakeAttention:
    def __init__(self, config):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
        self.num_key_value_groups = 2

class FakeMLP:
    def __init__(self, config):
        self.gate_proj = Linear()
        self.up_proj = Linear()
        self.down_proj = Linear()
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ir = config_to_ir(LLAMA_TINY_CONFIG, inspect_code=True, code_source=str(tmp_path))
    data = unfold(LLAMA_TINY_CONFIG, inspect_code=True, code_source=str(tmp_path), return_json=True)

    assert ir.extras["code_evidence"]["provenance"]["source"] == "path"
    evidence = data["code_evidence"]
    assert evidence["schema_version"] == "1.0"
    assert evidence["provenance"]["source"] == "path"
    assert "grouped_kv_attention" in evidence["detections"]["attention"]
    assert evidence["detections"]["attention"]["grouped_kv_attention"]["locations"][0]["class"] == "FakeAttention"
    # A global semantic detection can belong to a sibling owner. Expanded
    # JSON keeps the exact IR path but cites no finding until a fact-level,
    # owner-qualified receipt exists.
    assert data["layer_groups"][0]["attention"]["trace"]["code_finding_ids"] == []


def test_expanded_json_carries_structured_multimodal_inputs():
    data = unfold(GEMMA4_VISION_TINY_CONFIG, return_json=True)
    encoded = json.dumps(data["modalities"])

    vision = data["modalities"]["inputs"]["vision"]
    assert vision["kind"] == "image_to_soft_visual_tokens"
    assert vision["input"] == {
        "kind": "image_pixels",
        "shape": ["batch", "images", "channels", "height", "width"],
    }
    embedding = vision["embedding"]
    assert embedding["kind"] == "code_defined_embedding"
    # The exact active embedding route is not closed by U9 for this source
    # version.  The former elementwise+linear profile came from the config/
    # family shell; keep the stage navigable but mechanism-opaque.
    assert set(embedding) == {"kind"}
    assert vision["encoder"]["kind"] == "vision_encoder"
    assert vision["encoder"]["num_layers"] == 3
    assert "hidden_size" not in vision["encoder"]
    # Position encoding is derived structurally (learned table + 2D RoPE), no family hint.
    assert vision["encoder"]["position_encoding"] == {
        "kind": "learned_absolute_plus_rope",
        "application": "embedding_add_and_qk_rotation",
    }
    # A checkpoint pool-size declaration cannot create a pooling operation.
    # This source version proves no such post-stage call, so no reduction is
    # rendered.
    assert "reduction" not in vision
    # COR-4/COR-5 (§9/§10): the embedder projects through a raw Parameter
    # einsum — no construction-site Linear proves the output width, so the
    # v1 binder refuses (out_width_source="unavailable") and the card
    # carries NO out_features rather than the old text-width fabrication.
    # The decoder-side token width below stays 64 (interface truth).
    assert "out_features" not in vision["projector"]
    assert vision["projector"]["source_evidence"]["out_width_source"] == "unavailable"
    assert vision["tokens"] == {"kind": "soft_visual_tokens"}
    assert [step["operation"] for step in vision["pipeline"]] == [
        "input",
        "unknown",
        "encode",
        "unknown",
        "emit_soft_token_stream",
    ]
    assert vision["pipeline"][-1] == {
        "id": "vision_tokens",
        "operation": "emit_soft_token_stream",
        "kind": "soft_visual_tokens",
    }

    fusion = data["modalities"]["fusion"]
    assert fusion["kind"] == "placeholder_replace"
    assert fusion["operation"] == "scatter_soft_tokens_into_placeholder_slots"
    assert fusion["placeholder"] == {"kind": "image_placeholder", "token_id": 262144}
    assert fusion["mechanism"] == {
        "kind": "scatter",
        "source": "modalities.inputs.vision.tokens",
        "into": "io.token_embedding",
        "operation": "masked_scatter",
        "at": {"kind": "image_placeholder", "token_id": 262144},
    }
    assert fusion["target"] == "stack.input_embeddings"
    assert data["io"]["stack_input"] == {
        "kind": "mixed_embeddings",
        "width": 64,
        "source": "modalities.fusion",
        "trace": {"ir_path": "extras.modalities.fusion"},
    }

    assert "description" not in encoded
    assert "label" not in encoded
    assert "title" not in encoded


def test_expanded_json_carries_structured_audio_inputs():
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

    data = unfold(cfg, return_json=True)
    encoded = json.dumps(data["modalities"])

    audio = data["modalities"]["inputs"]["audio"]
    assert audio["kind"] == "audio_to_soft_tokens"
    assert audio["input"] == {
        "kind": "audio_features",
        "shape": ["batch", "segments", "frames", "features"],
    }
    assert audio["encoder"]["kind"] == "audio_encoder"
    assert audio["encoder"]["source_owner"] == "Gemma4AudioModel"
    assert audio["encoder"]["num_layers"] == 12
    assert "hidden_size" not in audio["encoder"]
    assert audio["projector"]["kind"] == "linear_projector"
    assert "out_features" not in audio["projector"]
    assert audio["projector"]["source_class"] == "Gemma4MultimodalEmbedder"
    assert [op["kind"] for op in audio["projector"]["ops"]] == ["norm", "linear"]
    assert audio["tokens"] == {"kind": "soft_audio_tokens"}
    assert [step["operation"] for step in audio["pipeline"]] == [
        "input",
        "encode",
        "unknown",
        "emit_soft_token_stream",
    ]

    fusion = data["modalities"]["fusion"]
    assert fusion["placeholders"]["audio"] == {
        "kind": "audio_placeholder",
        "token_id": 258881,
        "begin_token_id": 256000,
        "end_token_id": 258883,
    }
    assert fusion["mechanism"]["kind"] == "scatter_many"
    assert any(
        route["source"] == "modalities.inputs.audio.tokens"
        and route["at"]["kind"] == "audio_placeholder"
        for route in fusion["mechanism"]["routes"]
    )

    assert "description" not in encoded
    assert "label" not in encoded
    assert "title" not in encoded


def test_expanded_json_completes_qwen2_audio_sparse_text_config():
    data = unfold(QWEN2_AUDIO_SPARSE_CONFIG, return_json=True)

    assert data["dimensions"]["hidden_size"] == 4096
    assert data["stack"]["num_layers"] == 32
    assert data["layer_groups"][0]["attention"]["kind"] == "mha"
    assert data["layer_groups"][0]["ffn"]["intermediate_size"] == 11008

    audio = data["modalities"]["inputs"]["audio"]
    assert audio["encoder"]["hidden_size"] == 1280
    assert audio["encoder"]["num_layers"] == 32
    assert audio["encoder"]["num_attention_heads"] == 20
    assert data["modalities"]["fusion"]["placeholders"]["audio"]["token_id"] == 151646


def test_expanded_json_supports_mllama_cross_attention_vision():
    data = unfold(MLLAMA_VISION_TINY_CONFIG, return_json=True)

    vision = data["modalities"]["inputs"]["vision"]
    assert vision["kind"] == "image_to_cross_attention_states"
    assert vision["encoder"]["kind"] == "vision_encoder"
    assert vision["encoder"]["num_attention_heads"] == 16
    # Two exact occurrences of the same block class stay separate: 32 local
    # ungated blocks followed by 8 learned-gated global blocks.  The component
    # total is the sum of those source-bound counts, not the last stage's 8.
    variants = vision["encoder"]["variants"]
    assert [item["repeat"] for item in variants] == [32, 8]
    assert [item["residual_gated"] for item in variants] == [False, True]
    assert variants[1]["gate_activation"] == "tanh"
    assert variants[1]["gate_source"] == "parameter"
    assert vision["encoder"]["num_layers"] == 40
    # ``max_num_tiles`` is input geometry.  The modeling forward receives an
    # already-tiled tensor; processor-side tiling is not a model operation.
    assert "tiling" not in vision
    assert {key: vision["projector"][key] for key in
            ("kind", "in_features", "out_features", "source_class")} == {
        "kind": "linear_projector", "in_features": 7680,
        "out_features": 4096, "source_class": "Linear",
    }
    assert [op["kind"] for op in vision["projector"]["ops"]] == ["linear"]
    assert vision["tokens"] == {
        "kind": "vision_cross_attention_states",
        "width": 4096,
    }
    assert [step["operation"] for step in vision["pipeline"]] == [
        "input",
        "unknown",
        "encode",
        "unknown",
        "emit_cross_attention_states",
    ]

    fusion = data["modalities"]["fusion"]
    assert fusion["kind"] == "cross_attention"
    assert fusion["operation"] == "condition_decoder_hidden_states"
    assert fusion["target"] == "decoder.cross_attention_layers"
    assert fusion["mechanism"] == {
        "kind": "cross_attention",
        "operation": "cross_attention_states",
        "sources": ["vision"],
        "layers": [3, 8, 13, 18, 23, 28, 33, 38],
        "num_layers": 8,
    }


def test_mllama_cross_attention_is_layer_variant_only():
    diagram = unfold(MLLAMA_VISION_TINY_CONFIG)
    ir = diagram.to_ir()

    assert not ir["layers"][0]["attention"]["cross_attention"]
    assert ir["layers"][3]["attention"]["cross_attention"]
    assert not any(block.get("id") == "cross_attention_states" for block in ir["layers"][0]["blocks"])
    layer3_blocks = ir["layers"][3]["blocks"]
    assert not any(block.get("id") == "vision_path" for block in layer3_blocks)
    side_states = next(block for block in layer3_blocks if block.get("id") == "cross_attention_states")
    # Inspect title matches the block's visible label; the raw tensor name
    # lives in the description (and stays the node id).
    assert side_states["title"] == "Projected image states"
    assert "cross_attention_states" in side_states["description"]
    assert side_states["view"] == "vision_path"
    assert [
        i for i, layer in enumerate(ir["layers"])
        if layer["attention"]["cross_attention"]
    ] == [3, 8, 13, 18, 23, 28, 33, 38]

    # The layer role is exact, but its nested AutoModel attention owner is not
    # source-addressable yet.  Preserve cross-attention while refusing to
    # manufacture GQA from checkpoint head counts alone.
    assert ir["layers"][3]["attention"]["kind"] is None
    html = diagram.to_html(standalone=False)
    assert "Cross-Attention" in html
    assert "GQA XAttn" not in html
    assert "Cross-Attention" in html
    assert "cross_attention_states" in html
    assert "Projected image states" in html
    # The modeling forward receives an already-tiled tensor.  Tile creation
    # lives in the processor, so max_num_tiles is geometry only and cannot
    # fabricate a modeling-code operation in this diagram.
    assert "Flatten spatial grid" not in html
    assert 'data-id="vision_enc_g0_op_selfattn"' in html
    # The vision reader now proves more than a softmax kernel: one exact
    # owner-bound head-count path shapes Q, K and V equally.  That is the MHA
    # mechanism law itself, so this tower may open without borrowing the
    # checkpoint's head counts or a model-family label.
    vision_groups = ir["extras"]["modalities"]["inputs"]["vision"]["encoder"][
        "sub_model"
    ]["groups"]
    assert all(group["attention"]["kind"] == "mha" for group in vision_groups)
    assert all(
        group["attention"]["projection_mode"] == "split_qkv"
        for group in vision_groups
    )
    assert "vision_enc_g0_attn_scaled_scores" in html
    # The separate text-decoder cross-attention remains honestly unresolved;
    # this assertion is scoped to the source-proven vision owner above.
    assert "vision_enc_g0_ffn_" in html
    assert "gated residuals" in html
    assert "Vision context" not in html


def test_expanded_json_supports_qwen_style_unified_grid_stream():
    data = unfold(QWEN2_VL_TINY_CONFIG, return_json=True)
    encoded = json.dumps(data["modalities"])

    vision = data["modalities"]["inputs"]["vision"]
    assert vision["kind"] == "image_to_grid_tokens"
    assert vision["encoder"]["kind"] == "vision_encoder"
    assert vision["encoder"]["num_layers"] == 3
    assert "hidden_size" not in vision["encoder"]
    assert "position_encoding" not in vision["encoder"]
    assert [op["kind"] for op in vision["embedding"]["ops"]][:2] == [
        "reshape", "conv3d"]
    projector = vision["projector"]
    assert projector["kind"] == "patch_merger"
    assert "in_features" not in projector and projector["out_features"] == 64
    assert "profile" not in projector
    assert projector["source_class"] == "PatchMerger"
    assert [op["kind"] for op in projector["ops"]] == [
        "norm", "reshape", "linear", "activation", "linear"
    ]
    assert projector["ops"][3]["fn"] == "gelu"
    assert vision["tokens"] == {
        "kind": "grid_visual_tokens",
        "width": 64,
    }
    assert vision["pipeline"][-1]["operation"] == "emit_grid_token_stream"

    video = data["modalities"]["inputs"]["video"]
    assert video["kind"] == "video_to_grid_tokens"
    assert video["encoder"]["num_layers"] == 3
    assert "hidden_size" not in video["encoder"]
    video_projector = video["projector"]
    assert video_projector["kind"] == "patch_merger"
    assert "in_features" not in video_projector
    assert video_projector["out_features"] == 64
    assert "profile" not in video_projector
    assert video_projector["source_class"] == "PatchMerger"
    assert [op["kind"] for op in video_projector["ops"]] == [
        "norm", "reshape", "linear", "activation", "linear"
    ]
    assert video["tokens"] == {
        "kind": "grid_video_tokens",
        "width": 64,
    }

    fusion = data["modalities"]["fusion"]
    assert fusion["kind"] == "unified_multimodal_stream"
    assert fusion["operation"] == "scatter_grid_tokens_into_placeholder_slots"
    assert fusion["target"] == "stack.input_embeddings"
    assert fusion["mechanism"] == {
        "kind": "grid_placeholder_replace",
        "operation": "scatter_grid_tokens_into_placeholder_slots",
        "sources": ["vision", "video"],
        "position_encoding": "multimodal_rope",
    }
    assert fusion["placeholders"]["image"] == {
        "kind": "image_placeholder",
        "token_id": 151655,
        "begin_token_id": 151652,
        "end_token_id": 151653,
    }
    assert fusion["placeholders"]["video"] == {
        "kind": "video_placeholder",
        "token_id": 151656,
        "begin_token_id": 151652,
        "end_token_id": 151653,
    }

    assert "description" not in encoded
    assert "label" not in encoded
    assert "title" not in encoded


def test_source_missing_modalities_keep_only_opaque_declared_lanes():
    qwen_like = deepcopy(QWEN2_VL_TINY_CONFIG)
    qwen_like.pop("model_type", None)
    qwen_like["architectures"] = []
    qwen_like["vision_config"].pop("model_type", None)
    qwen_like["vision_config"]["architectures"] = []

    qwen_data = unfold(qwen_like, return_json=True)
    qwen_inputs = qwen_data["modalities"]["inputs"]
    assert qwen_inputs["vision"]["kind"] == "code_defined_modality_path"
    assert qwen_inputs["vision"]["encoder"] == {
        "kind": "code_defined_encoder"}
    assert qwen_inputs["vision"]["tokens"] == {
        "kind": "code_defined_tokens"}
    assert qwen_inputs["video"]["kind"] == "code_defined_modality_path"
    assert qwen_data["modalities"]["fusion"]["kind"] == "code_defined_fusion"

    mllama_like = deepcopy(MLLAMA_VISION_TINY_CONFIG)
    mllama_like.pop("model_type", None)
    mllama_like["architectures"] = []
    mllama_like["vision_config"].pop("model_type", None)

    mllama_data = unfold(mllama_like, return_json=True)
    mllama_vision = mllama_data["modalities"]["inputs"]["vision"]
    assert mllama_vision["kind"] == "code_defined_modality_path"
    assert mllama_vision["encoder"] == {"kind": "code_defined_encoder"}
    assert mllama_vision["tokens"] == {"kind": "code_defined_tokens"}
    assert mllama_data["modalities"]["fusion"]["kind"] == "code_defined_fusion"


def test_config_only_patch_fields_do_not_emit_a_dynamic_grid():
    cfg = {
        "architectures": ["Qwen2VLForConditionalGeneration"],
        "model_type": "qwen2_vl",
        "_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
        "vocab_size": 152064,
        "image_token_id": 151655,
        "text_config": {
            "model_type": "qwen2",
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 152064,
            "rms_norm_eps": 1e-6,
        },
        "vision_config": {
            "model_type": "qwen2_vl_vision",
            "embed_dim": 80,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "depth": 2,
            "num_heads": 4,
        },
    }
    vision = unfold(cfg, return_json=True)["modalities"]["inputs"]["vision"]
    assert vision["embedding"]["kind"] == "code_defined_embedding"
    assert "grid" not in vision["embedding"]


def test_config_only_non_square_patch_fields_do_not_create_a_grid():
    cfg = dict(LLAMA_TINY_CONFIG)
    cfg.update({
        "architectures": ["LlavaForConditionalGeneration"],
        "model_type": "llava",
        "vision_config": {
            "model_type": "clip_vision_model",
            "hidden_size": 32,
            "image_size": 448,
            "patch_size_h": 14,
            "patch_size_w": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
        },
    })
    vision = unfold(cfg, return_json=True)["modalities"]["inputs"]["vision"]
    assert "grid" not in vision["embedding"]


def test_config_only_pixel_shuffle_declaration_cannot_create_reduction():
    cfg = {
        "architectures": ["InternVLForConditionalGeneration"], "model_type": "internvl",
        "image_token_id": 151667, "downsample_ratio": 0.5,
        "vision_feature_layer": -1, "vision_feature_select_strategy": "default",
        "projector_hidden_act": "gelu",
        "vision_config": {"model_type": "internvl_vision", "hidden_size": 64, "num_hidden_layers": 4,
                          "num_attention_heads": 4, "patch_size": 14, "image_size": 448, "intermediate_size": 256},
        "text_config": {"model_type": "qwen2", "hidden_size": 128, "intermediate_size": 512,
                        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                        "vocab_size": 1000, "rms_norm_eps": 1e-6},
    }
    v = unfold(cfg, return_json=True)["modalities"]["inputs"]["vision"]
    assert "reduction" not in v
    assert "feature_layer" not in v["encoder"]
    assert "feature_select_strategy" not in v["encoder"]
    # Installed source proves an MLP connector independently of the unproven
    # pixel-shuffle declaration; preserving that projector does not license a
    # reduction stage.
    assert v["projector"]["kind"] == "mlp_projector"
    assert [item["kind"] for item in v["projector"]["ops"]] == [
        "norm", "linear", "activation", "linear"]
    assert all(step["operation"] != "pixel_shuffle"
               for step in v["pipeline"])


def test_vision_perceiver_resampler_idefics2_style():
    """Idefics2: perceiver_config -> a resampler connector emitting fixed latents."""
    cfg = {
        "architectures": ["Idefics2ForConditionalGeneration"], "model_type": "idefics2",
        "image_token_id": 32001,
        "perceiver_config": {"model_type": "idefics2", "resampler_n_latents": 64},
        "vision_config": {"model_type": "idefics2", "hidden_size": 64, "num_hidden_layers": 4,
                          "num_attention_heads": 4, "patch_size": 14, "image_size": 980, "intermediate_size": 256},
        "text_config": {"model_type": "mistral", "hidden_size": 128, "intermediate_size": 512,
                        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                        "vocab_size": 1000, "rms_norm_eps": 1e-6},
    }
    v = unfold(cfg, return_json=True)["modalities"]["inputs"]["vision"]
    assert v["projector"]["kind"] == "perceiver_resampler"
    assert v["projector"]["learned_queries"] is True
    # Source proves a learned-query repeated resampler.  It does not yet prove
    # which Parameter dimension is the emitted token axis, so the checkpoint's
    # ``resampler_n_latents`` value cannot be copied as an output count.
    assert "num_latents" not in v["projector"]
    assert "count" not in v["tokens"]


def test_processor_anyres_config_does_not_create_a_model_tiling_stage():
    cfg = {
        "architectures": ["LlavaOnevisionForConditionalGeneration"], "model_type": "llava_onevision",
        "image_token_index": 151646, "vision_feature_layer": -1, "vision_feature_select_strategy": "full",
        "image_grid_pinpoints": [[384, 384], [384, 768], [768, 384]], "vision_aspect_ratio": "anyres_max_9",
        "projector_hidden_act": "gelu",
        "vision_config": {"model_type": "siglip_vision_model", "hidden_size": 64, "num_hidden_layers": 4,
                          "num_attention_heads": 4, "patch_size": 14, "image_size": 384, "intermediate_size": 256},
        "text_config": {"model_type": "qwen2", "hidden_size": 128, "intermediate_size": 512,
                        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                        "vocab_size": 1000, "rms_norm_eps": 1e-6},
    }
    v = unfold(cfg, return_json=True)["modalities"]["inputs"]["vision"]
    assert "tiling" not in v
    # Source may prove the layer/token-selection operations, but the config's
    # selector spelling/value is not itself architecture and is not copied.
    if "feature_operations" in v["encoder"]:
        assert {item["kind"] for item in v["encoder"]["feature_operations"]} >= {
            "single_layer_select", "drop_first_token"}
    assert all(step["operation"] != "tile_image" for step in v["pipeline"])
