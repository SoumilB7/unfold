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

GEMMA4_VISION_TINY_CONFIG = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "_name_or_path": "google/gemma-4-e2b",
    "image_token_id": 262144,
    "image_seq_length": 280,
    "image_token_count_options": [70, 140, 280, 560, 1120],
    "projector_hidden_act": "gelu_pytorch_tanh",
    "text_config": {
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4_text",
        "vocab_size": 262208,
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "max_position_embeddings": 1024,
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
    },
    "vision_config": {
        "architectures": ["Gemma4VisionModel"],
        "model_type": "gemma4_vision",
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "image_size": 896,
        "patch_size": 16,
        # Real gemma-4 vision structure: learned 2D positions + 2D RoPE, and a
        # k×k average pool that reduces the token count after the encoder.
        "position_embedding_size": 256,
        "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
        "pooling_kernel_size": 3,
        "global_head_dim": 8,
    },
}


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


MLLAMA_VISION_TINY_CONFIG = {
    "architectures": ["MllamaForConditionalGeneration"],
    "model_type": "mllama",
    "_name_or_path": "meta-llama/Llama-3.2-11B-Vision",
    "image_token_index": 128256,
    "vision_config": {
        "model_type": "mllama_vision_model",
        "hidden_size": 1280,
        "vision_output_dim": 7680,
        "num_hidden_layers": 32,
        "num_global_layers": 8,
        "attention_heads": 16,
        "image_size": 448,
        "patch_size": 14,
        "max_num_tiles": 4,
    },
    "text_config": {
        "model_type": "mllama_text_model",
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "cross_attention_layers": [3, 8, 13, 18, 23, 28, 33, 38],
        "max_position_embeddings": 131072,
        "hidden_act": "silu",
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
        "model_type": "qwen2_vl",
        "embed_dim": 32,
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
    },
}


def test_expanded_json_is_structural_not_renderer_copy():
    data = unfold(LLAMA_TINY_CONFIG, return_json=True)
    encoded = json.dumps(data)

    assert data["schema_version"] == "3.0"
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
    assert data["layer_groups"][0]["attention"]["trace"]["code_finding_ids"]


def test_expanded_json_carries_structured_multimodal_inputs():
    data = unfold(GEMMA4_VISION_TINY_CONFIG, return_json=True)
    encoded = json.dumps(data["modalities"])

    vision = data["modalities"]["inputs"]["vision"]
    assert vision["kind"] == "image_to_soft_visual_tokens"
    assert vision["input"] == {
        "kind": "image_pixels",
        "shape": ["batch", "images", "channels", "height", "width"],
        "image_size": 896,
        "patch_size": 16,
    }
    assert vision["embedding"] == {
        "kind": "patch_embedding",
        "patch_size": 16,
        "out_features": 32,
        "grid": {
            "kind": "static_patch_grid",
            "patch": {"h": 16, "w": 16},
            "input": {"h": 896, "w": 896},
            "tiles": {"h": 56, "w": 56},
        },
    }
    assert vision["encoder"]["kind"] == "gemma4_vision"
    assert vision["encoder"]["hidden_size"] == 32
    # Position encoding is derived structurally (learned table + 2D RoPE), no family hint.
    assert vision["encoder"]["position_encoding"] == {
        "kind": "learned_2d_plus_rope_2d",
        "rope": {"rope_theta": 100.0, "rope_type": "default"},
    }
    assert vision["encoder"]["global_head_dim"] == 8
    # k=3 average pool surfaces as a post-encoder token-reduction section + stage.
    assert vision["reduction"] == {
        "kind": "token_pooling",
        "kernel_size": 3,
        "reduces_tokens_by": 9,
    }
    assert vision["projector"]["out_features"] == 64
    assert vision["tokens"] == {
        "kind": "soft_visual_tokens",
        "count": 280,
        "count_options": [70, 140, 280, 560, 1120],
        "width": 64,
    }
    assert [step["operation"] for step in vision["pipeline"]] == [
        "input",
        "patch_embedding",
        "encode",
        "pool_tokens",
        "project_to_text_width",
        "emit_soft_token_stream",
    ]
    assert vision["pipeline"][-1] == {
        "id": "soft_visual_tokens",
        "operation": "emit_soft_token_stream",
        "kind": "soft_visual_tokens",
        "count": 280,
        "width": 64,
    }

    fusion = data["modalities"]["fusion"]
    assert fusion["kind"] == "placeholder_replace"
    assert fusion["operation"] == "scatter_soft_tokens_into_placeholder_slots"
    assert fusion["placeholder"] == {"kind": "image_placeholder", "token_id": 262144}
    assert fusion["mechanism"] == {
        "kind": "scatter",
        "source": "modalities.inputs.vision.tokens",
        "into": "io.token_embedding",
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
        "feature_size": 128,
    }
    assert audio["encoder"]["kind"] == "gemma4_audio"
    assert audio["encoder"]["hidden_size"] == 1024
    assert audio["encoder"]["num_layers"] == 12
    assert audio["projector"] == {
        "kind": "linear_projector",
        "in_features": 1024,
        "out_features": 64,
    }
    assert audio["tokens"] == {
        "kind": "soft_audio_tokens",
        "count": 750,
        "ms_per_token": 40,
        "width": 64,
    }
    assert [step["operation"] for step in audio["pipeline"]] == [
        "input",
        "encode",
        "project_to_text_width",
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
    assert vision["encoder"]["kind"] == "mllama_vision_model"
    assert vision["encoder"]["num_attention_heads"] == 16
    # Structural: local+global layer split and the wide concatenated output.
    assert vision["encoder"]["num_global_layers"] == 8
    assert vision["encoder"]["output_dim"] == 7680
    # max_num_tiles -> an image-tiling section + stage (image split into N tiles).
    assert vision["tiling"] == {"kind": "image_tiling", "mode": "fixed_tiles", "max_tiles": 4}
    assert vision["projector"] == {
        "kind": "linear_projector",
        "in_features": 7680,
        "out_features": 4096,
    }
    assert vision["tokens"] == {
        "kind": "vision_cross_attention_states",
        "count": 1025,
        "width": 4096,
    }
    assert [step["operation"] for step in vision["pipeline"]] == [
        "input",
        "tile_image",
        "patch_embedding",
        "encode",
        "project_to_decoder_width",
        "emit_cross_attention_states",
    ]

    fusion = data["modalities"]["fusion"]
    assert fusion["kind"] == "cross_attention"
    assert fusion["operation"] == "condition_decoder_hidden_states"
    assert fusion["target"] == "decoder.cross_attention_layers"
    assert fusion["mechanism"] == {
        "kind": "cross_attention",
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
    assert side_states["detail_view"] == "vision_path"
    assert [
        i for i, layer in enumerate(ir["layers"])
        if layer["attention"]["cross_attention"]
    ] == [3, 8, 13, 18, 23, 28, 33, 38]

    html = diagram.to_html(standalone=False)
    assert "GQA XAttn" in html
    assert "Cross-Attention" in html
    assert "cross_attention_states" in html
    assert "Projected image states" in html
    assert "Flatten patches" in html
    assert "Vision self-attention" in html
    assert "vision_attn_scaled" in html
    assert "vision_mlp_fc1" in html
    assert "separate vision tower" in html
    assert "Vision context" not in html


def test_expanded_json_supports_qwen_style_unified_grid_stream():
    data = unfold(QWEN2_VL_TINY_CONFIG, return_json=True)
    encoded = json.dumps(data["modalities"])

    vision = data["modalities"]["inputs"]["vision"]
    assert vision["kind"] == "image_to_grid_tokens"
    assert vision["encoder"]["kind"] == "qwen_vl_vision_transformer"
    assert vision["embedding"]["out_features"] == 32
    assert vision["encoder"]["hidden_size"] == 32
    assert vision["encoder"]["position_encoding"] == {"kind": "multimodal_rope"}
    assert vision["projector"] == {
        "kind": "patch_merger",
        "in_features": 128,
        "out_features": 64,
    }
    assert vision["tokens"] == {
        "kind": "grid_visual_tokens",
        "width": 64,
        "grid": {
            "kind": "dynamic_thw_grid",
            "runtime_input": "image_grid_thw",
            "axes": ["time", "height", "width"],
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "position_encoding": "multimodal_rope",
        },
    }
    assert vision["pipeline"][-1]["operation"] == "emit_grid_token_stream"

    video = data["modalities"]["inputs"]["video"]
    assert video["kind"] == "video_to_grid_tokens"
    assert video["embedding"]["out_features"] == 32
    assert video["encoder"]["hidden_size"] == 32
    assert video["projector"] == {
        "kind": "patch_merger",
        "in_features": 128,
        "out_features": 64,
    }
    assert video["tokens"] == {
        "kind": "grid_video_tokens",
        "width": 64,
        "grid": {
            "kind": "dynamic_thw_grid",
            "runtime_input": "video_grid_thw",
            "axes": ["time", "height", "width"],
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "position_encoding": "multimodal_rope",
        },
    }

    fusion = data["modalities"]["fusion"]
    assert fusion["kind"] == "unified_multimodal_stream"
    assert fusion["operation"] == "interleave_modal_tokens"
    assert fusion["target"] == "stack.input_embeddings"
    assert fusion["mechanism"] == {
        "kind": "interleave_grid_streams",
        "sources": ["vision", "video"],
        "position_encoding": "multimodal_rope",
        "runtime_grid_inputs": ["image_grid_thw", "video_grid_thw"],
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


def test_multimodal_detection_uses_structural_fields_without_family_model_type():
    qwen_like = deepcopy(QWEN2_VL_TINY_CONFIG)
    qwen_like.pop("model_type", None)
    qwen_like["architectures"] = []
    qwen_like["vision_config"].pop("model_type", None)
    qwen_like["vision_config"]["architectures"] = []

    qwen_data = unfold(qwen_like, return_json=True)
    assert qwen_data["modalities"]["inputs"]["vision"]["kind"] == "image_to_grid_tokens"
    assert qwen_data["modalities"]["inputs"]["vision"]["tokens"]["grid"]["runtime_input"] == "image_grid_thw"
    assert qwen_data["modalities"]["inputs"]["video"]["kind"] == "video_to_grid_tokens"
    assert qwen_data["modalities"]["fusion"]["kind"] == "unified_multimodal_stream"

    mllama_like = deepcopy(MLLAMA_VISION_TINY_CONFIG)
    mllama_like.pop("model_type", None)
    mllama_like["architectures"] = []
    mllama_like["vision_config"].pop("model_type", None)

    mllama_data = unfold(mllama_like, return_json=True)
    assert mllama_data["modalities"]["inputs"]["vision"]["kind"] == "image_to_cross_attention_states"
    assert mllama_data["modalities"]["inputs"]["vision"]["tokens"]["count"] == 1025
    assert mllama_data["modalities"]["fusion"]["kind"] == "cross_attention"


def test_dynamic_resolution_vision_emits_dynamic_patch_grid():
    """A Qwen2-VL-style tower has no fixed image_size; grid must be dynamic."""
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
    grid = unfold(cfg, return_json=True)["modalities"]["inputs"]["vision"]["embedding"]["grid"]
    assert grid == {
        "kind": "dynamic_patch_grid",
        "patch": {"h": 14, "w": 14, "t": 2},
        "spatial_merge_size": 2,
    }


def test_non_square_patch_grid_keeps_both_axes():
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
    grid = unfold(cfg, return_json=True)["modalities"]["inputs"]["vision"]["embedding"]["grid"]
    assert grid["patch"] == {"h": 14, "w": 16}
    assert grid["tiles"] == {"h": 32, "w": 28}


def test_vision_pixel_shuffle_connector_internvl_style():
    """InternVL: downsample_ratio -> a pixel-shuffle token-reduction stage."""
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
    assert v["reduction"] == {"kind": "pixel_shuffle", "downsample_ratio": 0.5, "reduces_tokens_by": 4}
    assert v["encoder"]["feature_layer"] == -1
    assert v["encoder"]["feature_select_strategy"] == "default"
    assert v["projector"]["kind"] == "mlp_projector"
    assert [s["operation"] for s in v["pipeline"]] == [
        "input", "patch_embedding", "encode", "pixel_shuffle", "project_to_text_width", "emit_soft_token_stream",
    ]


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
    assert v["projector"]["num_latents"] == 64
    # the resampler emits a fixed token count, not the input patch count
    assert v["tokens"]["count"] == 64


def test_vision_anyres_tiling_llava_onevision_style():
    """LLaVA-OneVision: image_grid_pinpoints -> any-res tiling + 'full' feature select."""
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
    assert v["tiling"] == {"kind": "image_tiling", "mode": "anyres", "num_layouts": 3,
                           "aspect_ratio_policy": "anyres_max_9"}
    assert v["encoder"]["feature_select_strategy"] == "full"
    assert [s["operation"] for s in v["pipeline"]] == [
        "input", "tile_image", "patch_embedding", "encode", "project_to_text_width", "emit_soft_token_stream",
    ]
