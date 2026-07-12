"""Importable, top-level shared test fixtures (§16.1 fixture-isolation).

These model config dicts were previously defined inside individual test
modules and imported cross-module (`from tests.test_diffusion import FLUX`,
etc.), which made the grouped hardening gate invalid — a test file's collection
depended on ANOTHER test module importing cleanly.  They now live here, in a
package that is NOT a test module, so every audit/hardening file collects and
runs alone.  A static gate (`tests/test_isolation.py`) forbids any production
or test file from importing a `test_*` module.

Real, public config VALUES only — no network, no model code executed.
"""
from __future__ import annotations

# --- diffusion (from the former tests/test_diffusion.py) --------------------

FLUX = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.30.0",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "_scheduler_config": {"num_train_timesteps": 1000, "shift": 3.0},
    "_vae_config": {
        "_class_name": "AutoencoderKL",
        "block_out_channels": [128, 256, 512, 512],
        "latent_channels": 16,
        "out_channels": 3,
        "layers_per_block": 2,
        "scaling_factor": 0.3611,
    },
    "_text_encoder_configs": {
        "text_encoder": {
            "_class_name": "CLIPTextModel", "architectures": ["CLIPTextModel"],
            "model_type": "clip_text_model",
            "num_hidden_layers": 12, "hidden_size": 768,
            "num_attention_heads": 12, "intermediate_size": 3072, "hidden_act": "quick_gelu",
            "max_position_embeddings": 77, "vocab_size": 49408,
        },
        "text_encoder_2": {
            "_class_name": "T5EncoderModel", "architectures": ["T5EncoderModel"],
            "model_type": "t5", "num_layers": 24, "d_model": 4096,
            "num_heads": 64, "d_ff": 10240, "dense_act_fn": "gelu_new", "vocab_size": 32128,
            "is_gated_act": True, "feed_forward_proj": "gated-gelu",
        },
    },
}

PIXART = {
    "_class_name": "PixArtTransformer2DModel",
    "_diffusers_version": "0.27.0",
    "num_layers": 28,
    "num_attention_heads": 16,
    "attention_head_dim": 72,
    "cross_attention_dim": 1152,
    "caption_channels": 4096,
    "patch_size": 2,
    "in_channels": 4,
    "sample_size": 128,
    "norm_type": "ada_norm_single",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
}

LLAMA = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
    "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
}

# --- multimodal styles (from the former tests/test_declared_ops.py) ----------

PIXTRAL_STYLE = {
    "architectures": ["LlavaForConditionalGeneration"], "model_type": "llava",
    "image_token_index": 10, "projector_hidden_act": "gelu",
    "text_config": {"model_type": "mistral", "hidden_size": 5120, "num_hidden_layers": 4,
                    "num_attention_heads": 32, "num_key_value_heads": 8,
                    "intermediate_size": 14336, "vocab_size": 131072,
                    "rms_norm_eps": 1e-5, "head_dim": 128},
    "vision_config": {"model_type": "pixtral", "hidden_size": 1024, "image_size": 1024,
                      "patch_size": 16, "num_hidden_layers": 24,
                      "num_attention_heads": 16, "intermediate_size": 4096},
}

QWEN2VL_STYLE = {
    "architectures": ["Qwen2VLForConditionalGeneration"], "model_type": "qwen2_vl",
    "image_token_id": 151655,
    "text_config": {"model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 4,
                    "num_attention_heads": 28, "num_key_value_heads": 4,
                    "intermediate_size": 18944, "vocab_size": 152064,
                    "rms_norm_eps": 1e-6},
    "vision_config": {"model_type": "qwen2_vl_vision", "embed_dim": 1280,
                      "hidden_size": 3584, "patch_size": 14, "temporal_patch_size": 2,
                      "spatial_merge_size": 2, "depth": 32, "num_heads": 16},
}

MISTRAL3_STYLE = {
    "architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3",
    "image_token_index": 10, "spatial_merge_size": 2,
    "projector_hidden_act": "gelu", "vision_feature_layer": -1,
    "text_config": {"model_type": "mistral", "hidden_size": 5120, "num_hidden_layers": 4,
                    "num_attention_heads": 32, "num_key_value_heads": 8,
                    "intermediate_size": 14336, "vocab_size": 131072,
                    "rms_norm_eps": 1e-5, "head_dim": 128},
    "vision_config": {"model_type": "pixtral", "hidden_size": 1024, "image_size": 1024,
                      "patch_size": 16, "num_hidden_layers": 24,
                      "num_attention_heads": 16, "intermediate_size": 4096},
}

# --- vision tiny configs (from the former tests/test_expanded_json.py) -------

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
        "position_embedding_size": 256,
        "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
        "pooling_kernel_size": 3,
        "global_head_dim": 8,
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

# --- gemma-4 base + builders (from the former tests/test_smoke.py) -----------

GEMMA4_31B_LAYER_TYPES = [
    "sliding_attention" if (i % 6) != 5 else "full_attention" for i in range(60)
]

GEMMA4_31B_CONFIG = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "_name_or_path": "google/gemma-4-31B",
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "gemma4_text",
        "vocab_size": 262144,
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "num_global_key_value_heads": 4,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "layer_types": GEMMA4_31B_LAYER_TYPES,
        "enable_moe_block": False,
        "attention_k_eq_v": True,
    },
}


def gemma4_e4b_config():
    cfg = dict(GEMMA4_31B_CONFIG)
    text_cfg = dict(GEMMA4_31B_CONFIG["text_config"])
    text_cfg.update(
        {
            "num_hidden_layers": 42,
            "layer_types": [
                "sliding_attention" if (i % 6) != 5 else "full_attention"
                for i in range(42)
            ],
            "hidden_size_per_layer_input": 1024,
            "vocab_size_per_layer_input": text_cfg["vocab_size"],
        }
    )
    cfg["_name_or_path"] = "google/gemma-4-E4B"
    cfg["text_config"] = text_cfg
    return cfg


def gemma4_e2b_vision_config():
    cfg = dict(GEMMA4_31B_CONFIG)
    text_cfg = dict(GEMMA4_31B_CONFIG["text_config"])
    text_cfg.update(
        {
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 256,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "max_position_embeddings": 131072,
        }
    )
    cfg.update(
        {
            "_name_or_path": "google/gemma-4-E2B",
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "boi_token_id": 255999,
            "boa_token_id": 256000,
            "eoi_token_id": 258882,
            "eoa_token_id": 258883,
            "image_seq_length": 280,
            "audio_seq_length": 750,
            "audio_ms_per_token": 40,
            "image_token_count_options": [70, 140, 280, 560, 1120],
            "projector_hidden_act": "gelu_pytorch_tanh",
            "text_config": text_cfg,
            "vision_config": {
                "architectures": ["Gemma4VisionModel"],
                "model_type": "gemma4_vision",
                "hidden_size": 768,
                "num_hidden_layers": 16,
                "num_attention_heads": 12,
                "image_size": 896,
                "patch_size": 16,
            },
            "audio_config": {
                "architectures": ["Gemma4AudioModel"],
                "model_type": "gemma4_audio",
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "output_proj_dims": 1536,
                "feature_size": 128,
            },
        }
    )
    return cfg


# Back-compat aliases for the private-named builders the old test modules used.
_gemma4_e4b_config = gemma4_e4b_config
_gemma4_e2b_vision_config = gemma4_e2b_vision_config


# --- additional shared fixtures relocated for §16.1 isolation ---

SDXL_UNET = {
    "_class_name": "UNet2DConditionModel", "_repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "in_channels": 4, "out_channels": 4, "block_out_channels": [320, 640, 1280],
    "layers_per_block": 2, "cross_attention_dim": 2048, "transformer_layers_per_block": [1, 2, 10],
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    "mid_block_type": "UNetMidBlock2DCrossAttn", "addition_embed_type": "text_time",
    "scheduler": ["diffusers", "EulerDiscreteScheduler"], "_scheduler_config": {"num_train_timesteps": 1000},
    "text_encoder": ["transformers", "CLIPTextModel"], "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
}

HYBRID_ENC = {**FLUX, "_text_encoder_configs": {
    "text_encoder": {
        "_class_name": "LlamaModel", "architectures": ["LlamaForCausalLM"],
        "model_type": "llama", "num_hidden_layers": 24, "hidden_size": 2048,
        "num_attention_heads": 16, "num_key_value_heads": 4,
        "intermediate_size": 5632, "hidden_act": "silu", "rms_norm_eps": 1e-5,
        "vocab_size": 32000, "max_position_embeddings": 8192,
        "rope_theta": 10000.0, "sliding_window": 4096,
        "layer_types": ["sliding_attention", "full_attention"] * 12,
    },
}}

MOE_ENC = {**FLUX, "_text_encoder_configs": {
    "text_encoder": {
        "_class_name": "MixtralModel", "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral", "num_hidden_layers": 32, "hidden_size": 4096,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 14336, "hidden_act": "silu", "rms_norm_eps": 1e-5,
        "vocab_size": 32000, "max_position_embeddings": 32768, "rope_theta": 1e6,
        "num_local_experts": 8, "num_experts_per_tok": 2,
    },
}}

MUSICGEN_SMALL = {
    "model_type": "musicgen",
    "architectures": ["MusicgenForConditionalGeneration"],
    "is_encoder_decoder": True,
    "text_encoder": {
        "model_type": "t5", "d_model": 768, "num_layers": 12, "num_heads": 12,
        "d_ff": 3072, "d_kv": 64, "vocab_size": 32128,
        "feed_forward_proj": "relu", "dense_act_fn": "relu", "is_gated_act": False,
        "relative_attention_num_buckets": 32,
    },
    "audio_encoder": {
        "model_type": "encodec", "sampling_rate": 32000, "audio_channels": 1,
        "num_filters": 64, "upsampling_ratios": [8, 5, 4, 4],
        "codebook_size": 2048, "codebook_dim": 128, "num_lstm_layers": 2,
        "hidden_size": 128, "num_residual_layers": 1,
    },
    "decoder": {
        "model_type": "musicgen_decoder", "vocab_size": 2048,
        "max_position_embeddings": 2048, "num_hidden_layers": 24,
        "ffn_dim": 4096, "num_attention_heads": 16, "hidden_size": 1024,
        "activation_function": "gelu", "num_codebooks": 4, "audio_channels": 1,
        "scale_embedding": False, "tie_word_embeddings": False,
    },
}

STABLE_AUDIO = {
    "_class_name": "StableAudioDiTModel",
    "sample_size": 1024, "in_channels": 64, "out_channels": 64,
    "num_layers": 24, "attention_head_dim": 64, "num_attention_heads": 24,
    "num_key_value_attention_heads": 12,
    "cross_attention_dim": 768, "cross_attention_input_dim": 768,
    "global_states_input_dim": 1536, "time_proj_dim": 256,
    "text_encoder": ["transformers", "T5EncoderModel"],
    "_vae_config": {
        "_class_name": "AutoencoderOobleck", "audio_channels": 2,
        "sampling_rate": 44100, "decoder_channels": 128,
        "channel_multiples": [1, 2, 4, 8, 16],
        "downsampling_ratios": [2, 4, 4, 8, 8], "decoder_input_channels": 64,
    },
}

_BASE = dict(num_hidden_layers=2, hidden_size=128, num_attention_heads=8,
             num_key_value_heads=2, intermediate_size=256, vocab_size=1000, rms_norm_eps=1e-5)

_VISION_CFG = {"model_type": "qwen2_vl", "architectures": ["Qwen2VisionTransformerPretrainedModel"],
               "depth": 4, "hidden_size": 128, "num_heads": 8, "patch_size": 14, "in_channels": 3}

_GRID_VISION = {**_VISION_CFG, "spatial_merge_size": 2, "temporal_patch_size": 2}

CORPUS = {
    "dense_gated":  dict(_BASE, model_type="llama", hidden_act="silu"),          # attention, gated_ffn
    "dense_mlp":    dict(_BASE, model_type="phi", num_key_value_heads=8, hidden_act="gelu_new"),  # dense_ffn
    "moe_mla_mtp":  dict(_BASE, model_type="deepseek_v3", kv_lora_rank=64, q_lora_rank=96,
                         qk_nope_head_dim=64, qk_rope_head_dim=32, n_routed_experts=8,
                         num_experts_per_tok=2, moe_intermediate_size=128, first_k_dense_replace=1,
                         n_shared_experts=1, scoring_func="sigmoid", topk_method="noaux_tc",
                         n_group=4, topk_group=2, norm_topk_prob=True, routed_scaling_factor=2.5,
                         num_nextn_predict_layers=1),  # moe, moe_router, moe_expert, mla_*, mtp_*
    "dsa":          dict(_BASE, model_type="deepseek_v32", kv_lora_rank=64, q_lora_rank=96,
                         qk_nope_head_dim=64, qk_rope_head_dim=32, n_routed_experts=8,
                         num_experts_per_tok=2, moe_intermediate_size=128, first_k_dense_replace=1,
                         index_topk=2048, index_n_heads=64, index_head_dim=128),  # dsa_indexer
    "ple":          dict(_BASE, model_type="m", hidden_size_per_layer_input=64,
                         vocab_size_per_layer_input=1000),                         # per_layer_embedding
    "self_cond":    dict(model_type="diffusion_gemma", canvas_length=256,
                         text_config=dict(_BASE, n_routed_experts=8, num_experts_per_tok=2,
                                          moe_intermediate_size=128)),             # self_conditioning
                         # canvas_length is what the REAL config declares — block-diffusion
                         # detection is config-first, never a model_type spelling.
    "vision":       dict(_BASE, model_type="qwen2_vl", vision_config=_VISION_CFG,
                         image_token_id=4),  # vision_path/encoder/self_attention/mlp/patch_embedding, multimodal_fusion
    "audio":        dict(_BASE, model_type="qwen2_audio",
                         audio_config={"num_hidden_layers": 4, "d_model": 128,
                                       "encoder_attention_heads": 8}),             # audio_path/encoder
    "video":        dict(_BASE, model_type="qwen2_vl", vision_config=_GRID_VISION,
                         image_token_id=4, video_token_id=5),                      # video_path/encoder
    "codec_lm":     MUSICGEN_SMALL,  # conditioning_path + text_encoder tower, K-codebook heads
    "dit_audio":    STABLE_AUDIO,    # 1-D audio latent, no patchify, oobleck ladder
    "dit_mmdit":    FLUX,        # attention, ffn, scheduler_step, vae_decoder(_block), text_encoder, encoded_text_concat
    "dit_cross":    PIXART,      # cross_attention
    "unet":         SDXL_UNET,   # unet, unet_stage, unet_resnet, unet_transformer
    # A pipeline whose text encoder is a HETEROGENEOUS stack (sliding/global
    # alternation) — exercises the grouped encoder tower (per-layer-type cells
    # + per-group drills) through every universal net.
    "dit_hybrid_encoder": {**FLUX, "_text_encoder_configs": {
        "text_encoder": {
            "_class_name": "LlamaModel", "architectures": ["LlamaForCausalLM"],
            "model_type": "llama", "num_hidden_layers": 24, "hidden_size": 2048,
            "num_attention_heads": 16, "num_key_value_heads": 4,
            "intermediate_size": 5632, "hidden_act": "silu", "rms_norm_eps": 1e-5,
            "vocab_size": 32000, "max_position_embeddings": 8192,
            "rope_theta": 10000.0, "sliding_window": 4096,
            "layer_types": ["sliding_attention", "full_attention"] * 12,
        },
    }},
    # An MoE text encoder — the canonical router/top-k/expert drill transplanted
    # into a supporting tower, checked by every universal net.
    # Refiner-bearing DiT: exercises the GENERAL secondary-stack detection
    # (config-count-bound ModuleList that is not the root stack) and the
    # refiner_tower view — the token refiner drawn on the text rail.
    "dit_refiner": {
        "_class_name": "HunyuanVideoTransformer3DModel",
        "num_layers": 2, "num_single_layers": 2, "num_refiner_layers": 2,
        "num_attention_heads": 4, "attention_head_dim": 32,
        "in_channels": 16, "out_channels": 16,
        "patch_size": 2, "patch_size_t": 1, "mlp_ratio": 4.0,
        "pooled_projection_dim": 64, "text_embed_dim": 128,
        "qk_norm": "rms_norm", "rope_axes_dim": [8, 12, 12],
        "rope_theta": 256.0, "guidance_embeds": True,
    },
    "dit_moe_encoder": {**FLUX, "_text_encoder_configs": {
        "text_encoder": {
            "_class_name": "MixtralModel", "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral", "num_hidden_layers": 32, "hidden_size": 4096,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "intermediate_size": 14336, "hidden_act": "silu", "rms_norm_eps": 1e-5,
            "vocab_size": 32000, "max_position_embeddings": 32768,
            "rope_theta": 1e6, "num_local_experts": 8, "num_experts_per_tok": 2,
        },
    }},
}
