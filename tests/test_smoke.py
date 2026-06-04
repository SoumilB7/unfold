"""Smoke test: parse real configs and verify IR + HTML output."""
import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_unfolder import unfold
from model_unfolder.adapters.transformer.parser import parse

KIMI_K2_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "_name_or_path": "moonshotai/Kimi-K2-Instruct",
    "vocab_size": 163840,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "num_hidden_layers": 61,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "max_position_embeddings": 131072,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "v_head_dim": 128,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
}

DEEPSEEK_V3_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "_name_or_path": "deepseek-ai/DeepSeek-V3",
    "vocab_size": 129280,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "v_head_dim": 128,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "moe_layer_freq": 1,
}

LLAMA3_8B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "_name_or_path": "meta-llama/Meta-Llama-3-8B",
    "vocab_size": 128256,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 8192,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}

MISTRAL_7B_CONFIG = {
    "architectures": ["MistralForCausalLM"],
    "model_type": "mistral",
    "_name_or_path": "mistralai/Mistral-7B-v0.3",
    "vocab_size": 32768,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}


GPT_NEOX_CONFIG = {
    "architectures": ["GPTNeoXForCausalLM"],
    "model_type": "gpt_neox",
    "_name_or_path": "EleutherAI/gpt-neox-20b",
    "vocab_size": 50432,
    "hidden_size": 6144,
    "intermediate_size": 24576,
    "num_hidden_layers": 44,
    "num_attention_heads": 64,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
    "hidden_act": "gelu",
    "use_parallel_residual": True,
}


FALCON_PARALLEL_CONFIG = {
    "architectures": ["FalconForCausalLM"],
    "model_type": "falcon",
    "_name_or_path": "tiiuae/falcon-7b",
    "vocab_size": 65024,
    "hidden_size": 4544,
    "intermediate_size": 18176,
    "num_hidden_layers": 32,
    "num_attention_heads": 71,
    "multi_query": True,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
    "parallel_attn": True,
}

GEMMA1_CONFIG = {
    "architectures": ["GemmaForCausalLM"],
    "model_type": "gemma",
    "_name_or_path": "google/gemma-7b",
    "vocab_size": 256000,
    "hidden_size": 3072,
    "intermediate_size": 24576,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 1,
    "max_position_embeddings": 8192,
    "tie_word_embeddings": True,
    "hidden_activation": "gelu_pytorch_tanh",
}

PHI2_CONFIG = {
    "architectures": ["PhiForCausalLM"],
    "model_type": "phi",
    "_name_or_path": "microsoft/phi-2",
    "vocab_size": 51200,
    "hidden_size": 2560,
    "intermediate_size": 10240,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
    "hidden_act": "gelu_new",
    "partial_rotary_factor": 0.4,
}

YI_34B_CONFIG = {
    "architectures": ["YiForCausalLM"],
    "model_type": "yi",
    "_name_or_path": "01-ai/Yi-34B",
    "vocab_size": 64000,
    "hidden_size": 7168,
    "intermediate_size": 20480,
    "num_hidden_layers": 60,
    "num_attention_heads": 56,
    "num_key_value_heads": 8,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}

OLMO_7B_CONFIG = {
    "architectures": ["OlmoForCausalLM"],
    "model_type": "olmo",
    "_name_or_path": "allenai/OLMo-7B",
    "vocab_size": 50304,
    "d_model": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "mlp_ratio": 8,
    "max_sequence_length": 2048,
    "tie_word_embeddings": False,
    "activation_type": "swiglu",
    "norm_type": "layer_norm",
}

OLMOE_CONFIG = {
    "architectures": ["OlmoeForCausalLM"],
    "model_type": "olmoe",
    "_name_or_path": "allenai/OLMoE-1B-7B-0924",
    "vocab_size": 50304,
    "hidden_size": 2048,
    "intermediate_size": 1024,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "hidden_act": "swiglu",
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "expert_intermediate_size": 1024,
    "use_qk_norm": True,
    "norm_type": "rms_norm",
}

DBRX_CONFIG = {
    "architectures": ["DbrxForCausalLM"],
    "model_type": "dbrx",
    "_name_or_path": "databricks/dbrx-base",
    "vocab_size": 100352,
    "d_model": 6144,
    "n_layers": 40,
    "n_heads": 48,
    "max_seq_len": 32768,
    "tie_word_embeddings": False,
    "attn_config": {
        "kv_n_heads": 8,
        "clip_qkv": 8,
        "rope_theta": 500000,
    },
    "ffn_config": {
        "ffn_act_fn": {"name": "silu"},
        "ffn_hidden_size": 3584,
        "moe_num_experts": 16,
        "moe_top_k": 4,
    },
    "router_aux_loss_coef": 0.05,
}


# Gemma 4 31B (dense) — alternates 5 sliding + 1 full across 60 layers, with
# distinct head_dim and num_kv_heads on global vs sliding layers.
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


def _gemma4_e4b_config():
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


def _gemma4_e2b_vision_config():
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
        "activation_function": "gelu",
        "scale_embedding": False,
        "max_source_positions": 1500,
    },
    "text_config": {
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "intermediate_size": 11008,
        "max_position_embeddings": 8192,
        "model_type": "qwen2",
        "rope_theta": 10000,
        "rms_norm_eps": 1e-5,
        "sliding_window": 32768,
        "torch_dtype": "bfloat16",
        "use_mrope": False,
        "vocab_size": 156032,
    },
}


def test_kimi_k2():
    d = unfold(KIMI_K2_CONFIG)
    ir = d.to_ir()
    assert ir["name"] == "Kimi-K2-Instruct"
    assert ir["vocab_size"] == 163840
    assert len(ir["layers"]) == 61
    assert ir["layers"][0]["ffn"]["kind"] == "dense"
    assert ir["layers"][1]["ffn"]["kind"] == "moe"
    assert ir["layers"][1]["ffn"]["num_experts"] == 384
    assert ir["layers"][0]["attention"]["kind"] == "mla"

    # param estimates surface on the IR
    assert ir["params"]["is_sparse"] is True
    assert ir["params"]["total"] > ir["params"]["active"]

    html = d.to_html(standalone=True)
    assert "<!doctype html>" in html.lower()
    assert "Unfold" in html
    assert "<script" in html.lower()                # click-to-inspect handler
    assert "uf-card-detail" in html                 # detail panels are present
    assert "<svg" in html.lower()

    fragment = d._repr_html_()
    assert "<script" in fragment.lower()
    assert "<style>" in fragment
    assert "<svg" in fragment.lower()
    assert d._mount_id in fragment

    print(f"Kimi K2 OK  — ~{ir['params']['total_h']} total / {ir['params']['active_h']} active")


def test_deepseek_v3_phase_change():
    d = unfold(DEEPSEEK_V3_CONFIG)
    ir = d.to_ir()
    assert len(ir["layers"]) == 61
    for i in range(3):
        assert ir["layers"][i]["ffn"]["kind"] == "dense"
    for i in range(3, 61):
        assert ir["layers"][i]["ffn"]["kind"] == "moe"
    assert ir["layers"][3]["ffn"]["num_experts"] == 256
    print(f"DeepSeek-V3 phase change OK  — ~{ir['params']['total_h']} total / {ir['params']['active_h']} active")


def test_mtp_head_detected_and_rendered():
    d = unfold({**DEEPSEEK_V3_CONFIG, "num_nextn_predict_layers": 1})
    ir = d.to_ir()

    assert ir["extras"]["mtp"]["num_modules"] == 1
    mtp = next(b for b in ir["extras"]["render"]["model_blocks"] if b["id"] == "mtp")
    assert mtp["role"] == "mtp" and mtp["view"] == "mtp_head"

    # Surfaced in the prose-free expanded schema too.
    assert d.to_json()["multi_token_prediction"]["num_modules"] == 1

    html = d.to_html(standalone=True)
    assert "MTP head" in html
    assert 'data-card-id="mtp"' in html
    assert "eh_proj" in html  # detail-view internals rendered

    # The transformer block opens into its own tower (like the vision encoder).
    assert "separate tower" in html
    assert "Multi-Head Latent" in html

    # The block REUSES the real decoder-layer blocks as its children (no
    # synthesized internals, no per-block plumbing), so they route through the
    # central router into the same MLA / MoE drill-downs as the main stack.
    mtp = next(b for b in ir["extras"]["render"]["model_blocks"] if b["id"] == "mtp")
    tblock = next(c for c in mtp["children"] if c["id"] == "mtp_block")
    child_kinds = {c.get("kind") for c in tblock["children"]}
    assert {"attention", "ffn", "norm"} <= child_kinds
    attn = next(c for c in tblock["children"] if c.get("kind") == "attention")
    assert attn["view"] == "attention"
    assert {"mla_query_path", "mla_kv_path"} <= {c["id"] for c in attn.get("children", [])}
    # the reused attention's drill-downs are present as cards
    assert 'data-card-id="mla_query_path"' in html

    # The module count is surfaced when > 1.
    assert "MTP head x2" in unfold({**DEEPSEEK_V3_CONFIG, "num_nextn_predict_layers": 2}).to_html()


def test_models_without_mtp_have_no_mtp_block():
    ir = unfold(LLAMA3_8B_CONFIG).to_ir()
    assert "mtp" not in ir["extras"]
    assert all(b["id"] != "mtp" for b in ir["extras"]["render"]["model_blocks"])
    assert "MTP head" not in unfold(LLAMA3_8B_CONFIG).to_html()


def test_gemma4_31b():
    d = unfold(GEMMA4_31B_CONFIG)
    ir = d.to_ir()
    assert ir["name"] == "gemma-4-31B"
    assert len(ir["layers"]) == 60

    # Sliding/full pattern: every 6th layer is full, rest are sliding.
    for i, layer in enumerate(ir["layers"]):
        attn = layer["attention"]
        if (i % 6) == 5:
            assert attn["mask"] == "global", f"layer {i} should be full attention"
            assert attn["num_kv_heads"] == 4
            assert attn["head_dim"] == 512
        else:
            assert attn["mask"] == "sliding", f"layer {i} should be sliding"
            assert attn["window_size"] == 1024
            assert attn["num_kv_heads"] == 16
            assert attn["head_dim"] == 256
        assert attn["num_heads"] == 32
        assert layer["ffn"]["kind"] == "dense"

    # Two distinct layer signatures → layer-map shows two colored groups.
    sigs = {(l["attention"]["mask"], l["attention"]["window_size"]) for l in ir["layers"]}
    assert sigs == {("sliding", 1024), ("global", None)}

    assert ir["params"]["is_sparse"] is False

    html = d.to_html(standalone=True)
    assert "<!doctype html>" in html.lower()
    assert "<svg" in html.lower()
    print(f"Gemma 4 31B OK  — ~{ir['params']['total_h']} params")


def test_sliding_window_toggle_and_split():
    """AP-1: respect use_sliding_window and the Qwen max_window_layers split."""
    base = dict(
        model_type="qwen2", num_hidden_layers=6, hidden_size=64,
        num_attention_heads=8, intermediate_size=128, vocab_size=100,
        rms_norm_eps=1e-5,
    )

    # Window declared but explicitly disabled -> every layer is full attention.
    ir = unfold({**base, "sliding_window": 4096, "use_sliding_window": False}).to_ir()
    assert all(l["attention"]["mask"] == "causal" for l in ir["layers"])
    assert "sliding_window" not in ir.get("extras", {})

    # Mistral-style: window set, no toggle flag -> all layers slide (preserved).
    ir = unfold({**base, "sliding_window": 4096}).to_ir()
    assert all(l["attention"]["mask"] == "sliding" for l in ir["layers"])
    assert all(l["attention"]["window_size"] == 4096 for l in ir["layers"])

    # Enabled with a split: bottom max_window_layers full, the rest slide.
    ir = unfold({**base, "sliding_window": 4096,
                 "use_sliding_window": True, "max_window_layers": 2}).to_ir()
    masks = [l["attention"]["mask"] for l in ir["layers"]]
    assert masks == ["global", "global", "sliding", "sliding", "sliding", "sliding"]


def test_qwen3_moe_dense_sparse_pattern():
    """AP-2: decoder_sparse_step + mlp_only_layers decide dense-vs-MoE layers."""
    base = dict(
        model_type="qwen3_moe", num_hidden_layers=6, hidden_size=64,
        num_attention_heads=8, intermediate_size=128, moe_intermediate_size=64,
        vocab_size=100, rms_norm_eps=1e-5, num_experts=8, num_experts_per_tok=2,
    )

    def kinds(**over):
        ir = unfold({**base, **over}).to_ir()
        return [l["ffn"]["kind"] for l in ir["layers"]]

    # step=1 -> every layer MoE (Qwen3-30B-A3B shape).
    assert kinds(decoder_sparse_step=1) == ["moe"] * 6
    # step=2 -> MoE only where (i + 1) % 2 == 0.
    assert kinds(decoder_sparse_step=2) == ["dense", "moe"] * 3
    # mlp_only_layers force those indices dense even on the sparse step.
    assert kinds(decoder_sparse_step=1, mlp_only_layers=[0, 2]) == \
        ["dense", "moe", "dense", "moe", "moe", "moe"]


def test_omni_nested_thinker_text_config_unwrapped():
    """AP-3: an LM nested under thinker_config.text_config is unwrapped, not dropped."""
    inner = dict(
        model_type="qwen3_moe", num_hidden_layers=4, hidden_size=64,
        num_attention_heads=8, intermediate_size=128, vocab_size=100,
        rms_norm_eps=1e-5,
    )
    cfg = {
        "model_type": "qwen3_omni_moe",
        "thinker_config": {"model_type": "thinker", "text_config": inner,
                           "vision_config": {}, "audio_config": {}},
    }
    ir = unfold(cfg).to_ir()
    assert len(ir["layers"]) == 4
    assert ir["layers"][0]["attention"]["num_heads"] == 8
    assert not ir.get("warnings")


def test_attention_bias_and_rope_theta():
    """AP-4/AP-5: read attention_bias onto the spec, surface rope_theta always."""
    base = dict(
        model_type="qwen2", num_hidden_layers=2, hidden_size=64,
        num_attention_heads=8, intermediate_size=128, vocab_size=100,
        rms_norm_eps=1e-5, rope_theta=1000000,
    )

    # attention_bias=True -> per-layer spec flag + "+bias" in the label.
    d = unfold({**base, "attention_bias": True})
    assert all(l["attention"]["bias"] for l in d.to_ir()["layers"])
    assert "+bias" in d.to_html()

    # The bare rope_theta (no scaling dict) is surfaced on the IR extras.
    assert parse({**base, "attention_bias": True}).extras["rope"]["rope_theta"] == 1000000

    # attention_bias absent/False -> not flagged.
    assert not any(l["attention"]["bias"] for l in unfold({**base, "attention_bias": False}).to_ir()["layers"])
    assert not any(l["attention"]["bias"] for l in unfold(base).to_ir()["layers"])


def test_compress_rates_alias_derives_csa_hca_masks():
    """DeepSeek-V4 'compress_rates' is an alias of compress_ratios (0/4/128 -> SWA/CSA/HCA)."""
    cfg = dict(
        model_type="deepseek_v4", num_hidden_layers=4, hidden_size=128,
        num_attention_heads=16, num_key_value_heads=16, intermediate_size=512,
        vocab_size=1000, rms_norm_eps=1e-6, kv_lora_rank=64, q_lora_rank=96,
        compress_rates=[0, 4, 128, 4],
    )
    masks = [l["attention"]["mask"] for l in unfold(cfg).to_ir()["layers"]]
    assert masks == ["sliding", "compressed_sparse", "heavily_compressed", "compressed_sparse"]


def test_moe_routing_detail():
    """AP-7: surface gating / grouped routing / top-k renorm / scale on the router."""
    cfg = dict(
        model_type="deepseek_v3", num_hidden_layers=3, hidden_size=128,
        num_attention_heads=16, num_key_value_heads=16, intermediate_size=512,
        moe_intermediate_size=128, vocab_size=1000, rms_norm_eps=1e-6,
        kv_lora_rank=64, q_lora_rank=96, n_routed_experts=64, num_experts_per_tok=8,
        first_k_dense_replace=1,
        scoring_func="sigmoid", topk_method="noaux_tc", n_group=8, topk_group=4,
        norm_topk_prob=True, routed_scaling_factor=2.5,
    )
    d = unfold(cfg)
    ffn = next(l["ffn"] for l in d.to_ir()["layers"] if l["ffn"]["kind"] == "moe")
    assert ffn["routing"] == {
        "scoring_func": "sigmoid", "topk_method": "noaux_tc",
        "n_group": 8, "topk_group": 4, "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
    }
    from model_unfolder.labels import moe_router_lines
    lines = moe_router_lines(ffn)
    assert lines[0] == "Router"
    assert "sigmoid gating · top-8 of 64" in lines[1]
    assert any("keep 4/8 groups" in ln for ln in lines)

    # n_group == 1 is "no grouping" -> no group line.
    ir = unfold({**cfg, "n_group": 1, "topk_group": 1}).to_ir()
    ffn = next(l["ffn"] for l in ir["layers"] if l["ffn"]["kind"] == "moe")
    assert not any("groups" in ln for ln in moe_router_lines(ffn))


def test_single_kv_gemma4_stays_gqa_view():
    cfg = dict(GEMMA4_31B_CONFIG)
    text_cfg = dict(GEMMA4_31B_CONFIG["text_config"])
    text_cfg.update(
        {
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 256,
            "layer_types": ["sliding_attention"],
        }
    )
    cfg["text_config"] = text_cfg

    d = unfold(cfg)
    ir = d.to_ir()
    assert ir["layers"][0]["attention"]["kind"] == "gqa"
    assert ir["layers"][0]["attention"]["num_kv_heads"] == 1

    html = d.to_html(standalone=True)
    assert "Grouped-query attention" in html
    assert "Grouped SDPA" in html
    assert "GQA 8/1" in html
    assert "Q0-Q7" in html
    assert "use KV0" in html
    assert "Multi-query attention" not in html
    assert "Multi-Query SDPA" not in html


def test_gemma4_ple_uses_reusable_part_contract():
    d = unfold(_gemma4_e4b_config())
    ir = d.to_ir()
    blocks = ir["layers"][0]["blocks"]
    ple = next(block for block in blocks if block["id"] == "ple")

    assert ple["view"] == "per_layer_embedding"
    assert ple["detail"]["view"] == "per_layer_embedding"
    assert ple["detail"]["nodes"]["multiply"] == "ple_mul"
    assert ir["extras"]["per_layer_embeddings"]["hidden"] == 1024
    assert ir["extras"]["external_pathways"][0]["tap_block"] == "ple_mul"

    html = d.to_html(standalone=True)
    assert "uf-card-ple" in html
    assert 'data-card-id="ple_gate"' in html
    assert 'data-card-id="per_layer_input"' in html
    assert "per_layer_input[L]" in html
    assert "gelu_pytorch_tanh" not in html.lower()


def test_gemma4_multimodal_fusion_render():
    d = unfold(_gemma4_e2b_vision_config())
    ir = d.to_ir()
    assert ir["extras"]["modalities"]["inputs"]["vision"]["encoder"]["kind"] == "gemma4_vision"
    assert ir["extras"]["modalities"]["inputs"]["audio"]["encoder"]["kind"] == "gemma4_audio"
    assert ir["extras"]["modalities"]["inputs"]["audio"]["tokens"]["ms_per_token"] == 40
    assert ir["extras"]["modalities"]["fusion"]["kind"] == "placeholder_replace"
    assert ir["extras"]["modalities"]["fusion"]["mechanism"]["kind"] == "scatter_many"

    html = d.to_html(standalone=True)
    assert "Vision input" not in html
    assert "Soft visual tokens" in html
    assert "Soft audio tokens" in html
    assert "Vision -&gt; tokens" in html
    assert "Audio -&gt; tokens" in html
    assert "Multimodal fusion" in html
    assert "Visual tokens" in html
    assert "Audio tokens" in html
    assert "scatter modality features into token slots" in html
    assert "Text embeddings with modality slots" in html
    assert "Decoder stack input" in html
    assert "BOI" in html
    assert "BOA" in html
    assert "EOI" in html
    assert "EOA" in html
    assert "&lt;image&gt; x 280" in html
    assert "&lt;audio&gt; x 750" in html
    assert "AUD x 750" in html
    assert "280 x 1,536" in html
    assert 'data-card-id="vision_path"' in html
    assert 'data-card-id="audio_path"' in html
    assert 'data-card-id="vision_pixels"' in html
    assert 'data-card-id="vision_patches"' in html
    assert 'data-card-id="vision_encoder"' in html
    assert 'data-card-id="vision_projector"' in html
    assert 'data-card-id="visual_tokens"' in html
    assert 'data-card-id="audio_features"' in html
    assert 'data-card-id="audio_encoder"' in html
    assert 'data-card-id="audio_projector"' in html
    assert 'data-card-id="audio_tokens"' in html
    assert 'data-card-id="fusion"' in html
    assert 'data-card-id="stack_input"' in html
    assert 'data-id="fusion_image_slots"' in html
    assert 'data-id="fusion_vision_tokens"' in html
    assert 'data-id="fusion_audio_slots"' in html
    assert 'data-id="fusion_audio_tokens"' in html
    assert 'data-card-id="fusion_image_slots"' in html
    assert 'data-card-id="fusion_audio_tokens"' in html
    assert 'data-card-id="fusion_mixed_stream"' in html


def test_gemma4_video_token_does_not_create_grid_video_path():
    cfg = _gemma4_e2b_vision_config()
    cfg.update({"video_token_id": 258884, "video_seq_length": 64})

    d = unfold(cfg)
    modalities = d.to_ir()["extras"]["modalities"]["inputs"]
    assert "vision" in modalities
    assert "audio" in modalities
    assert "video" not in modalities

    html = d.to_html(standalone=True)
    assert "Video -&gt; grid" not in html
    assert 'data-card-id="video_path"' not in html


def test_qwen2_audio_sparse_text_config_is_completed():
    d = unfold(QWEN2_AUDIO_SPARSE_CONFIG)
    ir = d.to_ir()

    assert ir["warnings"] == []
    assert ir["name"] == "Qwen2-Audio-7B"
    assert ir["hidden_size"] == 4096
    assert ir["vocab_size"] == 156032
    assert len(ir["layers"]) == 32
    assert ir["layers"][0]["attention"]["kind"] == "mha"
    assert ir["layers"][0]["attention"]["num_heads"] == 32
    assert ir["layers"][0]["ffn"]["intermediate_size"] == 11008

    audio = ir["extras"]["modalities"]["inputs"]["audio"]
    assert audio["input"]["feature_size"] == 128
    assert audio["encoder"]["hidden_size"] == 1280
    assert audio["encoder"]["num_layers"] == 32
    assert audio["encoder"]["num_attention_heads"] == 20
    assert ir["extras"]["modalities"]["fusion"]["placeholders"]["audio"]["token_id"] == 151646

    html = d.to_html(standalone=True)
    assert "partial config" not in html
    assert "Audio -&gt; tokens" in html


def test_qwen2_audio_code_evidence_does_not_mark_config_partial():
    d = unfold(QWEN2_AUDIO_SPARSE_CONFIG, inspect_code=True)
    ir = d.to_ir()

    assert ir["warnings"] == []
    assert ir["extras"]["code_evidence"]["provenance"]["model_type"] == "qwen2_audio"
    assert ir["extras"]["code_evidence"]["provenance"]["files"]

    html = d.to_html(standalone=True)
    assert "partial config" not in html
    assert "CODE EVIDENCE" in html


def test_llama3():
    d = unfold(LLAMA3_8B_CONFIG)
    ir = d.to_ir()
    assert len(ir["layers"]) == 32
    assert ir["layers"][0]["attention"]["kind"] == "gqa"
    assert ir["layers"][0]["attention"]["num_heads"] == 32
    assert ir["layers"][0]["attention"]["num_kv_heads"] == 8
    assert ir["layers"][0]["ffn"]["kind"] == "dense"
    assert ir["params"]["is_sparse"] is False

    html = d.to_html(standalone=True)
    assert "Grouped SDPA" in html
    assert "KV sharing pattern" in html
    assert "Q0-Q3" in html
    assert "use KV0" in html
    assert "KV cache 4x smaller" in html
    assert "Grouped scaled dot-product attention" in html

    print(f"Llama-3 OK  — ~{ir['params']['total_h']} params")


def test_non_gated_dense_ffn_has_plain_mlp_view():
    d = unfold(GPT_NEOX_CONFIG)
    ir = d.to_ir()
    ffn_block = next(block for block in ir["layers"][0]["blocks"] if block["id"] == "ffn")
    child_ids = {child["id"] for child in ffn_block["children"]}

    assert ir["layers"][0]["ffn"]["gated"] is False
    assert ffn_block["view"] == "dense_ffn"
    assert {"up_proj", "silu", "down_proj"} <= child_ids
    assert "gate_proj" not in child_ids
    assert "mul" not in child_ids

    html = d.to_html(standalone=True)
    assert "Linear (in)" in html
    assert "Linear (gate)" not in html
    assert 'data-card-id="gate_proj"' not in html
    assert 'data-card-id="mul"' not in html


def test_falcon_parallel_attn_uses_parallel_topology():
    d = unfold(FALCON_PARALLEL_CONFIG)
    ir = d.to_ir()
    blocks = ir["layers"][0]["blocks"]
    block_by_id = {block["id"]: block for block in blocks}

    assert ir["layers"][0]["attention"]["kind"] == "mqa"
    assert ir["extras"]["parallel_residual"] is True
    assert block_by_id["rms1"]["label"] == "LayerNorm"
    assert block_by_id["add1"]["title"] == "Residual add (parallel)"
    assert block_by_id["ffn"]["lane"] == "left"
    assert block_by_id["ffn"]["tap_from"] == "attn"
    assert block_by_id["ffn"]["feeds"] == "add1"
    assert block_by_id["ffn"]["side_align"] == "tap"
    assert "add2" not in block_by_id

    html = d.to_html(standalone=True)
    assert "Multi-Query SDPA" in html
    assert "Shared K/V cache" in html
    assert "1 K + 1 V reused by 71 Q" in html
    assert "KV cache 71x smaller" in html
    assert "Multi-query scaled dot-product attention" in html


def test_new_should_support_family_routes():
    cases = [
        # Gemma 7B has num_kv=1 but no `multi_query` flag — extreme GQA,
        # matching Google's "GQA throughout" naming.
        (GEMMA1_CONFIG, "gqa", "dense", "rmsnorm"),
        (PHI2_CONFIG, "mha", "dense", "layernorm"),
        (YI_34B_CONFIG, "gqa", "dense", "rmsnorm"),
        (OLMO_7B_CONFIG, "mha", "dense", "layernorm"),
        (OLMOE_CONFIG, "mha", "moe", "rmsnorm"),
    ]

    for cfg, attn_kind, ffn_kind, norm_kind in cases:
        d = unfold(cfg)
        ir = d.to_ir()
        layer = ir["layers"][0]

        assert not ir["warnings"]
        assert layer["attention"]["kind"] == attn_kind
        assert layer["ffn"]["kind"] == ffn_kind
        assert layer["norm_kind"] == norm_kind

    phi = unfold(PHI2_CONFIG).to_ir()
    assert phi["layers"][0]["ffn"]["gated"] is False
    assert phi["extras"]["partial_rotary_factor"] == 0.4

    olmoe = unfold(OLMOE_CONFIG).to_ir()
    assert olmoe["layers"][0]["attention"]["qk_norm"] is True
    assert olmoe["layers"][0]["ffn"]["num_experts"] == 64
    assert olmoe["layers"][0]["ffn"]["num_experts_per_tok"] == 8


def test_dbrx_nested_config_routes_to_gqa_moe():
    d = unfold(DBRX_CONFIG)
    ir = d.to_ir()
    layer = ir["layers"][0]

    assert not ir["warnings"]
    assert ir["name"] == "dbrx-base"
    assert len(ir["layers"]) == 40
    assert layer["attention"]["kind"] == "gqa"
    assert layer["attention"]["num_heads"] == 48
    assert layer["attention"]["num_kv_heads"] == 8
    assert layer["attention"]["head_dim"] == 128
    assert layer["ffn"]["kind"] == "moe"
    assert layer["ffn"]["num_experts"] == 16
    assert layer["ffn"]["num_experts_per_tok"] == 4
    assert layer["ffn"]["expert_intermediate_size"] == 3584
    assert ir["extras"]["attention"]["clip_qkv"] == 8

    html = d.to_html(standalone=True)
    assert "GQA 48/8" in html
    assert "MoE" in html
    assert "16 experts" in html


def test_attention_detail_views_dispatch_by_kind():
    # Scope: transformer-LLM attention kinds (MHA / GQA / MQA / MLA).
    # Non-attention mixers (SSM, RWKV, linear, recurrent) are out of scope.
    d = unfold(KIMI_K2_CONFIG)
    ir = d.to_ir()
    attn_block = next(block for block in ir["layers"][0]["blocks"] if block["id"] == "attn")
    child_ids = {child["id"] for child in attn_block["children"]}
    html = d.to_html(standalone=True)

    assert ir["layers"][0]["attention"]["kind"] == "mla"
    assert "mla_kv_path" in child_ids
    assert "Multi-Head Latent" in html
    assert "mla_kv_down" in html
    assert "Scaled Dot-Product Attention" not in html


def test_model_id_uses_hf_token_env_and_explicit_override():
    calls = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, dict(kwargs)))
            return LLAMA3_8B_CONFIG

    old_token = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = "hf_env_token"
    try:
        _with_fake_transformers(
            FakeAutoConfig,
            lambda: unfold("meta-llama/Meta-Llama-3-8B"),
        )
        _with_fake_transformers(
            FakeAutoConfig,
            lambda: unfold("meta-llama/Meta-Llama-3-8B", token="hf_call_token"),
        )
    finally:
        _restore_env("HF_TOKEN", old_token)

    assert calls[0][0] == "meta-llama/Meta-Llama-3-8B"
    assert calls[0][1]["trust_remote_code"] is False  # always explicit, never prompts
    assert calls[0][1]["token"] == "hf_env_token"
    assert calls[1][1]["token"] == "hf_call_token"


def test_model_id_falls_back_to_legacy_hf_auth_kwarg():
    calls = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(dict(kwargs))
            if "token" in kwargs:
                raise TypeError("got an unexpected keyword argument 'token'")
            return LLAMA3_8B_CONFIG

    _with_fake_transformers(
        FakeAutoConfig,
        lambda: unfold("meta-llama/Meta-Llama-3-8B", token="hf_call_token"),
    )

    assert calls[0]["token"] == "hf_call_token"
    assert calls[1]["use_auth_token"] == "hf_call_token"
    assert calls[1]["trust_remote_code"] is False


def test_model_id_custom_code_falls_back_to_raw_config_json():
    """Custom-code repos must NEVER prompt or run remote code — fall back to config.json."""
    calls = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(dict(kwargs))
            raise ValueError(
                "Loading custom/model requires you to execute the configuration "
                "file; set trust_remote_code=True"
            )

    import json
    import tempfile

    cfg = {
        "model_type": "custom_moe", "num_hidden_layers": 2, "hidden_size": 64,
        "num_attention_heads": 8, "intermediate_size": 128, "vocab_size": 100,
        "rms_norm_eps": 1e-5,
    }
    tmpdir = tempfile.mkdtemp()
    cfg_path = os.path.join(tmpdir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **kw: cfg_path
    sys.modules["huggingface_hub"] = fake_hub
    try:
        d = _with_fake_transformers(FakeAutoConfig, lambda: unfold("custom/model"))
    finally:
        sys.modules.pop("huggingface_hub", None)

    # Exactly one attempt, explicitly False — no prompt, no retry-with-True.
    assert len(calls) == 1
    assert calls[0]["trust_remote_code"] is False
    assert len(d.to_ir()["layers"]) == 2


def test_typed_errors_for_access_notfound_and_parse():
    """Load/parse failures surface as the right typed UnfoldError subclass."""
    import pytest
    from model_unfolder import (
        ConfigParseError,
        ModelAccessError,
        ModelNotFoundError,
    )

    class Gated:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            raise OSError("401 Client Error. Access to model X is restricted and gated.")

    class Missing:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            raise OSError("404 Client Error. Repository Not Found for url ...")

    with pytest.raises(ModelAccessError):
        _with_fake_transformers(Gated, lambda: unfold("meta-llama/Llama-2-7b-hf"))
    with pytest.raises(ModelNotFoundError):
        _with_fake_transformers(Missing, lambda: unfold("nope/does-not-exist"))

    # A loaded-but-broken config (no layers) is a hard ConfigParseError.
    with pytest.raises(ConfigParseError):
        unfold({"model_type": "mystery", "some_blob": 123})


def _with_fake_transformers(auto_config, fn):
    module = types.ModuleType("transformers")
    module.AutoConfig = auto_config
    had_previous = "transformers" in sys.modules
    previous = sys.modules.get("transformers")
    sys.modules["transformers"] = module
    try:
        return fn()
    finally:
        if had_previous:
            sys.modules["transformers"] = previous
        else:
            del sys.modules["transformers"]


def _restore_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    test_kimi_k2()
    test_deepseek_v3_phase_change()
    test_mtp_head_detected_and_rendered()
    test_models_without_mtp_have_no_mtp_block()
    test_gemma4_31b()
    test_gemma4_ple_uses_reusable_part_contract()
    test_llama3()
    test_new_should_support_family_routes()
    test_dbrx_nested_config_routes_to_gqa_moe()
    test_model_id_uses_hf_token_env_and_explicit_override()
    test_model_id_falls_back_to_legacy_hf_auth_kwarg()
    test_model_id_custom_code_falls_back_to_raw_config_json()
    print("\nAll smoke tests passed.")
