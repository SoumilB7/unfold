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
DIFFUSION_GEMMA_LAYER_TYPES = [
    "sliding_attention" if (i % 6) != 5 else "full_attention" for i in range(30)
]
DIFFUSION_GEMMA_CONFIG = {
    "architectures": ["DiffusionGemmaForBlockDiffusion"],
    "model_type": "diffusion_gemma",
    "_name_or_path": "google/diffusiongemma-26B-A4B-it",
    "canvas_length": 256,
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "gemma4_text",
        "vocab_size": 262144,
        "hidden_size": 2816,
        "intermediate_size": 2112,
        "num_hidden_layers": 30,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "layer_types": DIFFUSION_GEMMA_LAYER_TYPES,
        "final_logit_softcapping": 30.0,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 704,
    },
}

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


def _arch_variant_badges(html):
    """The ``x N`` repeat badge inside each architecture-view variant, keyed by
    variant index — what a viewer sees on the layer cell after toggling pills."""
    import re
    chunks = re.split(r'class="uf-arch-variant uf-arch-variant-(\d+)"', html)
    out = {}
    for i in range(1, len(chunks), 2):
        m = re.search(r">x (\d+)<", chunks[i + 1])
        if m:
            out[int(chunks[i])] = int(m.group(1))
    return out


def test_layer_repeat_badge_is_per_variant_not_global_total():
    """A heterogeneous model renders one architecture variant per layer type; the
    ``x N`` badge on each must count THAT group's layers (matching its own toggle
    pill), never the global total. Regression: the badge hardcoded
    len(ir["layers"]) so every DeepSeek-V3 variant wrongly read "x 61"."""
    # 1 dense + 3 MoE layers ⇒ neither group equals the total (4), so a global
    # leak (both "x 4") is unmistakably distinguishable from the correct 1 / 3.
    cfg = {**DEEPSEEK_V3_CONFIG, "num_hidden_layers": 4, "first_k_dense_replace": 1}
    badges = _arch_variant_badges(unfold(cfg).to_html(standalone=True))
    assert sorted(badges.values()) == [1, 3], badges
    assert sum(badges.values()) == cfg["num_hidden_layers"]

    # A homogeneous stack still shows the total on its single variant.
    homo = _arch_variant_badges(unfold(LLAMA3_8B_CONFIG).to_html(standalone=True))
    assert set(homo.values()) == {len(unfold(LLAMA3_8B_CONFIG).to_ir()["layers"])}, homo


def test_reused_router_drill_resolves_its_own_routing_not_the_ambient_variant():
    """An MTP block reuses the grouped-MoE decoder layer; its Top-k drill must show
    the SAME grouped torch sequence everywhere — never collapse to a plain (non-grouped)
    torch.topk just because a dense-layer variant tab is the ambient dominant. Regression:
    the drill read info['dominant'] instead of its own block-local detail.ffn."""
    from model_unfolder.preview import svg_views
    # Needs all three: dense+MoE variants (first_k_dense_replace), GROUPED routing
    # (so the drill is multi-step), and an MTP block that reuses the layer.
    cfg = {**DEEPSEEK_V3_CONFIG, "num_hidden_layers": 4, "first_k_dense_replace": 1,
           "num_nextn_predict_layers": 1, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
           "n_group": 8, "topk_group": 4, "norm_topk_prob": True, "routed_scaling_factor": 2.5}
    html = unfold(cfg).to_html(standalone=True)
    drills = [("Group scores" in svg) for _lbl, svg in svg_views(html)
              if "Top-k experts" in svg and "Gather weights" in svg]
    assert drills, "no top-k selection drill was baked"
    assert all(drills), "a reused-router drill rendered non-grouped — variant leaked into block resolution"


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
    assert "decoder layer" in html
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


def test_diffusion_gemma_block_diffusion():
    """Gate A pin: DiffusionGemma gets the block-diffusion loop view, not decoder-only."""
    d = unfold(DIFFUSION_GEMMA_CONFIG)
    ir = d.to_ir()

    # Basic IR shape — 30 shared encoder/decoder layers
    assert ir["name"] == "diffusiongemma-26B-A4B-it"
    assert len(ir["layers"]) == 30

    # Render extras must declare block_diffusion layout
    render = ir["extras"]["render"]
    assert render["layout"] == "block_diffusion", "must route to block-diffusion view"
    assert ir["extras"]["block_diffusion"]["canvas_length"] == 256

    # All loop blocks must be present with the right ids.  The committed output
    # is the arrow leaving the loop (folded into the sampler card) — not a
    # standalone "just there" output block.
    expected_ids = {
        "bd_prompt", "bd_encoder", "bd_kv_cache", "bd_canvas",
        "bd_self_cond", "bd_decoder", "bd_lm_head", "bd_sampler",
    }
    loop_ids = {b["id"] for b in render["loop_blocks"]}
    assert loop_ids == expected_ids, f"loop block mismatch: {loop_ids ^ expected_ids}"
    assert "bd_output" not in loop_ids, "output must not be a standalone block"

    # Every loop block must have a non-empty description (no bare undescribed blocks)
    for block in render["loop_blocks"]:
        assert block.get("description"), f"block {block['id']!r} has no description"

    html = d.to_html()
    # Loop view SVG must be present
    assert "<svg" in html
    # Key block ids must appear as data-id attributes (clickable in the diagram)
    for bid in expected_ids:
        assert f'data-id="{bid}"' in html, f"block {bid!r} not clickable in SVG"

    # The section label announces the block diffusion loop
    assert "BLOCK DIFFUSION LOOP" in html

    # Softcap value from config is surfaced in the lm_head description
    lm_block = next(b for b in render["loop_blocks"] if b["id"] == "bd_lm_head")
    assert "30.0" in lm_block["description"], "softcap value not surfaced in lm_head card"

    print(f"DiffusionGemma OK  — canvas={ir['extras']['block_diffusion']['canvas_length']}")


def test_diffusion_gemma_block_worthiness():
    """Gate C pin: the per-layer view obeys the three-tier block paradigm AND
    divides the parallel FFN inline (no collapsed '∥' block).

    Tier-1 blocks (attention, the two FFN branches, norms) are clickable; Tier-2
    connectors (residual ⊕ and the FFN merge ⊕) are static glyphs with no card;
    the Tier-3 learned layer scalar is an annotation, never a block.
    """
    d = unfold(DIFFUSION_GEMMA_CONFIG)
    ir = d.to_ir()
    blocks = {b["id"]: b for b in ir["layers"][0]["blocks"]}

    # Tier-3: layer_scalar is NOT a block (Gate C) — and the frame caption was
    # dropped as not worth the space, so it is not surfaced as an annotation either.
    assert "layer_scalar" not in blocks, "learned scalar must not be a block (Tier-3)"
    assert not ir["extras"]["render"].get("layer_annotations"), "scalar caption was removed"

    # Tier-2: residual adds AND the FFN merge are connector GLYPHS (⊕, not boxes) —
    # clickable for a describing card (not static, but still kind residual_add).
    for add_id in ("add1", "add2", "ffn_merge"):
        assert blocks[add_id]["kind"] == "residual_add"
        assert not blocks[add_id].get("static"), f"{add_id} connector is now clickable, not static"
        assert blocks[add_id].get("description"), f"{add_id} must describe itself on click"

    # The parallel FFN is divided inline: two branch blocks, no collapsed block.
    assert "ffn" not in blocks, "the collapsed 'MLP ∥ MoE' block must be gone"
    assert blocks["ffn_mlp"]["branch_side"] == "left" and blocks["ffn_mlp"]["feeds"] == "ffn_merge"
    assert blocks["ffn_moe"]["branch_side"] == "right" and blocks["ffn_moe"]["feeds"] == "ffn_merge"
    assert blocks["ffn_mlp"]["view"] and blocks["ffn_moe"]["view"] == "moe"  # each branch drills down
    assert blocks["attn"]["kind"] == "attention"

    html = d.to_html()
    # The connector glyphs are clickable with a describing card (still glyphs, not boxes).
    for add_id in ("add1", "add2", "ffn_merge"):
        assert f'data-id="{add_id}"' in html, f"{add_id} connector must be clickable"
        assert f'data-card-id="{add_id}"' in html, f"{add_id} must have a describing card"
    # Both branches ARE clickable blocks inline in the architecture.
    for bid in ("ffn_mlp", "ffn_moe"):
        assert f'data-id="{bid}"' in html and f'data-card-id="{bid}"' in html
    # layer_scalar is not surfaced anywhere — not a block, not a caption.
    assert 'data-id="layer_scalar"' not in html
    assert "learned per-layer scalar" not in html
    # Coupling stays clean with connectors demoted.
    from model_unfolder.block_schema import validate_click_coupling
    assert validate_click_coupling(html) == []


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
    from model_unfolder.labels import router_facts
    facts = router_facts(ffn)
    assert "64 experts" in facts and "top-8" in facts and "sigmoid" in facts
    assert "keep 4/8 groups" in facts

    # n_group == 1 is "no grouping" -> no group chip.
    ir = unfold({**cfg, "n_group": 1, "topk_group": 1}).to_ir()
    ffn = next(l["ffn"] for l in ir["layers"] if l["ffn"]["kind"] == "moe")
    assert not any("groups" in f for f in router_facts(ffn))


def test_moe_gate_view_is_config_driven_and_shared_expert_drawn():
    """The MoE diagram must SHOW the modern router (not just card facts), draw the
    always-on shared expert, and adapt the gate pipeline to the config — closing
    the 'diagram shows plain top-k' gap across DeepSeek/Kimi/GLM/Qwen3-MoE."""
    from model_unfolder.renderers.html.block_views.registry import (
        render_block_detail, render_sub_block_detail,
    )
    from model_unfolder.renderers.html.metadata import _make_info
    from model_unfolder.block_schema import validate_click_coupling

    def moe_and_router(cfg):
        ir = parse(cfg).to_dict()
        moe = next(b for L in ir["layers"] for b in L["blocks"]
                   if b.get("id") == "ffn" and b.get("view") == "moe")
        info = _make_info(ir)
        return ir, info, moe, next(c for c in moe["children"] if c["id"] == "router")

    base = dict(
        model_type="deepseek_v3", num_hidden_layers=3, hidden_size=128,
        num_attention_heads=16, num_key_value_heads=16, intermediate_size=512,
        moe_intermediate_size=128, vocab_size=1000, rms_norm_eps=1e-6,
        kv_lora_rank=64, q_lora_rank=96, n_routed_experts=64, num_experts_per_tok=8,
        first_k_dense_replace=1, scoring_func="sigmoid", topk_method="noaux_tc",
        n_group=8, topk_group=4, norm_topk_prob=True, routed_scaling_factor=2.5,
        n_shared_experts=1,
    )
    ir, info, moe, router = moe_and_router(base)
    # Router drills into the gate policy view.
    assert router.get("view") == "moe_router"
    # The shared expert is a real Tier-1 block in the MoE view + has a card.
    assert any(c["id"] == "shared_expert" for c in moe["children"])
    moe_html = render_block_detail(ir, info, "m", moe)
    assert "Shared" in moe_html and 'data-id="shared_expert"' in moe_html
    # The weighted-sum ⊕ is a Tier-2 connector glyph — clickable for its describing card.
    add = next(c for c in moe["children"] if c["id"] == "add_moe")
    assert add["kind"] == "residual_add" and not add.get("static") and add.get("description")

    # De-blocked per Gate C: the router view shows only bare OP-NAME labels —
    # every count/flag is a chip on a card, never on a block. The gate's scoring
    # fn, the selection counts, the scale value all live in cards now.
    gate = render_sub_block_detail(ir, info, "m", router)
    for token in ("Gate", "Top-k", "renormalize", "learned bias"):
        assert token in gate, f"router view missing label {token!r}"
    # The descriptive text moved OFF the blocks (the user's "why not in description"):
    # no scoring fn, no expert/group counts, no scale value painted on the diagram.
    for leaked in ("sigmoid", "256 scores", "group-limited", "keep 4 of 8",
                   "select top-8", "routed scale"):
        assert leaked not in gate, f"label text {leaked!r} should be in a card, not the diagram"

    rcards = {c["id"]: c for c in router["children"]}
    # Gate card carries the Linear + scoring detail and count chips.
    assert "sigmoid" in rcards["g_gate"]["description"] and "Linear" in rcards["g_gate"]["description"]
    assert any("experts" in f for f in rcards["g_gate"]["facts"])
    # routed scale is a × glyph, but its CONSTANT OPERAND is shown beside the glyph
    # (answers "× what?") — a labelled-constant connector, not a bare ×.
    assert ">2.5</text>" in gate, "the × must show its constant operand (2.5) on the diagram"
    assert "2.5" in rcards["g_scale"]["title"]

    # "Top-k" is not hand-wavy logic: it names torch.topk and DRILLS into the real
    # PyTorch sequence DeepSeek runs (two torch.topk + mask + gather).
    topk = rcards["g_topk"]
    assert topk["view"] == "topk_selection" and "torch.topk" in topk["description"]
    drill_ids = {c["id"] for c in topk["children"]}
    assert {"ts_group", "ts_topk_groups", "ts_mask", "ts_topk_experts", "ts_gather"} == drill_ids
    sel = render_sub_block_detail(ir, info, "m", topk)
    for token in ("Group scores", "Top-k groups", "Mask groups", "Top-k experts", "Gather weights"):
        assert token in sel, f"top-k drill missing {token!r}"
    # the leaf cards name the actual torch ops
    cards_by_id = {c["id"]: c for c in topk["children"]}
    assert "torch.topk" in cards_by_id["ts_topk_experts"]["description"]
    assert "gather" in cards_by_id["ts_gather"]["description"]
    assert "masked_fill" in cards_by_id["ts_mask"]["description"]

    # Dynamic (Gate A.3): a plain softmax top-k router collapses — no group / bias /
    # scale, and its single torch.topk is an honest LEAF (no drill).
    plain = dict(model_type="m", num_hidden_layers=2, hidden_size=128,
                 num_attention_heads=8, num_key_value_heads=2, intermediate_size=256,
                 vocab_size=1000, rms_norm_eps=1e-5, num_local_experts=8,
                 num_experts_per_tok=2)
    pir, pinfo, _pmoe, prouter = moe_and_router(plain)
    pgate = render_sub_block_detail(pir, pinfo, "m", prouter)
    assert "Gate" in pgate and "Top-k" in pgate
    pcards = {c["id"]: c for c in prouter["children"]}
    assert pcards["g_topk"].get("view") is None and not pcards["g_topk"].get("children")
    assert "torch.topk" in pcards["g_topk"]["description"]
    assert "learned bias" not in pgate

    # The whole rendered model stays click-coupled with the new gate drill embedded.
    assert validate_click_coupling(unfold(base).to_html(standalone=True)) == []


def test_router_topk_drill_adapts_per_family_not_inherited():
    """Lock the router DESIGN across families while the STEPS adapt per config: the
    Top-k drill must add group steps only when grouped, a gather only when a bias
    splits selection-scores from weight-scores, and never inherit one family's
    specifics for another (DeepSeek-V2's max-per-group ≠ V3's top-2-sum). Every
    drilled child card must have a matching node in its view (no orphans)."""
    from model_unfolder.renderers.html.metadata import _make_info
    from model_unfolder.renderers.html.block_views.registry import render_sub_block_detail

    base = dict(num_hidden_layers=2, hidden_size=128, num_attention_heads=8,
                num_key_value_heads=8, intermediate_size=256, vocab_size=1000,
                rms_norm_eps=1e-5, moe_intermediate_size=128)

    def find(blocks, tid):
        for b in blocks:
            if isinstance(b, dict):
                if b.get("id") == tid:
                    return b
                hit = find(b.get("children") or [], tid)
                if hit:
                    return hit

    def topk_block(cfg):
        d = unfold(cfg); ir = d.to_ir(); info = _make_info(ir)
        gt = find(info["dominant"]["spec"]["blocks"], "g_topk")
        svg = render_sub_block_detail(ir, info, "m", gt) if gt.get("view") else ""
        return gt, svg

    # plain softmax (Mixtral): single torch.topk → an honest LEAF, no drill.
    gt, _ = topk_block(dict(base, model_type="mixtral", num_local_experts=8, num_experts_per_tok=2))
    assert gt.get("view") is None and not gt.get("children")

    # grouped + group_limited_greedy, NO bias (DeepSeek-V2): group steps, NO gather,
    # and the group score is the MAX (never V3's top-2-sum).
    gt, svg = topk_block(dict(base, model_type="deepseek_v2", n_routed_experts=64,
        num_experts_per_tok=6, n_group=8, topk_group=3, topk_method="group_limited_greedy",
        scoring_func="softmax", routed_scaling_factor=16.0))
    assert [c["id"] for c in gt["children"]] == ["ts_group", "ts_topk_groups", "ts_mask", "ts_topk_experts"]
    assert "Gather weights" not in svg
    gs = find(gt["children"], "ts_group")
    assert "top expert" in gs["description"] and "top-2" not in gs["description"]

    # grouped + noaux bias (DeepSeek-V3): full sequence WITH gather.
    gt, svg = topk_block(dict(base, model_type="deepseek_v3", n_routed_experts=256,
        num_experts_per_tok=8, n_group=8, topk_group=4, topk_method="noaux_tc",
        scoring_func="sigmoid", norm_topk_prob=True, routed_scaling_factor=2.5))
    assert [c["id"] for c in gt["children"]] == ["ts_group", "ts_topk_groups", "ts_mask", "ts_topk_experts", "ts_gather"]
    assert "top-2" in find(gt["children"], "ts_group")["description"]

    # noaux bias but NOT grouped (edge): gather WITHOUT group steps.
    gt, svg = topk_block(dict(base, model_type="deepseek_v3", n_routed_experts=64,
        num_experts_per_tok=8, topk_method="noaux_tc", scoring_func="sigmoid"))
    assert [c["id"] for c in gt["children"]] == ["ts_topk_experts", "ts_gather"]

    # No orphan cards: every drilled child card has a node (its title) in the view.
    for cfg in (dict(base, model_type="deepseek_v2", n_routed_experts=64, num_experts_per_tok=6,
                     n_group=8, topk_group=3, topk_method="group_limited_greedy", scoring_func="softmax"),
                dict(base, model_type="deepseek_v3", n_routed_experts=256, num_experts_per_tok=8,
                     n_group=8, topk_group=4, topk_method="noaux_tc", scoring_func="sigmoid")):
        gt, svg = topk_block(cfg)
        for child in gt["children"]:
            assert child["title"] in svg, f"orphan card {child['id']!r}: no node in its drill view"


def test_dsa_indexer_and_clamped_swiglu_are_surfaced():
    """Two Tier-3 properties that used to be silently dropped must now read from
    config: DeepSeek-V3.2's sparse-attention indexer and gpt-oss's clamped SwiGLU."""
    from model_unfolder.labels import attention_summary, ffn_summary

    # DSA: indexer geometry + top-k are consumed and surfaced (not 'unparsed').
    dsa = parse(dict(
        model_type="deepseek_v32", num_hidden_layers=2, hidden_size=128,
        num_attention_heads=16, num_key_value_heads=16, intermediate_size=512,
        moe_intermediate_size=128, vocab_size=1000, rms_norm_eps=1e-6,
        kv_lora_rank=64, q_lora_rank=96, qk_nope_head_dim=64, qk_rope_head_dim=32,
        n_routed_experts=16, num_experts_per_tok=4, first_k_dense_replace=1,
        index_topk=2048, index_n_heads=64, index_head_dim=128,
    )).to_dict()
    att = dsa["layers"][0]["attention"]
    assert (att["index_topk"], att["index_n_heads"], att["index_head_dim"]) == (2048, 64, 128)
    desc, facts = attention_summary(att)
    assert "DeepSeek Sparse Attention" in desc
    assert any("DSA top-2,048" in f for f in facts) and any("indexer 64×128" in f for f in facts)

    # Clamped SwiGLU: gpt-oss swiglu_limit becomes a Tier-3 chip on the FFN.
    oss = parse(dict(
        model_type="gpt_oss", num_hidden_layers=2, hidden_size=128,
        num_attention_heads=8, num_key_value_heads=8, intermediate_size=256,
        vocab_size=1000, rms_norm_eps=1e-5, num_local_experts=8,
        num_experts_per_tok=2, swiglu_limit=7.0,
    )).to_dict()
    ffn = next(l["ffn"] for l in oss["layers"] if l["ffn"]["kind"] == "moe")
    assert ffn["activation_clip"] == 7.0
    assert any("clamped ±7" in f for f in ffn_summary(ffn)[1])

    # M-RoPE: Qwen-VL's rope_scaling.mrope_section becomes a Tier-3 chip.
    vl = parse(dict(
        model_type="qwen2_vl", num_hidden_layers=2, hidden_size=128,
        num_attention_heads=8, num_key_value_heads=2, intermediate_size=256,
        vocab_size=1000, rms_norm_eps=1e-5,
        rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
    )).to_dict()
    att = vl["layers"][0]["attention"]
    assert att["mrope_section"] == [16, 24, 24]
    assert any("M-RoPE 16/24/24" in f for f in attention_summary(att)[1])


def test_dsa_indexer_is_a_clickable_subblock_only_when_declared():
    """The DSA lightning indexer is a Tier-1 drill-down on V3.2's attention — a
    third path into the scores — and is strictly gated on index_n_heads so no
    other MLA model (V3 / Kimi) grows a phantom indexer."""
    from model_unfolder.renderers.html.block_views.registry import (
        render_block_detail, render_sub_block_detail,
    )
    from model_unfolder.renderers.html.metadata import _make_info

    def attn_block(cfg):
        ir = parse(cfg).to_dict()
        b = next(b for L in ir["layers"] for b in L["blocks"] if b.get("id") == "attn")
        return ir, _make_info(ir), b

    mla = dict(
        num_hidden_layers=2, hidden_size=128, num_attention_heads=16,
        num_key_value_heads=16, intermediate_size=512, moe_intermediate_size=128,
        vocab_size=1000, rms_norm_eps=1e-6, kv_lora_rank=64, q_lora_rank=96,
        qk_nope_head_dim=64, qk_rope_head_dim=32, n_routed_experts=16,
        num_experts_per_tok=4, first_k_dense_replace=1,
    )
    # V3.2: indexer present, clickable, and its drill describes the scorer.
    ir, info, attn = attn_block(dict(mla, model_type="deepseek_v32",
                                     index_topk=2048, index_n_heads=64, index_head_dim=128))
    assert any(c["id"] == "mla_indexer" for c in attn["children"])
    assert 'data-id="mla_indexer"' in render_block_detail(ir, info, "a", attn)
    idx = next(c for c in attn["children"] if c["id"] == "mla_indexer")
    drill = render_sub_block_detail(ir, info, "a", idx)
    # Locked label standard (same as the router): bare op-name blocks, counts as
    # card chips — the selection names its real op (Top-k keys = torch.topk).
    assert "Linear (Indexer)" in drill and "Index scores" in drill and "Top-k keys" in drill
    assert "top-2,048" not in drill and "64 heads" not in drill   # counts are NOT on the blocks
    kids = {c["id"]: c for c in idx["children"]}
    assert "torch.topk" in kids["dsa_topk"]["description"]
    assert any("2,048" in f for f in kids["dsa_topk"]["facts"])   # k = 2,048 chip
    assert any("64" in f for f in kids["dsa_proj"]["facts"])      # 64 heads chip

    # V3 (no index fields): no phantom indexer node or child.
    ir2, info2, attn2 = attn_block(dict(mla, model_type="deepseek_v3"))
    assert not any(c["id"] == "mla_indexer" for c in attn2["children"])
    assert 'data-id="mla_indexer"' not in render_block_detail(ir2, info2, "a", attn2)


def test_layer_norm_placement_matches_source_topology():
    """Connection fidelity (Gate A.6): the per-layer norm/residual topology must
    match the real HF DecoderLayer.forward for families that depart from the
    default sequential pre-norm — Gemma sandwich, OLMo-2 post-norm, Cohere parallel."""
    base = dict(num_hidden_layers=2, hidden_size=128, num_attention_heads=8,
                num_key_value_heads=2, intermediate_size=256, vocab_size=1000, rms_norm_eps=1e-5)

    def ids(cfg):
        L = parse(cfg).layers[-1]
        return L.norm_placement, [b["id"] for b in L.blocks], {b["id"]: b for b in L.blocks}

    # Gemma-2/3 sandwich: input_ln → attn → post_attn_ln → ⊕ → pre_ffn_ln → ffn → post_ffn_ln → ⊕
    pl, order, by = ids(dict(base, model_type="gemma2", sliding_window=512))
    assert pl == "double"
    assert order == ["rms1", "attn", "post_attn_ln", "add1", "rms2", "ffn", "post_ffn_ln", "add2"]
    assert by["add1"]["residual_from"] == "rms1" and by["add1"]["kind"] == "residual_add"

    # OLMo-2 post-norm: attn → post_attn_ln → ⊕ → ffn → post_ffn_ln → ⊕ (no pre-norms)
    pl, order, by = ids(dict(base, model_type="olmo2"))
    assert pl == "post"
    assert order == ["attn", "post_attn_ln", "add1", "ffn", "post_ffn_ln", "add2"]
    assert by["add1"]["residual_from"] == "attn"  # taps the sublayer input (= layer input)

    # Cohere parallel (no config flag): one shared norm feeds attn ∥ ffn → one combined ⊕
    pl, order, by = ids(dict(base, model_type="cohere"))
    assert by["ffn"].get("lane") and by["ffn"].get("feeds") == "add1"
    assert "add2" not in by  # single combined residual add

    # Regression: a standard family stays sequential pre-norm.
    pl, order, _ = ids(dict(base, model_type="llama"))
    assert pl == "pre"
    assert order == ["rms1", "attn", "add1", "rms2", "ffn", "add2"]


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
    assert "Grouped scaled dot-product attention" in html
    assert "GQA 8/1" in html
    assert "Q0-Q7" in html
    assert "use KV0" in html
    assert "Multi-query attention" not in html
    assert "Multi-query scaled dot-product attention" not in html


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
    assert "Vision → tokens" in html
    assert "Audio → tokens" in html
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
    assert "&lt;image&gt; × 280" in html
    assert "&lt;audio&gt; × 750" in html
    assert "AUD × 750" in html
    assert "280 × 1,536" in html
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
    assert "Audio → tokens" in html


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
    assert "Grouped scaled dot-product attention" in html
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
    # card ids match the op-graph region ops (activation/multiply), not silu/mul.
    assert {"up_proj", "activation", "down_proj"} <= child_ids
    assert "gate_proj" not in child_ids
    assert "multiply" not in child_ids

    html = d.to_html(standalone=True)
    assert "Linear (in)" in html
    assert "Linear (gate)" not in html
    assert 'data-card-id="gate_proj"' not in html
    assert 'data-card-id="multiply"' not in html


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
    assert "Multi-query scaled dot-product attention" in html
    assert "Shared K/V cache" in html
    # the aside chip splits the fact into a strong half and a detail half
    assert "1 K + 1 V" in html and "reused by 71 Q" in html
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


def test_param_counts_match_published_within_tolerance():
    """MLA head geometry must be read so total/active params are accurate.

    Regression for the bug where MLA used head_dim = hidden/num_heads (≈56 for
    DeepSeek) instead of qk_nope+qk_rope (192) / v_head_dim (128), undercounting
    every attention layer and pushing active params ~15% low.
    """
    # (config, published_total_B, published_active_B)
    cases = [
        (DEEPSEEK_V3_CONFIG, 671, 37),
        (KIMI_K2_CONFIG, 1000, 32),  # Kimi-K2: ~1T total / 32B active
    ]
    for cfg, pub_total, pub_active in cases:
        p = unfold(cfg).to_ir()["params"]
        total_b = p["total"] / 1e9
        active_b = p["active"] / 1e9
        assert abs(total_b - pub_total) / pub_total < 0.05, \
            f"total {total_b:.1f}B vs published {pub_total}B"
        assert abs(active_b - pub_active) / pub_active < 0.08, \
            f"active {active_b:.1f}B vs published {pub_active}B"


def test_param_estimation_never_crashes_on_list_valued_counts():
    """Some configs declare per-layer/per-block LISTS where a scalar count is
    expected (a heterogeneous MoE schedule). Parameter estimation must degrade to
    an approximate number, never raise — a crash here failed the whole render."""
    from model_unfolder.params import _ffn_params, _attn_params
    from model_unfolder.ir import FFNSpec, AttentionSpec
    f = FFNSpec(kind="moe", activation="silu", intermediate_size=[100, 200],
                gated=True, num_experts=[8, 16], num_experts_per_tok=2,
                num_shared_experts=[1])
    total, active = _ffn_params(f, 128)
    assert total > 0 and active > 0
    a = AttentionSpec(kind="gqa", num_heads=[8, 8], num_kv_heads=[2], head_dim=None)
    assert _attn_params(a, 128) > 0


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
