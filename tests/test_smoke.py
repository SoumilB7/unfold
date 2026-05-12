"""Smoke test: parse real configs and verify IR + HTML output."""
import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_unfolder import unfold

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

    assert ple["detail_view"] == "per_layer_embedding"
    assert ple["detail"]["view"] == "per_layer_embedding"
    assert ple["detail"]["nodes"]["multiply"] == "ple_mul"
    assert ir["extras"]["per_layer_embeddings"]["hidden"] == 1024
    assert ir["extras"]["external_pathways"][0]["tap_block"] == "ple_mul"

    html = d.to_html(standalone=True)
    assert "uf-card-ple" in html
    assert "uf-l3-ple_gate" in html
    assert "uf-l3-per_layer_input" in html
    assert "per_layer_input[L]" in html
    assert "gelu_pytorch_tanh" not in html.lower()


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
    assert ffn_block["detail_view"] == "dense_ffn"
    assert {"up_proj", "silu", "down_proj"} <= child_ids
    assert "gate_proj" not in child_ids
    assert "mul" not in child_ids

    html = d.to_html(standalone=True)
    assert "Linear (in)" in html
    assert "Linear (gate)" not in html
    assert "uf-l3-gate_proj" not in html
    assert "uf-l3-mul" not in html


def test_falcon_parallel_attn_uses_parallel_topology():
    d = unfold(FALCON_PARALLEL_CONFIG)
    ir = d.to_ir()
    blocks = ir["layers"][0]["blocks"]
    block_by_id = {block["id"]: block for block in blocks}

    assert ir["layers"][0]["attention"]["kind"] == "mqa"
    assert ir["extras"]["parallel_attn"] is True
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


def test_attention_detail_views_dispatch_by_kind():
    base = {
        "vocab_size": 32000,
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "hidden_act": "silu",
    }
    cases = [
        (
            "mla",
            KIMI_K2_CONFIG,
            "mla_kv_down",
            "Multi-Head Latent",
            True,
        ),
        (
            "ssm",
            {
                **base,
                "architectures": ["FalconH1ForCausalLM"],
                "model_type": "falcon_h1",
                "mamba_d_state": 16,
                "mamba_d_ssm": 64,
                "attn_layer_indices": None,
            },
            "ssm_scan",
            "Selective Scan",
            True,
        ),
        (
            "recurrent",
            {
                **base,
                "architectures": ["RecurrentGemmaForCausalLM"],
                "model_type": "recurrent_gemma",
                "block_types": ["recurrent", "recurrent", "attention"],
                "attention_window_size": 128,
                "lru_width": 32,
            },
            "lru_state",
            "Recurrent State",
            False,
        ),
        (
            "rwkv",
            {
                "architectures": ["RwkvForCausalLM"],
                "model_type": "rwkv",
                "vocab_size": 32000,
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "attention_hidden_size": 64,
                "head_size": 16,
                "intermediate_size": 224,
            },
            "rwkv_time_mix",
            "Time-Mix",
            True,
        ),
        (
            "linear",
            {
                **base,
                "architectures": ["MiniMaxText01ForCausalLM"],
                "model_type": "minimax_text_01",
                "attn_type_list": [0, 0, 0, 0],
                "num_local_experts": 8,
                "num_experts_per_tok": 2,
            },
            "linear_mix",
            "Linear Attention Mix",
            True,
        ),
    ]

    for kind, cfg, expected_child, expected_html, forbid_sdpa in cases:
        d = unfold(cfg)
        ir = d.to_ir()
        attn_block = next(block for block in ir["layers"][0]["blocks"] if block["id"] == "attn")
        child_ids = {child["id"] for child in attn_block["children"]}
        html = d.to_html(standalone=True)

        assert ir["layers"][0]["attention"]["kind"] == kind
        assert expected_child in child_ids
        assert expected_html in html
        if forbid_sdpa:
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
    assert "trust_remote_code" not in calls[0][1]
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
    assert "trust_remote_code" not in calls[1]


def test_model_id_uses_remote_code_only_when_required():
    calls = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(dict(kwargs))
            if not kwargs.get("trust_remote_code"):
                raise ValueError("set trust_remote_code=True to execute the configuration file")
            return LLAMA3_8B_CONFIG

    _with_fake_transformers(
        FakeAutoConfig,
        lambda: unfold("custom/model"),
    )

    assert "trust_remote_code" not in calls[0]
    assert calls[1]["trust_remote_code"] is True


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
    test_gemma4_31b()
    test_gemma4_ple_uses_reusable_part_contract()
    test_llama3()
    test_model_id_uses_hf_token_env_and_explicit_override()
    test_model_id_falls_back_to_legacy_hf_auth_kwarg()
    test_model_id_uses_remote_code_only_when_required()
    print("\nAll smoke tests passed.")
