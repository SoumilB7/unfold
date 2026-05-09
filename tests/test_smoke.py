"""Smoke test: parse real configs and verify IR + HTML output."""
import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unfold import unfold

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
    assert "<script" not in html.lower()
    assert "<svg" in html.lower()

    fragment = d._repr_html_()
    assert "<script" not in fragment.lower()
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


def test_llama3():
    d = unfold(LLAMA3_8B_CONFIG)
    ir = d.to_ir()
    assert len(ir["layers"]) == 32
    assert ir["layers"][0]["attention"]["kind"] == "gqa"
    assert ir["layers"][0]["attention"]["num_heads"] == 32
    assert ir["layers"][0]["attention"]["num_kv_heads"] == 8
    assert ir["layers"][0]["ffn"]["kind"] == "dense"
    assert ir["params"]["is_sparse"] is False
    print(f"Llama-3 OK  — ~{ir['params']['total_h']} params")


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
    assert calls[0][1]["trust_remote_code"] is True
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
    test_llama3()
    test_model_id_uses_hf_token_env_and_explicit_override()
    test_model_id_falls_back_to_legacy_hf_auth_kwarg()
    print("\nAll smoke tests passed.")
