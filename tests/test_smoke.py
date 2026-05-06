"""Smoke test: parse real configs and verify IR + HTML output."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_viz import visualize

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
    d = visualize(KIMI_K2_CONFIG)
    ir = d.to_ir()
    assert ir["name"] == "Kimi-K2-Instruct"
    assert ir["vocab_size"] == 163840
    assert len(ir["layers"]) == 61
    assert ir["layers"][0]["ffn"]["kind"] == "dense"
    assert ir["layers"][1]["ffn"]["kind"] == "moe"
    assert ir["layers"][1]["ffn"]["num_experts"] == 384
    assert ir["layers"][0]["attention"]["kind"] == "mla"

    html = d.to_html(standalone=True)
    assert "<!doctype html>" in html.lower()
    assert "TransformerViz" in html

    fragment = d._repr_html_()
    assert "<script>" in fragment
    assert d._mount_id in fragment

    print("Kimi K2 OK")


def test_deepseek_v3_phase_change():
    d = visualize(DEEPSEEK_V3_CONFIG)
    ir = d.to_ir()
    assert len(ir["layers"]) == 61
    for i in range(3):
        assert ir["layers"][i]["ffn"]["kind"] == "dense"
    for i in range(3, 61):
        assert ir["layers"][i]["ffn"]["kind"] == "moe"
    assert ir["layers"][3]["ffn"]["num_experts"] == 256
    print("DeepSeek-V3 phase change OK")


def test_llama3():
    d = visualize(LLAMA3_8B_CONFIG)
    ir = d.to_ir()
    assert len(ir["layers"]) == 32
    assert ir["layers"][0]["attention"]["kind"] == "gqa"
    assert ir["layers"][0]["attention"]["num_heads"] == 32
    assert ir["layers"][0]["attention"]["num_kv_heads"] == 8
    assert ir["layers"][0]["ffn"]["kind"] == "dense"
    print("Llama-3 OK")


if __name__ == "__main__":
    test_kimi_k2()
    test_deepseek_v3_phase_change()
    test_llama3()
    print("\nAll smoke tests passed.")
