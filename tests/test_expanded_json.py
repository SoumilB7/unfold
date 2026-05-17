"""Expanded architecture JSON tests."""
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
