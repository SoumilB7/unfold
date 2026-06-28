"""Static code-evidence tests."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_unfolder import config_to_ir, inspect_model_code, unfold


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


def test_static_code_evidence_detects_attention_ffn_and_cache(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class FakeMLP:
    def __init__(self, config):
        self.gate_proj = Linear()
        self.up_proj = Linear()
        self.down_proj = Linear()

class FakeAttention:
    def __init__(self, config):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

    def forward(self, hidden_states, past_key_value=None):
        if past_key_value is not None:
            key_states, value_states = past_key_value.update()
        return attention_interface()

class FakeDecoderLayer:
    def __init__(self, config):
        self.self_attn = FakeAttention(config)
        self.mlp = FakeMLP(config)
        self.input_layernorm = Norm()
        self.post_attention_layernorm = Norm()
""",
    )

    evidence = inspect_model_code(tmp_path)

    assert "split_qkv_attention" in evidence.components["attention"]
    assert "grouped_kv_attention" in evidence.components["attention"]
    assert "gated_dense_ffn" in evidence.components["ffn"]
    assert "kv_cache_update" in evidence.components["feature"]
    assert "decoder_layer" in evidence.components["topology"]


def test_static_code_evidence_detects_mla_and_moe(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class DeepseekLikeAttention:
    def __init__(self, config):
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.q_a_proj = Linear()
        self.q_b_proj = Linear()
        self.kv_a_proj_with_mqa = Linear()
        self.kv_b_proj = Linear()
        self.o_proj = Linear()
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim

    def forward(self, hidden_states, past_key_value=None):
        q_nope, q_pe = split(hidden_states)
        k_nope, k_pe = split(hidden_states)
        key_states, value_states = past_key_value.update()
        return attention_interface()

class SparseMoeBlock:
    def __init__(self, config):
        self.router = Router()
        self.experts = Experts()
        self.shared_experts = Experts()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
""",
    )

    evidence = inspect_model_code(tmp_path)

    assert "mla" in evidence.components["attention"]
    assert "latent_kv_cache" in evidence.components["feature"]
    assert "mixture_of_experts" in evidence.components["ffn"]
    assert "shared_experts" in evidence.components["feature"]


def test_config_to_ir_can_attach_code_evidence_from_path(tmp_path):
    _write_modeling_file(
        tmp_path,
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
""",
    )

    ir = config_to_ir(LLAMA_TINY_CONFIG, inspect_code=True, code_source=str(tmp_path))

    assert "code_evidence" in ir.extras
    assert ir.extras["code_evidence"]["provenance"]["source"] == "path"
    assert "grouped_kv_attention" in ir.extras["code_evidence"]["components"]["attention"]

    diagram = unfold(LLAMA_TINY_CONFIG, inspect_code=True, code_source=str(tmp_path))
    html = diagram.to_html(standalone=True)

    assert "code_evidence" in diagram.to_ir()["extras"]
    assert "CODE EVIDENCE" in html
    assert "grouped K/V" in html
    assert "split QKV" in html
    assert "gated dense FFN" in html


def test_code_evidence_section_is_hidden_without_inspection():
    html = unfold(LLAMA_TINY_CONFIG).to_html(standalone=True)

    assert "CODE EVIDENCE" not in html


# ---------------------------------------------------------------------------
# Detectors for custom/quirky setups across families
# ---------------------------------------------------------------------------


def test_detects_per_layer_embedding_pathway(tmp_path):
    """Gemma 3n / Gemma 4-style Per-Layer Embeddings."""
    _write_modeling_file(
        tmp_path,
        """
class Gemma3nTextDecoderLayer:
    def __init__(self, config):
        self.self_attn = FakeAttention()
        self.mlp = FakeMLP()
        self.input_layernorm = Norm()
        self.post_attention_layernorm = Norm()
        self.pre_feedforward_layernorm = Norm()
        self.post_feedforward_layernorm = Norm()
        self.per_layer_input_gate = Linear()
        self.per_layer_projection = Linear()
        self.post_per_layer_input_norm = Norm()
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.altup = AltUp()
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "per_layer_embedding_pathway" in ev.components["topology"]
    assert "double_ffn_norm" in ev.components["topology"]


def test_detects_altup_routing(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class Gemma3nTextAltUp:
    def __init__(self, config):
        self.modality_router = Linear()
        self.router_norm = Norm()
        self.prediction_coefs = Param()
        self.correction_coefs = Param()
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "altup_routing" in ev.components["topology"]


def test_detects_cross_layer_kv_sharing(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class Gemma3nTextAttention:
    def __init__(self, config, layer_idx):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
        self.is_kv_shared_layer = layer_idx >= config.first_kv_shared_layer
        self.kv_shared_layer_index = config.kv_shared_layer_index
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "cross_layer_kv_sharing" in ev.components["feature"]


def test_detects_attention_logit_softcap(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class Gemma2Attention:
    def __init__(self, config):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
        self.attn_logit_softcapping = config.attn_logit_softcapping
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "attention_logit_softcap" in ev.components["feature"]


def test_detects_alibi_via_calls(tmp_path):
    """BLOOM/MPT/Falcon ALiBi: computed in model.forward, never stored as field."""
    _write_modeling_file(
        tmp_path,
        """
class BloomModel:
    def __init__(self, config):
        self.num_heads = config.num_attention_heads

    def forward(self, input_ids):
        alibi = build_alibi_tensor(self.num_heads, input_ids.shape[-1])
        return alibi
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "alibi_position_bias" in ev.components["feature"]


def test_detects_partial_rotary_via_config_refs(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class GPTNeoXAttention:
    def __init__(self, config):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
        self.rotary_pct = config.rotary_pct
        self.rotary_ndims = int(config.rotary_pct * config.head_dim)
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "partial_rotary_embedding" in ev.components["feature"]


def test_detects_fine_grained_moe_routing_deepseek_style(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class DeepseekV3MoE:
    def __init__(self, config):
        self.experts = Experts()
        self.shared_experts = Experts()
        self.gate = Gate()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, x):
        return route_tokens_to_experts(x, self.gate, self.experts)
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "mixture_of_experts" in ev.components["ffn"]
    assert "shared_experts" in ev.components["feature"]
    assert "fine_grained_expert_routing" in ev.components["feature"]


def test_detects_mla_with_decoupled_rope_heads(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class DeepseekV3Attention:
    def __init__(self, config):
        self.q_a_proj = Linear()
        self.q_b_proj = Linear()
        self.kv_a_proj_with_mqa = Linear()
        self.kv_b_proj = Linear()
        self.o_proj = Linear()
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "mla" in ev.components["attention"]
    assert "latent_kv_cache" in ev.components["feature"]
    assert "decoupled_rope_heads" in ev.components["feature"]


def test_detects_multi_token_prediction(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class DeepseekV3MTPLayer:
    def __init__(self, config):
        self.mtp_proj = Linear()
        self.mtp_norm = Norm()
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "multi_token_prediction" in ev.components["topology"]


def test_detects_double_ffn_norm_gemma2_style(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class Gemma2DecoderLayer:
    def __init__(self, config):
        self.self_attn = FakeAttention()
        self.mlp = FakeMLP()
        self.input_layernorm = Norm()
        self.post_attention_layernorm = Norm()
        self.pre_feedforward_layernorm = Norm()
        self.post_feedforward_layernorm = Norm()
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "double_ffn_norm" in ev.components["topology"]
    assert "decoder_layer" in ev.components["topology"]


def test_detects_falcon_parallel_residual_candidates(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class FalconDecoderLayer:
    def __init__(self, config):
        self.self_attention = FakeAttention()
        self.mlp = FakeMLP()
        self.ln_attn = Norm()
        self.ln_mlp = Norm()
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "parallel_residual_candidates" in ev.components["topology"]


def test_detects_qk_norm_cohere_style(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class CohereAttention:
    def __init__(self, config):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
        self.use_qk_norm = config.use_qk_norm
        self.q_norm = Norm()
        self.k_norm = Norm()
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "qk_norm" in ev.components["feature"]


def test_detects_nope_layer_interleaving(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class Llama4TextAttention:
    def __init__(self, config, layer_idx):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
        self.use_rope = (layer_idx + 1) % config.no_rope_layer_interval != 0
        self.qk_norm = config.qk_norm
""",
    )
    ev = inspect_model_code(tmp_path)

    assert "nope_layer_interleaving" in ev.components["feature"]


# ---------------------------------------------------------------------------
# Typed positional evidence — model stage + attention stage, config selected
# ---------------------------------------------------------------------------


def test_positional_evidence_real_counterexample_matrix():
    """The exact regressions which disprove a flat/file-presence detector."""
    from transformers import AutoConfig
    from model_unfolder.evidence.position import decoder_positional_evidence

    expected = {
        "bloom": "alibi",
        "mpt": "alibi",
        "gpt2": "learned_absolute",
        "gpt_neo": "learned_absolute",
        "opt": "learned_absolute",
        "gpt_bigcode": "learned_absolute",
        "openai-gpt": "learned_absolute",
        "xglm": "fixed_absolute",
        "ctrl": "fixed_absolute",
        "gptj": "rope",
        "llama": "rope",
        "mistral": "rope",
        "phi3": "rope",
        "mixtral": "rope",
    }
    seen = 0
    for model_type, kind in expected.items():
        try:
            cfg = AutoConfig.for_model(model_type).to_dict()
        except (KeyError, ValueError):
            continue
        evidence = decoder_positional_evidence(cfg)
        assert evidence.status == "proven", (model_type, evidence)
        assert evidence.kinds == {kind}, (model_type, evidence)
        seen += 1
    assert seen >= 11


def test_falcon_config_selects_rope_or_alibi_from_same_source():
    from transformers import AutoConfig
    from model_unfolder.evidence.position import decoder_positional_evidence

    cfg = AutoConfig.for_model("falcon").to_dict()
    rope = decoder_positional_evidence({**cfg, "alibi": False})
    alibi = decoder_positional_evidence({**cfg, "alibi": True})
    assert rope.status == alibi.status == "proven"
    assert rope.kinds == {"rope"}
    assert alibi.kinds == {"alibi"}
    assert rope.mechanisms[0].source_file == alibi.mechanisms[0].source_file


def test_zero_rotary_geometry_is_a_proven_noop():
    from transformers import AutoConfig
    from model_unfolder.evidence.position import decoder_positional_evidence

    cfg = AutoConfig.for_model("gpt_neox").to_dict()
    evidence = decoder_positional_evidence({**cfg, "rotary_pct": 0.0})
    assert evidence.status == "proven"
    assert evidence.kinds == {"none"}


def test_hybrid_schedule_proves_rope_and_positionless_mixer():
    from transformers import AutoConfig
    from model_unfolder.evidence.position import decoder_positional_evidence

    try:
        cfg = AutoConfig.for_model("qwen3_5").to_dict()
    except (KeyError, ValueError):
        pytest.skip("installed transformers has no qwen3_5")
    evidence = decoder_positional_evidence(cfg)
    assert evidence.status == "proven"
    assert evidence.kinds == {"rope", "none"}


def test_multimodal_shared_file_uses_the_qualified_text_component():
    from transformers import AutoConfig
    from model_unfolder.evidence.position import decoder_positional_evidence

    try:
        cfg = AutoConfig.for_model("qwen3_5").to_dict()
    except (KeyError, ValueError):
        pytest.skip("installed transformers has no qwen3_5")
    evidence = decoder_positional_evidence(cfg)
    assert evidence.status == "proven"
    assert evidence.component == "text_config"
    assert {item.class_name for item in evidence.mechanisms} == {
        "Qwen3_5Attention", "Qwen3_5GatedDeltaNet",
    }
    assert not any("Vision" in item.class_name for item in evidence.mechanisms)


def test_model_input_absolute_add_can_coexist_with_attention_rope(tmp_path, monkeypatch):
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.evidence.position import decoder_positional_evidence
    from model_unfolder.evidence import position as position_module

    source = _write_modeling_file(
        tmp_path,
        """
def apply_rotary_pos_emb(q, k):
    return q, k

class FakeAttention:
    def forward(self, x):
        q, k = apply_rotary_pos_emb(x, x)
        return q + k

class FakeMLP:
    def forward(self, x):
        return x

class FakeBlock:
    def __init__(self):
        self.attn = FakeAttention()
        self.mlp = FakeMLP()
    def forward(self, x):
        return self.mlp(self.attn(x))

class FakeModel:
    def __init__(self):
        self.wpe = Embedding()
        self.layers = ModuleList([FakeBlock()])
    def forward(self, input_ids, position_ids):
        x = input_ids + self.wpe(position_ids)
        for layer in self.layers:
            x = layer(x)
        return x
""",
    )
    bundle = SourceBundle(
        source="path", files=(str(source),), model_type="fake",
        component_files={"root": (str(source),)},
        component_architectures={"root": "FakeModel"},
    )
    monkeypatch.setattr(position_module, "resolve_source_files", lambda *a, **k: bundle)
    evidence = decoder_positional_evidence({"model_type": "fake"})
    assert evidence.status == "proven"
    assert evidence.kinds == {"learned_absolute", "rope"}


def test_dead_rotary_helper_does_not_count_as_applied(tmp_path, monkeypatch):
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.evidence.position import decoder_positional_evidence
    from model_unfolder.evidence import position as position_module

    source = _write_modeling_file(
        tmp_path,
        """
def apply_rotary_pos_emb(q, k):
    return q, k

class FakeAttention:
    def forward(self, x):
        return x

class FakeMLP:
    def forward(self, x):
        return x

class FakeBlock:
    def __init__(self):
        self.attn = FakeAttention()
        self.mlp = FakeMLP()
    def forward(self, x, past_key_value=None):
        return self.mlp(self.attn(x))

class FakeModel:
    def __init__(self):
        self.layers = ModuleList([FakeBlock()])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
""",
    )
    bundle = SourceBundle(
        source="path", files=(str(source),), model_type="fake",
        component_files={"root": (str(source),)},
        component_architectures={"root": "FakeModel"},
    )
    monkeypatch.setattr(position_module, "resolve_source_files", lambda *a, **k: bundle)
    evidence = decoder_positional_evidence({"model_type": "fake"})
    assert evidence.status == "ambiguous"
    assert not evidence.mechanisms


def test_oracle_missing_is_distinct_from_present_but_ambiguous(monkeypatch):
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.evidence.position import decoder_positional_evidence
    from model_unfolder.evidence import position as position_module

    monkeypatch.setattr(
        position_module, "resolve_source_files",
        lambda *a, **k: SourceBundle(source="local", model_type="missing"),
    )
    evidence = decoder_positional_evidence({"model_type": "missing"})
    assert evidence.status == "oracle_missing"


# ---------------------------------------------------------------------------
# Validation cross-checks
# ---------------------------------------------------------------------------


def test_validate_warns_on_mla_in_code_but_not_in_ir(tmp_path):
    """If MLA-shaped attention is in the modeling file but the parsed IR
    has none, validation must emit a warning."""
    _write_modeling_file(
        tmp_path,
        """
class DeepseekV3Attention:
    def __init__(self, config):
        self.q_a_proj = Linear()
        self.q_b_proj = Linear()
        self.kv_a_proj_with_mqa = Linear()
        self.kv_b_proj = Linear()
        self.o_proj = Linear()
""",
    )

    # Llama IR (no MLA) + code evidence that says MLA → warning expected.
    ir = config_to_ir(LLAMA_TINY_CONFIG, inspect_code=True, code_source=str(tmp_path))

    assert any("MLA" in w for w in ir.warnings)


def test_validate_warns_on_ple_in_code_but_not_in_ir(tmp_path):
    _write_modeling_file(
        tmp_path,
        """
class GemmaLikeDecoderLayer:
    def __init__(self, config):
        self.self_attn = FakeAttention()
        self.mlp = FakeMLP()
        self.input_layernorm = Norm()
        self.post_attention_layernorm = Norm()
        self.per_layer_input_gate = Linear()
        self.per_layer_projection = Linear()
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
""",
    )

    ir = config_to_ir(LLAMA_TINY_CONFIG, inspect_code=True, code_source=str(tmp_path))

    assert any("Per-Layer Embedding" in w or "PLE" in w for w in ir.warnings)


def _write_modeling_file(tmp_path, body: str):
    path = tmp_path / "modeling_fake.py"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path
