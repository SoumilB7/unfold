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


def test_field_types_resolve_constructor_classmethod_factories():
    """``self.x = Klass._from_config(cfg)`` types the field as ``Klass`` (never
    the factory's method name) — the general delegated-construction signature
    behind the transformers ``*WithProjection`` wrappers.  ``AutoModel`` stays
    the honest address name; concrete resolution belongs to the qualified
    component rail, not the shared extractor."""
    import ast
    from model_unfolder.evidence.forward_ops import _field_types, _method

    tree = ast.parse(
        "class Wrapper:\n"
        "    def __init__(self, config):\n"
        "        self.inner = InnerTower._from_config(config)\n"
        "        self.other = OtherTower.from_config(config.sub_config)\n"
        "        self.auto = AutoModel.from_config(config.vision_config)\n"
        "        self.plain = PlainLayer(config)\n"
    )
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    types = _field_types(_method(node, "__init__"))
    assert types["inner"] == "InnerTower"
    assert types["other"] == "OtherTower"
    assert types["auto"] == "AutoModel"
    assert types["plain"] == "PlainLayer"


def test_relative_bias_is_a_proven_positional_mechanism(tmp_path, monkeypatch):
    """A learned bias over bucketed relative distances added to the scores
    (the T5-family code shape, matched by general markers) proves
    ``relative_bias`` at the ``attention_bias`` altitude — no more
    present-but-ambiguous for encoders built this way."""
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.evidence.position import decoder_positional_evidence
    from model_unfolder.evidence import position as position_module

    source = _write_modeling_file(
        tmp_path,
        """
class NovelBiasAttention:
    def __init__(self):
        self.relative_attention_bias = Embedding()
    def compute_bias(self, q_len, k_len):
        return self.relative_attention_bias(q_len)
    def forward(self, x):
        scores = matmul(x, x)
        position_bias = self.compute_bias(1, 1)
        scores += position_bias
        return softmax(scores)

class FakeMLP:
    def forward(self, x):
        return x

class FakeBlock:
    def __init__(self):
        self.attn = NovelBiasAttention()
        self.mlp = FakeMLP()
    def forward(self, x):
        return self.mlp(self.attn(x))

class FakeModel:
    def __init__(self):
        self.layers = ModuleList([FakeBlock()])
    def forward(self, input_ids):
        x = input_ids
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
    assert evidence.kinds == {"relative_bias"}
    mechanism = evidence.mechanisms[0]
    assert mechanism.application == "attention_bias"
    assert mechanism.class_name == "NovelBiasAttention"


def test_attention_score_scaling_verdicts_are_code_derived(tmp_path):
    """scores_scaled: True on an explicit scale symbol or an SDPA terminal,
    False only for a provably raw matmul, None for wrapper-only/mixed files."""
    from model_unfolder.evidence.patterns import attention_score_scaling_from_files

    scaled = tmp_path / "scaled.py"
    scaled.write_text(
        "class ScaledAttention:\n"
        "    def __init__(self):\n"
        "        self.scaling = 0.125\n"
        "    def forward(self, q, k):\n"
        "        return matmul(q, k) * self.scaling\n"
    )
    unscaled = tmp_path / "unscaled.py"
    unscaled.write_text(
        "class RawAttention:\n"
        "    def forward(self, q, k):\n"
        "        scores = matmul(q, k)\n"
        "        return softmax(scores)\n"
    )
    delegated = tmp_path / "delegated.py"
    delegated.write_text(
        "class DelegatedAttention:\n"
        "    def forward(self, q, k):\n"
        "        return scaled_dot_product_attention(q, k, k)\n"
    )
    wrapper_only = tmp_path / "wrapper.py"
    wrapper_only.write_text(
        "class WrapperAttention:\n"
        "    def __init__(self):\n"
        "        self.inner = Something()\n"
        "    def forward(self, x):\n"
        "        return self.inner(x)\n"
    )
    assert attention_score_scaling_from_files((scaled,)) is True
    assert attention_score_scaling_from_files((unscaled,)) is False
    assert attention_score_scaling_from_files((delegated,)) is True
    assert attention_score_scaling_from_files((wrapper_only,)) is None
    # Mixed verdicts across one file's attention classes stay honestly unproven.
    assert attention_score_scaling_from_files((scaled, unscaled)) is None


def test_storage_fidelity_detectors_are_code_shaped(tmp_path):
    """Storage detectors fire on code shape, never a name.

    Embedding-stage normalization moved to the U3 exact-owner/flow reader and
    is pinned separately in ``test_embedding_bookend.py``.
    """
    from model_unfolder.evidence.patterns import (
        attention_fused_qkv_from_files,
        expert_fused_gate_up_from_files,
    )

    fused = tmp_path / "fused.py"
    fused.write_text(
        "class NovelExperts:\n"
        "    def __init__(self):\n"
        "        self.gate_up_proj = Parameter()\n"
        "        self.down_proj = Parameter()\n"
        "    def forward(self, x):\n"
        "        gate_up = linear(x, self.gate_up_proj)\n"
        "        gate = gate_up[..., ::2]\n"          # interleaved split, no chunk()
        "        up = gate_up[..., 1::2]\n"
        "        return linear(gate * up, self.down_proj)\n"
        "class NovelAttention:\n"
        "    def __init__(self):\n"
        "        self.query_key_value = Linear()\n"
        "    def forward(self, x):\n"
        "        qkv = self.query_key_value(x)\n"
        "        return qkv\n"
        "class NovelModel:\n"
        "    def __init__(self):\n"
        "        self.word_embeddings = Embedding()\n"
        "        self.word_embeddings_layernorm = LayerNorm()\n"
        "        self.layers = ModuleList([NovelAttention()])\n"
        "    def forward(self, input_ids):\n"
        "        h = self.word_embeddings(input_ids)\n"
        "        h = self.word_embeddings_layernorm(h)\n"
        "        return h\n"
    )
    split = tmp_path / "split.py"
    split.write_text(
        "class PlainAttention:\n"
        "    def __init__(self):\n"
        "        self.q_proj = Linear(); self.k_proj = Linear(); self.v_proj = Linear()\n"
        "    def forward(self, x):\n"
        "        return self.q_proj(x)\n"
        "class PlainModel:\n"
        "    def __init__(self):\n"
        "        self.embed_tokens = Embedding()\n"
        "        self.norm = RMSNorm()\n"
        "    def forward(self, input_ids):\n"
        "        h = self.embed_tokens(input_ids)\n"
        "        for layer in []:\n"
        "            h = layer(h)\n"
        "        return self.norm(h)\n"                # FINAL norm, not embed-stage?
    )
    assert expert_fused_gate_up_from_files((fused,)) is True
    assert expert_fused_gate_up_from_files((split,)) is None
    assert attention_fused_qkv_from_files((fused,)) is True
    assert attention_fused_qkv_from_files((split,)) is False
_CHATGLM_SHAPED = """
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_pos_emb(x, rope_cache):
    return x


class CoreAttention(nn.Module):
    def __init__(self, config, layer_number):
        super().__init__()

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, value_layer)


class SelfAttention(nn.Module):
    def __init__(self, config, layer_number):
        super().__init__()
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.core_attention = CoreAttention(config, layer_number)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):
        mixed = self.query_key_value(hidden_states)
        query_layer, key_layer, value_layer = mixed.split(3, dim=-1)
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)
        context = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        return self.dense(context)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size * 2)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        intermediate = self.dense_h_to_4h(hidden_states)
        intermediate = self.activation_func(intermediate)
        return self.dense_4h_to_h(intermediate)


class GLMBlock(nn.Module):
    def __init__(self, config, layer_number):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.self_attention = SelfAttention(config, layer_number)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):
        out = self.input_layernorm(hidden_states)
        out = self.self_attention(out, attention_mask, rotary_pos_emb)
        hidden_states = hidden_states + out
        out = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.mlp(out)
"""


def test_init_local_fn_and_inner_kernel_evidence(tmp_path):
    """The ChatGLM-shaped code signatures, read GENERALLY (no names):

    1. a nested fn bound to ``self.F`` in __init__ (swiglu) is FOLDED into the
       class's op scan — so a 2-linear MLP that chunks one fused projection and
       multiplies the halves is proven GATED with fused storage;
    2. an inner attention kernel constructed as a FIELD of the attention class
       (core_attention) is not a rival positional candidate — the owner's RoPE
       application is proven, never "candidates disagree".
    """
    f = tmp_path / "modeling_x.py"
    f.write_text(_CHATGLM_SHAPED)
    files = (str(f),)

    from model_unfolder.evidence.patterns import decoder_ffn_gated_from_files
    assert decoder_ffn_gated_from_files(files, cfg={}) is True

    from model_unfolder.evidence.ffn import ffn_structure_evidence
    ev = ffn_structure_evidence(files, expected_gated=True)
    assert ev.status == "proven" and ev.projection_mode == "fused_gate_up"

    from model_unfolder.evidence.position import decoder_positional_evidence
    from model_unfolder.evidence.models import SourceBundle
    bundle = SourceBundle(source="path", files=files,
                          component_files={"root": files})
    pos = decoder_positional_evidence({}, bundle=bundle)
    assert pos.status == "proven"
    assert any(m.kind == "rope" for m in pos.mechanisms)


def test_remote_code_declaration_reaches_the_hub_rail(monkeypatch, tmp_path):
    """A config that DECLARES auto_map (remote code) resolves its evidence from
    the repo's own .py files when nothing is installed — the address is the
    id, the DECLARATION is the config's, and offline failure stays honest."""
    f = tmp_path / "modeling_x.py"
    f.write_text(_CHATGLM_SHAPED)
    import model_unfolder.evidence.sources as S
    calls = {}

    def fake_hub(target, *, token=None):
        calls["hit"] = True
        return S.SourceBundle(source="hub", files=(str(f),),
                              component_files={"root": (str(f),)})
    monkeypatch.setattr(S, "_hub_bundle", fake_hub)
    cfg = {"model_type": "notinstalled_xyz", "auto_map": {"AutoModel": "modeling_x.X"},
           "_name_or_path": "some/repo"}
    bundle = S.resolve_source_files(cfg, source="local")
    assert calls.get("hit") and bundle.files
    # without the declaration, no hub attempt is made
    calls.clear()
    bundle = S.resolve_source_files({"model_type": "notinstalled_xyz"}, source="local")
    assert not calls.get("hit")


# ---------------------------------------------------------------------------
# UNIT 1 — SOURCE PARITY (run_77 R1/R2/R3): the loader stamps the address,
# unknown model_type falls through to the declared class, and the true
# refusal cause is never masked. One resolution context for ship AND audit.
# ---------------------------------------------------------------------------


def test_raw_json_loader_stamps_repo_id(monkeypatch, tmp_path):
    """The raw-JSON rung (remote-code / registry-predating repos) must stamp
    ``_repo_id`` so ``resolve_source_files`` can fetch the repo's own modeling
    source — without it every model on this rung parses evidence-blind."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        '{"model_type": "notinstalled_xyz", "hidden_size": 64, '
        '"num_hidden_layers": 2, "auto_map": {"AutoModel": "modeling_x.X"}}'
    )
    import model_unfolder.parser as P

    def fake_download(**kwargs):
        assert kwargs["repo_id"] == "some/remote-repo"
        return str(cfg_file)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    cfg = P._load_raw_config_json("some/remote-repo", None)
    assert cfg["_repo_id"] == "some/remote-repo"
    # the stamp is exactly what the source resolver's id lookup prioritizes
    import model_unfolder.evidence.sources as S
    assert S._model_id(cfg) == "some/remote-repo"


def test_unknown_model_type_falls_through_to_declared_class():
    """A PRESENT-but-unregistered model_type (the rebrand pattern: kimi_k2
    running the installed DeepseekV3 class) must resolve source by the
    declared architecture, exactly as an ABSENT model_type already does."""
    import model_unfolder.evidence.sources as S
    cfg = {"model_type": "totally_unknown_rebrand_zz",
           "architectures": ["LlamaForCausalLM"]}
    bundle = S._installed_transformers_bundle(cfg)
    assert bundle.files, bundle.warnings
    assert any("modeling_llama" in f for f in bundle.files)
    assert bundle.component_architectures.get("root") == "LlamaForCausalLM"
    # absent architecture keeps the honest no-source warning
    empty = S._installed_transformers_bundle({"model_type": "totally_unknown_rebrand_zz"})
    assert not empty.files and empty.warnings


def test_source_directory_uses_installed_registry_not_a_project_identity_table(
        monkeypatch, tmp_path):
    """Source addressing follows the installed library's own registry.

    The project must not grow a model-type-to-directory table or guess parent
    packages by chopping role suffixes.
    """
    import model_unfolder.evidence.sources as S
    from transformers.models.auto import configuration_auto

    models_root = tmp_path / "models"
    declared = models_root / "registry_selected_module"
    declared.mkdir(parents=True)
    monkeypatch.setattr(
        configuration_auto, "model_type_to_module_name",
        lambda _declared_type: declared.name,
    )
    assert S._transformers_family_dir(
        models_root, "arbitrary_declared_type") == declared.name
    assert not hasattr(S, "MODEL_TYPE_TO_TRANSFORMERS_DIR")

    # If registry metadata names no installed module, an invented suffix-parent
    # must not be selected merely because that directory happens to exist.
    (models_root / "ambiguous").mkdir()
    monkeypatch.setattr(
        configuration_auto, "model_type_to_module_name",
        lambda _declared_type: "not_installed",
    )
    assert S._transformers_family_dir(models_root, "ambiguous_text") is None


def test_fileless_hub_warning_is_not_masked(monkeypatch):
    """When the remote-code hub lookup fails (offline/gated/id-less), its TRUE
    cause must surface beside the local fallback warning instead of being
    replaced by the generic 'no installed source' line."""
    import model_unfolder.evidence.sources as S

    def failing_hub(target, *, token=None):
        raise RuntimeError("simulated offline")

    monkeypatch.setattr(S, "_hub_bundle", failing_hub)
    cfg = {"model_type": "notinstalled_xyz",
           "auto_map": {"AutoModel": "modeling_x.X"}}
    bundle = S.resolve_source_files(cfg, source="local")
    assert not bundle.files
    assert any("remote-code source fetch failed" in w for w in bundle.warnings)



# ---------------------------------------------------------------------------
# QK-norm — code-first (the code decides the SHAPE and names its own gate)
# ---------------------------------------------------------------------------

def test_qk_norm_resolution_states():
    """Parser arbitration: positive code evidence or checkpoint fallback."""
    from model_unfolder.adapters.transformer.parser import _resolve_qk_norm_layers
    from model_unfolder.evidence.qk_norm import (
        QKNormCodeEvidence,
        QKNormGateAtom,
    )

    # unconditional: config not consulted
    assert _resolve_qk_norm_layers(
        QKNormCodeEvidence(present=True), {}, False, 3) == [True] * 3
    # gated: the named field's VALUE decides
    gated = QKNormCodeEvidence(
        present=None, gate=(QKNormGateAtom(
            "qk_layernorm", ("qk_layernorm",)),))
    assert _resolve_qk_norm_layers(gated, {"qk_layernorm": True}, False, 2) == [True] * 2
    assert _resolve_qk_norm_layers(gated, {"qk_layernorm": False}, True, 2) == [False] * 2
    assert _resolve_qk_norm_layers(gated, {}, True, 2) == [True] * 2
    # per-layer atom: the code indexes its own field by layer
    comp = QKNormCodeEvidence(present=None, gate=(
        QKNormGateAtom(
            "no_rope_layers", ("no_rope_layers",), per_layer=True),
        QKNormGateAtom("use_qk_norm", ("use_qk_norm",)),
    ))
    cfg = {"use_qk_norm": True, "no_rope_layers": [1, 1, 0, 1]}
    assert _resolve_qk_norm_layers(comp, cfg, False, 4) == [True, True, False, True]
    # unresolvable gate value -> honest fallback to the declared spelling
    assert _resolve_qk_norm_layers(comp, {"use_qk_norm": True}, True, 4) == [True] * 4
    # no source at all -> the declaration stands (a declaration is evidence)
    assert _resolve_qk_norm_layers(None, {}, True, 2) == [True] * 2


def test_qk_norm_ships_for_config_silent_oracle_models():
    """The ship-path fix the 21-LLM sweep demanded: Qwen3/OLMo-2/Gemma-3 build
    q/k norms unconditionally with SILENT configs — the default parse (no
    inspect_code flag) must now carry the fact."""
    from transformers import AutoConfig
    for mt in ("qwen3", "olmo2", "gemma3_text"):
        ir = config_to_ir(AutoConfig.for_model(mt))
        assert ir.layers and all(l.attention.qk_norm for l in ir.layers), mt


def test_qk_norm_declared_gate_still_decides_for_stablelm():
    from transformers import AutoConfig
    cfg = AutoConfig.for_model("stablelm")          # qk_layernorm defaults False
    assert not any(l.attention.qk_norm for l in config_to_ir(cfg).layers)
    cfg.qk_layernorm = True
    assert all(l.attention.qk_norm for l in config_to_ir(cfg).layers)


def test_llama4_qk_norm_skips_nope_layers_and_positions_follow_the_code():
    """Fabrication half of the sweep finding + the positional bug the gate
    exposed: NoPE placement must follow ``config.no_rope_layers`` (the field
    the code indexes — NoPE at layers 3, 7, 11…), and QK-norm must sit on
    exactly the rope layers."""
    from transformers import AutoConfig
    cfg = AutoConfig.for_model("llama4_text")
    ir = config_to_ir(cfg)
    qk = [bool(l.attention.qk_norm) for l in ir.layers]
    nope = [bool(l.attention.no_rope) for l in ir.layers]
    assert any(nope) and sum(nope) * 4 == len(ir.layers)
    assert all(q == (not n) for q, n in zip(qk, nope))
    assert [i for i, n in enumerate(nope) if n][:3] == [3, 7, 11]


def test_qk_norm_stays_absent_for_mla_and_plain_oracle_models():
    from transformers import AutoConfig
    for mt in ("llama", "deepseek_v3", "gemma2", "gpt_neox", "bloom"):
        ir = config_to_ir(AutoConfig.for_model(mt))
        assert not any(l.attention.qk_norm for l in ir.layers), mt


_SHARED_ATTENTION_TOWER = '''
import torch
from torch import nn


class SharedAttention(nn.Module):
    """diffusers-Attention-shaped: lane norms gated on an __init__ PARAM."""

    def __init__(self, dim, heads=8, qk_norm=None):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "rms_norm":
            self.norm_q = nn.RMSNorm(dim)
            self.norm_k = nn.RMSNorm(dim)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
        return torch.matmul(torch.matmul(q, k.transpose(-1, -2)).softmax(-1), v)


class NormedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SharedAttention(dim, qk_norm="rms_norm")
        self.ff = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.ff(self.attn(self.norm1(x)))


class PlainBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SharedAttention(dim)
        self.ff = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.ff(self.attn(self.norm1(x)))
'''


def test_tower_lane_norms_read_the_construction_site_not_field_presence(tmp_path):
    """The HunyuanVideo-refiner fabrication: a SHARED attention class carries
    ``norm_q``/``norm_k`` fields for every caller — the per-lane tower fact
    must come from the value THIS block passes at its construction site
    (omitted kwarg ⇒ the class's own None default ⇒ no norm), not from bare
    field presence."""
    f = tmp_path / "modeling_shared_tower.py"
    f.write_text(_SHARED_ATTENTION_TOWER)
    from model_unfolder.evidence.transitive import build_registry
    from model_unfolder.everchanging import load_conformance_transitive
    from model_unfolder.evidence.vision import layer_facts_from_block

    registry = build_registry([str(f)])
    vocab = load_conformance_transitive()
    normed = layer_facts_from_block("NormedBlock", registry, vocab)
    plain = layer_facts_from_block("PlainBlock", registry, vocab)
    assert normed["q_norm"] is True and normed["k_norm"] is True
    assert plain["q_norm"] is False and plain["k_norm"] is False


# ---------------------------------------------------------------------------
# Partial rotary — surfaced from every dialect, incl. code-only (S1b)
# ---------------------------------------------------------------------------

_CHATGLM_ROTARY_INIT = '''
import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x


class XModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        rotary_dim = (
            config.hidden_size // config.num_attention_heads
            if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=True)

    def forward(self, x):
        return self.rotary_pos_emb(x)
'''

_CONFIG_DRIVEN_ROTARY_INIT = '''
from torch import nn


class YRotaryEmbedding(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x


class YModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rotary_emb = YRotaryEmbedding(config=config)

    def forward(self, x):
        return self.rotary_emb(x)
'''


def test_code_rope_dim_evaluates_the_constructor_arithmetic(tmp_path):
    """ChatGLM shape: ``RotaryEmbedding(rotary_dim // 2)`` with the kv_channels
    ternary — the halving exists nowhere in config; the evaluator reads it
    from the code's own expression."""
    from model_unfolder.evidence.patterns import decoder_rope_dim_from_files
    f = tmp_path / "modeling_x.py"
    f.write_text(_CHATGLM_ROTARY_INIT)
    cfg = {"hidden_size": 4096, "num_attention_heads": 32, "kv_channels": 128}
    assert decoder_rope_dim_from_files((str(f),), cfg=cfg) == 64
    cfg_no_kv = {"hidden_size": 4096, "num_attention_heads": 32, "kv_channels": None}
    assert decoder_rope_dim_from_files((str(f),), cfg=cfg_no_kv) == 64


def test_code_rope_dim_ignores_config_driven_rotary(tmp_path):
    """Modern classes pass ``config=config`` — no explicit dim argument, so
    the code channel stays silent (the fraction is config-declared there)."""
    from model_unfolder.evidence.patterns import decoder_rope_dim_from_files
    f = tmp_path / "modeling_y.py"
    f.write_text(_CONFIG_DRIVEN_ROTARY_INIT)
    assert decoder_rope_dim_from_files((str(f),), cfg={"hidden_size": 64}) is None


def test_partial_rotary_surfaces_in_drill_and_chips():
    """StableLM (partial_rotary_factor=0.25): the RoPE op states the real
    rot/pass split and the Partial RoPE chip appears; a full-rotary model
    stays chip-free."""
    from transformers import AutoConfig
    import model_unfolder as mu
    doc = mu.unfold(AutoConfig.for_model("stablelm"))
    html = doc.to_html()
    assert "rot 20 · pass 60 dims" in html          # 0.25 × 80
    assert "Partial RoPE" in html
    full = mu.unfold(AutoConfig.for_model("llama")).to_html()
    assert "Partial RoPE" not in full and "rot " not in full


def test_nested_rope_parameters_dialect_carries_the_fraction():
    """GPT-NeoX modern dialect: partial_rotary_factor nested INSIDE
    rope_scaling/rope_parameters (the legacy top-level rotary_pct no longer
    exists on the config class)."""
    import model_unfolder as mu
    cfg = {
        "model_type": "gpt_neox", "hidden_size": 96 * 8, "num_attention_heads": 8,
        "num_hidden_layers": 2, "vocab_size": 1000,
        "rope_scaling": {"rope_type": "default", "partial_rotary_factor": 0.25},
    }
    ir = mu.config_to_ir(cfg)
    assert ir.layers[0].attention.rope_dim == 24


def test_qk_norm_draws_real_ops_in_the_drill():
    """Her Eyes' lawfulness line: the drill may not omit an op its parent
    label advertises — a QK-norm model's attention drill draws Q Norm/K Norm
    on the lanes (projection → norm → RoPE); MLA latent norms never do."""
    from transformers import AutoConfig
    import model_unfolder as mu
    html = mu.unfold(AutoConfig.for_model("qwen3")).to_html()
    assert html.count("Q Norm") >= 1 and html.count("K Norm") >= 1
    mla = mu.unfold(AutoConfig.for_model("deepseek_v3")).to_html()
    assert mla.count("Q Norm") == 0 and mla.count("K Norm") == 0


# ---------------------------------------------------------------------------
# Code-derived FFN intermediate width (GPT-J/GPT-2/CodeGen n_inner=None -> 4*hidden)
# ---------------------------------------------------------------------------

_GPTJ_SHAPED_INNER = '''
import torch
from torch import nn


class XMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, config.n_embd)

    def forward(self, x):
        return self.fc_out(torch.nn.functional.gelu(self.fc_in(x)))


class XAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states, layer_past=None, use_cache=None):
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        return self.out_proj(v)


class XBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = XAttention(config)
        self.mlp = XMLP(inner_dim, config)

    def forward(self, hidden_states, layer_past=None, use_cache=None):
        return hidden_states + self.mlp(self.attn(self.ln_1(hidden_states)))
'''


def test_intermediate_size_from_the_constructor_default_expression(tmp_path):
    """GPT-J shape: ``inner_dim = config.n_inner if config.n_inner is not None
    else 4 * config.n_embd`` — with n_inner absent, the FFN width is read from
    the code's own default expression (4×hidden), never a per-model table."""
    from model_unfolder.evidence.patterns import decoder_intermediate_size_from_files
    f = tmp_path / "modeling_x.py"
    f.write_text(_GPTJ_SHAPED_INNER)
    files = (str(f),)
    aliases = ("intermediate_size", "n_inner", "d_ff", "ffn_hidden_size")
    # n_inner absent -> 4 * n_embd
    assert decoder_intermediate_size_from_files(files, {"n_embd": 4096}, aliases) == 16384
    # n_inner present -> the ternary yields it (code default not applied)
    assert decoder_intermediate_size_from_files(
        files, {"n_embd": 4096, "n_inner": 9000}, aliases) == 9000


def test_intermediate_size_reader_ignores_a_sibling_rope_ternary(tmp_path):
    """The reader is keyed on the intermediate_size vocabulary, so a different
    config-default ternary in the same __init__ is not mistaken for the FFN
    width."""
    src = _GPTJ_SHAPED_INNER.replace(
        "        self.ln_1 = nn.LayerNorm(config.n_embd)",
        "        rot = config.rotary_dim if config.rotary_dim is not None else 64\n"
        "        self.ln_1 = nn.LayerNorm(config.n_embd)")
    f = tmp_path / "modeling_y.py"
    f.write_text(src)
    from model_unfolder.evidence.patterns import decoder_intermediate_size_from_files
    aliases = ("intermediate_size", "n_inner")
    # rotary_dim ternary must NOT be picked; n_inner default (4*n_embd) wins
    assert decoder_intermediate_size_from_files(
        (str(f),), {"n_embd": 1024, "rotary_dim": None}, aliases) == 4096


def test_gptj_family_derives_the_ffn_width_end_to_end():
    """GPT-J/CodeGen/GPT-2 carry ``n_inner=None`` and compute 4×hidden — the
    parse must surface the real FFN width, not undercount it to zero."""
    from transformers import AutoConfig
    for mt, embd_field in (("gptj", "n_embd"), ("codegen", "n_embd"), ("gpt2", "n_embd")):
        cfg = AutoConfig.for_model(mt)
        cfg.n_inner = None
        ir = config_to_ir(cfg)
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, embd_field)
        assert ir.layers[0].ffn.intermediate_size == 4 * hidden, mt


def test_declared_intermediate_size_is_never_overridden_by_code():
    """The code reader fires ONLY when the config field is absent — a declared
    intermediate_size (Llama) is authoritative and untouched."""
    from transformers import AutoConfig
    cfg = AutoConfig.for_model("llama")
    ir = config_to_ir(cfg)
    assert ir.layers[0].ffn.intermediate_size == cfg.intermediate_size


# ---------------------------------------------------------------------------
# Norm-kind: code MATH outranks BOTH eps spellings (PhiMoE/Persimmon)
# ---------------------------------------------------------------------------

def test_norm_kind_math_outranks_the_rms_eps_spelling():
    """PhiMoE constructs ``nn.LayerNorm`` while carrying ``rms_norm_eps`` — the
    RMS eps spelling lies, the code math (torch-builtin LayerNorm) tells the
    truth.  T5 exposes rival encoder/decoder stages, so its epsilon spelling
    cannot stand in for an exact primitive and the diagram stays generic.
    Every unambiguous plain RMS/LN control is unchanged."""
    from transformers import AutoConfig
    expect = {"phimoe": "LayerNorm", "t5": "Normalization", "llama": "RMSNorm",
              "bloom": "LayerNorm", "gemma2": "RMSNorm", "qwen3": "RMSNorm"}
    for mt, want in expect.items():
        ir = config_to_ir(AutoConfig.for_model(mt))
        drawn = {b.get("label") for l in ir.layers for b in (l.blocks or [])
                 if isinstance(b, dict) and b.get("kind") == "norm"}
        assert drawn == {want}, f"{mt}: drew {drawn}, expected {want}"


def test_t5_two_stage_norm_stays_unknown_until_stage_selection_is_proven():
    """T5Model delegates to both encoder and decoder stacks.

    The exact address rail preserves both rivals; an epsilon field spelling
    must not pick a primitive or pretend one stage was selected.
    """
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.decoder_norm import decoder_norm_kind_for_path
    cfg = AutoConfig.for_model("t5")
    context = ParseContext.build(cfg)
    result = decoder_norm_kind_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "ambiguous"
    ir = config_to_ir(cfg, parse_context=context)
    assert {layer.norm_kind for layer in ir.layers} == {"unknown"}


# ---------------------------------------------------------------------------
# Raw-JSON rung: config-class default hydration (identity-as-address at load)
# ---------------------------------------------------------------------------

def test_raw_json_hydration_fills_class_defaults_and_preserves_raw_and_stamps():
    from model_unfolder.parser import _hydrate_config_class_defaults as H
    raw = {"model_type": "gemma2", "num_hidden_layers": 4, "hidden_size": 256,
           "num_attention_heads": 8, "_repo_id": "stamp/keep"}
    h = H(raw)
    assert h.get("query_pre_attn_scalar") is not None      # class default materialized
    assert h["num_hidden_layers"] == 4                      # raw wins over any default
    assert h["_repo_id"] == "stamp/keep"                    # loader stamp survives
    assert h["model_type"] == "gemma2"


def test_raw_json_hydration_raw_value_overrides_a_class_default():
    from model_unfolder.parser import _hydrate_config_class_defaults as H
    over = {"model_type": "gemma2", "sliding_window": 999, "num_hidden_layers": 2,
            "hidden_size": 256, "num_attention_heads": 8}
    assert H(over)["sliding_window"] == 999


def test_raw_json_hydration_is_a_noop_for_unknown_or_typeless_configs():
    from model_unfolder.parser import _hydrate_config_class_defaults as H
    unk = {"model_type": "totally_unknown_xyz", "hidden_size": 128}
    assert H(unk) == unk
    assert H({"hidden_size": 128}) == {"hidden_size": 128}
    assert H("not a dict") == "not a dict"


# ---------------------------------------------------------------------------
# MoE router facts from code (GLM-4.5 sigmoid+bias; Phi sparsemixer)
# ---------------------------------------------------------------------------

_DSV3_SHAPED_ROUTER = '''
import torch
from torch import nn
import torch.nn.functional as F


class XTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = nn.Parameter(torch.empty((config.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((config.n_routed_experts)))

    def forward(self, hidden_states):
        return hidden_states


class XMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = XTopkRouter(config)
        self.experts = nn.ModuleList()

    def route_tokens_to_experts(self, hidden_states):
        router_logits = F.linear(hidden_states.type(torch.float32), self.gate.weight)
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = router_logits_for_choice.view(-1, self.gate.n_group, 4).topk(2, dim=-1)[0].sum(-1)
        group_idx = torch.topk(group_scores, k=self.gate.topk_group, dim=-1)[1]
        group_mask = torch.zeros_like(group_scores)
        topk_indices = torch.topk(router_logits_for_choice, k=self.gate.top_k, dim=-1)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        return hidden_states
'''

_MIXTRAL_SHAPED_ROUTER = '''
import torch
from torch import nn


class XSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        return routing_weights
'''

_SPARSEMIXER_SHAPED_ROUTER = '''
import torch
from torch import nn


def sparsemixer(scores, top_k, jitter_eps):
    masked_scores = scores.masked_fill(scores < 0, float("-inf"))
    masked_scores = torch.softmax(masked_scores, dim=-1)
    selected = torch.topk(masked_scores, top_k, dim=-1)[1]
    weights = masked_scores.gather(dim=-1, index=selected)
    return weights, selected


class XPhiMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = sparsemixer(router_logits, self.top_k, jitter_eps=0.01)
        return routing_weights
'''

_GPTOSS_SHAPED_ROUTER = '''
import torch
from torch import nn
import torch.nn.functional as F


class XExperts(nn.Module):
    """Expert compute uses a sigmoid GLU — must NOT be read as router scoring."""
    def __init__(self, config):
        super().__init__()
        self.alpha = 1.702
        self.gate_up_proj = nn.Parameter(torch.empty(config.num_local_experts, config.hidden_size, 2))

    def forward(self, hidden_states, routing_weights):
        gate = hidden_states
        glu = gate * torch.sigmoid(gate * self.alpha)
        return glu * routing_weights


class XTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(torch.empty(config.num_local_experts, config.hidden_size))

    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_scores = torch.nn.functional.softmax(router_top_value, dim=1)
        return router_scores, router_indices
'''


def _router_ev(tmp_path, src):
    from model_unfolder.evidence.patterns import decoder_router_evidence_from_files
    import hashlib
    # unique filename per distinct source: _parse_defs is lru_cached on path, so
    # two calls to the same path in one test would reuse the first parse.
    f = tmp_path / f"modeling_{hashlib.md5(src.encode()).hexdigest()[:8]}.py"
    f.write_text(src)
    return decoder_router_evidence_from_files((str(f),))


def test_router_sigmoid_and_aux_free_bias_from_split_router_and_block(tmp_path):
    """DeepSeek-V3/GLM-4.5 shape: the bias buffer lives in the router class, the
    sigmoid + group + gather in the MoE block's route_tokens_to_experts — the
    reader scans the UNION and reports sigmoid + bias + grouped."""
    ev = _router_ev(tmp_path, _DSV3_SHAPED_ROUTER)
    assert ev is not None
    assert ev.scoring_fn == "sigmoid" and ev.bias_correction and ev.grouped
    assert not ev.sparsemixer


def test_router_plain_softmax_topk(tmp_path):
    ev = _router_ev(tmp_path, _MIXTRAL_SHAPED_ROUTER)
    assert ev.scoring_fn == "softmax"
    assert not ev.bias_correction and not ev.grouped and not ev.sparsemixer


def test_router_sparsemixer_followed_into_the_free_function(tmp_path):
    """Phi shape: routing delegates to a module-level ``sparsemixer`` — the
    reader follows it one hop, reports sparsemixer + its softmax scoring."""
    ev = _router_ev(tmp_path, _SPARSEMIXER_SHAPED_ROUTER)
    assert ev.sparsemixer and ev.scoring_fn == "softmax"
    assert not ev.bias_correction


def test_router_ignores_expert_activation_sigmoid(tmp_path):
    """gpt-oss shape: the EXPERT GLU uses ``torch.sigmoid(gate * alpha)`` — an
    activation, not routing.  The score-transform detector keys on routing-logit
    NAMES, so scoring resolves to the router's softmax, never the expert sigmoid."""
    ev = _router_ev(tmp_path, _GPTOSS_SHAPED_ROUTER)
    assert ev.scoring_fn == "softmax", f"expert sigmoid leaked: {ev}"
    assert not ev.bias_correction and not ev.sparsemixer


def test_router_none_for_dense_model(tmp_path):
    from model_unfolder.evidence.patterns import decoder_router_evidence_from_files
    f = tmp_path / "modeling_dense.py"
    f.write_text("import torch\nfrom torch import nn\n\nclass XMLP(nn.Module):\n"
                 "    def __init__(self, config):\n        super().__init__()\n"
                 "        self.fc = nn.Linear(4, 4)\n    def forward(self, x):\n        return self.fc(x)\n")
    assert decoder_router_evidence_from_files((str(f),)) is None


def test_glm45_router_draws_sigmoid_bias_and_gather_from_code():
    """The headline S2 fix: GLM-4.5's config lacks scoring_func/topk_method but
    its code enacts DeepSeek-V3 routing — the drawn router must be sigmoid with
    the aux-loss-free bias and the raw-weight gather, not a plain softmax."""
    from transformers import AutoConfig
    import model_unfolder as mu
    html = mu.unfold(AutoConfig.for_model("glm4_moe")).to_html()
    assert "sigmoid" in html
    assert "load-balancing" in html          # aux-loss-free bias card
    assert "Gather weights" in html          # raw-weight gather step
    assert "softmax gating" not in html


def test_deepseek_v3_declared_and_code_agree_no_drift():
    """DSV3 declares scoring_func='sigmoid'+topk_method='noaux_tc' AND enacts
    them — code agrees with config, so no disagreement note is recorded."""
    import json
    from model_unfolder.sable import DEFAULT_CORPUS
    import model_unfolder as mu
    cfg = json.loads((DEFAULT_CORPUS / "deepseek-v3.json").read_text())["config"]
    routing = None
    for l in mu.config_to_ir(cfg).layers:
        if l.ffn.routing:
            routing = l.ffn.routing
    assert routing["scoring_func"] == "sigmoid" and routing.get("bias_correction")
    assert "_scoring_declared" not in routing      # config and code agree


def test_router_scoring_position_before_vs_after_topk(tmp_path):
    """The score transform's POSITION relative to top-k is a code fact: DSV3/GLM
    (softmax/sigmoid the full logits, THEN select) score before; gpt-oss/Granite
    (top-k the raw logits, THEN softmax the winners) score after.  Drives whether
    a scoring node is drawn before top-k — a node before top-k would misdraw
    gpt-oss."""
    before_shape = _DSV3_SHAPED_ROUTER                 # sigmoid() ... then torch.topk
    after_shape = _GPTOSS_SHAPED_ROUTER                # torch.topk ... then softmax
    assert _router_ev(tmp_path, before_shape).scoring_before_topk is True
    assert _router_ev(tmp_path, after_shape).scoring_before_topk is False


def test_router_ignores_framework_container_aux_loss_softmax(tmp_path):
    """A ``*ForCausalLM`` wrapper that softmaxes router logits for the load-balance
    STAT (output_router_logits) must not be read as the selection scoring — the
    container is excluded, so a top-k-then-softmax block still resolves to
    scoring-after-topk, not an ambiguous None."""
    src = _GPTOSS_SHAPED_ROUTER + '''

class XForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = XTopKRouter(config)

    def forward(self, hidden_states):
        router_logits = hidden_states
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        return routing_weights
'''
    ev = _router_ev(tmp_path, src)
    assert ev.scoring_fn == "softmax" and ev.scoring_before_topk is False


def test_glm45_draws_the_sigmoid_scoring_node_before_topk():
    """Her Eyes lawfulness fix: GLM-4.5's router draws the sigmoid score
    transform as its OWN node between Linear (Gate) and Top-k, so the drill's
    'expert scores' has a visible origin."""
    from transformers import AutoConfig
    import model_unfolder as mu
    html = mu.unfold(AutoConfig.for_model("glm4_moe")).to_html()
    assert ">sigmoid<" in html                # a drawn node, not only a chip


# ---------------------------------------------------------------------------
# MoE-vs-dense layer SCHEDULE from construction evidence (code-authoritative)
# ---------------------------------------------------------------------------

_MOE_SCAFFOLD = '''
import torch
from torch import nn


class XExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList()
        self.gate = nn.Linear(config.hidden_size, config.num_experts)

    def forward(self, x):
        return x


class XMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = XExperts(config)

    def forward(self, x):
        return self.experts(x)


class XMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x))


class XAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, past_key_values=None):
        return self.o_proj(self.v_proj(hidden_states))


class XDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = XAttention(config, layer_idx)
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
{ffn_ctor}

    def forward(self, hidden_states, past_key_values=None):
        return hidden_states + self.mlp(self.self_attn(self.input_layernorm(hidden_states)))
'''

# Gate forms — each substitutes the FFN-field construction block:
_CTOR_UNCONDITIONAL = "        self.mlp = XMoE(config)"
_CTOR_THRESHOLD = (
    "        if layer_idx >= config.first_k_dense_replace:\n"
    "            self.mlp = XMoE(config)\n"
    "        else:\n"
    "            self.mlp = XMLP(config)")
_CTOR_MEMBERSHIP_SELFFLAG = (
    "        self.is_moe_layer = layer_idx in config.moe_layers\n"
    "        if self.is_moe_layer:\n"
    "            self.mlp = XMoE(config)\n"
    "        else:\n"
    "            self.mlp = XMLP(config)")
_CTOR_EXCLUSION_AND = (
    "        if (layer_idx not in config.mlp_only_layers) and (config.num_experts > 0):\n"
    "            self.mlp = XMoE(config)\n"
    "        else:\n"
    "            self.mlp = XMLP(config)")
_CTOR_MODULO_AND = (
    "        if ((layer_idx + 1) % config.moe_layer_interval == 0) and (layer_idx >= config.moe_layer_start_index):\n"
    "            self.mlp = XMoE(config)\n"
    "        else:\n"
    "            self.mlp = XMLP(config)")
_CTOR_TERNARY = (
    "        self.mlp = XMoE(config) if layer_idx >= config.first_k_dense_replace else XMLP(config)")
_CTOR_DENSE = "        self.mlp = XMLP(config)"
# a MoE class MISLEADINGLY named MLP (gpt-oss shape) — must still read MoE
_CTOR_MISNAMED = "        self.mlp = XExperts(config)"   # XExperts builds experts → MoE despite call
# hybrid: two ffn fields (shared + moe) → ambiguous → None
_CTOR_MULTI = (
    "        self.shared_mlp = XMLP(config)\n"
    "        self.mlp = XMoE(config)")


def _sched(tmp_path, ctor, cfg):
    from model_unfolder.evidence.patterns import decoder_moe_schedule_from_files
    import hashlib
    f = tmp_path / f"modeling_{hashlib.md5(ctor.encode()).hexdigest()[:8]}.py"
    f.write_text(_MOE_SCAFFOLD.format(ffn_ctor=ctor))
    return decoder_moe_schedule_from_files((str(f),), cfg)


def test_moe_schedule_unconditional_all_moe(tmp_path):
    assert _sched(tmp_path, _CTOR_UNCONDITIONAL, {"num_hidden_layers": 4}) == [True]*4


def test_moe_schedule_threshold_dense_prefix(tmp_path):
    cfg = {"num_hidden_layers": 5, "first_k_dense_replace": 2}
    assert _sched(tmp_path, _CTOR_THRESHOLD, cfg) == [False, False, True, True, True]


def test_moe_schedule_membership_via_self_flag(tmp_path):
    """Llama-4 shape: self.is_moe_layer = layer_idx in config.moe_layers."""
    cfg = {"num_hidden_layers": 4, "moe_layers": [1, 3]}
    assert _sched(tmp_path, _CTOR_MEMBERSHIP_SELFFLAG, cfg) == [False, True, False, True]


def test_moe_schedule_exclusion_and_threshold(tmp_path):
    """Qwen shape: (layer_idx not in mlp_only_layers) and (num_experts > 0)."""
    cfg = {"num_hidden_layers": 4, "mlp_only_layers": [2], "num_experts": 8}
    assert _sched(tmp_path, _CTOR_EXCLUSION_AND, cfg) == [True, True, False, True]
    cfg0 = {"num_hidden_layers": 3, "mlp_only_layers": [], "num_experts": 0}
    assert _sched(tmp_path, _CTOR_EXCLUSION_AND, cfg0) == [False, False, False]


def test_moe_schedule_modulo_and_threshold(tmp_path):
    """Ernie shape: (layer_idx+1) % interval == 0 and layer_idx >= start."""
    cfg = {"num_hidden_layers": 6, "moe_layer_interval": 2, "moe_layer_start_index": 1}
    # (i+1)%2==0 → i in {1,3,5}; AND i>=1 → all of them
    assert _sched(tmp_path, _CTOR_MODULO_AND, cfg) == [False, True, False, True, False, True]


def test_moe_schedule_ternary(tmp_path):
    cfg = {"num_hidden_layers": 4, "first_k_dense_replace": 1}
    assert _sched(tmp_path, _CTOR_TERNARY, cfg) == [False, True, True, True]


def test_moe_schedule_dense_model_all_false(tmp_path):
    assert _sched(tmp_path, _CTOR_DENSE, {"num_hidden_layers": 3}) == [False]*3


def test_moe_schedule_detects_misnamed_moe_class(tmp_path):
    """gpt-oss shape: the MoE class is named like an MLP but builds experts —
    structural (name-independent) detection must still resolve MoE."""
    assert _sched(tmp_path, _CTOR_MISNAMED, {"num_hidden_layers": 2}) == [True]*2


def test_moe_schedule_multiple_ffn_fields_returns_none(tmp_path):
    """Ambiguous (shared_mlp + moe — hybrid shape) → None → config fallback."""
    assert _sched(tmp_path, _CTOR_MULTI, {"num_hidden_layers": 4}) is None


def test_moe_schedule_unresolvable_gate_returns_none(tmp_path):
    """A gate referencing an absent config field → None (never a wrong guess)."""
    cfg = {"num_hidden_layers": 4}   # first_k_dense_replace missing
    assert _sched(tmp_path, _CTOR_THRESHOLD, cfg) is None


def test_llama4_moe_schedule_now_drawn_moe_end_to_end():
    """The headline fix: Llama-4's MoE was drawn all-dense (moe_layers unread +
    interleave==1 inversion); the code schedule now draws it MoE."""
    from transformers import AutoConfig
    import model_unfolder as mu
    ir = mu.config_to_ir(AutoConfig.for_model("llama4_text"))
    moe = sum(1 for l in ir.layers if l.ffn.kind == "moe")
    assert moe == len(ir.layers) and moe > 0, f"only {moe}/{len(ir.layers)} MoE"


def test_ernie_moe_schedule_first_layer_dense_from_code():
    """The bug the sweep FOUND: Ernie's config path drew all-MoE; the code gate
    ((i+1)%interval==0 and i>=start=1) makes layer 0 dense."""
    from transformers import AutoConfig
    import model_unfolder as mu
    ir = mu.config_to_ir(AutoConfig.for_model("ernie4_5_moe"))
    kinds = [l.ffn.kind for l in ir.layers]
    assert kinds[0] == "dense" and kinds[1] == "moe"


def test_moe_schedule_working_families_unchanged():
    """The 12 agreeing families must stay MoE exactly as before (byte-stable):
    DeepSeek dense-prefix, Mixtral/Qwen3/gpt-oss all-MoE."""
    from transformers import AutoConfig
    import model_unfolder as mu
    ds = [l.ffn.kind for l in mu.config_to_ir(AutoConfig.for_model("deepseek_v3")).layers]
    assert ds[:3] == ["dense"]*3 and all(k == "moe" for k in ds[3:])
    for mt in ("mixtral", "qwen3_moe", "gpt_oss"):
        ir = mu.config_to_ir(AutoConfig.for_model(mt))
        assert all(l.ffn.kind == "moe" for l in ir.layers), mt


def test_moe_schedule_hybrid_returns_none_falls_back():
    """granitemoehybrid (Mamba-MoE) → code=None → config path (no crash)."""
    from transformers import AutoConfig
    import model_unfolder as mu
    ir = mu.config_to_ir(AutoConfig.for_model("granitemoehybrid"))  # must not raise
    assert ir.layers


# ---------------------------------------------------------------------------
# Per-layer SCHEDULE conformance lock (Group 1): the drawn per-layer type
# schedule must match the code's AUTHORITATIVE per-layer list — the net that
# would have caught the MoE interleave==1 inversion, locking sliding/NoPE/MoE
# against future re-derivation drift.
# ---------------------------------------------------------------------------

def test_sliding_schedule_matches_code_layer_types():
    """Every model declaring ``layer_types`` (the list the attention class reads
    as ``config.layer_types[layer_idx] == 'sliding_attention'``) must have its
    drawn per-layer sliding schedule EXACTLY match that list."""
    from transformers import AutoConfig
    import model_unfolder as mu
    for mt in ("gemma2", "gemma3_text", "cohere2", "gpt_oss", "qwen2", "qwen3"):
        cfg = AutoConfig.for_model(mt)
        lt = getattr(cfg, "layer_types", None)
        if not lt:
            continue
        drawn = [("slid" in str(l.attention.mask).lower())
                 for l in mu.config_to_ir(cfg).layers]
        code = [(x == "sliding_attention") for x in lt][:len(drawn)]
        assert drawn == code, f"{mt}: sliding schedule diverged from code layer_types"


def test_nope_schedule_matches_code_no_rope_layers():
    """A model with a ``no_rope_layers`` list (Llama-4 iRoPE) must draw NoPE on
    exactly the layers the code marks (``no_rope_layers[i]`` truthy = uses rope)."""
    from transformers import AutoConfig
    import model_unfolder as mu
    for mt in ("llama4_text",):
        cfg = AutoConfig.for_model(mt)
        nrl = getattr(cfg, "no_rope_layers", None)
        if not isinstance(nrl, (list, tuple)):
            continue
        drawn = [bool(l.attention.no_rope) for l in mu.config_to_ir(cfg).layers]
        code = [not bool(x) for x in nrl][:len(drawn)]
        assert drawn == code, f"{mt}: NoPE schedule diverged from code no_rope_layers"


def test_moe_schedule_matches_code_construction():
    """The MoE schedule (now code-authoritative) must match the code's per-layer
    experts-class construction — the exact regression the interleave==1 bug was."""
    from transformers import AutoConfig
    import model_unfolder as mu
    from model_unfolder.evidence.patterns import decoder_moe_schedule_from_files
    import transformers, pathlib
    base = pathlib.Path(transformers.__file__).parent / "models"
    for mt, ff in (("llama4_text", "llama4/modeling_llama4.py"),
                   ("deepseek_v3", "deepseek_v3/modeling_deepseek_v3.py"),
                   ("ernie4_5_moe", "ernie4_5_moe/modeling_ernie4_5_moe.py")):
        cfg = AutoConfig.for_model(mt)
        code = decoder_moe_schedule_from_files((str(base / ff),), cfg)
        if code is None:
            continue
        drawn = [(l.ffn.kind == "moe") for l in mu.config_to_ir(cfg).layers]
        assert drawn == code[:len(drawn)], f"{mt}: drawn MoE schedule != code construction"


def test_parallel_norm_count_from_construction():
    """Parallel-residual input-norm count from the code dataflow: GPT-J shares
    one norm (1 — the pinned negative control), GPT-NeoX applies two separate
    norms (2 — the fix); Falcon's conditional 4-field case → None (fallback)."""
    import transformers, pathlib
    from model_unfolder.evidence.patterns import decoder_parallel_norm_count_from_files
    base = pathlib.Path(transformers.__file__).parent / "models"
    for mt, ff, want in (("gptj", "gptj/modeling_gptj.py", 1),
                         ("gpt_neox", "gpt_neox/modeling_gpt_neox.py", 2),
                         ("falcon", "falcon/modeling_falcon.py", None)):
        p = base / ff
        if not p.exists():
            continue
        assert decoder_parallel_norm_count_from_files((str(p),)) == want, mt


def test_diffusion_rope_component_scoping():
    """The Sana leak, pinned: the DiT's own source says NO rotary; the pipeline
    UNION (DiT + Gemma-2 text encoder) says rotary — because Gemma's markers
    ride in.  Code readers must therefore consume ROOT-scoped files only."""
    import diffusers, transformers, pathlib
    from model_unfolder.evidence.patterns import diffusion_rope_from_files
    sana = pathlib.Path(diffusers.__file__).parent / "models" / "transformers" / "sana_transformer.py"
    gemma = pathlib.Path(transformers.__file__).parent / "models" / "gemma2" / "modeling_gemma2.py"
    if not (sana.exists() and gemma.exists()):
        return
    assert diffusion_rope_from_files((str(sana),)) is False
    assert diffusion_rope_from_files((str(sana), str(gemma))) is True  # the leak shape


def test_diffusor_source_files_root_scoped():
    """_source_files returns the ROOT component's files, never the pipeline
    union; bundles without a component map keep the flat files (fallback)."""
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.adapters.diffusor.parser import _source_files

    class _Ctx:
        def __init__(self, bundle):
            self.source_bundle = bundle

    scoped = SourceBundle(
        source="local", files=("dit.py", "enc.py"),
        component_files={"root": ("dit.py",), "text_encoder": ("enc.py",)})
    assert _source_files({}, _Ctx(scoped)) == ("dit.py",)

    flat = SourceBundle(source="local", files=("dit.py", "enc.py"))
    assert _source_files({}, _Ctx(flat)) == ("dit.py", "enc.py")


def test_intermediate_size_follows_into_ffn_class():
    """Width-default idioms BELOW the layer init: the ternary or the inline
    widened-Linear in the FFN-role class's own __init__ (BLOOM's
    ``nn.Linear(hidden, 4*hidden)``); layer-init ternary (GPT-J) unchanged;
    ambiguity (two distinct widened values) → None, never a guess."""
    import transformers, pathlib
    from transformers import AutoConfig
    from model_unfolder.evidence.patterns import decoder_intermediate_size_from_files
    base = pathlib.Path(transformers.__file__).parent / "models"
    cfg = AutoConfig.for_model("bloom"); cfg.hidden_size = 14336
    assert decoder_intermediate_size_from_files(
        (str(base / "bloom/modeling_bloom.py"),), cfg,
        ("intermediate_size", "n_inner", "ffn_dim")) == 4 * 14336
    cfgj = AutoConfig.for_model("gptj")
    assert decoder_intermediate_size_from_files(
        (str(base / "gptj/modeling_gptj.py"),), cfgj,
        ("intermediate_size", "n_inner")) == 4 * cfgj.n_embd


def test_intermediate_size_ambiguous_widths_refuse(tmp_path):
    """Two DIFFERENT widened Linears in one FFN init → None (tri-state law)."""
    import hashlib
    from transformers import AutoConfig
    from model_unfolder.evidence.patterns import decoder_intermediate_size_from_files
    src = '''
import torch.nn as nn
class OddMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.a = nn.Linear(hidden_size, 4 * hidden_size)
        self.b = nn.Linear(hidden_size, 8 * hidden_size)
        self.down = nn.Linear(4 * hidden_size, hidden_size)
    def forward(self, x):
        return self.down(self.a(x) * self.b(x))
class OddAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
    def forward(self, x, past_key_value=None):
        return x
class OddBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = OddAttention(config)
        self.mlp = OddMLP(config)
    def forward(self, x, past_key_value=None):
        return self.mlp(self.self_attn(x, past_key_value))
'''
    f = tmp_path / f"modeling_{hashlib.md5(src.encode()).hexdigest()[:8]}.py"
    f.write_text(src)
    cfg = AutoConfig.for_model("bloom"); cfg.hidden_size = 128
    assert decoder_intermediate_size_from_files(
        (str(f),), cfg, ("intermediate_size",)) is None


def test_scores_scaling_wired_to_main_paths():
    """B1: the unscaled-scores verdict reaches the MAIN paths' spec + all three
    projections — reader witnesses (T5 raw QK^T → False, Llama SDPA → True),
    wrapper root-scoping, spec emission only-when-False (byte-stability)."""
    import transformers, pathlib
    from model_unfolder.evidence.patterns import attention_score_scaling_from_files
    from model_unfolder.evidence.models import SourceBundle

    base = pathlib.Path(transformers.__file__).parent / "models"
    t5 = base / "t5" / "modeling_t5.py"
    llama = base / "llama" / "modeling_llama.py"
    assert attention_score_scaling_from_files((str(t5),)) is False
    assert attention_score_scaling_from_files((str(llama),)) is True

    class _Ctx:
        def __init__(self, bundle):
            self.source_bundle = bundle

    from model_unfolder.adapters.transformer.parser import (
        _code_scores_scaled as t_scaled)
    from model_unfolder.adapters.diffusor.parser import (
        _code_scores_scaled as d_scaled)
    t5_bundle = SourceBundle(source="local", files=(str(t5),))
    assert t_scaled({}, _Ctx(t5_bundle)) is False
    # Diffusor wrapper is ROOT-scoped: an unscaled encoder in the union must
    # not flip the denoiser's verdict (the A1 scoping discipline).
    mixed = SourceBundle(
        source="local", files=(str(llama), str(t5)),
        component_files={"root": (str(llama),), "text_encoder": (str(t5),)})
    assert d_scaled({}, _Ctx(mixed)) is True


def test_scores_scaled_projections_only_when_false():
    """The spec field lands in BOTH dict projections ONLY when False — a
    scaled/unknown model's output stays byte-identical (the scores_scale rule)."""
    from model_unfolder.ir import AttentionSpec, _attention_to_dict
    from model_unfolder.adapters.transformer.blocks.attention import attention_detail

    base = AttentionSpec(kind="mha", num_heads=8, num_kv_heads=8, head_dim=64)
    assert "scores_scaled" not in _attention_to_dict(base)
    assert "scores_scaled" not in attention_detail(base)
    proven = AttentionSpec(kind="mha", num_heads=8, num_kv_heads=8, head_dim=64,
                           scores_scaled=False)
    assert _attention_to_dict(proven)["scores_scaled"] is False
    assert attention_detail(proven)["scores_scaled"] is False
    scaled = AttentionSpec(kind="mha", num_heads=8, num_kv_heads=8, head_dim=64,
                           scores_scaled=True)
    assert "scores_scaled" not in _attention_to_dict(scaled)
    assert "scores_scaled" not in attention_detail(scaled)


def test_scores_scaled_region_drops_denominator():
    """Opgraph spine: scores_scaled=False → numerator-only Q K^T (no sqrt
    fraction); absent/True keeps the sqrt(dim) denominator."""
    from model_unfolder.opgraph import attention_region

    def _scores_meta(attn):
        region = attention_region(attn, 512)
        for op in region.ops:
            meta = getattr(op, "meta", None) or {}
            if "denominator" in meta:
                return meta
        raise AssertionError("no scores op with a formula meta found")

    plain = {"kind": "mha", "num_heads": 8, "num_kv_heads": 8, "head_dim": 64}
    assert _scores_meta(plain)["denominator"]          # sqrt(dim) kept
    unscaled = dict(plain, scores_scaled=False)
    assert _scores_meta(unscaled)["denominator"] is None
    assert _scores_meta(unscaled)["formula"] == "QK^T"


def test_cross_qk_norm_per_site_evidence(tmp_path):
    """A3: the CROSS sublayer's Q/K-norm comes from the cross class's OWN
    construction, per site — unconditional lane norms in a same-file cross
    class prove it (Wan); a shared imported class or ctor-gated lane norms
    stay None (PixArt/SD3 — only positive evidence draws the op)."""
    import pathlib, hashlib
    import diffusers
    from model_unfolder.evidence.patterns import diffusion_cross_qk_norm_from_files

    base = pathlib.Path(diffusers.__file__).parent / "models" / "transformers"
    wan = base / "transformer_wan.py"
    if wan.exists():
        assert diffusion_cross_qk_norm_from_files((str(wan),)) == "rms_norm"
    for stem in ("pixart_transformer_2d.py", "transformer_sd3.py"):
        p = base / stem
        if p.exists():
            assert diffusion_cross_qk_norm_from_files((str(p),)) is None, stem

    # ctor-GATED lane norm in a same-file cross class → None (guarded ≠ proven)
    src = '''
import torch, torch.nn as nn
class GatedAttention(nn.Module):
    def __init__(self, dim, qk_norm=None):
        super().__init__()
        if qk_norm:
            self.norm_q = nn.RMSNorm(dim)
            self.norm_k = nn.RMSNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
    def forward(self, x, encoder_hidden_states=None):
        return x
class SomeBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn1 = GatedAttention(dim)
        self.attn2 = GatedAttention(dim)
        self.ffn = nn.Linear(dim, dim)
    def forward(self, hidden_states, encoder_hidden_states):
        h = self.attn1(hidden_states)
        h = self.attn2(h, encoder_hidden_states)
        return self.ffn(h)
'''
    f = tmp_path / f"modeling_{hashlib.md5(src.encode()).hexdigest()[:8]}.py"
    f.write_text(src)
    assert diffusion_cross_qk_norm_from_files((str(f),)) is None


def test_cross_spec_qk_norm_trusted_not_suppressed():
    """The render layer trusts the spec: a cross spec whose qk_norm carries
    per-site evidence draws Q/K Norm (detail keys + cards + region ops); a
    cross spec without evidence draws none (the parse-time guard)."""
    from model_unfolder.ir import AttentionSpec
    from model_unfolder.adapters.transformer.blocks.attention import (
        attention_detail, attention_child_blocks)
    from model_unfolder.opgraph import attention_region

    proven = AttentionSpec(kind="mha", num_heads=8, num_kv_heads=8, head_dim=64,
                           cross_attention=True, cross_kv_source="encoded text prompt",
                           qk_norm=True, cached=False)
    d = attention_detail(proven)
    assert d["q_norm"] and d["k_norm"]
    ids = {c["id"] for c in attention_child_blocks(proven, 512, id_prefix="x_")}
    assert {"x_q_norm", "x_k_norm"} <= ids
    ops = {op.id for op in attention_region(d, 512).ops}
    assert {"q_norm", "k_norm"} <= ops

    unproven = AttentionSpec(kind="mha", num_heads=8, num_kv_heads=8, head_dim=64,
                             cross_attention=True, cross_kv_source="encoded text prompt",
                             qk_norm=False, cached=False)
    d2 = attention_detail(unproven)
    assert not d2["q_norm"] and not d2["k_norm"]
    ids2 = {c["id"] for c in attention_child_blocks(unproven, 512, id_prefix="x_")}
    assert "x_q_norm" not in ids2


def test_attention_sinks_spec_and_rendering_surface():
    """The exact-owner reader is pinned in test_attention_sinks; this control
    proves its positive fact remains an only-when-True rendered mechanism."""
    from model_unfolder.ir import AttentionSpec, _attention_to_dict
    from model_unfolder.adapters.transformer.blocks.attention import (
        attention_detail, attention_child_blocks)
    from model_unfolder.opgraph import attention_region

    plain = AttentionSpec(kind="gqa", num_heads=8, num_kv_heads=2, head_dim=64)
    assert "sinks" not in _attention_to_dict(plain)
    assert "sinks" not in attention_detail(plain)

    sunk = AttentionSpec(kind="gqa", num_heads=8, num_kv_heads=2, head_dim=64,
                         sinks=True)
    assert _attention_to_dict(sunk)["sinks"] is True
    d = attention_detail(sunk)
    assert d["sinks"] is True
    region = attention_region(d, 512)
    ids = {op.id for op in region.ops}
    # ONE spine box between scores and softmax — the sink logits are learned
    # PARAMETERS of the append op, never a side input node (side inputs made
    # the layout duplicate the downstream chain; U5 caught it twice)
    assert "sink_concat" in ids and "attention_sinks" not in ids
    assert any(e.src == "scaled_scores" and e.dst == "sink_concat"
               for e in region.edges)
    assert any(e.src == "sink_concat" and e.dst == "attn_softmax"
               for e in region.edges)
    assert not any(e.src == "scaled_scores" and e.dst == "attn_softmax"
                   for e in region.edges)
    card_ids = {c["id"] for c in attention_child_blocks(sunk, 512)}
    assert "sink_concat" in card_ids and "attention_sinks" not in card_ids


def test_instance_gate_pruned_by_construction_site(tmp_path):
    """A5: a SHARED block class instance-parameterized at the construction
    site — the falsy-literal site loses the gated-branch facts; truthy and
    computed sites keep them; branch-independent facts (sandwich norms whose
    second norm runs in BOTH branches) survive the prune."""
    import hashlib
    from model_unfolder.evidence.stacks import secondary_stacks_from_files
    from model_unfolder.evidence.transitive import build_registry
    from model_unfolder.evidence.vision import layer_facts_from_block
    from model_unfolder.everchanging import load_conformance_transitive

    src = '''
import torch, torch.nn as nn
class SharedAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
    def forward(self, x):
        return x
class SharedFeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
    def forward(self, x):
        return self.fc2(self.fc1(x))
class SharedBlock(nn.Module):
    def __init__(self, dim, modulation=True):
        super().__init__()
        self.modulation = modulation
        self.attn = SharedAttention(dim)
        self.feed_forward = SharedFeedForward(dim)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        if modulation:
            self.adaLN_modulation = nn.Linear(dim, 4 * dim)
    def forward(self, x, temb=None):
        if self.modulation:
            scale, gate = self.adaLN_modulation(temb).chunk(2, dim=-1)
            h = self.norm1(x)
            x = x + gate.tanh() * self.norm2(self.attn(h))
        else:
            h = self.norm1(x)
            x = x + self.norm2(self.attn(h))
        return x + self.feed_forward(x)
class SharedHost(nn.Module):
    def __init__(self, dim, num_refiner_layers=2):
        super().__init__()
        self.context_refiner = nn.ModuleList(
            [SharedBlock(dim, modulation=False) for _ in range(num_refiner_layers)])
        self.noise_refiner = nn.ModuleList(
            [SharedBlock(dim, modulation=True) for _ in range(num_refiner_layers)])
    def forward(self, x, temb):
        for blk in self.context_refiner:
            x = blk(x)
        for blk in self.noise_refiner:
            x = blk(x, temb)
        return x
'''
    f = tmp_path / f"modeling_{hashlib.md5(src.encode()).hexdigest()[:8]}.py"
    f.write_text(src)
    registry = build_registry([str(f)])
    vocab = load_conformance_transitive()
    stacks = {s.field_name: s for s in
              secondary_stacks_from_files((str(f),), "SharedHost")}
    assert dict(stacks["context_refiner"].ctor_kwargs) == {"modulation": False}
    assert dict(stacks["noise_refiner"].ctor_kwargs) == {"modulation": True}

    ctx = layer_facts_from_block("SharedBlock", registry, vocab,
                                 ctor_kwargs=stacks["context_refiner"].ctor_kwargs)
    assert ctx["residual_gated"] is False and ctx["gate_activation"] is None
    noise = layer_facts_from_block("SharedBlock", registry, vocab,
                                   ctor_kwargs=stacks["noise_refiner"].ctor_kwargs)
    assert noise["residual_gated"] is True and noise["gate_activation"] == "tanh"
    # no ctor info at all → class facts kept (never a wrong prune)
    bare = layer_facts_from_block("SharedBlock", registry, vocab)
    assert bare["residual_gated"] is True


def test_unparsed_router_raises_ambiguity_not_softmax(tmp_path):
    """B4: config + code both silent on the score transform while source IS
    installed → the router block carries the ambiguous-evidence envelope (the
    BLOCKING net's food) and the card names no transform; resolved corpora
    (config-declared or code-read) carry no envelope."""
    import hashlib, json, pathlib
    from model_unfolder.adapters.transformer.parser import _moe_routing
    from model_unfolder.evidence.models import SourceBundle

    class _Ctx:
        def __init__(self, bundle):
            self.source_bundle = bundle

    # A MoE-ish source whose router class the extractor can't resolve (no
    # routing tokens at all) — scoring stays unread while source exists.
    src = '''
import torch.nn as nn
class OpaqueBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlp = nn.Linear(config.hidden_size, config.hidden_size)
    def forward(self, x, past_key_value=None):
        return self.mlp(self.self_attn(x))
'''
    f = tmp_path / f"modeling_{hashlib.md5(src.encode()).hexdigest()[:8]}.py"
    f.write_text(src)

    class _Cfg:
        num_experts = 8
        num_experts_per_tok = 2
    routing = _moe_routing(_Cfg(), _Ctx(SourceBundle(source="local", files=(str(f),))))
    assert routing and routing.get("evidence", {}).get("status") == "ambiguous"
    assert "scoring_func" not in routing

    # no source at all → oracle_missing territory, NO ambiguity stamp
    routing2 = _moe_routing(_Cfg(), _Ctx(SourceBundle(source="local", files=())))
    assert not (routing2 or {}).get("evidence")

    # blessed MoE fixtures stay envelope-free (config- or code-resolved)
    from model_unfolder.sable import DEFAULT_CORPUS
    from model_unfolder import unfold
    for stem in ("deepseek-v3", "glm-4-5", "gpt-oss-20b"):
        fx = json.loads((pathlib.Path(DEFAULT_CORPUS) / f"{stem}.json").read_text())
        d = unfold(fx["config"], inspect_code=True, code_source="local")
        js = d.to_json() if not callable(getattr(d, "to_json", None)) else d.to_json()
        blob = js if isinstance(js, str) else json.dumps(js)
        assert '"status": "ambiguous"' not in blob.replace("'", '"'), stem


def test_router_ambiguity_reaches_blocking_net(tmp_path):
    """B4 end-to-end: the routing envelope lands on the router BLOCK's detail
    and the blocking evidence_ambiguity finder reports it."""
    import hashlib
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.sable import _ambiguous_evidence_findings

    src = '''
import torch.nn as nn
class OpaqueBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlp = nn.Linear(config.hidden_size, config.hidden_size)
    def forward(self, x, past_key_value=None):
        return self.mlp(self.self_attn(x))
'''
    f = tmp_path / f"modeling_{hashlib.md5(src.encode()).hexdigest()[:8]}.py"
    f.write_text(src)
    cfg = {"model_type": "llama", "hidden_size": 256, "num_hidden_layers": 2,
           "num_attention_heads": 8, "num_key_value_heads": 8,
           "intermediate_size": 512, "vocab_size": 1000,
           "num_experts": 8, "num_experts_per_tok": 2}
    ctx = ParseContext(source_bundle=SourceBundle(source="local", files=(str(f),)),
                       source="local")
    from model_unfolder.parser import config_to_ir
    from model_unfolder.diagram import Diagram
    diagram = Diagram(config_to_ir(cfg, parse_context=ctx))
    findings = _ambiguous_evidence_findings(diagram.to_ir())
    assert any("router" in x for x in findings), findings


def test_companion_denoiser_notes_from_vocabulary():
    """C2: companion-denoiser keys are YAML vocabulary — the expert-switch
    spelling (`transformer_2` + boundary_ratio) and the CFG-twin spelling both
    produce their note; neither key present → no note."""
    import json, pathlib
    from model_unfolder.sable import DEFAULT_CORPUS
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.adapters.diffusor.parser import parse as parse_diffusor

    fx = json.loads((pathlib.Path(DEFAULT_CORPUS)
                     / "wan2-2-t2v-a14b-diffusers.json").read_text())
    cfg = fx["config"]
    ir = parse_diffusor(cfg, context=ParseContext.build(cfg, source="local"))
    joined = " ".join(ir.notes or [])
    assert "transformer_2" in joined and "0.875" in joined

    twin_cfg = {"_class_name": "SomePipeline",
                "transformer": ["diffusers", "SomeTransformer2DModel"],
                "unconditional_transformer": ["diffusers", "SomeTransformer2DModel"],
                "num_layers": 2, "num_attention_heads": 8, "attention_head_dim": 64,
                "in_channels": 4, "joint_attention_dim": 2048}
    ir2 = parse_diffusor(twin_cfg)
    joined2 = " ".join(ir2.notes or [])
    assert "CFG twin" in joined2 and "unconditional_transformer" in joined2

    plain = {"_class_name": "SomePipeline",
             "transformer": ["diffusers", "SomeTransformer2DModel"],
             "num_layers": 2, "num_attention_heads": 8, "attention_head_dim": 64,
             "in_channels": 4, "joint_attention_dim": 2048}
    assert not [n for n in (parse_diffusor(plain).notes or [])
                if "denoiser" in n or "twin" in n]


def test_dit_attention_bias_from_declared_config():
    """C3: a diffusers config is a constructor record — a declared
    `attention_bias: true` reaches the DiT AttentionSpec; silence keeps the
    bias-less default (never a guess)."""
    import json, pathlib
    from model_unfolder.sable import DEFAULT_CORPUS
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.adapters.diffusor.parser import parse as dparse
    fx = json.loads((pathlib.Path(DEFAULT_CORPUS)
                     / "pixart-sigma-xl-2-1024-ms.json").read_text())
    ir = dparse(fx["config"], context=ParseContext.build(fx["config"], source="local"))
    assert ir.layers[0].attention.bias is True
    fx2 = json.loads((pathlib.Path(DEFAULT_CORPUS)
                      / "stable-diffusion-3-5-large.json").read_text())
    ir2 = dparse(fx2["config"], context=ParseContext.build(fx2["config"], source="local"))
    assert ir2.layers[0].attention.bias is False


def test_dit_norm_kind_resolved_from_classes_when_config_silent():
    """C4: config-silent DiT norms resolve from ROOT-scoped class evidence
    (never the pale "Normalization"), with code provenance in the prose;
    an explicit config `norm_type` still wins (config-first on diffusers)."""
    import json, pathlib
    from model_unfolder.sable import DEFAULT_CORPUS
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.adapters.diffusor.parser import parse as dparse

    want = {"stable-diffusion-3-5-large": "LayerNorm",
            "lumina-image-2-0": "RMSNorm"}
    for stem, kind in want.items():
        fx = json.loads((pathlib.Path(DEFAULT_CORPUS) / f"{stem}.json").read_text())
        ir = dparse(fx["config"], context=ParseContext.build(fx["config"], source="local"))
        norm = next(b for b in ir.layers[0].blocks
                    if isinstance(b, dict) and b.get("kind") == "norm")
        assert norm["label"] == kind, (stem, norm["label"])
        assert "read from the model code" in (norm.get("description") or ""), stem


def test_gemma2_softcaps_drawn_not_parked():
    """C1: attn_logit_softcapping becomes a REAL drawn node between QK^T and
    the softmax (+card), final_logit_softcapping reshapes the LM-head card;
    models without either stay byte-identical."""
    import json, pathlib
    from model_unfolder.sable import DEFAULT_CORPUS
    from model_unfolder.parser import config_to_ir
    from model_unfolder.diagram import Diagram
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.opgraph import attention_region
    from model_unfolder.ir import AttentionSpec, _attention_to_dict
    from model_unfolder.adapters.transformer.blocks.attention import (
        attention_detail, attention_child_blocks)

    capped = AttentionSpec(kind="gqa", num_heads=8, num_kv_heads=4, head_dim=64,
                           logit_softcap=50.0)
    d = attention_detail(capped)
    assert d["logit_softcap"] == 50.0
    region = attention_region(d, 512)
    ids = {op.id for op in region.ops}
    assert "attn_softcap" in ids
    assert any(e.src == "scaled_scores" and e.dst == "attn_softcap" for e in region.edges)
    assert any(e.src == "attn_softcap" and e.dst == "attn_softmax" for e in region.edges)
    assert "attn_softcap" in {c["id"] for c in attention_child_blocks(capped, 512)}

    plain = AttentionSpec(kind="gqa", num_heads=8, num_kv_heads=4, head_dim=64)
    assert "logit_softcap" not in _attention_to_dict(plain)
    assert "attn_softcap" not in {op.id for op in attention_region(attention_detail(plain), 512).ops}

    # end-to-end on the gemma-2 fixture: node + LM-head softcap card present
    fx = json.loads((pathlib.Path(DEFAULT_CORPUS) / "gemma-2-2b-it.json").read_text())
    cfg = fx["config"]
    diagram = Diagram(config_to_ir(cfg, parse_context=ParseContext.build(cfg, source="local")))
    ir = diagram.to_ir()
    attn = (ir.get("layers") or [{}])[0].get("attention") or {}
    assert attn.get("logit_softcap") == 50.0
    heads = [b for b in ((ir.get("extras") or {}).get("render") or {}).get("model_blocks", [])
             if b.get("id") == "lm_head"]
    assert heads and "softcap" in (heads[0].get("title") or "").lower()


def test_norm_placement_defaults_two_unknown_tiers():
    """B2 (U2 endpoint): placement has TWO unknown tiers.

    * source ABSENT (oracle_missing / zero evidence) → the pale
      "code-defined wiring" block — no fabricated pre-norm cells, no
      residual-tap claims (the tower_cell primitive on the main path);
    * source PRESENT but the reader abstained on the idiom → the
      conventional pre cell stays DRAWN (the op/nested-conformance oracle
      still checks its norms/residual ops against the readable forward())
      — recorded ambiguous + tagged asserted;
    * a proven placement (real source) carries no tag at all; a proven DiT
      sandwich is stated on the norm cards."""
    import json, pathlib
    from model_unfolder.adapters.transformer.parser import parse as parse_transformer
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.models import SourceBundle

    cfg = {"model_type": "llama", "hidden_size": 256, "num_hidden_layers": 2,
           "num_attention_heads": 8, "num_key_value_heads": 8,
           "intermediate_size": 512, "vocab_size": 1000}
    # Tier 1: no source at all → pale wiring block, nothing conventional.
    ctx = ParseContext(source_bundle=SourceBundle(source="local", files=()),
                       source="local")
    ir = parse_transformer(cfg, context=ctx)
    assert ir.layers[0].norm_placement == "unknown"
    ids = [b.get("id") for b in ir.layers[0].blocks if isinstance(b, dict)]
    assert "wiring_unresolved" in ids and "rms1" not in ids
    assert ctx.facts.records["decoder.layer.norm_placement"].status == "oracle_missing"

    # Tier 2/proven: real source (llama's topology reader proves pre) → the
    # concrete cell, no tag, code_proven ledger status.
    ctx2 = ParseContext.build(cfg, source="local")
    ir2 = parse_transformer(cfg, context=ctx2)
    assert ir2.layers[0].norm_placement == "pre"
    assert "norm_placement" not in (ir2.layers[0].ffn.asserted or ())
    assert ctx2.facts.records["decoder.layer.norm_placement"].status == "code_proven"

    # proven DiT sandwich → stated on the norm cards
    from model_unfolder.sable import DEFAULT_CORPUS
    from model_unfolder.adapters.diffusor.parser import parse as dparse
    fx = json.loads((pathlib.Path(DEFAULT_CORPUS) / "lumina-image-2-0.json").read_text())
    dir_ = dparse(fx["config"], context=ParseContext.build(fx["config"], source="local"))
    norm_descs = [b.get("description", "") for b in dir_.layers[0].blocks
                  if isinstance(b, dict) and b.get("kind") == "norm"]
    assert any("sandwich placement" in d for d in norm_descs)



def test_unet_prose_on_code_evidence_rail():
    """B3: UNet ResNet/conv activation follows the config's act_fn with
    provenance (class default stated as default); the Transformer2D FFN is
    honest-undeclared (never the old hardcoded GEGLU assertion)."""
    import json, pathlib
    from model_unfolder.sable import DEFAULT_CORPUS
    from model_unfolder.adapters.diffusor.unet import (
        parse_unet, unet_denoiser_children, _unet_transformer_subblocks)

    fx = json.loads((pathlib.Path(DEFAULT_CORPUS)
                     / "stable-diffusion-xl-base-1-0.json").read_text())
    cfg = fx["config"]
    sub = cfg.get("unet") or cfg          # pipeline index or bare unet config
    if not isinstance(sub, dict) or "block_out_channels" not in sub:
        sub = next((v for v in cfg.values()
                    if isinstance(v, dict) and "block_out_channels" in v), cfg)
    unet = parse_unet(sub)
    assert unet["act_fn"] == "silu"
    blob = json.dumps(unet_denoiser_children(unet))
    assert "GEGLU" not in blob
    assert "act_fn" in blob               # provenance sentence present

    st = {"id": "unet_down_1", "channels": 640, "num_heads": 10, "head_dim": 64}
    ff = next(b for b in _unet_transformer_subblocks(st, 2048)
              if b["id"].endswith("__ff"))
    assert "not declare" in ff["description"]
    assert ff["detail"]["ffn"].get("gated") is None

    # a synthetic gelu act_fn flows into every label
    unet2 = parse_unet({"block_out_channels": [64, 128], "in_channels": 4,
                        "out_channels": 4, "down_block_types":
                        ["DownBlock2D", "AttnDownBlock2D"],
                        "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
                        "act_fn": "gelu"})
    blob2 = json.dumps(unet_denoiser_children(unet2))
    assert "GroupNorm+GELU" in blob2 or "GroupNorm + GELU" in blob2
    assert "declared by the config" in blob2


def test_asserted_facts_tagged_and_advisory():
    """B5 (U2): the `asserted` tuple now carries ONLY the drawn conventions
    the doctrine keeps (sqrt(dim) scores, split storage, kept-pre placement) —
    everything else is either evidence-backed or a TYPED unknown, never a
    default presented as fact. mask specifically (strengthened by P2d): no
    architectures + no is_decoder, but the INSTALLED llama source
    unconditionally builds a causal mask — CODE-PROVEN causal now outranks
    the undeclared config (was a typed unknown before the reader existed);
    still no asserted tag either way."""
    import json, pathlib
    from model_unfolder import unfold
    from model_unfolder.sable import DEFAULT_CORPUS

    # bare llama-typed config WITHOUT architectures: decoder-ness undeclared
    # by CONFIG, but code-proven causal by the P2d reader — evidence-backed,
    # never a default presented as fact, no asserted tag.
    d = unfold({"model_type": "llama", "hidden_size": 256, "num_hidden_layers": 2,
                "num_attention_heads": 8, "num_key_value_heads": 8,
                "intermediate_size": 512, "vocab_size": 1000})
    ir = d.to_ir()
    attn = ir["layers"][0]["attention"]
    assert attn["mask"] == "causal"
    prov = (ir.get("extras") or {}).get("fact_provenance") or {}
    assert prov["decoder.attention.mask"]["status"] == "code_proven"
    assert "mask" not in (attn.get("asserted") or [])
    # scores_scale NOT tagged: the B1 code read backs sqrt(dim) (source installed)
    assert "scores_scale" not in (attn.get("asserted") or [])

    # the SAME config with the real checkpoint's declaration draws causal
    # with no tag (config_declared, not asserted).
    d2 = unfold({"model_type": "llama", "architectures": ["LlamaForCausalLM"],
                 "hidden_size": 256, "num_hidden_layers": 2,
                 "num_attention_heads": 8, "num_key_value_heads": 8,
                 "intermediate_size": 512, "vocab_size": 1000})
    attn2 = d2.to_ir()["layers"][0]["attention"]
    assert attn2["mask"] == "causal"
    assert "mask" not in (attn2.get("asserted") or [])

    # a fully declared+code-backed model: FFN carries no asserted key at all
    fx = json.loads((pathlib.Path(DEFAULT_CORPUS) / "llama-7b.json").read_text())
    ir2 = unfold(fx["config"], inspect_code=True, code_source="local").to_ir()
    assert "asserted" not in ir2["layers"][0]["ffn"]


def test_unet_ffn_activation_anchored_to_declared_blocks():
    """U1: the UNet Transformer2D FFN activation is read from the block class
    the config's block-type strings NAME (identity-as-address), never from an
    import-closure vote (which proves the wrong family) — SDXL proves geglu;
    no declared types or no source → None (honest-undeclared card stays)."""
    import json, pathlib
    from model_unfolder.sable import DEFAULT_CORPUS
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.conformance import _augment_diffusion_files
    from model_unfolder.evidence.patterns import unet_transformer_ffn_activation_from_files

    fx = json.loads((pathlib.Path(DEFAULT_CORPUS)
                     / "stable-diffusion-xl-base-1-0.json").read_text())
    cfg = fx["config"]
    ctx = ParseContext.build(cfg, source="local")
    root = (ctx.source_bundle.component_files or {}).get("root") or ctx.source_bundle.files
    files = _augment_diffusion_files(tuple(root))
    types = list(cfg.get("down_block_types") or []) + list(cfg.get("up_block_types") or []) \
        + [cfg.get("mid_block_type") or ""]
    assert unet_transformer_ffn_activation_from_files(files, types) == "geglu"
    assert unet_transformer_ffn_activation_from_files(files, []) is None
    assert unet_transformer_ffn_activation_from_files((), types) is None

    # end-to-end: the drawn FFN is geglu-with-provenance, and the naive vote's
    # wrong answer (flux's gelu-approximate) appears nowhere on the UNet card
    from model_unfolder import unfold
    html = unfold(cfg, inspect_code=True, code_source="local").to_html()
    assert "GEGLU" in html and "block types name" in html
    assert "does not declare its inner structure" not in html


def test_mla_kind_cross_check_both_directions(tmp_path):
    """U3: fact-conformance polices the attention KIND — a code-MLA drawn as
    GQA flags wrong_attention; drawn-MLA with no code MLA flags fabricated;
    agreeing models (DeepSeek drawn+code MLA, Llama neither) stay clean."""
    import json, pathlib
    from model_unfolder.sable import DEFAULT_CORPUS
    from model_unfolder.evidence.conformance import check_fact_conformance
    from model_unfolder import unfold

    for stem in ("deepseek-v3", "llama-7b"):
        fx = json.loads((pathlib.Path(DEFAULT_CORPUS) / f"{stem}.json").read_text())
        d = unfold(fx["config"], inspect_code=True, code_source="local")
        probs = [p for p in check_fact_conformance(fx["config"], d.to_ir())
                 if "attention_kind" in (p.view or "")]
        assert not probs, (stem, [p.message for p in probs])

    # a code-MLA whose config hides the latent ranks: drawn GQA must flag
    fx = json.loads((pathlib.Path(DEFAULT_CORPUS) / "deepseek-v3.json").read_text())
    hidden = {k: v for k, v in fx["config"].items()
              if k not in ("kv_lora_rank", "q_lora_rank",
                           "qk_nope_head_dim", "qk_rope_head_dim")}
    d2 = unfold(hidden, inspect_code=True, code_source="local")
    kinds = {(l.get("attention") or {}).get("kind") for l in d2.to_ir()["layers"]}
    if "mla" not in kinds:                      # config path indeed drew GQA/MHA
        probs2 = check_fact_conformance(hidden, d2.to_ir())
        assert any(p.kind == "wrong_attention" and "attention_kind" in (p.view or "")
                   for p in probs2), [p.message for p in probs2]


def test_cross_attention_schedule_matches_declared_layers():
    """U4a: the drawn cross-attention layer schedule EXACTLY matches the
    declared ``cross_attention_layers`` list — on the wrapper AND on the bare
    component config (whose cross layers were silently suppressed by a
    vision_config gate until this lock caught it)."""
    from transformers import AutoConfig
    import model_unfolder as mu
    for mt in ("mllama", "mllama_text_model"):
        cfg = AutoConfig.for_model(mt)
        text = getattr(cfg, "text_config", None) or cfg
        declared = list(getattr(text, "cross_attention_layers", None) or [])
        if not declared:
            continue
        ir = mu.config_to_ir(cfg)
        drawn = [i for i, l in enumerate(ir.layers) if l.attention.cross_attention]
        assert drawn == declared[:len(ir.layers)], \
            f"{mt}: cross schedule diverged (drawn {drawn[:6]}… vs declared {declared[:6]}…)"


def test_hybrid_mixer_schedule_matches_code_layer_types():
    """U4b: the linear-mixer vs full-attention schedule matches the
    ``layer_types`` list the attention/mixer classes read per index
    (qwen3_next: gated_delta on 'linear_attention', softmax kinds on
    'full_attention')."""
    from transformers import AutoConfig
    import model_unfolder as mu
    for mt in ("qwen3_next",):
        cfg = AutoConfig.for_model(mt)
        lt = list(getattr(cfg, "layer_types", None) or [])
        if not lt:
            continue
        ir = mu.config_to_ir(cfg)
        drawn = [l.attention.kind in ("gated_delta", "linear", "ssm", "rwkv")
                 for l in ir.layers]
        code = [x == "linear_attention" for x in lt][:len(drawn)]
        assert drawn == code, f"{mt}: mixer schedule diverged from layer_types"


def test_patch_ops_humanized_and_plumbing_collapsed():
    """Theme-L (all three Her Eyes DISLIKEs): a raw implementation CLASS NAME
    is never a drawn patch-op label (structure names the op; the class rides
    as provenance), and CONSECUTIVE reshape-kind plumbing collapses into one
    step whose card enumerates every move — the Tower-Census fallback rule."""
    from transformers import AutoConfig
    import model_unfolder as mu
    from model_unfolder.evidence.models import SourceOp
    from model_unfolder.evidence.vision import _collapse_plumbing_runs

    # collapse: 3 consecutive reshapes → 1 op, description enumerates in order
    ops = [SourceOp("conv", "Patch convolution", "X", "f.py", 1),
           SourceOp("reshape", "Flatten spatial grid", "X", "f.py", 2),
           SourceOp("reshape", "Transpose to tokens", "X", "f.py", 3),
           SourceOp("reshape", "Regroup patch tokens", "X", "f.py", 4),
           SourceOp("norm", "LayerNorm", "X", "f.py", 5)]
    got = _collapse_plumbing_runs(ops)
    assert [o.kind for o in got] == ["conv", "reshape", "norm"]
    assert "Flatten spatial grid → Transpose to tokens → Regroup patch tokens" \
        in got[1].description
    # single reshapes pass through untouched
    assert _collapse_plumbing_runs(ops[:2])[1].label == "Flatten spatial grid"

    # end-to-end: no camelCase class-name labels in any modality op list
    for mt in ("llama4", "gemma3", "qwen2_vl"):
        try:
            cfg = AutoConfig.for_model(mt)
        except Exception:
            continue
        ir = mu.unfold(cfg, inspect_code=True, code_source="local").to_ir()
        mods = ((ir.get("extras") or {}).get("modalities") or {}).get("inputs") or {}
        for name in ("vision", "video"):
            for op in ((mods.get(name) or {}).get("embedding") or {}).get("ops") or []:
                label = op["label"]
                assert not (label[:1].isupper() and any(c.islower() for c in label)
                            and any(c.isupper() for c in label[1:])
                            and " " not in label and label not in ("LayerNorm", "RMSNorm")), \
                    f"{mt}/{name}: class-name-like label {label!r}"


# ---------------------------------------------------------------------------
# U2 P2b — lm_head tying reader: unconditional manual assignment is proof;
# _tied_weights_keys is CAPABILITY and must never count (the mandatory
# negative — post_init ties gated on the config flag, so treating the class
# attribute as proof would fabricate tying on every llama).
# ---------------------------------------------------------------------------

def test_lm_head_tying_reader_unconditional_assign_is_proof(tmp_path):
    from model_unfolder.evidence.patterns import lm_head_tying_from_files

    tied = tmp_path / "tied.py"
    tied.write_text(
        "class FakeForCausalLM:\n"
        "    def __init__(self, config):\n"
        "        self.transformer = FakeModel(config)\n"
        "        self.lm_head = nn.Linear(4, 8, bias=False)\n"
        "        self.lm_head.weight = self.transformer.embed_tokens.weight\n"
        "    def forward(self, x):\n"
        "        return self.lm_head(self.transformer(x))\n"
    )
    assert lm_head_tying_from_files((str(tied),)) is True

    # deep chains and legacy field spellings resolve the same way
    wte = tmp_path / "wte.py"
    wte.write_text(
        "class LegacyLMHeadModel:\n"
        "    def __init__(self, config):\n"
        "        self.output_layer.weight = self.model.decoder.wte.weight\n"
    )
    assert lm_head_tying_from_files((str(wte),)) is True


def test_lm_head_tying_capability_is_never_proof(tmp_path):
    """THE mandatory negative: _tied_weights_keys alone (llama's shape —
    tying happens in post_init gated on config.tie_word_embeddings) must
    yield None, not True."""
    from model_unfolder.evidence.patterns import lm_head_tying_from_files

    capability = tmp_path / "capability.py"
    capability.write_text(
        "class FakeForCausalLM:\n"
        "    _tied_weights_keys = [\"lm_head.weight\"]\n"
        "    def __init__(self, config):\n"
        "        self.model = FakeModel(config)\n"
        "        self.lm_head = nn.Linear(4, 8, bias=False)\n"
        "        self.post_init()\n"
        "    def forward(self, x):\n"
        "        return self.lm_head(self.model(x))\n"
    )
    assert lm_head_tying_from_files((str(capability),)) is None


def test_lm_head_tying_conditional_assign_is_config_gated_not_proof(tmp_path):
    """An if-nested manual tie is the config's decision, not code truth."""
    from model_unfolder.evidence.patterns import lm_head_tying_from_files

    gated = tmp_path / "gated.py"
    gated.write_text(
        "class FakeForCausalLM:\n"
        "    def __init__(self, config):\n"
        "        self.model = FakeModel(config)\n"
        "        self.lm_head = nn.Linear(4, 8, bias=False)\n"
        "        if config.tie_word_embeddings:\n"
        "            self.lm_head.weight = self.model.embed_tokens.weight\n"
    )
    assert lm_head_tying_from_files((str(gated),)) is None
    # source absent → honest None
    assert lm_head_tying_from_files(()) is None


# ---------------------------------------------------------------------------
# U2 P2c — ACT2FN dispatch reader: code proves THAT an activation applies and
# NAMES the deciding config field; config supplies WHICH (code_and_config).
# ---------------------------------------------------------------------------

def test_ffn_activation_dispatch_field_reader(tmp_path):
    from model_unfolder.evidence.patterns import (
        ffn_activation_dispatch_field_from_files,
    )

    dispatch = tmp_path / "dispatch.py"
    dispatch.write_text(
        "class FakeAttention:\n"
        "    def __init__(self, config):\n"
        "        self.q_proj = nn.Linear(4, 4)\n"
        "        self.k_proj = nn.Linear(4, 4)\n"
        "        self.v_proj = nn.Linear(4, 4)\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "class FakeMLP:\n"
        "    def __init__(self, config):\n"
        "        self.up_proj = nn.Linear(4, 8)\n"
        "        self.down_proj = nn.Linear(8, 4)\n"
        "        self.act_fn = ACT2FN[config.hidden_act]\n"
        "    def forward(self, x):\n"
        "        return self.down_proj(self.act_fn(self.up_proj(x)))\n"
        "class FakeDecoderLayer:\n"
        "    def __init__(self, config):\n"
        "        self.self_attn = FakeAttention(config)\n"
        "        self.mlp = FakeMLP(config)\n"
        "    def forward(self, x, past_key_value=None):\n"
        "        return x\n"
    )
    assert ffn_activation_dispatch_field_from_files((str(dispatch),)) == "hidden_act"

    # get_activation(config.X) — the diffusers/self.config spelling
    getact = tmp_path / "getact.py"
    getact.write_text(
        "class OtherAttention:\n"
        "    def __init__(self, config):\n"
        "        self.q_proj = nn.Linear(4, 4)\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "class OtherMLP:\n"
        "    def __init__(self, config):\n"
        "        self.fc1 = nn.Linear(4, 8)\n"
        "        self.fc2 = nn.Linear(8, 4)\n"
        "        self.act = get_activation(self.config.activation_function)\n"
        "    def forward(self, x):\n"
        "        return self.fc2(self.act(self.fc1(x)))\n"
        "class OtherDecoderLayer:\n"
        "    def __init__(self, config):\n"
        "        self.self_attn = OtherAttention(config)\n"
        "        self.mlp = OtherMLP(config)\n"
        "    def forward(self, x, past_key_value=None):\n"
        "        return x\n"
    )
    assert (ffn_activation_dispatch_field_from_files((str(getact),))
            == "activation_function")

    # HARDCODED module (BLOOM shape): the dispatch reader abstains — that is
    # the name-match reader's fact (code_proven), not code_and_config.
    hardcoded = tmp_path / "hardcoded.py"
    hardcoded.write_text(
        "class HardAttention:\n"
        "    def __init__(self, config):\n"
        "        self.q_proj = nn.Linear(4, 4)\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "class HardMLP:\n"
        "    def __init__(self, config):\n"
        "        self.fc1 = nn.Linear(4, 8)\n"
        "        self.gelu_impl = FastGelu()\n"
        "        self.fc2 = nn.Linear(8, 4)\n"
        "    def forward(self, x):\n"
        "        return self.fc2(self.gelu_impl(self.fc1(x)))\n"
        "class HardDecoderLayer:\n"
        "    def __init__(self, config):\n"
        "        self.self_attn = HardAttention(config)\n"
        "        self.mlp = HardMLP(config)\n"
        "    def forward(self, x, past_key_value=None):\n"
        "        return x\n"
    )
    assert ffn_activation_dispatch_field_from_files((str(hardcoded),)) is None
    # source absent → None
    assert ffn_activation_dispatch_field_from_files(()) is None


def test_activation_dispatch_installed_witnesses():
    """Installed-source witnesses: llama names hidden_act; gpt2 names
    activation_function (the alias spelling its own code reads)."""
    import transformers, pathlib
    from model_unfolder.evidence.patterns import (
        ffn_activation_dispatch_field_from_files,
    )
    base = pathlib.Path(transformers.__file__).parent / "models"
    for ff, want in (("llama/modeling_llama.py", "hidden_act"),
                     ("gpt2/modeling_gpt2.py", "activation_function")):
        p = base / ff
        if not p.exists():
            continue
        assert ffn_activation_dispatch_field_from_files((str(p),)) == want, ff


def test_moe_container_defers_to_expert_class_gating(tmp_path):
    """U2 P2c extension: a routing container is not the FFN — its EXPERT
    classes are. Gated experts (3-linear or Parameter-fused gate_up) prove
    True; dense 2-linear experts prove False; an unresolvable container
    still abstains (None)."""
    from model_unfolder.evidence.patterns import decoder_ffn_gated_from_files

    def _file(name, expert_body):
        p = tmp_path / name
        p.write_text(
            "class FakeAttention:\n"
            "    def __init__(self, config):\n"
            "        self.q_proj = nn.Linear(4, 4)\n"
            "    def forward(self, x):\n"
            "        return x\n"
            + expert_body +
            "class FakeDecoderLayer:\n"
            "    def __init__(self, config):\n"
            "        self.self_attn = FakeAttention(config)\n"
            "        self.mlp = FakeMoEBlock(config)\n"
            "    def forward(self, x, past_key_value=None):\n"
            "        return x\n"
        )
        return (str(p),)

    gated = _file("gated.py",
        "class ExpertMLP:\n"
        "    def __init__(self, config):\n"
        "        self.gate_proj = nn.Linear(4, 8)\n"
        "        self.up_proj = nn.Linear(4, 8)\n"
        "        self.down_proj = nn.Linear(8, 4)\n"
        "    def forward(self, x):\n"
        "        return self.down_proj(act(self.gate_proj(x)) * self.up_proj(x))\n"
        "class FakeMoEBlock:\n"
        "    def __init__(self, config):\n"
        "        self.experts = ExpertMLP(config)\n"
        "        self.router = nn.Linear(4, 8)\n"
        "    def forward(self, x):\n"
        "        weights, idx = topk(softmax(self.router(x)), 2)\n"
        "        return self.experts(x) * weights\n")
    assert decoder_ffn_gated_from_files(gated, cfg={}) is True

    dense = _file("dense.py",
        "class ExpertMLP:\n"
        "    def __init__(self, config):\n"
        "        self.fc1 = nn.Linear(4, 8)\n"
        "        self.fc2 = nn.Linear(8, 4)\n"
        "    def forward(self, x):\n"
        "        return self.fc2(relu(self.fc1(x)))\n"
        "class FakeMoEBlock:\n"
        "    def __init__(self, config):\n"
        "        self.experts = ExpertMLP(config)\n"
        "        self.router = nn.Linear(4, 8)\n"
        "    def forward(self, x):\n"
        "        weights, idx = topk(softmax(self.router(x)), 2)\n"
        "        return self.experts(x) * weights\n")
    assert decoder_ffn_gated_from_files(dense, cfg={}) is False

    # Parameter-fused experts (gpt-oss shape): gate-named Parameter + gate_mul
    fused = _file("fused.py",
        "class FusedExpertsMLP:\n"
        "    def __init__(self, config):\n"
        "        self.gate_up_proj = nn.Parameter(t)\n"
        "        self.down_proj = nn.Parameter(t)\n"
        "    def forward(self, x):\n"
        "        gate = x[..., ::2]\n"
        "        up = x[..., 1::2]\n"
        "        return (up * glu(gate)) @ self.down_proj\n"
        "class FakeMoEBlock:\n"
        "    def __init__(self, config):\n"
        "        self.experts = FusedExpertsMLP(config)\n"
        "        self.router = nn.Linear(4, 8)\n"
        "    def forward(self, x):\n"
        "        weights, idx = topk(softmax(self.router(x)), 2)\n"
        "        return self.experts(x) * weights\n")
    assert decoder_ffn_gated_from_files(fused, cfg={}) is True


# ---------------------------------------------------------------------------
# U2 P2d — attention causality reader (the mask direction): machinery calls
# with config-resolved is_decoder gates; is_causal literals; the sdpa
# is_causal=False trap stays OUT of bidirectional evidence.
# ---------------------------------------------------------------------------

def test_attention_causality_same_source_flips_with_config(tmp_path):
    """Counterexample class 1: ONE source file (is_decoder-gated machinery,
    the BERT/T5 shape) yields bidirectional for the plain checkpoint and
    causal for an is_decoder=True checkpoint."""
    from model_unfolder.evidence.patterns import attention_causality_from_files

    gated = tmp_path / "gated.py"
    gated.write_text(
        "class FakeModel:\n"
        "    def _make_masks(self, attention_mask, embedding_output):\n"
        "        if self.config.is_decoder:\n"
        "            attention_mask = create_causal_mask(config=self.config)\n"
        "        else:\n"
        "            attention_mask = create_bidirectional_mask(config=self.config)\n"
        "        return attention_mask\n"
    )
    assert attention_causality_from_files((str(gated),), {}) == "bidirectional"
    assert attention_causality_from_files((str(gated),),
                                          {"is_decoder": True}) == "causal"


def test_attention_causality_unconditional_and_literals(tmp_path):
    from model_unfolder.evidence.patterns import attention_causality_from_files

    # unconditional machinery ignores the config entirely (llama shape)
    uncond = tmp_path / "uncond.py"
    uncond.write_text(
        "class FakeModel:\n"
        "    def forward(self, x):\n"
        "        mask = create_causal_mask(config=self.config)\n"
        "        return mask\n"
    )
    assert attention_causality_from_files((str(uncond),), {}) == "causal"
    assert attention_causality_from_files((str(uncond),),
                                          {"is_decoder": False}) == "causal"

    # self.is_causal = True literal in an attention class (no machinery)
    literal = tmp_path / "literal.py"
    literal.write_text(
        "class FakeAttention:\n"
        "    def __init__(self, config):\n"
        "        self.is_causal = True\n"
        "    def forward(self, x):\n"
        "        return x\n"
    )
    assert attention_causality_from_files((str(literal),), {}) == "causal"

    # source absent → None (counterexample class 4)
    assert attention_causality_from_files((), {}) is None


def test_attention_causality_traps_stay_unproven(tmp_path):
    """is_causal=False is NOT bidirectional evidence (cross-attn and
    sdpa-with-additive-mask both spell it); mixed machinery is honest None;
    a cross-attn bidirectional mask gated on encoder inputs is skipped."""
    from model_unfolder.evidence.patterns import attention_causality_from_files

    trap = tmp_path / "trap.py"
    trap.write_text(
        "class FakeAttention:\n"
        "    def __init__(self, config):\n"
        "        self.is_causal = False\n"
        "    def forward(self, q, k, v, mask):\n"
        "        return scaled_dot_product_attention(q, k, v, attn_mask=mask,\n"
        "                                            is_causal=False)\n"
    )
    assert attention_causality_from_files((str(trap),), {}) is None

    mixed = tmp_path / "mixed.py"
    mixed.write_text(
        "class EncModel:\n"
        "    def forward(self, x):\n"
        "        return create_bidirectional_mask(config=self.config)\n"
        "class DecModel:\n"
        "    def forward(self, x):\n"
        "        return create_causal_mask(config=self.config)\n"
    )
    assert attention_causality_from_files((str(mixed),), {}) is None

    crossattn = tmp_path / "crossattn.py"
    crossattn.write_text(
        "class FakeModel:\n"
        "    def forward(self, x, encoder_hidden_states=None):\n"
        "        mask = create_causal_mask(config=self.config)\n"
        "        if encoder_hidden_states is not None:\n"
        "            enc_mask = create_bidirectional_mask(config=self.config)\n"
        "        return mask\n"
    )
    assert attention_causality_from_files((str(crossattn),), {}) == "causal"


def test_attention_causality_installed_witnesses():
    """Installed-source witnesses across the counterexample classes: same
    arch family ≠ same verdict (bert vs bert-as-decoder), similar config ≠
    same source (gpt2 vs bert)."""
    import pathlib, transformers
    from model_unfolder.evidence.patterns import attention_causality_from_files
    base = pathlib.Path(transformers.__file__).parent / "models"
    cases = [("bert/modeling_bert.py", {}, "bidirectional"),
             ("bert/modeling_bert.py", {"is_decoder": True}, "causal"),
             ("gpt2/modeling_gpt2.py", {}, "causal"),
             ("llama/modeling_llama.py", {}, "causal"),
             ("t5/modeling_t5.py", {}, "bidirectional")]
    for ff, cfg, want in cases:
        p = base / ff
        if not p.exists():
            continue
        assert attention_causality_from_files((str(p),), cfg) == want, (ff, cfg)
