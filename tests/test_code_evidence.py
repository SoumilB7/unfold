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
    """The three storage/bookend detectors fire on the code SHAPE, never a
    name: fused experts (stacked gate_up split any way), fused QKV (one
    projection, no split q/k/v), and an embedding-stage norm applied to the
    embedding OUTPUT.  Negative controls prove absence stays None."""
    from model_unfolder.evidence.patterns import (
        attention_fused_qkv_from_files,
        embedding_stage_norm_from_files,
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
    assert embedding_stage_norm_from_files((fused,)) == "LayerNorm"
    # A FINAL norm applied to a reused variable name is NOT an embedding-stage
    # norm — the order-aware dataflow must not misread llama-shaped code.
    assert embedding_stage_norm_from_files((split,)) is None

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
# QK-norm — code-first (the code decides the SHAPE and names its own gate)
# ---------------------------------------------------------------------------

_QK_SCAFFOLD = '''
import torch
from torch import nn


class XRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(v + 1e-6))


class XMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


{attention}


class XDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = XAttention(config, layer_idx)
        self.mlp = XMLP(config)
        self.input_layernorm = XRMSNorm(config.hidden_size)

    def forward(self, hidden_states, past_key_values=None):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(hidden_states)
'''

_QK_UNCONDITIONAL_ATTN = '''
class XAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_norm = XRMSNorm(config.head_dim)
        self.k_norm = XRMSNorm(config.head_dim)

    def forward(self, hidden_states, past_key_values=None):
        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)
        attn = torch.matmul(query_states, key_states.transpose(-1, -2)).softmax(-1)
        return self.o_proj(torch.matmul(attn, value_states))
'''

_QK_GATED_ATTN = '''
class XAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(config.head_dim)
            self.k_layernorm = nn.LayerNorm(config.head_dim)

    def forward(self, hidden_states, past_key_values=None):
        fused = self.query_key_value(hidden_states)
        query_states, key_states, value_states = fused.chunk(3, dim=-1)
        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)
        attn = torch.matmul(query_states, key_states.transpose(-1, -2)).softmax(-1)
        return self.dense(torch.matmul(attn, value_states))
'''

_QK_COMPOSITE_ATTN = '''
class XAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.use_rope = config.no_rope_layers[layer_idx]
        if config.use_qk_norm and self.use_rope:
            self.qk_norm = XRMSNorm(config.head_dim)

    def forward(self, hidden_states, past_key_values=None):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if hasattr(self, "qk_norm"):
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)
        attn = torch.matmul(query_states, key_states.transpose(-1, -2)).softmax(-1)
        return self.o_proj(torch.matmul(attn, value_states))
'''

_QK_PLAIN_ATTN = '''
class XAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, past_key_values=None):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        attn = torch.matmul(query_states, key_states.transpose(-1, -2)).softmax(-1)
        return self.o_proj(torch.matmul(attn, value_states))
'''

_QK_MLA_LATENT_ATTN = '''
class XAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank)
        self.q_a_layernorm = XRMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, config.hidden_size)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, config.kv_lora_rank)
        self.kv_a_layernorm = XRMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, past_key_values=None):
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        compressed = self.kv_a_proj_with_mqa(hidden_states)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed))
        attn = torch.matmul(q, kv.transpose(-1, -2)).softmax(-1)
        return self.o_proj(torch.matmul(attn, kv))
'''


def _qk_files(tmp_path, attention_src):
    f = tmp_path / "modeling_x.py"
    f.write_text(_QK_SCAFFOLD.format(attention=attention_src))
    return (str(f),)


def test_qk_norm_unconditional_construction_is_present_without_config(tmp_path):
    """Qwen3/OLMo-2 shape: q/k norms built unconditionally and applied on the
    projection path ⇒ present, no config consulted (their configs are silent)."""
    from model_unfolder.evidence.patterns import decoder_qk_norm_from_files
    ev = decoder_qk_norm_from_files(_qk_files(tmp_path, _QK_UNCONDITIONAL_ATTN))
    assert ev is not None and ev.present is True and ev.gate == ()


def test_qk_norm_gate_is_the_field_the_code_reads(tmp_path):
    """StableLM/Persimmon shape: construction and application sit behind
    ``self.qk_layernorm = config.qk_layernorm`` — the atom is the config field
    the CODE names (never a spelling we guessed), through fused-QKV chunking."""
    from model_unfolder.evidence.patterns import decoder_qk_norm_from_files
    ev = decoder_qk_norm_from_files(_qk_files(tmp_path, _QK_GATED_ATTN))
    assert ev is not None and ev.present is None
    assert [(a.field, a.per_layer) for a in ev.gate] == [("qk_layernorm", False)]


def test_qk_norm_composite_gate_extracts_the_per_layer_term(tmp_path):
    """Llama-4 shape: ``config.use_qk_norm and self.use_rope`` where use_rope
    indexes ``config.no_rope_layers[layer_idx]`` — both atoms extracted, the
    per-layer one marked so the parser evaluates it per layer.  The
    ``hasattr(self, "qk_norm")`` application guard adds no atom."""
    from model_unfolder.evidence.patterns import decoder_qk_norm_from_files
    ev = decoder_qk_norm_from_files(_qk_files(tmp_path, _QK_COMPOSITE_ATTN))
    assert ev is not None and ev.present is None
    assert [(a.field, a.per_layer) for a in ev.gate] == [
        ("no_rope_layers", True), ("use_qk_norm", False)]


def test_qk_norm_proven_absent_when_the_class_builds_none(tmp_path):
    from model_unfolder.evidence.patterns import decoder_qk_norm_from_files
    ev = decoder_qk_norm_from_files(_qk_files(tmp_path, _QK_PLAIN_ATTN))
    assert ev is not None and ev.present is False


def test_qk_norm_mla_latent_norms_are_not_qk_norms(tmp_path):
    """DeepSeek MLA shape: ``q_a_layernorm``/``kv_a_layernorm`` results feed
    ANOTHER projection — intermediate norms by dataflow, so proven absent."""
    from model_unfolder.evidence.patterns import decoder_qk_norm_from_files
    ev = decoder_qk_norm_from_files(_qk_files(tmp_path, _QK_MLA_LATENT_ATTN))
    assert ev is not None and ev.present is False


def test_qk_norm_resolution_states():
    """The parser-side 5-state resolution: code shape × checkpoint values."""
    from model_unfolder.adapters.transformer.parser import _resolve_qk_norm_layers
    from model_unfolder.evidence.patterns import QKNormCodeEvidence, QKNormGateAtom

    # proven absent beats a declared spelling — a flag the code never reads is dead
    assert _resolve_qk_norm_layers(
        QKNormCodeEvidence(present=False), {"qk_layernorm": True}, True, 4
    ) == [False] * 4
    # unconditional: config not consulted
    assert _resolve_qk_norm_layers(
        QKNormCodeEvidence(present=True), {}, False, 3) == [True] * 3
    # gated: the named field's VALUE decides
    gated = QKNormCodeEvidence(
        present=None, gate=(QKNormGateAtom("qk_layernorm"),))
    assert _resolve_qk_norm_layers(gated, {"qk_layernorm": True}, False, 2) == [True] * 2
    assert _resolve_qk_norm_layers(gated, {"qk_layernorm": False}, True, 2) == [False] * 2
    # per-layer atom: the code indexes its own field by layer
    comp = QKNormCodeEvidence(present=None, gate=(
        QKNormGateAtom("no_rope_layers", per_layer=True),
        QKNormGateAtom("use_qk_norm"),
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
    truth; and T5 (``layer_norm_epsilon`` + RMS math) stays RMS.  Every plain
    RMS/LN control is unchanged."""
    from transformers import AutoConfig
    expect = {"phimoe": "LayerNorm", "t5": "RMSNorm", "llama": "RMSNorm",
              "bloom": "LayerNorm", "gemma2": "RMSNorm", "qwen3": "RMSNorm"}
    for mt, want in expect.items():
        ir = config_to_ir(AutoConfig.for_model(mt))
        drawn = {b.get("label") for l in ir.layers for b in (l.blocks or [])
                 if isinstance(b, dict) and b.get("kind") == "norm"}
        assert drawn == {want}, f"{mt}: drew {drawn}, expected {want}"


def test_norm_math_verdict_maps_torch_builtin_names():
    """The math reader classifies a torch-builtin norm by its API name (fixed
    library math), so a class with no in-file forward still resolves."""
    from model_unfolder.evidence.patterns import _norm_math_verdict
    import ast as _ast
    assert _norm_math_verdict(None, {}, "LayerNorm", _ast) == "layernorm"
    assert _norm_math_verdict(None, {}, "RMSNorm", _ast) == "rmsnorm"
    assert _norm_math_verdict(None, {}, "SomethingElse", _ast) is None


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


# ---------------------------------------------------------------------------
# Attention bias from construction (Group 2): code-authoritative QKV bias
# ---------------------------------------------------------------------------

def test_attention_bias_from_construction():
    """The QKV-projection bias read from the attention class's construction:
    Bloom/Qwen2 hardcode bias=True (config declares nothing → was drawn
    bias-less); Llama gates on config.attention_bias; Phi-3 is bias=False."""
    import transformers, pathlib
    from transformers import AutoConfig
    from model_unfolder.evidence.patterns import decoder_attention_bias_from_files
    base = pathlib.Path(transformers.__file__).parent / "models"
    cases = [("bloom", "bloom/modeling_bloom.py", None, True),
             ("qwen2", "qwen2/modeling_qwen2.py", None, True),
             ("phi3", "phi3/modeling_phi3.py", None, False),
             ("llama", "llama/modeling_llama.py", {"attention_bias": False}, False),
             ("llama", "llama/modeling_llama.py", {"attention_bias": True}, True)]
    for mt, ff, override, want in cases:
        cfg = AutoConfig.for_model(mt)
        if override:
            for k, v in override.items():
                setattr(cfg, k, v)
        got = decoder_attention_bias_from_files((str(base / ff),), cfg)
        assert got == want, f"{mt} {override}: got {got}, want {want}"


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
