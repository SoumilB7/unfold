"""U3-F exact-owner Q/K-normalization evidence controls."""
from __future__ import annotations

from dataclasses import replace

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.qk_norm import (
    QKNormGateAtom,
    decoder_qk_norm_evidence_for_path,
)


def _reader(tmp_path, init_body, forward_body):
    source = f"""
import torch
from torch import nn

class Attention:
    def __init__(self, config, layer_idx):
{_indent(init_body, 8)}

    def forward(self, hidden_states):
{_indent(forward_body, 8)}

class Block:
    def __init__(self, config, layer_idx):
        self.attn = Attention(config, layer_idx)

    def forward(self, hidden_states):
        return self.attn(hidden_states)

class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config, i) for i in range(config.layers)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class Wrapper:
    base_model_prefix = "model"

    def __init__(self, config):
        self.model = Model(config)
"""
    path = tmp_path / "model.py"
    path.write_text(source, encoding="utf-8")
    bundle = SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    return decoder_qk_norm_evidence_for_path(
        index, bundle, (), allow_root_stage=True)


def _reader_from_source(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(source, encoding="utf-8")
    bundle = SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    return decoder_qk_norm_evidence_for_path(
        index, bundle, (), allow_root_stage=True)


def _indent(text, spaces):
    prefix = " " * spaces
    return "\n".join(
        prefix + line if line else ""
        for line in text.strip().splitlines())


_LINEARS = """
self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
"""

_SCORE = """
value_states = self.v_proj(hidden_states)
scores = torch.matmul(query_states, key_states.transpose(-1, -2))
weights = torch.softmax(scores, dim=-1)
return torch.matmul(weights, value_states)
"""


def test_unconditional_q_and_k_norms_resolve_from_exact_dataflow(tmp_path):
    result = _reader(
        tmp_path,
        _LINEARS + """
self.q_norm = nn.RMSNorm(config.head_dim)
self.k_norm = nn.RMSNorm(config.head_dim)
""",
        """
query_states = self.q_norm(self.q_proj(hidden_states))
key_states = self.k_norm(self.k_proj(hidden_states))
""" + _SCORE,
    )
    assert result.status == "resolved", result.failures
    assert result.value.present is True
    assert result.value.gate == ()
    assert result.provenance[0].kind == "source"


def test_config_gate_is_exact_and_owner_qualified(tmp_path):
    result = _reader(
        tmp_path,
        _LINEARS + """
self.enabled = config.qk_layernorm
if self.enabled:
    self.q_norm = nn.LayerNorm(config.head_dim)
    self.k_norm = nn.LayerNorm(config.head_dim)
""",
        """
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
if self.enabled:
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)
""" + _SCORE,
    )
    assert result.status == "resolved", result.failures
    assert result.value.present is None
    assert result.value.gate == (
        QKNormGateAtom("qk_layernorm", ("qk_layernorm",)),)
    assert result.provenance[0].config_paths == (("qk_layernorm",),)


def test_per_layer_gate_preserves_both_code_selected_fields(tmp_path):
    result = _reader(
        tmp_path,
        _LINEARS + """
self.use_rope = config.no_rope_layers[layer_idx]
if config.use_qk_norm and self.use_rope:
    self.qk_norm = nn.RMSNorm(config.head_dim)
""",
        """
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
if hasattr(self, "qk_norm"):
    query_states = self.qk_norm(query_states)
    key_states = self.qk_norm(key_states)
""" + _SCORE,
    )
    assert result.status == "resolved", result.failures
    assert result.value.gate == (
        QKNormGateAtom(
            "no_rope_layers", ("no_rope_layers",), per_layer=True),
        QKNormGateAtom("use_qk_norm", ("use_qk_norm",)),
    )


def test_two_norms_on_only_query_lane_do_not_launder_missing_key_norm(
        tmp_path):
    result = _reader(
        tmp_path,
        _LINEARS + """
self.q_norm_a = nn.RMSNorm(config.head_dim)
self.q_norm_b = nn.RMSNorm(config.head_dim)
""",
        """
projected = self.q_proj(hidden_states)
query_a = self.q_norm_a(projected)
query_b = self.q_norm_b(projected)
query_states = query_a + query_b
key_states = self.k_proj(hidden_states)
""" + _SCORE,
    )
    assert result.status == "failed"
    assert "both exact score operands" in result.failures[0].detail


def test_query_norm_contamination_cannot_impersonate_key_normalization(
        tmp_path):
    result = _reader(
        tmp_path,
        _LINEARS + """
self.q_norm_a = nn.RMSNorm(config.head_dim)
self.q_norm_b = nn.RMSNorm(config.head_dim)
""",
        """
projected = self.q_proj(hidden_states)
query_a = self.q_norm_a(projected)
query_b = self.q_norm_b(projected)
query_states = query_a + query_b
key_states = self.k_proj(hidden_states) + query_a
""" + _SCORE,
    )
    assert result.status == "failed"


def test_normalized_scalar_in_free_compute_cannot_impersonate_query_norm(
        tmp_path):
    """A score dependency is not automatically the Q tensor lane.

    The free compute function multiplies an unnormalized query by a normalized
    scale.  Parameter-union rebinding must not treat that scale's norm as proof
    that the query itself was normalized.
    """
    result = _reader_from_source(tmp_path, """
import torch
from torch import nn

def eager_attention(module, query, key, value, scale):
    scores = torch.matmul(query * scale, key.transpose(-1, -2))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)

class Attention:
    def __init__(self, config, layer_idx):
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.scale_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.scale_norm = nn.RMSNorm(config.hidden_size)
        self.k_norm = nn.RMSNorm(config.hidden_size)

    def forward(self, hidden_states):
        query = self.q_proj(hidden_states)
        key = self.k_norm(self.k_proj(hidden_states))
        value = self.v_proj(hidden_states)
        scale = self.scale_norm(self.scale_proj(hidden_states))
        return eager_attention(self, query, key, value, scale)

class Block:
    def __init__(self, config, layer_idx):
        self.attn = Attention(config, layer_idx)
    def forward(self, hidden_states):
        return self.attn(hidden_states)

class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config, i) for i in range(config.layers)])
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
""")
    assert result.status == "failed"


def test_normalized_scalar_in_local_score_cannot_impersonate_query_norm(
        tmp_path):
    result = _reader(
        tmp_path,
        _LINEARS + """
self.scale_proj = nn.Linear(config.hidden_size, config.hidden_size)
self.scale_norm = nn.RMSNorm(config.hidden_size)
self.k_norm = nn.RMSNorm(config.hidden_size)
""",
        """
query_states = self.q_proj(hidden_states)
key_states = self.k_norm(self.k_proj(hidden_states))
scale = self.scale_norm(self.scale_proj(hidden_states))
value_states = self.v_proj(hidden_states)
scores = torch.matmul(
    query_states * scale, key_states.transpose(-1, -2))
weights = torch.softmax(scores, dim=-1)
return torch.matmul(weights, value_states)
""",
    )
    assert result.status == "failed"


def test_latent_norm_before_another_projection_is_not_qk_norm(tmp_path):
    result = _reader(
        tmp_path,
        """
self.q_a = nn.Linear(config.hidden_size, config.hidden_size)
self.q_b = nn.Linear(config.hidden_size, config.hidden_size)
self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
self.latent_norm = nn.RMSNorm(config.hidden_size)
""",
        """
query_states = self.q_b(self.latent_norm(self.q_a(hidden_states)))
key_states = self.k_proj(hidden_states)
""" + _SCORE,
    )
    assert result.status == "failed"


def test_plain_attention_is_unknown_not_proven_absent(tmp_path):
    result = _reader(
        tmp_path,
        _LINEARS,
        """
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
""" + _SCORE,
    )
    assert result.status == "failed"


def test_gate_atom_rejects_a_path_that_does_not_end_in_its_field():
    with pytest.raises(ValueError):
        QKNormGateAtom("use_qk_norm", ("foreign",))
    atom = QKNormGateAtom("use_qk_norm", ("text_config", "use_qk_norm"))
    with pytest.raises(TypeError):
        replace(atom, config_path=["text_config", "use_qk_norm"])
