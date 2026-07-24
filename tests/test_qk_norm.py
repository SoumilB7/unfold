"""U3-F exact-owner Q/K normalization controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.qk_norm import (
    QKNormCodeEvidence,
    QKNormGateAtom,
    decoder_qk_norm_evidence_for_path,
)


_SOURCE = """
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Norm:
        def __init__(self, dim):
            self.weight = torch.ones(dim)
        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            return self.weight * (x * torch.rsqrt(variance + 1e-6))

    def preserve_pair(q, k):
        q_out = q * 2
        k_out = k * 3
        return q_out, k_out

    def collapse_pair(q, k):
        return q, q

    class Attention:
        def __init__(self, config, layer_idx):
            self.q = nn.Linear(config.hidden, config.hidden)
            self.k = nn.Linear(config.hidden, config.hidden)
            self.v = nn.Linear(config.hidden, config.hidden)
            self.out = nn.Linear(config.hidden, config.hidden)
            {norm_init}
        def forward(self, x):
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            {norm_forward}
            weights = F.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
            return self.out(torch.matmul(weights, v))

    class Other:
        def __init__(self, config):
            self.up = nn.Linear(config.hidden, config.wide)
            self.norm = Norm(config.wide)
            self.down = nn.Linear(config.wide, config.hidden)
        def forward(self, x):
            return self.down(self.norm(self.up(x)))

    class Block:
        def __init__(self, config, layer_idx):
            self.left = Attention(config, layer_idx)
            self.right = Other(config)
        def forward(self, x):
            x = self.left(x)
            return self.right(x)

    class Core:
        def __init__(self, config):
            self.items = nn.ModuleList(
                [Block(config, i) for i in range(config.layers)])
        def forward(self, x):
            for item in self.items:
                x = item(x)
            return x

    class Wrapper:
        base_model_prefix = "core"
        def __init__(self, config):
            self.core = Core(config)
"""


def _reader(tmp_path, norm_init, norm_forward):
    path = tmp_path / "model.py"
    source = textwrap.dedent(_SOURCE.format(
        norm_init=textwrap.indent(
            textwrap.dedent(norm_init).strip(), " " * 12).lstrip(),
        norm_forward=textwrap.indent(
            textwrap.dedent(norm_forward).strip(), " " * 12).lstrip(),
    ))
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


def test_unconditional_q_and_k_norms_are_positive_code_evidence(tmp_path):
    result = _reader(
        tmp_path,
        """
        self.first = Norm(config.hidden)
        self.second = Norm(config.hidden)
        """,
        """
        q = self.first(q)
        k = self.second(k)
        """,
    )
    assert result.status == "resolved", result.failures
    assert result.value == QKNormCodeEvidence(True)
    assert result.provenance[0].kind == "source"


def test_gate_is_the_exact_config_path_read_by_the_code(tmp_path):
    result = _reader(
        tmp_path,
        """
        self.enabled = config.qk_layernorm
        if self.enabled:
            self.first = Norm(config.hidden)
            self.second = Norm(config.hidden)
        """,
        """
        if self.enabled:
            q = self.first(q)
            k = self.second(k)
        """,
    )
    assert result.status == "resolved", result.failures
    assert result.value == QKNormCodeEvidence(
        None, (QKNormGateAtom(
            "qk_layernorm", ("qk_layernorm",)),))
    assert result.provenance[0].config_paths == (("qk_layernorm",),)


def test_shared_norm_extracts_composite_and_per_layer_gate(tmp_path):
    result = _reader(
        tmp_path,
        """
        self.per_layer = config.no_rope_layers[layer_idx]
        if config.use_qk_norm and self.per_layer:
            self.shared = Norm(config.hidden)
        """,
        """
        if hasattr(self, "shared"):
            q = self.shared(q)
            k = self.shared(k)
        """,
    )
    assert result.status == "resolved", result.failures
    assert result.value.gate == (
        QKNormGateAtom(
            "no_rope_layers", ("no_rope_layers",), per_layer=True),
        QKNormGateAtom("use_qk_norm", ("use_qk_norm",)),
    )


def test_q_and_value_norms_cannot_impersonate_q_and_k(tmp_path):
    result = _reader(
        tmp_path,
        """
        self.first = Norm(config.hidden)
        self.second = Norm(config.hidden)
        """,
        """
        q = self.first(q)
        v = self.second(v)
        """,
    )
    assert result.status == "failed"
    assert "both exact score operands" in result.failures[0].detail


def test_exact_local_tuple_helper_preserves_independent_q_and_k_lanes(
        tmp_path):
    result = _reader(
        tmp_path,
        """
        self.first = Norm(config.hidden)
        self.second = Norm(config.hidden)
        """,
        """
        q = self.first(q)
        k = self.second(k)
        q, k = preserve_pair(q, k)
        """,
    )
    assert result.status == "resolved", result.failures


def test_local_tuple_helper_cannot_launder_one_norm_lane_into_both(tmp_path):
    result = _reader(
        tmp_path,
        """
        self.first = Norm(config.hidden)
        self.second = Norm(config.hidden)
        """,
        """
        q = self.first(q)
        k = self.second(k)
        q, k = collapse_pair(q, k)
        """,
    )
    assert result.status == "failed"
    assert "share a norm application" in result.failures[0].detail


def test_latent_norm_before_another_projection_is_not_qk_norm(tmp_path):
    result = _reader(
        tmp_path,
        """
        self.pre = nn.Linear(config.hidden, config.hidden)
        self.middle = Norm(config.hidden)
        self.post = nn.Linear(config.hidden, config.hidden)
        self.key_norm = Norm(config.hidden)
        """,
        """
        q = self.post(self.middle(self.pre(q)))
        k = self.key_norm(k)
        """,
    )
    assert result.status == "failed"


def test_plain_attention_and_sibling_norm_do_not_prove_absence_or_presence(
        tmp_path):
    result = _reader(tmp_path, "pass", "pass")
    assert result.status == "failed"
    assert result.value is None


def test_unknown_guard_never_becomes_an_unconditional_positive(tmp_path):
    result = _reader(
        tmp_path,
        """
        if runtime_probe(config):
            self.first = Norm(config.hidden)
            self.second = Norm(config.hidden)
        """,
        """
        if self.training:
            q = self.first(q)
            k = self.second(k)
        """,
    )
    assert result.status == "failed"
    assert result.failures[0].kind == "unsupported_syntax"


def test_qk_evidence_dto_rejects_absence_and_path_forgery():
    with pytest.raises(ValueError):
        QKNormCodeEvidence(False)
    atom = QKNormGateAtom("enabled", ("enabled",))
    with pytest.raises(ValueError):
        replace(atom, config_path=("other",))
    with pytest.raises(ValueError):
        QKNormCodeEvidence(True, (atom,))
