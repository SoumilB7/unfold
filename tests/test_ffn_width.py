"""U7 exact-owner FFN intermediate-width controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest
from transformers import AutoConfig

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.ffn_width import (
    FFNIntermediateWidth,
    decoder_ffn_intermediate_width_for_path,
)
from model_unfolder.evidence.models import SourceBundle


def _read(config):
    context = ParseContext.build(config, source="local")
    return decoder_ffn_intermediate_width_for_path(
        context.program_index(), context.source_bundle, (), config,
        allow_root_stage=True)


@pytest.mark.parametrize("model_type,expected", [
    ("gptj", 16384),
    ("codegen", 16384),
    ("gpt2", 3072),
    ("bloom", 256),
])
def test_real_source_default_and_inline_widths_are_exact(model_type, expected):
    result = _read(AutoConfig.for_model(model_type))
    assert result.status == "resolved", result.failures
    assert result.value.value == expected
    assert result.value.premises
    assert result.value.spans


def test_explicit_gptj_inner_width_wins_through_the_same_code_expression():
    config = AutoConfig.for_model("gptj")
    config.n_inner = 9000
    result = _read(config)
    assert result.status == "resolved", result.failures
    assert result.value.value == 9000
    assert result.value.premises == ((('n_inner',), 9000),)


def test_missing_condition_operand_is_not_treated_as_explicit_none():
    config = AutoConfig.for_model("gptj").to_dict()
    config.pop("n_inner", None)
    result = _read(config)
    assert result.status == "failed"
    assert result.value is None


def test_declared_llama_width_round_trips_without_a_special_case():
    config = AutoConfig.for_model("llama")
    result = _read(config)
    assert result.status == "resolved", result.failures
    assert result.value.value == config.intermediate_size
    assert result.value.premises == (
        (("intermediate_size",), config.intermediate_size),)


def test_width_dto_rejects_value_and_provenance_forgery():
    result = _read(AutoConfig.for_model("bloom"))
    assert result.status == "resolved"
    value = result.value
    with pytest.raises(ValueError):
        replace(value, value=0)
    with pytest.raises(ValueError):
        replace(value, premises=())
    with pytest.raises(ValueError):
        replace(value, premises=(value.premises[0], value.premises[0]))
    with pytest.raises(ValueError):
        FFNIntermediateWidth(
            value.owner_occurrence, value.owner_symbol, value.value,
            value.premises, ())


def test_rival_up_projection_widths_do_not_certify_the_down_width(tmp_path):
    """The down projection alone cannot vouch for a gated pair whose two
    upstream lanes disagree.  Every exact affine lane must join on one width.
    """
    source = """
import torch
from torch import nn
from torch.nn import functional as F
class Attention:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        return torch.matmul(F.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1), v)
class FeedForward:
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden, config.wide_a)
        self.up = nn.Linear(config.hidden, config.wide_b)
        self.down = nn.Linear(config.wide_a, config.hidden)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
class Block:
    def __init__(self, config):
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
    def forward(self, x):
        return self.ffn(self.attn(x))
class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
"""
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = pi.build_program_index(bundle)
    result = decoder_ffn_intermediate_width_for_path(
        index, bundle, (),
        {"hidden": 8, "wide_a": 32, "wide_b": 64, "layers": 2},
        allow_root_stage=True)
    assert result.status == "failed"
    assert result.value is None
