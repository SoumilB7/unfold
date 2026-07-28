"""U3-F exact-owner attention/ordinary-FFN projection bias controls."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.projection_bias import (
    EquivalentProjectionBiasEvidence,
    decoder_attention_bias_for_path,
    decoder_ffn_bias_for_path,
)


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _bundle(tmp_path, *, attention_bias="", ffn_bias=""):
    source = f"""
import torch
from torch import nn
from torch.nn import functional as F

class Attention:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden{attention_bias})
        self.k = nn.Linear(config.hidden, config.hidden{attention_bias})
        self.v = nn.Linear(config.hidden, config.hidden{attention_bias})
        self.unrelated = nn.Linear(config.hidden, config.hidden, bias=False)
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-1, -2))
        return torch.matmul(F.softmax(scores, dim=-1), v)

class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide{ffn_bias})
        self.down = nn.Linear(config.wide, config.hidden{ffn_bias})
        self.unrelated = nn.Linear(config.hidden, config.hidden, bias=False)
        self.act = nn.GELU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))

class Block:
    def __init__(self, config):
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
    def forward(self, x):
        x = self.attn(x)
        return self.ffn(x)

class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
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
    return SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )


@pytest.mark.parametrize(("suffix", "expected"), [
    ("", True),
    (", bias=True", True),
    (", bias=False", False),
])
def test_attention_bias_uses_only_exact_qkv_projection_occurrences(
        tmp_path, suffix, expected):
    bundle = _bundle(tmp_path, attention_bias=suffix)
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.value is expected
    assert result.value.mechanism == "attention"
    assert len(result.value.projections) == 3


@pytest.mark.parametrize(("suffix", "expected"), [
    ("", True),
    (", bias=True", True),
    (", bias=False", False),
])
def test_ffn_bias_uses_only_exact_reaching_projection_occurrences(
        tmp_path, suffix, expected):
    bundle = _bundle(tmp_path, ffn_bias=suffix)
    result = decoder_ffn_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.value is expected
    assert result.value.mechanism == "ordinary_ffn"
    assert len(result.value.projections) == 2


def test_config_gated_bias_is_not_mislabeled_code_only(tmp_path):
    bundle = _bundle(
        tmp_path,
        attention_bias=", bias=config.attention_bias",
        ffn_bias=", bias=config.mlp_bias",
    )
    index = build_program_index(bundle)
    attention = decoder_attention_bias_for_path(
        index, bundle, (), allow_root_stage=True)
    ffn = decoder_ffn_bias_for_path(
        index, bundle, (), allow_root_stage=True)
    assert attention.status == "failed"
    assert ffn.status == "failed"
    assert {item.kind for item in attention.failures} == {
        "unsupported_syntax"}
    assert {item.kind for item in ffn.failures} == {
        "unsupported_syntax"}


def test_disagreeing_exact_projections_are_ambiguous(tmp_path):
    bundle = _bundle(tmp_path)
    source = Path(bundle.files[0])
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "self.k = nn.Linear(config.hidden, config.hidden)",
        "self.k = nn.Linear(config.hidden, config.hidden, bias=False)")
    source.write_text(text, encoding="utf-8")
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "ambiguous"


def test_sibling_class_linear_cannot_vote(tmp_path):
    bundle = _bundle(tmp_path, attention_bias=", bias=True")
    source = Path(bundle.files[0])
    text = source.read_text(encoding="utf-8") + """
class Distractor:
    def __init__(self, config):
        self.a = nn.Linear(config.hidden, config.hidden, bias=False)
        self.b = nn.Linear(config.hidden, config.hidden, bias=False)
        self.c = nn.Linear(config.hidden, config.hidden, bias=False)
"""
    source.write_text(text, encoding="utf-8")
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved"
    assert result.value.value is True


@pytest.mark.parametrize(("slug", "path", "mechanism", "expected"), [
    ("bloom", (), "attention", True),
    ("bloom", (), "ordinary_ffn", True),
    ("gemma-2-2b-it", (), "ordinary_ffn", False),
    # Both the dense alternative and the invoked shared-expert alternative
    # independently prove bias=False; neither branch may certify the other.
    ("deepseek-v3", (), "ordinary_ffn", False),
    ("glm-4-5", (), "ordinary_ffn", False),
    ("qwen2-vl-7b-instruct", ("text_config",), "attention", True),
])
def test_real_source_only_projection_bias_examples(
        slug, path, mechanism, expected):
    from model_unfolder.evidence.context import ParseContext

    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    reader = (
        decoder_attention_bias_for_path
        if mechanism == "attention" else decoder_ffn_bias_for_path)
    result = reader(
        context.program_index(), context.source_bundle, path,
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.value is expected


def test_equivalent_projection_bias_rejects_cross_branch_disagreement():
    from model_unfolder.evidence.context import ParseContext

    config = json.loads(
        (_CORPUS / "deepseek-v3.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_ffn_bias_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved"
    first, second = result.value.variants
    with pytest.raises(ValueError, match="unanimously agree"):
        EquivalentProjectionBiasEvidence(
            "ordinary_ffn", (first, replace(second, value=not first.value)))
    with pytest.raises(ValueError, match="distinct branch evidence"):
        EquivalentProjectionBiasEvidence(
            "ordinary_ffn", (first, first))


def test_parser_consumes_the_same_exact_projection_bias_results():
    from model_unfolder import config_to_ir
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    config = json.loads(
        (_CORPUS / "bloom.json").read_text(encoding="utf-8"))["config"]
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    attention = context.reader_results[
        ("decoder.attention.projection_bias", ())]
    ffn = context.reader_results[
        ("decoder.ordinary_ffn.projection_bias", ())]
    assert attention.status == "resolved"
    assert ffn.status == "resolved"
    assert attention.value.value is True
    assert ffn.value.value is True
    assert all(layer.attention.bias is True for layer in ir.layers)
    assert all(layer.ffn.bias is True for layer in ir.layers)
    fact = context.facts.records["decoder.attention.bias"]
    assert fact.status == "code_proven"
    assert fact.source == "decoder_attention_bias_for_path"


def test_legacy_whole_file_projection_bias_readers_are_deleted():
    from model_unfolder.evidence import patterns

    assert not hasattr(patterns, "decoder_attention_bias_from_files")
    assert not hasattr(patterns, "decoder_mlp_bias_from_files")
