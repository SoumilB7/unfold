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
        self.o = nn.Linear(config.hidden, config.hidden{attention_bias})
        self.unrelated = nn.Linear(config.hidden, config.hidden, bias=False)
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-1, -2))
        context = torch.matmul(F.softmax(scores, dim=-1), v)
        return self.o(context)

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
def test_attention_bias_uses_only_exact_qkvo_projection_occurrences(
        tmp_path, suffix, expected):
    bundle = _bundle(tmp_path, attention_bias=suffix)
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.value is expected
    assert result.value.mechanism == "attention"
    assert len(result.value.projections) == 4


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


def test_config_gated_bias_carries_exact_path_without_reading_value(tmp_path):
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
    assert attention.status == "resolved"
    assert attention.value.value is None
    assert attention.value.config_path == ("attention_bias",)
    assert ffn.status == "failed"
    assert {item.kind for item in ffn.failures} == {
        "unsupported_syntax"}


def test_config_gated_bias_refuses_rival_projection_paths(tmp_path):
    bundle = _bundle(
        tmp_path, attention_bias=", bias=config.attention_bias")
    source = Path(bundle.files[0])
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "self.k = nn.Linear(config.hidden, config.hidden, "
            "bias=config.attention_bias)",
            "self.k = nn.Linear(config.hidden, config.hidden, "
            "bias=config.other_bias)"),
        encoding="utf-8")
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "ambiguous"


def test_config_gated_bias_follows_one_exact_local_config_alias(tmp_path):
    bundle = _bundle(
        tmp_path, attention_bias=", bias=section.attention_bias")
    source = Path(bundle.files[0])
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "    def __init__(self, config):\n"
            "        self.q = nn.Linear(config.hidden, config.hidden, "
            "bias=section.attention_bias)",
            "    def __init__(self, config):\n"
            "        section = config.attention\n"
            "        self.q = nn.Linear(config.hidden, config.hidden, "
            "bias=section.attention_bias)",
            1),
        encoding="utf-8")
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.value is None
    assert result.value.config_path == ("attention", "attention_bias")


def test_config_gated_bias_rejects_a_dynamic_expression(tmp_path):
    bundle = _bundle(
        tmp_path, attention_bias=", bias=bool(config.attention_bias)")
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "failed"
    assert {item.kind for item in result.failures} == {"unsupported_syntax"}


def test_projection_bias_dto_cannot_carry_value_and_config_path(tmp_path):
    bundle = _bundle(tmp_path, attention_bias=", bias=False")
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved"
    with pytest.raises(ValueError, match="exactly one"):
        replace(result.value, config_path=("attention_bias",))
    with pytest.raises(ValueError, match="exactly one"):
        replace(result.value, value=None)


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


def test_output_projection_must_agree_with_qkv_bias(tmp_path):
    bundle = _bundle(tmp_path, attention_bias=", bias=True")
    source = Path(bundle.files[0])
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "self.o = nn.Linear(config.hidden, config.hidden, bias=True)",
            "self.o = nn.Linear(config.hidden, config.hidden, bias=False)"),
        encoding="utf-8")
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "ambiguous"


def test_unrelated_later_linear_cannot_pose_as_output_projection(tmp_path):
    bundle = _bundle(tmp_path, attention_bias=", bias=False")
    source = Path(bundle.files[0])
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "return self.o(context)",
            "unused = self.unrelated(x)\n        return self.o(context)"),
        encoding="utf-8")
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.value is False
    assert len(result.value.projections) == 4


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
    ("bloom", (), "ordinary_ffn", True),
    ("gemma-2-2b-it", (), "ordinary_ffn", False),
    # Both the dense alternative and the invoked shared-expert alternative
    # independently prove bias=False; neither branch may certify the other.
    ("deepseek-v3", (), "ordinary_ffn", False),
    ("glm-4-5", (), "ordinary_ffn", False),
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


@pytest.mark.parametrize(("slug", "path"), [
    ("bloom", ()),
    ("qwen2-vl-7b-instruct", ("text_config",)),
])
def test_attention_bias_does_not_let_qkv_certify_an_unproven_output_projection(
        slug, path):
    from model_unfolder.evidence.context import ParseContext

    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_bias_for_path(
        context.program_index(), context.source_bundle, path,
        allow_root_stage=True)
    assert result.status == "failed"
    assert {item.kind for item in result.failures} == {"incomplete_graph"}


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
    assert attention.status == "failed"
    assert ffn.status == "resolved"
    assert ffn.value.value is True
    assert all(layer.attention.bias is None for layer in ir.layers)
    assert all(layer.ffn.bias is True for layer in ir.layers)
    fact = context.facts.records["decoder.attention.bias"]
    assert fact.status == "ambiguous"
    assert fact.value is None


def test_parser_consumes_only_the_bias_path_named_by_the_constructor(tmp_path):
    from model_unfolder import config_to_ir
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    bundle = _bundle(
        tmp_path, attention_bias=", bias=config.attention_bias")
    config = json.loads(
        (_CORPUS / "llama-7b.json").read_text(encoding="utf-8"))["config"]
    config["attention_bias"] = False
    # A disagreeing lookalike must be powerless: the code names the exact
    # ``attention_bias`` occurrence, not an alias family selected by the parser.
    config["use_qkv_bias"] = True
    cfg = _coerce(config)
    context = ParseContext(
        source_bundle=bundle,
        declared_decoderness="decoder_only_wrapper",
    )
    ir = config_to_ir(cfg, parse_context=context)
    assert all(layer.attention.bias is False for layer in ir.layers)
    fact = context.facts.records["decoder.attention.bias"]
    assert fact.value is False
    assert fact.status == "code_and_config"
    typed = context.facts.typed["decoder.attention.bias"]
    assert typed.config_paths == ("attention_bias",)
    consumed = tuple(
        event for event in context.config_access.events
        if event.intent == "consumed"
        and event.fact_owner == "decoder.attention"
        and event.fact_key == "bias")
    assert len(consumed) == 1
    assert consumed[0].config_path == "attention_bias"
    assert consumed[0].alias == "attention_bias"


def test_legacy_whole_file_projection_bias_readers_are_deleted():
    from model_unfolder.evidence import patterns

    assert not hasattr(patterns, "decoder_attention_bias_from_files")
    assert not hasattr(patterns, "decoder_mlp_bias_from_files")
