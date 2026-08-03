"""U3-F exact-owner attention/ordinary-FFN projection bias controls."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.parser import _coerce
from model_unfolder.evidence.attention_output import (
    AttentionOutputProjectionEvidence,
    decoder_attention_output_projection_for_path,
)
from model_unfolder.evidence.projection_bias import (
    EquivalentProjectionBiasEvidence,
    ProjectionBiasPatternEvidence,
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


def _helper_output_bundle(tmp_path, *, project_lane="context"):
    bundle = _bundle(tmp_path)
    source = Path(bundle.files[0])
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "class Attention:",
        "def attention_math(q, k, v):\n"
        "    scores = torch.matmul(q, k.transpose(-1, -2))\n"
        "    weights = F.softmax(scores, dim=-1)\n"
        "    context = torch.matmul(weights, v)\n"
        "    return context, weights\n\n"
        "class Attention:")
    text = text.replace(
        "        scores = torch.matmul(q, k.transpose(-1, -2))\n"
        "        context = torch.matmul(F.softmax(scores, dim=-1), v)\n"
        "        return self.o(context)",
        "        context, weights = attention_math(q, k, v)\n"
        f"        return self.o({project_lane})")
    source.write_text(text, encoding="utf-8")
    return bundle


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


def test_attention_output_requires_exact_terminal_to_linear_path(tmp_path):
    bundle = _bundle(tmp_path)
    result = decoder_attention_output_projection_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.projection not in result.value.input_projections
    assert result.value.output_source.span.line < result.value.call.span.line


def test_attention_output_does_not_follow_a_field_name_or_module_count(tmp_path):
    bundle = _bundle(tmp_path)
    source = Path(bundle.files[0])
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "return self.o(context)", "return context"),
        encoding="utf-8")
    result = decoder_attention_output_projection_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "failed"
    assert {failure.kind for failure in result.failures} == {"incomplete_graph"}


def test_attention_output_refuses_two_rival_linear_consumers(tmp_path):
    bundle = _bundle(tmp_path)
    source = Path(bundle.files[0])
    source.write_text(
        source.read_text(encoding="utf-8")
        .replace(
            "self.o = nn.Linear(config.hidden, config.hidden)",
            "self.o = nn.Linear(config.hidden, config.hidden)\n"
            "        self.o2 = nn.Linear(config.hidden, config.hidden)")
        .replace(
            "return self.o(context)",
            "left = self.o(context)\n        right = self.o2(context)\n"
            "        return left + right"),
        encoding="utf-8")
    result = decoder_attention_output_projection_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "failed"


def test_attention_output_dto_rejects_qkv_occurrence_as_output(tmp_path):
    bundle = _bundle(tmp_path)
    result = decoder_attention_output_projection_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved"
    value = result.value
    with pytest.raises(ValueError, match="exact construction site"):
        AttentionOutputProjectionEvidence(
            value.attention, value.input_projections,
            value.input_projections[0], value.projection_site, value.call,
            value.compute_terminal, value.output_source, value.spans)
    with pytest.raises((TypeError, ValueError)):
        replace(value, input_projections=(
            value.input_projections[0], value.input_projections[0]))
    with pytest.raises(ValueError, match="exact constructed field"):
        replace(value, call=value.compute_terminal)


def test_attention_output_follows_the_exact_helper_return_lane(tmp_path):
    bundle = _helper_output_bundle(tmp_path, project_lane="context")
    result = decoder_attention_output_projection_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures


def test_attention_output_rejects_projecting_the_helper_weights_lane(tmp_path):
    bundle = _helper_output_bundle(tmp_path, project_lane="weights")
    result = decoder_attention_output_projection_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "failed"
    assert {failure.kind for failure in result.failures} == {"incomplete_graph"}


@pytest.mark.parametrize("slug", ["llama-7b", "bloom", "qwen3-8b"])
def test_real_decoder_output_projection_is_exactly_proven(slug):
    data = json.loads((_CORPUS / f"{slug}.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    result = decoder_attention_output_projection_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", (slug, result.failures)


def test_parser_fact_region_and_receipt_share_output_projection_claim():
    from model_unfolder import Diagram, config_to_ir
    from model_unfolder.opgraph import attention_region

    data = json.loads((_CORPUS / "llama-7b.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    diagram = Diagram(config_to_ir(_coerce(data["config"]), parse_context=context))
    assert {layer.attention.output_projection for layer in diagram.ir.layers} == {
        True}
    fact = context.facts.typed["decoder.attention.output_projection"]
    assert fact.status == "code_proven" and fact.value is True
    region = attention_region(
        diagram.ir.to_dict()["layers"][0]["attention"],
        diagram.ir.hidden_size)
    assert "o_proj" in region.by_id()
    assert "attention_output_unresolved" not in region.by_id()
    diagram.to_html(standalone=True)
    receipts = tuple(
        receipt for event in diagram.render_events()
        for receipt in event.receipts
        if receipt.fact_key == "output_projection")
    assert receipts
    assert {receipt.node_ids for receipt in receipts} == {("o_proj",)}
    assert {receipt.mechanism for receipt in receipts} == {
        "attention_output_projection"}


def test_real_mla_output_uses_its_exact_latent_input_and_output_paths():
    from model_unfolder import config_to_ir
    from model_unfolder.opgraph import attention_region

    data = json.loads((_CORPUS / "deepseek-v3.json").read_text())
    cfg = _coerce(data["config"])
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    assert {layer.attention.output_projection for layer in ir.layers} == {True}
    region = attention_region(ir.to_dict()["layers"][0]["attention"], ir.hidden_size)
    assert region.by_id()["o_proj"].kind == "linear"
    assert "attention_output_unresolved" not in region.by_id()
    assert context.facts.typed[
        "decoder.attention.output_projection"].value is True


def test_real_mla_bias_keeps_every_affine_stage_and_joins_checkpoint_value():
    from model_unfolder import config_to_ir

    config = json.loads(
        (_CORPUS / "deepseek-v3.json").read_text(encoding="utf-8"))["config"]
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    result = decoder_attention_bias_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, ProjectionBiasPatternEvidence)
    # q_a -> q_b, kv_a -> kv_b, then the exact output projection.  Terminal
    # producers alone would silently omit both compression stages.
    assert len(result.value.projections) == 5
    assert result.value.config_paths == (("attention_bias",),)
    with pytest.raises(ValueError, match="distinct exact expressions"):
        replace(result.value, terms=(
            result.value.terms[0], result.value.terms[0]))
    bound_term = next(
        term for term in result.value.terms
        if term.config_path is not None)
    with pytest.raises(TypeError, match="exact config path"):
        replace(bound_term, config_path=())

    ir = config_to_ir(cfg, parse_context=context)
    assert {layer.attention.bias for layer in ir.layers} == {False}
    fact = context.facts.typed["decoder.attention.bias"]
    assert fact.value is False
    assert fact.status == "code_and_config"
    assert fact.config_paths == ("attention_bias",)
    assert not any(
        row.config_path == "attention_bias"
        for row in context.config_access.unconsumed_occurrences())


def test_latent_bias_true_is_reported_as_exact_mixed_layout():
    from model_unfolder import config_to_ir
    from model_unfolder.labels import attention_summary

    config = json.loads(
        (_CORPUS / "deepseek-v3.json").read_text(encoding="utf-8"))["config"]
    config["attention_bias"] = True
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    assert {layer.attention.bias for layer in ir.layers} == {"mixed"}
    assert context.facts.typed["decoder.attention.bias"].value == "mixed"
    _description, facts = attention_summary(
        ir.to_dict()["layers"][0]["attention"])
    assert "mixed projection bias" in facts


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


def test_config_gated_bias_retains_each_exact_projection_path(tmp_path):
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
    assert result.status == "resolved"
    assert isinstance(result.value, ProjectionBiasPatternEvidence)
    assert set(result.value.config_paths) == {
        ("attention_bias",), ("other_bias",)}


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


def test_disagreeing_exact_projections_are_an_exact_mixed_pattern(tmp_path):
    bundle = _bundle(tmp_path)
    source = Path(bundle.files[0])
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "self.k = nn.Linear(config.hidden, config.hidden)",
        "self.k = nn.Linear(config.hidden, config.hidden, bias=False)")
    source.write_text(text, encoding="utf-8")
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved"
    assert isinstance(result.value, ProjectionBiasPatternEvidence)
    assert not result.value.config_paths


def test_output_projection_disagreement_is_retained_as_mixed(tmp_path):
    bundle = _bundle(tmp_path, attention_bias=", bias=True")
    source = Path(bundle.files[0])
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "self.o = nn.Linear(config.hidden, config.hidden, bias=True)",
            "self.o = nn.Linear(config.hidden, config.hidden, bias=False)"),
        encoding="utf-8")
    result = decoder_attention_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved"
    assert isinstance(result.value, ProjectionBiasPatternEvidence)


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


def test_attention_bias_promotes_only_after_exact_output_projection_proof():
    from model_unfolder.evidence.context import ParseContext

    config = json.loads(
        (_CORPUS / "bloom.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_bias_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.value is True
    # BLOOM stores Q/K/V in one fused affine plus one output affine.
    assert len(result.value.projections) == 2


def test_attention_bias_preserves_real_nested_mixed_projection_layout():
    from model_unfolder.evidence.context import ParseContext

    config = json.loads(
        (_CORPUS / "qwen2-vl-7b-instruct.json").read_text(
            encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_bias_for_path(
        context.program_index(), context.source_bundle, ("text_config",),
        allow_root_stage=True)
    assert result.status == "resolved"
    assert isinstance(result.value, ProjectionBiasPatternEvidence)
    assert not result.value.config_paths


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
    assert attention.value.value is True
    assert ffn.status == "resolved"
    assert ffn.value.value is True
    assert all(layer.attention.bias is True for layer in ir.layers)
    assert all(layer.ffn.bias is True for layer in ir.layers)
    fact = context.facts.records["decoder.attention.bias"]
    assert fact.status == "code_proven"
    assert fact.value is True


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
