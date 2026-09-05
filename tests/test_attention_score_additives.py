"""U8 exact score-side additive-application controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.context import slot_parse_context
from model_unfolder.evidence.decoder_block import decoder_block_path_at_root
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.attention_score_additives import (
    AttentionScoreAdditiveInventory,
    BaddbmmReceiverApplication,
    ExplicitAttentionScoreAdditiveApplication,
    attention_score_additives_at_block,
    decoder_attention_score_additives_for_path,
)


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _source(*, beta="1", call=None):
    score = call or (
        "bias.baddbmm(batch1=q, batch2=k.transpose(-1, -2), "
        "beta=self.beta, alpha=self.scale)")
    return f"""
import torch
from torch import nn
from torch.nn import functional as F

class Mixer:
    def __init__(self, config):
        self.beta = {beta}
        self.scale = 0.5
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x, bias):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        score = {score}
        weights = F.softmax(score, dim=-1)
        return torch.matmul(weights, v)

class Cell:
    def __init__(self, config):
        self.mixer = Mixer(config)
    def forward(self, x, bias):
        return self.mixer(x, bias)

class Core:
    def __init__(self, config):
        self.items = nn.ModuleList([Cell(config) for _ in range(config.layers)])
    def forward(self, x, bias):
        for item in self.items:
            x = item(x, bias)
        return x

class Wrapper:
    base_model_prefix = "core"
    def __init__(self, config):
        self.core = Core(config)
"""


def _pipeline(tmp_path, **kwargs):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(_source(**kwargs)), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    block = decoder_block_path_at_root(
        index, root, allow_root_stage=True)
    assert block.status == "resolved", block.failures
    return index, root, block.value.block_occurrence


def test_real_bloom_proves_exact_enabled_score_bias_application():
    config = json.loads(
        (_CORPUS / "bloom.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_score_additives_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, AttentionScoreAdditiveInventory)
    assert tuple(type(item) for item in result.value.applications) == (
        BaddbmmReceiverApplication,
        ExplicitAttentionScoreAdditiveApplication,
    )
    alibi, mask = result.value.applications
    assert alibi.protocol == "baddbmm_receiver"
    assert alibi.bias_operand.name == "alibi"
    assert alibi.beta_value == 1
    assert alibi.beta_premises == ()
    assert mask.protocol == "binary_add"
    assert mask.additive_operand.name == "attention_mask"
    assert tuple(name for name, _value in alibi.score_call.kwargs) == (
        "batch1", "batch2", "beta", "alpha")


def test_real_llama_exposes_only_its_exact_mask_addition():
    config = json.loads(
        (_CORPUS / "llama-7b.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_score_additives_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert len(result.value.applications) == 1
    application = result.value.applications[0]
    assert isinstance(application, ExplicitAttentionScoreAdditiveApplication)
    assert application.protocol == "binary_add"
    assert application.additive_operand.name == "attention_mask"


def test_real_t5_exposes_its_combined_score_side_operand_without_semantic_label():
    root_config = json.loads(
        (_CORPUS / "fluxtransformer2dmodel.json").read_text(
            encoding="utf-8"))["config"]
    context = slot_parse_context(
        ParseContext.build(root_config), "text_encoder_2")
    result = decoder_attention_score_additives_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert len(result.value.applications) == 1
    application = result.value.applications[0]
    assert isinstance(application, ExplicitAttentionScoreAdditiveApplication)
    assert application.protocol == "augmented_add"
    assert application.additive_operand.name == "position_bias_masked"


def test_explicit_zero_beta_proves_receiver_is_not_applied(tmp_path):
    index, root, block = _pipeline(tmp_path, beta="0")
    result = attention_score_additives_at_block(index, root, block)
    assert result.status == "absent"


def test_dynamic_beta_cannot_be_guessed_from_its_spelling(tmp_path):
    index, root, block = _pipeline(tmp_path, beta="config.bias_beta")
    result = attention_score_additives_at_block(index, root, block)
    assert result.status == "failed"
    assert "beta" in result.failures[0].detail


def test_exact_config_beta_can_join_without_making_config_the_mechanism(
        tmp_path):
    index, root, block = _pipeline(tmp_path, beta="config.bias_beta")
    result = attention_score_additives_at_block(
        index, root, block,
        config_selector=lambda path: (
            True, 2, "config_declared")
        if path == ("bias_beta",) else None)
    assert result.status == "resolved", result.failures
    application = result.value.applications[0]
    assert application.beta_value == 2
    assert application.beta_premises == (
        (("bias_beta",), "config_declared", 2),)
    assert result.provenance[-1].kind == "code_and_config"


def test_complete_local_renaming_does_not_change_the_protocol(tmp_path):
    renamed = _source().replace("bias", "offset").replace(
        "q =", "first =").replace("batch1=q", "batch1=first").replace(
        "k =", "second =").replace("batch2=k.", "batch2=second.")
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(renamed), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    block = decoder_block_path_at_root(
        index, root, allow_root_stage=True).value.block_occurrence
    result = attention_score_additives_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    assert result.value.applications[0].bias_operand.name == "offset"


def test_plain_score_to_softmax_proves_no_additive_application(tmp_path):
    index, root, block = _pipeline(
        tmp_path, call="torch.matmul(q, k.transpose(-1, -2))")
    result = attention_score_additives_at_block(index, root, block)
    assert result.status == "absent"


@pytest.mark.parametrize("call", [
    "bias.baddbmm(q, beta=self.beta, alpha=self.scale)",
    "bias.baddbmm(batch1=q, beta=self.beta, alpha=self.scale)",
    "bias.baddbmm(q, k, v, beta=self.beta, alpha=self.scale)",
    "bias.baddbmm(q, k)",
])
def test_incomplete_or_implicit_baddbmm_shapes_do_not_resolve(
        tmp_path, call):
    index, root, block = _pipeline(tmp_path, call=call)
    result = attention_score_additives_at_block(index, root, block)
    assert result.status != "resolved"


def test_dto_cannot_be_forged_from_a_different_receiver_or_zero_beta(tmp_path):
    index, root, block = _pipeline(tmp_path)
    result = attention_score_additives_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    value = result.value.applications[0]
    with pytest.raises(ValueError, match="derive from the score call"):
        replace(value, bias_operand=value.batch_operands[0])
    with pytest.raises(ValueError, match="finite non-zero beta"):
        replace(value, beta_value=0)
    with pytest.raises(ValueError, match="decisive operand"):
        replace(value, spans=(value.score_call.span, value.softmax_call.span))


def test_inventory_and_explicit_application_closure_reject_forgery():
    config = json.loads(
        (_CORPUS / "bloom.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_score_additives_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    baddbmm, explicit = result.value.applications
    with pytest.raises(ValueError, match="source order"):
        replace(result.value, applications=(explicit, baddbmm))
    with pytest.raises(ValueError, match="one exact score lane"):
        replace(explicit, score_lane="not_the_written_target")
    with pytest.raises(ValueError, match="between exact score and softmax"):
        replace(result.value, applications=(
            baddbmm,
            replace(explicit, application=replace(
                explicit.application, span=result.value.softmax_call.span),
                operation=replace(
                    explicit.operation, span=result.value.softmax_call.span),
                spans=(explicit.additive_operand.span,
                       result.value.softmax_call.span)),
        ))
