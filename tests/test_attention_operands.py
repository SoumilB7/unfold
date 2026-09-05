"""Exact attention-score operand boundary controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.attention_child import attention_child_evidence
from model_unfolder.evidence.attention_operands import (
    attention_qk_operands_evidence,
)
from model_unfolder.evidence.decoder_block import decoder_block_path_for_config
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


_SOURCE = """
import torch
from torch import nn

class Lane(nn.Module):
    def __init__(self, config):
        self.a = nn.Linear(config.hidden_size, config.hidden_size)
        self.b = nn.Linear(config.hidden_size, config.hidden_size)
        self.c = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden):
        left = self.a(hidden)
        right = self.b(hidden)
        payload = self.c(hidden)
        return torch.nn.functional.scaled_dot_product_attention(
            left, right, payload)

class Cell(nn.Module):
    def __init__(self, config):
        self.lane = Lane(config)

    def forward(self, hidden):
        return self.lane(hidden)

class Stage(nn.Module):
    def __init__(self, config):
        self.cells = nn.ModuleList(
            [Cell(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden):
        for cell in self.cells:
            hidden = cell(hidden)
        return hidden

class Wrapper(nn.Module):
    base_model_prefix = "model"

    def __init__(self, config):
        self.model = Stage(config)
"""


def _evidence(tmp_path, source=_SOURCE):
    path = tmp_path / "modeling_operands.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = build_program_index(bundle)
    block = decoder_block_path_for_config(
        index, bundle, (), allow_root_stage=True)
    assert block.status == "resolved"
    attention = attention_child_evidence(
        index, block.value.component_root, block.value.block_occurrence)
    assert attention.status == "resolved"
    return attention_qk_operands_evidence(
        index, block.value.component_root, attention.value)


def test_direct_framework_attention_exposes_exact_score_slots(tmp_path):
    result = _evidence(tmp_path)
    assert result.status == "resolved"
    assert result.value.protocol == "direct_attention"
    assert tuple(item.source_segment for item in (
        result.value.query_operand, result.value.key_operand)) == (
            "left", "right")


def test_operand_roles_do_not_require_distinct_projection_storage(tmp_path):
    source = _SOURCE.replace(
        "left, right, payload)", "left, left, payload)")
    result = _evidence(tmp_path, source)
    assert result.status == "resolved"
    assert result.value.query_operand.source_segment == "left"
    assert result.value.key_operand.source_segment == "left"


def test_complete_local_and_field_rename_does_not_change_operand_roles(tmp_path):
    source = (_SOURCE
              .replace("Lane", "OpaqueCompute")
              .replace("self.a", "self.first")
              .replace("self.b", "self.second")
              .replace("self.c", "self.third")
              .replace("left", "alpha")
              .replace("right", "beta")
              .replace("payload", "gamma"))
    result = _evidence(tmp_path, source)
    assert result.status == "resolved"
    assert tuple(item.source_segment for item in (
        result.value.query_operand, result.value.key_operand)) == (
            "alpha", "beta")


def test_direct_dot_softmax_uses_the_score_dot_not_the_output_dot(tmp_path):
    source = _SOURCE.replace(
        "return torch.nn.functional.scaled_dot_product_attention(\n"
        "            left, right, payload)",
        "scores = torch.matmul(left, right.transpose(-1, -2))\n"
        "        weights = torch.softmax(scores, dim=-1)\n"
        "        return torch.matmul(weights, payload)")
    result = _evidence(tmp_path, source)
    assert result.status == "resolved", result
    assert result.value.protocol == "dot_softmax"
    assert result.value.query_operand.source_segment == "left"
    assert result.value.key_operand.source_segment == "right.transpose(-1, -2)"


def test_operand_evidence_closure_rejects_foreign_provenance(tmp_path):
    value = _evidence(tmp_path).value
    with pytest.raises(ValueError):
        replace(value, spans=tuple(
            span for span in value.spans if span != value.score_call.span))
    with pytest.raises(ValueError):
        replace(value, attention_symbol=value.compute_callable)


def test_dot_softmax_helper_maps_exact_formals_back_to_entry_actuals():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext

    context = ParseContext.build(AutoConfig.for_model("llama"))
    index = context.program_index()
    block = decoder_block_path_for_config(
        index, context.source_bundle, (), allow_root_stage=True)
    attention = attention_child_evidence(
        index, block.value.component_root, block.value.block_occurrence)
    result = attention_qk_operands_evidence(
        index, block.value.component_root, attention.value)
    assert result.status == "resolved", result
    assert result.value.protocol == "dot_softmax"
    assert tuple(item.source_segment for item in (
        result.value.query_operand, result.value.key_operand)) == (
            "query_states", "key_states")
