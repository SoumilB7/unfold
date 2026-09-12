"""U8-D regression controls for retired config-authored layer schedules.

These spellings remain useful checkpoint syntax only when exact modeling code
reads them to select an already-proven mechanism.  In isolation none may create
attention, recurrent-mixer, compressed-attention, or MoE structure.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_unfolder import config_to_ir
from model_unfolder.evidence.context import ParseContext


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _base(**extra):
    return {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "vocab_size": 128,
        **extra,
    }


@pytest.mark.parametrize("declaration", [
    {"attn_type_list": [0, 0, 0, 1]},
    {"block_types": ["recurrent", "attention"]},
    {"attention_types": [[["global", "local"], 2]]},
    {"dense_attention_every_n_layers": 2},
    {"layer_types": [
        "linear_attention", "full_attention",
        "linear_attention", "full_attention"]},
    {"compress_ratios": [0, 4, 128, 4]},
])
def test_config_schedule_without_source_authors_no_mixer_or_compression(
        declaration):
    ir = config_to_ir(_base(**declaration))
    assert {layer.attention.kind for layer in ir.layers} == {None}
    assert {layer.attention.compress_ratio for layer in ir.layers} == {None}
    facts = ir.extras.get("fact_provenance") or {}
    assert "decoder.layer.layer_schedule" not in facts
    assert "decoder.attention.mixer_schedule" not in facts


def test_config_only_moe_membership_does_not_create_experts():
    ir = config_to_ir(_base(
        num_local_experts=8,
        num_experts_per_tok=2,
        moe_layers_enum="1,3"))
    assert all(layer.ffn.kind != "moe" for layer in ir.layers)


def test_real_qwen35_source_selects_exact_hybrid_schedule():
    config = json.loads(
        (_CORPUS / "qwen3-5-27b-text.json").read_text(encoding="utf-8")
    )["config"]
    context = ParseContext.build(config)
    ir = config_to_ir(config, parse_context=context)
    assert tuple(layer.attention.kind for layer in ir.layers[:8]) == (
        "gated_delta", "gated_delta", "gated_delta", "gqa",
        "gated_delta", "gated_delta", "gated_delta", "gqa",
    )
    fact = context.facts.typed["decoder.attention.mixer_schedule"]
    assert fact.status == "code_and_config"
    assert set(fact.config_paths) == {"num_hidden_layers", "layer_types"}


def test_plain_model_ignores_a_familiar_mixer_token_list():
    config = json.loads(
        (_CORPUS / "llama-7b.json").read_text(encoding="utf-8"))["config"]
    config["layer_types"] = ["linear_attention"] * config["num_hidden_layers"]
    context = ParseContext.build(config)
    ir = config_to_ir(config, parse_context=context)
    assert {layer.attention.kind for layer in ir.layers} == {"mha"}
    assert context.facts.typed["decoder.attention.mixer_schedule"].value \
        == ("ordinary_attention",) * config["num_hidden_layers"]


def test_block_configs_nas_projection_stays_quarantined():
    config = _base(block_configs=[
        {"attention": {"attention_type": "sliding"}, "ffn": {"type": "moe"}},
        {"attention": {"attention_type": "global"}, "ffn": {"type": "dense"}},
    ])
    ir = config_to_ir(config)
    assert {layer.attention.kind for layer in ir.layers} == {None}
    assert {layer.ffn.kind for layer in ir.layers} != {"moe"}
