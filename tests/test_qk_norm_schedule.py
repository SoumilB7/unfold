"""U8-E exact per-layer Q/K-normalization schedule controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.qk_norm_schedule import (
    DecoderQKNormSchedule,
    decoder_qk_norm_schedule_for_path,
)


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _selector(document):
    def select(path):
        current = document
        for part in tuple(path):
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"
    return select


def _result(config):
    context = ParseContext.build(config)
    return decoder_qk_norm_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_selector(config))


@pytest.mark.parametrize("model_type", ["qwen3", "olmo2", "gemma3_text"])
def test_unconditional_real_qk_norm_is_true_at_every_exact_layer(model_type):
    from transformers import AutoConfig

    config = AutoConfig.for_model(model_type).to_dict()
    result = _result(config)
    assert result.status == "resolved", result.failures
    assert set(result.value.decisions) == {True}
    assert result.value.mechanism.present is True


def test_real_per_layer_gate_is_not_collapsed_to_one_boolean():
    from transformers import AutoConfig

    config = AutoConfig.for_model("llama4_text").to_dict()
    result = _result(config)
    assert result.status == "resolved", result.failures
    assert result.value.decisions[:8] == (
        True, True, True, False, True, True, True, False)
    assert {path for path, _kind in result.value.config_dependencies} >= {
        ("num_hidden_layers",), ("use_qk_norm",), ("no_rope_layers",)}


def test_real_false_gate_is_a_proven_negative_not_unknown():
    from transformers import AutoConfig

    config = AutoConfig.for_model("stablelm").to_dict()
    result = _result(config)
    assert result.status == "resolved", result.failures
    assert set(result.value.decisions) == {False}


def test_familiar_config_token_cannot_create_qk_norm_on_llama():
    config = json.loads(
        (_CORPUS / "llama-7b.json").read_text(encoding="utf-8"))["config"]
    config["use_qk_norm"] = True
    config["qk_layernorm"] = True
    result = _result(config)
    assert result.status == "failed"
    assert result.value is None


def test_short_per_layer_operand_never_defaults_or_repeats():
    from transformers import AutoConfig

    config = AutoConfig.for_model("llama4_text").to_dict()
    config["no_rope_layers"] = [True]
    result = _result(config)
    assert result.status == "failed"
    assert result.value is None


def test_schedule_closure_rejects_cross_occurrence_and_missing_dependencies():
    from transformers import AutoConfig

    value = _result(AutoConfig.for_model("qwen3").to_dict()).value
    assert isinstance(value, DecoderQKNormSchedule)
    with pytest.raises(ValueError):
        replace(value, attention_occurrence=value.block_occurrence)
    with pytest.raises(ValueError):
        replace(value, decisions=value.decisions[:-1])
    with pytest.raises(ValueError):
        replace(value, config_dependencies=())


def test_parser_projects_the_exact_mixed_schedule_and_ledger_fact():
    from transformers import AutoConfig
    from model_unfolder import config_to_ir

    config = AutoConfig.for_model("llama4_text").to_dict()
    context = ParseContext.build(config)
    ir = config_to_ir(config, parse_context=context)
    assert tuple(layer.attention.qk_norm for layer in ir.layers[:8]) == (
        True, True, True, False, True, True, True, False)
    fact = context.facts.typed["decoder.attention.qk_norm_schedule"]
    assert fact.value[:8] == (
        True, True, True, False, True, True, True, False)
