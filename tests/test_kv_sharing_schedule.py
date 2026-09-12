"""U8-E exact cross-layer K/V reuse controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from transformers import AutoConfig

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.kv_sharing_schedule import (
    DecoderKVSharingSchedule,
    decoder_kv_sharing_schedule_for_path,
)
from model_unfolder.evidence.program_index import build_program_index


def _selector(document):
    def select(path):
        current = document
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"
    return select


def _result(model_type, document=None):
    document = document or AutoConfig.for_model(model_type).to_dict()
    context = ParseContext.build(document)
    return decoder_kv_sharing_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_selector(document))


def _mutated_result(tmp_path, old, new):
    document = AutoConfig.for_model("gemma3n_text").to_dict()
    context = ParseContext.build(document)
    modeling = next(
        Path(item) for item in context.source_bundle.files
        if Path(item).name == "modeling_gemma3n.py")
    source = modeling.read_text(encoding="utf-8")
    assert old in source
    changed = tmp_path / modeling.name
    changed.write_text(source.replace(old, new, 1), encoding="utf-8")
    def swap(items):
        return tuple(str(changed) if Path(item) == modeling else item
                     for item in items)
    bundle = replace(
        context.source_bundle,
        files=swap(context.source_bundle.files),
        component_files={
            key: swap(items)
            for key, items in context.source_bundle.component_files.items()
        })
    return decoder_kv_sharing_schedule_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=_selector(document))


def test_real_gemma3n_reuses_the_last_pre_share_layer_of_the_same_type():
    result = _result("gemma3n_text")
    assert result.status == "resolved", result.failures
    assert result.value.decisions == (
        (None,) * 20
        + (18, 18, 18, 18, 19) * 3
    )
    assert set(result.value.config_dependencies) >= {
        (("num_hidden_layers",), "config_declared"),
        (("num_kv_shared_layers",), "config_declared"),
        (("layer_types",), "config_declared"),
    }


def test_real_gemma4_proves_no_reuse_when_exact_count_is_zero():
    result = _result("gemma4_text")
    assert result.status == "resolved", result.failures
    assert result.value.decisions == (None,) * 30


def test_parser_projects_only_the_exact_positive_reuse_schedule():
    from model_unfolder import config_to_ir

    document = AutoConfig.for_model("gemma3n_text").to_dict()
    context = ParseContext.build(document)
    ir = config_to_ir(document, parse_context=context)
    expected = (None,) * 20 + (18, 18, 18, 18, 19) * 3
    assert tuple(layer.attention.kv_source_layer for layer in ir.layers) == expected
    assert tuple((edge.from_layer, edge.to_layer)
                 for edge in ir.cross_layer_edges) == tuple(
        (source, index) for index, source in enumerate(expected)
        if source is not None)
    fact = context.facts.typed["decoder.attention.kv_sharing_schedule"]
    assert fact.value == expected
    assert "num_kv_shared_layers" not in ir.extras


def test_parser_ledgers_the_code_proven_all_none_schedule():
    from model_unfolder import config_to_ir

    document = AutoConfig.for_model("gemma4_text").to_dict()
    context = ParseContext.build(document)
    ir = config_to_ir(document, parse_context=context)
    assert all(layer.attention.kv_source_layer is None for layer in ir.layers)
    assert not ir.cross_layer_edges
    fact = context.facts.typed["decoder.attention.kv_sharing_schedule"]
    assert fact.value == (None,) * document["num_hidden_layers"]
    assert fact.status == "code_and_config"
    assert set(fact.config_paths) >= {
        "num_hidden_layers", "num_kv_shared_layers", "layer_types"}
    assert "num_kv_shared_layers" not in ir.extras


def test_plain_llama_has_no_kv_sharing_application():
    result = _result("llama")
    assert result.status == "absent"
    assert result.value is None


def test_config_count_cannot_create_kv_sharing_on_plain_llama():
    document = AutoConfig.for_model("llama").to_dict()
    document["num_kv_shared_layers"] = 8
    result = _result("llama", document)
    assert result.status == "absent"


def test_forward_must_read_both_lanes_from_the_shared_mapping(tmp_path):
    result = _mutated_result(
        tmp_path,
        "key_states, value_states = shared_kv_states[self.kv_shared_layer_index]",
        "key_states, value_states = hidden_states, hidden_states",
    )
    assert result.status == "absent"


def test_read_value_must_reach_the_attention_compute(tmp_path):
    result = _mutated_result(
        tmp_path,
        "key_states = key_states.to(query_states.device)",
        "key_states = self.k_proj(hidden_states)",
    )
    assert result.status == "failed"


def test_store_and_read_must_use_one_exact_key_protocol(tmp_path):
    result = _mutated_result(
        tmp_path,
        "shared_kv_states[self.layer_idx] = key_states, value_states",
        "shared_kv_states[self.kv_shared_layer_index] = key_states, value_states",
    )
    assert result.status == "failed"


def test_short_layer_type_operand_cannot_guess_a_source():
    document = AutoConfig.for_model("gemma3n_text").to_dict()
    document["layer_types"] = document["layer_types"][:-1]
    result = _result("gemma3n_text", document)
    assert result.status == "failed"


def test_schedule_closure_rejects_future_and_foreign_sources():
    value = _result("gemma3n_text").value
    with pytest.raises(ValueError):
        replace(value, decisions=(20, *value.decisions[1:]))
    with pytest.raises(ValueError):
        replace(value, decisions=(-1, *value.decisions[1:]))


def test_layer_field_schedule_is_not_a_family_table():
    value = _result("gemma3n_text").value
    assert isinstance(value, DecoderKVSharingSchedule)
    assert all("gemma" not in item.field.lower()
               for item in value.field_schedules)
