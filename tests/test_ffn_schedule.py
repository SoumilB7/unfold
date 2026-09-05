"""U8-E occurrence-exact dense/routed FFN schedule controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
from transformers import AutoConfig
from test_support import FLUX

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.ffn_schedule import (
    DecoderFFNSchedule,
    FFNLayerDecision,
    decoder_ffn_schedule_for_path,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _config(name="deepseek-v3"):
    return json.loads(
        (_CORPUS / f"{name}.json").read_text(encoding="utf-8"))[
            "config"]


def _selector(document):
    def select(path):
        current = document
        for part in tuple(path):
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"
    return select


def _real_result(name="deepseek-v3", config=None):
    config = config or _config(name)
    context = ParseContext.build(config)
    return decoder_ffn_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_selector(config))


def _source_result(tmp_path, name, source, config):
    original = ParseContext.build(config).source_bundle
    path = tmp_path / f"modeling_{name.replace('-', '_')}.py"
    path.write_text(source, encoding="utf-8")
    architecture = original.architecture
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": architecture},
        architecture=architecture)
    return decoder_ffn_schedule_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=_selector(config))


@pytest.mark.parametrize(("name", "prefix", "tail"), [
    ("deepseek-v3", ("dense", "dense", "dense"), "moe"),
    ("glm-4-5", ("dense", "dense", "dense"), "moe"),
])
def test_real_mixed_schedule_is_construction_and_mechanism_exact(
        name, prefix, tail):
    result = _real_result(name)
    assert result.status == "resolved", result.failures
    states = tuple(item.state for item in result.value.decisions)
    assert states[:3] == prefix
    assert set(states[3:]) == {tail}
    assert {item.kind for item in result.value.candidates} == {"dense", "moe"}
    assert {path for path, _kind in result.value.config_dependencies} == {
        ("num_hidden_layers",), ("first_k_dense_replace",)}


@pytest.mark.parametrize(("name", "state"), [
    ("llama-7b", "dense"),
    ("granite-3-0-8b-instruct", "dense"),
    ("gpt-oss-20b", "moe"),
    ("dbrx-base", "moe"),
])
def test_real_uniform_schedules_need_no_family_or_schedule_table(name, state):
    result = _real_result(name)
    assert result.status == "resolved", result.failures
    assert {item.state for item in result.value.decisions} == {state}
    assert {item.kind for item in result.value.candidates} == {state}


@pytest.mark.parametrize(("slot", "state"), [
    ("text_encoder", "dense"),
    ("text_encoder_2", "dense"),
])
def test_real_supporting_encoder_uniform_ffn_is_source_complete(slot, state):
    config = FLUX["_text_encoder_configs"][slot]
    result = _real_result(config=config)
    assert result.status == "resolved", result.failures
    assert {item.state for item in result.value.decisions} == {state}
    assert {item.kind for item in result.value.candidates} == {state}
    if slot == "text_encoder_2":
        calls = result.value.candidates[0].invocations
        assert len(calls) == 1
        assert calls[0].callee.source_segment == "self.layer[-1]"


def test_uniform_constructor_with_rival_site_never_defaults_dense(tmp_path):
    config = FLUX["_text_encoder_configs"]["text_encoder"]
    context = ParseContext.build(config)
    source = Path(context.source_bundle.files[0]).read_text(encoding="utf-8")
    marker = "self.mlp = CLIPMLP(config)"
    assert marker in source
    changed = source.replace(
        marker,
        "self.mlp = CLIPMLP(config) if config.unseen else CLIPMLP(config)",
        1)
    result = _source_result(tmp_path, "clip-rival", changed, config)
    assert result.status == "failed"


def test_container_ffn_slot_must_match_its_literal_invocation(tmp_path):
    config = FLUX["_text_encoder_configs"]["text_encoder_2"]
    context = ParseContext.build(config)
    source = Path(context.source_bundle.files[0]).read_text(encoding="utf-8")
    marker = "self.layer[-1]("
    assert marker in source
    changed = source.replace(marker, "self.layer[0](", 1)
    result = _source_result(tmp_path, "t5-wrong-slot", changed, config)
    assert result.status == "failed"


def test_dense_model_cannot_be_turned_into_moe_by_config_tokens():
    config = _config("llama-7b")
    config.update({
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "num_experts": 128,
        "moe_layers": list(range(config["num_hidden_layers"])),
    })
    result = _real_result("llama-7b", config)
    assert result.status == "resolved", result.failures
    assert {item.state for item in result.value.decisions} == {"dense"}
    assert {path for path, _kind in result.value.config_dependencies} == {
        ("num_hidden_layers",)}


def test_inactive_routed_candidate_remains_a_negative_schedule_premise():
    """A dense result is sound only when the exact disabled rival survives.

    Gemma4 always constructs/calls its ordinary MLP and conditionally adds a
    routed-expert branch.  Dropping the false candidate used to produce the
    right word (dense) for the wrong reason and left ``enable_moe_block``
    unaccounted for.
    """
    config = AutoConfig.for_model("gemma4_text").to_dict()
    result = _real_result(config=config)
    assert result.status == "resolved", result.failures
    assert {item.state for item in result.value.decisions} == {"dense"}
    assert {item.kind for item in result.value.candidates} == {"dense", "moe"}
    assert dict(result.value.config_dependencies)[("enable_moe_block",)] \
        == "config_declared"


def test_unknown_or_missing_selector_operand_never_defaults(tmp_path):
    config = _config("deepseek-v3")
    config["first_k_dense_replace"] = "unknown"
    result = _real_result("deepseek-v3", config)
    assert result.status == "failed"
    assert result.value is None


def test_construction_without_matching_block_invocation_is_not_an_ffn(tmp_path):
    config = _config("deepseek-v3")
    context = ParseContext.build(config)
    source = Path(context.source_bundle.files[0]).read_text(encoding="utf-8")
    marker = "hidden_states = self.mlp(hidden_states)"
    assert marker in source
    changed = source.replace(marker, "hidden_states = hidden_states", 1)
    result = _source_result(tmp_path, "deepseek-v3", changed, config)
    assert result.status == "failed"
    assert result.value is None


def test_class_and_field_renaming_preserves_the_schedule(tmp_path):
    config = _config("deepseek-v3")
    context = ParseContext.build(config)
    source = Path(context.source_bundle.files[0]).read_text(encoding="utf-8")
    changed = source
    for old, new in (
        ("DeepseekV3MoE", "ArbitraryRoutedTransform"),
        ("DeepseekV3MLP", "ArbitraryDenseTransform"),
        ("self.mlp", "self.transform"),
    ):
        changed = changed.replace(old, new)
    result = _source_result(tmp_path, "deepseek-v3", changed, config)
    assert result.status == "resolved", result.failures
    assert tuple(item.state for item in result.value.decisions) \
        == tuple(item.state for item in _real_result().value.decisions)


def test_parser_projects_only_the_exact_ffn_schedule():
    from model_unfolder import config_to_ir

    config = _config("deepseek-v3")
    context = ParseContext.build(config)
    ir = config_to_ir(config, parse_context=context)
    assert tuple(layer.ffn.kind for layer in ir.layers[:5]) == (
        "dense", "dense", "dense", "moe", "moe")
    fact = context.facts.typed["decoder.ffn.ffn_schedule"]
    assert fact.value[:5] == ("dense", "dense", "dense", "moe", "moe")
    assert fact.status == "code_and_config"
    assert set(fact.config_paths) == {
        "num_hidden_layers", "first_k_dense_replace"}


def test_result_closure_rejects_unresolved_and_mismatched_decisions():
    value = _real_result().value
    with pytest.raises(ValueError):
        replace(value, decisions=(
            FFNLayerDecision(0, "unresolved", reason="forged"),
            *value.decisions[1:]))
    with pytest.raises(ValueError):
        replace(value, decisions=(
            replace(value.decisions[0], site_id=value.decisions[3].site_id),
            *value.decisions[1:]))
    with pytest.raises(ValueError):
        replace(value, config_dependencies=tuple(
            item for item in value.config_dependencies
            if item[0] != ("first_k_dense_replace",)))
    assert isinstance(value, DecoderFFNSchedule)
