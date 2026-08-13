"""U8-D exact construction + invocation + U6 mixer schedule controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.mixer_schedule import (
    DecoderMixerSchedule,
    MixerLayerDecision,
    decoder_mixer_schedule_for_path,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _config():
    return json.loads(
        (_CORPUS / "qwen3-5-27b-text.json").read_text(encoding="utf-8")
    )["config"]


def _selector(document):
    def select(path):
        current = document
        for part in tuple(path):
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"
    return select


def _real_result(config=None):
    config = config or _config()
    context = ParseContext.build(config)
    return decoder_mixer_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_selector(config))


def _result_from_source(tmp_path, source, config):
    path = tmp_path / "modeling_renamed_hybrid.py"
    path.write_text(source, encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Qwen3_5ForCausalLM"},
        architecture="Qwen3_5ForCausalLM")
    return decoder_mixer_schedule_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=_selector(config))


def test_real_qwen35_schedule_joins_construction_invocation_and_mechanism():
    result = _real_result()
    assert result.status == "resolved", result.failures
    value = result.value
    assert value.transport.layer_count == 64
    assert value.transport.count_config_path == ("num_hidden_layers",)
    assert tuple(item.state for item in value.decisions[:8]) == (
        "gated_delta", "gated_delta", "gated_delta", "ordinary_attention",
        "gated_delta", "gated_delta", "gated_delta", "ordinary_attention",
    )
    assert {item.kind for item in value.candidates} == {
        "ordinary_attention", "gated_delta"}
    assert {path for path, _kind in value.config_dependencies} == {
        ("num_hidden_layers",), ("layer_types",)}
    assert all(
        item.construction.selected_candidates[0].site_id
        == item.occurrence.sites[-1]
        and item.invocation.callee_owner_occurrence == item.occurrence
        for item in value.decisions)


@pytest.mark.parametrize("mutation", [
    lambda config: config["layer_types"].__setitem__(0, "unproved-token"),
    lambda config: config.__setitem__(
        "layer_types", config["layer_types"][:-1]),
])
def test_unknown_or_short_selector_never_defaults_to_a_mixer(mutation):
    config = _config()
    mutation(config)
    result = _real_result(config)
    assert result.status == "failed"
    assert result.value is None


def test_config_token_cannot_turn_a_plain_llama_block_into_gated_delta():
    config = json.loads(
        (_CORPUS / "llama-7b.json").read_text(encoding="utf-8"))["config"]
    config["layer_types"] = ["linear_attention"] * config["num_hidden_layers"]
    result = _real_result(config)
    assert result.status == "resolved", result.failures
    assert {item.state for item in result.value.decisions} == {
        "ordinary_attention"}
    assert {item.kind for item in result.value.candidates} == {
        "ordinary_attention"}


def test_constructor_and_forward_must_select_the_same_candidate(tmp_path):
    config = _config()
    context = ParseContext.build(config)
    source = Path(context.source_bundle.files[0]).read_text(encoding="utf-8")
    marker = 'if self.layer_type == "linear_attention":'
    assert source.count(marker) >= 2
    head, tail = source.rsplit(marker, 1)
    changed = head + 'if self.layer_type == "full_attention":' + tail
    result = _result_from_source(tmp_path, changed, config)
    assert result.status == "failed"
    assert result.value is None


def test_complete_class_and_field_renaming_does_not_change_schedule(tmp_path):
    config = _config()
    context = ParseContext.build(config)
    source = Path(context.source_bundle.files[0]).read_text(encoding="utf-8")
    renamed = source
    for old, new in (
        ("Qwen3_5GatedDeltaNet", "ArbitraryRecurrenceUnit"),
        ("Qwen3_5Attention", "ArbitrarySoftmaxUnit"),
        ("linear_attn", "first_lane"),
        ("self_attn", "second_lane"),
    ):
        renamed = renamed.replace(old, new)
    result = _result_from_source(tmp_path, renamed, config)
    assert result.status == "resolved", result.failures
    assert tuple(item.state for item in result.value.decisions) \
        == tuple(item.state for item in _real_result(config).value.decisions)


def test_integer_dispatch_is_only_an_operand_not_a_semantic_label(tmp_path):
    config = _config()
    context = ParseContext.build(config)
    source = Path(context.source_bundle.files[0]).read_text(encoding="utf-8")
    changed = source.replace('"linear_attention"', "0").replace(
        '"full_attention"', "1")
    config["layer_types"] = [
        0 if item == "linear_attention" else 1
        for item in config["layer_types"]]
    result = _result_from_source(tmp_path, changed, config)
    assert result.status == "resolved", result.failures
    assert tuple(item.state for item in result.value.decisions[:4]) == (
        "gated_delta", "gated_delta", "gated_delta", "ordinary_attention")


def test_same_attention_class_at_two_sites_remains_occurrence_exact(tmp_path):
    """A class identity cannot collapse two construction occurrences."""
    config = _config()
    context = ParseContext.build(config)
    source = Path(context.source_bundle.files[0]).read_text(encoding="utf-8")
    init_marker = (
        'elif self.layer_type == "full_attention":\n'
        '            self.self_attn = Qwen3_5Attention(config, layer_idx)')
    forward_marker = '''elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )'''
    assert init_marker in source and forward_marker in source
    source = source.replace(
        init_marker,
        init_marker + (
            '\n        elif self.layer_type == "alternate_attention":\n'
            '            self.other_attn = Qwen3_5Attention(config, layer_idx)'),
        1)
    # Copy the real ordinary branch, changing only its guard and exact child
    # occurrence.  Both sites instantiate the same class; the schedule must
    # retain which site is selected rather than unioning by class.
    alternate_branch = forward_marker.replace(
        'elif self.layer_type == "full_attention":',
        'elif self.layer_type == "alternate_attention":', 1).replace(
            "self.self_attn(", "self.other_attn(")
    source = source.replace(
        forward_marker, forward_marker + "\n        " + alternate_branch, 1)
    full_indices = [
        index for index, item in enumerate(config["layer_types"])
        if item == "full_attention"]
    assert len(full_indices) >= 2
    config["layer_types"][full_indices[1]] = "alternate_attention"
    result = _result_from_source(tmp_path, source, config)
    assert result.status == "resolved", result.failures
    value = result.value
    ordinary = tuple(
        item for item in value.candidates
        if item.kind == "ordinary_attention")
    assert len(ordinary) == 2
    assert ordinary[0].owner_symbol == ordinary[1].owner_symbol
    assert ordinary[0].occurrence != ordinary[1].occurrence
    selected = value.decisions[full_indices[1]]
    assert selected.state == "ordinary_attention"
    assert selected.occurrence == ordinary[1].occurrence


def test_unrecognised_candidate_remains_opaque_not_gated_delta(tmp_path):
    config = _config()
    context = ParseContext.build(config)
    source = Path(context.source_bundle.files[0]).read_text(encoding="utf-8")
    # Break the decisive framework primitive without changing the selector,
    # construction address, field, class or config token.  Those surrounding
    # hints are powerless when the candidate mechanism itself is unproved.
    assert "nn.Conv1d(" in source
    changed = source.replace("nn.Conv1d(", "OpaqueConv(", 1)
    result = _result_from_source(tmp_path, changed, config)
    assert result.status == "failed"
    assert result.value is None


def test_parser_projects_only_the_exact_mixer_schedule():
    from model_unfolder import config_to_ir

    config = _config()
    context = ParseContext.build(config)
    ir = config_to_ir(config, parse_context=context)
    assert tuple(layer.attention.kind for layer in ir.layers[:8]) == (
        "gated_delta", "gated_delta", "gated_delta", "gqa",
        "gated_delta", "gated_delta", "gated_delta", "gqa",
    )
    # Mixer placement is now exact, but it cannot fabricate a positionless
    # claim.  Position stays unknown until its independent per-occurrence
    # application schedule is cut over.
    assert {layer.attention.position_kind for layer in ir.layers} == {"unknown"}
    assert {layer.attention.rope for layer in ir.layers} == {None}
    fact = context.facts.typed["decoder.attention.mixer_schedule"]
    assert fact.value[:8] == (
        "gated_delta", "gated_delta", "gated_delta", "ordinary_attention",
        "gated_delta", "gated_delta", "gated_delta", "ordinary_attention",
    )
    assert fact.status == "code_and_config"
    assert set(fact.config_paths) == {"num_hidden_layers", "layer_types"}


def test_parser_does_not_project_config_only_mixer_tokens():
    from model_unfolder import config_to_ir

    config = json.loads(
        (_CORPUS / "llama-7b.json").read_text(encoding="utf-8"))["config"]
    config["layer_types"] = ["linear_attention"] * config["num_hidden_layers"]
    context = ParseContext.build(config)
    ir = config_to_ir(config, parse_context=context)
    assert {layer.attention.kind for layer in ir.layers} == {"mha"}
    assert context.facts.typed["decoder.attention.mixer_schedule"].value \
        == ("ordinary_attention",) * config["num_hidden_layers"]


def test_result_closure_rejects_unresolved_layer_and_foreign_occurrence():
    value = _real_result().value
    with pytest.raises(ValueError):
        replace(value, decisions=(
            MixerLayerDecision(0, "unresolved", reason="forged"),
            *value.decisions[1:],
        ))
    with pytest.raises(ValueError):
        replace(value, decisions=(
            replace(value.decisions[0], occurrence=value.block_occurrence),
            *value.decisions[1:],
        ))
    with pytest.raises(ValueError):
        first_candidate = next(
            item for item in value.candidates
            if item.kind == value.decisions[0].state
            and item.occurrence == value.decisions[0].occurrence)
        replace(value, decisions=(
            replace(
                value.decisions[0],
                construction=first_candidate.selector.decisions[1]),
            *value.decisions[1:],
        ))
    with pytest.raises(ValueError):
        replace(value, config_dependencies=tuple(
            item for item in value.config_dependencies
            if item[0] != ("layer_types",)))
    assert isinstance(value, DecoderMixerSchedule)
