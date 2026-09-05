"""U7 exact fused routed-expert width qualification controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.expert_width import (
    decoder_expert_intermediate_width_for_path,
    decoder_shared_expert_count_for_path,
)
from model_unfolder.parser import config_to_ir
from model_unfolder.evidence.program_index import build_program_index


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _read(slug):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_expert_intermediate_width_for_path(
        context.program_index(), context.source_bundle, (), config,
        allow_root_stage=True)
    return config, context, result


def _mutated_bundle(tmp_path, context, old, new):
    source_path = Path(context.source_bundle.files[0])
    source = source_path.read_text(encoding="utf-8")
    assert old in source
    path = tmp_path / source_path.name
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    bundle = replace(
        context.source_bundle,
        files=(str(path),),
        component_files={"root": (str(path),)},
        supporting_files={},
    )
    return bundle, build_program_index(bundle)


@pytest.mark.parametrize(("slug", "expected", "path"), [
    ("deepseek-v3", 2048, ("moe_intermediate_size",)),
    ("glm-4-5", 1536, ("moe_intermediate_size",)),
    ("gpt-oss-20b", 2880, ("intermediate_size",)),
])
def test_real_fused_expert_width_uses_the_exact_shape_operand(
        slug, expected, path):
    _config, _context, result = _read(slug)
    assert result.status == "resolved", result.failures
    assert result.value.value == expected
    assert result.value.premises == ((path, expected),)
    assert result.value.spans


def test_flattened_split_storage_does_not_guess_a_per_expert_factor():
    _config, _context, result = _read("dbrx-base")
    assert result.status == "failed"
    assert result.value is None
    assert "split/flattened" in result.failures[0].detail


def test_unqualified_dbrx_width_candidate_is_visible_but_powerless():
    """A plausible config number is not an architectural width by itself.

    DBRX stores experts in flattened split matrices.  Its ``ffn_hidden_size``
    is therefore a real input the parser inspected, but neither the ordinary
    nor expert reader can prove a per-FFN projection factor from it.  Keep that
    occurrence explicitly classified without projecting it or laundering it
    into a typed geometry fact.
    """
    config, context, _result = _read("dbrx-base")
    ir = config_to_ir(config, parse_context=context)

    events = [
        event for event in context.config_access.events
        if event.config_path == "ffn_config.ffn_hidden_size"
    ]
    assert any(event.intent == "ignored" for event in events)
    assert any(
        event.intent == "ignored"
        and "no exact ordinary output-projection" in event.reason
        for event in events
    )
    assert {
        layer.ffn.intermediate_size for layer in ir.layers
    } == {None}
    assert {
        layer.ffn.expert_intermediate_size for layer in ir.layers
    } == {None}
    assert "decoder.ffn.intermediate_size" not in context.facts.typed
    assert (
        "decoder.ffn.expert.expert_intermediate_size"
        not in context.facts.typed
    )


def test_parser_projects_exact_expert_width_and_the_same_instance_fact():
    config, context, result = _read("deepseek-v3")
    assert result.status == "resolved"
    ir = config_to_ir(config, parse_context=context)
    expert_widths = {
        layer.ffn.expert_intermediate_size
        for layer in ir.layers if layer.ffn.kind == "moe"}
    assert expert_widths == {2048}
    fact = ir.extras["fact_provenance"][
        "decoder.ffn.expert.expert_intermediate_size"]
    assert fact["value"] == 2048
    assert fact["status"] == "code_and_config"
    typed = context.facts.typed[
        "decoder.ffn.expert.expert_intermediate_size"]
    assert typed.config_paths == ("moe_intermediate_size",)


def test_expert_width_dto_rejects_invalid_geometry_and_provenance():
    _config, _context, result = _read("gpt-oss-20b")
    value = result.value
    with pytest.raises(ValueError):
        replace(value, value=0)
    with pytest.raises(ValueError):
        replace(value, premises=(value.premises[0], value.premises[0]))
    with pytest.raises(ValueError):
        replace(value, premises=(((), value.value),))
    with pytest.raises(ValueError):
        replace(value, spans=())


@pytest.mark.parametrize(("slug", "width", "count_path"), [
    ("deepseek-v3", 2048, ("n_shared_experts",)),
    ("glm-4-5", 1536, ("n_shared_experts",)),
])
def test_shared_expert_count_requires_application_and_multiplicative_width(
        slug, width, count_path):
    config, context, _width_result = _read(slug)
    result = decoder_shared_expert_count_for_path(
        context.program_index(), context.source_bundle, (), config,
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.value == 1
    assert result.value.count_path == count_path
    assert dict(result.value.premises)[("moe_intermediate_size",)] == width
    assert dict(result.value.premises)[count_path] == 1


@pytest.mark.parametrize("slug", ["gpt-oss-20b", "dbrx-base"])
def test_expert_count_declaration_cannot_manufacture_a_shared_ffn(slug):
    config, context, _width_result = _read(slug)
    config["n_shared_experts"] = 99
    result = decoder_shared_expert_count_for_path(
        context.program_index(), context.source_bundle, (), config,
        allow_root_stage=True)
    assert result.status == "failed"
    ir = config_to_ir(config, parse_context=ParseContext.build(config))
    assert {
        layer.ffn.num_shared_experts
        for layer in ir.layers if layer.ffn.kind == "moe"
    } == {None}


def test_parser_expert_counts_follow_exact_source_paths_not_alias_priority():
    config, _context, _result = _read("deepseek-v3")
    config["num_experts"] = 999
    config["experts_per_token"] = 997
    config["num_shared_experts"] = 995
    context = ParseContext.build(config)
    ir = config_to_ir(config, parse_context=context)
    geometry = {
        (layer.ffn.num_experts, layer.ffn.num_experts_per_tok,
         layer.ffn.num_shared_experts)
        for layer in ir.layers if layer.ffn.kind == "moe"
    }
    assert geometry == {(256, 8, 1)}
    routing = context.facts.typed["decoder.ffn.routing_policy"].value
    assert routing["num_experts"] == 256
    assert routing["num_experts_per_tok"] == 8
    assert context.facts.typed[
        "decoder.ffn.expert.shared_expert_count"].value == 1


def test_shared_expert_count_dto_rejects_forged_count_and_provenance():
    config, context, _result = _read("deepseek-v3")
    result = decoder_shared_expert_count_for_path(
        context.program_index(), context.source_bundle, (), config,
        allow_root_stage=True)
    value = result.value
    with pytest.raises(ValueError):
        replace(value, value=0)
    with pytest.raises(ValueError):
        replace(value, count_path=("not_the_premise",))
    with pytest.raises(ValueError):
        replace(value, spans=())


def test_constructed_but_unused_shared_ffn_does_not_prove_a_count(tmp_path):
    config, context, _result = _read("deepseek-v3")
    bundle, index = _mutated_bundle(
        tmp_path, context,
        "hidden_states = hidden_states + self.shared_experts(residuals)",
        "unused_shared = self.shared_experts(residuals)",
    )
    result = decoder_shared_expert_count_for_path(
        index, bundle, (), config, allow_root_stage=True)
    assert result.status == "failed"


def test_shared_count_requires_multiplicative_width_not_a_familiar_path(
        tmp_path):
    config, context, _result = _read("deepseek-v3")
    bundle, index = _mutated_bundle(
        tmp_path, context,
        "config.moe_intermediate_size * config.n_shared_experts",
        "config.moe_intermediate_size + config.n_shared_experts",
    )
    result = decoder_shared_expert_count_for_path(
        index, bundle, (), config, allow_root_stage=True)
    assert result.status == "failed"
    assert "multiplicative" in result.failures[0].detail \
        or "two-factor product" in result.failures[0].detail
