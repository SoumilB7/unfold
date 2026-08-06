"""U8-B positive Q/K-rotation protocol and counterexamples."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_application import (
    decoder_qk_half_turn_application_for_path,
)
from model_unfolder.evidence.program_index import build_program_index


_BASE = """
import torch
from torch import nn

def half_turn(x):
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)

def apply_pair(a, b, factor_a, factor_b):
    factor_a = factor_a.unsqueeze(1)
    factor_b = factor_b.unsqueeze(1)
    out_a = (a * factor_a) + (half_turn(a) * factor_b)
    out_b = (b * factor_a) + (half_turn(b) * factor_b)
    return out_a, out_b

class AttentionLane(nn.Module):
    def __init__(self, config):
        self.one = nn.Linear(config.hidden_size, config.hidden_size)
        self.two = nn.Linear(config.hidden_size, config.hidden_size)
        self.three = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden, first_factor, second_factor):
        q = self.one(hidden)
        k = self.two(hidden)
        v = self.three(hidden)
        q, k = apply_pair(q, k, first_factor, second_factor)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

class Cell(nn.Module):
    def __init__(self, config):
        self.lane = AttentionLane(config)

    def forward(self, hidden, first_factor, second_factor):
        return self.lane(hidden, first_factor, second_factor)

class Stage(nn.Module):
    def __init__(self, config):
        self.cells = nn.ModuleList(
            [Cell(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden, first_factor, second_factor):
        for cell in self.cells:
            hidden = cell(hidden, first_factor, second_factor)
        return hidden

class Wrapper(nn.Module):
    base_model_prefix = "model"

    def __init__(self, config):
        self.model = Stage(config)
"""


def _result(tmp_path, source):
    path = tmp_path / "modeling_protocol.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = build_program_index(bundle)
    return decoder_qk_half_turn_application_for_path(
        index, bundle, (), allow_root_stage=True)


def test_exact_two_lane_half_turn_protocol_resolves(tmp_path):
    result = _result(tmp_path, _BASE)
    assert result.status == "resolved"
    assert result.value.helper_callable.qualified_name == "apply_pair"
    assert result.value.half_turn_callable.qualified_name == "half_turn"
    assert len(result.value.qk_projection_sources) == 2
    assert tuple(item.source_segment for item in result.value.factor_arguments) \
        == ("first_factor", "second_factor")


def test_evidence_closure_rejects_forged_storage_and_factor_links(tmp_path):
    value = _result(tmp_path, _BASE).value
    with pytest.raises(ValueError):
        replace(value, storage_mode="fused_qkv")
    with pytest.raises(ValueError):
        replace(value, qk_projection_sources=(
            value.qk_projection_sources[0], value.qk_projection_sources[0]))
    with pytest.raises(ValueError):
        replace(value, factor_arguments=tuple(reversed(value.factor_arguments)))
    with pytest.raises(ValueError):
        replace(value, spans=tuple(
            span for span in value.spans
            if span != value.application_call.span))


def test_complete_function_class_field_and_local_rename_still_resolves(tmp_path):
    renamed = (_BASE
               .replace("half_turn", "arbitrary_permutation")
               .replace("apply_pair", "combine_values")
               .replace("AttentionLane", "OpaqueUnit")
               .replace("self.one", "self.alpha")
               .replace("self.two", "self.beta")
               .replace("self.three", "self.gamma")
               .replace("        q = self.alpha(hidden)",
                        "        left = self.alpha(hidden)")
               .replace("        k = self.beta(hidden)",
                        "        right = self.beta(hidden)")
               .replace("        v = self.gamma(hidden)",
                        "        payload = self.gamma(hidden)")
               .replace("q, k = combine_values(q, k,",
                        "left, right = combine_values(left, right,")
               .replace("(q, k, v)", "(left, right, payload)"))
    result = _result(tmp_path, renamed)
    assert result.status == "resolved"


def test_tempting_helper_name_with_wrong_math_does_not_resolve(tmp_path):
    source = _BASE.replace(
        "return torch.cat((-second, first), dim=-1)", "return x")
    result = _result(tmp_path, source)
    assert result.status == "failed"


def test_only_one_rotated_lane_does_not_resolve(tmp_path):
    source = _BASE.replace(
        "out_b = (b * factor_a) + (half_turn(b) * factor_b)",
        "out_b = b")
    result = _result(tmp_path, source)
    assert result.status == "failed"


def test_half_turn_must_preserve_the_exact_half_order(tmp_path):
    source = _BASE.replace(
        "return torch.cat((-second, first), dim=-1)",
        "return torch.cat((-first, second), dim=-1)")
    assert _result(tmp_path, source).status == "failed"


def test_empty_or_overlapping_half_slices_do_not_resolve(tmp_path):
    source = _BASE.replace(
        "first = x[..., : x.shape[-1] // 2]",
        "first = x[..., x.shape[-1] // 2 : x.shape[-1] // 2]")
    assert _result(tmp_path, source).status == "failed"


def test_factor_expression_cannot_launder_an_unrelated_origin(tmp_path):
    source = _BASE.replace(
        "factor_a = factor_a.unsqueeze(1)",
        "factor_a = factor_a.unsqueeze(1) + a")
    assert _result(tmp_path, source).status == "failed"


def test_rotation_outputs_must_reach_both_attention_operands(tmp_path):
    source = _BASE.replace(
        "q, k = apply_pair(q, k, first_factor, second_factor)\n"
        "        return torch.nn.functional.scaled_dot_product_attention(q, k, v)",
        "q, k = apply_pair(q, k, first_factor, second_factor)\n"
        "        q = self.one(hidden)\n"
        "        return torch.nn.functional.scaled_dot_product_attention(q, k, v)")
    result = _result(tmp_path, source)
    assert result.status == "failed"


def test_rotation_must_consume_exact_projection_lanes(tmp_path):
    source = _BASE.replace(
        "q, k = apply_pair(q, k, first_factor, second_factor)",
        "q, k = apply_pair(hidden, hidden, first_factor, second_factor)")
    result = _result(tmp_path, source)
    assert result.status == "failed"


def test_each_rotation_lane_has_one_exact_projection_origin(tmp_path):
    source = _BASE.replace(
        "q, k = apply_pair(q, k, first_factor, second_factor)",
        "q, k = apply_pair(q + k, k, first_factor, second_factor)")
    assert _result(tmp_path, source).status == "failed"


def test_valid_protocol_in_uninvoked_sibling_is_not_laundered(tmp_path):
    source = _BASE.replace(
        "q, k = apply_pair(q, k, first_factor, second_factor)",
        "q, k = q, k")
    result = _result(tmp_path, source)
    assert result.status == "failed"


def test_two_real_application_paths_are_ambiguous_not_ranked(tmp_path):
    source = _BASE.replace(
        "q, k = apply_pair(q, k, first_factor, second_factor)",
        "q, k = apply_pair(q, k, first_factor, second_factor)\n"
        "        q, k = apply_pair(q, k, first_factor, second_factor)")
    assert _result(tmp_path, source).status == "ambiguous"


def test_partial_source_cannot_prove_a_negative_or_positive(tmp_path):
    source = _BASE.replace("class AttentionLane", "class AttentionLane(")
    result = _result(tmp_path, source)
    assert result.status == "failed"


def test_bloom_is_a_negative_control_not_mislabelled_as_rotation():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext

    context = ParseContext.build(AutoConfig.for_model("bloom"))
    result = decoder_qk_half_turn_application_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "failed"


def test_real_llama_gemma_qwen_and_olmo_controls_resolve():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext

    for model_type in ("llama", "gemma2", "qwen3", "olmo2"):
        context = ParseContext.build(AutoConfig.for_model(model_type))
        result = decoder_qk_half_turn_application_for_path(
            context.program_index(), context.source_bundle, (),
            allow_root_stage=True)
        assert result.status == "resolved", (model_type, result)
