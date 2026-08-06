"""U8-B positive Q/K-rotation protocol and counterexamples."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.attention_child import attention_child_evidence
from model_unfolder.evidence.attention_operands import (
    attention_qk_operands_evidence,
)
from model_unfolder.evidence.decoder_block import decoder_block_path_for_config
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
    assert result.value.rotation_callable.qualified_name == "half_turn"
    assert result.value.rotation_protocol == "split_half_turn"
    assert result.value.attention_operands.query_operand.source_segment == "q"
    assert result.value.attention_operands.key_operand.source_segment == "k"
    assert tuple(item.source_segment for item in result.value.factor_arguments) \
        == ("first_factor", "second_factor")


def test_evidence_closure_rejects_forged_operand_and_factor_links(tmp_path):
    value = _result(tmp_path, _BASE).value
    with pytest.raises(ValueError):
        replace(value, attention_operands=replace(
            value.attention_operands,
            attention_occurrence=value.block_occurrence))
    with pytest.raises(ValueError):
        replace(value, rotation_protocol="name_says_rope")
    with pytest.raises(ValueError):
        replace(value.attention_operands,
                key_operand=value.factor_arguments[0])
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


def test_exact_chunk_pair_rotation_protocol_resolves(tmp_path):
    chunk_helpers = """
def apply_one(x, direct, rotated):
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_out = first_half * direct - second_half * rotated
    second_out = second_half * direct + first_half * rotated
    return torch.cat((first_out, second_out), dim=-1)

def apply_pair(a, b, factor_a, factor_b):
    factor_a = factor_a.unsqueeze(1)
    factor_b = factor_b.unsqueeze(1)
    out_a = apply_one(a, factor_a, factor_b)
    out_b = apply_one(b, factor_a, factor_b)
    return out_a, out_b
"""
    start = _BASE.index("def apply_pair")
    end = _BASE.index("\nclass AttentionLane")
    source = _BASE[:start] + chunk_helpers + _BASE[end:]
    result = _result(tmp_path, source)
    assert result.status == "resolved"
    assert result.value.rotation_callable.qualified_name == "apply_one"
    assert result.value.rotation_protocol == "chunk_pair"


def test_chunk_pair_wrong_sign_does_not_resolve(tmp_path):
    chunk_helpers = """
def apply_one(x, direct, rotated):
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_out = first_half * direct + second_half * rotated
    second_out = second_half * direct + first_half * rotated
    return torch.cat((first_out, second_out), dim=-1)

def apply_pair(a, b, factor_a, factor_b):
    out_a = apply_one(a, factor_a, factor_b)
    out_b = apply_one(b, factor_a, factor_b)
    return out_a, out_b
"""
    start = _BASE.index("def apply_pair")
    end = _BASE.index("\nclass AttentionLane")
    source = _BASE[:start] + chunk_helpers + _BASE[end:]
    assert _result(tmp_path, source).status == "failed"


def _interleaved_source(*, wrong_sign=False):
    helper = """
def apply_interleaved(a, b, factor_a, factor_b):
    factor_a = factor_a[..., : factor_a.shape[-1] // 2].unsqueeze(1)
    factor_b = factor_b[..., : factor_b.shape[-1] // 2].unsqueeze(1)
    a0, a1 = a[..., 0::2], a[..., 1::2]
    b0, b1 = b[..., 0::2], b[..., 1::2]
    out_a = torch.cat([a0 * factor_a - a1 * factor_b,
                       a1 * factor_a + a0 * factor_b], dim=-1)
    out_b = torch.cat([b0 * factor_a - b1 * factor_b,
                       b1 * factor_a + b0 * factor_b], dim=-1)
    return out_a, out_b
"""
    if wrong_sign:
        helper = helper.replace(
            "a0 * factor_a - a1 * factor_b",
            "a0 * factor_a + a1 * factor_b")
    source = _BASE.replace(
        "import torch\n", "import torch\n" + helper + "\n")
    source = source.replace(
        "self.three = nn.Linear(config.hidden_size, config.hidden_size)",
        "self.three = nn.Linear(config.hidden_size, config.hidden_size)\n"
        "        self.config = config")
    source = source.replace(
        "q, k = apply_pair(q, k, first_factor, second_factor)",
        "if self.config.interleaved:\n"
        "            q, k = apply_interleaved(\n"
        "                q, k, first_factor, second_factor)\n"
        "        else:\n"
        "            q, k = apply_pair(q, k, first_factor, second_factor)")
    return source


@pytest.mark.parametrize(("selected", "protocol"), [
    (True, "interleaved_pair"),
    (False, "split_half_turn"),
])
def test_exact_config_guard_selects_the_proven_rotation_branch(
        tmp_path, selected, protocol):
    path = tmp_path / "modeling_protocol.py"
    path.write_text(textwrap.dedent(_interleaved_source()), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"}, architecture="Wrapper")
    result = decoder_qk_half_turn_application_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=lambda path: (
            (True, selected, "config_declared")
            if path == ("interleaved",) else (False, None, "")))
    assert result.status == "resolved", result
    assert result.value.rotation_protocol == protocol
    assert result.value.guard_config_paths == (("interleaved",),)


def test_missing_guard_operand_cannot_choose_a_rotation_branch(tmp_path):
    path = tmp_path / "modeling_protocol.py"
    path.write_text(textwrap.dedent(_interleaved_source()), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"}, architecture="Wrapper")
    result = decoder_qk_half_turn_application_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=lambda _path: (False, None, ""))
    assert result.status == "failed"


def test_selected_interleaved_branch_requires_exact_rotation_algebra(tmp_path):
    path = tmp_path / "modeling_protocol.py"
    path.write_text(
        textwrap.dedent(_interleaved_source(wrong_sign=True)), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"}, architecture="Wrapper")
    result = decoder_qk_half_turn_application_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=lambda path: (
            (True, True, "config_declared")
            if path == ("interleaved",) else (False, None, "")))
    assert result.status == "failed"


def _complex_pair_source():
    helper = """
def apply_pair(a, b, phase):
    a_complex = torch.view_as_complex(
        a.float().reshape(*a.shape[:-1], -1, 2))
    b_complex = torch.view_as_complex(
        b.float().reshape(*b.shape[:-1], -1, 2))
    out_a = torch.view_as_real(
        a_complex * phase[:, :, None, :]).flatten(3)
    out_b = torch.view_as_real(
        b_complex * phase[:, :, None, :]).flatten(3)
    return out_a.type_as(a), out_b.type_as(b)
"""
    start = _BASE.index("def apply_pair")
    end = _BASE.index("\nclass AttentionLane")
    source = _BASE[:start] + helper + _BASE[end:]
    return source.replace(
        "apply_pair(q, k, first_factor, second_factor)",
        "apply_pair(q, k, first_factor)")


def test_exact_complex_pair_rotation_protocol_resolves(tmp_path):
    result = _result(tmp_path, _complex_pair_source())
    assert result.status == "resolved", result
    assert result.value.rotation_protocol == "complex_pair"
    assert result.value.factor_parameter_indices == (2,)
    assert tuple(item.source_segment for item in result.value.factor_arguments) \
        == ("first_factor",)


@pytest.mark.parametrize("old,new", [
    ("torch.view_as_complex(\n        a.float()", "torch.as_tensor(\n        a.float()"),
    ("a.float().reshape(*a.shape[:-1], -1, 2)",
     "a.float().reshape(*a.shape[:-1], -1, 4)"),
    ("a_complex * phase[:, :, None, :]",
     "a_complex + phase[:, :, None, :]"),
    ("b_complex * phase[:, :, None, :]",
     "b_complex * phase[:, :, :, 0]"),
    (").flatten(3)", ").flatten(2)"),
])
def test_complex_pair_protocol_rejects_inexact_algebra(tmp_path, old, new):
    source = _complex_pair_source().replace(old, new, 1)
    assert source != _complex_pair_source()
    assert _result(tmp_path, source).status == "failed"


def test_real_llama4_complex_rotation_and_layer_guard():
    import inspect

    from transformers import AutoConfig
    from transformers.models.llama4 import modeling_llama4

    path = inspect.getsourcefile(modeling_llama4)
    bundle = SourceBundle(
        source="local", files=(path,),
        component_files={"root": (path,)},
        component_architectures={"root": "Llama4TextModel"},
        architecture="Llama4TextModel")
    index = build_program_index(bundle)
    config = AutoConfig.for_model("llama4").text_config

    def select(parts):
        current = config
        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    return False, None, ""
                current = current[part]
            elif not hasattr(current, part):
                return False, None, ""
            else:
                current = getattr(current, part)
        return True, current, "config_declared"

    active = decoder_qk_half_turn_application_for_path(
        index, bundle, (), allow_root_stage=True,
        config_selector=select,
        constructor_parameter_values={"layer_idx": 0})
    inactive = decoder_qk_half_turn_application_for_path(
        index, bundle, (), allow_root_stage=True,
        config_selector=select,
        constructor_parameter_values={"layer_idx": 3})
    assert active.status == "resolved", active
    assert active.value.rotation_protocol == "complex_pair"
    assert active.value.guard_config_paths == (("no_rope_layers",),)
    assert inactive.status == "failed"


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


def test_rotation_proof_does_not_require_simple_projection_storage(tmp_path):
    source = _BASE.replace(
        "q = self.one(hidden)\n        k = self.two(hidden)\n"
        "        v = self.three(hidden)",
        "if hidden.shape[-1] > 1:\n"
        "            q = self.one(hidden)\n"
        "        else:\n"
        "            q = self.two(hidden)\n"
        "        packed = self.three(hidden)\n"
        "        k = packed[..., : packed.shape[-1] // 2]\n"
        "        v = packed[..., packed.shape[-1] // 2 :]")
    result = _result(tmp_path, source)
    assert result.status == "resolved"


def test_unpack_targets_must_be_two_distinct_score_lanes(tmp_path):
    source = _BASE.replace(
        "q, k = apply_pair(q, k, first_factor, second_factor)",
        "q, q = apply_pair(q, k, first_factor, second_factor)")
    assert _result(tmp_path, source).status == "failed"


def test_rotating_query_and_value_cannot_masquerade_as_query_and_key(tmp_path):
    source = _BASE.replace(
        "scaled_dot_product_attention(q, k, v)",
        "scaled_dot_product_attention(q, v, k)")
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


def test_real_gpt_oss_chunk_pair_control_resolves():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext

    context = ParseContext.build(AutoConfig.for_model("gpt_oss"))
    result = decoder_qk_half_turn_application_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result
    assert result.value.rotation_protocol == "chunk_pair"


def test_real_deepseek_score_operands_resolve_without_projection_storage():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext

    context = ParseContext.build(AutoConfig.for_model("deepseek_v3"))
    index = context.program_index()
    block = decoder_block_path_for_config(
        index, context.source_bundle, (), allow_root_stage=True)
    assert block.status == "resolved"
    attention = attention_child_evidence(
        index, block.value.component_root, block.value.block_occurrence)
    assert attention.status == "resolved"
    operands = attention_qk_operands_evidence(
        index, block.value.component_root, attention.value)
    assert operands.status == "resolved", operands
    assert operands.value.protocol == "dot_softmax"
    assert tuple(item.source_segment for item in (
        operands.value.query_operand, operands.value.key_operand)) == (
            "query_states", "key_states")


def test_real_deepseek_interleaved_application_uses_exact_config_guard():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext

    config = AutoConfig.for_model("deepseek_v3")
    context = ParseContext.build(config)

    def select(path):
        if path == ("rope_interleave",):
            return True, config.rope_interleave, "config_declared"
        return False, None, ""

    result = decoder_qk_half_turn_application_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=select)
    assert result.status == "resolved", result
    assert result.value.rotation_protocol == "interleaved_pair"
    assert result.value.guard_config_paths == (("rope_interleave",),)
