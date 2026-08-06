"""U7 exact-owner norm placement and residual-equation controls."""
from __future__ import annotations

from dataclasses import replace
import math
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.cell_topology import (
    decoder_cell_topology_for_path,
)
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _source(block_forward, *, norm_fields=None, attention_return="return out",
            ffn_return="return out", helper="",
            attention_params="signal, residual=None",
            ffn_params="signal, residual=None", cell_fields="",
            extra_classes="", attention_class="Compute"):
    norm_fields = norm_fields or """
        self.first = nn.LayerNorm(config.hidden)
        self.second = nn.LayerNorm(config.hidden)
"""
    return f"""
import torch
from torch import nn
from torch.nn import functional as F

{helper}

class Compute:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, {attention_params}):
        q = self.q(signal)
        k = self.k(signal)
        v = self.v(signal)
        score = torch.matmul(q, k.transpose(-1, -2))
        out = torch.matmul(F.softmax(score, dim=-1), v)
        {attention_return}

class Transform:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, {ffn_params}):
        out = self.down(self.act(self.up(signal)))
        {ffn_return}

{extra_classes}

class Cell:
    def __init__(self, config):
        self.compute = {attention_class}(config)
        self.transform = Transform(config)
{textwrap.indent(textwrap.dedent(cell_fields).strip(), " " * 8) if cell_fields else ""}
{norm_fields}
    def forward(self, signal):
{block_forward}

class Body:
    def __init__(self, config):
        self.stack = nn.ModuleList(
            [Cell(config) for _ in range(config.layers)])
    def forward(self, signal):
        for cell in self.stack:
            signal = cell(signal)
        return signal

class Outer:
    base_model_prefix = "body"
    def __init__(self, config):
        self.body = Body(config)
"""


def _read(
        tmp_path, block_forward, *, selected_config=None,
        selected_source_kinds=None, **kwargs):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(_source(
        textwrap.indent(textwrap.dedent(block_forward).strip(), " " * 8),
        **kwargs)), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Outer"}, architecture="Outer")
    missing = object()

    def selected_value(path):
        current = selected_config
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return missing
            current = current[part]
        return current

    def select(path):
        value = selected_value(path)
        return None if value is missing else value

    def select_guard(path):
        value = selected_value(path)
        return (
            value is not missing,
            None if value is missing else value,
            (selected_source_kinds or {}).get(
                tuple(path), "config_declared"),
        )

    return decoder_cell_topology_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=(select if selected_config is not None else None),
        guard_config_selector=(
            select_guard if selected_config is not None else None))


_GUARDED_AUGMENTED_CELL = """
residual = signal
if self.config.new_decoder_architecture and self.config.num_norms == 2:
    attention_input = self.attention_norm(signal)
    mlp_input = self.mlp_norm(signal)
else:
    attention_input = self.input_norm(signal)
attention_output = self.compute(attention_input)
if not self.config.new_decoder_architecture:
    if self.config.parallel:
        mlp_input = attention_input
    else:
        residual = residual + attention_output
        mlp_input = self.post_norm(residual)
if self.config.new_decoder_architecture and self.config.parallel and self.config.num_norms == 1:
    mlp_input = attention_input
mlp_output = self.transform(mlp_input)
if self.config.new_decoder_architecture or self.config.parallel:
    mlp_output += attention_output
output = residual + mlp_output
return output
"""


_GUARDED_NORMS = """
        self.config = config
        self.attention_norm = nn.LayerNorm(config.hidden)
        self.mlp_norm = nn.LayerNorm(config.hidden)
        self.input_norm = nn.LayerNorm(config.hidden)
        self.post_norm = nn.LayerNorm(config.hidden)
"""


@pytest.mark.parametrize(("selected", "topology"), (
    ({"new_decoder_architecture": False, "parallel": True, "num_norms": 1},
     "parallel"),
    ({"new_decoder_architecture": False, "parallel": False, "num_norms": 1},
     "sequential"),
))
def test_nested_config_guards_and_augmented_add_prove_exact_cell(
        tmp_path, selected, topology):
    result = _read(
        tmp_path, _GUARDED_AUGMENTED_CELL,
        selected_config=selected, norm_fields=_GUARDED_NORMS)
    assert result.status == "resolved", result.failures
    assert result.value.norm_placement == "pre"
    assert result.value.residual_topology == topology
    assert set(result.value.residual_config_paths) == {
        ("new_decoder_architecture",), ("parallel",),
    }
    assert result.value.norm_config_paths == result.value.residual_config_paths


def test_nested_config_guards_never_select_without_exact_values(tmp_path):
    result = _read(
        tmp_path, _GUARDED_AUGMENTED_CELL,
        selected_config={"parallel": True}, norm_fields=_GUARDED_NORMS)
    assert result.status == "failed"
    assert any("unresolved guard" in failure.detail
               for failure in result.failures)


_SCALED_SEQUENTIAL = """
residual = signal
signal = self.first(signal)
attention_output = self.compute(signal)
signal = residual + attention_output * self.residual_multiplier
residual = signal
signal = self.second(signal)
ffn_output = self.transform(signal)
signal = residual + ffn_output * self.residual_multiplier
return signal
"""


def test_exact_residual_equations_bind_their_config_scale(tmp_path):
    result = _read(
        tmp_path, _SCALED_SEQUENTIAL,
        selected_config={"residual_multiplier": 0.22},
        cell_fields="self.residual_multiplier = config.residual_multiplier")
    assert result.status == "resolved", result.failures
    assert result.value.residual_scale_path == ("residual_multiplier",)
    assert result.value.residual_scale_value is None
    assert result.value.residual_scale_spans


def test_unused_residual_multiplier_declaration_is_powerless(tmp_path):
    result = _read(
        tmp_path, _SCALED_SEQUENTIAL.replace(
            " * self.residual_multiplier", ""),
        selected_config={"residual_multiplier": 0.22},
        cell_fields="self.residual_multiplier = config.residual_multiplier")
    assert result.status == "resolved", result.failures
    assert result.value.residual_scale_path is None
    assert result.value.residual_scale_value is None


def test_one_scaled_branch_cannot_manufacture_two_scale_connectors(tmp_path):
    result = _read(
        tmp_path, _SCALED_SEQUENTIAL.replace(
            "ffn_output * self.residual_multiplier", "ffn_output"),
        selected_config={"residual_multiplier": 0.22},
        cell_fields="self.residual_multiplier = config.residual_multiplier")
    assert result.status == "resolved", result.failures
    assert result.value.residual_scale_path is None
    assert result.value.residual_scale_value is None


def test_equal_source_literal_scales_are_code_proven(tmp_path):
    result = _read(
        tmp_path, _SCALED_SEQUENTIAL.replace(
            "self.residual_multiplier", "0.5"))
    assert result.status == "resolved", result.failures
    assert result.value.residual_scale_path is None
    assert result.value.residual_scale_value == 0.5
    assert result.value.residual_scale_spans


def test_rival_scale_operands_cannot_be_collapsed(tmp_path):
    source = _SCALED_SEQUENTIAL.replace(
        "attention_output * self.residual_multiplier",
        "attention_output * self.attention_scale",
    ).replace(
        "ffn_output * self.residual_multiplier",
        "ffn_output * self.ffn_scale",
    )
    result = _read(
        tmp_path, source,
        selected_config={"attention_scale": 0.2, "ffn_scale": 0.3},
        cell_fields="""
        self.attention_scale = config.attention_scale
        self.ffn_scale = config.ffn_scale
        """)
    assert result.status == "resolved", result.failures
    assert result.value.residual_scale_path is None
    assert result.value.residual_scale_value is None


def test_residual_scale_evidence_shape_is_closed(tmp_path):
    result = _read(
        tmp_path, _SCALED_SEQUENTIAL,
        selected_config={"residual_multiplier": 0.22},
        cell_fields="self.residual_multiplier = config.residual_multiplier")
    assert result.status == "resolved", result.failures
    with pytest.raises(ValueError, match="config-bound or source-literal"):
        replace(result.value, residual_scale_value=0.22)
    with pytest.raises(ValueError, match="retains exact source spans"):
        replace(result.value, residual_scale_spans=())
    with pytest.raises(TypeError, match="scale value is numeric"):
        replace(result.value, residual_scale_path=None,
                residual_scale_value=math.inf)


def test_exact_constructor_config_normalization_feeds_forward_guards(tmp_path):
    result = _read(
        tmp_path, _GUARDED_AUGMENTED_CELL,
        selected_config={
            "new_decoder_architecture": True,
            "parallel": True,
            "num_norms": None,
        },
        cell_fields="""
        if config.num_norms is None and config.new_decoder_architecture:
            config.num_norms = 2
        """,
        norm_fields=_GUARDED_NORMS)
    assert result.status == "resolved", result.failures
    assert (result.value.norm_placement,
            result.value.residual_topology,
            result.value.parallel_input_norm_count) \
        == ("pre", "parallel", 2)
    assert set(result.value.norm_config_paths) == {
        ("new_decoder_architecture",), ("num_norms",), ("parallel",),
    }


def test_dynamic_constructor_config_mutation_cannot_select_a_cell(tmp_path):
    result = _read(
        tmp_path, _GUARDED_AUGMENTED_CELL,
        selected_config={
            "new_decoder_architecture": True,
            "parallel": True,
            "num_norms": None,
        },
        cell_fields="""
        if config.num_norms is None and config.new_decoder_architecture:
            config.num_norms = choose_norm_count()
        """,
        norm_fields=_GUARDED_NORMS)
    assert result.status == "failed"


def test_constructor_normalization_cannot_execute_after_selected_return(
        tmp_path):
    result = _read(
        tmp_path, _GUARDED_AUGMENTED_CELL,
        selected_config={
            "skip_setup": True,
            "new_decoder_architecture": True,
            "parallel": True,
            "num_norms": None,
        },
        cell_fields="""
        if config.skip_setup:
            return
        config.num_norms = 2
        """,
        norm_fields=_GUARDED_NORMS)
    assert result.status == "failed"


def test_constructor_normalization_executes_after_proven_inactive_return(
        tmp_path):
    result = _read(
        tmp_path, _GUARDED_AUGMENTED_CELL,
        selected_config={
            "skip_setup": False,
            "new_decoder_architecture": True,
            "parallel": True,
            "num_norms": None,
        },
        cell_fields="""
        if config.skip_setup:
            return
        config.num_norms = 2
        """,
        norm_fields=_GUARDED_NORMS)
    assert result.status == "resolved", result.failures
    assert result.value.parallel_input_norm_count == 2


def test_constructor_normalization_inside_unsupported_region_is_unknown(
        tmp_path):
    result = _read(
        tmp_path, _GUARDED_AUGMENTED_CELL,
        selected_config={
            "new_decoder_architecture": True,
            "parallel": True,
            "num_norms": None,
        },
        cell_fields="""
        try:
            config.num_norms = 2
        finally:
            pass
        """,
        norm_fields=_GUARDED_NORMS)
    assert result.status == "failed"


def test_guard_dependencies_retain_class_default_origin_in_reader_result(
        tmp_path):
    result = _read(
        tmp_path, _GUARDED_AUGMENTED_CELL,
        selected_config={
            "new_decoder_architecture": False,
            "parallel": True,
            "num_norms": 1,
        },
        selected_source_kinds={
            ("new_decoder_architecture",): "class_default",
            ("num_norms",): "class_default",
        },
        norm_fields=_GUARDED_NORMS)
    assert result.status == "resolved", result.failures
    kinds = dict(result.value.config_source_kinds)
    assert kinds[("new_decoder_architecture",)] == "class_default"
    assert kinds[("parallel",)] == "config_declared"


def test_non_add_augmented_assignment_cannot_prove_residual_merge(tmp_path):
    result = _read(
        tmp_path,
        """
        residual = signal
        attention_output = self.compute(self.first(signal))
        mlp_output = self.transform(self.second(signal))
        mlp_output *= attention_output
        return residual + mlp_output
        """)
    assert result.status == "failed"


def test_parallel_norm_count_cannot_disagree_with_exact_branch_sites(tmp_path):
    result = _read(
        tmp_path,
        """
        residual = signal
        shared = self.first(signal)
        attention_output = self.compute(shared)
        mlp_output = self.transform(shared)
        return residual + attention_output + mlp_output
        """)
    assert result.status == "resolved", result.failures
    assert result.value.parallel_input_norm_count == 1
    with pytest.raises(ValueError, match="exact branch norm sites"):
        replace(result.value, parallel_input_norm_count=2)


@pytest.mark.parametrize(("forward", "placement"), (
    ("""
    residual = signal
    signal = self.first(signal)
    signal = self.compute(signal)
    signal = residual + signal
    residual = signal
    signal = self.second(signal)
    signal = self.transform(signal)
    signal = residual + signal
    return signal
    """, "pre"),
    ("""
    residual = signal
    signal = self.compute(signal)
    signal = self.first(signal)
    signal = residual + signal
    residual = signal
    signal = self.transform(signal)
    signal = self.second(signal)
    signal = residual + signal
    return signal
    """, "post"),
    ("""
    residual = signal
    signal = self.first(signal)
    signal = self.compute(signal)
    signal = self.second(signal)
    signal = residual + signal
    residual = signal
    signal = self.first(signal)
    signal = self.transform(signal)
    signal = self.second(signal)
    signal = residual + signal
    return signal
    """, "double"),
))
def test_sequential_placement_is_two_positive_residual_equations(
        tmp_path, forward, placement):
    result = _read(tmp_path, forward)
    assert result.status == "resolved", result.failures
    assert (result.value.norm_placement,
            result.value.residual_topology) == (placement, "sequential")


def test_parallel_requires_shared_input_and_one_three_term_merge(tmp_path):
    result = _read(tmp_path, """
    normalized = self.first(signal)
    left = self.compute(normalized)
    right = self.transform(normalized)
    return signal + left + right
    """)
    assert result.status == "resolved", result.failures
    assert result.value.norm_placement == "pre"
    assert result.value.residual_topology == "parallel"
    assert result.value.attention.merge_span == result.value.ffn.merge_span


def test_two_calls_in_one_lexical_segment_are_not_parallel_without_common_merge(
        tmp_path):
    result = _read(tmp_path, """
    left = self.compute(self.first(signal))
    first_result = signal + left
    right = self.transform(self.second(first_result))
    return first_result + right
    """)
    assert result.status == "resolved", result.failures
    assert result.value.residual_topology == "sequential"


def test_dead_parallel_looking_equation_cannot_certify_the_return(tmp_path):
    result = _read(tmp_path, """
    normalized = self.first(signal)
    left = self.compute(normalized)
    right = self.transform(normalized)
    unused = signal + left + right
    return signal
    """)
    assert result.status == "failed"
    assert "residual equation" in result.failures[0].detail


def test_guarded_rival_ffn_paths_remain_unresolved(tmp_path):
    result = _read(tmp_path, """
    normalized = self.first(signal)
    left = self.compute(normalized)
    if self.training:
        right = self.transform(normalized)
        return signal + left + right
    right = self.transform(self.second(signal))
    return signal + left + right
    """)
    assert result.status == "failed"


def test_guarded_only_norm_definition_cannot_become_unconditional(tmp_path):
    result = _read(tmp_path, """
    if self.training:
        normalized = self.first(signal)
    left = self.compute(normalized)
    first_result = signal + left
    right = self.transform(self.second(first_result))
    return first_result + right
    """)
    assert result.status == "failed"


def test_child_integrated_add_is_proved_from_exact_child_and_helper(tmp_path):
    result = _read(
        tmp_path,
        """
        normalized = self.first(signal)
        signal = self.compute(normalized, signal)
        normalized = self.second(signal)
        return self.transform(normalized, signal)
        """,
        helper="""
def combine(alpha, beta):
    return beta + alpha
""",
        attention_return="return combine(out, residual)",
        ffn_return="return combine(out, residual)")
    assert result.status == "resolved", result.failures
    assert result.value.residual_topology == "sequential"
    assert {
        result.value.attention.merge_kind,
        result.value.ffn.merge_kind,
    } == {"child_integrated_add"}


def test_two_calls_to_one_residual_helper_remain_distinct_occurrences(tmp_path):
    """One helper source span does not mean one runtime residual equation."""
    result = _read(
        tmp_path,
        """
        residual = signal
        attention_input = self.first(signal)
        attention_output = self.compute(attention_input)
        residual = combine(attention_output, residual)
        mlp_input = self.second(residual)
        mlp_output = self.transform(mlp_input)
        output = combine(mlp_output, residual)
        return output
        """,
        helper="""
def combine(value, residual):
    kept = F.dropout(value, p=0.0)
    output = residual + kept
    return output
""")
    assert result.status == "resolved", result.failures
    assert (result.value.norm_placement,
            result.value.residual_topology) == ("pre", "sequential")
    assert result.value.attention.merge_span \
        == result.value.ffn.merge_span
    assert result.value.attention.invocation.node.call_site \
        != result.value.ffn.invocation.node.call_site


def test_keyword_residual_cannot_shift_into_an_omitted_optional_formal(tmp_path):
    result = _read(
        tmp_path,
        """
        normalized = self.first(signal)
        signal = self.compute(normalized, residual=signal)
        normalized = self.second(signal)
        return self.transform(normalized, residual=signal)
        """,
        attention_params="signal, optional=None, residual=None",
        ffn_params="signal, optional=None, residual=None",
        attention_return="return out + residual",
        ffn_return="return out + residual")
    assert result.status == "resolved", result.failures
    assert result.value.residual_topology == "sequential"


@pytest.mark.parametrize("child_return", (
    "return residual + residual",
    "return out + residual + signal",
))
def test_child_add_requires_one_residual_and_one_computed_term(
        tmp_path, child_return):
    result = _read(
        tmp_path,
        """
        normalized = self.first(signal)
        signal = self.compute(normalized, signal)
        normalized = self.second(signal)
        return self.transform(normalized, signal)
        """,
        attention_return=child_return,
        ffn_return=child_return)
    assert result.status == "failed"


def test_dead_helper_add_cannot_certify_a_child_residual(tmp_path):
    result = _read(
        tmp_path,
        """
        normalized = self.first(signal)
        signal = self.compute(normalized, signal)
        normalized = self.second(signal)
        return self.transform(normalized, signal)
        """,
        helper="""
def combine(alpha, beta):
    unused = beta + alpha
    return alpha
""",
        attention_return="return combine(out, residual)",
        ffn_return="return combine(out, residual)")
    assert result.status == "failed"


def test_dead_child_integrated_ffn_cannot_certify_the_block_return(tmp_path):
    result = _read(
        tmp_path,
        """
        normalized = self.first(signal)
        signal = self.compute(normalized, signal)
        normalized = self.second(signal)
        unused = self.transform(normalized, signal)
        return signal
        """,
        attention_return="return out + residual",
        ffn_return="return out + residual")
    assert result.status == "failed"
    assert "residual equation" in result.failures[0].detail


@pytest.mark.parametrize("live_merge", (True, False))
def test_nested_attention_norm_wrapper_requires_its_live_residual_equation(
        tmp_path, live_merge):
    merge = (
        "signal = signal + residual" if live_merge
        else "unused = signal + residual")
    returned = "signal" if live_merge else "residual"
    wrapper = f"""
class Composite:
    def __init__(self, config):
        self.before = nn.LayerNorm(config.hidden)
        self.inner = Compute(config)
        self.after = nn.LayerNorm(config.hidden)
    def forward(self, signal):
        residual = signal
        signal = self.before(signal)
        signal = self.inner(signal)
        {merge}
        return {returned}, self.after({returned})
"""
    result = _read(
        tmp_path,
        """
        residual, normalized = self.compute(signal)
        transformed = self.transform(normalized)
        return residual + transformed
        """,
        extra_classes=wrapper,
        attention_class="Composite")
    if live_merge:
        assert result.status == "resolved", result.failures
        assert (result.value.norm_placement,
                result.value.residual_topology) == ("pre", "sequential")
    else:
        assert result.status == "failed"


def test_unrelated_sibling_residual_equations_cannot_vote(tmp_path):
    result = _read(
        tmp_path,
        """
        residual = signal
        signal = self.first(signal)
        signal = self.compute(signal)
        signal = residual + signal
        residual = signal
        signal = self.second(signal)
        signal = self.transform(signal)
        signal = residual + signal
        return signal
        """,
        helper="""
class MisleadingSibling:
    def forward(self, signal, left, right):
        return signal + left + right
""")
    assert result.status == "resolved"
    assert result.value.residual_topology == "sequential"


def test_opaque_transform_on_branch_boundary_is_not_negative_norm_proof(
        tmp_path):
    result = _read(tmp_path, """
    normalized = self.first(signal)
    left = self.compute(normalized)
    left = torch.clone(left)
    first_result = signal + left
    right = self.transform(self.second(first_result))
    return first_result + right
    """)
    assert result.status == "failed"


def test_result_types_reject_forged_topology_and_provenance(tmp_path):
    result = _read(tmp_path, """
    residual = signal
    signal = self.first(signal)
    signal = self.compute(signal)
    signal = residual + signal
    residual = signal
    signal = self.second(signal)
    signal = self.transform(signal)
    signal = residual + signal
    return signal
    """)
    value = result.value
    with pytest.raises(ValueError, match="derives"):
        replace(value, norm_placement="post")
    with pytest.raises(ValueError, match="closed residual topology"):
        replace(value, residual_topology="mostly_sequential")
    with pytest.raises(ValueError, match="provenance is closed"):
        replace(value, spans=value.spans[:-1])
    with pytest.raises(ValueError, match="closed mechanism"):
        replace(value.attention, mechanism="mixer")
    with pytest.raises(ValueError, match="distinct exact calls"):
        replace(
            value.attention,
            post_norm_site=value.attention.pre_norm_site)


@pytest.mark.parametrize(("slug", "placement", "merge_kind"), (
    ("llama-7b", "pre", "block_add"),
    ("gemma-2-2b-it", "double", "block_add"),
    ("olmo-2-1124-7b", "post", "block_add"),
    ("bloom", "pre", "child_integrated_add"),
    ("qwen3-8b", "pre", "block_add"),
))
def test_real_source_controls_prove_exact_sequential_cells(
        slug, placement, merge_kind):
    cfg = json.loads((_CORPUS / f"{slug}.json").read_text())["config"]
    context = ParseContext.build(cfg, source="local")
    result = decoder_cell_topology_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.norm_placement == placement
    assert result.value.residual_topology == "sequential"
    assert result.value.attention.merge_kind == merge_kind
    assert result.value.ffn.merge_kind == merge_kind


def test_real_conditional_parallel_selector_is_not_guessed_without_operand():
    cfg = json.loads(
        (_CORPUS / "stablelm-2-1-6b.json").read_text())["config"]
    context = ParseContext.build(cfg, source="local")
    result = decoder_cell_topology_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "failed"
    assert result.failures[0].kind == "conflict"


@pytest.mark.parametrize(("selected", "topology"), (
    (False, "sequential"),
    (True, "parallel"),
))
def test_real_conditional_parallel_selector_uses_exact_source_bound_operand(
        selected, topology):
    cfg = json.loads(
        (_CORPUS / "stablelm-2-1-6b.json").read_text())["config"]
    cfg["use_parallel_residual"] = selected
    context = ParseContext.build(cfg, source="local")
    result = decoder_cell_topology_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True,
        config_selector=lambda path: cfg[path[0]])
    assert result.status == "resolved", result.failures
    assert (result.value.norm_placement,
            result.value.residual_topology) == ("pre", topology)
    assert result.value.norm_config_paths == ()
    assert result.value.residual_config_paths == (("use_parallel_residual",),)
    assert any(
        provenance.config_paths == (("use_parallel_residual",),)
        for provenance in result.provenance)


def test_real_falcon_source_proves_all_guarded_cell_variants():
    """One installed source contains sequential and one/two-norm parallel cells."""
    base = {
        "architectures": ["FalconForCausalLM"],
        "model_type": "falcon",
        "hidden_size": 4544,
        "intermediate_size": 18176,
        "num_hidden_layers": 32,
        "num_attention_heads": 71,
        "num_kv_heads": 1,
        "multi_query": True,
    }
    context = ParseContext.build(base, source="local")
    index = context.program_index()
    variants = (
        (False, True, None, "parallel", 1),
        (False, False, None, "sequential", None),
        # The exact constructor normalizes absent/None to two norms for the
        # new architecture before the same config object reaches forward().
        (True, True, None, "parallel", 2),
        (True, True, 1, "parallel", 1),
        (True, True, 2, "parallel", 2),
    )
    for new_architecture, parallel, norm_count, topology, expected_count \
            in variants:
        selected = {
            **base,
            "new_decoder_architecture": new_architecture,
            "parallel_attn": parallel,
            "num_ln_in_parallel_attn": norm_count,
        }

        def choose(path):
            current = selected
            for part in path:
                if not isinstance(current, dict) or part not in current:
                    return False, None, ""
                current = current[part]
            return True, current, "config_declared"

        result = decoder_cell_topology_for_path(
            index, context.source_bundle, (), allow_root_stage=True,
            config_selector=choose, guard_config_selector=choose)
        assert result.status == "resolved", (
            new_architecture, parallel, norm_count, result.failures)
        assert (result.value.norm_placement,
                result.value.residual_topology,
                result.value.parallel_input_norm_count) \
            == ("pre", topology, expected_count)


def _installed_transformers_topology(relative, architecture, **kwargs):
    import transformers

    source = Path(transformers.__file__).parent / "models" / relative
    if not source.exists():
        pytest.skip(f"{relative} modeling source is unavailable")
    bundle = SourceBundle(
        source="local",
        files=(str(source),),
        component_files={"root": (str(source),)},
        component_architectures={"root": architecture},
        architecture=architecture,
    )
    return decoder_cell_topology_for_path(
        build_program_index(bundle), bundle, (),
        allow_root_stage=True, **kwargs)


def test_real_gptj_source_proves_one_shared_parallel_input_norm():
    result = _installed_transformers_topology(
        "gptj/modeling_gptj.py", "GPTJForCausalLM")
    assert result.status == "resolved", result.failures
    assert (result.value.norm_placement,
            result.value.residual_topology,
            result.value.parallel_input_norm_count) \
        == ("pre", "parallel", 1)


def test_real_gpt_neox_selected_path_proves_two_parallel_input_norms():
    selected = {"use_parallel_residual": True}
    result = _installed_transformers_topology(
        "gpt_neox/modeling_gpt_neox.py", "GPTNeoXForCausalLM",
        config_selector=lambda path: selected.get(path[0]),
        guard_config_selector=lambda path: (
            path[0] in selected,
            selected.get(path[0]),
            "config_declared",
        ),
    )
    assert result.status == "resolved", result.failures
    assert (result.value.norm_placement,
            result.value.residual_topology,
            result.value.parallel_input_norm_count) \
        == ("pre", "parallel", 2)
    assert result.value.residual_config_paths == (
        ("use_parallel_residual",),)


def test_real_routed_ffn_is_still_the_exact_sequential_cell_stage():
    cfg = json.loads(
        (_CORPUS / "gpt-oss-20b.json").read_text())["config"]
    context = ParseContext.build(cfg, source="local")
    result = decoder_cell_topology_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.ffn.mechanism == "routed_ffn"
    assert (result.value.norm_placement,
            result.value.residual_topology) == ("pre", "sequential")


def test_split_expert_and_nested_attention_cell_is_proved_recursively():
    cfg = json.loads(
        (_CORPUS / "dbrx-base.json").read_text())["config"]
    context = ParseContext.build(cfg, source="local")
    result = decoder_cell_topology_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert (result.value.norm_placement,
            result.value.residual_topology) == ("pre", "sequential")
    assert result.value.attention.merge_kind == "block_add"
    assert result.value.ffn.mechanism == "routed_ffn"


def test_qwen35_guarded_topology_requires_its_exact_declared_domain():
    cfg = json.loads(
        (_CORPUS / "qwen3-5-27b-text.json").read_text())["config"]
    context = ParseContext.build(cfg, source="local")
    result = decoder_cell_topology_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"
    assert "selector" in result.failures[0].detail


def test_qwen35_attention_and_recurrent_variants_prove_one_cell_shell():
    cfg = json.loads(
        (_CORPUS / "qwen3-5-27b-text.json").read_text())["config"]
    context = ParseContext.build(cfg, source="local")
    result = decoder_cell_topology_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True,
        config_selector=lambda path: cfg[path[0]])
    assert result.status == "resolved", result.failures
    assert (result.value.norm_placement,
            result.value.residual_topology) == ("pre", "sequential")
    assert {item.mechanism for item in result.value.mixers} \
        == {"attention", "gated_delta_mixer"}
    assert result.value.norm_config_paths == (("layer_types",),)
    assert result.value.residual_config_paths == (("layer_types",),)
    assert any(
        provenance.config_paths == (("layer_types",),)
        for provenance in result.provenance)


def test_qwen35_unknown_declared_variant_cannot_be_laundered_into_known_shell():
    cfg = json.loads(
        (_CORPUS / "qwen3-5-27b-text.json").read_text())["config"]
    context = ParseContext.build(cfg, source="local")
    result = decoder_cell_topology_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True,
        config_selector=lambda _path: ["full_attention", "future_mixer"])
    assert result.status == "failed"
    assert "selector value" in result.failures[0].detail


def test_dynamic_config_subscript_cannot_become_a_per_layer_selector(tmp_path):
    result = _read(
        tmp_path,
        """
        if self.mode == "left":
            branch = self.compute(self.first(signal))
        elif self.mode == "right":
            branch = self.compute(self.first(signal))
        first_result = signal + branch
        transformed = self.transform(self.second(first_result))
        return first_result + transformed
        """,
        cell_fields="""
        self.runtime_key = config.runtime_key
        self.mode = config.layer_types[self.runtime_key]
        """)
    assert result.status != "resolved"
