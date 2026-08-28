"""U10-E exact root-bookend and 3-D-operation controls."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_block import read_diffusion_block_facts
from model_unfolder.evidence.diffusion_bookends import (
    read_diffusion_bookends,
)
from model_unfolder.evidence.diffusion_conditioning import (
    read_diffusion_conditioning_graph,
)
from model_unfolder.evidence.diffusion_stack import read_diffusion_stack_inventory
from model_unfolder.evidence.diffusion_stream import read_diffusion_stream_graph
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


PREFIX = """
import torch
from torch import nn
from torch.nn import functional as F

class Mixer:
    def __init__(self, config):
        self.q = nn.Linear(config.width, config.width)
        self.k = nn.Linear(config.width, config.width)
        self.v = nn.Linear(config.width, config.width)
    def forward(self, x):
        return F.scaled_dot_product_attention(self.q(x), self.k(x), self.v(x))

class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.gate = nn.Linear(config.width, config.width)
    def forward(self, state, condition):
        scale = self.gate(condition)
        delta = self.mix(state)
        state = state + scale * delta
        return state
"""


ROOT = """
class Root:
    def __init__(self, config):
        self.enter = nn.Conv3d(config.in_channels, config.width, 1)
        self.condition = nn.Linear(config.width, config.width)
        self.units = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.leave = nn.Conv3d(config.width, config.out_channels, 1)
    def forward(self, sample, timestep):
        state = self.enter(sample)
        condition = self.condition(timestep)
        for unit in self.units:
            state = unit(state, condition)
        return self.leave(state)
"""


def _write(tmp_path, source):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _read(tmp_path, root_source=ROOT):
    path = _write(tmp_path, PREFIX + root_source)
    bundle = SourceBundle(
        source="test", files=(path,), architecture="Root",
        component_files={"root": (path,)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stacks = read_diffusion_stack_inventory(index, root)
    blocks = read_diffusion_block_facts(index, root, stacks)
    streams = read_diffusion_stream_graph(index, root, blocks)
    conditioning = read_diffusion_conditioning_graph(index, root, streams)
    result = read_diffusion_bookends(
        index, root, stacks, streams, conditioning)
    return result, (root, stacks, streams, conditioning), index


def test_exact_input_output_and_conditioning_routes_are_separate(tmp_path):
    result, _deps, _index = _read(tmp_path)
    assert result.status == "incomplete"
    rows = result.require_value().applications
    assert [(item.role, tuple(op.kind for op in item.operations))
            for item in rows] == [
                ("conditioning_input", ("linear",)),
                ("state_input", ("conv3d",)),
                ("state_output", ("conv3d",)),
            ]
    assert len({item.call.span for item in rows}) == 3
    assert [
        (item.role,
         tuple((dimension.operation_kind, dimension.dimension_role,
                dimension.expression.source_segment)
               for dimension in item.dimension_operands))
        for item in rows
    ] == [
        ("conditioning_input", (
            ("linear", "input_width", "config.width"),
            ("linear", "output_width", "config.width"),
        )),
        ("state_input", (
            ("conv3d", "input_channels", "config.in_channels"),
            ("conv3d", "output_channels", "config.width"),
        )),
        ("state_output", (
            ("conv3d", "input_channels", "config.width"),
            ("conv3d", "output_channels", "config.out_channels"),
        )),
    ]


def test_conv3d_is_source_proven_3d_not_a_config_video_claim(tmp_path):
    result, _deps, _index = _read(tmp_path)
    rows = result.require_value().temporal_operations
    assert [(item.kind, item.application.role) for item in rows] == [
        ("three_dimensional_convolution", "state_input"),
        ("three_dimensional_convolution", "state_output"),
    ]
    assert not hasattr(result.require_value(), "video")
    assert not hasattr(result.require_value(), "temporal_config")


def test_three_entry_patch_config_without_source_operation_changes_nothing(tmp_path):
    # The evidence API has no config input.  This is the anti-fabrication law:
    # no patch tuple can enter the source-only result at all.
    plain = ROOT.replace("nn.Conv3d", "nn.Linear")
    result, _deps, _index = _read(tmp_path, plain)
    assert result.require_value().temporal_operations == ()


def test_rank_five_reshape_is_geometry_not_a_temporal_operation(tmp_path):
    source = ROOT.replace(
        "state = self.enter(sample)",
        "state = self.enter(sample)\n        state = state.reshape(1, 2, 3, 4, 5)")
    result, _deps, _index = _read(tmp_path, source)
    value = result.require_value()
    assert [item.kind for item in value.temporal_operations].count(
        "three_dimensional_convolution") == 2
    assert [item.kind for item in value.tensor_geometry] == ["rank_five_shape"]


def test_reshape_rank_four_is_not_promoted_to_a_3d_operation(tmp_path):
    source = ROOT.replace(
        "state = self.enter(sample)",
        "state = self.enter(sample)\n        state = state.reshape(1, 2, 3, 4)")
    result, _deps, _index = _read(tmp_path, source)
    assert result.require_value().tensor_geometry == ()


def test_unused_projection_is_not_a_bookend(tmp_path):
    source = ROOT.replace(
        "state = self.enter(sample)",
        "unused = self.enter(sample)\n        state = sample")
    result, _deps, _index = _read(tmp_path, source)
    rows = result.require_value().applications
    assert not any(item.role == "state_input" and
                   any(op.kind == "conv3d" for op in item.operations)
                   for item in rows)


def test_output_call_not_reaching_return_is_not_a_bookend(tmp_path):
    source = ROOT.replace(
        "return self.leave(state)",
        "discarded = self.leave(state)\n        return state")
    result, _deps, _index = _read(tmp_path, source)
    assert not any(item.role == "state_output"
                   for item in result.require_value().applications)


def test_condition_projection_not_used_by_proven_gate_is_not_conditioning(tmp_path):
    source = ROOT.replace(
        "condition = self.condition(timestep)",
        "unused = self.condition(timestep)\n        condition = timestep")
    result, _deps, _index = _read(tmp_path, source)
    assert not any(item.role == "conditioning_input"
                   for item in result.require_value().applications)


def test_complete_class_field_and_local_rename_preserves_routes(tmp_path):
    renamed = (ROOT.replace("Root", "Opaque")
               .replace("enter", "alpha")
               .replace("condition", "beta")
               .replace("units", "gamma")
               .replace("leave", "delta")
               .replace("state", "carrier")
               .replace("sample", "source_value")
               .replace("timestep", "side_value")
               .replace("unit", "element"))
    # Architecture is address metadata only; adjust it without changing the
    # helper's mechanism assertions.
    path = _write(tmp_path, PREFIX + renamed)
    bundle = SourceBundle(
        source="test", files=(path,), architecture="Opaque",
        component_files={"root": (path,)},
        component_architectures={"root": "Opaque"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stacks = read_diffusion_stack_inventory(index, root)
    blocks = read_diffusion_block_facts(index, root, stacks)
    streams = read_diffusion_stream_graph(index, root, blocks)
    cond = read_diffusion_conditioning_graph(index, root, streams)
    result = read_diffusion_bookends(index, root, stacks, streams, cond)
    assert [(item.role, tuple(op.kind for op in item.operations))
            for item in result.require_value().applications] == [
                ("conditioning_input", ("linear",)),
                ("state_input", ("conv3d",)),
                ("state_output", ("conv3d",)),
            ]


def test_dependency_belongs_to_another_root_is_rejected(tmp_path):
    first, deps, index = _read(tmp_path / "a")
    _other, other_deps, _other_index = _read(tmp_path / "b")
    root, stacks, streams, conditioning = deps
    out = read_diffusion_bookends(
        index, root, other_deps[1], streams, conditioning)
    assert out.status == "failed"
    assert out.failures[0].kind == "out_of_owner"
    assert first.has_value


def test_dto_rejects_forged_role_and_dependency(tmp_path):
    result, _deps, _index = _read(tmp_path)
    value = result.require_value()
    row = value.applications[0]
    with pytest.raises(ValueError, match="role"):
        replace(row, role="patchify")
    with pytest.raises(ValueError, match="stack"):
        replace(row, owner_occurrence=row.stack_execution.block_occurrence)
    temporal = value.temporal_operations[0]
    with pytest.raises(ValueError, match="temporal"):
        replace(temporal, kind="rank_five_shape")


def test_geometry_cannot_be_forged_into_temporal_mechanism(tmp_path):
    source = ROOT.replace(
        "state = self.enter(sample)",
        "state = self.enter(sample)\n        state = state.reshape(1, 2, 3, 4, 5)")
    result, _deps, _index = _read(tmp_path, source)
    geometry = result.require_value().tensor_geometry[0]
    with pytest.raises(ValueError, match="geometry"):
        replace(geometry, kind="three_dimensional_convolution")


def test_source_missing_dependencies_stay_typed_failure(tmp_path):
    result, deps, index = _read(tmp_path)
    root, stacks, streams, conditioning = deps
    failed = replace(stacks, status="failed", value=None,
                     completeness="none", failures=stacks.failures[:1],
                     provenance=())
    out = read_diffusion_bookends(
        index, root, failed, streams, conditioning)
    assert out.status == "failed"
    assert result.has_value


def test_parser_shadow_publishes_the_exact_same_bookend_object(tmp_path):
    from model_unfolder.adapters.diffusor.parser import (
        _shadow_diffusion_bookends,
    )
    from model_unfolder.evidence.context import ParseContext

    path = _write(tmp_path, PREFIX + ROOT)
    bundle = SourceBundle(
        source="test", files=(path,), architecture="Root",
        component_files={"root": (path,)},
        component_architectures={"root": "Root"})
    context = ParseContext(source_bundle=bundle)
    first = _shadow_diffusion_bookends(context)
    second = _shadow_diffusion_bookends(context)
    assert first is second
    assert context.reader_results[("root.denoiser.bookends", ())] is first


@lru_cache(maxsize=32)
def _real_bookend_counts(witness):
    import model_unfolder as mu
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    data = json.loads((corpus / f"{witness}.json").read_text())
    context = ParseContext.build(_coerce(data.get("config") or data))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    stacks = read_diffusion_stack_inventory(index, root)
    blocks = read_diffusion_block_facts(index, root, stacks)
    if not blocks.has_value:
        return (0, 0, 0, 0)
    streams = read_diffusion_stream_graph(index, root, blocks)
    conditioning = read_diffusion_conditioning_graph(index, root, streams)
    result = read_diffusion_bookends(
        index, root, stacks, streams, conditioning)
    roles = [item.role for item in result.require_value().applications]
    return (roles.count("state_input"), roles.count("state_output"),
            roles.count("conditioning_input"),
            len(result.require_value().temporal_operations))


@pytest.mark.parametrize("witness,expected", [
    ("auraflow-v0-3", (7, 0, 0, 0)),
    ("cogvideox-5b", (0, 2, 0, 0)),
    ("flux-2-dev", (5, 3, 3, 0)),
    ("fluxtransformer2dmodel", (3, 2, 0, 0)),
    ("hunyuanvideo", (0, 0, 0, 0)),
    ("ltx-video", (1, 2, 4, 0)),
    ("lumina-image-2-0", (5, 0, 0, 0)),
    ("mochi-1-preview", (3, 1, 0, 0)),
    ("pixart-sigma-xl-2-1024-ms", (0, 0, 0, 0)),
    ("prxpixel-t2i", (1, 1, 2, 0)),
    ("qwen-image", (2, 1, 0, 0)),
    ("sana-1600m-1024px-diffusers", (0, 2, 0, 0)),
    ("stable-diffusion-3-5-large", (0, 0, 0, 0)),
    ("stable-diffusion-xl-base-1-0", (0, 0, 0, 0)),
    ("wan2-2-t2v-a14b-diffusers", (3, 1, 0, 1)),
])
def test_real_diffusion_bookend_matrix_is_an_honest_lower_bound(
        witness, expected):
    assert _real_bookend_counts(witness) == expected
