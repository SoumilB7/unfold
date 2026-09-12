"""U10-F1 closed source-projection controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.adapters.diffusor.schema import project_diffusion_source
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_block import read_diffusion_block_facts
from model_unfolder.evidence.diffusion_bookends import read_diffusion_bookends
from model_unfolder.evidence.diffusion_conditioning import (
    read_diffusion_conditioning_graph,
)
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.diffusion_stack import read_diffusion_stack_inventory
from model_unfolder.evidence.diffusion_stream import read_diffusion_stream_graph
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reader_result import ReaderFailure, ReaderResult


SOURCE = """
import torch
from torch import nn
from torch.nn import functional as F

class Mixer:
    def __init__(self, config):
        self.width = config.hidden // config.query_heads
        self.q = nn.Linear(config.hidden, config.query_heads * self.width)
        self.k = nn.Linear(config.hidden, config.kv_heads * self.width)
        self.v = nn.Linear(config.hidden, config.kv_heads * self.width)
    def forward(self, state, context):
        query = self.q(state)
        key = self.k(context)
        value = self.v(context)
        return F.scaled_dot_product_attention(query, key, value)

class FeedForward:
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden, config.wide)
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, state):
        return self.down(F.silu(self.gate(state)) * self.up(state))

class Block:
    def __init__(self, config):
        self.first_norm = nn.LayerNorm(config.hidden)
        self.mix = Mixer(config)
        self.second_norm = nn.LayerNorm(config.hidden)
        self.ffn = FeedForward(config)
    def forward(self, state, context):
        state = state + self.mix(self.first_norm(state), context)
        state = state + self.ffn(self.second_norm(state))
        return state

class Root:
    def __init__(self, config):
        self.enter = nn.Linear(config.in_channels, config.hidden)
        self.units = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
        self.leave = nn.Linear(config.hidden, config.out_channels)
    def forward(self, sample, context):
        state = self.enter(sample)
        state = state.reshape(1, 2, 3, 4, 5)
        for unit in self.units:
            state = unit(state, context)
        return self.leave(state)
"""


CONFIG = {
    "hidden": 64,
    "wide": 128,
    "query_heads": 8,
    "kv_heads": 2,
    "layers": 4,
    "in_channels": 16,
    "out_channels": 16,
}


def _write(tmp_path, source):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _selector(document, *, with_status=False):
    def select(path):
        value = document
        for part in path:
            if not isinstance(value, dict) or part not in value:
                return (False, None, "") if with_status else None
            value = value[part]
        return ((True, value, "config_declared")
                if with_status else value)
    return select


def _read(tmp_path, source=SOURCE, *, architecture="Root", config=CONFIG):
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="test", files=(path,), architecture=architecture,
        component_files={"root": (path,)},
        component_architectures={"root": architecture})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.address_resolved
    topology = read_diffusion_root_topology(index, root)
    stacks = read_diffusion_stack_inventory(index, root)
    blocks = read_diffusion_block_facts(
        index, root, stacks, config_document=config,
        config_value_selector=_selector(config),
        config_guard_selector=_selector(config, with_status=True))
    streams = read_diffusion_stream_graph(index, root, blocks)
    conditioning = read_diffusion_conditioning_graph(index, root, streams)
    bookends = read_diffusion_bookends(
        index, root, stacks, streams, conditioning)
    companions = ReaderResult.absent(root.graph.root.occurrence)
    result = project_diffusion_source(
        topology, blocks, streams, conditioning, bookends, companions)
    return result, (topology, blocks, streams, conditioning, bookends)


def test_projection_copies_exact_protocols_without_inventing_diagram_kinds(
        tmp_path):
    result, _dependencies = _read(tmp_path)
    assert result.status == "incomplete"
    value = result.require_value()
    assert len(value.blocks) == 1
    block = value.blocks[0]
    assert block.norm_kind == "layernorm"
    assert block.ffn is not None
    assert block.ffn.gated is True
    assert block.ffn.projection_mode == "split"
    lane = block.attention[0]
    assert lane.compute_protocol == "scaled_dot_product_attention"
    assert lane.head_protocol == "grouped_kv"
    assert lane.projection_storage == "split"
    assert lane.stream_kind == "contextual_single_state"
    assert not hasattr(lane, "attention_kind")
    assert not hasattr(lane, "mha")


def test_rank_five_geometry_never_becomes_a_temporal_mechanism(tmp_path):
    result, _dependencies = _read(tmp_path)
    value = result.require_value()
    assert value.tensor_geometry_kinds == ("rank_five_shape",)
    assert value.temporal_operation_kinds == ()
    assert not hasattr(value, "video")
    assert not hasattr(value, "temporal_axis")


def test_same_block_class_in_two_stacks_remains_two_occurrences(tmp_path):
    source = SOURCE.replace(
        "self.leave = nn.Linear(config.hidden, config.out_channels)",
        "self.more = nn.ModuleList(\n"
        "            [Block(config) for _ in range(config.other_layers)])\n"
        "        self.leave = nn.Linear(config.hidden, config.out_channels)")
    source = source.replace(
        "return self.leave(state)",
        "for other in self.more:\n"
        "            state = other(state, context)\n"
        "        return self.leave(state)")
    result, dependencies = _read(
        tmp_path, source, config={**CONFIG, "other_layers": 2})
    value = result.require_value()
    assert len(value.blocks) == 2
    assert len({item.block_occurrence for item in value.blocks}) == 2
    assert len({item.evidence.stack.block_symbol for item in value.blocks}) == 1
    with pytest.raises(ValueError, match="authorities disagree"):
        replace(value.blocks[0], stream=value.blocks[1].stream)
    assert len(dependencies[1].require_value().blocks) == 2


def test_unresolved_stack_is_preserved_not_converted_to_a_default_block(tmp_path):
    source = SOURCE.replace(
        "self.units = nn.ModuleList(\n"
        "            [Block(config) for _ in range(config.layers)])",
        "self.units = nn.ModuleList(\n"
        "            [factory(config) for _ in range(config.layers)])")
    result, _dependencies = _read(tmp_path, source)
    value = result.require_value()
    assert value.blocks == ()
    assert value.unresolved_stacks


def test_complete_class_field_and_local_rename_preserves_projection(tmp_path):
    original, _dependencies = _read(tmp_path / "original")
    renamed = (SOURCE.replace("Root", "Opaque")
               .replace("Block", "Cell")
               .replace("Mixer", "Kernel")
               .replace("FeedForward", "DensePath")
               .replace("units", "sequence")
               .replace("unit", "element")
               .replace("first_norm", "alpha")
               .replace("second_norm", "beta")
               .replace("mix", "combine")
               .replace("ffn", "transform"))
    changed, _changed_dependencies = _read(
        tmp_path / "renamed", renamed, architecture="Opaque")

    def signature(result):
        value = result.require_value()
        block = value.blocks[0]
        lane = block.attention[0]
        return (
            value.topology_kind,
            lane.compute_protocol, lane.head_protocol, lane.stream_kind,
            block.norm_kind, block.norm_placement, block.residual_topology,
            block.ffn.gated, block.ffn.projection_mode,
            value.temporal_operation_kinds, value.tensor_geometry_kinds,
        )

    assert signature(changed) == signature(original)


def test_missing_dependencies_remain_typed_failure():
    missing = ReaderResult.failed(None, (ReaderFailure(
        "missing_source", "source bundle is unavailable"),))
    result = project_diffusion_source(
        missing, missing, missing, missing, missing, ReaderResult.absent())
    assert result.status == "failed"
    assert result.failures[0].kind == "missing_source"


def test_parser_publishes_projection_call_locally_without_ir_authority(tmp_path):
    from model_unfolder.adapters.diffusor.parser import (
        _shadow_diffusion_source_projection,
    )
    from model_unfolder.evidence.context import ParseContext

    path = _write(tmp_path, SOURCE)
    bundle = SourceBundle(
        source="test", files=(path,), architecture="Root",
        component_files={"root": (path,)},
        component_architectures={"root": "Root"})
    context = ParseContext(source_bundle=bundle)
    first = _shadow_diffusion_source_projection(context)
    second = _shadow_diffusion_source_projection(context)
    assert first is second
    assert first.status == "incomplete"
    assert context.reader_results[
        ("root.denoiser.source_projection", ())] is first
    assert not hasattr(first.require_value(), "ir")
    assert not hasattr(first.require_value(), "render")


@pytest.mark.parametrize(("witness", "blocks", "unresolved"), (
    ("cogvideox-5b", 1, 0),
    ("flux-2-dev", 2, 0),
    ("hunyuanvideo", 1, 2),
    ("sana-1600m-1024px-diffusers", 1, 2),
    ("stable-diffusion-xl-base-1-0", 0, 4),
    ("wan2-2-t2v-a14b-diffusers", 1, 1),
))
def test_real_diffusion_projection_is_occurrence_exact_and_config_powerless(
        witness, blocks, unresolved):
    import model_unfolder as mu
    from model_unfolder.adapters.diffusor.parser import (
        _shadow_diffusion_source_projection,
    )
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    data = json.loads((corpus / f"{witness}.json").read_text())
    context = ParseContext.build(_coerce(data.get("config") or data))
    before = len(context.config_access.events)
    result = _shadow_diffusion_source_projection(context)
    assert result.status == "incomplete"
    projection = result.require_value()
    assert len(projection.blocks) == blocks
    assert len(projection.unresolved_stacks) == unresolved
    assert len({item.block_occurrence for item in projection.blocks}) == blocks
    assert len(context.config_access.events) == before
