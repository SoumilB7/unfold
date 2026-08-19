"""U10-D exact local diffusion-stream controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_block import read_diffusion_block_facts
from model_unfolder.evidence.diffusion_stack import read_diffusion_stack_inventory
from model_unfolder.evidence.diffusion_stream import (
    DiffusionStreamInventory,
    read_diffusion_stream_graph,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


PREFIX = """
import torch
from torch import nn
from torch.nn import functional as F
"""

SELF_MIXER = """
class Mixer:
    def __init__(self, config):
        self.q = nn.Linear(config.width, config.width)
        self.k = nn.Linear(config.width, config.width)
        self.v = nn.Linear(config.width, config.width)
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return F.scaled_dot_product_attention(q, k, v)
"""

CROSS_MIXER = """
class Mixer:
    def __init__(self, config):
        self.q = nn.Linear(config.width, config.width)
        self.k = nn.Linear(config.width, config.width)
        self.v = nn.Linear(config.width, config.width)
    def forward(self, x, other):
        q = self.q(x)
        k = self.k(other)
        v = self.v(other)
        return F.scaled_dot_product_attention(q, k, v)
"""


def _write(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _read(tmp_path, block, *, mixer=CROSS_MIXER, root_args="x, context",
          block_args="x, context"):
    source = PREFIX + mixer + block + f"""
class Root:
    def __init__(self, config):
        self.units = nn.ModuleList([Block(config) for _ in range(config.layers)])
    def forward(self, {root_args}):
        for unit in self.units:
            x = unit({block_args})
        return x
"""
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="test", files=(path,), architecture="Root",
        component_files={"root": (path,)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stacks = read_diffusion_stack_inventory(index, root)
    blocks = read_diffusion_block_facts(index, root, stacks)
    result = read_diffusion_stream_graph(index, root, blocks)
    return result, blocks, root, index


def test_nonreturned_second_operand_is_context_not_a_second_state(tmp_path):
    result, _blocks, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, x, context):
        mixed = self.mix(x, context)
        x = x + mixed
        return x
""")
    row = result.require_value().blocks[0]
    assert row.relations, row.unresolved[0].reason
    relation = row.relations[0]
    assert relation.kind == "contextual_single_state"
    assert relation.state_formals == ("x",)
    assert relation.context_formals == ("context",)
    assert [(item.formal.name, item.role) for item in row.roots] == [
        ("x", "state"), ("context", "context")]


def test_two_returned_roots_make_an_exact_dual_state_lane(tmp_path):
    result, _blocks, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, left, right):
        delta = self.mix(left, right)
        left = left + delta
        right = right + delta
        return left, right
""", root_args="left, right", block_args="left, right")
    relation = result.require_value().blocks[0].relations[0]
    assert relation.kind == "dual_state"
    assert relation.state_formals == ("left", "right")
    assert relation.context_formals == ()


def test_exact_concat_is_joined_but_storage_order_is_not_execution(tmp_path):
    result, _blocks, _root, _index = _read(
        tmp_path, """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, left, right):
        joined = torch.cat([left, right], dim=1)
        delta = self.mix(joined)
        left = left + delta
        return left
""", mixer=SELF_MIXER, root_args="left, right", block_args="left, right")
    relation = result.require_value().blocks[0].relations[0]
    assert relation.kind == "joined_inputs"
    assert relation.joins[0].input_formals == ("left", "right")
    assert relation.state_formals == ("left",)
    assert relation.auxiliary_formals == ("right",)
    assert not hasattr(relation, "execution_complete")
    assert not hasattr(relation, "modality")


def test_self_lane_with_decoy_dimension_and_unused_text_is_single_state(tmp_path):
    result, _blocks, _root, _index = _read(
        tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.cross_attention_dim = config.cross_attention_dim
    def forward(self, state, unused_text):
        delta = self.mix(state)
        state = state + delta
        return state
""", mixer=SELF_MIXER, root_args="state, unused_text",
        block_args="state, unused_text")
    row = result.require_value().blocks[0]
    assert row.relations[0].kind == "single_state"
    assert [item.formal.name for item in row.roots] == ["state"]


def test_guarded_rival_reaching_definitions_stay_unresolved(tmp_path):
    result, _blocks, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, x, context, choose):
        if choose:
            lane = x
        else:
            lane = context
        mixed = self.mix(lane, context)
        x = x + mixed
        return x
""", root_args="x, context, choose", block_args="x, context, choose")
    row = result.require_value().blocks[0]
    assert not row.relations
    assert len(row.unresolved) == 1
    assert "rival" in row.unresolved[0].reason


def test_invoked_but_unused_lane_cannot_claim_a_stream_relation(tmp_path):
    result, _blocks, _root, _index = _read(
        tmp_path, """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, state, unused_text):
        ignored = self.mix(state)
        return state
""", mixer=SELF_MIXER, root_args="state, unused_text",
        block_args="state, unused_text")
    row = result.require_value().blocks[0]
    assert not row.relations
    assert len(row.unresolved) == 1
    assert "return route" in row.unresolved[0].reason


def test_reassigned_return_slot_follows_its_actual_source(tmp_path):
    result, _blocks, _root, _index = _read(
        tmp_path, """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, state, replacement):
        ignored = self.mix(state)
        state = replacement
        return state
""", mixer=SELF_MIXER, root_args="state, replacement",
        block_args="state, replacement")
    row = result.require_value().blocks[0]
    assert not row.relations
    assert [item.formal.name for item in row.roots
            if item.role == "state"] == ["replacement"]


def test_guarded_state_preserving_return_transform_keeps_the_state(tmp_path):
    result, _blocks, _root, _index = _read(
        tmp_path, """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, state, clamp):
        state = state + self.mix(state)
        if clamp:
            state = state.clip(-1, 1)
        return state
""", mixer=SELF_MIXER, root_args="state, clamp",
        block_args="state, clamp")
    row = result.require_value().blocks[0]
    assert row.relations[0].kind == "single_state"
    assert not row.returns[0].unresolved


def test_source_rename_preserves_relation_while_source_change_does_not(tmp_path):
    block = """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, alpha, beta):
        out = self.mix(alpha, beta)
        alpha = alpha + out
        return alpha
"""
    first, *_ = _read(
        tmp_path, block, root_args="alpha, beta", block_args="alpha, beta")
    renamed = block.replace("alpha", "u").replace("beta", "v")
    second, *_ = _read(
        tmp_path, renamed, root_args="u, v", block_args="u, v")
    assert first.require_value().blocks[0].relations[0].kind \
        == second.require_value().blocks[0].relations[0].kind \
        == "contextual_single_state"

    changed = renamed.replace("self.mix(u, v)", "self.mix(u, u)")
    third, *_ = _read(
        tmp_path, changed, root_args="u, v", block_args="u, v")
    assert third.require_value().blocks[0].relations[0].kind == "single_state"


def test_result_closure_rejects_cross_block_and_omitted_rows(tmp_path):
    result, blocks, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, x, context):
        x = x + self.mix(x, context)
        return x
""")
    inventory = result.require_value()
    with pytest.raises(ValueError, match="exactly cover"):
        DiffusionStreamInventory(
            inventory.component_root, inventory.block_inventory, ())
    relation = inventory.blocks[0].relations[0]
    with pytest.raises(ValueError, match="single-state"):
        replace(relation, kind="single_state")
    with pytest.raises(ValueError, match="canonical norm"):
        replace(inventory.blocks[0], norm_invocations=(relation.lane_call,))
    with pytest.raises(ValueError, match="round-trip"):
        replace(inventory.blocks[0], roots=())


def test_foreign_index_and_unresolved_root_are_not_accepted(tmp_path):
    result, blocks, root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, x, context): return x + self.mix(x, context)
""")
    assert result.status == "incomplete"
    path = _write(tmp_path, (PREFIX + SELF_MIXER + """
class Other:
    def forward(self, x): return x
"""))
    other_bundle = SourceBundle(
        source="other", files=(path,), architecture="Other",
        component_files={"root": (path,)},
        component_architectures={"root": "Other"})
    other_index = build_program_index(other_bundle)
    foreign = read_diffusion_stream_graph(other_index, root, blocks)
    assert foreign.status == "failed"
    assert foreign.failures[0].kind == "out_of_owner"


def test_parser_shadow_preserves_source_missing_as_typed_unknown():
    from model_unfolder.adapters.diffusor.parser import (
        _shadow_diffusion_stream_and_conditioning,
        _source_only_diffusion_stream_and_conditioning,
    )
    from model_unfolder.evidence.context import ParseContext

    _source_only_diffusion_stream_and_conditioning.cache_clear()
    context = ParseContext(
        source_bundle=SourceBundle(source="local", files=()))
    result = _shadow_diffusion_stream_and_conditioning(context)
    streams = context.reader_results[("root.denoiser.streams", ())]
    assert result.status == streams.status == "failed"
    assert [item.kind for item in result.failures] == ["missing_source"]
    assert context.reader_results[
        ("root.denoiser.conditioning", ())] is result


@pytest.mark.parametrize(
    "witness,relations,unresolved,ffn_relations,ffn_unresolved,applications", [
    ("auraflow-v0-3", ("dual_state", "single_state"), 0,
     ("single_state", "single_state", "single_state"), 0,
     (("bare_gate", "attention"), ("bare_gate", "attention"),
      ("bare_gate", "attention"), ("bare_gate", "ffn"))),
    ("cogvideox-5b", ("dual_state",), 0, (), 0,
     (("bare_gate", "attention"), ("bare_gate", "attention"))),
    ("flux-2-dev", ("dual_state",), 1, (), 2,
     (("bare_gate", "attention"), ("bare_gate", "attention"),
      ("bare_gate", "ffn"), ("bare_gate", "ffn"),
      ("bare_gate", "attention"))),
    ("fluxtransformer2dmodel", ("joined_inputs",), 1, (), 0, ()),
    ("hunyuanvideo", ("single_state",), 0, (), 0,
     (("bare_gate", "attention"),)),
    ("ltx-video", ("single_state", "contextual_single_state"), 0,
     (), 0, (("bare_gate", "attention"),)),
    ("lumina-image-2-0", (), 6, (), 0, ()),
    ("mochi-1-preview", (), 0, (), 0, ()),
    ("pixart-sigma-xl-2-1024-ms", (), 0, (), 0, ()),
    ("prxpixel-t2i", ("contextual_single_state",), 0, (), 0,
     (("bare_gate", "attention"),)),
    ("qwen-image", (), 1, (), 0, ()),
    ("sana-1600m-1024px-diffusers", (), 0, (), 0, ()),
    ("stable-diffusion-3-5-large", (), 0, (), 0, ()),
    ("stable-diffusion-xl-base-1-0", (), 0, (), 0, ()),
    ("wan2-2-t2v-a14b-diffusers", (), 2, (), 0, ()),
])
def test_real_u10d_matrix_is_local_source_proof_not_family_classification(
        witness, relations, unresolved, ffn_relations, ffn_unresolved,
        applications):
    import model_unfolder as mu
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.diffusion_conditioning import (
        read_diffusion_conditioning_graph,
    )
    from model_unfolder.parser import _coerce

    corpus = Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    data = json.loads((corpus / f"{witness}.json").read_text())
    context = ParseContext.build(_coerce(data.get("config") or data))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    stacks = read_diffusion_stack_inventory(index, root)
    blocks = read_diffusion_block_facts(index, root, stacks)
    streams = read_diffusion_stream_graph(index, root, blocks)
    conditioning = read_diffusion_conditioning_graph(index, root, streams)

    assert streams.status == conditioning.status == "incomplete"
    assert tuple(item.kind for block in streams.value.blocks
                 for item in block.relations) == relations
    assert sum(len(block.unresolved) for block in streams.value.blocks) \
        == unresolved
    assert tuple(item.kind for block in streams.value.blocks
                 for item in block.ffn_relations) == ffn_relations
    assert sum(len(block.unresolved_ffns) for block in streams.value.blocks) \
        == ffn_unresolved
    assert tuple((item.kind, item.branch_kind)
                 for block in conditioning.value.blocks
                 for item in block.applications) == applications
    assert all(not hasattr(item, "model_family")
               for block in streams.value.blocks for item in block.relations)
