"""U10-C exact diffusion block-fact composition controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.component_owner import (
    ComponentRootResolution,
    resolve_component_root,
)
from model_unfolder.evidence.diffusion_block import (
    DiffusionAttentionLaneFacts,
    DiffusionBlockFactInventory,
    read_diffusion_block_facts,
)
from model_unfolder.evidence.attention_lane import (
    FrameworkAttentionLaneEvidence,
)
from model_unfolder.evidence.diffusion_stack import read_diffusion_stack_inventory
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(files, architecture="Root"):
    return SourceBundle(
        source="test", files=tuple(files), architecture=architecture,
        component_files={"root": tuple(files)},
        component_architectures={"root": architecture})


def _selector(document, *, triple=False):
    def select(path):
        value = document
        for part in path:
            if not isinstance(value, dict) or part not in value:
                return (False, None, "") if triple else None
            value = value[part]
        return (True, value, "config_declared") if triple else value
    return select


def _read(tmp_path, source, *, config=None, extra_files=(), architecture="Root"):
    path = _write(tmp_path, "model.py", source)
    bundle = _bundle((path, *extra_files), architecture)
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved", (
        root.status, root.declared_architecture,
        root.candidates, root.parse_failures)
    stacks = read_diffusion_stack_inventory(index, root)
    document = config
    result = read_diffusion_block_facts(
        index, root, stacks, config_document=document,
        config_value_selector=(
            _selector(document) if document is not None else None),
        config_guard_selector=(
            _selector(document, triple=True) if document is not None else None))
    return result, stacks, root, index


PREFIX = textwrap.dedent("""
    import torch
    from torch import nn
    from torch.nn import functional as F
""")


ATTENTION = textwrap.dedent("""
    class Mixer:
        def __init__(self, config):
            self.width = config.hidden // config.query_groups
            self.q = nn.Linear(config.hidden, config.query_groups * self.width)
            self.k = nn.Linear(config.hidden, config.shared_groups * self.width)
            self.v = nn.Linear(config.hidden, config.shared_groups * self.width)
        def forward(self, x):
            query = self.q(x)
            key = self.k(x)
            value = self.v(x)
            score = torch.matmul(query, key.transpose(-1, -2))
            return torch.matmul(F.softmax(score, dim=-1), value)
""")


def _source(*, gated=False, two_stacks=False, config_rope_decoy=False):
    ffn = textwrap.dedent("""
        class FeedForward:
            def __init__(self, config):
                self.up = nn.Linear(config.hidden, config.wide)
                self.down = nn.Linear(config.wide, config.hidden)
                self.act = nn.GELU()
            def forward(self, x):
                return self.down(self.act(self.up(x)))
    """) if not gated else textwrap.dedent("""
        class FeedForward:
            def __init__(self, config):
                self.gate = nn.Linear(config.hidden, config.wide)
                self.up = nn.Linear(config.hidden, config.wide)
                self.down = nn.Linear(config.wide, config.hidden)
            def forward(self, x):
                return self.down(F.silu(self.gate(x)) * self.up(x))
    """)
    decoy = "self.declared_rotation = config.rope" if config_rope_decoy else ""
    root = textwrap.dedent("""
        class Root:
            def __init__(self, config):
                self.first = nn.ModuleList(
                    [Block(config) for _ in range(config.layers)])
                self.second = nn.ModuleList(
                    [Block(config) for _ in range(config.other_layers)])
            def forward(self, x):
                for item in self.first:
                    x = item(x)
                for item in self.second:
                    x = item(x)
                return x
    """) if two_stacks else textwrap.dedent("""
        class Root:
            def __init__(self, config):
                self.layers = nn.ModuleList(
                    [Block(config) for _ in range(config.layers)])
            def forward(self, x):
                for item in self.layers:
                    x = item(x)
                return x
    """)
    block = textwrap.dedent(f"""
        class Block:
            def __init__(self, config):
                self.n1 = nn.LayerNorm(config.hidden)
                self.attn = Mixer(config)
                self.n2 = nn.LayerNorm(config.hidden)
                self.ffn = FeedForward(config)
                {decoy}
            def forward(self, x):
                residual = x
                x = self.n1(x)
                x = self.attn(x)
                x = residual + x
                residual = x
                x = self.n2(x)
                x = self.ffn(x)
                return residual + x
    """)
    return PREFIX + ATTENTION + ffn + block + root


CONFIG = {
    "hidden": 64,
    "wide": 128,
    "query_groups": 8,
    "shared_groups": 2,
    "layers": 4,
    "other_layers": 2,
    "rope": True,
}


@pytest.mark.parametrize(("gated", "mode"), (
    (False, "dense"),
    (True, "split"),
))
def test_exact_block_composes_attention_ffn_norm_without_role_guess(
        tmp_path, gated, mode):
    result, stacks, _root, _index = _read(
        tmp_path, _source(gated=gated), config=CONFIG)
    assert result.status == stacks.status == "incomplete"
    inventory = result.require_value()
    assert len(inventory.blocks) == 1
    block = inventory.blocks[0]
    assert block.stack == stacks.require_value().stacks[0]
    assert block.norm_result.status == "resolved"
    assert block.norm_result.value == "layernorm"
    assert block.ffn_result.status == "resolved", block.ffn_result.failures
    assert block.ffn_result.value.gated is gated
    assert block.ffn_result.value.projection_mode == mode
    assert len(block.attention_lanes) == 1
    lane = block.attention_lanes[0]
    assert lane.compute_protocol == "dot_softmax"
    assert lane.score_scaling_result.status == "resolved"
    assert lane.head_binding_result.status == "resolved"
    assert lane.projection_storage_result.status == "resolved"
    assert not hasattr(lane, "role")
    assert not hasattr(lane, "self_attention")


def test_config_declared_rope_without_application_stays_unknown(tmp_path):
    result, _stacks, _root, _index = _read(
        tmp_path, _source(config_rope_decoy=True), config=CONFIG)
    lane = result.require_value().blocks[0].attention_lanes[0]
    assert lane.position_application_result.status != "resolved"
    assert lane.separate_position_application_result.status != "resolved"
    assert all("rope" not in failure.detail.lower()
               for failure in lane.position_application_result.failures)


def test_same_block_class_in_two_containers_never_unions_occurrences(tmp_path):
    result, stacks, _root, _index = _read(
        tmp_path, _source(two_stacks=True), config=CONFIG)
    inventory = result.require_value()
    assert len(inventory.blocks) == len(stacks.require_value().stacks) == 2
    assert len({item.stack.block_occurrence for item in inventory.blocks}) == 2
    assert {item.stack.block_symbol for item in inventory.blocks} \
        == {inventory.blocks[0].stack.block_symbol}
    assert all(len(item.attention_lanes) == 1 for item in inventory.blocks)


def test_two_attention_children_remain_two_exact_lanes(tmp_path):
    source = PREFIX + textwrap.dedent("""
        class Norm:
            def __init__(self, dim): self.weight = torch.ones(dim)
            def forward(self, x):
                variance = x.pow(2).mean(-1, keepdim=True)
                return self.weight * (x * torch.rsqrt(variance + 1e-6))
        class Left:
            def __init__(self, config):
                self.width = config.hidden // config.left_heads
                self.q = nn.Linear(config.hidden, config.left_heads * self.width)
                self.k = nn.Linear(config.hidden, config.left_kv * self.width)
                self.v = nn.Linear(config.hidden, config.left_kv * self.width)
                self.qn = Norm(config.hidden)
                self.kn = Norm(config.hidden)
            def forward(self, x):
                q, k, v = self.q(x), self.k(x), self.v(x)
                q, k = self.qn(q), self.kn(k)
                return torch.matmul(
                    F.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1), v)
        class Right:
            def __init__(self, config):
                self.width = config.hidden // config.right_heads
                self.q = nn.Linear(config.hidden, config.right_heads * self.width)
                self.k = nn.Linear(config.hidden, config.right_kv * self.width)
                self.v = nn.Linear(config.hidden, config.right_kv * self.width)
            def forward(self, x):
                q, k, v = self.q(x), self.k(x), self.v(x)
                return torch.matmul(
                    F.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1), v)
        class Block:
            def __init__(self, config):
                self.a = Left(config)
                self.b = Right(config)
            def forward(self, x, context):
                x = self.a(x)
                return self.b(context) + x
        class Root:
            def __init__(self, config):
                self.layers = nn.ModuleList(
                    [Block(config) for _ in range(config.layers)])
            def forward(self, x, context):
                for item in self.layers:
                    x = item(x, context)
                return x
    """)
    config = {
        **CONFIG,
        "left_heads": 8, "left_kv": 8,
        "right_heads": 16, "right_kv": 4,
    }
    result, _stacks, _root, _index = _read(tmp_path, source, config=config)
    block = result.require_value().blocks[0]
    assert len(block.attention_lanes) == 2
    paths = {
        lane.head_binding_result.value.query_heads_path
        for lane in block.attention_lanes
        if lane.head_binding_result.status == "resolved"
    }
    assert paths == {("left_heads",), ("right_heads",)}
    assert len({lane.child.compute_occurrence
                for lane in block.attention_lanes}) == 2
    assert sorted(lane.qk_norm_result.status for lane in block.attention_lanes) \
        == ["failed", "resolved"]
    assert all(not hasattr(lane, "role") for lane in block.attention_lanes)


def test_non_softmax_mixer_is_positive_and_not_laundered_as_attention(tmp_path):
    source = PREFIX + textwrap.dedent("""
        def step_one(q, k, v, **kwargs): return q, k
        def step_two(q, k, v, **kwargs): return q, k
        class Recurrent:
            def __init__(self, config):
                self.red = config.key_heads
                self.green = config.value_heads
                self.blue = config.key_dim
                self.gold = config.value_dim
                self.kw = config.kernel
                self.qk_width = self.blue * self.red
                self.v_width = self.gold * self.green
                self.conv = nn.Conv1d(
                    self.qk_width * 2 + self.v_width,
                    self.qk_width * 2 + self.v_width, kernel_size=self.kw)
                self.first = step_one
                self.second = step_two
            def forward(self, x):
                q, k, v = torch.split(
                    x, [self.qk_width, self.qk_width, self.v_width], dim=-1)
                q = q.reshape(1, 1, -1, self.blue)
                k = k.reshape(1, 1, -1, self.blue)
                v = v.reshape(1, 1, -1, self.gold)
                beta = x.sigmoid()
                decay = F.softplus(x)
                if x.shape[0] == 1:
                    out, state = self.first(q, k, v, decay=decay, beta=beta)
                else:
                    out, state = self.second(q, k, v, decay=decay, beta=beta)
                if self.green // self.red > 1:
                    q = q.repeat_interleave(self.green // self.red)
                    k = k.repeat_interleave(self.green // self.red)
                return out
        class Block:
            def __init__(self, config): self.unit = Recurrent(config)
            def forward(self, x): return self.unit(x)
        class Root:
            def __init__(self, config):
                self.layers = nn.ModuleList(
                    [Block(config) for _ in range(config.layers)])
            def forward(self, x):
                for item in self.layers: x = item(x)
                return x
    """)
    config = {
        **CONFIG, "key_heads": 2, "value_heads": 4,
        "key_dim": 8, "value_dim": 4, "kernel": 3,
    }
    result, _stacks, _root, _index = _read(tmp_path, source, config=config)
    block = result.require_value().blocks[0]
    assert not block.attention_lanes
    assert len(block.non_softmax_mixers) == 1
    assert block.non_softmax_mixers[0].status == "resolved"


def test_imported_block_stays_joined_to_exact_cross_file_symbol(tmp_path):
    part = _write(tmp_path, "part.py", PREFIX + ATTENTION + textwrap.dedent("""
        class FeedForward:
            def __init__(self, config):
                self.up = nn.Linear(config.hidden, config.wide)
                self.down = nn.Linear(config.wide, config.hidden)
            def forward(self, x): return self.down(F.silu(self.up(x)))
        class Neutral:
            def __init__(self, config):
                self.attn = Mixer(config)
                self.ffn = FeedForward(config)
            def forward(self, x): return self.ffn(self.attn(x))
    """))
    source = PREFIX + textwrap.dedent("""
        from part import Neutral as Unit
        class Root:
            def __init__(self, config):
                self.items = nn.ModuleList(
                    [Unit(config) for _ in range(config.layers)])
            def forward(self, x):
                for item in self.items: x = item(x)
                return x
    """)
    result, _stacks, _root, _index = _read(
        tmp_path, source, config=CONFIG, extra_files=(part,))
    block = result.require_value().blocks[0]
    assert block.stack.block_symbol.qualified_name == "Neutral"
    assert block.attention_lanes[0].child.compute_owner_symbol.source \
        == block.stack.block_symbol.source


def test_upstream_unresolved_stack_remains_opaque(tmp_path):
    source = PREFIX + textwrap.dedent("""
        class A:
            def forward(self, x): return x
        class B:
            def forward(self, x): return x
        class Root:
            def __init__(self, config):
                if config.pick:
                    self.items = nn.ModuleList([A()])
                else:
                    self.items = nn.ModuleList([B()])
            def forward(self, x):
                for item in self.items: x = item(x)
                return x
    """)
    result, stacks, _root, _index = _read(
        tmp_path, source, config={**CONFIG, "pick": True})
    assert result.status == "incomplete"
    assert not result.require_value().blocks
    assert result.require_value().unresolved_stacks \
        == stacks.require_value().unresolved


def test_lane_dto_rejects_cross_block_laundering(tmp_path):
    result, _stacks, _root, _index = _read(
        tmp_path, _source(two_stacks=True), config=CONFIG)
    first, second = result.require_value().blocks
    lane = first.attention_lanes[0]
    with pytest.raises(ValueError, match="belongs|exact block"):
        DiffusionAttentionLaneFacts(
            second.stack.block_occurrence, lane.block_symbol, lane.child,
            lane.score_scaling_result, lane.head_binding_result,
            lane.head_geometry_result, lane.projection_storage_result,
            lane.qk_norm_result, lane.position_application_result,
            lane.separate_position_application_result, lane.spans)


def test_inventory_rejects_omitted_positive_stack(tmp_path):
    result, _stacks, _root, _index = _read(
        tmp_path, _source(two_stacks=True), config=CONFIG)
    inventory = result.require_value()
    with pytest.raises(ValueError, match="exact stack"):
        DiffusionBlockFactInventory(
            inventory.component_root, inventory.stack_inventory,
            inventory.blocks[:1], inventory.unresolved_stacks)


def test_bad_root_or_foreign_stack_result_is_rejected(tmp_path):
    result, stacks, root, index = _read(
        tmp_path, _source(), config=CONFIG)
    assert result.status == "incomplete"
    with pytest.raises(ValueError, match="resolved D0"):
        read_diffusion_block_facts(
            index, ComponentRootResolution("absent", "root"), stacks)

    other_result, other_stacks, other_root, _other_index = _read(
        tmp_path, _source().replace("class Root:", "class OtherRoot:"),
        config=CONFIG, architecture="OtherRoot")
    assert other_result.status == "incomplete"
    foreign = read_diffusion_block_facts(index, other_root, other_stacks)
    assert foreign.status == "failed"
    assert foreign.failures[0].kind == "out_of_owner"


def test_parser_shadow_is_call_local_and_has_no_config_authority(tmp_path):
    from model_unfolder.adapters.diffusor.parser import (
        _shadow_diffusion_block_facts,
        _source_only_diffusion_stack_and_blocks,
    )
    from model_unfolder.evidence.context import ParseContext

    path = _write(tmp_path, "shadow.py", _source(config_rope_decoy=True))
    bundle = _bundle((path,))
    _source_only_diffusion_stack_and_blocks.cache_clear()
    context = ParseContext(source_bundle=bundle)
    first = _shadow_diffusion_block_facts(context)
    second = _shadow_diffusion_block_facts(context)
    assert first is second
    lane = first.require_value().blocks[0].attention_lanes[0]
    assert lane.position_application_result.status != "resolved"
    assert context.reader_results[
        ("root.denoiser.blocks", ())] is first
    other_context = ParseContext(source_bundle=bundle)
    assert _shadow_diffusion_block_facts(other_context) is first
    assert _source_only_diffusion_stack_and_blocks.cache_info().hits == 1

    # Same path but different bytes must never reuse the source-only result.
    _write(tmp_path, "shadow.py", _source(gated=True))
    changed_context = ParseContext(source_bundle=bundle)
    changed = _shadow_diffusion_block_facts(changed_context)
    assert changed is not first
    assert changed.require_value().blocks[0].ffn_result.value.gated is True


def test_parser_shadow_keeps_an_unresolved_root_as_typed_unknown():
    from model_unfolder.adapters.diffusor.parser import (
        _shadow_diffusion_block_facts,
        _source_only_diffusion_stack_and_blocks,
    )
    from model_unfolder.evidence.context import ParseContext

    _source_only_diffusion_stack_and_blocks.cache_clear()
    context = ParseContext(
        source_bundle=SourceBundle(source="local", files=()))
    result = _shadow_diffusion_block_facts(context)

    assert result.status == "failed"
    assert result.owner is None
    assert [failure.kind for failure in result.failures] == ["missing_source"]
    assert context.reader_results[("root.denoiser.blocks", ())] is result


def _processor_attention_source(*, internal_container=False,
                                processor_math=True, guarded=False):
    processor_body = textwrap.dedent("""
        q = x
        k = x
        v = x
        return F.scaled_dot_product_attention(q, k, v)
    """) if processor_math else "return x\n"
    processor = (
        "class Worker:\n"
        "    def __call__(self, owner, x):\n"
        + textwrap.indent(processor_body, " " * 8))
    container = textwrap.dedent("""
        from diffusers.models.attention import AttentionModuleMixin
        class NeutralContainer(nn.Module, AttentionModuleMixin):
            def __init__(self, config, processor=None):
                super().__init__()
                self.set_processor(processor)
            def forward(self, x):
                return self.processor(self, x)
    """) if internal_container else textwrap.dedent("""
        from diffusers.models.attention_processor import Attention as NeutralContainer
    """)
    assignment = textwrap.indent(
        textwrap.dedent("""
            if config.use_lane:
                self.unit = NeutralContainer(
                    query_dim=config.hidden, processor=Worker())
        """) if guarded else textwrap.dedent("""
            self.unit = NeutralContainer(
                query_dim=config.hidden, processor=Worker())
        """), " " * 8)
    block = (
        "class Block:\n"
        "    def __init__(self, config):\n"
        + assignment
        + "    def forward(self, x):\n"
        + "        return self.unit(x)\n")
    root = textwrap.dedent("""
        class Root:
            def __init__(self, config):
                self.layers = nn.ModuleList(
                    [Block(config) for _ in range(config.layers)])
            def forward(self, x):
                for item in self.layers:
                    x = item(x)
                return x
    """)
    return PREFIX + container + processor + block + root


@pytest.mark.parametrize("internal_container", (False, True))
def test_exact_framework_or_source_delegate_is_a_positive_lane(
        tmp_path, internal_container):
    result, _stacks, _root, _index = _read(
        tmp_path,
        _processor_attention_source(internal_container=internal_container),
        config=CONFIG)
    block = result.require_value().blocks[0]
    assert len(block.attention_lanes) == 1
    lane = block.attention_lanes[0]
    assert isinstance(lane.child, FrameworkAttentionLaneEvidence)
    assert lane.compute_protocol == "scaled_dot_product_attention"
    assert lane.head_binding_result.status == "failed"
    assert lane.position_application_result.status == "failed"


def test_attention_spelling_without_exact_protocol_is_powerless(tmp_path):
    source = _processor_attention_source().replace(
        "from diffusers.models.attention_processor import Attention as "
        "NeutralContainer",
        "class NeutralContainer(nn.Module):\n"
        "    def __init__(self, config, processor=None): pass\n"
        "    def forward(self, x): return x")
    result, _stacks, _root, _index = _read(tmp_path, source, config=CONFIG)
    assert not result.require_value().blocks[0].attention_lanes


def test_injected_processor_without_compute_cannot_prove_source_delegate(
        tmp_path):
    result, _stacks, _root, _index = _read(
        tmp_path,
        _processor_attention_source(
            internal_container=True, processor_math=False),
        config=CONFIG)
    assert not result.require_value().blocks[0].attention_lanes


def test_exact_framework_container_does_not_invent_processor_math(tmp_path):
    result, _stacks, _root, _index = _read(
        tmp_path,
        _processor_attention_source(processor_math=False),
        config=CONFIG)
    lane = result.require_value().blocks[0].attention_lanes[0]
    assert isinstance(lane.child, FrameworkAttentionLaneEvidence)
    assert lane.child.processor is None
    assert lane.compute_protocol == "framework_attention_container"


def _framework_geometry_source(*, include_kv=True, renamed=False):
    kv = ", kv_heads=kv_heads" if include_kv else ""
    heads_name = "head_count" if renamed else "heads"
    return PREFIX + textwrap.dedent(f"""
        from diffusers.models.attention_processor import Attention as Container
        class Block:
            def __init__(self, heads, kv_heads, dim_head):
                self.unit = Container(
                    {heads_name}=heads{kv}, dim_head=dim_head)
            def forward(self, x):
                return self.unit(x)
        class Root:
            def __init__(self, config):
                self.layers = nn.ModuleList([Block(
                    config.query_groups, config.shared_groups, config.hidden)
                    for _ in range(config.layers)])
            def forward(self, x):
                for item in self.layers:
                    x = item(x)
                return x
    """)


def test_framework_attention_geometry_is_api_role_not_model_identity(tmp_path):
    result, _stacks, _root, _index = _read(
        tmp_path, _framework_geometry_source(), config=CONFIG)
    geometry = result.require_value().blocks[0].attention_lanes[0].child.geometry
    assert geometry is not None
    assert geometry.query_heads.name == "heads"
    assert geometry.key_value_heads.name == "kv_heads"
    assert geometry.head_dim.name == "dim_head"


def test_framework_attention_omitted_kv_default_remains_unknown(tmp_path):
    result, _stacks, _root, _index = _read(
        tmp_path, _framework_geometry_source(include_kv=False), config=CONFIG)
    geometry = result.require_value().blocks[0].attention_lanes[0].child.geometry
    assert geometry is not None
    assert geometry.key_value_heads is None


def test_framework_attention_unknown_keyword_cannot_launder_geometry(tmp_path):
    result, _stacks, _root, _index = _read(
        tmp_path, _framework_geometry_source(renamed=True), config=CONFIG)
    lane = result.require_value().blocks[0].attention_lanes[0]
    assert lane.child.geometry is None


def test_guarded_framework_lane_uses_exact_occurrence_guard(tmp_path):
    source = _processor_attention_source(guarded=True)
    active, _stacks, _root, _index = _read(
        tmp_path, source, config={**CONFIG, "use_lane": True})
    assert len(active.require_value().blocks[0].attention_lanes) == 1
    inactive, _stacks, _root, _index = _read(
        tmp_path, source, config={**CONFIG, "use_lane": False})
    assert not inactive.require_value().blocks[0].attention_lanes


def test_exact_attention_dispatch_protocol_is_not_an_sdpa_spelling_guess(
        tmp_path):
    source = _processor_attention_source(internal_container=True).replace(
        "from diffusers.models.attention import AttentionModuleMixin",
        "from diffusers.models.attention import AttentionModuleMixin\n"
        "from diffusers.models.attention_dispatch import dispatch_attention_fn",
    ).replace(
        "return F.scaled_dot_product_attention(q, k, v)",
        "return dispatch_attention_fn(q, k, v)")
    result, _stacks, _root, _index = _read(tmp_path, source, config=CONFIG)
    lane = result.require_value().blocks[0].attention_lanes[0]
    assert lane.compute_protocol == "attention_dispatch"


def test_source_mixin_default_processor_is_exact_positive_evidence(tmp_path):
    source = PREFIX + textwrap.dedent("""
        from diffusers.models.attention import AttentionModuleMixin
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        class Worker:
            def __call__(self, owner, x):
                return dispatch_attention_fn(x, x, x)
        class NeutralContainer(nn.Module, AttentionModuleMixin):
            _default_processor_cls = Worker
            def __init__(self, config, processor=None):
                super().__init__()
                if processor is None:
                    processor = self._default_processor_cls()
                self.set_processor(processor)
            def forward(self, x):
                return self.processor(self, x)
        class Block:
            def __init__(self, config):
                self.unit = NeutralContainer(config)
            def forward(self, x):
                return self.unit(x)
        class Root:
            def __init__(self, config):
                self.layers = nn.ModuleList(
                    [Block(config) for _ in range(config.layers)])
            def forward(self, x):
                for item in self.layers:
                    x = item(x)
                return x
    """)
    result, _stacks, _root, _index = _read(tmp_path, source, config=CONFIG)
    lane = result.require_value().blocks[0].attention_lanes[0]
    assert lane.compute_protocol == "attention_dispatch"
    assert lane.child.processor.argument_name == "class_default_processor"


def test_default_processor_call_must_reach_the_processor_setter(tmp_path):
    source = PREFIX + textwrap.dedent("""
        from diffusers.models.attention import AttentionModuleMixin
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        class Worker:
            def __call__(self, owner, x):
                return dispatch_attention_fn(x, x, x)
        class NeutralContainer(nn.Module, AttentionModuleMixin):
            _default_processor_cls = Worker
            def __init__(self, config, processor=None):
                super().__init__()
                if processor is None:
                    self._default_processor_cls()
                self.set_processor(processor)
            def forward(self, x):
                return self.processor(self, x)
        class Block:
            def __init__(self, config): self.unit = NeutralContainer(config)
            def forward(self, x): return self.unit(x)
        class Root:
            def __init__(self, config):
                self.layers = nn.ModuleList(
                    [Block(config) for _ in range(config.layers)])
            def forward(self, x):
                for item in self.layers: x = item(x)
                return x
    """)
    result, _stacks, _root, _index = _read(tmp_path, source, config=CONFIG)
    assert not result.require_value().blocks[0].attention_lanes


def test_fake_mixin_spelling_and_guarded_delegate_are_powerless(tmp_path):
    fake = _processor_attention_source(internal_container=True).replace(
        "from diffusers.models.attention import AttentionModuleMixin",
        "class AttentionModuleMixin: pass")
    result, _stacks, _root, _index = _read(tmp_path, fake, config=CONFIG)
    assert not result.require_value().blocks[0].attention_lanes

    guarded = _processor_attention_source(internal_container=True).replace(
        "return self.processor(self, x)",
        "if x is not None:\n            return self.processor(self, x)\n"
        "        return x")
    result, _stacks, _root, _index = _read(tmp_path, guarded, config=CONFIG)
    assert not result.require_value().blocks[0].attention_lanes


def test_framework_lane_dto_rejects_forged_protocol_and_missing_provenance(
        tmp_path):
    result, _stacks, _root, _index = _read(
        tmp_path, _processor_attention_source(), config=CONFIG)
    lane = result.require_value().blocks[0].attention_lanes[0].child
    with pytest.raises(ValueError, match="vocabulary"):
        replace(lane, protocol="familiar_attention_name")
    with pytest.raises(ValueError, match="provenance"):
        replace(lane, spans=tuple(
            span for span in lane.spans
            if span != lane.invocation.call.span))


def test_two_framework_containers_remain_two_invocation_lanes(tmp_path):
    source = _processor_attention_source().replace(
        "self.unit = NeutralContainer(\n"
        "            query_dim=config.hidden, processor=Worker())",
        "self.unit = NeutralContainer(\n"
        "            query_dim=config.hidden, processor=Worker())\n"
        "        self.other = NeutralContainer(\n"
        "            query_dim=config.hidden, processor=Worker())",
    ).replace("return self.unit(x)", "return self.other(self.unit(x))")
    result, _stacks, _root, _index = _read(tmp_path, source, config=CONFIG)
    lanes = result.require_value().blocks[0].attention_lanes
    assert len(lanes) == 2
    assert len({lane.child.invocation.call_site for lane in lanes}) == 2


def test_framework_lane_cannot_launder_an_ambiguous_ordinary_rival(tmp_path):
    source = _processor_attention_source().replace(
        "self.unit = NeutralContainer(\n"
        "            query_dim=config.hidden, processor=Worker())",
        "self.unit = NeutralContainer(\n"
        "            query_dim=config.hidden, processor=Worker())\n"
        "        self.ordinary = Mixer(config)\n"
        "        if config.optional_lane:\n"
        "            self.optional = Mixer(config)",
    ).replace(
        "return self.unit(x)",
        "return self.unit(x) + self.ordinary(x) + self.optional(x)")
    source = source.replace(PREFIX, PREFIX + ATTENTION, 1)
    config = dict(CONFIG)  # intentionally lacks optional_lane
    result, _stacks, _root, _index = _read(tmp_path, source, config=config)
    block = result.require_value().blocks[0]
    assert block.attention_census_result.status == "ambiguous"
    assert not block.attention_lanes


def test_external_attention_implementation_stays_typed_unknown_not_name_proven(
        tmp_path):
    source = _processor_attention_source().replace(
        "from diffusers.models.attention_processor import Attention as "
        "NeutralContainer",
        "from .attention_processor import MochiAttention as NeutralContainer")
    result, _stacks, _root, _index = _read(tmp_path, source, config=CONFIG)
    block = result.require_value().blocks[0]
    assert not block.attention_lanes
    assert block.attention_census_result.status == "failed"
    assert {failure.kind for failure in block.attention_census_result.failures} \
        == {"external_unavailable"}
    assert "MochiAttention" in block.attention_census_result.failures[0].detail


def test_positive_half_turn_application_is_composed_at_the_exact_lane(tmp_path):
    source = PREFIX + textwrap.dedent("""
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
        class Lane:
            def __init__(self, config):
                self.q = nn.Linear(config.hidden, config.hidden)
                self.k = nn.Linear(config.hidden, config.hidden)
                self.v = nn.Linear(config.hidden, config.hidden)
            def forward(self, x, first_factor, second_factor):
                q, k, v = self.q(x), self.k(x), self.v(x)
                q, k = apply_pair(q, k, first_factor, second_factor)
                return F.scaled_dot_product_attention(q, k, v)
        class Block:
            def __init__(self, config): self.lane = Lane(config)
            def forward(self, x, first_factor, second_factor):
                return self.lane(x, first_factor, second_factor)
        class Root:
            def __init__(self, config):
                self.layers = nn.ModuleList(
                    [Block(config) for _ in range(config.layers)])
            def forward(self, x, first_factor, second_factor):
                for item in self.layers:
                    x = item(x, first_factor, second_factor)
                return x
    """)
    result, _stacks, _root, _index = _read(tmp_path, source, config=CONFIG)
    lane = result.require_value().blocks[0].attention_lanes[0]
    assert lane.position_application_result.status == "resolved"
    assert lane.position_application_result.value.rotation_protocol \
        == "split_half_turn"


@pytest.mark.parametrize("ffn_body", (
    """
    class ConvGLU:
        def __init__(self, config):
            self.proj = nn.Conv1d(config.hidden, config.wide * 2, 1)
            self.out = nn.Conv1d(config.wide, config.hidden, 1)
        def forward(self, x):
            left, right = self.proj(x).chunk(2, dim=1)
            return self.out(F.silu(left) * right)
    """,
    """
    class Opaque:
        def __init__(self, config): self.width = config.wide
        def forward(self, x): return x
    """,
))
def test_conv_glu_and_unproven_ffn_stay_unknown_at_u10_c(tmp_path, ffn_body):
    ffn_name = "ConvGLU" if "class ConvGLU" in ffn_body else "Opaque"
    source = PREFIX + ATTENTION + textwrap.dedent(ffn_body) + textwrap.dedent(f"""
        class Block:
            def __init__(self, config):
                self.attn = Mixer(config)
                self.ffn = {ffn_name}(config)
            def forward(self, x): return self.ffn(self.attn(x))
        class Root:
            def __init__(self, config):
                self.layers = nn.ModuleList(
                    [Block(config) for _ in range(config.layers)])
            def forward(self, x):
                for item in self.layers: x = item(x)
                return x
    """)
    result, _stacks, _root, _index = _read(tmp_path, source, config=CONFIG)
    block = result.require_value().blocks[0]
    assert block.ffn_result.status == "failed"
    assert block.ffn_census_result.status == "failed"


def test_plain_norm_is_not_upgraded_to_adaln_by_a_modulated_sibling(tmp_path):
    source = PREFIX + ATTENTION + textwrap.dedent("""
        class Modulated:
            def __init__(self, config):
                self.norm = nn.LayerNorm(config.hidden)
            def forward(self, x, scale, shift):
                return self.norm(x) * (1 + scale) + shift
        class Block:
            def __init__(self, config):
                self.plain = nn.RMSNorm(config.hidden)
                self.modulated = Modulated(config)
                self.attn = Mixer(config)
            def forward(self, x, scale, shift):
                x = self.plain(x)
                x = self.modulated(x, scale, shift)
                return self.attn(x)
        class Root:
            def __init__(self, config):
                self.layers = nn.ModuleList(
                    [Block(config) for _ in range(config.layers)])
            def forward(self, x, scale, shift):
                for item in self.layers: x = item(x, scale, shift)
                return x
    """)
    result, _stacks, _root, _index = _read(tmp_path, source, config=CONFIG)
    block = result.require_value().blocks[0]
    assert block.norm_result.status == "resolved"
    assert block.norm_result.value == "rmsnorm"
    assert not hasattr(block, "adaptive_norm")
    assert not hasattr(block, "adaln")


def test_model_stage_learned_positions_do_not_become_block_rotation(tmp_path):
    source = _source().replace(
        "self.layers = nn.ModuleList(",
        "self.position = nn.Embedding(512, config.hidden)\n"
        "        self.layers = nn.ModuleList(")
    result, _stacks, _root, _index = _read(tmp_path, source, config=CONFIG)
    lane = result.require_value().blocks[0].attention_lanes[0]
    assert lane.position_application_result.status != "resolved"
    assert lane.separate_position_application_result.status != "resolved"


def test_guarded_same_class_stack_occurrences_stay_exact_and_separate(tmp_path):
    source = PREFIX + textwrap.dedent("""
        class Block:
            def forward(self, x): return x
        class Root:
            def __init__(self, config):
                if config.use_first:
                    self.first = nn.ModuleList([Block()])
                if config.use_second:
                    self.second = nn.ModuleList([Block()])
            def forward(self, x):
                if x is not None:
                    for item in self.first: x = item(x)
                if x is not None:
                    for item in self.second: x = item(x)
                return x
    """)
    result, stacks, _root, _index = _read(
        tmp_path, source,
        config={**CONFIG, "use_first": True, "use_second": True})
    blocks = result.require_value().blocks
    assert len(blocks) == len(stacks.require_value().stacks) == 2
    assert len({item.stack.block_occurrence for item in blocks}) == 2
    assert len({item.stack.container.field for item in blocks}) == 2
    guards = tuple(
        item.stack.container.element_sites[0].guard for item in blocks)
    assert all(guards)
    assert len(set(guards)) == 2


@pytest.mark.parametrize("witness,lanes,protocols", [
    ("auraflow-v0-3", 2, {"framework_attention_container"}),
    ("cogvideox-5b", 1, {"framework_attention_container"}),
    ("flux-2-dev", 2, {"attention_dispatch"}),
    ("fluxtransformer2dmodel", 2, {"attention_dispatch"}),
    ("hunyuanvideo", 1, {"framework_attention_container"}),
    ("ltx-video", 2, {"attention_dispatch"}),
    ("lumina-image-2-0", 6, {"scaled_dot_product_attention"}),
    ("mochi-1-preview", 0, set()),
    ("pixart-sigma-xl-2-1024-ms", 0, set()),
    ("prxpixel-t2i", 1, {"attention_dispatch"}),
    ("qwen-image", 1, {"attention_dispatch"}),
    ("sana-1600m-1024px-diffusers", 0, set()),
    ("stable-diffusion-3-5-large", 0, set()),
    ("stable-diffusion-xl-base-1-0", 0, set()),
    ("wan2-2-t2v-a14b-diffusers", 2, {"attention_dispatch"}),
])
def test_real_diffusion_block_fact_matrix_is_source_only_and_occurrence_exact(
        witness, lanes, protocols):
    import json
    import pathlib

    import model_unfolder as mu
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    data = json.loads((corpus / f"{witness}.json").read_text())
    context = ParseContext.build(_coerce(data.get("config") or data))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    stacks = read_diffusion_stack_inventory(index, root)
    # This is the same source-only contract used by the parser's U10-C shadow:
    # no raw config document or selector may make a mechanism true here.
    result = read_diffusion_block_facts(index, root, stacks)
    assert result.status == "incomplete"
    inventory = result.require_value()
    all_lanes = tuple(
        lane for block in inventory.blocks for lane in block.attention_lanes)
    assert len(all_lanes) == lanes
    assert {lane.compute_protocol for lane in all_lanes} == protocols
    assert all(
        lane.block_occurrence == block.stack.block_occurrence
        for block in inventory.blocks for lane in block.attention_lanes)
    assert inventory.unresolved_stacks == stacks.require_value().unresolved

    if witness == "mochi-1-preview":
        block = inventory.blocks[0]
        assert block.attention_census_result.status == "failed"
        assert {failure.kind
                for failure in block.attention_census_result.failures} \
            == {"external_unavailable"}
    elif witness == "sana-1600m-1024px-diffusers":
        # attn2 is constructor-guarded by a config operand. U10-C keeps the
        # whole lane census ambiguous; U10-F will join that exact operand.
        assert inventory.blocks[0].attention_census_result.status == "ambiguous"
