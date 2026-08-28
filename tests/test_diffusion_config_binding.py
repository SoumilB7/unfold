"""U10-F2 exact binding controls; no production consumption is permitted."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.adapters.diffusor.config_binding import (
    BoundDiffusionSourceProjection,
    bind_diffusion_source_projection,
)
from model_unfolder.adapters.diffusor import config_binding
from model_unfolder.adapters.diffusor.projection_ir import (
    _project_cross_context_block,
    project_diffusion_ir,
)
from model_unfolder.adapters.transformer.blocks.attention import (
    attention_child_blocks,
)
from model_unfolder.evidence import config_access
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.document import DocumentBinding, prepare_document
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reader_result import ReaderResult
from model_unfolder.evidence.receipts import (
    fabrication_findings,
    join_obligation_receipts,
    stamp_context,
)
from model_unfolder.ir import AttentionSpec


def test_contextual_primary_attention_gets_its_exact_external_rail():
    primary = AttentionSpec(
        kind=None, num_heads=None, cross_attention=True,
        cross_kv_source="external context")
    blocks = [{"id": "attn", "kind": "attention"}]
    projected = _project_cross_context_block(blocks, primary, None)
    assert projected[0]["id"] == "cross_attention_states"
    assert projected[0]["feeds"] == "attn"
    assert projected[1:] == blocks


def test_contextual_attention_authors_the_matching_external_rail_card():
    attention = AttentionSpec(
        kind=None, num_heads=None, cross_attention=True,
        cross_kv_source="external context")
    cards = attention_child_blocks(attention, 64, generic=True)
    card = next(item for item in cards
                if item["id"] == "cross_attention_states")
    assert card["title"] == "Cross-attention K/V states"
    assert "external context" in card["description"]


def test_multiple_contextual_lanes_cannot_share_one_anonymous_rail():
    lane = AttentionSpec(
        kind=None, num_heads=None, cross_attention=True,
        cross_kv_source="external context")
    with pytest.raises(ValueError, match="multiple external-context rails"):
        _project_cross_context_block(
            [{"id": "attn"}, {"id": "cross_attn"}], lane, lane)


SOURCE = """
import torch
from torch import nn
from torch.nn import functional as F

class Kernel:
    def __init__(self, config):
        self.width = config.hidden // config.query_heads
        self.q = nn.Linear(config.hidden, config.query_heads * self.width)
        self.k = nn.Linear(config.hidden, config.kv_heads * self.width)
        self.v = nn.Linear(config.hidden, config.kv_heads * self.width)
    def forward(self, state, context):
        return F.scaled_dot_product_attention(
            self.q(state), self.k(context), self.v(context))

class DensePath:
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden, config.wide)
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, state):
        return self.down(F.silu(self.gate(state)) * self.up(state))

class Cell:
    def __init__(self, config):
        self.a = nn.LayerNorm(config.hidden)
        self.kernel = Kernel(config)
        self.b = nn.LayerNorm(config.hidden)
        self.dense = DensePath(config)
    def forward(self, state, context):
        state = state + self.kernel(self.a(state), context)
        state = state + self.dense(self.b(state))
        return state

class Root:
    def __init__(self, config):
        self.sequence = nn.ModuleList(
            [Cell(config) for _ in range(config.layers)])
    def forward(self, state, context):
        for element in self.sequence:
            state = element(state, context)
        return state
"""


REGISTERED_SOURCE = """
from diffusers.configuration_utils import register_to_config as rtc
import torch
from torch import nn
from torch.nn import functional as F

class Kernel:
    def __init__(self, hidden, query_heads, kv_heads):
        self.width = hidden // query_heads
        self.q = nn.Linear(hidden, query_heads * self.width)
        self.k = nn.Linear(hidden, kv_heads * self.width)
        self.v = nn.Linear(hidden, kv_heads * self.width)
    def forward(self, state, context):
        return F.scaled_dot_product_attention(
            self.q(state), self.k(context), self.v(context))

class DensePath:
    def __init__(self, hidden, wide):
        self.gate = nn.Linear(hidden, wide)
        self.up = nn.Linear(hidden, wide)
        self.down = nn.Linear(wide, hidden)
    def forward(self, state):
        return self.down(F.silu(self.gate(state)) * self.up(state))

class Cell:
    def __init__(self, hidden, wide, query_heads, kv_heads):
        self.a = nn.LayerNorm(hidden)
        self.kernel = Kernel(hidden, query_heads, kv_heads)
        self.b = nn.LayerNorm(hidden)
        self.dense = DensePath(hidden, wide)
    def forward(self, state, context):
        state = state + self.kernel(self.a(state), context)
        state = state + self.dense(self.b(state))
        return state

class Root:
    @rtc
    def __init__(self, layers=4, hidden=64, wide=128,
                 query_heads=8, kv_heads=2):
        self.sequence = nn.ModuleList([
            Cell(hidden, wide, query_heads, kv_heads)
            for _ in range(layers)
        ])
    def forward(self, state, context):
        for element in self.sequence:
            state = element(state, context)
        return state
"""


GATED_DELTA_SOURCE = """
from diffusers.configuration_utils import register_to_config as rtc
import torch
from torch import nn
from torch.nn import functional as F

def step_one(q, k, v, **kwargs): return q, k
def step_two(q, k, v, **kwargs): return q, k

class Recurrent:
    def __init__(self, key_heads, value_heads, key_dim, value_dim, kernel):
        self.red = key_heads
        self.green = value_heads
        self.blue = key_dim
        self.gold = value_dim
        self.kw = kernel
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

class Cell:
    def __init__(self, key_heads, value_heads, key_dim, value_dim, kernel):
        self.unit = Recurrent(
            key_heads, value_heads, key_dim, value_dim, kernel)
    def forward(self, x): return self.unit(x)

class Root:
    @rtc
    def __init__(self, layers=2, key_heads=2, value_heads=4,
                 key_dim=8, value_dim=4, kernel=3):
        self.layers = nn.ModuleList([
            Cell(key_heads, value_heads, key_dim, value_dim, kernel)
            for _ in range(layers)
        ])
    def forward(self, x):
        for item in self.layers: x = item(x)
        return x
"""


CONFIG = {
    "hidden": 64,
    "wide": 128,
    "query_heads": 8,
    "kv_heads": 2,
    "layers": 4,
    # Familiar declarations that source never reads are powerless.
    "attention_kind": "mha",
    "video": True,
}


def _write(tmp_path, source=SOURCE):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _inputs(tmp_path, source=SOURCE, config=CONFIG):
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="test", files=(path,), architecture="Root",
        component_files={"root": (path,)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.address_resolved
    topology = read_diffusion_root_topology(index, root)
    prepared = prepare_document(dict(config), merge=False)
    binding = DocumentBinding("root", (), prepared)
    companions = ReaderResult.absent(root.graph.root.occurrence)
    return index, root, binding, topology, companions


def _bind(tmp_path, source=SOURCE, config=CONFIG):
    index, root, binding, topology, companions = _inputs(
        tmp_path, source, config)
    ledger = config_access.ConfigAccessLedger()
    with config_access.capture_events(ledger), \
            config_access.bound_document(binding), \
            config_access.owner_scope("root.denoiser"):
        result = bind_diffusion_source_projection(
            index, root, binding, topology, companions)
    return result, ledger


def test_only_code_retained_exact_paths_are_bound_and_none_are_consumed(tmp_path):
    result, ledger = _bind(tmp_path)
    assert result.status == "incomplete"
    value = result.require_value()
    assert isinstance(value, BoundDiffusionSourceProjection)
    assert value.source.blocks[0].attention[0].head_protocol == "grouped_kv"
    paths = {item.path for item in value.operands}
    assert {("layers",), ("query_heads",), ("kv_heads",), ("hidden",)} \
        <= paths
    assert ("attention_kind",) not in paths
    assert ("video",) not in paths
    assert all(event.intent != "consumed" for event in ledger.events)
    bound = [event for event in ledger.events if event.intent == "bound"]
    assert {tuple(event.config_path.split(".")) for event in bound} == paths
    assert all(event.component == "root.denoiser" for event in bound)
    assert all(item.source_owner.root == value.source.component_root.root
               and item.source_spans for item in value.operands)


def test_count_operand_is_found_inside_the_normalized_expression(tmp_path):
    result, _ledger = _bind(tmp_path)
    rows = tuple(
        item for item in result.require_value().operands
        if item.fact_owner == "denoiser.stacks[0]"
        and item.fact_key == "num_layers")
    assert len(rows) == 1
    assert rows[0].path == ("layers",)
    # The decisive source is the parameter occurrence inside range(layers),
    # not a string search over the whole diagnostic expression.
    assert len(rows[0].source_spans) == 1
    assert rows[0].source_spans[0].col > \
        result.require_value().source.blocks[0].evidence.stack \
        .count_expression.span.col


def test_import_proven_registration_binds_scalar_constructor_operands(tmp_path):
    result, ledger = _bind(tmp_path, source=REGISTERED_SOURCE)
    value = result.require_value()
    assert value.registration_result.status == "resolved"
    assert {item.path for item in value.operands} >= {
        ("layers",), ("query_heads",), ("kv_heads",),
    }
    assert value.source.blocks[0].attention[0].head_protocol == "grouped_kv"
    assert all(event.intent != "consumed" for event in ledger.events)


def test_registered_self_config_count_binds_the_exact_parameter_path(tmp_path):
    source = REGISTERED_SOURCE.replace(
        "for _ in range(layers)",
        "for _ in range(self.config.layers)")
    result, _ledger = _bind(tmp_path, source=source)
    rows = tuple(
        item for item in result.require_value().operands
        if item.fact_owner == "denoiser.stacks[0]"
        and item.fact_key == "num_layers")
    assert len(rows) == 1
    assert rows[0].path == ("layers",)


def test_local_self_config_write_disables_the_framework_address(tmp_path):
    source = REGISTERED_SOURCE.replace(
        "query_heads=8, kv_heads=2):\n        self.sequence",
        "query_heads=8, kv_heads=2):\n"
        "        self.config = object()\n        self.sequence").replace(
            "for _ in range(layers)",
            "for _ in range(self.config.layers)")
    result, _ledger = _bind(tmp_path, source=source)
    assert not any(
        item.fact_owner == "denoiser.stacks[0]"
        and item.fact_key == "num_layers"
        for item in result.require_value().operands)


def test_local_registration_spelling_cannot_bind_scalar_constructor(tmp_path):
    source = REGISTERED_SOURCE.replace(
        "from diffusers.configuration_utils import register_to_config as rtc",
        "def register_to_config(fn): return fn").replace(
            "    @rtc\n", "    @register_to_config\n")
    result, ledger = _bind(tmp_path, source=source)
    value = result.require_value()
    assert value.registration_result.status == "failed"
    assert value.registration_result.failures[0].kind == "unresolved_import"
    assert value.operands == ()
    assert all(event.intent != "bound" for event in ledger.events)


def test_missing_operand_cannot_be_bound_or_replaced_by_a_decoy(tmp_path):
    config = {key: value for key, value in CONFIG.items() if key != "kv_heads"}
    config["num_key_value_heads"] = 2
    result, ledger = _bind(tmp_path, config=config)
    projection = result.require_value()
    lane = projection.source.blocks[0].attention[0]
    # The source protocol remains positively known, but no final GQA/MQA kind
    # can be derived until its exact KV operand is bound.
    assert lane.head_protocol == "grouped_kv"
    assert not hasattr(lane, "attention_kind")
    paths = {item.path for item in projection.operands}
    assert ("kv_heads",) not in paths
    assert ("num_key_value_heads",) not in paths
    assert ("kv_heads",) in {
        item.path for item in projection.unresolved_operands}
    assert all(event.intent != "consumed" for event in ledger.events)


def test_binding_result_rejects_cross_component_operand_forgery(tmp_path):
    result, _ledger = _bind(tmp_path)
    value = result.require_value()
    operand = value.operands[0]
    foreign_root = replace(
        operand.component_root,
        root=replace(
            operand.component_root.root,
            qualified_name="Foreign"))
    with pytest.raises(ValueError, match="exact owner in its component"):
        replace(value, operands=(replace(
            operand, component_root=foreign_root), *value.operands[1:]))
    with pytest.raises(ValueError, match="exactly partition"):
        replace(value, operands=value.operands[1:])
    with pytest.raises(ValueError, match="declared by the checkpoint"):
        replace(operand, resolution=replace(
            operand.resolution, provenance=config_access.CLASS_DEFAULT))
    with pytest.raises(ValueError, match="round-trips"):
        replace(operand, resolution=replace(operand.resolution, value=999999))


def test_same_class_and_path_in_two_stack_occurrences_never_collapse(tmp_path):
    source = SOURCE.replace(
        "self.sequence = nn.ModuleList(\n            [Cell(config) for _ in range(config.layers)])",
        "self.first = nn.ModuleList(\n"
        "            [Cell(config) for _ in range(config.layers)])\n"
        "        self.second = nn.ModuleList(\n"
        "            [Cell(config) for _ in range(config.layers)])",
    ).replace(
        "for element in self.sequence:\n            state = element(state, context)",
        "for element in self.first:\n"
        "            state = element(state, context)\n"
        "        for element in self.second:\n"
        "            state = element(state, context)",
    )
    result, _ledger = _bind(tmp_path, source=source)
    value = result.require_value()
    assert len(value.source.blocks) == 2
    query_rows = tuple(
        item for item in value.operands
        if item.path == ("query_heads",)
        and item.fact_key == "head_protocol")
    assert len(query_rows) == 2
    assert len({item.source_owner for item in query_rows}) == 2


def test_class_supplied_value_is_visible_but_never_bound(tmp_path):
    index, root, binding, topology, companions = _inputs(tmp_path)
    prepared = replace(
        binding.prepared,
        provenance={
            **binding.prepared.provenance,
            "query_heads": config_access.CLASS_DEFAULT,
        })
    binding = DocumentBinding("root", (), prepared)
    ledger = config_access.ConfigAccessLedger()
    with config_access.capture_events(ledger), \
            config_access.bound_document(binding), \
            config_access.owner_scope("root.denoiser"):
        result = bind_diffusion_source_projection(
            index, root, binding, topology, companions)
    value = result.require_value()
    assert ("query_heads",) in {
        item.path for item in value.unresolved_operands}
    assert ("query_heads",) not in {item.path for item in value.operands}
    assert not any(
        event.intent == "bound" and event.config_path == "query_heads"
        for event in ledger.events)


def test_projected_reader_path_cannot_hide_outside_operand_partition(
        tmp_path, monkeypatch):
    result, _ledger = _bind(tmp_path)
    value = result.require_value()
    original = config_binding._projected_config_paths
    monkeypatch.setattr(
        config_binding, "_projected_config_paths",
        lambda source: original(source) | {("hidden_dependency",)})
    with pytest.raises(ValueError, match="every projected config dependency"):
        replace(value)


def test_source_and_config_rename_changes_addresses_not_semantics(tmp_path):
    first, _ledger = _bind(tmp_path / "first")
    source = (SOURCE.replace("query_heads", "alpha_count")
              .replace("kv_heads", "beta_count")
              .replace("hidden", "model_width")
              .replace("wide", "inner_width")
              .replace("layers", "depth"))
    config = {
        "model_width": 64, "inner_width": 128,
        "alpha_count": 8, "beta_count": 2, "depth": 4,
        "attention_kind": "gqa", "video": False,
    }
    second, _other_ledger = _bind(tmp_path / "second", source, config)
    left = first.require_value()
    right = second.require_value()
    assert left.source.blocks[0].attention[0].head_protocol \
        == right.source.blocks[0].attention[0].head_protocol == "grouped_kv"
    assert left.source.blocks[0].ffn.gated \
        == right.source.blocks[0].ffn.gated is True
    assert {item.path for item in left.operands} \
        != {item.path for item in right.operands}


def test_unestablished_document_provenance_cannot_be_promoted_to_checkpoint(
        tmp_path):
    index, root, binding, topology, companions = _inputs(tmp_path)
    binding = DocumentBinding(
        "root", (), replace(binding.prepared, provenance={}))
    ledger = config_access.ConfigAccessLedger()
    with config_access.capture_events(ledger), \
            config_access.owner_scope("root.denoiser"):
        result = bind_diffusion_source_projection(
            index, root, binding, topology, companions)
    value = result.require_value()
    assert value.operands == ()
    assert value.unresolved_operands
    assert all(event.intent != "bound" for event in ledger.events)


def test_nested_document_binding_cannot_masquerade_as_the_root(tmp_path):
    index, root, binding, topology, companions = _inputs(tmp_path)
    nested = DocumentBinding("root", ("nested",), binding.prepared)
    with pytest.raises(ValueError, match="prepared root document"):
        bind_diffusion_source_projection(
            index, root, nested, topology, companions)


@pytest.mark.parametrize(
        "witness, expected_blocks, expected_unresolved, expected_bound", [
    ("auraflow-v0-3", 2, None, 2),
        ("cogvideox-5b", 1, 0, 3),
        ("flux-2-dev", 2, 0, 4),
        ("fluxtransformer2dmodel", 2, None, 4),
        ("hunyuanvideo", 3, 0, 9),
        ("ltx-video", 1, None, 4),
    ("lumina-image-2-0", 3, None, 4),
    ("mochi-1-preview", 1, None, 1),
    ("pixart-sigma-xl-2-1024-ms", 0, None, 0),
        ("prxpixel-t2i", 1, None, 2),
        ("qwen-image", 1, None, 5),
    ("sana-1600m-1024px-diffusers", 1, 2, 1),
    ("stable-diffusion-3-5-large", 0, None, 0),
    ("stable-diffusion-xl-base-1-0", 0, 4, 0),
    ("wan2-2-t2v-a14b-diffusers", 1, 1, 2),
])
def test_real_witness_operands_are_exact_and_never_consumed(
        witness, expected_blocks, expected_unresolved, expected_bound):
    import model_unfolder as mu
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce_prepared

    corpus = Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    data = json.loads((corpus / f"{witness}.json").read_text())
    prepared = _coerce_prepared(data.get("config") or data)
    context = ParseContext.build(prepared.document)
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    topology = read_diffusion_root_topology(index, root)
    binding = DocumentBinding("root", (), prepared)
    context.prepared_documents["root"] = binding
    companions = ReaderResult.absent(root.graph.root.occurrence)
    before = len(context.config_access.events)
    with config_access.capture_events(context.config_access), \
            config_access.bound_document(binding), \
            config_access.owner_scope("root.denoiser"):
        result = bind_diffusion_source_projection(
            index, root, binding, topology, companions)
    value = result.require_value()
    assert len(value.source.blocks) == expected_blocks
    if expected_unresolved is not None:
        assert len(value.source.unresolved_stacks) == expected_unresolved
    for operand in value.operands:
        current = binding.document
        for part in operand.path:
            assert isinstance(current, dict) and part in current
            current = current[part]
        assert current == operand.value
        assert operand.provenance == config_access.CHECKPOINT_DECLARED
        assert operand.source_owner.root == value.source.component_root.root
        assert operand.source_spans
    emitted = context.config_access.events[before:]
    assert all(event.intent != "consumed" for event in emitted)
    assert sum(event.intent == "bound" for event in emitted) \
        == len(value.operands)
    assert len(value.operands) == expected_bound


def test_f3_consumes_exact_rows_once_and_projects_source_proven_structure(
        tmp_path):
    result, _binding_ledger = _bind(
        tmp_path, source=REGISTERED_SOURCE)
    bound = result.require_value()
    ledger = config_access.ConfigAccessLedger()
    with config_access.capture_events(ledger), \
            config_access.owner_scope("root.denoiser"):
        projected = project_diffusion_ir(bound)

    assert len(projected.templates) == 1
    assert len(projected.layers) == 4
    layer = projected.layers[0]
    assert layer.attention.kind == "gqa"
    assert (layer.attention.num_heads, layer.attention.num_kv_heads) == (8, 2)
    # The shared exact-expression rail closes the constructor-local
    # ``self.width = hidden // query_heads`` assignment.  This is a derived
    # source+config value backed by both exact operands, not a conventional
    # hidden/query reconstruction in F3.
    assert layer.attention.head_dim == 8
    assert layer.attention.projection_mode == "split_qkv"
    assert layer.attention.mask == "unknown"
    assert layer.attention.scores_scaled is True
    assert layer.ffn.kind == "dense"
    assert layer.ffn.gated is True
    assert layer.ffn.projection_mode == "split"
    assert layer.norm_kind == "layernorm"
    assert layer.norm_placement == "pre"
    assert layer.residual_topology == "sequential"
    consumed = tuple(event for event in ledger.events
                     if event.intent == "consumed")
    assert len(consumed) == len(bound.operands)
    assert len({(event.fact_owner, event.fact_key, event.config_path)
                for event in consumed}) == len(consumed)
    assert all(event.mechanism.startswith("diffusion_") for event in consumed)
    assert all(event.value_status_hash for event in consumed)


def test_f3_derived_head_dimension_consumes_every_exact_premise(tmp_path):
    """One derived field may lawfully depend on multiple config occurrences.

    ``hidden // query_heads`` is one code-derived head dimension backed by two
    exact checkpoint operands.  The projector must join and consume both rows;
    requiring a single row would leave real source evidence unprojectable.
    """
    result, _binding_ledger = _bind(tmp_path, source=SOURCE)
    bound = result.require_value()
    head_rows = tuple(item for item in bound.operands
                      if item.fact_key == "head_dim")
    assert {item.path for item in head_rows} == {
        ("hidden",), ("query_heads",),
    }
    ledger = config_access.ConfigAccessLedger()
    with config_access.capture_events(ledger), \
            config_access.owner_scope("root.denoiser"):
        projected = project_diffusion_ir(bound)
    assert projected.layers[0].attention.head_dim == 8
    consumed_paths = {
        tuple(event.config_path.split("."))
        for event in ledger.events
        if event.intent == "consumed"
        and event.fact_key == "diffusion_attention_head_dim"
    }
    assert consumed_paths == {("hidden",), ("query_heads",)}


def test_f3_projection_closure_rejects_same_signature_block_forgery(tmp_path):
    result, _ledger = _bind(tmp_path, source=REGISTERED_SOURCE)
    with config_access.capture_events(config_access.ConfigAccessLedger()), \
            config_access.owner_scope("root.denoiser"):
        projected = project_diffusion_ir(result.require_value())
    layer = projected.layers[0]
    forged_blocks = [dict(block) for block in layer.blocks]
    forged_blocks[0]["description"] = "fabricated presentation claim"
    forged = replace(
        layer,
        blocks=forged_blocks,
    )
    # The grouping signature is deliberately coarser than the full block list;
    # projection closure must nevertheless reject the altered structure.
    assert forged.signature() == layer.signature()
    with pytest.raises(ValueError, match="derive solely from templates"):
        replace(projected, layers=(forged, *projected.layers[1:]))


def test_f3_never_converts_dispatch_or_missing_kv_operand_to_mha(tmp_path):
    dispatch = REGISTERED_SOURCE.replace(
        "from torch.nn import functional as F",
        "from torch.nn import functional as F\n"
        "from diffusers.models.attention_dispatch import dispatch_attention_fn",
    ).replace(
        "return F.scaled_dot_product_attention(\n"
        "            self.q(state), self.k(context), self.v(context))",
        "return dispatch_attention_fn(\n"
        "            self.q(state), self.k(context), self.v(context))",
    )
    result, _ledger = _bind(tmp_path / "dispatch", source=dispatch)
    with config_access.capture_events(config_access.ConfigAccessLedger()), \
            config_access.owner_scope("root.denoiser"):
        projected = project_diffusion_ir(result.require_value())
    assert projected.layers[0].attention.kind is None

    missing = {key: value for key, value in CONFIG.items()
               if key != "kv_heads"}
    result, _ledger = _bind(
        tmp_path / "missing", source=REGISTERED_SOURCE, config=missing)
    with config_access.capture_events(config_access.ConfigAccessLedger()), \
            config_access.owner_scope("root.denoiser"):
        projected = project_diffusion_ir(result.require_value())
    assert projected.layers[0].attention.kind is None
    assert projected.layers[0].attention.num_kv_heads is None


def test_f3_partial_gated_delta_geometry_keeps_mechanism_and_unknown_dims(
        tmp_path):
    config = {
        "layers": 2, "key_heads": 2, "value_heads": 4,
        "key_dim": 8, "value_dim": 4,
        # ``kernel`` is intentionally absent: source still proves the mixer,
        # while its five-part geometry is incomplete.
    }
    result, _ledger = _bind(
        tmp_path, source=GATED_DELTA_SOURCE, config=config)
    bound = result.require_value()
    assert any(item.fact_key == "conv_kernel"
               for item in bound.unresolved_operands)
    ledger = config_access.ConfigAccessLedger()
    with config_access.capture_events(ledger), \
            config_access.owner_scope("root.denoiser"):
        projected = project_diffusion_ir(bound)
    attention = projected.layers[0].attention
    assert attention.kind == "gated_delta"
    assert attention.mixer_state == "gated_delta"
    assert attention.num_heads is attention.num_kv_heads is None
    assert attention.head_dim is attention.conv_kernel_size is None
    assert len([event for event in ledger.events
                if event.intent == "consumed"]) == len(bound.operands)


def test_f3_missing_exact_stack_count_keeps_symbolic_template_unexpanded(
        tmp_path):
    missing = {key: value for key, value in CONFIG.items() if key != "layers"}
    result, _ledger = _bind(
        tmp_path, source=REGISTERED_SOURCE, config=missing)
    with config_access.capture_events(config_access.ConfigAccessLedger()), \
            config_access.owner_scope("root.denoiser"):
        projected = project_diffusion_ir(result.require_value())
    assert len(projected.templates) == 1
    assert projected.templates[0].count is None
    assert projected.layers == ()
    assert projected.unresolved == (
        "denoiser.stacks[0]: exact repetition count is not checkpoint-bound",
    )


def test_f3_production_render_is_honest_and_click_coupled():
    from model_unfolder import unfold
    from model_unfolder.block_schema import (
        validate_block_tree, validate_click_coupling,
    )
    from test_support import FLUX, PIXART

    flux = unfold(FLUX)
    assert flux.ir.num_layers == 57
    assert all(layer.attention.mask == "unknown" for layer in flux.ir.layers)
    flux_html = flux.to_html(standalone=True)
    assert validate_block_tree(flux.ir) == []
    assert validate_click_coupling(flux_html) == []
    assert all(term not in flux_html for term in (
        "Patchify", "Unpatchify", "AdaLN-Out"))
    expanded = flux.to_json()
    io = expanded["io"]
    assert "patchify" not in io
    assert io["input"]["kind"] == "denoiser_state"
    assert io["output"]["kind"] == "denoiser_state"
    assert io["output"]["domain"] is None
    assert io["input_transform"].get("operations") == ["linear"]
    assert "noise_prediction" not in str(io)

    pixart = unfold(PIXART)
    assert pixart.ir.num_layers == 0
    assert pixart.ir.extras["render"]["opaque_layer_block"]["resolved"] is False
    pixart_html = pixart.to_html(standalone=True)
    assert "Repeated denoiser" in pixart_html
    assert validate_block_tree(pixart.ir) == []
    assert validate_click_coupling(pixart_html) == []
    assert all(term not in pixart_html for term in (
        "Patchify", "Unpatchify", "AdaLN-Out"))


def _production_flux_spec_chain():
    """Return the actual parser-owned U10 fact/obligation/receipt chain."""
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import config_to_ir
    from test_support import FLUX

    context = ParseContext.build(FLUX)
    ir = config_to_ir(FLUX, parse_context=context)
    extras = ir.extras
    obligations = tuple(
        item for item in extras["config_access"]["projection_obligations"]
        if item["target"]["key"] == "diffusion_stack_depth")
    receipts = tuple(
        item for item in context.projection_receipts
        if item.fact_key == "diffusion_stack_depth")
    return extras, obligations, receipts


def test_f4_production_spec_receipts_join_exact_flux_stack_facts():
    """F4 closes config occurrence -> typed fact -> actual LayerSpec field.

    The receipt originates inside ``project_diffusion_ir`` and stays on the
    ParseContext until the audit context stamps it.  The renderer does not
    manufacture a receipt for this non-visual structural surface.
    """
    extras, obligations, receipts = _production_flux_spec_chain()
    assert len(obligations) == len(receipts) == 2
    assert {item.fact_id for item in receipts} == {
        "root.denoiser.stacks[0].diffusion_stack_depth",
        "root.denoiser.stacks[1].diffusion_stack_depth",
    }
    assert all(
        item.surface == "spec"
        and item.structural_target == "diffusion_stack_depth"
        and item.node_ids == ("diffusion_stack_depth",)
        and item.projector_symbol ==
        "adapters.diffusor.projection_ir.project_diffusion_ir"
        and item.context_token == ""
        for item in receipts)

    token = "test-flux-spec-context"
    joined = join_obligation_receipts(
        obligations, stamp_context(receipts, token),
        extras["fact_provenance"], context_token=token)
    assert joined["findings"] == []
    assert len(joined["receipted_targets"]) == 2


def test_f4_missing_production_spec_receipt_blocks():
    extras, obligations, receipts = _production_flux_spec_chain()
    token = "test-flux-spec-context"
    joined = join_obligation_receipts(
        obligations, stamp_context(receipts[:1], token),
        extras["fact_provenance"], context_token=token)
    assert len(joined["receipted_targets"]) == 1
    assert any("no projector emitted a matching receipt" in finding
               for finding in joined["findings"])


def test_f4_spec_receipt_value_drift_blocks():
    extras, obligations, receipts = _production_flux_spec_chain()
    token = "test-flux-spec-context"
    forged = replace(receipts[0], fact_value_status_hash="0" * 16)
    joined = join_obligation_receipts(
        obligations, stamp_context((forged, *receipts[1:]), token),
        extras["fact_provenance"], context_token=token)
    assert len(joined["receipted_targets"]) == 1
    assert any("drawing drifted from the ledgered fact" in finding
               for finding in joined["findings"])


def test_f4_source_only_spec_value_has_fact_and_real_consumer_receipt():
    """A code-fixed mechanism is not exempt from the receipt law.

    FLUX's projected FFN form is fixed by the exact source occurrence and has
    no F2 activation operand.  Before this poison, the value reached
    ``FFNSpec`` but the config-only receipt path emitted neither a typed fact
    nor a receipt.  Source-only and source+config projections must be equally
    auditable.
    """
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import config_to_ir
    from test_support import FLUX

    context = ParseContext.build(FLUX)
    ir = config_to_ir(FLUX, parse_context=context)
    facts = ir.extras["fact_provenance"]
    expected = {
        "root.denoiser.diffusion_root_topology": (
            "diffusion_root_topology", "spec", "diffusion_root_topology"),
        "root.denoiser.diffusion_bookend_operations": (
            "diffusion_bookend_operations", "block", "denoiser_bookends"),
        "root.denoiser.stacks[0].cell.diffusion_norm_mechanism": (
            "diffusion_norm_mechanism", "spec", "diffusion_norm_mechanism"),
        "root.denoiser.stacks[1].ffn.diffusion_ffn_mechanism": (
            "diffusion_ffn_mechanism", "spec", "diffusion_ffn_mechanism"),
    }
    assert all(facts[fact_id]["status"] == "code_proven"
               for fact_id in expected)
    # An occurrence's ``root`` is the component-root class, not necessarily
    # the nested class owning each decisive span.  U10 must retain exact
    # file/line provenance without laundering every span through that root
    # class label.
    typed = context.facts.typed_records()
    for fact_id in expected:
        assert typed[fact_id].source_spans
        assert all(span.file and span.line is not None
                   and span.class_name is None
                   for span in typed[fact_id].source_spans)
    receipts = tuple(item for item in context.projection_receipts
                     if item.fact_id in expected)
    assert len(receipts) == len(expected)
    assert {
        item.fact_id: (
            item.mechanism, item.surface, item.structural_target)
        for item in receipts
    } == expected
    assert fabrication_findings(receipts, facts, set()) == []


@pytest.mark.parametrize(("witness", "expected_counts", "expected_layers"), (
    ("flux-2-dev", (8, 48), 56),
    # The root dual/single stacks are guarded rivals. The exact imported
    # registration protocol proves the omitted selector's literal default,
    # selecting their ordinary branches; the nested refiner remains non-root.
    ("hunyuanvideo", (20, 40), 60),
    # One shared class contains guarded rival attention calls.  Exact literal
    # constructor arguments select one runtime lane per occurrence; the
    # inactive rival must not inflate the F3 lane cardinality.
    ("lumina-image-2-0", (2, 2, 26), 30),
    ("sana-1600m-1024px-diffusers", (20,), 20),
))
def test_f3_materializes_only_exact_root_topology_stages(
        witness, expected_counts, expected_layers):
    import model_unfolder as mu
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce_prepared

    corpus = Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    data = json.loads((corpus / f"{witness}.json").read_text())
    prepared = _coerce_prepared(data.get("config") or data)
    context = ParseContext.build(prepared.document)
    binding = DocumentBinding("root", (), prepared)
    context.prepared_documents["root"] = binding
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    topology = read_diffusion_root_topology(index, root)
    companions = ReaderResult.absent(root.graph.root.occurrence)
    with config_access.capture_events(context.config_access), \
            config_access.bound_document(binding), \
            config_access.owner_scope("root.denoiser"):
        bound = bind_diffusion_source_projection(
            index, root, binding, topology, companions).require_value()
        projected = project_diffusion_ir(bound)
    assert tuple(template.count for template in projected.templates
                 if template.root_stage) == expected_counts
    assert len(projected.layers) == expected_layers
    if witness == "hunyuanvideo":
        assert len(projected.templates) == 3
        assert [item.stack_variant for item in projected.templates[:2]] == [
            {"selected_branch": 1, "candidate_count": 2},
            {"selected_branch": 1, "candidate_count": 2},
        ]
        assert [item.source.evidence.stack.selection.premises
                for item in projected.templates[:2]] == [
            ((("image_condition_type",), "class_default", None),),
            ((("image_condition_type",), "class_default", None),),
        ]
        assert projected.templates[2].count == 2
        assert projected.templates[2].root_stage is False
        assert any("not an exact root-topology stage" in item
                   for item in projected.unresolved)
    if witness == "sana-1600m-1024px-diffusers":
        template = projected.templates[0]
        assert template.materialization_blocked is False
        assert template.attention.cross_attention is False
        assert template.cross_attention is not None
        assert template.cross_attention.cross_attention is True
    if witness == "lumina-image-2-0":
        assert all(not item.materialization_blocked
                   for item in projected.templates)
        assert all(item.attention.variant["stream_relation"] == "single_state"
                   for item in projected.templates)
