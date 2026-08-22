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
from model_unfolder.evidence import config_access
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.document import DocumentBinding, prepare_document
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reader_result import ReaderResult


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
        if item.fact_owner == "denoiser.stack"
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
    ("auraflow-v0-3", 2, None, 0),
    ("cogvideox-5b", 1, 0, 1),
    ("flux-2-dev", 2, 0, 2),
    ("fluxtransformer2dmodel", 2, None, 2),
    ("hunyuanvideo", 1, 2, 1),
    ("ltx-video", 1, None, 1),
    ("lumina-image-2-0", 3, None, 3),
    ("mochi-1-preview", 1, None, 1),
    ("pixart-sigma-xl-2-1024-ms", 0, None, 0),
    ("prxpixel-t2i", 1, None, 1),
    ("qwen-image", 1, None, 1),
    ("sana-1600m-1024px-diffusers", 1, 2, 1),
    ("stable-diffusion-3-5-large", 0, None, 0),
    ("stable-diffusion-xl-base-1-0", 0, 4, 0),
    ("wan2-2-t2v-a14b-diffusers", 1, 1, 1),
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
