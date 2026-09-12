"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.adapters.diffusor.config_binding import bind_diffusion_source_projection
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
