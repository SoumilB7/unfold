"""U3-F exact-address routed-expert storage controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.decoder_block import decoder_block_path_at_root
from model_unfolder.evidence.expert_storage import (
    decoder_routed_expert_storage_for_path,
    routed_expert_storage_at_block,
)
from model_unfolder.evidence.models import SourceBundle


_CORPUS = Path(__file__).parent / "sable_test_corpus"
_PREFIX = """
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

class Attention:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        return F.scaled_dot_product_attention(q, k, v)
"""


def _expert_class(name="ShardedCompute", *, split=True, product=True):
    split_line = (
        "gate, up = F.linear(current, self.fused[index]).chunk(2, dim=-1)"
        if split else
        "gate = F.linear(current, self.fused[index])\n            up = current"
    )
    product_line = (
        "mixed = self.act(gate) * up" if product else
        "mixed = self.act(gate)"
    )
    return f"""
class {name}(nn.Module):
    def __init__(self, config):
        self.count = config.num_experts
        self.width = config.intermediate
        self.hidden = config.hidden
        self.fused = nn.Parameter(
            torch.empty(self.count, 2 * self.width, self.hidden))
        self.down = nn.Parameter(
            torch.empty(self.count, self.hidden, self.width))
        self.act = ACT2FN[config.hidden_act]
    def forward(self, hidden, routes, weights):
        output = torch.zeros_like(hidden)
        for index in routes:
            current = hidden[index]
            {split_line}
            {product_line}
            projected = F.linear(mixed, self.down[index])
            output.index_add_(0, index, projected)
        return output
"""


def _bundle(tmp_path, *, expert_source=None, block_init=None, extra=""):
    expert_source = expert_source or _expert_class()
    block_init = block_init or "self.compute = Routed(config)"
    block_init = textwrap.indent(
        textwrap.dedent(block_init).strip(), "        ")
    source = _PREFIX + expert_source + f"""
class Routed(nn.Module):
    def __init__(self, config):
        self.worker = ShardedCompute(config)
    def forward(self, x):
        return self.worker(x, x, x)

class Block(nn.Module):
    def __init__(self, config):
        self.attn = Attention(config)
{block_init}
    def forward(self, x):
        x = self.attn(x)
        return self.compute(x)

class Model(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapper(nn.Module):
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
""" + extra
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    block = decoder_block_path_at_root(index, root, allow_root_stage=True)
    assert block.status == "resolved", block.failures
    return index, root, block.value.block_occurrence


def test_fused_expert_requires_exact_stacked_two_lane_flow(tmp_path):
    index, root, block = _bundle(tmp_path)
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "fused_gate_up"
    assert result.value.owner_symbol.qualified_name == "ShardedCompute"
    assert [symbol.qualified_name for symbol in result.value.owner_trace] == [
        "Block", "Routed", "ShardedCompute"]
    assert len(result.value.construction_path) == 2


@pytest.mark.parametrize(("split", "product"), [
    (False, True),
    (True, False),
])
def test_missing_split_or_lane_product_cannot_prove_fused_expert(
        tmp_path, split, product):
    index, root, block = _bundle(
        tmp_path, expert_source=_expert_class(
            split=split, product=product))
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "failed"


def test_uninvoked_stacked_parameter_class_cannot_launder_the_block(tmp_path):
    distractor = _expert_class("UnusedStack")
    ordinary = """
class ShardedCompute(nn.Module):
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.intermediate * 2)
        self.down = nn.Linear(config.intermediate, config.hidden)
    def forward(self, x, routes, weights):
        gate, up = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)
"""
    index, root, block = _bundle(
        tmp_path, expert_source=ordinary, extra=distractor)
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "failed"


def test_two_factor_on_expert_axis_cannot_certify_a_fused_output_lane(tmp_path):
    source = _expert_class().replace(
        "self.count, 2 * self.width, self.hidden",
        "2 * self.count, self.width, self.hidden")
    index, root, block = _bundle(tmp_path, expert_source=source)
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "failed"


def test_two_reachable_fused_expert_variants_remain_ambiguous(tmp_path):
    second = _expert_class("OtherCompute")
    block_init = """
    if config.choice:
        self.compute = Routed(config)
    else:
        self.compute = OtherRouted(config)
"""
    extra = second + """
class OtherRouted(nn.Module):
    def __init__(self, config):
        self.worker = OtherCompute(config)
    def forward(self, x):
        return self.worker(x, x, x)
"""
    index, root, block = _bundle(
        tmp_path, block_init=block_init, extra=extra)
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_result_closure_rejects_cross_owner_parameter_forgery(tmp_path):
    index, root, block = _bundle(tmp_path)
    value = routed_expert_storage_at_block(index, root, block).value
    with pytest.raises(ValueError):
        replace(value, down_parameter=replace(
            value.down_parameter, owner=value.block_symbol))


@pytest.mark.parametrize(("slug", "expected"), [
    ("deepseek-v3", "resolved"),
    ("glm-4-5", "resolved"),
    ("gpt-oss-20b", "resolved"),
    ("llama-7b", "failed"),
])
def test_real_routed_expert_controls(slug, expected):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_routed_expert_storage_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == expected, result.failures
    if expected == "resolved":
        assert result.value.projection_mode == "fused_gate_up"


@pytest.mark.parametrize("slug", [
    "deepseek-v3",
    "glm-4-5",
    "gpt-oss-20b",
])
def test_parser_consumes_the_same_exact_expert_storage_result(slug):
    from model_unfolder import config_to_ir
    from model_unfolder.parser import _coerce

    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    result = context.reader_results[("decoder.ffn.expert_storage", ())]
    assert result.status == "resolved", result.failures
    moe_layers = [layer for layer in ir.layers if layer.ffn.kind == "moe"]
    assert moe_layers
    assert {
        layer.ffn.expert_projection_mode for layer in moe_layers
    } == {"fused_gate_up"}


@pytest.mark.parametrize("slug", [
    "deepseek-v3",
    "glm-4-5",
    "gpt-oss-20b",
])
def test_moe_block_does_not_prepend_an_ordinary_ffn_path(slug):
    """The MoE container may draw only mechanisms it actually owns."""
    from model_unfolder import config_to_ir
    from model_unfolder.parser import _coerce

    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    block = next(
        item
        for layer in ir.layers if layer.ffn.kind == "moe"
        for item in layer.blocks
        if item.get("id") == "ffn")
    child_ids = {item["id"] for item in block["children"]}
    assert not child_ids & {
        "gate_proj", "up_proj", "activation", "multiply", "down_proj"}
    assert {"router", "expert_1", "expert_k", "expert_kp1",
            "expert_n", "add_moe"} <= child_ids


def test_parser_and_conformance_share_one_exact_expert_result(monkeypatch):
    from model_unfolder import config_to_ir
    from model_unfolder.evidence import expert_storage as storage_module
    from model_unfolder.evidence.conformance import check_fact_conformance
    from model_unfolder.parser import _coerce

    config = json.loads(
        (_CORPUS / "gpt-oss-20b.json").read_text(
            encoding="utf-8"))["config"]
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    real_reader = storage_module.decoder_routed_expert_storage_for_path
    calls = []

    def counted(*args, **kwargs):
        calls.append((args[2], kwargs.get("allow_root_stage")))
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(
        storage_module, "decoder_routed_expert_storage_for_path", counted)
    ir = config_to_ir(cfg, parse_context=context)
    key = ("decoder.ffn.expert_storage", ())
    parsed_result = context.reader_results[key]
    assert calls == [((), True)]
    problems = check_fact_conformance(
        cfg, ir.to_dict(), bundle=context.source_bundle,
        program_index=context.program_index(), parse_context=context)
    assert not [problem for problem in problems
                if problem.kind == "wrong_storage"]
    assert context.reader_results[key] is parsed_result
    assert calls == [((), True)]
    fact = context.facts.records[
        "decoder.ffn.expert.expert_projection_mode"]
    assert fact.value == "fused_gate_up"
    assert fact.status == "code_proven"
    assert fact.source == "decoder_routed_expert_storage_for_path"


def test_renaming_classes_and_fields_does_not_change_the_mechanism(tmp_path):
    renamed = _expert_class("OpaqueB").replace(
        "self.fused", "self.matrix_a").replace(
        "self.down", "self.matrix_b")
    source = renamed.replace("class OpaqueB", "class ShardedCompute")
    index, root, block = _bundle(tmp_path, expert_source=source)
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "resolved", result.failures
