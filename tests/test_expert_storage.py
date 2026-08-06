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
    ExpertActivationEvidence,
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


def _split_expert_class(*, product=True, return_down=True):
    product_line = (
        "mixed = F.silu(left) * right" if product else
        "mixed = F.silu(left)"
    )
    return f"""
class OpaqueKernel(nn.Module):
    def __init__(self, config):
        self.count = config.num_experts
        self.width = config.intermediate
        self.hidden = config.hidden
        self.alpha = nn.Parameter(torch.empty(
            self.count * self.width, self.hidden))
        self.beta = nn.Parameter(torch.empty(
            self.count * self.width, self.hidden))
        self.omega = nn.Parameter(torch.empty(
            self.count * self.width, self.hidden))
    def forward(self, signal, one, two, three):
        left = signal.matmul(one)
        right = signal.matmul(two)
        {product_line}
        down = mixed.matmul(three.t())
        return {"down" if return_down else "signal"}

class ShardedCompute(nn.Module):
    def __init__(self, config):
        self.unit = OpaqueKernel(config)
        self.count = config.num_experts
        self.width = config.intermediate
        self.hidden = config.hidden
    def forward(self, hidden, routes, weights):
        output = torch.zeros_like(hidden)
        shape = (-1, self.width, self.hidden)
        for index in routes:
            first = self.unit.alpha.view(shape)[index]
            second = self.unit.beta.view(shape)[index]
            third = self.unit.omega.view(shape)[index]
            value = self.unit(hidden[index], first, second, third)
            output.index_add_(0, index, value)
        return output
"""


def _split_expert_with_dead_product():
    """A dead gate/up product must not certify a different down path."""
    return _split_expert_class().replace(
        "mixed = F.silu(left) * right\n        down = mixed.matmul(three.t())",
        "dead = F.silu(left) * right\n"
        "        mixed = left + right\n"
        "        down = mixed.matmul(three.t())",
    )


def _literal_swish_expert(*, distractions=False):
    source = _expert_class().replace(
        "self.act = ACT2FN[config.hidden_act]",
        "self.alpha = 1.702\n        self.limit = 7.0",
    ).replace(
        "mixed = self.act(gate) * up",
        "gate = gate.clamp(max=self.limit)\n"
        "            up = up.clamp(min=-self.limit, max=self.limit)\n"
        "            glu = gate * torch.sigmoid(gate * self.alpha)\n"
        "            mixed = (up + 1) * glu",
    )
    if distractions:
        source = source.replace(
            "mixed = (up + 1) * glu",
            "mixed = (up + 1) * glu\n"
            "            dead_clip = hidden.clamp(max=99)\n"
            "            dead_offset = hidden + 9",
        )
    return source


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
    assert result.value.activation is not None
    assert result.value.activation.kind is None
    assert result.value.activation.config_path == ("hidden_act",)


def test_unrelated_activation_cannot_launder_the_expert_gate(tmp_path):
    source = _expert_class().replace(
        "mixed = self.act(gate) * up",
        "dead = F.gelu(gate)\n            mixed = gate * up",
    )
    index, root, block = _bundle(tmp_path, expert_source=source)
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "fused_gate_up"
    assert result.value.activation is None


def test_activation_after_the_down_projection_cannot_qualify(tmp_path):
    source = _expert_class().replace(
        "mixed = self.act(gate) * up",
        "mixed = gate * up",
    ).replace(
        "projected = F.linear(mixed, self.down[index])",
        "projected = F.linear(mixed, self.down[index])\n"
        "            dead = self.act(projected)",
    )
    index, root, block = _bundle(tmp_path, expert_source=source)
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    assert result.value.activation is None


def test_split_expert_requires_repeated_selection_and_three_stage_dataflow(
        tmp_path):
    index, root, block = _bundle(
        tmp_path, expert_source=_split_expert_class())
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "split"
    assert len(result.value.input_parameters) == 2
    assert len(result.value.construction_path) == 3
    assert result.value.activation.kind == "silu"


def test_literal_formula_ignores_unrelated_clamp_and_offset(tmp_path):
    index, root, block = _bundle(
        tmp_path, expert_source=_literal_swish_expert(distractions=True))
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    activation = result.value.activation
    assert activation.kind == "swish"
    assert activation.alpha == 1.702
    assert activation.gate_clip == (None, 7.0)
    assert activation.up_clip == (-7.0, 7.0)
    assert activation.up_offset == 1.0


@pytest.mark.parametrize(("product", "return_down"), [
    (False, True),
    (True, False),
])
def test_split_parameter_trio_without_live_gate_down_flow_abstains(
        tmp_path, product, return_down):
    index, root, block = _bundle(
        tmp_path, expert_source=_split_expert_class(
            product=product, return_down=return_down))
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "failed"


def test_dead_gate_product_cannot_certify_an_unrelated_down_path(tmp_path):
    index, root, block = _bundle(
        tmp_path, expert_source=_split_expert_with_dead_product())
    result = routed_expert_storage_at_block(index, root, block)
    assert result.status == "failed"


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


def test_activation_evidence_closure_rejects_competing_authorities(tmp_path):
    index, root, block = _bundle(tmp_path)
    value = routed_expert_storage_at_block(index, root, block).value
    activation = value.activation
    with pytest.raises(ValueError):
        replace(activation, kind="silu")
    with pytest.raises(TypeError):
        replace(activation, gate_clip=(False, 7.0))
    with pytest.raises(ValueError):
        replace(activation, gate_clip=(None, None))
    with pytest.raises(ValueError):
        replace(activation, gate_clip=(8.0, 7.0))


def test_storage_closure_rejects_foreign_activation_provenance(tmp_path):
    index, root, block = _bundle(tmp_path)
    value = routed_expert_storage_at_block(index, root, block).value
    foreign = replace(value.spans[0], source=replace(
        value.spans[0].source, component_key="other"))
    with pytest.raises(ValueError):
        replace(value, activation=ExpertActivationEvidence(
            config_path=("hidden_act",), spans=(foreign,)))


@pytest.mark.parametrize(("slug", "expected"), [
    ("deepseek-v3", "resolved"),
    ("glm-4-5", "resolved"),
    ("gpt-oss-20b", "resolved"),
    ("dbrx-base", "resolved"),
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
        assert result.value.projection_mode == (
            "split" if slug == "dbrx-base" else "fused_gate_up")


@pytest.mark.parametrize(("slug", "kind", "path"), [
    ("deepseek-v3", None, ("hidden_act",)),
    ("glm-4-5", None, ("hidden_act",)),
    ("dbrx-base", None, ("ffn_config", "ffn_act_fn", "name")),
])
def test_real_expert_activation_dispatch_keeps_the_exact_config_path(
        slug, kind, path):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_routed_expert_storage_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.activation.kind is kind
    assert result.value.activation.config_path == path


def test_gpt_oss_expert_formula_is_literal_and_asymmetric():
    config = json.loads(
        (_CORPUS / "gpt-oss-20b.json").read_text(
            encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_routed_expert_storage_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    activation = result.value.activation
    assert activation.kind == "swish"
    assert activation.config_path == ()
    assert activation.alpha == 1.702
    assert activation.gate_clip == (None, 7.0)
    assert activation.up_clip == (-7.0, 7.0)
    assert activation.up_offset == 1.0


def test_dbrx_expert_activation_cannot_borrow_a_root_sibling():
    from model_unfolder import config_to_ir
    from model_unfolder.parser import _coerce

    config = json.loads(
        (_CORPUS / "dbrx-base.json").read_text(encoding="utf-8"))["config"]
    config["hidden_act"] = "gelu"
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    moe = next(layer.ffn for layer in ir.layers if layer.ffn.kind == "moe")
    assert moe.expert_activation_formula == {"kind": "silu"}
    fact = context.facts.records[
        "decoder.ffn.expert.expert_activation_formula"]
    assert fact.value == {"kind": "silu"}
    assert fact.status == "code_and_config"


def test_dbrx_source_literal_fallback_is_not_reported_as_checkpoint_config():
    from model_unfolder import config_to_ir
    from model_unfolder.parser import _coerce

    config = json.loads(
        (_CORPUS / "dbrx-base.json").read_text(encoding="utf-8"))["config"]
    config["ffn_config"]["ffn_act_fn"] = {}
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    moe = next(layer.ffn for layer in ir.layers if layer.ffn.kind == "moe")
    assert moe.expert_activation_formula == {"kind": "silu"}
    fact = context.facts.records[
        "decoder.ffn.expert.expert_activation_formula"]
    assert fact.value == {"kind": "silu"}
    assert fact.status == "code_proven"


@pytest.mark.parametrize("slug", [
    "deepseek-v3",
    "glm-4-5",
    "gpt-oss-20b",
    "dbrx-base",
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
    } == {"split" if slug == "dbrx-base" else "fused_gate_up"}
    expected = (
        {"kind": "swish", "alpha": 1.702,
         "gate_clip": (None, 7.0), "up_clip": (-7.0, 7.0),
         "up_offset": 1.0}
        if slug == "gpt-oss-20b" else {"kind": "silu"})
    assert {repr(layer.ffn.expert_activation_formula)
            for layer in moe_layers} == {repr(expected)}
    dense_layers = [layer for layer in ir.layers if layer.ffn.kind == "dense"]
    assert all(
        layer.ffn.expert_activation_formula is None for layer in dense_layers)
    fact = context.facts.records[
        "decoder.ffn.expert.expert_activation_formula"]
    assert fact.value == expected
    assert fact.status == (
        "code_proven" if slug == "gpt-oss-20b" else "code_and_config")
    pending = ir.extras["config_access"]["accessed_unconsumed_exact"]
    assert not [
        row for row in pending
        if row["path"].endswith((
            "hidden_act", "feed_forward_proj", "is_gated_act",
            "swiglu_limit"))]
    if slug == "gpt-oss-20b":
        block = next(
            item for layer in moe_layers for item in layer.blocks
            if item.get("id") == "ffn")
        expert = next(
            item for item in block["children"]
            if item.get("id") == "expert_1")
        assert [item["id"] for item in expert["children"]] == [
            "expert_hidden", "expert_gate_up_proj", "expert_gate_up_split",
            "expert_gate_clip", "expert_act", "expert_up_clip",
            "expert_up_offset", "expert_mul", "expert_down_proj",
        ]


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
