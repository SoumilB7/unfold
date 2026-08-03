"""U6 exact K/V-cache update evidence."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.attention import decoder_attention_cache_for_path
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.parser import _coerce


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _bundle(tmp_path, cache_lines: str, *, parameters: str = "memory=None"):
    source = f"""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Mixer:
        def __init__(self, config):
            self.qa = nn.Linear(config.hidden, config.heads * config.dim)
            self.kb = nn.Linear(config.hidden, config.heads * config.dim)
            self.vc = nn.Linear(config.hidden, config.heads * config.dim)
            self.out = nn.Linear(config.hidden, config.hidden)
            self.slot = 0
        def forward(self, x, {parameters}):
            alpha = self.qa(x)
            beta = self.kb(x)
            gamma = self.vc(x)
    __CACHE_LINES__
            value = F.scaled_dot_product_attention(alpha, beta, gamma)
            return self.out(value)

    class Cell:
        def __init__(self, config):
            self.mix = Mixer(config)
            self.ffn = nn.Linear(config.hidden, config.hidden)
        def forward(self, x):
            return self.ffn(self.mix(x))

    class Core:
        def __init__(self, config):
            self.layers = nn.ModuleList(
                [Cell(config) for _ in range(config.layers)])
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class Wrapper:
        base_model_prefix = "model"
        def __init__(self, config):
            self.model = Core(config)
    """
    source = textwrap.dedent(source).replace(
        "__CACHE_LINES__", textwrap.indent(cache_lines, "        "))
    path = Path(tmp_path) / "model.py"
    path.write_text(source, encoding="utf-8")
    return SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")


def _read(bundle):
    return decoder_attention_cache_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)


def test_exact_parameter_update_replacing_both_live_lanes_is_resolved(tmp_path):
    result = _read(_bundle(
        tmp_path,
        "if memory is not None:\n            beta, gamma = memory.update(beta, gamma, self.slot)"))
    assert result.status == "resolved", result.failures
    assert result.value.receiver_parameter == "memory"
    assert result.value.output_names == ("beta", "gamma")
    assert result.value.conditional is True
    assert len(set(result.value.input_projections)) == 2


def test_cache_parameter_or_config_declaration_cannot_author_update(tmp_path):
    result = _read(_bundle(tmp_path, "beta = beta"))
    assert result.status == "failed"


def test_unrelated_parameter_update_is_powerless(tmp_path):
    result = _read(_bundle(
        tmp_path,
        "if memory is not None:\n            memory.update({'seen': 1})"))
    assert result.status == "failed"


def test_unused_update_result_cannot_author_cache(tmp_path):
    result = _read(_bundle(
        tmp_path,
        "if memory is not None:\n            memory.update(beta, gamma, self.slot)"))
    assert result.status == "failed"


def test_generic_two_argument_update_without_layer_address_is_powerless(tmp_path):
    result = _read(_bundle(
        tmp_path,
        "if memory is not None:\n            beta, gamma = memory.update(beta, gamma)"))
    assert result.status == "failed"


def test_swapped_return_binding_cannot_launder_kv_lane_identity(tmp_path):
    result = _read(_bundle(
        tmp_path,
        "if memory is not None:\n            gamma, beta = memory.update(beta, gamma, self.slot)"))
    assert result.status == "failed"


def test_only_one_returned_lane_reaching_attention_is_insufficient(tmp_path):
    result = _read(_bundle(
        tmp_path,
        "before = gamma\n        if memory is not None:\n            beta, gamma = memory.update(beta, gamma, self.slot)\n        gamma = before"))
    assert result.status == "failed"


def test_unrelated_guard_cannot_certify_receiver_presence(tmp_path):
    result = _read(_bundle(
        tmp_path,
        "if enabled:\n            beta, gamma = memory.update(beta, gamma, self.slot)",
        parameters="memory=None, enabled=False"))
    assert result.status == "failed"


def test_parameter_and_lane_spellings_are_not_the_protocol(tmp_path):
    result = _read(_bundle(
        tmp_path,
        "if archive is not None:\n            beta, gamma = archive.update(beta, gamma, self.slot)",
        parameters="archive=None"))
    assert result.status == "resolved", result.failures
    assert result.value.receiver_parameter == "archive"


def test_cache_binding_rejects_forged_receiver_lane_and_layer_identity(tmp_path):
    result = _read(_bundle(
        tmp_path,
        "if memory is not None:\n            beta, gamma = memory.update(beta, gamma, self.slot)"))
    assert result.status == "resolved", result.failures
    value = result.value
    with pytest.raises(ValueError, match="parameter receiver"):
        replace(value, receiver_parameter="someone_else")
    with pytest.raises(ValueError, match="replacement lanes"):
        replace(value, output_names=("gamma", "beta"))
    with pytest.raises(ValueError, match="layer-index"):
        replace(value, layer_index=value.update_call.args[0])


@pytest.mark.parametrize("slug", [
    "llama-7b", "qwen3-8b", "bloom", "stablelm-2-1-6b",
])
def test_real_decoder_cache_examples(slug):
    data = json.loads((_CORPUS / f"{slug}.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    result = decoder_attention_cache_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.conditional is True


def test_unresolved_storage_never_fabricates_cache():
    from model_unfolder import config_to_ir
    from model_unfolder.opgraph import attention_region

    data = json.loads((_CORPUS / "deepseek-v3.json").read_text())
    cfg = _coerce(data["config"])
    context = ParseContext.build(cfg)
    result = decoder_attention_cache_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status != "resolved"
    ir = config_to_ir(cfg, parse_context=context)
    assert ir.layers and {layer.attention.cached for layer in ir.layers} == {None}
    assert "kv_cache" not in {
        op.id for op in attention_region(
            ir.to_dict()["layers"][0]["attention"], ir.hidden_size).ops}


def test_real_parser_fact_region_json_and_receipt_share_one_cache_claim():
    from model_unfolder import Diagram, config_to_ir
    from model_unfolder.expanded.attention import build_attention

    data = json.loads((_CORPUS / "llama-7b.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    diagram = Diagram(config_to_ir(_coerce(data["config"]), parse_context=context))
    assert diagram.ir.layers
    assert {layer.attention.cached for layer in diagram.ir.layers} == {True}
    fact = context.facts.typed["decoder.attention.cached"]
    assert fact.status == "code_proven" and fact.value is True

    serialized = diagram.to_ir()
    attn = serialized["layers"][0]["attention"]
    expanded = build_attention(attn, serialized["hidden_size"], "layers[0]")
    assert expanded["cache"]["enabled"] is True
    nodes = {node["id"] for node in expanded["operation_graph"]["nodes"]}
    assert "kv_cache" in nodes

    diagram.to_html(standalone=True)
    receipts = tuple(
        receipt for event in diagram.render_events()
        for receipt in event.receipts if receipt.fact_key == "cached")
    assert receipts
    assert {receipt.node_ids for receipt in receipts} == {("kv_cache",)}
    assert {receipt.mechanism for receipt in receipts} == {
        "attention_cache_update"}
