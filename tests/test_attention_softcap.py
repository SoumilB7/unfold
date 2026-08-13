"""U6 exact owner/config-bound attention-logit softcap controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.attention import (
    AttentionLogitSoftcapBinding,
    attention_logit_softcap_at_block,
    decoder_attention_logit_softcap_for_path,
)
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.decoder_block import decoder_block_path_at_root
from model_unfolder.evidence.models import SourceBundle


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _source(protocol: str, *, cap_argument="self.limit", sibling=""):
    return f"""
import torch
from torch import nn
from torch.nn import functional as F

def eager(module, query, key, value, cap=None):
    weights = torch.matmul(query, key.transpose(-1, -2))
{textwrap.indent(textwrap.dedent(protocol).strip(), '    ')}
    weights = F.softmax(weights, dim=-1)
    return torch.matmul(weights, value)

class Mixer:
    def __init__(self, config):
        self.config = config
        self.limit = self.config.attention_cap
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        return eager(self, query, key, value, cap={cap_argument})

{sibling}
class Cell:
    def __init__(self, config):
        self.mixer = Mixer(config)
    def forward(self, x):
        return self.mixer(x)

class Core:
    def __init__(self, config):
        self.items = nn.ModuleList([Cell(config) for _ in range(config.layers)])
    def forward(self, x):
        for item in self.items:
            x = item(x)
        return x

class Wrapper:
    base_model_prefix = "core"
    def __init__(self, config):
        self.core = Core(config)
"""


_EXACT = """
if cap is not None:
    weights = weights / cap
    weights = torch.tanh(weights)
    weights = weights * cap
"""


def _pipeline(tmp_path, protocol=_EXACT, *, cap_argument="self.limit",
              sibling=""):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(_source(
        protocol, cap_argument=cap_argument, sibling=sibling)),
        encoding="utf-8")
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
    return index, bundle, root, block.value.block_occurrence


def test_exact_guarded_divide_tanh_multiply_binds_config_path(tmp_path):
    index, bundle, root, block = _pipeline(tmp_path)
    result = attention_logit_softcap_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, AttentionLogitSoftcapBinding)
    assert result.value.config_path == ("attention_cap",)
    assert result.value.parameter == "cap"
    assert result.provenance[0].kind == "code_and_config"
    by_path = decoder_attention_logit_softcap_for_path(
        index, bundle, (), allow_root_stage=True)
    assert by_path.status == "resolved"
    assert by_path.value == result.value


@pytest.mark.parametrize("protocol", [
    """
if cap is not None:
    weights = weights / cap
    unused = torch.tanh(weights)
    weights = weights * cap
""",
    """
if cap is not None:
    weights = weights / cap
    weights = torch.tanh(weights)
""",
    """
if cap is not None:
    weights = weights / cap
    weights = torch.tanh(weights)
    weights = weights * other
""",
    """
weights = weights / cap
weights = torch.tanh(weights)
weights = weights * cap
""",
])
def test_incomplete_or_unrelated_tanh_protocol_cannot_prove_softcap(
        tmp_path, protocol):
    index, _bundle, root, block = _pipeline(tmp_path, protocol)
    assert attention_logit_softcap_at_block(
        index, root, block).status == "failed"


def test_dynamic_or_unbound_operand_cannot_become_config_evidence(tmp_path):
    index, _bundle, root, block = _pipeline(
        tmp_path, cap_argument="compute_limit()")
    assert attention_logit_softcap_at_block(
        index, root, block).status == "failed"


def test_sibling_softcap_cannot_vote_for_selected_attention(tmp_path):
    sibling = """
class Decoy:
    def __init__(self, config):
        self.limit = config.decoy_cap
    def forward(self, q, k, v):
        return eager(self, q, k, v, cap=self.limit)
"""
    index, _bundle, root, block = _pipeline(
        tmp_path, protocol="weights = weights + 0", sibling=sibling)
    assert attention_logit_softcap_at_block(
        index, root, block).status == "failed"


def test_real_gemma2_positive_and_llama_negative():
    outcomes = {}
    for slug in ("gemma-2-2b-it", "llama-7b"):
        config = json.loads((_CORPUS / f"{slug}.json").read_text())["config"]
        context = ParseContext.build(config)
        outcomes[slug] = decoder_attention_logit_softcap_for_path(
            context.program_index(), context.source_bundle, (),
            allow_root_stage=True)
    assert outcomes["gemma-2-2b-it"].status == "resolved"
    assert outcomes["gemma-2-2b-it"].value.config_path == (
        "attn_logit_softcapping",)
    assert outcomes["llama-7b"].status == "failed"


def test_parser_consumes_only_the_exact_code_bound_softcap_path(tmp_path):
    from model_unfolder import config_to_ir
    from model_unfolder.parser import _coerce

    index, bundle, _root, _block = _pipeline(tmp_path)
    del index  # parser must build/use its own call-local ProgramIndex
    config = json.loads((_CORPUS / "llama-7b.json").read_text())["config"]
    config.update({"hidden": 128, "layers": 2, "attention_cap": 17.0})
    cfg = _coerce(config)
    context = ParseContext(
        source_bundle=bundle,
        declared_decoderness="decoder_only_wrapper",
    )
    ir = config_to_ir(cfg, parse_context=context)
    assert ir.layers and all(
        layer.attention.logit_softcap == 17.0 for layer in ir.layers)
    typed = context.facts.typed["decoder.attention.logit_softcap"]
    assert typed.status == "code_and_config"
    assert typed.config_paths == ("attention_cap",)
    consumed = tuple(
        event for event in context.config_access.events
        if event.intent == "consumed"
        and event.fact_owner == "decoder.attention"
        and event.fact_key == "logit_softcap")
    assert len(consumed) == 1
    assert consumed[0].config_path == "attention_cap"


def test_raw_softcap_number_cannot_fabricate_mechanism_on_llama():
    from model_unfolder import config_to_ir
    from model_unfolder.parser import _coerce

    config = json.loads((_CORPUS / "llama-7b.json").read_text())["config"]
    config["attn_logit_softcapping"] = 99.0
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    assert ir.layers and all(
        layer.attention.logit_softcap is None for layer in ir.layers)
    assert "decoder.attention.logit_softcap" not in context.facts.typed
    assert not any(
        event.intent == "consumed"
        and event.fact_key == "logit_softcap"
        for event in context.config_access.events)


def test_real_attention_projector_emits_exact_softcap_receipt():
    from model_unfolder import config_to_ir
    from model_unfolder.diagram import Diagram
    from model_unfolder.evidence.receipts import join_obligation_receipts
    from model_unfolder.parser import _coerce

    config = json.loads((_CORPUS / "gemma-2-2b-it.json").read_text())["config"]
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    diagram = Diagram(config_to_ir(cfg, parse_context=context))
    diagram.to_html(standalone=True)
    events = diagram.render_events()
    receipts = tuple(
        receipt for event in events for receipt in event.receipts
        if receipt.fact_key == "logit_softcap")
    # Gemma-2 has sliding and global layer groups, so the one owner-level fact
    # is truthfully projected by both group drills.  Every emitted receipt must
    # nevertheless cite the same exact route and render context.
    assert len(receipts) == 2
    assert {receipt.owner for receipt in receipts} == {"decoder.attention"}
    assert {receipt.mechanism for receipt in receipts} == {
        "attention_logit_softcap"}
    assert {receipt.surface for receipt in receipts} == {"opgraph"}
    assert {receipt.structural_target for receipt in receipts} == {
        "attn_softcap"}
    assert {receipt.node_ids for receipt in receipts} == {("attn_softcap",)}
    receipt = receipts[0]
    ir = diagram.to_ir()
    access = ir["extras"]["config_access"]
    obligations = [
        item for item in access["projection_obligations"]
        if item["target"]["key"] == "logit_softcap"
    ]
    result = join_obligation_receipts(
        obligations, receipts,
        ir["extras"]["fact_provenance"],
        context_token=receipt.context_token,
    )
    assert result["findings"] == []
    assert result["receipted_targets"] == [(
        "decoder.attention", "decoder.attention.logit_softcap",
        "attention_logit_softcap")]


def test_softcap_dto_rejects_forged_path_and_order(tmp_path):
    index, _bundle, root, block = _pipeline(tmp_path)
    value = attention_logit_softcap_at_block(index, root, block).value
    with pytest.raises(TypeError, match="exact config path"):
        replace(value, config_path=())
    with pytest.raises(ValueError, match="precede softmax"):
        replace(value, divide_span=value.multiply_span,
                multiply_span=value.divide_span)
    with pytest.raises(ValueError, match="every decisive"):
        replace(value, spans=(value.score_call.span,))
