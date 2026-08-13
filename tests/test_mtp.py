from __future__ import annotations

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.mtp import decoder_mtp_construction_for_path
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.receipts import join_obligation_receipts
from model_unfolder.parser import config_to_ir
from model_unfolder.diagram import Diagram
from model_unfolder.renderers.html.render_context import (
    RenderContext, activate_render_context,
)


def _source(*, share_embedding: bool = True, share_head: bool = True,
            include_concat: bool = True, include_block: bool = True) -> str:
    predictor_embedding = (
        ""
        if share_embedding else
        "        self.own_embedding = Embedding(config.vocab_size, 4)\n"
    )
    predictor_head = (
        "        self.head = head\n"
        if share_head else
        "        self.head = Linear(4, config.vocab_size)\n"
    )
    second_input = "embedding" if share_embedding else "token_ids"
    embed_value = "embedding" if share_embedding else "self.own_embedding(token_ids)"
    concat = (
        "        joined = torch.cat((a, b), dim=-1)\n"
        if include_concat else
        "        joined = a + b\n"
    )
    block = (
        "        x = self.block(x)\n"
        if include_block else
        ""
    )
    stage_embedding = (
        "        embedding = self.embed(token_ids)\n"
        "        for predictor in self.predictors:\n"
        "            logits = predictor(hidden, embedding)\n"
        if share_embedding else
        "        for predictor in self.predictors:\n"
        "            logits = predictor(hidden, token_ids)\n"
    )
    return f"""import torch
from torch.nn import Embedding, LayerNorm, Linear, ModuleList

class Layer:
    def __init__(self, config):
        self.proj = Linear(4, 4)
    def forward(self, hidden):
        return self.proj(hidden)

class Predictor:
    def __init__(self, config, head):
        self.hnorm = LayerNorm(4)
        self.enorm = LayerNorm(4)
        self.proj = Linear(8, 4)
        self.block = Layer(config)
{predictor_embedding}{predictor_head}
    def forward(self, hidden, {second_input}):
        a = self.hnorm(hidden)
        b = self.enorm({embed_value})
{concat}        x = self.proj(joined)
{block}        return self.head(x)

class Root:
    base_model_prefix = ""
    def __init__(self, config):
        self.embed = Embedding(config.vocab_size, 4)
        self.head = Linear(4, config.vocab_size)
        self.layers = ModuleList(
            [Layer(config) for _ in range(config.num_layers)])
        self.predictors = ModuleList(
            [Predictor(config, self.head)
             for _ in range(config.num_predictors)])
    def forward(self, token_ids):
        hidden = self.embed(token_ids)
        for layer in self.layers:
            hidden = layer(hidden)
{stage_embedding}        return self.head(hidden), logits
"""


def _read(tmp_path, **changes):
    path = tmp_path / "model.py"
    path.write_text(_source(**changes), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"}, architecture="Root")
    return decoder_mtp_construction_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)


def _parse(tmp_path, **changes):
    path = tmp_path / "model.py"
    path.write_text(_source(**changes), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"}, architecture="Root")
    config = {
        "architectures": ["Root"], "model_type": "synthetic_mtp",
        "is_decoder": True, "vocab_size": 32, "hidden_size": 4,
        "intermediate_size": 8, "num_hidden_layers": 2,
        "num_layers": 2, "num_attention_heads": 1,
        "num_key_value_heads": 1, "num_predictors": 2,
        "hidden_act": "relu", "tie_word_embeddings": False,
        "max_position_embeddings": 32,
    }
    context = ParseContext(source_bundle=bundle, source="local")
    ir = config_to_ir(config, parse_context=context)
    return ir, context


def test_exact_shared_mtp_construction_resolves(tmp_path):
    result = _read(tmp_path)
    assert result.status == "resolved", result.failures
    assert result.value.count_path == ("num_predictors",)
    assert result.value.shares_embedding is True
    assert result.value.shares_output_head is True


def test_count_without_complete_operation_path_is_powerless(tmp_path):
    assert _read(tmp_path, include_concat=False).status == "absent"
    assert _read(tmp_path, include_block=False).status == "absent"


def test_complete_field_and_class_rename_does_not_change_result(tmp_path):
    source = (_source()
              .replace("Predictor", "FutureUnit")
              .replace("predictors", "future_units")
              .replace("hnorm", "left_op")
              .replace("enorm", "right_op")
              .replace("proj", "map_op")
              .replace("block", "cell_op")
              .replace("head", "readout"))
    path = tmp_path / "renamed.py"
    path.write_text(source, encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"}, architecture="Root")
    result = decoder_mtp_construction_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures


def test_unshared_variants_are_not_misreported_as_shared(tmp_path):
    own_embedding = _read(tmp_path, share_embedding=False)
    assert own_embedding.status == "resolved", own_embedding.failures
    assert own_embedding.value.shares_embedding is False
    assert own_embedding.value.shares_output_head is True

    own_head = _read(tmp_path, share_head=False)
    assert own_head.status == "resolved", own_head.failures
    assert own_head.value.shares_embedding is True
    assert own_head.value.shares_output_head is False


def test_exact_mtp_path_authors_one_fact_block_and_real_receipt(tmp_path):
    ir, _context = _parse(tmp_path)
    extras = ir.extras or {}
    facts = extras.get("fact_provenance") or {}
    expected = {
        "num_modules": 2,
        "shares_embedding": True,
        "shares_output_head": True,
        "hidden_norm_kind": "LayerNorm",
        "embedding_norm_kind": "LayerNorm",
        "reuses_stage_block_class": True,
    }
    assert facts["decoder.mtp_modules"] == {
        "value": expected, "status": "code_and_config",
        "source": "decoder_mtp_construction_for_path",
    }
    blocks = extras["render"]["model_blocks"]
    mtp = [block for block in blocks if block.get("id") == "mtp"]
    assert len(mtp) == 1
    assert {key: mtp[0]["detail"][key] for key in expected} == expected
    # The source proves equality of the repeated block CLASS, not which
    # heterogeneous occurrence may donate its internals.  Never borrow layer 0.
    inner = next(item for item in mtp[0]["children"]
                 if item.get("id") == "mtp_block")
    assert "children" not in inner and "view" not in inner

    diagram = Diagram(ir)
    render = RenderContext(theme="teal", fact_rows=dict(facts))
    with activate_render_context(render):
        diagram.to_html(standalone=True)
    receipts = [receipt for event in render.events
                for receipt in event.receipts
                if receipt.fact_id == "decoder.mtp_modules"]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert (
        receipt.surface, receipt.structural_target, receipt.node_ids,
        receipt.projector_symbol, receipt.context_token,
    ) == (
        "html", "mtp_modules", ("mtp",),
        "renderers.html.views._build_architecture_view",
        render.context_token,
    )
    obligations = [item for item in
                   extras["config_access"]["projection_obligations"]
                   if item["target"]["key"] == "mtp_modules"]
    assert len(obligations) == 1
    assert obligations[0]["source"]["path"] == "num_predictors"
    joined = join_obligation_receipts(
        obligations, receipts, facts, context_token=render.context_token)
    assert joined["findings"] == []


def test_count_only_or_incomplete_source_never_authors_mtp(tmp_path):
    for changes in ({"include_concat": False}, {"include_block": False}):
        ir, _context = _parse(tmp_path, **changes)
        extras = ir.extras or {}
        assert "decoder.mtp_modules" not in (
            extras.get("fact_provenance") or {})
        assert not [block for block in
                    extras["render"]["model_blocks"]
                    if block.get("id") == "mtp"]
