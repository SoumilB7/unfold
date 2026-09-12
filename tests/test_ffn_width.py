"""U7 exact-owner FFN intermediate-width controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest
from transformers import AutoConfig

from model_unfolder.evidence.context import ParseContext, slot_parse_context
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.ffn_mechanism import (
    ConfigSelectedFFNMechanism,
    decoder_ffn_mechanism_for_path,
)
from model_unfolder.evidence.ffn_width import (
    FFNIntermediateWidth,
    decoder_ffn_intermediate_width_for_path,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.parser import config_to_ir


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def test_conflicting_expression_premises_never_become_code_only_evidence():
    from model_unfolder.evidence.expression_eval import (
        EvaluatedExpression, combined)
    from model_unfolder.evidence.program_index import ExprNode

    expression = ExprNode(kind="constant", const_value=3)
    left = EvaluatedExpression(1, ((("width",), 1),))
    right = EvaluatedExpression(2, ((("width",), 2),))
    assert combined(3, expression, left, right) is None
    boolean = EvaluatedExpression(True, ((("width",), True),))
    integer = EvaluatedExpression(1, ((("width",), 1),))
    assert combined(1, expression, boolean, integer) is None


def _read(config):
    context = ParseContext.build(config, source="local")
    return decoder_ffn_intermediate_width_for_path(
        context.program_index(), context.source_bundle, (), config,
        allow_root_stage=True)


@pytest.mark.parametrize("model_type,expected", [
    ("gptj", 16384),
    ("codegen", 16384),
    ("gpt2", 3072),
    ("bloom", 256),
])
def test_real_source_default_and_inline_widths_are_exact(model_type, expected):
    result = _read(AutoConfig.for_model(model_type))
    assert result.status == "resolved", result.failures
    assert result.value.value == expected
    assert result.value.premises
    assert result.value.spans


def test_gpt2_fact_cites_checkpoint_spellings_not_runtime_properties():
    config = AutoConfig.for_model("gpt2").to_dict()
    context = ParseContext.build(config)
    config_to_ir(config, parse_context=context)
    fact = context.facts.typed["decoder.ffn.intermediate_size"]
    assert fact.config_paths == ("n_inner", "n_embd")
    assert "hidden_size" not in fact.config_paths


def test_t5_width_consumes_the_exact_config_selected_ffn_branch():
    """The wrapper's gated/dense construction alternatives are resolved once.

    The selected branch is represented as the root of its isolated exact owner
    graph, so downstream width evaluation must consume that root rather than
    independently rerunning the unresolved wrapper selection.
    """
    root_config = json.loads(
        (_CORPUS / "fluxtransformer2dmodel.json").read_text(
            encoding="utf-8"))["config"]
    sub_config = root_config["_text_encoder_configs"]["text_encoder_2"]
    context = slot_parse_context(
        ParseContext.build(root_config), "text_encoder_2",
        document=sub_config)
    document = dict(context.class_defaults or {})
    document.update(sub_config)
    mechanism = decoder_ffn_mechanism_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True,
        config_selector=lambda path: document.get(path[-1]))
    assert mechanism.status == "resolved", mechanism.failures
    assert isinstance(mechanism.value, ConfigSelectedFFNMechanism)
    assert mechanism.value.gated is True

    result = decoder_ffn_intermediate_width_for_path(
        context.program_index(), context.source_bundle, (), document,
        allow_root_stage=True, mechanism_result=mechanism)
    assert result.status == "resolved", result.failures
    assert result.value.value == 10240
    assert result.value.premises == ((("d_ff",), 10240),)


def test_musicgen_repeated_block_width_uses_its_exact_config_binding():
    """A symbolic repeated-block construction need not expose a one-shot
    constructor environment when the exact owner already binds ``config``.
    Unbound constructor formals stay unknown; the directly bound ``ffn_dim``
    expression remains exact.
    """
    from test_support import MUSICGEN_SMALL

    context = ParseContext.build(MUSICGEN_SMALL)
    config_path = ("decoder",)
    mechanism = decoder_ffn_mechanism_for_path(
        context.program_index(), context.source_bundle, config_path,
        allow_root_stage=True,
        config_selector=lambda path: MUSICGEN_SMALL["decoder"].get(path[-1]))
    assert mechanism.status == "resolved", mechanism.failures
    result = decoder_ffn_intermediate_width_for_path(
        context.program_index(), context.source_bundle, config_path,
        MUSICGEN_SMALL, allow_root_stage=True,
        mechanism_result=mechanism)
    assert result.status == "resolved", result.failures
    assert result.value.value == 4096
    assert result.value.premises == ((("decoder", "ffn_dim"), 4096),)


def test_explicit_gptj_inner_width_wins_through_the_same_code_expression():
    config = AutoConfig.for_model("gptj")
    config.n_inner = 9000
    result = _read(config)
    assert result.status == "resolved", result.failures
    assert result.value.value == 9000
    assert result.value.premises == ((('n_inner',), 9000),)


def test_missing_condition_operand_is_not_treated_as_explicit_none():
    config = AutoConfig.for_model("gptj").to_dict()
    config.pop("n_inner", None)
    result = _read(config)
    assert result.status == "failed"
    assert result.value is None


def test_declared_llama_width_round_trips_without_a_special_case():
    config = AutoConfig.for_model("llama")
    result = _read(config)
    assert result.status == "resolved", result.failures
    assert result.value.value == config.intermediate_size
    assert result.value.premises == (
        (("intermediate_size",), config.intermediate_size),)


def test_parser_withholds_present_width_when_exact_source_is_missing():
    """A plausible config leaf cannot author geometry by itself.  This pins
    the parser integration seam, rather than only the standalone reader."""
    config = {
        "model_type": "unknown_width_model",
        "architectures": ["UnknownWidthForCausalLM"],
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "intermediate_size": 257,
        "vocab_size": 128,
    }
    context = ParseContext(
        source_bundle=SourceBundle(source="local", files=()),
        source="local")
    ir = config_to_ir(config, parse_context=context)
    assert ir.layers[0].ffn.intermediate_size is None
    assert "decoder.ffn.intermediate_size" not in (
        ir.extras.get("fact_provenance") or {})


def test_width_dto_rejects_value_and_provenance_forgery():
    result = _read(AutoConfig.for_model("bloom"))
    assert result.status == "resolved"
    value = result.value
    with pytest.raises(ValueError):
        replace(value, value=0)
    with pytest.raises(ValueError):
        replace(value, premises=(value.premises[0], value.premises[0]))
    with pytest.raises(ValueError):
        replace(value, premises=(((), value.value),))
    with pytest.raises(ValueError):
        FFNIntermediateWidth(
            value.owner_occurrence, value.owner_symbol, value.value,
            value.premises, ())


def test_rival_up_projection_widths_do_not_certify_the_down_width(tmp_path):
    """The down projection alone cannot vouch for a gated pair whose two
    upstream lanes disagree.  Every exact affine lane must join on one width.
    """
    source = """
import torch
from torch import nn
from torch.nn import functional as F
class Attention:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        return torch.matmul(F.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1), v)
class FeedForward:
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden, config.wide_a)
        self.up = nn.Linear(config.hidden, config.wide_b)
        self.down = nn.Linear(config.wide_a, config.hidden)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
class Block:
    def __init__(self, config):
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
    def forward(self, x):
        return self.ffn(self.attn(x))
class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
"""
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = pi.build_program_index(bundle)
    result = decoder_ffn_intermediate_width_for_path(
        index, bundle, (),
        {"hidden": 8, "wide_a": 32, "wide_b": 64, "layers": 2},
        allow_root_stage=True)
    assert result.status == "failed"
    assert result.value is None


def test_exact_literal_ffn_width_is_code_proven_without_config_authority(
        tmp_path):
    source = """
import torch
from torch import nn
from torch.nn import functional as F
class Attention:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        return torch.matmul(F.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1), v)
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, 32)
        self.down = nn.Linear(32, config.hidden)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))
class Block:
    def __init__(self, config):
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
    def forward(self, x):
        return self.ffn(self.attn(x))
class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
"""
    path = tmp_path / "model_literal.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    config = {
        "model_type": "literal_ffn", "architectures": ["Wrapper"],
        "hidden": 8, "hidden_size": 8, "layers": 1,
        "num_hidden_layers": 1, "intermediate_size": 999,
        "vocab_size": 64,
    }
    context = ParseContext(
        source_bundle=bundle, source="local",
        _program_index=pi.build_program_index(bundle))
    result = decoder_ffn_intermediate_width_for_path(
        context.program_index(), bundle, (), config, allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.value == 32
    assert result.value.premises == ()

    ir = config_to_ir(config, parse_context=context)
    assert ir.layers[0].ffn.intermediate_size == 32
    fact = context.facts.typed["decoder.ffn.intermediate_size"]
    assert fact.status == "code_proven"
    assert fact.config_paths == ()
