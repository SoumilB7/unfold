"""U3 output-lineage repeated-stage boundary counterexamples."""
from __future__ import annotations

import textwrap
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.output_repeated_stage import (
    OutputLineageRelation,
    OutputRepeatedStage,
    resolve_output_repeated_stage,
)
from model_unfolder.evidence.program_index import build_program_index


_BASE = """
class Cell:
    def __init__(self, config):
        pass
    def forward(self, x):
        return x

class Prep:
    def __init__(self, config):
        pass
    def forward(self, x):
        return x

class Post:
    def __init__(self, config):
        pass
    def forward(self, x):
        return x
"""

_STAGE = """
class Stage:
    def __init__(self, config):
        self.layers = ModuleList([Cell(config) for _ in range(config.depth)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
"""

_PREFIX = _BASE + _STAGE


def _read(tmp_path, forward, *, init="", extra=""):
    source = _PREFIX + extra + """
class Wrapper:
    def __init__(self, config):
        self.prep = Prep(config)
        self.stage = Stage(config)
        self.post = Post(config)
""" + textwrap.indent(textwrap.dedent(init).strip(), "        ") + """
    def forward(self, x):
""" + textwrap.indent(textwrap.dedent(forward).strip(), "        ")
    path = tmp_path / "modeling_output_stage.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    return index, root, resolve_output_repeated_stage(
        index, root, root.graph.root.occurrence)


def test_exact_transformed_output_lineage_resolves_the_repeated_stage(tmp_path):
    index, root, result = _read(tmp_path, """
        hidden = self.prep(x)
        hidden = self.stage(hidden)
        hidden = hidden[0]
        hidden = self.post(hidden)
        return Output(last_hidden_state=hidden)
    """)
    assert result.status == "resolved", result.failures
    assert root.graph.node_for(
        result.value.stage_occurrence).symbol.qualified_name == "Stage"
    assert result.value.repeated_child.child_symbol.qualified_name == "Cell"
    assert {item.kind for item in result.value.lineage} == {
        "proven_def_use", "transformed_candidate"}


def test_class_field_and_local_renaming_do_not_change_the_address_proof(
        tmp_path):
    _index, root, result = _read(
        tmp_path,
        """
        renamed = self.before(x)
        renamed = self.body(renamed)
        renamed = renamed[0]
        renamed = self.after(renamed)
        return Output(last_hidden_state=renamed)
        """,
        init="""
        self.before = Prep(config)
        self.body = RenamedStage(config)
        self.after = Post(config)
        """,
        extra=_STAGE.replace("class Stage", "class RenamedStage"),
    )
    assert result.status == "resolved", result.failures
    assert root.graph.node_for(
        result.value.stage_occurrence).symbol.qualified_name == "RenamedStage"


def test_two_output_reaching_repeated_stages_are_ambiguity_not_a_pick(
        tmp_path):
    _index, _root, result = _read(
        tmp_path,
        """
        hidden = self.stage(x)
        hidden = self.other(hidden)
        return Output(last_hidden_state=hidden)
        """,
        init="self.other = OtherStage(config)",
        extra=_STAGE.replace("class Stage", "class OtherStage"),
    )
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_uninvoked_repeated_sibling_cannot_vote(tmp_path):
    _index, root, result = _read(
        tmp_path,
        """
        hidden = self.stage(x)
        return Output(last_hidden_state=hidden)
        """,
        init="self.other = OtherStage(config)",
        extra=_STAGE.replace("class Stage", "class OtherStage"),
    )
    assert result.status == "resolved"
    assert root.graph.node_for(
        result.value.stage_occurrence).symbol.qualified_name == "Stage"


def test_unresolved_self_child_on_the_output_path_blocks_selection(tmp_path):
    _index, _root, result = _read(
        tmp_path,
        """
        hidden = self.stage(x)
        hidden = self.dynamic(hidden)
        return Output(last_hidden_state=hidden)
        """,
        init=(
            "self.dynamic = OtherStage(config) if config.flag "
            "else RivalStage(config)"
        ),
        extra=(
            _STAGE.replace("class Stage", "class OtherStage")
            + _STAGE.replace("class Stage", "class RivalStage")
        ),
    )
    assert result.status == "failed"
    assert "unresolved self-child" in result.failures[0].detail


def test_returning_a_bare_name_is_not_upgraded_to_a_structured_sink(tmp_path):
    _index, _root, result = _read(tmp_path, """
        hidden = self.stage(x)
        return hidden
    """)
    assert result.status == "failed"
    assert "returned expression" in result.failures[0].detail


def test_result_closure_rejects_a_broken_lineage(tmp_path):
    _index, _root, result = _read(tmp_path, """
        hidden = self.stage(x)
        hidden = self.post(hidden)
        return Output(last_hidden_state=hidden)
    """)
    assert result.status == "resolved"
    value = result.value
    with pytest.raises(ValueError):
        OutputRepeatedStage(
            value.owner_occurrence,
            value.stage_occurrence,
            value.invocation,
            value.repeated_child,
            value.return_sink,
            tuple(reversed(value.lineage)),
            value.spans,
        )
    relation = value.lineage[0]
    with pytest.raises(ValueError):
        OutputLineageRelation(
            relation.source, relation.source, relation.kind, relation.spans)


def test_real_clip_structured_output_resolves_its_exact_encoder_stage():
    from model_unfolder.evidence.context import ParseContext, slot_parse_context
    from model_unfolder.evidence.decoder_block import decoder_block_path_at_root
    from model_unfolder.evidence.decoder_norm import decoder_norm_kind_for_path
    from model_unfolder.evidence.ffn_mechanism import (
        decoder_ffn_mechanism_for_path,
    )
    from model_unfolder.evidence.projection_bias import (
        decoder_attention_bias_for_path,
    )

    config = json.loads(
        (Path(__file__).parent / "sable_test_corpus"
         / "stable-diffusion-xl-base-1-0.json").read_text(
             encoding="utf-8"))["config"]
    outer = ParseContext.build(config)
    context = slot_parse_context(outer, "text_encoder")
    assert context is not None
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    assert root.status == "resolved"
    result = resolve_output_repeated_stage(
        index, root, root.graph.root.occurrence)
    assert result.status == "resolved", result.failures
    stage = root.graph.node_for(result.value.stage_occurrence)
    assert stage is not None
    assert stage.symbol.qualified_name == "CLIPEncoder"
    assert result.value.repeated_child.child_symbol.qualified_name \
        == "CLIPEncoderLayer"
    integrated = decoder_block_path_at_root(
        index, root, allow_root_stage=True)
    assert integrated.status == "resolved", integrated.failures
    assert root.graph.node_for(
        integrated.value.stage_occurrence).symbol == stage.symbol
    assert root.graph.node_for(
        integrated.value.block_occurrence).symbol.qualified_name \
        == "CLIPEncoderLayer"
    ffn = decoder_ffn_mechanism_for_path(
        index, context.source_bundle, (), allow_root_stage=True)
    assert ffn.status == "resolved", ffn.failures
    assert ffn.value.owner_symbol.qualified_name == "CLIPMLP"
    assert ffn.value.projection_mode == "dense"
    norm = decoder_norm_kind_for_path(
        index, context.source_bundle, (), allow_root_stage=True)
    assert norm.status == "resolved", norm.failures
    assert norm.value == "layernorm"
    attention_bias = decoder_attention_bias_for_path(
        index, context.source_bundle, (), allow_root_stage=True)
    assert attention_bias.status == "resolved", attention_bias.failures
    assert attention_bias.value.value is True
