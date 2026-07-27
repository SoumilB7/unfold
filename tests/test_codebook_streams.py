"""U3-G — exact root/stage ownership for multi-codebook aggregation."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.codebook_streams import (
    CodebookStreamsEvidence,
    decoder_codebook_streams_for_path,
)
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _source(
    *,
    embedding_output="self.embeds[i](tokens[i])",
    embedding_iterable="range(count)",
    embedding_filter="",
    head_output="head(hidden)",
    head_iterable="self.heads",
    aggregate="torch.stack",
    extra_stage="",
    extra_root="",
    sum_parameter="",
):
    return f"""
        import torch
        from torch import nn

        class Block:
            def forward(self, hidden):
                return hidden

        class Stage:
            def __init__(self):
                self.embeds = nn.ModuleList(
                    [nn.Embedding(16, 8) for _ in range(4)])
                self.layers = nn.ModuleList(
                    [Block() for _ in range(2)])
                {extra_stage}

            def forward(self, tokens, count{sum_parameter}):
                hidden = sum(
                    {embedding_output}
                    for i in {embedding_iterable}{embedding_filter})
                for layer in self.layers:
                    hidden = layer(hidden)
                return hidden

        class Wrapper:
            base_model_prefix = "model"

            def __init__(self):
                self.model = Stage()
                self.heads = nn.ModuleList(
                    [nn.Linear(8, 16) for _ in range(4)])
                {extra_root}

            def forward(self, tokens, count):
                hidden = self.model(tokens, count)
                return {aggregate}(
                    [{head_output} for head in {head_iterable}])
    """


def _reader(tmp_path, source=None):
    path = tmp_path / "model.py"
    path.write_text(
        textwrap.dedent(source or _source()), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"})
    return decoder_codebook_streams_for_path(
        pi.build_program_index(bundle), bundle, ())


def test_exact_root_and_stage_lanes_resolve_independently(tmp_path):
    result = _reader(tmp_path)
    assert result.status == "resolved", result
    value = result.value
    assert value.embeddings_summed is True
    assert value.heads_stacked is True
    assert value.embedding_sum.owner_occurrence == value.stage_occurrence
    root = value.component_root.graph.root.occurrence
    assert value.head_stack.owner_occurrence == root
    assert value.embedding_sum.element_primitive.qualified_target \
        == "torch.nn.Embedding"
    assert value.head_stack.element_primitive.qualified_target \
        == "torch.nn.Linear"
    assert value.embedding_sum.comprehension.clauses[0].target.name == "i"
    assert value.head_stack.comprehension.clauses[0].target.name == "head"


def test_wrong_embedding_bank_cannot_borrow_the_real_container(tmp_path):
    result = _reader(
        tmp_path,
        _source(
            extra_stage=(
                "self.other = nn.ModuleList("
                "[nn.Embedding(16, 8) for _ in range(4)])"),
            embedding_output="self.other[i](tokens[i])"))
    assert result.status == "ambiguous"


def test_head_target_must_be_bound_by_the_exact_bank_clause(tmp_path):
    result = _reader(
        tmp_path, _source(head_iterable="other_heads"))
    assert result.status == "incomplete"
    assert result.value.embeddings_summed is True
    assert result.value.heads_stacked is None


def test_embedding_index_must_be_the_comprehension_target(tmp_path):
    result = _reader(
        tmp_path, _source(embedding_output="self.embeds[j](tokens[i])"))
    assert result.status == "incomplete"
    assert result.value.embeddings_summed is None
    assert result.value.heads_stacked is True


def test_filtered_comprehension_cannot_claim_all_streams(tmp_path):
    result = _reader(
        tmp_path, _source(embedding_filter=" if i > 0"))
    assert result.status == "incomplete"
    assert result.value.embeddings_summed is None


def test_sum_must_be_the_unshadowed_builtin(tmp_path):
    result = _reader(
        tmp_path, _source(sum_parameter=", sum=None"))
    assert result.status == "incomplete"
    assert result.value.embeddings_summed is None
    assert result.value.heads_stacked is True


def test_stack_must_resolve_through_the_exact_torch_import(tmp_path):
    result = _reader(tmp_path, _source(aggregate="sum"))
    assert result.status == "incomplete"
    assert result.value.embeddings_summed is True
    assert result.value.heads_stacked is None


def test_sibling_class_bank_cannot_vote_for_the_selected_stage(tmp_path):
    source = _source(
        embedding_output="tokens[i]",
        extra_root="")
    source += """
        class Sibling:
            def __init__(self):
                self.bank = nn.ModuleList(
                    [nn.Embedding(16, 8) for _ in range(4)])

            def forward(self, tokens, count):
                return sum(
                    self.bank[i](tokens[i]) for i in range(count))
    """
    result = _reader(tmp_path, source)
    assert result.status == "incomplete"
    assert result.value.embeddings_summed is None
    assert result.value.heads_stacked is True


def test_two_exact_embedding_banks_are_ambiguity_not_a_pick(tmp_path):
    result = _reader(
        tmp_path,
        _source(
            extra_stage=(
                "self.other = nn.ModuleList("
                "[nn.Embedding(16, 8) for _ in range(4)])")))
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_real_musicgen_positive_and_llama_negative():
    musicgen = json.loads(
        (_CORPUS / "musicgen-small.json").read_text())["config"]
    music_context = ParseContext.build(musicgen)
    result = decoder_codebook_streams_for_path(
        music_context.program_index(), music_context.source_bundle,
        ("decoder",))
    assert result.status == "resolved", result
    assert result.value.embeddings_summed is True
    assert result.value.heads_stacked is True

    llama = json.loads(
        (_CORPUS / "llama-7b.json").read_text())["config"]
    llama_context = ParseContext.build(llama)
    negative = decoder_codebook_streams_for_path(
        llama_context.program_index(), llama_context.source_bundle, ())
    assert negative.status == "absent"


def test_evidence_closure_rejects_lane_owner_laundering(tmp_path):
    value = _reader(tmp_path).value
    with pytest.raises(ValueError):
        CodebookStreamsEvidence(
            value.component_root,
            value.component_root.graph.root.occurrence,
            value.embedding_sum,
            value.head_stack)
    with pytest.raises(ValueError):
        replace(
            value.embedding_sum,
            owner_occurrence=value.component_root.graph.root.occurrence)


def test_lane_closure_rejects_self_certified_aggregate_argument(tmp_path):
    value = _reader(tmp_path).value
    call = value.head_stack.aggregate_call
    forged = replace(call, args=value.embedding_sum.aggregate_call.args)
    with pytest.raises(ValueError):
        replace(value.head_stack, aggregate_call=forged)
