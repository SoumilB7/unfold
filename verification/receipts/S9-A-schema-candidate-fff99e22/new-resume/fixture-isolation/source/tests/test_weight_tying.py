"""U3-G — exact returned-head ↔ stack-feeding-embedding tying."""
from __future__ import annotations

from test_support.s9_fixtures.weight_tying import _reader, _source

from dataclasses import replace
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.weight_tying import (
    ManualWeightTyingEvidence,
    manual_weight_tying_for_path,
)


_CORPUS = Path(__file__).parent / "sable_test_corpus"






def test_exact_returned_head_and_stack_feeding_embedding_resolve(tmp_path):
    result = _reader(tmp_path)
    assert result.status == "resolved", result
    value = result.value
    assert value.output_endpoint.primitive.qualified_target \
        == "torch.nn.Linear"
    assert value.embedding_endpoint.primitive.qualified_target \
        == "torch.nn.Embedding"
    assert value.output_endpoint.chain == ("output",)
    assert value.embedding_endpoint.chain == ("core", "tokens")
    assert value.embedding_edge.proof_kind == "versioned_def_use"


def test_class_and_field_renaming_preserves_the_same_mechanism(tmp_path):
    result = _reader(
        tmp_path,
        _source(
            wrapper="Outer", stage="Inner", block="Cell",
            model_field="body", head_field="emit",
            embedding_field="lookup"),
        architecture="Outer")
    assert result.status == "resolved"
    assert result.value.output_endpoint.chain == ("emit",)
    assert result.value.embedding_endpoint.chain == ("body", "lookup")


def test_capability_declaration_without_assignment_is_not_proof(tmp_path):
    source = _source(
        assignment="",
        extra_root='_tied_weights_keys = ["output.weight"]')
    assert _reader(tmp_path, source).status == "absent"


def test_config_guarded_assignment_is_not_code_truth(tmp_path):
    source = _source(
        assignment=(
            "if config.tie_word_embeddings:\n"
            "                    "
            "self.output.weight = self.core.tokens.weight"))
    assert _reader(tmp_path, source).status == "absent"


def test_embedding_must_feed_the_repeated_stack(tmp_path):
    source = _source(
        extra_stage="self.position = nn.Embedding(16, 8)",
        stage_embedding_call="self.position(token_ids)")
    assert _reader(tmp_path, source).status == "absent"


def test_output_projection_must_be_on_the_returned_path(tmp_path):
    source = _source(
        extra_root="self.other = nn.Linear(8, 16, bias=False)",
        returned="self.other(self.core(token_ids))")
    assert _reader(tmp_path, source).status == "absent"


def test_sibling_class_assignment_cannot_vote(tmp_path):
    source = _source(assignment="")
    source += """
        class Sibling:
            def __init__(self):
                self.core = Stage()
                self.output = nn.Linear(8, 16, bias=False)
                self.output.weight = self.core.tokens.weight

            def forward(self, token_ids):
                return self.output(self.core(token_ids))
    """
    assert _reader(tmp_path, source).status == "absent"


def test_two_qualifying_assignments_are_ambiguity_not_a_pick(tmp_path):
    assignment = (
        "self.output.weight = self.core.tokens.weight\n"
        "                self.output.weight = self.core.tokens.weight")
    result = _reader(tmp_path, _source(assignment=assignment))
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_similar_non_weight_assignment_is_not_proof(tmp_path):
    source = _source(
        assignment="self.output.bias = self.core.tokens.weight")
    assert _reader(tmp_path, source).status == "absent"


def test_similarly_named_custom_primitives_are_not_framework_proof(tmp_path):
    source = _source().replace(
        "class Block:",
        "class OutputLinear:\n"
        "            pass\n"
        "\n"
        "        class Block:",
    ).replace(
        "self.output = nn.Linear(8, 16, bias=False)",
        "self.output = OutputLinear()",
    )
    assert _reader(tmp_path, source).status == "absent"


def test_rival_exact_endpoint_constructions_are_ambiguous(tmp_path):
    source = _source().replace(
        "self.output = nn.Linear(8, 16, bias=False)",
        "self.output = nn.Linear(8, 16, bias=False)\n"
        "                self.output = nn.Linear(8, 16, bias=False)")
    result = _reader(tmp_path, source)
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_real_llama_does_not_turn_capability_into_code_proof():
    config = json.loads(
        (_CORPUS / "llama-7b.json").read_text())["config"]
    context = ParseContext.build(config)
    result = manual_weight_tying_for_path(
        context.program_index(), context.source_bundle, ())
    assert result.status == "absent"


def test_evidence_closure_rejects_swapped_endpoints(tmp_path):
    value = _reader(tmp_path).value
    with pytest.raises(ValueError):
        ManualWeightTyingEvidence(
            value.component_root,
            value.stage_occurrence,
            value.assignment,
            value.embedding_endpoint,
            value.output_endpoint,
            value.output_call,
            value.embedding_edge,
            value.spans)
    with pytest.raises(ValueError):
        replace(
            value,
            assignment=replace(value.assignment, guard=("fabricated",)))


def test_endpoint_closure_rejects_a_foreign_parent(tmp_path):
    value = _reader(tmp_path).value
    with pytest.raises(ValueError):
        replace(value, output_endpoint=replace(
            value.output_endpoint,
            parent_occurrence=value.stage_occurrence))
    with pytest.raises(ValueError):
        replace(value, output_endpoint=replace(
            value.output_endpoint,
            parent_symbol=value.embedding_endpoint.parent_symbol))


@pytest.mark.parametrize('declared', [True, False])
def test_declared_tying_flag_proves_only_its_checkpoint_value(declared):
    from model_unfolder import config_to_ir
    config = json.loads((_CORPUS / 'llama-7b.json').read_text())['config']
    config['tie_word_embeddings'] = declared
    context = ParseContext.build(config)
    config_to_ir(config, parse_context=context)
    fact = context.facts.typed_records()['model.tie_word_embeddings']
    assert fact.value is declared
    assert fact.status == 'config_declared'
    assert fact.claim_kind == 'value'
    assert fact.config_paths == ('tie_word_embeddings',)
    assert fact.claim_evidence is not None
    with pytest.raises(ValueError, match='semantic kind'):
        replace(fact, claim_kind='relation')
    with pytest.raises(ValueError):
        replace(fact, value=not declared)


def test_truthy_nonboolean_tying_declaration_is_not_value_qualified():
    from model_unfolder import config_to_ir
    config = json.loads((_CORPUS / 'llama-7b.json').read_text())['config']
    config['tie_word_embeddings'] = 'false'
    context = ParseContext.build(config)
    config_to_ir(config, parse_context=context)
    fact = context.facts.typed_records()['model.tie_word_embeddings']
    # Preserve the old parser result while refusing to certify its coercion.
    assert fact.claim_evidence is None


def test_class_default_tying_flag_declares_value_without_claiming_sharing():
    from model_unfolder import config_to_ir

    config = json.loads((_CORPUS / 'llama-7b.json').read_text())['config']
    config.pop('tie_word_embeddings', None)
    context = ParseContext.build(config)
    config_to_ir(config, parse_context=context)
    fact = context.facts.typed_records()['model.tie_word_embeddings']
    assert fact.status == 'class_default'
    assert fact.claim_kind == 'value'
    assert fact.claim_readers == ('model_unfolder.adapters.transformer.parser.parse',)
    assert fact.claim_evidence is None
    assert not fact.claim_document_token
