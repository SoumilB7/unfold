"""U8-F exact per-layer side-input pathway controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from transformers import AutoConfig

from model_unfolder import config_to_ir
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.per_layer_side_input import (
    decoder_per_layer_side_input_for_path,
)
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.qualification import qualification_findings


@pytest.fixture(scope="module")
def gemma3n_case():
    document = AutoConfig.for_model("gemma3n_text").to_dict()
    context = ParseContext.build(document)
    return document, context


def _result(document, context):
    return decoder_per_layer_side_input_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)


def _mutated_result(document, context, tmp_path, old, new):
    modeling = next(
        Path(item) for item in context.source_bundle.files
        if Path(item).name == "modeling_gemma3n.py")
    source = modeling.read_text(encoding="utf-8")
    assert old in source
    changed = tmp_path / modeling.name
    changed.write_text(source.replace(old, new, 1), encoding="utf-8")

    def swap(items):
        return tuple(str(changed) if Path(item) == modeling else item
                     for item in items)

    bundle = replace(
        context.source_bundle,
        files=swap(context.source_bundle.files),
        component_files={
            key: swap(items)
            for key, items in context.source_bundle.component_files.items()
        })
    return decoder_per_layer_side_input_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)


def _renamed_result(document, context, tmp_path):
    modeling = next(
        Path(item) for item in context.source_bundle.files
        if Path(item).name == "modeling_gemma3n.py")
    source = modeling.read_text(encoding="utf-8")
    for old, new in (
        ("embed_tokens_per_layer", "side_bank"),
        ("per_layer_model_projection", "stage_side_projection"),
        ("per_layer_projection_norm", "stage_side_norm"),
        ("per_layer_input_gate", "side_gate"),
        ("post_per_layer_input_norm", "post_side_norm"),
        ("per_layer_projection", "side_projection"),
    ):
        source = source.replace(old, new)
    changed = tmp_path / modeling.name
    changed.write_text(source, encoding="utf-8")
    swap = lambda items: tuple(
        str(changed) if Path(item) == modeling else item for item in items)
    bundle = replace(
        context.source_bundle,
        files=swap(context.source_bundle.files),
        component_files={key: swap(items) for key, items
                         in context.source_bundle.component_files.items()})
    return decoder_per_layer_side_input_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)


def test_real_gemma3n_proves_exact_side_input_chain(gemma3n_case):
    document, context = gemma3n_case
    result = _result(document, context)
    assert result.status == "resolved", result.failures
    assert result.value.width_path == ("hidden_size_per_layer_input",)
    assert result.value.vocabulary_path == ("vocab_size_per_layer_input",)
    assert result.value.gate_call.lexical_order \
        < result.value.multiply_call.lexical_order \
        < result.value.projection_call.lexical_order \
        < result.value.norm_call.lexical_order


def test_parser_projects_ple_only_from_the_typed_fact(gemma3n_case):
    document, _context = gemma3n_case
    context = ParseContext.build(document)
    ir = config_to_ir(document, parse_context=context)
    fact = context.facts.typed["decoder.per_layer_embedding_pathway"]
    assert fact.status == "code_and_config"
    assert fact.value == {
        "hidden": document["hidden_size_per_layer_input"],
        "vocab": document["vocab_size_per_layer_input"],
    }
    pathway = ir.extras["external_pathways"][0]
    assert pathway["detail"] == fact.value
    assert not [item for item in qualification_findings(ir.to_dict())
                if "per_layer_embedding_pathway" in item]


def test_real_architecture_consumer_emits_exact_ple_receipt(gemma3n_case):
    from model_unfolder import unfold
    document, _context = gemma3n_case
    receipts = [
        receipt for event in unfold(document).render_events()
        for receipt in event.receipts
        if receipt.fact_key == "per_layer_embedding_pathway"
    ]
    assert receipts
    assert {(item.owner, item.mechanism, item.surface,
             item.structural_target, item.node_ids)
            for item in receipts} == {(
                "decoder", "per_layer_embedding_pathway", "html",
                "per_layer_embedding_pathway", ("ple",))}


def test_config_dimensions_cannot_create_ple_on_plain_llama():
    document = AutoConfig.for_model("llama").to_dict()
    document.update({
        "hidden_size_per_layer_input": 64,
        "vocab_size_per_layer_input": 4096,
    })
    context = ParseContext.build(document)
    result = _result(document, context)
    assert result.status != "resolved"
    ir = config_to_ir(document, parse_context=context)
    assert "per_layer_embeddings" not in ir.extras
    assert "decoder.per_layer_embedding_pathway" not in context.facts.typed


def test_removing_the_exact_multiply_breaks_the_proof(
        tmp_path, gemma3n_case):
    document, context = gemma3n_case
    result = _mutated_result(
        document, context, tmp_path,
        "first_prediction = torch.multiply(first_prediction, per_layer_input)",
        "first_prediction = torch.add(first_prediction, per_layer_input)")
    assert result.status != "resolved"


def test_passing_an_unindexed_side_tensor_breaks_the_proof(
        tmp_path, gemma3n_case):
    document, context = gemma3n_case
    result = _mutated_result(
        document, context, tmp_path,
        "per_layer_input = per_layer_inputs[:, :, i, :]",
        "per_layer_input = per_layer_inputs")
    assert result.status != "resolved"


def test_complete_operational_field_rename_does_not_change_the_answer(
        tmp_path, gemma3n_case):
    document, context = gemma3n_case
    result = _renamed_result(document, context, tmp_path)
    assert result.status == "resolved", result.failures
    assert result.value.width_path == ("hidden_size_per_layer_input",)
