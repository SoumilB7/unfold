"""U4-D: normalization, residual wiring, and bookends are independent facts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import replace

import pytest

from model_unfolder.adapters.diffusor.parser import parse as parse_diffusion
from model_unfolder.adapters.transformer.assembly import (
    decoder_layer,
    parallel_decoder_layer,
    single_stream_decoder_layer,
)
from model_unfolder.adapters.transformer.blocks.attention import attention_detail
from model_unfolder.adapters.transformer.blocks.feed_forward import ffn_detail
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.expanded import build_expanded
from model_unfolder.ir import (
    AttentionSpec,
    FFNSpec,
    LayerSpec,
    ModelIR,
    canonical_norm_kind,
)
from model_unfolder.params import estimate_params
from model_unfolder.parser import config_to_ir
from model_unfolder.renderers.html.tower import tower_cell
from model_unfolder.sable import DEFAULT_CORPUS
from model_unfolder.submodel import submodel_cell_blocks


def _attention() -> AttentionSpec:
    return AttentionSpec(
        kind="mha", num_heads=4, num_kv_heads=4, head_dim=16,
        mask="causal", rope=False,
    )


def _ffn() -> FFNSpec:
    return FFNSpec(
        kind="dense", activation="gelu", intermediate_size=128,
        gated=False, projection_mode="dense",
    )


def _model(
        layer: LayerSpec, *, embedding_norm_kind=None, final_norm_kind=None
) -> ModelIR:
    return ModelIR(
        name="fixture",
        architecture="Fixture",
        vocab_size=100,
        hidden_size=64,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        layers=[layer],
        embedding_norm_kind=embedding_norm_kind,
        final_norm_kind=final_norm_kind,
    )


def _ids(layer: LayerSpec) -> list[str]:
    return [block["id"] for block in layer.blocks]


def _corpus_config(slug: str) -> dict:
    payload = json.loads((Path(DEFAULT_CORPUS) / f"{slug}.json").read_text())
    return payload["config"]


def test_layer_topology_fields_are_closed_and_unknown_by_default():
    layer = LayerSpec(index=0, attention=_attention(), ffn=_ffn())
    assert (
        layer.norm_kind,
        layer.norm_placement,
        layer.residual_topology,
        layer.parallel_norm_count,
    ) == ("unknown", "unknown", "unknown", None)

    with pytest.raises(ValueError, match="norm kind"):
        LayerSpec(
            index=0, attention=_attention(), ffn=_ffn(),
            norm_kind="RMSNorm",
        )
    with pytest.raises(ValueError, match="norm placement"):
        LayerSpec(
            index=0, attention=_attention(), ffn=_ffn(),
            norm_placement="conventional_pre",
        )
    with pytest.raises(ValueError, match="residual topology"):
        LayerSpec(
            index=0, attention=_attention(), ffn=_ffn(),
            residual_topology="decoder",
        )
    with pytest.raises(ValueError, match="requires"):
        LayerSpec(
            index=0, attention=_attention(), ffn=_ffn(),
            residual_topology="sequential", parallel_norm_count=1,
        )
    with pytest.raises(ValueError, match="model-stage norm kind"):
        ModelIR(
            name="fixture", architecture="Fixture", vocab_size=100,
            hidden_size=64, max_position_embeddings=128,
            tie_word_embeddings=True, layers=[],
            final_norm_kind="borrowed_from_layer",
        )


def test_norm_vocabulary_normalization_never_classifies_a_class_name():
    assert canonical_norm_kind("LayerNorm") == "layernorm"
    assert canonical_norm_kind("rms_norm") == "rmsnorm"
    assert canonical_norm_kind("LlamaRMSNorm") is None
    assert canonical_norm_kind("SomeLayerNorm") is None


def test_norm_kind_or_placement_alone_cannot_draw_cell_wiring():
    known_kind = decoder_layer(
        0, _attention(), _ffn(), 64, norm_kind="rmsnorm",
    )
    known_placement = decoder_layer(
        0, _attention(), _ffn(), 64,
        norm_kind="rmsnorm", norm_placement="pre",
    )
    for layer in (known_kind, known_placement):
        assert _ids(layer) == ["attn", "wiring_unresolved", "ffn"]
        assert not any(
            block["kind"] == "residual_add" for block in layer.blocks
        )

    proven = decoder_layer(
        0, _attention(), _ffn(), 64,
        norm_kind="rmsnorm",
        norm_placement="pre",
        residual_topology="sequential",
    )
    assert _ids(proven) == ["rms1", "attn", "add1", "rms2", "ffn", "add2"]


def test_shared_tower_projector_has_no_pre_norm_default():
    blocks = tower_cell(
        "tower", attn_label="Attention", norm_label="LayerNorm",
    )
    assert [block["id"] for block in blocks] == [
        "tower_op_selfattn", "tower_op_wiring_unknown", "tower_op_ffn",
    ]
    assert blocks[1]["label"] == ["LayerNorm", "wiring unresolved"]
    assert blocks[1]["resolved"] is False and blocks[1]["static"] is True
    assert not any(block["kind"] == "residual_add" for block in blocks)


def test_unknown_norm_kind_and_placement_say_wiring_unresolved():
    main = decoder_layer(0, _attention(), _ffn(), 64)
    assert main.blocks[1]["label"] == "Wiring unresolved"

    shared = tower_cell(
        "tower", attn_label="Attention", norm_label="Norm",
    )
    assert shared[1]["label"] == "Wiring unresolved"


def test_submodel_card_projector_has_no_missing_placement_default():
    spec = {
        "component": "root.text_encoder",
        "groups": [{
            "attention": attention_detail(_attention()),
            "ffn": ffn_detail(_ffn()),
            # Intentionally no norm_placement: missing is unknown, never pre.
        }],
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unknown wiring must not request norm/residual cards")

    cards = submodel_cell_blocks(
        spec,
        "encoder",
        attn_description="attention",
        norm_fallback="LayerNorm",
        norm_card=forbidden,
        residual_card=forbidden,
    )
    assert [card["id"] for card in cards] == [
        "encoder_op_selfattn", "encoder_op_ffn",
    ]


def test_parallel_topology_never_invents_one_shared_norm():
    unresolved = parallel_decoder_layer(
        0, _attention(), _ffn(), 64,
        norm_kind="layernorm", norm_placement="unknown", norm_count=None,
    )
    assert unresolved.residual_topology == "parallel"
    assert unresolved.norm_placement == "unknown"
    assert unresolved.parallel_norm_count is None
    norm = unresolved.blocks[0]
    assert norm["label"] == ["Norm inputs", "unresolved"]
    assert "share one" in norm["description"]
    assert "shared)" not in norm["title"].lower()

    shared = parallel_decoder_layer(
        0, _attention(), _ffn(), 64,
        norm_kind="layernorm", norm_placement="pre", norm_count=1,
    )
    assert shared.norm_placement == "pre"
    assert shared.parallel_norm_count == 1
    assert "shared" in shared.blocks[0]["title"].lower()


def test_parallel_residual_scale_is_not_dropped_between_fact_and_view():
    layer = parallel_decoder_layer(
        0, _attention(), _ffn(), 64,
        norm_kind="layernorm", norm_placement="pre", norm_count=1,
        residual_scale=0.25,
    )
    assert layer.residual_scale == 0.25
    assert _ids(layer) == [
        "rms1", "attn", "parallel_sum", "res_scale1", "add1", "ffn",
    ]
    branch_sum = layer.blocks[2]
    scale = layer.blocks[3]
    add = layer.blocks[4]
    side_ffn = layer.blocks[5]
    assert branch_sum["kind"] == "residual_add"
    assert branch_sum["static"] is True
    assert scale["kind"] == "gate_mul" and scale["sub"] == "× 0.25"
    assert add["residual_from"] == "rms1"
    assert side_ffn["feeds"] == "parallel_sum"
    assert "scaled(attention output + FFN output)" in add["description"]


def test_layer_spec_rejects_a_non_finite_residual_scale():
    with pytest.raises(TypeError, match="residual_scale must be numeric"):
        LayerSpec(
            index=0,
            attention=_attention(),
            ffn=_ffn(),
            residual_topology="sequential",
            residual_scale=math.nan,
        )


def test_self_ffn_scale_cannot_leak_into_additive_cross_attention():
    layer = decoder_layer(
        0, _attention(), _ffn(), 64,
        norm_kind="layernorm", norm_placement="pre",
        residual_topology="sequential", residual_scale=0.25,
        cross_attention_spec=_attention(),
    )
    assert _ids(layer) == [
        "rms1", "attn", "res_scale1", "add1",
        "rms_cross", "cross_attn", "add_cross",
        "rms2", "ffn", "res_scale2", "add2",
    ]
    assert sum(
        block.get("kind") == "gate_mul" for block in layer.blocks
    ) == 2
    cross_add = next(
        block for block in layer.blocks if block["id"] == "add_cross")
    assert cross_add["residual_from"] == "rms_cross"


def test_topology_and_parallel_norm_count_cannot_collapse_layer_groups():
    base = LayerSpec(index=0, attention=_attention(), ffn=_ffn())
    parallel_unknown = replace(base, residual_topology="parallel")
    parallel_one = replace(
        base, residual_topology="parallel", parallel_norm_count=1,
    )
    assert base.signature() != parallel_unknown.signature()
    assert parallel_unknown.signature() != parallel_one.signature()


def test_fused_topology_keeps_its_proven_relation_without_inventing_a_norm():
    layer = single_stream_decoder_layer(
        0, _attention(), _ffn(), 64,
        norm_kind="layernorm", norm_placement="unknown",
    )
    assert layer.residual_topology == "fused_parallel"
    assert layer.norm_placement == "unknown"
    assert "ss_input" in _ids(layer)
    assert "rms1" not in _ids(layer)
    assert {"ss_concat", "ss_proj", "gate_single", "ss_add"} <= set(_ids(layer))


def test_unknown_single_stream_fusion_does_not_default_to_flux_wiring():
    cfg = {
        "_class_name": "SyntheticTransformer2DModel",
        "num_layers": 0,
        "num_single_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 4,
        "joint_attention_dim": 64,
        "in_channels": 16,
        "patch_size": 1,
    }
    context = ParseContext(
        source_bundle=SourceBundle(source="local", files=()),
        source="local",
    )
    layer = parse_diffusion(cfg, context=context).layers[0]
    assert layer.residual_topology == "unknown"
    assert _ids(layer) == ["attn", "wiring_unresolved", "ffn"]
    assert not {"ss_concat", "ss_proj", "gate_single"} & set(_ids(layer))


def test_final_norm_is_not_borrowed_from_the_repeated_layer():
    layer = decoder_layer(
        0, _attention(), _ffn(), 64,
        norm_kind="rmsnorm",
        norm_placement="pre",
        residual_topology="sequential",
    )
    unknown = _model(layer)
    exact = _model(layer, final_norm_kind="rmsnorm")

    unknown_expanded = build_expanded(unknown)
    exact_expanded = build_expanded(exact)
    assert unknown_expanded["io"]["final_stage"]["status"] == "unresolved"
    assert "final_norm" not in unknown_expanded["io"]
    assert exact_expanded["io"]["final_norm"]["kind"] == "rmsnorm"

    unknown_params = estimate_params(unknown)
    exact_params = estimate_params(exact)
    assert exact_params["total"] - unknown_params["total"] == 64
    assert any(
        "final-stage normalization unresolved" in note
        for note in unknown_params["assumptions"]
    )


def test_entry_norm_parameters_are_counted_only_when_the_bookend_is_proven():
    layer = decoder_layer(
        0, _attention(), _ffn(), 64,
        norm_kind="rmsnorm",
        norm_placement="pre",
        residual_topology="sequential",
    )
    unknown = _model(layer)
    exact = _model(layer, embedding_norm_kind="layernorm")

    unknown_params = estimate_params(unknown)
    exact_params = estimate_params(exact)
    assert exact_params["total"] - unknown_params["total"] == 128
    assert any(
        "embedding-stage normalization not proven" in note
        for note in unknown_params["assumptions"]
    )


def test_entry_norm_is_separate_code_proven_bookend():
    cfg = _corpus_config("bloom")
    context = ParseContext.build(cfg, source="local")
    ir = config_to_ir(cfg, parse_context=context)

    assert ir.embedding_norm_kind == "layernorm"
    assert ir.final_norm_kind == "layernorm"
    entry_fact = context.facts.records["model.embedding_norm_kind"]
    final_fact = context.facts.records["model.final_norm_kind"]
    assert entry_fact.status == final_fact.status == "code_proven"
    assert entry_fact.value == final_fact.value == "layernorm"
    assert entry_fact.source == "embedding_stage_norm_evidence"
    assert final_fact.source == "final_stage_norm_evidence"
    embed_norm = next(
        block for block in ir.extras["render"]["model_blocks"]
        if block["id"] == "embed_norm"
    )
    assert "BLOOM" not in embed_norm["description"]


@pytest.mark.parametrize(
    ("slug", "placement", "topology", "expected_ids"),
    (
        (
            "llama-7b", "pre", "sequential",
            ["rms1", "attn", "add1", "rms2", "ffn", "add2"],
        ),
        (
            "gemma-2-2b-it", "double", "sequential",
            [
                "rms1", "attn", "post_attn_ln", "add1",
                "rms2", "ffn", "post_ffn_ln", "add2",
            ],
        ),
        (
            "olmo-2-1124-7b", "post", "sequential",
            ["attn", "post_attn_ln", "add1", "ffn", "post_ffn_ln", "add2"],
        ),
    ),
)
def test_exact_source_controls_keep_their_detailed_cells(
        slug, placement, topology, expected_ids):
    cfg = _corpus_config(slug)
    ir = config_to_ir(
        cfg, parse_context=ParseContext.build(cfg, source="local")
    )
    layer = ir.layers[0]
    assert layer.norm_placement == placement
    assert layer.residual_topology == topology
    assert _ids(layer) == expected_ids


def test_musicgen_does_not_borrow_a_pre_norm_convention():
    cfg = _corpus_config("musicgen-small")
    ir = config_to_ir(
        cfg, parse_context=ParseContext.build(cfg, source="local")
    )
    assert ir.layers
    layer = ir.layers[0]
    assert layer.norm_kind == "layernorm"
    assert layer.norm_placement == "unknown"
    assert layer.residual_topology == "unknown"
    assert "wiring_unresolved" in _ids(layer)
    assert not {"rms1", "rms2", "add1", "add2"} & set(_ids(layer))


def test_stablelm_selector_is_consumed_only_for_the_fact_it_decides():
    cfg = _corpus_config("stablelm-2-1-6b")
    ir = config_to_ir(
        cfg, parse_context=ParseContext.build(cfg, source="local")
    )
    assert ir.layers[0].norm_placement == "pre"
    assert ir.layers[0].residual_topology == "sequential"
    facts = ir.extras["fact_provenance"]
    assert facts["decoder.layer.norm_placement"]["status"] == "code_proven"
    assert facts["decoder.layer.residual_topology"]["status"] \
        == "code_and_config"
    assert not any(
        item.endswith(":use_parallel_residual")
        for item in ir.extras["config_access"]["accessed_unconsumed"]
    )
    obligations = [
        item for item in ir.extras["config_access"]["projection_obligations"]
        if item["mechanism"] == "cell_topology"
    ]
    assert [(item["source"]["path"], item["target"]["key"])
            for item in obligations] \
        == [("use_parallel_residual", "residual_topology")]


def test_qwen35_declared_variant_domain_is_cited_by_both_cell_facts():
    cfg = _corpus_config("qwen3-5-27b-text")
    ir = config_to_ir(
        cfg, parse_context=ParseContext.build(cfg, source="local")
    )
    assert (ir.layers[0].norm_placement,
            ir.layers[0].residual_topology) == ("pre", "sequential")
    facts = ir.extras["fact_provenance"]
    assert facts["decoder.layer.norm_placement"]["status"] \
        == "code_and_config"
    assert facts["decoder.layer.residual_topology"]["status"] \
        == "code_and_config"
    obligations = [
        item for item in ir.extras["config_access"]["projection_obligations"]
        if item["mechanism"] == "cell_topology"
    ]
    assert sorted(
        (item["source"]["path"], item["target"]["key"])
        for item in obligations) == [
            ("layer_types", "norm_placement"),
            ("layer_types", "residual_topology"),
        ]
