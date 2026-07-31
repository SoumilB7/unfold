"""U4-F: unknown architecture has one meaning on every consumer surface."""
from __future__ import annotations

import json

import pytest

from model_unfolder.adapters.diffusor.unet import (
    _unet_stage_children,
    parse_unet,
    unet_denoiser_children,
)
from model_unfolder.adapters.transformer.assembly import (
    decoder_extras,
    decoder_layer,
)
from model_unfolder.diagram import Diagram
from model_unfolder.ir import AttentionSpec, FFNSpec, ModelIR
from model_unfolder.opgraph import attention_region, ffn_region
from model_unfolder.opgraph import Op, Region
from model_unfolder.renderers.html.op_render import _node_for, region_to_graph
from model_unfolder.renderers.html.block_views.unet import (
    build_unet_resnet_view,
)
from model_unfolder.renderers.html.block_views.mtp_head import (
    build_mtp_transformer_block_view,
)
from model_unfolder.renderers.html.metadata import _make_info
from model_unfolder.renderers.html.graph import Graph, Node
from model_unfolder.renderers.html.graph_engine import render_graph
from model_unfolder.renderers.html.tower import tower_graph
from model_unfolder.renderers.html.views import (
    _block_layout,
    _build_architecture_view,
)
from model_unfolder.expanded.grouping import _group_name


def _unknown_model() -> ModelIR:
    attention = AttentionSpec(
        kind=None,
        num_heads=8,
        num_kv_heads=2,
        head_dim=8,
        mask=None,
    )
    ffn = FFNSpec(
        kind=None,
        activation=None,
        intermediate_size=128,
        gated=None,
        projection_mode=None,
    )
    layer = decoder_layer(0, attention, ffn, 64)
    return ModelIR(
        name="unknown-fixture",
        architecture="UnknownFixture",
        vocab_size=128,
        hidden_size=64,
        max_position_embeddings=32,
        tie_word_embeddings=None,
        layers=[layer],
        extras=decoder_extras(128, 64, None),
    )


def test_one_unknown_state_survives_every_projection():
    model = _unknown_model()
    diagram = Diagram(model)
    raw = diagram.to_ir()
    expanded = diagram.to_json()
    html = diagram.to_html(standalone=True)
    info = _make_info(raw)
    layer = raw["layers"][0]

    # Canonical IR
    assert layer["attention"]["kind"] is None
    assert layer["attention"]["mask"] is None
    assert layer["attention"]["projection_mode"] is None
    assert layer["attention"]["cached"] is None
    assert layer["ffn"]["kind"] is None
    assert layer["ffn"]["gated"] is None
    assert layer["ffn"]["projection_mode"] is None
    assert layer["norm_kind"] == "unknown"
    assert layer["norm_placement"] == "unknown"
    assert layer["residual_topology"] == "unknown"

    # Canonical regions
    attention = attention_region(layer["attention"], 64)
    ffn = ffn_region(layer["ffn"], 64)
    assert attention.resolved is False
    assert [(op.kind, op.label) for op in attention.ops] == [
        ("opaque", "Attention mechanism unresolved"),
    ]
    assert ffn.resolved is False
    assert [op.kind for op in ffn.ops] == ["opaque"]

    # Expanded JSON
    group = expanded["layer_groups"][0]
    assert group["attention"]["kind"] is None
    assert group["attention"]["projections"] == {}
    assert [node["operation"] for node in
            group["attention"]["operation_graph"]["nodes"]] == ["opaque"]
    assert group["attention"]["cache"] == {
        "enabled": None,
        "status": "not_applicable",
    }
    assert group["ffn"]["structure_state"] == "mechanism_unresolved"
    assert group["ffn"]["structure_declared"] is False
    assert [node["operation"] for node in
            group["ffn"]["operation_graph"]["nodes"]] == ["opaque"]
    assert group["norm"]["kind"] == "unknown"
    assert group["norm"]["placement"] == "unknown"
    assert group["residual_topology"] == {
        "mode": "unknown",
        "residual_adds": [],
    }
    assert expanded["io"]["final_stage"]["status"] == "unresolved"
    assert "final_norm" not in expanded["io"]

    # Parameter assumptions are structural qualifications.  They must survive
    # the expanded machine schema just as they survive raw IR and HTML.
    expanded_assumptions = expanded["parameters"].get("assumptions") or []
    assert any("attention mechanism unknown" in note
               for note in expanded_assumptions)
    assert any("FFN structure unknown" in note
               for note in expanded_assumptions)
    assert any("layer normalization parameters unresolved" in note
               for note in expanded_assumptions)
    assert any("final-stage normalization unresolved" in note
               for note in expanded_assumptions)

    # HTML/cards/metadata
    assert "Attention mechanism unresolved" in html
    assert "Feed-forward mechanism unresolved" in html
    assert "Wiring unresolved" in html
    assert "Pre-head path unresolved" in html
    for fabricated_id in (
        "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj",
    ):
        assert fabricated_id not in info["blocks"]
        assert f'data-card-id="{fabricated_id}"' not in html

    # Parameters: a deterministic estimate may remain during U4, but it must
    # expose the convention rather than presenting Q/K/V/O as code-proven.
    assumptions = diagram.param_count().get("assumptions") or []
    assert any("attention mechanism unknown" in note for note in assumptions)
    assert any("FFN structure unknown" in note for note in assumptions)
    assert any("layer normalization parameters unresolved" in note
               for note in assumptions)
    assert any("final-stage normalization unresolved" in note
               for note in assumptions)


def test_unet_missing_activation_never_becomes_silu():
    unet = parse_unet({
        "block_out_channels": [32, 64],
        "down_block_types": ["DownBlock2D", "DownBlock2D"],
        "up_block_types": ["UpBlock2D", "UpBlock2D"],
        "layers_per_block": 1,
        "in_channels": 4,
        "out_channels": 4,
    })
    assert unet["act_fn"] is None
    cards = unet_denoiser_children(unet)
    payload = json.dumps(cards)
    assert "SiLU" not in payload
    assert "GroupNorm+Activation" in payload
    assert "no owner-bound act_fn evidence" in payload
    resnet = _unet_stage_children(
        unet["down"][0],
        "down",
        unet.get("cross_attention_dim"),
        act_label="Activation",
        act_note="unresolved because no owner-bound act_fn evidence is available",
    )[0]
    html = build_unet_resnet_view(
        {"name": "unknown-unet"},
        {"blocks": {}},
        "unknown-unet",
        resnet,
    )
    assert "GroupNorm + Activation" in html
    assert "SiLU" not in html


def test_unet_declared_activation_survives_the_same_path():
    unet = parse_unet({
        "block_out_channels": [32],
        "down_block_types": ["DownBlock2D"],
        "up_block_types": ["UpBlock2D"],
        "layers_per_block": 1,
        "in_channels": 4,
        "out_channels": 4,
        "act_fn": "silu",
    })
    assert unet["act_fn"] == "silu"
    payload = json.dumps(unet_denoiser_children(unet))
    assert "GroupNorm+SiLU" in payload
    assert "resolved through the denoiser's act_fn input" in payload


@pytest.mark.parametrize("kind", [None, "", "novel_block"])
def test_tower_renderer_never_turns_unknown_block_kind_into_norm(kind):
    graph = tower_graph({
        "cell": [{
            "id": "unresolved_op",
            "kind": kind,
            "label": "Mechanism unresolved",
        }],
        "repeat": 1,
    })
    assert [(node.id, node.kind) for node in graph.nodes] == [
        ("unresolved_op", "opaque"),
    ]


@pytest.mark.parametrize("kind", [None, "", "novel_block"])
def test_architecture_layout_uses_neutral_geometry_for_unknown_kind(kind):
    layout, _, _, _ = _block_layout({
        "id": "unresolved_op",
        "kind": kind,
        "label": "Mechanism unresolved",
    })
    assert layout == {
        "shape": "rect",
        "w": 200,
        "h": 44,
        "font": 15,
    }


def test_mtp_renderer_does_not_reconstruct_an_empty_decoder_cell():
    html = build_mtp_transformer_block_view(
        {"name": "unknown-mtp"},
        {"blocks": {}},
        "uf-u4f",
        {"children": []},
    )
    for fabricated_id in (
        "mtp_block_norm1",
        "mtp_block_attn",
        "mtp_block_add1",
        "mtp_block_norm2",
        "mtp_block_ffn",
        "mtp_block_add2",
    ):
        assert fabricated_id not in html


def test_mtp_renderer_preserves_declared_unknown_child_as_opaque():
    html = build_mtp_transformer_block_view(
        {"name": "unknown-mtp"},
        {"blocks": {}},
        "uf-u4f",
        {
            "children": [{
                "id": "declared_unknown",
                "kind": None,
                "label": "Mechanism unresolved",
                "resolved": False,
            }],
        },
    )
    assert "declared_unknown" in html
    assert "Mechanism unresolved" in html
    assert "mtp_block_norm1" not in html


def test_unknown_elementwise_operation_is_not_recast_as_activation_or_matmul():
    region = Region(
        "unknown",
        "custom",
        "Unknown operation",
        [Op("mystery", "elementwise", "Custom elementwise", fn=None)],
        [],
        resolved=False,
    )
    node = _node_for(region.ops[0], region, clickable=False, primary="hidden")
    assert node.kind == "opaque"
    assert node.label == "Custom elementwise"
    assert node.resolved is False


def test_formula_renderer_does_not_install_qk_or_sqrt_defaults():
    html = render_graph(
        Graph(nodes=[Node("unknown_formula", "formula", meta={})],
              flow=["unknown_formula"]),
        {"blocks": {}},
        "uf-u4f",
        "unknown-formula",
        "Unknown formula",
    )
    assert "Scores" in html
    assert "Q K^T" not in html
    assert "sqrt(dim)" not in html


def test_two_inputs_into_one_opaque_mechanism_render_that_mechanism_once():
    region = attention_region(
        {
            "kind": None,
            "cross_attention": True,
            "cross_kv_source": "encoded text prompt",
        },
        128,
    )
    graph = region_to_graph(region)
    assert graph.flow == ["hidden", "block", "region_out"]
    assert sum(node.id == "block" for node in graph.nodes) == 1
    assert all("block" not in lane.ids
               for parallel in graph.parallels for lane in parallel.lanes)
    assert graph.parallels == []
    assert [
        (side.node, side.target, side.side)
        for side in graph.side_inputs
    ] == [
        ("cross_attention_states", "block", "right"),
    ]


def test_heterogeneous_group_name_calls_unknown_unresolved_not_default():
    unknown_mask = {
        "representative": {
            "attention": {"mask": None},
            "ffn": {"kind": "dense"},
        },
    }
    causal = {
        "representative": {
            "attention": {"mask": "causal"},
            "ffn": {"kind": "dense"},
        },
    }
    assert _group_name(unknown_mask, [unknown_mask, causal]) == "unresolved"

    unknown_ffn = {
        "representative": {
            "attention": {"mask": "causal"},
            "ffn": {"kind": None},
        },
    }
    moe = {
        "representative": {
            "attention": {"mask": "causal"},
            "ffn": {"kind": "moe"},
        },
    }
    assert _group_name(unknown_ffn, [unknown_ffn, moe]) == "unresolved"


def test_missing_final_stage_card_never_becomes_final_rmsnorm():
    raw = _unknown_model().to_dict()
    render = raw["extras"]["render"]
    render["model_blocks"] = [
        block
        for block in render.get("model_blocks") or []
        if block.get("id") != "final_rms"
    ]
    info = _make_info(raw)
    html = _build_architecture_view(raw, info, "uf-u4f")
    assert "Final RMSNorm" not in html
    assert "Pre-head path" in html


def test_missing_model_bookend_cards_render_only_neutral_placeholders():
    raw = _unknown_model().to_dict()
    raw["extras"]["render"]["model_blocks"] = []
    info = _make_info(raw)
    html = _build_architecture_view(raw, info, "uf-u4f")
    for fabricated in (
        "Tokenized text",
        "Token Embedding layer",
        "Final RMSNorm",
        "Linear output layer",
    ):
        assert fabricated not in html
    for neutral in (
        "Model input",
        "Embedding stage unresolved",
        "Pre-head path",
        "Output stage unresolved",
    ):
        assert neutral in html


@pytest.mark.parametrize(
    "cell_kind",
    ["transformer2d", "simple_cross", "code_defined", None],
)
def test_unet_cell_kind_never_fabricates_inner_mha(cell_kind):
    """A container-cell classification is not an attention-mechanism proof."""
    stage = {
        "id": f"stage_{cell_kind or 'unknown'}",
        "channels": 64,
        "resnets": 1,
        "attn": True,
        "transformers": 1,
        "num_heads": 8,
        "head_dim": 8,
        "attn_kind": cell_kind,
        "has_self": True,
        "has_cross": True,
    }
    cards = _unet_stage_children(
        stage,
        "down",
        128,
        act_label="Activation",
        act_note="unresolved",
    )
    payload = json.dumps(cards)
    assert '"kind": "mha"' not in payload
    assert '"num_kv_heads": 8' not in payload
    assert "q_proj" not in payload
    assert "k_proj" not in payload
    assert "v_proj" not in payload
