"""U4-C poisons: FFN unknowns and layer grouping stay identical everywhere."""
from __future__ import annotations

from dataclasses import fields, replace
import math

import pytest

from model_unfolder.expanded.ffn import build_ffn
from model_unfolder.expanded.grouping import signature as expanded_signature
from model_unfolder.adapters.transformer.special_parts.modalities.schema import (
    tower_submodel_spec,
)
from model_unfolder.ir import (
    AttentionSpec,
    FFNSpec,
    LayerSpec,
    ModelIR,
    layer_signature,
)
from model_unfolder.labels import cards_from_region, ffn_label, ffn_short, ffn_title
from model_unfolder.opgraph import (
    ffn_region,
    ffn_structure_declared,
    ffn_structure_state,
)
from model_unfolder.renderers.html.metadata import _signature as html_signature


def _attention() -> AttentionSpec:
    return AttentionSpec(
        kind="mha",
        num_heads=8,
        num_kv_heads=8,
        head_dim=16,
        mask="causal",
        qk_norm=False,
        rope=False,
        position_kind="none",
        position_application="none",
        bias=False,
        cached=False,
        scores_scaled=False,
        projection_mode="split_qkv",
    )


def _ffn() -> FFNSpec:
    return FFNSpec(
        kind="dense",
        activation="gelu",
        intermediate_size=256,
        gated=False,
        bias=False,
        projection_mode="dense",
    )


def _layer(**values) -> LayerSpec:
    return LayerSpec(
        index=0,
        attention=values.pop("attention", _attention()),
        ffn=values.pop("ffn", _ffn()),
        norm_kind=values.pop("norm_kind", "layernorm"),
        norm_placement=values.pop("norm_placement", "pre"),
        **values,
    )


def _as_dict(layer: LayerSpec) -> dict:
    return ModelIR(
        name="fixture",
        architecture="Fixture",
        vocab_size=10,
        hidden_size=128,
        max_position_embeddings=16,
        tie_word_embeddings=False,
        layers=[layer],
    ).to_dict()["layers"][0]


def _all_signatures(layer: LayerSpec) -> tuple:
    projected = _as_dict(layer)
    return (
        layer.signature(),
        layer_signature(projected),
        expanded_signature(projected),
        html_signature(projected),
    )


def test_all_layer_grouping_consumers_use_the_same_signature():
    signatures = _all_signatures(_layer())
    assert signatures.count(signatures[0]) == len(signatures)


_ATTENTION_MUTATIONS = {
    "kind": "gqa",
    "mixer_state": "ordinary_attention",
    "num_heads": 16,
    "num_kv_heads": 2,
    "head_dim": 32,
    "kv_lora_rank": 4,
    "q_lora_rank": 5,
    "rope_dim": 8,
    "rope_theta": 500000.0,
    "rope_initialization": {"protocol": "inverse_frequency"},
    "qk_nope_head_dim": 12,
    "qk_rope_head_dim": 4,
    "v_head_dim": 14,
    "mask": "sliding",
    "window_size": 32,
    "kv_source_layer": 0,
    "qk_norm": True,
    "sinks": True,
    "logit_softcap": 10.0,
    "qkv_clip": 8.0,
    "rope": True,
    "position_kind": "rope",
    "position_application": "qk_rotation",
    "bias": True,
    "shared": True,
    "no_rope": True,
    "rope_3d": True,
    "cached": True,
    "output_projection": True,
    "cross_attention": True,
    "cross_kv_source": "encoded states",
    "cross_kv_source_kind": "conditioning_encoder",
    "compress_ratio": 2,
    "index_topk": 4,
    "index_n_heads": 2,
    "index_head_dim": 8,
    "mrope_section": [2, 3, 3],
    "conv_kernel_size": 3,
    "output_gate": "sigmoid",
    "scores_scale": 0.25,
    "scores_scaled": True,
    "projection_mode": "fused_qkv",
    "variant": {"tag": "variant-b"},
}


def test_every_attention_architecture_field_changes_every_grouping_consumer():
    base = _layer()
    baseline = base.signature()
    structural = {item.name for item in fields(AttentionSpec)} - {
        "asserted", "cross_kv_source_evidence",
    }
    assert structural == set(_ATTENTION_MUTATIONS)
    for name, value in _ATTENTION_MUTATIONS.items():
        updates = {name: value}
        if name == "cross_kv_source_kind":
            updates.update(
                cross_attention=True,
                cross_kv_source_evidence={
                    "status": "proven", "kind": "cross_attention",
                    "owner_class": "Wrapper",
                    "source_file": "/tmp/modeling.py", "line": 41,
                    "routes": [{"modality": "conditioning"}],
                },
            )
        candidate = replace(base, attention=replace(base.attention, **updates))
        signatures = _all_signatures(candidate)
        assert signatures.count(signatures[0]) == len(signatures), name
        assert signatures[0] != baseline, name
    # asserted is provenance/debt, not architecture.
    assert replace(
        base, attention=replace(base.attention, asserted=("legacy",))
    ).signature() == baseline
    # Exact evidence locations are provenance, not architecture: they must not
    # split otherwise identical repeated layers into separate groups.
    typed = replace(
        base.attention,
        cross_attention=True,
        cross_kv_source_kind="conditioning_encoder",
        cross_kv_source_evidence={
            "status": "proven", "kind": "cross_attention",
            "owner_class": "Wrapper", "source_file": "/tmp/a.py", "line": 1,
            "routes": [{"modality": "conditioning"}],
        },
    )
    relocated = replace(
        typed,
        cross_kv_source_evidence={
            "status": "proven", "kind": "cross_attention",
            "owner_class": "Wrapper", "source_file": "/tmp/b.py", "line": 9,
            "routes": [{"modality": "conditioning"}],
        },
    )
    assert replace(base, attention=typed).signature() \
        == replace(base, attention=relocated).signature()


_FFN_MUTATIONS = {
    "kind": "moe",
    "activation": "silu",
    "intermediate_size": 512,
    "gated": True,
    "activation_assumed": True,
    "activation_from_class": True,
    "bias": True,
    "projection_mode": "split",
    "expert_projection_mode": "fused_gate_up",
    "expert_activation_formula": {"kind": "silu"},
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_shared_experts": 1,
    "expert_intermediate_size": 64,
    "routing": {"top_k": 2},
}


def test_every_ffn_architecture_field_changes_every_grouping_consumer():
    base = _layer()
    baseline = base.signature()
    structural = {item.name for item in fields(FFNSpec)} - {"asserted"}
    assert structural == set(_FFN_MUTATIONS)
    for name, value in _FFN_MUTATIONS.items():
        updates = {name: value}
        if name == "expert_activation_formula":
            updates.update(
                kind="moe", expert_projection_mode="fused_gate_up")
        candidate = replace(base, ffn=replace(base.ffn, **updates))
        signatures = _all_signatures(candidate)
        assert signatures.count(signatures[0]) == len(signatures), name
        assert signatures[0] != baseline, name
    assert replace(
        base, ffn=replace(base.ffn, asserted=("legacy",))
    ).signature() == baseline


def test_expert_formula_cannot_be_attached_to_a_dense_ffn():
    with pytest.raises(ValueError, match="requires kind='moe'"):
        FFNSpec(
            kind="dense",
            expert_projection_mode="fused_gate_up",
            expert_activation_formula={"kind": "silu"},
        )


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_expert_formula_rejects_non_finite_operands(value):
    with pytest.raises(TypeError, match="is numeric"):
        FFNSpec(
            kind="moe",
            expert_projection_mode="fused_gate_up",
            expert_activation_formula={"kind": "silu", "alpha": value},
        )


def test_cross_attention_and_cell_topology_cannot_collapse():
    base = _layer(blocks=[{
        "id": "attn", "role": "attention", "kind": "attention",
        "title": "presentation A", "description": "presentation only",
    }])
    renamed_prose = _layer(blocks=[{
        "id": "attn", "role": "attention", "kind": "attention",
        "title": "presentation B", "description": "different prose",
    }])
    changed_topology = _layer(blocks=[{
        "id": "attn", "role": "attention", "kind": "attention",
        "branch_side": "left",
    }])
    additive_cross = replace(base, cross_attention=replace(_attention(), cross_attention=True))
    assert base.signature() == renamed_prose.signature()
    assert base.signature() != changed_topology.signature()
    assert base.signature() != additive_cross.signature()


def test_ffn_state_is_one_closed_cross_surface_vocabulary():
    cases = [
        ({}, "mechanism_unresolved", False),
        ({"kind": "novel"}, "unsupported", False),
        ({"kind": "dense"}, "gating_unresolved", False),
        ({"kind": "dense", "gated": True}, "storage_unresolved", False),
        ({"kind": "dense", "gated": False, "projection_mode": "split"},
         "storage_unresolved", False),
        ({"kind": "dense", "gated": False, "projection_mode": "dense"},
         "dense", True),
        ({"kind": "dense", "gated": True, "projection_mode": "split"},
         "gated", True),
        ({"kind": "conv_glu"}, "conv_glu", True),
        ({"kind": "moe", "expert_projection_mode": None}, "moe", False),
        ({"kind": "moe", "expert_projection_mode": "split"}, "moe", True),
    ]
    for fact, state, declared in cases:
        assert ffn_structure_state(fact) == state
        assert ffn_structure_declared(fact) is declared
        expanded = build_ffn(fact, 64, "layers[0]", None)
        assert expanded["structure_state"] == state
        assert expanded["structure_declared"] is declared


def test_every_opaque_ffn_names_why_it_is_opaque_on_graph_json_and_cards():
    cases = [
        ({}, "mechanism_unresolved", "mechanism unresolved"),
        ({"kind": "dense"}, "gating_unresolved", "gating unresolved"),
        (
            {"kind": "dense", "gated": True, "projection_mode": None},
            "storage_unresolved",
            "storage unresolved",
        ),
        ({"kind": "novel"}, "unsupported", "unsupported"),
    ]
    for fact, status, wording in cases:
        region = ffn_region(fact, 64)
        assert region.resolved is False
        assert region.ops[0].meta["status"] == status
        assert wording in " ".join(
            region.ops[0].label
            if isinstance(region.ops[0].label, list)
            else [str(region.ops[0].label)]
        ).lower()
        assert wording in cards_from_region(region)[0]["title"].lower()
        assert wording.split("_")[0] in str(ffn_label(fact)).lower()
        assert ffn_title(fact)
        assert ffn_short(fact)
        expanded = build_ffn(fact, 64, "layers[0]", None)
        assert expanded["operation_graph"]["nodes"][0]["status"] == status


def test_moe_expert_unknown_storage_is_explicit_not_borrowed():
    fact = {
        "kind": "moe",
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "expert_projection_mode": None,
    }
    expert = next(op for op in ffn_region(fact, 64).ops if op.id == "expert")
    assert expert.meta["status"] == "storage_unresolved"
    assert "storage unresolved" in " ".join(expert.label).lower()


def test_tower_softmax_and_unknown_storage_do_not_default_to_mha_or_split_ffn():
    spec = tower_submodel_spec(
        {
            "num_attention_heads": 8,
            "hidden_size": 128,
            "intermediate_size": 256,
            "activation": None,
            "num_layers": 1,
        },
        [{
            "repeat": 1,
            "attention_kind": "softmax",
            "projection_mode": "unknown",
            "ffn_gated": True,
            "ffn_projection_mode": "unknown",
            "norm_kind": "layernorm",
            "norm_placement": "pre",
        }],
        component="root.vision",
    )
    group = spec["groups"][0]
    assert group["attention"]["kind"] is None
    assert group["attention"]["projection_mode"] is None
    assert group["ffn"]["kind"] is None
    assert group["ffn"]["gated"] is None
    assert group["ffn"]["projection_mode"] is None


def test_tower_preserves_an_exact_supported_attention_and_ffn_kind():
    spec = tower_submodel_spec(
        {
            "num_attention_heads": 8,
            "hidden_size": 128,
            "intermediate_size": 256,
            "activation": "gelu",
            "num_layers": 1,
        },
        [{
            "repeat": 1,
            "attention_kind": "linear",
            "projection_mode": "separate_qkv",
            "ffn_gated": False,
            "ffn_projection_mode": "dense",
            "norm_kind": "layernorm",
            "norm_placement": "pre",
        }],
        component="root.vision",
    )
    group = spec["groups"][0]
    assert group["attention"]["kind"] == "linear"
    assert group["attention"]["projection_mode"] == "split_qkv"
    assert group["ffn"]["kind"] == "dense"
    assert group["ffn"]["gated"] is False
    assert group["ffn"]["projection_mode"] == "dense"


def test_top_level_unknown_safe_ffn_label_grows_its_architecture_box():
    """The legacy architecture renderer must match the graph engine's fit law."""
    from model_unfolder.renderers.html.views import _block_layout

    one_line = {
        "id": "ffn", "kind": "ffn", "label": "Feed-forward",
    }
    honest_unknown = {
        "id": "ffn", "kind": "ffn",
        "label": ["Gated FFN", "storage unresolved"],
    }
    _, one_w, one_h, _ = _block_layout(one_line)
    _, unknown_w, unknown_h, _ = _block_layout(honest_unknown)
    assert unknown_w > one_w
    assert unknown_h > one_h
    assert unknown_w >= 340 and unknown_h >= 72


def test_top_level_single_line_truth_cannot_overflow_or_shrink_kind_floor():
    from model_unfolder.renderers.html.views import _block_layout

    _, width, height, _ = _block_layout({
        "kind": "ffn", "label": "Feed-forward (FFN)", "w": 40, "h": 10,
    })
    assert width >= 300
    assert height >= 44


def test_diffusion_bookend_label_uses_the_same_growth_law():
    """The final input/output scaffold is outside the repeated block loop, but
    its truth-bearing labels must use the identical content-fit calculation.
    This pins the Sana ``Output operations`` clipping regression.
    """
    from model_unfolder.renderers.html.views import _block_layout

    _, width, height, _ = _block_layout({
        "kind": "norm", "label": "Output operations",
        "w": 180, "h": 36, "font": 16,
    })
    assert width > 180
    assert height >= 36
