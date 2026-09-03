"""U4 anti-fabrication controls for unresolved architectural facts."""

import json
from pathlib import Path

import model_unfolder as mu

from model_unfolder.ir import AttentionSpec, FFNSpec
from model_unfolder.labels import (
    attention_label,
    attention_title,
    kind_long,
    kind_short,
    mask_long,
    mask_short,
    ffn_summary,
)
from model_unfolder.opgraph import attention_region, ffn_region
from model_unfolder.opgraph import mla_kv_region, mla_query_region
from model_unfolder.expanded.attention import build_attention
from model_unfolder.expanded.ffn import build_ffn
from model_unfolder.adapters.transformer.blocks.feed_forward import (
    ffn_child_blocks,
)
from model_unfolder.adapters.transformer.blocks.attention import (
    attention_detail,
)


def _unknown_attention(**overrides):
    values = {
        "kind": None,
        "num_heads": 8,
        "num_kv_heads": 8,
        "head_dim": 64,
        "mask": None,
    }
    values.update(overrides)
    return AttentionSpec(**values)


def test_missing_or_unrecognized_dict_kind_never_formats_as_mha():
    for value in (None, "", "unknown", "novel_mixer"):
        attention = {"kind": value}
        assert kind_short(attention) == "Attn unresolved"
        assert kind_long(attention) == "Attention mechanism unresolved"


def test_missing_or_unrecognized_mask_never_formats_as_causal():
    for value in (None, "", "unknown", "novel_mask"):
        attention = {"mask": value}
        assert mask_short(attention) == "unresolved"
        assert mask_long(attention) == "Mask unresolved"


def test_known_attention_and_mask_vocabulary_is_preserved():
    assert kind_short({"kind": "mha"}) == "MHA"
    assert kind_long({"kind": "gqa"}) == "Grouped-query attention"
    assert mask_short({"mask": "causal"}) == "causal"
    assert mask_long({"mask": "full"}) == "Full (bidirectional)"


def test_typed_unknown_attention_is_explicit_on_layer_surfaces():
    attention = _unknown_attention()
    assert attention_label(attention) == ["Attention", "(mechanism unresolved)"]
    assert attention_title(attention) == "Attention mechanism unresolved"


def test_typed_unknown_cross_attention_stays_cross_but_not_mha():
    attention = _unknown_attention(cross_attention=True)
    assert attention_label(attention) == [
        "Cross-Attention",
        "(K/V source unresolved)",
    ]
    assert attention_title(attention) == "Cross-attention mechanism unresolved"
    assert kind_short({"kind": None, "cross_attention": True}) == "XAttn unresolved"


def test_unknown_cross_attention_opgraph_preserves_its_proven_role():
    region = attention_region(
        {
            "kind": None,
            "cross_attention": True,
            "cross_kv_source": "encoded text prompt",
        },
        128,
    )
    assert region.resolved is False
    assert region.label == "Cross-attention mechanism unresolved"
    assert [(op.kind, op.label) for op in region.ops] == [
        ("input", None),
        ("opaque", "Cross-attention mechanism unresolved"),
        ("input", ["Encoded text"]),
    ]
    assert [(edge.src, edge.dst) for edge in region.edges] == [
        ("hidden", "block"),
        ("cross_attention_states", "block"),
    ]
    assert not {"q_proj", "k_proj", "v_proj", "scaled_scores"} & set(
        region.by_id()
    )


def test_variant_cannot_hide_an_unknown_attention_mechanism():
    variant = {
        "short": "Joint Attn",
        "tag": "MM-DiT",
        "title": "Joint attention over text and image tokens",
        "label": ["Joint Attention", "(dual-stream)"],
    }
    as_dict = {"kind": None, "variant": variant}
    assert kind_short(as_dict).endswith("unresolved")
    assert kind_long(as_dict).endswith("attention mechanism unresolved")
    typed = _unknown_attention(variant=variant)
    assert attention_label(typed)[-1] == "(unresolved)"
    assert attention_title(typed).endswith("attention mechanism unresolved")


def _corpus_ir(slug: str) -> dict:
    path = Path(__file__).parent / "sable_test_corpus" / f"{slug}.json"
    config = json.loads(path.read_text())["config"]
    return mu.unfold(config).to_ir()


def _layer_kinds(slug: str) -> set:
    return {
        layer["attention"].get("kind")
        for layer in (_corpus_ir(slug).get("layers") or [])
    }


def test_real_transformer_head_geometry_keeps_its_known_gqa_kind():
    assert _layer_kinds("qwen3-8b") == {"gqa"}


def test_config_selected_sana_lane_projects_only_positive_relations():
    """Sana's applications come from its selected occurrence, not a file vote.

    Exact constructor/config binding selects the guarded block branch.  Source
    then proves the self/context applications, while their internal attention
    and FFN mechanisms remain unknown and therefore opaque.
    """
    ir = _corpus_ir("sana-1600m-1024px-diffusers")
    assert _layer_kinds("sana-1600m-1024px-diffusers") == {None}
    assert {
        tuple(block.get("kind") for block in layer.get("blocks", ()))
        for layer in ir["layers"]
    } == {("block", "attention", "gate_mul", "attention", "norm", "ffn")}
    attention_blocks = [
        block for block in ir["layers"][0]["blocks"]
        if block["kind"] == "attention"
    ]
    assert [block["id"] for block in attention_blocks] == ["attn", "cross_attn"]
    assert all(block["detail"]["attention"]["kind"] is None
               for block in attention_blocks)
    assert [child["id"] for child in attention_blocks[0]["children"]] == [
        "opaque_mixer",
    ]
    assert [child["id"] for child in attention_blocks[1]["children"]] == [
        "opaque_mixer", "cross_attention_states",
    ]
    assert all(block["children"][0]["resolved"] is False
               for block in attention_blocks)
    # The cross-attention placement/source is exact even though its inner
    # mechanism is opaque, so the external input keeps its own truthful card.
    assert "external context" in (
        attention_blocks[1]["children"][1]["description"])


def test_source_proven_sana_cross_attention_keeps_mechanism_unknown():
    ir = _corpus_ir("sana-1600m-1024px-diffusers")
    layer = ir["layers"][0]
    assert layer["attention"]["kind"] is None
    assert layer["attention"]["cross_attention"] is False
    assert layer["cross_attention"]["cross_attention"] is True
    assert layer["cross_attention"]["cross_kv_source"] == "external context"
    assert layer["cross_attention"]["kind"] is None
    assert [block["id"] for block in layer["blocks"]] == [
        "cross_attention_states", "attn", "attn_condition_gate_0",
        "cross_attn", "wiring_unresolved", "ffn",
    ]


def test_real_unproven_diffusion_attention_no_longer_defaults_to_mha():
    for slug in ("flux-2-dev", "stable-diffusion-3-5-large"):
        kinds = _layer_kinds(slug)
        assert "mha" not in kinds
        # A source-proven count may materialize opaque layers (Flux); an
        # unresolved root count may materialize none (SD3).  Neither case is a
        # license to manufacture an attention mechanism.
        assert kinds in ({None}, set())


def test_expanded_unknown_attention_keeps_geometry_without_qkv_or_sdpa():
    expanded = mu.unfold(
        {
            "model_type": "unknown-safe-fixture",
            "architectures": ["UnknownSafeForCausalLM"],
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 0,
            "intermediate_size": 256,
        },
        return_json=True,
    )
    attention = expanded["layer_groups"][0]["attention"]
    assert attention["kind"] is None
    assert attention["projections"] == {}
    assert [node["operation"] for node in
            attention["operation_graph"]["nodes"]] == ["opaque"]
    # Cache is not applicable until an attention mechanism is known; it must
    # not be inferred from mask/geometry or represented as a proven negative.
    assert attention["cache"] == {
        "enabled": None,
        "status": "not_applicable",
    }


def test_attention_internal_defaults_are_unknown_not_conventional():
    attention = AttentionSpec(kind="gqa", num_heads=8)
    assert attention.qk_norm is None
    assert attention.bias is None
    assert attention.rope is None
    assert attention.cached is None
    assert attention.projection_mode is None
    assert attention.scores_scaled is None
    detail = attention_detail(attention)
    assert detail["q_norm"] is None
    assert detail["k_norm"] is None


def test_attention_summary_distinguishes_false_unknown_and_position_unknown():
    from model_unfolder.labels import attention_summary

    _, unknown = attention_summary({
        "kind": "gqa", "num_heads": 8, "num_kv_heads": 2,
        "head_dim": 64, "qk_norm": None, "bias": None,
        "cached": None, "projection_mode": None, "scores_scaled": None,
        "position_kind": "unknown", "position_application": "unknown",
    })
    assert {
        "bias unresolved", "QK norm unresolved", "cache unresolved",
        "QKV storage unresolved", "score scaling unresolved",
        "position application unresolved",
    } <= set(unknown)

    _, negative = attention_summary({
        "kind": "gqa", "num_heads": 8, "num_kv_heads": 2,
        "head_dim": 64, "qk_norm": False, "bias": False,
        "cached": False, "projection_mode": "split_qkv",
        "scores_scaled": False,
        "position_kind": "none", "position_application": "none",
    })
    assert {"bias-free projections", "no QK norm", "no KV cache",
            "no attention-stage position op"} <= set(negative)


def test_u6_projection_bias_debt_rows_are_retired_at_both_owner_altitudes():
    """Exact source readers replaced both root and recursive pending rows."""
    from model_unfolder.evidence.structural_debt import pending_projection_paths

    pending = pending_projection_paths()
    assert ("root", "attention_bias") not in pending
    assert ("root.text_encoder", "attention_bias") not in pending


def test_unknown_projection_storage_never_becomes_split_qkv():
    region = attention_region(
        {
            "kind": "gqa",
            "num_heads": 8,
            "num_kv_heads": 2,
            "head_dim": 64,
            "projection_mode": None,
            "scores_scaled": True,
        },
        512,
    )
    ids = {op.id for op in region.ops}
    assert "qkv_projection_unresolved" in ids
    assert not {"q_proj", "k_proj", "v_proj", "qkv_proj"} & ids


def test_projection_storage_requires_an_explicit_split_or_fused_fact():
    split = attention_region(
        {
            "kind": "gqa", "num_heads": 8, "num_kv_heads": 2,
            "head_dim": 64, "projection_mode": "split_qkv",
            "scores_scaled": True,
        },
        512,
    )
    fused = attention_region(
        {
            "kind": "gqa", "num_heads": 8, "num_kv_heads": 2,
            "head_dim": 64, "projection_mode": "fused_qkv",
            "scores_scaled": True,
        },
        512,
    )
    assert {"q_proj", "k_proj", "v_proj"} <= {op.id for op in split.ops}
    assert "qkv_proj" in {op.id for op in fused.ops}


def test_cache_and_score_scaling_are_independent_tristate_facts():
    base = {
        "kind": "gqa", "num_heads": 8, "num_kv_heads": 2,
        "head_dim": 64, "projection_mode": "split_qkv",
        "position_kind": "none", "position_application": "none",
    }
    unknown = build_attention(base, 512, "groups[0]", None)
    assert unknown["cache"] == {"enabled": None, "status": "unresolved"}
    unknown_scores = next(
        node for node in unknown["operation_graph"]["nodes"]
        if node["id"] == "scores"
    )
    assert "formula" not in unknown_scores
    assert unknown_scores["status"] == "unresolved"

    cached = build_attention(
        {**base, "cached": True, "scores_scaled": True},
        512, "groups[0]", None,
    )
    assert cached["cache"]["enabled"] is True
    assert "kv_cache" in {
        node["id"] for node in cached["operation_graph"]["nodes"]
    }
    scaled_scores = next(
        node for node in cached["operation_graph"]["nodes"]
        if node["id"] == "scores"
    )
    assert scaled_scores["formula"] == "QK^T/sqrt(dim)"

    uncached = build_attention(
        {**base, "cached": False, "scores_scaled": False},
        512, "groups[0]", None,
    )
    assert uncached["cache"] == {"enabled": False, "kind": "none"}
    assert "kv_cache" not in {
        node["id"] for node in uncached["operation_graph"]["nodes"]
    }
    raw_scores = next(
        node for node in uncached["operation_graph"]["nodes"]
        if node["id"] == "scores"
    )
    assert raw_scores["formula"] == "QK^T"


def test_declared_score_constant_cannot_author_the_scale_operation():
    base = {
        "kind": "gqa", "num_heads": 8, "num_kv_heads": 2,
        "head_dim": 64, "projection_mode": "split_qkv",
        "scores_scale": 1 / 32,
    }
    unproved = next(
        op for op in attention_region(base, 512).ops
        if op.id == "scaled_scores"
    )
    assert unproved.meta["status"] == "unresolved"
    assert "formula" not in unproved.meta

    proved = next(
        op for op in attention_region(
            {**base, "scores_scaled": True}, 512
        ).ops
        if op.id == "scaled_scores"
    )
    assert proved.meta["formula"] == "QK^T/32"


def test_rope_requires_the_exact_application_fact():
    base = {
        "kind": "gqa", "num_heads": 8, "num_kv_heads": 2,
        "head_dim": 64, "projection_mode": "split_qkv",
        "scores_scaled": True,
    }
    for incomplete in (
        {**base, "rope": True},
        {**base, "position_kind": "rope"},
        {**base, "position_application": "qk_rotation"},
    ):
        ids = {op.id for op in attention_region(incomplete, 512).ops}
        assert not {"q_rope", "k_rope"} & ids
    proven = {
        **base, "rope": True, "position_kind": "rope",
        "position_application": "qk_rotation",
    }
    assert {"q_rope", "k_rope"} <= {
        op.id for op in attention_region(proven, 512).ops
    }


def test_mla_latent_structure_does_not_imply_cache_rope_or_scaling():
    attention = {
        "kind": "mla",
        "num_heads": 16,
        "head_dim": 128,
        "kv_lora_rank": 256,
        "qk_nope_head_dim": 96,
        "qk_rope_head_dim": 32,
    }
    parent = attention_region(attention, 2048)
    kv = mla_kv_region(attention, 2048)
    query = mla_query_region(attention, 2048)
    assert "cache" not in {
        op.kind for op in (*parent.ops, *kv.ops, *query.ops)
    }
    assert "rope" not in {
        op.kind for op in (*parent.ops, *kv.ops, *query.ops)
    }
    scores = next(op for op in parent.ops if op.id == "scaled_scores")
    assert scores.meta["status"] == "unresolved"
    assert "formula" not in scores.meta


# ---------------------------------------------------------------------------
# U4-C — FFN facts are independent and unknown never picks a familiar MLP.
# ---------------------------------------------------------------------------


def test_omitted_ffn_fields_are_typed_unknown_not_modern_decoder_defaults():
    ffn = FFNSpec()
    assert ffn.kind is None
    assert ffn.activation is None
    assert ffn.intermediate_size is None
    assert ffn.gated is None
    assert ffn.projection_mode is None
    assert ffn.expert_projection_mode is None


def test_unknown_ffn_mechanism_keeps_width_without_gate_up_down():
    region = ffn_region(
        {
            "kind": None,
            "intermediate_size": 256,
            "activation": "gelu",
        },
        64,
    )
    assert region.template == "undeclared"
    assert region.resolved is False
    assert [op.kind for op in region.ops] == ["opaque"]
    assert region.ops[0].meta["intermediate_size"] == 256
    assert region.ops[0].meta["activation"] == "gelu"
    assert not {
        "gate_proj", "up_proj", "down_proj", "multiply",
    } & {op.id for op in region.ops}


def test_known_gate_without_storage_stays_opaque():
    for gated in (True, False):
        region = ffn_region(
            {
                "kind": "dense",
                "gated": gated,
                "activation": "silu",
                "intermediate_size": 256,
                "projection_mode": None,
            },
            64,
        )
        assert region.template == "unresolved_storage"
        assert [op.kind for op in region.ops] == ["opaque"]


def test_ffn_storage_must_agree_with_gate_topology():
    for values in (
        {"gated": False, "projection_mode": "split"},
        {"gated": False, "projection_mode": "fused_gate_up"},
        {"gated": True, "projection_mode": "dense"},
    ):
        region = ffn_region(
            {
                "kind": "dense",
                "activation": "silu",
                "intermediate_size": 256,
                **values,
            },
            64,
        )
        assert region.template == "unresolved_storage"


def test_known_ffn_shape_does_not_require_a_guessed_activation():
    dense = ffn_region(
        {
            "kind": "dense", "gated": False,
            "projection_mode": "dense", "activation": None,
            "intermediate_size": 256,
        },
        64,
    )
    gated = ffn_region(
        {
            "kind": "dense", "gated": True,
            "projection_mode": "split", "activation": None,
            "intermediate_size": 256,
        },
        64,
    )
    assert dense.template == "dense_mlp"
    assert gated.template == "gated_mlp"
    assert next(op for op in dense.ops if op.id == "activation").fn is None
    assert next(op for op in gated.ops if op.id == "activation").fn is None


def test_routed_expert_never_borrows_ordinary_shared_ffn_facts():
    spec = FFNSpec(
        kind="moe",
        gated=True,
        projection_mode="split",
        activation="silu",
        intermediate_size=128,
        expert_intermediate_size=64,
        expert_projection_mode=None,
        num_experts=8,
        num_experts_per_tok=2,
    )
    children = ffn_child_blocks(spec, 32)
    expert = next(child for child in children if child["id"] == "expert_1")
    assert [child["id"] for child in expert["children"]] == ["block"]
    detail = expert["detail"]["ffn"]
    assert detail["gated"] is None
    assert detail["expert_projection_mode"] is None
    assert detail["activation"] is None

    region = ffn_region(
        {
            "kind": "moe",
            "gated": True,
            "projection_mode": "split",
            "activation": "silu",
            "expert_projection_mode": None,
            "intermediate_size": 128,
            "expert_intermediate_size": 64,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        },
        32,
    )
    expert_op = next(op for op in region.ops if op.id == "expert")
    assert expert_op.meta["gated"] is None
    assert expert_op.meta["intermediate_size"] == 64
    assert expert_op.meta["activation"] is None


def test_expanded_ffn_projects_exact_tri_states_and_expert_width():
    ffn = {
        "kind": "moe",
        "gated": None,
        "projection_mode": None,
        "activation": None,
        "intermediate_size": 128,
        "expert_intermediate_size": 64,
        "expert_projection_mode": None,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_shared_experts": None,
    }
    expanded = build_ffn(ffn, 32, "layers[0]", None)
    assert expanded["gated"] is None
    assert expanded["structure_declared"] is False
    assert expanded["experts"]["expert_intermediate_size"] == 64
    assert expanded["experts"]["shared"] is None
    expert_graph = next(
        node for node in expanded["operation_graph"]["nodes"]
        if node["id"] == "expert_template"
    )["graph"]
    assert [node["operation"] for node in expert_graph["nodes"]] == ["opaque"]


def test_ffn_summary_never_calls_unknown_dense_or_swiglu():
    desc, facts = ffn_summary({
        "kind": None,
        "gated": None,
        "activation": None,
        "intermediate_size": 256,
    })
    assert "mechanism unresolved" in desc
    assert "mechanism unresolved" in facts
    assert "Dense" not in desc
    assert "SwiGLU" not in desc
