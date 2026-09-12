"""Instance-level value qualification poisons for U6/U7/U8."""
from __future__ import annotations

from model_unfolder.evidence.qualification import (
    QUALIFICATION_MATRIX,
    qualification_findings,
)
from model_unfolder.evidence.expression_eval import canonical_alias_view
from model_unfolder.adapters.transformer.assembly import decoder_layer
from model_unfolder.adapters.transformer.blocks.attention import (
    attention_child_blocks,
)
from model_unfolder.ir import AttentionSpec, FFNSpec, ModelIR
from model_unfolder.labels import attention_summary
from model_unfolder.opgraph import attention_region
from model_unfolder.params import estimate_params


def _ir(attention=None, ffn=None, facts=None):
    facts = dict(facts or {})
    kind = (attention or {}).get("kind")
    if kind is not None and facts \
            and "decoder.attention.mixer_schedule" not in facts:
        facts["decoder.attention.mixer_schedule"] = {
            "value": [(
                "ordinary_attention"
                if kind in {"mha", "gqa", "mqa", "mla"} else kind)],
            "status": "code_and_config",
        }
    ffn_kind = (ffn or {}).get("kind")
    if ffn_kind is not None and facts \
            and "decoder.ffn.ffn_schedule" not in facts:
        facts["decoder.ffn.ffn_schedule"] = {
            "value": [ffn_kind],
            "status": "code_and_config",
        }
    return {
        "layers": [{
            "attention": attention or {},
            "ffn": ffn or {},
        }],
        "extras": {"fact_provenance": facts},
    }


def test_source_runtime_alias_bridge_requires_value_agreement():
    aliases = {"hidden_size": ["hidden_size", "n_embd"]}
    bridged = canonical_alias_view({"n_embd": 64}, aliases)
    assert bridged["hidden_size"] == 64

    conflicted = canonical_alias_view(
        {"hidden_size": 96, "n_embd": 64}, aliases)
    assert "hidden_size" not in conflicted
    assert conflicted["n_embd"] == 64


def test_source_runtime_alias_bridge_is_evaluation_only():
    raw = {"n_embd": 64}
    bridged = canonical_alias_view(
        raw, {"hidden_size": ["hidden_size", "n_embd"]})
    assert raw == {"n_embd": 64}
    assert bridged is not raw and bridged["hidden_size"] == 64


def test_matrix_covers_every_authoritative_geometry_sink():
    expected = {"spec", "opgraph", "card", "json", "params"}
    assert len(QUALIFICATION_MATRIX) == 17
    assert all(
        rule.surfaces == expected
        for rule in QUALIFICATION_MATRIX
        if rule.fact_key not in {
            "mask_schedule", "qk_norm_schedule", "kv_sharing_schedule",
            "position_schedule", "position_addition",
            "cross_attention_schedule", "codebook_streams",
            "per_layer_embedding_pathway"})
    assert next(rule for rule in QUALIFICATION_MATRIX
                if rule.fact_key == "codebook_streams").surfaces == {
                    "spec", "card", "json", "params"}
    assert next(rule for rule in QUALIFICATION_MATRIX
                if rule.fact_key == "per_layer_embedding_pathway").surfaces == {
                    "spec", "card", "json", "params"}
    assert next(rule for rule in QUALIFICATION_MATRIX
                if rule.fact_key == "qk_norm_schedule").surfaces == {
                    "spec", "opgraph", "card", "json"}
    assert next(rule for rule in QUALIFICATION_MATRIX
                if rule.fact_key == "position_addition").surfaces == {
                    "spec", "card", "json"}
    mask_rule = next(
        rule for rule in QUALIFICATION_MATRIX
        if rule.fact_key == "mask_schedule")
    assert mask_rule.surfaces == {"spec", "card", "json"}
    position_rule = next(
        rule for rule in QUALIFICATION_MATRIX
        if rule.fact_key == "position_schedule")
    assert position_rule.surfaces == {"spec", "opgraph", "card", "json"}
    kv_rule = next(
        rule for rule in QUALIFICATION_MATRIX
        if rule.fact_key == "kv_sharing_schedule")
    assert kv_rule.surfaces == {"spec", "card", "json"}
    cross_rule = next(
        rule for rule in QUALIFICATION_MATRIX
        if rule.fact_key == "cross_attention_schedule")
    assert cross_rule.surfaces == {"spec", "opgraph", "card", "json"}
    assert all(rule.ir_scope == "transformer_decoder"
               for rule in QUALIFICATION_MATRIX)


def test_complete_negative_kv_schedule_matches_withheld_positive_structure():
    ir = _ir(facts={"decoder.attention.kv_sharing_schedule": {
        "value": [None, None], "status": "code_and_config"}})
    ir["layers"].append({"attention": {}, "ffn": {}})
    assert qualification_findings(ir) == []

    ir["extras"]["fact_provenance"][
        "decoder.attention.kv_sharing_schedule"]["value"] = [None, 0]
    findings = qualification_findings(ir)
    assert len(findings) == 1
    assert "kv_sharing_schedule" in findings[0]


def test_head_geometry_schedule_qualifies_each_occurrence_not_a_global_mean():
    ir = _ir(
        attention={
            "kind": "gqa", "num_heads": 8,
            "num_kv_heads": 4, "head_dim": 256},
        facts={"decoder.attention.head_geometry_schedule": {
            "value": [
                ("gqa", 8, 4, 256),
                ("mqa", 8, 1, 512),
            ],
            "status": "code_and_config"},
            "decoder.attention.mixer_schedule": {
                "value": ["ordinary_attention", "ordinary_attention"],
                "status": "code_and_config"}})
    ir["layers"].append({
        "attention": {
            "kind": "mqa", "num_heads": 8,
            "num_kv_heads": 1, "head_dim": 512},
        "ffn": {},
    })
    assert qualification_findings(ir) == []
    ir["layers"][1]["attention"]["num_kv_heads"] = 2
    findings = qualification_findings(ir)
    assert len(findings) == 1
    assert "head_geometry_schedule" in findings[0]


def test_mask_schedule_fact_must_match_every_layer_value_exactly():
    ir = _ir(
        attention={"mask": "sliding", "window_size": 128},
        facts={"decoder.attention.mask_schedule": {
            "value": [("sliding", 128), ("global", None)],
            "status": "code_and_config"}})
    ir["layers"].append({
        "attention": {"mask": "global", "window_size": None}, "ffn": {}})
    assert qualification_findings(ir) == []
    ir["layers"][1]["attention"]["mask"] = "sliding"
    findings = qualification_findings(ir)
    assert len(findings) == 1
    assert "fact schedule" in findings[0]


def test_unknown_mask_is_withheld_not_an_unreceipted_schedule():
    assert qualification_findings(_ir(
        attention={"mask": "unknown", "window_size": None})) == []


def test_transformer_matrix_does_not_claim_the_diffusion_owner_altitude():
    ir = _ir(attention={
        "kind": None, "num_heads": 24,
        "num_kv_heads": 24, "head_dim": 128})
    ir["extras"]["diffusion"] = {"kind": "transformer"}
    assert qualification_findings(ir) == []


def test_registered_leaf_without_instance_fact_cannot_green():
    findings = qualification_findings(_ir(
        attention={
            "kind": "gqa", "num_heads": 8,
            "num_kv_heads": 2, "head_dim": 8}))
    assert len(findings) == 2
    assert all("without an owner-qualified instance fact" in item
               for item in findings)


def test_wrong_fact_value_cannot_green():
    findings = qualification_findings(_ir(
        ffn={"intermediate_size": 256},
        facts={
            "decoder.ffn.intermediate_size": {
                "value": 512, "status": "code_and_config"}}))
    assert len(findings) == 1
    assert "does not match" in findings[0]


def test_matching_attention_and_ffn_values_green():
    geometry = {
        "kind": "gqa", "num_heads": 8,
        "num_kv_heads": 2, "head_dim": 8,
        "q_lora_rank": None, "kv_lora_rank": None,
        "qk_nope_head_dim": None, "qk_rope_head_dim": None,
        "v_head_dim": None}
    findings = qualification_findings(_ir(
        attention=geometry,
        ffn={"intermediate_size": 256},
        facts={
            "decoder.attention.head_geometry": {
                "value": geometry, "status": "code_and_config"},
            "decoder.ffn.intermediate_size": {
                "value": 256, "status": "code_and_config"},
        }))
    assert findings == []


def test_mla_label_cannot_green_while_its_auxiliary_geometry_disappears():
    projected = {
        "kind": "mla", "num_heads": 8, "num_kv_heads": 8,
        "head_dim": 24, "q_lora_rank": 16, "kv_lora_rank": 12,
        "qk_nope_head_dim": 16, "qk_rope_head_dim": 8,
        "v_head_dim": 16}
    incomplete_fact = dict(projected)
    incomplete_fact["q_lora_rank"] = None
    findings = qualification_findings(_ir(
        attention=projected,
        facts={"decoder.attention.head_geometry": {
            "value": incomplete_fact, "status": "code_and_config"}}))
    assert len(findings) == 1
    assert "does not match" in findings[0]


def test_hybrid_attention_lanes_join_their_own_mechanism_facts():
    ordinary = {
        "kind": "gqa", "num_heads": 8, "num_kv_heads": 2,
        "head_dim": 16, "q_lora_rank": None, "kv_lora_rank": None,
        "qk_nope_head_dim": None, "qk_rope_head_dim": None,
        "v_head_dim": None}
    recurrent = {
        "kind": "gated_delta", "num_heads": 12, "num_kv_heads": 4,
        "head_dim": 8, "v_head_dim": 10, "conv_kernel_size": 4}
    ir = _ir(attention=ordinary, facts={
        "decoder.attention.head_geometry": {
            "value": ordinary, "status": "code_and_config"},
        "decoder.attention.gated_delta_geometry": {
            "value": (4, 12, 8, 10, 4), "status": "code_and_config"},
        "decoder.attention.mixer_schedule": {
            "value": ["ordinary_attention", "gated_delta"],
            "status": "code_and_config"},
    })
    ir["layers"].append({"attention": recurrent, "ffn": {}})
    assert qualification_findings(ir) == []


def test_mixer_schedule_fact_must_match_every_layer_occurrence_exactly():
    """A correct set of mechanism facts cannot launder wrong placement.

    The schedule is its own architectural claim: swapping two otherwise
    well-proven mechanisms must fail even though both candidate mechanisms are
    present and individually qualified.
    """
    ordinary = {
        "kind": "gqa", "num_heads": 8, "num_kv_heads": 2,
        "head_dim": 16, "q_lora_rank": None, "kv_lora_rank": None,
        "qk_nope_head_dim": None, "qk_rope_head_dim": None,
        "v_head_dim": None}
    recurrent = {
        "kind": "gated_delta", "num_heads": 12, "num_kv_heads": 4,
        "head_dim": 8, "v_head_dim": 10, "conv_kernel_size": 4}
    ir = _ir(attention=ordinary, facts={
        "decoder.attention.head_geometry": {
            "value": ordinary, "status": "code_and_config"},
        "decoder.attention.gated_delta_geometry": {
            "value": (4, 12, 8, 10, 4), "status": "code_and_config"},
        "decoder.attention.mixer_schedule": {
            "value": ["gated_delta", "ordinary_attention"],
            "status": "code_and_config"},
    })
    ir["layers"].append({"attention": recurrent, "ffn": {}})
    findings = qualification_findings(ir)
    assert len(findings) == 1
    assert "mixer_schedule fact schedule" in findings[0]


def test_ffn_schedule_fact_must_match_every_layer_occurrence_exactly():
    ir = _ir(ffn={"kind": "dense"}, facts={
        "decoder.ffn.ffn_schedule": {
            "value": ["moe", "dense"],
            "status": "code_and_config"},
    })
    ir["layers"].append({"attention": {}, "ffn": {"kind": "moe"}})
    findings = qualification_findings(ir)
    assert len(findings) == 1
    assert "ffn_schedule fact schedule" in findings[0]


def test_hybrid_attention_lane_cannot_borrow_the_other_mechanisms_fact():
    ir = _ir(attention={
        "kind": "gated_delta", "num_heads": 12, "num_kv_heads": 4,
        "head_dim": 8, "v_head_dim": 10, "conv_kernel_size": 4}, facts={
        "decoder.attention.head_geometry": {"value": {
            "kind": "gqa", "num_heads": 8, "num_kv_heads": 2,
            "head_dim": 16}, "status": "code_and_config"}})
    findings = qualification_findings(ir)
    assert len(findings) == 2
    assert any("gated_delta_geometry" in finding for finding in findings)
    assert any("every authoritative structural surface withholds" in finding
               for finding in findings)


def test_expert_width_is_withheld_until_its_exact_fact_exists():
    findings = qualification_findings(_ir(
        ffn={"expert_intermediate_size": 64}))
    assert len(findings) == 1
    assert "decoder.ffn.expert.expert_intermediate_size" in findings[0]


def test_moe_counts_require_router_and_shared_application_facts():
    ffn = {
        "kind": "moe", "num_experts": 8,
        "num_experts_per_tok": 2, "num_shared_experts": 1,
    }
    facts = {
        "decoder.ffn.routing_policy": {
            "value": {"num_experts": 8, "num_experts_per_tok": 2},
            "status": "code_and_config",
        },
        "decoder.ffn.expert.shared_expert_count": {
            "value": 1, "status": "code_and_config",
        },
    }
    assert qualification_findings(_ir(ffn=ffn, facts=facts)) == []

    missing = _ir(ffn=ffn, facts={
        "decoder.ffn.routing_policy": facts["decoder.ffn.routing_policy"]})
    findings = qualification_findings(missing)
    assert any("shared_expert_count" in item for item in findings)

    wrong = _ir(ffn=ffn, facts=facts)
    wrong["extras"]["fact_provenance"][
        "decoder.ffn.routing_policy"]["value"]["num_experts"] = 16
    findings = qualification_findings(wrong)
    assert any("routing_policy" in item and "does not match" in item
               for item in findings)


def test_global_fact_cannot_launder_heterogeneous_occurrences():
    ir = _ir(
        ffn={"intermediate_size": 256},
        facts={"decoder.ffn.intermediate_size": {
            "value": 256, "status": "code_and_config"}})
    ir["layers"].append({
        "attention": {}, "ffn": {"intermediate_size": 512}})
    findings = qualification_findings(ir)
    assert len(findings) == 1
    assert "per-occurrence facts" in findings[0]


def test_codebook_structure_requires_the_exact_instance_fact():
    value = {"num": 4, "embeddings_summed": True, "heads_stacked": True}
    ir = _ir(facts={"decoder.codebook_streams": {
        "value": value, "status": "code_and_config"}})
    ir["extras"]["render"] = {"model_blocks": ({
        "id": "tok_text", "detail": dict(value),
    },)}
    assert qualification_findings(ir) == []

    del ir["extras"]["fact_provenance"]["decoder.codebook_streams"]
    findings = qualification_findings(ir)
    assert any("codebook_streams" in item for item in findings)


def test_legacy_raw_codebook_extras_cannot_satisfy_qualification():
    value = {"num": 4, "embeddings_summed": True, "heads_stacked": True}
    ir = _ir(facts={"decoder.codebook_streams": {
        "value": value, "status": "code_and_config"}})
    ir["extras"]["codebooks"] = {**value, "vocab_per_book": 2048}
    findings = qualification_findings(ir)
    assert any("codebook_streams" in item and "withholds" in item
               for item in findings)


def test_cross_attention_schedule_qualifies_replacement_and_additive_layers():
    ir = _ir(facts={"decoder.attention.cross_attention_schedule": {
        "value": ["self", "replacement_cross", "additive_cross"],
        "status": "code_and_config",
    }})
    ir["layers"] = [
        {"attention": {"cross_attention": False}, "ffn": {}},
        {"attention": {"cross_attention": True}, "ffn": {}},
        {"attention": {"cross_attention": False},
         "cross_attention": {"cross_attention": True}, "ffn": {}},
    ]
    assert qualification_findings(ir) == []

    ir["extras"]["fact_provenance"][
        "decoder.attention.cross_attention_schedule"]["value"][1] = "self"
    findings = qualification_findings(ir)
    assert any("cross_attention_schedule fact schedule" in item
               for item in findings)


def test_consumers_do_not_reconstruct_withheld_head_geometry():
    """Unknown must stay unknown after the parser, on every qualified sink."""
    attention = AttentionSpec(
        kind="gqa", num_heads=8, num_kv_heads=2, head_dim=None,
        projection_mode="split_qkv")

    region = attention_region(vars(attention), 64)
    assert region.by_id()["q_proj"].out_features is None
    assert region.by_id()["k_proj"].out_features is None

    cards = attention_child_blocks(attention, 64)
    facts = [fact for card in cards for fact in (card.get("facts") or ())]
    assert "64 → 64" not in facts
    assert "64 → ?" in facts

    _description, summary_facts = attention_summary(vars(attention))
    assert "8 Q heads" in summary_facts
    assert "2 KV heads" in summary_facts
    assert not any("head dim" in fact for fact in summary_facts)

    layer = decoder_layer(
        0, attention, FFNSpec(kind=None), hidden_size=64)
    model = ModelIR(
        name="withheld-geometry", architecture="Synthetic",
        vocab_size=128, hidden_size=64, max_position_embeddings=16,
        tie_word_embeddings=True, layers=[layer])
    estimate = estimate_params(model)
    assert estimate["per_layer"][0]["attn"] == 0
    assert any("attention geometry unresolved" in item
               for item in estimate["assumptions"])


def test_params_do_not_count_gated_delta_as_ordinary_attention():
    """Exact recurrent geometry cannot authorize an ordinary Q/K/V/O formula."""
    attention = AttentionSpec(
        kind="gated_delta", num_heads=12, num_kv_heads=4,
        head_dim=8, v_head_dim=10, conv_kernel_size=4)
    layer = decoder_layer(
        0, attention, FFNSpec(kind=None), hidden_size=64)
    model = ModelIR(
        name="recurrent", architecture="Synthetic",
        vocab_size=128, hidden_size=64, max_position_embeddings=16,
        tie_word_embeddings=True, layers=[layer])
    estimate = estimate_params(model)
    assert estimate["per_layer"][0]["attn"] == 0
    assert any("gated-delta parameter formula not yet migrated" in item
               for item in estimate["assumptions"])
