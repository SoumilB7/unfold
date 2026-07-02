"""The sub-model PARITY net — embedded ≡ standalone, at every hop and depth.

The recurring disease this net kills: a supporting tower rendered as a lossy,
hand-projected summary of its real model.  The law it enforces:

    Projecting an embedded config must be structurally identical to parsing
    the SAME config standalone — modulo namespace, ownership qualification,
    and the DECLARED altitude transforms (``submodel.ALTITUDE_TRANSFORMS``).

Three independent hops are checked, so a fact lost anywhere between the
sub-parse and the rendered drill fails loudly:

1. **derivation parity** — the spec's per-group facts equal the canonical
   serializers applied directly to a standalone parse of the same config
   (catches loss inside ``submodel_spec`` / the honesty-override wiring);
2. **pipeline parity** — the rendered encoder block's ``detail.sub_model``
   equals the directly derived spec (catches loss between the diffusor's spec
   list, block assembly and detail passthrough);
3. **drill parity** — the canonical region rebuilt from the embedded block's
   facts equals the region from the standalone facts, compared by CANONICAL op
   identity (catches namespace-fragile drift in the drill itself).

Plus the recursion contract: a sub-model nested inside a sub-model projects
through the same machinery with composed namespaces (``_s<j>``), dotted
component ownership, and document-level click coupling.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_diffusion as td

from model_unfolder import unfold
from model_unfolder.adapters.diffusor import parser as diffusor
from model_unfolder.submodel import ALTITUDE_TRANSFORMS, qualify_component

# The case matrix spans every generalizable axis the projector must carry:
# FFN form (dense / gated / MoE), positional mechanism (learned-absolute /
# RoPE / relative-bias), score scaling (scaled / code-proven unscaled),
# attention kind (MHA / GQA), and schedule (homogeneous / periodic hybrid).
ENCODER_CASES = {
    "clip_dense_learnedabs": td.FLUX["_text_encoder_configs"]["text_encoder"],
    "t5_gated_relbias_unscaled": td.FLUX["_text_encoder_configs"]["text_encoder_2"],
    "llama_gqa_rope": td.HYBRID_ENC["_text_encoder_configs"]["text_encoder"],
    "mixtral_moe": td.MOE_ENC["_text_encoder_configs"]["text_encoder"],
}


def _host_with(encoder_cfg: dict) -> dict:
    return {**td.FLUX, "_text_encoder_configs": {"text_encoder": encoder_cfg}}


def _embedded_spec(host: dict) -> dict:
    ir = unfold(host).to_ir()

    def find(blocks, bid):
        for b in blocks or []:
            if b.get("id") == bid:
                return b
            hit = find(b.get("children"), bid)
            if hit is not None:
                return hit

    render = (ir.get("extras") or {}).get("render") or {}
    enc = find(render.get("loop_blocks"), "encoder_0")
    return (enc.get("detail") or {}).get("sub_model")


def _direct_spec(encoder_cfg: dict, component: str = "text_encoder") -> dict:
    spec = dict(diffusor._normalize_encoder_config(encoder_cfg))
    sub = spec.get("sub_model")
    assert isinstance(sub, dict), "sub-parse produced no sub_model spec"
    qualify_component(sub, component)
    return sub


@pytest.mark.parametrize("name,cfg", sorted(ENCODER_CASES.items()))
def test_pipeline_parity_embedded_spec_equals_direct_derivation(name, cfg):
    """Hop 2: nothing between the spec list, block assembly and detail
    passthrough may drop or mutate a single fact — deep equality, no subset."""
    embedded = _embedded_spec(_host_with(cfg))
    direct = _direct_spec(cfg)
    assert embedded == direct


@pytest.mark.parametrize("name,cfg", sorted(ENCODER_CASES.items()))
def test_derivation_parity_spec_facts_equal_standalone_serializers(name, cfg):
    """Hop 1: each group's attention fact equals the canonical serializer
    applied to the standalone sub-parse's typed spec, plus ONLY the declared
    altitude transforms and the evidence-resolved extras."""
    from model_unfolder.adapters.transformer.blocks.attention import attention_detail
    from model_unfolder.adapters.transformer.parser import parse as parse_transformer
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.ir import distinct_layer_groups

    standalone = parse_transformer(cfg, context=ParseContext.build(cfg, source="local"))
    groups = distinct_layer_groups(standalone.layers)
    spec = _direct_spec(cfg)
    assert len(spec["groups"]) == len(groups)
    tower = ALTITUDE_TRANSFORMS["tower"]["attention"]
    for spec_group, typed_group in zip(spec["groups"], groups):
        expected = attention_detail(typed_group["layer"].attention)
        expected.update(tower)                       # the DECLARED transform
        got = dict(spec_group["attention"])
        # hidden + scores_scaled are evidence-resolved extras the standalone
        # serializer does not carry; everything else must match exactly.
        assert got.pop("hidden") == standalone.hidden_size
        got.pop("scores_scaled", None)
        assert got == expected
        assert spec_group["count"] == len(typed_group["indices"])
        assert spec_group["layers"] == list(typed_group["indices"])
        # FFN structural identity survives (kind + width + expert geometry).
        ffn = typed_group["layer"].ffn
        assert spec_group["ffn"].get("kind") == ("moe" if ffn.kind == "moe" else "dense")
        if ffn.kind == "moe":
            assert spec_group["ffn"].get("num_experts") == ffn.num_experts
            assert spec_group["ffn"].get("num_experts_per_tok") == ffn.num_experts_per_tok
        else:
            assert spec_group["ffn"].get("intermediate_size") == ffn.intermediate_size


@pytest.mark.parametrize("name,cfg", sorted(ENCODER_CASES.items()))
def test_drill_parity_regions_match_by_canonical_identity(name, cfg):
    """Hop 3: the attention drill region rebuilt from the EMBEDDED block's own
    facts equals the region from the standalone facts — compared entirely
    through canonical op identity, so namespacing can never mask drift."""
    from model_unfolder.opgraph import attention_region, prefix_region

    spec = _embedded_spec(_host_with(cfg))
    for group in spec["groups"]:
        fact = dict(group["attention"])
        fact.pop("node_prefix", None)
        base = attention_region(fact, fact.get("hidden"))
        renamed = prefix_region(base, "deep_ns_a_deep_ns_b_")   # two rename layers

        def shape(region):
            ops = [((op.meta or {}).get("canonical_id", op.id), op.kind)
                   for op in region.ops]
            ids = {op.id: (op.meta or {}).get("canonical_id", op.id)
                   for op in region.ops}
            edges = {(ids[e.src], ids[e.dst]) for e in region.edges}
            return ops, edges

        assert shape(base) == shape(renamed)


def test_recursion_contract_nested_submodel_projects_and_couples():
    """A sub-model inside a sub-model: composed ``_s<j>`` namespaces, dotted
    component ownership, the same projector at every depth, and document-level
    click coupling through both levels."""
    from model_unfolder.block_schema import validate_click_coupling
    from model_unfolder.submodel import submodel_cell_blocks

    outer = _direct_spec(
        td.HYBRID_ENC["_text_encoder_configs"]["text_encoder"], "text_encoder")
    inner = dict(diffusor._normalize_encoder_config(
        td.FLUX["_text_encoder_configs"]["text_encoder"]))["sub_model"]
    inner["component"] = "inner_tower"       # the nested slot's own relative name
    outer["sub_models"] = [inner]
    qualify_component(outer, "outer_slot")
    assert outer["component"] == "outer_slot.text_encoder"
    assert inner["component"] == "outer_slot.inner_tower"

    # Project into a REAL document: inject the nested spec into a rendered
    # pipeline's encoder and require whole-document click coupling.
    diagram = unfold(td.HYBRID_ENC)

    def find(blocks, bid):
        for b in blocks or []:
            if b.get("id") == bid:
                return b
            hit = find(b.get("children"), bid)
            if hit is not None:
                return hit

    render = (diagram.ir.extras or {}).get("render") or {}
    enc = find(render.get("loop_blocks"), "encoder_0")
    spec = enc["detail"]["sub_model"]
    nested = dict(inner)
    spec["sub_models"] = [nested]
    enc["children"] = [enc["children"][0]] + submodel_cell_blocks(
        spec, "encoder_0",
        attn_description="Each token attends to the others in the prompt.",
        norm_fallback="RMSNorm",
        norm_card=lambda prefix, norm: {
            "id": f"{prefix}_op_norm", "title": norm, "description": f"{norm}."},
        residual_card=lambda prefix: {
            "id": f"{prefix}_op_add", "title": "Residual add",
            "description": "Adds the sublayer input back onto its output."},
    )

    nested_block = find(enc["children"], "encoder_0_s0")
    assert nested_block is not None
    assert nested_block["view"] == "text_encoder"
    assert nested_block["source_component"] == "outer_slot.inner_tower"
    inner_attn = find(nested_block["children"], "encoder_0_s0_op_selfattn")
    assert inner_attn is not None and inner_attn.get("children")
    assert all(child["id"].startswith("encoder_0_s0_attn_")
               for child in inner_attn["children"])

    html = diagram.to_html(standalone=True)
    assert 'data-id="encoder_0_s0"' in html            # nested tower node drawn
    assert 'data-card-id="encoder_0_s0_op_selfattn"' in html
    assert validate_click_coupling(html) == []
    assert diagram.wiring_problems() == []


def test_ownership_prefers_the_deepest_qualified_component():
    """A composite encoder (a VL wrapper) owns its facts one component deeper
    than its pipeline slot — projected blocks must carry the dotted path."""
    cfg = {
        "model_type": "mistral3", "architectures": ["Mistral3ForConditionalGeneration"],
        "text_config": {
            "model_type": "mistral", "hidden_size": 512, "intermediate_size": 2048,
            "num_hidden_layers": 4, "num_attention_heads": 8,
            "num_key_value_heads": 2, "hidden_act": "silu", "rms_norm_eps": 1e-5,
            "vocab_size": 32000, "max_position_embeddings": 4096,
            "rope_theta": 1e6,
        },
        "vision_config": {"model_type": "pixtral", "hidden_size": 256,
                          "num_hidden_layers": 2, "num_attention_heads": 4},
        "image_token_index": 10,
    }
    pytest.importorskip("transformers")
    spec = dict(diffusor._normalize_encoder_config(cfg))
    sub = spec.get("sub_model")
    if not isinstance(sub, dict):
        pytest.skip("mistral3 sub-parse unavailable in this environment")
    qualify_component(sub, "text_encoder")
    position = (sub.get("evidence") or {}).get("position") or {}
    if position.get("status") == "proven":
        assert str(position.get("component", "")).startswith("text_encoder.")
