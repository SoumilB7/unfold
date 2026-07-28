"""U4 anti-fabrication controls for unresolved architectural facts."""

import json
from pathlib import Path

import model_unfolder as mu

from model_unfolder.ir import AttentionSpec
from model_unfolder.labels import (
    attention_label,
    attention_title,
    kind_long,
    kind_short,
    mask_long,
    mask_short,
)
from model_unfolder.opgraph import attention_region


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
        "(unresolved)",
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
        ("opaque", "Cross-attention mechanism unresolved"),
    ]


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


def test_real_code_proven_special_diffusion_attention_is_preserved():
    assert _layer_kinds("sana-1600m-1024px-diffusers") == {"linear"}


def test_real_cross_attention_existence_does_not_fabricate_its_mechanism():
    ir = _corpus_ir("sana-1600m-1024px-diffusers")
    cross = next(
        block
        for block in ir["layers"][0]["blocks"]
        if block.get("id") == "cross_attn"
    )
    assert cross["detail"]["attention"]["cross_attention"] is True
    assert cross["detail"]["attention"]["cross_kv_source"] == "encoded text prompt"
    assert cross["detail"]["attention"]["kind"] is None
    assert cross["label"] == [
        "Cross-Attention",
        "(unresolved)",
    ]
    assert cross["title"] == "Cross-attention mechanism unresolved"
    assert "mechanism unresolved" in cross["facts"]


def test_real_unproven_diffusion_attention_no_longer_defaults_to_mha():
    for slug in ("flux-2-dev", "stable-diffusion-3-5-large"):
        kinds = _layer_kinds(slug)
        assert "mha" not in kinds
        assert kinds == {None}


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
    assert attention["cache"] == {"enabled": False}
