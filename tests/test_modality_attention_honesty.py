"""Modality summary cards must not classify attention from head counts."""

from model_unfolder.renderers.html.metadata_modalities import (
    _encoder_attention_child,
)


def test_grouped_head_geometry_does_not_fabricate_gqa():
    child = _encoder_attention_child(
        "audio_enc",
        {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
        },
    )[0]

    assert child["title"] == "Attention mechanism unresolved"
    assert "no MHA/GQA graph is inferred" in child["description"]
    assert "Grouped-query" not in child["title"]
    assert any("8" in fact for fact in child["facts"])
    assert any("2" in fact for fact in child["facts"])


def test_code_proven_modality_attention_kind_is_preserved():
    child = _encoder_attention_child(
        "video_enc",
        {
            "attention_kind": "gqa",
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
        },
    )[0]

    assert child["title"] == "Grouped-query self-attention"
    assert child["description"] == (
        "The exact tower source proves this token-mixing mechanism."
    )


def test_missing_kv_head_count_is_not_replaced_with_query_head_count():
    child = _encoder_attention_child(
        "audio_enc",
        {"hidden_size": 1024, "num_attention_heads": 8},
    )[0]

    assert child["title"] == "Attention mechanism unresolved"
    assert not any("KV" in fact for fact in child["facts"])
