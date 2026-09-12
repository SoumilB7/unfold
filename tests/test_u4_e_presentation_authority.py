"""U4-E: presentation may expose canonical structure, never manufacture it."""
from __future__ import annotations

from model_unfolder.block_schema import validate_click_coupling
from model_unfolder.renderers.html.cards import (
    _build_inspect_cards,
    _build_nested_inspect_panels,
)
from model_unfolder.renderers.html.document import _render_fragment_body
from model_unfolder.renderers.html.metadata import _make_info, _meta_for


def _ir(*, layers=None, model_blocks=None) -> dict:
    return {
        "name": "fixture",
        "architecture": "Fixture",
        "vocab_size": 32,
        "hidden_size": 16,
        "max_position_embeddings": 8,
        "tie_word_embeddings": None,
        "layers": list(layers or []),
        "cross_layer_edges": [],
        "extras": {
            "render": {
                "family": "transformer",
                "layout": "decoder_only",
                "model_blocks": list(model_blocks or []),
            },
        },
        "warnings": [],
        "notes": [],
    }


def _layer(*, blocks=None, ffn=None) -> dict:
    return {
        "index": 0,
        "attention": {
            "kind": None,
            "num_heads": 0,
            "num_kv_heads": None,
            "mask": None,
        },
        "ffn": ffn or {
            "kind": None,
            "activation": None,
            "intermediate_size": 64,
            "gated": None,
            "projection_mode": None,
        },
        "norm_kind": "unknown",
        "norm_placement": "unknown",
        "residual_topology": "unknown",
        "parallel_norm_count": None,
        "blocks": list(blocks or []),
    }


def _block(node_id: str, *, children=None) -> dict:
    return {
        "id": node_id,
        "role": node_id,
        "kind": "opaque",
        "label": node_id,
        "title": f"{node_id} title",
        "description": f"{node_id} description",
        **({"children": list(children)} if children is not None else {}),
    }


def test_empty_stack_has_no_synthetic_dominant_layer_or_metadata():
    info = _make_info(_ir())
    assert info["groups"] == []
    assert info["dominant"] is None
    assert info["blocks"] == {}
    assert info["meta"] == {}


def test_empty_stack_renders_one_explicit_state_not_a_conventional_cell():
    html = _render_fragment_body(_ir(), "empty", include_font_import=False)
    assert "Repeated layer structure unavailable" in html
    assert "Attention, feed-forward and residual wiring are not inferred" in html
    assert "LAYER MAP" not in html
    assert 'data-depth="2"' not in html
    for fabricated in ("MHA", "SwiGLU", "RMSNorm", "Pre-attention norm"):
        assert fabricated not in html
    assert validate_click_coupling(html) == []


def test_cards_exist_only_for_declared_canonical_blocks():
    raw = _ir(layers=[_layer()])
    info = _make_info(raw)
    html = _build_inspect_cards(raw, info, "cards")
    assert 'data-card-id="default"' in html
    for undeclared in ("tok_text", "embed", "final_rms", "lm_head"):
        assert f'data-card-id="{undeclared}"' not in html

    raw = _ir(
        layers=[_layer()],
        model_blocks=[_block("embed"), _block("lm_head")],
    )
    info = _make_info(raw)
    html = _build_inspect_cards(raw, info, "cards")
    assert 'data-card-id="embed"' in html
    assert 'data-card-id="lm_head"' in html
    assert 'data-card-id="tok_text"' not in html
    assert 'data-card-id="final_rms"' not in html


def test_nested_cards_do_not_reconstruct_ffn_from_fact_values():
    raw = _ir(layers=[_layer(ffn={
        "kind": "dense",
        "activation": "gelu",
        "intermediate_size": 64,
        "gated": False,
        "projection_mode": "dense",
    })])
    info = _make_info(raw)
    assert _build_nested_inspect_panels(raw, info, "nested") == []
    assert _meta_for(raw, raw["layers"][0], {}) == {}


def test_declared_child_cards_survive_without_a_fallback():
    child = _block("inner_projection")
    raw = _ir(layers=[_layer(blocks=[
        _block("ffn", children=[child]),
    ])])
    info = _make_info(raw)
    levels = _build_nested_inspect_panels(raw, info, "nested")
    assert len(levels) == 1
    assert 'data-card-id="inner_projection"' in levels[0]
