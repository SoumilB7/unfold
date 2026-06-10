"""The universal card declarer (view:"ops") and the push rule behind it.

The class of bug these guard: a card whose facts describe drawable structure
(dims + an activation — i.e. a chain, not a single op) but which renders as
prose because nobody attached a view.  Structure must be *declared* in the op
alphabet and projected by the one renderer — never flattened into a sentence.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_unfolder import unfold
from model_unfolder.opgraph import ops_region
from model_unfolder.renderers.html.block_views.registry import VIEW_REGISTRY

PIXTRAL_STYLE = {
    "architectures": ["LlavaForConditionalGeneration"], "model_type": "llava",
    "image_token_index": 10, "projector_hidden_act": "gelu",
    "text_config": {"model_type": "mistral", "hidden_size": 5120, "num_hidden_layers": 4,
                    "num_attention_heads": 32, "num_key_value_heads": 8,
                    "intermediate_size": 14336, "vocab_size": 131072,
                    "rms_norm_eps": 1e-5, "head_dim": 128},
    "vision_config": {"model_type": "pixtral", "hidden_size": 1024, "image_size": 1024,
                      "patch_size": 16, "num_hidden_layers": 24,
                      "num_attention_heads": 16, "intermediate_size": 4096},
}

QWEN2VL_STYLE = {
    "architectures": ["Qwen2VLForConditionalGeneration"], "model_type": "qwen2_vl",
    "image_token_id": 151655,
    "text_config": {"model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 4,
                    "num_attention_heads": 28, "num_key_value_heads": 4,
                    "intermediate_size": 18944, "vocab_size": 152064,
                    "rms_norm_eps": 1e-6},
    "vision_config": {"model_type": "qwen2_vl_vision", "embed_dim": 1280,
                      "hidden_size": 3584, "patch_size": 14, "temporal_patch_size": 2,
                      "spatial_merge_size": 2, "depth": 32, "num_heads": 16},
}


def test_ops_region_builds_a_chain_with_implicit_wiring():
    r = ops_region([
        {"kind": "linear", "label": "Linear", "in": 1024, "out": 5120},
        {"kind": "activation", "fn": "gelu"},
        {"kind": "linear", "label": "Linear", "in": 5120, "out": 5120},
    ], rid="proj")
    assert [o.id for o in r.ops] == ["hidden", "proj_op0", "proj_op1", "proj_op2"]
    assert r.ops[0].out_features == 1024          # in-port caption width
    assert [(e.src, e.dst) for e in r.edges] == [
        ("hidden", "proj_op0"), ("proj_op0", "proj_op1"), ("proj_op1", "proj_op2")]
    assert r.template == "declared" and r.merges() == []


def test_ops_region_wires_branches_by_from():
    r = ops_region([
        {"id": "a", "kind": "linear"},
        {"id": "b", "kind": "linear", "from": "hidden"},
        {"id": "join", "kind": "elementwise", "fn": "mul", "from": ["a", "b"]},
    ], rid="g")
    assert r.merges() == ["join"]


def test_ops_region_fails_loudly_on_a_typo():
    with pytest.raises(ValueError, match="liner"):
        ops_region([{"kind": "liner"}], rid="x")
    with pytest.raises(ValueError, match="unknown op"):
        ops_region([{"kind": "linear", "from": "nope"}], rid="x")


def test_mlp_projector_card_declares_its_ops():
    """The pixtral case: the projector card embeds linear→act→linear, with the
    chips and diagram derived from the same facts."""
    assert "ops" in VIEW_REGISTRY
    html = unfold(PIXTRAL_STYLE).to_html(standalone=True)
    i = html.find('data-card-id="vision_projector"')
    seg = html[i:i + 4000]
    assert "MLP projector" in seg
    assert "1,024 → 5,120" in seg                  # chip
    assert "<svg" in seg and "in (1,024)" in seg   # declared-ops diagram
    assert "GELU" in seg


def test_patch_merger_card_declares_its_ops():
    html = unfold(QWEN2VL_STYLE).to_html(standalone=True)
    i = html.find('data-card-id="vision_projector"')
    seg = html[i:i + 4000]
    assert "Patch merger" in seg and "Concat" in seg and "<svg" in seg


# Views that render their children as *op nodes* of one canonical region —
# those child cards are leaves of an already-drawn diagram, so prose is the
# right format for them.  Children of path/tower/encoder views are diagram
# *blocks* and stay auditable.
_OP_LEVEL_VIEWS = {
    "ffn", "gated_ffn", "dense_ffn", "moe", "moe_expert", "attention",
    "mla_query_path", "mla_kv_cache_path", "vision_self_attention",
    "vision_mlp", "vision_patch_embedding", "vae_decoder_block", "ops",
}


def _walk_cards(ir: dict):
    """Every diagram-node card the renderer can show, wherever authored —
    minus the op-leaf children of canonical-region views (the click lookup
    flattens those to the top level, so exempt them by walking the tree)."""
    from model_unfolder.renderers.html.metadata import _block_lookup, _make_info
    info = _make_info(ir)
    cards = {}
    for group in info["groups"]:
        cards.update(_block_lookup(ir, group["spec"]))
    cards.update(info["blocks"])
    exempt = set()
    stack = list(cards.values())
    while stack:
        b = stack.pop()
        if not isinstance(b, dict):
            continue
        kids = b.get("children") or []
        if b.get("view") in _OP_LEVEL_VIEWS:
            exempt.update(id(k) for k in kids)
        stack.extend(kids)
    return [b for b in cards.values() if id(b) not in exempt]


def test_no_structural_card_renders_as_prose():
    """The push rule: a card whose own facts describe a *chain* (dims plus an
    activation) must declare a view — named template or declared ops.  This is
    what turns 'Soumil finds prose cards by clicking' into a CI failure."""
    import re
    acts = re.compile(r"\b(gelu|silu|relu|swiglu|geglu|quick_gelu|gelu_new|swish)\b", re.I)
    dims = re.compile(r"\d[\d,]*\s*(?:→|->)\s*\d")
    offenders = []
    for cfg in (PIXTRAL_STYLE, QWEN2VL_STYLE):
        ir = unfold(cfg).to_ir()
        for card in _walk_cards(ir):
            text = " ".join([card.get("description") or ""] + list(card.get("facts") or []))
            if dims.search(text) and acts.search(text) and not card.get("view"):
                offenders.append((card.get("id"), text[:80]))
    assert not offenders, f"structural cards with no view: {offenders}"


def test_every_declared_op_gets_a_derived_card_automatically():
    """Cards are the THIRD projection of the region: a new ops view needs no
    hand-written per-node descriptions — title/sentence/chips derive from the
    same op list that draws the SVG, and the nodes become click targets."""
    html = unfold(PIXTRAL_STYLE).to_html(standalone=True)
    for i in range(3):
        assert f'data-id="vision_projector_op{i}"' in html      # clickable node
        assert f'data-card-id="vision_projector_op{i}"' in html  # derived card
    assert "Element-wise non-linearity." in html                 # kind vocabulary


def test_op_card_vocabulary_derives_titles_and_facts():
    from model_unfolder.labels import cards_from_region, op_card
    from model_unfolder.opgraph import Op

    linear = op_card(Op("x", "linear", "Linear", in_features=1024, out_features=5120))
    assert linear["description"] and linear["facts"] == ["1,024 → 5,120"]
    act = op_card(Op("a", "activation", fn="gelu"))
    assert act["title"] == "GELU" and act["facts"] == ["gelu"]
    opaque = op_card(Op("o", "opaque", meta={"class_name": "MyBlock"}))
    assert opaque["title"] == "MyBlock"
    region = ops_region([{"kind": "linear"}, {"kind": "norm"}], rid="z")
    assert [c["id"] for c in cards_from_region(region)] == ["z_op0", "z_op1"]
    assert all(c["description"] for c in cards_from_region(region))


def test_authored_children_win_over_derived_cards():
    """The derivation is a floor, not a cage: a block that already declares
    children keeps them untouched."""
    from model_unfolder.renderers.html.metadata import _ensure_declared_op_cards
    block = {"id": "p", "view": "ops",
             "detail": {"ops": [{"kind": "linear"}]},
             "children": [{"id": "custom", "title": "Mine", "description": "kept"}]}
    _ensure_declared_op_cards(block)
    assert [c["id"] for c in block["children"]] == ["custom"]
