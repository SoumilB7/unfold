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

# Shared fixtures live in the importable test_support package (§16.1).
from test_support import PIXTRAL_STYLE, QWEN2VL_STYLE, MISTRAL3_STYLE


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
    seg = html[i:i + 9000]
    assert "Patch merger" in seg and "Reshape / merge patches" in seg
    assert "LayerNorm" in seg and "GELU" in seg and seg.count("Linear") >= 2


def test_vision_patch_profiles_preserve_source_order_and_concrete_backend():
    pixtral = unfold(PIXTRAL_STYLE).to_ir()["extras"]["modalities"]["inputs"]["vision"]
    qwen = unfold(QWEN2VL_STYLE).to_ir()["extras"]["modalities"]["inputs"]["vision"]
    # Labels are humanized STRUCTURAL names (a raw torch class on a box was
    # the Theme-L leak); the concrete backend survives as card provenance in
    # the conv op's description, and consecutive tensor moves collapse into
    # ONE regroup step whose description enumerates them in source order.
    pix_ops = pixtral["embedding"]["ops"]
    # SEMANTIC reshapes (the variable-size image JOIN) stand alone — only the
    # axis plumbing (flatten/transpose) folds into the regroup step.
    assert [op["label"] for op in pix_ops] == [
        "Patch convolution", "Crop patches", "Regroup patch tokens",
        "Join patch sequences", "RMSNorm"
    ]
    assert "Conv2d" in (pix_ops[0].get("description") or "")
    assert ("Flatten spatial grid → Transpose to tokens"
            in (pix_ops[2].get("description") or ""))
    qwen_ops = qwen["embedding"]["ops"]
    assert [op["label"] for op in qwen_ops] == [
        "Reshape patches", "Patch convolution", "Flatten tokens"
    ]
    assert "Conv3d" in (qwen_ops[1].get("description") or "")
    for cfg in (PIXTRAL_STYLE, QWEN2VL_STYLE):
        html = unfold(cfg).to_html(standalone=True)
        assert "Linear / Conv2d" not in html
        assert 'data-card-id="vision_patches"' in html


def test_pixtral_encoder_uses_rmsnorm_and_a_gated_vision_mlp():
    vision = unfold(PIXTRAL_STYLE).to_ir()["extras"]["modalities"]["inputs"]["vision"]
    assert vision["encoder"]["norm_kind"] == "RMSNorm"
    assert vision["encoder"]["ffn_gated"] is True
    html = unfold(PIXTRAL_STYLE).to_html(standalone=True)
    for node_id in ("vision_enc_ffn_gate_proj", "vision_enc_ffn_up_proj",
                    "vision_enc_ffn_multiply"):
        assert f'data-id="{node_id}"' in html
        assert f'data-card-id="{node_id}"' in html


def test_mistral3_projector_includes_norm_patch_merge_and_two_linear_mlp():
    vision = unfold(MISTRAL3_STYLE).to_ir()["extras"]["modalities"]["inputs"]["vision"]
    projector = vision["projector"]
    assert "profile" not in projector
    assert projector["source_class"] == "Mistral3MultiModalProjector"
    assert [op.get("label") or op.get("fn") for op in projector["ops"]] == [
        "RMSNorm", "Split image sequences", "Arrange spatial grid",
        "Extract merge windows", "Flatten merge windows", "Join image sequences",
        "Patch merge", "Linear (in)", "gelu", "Linear (out)"
    ]
    html = unfold(MISTRAL3_STYLE).to_html(standalone=True)
    assert "Patch merger" in html
    assert "Extract merge windows" in html and "RMSNorm" in html


def test_qwen_video_path_reuses_the_same_conv3d_and_patch_merger_profiles():
    cfg = {**QWEN2VL_STYLE, "video_token_id": 151656}
    video = unfold(cfg).to_ir()["extras"]["modalities"]["inputs"]["video"]
    # "Conv3d" (a raw torch class on a box) became the humanized structural
    # label — the Theme-L leak fix; the video path still reuses the exact
    # same three-op profile as the image path.
    assert [op["label"] for op in video["embedding"]["ops"]] == [
        "Reshape patches", "Patch convolution", "Flatten tokens"
    ]
    assert "profile" not in video["projector"]
    assert video["projector"]["source_class"] == "PatchMerger"
    html = unfold(cfg).to_html(standalone=True)
    assert 'data-card-id="video_patches"' in html
    assert 'data-card-id="video_projector"' in html


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
