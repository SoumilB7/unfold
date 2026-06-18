"""Block-schema + click-coupling regression gate.

Turns the "silently renders wrong" class of bugs into a failing test:
  * the block tree only uses known keys, ids are present/unique, and every
    `view` is registered;
  * every clickable node in the rendered HTML resolves to a card.

If you add a block key, a view, or a detail diagram, this is what catches an
unregistered view, a typo'd key, or a view drawing a node-id no block declares.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from model_unfolder import unfold
from model_unfolder.adapters.transformer.parser import parse
from model_unfolder.block_schema import (
    KNOWN_BLOCK_KEYS,
    validate_block_tree,
    validate_click_coupling,
)

_BASE = dict(
    model_type="m", num_hidden_layers=3, hidden_size=128, num_attention_heads=8,
    num_key_value_heads=2, intermediate_size=256, vocab_size=1000, rms_norm_eps=1e-5,
)

# A corpus spanning the topologies that produce distinct block shapes.
CORPUS = {
    "dense": _BASE,
    "moe_mla": dict(
        _BASE, kv_lora_rank=64, q_lora_rank=96, n_routed_experts=8,
        num_experts_per_tok=2, moe_intermediate_size=128, first_k_dense_replace=1,
        scoring_func="sigmoid", n_group=4, topk_group=2, norm_topk_prob=True,
        routed_scaling_factor=2.5,
    ),
    "parallel_residual": dict(_BASE, use_parallel_residual=True, hidden_act="gelu"),
    "mtp": dict(_BASE, num_nextn_predict_layers=2),
    "per_layer_embedding": dict(_BASE, hidden_size_per_layer_input=64, vocab_size_per_layer_input=1000),
    "sliding_window": dict(_BASE, sliding_window=1024, use_sliding_window=True, max_window_layers=1),
    # Block-worthiness paradigm in anger: DiffusionGemma is the one family that
    # exercises Tier-2 `static` connectors (⊕ merges) and inline parallel
    # `branch_side` branches (dense MLP ∥ MoE).  Pinning it here means the schema
    # must keep blessing the paradigm keys — they can't silently become "typos".
    "block_paradigm": dict(
        model_type="diffusion_gemma",
        text_config=dict(
            _BASE, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=128,
        ),
    ),
}


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_block_tree_is_schema_valid(name):
    ir = parse(CORPUS[name])
    problems = validate_block_tree(ir)
    assert problems == [], f"{name} block-tree schema problems:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_every_clickable_node_has_a_card(name):
    html = unfold(CORPUS[name]).to_html(standalone=True)
    problems = validate_click_coupling(html)
    assert problems == [], f"{name} click-coupling problems:\n  " + "\n  ".join(problems)


# --- the validator must actually catch each silent-failure class ------------

def _fake_ir(blocks):
    return SimpleNamespace(layers=[SimpleNamespace(blocks=blocks)], extras={})


def test_validator_catches_unregistered_view():
    p = validate_block_tree(_fake_ir([{"id": "a", "view": "not_a_real_view"}]))
    assert any("not registered" in m for m in p)


def test_validator_catches_unknown_key_typo():
    p = validate_block_tree(_fake_ir([{"id": "a", "lable": "typo"}]))
    assert any("unknown key" in m for m in p)


def test_validator_catches_missing_and_duplicate_id():
    assert any("no string 'id'" in m for m in validate_block_tree(_fake_ir([{"role": "attention"}])))
    assert any("duplicate id" in m for m in validate_block_tree(_fake_ir([{"id": "x"}, {"id": "x"}])))


def test_click_coupling_flags_orphan_node():
    html = '<g data-id="router_gate"></g><div data-card-id="router"></div>'
    assert validate_click_coupling(html)  # router_gate has no card
    assert not validate_click_coupling('<g data-id="router"></g><div data-card-id="router"></div>')


def test_known_keys_cover_the_real_tree():
    # Every key the real render tree emits must be in the schema, else valid
    # blocks would be falsely flagged as typos.
    ir = parse(CORPUS["moe_mla"])
    from model_unfolder.block_schema import iter_block_tree
    used = set()
    for _scope, block in iter_block_tree(ir):
        used |= set(block)
    assert used <= KNOWN_BLOCK_KEYS, f"render tree uses keys not in schema: {sorted(used - KNOWN_BLOCK_KEYS)}"


def test_ffn_detail_view_uses_clicked_block_not_dominant_group():
    """Dense/gated/MoE detail views must not drift to info['dominant'].

    The deliberately mismatched contexts below reproduce the old failure mode:
    a clicked FFN could render as whichever layer type happened to be dominant.
    """
    from model_unfolder.renderers.html.block_views.registry import render_block_detail
    from model_unfolder.renderers.html.metadata import _make_info

    dense_ir = parse(dict(
        model_type="phi", num_hidden_layers=1, hidden_size=128, num_attention_heads=8,
        intermediate_size=256, vocab_size=1000, hidden_act="gelu", layer_norm_eps=1e-5,
    )).to_dict()
    moe_ir = parse(CORPUS["moe_mla"]).to_dict()

    dense_block = next(b for b in dense_ir["layers"][0]["blocks"] if b["id"] == "ffn")
    moe_block = next(b for b in moe_ir["layers"][1]["blocks"] if b["id"] == "ffn")

    dense_in_moe_context = render_block_detail(dense_ir, _make_info(moe_ir), "ffn-dense", dense_block)
    assert "Linear (in)" in dense_in_moe_context
    assert "Linear (gate)" not in dense_in_moe_context
    assert "Expert 1" not in dense_in_moe_context

    moe_in_dense_context = render_block_detail(moe_ir, _make_info(dense_ir), "ffn-moe", moe_block)
    assert "Expert 1" in moe_in_dense_context
    assert "Router" in moe_in_dense_context
    assert "Linear (in)" not in moe_in_dense_context


def test_attention_detail_view_uses_clicked_block_not_dominant_group():
    from model_unfolder.renderers.html.block_views.registry import render_block_detail
    from model_unfolder.renderers.html.metadata import _make_info

    mha_ir = parse(dict(
        model_type="phi", num_hidden_layers=1, hidden_size=128, num_attention_heads=8,
        intermediate_size=256, vocab_size=1000, hidden_act="gelu", layer_norm_eps=1e-5,
    )).to_dict()
    gqa_ir = parse(dict(
        model_type="m", num_hidden_layers=1, hidden_size=128, num_attention_heads=8,
        num_key_value_heads=2, intermediate_size=256, vocab_size=1000, rms_norm_eps=1e-5,
    )).to_dict()

    mha_block = next(b for b in mha_ir["layers"][0]["blocks"] if b["id"] == "attn")
    gqa_block = next(b for b in gqa_ir["layers"][0]["blocks"] if b["id"] == "attn")

    mha_in_gqa_context = render_block_detail(mha_ir, _make_info(gqa_ir), "attn-mha", mha_block)
    assert "grouped-query attention" not in mha_in_gqa_context
    assert "gqa-attn-arrow" not in mha_in_gqa_context

    gqa_in_mha_context = render_block_detail(gqa_ir, _make_info(mha_ir), "attn-gqa", gqa_block)
    assert "grouped-query attention" in gqa_in_mha_context
    assert "gqa-attn-arrow" in gqa_in_mha_context
