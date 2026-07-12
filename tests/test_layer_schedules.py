"""U3 — the general per-layer TYPE-SCHEDULE engine.

One engine, six spellings: every config dialect below answers the SAME question
("which type is layer i, along dimension D?") and normalizes into the canonical
``layer_types`` list / MoE-membership list the per-layer walk already consumes.
Tests pin the FORM (value-list / pattern-tile / nested-tile / dense-interval /
comma-string) against tmp configs, the None-on-doubt guardrail, the byte-stable
fallback (a config that already carries ``layer_types`` is untouched), and — when
cached — the real remote-code/installed witnesses.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

import pytest

from model_unfolder.parser import config_to_ir
from model_unfolder.adapters.transformer.blocks.attention import attention_child_blocks
from model_unfolder.opgraph import attention_region

HUB = os.path.expanduser("~/.cache/huggingface/hub")


def _cached(repo: str):
    for snap in glob.glob(os.path.join(
            HUB, f"models--{repo.replace('/', '--')}", "snapshots", "*")):
        path = os.path.join(snap, "config.json")
        if os.path.exists(path):
            return json.load(open(path))
    return None


def _kinds(ir):
    return Counter(l.attention.kind for l in ir.layers)


def _masks(ir):
    return Counter(l.attention.mask for l in ir.layers)


def _ffns(ir):
    return Counter(l.ffn.kind for l in ir.layers)


def _sched_fact(ir, name="layer_schedule"):
    fp = ir.extras.get("fact_provenance") or {}
    return fp.get(f"decoder.layer.{name}") or fp.get(f"decoder.ffn.{name}")


# ---------------------------------------------------------------------------
# FORM: value_list  (attn_type_list — MiniMax lightning/full mixer codes)
# ---------------------------------------------------------------------------

def test_attn_type_list_int_codes_split_lightning_and_full():
    """0 = lightning (linear) mixer, 1 = full softmax attention — a per-layer
    int-coded list, one token per layer."""
    cfg = {
        "model_type": "minimax_text_01",
        "architectures": ["MiniMaxText01ForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 8, "num_attention_heads": 8, "vocab_size": 128,
        "attn_type_list": [0, 0, 0, 0, 0, 0, 0, 1],
    }
    ir = config_to_ir(cfg)
    assert _kinds(ir) == {"declared_lightning_mixer": 7, "mha": 1}
    # The schedule proves placement/name, not Lightning's internal graph or
    # positional behavior. The mixer remains opaque and position-unknown.
    mixers = [l.attention for l in ir.layers
              if l.attention.kind == "declared_lightning_mixer"]
    assert all(a.position_kind == "unknown" for a in mixers)
    fact = _sched_fact(ir)
    assert fact and fact["status"] == "config_declared" and fact["source"] == "attn_type_list"


def test_declared_mixer_cannot_fall_through_to_familiar_internals():
    """Name/placement evidence must remain opaque at both operation and card depth."""
    cfg = {
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 1, "num_attention_heads": 8, "vocab_size": 128,
        "attn_type_list": [0],
    }
    attn = config_to_ir(cfg).layers[0].attention
    region = attention_region({"kind": attn.kind}, 64)
    assert region.template == "opaque"
    assert len(region.ops) == 1
    cards = attention_child_blocks(attn, 64)
    assert [card["id"] for card in cards] == ["opaque_mixer"]
    assert cards[0].get("resolved") is False


def test_attn_type_list_all_full_stays_uniform():
    """MiniMax-M2 ships attn_type_list all-1s — every layer is full attention, so
    the honest verdict is ONE type (no false split)."""
    cfg = {
        "model_type": "minimax_m2", "architectures": ["MiniMaxM2ForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 6, "num_attention_heads": 8, "vocab_size": 128,
        "attn_type_list": [1, 1, 1, 1, 1, 1],
    }
    ir = config_to_ir(cfg)
    assert set(_kinds(ir)) == {"mha"}


def test_attn_type_list_shorter_than_stack_is_doubt():
    """A partial per-layer list can't decide the whole stack — None-on-doubt (no
    schedule, the uniform fallback stands rather than a fabricated split)."""
    cfg = {
        "model_type": "minimax_text_01", "architectures": ["MiniMaxText01ForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 8, "num_attention_heads": 8, "vocab_size": 128,
        "attn_type_list": [0, 0, 1],   # only 3 of 8
    }
    ir = config_to_ir(cfg)
    assert "declared_lightning_mixer" not in _kinds(ir)
    assert _sched_fact(ir) is None


# ---------------------------------------------------------------------------
# FORM: pattern_tile  (block_types / _block_types — recurrentgemma GRIFFIN)
# ---------------------------------------------------------------------------

def test_block_types_tiles_recurrent_and_local_attention():
    """A short (recurrent, recurrent, attention) pattern tiles to the stack; the
    attention blocks are LOCAL (windowed) when a window is declared."""
    cfg = {
        "model_type": "recurrent_gemma",
        "architectures": ["RecurrentGemmaForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 6, "num_attention_heads": 8, "vocab_size": 128,
        "_block_types": ["recurrent", "recurrent", "attention"],
        "attention_window_size": 2048,
    }
    ir = config_to_ir(cfg)
    assert _kinds(ir) == {"declared_recurrent_mixer": 4, "mha": 2}
    # the attention layers slide (local window), the recurrent ones are mixers
    assert _masks(ir)["sliding"] == 2
    assert all(l.attention.position_kind == "unknown" for l in ir.layers
               if l.attention.kind == "declared_recurrent_mixer")


def test_block_types_attention_stays_full_without_window():
    """No declared window -> the "attention" blocks are plain (not fabricated as
    sliding); the recurrent split still holds."""
    cfg = {
        "model_type": "recurrent_gemma", "architectures": ["RecurrentGemmaForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 6, "num_attention_heads": 8, "vocab_size": 128,
        "block_types": ["recurrent", "recurrent", "attention"],
    }
    ir = config_to_ir(cfg)
    assert _kinds(ir)["declared_recurrent_mixer"] == 4
    assert _masks(ir)["sliding"] == 0


# ---------------------------------------------------------------------------
# FORM: nested_tile  (attention_types — gpt-neo global/local)
# ---------------------------------------------------------------------------

def test_attention_types_nested_pattern_global_local():
    """[[["global","local"], N]] expands to N x [global, local]; global -> full
    (causal/global), local -> sliding."""
    cfg = {
        "model_type": "gpt_neo", "architectures": ["GPTNeoForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_layers": 6, "num_heads": 8, "vocab_size": 128,
        "attention_types": [[["global", "local"], 3]],
        "window_size": 256,
    }
    ir = config_to_ir(cfg)
    assert _masks(ir) == {"global": 3, "sliding": 3}


# ---------------------------------------------------------------------------
# FORM: dense_interval  (dense_attention_every_n_layers — Phi-3-small)
# ---------------------------------------------------------------------------

def test_dense_interval_alternates_dense_and_sparse():
    """Every Nth layer is dense; the rest are compressed-sparse."""
    cfg = {
        "model_type": "phi3small", "architectures": ["Phi3SmallForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 6, "num_attention_heads": 8, "vocab_size": 128,
        "dense_attention_every_n_layers": 2,
    }
    ir = config_to_ir(cfg)
    assert _masks(ir)["compressed_sparse"] == 3
    assert _masks(ir)["causal"] == 3


# ---------------------------------------------------------------------------
# FORM: moe_comma_string  (moe_layers_enum — step3)
# ---------------------------------------------------------------------------

def test_moe_layers_enum_comma_string_membership():
    """"2,3,4,5" carves the MoE layers out of the dense ones (layers 0,1 dense)."""
    cfg = {
        "model_type": "step3_text", "architectures": ["Step3TextForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 6, "num_attention_heads": 8, "vocab_size": 128,
        "num_experts": 8, "num_experts_per_tok": 2,
        "moe_layers_enum": "2,3,4,5",
    }
    ir = config_to_ir(cfg)
    assert _ffns(ir) == {"dense": 2, "moe": 4}
    assert [l.index for l in ir.layers if l.ffn.kind == "dense"] == [0, 1]
    fact = _sched_fact(ir, "moe_schedule")
    assert fact and fact["source"] == "moe_layers_enum"


# ---------------------------------------------------------------------------
# Byte-stable fallback: a canonical layer_types list is NOT overridden.
# ---------------------------------------------------------------------------

def test_canonical_layer_types_wins_over_schedule_spellings():
    """When the config already carries the canonical ``layer_types`` list it is
    used verbatim — the schedule normalizer is a fallback, never an override."""
    cfg = {
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 128,
        "layer_types": ["full_attention", "full_attention",
                        "full_attention", "full_attention"],
        # a stray schedule spelling must NOT re-carve the stack
        "attn_type_list": [0, 0, 1, 1],
    }
    ir = config_to_ir(cfg)
    assert set(_kinds(ir)) == {"mha"}          # not split by attn_type_list
    assert _sched_fact(ir) is None


def test_linear_attention_token_still_gated_delta():
    """The existing Qwen3-Next spelling (layer_types == linear_attention) keeps
    mapping to the gated_delta cell (byte-stability)."""
    cfg = {
        "model_type": "qwen3_next", "architectures": ["Qwen3NextForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 128,
        "layer_types": ["linear_attention", "linear_attention",
                        "full_attention", "full_attention"],
    }
    ir = config_to_ir(cfg)
    assert _kinds(ir)["gated_delta"] == 2
    gd = next(l for l in ir.layers if l.attention.kind == "gated_delta")
    assert (gd.attention.variant or {}).get("short") == "Gated DeltaNet"


def test_block_configs_nas_projection_is_quarantined_until_source_bound():
    """A config schedule alone may not invent NAS topology or width formulas.

    This deliberately pins the pre-H8 quarantine: once the exact model source
    binds no_op/replacement/width/norm semantics, this test must be replaced by
    source-backed counterexamples rather than weakened.
    """
    base = {
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 2, "num_attention_heads": 8, "vocab_size": 128,
    }
    declared = {
        **base,
        "block_configs": [
            {"attention": {"no_op": True},
             "ffn": {"ffn_mult": 3.5}},
            {"attention": {"replace_with_linear": True},
             "ffn": {"replace_with_linear": True}},
        ],
    }
    plain_ir = config_to_ir(base)
    declared_ir = config_to_ir(declared)
    assert [layer.signature() for layer in declared_ir.layers] == [
        layer.signature() for layer in plain_ir.layers
    ]
    assert all("pruned" not in str(layer.blocks).lower()
               for layer in declared_ir.layers)


# ---------------------------------------------------------------------------
# Real cached witnesses (skip when the snapshot isn't on disk).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("repo,expect_kinds,expect_masks", [
    ("MiniMaxAI/MiniMax-Text-01", {"declared_lightning_mixer": 70, "gqa": 10}, None),
    ("google/recurrentgemma-2b", {"declared_recurrent_mixer": 18, "gqa": 8}, {"sliding": 8}),
    ("microsoft/Phi-3-small-8k-instruct", None, {"causal": 16, "compressed_sparse": 16}),
    ("EleutherAI/gpt-neo-125M", None, {"global": 6, "sliding": 6}),
])
def test_real_config_schedules(repo, expect_kinds, expect_masks):
    cfg = _cached(repo)
    if cfg is None:
        pytest.skip(f"{repo} not cached")
    ir = config_to_ir(cfg)
    if expect_kinds is not None:
        assert dict(_kinds(ir)) == expect_kinds
    if expect_masks is not None:
        got = _masks(ir)
        for k, v in expect_masks.items():
            assert got[k] == v


def test_real_step3_moe_membership():
    cfg = _cached("stepfun-ai/step3")
    if cfg is None:
        pytest.skip("step3 not cached")
    ir = config_to_ir(cfg)
    dense = [l.index for l in ir.layers if l.ffn.kind == "dense"]
    # enum "4..59" -> layers 0-3 and 60 are dense (5 dense)
    assert dense == [0, 1, 2, 3, 60]
