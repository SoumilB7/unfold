"""Tests for the Sable harness: the two new mechanical nets (label-lint,
wiring-conformance), the orchestrator, and the CI regression lock.

Each net has a NEGATIVE CONTROL — proof it actually fires — because a net that
can't fail is worthless (the doctrine's rule). The corpus uses the offline config
fixtures from test_diffusion so everything runs without network.
"""
from __future__ import annotations

import json

import pytest

import model_unfolder as mu
from model_unfolder import lint_labels, sable, bless, check_regression, load_corpus
from model_unfolder.evidence import check_wiring_conformance
from tests.test_diffusion import FLUX, PIXART, LLAMA

CORPUS = [("FLUX", FLUX), ("PIXART", PIXART), ("LLAMA", LLAMA)]


# --------------------------------------------------------------------------- #
# label-lint
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,cfg", CORPUS)
def test_label_lint_clean_on_corpus(name, cfg):
    assert lint_labels(mu.unfold(cfg).to_ir()) == []


def test_label_lint_flags_nested_parens_and_raw_activation():
    """The two real regressions it exists for: a tag wrapped in parens it already
    had, and a raw backend activation spelling on an activation block."""
    ir = {
        "layers": [{"blocks": [
            {"id": "attn", "kind": "attention", "label": ["Joint Attention", "(MM-DiT (dual-stream))"]},
            {"id": "activation", "kind": "activation", "label": "gelu-approximate"},
            {"id": "ok_attn", "kind": "attention", "label": ["Joint Attention", "(dual-stream)"]},
            {"id": "ok_act", "kind": "activation", "label": "GELU"},
        ]}],
        "extras": {},
    }
    problems = lint_labels(ir)
    assert any("nested/doubled parentheses" in p and "attn" in p for p in problems)
    assert any("raw backend" in p and "activation" in p for p in problems)
    # the clean siblings are NOT flagged
    assert not any("ok_attn" in p or "ok_act" in p for p in problems)


# --------------------------------------------------------------------------- #
# wiring-conformance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,cfg", [("FLUX", FLUX), ("PIXART", PIXART)])
def test_wiring_conformance_clean_on_corpus(name, cfg):
    """Every conditioning rail the diffusion diagrams draw maps to a real forward
    argument (FLUX/PixArt blocks all take temb + encoder_hidden_states)."""
    ir = mu.unfold(cfg).to_ir()
    assert [p.message for p in check_wiring_conformance(cfg, ir)] == []


def test_wiring_conformance_flags_fabricated_text_rail():
    """NEGATIVE CONTROL: a text-conditioning rail drawn into a block whose
    forward() takes no text argument is flagged. LlamaDecoderLayer.forward has no
    encoder_hidden_states, so a fabricated text rail on a llama layer must fire."""
    ir = mu.unfold(LLAMA).to_ir()
    ir["layers"][0]["blocks"].append({
        "id": "text_cond", "lane": "external_bottom_right",
        "diffusion_stage": "text_conditioning", "kind": "conditioning",
    })
    probs = check_wiring_conformance(LLAMA, ir)
    assert any(p.kind == "fabricated_input" and p.op == "text" for p in probs), \
        [p.message for p in probs]


def test_wiring_conformance_flags_missing_text_rail():
    """NEGATIVE CONTROL (missing direction): when a block's forward() TAKES a text
    input (Flux's dual block has encoder_hidden_states) but the diagram draws no
    text rail and shows no joined-sequence indication, the dropped text is flagged.
    This is the direction that caught PRX (text K/V concatenated, drawn as plain
    self-attention)."""
    ir = mu.unfold(FLUX).to_ir()
    for L in ir["layers"]:                       # strip the rail from a dual-stream layer
        tag = str((L.get("attention") or {}).get("variant", {}).get("tag") or "").lower()
        if "dual-stream" in tag:
            L["blocks"] = [b for b in L["blocks"] if b.get("id") != "text_cond"]
            break
    probs = check_wiring_conformance(FLUX, ir)
    assert any(p.kind == "missing_input" and p.op == "text" for p in probs), \
        [p.message for p in probs]


# --------------------------------------------------------------------------- #
# the orchestrator
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,cfg", CORPUS)
def test_sable_mechanical_pass_on_corpus(name, cfg):
    r = sable(cfg, render_images=False)
    assert r.oracle == "present", r.oracle      # modeling source resolved
    assert r.mechanical_passed, r.summary()
    assert r.view_hashes                         # at least one distinct view locked
    assert r.visual_review == "PENDING"          # never auto-passes the eye step


# --------------------------------------------------------------------------- #
# the CI lock
# --------------------------------------------------------------------------- #

def test_bless_requires_visual_review_and_round_trips(tmp_path):
    r = sable(FLUX, render_images=False)
    # mechanical-clean but visual PENDING -> NOT blessable.
    with pytest.raises(ValueError):
        bless(r, FLUX, corpus_dir=str(tmp_path))
    # Approve the eye step, then it locks and reproduces with no drift.
    r.visual_review = "CLEAN"
    path = bless(r, FLUX, corpus_dir=str(tmp_path))
    fixture = json.loads(open(path).read())
    assert check_regression(fixture) == []
    # Tamper the locked signature -> drift is detected.
    fixture["hash_signature"] = ["deadbeef"] + fixture["hash_signature"][1:]
    assert any("view drift" in m for m in check_regression(fixture))


def test_sable_regression_corpus():
    """Every blessed model re-renders identically — same clean mechanical pass AND
    the same per-view SVG hashes. Drift here means a diagram changed under a model
    that was signed off: re-review its gallery and re-bless if intended."""
    corpus = load_corpus()
    if not corpus:
        pytest.skip("no blessed models in tests/sable_corpus/ yet")
    for filename, fixture in corpus:
        drift = check_regression(fixture)
        assert drift == [], f"{filename} regressed:\n  " + "\n  ".join(drift)
