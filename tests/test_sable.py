"""Tests for the Sable harness: the two new mechanical nets (label-lint,
wiring-conformance), the orchestrator, and the CI regression lock.

Each net has a NEGATIVE CONTROL — proof it actually fires — because a net that
can't fail is worthless (the doctrine's rule). The corpus uses the offline config
fixtures from test_diffusion so everything runs without network.
"""
from __future__ import annotations

import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import pytest

import model_unfolder as mu
from model_unfolder import lint_labels, sable, bless, check_regression, load_corpus
from model_unfolder.evidence import check_fact_conformance, check_wiring_conformance
from tests.test_diffusion import FLUX, PIXART, LLAMA

CORPUS = [("FLUX", FLUX), ("PIXART", PIXART), ("LLAMA", LLAMA)]


def test_plain_parse_resolves_model_source_once(monkeypatch):
    """Every architectural detector consumes one call-local source bundle."""
    from model_unfolder.evidence import context as context_module

    real = context_module.resolve_source_files
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(context_module, "resolve_source_files", counted)
    mu.unfold(LLAMA)
    assert len(calls) == 1


def test_sable_parse_and_all_conformance_nets_share_one_source_bundle(monkeypatch):
    """Sable must not rediscover source separately for parse and each net."""
    from model_unfolder.evidence import context as context_module

    real = context_module.resolve_source_files
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(context_module, "resolve_source_files", counted)
    sable(LLAMA, render_images=False)
    assert len(calls) == 1


def test_concurrent_transformer_and_diffusion_renders_are_call_local():
    """Theme and graph diagnostics cannot cross-contaminate concurrent calls."""
    from model_unfolder.renderers.html.document import render_document
    from model_unfolder.renderers.html.render_context import (
        RenderContext,
        activate_render_context,
    )

    llama_ir = mu.unfold(LLAMA).to_ir()
    flux_ir = deepcopy(mu.unfold(FLUX).to_ir())
    # Production currently chooses teal for both domains. Force the registered
    # blue palette here so the test exercises actual cross-theme isolation.
    flux_ir["extras"]["render"]["theme"] = "blue"

    def render(ir, mount, theme):
        context = RenderContext(theme=theme)
        with activate_render_context(context):
            html = render_document(ir, mount)
        return html, context

    llama_expected, _ = render(llama_ir, "concurrent-llama", "teal")
    flux_expected, _ = render(flux_ir, "concurrent-flux", "blue")

    for _ in range(3):
        with ThreadPoolExecutor(max_workers=2) as pool:
            llama_future = pool.submit(render, llama_ir, "concurrent-llama", "teal")
            flux_future = pool.submit(render, flux_ir, "concurrent-flux", "blue")
            llama_html, llama_context = llama_future.result()
            flux_html, flux_context = flux_future.result()

        assert llama_html == llama_expected
        assert flux_html == flux_expected
        assert "#0F6E56" in llama_html and "#1E5FB0" not in llama_html
        assert "#1E5FB0" in flux_html and "#0F6E56" not in flux_html
        assert llama_context.events and flux_context.events
        assert not llama_context.wiring_findings
        assert not flux_context.wiring_findings


def test_render_events_carry_block_path_component_and_variant():
    diagram = mu.unfold(FLUX)
    diagram.to_html(standalone=True)
    events = diagram.render_events()
    assert events
    attn = next(event for event in events if event.view == "attn")
    assert attn.block_path == ("attn",)
    assert attn.component == "root"
    assert "MM-DiT" in attn.variant
    assert attn.source_owner == "FluxTransformer2DModel"
    assert "linear" in attn.drawn_ops


# --------------------------------------------------------------------------- #
# label-lint
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,cfg", CORPUS)
def test_label_lint_clean_on_corpus(name, cfg):
    assert lint_labels(mu.unfold(cfg).to_ir()) == []


def test_label_lint_flags_nested_parens_and_raw_activation():
    """Negative controls for text/design classes that repeatedly slipped through
    non-visual checks: nested labels, raw backend activation, numeric facts on
    blocks, backend alternatives, and static Tier-2 connectors."""
    ir = {
        "layers": [{"blocks": [
            {"id": "attn", "kind": "attention", "label": ["Joint Attention", "(MM-DiT (dual-stream))"]},
            {"id": "activation", "kind": "activation", "label": "gelu-approximate"},
            {"id": "clip", "kind": "embedding", "label": "CLIP (768-d)"},
            {"id": "patch", "kind": "linear", "label": "Linear / Conv2d"},
            {"id": "encoder", "kind": "attention", "label": "Encoder ×30"},
            {"id": "bad_add", "kind": "residual_add", "label": "⊕", "static": True},
            {"id": "ok_attn", "kind": "attention", "label": ["Joint Attention", "(dual-stream)"]},
            {"id": "ok_act", "kind": "activation", "label": "GELU"},
            {"id": "ok_conv", "kind": "linear", "label": "Conv2d"},
        ]}],
        "extras": {},
    }
    problems = lint_labels(ir)
    assert any("nested/doubled parentheses" in p and "attn" in p for p in problems)
    assert any("raw backend" in p and "activation" in p for p in problems)
    assert any("dimensions/counts" in p and "clip" in p for p in problems)
    assert any("backend ops" in p and "patch" in p for p in problems)
    assert any("dimensions/counts" in p and "encoder" in p for p in problems)
    assert any("Tier-2 connector" in p and "bad_add" in p for p in problems)
    # the clean siblings are NOT flagged
    assert not any("ok_attn" in p or "ok_act" in p or "ok_conv" in p for p in problems)


def test_numeric_lint_separates_dimensions_from_topology_descriptors():
    """A *dimension* (768-d / 1024d) is a fact that belongs on a chip and must
    flag; a bare single-digit + D (2D / 3D / axial 3D) is an N-dimensional
    TOPOLOGY descriptor (3D-RoPE, 2D patch grid) — operation identity, not a
    channel count — and must NOT flag.  This is the false-positive that would
    otherwise reject an honest ``3D RoPE`` attention label."""
    from model_unfolder.lint import _leaks_numeric_fact

    # dimensions / counts -> flag
    for fact in ("768-d", "1024d", "1,280-d", "768d", "8-d", "12 heads", "Encoder ×30"):
        assert _leaks_numeric_fact(fact), f"{fact!r} is a numeric fact and must flag"
    # qualitative N-dimensional topology -> do NOT flag
    for topo in ("3D", "2D", "axial 3D", "3D RoPE", "2D patch", "3D-RoPE"):
        assert not _leaks_numeric_fact(topo), f"{topo!r} is a topology descriptor, not a dimension"

    # …and the same at the label-lint level: a 3D-RoPE attention block is clean.
    ir = {
        "layers": [{"blocks": [
            {"id": "rope", "kind": "attention", "label": "Attention (3D RoPE)"},
            {"id": "patch", "kind": "linear", "label": "2D Patchify"},
            {"id": "dim", "kind": "embedding", "label": "Embedding 768-d"},
        ]}],
        "extras": {},
    }
    problems = lint_labels(ir)
    assert not any("rope" in p or "patch" in p for p in problems)
    assert any("dimensions/counts" in p and "dim" in p for p in problems)


def test_config_access_capture_survives_nested_reset_and_reports_dotted_paths():
    """Sable's outer audit cannot be erased by a nested component parser."""
    from model_unfolder.adapters.transformer import debug

    cfg = {
        "model_type": "outer",
        "vision_config": {"hidden_size": 128, "new_architecture_switch": True},
        "torch_dtype": "float16",  # intentionally ignored vocabulary
    }
    with debug.capture_accesses() as touched:
        debug.note_access("model_type")
        debug.note_access("vision_config")
        debug.reset()  # a nested parser's legacy reset must not erase capture
        debug.note_access("hidden_size")
    assert "model_type" in touched and "hidden_size" in touched
    assert debug.unparsed_fields([cfg], touched=touched, recursive=True) == [
        "vision_config.new_architecture_switch"
    ]


def test_config_field_audit_is_visible_but_staged_non_blocking():
    """Unread architecture switches must be reported without breaking every
    existing blessed model before the ownership backlog has been worked down."""
    cfg = {**LLAMA, "brand_new_architecture_switch": True}
    report = sable(cfg, render_images=False)
    audit = next(c for c in report.checks if c.name == "config_field_audit")
    assert audit.blocking is False
    assert any("brand_new_architecture_switch" in finding for finding in audit.findings)
    assert report.mechanical_passed


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
# fact-conformance — the SAME-op-kind, different-SEMANTICS axis (positional
# scheme, attention algorithm) that op-presence conformance is blind to.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,cfg", [("FLUX", FLUX), ("PIXART", PIXART)])
def test_fact_conformance_clean_on_corpus(name, cfg):
    """FLUX (axial RoPE, softmax) and PixArt (learned-pos, softmax) make no
    fabricated-NoPE or wrong-attention claim — fact-conformance is clean."""
    ir = mu.unfold(cfg).to_ir()
    assert [p.message for p in check_fact_conformance(cfg, ir)] == []


def test_fact_conformance_flags_fabricated_nope():
    """NEGATIVE CONTROL: a block whose forward() applies rotary (FLUX threads
    image_rotary_emb) but whose diagram asserts NoPE must fire. This is the
    recurring fabricated-NoPE class (Wan/CogVideoX/Mochi/LTX/Lumina2)."""
    ir = mu.unfold(FLUX).to_ir()
    for L in ir["layers"]:                       # strip the positional scheme
        att = L.get("attention") or {}
        att["no_rope"], att["rope"], att["rope_dim"] = True, False, None
    probs = check_fact_conformance(FLUX, ir)
    assert any(p.kind == "missing_position" for p in probs), [p.message for p in probs]


def test_fact_conformance_is_symmetric_for_fabricated_rope_and_missing_learned_position():
    """StarCoder's old drawing must fail in BOTH directions: invented RoPE and
    omitted learned-position addition, using one typed source decision."""
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("gpt_bigcode").to_dict()
    ir = mu.unfold(cfg).to_ir()
    ir["extras"].pop("position_encoding", None)
    for layer in ir["layers"]:
        attn = layer["attention"]
        attn["rope"] = True
        attn["position_kind"] = "rope"
        attn["position_application"] = "qk_rotation"
    probs = check_fact_conformance(cfg, ir)
    assert any(p.kind == "fabricated_position" and p.op == "rope" for p in probs)
    assert any(p.kind == "missing_position" and p.op == "learned_absolute" for p in probs)


def test_fact_conformance_flags_one_non_rope_scheme_substituted_for_another():
    """Wrong alternatives must fail even though both correctly omit RoPE."""
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("bloom").to_dict()
    ir = mu.unfold(cfg).to_ir()
    for layer in ir["layers"]:
        layer["attention"]["position_kind"] = "learned_absolute"
        layer["attention"]["position_application"] = "embedding_add"
    probs = check_fact_conformance(cfg, ir)
    assert any(p.kind == "fabricated_position" and p.op == "learned_absolute" for p in probs)
    assert any(p.kind == "missing_position" and p.op == "alibi" for p in probs)


def test_present_but_ambiguous_position_is_blocking_unresolved(monkeypatch):
    from model_unfolder.evidence.models import PositionalEvidence
    from model_unfolder.evidence import position as position_module

    monkeypatch.setattr(
        position_module, "decoder_positional_evidence",
        lambda *a, **k: PositionalEvidence("ambiguous", reason="negative control"),
    )
    probs = check_fact_conformance(LLAMA, mu.unfold(LLAMA).to_ir())
    assert any(p.kind == "unresolved" and p.op == "position" for p in probs)


def test_true_oracle_missing_remains_visible_in_sable_report():
    cfg = {
        "model_type": "definitely_uninstalled_decoder",
        "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 1, "num_attention_heads": 4,
    }
    report = sable(cfg, render_images=False)
    assert report.oracle.startswith("MISSING")


def test_fact_conformance_flags_wrong_attention_kind():
    """NEGATIVE CONTROL: a diagram that draws LINEAR attention for a block whose
    code uses softmax (FLUX has no *LinearAttn* processor) must fire — the inverse
    of the Sana miss (softmax drawn for a linear-attention block)."""
    ir = mu.unfold(FLUX).to_ir()
    for L in ir["layers"]:
        (L.get("attention") or {})["kind"] = "linear"
    probs = check_fact_conformance(FLUX, ir)
    assert any(p.kind == "wrong_attention" for p in probs), [p.message for p in probs]


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
    """Every blessed model retains its SVG lock.  Old blessings newly invalidated
    by exact source attribution stay pinned as explicit unresolved debt; they are
    not silently re-blessed without a fresh Dable review."""
    expected_unresolved = {
        "stable-diffusion-xl-base-1-0.json": {
            "unet2dcondition/attn", "unet2dcondition/ffn",
        },
    }
    corpus = load_corpus()
    if not corpus:
        pytest.skip("no blessed models in tests/sable_corpus/ yet")
    for filename, fixture in corpus:
        drift = check_regression(fixture)
        expected = expected_unresolved.get(filename, set())
        actual_expected = {
            view for view in expected
            if any(item.startswith(f"nested_conformance: {view}: no code unit resolved")
                   for item in drift)
        }
        unexpected = [item for item in drift
                      if not any(item.startswith(
                          f"nested_conformance: {view}: no code unit resolved"
                      ) for view in expected)]
        assert actual_expected == expected, f"{filename} lost pinned unresolved coverage: {drift}"
        assert unexpected == [], f"{filename} regressed:\n  " + "\n  ".join(unexpected)
