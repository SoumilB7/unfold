"""Op-conformance: the diagram's op-set must match the model's HF forward() code.

The net for the class of bug where the picture is internally perfect (coupling /
wiring / unique-ids all green) yet diverges from what the code actually does —
e.g. Flux's single-stream block once drawn as a parallel-sum (no ``concat`` /
``gate_mul``) when ``FluxSingleTransformerBlock.forward`` does
``torch.cat([attn, mlp]) -> proj_out -> gate* -> residual+``.

Three layers here:
  * the EXTRACTOR (code side) — what does a forward() do;
  * the DIFF (both directions) over the offline corpus — does the picture match;
  * the NEGATIVE CONTROL — the old buggy rendering MUST be caught.
"""
from __future__ import annotations

import pytest

import model_unfolder as mu
from model_unfolder.evidence import check_model_conformance, extract_forward_ops
from model_unfolder.evidence.conformance import diff_conformance, resolve_view_code
from model_unfolder.evidence.sources import resolve_source_files
from model_unfolder.everchanging import load_conformance_abstractions, load_conformance_map

from tests.test_diffusion import FLUX, PIXART
from tests import test_coverage as tc


def _flux_forward_ops():
    bundle = resolve_source_files(FLUX, source="local")
    if not bundle.files:
        pytest.skip("diffusers Flux modeling source not installed locally")
    return extract_forward_ops(bundle.files)


# --------------------------------------------------------------------------
# Stage 1 — the extractor (code side)
# --------------------------------------------------------------------------

def test_extractor_finds_flux_single_stream_fused_topology():
    """The single-stream block's forward fuses attn∥mlp via a concat + an AdaLN
    gate — the exact ops a parallel-sum drawing would be MISSING."""
    fo = _flux_forward_ops().get("FluxSingleTransformerBlock")
    assert fo is not None, "FluxSingleTransformerBlock not found in Flux source"
    assert "concat" in fo.op_kinds and "gate_mul" in fo.op_kinds, fo.op_kinds
    assert {"attention", "linear", "norm", "residual_add", "activation"} <= fo.op_kinds, fo.op_kinds
    assert fo.field_types.get("attn", "").endswith("Attention")
    assert fo.field_types.get("proj_out") == "Linear"


def test_extractor_distinguishes_dual_stream_block():
    """The dual-stream block is a DIFFERENT topology: sequential attn then a real
    FeedForward (ffn), AdaLN-gated — not the single-stream concat fusion."""
    fo = _flux_forward_ops().get("FluxTransformerBlock")
    assert fo is not None
    assert {"attention", "ffn", "gate_mul", "norm", "residual_add"} <= fo.op_kinds, fo.op_kinds
    assert any(v == "FeedForward" for v in fo.field_types.values())


# --------------------------------------------------------------------------
# Stage 2 — the FLUX regression + the negative control
# --------------------------------------------------------------------------

def test_flux_conformance_clean_both_directions():
    """The corrected Flux renders both blocks faithfully — zero conformance gaps."""
    ir = mu.unfold(FLUX).to_ir()
    problems = check_model_conformance(FLUX, ir)
    real = [p for p in problems if p.kind in ("missing", "fabricated", "stale")]
    assert real == [], "\n".join(p.message for p in real)


def test_negative_control_parallel_sum_rendering_is_caught():
    """THE pin: a GPT-J parallel-sum single-stream rendering (no concat, no gate)
    MUST fail the diff with both ops flagged missing — citing the forward()."""
    code = _flux_forward_ops()["FluxSingleTransformerBlock"]
    ab = load_conformance_abstractions()
    buggy_diagram = frozenset({"norm", "attention", "ffn", "residual_add"})  # the old wrong picture
    problems = diff_conformance(buggy_diagram, code, "flux", "single_stream", ab)
    missing = {p.op for p in problems if p.kind == "missing"}
    assert {"concat", "gate_mul"} <= missing, [p.message for p in problems]
    assert any("transformer_flux" in p.source_file for p in problems if p.kind == "missing")


def test_negative_control_end_to_end_pipeline_catches_buggy_render():
    """The FULL path (parser → IR → conformance) catches the bug: mutate Flux's
    single-stream group back to the buggy parallel-sum (no concat/gate) and the
    net flags both — classified by the parser's variant tag, so the mis-render
    can't dodge the check by looking like a plain block."""
    ir = mu.unfold(FLUX).to_ir()
    mutated = False
    for layer in ir["layers"]:
        if "concat" in {b.get("kind") for b in (layer.get("blocks") or [])}:
            layer["blocks"] = [{"id": "rms1", "kind": "norm"}, {"id": "attn", "kind": "attention"},
                               {"id": "ffn", "kind": "ffn"}, {"id": "add1", "kind": "residual_add"}]
            mutated = True
    assert mutated, "no single-stream group to mutate — Flux fixture changed?"
    missing = {p.op for p in check_model_conformance(FLUX, ir) if p.kind == "missing"}
    assert {"concat", "gate_mul"} <= missing, missing


# --------------------------------------------------------------------------
# Stage 3 — the corpus net + resolver honesty + staleness
# --------------------------------------------------------------------------

def test_op_conformance_both_directions_over_corpus():
    """Across the offline archetype corpus, no view's diagram diverges from its
    forward() code (missing / fabricated / stale). Unresolved views (a family
    whose source isn't installed) are gaps, not failures — see the honesty test."""
    failures: list[str] = []
    for name, cfg in tc.CORPUS.items():
        ir = mu.unfold(cfg).to_ir()
        for p in check_model_conformance(cfg, ir):
            if p.kind in ("missing", "fabricated", "stale"):
                failures.append(f"{name}: {p.message}")
    assert not failures, "op-conformance gaps:\n  " + "\n  ".join(failures)


def test_resolver_binds_the_diffusion_block_views():
    """The net can't silently no-op on the hero cases: Flux's TWO block views and
    PixArt's block view MUST resolve to a real forward() to diff against."""
    flux_ops = _flux_forward_ops()
    cmap = load_conformance_map()
    single = resolve_view_code("flux", "single_stream", {}, flux_ops, cmap)
    dual = resolve_view_code("flux", "block", {}, flux_ops, cmap)
    assert single is not None and single.class_name == "FluxSingleTransformerBlock"
    assert dual is not None and dual.class_name == "FluxTransformerBlock"
    # PixArt's block class lives in models/attention.py — resolved via file augmentation.
    pix_problems = check_model_conformance(PIXART, mu.unfold(PIXART).to_ir())
    assert not [p for p in pix_problems if p.kind == "unresolved"], \
        [p.view for p in pix_problems if p.kind == "unresolved"]


def test_conformance_citations_not_stale():
    """Every `since` citation token still appears in its cited forward() — so a
    silent upstream rename can't rot the allow-list."""
    ir = mu.unfold(FLUX).to_ir()
    stale = [p.message for p in check_model_conformance(FLUX, ir) if p.kind == "stale"]
    assert not stale, stale


# --------------------------------------------------------------------------
# Stage 4 — render every variant (no silent dominant-only collapse)
# --------------------------------------------------------------------------

def test_heterogeneous_denoiser_renders_every_variant():
    """A multi-block-type denoiser (Flux: dual-stream + single-stream) must render
    EVERY variant's architecture, not collapse to the dominant — so non-dominant
    blocks are drillable and enter the image surface. Pins Fix 4 / the invisibility
    root cause."""
    from model_unfolder.renderers.html.metadata import _make_info
    ir = mu.unfold(FLUX).to_ir()
    n_groups = len(_make_info(ir)["groups"])
    assert n_groups >= 2, f"expected Flux dual+single groups, got {n_groups}"
    html = mu.unfold(FLUX).to_html(standalone=True)
    n_arch = html.count('class="uf-arch-variant uf-arch-variant-')
    assert n_arch >= n_groups, (
        f"{n_arch} architecture variants rendered for {n_groups} block-type groups "
        "— a non-dominant variant collapsed (invisible).")
