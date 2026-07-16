"""U2 projection receipts — the render half of the fact/projection contract.

Commit 1 pilot: the source-bound vision/video projector output width.  The
drill that draws the width emits a typed receipt; Net 2 joins it to the
config-consumption obligation; coverage is owner/mechanism-scoped; and no
pixel changes (a receipt rides the render event, never the SVG).
"""
from __future__ import annotations

import json
import pathlib

import pytest

import model_unfolder as mu
from model_unfolder.evidence.receipts import (
    ProjectionReceipt, RECEIPTED_SCOPES, join_obligation_receipts,
    fabrication_findings, value_status_hash,
)
from model_unfolder.sable import sable

_CORPUS = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"


def _qwen2vl():
    return json.loads((_CORPUS / "qwen2-vl-7b-instruct.json").read_text())["config"]


def _render_receipts(cfg):
    from model_unfolder.parser import config_to_ir
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.diagram import Diagram
    from model_unfolder.renderers.html.render_context import (
        RenderContext, activate_render_context)

    ctx = ParseContext.build(cfg, source="local")
    diagram = Diagram(config_to_ir(cfg, parse_context=ctx))
    rc = RenderContext(theme="teal")
    with activate_render_context(rc):
        diagram.to_html(standalone=True)
    return [r for e in rc.events for r in getattr(e, "receipts", ()) or ()]


def test_projector_width_receipt_comes_from_the_drill_that_draws_it():
    """The pilot: the vision and video projector op drills emit a receipt for
    <owner>.projector_out_features — the surface that puts the width on the
    page, not a parse-time assertion."""
    receipts = {(r.owner, r.mechanism): r for r in _render_receipts(_qwen2vl())}
    for owner in ("root.vision", "root.video"):
        r = receipts[(owner, "projector_out_width")]
        assert r.fact_key == f"{owner}.projector_out_features"
        assert r.surface.startswith("ops_") and "projector" in r.surface
        assert r.projection_kind == "op"
        assert r.value_status_hash == value_status_hash(3584, "config_bound")


def test_net2_covered_scope_passes_and_is_blocking():
    """Net 2 joins occurrence -> target -> receipt: the covered projector scope
    has matching receipts, so the blocking check passes."""
    rep = sable(_qwen2vl(), render_images=False)
    net2 = next(c for c in rep.checks if c.name == "config_consumed_unreceipted")
    assert net2.blocking is True
    assert net2.passed and net2.findings == []


def test_coverage_is_scoped_and_the_global_bool_is_gone():
    ca = mu.unfold(_qwen2vl()).to_ir()["extras"]["config_access"]
    assert "projection_receipts_available" not in ca
    scopes = {tuple(s) for s in ca["projection_coverage"]["receipted_scopes"]}
    assert ("root.vision", "projector_out_width") in scopes
    assert ("root.video", "projector_out_width") in scopes


def test_receipt_fabrication_net_is_clean_and_would_fire():
    """Reverse-fabrication: real receipts reference a declared claim target and
    pass; a receipt for an unregistered target fires."""
    rep = sable(_qwen2vl(), render_images=False)
    fab = next(c for c in rep.checks if c.name == "receipt_fabrication")
    assert fab.blocking is True and fab.passed and fab.findings == []

    ghost = ProjectionReceipt(
        fact_key="root.vision.never_registered_fact", owner="root.vision",
        mechanism="projector_out_width", surface="ops_vision_projector",
        node_path=("vision_projector",), projection_kind="op",
        value_status_hash="deadbeef")
    findings = fabrication_findings(
        [ghost], registered_keys=set(), claimed_targets=set(), debt_keys=set())
    assert findings and "nothing behind it" in findings[0]


def test_unrelated_obligations_never_become_blocking():
    """Enabling projector receipts must not make a text/LM obligation blocking:
    an obligation outside a receipted scope produces no finding even with an
    empty receipt set."""
    ca = mu.unfold(_qwen2vl()).to_ir()["extras"]["config_access"]
    obligations = ca["projection_obligations"]
    text_obs = [o for o in obligations
                if (o["target"]["owner"], o["mechanism"]) not in RECEIPTED_SCOPES]
    assert text_obs, "the witness must carry un-migrated obligations"
    result = join_obligation_receipts(obligations, [], RECEIPTED_SCOPES)
    # every finding names a RECEIPTED scope target; no text/LM obligation appears
    for f in result["findings"]:
        assert any(f"{o}/projector_out_width" in f
                   for o in ("root.vision", "root.video"))


def _projector_consumptions(cfg):
    from model_unfolder.evidence.config_access import capture_events, owner_scope
    with capture_events() as led:
        with owner_scope("root"):
            mu.unfold(cfg).to_ir()
    return sorted({(e.component, e.config_path) for e in led.events
                   if e.fact_key == "projector_out_features"
                   and e.intent == "consumed"})


def test_negative_control_flux_owns_its_encoder_vision_not_root_vision():
    """NEGATIVE control (producer fix): flux's text encoder is mistral3, a real
    VLM.  Its vision projector is OWNED by root.text_encoder.vision — never
    falsely attributed to the pipeline's top-level root.vision.  The false
    producer (a context-less second sub-parse) is deleted, so no root.vision
    consumption exists and Net 2 is clean without any render-drawn workaround."""
    fx = json.loads((_CORPUS / "flux-2-dev.json").read_text())["config"]
    consumptions = _projector_consumptions(fx)
    owners = {owner for owner, _ in consumptions}
    assert "root.vision" not in owners, consumptions
    assert "root.text_encoder.vision" in owners, consumptions
    rep = sable(fx, render_images=False)
    net2 = next(c for c in rep.checks if c.name == "config_consumed_unreceipted")
    assert net2.passed and net2.findings == []
    # the advisory phantom check is GONE — a known false consumption is a bug
    # fixed at the producer, not permanent advisory debt.
    assert all(c.name != "config_phantom_consumption" for c in rep.checks)


def test_positive_control_qwen2vl_owns_top_level_root_vision():
    """POSITIVE control: a genuine top-level VLM keeps root.vision ownership,
    is drawn, and its receipt satisfies the unconditionally-blocking Net 2."""
    consumptions = _projector_consumptions(_qwen2vl())
    assert ("root.vision", "vision_config.hidden_size") in consumptions
    assert not any(owner.startswith("root.text_encoder")
                   for owner, _ in consumptions)


def test_net2_is_unconditionally_blocking_in_a_receipted_scope():
    """A source-proven consumption in a receipted scope owes a receipt
    UNCONDITIONALLY — absence of render output is never proof of
    non-applicability.  Dropping qwen2-vl's receipts must block."""
    from model_unfolder.evidence import receipts as receipts_mod

    real = receipts_mod.join_obligation_receipts

    def blind(obligations, receipts, receipted_scopes=receipts_mod.RECEIPTED_SCOPES):
        return real(obligations, [], receipted_scopes)   # render drew nothing

    import pytest
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(receipts_mod, "join_obligation_receipts", blind)
        rep = sable(_qwen2vl(), render_images=False)
    finally:
        monkeypatch.undo()
    net2 = next(c for c in rep.checks if c.name == "config_consumed_unreceipted")
    assert net2.blocking and not net2.passed and net2.findings


def test_namespace_predicate_is_ownership_driven_not_model_driven():
    """The fix is a reusable OWNERSHIP predicate: the SAME VLM config, parsed
    under a sub-component namespace, owns its vision under that namespace.
    Proven by parsing qwen2-vl's config through a ParseContext whose
    component_namespace is a sub-slot — its vision projector consumption is
    owned by <namespace>.vision, and never collides with the top-level
    root.vision a real top-level parse would produce."""
    from model_unfolder.parser import config_to_ir
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.config_access import capture_events, owner_scope

    cfg = _qwen2vl()
    ctx = ParseContext.build(cfg, source="local")
    ctx.component_namespace = "root.text_encoder_2"      # parsed as a sub-slot
    with capture_events() as led:
        with owner_scope("root"):
            config_to_ir(cfg, parse_context=ctx)
    owners = {e.component for e in led.events
              if e.fact_key == "projector_out_features" and e.intent == "consumed"}
    assert owners == {"root.text_encoder_2.vision", "root.text_encoder_2.video"}, owners
    assert "root.vision" not in owners


def test_no_visual_delta_projector_views_unchanged():
    """A receipt rides the render event, never the SVG: the vision and video
    projector view hashes are identical to their blessed manifest values."""
    from test_support.preservation import _view_hashes

    manifest = json.loads(
        (pathlib.Path(mu.__file__).parent.parent / "tests"
         / "preservation_expected_manifest.json").read_text())["witnesses"]
    cfg = _qwen2vl()
    views = _view_hashes(cfg)
    expected = manifest["qwen2-vl-7b-instruct"]["views"]
    drifted = [v for v in sorted(set(views) | set(expected))
               if views.get(v) != expected.get(v)]
    assert drifted == [], f"projector receipt changed pixels: {drifted}"
