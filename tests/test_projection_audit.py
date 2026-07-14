"""U2 P4 — the two permanent evidence nets.

net #13 (projection-audit): every code/config-proven structural fact on a
drawable family must have a DRAWN witness (a ``RenderEvent.facts_projected``
entry) — a fact read from the modeling source but projected nowhere is the
granite-score-multiplier class.

net #14 (zero-asserted census): a parse with no code and a numbers-only config
must fall to honest-unknown for every family; only a doctrine-allowed set may
still carry a generic default.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import model_unfolder as mu
from model_unfolder import sable
from model_unfolder.sable import (
    _projection_audit_findings,
    _zero_asserted_census_findings,
    _numbers_only,
    _accessed_unprojected_findings,
    _CENSUS_ALLOWED,
    _PROJECTION_AUDIT_BLOCKING,
)
from test_support import LLAMA, FLUX


def _event(facts):
    return SimpleNamespace(facts_projected=frozenset(facts))


def _ir(fact_provenance):
    return {"extras": {"fact_provenance": fact_provenance}}


# --------------------------------------------------------------------------- #
# net #13 — projection-audit
# --------------------------------------------------------------------------- #

def test_projection_audit_flags_a_proven_fact_that_no_surface_draws():
    """The core class: a fact proven from evidence but projected NOWHERE."""
    ir = _ir({"decoder.attention.scores_scale": {"value": "x", "status": "code_proven"}})
    findings = _projection_audit_findings(ir, render_log=[])
    assert len(findings) == 1
    assert "decoder.attention.scores_scale" in findings[0]


def test_projection_audit_passes_when_a_render_event_witnesses_the_fact():
    ir = _ir({"decoder.attention.scores_scale": {"value": "x", "status": "code_proven"}})
    log = [_event({"decoder.attention.scores_scale"})]
    assert _projection_audit_findings(ir, log) == []


@pytest.mark.parametrize("status", ["unknown", "asserted", "oracle_missing", "ambiguous", "derived"])
def test_projection_audit_exempts_non_evidenced_statuses(status):
    """unknown / oracle_missing / ambiguous render pale-honest; asserted is the
    census net's target; derived is computed — none owe a drawn witness."""
    ir = _ir({"decoder.attention.scores_scale": {"value": "x", "status": status}})
    assert _projection_audit_findings(ir, render_log=[]) == []


def test_projection_audit_exempts_non_drawable_families():
    """A proven fact whose owner family has no v1 render surface is not owed a
    witness yet (the net scopes to attention/ffn/layer/model)."""
    ir = _ir({"embeddings.scale.value": {"value": 1, "status": "code_proven"}})
    assert _projection_audit_findings(ir, render_log=[]) == []


def test_projection_audit_union_is_across_all_events():
    """The witness may come from ANY render event — the audit unions them."""
    ir = _ir({
        "decoder.attention.scores_scale": {"value": "x", "status": "code_proven"},
        "decoder.layer.norm_kind": {"value": "rmsnorm", "status": "code_proven"},
    })
    log = [_event({"decoder.attention.scores_scale"}), _event({"decoder.layer.norm_kind"})]
    assert _projection_audit_findings(ir, log) == []


def test_projection_audit_is_wired_into_sable_as_blocking_and_clean():
    """Wired like config_field_audit / evidence_ambiguity: present, blocking, and
    green on a real decoder (every proven fact has a drawn witness)."""
    report = sable(LLAMA, render_images=False)
    check = next(c for c in report.checks if c.name == "projection_audit")
    assert check.blocking is _PROJECTION_AUDIT_BLOCKING is True
    assert check.passed, check.findings


def test_render_emits_facts_projected_for_the_proven_families():
    """End-to-end: a real render stamps the ledger keys onto render events —
    attention facts on the attention drill, layer/model facts on the
    architecture view.  Proves the emission, not just the audit logic."""
    from model_unfolder.parser import config_to_ir, _coerce
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.diagram import Diagram
    from model_unfolder.renderers.html.render_context import (
        RenderContext, activate_render_context,
    )

    cfg = _coerce(LLAMA)
    ctx = ParseContext.build(cfg, source="local")
    diagram = Diagram(config_to_ir(cfg, parse_context=ctx))
    rc = RenderContext(theme="teal")
    with activate_render_context(rc):
        diagram.to_html(standalone=True)
    projected = set()
    for e in rc.events:
        projected |= set(e.facts_projected)
    assert "decoder.attention.scores_scale" in projected      # the attention drill
    assert "decoder.ffn.activation" in projected              # the FFN drill
    assert "decoder.layer.norm_kind" in projected             # the architecture view
    assert "model.tie_word_embeddings" in projected           # the architecture view
    # The architecture facts-only event carries no drawn ops (nested-conformance
    # keys on drill roles and skips it).
    arch = [e for e in rc.events if e.view == "architecture"]
    assert arch and all(not e.drawn_ops for e in arch)


# --------------------------------------------------------------------------- #
# net #14 — zero-asserted census
# --------------------------------------------------------------------------- #

def test_numbers_only_keeps_address_and_numbers_strips_declarations():
    stripped = _numbers_only({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096, "rope_theta": 10000.0,
        "hidden_act": "silu", "tie_word_embeddings": False,
        "rope_scaling": {"type": "linear"}, "some_list": [1, 2],
    })
    assert stripped["model_type"] == "llama"
    assert stripped["architectures"] == ["LlamaForCausalLM"]   # address survives verbatim
    assert stripped["hidden_size"] == 4096                     # numeric value survives
    assert stripped["rope_theta"] == 10000.0
    assert "hidden_act" not in stripped                        # string declaration stripped
    assert "tie_word_embeddings" not in stripped               # bool declaration stripped
    assert "rope_scaling" not in stripped and "some_list" not in stripped


def test_census_clean_on_a_real_decoder():
    """Post-P1 a llama's zero-evidence parse asserts only doctrine-allowed
    defaults (scores_scale / ffn_storage / projection_mode)."""
    assert _zero_asserted_census_findings(LLAMA, "local") == []


def test_census_skips_a_model_it_cannot_parse_from_numbers_only():
    """A pipeline / wrapper config whose transformer lives in a stripped nested
    dict cannot be reconstructed from numbers-only — it asserts nothing, so the
    net skips it instead of crashing the pass."""
    assert _zero_asserted_census_findings(FLUX, "local") == []


def test_census_flags_a_disallowed_zero_evidence_default(monkeypatch):
    """NEGATIVE CONTROL: if a zero-evidence parse asserted a fact OUTSIDE the
    doctrine-allowed set (here a fabricated ``mask`` default), the census fires.
    Injected at the re-parse so the gate itself is exercised."""
    import model_unfolder.parser as parser_mod

    def fake_config_to_ir(cfg, *a, parse_context=None, **kw):
        parse_context.facts.record("decoder.attention", "mask", "causal", "asserted")
        parse_context.facts.record("decoder.ffn", "ffn_storage", "cat", "asserted")  # allowed
        return SimpleNamespace(extras={})

    monkeypatch.setattr(parser_mod, "config_to_ir", fake_config_to_ir)
    findings = _zero_asserted_census_findings(LLAMA, "local")
    # Only the disallowed 'mask' is flagged; the allowed 'ffn_storage' assertion
    # is not (len == 1 proves it — the allowed set is merely NAMED in the advice).
    assert len(findings) == 1
    assert "asserts 'mask'" in findings[0]


def test_census_is_wired_into_sable_as_blocking():
    report = sable(LLAMA, render_images=False)
    check = next(c for c in report.checks if c.name == "zero_asserted_census")
    assert check.blocking is True
    assert check.passed, check.findings


def test_census_allowed_set_is_the_three_presentation_conventions():
    assert _CENSUS_ALLOWED == {"scores_scale", "ffn_storage", "projection_mode"}


# --------------------------------------------------------------------------- #
# config_field_audit upgrade — accessed-but-unprojected (advisory)
# --------------------------------------------------------------------------- #

def test_accessed_unprojected_is_inert_without_a_consumed_census():
    """H3 (§16.5): the net reads the OWNER-SCOPED ``config_access`` ledger, whose
    ``accessed_unconsumed`` is already gated at parse time to owners that have a
    consumed census.  An extras block with no such ledger entry (an adapter/owner
    on inspected-only reads) is inert."""
    ir = {"extras": {"config_access": {"accessed_unconsumed": []}}}
    assert _accessed_unprojected_findings(ir) == []
    # and a legacy extras block with no config_access at all is likewise inert
    assert _accessed_unprojected_findings({"extras": {"config_audit": {}}}) == []


def test_real_transformer_parse_publishes_a_consumed_census():
    """H3 Phase B activation (§11 step 4): the geometry/embedding family is
    migrated to ``consume()``, so a real transformer parse now publishes a
    non-empty ``config_consumed`` — the net is no longer inert in production."""
    ir = mu.unfold(LLAMA).to_ir()
    consumed = (ir.get("extras") or {}).get("config_consumed") or []
    assert "hidden_size" in consumed and "num_hidden_layers" in consumed


def test_accessed_unprojected_fires_once_a_consumed_census_exists():
    """When the owner-scoped ledger surfaces an accessed-but-unconsumed field,
    the net reports it — OWNER-QUALIFIED (``owner:field``), so a sibling's
    consumption of the same leaf key does not clear it (the granite-multiplier
    class, now free of the flat-global collision)."""
    ir = {"extras": {"config_access": {
        "accessed_unconsumed": ["root:sliding_window", "root.vision:hidden_size"],
    }}}
    findings = _accessed_unprojected_findings(ir)
    assert len(findings) == 2
    assert any("root:sliding_window" in f for f in findings)
    # the vision hidden_size is flagged for vision even though text consumed one
    assert any("root.vision:hidden_size" in f for f in findings)


def test_accessed_unprojected_is_wired_advisory_not_blocking():
    report = sable(LLAMA, render_images=False)
    check = next(c for c in report.checks if c.name == "config_accessed_unprojected")
    assert check.blocking is False


# --------------------------------------------------------------------------- #
# COR-5 (§10) — migration claims: Net 1 blocks claimed exact scopes; poisons
# prove a violated, bare-funnel, or fabricated-receipt state cannot pass.
# --------------------------------------------------------------------------- #

def test_cor5_migration_claim_constructor_rejects_empty_declarations():
    from model_unfolder.evidence.registry import MigrationClaim

    with pytest.raises(ValueError):
        MigrationClaim("root.vision", "projector_out_width", "COR-4", ())
    with pytest.raises(ValueError):
        MigrationClaim("", "projector_out_width", "COR-4", ("a.b",))
    with pytest.raises(ValueError):
        MigrationClaim("root.vision", "", "COR-4", ("a.b",))
    with pytest.raises(ValueError):
        MigrationClaim("root.vision", "projector_out_width", "COR-4", ("a.b", ""))


def test_cor5_poison_claimed_scope_with_unconsumed_read_blocks(monkeypatch):
    """POISON: a claim over a path the parse reads without consuming MUST fire
    — proves the blocking net cannot be vacuously green."""
    from model_unfolder.evidence import registry as reg

    poisoned = (reg.MigrationClaim(
        "root", "norm_epsilon", "POISON",
        ("rms_norm_eps",),
    ),)
    monkeypatch.setattr(reg, "MIGRATED_SCOPES", poisoned)
    rep = sable(LLAMA, render_images=False)
    check = next(c for c in rep.checks if c.name == "config_migration_claims")
    assert check.blocking
    assert not check.passed
    assert any("rms_norm_eps" in f for f in check.findings)
    assert not rep.mechanical_passed


def test_cor5_poison_bare_funnel_read_in_claimed_scope_is_a_violation(monkeypatch):
    """A claimed scope may not read through the bare funnel: llama's
    ``rms_norm_eps`` read is path-inexact today, so the SAME poison must name
    the inexact-read law (not only unconsumed-ness)."""
    from model_unfolder.evidence import registry as reg

    poisoned = (reg.MigrationClaim(
        "root", "norm_epsilon", "POISON", ("rms_norm_eps",)),)
    monkeypatch.setattr(reg, "MIGRATED_SCOPES", poisoned)
    ir = mu.unfold(LLAMA).to_ir()
    rows = ir["extras"]["config_access"]["migration_claims"]
    assert len(rows) == 1
    assert rows[0]["observed_events"] >= 1
    assert any("inexact read" in v or "neither consumed" in v
               for v in rows[0]["violations"])


def test_cor5_real_claim_is_earned_on_the_multimodal_witness():
    """The COR-4 projector-width claim holds on the source-present witness:
    events observed, zero violations, and the claimed path actually consumed."""
    import json
    import pathlib

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    cfg = json.loads((corpus / "qwen2-vl-7b-instruct.json").read_text())["config"]
    ir = mu.unfold(cfg).to_ir()
    ca = ir["extras"]["config_access"]
    rows = {row["scope"]: row for row in ca["migration_claims"]}
    for scope in ("root.vision/projector_out_width", "root.video/projector_out_width"):
        assert rows[scope]["observed_events"] > 0
        assert rows[scope]["violations"] == []
    assert "root.vision:hidden_size" in ca["consumed"]
    assert "root.video:hidden_size" in ca["consumed"]


def test_cor5_net2_blocks_exactly_when_receipts_are_declared_available(monkeypatch):
    """Net 2 is advisory while ``projection_receipts_available=False`` and
    BLOCKING the moment a parse claims receipts: claiming with unreceipted
    obligations standing must fail — completion cannot be claimed on an
    unavailable ledger."""
    rep = sable(LLAMA, render_images=False)
    net2 = next(c for c in rep.checks if c.name == "config_consumed_unprojected")
    assert net2.blocking is False
    assert "projection_receipts_unavailable" in (net2.note or "")

    from model_unfolder.evidence import context as ctx_mod

    real_build = ctx_mod.ParseContext.build

    def claiming_build(*args, **kwargs):
        built = real_build(*args, **kwargs)
        built.projection_receipts_available = True   # fabricated claim
        return built

    monkeypatch.setattr(ctx_mod.ParseContext, "build", claiming_build)
    rep2 = sable(LLAMA, render_images=False)
    net2b = next(c for c in rep2.checks if c.name == "config_consumed_unprojected")
    assert net2b.blocking is True
    assert net2b.findings, "unreceipted obligations must surface as findings"
    assert not net2b.passed
    assert not rep2.mechanical_passed
