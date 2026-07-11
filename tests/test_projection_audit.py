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
from tests.test_diffusion import LLAMA, FLUX


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
    """The consumed rail is not populated in production yet; without it the
    accessed-but-unprojected signal is not computable, so it stays silent."""
    ir = mu.unfold(LLAMA).to_ir()
    assert "config_consumed" not in (ir.get("extras") or {})
    assert _accessed_unprojected_findings(ir) == []


def test_accessed_unprojected_fires_once_a_consumed_census_exists():
    """When the consumed census IS available, a field accessed but not consumed
    into a spec is surfaced (the granite-multiplier class)."""
    ir = {"extras": {
        "config_consumed": ["hidden_size"],
        "config_audit": {"accessed": ["hidden_size", "sliding_window"]},
    }}
    findings = _accessed_unprojected_findings(ir)
    assert len(findings) == 1 and "sliding_window" in findings[0]


def test_accessed_unprojected_is_wired_advisory_not_blocking():
    report = sable(LLAMA, render_images=False)
    check = next(c for c in report.checks if c.name == "config_accessed_unprojected")
    assert check.blocking is False
