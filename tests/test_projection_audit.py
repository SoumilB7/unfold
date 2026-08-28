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
from model_unfolder.evidence import config_access as _config_access
from model_unfolder.renderers.html.fact_projection import layer_and_model_facts
from test_support import bind_document, LLAMA, FLUX


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


def test_diffusion_root_and_stack_facts_are_not_outside_the_drawn_net():
    """U10 facts live under denoiser/stack occurrence owners, not decoder.

    Treating those owner families as non-drawable would let an exact root
    topology, bookend route, or repetition count disappear while the blocking
    projection audit remained vacuously green.
    """
    facts = {
        "root.denoiser.diffusion_root_topology": {
            "value": "repeated_stack", "status": "code_proven"},
        "root.denoiser.diffusion_bookend_operations": {
            "value": {}, "status": "code_proven"},
        "root.denoiser.stacks[0].diffusion_stack_depth": {
            "value": 8, "status": "code_and_config"},
        "root.denoiser.stacks[0].diffusion_stack_variant": {
            "value": {"selected_branch": 1, "candidate_count": 2},
            "status": "class_default"},
    }
    ir = _ir(facts)
    assert set(layer_and_model_facts(ir)) == set(facts)
    assert len(_projection_audit_findings(ir, render_log=[])) == 4
    assert _projection_audit_findings(ir, [_event(facts)]) == []


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
    """A zero-evidence parse asserts no architectural convention."""
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
        return SimpleNamespace(extras={})

    monkeypatch.setattr(parser_mod, "config_to_ir", fake_config_to_ir)
    findings = _zero_asserted_census_findings(LLAMA, "local")
    assert len(findings) == 1
    assert "asserts 'mask'" in findings[0]


def test_census_is_wired_into_sable_as_blocking():
    report = sable(LLAMA, render_images=False)
    check = next(c for c in report.checks if c.name == "zero_asserted_census")
    assert check.blocking is True
    assert check.passed, check.findings


def test_census_has_no_allowed_architecture_convention():
    assert _CENSUS_ALLOWED == set()


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
    """When the owner-scoped ledger surfaces an accessed-but-unconsumed
    occurrence, the net reports it — OCCURRENCE-EXACT (fourth vet §10.3:
    component + exact path + actual spelling), so a sibling's consumption of
    the same leaf key does not clear it AND two paths sharing a canonical
    leaf stay two findings."""
    ir = {"extras": {"config_access": {
        "accessed_unconsumed": ["root:sliding_window", "root.vision:hidden_size"],
        "accessed_unconsumed_exact": [
            {"component": "root", "path": "sliding_window",
             "spelling": "sliding_window", "canonical": "sliding_window"},
            {"component": "root.vision", "path": "vision_config.hidden_size",
             "spelling": "hidden_size", "canonical": "hidden_size"},
            {"component": "root.vision", "path": "vision_config.sub.hidden_size",
             "spelling": "hidden_size", "canonical": "hidden_size"},
        ],
    }}}
    findings = _accessed_unprojected_findings(ir)
    assert len(findings) == 3          # exact rows, not the collapsed summary
    assert any("root:'sliding_window'" in f for f in findings)
    assert any("'vision_config.hidden_size'" in f for f in findings)
    assert any("'vision_config.sub.hidden_size'" in f for f in findings)


def test_accessed_unprojected_is_wired_advisory_not_blocking():
    report = sable(LLAMA, render_images=False)
    check = next(c for c in report.checks if c.name == "config_accessed_unprojected")
    assert check.blocking is False


# --------------------------------------------------------------------------- #
# COR-5 (§10) — migration claims: Net 1 blocks claimed exact scopes; poisons
# prove a violated, bare-funnel, or fabricated-receipt state cannot pass.
# --------------------------------------------------------------------------- #

def test_cor5_migration_claim_constructor_rejects_empty_declarations():
    from model_unfolder.evidence.config_access import ProjectionTarget
    from model_unfolder.evidence.registry import ClaimBinding, MigrationClaim

    target = ProjectionTarget("root.vision", "projector_out_features")
    binding = ClaimBinding("a.b", target)
    with pytest.raises(ValueError):
        MigrationClaim("root.vision", "projector_out_width", "COR-4", ())
    with pytest.raises(ValueError):
        MigrationClaim("", "projector_out_width", "COR-4", (binding,))
    with pytest.raises(ValueError):
        MigrationClaim("root.vision", "", "COR-4", (binding,))
    with pytest.raises(ValueError):
        ClaimBinding("", target)
    with pytest.raises(ValueError):
        ClaimBinding("a.b", ProjectionTarget("", "projector_out_features"))
    with pytest.raises(ValueError):
        ClaimBinding("a.b", ProjectionTarget("root.vision", ""))


def test_cor5_poison_claimed_scope_with_unconsumed_read_blocks(monkeypatch):
    """POISON: a claim over a path the parse reads without consuming MUST fire
    — proves the blocking net cannot be vacuously green."""
    from model_unfolder.evidence import registry as reg

    poisoned = (reg.MigrationClaim(
        "root", "norm_epsilon", "POISON",
        (reg.ClaimBinding("rms_norm_eps",
                          _config_access.ProjectionTarget("root", "norm_eps")),),
    ),)
    monkeypatch.setattr(reg, "MIGRATED_SCOPES", poisoned)
    rep = sable(LLAMA, render_images=False)
    check = next(c for c in rep.checks if c.name == "config_migration_claims")
    assert check.blocking
    assert not check.passed
    assert any("rms_norm_eps" in f for f in check.findings)
    assert not rep.mechanical_passed


def test_cor5_poison_unconsumed_read_in_claimed_scope_is_a_violation(monkeypatch):
    """A claimed scope may not contain a present read that no declared mechanism
    binding consumes.

    U2.2a note: this poison used to ride on llama's ``rms_norm_eps`` read being
    path-INEXACT, and accepted either law's message. That read is a true
    top-level field and is now honestly exact, so the inexact limb no longer
    applies here — and an ``or`` across two laws could never prove which one
    fired anyway. The two laws are now poisoned separately, each against a
    construction that cannot rot: this one on unconsumed-ness, the next on
    inexactness."""
    from model_unfolder.evidence import registry as reg

    poisoned = (reg.MigrationClaim(
        "root", "norm_epsilon", "POISON",
        (reg.ClaimBinding("rms_norm_eps",
                          _config_access.ProjectionTarget("root", "norm_eps")),),
    ),)
    monkeypatch.setattr(reg, "MIGRATED_SCOPES", poisoned)
    ir = mu.unfold(LLAMA).to_ir()
    rows = ir["extras"]["config_access"]["migration_claims"]
    assert len(rows) == 1
    assert rows[0]["observed_events"] >= 1
    assert rows[0]["target_matches"] == 0
    assert any("rms_norm_eps" in v and "not consumed" in v
               for v in rows[0]["violations"]), rows[0]["violations"]


def test_cor5_poison_bare_funnel_read_in_claimed_scope_is_a_violation():
    """A claimed scope may not read through the bare funnel.

    Built from a constructed inexact read rather than a model's incidental one:
    the law must hold for ANY claimed scope, and tying it to whichever witness
    happens to be unpathed today means the poison silently retires the moment
    that reader is fixed."""
    from model_unfolder.evidence.claims_audit import validate_claims
    from model_unfolder.evidence import registry as reg

    claim = reg.MigrationClaim(
        "root", "norm_epsilon", "POISON",
        (reg.ClaimBinding("rms_norm_eps",
                          _config_access.ProjectionTarget("root", "norm_eps")),))
    # the SAME occurrence, consumed into exactly the declared target — lawful in
    # every respect except that its reader never said where the value lives
    inexact = _config_access.ConfigAccessEvent(
        component="root", config_path="rms_norm_eps", canonical="rms_norm_eps",
        alias="rms_norm_eps", present=True, intent="consumed",
        fact_owner="root", fact_key="norm_eps", mechanism="norm_epsilon",
        path_exact=False)
    rows = validate_claims([inexact], (claim,))
    assert rows[0]["observed_events"] == 1
    assert rows[0]["target_matches"] == 0, "an inexact read may never MATCH"
    assert any("inexact read" in v and "bare funnel" in v
               for v in rows[0]["violations"]), rows[0]["violations"]

    # and the control: the identical event, exactly pathed, is lawful
    exact = _config_access.ConfigAccessEvent(
        component="root", config_path="rms_norm_eps", canonical="rms_norm_eps",
        alias="rms_norm_eps", present=True, intent="consumed",
        fact_owner="root", fact_key="norm_eps", mechanism="norm_epsilon",
        path_exact=True)
    lawful = validate_claims([exact], (claim,))
    assert lawful[0]["target_matches"] == 1
    assert lawful[0]["violations"] == []


def test_cor5_projector_claim_is_earned_without_laundering_encoder_width():
    """Qwen2-VL earns both projector-width claims from exact source-bound
    consumption. Its tower ``embed_dim`` is deliberately not consumed until
    U14 installs the modality FactLedger/receipt route, and the retired
    encoder-width claim is absent rather than dormant."""
    import json
    import pathlib

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    cfg = json.loads((corpus / "qwen2-vl-7b-instruct.json").read_text())["config"]
    ir = mu.unfold(cfg).to_ir()
    ca = ir["extras"]["config_access"]
    rows = {row["scope"]: row for row in ca["migration_claims"]}
    for scope in ("root.vision/projector_out_width",
                  "root.video/projector_out_width"):
        assert rows[scope]["observed_events"] > 0
        assert rows[scope]["target_matches"] > 0, scope
        assert rows[scope]["violations"] == []
    # U9 retired COR-5's config-only encoder-width claim.  The exact modality
    # DTO may carry a width established by source, but U14 owns its typed fact
    # and receipt cutover; a dormant migration claim must not pretend that
    # cutover already happened.
    assert "root.vision/encoder_width" not in rows
    assert "root.vision:hidden_size" in ca["consumed"]
    assert "root.video:hidden_size" in ca["consumed"]


def test_cor5_net2_is_scope_gated_not_globally_gated(monkeypatch):
    """U2 cutover: Net 2 is no longer gated by a global boolean.  It BLOCKS
    inside receipted (owner, mechanism) scopes and is advisory-empty for a
    model whose obligations all fall outside them.  A receipted scope whose
    consumer emits NO receipt must block (poisoned by shrinking the receipt
    set to empty)."""
    # LLAMA: no receipted-scope obligations -> advisory, empty, passes.
    rep = sable(LLAMA, render_images=False)
    net2 = next(c for c in rep.checks if c.name == "config_consumed_unreceipted")
    assert net2.blocking is True and net2.passed and net2.findings == []

    # POISON: declare a scope receipted but drop its receipts -> must block.
    from model_unfolder.evidence import receipts as receipts_mod

    real_join = receipts_mod.join_obligation_receipts

    def blind_join(obligations, receipts, facts=None, **kwargs):
        # render drew nothing — the receipts are dropped, everything else rides
        # through unchanged (U2-R5 signature: facts + context_token/scopes/routes)
        return real_join(obligations, [], facts, **kwargs)

    import json
    import pathlib

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    qwen = json.loads((corpus / "qwen2-vl-7b-instruct.json").read_text())["config"]
    monkeypatch.setattr(receipts_mod, "join_obligation_receipts", blind_join)
    rep2 = sable(qwen, render_images=False)
    net2b = next(c for c in rep2.checks if c.name == "config_consumed_unreceipted")
    assert net2b.blocking is True
    assert net2b.findings, "a receipted scope with no receipt must surface findings"
    assert not net2b.passed
    assert not rep2.mechanical_passed


# --------------------------------------------------------------------------- #
# Fourth vet (2026-07-15) — the three COR-5 soundness corrections.
# --------------------------------------------------------------------------- #

def _qwen2vl_cfg():
    import json
    import pathlib

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    return json.loads((corpus / "qwen2-vl-7b-instruct.json").read_text())["config"]


def test_cor5_poison_right_path_consumed_into_wrong_fact_blocks(monkeypatch):
    """Correction 1 POISON: the claimed path IS consumed — but into a fact the
    claim did not declare.  A path-only guard would pass this; the
    target-bound guard must name the drift and block."""
    from model_unfolder.evidence import registry as reg

    poisoned = (reg.MigrationClaim(
        "root.vision", "projector_out_width", "POISON",
        (reg.ClaimBinding(
            "vision_config.hidden_size",
            _config_access.ProjectionTarget("root.vision", "some_other_fact")),),
    ),)
    monkeypatch.setattr(reg, "MIGRATED_SCOPES", poisoned)
    ir = mu.unfold(_qwen2vl_cfg()).to_ir()
    rows = ir["extras"]["config_access"]["migration_claims"]
    assert len(rows) == 1
    violations = rows[0]["violations"]
    assert any("WRONG fact" in v and "source-to-target drift" in v
               for v in violations)
    assert rows[0]["target_matches"] == 0
    rep = sable(_qwen2vl_cfg(), render_images=False)
    check = next(c for c in rep.checks if c.name == "config_migration_claims")
    assert check.blocking and not check.passed


def _claim_rows_of(cfg):
    ca = (mu.unfold(cfg).to_ir().get("extras") or {}).get("config_access") or {}
    return ca.get("migration_claims") or []


@pytest.fixture(scope="session")
def corpus_claim_rows():
    """One pass over corpus + synthetic binding witnesses (alphabetical,
    early-exit once every declared BINDING is witnessed) — the anti-vacuity
    law and its poisons all consume this through the ONE audit function."""
    import json
    import pathlib

    from model_unfolder.evidence.claims_audit import audit_claim_coverage
    from model_unfolder.evidence.registry import MIGRATED_SCOPES
    from test_support.claim_witnesses import CLAIM_SYNTHETIC_WITNESSES

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    rows_by_witness: dict[str, list] = {}
    for name, cfg in sorted(CLAIM_SYNTHETIC_WITNESSES.items()):
        rows_by_witness[name] = _claim_rows_of(cfg)
    for path in sorted(corpus.glob("*.json")):
        rows_by_witness[path.stem] = _claim_rows_of(
            json.loads(path.read_text())["config"])
        coverage = audit_claim_coverage(rows_by_witness, MIGRATED_SCOPES)
        if not coverage["unwitnessed"]:
            break
    return rows_by_witness


def test_cor5_every_declared_binding_is_witnessed_on_corpus(corpus_claim_rows):
    """Fifth directive: coverage is BINDING-level — every declared
    path-to-target binding must be observed AND target-matched on at least
    one real or synthetic witness, through the SAME audit function the
    poisons call.  Per-witness zero observations stay lawful."""
    from model_unfolder.evidence.claims_audit import audit_claim_coverage
    from model_unfolder.evidence.registry import MIGRATED_SCOPES

    coverage = audit_claim_coverage(corpus_claim_rows, MIGRATED_SCOPES)
    assert coverage["unwitnessed"] == [], (
        "bindings never witnessed anywhere (add a witness or remove the "
        f"binding): {coverage['unwitnessed']}")
    assert coverage["witness_violations"] == {}


def test_cor5_poison_nonexistent_path_claim_fails_the_corpus_gate(monkeypatch):
    """Correction 2 POISON: a claim over a path no model carries produces
    observed_events=0 everywhere — the corpus-level law must flag it (while
    each individual model still passes: zero observations are lawful
    per-model)."""
    from model_unfolder.evidence import registry as reg

    bogus = reg.MigrationClaim(
        "root.vision", "phantom_mechanism", "POISON",
        (reg.ClaimBinding(
            "vision_config.no_such_field_xyz",
            _config_access.ProjectionTarget("root.vision", "phantom_fact")),),
    )
    monkeypatch.setattr(reg, "MIGRATED_SCOPES", (*reg.MIGRATED_SCOPES, bogus))
    ir = mu.unfold(_qwen2vl_cfg()).to_ir()
    rows = {r["scope"]: r for r in ir["extras"]["config_access"]["migration_claims"]}
    phantom = rows["root.vision/phantom_mechanism"]
    assert phantom["observed_events"] == 0
    assert phantom["violations"] == []           # per-model: lawful
    satisfied = {scope for scope, row in rows.items()
                 if row["observed_events"] > 0 and row["target_matches"] > 0}
    assert "root.vision/phantom_mechanism" not in satisfied
    # the corpus law (previous test) computes exactly this set over all
    # witnesses — a scope satisfied nowhere fails it.


# U2.2a vet: the fixture below carries a real DOCUMENT and each emit names the
# object it read.  It used to assert exact paths against nothing at all, which
# only worked while an explicit string could certify itself — the hole the vet
# closed.  A census row is now an occurrence PROVEN against the document, so a
# fixture that proves nothing rightly produces no rows; the law is unchanged and
# the fixture is what became honest.
_VAE_DOC = {"_vae_config": {"scaling_factor": 0.18,
                            "decoder": {"scale": 0.5}}}
# U2-R2b: the census admits only CHECKPOINT-provenance occurrences, so the VAE
# reads must be marked as the checkpoint's own words (they are — this is the
# VAE's config.json).  Without a map they read "" (unestablished) and are
# excluded, which is what regressed this test after ac860e6.
_VAE_PROV = {"_vae_config.scaling_factor": _config_access.CHECKPOINT_DECLARED,
             "_vae_config.decoder.scale": _config_access.CHECKPOINT_DECLARED}


def _vae_emit(intent="inspected", **kw):
    """Emit a read of the VAE document at its true location."""
    inner = _VAE_DOC["_vae_config"]
    which = kw.pop("of")
    obj = inner if which == "outer" else inner["decoder"]
    _config_access.emit(
        "scaling_factor", intent=intent, present=True, source_obj_id=id(obj), **kw)


def test_cor5_census_view_is_occurrence_exact():
    """Correction 3: two exact occurrences sharing one canonical leaf under one
    owner are TWO rows in the authoritative view (full ConfigOccurrenceKey);
    the (owner, canonical) view collapses them and is compatibility-only."""
    with _config_access.capture_events() as ledger:
        with _config_access.owner_scope("root.vae"), \
                _config_access.bound_document(
                    bind_document(_VAE_DOC, _VAE_PROV)):
            _vae_emit(of="outer", alias="scaling_factor",
                      config_path="_vae_config.scaling_factor")
            _vae_emit(of="decoder", alias="scale",
                      config_path="_vae_config.decoder.scale")
    exact = ledger.unconsumed_occurrences()
    assert [(k.config_path, k.actual_spelling) for k in exact] == [
        ("_vae_config.decoder.scale", "scale"),
        ("_vae_config.scaling_factor", "scaling_factor"),
    ]
    assert len(ledger.accessed_but_unconsumed()) == 1     # documented collapse
    with _config_access.capture_events() as ledger2:
        with _config_access.owner_scope("root.vae"), \
                _config_access.bound_document(
                    bind_document(_VAE_DOC, _VAE_PROV)):
            _vae_emit(of="outer", alias="scaling_factor",
                      config_path="_vae_config.scaling_factor")
            _vae_emit(of="decoder", alias="scale",
                      config_path="_vae_config.decoder.scale")
            _vae_emit(intent="consumed", of="outer", alias="scaling_factor",
                      config_path="_vae_config.scaling_factor",
                      fact_owner="root.vae", fact_key="scaling_factor")
    remaining = ledger2.unconsumed_occurrences()
    assert [k.config_path for k in remaining] == ["_vae_config.decoder.scale"]


def test_cor5_obligation_truth_has_no_canonical_fallback():
    """Correction 3: registered debt excuses ONLY the exact source occurrence.
    A sibling occurrence sharing the canonical leaf stays unreceipted — the
    leaf-name coincidence can no longer flip its truth state."""
    with _config_access.capture_events() as ledger:
        with _config_access.owner_scope("root.vae"):
            _config_access.emit("scaling_factor", intent="consumed", present=True,
                                alias="scaling_factor",
                                config_path="_vae_config.scaling_factor",
                                fact_owner="root.vae", fact_key="scaling_factor")
            _config_access.emit("scaling_factor", intent="consumed", present=True,
                                alias="scale",
                                config_path="_vae_config.decoder.scaling_factor",
                                fact_owner="root.vae", fact_key="decoder_scale")
    obligations = ledger.projection_obligations(
        pending_sources={("root.vae", "_vae_config.scaling_factor")})
    states = {ob.source_occurrence.config_path: ob.state for ob in obligations}
    assert states["_vae_config.scaling_factor"] == "pending"
    assert states["_vae_config.decoder.scaling_factor"] == "unreceipted"


# --------------------------------------------------------------------------- #
# Fifth directive (U0/U1 close) — exact mechanism matching, binding-level
# anti-vacuity, and the named positive/negative controls.  Every check here
# flows through the SAME functions the corpus gate uses (claims_audit).
# --------------------------------------------------------------------------- #

def test_final_poison_wrong_sink_kind_blocks(monkeypatch):
    """A binding declaring sink kind 'geometry' while the real consumption
    lands as a 'fact' must fail with the sink-kind law named."""
    from model_unfolder.evidence import registry as reg

    poisoned = (reg.MigrationClaim(
        "root.vision", "projector_out_width", "POISON",
        (reg.ClaimBinding(
            "vision_config.hidden_size",
            _config_access.ProjectionTarget(
                "root.vision", "projector_out_features",
                structural_sink_kind="geometry")),),
    ),)
    monkeypatch.setattr(reg, "MIGRATED_SCOPES", poisoned)
    rows = _claim_rows_of(_qwen2vl_cfg())
    assert any("WRONG sink kind" in v for v in rows[0]["violations"])


def test_final_poison_wrong_mechanism_blocks(monkeypatch):
    """A consumption tagged with a mechanism that declares NO binding for the
    path is a violation — another mechanism's binding never clears it (the
    cross-mechanism union is gone)."""
    from model_unfolder.evidence import registry as reg

    poisoned = (reg.MigrationClaim(
        "root.vision", "phantom_mechanism", "POISON",
        (reg.ClaimBinding(
            "vision_config.hidden_size",
            _config_access.ProjectionTarget("root.vision", "phantom_fact")),),
    ),)
    monkeypatch.setattr(reg, "MIGRATED_SCOPES", poisoned)
    rows = _claim_rows_of(_qwen2vl_cfg())
    violations = rows[0]["violations"]
    assert any("wrong mechanism" in v and "projector_out_width" in v
               for v in violations), violations


def test_final_poison_unwitnessed_binding_fails_coverage(corpus_claim_rows):
    """A declared binding no witness exercises must surface in
    ``unwitnessed`` — through the SAME audit function as the real gate."""
    from model_unfolder.evidence.claims_audit import audit_claim_coverage
    from model_unfolder.evidence import registry as reg

    ghost = reg.MigrationClaim(
        "root.vision", "encoder_width", "POISON",
        (reg.ClaimBinding(
            "vision_config.never_spelled_width",
            _config_access.ProjectionTarget("root.vision", "hidden_size")),),
    )
    coverage = audit_claim_coverage(
        corpus_claim_rows, (*reg.MIGRATED_SCOPES, ghost))
    assert coverage["unwitnessed"] == [
        "root.vision/encoder_width::vision_config.never_spelled_width -> "
        "root.vision.hidden_size[fact]"]


def test_final_control_config_width_cannot_author_an_encoder_claim():
    """The same config spelling can parameterize a proven projector, but a
    source-less tower cannot turn it into an encoder architecture claim.

    This pins the U9/U14 boundary: the former COR-5 config-only migration is
    gone, while the exact Qwen projector binding remains independently lawful.
    """
    from test_support.claim_witnesses import HIDDEN_SIZE_WITNESS

    rows_q = {r["scope"]: r for r in _claim_rows_of(_qwen2vl_cfg())}
    q_proj = next(b for b in rows_q["root.vision/projector_out_width"]["bindings"]
                  if b["path"] == "vision_config.hidden_size")
    assert q_proj["target_matches"] > 0
    assert rows_q["root.vision/projector_out_width"]["violations"] == []
    assert "root.vision/encoder_width" not in rows_q

    rows_e = {r["scope"]: r for r in _claim_rows_of(HIDDEN_SIZE_WITNESS)}
    e_proj = next(b for b in rows_e["root.vision/projector_out_width"]["bindings"]
                  if b["path"] == "vision_config.hidden_size")
    assert e_proj["target_matches"] == 0        # no source-bound projector
    assert "root.vision/encoder_width" not in rows_e
    assert all(r["violations"] == [] for r in rows_e.values())


def test_final_negative_control_flux_qwen_image_author_no_top_level_vision():
    """NEGATIVE projector controls (producer-fix strengthened): flux and
    qwen-image are diffusion pipelines whose text encoder is a VLM (mistral3).
    That VLM's vision is OWNED by root.text_encoder.vision — the pipelines
    author NO top-level root.vision projector or encoder consumption at all, so
    no language-width fabrication can leak into the pipeline's vision scope."""
    import json
    import pathlib

    from model_unfolder.evidence.config_access import capture_events, owner_scope

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    for slug, has_embedded_vision in (
            ("flux-2-dev", True), ("qwen-image", True)):
        cfg = json.loads((corpus / f"{slug}.json").read_text())["config"]
        rows = {r["scope"]: r for r in _claim_rows_of(cfg)}
        # no top-level projector consumption is claimed; the retired
        # config-only encoder-width scope must not reappear.
        assert rows["root.vision/projector_out_width"]["target_matches"] == 0, slug
        assert "root.vision/encoder_width" not in rows, slug
        assert all(r["violations"] == [] for r in rows.values()), slug
        # the VLM text encoder's vision is namespaced under its slot, not root
        with capture_events() as led:
            with owner_scope("root"):
                mu.unfold(cfg).to_ir()
        vision_owners = {e.component for e in led.events if "vision" in e.component}
        assert "root.vision" not in vision_owners, (slug, vision_owners)
        assert bool(any(
            o.startswith("root.text_encoder") and o.endswith("vision")
            for o in vision_owners)) is has_embedded_vision, (
                slug, vision_owners)


def test_final_qwen2vl_control_receipts_projector_and_withholds_encoder_width():
    """The multimodal witness receipts both exact projector lanes while its
    unreceipted encoder width has no migration claim and is not falsely
    consumed. Qwen2-VL is the permanent U9/U14 boundary control."""
    cfg = _qwen2vl_cfg()
    ir = mu.unfold(cfg).to_ir()
    rows = {r["scope"]: r
            for r in ir["extras"]["config_access"]["migration_claims"]}
    assert "root.vision/encoder_width" not in rows
    for scope in ("root.vision/projector_out_width",
                  "root.video/projector_out_width"):
        hidden = next(b for b in rows[scope]["bindings"]
                      if b["path"] == "vision_config.hidden_size")
        assert hidden["target_matches"] > 0, scope
        assert rows[scope]["violations"] == []
    projector = ir["extras"]["modalities"]["inputs"]["vision"]["projector"]
    assert projector["out_features"] == 3584
