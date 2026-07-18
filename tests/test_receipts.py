"""U2-R5 projection receipts — FactDefinition is the sole route authority.

The pilot vertical: the source-bound vision/video projector output width.  The
consumption records its fingerprint; the typed FACT carries the value and the
``code_and_config`` status the source proves; the ACTUAL projector
(``declared_ops.build_declared_ops_view``) emits a receipt on the canonical
``card`` surface at the exact structural node; the render context stamps its own
token; and the validator joins occurrence -> fact -> registered route ->
receipt, strict on EVERY field.  The expected hash originates from the typed
fact and the consumption — never a renderer descriptor.

Poisons prove each field participates: a receipt wrong in context, owner, fact,
mechanism, surface, structural target, node identity, projector symbol, or
value/status hash blocks; a missing fact blocks; a consumption/fact fingerprint
disagreement blocks; a silent registry blocks; and every one of the NINE
canonical surfaces (including ``spec``) is exercised as a wrong-surface poison.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import model_unfolder as mu
from model_unfolder.evidence.receipts import (
    ProjectionReceipt,
    RECEIPTED_SCOPES,
    fabrication_findings,
    join_obligation_receipts,
    receipted_scopes,
    value_status_hash,
)
from model_unfolder.evidence.registry import (
    PROJECTION_ROUTE_SURFACES,
    ProjectionRoute,
    REGISTRY,
)
from model_unfolder.sable import sable

_CORPUS = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"


def _qwen2vl():
    return json.loads((_CORPUS / "qwen2-vl-7b-instruct.json").read_text())["config"]


# ---------------------------------------------------------------------------
# Hermetic fixture: one migrated scope, fully specified
# ---------------------------------------------------------------------------

_SYMBOL = ("renderers.html.block_views.declared_ops."
           "build_declared_ops_view")
_ROUTE = ProjectionRoute("root.vision", "projector_out_width", "card",
                         "vision_projector", frozenset({"op"}),
                         frozenset({("vision_projector",)}),
                         frozenset({_SYMBOL}))
_ROUTES = {"projector_out_features": (_ROUTE,)}
_FACT_ROWS = {"root.vision.projector_out_features":
              {"value": 3584, "status": "code_and_config"}}
_EXPECTED = value_status_hash(3584, "code_and_config")
_TOKEN = "render-token-1"


def _obligation(expected=_EXPECTED):
    return {
        "source": {"component": "root.vision",
                   "path": "vision_config.hidden_size"},
        "target": {"owner": "root.vision", "key": "projector_out_features"},
        "mechanism": "projector_out_width",
        "expected_value_status_hash": expected,
    }


def _receipt(**overrides):
    base = dict(
        fact_id="root.vision.projector_out_features", owner="root.vision",
        fact_key="projector_out_features", mechanism="projector_out_width",
        fact_value_status_hash=_EXPECTED, surface="card",
        structural_target="vision_projector",
        projector_symbol=_SYMBOL, projection_kind="op",
        node_ids=("vision_projector",), context_token=_TOKEN)
    base.update(overrides)
    return ProjectionReceipt(**base)


def _join(receipts, obligations=None, facts=None, token=_TOKEN, routes=None):
    return join_obligation_receipts(
        obligations if obligations is not None else [_obligation()],
        receipts, facts if facts is not None else _FACT_ROWS,
        context_token=token,
        scopes=frozenset({("root.vision", "projector_out_width")}),
        routes=routes if routes is not None else _ROUTES)


# ---------------------------------------------------------------------------
# The positive control — a valid chain clears, non-vacuously
# ---------------------------------------------------------------------------

def test_a_valid_receipt_clears_its_obligation():
    result = _join([_receipt()])
    assert result["findings"] == []
    assert result["receipted_targets"] == [
        ("root.vision", "root.vision.projector_out_features",
         "projector_out_width")]


def test_an_unreceipted_scope_stays_advisory():
    """Coverage is owner/mechanism-scoped: an obligation OUTSIDE the migrated
    scopes yields no finding — it remains the advisory census (exact R6 debt),
    never a silently-blocking one."""
    embedded = _obligation()
    embedded["target"] = {"owner": "root.text_encoder.vision",
                          "key": "projector_out_features"}
    embedded["source"] = {"component": "root.text_encoder.vision",
                          "path": "vision_config.hidden_size"}
    result = _join([], obligations=[embedded])
    assert result["findings"] == []
    assert result["receipted_targets"] == []


# ---------------------------------------------------------------------------
# Strict on EVERY field — each poison blocks for its own reason
# ---------------------------------------------------------------------------

def test_missing_receipt_blocks():
    result = _join([])
    assert any("no projector emitted" in f for f in result["findings"])


def test_foreign_context_token_blocks():
    """A receipt from another parse/render cannot clear this one."""
    result = _join([_receipt(context_token="some-other-render")])
    assert any("FOREIGN render-context token" in f for f in result["findings"])
    assert result["receipted_targets"] == []


def test_missing_typed_fact_blocks():
    """A consumption that never became a ledgered fact cannot be receipted —
    the expected hash originates from the FACT."""
    result = _join([_receipt()], facts={})
    assert any("NO typed fact is ledgered" in f for f in result["findings"])


def test_consumption_and_fact_disagreement_blocks():
    """The two upstream authorities must agree; drift between them is a finding,
    never a tie the validator breaks."""
    stale = value_status_hash(3584, "config_declared")
    result = _join([_receipt()], obligations=[_obligation(expected=stale)])
    assert any("disagrees with the typed fact" in f for f in result["findings"])


def test_missing_expectation_blocks():
    result = _join([_receipt()], obligations=[_obligation(expected="")])
    assert any("NO expected fingerprint" in f for f in result["findings"])


def test_wrong_owner_receipt_does_not_clear():
    result = _join([_receipt(owner="root.video",
                             fact_id="root.video.projector_out_features")])
    assert any("no projector emitted" in f for f in result["findings"])


def test_wrong_fact_receipt_does_not_clear():
    result = _join([_receipt(fact_id="root.vision.hidden_size",
                             fact_key="hidden_size")])
    assert any("no projector emitted" in f for f in result["findings"])


def test_wrong_mechanism_receipt_does_not_clear():
    result = _join([_receipt(mechanism="encoder_width")])
    assert any("no projector emitted" in f for f in result["findings"])


def test_wrong_structural_target_blocks():
    result = _join([_receipt(structural_target="video_projector")])
    assert any("matches no registered projection route" in f
               for f in result["findings"])


def test_wrong_node_identity_blocks():
    result = _join([_receipt(node_ids=("some_other_node",))])
    assert any("matches no registered projection route" in f
               for f in result["findings"])


def test_projector_symbol_requires_exact_membership():
    """R5-vet: "any nonempty symbol" validated nothing — a fake projector that
    is not in the route's allowed set blocks, and an empty one blocks too."""
    for fake in ("", "some.fake.projector"):
        result = _join([_receipt(projector_symbol=fake)])
        assert any("matches no registered projection route" in f
                   for f in result["findings"]), repr(fake)


def test_drifted_value_hash_blocks():
    """The renderer drew something other than the ledgered fact."""
    result = _join([_receipt(
        fact_value_status_hash=value_status_hash(9999, "code_and_config"))])
    assert any("does not match the typed fact" in f for f in result["findings"])


def test_silent_registry_blocks():
    """A scope treated as receipted with NO registered route is a finding —
    absence of the validator is never permission."""
    result = _join([_receipt()], routes={})
    assert any("registry is the sole route authority and it is silent" in f
               for f in result["findings"])


@pytest.mark.parametrize("surface", sorted(PROJECTION_ROUTE_SURFACES - {"card"}))
def test_every_wrong_canonical_surface_blocks(surface):
    """All NINE canonical surfaces participate — including ``spec``: a receipt
    on any surface the route does not name is rejected."""
    result = _join([_receipt(surface=surface)])
    assert any("matches no registered projection route" in f
               for f in result["findings"]), surface


def test_spec_is_a_registrable_surface_not_an_informal_category():
    """``spec`` is the explicit ninth surface: a route may target it, and an
    unknown surface is a constructor error."""
    route = ProjectionRoute("root.x", "m", "spec", "SpecTarget",
                            frozenset({"field"}),
                            projector_symbols=frozenset({"sym"}))
    assert route.surface == "spec"
    with pytest.raises(ValueError, match="nine canonical"):
        ProjectionRoute("root.x", "m", "not_a_surface", "t", frozenset({"op"}),
                        projector_symbols=frozenset({"sym"}))


# ---------------------------------------------------------------------------
# R5-vet corrections — each hole gets its own poison
# ---------------------------------------------------------------------------

def test_wrong_fact_key_with_correct_fact_id_blocks():
    """R5-vet hole 1: the candidate index keyed on fact_id alone, so a receipt
    with the RIGHT fact_id but a WRONG fact_key was accepted.  Coherence is now
    enforced both ways."""
    result = _join([_receipt(fact_key="hidden_size")])   # fact_id still correct
    assert any("MALFORMED" in f or "cites fact_key" in f
               for f in result["findings"])
    assert result["receipted_targets"] == []


def test_omitted_join_context_blocks_not_disables():
    """R5-vet hole 3: an empty context must BLOCK the receipted join, never
    disable the context check."""
    result = _join([_receipt(context_token="")], token="")
    assert any("NO render-context token" in f for f in result["findings"])
    assert result["receipted_targets"] == []


def test_wrong_projection_kind_blocks():
    """R5-vet hole 4: the route's projection_kinds now participate."""
    result = _join([_receipt(projection_kind="prose")])
    assert any("matches no registered projection route" in f
               for f in result["findings"])


def test_same_scope_two_fact_collision_does_not_cross_clear():
    """R5-vet: two facts sharing (owner, mechanism) — a receipt for fact A must
    never clear fact B's obligation.  Routes are keyed BY FACT."""
    other_route = ProjectionRoute(
        "root.vision", "projector_out_width", "card", "other_node",
        frozenset({"op"}), frozenset({("other_node",)}),
        frozenset({_SYMBOL}))
    routes = {"projector_out_features": (_ROUTE,),
              "other_width": (other_route,)}
    other_receipt = _receipt(fact_id="root.vision.other_width",
                             fact_key="other_width",
                             structural_target="other_node",
                             node_ids=("other_node",))
    result = _join([other_receipt], routes=routes)
    # the projector_out_features obligation is NOT cleared by the other fact
    assert any("no projector emitted" in f for f in result["findings"])
    assert result["receipted_targets"] == []


def test_route_construction_is_validated():
    """R5-vet item 9: bad routes are registry errors AT CONSTRUCTION."""
    from model_unfolder.evidence.registry import FactDefinition
    # kind outside the closed vocabulary
    with pytest.raises(ValueError, match="kind"):
        ProjectionRoute("root.x", "m", "card", "t", frozenset({"vibes"}),
                        projector_symbols=frozenset({"sym"}))
    # missing projector symbols
    with pytest.raises(ValueError, match="projector symbol"):
        ProjectionRoute("root.x", "m", "card", "t", frozenset({"op"}))
    good = ProjectionRoute("root.x", "m", "card", "t", frozenset({"op"}),
                           projector_symbols=frozenset({"sym"}))
    # route owner outside the fact's owner patterns
    with pytest.raises(ValueError, match="outside this fact's"):
        FactDefinition(
            key="projector_out_features",
            value_types=frozenset({"int"}),
            allowed_statuses=frozenset({"code_and_config"}),
            owner_patterns=frozenset({"root.vision"}),
            projection_routes=(ProjectionRoute(
                "root.GHOST", "m", "card", "t", frozenset({"op"}),
                projector_symbols=frozenset({"sym"})),))
    # duplicate route
    with pytest.raises(ValueError, match="duplicate projection route"):
        FactDefinition(
            key="projector_out_features",
            value_types=frozenset({"int"}),
            allowed_statuses=frozenset({"code_and_config"}),
            owner_patterns=frozenset({"root.x"}),
            projection_routes=(good, good))


def test_normalized_owner_pattern_matching_covers_concrete_indices():
    """R5-vet item 10: a ``layers[i]`` route pattern matches concrete owners,
    so future per-layer routes need no new machinery."""
    from model_unfolder.evidence.receipts import scope_is_receipted
    scopes = frozenset({("layers[i].ffn", "activation_width")})
    assert scope_is_receipted("layers[3].ffn", "activation_width", scopes)
    assert scope_is_receipted("layers[17].ffn", "activation_width", scopes)
    assert not scope_is_receipted("layers[3].attention", "activation_width",
                                  scopes)


def test_renderer_can_never_supply_a_status():
    """R5-vet item 5: the spec["status"] fallback is REMOVED — with no ledgered
    fact the cited status is the explicit non-status, whose hash can never
    match a real fact."""
    from model_unfolder.evidence.receipts import receipts_from_projects
    receipts = receipts_from_projects(
        [{"owner": "root.vision", "fact": "projector_out_features",
          "mechanism": "projector_out_width", "value": 3584,
          "status": "code_and_config"}],       # renderer-supplied — IGNORED
        surface="card", structural_target="vision_projector",
        projector_symbol=_SYMBOL, node_ids=("vision_projector",),
        projection_kind="op", fact_rows={})     # no ledgered fact
    assert receipts[0].fact_value_status_hash == value_status_hash(
        3584, "unledgered")
    assert receipts[0].fact_value_status_hash != _EXPECTED


def test_code_bound_width_records_a_typed_code_proven_fact():
    """R5-vet item 7: a pure-code width records a typed fact (with its exact
    projector source span) even though it has no config obligation."""
    from types import SimpleNamespace
    from model_unfolder.evidence.context import FactLedger, capture_facts
    from model_unfolder.adapters.transformer.special_parts.modalities.vision         import _bound_out_width
    evidence = SimpleNamespace(
        status="proven", out_width_source="code_bound", out_width_value=4096,
        component="vision_tower", projector_class="VisionProjector",
        source_file="modeling_x.py", line=123)
    ledger = FactLedger()
    with capture_facts(ledger):
        width, status = _bound_out_width(evidence, None, owner="root.vision")
    assert (width, status) == (4096, "code_proven")
    fact = ledger.typed.get("root.vision.projector_out_features")
    assert fact is not None and fact.status == "code_proven"
    assert fact.source_spans and fact.source_spans[0].file == "modeling_x.py"
    assert fact.source_spans[0].line == 123


def test_code_and_config_fact_cites_both_halves():
    """R5-vet item 8: code_and_config substantiates BOTH the config path and
    the projector source span."""
    extras = mu.unfold(_qwen2vl()).to_ir()["extras"]
    row = extras["fact_provenance"]["root.vision.projector_out_features"]
    assert row["status"] == "code_and_config"
    assert "vision_config.hidden_size" in row["source"]
    assert ".py:" in row["source"]           # the projector source span


def test_the_canonical_surface_list_is_exactly_nine():
    assert PROJECTION_ROUTE_SURFACES == {
        "ir", "spec", "opgraph", "block", "card", "html", "json", "params",
        "conformance"}


# ---------------------------------------------------------------------------
# Route authority — the registry, nowhere else
# ---------------------------------------------------------------------------

def test_receipted_scopes_derive_from_fact_definitions():
    """FactDefinition is the SOLE authority: the pilot scopes come from the
    registry's projection_routes."""
    scopes = receipted_scopes()
    assert ("root.vision", "projector_out_width") in scopes
    assert ("root.video", "projector_out_width") in scopes


def test_migration_claims_carry_no_projection_policy():
    """The claim-side policy is DELETED — a claim binds a source occurrence to
    a fact; the fact owns where it may project."""
    from model_unfolder.evidence.registry import MIGRATED_SCOPES
    for claim in MIGRATED_SCOPES:
        assert not hasattr(claim, "projection"), (
            f"{claim.owner}/{claim.mechanism} still carries a claim-side "
            "projection policy — FactDefinition is the sole route authority")


def test_receipted_scopes_lazy_view_tracks_the_registry():
    assert ("root.vision", "projector_out_width") in RECEIPTED_SCOPES
    assert len(RECEIPTED_SCOPES) >= 2


def test_coverage_is_scoped_and_the_global_bool_is_gone():
    ca = mu.unfold(_qwen2vl()).to_ir()["extras"]["config_access"]
    assert "projection_receipts_available" not in ca
    scopes = {tuple(s) for s in ca["projection_coverage"]["receipted_scopes"]}
    assert ("root.vision", "projector_out_width") in scopes
    assert ("root.video", "projector_out_width") in scopes


def test_an_empty_registry_cannot_vacuously_green():
    """Anti-vacuity: with no routes, nothing is receipted — an obligation in a
    claimed scope with an empty registry is a silent-registry finding, and no
    receipt is ever accepted."""
    result = _join([_receipt()], routes={})
    assert result["receipted_targets"] == []
    assert result["findings"]


# ---------------------------------------------------------------------------
# Reverse fabrication
# ---------------------------------------------------------------------------

def test_reverse_fabrication_catches_an_unregistered_receipt():
    ghost = _receipt(fact_id="root.vision.some_invented_fact",
                     fact_key="some_invented_fact")
    findings = fabrication_findings([ghost], _FACT_ROWS, set())
    assert any("nothing behind it" in f for f in findings)


def test_reverse_fabrication_requires_a_LEDGERED_fact_not_a_leaf_name():
    """R5-vet: root.ghost.projector_out_features shares a REGISTERED leaf name
    and used to pass — a leaf name is not evidence.  With no ledgered fact it
    is a fabrication; with a ledgered fact under a ghost owner it fails the
    definition's owner patterns."""
    ghost = _receipt(owner="root.ghost",
                     fact_id="root.ghost.projector_out_features")
    findings = fabrication_findings([ghost], _FACT_ROWS, set())
    assert any("nothing behind it" in f for f in findings)
    ghost_facts = dict(_FACT_ROWS)
    ghost_facts["root.ghost.projector_out_features"] = {
        "value": 3584, "status": "code_and_config"}
    findings = fabrication_findings([ghost], ghost_facts, set())
    assert any("outside" in f and "owner patterns" in f for f in findings)


def test_reverse_fabrication_validates_the_route_too():
    """A ledgered fact drawn OUTSIDE its registered routes is a fabrication of
    placement even though the fact is real."""
    off_route = _receipt(surface="html")
    findings = fabrication_findings([off_route], _FACT_ROWS, set())
    assert any("registered projection routes" in f for f in findings)


def test_registered_receipt_is_not_fabrication():
    findings = fabrication_findings([_receipt()], _FACT_ROWS, set())
    assert findings == []


def test_sable_fabrication_net_is_clean_and_blocking():
    rep = sable(_qwen2vl(), render_images=False)
    fab = next(c for c in rep.checks if c.name == "receipt_fabrication")
    assert fab.blocking is True and fab.passed and fab.findings == []


# ---------------------------------------------------------------------------
# The live pilot — end to end on the real witnesses
# ---------------------------------------------------------------------------

def _render_chain(cfg_dict):
    from model_unfolder.parser import config_to_ir, _coerce
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.diagram import Diagram
    from model_unfolder.renderers.html.render_context import (
        RenderContext, activate_render_context,
    )
    cfg = _coerce(cfg_dict)
    ctx = ParseContext.build(cfg, source="local")
    ir = config_to_ir(cfg, parse_context=ctx)
    diagram = Diagram(ir)
    rc = RenderContext(theme="teal")
    with activate_render_context(rc):
        diagram.to_html(standalone=True)
    receipts = [r for e in rc.events for r in getattr(e, "receipts", ()) or ()]
    return dict(ir.extras or {}), receipts, rc


def test_qwen2vl_positive_one_exact_chain():
    """The pilot: ONE exact obligation/fact/receipt chain per lane, emitted by
    the actual projector, canonical surface, context-stamped, zero findings."""
    extras, receipts, rc = _render_chain(_qwen2vl())
    projector_receipts = [r for r in receipts
                          if r.fact_key == "projector_out_features"]
    assert {r.owner for r in projector_receipts} == {"root.vision", "root.video"}
    for r in projector_receipts:
        assert r.surface == "card"
        assert r.projector_symbol.endswith("build_declared_ops_view")
        assert r.context_token == rc.context_token
    obls = (extras.get("config_access") or {}).get("projection_obligations") or []
    facts = extras.get("fact_provenance") or {}
    assert facts["root.vision.projector_out_features"]["status"] == "code_and_config"
    result = join_obligation_receipts(obls, receipts, facts,
                                      context_token=rc.context_token)
    assert result["findings"] == []
    assert len(result["receipted_targets"]) == 2


@pytest.mark.parametrize("witness", ["flux-2-dev", "qwen-image"])
def test_negative_controls_no_phantom_consumption_or_receipt(witness):
    """FLUX / Qwen-Image: no TOP-LEVEL vision modality, so no root.vision
    projector consumption, fact, or receipt.  The embedded encoder's own tower
    (root.text_encoder.vision) keeps its consumption VISIBLE on the advisory
    census — an unmigrated scope, exact R6 debt, never silently cleared."""
    cfg = json.loads((_CORPUS / f"{witness}.json").read_text())["config"]
    extras, receipts, rc = _render_chain(cfg)
    assert not [r for r in receipts if r.fact_key == "projector_out_features"]
    facts = extras.get("fact_provenance") or {}
    assert not [k for k in facts if "projector_out_features" in k]
    obls = [o for o in ((extras.get("config_access") or {})
                        .get("projection_obligations") or [])
            if o["target"]["key"] == "projector_out_features"]
    assert obls, "the embedded consumption must stay VISIBLE"
    for o in obls:      # visible, embedded-owned, unmigrated
        assert o["target"]["owner"].startswith("root.text_encoder"), o
    result = join_obligation_receipts(
        obls, receipts, facts, context_token=rc.context_token)
    assert result["findings"] == []          # advisory, not silently cleared
    assert result["receipted_targets"] == []


def test_sable_net2_is_green_and_blocking_on_the_pilot_witness():
    report = sable(_qwen2vl(), render_images=False)
    net2 = next(c for c in report.checks
                if c.name == "config_consumed_unreceipted")
    assert net2.blocking is True
    assert net2.passed and net2.findings == []


def test_unrelated_obligations_never_become_blocking():
    """Enabling projector receipts must not make a text/LM obligation blocking:
    an obligation outside a receipted scope produces no finding even with an
    empty receipt set."""
    ca = mu.unfold(_qwen2vl()).to_ir()["extras"]["config_access"]
    obligations = ca["projection_obligations"]
    outside = [o for o in obligations
               if (o["target"]["owner"], o.get("mechanism", ""))
               not in RECEIPTED_SCOPES]
    assert outside, "the witness must carry un-migrated obligations"
    result = join_obligation_receipts(outside, [], {}, context_token="t")
    assert result["findings"] == []
