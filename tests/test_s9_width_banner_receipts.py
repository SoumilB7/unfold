"""The existing numeric Hidden cell receipts only the scalar it really emits."""
from dataclasses import replace

import pytest

from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.receipts import join_obligation_receipts, value_status_hash
from model_unfolder.renderers.html import sections
from model_unfolder.renderers.html.render_context import RenderContext, activate_render_context
from test_support.s9_fixtures.s9_scalar_decisions import _config, _parse


def _chain(wrapper=False):
    checkpoint = _config()
    if wrapper:
        checkpoint = {"text_config": checkpoint}
    ir, facts, events = _parse(checkpoint)
    fact = facts["model.hidden_size"]
    validate_fact_claim(fact, fact.claim_evidence)
    rows = {"model.hidden_size": {"value": fact.value, "status": fact.status}}
    obligations = [{"source": {"component": event.component, "path": event.config_path},
                    "target": {"owner": event.fact_owner, "key": event.fact_key},
                    "mechanism": event.mechanism,
                    "expected_value_status_hash": event.value_status_hash}
                   for event in events if event.intent == "consumed"
                   and event.fact_owner == "model" and event.fact_key == "hidden_size"]
    assert len(obligations) == 1
    return ir.to_dict(), rows, obligations


def _render(ir, rows):
    context = RenderContext(fact_rows=rows)
    with activate_render_context(context):
        html = sections._stats_banner(ir)
    receipts = [receipt for event in context.events for receipt in event.receipts]
    return html, receipts, context


@pytest.mark.parametrize("wrapper", [False, True])
def test_existing_hidden_cell_closes_real_scalar_consumption_with_actual_node(wrapper):
    ir, rows, obligations = _chain(wrapper)
    html, receipts, context = _render(ir, rows)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.fact_id == "model.hidden_size"
    assert receipt.node_ids == ("stats_hidden_size",)
    assert receipt.surface == "html" and receipt.structural_target == "stats_banner.hidden_size"
    assert receipt.projector_symbol == "renderers.html.sections._stats_banner"
    assert receipt.projection_kind == "field" and receipt.context_token == context.context_token
    assert html.count('data-uf-receipt-node="stats_hidden_size"') == 1
    assert ('<div class="uf-stat" data-uf-receipt-node="stats_hidden_size">'
            '<div class="uf-stat-key">HIDDEN</div><div class="uf-stat-val">64</div></div>') in html
    assert receipt.fact_value_status_hash == value_status_hash(64, "config_declared")
    assert obligations[0]["source"]["path"] == ("text_config.hidden_size" if wrapper else "hidden_size")
    assert join_obligation_receipts(obligations, receipts, rows,
                                   context_token=context.context_token)["findings"] == []


def test_wrong_displayed_scalar_is_not_replaced_with_the_fact_value():
    ir, rows, obligations = _chain()
    ir["hidden_size"] = 96
    html, receipts, context = _render(ir, rows)
    assert '<div class="uf-stat-val">96</div>' in html
    assert receipts[0].fact_value_status_hash == value_status_hash(96, "config_declared")
    assert join_obligation_receipts(obligations, receipts, rows,
                                   context_token=context.context_token)["findings"]


def test_sparse_source_default_cell_receipt_cites_its_actual_weaker_tier(tmp_path):
    from model_unfolder.evidence.context import ParseContext
    from test_support.s9_fixtures.s9_class_default_value import _read
    checkpoint = _config()
    del checkpoint["hidden_size"]
    result, index, document = _read(tmp_path, checkpoint=checkpoint)
    assert result.status == "resolved"
    defaults = {"hidden_size": 64}
    context = ParseContext(result.claim_witness.bundle, class_defaults=defaults,
                           class_defaults_by_path={(): defaults})
    context._program_index = index
    ir, facts, _events = _parse(checkpoint, context=context, document=document)
    fact = facts["model.hidden_size"]
    validate_fact_claim(fact, fact.claim_evidence)
    assert fact.status == "class_default" and fact.config_paths == ()
    rows = {"model.hidden_size": {"value": fact.value, "status": fact.status}}
    html, receipts, render_context = _render(ir.to_dict(), rows)
    assert '<div class="uf-stat-val">64</div>' in html
    assert len(receipts) == 1
    assert receipts[0].fact_value_status_hash == value_status_hash(64, "class_default")

    # Actual default-reader spans reach the source constructor. A numeric cell
    # must not turn that legacy class/span match into a drawn module.
    from pathlib import Path
    from model_unfolder.evidence.component_owner import resolve_component_root
    from model_unfolder.evidence.config_access import checkpoint_fingerprint
    from model_unfolder.evidence.reconciliation import (
        ProjectionFactFinding, _fact_static_claims,
        projection_claims_from_product, static_claims_from_owner_graph,
    )
    from physics.instance_inventory import ModuleNode, ResolvedClass, SourceFile
    from test_support.s9_fixtures.s7_reconciliation import _inventory
    root = resolve_component_root(index, result.claim_witness.bundle, "root")
    assert root.status == "resolved"
    static_claims = static_claims_from_owner_graph(root.graph)
    assert any(claim.path_pattern == () for claim in
               _fact_static_claims(index, fact, static_claims))
    symbol = root.graph.root.symbol
    class_ref = ResolvedClass(Path(symbol.source.canonical_path).stem,
                              symbol.qualified_name)
    source = SourceFile(class_ref.module, symbol.source.canonical_path,
                        symbol.source.content_fingerprint)
    base_inventory = _inventory(count=1)
    inventory = replace(base_inventory,
        provenance=replace(base_inventory.provenance, source_files=(source,),
            config_sha256=checkpoint_fingerprint(checkpoint),
            resolved_class=class_ref,
            requested_factory=f"{class_ref.module}.{class_ref.qualname}",
            constructor_used=f"{class_ref.module}.{class_ref.qualname}(config)"),
        modules=(ModuleNode("", class_ref, class_ref.module, (class_ref,),
                            (), (), {}, ()),), repetition_groups=())
    assert len(render_context.events) == 1
    event = render_context.events[0]
    assert event.view == "stats_banner" and event.block_path == ()
    assert event.facts_projected == frozenset({fact.ledger_key()})
    assert event.receipts == tuple(receipts)
    arguments = dict(index=index, inventory=inventory, static_claims=static_claims,
                     ir=ir, facts={fact.ledger_key(): fact})
    assert projection_claims_from_product(**arguments, render_events=()) == ()
    assert projection_claims_from_product(
        **arguments, render_events=render_context.events) == ()
    assert render_context.events == [event]  # Scalar denominator/receipt retained.

    # Validate referenced fact identities before applying the scalar boundary.
    with pytest.raises(ValueError, match="facts absent from the typed ledger"):
        projection_claims_from_product(**arguments, render_events=(replace(
            event, facts_projected=frozenset({"model.absent_scalar"})),))

    # The same source-default fact can accompany a real canonical occurrence
    # citation; R4 keeps that block drawn even if its semantic stamp is missing.
    ir.extras.setdefault("render", {}).setdefault("model_blocks", []).append({
        "id": "actual_root_card", "kind": "opaque", "label": "Actual root",
        "source_instance_path": "", "source_fact_keys": [fact.ledger_key()]})
    for projected_fact in (fact, replace(fact, claim_evidence=None,
                                         claim_document_token="")):
        explicit = projection_claims_from_product(
            **{**arguments, "facts": {fact.ledger_key(): projected_fact}},
            render_events=render_context.events)
        assert len(explicit) == 1 and explicit[0].instance_path == ""
        assert explicit[0].axis.kind == "rendered"
        expected_findings = (() if projected_fact.claim_evidence is not None else
                             (ProjectionFactFinding(fact.ledger_key()),))
        assert explicit[0].axis.fact_findings == expected_findings


@pytest.mark.parametrize("formatted", ["65", "~64", "?"])
def test_changed_or_approximate_formatted_cell_cannot_receipt_original_scalar(monkeypatch, formatted):
    ir, rows, obligations = _chain()
    original = sections._fmt_int
    monkeypatch.setattr(sections, "_fmt_int", lambda value: formatted if value == 64 else original(value))
    html, receipts, context = _render(ir, rows)
    assert f'<div class="uf-stat-val">{formatted}</div>' in html
    assert receipts == []
    assert join_obligation_receipts(obligations, receipts, rows,
                                   context_token=context.context_token)["findings"]


def test_changed_final_html_text_is_not_receipted_by_pre_escape_value(monkeypatch):
    ir, rows, obligations = _chain()
    original = sections._html
    monkeypatch.setattr(sections, "_html", lambda value: "65" if value == "64" else original(value))
    html, receipts, context = _render(ir, rows)
    assert '<div class="uf-stat-val">65</div>' in html and receipts == []
    assert join_obligation_receipts(obligations, receipts, rows,
                                   context_token=context.context_token)["findings"]


def test_omitted_hidden_item_never_acquires_a_scalar_receipt(monkeypatch):
    ir, rows, obligations = _chain()
    ir.setdefault("extras", {}).setdefault("render", {})["family"] = "diffusion"
    monkeypatch.setattr(sections, "_diffusion_stats", lambda *_args: [("Params", "?")])
    html, receipts, context = _render(ir, rows)
    assert "HIDDEN" not in html and "stats_hidden_size" not in html and receipts == []
    assert join_obligation_receipts(obligations, receipts, rows,
                                   context_token=context.context_token)["findings"]


def test_foreign_fact_tier_or_render_context_cannot_clear_actual_consumption():
    ir, rows, obligations = _chain()
    _html, receipts, context = _render(ir, rows)
    foreign = RenderContext(fact_rows=rows)
    assert join_obligation_receipts(obligations, receipts, rows,
                                   context_token=foreign.context_token)["findings"]
    wrong_rows = {"model.hidden_size": {"value": 64, "status": "class_default"}}
    _html, wrong_receipts, wrong_context = _render(ir, wrong_rows)
    assert wrong_receipts[0].fact_value_status_hash == value_status_hash(64, "class_default")
    assert join_obligation_receipts(obligations, wrong_receipts, rows,
                                   context_token=wrong_context.context_token)["findings"]
    missing_html, missing, _missing_context = _render(ir, {})
    assert "HIDDEN" in missing_html and missing == []
    assert join_obligation_receipts(obligations, missing, rows,
                                   context_token=context.context_token)["findings"]


@pytest.mark.parametrize("change", [
    {"node_ids": ("embed",)}, {"surface": "ir"},
    {"projector_symbol": "adapters.transformer.parser.parse"},
    {"projection_kind": "connection"},
    {"owner": "decoder.attention", "fact_id": "decoder.attention.hidden_size"},
])
def test_banner_receipt_cannot_borrow_another_route_identity(change):
    ir, rows, obligations = _chain()
    _html, receipts, context = _render(ir, rows)
    assert join_obligation_receipts(obligations, [replace(receipts[0], **change)], rows,
                                   context_token=context.context_token)["findings"]


def test_banner_capability_without_an_actual_emission_cannot_clear_consumption():
    from model_unfolder.renderers.html.fact_projection import CONDITIONAL_PROJECTORS
    _ir, rows, obligations = _chain()
    assert any(row.owner == "model" and row.leaf == "hidden_size"
               and row.surface == "stats_banner" for row in CONDITIONAL_PROJECTORS)
    context = RenderContext(fact_rows=rows)
    assert context.events == []
    assert join_obligation_receipts(obligations, (), rows,
                                   context_token=context.context_token)["findings"]
