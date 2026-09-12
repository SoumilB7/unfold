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
    html, receipts, _render_context = _render(ir.to_dict(), rows)
    assert '<div class="uf-stat-val">64</div>' in html
    assert len(receipts) == 1
    assert receipts[0].fact_value_status_hash == value_status_hash(64, "class_default")


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
