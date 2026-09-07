"""Projection receipts follow exact rendered lines and the actual FFN graph."""
from copy import deepcopy

import pytest

from model_unfolder.renderers.html import cards
from model_unfolder.renderers.html.block_views.registry import render_block_detail
from model_unfolder.renderers.html.render_context import RenderContext, activate_render_context
from model_unfolder.renderers.html.views_diffusion import _build_loop_cards, _cards_for_children


KEY = "root.denoiser.declared_constructor_defaults"
OTHER = "root.denoiser.spatial_mechanisms"
FFN = "root.denoiser.ffn_mechanisms"
LINES = ['Declared class default · mode: "a&b" (checkpoint omitted)', "Second exact line"]


def block():
    return {"id": "denoiser", "title": "Card", "source_component": "root",
            "source_owner": "ExactOwner", "facts": list(LINES),
            "source_fact_keys": [KEY, OTHER],
            "detail": {"fact_display_lines": {KEY: list(LINES)}}}


def captured(render):
    context = RenderContext()
    with activate_render_context(context):
        html = render()
    return html, context.events


@pytest.mark.parametrize("kind", ["simple", "rich", "nested", "loop", "descendant"])
def test_actual_card_lines_emit_only_the_declared_displayed_fact(kind):
    card = block()
    def render():
        if kind == "simple":
            return cards._simple_card("denoiser", "Card", "", card["facts"], block=card)
        if kind == "rich":
            return cards._rich_card("denoiser", "Card", "", "<svg></svg>", card["facts"], block=card)
        if kind == "nested":
            return cards._nested_panel({}, {}, "test", [card])
        if kind == "loop":
            return _build_loop_cards({"extras": {"render": {"loop_blocks": [card]}}}, {}, "test",
                                     denoiser_arch="<svg></svg>")
        return _cards_for_children({}, {}, "test", [card])
    html, events = captured(render)
    assert "a&amp;b" in html
    assert len(events) == 1
    assert events[0].facts_projected == frozenset({KEY})
    assert events[0].block_path == ("denoiser",)
    assert events[0].source_owner == "ExactOwner"
    assert events[0].node_ids == frozenset({"denoiser"})


@pytest.mark.parametrize("poison", ["missing", "changed", "empty", "blank", "uncited", "wrong_card", "description_only"])
def test_no_receipt_when_exact_lines_do_not_render_on_that_card(poison):
    card = block()
    facts = card["facts"]
    node = card["id"]
    if poison == "missing": facts.pop()
    elif poison == "changed": facts[0] += " altered"
    elif poison == "empty": card["detail"]["fact_display_lines"][KEY] = []
    elif poison == "blank": card["detail"]["fact_display_lines"][KEY] = [" "]
    elif poison == "uncited": card["source_fact_keys"] = [OTHER]
    elif poison == "wrong_card": node = "other"
    elif poison == "description_only": facts = []
    _, events = captured(lambda: cards._simple_card(node, "Card", " ".join(LINES), facts, block=card))
    assert events == []


def test_chip_renderer_omission_cannot_be_satisfied_by_input_metadata(monkeypatch):
    card = block()
    monkeypatch.setattr(cards, "facts_html", lambda facts: "")
    html, events = captured(lambda: cards._simple_card("denoiser", "Card", "", LINES, block=card))
    assert "uf-fact" not in html
    assert events == []


def test_receipt_metadata_does_not_change_html():
    card = block()
    with_metadata, _ = captured(lambda: cards._simple_card("denoiser", "Card", "", LINES, block=card))
    without_metadata, events = captured(lambda: cards._simple_card("denoiser", "Card", "", LINES))
    assert with_metadata == without_metadata
    assert not events


def ffn_block():
    return {"id": "exact_ffn", "view": "runtime_ffn", "kind": "ffn",
            "source_component": "root", "source_owner": "ExactFFN",
            "source_fact_keys": [FFN, OTHER],
            "detail": {"ffn": {"kind": "dense", "activation": "gelu", "gated": True,
                                  "projection_mode": "fused_gate_up", "intermediate_size": 8}},
            "children": [{"id": node, "role": "operation"} for node in (
                "gate_up_proj", "gate_up_split", "activation", "multiply", "down_proj")]}


def test_actual_runtime_ffn_graph_receipts_only_its_mechanism():
    card = ffn_block()
    svg, events = captured(lambda: render_block_detail({"name": "fixture", "hidden_size": 4}, {}, "test", card))
    assert "<svg" in svg
    receipt = [event for event in events if event.view == "runtime_ffn_fact"]
    assert len(receipt) == 1
    assert receipt[0].facts_projected == frozenset({FFN})
    assert receipt[0].block_path == ("exact_ffn",)
    assert receipt[0].node_ids
    uncited = deepcopy(card)
    uncited["source_fact_keys"] = [OTHER]
    _, events = captured(lambda: render_block_detail({"name": "fixture", "hidden_size": 4}, {}, "test", uncited))
    assert not any(event.view == "runtime_ffn_fact" for event in events)


@pytest.mark.parametrize("empty_graph", ["", "<svg></svg>"])
def test_containment_only_or_empty_ffn_has_no_mechanism_receipt(monkeypatch, empty_graph):
    from model_unfolder.renderers.html.block_views import feed_forward
    monkeypatch.setattr(feed_forward, "build_ffn_view", lambda *args: empty_graph)
    card = ffn_block()
    card["children"] = [{"id": "net", "label": "Constructed net", "source_instance_path": "net"}]
    svg, events = captured(lambda: render_block_detail({"hidden_size": 4}, {}, "test", card))
    assert "Constructed net" in svg
    assert not any(event.view == "runtime_ffn_fact" for event in events)


@pytest.mark.parametrize("poison", ["discard", "partial", "foreign_block"])
def test_graph_event_cannot_substitute_for_the_returned_bound_svg(monkeypatch, poison):
    from model_unfolder.renderers.html.block_views import feed_forward
    from model_unfolder.renderers.html.render_context import current_render_context
    original = feed_forward.build_ffn_view
    def altered(*args):
        if poison == "foreign_block":
            with current_render_context().block({"id": "another_ffn"}):
                return original(*args)
        svg = original(*args)
        if poison == "discard":
            return "<svg></svg>"
        return svg.replace('data-id="activation"', 'data-id="removed_activation"')
    monkeypatch.setattr(feed_forward, "build_ffn_view", altered)
    svg, events = captured(lambda: render_block_detail(
        {"name": "fixture", "hidden_size": 4}, {}, "test", ffn_block()))
    assert "<svg" in svg
    assert any(event.view == "ffn" for event in events)
    assert not any(event.view == "runtime_ffn_fact" for event in events)
