"""Containment labels receipt only their own visible, qualified primitive node."""
from copy import deepcopy
from xml.etree import ElementTree

import pytest

from model_unfolder.renderers.html.block_views import unet
from model_unfolder.renderers.html.render_context import RenderContext, activate_render_context

KEY = "root.denoiser.runtime_primitives"


def _child(name):
    return {"id": name, "source_instance_path": "block." + name,
            "source_component": "root", "source_owner": "Linear", "kind": "linear",
            "label": "Linear", "source_fact_keys": [KEY]}


def _render(parent, supplied=None, rows=None):
    if rows is None:
        rows = {KEY: {"status": "code_proven", "value": {
            child["source_instance_path"]: {"kind": child["kind"], "label": child["label"]}
            for child in parent["children"]}}}
    context = RenderContext(fact_rows=rows)
    with activate_render_context(context), context.block(parent):
        svg = unet.build_constructed_children_view({}, {}, "uf-primitive-control",
                                                  supplied or parent)
    return svg, context.events


def test_filtered_actual_child_records_only_its_own_label():
    first, second = _child("first"), _child("second")
    parent = {"id": "parent", "children": [first, second], "source_fact_keys": ["wrong.parent"]}
    svg, events = _render(parent, {"children": [deepcopy(second)]})
    assert 'data-id="second"' in svg and 'data-id="first"' not in svg
    assert len(events) == 1
    event = events[0]
    assert event.view == "constructed_primitive_label"
    assert event.block_path == ("parent", "second")
    assert event.node_ids == frozenset({"second"})
    assert event.facts_projected == frozenset({KEY})
    assert event.source_owner == "Linear" and event.component == "root"
    assert not event.drawn_ops


@pytest.mark.parametrize("field,value", [
    ("source_instance_path", "foreign.first"), ("source_owner", "Foreign"),
    ("source_component", "other"), ("label", "Changed"),
    ("source_fact_keys", []), ("id", "foreign"),
])
def test_foreign_or_uncited_filtered_child_cannot_borrow_parent(field, value):
    child = _child("first")
    supplied = deepcopy(child)
    supplied[field] = value
    svg, events = _render({"id": "parent", "children": [child]}, {"children": [supplied]})
    assert "<svg" in svg
    assert events == []


@pytest.mark.parametrize("rows", [{}, {KEY: {"status": "asserted", "value": {}}},
                                   {KEY: {"status": "code_proven", "value": {}}},
                                   {KEY: {"status": "code_proven", "value": {
                                       "block.first": {"kind": "conv2d", "label": "Linear"}}}}])
def test_absent_or_mismatched_fact_member_cannot_receipt(rows):
    svg, events = _render({"id": "parent", "children": [_child("first")]}, rows=rows)
    assert 'data-id="first"' in svg
    assert events == []


def test_real_layout_then_discarded_svg_has_no_receipt(monkeypatch):
    fit = unet.fit_svg
    def discard(*args, **kwargs):
        assert 'data-id="first"' in fit(*args, **kwargs)
        return "<svg></svg>"
    monkeypatch.setattr(unet, "fit_svg", discard)
    svg, events = _render({"id": "parent", "children": [_child("first")]})
    assert svg == "<svg></svg>" and events == []


def test_partial_return_receipts_only_the_surviving_node(monkeypatch):
    fit = unet.fit_svg
    def partial(*args, **kwargs):
        root = ElementTree.fromstring(fit(*args, **kwargs))
        for container in root.iter():
            for node in list(container):
                if node.get("data-id") == "second":
                    container.remove(node)
        return ElementTree.tostring(root, encoding="unicode")
    monkeypatch.setattr(unet, "fit_svg", partial)
    svg, events = _render({"id": "parent", "children": [_child("first"), _child("second")]})
    assert 'data-id="first"' in svg and 'data-id="second"' not in svg
    assert len(events) == 1 and events[0].node_ids == frozenset({"first"})


def test_actual_returned_label_must_match_the_qualified_child(monkeypatch):
    fit = unet.fit_svg
    def relabel(*args, **kwargs):
        return fit(*args, **kwargs).replace(">Linear</text>", ">Different</text>")
    monkeypatch.setattr(unet, "fit_svg", relabel)
    svg, events = _render({"id": "parent", "children": [_child("first")]})
    assert ">Different</text>" in svg and events == []


def test_no_active_parent_cannot_claim_a_child_receipt():
    context = RenderContext()
    with activate_render_context(context):
        svg = unet.build_constructed_children_view({}, {}, "uf-no-parent",
                                                  {"children": [_child("first")]})
    assert 'data-id="first"' in svg and context.events == []
