"""Declared conditional capabilities need real, occurrence-local visible receipts."""
from copy import deepcopy
import importlib
from xml.etree import ElementTree

import pytest

from model_unfolder.renderers.html.fact_projection import CONDITIONAL_PROJECTORS
from model_unfolder.renderers.html.render_context import RenderContext, activate_render_context


OWNER = "root.denoiser"
CAPABILITIES = tuple(row for row in CONDITIONAL_PROJECTORS if row.owner == OWNER)
CHIP = "Source-proven spatial primitive: torch.nn.Conv2d"
IR = {"name": "tiny canonical block", "hidden_size": 4}


def _child(name, kind="opaque"):
    return {"id": name, "kind": kind, "label": name}


def _block(capability):
    key = capability.owner + "." + capability.leaf
    block = {"id": "subject", "source_component": "root", "source_owner": "ActualClass",
             "source_fact_keys": [key], "children": [], "detail": {}}
    leaf = capability.leaf
    if leaf == "spatial_mechanisms":
        block.update(facts=[CHIP], detail={"fact_display_lines": {key: [CHIP]}})
        visible = {"subject"}
    elif leaf == "ffn_mechanisms":
        names = ("gate_up_proj", "gate_up_split", "activation", "multiply", "down_proj")
        block.update(kind="ffn", view="runtime_ffn", detail={
            "ffn": {"kind": "dense", "activation": "gelu", "gated": True,
                    "projection_mode": "fused_gate_up", "intermediate_size": 8}},
            children=[{"id": name, "role": "operation"} for name in names])
        visible = {"gate_up_proj", "activation", "down_proj"}
    elif leaf == "context_connections":
        block["detail"] = {"source": "context", "source_label": "Context input",
                           "target_formal": "encoder_hidden_states"}
        visible = {"context"}
    elif leaf in {"cell_connections", "runtime_primitives"}:
        block["detail"] = {
            "connection_calls": {name: {"id": name, "kind": "conv2d", "label": name,
                                          "target": name, "guard": "unconditional"}
                                 for name in ("first", "second")},
            "connections": [{"source": "first", "target": "second"}]}
        visible = {"first", "second"}
    elif leaf == "cell_arithmetic":
        block.update(children=[_child("left"), _child("right")], detail={
            "connection_calls": {}, "connections": [],
            "return_arithmetic": {"operands": ["left", "right"], "scale": "divide"}})
        visible = {"left", "right"}
    elif leaf in {"stage_join_connections", "constructed_stage_relations", "constructed_modules"}:
        block.update(children=[_child("state"), _child("skip")], detail={
            "join_routes": [{"operands": ["state", "skip"], "join": "concat",
                             "target": "child_calls"}]})
        visible = {"state", "skip", "child_calls"}
    elif leaf == "primary_state_ports":
        block.update(children=[_child("argument"), _child("boundary")], detail={
            "port_route_kind": "call_result", "argument_ids": ["argument"],
            "boundary_id": "boundary", "result_slot": []})
        visible = {"argument", "boundary"}
    else:
        raise AssertionError("New capability needs its own real canonical-block witness: " + leaf)
    return block, visible


def _invoke(capability, block):
    """Invoke the declaration's actual emitter; no test-only receipt producer."""
    module, name = capability.emitter.rsplit(".", 1)
    emitter = getattr(importlib.import_module(module), name)
    context = RenderContext()
    with activate_render_context(context):
        if capability.surface == "card_chip":
            rendered = emitter(block["id"], block["facts"], block)
        else:
            with context.block(block):
                rendered = emitter(IR, {}, "conditional-contract", block)
    return rendered, context.events


def _matching(capability, events):
    key = capability.owner + "." + capability.leaf
    return [event for event in events
            if key in event.facts_projected and event.view == capability.event_view
            and event.block_path == ("subject",)]


def _visible_node_ids(svg):
    root = ElementTree.fromstring("<root>" + svg + "</root>")
    return {element.attrib["data-id"] for element in root.iter()
            if element.attrib.get("data-id") and "uf-node" in element.attrib.get("class", "").split()
            and len(element)}


@pytest.mark.parametrize("capability", CAPABILITIES, ids=lambda row: row.leaf)
def test_conditional_capability_has_real_visible_fresh_owner_qualified_receipt(capability):
    block, expected_visible = _block(capability)
    html, events = _invoke(capability, block)
    matching = _matching(capability, events)
    assert matching, "A declaration alone is not a render receipt"
    key = capability.owner + "." + capability.leaf
    assert all(event.facts_projected == frozenset({key}) for event in matching)
    assert all(event.component == "root" and event.source_owner == "ActualClass" for event in matching)
    if capability.surface == "card_chip":
        assert '<span class="uf-fact">' + CHIP + "</span>" in html
        assert matching[0].node_ids == frozenset(expected_visible)
    else:
        visible = _visible_node_ids(html)
        assert expected_visible <= visible
        assert expected_visible <= set().union(*(event.node_ids for event in matching))


@pytest.mark.parametrize("capability", CAPABILITIES, ids=lambda row: row.leaf)
@pytest.mark.parametrize("poison", ["wrong_fact_owner", "uncited"])
def test_real_renderer_cannot_satisfy_declared_owner_with_wrong_or_uncited_fact(capability, poison):
    block, _ = _block(capability)
    key = capability.owner + "." + capability.leaf
    other = "root.other_denoiser." + capability.leaf
    block["source_fact_keys"] = [other] if poison == "wrong_fact_owner" else []
    if capability.surface == "card_chip" and poison == "wrong_fact_owner":
        block["detail"]["fact_display_lines"] = {other: block["detail"]["fact_display_lines"][key]}
    html, events = _invoke(capability, block)
    assert html, "The wrong-owner/uncited control must still invoke the real drawing"
    assert not _matching(capability, events)
    assert all(key not in event.facts_projected for event in events)


@pytest.mark.parametrize("capability", CAPABILITIES, ids=lambda row: row.leaf)
def test_declared_capability_without_its_visible_payload_has_no_receipt(capability, monkeypatch):
    block, _ = _block(capability)
    leaf = capability.leaf
    if leaf == "spatial_mechanisms":
        block["facts"] = []
    elif leaf == "ffn_mechanisms":
        from model_unfolder.renderers.html.block_views import feed_forward
        monkeypatch.setattr(feed_forward, "build_ffn_view", lambda *args: "")
    elif leaf == "context_connections":
        from model_unfolder.renderers.html.block_views import unet
        monkeypatch.setattr(unet, "render_graph", lambda *args, **kwargs: "")
    elif leaf in {"cell_connections", "runtime_primitives"}:
        block["detail"]["connections"] = []
    elif leaf == "cell_arithmetic":
        block["detail"].pop("return_arithmetic")
    elif leaf in {"stage_join_connections", "constructed_stage_relations", "constructed_modules"}:
        block["detail"]["join_routes"] = []
        block["children"] = []
    elif leaf == "primary_state_ports":
        block["children"] = []
    else:
        raise AssertionError(leaf)
    _, events = _invoke(capability, block)
    assert not _matching(capability, events)


@pytest.mark.parametrize("poison", ["discarded", "partial"])
def test_ffn_capability_requires_nodes_in_the_actual_returned_svg(monkeypatch, poison):
    from model_unfolder.renderers.html.block_views import feed_forward
    capability = next(row for row in CAPABILITIES if row.leaf == "ffn_mechanisms")
    block, _ = _block(capability)
    original = feed_forward.build_ffn_view
    invoked = []

    def altered(*args):
        invoked.append(True)
        svg = original(*args)
        return "<svg></svg>" if poison == "discarded" else svg.replace(
            'data-id="activation"', 'data-id="missing_activation"')

    monkeypatch.setattr(feed_forward, "build_ffn_view", altered)
    _, events = _invoke(capability, deepcopy(block))
    assert invoked == [True]
    assert any(event.view == "ffn" for event in events)
    assert not _matching(capability, events)
