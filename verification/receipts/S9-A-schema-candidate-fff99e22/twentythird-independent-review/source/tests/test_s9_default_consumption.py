"""Default accounting retains a channel address, never a checkpoint occurrence."""
from dataclasses import replace

import pytest

from model_unfolder.evidence.config_access import (
    CLASS_DEFAULT, bound_document, capture_events, resolve,
)
from model_unfolder.evidence.document import DocumentBinding, PreparationFailure, PreparedDocument
from model_unfolder.evidence.receipts import value_status_hash


def test_prepared_default_has_exact_consumption_address_without_occurrence():
    cfg = {}
    prepared = PreparedDocument(cfg, {}, class_overlay={"hidden_act": "silu"})
    with bound_document(DocumentBinding("root", (), prepared)), capture_events() as ledger:
        resolution = resolve(cfg, "hidden_act", ("activation",),
                             class_defaults={"hidden_act": "silu"})
        resolution.bind("actual_reader", "decoder.ffn", "activation")
        decision = resolution.consume_decision(
            mechanism="source-selected operand", fact_owner="decoder.ffn",
            fact_key="activation", reader="actual_reader", status="class_default")
    assert resolution.selected_path is None and decision.occurrence is None
    assert decision.default_path == ("hidden_act",)
    assert decision.provenance == CLASS_DEFAULT and not decision.present
    assert len(ledger.events) == 3
    for event in ledger.events:
        assert event.config_path == "hidden_act" and event.path_exact
        assert event.provenance == CLASS_DEFAULT
        assert event.intent == "absent_default" and not event.present and event.alias is None
    assert ledger.events[-1].value_status_hash == value_status_hash("silu", "class_default")
    assert not ledger.consumed()


def test_default_address_is_relative_to_actual_nested_parent_and_document():
    cfg = {"vision_config": {}}
    prepared = PreparedDocument(cfg, {"vision_config": {}},
                                class_overlay={"vision_config.width": 8})
    with bound_document(DocumentBinding("root.encoder", ("encoder",), prepared)), capture_events() as ledger:
        resolution = resolve(cfg["vision_config"], "width", path=("vision_config",),
                             class_defaults={"width": 8})
        decision = resolution.consume_decision(mechanism="dimension", fact_owner="vision",
                                               fact_key="width", reader="actual_reader")
    assert decision.default_path == ("vision_config", "width")
    assert ledger.events[-1].document_path == ("encoder",)
    assert ledger.events[-1].config_path == "vision_config.width"
    assert ledger.events[-1].path_exact and not ledger.events[-1].present


@pytest.mark.parametrize("case", ["bare", "foreign", "override", "failure", "null_alias", "null_parent"])
def test_unestablished_default_does_not_acquire_exact_channel_provenance(case):
    cfg = {}
    prepared = PreparedDocument(cfg, {}, class_overlay={"hidden_act": "silu"})
    defaults = {"hidden_act": "silu"}
    aliases = ("activation",)
    path = ()
    if case == "bare":
        prepared = PreparedDocument(cfg, {})
    elif case == "foreign":
        supplied = {}
    elif case == "override":
        defaults = {"hidden_act": "gelu"}
    elif case == "failure":
        prepared = replace(prepared, failure=PreparationFailure("class_rejected", "hydrate"))
    elif case == "null_alias":
        cfg["activation"] = None
        prepared = replace(prepared, checkpoint=dict(cfg))
    elif case == "null_parent":
        cfg["vision_config"] = None
        prepared = PreparedDocument(cfg, dict(cfg), class_overlay={"vision_config.hidden_act": "silu"})
        supplied = {}
        path = ("vision_config",)
    with bound_document(DocumentBinding("root", (), prepared)), capture_events() as ledger:
        resolution = resolve(supplied if case in {"foreign", "null_parent"} else cfg,
                             "hidden_act", aliases, path=path, class_defaults=defaults)
        assert resolution.default_premise is None
        resolution.consume("decoder.ffn", "activation", status="class_default")
    assert all(event.provenance != CLASS_DEFAULT for event in ledger.events)


@pytest.mark.parametrize("mutation", ["value", "overlay", "document", "checkpoint", "foreign_document"])
def test_retained_default_rejects_changed_value_or_document_before_consumption(mutation):
    cfg = {}
    prepared = PreparedDocument(cfg, {}, class_overlay={"hidden_act": "silu"})
    with bound_document(DocumentBinding("root", (), prepared)):
        resolution = resolve(cfg, "hidden_act", class_defaults={"hidden_act": "silu"})
        assert resolution.default_premise is not None
        if mutation == "value":
            resolution = replace(resolution, value="gelu")
        elif mutation == "overlay":
            prepared.class_overlay["hidden_act"] = "gelu"
        elif mutation == "document":
            prepared.document["hidden_act"] = None
        elif mutation == "checkpoint":
            prepared.checkpoint["hidden_act"] = None
        if mutation == "foreign_document":
            foreign = PreparedDocument({}, {}, class_overlay={"hidden_act": "silu"})
            with bound_document(DocumentBinding("root", (), foreign)), pytest.raises(ValueError):
                resolution.consume("decoder.ffn", "activation", status="class_default")
        else:
            # No capture is necessary: validating the operand precedes emission.
            with pytest.raises(ValueError):
                resolution.consume("decoder.ffn", "activation", status="class_default")


@pytest.mark.parametrize("change", [
    {"canonical": "other"}, {"component": "foreign"}, {"state": "present"},
    {"selected_path": "hidden_act"}, {"source_kind": "checkpoint"},
])
def test_default_consumption_revalidates_resolution_metadata_without_capture(change):
    cfg = {}
    prepared = PreparedDocument(cfg, {}, class_overlay={"hidden_act": "silu"})
    with bound_document(DocumentBinding("root", (), prepared)):
        resolution = resolve(cfg, "hidden_act", class_defaults={"hidden_act": "silu"})
        with pytest.raises(ValueError, match="metadata differs"):
            replace(resolution, **change).consume("decoder.ffn", "activation")


def test_default_consumer_refuses_duck_provider_without_capture():
    class Fake:
        path = ("hidden_act",)
        def validate_value(self, value):
            pass
    cfg = {}
    prepared = PreparedDocument(cfg, {}, class_overlay={"hidden_act": "silu"})
    with bound_document(DocumentBinding("root", (), prepared)):
        resolution = resolve(cfg, "hidden_act", class_defaults={"hidden_act": "silu"})
        forged = replace(resolution, default_premise=Fake())
        with pytest.raises(TypeError, match="exact supported typed"):
            forged.consume("decoder.ffn", "activation")
        with pytest.raises(TypeError, match="exact supported typed"):
            forged.bind("reader", "decoder.ffn", "activation")
