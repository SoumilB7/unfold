"""A partially sparse decoder count uses only its bound class default."""
import copy
import json
from pathlib import Path

import pytest

from model_unfolder.adapters.transformer.parser import parse
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.config_access import current_prepared_document


def _config():
    return json.loads((Path(__file__).parent / "sable_test_corpus" / "musicgen-small.json").read_text())["config"]


def test_context_only_nested_layer_default_never_constructs_layers():
    cfg = _config()
    assert cfg["decoder"].pop("num_hidden_layers") == 24
    context = ParseContext.build(cfg)
    context.class_defaults_by_path.setdefault(("decoder",), {})["num_hidden_layers"] = 99
    assert current_prepared_document.get() is None
    ir = parse(cfg, context=context)
    assert len(ir.layers) == 0
    assert current_prepared_document.get() is None
    assert not any(event.fact_key == "num_layers" and event.provenance == "class_default"
                   for event in context.config_access.events)
    assert "decoder.attention.cross_attention_schedule" not in context.facts.typed


@pytest.mark.parametrize("count", [None, 0])
def test_explicit_null_or_zero_count_is_not_repaired_from_class_default(count):
    cfg = _config()
    cfg["decoder"]["num_hidden_layers"] = count
    context = ParseContext.build(cfg)
    # Direct parsing exposes the limited IR before the public nonempty check.
    ir = parse(cfg, context=context)
    assert len(ir.layers) == 0
    assert not any(event.fact_key == "num_layers" and event.provenance == "class_default"
                   for event in context.config_access.events)


def test_failed_preparation_does_not_enable_a_context_count_candidate():
    from model_unfolder.evidence.config_access import bound_document
    from model_unfolder.evidence.document import DocumentBinding, PreparedDocument, PreparationFailure
    cfg = _config()
    cfg["decoder"].pop("num_hidden_layers")
    context = ParseContext.build(cfg)
    context.class_defaults_by_path.setdefault(("decoder",), {})["num_hidden_layers"] = 99
    document = PreparedDocument(cfg, copy.deepcopy(cfg),
        class_overlay={"decoder.num_hidden_layers": 99},
        failure=PreparationFailure("class_rejected", "hydrate", "intentional refusal control"))
    with bound_document(DocumentBinding("root", (), document)):
        ir = parse(cfg, context=context)
    assert len(ir.layers) == 0
    assert not any(event.fact_key == "num_layers" and event.provenance == "class_default"
                   for event in context.config_access.events)


# The actual successful nested default witness remains in
# test_s9_repeated_schedule_qualification.py: remove exactly the checkpoint's
# count, retain24 real layers and the qualified additive schedule, class_default
# tier and final schedule hash on its absent_default decision. Never skip it.
