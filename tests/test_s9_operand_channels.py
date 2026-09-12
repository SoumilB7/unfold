"""Addressed operands retain checkpoint/default/null distinctions in production."""
from copy import deepcopy

import pytest

from model_unfolder.evidence.config_access import bound_document, capture_events, owner_scope
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.reader_claims import resolve_reader_operand


@pytest.mark.parametrize("path,value", [
    (("choose_split",), True),
    (("hidden_act",), "silu"),
    (("vision_config", "width"), 8),
    (("attention_multiplier",), 0.125),
])
def test_exact_operand_uses_only_the_active_prepared_default_and_preserves_null(path, value):
    empty = {}
    document = PreparedDocument(empty, {}, class_overlay={".".join(path): value})
    with bound_document(DocumentBinding("root", (), document)):
        default = resolve_reader_operand(empty, path)
        assert default.value == value and default.source_kind == "class_default"
        assert not default.present and default.selected_path is None
    checkpoint = {}
    parent = checkpoint
    for part in path[:-1]:
        parent = parent.setdefault(part, {})
    parent[path[-1]] = value
    document = PreparedDocument(checkpoint, deepcopy(checkpoint),
                                class_overlay={".".join(path): "rival default"})
    with bound_document(DocumentBinding("root", (), document)):
        ordinary = resolve_reader_operand(checkpoint, path)
        assert ordinary.value == value and ordinary.present
        assert ordinary.selected_path == ".".join(path)
        assert ordinary.source_kind != "class_default"
    parent[path[-1]] = None
    document = PreparedDocument(checkpoint, deepcopy(checkpoint),
                                class_overlay={".".join(path): value})
    with bound_document(DocumentBinding("root", (), document)):
        null = resolve_reader_operand(checkpoint, path)
        assert null.present and null.value is None
        assert null.source_kind != "class_default"


def test_conflicting_aliases_do_not_fall_through_to_class_default():
    checkpoint = {"hidden_size": 8, "n_embd": 16}
    document = PreparedDocument(checkpoint, deepcopy(checkpoint),
                                class_overlay={"hidden_size": 32})
    with bound_document(DocumentBinding("root", (), document)):
        conflict = resolve_reader_operand(checkpoint, ("hidden_size",), allow_aliases=True)
        assert conflict.ambiguous and conflict.source_kind != "class_default"


def test_foreign_supplied_operand_cannot_replace_the_prepared_default():
    document = PreparedDocument({}, {}, class_overlay={"hidden_act": "silu"})
    with bound_document(DocumentBinding("root", (), document)):
        with pytest.raises(ValueError, match="differs from its prepared document"):
            resolve_reader_operand({"hidden_act": "gelu"}, ("hidden_act",))
    # Outside the prepared channel an omitted operand stays missing.
    missing = resolve_reader_operand({}, ("hidden_act",))
    assert not missing.present and missing.source_kind != "class_default"


@pytest.mark.parametrize("channel", ["checkpoint", "class_default"])
def test_boolean_default_reaches_actual_config_selected_ffn_reader(tmp_path, channel):
    from test_support.s9_fixtures.ffn_mechanism import _config_selected_wrapper_reader
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.adapters.transformer.parser import _ffn_mechanism_result
    _config_selected_wrapper_reader(tmp_path, lambda path: True)
    source = str(tmp_path / "selected.py")
    bundle = SourceBundle(source="local", files=(source,), architecture="Wrapper",
                          component_files={"root": (source,)},
                          component_architectures={"root": "Wrapper"})
    context = ParseContext(bundle)
    checkpoint = {"choose_split": True} if channel == "checkpoint" else {}
    document = PreparedDocument(checkpoint, dict(checkpoint),
                                class_overlay={"choose_split": True})
    with bound_document(DocumentBinding("root", (), document)):
        result = _ffn_mechanism_result(context, config_root=checkpoint)
        assert result.status == "resolved", result.failures
        assert result.value.selector_value is True
        projection = result.claim_witness.project("decoder.ffn", "gated", document)
        assert projection.value is True
        assert projection.status == ("class_default" if channel == "class_default"
                                     else "code_and_config")
        assert projection.config_paths == (() if channel == "class_default"
                                          else (("choose_split",),))


@pytest.mark.parametrize("channel", ["checkpoint", "class_default", "null"])
def test_projector_actual_width_writer_preserves_default_channel(tmp_path, channel):
    from model_unfolder.evidence.models import SourceBundle
    from model_unfolder.evidence.context import ParseContext, capture_facts
    from model_unfolder.evidence.projector import projector_result_for_context
    from model_unfolder.adapters.transformer.special_parts.modalities.vision import apply_projector_evidence
    source = tmp_path / "modeling_projector.py"
    source.write_text('''from torch.nn import Linear
class Root:
    def __init__(self, config): self.bridge = Linear(config.input_width, config.output_width)
    def forward(self, inputs_embeds, image_features, mask):
        image_features = self.bridge(image_features)
        return inputs_embeds.masked_scatter(mask, image_features)
''')
    bundle = SourceBundle(source="test", files=(str(source),), architecture="Root",
                          component_files={"root": (str(source),)},
                          component_architectures={"root": "Root"})
    context = ParseContext(bundle)
    checkpoint = ({"input_width": 4, "output_width": 6} if channel == "checkpoint" else
                  {"input_width": None, "output_width": None} if channel == "null" else {})
    document = PreparedDocument(checkpoint, dict(checkpoint),
                                class_overlay={"input_width": 4, "output_width": 6})
    with bound_document(DocumentBinding("root", (), document)), capture_facts(context.facts):
        result = projector_result_for_context(context)
        assert result.status == "resolved", result.failures
        payload = {"modalities": {"inputs": {"vision": {"projector": {}, "pipeline": []}}}}
        apply_projector_evidence(payload, result, checkpoint)
        for key, value, path in (("projector_in_features", 4, "input_width"),
                                 ("projector_out_features", 6, "output_width")):
            fact = context.facts.typed.get("root.vision." + key)
            if channel == "null":
                assert fact is None
                continue
            assert fact.value == value
            assert fact.status == ("class_default" if channel == "class_default"
                                   else "code_and_config")
            assert fact.config_paths == (() if channel == "class_default" else (path,))
            assert fact.claim_kind == "value" and fact.claim_evidence is None


@pytest.mark.parametrize("channel", ["checkpoint", "class_default"])
@pytest.mark.parametrize("slug,leaf,fact_key", [
    ("granite-3-0-8b-instruct", "attention_multiplier", "decoder.attention.scores_scale"),
    ("deepseek-v3", "hidden_act", "decoder.ffn.expert.expert_activation_formula"),
])
def test_actual_parser_source_operand_keeps_checkpoint_and_default_tiers(channel, slug, leaf, fact_key):
    import json
    from pathlib import Path
    from model_unfolder.evidence.context import ParseContext, capture_facts
    from model_unfolder.adapters.transformer.parser import parse
    config = json.loads((Path(__file__).parent / "sable_test_corpus" / (slug + ".json")).read_text())["config"]
    value = config[leaf]
    if channel == "class_default":
        del config[leaf]
    context = ParseContext.build(config)
    # This control supplies a prepared value channel, not a new class-origin
    # proof. The mechanism remains independently source-bound and any stronger
    # application qualification debt must remain visible.
    context.class_defaults = {leaf: value}
    context.class_defaults_by_path = {(): {leaf: value}}
    document = PreparedDocument(config, deepcopy(config), class_overlay={leaf: value})
    with capture_events(context.config_access), owner_scope("root"), \
            bound_document(DocumentBinding("root", (), document)), capture_facts(context.facts):
        parse(config, context=context)
        fact = context.facts.typed[fact_key]
    assert fact.status == ("class_default" if channel == "class_default" else "code_and_config")
    assert fact.config_paths == (() if channel == "class_default" else (leaf,))
    assert fact.claim_kind == "applied_function"
    assert fact.value == ("declared" if leaf == "attention_multiplier" else {"kind": value})


@pytest.mark.parametrize("ancestor", [None, 0, False, [], "invalid"])
def test_explicit_noncontainer_ancestor_cannot_supply_a_nested_default(ancestor):
    from model_unfolder.evidence.reader_claims import reader_operand
    checkpoint = {"vision_config": ancestor}
    document = PreparedDocument(checkpoint, deepcopy(checkpoint),
                                class_overlay={"vision_config.width": 8})
    with bound_document(DocumentBinding("root", (), document)):
        assert resolve_reader_operand(checkpoint, ("vision_config", "width")) is None
        with pytest.raises(ValueError, match="no exact checkpoint or class-default evidence"):
            reader_operand(document, ("vision_config", "width"))


def test_failed_preparation_cannot_supply_defaults_but_keeps_checkpoint_operands():
    from model_unfolder.evidence.document import PreparationFailure
    from model_unfolder.evidence.reader_claims import reader_operand
    document = PreparedDocument({"ordinary": 3}, {"ordinary": 3},
                                class_overlay={"hidden_act": "silu"},
                                failure=PreparationFailure("class_rejected", "hydrate"))
    with bound_document(DocumentBinding("root", (), document)):
        missing = resolve_reader_operand(document.document, ("hidden_act",))
        assert not missing.present and missing.source_kind != "class_default"
        assert resolve_reader_operand(document.document, ("ordinary",)).value == 3
        with pytest.raises(ValueError, match="no exact checkpoint or class-default evidence"):
            reader_operand(document, ("hidden_act",))


@pytest.mark.parametrize("cfg", [{}, {"hidden_act": None}])
def test_foreign_absence_or_null_cannot_activate_a_source_literal_fallback(cfg):
    document = PreparedDocument({"hidden_act": "silu"}, {"hidden_act": "silu"})
    with bound_document(DocumentBinding("root", (), document)):
        with pytest.raises(ValueError, match="differs from its prepared document"):
            resolve_reader_operand(cfg, ("hidden_act",))


@pytest.mark.parametrize("cfg", [None, False, 0, [], "invalid"])
def test_noncontainer_root_cannot_be_repaired_with_a_prepared_default(cfg):
    document = PreparedDocument({}, {}, class_overlay={"hidden_act": "silu"})
    with bound_document(DocumentBinding("root", (), document)):
        with pytest.raises(ValueError, match="differs from its prepared document"):
            resolve_reader_operand(cfg, ("hidden_act",))
