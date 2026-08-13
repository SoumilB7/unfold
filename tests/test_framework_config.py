"""U8-C prerequisite — exact framework-stored config address evidence."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.context import ParseContext, slot_parse_context
from model_unfolder.evidence.decoder_block import decoder_block_candidates_for_config
from model_unfolder.evidence.framework_config import (
    FrameworkConfigChildRelay,
    FrameworkConfigDefaultValue,
    config_path_from_framework_alias,
    framework_config_alias,
    framework_config_child_relay,
    framework_config_class,
    framework_config_class_default,
    framework_config_default_selector,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import ExprNode, build_program_index


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _source(*, base_import="transformers.modeling_utils",
            super_actual="config", base_init="", before_super="",
            stage_actual="config"):
    return f"""
from {base_import} import PreTrainedModel

class Base(PreTrainedModel):
{textwrap.indent(base_init, '    ') if base_init else '    pass'}

class Stage(Base):
    def __init__(self, config, other=None):
{textwrap.indent(before_super, '        ') if before_super else ''}
        super().__init__({super_actual})
        self.layers = []

class Wrapper(Base):
    base_model_prefix = "stage"
    def __init__(self, config):
        super().__init__(config)
        self.stage = Stage({stage_actual})
"""


def _synthetic(tmp_path, **kwargs):
    path = tmp_path / "modeling_framework_config.py"
    path.write_text(textwrap.dedent(_source(**kwargs)), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    assert root.status == stage.status == "resolved"
    return index, root, stage.occurrence


def _synthetic_default(tmp_path, *, default="False", annotation="ExactConfig",
                       support_component="root"):
    model = tmp_path / "modeling_exact.py"
    support = tmp_path / "configuration_exact.py"
    model.write_text(textwrap.dedent(f"""
        from transformers.modeling_utils import PreTrainedModel
        from .configuration_exact import ExactConfig

        class Base(PreTrainedModel):
            config: {annotation}

        class Stage(Base):
            def __init__(self, config):
                super().__init__(config)
                self.layers = []

        class Wrapper(Base):
            base_model_prefix = "stage"
            def __init__(self, config):
                super().__init__(config)
                self.stage = Stage(config)
    """), encoding="utf-8")
    support.write_text(textwrap.dedent(f"""
        class ExactConfig:
            decoder: bool = {default}
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(model),),
        component_files={"root": (str(model),)},
        supporting_files={support_component: (str(support),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    if root.status != "resolved" or stage.status != "resolved":
        return bundle, index, root, stage, None
    alias = framework_config_alias(index, root, stage.occurrence)
    return bundle, index, root, stage, alias


def test_exact_super_argument_proves_framework_stored_config_alias(tmp_path):
    index, root, stage = _synthetic(tmp_path)
    result = framework_config_alias(index, root, stage)
    assert result.status == "resolved", result.failures
    assert result.value.stored_field == "config"
    expression = ExprNode(
        "attribute", name="layer_types", children=(ExprNode(
            "attribute", name="config", children=(ExprNode(
                "name", name="self"),)),))
    assert config_path_from_framework_alias(expression, result.value) \
        == ("layer_types",)


def test_exact_single_parameter_root_binding_is_the_empty_config_prefix(
        tmp_path):
    index, root, _stage = _synthetic(tmp_path)
    result = framework_config_alias(
        index, root, root.graph.root.occurrence)
    assert result.status == "resolved", result.failures
    assert result.value.config_binding.origin == "root_argument"
    assert result.value.config_binding.prefixes == ((),)


def test_framework_stored_config_relays_to_one_exact_child_actual(tmp_path):
    index, root, stage = _synthetic(tmp_path, stage_actual="self.config")
    result = framework_config_child_relay(index, root, stage)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, FrameworkConfigChildRelay)
    assert result.value.child_occurrence == stage
    assert result.value.child_symbol.qualified_name == "Stage"
    assert result.value.child_binding.parameter == "config"
    assert result.value.child_binding.resolved_prefix == ()
    assert result.value.child_actual.source_segment == "self.config"
    assert result.value.child_actual.span in result.value.spans


def test_framework_child_relay_rejects_unrelated_or_rival_actuals(tmp_path):
    index, root, stage = _synthetic(tmp_path, stage_actual="other")
    result = framework_config_child_relay(index, root, stage)
    assert result.status == "failed"
    assert result.failures[0].kind in {"incomplete_graph", "conflict"}

    source = _source(stage_actual="self.config, self.config").replace(
        "def __init__(self, config):\n        super().__init__(config)\n"
        "        self.stage = Stage(self.config, self.config)",
        "def __init__(self, config):\n        super().__init__(config)\n"
        "        self.stage = Stage(self.config, self.config)")
    path = tmp_path / "modeling_rival_actual.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"}, architecture="Wrapper")
    rival_index = build_program_index(bundle)
    rival_root = resolve_component_root(rival_index, bundle, "root")
    rival_stage = resolve_declared_model_stage(rival_index, rival_root)
    assert rival_stage.status == "resolved"
    rival = framework_config_child_relay(
        rival_index, rival_root, rival_stage.occurrence)
    assert rival.status == "failed"
    assert rival.failures[0].kind == "conflict"


def test_framework_child_relay_dto_rejects_cross_site_forgery(tmp_path):
    index, root, stage = _synthetic(tmp_path, stage_actual="self.config")
    value = framework_config_child_relay(index, root, stage).value
    with pytest.raises(ValueError, match="exact child actual"):
        replace(value, child_actual=ExprNode(
            "name", name="config", span=value.child_actual.span,
            source_segment="config"))
    with pytest.raises(ValueError, match="exact parent site"):
        replace(value, child_occurrence=root.graph.root.occurrence)
    with pytest.raises(ValueError, match="exact site candidate"):
        replace(value, child_symbol=root.graph.root.symbol)
    with pytest.raises(ValueError, match="preserves its exact address"):
        replace(value, child_binding=replace(
            value.child_binding, prefixes=(("foreign",),)))


def test_exact_framework_config_formal_recovers_multi_parameter_root_address(
        tmp_path):
    source = _source().replace(
        "def __init__(self, config):\n        super().__init__(config)",
        "def __init__(self, config, enabled=True):\n"
        "        super().__init__(config)")
    path = tmp_path / "modeling_multi_root.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    result = framework_config_alias(
        index, root, root.graph.root.occurrence)
    assert result.status == "resolved", result.failures
    assert result.value.config_binding.origin == "component_root"
    assert result.value.config_binding.prefixes == ((),)
    with pytest.raises(ValueError, match="empty prefix"):
        replace(
            result.value,
            config_binding=replace(
                result.value.config_binding, prefixes=(("foreign",),)))


@pytest.mark.parametrize("super_actual", ["other", "42"])
def test_wrong_super_actual_cannot_borrow_the_config_binding(
        tmp_path, super_actual):
    index, root, stage = _synthetic(tmp_path, super_actual=super_actual)
    result = framework_config_alias(index, root, stage)
    assert result.status == "failed"
    assert result.failures[0].kind in {
        "incomplete_graph", "unsupported_syntax"}


def test_intervening_constructor_blocks_framework_alias(tmp_path):
    index, root, stage = _synthetic(
        tmp_path, base_init="def __init__(self, config):\n    super().__init__(config)")
    result = framework_config_alias(index, root, stage)
    assert result.status == "failed"
    assert result.failures[0].kind == "unsupported_syntax"
    assert "transform" in result.failures[0].detail


def test_local_lookalike_base_has_no_framework_storage_semantics(tmp_path):
    source = _source().replace(
        "from transformers.modeling_utils import PreTrainedModel",
        "class PreTrainedModel:\n    pass")
    path = tmp_path / "lookalike.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"}, architecture="Wrapper")
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    result = framework_config_alias(index, root, stage.occurrence)
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"


def test_wrong_external_framework_target_is_not_accepted(tmp_path):
    index, root, stage = _synthetic(
        tmp_path, base_import="somewhere.modeling_utils")
    result = framework_config_alias(index, root, stage)
    assert result.status == "failed"
    assert result.failures[0].kind == "external_unavailable"


def test_local_super_shadowing_blocks_builtin_protocol(tmp_path):
    index, root, stage = _synthetic(tmp_path, before_super="super = lambda: other")
    result = framework_config_alias(index, root, stage)
    assert result.status == "failed"
    assert result.failures[0].kind == "unsupported_syntax"


def test_framework_alias_dto_rejects_a_forged_storage_field(tmp_path):
    index, root, stage = _synthetic(tmp_path)
    result = framework_config_alias(index, root, stage)
    assert result.status == "resolved"
    with pytest.raises(ValueError, match="stored field"):
        replace(result.value, stored_field="architecture")
    with pytest.raises(ValueError, match="owner node"):
        replace(result.value, owner_symbol=root.graph.root.symbol)


def test_exact_inherited_annotation_reaches_support_config_literal(tmp_path):
    bundle, index, _root, _stage, alias = _synthetic_default(tmp_path)
    assert bundle.files == bundle.component_files["root"]
    assert bundle.supporting_files["root"][0].endswith("configuration_exact.py")
    assert alias.status == "resolved"
    config_class = framework_config_class(index, alias.value)
    assert config_class.status == "resolved", config_class.failures
    assert config_class.value.annotation_owner.qualified_name == "Base"
    assert config_class.value.config_class.qualified_name == "ExactConfig"
    default = framework_config_class_default(
        index, config_class.value, ("decoder",))
    assert default.status == "resolved", default.failures
    assert default.value.value is False
    assert default.value.assignment.span.source.canonical_path.endswith(
        "configuration_exact.py")


def test_checkpoint_value_wins_over_exact_class_default(tmp_path):
    _bundle, index, _root, _stage, alias = _synthetic_default(tmp_path)
    selector = framework_config_default_selector(
        index, alias.value,
        lambda path: (True, True, "config_declared")
        if path == ("decoder",) else (False, None, ""))
    assert selector(("decoder",)) == (True, True, "config_declared")


def test_omitted_value_receives_typed_exact_class_default(tmp_path):
    _bundle, index, _root, _stage, alias = _synthetic_default(tmp_path)
    selector = framework_config_default_selector(
        index, alias.value, lambda _path: (False, None, ""))
    selected = selector(("decoder",))
    assert isinstance(selected, FrameworkConfigDefaultValue)
    assert selected.value is False
    assert selected.path == ("decoder",)


@pytest.mark.parametrize("default", ["dynamic_default()", "False if flag else True"])
def test_dynamic_or_computed_class_default_stays_unavailable(tmp_path, default):
    _bundle, index, _root, _stage, alias = _synthetic_default(
        tmp_path, default=default)
    selector = framework_config_default_selector(
        index, alias.value, lambda _path: (False, None, ""))
    assert selector(("decoder",)) == (False, None, "")


def test_same_named_support_class_in_a_sibling_component_cannot_be_borrowed(
        tmp_path):
    _bundle, _index, root, stage, _alias = _synthetic_default(
        tmp_path, support_component="vision")
    # D0 remains exact for root modeling source, but the annotation cannot bind
    # a config class from a sibling component.
    assert root.status == stage.status == "resolved"
    alias = framework_config_alias(_index, root, stage.occurrence)
    result = framework_config_class(_index, alias.value)
    assert result.status == "failed"
    assert result.failures[0].kind == "external_unavailable"


def test_unresolvable_annotation_never_falls_back_by_class_spelling(tmp_path):
    _bundle, index, _root, _stage, alias = _synthetic_default(
        tmp_path, annotation="MissingConfig")
    result = framework_config_class(index, alias.value)
    assert result.status == "failed"
    assert result.failures[0].kind == "unresolved_import"


@pytest.mark.parametrize("fixture,slot", [
    ("gemma-2-2b-it.json", None),
    ("gpt-oss-20b.json", None),
    ("fluxtransformer2dmodel.json", "text_encoder_2"),
])
def test_real_transformers_stage_proves_framework_config_alias(fixture, slot):
    config = json.loads((_CORPUS / fixture).read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    if slot is not None:
        context = slot_parse_context(context, slot)
    index = context.program_index()
    candidates = decoder_block_candidates_for_config(
        index, context.source_bundle, (), allow_root_stage=True)
    assert candidates.status == "resolved"
    result = framework_config_alias(
        index, candidates.value.component_root,
        candidates.value.stage_occurrence)
    assert result.status == "resolved", result.failures
    assert result.value.stored_field == "config"
