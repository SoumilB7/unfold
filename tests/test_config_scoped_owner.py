"""U3-F5e — selected config scope -> exact config-constructed component root."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.config_scoped_owner import (
    ConfigConstructedRootResolution,
    resolve_config_constructed_root,
)
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle


def _auto_registry_source(tmp_path, *, architecture="Child"):
    """The smallest exact Transformers AutoModel registry proof.

    A bare ``AutoModel.from_config`` spelling is not class authority.  The
    production resolver requires the official implementation address, its
    literal lazy-map binding, and the checkpoint's exact config-class key.
    Synthetic positive controls must therefore carry that same proof rather
    than quietly reintroducing the old component-architecture shortcut.
    """
    path = tmp_path / "transformers" / "models" / "auto" / "modeling_auto.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(f"""
        from collections import OrderedDict
        from .auto_factory import _LazyAutoMapping

        CONFIG_MAPPING_NAMES = OrderedDict([("child_cfg", "ChildConfig")])
        MODEL_MAPPING_NAMES = OrderedDict([("child_cfg", "{architecture}")])
        MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)

        class AutoModel:
            _model_mapping = MODEL_MAPPING
    """), encoding="utf-8")
    return path


def _pipeline(tmp_path, body: str, *, include_child_component=True):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    components = {"root": (str(path),)}
    if include_child_component:
        components["child"] = (str(path),)
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files=components,
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    return index, bundle, root


_DEFAULT = """
    class Child:
        def __init__(self, config):
            self.config = config

    class Wrapper:
        def __init__(self, config=None, injected=None):
            if injected is None:
                injected = Child._from_config(config.child)
            self.slot = injected
"""


def test_default_active_local_factory_and_field_alias_resolve(tmp_path):
    index, bundle, root = _pipeline(tmp_path, _DEFAULT)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "resolved"
    proof = result.candidate
    assert proof.construction_site.target_kind == "local"
    assert proof.construction_site.target == "injected"
    assert proof.installation_field == "slot"
    assert proof.installation_kind == "local_alias"
    assert proof.root_parameter.name == "config"
    assert proof.root_binding.parameter == "config"
    assert proof.root_binding.resolved_prefix == ()
    assert proof.local_config_path == ("child",)
    assert [item.name for item in proof.defaulted_parameters] == ["injected"]
    assert proof.component_symbol.qualified_name == "Child"
    assert proof.component_root.graph.root.occurrence.root == proof.component_symbol
    assert {span.line for span in proof.spans} == {
        proof.construction_site.span.line,
        proof.installation_binding.span.line}


def test_config_construction_may_live_on_one_exact_reachable_owner(tmp_path):
    source = """
        class Child:
            def __init__(self, config): pass
        class Stage:
            def __init__(self, config):
                child = Child._from_config(config.child)
                self.slot = child
        class Wrapper:
            def __init__(self, config):
                self.stage = Stage(config)
    """
    index, bundle, root = _pipeline(tmp_path, source)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "resolved"
    assert result.candidate.outer_root == root.graph.root.occurrence
    assert result.candidate.construction_owner_symbol.qualified_name == "Stage"
    assert result.candidate.construction_owner.sites
    assert result.candidate.component_root.outer_root \
        == result.candidate.construction_owner


def test_nested_owner_parameter_prefix_is_part_of_the_exact_address(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent("""
        class Child:
            def __init__(self, config): pass
        class Stage:
            def __init__(self, config):
                child = Child._from_config(config.child)
                self.slot = child
        class Wrapper:
            def __init__(self, config):
                self.stage = Stage(config.other)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={
            "root": (str(path),),
            "child": (str(path),),
            "other.child": (str(path),),
        },
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"

    wrong = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert wrong.status == "absent"

    exact = resolve_config_constructed_root(
        index, bundle, root, ("other", "child"))
    assert exact.status == "resolved"
    assert exact.candidate.root_binding.resolved_prefix == ("other",)
    assert exact.candidate.local_config_path == ("child",)


def test_nested_owner_may_forward_its_already_scoped_config_whole(tmp_path):
    child_path = tmp_path / "child.py"
    child_path.write_text(textwrap.dedent("""
        class Child:
            def __init__(self, config): pass
    """), encoding="utf-8")
    wrapper_path = tmp_path / "wrapper.py"
    wrapper_path.write_text(textwrap.dedent("""
        from child import Child
        class Stage:
            def __init__(self, config):
                child = Child._from_config(config)
                self.slot = child
        class Wrapper:
            def __init__(self, config):
                self.stage = Stage(config.child)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(wrapper_path), str(child_path)),
        component_files={
            "root": (str(wrapper_path), str(child_path)),
            "child": (str(child_path),),
        },
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"

    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "resolved"
    assert result.candidate.root_binding.resolved_prefix == ("child",)
    assert result.candidate.local_config_path == ()


def test_exact_non_component_wrapper_is_not_a_rival_component_root(tmp_path):
    wrapper_path = tmp_path / "wrapper.py"
    child_path = tmp_path / "child.py"
    wrapper_path.write_text(textwrap.dedent("""
        class Stage:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                self.stage = Stage(config.child)
    """), encoding="utf-8")
    child_path.write_text(
        "class Child:\n"
        "    def __init__(self, config): pass\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(
        source="local", files=(str(wrapper_path), str(child_path)),
        component_files={
            "root": (str(wrapper_path),),
            "child": (str(child_path),),
        },
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "absent"


def test_rival_owner_parameter_prefixes_never_select_one_path(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent("""
        class Child:
            def __init__(self, config): pass
        class Stage:
            def __init__(self, config):
                child = Child._from_config(config.child)
                self.slot = child
        class Wrapper:
            def __init__(self, config, flag):
                self.stage = Stage(
                    config.left if flag else config.right)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={
            "root": (str(path),),
            "left.child": (str(path),),
        },
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    result = resolve_config_constructed_root(
        index, bundle, root, ("left", "child"))
    assert result.status == "failed"
    assert result.failure_kind == "unresolved_config_prefix"


def test_same_nested_constructor_at_two_occurrences_is_ambiguous(tmp_path):
    source = """
        class Child:
            def __init__(self, config): pass
        class Stage:
            def __init__(self, config):
                child = Child._from_config(config.child)
                self.slot = child
        class Wrapper:
            def __init__(self, config):
                self.left = Stage(config)
                self.right = Stage(config)
    """
    index, bundle, root = _pipeline(tmp_path, source)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "ambiguous"
    assert len(result.rivals) == 2
    assert len({item.construction_owner for item in result.rivals}) == 2


def test_names_do_not_select_a_sibling_config_scope(tmp_path):
    index, bundle, root = _pipeline(tmp_path, _DEFAULT)
    result = resolve_config_constructed_root(
        index, bundle, root, ("other",))
    assert result.status == "failed"
    assert result.failure_kind == "component_source_absent"


def test_non_none_injected_default_cannot_activate_guard(tmp_path):
    source = _DEFAULT.replace(
        "def __init__(self, config=None, injected=None):",
        "def __init__(self, config=None, injected=sentinel):",
    )
    index, bundle, root = _pipeline(tmp_path, source)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "failed"
    assert result.failure_kind == "unresolved_config_construction"
    assert "guard is not exactly decidable" in result.failure_detail


def test_dynamic_factory_candidate_is_failed_never_guessed(tmp_path):
    source = _DEFAULT.replace(
        "Child._from_config(config.child)",
        "factory_for(config.kind)(config.child)",
    )
    index, bundle, root = _pipeline(tmp_path, source)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "failed"
    assert result.failure_kind == "unsupported_config_construction"


def test_local_reassignment_before_field_alias_is_failed(tmp_path):
    source = _DEFAULT.replace(
        "            self.slot = injected\n",
        "            injected = other\n            self.slot = injected\n",
    )
    index, bundle, root = _pipeline(tmp_path, source)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "failed"
    assert "redefined before its field alias" in result.failure_detail


def test_control_exit_before_field_alias_is_failed(tmp_path):
    source = _DEFAULT.replace(
        "            self.slot = injected\n",
        "            if stop:\n                return\n"
        "            self.slot = injected\n",
    )
    index, bundle, root = _pipeline(tmp_path, source)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "failed"
    assert "control can exit" in result.failure_detail


def test_two_exact_local_constructions_are_ambiguous(tmp_path):
    source = """
        class A:
            def __init__(self, config): pass
        class B:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                a = A._from_config(config.child)
                b = B._from_config(config.child)
                self.left = a
                self.right = b
    """
    index, bundle, root = _pipeline(tmp_path, source)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "ambiguous"
    assert {item.component_symbol.qualified_name for item in result.rivals} == {
        "A", "B"}


@pytest.mark.parametrize("exact_count", [1, 2])
def test_unresolved_candidate_prevents_a_closed_exact_result(
        tmp_path, exact_count):
    exact = (
        "        a = A._from_config(config.child)\n"
        "        self.a = a\n"
    )
    if exact_count == 2:
        exact += (
            "        b = B._from_config(config.child)\n"
            "        self.b = b\n"
        )
    source = (
        "class A:\n"
        "    def __init__(self, config): pass\n"
        "class B:\n"
        "    def __init__(self, config): pass\n"
        "class Wrapper:\n"
        "    def __init__(self, config):\n"
        f"{exact}"
        "        unknown = factory_for(config.kind)(config.child)\n"
        "        self.unknown = unknown\n"
    )
    index, bundle, root = _pipeline(tmp_path, source)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "failed"
    assert result.failure_kind == "unsupported_config_construction"


def test_imported_factory_class_binds_to_exact_component_source(tmp_path):
    child_path = tmp_path / "child.py"
    child_path.write_text(
        "class Child:\n"
        "    @classmethod\n"
        "    def _from_config(cls, config): return cls(config)\n",
        encoding="utf-8",
    )
    wrapper_path = tmp_path / "wrapper.py"
    wrapper_path.write_text(
        "from child import Child\n"
        "class Wrapper:\n"
        "    def __init__(self, config, injected=None):\n"
        "        if injected is None:\n"
        "            injected = Child._from_config(config.child)\n"
        "        self.slot = injected\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(
        source="local",
        files=(str(wrapper_path), str(child_path)),
        component_files={
            "root": (str(wrapper_path), str(child_path)),
            "child": (str(child_path),),
        },
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "resolved"
    assert result.candidate.component_symbol.source.canonical_path \
        == str(child_path.resolve())
    assert result.candidate.component_symbol.source.component_key == "child"


def test_ambiguous_import_cannot_be_narrowed_by_component_membership(tmp_path):
    """Component metadata cannot select one of several exact code bindings."""
    selected_dir = tmp_path / "selected"
    rival_dir = tmp_path / "rival"
    selected_dir.mkdir()
    rival_dir.mkdir()
    selected = selected_dir / "child.py"
    rival = rival_dir / "child.py"
    for path in (selected, rival):
        path.write_text(
            "class Child:\n"
            "    @classmethod\n"
            "    def _from_config(cls, config): return cls(config)\n",
            encoding="utf-8",
        )
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "from child import Child\n"
        "class Wrapper:\n"
        "    def __init__(self, config):\n"
        "        self.slot = Child._from_config(config.child)\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(
        source="local",
        files=(str(wrapper), str(selected), str(rival)),
        component_files={
            "root": (str(wrapper), str(selected), str(rival)),
            "child": (str(selected),),
        },
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"

    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "failed"
    assert result.failure_kind == "unresolved_config_construction"
    assert "exact import rivals" in result.failure_detail


def test_missing_field_alias_fails_instead_of_promoting_local(tmp_path):
    source = _DEFAULT.replace("            self.slot = injected\n", "")
    index, bundle, root = _pipeline(tmp_path, source)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "failed"
    assert "0 exact field aliases" in result.failure_detail


def test_selected_scope_without_component_source_is_typed_failure(tmp_path):
    index, bundle, root = _pipeline(
        tmp_path, _DEFAULT, include_child_component=False)
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "failed"
    assert result.failure_kind == "component_source_absent"


def test_foreign_index_cannot_reuse_a_d0_root(tmp_path):
    index, bundle, root = _pipeline(tmp_path, _DEFAULT)
    other_path = tmp_path / "other.py"
    other_path.write_text("class Other: pass\n")
    other_bundle = SourceBundle(
        source="local", files=(str(other_path),),
        component_files={"root": (str(other_path),)})
    other = pi.build_program_index(other_bundle)
    result = resolve_config_constructed_root(
        other, bundle, root, ("child",))
    assert result.status == "failed"
    assert result.failure_kind == "index_mismatch"


def test_result_closure_rejects_payload_on_absent(tmp_path):
    index, bundle, root = _pipeline(tmp_path, _DEFAULT)
    resolved = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    with pytest.raises(ValueError):
        replace(resolved, status="absent")
    with pytest.raises(ValueError):
        ConfigConstructedRootResolution(
            "failed", root.occurrence, ("child",), "child")


def test_real_musicgen_decoder_scope_resolves_from_code_not_identity():
    cfg = json.loads(
        (Path("tests/sable_test_corpus") / "musicgen-small.json").read_text()
    )["config"]
    context = ParseContext.build(cfg)
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    result = resolve_config_constructed_root(
        index, context.source_bundle, root, ("decoder",))
    assert result.status == "resolved"
    assert result.candidate.component_symbol.qualified_name \
        == "MusicgenForCausalLM"
    assert result.candidate.construction_site.constructor.source_segment \
        == "MusicgenForCausalLM._from_config(config.decoder)"


def test_real_musicgen_auto_dispatch_uses_the_exact_framework_registry():
    cfg = json.loads(
        (Path("tests/sable_test_corpus") / "musicgen-small.json").read_text()
    )["config"]
    context = ParseContext.build(cfg)
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    text = resolve_config_constructed_root(
        index, context.source_bundle, root, ("text_encoder",))
    audio = resolve_config_constructed_root(
        index, context.source_bundle, root, ("audio_encoder",))
    assert text.status == audio.status == "resolved"
    assert text.candidate.component_symbol.qualified_name == "T5EncoderModel"
    assert audio.candidate.component_symbol.qualified_name == "EncodecModel"
    # The config's generic architecture is T5Model; the exact task-specific
    # AutoModelForTextEncoding registry is the runtime authority instead.
    assert context.source_bundle.component_architectures["text_encoder"] \
        == "T5Model"


def test_real_qwen2_vl_text_scope_resolves_at_reachable_direct_field():
    cfg = json.loads(
        (Path("tests/sable_test_corpus")
         / "qwen2-vl-7b-instruct.json").read_text()
    )["config"]
    context = ParseContext.build(cfg)
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    result = resolve_config_constructed_root(
        index, context.source_bundle, root, ("text_config",))
    assert result.status == "resolved"
    proof = result.candidate
    assert proof.component_symbol.qualified_name == "Qwen2VLTextModel"
    assert proof.construction_owner_symbol.qualified_name == "Qwen2VLModel"
    assert proof.installation_kind == "direct_field"
    assert proof.installation_field == "language_model"
    assert proof.construction_owner.sites
    with pytest.raises(ValueError):
        replace(proof, construction_owner_symbol=proof.component_symbol)
    with pytest.raises(TypeError):
        replace(proof, installation_kind="local_alias")
    with pytest.raises(ValueError):
        replace(
            proof.component_root,
            outer_owner_symbol=proof.component_symbol)


def test_transformers_auto_model_from_config_joins_declared_component(tmp_path):
    auto_path = _auto_registry_source(tmp_path)
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent("""
        from transformers.models.auto.modeling_auto import AutoModel

        class Wrapper:
            def __init__(self, config):
                self.slot = AutoModel.from_config(config.child)
    """), encoding="utf-8")
    child_path = tmp_path / "modeling_child.py"
    child_path.write_text(textwrap.dedent("""
        class Child:
            def __init__(self, config): pass
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path), str(auto_path), str(child_path)),
        component_files={
            "root": (str(path), str(auto_path)), "child": (str(child_path),)},
        component_architectures={"root": "Wrapper", "child": "Child"},
        component_model_types={"child": "child_cfg"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "resolved", result.failure_detail
    assert result.candidate.component_symbol.qualified_name == "Child"
    assert result.candidate.installation_field == "slot"
    assert result.candidate.construction_site.constructor.source_segment \
        == "AutoModel.from_config(config.child)"


def test_exact_direct_construction_resolves_when_component_map_has_no_architecture(
        tmp_path):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent("""
        class Child:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                self.slot = Child(config.child)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),), "child": (str(path),)},
        component_architectures={"root": "Wrapper"}, architecture="Wrapper")
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "resolved", result.failure_detail
    assert result.candidate.component_symbol.qualified_name == "Child"


def test_missing_component_architecture_keeps_two_config_consumers_as_rivals(
        tmp_path):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent("""
        class First:
            def __init__(self, config): pass
        class Second:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                self.first = First(config.child)
                self.second = Second(config.child)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),), "child": (str(path),)},
        component_architectures={"root": "Wrapper"}, architecture="Wrapper")
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "ambiguous"
    assert {item.component_symbol.qualified_name for item in result.rivals} \
        == {"First", "Second"}


def test_declared_component_architecture_selects_only_an_exact_candidate(
        tmp_path):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent("""
        class Tower:
            def __init__(self, config): pass
        class Connector:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                self.tower = Tower(config.child)
                self.connector = Connector(config.child)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),), "child": (str(path),)},
        component_architectures={"root": "Wrapper", "child": "Tower"},
        architecture="Wrapper")
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "resolved"
    assert result.candidate.component_symbol.qualified_name == "Tower"


def test_non_candidate_component_architecture_cannot_hide_exact_source(
        tmp_path):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent("""
        class Concrete:
            def __init__(self, config): pass
        class PretrainedBase: pass
        class Wrapper:
            def __init__(self, config):
                self.child = Concrete(config.child)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),), "child": (str(path),)},
        component_architectures={
            "root": "Wrapper", "child": "PretrainedBase"},
        architecture="Wrapper")
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "resolved"
    assert result.candidate.component_symbol.qualified_name == "Concrete"


def test_unrelated_from_config_factory_cannot_claim_framework_dispatch(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent("""
        from somewhere import OtherFactory

        class Child:
            def __init__(self, config): pass

        class Wrapper:
            def __init__(self, config):
                self.slot = OtherFactory.from_config(config.child)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),), "child": (str(path),)},
        component_architectures={"root": "Wrapper", "child": "Child"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "failed"
    assert result.failure_kind == "unresolved_config_construction"


def test_lookalike_auto_model_namespace_cannot_claim_framework_dispatch(
        tmp_path):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent("""
        from impostor.models.auto.modeling_auto import AutoModel

        class Child:
            def __init__(self, config): pass

        class Wrapper:
            def __init__(self, config):
                self.slot = AutoModel.from_config(config.child)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),), "child": (str(path),)},
        component_architectures={"root": "Wrapper", "child": "Child"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == "failed"
    assert result.failure_kind == "unresolved_config_construction"


@pytest.mark.parametrize(("official_path", "expected"), [
    (True, "resolved"),
    (False, "failed"),
])
def test_relative_auto_dispatch_requires_the_exact_transformers_package_path(
        tmp_path, official_path, expected):
    base = (
        tmp_path / "transformers" / "models" / "opaque"
        if official_path else tmp_path / "unrelated")
    base.mkdir(parents=True)
    path = base / "modeling_opaque.py"
    path.write_text(textwrap.dedent("""
        from ..auto import AutoModel

        class Child:
            def __init__(self, config): pass

        class Wrapper:
            def __init__(self, config):
                self.slot = AutoModel.from_config(config.child)
    """), encoding="utf-8")
    auto_path = _auto_registry_source(tmp_path)
    bundle = SourceBundle(
        source="local", files=(str(path), str(auto_path)),
        component_files={
            "root": (str(path), str(auto_path)), "child": (str(path),)},
        component_architectures={"root": "Wrapper", "child": "Child"},
        component_model_types={"child": "child_cfg"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    result = resolve_config_constructed_root(
        index, bundle, root, ("child",))
    assert result.status == expected
