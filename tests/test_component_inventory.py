"""U9-A: active nested components require exact construction ownership."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.component_inventory import (
    ComponentOwnerEntry,
    resolve_component_inventory,
)
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle


def _bundle(tmp_path, body, *, components=("vision_config",), root="Wrapper"):
    path = tmp_path / "modeling.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    files = {"root": (str(path),)}
    architectures = {"root": root}
    for component in components:
        files[component] = (str(path),)
        architectures[component] = "Tower"
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture=root,
        component_files=files, component_architectures=architectures,
    )
    return bundle, ParseContext(bundle).program_index()


def test_exact_constructed_component_is_active(tmp_path):
    bundle, index = _bundle(tmp_path, """
        class Tower:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                self.tower = Tower(config.vision_config)
    """)
    inventory = resolve_component_inventory(index, bundle)
    entry = inventory.entry("vision_config")
    assert entry.status == "active"
    assert entry.component_root.config_path == ("vision_config",)
    assert entry.component_root.installation_field == "tower"
    assert [item.component_key for item in inventory.active] == [
        "root", "vision_config"]


def test_parse_context_memoizes_one_index_bound_inventory(tmp_path):
    bundle, _ = _bundle(tmp_path, """
        class Tower:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                self.tower = Tower(config.vision_config)
    """)
    context = ParseContext(bundle)
    first = context.component_inventory()
    second = context.component_inventory()
    assert first is second
    assert first.root.graph.root.symbol \
        == context.program_index().class_by_symbol(
            first.root.graph.root.symbol).symbol


def test_declared_but_unconstructed_component_is_inventory_only(tmp_path):
    bundle, index = _bundle(tmp_path, """
        class Tower: pass
        class Wrapper:
            def __init__(self, config):
                self.other = object()
    """)
    entry = resolve_component_inventory(index, bundle).entry("vision_config")
    assert entry.status == "declared_unused"
    assert entry.component_root is None


def test_same_component_constructed_twice_preserves_rivals(tmp_path):
    bundle, index = _bundle(tmp_path, """
        class Tower:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                self.left = Tower(config.vision_config)
                self.right = Tower(config.vision_config)
    """)
    entry = resolve_component_inventory(index, bundle).entry("vision_config")
    assert entry.status == "ambiguous"
    assert len(entry.rival_spans) == 2


def test_broken_root_makes_nested_ownership_unavailable(tmp_path):
    healthy = tmp_path / "healthy.py"
    broken = tmp_path / "broken.py"
    healthy.write_text("class Wrapper: pass\nclass Tower: pass\n", encoding="utf-8")
    broken.write_text("class Broken(:\n", encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(healthy), str(broken)), architecture="Wrapper",
        component_files={
            "root": (str(healthy), str(broken)),
            "vision_config": (str(healthy),),
        },
        component_architectures={"root": "Wrapper", "vision_config": "Tower"},
    )
    inventory = resolve_component_inventory(
        ParseContext(bundle).program_index(), bundle)
    assert inventory.entry("root").status == "failed"
    nested = inventory.entry("vision_config")
    assert nested.status == "unavailable"
    assert nested.failure_kind == "root_unresolved"


def test_pipeline_slot_is_not_relabelled_as_nested_component(tmp_path):
    bundle, index = _bundle(tmp_path, """
        class Tower: pass
        class Wrapper: pass
    """, components=("text_encoder",))
    bundle = SourceBundle(
        **{**bundle.__dict__, "pipeline_components": ("text_encoder",)})
    inventory = resolve_component_inventory(index, bundle)
    assert inventory.entry("text_encoder") is None


def test_pipeline_slot_descendant_is_not_relabelled_as_root_child(tmp_path):
    bundle, _ = _bundle(tmp_path, """
        class Tower: pass
        class Wrapper: pass
    """, components=("text_encoder.vision_config",))
    bundle = SourceBundle(
        **{**bundle.__dict__, "pipeline_components": ("text_encoder",)})
    inventory = resolve_component_inventory(
        ParseContext(bundle).program_index(), bundle)
    assert inventory.entry("text_encoder.vision_config") is None


def test_two_sibling_paths_do_not_launder_ownership(tmp_path):
    bundle, index = _bundle(tmp_path, """
        class Tower:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                self.left = Tower(config.left)
    """, components=("left", "right"))
    inventory = resolve_component_inventory(index, bundle)
    assert inventory.entry("left").status == "active"
    assert inventory.entry("right").status == "declared_unused"


def test_entry_closure_rejects_active_without_proof():
    try:
        ComponentOwnerEntry("vision_config", ("vision_config",), "Tower", "active")
    except ValueError as exc:
        assert "exact component root" in str(exc)
    else:
        raise AssertionError("active entry accepted without ownership proof")
