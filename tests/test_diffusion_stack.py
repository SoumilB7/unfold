"""U10-B occurrence-exact diffusion stack inventory controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_stack import (
    DiffusionStackInventory,
    DiffusionStackOccurrence,
    read_diffusion_stack_inventory,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(files, architecture="Root"):
    return SourceBundle(
        source="test", files=tuple(files), architecture=architecture,
        component_files={"root": tuple(files)},
        component_architectures={"root": architecture})


def _read(tmp_path, source, *, architecture="Root", extra_files=(),
          config_guard_selector=None):
    path = _write(tmp_path, "model.py", source)
    bundle = _bundle((path, *extra_files), architecture)
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    return read_diffusion_stack_inventory(
        index, root, config_guard_selector=config_guard_selector), root, index


BASE = """
    from torch.nn import ModuleList
    class Cell:
        def forward(self, value, context=None): return value
    class Root:
        def __init__(self, config):
            self.items = ModuleList([Cell() for _ in range(config.depth)])
        def forward(self, value, context=None):
            for item in self.items:
                value = item(value, context=context)
            return value
"""


def test_symbolic_container_is_one_occurrence_not_n_layers(tmp_path):
    result, root, _index = _read(tmp_path, BASE)
    assert result.status == "incomplete"
    inventory = result.require_value()
    assert inventory.component_root == root.graph.root.occurrence
    assert len(inventory.stacks) == 1
    stack = inventory.stacks[0]
    assert stack.container.count_expression is not None
    assert tuple(part.name for part in stack.count_config_path.segments) == ("depth",)
    assert len(stack.container.element_sites) == 1
    assert len(stack.executions) == 1
    assert stack.block_occurrence.sites[-1] \
        == stack.container.element_sites[0].site_id


def test_two_containers_reusing_one_class_are_two_occurrences(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value, context=None): return value
        class Root:
            def __init__(self, config):
                self.first = ModuleList([Cell() for _ in range(config.a)])
                self.second = ModuleList([Cell() for _ in range(config.b)])
            def forward(self, value, context=None):
                for left in self.first:
                    value = left(value, context=context)
                for right in self.second:
                    value = right(value, context=context)
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    stacks = result.require_value().stacks
    assert len(stacks) == 2
    assert {item.block_symbol for item in stacks} == {stacks[0].block_symbol}
    assert len({item.block_occurrence for item in stacks}) == 2


def test_same_container_invoked_twice_is_one_occurrence_two_executions(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value, context=None): return value
        class Root:
            def __init__(self, config):
                self.items = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, value, context=None):
                for item in self.items:
                    value = item(value, context=context)
                for again in self.items:
                    value = again(value)
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    assert len(result.require_value().stacks) == 1
    assert len(result.require_value().stacks[0].executions) == 2


def test_sliced_iteration_keeps_exact_symbolic_occurrence(tmp_path):
    result, _root, _index = _read(
        tmp_path, BASE.replace("self.items:", "self.items[:2]:"))
    execution = result.require_value().stacks[0].executions[0]
    assert execution.kind == "repeated_template"
    assert execution.template.iteration_kind == "sliced"


def test_enumerated_iteration_keeps_exact_element_binding(tmp_path):
    source = BASE.replace(
        "for item in self.items:", "for number, item in enumerate(self.items):")
    result, _root, _index = _read(tmp_path, source)
    execution = result.require_value().stacks[0].executions[0]
    assert execution.template.iteration_kind == "enumerated"


def test_literal_indexed_container_call_is_exact(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value, context=None): return value
        class Root:
            def __init__(self, config):
                self.items = ModuleList([Cell(), Cell()])
            def forward(self, value, context=None):
                value = self.items[0](value, context=context)
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    assert len(result.require_value().stacks) == 1
    execution = result.require_value().stacks[0].executions[0]
    assert execution.kind == "literal_index"


def test_dynamic_index_is_unresolved_never_first_element(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value, context=None): return value
        class Root:
            def __init__(self, config):
                self.items = ModuleList([Cell(), Cell()])
            def forward(self, value, index, context=None):
                value = self.items[index](value, context=context)
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    assert result.status == "incomplete"
    assert not result.require_value().stacks
    assert result.require_value().unresolved


def test_heterogeneous_loop_is_unresolved_never_element_zero(tmp_path):
    source = BASE.replace(
        "class Root:",
        "class Other:\n        def forward(self, value, context=None): return value\n"
        "    class Root:").replace(
        "[Cell() for _ in range(config.depth)]", "[Cell(), Other()]")
    result, _root, _index = _read(tmp_path, source)
    assert result.status == "incomplete"
    assert not result.require_value().stacks
    assert result.require_value().unresolved


def test_guarded_container_constructors_remain_unresolved_rivals(tmp_path):
    source = """
        from torch.nn import ModuleList
        class A:
            def forward(self, x): return x
        class B:
            def forward(self, x): return x
        class Root:
            def __init__(self, config):
                if config.pick:
                    self.items = ModuleList([A()])
                else:
                    self.items = ModuleList([B()])
            def forward(self, x):
                for item in self.items:
                    x = item(x)
                return x
    """
    result, _root, _index = _read(tmp_path, source)
    assert result.status == "incomplete"
    assert not result.require_value().stacks
    assert result.require_value().unresolved


def test_exact_guard_partition_selects_one_rival_and_retains_proof(tmp_path):
    source = """
        from torch.nn import ModuleList
        class A:
            def forward(self, x): return x
        class B:
            def forward(self, x): return x
        class Root:
            def __init__(self, config):
                if config.pick:
                    self.items = ModuleList([A()])
                else:
                    self.items = ModuleList([B()])
            def forward(self, x):
                for item in self.items:
                    x = item(x)
                return x
    """
    selector = lambda path: (
        (True, False, "config_declared")
        if tuple(path) == ("pick",) else (False, None, ""))
    result, _root, _index = _read(
        tmp_path, source, config_guard_selector=selector)
    inventory = result.require_value()
    assert len(inventory.stacks) == 1
    stack = inventory.stacks[0]
    assert stack.block_symbol.qualified_name == "B"
    assert stack.selection is not None
    assert stack.selection.selected_branch == 1
    assert len(stack.selection.rival.records) == 2
    assert stack.selection.premises == (
        (("pick",), "config_declared", False),)
    assert not inventory.unresolved


def test_unresolved_guard_never_selects_a_container_rival(tmp_path):
    source = """
        from torch.nn import ModuleList
        class A:
            def forward(self, x): return x
        class B:
            def forward(self, x): return x
        class Root:
            def __init__(self, config):
                if config.pick:
                    self.items = ModuleList([A()])
                else:
                    self.items = ModuleList([B()])
            def forward(self, x):
                for item in self.items:
                    x = item(x)
                return x
    """
    result, _root, _index = _read(
        tmp_path, source,
        config_guard_selector=lambda _path: (False, None, ""))
    assert not result.require_value().stacks
    assert result.require_value().unresolved


def test_constructed_but_uninvoked_container_is_not_active_stack(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value): return value
        class Root:
            def __init__(self, config):
                self.items = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, value): return value
    """
    result, _root, _index = _read(tmp_path, source)
    assert result.status == "incomplete"
    assert not result.require_value().stacks
    assert result.require_value().unresolved


def test_same_loop_variable_spelling_outside_body_does_not_bind_container(
        tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value): return value
        class Root:
            def __init__(self, config):
                self.items = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, value):
                for item in self.items:
                    value = value
                value = item(value)
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    unresolved = result.require_value().unresolved
    assert not result.require_value().stacks
    assert [(item.field, item.reason) for item in unresolved] == [
        ("items", "container_execution_unobserved")]


def test_class_field_and_local_renaming_preserves_occurrence_count(tmp_path):
    source = (BASE.replace("Root", "Opaque")
              .replace("Cell", "Unit")
              .replace("items", "collection")
              .replace("item", "member")
              .replace("value", "tensor")
              .replace("context", "side"))
    result, _root, _index = _read(
        tmp_path, source, architecture="Opaque")
    assert len(result.require_value().stacks) == 1


def test_exact_call_argument_bindings_are_carried_without_lane_labels(tmp_path):
    result, _root, _index = _read(tmp_path, BASE)
    stack = result.require_value().stacks[0]
    binding = stack.executions[0].binding
    assert binding.status == "resolved"
    assert {item.formal.name for item in binding.bindings} == {"value", "context"}
    assert not hasattr(stack, "lane")
    assert not hasattr(stack, "role")


def test_symbolic_non_config_count_is_retained_without_fabricated_path(tmp_path):
    source = BASE.replace("range(config.depth)", "range(depth)").replace(
        "def __init__(self, config):", "def __init__(self, config, depth):")
    result, _root, _index = _read(tmp_path, source)
    stack = result.require_value().stacks[0]
    assert stack.count_expression.source_segment == "range(depth)"
    assert stack.count_config_path is None


def test_unrelated_config_depth_decoys_create_no_stack(tmp_path):
    source = """
        class Root:
            def __init__(self, config):
                self.depth = config.num_layers
                self.other = config.num_single_layers
            def forward(self, value): return value
    """
    result, _root, _index = _read(tmp_path, source)
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"


def test_nested_invoked_stack_has_an_exact_owner_route(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, x): return x
        class Inner:
            def __init__(self, config):
                self.items = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, x):
                for item in self.items:
                    x = item(x)
                return x
        class Root:
            def __init__(self, config): self.inner = Inner(config)
            def forward(self, x): return self.inner(x)
    """
    result, root, _index = _read(tmp_path, source)
    stack = result.require_value().stacks[0]
    assert stack.owner_occurrence != root.graph.root.occurrence
    assert len(stack.owner_route) == 1
    assert stack.owner_route[0].callee_owner_occurrence == stack.owner_occurrence


def test_root_and_nested_stacks_remain_separate_occurrences(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, x): return x
        class Inner:
            def __init__(self, config):
                self.items = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, x):
                for item in self.items: x = item(x)
                return x
        class Root:
            def __init__(self, config):
                self.inner = Inner(config)
                self.items = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, x):
                x = self.inner(x)
                for item in self.items: x = item(x)
                return x
    """
    result, _root, _index = _read(tmp_path, source)
    assert len(result.require_value().stacks) == 2
    assert len({item.owner_occurrence for item in result.require_value().stacks}) == 2


def test_stack_nested_inside_repeated_block_retains_full_occurrence_route(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Leaf:
            def forward(self, x): return x
        class Cell:
            def __init__(self): self.inner = ModuleList([Leaf()])
            def forward(self, x):
                for leaf in self.inner: x = leaf(x)
                return x
        class Root:
            def __init__(self, config):
                self.items = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, x):
                for item in self.items: x = item(x)
                return x
    """
    result, _root, _index = _read(tmp_path, source)
    stacks = result.require_value().stacks
    assert len(stacks) == 2
    nested = max(stacks, key=lambda item: len(item.owner_route))
    assert len(nested.owner_route) == 1
    assert nested.owner_route[0].block_occurrence == nested.owner_occurrence


def test_helper_constructed_symbolic_element_without_owner_join_stays_unresolved(
        tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, x): return x
        def make(): return Cell()
        class Root:
            def __init__(self, config):
                self.items = ModuleList([make() for _ in range(config.depth)])
            def forward(self, x):
                for item in self.items: x = item(x)
                return x
    """
    result, _root, _index = _read(tmp_path, source)
    assert result.status == "incomplete"
    assert not result.require_value().stacks
    assert result.require_value().unresolved
    assert result.failures[0].kind == "incomplete_graph"


def test_imported_alias_joins_only_through_exact_owner_graph_site(tmp_path):
    helper = _write(tmp_path, "parts.py", """
        class Neutral:
            def forward(self, x): return x
    """)
    source = """
        from torch.nn import ModuleList
        from parts import Neutral as Unit
        class Root:
            def __init__(self, config):
                self.items = ModuleList([Unit() for _ in range(config.depth)])
            def forward(self, x):
                for item in self.items: x = item(x)
                return x
    """
    result, _root, _index = _read(tmp_path, source, extra_files=(helper,))
    assert result.status == "incomplete"
    stack = result.require_value().stacks[0]
    assert stack.block_symbol.qualified_name == "Neutral"
    assert stack.executions[0].kind == "owner_graph_template"
    assert stack.executions[0].unresolved_source is not None


def test_construction_order_and_call_location_are_not_conflated(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value): return value
        class Root:
            def __init__(self, config):
                self.first = ModuleList([Cell()])
                self.second = ModuleList([Cell()])
            def forward(self, value):
                for later in self.second:
                    value = later(value)
                for earlier in self.first:
                    value = earlier(value)
                return value
    """
    result, _root, _index = _read(tmp_path, source)
    stacks = result.require_value().stacks
    by_field = {item.container.field: item for item in stacks}
    assert by_field["first"].container.source_order \
        < by_field["second"].container.source_order
    assert by_field["second"].executions[0].call.span.line \
        < by_field["first"].executions[0].call.span.line
    assert not hasattr(stacks[0], "execution_order")


def test_broken_sibling_source_blocks_inventory(tmp_path):
    broken = _write(tmp_path, "broken.py", "class Root(:\n")
    result, root, _index = _read(tmp_path, BASE, extra_files=(broken,))
    assert root.status == "failed"
    assert result.status == "failed"
    assert result.failures[0].kind == "parse_failure"


def test_missing_root_is_typed_failure(tmp_path):
    result, root, _index = _read(
        tmp_path, "class Other:\n    pass\n", architecture="Missing")
    assert root.status == "absent"
    assert result.status == "failed"
    assert result.failures[0].kind == "missing_source"


def test_inventory_rejects_duplicate_stack_identity(tmp_path):
    result, _root, _index = _read(tmp_path, BASE)
    inventory = result.require_value()
    with pytest.raises(ValueError, match="unique"):
        DiffusionStackInventory(
            inventory.component_root,
            (inventory.stacks[0], inventory.stacks[0]), inventory.unresolved)


def test_occurrence_rejects_wrong_component_root(tmp_path):
    result, _root, _index = _read(tmp_path, BASE)
    stack = result.require_value().stacks[0]
    wrong = replace(stack.component_root, sites=stack.block_occurrence.sites)
    with pytest.raises(ValueError, match="component root|owner route"):
        DiffusionStackOccurrence(
            wrong, stack.owner_occurrence, stack.owner_symbol,
            stack.container, stack.block_occurrence, stack.block_symbol,
            stack.executions, stack.owner_route)


@pytest.mark.parametrize("witness,positive,unresolved", [
    ("auraflow-v0-3", 2, 0),
    ("cogvideox-5b", 1, 0),
    ("flux-2-dev", 2, 0),
    ("fluxtransformer2dmodel", 2, 2),
    ("hunyuanvideo", 1, 2),
    ("ltx-video", 1, 0),
    ("lumina-image-2-0", 3, 7),
    ("mochi-1-preview", 1, 4),
    ("pixart-sigma-xl-2-1024-ms", 0, 1),
    ("prxpixel-t2i", 1, 1),
    ("qwen-image", 1, 3),
    ("sana-1600m-1024px-diffusers", 1, 2),
    ("stable-diffusion-3-5-large", 0, 1),
    ("stable-diffusion-xl-base-1-0", 0, 4),
    ("wan2-2-t2v-a14b-diffusers", 1, 1),
])
def test_real_diffusion_witness_occurrence_inventory_matrix(
        witness, positive, unresolved):
    import json
    import pathlib

    import model_unfolder as mu
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    data = json.loads((corpus / f"{witness}.json").read_text())
    context = ParseContext.build(_coerce(data.get("config") or data))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    result = read_diffusion_stack_inventory(index, root)
    assert result.status == "incomplete"
    inventory = result.require_value()
    assert len(inventory.stacks) == positive
    assert len(inventory.unresolved) == unresolved
    assert inventory.stacks or inventory.unresolved
    assert all(stack.executions for stack in inventory.stacks)
    assert len({(stack.owner_occurrence, stack.container.record,
                stack.block_occurrence) for stack in inventory.stacks}) \
        == len(inventory.stacks)
    fields = {stack.container.field for stack in inventory.stacks}
    unresolved_fields = {item.field for item in inventory.unresolved}
    if witness == "hunyuanvideo":
        # The exact nested token-refiner is proven. The two root stacks each
        # have real config-guarded block-class rivals, which U10-B preserves
        # rather than selecting from model/config identity.
        assert fields == {"refiner_blocks"}
        assert unresolved_fields == {
            "transformer_blocks", "single_transformer_blocks"}
    elif witness == "lumina-image-2-0":
        # One block class is constructed at three separate stack addresses;
        # class equality must never collapse those occurrences.
        assert fields == {"layers", "context_refiner", "noise_refiner"}
        assert len({stack.block_symbol for stack in inventory.stacks}) == 1
        assert len({stack.block_occurrence for stack in inventory.stacks}) == 3
    elif witness == "stable-diffusion-xl-base-1-0":
        assert not inventory.stacks
        assert unresolved_fields == {"down_blocks", "up_blocks"}
    elif witness in {
            "pixart-sigma-xl-2-1024-ms",
            "stable-diffusion-3-5-large"}:
        assert not inventory.stacks
        assert unresolved_fields == {"transformer_blocks"}
