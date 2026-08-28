"""U11-B exact stage-construction evidence controls."""
from __future__ import annotations

import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_stage_construction import (
    StageClassCandidate,
    UNetStageConstructionInventory,
    read_unet_stage_construction,
)


ROOT = """
    from torch.nn import ModuleList
    from .factory import build

    class Root:
        def __init__(self, config):
            self.left = ModuleList([])
            self.right = ModuleList([])
            for token in config.left_types:
                item = build(token)
                self.left.append(item)
            for token in config.right_types:
                item = build(token)
                self.right.append(item)

        def forward(self, value):
            saved = (value,)
            for first in self.left:
                value, branch = first(value)
                saved += branch
            for second in self.right:
                side = saved[-1:]
                value = second(value, side)
            return value
"""


FACTORY = """
    class Alpha:
        def forward(self, value, side=None): return value, (value,)

    class Beta:
        def forward(self, value, side=None): return value, (value,)

    def build(token):
        if token == "first":
            return Alpha()
        return Beta()
"""


def _write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(tmp_path, root=ROOT, factory=FACTORY, *, extra=()):
    package = tmp_path / "pkg"
    _write(package / "__init__.py", "")
    root_path = _write(package / "root.py", root)
    _write(package / "factory.py", factory)
    for name, source in extra:
        _write(package / name, source)
    return SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (root_path,)},
        component_architectures={"root": "Root"},
        import_roots={"root": (SourceImportRoot("pkg", str(package)),)},
    )


def _read(bundle):
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root)
    assert topology.status == "incomplete"
    assert topology.require_value().kind == "u_shaped"
    result = read_unet_stage_construction(
        index, bundle, root, topology.require_value())
    return result, root, topology, index


def test_exact_imported_factory_branches_feed_both_u10_containers(tmp_path):
    result, _root, topology, index = _read(_bundle(tmp_path))
    assert result.status == "incomplete"
    inventory = result.require_value()
    assert isinstance(inventory, UNetStageConstructionInventory)
    assert len(inventory.stages) == 2
    assert len(inventory.index.source_nodes) == len(index.source_nodes) + 1
    assert [stage.topology_stage.field for stage in inventory.stages] == [
        topology.require_value().stages[0].field,
        topology.require_value().stages[1].field,
    ]
    for stage in inventory.stages:
        assert stage.producer_call.callee.source_segment == "build"
        assert stage.storage_call.callee.name == "append"
        assert {candidate.symbol.qualified_name for candidate in stage.candidates} \
            == {"Alpha", "Beta"}
        assert all(candidate.import_chain
                   and candidate.import_chain[0].call == stage.producer_call
                   for candidate in stage.candidates)
        assert any(candidate.guard for candidate in stage.candidates)
        assert not stage.issues
    assert inventory.stages[0].producer_call.span != \
        inventory.stages[1].producer_call.span


def test_factory_token_is_retained_as_operand_not_promoted_to_mechanism(tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    for stage in result.require_value().stages:
        assert stage.producer_call.args[0].source_segment == "token"
        assert not hasattr(stage, "attention")
        assert not hasattr(stage, "resnet")
        assert not hasattr(stage, "stage_kind")


def test_complete_class_field_factory_and_local_rename_preserves_evidence(tmp_path):
    root = (ROOT.replace("Root", "Opaque")
            .replace("build", "assemble")
            .replace("left", "alpha_store")
            .replace("right", "omega_store")
            .replace("first", "one")
            .replace("second", "two")
            .replace("item", "unit"))
    factory = (FACTORY.replace("build", "assemble")
               .replace("Alpha", "One")
               .replace("Beta", "Two"))
    bundle = _bundle(tmp_path, root, factory)
    bundle = SourceBundle(
        source=bundle.source, architecture="Opaque",
        component_files=bundle.component_files,
        component_architectures={"root": "Opaque"},
        import_roots=bundle.import_roots)
    result, *_ = _read(bundle)
    assert [{c.symbol.qualified_name for c in stage.candidates}
            for stage in result.require_value().stages] == [
                {"One", "Two"}, {"One", "Two"}]


def test_unimported_same_name_decoy_cannot_enter(tmp_path):
    result, *_ = _read(_bundle(
        tmp_path, extra=(("decoy.py", "class Alpha: pass\nclass Beta: pass"),)))
    paths = {candidate.symbol.source.canonical_path
             for stage in result.require_value().stages
             for candidate in stage.candidates}
    assert len(paths) == 1
    assert next(iter(paths)).endswith("factory.py")


def test_direct_imported_class_carries_the_exact_import_proof(tmp_path):
    root = (ROOT.replace("from .factory import build",
                         "from .factory import Alpha")
            .replace("build(token)", "Alpha()"))
    result, *_ = _read(_bundle(tmp_path, root))
    for stage in result.require_value().stages:
        assert {candidate.symbol.qualified_name for candidate in stage.candidates} \
            == {"Alpha"}
        candidate = stage.candidates[0]
        assert candidate.import_chain[-1].imported_symbol == candidate.symbol


def test_exact_reexport_chain_reaches_factory_without_package_search(tmp_path):
    root = ROOT.replace("from .factory import build", "from .api import build")
    result, *_ = _read(_bundle(
        tmp_path, root, extra=(("api.py", "from .factory import build"),)))
    for stage in result.require_value().stages:
        assert {candidate.symbol.qualified_name for candidate in stage.candidates} \
            == {"Alpha", "Beta"}
        assert len(stage.candidates[0].import_chain[0].source_chain) == 2


def test_exact_local_helper_return_chain_is_preserved(tmp_path):
    factory = FACTORY.replace(
        "def build(token):",
        "def helper(token):").replace(
        "        return Beta()",
        "        return Beta()\n\n    def build(token):\n        return helper(token)")
    result, *_ = _read(_bundle(tmp_path, factory=factory))
    for stage in result.require_value().stages:
        assert all(tuple(symbol.qualified_name for symbol in candidate.factory_chain)
                   == ("build", "helper") for candidate in stage.candidates)


def test_dynamic_factory_dispatch_remains_typed_unknown(tmp_path):
    factory = """
        class Alpha: pass
        TABLE = {"a": Alpha}
        def build(token):
            return TABLE[token]()
    """
    result, *_ = _read(_bundle(tmp_path, factory=factory))
    for stage in result.require_value().stages:
        assert not stage.candidates
        assert stage.issues[0].kind == "dynamic_constructor"


def test_broken_exact_factory_is_typed_and_never_replaced_by_decoy(tmp_path):
    result, *_ = _read(_bundle(
        tmp_path, factory="def build(:\n    pass",
        extra=(("decoy.py", FACTORY),)))
    assert result.status == "incomplete"
    assert all(not stage.candidates for stage in result.require_value().stages)
    assert all(stage.issues[0].kind == "parse_failure"
               for stage in result.require_value().stages)


def test_guarded_import_is_typed_incomplete(tmp_path):
    root = ROOT.replace(
        "from .factory import build",
        "if FLAG:\n        from .factory import build")
    result, *_ = _read(_bundle(tmp_path, root))
    assert all(not stage.candidates for stage in result.require_value().stages)
    assert all(stage.issues[0].kind == "import_incomplete"
               for stage in result.require_value().stages)


def test_config_path_is_cited_only_when_inside_exact_producer_call(tmp_path):
    root = ROOT.replace("build(token)", "build(config.selected)")
    result, *_ = _read(_bundle(tmp_path, root))
    for stage in result.require_value().stages:
        assert len(stage.config_paths) == 1
        assert tuple(segment.name for segment in stage.config_paths[0].segments) \
            == ("selected",)


def test_sibling_config_read_cannot_be_laundered_into_factory_operand(tmp_path):
    root = ROOT.replace(
        "for token in config.left_types:",
        "seen = config.selected\n            for token in config.left_types:")
    result, *_ = _read(_bundle(tmp_path, root))
    assert not result.require_value().stages[0].config_paths


def test_guarded_rival_producers_for_one_append_are_not_guessed(tmp_path):
    root = ROOT.replace(
        "            for token in config.left_types:\n"
        "                item = build(token)\n"
        "                self.left.append(item)",
        "            for token in config.left_types:\n"
        "                if config.flag:\n"
        "                    item = build(token)\n"
        "                else:\n"
        "                    item = build(token)\n"
        "                self.left.append(item)")
    bundle = _bundle(tmp_path, root)
    index = build_program_index(bundle)
    resolved = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, resolved).require_value()
    result = read_unet_stage_construction(index, bundle, resolved, topology)
    assert result.status == "incomplete"
    inventory = result.require_value()
    assert len(inventory.unresolved_stages) == 1
    assert inventory.unresolved_stages[0].issues[0].kind == "storage_route_rival"


def test_symbolic_comprehension_is_one_template_not_n_occurrences(tmp_path):
    source = """
        from torch.nn import ModuleList
        class Cell:
            def forward(self, value, side=None): return value, (value,)
        class Root:
            def __init__(self, config):
                self.alpha = ModuleList([Cell() for _ in range(config.depth)])
                self.omega = ModuleList([Cell() for _ in range(config.depth)])
            def forward(self, value):
                saved = (value,)
                for first in self.alpha:
                    value, branch = first(value)
                    saved += branch
                for second in self.omega:
                    side = saved[-1:]
                    value = second(value, side)
                return value
    """
    result, *_ = _read(_bundle(tmp_path, source))
    assert len(result.require_value().stages) == 2
    assert all(len(stage.candidates) == 1
               for stage in result.require_value().stages)
    assert all(stage.direct_site is not None
               for stage in result.require_value().stages)


def test_config_stage_lists_without_u10_shape_cannot_enter_u11(tmp_path):
    source = """
        class Root:
            def __init__(self, config): self.claim = config.left_types
            def forward(self, value): return value
    """
    bundle = _bundle(tmp_path, source)
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root)
    assert topology.status == "failed"
    result = read_unet_stage_construction(
        index, bundle, root, topology.value)
    assert result.status == "failed"
    assert result.failures[0].kind == "out_of_owner"


def test_result_rejects_candidate_without_an_address_chain(tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    candidate = result.require_value().stages[0].candidates[0]
    with pytest.raises(TypeError):
        StageClassCandidate(candidate.symbol, (), call=candidate.call,
                            returned_by=candidate.returned_by)


def test_cross_source_candidate_without_import_proof_is_rejected(tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    candidate = result.require_value().stages[0].candidates[0]
    with pytest.raises(ValueError):
        StageClassCandidate(
            result.owner.root, candidate.factory_chain, (),
            call=candidate.call, returned_by=candidate.returned_by)


def test_inventory_rejects_a_foreign_program_index(tmp_path):
    result, *_ = _read(_bundle(tmp_path / "one"))
    foreign, *_ = _read(_bundle(tmp_path / "two"))
    with pytest.raises(ValueError):
        replace(result.require_value(), index=foreign.require_value().index)


def test_topology_from_a_different_root_is_rejected(tmp_path):
    bundle = _bundle(tmp_path / "one")
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    other, _other_root, other_topology, _ = _read(_bundle(tmp_path / "two"))
    assert other.has_value
    result = read_unet_stage_construction(
        index, bundle, root, other_topology.require_value())
    assert result.status == "failed"
    assert result.failures[0].kind == "out_of_owner"


def test_real_sdxl_factories_resolve_as_conditional_candidates():
    import diffusers

    package = Path(diffusers.__file__).resolve().parent
    source = package / "models" / "unets" / "unet_2d_condition.py"
    bundle = SourceBundle(
        source="test", architecture="UNet2DConditionModel",
        component_files={"root": (str(source),)},
        component_architectures={"root": "UNet2DConditionModel"},
        import_roots={"root": (SourceImportRoot("diffusers", str(package)),)},
    )
    result, *_ = _read(bundle)
    assert result.status == "incomplete"
    inventory = result.require_value()
    assert len(inventory.stages) == 2
    names = [{candidate.symbol.qualified_name
              for candidate in stage.candidates}
             for stage in inventory.stages]
    assert "CrossAttnDownBlock2D" in names[0]
    assert "CrossAttnUpBlock2D" in names[1]
    assert all(candidate.symbol.source.canonical_path.endswith(
        "diffusers/models/unets/unet_2d_blocks.py")
        for stage in inventory.stages for candidate in stage.candidates)
