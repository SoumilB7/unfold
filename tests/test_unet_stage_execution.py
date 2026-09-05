"""U11-C partial exact stage-execution graph controls."""
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
    DirectFieldInvocationAddress,
    read_direct_field_construction,
    read_unet_stage_construction,
)
from model_unfolder.evidence.unet_stage_execution import (
    DirectStageExecution,
    StageDataflowEdge,
    read_unet_stage_execution,
)


ROOT = """
    from torch.nn import ModuleList
    from .factory import build

    class Root:
        def __init__(self, config):
            self.alpha = ModuleList([])
            self.omega = ModuleList([])
            self.bridge = build(config.bridge_type)
            for token in config.alpha_types:
                item = build(token)
                self.alpha.append(item)
            for token in config.omega_types:
                item = build(token)
                self.omega.append(item)

        def forward(self, value):
            saved = (value,)
            for first in self.alpha:
                value, branch = first(value)
                saved += branch
            value = self.bridge(value)
            for second in self.omega:
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
        if token == "one": return Alpha()
        return Beta()
"""


def _write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(tmp_path, root=ROOT, factory=FACTORY):
    package = tmp_path / "pkg"
    _write(package / "__init__.py", "")
    root_path = _write(package / "root.py", root)
    _write(package / "factory.py", factory)
    return SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (root_path,)},
        component_architectures={"root": "Root"},
        import_roots={"root": (SourceImportRoot("pkg", str(package)),)},
    )


def _read(bundle):
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(
        index, bundle, root, topology).require_value()
    result = read_unet_stage_execution(construction, bundle, root)
    return result, construction, topology, root


def test_constructed_direct_field_inside_u_interval_is_preserved(tmp_path):
    result, construction, topology, _root = _read(_bundle(tmp_path))
    assert result.status == "incomplete"
    graph = result.require_value()
    assert graph.construction == construction
    assert graph.topology == topology
    assert len(graph.repeated) == 2
    assert len(graph.direct) == 1
    direct = graph.direct[0]
    assert direct.node_id.field == "bridge"
    assert {candidate.symbol.qualified_name
            for item in direct.construction.constructions
            for candidate in item.candidates} == {"Alpha", "Beta"}


def test_direct_field_spelling_assigns_no_architectural_role(tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    direct = result.require_value().direct[0]
    assert not hasattr(direct, "mid")
    assert not hasattr(direct, "bottleneck")
    assert not hasattr(direct, "stage_kind")
    assert direct.node_id.kind == "direct"


def test_complete_field_and_class_rename_preserves_graph_shape(tmp_path):
    root = (ROOT.replace("Root", "Opaque")
            .replace("bridge", "centerpiece")
            .replace("alpha", "left_store")
            .replace("omega", "right_store")
            .replace("first", "one")
            .replace("second", "two"))
    factory = FACTORY.replace("Alpha", "One").replace("Beta", "Two")
    bundle = _bundle(tmp_path, root, factory)
    bundle = SourceBundle(
        source="test", architecture="Opaque",
        component_files=bundle.component_files,
        component_architectures={"root": "Opaque"},
        import_roots=bundle.import_roots)
    result, *_ = _read(bundle)
    graph = result.require_value()
    assert len(graph.repeated) == 2
    assert [node.node_id.kind for node in graph.direct] == ["direct"]
    assert {candidate.symbol.qualified_name
            for item in graph.direct[0].construction.constructions
            for candidate in item.candidates} == {"One", "Two"}


def test_unconstructed_self_method_between_loops_is_not_a_stage(tmp_path):
    root = ROOT.replace(
        "value = self.bridge(value)",
        "value = self.helper(value)\n            value = self.bridge(value)").replace(
        "        def forward(self, value):",
        "        def helper(self, value): return value\n\n"
        "        def forward(self, value):")
    result, *_ = _read(_bundle(tmp_path, root))
    assert [node.node_id.field for node in result.require_value().direct] == [
        "bridge"]


def test_helper_only_field_assignment_stays_unresolved_until_helper_execution_is_proven(
        tmp_path):
    root = ROOT.replace(
        "self.bridge = build(config.bridge_type)",
        "self.prepare(config)").replace(
        "        def forward(self, value):",
        "        def prepare(self, config):\n"
        "            self.bridge = build(config.bridge_type)\n\n"
        "        def forward(self, value):")
    result, *_ = _read(_bundle(tmp_path, root))
    graph = result.require_value()
    assert not graph.direct
    assert any(item.kind == "direct_construction_incomplete"
               for item in graph.unresolved)


def test_invoked_field_with_broken_factory_keeps_execution_but_unknown_construction(
        tmp_path):
    result, *_ = _read(_bundle(tmp_path, factory="def build(:\n    pass"))
    graph = result.require_value()
    assert len(graph.direct) == 1
    construction = graph.direct[0].construction.constructions[0]
    assert not construction.candidates
    assert construction.issues[0].kind == "parse_failure"


def test_constructed_field_called_outside_interval_is_not_a_direct_node(tmp_path):
    root = ROOT.replace(
        "saved = (value,)",
        "value = self.bridge(value)\n            saved = (value,)").replace(
        "            value = self.bridge(value)\n            for second",
        "            for second")
    result, *_ = _read(_bundle(tmp_path, root))
    assert not result.require_value().direct


def test_constructed_call_after_unconditional_return_is_not_promoted(tmp_path):
    root = ROOT.replace(
        "value = self.bridge(value)",
        "return value\n            value = self.bridge(value)")
    result, *_ = _read(_bundle(tmp_path, root))
    graph = result.require_value()
    assert not graph.direct
    assert any(item.kind == "direct_call_tainted"
               and "unreachable_after_return" in item.detail
               for item in graph.unresolved)


def test_constructed_call_in_unsupported_expression_is_not_promoted(tmp_path):
    root = ROOT.replace(
        "value = self.bridge(value)",
        "value = flag and self.bridge(value)")
    result, *_ = _read(_bundle(tmp_path, root))
    graph = result.require_value()
    assert not graph.direct
    assert any(item.kind == "direct_call_tainted"
               and "unsupported_execution_region" in item.detail
               for item in graph.unresolved)


def test_two_constructed_fields_inside_interval_are_both_preserved(tmp_path):
    root = ROOT.replace(
        "self.bridge = build(config.bridge_type)",
        "self.bridge = build(config.bridge_type)\n"
        "            self.other = build(config.other_type)").replace(
        "value = self.bridge(value)",
        "value = self.bridge(value)\n            value = self.other(value)")
    result, *_ = _read(_bundle(tmp_path, root))
    assert [node.node_id.field for node in result.require_value().direct] == [
        "bridge", "other"]


def test_guarded_calls_to_same_field_remain_two_exact_invocation_nodes(tmp_path):
    root = ROOT.replace(
        "value = self.bridge(value)",
        "if flag:\n                value = self.bridge(value)\n"
        "            else:\n                value = self.bridge(value)")
    result, *_ = _read(_bundle(tmp_path, root))
    assert len(result.require_value().direct) == 2
    assert all(node.call.guard for node in result.require_value().direct)
    assert len({node.node_id for node in result.require_value().direct}) == 2


def test_config_declared_field_without_execution_cannot_enter_graph(tmp_path):
    root = ROOT.replace("            value = self.bridge(value)\n", "")
    result, *_ = _read(_bundle(tmp_path, root))
    assert not result.require_value().direct


def test_u10_skip_route_is_the_only_positive_interstage_edge(tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    graph = result.require_value()
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.proof_kind == "u10_skip_route"
    assert edge.route == graph.topology.skip_route
    assert edge.route.producer in edge.source.topology_stage.calls
    assert edge.route.consumer in edge.target.topology_stage.calls
    assert {item.kind for item in graph.unresolved} >= {
        "whole_callable_open", "direct_stage_order_open"}
    assert not hasattr(graph, "execution_order")
    assert not hasattr(graph, "happens_before")


def test_two_sequential_calls_to_same_field_remain_two_occurrences(tmp_path):
    root = ROOT.replace(
        "value = self.bridge(value)",
        "value = self.bridge(value)\n            value = self.bridge(value)")
    result, *_ = _read(_bundle(tmp_path, root))
    graph = result.require_value()
    assert len(graph.direct) == 2
    assert graph.direct[0].node_id.source_position < \
        graph.direct[1].node_id.source_position


def test_open_cfg_cannot_be_forged_into_a_graph_without_unresolved_relations(
        tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    with pytest.raises(ValueError):
        replace(result.require_value(), unresolved=())


def test_graph_cannot_replace_or_duplicate_the_u10_skip_edge(tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    graph = result.require_value()
    with pytest.raises(ValueError):
        replace(graph, edges=())
    with pytest.raises(ValueError):
        replace(graph, edges=(graph.edges[0], graph.edges[0]))


def test_unresolved_relation_cannot_cite_a_foreign_node(tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    graph = result.require_value()
    forged = replace(
        graph.unresolved[0],
        nodes=(replace(graph.repeated[0].node_id, field="foreign"),))
    with pytest.raises(ValueError):
        replace(graph, unresolved=(forged, *graph.unresolved[1:]))


def test_reverse_skip_edge_cannot_be_forged(tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    graph = result.require_value()
    with pytest.raises(ValueError):
        StageDataflowEdge(
            graph.repeated[-1], graph.repeated[0],
            "u10_skip_route", graph.topology.skip_route)


def test_direct_node_field_must_match_calls_and_construction(tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    direct = result.require_value().direct[0]
    with pytest.raises(ValueError):
        DirectStageExecution(
            replace(direct.node_id, field="forged"),
            direct.call, direct.construction)


def test_raw_familiar_field_spelling_cannot_enter_construction_reader(tmp_path):
    result, _construction, _topology, root = _read(_bundle(tmp_path))
    graph = result.require_value()
    with pytest.raises(TypeError):
        read_direct_field_construction(
            graph.index, _bundle(tmp_path / "unused"), root, "mid_block")


def test_invocation_address_rejects_a_forged_field_spelling(tmp_path):
    result, *_ = _read(_bundle(tmp_path))
    address = result.require_value().direct[0].construction.address
    with pytest.raises(ValueError):
        DirectFieldInvocationAddress(
            address.owner, "mid_block", address.calls,
            address.earlier_stage, address.later_stage)


def test_root_mismatch_is_typed_failure(tmp_path):
    result, construction, _topology, _root = _read(_bundle(tmp_path / "one"))
    assert result.has_value
    other_bundle = _bundle(tmp_path / "two")
    other_index = build_program_index(other_bundle)
    other_root = resolve_component_root(other_index, other_bundle, "root")
    mismatch = read_unet_stage_execution(
        construction, other_bundle, other_root)
    assert mismatch.status == "failed"
    assert mismatch.failures[0].kind == "out_of_owner"


def test_real_sdxl_selects_exact_interloop_constructed_field_only():
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
    graph = result.require_value()
    assert len(graph.direct) == 2  # guarded cross-attn and non-cross invocations
    assert {node.node_id.field for node in graph.direct} == {"mid_block"}
    assert {candidate.symbol.qualified_name
            for node in graph.direct
            for item in node.construction.constructions
            for candidate in item.candidates} == {
                "UNetMidBlock2D", "UNetMidBlock2DCrossAttn",
                "UNetMidBlock2DSimpleCrossAttn"}
