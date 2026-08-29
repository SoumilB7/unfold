"""U11-E1 exact nested attention/FFN mechanism controls."""
from __future__ import annotations

import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from model_unfolder.evidence.attention_lane import (
    FrameworkAttentionLaneEvidence,
    framework_attention_lane_positive_proof_in_graph,
)
from model_unfolder.evidence.attention_invocation_role import (
    framework_attention_invocation_role,
)
from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_owner_graph,
)
from model_unfolder.evidence.container_inventory import (
    resolve_container_inventory_in_graph,
)
from model_unfolder.evidence.constructor_values import (
    canonical_construction_target,
    constructor_frame,
    resolve_effective_constructor_parameter,
)
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.execution_flow import (
    resolve_addressed_invocations_in_graph,
)
from model_unfolder.evidence.import_source import (
    canonical_called_import_target,
    resolve_called_import_source,
)
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.selected_composite_ffn import (
    selected_composite_ffn_mechanism,
)
from model_unfolder.evidence.unet_cell_mechanism import read_unet_cell_mechanisms
from model_unfolder.evidence.unet_nested_mechanism import (
    AlternativeNestedOccurrenceId,
    read_unet_nested_mechanisms,
)
from model_unfolder.evidence.unet_stage_cells import read_unet_stage_cells
from model_unfolder.evidence.unet_stage_construction import (
    read_unet_stage_construction,
)
from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution


ROOT = """
    from torch.nn import ModuleList
    from .stages import build

    class Root:
        def __init__(self, config):
            self.down = ModuleList([build(config.kind)])
            self.bridge = build(config.kind)
            self.up = ModuleList([build(config.kind)])
        def forward(self, value, side):
            saved = (value,)
            for stage in self.down:
                value, branch = stage(value, side)
                saved += branch
            value = self.bridge(value, side)
            for stage in self.up:
                value = stage(value, saved[-1:], side)
            return value
"""


STAGES = """
    from torch.nn import ModuleList
    from .cells import NestedCell, OpaqueCell

    class Stage:
        def __init__(self):
            self.units = ModuleList([NestedCell(), OpaqueCell()])
        def forward(self, value, skip=None, side=None):
            for unit in self.units:
                value = unit(value, side)
            return value, (value,)

    def build(token):
        return Stage()
"""


CELLS = """
    from torch import nn
    from torch.nn import functional as F

    class MathLane:
        def __init__(self):
            self.q = nn.Linear(8, 8)
            self.k = nn.Linear(8, 8)
            self.v = nn.Linear(8, 8)
            self.o = nn.Linear(8, 8)
        def forward(self, value, other=None):
            query = self.q(value)
            source = value if other is None else other
            key = self.k(source)
            val = self.v(source)
            mixed = F.scaled_dot_product_attention(query, key, val)
            return self.o(mixed)

    class SplitPath:
        def __init__(self):
            self.a = nn.Linear(8, 16)
            self.b = nn.Linear(8, 16)
            self.c = nn.Linear(16, 8)
        def forward(self, value):
            return self.c(F.silu(self.a(value)) * self.b(value))

    class InnerBlock:
        def __init__(self):
            self.first = MathLane()
            self.second = MathLane()
            self.third = SplitPath()
        def forward(self, value, side):
            value = self.first(value)
            value = self.second(value, side)
            return self.third(value)

    class NestedCell:
        def __init__(self):
            self.stack = nn.ModuleList([InnerBlock()])
        def forward(self, value, side):
            for item in self.stack:
                value = item(value, side)
            return value

    class OpaqueCell:
        def forward(self, value, side):
            return value
"""


def _write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(tmp_path, *, stages=STAGES, cells=CELLS):
    package = tmp_path / "pkg"
    _write(package / "__init__.py", "")
    root = _write(package / "root.py", ROOT)
    _write(package / "stages.py", stages)
    _write(package / "cells.py", cells)
    return SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (root,)},
        component_architectures={"root": "Root"},
        import_roots={"root": (SourceImportRoot("pkg", str(package)),)},
    )


def _read(bundle):
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root).require_value()
    stages = read_unet_stage_construction(
        index, bundle, root, topology).require_value()
    execution = read_unet_stage_execution(stages, bundle, root).require_value()
    cells = read_unet_stage_cells(execution, bundle).require_value()
    mechanisms = read_unet_cell_mechanisms(cells).require_value()
    return read_unet_nested_mechanisms(mechanisms)


def _for_parent(result, name="NestedCell"):
    value = result.require_value()
    parents = tuple(
        item.occurrence_id for item in value.cells.mechanisms
        if item.occurrence_id.symbol.qualified_name == name)
    assert parents
    parent = parents[0]
    return parent, tuple(
        item for item in value.mechanisms
        if (item.occurrence_id.parent
            if hasattr(item.occurrence_id, "parent")
            else item.occurrence_id.alternative.parent) == parent)


def test_nested_modulelist_reaches_exact_attention_and_ffn_children(tmp_path):
    result = _read(_bundle(tmp_path))
    assert result.status == "incomplete"
    _parent, rows = _for_parent(result)
    assert [item.kind for item in rows] == ["attention", "attention", "ffn"]
    assert [item.occurrence_id.local.sites[-1].owner.qualified_name
            for item in rows] == ["InnerBlock", "InnerBlock", "InnerBlock"]
    assert rows[0].attention.protocol == "scaled_dot_product_attention"
    assert rows[-1].ffn.gated is True
    assert rows[-1].ffn.projection_mode == "split"


def test_call_inputs_retain_formal_origins_without_self_cross_labels(tmp_path):
    _parent, rows = _for_parent(_read(_bundle(tmp_path)))
    first, second = rows[:2]
    assert [item.roots for item in first.inputs] == [("value",)]
    assert [item.roots for item in second.inputs] == [("value",), ("side",)]
    assert not hasattr(first, "self_attention")
    assert not hasattr(second, "cross_attention")
    assert all(not item.unresolved for item in (*first.inputs, *second.inputs))


def test_repeated_container_is_symbolic_and_occurrence_exact(tmp_path):
    parent, rows = _for_parent(_read(_bundle(tmp_path)))
    assert rows
    # The InnerBlock construction is one symbolic template, not N fabricated
    # runtime blocks, and every nested owner stays rooted at the exact cell.
    assert all(item.occurrence_id.local.root == parent.symbol for item in rows)
    assert len({item.occurrence_id.local for item in rows}) == 3


def test_rival_helper_container_routes_are_preserved_not_selected(tmp_path):
    stages = (STAGES.replace(
        "from .cells import NestedCell, OpaqueCell",
        "from .cells import NestedCell, OpaqueCell, RivalCell")
        .replace("[NestedCell(), OpaqueCell()]",
                 "[NestedCell(), OpaqueCell(), RivalCell(True)]"))
    cells = CELLS + """
    class RivalCell:
        def __init__(self, choose):
            if choose:
                self.make_one()
            else:
                self.make_two()
        def make_one(self):
            self.stack = nn.ModuleList([InnerBlock()])
        def make_two(self):
            self.stack = nn.ModuleList([InnerBlock()])
        def forward(self, value, side):
            for item in self.stack:
                value = item(value, side)
            return value
    """
    _parent, rows = _for_parent(
        _read(_bundle(tmp_path, stages=stages, cells=cells)), "RivalCell")
    assert rows
    assert all(isinstance(item.occurrence_id, AlternativeNestedOccurrenceId)
               for item in rows)
    assert len({item.occurrence_id.alternative.site.site_id for item in rows}) == 2
    assert {item.kind for item in rows} == {"attention", "ffn"}


def test_opaque_child_does_not_become_negative_attention_or_ffn(tmp_path):
    result = _read(_bundle(tmp_path))
    _parent, rows = _for_parent(result, "OpaqueCell")
    assert rows == ()
    value = result.require_value()
    assert any(item.parent.symbol.qualified_name == "OpaqueCell"
               and item.kind == "whole_callable_open" for item in value.issues)


def test_complete_class_and_field_renaming_preserves_mechanism_shape(tmp_path):
    renamed_stages = (STAGES.replace("NestedCell", "UnitOuter")
                      .replace("OpaqueCell", "UnitBlank")
                      .replace("units", "bucket"))
    renamed_cells = (CELLS.replace("MathLane", "UnitMath")
                     .replace("SplitPath", "UnitDense")
                     .replace("InnerBlock", "UnitInner")
                     .replace("NestedCell", "UnitOuter")
                     .replace("OpaqueCell", "UnitBlank")
                     .replace("self.stack", "self.bucket")
                     .replace("self.first", "self.alpha")
                     .replace("self.second", "self.beta")
                     .replace("self.third", "self.gamma")
                     .replace("self.a", "self.left")
                     .replace("self.b", "self.middle")
                     .replace("self.c", "self.right"))
    old = _for_parent(_read(_bundle(tmp_path / "old")))[1]
    new = _for_parent(_read(_bundle(
        tmp_path / "new", stages=renamed_stages, cells=renamed_cells)),
        "UnitOuter")[1]
    assert [item.kind for item in old] == [item.kind for item in new]
    assert [(item.ffn.gated if item.ffn else None) for item in old] == [
        (item.ffn.gated if item.ffn else None) for item in new]
    assert [[item.roots for item in row.inputs] for row in old] == [
        [item.roots for item in row.inputs] for row in new]


def test_dto_rejects_cross_parent_and_mechanism_forgery(tmp_path):
    value = _read(_bundle(tmp_path)).require_value()
    attention = next(item for item in value.mechanisms
                     if item.kind == "attention")
    ffn = next(item for item in value.mechanisms if item.kind == "ffn")
    with pytest.raises(ValueError):
        replace(attention, occurrence_id=ffn.occurrence_id)
    with pytest.raises(ValueError):
        replace(attention, kind="ffn")
    with pytest.raises(ValueError):
        replace(attention.inputs[0], roots=("forged", "forged"))


def test_indexed_framework_container_requires_exact_canonical_import_join(
        tmp_path):
    package = tmp_path / "diffusers"
    _write(package / "__init__.py", "")
    _write(package / "models" / "__init__.py", "")
    _write(package / "models" / "attention_processor.py", """
        class Attention:
            def forward(self, value): return value
    """)
    block_file = _write(package / "models" / "block.py", """
        from .attention_processor import Attention
        class Block:
            def __init__(self): self.unit = Attention()
            def forward(self, value): return self.unit(value)
    """)
    bundle = SourceBundle(
        source="test", architecture="Block",
        component_files={"root": (block_file,)},
        component_architectures={"root": "Block"},
        import_roots={"root": (SourceImportRoot(
            "diffusers", str(package)),)},
    )
    index = build_program_index(bundle)
    site = next(item for item in index.construction_sites
                if item.target == "unit")
    constructor = next(item for item in index.calls_in(site.enclosing_callable)
                       if item.span == site.span)
    imported = resolve_called_import_source(
        index, bundle, "root", constructor)
    assert imported.status == "resolved"
    expanded = imported.index
    block = next(item.symbol for item in expanded.classes
                 if item.symbol.qualified_name == "Block")
    graph = resolve_owner_graph(expanded, block)
    invocations = resolve_addressed_invocations_in_graph(
        expanded, graph, graph.root.occurrence,
        resolve_container_inventory_in_graph(
            expanded, graph, graph.root.occurrence))
    invocation = next(item for item in invocations.addressed
                      if item.call.callee.name == "unit")
    target = canonical_called_import_target(bundle, imported)
    assert target.qualified_target == \
        "diffusers.models.attention_processor.Attention"
    with pytest.raises(ValueError, match="derives from root"):
        replace(target, qualified_target=
                "other.models.attention_processor.Attention")
    with pytest.raises(ValueError, match="resolved declared root"):
        replace(target, import_root=SourceImportRoot(
            "diffusers", str(tmp_path / "foreign")))
    proof = framework_attention_lane_positive_proof_in_graph(
        expanded, graph, invocation, canonical_import=target)
    assert proof.protocol == "indexed_framework_container"
    assert proof.child_symbol == imported.imported_symbol
    assert proof.compute_protocol == "framework_attention_container"
    # The same lexical short import under another package root is powerless.
    with pytest.raises(TypeError, match="typed called-import"):
        framework_attention_lane_positive_proof_in_graph(
            expanded, graph, invocation,
            canonical_import="diffusers.models.attention_processor.Attention")


def test_real_sdxl_preserves_rival_transformer_routes_and_framework_attention():
    import diffusers

    package = Path(diffusers.__file__).resolve().parent
    source = package / "models" / "unets" / "unet_2d_condition.py"
    bundle = SourceBundle(
        source="test", architecture="UNet2DConditionModel",
        component_files={"root": (str(source),)},
        component_architectures={"root": "UNet2DConditionModel"},
        import_roots={"root": (SourceImportRoot("diffusers", str(package)),)},
    )
    value = _read(bundle).require_value()
    transformer = tuple(
        item for item in value.mechanisms
        if (item.occurrence_id.parent
            if hasattr(item.occurrence_id, "parent")
            else item.occurrence_id.alternative.parent
            ).symbol.qualified_name == "Transformer2DModel")
    assert transformer, [
        (item.kind, item.detail) for item in value.issues
        if item.parent.symbol.qualified_name == "Transformer2DModel"]
    attention = tuple(item for item in transformer
                      if item.kind == "attention")
    assert attention
    assert all(isinstance(item.attention, FrameworkAttentionLaneEvidence)
               for item in attention)
    assert {item.attention.compute_protocol for item in attention} == {
        "framework_attention_container"}
    # Transformer2DModel constructs the same BasicTransformerBlock template in
    # three mutually-exclusive initialization helpers.  Preserve all three
    # exact routes; never select one because the resulting class is familiar.
    alternatives = tuple(item.occurrence_id.alternative for item in attention
                         if isinstance(item.occurrence_id,
                                       AlternativeNestedOccurrenceId))
    assert alternatives
    assert len({item.site for item in alternatives}) == 3
    assert all(len(item.rival_sites) == 3 for item in alternatives)
    # FeedForward selects both dense and gated activation classes from the
    # activation_fn operand.  Source alone does not prove SDXL's checkpoint
    # choice, so E1 must not manufacture GEGLU/gating before the config join.
    assert not any(item.kind == "ffn" for item in transformer)

    # E2's neutral constructor-value boundary can nevertheless prove the exact
    # runtime operand independently for every preserved route.  The route is:
    # Transformer2DModel's literal class default -> its imported
    # register_to_config self.config access -> BasicTransformerBlock formal ->
    # FeedForward formal.  This assertion still does not interpret "geglu" as
    # an FFN mechanism; that semantic selection belongs to E2's U7 join.
    def target_for(site, candidate):
        imported = (canonical_called_import_target(bundle,
                    candidate.import_chain[-1])
                    if candidate.import_chain else None)
        return canonical_construction_target(
            value.index, site, candidate.symbol,
            canonical_import=imported)

    proven = []
    selected_mechanisms = []
    input_roles = []
    input_role_statuses = []
    input_role_failures = []
    alternatives = []
    for item in attention:
        if not isinstance(item.occurrence_id,
                          AlternativeNestedOccurrenceId):
            continue
        alternative = item.occurrence_id.alternative
        if all(kept.site != alternative.site for kept in alternatives):
            alternatives.append(alternative)
    for alternative in alternatives:
        parent_candidates = tuple(dict.fromkeys(
            (construction, candidate)
            for invocation in value.cells.cells.invocations
            for construction in invocation.constructions
            for candidate in construction.candidates
            if invocation.parent.occurrence_id == alternative.parent.parent
            and candidate.symbol == alternative.parent.symbol
            and candidate.span == alternative.parent.candidate_span
            and (construction.site.span if construction.site is not None
                 else construction.field_assign.value.span)
            == alternative.parent.construction_span))
        assert len(parent_candidates) == 1
        construction, parent_candidate = parent_candidates[0]
        assert construction.site is not None
        transformer_target = target_for(
            construction.site, parent_candidate)
        transformer_frame = constructor_frame(
            value.index, transformer_target)
        block_target = target_for(alternative.site, alternative.candidate)
        block_frame = constructor_frame(
            value.index, block_target, transformer_frame)
        alternative_attention = tuple(
            item for item in attention
            if isinstance(item.occurrence_id,
                          AlternativeNestedOccurrenceId)
            and item.occurrence_id.alternative == alternative
            and item.attention.block_occurrence
            == block_frame.graph.root.occurrence)
        assert alternative_attention
        for lane_row in alternative_attention:
            role = framework_attention_invocation_role(
                value.index, block_frame, lane_row.attention)
            if role.status not in {"resolved", "incomplete"}:
                input_role_failures.append((
                    alternative.site.span.line,
                    lane_row.attention.construction.target,
                    tuple(item.kind for item in role.failures)))
                continue
            input_role_statuses.append(role.status)
            input_roles.append(role.require_value())
        feed_sites = tuple(
            site for site in value.index.construction_sites_of(
                alternative.symbol)
            if site.target == "ff")
        assert len(feed_sites) == 1
        feed_site = feed_sites[0]
        feed_candidates = tuple(
            item.symbol for item in feed_site.candidates
            if item.symbol is not None)
        assert len(feed_candidates) == 1
        feed_target = canonical_construction_target(
            value.index, feed_site, feed_candidates[0])
        feed_frame = constructor_frame(
            value.index, feed_target, block_frame)
        result = resolve_effective_constructor_parameter(
            value.index, feed_frame, "activation_fn")
        assert result.status == "resolved", result.failures
        proven.append(result.require_value())
        mechanism = selected_composite_ffn_mechanism(
            value.index, bundle, feed_frame)
        assert mechanism.status == "resolved", mechanism.failures
        selected_mechanisms.append(mechanism.require_value())
    assert len(proven) == 3
    assert {(item.value, item.source_kind) for item in proven} == {
        ("geglu", "class_default")}
    assert all([step.access_kind for step in item.steps] == [
        "parameter_forward", "registered_config_forward", "class_default"]
               for item in proven)
    assert len(selected_mechanisms) == 3
    assert {(item.gated, item.projection_mode, item.activation)
            for item in selected_mechanisms} == {
        (True, "fused_gate_up", "gelu")}
    # Each exact BasicTransformerBlock route proves attn1's two source-level
    # alternatives (self vs distinct context slot).  The direct attn2 lanes
    # remain typed unknown because an optional runtime GLIGEN/fuser transform
    # can replace their primary state before the call.  A nested fuser's own
    # ``attn`` belongs to the child occurrence and is never queried with this
    # parent frame.  U11-F may shrink the attn2 unknowns only with an exact
    # component-bound runtime/config proof; E2c must not infer conventional
    # self+cross structure from the familiar two-lane shape.
    assert len(input_roles) == 3
    assert {item.kind for item in input_roles} == {"conditional"}
    assert input_role_statuses == ["incomplete"] * 3
    assert len(input_role_failures) == 3
    assert {target for _line, target, _kinds in input_role_failures} == {
        "attn2"}
    assert {kinds for _line, _target, kinds in input_role_failures} == {
        ("incomplete_graph",)}
    assert len({line for line, _target, _kinds in input_role_failures}) == 3
