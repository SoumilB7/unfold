"""U11-E2 selector-to-composite-FFN mechanism controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.constructor_values import (
    canonical_construction_target,
    constructor_frame,
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


SOURCE = """
import torch_npu
from torch import nn
from torch.nn import functional as F

class GatePath:
    def __init__(self, width): self.proj = nn.Linear(width, width * 2)
    def curve(self, value): return F.gelu(value)
    def forward(self, value):
        projected = self.proj(value)
        if ready(): return torch_npu.npu_geglu(projected, dim=-1)[0]
        else:
            left, right = projected.chunk(2, dim=-1)
            return left * self.curve(right)

class OtherPath:
    def forward(self, value): return value

class Composite:
    def __init__(self, width, mode="gate", add_extra=False):
        if mode == "gate": selected = GatePath(width)
        elif mode == "other": selected = OtherPath()
        self.parts = nn.ModuleList([])
        self.parts.append(selected)
        self.parts.append(nn.Dropout(0.0))
        self.parts.append(nn.Linear(width, width))
        if add_extra:
            self.parts.append(OtherPath())
    def forward(self, state):
        for operation in self.parts:
            state = operation(state)
        return state

class Root:
    def __init__(self): self.unit = Composite(8)
"""


def _read(tmp_path, source=SOURCE):
    path = Path(tmp_path) / "model.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    site = next(item for item in index.construction_sites
                if item.owner.qualified_name == "Root"
                and item.target == "unit")
    target = canonical_construction_target(
        index, site, site.candidates[0].symbol)
    frame = constructor_frame(index, target)
    return selected_composite_ffn_mechanism(index, bundle, frame)


def test_literal_selector_reaches_code_proven_gated_composite(tmp_path):
    value = _read(tmp_path).require_value()
    assert (value.gated, value.projection_mode, value.activation) == (
        True, "fused_gate_up", "gelu")
    assert value.branch.selector.value == "gate"
    assert value.branch.selector.parameter.name == "mode"
    assert len(value.branch.rival_bindings) == 2
    assert len(value.execution.transparent_sites) == 1
    assert value.execution.output_site.target == value.execution.field


def test_complete_symbol_field_local_and_token_renaming_preserves_result(
        tmp_path):
    renamed = (SOURCE.replace("GatePath", "First")
               .replace("OtherPath", "Second")
               .replace("Composite", "Bucket")
               .replace("Root", "Top")
               .replace('"gate"', '"choice_a"')
               .replace('"other"', '"choice_b"')
               .replace("mode", "choice")
               .replace("selected", "picked")
               .replace("parts", "items")
               .replace("operation", "entry"))
    path = Path(tmp_path) / "model.py"
    path.write_text(textwrap.dedent(renamed), encoding="utf-8")
    bundle = SourceBundle(
        source="test", architecture="Top",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Top"})
    index = build_program_index(bundle)
    site = next(item for item in index.construction_sites
                if item.owner.qualified_name == "Top" and item.target == "unit")
    frame = constructor_frame(index, canonical_construction_target(
        index, site, site.candidates[0].symbol))
    value = selected_composite_ffn_mechanism(
        index, bundle, frame).require_value()
    assert (value.gated, value.projection_mode, value.activation) == (
        True, "fused_gate_up", "gelu")
    assert value.branch.selector.parameter.name == "choice"


@pytest.mark.parametrize("old,new", [
    ('Composite(8)', 'Composite(8, mode=choose())'),
    ('if mode == "gate"', 'if unknown == "gate"'),
    ('self.parts.append(selected)', 'self.parts.append(OtherPath())'),
    ('self.parts.append(nn.Linear(width, width))',
     'self.parts.append(nn.Dropout(0.0))'),
    ('for operation in self.parts', 'for operation in self.other'),
    ('state = operation(state)', 'other = operation(state)'),
    ('self.parts = nn.ModuleList([])',
     'self.parts = nn.ModuleList([OtherPath()])'),
])
def test_every_selector_container_and_execution_edge_is_required(
        tmp_path, old, new):
    assert _read(tmp_path, SOURCE.replace(old, new)).status == "failed"


def test_two_true_selector_branches_are_ambiguous(tmp_path):
    source = SOURCE.replace(
        'elif mode == "other": selected = OtherPath()',
        'if mode == "gate": selected = OtherPath()')
    assert _read(tmp_path, source).status == "failed"


@pytest.mark.parametrize("replacement", [
    "projected.chunk(2, dim=0)",
    "projected.chunk(2)",
    "projected.chunk(2, dim=choose_dim())",
    "projected.chunk(2, -2)",
])
def test_source_split_must_be_exactly_on_the_last_axis(
        tmp_path, replacement):
    source = SOURCE.replace("projected.chunk(2, dim=-1)", replacement)
    assert _read(tmp_path, source).status == "failed"


def test_explicit_selector_actual_overrides_the_class_default(tmp_path):
    source = SOURCE.replace('Composite(8)', 'Composite(8, mode="other")')
    assert _read(tmp_path, source).status == "failed"


def test_unclassified_extra_container_element_blocks_the_claim(tmp_path):
    source = SOURCE.replace(
        "self.parts.append(nn.Dropout(0.0))",
        "self.parts.append(OtherPath())")
    assert _read(tmp_path, source).status == "failed"


def test_proven_inactive_optional_element_cannot_pollute_the_live_graph(
        tmp_path):
    value = _read(tmp_path).require_value()
    assert len(value.execution.append_calls) == 3
    assert all(site.target == value.execution.field
               for site in value.execution.transparent_sites)


def test_active_optional_unclassified_element_blocks_the_claim(tmp_path):
    source = SOURCE.replace("Composite(8)", "Composite(8, add_extra=True)")
    assert _read(tmp_path, source).status == "failed"


def test_unknown_optional_element_guard_blocks_the_claim(tmp_path):
    source = SOURCE.replace("if add_extra:", "if choose_extra():")
    assert _read(tmp_path, source).status == "failed"


def test_early_return_inside_the_loop_blocks_execution_proof(tmp_path):
    source = SOURCE.replace(
        "state = operation(state)",
        "state = operation(state)\n            return state")
    assert _read(tmp_path, source).status == "failed"


def test_conditionally_executed_element_is_not_a_complete_container_route(
        tmp_path):
    source = SOURCE.replace(
        "state = operation(state)",
        "if enabled:\n                state = operation(state)")
    assert _read(tmp_path, source).status == "failed"


def test_dto_rejects_selector_and_execution_forgery(tmp_path):
    value = _read(tmp_path).require_value()
    with pytest.raises(ValueError, match="closes selector"):
        replace(value, spans=())
    with pytest.raises(ValueError, match="complete typed route"):
        replace(value.execution, append_calls=())
    with pytest.raises(ValueError, match="all local rivals"):
        replace(value.branch, rival_bindings=tuple(
            reversed(value.branch.rival_bindings)))


def test_real_diffusers_feed_forward_default_reaches_code_proven_geglu(
        tmp_path):
    import diffusers

    package = Path(diffusers.__file__).resolve().parent
    mirror = Path(tmp_path) / "diffusers"
    (mirror / "models").mkdir(parents=True)
    (mirror / "__init__.py").write_text("", encoding="utf-8")
    (mirror / "models" / "__init__.py").write_text("", encoding="utf-8")
    for name in ("attention.py", "activations.py"):
        (mirror / "models" / name).write_text(
            (package / "models" / name).read_text(encoding="utf-8"),
            encoding="utf-8")
    source = mirror / "root.py"
    source.write_text(textwrap.dedent("""
        from diffusers.models.attention import FeedForward
        class Root:
            def __init__(self):
                self.unit = FeedForward(8)
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (str(source),)},
        component_architectures={"root": "Root"},
        import_roots={"root": (SourceImportRoot(
            "diffusers", str(mirror)),)})
    index = build_program_index(bundle)
    site = next(item for item in index.construction_sites
                if item.owner.qualified_name == "Root"
                and item.target == "unit")
    call = next(item for item in index.calls_in(site.enclosing_callable)
                if item.span == site.span)
    imported = resolve_called_import_source(index, bundle, "root", call)
    assert imported.status == "resolved", (
        imported.status, imported.failure_kind, imported.failure_detail)
    target = canonical_construction_target(
        imported.index, site, imported.imported_symbol,
        canonical_import=canonical_called_import_target(bundle, imported))
    frame = constructor_frame(imported.index, target)
    result = selected_composite_ffn_mechanism(
        imported.index, bundle, frame)
    assert result.status == "resolved", result.failures[0].detail
    value = result.require_value()
    assert (value.gated, value.projection_mode, value.activation) == (
        True, "fused_gate_up", "gelu")
