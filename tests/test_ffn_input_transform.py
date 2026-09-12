"""U11-E2b code-proven FFN input-transform controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.ffn_input_transform import (
    fused_input_projection_transform_at_symbol,
)
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index


SOURCE = """
import torch_npu
from torch import nn
from torch.nn import functional as F

class Unit:
    def __init__(self, width):
        self.proj = nn.Linear(width, width * 2)
    def activate(self, value):
        if value.device.type == "mps":
            return F.gelu(value.float()).to(value.dtype)
        return F.gelu(value)
    def forward(self, value):
        projected = self.proj(value)
        if ready():
            return torch_npu.npu_geglu(projected, dim=-1)[0]
        else:
            left, right = projected.chunk(2, dim=-1)
            return left * self.activate(right)
"""


def _read(tmp_path, source=SOURCE, architecture="Unit"):
    path = Path(tmp_path) / "model.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    index = build_program_index(SourceBundle(
        source="test", architecture=architecture,
        component_files={"root": (str(path),)},
        component_architectures={"root": architecture}))
    symbol = next(item.symbol for item in index.classes
                  if item.symbol.qualified_name == architecture)
    return index, symbol, fused_input_projection_transform_at_symbol(
        index, symbol)


def test_equivalent_framework_and_source_paths_prove_fused_gelu(tmp_path):
    _index, _symbol, result = _read(tmp_path)
    assert result.status == "resolved"
    value = result.require_value()
    assert (value.gated, value.mode, value.activation) == (
        True, "fused_gate_up", "gelu")
    assert [item.kind for item in value.alternatives] == [
        "framework_fused", "source_split"]
    assert value.projection.parent == value.owner


def test_complete_class_field_and_local_renaming_is_semantically_powerless(
        tmp_path):
    renamed = (SOURCE.replace("Unit", "Other")
               .replace("self.proj", "self.map")
               .replace("projected", "made")
               .replace("left", "alpha")
               .replace("right", "beta")
               .replace("activate", "curve"))
    old = _read(tmp_path / "old")[2].require_value()
    new = _read(tmp_path / "new", renamed, "Other")[2].require_value()
    assert (old.mode, old.activation) == (new.mode, new.activation)


@pytest.mark.parametrize("old,new", [
    ("npu_geglu", "npu_other"),
    ("dim=-1)[0]", "dim=1)[0]"),
    ("dim=-1)[0]", "dim=-1)[1]"),
    ("projected.chunk(2", "projected.split(2"),
    ("projected.chunk(2", "projected.chunk(3"),
    ("left * self.activate(right)", "left + self.activate(right)"),
    ("self.activate(right)", "self.activate(right, 1)"),
    ("return F.gelu(value)", "return value"),
    ("return F.gelu(value)", "return F.gelu(value) + value"),
    ("return F.gelu(value)", "return F.gelu(value + 1)"),
])
def test_each_required_mechanism_edge_is_a_kill_shot(tmp_path, old, new):
    result = _read(tmp_path, SOURCE.replace(old, new))[2]
    assert result.status == "failed"


def test_one_guarded_return_without_a_fallback_is_not_complete(tmp_path):
    source = SOURCE.replace(
        "        else:\n"
        "            left, right = projected.chunk(2, dim=-1)\n"
        "            return left * self.activate(right)\n", "")
    result = _read(tmp_path, source)[2]
    assert result.status == "failed"


def test_unrelated_activation_cannot_certify_the_live_return(tmp_path):
    source = SOURCE.replace(
        "return left * self.activate(right)",
        "unused = self.activate(right)\n            return left * right")
    result = _read(tmp_path, source)[2]
    assert result.status == "failed"


def test_dto_rejects_semantic_and_provenance_forgery(tmp_path):
    value = _read(tmp_path)[2].require_value()
    with pytest.raises(ValueError, match="unanimous"):
        replace(value, activation="silu")
    with pytest.raises(ValueError, match="same-source provenance"):
        replace(value.alternatives[0], spans=())


def test_real_diffusers_geglu_is_proven_from_both_implementation_paths():
    import diffusers

    package = Path(diffusers.__file__).resolve().parent
    source = package / "models" / "activations.py"
    index = build_program_index(SourceBundle(
        source="test", architecture="GEGLU",
        component_files={"root": (str(source),)},
        component_architectures={"root": "GEGLU"},
        import_roots={"root": (SourceImportRoot(
            "diffusers", str(package)),)}))
    symbol = next(item.symbol for item in index.classes
                  if item.symbol.qualified_name == "GEGLU")
    value = fused_input_projection_transform_at_symbol(
        index, symbol).require_value()
    assert (value.mode, value.activation) == ("fused_gate_up", "gelu")
    assert {item.kind for item in value.alternatives} == {
        "framework_fused", "source_split"}
