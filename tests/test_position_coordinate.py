"""Neutral coordinate-origin kernel controls."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_coordinate import coordinate_origin
from model_unfolder.evidence.program_index import build_program_index


def _origin(tmp_path, unpack):
    path = tmp_path / "coordinates.py"
    body = textwrap.indent(unpack, "    ")
    path.write_text(
        "import torch\n"
        "def forward(tensor: torch.Tensor, offset: int = 0):\n"
        f"{body}\n"
        "    coords = (torch.arange(length) + offset).to(tensor.device)\n"
        "    return coords.view(-1)\n",
        encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Missing"},
        architecture="Missing")
    index = build_program_index(bundle)
    callable_symbol = next(item.symbol for item in index.callables
                           if item.symbol.qualified_name == "forward")
    returned = index.return_observations_in(callable_symbol)[0]
    return coordinate_origin(
        index, callable_symbol, returned.value, returned.span)


def test_exact_parameter_size_unpack_can_bound_arange(tmp_path):
    result = _origin(tmp_path, "batch, lanes, length = tensor.size()")
    assert result is not None
    assert result.protocol == "arange"


def test_size_call_with_arguments_is_not_a_complete_unpack(tmp_path):
    result = _origin(tmp_path, "batch, lanes, length = tensor.size(0)")
    assert result is None


def test_unrelated_object_size_unpack_is_not_a_parameter_shape(tmp_path):
    result = _origin(
        tmp_path,
        "other = Factory()\nbatch, lanes, length = other.size()")
    assert result is None


def test_untyped_parameter_size_unpack_is_not_promoted(tmp_path):
    path = tmp_path / "untyped.py"
    path.write_text(textwrap.dedent("""
        import torch
        def forward(tensor, offset: int = 0):
            batch, lanes, length = tensor.size()
            return torch.arange(length) + offset
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Missing"}, architecture="Missing")
    index = build_program_index(bundle)
    callable_symbol = index.callables[0].symbol
    returned = index.return_observations_in(callable_symbol)[0]
    assert coordinate_origin(
        index, callable_symbol, returned.value, returned.span) is None
