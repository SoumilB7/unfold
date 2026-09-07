"""Independent initial S8 RETURN probes; run from the reviewed checkout.

This script changes no checkout files, imports no test module, and runs no
pytest lane. Temporary source fixtures are deleted when each probe finishes.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_cell_connections import (
    _direct_origin, _member, _member_stays_bound,
)
from model_unfolder.evidence.unet_cell_mechanism import (
    _conditioning, _origin_snapshots, read_unet_stage_join_connections,
)


def fingerprint():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    digest = hashlib.sha256()
    for raw in sorted(path for path in paths if path):
        path = ROOT / raw.decode()
        if path.is_file():
            digest.update(raw + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def index_source(directory, source):
    path = Path(directory) / "cell.py"
    path.write_text(source)
    return build_program_index(SourceBundle(
        source="path", files=(str(path),), component_files={"root": (str(path),)}))


def alias_escape():
    source = """class Arbitrary:
    def forward(self, value):
        alias = self
        helper(alias)
        value = self.first(value)
        return self.second(value)
"""
    with tempfile.TemporaryDirectory(prefix="s8-review-alias-") as directory:
        index = index_source(directory, source)
        forward = next(row for row in index.callables
                       if row.symbol.qualified_name == "Arbitrary.forward")
        calls = tuple(index.calls_in(forward.symbol))
        target = next(row for row in calls if _member(row.callee) == "second")
        bindings = SimpleNamespace(_modules={
            "cell": SimpleNamespace(init_attributes={}),
            "cell.first": object(), "cell.second": object(),
        })
        sources = {row.span: row for row in calls if _member(row.callee) == "first"}
        result = {
            "member_stays_bound": _member_stays_bound(index, forward, target, "cell", bindings),
            "direct_edge_found": _direct_origin(index, forward, target, target.args[0], sources) is not None,
            "source": source,
        }
        assert result["member_stays_bound"] and result["direct_edge_found"]
        return result


def forged_primitive():
    import torch
    from physics.instance_inventory import _class_ref
    from model_unfolder.evidence.primitive_semantics import runtime_primitive_definition

    class SiLU(torch.nn.Module):
        def forward(self, value):
            return value + 10

    SiLU.__module__ = "torch.nn.modules.activation"
    SiLU.__qualname__ = "SiLU"
    obj = SiLU()
    result = {
        "is_framework_SiLU": type(obj) is torch.nn.SiLU,
        "serialized_ref": str(_class_ref(type(obj))),
        "claimed_meaning": runtime_primitive_definition(_class_ref(type(obj))),
        "actual_at_zero": obj(torch.tensor(0.)).item(),
        "framework_SiLU_at_zero": torch.nn.SiLU()(torch.tensor(0.)).item(),
    }
    assert not result["is_framework_SiLU"]
    assert result["claimed_meaning"] == ("activation", "SiLU", "silu")
    assert result["actual_at_zero"] != result["framework_SiLU_at_zero"]
    return result


def unrelated_concat():
    from model_unfolder.evidence.component_owner import resolve_component_root
    from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
    from model_unfolder.evidence.unet_stage_construction import read_unet_stage_construction
    from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution
    from model_unfolder.evidence.unet_stage_cells import read_unet_stage_cells

    # Existing source examples are read as literal data, never imported/executed.
    parsed = ast.parse((ROOT / "tests/test_unet_stage_cells.py").read_text())
    examples = {node.targets[0].id: ast.literal_eval(node.value)
                for node in parsed.body if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}
    stages = examples["STAGES"].replace(
        "from torch.nn import ModuleList", "from torch.nn import ModuleList\n    from torch import cat")
    stages = stages.replace("value = unit(value)",
        "ignored = cat([side, side], dim=0)\n                joined = cat([value, side], dim=1)\n                value = unit(joined)")
    with tempfile.TemporaryDirectory(prefix="s8-review-join-") as directory:
        package = Path(directory) / "pkg"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name, source in (("root.py", examples["ROOT"]), ("stages.py", stages),
                             ("cells.py", examples["CELLS"])):
            (package / name).write_text(textwrap.dedent(source))
        bundle = SourceBundle(
            source="test", architecture="Root", component_files={"root": (str(package / "root.py"),)},
            component_architectures={"root": "Root"},
            import_roots={"root": (SourceImportRoot("pkg", str(package)),)})
        index = build_program_index(bundle)
        root = resolve_component_root(index, bundle, "root")
        topology = read_diffusion_root_topology(index, root).require_value()
        construction = read_unet_stage_construction(index, bundle, root, topology).require_value()
        execution = read_unet_stage_execution(construction, bundle, root).require_value()
        cells = read_unet_stage_cells(execution, bundle).require_value()
        row = read_unet_stage_join_connections(cells).require_value()[0]
        rival = next(call for call in cells.index.calls_in(row.join.enclosing_callable)
                     if call.callee.name == "cat" and call.span != row.join.span)
        forged = replace(row, join=rival)
        result = {
            "actual_join_line": row.join.span.line, "unused_join_line": rival.span.line,
            "forged_connection_constructed": forged.join == rival,
            "actual_child_call_line": row.invocation.call.span.line,
            "bindings_still_justify_original_join": forged.bindings == row.bindings,
            "stage_source": textwrap.dedent(stages),
            "limit": "DTO constructor reproduced; wrapper acceptance checked by source inspection, not a full forged EvidenceFact probe",
        }
        assert result["forged_connection_constructed"] and result["bindings_still_justify_original_join"]
        return result


def discarded_side():
    source = """def ignore(arg):
    return 7
class Cell:
    def forward(self, value, side):
        hidden = value * 2
        detached_side = ignore(side)
        hidden = hidden + detached_side
        return value + hidden
"""
    with tempfile.TemporaryDirectory(prefix="s8-review-side-") as directory:
        index = index_source(directory, source)
        forward = next(row for row in index.callables if row.symbol.qualified_name == "Cell.forward")
        snapshots, _ = _origin_snapshots(index, forward.symbol, ("value", "side"))
        claims = _conditioning(index, forward.symbol, "value", ("side",), snapshots)
        namespace = {}
        exec(source, namespace)
        result = {
            "claims": [{"kind": row.kind, "claimed_side_formal": row.side_parameter,
                        "binding_line": row.binding.span.line} for row in claims],
            "result_side_zero": namespace["Cell"]().forward(1, 0),
            "result_side_999": namespace["Cell"]().forward(1, 999),
            "source": source,
        }
        assert result["claims"] and result["claims"][0]["claimed_side_formal"] == "side"
        assert result["result_side_zero"] == result["result_side_999"] == 10
        return result


before = fingerprint()
result = {
    "reviewed_implementation": "bb48a31e336c1721f0a111862bd081acf390bb25",
    "checkout_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "checkout": str(ROOT), "tracked_fingerprint_before": before,
    "verdict": "RETURN", "scope": "initial authority review; not the full S8 acceptance bracket",
    "R1_self_alias_escape": alias_escape(), "R2_serialized_type_spoof": forged_primitive(),
    "R3_unrelated_concat": unrelated_concat(), "R4_discarded_conditioning_argument": discarded_side(),
}
result["tracked_fingerprint_after"] = fingerprint()
assert result["tracked_fingerprint_after"] == before
print(json.dumps(result, indent=2, sort_keys=True))
