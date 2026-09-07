"""Replay existing worker witnesses against indexed installed source; no model run."""
import json
from pathlib import Path
import sysconfig
from types import SimpleNamespace

from physics.attribute_bindings import binding_from_dict
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_lookup_closure import read_lookup_closure


base = Path(sysconfig.get_paths()["purelib"]) / "diffusers"
files = tuple(str(base / name) for name in (
    "models/unets/unet_2d_condition.py", "models/modeling_utils.py",
    "configuration_utils.py", "utils/peft_utils.py"))
index = build_program_index(SourceBundle(source="path", files=files, component_files={"root": files}))
receipt = Path(__file__).parent.parent / "S8-attribute-binding-observation" / "installed-class-probe.json"
rows = [binding_from_dict(row) for row in json.loads(receipt.read_text())["bindings"]]
bindings = SimpleNamespace(index=index, _modules={
    "": SimpleNamespace(init_attributes={}),
    **{row.child_path: SimpleNamespace(init_attributes={}) for row in rows if row.child_path}})
results = []
for row in rows:
    result = read_lookup_closure(bindings, row)
    results.append({"attribute": row.attribute, "kind": result.kind, "reason": result.reason,
                    "source_sha256": (row.function or row.lookup_getattr).source_sha256,
                    "conditions": [{"return_line": condition.return_span.line,
                                    "branch_kinds": [step.kind for step in condition.branch]}
                                   for condition in result.conditions]})
print(json.dumps({"scope": "existing small constructed witness replay plus current exact source index; "
                          "not full SDXL inventory, product output or family completion",
                  "results": results}, indent=2, sort_keys=True))
