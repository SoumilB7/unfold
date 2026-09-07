"""Reproduce the S8 class-lookup evidence gap without authoring a new reader."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from physics.instance_inventory import BuildRequest, inventory_in_subprocess
from physics.execution_observation import ExecutionRecipe, TensorArgument, observe_in_subprocess
from physics.source_override import SourceOverride


SOURCE = '''from torch import nn

class ClassLookupFixture(nn.Module):
    if FLAG:
        optional_scale = 2.0

    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, value):
        result = self.projection(value)
        if getattr(self, "optional_scale", None):
            result = result * self.optional_scale
        return result
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, populations = {}, []
    for label, flag in (("present", "True"), ("absent", "False")):
        path = (args.output / (label + ".py")).resolve()
        path.write_text(SOURCE.replace("FLAG", flag))
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        override = SourceOverride("test_support.s6_models", str(path), source_hash)
        request = BuildRequest(config={}, framework="custom", factory_module=override.module,
                               factory_qualname="ClassLookupFixture", source_overrides=(override,),
                               timeout_seconds=45, memory_limit_bytes=8 * 1024**3)
        built = inventory_in_subprocess(request)
        assert built.status == "ok", built.failure
        inventory = built.inventory
        recipe = ExecutionRecipe("class-lookup", "synthetic_tensor", "eval", "disabled",
                                 "unspecified", False, "float32",
                                 {row.package: row.version for row in inventory.provenance.packages},
                                 tensor_arguments=(TensorArgument("value", (1, 4), "float32"),))
        observed = observe_in_subprocess(request, recipe)
        assert observed.status == "ok", observed.failure
        index = build_program_index(SourceBundle(source="path", files=(str(path),),
                                                 component_files={"root": (str(path),)}))
        cls = next(row for row in index.classes if row.symbol.qualified_name == "ClassLookupFixture")
        populations.append(tuple(dataclasses.asdict(row) for row in inventory.modules))
        records[label] = {
            "source_sha256": source_hash,
            "indexed_conditional_member_declarations": [row.attr for row in cls.body_assigns if row.attr == "optional_scale"],
            "instance_attribute_present": "optional_scale" in inventory.modules[0].init_attributes,
            "observed_ops_under_recipe": [row.op for row in observed.observation.functional_ops],
            "observation_status": observed.status,
        }
        (args.output / (label + "-inventory.json")).write_text(json.dumps(built.to_dict(), indent=2))
        (args.output / (label + "-observation.json")).write_text(json.dumps(observed.to_dict(), indent=2))
    result = {"cases": records, "module_populations_identical": populations[0] == populations[1],
              "gap": "ClassRecord has no conditional class-member address or class-body coverage record; an absent instance attribute cannot establish getattr's default branch.",
              "scope": "The trace is a positive-only counterexample, never a static negative proof."}
    assert result["module_populations_identical"]
    assert all(not row["indexed_conditional_member_declarations"] and not row["instance_attribute_present"]
               for row in records.values())
    assert "mul" in records["present"]["observed_ops_under_recipe"]
    assert "mul" not in records["absent"]["observed_ops_under_recipe"]
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
