"""Source address is insufficient to choose a helper after runtime rebinding.

Standalone source-index/Python-language counterexample, not a model run.
"""
import hashlib
import json
from pathlib import Path
import tempfile

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index

SOURCE = '''def replace_child(owner):
    owner.second = lambda value: value + 100

class Cell:
    def __init__(self):
        self.second = lambda value: value * 2

    def helper(self):
        return None

    def forward(self, value):
        self.helper()
        return self.second(value)
'''

with tempfile.TemporaryDirectory(prefix="s8-helper-lookup-") as directory:
    path = Path(directory) / "model.py"
    path.write_text(SOURCE)
    index = build_program_index(SourceBundle(
        source="path", files=(str(path),), component_files={"root": (str(path),)}))
    namespace = {"__name__": "s8_lookup_counterexample"}
    exec(compile(SOURCE, str(path), "exec"), namespace)
    instance = namespace["Cell"]()
    recorded_before = {"class": type(instance).__qualname__,
                       "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                       "instance_attribute_keys": sorted(vars(instance))}
    lookup_before = instance.helper.__func__.__qualname__
    # Simulate a process-local import/plugin replacement. The class's source
    # bytes, MRO, class address and instance attribute census do not change.
    namespace["Cell"].helper = namespace["replace_child"]
    recorded_after = {"class": type(instance).__qualname__,
                      "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                      "instance_attribute_keys": sorted(vars(instance))}
    helper = next(row for row in index.callables
                  if row.symbol.qualified_name == "Cell.helper")
    result = {
        "source": SOURCE,
        "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
        "indexed_candidate": helper.symbol.qualified_name,
        "indexed_candidate_writes": [str(row.target) for row in index.attribute_accesses
                                     if row.enclosing_callable == helper.symbol and row.mode == "write"],
        "actual_lookup_function": instance.helper.__func__.__qualname__,
        "lookup_before": lookup_before,
        "existing_address_channels_before": recorded_before,
        "existing_address_channels_after": recorded_after,
        "child_before_call": instance.second(3),
        "forward_result": instance.forward(3),
    }
    assert result["indexed_candidate_writes"] == []
    assert result["actual_lookup_function"] == "replace_child"
    assert recorded_before == recorded_after
    assert result["child_before_call"] == 6 and result["forward_result"] == 103
    print(json.dumps(result, indent=2, sort_keys=True))
