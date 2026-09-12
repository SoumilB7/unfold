"""A flattened try-handler body cannot certify the original owner binding."""
import importlib.util
import json
from pathlib import Path
import tempfile

from physics.attribute_bindings import AttributeBindingWitness, _function_witness, capture_attribute_lookup_types
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_wrapper_binding import read_wrapper_binding

SOURCE = '''class Cell:
    def forward(owner, value):
        return owner.child(value)

def wrap(captured):
    def wrapper(owner, value):
        try:
            unknown()
        except Exception as owner:
            return captured(owner, value)
    return wrapper
'''

with tempfile.TemporaryDirectory(prefix="s8-wrapper-handler-") as directory:
    path = Path(directory) / "model.py"
    path.write_text(SOURCE)
    spec = importlib.util.spec_from_file_location("s8_wrapper_handler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = build_program_index(SourceBundle(source="path", component_files={"root": (str(path),)}))
    function, reason = _function_witness(module.wrap(module.Cell.forward), capture_attribute_lookup_types())
    assert function is not None, reason
    witness = AttributeBindingWitness("", "forward", "plain_bound_method",
                                      function=function, lookup_kind="object_getattribute")
    target = next(row.symbol for row in index.callables if row.symbol.qualified_name == "Cell.forward")
    wrapper = next(row for row in index.callables if row.symbol.qualified_name == "wrap.wrapper")
    result = read_wrapper_binding(index, witness, target)
    out = {"source": SOURCE, "result_kind": result.kind, "reason": result.reason,
           "expected_kind": "unresolved",
           "owner_identifiers": [{"context": x.context, "line": x.span.line}
                                 for x in index.identifiers_in(wrapper.symbol) if x.name == "owner"],
           "unsupported": [{"kind": x.construct_kind, "line": x.span.line}
                           for x in index.unsupported_execution_in(wrapper.symbol)]}
    print(json.dumps(out, indent=2, sort_keys=True))
    raise SystemExit(0 if result.kind == "unresolved" else 1)
