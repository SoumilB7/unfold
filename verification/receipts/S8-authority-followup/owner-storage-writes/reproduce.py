"""Owner R1 controls: existing self-reachable storage writes must be accounted.

Run from the checkout under review. No pytest, model execution, or source edits.
Exit 1 means at least one explicit write incorrectly retained member identity.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_cell_connections import _member, _member_stays_bound

reader = Path("model_unfolder/evidence/unet_cell_connections.py")
before = hashlib.sha256(reader.read_bytes()).hexdigest()
results = []
for change in ("pass", 'self._modules["second"] = replacement',
               'self.__dict__["second"] = replacement', "del self.second"):
    source = ("class Cell:\n def forward(self,value,replacement):\n  " + change
              + "\n  value=self.first(value)\n  return self.second(value)\n")
    with tempfile.TemporaryDirectory(prefix="s8-owner-write-") as directory:
        path = Path(directory) / "model.py"
        path.write_text(source)
        index = build_program_index(SourceBundle(
            source="path", files=(str(path),), component_files={"root": (str(path),)}))
        forward = next(row for row in index.callables
                       if row.symbol.qualified_name == "Cell.forward")
        call = next(row for row in index.calls_in(forward.symbol)
                    if _member(row.callee) == "second")
        bindings = SimpleNamespace(_modules={
            "cell": SimpleNamespace(init_attributes={}),
            "cell.first": object(), "cell.second": object()})
        retained = _member_stays_bound(index, forward, call, "cell", bindings)
        expected = change == "pass"
        results.append({"source": source, "change": change,
                        "member_stays_bound": retained, "expected": expected,
                        "passed": retained is expected})
after = hashlib.sha256(reader.read_bytes()).hexdigest()
print(json.dumps({"reader_sha256_before": before, "reader_sha256_after": after,
                  "reader_unchanged": before == after, "results": results}, indent=2))
raise SystemExit(0 if before == after and all(row["passed"] for row in results) else 1)
