"""Read-only owner receipt verification; no models, rendering, or blessing."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
result = json.loads((HERE / "result.json").read_text())
for relative, expected in result["input_sha256"].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
base = ROOT / result["source_receipt"]
for name in result["matrix_gzip_byte_identity"]:
    assert (base / "pixart-before" / name).read_bytes() == (base / "pixart-after" / name).read_bytes()

def differences(a, b, path="$", rows=None):
    rows = [] if rows is None else rows
    if type(a) is not type(b):
        rows.append({"path": path, "before": a, "after": b})
    elif isinstance(a, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                rows.append({"path": path + "." + key, "before_present": key in a,
                             "after_present": key in b, "before": a.get(key), "after": b.get(key)})
            else:
                differences(a[key], b[key], path + "." + key, rows)
    elif isinstance(a, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            differences(x, y, path + f"[{i}]", rows)
    elif a != b:
        rows.append({"path": path, "before": a, "after": b})
    return rows

for name, rows in json.loads((HERE / "delta-ledger.json").read_text()).items():
    before, after = [json.loads(gzip.decompress((base / side / name).read_bytes()))
                     for side in ["pixart-before", "pixart-after"]]
    expected = [{k: v for k, v in row.items() if k != "owner_disposition"} for row in rows]
    assert differences(before, after) == expected
assert result["unexplained"] == 0 and result["blessed"] is False
print("PASS: input pins, matrix bytes, and complete before/after delta values")
