"""Replay trace selection against saved actual HTML observations; no model run."""
import hashlib
import json
from pathlib import Path
from scripts.report_s8_demonstration import blocks

out = Path(__file__).resolve().parent
reporter = out.parents[2] / "scripts/report_s8_demonstration.py"
source = reporter.read_bytes()
source_hash = hashlib.sha256(source).hexdigest()
if (out / "results.json").exists():
    assert json.loads((out / "results.json").read_text())["reporter_sha256"] == source_hash
base = Path("/private/tmp/unfold-s8-final-4560449/sdxl/ordinary")
raw = {name: (base / (name + ".json")).read_bytes() for name in ("ir", "facts", "observation")}
ir, facts, observation = (json.loads(raw[name]) for name in ("ir", "facts", "observation"))
all_blocks = list(blocks(ir["extras"]["render"]))
by = {row["source_instance_path"]: row for row in all_blocks if row.get("source_instance_path")}
visible = observation["page"]["cards"]["denoiser"]["node_ids"]
rows = []
for short in ("ffn_mechanisms", "cell_connections", "spatial_mechanisms"):
    occurrence = next(path for path in sorted(facts["root.denoiser." + short]["value"]) if path in by)
    stage = next(row for path, row in by.items() if occurrence.startswith(path + ".") and path.count(".") == 1)
    rows.append({
        "claim": short, "occurrence": occurrence,
        "selected_stage_id": stage["id"], "selected_stage_visible": stage["id"] in visible,
        "same_occurrence_stage_cards": [
            {"id": row["id"], "visible": row["id"] in visible}
            for row in all_blocks
            if row.get("source_instance_path") == ".".join(occurrence.split(".")[:2])],
    })
assert reporter.read_bytes() == source
result = {"scope": "Trace report selection only; saved actual 4560449 artifacts",
          "reporter_sha256": source_hash, "source_unchanged": True,
          "artifact_paths": {name: str(base / (name + ".json")) for name in raw},
          "artifact_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
          "rows": rows}
(out / "reporter.py.txt").write_bytes(source)
(out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(rows, indent=2))
