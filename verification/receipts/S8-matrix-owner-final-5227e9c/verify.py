"""Read-only full matrix row comparison against accepted S7; no model runs."""
from pathlib import Path
import collections, gzip, hashlib, json, subprocess, sys
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FINAL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "verification/s7"

def previous(relative):
    raw = subprocess.check_output(["git", "show", "83140f1:verification/s7/" + relative], cwd=ROOT)
    return json.loads(gzip.decompress(raw) if relative.endswith(".gz") else raw), hashlib.sha256(raw).hexdigest()

def current(relative):
    raw = (FINAL / relative).read_bytes()
    return json.loads(gzip.decompress(raw) if relative.endswith(".gz") else raw), hashlib.sha256(raw).hexdigest()

old_matrix, old_hash = previous("matrix.json")
new_matrix, new_hash = current("matrix.json")
assert old_matrix["denominator"] == new_matrix["denominator"] == {"corpus": 29, "to_serve": 10}
old_summaries = {r["slug"]: r for r in old_matrix["models"]}
new_summaries = {r["slug"]: r for r in new_matrix["models"]}
assert old_summaries.keys() == new_summaries.keys() and len(new_summaries) == 39
rows = []
for slug in sorted(new_summaries):
    old, old_pin = previous(f"models/{slug}.json.gz")
    new, new_pin = current(f"models/{slug}.json.gz")
    a, b = old["table"]["occurrences"], new["table"]["occurrences"]
    assert len(a) == len(b)
    assert old["table"]["config_sha256"] == new["table"]["config_sha256"]
    assert old["table"]["relations"] == new["table"]["relations"]
    assert old["table"]["input_failures"] == new["table"]["input_failures"]
    deltas = []
    for x, y in zip(a, b):
        assert x["provenance"]["instance_path"] == y["provenance"]["instance_path"]
        assert x["construction"] == y["construction"]
        changed = sorted(k for k in x.keys() | y.keys() if x.get(k) != y.get(k))
        if changed:
            deltas.append({"path": x["provenance"]["instance_path"], "fields": changed})
        if x["execution"] != y["execution"]:
            assert x["provenance"]["instance_path"] == ""
            assert x["execution"]["kind"] == "execution_unresolved" and y["execution"]["kind"] == "observed"
    root_delta = any("execution" in r["fields"] for r in deltas)
    raw_pin = None
    pilot_pins = {}
    if root_delta:
        raw, raw_pin = previous(f"observations/{slug}.json.gz")
        observed_ids = {attempt["recipe"]["recipe_id"] for attempt in raw["attempts"]
                        if attempt["status"] == "ok" and any(c["path"] == "" for c in (attempt.get("observation") or {}).get("module_calls", []))}
        pilot_paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "83140f1", "--", f"verification/s6/pilots/{slug}"], cwd=ROOT, text=True).splitlines()
        for pilot_path in pilot_paths:
            if not Path(pilot_path).name.startswith("observation-"):
                continue
            pilot_bytes = subprocess.check_output(["git", "show", "83140f1:" + pilot_path], cwd=ROOT)
            pilot = json.loads(pilot_bytes)
            if pilot["status"] == "ok" and pilot["provenance"]["config_sha256"] == old["table"]["config_sha256"]:
                if any(c["path"] == "" for c in pilot["observation"]["module_calls"]):
                    observed_ids.add(pilot["observation"]["recipe"]["recipe_id"])
                    pilot_pins[pilot_path] = hashlib.sha256(pilot_bytes).hexdigest()
        assert set(b[0]["execution"]["recipe_ids"]) <= observed_ids and observed_ids, (slug, b[0]["execution"]["recipe_ids"], observed_ids)
    if slug != "stable-diffusion-xl-base-1-0":
        assert not deltas or deltas == [{"path": "", "fields": ["execution"]}]
        disposition = "Historical empty-root positive joined" if deltas else "All occurrence axes and relations unchanged"
    else:
        assert len(deltas) == 1928
        assert all(set(r["fields"]) <= {"projection", "provenance", "execution"} for r in deltas)
        assert sum(x["projection"]["kind"] == "rendered" for x in b) == 1928
        assert sum(x["projection"]["kind"] == "non_architectural" for x in b) == 2
        assert not any(x["projection"]["fact_findings"] for x in b)
        assert all(x["construction"]["kind"] == "not_constructed" for x in b if x["projection"]["kind"] == "projection_unresolved")
        disposition = "Reviewed qualified UNet placement/provenance plus historical empty-root join; exact per-output owner ledgers and family receipt retained"
    rows.append({"slug": slug, "disposition": disposition, "root_positive_was_in_accepted_raw_observation": root_delta,
                 "old_model_sha256": old_pin, "new_model_sha256": new_pin, "old_raw_observation_sha256": raw_pin, "old_pilot_observation_sha256": pilot_pins,
                 "occurrence_deltas": deltas, "relations_identical": True, "construction_identical": True,
                 "summary_before": old_summaries[slug], "summary_after": new_summaries[slug]})
summary = {"status": "PASS bounded owner matrix review", "denominator": 39, "unexplained": 0, "blessed": False,
           "old_matrix_sha256": old_hash, "new_matrix_sha256": new_hash,
           "root_join_repairs": sum(r["root_positive_was_in_accepted_raw_observation"] for r in rows),
           "all_relation_rows_unchanged": True, "all_construction_axes_unchanged": True,
           "recipe_status_counts": dict(collections.Counter(r["recipe_status"] for r in new_matrix["models"]))}
(HERE / "rows.json.gz").write_bytes(gzip.compress((json.dumps(rows, indent=2, sort_keys=True) + "\n").encode(), mtime=0))
(HERE / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary))
