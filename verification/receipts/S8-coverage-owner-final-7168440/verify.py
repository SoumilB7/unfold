"""Read-only independent comparison of actual final44 coverage artifacts."""
from pathlib import Path
import hashlib
import json
import sys


def read(path):
    return json.loads(path.read_bytes())


def main():
    final = Path(sys.argv[1])
    root = Path(__file__).resolve().parents[3]
    receipt = Path(__file__).resolve().parent
    prior_dir = root / "verification/receipts/S8-coverage-5227e9c-return"
    paths = {"final": final, "accepted": prior_dir / "accepted-coverage.json",
             "prior_red": prior_dir / "current-coverage.json"}
    docs = {key: read(path) for key, path in paths.items()}
    rows = {key: {row["input"]: row for row in doc["models"]}
            for key, doc in docs.items()}
    assert all(len(doc["models"]) == len(rows[key]) == 44
               for key, doc in docs.items())
    assert rows["final"].keys() == rows["accepted"].keys() == rows["prior_red"].keys()
    assert {row["cohort"] for row in rows["final"].values()} == {"corpus", "unseen"}
    assert sum(row["cohort"] == "corpus" for row in rows["final"].values()) == 29
    changed = []
    deltas = []
    for key, after in rows["final"].items():
        before = rows["accepted"][key]
        red = rows["prior_red"][key]
        assert after["silent"] == 0 and after["silent_findings"] == [], key
        assert {k: v for k, v in after.items() if k not in {"silent", "silent_findings"}} == {
            k: v for k, v in red.items() if k not in {"silent", "silent_findings"}}, key
        differences = {k: {"before": before.get(k), "after": after.get(k)}
                       for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        if differences:
            changed.append(key)
        deltas.append({"input": key, "differences_vs_accepted": differences,
                       "removed_silent_findings": red["silent_findings"]})
    assert set(changed) == {"tests/sable_test_corpus/stable-diffusion-xl-base-1-0.json",
                            "tests/unseen_model_configs/sd-v1-4.json"}, changed
    totals = {key: {metric: sum(row[metric] for row in values.values())
                    for metric in ("proven", "flagged", "silent")}
              for key, values in rows.items()}
    assert totals["final"] == {"proven": 645, "flagged": 294, "silent": 0}, totals
    assert totals["accepted"] == {"proven": 621, "flagged": 241, "silent": 0}, totals
    assert totals["prior_red"] == {"proven": 645, "flagged": 294, "silent": 620}, totals
    result = {"status": "PASS", "denominator": 44,
              "other42_complete_coverage_rows_equal_accepted": True,
              "all44_non_silent_fields_equal_prior_red": True,
              "totals": totals, "changed_inputs": changed,
              "pins": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for key, path in paths.items()},
              "new_model_execution_by_this_audit": False, "blessed": False}
    (receipt / "per-model-deltas.json").write_text(json.dumps(deltas, indent=2, sort_keys=True) + "\n")
    (receipt / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
