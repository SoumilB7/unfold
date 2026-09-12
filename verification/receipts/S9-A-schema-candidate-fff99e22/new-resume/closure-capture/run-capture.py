"""Root-only serial before/after closure replay with independent brackets."""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-tree", required=True, type=Path)
    parser.add_argument("--after-tree", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    plan = here / "targets.json"
    source = here / "capture.py"
    expected_commit = json.loads(plan.read_text())["before_commit"]
    pins_before = {path.name: sha(path) for path in (plan, source, Path(__file__))}
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for phase, supplied in (("before", args.before_tree), ("after", args.after_tree)):
        tree = supplied.resolve()
        name = "verify-s9-a-closure-baseline" if phase == "before" else "verify-s9-a-twentysixth"
        if tree.name != name:
            raise ValueError("phase worktree name differs from the reviewed plan")
        if phase == "before":
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tree, text=True,
                                  capture_output=True, check=True).stdout.strip()
            status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                                    cwd=tree, text=True, capture_output=True, check=True).stdout.strip()
            if head != expected_commit or status:
                raise ValueError("before tree must be clean at fff99e22")
        lane = out / phase
        lane.mkdir()
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", UNFOLD_EVIDENCE_CACHE_DIR=str(lane / "cache"))
        spec = importlib.util.spec_from_file_location("closure_verify_" + phase, tree / "scripts/verify_commit.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result = module._run_lane(module.Lane("portable-source-closure", (
            sys.executable, str(source), "--plan", str(plan), "--phase", phase,
            "--output", str(lane / "capture"))), tree, lane)
        row = {**dataclasses.asdict(result), "log_path": str(result.log_path),
               "phase": phase, "passed": result.passed}
        results.append(row)
        (lane / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        # A typed mismatch in the before replay does not suppress after data.
    pins_after = {path.name: sha(path) for path in (plan, source, Path(__file__))}
    record = {"scope": "Two serial 39-target closure captures with clean baseline HTML/IR; no architecture acceptance.",
              "pins_before": pins_before, "pins_after": pins_after, "lanes": results,
              "passed": pins_before == pins_after and all(row["passed"] for row in results)}
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
