"""Root-owned exact37 capture; source, script and address-input brackets."""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    plan_path = here / "plan.json"
    plan = json.loads(plan_path.read_text())
    tree = Path(plan["tree"])
    if tree.name != "verify-s9-a-thirtyseventh" or tree.resolve() != tree:
        raise ValueError("capture requires the reviewed immutable37 worktree")
    from capture_contract import validate_execution_inputs

    def pins():
        return validate_execution_inputs(plan_path, args.source_manifest_sha256)

    before = pins()
    out = Path(plan["output_root"])
    if Path(plan["capture_output"]) != out / "capture":
        raise ValueError("capture path differs from the reviewed output root")
    out.mkdir(parents=True, exist_ok=False)
    (out / "pins-before.json").write_text(json.dumps(before, indent=2) + "\n")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", UNFOLD_EVIDENCE_CACHE_DIR=str(out / "cache"))
    spec = importlib.util.spec_from_file_location("current37_capture_verify", tree / "scripts/verify_commit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module._run_lane(module.Lane("actual37-pages-presentation-closures", (
        sys.executable, str(here / "census.py"), "--plan", str(plan_path),
        "--output", plan["capture_output"], "--pages-root", plan["pages_root"],
        "--source-manifest-sha256", args.source_manifest_sha256)), tree, out)
    after = pins()
    (out / "pins-after.json").write_text(json.dumps(after, indent=2) + "\n")
    record = {**dataclasses.asdict(result), "log_path": str(result.log_path),
              "source_manifest_sha256": args.source_manifest_sha256,
              "pins_equal": before == after, "passed": result.passed and before == after,
              "scope": "Actual37 capture and presentation findings; no blessing, browser verdict or normalization."}
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
