"""Root-owned exact28 capture; source, script and address-input brackets."""
from pathlib import Path
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    plan_path = here / "plan.json"
    plan = json.loads(plan_path.read_text())
    tree = Path(plan["tree"])
    if tree.name != "verify-s9-a-twentyeighth" or tree.resolve() != tree:
        raise ValueError("capture requires the reviewed immutable28 worktree")
    source_manifest = Path(plan["source_manifest"])
    if sha(source_manifest) != plan["source_manifest_sha256"]:
        raise ValueError("frozen28 source manifest changed")
    source_rows = json.loads(source_manifest.read_text())
    scripts = [here / name for name in (
        "run-census.py", "census.py", "portable_source.py", "plan.json", "manifest.json")]
    manifest = json.loads((here / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(here / name) != expected:
            raise ValueError("capture source differs from its held manifest: " + name)

    def pins():
        sources = {}
        for row in source_rows:
            path = tree / row["path"]
            actual = sha(path) if path.is_file() else None
            if actual != row["sha256"]:
                raise ValueError("frozen28 source differs: " + row["path"])
            sources[row["path"]] = actual
        inputs = {}
        for target in plan["targets"]:
            for name, expected in ((target["input"], target["input_sha256"]),
                                   (target["inventory"], target["inventory_sha256"])):
                inputs[name] = sha(tree / name)
                if inputs[name] != expected:
                    raise ValueError("reviewed address input changed: " + name)
        inputs["verification/s7/matrix.json"] = sha(tree / "verification/s7/matrix.json")
        if inputs["verification/s7/matrix.json"] != plan["active_matrix_sha256"]:
            raise ValueError("active39 address matrix changed")
        return {"scripts": {path.name: sha(path) for path in scripts},
                "source_manifest_sha256": sha(source_manifest),
                "source": sources, "inputs": inputs}

    before = pins()
    out = Path(plan["output_root"])
    if Path(plan["capture_output"]) != out / "capture":
        raise ValueError("capture path differs from the reviewed output root")
    out.mkdir(parents=True, exist_ok=False)
    (out / "pins-before.json").write_text(json.dumps(before, indent=2) + "\n")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", UNFOLD_EVIDENCE_CACHE_DIR=str(out / "cache"))
    spec = importlib.util.spec_from_file_location("current28_capture_verify", tree / "scripts/verify_commit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module._run_lane(module.Lane("actual28-pages-presentation-closures", (
        sys.executable, str(here / "census.py"), "--plan", str(plan_path),
        "--output", plan["capture_output"], "--pages-root", plan["pages_root"])), tree, out)
    after = pins()
    (out / "pins-after.json").write_text(json.dumps(after, indent=2) + "\n")
    record = {**dataclasses.asdict(result), "log_path": str(result.log_path),
              "source_manifest_sha256": plan["source_manifest_sha256"],
              "pins_equal": before == after, "passed": result.passed and before == after,
              "scope": "Actual28 capture and presentation findings; no blessing, browser verdict or normalization."}
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
