"""Root-only serial launcher for actual frozen26 pages and declarations."""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pages-root", required=True, type=Path)
    args = parser.parse_args()
    tree = args.tree.resolve()
    if tree.name != "verify-s9-a-twentysixth":
        raise ValueError("launcher requires the immutable frozen26 worktree")
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("census.py").resolve()
    before_script = hashlib.sha256(source.read_bytes()).hexdigest()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", UNFOLD_EVIDENCE_CACHE_DIR=str(out / "cache"))
    spec = importlib.util.spec_from_file_location("s9_page_capture_verify", tree / "scripts/verify_commit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module._run_lane(module.Lane("actual-pages-declarations", (
        sys.executable, str(source), "--output", str(out / "capture"),
        "--pages-root", str(args.pages_root.resolve()))), tree, out)
    after_script = hashlib.sha256(source.read_bytes()).hexdigest()
    record = {**dataclasses.asdict(result), "log_path": str(result.log_path),
              "script_before": before_script, "script_after": after_script,
              "passed": result.passed and before_script == after_script}
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
