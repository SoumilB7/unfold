"""Capture exact accepted-S7 sidecars; never update a preservation baseline."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("slugs", nargs="+")
    args = parser.parse_args()
    source = args.source.resolve()
    pins = json.loads(args.pins.read_text())
    for relative, expected in pins["files"].items():
        actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Snapshot source changed: " + relative)
    if args.output.exists():
        raise ValueError("Use a fresh output directory; preserve earlier attempts")
    args.output.mkdir(parents=True)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE"):
        os.environ[key] = "1"
    sys.path.insert(0, str(source))
    import model_unfolder as mu
    from test_support import preservation as preservation

    if not Path(mu.__file__).resolve().is_relative_to(source):
        raise ValueError("Model parser did not import the pinned source")
    if not Path(preservation.__file__).resolve().is_relative_to(source):
        raise ValueError("Comparator did not import the pinned source")
    corpus = source / "tests" / "sable_test_corpus"
    manifest = json.loads((source / "tests" / "preservation_expected_manifest.json").read_text())
    results = []
    for slug in args.slugs:
        expected = manifest["witnesses"][slug]
        input_path = corpus / (slug + ".json")
        if preservation.input_sha256(input_path) != expected["input_sha256"]:
            raise ValueError("Frozen input changed: " + slug)
        target = args.output / slug
        target.mkdir()
        started = time.monotonic()
        print("START " + slug, flush=True)
        config = json.loads(input_path.read_text())["config"]
        diagram = mu.unfold(config)
        ir = diagram.to_ir()
        _, sidecar = preservation.split_structural_ir(ir)
        payload = preservation._canon_bytes(sidecar)
        actual_sha = hashlib.sha256(payload).hexdigest()
        (target / "ir.json.gz").write_bytes(gzip.compress(preservation._canon_bytes(ir), mtime=0))
        (target / "ledgers.json.gz").write_bytes(gzip.compress(payload, mtime=0))
        row = {
            "slug": slug,
            "source_commit": pins["commit"],
            "input_sha256": expected["input_sha256"],
            "expected_ledger_sha256": expected["surfaces"]["ledgers"],
            "actual_ledger_sha256": actual_sha,
            "expected_hash_recovered": actual_sha == expected["surfaces"]["ledgers"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "parser_module": mu.__file__,
            "comparator_module": preservation.__file__,
        }
        (target / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if not row["expected_hash_recovered"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
