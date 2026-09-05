#!/usr/bin/env python3
"""Regenerate/check the S4 29+15 no-silence support manifest.

The inputs are frozen config documents.  They provide checkpoint values; the
installed source bundle remains the authority for mechanism evidence.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_unfolder import sable
from model_unfolder.evidence.coverage import coverage_problems

CORPUS = ROOT / "tests" / "sable_test_corpus"
UNSEEN = ROOT / "tests" / "unseen_model_configs"
OUTPUT = ROOT / "coverage.json"


def _inputs():
    for path in sorted(CORPUS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        model = str(payload.get("model") or config.get("_name_or_path") or path.stem)
        yield "corpus", path, model, config
    for path in sorted(UNSEEN.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield "unseen", path, str(payload["model_id"]), payload["config"]


def generate() -> dict:
    rows = []
    for cohort, path, model, config in _inputs():
        try:
            report = sable(config, render_images=False)
            coverage = report.coverage
            row = {
                "cohort": cohort,
                "flagged": coverage["flagged"],
                "flagged_findings": coverage.get("flagged_findings") or [],
                "input": str(path.relative_to(ROOT)),
                "model": model,
                "proven": coverage["proven"],
                "rendered_name": report.model,
                "silent": coverage["silent"],
                "silent_findings": coverage.get("silent_findings") or [],
            }
        except Exception as exc:
            # A crash is part of the denominator and is silent by definition;
            # recording it makes the manifest fail instead of truncating.
            row = {
                "cohort": cohort,
                "flagged": 0,
                "flagged_findings": [],
                "input": str(path.relative_to(ROOT)),
                "model": model,
                "proven": 0,
                "rendered_name": "",
                "silent": 1,
                "silent_findings": [f"{type(exc).__name__}: {exc}"],
            }
        rows.append(row)
        print(
            f"{cohort:6} {model}: proven={row['proven']} "
            f"flagged={row['flagged']} silent={row['silent']}",
            file=sys.stderr,
            flush=True,
        )
    import diffusers
    import huggingface_hub
    import transformers
    return {
        "schema": 1,
        "source_environment": {
            "diffusers": diffusers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "transformers": transformers.__version__,
        },
        "models": rows,
    }


def _render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    document = generate()
    problems = coverage_problems(document)
    rendered = _render(document)
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != rendered:
            print("coverage.json is stale", file=sys.stderr)
            print("".join(difflib.unified_diff(
                current.splitlines(True), rendered.splitlines(True),
                fromfile="coverage.json", tofile="generated")), file=sys.stderr)
            return 1
    else:
        OUTPUT.write_text(rendered, encoding="utf-8")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
