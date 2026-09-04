#!/usr/bin/env python3
"""Regenerate the release showcase from reviewed, frozen corpus inputs.

The old showcase fetched mutable model IDs and eventually mislabeled a Gemma
render as DeepSeek.  S5 makes every public example a deterministic projection of
an exact reviewed fixture.  The script deletes stale generated pages, fixes the
mount id, records the source fixture/hash, and can compare a fresh temporary
render against the committed directory with ``--check``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from model_unfolder import unfold


ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests" / "sable_test_corpus"
OUTPUT = ROOT / "examples"

EXAMPLES = {
    "deepseek-v3": "deepseek-v3",
    "granite-3-0-8b-instruct": "granite-3-0-8b-instruct",
    "hunyuanvideo": "hunyuanvideo",
    "llama-7b": "llama-7b",
    "musicgen-small": "musicgen-small",
    "pixart-sigma-xl-2-1024-ms": "pixart-sigma-xl-2-1024-ms",
    "qwen2-vl-7b-instruct": "qwen2-vl-7b-instruct",
    "stable-diffusion-xl-base-1-0": "stable-diffusion-xl-base-1-0",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    images = destination / "images"
    images.mkdir(exist_ok=True)
    for stale in (*destination.glob("*.html"), *images.glob("*.png")):
        stale.unlink()

    rows = []
    for public_name, fixture_name in EXAMPLES.items():
        fixture_path = CORPUS / f"{fixture_name}.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        config = fixture["config"]
        diagram = unfold(config)
        diagram._mount_id = f"uf-example-{public_name}"
        provenance = (
            "<!-- data-generated-by=scripts/generate_examples.py "
            f"source={fixture_path.relative_to(ROOT).as_posix()} "
            f"input-sha256={_sha(json.dumps(config, sort_keys=True, separators=(',', ':')).encode())} -->\n"
        )
        html_path = destination / f"{public_name}.html"
        html_path.write_text(
            provenance + diagram.to_html(standalone=True), encoding="utf-8")
        row = {
            "file": html_path.name,
            "fixture": fixture_path.relative_to(ROOT).as_posix(),
            "rendered_name": diagram.to_ir()["name"],
            "sha256": _sha(html_path.read_bytes()),
        }
        if public_name == "llama-7b":
            png = images / "llama-7b.png"
            diagram.to_png(str(png))
            row["hero_png"] = {
                "file": png.relative_to(destination).as_posix(),
                "sha256": _sha(png.read_bytes()),
            }
        rows.append(row)

    manifest = {"schema": 1, "examples": rows}
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def check() -> int:
    with tempfile.TemporaryDirectory(prefix="model-unfolder-examples-") as tmp:
        candidate = Path(tmp) / "examples"
        generate(candidate)
        expected = _files(OUTPUT)
        actual = _files(candidate)
        if expected == actual:
            print(f"examples are current: {len(actual)} files")
            return 0
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(expected) & set(actual)
                         if expected[name] != actual[name])
        print("examples are stale")
        print("missing from candidate:", missing)
        print("extra in candidate:", extra)
        print("byte-changed:", changed)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    manifest = generate(OUTPUT)
    print(f"generated {len(manifest['examples'])} reviewed examples in {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
