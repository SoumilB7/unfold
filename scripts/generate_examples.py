#!/usr/bin/env python3
"""Regenerate or verify the reviewed release showcase.

HTML and the exact SVG input of the hero image are deterministic product
surfaces. The reviewed PNG is a platform-produced artifact: its bytes are
sealed in the manifest, but ``--check`` deliberately does not re-rasterize it.
Different conforming SVG renderers may encode identical pixels differently.
The SVG-input seal makes a product change stale the PNG; the PNG seal makes an
unreviewed artifact change stale the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile

from model_unfolder import unfold
from model_unfolder.preview import (
    _with_clickable_highlight,
    architecture_svg,
    svg_to_png,
)


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

HERO = "llama-7b"
HERO_RENDER = {
    "background": "white",
    "highlight_clickable": True,
    "scale": 2.0,
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hero_svg_input(html: str) -> bytes:
    """Return the exact bytes passed to the canonical hero rasterizer."""
    svg = architecture_svg(html)
    if HERO_RENDER["highlight_clickable"]:
        svg = _with_clickable_highlight(svg)
    return svg.encode("utf-8")


def _render_rows(destination: Path, *, rasterize_hero: bool) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    images = destination / "images"
    if rasterize_hero:
        images.mkdir(exist_ok=True)

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
        product_html = diagram.to_html(standalone=True)
        html_path = destination / f"{public_name}.html"
        html_path.write_text(provenance + product_html, encoding="utf-8")
        row = {
            "file": html_path.name,
            "fixture": fixture_path.relative_to(ROOT).as_posix(),
            "rendered_name": diagram.to_ir()["name"],
            "sha256": _sha(html_path.read_bytes()),
        }
        if public_name == HERO:
            source = _hero_svg_input(product_html)
            hero = {
                "file": f"images/{HERO}.png",
                "render": dict(HERO_RENDER),
                "source_svg_sha256": _sha(source),
            }
            if rasterize_hero:
                png = destination / hero["file"]
                # ``source`` already contains the requested clickable overlay.
                svg_to_png(
                    source.decode("utf-8"), str(png),
                    scale=HERO_RENDER["scale"],
                    background=HERO_RENDER["background"],
                    highlight_clickable=False,
                )
                hero["sha256"] = _sha(png.read_bytes())
            row["hero_png"] = hero
        rows.append(row)
    return rows


def generate(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    images = destination / "images"
    images.mkdir(exist_ok=True)
    for stale in (*destination.glob("*.html"), *images.glob("*.png")):
        stale.unlink()

    manifest = {"schema": 2, "examples": _render_rows(
        destination, rasterize_hero=True)}
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Validate the minimum PNG envelope and return its IHDR dimensions."""
    if len(data) < 33 or data[:8] != PNG_SIGNATURE:
        raise ValueError("hero artifact is not a PNG")
    length = struct.unpack(">I", data[8:12])[0]
    if length != 13 or data[12:16] != b"IHDR":
        raise ValueError("hero artifact has no canonical IHDR")
    width, height = struct.unpack(">II", data[16:24])
    if width < 1 or height < 1:
        raise ValueError("hero artifact has invalid dimensions")
    if b"IEND" not in data[24:]:
        raise ValueError("hero artifact is truncated")
    return width, height


def _candidate_projection(rows: list[dict]) -> list[dict]:
    projected = json.loads(json.dumps(rows))
    for row in projected:
        if "hero_png" in row:
            row["hero_png"].pop("sha256", None)
    return projected


def _validation_errors(expected_root: Path, candidate_root: Path,
                       candidate_rows: list[dict]) -> list[str]:
    """Validate both deterministic product content and reviewed raster seals."""
    errors: list[str] = []
    manifest_path = expected_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"manifest unreadable: {exc}"]
    if manifest.get("schema") != 2 or not isinstance(manifest.get("examples"), list):
        return ["manifest must use release-example schema 2"]

    expected_rows = manifest["examples"]
    if _candidate_projection(expected_rows) != candidate_rows:
        errors.append("manifest metadata or hero SVG-input seal is stale")

    expected_html = {row.get("file") for row in expected_rows}
    candidate_html = {path.name for path in candidate_root.glob("*.html")}
    committed_html = {path.name for path in expected_root.glob("*.html")}
    if expected_html != candidate_html or expected_html != committed_html:
        errors.append("example HTML file set differs from the manifest")
    else:
        for name in sorted(expected_html):
            if (expected_root / name).read_bytes() != (candidate_root / name).read_bytes():
                errors.append(f"deterministic HTML is stale: {name}")

    hero_rows = [row for row in expected_rows if "hero_png" in row]
    if len(hero_rows) != 1:
        errors.append("manifest must carry exactly one reviewed hero PNG")
        return errors
    hero = hero_rows[0]["hero_png"]
    required = {"file", "render", "sha256", "source_svg_sha256"}
    if set(hero) != required or hero.get("render") != HERO_RENDER:
        errors.append("hero PNG receipt shape or render parameters are invalid")
        return errors
    png_path = expected_root / hero["file"]
    try:
        png = png_path.read_bytes()
        _png_dimensions(png)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    else:
        if _sha(png) != hero["sha256"]:
            errors.append("reviewed hero PNG bytes do not match their manifest seal")

    allowed = expected_html | {"manifest.json", hero["file"]}
    actual = {path.relative_to(expected_root).as_posix()
              for path in expected_root.rglob("*") if path.is_file()}
    if actual != allowed:
        errors.append("release-example directory has missing or unmanifested files")
    return errors


def check(output: Path = OUTPUT) -> int:
    # Verification renders only deterministic HTML/SVG. It never invokes the
    # platform rasterizer; the reviewed PNG is checked against its two seals.
    with tempfile.TemporaryDirectory(prefix="model-unfolder-examples-") as tmp:
        candidate = Path(tmp) / "examples"
        rows = _render_rows(candidate, rasterize_hero=False)
        errors = _validation_errors(output, candidate, rows)
    if errors:
        print("examples are stale")
        for error in errors:
            print("-", error)
        return 1
    print(f"examples are current: {len(rows)} reviewed models")
    return 0


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
