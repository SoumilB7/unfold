"""Render the baked diagram to PNG images — pixels as a first-class oracle.

The bugs that keep slipping (a `×`/`⊙`/`⊕` with a missing input, an unclickable
"block", a crooked/overlapping arrow) are *visual* facts: coupling-clean HTML and
green structural tests cannot see them. The only thing that can is the rendered
image. So seeing the output as an image is a built-in operation, not an ad-hoc
``rsvg-convert`` dance done by hand.

Every diagram view — the top architecture diagram and every drill a click opens,
at every depth — is baked as an ``<svg>`` in the standalone HTML up front (the
project's ground-truth invariant: the JS only toggles visibility, it builds
nothing). Two facts shape what we image:

* **Description-only leaf cards carry no ``<svg>``** and so get no image (we don't
  want pictures of prose) — the natural, correct stopping point of each drill.
* **Each repeated layer-group bakes its OWN scoped copy** of the drill panels, so
  one architecture renders many byte-identical svgs (4× the same ``attn``, 12× the
  same ``expert``…). Imaging all of them is not exhaustiveness — it is noise that
  *hides* problems (you glance at one of four identical pictures). Exhaustive means
  every **distinct** diagram, to its leaves, each looked at exactly once.

So this module pulls **every** baked svg with its drill path, **deduplicates by
visual identity** (ids/instance refs normalised away — pixel-identical ⇒ one
image), writes a ``MANIFEST.txt`` recording every collapse and skip, and converts
each distinct diagram with ``rsvg-convert``.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser


def default_image_dir(model_name: str) -> str:
    """Default output dir: ``previews/individual_images/<model>``.

    The ``previews/`` folder is found by walking up from the cwd (so it resolves
    whether run from the repo root or ``unfold-pkg/``); if none exists, one is
    created under the cwd."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model_name or "model")
    d = os.getcwd()
    base = None
    while True:
        if os.path.isdir(os.path.join(d, "previews")):
            base = os.path.join(d, "previews")
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    base = base or os.path.join(os.getcwd(), "previews")
    return os.path.join(base, "individual_images", safe)


def _ensure_rsvg() -> None:
    if shutil.which("rsvg-convert") is None:
        raise RuntimeError(
            "rsvg-convert not found — install librsvg (macOS: `brew install librsvg`) "
            "to render diagram images."
        )


def _line_offsets(text: str) -> list[int]:
    offs, total = [0], 0
    for line in text.splitlines(keepends=True):
        total += len(line)
        offs.append(total)
    return offs


class _ViewExtractor(HTMLParser):
    """Walk the HTML tracking the stack of enclosing panels so every ``<svg>`` is
    captured with a label: the ``data-card-id`` drill path for a drill view, or
    the ``uf-arch-variant-N`` index for a top architecture diagram. Captures
    **every** svg — completeness by construction, at any drill depth."""

    def __init__(self, html: str):
        super().__init__(convert_charrefs=False)
        self._html = html
        self._offs = _line_offsets(html)
        self._stack: list[tuple[str | None, str]] = []   # (data-card-id, class) per open tag
        self.views: list[tuple[str, str]] = []           # (label, svg_text)

    def _offset(self) -> int:
        line, col = self.getpos()
        return self._offs[line - 1] + col

    def handle_starttag(self, tag, attrs):
        if tag == "svg":
            path = [cid for cid, _ in self._stack if cid]
            if path:
                label = "/".join(path)
            else:
                variant = next((m.group(1) for _, cls in reversed(self._stack)
                                if (m := re.search(r"uf-arch-variant-(\d+)", cls or ""))), None)
                label = "architecture" if variant in (None, "0") else f"architecture_v{variant}"
            start = self._offset()
            end = self._html.index("</svg>", start) + len("</svg>")
            self.views.append((label, self._html[start:end]))
        a = dict(attrs)
        self._stack.append((a.get("data-card-id") if tag == "div" else None, a.get("class", "")))

    def handle_endtag(self, tag):
        if self._stack:
            self._stack.pop()


def svg_views(html: str) -> list[tuple[str, str]]:
    """Every baked diagram view as ``(label, svg)`` in document order — labelled
    by drill path / architecture variant. Includes the per-layer-group duplicate
    copies; :func:`render_images` is what dedups them. Leaf cards never appear."""
    ex = _ViewExtractor(html)
    ex.feed(html)
    return ex.views


def architecture_svg(html: str) -> str:
    """The top architecture-view svg (the first baked view)."""
    views = svg_views(html)
    if not views:
        raise RuntimeError("no <svg> views found in the rendered HTML")
    return views[0][1]


def _visual_hash(svg: str) -> str:
    """Identity of a diagram by what it DRAWS — per-instance ids / ref targets
    normalised away, text and geometry kept. Pixel-identical diagrams hash equal."""
    s = re.sub(r'\b(id|data-id|data-card-id|data-target|href|xlink:href|aria-labelledby)="[^"]*"', "", svg)
    s = re.sub(r"url\(#[^)]*\)", "url()", s)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def svg_to_png(svg: str, path: str, *, scale: float = 2.0, background: str = "white") -> str:
    """Convert one svg string to a PNG file via ``rsvg-convert``."""
    _ensure_rsvg()
    subprocess.run(
        ["rsvg-convert", "-b", background, "-z", str(scale), "-o", path],
        input=svg.encode("utf-8"), check=True,
    )
    return path


def render_images(diagram, outdir: str, *, scale: float = 2.0,
                  background: str = "white", dedup: bool = True) -> list[str]:
    """Render every DISTINCT diagram view (architecture + every drill, to the
    leaves) to a PNG in *outdir*, plus a ``MANIFEST.txt``.

    With ``dedup`` (default) the per-layer-group duplicate copies collapse to one
    image each — exhaustive over distinct diagrams, nothing identical repeated.
    Returns the written image paths."""
    _ensure_rsvg()
    os.makedirs(outdir, exist_ok=True)
    html = diagram.to_html(standalone=True)
    views = svg_views(html)

    # Group by visual identity (or keep all if dedup off).
    distinct: list[dict] = []
    by_hash: dict[str, dict] = {}
    for label, svg in views:
        key = _visual_hash(svg) if dedup else f"{len(distinct)}"
        if key in by_hash:
            by_hash[key]["aliases"].append(label)
            continue
        entry = {"label": label, "svg": svg, "aliases": []}
        by_hash[key] = entry
        distinct.append(entry)

    # Unique, filesystem-safe filenames (distinct diagrams may share a base label).
    used: dict[str, int] = {}
    paths, manifest = [], []
    for i, entry in enumerate(distinct):
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", entry["label"])
        n = used.get(base, 0)
        used[base] = n + 1
        name = base if not n else f"{base}__{n}"
        fname = f"{i:02d}__{name}.png"
        svg_to_png(entry["svg"], os.path.join(outdir, fname), scale=scale, background=background)
        paths.append(os.path.join(outdir, fname))
        dup = f"   (+{len(entry['aliases'])} identical copies collapsed)" if entry["aliases"] else ""
        manifest.append(f"{fname}{dup}")

    n_leaves = len(re.findall(r'data-card-id="', html)) - len(views)
    collapsed = len(views) - len(distinct)
    with open(os.path.join(outdir, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {len(distinct)} DISTINCT diagram views (architecture + every drill, to the leaves)\n")
        f.write(f"# {len(views)} baked svgs total — {collapsed} identical per-layer-group copies collapsed\n")
        f.write(f"# {max(n_leaves, 0)} description-only leaf cards skipped (no diagram)\n\n")
        f.write("\n".join(manifest) + "\n")
    return paths
