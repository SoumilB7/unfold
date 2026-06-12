"""Patch-grid geometry -> display strings.

One formatter over the normalized ``grid`` object the parser emits
(``modalities.inputs.vision.embedding.grid``).  Square, non-square,
dynamic-resolution, temporal (video), and patch-merged towers all flow
through the same functions instead of each needing a new branch in the
renderer.  Every function degrades gracefully when fields are missing — it
returns a generic phrase, never raises and never omits a node.
"""
from __future__ import annotations

from .utils import _fmt_int


def coerce_grid(grid: dict | None, image_size=None, patch_size=None) -> dict | None:
    """Return a grid object, synthesizing a minimal square one from scalars.

    The parser now attaches ``grid`` directly; the scalar arguments are a
    fallback for older/partial paths so callers never have to branch.
    """
    if grid:
        return grid
    p = _as_int(patch_size)
    img = _as_int(image_size)
    if p is None and img is None:
        return None
    tiles = None
    inp = None
    if img is not None:
        inp = {"h": img, "w": img}
        if p and img % p == 0:
            tiles = {"h": img // p, "w": img // p}
    return {
        "kind": "static_patch_grid",
        "patch": {"h": p, "w": p} if p else {},
        "input": inp,
        "tiles": tiles,
    }


def grid_title(grid: dict | None) -> str:
    """Compact title, e.g. ``"32x32 patch grid"`` / ``"dynamic patch grid"``."""
    if not grid:
        return "patch grid"
    tiles = grid.get("tiles") or {}
    if tiles.get("h") and tiles.get("w"):
        dims = f"{_fmt_int(tiles['h'])}×{_fmt_int(tiles['w'])}"
        t = (grid.get("patch") or {}).get("t")
        if t:
            dims += f"x{_fmt_int(t)}"
        return f"{dims} patch grid"
    if grid.get("kind") == "dynamic_patch_grid":
        return "dynamic patch grid"
    return "patch grid"


def grid_subtitle(grid: dict | None) -> str:
    """Detail line, e.g. ``"14px patch from 448px image"``."""
    if not grid:
        return "image split into patch tiles"
    patch_str = _patch_phrase(grid)
    image_str = _image_phrase(grid)
    if patch_str and image_str:
        sep = " from " if grid.get("input") else " · "
        line = f"{patch_str}{sep}{image_str}"
    else:
        line = patch_str or image_str or "image split into patch tiles"
    merge = grid.get("spatial_merge_size")
    if merge:
        line += f" · {_fmt_int(merge)}×{_fmt_int(merge)} merge"
    return line


def grid_card_phrase(grid: dict | None) -> str | None:
    """Phrase for the metadata card: ``"Split image into {phrase}"``.

    Returns ``None`` when there is nothing concrete to say so the caller can
    fall back to a bare "patches" label.
    """
    if not grid:
        return None
    p = grid.get("patch") or {}
    ph, pw = p.get("h"), p.get("w")
    if ph and pw:
        patch_px = f"{_fmt_int(ph)}px each" if ph == pw else f"{_fmt_int(ph)}×{_fmt_int(pw)}px each"
    else:
        patch_px = None
    tiles = grid.get("tiles") or {}
    if tiles.get("h") and tiles.get("w"):
        dims = f"{_fmt_int(tiles['h'])}×{_fmt_int(tiles['w'])}"
        return f"{dims} patches ({patch_px})" if patch_px else f"{dims} patches"
    if grid.get("kind") == "dynamic_patch_grid":
        return f"patches ({patch_px}); grid size varies with input" if patch_px else "patches; grid size varies with input"
    return f"patches ({patch_px})" if patch_px else None


def _patch_phrase(grid: dict) -> str | None:
    p = grid.get("patch") or {}
    ph, pw, t = p.get("h"), p.get("w"), p.get("t")
    if not ph and not pw:
        return None
    if ph and pw and ph != pw:
        s = f"{_fmt_int(ph)}×{_fmt_int(pw)}px patch"
    else:
        s = f"{_fmt_int(ph or pw)}px patch"
    if t:
        s += f" \u00d7 {_fmt_int(t)} frames"
    return s


def _image_phrase(grid: dict) -> str | None:
    inp = grid.get("input") or {}
    ih, iw = inp.get("h"), inp.get("w")
    if ih and iw:
        return f"{_fmt_int(ih)}px image" if ih == iw else f"{_fmt_int(ih)}×{_fmt_int(iw)}px image"
    if grid.get("kind") == "dynamic_patch_grid":
        return "grid size varies with input"
    return None


def _as_int(value) -> int | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = ["coerce_grid", "grid_title", "grid_subtitle", "grid_card_phrase"]
