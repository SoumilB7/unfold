"""Canonical patch-grid facts -> display strings.

The renderer formats only the normalized ``grid`` object the parser emits.  It
does not synthesize a grid from image/patch scalar fields; establishing that
those operands form a patchification mechanism belongs to U9 source evidence.
"""
from __future__ import annotations

from .utils import _fmt_int


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


__all__ = ["grid_title", "grid_subtitle", "grid_card_phrase"]
