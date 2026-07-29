"""Layer grouping by structural signature.

Two layers belong to the same group iff their attention spec, FFN spec,
norm choice, and block tree all match.  The largest group is marked
``dominant`` — non-dominant groups are emitted in the same shape and can
be diffed against the dominant by callers.
"""
from __future__ import annotations

from ..ir import layer_signature


def signature(layer: dict) -> tuple:
    """Project the IR's one canonical layer grouping contract."""
    return layer_signature(layer)


def group_layers(layers: list[dict]) -> list[dict]:
    """Bucket layers by signature; mark dominant; assign short names + ids."""
    by_sig: dict[tuple, dict] = {}
    order: list[tuple] = []
    for layer in layers:
        sig = signature(layer)
        if sig not in by_sig:
            by_sig[sig] = {"signature": sig, "representative": layer, "indices": []}
            order.append(sig)
        by_sig[sig]["indices"].append(layer.get("index"))

    groups = [by_sig[s] for s in order]
    if not groups:
        return groups

    dominant_sig = max(order, key=lambda s: len(by_sig[s]["indices"]))
    for i, group in enumerate(groups):
        group["dominant"] = group["signature"] == dominant_sig
        group["id"]       = f"layer_group_{i}"
        group["name"]     = _group_name(group, groups)
    return groups


def _group_name(group: dict, groups: list[dict]) -> str:
    """Short label distinguishing this group from others (or 'main' when alone)."""
    rep = group["representative"]
    attn = rep.get("attention") or {}
    ffn  = rep.get("ffn") or {}
    masks = {(g["representative"].get("attention") or {}).get("mask") for g in groups}
    if len(masks) > 1:
        return str(attn.get("mask") or "default")
    ffn_kinds = {(g["representative"].get("ffn") or {}).get("kind") for g in groups}
    if len(ffn_kinds) > 1:
        return str(ffn.get("kind") or "default")
    return "main"


def group_id_for_layer(layer: dict, groups: list[dict]) -> str:
    sig = signature(layer)
    for g in groups:
        if g["signature"] == sig:
            return g["id"]
    return "unknown"
