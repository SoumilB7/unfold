"""Layer grouping by structural signature.

Two layers belong to the same group iff their attention spec, FFN spec,
norm choice, and block tree all match.  The largest group is marked
``dominant`` — non-dominant groups are emitted in the same shape and can
be diffed against the dominant by callers.
"""
from __future__ import annotations

from typing import Any


_SIG_ATTN = (
    "kind", "num_heads", "num_kv_heads", "head_dim",
    "kv_lora_rank", "q_lora_rank", "rope_dim",
    "mask", "window_size", "kv_source_layer",
    "qk_norm", "shared", "no_rope",
)
_SIG_FFN = (
    "kind", "activation", "intermediate_size", "gated",
    "num_experts", "num_experts_per_tok", "num_shared_experts",
    "expert_intermediate_size",
)


def signature(layer: dict) -> tuple:
    """Hashable structural fingerprint of one layer."""
    attn = layer.get("attention") or {}
    ffn  = layer.get("ffn") or {}
    return (
        tuple((k, attn.get(k)) for k in _SIG_ATTN),
        tuple((k, ffn.get(k))  for k in _SIG_FFN),
        layer.get("norm_kind"),
        layer.get("norm_placement"),
        tuple(_block_sig(b) for b in layer.get("blocks") or [] if isinstance(b, dict)),
    )


def _block_sig(block: dict) -> tuple:
    return (
        block.get("id"),
        block.get("role"),
        block.get("kind"),
        tuple(_block_sig(c) for c in block.get("children") or [] if isinstance(c, dict)),
    )


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
