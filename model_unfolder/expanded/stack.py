"""The decoder stack viewed three ways.

* ``loop``           — one entry per group: "apply this group to these
                       layer indices" (compressed into ``ranges``).  This
                       is the for-loop your reader actually wants.
* ``layer_pattern``  — flat ``sequence`` of group ids per layer index,
                       plus the smallest repeating ``cycle`` when one
                       exists (so the period/repeats are explicit too).
"""
from __future__ import annotations

from typing import Any

from .grouping import group_id_for_layer
from .utils import fmt_indices


def build_stack(layers: list[dict], groups: list[dict]) -> dict[str, Any]:
    sequence = [group_id_for_layer(l, groups) for l in layers]
    return {
        "kind":             "decoder_only",
        "num_layers":       len(layers),
        "num_layer_groups": len(groups),
        "loop":             _loop_view(groups),
        "layer_pattern":    _pattern(sequence),
    }


def _loop_view(groups: list[dict]) -> list[dict[str, Any]]:
    """For each group, the layer indices it applies to."""
    return [
        {
            "applies":    g["id"],
            "name":       g["name"],
            "dominant":   bool(g.get("dominant")),
            "applies_to": fmt_indices(g["indices"]),
        }
        for g in groups
    ]


def _pattern(sequence: list[str]) -> dict[str, Any]:
    cycle = _smallest_cycle(sequence)
    return {
        "sequence": sequence,
        "cycle":    cycle,
        "period":   len(cycle) if cycle else None,
        "repeats":  (len(sequence) // len(cycle)) if cycle else None,
    }


def _smallest_cycle(seq: list[str]) -> list[str] | None:
    n = len(seq)
    if not seq:
        return None
    for p in range(1, n // 2 + 1):
        if n % p == 0 and all(seq[i] == seq[i % p] for i in range(n)):
            return seq[:p]
    return seq if len(set(seq)) == 1 else None
