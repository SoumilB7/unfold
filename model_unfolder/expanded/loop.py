"""Project the declared diffusion sampling loop onto the JSON schema.

The loop's structure is authored once, as data, in
``adapters/diffusor/blocks.py`` (``diffusion_loop_blocks`` for the nodes,
``diffusion_loop_edges`` for the wiring, ``diffusion_loop_region`` for the
recurrence).  The HTML view draws from it; this module is the *other* projection
of the same declaration — so the JSON can never drift from the diagram.

Emits the five structural parts an architecture is made of:

* **blocks**  → ``nodes`` (id / role / kind / stage, and whether it drills down)
* **arrows**  → ``edges`` (from/to + ports + label + ``when`` + ``back_edge``)
* **repeating region** → ``repeating_region`` (members + loop-carried back-edge)
* **connectors** (fan-in)  → derived: a node with ≥2 incoming edges
* **splitters** (fan-out)   → derived: a node with ≥2 outgoing edges

``route``/``gap``/``lane_index``/``label_at`` on an edge are SVG-only routing
hints; they are dropped here — this view keeps topology, not pixels.
"""
from __future__ import annotations

from typing import Any

from .utils import drop_none


def build_sampling_loop(extras: dict) -> dict | None:
    """Return the ``sampling_loop`` JSON object, or ``None`` for non-diffusion."""
    render = extras.get("render") or {}
    blocks = render.get("loop_blocks") or []
    if not blocks:
        return None

    nodes = [
        drop_none({
            "id":         b.get("id"),
            "role":       b.get("role"),
            "kind":       b.get("kind"),
            "stage":      b.get("diffusion_stage"),
            "expandable": bool(b.get("children") or b.get("view")) or None,
        })
        for b in blocks if b.get("id")
    ]

    edges_raw = render.get("loop_edges") or []
    edges = [
        drop_none({
            "from":      e.get("from"),
            "to":        e.get("to"),
            "from_port": e.get("from_port"),
            "to_port":   e.get("to_port"),
            "label":     e.get("label"),
            "when":      e.get("when"),
            "back_edge": e.get("back_edge") or None,
        })
        for e in edges_raw
    ]

    # Connectors (fan-in) and splitters (fan-out) are derived from multiplicity,
    # never stored twice — exactly how a structural conformer would find them.
    incoming: dict[str, list[str]] = {}
    outgoing: dict[str, list[str]] = {}
    for e in edges_raw:
        src, dst = e.get("from"), e.get("to")
        if src is None or dst is None:
            continue
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(dst, []).append(src)
    connectors = [{"at": n, "inputs":  v} for n, v in incoming.items() if len(v) >= 2]
    splitters  = [{"at": n, "outputs": v} for n, v in outgoing.items() if len(v) >= 2]

    out: dict[str, Any] = {
        "nodes":            nodes,
        "edges":            edges,
        "repeating_region": render.get("loop_region") or None,
        "connectors":       connectors or None,
        "splitters":        splitters or None,
    }
    return drop_none(out)
