"""The declared-ops view — ANY card-declared op chain, rendered by the ONE engine.

A card that isn't one of the named templates (attention / FFN / tower) never
needs a bespoke view or hand-written prose: it declares its internals in the
op alphabet (``view: "ops"`` + ``detail.ops``) and this view projects them
through the same ``region_to_graph`` / ``render_graph`` pipeline as every
canonical template.  MLP projectors, patch mergers, conv stems, pixel-shuffle
reductions — all compositions of existing ops, never new view code.  Leaf view.
"""
from __future__ import annotations

from ....opgraph import ops_region
from ..graph_engine import render_graph
from ..op_render import region_to_graph


def build_declared_ops_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    declared = (block.get("detail") or {}).get("ops") or []
    rid = block.get("id") or "ops"
    title = block.get("title") or block.get("label") or "declared ops"
    region = ops_region(declared, rid=rid, label=title)
    return render_graph(
        region_to_graph(region), info, mount_id, f"ops_{rid}",
        f"{ir.get('name', 'model')} {title}", min_width=640,
    )
