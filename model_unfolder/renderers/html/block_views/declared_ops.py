"""The declared-ops view — ANY card-declared op chain, rendered by the ONE engine.

A card that isn't one of the named templates (attention / FFN / tower) never
needs a bespoke view or hand-written prose: it declares its internals in the
op alphabet (``view: "ops"`` + ``detail.ops``) and this view projects them
through the same ``region_to_graph`` / ``render_graph`` pipeline as every
canonical template.  MLP projectors, patch mergers, conv stems, pixel-shuffle
reductions — all compositions of existing ops, never new view code.  Leaf view.
"""
from __future__ import annotations

from ....evidence.receipts import receipts_from_projects
from ....opgraph import ops_region
from ..graph import Group
from ..graph_engine import render_graph
from ..op_render import region_to_graph


def build_declared_ops_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    declared = (block.get("detail") or {}).get("ops") or []
    rid = block.get("id") or "ops"
    title = block.get("title") or block.get("label") or "declared ops"
    region = ops_region(declared, rid=rid, label=title)
    # Op nodes are drill targets whenever cards exist for them — and they
    # always do: the lookup derives per-op cards from this same region.
    graph = region_to_graph(region, clickable=bool(block.get("children")))
    for index, op in enumerate(declared):
        if "repeat" not in op or op.get("repeat") == 1:
            continue
        repeat = op.get("repeat")
        graph.groups.append(Group(
            [op.get("id") or f"{rid}_op{index}"],
            repeat=repeat if isinstance(repeat, int) and repeat > 1 else None,
        ))
    # U2 receipts: a block may declare the exact fact targets it draws here
    # (``projects``).  This generic op view emits their typed receipts onto the
    # drill's own render event — no per-mechanism code, so scheduler and any
    # future op drill reuse the same channel.
    view_key = f"ops_{rid}"
    # U2-R5: THIS function is the actual projector — it draws the declared ops
    # (the projector width among them) on the CARD surface at the block's own
    # structural node.  The receipt is emitted here, at the drawing site, with
    # the canonical surface and this projector's symbol; the render context
    # stamps its own token when the event is recorded.
    return render_graph(
        graph,
        info, mount_id, view_key,
        f"{ir.get('name', 'model')} {title}", min_width=640,
        receipts=receipts_from_projects(
            block.get("projects"), surface="card", structural_target=rid,
            projector_symbol="renderers.html.block_views.declared_ops.build_declared_ops_view",
            node_ids=(rid,), projection_kind="op",
            fact_rows=(ir.get("extras") or {}).get("fact_provenance") or {}),
    )
