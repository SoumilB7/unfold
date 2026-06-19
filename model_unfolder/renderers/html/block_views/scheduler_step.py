"""Detail view for one scheduler/sampler step (z_t → z_{t-1}).

The denoiser's prediction (ε̂ / v̂) is scaled by the step size and combined with
the current latent to take one step toward the clean latent.  Built as a
purpose-built graph (one combine ⊕, the prediction on the flow, z_t entering from
the side) — the declared-ops chain mis-laid this out because the combine merges the
primary input (z_t) with a side-scaled input, floating the ⊕ (same failure mode the
self-conditioning view hit).  The named blocks (prediction, scale, current latent) are
clickable with their own cards; the combine ⊕ is a static connector and z_{t-1} a port.
"""
from __future__ import annotations

from ..graph import Graph, Node, SideInput
from ..graph_engine import render_graph


def build_scheduler_step_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    s = ((block or {}).get("detail") or {}).get("scheduler_step") or {}
    sym = s.get("sym", "ε̂")
    what = s.get("what", "noise")
    scale_label = s.get("scale_label", "scale")
    step_label = s.get("step_label", "z_t → z_{t-1}")

    # Tier-1 named blocks are clickable (open their card); the combine ⊕ is a
    # Tier-2 connector and the output is a port — those stay static.
    nodes = [
        Node("sch_pred", "embedding", [f"{sym} {what}", "from denoiser"]),
        Node("sch_scale", "select", scale_label, w=210),
        Node("sch_step", "residual_add"),     # combine z_t with the scaled prediction (clickable)
        Node("sch_out", "port", ["z_{t−1}", "(one step)"], static=True),
        # the current latent enters the combine from the side (the loop-carried value)
        Node("sch_zt", "embedding", ["z_t", "current latent"]),
    ]
    graph = Graph(
        nodes=nodes,
        flow=["sch_pred", "sch_scale", "sch_step", "sch_out"],
        side_inputs=[SideInput("sch_zt", "sch_step", side="left")],
        note=step_label,
    )
    return render_graph(
        graph, info, mount_id, "scheduler_step",
        f"{ir.get('name', 'model')} scheduler step", min_width=520,
    )
