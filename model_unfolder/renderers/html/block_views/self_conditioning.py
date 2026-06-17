"""Detail view for DiffusionGemma's self-conditioning block.

Self-conditioning gives the decoder a low-noise prior from its OWN previous
prediction.  The previous denoising step's logits are turned into soft
embeddings; this module norms that signal, runs it through a gated SwiGLU MLP,
**adds it to the canvas input embeddings**, then post-norms — the sum is what the
decoder actually sees this step.  (HF: ``DiffusionGemmaSelfConditioning.forward``.)

It is a branch-and-merge, not a chain: the signal path (norm → gated MLP) joins
the canvas embeddings at one ``⊕`` and a single post-norm follows.  Rendering it
through the graph engine — a ``Parallel`` for the SwiGLU and a ``SideInput`` for
the canvas — keeps that one merge / one norm honest (the declared-ops chain
duplicated them).
"""
from __future__ import annotations

from ..graph import Graph, Node, Parallel, SideInput
from ..graph_engine import render_graph


def build_self_conditioning_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    nodes = [
        Node("sc_signal", "port", ["prev-step soft", "embeddings"], static=True),
        Node("sc_pre_norm", "norm", "pre_norm (RMSNorm)"),
        Node("sc_gate", "linear", "gate_proj"),
        Node("sc_act", "activation", "GELU"),
        Node("sc_up", "linear", "up_proj"),
        Node("sc_gate_up", "gate_mul", static=True),       # SwiGLU ⊗ — connector glyph
        Node("sc_down", "linear", "down_proj"),
        Node("sc_add", "residual_add", static=True),       # ⊕ add canvas — connector glyph
        Node("sc_post_norm", "norm", ["post_norm", "RMSNorm · no scale"]),
        Node("sc_out", "port", static=True),
        # The canvas embeddings enter the ⊕ laterally (the thing being enriched).
        Node("sc_canvas", "embedding", ["Canvas embeddings", "(inputs_embeds)"]),
    ]
    graph = Graph(
        nodes=nodes,
        flow=["sc_signal", "sc_pre_norm", "sc_gate_up", "sc_down", "sc_add", "sc_post_norm", "sc_out"],
        parallels=[Parallel("sc_pre_norm", "sc_gate_up", [["sc_gate", "sc_act"], ["sc_up"]])],
        side_inputs=[SideInput("sc_canvas", "sc_add", side="right")],
        note="prev-step prediction → gated MLP → ⊕ into the canvas",
    )
    return render_graph(
        graph, info, mount_id, "self_conditioning",
        f"{ir.get('name', 'model')} self-conditioning", min_width=640,
    )
