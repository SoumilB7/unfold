"""Source-proven prefix-concatenation modality fusion view."""
from __future__ import annotations

from ...graph import Graph, Node, SideInput
from ...graph_engine import render_graph


def build_prefix_fusion_view(ir: dict, info: dict, mount_id: str) -> str:
    graph = Graph(
        nodes=[
            Node("embed", "embedding", "Text embeddings"),
            Node("vision_path", "embedding", "Visual tokens"),
            Node("prefix_concat", "concat"),
            Node("stack_input", "output", "Decoder input"),
        ],
        flow=["embed", "prefix_concat", "stack_input"],
        side_inputs=[SideInput("vision_path", "prefix_concat", side="right")],
    )
    return render_graph(
        graph, info, mount_id, "prefix-modality-fusion",
        f"{ir.get('name', 'model')} prefix modality fusion",
    )


__all__ = ["build_prefix_fusion_view"]
