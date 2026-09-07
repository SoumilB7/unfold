from unittest.mock import patch
import json
from model_unfolder.renderers.html.block_views import feed_forward
from model_unfolder.renderers.html.block_views.registry import render_block_detail
from model_unfolder.renderers.html.render_context import RenderContext, activate_render_context

block = {"id": "exact_ffn", "view": "runtime_ffn", "kind": "ffn",
         "source_fact_keys": ["root.denoiser.ffn_mechanisms", "unrelated.fact"],
         "detail": {"ffn": {"kind": "dense", "activation": "gelu", "gated": True,
                    "projection_mode": "fused_gate_up", "intermediate_size": 8}}, "children": []}
real = feed_forward.build_ffn_view

def discarded_output(*args, **kwargs):
    real(*args, **kwargs)  # legitimate event, but its SVG does not reach the product
    return "<svg></svg>"

context = RenderContext()
with patch.object(feed_forward, "build_ffn_view", discarded_output), activate_render_context(context):
    svg = render_block_detail({"name": "fixture", "hidden_size": 4}, {}, "review", block)
rows = [{"view": event.view, "nodes": sorted(event.node_ids),
         "facts": sorted(event.facts_projected), "block_path": event.block_path}
        for event in context.events]
print(json.dumps({"returned_svg": svg, "events": rows,
                  "false_ffn_receipt": any(event.view == "runtime_ffn_fact" for event in context.events)},
                 indent=2, sort_keys=True))
