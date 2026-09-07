from unittest.mock import patch
from copy import deepcopy
import json
from model_unfolder.renderers.html.block_views import feed_forward
from model_unfolder.renderers.html.block_views.registry import render_block_detail
from model_unfolder.renderers.html.render_context import RenderContext, activate_render_context, current_render_context
FFN = "root.denoiser.ffn_mechanisms"
card = {"id": "exact_ffn", "view": "runtime_ffn", "kind": "ffn",
        "source_fact_keys": [FFN, "unrelated.fact"],
        "detail": {"ffn": {"kind": "dense", "activation": "gelu", "gated": True,
                   "projection_mode": "fused_gate_up", "intermediate_size": 8}},
        "children": [{"id": node, "role": "operation"} for node in
                     ("gate_up_proj", "gate_up_split", "activation", "multiply", "down_proj")]}
original = feed_forward.build_ffn_view
results = []
for case in ("ordinary", "discard", "partial", "foreign_block", "uncited"):
    block = deepcopy(card)
    if case == "uncited": block["source_fact_keys"] = ["unrelated.fact"]
    def altered(*args, **kwargs):
        if case == "foreign_block":
            with current_render_context().block({"id": "other_ffn"}):
                return original(*args, **kwargs)
        svg = original(*args, **kwargs)
        if case == "discard": return "<svg></svg>"
        if case == "partial": return svg.replace('data-id="activation"', 'data-id="removed_activation"')
        return svg
    context = RenderContext()
    with patch.object(feed_forward, "build_ffn_view", altered), activate_render_context(context):
        svg = render_block_detail({"name": "fixture", "hidden_size": 4}, {}, "review", block)
    receipts = [e for e in context.events if e.view == "runtime_ffn_fact"]
    assert bool(receipts) == (case == "ordinary"), case
    assert all(e.facts_projected == frozenset({FFN}) for e in receipts)
    results.append({"case": case, "pass": True, "actual_svg_length": len(svg),
                    "ffn_events": sum(e.view == "ffn" for e in context.events),
                    "receipts": [{"facts": sorted(e.facts_projected), "nodes": sorted(e.node_ids),
                                  "block_path": e.block_path} for e in receipts]})
print(json.dumps(results, indent=2, sort_keys=True))
