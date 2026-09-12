"""The ONE tower backbone: every transformer tower (text encoder, vision
encoder, MTP block, custom) renders through ``tower.tower_graph`` with the
same block vocabulary as the main model view."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_unfolder.renderers.html.block_views.registry import VIEW_REGISTRY
from model_unfolder.renderers.html.graph_engine import render_graph
from model_unfolder.renderers.html.graph import Graph, Node, Parallel
from model_unfolder.renderers.html.tower import tower_cell, tower_graph

SPEC = {
    "title": "my custom tower",
    "source": {"id": "ct_in", "label": "Custom input", "sub": "anything"},
    "pre": [{"id": "ct_embed", "kind": "embedding", "label": "Embed"}],
    "cell": [
        {"id": "ct_norm", "kind": "norm", "label": "RMSNorm"},
        {"id": "ct_mix", "kind": "attention", "label": "Custom mixer"},
        {"id": "ct_add", "kind": "residual_add", "static": True, "residual_from": "ct_norm"},
    ],
    "repeat": 24,
    "output": {"id": "ct_out", "label": "Custom output"},
}


def test_tower_graph_builds_cell_group_and_residuals_from_the_spec():
    g = tower_graph(SPEC)
    assert g.flow == ["ct_in", "ct_embed", "ct_norm", "ct_mix", "ct_add", "ct_out"]
    assert g.groups[0].members == ["ct_norm", "ct_mix", "ct_add"]
    assert g.groups[0].repeat == 24
    assert [(e.src, e.dst) for e in g.residuals()] == [("ct_norm", "ct_add")]


def test_custom_tower_renders_with_no_view_code():
    """An adapter that emits view:'tower' + detail.tower gets the backbone."""
    from model_unfolder.renderers.html.render_context import current_render_context

    assert "tower" in VIEW_REGISTRY
    svg = render_graph(tower_graph(SPEC), {}, "t0", "tower-test", "custom tower")
    for marker in ("Custom mixer", "Embed", "× 24", "Custom output"):
        assert marker in svg
    assert current_render_context() is None


def test_low_level_svg_id_allocation_never_creates_ambient_render_state():
    """Manual SVG builders may consume a document context, never install one."""
    from model_unfolder.renderers.html.render_context import current_render_context
    from model_unfolder.renderers.html.svg import _ids

    assert current_render_context() is None
    assert _ids("standalone", "view") == (
        "standalone-view-0-arrow",
        "standalone-view-0-shadow",
    )
    assert current_render_context() is None


def test_single_cell_never_draws_repeat_frame_or_pill_even_with_a_label():
    spec = {
        "cell": [{"id": "only", "kind": "attention", "label": "One block"}],
        "repeat": 1,
        "repeat_label": "decoder layer",
    }
    graph = tower_graph(spec)
    assert graph.groups == []
    svg = render_graph(graph, {}, "single", "single-tower", "single tower")
    assert "decoder layer" not in svg and "× 1" not in svg


def test_unknown_safe_ffn_tower_label_gets_truth_readable_geometry():
    """The second honesty line must remain inside its tower node.

    The handwritten SVG font is wider than the generic character estimate;
    pin the explicit geometry at the shared tower primitive so every
    denoiser/vision/audio/refiner caller receives the same correction.
    """
    cell = tower_cell(
        "cell",
        attn_label="Self-attention",
        norm_label="LayerNorm",
        ffn_fact={
            "kind": "dense",
            "gated": True,
            "projection_mode": None,
        },
    )
    ffn = next(block for block in cell if block["id"] == "cell_op_ffn")
    assert ffn["label"] == ["Gated FFN", "storage unresolved"]
    assert ffn["w"] >= 340 and ffn["h"] >= 76
    node = next(
        node for node in tower_graph({"cell": cell, "repeat": 2}).nodes
        if node.id == "cell_op_ffn"
    )
    assert node.width() == ffn["w"] and node.height() == ffn["h"]


def test_parallel_circle_fanin_distributes_arrowheads_around_connector_edge():
    """Several expert lanes must not stack arrowheads into one X-shaped blot."""
    import re

    graph = Graph(
        nodes=[
            Node("src", "router"),
            *[Node(f"expert_{i}", "expert") for i in range(5)],
            Node("sum", "residual_add"),
            Node("out", "port", static=True),
        ],
        flow=["src", "sum", "out"],
        parallels=[Parallel("src", "sum", [[f"expert_{i}"] for i in range(5)])],
    )
    svg = render_graph(graph, {}, "fanin", "fanin", "fanin")
    paths = re.findall(r'<path\s+d="([^"]+)"[^>]*marker-end=', svg)
    endpoints = []
    for path in paths:
        coords = re.findall(r'L\s+(-?[\d.]+)\s+(-?[\d.]+)', path)
        if coords:
            endpoints.append(coords[-1])
    assert len(endpoints) == len(set(endpoints)), "two routes stack arrowheads at one connector point"


def test_unknown_audio_encoder_opens_one_opaque_tower_cell():
    """Depth/width/head numbers do not prove an attention+FFN mechanism."""
    from model_unfolder.renderers.html.block_views.modality_views.audio import encoder_tower_spec

    spec = encoder_tower_spec(
        {"hidden_size": 1024, "num_layers": 12, "num_attention_heads": 8})
    g = tower_graph(spec)
    assert g.groups[0].repeat == 12
    kinds = {n.id: n.kind for n in g.nodes}
    assert kinds["enc_opaque"] == "opaque"
    assert not ({"attention", "ffn", "norm"} & set(kinds.values()))
    assert {"audio_encoder", "video_encoder"} <= set(VIEW_REGISTRY)


def test_video_encoder_reuses_the_canonical_vision_backbone():
    import inspect
    from model_unfolder.renderers.html.block_views.modality_views.video import build_video_encoder_view

    assert "build_vision_encoder_view(" in inspect.getsource(build_video_encoder_view)


def test_all_three_builtin_towers_route_through_the_backbone():
    import inspect
    from model_unfolder.renderers.html.block_views import text_encoder, mtp_head
    from model_unfolder.renderers.html.block_views.modality_views import vision_details

    for mod, fn in ((text_encoder, "build_text_encoder_view"),
                    (mtp_head, "build_mtp_transformer_block_view"),
                    (vision_details, "build_vision_encoder_view")):
        assert "tower_graph(" in inspect.getsource(getattr(mod, fn))
