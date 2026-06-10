"""The ONE tower backbone: every transformer tower (text encoder, vision
encoder, MTP block, custom) renders through ``tower.tower_graph`` with the
same block vocabulary as the main model view."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_unfolder.renderers.html.block_views.registry import VIEW_REGISTRY
from model_unfolder.renderers.html.graph_engine import render_graph
from model_unfolder.renderers.html.tower import tower_graph

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
    assert "tower" in VIEW_REGISTRY
    svg = render_graph(tower_graph(SPEC), {}, "t0", "tower-test", "custom tower")
    for marker in ("Custom mixer", "Embed", "× 24 layers", "Custom output"):
        assert marker in svg


def test_all_three_builtin_towers_route_through_the_backbone():
    import inspect
    from model_unfolder.renderers.html.block_views import text_encoder, mtp_head
    from model_unfolder.renderers.html.block_views.modality_views import vision_details

    for mod, fn in ((text_encoder, "build_text_encoder_view"),
                    (mtp_head, "build_mtp_transformer_block_view"),
                    (vision_details, "build_vision_encoder_view")):
        assert "tower_graph(" in inspect.getsource(getattr(mod, fn))
