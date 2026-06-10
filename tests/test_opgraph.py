"""The canonical op-graph: one Region resolved once, projected to render + JSON.

These guard the IR-level unification — that a gated/dense/custom FFN is described
in exactly one place (``opgraph.ffn_region``) and that both the HTML renderer and
the JSON exporter project from that same region rather than re-deriving it.
"""
from model_unfolder.expanded.ffn import _region_to_json
from model_unfolder.opgraph import ffn_region
from model_unfolder.renderers.html.op_render import region_to_graph


def test_gated_ffn_resolves_to_a_gated_region():
    r = ffn_region({"kind": "dense", "gated": True, "activation": "silu",
                    "intermediate_size": 256}, 64)
    assert r.template == "gated_mlp" and r.resolved
    ids = {o.id for o in r.ops}
    assert {"gate_proj", "up_proj", "activation", "multiply", "down_proj"} <= ids
    assert r.merges() == ["multiply"]          # the single branch-merge point


def test_dense_ffn_is_a_plain_chain():
    r = ffn_region({"kind": "dense", "gated": False, "activation": "gelu",
                    "intermediate_size": 256}, 64)
    assert r.template == "dense_mlp"
    assert r.merges() == []                     # no branch — a straight column


def test_custom_ffn_falls_back_to_one_honest_opaque_node():
    """An unrecognised FFN is a single opaque node labelled from the class name —
    flagged unresolved, never a fabricated gate/up/down structure."""
    r = ffn_region({"kind": "some_novel_glu", "class_name": "MyFancyMLP"}, 4096)
    assert r.template == "opaque" and r.resolved is False
    assert [o.kind for o in r.ops] == ["opaque"]
    assert r.ops[0].meta["class_name"] == "MyFancyMLP"


def test_render_and_json_project_from_the_same_region():
    """The whole point: render and JSON consume ONE region (no second authoring)."""
    r = ffn_region({"kind": "dense", "gated": True, "intermediate_size": 256}, 64)
    json_ids = {n["id"] for n in _region_to_json(r)["nodes"]}
    graph_ids = {n.id for n in region_to_graph(r).nodes}
    # every structural op id appears in both projections (the render adds only a
    # presentation-only output bookend on top).
    assert json_ids <= graph_ids
    assert "multiply" in json_ids and "gate_proj" in json_ids


def test_custom_ffn_json_is_one_opaque_node():
    r = ffn_region({"kind": "weird", "class_name": "MyMLP"}, 64)
    nodes = _region_to_json(r)["nodes"]
    assert [n["operation"] for n in nodes] == ["opaque"]
    assert nodes[0]["class_name"] == "MyMLP"
