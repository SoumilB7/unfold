"""The canonical op-graph: one Region resolved once, projected to render + JSON.

These guard the IR-level unification — that a gated/dense/custom FFN and every
attention family are described in exactly one place (``opgraph.ffn_region`` /
``opgraph.attention_region``) and that both the HTML renderer and the JSON
exporter project from that same region rather than re-deriving it.
"""
from model_unfolder.expanded.ffn import _region_to_json
from model_unfolder.expanded.region import region_to_json
from model_unfolder.opgraph import (
    attention_region,
    ffn_region,
    mla_kv_region,
    mla_query_region,
)
from model_unfolder.renderers.html.op_render import region_to_graph
from model_unfolder.renderers.html.graph import wiring_problems

GQA = {"kind": "gqa", "num_heads": 32, "num_kv_heads": 8, "head_dim": 128}


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


# ---------------------------------------------------------------------------
# attention regions
# ---------------------------------------------------------------------------


def test_gqa_attention_region_is_a_multi_merge_dag():
    r = attention_region(GQA, 4096)
    assert r.template == "gqa" and r.resolved
    # two merge points: Q,K meet at the scores; softmax,V meet at apply-V.
    assert set(r.merges()) == {"scaled_scores", "attn_apply_v"}
    # op ids ARE the inspect-card ids — one identity for structure and clicks.
    assert {"q_proj", "k_proj", "v_proj", "scaled_scores", "attn_softmax",
            "attn_apply_v", "concat_heads", "o_proj"} <= {o.id for o in r.ops}


def test_attention_lanes_and_spine_are_derived_from_edges():
    g = region_to_graph(attention_region(GQA, 4096), clickable=True)
    assert g.flow[:2] == ["hidden", "scaled_scores"]          # spine jumps to the join
    lanes = g.parallels[0].norm_lanes()
    # Q and K pass through RoPE before the scores (apply_rotary_pos_emb); V does not.
    assert [lane.ids for lane in lanes] == [["q_proj", "q_rope"], ["k_proj", "k_rope"], ["v_proj"]]
    assert lanes[2].dst == ["attn_apply_v"]                   # V merges above the join


def test_non_rope_family_omits_the_rope_step():
    # ALiBi / learned-absolute families must NOT draw a fabricated RoPE node.
    g = region_to_graph(attention_region(dict(GQA, rope=False), 4096), clickable=True)
    assert [lane.ids for lane in g.parallels[0].norm_lanes()] == [["q_proj"], ["k_proj"], ["v_proj"]]


def test_alibi_bias_is_a_real_two_input_score_add():
    attn = dict(
        GQA, rope=False, position_kind="alibi",
        position_application="attention_bias",
    )
    region = attention_region(attn, 4096)
    assert {"alibi_offsets", "alibi_bias", "score_bias_add"} <= {
        op.id for op in region.ops
    }
    assert set(region.inputs_of("score_bias_add")) == {"scaled_scores", "alibi_bias"}
    graph = region_to_graph(region, clickable=True)
    assert wiring_problems(graph) == []
    from model_unfolder.renderers.html.graph_engine import _lane_draw_order
    parallel = graph.parallels[0]
    ordered = _lane_draw_order(parallel.norm_lanes(), parallel.dst)
    alibi_lane = next(lane for lane in ordered if lane.ids == ["alibi_bias"])
    v_lane = next(lane for lane in ordered if lane.ids == ["v_proj"])
    assert ordered.index(v_lane) == 0
    assert ordered.index(alibi_lane) == len(ordered) - 1


def test_cross_attention_kv_lanes_take_a_side_source():
    r = attention_region(dict(GQA, cross_attention=True), 4096)
    g = region_to_graph(r, clickable=True)
    lanes = g.parallels[0].norm_lanes()
    srcs = {tuple(lane.ids): lane.src for lane in lanes}
    assert srcs[("q_proj",)] is None                          # Q taps decoder hidden
    assert srcs[("k_proj",)] == srcs[("v_proj",)] == "cross_attention_states"


def test_mla_indexer_is_additive_and_keeps_the_v_lane_off_the_spine():
    """DSA must REUSE the MLA region and only ADD an indexer (gated on index_n_heads),
    never reimplement it; and adding that 3rd lane must not collapse the V→⊙ elbow.
    Regression: the indexer pushed KV to the centre column, so its tap to ⊙ ran
    straight up the spine and vanished (a pixel break the dangling flag can't see)."""
    from model_unfolder.renderers.html.graph_engine import _lane_draw_order
    from model_unfolder.renderers.html.op_render import region_to_graph

    mla = {"kind": "mla", "num_heads": 128, "head_dim": 192, "q_lora_rank": 1536,
           "kv_lora_rank": 512, "rope_dim": 64}
    dsa = {**mla, "index_n_heads": 64, "index_head_dim": 128, "index_topk": 2048}

    # Additive: V3 and V3.2 share the SAME query/kv/V structure; DSA only adds the indexer.
    base_ids = {o.id for o in attention_region(mla, 7168).ops}
    dsa_ids = {o.id for o in attention_region(dsa, 7168).ops}
    assert dsa_ids - base_ids == {"mla_indexer"}                  # nothing else changed
    assert "mla_indexer" not in base_ids                          # V3/Kimi/GLM untouched

    # The KV lane still feeds BOTH the scores and the V join, in both models.
    for attn in (mla, dsa):
        par = region_to_graph(attention_region(attn, 7168), clickable=True).parallels[0]
        kv = next(ln for ln in par.norm_lanes() if ln.ids == ["mla_kv_path"])
        assert "attn_apply_v" in (kv.dst or [])                   # V→⊙ edge present
        ordered = _lane_draw_order(par.norm_lanes(), par.dst)
        n = len(ordered)
        # the V-tapping lane must be on an OUTER column (never the centre), so its
        # elbow to ⊙ renders instead of collapsing onto the spine.
        assert ordered.index(kv) in (0, n - 1), f"V lane centred among {n} lanes"


def test_mla_region_nests_the_two_path_subgraphs():
    attn = {"kind": "mla", "num_heads": 128, "head_dim": 192,
            "q_lora_rank": 1536, "kv_lora_rank": 512, "rope_dim": 64}
    r = attention_region(attn, 7168)
    kinds = {o.id: o.kind for o in r.ops}
    assert kinds["mla_query_path"] == kinds["mla_kv_path"] == "subgraph"
    # the drill regions carry the same card-id namespace
    assert {"mla_q", "mla_q_nope", "mla_q_rope", "mla_q_concat"} <= {
        o.id for o in mla_query_region(attn, 7168).ops}
    assert {"mla_kv_down", "mla_cache", "mla_kv_up", "mla_k_merge", "mla_v"} <= {
        o.id for o in mla_kv_region(attn, 7168).ops}


def test_concat_is_a_merge_glyph_while_head_merge_is_a_reshape_box():
    """A TRUE two-lane merge (MLA NoPE+RoPE rejoin) is a ‖ connector glyph; merging
    per-head outputs back to the model dim is a single-stream RESHAPE → a box. So a
    ‖ always means 'two named lanes joined here', never a relabelled 1-input op."""
    from model_unfolder.renderers.html.graph import KIND
    attn = {"kind": "mla", "num_heads": 128, "head_dim": 192,
            "q_lora_rank": 1536, "kv_lora_rank": 512, "rope_dim": 64}

    # SDPA spine: concat-of-heads is a reshape, NOT a concat.
    ch = next(o for o in attention_region(GQA, 4096).ops if o.id == "concat_heads")
    assert ch.kind == "reshape"

    # MLA query/KV: the NoPE/RoPE rejoin is a real concat (two lanes meeting).
    qcat = next(o for o in mla_query_region(attn, 7168).ops if o.id == "mla_q_concat")
    kcat = next(o for o in mla_kv_region(attn, 7168).ops if o.id == "mla_k_merge")
    assert qcat.kind == "concat" and kcat.kind == "concat"

    # Glyphs: concat → the ‖ circle connector; reshape → a plain box.
    assert KIND["concat"].shape == "circle" and KIND["concat"].sym == "‖"
    assert KIND["reshape"].shape == "rect"

    # And the merge really registers two inbound lanes (so it is not dangling).
    g = region_to_graph(mla_query_region(attn, 7168), clickable=True)
    from model_unfolder.renderers.html.graph import wiring_problems
    assert wiring_problems(g) == []


def test_block_width_grows_to_fit_a_long_label():
    """A long label must not overflow its box: width() grows to fit, the
    horizontal mirror of height()'s line-fit (the bug that clipped the router's
    'Linear → 256 scores · sigmoid'). A floor only — explicit w wins, glyph
    default is the minimum, and a sub-line is counted too."""
    from model_unfolder.renderers.html.graph import Node
    short = Node("a", "linear", "Gate")
    longer = Node("b", "linear", "Linear → 256 scores · sigmoid")
    assert longer.width() > short.width()
    assert longer.width() >= Node("z", "linear").glyph().w        # never below the floor
    assert Node("c", "linear", "x", w=999).width() == 999         # explicit w wins
    # a long sub-line widens the box even when the heading is short
    assert Node("d", "select", "top-8", sub="group-limited: keep 4 of 8").width() \
        > Node("e", "select", "top-8").width()


def test_a_one_input_concat_is_flagged_dangling():
    """Now that single-stream regroups are ``reshape`` boxes, a ‖ with one input
    is a genuine wiring bug — the dangling flag (Dable) must catch it."""
    from model_unfolder.renderers.html.graph import Graph, Node, wiring_problems
    bad = Graph(nodes=[Node("in", "port", "x", static=True),
                       Node("c", "concat", "Concat")],
                flow=["in", "c"])
    probs = wiring_problems(bad)
    assert len(probs) == 1 and "‖" in probs[0]


def test_mla_kv_path_v_exits_as_a_labelled_output_lane():
    attn = {"kind": "mla", "kv_lora_rank": 512, "rope_dim": 64}
    g = region_to_graph(mla_kv_region(attn, 7168), clickable=True)
    # spine runs through compression -> latent cache -> expansion -> K concat
    assert g.flow[:5] == ["hidden", "mla_kv_down", "mla_cache", "mla_kv_up", "mla_k_merge"]
    lanes = g.parallels[0].norm_lanes()
    v = next(lane for lane in lanes if lane.ids == ["mla_v"])
    assert v.dst == [] and v.out_label == "V"
    rope = next(lane for lane in lanes if lane.ids == ["mla_k_rope", "mla_k_rope_apply"])
    assert rope.src == "mla_kv_down"                          # taps before the cache


def test_ssm_region_is_an_honest_chain_not_a_fabricated_qkv():
    r = attention_region({"kind": "ssm", "head_dim": 16}, 768)
    assert [o.id for o in r.ops] == ["hidden", "ssm_in_proj", "ssm_conv",
                                     "ssm_scan", "ssm_gate", "ssm_out_proj"]
    assert r.merges() == []


def test_gated_delta_region_preserves_conv_gates_recurrence_and_gated_norm():
    r = attention_region({
        "kind": "gated_delta", "num_heads": 32, "num_kv_heads": 16,
        "head_dim": 128, "v_head_dim": 128, "conv_kernel_size": 4,
    }, 4096)
    ids = {o.id for o in r.ops}
    assert {
        "delta_qkv_proj", "delta_z_proj", "delta_beta_proj", "delta_decay_proj",
        "delta_conv", "delta_qkv_split", "delta_beta", "delta_decay",
        "delta_rule", "delta_gated_norm", "delta_out_proj",
    } <= ids
    assert r.template == "gated_delta"
    assert set(r.inputs_of("delta_rule")) == {"delta_qkv_split", "delta_beta", "delta_decay"}
    assert set(r.inputs_of("delta_gated_norm")) == {"delta_rule", "delta_z_proj"}
    graph = region_to_graph(r, clickable=True)
    assert not wiring_problems(graph)


def test_sdpa_output_gate_splits_q_and_gates_before_output_projection():
    r = attention_region({**GQA, "output_gate": "sigmoid"}, 4096)
    assert set(r.inputs_of("attn_output_mul")) == {"concat_heads", "attn_output_gate"}
    assert r.inputs_of("o_proj") == ["attn_output_mul"]
    assert r.inputs_of("q_gate_split") == ["q_proj"]
    assert not wiring_problems(region_to_graph(r, clickable=True))


def test_unknown_attention_kind_is_one_honest_opaque_node():
    r = attention_region({"kind": "brand_new_mixer", "class_name": "MyMixer"}, 1024)
    assert r.template == "opaque" and r.resolved is False
    assert [o.kind for o in r.ops] == ["opaque"]


def test_attention_render_and_json_project_from_the_same_region():
    """The whole point, again: render and JSON consume ONE region."""
    region = attention_region(GQA, 4096)
    rename = {"scaled_scores": "scores", "attn_softmax": "softmax",
              "attn_apply_v": "context"}
    json_nodes = {n["id"]: n for n in region_to_json(region, rename=rename)["nodes"]}
    graph_ids = {n.id for n in region_to_graph(region, clickable=True).nodes}
    # same structure, two namings: the schema's public ids map 1:1 onto the
    # region/card ids the renderer draws.
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(json_nodes) & graph_ids
    assert json_nodes["scores"]["operation"] == "scaled_dot_product"
    assert json_nodes["scores"]["formula"] == "QK^T/sqrt(dim)"
    assert {"scaled_scores", "attn_softmax", "attn_apply_v"} <= graph_ids
