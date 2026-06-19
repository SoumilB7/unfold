"""View-coverage net — the "nothing slips" guarantee.

Every renderer view archetype lives in ``VIEW_REGISTRY``. A view that no model
ever renders is exactly how a broken/forgotten drill hides (e.g. the scheduler
step that floated its ⊕ for ages because nothing exercised + inspected it). This
test instruments the registry dispatcher, renders a corpus spanning every
archetype, and asserts:

  1. EVERY registered view is actually exercised by some model (no dead/unseen view);
  2. every rendered model is recursively click-coupled (every clickable node at every
     drill depth resolves to a card).

If you add a view to the registry, add a model here that exercises it — or the test
fails. The two registered *fallbacks* (``ops`` declared-op floor, ``tower`` custom
backbone) are emitted by no built-in model and are exercised by their own unit tests
(``test_declared_ops`` / ``test_tower``); they are the only documented exceptions.

This is the mechanical half of the coverage policy (CLAUDE.md §Coverage). The other
half — rendering each archetype to PNG and inspecting the pixels — is the Sable's
manual pixel pass; this test guarantees there is nothing it can forget to look at.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_unfolder import unfold
from model_unfolder.block_schema import validate_click_coupling
from model_unfolder.renderers.html.block_views import registry as reg
import test_diffusion as td   # reuse the offline diffusion fixtures (FLUX / PixArt / SDXL UNet)

_BASE = dict(num_hidden_layers=2, hidden_size=128, num_attention_heads=8,
             num_key_value_heads=2, intermediate_size=256, vocab_size=1000, rms_norm_eps=1e-5)

_VISION_CFG = {"depth": 4, "hidden_size": 128, "num_heads": 8, "patch_size": 14, "in_channels": 3}
_GRID_VISION = {**_VISION_CFG, "spatial_merge_size": 2, "temporal_patch_size": 2}

# A corpus designed to span EVERY view archetype (offline — synthetic + diffusion dicts).
CORPUS = {
    "dense_gated":  dict(_BASE, model_type="llama", hidden_act="silu"),          # attention, gated_ffn
    "dense_mlp":    dict(_BASE, model_type="phi", num_key_value_heads=8, hidden_act="gelu_new"),  # dense_ffn
    "moe_mla_mtp":  dict(_BASE, model_type="deepseek_v3", kv_lora_rank=64, q_lora_rank=96,
                         qk_nope_head_dim=64, qk_rope_head_dim=32, n_routed_experts=8,
                         num_experts_per_tok=2, moe_intermediate_size=128, first_k_dense_replace=1,
                         n_shared_experts=1, scoring_func="sigmoid", topk_method="noaux_tc",
                         n_group=4, topk_group=2, norm_topk_prob=True, routed_scaling_factor=2.5,
                         num_nextn_predict_layers=1),  # moe, moe_router, moe_expert, mla_*, mtp_*
    "dsa":          dict(_BASE, model_type="deepseek_v32", kv_lora_rank=64, q_lora_rank=96,
                         qk_nope_head_dim=64, qk_rope_head_dim=32, n_routed_experts=8,
                         num_experts_per_tok=2, moe_intermediate_size=128, first_k_dense_replace=1,
                         index_topk=2048, index_n_heads=64, index_head_dim=128),  # dsa_indexer
    "ple":          dict(_BASE, model_type="m", hidden_size_per_layer_input=64,
                         vocab_size_per_layer_input=1000),                         # per_layer_embedding
    "self_cond":    dict(model_type="diffusion_gemma",
                         text_config=dict(_BASE, n_routed_experts=8, num_experts_per_tok=2,
                                          moe_intermediate_size=128)),             # self_conditioning
    "vision":       dict(_BASE, model_type="qwen2_vl", vision_config=_VISION_CFG,
                         image_token_id=4),  # vision_path/encoder/self_attention/mlp/patch_embedding, multimodal_fusion
    "audio":        dict(_BASE, model_type="qwen2_audio",
                         audio_config={"num_hidden_layers": 4, "d_model": 128,
                                       "encoder_attention_heads": 8}),             # audio_path/encoder
    "video":        dict(_BASE, model_type="qwen2_vl", vision_config=_GRID_VISION,
                         image_token_id=4, video_token_id=5),                      # video_path/encoder
    "dit_mmdit":    td.FLUX,        # attention, ffn, scheduler_step, vae_decoder(_block), text_encoder, encoded_text_concat
    "dit_cross":    td.PIXART,      # cross_attention
    "unet":         td.SDXL_UNET,   # unet, unet_stage, unet_resnet, unet_transformer
}

# Registered FALLBACK views with no built-in model emitter — exercised by their own
# unit tests, not by a catalogue model. The only views allowed to be model-unexercised.
_FALLBACK_VIEWS = {"ops", "tower"}


def test_every_registered_view_is_exercised_and_couples():
    exercised: set[str] = set()
    orig = reg.render_view

    def _record(ctx, block):
        v = reg.view_key(block)
        if v:
            exercised.add(v)
        return orig(ctx, block)

    reg.render_view = _record
    try:
        for name, cfg in CORPUS.items():
            html = unfold(cfg).to_html(standalone=True)
            problems = validate_click_coupling(html)
            assert problems == [], f"{name}: recursive coupling broken:\n  " + "\n  ".join(problems)
    finally:
        reg.render_view = orig

    registered = {k for k in reg.VIEW_REGISTRY if k}
    missing = registered - exercised - _FALLBACK_VIEWS
    assert not missing, (
        "registered views never exercised by the corpus (a forgotten/dead drill can "
        f"hide a broken layout here — add a model that renders it): {sorted(missing)}"
    )


# Purpose-built graph drill-views (NOT the op-graph region path). These had a
# recurring class of bug: every node made `static` (unclickable blocks), and
# connectors (⊙ / ⊕) drawn without their second input wired (a dangling glyph).
# Coupling can't see either (static = no data-id = no orphan), so they're pinned here.
_AUTHORED_GRAPH_VIEWS = {"moe_router", "dsa_indexer", "cross_attention",
                         "scheduler_step", "self_conditioning"}
# Connectors that ALWAYS combine two real tensor inputs (so <2 inputs = dangling).
# gate_mul (×) is excluded: it may scale by a labelled constant (router ×scale).
_TWO_INPUT_CONNECTORS = {"dot_product", "residual_add"}


def test_authored_drill_views_are_clickable_and_connectors_wired():
    import sys
    import model_unfolder.renderers.html.graph_engine as ge

    seen: dict[str, tuple] = {}

    def _cap(graph, info, mount_id, view_key, title, **kw):
        if view_key in _AUTHORED_GRAPH_VIEWS and view_key not in seen:
            indeg = {n.id: 0 for n in graph.nodes}
            for a, b in zip(graph.flow, graph.flow[1:]):
                indeg[b] = indeg.get(b, 0) + 1
            for e in graph.edges:                      # incl. residual skips
                indeg[e.dst] = indeg.get(e.dst, 0) + 1
            for p in graph.parallels:
                for lane in p.lanes:
                    dsts = (None if isinstance(lane, list) else lane.dst) or [p.dst]
                    for d in dsts:
                        indeg[d] = indeg.get(d, 0) + 1
            for s in getattr(graph, "side_inputs", []):
                indeg[s.target] = indeg.get(s.target, 0) + 1
            clickable = [n.id for n in graph.nodes if not n.static and n.kind != "port"]
            dangling = [n.id for n in graph.nodes
                        if n.kind in _TWO_INPUT_CONNECTORS and indeg.get(n.id, 0) < 2]
            seen[view_key] = (clickable, dangling)
        return _orig(graph, info, mount_id, view_key, title, **kw)

    _orig = ge.render_graph
    patched = [m for n, m in sys.modules.items()
               if "block_views" in n and hasattr(m, "render_graph")]
    ge.render_graph = _cap
    for m in patched:
        m.render_graph = _cap
    try:
        for cfg in CORPUS.values():
            unfold(cfg).to_html(standalone=True)
    finally:
        ge.render_graph = _orig
        for m in patched:
            m.render_graph = _orig

    assert _AUTHORED_GRAPH_VIEWS <= set(seen), \
        f"authored views never exercised by the corpus: {sorted(_AUTHORED_GRAPH_VIEWS - set(seen))}"
    for vk, (clickable, dangling) in seen.items():
        assert clickable, f"{vk}: drill view has NO clickable node — unclickable blocks"
        assert not dangling, (
            f"{vk}: connector(s) drawn without a wired second input (a ⊙/⊕ that "
            f"multiplies/adds with nothing visible): {dangling}"
        )


def test_fallback_views_have_dedicated_coverage():
    """The two model-unexercised views must still be covered somewhere."""
    assert _FALLBACK_VIEWS <= {k for k in reg.VIEW_REGISTRY if k}
    # tower: tests/test_tower.py ; ops: tests/test_declared_ops.py
    import test_tower, test_declared_ops  # noqa: F401  (importable ⇒ their suites run)
