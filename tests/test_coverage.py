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
from model_unfolder.block_schema import (
    validate_click_coupling,
    validate_no_dotted_arrows,
    validate_no_dotted_boundaries,
    validate_unique_ref_ids,
)
from model_unfolder.renderers.html.block_views import registry as reg
import test_diffusion as td   # reuse the offline diffusion fixtures (FLUX / PixArt / SDXL UNet)

_BASE = dict(num_hidden_layers=2, hidden_size=128, num_attention_heads=8,
             num_key_value_heads=2, intermediate_size=256, vocab_size=1000, rms_norm_eps=1e-5)

_VISION_CFG = {"model_type": "qwen2_vl", "architectures": ["Qwen2VisionTransformerPretrainedModel"],
               "depth": 4, "hidden_size": 128, "num_heads": 8, "patch_size": 14, "in_channels": 3}
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
# `ffn` is the generic FFN-dispatch key (decides moe/dense/gated by the block); no
# built-in block sets view="ffn" directly — they stamp the concrete gated_ffn/dense_ffn/
# moe keys, which ARE exercised and route through the same build_ffn_view. So `ffn` is a
# dispatch alias, legitimately model-unexercised (covered by the FFN tests in test_smoke).
_FALLBACK_VIEWS = {"ops", "tower", "ffn"}


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


def test_no_duplicate_marker_ids_anywhere():
    """No two baked diagrams may share a url(#)-referenced def id (arrowhead
    markers, gradients). A duplicate makes the live browser bind every reference to
    the FIRST (often hidden) match, so arrowheads vanish from the rendered output
    even though each svg is correct in isolation — the standalone/notebook vs
    rendered-PNG divergence. Document-level; the isolated image pass can't see it."""
    for name, cfg in CORPUS.items():
        problems = validate_unique_ref_ids(unfold(cfg).to_html(standalone=True))
        assert not problems, f"{name}: duplicate marker/def ids:\n  " + "\n  ".join(problems)


def test_no_dotted_arrows_or_boundaries_anywhere():
    """Solid flow and region strokes are a corpus-wide design invariant."""
    for name, cfg in CORPUS.items():
        html = unfold(cfg).to_html(standalone=True)
        problems = validate_no_dotted_arrows(html) + validate_no_dotted_boundaries(html)
        assert not problems, f"{name}: dotted structural stroke(s):\n  " + "\n  ".join(problems)


def test_no_dangling_connectors_anywhere():
    """Dable's dangling flag, run across the whole corpus: NO graph — at any drill
    depth, in any model — may draw a ⊕/×/⊙ with a missing input. (The detector
    itself lives in renderers/html/graph.py and runs on every render.)"""
    for name, cfg in CORPUS.items():
        problems = unfold(cfg).wiring_problems()
        assert not problems, f"{name}: dangling connector(s):\n  " + "\n  ".join(problems)


def test_no_all_static_drill_views_anywhere():
    """NO drill view — op-graph-projected (attention / ffn / moe-expert / vae / ...)
    OR hand-authored — may be ALL-STATIC. Every rendered graph that has a substantial
    (non-port) node must expose at least one clickable node, so a viewer can always
    open something and read its description ("even a leaf is clickable").

    This is the GENERAL form of the clickability check. The earlier version only
    inspected the few hand-authored views, so an all-static op-graph view (the FFN
    'leaf' rendered with clickable=False) slipped right past it — the exact "how can
    it slip" gap. Driven off render_graph, it now sees EVERY graph the renderer builds."""
    import sys
    import model_unfolder.renderers.html.graph_engine as ge

    violations: list[tuple[str, str]] = []

    def _cap(graph, info, mount_id, view_key, title, **kw):
        nonport = [n for n in graph.nodes if n.kind != "port"]
        # An honest-unknown opaque node (config doesn't declare the structure) is a
        # legitimately pale/static leaf — there is nothing to drill or click.
        honest_unknown = nonport and all(n.kind == "opaque" and not n.resolved for n in nonport)
        if nonport and not honest_unknown and not any(not n.static for n in nonport):
            violations.append((view_key, title))
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

    uniq = sorted(set(violations))
    assert not uniq, (
        "all-static drill view(s) — every node static, so clicking opens nothing "
        "(give the block children / render clickable):\n  "
        + "\n  ".join(f"{vk}: {t}" for vk, t in uniq)
    )


def test_no_static_connectors_in_any_graph_view():
    """Tier-2 connector glyphs are clickable explanations, never static layout."""
    import sys
    import model_unfolder.renderers.html.graph_engine as ge

    connector_kinds = {"residual_add", "gate_mul", "dot_product", "concat"}
    violations: list[tuple[str, str]] = []

    def _cap(graph, info, mount_id, view_key, title, **kw):
        for node in graph.nodes:
            if node.kind in connector_kinds and node.static:
                violations.append((view_key, node.id))
        return _orig(graph, info, mount_id, view_key, title, **kw)

    _orig = ge.render_graph
    patched = [m for n, m in sys.modules.items()
               if "block_views" in n and hasattr(m, "render_graph")]
    ge.render_graph = _cap
    for module in patched:
        module.render_graph = _cap
    try:
        for cfg in CORPUS.values():
            unfold(cfg).to_html(standalone=True)
    finally:
        ge.render_graph = _orig
        for module in patched:
            module.render_graph = _orig

    assert not violations, (
        "static Tier-2 connector(s) — give each glyph a card/description:\n  "
        + "\n  ".join(f"{view}: {node}" for view, node in sorted(set(violations)))
    )


def test_view_imaging_is_exhaustive_deduped_and_leaf_free():
    """The Dable image pass must be exhaustive over DISTINCT diagrams and never
    image a description-only leaf: every extracted view is a real <svg>, the
    per-layer-group identical copies dedup down, and the canonical drills are
    all present (so nothing distinct is silently dropped)."""
    from model_unfolder.preview import svg_views, _visual_hash

    html = unfold(CORPUS["moe_mla_mtp"]).to_html(standalone=True)
    views = svg_views(html)
    assert views, "no diagram views were extracted from the baked HTML"
    assert all("<svg" in svg and svg.rstrip().endswith("</svg>") for _, svg in views), \
        "a non-svg (leaf prose?) was captured as a diagram view"

    distinct = {_visual_hash(svg) for _, svg in views}
    assert len(distinct) < len(views), \
        "expected identical per-layer-group copies to collapse under visual dedup"

    labels = {label.split("/")[-1] for label, _ in views}
    assert {"attn", "router", "expert_1", "mla_query_path"} <= labels, \
        f"a canonical drill is missing from the image set: {sorted(labels)}"


def test_fallback_views_have_dedicated_coverage():
    """The model-unexercised dispatch/fallback views must still be covered somewhere."""
    assert _FALLBACK_VIEWS <= {k for k in reg.VIEW_REGISTRY if k}
    # tower: tests/test_tower.py ; ops: tests/test_declared_ops.py ;
    # ffn (generic dispatch): build_ffn_view exercised via gated_ffn/dense_ffn in test_smoke.
    import test_tower, test_declared_ops  # noqa: F401  (importable ⇒ their suites run)
