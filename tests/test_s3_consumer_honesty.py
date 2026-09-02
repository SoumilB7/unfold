"""S3: terminal consumers expose unknowns instead of completing them."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_unfolder import ConfigParseError, unfold
from model_unfolder.adapters.transformer.blocks.model import (
    block_diffusion_loop_blocks,
)
from model_unfolder.adapters.diffusor.blocks import _encoder_norm_card
from model_unfolder.ir import AttentionSpec
from model_unfolder.labels import attention_label
from model_unfolder.evidence.conformance import diagram_op_set
from model_unfolder.opgraph import Edge, Op, Region
from model_unfolder.renderers.html.block_views.attention import (
    _apply_presentation,
)
from model_unfolder.renderers.html.graph import Graph, KIND, Node
from model_unfolder.renderers.html.graph_engine import render_graph
from model_unfolder.renderers.html.render_context import (
    RenderContext,
    activate_render_context,
)
from model_unfolder.renderers.html.metadata_modalities import (
    _audio_norm_card,
    _conditioning_norm_card,
    _vision_norm_card,
)
from model_unfolder.renderers.html.op_render import region_to_graph


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def test_unknown_graph_kind_is_visibly_unknown_not_norm():
    node = Node("future", "future_operation")
    assert node.glyph() is KIND["unknown"]
    assert node.glyph() is not KIND["norm"]
    assert node.glyph().label == "Operation unresolved"
    assert node.presentation_resolved() is False


def test_unknown_op_kind_projects_to_an_unresolved_operation():
    region = Region(
        "future", "test", "Future operation",
        [Op("hidden", "input"), Op("future", "future_operation")],
        [Edge("hidden", "future")],
    )
    node = region_to_graph(region).by_id()["future"]
    assert node.kind == "unknown"
    assert node.resolved is False
    assert node.static is True
    assert node.meta == {"source_kind": "future_operation"}


def test_visible_unknown_is_a_censused_non_compute_presentation_kind():
    from model_unfolder.everchanging import load_conformance_transitive

    assert diagram_op_set({"blocks": [{"kind": "unknown"}]}) == frozenset()
    assert "unknown" in load_conformance_transitive()["drawn_ignore"]
    context = RenderContext()
    graph = Graph(
        nodes=[Node("future", "unknown", resolved=False, static=True)],
        flow=["future"],
    )
    with activate_render_context(context):
        render_graph(graph, {"blocks": {}}, "s3", "future", "Future")
    assert context.events[0].node_ids == frozenset({"future"})
    assert context.events[0].drawn_ops == frozenset({"unknown"})


def test_visible_unknown_does_not_hide_a_missing_source_closure(monkeypatch):
    """Presentation neutrality must not weaken the existing closure gate."""
    from model_unfolder.evidence import conformance as conformance_module

    monkeypatch.setattr(
        conformance_module, "resolve_source_files",
        lambda *_args, **_kwargs: SimpleNamespace(files=("modeling.py",),
                                                   component_files={}),
    )
    monkeypatch.setattr(
        conformance_module, "build_registry", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        conformance_module, "_resolve_drill_closure",
        lambda *_args, **_kwargs: None,
    )
    problems = conformance_module.check_nested_conformance(
        {"model_type": "synthetic"},
        [("ffn", frozenset({"unknown", "port"}), frozenset({"future"}))],
    )
    assert [(problem.kind, problem.view) for problem in problems] == [
        ("unresolved", "synthetic/ffn"),
    ]


def test_unknown_norm_placement_is_chipped_not_defaulted_to_pre_norm():
    builders = (
        _audio_norm_card, _conditioning_norm_card, _vision_norm_card,
    )
    for builder in builders:
        unknown = builder("tower", "LayerNorm", None)
        assert "unresolved" in unknown["description"]
        assert unknown["facts"] == ["placement unresolved"]
        assert "pre-norm" not in unknown["description"]

        known = builder("tower", "LayerNorm", "pre")
        assert "pre-norm" in known["description"]
        assert "facts" not in known

    # The diffusion encoder fallback is a separate consumer surface and must
    # obey the same unknown-preservation law.
    unknown = _encoder_norm_card("encoder", "RMSNorm", None)
    assert unknown["facts"] == ["placement unresolved"]
    assert "pre-norm" not in unknown["description"]
    known = _encoder_norm_card("encoder", "RMSNorm", "pre")
    assert "pre-norm" in known["description"]
    assert "facts" not in known


def test_cross_attention_label_does_not_classify_kv_source_from_prose():
    prompt = AttentionSpec(
        "mha", 8, cross_attention=True,
        cross_kv_source="encoded prompt states from an encoder")
    arbitrary = AttentionSpec(
        "mha", 8, cross_attention=True,
        cross_kv_source="some source prose without a modality")
    assert attention_label(prompt) == attention_label(arbitrary) == [
        "Cross-Attention", "(K/V source unresolved)",
    ]


def test_block_diffusion_missing_values_stay_visible_and_unresolved():
    blocks = {
        block["id"]: block for block in block_diffusion_loop_blocks(
            2, 64, 128, 16,
            final_logit_softcap=None,
            ffn_intermediate_size=96,
        )
    }
    text = json.dumps(blocks, ensure_ascii=False)
    assert "48 steps" not in text
    assert "ε=0.1" not in text
    assert "softcap ±30" not in text
    assert "step bound unresolved" in blocks["bd_canvas"]["facts"]
    assert "entropy bound unresolved" in blocks["bd_sampler"]["facts"]
    assert "softcap unresolved" in blocks["bd_lm_head"]["facts"]

    exact = {
        block["id"]: block for block in block_diffusion_loop_blocks(
            2, 64, 128, 16, final_logit_softcap=17)
    }
    assert "softcap ±17.0" in exact["bd_lm_head"]["facts"]


def test_sliding_window_depiction_has_no_fabricated_token_strip():
    graph = Graph(
        nodes=[Node("hidden", "input", meta={"canonical_id": "hidden"})],
        flow=["hidden"],
    )
    _apply_presentation(graph, {"mask": "sliding", "window_size": None})
    node = graph.by_id()["hidden"]
    assert node.kind == "context_window"
    assert node.glyph().shape == "rect"
    assert node.label == ["Sliding context", "window size unresolved"]
    assert not hasattr(node, "active_start")


def test_bare_sd35_is_an_unresolved_denoiser_not_a_text_tower():
    config = json.loads(
        (_CORPUS / "stable-diffusion-3-5-large.json").read_text(
            encoding="utf-8"))["config"]
    diagram = unfold(config)
    ir = diagram.to_ir()
    html = diagram.to_html()
    assert ir["layers"] == []
    assert any("No root denoiser layers were materialized" in warning
               for warning in ir["warnings"])
    assert "No root denoiser layers materialized" in html
    assert "Repeated denoiser" in html and "structure unresolved" in html
    assert 'data-id="embed"' not in html
    assert 'data-id="final_rms"' not in html
    assert "Tokenized text" not in html
    assert "Token Embedding" not in html
    assert "LM head" not in html


def test_qwen3_conditional_generation_plain_unfold_has_no_projector_traceback(
    monkeypatch,
):
    """Exercise the original crash class through the public plain-dict path.

    The fake graph reproduces the exact producer defect: lineage found a caller
    which the component-root graph cannot address.  That is an address-integrity
    failure, not an unknown mechanism, so the public path must return the typed
    crash-class refusal required by C-3 rather than silently dropping it.
    """
    transformers = pytest.importorskip("transformers")
    from model_unfolder.evidence import projector as projector_module

    config = transformers.AutoConfig.for_model("qwen3_vl").to_dict()
    config["architectures"] = ["Qwen3VLForConditionalGeneration"]
    calls = []

    class MissingCallerGraph:
        def node_for(self, occurrence):
            calls.append(occurrence)
            return None

    monkeypatch.setattr(
        projector_module, "resolve_component_root",
        lambda *_args, **_kwargs: SimpleNamespace(
            graph=MissingCallerGraph()))
    with pytest.raises(ConfigParseError, match="Projector evidence"):
        unfold(config)
    assert calls, "the poison must exercise the missing-caller join"
