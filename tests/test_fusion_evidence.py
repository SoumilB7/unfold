"""Wrapper-qualified modality fusion evidence and its shared projections."""
from __future__ import annotations

from copy import deepcopy

import pytest

from model_unfolder import unfold
from model_unfolder.diagram import Diagram
from model_unfolder.evidence.conformance import check_fact_conformance
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.fusion import fusion_evidence
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.sources import resolve_source_files
from model_unfolder.parser import config_to_ir


def _tiny_multimodal_cfg():
    return {
        "architectures": ["Root"], "model_type": "custom_multimodal",
        "image_token_id": 99,
        "vision_config": {"hidden_size": 16, "num_hidden_layers": 1,
                          "num_attention_heads": 2, "patch_size": 4, "image_size": 8},
        "text_config": {"model_type": "llama", "hidden_size": 32,
                        "intermediate_size": 64, "num_hidden_layers": 1,
                        "num_attention_heads": 4, "num_key_value_heads": 2,
                        "vocab_size": 128, "rms_norm_eps": 1e-6},
    }


@pytest.mark.parametrize(("model_type", "owner", "kind", "modalities"), [
    ("paligemma", "PaliGemmaModel", "placeholder_replace", ["vision"]),
    ("llava", "LlavaModel", "placeholder_replace", ["vision"]),
    ("qwen2_vl", "Qwen2VLModel", "unified_multimodal_stream", ["vision", "video"]),
    ("mllama", "MllamaModel", "cross_attention", ["vision"]),
    ("gemma4", "Gemma4Model", "placeholder_replace", ["vision", "video", "audio"]),
])
def test_real_wrapper_fusion_counterexample_matrix(model_type, owner, kind, modalities):
    transformers = pytest.importorskip("transformers")
    evidence = fusion_evidence(transformers.AutoConfig.for_model(model_type).to_dict())
    assert evidence.status == "proven"
    assert evidence.owner_class == owner
    assert evidence.kind == kind
    assert [route.modality for route in evidence.routes] == modalities


def test_paligemma_is_masked_scatter_not_family_prefix():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.AutoConfig.for_model("paligemma").to_dict()
    fusion = unfold(cfg).to_ir()["extras"]["modalities"]["fusion"]
    assert fusion["kind"] == "placeholder_replace"
    assert fusion["operation"] == "scatter_soft_tokens_into_placeholder_slots"
    assert fusion["mechanism"]["operation"] == "masked_scatter"
    assert fusion["source_owner"] == "PaliGemmaModel"


def test_prefix_concat_requires_an_actual_concat_with_text_embeddings(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "import torch\n"
        "class Root:\n"
        "    def forward(self, inputs_embeds, image_features):\n"
        "        return torch.cat([image_features, inputs_embeds], dim=1)\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(source="test", files=(str(source),), architecture="Root")
    evidence = fusion_evidence({}, bundle=bundle)
    assert evidence.status == "proven"
    assert evidence.kind == "prefix_soft_tokens"
    assert [(route.modality, route.operation) for route in evidence.routes] == [
        ("vision", "prefix_concat")
    ]

    cfg = _tiny_multimodal_cfg()
    diagram = Diagram(config_to_ir(cfg, parse_context=ParseContext(bundle)))
    fusion = diagram.to_ir()["extras"]["modalities"]["fusion"]
    assert fusion["kind"] == "prefix_soft_tokens"
    assert diagram.wiring_problems() == []
    html = diagram.to_html(standalone=True)
    assert "Prefix concatenation" in html
    assert "scatter vision features into image-token slots" not in html
    assert "x 1</text>" not in html


def test_unknown_wrapper_is_ambiguous_instead_of_receiving_a_template(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "class Root:\n"
        "    def forward(self, inputs_embeds, image_features):\n"
        "        return self.custom_join(inputs_embeds, image_features)\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(source="test", files=(str(source),), architecture="Root")
    evidence = fusion_evidence({}, bundle=bundle)
    assert evidence.status == "ambiguous"

    cfg = _tiny_multimodal_cfg()
    diagram = Diagram(config_to_ir(cfg, parse_context=ParseContext(bundle)))
    fusion = diagram.to_ir()["extras"]["modalities"]["fusion"]
    assert fusion["kind"] == "code_defined_fusion"
    html = diagram.to_html(standalone=True)
    assert "Code-defined fusion" in html
    assert "scatter vision features into image-token slots" not in html


def test_multi_input_wrapper_keeps_only_configured_modality_routes():
    from tests.test_smoke import _gemma4_e2b_vision_config

    fusion = unfold(_gemma4_e2b_vision_config()).to_ir()["extras"]["modalities"]["fusion"]
    assert fusion["mechanism"]["kind"] == "scatter_many"
    assert [route["source"] for route in fusion["mechanism"]["routes"]] == [
        "modalities.inputs.vision.tokens", "modalities.inputs.audio.tokens",
    ]
    assert all(route["operation"] == "masked_scatter"
               for route in fusion["mechanism"]["routes"])


def test_fusion_fact_conformance_catches_kind_operation_and_routes():
    from tests.test_declared_ops import QWEN2VL_STYLE

    diagram = unfold(QWEN2VL_STYLE)
    bundle = resolve_source_files(QWEN2VL_STYLE)
    clean = [problem for problem in check_fact_conformance(
        QWEN2VL_STYLE, diagram.to_ir(), bundle=bundle,
    ) if problem.kind == "wrong_fusion_fact"]
    assert clean == []

    broken = deepcopy(diagram.to_ir())
    fusion = broken["extras"]["modalities"]["fusion"]
    fusion["operation"] = "interleave_modal_tokens"
    problems = [problem for problem in check_fact_conformance(
        QWEN2VL_STYLE, broken, bundle=bundle,
    ) if problem.kind == "wrong_fusion_fact"]
    assert len(problems) == 1
    assert "operation" in problems[0].message
