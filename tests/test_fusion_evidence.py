"""Wrapper-qualified modality fusion evidence and its shared projections."""
from __future__ import annotations

from copy import deepcopy

import pytest

from model_unfolder import unfold
from model_unfolder.diagram import Diagram
from model_unfolder.evidence.conformance import check_fact_conformance
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.fusion import fusion_evidence
from model_unfolder.evidence.fusion import fusion_result_for_context
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
    # Fusion proves replacement only.  Multi-axis position construction is an
    # independent U9-E fact and upgrades this at projection time.
    ("qwen2_vl", "Qwen2VLModel", "placeholder_replace", ["vision", "video"]),
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


def test_qwen2vl_unified_stream_requires_the_independent_multiaxis_route():
    transformers = pytest.importorskip("transformers")
    from model_unfolder.evidence.multiaxis_position import (
        multimodal_multiaxis_position_result,
    )

    cfg = transformers.AutoConfig.for_model("qwen2_vl").to_dict()
    context = ParseContext.build(cfg)
    base = fusion_evidence(cfg, parse_context=context)
    position = multimodal_multiaxis_position_result(
        context.program_index(), context.source_bundle)
    assert base.kind == "placeholder_replace"
    assert position.status == "resolved" and position.value

    projected = unfold(cfg).to_ir()["extras"]["modalities"]["fusion"]
    assert projected["kind"] == "unified_multimodal_stream"
    assert projected["operation"] == "scatter_grid_tokens_into_placeholder_slots"


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


def test_parser_and_conformance_share_one_call_local_fusion_result(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "import torch\n"
        "class Root:\n"
        "    def forward(self, inputs_embeds, image_features):\n"
        "        return torch.cat([image_features, inputs_embeds], dim=1)\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(source="test", files=(str(source),), architecture="Root")
    context = ParseContext(bundle)
    first = fusion_result_for_context(context)
    second = fusion_result_for_context(context)
    assert first is second
    assert context.reader_results[("root.fusion", ())] is first
    assert fusion_evidence({}, parse_context=context).kind == "prefix_soft_tokens"


def test_non_equivalent_exact_owner_routes_are_ambiguous(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "import torch\n"
        "class Prefix:\n"
        "    def forward(self, inputs_embeds, image_features):\n"
        "        return torch.cat([image_features, inputs_embeds], dim=1)\n"
        "class Scatter:\n"
        "    def forward(self, inputs_embeds, image_features, mask):\n"
        "        return inputs_embeds.masked_scatter(mask, image_features)\n"
        "class Root:\n"
        "    def __init__(self):\n"
        "        self.prefix = Prefix()\n"
        "        self.scatter = Scatter()\n"
        "    def forward(self, x, image_features, mask):\n"
        "        a = self.prefix(x, image_features)\n"
        "        return self.scatter(a, image_features, mask)\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(source="test", files=(str(source),), architecture="Root")
    context = ParseContext(bundle)
    result = fusion_result_for_context(context)
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_constructed_but_uninvoked_fusion_child_is_not_architecture(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "import torch\n"
        "class Unused:\n"
        "    def forward(self, inputs_embeds, image_features):\n"
        "        return torch.cat([image_features, inputs_embeds], dim=1)\n"
        "class Root:\n"
        "    def __init__(self):\n"
        "        self.unused = Unused()\n"
        "    def forward(self, x):\n"
        "        return x\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(source="test", files=(str(source),), architecture="Root")
    context = ParseContext(bundle)
    result = fusion_result_for_context(context)
    assert result.status == "absent"
    assert fusion_evidence({}, parse_context=context).status == "ambiguous"


def test_exact_same_file_helper_is_followed_instead_of_hidden(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "import torch\n"
        "def merge(inputs_embeds, image_features):\n"
        "    return torch.cat([image_features, inputs_embeds], dim=1)\n"
        "class Root:\n"
        "    def forward(self, inputs_embeds, image_features):\n"
        "        return merge(inputs_embeds, image_features)\n",
        encoding="utf-8",
    )
    context = ParseContext(SourceBundle(
        source="test", files=(str(source),), architecture="Root"))
    result = fusion_result_for_context(context)
    assert result.status == "resolved"
    assert result.value.kind == "prefix_soft_tokens"
    assert result.value.operation == "prepend_soft_tokens"


def test_unresolved_helper_cannot_prove_fusion_absent(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "from elsewhere import merge\n"
        "class Root:\n"
        "    def forward(self, inputs_embeds, image_features):\n"
        "        return merge(inputs_embeds, image_features)\n",
        encoding="utf-8",
    )
    context = ParseContext(SourceBundle(
        source="test", files=(str(source),), architecture="Root"))
    result = fusion_result_for_context(context)
    assert result.status == "failed"
    assert result.failures[0].kind == "unsupported_syntax"
    assert fusion_evidence({}, parse_context=context).status == "ambiguous"


def test_visible_fusion_plus_unresolved_rival_is_incomplete(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "import torch\n"
        "from elsewhere import maybe_merge\n"
        "class Root:\n"
        "    def forward(self, inputs_embeds, image_features, flag):\n"
        "        joined = torch.cat([image_features, inputs_embeds], dim=1)\n"
        "        if flag:\n"
        "            joined = maybe_merge(inputs_embeds, image_features)\n"
        "        return joined\n",
        encoding="utf-8",
    )
    context = ParseContext(SourceBundle(
        source="test", files=(str(source),), architecture="Root"))
    result = fusion_result_for_context(context)
    assert result.status == "incomplete"
    assert result.value.kind == "prefix_soft_tokens"
    assert result.failures[0].kind == "unsupported_syntax"
    assert fusion_evidence({}, parse_context=context).status == "ambiguous"


def test_unresolved_feature_tower_return_does_not_compete_with_wrapper_fusion(tmp_path):
    """An operand producer cannot replace a later wrapper fusion operation."""
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "import torch\n"
        "from elsewhere import Output\n"
        "class Tower:\n"
        "    def forward(self, image_features):\n"
        "        return Output(pooler_output=image_features)\n"
        "class Root:\n"
        "    def __init__(self): self.tower = Tower()\n"
        "    def forward(self, inputs_embeds, image_features, mask):\n"
        "        image_features = self.tower(image_features).pooler_output\n"
        "        return inputs_embeds.masked_scatter(mask, image_features)\n",
        encoding="utf-8",
    )
    context = ParseContext(SourceBundle(
        source="test", files=(str(source),), architecture="Root"))
    result = fusion_result_for_context(context)
    assert result.status == "resolved"
    assert result.value.kind == "placeholder_replace"


def test_unresolved_output_packaging_downstream_of_fusion_is_not_a_rival(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "from elsewhere import Output\n"
        "class Root:\n"
        "    def forward(self, x, cross_attention_states):\n"
        "        cross_attention_states = cross_attention_states.to(x.device)\n"
        "        outputs = consume(x, cross_attention_states=cross_attention_states)\n"
        "        return Output(last_hidden_state=outputs.last_hidden_state)\n",
        encoding="utf-8",
    )
    context = ParseContext(SourceBundle(
        source="test", files=(str(source),), architecture="Root"))
    result = fusion_result_for_context(context)
    assert result.status == "resolved"
    assert result.value.kind == "cross_attention"


def test_multi_input_wrapper_keeps_only_configured_modality_routes():
    from test_support import _gemma4_e2b_vision_config

    fusion = unfold(_gemma4_e2b_vision_config()).to_ir()["extras"]["modalities"]["fusion"]
    assert fusion["mechanism"]["kind"] == "scatter_many"
    assert [route["source"] for route in fusion["mechanism"]["routes"]] == [
        "modalities.inputs.vision.tokens", "modalities.inputs.audio.tokens",
    ]
    assert all(route["operation"] == "masked_scatter"
               for route in fusion["mechanism"]["routes"])


def test_fusion_fact_conformance_catches_kind_operation_and_routes():
    from test_support import QWEN2VL_STYLE

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
