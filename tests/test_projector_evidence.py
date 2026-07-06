"""Qualified multimodal connector evidence and its shared IR/conformance rail."""
from __future__ import annotations

from copy import deepcopy

import pytest

from model_unfolder import unfold
from model_unfolder.evidence.conformance import check_fact_conformance
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.projector import projector_evidence
from model_unfolder.evidence.sources import resolve_source_files


@pytest.mark.parametrize(("model_type", "expected_class", "expected_kind", "expected_ops"), [
    ("paligemma", "PaliGemmaMultiModalProjector", "linear_projector",
     ["linear"]),
    ("llava", "LlavaMultiModalProjector", "mlp_projector",
     ["linear", "activation", "linear"]),
    ("qwen2_vl", "PatchMerger", "patch_merger",
     ["norm", "reshape", "linear", "activation", "linear"]),
    ("mistral3", "Mistral3MultiModalProjector", "patch_merger",
     ["norm", "reshape", "reshape", "reshape", "reshape", "reshape",
      "linear", "linear", "activation", "linear"]),
    ("gemma4", "Gemma4MultimodalEmbedder", "linear_projector",
     ["norm", "linear"]),
    ("mllama", "Linear", "linear_projector", ["linear"]),
])
def test_real_projector_counterexample_matrix(
    model_type, expected_class, expected_kind, expected_ops,
):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.AutoConfig.for_model(model_type).to_dict()
    evidence = projector_evidence(cfg)
    assert evidence.status == "proven"
    assert evidence.projector_class == expected_class
    assert evidence.kind == expected_kind
    assert [op.kind for op in evidence.ops] == expected_ops


def test_idefics_connector_follows_factory_resampler_and_learned_queries():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.AutoConfig.for_model("idefics2").to_dict()
    evidence = projector_evidence(cfg)
    assert evidence.projector_class == "Idefics2Connector"
    assert evidence.kind == "perceiver_resampler"
    assert evidence.learned_queries is True
    assert [(op.kind, op.label) for op in evidence.ops] == [
        ("linear", "Linear (gate)"),
        ("linear", "Linear (up)"),
        ("activation", "gelu_pytorch_tanh"),
        ("elementwise", "Multiply"),
        ("linear", "Linear (out)"),
        ("opaque", "Perceiver layer"),
        ("norm", "RMSNorm"),
    ]
    layer = evidence.ops[5]
    assert layer.repeat == cfg["perceiver_config"]["resampler_depth"]
    assert "cross-attend" in layer.description and "MLP" in layer.description


def test_generic_projection_requires_execution_shaped_wrapper_proof(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "class Root:\n"
        "    def __init__(self):\n"
        "        self.embedding_projection = Linear()\n"
        "        self.pre = LayerNorm()\n"
        "    def forward(self, x):\n"
        "        return self.embedding_projection(self.pre(x))\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(source="test", files=(str(source),), architecture="Root")
    evidence = projector_evidence({}, bundle=bundle)
    assert evidence.status == "proven"
    assert evidence.owner_class == "Root"
    assert evidence.projector_class == "Root"
    assert evidence.kind == "linear_projector"
    assert [op.kind for op in evidence.ops] == ["norm", "linear"]

    source.write_text(
        "class Root:\n"
        "    def __init__(self):\n"
        "        self.projection = Linear()\n"
        "    def forward(self, x):\n"
        "        return self.projection(x)\n",
        encoding="utf-8",
    )
    evidence = projector_evidence({}, bundle=bundle)
    assert evidence.status == "ambiguous"


def test_projector_fact_conformance_is_bidirectional():
    from tests.test_declared_ops import QWEN2VL_STYLE

    diagram = unfold(QWEN2VL_STYLE)
    bundle = resolve_source_files(QWEN2VL_STYLE)
    clean = [problem for problem in check_fact_conformance(
        QWEN2VL_STYLE, diagram.to_ir(), bundle=bundle,
    ) if problem.kind == "wrong_projector_fact"]
    assert clean == []

    broken = deepcopy(diagram.to_ir())
    projector = broken["extras"]["modalities"]["inputs"]["vision"]["projector"]
    projector["ops"] = projector["ops"][:-1]
    problems = [problem for problem in check_fact_conformance(
        QWEN2VL_STYLE, broken, bundle=bundle,
    ) if problem.kind == "wrong_projector_fact"]
    assert len(problems) == 1
    assert "vision.ops" in problems[0].message


def test_projector_ir_has_no_family_profile_or_family_title():
    from tests.test_declared_ops import MISTRAL3_STYLE

    diagram = unfold(MISTRAL3_STYLE)
    projector = diagram.to_ir()["extras"]["modalities"]["inputs"]["vision"]["projector"]
    assert "profile" not in projector
    html = diagram.to_html(standalone=True)
    assert "Mistral3 multimodal projector" not in html
    assert "Patch merger" in html
