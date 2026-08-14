"""Qualified multimodal connector evidence and its shared IR/conformance rail."""
from __future__ import annotations

from copy import deepcopy

import pytest

from model_unfolder import unfold
from model_unfolder.evidence.conformance import check_fact_conformance
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.projector import projector_evidence
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.sources import resolve_source_files


def _exact_projector_evidence(cfg, bundle=None):
    bundle = bundle or resolve_source_files(cfg)
    return projector_evidence(
        cfg, bundle=bundle, index=build_program_index(bundle),
        config_selector=lambda path: _select(cfg, path))


def _select(cfg, path):
    value = cfg
    for part in path:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return False, None, ""
    return True, value, "config_declared"


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
    if model_type == "gemma4":
        from test_support import _gemma4_e2b_vision_config
        cfg = _gemma4_e2b_vision_config()
    else:
        cfg = transformers.AutoConfig.for_model(model_type).to_dict()
    evidence = _exact_projector_evidence(cfg)
    assert evidence.status == "proven"
    assert evidence.projector_class == expected_class
    assert evidence.kind == expected_kind
    assert [op.kind for op in evidence.ops] == expected_ops


# COR-4 (§9): the OUT width the construction site actually wires, per shape —
# param-fed through a factory chain (qwen2_vl), config-fed (paligemma: the
# VISION config's own projection_dim, not the language width; llava/mistral3:
# the text width, lawful because the source names it), and a primitive Linear
# field bound at the owner's own site (mllama).  idefics2's perceiver hides
# its widths behind loop-built layers: the binder must refuse, not guess.
# The IN column pins the other review shapes: a derived arithmetic entry
# (qwen2_vl context_dim*merge², llava's feature-concat multiply) is
# established-not-reduced, and mllama's primitive binds both ends.
@pytest.mark.parametrize(("model_type", "out_source", "out_path", "in_source"), [
    ("qwen2_vl", "config_bound", ("vision_config", "hidden_size"), "derived"),
    ("paligemma", "config_bound", ("vision_config", "projection_dim"), "config_bound"),
    ("llava", "config_bound", ("text_config", "hidden_size"), "derived"),
    ("mistral3", "config_bound", ("text_config", "hidden_size"), "derived"),
    ("mllama", "config_bound", ("text_config", "hidden_size"), "config_bound"),
    ("idefics2", "unavailable", (), "unavailable"),
])
def test_out_width_binding_is_construction_site_exact(
    model_type, out_source, out_path, in_source,
):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.AutoConfig.for_model(model_type).to_dict()
    evidence = _exact_projector_evidence(cfg)
    assert evidence.out_width_source == out_source
    assert tuple(evidence.out_width_path) == out_path
    assert evidence.in_width_source == in_source
    if out_source == "config_bound":
        assert evidence.out_width_value is None    # values resolve at the
        # consumer through the evented accessor, never inside evidence


def test_inactive_optional_component_does_not_author_a_projector():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.AutoConfig.for_model("gemma4").to_dict()
    assert cfg.get("vision_config") is None
    evidence = _exact_projector_evidence(cfg)
    assert evidence.status != "proven"
    assert evidence.ops == ()


def test_width_binding_is_occurrence_exact_and_rivals_refuse(tmp_path):
    """The same class at two prefixes is not a union: the terminal occurrence
    reaching fusion binds its own path. Rival writes to that occurrence refuse."""
    prefix_conflict = tmp_path / "modeling_prefix_conflict.py"
    prefix_conflict.write_text(
        "from torch.nn import Linear\n"
        "class Proj:\n"
        "    def __init__(self, config):\n"
        "        self.out = Linear(4, config.width)\n"
        "    def forward(self, x):\n"
        "        return self.out(x)\n"
        "class Wrap:\n"
        "    def __init__(self, config):\n"
        "        self.projector = Proj(config.vision_config)\n"
        "    def forward(self, x):\n"
        "        return self.projector(x)\n"
        "class Root:\n"
        "    def __init__(self, config):\n"
        "        self.projector = Proj(config.audio_config)\n"
        "        self.wrap = Wrap(config)\n"
        "    def forward(self, inputs_embeds, image_features, mask):\n"
        "        image_features = self.projector(image_features)\n"
        "        image_features = self.wrap(image_features)\n"
        "        return inputs_embeds.masked_scatter(mask, image_features)\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(source="test", files=(str(prefix_conflict),), architecture="Root")
    evidence = _exact_projector_evidence({}, bundle)
    assert evidence.status == "proven"
    assert evidence.out_width_source == "config_bound"
    assert evidence.out_width_path == ("vision_config", "width")

    field_conflict = tmp_path / "modeling_field_conflict.py"
    field_conflict.write_text(
        "from torch.nn import Linear\n"
        "class Root:\n"
        "    def __init__(self, config, flag):\n"
        "        if flag:\n"
        "            self.projector = Linear(4, config.a_width)\n"
        "        else:\n"
        "            self.projector = Linear(4, config.b_width)\n"
        "    def forward(self, inputs_embeds, image_features, mask):\n"
        "        image_features = self.projector(image_features)\n"
        "        return inputs_embeds.masked_scatter(mask, image_features)\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(source="test", files=(str(field_conflict),), architecture="Root")
    evidence = _exact_projector_evidence({}, bundle)
    assert evidence.status != "proven"
    assert evidence.out_width_source == "unavailable"

    control = tmp_path / "modeling_control.py"
    control.write_text(
        "from torch.nn import Linear\n"
        "class Proj:\n"
        "    def __init__(self, config):\n"
        "        self.out = Linear(4, config.width)\n"
        "    def forward(self, x):\n"
        "        return self.out(x)\n"
        "class Wrap:\n"
        "    def __init__(self, config):\n"
        "        self.projector = Proj(config.vision_config)\n"
        "    def forward(self, x):\n"
        "        return self.projector(x)\n"
        "class Root:\n"
        "    def __init__(self, config):\n"
        "        self.wrap = Wrap(config)\n"
        "    def forward(self, inputs_embeds, image_features, mask):\n"
        "        image_features = self.wrap(image_features)\n"
        "        return inputs_embeds.masked_scatter(mask, image_features)\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(source="test", files=(str(control),), architecture="Root")
    proven = _exact_projector_evidence({}, bundle)
    assert proven.status == "proven"
    assert proven.out_width_source == "config_bound"
    assert tuple(proven.out_width_path) == ("vision_config", "width")
    assert proven.out_width_value is None


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


def test_generic_projection_without_fusion_is_not_relabelled_multimodal(tmp_path):
    source = tmp_path / "modeling_custom.py"
    source.write_text(
        "from torch.nn import Linear, LayerNorm\n"
        "class Root:\n"
        "    def __init__(self):\n"
        "        self.embedding_projection = Linear()\n"
        "        self.pre = LayerNorm()\n"
        "    def forward(self, x):\n"
        "        return self.embedding_projection(self.pre(x))\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(source="test", files=(str(source),), architecture="Root")
    evidence = _exact_projector_evidence({}, bundle)
    assert evidence.status == "ambiguous"
    assert evidence.ops == ()

    source.write_text(
        "from torch.nn import Linear\n"
        "class Root:\n"
        "    def __init__(self):\n"
        "        self.projection = Linear()\n"
        "    def forward(self, x):\n"
        "        return self.projection(x)\n",
        encoding="utf-8",
    )
    evidence = _exact_projector_evidence({}, bundle)
    assert evidence.status == "ambiguous"


def test_projector_fact_conformance_is_bidirectional():
    from test_support import QWEN2VL_STYLE

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
    from test_support import MISTRAL3_STYLE

    diagram = unfold(MISTRAL3_STYLE)
    projector = diagram.to_ir()["extras"]["modalities"]["inputs"]["vision"]["projector"]
    assert "profile" not in projector
    html = diagram.to_html(standalone=True)
    assert "Mistral3 multimodal projector" not in html
    assert "Patch merger" in html
