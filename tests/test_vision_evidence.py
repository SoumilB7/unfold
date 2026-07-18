"""Counterexample matrix for component-qualified vision source evidence."""
from copy import deepcopy

from model_unfolder import unfold
from model_unfolder.evidence.conformance import check_fact_conformance
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.vision import vision_tower_evidence
from test_support import PIXTRAL_STYLE, QWEN2VL_STYLE
from test_support import GEMMA4_VISION_TINY_CONFIG, MLLAMA_VISION_TINY_CONFIG


def _wrapper(root_type, root_arch, vision_type, vision_arch):
    return {
        "model_type": root_type,
        "architectures": [root_arch],
        "vision_config": {"model_type": vision_type, "architectures": [vision_arch]},
    }


SIGLIP = _wrapper("paligemma", "PaliGemmaForConditionalGeneration",
                  "siglip_vision_model", "SiglipVisionModel")
CLIP = _wrapper("llava", "LlavaForConditionalGeneration",
                "clip_vision_model", "CLIPVisionModel")
QWEN25 = _wrapper("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration",
                  "qwen2_5_vl", "Qwen2_5_VisionTransformerPretrainedModel")
QWEN3 = _wrapper("qwen3_vl", "Qwen3VLForConditionalGeneration",
                 "qwen3_vl", "Qwen3VLVisionModel")


def _evidence(cfg):
    context = ParseContext.build(cfg)
    return vision_tower_evidence(cfg, bundle=context.source_bundle)


def test_siglip_and_clip_are_dense_layernorm_towers_with_real_conv_order():
    for cfg in (SIGLIP, CLIP):
        evidence = _evidence(cfg)
        assert evidence.status == "proven"
        assert evidence.position_kind == "learned_absolute"
        # Humanized structural label (a raw torch class name on a box was
        # the Theme-L leak) + the flatten/transpose pair collapsed into ONE
        # regroup step — CONV ORDER still locked: conv first, then the
        # regroup whose card enumerates the moves in execution order.
        assert evidence.patch_ops[0].label == "Patch convolution"
        assert evidence.patch_ops[1].kind == "reshape"
        assert ("Flatten spatial grid → Transpose to tokens"
                in evidence.patch_ops[1].description)
        layer = evidence.variants[0]
        assert (layer.norm_kind, layer.ffn_gated, layer.projection_mode) == (
            "LayerNorm", False, "separate_qkv")


def test_pixtral_is_gated_rmsnorm_without_affecting_dense_counterexamples():
    layer = _evidence(PIXTRAL_STYLE).variants[0]
    assert (layer.norm_kind, layer.norm_placement, layer.ffn_gated) == (
        "RMSNorm", "pre", True)
    assert _evidence(SIGLIP).variants[0].ffn_gated is False


def test_qwen_generations_keep_dense_vs_gated_and_fused_qkv_distinct():
    qwen2 = _evidence(QWEN2VL_STYLE).variants[0]
    qwen25 = _evidence(QWEN25).variants[0]
    qwen3 = _evidence(QWEN3).variants[0]
    assert (qwen2.norm_kind, qwen2.ffn_gated, qwen2.projection_mode) == (
        "LayerNorm", False, "fused_qkv")
    assert (qwen25.norm_kind, qwen25.ffn_gated, qwen25.projection_mode) == (
        "RMSNorm", True, "fused_qkv")
    assert qwen3.projection_mode == "fused_qkv"


def test_mllama_preserves_local_and_global_constructor_variants():
    evidence = _evidence(MLLAMA_VISION_TINY_CONFIG)
    assert [(item.variant_key, item.repeat_field, item.residual_gated)
            for item in evidence.variants] == [
        ("transformer", "num_hidden_layers", False),
        ("global_transformer", "num_global_layers", True),
    ]
    html = unfold(MLLAMA_VISION_TINY_CONFIG).to_html(standalone=False)
    assert "× 32" in html and "× 8" in html
    assert ">transformer<" not in html and ">global_transformer<" not in html
    # Both constructor variants render as layer-type GROUPS through the one
    # tower projector; only the gated (global) group draws the × gates, and
    # every gate couples to its card.
    for node in ("vision_enc_g0_op_selfattn", "vision_enc_g1_op_selfattn",
                 "vision_enc_g1_op_selfattn_gate", "vision_enc_g1_op_ffn_gate"):
        assert f'data-id="{node}"' in html
        assert f'data-card-id="{node}"' in html
    assert 'data-id="vision_enc_g0_op_selfattn_gate"' not in html


def test_gemma4_surfaces_double_norm_and_qkv_norms():
    evidence = _evidence(GEMMA4_VISION_TINY_CONFIG)
    layer = evidence.variants[0]
    assert [(op.kind, op.label) for op in evidence.patch_ops[:2]] == [
        ("elementwise", "Normalize pixels"), ("linear", "Linear")]
    assert layer.norm_placement == "double"
    assert (layer.q_norm, layer.k_norm, layer.v_norm) == (True, True, True)
    html = unfold(GEMMA4_VISION_TINY_CONFIG).to_html(standalone=False)
    # Q/K/V norms surface inside the ONE namespaced canonical attention region.
    for node in ("vision_enc_attn_q_norm", "vision_enc_attn_k_norm",
                 "vision_enc_attn_v_norm"):
        assert f'data-id="{node}"' in html
        assert f'data-card-id="{node}"' in html
    # Sandwich placement: 4 norm NODES share the cell's one norm card (pre +
    # post per sublayer) — a pre-norm tower draws exactly 2.
    assert html.count('data-id="vision_enc_op_norm"') == 4


def test_missing_vision_oracle_is_unknown_not_a_standard_vit_cell():
    evidence = vision_tower_evidence({}, bundle=SourceBundle(source="local"))
    assert evidence.status == "oracle_missing"


_ROOT_FALLBACK_SOURCE = '''
class OpticalAttention:
    def __init__(self):
        self.q_proj = Linear()
    def forward(self, x):
        return self.q_proj(x)

class OpticalMLP:
    def __init__(self):
        self.fc1 = Linear()
        self.fc2 = Linear()
    def forward(self, x):
        return self.fc2(self.fc1(x))

class OpticalCell:
    def __init__(self):
        self.norm = LayerNorm()
        self.attn = OpticalAttention()
        self.mlp = OpticalMLP()
    def forward(self, x):
        x = self.norm(x)
        x = self.attn(x)
        return self.mlp(x)

class OpticalTower:
    def __init__(self):
        self.patch_embed = Conv2d()
        self.layers = ModuleList([OpticalCell()])
    def forward(self, pixels):
        x = self.patch_embed(pixels)
        for layer in self.layers:
            x = layer(x)
        return x

class TextDecoderLayer:
    def __init__(self):
        self.attn = OpticalAttention()
        self.mlp = OpticalMLP()
    def forward(self, x):
        return self.mlp(self.attn(x))

class Inner:
    def __init__(self, config):
        self.visual = OpticalTower._from_config(config.vision_config)
        self.layers = ModuleList([TextDecoderLayer()])
    def forward(self, x):
        return self.visual(x)

class Wrapper:
    def __init__(self, config):
        self.model = Inner(config)
    def forward(self, x):
        return self.model(x)
'''


def _root_fallback_bundle(path):
    return SourceBundle(
        source="path", files=(str(path),), architecture="Wrapper",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
    )


def test_root_fallback_follows_wrapper_assignment_not_a_vision_name(tmp_path):
    source = tmp_path / "modeling_wrapper.py"
    source.write_text(_ROOT_FALLBACK_SOURCE)
    evidence = vision_tower_evidence(
        {"vision_config": {}}, bundle=_root_fallback_bundle(source),
    )
    assert evidence.status == "proven"
    assert evidence.owner_class == "OpticalTower"
    assert [item.block_class for item in evidence.variants] == ["OpticalCell"]


def test_root_fallback_does_not_guess_automodel_delegate_from_class_names(tmp_path):
    source = tmp_path / "modeling_wrapper.py"
    source.write_text(_ROOT_FALLBACK_SOURCE.replace(
        "OpticalTower._from_config(config.vision_config)",
        "AutoModel.from_config(config.vision_config)",
    ))
    evidence = vision_tower_evidence(
        {"vision_config": {}}, bundle=_root_fallback_bundle(source),
    )
    assert evidence.status == "ambiguous"
    assert evidence.reason == "root wrapper does not prove the delegated vision class"


def test_vision_fact_conformance_consumes_the_same_typed_evidence():
    context = ParseContext.build(PIXTRAL_STYLE)
    ir = unfold(PIXTRAL_STYLE).to_ir()
    clean = check_fact_conformance(PIXTRAL_STYLE, ir, bundle=context.source_bundle)
    assert not [problem for problem in clean if problem.kind == "wrong_vision_fact"]
    broken = deepcopy(ir)
    broken["extras"]["modalities"]["inputs"]["vision"]["encoder"]["variants"][0]["norm_kind"] = "LayerNorm"
    problems = check_fact_conformance(PIXTRAL_STYLE, broken, bundle=context.source_bundle)
    assert any(problem.kind == "wrong_vision_fact" and "norm_kind" in problem.op
               for problem in problems)


def test_rec5_projector_width_is_code_bound_or_honest_debt():
    """REC-5 (§11.2/§11.4, R-10) as amended by COR-4 (§9): the width-comparison
    heuristic is DELETED at source level (no family branch may replace it) and
    so is the whole generic out-width author.  On the source-present witness
    the construction-site binding CONSUMES ``vision_config.hidden_size``
    exactly (fact ``projector_out_features``), so the registered debt row is
    DISCHARGED there — it remains registered for source-less grid towers."""
    import json
    import pathlib

    import model_unfolder as mu
    from model_unfolder.evidence.structural_debt import (
        pending_projection_paths)

    src = (pathlib.Path(mu.__file__).parent / "adapters" / "transformer" /
           "special_parts" / "modalities" / "vision.py").read_text()
    assert "_resolve_out_width" not in src   # the heuristic (and any wrapper)
    assert "def vision_projector_out" not in src   # COR-4: the generic author

    assert ("root.vision", "vision_config.hidden_size") \
        in pending_projection_paths()

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    cfg = json.loads((corpus / "qwen2-vl-7b-instruct.json").read_text())["config"]
    extras = mu.unfold(cfg).to_ir().get("extras") or {}
    audit = extras.get("config_audit", {})
    assert audit.get("unread") == []
    assert "vision_config.hidden_size" not in (audit.get("pending_projection") or [])
    rows = (extras.get("config_access") or {}).get("projection_obligations") or []
    bound = [r for r in rows
             if (r.get("source") or {}).get("path") == "vision_config.hidden_size"]
    assert bound, "the consumed width must surface as a projection obligation"
    # the root.vision entry cannot excuse a TRANSFORMER-root hidden_size —
    # a root-level unread hidden_size would still flag (P5/P6 family guard).
