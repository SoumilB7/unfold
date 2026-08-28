"""Op-conformance: the diagram's op-set must match the model's HF forward() code.

The net for the class of bug where the picture is internally perfect (coupling /
wiring / unique-ids all green) yet diverges from what the code actually does —
e.g. Flux's single-stream block once drawn as a parallel-sum (no ``concat`` /
``gate_mul``) when ``FluxSingleTransformerBlock.forward`` does
``torch.cat([attn, mlp]) -> proj_out -> gate* -> residual+``.

Three layers here:
  * the EXTRACTOR (code side) — what does a forward() do;
  * the DIFF (both directions) over the offline corpus — does the picture match;
  * the NEGATIVE CONTROL — the old buggy rendering MUST be caught.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import model_unfolder as mu
from model_unfolder.evidence import check_model_conformance, extract_forward_ops
from model_unfolder.evidence.conformance import (
    _typed_stream_relation_unknown,
    diff_conformance,
    resolve_view_code,
)
from model_unfolder.evidence.sources import resolve_source_files
from model_unfolder.everchanging import load_conformance_abstractions, load_conformance_map

from test_support import FLUX, PIXART
from tests import test_coverage as tc


def _flux_forward_ops():
    bundle = resolve_source_files(FLUX, source="local")
    if not bundle.files:
        pytest.skip("diffusers Flux modeling source not installed locally")
    return extract_forward_ops(bundle.files)


def test_typed_unknown_stream_cannot_be_strengthened_by_legacy_param_presence():
    assert _typed_stream_relation_unknown({
        "attention": {"variant": {"stream_relation": None}},
    }) is True
    assert _typed_stream_relation_unknown({
        "attention": {"variant": {"stream_relation": "single_state"}},
    }) is False
    assert _typed_stream_relation_unknown({"attention": {}}) is False


# --------------------------------------------------------------------------
# Stage 1 — the extractor (code side)
# --------------------------------------------------------------------------

def test_extractor_finds_flux_single_stream_fused_topology():
    """The single-stream block's forward fuses attn∥mlp via a concat + an AdaLN
    gate — the exact ops a parallel-sum drawing would be MISSING."""
    fo = _flux_forward_ops().get("FluxSingleTransformerBlock")
    assert fo is not None, "FluxSingleTransformerBlock not found in Flux source"
    assert "concat" in fo.op_kinds and "gate_mul" in fo.op_kinds, fo.op_kinds
    assert {"attention", "linear", "norm", "residual_add", "activation"} <= fo.op_kinds, fo.op_kinds
    assert fo.field_types.get("attn", "").endswith("Attention")
    assert fo.field_types.get("proj_out") == "Linear"


def test_extractor_distinguishes_dual_stream_block():
    """The dual-stream block is a DIFFERENT topology: sequential attn then a real
    FeedForward (ffn), AdaLN-gated — not the single-stream concat fusion."""
    fo = _flux_forward_ops().get("FluxTransformerBlock")
    assert fo is not None
    assert {"attention", "ffn", "gate_mul", "norm", "residual_add"} <= fo.op_kinds, fo.op_kinds
    assert any(v == "FeedForward" for v in fo.field_types.values())


# --------------------------------------------------------------------------
# Stage 2 — the FLUX regression + the negative control
# --------------------------------------------------------------------------

def test_flux_conformance_clean_both_directions():
    """The corrected Flux renders both blocks faithfully — zero conformance gaps."""
    ir = mu.unfold(FLUX).to_ir()
    problems = check_model_conformance(FLUX, ir)
    real = [p for p in problems if p.kind in ("missing", "fabricated", "stale")]
    assert real == [], "\n".join(p.message for p in real)


def test_source_join_relation_selects_single_stream_without_label_inference():
    from model_unfolder.evidence.conformance import classify_group

    spec = {"attention": {"variant": {
        "stream_relation": "joined_inputs",
        "tag": "presentation wording may change",
    }}}
    assert classify_group(spec) == "single_stream"
    spec["attention"]["variant"]["stream_relation"] = "dual_state"
    assert classify_group(spec) == "block"


def test_source_join_relation_projects_the_concat_operation():
    ir = mu.unfold(FLUX).to_ir()
    joined = [layer for layer in ir["layers"]
              if (layer.get("attention") or {}).get(
                  "variant", {}).get("stream_relation") == "joined_inputs"]
    assert joined
    assert all(any(block.get("kind") == "concat"
                   and block.get("feeds") == "attn"
                   for block in layer["blocks"])
               for layer in joined)
    assert all(not any(block.get("kind") == "concat"
                       for block in layer["blocks"])
               for layer in ir["layers"] if layer not in joined)


def test_negative_control_parallel_sum_rendering_is_caught():
    """THE pin: a GPT-J parallel-sum single-stream rendering (no concat, no gate)
    MUST fail the diff with both ops flagged missing — citing the forward()."""
    code = _flux_forward_ops()["FluxSingleTransformerBlock"]
    ab = load_conformance_abstractions()
    buggy_diagram = frozenset({"norm", "attention", "ffn", "residual_add"})  # the old wrong picture
    problems = diff_conformance(buggy_diagram, code, "flux", "single_stream", ab)
    missing = {p.op for p in problems if p.kind == "missing"}
    assert {"concat", "gate_mul"} <= missing, [p.message for p in problems]
    assert any("transformer_flux" in p.source_file for p in problems if p.kind == "missing")


def test_typed_unknown_cell_waives_only_its_wiring_operations():
    """An honest topology abstention must be mechanically blessable without
    becoming a blanket conformance escape hatch."""
    code = _flux_forward_ops()["FluxSingleTransformerBlock"]
    ab = load_conformance_abstractions()
    drawn = frozenset({"attention", "ffn"})
    spec = {
        "norm_placement": "unknown",
        "residual_topology": "unknown",
    }
    problems = diff_conformance(
        drawn, code, "flux", "single_stream", ab, spec=spec,
    )
    missing = {p.op for p in problems if p.kind == "missing"}
    assert not ({"norm", "gate_mul", "residual_add"} & missing)
    # The typed abstention is narrow: unrelated code operations remain visible.
    assert "concat" in missing


def test_exact_opaque_cell_declares_the_whole_cell_abstraction():
    """Only the closed U10 opaque-cell DTO may abstain on all internals."""
    code = _flux_forward_ops()["FluxTransformerBlock"]
    spec = {
        "norm_placement": "unknown",
        "residual_topology": "unknown",
        "blocks": [{
            "id": "cell_structure_unresolved",
            "kind": "opaque",
            "role": "opaque",
            "resolved": False,
        }],
    }
    assert diff_conformance(
        frozenset(), code, "flux", "block",
        load_conformance_abstractions(), spec=spec,
    ) == []

    # A lookalike opaque block is not an escape hatch.
    forged = {**spec, "blocks": [{**spec["blocks"][0], "id": "anything"}]}
    missing = {p.op for p in diff_conformance(
        frozenset(), code, "flux", "block",
        load_conformance_abstractions(), spec=forged,
    ) if p.kind == "missing"}
    assert {"attention", "ffn"} <= missing


def test_absent_unknown_fields_grant_no_conformance_waiver():
    code = _flux_forward_ops()["FluxTransformerBlock"]
    problems = diff_conformance(
        frozenset({"attention", "ffn"}), code, "flux", "block",
        load_conformance_abstractions(), spec={},
    )
    missing = {p.op for p in problems if p.kind == "missing"}
    assert {"norm", "gate_mul", "residual_add"} <= missing


def test_negative_control_end_to_end_pipeline_catches_buggy_render():
    """A localized wiring unknown cannot launder known attention/FFN ops."""
    ir = mu.unfold(FLUX).to_ir()
    partial = next((layer for layer in ir["layers"]
                    if any(block.get("id") == "wiring_unresolved"
                           for block in (layer.get("blocks") or []))), None)
    assert partial is not None, "no localized Flux wiring unknown to attack"
    # Preserve the narrow typed wiring abstention but erase the mechanisms
    # source did prove.  The abstention must not become a whole-cell waiver.
    partial["blocks"] = [
        block for block in partial["blocks"]
        if block.get("id") == "wiring_unresolved"
    ]
    missing = {p.op for p in check_model_conformance(FLUX, ir) if p.kind == "missing"}
    assert {"attention", "ffn"} <= missing, missing


# --------------------------------------------------------------------------
# Stage 3 — the corpus net + resolver honesty + staleness
# --------------------------------------------------------------------------

def test_op_conformance_both_directions_over_corpus():
    """Across the offline archetype corpus, no view's diagram diverges from its
    forward() code (missing / fabricated / stale). Unresolved views (a family
    whose source isn't installed) are gaps, not failures — see the honesty test."""
    failures: list[str] = []
    for name, cfg in tc.CORPUS.items():
        ir = mu.unfold(cfg).to_ir()
        for p in check_model_conformance(cfg, ir):
            if p.kind in ("missing", "fabricated", "stale"):
                failures.append(f"{name}: {p.message}")
    assert not failures, "op-conformance gaps:\n  " + "\n  ".join(failures)


def test_resolver_binds_the_diffusion_block_views():
    """The net can't silently no-op on the hero cases: Flux's TWO block views and
    PixArt's block view MUST resolve to a real forward() to diff against."""
    flux_ops = _flux_forward_ops()
    cmap = load_conformance_map()
    single = resolve_view_code("flux", "single_stream", {}, flux_ops, cmap)
    dual = resolve_view_code("flux", "block", {}, flux_ops, cmap)
    assert single is not None and single.class_name == "FluxSingleTransformerBlock"
    assert dual is not None and dual.class_name == "FluxTransformerBlock"
    # PixArt's block class lives in models/attention.py — resolved via file augmentation.
    pix_problems = check_model_conformance(PIXART, mu.unfold(PIXART).to_ir())
    assert not [p for p in pix_problems if p.kind == "unresolved"], \
        [p.view for p in pix_problems if p.kind == "unresolved"]


def test_resolver_recurses_into_delegated_transformer_components():
    """A multimodal wrapper is not the full oracle: its nested text and vision
    configs select the concrete classes used by ``AutoModel.from_config``.  The
    source bundle must include all three families, or Sable can falsely pass while
    never reading the delegated tower's forward()."""
    cfg = {
        "model_type": "paligemma",
        "vision_config": {"model_type": "siglip_vision_model"},
        "text_config": {"model_type": "gemma2"},
    }
    bundle = resolve_source_files(cfg, source="local")
    names = {Path(p).name for p in bundle.files}
    required = {"modeling_paligemma.py", "modeling_siglip.py", "modeling_gemma2.py"}
    if not required <= names:
        pytest.skip("delegated PaliGemma component sources are not installed locally")
    assert required <= names
    assert {Path(p).name for p in bundle.component_files["root"]} == {"modeling_paligemma.py"}
    assert {Path(p).name for p in bundle.component_files["vision_config"]} == {"modeling_siglip.py"}
    assert {Path(p).name for p in bundle.component_files["text_config"]} == {"modeling_gemma2.py"}
    assert bundle.component_model_types == {
        "root": "paligemma", "vision_config": "siglip_vision_model", "text_config": "gemma2",
    }
    assert bundle.component_architectures["vision_config"] == "SiglipVisionModel"
    assert bundle.component_architectures["text_config"] == "Gemma2Model"


def test_resolver_deduplicates_shared_component_implementation():
    """Text/vision config types may share one family module.  Recursive lookup
    must not parse the same source twice."""
    cfg = {
        "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text"},
        "vision_config": {"model_type": "qwen3_5_vision"},
    }
    bundle = resolve_source_files(cfg, source="local")
    qwen_files = [p for p in bundle.files if Path(p).name == "modeling_qwen3_5.py"]
    if not qwen_files:
        pytest.skip("Qwen3.5 modeling source is not installed locally")
    assert len(qwen_files) == 1
    assert set(bundle.component_files) == {"root", "text_config", "vision_config"}


def test_component_scoped_evidence_keeps_text_and_vision_oracles_separate():
    """Finding delegated files is insufficient if their classes are unioned again.
    The text block resolver must read Gemma2 while the vision closure reads only
    SigLIP for the same composite model, with qualified provenance on both."""
    from model_unfolder.evidence import conformance as conf
    from model_unfolder.evidence.forward_ops import extract_forward_ops

    cfg = {
        "model_type": "paligemma",
        "vision_config": {"model_type": "siglip_vision_model"},
        "text_config": {"model_type": "gemma2"},
    }
    bundle = resolve_source_files(cfg, source="local")
    required = {"root", "vision_config", "text_config"}
    if not required <= set(bundle.component_files):
        pytest.skip("PaliGemma delegated component sources are not installed locally")

    text_component, text_files = conf._component_source(bundle, "text")
    text_ops = extract_forward_ops(text_files, component=text_component)
    text_code = resolve_view_code("paligemma", "block", {}, text_ops, load_conformance_map())
    assert text_code is not None
    assert text_code.component == "text_config"
    assert Path(text_code.source_file).name == "modeling_gemma2.py"

    from model_unfolder.evidence.component_tower import \
        recursive_component_mechanisms
    from model_unfolder.evidence.program_index import build_program_index
    mechanisms = recursive_component_mechanisms(
        build_program_index(bundle), bundle, config_document=cfg,
        config_selector=conf._exact_document_selector(cfg))
    vision = [item for item in mechanisms.towers
              if item.component.component_key == "vision_config"]
    assert vision, "SigLIP vision tower did not resolve"
    assert all(Path(item.stage_symbol.source.canonical_path).name
               == "modeling_siglip.py" for item in vision)


def test_shared_source_file_is_rooted_at_each_components_auto_model():
    """Qwen text and vision classes share one modeling file.  File ownership
    alone cannot separate them; the AutoModel roots must lead to distinct block
    classes before role closures are built."""
    from model_unfolder.evidence import conformance as conf
    from model_unfolder.evidence.transitive import build_registry

    cfg = {
        "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text"},
        "vision_config": {"model_type": "qwen3_5_vision"},
    }
    bundle = resolve_source_files(cfg, source="local")
    if not bundle.component_files:
        pytest.skip("Qwen3.5 modeling source is not installed locally")
    expected = {
        "text": ("Qwen3_5TextModel", ["Qwen3_5DecoderLayer"]),
        "vision": ("Qwen3_5VisionModel", ["Qwen3_5VisionBlock"]),
    }
    for domain, (architecture, block_classes) in expected.items():
        component, files = conf._component_source(bundle, domain)
        registry = build_registry(files, component=component)
        assert bundle.component_architectures[component] == architecture
        assert conf._component_block_classes(registry, architecture) == block_classes


def test_conformance_citations_not_stale():
    """Every `since` citation token still appears in its cited forward() — so a
    silent upstream rename can't rot the allow-list."""
    ir = mu.unfold(FLUX).to_ir()
    stale = [p.message for p in check_model_conformance(FLUX, ir) if p.kind == "stale"]
    assert not stale, stale


# --------------------------------------------------------------------------
# Stage 4 — render every variant (no silent dominant-only collapse)
# --------------------------------------------------------------------------

def test_heterogeneous_denoiser_renders_every_variant():
    """A multi-block-type denoiser (Flux: dual-stream + single-stream) must render
    EVERY variant's architecture, not collapse to the dominant — so non-dominant
    blocks are drillable and enter the image surface. Pins Fix 4 / the invisibility
    root cause."""
    from model_unfolder.renderers.html.metadata import _make_info
    ir = mu.unfold(FLUX).to_ir()
    n_groups = len(_make_info(ir)["groups"])
    assert n_groups >= 2, f"expected Flux dual+single groups, got {n_groups}"
    html = mu.unfold(FLUX).to_html(standalone=True)
    n_arch = html.count('class="uf-arch-variant uf-arch-variant-')
    assert n_arch >= n_groups, (
        f"{n_arch} architecture variants rendered for {n_groups} block-type groups "
        "— a non-dominant variant collapsed (invisible).")


# --------------------------------------------------------------------------
# Indirect class construction must not read as a FABRICATED op (the op-conf
# false-positive class: Snowflake's MIXTRAL_ATTENTION_CLASSES[...] registry, and
# the "Moe"-spelled MoE blocks of OLMoE / Qwen3-MoE that case-sensitivity missed).
# --------------------------------------------------------------------------

def test_role_mapping_is_case_insensitive():
    from model_unfolder.evidence.forward_ops import _role_of
    # MoE block classes spell it "Moe", not "MoE" — must still type as the FFN family
    # (else their MoE field goes untyped and op-conformance falsely flags the drawn FFN).
    assert _role_of("OlmoeSparseMoeBlock") == "ffn"
    assert _role_of("Qwen3MoeSparseMoeBlock") == "ffn"
    # an ALL-CAPS class registry name still reads as attention.
    assert _role_of("MIXTRAL_ATTENTION_CLASSES") == "attention"


def test_call_name_resolves_registry_subscript_construction():
    import ast
    from model_unfolder.evidence.ast_scanner import _call_name
    # `self.self_attn = MIXTRAL_ATTENTION_CLASSES[impl](config)` — the constructed
    # func is a Subscript; it must resolve to the registry base name so the field
    # gets TYPED (not None, which would drop the attention op).
    call = ast.parse("MIXTRAL_ATTENTION_CLASSES[impl](config)", mode="eval").body
    assert _call_name(call.func) == "MIXTRAL_ATTENTION_CLASSES"


def test_literal_attention_class_map_preserves_candidates_and_uses_eager_default(tmp_path):
    """Keep every registry candidate for semantic checks while the generic
    transitive path follows the library's eager default."""
    from model_unfolder.evidence.transitive import build_registry, transitive_closure
    from model_unfolder.everchanging import load_conformance_transitive

    source = tmp_path / "modeling_dispatch.py"
    source.write_text(
        "class EagerAttention:\n"
        "    def forward(self, x):\n        return apply_rotary_pos_emb(x)\n"
        "class FlashAttention:\n"
        "    def forward(self, x):\n        return apply_rotary_pos_emb(x)\n"
        "ATTENTION_CLASSES = {'eager': EagerAttention, 'flash_attention_2': FlashAttention}\n"
        "class Block:\n"
        "    def __init__(self, config):\n"
        "        self.attn = ATTENTION_CLASSES[config._attn_implementation](config)\n"
        "    def forward(self, x):\n        return self.attn(x)\n"
    )
    registry = build_registry([str(source)])
    info = registry["Block"]
    assert info.field_type_candidates["attn"] == {"EagerAttention", "FlashAttention"}
    assert info.field_type_dispatch["attn"]["eager"] == "EagerAttention"
    _ops, tokens = transitive_closure("Block", registry, load_conformance_transitive())
    assert "apply_rotary_pos_emb" in tokens


def test_indirect_construction_yields_real_ops_not_fabrications(tmp_path):
    """End-to-end: a layer that builds attention via a class REGISTRY and a MoE
    FFN whose class spells "Moe" must expose BOTH ops — so op-conformance does not
    flag the diagram's attention/ffn as fabricated. Locks the FP class generally,
    via the code shape (registry subscript + case-insensitive role), no model name."""
    src = (
        "import torch.nn as nn\n"
        "ATTENTION_CLASSES = {'eager': object}\n"
        "class MyDecoderLayer(nn.Module):\n"
        "    def __init__(self, config):\n"
        "        super().__init__()\n"
        "        self.input_layernorm = nn.LayerNorm(8)\n"
        "        self.self_attn = ATTENTION_CLASSES[config._attn_implementation](config)\n"
        "        self.mlp = MyModelSparseMoeBlock(config)\n"
        "    def forward(self, x):\n"
        "        x = x + self.self_attn(self.input_layernorm(x))\n"
        "        x = x + self.mlp(x)\n"
        "        return x\n"
    )
    f = tmp_path / "modeling_my.py"
    f.write_text(src)
    ops = extract_forward_ops([str(f)])["MyDecoderLayer"]
    assert "attention" in ops.op_kinds, ops.op_kinds   # registry-built attention is REAL
    assert "ffn" in ops.op_kinds, ops.op_kinds          # "Moe"-spelled block reads as ffn


# --------------------------------------------------------------------------
# Diffusion FFN activation/gating read from the SOURCE — no per-model table.
# (T3: the gated==None pale-FFN class. The fact lives in the block's FFN
# construction, never the config.)
# --------------------------------------------------------------------------

def test_diffusion_source_resolves_for_dit_named_classes():
    """Every exact installed Diffusers model class resolves without markers.

    This pins both directions: unfamiliar DiT/UNet spellings resolve because
    the class definition exists, while a Transformers class cannot qualify by
    a suggestive or familiar name.
    """
    from model_unfolder.evidence.sources import (
        _installed_diffusers_model_class_file,
    )
    assert _installed_diffusers_model_class_file("HunyuanDiT2DModel")
    assert _installed_diffusers_model_class_file("LuminaNextDiT2DModel")
    assert _installed_diffusers_model_class_file("FluxTransformer2DModel")
    assert _installed_diffusers_model_class_file("StableCascadeUNet")
    assert not _installed_diffusers_model_class_file("LlamaForCausalLM")


def test_diffusor_class_defaults_mechanism_is_eradicated():
    """The per-model diffusor class_defaults MECHANISM is gone — loader, dead YAML
    data file and every ``_class_default`` call site are deleted, not merely emptied.
    Every architectural fact (qk_norm, ffn activation/kind, rope/axial dims, gate
    dialect, single-stream fusion, attn kind, cross-attn norm) is read from the
    modeling SOURCE, never tabulated by class name. This is the regrowth guard:
    re-introducing the table is a deliberate, reviewable act — not a silent one."""
    import importlib
    import pathlib

    from model_unfolder import everchanging

    assert not hasattr(everchanging, "load_diffusion_class_defaults"), (
        "the class-name fact-table loader was re-introduced — derive from code instead")

    parser_src = pathlib.Path(
        importlib.import_module("model_unfolder.adapters.diffusor.parser").__file__
    ).read_text()
    assert "_class_default" not in parser_src, (
        "a `_class_default` class-name fallback was re-introduced in the diffusor parser")

    yaml_path = (pathlib.Path(everchanging.__file__).parent
                 / "diffusor" / "class_defaults.yaml")
    assert not yaml_path.exists(), (
        f"{yaml_path.name} is a dead regrowth point with no loader — delete it")


# ---------------------------------------------------------------------------
def test_dormant_config_gated_op_is_not_required_but_an_active_one_is(tmp_path):
    """An op the code performs ONLY inside a positive config-gated ``if`` branch
    (PLE's ``hidden_states * per_layer_input`` under ``if self.flag:``) is not
    required of a diagram when the gate field is present-and-falsy in config — the
    same predicate the parser draws by. With the gate truthy it is still required;
    an unconditional op of the same kind is always required. Fixes the gemma-4
    false ``gate_mul`` while never hiding a real, active miss."""
    from model_unfolder.evidence.forward_ops import extract_forward_ops
    from model_unfolder.evidence.conformance import diff_conformance
    from model_unfolder.everchanging import load_conformance_abstractions

    gated = (
        "class GatedBlock:\n"
        "    def __init__(self):\n"
        "        self.attn = FooAttention(8)\n"
        "        self.mlp = FooMLP(8)\n"
        "    def forward(self, x):\n"
        "        x = self.attn(x)\n"
        "        x = self.mlp(x)\n"
        "        if self.flag:\n"
        "            x = x * gate\n"          # gate_mul ONLY under the config gate
        "        return x\n"
    )
    f = tmp_path / "m_gated.py"; f.write_text(gated)
    ops = extract_forward_ops([str(f)])["GatedBlock"]
    assert "gate_mul" in ops.op_kinds                      # the op is present in code
    assert "gate_mul" in ops.gated_op_kinds                # but only as a gated occurrence
    assert frozenset({"flag"}) in ops.gated_op_kinds["gate_mul"]

    ab = load_conformance_abstractions()
    drawn = frozenset({"attention", "ffn"})               # diagram correctly omits the gate
    # gate OFF (present, falsy) -> not required
    off = diff_conformance(drawn, ops, "x", "block", ab, cfg={"flag": 0})
    assert [p.op for p in off if p.kind == "missing"] == []
    # gate ON (present, truthy) -> still required
    on = diff_conformance(drawn, ops, "x", "block", ab, cfg={"flag": 1})
    assert "gate_mul" in [p.op for p in on if p.kind == "missing"]
    # no config at all -> conservative, stays required (never hide a real miss)
    blind = diff_conformance(drawn, ops, "x", "block", ab, cfg=None)
    assert "gate_mul" in [p.op for p in blind if p.kind == "missing"]


def test_unconditional_op_is_never_treated_as_gated(tmp_path):
    """An op that also occurs unconditionally must never be suppressed even if it
    ALSO appears under a config gate — its unconditional path always runs."""
    from model_unfolder.evidence.forward_ops import extract_forward_ops
    src = (
        "class B:\n"
        "    def __init__(self):\n"
        "        self.attn = FooAttention(8)\n"
        "    def forward(self, x):\n"
        "        x = x * scale\n"            # unconditional gate_mul
        "        if self.flag:\n"
        "            x = x * other\n"        # also gated, but the op already runs above
        "        return x\n"
    )
    f = tmp_path / "m_uncond.py"; f.write_text(src)
    ops = extract_forward_ops([str(f)])["B"]
    assert "gate_mul" in ops.op_kinds
    assert "gate_mul" not in ops.gated_op_kinds            # has an unconditional occurrence


def test_is_not_none_branch_is_not_suppressed_by_zero_config(tmp_path):
    """``0 is not None`` is true.  Treating this predicate as ordinary
    truthiness hides an active op and creates a false PASS."""
    src = (
        "class B:\n"
        "    def forward(self, x):\n"
        "        if self.scale is not None:\n"
        "            x = x * gate\n"
        "        return x\n"
    )
    f = tmp_path / "m_not_none.py"; f.write_text(src)
    ops = extract_forward_ops([str(f)])["B"]
    assert "gate_mul" in ops.op_kinds
    assert "gate_mul" not in ops.gated_op_kinds
    probs = diff_conformance(
        frozenset(), ops, "x", "block", load_conformance_abstractions(),
        cfg={"scale": 0},
    )
    assert "gate_mul" in [p.op for p in probs if p.kind == "missing"]


def test_projection_matmul_is_linear_but_qk_matmul_is_not(tmp_path):
    src = (
        "class B:\n"
        "    def forward(self, hidden, query, key):\n"
        "        projected = hidden @ self.gate_up_proj\n"
        "        scores = query @ key.transpose(-1, -2)\n"
        "        return projected, scores\n"
    )
    f = tmp_path / "m_matmul.py"; f.write_text(src)
    ops = extract_forward_ops([str(f)])["B"]
    assert {"linear", "dot_product"} <= ops.op_kinds


def test_index_copy_is_not_an_additive_residual(tmp_path):
    src = (
        "class B:\n"
        "    def forward(self, target, index, source):\n"
        "        return target.index_copy_(0, index, source)\n"
    )
    f = tmp_path / "m_index_copy.py"; f.write_text(src)
    ops = extract_forward_ops([str(f)])["B"]
    assert "residual_add" not in ops.op_kinds


# ===========================================================================
# RECURSIVE nested-drill conformance — diff each leaf-compute drill (attention /
# FFN / expert internals) against the TRANSITIVE forward() closure of its backing
# sub-module (following sdpa / rotary / the diffusers processor / the FeedForward
# ModuleList).  One altitude below the per-layer check above.
# ===========================================================================

from model_unfolder.evidence import check_nested_conformance
from model_unfolder.evidence.transitive import build_registry, transitive_closure
from model_unfolder.everchanging import load_conformance_transitive
def _render_log(cfg):
    from model_unfolder.renderers.html.render_context import RenderContext, activate_render_context

    diagram = mu.unfold(cfg)
    context = RenderContext()
    with activate_render_context(context):
        diagram.to_html(standalone=True)
    # Preserve the corpus's historical view-level union until its pinned broad
    # attribution debt is migrated, while exercising exact provenance for the
    # newly source-bound supporting text encoders. Production Sable consumes
    # the full typed event stream directly.
    return [
        event if (
            event.component.startswith("text_encoder")
            or (event.block_path and event.block_path[-1].startswith("encoder_")
                and event.view == "ffn")
        ) else event.legacy_tuple()
        for event in context.events
    ]


_EXPECTED_NESTED_UNRESOLVED = {
    # Several genuine source variants render one shared drill; exact variant
    # provenance must be carried by the render log before these can be proved.
    # Expert drills are intentionally absent here: their storage/mechanism is
    # unknown, so they render the explicit one-node ``opaque`` view and make no
    # decomposed compute claim for conformance to resolve.
    # The attention drill is now source-resolved: the tuple/frozenset
    # ForwardOps record-shape fix lets the conformance reader see the real
    # rotary/signature evidence instead of silently abstaining.  FFN/router
    # attribution remains genuinely unresolved.
    # U4-C: config-only MoE/FFN identity no longer authors a drill, so there is
    # no fabricated nested compute region to excuse.
    "self_cond": set(),
    # MusicGen's nested decoder is now joined to its exact component/config
    # path and exact inline FFN owner; the former audio/ffn unresolved pin was
    # retired by the shared U3 FFN result.
    # (UNet block factories resolved 2026-07-03: the string factory
    # ``get_down_block`` is followed generally — no unet pin remains.)
}


@pytest.mark.parametrize("name", list(tc.CORPUS))
def test_nested_conformance_over_corpus_is_clean_or_explicitly_unresolved(name):
    """No corpus drill may silently skip.  Exact matches are clean; known gaps are
    pinned as blocking ``unresolved`` results until provenance is implemented."""
    cfg = tc.CORPUS[name]
    problems = check_nested_conformance(cfg, _render_log(cfg))
    mismatches = [p for p in problems if p.kind != "unresolved"]
    assert mismatches == [], "\n".join(p.message for p in mismatches)
    actual_unresolved = {p.view.split("/", 1)[1] for p in problems}
    assert actual_unresolved == _EXPECTED_NESTED_UNRESOLVED.get(name, set()), \
        "\n".join(p.message for p in problems)


def test_transitive_closure_follows_sdpa_and_rotary():
    """The attention closure must FOLLOW the delegated compute: the score/softmax
    pair (via the ``attention_interface``/SDPA leaf) and the rotary helper token —
    even though ``LlamaAttention.forward`` itself extracts to only {linear, reshape}."""
    bundle = resolve_source_files({"model_type": "llama"}, source="local")
    if not bundle.files:
        pytest.skip("transformers llama source not installed")
    reg = build_registry(bundle.files)
    vocab = load_conformance_transitive()
    ops, tokens = transitive_closure("LlamaAttention", reg, vocab)
    assert {"dot_product", "activation", "linear"} <= ops      # sdpa compute followed
    assert any("rotary" in t.lower() for t in tokens)          # rope helper reachable
    # The eager helper's `* scaling` / `+ mask` noise must NOT leak as gate/residual.
    assert "gate_mul" not in ops and "residual_add" not in ops


def test_transitive_closure_follows_self_method_helper():
    """A ``self.route_tokens_to_experts(...)`` self-METHOD (where the router's
    ``torch.topk`` lives, NOT in forward) must be folded into the class op-set —
    the general self-method-following the engine does."""
    bundle = resolve_source_files({"model_type": "deepseek_v3"}, source="local")
    if not bundle.files:
        pytest.skip("transformers deepseek_v3 source not installed")
    reg = build_registry(bundle.files)
    vocab = load_conformance_transitive()
    ops, _ = transitive_closure("DeepseekV3MoE", reg, vocab)
    assert "route" in ops          # topk in route_tokens_to_experts, folded in


def test_diffusion_attention_closure_injects_block_processor():
    """A diffusers ``Attention`` delegates to a PROCESSOR built by the PARENT block
    (``Attention(processor=CogVideoXAttnProcessor2_0())``).  The union attention
    closure must inject it so the SDPA compute (and ``apply_rotary_emb``) is seen —
    otherwise the attention drill's rope/softmax reads as fabricated."""
    from model_unfolder.evidence.conformance import (
        _augment_diffusion_files, _block_classes, _resolve_drill_closure)
    bundle = resolve_source_files(_cogvideox_cfg(), source="local")
    if not bundle.files:
        pytest.skip("diffusers cogvideox source not installed")
    reg = build_registry(_augment_diffusion_files(bundle.files))
    vocab = load_conformance_transitive()
    blocks = [b for b in _block_classes(reg) if "CogVideoX" in b]
    assert blocks, "no CogVideoX block class resolved"
    closure = _resolve_drill_closure(blocks, reg, vocab, "attention", "attn")
    assert closure is not None, "no attention sub-module resolved"
    ops, _evidence = closure
    # the diffusers Attention.forward itself is empty (it delegates to the processor);
    # the SDPA compute appears ONLY if the block-supplied processor was injected.
    assert "dot_product" in ops, "block-supplied processor not injected into the closure"


def _cogvideox_cfg():
    return {
        "_class_name": "CogVideoXTransformer3DModel",
        "num_attention_heads": 4, "attention_head_dim": 16, "num_layers": 2,
        "in_channels": 4, "out_channels": 4, "text_embed_dim": 32,
        "time_embed_dim": 32, "sample_width": 8, "sample_height": 8, "sample_frames": 9,
    }


def test_nested_conformance_catches_fabricated_op(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: a drill that draws a compute op its sub-module never does
    (a fabricated ``concat`` in an FFN drill) MUST be flagged."""
    from model_unfolder.evidence import conformance as conf

    # a model whose ffn closure is {linear, activation, gate_mul} (a gated MLP)
    fake_closure = (frozenset({"linear", "activation", "gate_mul"}), "FakeMLP")
    monkeypatch.setattr(conf, "_resolve_drill_closure", lambda *a, **k: fake_closure)
    monkeypatch.setattr(conf, "resolve_source_files",
                        lambda *a, **k: type("B", (), {"files": ("x.py",)})())
    monkeypatch.setattr(conf, "build_registry", lambda *a, **k: {})
    # the drill DRAWS a concat the code never does -> fabricated
    log = [("ffn", frozenset({"linear", "activation", "gate_mul", "concat", "port"}), frozenset())]
    problems = conf.check_nested_conformance({"model_type": "x"}, log)
    assert any(p.kind == "fabricated" and p.op == "concat" for p in problems), \
        [p.message for p in problems]


def test_dense_ffn_drill_not_false_flagged_when_a_sibling_is_gated(monkeypatch):
    """SOUNDNESS REGRESSION: a dense drill is checked against its exact dense MLP,
    not a gated sibling's union.  The sibling can no longer inject ``gate_mul``
    into this view's evidence."""
    from model_unfolder.evidence import conformance as conf
    exact_dense = (frozenset({"linear", "activation"}), "DenseVisionMLP")
    monkeypatch.setattr(conf, "_resolve_drill_closure", lambda *a, **k: exact_dense)
    monkeypatch.setattr(conf, "resolve_source_files",
                        lambda *a, **k: type("B", (), {"files": ("x.py",)})())
    monkeypatch.setattr(conf, "build_registry", lambda *a, **k: {})
    log = [("ffn", frozenset({"linear", "activation", "port"}), frozenset())]   # dense drawn
    assert conf.check_nested_conformance({"model_type": "x"}, log) == []


def test_opaque_drill_makes_no_claim(monkeypatch):
    """An honest-unknown OPAQUE drill (the parser could not decompose the
    sub-module, so it drew one ``opaque`` block) must NOT be held to fabrication or
    salient-omission — Sana's GLUMBConv FFN is drawn opaque and must stay clean."""
    from model_unfolder.evidence import conformance as conf
    fake_closure = (frozenset({"linear", "activation", "gate_mul"}), "FakeMLP")
    monkeypatch.setattr(conf, "_resolve_drill_closure", lambda *a, **k: fake_closure)
    monkeypatch.setattr(conf, "resolve_source_files",
                        lambda *a, **k: type("B", (), {"files": ("x.py",)})())
    monkeypatch.setattr(conf, "build_registry", lambda *a, **k: {})
    log = [("ffn", frozenset({"opaque", "port"}), frozenset())]
    assert conf.check_nested_conformance({"model_type": "x"}, log) == []


def test_code_derived_ffn_gating_overrides_rmsnorm_heuristic():
    """The exact selected FFN, not RMSNorm, controls the rendered mechanism."""
    from transformers import AutoConfig
    for mt, expected in (("phi", False), ("llama", True)):
        cfg = AutoConfig.for_model(mt).to_dict()
        bundle = resolve_source_files(cfg, source="local")
        if not bundle.files:
            pytest.skip(f"transformers {mt} source not installed")
        assert mu.unfold(cfg).to_ir()["layers"][0]["ffn"]["gated"] is expected


def test_bloom_dormant_tensor_parallel_multiply_is_not_an_ffn_gate():
    """BLOOM's disabled slow path multiplies slice indices/weights; that is not
    gate*up and must not turn its dense GELU MLP into a gated SiLU diagram."""
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.ffn_mechanism import (
        decoder_ffn_mechanism_for_path,
    )

    cfg = AutoConfig.for_model("bloom").to_dict()
    context = ParseContext.build(cfg)
    if not context.source_bundle.files:
        pytest.skip("transformers BLOOM source not installed")
    exact = decoder_ffn_mechanism_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert exact.status == "resolved", exact.failures
    assert exact.value.activation == "gelu"
    ffn = mu.unfold(cfg).to_ir()["layers"][0]["ffn"]
    assert ffn["gated"] is False
    assert ffn["activation"] == "gelu"


def test_disabled_optional_cross_attention_param_is_not_a_missing_text_rail():
    """XGLM keeps optional encoder states in the signature while config disables
    construction of encoder_attn; a signature-only net must not fabricate it."""
    from transformers import AutoConfig
    from model_unfolder.evidence import check_wiring_conformance

    cfg = AutoConfig.for_model("xglm").to_dict()
    assert cfg["add_cross_attention"] is False
    ir = mu.unfold(cfg).to_ir()
    assert check_wiring_conformance(cfg, ir) == []
    report = mu.sable(cfg, render_images=False)
    nested = next(check for check in report.checks if check.name == "nested_conformance")
    assert nested.findings == []


# --- selection (router / indexer) + composite (moe container / vision-encoder /
#     mtp block) drill conformance, and the YAML multi-value-marker regression ---

def test_drill_role_markers_parse_all_multi_values():
    """REGRESSION: a flow-list `[role=a,b, …]` comma-splits the value and drops the
    tail (it once silently lost `attn,mla` -> only `attn`, so the MLA drill was
    NEVER checked). Block style preserves them — assert every multi-value marker
    survives and the three categories classify correctly."""
    from model_unfolder.evidence import conformance as conf
    v = load_conformance_transitive()
    m = v["drill_role_markers"]
    assert "mla" in m["attention"] and "topk" in m["route"]
    assert {"vision-encoder", "mtp-transformer-block"} <= set(m["composite"])
    cat = lambda vk: v["drill_category"].get(conf._drill_role(vk, v), "leaf_compute")
    assert cat("mla") == "leaf_compute"           # was skipped before the fix
    assert cat("moe_router") == "selection" and cat("dsa_indexer") == "selection"
    assert cat("moe") == "composite" and cat("vision-encoder") == "composite"


def test_selection_closure_carries_routing_topk():
    """The selection closure (ffn ∪ route sub-module closures) must carry the
    routing `route` (the top-k, folded from the MoE container's self-method) and the
    gate `linear` — i.e. it is genuinely exercised, not an empty no-op."""
    from model_unfolder.evidence import conformance as conf
    bundle = resolve_source_files({"model_type": "deepseek_v3"}, source="local")
    if not bundle.files:
        pytest.skip("transformers deepseek_v3 source not installed")
    reg = build_registry(conf._augment_diffusion_files(bundle.files))
    blocks = conf._block_classes(reg)
    closure = conf._resolve_selection_closure(blocks, reg, load_conformance_transitive())
    assert closure is not None, "selection closure did not resolve"
    sel, _evidence = closure
    assert "route" in sel and "linear" in sel


def test_selection_drill_requires_topk(monkeypatch):
    """NEGATIVE CONTROL: a router/indexer drill that omits its top-k (`select`)
    while the routing code DOES route MUST flag a salient `missing route` — the
    Mixtral-style 'router drawn without its selection step' bug."""
    from model_unfolder.evidence import conformance as conf
    v = load_conformance_transitive()
    ab = load_conformance_abstractions()
    sel = frozenset({"route", "linear", "gate_mul"})
    probs = conf._diff_selection("x", "moe_router", "route",
                                 frozenset({"linear", "gate_mul", "port"}), sel, v, ab)
    assert any(p.kind == "missing" and p.op == "route" for p in probs)


def test_selection_drill_drops_renormalize_and_bias_presentation():
    """The renormalize box (`norm`) and the e_score bias (`embedding`) are config-
    driven presentation, NOT compute ops — a selection drill drawing them against a
    closure that has no nn-norm / no embedding must stay clean (not fabricated)."""
    from model_unfolder.evidence import conformance as conf
    v = load_conformance_transitive()
    ab = load_conformance_abstractions()
    sel = frozenset({"route", "linear", "gate_mul"})
    drawn = frozenset({"select", "linear", "gate_mul", "norm", "embedding", "port"})
    assert conf._diff_selection("x", "moe_router", "route", drawn, sel, v, ab) == []


def test_composite_catches_impossible_container_combo():
    """NEGATIVE CONTROL: a composite drawing containers no single block has
    together (attention + routing, when the model has an attention block and a
    separate MoE block but none with both) MUST flag the orphan container."""
    from model_unfolder.evidence import conformance as conf
    v = load_conformance_transitive()
    ab = load_conformance_abstractions()
    blocks = [frozenset({"attention", "ffn", "norm", "residual_add"}),
              frozenset({"ffn", "route", "residual_add", "linear", "gate_mul"})]
    probs = conf._diff_composite("x", "moe", frozenset({"attention", "router", "port"}), blocks, v, ab)
    assert any(p.kind == "fabricated" and p.op == "route" for p in probs)
    # the real moe container (expert + router + combine) is clean
    assert conf._diff_composite("x", "moe",
                                frozenset({"expert", "router", "residual_add", "port"}), blocks, v, ab) == []


def test_moe_expert_combine_index_add_is_residual():
    """A fused MoE combine `final_hidden_states.index_add_(...)` (Mixtral/Qwen3/
    Olmoe) is the ⊕ the moe drill draws — it must be detected as `residual_add`,
    not missed (it is a scatter, not a `+`)."""
    from model_unfolder.evidence.transitive import build_registry, transitive_closure
    bundle = resolve_source_files({"model_type": "mixtral"}, source="local")
    if not bundle.files:
        pytest.skip("transformers mixtral source not installed")
    reg = build_registry(bundle.files)
    vocab = load_conformance_transitive()
    # the MoE block's transitive closure (experts included) carries the combine ⊕
    moe = next((n for n in reg if "SparseMoe" in n or n.endswith("MoE") or n.endswith("Moe")), None)
    assert moe is not None
    ops, _ = transitive_closure(moe, reg, vocab)
    assert "residual_add" in ops


@pytest.mark.parametrize("mt", ["mixtral", "qwen3_moe", "olmoe", "deepseek_v3"])
def test_real_moe_models_nested_clean(mt):
    """Real MoE models (selection + composite + expert leaf drills all active) are
    clean — the net resolves the router/indexer/container against real code."""
    cfg = {"model_type": mt, "hidden_size": 128, "num_hidden_layers": 2,
           "num_attention_heads": 8, "num_key_value_heads": 2, "intermediate_size": 256,
           "moe_intermediate_size": 128, "vocab_size": 1000, "num_experts": 8,
           "num_local_experts": 8, "num_experts_per_tok": 2}
    if not resolve_source_files(cfg, source="local").files:
        pytest.skip(f"{mt} source not installed")
    problems = check_nested_conformance(cfg, _render_log(cfg))
    assert problems == [], "\n".join(p.message for p in problems)


def test_parameter_matmul_expert_uses_exact_storage_proof_for_linear():
    """DBRX stores three repeated expert Parameters and applies them with
    ``matmul``.  A global dot-product→linear equivalence would be unsound for
    attention; the exact routed-storage proof must close only this expert
    occurrence, and every nested expert drill must remain conformance-clean.
    """
    fixture = Path(__file__).parent / "sable_test_corpus" / "dbrx-base.json"
    cfg = __import__("json").loads(fixture.read_text())["config"]
    if not resolve_source_files(cfg, source="local").files:
        pytest.skip("transformers DBRX source not installed")
    ir = mu.config_to_ir(cfg).to_dict()
    facts = (ir.get("extras") or {}).get("fact_provenance") or {}
    problems = check_nested_conformance(
        cfg, _render_log(cfg), fact_rows=facts)
    fabricated = [
        problem for problem in problems
        if problem.kind == "fabricated" and "expert" in problem.view
    ]
    assert fabricated == [], "\n".join(p.message for p in fabricated)


# ---------------------------------------------------------------------------
# storage / bookend fact conformance (fused-vs-split, embedding-stage norm)
# ---------------------------------------------------------------------------

def _fact_problems(cfg, ir):
    from model_unfolder.evidence.conformance import check_fact_conformance
    return check_fact_conformance(cfg, ir)


def test_storage_conformance_flags_fused_qkv_drawn_split():
    """Falcon stores one fused ``query_key_value`` projection.  The honest parse
    carries that fact and is clean; an IR that lost the fact (the exact plumbing
    drop this net exists for — the serializer omitted the spec field once) is
    flagged.  Symmetric direction: fabricating fused on a split model (Llama)."""
    import copy
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("falcon").to_dict()
    ir = mu.config_to_ir(cfg).to_dict()
    assert ir["layers"][0]["attention"]["projection_mode"] == "fused_qkv"
    assert not [p for p in _fact_problems(cfg, ir) if p.kind == "wrong_storage"]

    tampered = copy.deepcopy(ir)
    tampered["layers"][0]["attention"]["projection_mode"] = None
    assert any(p.kind == "wrong_storage" and "stored FUSED" in p.op
               for p in _fact_problems(cfg, tampered))

    split_cfg = AutoConfig.for_model("llama").to_dict()
    split_ir = mu.config_to_ir(split_cfg).to_dict()
    assert not [p for p in _fact_problems(split_cfg, split_ir) if p.kind == "wrong_storage"]
    fabricated = copy.deepcopy(split_ir)
    fabricated["layers"][0]["attention"]["projection_mode"] = "fused_qkv"
    assert any(p.kind == "wrong_storage" and "stored SPLIT" in p.op
               for p in _fact_problems(split_cfg, fabricated))


def test_storage_conformance_flags_fused_experts_drawn_split():
    """gpt-oss stores stacked fused ``gate_up_proj`` experts; dropping the drawn
    fact is flagged, and the honest parse is clean."""
    import copy
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("gpt_oss").to_dict()
    ir = mu.config_to_ir(cfg).to_dict()
    moe_layers = [l for l in ir["layers"] if (l.get("ffn") or {}).get("num_experts")]
    assert moe_layers and moe_layers[0]["ffn"]["expert_projection_mode"] == "fused_gate_up"
    assert not [p for p in _fact_problems(cfg, ir) if p.kind == "wrong_storage"]

    tampered = copy.deepcopy(ir)
    for layer in tampered["layers"]:
        if (layer.get("ffn") or {}).get("num_experts"):
            layer["ffn"]["expert_projection_mode"] = None
    assert any(p.kind == "wrong_storage" and "expert projection storage" in p.op
               for p in _fact_problems(cfg, tampered))


def test_bookend_conformance_flags_missing_positive_but_never_invents_absence():
    """BLOOM normalizes the word-embedding output (a drawn bookend): removing the
    drawn block is a MISSING bookend.  The execution substrate is open, so
    injecting one into Llama is not called fabricated until a future
    reader-specific completeness proof can establish absence."""
    import copy
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("bloom").to_dict()
    ir = mu.config_to_ir(cfg).to_dict()
    blocks = ir["extras"]["render"]["model_blocks"]
    assert any(b.get("id") == "embed_norm" for b in blocks)
    assert not [p for p in _fact_problems(cfg, ir)
                if p.kind in ("missing_bookend", "fabricated_bookend")]

    tampered = copy.deepcopy(ir)
    tampered["extras"]["render"]["model_blocks"] = [
        b for b in blocks if b.get("id") != "embed_norm"]
    assert any(p.kind == "missing_bookend" for p in _fact_problems(cfg, tampered))

    clean_cfg = AutoConfig.for_model("llama").to_dict()
    clean_ir = mu.config_to_ir(clean_cfg).to_dict()
    fabricated = copy.deepcopy(clean_ir)
    fabricated["extras"]["render"]["model_blocks"] = (
        list(fabricated["extras"]["render"]["model_blocks"])
        + [{"id": "embed_norm", "role": "norm", "kind": "norm"}])
    assert not any(p.kind == "fabricated_bookend"
                   for p in _fact_problems(clean_cfg, fabricated))


def test_component_storage_conformance_flags_encoder_tower_divergence():
    """The storage net runs per pipeline SLOT: claiming a fused QKV on FLUX's
    CLIP tower (whose source stores split q/k/v projections) is flagged with
    the slot as the component; the honest parse is clean."""
    import copy
    import json
    from model_unfolder.evidence.conformance import check_fact_conformance

    from model_unfolder.sable import DEFAULT_CORPUS
    cfg = json.loads((DEFAULT_CORPUS /
                      "fluxtransformer2dmodel.json").read_text())["config"]
    ir = mu.config_to_ir(cfg).to_dict()
    assert not [p for p in check_fact_conformance(cfg, ir) if p.kind == "wrong_storage"]

    tampered = copy.deepcopy(ir)
    for block in tampered["extras"]["render"]["loop_blocks"]:
        if isinstance(block, dict) and block.get("id") == "encoder_0":
            block["detail"]["sub_model"]["groups"][0]["attention"]["projection_mode"] = "fused_qkv"
    flagged = [p for p in check_fact_conformance(cfg, tampered) if p.kind == "wrong_storage"]
    assert flagged and flagged[0].source_component == "text_encoder"
    assert "stored SPLIT, drawn fused" in flagged[0].op
