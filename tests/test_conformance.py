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

import pytest

import model_unfolder as mu
from model_unfolder.evidence import check_model_conformance, extract_forward_ops
from model_unfolder.evidence.conformance import diff_conformance, resolve_view_code
from model_unfolder.evidence.sources import resolve_source_files
from model_unfolder.everchanging import load_conformance_abstractions, load_conformance_map

from tests.test_diffusion import FLUX, PIXART
from tests import test_coverage as tc


def _flux_forward_ops():
    bundle = resolve_source_files(FLUX, source="local")
    if not bundle.files:
        pytest.skip("diffusers Flux modeling source not installed locally")
    return extract_forward_ops(bundle.files)


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


def test_negative_control_end_to_end_pipeline_catches_buggy_render():
    """The FULL path (parser → IR → conformance) catches the bug: mutate Flux's
    single-stream group back to the buggy parallel-sum (no concat/gate) and the
    net flags both — classified by the parser's variant tag, so the mis-render
    can't dodge the check by looking like a plain block."""
    ir = mu.unfold(FLUX).to_ir()
    mutated = False
    for layer in ir["layers"]:
        if "concat" in {b.get("kind") for b in (layer.get("blocks") or [])}:
            layer["blocks"] = [{"id": "rms1", "kind": "norm"}, {"id": "attn", "kind": "attention"},
                               {"id": "ffn", "kind": "ffn"}, {"id": "add1", "kind": "residual_add"}]
            mutated = True
    assert mutated, "no single-stream group to mutate — Flux fixture changed?"
    missing = {p.op for p in check_model_conformance(FLUX, ir) if p.kind == "missing"}
    assert {"concat", "gate_mul"} <= missing, missing


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

def test_diffusion_ffn_activation_from_construction_kwarg(tmp_path):
    """A block that builds `FeedForward(activation_fn="geglu")` resolves to that
    activation from the modeling source — the CogView4 shape (config is silent)."""
    from model_unfolder.evidence.patterns import diffusion_ffn_activation_from_files
    src = (
        "class MyTransformerBlock:\n"
        "    def __init__(self, dim):\n"
        "        self.ff = FeedForward(dim=dim, activation_fn='geglu')\n"
    )
    f = tmp_path / "modeling_kwarg.py"
    f.write_text(src)
    assert diffusion_ffn_activation_from_files([str(f)]) == "geglu"


def test_diffusion_ffn_activation_from_named_swiglu_class(tmp_path):
    """A block that builds a structurally-gated SwiGLU FFN class (w1·w3·silu gate)
    resolves to "swiglu" from the class body — the HiDream/Lumina shape, where the
    activation is in the class structure, not a kwarg. No name token needed."""
    from model_unfolder.evidence.patterns import diffusion_ffn_activation_from_files
    src = (
        "class MyFusedFFN:\n"
        "    def __init__(self, dim, hidden):\n"
        "        self.w1 = nn.Linear(dim, hidden)\n"
        "        self.w2 = nn.Linear(hidden, dim)\n"
        "        self.w3 = nn.Linear(dim, hidden)\n"
        "    def forward(self, x):\n"
        "        return self.w2(F.silu(self.w1(x)) * self.w3(x))\n"
        "class MyTransformerBlock:\n"
        "    def __init__(self, dim):\n"
        "        self.feed_forward = MyFusedFFN(dim, 4 * dim)\n"
    )
    f = tmp_path / "modeling_struct.py"
    f.write_text(src)
    assert diffusion_ffn_activation_from_files([str(f)]) == "swiglu"


def test_diffusion_source_resolves_for_dit_named_classes():
    """A DiT denoiser named with "DiT" (not "Transformer"/"UNet") must still be
    recognised as a diffusion class so its installed source resolves — else
    conformance + the code-derived FFN silently skip (the HunyuanDiT/Lumina-Next
    MISSING-oracle-on-installed-source bug). Detected by the general marker
    vocabulary, never a hand-picked substring."""
    from model_unfolder.evidence.sources import _looks_like_diffusion_class
    assert _looks_like_diffusion_class("HunyuanDiT2DModel")
    assert _looks_like_diffusion_class("LuminaNextDiT2DModel")
    assert _looks_like_diffusion_class("FluxTransformer2DModel")
    assert _looks_like_diffusion_class("StableCascadeUNet")
    assert not _looks_like_diffusion_class("LlamaForCausalLM")


def test_rope_read_from_source_uses_fact_conformance_evidence(tmp_path):
    """RoPE presence is read from the SAME forward rotary evidence fact-conformance
    reads (so the parser asserts rope exactly when the net would flag its absence as
    a fabricated NoPE). A block whose forward applies rotary => rope; one with only
    learned positions => no rope. Fixes the Allegro/Lumina fabricated-NoPE class."""
    from model_unfolder.evidence.patterns import diffusion_rope_from_files
    rope = (
        "class RopeBlock:\n"
        "    def forward(self, hidden_states, image_rotary_emb=None):\n"
        "        return apply_rotary_emb(hidden_states, image_rotary_emb)\n"
    )
    learned = (
        "class LearnedPosBlock:\n"
        "    def forward(self, hidden_states):\n"
        "        return self.attn(hidden_states + self.pos_embed)\n"
    )
    fr = tmp_path / "modeling_rope.py"; fr.write_text(rope)
    fl = tmp_path / "modeling_learned.py"; fl.write_text(learned)
    assert diffusion_rope_from_files([str(fr)]) is True
    assert diffusion_rope_from_files([str(fl)]) is False


def test_attn_kind_read_from_source_linear_processor(tmp_path):
    """The attention ALGORITHM is read from the SAME *LinearAttn* processor signal
    fact-conformance reads (init_class_refs): a block constructing a LinearAttn
    processor => "linear"; a plain softmax block => None (caller's MHA default).
    Sana's class lives in a per-model table no longer — this is the rail."""
    from model_unfolder.evidence.patterns import diffusion_attn_kind_from_files
    linear = (
        "class MyBlock:\n"
        "    def __init__(self, dim):\n"
        "        self.attn = Attention(dim, processor=MyLinearAttnProcessor())\n"
        "    def forward(self, x):\n"
        "        return self.attn(x)\n"
    )
    softmax = (
        "class MyBlock:\n"
        "    def __init__(self, dim):\n"
        "        self.attn = Attention(dim)\n"
        "    def forward(self, x):\n"
        "        return self.attn(x)\n"
    )
    fl = tmp_path / "modeling_lin.py"; fl.write_text(linear)
    fs = tmp_path / "modeling_soft.py"; fs.write_text(softmax)
    assert diffusion_attn_kind_from_files([str(fl)]) == "linear"
    assert diffusion_attn_kind_from_files([str(fs)]) is None


# ---------------------------------------------------------------------------
# Decoder-layer MACRO-TOPOLOGY read from the forward() dataflow (code ->
# structure), the general replacement for the layer_topology.yaml model_type
# table. Asserts the GENERAL dataflow-classifier behavior on synthetic source,
# never a single family.
# ---------------------------------------------------------------------------

_PRE_LAYER = (
    "class FooDecoderLayer:\n"
    "    def __init__(self):\n"
    "        self.input_layernorm = RMSNorm(8)\n"
    "        self.post_attention_layernorm = RMSNorm(8)\n"
    "        self.self_attn = FooAttention(8)\n"
    "        self.mlp = FooMLP(8)\n"
    "    def forward(self, x, past_key_values=None):\n"
    "        residual = x\n"
    "        x = self.input_layernorm(x)\n"
    "        x = self.self_attn(x)\n"
    "        x = residual + x\n"
    "        residual = x\n"
    "        x = self.post_attention_layernorm(x)\n"
    "        x = self.mlp(x)\n"
    "        x = residual + x\n"
    "        return x\n"
)
_DOUBLE_LAYER = (
    "class FooDecoderLayer:\n"
    "    def __init__(self):\n"
    "        self.input_layernorm = RMSNorm(8)\n"
    "        self.post_attention_layernorm = RMSNorm(8)\n"
    "        self.pre_feedforward_layernorm = RMSNorm(8)\n"
    "        self.post_feedforward_layernorm = RMSNorm(8)\n"
    "        self.self_attn = FooAttention(8)\n"
    "        self.mlp = FooMLP(8)\n"
    "    def forward(self, x, past_key_values=None):\n"
    "        residual = x\n"
    "        x = self.input_layernorm(x)\n"
    "        x = self.self_attn(x)\n"
    "        x = self.post_attention_layernorm(x)\n"
    "        x = residual + x\n"
    "        residual = x\n"
    "        x = self.pre_feedforward_layernorm(x)\n"
    "        x = self.mlp(x)\n"
    "        x = self.post_feedforward_layernorm(x)\n"
    "        x = residual + x\n"
    "        return x\n"
)
_POST_LAYER = (
    "class FooDecoderLayer:\n"
    "    def __init__(self):\n"
    "        self.post_attention_layernorm = RMSNorm(8)\n"
    "        self.post_feedforward_layernorm = RMSNorm(8)\n"
    "        self.self_attn = FooAttention(8)\n"
    "        self.mlp = FooMLP(8)\n"
    "    def forward(self, x, past_key_values=None):\n"
    "        residual = x\n"
    "        x = self.self_attn(x)\n"
    "        x = self.post_attention_layernorm(x)\n"
    "        x = residual + x\n"
    "        residual = x\n"
    "        x = self.mlp(x)\n"
    "        x = self.post_feedforward_layernorm(x)\n"
    "        x = residual + x\n"
    "        return x\n"
)
_PARALLEL_LAYER = (
    "class FooDecoderLayer:\n"
    "    def __init__(self):\n"
    "        self.input_layernorm = LayerNorm(8)\n"
    "        self.self_attn = FooAttention(8)\n"
    "        self.mlp = FooMLP(8)\n"
    "    def forward(self, x, past_key_values=None):\n"
    "        residual = x\n"
    "        x = self.input_layernorm(x)\n"
    "        attn_out = self.self_attn(x)\n"
    "        mlp_out = self.mlp(x)\n"
    "        x = residual + attn_out + mlp_out\n"
    "        return x\n"
)


def _topo(tmp_path, src):
    from model_unfolder.evidence.patterns import decoder_layer_topology_from_files
    f = tmp_path / "modeling_topo.py"
    f.write_text(src)
    return decoder_layer_topology_from_files([str(f)])


def test_layer_topology_classifies_norm_placement_from_dataflow(tmp_path):
    """norm placement is read from where the norms sit relative to each sublayer in
    the forward() — not a model_type row. norm-before-sublayer => pre, norm-after =>
    post, both => double (sandwich)."""
    assert _topo(tmp_path, _PRE_LAYER)["norm_placement"] == "pre"
    assert _topo(tmp_path, _DOUBLE_LAYER)["norm_placement"] == "double"
    assert _topo(tmp_path, _POST_LAYER)["norm_placement"] == "post"


def test_layer_topology_detects_parallel_residual_from_shared_input(tmp_path):
    """parallel residual is read from the forward: attention and the FFN consumed in
    one residual segment (one norm feeds both, one combined add) => parallel; the
    sequential layer where the FFN follows the attention add => not parallel.
    Catches GPT-J / Phi / Cohere, all flagless and all missed by the old table."""
    assert _topo(tmp_path, _PARALLEL_LAYER)["parallel_residual"] is True
    assert _topo(tmp_path, _PRE_LAYER)["parallel_residual"] is False


def test_layer_topology_finds_decoder_not_encoder_in_multimodal_file(tmp_path):
    """When a modeling file bundles several attention+ffn classes (a multimodal
    file's vision/audio ENCODER layers + the text decoder), the decoder is picked
    by its KV-cache forward parameter — an encoder doesn't cache. Without this the
    first class (an encoder, often parallel) is misread as the decoder's topology."""
    src = (
        "class FooVisionEncoderLayer:\n"          # first, but an encoder (no cache)
        "    def __init__(self):\n"
        "        self.norm1 = LayerNorm(8)\n"
        "        self.attn = FooAttention(8)\n"
        "        self.mlp = FooMLP(8)\n"
        "    def forward(self, x):\n"
        "        residual = x\n"
        "        a = self.attn(self.norm1(x))\n"
        "        m = self.mlp(self.norm1(x))\n"
        "        return residual + a + m\n"        # parallel — would mislead
        "\n\n" + _PRE_LAYER
    )
    topo = _topo(tmp_path, src)
    assert topo["norm_placement"] == "pre"
    assert topo["parallel_residual"] is False     # decoder picked, not the encoder


def test_layer_topology_real_families_match_code(tmp_path):
    """The installed modeling source must classify each family as its known
    structure — zero drift from the emptied table — and CATCH the flagless parallels
    (GPT-J / Phi) the table never listed. Skips a family whose source isn't
    installed (a gap, not a failure)."""
    from model_unfolder.evidence.patterns import decoder_layer_topology_from_files
    from model_unfolder.evidence.sources import resolve_source_files
    expect = {
        "llama": ("pre", False), "gemma2": ("double", False), "olmo2": ("post", False),
        "cohere": ("pre", True), "gpt_j": ("pre", True), "phi": ("pre", True),
        "phi3": ("pre", False),
    }
    seen = 0
    for mt, (place, parallel) in expect.items():
        files = resolve_source_files({"model_type": mt}, source="local").files
        topo = decoder_layer_topology_from_files(files)
        if topo is None:
            continue
        seen += 1
        assert topo["norm_placement"] == place, f"{mt}: {topo} != {place}"
        assert topo["parallel_residual"] is parallel, f"{mt}: {topo} parallel != {parallel}"
    assert seen >= 4, "too few installed families exercised — resolver may be broken"


def test_norm_kind_read_from_decoder_norm_class(tmp_path):
    """config-silent norm KIND is read from the decoder's NORM submodule class
    (RMSNorm vs LayerNorm), not a legacy model_type family-set.  An attention
    HELPER class whose only attention signal is a flash-attn flag field (no norm)
    must NOT be mistaken for the decoder layer."""
    from model_unfolder.evidence.patterns import decoder_norm_kind_from_files
    rms = (
        "class FooDecoderLayer:\n"
        "    def __init__(self):\n"
        "        self.input_layernorm = FooRMSNorm(8)\n"
        "        self.self_attn = FooAttention(8)\n"
        "    def forward(self, x, past_key_values=None):\n"
        "        return x\n"
    )
    ln = rms.replace("FooRMSNorm", "FooLayerNorm")
    helper = (                                  # flash-attn flag matches 'attn', has no norm
        "class FooFlashAttention2:\n"
        "    def __init__(self):\n"
        "        self._flag = flash_attn_supports_top_left_mask()\n"
        "    def forward(self, x, past_key_values=None):\n"
        "        return x\n"
        "\n\n" + ln
    )
    fr = tmp_path / "m_rms.py"; fr.write_text(rms)
    fl = tmp_path / "m_ln.py"; fl.write_text(ln)
    fh = tmp_path / "m_helper.py"; fh.write_text(helper)
    assert decoder_norm_kind_from_files([str(fr)]) == "rmsnorm"
    assert decoder_norm_kind_from_files([str(fl)]) == "layernorm"
    assert decoder_norm_kind_from_files([str(fh)]) == "layernorm"   # helper skipped


def test_norm_kind_real_legacy_families_are_layernorm():
    """The installed pre-RMSNorm decoders read LayerNorm from code (zero drift from
    the deleted family-set), modern decoders RMSNorm — config-silently."""
    from model_unfolder.evidence.patterns import decoder_norm_kind_from_files
    from model_unfolder.evidence.sources import resolve_source_files
    expect = {"gpt2": "layernorm", "opt": "layernorm", "bloom": "layernorm",
              "gptj": "layernorm", "falcon": "layernorm", "phi": "layernorm",
              "llama": "rmsnorm", "gemma2": "rmsnorm"}
    seen = 0
    for mt, kind in expect.items():
        files = resolve_source_files({"model_type": mt}, source="local").files
        got = decoder_norm_kind_from_files(files)
        if got is None:
            continue
        seen += 1
        assert got == kind, f"{mt}: {got} != {kind}"
    assert seen >= 5


def test_multi_variant_file_detected_by_layer_class_count(tmp_path):
    """A multi-variant modeling file is detected by counting distinct LAYER classes
    (attention + ffn/norm), not a hardcoded family name: one decoder layer => 1
    (single tower), + a vision encoder layer => 2 (multi-variant)."""
    from model_unfolder.evidence.patterns import layer_class_count_from_files
    one = (
        "class FooDecoderLayer:\n"
        "    def __init__(self):\n"
        "        self.input_layernorm = FooRMSNorm(8)\n"
        "        self.self_attn = FooAttention(8)\n"
        "        self.mlp = FooMLP(8)\n"
        "    def forward(self, x, past_key_values=None):\n"
        "        return x\n"
    )
    two = one + (
        "\n\nclass FooVisionEncoderLayer:\n"
        "    def __init__(self):\n"
        "        self.norm1 = FooLayerNorm(8)\n"
        "        self.attn = FooAttention(8)\n"
        "        self.mlp = FooMLP(8)\n"
        "    def forward(self, x):\n"
        "        return x\n"
    )
    f1 = tmp_path / "m_one.py"; f1.write_text(one)
    f2 = tmp_path / "m_two.py"; f2.write_text(two)
    assert layer_class_count_from_files([str(f1)]) == 1
    assert layer_class_count_from_files([str(f2)]) == 2
