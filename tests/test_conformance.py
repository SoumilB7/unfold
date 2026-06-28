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


def test_ffn_kind_read_from_conv_glumbconv_construction(tmp_path):
    """The FFN KIND is read from the block's constructed ff class (the SAME
    init-construction evidence attn-kind reads): a block building GLUMBConv =>
    "conv_glu" (Sana's gated conv Mix-FFN); a plain FeedForward => None (caller's
    Linear-MLP default).  Replaces the per-model ffn_kind table."""
    from model_unfolder.evidence.patterns import diffusion_ffn_kind_from_files
    conv = (
        "class MyBlock:\n"
        "    def __init__(self, dim):\n"
        "        self.ff = GLUMBConv(dim)\n"
        "    def forward(self, x):\n"
        "        return self.ff(x)\n"
    )
    mlp = conv.replace("GLUMBConv", "FeedForward")
    fc = tmp_path / "m_conv.py"; fc.write_text(conv)
    fm = tmp_path / "m_mlp.py"; fm.write_text(mlp)
    assert diffusion_ffn_kind_from_files([str(fc)]) == "conv_glu"
    assert diffusion_ffn_kind_from_files([str(fm)]) is None


def test_gate_via_norm_distinguishes_gate_norm_from_film_norm(tmp_path):
    """gate-via-norm (Mochi) is read STRUCTURALLY: a *Modulated*Norm class whose
    forward gates the normed output by a scale (`*`) with NO additive FiLM shift
    (`+`).  A standard AdaLN FiLM norm (`norm*(1+scale)+shift`, e.g. Sana's
    SanaModulatedNorm) has the additive shift and is NOT gate-via-norm — the
    distinction that stops Sana being falsely flipped."""
    from model_unfolder.evidence.patterns import diffusion_gate_via_norm_from_files
    gate = (
        "class FooModulatedRMSNorm:\n"
        "    def __init__(self, eps):\n"
        "        self.norm = RMSNorm(eps)\n"
        "    def forward(self, x, scale=None):\n"
        "        x = self.norm(x)\n"
        "        x = x * scale\n"
        "        return x\n"
    )
    film = (
        "class FooModulatedNorm:\n"
        "    def __init__(self, dim):\n"
        "        self.norm = LayerNorm(dim)\n"
        "    def forward(self, x, temb, table):\n"
        "        x = self.norm(x)\n"
        "        shift, scale = (table[None] + temb).chunk(2, dim=1)\n"
        "        x = x * (1 + scale) + shift\n"
        "        return x\n"
    )
    fg = tmp_path / "m_gate.py"; fg.write_text(gate)
    ff = tmp_path / "m_film.py"; ff.write_text(film)
    assert diffusion_gate_via_norm_from_files([str(fg)]) is True
    assert diffusion_gate_via_norm_from_files([str(ff)]) is False


def test_qk_norm_type_read_from_four_code_spellings(tmp_path):
    """The Q/K-norm TYPE is read from the four code spellings observed across the
    DiT corpus: a norm_q field class, a literal kwarg, a variable kwarg resolved to
    its param default, and an IfExp constant — each yielding rms_norm vs layer_norm.
    Replaces the per-model qk_norm table (zero drift on all 7 corpus models)."""
    from model_unfolder.evidence.patterns import diffusion_qk_norm_from_files
    field_rms = (
        "class A:\n"
        "    def __init__(self):\n"
        "        self.norm_q = RMSNorm(8)\n"
        "        self.norm_added_q = RMSNorm(8)\n"
        "    def forward(self, x):\n        return x\n"
    )
    literal_layer = (
        "class B:\n"
        "    def __init__(self):\n"
        "        self.attn = Attention(8, qk_norm='fp32_layer_norm')\n"
        "    def forward(self, x):\n        return x\n"
    )
    param_default = (
        "class C:\n"
        "    def __init__(self, qk_norm='rms_norm'):\n"
        "        self.attn = Attention(8, qk_norm=qk_norm)\n"
        "    def forward(self, x):\n        return x\n"
    )
    ifexp = (
        "class D:\n"
        "    def __init__(self, qk_norm=True):\n"
        "        self.attn = Attention(8, qk_norm='layer_norm' if qk_norm else None)\n"
        "    def forward(self, x):\n        return x\n"
    )
    plain = (
        "class E:\n"
        "    def __init__(self):\n"
        "        self.attn = Attention(8)\n"
        "    def forward(self, x):\n        return x\n"
    )
    def w(name, s):
        f = tmp_path / name; f.write_text(s); return str(f)
    assert diffusion_qk_norm_from_files([w("a.py", field_rms)]) == "rms_norm"
    assert diffusion_qk_norm_from_files([w("b.py", literal_layer)]) == "layer_norm"
    assert diffusion_qk_norm_from_files([w("c.py", param_default)]) == "rms_norm"
    assert diffusion_qk_norm_from_files([w("d.py", ifexp)]) == "layer_norm"
    assert diffusion_qk_norm_from_files([w("e.py", plain)]) is None


def test_single_stream_fusion_anchored_to_built_block(tmp_path):
    """single-stream fusion is read from the block the model BUILDS into a single_*
    ModuleList (not any *Single* class): a real FFN submodule + no concat =>
    sequential; concat + MLP linears => concat_fused; concat + fused parallel attn
    => parallel.  A model that DEFINES a *Single* block but never stacks it (SD3) has
    no single-stream blocks => None — the false-positive this anchoring prevents."""
    from model_unfolder.evidence.patterns import diffusion_single_stream_fusion_from_files
    sequential = (
        "class FooSingleTransformerBlock:\n"
        "    def __init__(self):\n"
        "        self.norm1 = AdaLayerNormZero(8)\n"
        "        self.attn = Attention(8)\n"
        "        self.ff = FeedForward(8)\n"
        "    def forward(self, x):\n        return self.ff(self.attn(x))\n"
        "\nclass FooModel:\n"
        "    def __init__(self):\n"
        "        self.single_transformer_blocks = nn.ModuleList([FooSingleTransformerBlock() for _ in range(2)])\n"
        "    def forward(self, x):\n        return x\n"
    )
    concat_fused = (
        "class BarSingleTransformerBlock:\n"
        "    def __init__(self):\n"
        "        self.norm = AdaLayerNormZeroSingle(8)\n"
        "        self.proj_mlp = nn.Linear(8, 32)\n"
        "        self.proj_out = nn.Linear(40, 8)\n"
        "        self.attn = Attention(8)\n"
        "    def forward(self, x):\n        return self.proj_out(torch.cat([self.attn(x), self.proj_mlp(x)]))\n"
        "\nclass BarModel:\n"
        "    def __init__(self):\n"
        "        self.single_transformer_blocks = nn.ModuleList([BarSingleTransformerBlock() for _ in range(2)])\n"
        "    def forward(self, x):\n        return x\n"
    )
    # Defines a *Single* block but never stacks it -> not a single-stream model.
    defined_unused = (
        "class BazSingleTransformerBlock:\n"
        "    def __init__(self):\n"
        "        self.attn = Attention(8)\n"
        "        self.ff = FeedForward(8)\n"
        "    def forward(self, x):\n        return self.ff(self.attn(x))\n"
        "\nclass BazModel:\n"
        "    def __init__(self):\n"
        "        self.transformer_blocks = nn.ModuleList([SomeDualBlock() for _ in range(2)])\n"
        "    def forward(self, x):\n        return x\n"
    )
    def w(name, s):
        f = tmp_path / name; f.write_text(s); return str(f)
    assert diffusion_single_stream_fusion_from_files([w("seq.py", sequential)]) == "sequential"
    assert diffusion_single_stream_fusion_from_files([w("cf.py", concat_fused)]) == "concat_fused"
    assert diffusion_single_stream_fusion_from_files([w("unused.py", defined_unused)]) is None


def test_axes_dims_rope_read_from_init_default(tmp_path):
    """Axial-RoPE per-axis dims are read from the model __init__ default tuple
    (Flux axes_dims_rope=(16,56,56)); a model without the param => None."""
    from model_unfolder.evidence.patterns import diffusion_axes_dims_rope_from_files
    axial = (
        "class FooModel:\n"
        "    def __init__(self, axes_dims_rope=(16, 56, 56)):\n"
        "        self.x = 1\n"
        "    def forward(self, x):\n        return x\n"
    )
    none = (
        "class BarModel:\n"
        "    def __init__(self, dim=8):\n"
        "        self.x = 1\n"
        "    def forward(self, x):\n        return x\n"
    )
    fa = tmp_path / "ax.py"; fa.write_text(axial)
    fn = tmp_path / "no.py"; fn.write_text(none)
    assert diffusion_axes_dims_rope_from_files([str(fa)]) == [16, 56, 56]
    assert diffusion_axes_dims_rope_from_files([str(fn)]) is None


def test_ffn_activation_reads_inline_standalone_act_field(tmp_path):
    """A block whose FFN is INLINE (no FeedForward submodule) but builds a
    standalone activation field — PRX's self.mlp_act = GELU(approximate='tanh') —
    resolves to gelu-approximate; the fallback fires only when the standard FFN
    scan finds nothing, so standard-FFN models are unaffected."""
    from model_unfolder.evidence.patterns import diffusion_ffn_activation_from_files
    inline = (
        "class FooBlock:\n"
        "    def __init__(self, dim):\n"
        "        self.mlp_act = GELU(approximate='tanh')\n"
        "        self.linear1 = nn.Linear(dim, dim)\n"
        "    def forward(self, x):\n        return self.linear1(self.mlp_act(x))\n"
    )
    # a standard FeedForward block must still win via the normal scan, not the fallback
    standard = (
        "class BarBlock:\n"
        "    def __init__(self, dim):\n"
        "        self.ff = FeedForward(dim, activation_fn='geglu')\n"
        "        self.extra_act = SiLU()\n"
        "    def forward(self, x):\n        return self.ff(x)\n"
    )
    fi = tmp_path / "inline.py"; fi.write_text(inline)
    fs = tmp_path / "std.py"; fs.write_text(standard)
    assert diffusion_ffn_activation_from_files([str(fi)]) == "gelu-approximate"
    assert diffusion_ffn_activation_from_files([str(fs)]) == "geglu"   # standard scan wins


def test_diffusor_class_defaults_table_is_empty_all_code_derived():
    """The per-model diffusor class_defaults table is FULLY EMPTY — every
    architectural fact (qk_norm, ffn activation/kind, rope/axial dims, gate dialect,
    single-stream fusion, attn kind, cross-attn norm) is now read from the modeling
    SOURCE, not tabulated by class name. This is the conscious-abstraction gate: a
    NEW row is the law's tolerated 'truly-opaque source' exception, so adding one
    must be deliberate — update this test WITH the justification (why the evidence
    genuinely can't be read), never silently."""
    from model_unfolder.everchanging import load_diffusion_class_defaults
    table = load_diffusion_class_defaults()
    rows = {field: mapping for field, mapping in table.items() if mapping}
    assert rows == {}, (
        "diffusor class_defaults gained per-model rows — derive from code instead, "
        f"or justify as truly-opaque here: {rows}")


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


# ===========================================================================
# RECURSIVE nested-drill conformance — diff each leaf-compute drill (attention /
# FFN / expert internals) against the TRANSITIVE forward() closure of its backing
# sub-module (following sdpa / rotary / the diffusers processor / the FeedForward
# ModuleList).  One altitude below the per-layer check above.
# ===========================================================================

from model_unfolder.evidence import check_nested_conformance
from model_unfolder.evidence.transitive import build_registry, transitive_closure
from model_unfolder.everchanging import load_conformance_transitive
from model_unfolder.renderers.html.graph_engine import drain_render_log, reset_render_log


def _render_log(cfg):
    reset_render_log()
    mu.unfold(cfg).to_html(standalone=True)
    return drain_render_log()


@pytest.mark.parametrize("name", list(tc.CORPUS))
def test_nested_conformance_clean_over_corpus(name):
    """Every leaf-compute drill in the corpus conforms to its sub-module's
    transitive forward() closure — both directions, no fabrication / salient
    omission. The corpus spans attention (GQA / MLA), gated & dense FFN, experts,
    vision/audio/video, DiT/MMDiT, UNet."""
    cfg = tc.CORPUS[name]
    problems = check_nested_conformance(cfg, _render_log(cfg))
    assert problems == [], "\n".join(p.message for p in problems)


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
    from model_unfolder.evidence.conformance import _augment_diffusion_files, _role_union_closures
    bundle = resolve_source_files(_cogvideox_cfg(), source="local")
    if not bundle.files:
        pytest.skip("diffusers cogvideox source not installed")
    reg = build_registry(_augment_diffusion_files(bundle.files))
    vocab = load_conformance_transitive()
    closures = _role_union_closures(reg, vocab)
    assert "attention" in closures, "no attention sub-module resolved"
    ops, _cls = closures["attention"]
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
    fake_closures = {"ffn": (frozenset({"linear", "activation", "gate_mul"}), "FakeMLP")}
    monkeypatch.setattr(conf, "_role_union_closures", lambda *a, **k: fake_closures)
    monkeypatch.setattr(conf, "resolve_source_files",
                        lambda *a, **k: type("B", (), {"files": ("x.py",)})())
    monkeypatch.setattr(conf, "build_registry", lambda *a, **k: {})
    # the drill DRAWS a concat the code never does -> fabricated
    log = [("ffn", frozenset({"linear", "activation", "gate_mul", "concat", "port"}), frozenset())]
    problems = conf.check_nested_conformance({"model_type": "x"}, log)
    assert any(p.kind == "fabricated" and p.op == "concat" for p in problems), \
        [p.message for p in problems]


def test_dense_ffn_drill_not_false_flagged_when_a_sibling_is_gated(monkeypatch):
    """SOUNDNESS REGRESSION: there is NO leaf ffn/expert salient `gate_mul` check,
    because the diff resolves a role against the UNION of same-role sub-modules and
    a multimodal model has a GATED text MLP next to a DENSE vision MLP — so the
    union carries `gate_mul` even though THIS (dense) FFN drill correctly omits it.
    Drawing the dense FFN must stay CLEAN (the dense-vs-gated fact is fixed at the
    parser by the code-derived gating rail, not guessed here)."""
    from model_unfolder.evidence import conformance as conf
    # union has gate_mul (from a gated sibling) — but a dense drill must NOT flag it
    fake_closures = {"ffn": (frozenset({"linear", "activation", "gate_mul"}), "FakeMLP")}
    monkeypatch.setattr(conf, "_role_union_closures", lambda *a, **k: fake_closures)
    monkeypatch.setattr(conf, "resolve_source_files",
                        lambda *a, **k: type("B", (), {"files": ("x.py",)})())
    monkeypatch.setattr(conf, "build_registry", lambda *a, **k: {})
    monkeypatch.setattr(conf, "_block_classes", lambda *a, **k: [])
    log = [("ffn", frozenset({"linear", "activation", "port"}), frozenset())]   # dense drawn
    assert conf.check_nested_conformance({"model_type": "x"}, log) == []


def test_opaque_drill_makes_no_claim(monkeypatch):
    """An honest-unknown OPAQUE drill (the parser could not decompose the
    sub-module, so it drew one ``opaque`` block) must NOT be held to fabrication or
    salient-omission — Sana's GLUMBConv FFN is drawn opaque and must stay clean."""
    from model_unfolder.evidence import conformance as conf
    fake_closures = {"ffn": (frozenset({"linear", "activation", "gate_mul"}), "FakeMLP")}
    monkeypatch.setattr(conf, "_role_union_closures", lambda *a, **k: fake_closures)
    monkeypatch.setattr(conf, "resolve_source_files",
                        lambda *a, **k: type("B", (), {"files": ("x.py",)})())
    monkeypatch.setattr(conf, "build_registry", lambda *a, **k: {})
    log = [("ffn", frozenset({"opaque", "port"}), frozenset())]
    assert conf.check_nested_conformance({"model_type": "x"}, log) == []


def test_code_derived_ffn_gating_overrides_rmsnorm_heuristic():
    """The dense-vs-gated FACT is code-derived: ``PhiMLP`` is dense (no gate_mul) so
    a Phi config (RMSNorm + gelu_new) renders a DENSE FFN, NOT the gated FFN the
    rmsnorm heuristic would have drawn — and ``LlamaMLP`` (gate_mul) stays gated.
    This is what keeps the parser and the nested net from diverging."""
    from model_unfolder.evidence.patterns import decoder_ffn_gated_from_files
    for mt, expected in (("phi", False), ("llama", True)):
        bundle = resolve_source_files({"model_type": mt}, source="local")
        if not bundle.files:
            pytest.skip(f"transformers {mt} source not installed")
        assert decoder_ffn_gated_from_files(bundle.files) is expected


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
    rc = conf._role_union_closures(reg, load_conformance_transitive())
    sel = rc.get("ffn", (frozenset(),))[0] | rc.get("route", (frozenset(),))[0]
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
    import model_unfolder as mu
    cfg = {"model_type": mt, "hidden_size": 128, "num_hidden_layers": 2,
           "num_attention_heads": 8, "num_key_value_heads": 2, "intermediate_size": 256,
           "moe_intermediate_size": 128, "vocab_size": 1000, "num_experts": 8,
           "num_local_experts": 8, "num_experts_per_tok": 2}
    if not resolve_source_files(cfg, source="local").files:
        pytest.skip(f"{mt} source not installed")
    problems = check_nested_conformance(cfg, _render_log(cfg))
    assert problems == [], "\n".join(p.message for p in problems)
