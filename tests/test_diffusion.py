"""Diffusion (DiT/MMDiT) adapter — detection, IR shape, theme, render health.

These use real, public diffusers config values (FLUX.1-dev transformer,
PixArt-alpha transformer) as plain dicts — no network, no model code executed.
"""
import pytest

from model_unfolder import unfold, config_to_ir
from model_unfolder.adapters.diffusor import parser as diffusor
from model_unfolder.adapters.transformer import parser as transformer
from model_unfolder.block_schema import validate_block_tree, validate_click_coupling


# Real FLUX.1-dev transformer/config.json values (+ pipeline wiring the by-ID
# loader merges in: text encoders, scheduler, scheduler config).
FLUX = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.30.0",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "_scheduler_config": {"num_train_timesteps": 1000, "shift": 3.0},
    "_vae_config": {
        "_class_name": "AutoencoderKL",
        "block_out_channels": [128, 256, 512, 512],
        "latent_channels": 16,
        "out_channels": 3,
        "layers_per_block": 2,
        "scaling_factor": 0.3611,
    },
    # Real text-encoder configs the by-ID loader pulls from text_encoder/ and
    # text_encoder_2/ (CLIP ViT-L/14 + T5-v1.1-XXL encoder).
    "_text_encoder_configs": {
        "text_encoder": {
            "_class_name": "CLIPTextModel", "num_hidden_layers": 12, "hidden_size": 768,
            "num_attention_heads": 12, "intermediate_size": 3072, "hidden_act": "quick_gelu",
            "max_position_embeddings": 77, "vocab_size": 49408,
        },
        "text_encoder_2": {
            "_class_name": "T5EncoderModel", "num_layers": 24, "d_model": 4096,
            "num_heads": 64, "d_ff": 10240, "dense_act_fn": "gelu_new", "vocab_size": 32128,
            "is_gated_act": True, "feed_forward_proj": "gated-gelu",
        },
    },
}

# Real PixArt-alpha transformer config (single-stream, cross-attention to text).
PIXART = {
    "_class_name": "PixArtTransformer2DModel",
    "_diffusers_version": "0.27.0",
    "num_layers": 28,
    "num_attention_heads": 16,
    "attention_head_dim": 72,
    "cross_attention_dim": 1152,
    "caption_channels": 4096,
    "patch_size": 2,
    "in_channels": 4,
    "sample_size": 128,
    "norm_type": "ada_norm_single",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
}

LLAMA = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
    "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
    "rms_norm_eps": 1e-5,
}


def test_diffusor_matches_dit_not_transformer():
    assert diffusor.matches(FLUX) is True
    assert diffusor.matches(PIXART) is True
    # The diffusor adapter must NOT claim ordinary transformer-LLM configs;
    # it is registered before the catch-all transformer adapter.
    assert diffusor.matches(LLAMA) is False
    # The catch-all transformer adapter still matches everything (by design).
    assert transformer.matches(FLUX) is True


def test_flux_layer_count_and_geometry():
    ir = config_to_ir(FLUX)
    # 19 double-stream (joint) + 38 single-stream blocks.
    assert ir.num_layers == 57
    # hidden = num_attention_heads x attention_head_dim.
    assert ir.hidden_size == 24 * 128 == 3072
    # No token vocabulary in a denoiser.
    assert ir.vocab_size == 0
    # DiT attention is full bidirectional multi-head attention with axial RoPE.
    attn = ir.layers[0].attention
    assert attn.kind == "mha"
    assert attn.num_kv_heads == attn.num_heads == 24
    assert attn.mask == "full"
    assert attn.no_rope is False
    assert attn.rope_dim == 16 + 56 + 56   # axes_dims_rope sum to head_dim
    # Flux declares no activation_fn in config, but its model class fixes it
    # (FeedForward gelu-approximate, non-gated) — surfaced from class_defaults and
    # MARKED code-derived, never guessed (see test_flux_ffn_activation_is_code_derived_gelu).
    assert ir.layers[0].ffn.gated is False
    assert ir.layers[0].ffn.activation == "gelu-approximate"
    assert ir.layers[0].ffn.activation_from_class is True
    assert ir.layers[0].ffn.activation_assumed is False


def test_diffusion_render_spec_theme():
    ir = config_to_ir(FLUX)
    render = ir.extras["render"]
    assert render["family"] == "diffusion"
    # Green (teal) for now — same family theme as LLMs.
    assert render["theme"] == "teal"


def test_diffusion_renders_in_the_green_llm_theme():
    diffusion_html = unfold(FLUX).to_html(standalone=True)
    assert "#0F6E56" in diffusion_html         # teal/green block colour present
    assert "#1E5FB0" not in diffusion_html      # blue palette not used for now
    # Same palette as a transformer.
    transformer_html = unfold(LLAMA).to_html(standalone=True)
    assert "#0F6E56" in transformer_html


def test_denoiser_skeleton_is_drawn():
    html = unfold(FLUX).to_html(standalone=True)
    for label in ("Noisy latent", "Patchify", "Output projection", "Unpatchify", "VAE decode"):
        assert label in html, f"skeleton stage {label!r} not drawn"
    assert "AdaLN-Out" not in html
    assert "-&gt; noise eps" not in html


def test_diffusion_model_blocks_are_typed():
    ir = config_to_ir(FLUX)
    blocks = {
        block["id"]: block
        for block in ir.extras["render"]["model_blocks"]
    }
    assert blocks["tok_text"]["diffusion_stage"] == "latent_input"
    assert blocks["embed"]["diffusion_stage"] == "patchify"
    assert blocks["final_rms"]["diffusion_stage"] == "output_projection"
    assert blocks["lm_head"]["diffusion_stage"] == "unpatchify"

    side_blocks = {
        block["id"]: block
        for block in ir.layers[0].blocks
        if block.get("lane")
    }
    assert side_blocks["adaln_cond"]["diffusion_stage"] == "timestep_conditioning"
    assert side_blocks["text_cond"]["diffusion_stage"] == "text_conditioning"


def test_diffusion_name_from_model_tag_and_stats():
    """Header name comes from the model tag (repo id), and the stats banner shows
    diffusion-specific cells (Timesteps, Latent) instead of Vocab / Context."""
    cfg = {**FLUX, "_repo_id": "black-forest-labs/FLUX.1-dev"}
    ir = config_to_ir(cfg)
    assert ir.name == "FLUX.1-dev"           # not "transformer" / the component path
    html = unfold(cfg).to_html(standalone=True)
    assert "TIMESTEPS" in html and "1,000" in html
    assert "LATENT" in html and "64 ch" in html
    assert "VOCAB" not in html and "CONTEXT" not in html
    # Transformers keep Vocab / Context.
    assert "VOCAB" in unfold(LLAMA).to_html(standalone=True)


def test_loop_blocks_are_typed_with_approved_stages():
    """Every sampling-loop node carries an approved diffusion_stage, so the hero
    view is under the same type guard as the denoiser."""
    from model_unfolder.block_schema import DIFFUSION_STAGES
    ir = config_to_ir(FLUX)
    loop = {b["id"]: b for b in ir.extras["render"]["loop_blocks"]}
    expected = {
        "noise": "noise_input", "timestep": "timestep", "prompt": "prompt",
        "encoder_0": "text_encoder", "encoder_1": "text_encoder",
        "denoiser": "denoiser", "scheduler": "scheduler",
        "vae_decode": "vae_decode", "image": "image_output",
    }
    for bid, stage in expected.items():
        assert loop[bid]["diffusion_stage"] == stage, bid
        assert stage in DIFFUSION_STAGES, stage


def test_unknown_diffusion_blocks_render_unresolved():
    from model_unfolder.renderers.html.views import _is_resolved_diffusion_block

    info = {"blocks": {}}
    assert _is_resolved_diffusion_block(True, info, "embed", {"diffusion_stage": "patchify"})
    assert not _is_resolved_diffusion_block(True, info, "new_slot", {"diffusion_stage": "not_approved"})
    assert not _is_resolved_diffusion_block(True, info, "new_slot", {"kind": "linear", "label": "New slot"})
    # An unapproved stage on a loop node renders pale (unresolved) in the hero view.
    assert not _is_resolved_diffusion_block(True, info, "scheduler", {"diffusion_stage": "made_up"})


def test_main_view_is_the_sampling_loop():
    """The hero image is the recursive sampling loop, not the transformer stack."""
    html = unfold(FLUX).to_html(standalone=True)
    assert "SAMPLING LOOP" in html
    assert "sampling step" in html       # honest loop framing (no invented step count)
    assert "× T steps" not in html        # the old placeholder is gone
    assert "↺ t → 0" in html             # the loop frame's repeat pill (engine style)
    # The denoiser is a clickable loop node with a backing card.
    assert 'data-id="denoiser"' in html
    assert 'data-card-id="denoiser"' in html
    # Loop nodes present.
    for node in ("noise", "scheduler", "vae_decode", "image", "timestep"):
        assert f'data-id="{node}"' in html, f"loop node {node!r} missing"


def test_dit_layers_typed_correctly_not_as_llm():
    """The DiT blocks must NOT inherit LLM defaults: attention is full/bidirectional
    (not causal) with RoPE (not NoPE), and the two streams are named distinctly."""
    html = unfold(FLUX).to_html(standalone=True)
    assert "NoPE" not in html             # Flux uses axial RoPE, not NoPE
    assert "causal" not in html.lower()   # DiT attention is bidirectional
    assert "Joint Attn" in html
    assert "MM-DiT" in html               # double-stream named
    assert "single-stream" in html        # single-stream named
    ir = config_to_ir(FLUX)
    a0 = ir.layers[0].attention
    assert a0.mask == "full" and a0.no_rope is False


def test_denoiser_drills_into_the_dit_stack():
    """Clicking the denoiser opens the transformer architecture one panel deeper:
    its card must embed the DiT stack's clickable layer nodes (attention, etc.)."""
    import re
    html = unfold(FLUX).to_html(standalone=True)
    m = re.search(
        r'data-card-id="denoiser"(.*?)</div>\s*<div class="uf-card-detail', html, re.S
    )
    assert m, "denoiser card not found"
    denoiser_card = m.group(1)
    assert 'data-id="attn"' in denoiser_card      # DiT layer attention is reachable
    assert 'data-id="embed"' in denoiser_card      # patchify
    # Three drill depths exist: L2 loop, L3 DiT blocks, L4 internals.
    depths = sorted(set(re.findall(r'data-depth="(\d+)"', html)))
    assert depths == ["2", "3", "4"]


def test_text_encoders_render_as_separate_blocks():
    """The conditioning shows one block per real encoder (Flux: CLIP + T5) fed by
    a shared prompt — not a single combined 'CLIP + T5' block."""
    ir = config_to_ir(FLUX)
    loop_ids = [b["id"] for b in ir.extras["render"]["loop_blocks"]]
    assert "prompt" in loop_ids
    assert "encoder_0" in loop_ids and "encoder_1" in loop_ids
    html = unfold(FLUX).to_html(standalone=True)
    assert "CLIP" in html and "T5" in html
    # Each encoder is a clickable node with a backing card.
    for nid in ("prompt", "encoder_0", "encoder_1"):
        assert f'data-id="{nid}"' in html and f'data-card-id="{nid}"' in html


def test_text_encoder_breaks_into_drillable_ops():
    """Clicking an encoder opens its transformer-layer cell; each op (embedding,
    self-attention, FFN, add & norm) is a clickable node with a matching card,
    namespaced per encoder so CLIP and T5 don't share a card."""
    html = unfold(FLUX).to_html(standalone=True)
    for op in ("embed", "norm", "selfattn", "ffn", "add"):
        for enc in ("encoder_0", "encoder_1"):
            assert f'data-id="{enc}_op_{op}"' in html
            assert f'data-card-id="{enc}_op_{op}"' in html


def test_text_encoder_shows_real_config_dims():
    """When the loader fetched the encoders' configs, the view shows their real
    depth/width/heads/FFN (not a schematic 'N'), distinctly per encoder."""
    specs = diffusor._text_encoder_specs(FLUX)
    assert specs == [
        {"name": "CLIP", "layers": 12, "hidden": 768, "kind": "mha", "heads": 12,
         "kv_heads": 12, "head_dim": 64, "ffn": 3072,
         "activation": "quick_gelu", "vocab": 49408, "max_pos": 77, "gated": False},
        {"name": "T5", "layers": 24, "hidden": 4096, "kind": "mha", "heads": 64,
         "kv_heads": 64, "head_dim": 64, "ffn": 10240,
         "activation": "gelu_new", "vocab": 32128, "gated": True},
    ]
    html = unfold(FLUX).to_html(standalone=True)
    assert "× 12" in html and "× 24" in html   # real depths
    assert "12 heads" in html and "64 heads" in html
    assert "768 → 3,072" in html and "4,096 → 10,240" in html


def test_text_encoder_falls_back_when_no_config():
    """Without fetched encoder configs, the view stays honest: schematic '× N
    layers', no invented numbers."""
    flux_no_enc = {k: v for k, v in FLUX.items() if k != "_text_encoder_configs"}
    specs = diffusor._text_encoder_specs(flux_no_enc)
    assert specs == [{"name": "CLIP"}, {"name": "T5"}]
    html = unfold(flux_no_enc).to_html(standalone=True)
    assert "× N" in html


def test_vae_decoder_has_a_drill_view():
    """VAE decode opens its own view, built from the real VAE config (channels,
    upsampling) the loader fetched."""
    ir = config_to_ir(FLUX)
    vae_block = next(b for b in ir.extras["render"]["loop_blocks"] if b["id"] == "vae_decode")
    assert vae_block.get("view") == "vae_decoder"
    assert vae_block["detail"]["block_out_channels"] == [128, 256, 512, 512]
    html = unfold(FLUX).to_html(standalone=True)
    # Real decoder stages drawn compactly: 8x upscale (3 doublings), 128->3 output head.
    assert "Up stage" in html and "Output image head" in html
    assert "8× upscaled" in html
    assert "z₀ (clean)" not in html      # the removed loop-arrow label stays gone


def test_flux_splits_double_and_single_stream_groups():
    """The denoiser layer map must distinguish Flux's 19 double-stream (sequential)
    from its 38 single-stream (parallel) blocks — not collapse them into one."""
    from model_unfolder.renderers.html.metadata import _make_info
    info = _make_info(unfold(FLUX).to_ir())
    groups = info["groups"]
    assert len(groups) == 2
    assert sorted(len(g["indices"]) for g in groups) == [19, 38]


def test_flux_single_stream_block_has_no_text_rail_and_clean_labels():
    """Option-1 single-stream depiction: text + image are joined into ONE sequence
    ONCE before the stack, so the block takes no per-block text rail — only the
    timestep (AdaLN) conditions it.  A per-block text rail was both a
    cross-attention-like misread AND a line forced to cross the parallel MLP
    branch (it landed on attention's far port through the MLP box).

    The op-conformance net checks op-KINDS, not side-input wiring or labels, so
    these facts need their own pin:
    * a single-stream layer carries adaln_cond but NOT text_cond;
    * a dual-stream layer STILL carries text_cond (joint attn genuinely mixes the
      two streams there — the suppression must be single-stream-only);
    * neither attention label double-wraps its tag in parens
      ("(MM-DiT (dual-stream))" regressed to "(dual-stream)");
    * the single-stream variant declares the one-time-join stack caption (and the
      dual one does NOT);
    * the fused output projection is disambiguated from the model-level bookend.
    """
    ir = config_to_ir(FLUX)

    def _is_single(layer):
        return "single-stream" in str(
            (layer.attention.variant or {}).get("tag") or "").lower()

    single = next(L for L in ir.layers if _is_single(L))
    dual = next(L for L in ir.layers if not _is_single(L))

    single_side = {b["id"] for b in single.blocks if b.get("lane")}
    dual_side = {b["id"] for b in dual.blocks if b.get("lane")}
    assert "adaln_cond" in single_side and "text_cond" not in single_side
    assert "text_cond" in dual_side          # dual keeps its (correct) text input

    assert single.attention.variant["label"] == ["Joint Attention", "(single-stream)"]
    assert dual.attention.variant["label"] == ["Joint Attention", "(dual-stream)"]
    for L in (single, dual):                 # no nested parens in the block label
        suffix = L.attention.variant["label"][1]
        assert suffix.count("(") == 1 and suffix.count(")") == 1

    assert single.attention.variant.get("stack_note")      # join surfaced (single only)
    assert not dual.attention.variant.get("stack_note")

    proj = next(b for b in single.blocks if b.get("id") == "ss_proj")
    assert proj["label"] == "Fused projection"             # != model-level "Output projection"

    # The caption renders in the single-stream variant's architecture SVG, and no
    # "Text tokens conditioning" rail is drawn for that variant.
    html = unfold(FLUX).to_html(standalone=True)
    assert "joined once before this stack" in html


@pytest.mark.parametrize("cfg", [FLUX, PIXART])
def test_diffusion_blocks_and_clicks_valid(cfg):
    ir = config_to_ir(cfg)
    assert validate_block_tree(ir) == []
    html = unfold(cfg).to_html(standalone=True)
    assert validate_click_coupling(html) == []


# Real SDXL-base UNet config shape (+ pipeline wiring).
SDXL_UNET = {
    "_class_name": "UNet2DConditionModel", "_repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "in_channels": 4, "out_channels": 4, "block_out_channels": [320, 640, 1280],
    "layers_per_block": 2, "cross_attention_dim": 2048, "transformer_layers_per_block": [1, 2, 10],
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    "mid_block_type": "UNetMidBlock2DCrossAttn", "addition_embed_type": "text_time",
    "scheduler": ["diffusers", "EulerDiscreteScheduler"], "_scheduler_config": {"num_train_timesteps": 1000},
    "text_encoder": ["transformers", "CLIPTextModel"], "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
}


def test_unet_is_claimed_by_diffusor_not_transformer():
    assert diffusor.matches(SDXL_UNET) is True
    assert config_to_ir(SDXL_UNET).architecture == "UNet2DConditionModel"


def test_sdxl_unet_conditioning_is_honest():
    """SDXL conformance pins:
    * BOTH text encoders survive (CLIP-L + OpenCLIP-bigG), never folded into one;
    * conditioning is described with the UNet mechanism (additive into ResNets,
      cross-attention for text) — NEVER 'AdaLN modulation' (a DiT mechanism);
    * the text_time micro-conditioning (pooled + size/crop/target) is surfaced;
    * the mid block reports its real Transformer2D depth (transformer_layers_per_block[-1]).
    """
    ir = config_to_ir(SDXL_UNET)
    encoders = ir.extras["diffusion"]["text_encoders"]
    assert len(encoders) == 2                      # dual encoders, not collapsed

    html = unfold(SDXL_UNET).to_html(standalone=True)
    assert "encoded by 2 text encoder" in html
    # UNet mechanism, never AdaLN asserted as the mechanism.
    assert "as AdaLN modulation" not in html
    assert "additively, not through AdaLN" in html
    assert "cross-attention K/V" in html
    # SDXL micro-conditioning (addition_embed_type = text_time) surfaced.
    assert "addition_embed_type = text_time" in html
    # Mid block transformer depth is the real 10, not a hardcoded 1.
    assert ir.extras["unet"]["mid"]["transformers"] == 10


def test_unet_stage_part_kinds_resolve_for_all_block_variants():
    """Every down/up stage of a real UNet is a recognised part_kind (solid), even
    the Resnet*sample / Simple* block variants (DeepFloyd / Kandinsky) — they're
    classified by position, so none render pale."""
    from model_unfolder.block_schema import DIFFUSION_PART_KINDS
    DEEPFLOYD = {
        "_class_name": "UNet2DConditionModel", "in_channels": 3, "out_channels": 6,
        "block_out_channels": [320, 640, 1280, 1280], "layers_per_block": 3,
        "cross_attention_dim": 4096,
        "down_block_types": ["ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D",
                             "SimpleCrossAttnDownBlock2D", "SimpleCrossAttnDownBlock2D"],
        "up_block_types": ["SimpleCrossAttnUpBlock2D", "SimpleCrossAttnUpBlock2D",
                           "SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
        "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    }
    ir = config_to_ir(DEEPFLOYD)
    u = ir.extras["unet"]
    for st in u["down"] + u["up"]:
        assert st.get("diffusion_part_kind") in DIFFUSION_PART_KINDS, st["stage_type"]
    # The Resnet*sample stages have no cross-attention (resnet-only).
    assert u["down"][0]["attn"] is False
    assert u["up"][-1]["attn"] is False


def test_unet_structure_parsed():
    ir = config_to_ir(SDXL_UNET)
    u = ir.extras["unet"]
    assert ir.num_layers == 0                       # no flat transformer stack
    assert ir.hidden_size == 1280                    # widest stage
    assert [d["channels"] for d in u["down"]] == [320, 640, 1280]
    # First down stage is a plain DownBlock2D (no cross-attention); the rest have it.
    assert u["down"][0]["attn"] is False and u["down"][1]["attn"] is True
    # transformer_layers_per_block carried through.
    assert u["down"][2]["transformers"] == 10
    assert u["downscale"] == 4                        # 3 stages -> 2**2
    assert u["cross_attention_dim"] == 2048
    assert u["down"][0]["diffusion_part_kind"] == "down_stage"
    assert u["mid"]["diffusion_part_kind"] == "mid_stage"
    assert u["up"][0]["diffusion_part_kind"] == "up_stage"
    assert {"kind": "resnet_stack", "count": 2} in u["down"][0]["components"]


def test_unet_renders_loop_and_ushape(monkeypatch):
    ir = config_to_ir(SDXL_UNET)
    assert validate_block_tree(ir) == []
    html = unfold(SDXL_UNET).to_html(standalone=True)
    assert validate_click_coupling(html) == []
    assert "SAMPLING LOOP" in html                    # hero loop reused
    assert "U-Net" in html                            # denoiser node label
    assert "Conv U-Net" in html                       # header badge
    assert "skip connections" in html                 # the U-shape view
    assert "Down stage" in html and "Up stage" in html and "Mid stage" in html
    assert "DENOISER LAYER MAP" not in html           # no flat layer map for UNet
    # Name from the model tag.
    assert ir.name == "stable-diffusion-xl-base-1.0"


def test_unet_custom_stage_is_visible_but_unapproved():
    cfg = {
        **SDXL_UNET,
        "down_block_types": ["DownBlock2D", "CustomMagicDown", "CrossAttnDownBlock2D"],
    }
    ir = config_to_ir(cfg)
    custom = ir.extras["unet"]["down"][1]
    assert custom.get("custom_label") == "CustomMagicDown"
    assert "diffusion_part_kind" not in custom
    html = unfold(cfg).to_html(standalone=True)
    assert "CustomMagicDown" in html


def test_dit_dialect_field_aliases():
    """DiT variants with different field spellings parse correctly:
    AuraFlow's num_mmdit_layers/num_single_dit_layers, and Hunyuan-DiT declaring
    hidden_size directly (no per-head attention_head_dim)."""
    aura = config_to_ir({
        "_class_name": "AuraFlowTransformer2DModel", "num_mmdit_layers": 4,
        "num_single_dit_layers": 32, "attention_head_dim": 256,
        "num_attention_heads": 8, "joint_attention_dim": 2048, "in_channels": 4,
    })
    assert aura.num_layers == 36 and aura.hidden_size == 8 * 256

    hunyuan = config_to_ir({
        "_class_name": "HunyuanDiT2DModel", "num_layers": 40,
        "num_attention_heads": 16, "hidden_size": 1408, "cross_attention_dim": 1024,
    })
    assert hunyuan.num_layers == 40 and hunyuan.hidden_size == 1408


def test_pixart_single_stream_only():
    ir = config_to_ir(PIXART)
    assert ir.num_layers == 28
    assert ir.hidden_size == 16 * 72


# --- Diffusion auto-depth harness: recursive denoiser conformance ------------
# The package bakes every drill depth into the standalone HTML up front, so
# click-coupling over the FULL html is the recursive-leaf guarantee (every
# clickable node at every depth resolves to a card). This locks the structural
# depth (cross-attn sublayer, AdaLN gates, VAE/encoder drills) across families.

_AURAFLOW = {"_class_name": "AuraFlowTransformer2DModel", "num_mmdit_layers": 4,
             "num_single_dit_layers": 32, "attention_head_dim": 256,
             "num_attention_heads": 8, "joint_attention_dim": 2048, "in_channels": 4}
_HUNYUANDIT = {"_class_name": "HunyuanDiT2DModel", "num_layers": 40,
               "num_attention_heads": 16, "hidden_size": 1408, "cross_attention_dim": 1024}

_DEPTH_FIXTURES = {"flux_mmdit": FLUX, "pixart_cross": PIXART, "sdxl_unet": SDXL_UNET,
                   "auraflow": _AURAFLOW, "hunyuandit": _HUNYUANDIT}


def _walk_blocks(blocks):
    for b in blocks or []:
        if isinstance(b, dict):
            yield b
            yield from _walk_blocks(b.get("children"))


@pytest.mark.parametrize("name", sorted(_DEPTH_FIXTURES))
def test_diffusion_recursive_depth_conforms(name):
    cfg = _DEPTH_FIXTURES[name]
    d = unfold(cfg)
    ir = d.ir
    # 1. recursion bottoms out: every block carries a registered view OR a
    #    description (a leaf) — no bare, undrillable, undescribed block.
    from model_unfolder.renderers.html.block_views.registry import VIEW_REGISTRY
    render = (ir.extras or {}).get("render") or {}
    trees = [getattr(L, "blocks", []) for L in ir.layers] + \
            [render.get("model_blocks") or [], render.get("loop_blocks") or []]
    for blocks in trees:
        for b in _walk_blocks(blocks):
            if b.get("static"):
                continue
            assert b.get("view") in VIEW_REGISTRY or b.get("description") or b.get("children"), \
                f"{name}: block {b.get('id')!r} is a bare leaf (no view/description)"
    # 2. recursive coupling: every clickable node at every drill depth → a card.
    html = d.to_html(standalone=True)
    assert validate_block_tree(ir) == []
    assert validate_click_coupling(html) == []


# Real FLUX.1-dev model_index.json wiring (the pipeline component map).
FLUX_INDEX = {
    "_class_name": "FluxPipeline",
    "_diffusers_version": "0.30.0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "transformer": ["diffusers", "FluxTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}


def test_diffusion_loader_merges_pipeline_and_denoiser(tmp_path, monkeypatch):
    """By-ID loading: model_index.json + transformer/config.json -> merged config.

    Mocks hf_hub_download so the denoiser component config and the pipeline
    component map are combined the way a real diffusers repo would yield them.
    """
    import json
    import huggingface_hub
    from model_unfolder.adapters.diffusor.loader import load_diffusion_config_by_id

    def fake_download(repo_id, filename, subfolder=None, token=None):
        if subfolder == "transformer" and filename == "config.json":
            data = FLUX
        elif filename == "model_index.json" and not subfolder:
            data = FLUX_INDEX
        else:
            raise FileNotFoundError(f"404 {subfolder}/{filename}")
        p = tmp_path / f"{subfolder or ''}_{filename.replace('/', '_')}"
        p.write_text(json.dumps(data))
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    merged = load_diffusion_config_by_id("black-forest-labs/FLUX.1-dev")
    assert merged is not None
    # Denoiser identity is preserved (used for detection + arch name)...
    assert merged["_class_name"] == "FluxTransformer2DModel"
    # ...and the pipeline wiring is merged in for skeleton naming.
    assert merged["_pipeline_class_name"] == "FluxPipeline"
    assert merged["text_encoder"][1] == "CLIPTextModel"

    ir = config_to_ir(merged)
    assert ir.num_layers == 57
    assert ir.extras["render"]["theme"] == "teal"
    assert ir.extras["diffusion"]["text_encoders"] == ["CLIP", "T5"]


def test_diffusion_loader_returns_none_for_non_diffusion(tmp_path, monkeypatch):
    """A repo with no model_index.json isn't a diffusion pipeline -> None (so the
    by-ID path falls through to the normal raw-config handling)."""
    import huggingface_hub
    from model_unfolder.adapters.diffusor.loader import load_diffusion_config_by_id

    def fake_download(repo_id, filename, subfolder=None, token=None):
        raise FileNotFoundError("404")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert load_diffusion_config_by_id("some/llm") is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# DiT coverage audit classes (2026-06: video DiTs, UNet-by-id, MoE-DiT,
# non-KL VAEs, conditioning-style variants) — synthetic configs, no network.
# ---------------------------------------------------------------------------

WAN_STYLE = {
    "_class_name": "WanTransformer3DModel", "_diffusers_version": "0.33.0",
    "dim": 1536, "ffn_dim": 8960, "num_heads": 12, "num_layers": 30,
    "attention_head_dim": 128, "text_dim": 4096, "freq_dim": 256,
    "in_channels": 16, "out_channels": 16, "patch_size": [1, 2, 2],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
}

COGVIDEO_STYLE = {
    "_class_name": "CogVideoXTransformer3DModel", "_diffusers_version": "0.31.0",
    "num_attention_heads": 48, "attention_head_dim": 64, "num_layers": 42,
    "text_embed_dim": 4096, "time_embed_dim": 512, "in_channels": 16,
    "patch_size": 2, "activation_fn": "gelu-approximate",
}


def test_dit_norm_type_resolved_from_config_not_generic():
    """A config-declared norm_type names the norm — never the generic 'Normalization'.
    diffusers DiTs state AdaLN variants (ada_norm_single / ada_norm_zero / ...), which are
    LayerNorm-based, so they resolve to LayerNorm (the adaptive modulation is shown by the
    timestep wiring). A model that declares NOTHING stays honest-'Normalization'."""
    blocks = unfold(PIXART).ir.layers[0].blocks
    norms = [b for b in blocks if b.get("kind") == "norm"]
    assert norms and all(b.get("label") == "LayerNorm" for b in norms), \
        f"ada_norm_single must resolve to LayerNorm, got {[(b['id'], b.get('label')) for b in norms]}"
    # the AdaLN modulation must be NAMED in the self-attn & FFN norm cards (the defining
    # DiT mechanism); the cross-attn norm is noted as a plain norm.
    by = {b["id"]: b for b in blocks}
    assert all("AdaLN" in by[n]["description"] for n in ("rms1", "rms2"))
    assert "plain norm" in by["xattn_norm"]["description"]
    # rms_norm config → RMSNorm; an undeclared norm stays generic (honest-unknown).
    rms = {**PIXART, "norm_type": "rms_norm"}
    assert any(b.get("label") == "RMSNorm" for b in unfold(rms).ir.layers[0].blocks if b.get("kind") == "norm")
    bare = {k: v for k, v in PIXART.items() if k not in ("norm_type", "norm_eps")}
    assert any(b.get("label") == "Normalization" for b in unfold(bare).ir.layers[0].blocks if b.get("kind") == "norm")


def test_clickable_highlight_is_image_only():
    """The Dable amber-border overlay marks clickable blocks in the IMAGE pass only —
    it is injected into the extracted svg before rasterizing and must NEVER appear in
    the shipped HTML."""
    from model_unfolder.preview import _with_clickable_highlight, _CLICKABLE_HIGHLIGHT
    svg = '<svg viewBox="0 0 10 10"><g class="uf-node" data-id="x"><rect/></g></svg>'
    out = _with_clickable_highlight(svg)
    assert out.startswith("<svg") and _CLICKABLE_HIGHLIGHT in out, "overlay must inject into the svg"
    # the shipped document never carries the overlay's amber stroke (it has its own
    # .uf-node hover/select CSS — that's the real product, distinct from this overlay).
    html = unfold(FLUX).to_html(standalone=True)
    assert "FFC400" not in html and _CLICKABLE_HIGHLIGHT not in html, "overlay leaked into shipped HTML"


def test_inspect_code_resolves_diffusion_norm_from_diffusers_source():
    """When the config is silent on the norm type (FLUX & most DiTs), `inspect_code`
    reads it from the diffusers BLOCK class (AdaLayerNormZero → LayerNorm), tier-2 —
    so the norm card stops saying 'Normalization' and is marked code-derived. Without
    the flag it stays honest-'Normalization' (config-only)."""
    silent = sorted({b["label"] for b in unfold(FLUX).ir.layers[0].blocks if b.get("kind") == "norm"})
    assert silent == ["Normalization"], silent

    import importlib.util
    if importlib.util.find_spec("diffusers") is None:
        return  # diffusers not installed — the code path can't run
    from model_unfolder.evidence.sources import resolve_source_files
    if not resolve_source_files(FLUX, source="local").files:
        return  # installed diffusers doesn't define this class — skip

    norms = [b for b in unfold(FLUX, inspect_code=True).ir.layers[0].blocks if b.get("kind") == "norm"]
    assert norms and all(b["label"] == "LayerNorm" for b in norms), \
        f"inspect_code should resolve FLUX norm to LayerNorm, got {[(b['id'], b['label']) for b in norms]}"
    assert any("read from the model code" in b.get("description", "") for b in norms), \
        "a code-resolved norm must be marked as code-derived (tier-2), not config"


def test_cross_attn_dit_has_three_sublayers_and_adaln_gates():
    """Cross-attention DiTs (PixArt/Sana/Wan/video) have THREE sublayers —
    self-attn → cross-attn(to text) → FFN — each AdaLN-gated where the source gates
    (self + FFN). MM-DiT (joint) has NO separate cross-attention sublayer."""
    d = unfold(PIXART)
    ids = [b["id"] for b in d.ir.layers[0].blocks]
    # the three-sublayer chain in order
    assert ids[:11] == ["rms1", "attn", "gate_msa", "add1",
                        "xattn_norm", "cross_attn", "add_xattn",
                        "rms2", "ffn", "gate_mlp", "add2"]
    cross = next(b for b in d.ir.layers[0].blocks if b["id"] == "cross_attn")
    # The cross-attn drill is the CANONICAL attention view, hybridised with the input
    # change (cross_attention spec: image Q, encoded-text K/V, non-cached) — no bespoke fork.
    assert cross.get("view") == "attention" and cross.get("diffusion_stage") == "cross_attention"
    xattn = cross["detail"]["attention"]
    assert xattn["cross_attention"] is True and xattn["cached"] is False
    # cross-attn carries its OWN namespaced op cards (accurate dims), incl. the text K/V node.
    cross_ids = {c["id"] for c in cross["children"]}
    assert xattn["node_prefix"] == "x_" and "x_cross_attention_states" in cross_ids
    assert {"x_q_proj", "x_k_proj", "x_scaled_scores", "x_o_proj"} <= cross_ids
    # self-attention keeps its own specific (non-namespaced) cards, untouched.
    self_attn = next(b for b in d.ir.layers[0].blocks if b["id"] == "attn")
    assert {c["id"] for c in self_attn["children"]} >= {"q_proj", "k_proj", "scaled_scores", "o_proj"}
    # AdaLN gates are Tier-2 connectors (× glyph) on self-attn + FFN (not on cross-attn) —
    # glyphs, but now clickable with a describing card (not static).
    gates = [b for b in d.ir.layers[0].blocks if b["id"] in ("gate_msa", "gate_mlp")]
    assert all(g["kind"] == "gate_mul" and not g.get("static") and g.get("description") for g in gates)
    add_x = next(b for b in d.ir.layers[0].blocks if b["id"] == "add_xattn")
    assert add_x["kind"] == "residual_add" and not add_x.get("static") and add_x.get("description")
    html = d.to_html(standalone=True)
    assert "Cross-attention to text" in html and validate_click_coupling(html) == []

    # MM-DiT (joint_attention_dim) must NOT grow a cross-attention sublayer.
    mmdit_ids = [b["id"] for b in unfold(FLUX).ir.layers[0].blocks]
    assert "cross_attn" not in mmdit_ids


def test_video_dit_detected_and_honest():
    """Transformer3DModel classes are diffusion denoisers — never the LLM
    adapter (the Wan misparse: hidden 0, no loop, fake decoder)."""
    d = unfold(WAN_STYLE)
    ir = d.to_ir()
    assert (ir["extras"].get("render") or {}).get("family") == "diffusion"
    assert ir["hidden_size"] == 1536                      # dim spelling
    assert ir["layers"][0]["ffn"]["intermediate_size"] == 8960   # ffn_dim
    html = d.to_html(standalone=True)
    # cross-attn DiT: a SEPARATE cross-attention sublayer (self → cross → FFN), not MM-DiT
    assert "Cross-Attention" in html and "Cross-attention to text" in html
    block_ids = [b["id"] for b in d.ir.layers[0].blocks]
    assert "cross_attn" in block_ids and "add_xattn" in block_ids
    assert "MM-DiT" not in html
    assert ">Frames<" in html                             # video output, not "Image"
    assert validate_click_coupling(html) == []


def test_concat_joint_video_dit_is_not_called_dual_stream():
    """text_embed_dim-style joint sequences (CogVideoX/Mochi) share Q/K/V —
    'joint', honestly, but never the dual-stream MM-DiT claim."""
    html = unfold(COGVIDEO_STYLE).to_html(standalone=True)
    assert "Joint attention — concatenated text + latent sequence" in html
    assert "MM-DiT" not in html


def test_unet_is_a_loadable_denoiser_key():
    """UNet pipelines (SD1.5/SDXL/Kandinsky) load by id — the by-id loader must
    accept 'unet' as a denoiser component, not only 'transformer'."""
    from model_unfolder.adapters.diffusor.loader import _DENOISER_KEYS
    assert "unet" in _DENOISER_KEYS


def test_moe_dit_routes_experts_not_dense():
    """HiDream-I1 style: num_routed_experts/num_activated_experts in a DiT
    block ⇒ MoE FFN with router — never silently flattened to dense."""
    cfg = {"_class_name": "HiDreamImageTransformer2DModel",
           "num_layers": 4, "num_attention_heads": 20, "attention_head_dim": 128,
           "num_routed_experts": 4, "num_activated_experts": 2,
           "joint_attention_dim": 4096}
    ir = unfold(cfg).to_ir()
    ffn = ir["layers"][0]["ffn"]
    assert ffn["kind"] == "moe" and ffn["num_experts"] == 4
    assert ffn["num_experts_per_tok"] == 2


def test_non_kl_vae_stays_honest():
    """DC-AE (Sana) spells channels decoder_block_out_channels and mixes block
    types per stage: stages render with channel chips but NO invented
    'N× ResNet' claim and no dead output-head click."""
    cfg = dict(FLUX)
    cfg["_vae_config"] = {
        "_class_name": "AutoencoderDC",
        "decoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
        "latent_channels": 32,
    }
    html = unfold(cfg).to_html(standalone=True)
    assert validate_click_coupling(html) is None or validate_click_coupling(html) == []
    i = html.find('data-card-id="vae_decoder_block_1"')
    assert i > 0
    assert "× ResNet" not in html[i:i + 600]


def test_swiglu_video_dit_ffn_is_gated():
    """Mochi declares activation_fn swiglu — the FFN is gated, drawn with the
    gate path, not a plain MLP."""
    cfg = dict(COGVIDEO_STYLE, _class_name="MochiTransformer3DModel",
               activation_fn="swiglu")
    ffn = unfold(cfg).to_ir()["layers"][0]["ffn"]
    assert ffn["gated"] is True


def test_video_latent_shape_uses_declared_temporal_geometry():
    """CogVideoX declares sample_frames + temporal_compression_ratio: the z_T
    shape gains the frames axis ((49-1)/4+1 = 13 latent frames).  Models that
    declare nothing temporal keep the plain channel note — never invented."""
    cfg = dict(COGVIDEO_STYLE, sample_height=60, sample_width=90,
               sample_frames=49, temporal_compression_ratio=4)
    html = unfold(cfg).to_html(standalone=True)
    assert "16 × 13 × 30 × 45" in html
    html_wan = unfold(WAN_STYLE).to_html(standalone=True)
    assert "shape [16 channels]" in html_wan


def test_scheduler_step_renders_a_clean_combine_not_floating_ops():
    """The scheduler step opens the update rule as a purpose-built graph: the
    denoiser's prediction is scaled and combined with z_t into ONE ⊕ → z_{t-1}.
    (Regression: the declared-ops chain floated/duplicated the ⊕ because the
    combine merges the primary latent with a side-scaled input — same failure mode
    the self-conditioning view hit.)"""
    from model_unfolder.adapters.diffusor.parser import _scheduler_geom
    from model_unfolder.adapters.diffusor.blocks import _scheduler_step_view

    # flow-matching family (FLUX / FlowMatchEuler): velocity prediction, Euler step
    sv = _scheduler_step_view({"scheduler": "Flow Match Euler", "scheduler_flow_matching": True})
    assert sv["view"] == "scheduler_step"
    s = sv["detail"]["scheduler_step"]
    assert s["sym"] == "v̂" and "v̂" in s["step_label"] and "+" in s["step_label"]

    html = unfold(FLUX).to_html(standalone=True)
    assert "from denoiser" in html and "current latent" in html
    assert validate_click_coupling(html) == []            # no floating/orphan ⊕

    # epsilon family from a declared prediction_type → z_t − σ_t·ε̂
    eps = _scheduler_step_view({"scheduler": "DDIM", "scheduler_train_timesteps": 1000,
                                "scheduler_prediction_type": "epsilon"})
    assert eps["view"] == "scheduler_step"
    assert eps["detail"]["scheduler_step"]["sym"] == "ε̂"
    # an undeclared scheduler keeps the honest prose card (no fabricated step)
    assert _scheduler_step_view({}) == {}


def test_scheduler_display_names_handle_acronym_runs():
    """CogVideoXDDIMScheduler must not camel-split into 'Cog Video XDDIM' —
    oddballs live in typing.yaml's scheduler_display overrides."""
    from model_unfolder.adapters.diffusor.parser import _scheduler_geom
    assert _scheduler_geom({"scheduler": ["diffusers", "CogVideoXDDIMScheduler"]})["scheduler"] == "CogVideoX DDIM"
    assert _scheduler_geom({"scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]})["scheduler"] == "Flow Match Euler"
    assert _scheduler_geom({"scheduler": ["diffusers", "UniPCMultistepScheduler"]})["scheduler"] == "UniPC Multistep"


def test_ops_region_declared_side_inputs():
    """A declared {"kind": "input"} op is a side source: wired only by `from`,
    never advancing the implicit chain — and it gets a derived card."""
    from model_unfolder.labels import cards_from_region
    from model_unfolder.opgraph import ops_region
    r = ops_region([
        {"id": "pred", "kind": "input", "label": "prediction"},
        {"id": "scale", "kind": "elementwise", "fn": "mul", "from": ["pred"]},
        {"id": "step", "kind": "elementwise", "fn": "add", "from": ["hidden", "scale"]},
    ], rid="s")
    assert r.merges() == ["step"]
    assert [(e.src, e.dst) for e in r.edges] == [("pred", "scale"), ("hidden", "step"), ("scale", "step")]
    ids = [c["id"] for c in cards_from_region(r)]
    assert ids == ["pred", "scale", "step"]               # side input has a card; hidden doesn't


def test_sampling_loop_json_matches_html_nodes():
    """The JSON `sampling_loop` and the HTML loop view are two projections of ONE
    declared edge set (adapters/diffusor/blocks.diffusion_loop_*), so their node
    sets must be identical — the structural anti-drift gate."""
    import re
    d = unfold(FLUX)
    j = d.to_json()["sampling_loop"]
    json_nodes = {n["id"] for n in j["nodes"]}

    html = d.to_html(standalone=True)
    seg = html[html.index("SAMPLING LOOP"):]
    loop_svg = re.search(r"<svg.*?</svg>", seg, re.S).group(0)
    html_nodes = set(re.findall(r'data-id="([^"]+)"', loop_svg))

    assert html_nodes == json_nodes, (
        f"loop drift — only in HTML: {html_nodes - json_nodes}; "
        f"only in JSON: {json_nodes - html_nodes}")

    # Every edge endpoint is a real node; the recurrence has its one back-edge.
    ids = json_nodes
    for e in j["edges"]:
        assert e["from"] in ids and e["to"] in ids, f"dangling edge {e}"
    backs = [(e["from"], e["to"]) for e in j["edges"] if e.get("back_edge")]
    assert backs == [("scheduler", "latent")], backs
    # Fan-in (connectors) and fan-out (splitters) are derived, present, honest.
    assert {c["at"] for c in j["connectors"]} == {"latent", "denoiser"}
    assert {s["at"] for s in j["splitters"]} == {"denoiser", "prompt"}


def test_text_cond_rail_requires_attention_text_signal():
    """The text->attention K/V side-rail must be drawn ONLY when the config says
    attention consumes text (a joint/cross/text-embed dim or a stream split) — a
    text *encoder* alone must not draw it (it may condition purely via AdaLN, as
    in a plain class-conditional DiT). Pins the FAIL-1 fix for Ideogram4-style
    DiTs whose conditioning is an LLM feature with no attention-text dim."""
    from model_unfolder.adapters.diffusor.parser import _conditioning_side_blocks

    ids = lambda blks: {b["id"] for b in blks}
    # No attention-text signal: only the AdaLN rail, no text_cond.
    plain = _conditioning_side_blocks(text_in_attention=False, pooled_in_adaln=False,
                                      guidance=False)
    assert ids(plain) == {"adaln_cond"}
    assert "pooled text" not in plain[0]["description"]

    # Attention consumes text: the text_cond rail appears.
    joint = _conditioning_side_blocks(text_in_attention=True, pooled_in_adaln=True,
                                      guidance=False)
    assert ids(joint) == {"adaln_cond", "text_cond"}
    assert "pooled text" in joint[0]["description"]


def test_diffusion_json_does_not_leak_llm_io_fields():
    """A denoiser has no token vocabulary or LM head; the expanded JSON's
    dimensions + io must not leak vocab_size / tie_word_embeddings / token_ids /
    token_embedding (the IR carries vocab_size=0 only for param honesty). Pins
    the FAIL-2 fix."""
    j = unfold(FLUX).to_json()
    dims = j["dimensions"]
    assert "vocab_size" not in dims and "tie_word_embeddings" not in dims
    assert "in_channels" in dims and "hidden_size" in dims

    io = j["io"]
    assert io["input"]["kind"] == "noisy_latent"
    assert "token_embedding" not in io and "lm_head" not in io
    assert io["output"]["kind"] == "noise_prediction"

    # A real LLM still reports the token fields — the branch is diffusion-only.
    ld = unfold(LLAMA).to_json()
    assert "vocab_size" in ld["dimensions"]
    assert ld["io"]["input"]["kind"] == "token_ids"


def test_dit_ffn_undeclared_structure_is_honest_not_fabricated():
    """When NOTHING declares the FFN activation — neither the config NOR a
    ``class_defaults.yaml`` entry — the inner structure (activation AND gating) is
    rendered honestly as undeclared (gated=null, structure_declared=false), never a
    fabricated non-gated GELU MLP.  Uses IDEO_STYLE: a custom DiT class with no
    class default.  (Flux, whose model class DOES fix the activation, is the
    code-derived case — see ``test_flux_ffn_activation_is_code_derived_gelu``.)
    Pins WEAK-3 / honest-unknown.  A real LLM (declares its activation) is
    unaffected."""
    f = unfold(IDEO_STYLE).to_json()["layer_groups"][0]["ffn"]
    assert f.get("activation") is None          # no fabricated default
    assert f.get("gated") is None               # gating undeclared
    assert f.get("structure_declared") is False
    assert f.get("activation_assumed") is True
    assert not f.get("activation_from_class")   # not code-derived either — truly unknown
    # The FFN block's card says exactly that — never a fabricated GELU shape.
    import re
    html = unfold(IDEO_STYLE).to_html(standalone=True)
    m = re.search(r'data-card-id="ffn"[^>]*>.*?uf-card-desc">(.*?)</div>', html, re.S)
    assert m and "does not declare the gating or activation" in m.group(1)
    assert "GELU" not in m.group(1)   # activation is unknown → never fabricated
    # LLAMA declares its activation — gating/activation are real facts, not flagged.
    lf = unfold(LLAMA).to_json()["layer_groups"][0]["ffn"]
    assert lf["activation"] == "silu" and "activation_assumed" not in lf
    assert lf["gated"] is True


def test_flux_ffn_activation_is_code_derived_gelu():
    """Flux declares no FFN activation in config, but its model class fixes it
    (``FeedForward(activation_fn="gelu-approximate")`` / ``nn.GELU(approximate=
    "tanh")`` after proj_mlp, both mult=4, NON-gated).  We surface that from
    ``class_defaults.yaml`` — MARKED code-derived — so the FFN drill is informative
    (Linear → GELU → Linear) AND consistent with the single-stream MLP lane,
    instead of a pale opaque box.  This is the inverse of the honest-unknown pin
    above: a fact we CAN cite to the model class is surfaced, never fabricated
    silently.  In diffusers the activation_fn name fully specifies the FFN, so the
    non-gated shape is derived from it (a "*glu" name would mark it gated)."""
    import re
    ir = config_to_ir(FLUX)

    def _is_single(L):
        return "single-stream" in str((L.attention.variant or {}).get("tag") or "").lower()

    for L in (next(x for x in ir.layers if _is_single(x)),
              next(x for x in ir.layers if not _is_single(x))):
        assert L.ffn.activation == "gelu-approximate"
        assert L.ffn.gated is False                 # derived from the non-glu name
        assert L.ffn.activation_from_class is True   # code-derived, marked

    # JSON parity + the dense drill is real (Linear → GELU → Linear), not opaque.
    fj = unfold(FLUX).to_json()["layer_groups"][0]["ffn"]
    assert fj.get("activation") == "gelu-approximate"
    assert fj.get("activation_from_class") is True and fj.get("gated") is False
    dual = next(L for L in ir.layers if not _is_single(L))
    ffn_block = next(b for b in dual.blocks if b.get("id") == "ffn")
    assert [c["id"] for c in ffn_block["children"]] == ["up_proj", "activation", "down_proj"]

    html = unfold(FLUX).to_html(standalone=True)
    # The activation card marks the fact code-derived; the single-stream lane marks
    # it too and shows the clean math name (GELU), never the backend spelling.
    act = re.search(r'data-card-id="activation"[^>]*>.*?uf-card-desc">(.*?)</div>', html, re.S)
    assert act and "fixed in the model class" in act.group(1)
    ss = re.search(r'data-card-id="ss_mlp"[^>]*>.*?uf-card-desc">(.*?)</div>', html, re.S)
    assert ss and "code-derived" in ss.group(1) and "GELU" in ss.group(1)


# Ideogram-4-style DiT: custom class, LLM-feature conditioning, an AdaLN dim, a
# CFG twin, and NO declared activation / attention-text dim — exercises FAIL-1,
# WEAK-3 and GAP-4 offline (no network).
IDEO_STYLE = {
    "_class_name": "Ideogram4Transformer2DModel",
    "num_layers": 2, "num_attention_heads": 4, "attention_head_dim": 64,
    "in_channels": 16, "intermediate_size": 256,
    "adaln_dim": 512, "llm_features_dim": 53248,
    "unconditional_transformer": ["diffusers", "Ideogram4Transformer2DModel"],
}


def test_ideogram_style_dit_captures_declared_facts():
    """GAP-4: adaln_dim / llm_features_dim are captured; the CFG twin is a NOTE
    (by-design advisory, not a config gap — must NOT raise "partial config"); the
    AdaLN rail carries its dim — and (FAIL-1) no text->attention rail appears
    without an attention-text signal."""
    ir = config_to_ir(IDEO_STYLE)
    diff = (ir.extras or {}).get("diffusion") or {}
    assert diff["adaln_dim"] == 512 and diff["llm_features_dim"] == 53248
    # The CFG twin is advisory: it lives in notes, never in warnings, so a
    # faithful parse isn't mislabelled "⚠ partial config".
    assert any("unconditional_transformer" in n for n in ir.notes)
    assert not any("unconditional_transformer" in w for w in ir.warnings)

    side = {b["id"]: b for b in ir.layers[0].blocks if b.get("lane")}
    assert "text_cond" not in side                       # no attention-text dim
    assert "AdaLN dim 512" in (side["adaln_cond"].get("facts") or [])
    # WEAK-3: undeclared activation flagged assumed, gating undeclared (honest).
    assert ir.layers[0].ffn.activation_assumed is True
    assert ir.layers[0].ffn.gated is None
    # Norm kind is not declared (only a bare norm_eps) — don't assert RMS/Layer.
    assert ir.layers[0].norm_kind == "unknown"
    # FAIL-1/GAP-4: llm_features_dim is recognized as PRE-BLOCK text fusion, so
    # the attention description says text is fused before the stack — never the
    # wrong "class / timestep enters only through AdaLN".
    desc = ir.layers[0].attention.variant["desc"]
    assert "fused once before the stack" in desc
    assert "class / timestep) enters only" not in desc


def test_unet_view_shows_text_conditioning_rail():
    """The U-net denoiser diagram must SHOW the encoded text entering the
    cross-attention stages — a 'Encoded text' source broadcasting into the
    CrossAttn stages — not just the latent U-path.  A clickable, carded node."""
    import re
    html = unfold(SDXL_UNET).to_html(standalone=True)
    den = re.search(r'<svg[^>]*aria-label="[^"]*U-net denoiser".*?</svg>', html, re.S).group(0)
    assert "Encoded text" in den                       # the text source is drawn
    assert 'data-id="unet_text_cond"' in den           # clickable node
    assert 'data-card-id="unet_text_cond"' in html     # backing card (coupling)
    assert validate_click_coupling(html) == []


def test_encoded_text_box_drills_into_the_concat_view():
    """Clicking the 'Encoded text' source opens a view showing HOW the encoders
    make the cross-attention K/V: each CLIP's width feeding one concat (‖) into the
    2,048-d K/V (768 + 1,280 = 2,048).  This needs the per-encoder configs, which
    the by-ID loader fetches — exercised here with the SDXL fixture's configs."""
    cfg = dict(SDXL_UNET, _text_encoder_configs={
        "text_encoder": {"_class_name": "CLIPTextModel", "num_hidden_layers": 12,
                         "hidden_size": 768, "num_attention_heads": 12,
                         "intermediate_size": 3072, "hidden_act": "quick_gelu",
                         "max_position_embeddings": 77, "vocab_size": 49408},
        "text_encoder_2": {"_class_name": "CLIPTextModelWithProjection",
                           "num_hidden_layers": 32, "hidden_size": 1280,
                           "num_attention_heads": 20, "intermediate_size": 5120,
                           "hidden_act": "gelu", "max_position_embeddings": 77,
                           "vocab_size": 49408, "projection_dim": 1280},
    })
    import re
    d = unfold(cfg)
    html = d.to_html(standalone=True)
    i = html.find('data-card-id="unet_text_cond"')
    assert i >= 0
    seg = html[i:i + 8000]
    assert "<svg" in seg                                  # the box opens a real view
    assert "768-d" in seg and "1,280-d" in seg            # each encoder's width
    assert "K/V (2,048)" in seg                           # the concatenated K/V width
    assert "768 + 1,280 = 2,048" in html                  # the sum, in the op card prose
    # the ‖ concat operator is itself clickable, drilling into a card for the op
    assert 'data-id="text_concat_op"' in seg
    assert 'data-card-id="text_concat_op"' in html
    assert "torch.cat over the feature axis" in html
    assert validate_click_coupling(html) == []


def test_unet_stage_drills_show_per_stage_dims():
    """Each stage's drill must show ITS OWN width/heads, not the first stage's.
    Block ids are scoped per stage, so the panel's per-depth dedup can't collapse
    every stage's ResNet/attention card into one (the '320 ch everywhere' bug)."""
    import re
    # SDXL's real per-stage head counts come from attention_head_dim = [5,10,20]
    # (when num_attention_heads is unset, it IS the head count per stage).
    cfg = dict(SDXL_UNET, attention_head_dim=[5, 10, 20])
    html = unfold(cfg).to_html(standalone=True)

    def view_svg(cid: str) -> str:
        i = html.find(f'data-card-id="{cid}"')
        m = re.search(r'<svg.*?</svg>', html[i:i + 9000], re.S) if i >= 0 else None
        return m.group(0) if m else ""

    # down 320 / 640 / 1,280; up 1,280 / 640 / 320 — each its own card
    for sid, ch in [("unet_down_0", 320), ("unet_down_1", 640), ("unet_down_2", 1280),
                    ("unet_up_0", 1280), ("unet_up_2", 320)]:
        assert f"in ({ch:,} ch)" in view_svg(f"{sid}__resnet"), (sid, ch)
    # mid block: two resnets (pre/post), both at 1,280 ch
    assert f"in (1,280 ch)" in view_svg("unet_mid__resnet_pre"), "unet_mid__resnet_pre"
    assert f"in (1,280 ch)" in view_svg("unet_mid__resnet_post"), "unet_mid__resnet_post"
    # transformer head counts differ per stage (640→10 heads, 1,280→20 heads) —
    # not collapsed to the first cross-attn stage's count
    assert "10 heads" in html[html.find('data-card-id="unet_down_1__transformer"'):][:600]
    assert "20 heads" in html[html.find('data-card-id="unet_down_2__transformer"'):][:600]
    assert validate_click_coupling(html) == []


def test_unet_resnet_block_has_no_repeat_pill():
    """A ResNet block is ONE residual cell, not a repeated stack — its view must
    NOT show a '× N' / '× 1' repeat pill (the per-stage layers_per_block repeat is
    shown one level up, on the stage). The stage view still shows its real pill,
    and an unknown-count stack still legitimately reads '× N'."""
    import re
    html = unfold(SDXL_UNET).to_html(standalone=True)

    def view_svg(cid: str) -> str:
        i = html.find(f'data-card-id="{cid}"')
        m = re.search(r'<svg.*?</svg>', html[i:i + 9000], re.S)
        return m.group(0) if m else ""

    rn = view_svg("unet_down_1__resnet")                  # ids are scoped per stage
    assert "GroupNorm" in rn and "Conv 3" in rn          # ops still drawn
    assert "× N" not in rn and "× 1" not in rn            # but no repeat pill
    # the stage one level up keeps its REAL repeat (SDXL down_1 = 2 ResNet blocks)
    assert "× 2" in view_svg("unet_down_1")


def test_unet_attention_inner_ops_are_described_and_clickable():
    """Drilling into the UNet self/cross attention must give EVERY inner op a card
    (a description) and make it clickable — Q/K/V projections, scaled scores,
    softmax, apply-V, concat, output projection — plus cross-attention's distinct
    encoded-text K/V source.  The shared SDPA op cards use neutral wording (correct
    for both self and cross, which share op ids); the source difference is the
    cross_attention_states node."""
    html = unfold(SDXL_UNET).to_html(standalone=True)
    for op in ("q_proj", "k_proj", "v_proj", "scaled_scores", "attn_softmax",
               "attn_apply_v", "concat_heads", "o_proj"):
        assert f'data-id="{op}"' in html, f"{op} not clickable"
        assert f'data-card-id="{op}"' in html, f"{op} has no card"
    # cross-attention's distinguishing source node is described
    assert 'data-card-id="cross_attention_states"' in html
    assert "what makes it cross-attention" in html
    # shared op cards are source-neutral (not baked to one side)
    i = html.find('data-card-id="k_proj"')
    assert "the input" in html[i:i + 400]
    assert validate_click_coupling(html) == []


def test_unet_text_conditioning_propagates_through_drill_levels():
    """The encoded text is shown entering at EVERY level it's relevant, not just
    the deepest: the denoiser U (rail into cross-attn stages), the stage drill
    (into the Transformer block), the Transformer-block drill (beside the
    cross-attention sub-block), and the attention mechanism (the K/V node)."""
    import re
    html = unfold(SDXL_UNET).to_html(standalone=True)

    def card(cid: str) -> str:
        i = html.find(f'data-card-id="{cid}"')
        assert i >= 0, cid
        nxt = html.find('data-card-id=', i + 10)
        return html[i:(nxt if nxt > 0 else i + 9000)]

    den = re.search(r'<svg[^>]*aria-label="[^"]*U-net denoiser".*?</svg>', html, re.S).group(0)
    assert "Encoded text" in den                       # L1: the U-view rail
    assert "Encoded text" in card("unet_down_1")        # L2: stage → Transformer block
    assert "Encoded text" in card("unet_down_1__transformer")   # L3: beside Cross-attention
    assert "Encoded text" in card("unet_down_1__crossattn")     # L4: the attention K/V node
    # The two-CLIP origin is visible (768 + 1,280 → 2,048 concatenated), so the
    # single box doesn't read as "the second CLIP vanished".
    assert "2× CLIP" in den
    assert validate_click_coupling(html) == []


def test_unet_cross_attention_drill_shows_text_entering():
    """Opening a UNet Transformer block's Cross-attention (text) must render
    DIFFERENTLY from Self-attention: cross-attention pulls K/V from the encoded
    text, so its drilled diagram shows an external 'Encoded text' node feeding
    K/V — self-attention (K/V from the latent) does not.  Pins the bug where both
    opened the identical self-attention view."""
    html = unfold(SDXL_UNET).to_html(standalone=True)

    def card_seg(cid: str) -> str:
        i = html.find(f'data-card-id="{cid}"')
        assert i >= 0, cid
        nxt = html.find('data-card-id=', i + 10)
        return html[i:(nxt if nxt > 0 else i + 9000)]

    self_seg = card_seg("unet_down_1__selfattn")      # ids scoped per stage
    cross_seg = card_seg("unet_down_1__crossattn")
    assert "Encoded text" in cross_seg          # external text K/V enters
    assert "Encoded text" not in self_seg        # self-attention stays on the latent


def test_unet_hero_denoiser_labeled_unet_not_dit():
    """The hero loop's denoiser label must come from the parsed loop block, not a
    hardcoded 'DiT' — a UNet model (SDXL) must read 'U-Net Denoiser'. Pins the
    SDXL-shows-DiT regression."""
    import re
    html = unfold(SDXL_UNET).to_html(standalone=True)
    seg = html[html.index("SAMPLING LOOP"):]
    loop_svg = re.search(r"<svg.*?</svg>", seg, re.S).group(0)
    texts = re.findall(r"<text[^>]*>(.*?)</text>", loop_svg)
    assert "U-Net" in texts and "DiT Denoiser" not in texts
    # A real DiT still reads DiT.
    fseg = unfold(FLUX).to_html(standalone=True)
    fsvg = re.search(r"<svg.*?</svg>", fseg[fseg.index("SAMPLING LOOP"):], re.S).group(0)
    assert "DiT Denoiser" in re.findall(r"<text[^>]*>(.*?)</text>", fsvg)


def test_unet_view_stages_clickable_carded_and_clean():
    """The UNet U-shape passes the block gates: every stage box is a clickable
    node with a backing card (B.4/B.2), numbers live on card chips not the box
    (house style), skips use a concat connector, and conv-in/out are solid (no
    light bookend). Pins the SDXL UNet-view rework."""
    import re
    d = unfold(SDXL_UNET)
    html = d.to_html(standalone=True)
    usvg = re.search(r'<svg[^>]*aria-label="[^"]*U-net[^"]*".*?</svg>', html, re.S).group(0)

    node_ids = set(re.findall(r'data-id="(unet_[^"]+)"', usvg))
    card_ids = set(re.findall(r'data-card-id="(unet_[^"]+)"', html))
    assert node_ids and node_ids <= card_ids                 # B.4: every node carded
    assert validate_click_coupling(html) == []

    # House style: box labels are stage names only — no chips on the diagram.
    box_labels = re.findall(r'Caveat[^>]*>([^<]+)</text>', usvg)
    assert not any("ch" in l or "ResNet" in l for l in box_labels)
    assert "Down stage" in box_labels and "Conv in" in box_labels

    # Skips use a concat connector (circles); no cryptic ↓2/↑2 marks or "skip
    # connections" caption on the diagram (the cards/description carry that).
    assert usvg.count("<circle") >= 3
    assert "↓2" not in usvg and "↑2" not in usvg and "skip connections" not in usvg

    # No light-green accent on the conv bookends.
    from model_unfolder.renderers.html.theme import C
    assert C["bg_inner"] not in usvg

    # B.2 + B.7: the cards describe each stage and cite their config signature.
    den = [b for b in ((d.to_ir()["extras"] or {}).get("render") or {})["loop_blocks"]
           if b["id"] == "denoiser"][0]
    for c in den["children"]:
        assert c.get("description")
    joined = " ".join(c["description"] for c in den["children"])
    assert "block_out_channels" in joined and "down_block_types" in joined


def test_unet_hero_loop_has_arrows():
    """Regression: the UNet render spec must carry the SAME declared loop_edges
    as the DiT one, or the hero sampling loop draws no arrows."""
    import re
    d = unfold(SDXL_UNET)
    render = (d.to_ir()["extras"] or {}).get("render") or {}
    assert render.get("loop_edges") and render.get("loop_region")
    html = d.to_html(standalone=True)
    hero = re.search(r"<svg.*?</svg>", html[html.index("SAMPLING LOOP"):], re.S).group(0)
    assert hero.count("marker-end") >= 8         # noise→latent→denoiser⟳sched→vae→image + cond


def test_unet_stage_drills_into_resnet_and_transformer_reusing_openers():
    """A stage drills into a ResNet block and a Transformer block (B.1). The
    ResNet block opens its residual cell; the Transformer block opens
    self-attention / cross-attention / feed-forward — each REUSING the canonical
    attention / FFN opener (not a bespoke leaf). Skips merge solid."""
    import re
    d = unfold(SDXL_UNET)
    html = d.to_html(standalone=True)

    # ResNet block + Transformer block are real clickable, carded nodes.
    nodes = set(re.findall(r'data-id="([^"]+)"', html))
    cards = set(re.findall(r'data-card-id="([^"]+)"', html))
    # stage-level blocks are scoped by stage id; the channel-agnostic resnet ops
    # (GroupNorm/Conv/temb-inject/residual) stay shared (unscoped).
    for nid in ("unet_down_1__resnet", "unet_down_1__transformer", "unet_down_1__selfattn",
                "unet_down_1__crossattn", "unet_down_1__ff",
                "unet_op_norm1", "unet_op_temb", "unet_op_residual"):
        assert nid in nodes and nid in cards, nid

    # self/cross-attn reuse the ATTENTION opener; FF reuses the FFN opener.
    assert html.count('aria-label="stable-diffusion-xl-base-1.0 attention"') >= 2
    assert "feed-forward block" in html
    assert validate_click_coupling(html) == []

    # The transformer DEPTH chip is labelled as depth (a Transformer2D of depth
    # 10), not as a count of transformer blocks ("10× Transformer") nor "cross-attn ×N".
    assert "10-layer Transformer" in html
    assert "10× Transformer" not in html and "cross-attn ×10" not in html

    usvg = re.search(r'<svg[^>]*aria-label="[^"]*U-net denoiser".*?</svg>', html, re.S).group(0)
    assert usvg.count("stroke-dasharray") == 0          # no dotted skips
    assert usvg.count("<circle") >= 3                   # one concat connector per up stage


def test_unet_mid_block_is_resnet_transformer_resnet_sandwich():
    """UNetMidBlock2DCrossAttn.forward() is resnets[0] → attn[0] → resnets[1]:
    a sandwich, not a paired loop.  The mid stage view must show two separate resnet
    cards (pre/post) rather than a [ResNet, Transformer] × 2 repeat frame which would
    imply a non-existent second Transformer."""
    import re
    html = unfold(SDXL_UNET).to_html(standalone=True)
    nodes = set(re.findall(r'data-id="([^"]+)"', html))
    cards = set(re.findall(r'data-card-id="([^"]+)"', html))
    # Both pre and post resnets are present as separate carded nodes
    assert "unet_mid__resnet_pre" in nodes and "unet_mid__resnet_pre" in cards
    assert "unet_mid__resnet_post" in nodes and "unet_mid__resnet_post" in cards
    assert "unet_mid__transformer" in nodes and "unet_mid__transformer" in cards
    # NO paired-repeat pill: the mid view is a plain sequential chain
    def mid_svg(cid: str) -> str:
        i = html.find(f'data-card-id="{cid}"')
        m = re.search(r'<svg.*?</svg>', html[i:i + 9000], re.S) if i >= 0 else None
        return m.group(0) if m else ""
    mid_stage_svg = mid_svg("unet_mid")
    # No "× 2" frame in the mid stage view (sandwich is not a repeated pair)
    assert "× 2" not in mid_stage_svg, "mid stage must not show a ×2 repeat badge"
    assert validate_click_coupling(html) == []


def test_unet_resnet_view_shows_timestep_injection_and_correct_residual():
    """ResnetBlock2D.forward() injects temb between conv1 and norm2 (⊕ node), and
    the residual bypass goes around the ENTIRE cell from the block's raw input.
    Both must be present in the rendered ResNet drill view."""
    import re
    html = unfold(SDXL_UNET).to_html(standalone=True)
    nodes = set(re.findall(r'data-id="([^"]+)"', html))
    cards = set(re.findall(r'data-card-id="([^"]+)"', html))
    # Temb injection node is drawn and carded
    assert "unet_op_temb" in nodes, "⊕ timestep node must be drawn in ResNet drill"
    assert "unet_op_temb" in cards, "⊕ timestep node must have a card"
    # The ResNet drill SVG contains the ⊕ timestep label
    def view_svg(cid: str) -> str:
        i = html.find(f'data-card-id="{cid}"')
        m = re.search(r'<svg.*?</svg>', html[i:i + 9000], re.S) if i >= 0 else None
        return m.group(0) if m else ""
    rn = view_svg("unet_down_1__resnet")
    assert "Timestep emb" in rn, "Timestep source must appear in ResNet drill"
    assert validate_click_coupling(html) == []
