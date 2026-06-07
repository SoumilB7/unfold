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
    # DiT FFN is a non-gated GELU-family MLP.
    assert ir.layers[0].ffn.gated is False


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


def test_vae_decoder_has_a_drill_view():
    """VAE decode opens its own view, built from the real VAE config (channels,
    upsampling) the loader fetched."""
    ir = config_to_ir(FLUX)
    vae_block = next(b for b in ir.extras["render"]["loop_blocks"] if b["id"] == "vae_decode")
    assert vae_block.get("view") == "vae_decoder"
    assert vae_block["detail"]["block_out_channels"] == [128, 256, 512, 512]
    html = unfold(FLUX).to_html(standalone=True)
    # Real decoder stages drawn compactly: 8x upscale (3 doublings), 128->3 output head.
    assert "Decoder block" in html and "Output image head" in html
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


@pytest.mark.parametrize("cfg", [FLUX, PIXART])
def test_diffusion_blocks_and_clicks_valid(cfg):
    ir = config_to_ir(cfg)
    assert validate_block_tree(ir) == []
    html = unfold(cfg).to_html(standalone=True)
    assert validate_click_coupling(html) == []


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
