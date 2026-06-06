"""Diffusion (DiT/MMDiT) adapter — detection, IR shape, blue theme, render health.

These use real, public diffusers config values (FLUX.1-dev transformer,
PixArt-alpha transformer) as plain dicts — no network, no model code executed.
"""
import pytest

from model_unfolder import unfold, config_to_ir
from model_unfolder.adapters.diffusor import parser as diffusor
from model_unfolder.adapters.transformer import parser as transformer
from model_unfolder.block_schema import validate_block_tree, validate_click_coupling


# Real FLUX.1-dev transformer/config.json values.
FLUX = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.30.0",
    "attention_head_dim": 128,
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
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
    # DiT attention is full multi-head self-attention without rotary.
    attn = ir.layers[0].attention
    assert attn.kind == "mha"
    assert attn.num_kv_heads == attn.num_heads == 24
    assert attn.no_rope is True
    # DiT FFN is a non-gated GELU-family MLP.
    assert ir.layers[0].ffn.gated is False


def test_diffusion_render_spec_is_blue():
    ir = config_to_ir(FLUX)
    render = ir.extras["render"]
    assert render["family"] == "diffusion"
    assert render["theme"] == "blue"


def test_blue_theme_applied_and_does_not_leak_to_transformer():
    diffusion_html = unfold(FLUX).to_html(standalone=True)
    assert "#1E5FB0" in diffusion_html        # blue block colour present
    assert "#0F6E56" not in diffusion_html     # teal block colour absent

    # A transformer rendered after a diffusion render must be teal again
    # (use_theme restores the palette).
    transformer_html = unfold(LLAMA).to_html(standalone=True)
    assert "#0F6E56" in transformer_html
    assert "#1E5FB0" not in transformer_html


def test_denoiser_skeleton_is_drawn():
    html = unfold(FLUX).to_html(standalone=True)
    for label in ("Noisy latent", "Patchify", "AdaLN-Out", "Unpatchify", "VAE decode"):
        assert label in html, f"skeleton stage {label!r} not drawn"


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
    assert ir.extras["render"]["theme"] == "blue"
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
