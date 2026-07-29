"""Audio-gen support (SURGICAL_PLAN_AUDIO U-A/U-B/U-D) — composite seq2seq
wrapper walk, multi-codebook streams, and 1-D audio-DiT geometry.

Fixtures are real public config values (facebook/musicgen-small's config.json
sub-dicts; Stable Audio Open 1.0's values, which are the installed diffusers
class's own constructor defaults) as plain dicts — no network.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_unfolder import unfold, config_to_ir
from model_unfolder.block_schema import validate_click_coupling

# facebook/musicgen-small config.json — the composite contract: bare slots
# text_encoder / audio_encoder / decoder, each declaring its own model_type.

# stabilityai/stable-audio-open-1.0 — 1-D audio-latent DiT + oobleck VAE.


# ---- U-A: composite wrapper walk -------------------------------------------

from test_support import MUSICGEN_SMALL, STABLE_AUDIO


def test_musicgen_composite_parses_the_decoder_stack():
    ir = config_to_ir(MUSICGEN_SMALL)
    assert ir.num_layers == 24
    assert ir.hidden_size == 1024
    assert ir.vocab_size == 2048
    layer = ir.layers[0]
    assert layer.attention.num_heads == 16
    assert layer.ffn.intermediate_size == 4096       # ffn_dim alias
    assert layer.ffn.activation == "gelu"
    assert layer.ffn.gated is False


def test_musicgen_cross_attention_is_construction_proven_and_additive():
    """The decoder sub-config even says add_cross_attention: false — the
    unconditional ``encoder_attn`` construction is the truth (all layers).
    And it is ADDITIVE: the layer KEEPS self-attention and gains a cross
    sublayer (unlike mllama's replacement schedule)."""
    ir = config_to_ir(MUSICGEN_SMALL)
    for layer in ir.layers:
        assert layer.cross_attention is not None          # additive sublayer
        assert layer.attention.cross_attention is False   # self-attn KEPT
        assert layer.attention.mask == "causal"
    cross = ir.layers[0].cross_attention
    assert cross.cross_attention is True and cross.mask == "full"
    assert cross.cross_kv_source == "encoded prompt states (the conditioning encoder tower)"
    # The drawn cell has BOTH sublayers, in constructed order, and the side
    # states feed the CROSS block.
    ids = [b["id"] for b in ir.layers[0].blocks]
    assert ids.index("attn") < ids.index("cross_attn") < ids.index("ffn")
    side = next(b for b in ir.layers[0].blocks if b["id"] == "cross_attention_states")
    assert side["title"] == "Encoded prompt states"
    assert side["feeds"] == "cross_attn"
    assert side["view"] == "conditioning_path"
    # The cross block names the prompt, never "Vision".
    cross_block = next(b for b in ir.layers[0].blocks if b["id"] == "cross_attn")
    assert "Vision" not in " ".join(cross_block["label"])
    assert "Prompt" in " ".join(cross_block["label"])


def test_musicgen_conditioning_tower_rides_the_universal_roundtrip():
    ir = config_to_ir(MUSICGEN_SMALL)
    cond = ir.extras["modalities"]["inputs"]["conditioning"]
    encoder = cond["encoder"]
    assert encoder["model_type"] == "t5"
    assert encoder["hidden_size"] == 768 and encoder["num_layers"] == 12
    # Deep tower through the ONE encoder round-trip (embedded ≡ standalone).
    # T5Model exposes rival encoder/decoder stages here, so norm stays omitted
    # until the stage address is proven. The FFN's dense, ungated storage is
    # source-proven independently; its activation declaration is not yet bound
    # to this exact occurrence and must therefore remain absent/unknown.
    assert encoder["sub_model"]["groups"]
    assert "norm" not in encoder
    assert encoder["gated"] is False
    assert "activation" not in encoder
    group_ffn = encoder["sub_model"]["groups"][0]["ffn"]
    assert group_ffn["kind"] == "dense"
    assert group_ffn["gated"] is False
    assert group_ffn["activation"] is None
    # Width projection is shape-REQUIRED (768 -> 1024).
    assert cond["projector"] == {"kind": "code_defined_projector",
                                 "in_features": 768, "out_features": 1024}
    assert cond["tokens"]["kind"] == "cross_attention_states"
    assert ir.extras["modalities"]["fusion"]["kind"] == "cross_attention"


def test_musicgen_codec_slot_is_a_stated_omission():
    ir = config_to_ir(MUSICGEN_SMALL)
    assert any("Audio codec (encodec) not drawn" in w for w in ir.warnings)


def test_bare_decoder_flag_is_never_mistaken_for_a_composite():
    """A non-dict / model_type-less ``decoder`` value must not unwrap or fire
    the composite machinery — the slot's evidence is the child's own
    declaration, never the bare key."""
    cfg = {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
           "num_hidden_layers": 2, "hidden_size": 128, "num_attention_heads": 8,
           "intermediate_size": 256, "vocab_size": 1000, "rms_norm_eps": 1e-5,
           "decoder": True, "encoder": {"width": 3}}
    ir = config_to_ir(cfg)
    assert ir.num_layers == 2 and ir.hidden_size == 128
    assert not (ir.extras.get("modalities") or {})
    assert not any(layer.attention.cross_attention for layer in ir.layers)


# ---- U-B: multi-codebook streams --------------------------------------------

def test_musicgen_codebook_streams_are_construction_proven():
    ir = config_to_ir(MUSICGEN_SMALL)
    cb = ir.extras["codebooks"]
    assert cb["num"] == 4 and cb["vocab_per_book"] == 2048
    assert cb["embeddings_summed"] is True
    assert cb["heads_stacked"] is True
    blocks = {b["id"]: b for b in ir.extras["render"]["model_blocks"]}
    # Labels stay COUNT-FREE (label-lint law: numbers live on cards as chips).
    assert blocks["tok_text"]["label"] == ["Audio tokens", "(codebooks)"]
    assert "4" not in " ".join(blocks["tok_text"]["label"])
    assert blocks["embed"]["title"] == "Codebook embeddings (×4, summed)"
    assert "4 × (2,048 vocab)" in blocks["embed"]["facts"]
    assert blocks["lm_head"]["label"] == ["Audio-token", "heads"]
    assert "4 × (1,024 → 2,048)" in blocks["lm_head"]["facts"]
    assert "×4" in blocks["lm_head"]["title"]


def test_single_stream_decoders_stay_byte_stable():
    """No num_codebooks ⇒ the classic blocks, verbatim (corpus lock support)."""
    cfg = {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
           "num_hidden_layers": 2, "hidden_size": 128, "num_attention_heads": 8,
           "intermediate_size": 256, "vocab_size": 1000, "rms_norm_eps": 1e-5}
    ir = config_to_ir(cfg)
    assert "codebooks" not in ir.extras
    blocks = {b["id"]: b for b in ir.extras["render"]["model_blocks"]}
    assert blocks["tok_text"]["label"] == "Tokenized text"
    assert blocks["lm_head"]["label"] == "Linear output layer"


# ---- U-D: 1-D audio-DiT geometry ---------------------------------------------

def test_stable_audio_latent_is_one_dimensional_never_a_square_grid():
    ir = config_to_ir(STABLE_AUDIO)
    assert ir.num_layers == 24
    assert ir.hidden_size == 24 * 64
    loop = {b["id"]: b for b in ir.extras["render"]["loop_blocks"]}
    facts = " ".join(loop["latent"]["facts"] or [])
    assert "64 ch × 1,024 latent frames" in facts
    assert "x 1,024" not in facts and "x 1024" not in facts
    assert loop["image"]["label"] == "Waveform"
    vae_facts = " ".join(loop["vae_decode"]["facts"] or [])
    assert "44,100 Hz" in vae_facts and "2-channel audio" in vae_facts
    assert "temporal ↑2·4·4·8·8" in vae_facts


def test_stable_audio_has_no_fabricated_patchify():
    ir = config_to_ir(STABLE_AUDIO)
    blocks = {b["id"]: b for b in ir.extras["render"]["model_blocks"]}
    assert blocks["embed"]["label"] == "Input projection"
    assert "no patchify" in blocks["embed"]["description"]
    assert blocks["lm_head"]["label"] == "To latent channels"
    # An image DiT with a DECLARED patch keeps its Patchify labels.
    import test_support as td
    pix = config_to_ir(td.PIXART)
    pix_blocks = {b["id"]: b for b in pix.extras["render"]["model_blocks"]}
    assert pix_blocks["embed"]["label"] == "Patchify"


def test_stable_audio_gqa_kv_heads_from_the_declared_alias():
    ir = config_to_ir(STABLE_AUDIO)
    attn = ir.layers[0].attention
    assert attn.num_heads == 24
    assert attn.num_kv_heads == 12          # num_key_value_attention_heads


# ---- render health (both witnesses) ------------------------------------------

def test_audio_witnesses_render_and_couple():
    for cfg in (MUSICGEN_SMALL, STABLE_AUDIO):
        d = unfold(cfg)
        html = d.to_html()
        assert validate_click_coupling(html) == []
        assert d.wiring_problems() == []


# ---- TTS honesty: classic post-norm is sequential, never parallel ------------

def test_classic_post_norm_reads_sequential_post_never_parallel():
    """The original-Transformer idiom — wrapped ``norm(residual + x)`` (VITS)
    and the split-statement form ``x = residual + sub(x); x = norm(x)``
    (SpeechT5) — must classify as sequential post-norm.  Both misread as
    PARALLEL (+double) before: the add hid inside the norm's argument, fusing
    the sublayers into one segment (caught on the VITS TTS gallery)."""
    from model_unfolder.evidence.patterns import _classify_topology

    # Wrapped (VITS): attention, norm(+add)  |  ffn, norm(+add)
    wrapped = ["attention", "norm", "add", "ffn", "norm", "add"]
    assert _classify_topology(wrapped) == {
        "norm_placement": "post", "parallel_residual": False}

    # Split statements (SpeechT5): bare sublayer, add, then a leading norm
    # that CLOSES the previous sublayer.
    split = ["attention", "add", "norm", "ffn", "add", "norm"]
    assert _classify_topology(split) == {
        "norm_placement": "post", "parallel_residual": False}

    # Pre-norm families keep their leading norms (llama shape).
    pre = ["norm", "attention", "add", "norm", "ffn", "add"]
    assert _classify_topology(pre) == {
        "norm_placement": "pre", "parallel_residual": False}

    # True parallel residual (GPT-J): one segment, both sublayers.
    parallel = ["norm", "attention", "ffn", "add"]
    assert _classify_topology(parallel)["parallel_residual"] is True


def test_vits_layer_topology_from_installed_source():
    """End-to-end on the installed modeling source (offline)."""
    import os
    import pytest
    import transformers
    from model_unfolder.evidence.patterns import decoder_layer_topology_from_files
    path = os.path.join(os.path.dirname(transformers.__file__),
                        "models/vits/modeling_vits.py")
    if not os.path.exists(path):
        pytest.skip("vits source not installed")
    assert decoder_layer_topology_from_files([path]) == {
        "norm_placement": "post", "parallel_residual": False}


def test_tts_silent_drops_are_stated_omissions():
    """A config declaring flows / duration predictor / HiFiGAN ladder /
    speech pre-post-nets (VITS, SpeechT5 spellings) must WARN about what is
    not drawn — the silent-omission class, retired per family."""
    # The installed config classes ARE the witnesses (their defaults declare
    # the components; hub objects also resolve depth via attribute_map).
    from transformers import SpeechT5Config, VitsConfig
    ir = config_to_ir(VitsConfig())
    joined = " ".join(ir.warnings)
    assert "not drawn" in joined
    for label in ("normalizing flows", "duration predictor", "HiFiGAN"):
        assert label in joined
    # And the parallel/double misread stays dead: VITS is sequential post-norm.
    assert ir.layers[0].norm_placement == "post"

    ir2 = config_to_ir(SpeechT5Config())
    joined2 = " ".join(ir2.warnings)
    assert "encoder half" in joined2 and "6-layer decoder" in joined2
    assert "speech-decoder prenet" in joined2
