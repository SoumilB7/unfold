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
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.structural_debt import pending_projection_paths

# facebook/musicgen-small config.json — the composite contract: bare slots
# text_encoder / audio_encoder / decoder, each declaring its own model_type.

# stabilityai/stable-audio-open-1.0 — 1-D audio-latent DiT + oobleck VAE.


# ---- U-A: composite wrapper walk -------------------------------------------

from test_support import MUSICGEN_SMALL, STABLE_AUDIO


def test_musicgen_composite_parses_the_decoder_stack():
    context = ParseContext.build(MUSICGEN_SMALL)
    ir = config_to_ir(MUSICGEN_SMALL, parse_context=context)
    assert ir.num_layers == 24
    assert ir.hidden_size == 1024
    assert ir.vocab_size == 2048
    layer = ir.layers[0]
    assert layer.attention.num_heads == 16
    assert layer.ffn.intermediate_size == 4096       # ffn_dim alias
    assert layer.ffn.activation == "gelu"
    assert layer.ffn.gated is False
    mixer = context.facts.typed["decoder.attention.mixer_schedule"]
    assert mixer.status == "code_and_config"
    assert mixer.value == ("ordinary_attention",) * 24
    # The primary mixer fact cites only the exact self-attention occurrence;
    # the second proven attention occurrence remains owned by the independent
    # additive cross-attention schedule.
    assert context.facts.typed[
        "decoder.attention.cross_attention_schedule"].value \
        == ("additive_cross",) * 24


def test_musicgen_cross_attention_is_construction_proven_and_additive():
    """The decoder sub-config even says add_cross_attention: false — the
    unconditional ``encoder_attn`` construction is the truth (all layers).
    And it is ADDITIVE: the layer KEEPS self-attention and gains a cross
    sublayer (unlike mllama's replacement schedule)."""
    ir = config_to_ir(MUSICGEN_SMALL)
    for layer in ir.layers:
        assert layer.cross_attention is not None          # additive sublayer
        assert layer.attention.cross_attention is False   # self-attn KEPT
        # Cross-attention construction does not prove the independent self-mask
        # lane. MusicGen's exact mask transport is still unresolved, so U8
        # withholds it instead of reviving the former decoder/config default.
        assert layer.attention.mask == "unknown"
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
    # The compact block carries only the typed cross-attention role.  Prompt
    # provenance is independently typed by the side-state block above; the
    # renderer must not recover it by searching prose.
    cross_block = next(b for b in ir.layers[0].blocks if b["id"] == "cross_attn")
    assert "Vision" not in " ".join(cross_block["label"])
    assert "Cross-Attention" in " ".join(cross_block["label"])


def test_musicgen_conditioning_tower_rides_the_universal_roundtrip():
    ir = config_to_ir(MUSICGEN_SMALL)
    cond = ir.extras["modalities"]["inputs"]["conditioning"]
    encoder = cond["encoder"]
    assert "model_type" not in encoder
    assert encoder["source_component"] == "text_encoder"
    assert encoder["source_owner"] == "T5Stack"
    assert encoder["hidden_size"] == 768 and encoder["num_layers"] == 12
    # The task-specific AutoModelForTextEncoding source registry selects the
    # exact T5EncoderModel occurrence (not the config's generic T5Model), so
    # the encoder has one repeated stage rather than a false encoder+decoder
    # union.  Mechanisms the exact block reader has not closed remain unknown.
    assert len(encoder["variants"]) == 1
    variant = encoder["variants"][0]
    assert variant["repeat"] == 12
    assert variant["attention_kind"] == "mha"
    assert variant["ffn_gated"] is None
    assert variant["norm_kind"] is None
    assert encoder["num_attention_heads"] == 12
    assert encoder["final_norm_kind"] == "rmsnorm"
    assert encoder["position_encoding"] == {
        "kind": "relative_bias", "application": "attention_side_input"}
    # U9's ReaderResult is the architectural authority, but U14 still owns the
    # FactLedger/receipt migration for the legacy modality-extras surface.  Do
    # not let that later obligation disappear merely because unrelated parser
    # reads make the compatibility ``unread`` view order-sensitive.
    assert ("root.conditioning", "text_encoder.num_heads") \
        in pending_projection_paths()
    # Width projection is shape-REQUIRED (768 -> 1024).
    projector = cond["projector"]
    assert projector["kind"] == "linear_projector"
    assert (projector["in_features"], projector["out_features"]) == (768, 1024)
    assert projector["source_field"] == "enc_to_dec_proj"
    assert projector["source_evidence"]["in_width_path"] \
        == ["text_encoder", "d_model"]
    assert projector["source_evidence"]["out_width_path"] \
        == ["decoder", "hidden_size"]
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
    cb = ir.extras["fact_provenance"]["decoder.codebook_streams"]["value"]
    assert cb["num"] == 4
    assert cb["embeddings_summed"] is True
    assert cb["heads_stacked"] is True
    blocks = {b["id"]: b for b in ir.extras["render"]["model_blocks"]}
    # Labels stay COUNT-FREE (label-lint law: numbers live on cards as chips).
    assert blocks["tok_text"]["label"] == ["Parallel token", "streams"]
    assert "4" not in " ".join(blocks["tok_text"]["label"])
    assert blocks["embed"]["title"] == "Parallel embedding banks (×4, summed)"
    assert "4 × (2,048 vocab)" in blocks["embed"]["facts"]
    assert blocks["lm_head"]["label"] == ["Parallel token", "heads"]
    assert "4 × (1,024 → 2,048)" in blocks["lm_head"]["facts"]
    assert "×4" in blocks["lm_head"]["title"]


def test_musicgen_architecture_emits_codebook_fact_receipt():
    from model_unfolder import unfold
    from model_unfolder.evidence.receipts import join_obligation_receipts
    diagram = unfold(MUSICGEN_SMALL)
    extras = diagram.to_ir()["extras"]
    receipts = [
        receipt
        for event in diagram.render_events()
        for receipt in event.receipts
        if receipt.fact_key == "codebook_streams"
    ]
    assert receipts
    assert {(item.owner, item.mechanism, item.surface,
             item.structural_target, item.node_ids)
            for item in receipts} == {(
                "decoder", "codebook_streams", "html",
                "codebook_streams", ("tok_text", "embed", "lm_head"))}
    obligations = [
        item for item in extras["config_access"]["projection_obligations"]
        if item["target"]["key"] == "codebook_streams"]
    assert len(obligations) == 1
    joined = join_obligation_receipts(
        obligations, receipts, extras["fact_provenance"],
        context_token=receipts[0].context_token)
    assert joined["findings"] == []


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
    # ``sample_size`` is stored by the root class but does not participate in
    # the denoiser's forward shape; U10 therefore refuses to turn it into a
    # rendered temporal extent.  Most importantly, it also never squares it.
    assert facts == "VAE-space latent"
    assert "x 1,024" not in facts and "x 1024" not in facts
    # U12 still identifies the Oobleck codec facts, but U10 does not claim the
    # codec's output domain on the denoiser's behalf.
    assert loop["image"]["label"] == "Output"
    assert loop["image"]["title"] == "Output domain unresolved"
    vae_facts = " ".join(loop["vae_decode"]["facts"] or [])
    assert "44,100 Hz" in vae_facts and "2-channel audio" in vae_facts
    assert "temporal ↑2·4·4·8·8" in vae_facts


def test_stable_audio_has_no_fabricated_patchify():
    ir = config_to_ir(STABLE_AUDIO)
    blocks = {b["id"]: b for b in ir.extras["render"]["model_blocks"]}
    assert blocks["embed"]["label"] == "Input operations"
    assert blocks["embed"]["detail"]["operations"] == [
        "linear", "activation", "reshape", "conv1d", "concat"]
    assert blocks["final_rms"]["label"] == "Output operations"
    assert blocks["final_rms"]["detail"]["operations"] == ["linear", "conv1d"]
    assert blocks["lm_head"]["label"] == "Denoiser output"
    assert "patchify" not in str(blocks).lower()
    # A config-declared patch size cannot author a Patchify operation when the
    # exact source projection does not prove one either.
    import test_support as td
    pix = config_to_ir(td.PIXART)
    pix_blocks = {b["id"]: b for b in pix.extras["render"]["model_blocks"]}
    assert pix_blocks["embed"]["label"] == "Input unresolved"
    assert pix_blocks["embed"]["detail"]["operations"] == []


def test_stable_audio_gqa_kv_heads_from_the_declared_alias():
    ir = config_to_ir(STABLE_AUDIO)
    attn = ir.layers[0].attention
    assert attn.num_heads == 24
    assert attn.head_dim == 64
    # The self-attention construction omits kv_heads; the external API's
    # default is not indexed, so U10 keeps it unknown rather than borrowing
    # the cross lane's value.
    assert attn.num_kv_heads is None
    assert attn.kind is None
    cross = ir.layers[0].cross_attention
    assert cross is not None
    assert (cross.kind, cross.num_heads, cross.num_kv_heads, cross.head_dim) \
        == ("gqa", 24, 12, 64)
    facts = ir.extras["fact_provenance"]
    primary_fact = facts[
        "root.denoiser.stacks[0].attention[0].diffusion_attention_head_protocol"]
    cross_fact = facts[
        "root.denoiser.stacks[0].attention[1].diffusion_attention_head_protocol"]
    assert primary_fact["status"] == cross_fact["status"] == "code_and_config"
    assert primary_fact["value"]["num_kv_heads"] is None
    assert cross_fact["value"] == {
        "kind": "gqa", "num_heads": 24, "num_kv_heads": 12,
        "projection_mode": None, "output_gate": None}
    # An unresolved FFN does not erase these exact lanes. The shared
    # unknown-wiring projection retains both attentions and their external K/V
    # rail without inventing norms, residual taps, or an FFN mechanism.
    assert [block["id"] for block in ir.layers[0].blocks] == [
        "cross_attention_states", "attn", "cross_attn",
        "wiring_unresolved", "ffn"]
    assert ir.layers[0].ffn.kind is None

    geometry = facts["root.denoiser.diffusion_bookend_geometry"]
    assert geometry["status"] == "code_and_config"
    rows = geometry["value"]
    assert any(row == {
        "application_role": "conditioning_input",
        "operation_kind": "linear", "dimension_role": "input_width",
        "value": 768,
    } for row in rows)
    assert any(row["application_role"] == "state_input"
               and row["dimension_role"] == "input_width"
               and row["value"] == 1536 for row in rows)
    assert any(row["application_role"] == "state_input"
               and row["dimension_role"] == "input_width"
               and row["value"] == 256 for row in rows)
    embed = next(block for block in ir.extras["render"]["model_blocks"]
                 if block["id"] == "embed")
    assert embed["detail"]["dimensions"] == list(rows)
    assert ir.extras["config_access"].get("unparsed_fields") is None
    assert ir.extras.get("config_consumed_unreceipted") is None


# ---- render health (both witnesses) ------------------------------------------

def test_audio_witnesses_render_and_couple():
    for cfg in (MUSICGEN_SMALL, STABLE_AUDIO):
        d = unfold(cfg)
        html = d.to_html()
        assert validate_click_coupling(html) == []
        assert d.wiring_problems() == []


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
    # This bare config declares no exact architecture/root occurrence.  The old
    # whole-file scan selected a post-norm-looking class anyway; the exact U7
    # reader must keep rival model stages unknown instead of first-picking one.
    assert ir.layers[0].norm_placement == "unknown"

    ir2 = config_to_ir(SpeechT5Config())
    joined2 = " ".join(ir2.warnings)
    assert "encoder half" in joined2 and "6-layer decoder" in joined2
    assert "speech-decoder prenet" in joined2
