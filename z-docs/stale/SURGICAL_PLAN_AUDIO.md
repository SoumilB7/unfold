# SURGICAL PLAN — AUDIO-GEN + TTS SUPPORT (U-A … U-F)
*(2026-07-07 · recon-verified at file:line against installed transformers/
diffusers AND our seams · companions: AUDIO_SUPPORT_MAP.md (the why/what),
PROJECT_CONTEXT Part 13, TOWER_CENSUS.md.  Laws in force: evidence-never-
identity · tri-state · vocab in everchanging/ · general only · Soumil
commits.  SCOPE GATE: this extends the recorded scope line — do not start
until Soumil blesses.)*

**Recon headline finds (they shape the plan):**
1. `evidence/sources.py:306` `_component_configs` descends ONLY fields named
   `*_config` — MusicGen's composite keys (`text_encoder`/`audio_encoder`/
   `decoder`, configuration_musicgen.py:113-122) are INVISIBLE to the
   per-component file walk today.
2. `blocks.py:196-198` FABRICATES a square latent grid (`{ch} × {side} x
   {side}`) whenever sample_size is scalar and H/W absent — correct for 2D
   image DiTs by domain, a fabrication for 1D audio (StableAudio: 64 ×
   1024 × 1024 from a 1-D length!).  Latent today, live the moment an audio
   DiT parses.  Fix FIRST in U-D.
3. `StableAudioDiTModel` already matches the `DiT` class marker
   (typing.yaml) and `AudioLDM2UNet2DConditionModel` matches `is_unet` —
   family detection is free; geometry/conditioning facts are the work.
4. The talker warning (transformer/parser.py:575-577) is the ONLY current
   speech-gen acknowledgement — the honest-omission precedent to retire
   family by family.

**STATE (2026-07-07, Soumil blessed the scope):** U-D0 + U-A + U-B + U-D core
LANDED (witness tests in tests/test_audio_composite.py; coverage corpus gained
codec_lm + dit_audio so every net — nested conformance, token ratchet, view
coverage — runs them; first-iteration galleries in
unfold-pkg/previews/audio_first_iterations_2026-07-07/).  REMAINING: U-C codec
tower → U-E vocoder tail → U-D polish (global-prepend reader, hub witness) →
U-F (deferred).

**ORDER (original):** U-D0 → U-A → U-B → U-C → U-D → U-E → U-F.  Each unit: tests + full suite + 25-fixture zero-drift sweep +
witness render with Sable/Dable; bless new witnesses into the corpus.

---

## U-D0 — the 1D-latent honesty guard — ✅ LANDED 2026-07-07
*(evidence-gated square: declared patchify or conv-UNet family; test locks the
1-D form + PixArt/UNet squares; landed suite-green, corpus zero-drift.)*

**Site:** `adapters/diffusor/blocks.py:180-202` (`diffusion_loop_blocks`
latent_shape).  **Change:** the square inference `side = sample // patch`
may only fire when 2D-ness is EVIDENCED (patch_size present, or H/W fields,
or a spatial patchify in source); a scalar `sample_size` with `patch_size`
absent falls to the honest `"{in_ch} channels · {sample} frames"` 1-D form.
**Test:** StableAudio-shaped geom → no `×side x side`; SD3/PRX geoms
byte-identical (they declare patch/sample 2D-ness).  **Drift: zero** on the
corpus (all 2D models keep their evidence path).

## U-A — composite/seq2seq wrapper walk (MusicGen shape) — ✅ LANDED 2026-07-07
*(As built: composite_slots.yaml vocabulary + loader; _component_configs walks
declared bare slots gated on the child's own model_type (per-slot source
binding: text_encoder→modeling_t5.py, audio_encoder→modeling_encodec.py);
_unwrap_text walks main-role slots; cross-attention = construction reader
(decoder_cross_attention_all_layers_from_files) proving the UNCONDITIONAL
encoder_attn build — and it is ADDITIVE: LayerSpec.cross_attention carries the
extra sublayer spec, the pre-norm cell draws self-attn → cross-attn → FFN in
constructed order (the replacement-style first draw dropped self-attention —
caught by BOTH the pixel pass and the nested-conformance net); side block +
fusion + labels name "prompt", never "Vision"; T5 tower panel rides the ONE
encoder round-trip, LIFTED to the adapter-neutral model_unfolder/
encoder_panel.py (diffusor aliases kept; parity tests green); ownership
qualified per slot so the tower's drills diff against T5's own source.
EnCodec slot = stated omission warning until U-C.)*

**Recon facts:** MusicgenConfig sub-config KEYS `text_encoder`/
`audio_encoder`/`decoder` (configuration_musicgen.py:113-122, coerced in
__post_init__; `is_encoder_decoder=True`).  Our seams: `TEXT_WRAPPER_KEYS`
(transformer/common.py:49-52) + `_unwrap_text` (parser.py:416-445);
`_component_configs` `*_config`-suffix filter (sources.py:285-310); the
cross side-state gate is vision-only (parser.py:817-820).

**Changes:**
1. **Component-slot vocabulary → everchanging YAML** (new
   `transformer/composite_slots.yaml`): slot name → role
   (`text_encoder=encoder_tower · audio_encoder=codec · decoder=main`),
   consumed by BOTH `_unwrap_text`-style unwrap (main = `decoder`) and
   `_component_configs` (descend DECLARED slots whose value is a dict/config
   with a `model_type`, not only `*_config` names — general fix, also
   future-proofs other composite families).
2. Decoder parse: `is_encoder_decoder`+cross-attn construction (every
   MusicgenDecoderLayer builds `encoder_attn`, modeling_musicgen.py:311) —
   the cross schedule = ALL layers via CONSTRUCTION evidence (the U4
   cross-attn reader generalizes: unconditional `encoder_attn` field on the
   layer class → all-layers membership); side-state source wording:
   "encoded text prompt (T5 tower)" — generalize the vision-only gate at
   :817-820 into declared-encoder-slot presence.
3. Encoder tower panel: reuse the EXISTING universal round-trip
   (`_normalize_encoder_config` path, diffusor/parser.py:2007+ — lift the
   entry so the TRANSFORMER side can invoke it for a declared encoder slot;
   the projection `enc_to_dec_proj` (modeling:1211-1215) becomes the
   entry-projection chip, existing idiom).

**Guards:** slot vocabulary is DATA; unwrap depth caps stay; a composite
missing its decoder slot refuses honestly.  **Witnesses:**
facebook/musicgen-small (hub config), MAGNeT config (same family).
**Drift: zero** (new family; corpus untouched).  **Tests:** slot-walk unit
(fake composite), per-component files include decoder+text_encoder sources,
cross schedule = all layers from construction, tower panel present.

## U-B — multi-codebook streams — ✅ LANDED 2026-07-07
*(As built: decoder_codebook_streams_from_files proves ModuleList-of-Embedding
SUMMED in forward + ModuleList-of-Linear STACKED (structural, no names);
K = config's own num_codebooks; extras["codebooks"] + K-fan render (Audio
tokens ×K books / Codebook embeddings ×K summed / Audio-token heads ×K) —
all only-when-present, single-stream decoders byte-stable.)*

**Recon facts:** K embeddings ModuleList SUMMED (modeling_musicgen.py:
437-439, forward :505-518); K lm_heads ModuleList STACKED (:649-651, :735);
config `num_codebooks=4`, `audio_channels` (configuration:46-47).  Render
seam: `decoder_model_blocks` single embed/head (blocks/model.py:265-337);
multi-head precedent `mtp_head_block` (:21-84).

**Changes:**
1. IR: `extras["codebooks"] = {n, vocab_per_book, audio_channels}` —
   config-declared VALUES; code-proven SHAPE via a reader
   `decoder_codebook_streams_from_files`: an EMBEDDING-role ModuleList whose
   forward SUMS indexed members → K-summed input; a HEAD-role ModuleList
   stacked → K-head fan (construction evidence, tri-state None).
2. Render: embed block gains "×K codebooks (summed)" sub/chips; lm_head →
   a K-fan block (the mtp_head_block pattern: one box "Audio-token heads
   ×K", card lists per-book vocab); delay/interleave pattern = a token-strip
   chip on the hero ("delay pattern" — VALUE from config
   `audio_channels`/K, wording generic).
3. All three projections + only-when-present emission (byte-stability).

**Witnesses:** musicgen-small (K=4), stereo variant (audio_channels=2).
**Drift: zero** (fact absent on every current model).  **Tests:** reader on
real modeling file (K embeddings summed → verdict), spec emission
only-when-present, K-fan block ids/cards couple.

## U-C — codec/RVQ tower (EnCodec) — NEXT (not started; the stated-omission
warning from U-A covers the gap honestly meanwhile)

**Recon facts:** EncodecModel = encoder+decoder+quantizer (modeling_encodec
.py:485-497); SEANet ladders (:285-342: init conv → per-ratio resnet×N +
strided conv; decoder mirrors with ConvTranspose1d); RVQ = ModuleList of
quantizers, encode SUBTRACTS residual per stage (:430-437), decode SUMS
(:440-447); LSTM (:236) residual.

**Changes:**
1. `encodec`-shaped configs stop refusing: a CODEC adapter-branch (inside
   the transformer adapter's custom parts or a small `codec` special-part):
   hero = Encoder ladder → RVQ quantizer → Decoder ladder; per-stage
   channels/ratios from config (constructor record: `upsampling_ratios`,
   `num_filters`, `num_quantizers`, `codebook_size`).
2. The RVQ FILL-IN diagram (the MoE-router-style designed view): stage-k
   "codebook lookup → subtract residual" loop frame ×K, decode-side "sum
   stage outputs" — drawn from the construction+loop evidence above.
3. LSTM boxes: HONEST opaque ("recurrent context — recorded C1 scope"),
   never a fabricated attention cell.
4. Census: add "codec ladder + RVQ" to TOWER_CENSUS §4; when MusicGen's
   composite carries `audio_encoder`, the codec renders as its tower panel.

**Witnesses:** facebook/encodec_24khz standalone + inside musicgen-small.
**Drift: zero.**  **Tests:** codec parse (stages/ratios match config), RVQ
K-loop present, LSTM opaque not attention, musicgen composite embeds it.

## U-D — audio-DiT geometry (Stable Audio) — ✅ LANDED 2026-07-07 (core)
*(As built: audio_vae_fields typing vocab → _audio_latent_domain off the
pipeline's own oobleck declaration; 1-D latent chip ("64 ch × 1,024 latent
frames"), Waveform output, audio-worded noise/VAE prose; patchify/unpatchify
EVIDENCE-GATED (no declared patch + audio → "Input projection"/"To latent
channels", PixArt keeps Patchify); GQA cross via num_key_value_attention_heads
alias; oobleck ladder = decoder_channels × channel_multiples rail + Hz/
channels/temporal-↑ chips.  OPEN in-unit: global-prepend conditioning dialect
(reader not written — no config-only claim made), witness is the installed
class-defaults fixture: the HUB repo is license-gated (Soumil: accept the
stable-audio-open license to swap in the hub witness).)*

**Recon facts:** StableAudioDiTBlock = pre-LN, self-attn WITH rotary + GQA
(`num_key_value_attention_heads`), cross-attn EVERY block, FF swiglu
(stable_audio_transformer.py:64-133); conditioning is a GLOBAL PREPENDED
token (global_proj + time, :332-341, sliced :370) — NOT AdaLN; entry
Conv1d+Linear (:254-255, :334-338), NO patchify; config sample_size=1024 is
a 1-D length, no patch/H/W (:209-223).  Oobleck VAE: Snake1d + 1D
ladders, config `downsampling_ratios`/`channel_multiples` — none of the 2D
KL fields `_vae_geom` reads (diffusor/parser.py:1766-1806 → None today).

**Changes:**
1. U-D0 shipped the latent-shape guard.  Geometry: recognize the 1-D axis
   from EVIDENCE (`audio_channels`/`sampling_rate` in the pipeline VAE
   config + no patchify in DiT source) → `geom["audio"]` (the `video`
   precedent), latent chip "{ch} ch × {len} frames", loop output node
   "Waveform" (with U-E's tail when a vocoder exists; StableAudio's VAE IS
   the waveform decoder — tail stays VAE→Waveform).
2. Patchify/Unpatchify nodes (blocks.py:1119-1170) become EVIDENCE-GATED:
   drawn only when the DiT source patchifies (proj_in Conv with
   kernel=patch>1 or explicit patchify call); StableAudio draws
   "Conv1d + Linear in" / "Linear + Conv1d out" from its own entry/exit ops.
3. Conditioning: a "global conditioning token (prepended)" mechanism —
   NEW dialect beside AdaLN/gate-via-norm, read from source (global_proj +
   cat before the stack); seconds_start/seconds_total chips via
   config_facts.yaml rows.
4. Cross-attn every block: falls out of A3's per-site machinery + U-A's
   schedule generalization; GQA-in-cross (kv_heads) already an
   AttentionSpec fact.
5. Oobleck ladder: extend `_vae_geom` with the 1-D vocabulary
   (`downsampling_ratios`, `channel_multiples`, `decoder_input_channels`,
   `audio_channels`) → a 1-D up-ladder via a `vae_up_stage` 1-D variant
   ("↑r× temporal", Snake activation named from class evidence).

**Witnesses:** stabilityai/stable-audio-open-1.0.  **Drift: zero** (all
gates evidence-based; 2D corpus keeps its paths — locked by the sweep).
**Tests:** geom (no square grid, audio axis), patchify-gate negative
(StableAudio) + positive (SD3), global-cond dialect reader, oobleck ladder
from config, GQA cross facts.

## U-E — mel + vocoder tail (AudioLDM/2, Tango, MusicLDM; SpeechT5/VITS) — NEXT
*(TTS check 2026-07-08 landed the honesty FLOOR ahead of the unit:
(1) classic-post-norm classifier fix — ``norm(residual+x)`` wrapped AND
split-statement forms read sequential-post, killing the VITS parallel+double
FABRICATION and SpeechT5's false "pre" (locks in test_audio_composite);
(2) stated-omission warnings from the ``undrawn_component_fields`` vocabulary
(VITS flows/duration/HiFiGAN; SpeechT5 pre/post-nets) + flat-seq2seq
"encoder half only" warning + unproven-cross-schedule warning;
(3) raw-config loader fallback marker ("does not recognize") — Parler-TTS
now parses BY ID through the U-A composite walk (24×1024, ×9 books declared,
T5 tower; cross/K-fan honestly unproven until its package source is readable).
U-E proper still owes: the drawn decoder half + speech pre/post-nets, the
HiFiGAN ladder archetype, vocoder loader slots, mel loop tail.
LEDGER: classic post-norm draws the OLMo template (norm before ⊕; truth is
norm AFTER the add) — needs a ``post_classic`` cell variant; TTS output head
label ("Linear output layer" + input vocab) is the decoder-only template
speaking — U-E retires it for enc-only/seq2seq pages; Stable Audio hero's
decorative 2-D grid glyph should get an audio variant.)*

**Recon facts:** AudioLDM2 pipeline slots incl. `projection_model`,
`language_model` (GPT-2!), `vocoder` (SpeechT5HifiGan)
(pipeline_audioldm2.py:195-223); our loader knows only
transformer/unet/vae/text-encoders (loader.py:20,87-110); loop tail is
image-only (blocks.py:336-368); SpeechT5HifiGan = conv_pre → upsampler
ConvTranspose1d ModuleList → resblocks → conv_post
(modeling_speecht5.py:2970-3008); VITS embeds its HiFiGAN internally
(modeling_vits.py:1249).

**Changes:**
1. Loader: declared-component slots `vocoder`, `projection_model`,
   `language_model` (+ their configs) — spellings → everchanging YAML.
2. Loop-hero tail variant: when a `vocoder` component is DECLARED →
   `vae_decode` relabels "Latent → mel spectrogram", then `vocoder` node
   (drill = the HiFiGAN ladder archetype: conv_pre → ×N transposed-conv
   upsample stages → MRF resblocks → conv_post) → output "Waveform".
3. AudioLDM2's GPT-2 bridge = the EXISTING recursive sub-model machinery
   (an embedded decoder-only LM panel) + the ProjectionModel as an
   entry-projection card (two linears + SOS/EOS chips).
4. VITS/SpeechT5 honesty upgrade: their component walk (U-A vocabulary)
   surfaces flow/duration/vocoder as towers — flows = honest-opaque box v1
   ("normalizing flow — structure not yet drawn"), duration predictor =
   op-chain, HiFiGAN = the same vocoder archetype; retires today's
   misleading-partial parse (probed: they draw one component and silently
   drop the rest).

**Witnesses:** cvssp/audioldm2, declare-lab/tango2 (same shape),
microsoft/speecht5_tts (+ its separate vocoder), a VITS checkpoint.
**Drift: zero.**  **Tests:** slot loader, tail-variant gating on DECLARED
vocoder, HiFiGAN ladder from source, GPT-2 bridge panel parity, VITS no
longer silent-drops (warning or tower per component).

## U-F — masked-iteration loop hero (MAGNeT) — DEFERRED: the MAGNeT hub repos
are audiocraft-native (NO config.json — probed 2026-07-07), so there is no
hub-config witness; revisit when a transformers-format masked-decoding
witness exists.

**Recon facts:** `block_diffusion_loop_blocks` (blocks/model.py:86-263) is
the loop-hero precedent; MAGNeT = MusicGen family, bidirectional +
iterative masked decoding (generation-time loop).
**Changes:** generalize the block-diffusion layout into a declared
"iterative masked decoding" variant (canvas → mask → predict → re-mask →
commit), selected by EVIDENCE (non-causal decoder + the family's masked
scheduler declaration — config `span_len`/decoding steps are constructor
records) — never by name.  **Witness:** facebook/magnet-small-10secs.
**Drift: zero.**

---

## RIGOROUS GATE (2026-07-08, Soumil's "make a rigorous check on everything")

**Verdict: the two SHIPPED units are Sable-CLEAN.**  musicgen-small and
stable-audio both PASS every blocking net (config audit 0 · wiring · fact ·
op · nested · label-lint · coupling); asserted_facts advisories remain by
design (B5 tags on unproven mask/scale/storage defaults).  Findings the gate
surfaced and their dispositions:

FOUND + FIXED (all general):
- label-lint: my ×4/×9 counts were ON block labels → moved to titles/facts.
- wiring net blind to the conditioning rail → side block declares
  diffusion_stage cross_attention (maps to the text role).
- MusicGen position "unresolved" → root cause: the field-type extractor
  missed the LOCAL-VAR construction idiom (``decoder = X._from_config(...)``
  then ``self.decoder = decoder``) → two-pass local-ctor collection in
  forward_ops; position now PROVEN fixed_absolute (MusicgenDecoder).
- MusicGen fusion "no code unit" → the enc-dec conditioning route
  (owns encoder_hidden_states + passes it as a call keyword) added to
  fusion evidence — proven cross_attention route, modality "conditioning".
- Stable Audio cell FABRICATED AdaLN ×-gates + timestep→norm arrows +
  "(image tokens)" on an audio DiT (pixel pass catch): new reader
  denoiser_block_timestep_conditioning_from_files (block's own forward takes
  temb / Ada-norm fields → tri-state); False drops gates + rail + AdaLN
  prose (loop + tok_text wording gated); label "(latent tokens)".
- Wrong-oracle bind (SA diffed against FluxTransformerBlock): TWO pre-existing
  resolver defects — _is_block_class was a NAME-SUFFIX test that rejected
  StableAudioDiTBlock (now structural for module-list candidates: constructs
  an attention-role field; tier-3 name sweep stays conservative so a missing
  package still resolves to honest-unresolved, not a T5 sublayer), and
  candidates keyed by FIELD NAME let a shared-shim import
  (attention_processor → transformer_flux) shadow the true block (now an
  exact-ownership tier: the architecture's OWN ModuleLists win).
- Config-ownership audit → 0 unread on all five witnesses: SA declared dims
  as config_facts chips (global cond input / cross input width / time proj) +
  oobleck decoder_input_channels; audio_encoder codec scope → opaque_scopes
  (REMOVE at U-C); undrawn speech-component spellings + training/inference
  knobs classified; flat-seq2seq encoder aliases (encoder_layers /
  encoder_attention_heads / encoder_ffn_dim / max_text_positions) — speecht5
  dict-parses now too; vits use_bias → attention_bias alias; parler
  decoder-scope reads (cross-GQA num_cross_attention_key_value_heads feeds
  the cross spec; rope_embeddings + scale_embedding read, folded
  only-when-true).

THE LOCK WALK (what the resolver hardening shook loose — each disposition
general):  the structural block recognizer let the nets SEE four classes the
name-suffix test had silently skipped.  bloom: BloomBlock threads its
residual INTO the sublayers (dropout_add inside) → the op extractor gained
the linearizer's threaded-residual rule (a role-typed sublayer call passing
``residual`` also performs the merge).  diffusion_gemma: its text layer
INLINES MoE routing in the layer forward → composite row
``ffn=linear,activation,route`` (the drawn MoE box subsumes it; the router
is checked at drill depth).  pixart-sigma: the legacy shim architecture has
no own ModuleLists and the anchored tier collided on the ``transformer_blocks``
FIELD NAME with an IPAdapter helper → structural recognition is now scoped
to the EXACT-OWNERSHIP tier only; wider tiers keep the conservative suffix.
cogvideox: the newly-bound block re-cats the already-joint text ‖ video
sequence per block (API bookkeeping of one joined stream) → cited scoped
omission ``cogvideox/block=concat`` with a ``since: cat`` staleness pin.
Result: 25/25 blessed fixtures CLEAN by direct check_regression scan.

KNOWN-GATED (blocking findings that REMAIN, all warned on-page):
- parler: fusion + ffn drills unresolved — its modeling package is not
  installed (custom-package source rail = the ACE-Step/YuE backlog item).
- speecht5: decoder-half wiring bind + position — the undrawn seq2seq half
  (U-E draws it).
- vits: position (undrawn machinery) + the conv-FFN drill fabrication
  (pre-existing; U-E retires it — FFNSpec needs a conv projection mode
  through all three serializer projections).

## CLOSING EVERY UNIT
Suite green + 25/25 sweep zero-drift + witness render → Sable 12/12 →
pixel pass → bless the witness INTO the corpus (+ Her Eyes) → evolve
PROJECT_CONTEXT Part 7/13 (never append status) + AUDIO_SUPPORT_MAP status
column + toserve list rows.
