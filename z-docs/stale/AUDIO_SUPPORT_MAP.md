# AUDIO-GENERATION + TTS SUPPORT MAP — what the machinery needs, unit by unit
*(2026-07-07 · grounded by live probes: vits/speecht5 parse today but
misleadingly partial; encodec/bark refuse honestly; musicgen's composite
config needs the wrapper walk; StableAudioDiTModel/oobleck/AudioLDM2 modeling
are INSTALLED in diffusers and "StableAudioDiTModel" already matches the
`DiT` class marker.  Law 1 everywhere: every unit below is a general
mechanism; audio families are witnesses.  Companion: PROJECT_CONTEXT Part 13,
TOWER_CENSUS.md §2/§4.)*

## THE STRUCTURAL READ — every requested family is one of FOUR shapes

| family | shape | adapter home |
|---|---|---|
| MusicGen / AudioGen, Parler-TTS | codec-token LM: T5 encoder → cross-attn decoder → K codebook heads (+ EnCodec) | transformer |
| MAGNeT | same decoder, bidirectional + masked-iteration loop | transformer |
| YuE | llama-style decoder over audio tokens (dual-track) | transformer (parses today) |
| Bark | 3-stage GPT pipeline (semantic→coarse→fine) + EnCodec | transformer ×3 stages |
| Stable Audio 1/2, ACE-Step, F5-TTS | audio-latent DiT (1D/temporal latents) + conv1d VAE | diffusor |
| AudioLDM/2, Tango/2, MusicLDM | spectrogram-latent UNet + CLAP/T5 (+ embedded GPT-2 bridge in ALDM2) + HiFiGAN vocoder | diffusor (UNet exists) |
| Riffusion | literally SD-1.5 on spectrograms | works TODAY as SD |
| SpeechT5, VITS (TTS classics) | enc-dec / enc+flow+vocoder | transformer + new towers |

## THE UNITS (general mechanisms; ~6, ordered by unlock-per-effort)

**U-A. Seq2seq / conditional-generation wrapper walk.**  The composite
configs (musicgen: text_encoder + audio_encoder + decoder; speecht5; bark's
sub-model trio) need the wrapper vocabulary extended the way thinker/vision
wrappers were: component slots → per-component parse (encoder tower panel via
the EXISTING universal round-trip; decoder as the main stack;
cross-attn-every-layer via the EXISTING schedule machinery — "all layers" is
just the membership list).  Probe result: `AutoConfig.for_model("musicgen")`
can't even default-construct — real repo config.json drives this.
*Unlocks: MusicGen/AudioGen, Parler-TTS, SpeechT5 (honestly), MAGNeT base.*

**U-B. Multi-codebook token streams (the genuinely NEW LM fact).**  K
codebooks: K embedding tables summed at input, K lm_heads fanned at output,
the delay/interleave pattern as a token-strip chip (the fusion-strip idiom).
IR: `num_codebooks`, per-head vocab; render: head fan + pattern strip.
Config declares K (constructor record); code proves the sum/fan structurally
(ModuleList of K embeddings/heads — construction evidence, existing rails).
*Unlocks: MusicGen family, Parler, Bark stages, YuE dual-track chips.*

**U-C. Codec/RVQ tower — the census §4 audio twin.**  EnCodec/DAC:
conv1d SEANet ladder (down/up stages — near-twin of the VAE ladder archetype)
+ the RVQ quantizer as a designed fill-in ("codebook lookup, ×K residual
stages" — the exemplar fill-in shape, like the MoE router).  EnCodec's LSTM
stays an HONEST opaque box v1 (recorded scope line: recurrent cells are C1
territory).  Today encodec REFUSES ("no transformer layers") — the codec
tower gives it a first-class home instead.
*Unlocks: the audio-token story end-to-end for every codec-LM.*

**U-D. Audio-DiT geometry (1D latents) + conv1d VAE.**  The diffusor adapter
already detects the DiT (marker matches) and would draw blocks/attn/AdaLN;
the gaps are GEOMETRY facts: temporal-1D latents (no H×W patch grid — the
`video-ness` temporal-axis precedent generalizes to an AUDIO axis; evidence:
`audio_channels`/`sample_size`-seconds constructor fields + rotary-on-time),
`seconds_start/seconds_total` conditioning chips (config_facts.yaml rows),
and the oobleck conv1d up/down ladder (reuse the VAE stage renderer).
*Unlocks: Stable Audio 1/2; F5-TTS-class flow-matching DiTs; ACE-Step's DiT
half (its custom repo rides the remote-code rail).*

**U-E. Mel + vocoder tail (the audio sampling-loop hero).**  AudioLDM/Tango/
MusicLDM/SpeechT5/VITS all end "latent → mel-spectrogram → HiFiGAN →
waveform": (1) a loop-hero TAIL variant (mel + vocoder + waveform out instead
of Image/Frames) gated on DECLARED pipeline components (`vocoder` slot in
model_index — evidence, not vibes); (2) the HiFiGAN vocoder tower archetype
(transposed-conv upsample ladder + multi-receptive-field resblocks) — goes
straight into the tower census audio section.  AudioLDM2's embedded GPT-2
bridge LM = the EXISTING recursive sub-model machinery.
*Unlocks: AudioLDM/2, Tango/2, MusicLDM properly; upgrades SpeechT5/VITS
from misleading-partial to honest.*

**U-F. Masked-iteration loop hero (small).**  Generalize the EXISTING
block-diffusion loop layout into a declared "iterative masked decoding"
variant (canvas → mask → predict → re-mask).  *Unlocks: MAGNeT (and
MaskGIT-style image LMs later).*

## WHAT WORKS TODAY (probed, honest state)
- **Riffusion**: parses as SD-1.5 (it IS one).  MusicLDM-style audio wording
  only when the pipeline DECLARES audio components (evidence-based note).
- **YuE**: parses as a llama decoder (true); the audio-token semantics
  arrive with U-B chips.
- **vits/speecht5**: parse but draw ONE component as a plain stack and
  silently drop flow/duration/vocoder — the silent-omission class; U-A/U-E
  fix honestly.  VITS's normalizing FLOWS = genuinely new math → honest
  opaque v1, archetype later.
- **encodec/bark**: refuse honestly today (correct until U-A/U-C land).

## TTS SUGGESTIONS (same buckets, no new machinery beyond the six)
- **Parler-TTS** — MusicGen architecture; free with U-A+U-B (best first TTS witness).
- **Bark** — U-A (staged pipeline) + U-B + U-C.
- **F5-TTS / CosyVoice-class** — U-D (flow-matching DiT on mel) + U-E tail.
- **SpeechT5 / VITS** — U-A + U-E (+ honest-opaque flows).
- Recommend adding a TTS row-set to toserve_model.md before building.

## ORDER (recommended)
1. **U-A + U-B + U-C together = the MusicGen unit** (transformers-native,
   oracle installed, unlocks 4+ families incl. Parler-TTS; EnCodec refusal
   becomes a first-class codec tower).
2. **U-D = the Stable Audio unit** (diffusers-native, DiT detection already
   half-works).
3. **U-E = the AudioLDM/vocoder unit** (UNet exists; vocoder archetype feeds
   the census).
4. **U-F MAGNeT** (small, after 1).
5. ACE-Step/YuE polish via the remote-code rail once 1-2 exist.

**Scope-law note for Soumil:** the recorded scope line says decoder-only LLMs
+ diffusion.  Audio-gen fits BOTH existing adapters (codec-LM = decoder-only;
audio diffusion = DiT/UNet) — this is a scope EXTENSION to bless explicitly,
and the census's audio section (§2) grows two archetypes (codec/RVQ ladder,
vocoder ladder) rather than a new adapter.
