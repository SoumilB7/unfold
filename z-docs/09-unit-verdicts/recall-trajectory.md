# Recall trajectory — what each unit did to the drawn diagrams

Every unit's gates prove *soundness* (nothing fabricated). Nothing in the tree
measures *recall* (nothing proven was lost). This file is the missing recall
measurement, reconstructed from git: for every blessed witness, the number of
drawn diagram views locked in `tests/sable_test_corpus/<witness>.json`
(`len(hash_signature)`) at each unit's closing commit, plus the view-name
deltas from the tracked galleries (available from the U7 close onward).

Closing commits used: U2 `1d0c72b` · U3 `4bd1395` · U4 `e77f014` · U5 `569cd8b`
· U6 `a055591` · U7 `37f3b1b` · U8 `fd20ac4` · U9 `705f497` · U10 `7ade5cf`.

## 1. Drawn views per witness at each unit close

| witness | U2 | U3 | U4 | U5 | U6 | U7 | U8 | U9 | U10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auraflow-v0-3 | 16 | 16 | 16 | 16 | 15 | 15 | 15 | 15 | 15 |
| bloom | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| cogvideox-5b | 14 | 14 | 14 | 14 | 14 | 14 | 15 | 15 | 15 |
| dbrx-base | – | – | – | – | 6 | 6 | 6 | 6 | 6 |
| deepseek-v3 | 13 | 13 | 13 | 13 | 13 | 13 | **11** | 11 | 11 |
| flux-2-dev | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| fluxtransformer2dmodel | 18 | 18 | 18 | 18 | 18 | 18 | 19 | 19 | 20 |
| gemma-2-2b-it | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| glm-4-5 | 11 | 11 | 11 | 11 | 11 | 11 | **9** | 9 | 9 |
| gpt-oss-20b | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| granite-3-0-8b-instruct | – | – | – | – | – | 4 | 4 | 4 | 4 |
| hunyuanvideo | 22 | 22 | 21 | 21 | 21 | 21 | 21 | 21 | **20** |
| llama-7b | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| ltx-video | 12 | 12 | 12 | 12 | 12 | 12 | 13 | 13 | **12** |
| lumina-image-2-0 | 20 | 20 | 18 | 18 | 18 | 19 | 19 | 19 | **16** |
| mochi-1-preview | 10 | 10 | 10 | 10 | 10 | 10 | 11 | 11 | **9** |
| musicgen-small | 10 | 10 | 10 | 10 | 9 | 9 | 9 | **8** | 8 |
| olmo-2-1124-7b | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| pixart-sigma-xl-2-1024-ms | 16 | 16 | 16 | 16 | 16 | 16 | 17 | 17 | **13** |
| prxpixel-t2i | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |
| qwen-image | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | **13** |
| qwen2-vl-7b-instruct | 13 | 13 | 13 | 13 | 13 | 13 | 13 | **11** | 11 |
| qwen3-5-27b-text | – | – | – | – | 6 | 6 | 6 | 6 | 6 |
| qwen3-8b | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| sana-1600m-1024px-diffusers | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 13 | **12** |
| stable-diffusion-3-5-large | 22 | 22 | 22 | 22 | 20 | 20 | 21 | 21 | **19** |
| stable-diffusion-xl-base-1-0 | 29 | 29 | 30 | 30 | 29 | 29 | 29 | 29 | 29 |
| stablelm-2-1-6b | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| wan2-2-t2v-a14b-diffusers | 15 | 15 | 15 | 15 | 14 | 14 | 14 | 14 | **12** |
| **total** | 327 | 327 | 325 | 325 | 331 | 336 | 338 | 335 | **319** |
| witnesses | 26 | 26 | 26 | 26 | 28 | 29 | 29 | 29 | 29 |

View count is a coarse proxy (a view can change content without changing
count, and a removed fabricated view is a *gain* in honesty). It is still the
only longitudinal recall signal that exists, and it shows one thing clearly:
**two units reduced witness totals: U6 (five witnesses, collapses onto the
unresolved stub, masked by the two new witnesses) and U10 (six diffusion
witnesses, 16 views, the only net corpus-wide drop).**

## 2. View-name deltas from the tracked galleries (U7 → U10)

| witness | U8 (`37f3b1b`→`fd20ac4`) | U9 (→`705f497`) | U10 (→`7ade5cf`) |
|---|---|---|---|
| auraflow-v0-3 | – | – | −text_projection +ffn__1 |
| cogvideox-5b | encoder views split into exact per-group views (g0/g1) | – | – |
| deepseek-v3 | −mtp, −mtp_block | – | – |
| fluxtransformer2dmodel | encoder views split g0/g1 | – | +ffn__1 |
| glm-4-5 | −mtp, −mtp_block | – | – |
| hunyuanvideo | – | – | −text_refiner, −text_refiner_op_ffn +ffn__1 |
| ltx-video | encoder views split g0/g1 | – | −text_projection |
| lumina-image-2-0 | – | – | −entry_stage, −entry_stage_op_ffn, −text_refiner |
| mochi-1-preview | encoder views split g0/g1 | – | **−attn, −ffn** |
| musicgen-small | −conditioning_encoder_op_ffn +…op_selfattn | −conditioning_encoder, −…op_selfattn +conditioning_projector | – |
| pixart-sigma-xl-2-1024-ms | encoder views split g0/g1 | – | **−attn, −cross_attn, −ffn, −text_projection** |
| qwen-image | – | – | −encoder_0_op_ffn |
| qwen2-vl-7b-instruct | – | **−vision_enc_op_ffn, −vision_enc_op_selfattn** | – |
| sana-1600m-1024px-diffusers | – | – | −text_projection |
| stable-diffusion-3-5-large | encoder views split g0/g1 | – | **−attn, −ffn, −text_projection** +encoder_0_op_selfattn |
| wan2-2-t2v-a14b-diffusers | – | – | **−attn, −cross_attn, −ffn** +encoder_0_op_selfattn |

Unlisted witnesses (bloom, dbrx, flux-2-dev, gemma-2, gpt-oss, granite, llama,
olmo-2, prx, qwen3-5, qwen3-8b, sdxl, stablelm) had no view-name change.

## 3. How to read the drops

Each drop has to be classified as one of:

- **(a) fabricated detail removed** — the old view was authored by a
  convention/default/config token. Removing it is a correction.
- **(b) verified detail lost to a substrate gap** — the old view was
  conformance-verified against real source, and the new substrate simply
  could not see that source. Removing it is a *recall regression* that the
  lost-detail rule (`docs/U3_ACCOMPLISHMENTS_AND_CURRENT_PROCEDURE_AUDIT.md`
  §7) says must not be accepted without a re-proof attempt.
- **(c) verified detail deferred by explicit decision** — provable, not
  re-proven, consciously assigned to a later owner.

Classification of the bold rows (details in the per-unit files):

| drop | class | basis |
|---|---|---|
| DeepSeek/GLM `mtp`, `mtp_block` (U8) | (a) | the installed HF source declares `num_nextn_predict_layers` but constructs no MTP module; the drill was config-authored — a genuine correction |
| T5 encoder `g0/g1` split in six diffusion witnesses (U8) | **soundness miss** | `g1` (23 layers) now draws softmax with no relative-bias input and no chip; `modeling_t5.py` carries `position_bias` to every layer — a fabricated absence, re-blessed again by U10 |
| Qwen3.5 "(causal)" and "Partial RoPE" legends → unresolved (U8) | **(b)** | both provable from `modeling_qwen3_5.py` (`:1201` ternary mask selector; `:1177-1180`/`:153` text-stage coordinate); assigned to U9, which closed without touching them |
| MusicGen encoder views (U8/U9) | mixed | U9 proved the real T5 encoder (12L/12H/64d) and added the exact projector view; the removed generic cell drill was class-authored |
| Qwen2-VL `vision_enc_op_*` (U9) | **(b)** | the vision block is an ordinary pre-norm attention+MLP cell in `modeling_qwen2_vl.py`; U9 recorded "current exact occurrence readers do not close that cell" — a substrate gap, not an epistemic unknown |
| embedded CLIP/UMT5/T5 encoder self-attn + MusicGen decoder attn → stub (U6) | **(b)** | provable from installed `transformers`; MusicGen decoder later proven MHA 16/16/64; **none of the collapsed encoder drills has come back** — SD3.5 `encoder_0` at HEAD is byte-identical to the U6 stub (Experiment C) |
| PixArt, SD3.5, Mochi, Wan block drills (U10) | **(b)** | old views passed op/nested/wiring/fact conformance against real Diffusers source; the new ProgramIndex excludes the shared files (`attention.py`, `attention_processor.py`) their block classes are imported from |
| `text_projection` / `text_refiner` / `entry_stage` views (U10) | (a)/(c) | authored from config enums/conditioning templates; the refiner host-width claims were over-strong; some (Lumina refiners, HunyuanVideo refiner) are source-provable and are U10-D lanes awaiting the same import closure |

## 4. Verdict of the trajectory

- U2–U5: no recall loss; U7: none (Falcon re-proven in-unit, Sana restored).
- **U6: the first loss.** The drops at the U6 column (AuraFlow 16→15, SD3.5
  22→20, Wan 15→14, MusicGen 10→9, SDXL 30→29) are distinct CLIP-L / CLIP-G /
  UMT5 / T5 encoder self-attention drills and MusicGen's decoder attention
  collapsing onto the unresolved stub (identical hashes), approved under the
  claim that the 26 galleries were byte-identical — galleries were untracked
  then, so the claim was uncheckable. SD3.5 CLIP-G and AuraFlow UMT5 attention
  are still absent at HEAD.
- U8: MTP removal was a correction; but Qwen3.5's provable mask/position
  legends went unresolved, and the T5 encoder split shipped a silent negative
  into six galleries.
- U9: one verified cell (Qwen2-VL vision) dropped to opaque under a substrate
  gap.
- U10: the first net recall regression of the campaign — six witnesses,
  16 views, two flagship denoisers (PixArt, SD3.5) reduced to blank boxes.

The pattern is monotonic with distance from the transformer decoder core: the
further a mechanism lives from the model's own modeling file, the more the
exact-bundle boundary costs, and no gate reports it. The fix is one substrate
change (demand-driven import closure, which U11-A1 already built) plus one
blocking recall net; see `systemic-findings.md`.
