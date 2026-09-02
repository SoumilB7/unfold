# Baseline generation audit — verdict (HEAD `d778d46`, 2026-09-02)

Question: are we generating, in all model cases, what the contract promises?
Method and sets: `README.md`. Rows: `_table.md` (machine roll-up) and
`models/*.json` (per model: IR facts, Sable checks, instance inventory,
shadow comparison). 44 models: 14 corpus transformers (A), 15 corpus
diffusion (B), 15 out-of-corpus cached ids (C).

## 1. Verdict in one paragraph

**Corpus text: ok.** All 14 agree with the instantiated model on layer count
and tying; core mechanism slots are known on 10 of 14 and the unknowns are
few and named. **Corpus diffusion: honest but hollow.** All 15 draw the
opaque "Repeated denoiser structure unresolved" block; 60–100 % of mechanism
slots are unknown; three have zero layers in the IR (PixArt, SD3.5, SDXL).
**Unseen models: not ok, in three ways the corpus never showed.** Three of
15 crash with an unhandled exception (the whole Qwen3.x
`ForConditionalGeneration` family), two draw fabricated mechanisms that only
the Sable nets catch (Jamba, DeepSeek-Coder-V2-Lite), four render every
mechanism slot unknown (Jamba, GLM-4-9B, HunyuanDiT, Qwen3-Omni), and one
hybrid schedule is silently flattened (MiniMax-M2). Cold parse ranges from
1 s to 739 s.

## 2. The table (compressed; full in `_table.md`)

core = unknown share of the 9 core slots (attention kind/heads/head_dim/
position/mask, FFN kind/activation, norm kind/placement); all = unknown
share of the 20 applicable slots. fail = blocking/advisory Sable findings.
agree = IR layer count equals a stack (or the stack sum) of the instance.

| set | model | layers IR / instance | core unk | all unk | views | fail | agree | tie | parse s | state |
|---|---|---|---|---|---|---|---|---|---|---|
| A | bloom | 70 / 70 | 0 % | 12 % | 4 | 0/0 | ✓ | ✓ | 12 | clean |
| A | dbrx-base | 40 / 40 | 22 % | 30 % | 6 | 0/0 | ✓ | ✓ | 19 | activation + norm kind unknown |
| A | deepseek-v3 | 61 / 61 (3+58) | 0 % | 15 % | 11 | 0/0 | ✓ | ✓ | 4 | clean |
| A | gemma-2-2b-it | 26 / 26 | 0 % | 11 % | 6 | 0/0 | ✓ | ✓ | 6 | clean (IR 2 groups = sliding/full; instance 1 class) |
| A | glm-4-5 | 92 / 92 | 0 % | 10 % | 9 | 0/0 | ✓ | ✓ | 3 | clean |
| A | gpt-oss-20b | 24 / 24 | 11 % | 25 % | 8 | 0/0 | ✓ | ✓ | 3 | activation (clamp family) unknown |
| A | granite-3-0-8b | 40 / 40 | 0 % | 11 % | 4 | 0/1 | ✓ | ✓ | 2 | multipliers advisory |
| A | llama-7b | 32 / 32 | 0 % | 11 % | 4 | 0/0 | ✓ | ✓ | 2 | clean |
| A | musicgen-small | 24 / 24 (+T5 12, +16+16 enc) | 44 % | 47 % | 8 | 0/0 | ✓ | ✗ | 128 | mask/position/ffn kind/norm placement unknown; tying: IR None vs instance shared |
| A | olmo-2-1124-7b | 32 / 32 | 0 % | 6 % | 4 | 0/0 | ✓ | ✓ | 1 | clean |
| A | qwen2-vl-7b | 28 / 28 (+vision 32 not in IR) | 22 % | 29 % | 11 | 0/0 | ✓ | ✓ | 47 | position + ffn kind unknown; vision stack absent |
| A | qwen3-5-27b-text | 64 / 64 | 8 % | 9 % | 6 | 0/0 | ✓ | ✓ | 23 | mask + position unknown |
| A | qwen3-8b | 36 / 36 | 0 % | 6 % | 4 | 0/0 | ✓ | ✓ | 2 | clean |
| A | stablelm-2-1-6b | 24 / 24 | 0 % | 6 % | 4 | 0/0 | ✓ | ✓ | 2 | clean |
| B | auraflow | 36 / 4+32 | 80 % | 84 % | 15 | 0/0 | ✓ | — | 41 | OPAQUE |
| B | cogvideox-5b | 42 / 42 | 78 % | 88 % | 15 | 0/0 | ✓ | — | 33 | OPAQUE |
| B | flux-2-dev | 56 / 8+48 | 89 % | 94 % | 16 | 0/0 | ✓ | — | 84 | OPAQUE |
| B | flux | 57 / 19+38 | 82 % | 86 % | 20 | 0/0 | ✓ | — | 20 | OPAQUE |
| B | hunyuanvideo | 60 / 20+40 (+2 refiner) | 59 % | 75 % | 20 | 0/0 | ✓* | — | 15 | OPAQUE |
| B | ltx-video | 28 / 28 | 100 % | 100 % | 12 | 0/0 | ✓ | — | 2 | OPAQUE, nothing known |
| B | lumina-image-2 | 30 / 2+2+26 | 100 % | 100 % | 16 | 0/0 | ✓ | — | 5 | OPAQUE, nothing known |
| B | mochi-1 | 48 / 48 (inst 2 groups) | 100 % | 100 % | 9 | 0/0 | ✓ | — | 2 | OPAQUE, nothing known; last-block variant not in IR |
| B | pixart-sigma | **0** / 28 | — | — | 13 | 0/0 | ✗ | — | 3 | OPAQUE, no layers |
| B | prxpixel | 24 / 24 | 89 % | 94 % | 9 | 0/0 | ✓ | — | 31 | OPAQUE |
| B | qwen-image | 60 / 60 | 67 % | 82 % | 13 | 0/0 | ✓ | — | 71 | OPAQUE |
| B | sana-1600m | 20 / 20 | 89 % | 94 % | 12 | 0/0 | ✓ | — | 2 | OPAQUE |
| B | sd3.5-large | **0** / 38 (inst 2 groups) | — | — | 19 | 0/0 | ✗ | — | 13 | OPAQUE, no layers; all 19 views are encoders/VAE/scheduler |
| B | sdxl | **0** / UNet 3+3+10+2 | — | — | 29 | 0/1 | ✗ | — | 6 | UNet path draws 29 views; IR has no layer stack |
| B | wan2.2 | 40 / 40 | 100 % | 100 % | 12 | 0/0 | ✓ | — | 4 | OPAQUE, nothing known |
| C | command-a | 64 / 64 | 11 % | 24 % | 6 | 2/0 | ✓ | ✓ | 2 | `order_of_interleaved_layers` unread; position unknown |
| C | deepseek-coder-v2-lite | 27 / 27 | 44 % | 53 % | 8 | **4**/0 | ✓ | ✓ | 9 | **fabrication**: draws linear attention, code is MLA |
| C | glm-4-9b | 40 / — (chatglm not buildable) | 100 % | 100 % | 4 | 2/1 | — | ✓ | 5 | nothing known; `_repo_id` leaks into audit |
| C | gpt-neox-20b | 44 / 44 | 33 % | 33 % | 4 | 4/0 | ✓ | ✓ | 3 | attention kind unknown; partial-rotary unreceipted |
| C | hunyuandit | 40 / 40 (inst 2 groups) | 100 % | 100 % | 16 | 1/0 | ✓ | — | 41 | OPAQUE, nothing known |
| C | jamba | 32 / 32 (inst 3 groups) | 100 % | 100 % | 4 | 3/0 | ✓ | ✓ | 5 | **fabrication**: draws attention on Mamba layers; nothing known |
| C | lfm2-1.2b | 16 / 16 (inst 2 groups) | 11 % | 28 % | 4 | 3/0 | ✓ | ✓ | 2 | conv layers not represented (mixer schedule unqualified) |
| C | minimax-m2 | 62 / 62 | 11 % | 20 % | 6 | 2/0 | ✓ | ✓ | 22 | `attn_type_list` unread → hybrid schedule flattened to 1 group |
| C | qwen3.5-27b full | **CRASH** | — | — | — | — | — | — | 170 | `projector.py:159` |
| C | qwen3.6-35b-a3b | **CRASH** | — | — | — | — | — | — | 68 | `projector.py:159` (confirmed) |
| C | qwen3-omni-30b | 48 / — | 100 % | 100 % | 10 | 4/0 | — | ✓ | **739** | nothing known; fusion has no code unit; audio config unread |
| C | qwen3-vl-235b | **CRASH** | — | — | — | — | — | — | 128 | `projector.py:159` |
| C | sana-1.5-4.8b | 60 / 60 | 89 % | 94 % | 11 | 0/0 | ✓ | — | 14 | OPAQUE |
| C | sd-v1-4 | **0** / UNet 4+4 | — | — | 28 | 1/1 | ✗ | — | 23 | UNet path; `feature_extractor` unread |
| C | seed-oss-36b | 64 / 64 | 0 % | 11 % | 4 | 2/0 | ✓ | ✓ | 3 | clean mechanism; `residual_dropout` unread |

\* HunyuanVideo's 2-block token refiner lives under `context_embedder`; not a
stack disagreement.

## 3. The three silent classes (the ones the contract forbids)

1. **Crash instead of typed refusal** — Qwen3.5-27B, Qwen3.6-35B-A3B,
   Qwen3-VL-235B. Same site: `evidence/projector.py:159`,
   `root.graph.node_for(item.caller_occurrence).symbol` where `node_for`
   returns `None`. Earliest false producer: the projector's caller-name join
   assumes every caller occurrence is a root-graph node. A plain `unfold()`
   user gets a traceback.
2. **Fabrication on the ship path** — Jamba (attention drawn on
   `JambaMambaDecoderLayer`), DeepSeek-Coder-V2-Lite (linear attention drawn,
   MLA in code). `op_conformance` / `fact_conformance` fire in Sable, but
   `unfold()` does not run Sable; the drawing ships.
3. **Schedule flattening** — MiniMax-M2 `attn_type_list` unread: 62 layers
   become one group. The `config_field_audit` net flags it, again only in
   Sable.

Everything else is *visible*: the opaque block, the chips, the warnings.
Visible-but-hollow is a quality problem, not a contract violation; the three
above are contract violations.

## 4. What is genuinely good

- Layer count and tying agree with the real instantiated model on every
  buildable transformer (23 of 23), and on every diffusion model where the IR
  has layers (13 of 13, sum-of-stacks rule).
- The nets caught both fabrications and every unread schedule field on models
  they had never seen. Evidence-not-identity is doing its job at the audit
  layer.
- Ten corpus transformers have zero core unknowns; DeepSeek-V3's 3+58 split,
  Gemma-2's sliding/full alternation, Qwen3.5's 64-layer two-group schedule
  all render as distinct groups.
- Unseen dense models read well when their code is conventional: Seed-OSS
  (0 % core unknown), Command-A, LFM2's attention layers.

## 5. What this changes in the plan (→ `12` v2.4)

- **S3 gains a law**: every `unfold()` ends in an IR or a typed refusal, never
  a traceback. First producer-first fix: the projector caller join. Poison: a
  `ForConditionalGeneration` config through the plain path.
- **S4 gains the unseen-model lane**: the 15 set-C ids become a standing CI
  set with the same `proven / flagged / silent` denominator as the corpus;
  "silent" = crash or a blocking net finding not surfaced as a chip.
- **S4 gains ship-path chips**: a blocking conformance finding must appear on
  the drawing (`unfold()` output), not only in the Sable report.
- **S2's profiled targets** add Qwen3-Omni (739 s) and MusicGen (128 s).
- **S7/S9** already own the diffusion hollowness and the Qwen2-VL vision
  stack; this audit is their "before" row.

## 6. Reproduction

`python3 gen_audit.py <out_dir> name=corpus:<slug> | name=id:<Org/Repo>`,
then `python3 rollup.py`. Set `HF_HUB_OFFLINE=0` for composite diffusers ids
(offline resolution fails for them even when cached — a loader finding of its
own, `README.md`).
