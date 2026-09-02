# Reviewer log — running state (evolve in place; compress later, never lose)

Read this first after any context loss, then `10-full-research/14`, `12`,
`15`. Owner: the reviewer (Claude). Executor sheets: `S<n>.md`; verdicts:
`S<n>-review.md`. Soumil's rule (2026-09-02): keep writing here so context
survives compaction; redundancy is fine now, prune later.

## 1. Where the campaign is (2026-09-02 21:50 IST)

| step | executor | reviewer | notes |
|---|---|---|---|
| S0 hygiene | DONE `4a8b338`+`dbacd66` | **ACCEPT** (`S0-review.md`) | fingerprint hashes the `.git` pointer file → reconstruct with worktree name; fix owned by S4 |
| S1 close U11-F | DONE `d778d46`, pushed | **ACCEPT** (`S1-review.md`) | 101/101 F4/F2c tests reproduced; lanes reconstruct as `verify-9b8aa77d6b-<lane>` |
| S2 latency | not started | — | executor may start now |
| baseline generation audit | — | **DONE** → `baseline-2026-09-02/BASELINE_GENERATION_AUDIT.md` (44/44) | plan amended to v2.4 (no-traceback law, unseen-model lane, ship-path chips) |

Contract B1 adopted (text in `14` §B). Plan v2.3 final (`12`). Executor
brief `15`. Nothing else pending from Soumil.

## 2. How to resume the audit if this session dies

- Scripts now live in-repo: `baseline-2026-09-02/gen_audit.py` and `rollup.py` (copied 21:37).
- Roll-up: `scratchpad/rollup.py` → writes `baseline-2026-09-02/_table.md`.
- Per-model JSON already written: `baseline-2026-09-02/models/*.json`
  (44 planned; count the files to see progress).
- Rerun any model with `python3 gen_audit.py <out_dir> <name>=corpus:<slug>`
  or `<name>=id:<Org/Repo>` (set `HF_HUB_OFFLINE=0` for composite diffusers
  ids; offline resolution fails for them).
- Final deliverable to write: `baseline-2026-09-02/BASELINE_GENERATION_AUDIT.md`
  = the table + the reading (tier-1 / tier-2 / crash / fabrication findings
  below) + what it means for S2–S9.

## 3. Audit runs in flight / done

| batch | scope | state |
|---|---|---|
| A | 14 corpus transformer witnesses | DONE (4 MoE rows re-run in A2 after a script bug; all 14 JSONs valid) |
| B | 15 corpus diffusion witnesses | DONE |
| C | 15 out-of-corpus cached ids | STOPPED by the watcher after 10/15 (Omni row written at 21:38; parse **739 s**); remaining 5 moved to C3 |
| C2 | rerun: musicgen, deepseek-coder-v2-lite, sana-1-5, hunyuandit (network on) | DONE — all four rows valid (musicgen 24L; deepseek-coder 27L 4 blocking fails; SANA1.5 60L; HunyuanDiT 40L 100% unknown, 1 fail) |
| C3 | the 5 remaining C ids | DONE (qwen3-vl CRASH; command-a 64L 2 fails; gpt-neox 44L 4 fails; seed-oss 64L clean mech; sd-v1-4 UNet 28 views) |
| C watcher | stopped batch C after the Omni row | DONE |
| C4 | MiniMax-M2 62L 2 fails; Qwen3.6 traceback CONFIRMED same `projector.py:159` | DONE |
| known script bugs fixed mid-run | `dict in set` crash on MoE routing facts; instance builder now tries the declared architecture class first | rows affected were re-run |

## 4. Findings so far (all measured at HEAD `d778d46`)

**Tier 1 — structure**
- PixArt-Σ, SD3.5, SDXL: IR `layers = 0`; PixArt/SD3.5 draw one opaque
  "Repeated denoiser structure unresolved" box; the instance builds 28 / 38
  blocks. SDXL draws 29 views through the UNet path with layers=0 in the IR.
- Diffusion witnesses whose layer count is right (AuraFlow 36, CogVideoX 42,
  Mochi 48, LTX 28, Lumina 30, Wan 40, Qwen-Image 60, Sana 20, PRX 24,
  Flux 57, Flux-2 56, HunyuanVideo 60) still carry the opaque block and
  80–100 % unknown mechanism slots (Mochi/LTX/Lumina/Wan: 100 %).
- Instance vs IR layer count agrees on every transformer witness; on
  diffusion it agrees where IR has layers (sum-of-stacks rule for AuraFlow
  4+32, Flux 19+38).
- HunyuanVideo: IR 60 = 20 dual + 40 single; the instance also has a 2-block token refiner under `context_embedder` that the IR carries elsewhere (not a disagreement; the roll-up's sum rule flags it, read manually).
- Corpus transformer core-unknowns are few and named: DBRX activation+norm kind; gpt-oss activation (the swiglu clamp family); MusicGen mask/position/ffn kind/norm placement; Qwen2-VL position + ffn kind; Qwen3.5-text mask + position. DeepSeek-V3 core = 0 unknown.
- Qwen2-VL: IR 28 text layers; instance also has `visual.blocks` ×32 that the
  IR does not carry as a stack (the lost vision cell, `10` §3-4).

**Tier 2 — mechanism**
- Corpus transformers: 15–37 % of applicable slots unknown (qk_norm and
  score_scaling are None everywhere = never proven present/absent; Granite
  multipliers advisory; DeepSeek-V3 189/1220).
- Out-of-corpus text: Jamba 32 layers and GLM-4-9B 40 layers with **zero
  known facts per layer** (everything unknown, still draws attn/ffn views).
  LFM2 reads well (GQA 32/8, rope 1e6, silu gated) but its conv layers are
  not represented (mixer_schedule finding).

**Fabrications caught by the product's own nets on unseen models (good: the
nets fire; bad: the parser produced them)**
- Jamba: `op_conformance` — draws `attention` on `JambaMambaDecoderLayer`
  whose forward never does it.
- DeepSeek-Coder-V2-Lite: `fact_conformance` — draws linear attention where
  the code uses MLA; `qualified_projection_values` — mixer schedule
  projected without an owner-qualified fact.

**Crash (law violation: not a typed refusal)**
- `Qwen/Qwen3.6-35B-A3B` (hybrid MoE) and `Qwen/Qwen3-VL-235B-A22B-Instruct` show the identical error string → **the whole current Qwen3.x multimodal/hybrid family (`*ForConditionalGeneration`) crashes** in the projector caller join. Traceback for 3.6 being confirmed in C4.
- `Qwen/Qwen3.5-27B` full id (`Qwen3_5ForConditionalGeneration`):
  `AttributeError: 'NoneType' object has no attribute 'symbol'` at
  `evidence/projector.py:159` — `root.graph.node_for(item.caller_occurrence)`
  returns None. Corpus only ever held the text component. Earliest false
  producer: the projector's caller-name join assumes every caller occurrence
  is a root-graph node.

**Blocking config/boundary findings on unseen models**
- `config_field_audit` unread fields: Jamba `attn_layer_offset`, LFM2
  `block_auto_adjust_ff_dim`, GLM-4-9B `_repo_id` (loader-stamped key leaks
  into the audit), DeepSeek-Coder `attention_bias`.
- `document_boundary_completeness`: Jamba `hidden_act`, LFM2 `conv_bias`,
  DeepSeek-Coder `first_k_dense_replace`.

**Qwen3-Omni (unseen, composite)**: parses (48 thinker layers, vision/video/audio/fusion views drawn) but **100 % unknown mechanism**, 4 blocking nets (`thinker_config.audio_config` unread; fusion has no code unit to diff; `config_audit_incomplete` root.vision; unlocated `audio_config` read), and **739 s cold parse** — the single worst latency measured.
**MiniMax-M2 (unseen, hybrid)**: 62 layers read as ONE group; `attn_type_list` (the lightning/softmax schedule) is an **unread config field** (blocking) — the hybrid schedule is not represented; `head_dim` origin unestablished.

**Loader**
- Composite diffusers ids fail with `ModelNotFoundError` under
  `HF_HUB_OFFLINE=1` even though the snapshot is cached (SANA 1.5, HunyuanDiT).
  Being re-run with network on (C2).
- GLM-4-9B (`chatglm` remote code) cannot be instantiated by transformers
  5.12 (typed failure on the instance side, expected).

**Latency**
- Corpus parse+sable: 15–60 s per model; Qwen3-Omni > 6 min (cold, code
  inspection) — the S2 `_seg` fix target.

## 5. What the findings mean for the plan (provisional, finalise in the audit doc)

- S3's consumer-honesty list gains nothing new; the opaque box and chips are
  honest. The **silent** failures are: the Qwen3.5 crash (must become a typed
  refusal), Jamba/DeepSeek-Coder fabrications (the nets catch them in Sable,
  but a plain `unfold()` user sees the fabricated drawing — S4's "no waiver
  without a chip" must also gate the ship path, not only Sable).
- S4's `coverage.json` denominator: this audit's per-model unknown counts are
  the seed; add "silent" = crash or net-caught fabrication.
- S7 shadow matrix: the instance-vs-IR layer comparison here is its first
  row; multi-stack models (Qwen2-VL, AuraFlow, Flux) need the sum/any rule.

## 6. Next actions (in order)

1. Executor runs S2 (permitted). Reviewer reviews S2 against C-2 (now incl. Qwen3-Omni + MusicGen budgets).
2. S3 must include the no-traceback law fix (projector join) — check C-3 addendum.
3. S4 must include the unseen-model lane + ship-path chips — check C-4 v2.4 lines.
4. Prune this log's §3 once S2 is accepted (audit batches are history; keep §4/§5 findings).
