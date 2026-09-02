# Confirmation checklist — what must be seen before anything is called done

Written 2026-09-02 at Soumil's request: "anything you would need to judge
it on, any intent doc, any confirmation section or check, code-wise or
model-wise, written down fully so compaction never loses it." This file is
the judge's desk. Read it first after any context loss; then `12` (order),
`13` (hard models), `09` (judgment), `08` (register).

---

## §A The intent, in one paragraph (recachable)

`model-unfolder` takes a Hugging Face model id or config and draws the
model's architecture honestly: **tier 1** every structural element inside the
declared support set is present (a missing one is a blocking finding);
**tier 2** every mechanism is either proven from the model's own code or
visibly marked unresolved (never guessed, never inferred from a name);
**tier 3** every number is exact and sourced to a config path, a constructor
expression, or a parameter shape. The reader is a learner (Space / notebook),
the auditor second. Identity (class or model name) may locate code but may
never select architecture. Evidence flows one way: sources → typed facts →
canonical IR → renderers, params, conformance; consumers project, never
decide. As of 2026-09-02 the mechanism readers (U6–U11) are strong; structure
discovery is bounded by the static bundle; consumers still invent meaning in
a few places; recall was never measured; nothing has shipped since v0.2.17.
The agreed direction (both reviews): meta-instantiated tree + recipe trace
for construction/execution, static readers on exact classes for mechanism,
config for values, one reconciliation layer with per-question precedence,
relation axis for cross-occurrence structure, shadow mode before any
authority cutover, deletion responsibility by responsibility.

## §B Standing confirmations owed by Soumil (blocking where marked)

| # | confirmation | blocks | status |
|---|---|---|---|
| B1 | Ratify the tier contract (one line): tier 1 complete, tier 2 honest, tier 3 exact | S5 README wording, S7 blocking rule | **adopted 2026-09-02** via Soumil's "final decision, no more moving" instruction; text below is the contract |
| B2 | Torch / transformers / diffusers execution accepted; remote code isolated | — | **confirmed 2026-09-02** |
| B3 | Freeze U11 G/H/I after F4/F2c | S1 | **confirmed by both reviews**; agent message in `12` §4 |
| B4 | Approve the S3 preservation delta per witness before re-bless | S3 | open, arrives with S3 |
| B5 | Read the S7 disagreement matrix and name the first cutover family | S8 | open, arrives with S7 |
| B6 | Approve the one-page learner spec | S12 | open |
| B7 | Ship an honesty release (v0.3.0) at end of week 2 before authority changes | S5 | recommended, not yet confirmed |

**B1 proposed ratification text (2026-09-02, agreed wording pending Soumil's signature):**

> I ratify the product contract. Tier 1: every structural element and relation
> is present for every model in the declared support set, which is the
> enumerated list in `coverage.json`; a missing one is a blocking finding.
> Tier 2: every mechanism is proven from the model's own resolved code or
> visibly marked unresolved; never inferred from a name, a config flag, or a
> convention. Tier 3: every value is exact and sourced to a config path, a
> constructor expression, or a parameter shape.

Why each clause: "enumerated list" keeps tier 1 from being vacuous; "from the
model's own resolved code" is evidence-not-identity in one phrase; "and
relation" covers tying, KV sharing, multi-stream residual and side heads
(`13`).

## §C Checks I would demand before believing each step (code-wise)

Each check names the artifact that proves it. "Poison" = a deliberately
wrong input that must turn the gate red; a gate without a poison is not
trusted.

**C-0 hygiene (S0)**
- `git grep -I "hf_[A-Za-z0-9]\{30,\}"` empty on the tree **and** the two
  notebooks re-read cell by cell (`finalize.ipynb` cell 1, `done/tryrun.ipynb`
  cell 18); tokens revoked on the Hub side, not just deleted locally.
- Commit-body hook: a test commit with an empty body is rejected.
- `z-docs/` and `.claude/PROTOCOL.md` appear in `git ls-files`.

**C-1 U11-F closure (S1)**
- Coordinator receipt on `9a4e1e5` committed in-repo; upstream contains it;
  `U11_UNET_EXECUTION_PLAN.md` §10 row for F4/F2c says DONE with the receipt
  id. No commit after it touches `adapters/diffusor/unet.py` structure.

**C-2 latency (S2)**
- Fingerprints of the 5 profiled targets byte-identical before/after.
- A profile showing `ast.get_source_segment` no longer in the top 20.
- Two recorded budgets (post-import index build; end-to-end cold) from a
  quiet-machine baseline, committed as numbers, not prose.

**C-3 consumer honesty (S3)**
- For each removed site (`graph.py:105`, `op_render.py:287`,
  `metadata_modalities.py:494/668/897`, `labels.py:787`,
  `blocks/model.py:120,165,268`, `feed_forward.py:582-608`, placeholder tower,
  fixed sliding window): a witness where the old code produced a confident
  box and the new code produces a chip, shown as before/after PNG.
- `consumer_firewall` rule "no semantic default resolves an unknown" with a
  poison (reintroduce one `or "pre"` → red).
- Plain `unfold(dict)` on the SD3.5 corpus config no longer returns
  `stack.kind = decoder_only, num_layers = 0` silently; it returns a typed
  refusal or a visible zero-layer warning.

**C-4 recall and unknown-rate gates (S4)**
- Recall ratchet poison: take a blessed fixture, downgrade one proven fact to
  unresolved in a scratch copy → gate red with the fact named.
- Labels in the fixture signature: change one label → signature changes.
- `bless()` refuses a fixture whose `visual_review` was set by the same
  process that produced the render (test with a self-set CLEAN → refused).
- `config_accessed_unprojected` / `asserted_facts` are in the blocking set;
  `census.py --check` wired into CI; `zero_asserted_census` re-raises.
- `coverage.json` per model: `proven / flagged / silent`; silent = 0 for all
  29 witnesses; a poison witness with one silent fact → CI red.

**C-5 release (S5)**
- `pip install model-unfolder==0.3.0` in a clean venv renders the 29 witnesses.
- README states the supported corpus, known incomplete structure
  (SD3.5/PixArt denoiser opaque if still so), and the counts from
  `coverage.json`; the numbers match byte-for-byte.
- Space: unpinned, awake, a typed refusal is shown for a model outside the
  support set (not a generic spinner/error); `deepseek-v3.html` is DeepSeek.

**C-6 instance-oracle pilot (S6)**
- `physics/instance_inventory.py` runs in a subprocess with network off
  (test: a config whose `__init__` tries a download → typed
  `NetworkRefused`); timeout and memory cap tests; version record present.
- Inventory for the 8 pilot models committed as JSON with §1c records;
  `grep -rn "physics" model_unfolder/adapters model_unfolder/renderers` empty.
- `execution_observation.py` records module order **and** a function-mode
  op log; on SD3.5 block 0 the log contains `scaled_dot_product_attention`,
  16 `add`, 16 `mul` (the 2026-09-02 numbers); bf16 in the recipe record.
- Lazily constructed modules appear as `lazy_observed` (test with a module
  that builds a cache on first call).

**C-7 reconciliation + shadow (S7)**
- The three axes plus the relation axis are dataclasses with closed enums;
  a poison occurrence with two projection values is rejected at construction.
- Precedence is code: a synthetic conflict (instance says 38 blocks, static
  owner graph says 0) yields `construction_conflict`, blocking, never a
  merged number.
- Disagreement matrix for 29 + 10 models committed; every occurrence has all
  axes; **zero** silent drops (`named_modules()` count + lazily observed =
  rows).
- Relation rows present for: Gemma-2 tying; SD3.5 none; and, when added to
  the corpus, Gemma3n (`multi_stream_residual(4)`, `activation_reuse` 18→20…)
  and DeepSeek-V4 (`multi_stream_residual(4)`, `side_head hc_head`).
- Aperiodic schedule check: a Nemotron-H-style pattern renders as its
  distinct groups in encounter order, not forced into a cycle.
- **No pixel changed** in S7: preservation 29/29 identical.

**C-8 first family cutover (S8)**
- Differential report old-path vs new-path for every UNet witness; each
  difference has a named re-proof or a blocking finding; zero "unexplained".
- Each of the 14 `unet_*` readers: a production import path or a deletion
  commit; the count of readers with zero callers is 0.
- SDXL renders down/mid/up with cross-attention as proven / chipped /
  `not_constructed(guard)`; nothing "conventional".

**C-9 DiT/text families + restoration (S9)**
- SD3.5: 38 blocks, 37/1 split, QK-norm 64, FFN 2432→9728→2432 GELU-tanh,
  AdaLN 6×, processor named — the `12` §7 table reproduced by the product,
  not by a scratch script.
- 10 out-of-corpus `TO_SERVE` models: ≥ 90 % proven, 0 silent, each with a
  fleet visual verdict.
- Every item in the lost-detail list has a re-proof record naming the reader.

**C-10 deletion units (S10)**
- Per unit: parity receipt (corpus + out-of-corpus) + deletion in one commit;
  a grep proving no fallback import remains.

**C-11 U12–U15 (S11)** — empty firewall register with anti-vacuity poison;
one unknown vocabulary (grep for the old strings returns nothing).

**C-12 learner product (S12)** — journey acceptance per archetype; Her Eyes
verdict persisted per bless; Space cold render under the S2 budget.

## §D Checks I would demand model-wise (the witnesses that decide it)

| model | why it is in the set | what must be true when done |
|---|---|---|
| SD3.5 Large | the bundle-boundary failure; the refused grey FFN chip | `12` §7 right column, from the product |
| PixArt-Σ | runtime class remap | real class drawn; source lines point at `pixart_transformer_2d.py` |
| SDXL | UNet stages, factories, cross-attention | full stages; cross-attn never manufactured |
| DeepSeek-V3 | 3 dense + 58 MoE; MLA; fused experts | split drawn; score scale correct; MLA low-rank path |
| DeepSeek-V3.2 | DSA indexer inside attention | indexer drawn as part of attention or chipped, never omitted |
| **DeepSeek-V4** | mHC 4-stream residual, hash-MoE, `hc_head` | residual bus n = 4 with per-sublayer mixers or a visible `relation_unresolved`; hash router not drawn as learned |
| **Gemma3n** | AltUp 4 streams, LAuReL, per-layer inputs, KV sharing | all four relations rowed; KV edges 18→20… drawn or chipped |
| Gemma-2 | sandwich norm, tying, embed scale | tying as a relation; scale restored |
| Qwen3-Next / Jamba / Nemotron-H | hybrid and aperiodic stacks | distinct groups in encounter order; no forced cycle |
| Qwen2-VL | vision tower, m-RoPE | vision cell restored; position honest |
| Qwen3.5 | ternary, mask, position | resolved via trace or chipped, never guessed |
| T5 / UMT5 | loop-carried bias, per-layer bias | restored |
| MusicGen | composite, K codebooks, T5 tower | T5 tower view present |
| gpt-oss | sinks, sliding/full, swiglu clamp | sinks drawn; clamp chip never silent |
| LongCat-Flash | intra-layer shortcut MoE | shortcut as a relation row |

## §E The state to re-cache after compaction (facts, 2026-09-02)

- Branch `audio-composite-support`, HEAD `9a4e1e5`, 2 unpushed (F4 `9b3cb7b`,
  F2c `9a4e1e5`); U11 tracker row stale; 14 `unet_*` readers, zero production
  callers; legacy `adapters/diffusor/unet.py` 971 lines live; 3,991 tests.
- Last release v0.2.17 (2026-07-07); Space pinned, asleep; two plaintext HF
  tokens in notebooks (not copied anywhere).
- Measured: SD3.5 meta build 2.3 s / DeepSeek-V3 0.08 s / library import
  6.5 s; SD3.5 static route 42 s → one opaque denoiser box; plain dict route →
  zero-layer text tower.
- Probe scripts (scratchpad, session-local; copies of their logic are
  described in `12` §7 and `13` §5): `meta_demo.py`, `v22_test.py`,
  `hard_models.py`.
- Docs: `12` v2.3 binding; `13` hard models; this file; `08`/`07`/`10` carry
  amendment notes; `09-unit-verdicts/README.md` has the supersession note.
- Both reviewers converged; the counter-review's 7 corrections all tested and
  accepted; the only additions of mine still standing: the S7 time box with a
  lawful escape, the release cadence, and the relation axis.
