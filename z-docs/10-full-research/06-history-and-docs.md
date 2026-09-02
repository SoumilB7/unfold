# 06 — History and documentation: how the project got here, what is written down, what is current

Research pass written 2026-09-02 against `unfold-pkg` HEAD `9b3cb7b` (2026-09-01, branch
`audio-composite-support`, 12 uncommitted paths: 8 modified U11 files + `docs/MODULARIZATION.md`
deleted + 3 untracked plan docs + `docs/architecture_atlas.html`). Read-only; every number below
was measured with `git log`/`find`/`grep` on this tree. Unverified statements are marked
**(inference)**. Paths are relative to the workspace root unless prefixed `unfold-pkg/`.

Three facts frame everything else:

1. **The git repository is `unfold-pkg/` only** (remote `SoumilB7/unfold`, 17 branches). `z-docs/`,
   `.claude/PROTOCOL.md`, `done/`, `finalize.ipynb`, `push.txt` and `hf/` (its own repo) sit outside
   it. The curated documentation plane is therefore unversioned: no history, no diff, no receipt.
2. **487 commits, 2026-05-06 → 2026-09-01**, 474 authored as `Soumil`, 13 as `Soumil.Binhani` —
   the latter are exactly the 13 GitHub PR merges (#1 05-10 … #13 `transition` 07-07); no PR has
   been merged since, every U-era commit went straight onto `audio-composite-support`.
   396 (81 %) have an empty body; exactly **one** (`e3e8b85`, U0) uses the §13 commit template.
3. **Four documentation planes coexist**: `unfold-pkg/docs/` (35 tracked + 3 untracked execution
   plans, 20.4k lines), `z-docs/` (76 files, ~9.6k lines incl. `stale/`), `.claude/PROTOCOL.md`
   (451 lines, last edited 2026-07-05) and the 58-file agent memory directory. They disagree with
   each other and with the tree; §4 lists 23 concrete disagreements.

---

## 1. Timeline

### 1.1 Phase table

Phases are segmented by commit content, not by calendar. Line counts are `git log --numstat`
sums for the inclusive commit range; `code` = `model_unfolder/**/*.py`, `tests` = `tests/`+
`test_support/` (corpus JSON excluded), `docs` = `docs/` + any `*.md`.

| # | phase | dates | range | commits | code +/− | tests +/− | docs +/− | total +/− | docs share of added |
|---|---|---|---|---|---|---|---|---|---|
| P1 | initial build + PyPI releases 0.2.0–0.2.11 | 05-06 → 06-09 | `b8f0c73..a8b3e4c` | 118 | +23,092/−9,484 | +3,166/−182 | +910/−439 | +45,776/−17,639 (18k "other" = examples HTML) | 2 % |
| P2a | op-graph, one-tower, diffusion v1, normalization; 0.2.12–0.2.16 | 06-10 → 06-26 | `a6573fc..3b8c04c` | 77 | +11,360/−5,601 | +5,402/−232 | +1,003/−15 | +18,681/−5,887 | 5 % |
| P2b | Sable/Dable everywhere, LLM sweep, honesty waves, "deciding the future", 0.2.17 | 06-26 → 07-07 | `d5fad5d..a6c405f` | 39 | +17,921/−2,910 | +9,163/−5,203 | +1,271/−253 | +29,462/−9,051 | 4 % |
| P3 | audio-gen (PR #14), run_77 fleet review, evidence-hardening `procedure 1–9` | 07-08 → 07-13 | `55bd81a..d475d87` | 12 | +6,772/−974 | +5,082/−675 | +2,699/−8 | +14,905/−1,671 | 18 % |
| P4 | U0/U1 + REC-0..7 + COR-0..5 recovery | 07-13 → 07-15 | `e3e8b85..18ac007` | 19 | +2,337/−669 | +3,750/−357 | +2,762/−24 | +13,531/−1,070 (4.7k corpus JSON committed) | 20 % |
| P5 | U2 R0–R9 (receipts, debt register, all nets blocking) | 07-16 → 07-19 | `7c6a298..1d0c72b` | 19 | +6,162/−1,427 | +3,992/−807 | +330/−144 | +11,013/−2,412 (z-docs written outside git this week) | 3 % |
| P6 | U3 ProgramIndex + owner resolver + execution flow | 07-19 → 07-28 | `7753dcc..4c2288e` | 78 | +19,249/−2,680 | +14,295/−1,124 | +4,580/−169 | +38,864/−4,042 | 12 % |
| P7 | U4 unknown-safety, U5 consumer firewall | 07-28 → 08-02 | `56758d7..e4c8c21` | 14 | +4,157/−2,139 | +4,021/−1,500 | +652/−32 | +9,768/−4,436 | 7 % |
| P8 | U6 attention, U7 FFN/cell, U8 position/mask | 08-02 → 08-14 | `74c50ec..58239de` | 46 | +42,425/−6,587 | +19,414/−3,786 | +2,381/−102 | +65,787/−11,065 | 4 % |
| P9 | U9 recursive modality | 08-14 → 08-18 | `a6e334a..32a9fc7` | 17 | +11,205/−4,043 | +3,879/−846 | +1,275/−46 | +16,391/−4,973 | 8 % |
| P10 | U10 diffusion root/stack/stream | 08-18 → 08-28 | `92200e1..0b1ed89` | 15 | +9,994/−3,549 | +6,175/−1,531 | +806/−120 | +17,320/−5,736 | 5 % |
| P11 | U11 UNet (open) | 08-28 → 09-01 | `51d016e..9b3cb7b` | 33 | +14,598/−408 | +7,492/−67 | +1,303/−60 | +23,718/−543 | 5 % |
| | **total** | | | **487** | | | | **+305,216 / −68,525** | |

Package size today: `model_unfolder/` = 133,052 Python lines; `model_unfolder/evidence/` alone =
95,837 lines in 147 files (the 08-31 audit measured 92.6k/142 at `cd5d7ff`; +3.2k in two days).

### 1.2 Commits per ISO week

| week | commits | week | commits | week | commits |
|---|---|---|---|---|---|
| W19 (05-04) | 50 | W25 (06-15) | 29 | W31 (07-27) | 49 |
| W20 | 26 | W26 | 14 | W32 | 37 |
| W21 | 5 | W27 | 22 | W33 | 18 |
| W22 | 18 | W28 (07-06) | 11 | W34 | 15 |
| W23 | 17 | W29 (07-13) | 52 | W35 | 26 |
| W24 (06-08) | 45 | W30 | 44 | W36 (08-31) | 9 |

Busiest days: 05-10 (41 commits — library packaging + 6 releases), 07-24 (30, U3-F), 07-27 (25,
U3 closure), 06-10 (22, op-graph), 08-14 (18, U8 close + U9 start).

### 1.3 Phase narrative

**P1 — initial build (May).** A notebook-grown Streamlit/HTML renderer became a library on
05-10 (`b42eadc make it into a library`, `38be517 Rename package to model-unfolder and add PyPI
release workflow`) and shipped six PyPI versions the same day. Content: GQA/MLA/MoE attention
views (05-11..13), sliding-window and caching arrows (05-14/15), first code-level evidence
(`e338159 code level reinforcement`, 05-17), JSON output (05-17/18), vision soft tokens and
Mllama multimodal (05-21..29), MTP (05-31). The `hf/` Gradio Space was created 05-31 (§2.5).
Documentation: `README.md` (3.3 KB) and `docs/models_supported.md` (now 0 lines). The first
recorded rule is 05-31: *never detect architecture by model_type* (memory
`feedback-no-model-type-detection`), followed 06-07 by *all config vocabulary in `everchanging/`
YAML*.

**P2a — the canonical op-graph (06-10 → 06-26).** 22 commits on 06-10 built the single op-graph
Region with two projections, "ONE tower backbone for every transformer tower" (`5ab4e21`), bare
in/out ports, cards with chips instead of numbers on blocks (`f494928`). Diffusion v1 landed
06-06/07 (`8d5ba98`, `919f0a1 introduce unet`), DiTs finalized 06-11, VAE 06-13, video 06-22.
`docs/BLOCK_STANDARD.md` (the block-worthiness law, Gate C) and `docs/MODULARIZATION.md`
date from 06-18/19; `docs/llm_connection_audit.md` / `diffusion_connection_audit.md` are the
first Sable-style two-way conformance audits (06-18/19). Releases 0.2.12–0.2.16.

**P2b — Sable, Dable, Her Eyes and the honesty waves (06-26 → 07-07).** `6b2c15a sable dable
everywhere` (06-27) made the harness (`model_unfolder/sable.py`, 7 mechanical nets + PNG
gallery + `bless()`), followed by the `path to singularity` commits (06-28/29). The core law
was stated by Soumil on 06-26 (*"I never — and I mean never — want this to be model-specific"*)
and written into `.claude/PROTOCOL.md` (final edit 07-05). The 21-LLM Sable sweep ran 07-05
(`previews/old/llm_sable_sweep_2026-07-05/`, 21 Her Eyes reviews), producing the
`MODEL_DISTRIBUTIONS.md` (2,524 lines) provenance trace and the 14-unit honesty waves
(`previews/old/SURGICAL_PLAN_HONESTY_WAVES.md`, all landed 07-06, memory
`project-honesty-waves-2026-07-06`). Two master plans were written and merged
(`previews/old/MASTER_PLAN.md` 07-05, `new_MASTER_PLAN.md`), plus `Code_restructure.md`,
`RESTRUCTURE_PLAN.md`, `TOWER_CENSUS.md`, `DISTRIBUTION_OF_INVOCATION.md`. Soumil's 07-05
**scope upgrade** — "FULL CONVERSION to code-based" authority (`z-docs/stale/PROJECT_CONTEXT.md:1110`)
— is the pivot the whole U plan descends from. PR #13 `transition` merged 07-07; Release
0.2.17 the same day (`push.txt`) — **the last release to date**.

**P3 — audio, run_77, evidence hardening (07-08 → 07-13).** `55bd81a` shipped audio-gen (PR
#14: composite seq2seq walk, K-codebook streams, 1-D audio DiT; scope blessed 07-07,
`z-docs/stale/SURGICAL_PLAN_AUDIO.md:26`). The **run_77 campaign** (07-08 → 07-11) rendered
370 catalogue models (`previews/run_77/_summary.txt`: run started 2026-07-08 01:29) and put
5×6 Opus reviewer fleets through 337 of 371 galleries — 1,011 per-model reports
(`sable_report.md`/`dable_report.md`/`her_eyes_review.md`), 12 completion sheets, 33
consolidated batch reports, a 21-thread systemic rollup (`_RUN77_REVIEW_INDEX.md`) and the
`RUN77_PROBLEM_MAP.md` (41 verified-general findings, 5 roots). 17 models skipped by directive
(07-10), 17 cancelled when Soumil stopped the reviewers (07-11); the SVD 3D-UNet test never
ran. `SURGICAL_PLAN_EVIDENCE.md` (07-11/12) then ran the "Unit 1 source parity … U2 default
kill" sequence and, on 07-12, `EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md` (2,467 lines) was
written; its §16 executed overnight on 07-13 as nine commits `procedure 1`…`procedure 9`
(`daf056f`→`d475d87`, suite 1,093 green).

**P4 — the vets and the recovery (07-13 → 07-15).** Soumil's independent vet of the nine
procedures produced `EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md` the same day (07-13): it
ratifies each deliverable (§4) and rejects every completion claim (§5 findings 5.1–5.10,
probe-confirmed V1–V5). U0 (`e3e8b85`) and U1 (`b2316f5`, committed by Soumil himself) were
then themselves vetted twice: `U0_U1_RECOVERY_EXECUTION_PLAN.md` (R-01..R-11 → REC-0..7,
`50d2408`..`cc5e328`) and `U0_U1_FINAL_RECOVERY_CORRECTION_PLAN.md` (C0–C4 → COR-0..5,
`602dcb0`..`325fbdf`), a fourth vet (`b32d5d8`) and a fifth directive (`18ac007`, 07-15).
Two process rules changed here: commits as `procedure N` (07-13, superseding "never commit")
and **push after every unit commit** (07-14: "Why aren't you pushing any commits… Please do
that", `PROJECT_CONTEXT.md:712`). The 25 corpus configs were committed (COR-0) and the
migration-claim gate refined at `(owner, mechanism)` scope ("mass registration =
debt-laundering").

**P5 — U2 (07-16 → 07-19).** Nine correction rounds R0–R9 in four days (receipt rail,
StructuralDebt register replacing four allowlists, every config occurrence located, ten nets
blocking, witness 26 = MusicGen). Two vets by Soumil reopened work (`c376115` 07-17: "exactness
is a PROOF, provenance is first-class"; the 07-18 rule that every commit needs both the focused
and the broad gate after `a443661` fixed regressions the narrow receipts missed). **The z-docs
plane was created in this window** (all chapter files dated 07-14..07-16; `07-current-state`
last touched 07-24) — the third documentation plane, outside git.

**P6 — U3 (07-19 → 07-28).** Soumil's ten binding boundaries and revised phase ladder
(`docs/U3_RUNBOOK.md:19-100`, 07-19), then U3-A/B/C, an independent audit that corrected B/C
(`bfa2d75`), Codex V4 approvals for B1/B2 (`933bb90`, `85967d7`), execution-flow phases 2–5,
and the F-series reader migrations (30 commits on 07-24 alone). Closure was ratified 07-27
("Do it", `U3_COMPLETION_MASTER_PLAN.md:553`) with a frozen legacy quarantine, and the
DeepSeek/GLM FFN re-proof (`4bd1395`) produced the **§7 mandatory lost-detail procedure**
(`U3_ACCOMPLISHMENTS_AND_CURRENT_PROCEDURE_AUDIT.md:499`, 07-28). Docs: eleven `U3_*` files
(3,846 lines) plus `U_PLAN_RISK_OWNERSHIP_AND_STOP_GATES.md`.

**P7–P11 — the mechanism units (07-28 → 09-01).** U4 (typed unknown, six sub-units each with
Soumil's artifact approval 07-28..07-31), U5 (`569cd8b`, terminal-consumer firewall), U6 (25
commits, 28-witness bless `e758be4`), U7 (Granite + six witnesses approved 08-06), U8 (17k
lines of position/mask readers; 29-witness re-bless approved 08-14), U9 (08-14..18; §8
honesty delta approved 08-18), U10 (08-18..28; 29-witness delta approved 08-28), U11 (A1 →
F in progress; the uncommitted tree is U11-F work in `evidence/invocation_source.py`,
`unet_attention_source.py`, `unet_root_preprocess.py`). Every unit from U8 on has a
`docs/U<N>_*_EXECUTION_PLAN.md` of 277–1,665 lines that is simultaneously plan, ledger and
receipt log. The 08-31 audit (`z-docs/09-unit-verdicts/`) then graded U0–U10 and this research
program started 09-01.

### 1.4 Release cadence (tags on `origin`, all lightweight)

| version | date | commit | note |
|---|---|---|---|
| v0.2.0–v0.2.5 | 2026-05-10 | `38be517`…`d0e4345` | six releases in one day (library split, Qwen, fallbacks) |
| v0.2.6 / v0.2.7 | 05-12 / 05-13 | `3b3c9a7` / `42c30d7` | MoE explorable; MLA |
| v0.2.8 / v0.2.9 | 05-27 / 05-28 | `fff4613` / `26e65b9` | modularity; adaptability |
| (0.2.10) | 05-31 | `e90301a` | pyproject only, **no tag, never published** |
| v0.2.11 | 05-31 | `95a9cd1` | scale normalization, MTP |
| v0.2.12 / v0.2.13 | 06-11 / 06-17 | `a72f133` / `7ec099c` | op-graph; DiTs |
| v0.2.14 / v0.2.15 | 06-20 | `ef6ccd8` / `89ba6f9` | normalization; MoE fixes |
| v0.2.16 | 06-26 | `3b8c04c` | Sable harness era |
| v0.2.17 | 07-07 | `55513f6` | last release; `hf/` Space pins `model-unfolder[hf]==0.2.17` |

Since 0.2.17: 240 commits, +215k lines, zero releases. `model_unfolder/__init__.py:27` still
says `__version__ = "0.2.15"` (stale since 06-20; flagged in `z-docs/08-reference/verification-ledger.md:456`
on 07-16, never fixed). `dist/` holds only 0.2.0. `PROJECT_CONTEXT.md` Part 8 item 5 ("retro-tag
0.2.10–0.2.16") is moot: 0.2.11–0.2.16 are tagged on origin; only 0.2.10 never existed as a tag.

---

## 2. Documentation planes — inventory

### 2.1 `unfold-pkg/docs/` (tracked: 35 `.md`; untracked: 3 `.md` + 1 `.html`; `MODULARIZATION.md` deleted in the working tree)

| path | lines | purpose | status | evidence |
|---|---|---|---|---|
| `EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md` | 2,983 | THE binding plan U0→U15; §16 ledger, §20 runbook, §21 append-only tracker | **current authority**, internally stale (§3 H-table from 07-13 never updated; §21 has 29 rows, 8 "supersedes", U7 ×6, U0 ×4) | 38 commits, last `51d016e` 08-28 |
| `U11_UNET_EXECUTION_PLAN.md` | 1,243 | U11 plan + tracker | **live** (A1–E2c DONE, F active, G/H/I pending) | 23 commits, last 09-01 |
| `U10_DIFFUSION_ROOT_EXECUTION_PLAN.md` | 1,665 | U10 plan + receipts | DONE record | last 08-28 |
| `U9_RECURSIVE_MODALITY_EXECUTION_PLAN.md` | 277 | U9 plan | DONE record | last 08-18 |
| `U8_POSITION_MASK_SCHEDULE_EXECUTION_PLAN.md` | 1,320 | U8 plan | DONE, **header stale** ("ACTIVE… No manifest re-blessed" vs §U8-G21 approved and master §21 DONE) | last 08-14 |
| `U6_U7_U8_QUALIFICATION_MATRIX.md` | 374 | U6–U8 audit receipt | header stale ("Soumil's artifact decision remain" vs line 362 "subsequently approved") | last 08-14 |
| `U7_CONDITIONAL_SHARED_FFN_PROOF.md` | 508 | DeepSeek/GLM FFN proof | DONE record | last 08-06 |
| `U3_RUNBOOK.md` | 701 | U3 execution log (LIVE in title) | closed 08-14 by header correction; historical `[~]` rows | 27 commits |
| `U3_COMPLETION_MASTER_PLAN.md` | 732 | U3 closure plan, ratified 07-27 | DONE record | |
| `U3_ACCOMPLISHMENTS_AND_CURRENT_PROCEDURE_AUDIT.md` | 723 | audit + Gates A–J + **§7 lost-detail procedure** + Soumil's checklist | **current procedure** (binding, not a gate) | 07-28 |
| `U3_SEMANTIC_DELTA_ADJUDICATION.md` | 252 | U3-C5 delta receipt | DONE | 07-28 |
| `U3_D_TO_H_EXECUTION_AND_VET_PLAN.md` | 485 | reader-migration plan | superseded by completion plan | 07-24 |
| `U3_READER_INVENTORY.md` | 289 | pre-migration reader census | **historical** (says so; names deleted readers) | 07-27 |
| `U3_CURRENT_READER_INVENTORY.md` | 71 | generated inventory (5 quarantined readers, 20 `ast.parse` sites) | current, generated | 08-28 |
| `U3_A2_RECON.md`, `U3_EXECUTION_FLOW_CORPUS.md`, `U3_F0_REPEATED_CHILD_RECON.md`, `U3_F3_F4_BOOKEND_CUTOVER.md` | 203/92/176/158 | recon reports | historical | 07-23/24 |
| `U_PLAN_RISK_OWNERSHIP_AND_STOP_GATES.md` | 378 | who decides kernel decisions (Soumil + Codex), stop gates | current supplement | 07-20 |
| `U2_COMPLETION_SUMMARY.md` | 68 | R9 summary | DONE | 07-19 |
| `U2_DEFINITIVE_COMPLETION_SPEC.md` | 922 | U2 spec | **untracked** (Soumil's, deliberately) | mtime 07-18 |
| `U0_U1_RECOVERY_EXECUTION_PLAN.md` / `U0_U1_FINAL_RECOVERY_CORRECTION_PLAN.md` | 1,290 / 790 | the second and third vets of U0/U1 | **untracked** (`SCHEDULER_VERTICAL_TRACKER.md:45`: "Soumil's U0_U1_* plan docs stay untracked") | mtime 07-18 |
| `EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md` | 2,467 | the 07-12 plan; §16 procedures; §17 COR receipts | **superseded as tracker** (own header, 07-13); still the only record of H0–H10 rationale | 8 commits |
| `SCHEDULER_VERTICAL_TRACKER.md` | 71 | mis-scoped 5-commit plan | **retired** (own header; scope correction 07-16) | |
| `TRUE_CONFIG_PLAN.md` | 141 (uncommitted rewrite −326/+…) | `true_config()` worklist | `#TODO` — no code (`grep true_config model_unfolder` = 0 hits) | c404462 07-13 |
| `COR5_NET1_MIGRATION_DEBT.md` | 121 | generated census: standing 0, pending 16 | current, generated | 08-18 |
| `sable_dable_playbook.md` | 995 | agent playbook for Sable/Dable | **stale** ("Never run git commit"; pre-U vocabulary) | 06-28 |
| `BLOCK_STANDARD.md` | 365 | block-worthiness law (Gate C) | durable, references deleted `MODULARIZATION.md` | 06-19 |
| `blocks_supported.md` | 229 | view registry doc | stale (06-10 routing table) | |
| `llm_connection_audit.md` / `diffusion_connection_audit.md` | 157 / 131 | first two-way conformance audits (transformers 5.12, diffusers 0.38) | historical | 06-18/19 |
| `coverage_audit.md` / `serve_audit.md` | 89 / 155 | generated 07-04 (49 / 105 models) | stale generated | 07-04 |
| `models_supported.md` | 0 | empty since 05-31 | dead | |
| `architecture_atlas.html` | — | untracked HTML atlas (67 KB, 07-19) | unknown purpose, unreferenced | |

Also: `unfold-pkg/archive/hardening_completion_log_S14-S15.md` (§14/§15 of the hardening plan,
archived 07-12) and `unfold-pkg/README.md` (3.3 KB, 05-21, still says "only config.json is
downloaded" — true, but the product now also reads installed modeling source).

### 2.2 `z-docs/` (outside git; 76 files)

| chapter | files / lines | last mtime | status | evidence |
|---|---|---|---|---|
| `README.md` | 45 | 08-31 | current index; claims "07 is the only live status area" | contradicted by `unfold-pkg/docs/` trackers (§4) |
| `00-start-here/` (doctrine, how-to-use) | 3 / 164 | 07-16 | **durable, current** — the best short statement of the laws | matches master plan §1 |
| `01-product/` (contract, scope, true-config, next-architecture, finish-line) | 6 / 383 | 07-16 | durable; `true-config.md` describes unbuilt code | |
| `02-architecture/` | 5 / 293 | 07-16 | durable, pre-U3 vocabulary (no ProgramIndex/ReaderResult) | |
| `03-evidence/` | 4 / 208 | 07-16 | durable | |
| `04-domains/` | 4 / 141 | 07-14/16 | thin (diffusion 37 lines); pre-U10 | |
| `05-verification/` | 3 / 121 | 07-16 | pre-U2-R8 (does not know the ten blocking nets) | |
| `06-development/` | 3 / 128 | 07-16 | current rules (producer-first, failure reporting) | |
| `07-current-state/` | 12 / 1,773 | **07-24** | **stale**: README lists 5 files, directory holds 12 (u2-definitive-progress 647 lines, u2-receipt-accountability 389, u3-runbook 457, u3-reader-inventory 239, r6/r7 maps, U2 summary); nothing after U3-B1 | U4–U11 (07-28→09-01) absent |
| `08-reference/` (canonical-config 584, code-map 76, glossary, verification-ledger, documentation-boundary) | 6 / 852 | 07-16 | partly stale (§3 spot checks) | |
| `09-unit-verdicts/` | 18 / 2,346 | 08-31 / 09-02 | dated audit (U0–U10 grades, systemic findings, experiments, method judgment) | its own §Part 2 admits it is "a fourth documentation plane" |
| `10-full-research/` | this program | 09-02 | in progress | |
| `stale/` (PROJECT_CONTEXT 2,424, SURGICAL_PLAN_EVIDENCE/AUDIO, RUN77_PROBLEM_MAP, TO_SERVE2, CODEBASE_COHERENCE_AUDIT, CONFIG_* , AGENT_ORCHESTRATION_PLAYBOOK) | 11 / 4,810 | 07-07..07-14 | quarantined archive (own README) | `PROJECT_CONTEXT.md:8` links `../08-reference/legacy-removal.md` — **does not exist** |

### 2.3 `.claude/PROTOCOL.md` (451 lines, last mtime 2026-07-05) and its ancestor

The working "nature" document: core law (evidence never identity), Gate 0/A/B/C, Coverage
(three layers), Dable, Her Eyes (five questions, fixed template, "auto-run after every
bless"), closing rules ("Never `git commit`", "Scope: LLMs and diffusion only (no SSM/Mamba)"),
design habits. `done/forced/CLAUDE.md` (318 lines, mtime 06-20) is its direct predecessor
(same opening paragraph, no core-law section, no Her Eyes). PROTOCOL has not been edited since
the 07-05 scope upgrade; its opening sentence still says the product "turns a HuggingFace
**config** into an honest… diagram" and "only config-derived facts". Memory
`reference-procedures-in-protocol-md` (07-05) records that procedures moved here from CLAUDE.md.

### 2.4 Agent memory (`~/.claude/projects/…/memory/`, 58 files)

Nine `feedback-*` files carry Soumil's rulings verbatim with dates (05-31, 06-07, 06-26,
07-07, 07-13/14, 07-16 ×2, 07-18, 07-20); 47 `project-*` files are phase summaries. This is the
only plane where the *dates* of rulings survive; several rulings exist nowhere else (e.g. the
"procedure N" commit rule, the push flip, "PROCESS RUN" header). It is per-machine and invisible
to git.

### 2.5 `unfold-pkg/previews/` (gitignored, 412 MB) and the campaign record

| directory | size | dates | content | cited by docs as |
|---|---|---|---|---|
| `run_77/` | 238 MB | 07-08 → 07-11 | 371 model folders (425 entries), 337 with 3 reports each; `_RUN77_REVIEW_INDEX.md`, `_CAMPAIGN_STATE.md`, 12 sheets, 33 batch reports, `_problem_map/`, `_results.tsv` (370 run: OK 6, ACCESS 37, PARSE 3, ERROR 5, SKIP 319) | `previews/run_77/_problem_map/01..07_` |
| `old/` | 106 MB | 06-27 → 07-07 | 13 pre-U planning docs (MASTER_PLAN, new_MASTER_PLAN, RESTRUCTURE_PLAN, Code_restructure, MODEL_DISTRIBUTIONS, DISTRIBUTION_OF_INVOCATION, INVOCATION_FLOW_DEEPSEEK_V3, TOWER_CENSUS, HER_EYES_THEME_MAP, SURGICAL_PLAN_HONESTY_WAVES, MODEL_INVENTORY, toserve*.md) + `llm_sable_sweep_2026-07-05/` (21 galleries + Her Eyes) + `theme_l_fix_2026-07-07/` + `part1/` + 7 model galleries | docs cite `previews/llm_sable_sweep_2026-07-05/` and `previews/theme_l_fix_2026-07-07/` **without** the `old/` prefix — both paths now MISSING |
| `individual_images/` | 29 MB | 07-06 | 30 default `save_images()` galleries | |
| `cor5_visual_matrix_2026-07-14/`, `final_visual_matrix_2026-07-15/` | 15 MB each | 07-14/15 | the §11/§13.3 visual matrices (5 folders: transformer, multimodal, diffusion-transformers, diffusion-pipelines, unet); PNG only, no `.md` | hardening plan §17 |
| `u6_diffusion_rail_2026-07-12/`, `unit1_source_parity_2026-07-11/` | 4.8 / 2.2 MB | 07-11/12 | SURGICAL_PLAN_EVIDENCE galleries | cited |
| `audio_first_iterations_2026-07-07/`, `tts_first_iterations_2026-07-08/` | 1.6 / 1.2 MB | | audio unit galleries | cited |
| (cited, moved) `previews/u2_default_kill_2026-07-11/`, `u2_readers_2026-07-11/` | | | now under `done/forced/previews/` | MISSING at cited path |

Blessed galleries live in `unfold-pkg/tests/sable_test_corpus/galleries/<slug>/` (29, all
regenerated 08-28), **gitignored** (`.gitignore`: "Generated galleries only"), with hashes in
the committed `tests/preservation_manifest.json` (29 witnesses) and `tests/sable_test_corpus/*.json`
(29 fixtures). The pixels that every bless was approved on are therefore not in version control.

### 2.6 Workspace-root artefacts outside both planes

| artefact | what it is | dates | product intent it carries |
|---|---|---|---|
| `hf/` (own git repo, remote `huggingface.co/spaces/SoumilB7/Unfold`) | Gradio Space, `app.py` 4.4 KB: paste an HF repo id (+ optional token), renders `unfold()` HTML in an iframe; `DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"`; `requirements.txt` pins `model-unfolder[hf]==0.2.17`; README: "View any model's internal architecture just by HF tag" | 10 commits, 05-31 → 07-07 ("version update"); branch `transition` too | **the only deployed user surface**; frozen on 0.2.17, so none of U0–U11 is visible to a user |
| `done/TO_SERVE.md` | "The Big Grill 🔥" — 447-line catalogue of open-weight generative families by lab (Parts 1–5: text LLMs, VLMs, image, video, audio + exclusions: no SSM/RWKV/JEPA/CNN/encoder-only), 299 HF ids | mtime 07-07 18:17; no git | 156 of its 299 ids are `run_77` folders, and run_77 started 8 h later → **(inference)** it is the run_77 roster source. It is **not** the source of `previews/old/toserve*.md`: only 1–3 identical lines, 66/118 id overlap. `toserve.md` (118 ids, "the canonical list… coverage standard", June) fed `scripts/coverage_audit.py`; `toserve_model.md` is Soumil's own brief for a verification skill ("Each architecture will have n things: connecting lines, blocks, connectors…"); `toserve_layers.md` is the 06-27 IMP audit |
| `z-docs/stale/TO_SERVE2.md` (517 lines) | successor catalogue | 07-07 | fourth catalogue; no doc names which is the support set |
| `done/tryrun.ipynb` (884 KB) | Soumil's manual try-run, 21 code cells; `from unfold import unfold` → later `from model_unfolder import unfold`; targets: Meta-Llama-3-8B, Llama-2-7b(-hf), huggyllama/llama-7b, gemma-4-e4b/26B-A4B/31b, DeepSeek-V3, Mistral-7B-v0.3, Mistral-Medium-3.5-128B, Qwen3.5-122B-A10B, Qwen3.6-35B-A3B-NVFP4, c4ai-command-r-v01, gpt-neox-20b | 05-31 | the May target set; **contains an HF token literal** (not reproduced here) |
| `finalize.ipynb` (3.7 MB, root) | 24 code cells, `unfold(...)` calls incl. `inspect_code=True`; targets: HunyuanVideo, gemma-2-9b-it, **recurrentgemma-9b**, diffusiongemma-26B-A4B-it, SDXL-base, gemma-4-e4b, ideogram-4-nf4, DeepSeek-V3.1-Base, DeepSeek-V4-Flash, sarvam-30b, gpt-neox-20b, Qwen2-VL-7B, Wan2.2-TI2V-5B, CodeLlama-7b, DeepSeek-R1, Llama-3.2-11B-Vision, pixtral-12b, gpt-oss-120b | 06-29 | the late-June "does it work on the hard ones" set; recurrentgemma contradicts PROTOCOL's "no SSM" scope; **contains an HF token literal** |
| `done/model.json` | `model_unfolder.expanded` schema 3.1 JSON for gpt-oss-120b | 06-10 | the JSON consumer contract as of 06-10 |
| `done/previews/previews/` | `v1-done-ignore/` (71 MB, 14 HTML 06-20..06-27: DeepSeek-V3/V3.2-Exp, Llama-3.2-11B-Vision…), `individual_images/` (19 MB, 07-06), `new/` (7 jpgs) | June–July | pre-Sable output snapshots |
| `done/forced/` | `CLAUDE.md` (PROTOCOL ancestor) + `previews/u2_readers_2026-07-11`, `u2_default_kill_2026-07-11` | 06-20, 07-11 | |
| `push.txt` | release recipe for 0.2.17 (`git tag v0.2.17 … push`) + "# change pyproject toml, __init__" + "# Vonxtral left" | 07-07 | the author's own reminder that `__init__` must be bumped (still not done) and that Voxtral (audio) was the next target **(inference on the typo)** |
| `unfold-pkg/examples/` | 14 HTML (8 LLM: deepseek-v3, gemma-4-31b/e4b, kimi-k2, llama-3-8b, mistral-7b — 05-10; 6 UNet: sd1.5/2.1/sdxl-base/refiner/kandinsky2.2/deepfloyd-if — 06-07..10; flux-1-dev 06-10) + `images/` | 05-10, 06-10 | README's showcase; frozen pre-op-graph |
| `unfold-pkg/.github/workflows/release.yml` | tag-triggered PyPI publish (`v*`) | 05-10 | |
| `unfold-pkg/.claude/worktrees/verify-*` | harness-owned isolated receipt worktrees (REC-0 §6.2) | | |

---

## 3. Spot-checks: documentation claims vs the tree

| # | claim (source) | tree | verdict |
|---|---|---|---|
| 1 | code-map: readers at `model_unfolder/evidence/{ffn,position,vision,audio,projector,fusion}.py` (`08-reference/code-map.md:41`) | `ffn.py`, `position.py`, `audio.py` deleted (U7 `1ea4036`, U8, U9); `program_index.py`, `import_source.py`, `legacy_reader_quarantine.py` and ~140 others unlisted | **stale** |
| 2 | code-map: `tests/test_{ffn,…}_evidence.py` (`:76`) | `test_ffn_evidence.py` missing | stale |
| 3 | code-map: foreign formats via `everchanging/transformer/aliases.yaml` (`:13`) | exists only as `model_unfolder/everchanging/transformer/aliases.yaml` | path wrong (minor) |
| 4 | verification-ledger: "one owner-bound raw program index feeds all evidence readers — **target**" (`:34`) | `evidence/program_index.py` (`build_program_index`, line 2818) built in U3; readers migrated U3-F..U10 | superseded — now largely true |
| 5 | verification-ledger: `pyproject 0.2.17` vs `__version__ 0.2.15` "separate packaging fix required" (`:40`) | still `0.2.17` / `0.2.15` | **still true, unfixed 7 weeks** |
| 6 | implementation-state: "reproducible clean-checkout preservation for **25** representative models" (`:10`) | 29 witnesses in `tests/preservation_manifest.json` and 29 corpus fixtures | stale count |
| 7 | implementation-state: "census recorded **267** exact accessed-but-unconsumed occurrences" (`:99`) | `docs/COR5_NET1_MIGRATION_DEBT.md:53` standing 0, pending 16 | stale |
| 8 | implementation-state: "source readers still overlap instead of consuming one program index" (`:86`) | U3 closed 07-27/08-14 | stale |
| 9 | config-and-yaml-debt table rows for `layer_topology.yaml`, `layer_schedules.yaml`, `layer_types.yaml`, `diffusor/config_facts.yaml` (`:25-30`) | all four deleted (U7/U8/U10); `conditioning.yaml`, `decoderness.yaml` still exist | 4 of 12 rows stale |
| 10 | 07-current-state README lists five files | directory holds 12; seven dated trackers unindexed | stale index |
| 11 | z-docs README: "Chapter 07 … must be rewritten when state changes" | 07 last modified 07-24; eight units closed since | **6 weeks stale** |
| 12 | `stale/PROJECT_CONTEXT.md:8` → `08-reference/legacy-removal.md` | file absent | broken link |
| 13 | U8 plan header "ACTIVE… No manifest or gallery has been re-blessed" | master §21 U8 DONE `fd20ac4`, ruling 08-14 (`:2946`) | stale header |
| 14 | master plan §3 "Authoritative status" H0–H10 table | written 07-13, never updated; §21 rows are the real status | stale section inside the authority doc |
| 15 | PROTOCOL: "Scope: LLMs and diffusion only (no SSM/Mamba)" (`:438`) | audio-gen shipped 07-08; U6 `01a98cf bind recurrent mixer geometry`; `finalize.ipynb` targets recurrentgemma | stale scope |
| 16 | PROTOCOL / playbook / PROJECT_CONTEXT:182,903: "Never `git commit` — Soumil commits" | 487 commits; 78 bodies carry a Claude `Co-Authored-By` trailer; conventional `feat(u11):` subjects | superseded 07-13/14, never rewritten |
| 17 | PROTOCOL: "after every bless/re-bless, the new gallery gets her review" (`:405`, `:435`) | 8 of 29 blessed galleries have `her_eyes_review.md`; GLM-4.5's has **0 image rows** (void by its own completeness rule); qwen2-vl's is stale (3 stale, 1 missing image); U8/U9/U10 re-blesses (49+73 PNGs) got none | rule not followed since 08-06 |
| 18 | `sable_dable_playbook.md:11` "/previews … folders with model name → images, html, report and Manifest" | `previews/` gitignored; blessed galleries under `tests/…/galleries/` also gitignored | stale + evidence unversioned |
| 19 | docs cite `previews/llm_sable_sweep_2026-07-05/`, `previews/theme_l_fix_2026-07-07/`, `previews/u2_default_kill_2026-07-11/` | moved to `previews/old/` and `done/forced/previews/` | 3 broken citations |
| 20 | 09-verdicts method judgment: "57 of August's 105 commits have empty bodies" | `git log`: August = 111 commits, **111** empty bodies | the audit's own metric is wrong (direction unchanged) |
| 21 | 09-verdicts u03: master §21 U3 row quotes `58239de` quarantine counts (17/17) against closing `4bd1395` (25/23) | confirmed by reading the row | doc/tree disagreement inside the tracker |
| 22 | U3 boundary 8 "external import closure is EXPLICIT… external source nodes" (`U3_RUNBOOK.md:41`) | `build_program_index(bundle, *, external_nodes=())` — no caller passes `external_nodes` (grep: only the definition, `program_index.py:2818-2897`) | boundary declared, never crossed |
| 23 | `01-product/true-config.md` + `docs/TRUE_CONFIG_PLAN.md` describe `true_config()` | zero occurrences of `true_config` in `model_unfolder/` | design only, both docs say so |

Score: 23 checked, 3 still true (5, 22 as stated, 23 as stated), 17 stale or wrong, 3 path/link
breaks.

---

## 4. Decision log

### 4.1 Rulings by Soumil (dated; source; downstream consequence)

| date | unit | ruling | source | consequence |
|---|---|---|---|---|
| 05-31 | build | never detect structure by `model_type`/family name; config fields only | memory `feedback-no-model-type-detection` | `_MIXER_FIELD_SIGNATURES`; the legacy LayerNorm model_type list tolerated as sole exception |
| 06-07 | build | all config-parsing vocabulary is YAML in `everchanging/` | memory `feedback-config-vocab-in-everchanging` | `everchanging/{transformer,diffusor,conformance}/`; later half-deleted by U7/U8/U10 |
| 06-26 | Sable | "I never — and I mean never — want this to be model-specific" | memory `feedback-detect-from-evidence-never-identity`; PROTOCOL:10-36 | identity guard (H0/U-era), `class_defaults` tables deleted |
| 06 (undated) | Sable | "hand over a model id, run one thing, never see that id with an issue again" | memory `project-sable-harness` | `sable()`, `bless()`, `check_regression` |
| 07-05 | scope | **FULL CONVERSION to code-based authority**; config demoted to oracle-missing fallback; Tier-2 structural parse directly, not a Tier-1 "humanizer"; "complete Group 1, then Group 2" | `stale/PROJECT_CONTEXT.md:1110,1263,1512` | the entire U plan |
| 07-07 | docs | context docs must EVOLVE, never append; audio scope blessed; tier doctrine confirmed | memory `feedback-context-evolves-not-appends`; `SURGICAL_PLAN_AUDIO.md:26`; `PROJECT_CONTEXT.md:2221` | z-docs boundary rule (07-16) — then violated by §21 append rule |
| 07-08 | audio | "make a rigorous check on everything" | `SURGICAL_PLAN_AUDIO.md:279` | rigorous gate; PR #14 |
| 07-10 / 07-11 | run_77 | "just do the main ones" → 17 skipped; reviewers stopped → 17 cancelled | `run_77/_CAMPAIGN_STATE.md` | SVD 3D-UNet test never ran (still open) |
| 07-12 | evidence | "get it done perfectly, strategically, eradicated — nothing compromised"; "bleed it" (P2–P4) | `SURGICAL_PLAN_EVIDENCE.md:8,244` | hardening plan §16 → nine procedures |
| 07-13 | process | commit as `procedure N` (supersedes never-commit) | memory `feedback-user-commits-himself` | 9 commits; then U0/U1 |
| 07-13 | vet 1 | independent vet of procedures 1–9: deliverables ratified, completion rejected; tracker vocabulary PENDING/ACTIVE/BLOCKED/DONE only | master plan §3–§5, §13 | master plan becomes THE tracker; V1–V5 probe-confirmed |
| 07-14 | process | **push after every unit commit** | `PROJECT_CONTEXT.md:712`; memory | 23 commits pushed at once; every unit since pushed |
| 07-14 | COR | migration claims at exact `(owner, mechanism)` scope; "mass registration = debt-laundering"; COR-5 not ratified — 3 gaps | memory `project-cor-recovery-complete`; hardening plan §17 | `b32d5d8` corrections; Net-1 blocking |
| 07-15 | COR | green-conditional: "when all green, mark U0/U1 REVIEWED/DONE, unlock U2" | hardening plan `:2461`, memory | `18ac007`; the only DONE marks Claude was allowed to write |
| 07-16 | U2 | **producer-first one-way evidence** doctrine (binding); scheduler vertical mis-scoped → retired | memory `feedback-producer-first-one-way-evidence`; `SCHEDULER_VERTICAL_TRACKER.md:51` | `00-start-here/project-doctrine.md`; U2.1–U2.9 |
| 07-16 | U2.2a | gates assert the predicate, not a proxy | memory `feedback-gates-must-assert-predicate-not-proxy` | three green proxies removed |
| 07-17 | U2.2a | "exactness is a PROOF, provenance is first-class"; class-hydrated provenance required | `c376115`; `07-current-state/u2-receipt-accountability.md:74,325` | `inspected` provenance |
| 07-18 | U2-R5 | every R5–R9 commit needs focused AND broad gate; R5 locked details; MusicGen added as witness 26 at R9; "2 commits now" | memory `feedback-commit-receipt-runs-broad-gate`; `u2-definitive-progress.md:402,499,516` | `a443661` forward fix |
| 07-19 | U2/U3 | U2 final vet: both counterexamples killed; re-bless **delegated**; "go ahead" U3; ten binding boundaries; revised phase ladder; U3-A schema contract | `U2_COMPLETION_SUMMARY.md:8,25`; `U3_RUNBOOK.md:11,19-49,89,149` | boundary 8 (import closure) declared, never implemented |
| 07-20 | U3 | independent audit → B/C corrected baseline; "PROCESS RUN" header rule | `bfa2d75`; memory `feedback-process-run-writeup` | |
| 07-21/22 | U3 | B1, B2 approved via Codex V4 | `933bb90`, `85967d7` | Codex named as kernel-decision co-owner (`U_PLAN_RISK…:52-69`) |
| 07-27 | U3 | "Do it" — quarantine-based closure ratified | `U3_COMPLETION_MASTER_PLAN.md:553` | U3 DONE on `4bd1395` |
| 07-28 | U3-C5 / U4-A | nine `RMSNorm → Norm` T5/UMT5 views approved; U4-A attention delta approved (14 galleries re-blessed) | `U3_SEMANTIC_DELTA_ADJUDICATION.md:204`; master `:1548,1622` | first "honesty" bless |
| 07-29 / 07-30 / 07-31 | U4-B/D/F | 26-witness honesty deltas approved | master `:1549-1553,1743,1951,2075` | approval and bless in one commit (`c2a3c23`) |
| 08-03/04 | U6 | Qwen3.5/DBRX witness expansion + 27-witness re-bless approved "galleries byte-identical" | master §21 U6 row; `09/u06.md:61` | description false: 25/26 gallery hashes changed |
| 08-06 | U7 | six additional witnesses approved after seven PNGs inspected; Granite blessed | `U7_CONDITIONAL_SHARED_FFN_PROOF.md:423,467` | last dated PNG-level inspection on record |
| 08-14 | U8 / U3 | U8 29-witness position/mask honesty transition approved; U3 closure correction | master `:2946`; `58239de` | U8 DONE `fd20ac4` |
| 08-17/18 | U9 | §8 finite honesty delta approved ("two fabricated cell drills") | `U9_…PLAN.md:209-216` | Qwen2-VL vision cell lost (audit: it was correct) |
| 08-17 | docs | wants the three-plane doc sprawl restructured | memory `project-u8-done-u9-active` | not done; a fourth plane (09) added instead |
| 08-28 | U10 | exact 29-witness delta approved; 28 galleries re-blessed | `U10_…PLAN.md:1007,1657` | PixArt/SD3.5 blank boxes blessed |
| 08-31 / 09-01 | audit | U0–U10 verdicts requested; then "I won't let you judge until you have the entire codebase and the intent" | `09-unit-verdicts/`; memory `project-full-research-program` | this program |

### 4.2 Correction rounds

| round | dates | what was wrong | what fixed it |
|---|---|---|---|
| V1–V5 (vet of procedures 1–9) | 07-13 | `resolve_aliases` unwired (fictional consumed canonicals), bare-key unread audit, key-only pending-debt excusal, decorator whole-body exemption, weak `StructuralWrite.key` | master plan §5.1–5.10; R0 probe tests; U1 |
| REC-0..7 (2nd vet of U0/U1) | 07-13/14 | R-01..R-11: bind-as-consumed, owner+leaf paths, null=missing, vacuous nets, config-authored facts, **15/25 diffusion IR witnesses drifted undetected** because U0's baseline compared nothing | `50d2408`..`cc5e328`; deterministic harness + poison matrix + clean-checkout generator |
| COR-0..5 + 4th vet + 5th directive | 07-14/15 | C0–C4: preservation not self-contained, approximate joins, ambiguity reaching `dim 0`/SiLU retries, flattened modality scopes, generic projector width; then 3 soundness gaps in COR-5 (target-bound claims, corpus anti-vacuity, occurrence-exact debt 267 vs 250) | `602dcb0`..`325fbdf`, `b32d5d8`, `18ac007` |
| U2.1 / U2.1a / U2.2a vet | 07-16/17 | shipped Net-2 contract missing; receipt must reach an allowed consumer; **fabricated config paths** satisfied three green proxies; provenance | `81317d6`, `49d0d93`, `86f03ba`, `28a66ad`, `c376115` |
| U2 R0–R9 | 07-17 → 07-19 | narrow receipts let 8 regressions push (`a443661`); R5 fields decorative (`a778fb7`); four debt allowlists; unlocated occurrences; nets advisory; zero-drift red awaiting re-bless | `ac860e6`…`1d0c72b` |
| U3-B/C independent audit; A1 kernel correction; B1 V1→V4; B2 Codex V4; six execution-substrate rounds (closed-world claims removed); C5 semantic delta adjudication | 07-20 → 07-28 | occurrence ownership inexact; ReaderResult laws open; identifier census; closed-world coverage certificate; T5/UMT5 norm demotion needed adjudication; DeepSeek/GLM FFN lost then re-proven | `bfa2d75`, `c5b3c82`, `933bb90`, `85967d7`, `a1f5024`/`6676999`/`4bd1395` |
| U4-D contract alignment | 07-30 | tests contradicted code authority | `a08a561` |
| U6 | 08-04 | bless `e758be4` asserted byte-identical galleries — false (audit) | never corrected |
| U7 class-count cleanup; Falcon re-proof | 08-05 | owner-unqualified warning gate | `a8efdbe` |
| U9 fusion negatives | 08-14 | name-taint in negatives; modality ownership paths not executable | `b904d69`, `dd58aea`, `db02668` |
| U11 | 08-29/30 | cell operation protocols leaked; cutover dependency order wrong | `43e4e53`, `c0d26bc` |
| 09-unit-verdicts + experiments + method judgment | 08-31 → 09-02 | soundness ratcheted, recall never; unindexed ≠ unknown; U6/U8/U9/U10 verified-detail losses; two of its own claims corrected in Part 2 | this program |

### 4.3 Contradictions between recorded decisions

1. **U4 step 6 vs §7** — master plan `§20.7` step 6 (07-28): "If a known-good model becomes
   unknown, fix its evidence reader in the matching U6–U13 slice; do not restore the global
   default." `U3_ACCOMPLISHMENTS…:499-575` (same day): never call generic output honest before
   re-proof is attempted; "'the new reader returned None' is never an acceptable reason." Step 6
   won four times (U6 `e758be4`, U8 T5 bias, U9 Qwen2-VL cell, U10 PixArt/SD3.5).
2. **"Never git commit" vs agent commits** — PROTOCOL:437, playbook:31, hardening plan :1267
   ("Soumil alone decides and performs blessings/commits"), PROJECT_CONTEXT:182/903 vs the
   07-13 `procedure N` rule, the 07-14 push flip, and 474 commits under Soumil's name (only U1
   `b2316f5` is recorded as committed by him personally). None of the three "never commit"
   texts was rewritten; the rule survives only in memory.
3. **Her Eyes after every bless** (PROTOCOL:405-406, 435-436; "her review is stale when the
   image set changes") vs 21/29 blessed galleries with no review and three re-bless campaigns
   (08-14, 08-18, 08-28) with none; the surviving GLM-4.5 review is void by its own template
   rule. The 08-31 audit dates the lapse to 08-06.
4. **Bless authority** — "Soumil alone blesses" (hardening :1267, master :2636, playbook)
   vs "re-bless delegated by Soumil" (U2 R9) vs U8's "re-blessing performed through
   mechanically-passing Sable reports explicitly marked CLEAN" (master :2946) — a self-marked
   string (`SableReport.visual_review`), which the audit's Part 3 identifies as the gate defect.
5. **Append-only tracker vs closed status vocabulary** — §21 "never replace historical rows"
   vs §13 "statuses are only PENDING/ACTIVE/BLOCKED/DONE": the tracker contains `BLOCKING GREEN`,
   `SUPERSEDED (correction)`, `CLOSURE CANDIDATE`, `ACTIVE — … ACCEPTANCE RUNNING`, and both
   ACTIVE and DONE rows for U7 (×6), U8, U3, U5, U6. Also vs Soumil's 07-07 "evolve, never
   append" rule and `z-docs/08-reference/documentation-boundary.md`.
6. **Which plane is live** — `z-docs/README.md`: "Chapter 07 is the only live status/tracker
   area… implementation plans under `unfold-pkg/docs/` never override current status here" vs
   reality: 07 froze on 07-24 and every unit since tracked itself in `unfold-pkg/docs/U*_PLAN.md`
   + master §21. The 08-31 audit (chapter 09) says explicitly it "never overrides 07" — of a 07
   that is six weeks stale.
7. **Config vs code in the nature document** — PROTOCOL:3-5 "turns a HuggingFace config into…
   only config-derived facts (never invent dims/defaults)" vs master plan §1.1 "code is the only
   mechanism authority; config supplies values" (07-13) vs `01-product/product-contract.md`.
   The most-read rule file describes the pre-07-05 product.
8. **Scope** — PROTOCOL:438 "LLMs and diffusion only (no SSM/Mamba)" vs audio-gen (07-08),
   recurrent mixers (U6), `TO_SERVE.md` Part 5 audio, `finalize.ipynb` recurrentgemma, and
   `z-docs/01-product/supported-scope.md` (a fourth definition).
9. **Static-only physics** — `verification-ledger:17` "source inspection is static and does not
   instantiate the model — implemented" stated as a law; master plan §22.5 permits opt-in runtime
   probing; `09/first-principles-judgment.md` recommends meta-device instantiation. Three
   positions, no decision recorded.
10. **Import closure** — U3 boundary 8 (binding, 07-19) vs U3 DONE (07-27, 08-14) with
    `external_nodes` never passed vs U11-A1 `import_source.py` (08-28) building demand-driven
    closure for UNet readers only, while U11-A2 still plans to delete `_augment_diffusion_files`.
11. **Which catalogue is the support set** — `previews/old/toserve.md` ("canonical", 118 ids),
    `done/TO_SERVE.md` (299), `z-docs/stale/TO_SERVE2.md`, the run_77 roster (371), the 29-witness
    corpus, and `supported-scope.md`. `09/systemic-findings.md §3`: "coverage denominator against
    the declared support set is not measured anywhere."
12. **Evidence retention** — `PROJECT_CONTEXT.md` Part 9: blessed corpus "deliberately UNTRACKED
    — needs backup story" vs `.gitignore` treating galleries as "generated only" vs every bless
    approval resting on those PNGs. 22 of 30 cited `/private/tmp/model-unfolder-verification/<hash>`
    receipt directories are gone; 3 of the 8 that exist were recreated 2026-09-02 00:00
    (another pass of this program **(inference)**), so 5 original receipts survive.
13. **Untracked binding plans** — `U0_U1_*` and `U2_DEFINITIVE_COMPLETION_SPEC.md` are declared
    binding and deliberately untracked (`SCHEDULER_VERTICAL_TRACKER.md:45`); the master plan
    cites them as authority. A clone of the repo cannot see them.

---

## 5. Process metrics

| metric | value | source |
|---|---|---|
| commits / days / rate | 487 over 119 days; 4.1 per day; peak week 52 (W29) | `git log` |
| authors | Soumil 474, Soumil.Binhani 13 (= the 13 PR merges, none after 07-07); 78 bodies with a Claude `Co-Authored-By` trailer | |
| empty commit bodies | 396 / 487 (81 %); May 92/99, Jun 91/111, Jul 99/163, **Aug 111/111**, Sep 3/3 | `%b` |
| §13 commit template used | **1 / 487** (`e3e8b85` U0) | grep "Old unsound path" |
| longest bodies | U2 era: `c376115` 84 lines, `478e34a` 78, `28a66ad` 65 — the receipts moved from commit messages into plan docs after U2 | |
| docs share of added lines | 2–5 % in build phases; 18–20 % in P3/P4; 12 % in U3; ~5 % in U8–U11 — but plan docs grew to 20.4k lines while z-docs (9.6k) sit outside git | §1.1 |
| execution-plan volume | 20,443 lines in `unfold-pkg/docs/` + 9,595 in `z-docs/` + 451 PROTOCOL + ~150 KB memory | `wc` |
| tracker rows | §21: 29 rows for 16 units; 8 explicitly superseding; U7 6 rows, U0 4 | master plan :2650-2700 |
| receipt directories cited | 30 distinct hashes under `/private/tmp/model-unfolder-verification/`; 8 exist (3 recreated 09-02), 22 missing; `/tmp/sable_out`, `/private/tmp/u10-preservation-delta.json` missing; 3 `previews/` citations point at moved directories | §2.5, §3 |
| run_77 review artefacts | 1,011 per-model reports + 45 campaign docs, 238 MB, gitignored | |
| Her Eyes reviews | 21 (07-05 sweep, in `previews/old/`) + 337 (run_77) + 8 in blessed galleries (1 void, 1 stale) = 366; 0 since 08-06 | `find` |
| blessed witnesses | 25 (U0) → 26 (U2 R9, MusicGen) → 27/28 (U6) → 29 (U7, Qwen3.5+DBRX+Granite) | manifests |
| doc-vs-tree disagreements found in this pass | 23 spot checks: 17 stale/wrong, 3 broken paths; plus 13 recorded contradictions | §3, §4.3 |
| releases | 17 tags; last 07-07; `__version__` wrong since 06-20; Space frozen on 0.2.17 | §1.4 |
| receipt wall time | ~87 min per committed-tree receipt (audit); U10 receipt 3,607 tests | `U10_…PLAN.md:1662` |

---

## 6. What to keep, what to retire

**Keep (and put under version control).**
- `z-docs/00-start-here/project-doctrine.md`, `06-development/*`, `08-reference/documentation-boundary.md`
  — the laws are correct and short; commit `z-docs/` into the repo (or a sibling repo) so
  rulings get dates and diffs.
- `EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md` §1, §12, §14, §20 (the contracts and gates) and
  `U3_ACCOMPLISHMENTS…` §6–§8 (Gates A–J, §7 lost-detail, Soumil's checklist) — as the *procedure*
  plane, after the §7/U4-step-6 conflict is resolved in one sentence.
- `.claude/PROTOCOL.md` Gates A–C, Dable, Her Eyes — rewrite the opening paragraph and closing
  rules to the post-07-05 product (code authority, commit/push rule, scope) and make the Her Eyes
  auto-run a `bless()` precondition, not prose.
- The nine `feedback-*` memory files: fold their dated rulings into a committed
  `DECISIONS.md` (the table in §4.1 is a starting point).
- Generated artefacts only when generated: `COR5_NET1_MIGRATION_DEBT.md`, `U3_CURRENT_READER_INVENTORY.md`.
- The run_77 master index and problem map (the only recall-capable pixel↔truth trace ever run):
  compress the 45 campaign docs to the index + problem map and commit those two.

**Collapse.**
- §21 tracker → one row per unit, status ∈ {PENDING, ACTIVE, DONE} + a link to a machine
  `receipt.json` (already produced by the harness); delete the "supersedes" rows into git history.
- `07-current-state/` → one page regenerated from the tracker; delete `u2-*`, `r6/r7-*`, `u3-*`
  (their content is in `unfold-pkg/docs/` already).
- The per-unit `U*_EXECUTION_PLAN.md` files (7,700 lines for U8–U11 alone) → plan sections stay,
  narrative receipt logs move to `archive/` after the unit's DONE row cites the commit.
- `08-reference/code-map.md` → regenerate from the tree (a 20-line script) or delete; hand-maintained
  it was wrong within three weeks.

**Retire (archive, do not delete history).**
- `EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md` (already superseded), `SCHEDULER_VERTICAL_TRACKER.md`,
  `U3_D_TO_H_…`, the four `U3_*_RECON` files, `U3_READER_INVENTORY.md`, `sable_dable_playbook.md`,
  `blocks_supported.md`, `coverage_audit.md`, `serve_audit.md`, `models_supported.md` (empty),
  `architecture_atlas.html` (unreferenced), `TRUE_CONFIG_PLAN.md` until code exists.
- `z-docs/stale/` is correctly quarantined; fix the one broken link or drop the header.
- `previews/old/*.md` planning docs are the pre-U masters; they belong with `stale/`, not under a
  gitignored previews tree.
- `done/` and the two notebooks: keep offline; both notebooks contain a token literal and must
  never enter a repo.

**Fix now (trivial, long-stale).** `__version__` 0.2.15 → match `pyproject`; the three moved
`previews/` citations; the `legacy-removal.md` link; the 07-current-state README index; the U8
and U6/U7/U8-matrix headers; PROTOCOL's commit/push/scope lines.

---

## 7. Loose ends

- **Why does the audit chapter count 105 August commits (57 empty) when git shows 111 (111
  empty)?** Possibly a different date window or `%b` vs `%B`; not resolved here.
- **Was any bless PNG-inspected by a person after 08-06?** U8/U9/U10 plans say "inspected" /
  "individually inspected" (U7 08-06 names seven PNGs); U8–U10 name none. Not determinable from
  docs.
- **Which of the 474 `Soumil` commits were made by him?** Only `b2316f5` is recorded as his in
  the docs; the 13 `Soumil.Binhani` commits are the GitHub-UI PR merges (#1–#13, last 07-07), so
  the human-made set provably includes those 13 + `b2316f5`. Git cannot separate agent from human
  for the rest.
- **`done/TO_SERVE.md` as the run_77 roster source** is an inference from mtime and 156/299 id
  overlap; the runner (`run_77/_runner/`) was not read.
- **The `hf/` Space's `transition` branch** — content not inspected; may hold a newer app.
- **`docs/architecture_atlas.html`** (untracked, 67 KB, 07-19) — purpose unknown; nothing links
  to it.
- **`push.txt` "Vonxtral left"** — read as Voxtral (audio) TODO; unverified.
- **z-docs `08-reference/canonical-config.md`** (584 lines) was not spot-checked field by field.
- **The three receipt directories recreated on 2026-09-02 00:00** — presumably another pass of
  this program re-ran a committed-tree receipt; if so, the pass that did it should say so in
  `05-verification.md`.
- **The two notebooks contain HF token literals** at the workspace root, outside any `.gitignore`
  that protects them from a future `git add` if the workspace is ever initialised as a repo.
