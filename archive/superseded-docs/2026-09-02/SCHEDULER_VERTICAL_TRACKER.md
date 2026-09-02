# Scheduler vertical — SUPERSEDED

**This plan was mis-scoped and is retired.** Its 5-commit route (source
qualification → method index → scheduler evidence → render cutover → claims)
is **U3 + U13 work**, not U2. U2 may not derive scheduler semantics; it
registers the claims, receipts only the genuinely evidence-backed ones, and
files the class-name-selected Euler/flow semantics as exact U13 debt.

Live plan and state: `z-docs/07-current-state/u2-receipt-accountability.md`.
The history below is kept only for the Commit-1 record.

---

# Scheduler vertical — live tracker

**What this is:** the first full H7 domain burn-down (root.scheduler), executed
as 5 gated commits via the U2-receipt route Soumil specified. This file is
updated at every state change so progress is visible without reading test logs.

Last updated: 2026-07-15, mid Commit 1.

## Commits (in strict order)

| # | Commit | State | Notes |
|---|---|---|---|
| 1 | U2 projection-receipt core (pilot: vision projector width) | **IN PROGRESS** | code built + own tests green; fixing one exposed phantom before the gate |
| 2 | scheduler source qualification + general method-level program index | pending | |
| 3 | evidence/scheduler.py (Schedule + Step evidence, two mechanisms) | pending | |
| 4 | scheduler semantic + render cutover; delete flow-marker/Euler fabrication | pending | |
| 5 | 4 scheduler mechanisms, claims+receipts, remove _scheduler_config opacity | pending | |

## Commit 1 — sub-steps

- [x] `evidence/receipts.py` — typed `ProjectionReceipt` (key/value+status hash/owner/mechanism/surface/node-path/kind), scoped `RECEIPTED_SCOPES`, occurrence→target→receipt join, reverse-fabrication check
- [x] `RenderEvent.receipts` + `note_receipts` emission channel; `facts_projected` kept as compat
- [x] pilot wiring: vision/video projector op-drill emits a source-bound width receipt (2 receipts on qwen2-vl, from the real drill)
- [x] global `projection_receipts_available` bool → owner/mechanism-scoped `projection_coverage`
- [x] Net-2 rewritten to occurrence→target→receipt (blocking inside receipted scopes, advisory elsewhere)
- [x] reverse-fabrication net (receipt must reference a registered fact / claim target / shrinking debt)
- [x] `test_receipts.py` (pilot, poison, coverage, no-visual-delta) + 3 evolved net-2 tests — **all green (115p)**
- [x] no pixel change verified (qwen2-vl view hashes identical)
- [x] **PRODUCER FIX (per binding doctrine):** the flux 'phantom' was a real owner mis-attribution — its mistral3 text encoder (a genuine VLM) had its vision projector falsely attributed to top-level `root.vision` by a context-less second sub-parse. Fixed at the earliest producer: `ParseContext.component_namespace` threads ownership; modality owners are `<namespace>.<modality>`; the false `_detect_text_encoders` context-less path DELETED. Net-2 reverted to UNCONDITIONAL blocking (rejected the render-drawn workaround). flux → `root.text_encoder.vision` (clean); qwen2-vl → `root.vision` (strict); 10 receipt tests + pos/neg/adversarial controls green
- [x] sanity surfaced 4 fifth-directive tests that encoded the OLD buggy flux-under-root.vision behavior — updated to corrected ownership: flux/qwen-image are now stronger negative controls (VLM text encoder = root.text_encoder.vision, no top-level root.vision); added a legitimate synthetic top-level non-grid VLM witness for the hidden_size encoder binding; same-path-two-mechanisms control uses qwen2-vl(projector)+synthetic(encoder). 16 audit tests green
- [x] first full gate INVALIDATED: a concurrent actor modified the tree mid-run (new input_formats.py + mistral_params→aliases refactor, not mine) → fingerprint changed + their input_formats.py:37 broad-except failed the ratchet. Per Soumil: waited for their work; on 'done' applied the one-line typed-except fix to their finished file (doctrine: typed>broad; disclosed; not committed by me). My 19 files verified intact.
- [x] COMMITTED `7c6a298` — the whole green tree as ONE reproducible unit (incl. the previously-untracked `receipts.py`, `input_formats.py` and their tests, since tracked code imports them; Soumil's `U0_U1_*` plan docs stay untracked)
- [x] gate result: **full suite 1219 passed / 5 xfailed / 0 failed**, manifest all-25 = 0, STRUCTURAL deltas [] — but **FINGERPRINT CHANGED**: the concurrent actor modified `evidence/sources.py` + `tests/test_code_evidence.py` at 10:51, 28 min after FP_BEFORE. Layer 3 ran entirely AFTER that change (so the committed content IS full-suite-green), but the run is INVALID as a single-fingerprint receipt and is not cited.
- [x] U2.1 docs: `EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md` now states the shipped contract (`config_consumed_unreceipted`, occurrence→target→receipt, scoped `projection_coverage.receipted_scopes`, unconditional blocking in a receipted scope); the census doc was fixed at its GENERATOR (it is a generated artifact)
- [x] U2.1 census regenerated: **281 occurrence-exact rows** (was 267). The +14 is PRECISION, not new debt: the ownership fix un-merged falsely-shared rows — `root.vision` 25→15, new true owner `root.text_encoder.vision` = 24. Every other owner unchanged. The old 267 was an UNDERCOUNT caused by the producer bug.
- [ ] **CURRENT:** commit the U2.1 correction → ONE valid gate on the stable tree → clean-checkout → then mark ONLY 'U2 receipt core + projector-width pilot: DONE'

## Scope correction (Soumil, 2026-07-16)

U2's job is to make every EXISTING structural claim accountable — NOT to solve
future architecture semantics. A mechanism needing later U6–U13 interpretation
gets exact shrinking DEBT, never a fabricated receipt. Concretely the scheduler
splits: **U2** registers its claims, receipts only the genuinely evidence-backed
ones, and files the class-name-selected Euler/flow semantics as exact **U13**
debt; **U3** indexes the real `step()`/helper graph; **U13** derives the
semantics and deletes the fallbacks. The earlier 5-commit "scheduler vertical"
plan was mis-scoped (it was U3+U13 work) and is superseded by U2.1–U2.9.

## What's running / waited on right now

Committing the U2.1 correction, then ONE valid gate on a stable tree.

## How to read the gate runs

Each commit ends with the same 4-layer gate on ONE unchanged tree fingerprint:
1. each audit file alone, 2. grouped hardening gate, 3. full suite (~1h, the
long pole), 4. clean-checkout. "Waiting" almost always means layer 3 is
running. all-25 = the 25 blessed diffusion/LLM witnesses re-checked for drift.
