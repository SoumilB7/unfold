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
- [ ] **CURRENT:** Commit-1 full gate RE-RUNNING on the stabilized tree (structural-empty manifest is the check that their mistral refactor didn't move any corpus pixels)
- [ ] Commit-1 gate: grouped audit files + all-25 regression + manifest rebuild + preservation on one unchanged fingerprint
- [ ] commit + push

## What's running / waited on right now

**Commit-1 gate is running now** (~1.5h): diffusion sanity (the producer fix touched the shared diffusor text-encoder path) → manifest rebuild + all-25 + audit-files + grouped + full suite on one frozen fingerprint. Green → commit + push → Commit 2.

## How to read the gate runs

Each commit ends with the same 4-layer gate on ONE unchanged tree fingerprint:
1. each audit file alone, 2. grouped hardening gate, 3. full suite (~1h, the
long pole), 4. clean-checkout. "Waiting" almost always means layer 3 is
running. all-25 = the 25 blessed diffusion/LLM witnesses re-checked for drift.
