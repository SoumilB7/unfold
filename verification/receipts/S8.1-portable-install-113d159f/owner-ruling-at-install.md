# S8.1 — owner's ruling on the returned candidate (`ededf752`, 2026-09-09)

Status as submitted: FAIL by the executor's own coordinator (all four latency
medians over budget; two artifact locks; SDXL `html_meta` preservation row).
Page weight and browser behaviour PASS (2.17 MB / 1.71 MB; ready 174 ms /
127 ms; click 30 ms / 27 ms) — L7 restored. Semantic parity 29/29.

## What I measured on the same tree

| measure | value |
|---|---|
| SDXL warm `unfold()` in-process, profiled | 10.7 s: cache lookup + `validate_dependencies` **4.9 s** (29,435 digests over ~22 k files: whole code roots incl. stdlib "seals" and package metadata), static UNet readers 3.7 s, rest ~2 s |
| SDXL child inventory (`inventory_in_subprocess`) alone, incl. supervision | **15.0 s** — the cold floor of the instance path |
| executor's fresh-process medians | cold 37.9 s / warm 16.5 s (SDXL); explicit imports 5.1 s excluded, *lazy imports and cache validation inside* |

So the warm miss is two things, both decisions rather than hard limits: a
cache validation that seals far more than §1c requires, and a cache hit that
still pays for lazy heavy imports. The cold number is child 15 s + readers
~4 s + validation/store ~5 s + rest; with the rulings below it lands in the
low 20s.

## Rulings (the decisions the independent review asked the owner for)

**1. Cache identity = the §1c contract, not the machine.** The key is: config
hash; the exact source-file hashes of the `SourceBundle` closure the static
readers already trust (the same closure whose fingerprint the ProgramIndex
carries — bet 4's demand-driven closure, ~635 files for SDXL); package name +
version for every library in the resolved class's MRO; resolved class,
constructor/factory, build flags; recipe id/dtype/state for observations;
cache schema version; **plus** the package's own `physics/` and
`evidence/` source hashes (a reader or observer change must invalidate).
**Not** in the key: whole-stdlib or whole-site-packages file seals, package
metadata censuses, native-library bytes. Residual risk, stated: a change in
a helper outside the SourceBundle closure and outside the MRO packages'
declared versions is not detected by the cache — exactly the same blind spot
the static readers already accept for their own proofs, so the cache is no
weaker than the evidence it stores. Invalidation evidence required: poisons
(one byte changed in a closure file → miss; MRO package version changed →
miss; a `physics/` or reader file changed → miss; nothing changed → hit and
byte-identical result; a stale/corrupt entry → miss). Validation must be
milliseconds, not seconds.

**2. A cache hit imports nothing heavy.** Cache lookup and result decoding
must not import torch/transformers/diffusers; those load only when a child
must actually run. Warm budget **9.0 s stands**.

**3. Cold budget for the instance path: 30 s stands**, stated as the UNet
targets' cold-empty-cache number (child floor 15 s measured). Any target
above it after rulings 1–2 comes back with a profile, not a plea.

**4. Legacy seeding is dropped from S8.1.** The old S6/S7 receipts lack the
capture envelope; a seed is not a fresh observation; the first cold run
seeds the cache. No time is spent laundering old artifacts.

**5. The two artifact failures and the preservation row are consequences,
not defects.** (a) SDXL `html_meta` changed because the page is compact —
presentation only, semantic parity proven; (b) the SDXL example page is
stale for the same reason; (c) the S7 freshness stamp refuses 23 changed +
5 added production paths — reviewed S8.1 code, matrix rows unchanged. All
three are one enumerated output/metadata delta for **Soumil's approval**;
after it: regenerate the SDXL example, re-stamp S7 (no regeneration of
rows), re-bless the one preservation row through the guarded writer with an
independent verdict.

## Acceptance for S8.1 (unchanged otherwise)
Four medians under budget (cold ≤ 30 s, warm ≤ 9 s, three fresh-process
samples each); page ≤ 5 MB, ready ≤ 1 s, click ≤ 150 ms; 52/52 preservation
after the approved re-bless; receipts; fingerprints identical; Linux lane
green on push (still unverified — required before DONE). Then S9.


---

# Verdict on the resubmission `c7ec5526` (production `87cde8d8`, test-only `3d7a6332`) — 2026-09-09 evening

**Soumil's ruling, verbatim (2026-09-09, during this review):**

> honestly time is not such a big deal for me ok, i just need the accuracy for now and then we can go ahead with remaining stuff you see
> accuracy >>>>> design >> time
> basically

## What I verified myself (not from the sheet)

| check | result |
|---|---|
| Ruling 1 (cache identity = §1c + SourceBundle closure + own physics/evidence hashes; validation in ms) | PASS — `physics/result_cache.py` capability 3 stores only prepared documents, source addresses and worker DTOs; readers and IR construction always re-run (module docstring + `run_cached_parse`); campaign validation 29.8–42.4 ms; machine seals deleted (`cache_dependencies.py`, `cache_probes.py`, six seal test modules; −2,738 lines) |
| Ruling 1 poisons | PASS — I ran `tests/test_s81_result_cache.py` + `test_s81_source_bundle_codec.py` + `test_s6_physics.py` myself: 144 passed in 61 s; includes raw-member byte change → miss, declared-scope poison → miss, corrupt entry → normal miss, stale index → decline, request-axis mismatch → no reuse, own-source subtree change → new identity, CRLF/LF agreement, authenticated address never persisted |
| Ruling 2 (a hit imports nothing heavy) | PASS — my own fresh-process pair on an empty cache dir: warm `sys.modules` contains no torch/transformers/diffusers after `unfold()`; the diffusers class-file lookup now uses `importlib.util.find_spec` (non-executing) |
| Exactness of the hit | PASS — my cold and warm IRs are byte-identical (14,186,131 bytes); the campaign's twelve samples say the same; cache entry 3.0 MB |
| Ruling 3 budgets (30 / 9) | **FAIL as numbers:** SDXL 32.681 / 9.888 s, SD-v1-4 31.403 / 8.816 s (campaign, host load 5–10 on 10 cores); my sample 34.7 / 9.6 s at load 7–9. Profile supplied as required: warm remainder is the static UNet readers (`investigate_unet_runtime` 10.8 s cumulative of 17.6 s profiled), not the cache |
| Ruling 4 legacy seeding dropped | PASS |
| Ruling 5 output delta staged, originals untouched | PASS — four scratch files only (`examples/manifest.json`, compact `examples/stable-diffusion-xl-base-1-0.html`, one `html_meta` leaf, `verification/s7/matrix.json` sources only: 27 changed / 4 added / 0 removed; 117 payload files unchanged); the compact example strictly expands to the original 24,310,552-byte page |
| Page weight / browser | PASS — 2,170,839 / 1,708,364 bytes; Chrome 152 ready 252 / 183 ms; click 27 / 23 ms |
| Preservation | 51/52 + the one SDXL `html_meta` row awaiting approval; 29/29 semantic parity |
| Timer boundary commit `87cde8d8` | stricter, not looser: heavy/lazy imports now inside the interval; thresholds unchanged (I read the diff) |
| S6 boundary correction `3d7a6332` | lawful: a closed set of exactly five parent hooks admitted, with five positive and eight negative controls; production byte-identical |

## Ruling

**S8.1 is ACCEPTED on accuracy, conditional on three closing acts** (below). The two defects that made S8.1 necessary for *correctness* are gone: the cache is exact (an inexact cache would have been an accuracy defect) and cheap to validate, and a hit is heavy-import-free. The latency misses that remain (2.7 s / 0.9 s / 1.4 s over) are reader work, not evidence work, and Soumil has ruled time below accuracy and design.

**On the budgets — not a plea, the plan's own law.** `12` S2 says budgets "are set from that baseline, not asserted in advance." The 30 / 9 numbers were asserted in advance by me (9 s borrowed from the text-model end-to-end budget). The lawful correction is the S2 method: record the twelve measured samples as the UNet baseline in `latency_budgets.json` and set the gate at the S2 multiplier (1.3× the slowest median, rounded up: cold 43 s, warm 13 s) with `measurement_status` = measured, host load recorded. No gate is weakened by a proxy; the gate now asserts the measured predicate the plan prescribes. The 30 / 9 figures survive as the **target** of a named later item, not as a blocking gate: **"static UNet reader cost (warm ≈ 5 s unprofiled)"**, owner = executor, step = S10 (the first deletion unit is `adapters/diffusor/unet.py` + renderer UNet authorship; the reader cost is re-measured there and the gate ratchets down when it falls). Any future *rise* above the measured medians is a regression under the ratchet rule (proven count / budgets never worsen without a named cause).

**Two notes carried forward (not blocking, accuracy-neutral):**
- N1 — layering: `evidence/context.py`, `evidence/document.py`, `evidence/sources.py` and `parser.py` now import `physics.result_cache` (two at module level). The evidence layer depending on the physics package is a temporary coupling in the doctrine's sense. S11 module map: the cache hooks become a single injection point owned by the parser (or a `model_unfolder.cache` boundary), and `physics` moves under `model_unfolder.physics` — a top-level `physics` package in the wheel is a namespace hazard. (Related local fact, no action: the editable install on this machine predates `physics*` in `pyproject`, so `import model_unfolder` from a neutral directory fails locally until `pip install -e .` is rerun; the built wheel ships `physics`, receipt `S8-wheel-final-7168440`.)
- N2 — the S6 allowlist widening exists only because of N1; it is retired with N1.

## Closing acts (in order; then S8.1 DONE and S9 starts)

1. **Soumil's approval** of the enumerated output delta (one line; see the decision page `12-design/S8.1/owner/index.html`).
2. Executor: install the four staged files through the guarded procedure (independent verdict for the one `html_meta` row), re-run the two artifact checks (S5 example, S7 stamp) and the 52-row preservation lane on the committed tree; record the UNet baseline in `latency_budgets.json` by the S2 method (twelve samples, medians, host load, budgets 43 / 13, target 30 / 9 named as the S10 item); receipt in-repo; sheet updated to DONE-pending-Linux.
3. Push; Linux lane green (the isolation lane has never run natively for S8/S8.1). Then DONE.


---

# Addendum 2026-09-10 — proof-seal portability: two-file S7 renewal APPROVED (owner)

**What Linux found (run 34453756753, `evidence-receipts` job, read by me):** the live Linux S7 artifact for SDXL disagrees with the committed one at `fact_claim_proofs[*].index_fingerprints[0]` — two distinct committed values (`09fc7b6a…` ×8, `14f56c8a…` ×192) versus two live ones. Cause, verified in the diagnostic `proofs.json`: the persisted seal was `ProgramIndex.fingerprint`, whose `SourceId.canonical_path` values are absolute host paths (`/Library/Frameworks/Python.framework/…/site-packages/diffusers/models/unets/unet_2d_condition.py`). Same bytes, different machine → different seal. A verification artifact was carrying a machine address as if it were evidence identity.

**The fix (candidate `source.diff`, read by me):** the six proof-summary producers (`claim_evidence.py`, `instance_population_claim.py`, `unet_claims.py` ×7 sites, `unet_cell_connections.py`, `unet_primary_ports.py`) persist `portable_source_index_fingerprint(index)` instead of `index.fingerprint`; `program_index.py` memoises it per frozen index. The portable seal keeps the complete source multiset, component ownership, external provenance, the shortest unique trailing locator and the **content fingerprint** of every file; it drops only the host prefix. So relocating the same closure is stable; renaming, re-owning or changing bytes still changes the seal. The process-local `fingerprint` stays the live address identity. Producer-first (the false producer is the seal author, not the comparator); no normaliser touched (`normalizer_unchanged: true`); no proof waiver.

**What changes:** exactly 200 leaves in `verification/s7/models/stable-diffusion-xl-base-1-0.json.gz`, all `index_fingerprints` (I counted the delta file: 200 rows, every path ends in `index_fingerprints`), mapping 2→2 values; `verification/s7/matrix.json` raw/logical hash for that one payload plus six source stamps; all 39 summary rows unchanged; the other 116 payloads byte-identical (executor + independent reviewer fresh regeneration; consistent with the proposal touching two files). Preservation 52/52, S5 example, static, 79 controls pass.

**Ruling:** approved. This widens my earlier "117 payloads unchanged" condition by one payload, for a cause that is a portability bug in a verification receipt, not a change to any fact, drawing, number or product output; no re-bless of a product baseline is involved, so Soumil's e19 authority is not engaged (told to him, with "why not"). Executor: install the six-file source correction and the two S7 files through the guarded procedure, commit with the receipt, push, rerun the Linux lane at the exact head. S8.1 is DONE only on Linux green; S9 stays locked until then. Carry-forward: any *other* persisted receipt that embeds `ProgramIndex.fingerprint` or an absolute path is the same class of defect — the executor greps for it before the push and reports the count in the sheet (expected 0 after this fix).
