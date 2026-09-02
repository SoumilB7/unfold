# U2 Definitive Completion Spec — execution progress (live)

## ✅ U2 CLOSED (Soumil-confirmed, 2026-07-19). 1d0c72b pushed & synced.
1493p/11s/3xf/0 FAILED · fingerprint identical · 26-witness manifest
blessed & green · both audit rounds included. No further U2 action.
Future work = the explicitly tracked later-unit debts (the 112-row
StructuralDebt register by unit + position.py funnel walks + the
U10/U11 flags). Historical detail below.


## ⚠ SOUMIL'S FINAL-CORRECTION VET (2026-07-19) — U2 NOT COMPLETE. BINDING.

He confirmed BOTH failures live on 7110be6:
(1) `is_encoder_decoder` sits in ignored_fields.yaml (~line 110) → my R7
GLOBAL bare-key ledger rail records its ARCHITECTURAL reads (seq2seq half,
mask causality, cross-attn schedule) as "ignored … display/plumbing" —
factually wrong; R8's standing net never sees them.
(2) `fabrication_debt_keys()` (structural_debt.py) feeds reverse-
fabrication: a WHOLLY FABRICATED receipt root.is_encoder_decoder /
mechanism=fake / surface=card / target=fake / projector=fake returns []
findings — pending INPUT classification authorizes a fake DRAWN output.

REQUIRED CORRECTIONS (his exact list):
1. Replace global ignore-by-leaf with EXACT SCOPED rules (owner pattern +
   exact path + reader/intent + reason). Remove is_encoder_decoder from
   the global list; CONSUME it into its exact mask/decoderness/cross-attn
   targets (TP reads ~:801/:1198/:1311-era sites; the R7-added consume for
   is_decoder mirrors the owner/key decoder.attention.mask). Then DELETE
   the 4 is_encoder_decoder pending rows (debt shrinks same commit).
   NOTE: the redesign must stop the transformer ignored_fields keys/
   suffixes from feeding the ledger GLOBALLY — enumerate exact scoped
   rules (token-ids etc. per owner root/root.text_encoder/
   root.text_encoder.vision/root.vision + exact top-level paths); rerun
   census after; any resurfaced standing rows need real dispositions.
2. PROHIBIT config_read/classification debt from authorizing receipts:
   delete the debt-key lane (fabrication_findings' debt_keys param + the
   sable join + fabrication_debt_keys itself). A receipt cites a typed
   fact OR an exact migration claim tied to registered fact+route. Never
   weaken — this deletion strengthens.
3. PERMANENT poisons reproducing his two commands verbatim: (a) an
   is_encoder_decoder read can never ledger as ignored (and scoped rules
   cannot match without exact owner+path); (b) the fabricated receipt
   above MUST return a finding.
4. Complete R6 joins: drawn_leaf debt joins carry owner(+surface)
   identity, not bare names; fact_registered/fact_routed/status_retired/
   unknown_policy_retired conditions must ALSO require the row's owner ∈
   the FactDefinition.owner_patterns (owner-bound conditions);
   unrowed-extras growth gate becomes WRITER-EXACT (iterate census keys,
   demand exact writer_key rows), not extras-target-based.
5. MusicGen as witness 26 + inspect the EVIDENCE-ONLY manifest delta +
   perform the explicit re-bless (Soumil EXPLICITLY delegated this bless
   in the vet message — document the delta inspection in the commit).
   Need a musicgen-small fixture (check tests/ + test_support for an
   existing MusicGen config; R-fix directive earlier said "add MusicGen
   to the preservation witness set").
6. The R9 receipts + completion summary must live IN the repo (the z-docs
   summary is outside the pushed branch) — the correction commit IS the
   R9 commit ("U2-R8/R9 … final receipts" per spec §8).
7. Rerun the FULL R9 bracket on the resulting committed tree: ZERO
   failures, fingerprint identical, clean checkout. A known-explainable
   failure is NOT a passing bracket.


Executing `unfold-pkg/docs/U2_DEFINITIVE_COMPLETION_SPEC.md` (binding), unit by
unit, in strict order. Uncommitted tree above `c376115`. **Bless withheld. Not
yet committed.**

## The binding reframe (§3.3)

Do NOT flip `merge=True` in U2. Class-supplied candidates live in
`class_overlay`, audited, authoring NO architecture. Parity + Falcon/Gemma
domain fixes are U6/U8. Accordingly the parser production arbitration I had added
was **reverted**; production parsing is behaviour-preserving; the generic
`arbitration.py` and its tests remain.

## Vet of the first R0–R4 pass — corrections applied

Soumil's vet reopened R1–R4. Status of each correction:

- **R1 — embedded `merge=True` violation. FIXED.** The root prep was
  `merge=False` but the two `encoder_panel` prep sites used the default
  `merge=True`, so the class overlay WAS authoring an embedded Gemma-2 schedule
  (a live §3.3 breach + embedded≠standalone). Both now `merge=False`. Embedded
  Gemma-2 is uniform (shadow); embedded == standalone. **Zero structural drift on
  the 25** (`ir/expanded/params/html_meta/gallery` byte-identical) — no witness
  had a class-schedule-dependent encoder.

- **R2 — document path + reader in the consumed occurrence identity. FIXED.**
  `ConfigOccurrenceKey` gained `document_path` and `reader` (defaults keep every
  existing site safe); `consume_decision` populates them. The same
  `config_path` in two documents is now two occurrences; one source read by two
  readers is two occurrences. Back-compat verified (llama parses).

- **R3 — arbitration facts cite exact evidence + persist conflicts. FIXED.**
  `verdict_to_fact` now cites the winner's EXACT config path (not the prose
  premise) and REAL premise fact IDs (`Candidate.premise_fact_ids` →
  `Verdict.premise_fact_ids`); a derived fact with no premise is refused (I-5);
  conflicts are PERSISTED to a passed `conflict_sink` rather than dropped.

- **R4 — replace the gate + complete surfaces + poisons. PARTIAL.**
  - DONE: the writer-identity multiset is now THE gate — `STRUCTURAL_WRITERS`
    (pinned 286-key baseline) + `new_structural_writers` / `stale_structural_
    writers` on writer identity, REPLACING the `(sink,target)` set that could not
    see a second author. Poison verified: a second writer of an ALREADY-PINNED
    target is flagged (the old set could not). 20 previously-hidden multi-writer
    targets surfaced (286 vs 202).
  - NOT DONE: the COMPLETE static/runtime surface expansion §R4 lists — spec
    FIELD MUTATIONS (only definitions+constructors are scanned today), Region/Op
    DEFAULTS, block/card view kinds + structural PHRASES, expanded-JSON
    structural keys, parameter formula BRANCHES, conformance expectations;
    helper-returned-constant + table-driven resolution; recursive runtime visit;
    and the remaining ~11 poisons. **R4 is not complete and must not be committed
    as done.**

## Units R0–R5 (+R5 correction) — complete, pushed, vetted

R0–R4 (…2b1689a + regression fix a443661), R5 (478e34a), R5 forward
correction per Soumil's vet (a778fb7). Every unit: focused gate + broad
full-suite gate + isolated committed-tree receipt + pixels byte-identical.

## R6 — landed in working tree (commit pending gates)

The ONE StructuralDebt register (`evidence/structural_debt.py`): 67 exact
rows (one per writer×target), U3–U14 constructor-enforced, closed
deletion-condition DSL (8 verbs incl. `writer_gone`/`unknown_policy_retired`),
blocking gates (duplicate / dead-writer / dead-consumer / satisfied-condition
/ growth incl. nested — family excuses dead). Byte-parity proven on all 4
derived join surfaces vs the old registers before their deletion
(pending_projection_paths / pending_classification_paths /
fabrication_debt_keys / drawn_unledgered_names). Deleted: LEGACY_EXTRAS,
DRAWN_UNLEDGERED_DEBT, PENDING_PROJECTION_DEBT, PENDING_CONFIG_CLASSIFICATION.
Rewired: parser.py (excusals, classified_paths, pending_sources ×2), sable.py
(fabrication debt keys). Scanner extension (a0+): setdefault / dict-literal /
AnnAssign / update / variable-keyed subscript / extras-named-enclosing-symbol
(`_merge_extras`, `multimodal_extras`) — 10 new writer keys pinned; the 9
formerly writer-less rows all have censused writers. INFRA keys unified (ONE
source: registry, 7 keys). §5.8 xfail p11 flipped to positive (rows carry
owner). Row audit doc: `r6-debt-rows.md` (same dir).

## FINDING (R6 broad gate): pre-existing red on the zero-drift manifest net

`test_preservation.py::test_expected_manifest_zero_drift_zero_skip` fails —
ledgers drift on 25/25 witnesses, sable on 18 (pixels/ir/expanded/params/
html_meta/gallery all hold).  Bisected in an isolated worktree: the drift
enters at **28a66ad (U2.2a path truth)** — config paths in ledger bytes became
TRUE, which IS the intended evidence-surface change — and the expected
manifest (last rebuilt c061c92/U2.1a) was never re-blessed; re-bless is
DEFERRED TO R9 by Soumil's directive, so this net stays red until his bless.
Environment exonerated (c061c92 reproduces its own manifest byte-exactly
today).  **Receipt defect admission: my R4/R5 "broad gate green" receipts
cannot have contained a passing run of this test** (it fails deterministically
at 2b1689a, 478e34a, a778fb7 in pristine worktrees) — those receipts were
flawed on this one net; every other net was and is green.
**R6 adds ZERO drift**: all-25 × 6 regenerable surfaces hash-identical between
a778fb7 and the R6 tree; galleries untouched (ignored local bless artifacts).

## R6 — LANDED AND PUSHED (97a7d68)

Receipts: main-tree 1467p + exactly the known pre-existing red; isolated
worktree 1466p + same red + 1 gallery isolation artifact (galleries are
gitignored local bless artifacts; fails identically at a778fb7).

## R7 — code side DONE in working tree (dispositions in flight)

- ONE preparation boundary: diffusor slot walk prepares ONCE
  (prepare_document → DocumentBinding("root.<slot>", ("_text_encoder_configs",
  <slot>), prepared) → bound_document), passes the binding into
  normalize_encoder_config (no re-prepare, no double-entered address); the
  encoder round-trip holds its scope open over its WHOLE body (post-parse
  evidence reads included).
- Reader fixes: text-config reads named via config_container(_text_path,
  obj=text_cfg) at use_sliding_window/max_window_layers/layer_types/
  full_attention_interval, both rope_parameters|rope_scaling sites,
  rope_theta, and both _norm_kind_evidence(_src) callers.
- **Census after fixes: unlocated 15→0, origin-unknown 80→0** (two §R7
  blocking criteria); standing rows 227→294 (false merges separated —
  precision, per spec); root.vae/_scheduler_config rows now properly
  addressed.
- Loose document_scope overload DELETED (bound_document is the only entry;
  _enter_document private); all test callers converted via the new
  test_support.bind_document helper; deletion pinned by
  test_the_loose_document_scope_overload_is_deleted. Parity net 19p green.

## R7 — BLOCKING STATE ACHIEVED (suite receipt running)

**Standing accessed-but-unconsumed occurrences: 0** (was 227 at unit start;
294 after the precision gain). unlocated 0 · origin-unknown 0 · conflicts 0 ·
no-disposition rows 0 · ownerless pending 0. 116 CONSUMED conversions landed
via 3 parallel agents (transformer/diffusor+unet/modalities) — every
converted read records `consumed` with exact document-relative paths;
rival-spelling or-chains became one alias-law resolve (unequal rivals now
typed ambiguity, never silent first-match). 78 scoped-ignores ride the
everchanging vocabulary rails (address keys incl. wrapper probes
text_config/vision_config; declared chips cite their config_facts row;
sample-canvas dims via _display_geom; redundant co-declarations reasoned at
the reader). 36 PENDING occurrences excused by exact config_read
StructuralDebt rows (register 114 rows, key includes owner+occurrence; census
renders them in their own dispositioned section with units). audit_incomplete
now empty on the live corpus (mechanism re-pinned by a starved-scheduler
witness). Pixel surfaces verified byte-identical to the blessed manifest on 5
witnesses (ir/expanded/params/html_meta); ledgers/sable drift stays under the
known pre-existing red (28a66ad).

## R7 dispositions — integration state (historical)

Disposition audit complete (227/227: C 116 / SI 78 / P 33 / D 0; class-overlay
17/17). Landed so far: 33 PENDING rows in StructuralDebt (register now 100
rows, key extended to include owner+occurrence so per-owner reads are distinct
debts); ledger scoped-ignore rail moved to everchanging YAML
(evidence/ledger_ignores.yaml address keys — _ADDRESS_KEYS hardcode deleted —
plus transformer ignored_fields keys/suffixes ride the same rail; suffix
_token_index + keys tokens_per_second/position_id_per_seconds added);
diffusor chip reader records declared display reads as resolve().ignore
citing its config_facts.yaml row; _display_geom for sample-canvas dims;
out_channels consumed. Census: 294 → 182 standing. Class-overlay burden
collapsed to the 5 loader rows (shadow-mode slot documents dissolved the 12
class_default rows; te readers see checkpoint words — the raw rope_scaling
spellings now stand as their own occurrences and get pending rows after the
conversion pass). CONSUMED conversions running as 3 parallel agents
(transformer parser / diffusor+unet / modalities).

NAMED FOLLOW-UP (not R7-blocking, R8 net-1 territory): evidence/position.py
(~:115, :444-466) walks nested text_config RAW — bypasses the access funnel
entirely (invisible to the census); needs its own producer fix.

## R7 — LANDED AND PUSHED (9498eb4)

Receipts: main 1470p + only the known zero-drift red; isolated 1469p + same
red + the gallery isolation artifact. Pixels byte-identical on 5 manifest
witnesses. All seven §R7 blocking criteria hold (standing 0 / unlocated 0 /
origin-unknown 0 / conflicts 0 / no-disposition 0 / class-loader 0 /
ownerless 0).

## R8 RUNBOOK (spec 708-735: all 10 U2 nets blocking + anti-vacuous poisons)

Current sable advisory nets (sable.py:455-545): asserted_facts (b=False),
projection_audit (gated by _PROJECTION_AUDIT_BLOCKING), config_accessed_
unprojected (b=False, "inert until consumed rail populated" — the rail IS
populated post-R7), config_audit_incomplete (b=False, its own note says
"then blocking" once every adapter consumes — that arrived in R7).
Spec net → implementation mapping:
1. boundary completeness → NEW net: model's ledger has zero unlocated /
   origin-unknown reads (R7 achieved 0 corpus-wide; lock per-model).
2. location/origin conflict → extend config_ambiguity family (value
   conflicts covered; add location/origin conflict rows if the ledger
   distinguishes them).
3. accessed-but-neither(consumed/ignored/pending) → flip
   config_accessed_unprojected to blocking on the standing-occurrence set
   (unconsumed minus pending_projection/classification == empty).
4. consumed-but-no-exact-target → net over consumed events lacking fact
   owner/key (typed consume_decision makes this structural).
5. migrated-target-without-receipt → config_consumed_unreceipted (already
   blocking inside receipted scopes — verify + poison; do NOT widen beyond
   receipted scopes: "the exact debt row keeps the net honest").
6. receipt-without-fact/route/consumption → receipt_fabrication (blocking
   already; poison per R5 validator).
7. structural-writer-absent-from-registry/debt → NEW sable net over
   new_structural_writers() + debt_problems() (R4/R6 test gates become
   render-time nets).
8. stale-debt-or-growth → same debt_problems net (satisfied/unbacked/
   unconsumed/duplicate/unrowed).
9. cross-context/owner receipt collision → R5 join validator (context_token
   mandatory) — verify blocking + poison.
10. identity/config-name taint → identity_guard (verify blocking + poison).
Also flip: config_audit_incomplete → blocking (empty corpus-wide);
asserted_facts stays advisory unless spec's net-3 reading demands it (it
does not — the B5 hunt-list is not one of the 10).
Named follow-up to fold into net 1/2: evidence/position.py raw text_config
walks (~:115, :444-466) bypass the funnel — fix the producer or the
boundary net cannot see those reads.
Poisons: every net needs an anti-vacuous poison ("empty registry/claim set
cannot make any gate vacuously green" — spec 734-735).
Gates: both (focused + full suite + isolated receipt); expect exactly the
1 known zero-drift red; pixels byte-identical.

## R8 — implemented in working tree (broad gate running)

Landed: config_audit_incomplete flipped BLOCKING (staged period ended with
R7; pin test updated). Three NEW blocking nets in sable.py:
document_boundary_completeness (net 1 — reads the REAL extras keys
accessed_unresolved_path / unestablished_provenance, key names verified
against the producer), config_standing_unconsumed (net 3 — exact
owner+path join against the pending register), structural_debt_register
(nets 7+8 — new/stale writers + debt_problems at render time).
Nets 5/6/9/10 verified already-blocking (net-2 receipted-scope doctrine,
receipt_fabrication, R5 context-token validator, identity_guard);
projection_audit already blocking (_PROJECTION_AUDIT_BLOCKING=True).
asserted_facts stays advisory (B5 hunt-list, not one of the ten).
Anti-vacuous poisons: tests/test_u2_r8_blocking_nets.py (net-1 key-rename
guard, net-3 poison + exact-owner excusal + EMPTY-REGISTER-gets-stricter
anti-vacuity, net-7/8 live-state poison, all-blocking assertion on a live
witness). 34 tests green.
STILL OPEN in R8 (or documented follow-up): evidence/position.py
_config_value/_config_scopes raw walks (funnel-invisible reads of e.g.
layer_types as code-evidence tie-break) — needs wrapper-path naming per
scope hop to emit located events without turning net 1 red; NOT closed in
R8, carried as the named follow-up.

## Remaining

- R8: broad gate (running) → isolated receipt → commit + push.
- R9: full acceptance bracket on ONE unchanged fingerprint (no code edits
  while it runs); add musicgen-small.json preservation witness; summary
  doc for Soumil; MANIFEST RE-BLESS + gallery bless (Soumil's alone).

## Owed to Soumil (unchanged)

- The manifest re-bless (evidence-surface drift only; no structural delta).
- Any intentional structural/pixel delta (there are none — U2 is
  behaviour-preserving by construction).

---

## NEXT-STRETCH RUNBOOK (do this first, fresh context)

Soumil's directive: stop without committing; next stretch reconstruct + review
the boundaries, then land **R0/R1, R2, R3** as three commits (each passes its own
targeted tests and reproduces from its committed tree). **No `git add -A`, no
forced hunk split, no behavior change to obtain clean commits.** If R1 and R2
cannot separate without a broken intermediate tree, combine them into ONE
documented **R1+R2** commit explaining the dependency. After those are verified,
continue R4; **do not start R5** until R4 has complete static/runtime surface
coverage + the full 12-poison matrix + its own green receipt.

### CRITICAL reconstruction caveat — the tree has substrate BENEATH R0–R4

The uncommitted tree above `c376115` contains, in addition to R0–R4, the earlier
**U2.2a-vet substrate** these units build on: document preparation
(`evidence/document.py`), per-key provenance + path-existence-AND-object-identity
proofs + `_is_addressable` + the classifier/worklists (`config_access.py`), the
non-checkpoint/unestablished census + `document_roots` (`parser.py`,
`scripts/census.py`, `docs/COR5_NET1_MIGRATION_DEBT.md`), loader verbatim-subtree
provenance (`adapters/diffusor/loader.py`), the duplicate-hydration deletion
(`encoder_panel.py`), and the transformer parser's earlier path fixes.
**R1 depends on this substrate.** So the first commit is really
**substrate + R0/R1** unless the reviewer decides to split a precursor
"U2.2a substrate" commit first. Decide the precursor boundary BEFORE splitting.

### File → unit map (verify against `git status` at the time)

- **Substrate (precursor, or fold into commit 1):** `evidence/document.py` (new,
  minus `DocumentBinding`), `config_access.py` (provenance consts, `provenance_of`,
  `_prove_path`, `_is_addressable`, `_container_prefix_for`, classifier +
  worklists, `resolve_priority`), `parser.py` (`_coerce_prepared`/`_raw_input`,
  `checkpoint_provenance`, published census), `encoder_panel.py` (dup-hydration
  delete), `adapters/diffusor/loader.py`, `adapters/transformer/parser.py`
  (path/rope fixes), `scripts/census.py`, `docs/COR5_NET1_MIGRATION_DEBT.md`,
  `tests/test_config_paths.py`.
- **R0:** `tests/test_u2_r0_reproductions.py` (new).
- **R1:** `evidence/document.py` `DocumentBinding`; `context.py`
  `prepared_documents`; `config_access.py` `bound_document` + `document_scope`
  legacy-debt note + export; `parser.py` root-binding block; `encoder_panel.py`
  binding + **`merge=False` (R1-vet)**; `tests/test_u2_r1_document_binding.py`.
- **R2:** `config_access.py` `ConsumedConfigDecision` + `ConfigResolution.
  consume_decision` + `ConfigOccurrenceKey.{document_path,reader}` + exports;
  `tests/test_u2_r2_consumed_decision.py`, `tests/test_u2_r2_raw_consume_debt.py`.
- **R3:** `evidence/arbitration.py` (new, whole); `tests/test_arbitration.py`.
- **R4 (NOT for these three commits):** `structural_writes.py`
  `StructuralWriteKey` + `_scan_raw` + `scan_structural_write_multiset` +
  `STRUCTURAL_WRITERS` baseline + `new_/stale_structural_writers`;
  `tests/test_u2_r4_structural_multiset.py`.

### R1/R2 separability — assessed, likely CLEAN (re-verify on fresh eyes)

R2's `config_access.py` additions are purely ADDITIVE and nothing in R1
references them: `ConsumedConfigDecision` (new class), `consume_decision` (new
method that only calls the pre-existing `consume`), and two DEFAULTED
`ConfigOccurrenceKey` fields. R1 production (`bound_document`, root/encoder
bindings) never touches them; R1's test file never imports them. So a 3-way
split appears to reproduce without a broken tree. **If fresh inspection shows the
`config_access.py` hunks cannot be cleanly separated, fall back to R1+R2
combined** and document the dependency (shared file, additive-but-interleaved).

### Per-commit gate (each of the three)

1. `git stash` others is NOT the mechanism — stage NAMED hunks/files only.
2. After staging a commit, verify its tree reproduces: the committed set imports
   and its targeted tests pass. Targeted tests:
   - R0/R1: `test_u2_r0_reproductions.py`, `test_u2_r1_document_binding.py`,
     `test_config_paths.py`, `test_config_access.py`.
   - R2: `test_u2_r2_consumed_decision.py`, `test_u2_r2_raw_consume_debt.py`,
     `test_config_access.py`.
   - R3: `test_arbitration.py`.
3. NEVER stage `docs/TRUE_CONFIG_PLAN.md` (concurrent actor) or the untracked
   `docs/U0_U1_*` / `docs/U2_DEFINITIVE_COMPLETION_SPEC.md` plan docs.
4. Commit message per spec §8 (defect, invariant, files, deletion/debt, poison,
   controls, artifact delta, command/counts/fingerprint).

### R4 remaining (after the three commits)

Complete static/runtime surface coverage §R4: field MUTATIONS (not just
definitions), Region/Op DEFAULTS, card/view kinds + structural PHRASES,
expanded-JSON structural keys, param formula BRANCHES, conformance expectations,
helper-returned-constant + table-driven resolution, recursive runtime visit; the
FULL 12-poison matrix (§R4 list); its own green receipt. THEN R5.

### COMMITTED (pushed to audio-composite-support)

Reconstruction found the intended R0/R1, R2, R3 boundaries could not hold as
specified: `config_access.py` interleaves substrate+R1+R2 (no clean hunk split),
and `test_u2_r0_reproductions.py` cross-depends on R1/R2/R3/R4 so it cannot land
until R4.  Soumil chose **2 commits now**; R0 file rides with R4.

- **`ac860e6` U2 substrate + R1 + R2** — the document/provenance rail
  (`document.py`, `config_access.py`, prepared-document boundary, census) +
  R1 (`DocumentBinding`, `bound_document`, root/encoder bindings, **embedded
  `merge=False` R1-vet fix**) + R2 (`ConsumedConfigDecision`, `consume_decision`,
  `document_path`+`reader` in the occurrence identity, raw-`.consume()` freeze).
  Combined because they share `config_access.py` and cannot be file-separated
  without a forced hunk split.  **Receipt: 167 passed / 10 skipped / 0 failed
  from an isolated archive of the committed tree.**  The isolated receipt CAUGHT
  a real bug — a Gemma-2 test asserted the merge=True violation as success; it
  was rewritten as a §3.3 U8 counterfactual and the commit amended.
- **`313307c` U2-R3 arbitration** — the generic evidence-ordering rail
  (`arbitration.py` + `test_arbitration.py`), no domain migration.  Clean leaf.
  **Receipt: 17 passed from an isolated archive (R4/R0 confirmed absent).**

Each commit's receipt is from `git archive HEAD` into a temp tree — reproduction
proven in isolation, not inherited from the working tree.

- **`2b1689a` U2-R4 structural census** — the writer-identity multiset REPLACES
  the (sink,target) set gate (which hid second writers).  `StructuralWriteKey`
  + count-aware `new_/stale_structural_writers` (a second write in one symbol is
  caught via the count; 40 count>1 keys pinned).  Complete static surface
  286→308 keys across **6 new sink types** (nested extras, spec field mutations,
  opgraph defaults, expanded-JSON keys, conformance expectations, param
  formulas) + helper-returned/local-constant dataflow resolution.  Recursive
  runtime surface (7636 nodes on llama, reaching `ir[layers][0][attention]`,
  value-hashed).  **12-poison matrix** — every representation censused.  The R0
  cross-unit reproduction ledger rides here (it references all units' symbols).
  Scanner-only, zero structural drift.  **Receipt: 70 passed from an isolated
  archive; gate clean (new=0, stale=0).**

### State

R0–R4 complete and pushed as three reviewed commits, each with an isolated-tree
receipt.  Structural surfaces (`ir/expanded/params/html_meta/gallery`)
byte-identical across all 25; only `ledgers`/`sable` evidence surfaces drift
(cumulative from the substrate) — the manifest re-bless is Soumil's and is
**withheld**.  Plan docs stay untracked; `docs/TRUE_CONFIG_PLAN.md` is the
concurrent actor's.

### ⚠️ REGRESSION SWEEP (2026-07-18) — pushed R0–R4 has failures the narrow receipts missed

Starting R5, a broad check found the pushed commits (ac860e6/313307c/2b1689a)
carry **8 real regressions** the per-commit receipts missed because they ran
NARROW targeted sets, not the full gate.  The substrate changed shared machinery
(census provenance admission, broad-except ratchets, the identity-taint net, the
composite modality path) that the narrow sets never exercised.  Lesson saved to
memory: **a commit receipt must run the broad gate.**

FIXED in the working tree (UNCOMMITTED — do not lose):
- `test_projection_audit::test_cor5_census_view_is_occurrence_exact` — the VAE
  fixture needed a provenance map (census admits only checkpoint provenance now).
- `test_reader_exceptions` (3) — `document.py`'s two broad excepts narrowed to
  typed (`_values_agree` → `(TypeError,ValueError)`; `AutoConfig` → the concrete
  set); `encoder_panel` baseline lowered 2→1 (duplicate-hydration delete win).
- `test_identity_guard` / `test_h4_taint` — `prepare_document` decorated
  `@identity_address` (its `model_type` use is lawful identity-as-address, like
  `context._installed_config_defaults`).

STILL FAILING (root narrowed, needs fresh diagnosis):
- `test_audio_composite` (3, musicgen) + `test_coverage` (1).  **Real structural
  regression:** musicgen's composite conditioning modality is GONE post-substrate
  (`ir.extras["modalities"]` absent; present at c376115).  `multimodal_extras`
  returns None for musicgen in the REAL parse (namespace='root') even though
  `_modality_host` and `spec.build` both succeed in ISOLATION — so it is a
  context-dependent veto inside the modality loop, most likely the nested
  document/container scope interaction introduced by the R1 `builder.py`
  addressability change (`_wrapper_path(cfg, host)` + `config_container(_container,
  obj=sub_cfg)`) or the `bound_document` nesting.  musicgen is NOT a preservation
  witness, which is why the 25-witness gate did not catch it.

ROOT CAUSE (Soumil traced it): `prepare_document` passed nested values from
`raw` straight into `AutoConfig.for_model`, which MUTATES the nested component
dicts (pops `model_type` from a composite's `text_encoder`/`audio_encoder`).  In
shadow mode `document is raw`, so the mutation corrupted the checkpoint the
parser reads — MusicGen's conditioning tower lost `model_type`, its presence gate
rejected it, the modality vanished.  FIX: hydrate from a throwaway DEEP CLONE
(`_snapshot(raw)`); nothing reachable from `raw` reaches AutoConfig.  Verified:
raw unmutated, nested `model_type` survives (t5/encodec), modality restored.
7 permanent mutation tests pin the invariant (`tests/test_u2_document_mutation.py`).

Two follow-on corrections this surfaced:
- The `AutoConfig` except was WRONGLY narrowed earlier — `StrictDataclassField
  ValidationError` (huggingface_hub) is neither ValueError nor TypeError, so a
  typed list silently missed real rejections and crashed the parse. The boundary
  is GENUINELY open, so it is `except Exception` PINNED as justified ratchet debt
  (`evidence/document.py: 1`) — it converts any rejection into a typed
  `PreparationFailure`.  The `_values_agree` except stays typed (its modes are).
- Soumil's proviso #3: the `@identity_address` marker BLANKET-exempted, so a
  decorated function returning STRUCTURE was laundered.  Tightened: the marker
  exempts an address/display use only — a decorated branch that writes a
  STRUCTURAL SINK is still debt (poison added; `prepare_document` stays lawful,
  identity debt 0).  Closed a stale strict-xfail (`test_p8`) the fix un-blocked.

### ⭐ REQUIRED at R9 — add MusicGen to the preservation witness set

Soumil: the regression proved the 25-model set misses composite-conditioning
ownership.  Add `tests/sable_test_corpus/musicgen-small.json` (source: the
`MUSICGEN_SMALL` fixture in `test_support`) to the corpus and bless its manifest
entry.  DEFERRED to R9 because it changes the preservation MANIFEST (bless) and
ripples into census + every corpus-parametrized test — bless is withheld until
R9.  The behaviour is already restored, so it will pass on addition.

DONE: `a443661` "fix: substrate regressions missed by narrow commit receipts"
committed FORWARD (pushed history not amended) and pushed.  Broad receipt: full
suite `-p no:randomly` = 1411 passed / 10 skipped / 4 xfailed, sole failure
`test_expected_manifest_zero_drift_zero_skip` (manifest re-bless, withheld until
R9).  Isolated-tree receipt over the blast-radius set (mutation, audio, coverage,
reader-exceptions, identity, taint, authority-probes, projection-audit) = 141
passed / 4 xfailed / 0 failed; the archive confirms `registry.py` carries NO
`ProjectionRoute` (R5 foundation stays WIP, excluded from this commit).  R5 now
resumes.

### R5 — locked plan (Soumil's details, 2026-07-18) + the key finding

FOUNDATION DONE (green, uncommitted WIP in `registry.py`):
- 9 surfaces, `spec` explicit, verified EXACTLY equal to
  {ir, spec, opgraph, block, card, html, json, params, conformance}.
- `ProjectionRoute` type; `FactDefinition.projection_routes` field.
- `projection_routes` is registry METADATA, not a tenth projection surface.

KEY FINDING driving the core: `projector_out_features` is NOT an `EvidenceFact`
today — the pilot flows through the config-access OBLIGATION path, and the
renderer's `projects` descriptor supplies the value/status hash.  R5 requires it
to become a real fact so the expected hash originates from the FACT, not the
renderer.  Status must be **`code_and_config`** when source proves the projector
consumes that exact config value — NOT `config_declared` just because a number
exists.

CORE (one coupled unit; shadow-first so the rail never breaks mid-way):
1. RECORD `projector_out_features` (vision + video) as an `EvidenceFact`
   (`code_and_config` when source-proven) at the consumption site in
   `modalities/vision.py`.
2. Register it as a `FactDefinition` with a `ProjectionRoute` (closed-world:
   the fact must then be observed AND its route exercised).
3. `FactDefinition` becomes the SOLE route authority: derive `RECEIPTED_SCOPES`
   from `projection_routes`; REMOVE `MigrationClaim.projection` + `ProjectionPolicy`.
4. Upgrade `ProjectionReceipt` to §5.5 (fact_id, owner, fact_key, mechanism,
   fact_value_status_hash, surface, structural_target, projector_symbol,
   node_ids, output_hash).
5. Validator joins + validates ALL of: context, fact_id, owner, fact_key,
   mechanism, value/status hash, surface, structural_target, projector_symbol,
   node identity — expected hash from the FACT/consumption, never the renderer.
6. EMIT the receipt INSIDE the actual projector (`block_views/modality_views/
   vision.py` / `video.py` / `metadata_modalities.py`), not an upstream helper.
7. Controls: Qwen2-VL = ONE exact obligation/fact/receipt chain (positive);
   FLUX + Qwen-Image = NEITHER projector consumption NOR receipt, phantom
   consumption stays VISIBLE (negative).
8. Poisons for `spec` AND every other surface (wrong surface/owner/fact/mechanism/
   hash/target/projector/node/context each blocks).
9. Keep compat witnesses (`facts_projected`) in SHADOW until parity proven;
   delete only for the migrated routes.
10. Pixels byte-identical.  GATE: focused R5 tests + broad full-suite
    `-p no:randomly` + isolated committed-tree receipt.  Land R5 INDEPENDENTLY.

BINDING (Soumil): every R5–R9 commit needs BOTH its focused gate AND the broad
shared-infrastructure gate.  Manifest + MusicGen witness deferred to R9.  No
bless before R9.

### R5 CORE IMPLEMENTED (uncommitted; broad gate running)

All of Soumil's locked details landed and verified:
- **9 canonical surfaces**, `spec` explicit, `projection_routes` = registry
  metadata (verified not a tenth surface).
- **`projector_out_features` = `code_and_config`** typed fact (recorded at the
  consumption via the ambient `capture_facts` rail; registered `FactDefinition`
  with two `ProjectionRoute`s onto `card`/`vision_projector|video_projector`).
  Never `config_declared` for a bare number.
- **FactDefinition = sole route authority**: `ProjectionPolicy` DELETED,
  `MigrationClaim.projection` removed; `RECEIPTED_SCOPES` is a lazy
  registry-derived view.
- **§5.5 receipt** (fact_id/owner/fact_key/mechanism/fact_value_status_hash/
  surface/structural_target/projector_symbol/node_ids/output_hash) +
  `context_token` stamped by the RENDER CONTEXT itself (unforgeable by the
  projector).
- **Validator strict on every field**; expected hash from the FACT and the
  consumption; disagreement between those two upstream authorities BLOCKS.
- **Emission inside the actual projector** (`declared_ops.build_declared_ops_
  view`), which cites the LEDGERED fact's status (`fact_rows` param) — the
  descriptor carries the drawn value only; the renderer never derives a tier.
- **Controls**: Qwen2-VL = one exact obligation/fact/receipt chain per lane,
  0 findings, 2 receipted targets (non-vacuous).  FLUX/Qwen-Image = no
  top-level consumption/fact/receipt; the EMBEDDED tower's consumption
  (root.text_encoder.vision) stays VISIBLE as advisory census / exact R6 debt.
- **38 tests** incl. per-field poisons + all-9-surface wrong-surface poisons +
  anti-vacuity (empty registry cannot green).
- **Pixels byte-identical**: ir/expanded/params/html_meta/gallery unchanged
  across 25 (an early `out_width_fact_status` stamp DID drift ir/expanded on 1
  witness — caught by the preservation gate, fixed by citing the ledgered fact
  at emission instead of stamping the IR payload).
- **R4 gate proved itself**: it flagged the new `record_typed` author in
  `_bound_out_width`; pinned as a reviewed addition.
- Found + fixed en route: the closed-world registry rejection for EMBEDDED
  owners was being swallowed by the modality try/except, silently dropping the
  whole projector-evidence application for embedded VL encoders — now a
  deterministic registry-gated skip (unmigrated owner ⇒ obligation-only,
  visible).

**R5 LANDED: `478e34a`, pushed.**  Gates per the binding both-gates rule:
broad = full suite 1424 passed / 10 skipped / 4 xfailed, sole failure the
withheld manifest bless; isolated committed-tree receipt = 461 passed over the
blast-radius set (receipts, projection_audit, fact_registry, R4 multiset,
structural_writes, evidence_facts, h9_frontier).  The broad gate caught two
regressions pre-commit (a monkeypatch stub signature; the PARTIAL-SOURCE law
requiring `legacy_source` on the pilot fact) — both fixed before landing.

### Remaining after R5

R6 (StructuralDebt) → R7 (classify occurrences, delete `document_scope` legacy)
→ R8 (blocking cutover) → R9 (verify + MusicGen witness + summary + bless).

**These validate the WORKING TREE, not any commit.** They are NOT commit-level
receipts. Each of R0/R1, R2, R3 needs its OWN receipt: after staging that
commit's named files, run its targeted gate FROM THE COMMITTED TREE (a clean
checkout / `git stash` of everything else, or equivalent) and confirm it imports
and passes there. A commit whose gate only passes because sibling uncommitted
changes are present does NOT reproduce and is invalid — that is exactly what the
per-commit reproduction check must catch. R4 lands separately only after its
COMPLETE gate (full surfaces + all 12 poisons) passes as its own receipt. No R5,
no bless, before R4 is complete.

## R9-CORRECTION LIVE STATE (post-compaction: resume HERE)
DONE: kill-shots 1+2; scoped-rule rail (ledger_ignores rules; global feed
dead); 4 is_enc_dec pending rows deleted; owner-bound fact_* conditions;
writer-exact growth gate; drawn_unledgered_pairs; poisons file
test_u2_r9_final_corrections.py (12 tests); MusicGen made MECHANICAL CLEAN
(composite main-slot wrapper_path + prefix-owner mapping; conditioning slot
DocumentBinding boundary in conditioning.py; codebooks/audio_channels/
scale_embedding consumed; decoder/audio_encoder address keys; absolute-path
composition in parser.py coverage join); focused battery 156p; census 0.
REMAINING: (1) sable gallery for MUSICGEN_SMALL → inspect ALL PNGs vs
VISUAL_RUBRIC → report.visual_review="CLEAN" → bless(report,
MUSICGEN_SMALL, corpus_dir=tests/sable_test_corpus) [DELEGATED by Soumil];
(2) 25→26 pins: test_support/preservation.py:285-286 + test_preservation
poison param strings "witness_count != 25"; (3) rebuild
preservation_expected_manifest.json via P.build_expected_manifest + DIFF
old-vs-new: assert 25 witnesses' ir/expanded/params/html_meta/gallery
hashes UNCHANGED (evidence-only delta: ledgers/sable) + musicgen-small
added — document the inspection; (4) copy completion summary INTO repo
docs/U2_COMPLETION_SUMMARY.md (update: R9-correction content); (5) full
suite = ZERO failures; commit EVERYTHING as THE R9 commit (incl. docs,
census, manifest, fixture, gallery files if tracked — galleries are
GITIGNORED, fixture json is TRACKED); (6) rerun full bracket (zero
failures + fingerprint identical); (7) push. NEVER stage MODULARIZATION.md
/ TRUE_CONFIG_PLAN.md / untracked plan docs.
