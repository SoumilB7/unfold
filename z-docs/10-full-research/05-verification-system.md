# 05 — The verification system: what it proves, what it does not, what it costs

Pass written 2026-09-02 against tree `9b3cb7b` (branch `audio-composite-support`),
read-only. Every claim below was verified from the code at the cited `file:line`
(paths relative to `unfold-pkg/`); statements I could not verify are marked
**(unverified)** or **(inferred)**. Receipt timings come from the runner's own
logs under `/private/tmp/model-unfolder-verification/` (see §3).

One-paragraph answer. The system is a very strong **drift + self-consistency +
process** lock and a moderately strong **fabrication** net — it can prove that
nothing in 29 frozen renders changed and that no structural claim was authored
outside a censused register. It is a weak **recall** net: nothing measures how
much of a model's true structure the diagram *omits* except the two conformance
directions that compare the picture to the same AST reading the parser used, and
nothing compares any output to an independent oracle (an instantiated module tree,
a trace, a golden fact list). The human visual review — the only check aimed at
"does a person learn the right thing" — is an unpersisted string on a dataclass
(§5.2). The cost of one committed-tree receipt is ~90–97 min wall and ~8 CPU-hours.

---

## 1. Check inventory

### 1.1 The Sable nets (`model_unfolder/sable.py`, one `SableCheck` each)

`SableCheck.blocking` defaults to `True` (`sable.py:63`); `mechanical_passed`
consults only blocking nets (`:82`); `blessable` additionally requires
`oracle == "present"` and `visual_review == "CLEAN"` (`:90-91`). 25 nets are always
appended; the 26th (`gallery`) is appended only when PNG rendering fails and
carries a note, never findings (`:657`).

| # | net (`sable.py` line) | predicate, one sentence | blocking | input inspected | what it would miss |
|---|---|---|---|---|---|
| 1 | `click_coupling` (450) | every `data-id` clickable node resolves to a card in the next inspect panel (`block_schema.py:209-215`) | yes | baked HTML | wrong content behind a working click; invisible nodes |
| 2 | `dangling_connectors` (451) | every ⊕/×/⊙/‖ drawn has both inputs in the graph model (`Diagram.wiring_problems`) | yes | graph model, not pixels | an edge present in the model but rendered invisible (PROTOCOL.md Dable §2 admits this) |
| 3 | `unique_ref_ids` (452) | every `url(#id)` def is document-unique (`block_schema.py:306-312`) | yes | HTML | anything non-referential |
| 4 | `no_dotted_arrows` (453) | no arrow-bearing line is dotted (`block_schema.py:334-340`) | yes | HTML | semantics |
| 5 | `no_dotted_boundaries` (454) | no structural boundary is dotted (`:354-360`) | yes | HTML | semantics |
| 6 | `config_field_audit` (459-467) | every present config field is bound, exact-debt-registered, or scoped-ignored (`extras.config_audit.unread`) | yes | ledger sidecar | a field read and then misused; a field consumed into the wrong fact |
| 7 | `op_conformance` (468-470) | per layer-group representative, the diagram op-kind set equals the `forward()` op presence set both directions (`conformance.py:118-157`) | yes, **skipped w/o oracle** | IR + AST of the resolved block class | op order, wiring, operand identity, semantics of same-kind ops; only ONE representative per `classify_group` (`:141-143`); `abstractions.yaml` allow-list |
| 8 | `wiring_conformance` (471-475) | every drawn conditioning side-input matches a parameter ROLE present in the block's `forward()` signature (`conformance.py:439-452`) | yes, skipped w/o oracle | IR + forward params | exact edges ("role PRESENCE, never exact edges", `:449-451`) |
| 9 | `fact_conformance` (479-485) | same-op-kind/different-semantics facts agree with code: NoPE-vs-rotary, linear-vs-softmax, MLA kind, projector/recursive-component/fusion/storage facts (`conformance.py:504-528`, `:555-575`) | yes, skipped w/o oracle | IR + AST tokens + class-name substrings | any fact family not enumerated; relies on substring markers (`fact_markers.yaml`) |
| 10 | `nested_conformance` (490-495) | each drill's DRAWN op set vs the transitive `forward()` closure of its sub-module; fabrication strict, omission "scoped"; router/composite get weaker block-altitude checks; an EMPTY closure is skipped as honest-unknown (`conformance.py:884-915`) | yes, skipped w/o oracle | render log events + AST closure | omissions inside a skipped empty closure; anything an opaque delegation hides |
| 11 | `label_lint` (496) | no nested parens, raw HF class names, static Tier-2 connectors in labels (`lint.py:66-91`) | yes | IR blocks | wrong-but-well-formed labels |
| 12 | `evidence_ambiguity` (506-509) | no block's `detail.evidence.status == "ambiguous"` while source was scanned; `oracle_missing` exempt (`sable.py:138-182`) | yes | IR block tree | a wrong *resolved* answer |
| 13 | `asserted_facts` (514-518) | list every spec `asserted` default tuple (`:117-135`) | **no (advisory)** | IR layer specs | — (advisory; zero live findings in all 29 baselines) |
| 14 | `projection_audit` (524-528) | every code/config-proven fact on a DRAWABLE family appears in some `RenderEvent.facts_projected` (`:208-234`) | yes (`_PROJECTION_AUDIT_BLOCKING = True`, `:188`) | ledger + render log | facts on non-drawable families; a projection that draws the wrong value (key-set only) |
| 15 | `qualified_projection_values` (535-537) | for each of 17 `QualificationRule`s the canonical spec value equals the owner-qualified typed fact value (`qualification.py:142`) | yes | IR spec + ledger | the 17-family boundary (attention geometry/schedules, FFN width/routing, codebooks, PLE, position addition); other surfaces (card/json/params) are "consumer-test inventory", not checked here (`qualification.py` docstring) |
| 16 | `zero_asserted_census` (544-547) | strip config to numbers+address, re-parse with an EMPTY `SourceBundle`, require the `asserted` set ⊆ `_CENSUS_ALLOWED` (= ∅, `:196`) | yes | a synthetic re-parse | anything when the numbers-only parse raises — returns `[]` on `UnfoldError` AND on any `Exception` (`:289-292`) |
| 17 | `config_accessed_unprojected` (551-555) | list accessed-but-unconsumed occurrences (`extras.config_access.accessed_unconsumed_exact`) | **no (advisory)** | ledger | — (13 live findings: granite 2, SDXL 11 — `preservation_baseline/*/sable.json`) |
| 18 | `config_migration_claims` (562-567) | no violation inside a claimed (owner, mechanism, path) scope (`claims_audit.py:35`) | yes | ledger claim rows | reads outside the 3 `MIGRATED_SCOPES` (`registry.MIGRATED_SCOPES`, len 3) |
| 19 | `config_consumed_unreceipted` (575-580) | every consumption obligation in a receipted scope has a matching render receipt (`receipts.py:215`) | blocking **only inside** the 41 receipted scopes | obligations + receipts + fact rows | obligations in the unreceipted remainder (advisory census) |
| 20 | `receipt_fabrication` (585-587) | every emitted receipt cites a registered fact or a claimed target (`receipts.py:377`) | yes | receipts | a receipt whose *value* is wrong but whose hash was computed from the same wrong fact |
| 21 | `config_audit_incomplete` (597-601) | no owner with zero consumed events; no conflicting declaration left unreported | yes | ledger | — |
| 22 | `document_boundary_completeness` (606-613) | no unlocated read, no unestablished provenance | yes | ledger | — |
| 23 | `config_standing_unconsumed` (619-622) | every accessed-but-unconsumed occurrence is excused by an EXACT pending debt row (owner + dotted path) (`:307-323`) | yes | ledger ⋈ `STRUCTURAL_DEBT` | everything the 94 blanket closure rows excuse (§5.8) |
| 24 | `structural_debt_register` (628-631) | tree-wide: no unregistered/stale structural writer and `debt_problems() == []` (`structural_debt.py:1250-1273`) | yes | the source tree (not the model) | unit liveness (§5.7); re-run identically for every model |
| 25 | `config_ambiguity` (632-636) | no conflicting checkpoint declaration rows | yes | ledger | — |
| 26 | `gallery` (657) | note-only: PNG render failed | n/a | — | never fails |

Oracle dependence: nets 7–10 are the only ones that look at code, and all four
return `[]` when `context.source_bundle.files` is empty (`sable.py:446-495`); the
report then says `oracle: MISSING (conformance degraded …)` (`:444`).

### 1.2 Structural gates in `model_unfolder/evidence/` (tree-wide, test-time)

| gate | what it proves | size today | blocking test | what it misses |
|---|---|---|---|---|
| `structural_writes.py` (1097 lines) | static AST census of every structural author (spec class, extras leaf, opgraph `Op`/`Region`, card key, param assumption); a new `(module, symbol, sink, target)` or a stale pin fails | 200 pinned `STRUCTURAL_SURFACE` entries | `test_structural_writes.py` (13), `test_u2_r4_*` (10) | dynamic writes (setattr/dict merge) the scanner's sink vocabulary does not name; `StructuralWrite.key` is still `(sink, target)` (`structural_writes.py:87-88`; strict-xfail `test_authority_probes.py:219`) |
| `structural_debt.py` (1273 lines) | ONE register of remaining debt with owner/writer/consumer/unit/deletion-condition; `debt_problems` = duplicates + dead writer + dead consumer + satisfied condition + unrowed extras write + unrowed consumer read (`:1250-1273`) | 279 rows: config_read 186, consumer_read 83, extras 10; units U14 107, U15 80, U11 37, U12 34, U9 19, U5 2 | `test_structural_debt.py` (32) + Sable #24 | unit liveness; blanket closure rows (§5.7–5.8) |
| `consumer_firewall.py` (746) | HTML/JSON/params/conformance may import only canonical projection layers and may not read config/extras/source; 8 `ACCESS_KINDS` (`:28-31`) | 83 exact consumer-debt rows (`_CONSUMER_DEBT_BASELINE`) | `test_consumer_firewall.py` (21) | a consumer that re-derives structure from a *lawful* IR field |
| `identity_guard.py` (926) | AST guard for identity predicates + name-blind differential parse; class-keyed tables lawful only with `(path, table, fingerprint)` manifest entry | 21 lawful tables (16 `code_shape`, 2 display, 2 declared_role, 1 declared_component); debt = 0 | `test_identity_guard.py` (26), `test_h4_taint.py` (17), `test_authority_probes.py` (27, 2 strict xfail) | the conformance oracle's own class-name substrings (§5.10); an unconditional structural return under `@identity_display` (strict xfail `:202`) |
| `claims_audit.py` (188) | exact owner+mechanism+path → (owner, fact, sink) binding validation; corpus-level anti-vacuity | 3 migrated scopes | via Sable #18 + `test_config_*` | — |
| `qualification.py` (394) | value-exact spec ↔ fact for migrated families | 17 rules | Sable #15, `test_qualification.py` | non-migrated families |
| `legacy_reader_quarantine.py` (581) | the surviving `_from_files` readers, their callers and parse-authority sites are pinned by two content fingerprints (`:22-25`) | 5 readers (all U11), 20 parse-authority sites | `test_legacy_reader_quarantine.py` (13) | — (a quarantine, not a correctness check) |
| `config_guard.py` (464) | exact short-circuit evaluation of config-bound source guards; unknown syntax never picks a branch | reader helper | its own poisons | — |
| `receipts.py` (437) | occurrence → fact → route → receipt join with context token; reverse fabrication | 41 receipted scopes, 38 facts with routes, `REGISTRY` 62 facts | `test_receipts.py` (45) | value truth (hash of the same fact) |

### 1.3 Generated inventories

| script | output | staleness gate | last regenerated |
|---|---|---|---|
| `scripts/census.py` | `docs/COR5_NET1_MIGRATION_DEBT.md` (`--check` diffs) | **not wired** into any test or the receipt (grep of `tests/` and `verify_commit.py`) | 2026-08-18 |
| `scripts/generate_u3_reader_inventory.py` | `docs/U3_CURRENT_READER_INVENTORY.md` | `test_legacy_reader_quarantine.py:54-59` | 2026-08-28 |

### 1.4 Preservation harness (`test_support/preservation.py`, `tree_state.py`, `tests/test_preservation.py`, `test_isolation.py`)

- 7 canonical surfaces per witness (`ir`, `ledgers`, `expanded`, `params`, `html_meta`, `sable`, `gallery`; `preservation.py:40-42, 85-111`) hashed into `tests/preservation_expected_manifest.json` with input sha and an ORDERED `(label, sha)` view list (`:259-273`, verified at `:367-372`); count pinned at 29 (`:47`).
- Baseline dir `tests/preservation_baseline/<slug>/{7}.json` is the human-readable twin; `compare_model` (`:194-211`) treats the baseline's HTML sha as `None` (`:144`) — only ids/targets are compared there, but the expected-manifest path compares the full canonical `html_meta`.
- Poisons: 8 surface mutations (`test_preservation.py:62-82`), 8 manifest violations (`:185-240`), mount-UUID normalization on real renders (`:110-127`), determinism (`:130-137`).
- Tree quiescence: `tree_state.fingerprint` over every non-excluded file incl. exec bits (`tree_state.py:46-73`); excludes `preservation_baseline/` and `galleries/` (`:23-27`) which the runner hashes separately (`verify_commit.py:43-46, 104-115`).
- Clean-checkout reproducibility: `git archive HEAD` must carry all 29 inputs + gallery hashes (`test_isolation.py:154-194`) and must import + parse llama and flux (`:197-248`).
- `canonical_surfaces` calls `mu.unfold(cfg)` and `sable(cfg, render_images=False)` with **no `source` argument** (`preservation.py:91, 94`) — see §5.6.

### 1.5 Commit receipt runner (`scripts/verify_commit.py`)

Six lanes in detached worktrees (`:146-151`), each fingerprinted before/after with the tree AND the external artifacts (`:160-172`): `focused` (user `--focus` + 4 kernel files, 1 worker), `u2-authority` (4 files, ≤4 workers), `collect` (`--collect-only tests`), `static` (pyflakes on changed files, `git diff --check`, forbidden-symbol AST scan; `:175-210`), then `full` (everything minus preservation and the 4 authority files, via `pytest_file_bracket.py` in fresh 8-file batches; `:278-290`) and `preservation` (`--durations=10`, ≤4 workers). Fail-fast after preflight (`:379-380`); a receipt also fails if the coordinator's own two files or the source artifacts changed during the run (`:430-438`). `--focus` is mandatory (anti-vacuous, `:336-337`). Logs and `receipt.json` go to `/private/tmp/model-unfolder-verification/<run>/` (`:42`) — nothing is copied into the repository (§5.9).

### 1.6 Coverage tests (`tests/test_coverage.py`)

7 tests over the 17-entry SYNTHETIC `test_support.CORPUS` (keys: `audio, codec_lm, dense_gated, dense_mlp, dit_audio, dit_cross, dit_hybrid_encoder, dit_mmdit, dit_moe_encoder, dit_refiner, dsa, moe_mla_mtp, ple, self_cond, unet, video, vision`): every registered view key is exercised by some corpus model (`:57-81`) except 6 declared fallbacks (`ops, tower, ffn, mtp_head, mtp_transformer_block, per_layer_embedding`, `:48-54`); corpus-wide click-coupling, unique defs, no dotted strokes, no dangling connectors, no all-static drill, no static connector, and the image pass is exhaustive+deduped (`:84-211`).

### 1.7 Corpus (`tests/sable_test_corpus/`)

29 fixtures, each `{model, source:"local", config, hash_signature, checks{25 names→bool}, visual_evidence{gallery_dir, png_count, manifest}, superseded_hash_signature}` (all 29 carry a superseded signature, i.e. every one has been re-blessed at least once). 29 gallery dirs with PNG + `MANIFEST.txt`; **8 of 29** carry a `her_eyes_review.md` sidecar (gemma-2, glm-4.5, gpt-oss, llama, olmo-2, qwen2-vl, qwen3-8b, sdxl). `check_regression` (`sable.py:821-835`) re-runs Sable and compares blocking findings + the sorted hash multiset; `test_sable.py:573-597` runs it over the corpus with `expected_unresolved = {}`.

### 1.8 Audit scripts (network, not in any receipt)

| script | catalogue | last committed output |
|---|---|---|
| `scripts/coverage_audit.py` | 49 ids / 10 families (hard-coded `MODELS`) | `docs/coverage_audit.md` 2026-07-04: 35 parsed, 13 gated, 1 errored, 0 unparsed fields |
| `scripts/dit_coverage.py` | 31 ids / 7 families | no committed doc |
| `scripts/serve_audit.py` | parses `toserve.md` (111 ids seen) | `docs/serve_audit.md` 2026-07-04: 105 audited, 94 clean, 5 errored, 12 gated |
| `scripts/pytest_file_bracket.py` | — | exact-collection bracket used by the `full` lane (`:1-20`) |

All three model-sweep docs are two months stale relative to the U-plan changes and none is gated.

---

## 2. Direction-of-error matrix

Directions: **F** fabrication (claims false things), **R** omission/recall (loses or never states true things), **D** drift (anything changed), **S** self-consistency (fact vs spec vs render), **P** process (quiescence, isolation, exactness), **I** identity taint, **C** coverage (every view/scope exercised). Primary direction first; secondary in parentheses.

| check | F | R | D | S | P | I | C |
|---|---|---|---|---|---|---|---|
| Sable 1 click_coupling | | | | ● | | | |
| Sable 2 dangling_connectors | | | | ● | | | |
| Sable 3 unique_ref_ids | | | | ● | | | |
| Sable 4–5 dotted | | | | ● | | | |
| Sable 6 config_field_audit | | ● | | | | | |
| Sable 7 op_conformance | ● | ● | | | | | |
| Sable 8 wiring_conformance | ● | (●) | | | | | |
| Sable 9 fact_conformance | ● | | | | | | |
| Sable 10 nested_conformance | ● | (scoped) | | | | | |
| Sable 11 label_lint | | | | ● | | | |
| Sable 12 evidence_ambiguity | | ● | | | | | |
| Sable 13 asserted_facts (adv) | (●) | | | | | | |
| Sable 14 projection_audit | | ● | | | | | |
| Sable 15 qualified_projection_values | ● | | | ● | | | |
| Sable 16 zero_asserted_census | ● | | | | | | |
| Sable 17 config_accessed_unprojected (adv) | | (●) | | | | | |
| Sable 18 config_migration_claims | ● | | | ● | | | |
| Sable 19 config_consumed_unreceipted | | | | ● | | | |
| Sable 20 receipt_fabrication | ● | | | | | | |
| Sable 21 config_audit_incomplete | | ● | | | | | |
| Sable 22 document_boundary_completeness | | | | ● | ● | | |
| Sable 23 config_standing_unconsumed | | ● | | | | | |
| Sable 24 structural_debt_register | | | ● | | ● | | |
| Sable 25 config_ambiguity | ● | | | | | | |
| hash_signature lock / check_regression | | | ● | | | | |
| preservation 7 surfaces + expected manifest | | | ● | | | | |
| tree/artifact/coordinator fingerprints | | | | | ● | | |
| worktree lanes, collect exactness, bracket | | | | | ● | | |
| test_isolation (no test imports; clean checkout) | | | | | ● | | |
| structural_writes census | ● | | ● | | | | |
| structural_debt register gates | | | ● | | ● | | |
| consumer_firewall | ● | | ● | | | | |
| identity_guard + h4 taint + name-blind diff | ● | | | | | ● | |
| legacy_reader_quarantine fingerprints | | | ● | | | | |
| metamorphic (rename/collision/provenance/missing-source) | ● | | | | | ● | |
| test_coverage (every view exercised) | | | | | | | ● |
| claims_audit anti-vacuity (every binding observed) | | | | | | | ● |
| loud-miss token ratchet + oracle floor | | ● | | | ● | | |
| u4_unknown_safe / modality honesty | ● | | | | | | |
| submodel parity | | | | ● | | | |
| audit scripts (coverage/dit/serve) | | (●) | | | | | ● |

**Counts by primary direction** (42 rows): self-consistency **11**, fabrication **13**, drift **8**, recall **7** (of which 2 advisory and 2 "scoped"), process **8**, identity **3**, coverage **3**. Counting only *blocking, per-model, code-comparing* recall checks: **one** (`op_conformance`'s missing direction, on one representative layer per group). Nothing on the recall axis compares the render to anything other than the same AST reading that produced it.

What each direction actually establishes, and against what reference:

| direction | reference the check compares against | strength | why |
|---|---|---|---|
| drift | the tree's own past output (baseline JSON, manifest hashes, pinned censuses) | **strongest** | every surface of every witness is byte-hashed; 16 poisons prove the diff fires; tree quiescence is enforced around every lane |
| self-consistency | one output surface vs another (HTML vs cards, spec vs fact, obligation vs receipt) | strong | exact joins with context tokens; poisons per field (`test_receipts.py`, 45) |
| process | the run itself (fingerprints, exact collection, isolation) | strong | but receipts live in `/tmp` and are already lost for ≥7 cited runs (§5.9) |
| fabrication | the AST of the modeling file the parser also read | moderate | catches an op/input/scheme the code lacks; blind to a fact the AST reader mis-resolves the same way twice; skipped without oracle and `check_regression` does not notice (§5.5) |
| identity | the tree's own source (no identity predicate writes structure) | moderate | debt zero and blocking, but the oracle side is exempt (§5.10) and 2 strict-xfail shapes remain |
| coverage | the view registry and the claim bindings | narrow | "every view is drawn by SOME synthetic dict" — not "every mechanism of every catalogued model is drawn" |
| recall | the same AST reading, one representative per group | **weakest** | no golden facts, no instantiated-module oracle, no unknown-rate bound; `nested_conformance` explicitly scopes omission and skips empty closures (`conformance.py:900-915`) |

The asymmetry matters for the goal: "never guessing" is policed from six angles; "never silently omitting" is policed from one, and that one is only as good as the op-token vocabulary (`op_tokens.yaml`: 48 mapped tokens, **108 ignored**) that decides which `forward()` calls count.

---

## 3. Test-suite anatomy

- `pytest --collect-only -q` today: **3991 tests across 179 files** (my run, 6 s). At the last full receipt (`93b47f2c98`, commit `2401b2e`, 2026-09-01): 3970 collected = 3874 in the bracketed `full` lane (173 files) + 52 preservation + 44 authority.
- Largest files by tests: `test_evidence_facts.py` 330, `test_config_paths.py` 103, `test_diffusion.py` 96, `test_attention_mechanism.py` 95, `test_code_evidence.py` 83, `test_conformance.py` 71, `test_ffn_mechanism.py` 70, `test_program_index.py` 62, `test_model_stage.py` 57, `test_smoke.py` 57.
- **Slowest tests** (preservation lane `--durations=10`, `93b47f2c98/preservation.log`): `qwen-image` 435 s, `flux-2-dev` 315 s, `musicgen-small` 238 s, `prxpixel-t2i` 183 s, `hunyuanvideo` 169 s, `sd3.5-large` 155 s, `qwen2-vl` 143 s, `qwen3.5-27b` 141 s, `auraflow` 136 s, `fluxtransformer2dmodel` 133 s — each is one witness regenerated through `unfold` + `sable` twice (`canonical_surfaces` calls both, then `_view_hashes` unfolds again: `preservation.py:91, 94, 271`).
- **Slowest files** — only batch granularity exists (bracket logs have no per-test durations): batch-009 **2802 s** = `test_smoke, test_coverage, test_fact_registry, test_isolation, test_qk_norm, test_kv_sharing_schedule, test_container_inventory, test_unet_selected_stage_children`; batch-018 2156 s = `test_sable, test_tower, test_decoder_block, test_decoder_norm, test_fusion_evidence, test_projection_bias, test_attention_softcap, test_stage_operations`; batch-016 1832 s; batch-003 1452 s; batch-013 1409 s; batch-001 1392 s; batch-005 1339 s (contains `test_conformance`); batch-017 1156 s. Anything that renders the real corpus is the long pole.
- **Full lane**: 22 batches, 24,081 batch-seconds ≈ **6.7 CPU-hours**, 4910 s wall on 6 workers. Every lane's fingerprint was identical before/after in both receipts below.

| lane (`93b47f2c98`, 2026-09-01, commit `2401b2e`) | tests | workers | wall s | note |
|---|---|---|---|---|
| collect | 3970 collected | 1 | 18 | exact-collection baseline for the bracket |
| static | — | 1 | 0.4 | pyflakes on 8 changed files + `--check` + forbidden symbols |
| focused | 255 | 1 | 881 | `--focus` files + 4 kernel files |
| u2-authority | 44 | 3 | 765 | identity guard, multiset, R8 nets, reader exceptions |
| full | 3874 (3858 p / 14 s / 2 xf) | 6 | 4910 | 22 fresh-process batches, 173 files |
| preservation | 52 | 4 | 1008 | 29 witnesses regenerated + 23 poisons/controls |
| **receipt** | | | **5800 (97 min)** | preflight ≈ 15 min, heavy phase ≈ 82 min |

The older receipt `5564c3881a` (commit `7ade5cf`, 2026-08-28) ran 5217 s wall (full 3777 s, preservation 1059 s, focused 1432 s, authority 693 s). Between the two receipts the full lane grew ~30 % in wall time (3777 → 4910 s) with the same worker count; the corpus-rendering tests are the growth term. Estimated CPU budget of one receipt: full 6.7 h + preservation ~1.1 h (4 workers × 1008 s) + preflight ~0.5 h ≈ **8.3 CPU-hours**, of which roughly a quarter is the same 29 witnesses being unfolded repeatedly (preservation renders each witness three times: `preservation.py:91, 94, 271`; `test_sable`, `test_fact_registry`, `test_config_paths`, `test_conformance` and ~30 other files unfold them again).
- **skip/xfail**: 14 skipped (all in batch-001; the only per-witness `pytest.skip` in that batch is `test_config_paths.py:241` — witnesses without a `_text_encoder_configs` dict; 13 LLM witnesses + 1 diffusion witness **(inferred)**), 2 strict xfails = the two admitted holes `test_authority_probes.py:202` (decorator blesses an unconditional structural return) and `:219` (`StructuralWrite.key` still `(sink, target)`, `structural_writes.py:87-88`). 29 skip/xfail sites in source, most in `test_conformance.py` guarding uninstalled sources — all those sources are installed here (paligemma/qwen3_5/mixtral/mistral3 present), so they do not fire today.
- **Poisons vs positives**: 80 collected node ids contain "poison"; 54 `def test_*poison*`; 31 "NEGATIVE CONTROL / anti-vacuous" docstrings; a crude negative-wording regex over node names matches 1776 / 3991 (~45 %) — the suite is roughly half counterexamples. 429 node ids are parametrized over a real corpus slug across 36 files; 68 test files touch the real corpus.

---

## 4. Corpus adequacy

Ecosystem installed: transformers 5.12.1 (**466** `modeling_*.py` across 485 model dirs), diffusers 0.38.0 (40 `transformer_*.py`, 14 `unet_*.py`). Target catalogue: `previews/old/toserve_model.md` 165 ids / 34 families (LLM); `toserve.md` 111 ids incl. diffusion. The corpus is 13 LLM + 16 diffusion witnesses = **6 % of transformers modeling files, ~10 of 34 catalogue LLM families**.

Mechanism facts below are read from `tests/preservation_baseline/<slug>/ir.json` (`layers[].attention.kind/position_kind/qk_norm/window_size/mixer_state`, `ffn.kind/num_experts`, `norm_kind/placement`) and the view labels in the expected manifest.

Per-witness summary (L = layer count in the IR; "views" = distinct locked SVGs):

| witness | kind | L | attention | position | FFN | norm | distinctive mechanisms | views |
|---|---|---|---|---|---|---|---|---|
| llama-7b | LLM | 32 | MHA | RoPE | gated SiLU | RMS pre | canonical baseline | 4 |
| stablelm-2-1-6b | LLM | 24 | MHA | RoPE | gated SiLU | LayerNorm pre | LayerNorm+RoPE | 4 |
| olmo-2-1124-7b | LLM | 32 | MHA, QK-norm | RoPE | gated SiLU | RMS **post** | post-norm | 4 |
| bloom | LLM | 70 | MHA | **ALiBi** | dense GELU | LayerNorm pre | ALiBi, tied | 4 |
| qwen3-8b | LLM | 36 | GQA, QK-norm | RoPE | gated SiLU | RMS pre | — | 4 |
| granite-3-0-8b | LLM | 40 | GQA | RoPE | gated SiLU | RMS pre | multipliers (2 advisory findings) | 4 |
| gemma-2-2b-it | LLM | 26 | GQA, window 4096 | RoPE | gated gelu_tanh | RMS **double** | softcap, sandwich norm, tied | 6 |
| gpt-oss-20b | LLM | 24 | GQA, window 128, sinks | RoPE | MoE 32 | RMS pre | sinks + SWA + MoE | 8 |
| dbrx-base | LLM | 40 | GQA | RoPE | MoE 16 | unknown | MoE without shared expert | 6 |
| glm-4-5 | LLM | 92 | GQA, QK-norm | RoPE | dense + MoE 160 + shared | RMS pre | dense-first + shared expert | 9 |
| deepseek-v3 | LLM | 61 | **MLA** | RoPE | dense + MoE 256 + shared | RMS pre | MLA q/kv paths, group top-k | 11 |
| qwen3-5-27b-text | LLM | 64 | GQA + **gated_delta** | unknown | gated SiLU | RMS pre | hybrid mixer schedule | 6 |
| qwen2-vl-7b-instruct | VLM | 28 | GQA | unknown | gated SiLU | RMS pre | vision/video paths, fusion, projector | 11 |
| musicgen-small | audio | 24 | MHA + cross-attn | unknown | GELU | LayerNorm | enc-dec, codebooks, conditioning projector | 8 |
| fluxtransformer2dmodel | DiT | 57 | — | unknown | dense GELU | LayerNorm/unknown | dual+single stream, 2 encoders | 20 |
| flux-2-dev | DiT | 56 | — | unknown | — | LayerNorm | dual+single, 1 encoder | 16 |
| stable-diffusion-3-5-large | DiT | 0† | — | — | — | — | MMDiT, 3 encoders | 19 |
| auraflow-v0-3 | DiT | 36 | — | unknown | dense SiLU gated | unknown | MMDiT + single | 15 |
| lumina-image-2-0 | DiT | 30 | — | unknown | — | unknown | single-encoder MMDiT | 16 |
| qwen-image | DiT | 60 | — | unknown | — | LayerNorm | MMDiT | 13 |
| pixart-sigma-xl-2 | DiT | 0† | — | — | — | — | cross-attn DiT | 13 |
| sana-1600m | DiT | 20 | — | unknown | — | LayerNorm | **linear attention, conv FFN** | 12 |
| prxpixel-t2i | DiT | 24 | cross-attn | unknown | — | LayerNorm | cross-attn, no VAE view | 9 |
| hunyuanvideo | video DiT | 60 | — | unknown | dense GELU | LayerNorm/unknown | dual+single, 2 encoders, 3-D | 20 |
| cogvideox-5b | video DiT | 42 | — | unknown | — | unknown | 3-D RoPE | 15 |
| wan2-2-t2v-a14b | video DiT | 40 | — | unknown | — | unknown | cross-attn video | 12 |
| mochi-1-preview | video DiT | 48 | — | unknown | — | unknown | asymmetric, no attn/ffn drill locked | 9 |
| ltx-video | video DiT | 28 | — | unknown | — | unknown | cross-attn video | 12 |
| stable-diffusion-xl-base-1-0 | UNet | 0† | — | — | — | — | down/mid/up, resnet, crossattn, 2 encoders (11 advisory findings) | 29 |

† these witnesses carry their structure under `extras.render`/`extras.unet`, not `layers`; the `layers`-based nets 13, 15 and the numbers-only census see nothing there. Note how many diffusion witnesses record `position: unknown` and `norm: unknown` on the layer spec: the pale-honest policy makes these green, and nothing bounds how many facts a witness may leave unknown (§6.3).

| mechanism family | witnesses | count |
|---|---|---|
| GQA decoder | qwen3-8b, gemma-2, glm-4.5, gpt-oss, granite, dbrx, qwen2-vl, qwen3.5 | 8 |
| MHA decoder | llama-7b, bloom, stablelm, olmo-2, musicgen | 5 |
| RoPE (LLM) | llama, stablelm, olmo2, qwen3, gemma2, glm, gpt-oss, granite, dbrx, dsv3 | 10 |
| MoE FFN | dbrx (16), deepseek-v3 (256), glm-4.5 (160), gpt-oss (32) | 4 |
| shared expert / dense-first-layers | deepseek-v3, glm-4.5 | 2 |
| QK-norm | olmo-2, qwen3-8b, glm-4.5, qwen3.5 | 4 |
| sliding window | gemma-2 (4096), gpt-oss (128) | 2 |
| LayerNorm decoder | bloom, stablelm, musicgen | 3 |
| tied embeddings (LLM) | bloom, gemma-2, granite | 3 |
| MMDiT dual-stream | flux, flux-2, sd3.5, auraflow, hunyuanvideo, qwen-image, lumina-2 | 7 |
| single-stream DiT block | flux, flux-2, auraflow, hunyuanvideo | 4 |
| cross-attention DiT | pixart-Σ, sana, ltx-video, prx, wan2.2 | 5 |
| video (3-D) DiT | cogvideox, hunyuanvideo, wan2.2, mochi, ltx | 5 |
| pipeline text encoders (CLIP/T5/…) | 15 of 16 diffusion witnesses (prx has one) | 15 |
| VAE decoder drill | 13 diffusion witnesses | 13 |
| MLA attention | deepseek-v3 | **1** |
| ALiBi | bloom | **1** |
| post-norm | olmo-2 | **1** |
| sandwich/double norm + softcap | gemma-2 | **1** |
| attention sinks | gpt-oss (`ir.json` `"sinks": true`) | **1** |
| gated-delta hybrid mixer | qwen3.5-27b | **1** |
| vision tower / VLM fusion / mrope | qwen2-vl | **1** |
| encoder-decoder + cross-attn LLM + audio codebooks | musicgen | **1** |
| linear attention + conv FFN | sana | **1** |
| UNet (down/mid/up, resnet, crossattn) | sdxl | **1** |
| MQA, NoPE-layer LLM (Llama-4 iRoPE), learned-absolute LLM (GPT-2/OPT), parallel-residual (Falcon/GPT-J), T5/relative-bias standalone, encoder-only (BERT), MTP, DSA sparse indexer, self-conditioning, per-layer embedding (Gemma-3n), audio DiT, Jamba/RecurrentGemma hybrids, RWKV, Phi, Command-R, Nemotron, Kimi, Mixtral-style MoE without shared expert other than dbrx/gpt-oss | — | **0** (MTP/DSA/self-cond/PLE/audio-DiT exist only as synthetic `test_support.CORPUS` entries or unit controls, `test_coverage.py:48-54`) |

Near-duplicates by mechanism (differ only in norm type / multipliers / sizes): `llama-7b` ≈ `stablelm-2` ≈ `granite-3.0` ≈ `qwen3-8b` (dense SwiGLU pre-norm decoders); `flux` ≈ `flux-2` ≈ `hunyuanvideo` (dual+single MMDiT; hunyuan adds video); `lumina-2` ≈ `qwen-image` at the view-label level **(inferred from view labels, not from block internals)**.

Consequence: every one-witness family is locked by exactly one frozen render, so a regression that happens to reproduce that render (or a parser change that only affects the *second* model of a family) has no witness; and the whole recall side of §2 is exercised on one example for eleven families.

---

## 5. Known holes, verified from the code

1. **Hash-set signature with no labels.** `SableReport.hash_signature()` = `sorted(h for _, h in self.view_hashes)` (`sable.py:93-95`); the fixture stores only that list (`:770`); `check_regression` compares list equality (`:831-834`). Two views exchanging content, or a view's label changing, is invisible to the sable lock. The preservation expected manifest is NOT label-blind (`preservation.py:259-273, 367-372`), so this hole is closed for the 29 witnesses by the stronger gate, but `bless`/`check_regression` alone remain label-blind.
2. **`visual_review` is self-set and unpersisted.** It is a plain string field defaulting to `"PENDING"` (`sable.py:77`); the only writers in the tree are tests (`tests/test_sable.py:469, 545`); `bless` checks the string equals `"CLEAN"` plus artifact existence/count (`:678-706`), then writes a fixture containing `model, source, config, hash_signature, checks, visual_evidence` (`:762-777`) — no reviewer, date, per-view verdict, or rubric outcome. The durable record is the optional `her_eyes_review.md` sidecar, present for 8/29 witnesses.
3. **Rubric is layout-only, 7 of 8 items.** `VISUAL_RUBRIC` (`sable.py:44-53`): items 1–7 are line-through-block, arrowhead collision, overlap/clip, chip collision, duplicate labels, pale box, ambiguous arrow; only item 8 ("reads as the RIGHT mental model") is semantic. Nothing machine-checks any of the eight.
4. **Advisory nets.** `asserted_facts` (`:514-518`) and `config_accessed_unprojected` (`:551-555`) carry `blocking=False`; `check_regression` skips non-blocking nets (`:828-829`); 13 live `config_accessed_unprojected` findings sit in the blessed baselines (granite 2, SDXL 11).
5. **`check_regression` never reports a lost oracle.** It reads `rep.checks` and `rep.hash_signature()` only (`:825-835`); `rep.oracle` is never consulted. With sources uninstalled, nets 7–10 return `[]` (`:446-495`) and the corpus lock passes if the hashes still match. Only `test_loud_miss.py:80-88` (llama + one diffusers bundle) and `test_sable.py:449-454` (FLUX/PIXART/LLAMA) assert oracle presence.
6. **Preservation ignores fixture `source`.** `canonical_surfaces` calls `mu.unfold(cfg)` and `sable(cfg, render_images=False)` with no `source` (`preservation.py:91, 94`); `_view_hashes` likewise (`:271`). `check_regression` does honor `fixture.get("source")` (`sable.py:825`). Harmless today because all 29 fixtures say `"source": "local"`, latent otherwise.
7. **`debt_problems` never checks unit liveness.** The six gates (`structural_debt.py:1250-1273`) are duplicates, dead writer, dead consumer, satisfied deletion condition, unrowed extras write, unrowed consumer read. `MIGRATION_UNITS = U3…U15` (`:41`) is a spelling check on construction only. Rows assigned to U5 (2), U9 (19) and U11 (37) persist although memory records those units as DONE; nothing fails when a "done" unit still owns rows.
8. **Blanket closure excusal.** `_DIFFUSION_SOURCE_CLOSURE_PATHS` (69 bare leaf names, `:762-784`) and `_VAE_SOURCE_CLOSURE_PATHS` (25, `:786-807`) are expanded into `config_read` rows for `root.denoiser`/U15 and `root.vae`/U12 with `deletion_condition = "classified:<path>"` (`:809-825`). `pending_classification_paths` (`:971-977`) feeds Sable #23 (`sable.py:314`), so an accessed-but-unconsumed `hidden_size`, `num_layers`, `eps`, `patch_size`, … on ANY diffusion model is excused by rows that were never written for that model. 94 of the 115 `classified:` rows are these.
9. **Receipt directories are not retained.** `LOG_ROOT = /private/tmp/model-unfolder-verification` (`verify_commit.py:42`); nothing copies `receipt.json` into the repo. Evidence: `5564c3881a/` (commit `7ade5cf`, 2026-08-28) now holds only `receipt.json` — the six `.log` files it cites are gone; `docs/EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:1783-2279` cites runs `598777dc06, 6f07ace2b7, 1e6b8b09e8, a5b5caf788, c0604f8ef8, f9a358eae1`, none of which exist in the directory listing.
10. **`_role_of` identity heuristic inside the oracle.** `forward_ops._role_of` classifies a constructed field's class by case-insensitive SUBSTRING (`forward_ops.py:396-412`) from `everchanging/conformance/type_roles.yaml` (`attention: [Attention, Attn, SelfAttn, CrossAttn, DeltaNet]`, `ffn: [FeedForward, MoE, Experts, FFN, MLP, …]`). `conformance.py` uses it at 15 sites including the MLA kind cross-check (`:563-575` — "Structural fields, never a class name", but the *role* that selects which class gets `_has_mla` IS the class name). `identity_guard` registers these tables as lawful `code_shape` with `_CONFORMANCE_READERS` (`identity_guard.py:143-146, 173-187`), so the net that forbids identity-derived structure in the parser exempts identity-derived structure in the oracle that checks the parser.
11. (additional) **`zero_asserted_census` swallows every exception** (`sable.py:289-292`): a crash in the numbers-only parse is indistinguishable from "nothing asserted".
12. (additional) **Synthetic-only coverage.** `test_coverage.py` runs over the 17 synthetic configs, never the 29 real witnesses; the view-exercised set is therefore proven for hand-written dicts, not for any shipped model.
13. (additional) **`census.py --check` is ungated** (§1.3); `docs/COR5_NET1_MIGRATION_DEBT.md` can drift from the live corpus without any test noticing.

---

## 6. What a complete verification system for this goal would contain

The goal (README): render the *true, complete* structure, never guessing, never silently omitting. Missing checks, one sentence each:

1. **An independent oracle**: instantiate each witness on the `meta` device and diff the IR against the real `named_modules()` tree and a `torch.fx`/hook trace of `forward()` — today every code-facing net re-reads the same AST the parser read.
2. **Golden-fact recall**: a hand-curated per-witness list of architectural facts (from the paper/config) that the render MUST state, asserted as a subset of what is drawn — the only way to catch an omission the AST reader also missed.
3. **Unknown-rate ratchet**: per witness, the count of `unknown`/pale facts must not grow (recall today is unbounded — "honest unknown" is always green).
4. **Topology/order conformance**: compare edge sets and op ORDER of each drill against the traced forward, not op-kind presence sets.
5. **Shape/arithmetic conformance**: `heads × head_dim`, expert widths, KV dims and `estimate_params` checked against `model.safetensors.index.json` tensor shapes.
6. **Every layer variant, not one representative**: `op_conformance` on all distinct `classify_group` members and on every heterogeneous layer (representative dict at `conformance.py:141-143`).
7. **Labeled view signature**: lock `(label, hash)` in the sable fixture and diff labels in `check_regression`.
8. **Persisted visual verdicts**: `bless` writes reviewer, timestamp and per-view rubric outcomes into the fixture; a fixture without them is not blessable.
9. **Machine layout nets**: SVG bounding-box overlap/collision/clip/line-through-box detection for rubric items 1–4, run per view.
10. **Oracle assertion in the lock**: `check_regression` fails when `rep.oracle != "present"`.
11. **Preservation honors `fixture["source"]`** and records the resolved source file hashes so a `transformers` upgrade is a visible drift, not a silent re-baseline.
12. **Debt-register liveness**: a row whose `migration_unit` is marked complete anywhere fails; per-model closure rows replace the 94 blanket rows.
13. **Two witnesses per mechanism family** enforced by a corpus census test; the 11 single-witness and ~15 zero-witness families in §4 become a tracked list.
14. **Cached-config ecosystem sweep in CI**: `coverage_audit`/`dit_coverage`/`serve_audit` against committed config snapshots (no network) so "supported set" is a gated number, not a July doc.
15. **Skip budget**: `-rs` with an asserted maximum (14 today, silent).
16. **Property/metamorphic perturbations beyond rename**: change `num_key_value_heads`, `sliding_window`, `num_experts` in a witness config and assert the diagram changes in the expected place.
17. **Receipts in-repo**: a `receipts/<commit>.json` ledger written by `verify_commit.py`.
18. **Performance ratchet**: per-witness render time locked (qwen-image at 435 s per regeneration is the long pole of every receipt).

---

## 7. Loose ends

- Per-file durations for the `full` lane are not recoverable; the bracket runs 8-file batches without `--durations`. The "ten slowest files" above are batch-level.
- The 14 skips are attributed to `test_config_paths.py:241` by elimination (only per-witness skip in batch-001); not confirmed by an `-rs` run.
- `_role_of` use count (15) is from grep of `conformance.py`; I did not trace which of the 15 sites can change a *verdict* versus merely select a class to inspect.
- Whether `gpt-oss` is the only sinks witness and whether `qwen2-vl`'s `pos: unknown` is the intended mrope outcome or a recall gap belongs to pass 03; I only read the IR.
- I did not run `check_regression` or any Sable net; verdict-level claims (e.g. "13 live advisory findings") come from the committed baseline `sable.json` files.
- `flux` vs `flux-2` vs `hunyuanvideo` "near-duplicate" is a view-label judgement, not a block-internal diff.
- The 2 strict xfails are marked as *holes* by their authors; whether p10 is actually closed by the multiset key (`StructuralWriteKey`, `structural_writes.py:217`) while `StructuralWrite.key` remains `(sink, target)` is a naming question pass 08 should settle.
- `docs/dit_coverage.md` does not exist; `dit_coverage.py` prints to stdout only — whether a DiT sweep has ever been recorded is unknown from the tree.
