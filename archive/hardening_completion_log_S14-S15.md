> **ARCHIVED 2026-07-12.** This file is the sole copy of the Evidence Hardening
> plan's **§14 (Completion log)** and **§15 (Judgment handoff)**. §16 (the
> authoritative recovery plan in `docs/EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md`)
> supersedes §14/§15 *without deleting their historical record*, so it was moved
> out of `docs/` (which now holds only the current plan + reference docs) rather
> than deleted. Its inline references to `docs/H0_BASELINE.md`,
> `H0_RESPONSIBILITY_CENSUS.md`, `H3_PLAN.md`, `H3_SCOPE.md` are to per-unit
> scratch docs that were removed in the same cleanup — they are historical
> mentions, not live links. Current per-unit status lives in the plan's §6
> tracker; current exit criteria are the poison-control tests themselves.

---

# 14. Completion log (implementer's running record)

*Appended per Soumil's instruction: for each unit, what was ENCOUNTERED, what was
DONE, and what it TURNED OUT to be. Narrative depth behind the §6 tracker. Newest
work appended at the bottom. Nothing here is committed — Soumil commits and blesses.*

## H0 — Baseline and unsafe-path quarantine — DONE (2026-07-12)

**Encountered.** The tree at `ef9046c`+`7aeb815` was clean (no uncommitted work —
the plan was committed together with the U6 diffusion changes it governs, so there
was nothing to preserve *against* the plan). The forbidden U6 tables
(`unet_blocks.yaml`, `vae_classes.yaml`, `schedulers.yaml`) were already absent —
the concurrent correction had removed them. But the identity guard was still the
*enumerated* guard: `_ARCHITECTURAL_FACT_TABLES` knew table NAMES, so a new
class-keyed structural table under a fresh name (the exact U6 bypass class) passed
falsely green. Three isolated audit files were claimed to pass alone; I verified
rather than assumed (all four did: projection-audit 20, identity-guard 8,
fact-ledger 21, layer-schedules 15).

**Did.** (1) Froze metrics in `docs/H0_BASELINE.md`: full suite **628 passed**,
25 corpus fixtures, identity debt 0, **69** broad `except Exception` catches
(8 evidence / 54 adapters / 7 parser+renderers), **641** asserted ledger records
(543 DiT `attention_kind` + 94 `ffn_storage` + 4 `projection_mode`), projection
coverage {attention, ffn, layer, model}, and perf timings. (2) Wrote the
O/B/E/S/P/X responsibility census `docs/H0_RESPONSIBILITY_CENSUS.md` over the
high-risk entry points — spawned three agents for the transformer/evidence/diffusion
thirds; the diffusion agent died on a transient API error, so I read all 4,481 lines
of the three diffusion files myself and authored Section 3. (3) Hardened the guard's
no-growth gate: class-keyed tables are now detected by SHAPE (plain list, pair list,
dict map, file-root map in YAML; Python dict literal) under ANY name, plus a display
category pinned exactly, plus the PyYAML-free fallback parser extended so the nets
don't weaken without the optional dep.

**Turned out.** During the required self-audit (against G-1…G-15) I found a real
defect **in my own gate**: applying G-7 (negative proof requires complete
inspection), the "debt is zero" walk only covered `adapters/renderers/evidence` —
the package ROOT modules (`parser.py`, `opgraph.py`, `submodel.py`, `encoder_panel.py`)
were unscanned, and `opgraph.py` is the census's own conflation point C-8. A
class-keyed literal there would have escaped. I probed all 15 root modules first
(all clean, including `sable.py`), then widened the walk with zero exemptions and
added a poisoned-tree e2e control planting a literal in BOTH an adapters file and a
root module. Debt stayed 0 / display 2 / declared 18 (proof the widening is coverage,
not reclassification). Final full suite **636 passed** (628 + the 3 new shape-net /
poisoned-tree controls + walk-widening tests) in 40:06. Known residuals stated
honestly for H4: dict-comprehension tables, helper-hidden identity, consumer-side
taint, and single-capital class names (`_looks_like_class_name` needs ≥2 capitals).

## H1 — Evidence primitives and negative-proof law — implemented, gating (2026-07-12)

**Encountered.** `FactLedger`/`FactRecord` (context.py) already carried
value/status/source but no completeness, no source spans, no premises, no typed
failures — and nothing enforced the epistemic laws the doctrine documents. The laws
lived in prose and review practice (root cause R-4: the IR represented values, not
epistemic state).

**Did.** Wrote `evidence/facts.py` with `EvidenceFact` (owner path, value, status,
completeness, source_spans, config_paths, premises, reason, failure_kind) plus
`RawObservation`/`BoundObservation` as DISTINCT types. Made the laws
CONSTRUCTOR ERRORS, not documentation: a `code_proven`/`code_and_config` negative
(`False`/`None`/`"none"`) without `complete` inspection raises `NegativeProofError`
(I-3); a derived fact must name premises and a non-derived fact may not carry any
(I-5); a failure kind may only ride ambiguous/oracle-missing/unknown (I-9); passing
an observation where a fact is required is a `TypeError` (§4A.1 boundary); and
`bool(fact)` raises so `value or default` physically cannot coerce a tri-state
through this layer (H1.5). Added `resolve_strength` (derived strength = weakest
premise, recursively; missing premise / cycle → 0). Extended `FactLedger` with
`record_typed` where the serialized legacy row is DERIVED FROM the typed fact — one
author, so the two views cannot diverge for typed keys.

**Turned out.** The exhaustive round-trip matrix (9 statuses × 8 value shapes × 4
source forms) failed on legacy **"derived" rows that record no premises** (the
eps-spelling norm tier writes these today). Rather than fabricate a premise or waive
the law for native facts, I marked lifted premiseless-derived rows with an explicit
`<legacy:unrecorded-premises>` sentinel — mechanically countable H2 debt that
resolves to strength 0, never an invented rank. Isolated `test_evidence_facts.py`
**325 passed** including a real llama parse whose entire ledger lifts and
round-trips exactly; targeted fact-ledger + projection-audit **41 passed** against
the extended ledger. Behavior-neutral by construction (no production path
constructs an `EvidenceFact` yet); the full-suite behavior-neutrality gate is the
one open confirmation.

## H2 — Closed fact registry and legacy census — implemented, gating (2026-07-12)

**Encountered.** Projection-audit exempted new domains via an open whitelist; there
was no closed-world contract, and the asserted debt (641 records) was measured but
not pinned. I ran a corpus-wide inventory probe first (H2 step 1) rather than guess
the population — it returned exactly **11** ledger fact names with their observed
statuses/owners/value-types.

**Did.** Wrote `evidence/registry.py`: `FactDefinition` (value_types,
allowed_statuses, owner_patterns, projections, unknown_policy,
negative_requires_complete, parameter_consumer, conformance) with self-gates that
raise at import (duplicate key, unknown status/surface/policy, drawable-without-
unknown-policy, parameter-consumer-without-unknown-policy). Populated `REGISTRY`
from the MEASURED inventory — allowed sets are exactly the observed reality, so any
future tier a new model produces is a reviewed widening, never a silent acceptance.
Added a `legacy_convention` unknown-policy spelling to pin the opgraph compatibility
leaks (rope=True, kind-None→softmax-MHA, scores-scaled-when-silent, split-when-
unproven) HONESTLY instead of laundering them as lawful policy — the census counts
them like `legacy_asserted` debt, and H8's cutovers replace every row carrying it.

**Turned out.** Two truths the registry had to state where they live: (a)
`attention_kind`/`ffn_storage` records carry `NoneType` VALUES — the asserted-fold
writes `getattr(spec, fact, None)` for non-field tag names (census §0.7), so the
registry pins that, and H2's fold-unification makes them real values; (b) six DRAWN
leaf names (position_kind, qk_norm, q_norm, k_norm, sinks, logit_softcap) never
appear in any ledger — drawn-but-unledgered facts that get definitions when H8 gives
them writers. Corpus census `test_fact_registry.py` **12 passed** (3:15):
bidirectional closed world (unregistered fact/owner/status/value-type fails;
registered-but-never-observed fails), asserted pins EXACT at 543/94/4 = 641, and a
mechanism-name lint against the corpus's own declared model types (a family-keyed
registration cannot slip in). SCOPE HONESTY: H2 closes the LEDGER boundary; the
large unledgered raw-write population (census raw-write inventories) has no ledger
identity yet, so it is the H7/H8 migration surface, documented — not falsely
claimed closed.

**Amendment honored (mid-course plan edit).** After the initial H2 landing, the
plan gained an explicit clarification (§7-H2: "may not be a vacuous scaffold… a
poisoned unregistered fact/owner/status must make the corpus census fail. A test
module that validates only FactDefinition constructor errors does not satisfy
H2"). My H2 had the corpus positive-check and constructor-error tests but no
NEGATIVE control proving the census catches an injection. Closed it without
reinterpretation: extracted the checker into `registry.census_problems(rows)` (the
single function BOTH the corpus census and the poison controls consume) and added
four poison controls (unregistered fact / owner / status / value-type each fails)
plus a clean-row positive — 12 passed in 0.07s, synthetic so no corpus parse. The
corpus census now calls the same shared checker, so "corpus is clean" and "poison
fails" exercise identical logic.

## H4 — Semantic identity and taint guard (early slice) — IN PROGRESS (2026-07-12)

**Plan-order note.** §11 places the H4 *early slice* immediately after H1+H2 and
BEFORE H3 ("so U6 cannot add another bypass while the full taint system is built").
Following §11: H4 early slice now, H3 after. The full dataflow/taint net (item 2)
is explicitly the "full taint system built later"; the early slice is the
generalized negative controls (item 6) + the fact-provenance rule (item 1).

**Encountered.** H0's shape net closed the class-keyed-table growth vector but had
no coverage of item 1 (fact provenance) — a fact could still be *decided* by a name
as long as the deciding table wasn't itself class-keyed. And H1/H2 had just made
item 1 checkable for the first time: H1 gives every fact a typed `source`/`reason`,
H2 gives the registry, so "no structural fact cites identity as its deciding source"
became a mechanical invariant instead of a review sentiment. The subtlety to get
right: the DECLARED tiers legitimately read identity — the causal-LM role reads
`architectures[]`, the class-default hydration channel reads `model_type`, and the
pinned declared-component tables read `_class_name`. A naive "no fact mentions
identity" net would false-positive on all of those lawful config-declared reads.

**Did.** Added `scan_fact_provenance_identity(provenance)` to `identity_guard.py`:
a pure function over a serialized `fact_provenance` dict that flags only
`code_proven`/`code_and_config`/`derived` facts whose `source` cites an identity
field or a declared class-vocabulary/display table — the `config_declared`/
`class_default` tiers are exempt by construction. Matching is by EXACT provenance
SEGMENT (split on `: . / space`), so a reader named `block_architecture_reader`
never trips on the `architectures` field and `config:model_type` does. Added
`tests/test_h4_taint.py`: 11 synthetic controls for item 1 (identity-cited code
facts flagged across all three code-evidence statuses and the class tables; declared
tiers lawful; reader names and the generic word "architecture" no-false-positive;
non-evidence statuses ignored) plus the generalized item-6 negative controls
(intermediate-enum table caught; helper-hidden literal caught; domain-marker
generator caught) — and one EXPLICIT open-boundary control pinning the single
evasion the early slice does NOT close (a non-domain substring like `"SimpleCrossAttn"
in cls` flowing to structure), so the day the full dataflow net lands, that test
flips from "not caught" to "caught" and the boundary can never be silently
forgotten. Added the corpus invariant test (parse every fixture → assert no fact
cites identity) — deferred to run when the H1/H2 gate frees the machine.

**Turned out (so far).** Synthetic slice **11 passed in 0.04s** (static, no
parsing — no CPU competition with the running gate). Open before DONE: the corpus
fact-provenance invariant run (a hit would be a genuine identity leak → Problem
Report, not a relaxation), wiring the net BLOCKING into `sable` once the corpus is
proven clean, and the H4 full-suite gate. Deliberately deferred to the H4 FULL unit
(not the early slice): the dataflow/static taint net (item 2), YAML value inspection
(item 3 completion — the early slice inspects keys), and the renderer-firewall
dependency rule (item 4).

**Corpus invariant — turned out CLEAN.** Ran the corpus fact-provenance invariant
once the H1/H2 gate freed the machine (`test_h4_taint.py`, 354-test batch, 3:58):
**no fact in any of the 25 blessed fixtures cites identity as its deciding source.**
The invariant is real and satisfied — a meaningful "nothing to report" (had it
found a hit, that was a §13.4 Problem Report, never a relaxation). The net is
blocking via its place in the suite; wiring it into the `sable` report object is
deferred to the full H4.

## H3 — Config ownership and consumption — Phase A DONE, B–F pending (2026-07-12)

**Plan-order + amendment.** §11 makes H3 follow the H4 early slice. Wrote the deep
execution plan first (`docs/H3_PLAN.md`): the six §7-H3 items mapped to six
shadow→cutover→delete phases, a concrete 24-site `_note_fact` worklist mapped from
the tree, and — carrying forward the H2 anti-vacuous amendment as standing policy —
EVERY net H3 adds ships with a poison control proving it fires.

**Encountered (Phase A).** The config-access rail (`debug.py`) had ONE binary
distinction (touched vs the `consumed` set that supported `intent="consumed"` but
had ZERO production callers — so `_consumed` was always empty, `config_consumed`
never published, and the accessed-but-unconsumed sable net inert by construction).
The plan wants five mandatory intents. Design realization: only three are
rail-marked (inspected/bound/consumed) — `projected` is derived at IR assembly by
joining `consumed` with the #13 receipts, and `ignored` comes from the scoped
ignore rules; marking them in the hot accessor would be wrong.

**Did (Phase A).** Extended `note_access(name, intent)` to the three-value rail
enum with `_bound`/`_consumed` sets, `_touched` kept as the union (so the unread
diagnostic is byte-unchanged), a LOUD `ValueError` on an unknown intent (a typo
must never silently degrade to inspected), and `bound_fields()`/`consumed_fields()`
accessors; `reset()` clears all three. Added `tests/test_config_intents.py` (7):
the three intents route distinctly, reset clears, unknown intent raises, capture
includes intent-marked accesses, and the anti-vacuous control — a field READ but
never CONSUMED (the `embedding_multiplier`/`logits_scaling` granite class) is
VISIBLE in accessed−consumed, the exact signal the inert net will read.

**Turned out (Phase A).** 7 passed in 0.06s; behavior-neutral SHADOW (the new sets
are recorded but nothing reads them for output yet). Full-suite confirmation folds
into the next H3 gate.

**Encountered (Phase B).** First designed `consumed` as a DERIVATION from the
ledger's `source` fields — then caught it was insufficient by reading the actual
net (`_accessed_unprojected_findings`, sable.py:301, advisory `blocking=False`): it
computes `accessed − consumed`, and the GEOMETRY fields flow into the spec via the
`get()` closure with NO `_note_fact`, so ledger-derivation would leave every
geometry field falsely unconsumed — a net full of false positives. A second, deeper
issue: `_consumed` is a module global, and a NESTED component parse (qwen2-vl vision
tower, diffusion text encoders) calls `reset()`, which would clobber the root's
consumed set — the exact bug the `capture_accesses` ContextVar already fixes for
*accessed*.

**Did (Phase B, geometry+embedding family).** Introduced a `consume(field,
default)` accessor (`get` + `note_access(field, "consumed")`) and converted the
spec-flow reads to it — `num_hidden_layers`, `hidden_size`, `num_attention_heads`,
`num_key_value_heads`, `head_dim`, `intermediate_size`, `mlp_ratio`, `vocab_size`,
`max_position_embeddings` — leaving `get()` as `inspected` for branch tests, so the
intent is encoded by WHICH accessor a site uses. Made the capture nesting-safe: each
capture is now a `(touched, consumed)` pair that survives a nested `reset()`
(module globals reset; capture sets keep accumulating), and `config_to_ir` reads
consumed from the capture, not the clobbered module global. Added `config_consumed`
to the name-blind `_normalized_structure` drop list (it is diagnostic, not
architecture — a scrubbed-vs-original consumed asymmetry must not trip the
name-blind net). Extended the rail tests to 8 incl. a nesting-safe control proving
the capture keeps both fields while the module global loses the pre-reset one.

**Turned out (Phase B increment).** 8 rail tests green (0.02s); parser imports
clean. This is one coherent family + the hard capture infrastructure; the remaining
consume() families (activation and the other ledgered-fact deciding fields) are
mechanical follow-ons. End-to-end confirmation (a real parse publishes
`config_consumed` with the 9 fields; name-blind stays green; the advisory net wakes
to WARN on the still-unconverted fields — expected shadow noise until Phase F
cleanup) folds into the next full gate. The blocking flip stays Phase F.

## Plan re-sync + reconciliation (2026-07-12, on Soumil's instruction to obey the MD)

**Encountered.** Asked to verify my status matches the plan as the sole source of
truth, I re-read the CURRENT plan and found it had been edited since my initial
full read — three binding changes I was working without:
1. **§8.1 (new): a mandatory pre-change renderer-preservation gate P-1…P-6** that
   runs BEFORE editing production code (freeze baseline → blast-radius graph →
   witness matrix → delta ledger → increasing-radius no-break sequence → two-key
   release). I had been doing lightweight versions.
2. **H1 item-1 amendment:** a legacy `code_proven=False/None` lift must NOT
   manufacture `completeness="complete"`. My `from_record` did exactly that.
3. **H0 item-5 amendment:** quarantine the config-only `block_configs` NAS
   projection.

And a worktree fact: **Soumil is editing the tree concurrently.** HEAD is still
`7aeb815`, but six files I never touched (`params.py`, `blocks/layers.py`, `ir.py`,
`assembly.py`, `metadata.py`, `test_layer_schedules.py`) carry his uncommitted
edits — the COMPLETE H0 NAS quarantine (removing `has_attention/has_ffn` pruning
from params, the NAS sublayer logic from layers/ir/assembly, and adding the
`test_block_configs_nas_projection_is_quarantined_until_source_bound` pin).

**Did.** (1) Verified I had not clobbered his work — my edits and his six files do
not overlap; `parser.py` holds his block_configs removal AND my `consume()` in
different regions; his quarantine test passes (1). (2) **Fixed the H1 violation**
exactly per the amendment: the lift records `uninspected` + a `migrated_legacy`
flag (exempt from the native negative-proof law, counted by `migrated_legacy_debt`);
native code-negatives still must prove complete inspection; 326 facts tests green.
(3) Confirmed **§11 (the roadmap) is UNCHANGED** — my sequence
(H0→H1+H2→H4-slice→H3→H7…) matches it exactly, current position H3.

**Turned out.** The full A+B gate then caught **3 real Phase-B regressions** — the
exact preservation failures §8.1's P-2 blast-radius graph exists to catch BEFORE
editing, which I paid for by not running it: (a) my `capture_accesses` tuple change
broke a test caller in `test_sable.py` (I had grepped `model_unfolder/` for callers,
not `tests/`); (b/c) two committed tests pinned the pre-H3 "config_consumed not
published" state, which §11 step 4 explicitly authorizes H3 to change by activating
the net. Fixed all three FAITHFULLY (not by weakening): the caller unpacks the pair;
the inert-net test now exercises the no-census case synthetically while a NEW test
pins that real parses publish the census post-H3; the asserted-fold test asserts the
now-populated census. 43 affected tests green; full re-gate running.

**Standing correction adopted.** §8.1 is now binding for every remaining production
edit (H3 Phases C–F, H7, …): freeze baseline, enumerate EVERY consumer incl. tests
and the advisory nets, write the delta ledger, run the increasing-radius sequence
BEFORE the full gate — not after. The 3 regressions are the concrete lesson that the
pre-change gate is not optional ceremony.

## H3 Phase B — COMPLETE (spec-flow consumption), 2026-07-12

**Did (under a written §8.1 pre-change ledger — docs/H3_PLAN.md).** Extended the
`consume()` migration from the geometry/embedding family to all clear spec-flow
config reads: `hidden_act` (activation fact), `sliding_window` (window spec), the
four MoE-geometry fields (`num_experts`, `num_experts_per_tok`, `num_shared_experts`,
`moe_intermediate_size`), `tie_word_embeddings` (tie fact), `final_logit_softcapping`
(softcap extra) — 16 fields total. Every one is a MECHANISM config field; nothing
model-family-specific. The branch-test / discarded reads were DELIBERATELY left as
`get()`/`inspected` (verified: the MQA re-read `not get("num_key_value_heads")`
stays inspected), so a genuinely dropped read still surfaces in the advisory net —
that IS H3's signal, and consuming it would defeat the exit.

**Turned out.** Census activated and growing: llama `config_consumed` 8→15
(now incl. `hidden_act`, `tie_word_embeddings`); gpt-oss (MoE) consumes
`num_experts`. §8.1 P-5: 393 targeted census/rail tests green, config_consumed
growth probed on a dense + a MoE witness, full suite running as the final radius.
Plan-measure (§10 "config reads by intent") is now real — `consumed` populated,
`inspected` the honest default for discarded reads.

**Remaining to the H3 exit** ("discarded multipliers … cannot remain silent"),
each under §8.1: probe the residual `accessed − consumed` per corpus witness →
Phase D scope-ignore the non-architectural fields (by adapter/component/owner) →
allowlist the genuine read-never-drawn multipliers as pinned H3→H8 debt → Phase F
flip the advisory net to blocking. Phase C (`bound` reporting, I-2) and Phase E
(config_declared enum schema contract) harden the ownership binding alongside.

## Deep plan-vs-execution re-audit — H2 gap found (2026-07-12, on Soumil's instruction)

**Encountered.** Asked to compare the plan against actual execution deeply, I
checked my strongest claim (H2 DONE) against §7-H2.4's FOUR blocking gates and the
§6 DONE definition. Three gates are enforced (unknown fact key; drawable-without-
projection; parameter-consumer-without-unknown-policy). The FOURTH — **"new legacy
structural write"** — is NOT: a new unledgered `extras[...]=`/spec-field raw
structural write (the population the census documents: `parallel_residual`,
`qk_norm`, `sinks`, `mlp_bias`, the diffusion extras) passes every test. Confirmed
concretely: I could add `extras["fabricated_thing"] = True` to the parser today and
nothing would fail. And `legacy_asserted` (H2 item 3, the "count trends to zero"
measure) is never actually PRODUCED — the ledger uses `"asserted"`, so the typed
`legacy_asserted` status has no baseline. H2's exit "debt may remain but cannot
GROW or hide" is therefore violated for the raw-write debt, and my tracker outcome
literally claimed "raw-author census blocking against growth" — an over-claim.

**Did.** Downgraded H2 DONE→ACTIVE honestly (the ledger closed-world + asserted pins
ARE done and correct; the raw-write growth gate is the gap). Building the missing
piece: a corpus-based **raw structural-write census** — pin the set of `extras`
keys + spec-field writes the blessed corpus produces (excluding the audit/ledger
infrastructure keys), so a NEW raw structural write fails until consciously
registered/pinned, with a poison control proving it fires (the anti-vacuous rule
now standing policy). This is the faithful "cannot grow or hide" for the raw-write
surface; the deeper migration of those raw writes INTO the registry stays H7/H8.

**Turned out — H2.4 census DONE.** `evidence/registry.py::new_raw_structural_extras`
+ 12-key `RAW_EXTRAS_BASELINE` pinned (diffusion/modalities/moe/mtp/…); a new key
fails, a stale key fails (monotonic-shrink), poison controls green; 21 registry
tests green. The H2 gap is closed pending the full-suite fold.

## H3 — the advisory-net phantom-alias unlock (2026-07-12)

**Encountered.** Probing `accessed − consumed − ignored` for the blocking-flip
prerequisite showed a residual of **231 fields even for llama** — almost all
PHANTOM alias spellings (`n_layer`, `n_embd`, `d_model`, and even
`embedding_multiplier`, which llama does not carry). Root cause = the census's
"`_resolve` marks ALL sibling aliases accessed" finding PLUS
`get_config_value` marking a field accessed even when absent. So the advisory net
the plan wants activated (§11 step 4) was drowning in phantoms — meaningless.
Also learned: the granite multipliers are NOT in the blessed corpus, so the corpus
residual is redundant-aliases + genuinely-unconsumed spec-flow, all cleanable.

**Did.** Two surgical fixes so `accessed` = PRESENT fields only: (1)
`common.get_config_value` records the access only when the key is present (an alias
probe that misses is not a read); (2) transformer `_resolve` marks only the PRESENT
sibling spellings (the redundant-sibling intent is preserved; the absent phantoms
are dropped). `consume()` still marks regardless of presence (an absent field can
decide a fact). Residual collapsed: llama 231→7, bloom 231→4.

**Turned out — the flat-global-leak (H3.6) bit.** Because the config audit matches
accessed by key NAME across ALL scopes (the "flat global leak" H3.6 names), the old
phantom-marking was MASKING genuine unreads in nested configs. Removing it exposed
exactly 3: `_vae_config.act_fn` (SD/SDXL VAEs), `_vae_config.temporal_compression_ratio`
(HunyuanVideo), `max_sequence_length` (Mochi) — all genuine, all diffusion-scope.
Handled honestly (the diffusion adapter now reads its own VAE act/temporal fields
and Mochi's text-token limit — constructor records it should own), NOT by reverting
the correct fix. Corpus clean (sable regression + config_intents green). The
accessed-fix is a net correctness gain: phantom-masking gone, 3 real unreads owned.
Remaining to the blocking flip: consume the last transformer spec-flow fields +
mark present-sibling spellings consumed → residual ~0 for transformer models; the
diffusion/modality residual needs those adapters' consume() (H7/H8); THEN the
subtract-ignored blocking flip. The blocking flip is confirmed a LARGE, genuinely
deferred task spanning H3.6 + H7 + H8, exactly as §7-H3.5 ("after corpus cleanup")
anticipates.

---

# 15. Current state for judgment (handoff — Soumil decides moved parts + direction)

*Written 2026-07-12 at Soumil's request. Nothing is committed; HEAD is still
`7aeb815`. Two authors have uncommitted edits in this worktree — Soumil (the H0
NAS quarantine) and this agent (the evidence-hardening units). This section is
the map to judge what moved and to direct what happens next.*

## 15.1 Moved parts — every changed file, by author and unit

### Soumil's uncommitted edits (the H0 item-5 NAS quarantine)
| File | Δ | What |
|---|---|---|
| `adapters/transformer/blocks/layers.py` | −61 | removed NAS pruned-sublayer logic |
| `ir.py` | −13 | removed NAS `has_attention`/`has_ffn` spec fields |
| `params.py` | ~14 | removed NAS pruned-sublayer param accounting |
| `adapters/transformer/assembly.py` | −14 | removed NAS sublayer assembly |
| `renderers/html/metadata.py` | −4 | removed NAS metadata |
| `tests/test_layer_schedules.py` | +47 | added the block_configs-quarantine pin test |

Verified: no overlap with the agent's edits; quarantine test passes; 0 corpus
fixtures use `block_configs` (no pixel impact).

### Agent's uncommitted edits (evidence hardening, H0–H4)
| File | Δ | Unit | What (responsibility O/B/E/S/P/X) |
|---|---|---|---|
| `evidence/facts.py` | NEW | H1 | typed `EvidenceFact`/`Raw`/`Bound`/`SourceSpan`, negative-proof law, strength resolver, truthiness ban, `migrated_legacy` lift (X/E) |
| `evidence/registry.py` | NEW | H2 | closed fact registry (11 facts) + `census_problems` + `new_raw_structural_extras` raw-write census (X) |
| `evidence/identity_guard.py` | +270 | H0/H4 | shape-net no-growth gate (4 YAML shapes + Python literal, whole package root); H4 fact-provenance rule; `config_consumed` name-blind drop (X) |
| `evidence/context.py` | +30 | H1 | `FactLedger.record_typed` single-author; typed_records/strength (X) |
| `adapters/transformer/debug.py` | +66 | H3 | five-intent rail; nesting-safe `(touched,consumed)` capture; `bound_fields` (X) |
| `adapters/transformer/common.py` | +24 | H3 | `get_config_value` marks accessed **present-only** (phantom-alias fix) (O/X) |
| `adapters/transformer/parser.py` | mixed | H3 (+Soumil block_configs removal) | `consume()` accessor; 16 spec-flow fields migrated; `_resolve` present-only marking (O/B) |
| `adapters/diffusor/parser.py` | +12 | H3 | VAE `act_fn`/`temporal_compression_ratio` + Mochi `max_sequence_length` reads (O) |
| `parser.py` (root) | +13 | H3 | capture unpack → nesting-safe consumed census (X) |
| tests (5 files) | +211 | H0–H3 | new: config_intents, evidence_facts, fact_registry, h4_taint; edited: identity_guard, projection_audit, fact_ledger, sable (3 faithful regression fixes) |

## 15.2 Unit status (agent's honest self-assessment, per §6 DONE = code+counterexamples+isolated+full-suite+corpus+pixels)

| Unit | Status | Confidence |
|---|---|---|
| H0 | DONE | high — baseline, census, shape-net gate, NAS quarantine (Soumil) all verified |
| H1 | DONE | high — behavior-neutral, item-1 amendment honored, 326 tests + real-parse round-trip |
| H2 | DONE-pending-gate | high — ledger closed-world + asserted pins + raw-write census (21 tests); awaiting the running full suite to re-confirm |
| H3 | ACTIVE | mid — rail + consume (16 fields) + phantom-alias fix DONE; blocking flip is LARGE deferred (see 15.3) |
| H4 | ACTIVE (early slice) | high — fact-provenance rule + negative controls; full dataflow-taint net deferred |
| H5–H10 | PENDING | — correctly not started (I-9 broad-excepts, I-10 ownership, diffusion migration all await) |

## 15.3 Decision points requiring Soumil's judgment

1. **The accessed-fix scope (H3).** `get_config_value`/`_resolve` now mark
   accessed present-only. This is correct (residual 231→7 for llama) but it
   touched the GLOBAL audit and exposed 3 nested unreads I handled by adding
   reads to the diffusion adapter (`diffusor/parser.py` +12). *Judge:* keep the
   accessed-fix now, or defer it until the H3.6 scoped audit lands (the flat
   global leak is the real root)? The 3 diffusion reads are honest but are
   arguably H7-scope work pulled early.

2. **H3 blocking flip is far.** The plan's H3.5 blocking net requires the corpus
   `accessed−consumed−ignored` to reduce to only genuine read-never-drawn
   architecture. Reality: that needs (a) the H3.6 scoped audit, (b) the rest of
   the transformer `consume()` + present-sibling-consumed, and (c) the
   diffusion/modality adapters' own `consume()` (H7/H8). *Judge:* pursue the
   blocking flip as a cross-unit endgame, or keep the net advisory and move to
   H7 per §11 order now?

3. **H2 raw-write census granularity.** It pins top-level `extras` KEYS (12).
   It does NOT yet pin nested sub-facts or spec-field writes. *Judge:* is
   top-level-key granularity sufficient for "cannot grow", or extend to nested?

4. **Commit boundary.** Everything is uncommitted and mixes Soumil's NAS work
   with the agent's H0–H3 work. *Judge:* commit in slices (H0/H1/H2 are clean,
   behavior-neutral, gate-green candidates), or hold?

## 15.4 Next-step options (agent recommends, Soumil chooses)

- **Option A (finish H3 to advisory-meaningful, then H7):** consume the last
  transformer spec-flow fields + present-sibling-consumed → transformer residual
  ~0; keep the net advisory; move to H7.1 (diffusion conditioning) per §11.
  *Cleanest per the plan order; blocking flip becomes an endgame step.*
- **Option B (H3.6 scoped audit next):** build the per-owner audit so the
  accessed/unread/consumed sets are scoped, eliminating the flat global leak —
  then the accessed-fix and the blocking flip both become clean and per-adapter.
  *Higher upfront cost; unblocks a correct blocking flip.*
- **Option C (revert the accessed-fix, park H3 at rail+consume):** keep the
  advisory net phantom-polluted-but-known; return to it after H3.6. *Lowest
  risk; loses the meaningful net now.*

Agent's recommendation: **Option A** — it matches §11's order (H3 mechanism →
H7), keeps the net honestly advisory, and defers the blocking flip to the
cross-unit endgame the plan already frames. But this is Soumil's call.

## 15.5 Errors I made this session (consolidated, for judgment)

Each was caught and fixed; listed tight so the pattern is visible.

1. **H1 violated the item-1 amendment** — my lift manufactured
   `completeness="complete"`. → `uninspected` + `migrated_legacy`; native law intact.
2. **H2 over-claimed DONE** — the H2.4 "new legacy structural write" gate was never
   enforced; I claimed "raw-author census blocking against growth" and hadn't built
   it. → downgraded, built the raw-write census.
3. **3 Phase-B regressions** — I edited production WITHOUT running §8.1's pre-change
   blast-radius first. Root: grepped `model_unfolder/` for the capture-API caller,
   not `tests/`; and two committed tests pinned a state H3 legitimately changes. →
   fixed faithfully; §8.1 now binding.
4. **Advisory net was meaningless and I nearly built on it** — assumed the residual
   was near-multipliers; it was 231 PHANTOM aliases. Caught only by probing. → the
   present-only accessed-fix.
5. **The accessed-fix exposed the flat global leak** — cross-scope by-name audit had
   been MASKING 3 real diffusion unreads; my fix surfaced them and broke the blocking
   corpus test. → handled at source (diffusion adapter reads them), not by reverting.
6. **First Phase-B design was wrong** — I planned to DERIVE `consumed` from the
   ledger; it misses geometry (no `_note_fact`). → the `consume()` accessor instead.

**Meta-lesson:** every real error came from acting before the plan's own gate (§8.1
pre-change) or before measuring (the phantom-alias assumption). The plan was right;
skipping its discipline was the mistake.

## 15.6 The judgment, tight

Judge these four; everything else follows:
- **D1 accessed-fix:** keep now, or defer to H3.6 scoped audit? (it's correct but global-scope + pulled 3 diffusion reads early)
- **D2 blocking flip:** endgame across H3.6+H7+H8, or stay advisory and go H7 now?
- **D3 raw-write census:** top-level extras keys enough, or nest deeper?
- **D4 commit:** slice H0/H1/H2 (clean, gate-green) now, or hold all?

Agent leans: D1 keep, D2 advisory + go H7 (§11 order), D3 top-level enough for now,
D4 slice H0/H1/H2. Your call.
