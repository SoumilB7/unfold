# Executor brief — the instruction the implementing agent follows (final, 2026-09-02)

Status: **FINAL.** The plan is `12-execution-order.md` v2.3. No further
architecture debate; changes to the plan require Soumil's written ruling in
this file's §6 log. The product contract is the B1 text in `14` §B (ratified
by Soumil's "final decision" instruction of 2026-09-02).

Roles: **Executor** = the agent that implements. **Reviewer** = Claude
(this session or any successor reading `14` first), who grades each step
against `14` §C/§D and returns ACCEPT or RETURN. **Arbiter** = Soumil, who
approves deltas before re-bless and holds the B-confirmations.

---

## §1 Read before touching code (in this order)

1. `z-docs/10-full-research/14-confirmation-checklist.md` — intent, checks, state.
2. `z-docs/10-full-research/12-execution-order.md` — the order, contracts, bets.
3. `z-docs/10-full-research/13-hard-models-and-relations.md` — the relation axis.
4. `unfold-pkg/docs/U11_UNET_EXECUTION_PLAN.md` §3 (laws) and §9 (stop conditions) — still binding for anything touching evidence code.
5. `.claude/PROTOCOL.md` — Sable / bless procedure.

## §2 Laws the executor may never break

1. **Evidence, never identity.** No branch on class name, model name, repo id
   or family. Exact runtime type in the closed `torch.nn` set is allowed
   (it is the object). A user class named like a primitive is custom.
2. **One-way flow.** Sources → typed facts → canonical IR → consumers.
   Renderers, params, JSON and conformance project; they never decide, default,
   sniff or invent.
3. **Unknown is visible.** A mechanism you cannot prove is a chip, never a
   guess. A structural element or relation you cannot place is a **blocking**
   finding, never silence.
4. **Producer-first.** When a gate fires, fix the earliest false producer and
   delete the bad path. Never weaken a gate, never condition a fact on render
   output, never add an allowlist row without owner + reason + deletion step.
5. **No self-bless.** `bless()` takes a persisted verdict from a reviewer that
   did not implement the change. Every output delta is enumerated per witness
   with its evidence-level cause and approved by the arbiter **before**
   re-bless.
6. **No big bang.** New authorities enter in shadow with zero production
   consumers. Cutover is per family with the old path kept for differential
   testing. Deletion is per responsibility, in its own unit, after parity.
7. **Receipts in-repo.** Every step's broad-gate receipt (isolated worktree,
   full suite) is committed. `/private/tmp` receipts do not exist.
8. **Commit bodies.** Title + what / why / receipt id. The hook rejects empty.
9. **Never two pytest lanes at once** in this repo; verify test filenames
   before batch invocations ("no tests ran" = a missing file, not a pass).
10. **Do not start U11-G/H/I** on the static substrate. Do not add another
    structure-authoring reader.

## §3 The steps (execute strictly in order; each ends with a completion sheet)

| step | do | done when (from `14` §C) | ships |
|---|---|---|---|
| S0 | Rotate + scrub the two HF tokens; notebook secrets scan; track `z-docs/` + `.claude/PROTOCOL.md` (curate `done/`); commit-body hook; archive the 12 superseded docs (`06`) | C-0 | — |
| S1 | Coordinator on `9b3cb7b`/`9a4e1e5`; push; set U11 §10 tracker row F4/F2c DONE with receipt id | C-1 | — |
| S2 | `_SourceWalker._seg` linear; memoise the three hot lookups; record two budgets from a quiet baseline | C-2 | — |
| S3 | Remove the listed consumer unknown→known sites, label sniff, literals, phantom experts, sliding-window fixture, placeholder tower; zero-layer warning; blocking firewall rule "no semantic default resolves an unknown" | C-3 | — |
| S4 | Recall ratchet (+ labels in signature); no waiver without chip; flip advisory nets; wire `census.py --check`; `zero_asserted_census` re-raises; in-repo receipts; independent-verdict bless; `coverage.json` per model | C-4 | — |
| S5 | v0.3.0 **honesty release**: README (support set, known-incomplete, coverage numbers); Space unpinned + typed refusals; examples regenerated; `deepseek-v3.html` fixed; tag + PyPI + Space | C-5 | **v0.3.0** |
| S6 | `physics/instance_inventory.py` + `physics/execution_observation.py` (subprocess, network off, timeout, mem cap, §1c records, bf16 recipe, function-mode op log, lazily observed modules); 8-model pilot; **zero production imports** | C-6 | — |
| S7 | `evidence/reconciliation.py`: three axes + relation axis as closed dataclasses; precedence matrix as code; shadow on 29 + 10 models; disagreement matrix committed; Sable net blocking on unresolved axis; **no pixel change**; 3-week box with the lawful escape only | C-7 | — |
| S8 | UNet family cutover: G′ thin projection over the table; D/E/F readers on exact cell classes; 14 `unet_*` readers wired or deleted; old path behind a flag for differential; zero unexplained differences | C-8 | — |
| S9 | DiT/text families: rival-invocation pruning; readers from exact class (mixin lane, processors, `canonical_import`, partial cell, heterogeneous `FeedForward.net`, two-lane FFN); lost-detail restoration list with owners; negatives under closure; `multi_stream_residual` fact + bus rendering or visible chip | C-9 | **v0.4.0** |
| S10 | Rolling deletion units (unet.py structure authorship + renderer UNet authorship first) | C-10 | — |
| S11 | U12 VAE, U13 scheduler on the hybrid; U14 firewall rows → 0; U15 YAML → syntax, dead authorities, duplicate helpers, `parse()` split | C-11 | **v0.5.0** |
| S12 | Learner spec as acceptance; Her Eyes blocking per bless; renderer-only npm; cached-JSON Space | C-12 | **v1.0** |

Witnesses that decide each step are in `14` §D. Gemma3n and DeepSeek-V4
enter the corpus at S7 (they are the relation-axis witnesses).

## §4 Completion sheet (one per step, committed as `z-docs/11-execution/<step>.md`)

```
step: S<n>
tree: <commit sha>   receipt: <id>   receipt path: <in-repo path>
checks: C-<n> items, each: PASS | FAIL + artifact path
poisons run: <list, each with the red output captured>
deltas: per witness: <what changed> ← <evidence-level cause>   arbiter approval: <date/link>
coverage: proven/flagged/silent per model (silent must be 0)   delta vs previous: +/-
deleted: <files/functions> with the responsibility they carried and its replacement
open: <anything left, with owner and the step it moves to>
```

A sheet with a FAIL, a missing artifact, or a silent > 0 is not a completion;
it is a status report. Do not mark DONE.

## §5 Stop and report (do not improvise) when

- a step needs class/family/name semantics to work;
- a gate would have to be weakened or an allowlist grown to go green;
- a renderer/params path needs raw config or source access;
- a preservation delta has no evidence-level explanation;
- the reconciliation layer needs a precedence not in `12` §1b;
- a real model contradicts a general rule;
- the verification tree changes during a gate;
- the S7 box expires with unclassified occurrences in every family;
- a required library cannot be executed in isolation for a corpus witness.

## §6 Rulings log (append-only, arbiter only)

- 2026-09-02 — plan v2.3 declared final; B1 contract text adopted; executor
  brief issued.

---

## §7 Reviewer protocol (Claude)

For each completion sheet: (1) reproduce the receipt on the named tree in an
isolated worktree; (2) walk `14` §C-<n> item by item and run each poison;
(3) check every delta against its cause and the arbiter approval; (4) recompute
`coverage.json` and diff; (5) for shipping steps, install from PyPI in a clean
venv and open the Space; (6) return **ACCEPT** or **RETURN** with the failing
check ids. A RETURN names the earliest false producer, never a patch to the
gate. The reviewer never edits production code during review.
