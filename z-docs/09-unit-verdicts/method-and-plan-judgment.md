# Judgment of the U plan, of this chapter's own verdicts, and of the method

Written 2026-09-01. Cynical by request. Three subjects: the master plan as a
plan, the verdicts in this chapter as verdicts, and the way Soumil runs the
work — which is the part that decides everything else.

---

## Part 1 — The U plan

### What it is
`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md`: 2,983 lines; a §5 audit of
blocking findings; a §16 producer-to-consumer ledger (T-/D-/M-/U-/V-/J-/X-
rows); §12 unit acceptance gates; §13 a mandatory commit template; §14 a
ten-point definition of done; §19 "one pass = one campaign, separately
reviewable commits"; §20 a binding runbook per unit; §21 an append-only
tracker. Per-unit execution plans of 700–1,700 lines refine it.

### What is right about it
- It is a *ledger* before it is a plan: every unit names what it deletes,
  which witnesses must pass, which counterexamples must fail. That is why
  the campaign never regressed to family branches — the plan made fabrication
  structurally hard to commit.
- §12's gates are the best unit-acceptance contract I have read for this
  kind of work: production path not just tests, negatives fail before and
  pass after, equivalent + counterexample models, missing/partial/ambiguous
  source, alias collisions, unchanged-tree fingerprint, old path shrinks in
  the same commit, no bless without explicit decision.
- §19's "one campaign, many reviewable commits" and the sequencing
  (substrate → unknown-safety → firewall → mechanisms → consumers → deletion)
  are correct in the abstract.

### What is wrong with it
1. **It plans soundness and never plans recall.** Ten definition-of-done
   points, twelve acceptance gates — none says "previously proven detail
   must not become unknown." The plan's own U4 step 6 says the opposite
   ("do not restore the default; fix the reader later") and was never
   reconciled with the §7 lost-detail procedure written the same day. A plan
   that measures only one direction of error will be followed in only one
   direction.
2. **It fixed the physics before measuring it.** "Static source, never
   executed" was adopted as a law in §1 without an experiment; the
   meta-instantiation alternative (0.02–0.3 s for the entire module tree)
   appears only as an optional §22.5 afterthought. The most expensive
   decision in the project has no evidence attached to it.
3. **It sequenced the boundary wrong.** Import closure was declared in U3's
   boundary 8, scaffolded, and left; U9 and U10 were then planned on top of
   an index that could not see their subjects. The plan never had a gate
   "the index must contain every constructed class before a mechanism unit
   may start."
4. **It legislates a process it cannot enforce, and the process drifted.**
   §13's commit template appears in **1 of 487** commits; **every one of August's 111 commits has an empty body** (396 of 487 overall — corrected by `../10-full-research/06-history-and-docs.md`). The tracker is append-only by rule and now
   carries ACTIVE and DONE rows for the same unit. The plan's rules for
   *itself* were the first thing abandoned.
5. **It never names the user or the visible product.** 2,983 lines with no
   sentence about who opens the diagram and what they need to see. Priority
   order puts completeness fifth and "visual polish" sixth — which is how a
   plan produces blank flagship models with every gate green and calls it
   progress.
6. **It grows as fast as the code.** ~9.5k lines of execution plans plus the
   master plan, per-slice narrative records, three documentation planes that
   disagree. The plan became a second product.

**Plan grade: B.** Right structure, right gates for the half it measures,
wrong physics, wrong sequencing at the one boundary that mattered, blind to
the user, and not followed by its own process.

---

## Part 2 — This chapter's own verdicts, judged

The verdicts (`u00`–`u10`, `recall-trajectory`, `systemic-findings`) were
produced by six read-only audit agents and my direct checks, then tested by
five render experiments. Judged the same way:

### What holds
- Every claim that the experiments could test was confirmed; none was
  refuted. The central thesis — unindexed mistaken for unknown, soundness
  ratcheted while recall never was — survived rendering 42 models on three
  historical trees.
- Verification against the tree, not the trackers, found real doc/tree
  disagreements (U3's quarantine counts, U8's stale header, U9's receipt
  directory, "dead helpers deleted").

### What was wrong or weak, and corrected
- **Wrong**: "SD3.5 `encoder_0` restored in U10" (it is the U6 stub);
  "U2–U7: no recall loss" (U6 was the first loss); "no gate fails on a
  regression" (gates fail, undirected); "MTP drills provable, unassigned"
  (config-authored, correctly removed).
- **Under-stated**: the U10 root cause is two causes; the U6 loss is total,
  not partial; soundness has a real hole (DeepSeek yarn scale).
- **Weak method**: the first pass reasoned from fixture view *counts*, which
  are not a detail metric — the U4/U5 auditor and Experiment A had to
  re-parse at historical trees to see facts. Grades are one person's
  calibration; the A−/B−/C+ spread is defensible but not measured.
- **The chapter reproduces the flaw it criticises.** It is ~2,500 lines of
  dated prose — a fourth documentation plane — and the experimental evidence
  behind it lives in `/private/tmp/…/scratchpad/exp{A..E}/`, which will
  vanish exactly like the nine receipt directories it faults. The probe
  scripts should be committed under `unfold-pkg/scripts/audit/` or the
  chapter is another unreproducible claim.

**Chapter grade: B+ as an audit; C as a durable artifact until its
experiments are checked in.**

---

## Part 3 — The method: how Soumil runs this

### The shape of it, as observed across the whole campaign
Soumil does not write code and rarely reads it. He runs a **two-model
adversarial loop**: one system implements (Claude fleets, 5×6 batches with
a shared brief — the playbook), a second vets independently (Codex, named in
the U3 runbook as contract owner and reviewer), and he sits between them as
**arbiter**: issuing binding directives with exact specs, holding commits
unpushed until vetted ("keep 4cece0d unpushed and amend"), forcing correction
rounds (B1 V1→V4, B2 V2→V4, six execution-substrate rounds, U2 R0→R9,
COR-0..5), refusing self-certification ("mass registration = debt-
laundering"), ruling on semantic deltas before any bless, and periodically
demanding whole-campaign audits — the DeepSeek "full circle" question, the
U4/U5 check, the U9/U10 checks, this chapter, the render experiments. His
standing rules are in `.claude/PROTOCOL.md` and a set of feedback memories:
evidence-not-identity above everything, producer-first correction, gates
must assert the predicate not a proxy, receipts run the broad gate, context
docs evolve rather than append, config vocabulary lives in YAML, never
bless silently.

### What is genuinely excellent — and unusual
1. **He institutionalised distrust of his own tools.** Independent vetting,
   held commits, correction rounds, the rule that a green gate is never
   more important than evidence direction. Most solo builders using agents
   ship whatever the agent says is green; he built a court.
2. **His rulings are consistently right and general.** "Inert is not
   completion." "Never silently omit failed alias resolution." "Remove the
   closed-world coverage certificate." "Mass registration is debt-
   laundering." "A detail lost in migration is not an honest unknown until
   re-proof is attempted." Each one is the correct engineering principle,
   stated as a rule that applies to every future case, not a patch. That is
   the rarest skill in the whole project.
3. **He asks the meta-question.** "Why are we going in full circle?" "Judge
   this from your own standpoint." "Confirm it with experiments." The
   campaign's biggest flaws were all *found by him asking*, not by any gate.
4. **He converts corrections into standing law** (the feedback memories,
   PROTOCOL gates). The process learns.

### Correction after checking Sable's visual layer

My first draft graded "the one human duty — looking at pixels" a D. That
misread the design. Soumil built perception to be *delegated*, not performed:
`sable()` renders every distinct view to a PNG gallery with a manifest;
`SableReport.visual_review` is filled "by the caller (inline, or a
vision-subagent fleet) against `report.rubric`"; `bless()` refuses without a
gallery whose PNG count matches the distinct-view count; PROTOCOL's Dable
makes agents read PNGs as pixels by default; Her Eyes is an agent persona;
and the July run_77 campaign put 90+ agents through 425 models with a
three-way trace — *pixels ↔ HF truth (file:line) ↔ package site* — that
listed omissions per view ("omits norm2/ffn_norm2 post-norms — Sable
finding #1"). That is a recall-capable perception layer, designed before any
of the losses happened, and it is ahead of anything comparable. The human's
job in this design is to arbitrate the fleet's findings — which he did.

So the defect is not "he didn't look." It is that **the perception layer he
designed was never made a required, independent, persisted, recall-capable
input to bless — and the one instance of it that could catch lost detail was
run once and never again.** Precisely:

1. **The bless gate accepts a self-set string.** `bless()` requires
   `visual_review == "CLEAN"` and that PNGs exist; the fixture then stores
   only `gallery_dir` and `png_count` — no reviewer, no findings, no rubric
   verdicts (`visual_review` is `None` in every fixture at every bless). The
   agent that implemented the change can mark its own report CLEAN. The
   defendant certifies the verdict — automated.
2. **The rubric asks "is it drawn right?", never "is it still there?"** All
   seven `VISUAL_RUBRIC` items are layout: arrows through blocks, overlaps,
   clipping, duplicate labels, chip collisions. A blank denoiser has no
   overlaps. A perfect rubric pass blesses PixArt's empty box by
   construction.
3. **The recall-capable protocol existed and was not re-run.** run_77's
   pixel↔HF-truth trace would have caught the T5 bias (U8), the Qwen2-VL
   cell (U9) and the PixArt/SD3.5 boxes (U10) on the first image. It ran in
   July, at scale, and never after any migration bless. The U8–U10 records
   say galleries were "inspected" and "opaque/detail boundaries match the
   typed evidence" — checking the picture against the evidence that produced
   it, which is circular.
4. **Her Eyes — the only independent, persisted visual judgment — lapsed on
   2026-08-06.** Eight reviews exist; U8 (49 PNGs), U9 and U10 (73 PNGs)
   re-blessed with none, against PROTOCOL's own closing rule.

What remains true from the first draft: three consecutive blesses passed
every major loss through the arbiter on descriptions and hashes, because the
fleet output that should have been in front of him was never demanded by the
gate.

### The other failures (unchanged)

5. **When both models share a blind spot, the arbiter has no third witness.**
   Implementer and vetter both measured soundness; neither measured recall;
   he ruled on what they surfaced. §7 was written and then violated four
   times because a prose rule is not a gate and nobody demanded the gate.
6. **He let the process outgrow the product** — 1,300-line plans, append-only
   trackers, 87-minute receipts, three documentation planes — and let his own
   rules lapse: 1 of 487 commits follows §13; "never git commit, Soumil
   commits himself" became 474 agent commits under his identity.
7. **He never wrote the product spec.** Every directive is about evidence,
   ownership, deletion, receipts; none says "the learner sees X, clicks Y,
   the journey ends at Z." His best product insight lives in a chat message.
8. **Sequencing by doctrine instead of by value** put the decisive boundary
   sixth and let the visible regression accumulate for a month.

### Verdict on the method (revised)

**A for judgment, A− for perception-layer design, C− for making that layer
binding, and B+ overall.** The arbiter model is the right way for one person
to run agent fleets at this scale; his rulings are better than most senior
engineers'; and he built — in July — exactly the pixel↔truth review that
would have prevented every loss since. The failure is narrower and more
fixable than "didn't look": the gate never required the fleet's verdict, the
rubric never asked about loss, the verdict was never persisted, and the one
run that could catch loss was never repeated. Three obligations, none of them
on his eyes:

1. **`bless()` requires an independent review artifact**, not a string: a
   persisted per-view verdict file (rubric + pixel↔truth trace + before/after
   diff) produced by a vision fleet that did not implement the change, with
   the fixture recording who and what. Self-marked CLEAN is refused.
2. **Add the missing rubric direction**: for every changed view, "what did the
   previous gallery show that this one does not, and does the source still
   do it?" — the run_77 trace, made mandatory per bless, plus a directed
   recall net so the fleet's report lists views that got more generic first.
3. **A one-page product spec** naming the learner and the journey, and one
   question per unit that neither model wrote: "what is the product losing
   this week, and what are you both not measuring?" — answered in numbers.

He built the court, wrote good law, and designed the witnesses — then never
subpoenaed them. Subpoena the witnesses.
