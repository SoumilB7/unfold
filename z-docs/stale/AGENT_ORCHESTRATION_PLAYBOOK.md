# AGENT ORCHESTRATION PLAYBOOK — how to spin up agents that produce the best output
*(distilled from the run_77 review campaign, 2026-07-08→10: 90+ Opus agents,
5×6 review batches, aggregators, two API-outage recoveries. This doc exists so
context compaction never degrades orchestration quality — re-read it before
spinning any fleet. Written by the coordinator, from measured outcomes.)*

## 1. THE SHAPE THAT WORKS (review-fleet reference architecture)
- **5 parallel agents × 6 units each** is the sweet spot: enough parallelism
  to finish a 30-model batch in ~1 agent-runtime, few enough that failures
  are cheap to re-dispatch and the coordinator can actually verify each one.
- **One shared BRIEF FILE on disk**; per-agent prompts stay SHORT and only
  add: (a) the assigned six folders with image counts, (b) one line of
  architecture-specific verification per model, (c) the deliverable contract.
  Agents `Read` the brief as their first act. Upgrading the standard
  mid-campaign = append an ADDENDUM to the brief + SendMessage the running
  agents to re-read it (they pick it up at their next tool round).
- **Balance groups by WORK WEIGHT, not count** (image totals; greedy: sort
  desc, assign to lightest group), and tell each agent to do its heaviest
  model FIRST — so stragglers converge instead of one agent finishing last
  with the giant gallery still queued.
- **Aggregation is its own fleet** (one agent per report type), never the
  coordinator's job, and never the reviewers' job.

## 2. WHAT TO PUT IN THE PROMPT (each ingredient earned its place)
1. **Mandate context reading FIRST**: name the exact law/procedure files
   (PROTOCOL.md, PROJECT_CONTEXT Parts 1+12) and say "before any model."
   Agents that read the laws first produced tier-doctrine-correct judgments
   (dense-but-honest ≠ defect); agents without it over-flag honest pages.
2. **Per-unit hints name the ONE thing to verify + the red-flag condition**:
   "Granite-4 is a Mamba2/attention HYBRID — red-flag if SSM drawn as
   attention", "Arctic has a parallel dense-MLP branch — red-flag if
   dropped". Every P0 the campaign caught was pre-seeded by such a hint OR by
   the exhaustiveness rule. Generic models need no hint; exotic ones always
   get one. Hints must say VERIFY, never assert the answer (the phi-4 hint
   wrongly said "verify DENSE" — the agent correctly pushed back with
   evidence; prompts must leave room for that).
3. **Exhaustiveness as a voiding rule, not a request**: "open EVERY png — a
   skipped image voids the pass; list them in MANIFEST order." Soft phrasing
   gets sampling; voiding rules get 336/336 images opened.
4. **A FINAL MESSAGE contract in raw-data form**: "per model one line:
   slug | verdict (+counts) | ... | anything that blocks trusting the
   result." Their final text is their return value — say exactly what shape
   you want back, or you get prose.
5. **Read-only boundaries, stated twice**: "REPORTS ONLY — package bugs are
   findings, never fixes"; "write ONLY inside your assigned folders."
   Zero boundary violations across the campaign.
6. **The three-way trace (package metering)**: pixels ↔ HF truth (file:line)
   ↔ OUR package site (file:line + mechanism + fix direction), localized by
   grep-the-drawn-label then READING the site; "not localized — needs
   maintainer trace" allowed, guessing banned. This turns findings into
   directly actionable fix tickets.
7. **Honest-refusal handling for degenerate units** (rule E): attempt once,
   classify the typed error, write the reports anyway documenting the state.
   Every folder ends with deliverables; nothing silently skipped.
8. **Durability discipline inside the agent**: "after writing each file,
   confirm it exists before moving on" and, for long single documents,
   "append ONE section per Edit and re-check the file after each."
   This is what made two total-API-outage recoveries cheap.
9. **Contradiction policy for aggregators**: "note contradictions, never
   resolve them" + "derive the thread list from the files — do not copy my
   hint list blindly" + "preserve citations verbatim." Aggregators WILL
   otherwise harmonize away real reviewer disagreements (the DeepSeek-V3
   fused-gate_up contradiction was preserved because of this line).
10. **Output structure by numbered section list** ("EXACTLY this structure:
   1..6") — you get back a document you can verify mechanically
   (`grep -c '^## '`).

## 3. COORDINATOR DUTIES (what NOT to delegate)
- **Verification of existence + structure**: one `ls` loop per agent
  completion (n/3 files per folder), `grep -c '^## '` on aggregates, one
  content spot-check per agent (read one report's core section). Cheap, and
  it catches partial writes the agent believed complete.
- **The COMPLETION SHEET is the single source of truth**: one MD per batch,
  per-model status rows ticked only after on-disk verification. It is what
  makes failure recovery surgical (re-dispatch exactly the ⏳ rows).
- **Never re-do banked work**: files on disk are the ledger; a lost agent
  costs only its unwritten folders.
- **Don't read subagent JSONL transcripts** (context overflow); trust final
  messages + on-disk artifacts. Don't poll agents that will notify.
- Keep the coordinator's own context lean: batch the checks, one-line status
  updates, prompts reference files instead of inlining walls of text.
- zsh gotcha: unquoted `$VAR` does NOT word-split in zsh — use literal lists
  in `for` loops or `printf | while read`.

## 4. FAILURE PLAYBOOK (all three modes hit this campaign, all recovered)
- **Agent stalls but transcript exists** → `SendMessage` to its id resumes
  it with FULL context; tell it exactly what's on disk already ("X and Y are
  3/3; REMAINING: Z — do not re-do banked work") and restate durability
  discipline. If the send returns "queued for next tool round", the agent is
  ALIVE — that's mid-flight steering, not resurrection.
- **Agent dies, transcript lost** ("No transcript found") → fresh GAP agents
  scoped to exactly the pending rows from the completion sheet. Balance the
  gap into 2-3 agents, restate the brief pointer + per-model hints.
- **Writer dies mid-document** → the one-section-per-Edit discipline means
  the file on disk holds all completed sections; resume/redispatch says
  "the file is at N lines with sections A,B done — append the rest, one
  section per Edit."
- **Model availability errors** (rate/limit/unavailable) → resumes preserve
  everything; after a model switch, relaunch fresh (aggregators re-read
  inputs fast; reviewers should be gap-scoped).
- **Watch for the SAME phase failing fleet-wide** — that's infrastructure,
  not the agents; wait/retry rather than rewriting prompts.

## 5. MONITORS & PACING
- A file-count monitor (`while true; count reports; echo on change`) gives
  cheap fleet-wide progress without touching agent transcripts; kill it once
  done. One notification per emitted line — keep the filter tight.
- Completion notifications arrive unprompted; don't also poll. For a
  process you must wait on, arm ONE `until <condition>` background waiter.

## 6. AGGREGATOR FLEET SPEC (per batch, after 100% verification)
Three agents (Sable/Dable/HerEyes), each: read all 30 of its report type
(folder list passed as a FILE), write `_<TYPE>_BATCH<NN>_REPORT.md` with the
numbered-section contract; systemic threads ranked severity×spread with
models·defect·citations·fix-direction·which-net-missed-it; scoreboard table;
model-by-model appendix; (Dable) defect classes + geometry health + package
fix ledger keyed by model_unfolder file:line; (HerEyes) tallies + loves +
every DISLIKE + ranked work items + charter syntheses. One section per Edit.

## APPENDIX — THE CURRENT REVIEWER BRIEF, VERBATIM
(the live copy lives in the session scratchpad as review_brief.md; recreate
it from here if the session is gone)

```markdown
# run_77 REVIEW BRIEF (shared by all five reviewer agents)

You are a reviewer for the model-unfolder project: HF configs → honest interactive
architecture-diagram HTML. You review PRE-RENDERED galleries; you fix nothing.

## STEP 0 — CONTEXT (mandatory, before any model)
1. Read /Users/soumil/Code/Projects/Understand/llmvisualizer/.claude/PROTOCOL.md IN FULL.
   It defines the three procedures you will run VERBATIM: Gate A ("Sable", ~line 62),
   "Dable" (~line 299), "Her Eyes" (~line 352, includes the FIXED review template).
2. In /Users/soumil/Code/Projects/Understand/llmvisualizer/PROJECT_CONTEXT.md read
   PART 1 (the laws) and PART 12 (the tier doctrine: fallback pages state facts
   honestly BY DESIGN; collapse moves truth to cards; token-identity-changing ops
   keep boxes). Judge with these laws: evidence-never-identity; tri-state
   (proven / honest-unknown / never-a-wrong-guess); honesty outranks beauty.

## PER MODEL (folder → hub id: replace "__" with "/")
Work from /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg
(every Bash call: cd there first — the shell cwd resets between calls).

A. MECHANICAL (feeds the Sable report):
   python3 - <<'PY'
   from model_unfolder.sable import sable      # NEVER `from model_unfolder import sable` (module shadow)
   rep = sable("<hub/id>", render_images=False)
   print("oracle:", rep.oracle, "| mechanical:", "PASS" if rep.mechanical_passed else "FAIL")
   for c in rep.checks:
       print(("BLOCKING" if c.blocking else "advisory"), c.name, "OK" if c.passed else "FINDINGS")
       for f in c.findings: print("   -", f)
   PY
   Also validate the folder's HTML:
   from model_unfolder.block_schema import validate_click_coupling
   problems = validate_click_coupling(open("<folder>/<file>.html").read())
   Ignore pad_token_id/bos_token_id stderr noise. NEVER print tokens/env secrets.

B. SABLE REPORT → <folder>/sable_report.md
   Follow Gate A's numbered steps AGAINST THE RENDERED ARTIFACTS (the HTML + PNGs
   are the ground truth — never imagine what code would draw):
   decompose downward (list every drawn block from the HTML), map to existing
   block vocabulary, dynamism judgment, then the TWO-WAY reconciliation of the
   five structure types (arrows/blocks/repeats/connectors/splitters) vs the
   INSTALLED HF modeling source (transformers/diffusers site-packages — open the
   real modeling_*.py of this family and CITE file:line for every claim).
   code→structure: nothing the forward() does is missing (red-flag misses);
   structure→code: nothing drawn is fabricated (red-flag fabrications).
   Include the full mechanical net table from step A with per-net verdicts.
   End with: VERDICT (CLEAN / FINDINGS), ranked findings, oracle state.

C. DABLE REPORT → <folder>/dable_report.md
   Dable per PROTOCOL: EXHAUSTIVE pixel pass. Read EVERY .png in the folder with
   the Read tool (no skipping — a skipped image voids the pass; list them in
   MANIFEST.txt order). Per image: one row — filename | what it shows | pixel
   verdict (CLEAN or the defect: overlap / crooked elbow / overflowing text /
   wrong or misleading label / dangling arrow / duplicated lane / wrong count) |
   truth check against the modeling source where the image makes a structural
   claim. The dangling flag: any arrow/port that connects to nothing. End with:
   image tally, defect list ranked by severity, VERDICT.

D. HER EYES REVIEW → <folder>/her_eyes_review.md
   Run the Her Eyes persona EXACTLY per PROTOCOL: judge ONLY from the PNGs
   (never HTML/code); the FIXED template (the face, the tallies line, "## Every
   view" table with EVERY manifest image exactly once in manifest order,
   "## What she suggests", "## Her answers" — the five charter questions:
   delight per image LOVE/FINE/DISLIKE + one honest sentence; every DISLIKE
   MUST carry a concrete visual suggestion; ceiling; bundling (drills
   preserved); newcomer; journey end). Her suggestions may bundle/calm but may
   never hide a real operation — honesty outranks beauty.

E. EMPTY FOLDER (no PNGs/HTML): the run_77 renderer produced nothing. Try once:
   from model_unfolder import unfold
   d = unfold("<hub/id>"); d.save_images("<folder>"); open("<folder>/<slug>.html","w").write(d.to_html())
   If that succeeds → full procedure on the fresh gallery, and SAY in all three
   reports that you rendered it now (run_77 had produced nothing).
   If it fails/refuses → still write all three reports: sable_report.md records
   the exact error + classification (honest refusal / gated / loader gap) and
   whether the refusal is the CORRECT behavior per the laws; dable_report.md and
   her_eyes_review.md state "no gallery — nothing to review" (Her Eyes template
   with 0 images, tallies all zero).

## HARD RULES
- REPORTS ONLY: never modify package code, tests, YAML, or other previews.
  Package bugs you discover = findings in the reports.
- Write ONLY inside your six assigned model folders.
- Do NOT re-render folders that already have images (the run_77 output IS the
  review subject). Only rule-E empty folders may be rendered.
- No git commands. Never echo HF tokens or env values.
- Rigor bar: every PNG actually opened and looked at; every finding names the
  image/block and the code citation; no vague praise; both directions checked.
  If a page is a FALLBACK-tier page (plain honest stack), judge it against the
  tier doctrine — "dense but honest" is not a defect; a wrong claim is.

## FINAL MESSAGE (your return value — raw data, not prose for a human)
Per model one line: slug | sable VERDICT (+#findings) | dable VERDICT (+#defects)
| her-eyes tallies (LOVE/FINE/DISLIKE) | reports written (3/3?) | anything that
blocks trusting the result.

## ADDENDUM (mandatory from batch 3 on) — PACKAGE METERING: the three-way trace
A defect is NOT fully reported until it is traced to the exact site in OUR
codebase that produced it. For EVERY defect you find (Sable or Dable), report
the full three-way chain:
  (a) PIXELS — the image/view and the exact drawn claim (label/edge/count);
  (b) HF TRUTH — the modeling source contradiction, file:line, verbatim;
  (c) PACKAGE SITE — the site in
      /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/model_unfolder/
      that produced the wrong drawing: exact file:line + the MECHANISM
      (template default applied · config field unread/mis-aliased · evidence
      reader returned None and a default asserted · hardcoded string · wrong
      owner/class binding · missing YAML vocabulary row) + a one-line general
      fix direction.
How to localize fast: grep the package for the drawn LABEL text or block id
(labels live in adapters/*/blocks*, renderers/html/*, labels.py; facts come
from adapters/*/parser.py + evidence/*.py; vocabulary in everchanging/*.yaml).
Verify the site by reading it — never guess a line number.
The package is READ-ONLY for you: cite it, never modify it.
REPORT FORMAT: dable_report.md gains a final section `## PACKAGE METERING` —
a table: defect | image | HF truth (file:line) | package site (file:line) |
mechanism | fix direction. sable_report.md findings each carry the package
site inline. Defects you cannot localize after a genuine search: say so
explicitly ("package site not localized — needs maintainer trace") rather
than guessing.

```
