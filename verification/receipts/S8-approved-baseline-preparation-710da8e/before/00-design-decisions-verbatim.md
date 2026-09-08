# Design decisions — Soumil's answers, VERBATIM (golden context, 2026-09-04)

Round 1. Questions and pictures: `questions/index.html` (open with
`open /Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design/questions/index.html`).
This file is to design what `10-full-research/16` §4 is to intent: never
compress, never paraphrase away. Later rounds append below as Round 2, 3….

## The process rule (Soumil, verbatim)

> yes lets keep it a deisgn choice of ours on every verification will have this html which i can see and decide, this is perfect back and forth

## Round 1 answers (verbatim)

> Q0) yes i agree, but we will have to think abotu a way of representing beautifully something outside the tower
>
> Q1) i like the symbolic representaton of the one on the left side, the arrows need betterment but selected 8x mixtrues is not easy to see you see, so i want some what of the represntation of the interna working here, this is where i am going with a bit too, this is yes a deisgn decision
> things can hold symbolic meaning of the architecture you see - question 2 is similar you see i want the one on left side because it holds symbolic meaning on what a sliding window even is
>
> Q2) one on left and this is design decisions and more like this to come later
> -> add 1 and 2 to the deisgn doc part
>
> Q3) make 2 modes, i guess there are but unify thier toggles into 2 modes, dev mode should shout that light green for sure, and for the normal case we go with the best predicted case upto which we can garantee its right else we can show it broken liek that
>
> Q4) depth 1 can draw that i am perfectly fine
>
> Q5) leaning towards `a` but lets see first renders to design later on
>
> Q6) will choose later on
>
> now this along with the htlm page lets store verbatim because this is golden context like our last discussion and also please lets save it as very important parts
> also save your gathered context too,

## Owner's gathered context (my reading; Soumil corrects, never the other way)

**The new law of symbolic drawing (Q1, Q2).** Soumil wants pictures that
carry the *meaning* of a mechanism, not only its proven occurrences: the
expert fan (Expert 1 … k … k+1 … N) shows what routing *is*; the token strip
with a highlighted window shows what a sliding window *is*. S3 removed both
because they were drawn as if they were occurrences (an invented 15 and 5;
four boxes that looked like four modules). The reconciliation with L1/L2:
**a symbolic glyph is allowed when (a) it is derived only from proven facts
(k = 8, N = 256, window = 4,096), (b) it is marked symbolic in its card and
chips ("one of 256 experts; 8 are selected per token"), and (c) it never
adds an occurrence to the IR, the conformance census, or the parameter
count.** Symbolic is a *presentation* kind, like `port` and `formula`. This
goes into `12` §2a as L2a and into the brief. Q1 also asks for the fan's
arrows to be redrawn and for "8 selected" to be legible at a glance — the
fan should show the selection (e.g. the k chosen lanes solid, the rest
faint), not only the note line.

**Q0 — outside the tower.** DeepSeek's MTP beside the tower (and later
`hc_head`, per-layer projections, AltUp unembed) needs its own beautiful
placement rule, not a box glued above the LM head. Design-brief item; first
renders at S9 decide.

**Q3 — two modes.** Unify today's toggles (pills, ⚠ bar, ⓘ notes, pale
styling, chips, dev-only yellow outline) into exactly two modes:
- **dev mode**: shouts. Every unresolved element in light green, every
  finding row visible, counts on blocks, the openable-outline affordance.
- **normal mode**: the guaranteed-correct picture, as clean as v0.2.17; what
  is not guaranteed is *shown broken* (pale/absent with a marker), never
  filled in. "Best predicted case up to which we can guarantee it's right"
  = the proven subset; nothing predicted beyond proof. **Confirmed by Soumil 2026-09-04 ("yes correct"): normal mode never
  guesses; it only quiets the *reporting* of the unknown, not the unknown
  itself.**
The mode is a page-level switch (one toggle), not per-feature toggles.

**Q4 — depth 1 draws every deviation.** Gemma3n-class layers may be tall.
No collapse-with-marker.

**Q5 — multi-stream residual:** leaning (a) four lines with mixers; decide
on the first S9 renders.

**Q6 — deployment facts placement:** deferred to a later round.

**The process rule.** From now on every step's review ships a decision
page in `z-docs/12-design/` (`questions/`, `S<n>/`…) with pictures rendered
from the tree under review, and Soumil answers in chat; the answers are
appended here verbatim. This is the "perfect back and forth". Added to `15`
§7 (reviewer protocol) and §6 (rulings).

## How Soumil likes to be asked (durable; Soumil 2026-09-04: "this is perfect back and forth")

1. **A page, not prose.** One HTML page per round under `z-docs/12-design/<round>/index.html`,
   pictures rendered from the exact tree under review (playwright in the
   scratchpad venv, or a sub-agent). Before/after pairs side by side where a
   choice exists; a single picture where it is "look at the current state".
   Captions say what the picture is and which numbers in it are proven.
2. **One boxed question per decision**, numbered `Qn` continuing across
   rounds (round 1 = Q1–Q6, S6 page = Q7), under its picture. Plain words.
   State the options concretely; state what each costs in accuracy or beauty
   if that matters. Never assume his taste; "why not X?" is allowed.
3. **"Don't care" is a valid answer**: then I pick, and tell him what I picked.
4. **The absolute path in chat**, plus the one-line `open …` command.
5. **He answers in chat**, in any form; I append the answer **verbatim** to
   this file under a new round heading, then my reading under his, marked as
   mine. He corrects the reading, never the other way round.
6. In chat, the questions are also listed in plain numbered form with one
   sentence on why each matters, so he can answer without opening the page
   if he prefers.
7. Only questions whose answer changes the work. Routine calls I make and
   report.


## S8 family and ownership — Soumil, 2026-09-07

> B5 is recorded: first cutover family = UNet, first witness = SDXL. S7 is DONE.

Later in the same session, verbatim:

> can you make it the same branch and complete the given task and then revire it too and give me a run down text for review , actually i accidentally gave you the task where as i wanted to give it to executor agent, can you in parallel spin up a sub agent whcih you own as executor and i give you stuff so you can do it
>
> also please reread everythign in case anything is lost by. compaction, ask me anything and also make it same branch and get the work done via sub agent anyways

Owner reading: Codex now coordinates an executor subagent and reviews its work; the original S8 scope and approval boundaries remain. Work continues on `audio-composite-support`; isolated verification uses detached worktrees. The initial partial candidate was authored by Codex, so a separate non-implementing reviewer also audits it. This is not output-delta approval or release authorization.


## S8 output-delta approval — Soumil, 2026-09-08

The owner presented final commit `710da8e`, the actual red broad receipt, the separate 11-pass test correction, the 29-witness output-delta decision page, the actual SDXL page, and the explicit remaining limits. The question was: **“Do you approve the enumerated output changes for re-blessing?”**

Soumil answered verbatim:

> yes go ahead

Owner reading: approval applies to the enumerated S8 output changes in `S8/output-deltas/index.html` and its per-witness dispositions, including the separately reviewed example outputs and Bloom view correction. The executor may update the corresponding expected outputs through the existing guarded procedure and run the required verification on `audio-composite-support`. This approval does not resolve the recorded mechanism, execution, configuration or viewing limitations, expand the support set, or authorize push or release. S8 completion still requires the actual green gate and independent final review. Earlier pending-approval records remain historical evidence.
