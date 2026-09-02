# Unit verdicts — U0 through U10, audited against the goal

Audit date 2026-08-31, tree `cd5d7ff` (branch `audio-composite-support`,
U11 active). Every claim below was checked against git and the working tree,
not taken from the unit trackers; where the trackers and the tree disagree,
the per-unit file says so.

The goal each unit is judged against is `../01-product/next-product-architecture.md`:
a general evidence compiler — code is the only mechanism authority, config
supplies values, unknown stays visible, every surface projects one canonical
fact, and a new model is accurate when its mechanisms are already understood.

## Reading order

1. [Systemic findings](systemic-findings.md) — the campaign-level verdict,
   the two things it overlooked, and how it would be done from here.
2. [Recall trajectory](recall-trajectory.md) — the missing measurement: what
   each unit did to the drawn diagrams, per witness, and how each drop
   classifies.
3. Per-unit verdicts, one file each: [U0](u00.md) · [U1](u01.md) · [U2](u02.md)
   · [U3](u03.md) · [U4](u04.md) · [U5](u05.md) · [U6](u06.md) · [U7](u07.md)
   · [U8](u08.md) · [U9](u09.md) · [U10](u10.md).
4. [Experimental confirmation](experimental-confirmation.md) — every claim
   re-tested by rendering the witnesses and 13 out-of-corpus models; verdict
   table, findings the audit lacked, measured distance from intent, ranked fixes.
5. [Cross-unit notes](cross-unit-notes.md) — observations that span units.
6. [Method and plan judgment](method-and-plan-judgment.md) — the U plan as a
   plan, this chapter's own verdicts audited, and Soumil's arbiter method:
   what worked, exactly where it failed, and the three obligations that fix it.
7. [First-principles judgment](first-principles-judgment.md) — the cynical
   implementer's verdict on the whole project against its intent, with the
   alternative physics measured and a rebuild plan.

## Grade table

| unit | what it was for | grade | one-line verdict |
|---|---|---|---|
| U0 | freeze a provable baseline | C → B | measured nothing as shipped; made real by three recovery commits |
| U1 | exact config occurrences, no first-hit resolver | B− | correct end state after two audits reversed an attempt that *added* config authority |
| U2 | receipts, debt register, all nets blocking | B+ | genuine infrastructure closure through five honest vet rounds; left its own U0 pin red |
| U3 | ProgramIndex + exact ownership substrate | B− | A-grade kernel; defined the index boundary, wrote the rule for crossing it (boundary 8), never crossed it |
| U4 | unknown-safety | B− | defaults genuinely gone; step 6 licensed demote-now/re-prove-later and three provable facts were dropped, one silently |
| U5 | consumer firewall | B+ | real, non-vacuous ratchet; 330 backward reads still execute, U14 not started |
| U6 | exact attention evidence | C+ | strong readers; collapsed provable encoder/MusicGen drills onto the stub under a false parity claim; latent BLOOM geometry bug |
| U7 | FFN / router / cell / bookend | A− | five readers deleted with reconciling counts; Falcon re-proven in-unit; acceptance hole closed with poisons |
| U8 | position / mask / schedule | B− | 17k lines of exact readers, shadow-first, all corrections general; three provable facts left unresolved, T5 bias silently absent in six galleries |
| U9 | recursive modality ownership | B− | genuine substrate; Qwen2-VL's verified cell removed as "fabricated" — it was correct |
| U10 | diffusion root / stack / stream | A (execution) / C (outcome) | most rigorous receipts in the campaign; PixArt and SD3.5 reduced to blank boxes by a known, unindexed import boundary |

## The one-paragraph verdict

The doctrine is right and the transformer decoder core proves it (14/14
witnesses fully resolved, no family branch anywhere). Every unit ratcheted
*soundness* — nothing fabricated survives — and none ratcheted *recall*.
The plan itself licensed the loss (U4 step 6) the same day the procedure
forbidding it was written (§7), and the index boundary that would decide
whether a fact is *unknown* or merely *unindexed* was declared in U3, built
as an unused parameter, and only crossed in U11 for UNets. The result is a
product that is honest everywhere and complete only where a mechanism lives
in the model's own file: verified detail was lost in U6, U8, U9 and U10, each
time approved as honesty, and two flagship diffusion models now render as
blank boxes. Both fixes are bounded and already half-built.

## Method

Per unit: intent quoted from its plan; deletions verified by grep at HEAD;
gates checked for unconditional asserts; product delta measured from fixture
`hash_signature` lengths at each closing commit, gallery `MANIFEST.txt`
view-name diffs (tracked from the U7 close), and where necessary re-parsing
the frozen configs against the installed source at historical trees; every
"honesty delta" classified as fabricated-removed, verified-lost, or
consciously-deferred by reading the installed Hugging Face source. Six
parallel read-only audits plus direct verification of U10; no repository
file was modified by the audit.

## Superseded by the full research (2026-09-02)

`../10-full-research/` re-examined every finding here against nine research
passes; `08-findings-register.md` there records each as accepted, voided or
revised. Grade changes: **U4 → C+** (consumers still convert unknown → known),
**U5 → B** (the firewall does not cover semantic reconstruction); project
overall **C+**, plan **B−**, method **B**. The latency root cause claimed in
`first-principles-judgment.md` §3.1 is withdrawn (it is one quadratic bug);
the physics argument stands on construction evidence alone.

## What this chapter is not

Not a tracker. It is dated and will go stale; the live state remains
`07-current-state/`. It records a judgment at one tree so the next plan can
start from what was actually true rather than from what the unit trackers
claimed.
