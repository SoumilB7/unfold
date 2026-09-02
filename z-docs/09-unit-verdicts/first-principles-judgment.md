# First-principles judgment — the project as an implementer would build it

Written 2026-09-01 against tree `a00ae48`, after reading the doctrine, the
architecture chapters, the code and process metrics, and after five render
experiments (`experimental-confirmation.md`). This is deliberately the cynical
reading. The generous reading is in the per-unit files.

## 1. What was intended

The doctrine says it plainly: *"reconstruct a model's architecture from
artifacts the model actually ships; never fill a gap with a familiar template."*
The README says the product: *"your one click model unfolder"* — an interactive
diagram a person can learn from, in a notebook, from a model ID. Soumil's own
framing in conversation: the diagram must be worth looking at, honest, and
complete enough that nobody has to read the code.

So the intent has three legs: **true**, **complete**, **wanted**. Judged on all
three.

## 2. The verdict

The project built the strongest honesty engine for neural architecture I have
seen, and in doing so it confused the engine for the product. It chose the
hardest possible physics for the easiest part of the problem, it ratcheted the
one property that never fails visibly and left unmeasured the property that
does, and it wrapped the whole thing in a process whose weight now exceeds the
code. The result at HEAD: a text-model explainer that is genuinely correct and
close to complete; a diffusion/multimodal explainer that is honest and mostly
empty; a codebase of 136k lines with a 3,762-line function at its centre; and
no clear answer to the question "who is this for, and what happens when they
click."

**Against intent: true — achieved; complete — half; wanted — undefined.**

## 3. The five deepest judgments

### 3.1 Right north star, wrong physics

The campaign decided that *everything* — what modules exist, how many, which
class, what shapes, and what the forward does — must be recovered by static
analysis of Python source. The product contract even enshrines it: *"model
code is not executed."*

But there are two different questions here, and they have different physics:

- **What exists** (components, stacks, ×N, block classes, parameter shapes,
  which guarded branch was constructed). Python answers this in one call:

  ```
  llama-7b     meta-device instantiation: 0.30 s  → 423 modules, 291 shapes
  deepseek-v3                             0.03 s  → 1,215 modules incl. q_a_layernorm / kv_a_layernorm
  pixart                                  0.02 s  → 28 × BasicTransformerBlock, attn1 heads=16, attn2 cross=1152, ff=[GELU, Dropout, Linear]
  ```

  No weights, no forward pass, no GPU. Everything U3's ownership substrate
  (≈10–15k lines, six weeks, five vet rounds) approximates statically — and the
  things it still misses (DeepSeek's `q_a_layernorm`, PixArt's block class,
  DBRX's expert axes, every "external container") — are simply *there*.

- **What the forward does** (order of ops, residual taps, where the mask and
  rotation are applied, gating algebra). This *is* a static-analysis problem,
  and the readers for it are genuinely good.

The project applied the second method to both questions. That single choice
explains the bundle boundary, the blank denoisers, the DBRX ambiguity, the
30–100 s cold latency of a "one click" tool, and roughly half of `evidence/`.
"Model code is not executed" was a taboo written as a law; instantiating
`__init__` on the meta device is not running a model, and remote-code trust is
a policy switch, not an architecture. The master plan (§22.5) even admits the
runtime probe is legitimate — and nobody built it.

### 3.2 Doctrine became dogma

"Evidence, never identity" is exactly right for *models*: `model_type ==
"gemma2"` must never select a sandwich norm. The project extended it to
*primitives*, where a name is the definition. `nn.LayerNorm` is a LayerNorm.
`nn.GELU(approximate="tanh")` is tanh-GELU. The absence of any norm on q/k is
provable by inspection. Instead: DBRX draws `norm_kind: unknown` ×40 for a
plain `nn.LayerNorm`; seven of twelve decoders carry a "QK norm unresolved"
chip where no QK norm exists; tanh-GELU is labelled GELU. Unknown was turned
into a virtue and then over-applied to things that were never unknown.

Meanwhile the *actual* identity heuristic survives wherever it is load-bearing:
`everchanging/conformance/type_roles.yaml` maps class-name **substrings**
(`Attention`, `Norm`, `Router`, `GELU`) to roles, and `forward_ops._role_of`
consumes it at the heart of the conformance nets — whitelisted by the identity
guard as a "code-shape marker." The doctrine forbids in the parser what the
gates depend on. That is not a doctrine; that is a taboo with an exemption.

### 3.3 Soundness without recall — the product got emptier as it got "better"

Twenty-six nets ask "did we claim something false?" Zero ask "did we lose
something we had proven?" Every unit was therefore free to trade detail for
unknown and call it honesty — and U4's own plan licensed it in writing the same
day §7 forbade it. The trajectory file shows the consequence: the corpus lost
verified structure in U6, U8, U9 and U10; PixArt and SD3.5 went from correct,
conformance-verified skeletons to blank boxes; T5 encoders in six galleries
draw a bias the code applies as if it weren't there; Qwen2-VL's text FFN is
`code_proven` in the ledger and "unresolved" on screen — with every gate
green each time. A blanket excusal list (`_DIFFUSION_SOURCE_CLOSURE_PATHS`)
then makes the blank boxes pass the config audits too.

An engineer optimising a soundness metric will always converge on emptiness.
The gate that measures the *other* direction costs a few hundred lines. It was
never built, and the lost-detail rule lived as prose for five weeks while
being violated four times.

### 3.4 Process became product

| measure | value |
|---|---|
| commits, May 6 → Sep 1 | 487, one author identity (agent-driven) |
| `evidence/` | 95.8k lines, 142 modules |
| tests | 66k lines, 3,920 tests, full lane 63 min, receipt 87 min |
| execution prose | 20.6k lines in `unfold-pkg/docs/` + 9.4k in `z-docs/` |
| dataclasses | 497 |
| distinct words for "unknown" | 10 (`ambiguous`, `unknown`, `failed`, `absent`, `unresolved`, `incomplete`, `opaque`, `partial`, `external_unavailable`, `missing_source`); 1,388 "unresolved" literals |
| longest function | `adapters/transformer/parser.py::parse` — **3,762 lines, 470 locals, 277 ifs, nesting 7** |
| receipts reproducible today | 1 of 10 |

The doctrine says *"a helper that performs several of these jobs is temporary
coupling, not a pattern to copy"* — above a parser function that performs all
of them, at 3,762 lines, that grew *during* the campaign. Each unit produced an
800–1,300-line plan, per-slice narrative records, and tracker rows that are
append-only by rule; three documentation planes now disagree with each other
and with the tree. The receipt discipline caught real defects (it earned its
keep in U2, U7, U10). The narrative around it did not. This is what a fleet of
agents optimising to pass their own gates looks like: rigorous, exhausting,
and increasingly about itself. The ratio of ceremony to product is off by
something like five to one.

### 3.5 The product was never defined

The README sells "one click." The internals are an evidence compiler. Nobody
wrote down who the user is. A *learner* needs beauty, bundling, a journey
that ends before the leaves — Soumil's own "nobody gives a fuck about blocks
and blocks" point — and can tolerate a grey box. An *auditor* needs provenance
chips, receipts and the exact code line. These pull in opposite directions,
and the codebase serves the auditor while the README sells the learner. The
version string is inconsistent between `pyproject.toml` and `__version__`
(the docs' own ledger admits it). A cold `unfold()` takes 30–100 s and
nothing is cached across processes. None of this is fatal; all of it says
"no one is holding the user's seat at the table."

## 4. What is genuinely excellent — and I would not touch

- **The honesty guarantee is real and rare.** 42 rendered models, one wrong
  proven fact. No family branch anywhere. Unknown never becomes a default.
  Almost no one in this space has this, and it was the hardest thing to build.
- **The forward-semantics readers for text models are correct.** MLA widths,
  fused QKV, sinks, softcap, sandwich norms, Granite scaling, partial rotary —
  all checked against source, all right.
- **The counterexample culture.** Rename poisons, sibling poisons,
  source-missing poisons, forged-fingerprint poisons. This is how you keep a
  system honest and it should be kept exactly as is.
- **Deletion in the same unit, isolated-worktree receipts, U7 as a whole.**
  U7 is what every unit should have been: five readers deleted with counts
  that reconcile, a regression re-proven inside the unit, an acceptance hole
  closed with poisons instead of carried.

## 5. How I would build it from here

1. **Split the physics.** Structure (tier 1) from meta-device instantiation:
   module tree, classes, ×N, shapes, constructed branches — in milliseconds,
   opt-in gated for remote code. Mechanism (tier 2) from the existing static
   forward readers, now pointed at exact module instances instead of guessed
   occurrences. Checkpoint tensor headers as the third witness for shapes.
   This deletes most of the ownership substrate, closes the bundle boundary
   permanently, and drops cold latency by an order of magnitude.
2. **Fix the doctrine's scope.** Identity of a *model* never selects
   structure. Identity of a *torch primitive* is its definition. Prove
   negatives. Delete `type_roles.yaml` or promote it honestly to a primitive
   table under the same rule.
3. **Add the missing half of the gates.** A directed recall net (proven →
   unresolved requires a re-proof record), reverse-direction conformance
   (executed op must be drawn or chipped — "no waiver without a chip"),
   labels in fixture signatures, and a published per-model coverage receipt:
   proven / flagged / silent, silent = 0.
4. **Restore the lost detail with the seven ranked fixes** in
   `experimental-confirmation.md` §5 — most are one general slice each.
5. **Cut the ceremony to what earned its keep**: isolated receipts, poisons,
   one machine-readable tracker. Retire per-slice prose. Refactor the 3,762-
   line parser into the observe → bind → interpret → project stages the
   doctrine already names.
6. **Define the user and ship the learner product** on top of the audit
   engine: journeys, bundling, drills that end — the design work Soumil has
   been waiting to do.

## 6. Grade

| dimension | grade | why |
|---|---|---|
| engineering rigor | A− | receipts, poisons, deletion discipline, honesty guarantee |
| architectural judgment | C | static-only for construction; boundary never crossed; soundness-only gates |
| doctrine hygiene | C+ | right law, over-applied to primitives, exempted where load-bearing |
| product judgment | D+ | user undefined; "one click" at 30–100 s; the visible product regressed while the engine improved |
| delivery against intent | B− | true ✓ · complete ½ · wanted ? |

**Overall: B−.** A superb engine in search of its product, built with the
wrong physics for half the problem and the wrong metric for progress —
recoverable in weeks, because everything that is hard is already done and
everything that is missing is cheap.
