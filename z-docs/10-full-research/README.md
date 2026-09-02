# Full research — the whole project, understood before it is judged

Started 2026-09-01 at tree `a00ae48`+ (branch `audio-composite-support`); all passes and the synthesis complete 2026-09-02.
Purpose: understand the entire codebase — the engine in `unfold-pkg/` **and the sibling codebases of the `Understand/` umbrella it belongs to** — the intent behind it, and the
problem it solves — completely, with no loose ends — then judge, then void
or accept every earlier finding (`../09-unit-verdicts/`), then plan what
comes after the U plan.

**The goal we must never lose sight of** (to be refined in `00-goal-and-intent.md`):
a source-grounded architecture explainer that renders the true, complete
structure of any supported Hugging Face model as an interactive diagram a
person can learn from — never guessing, never silently omitting.

## Files (each written by a dedicated deep-research pass)

| file | question it answers |
|---|---|
| `00-goal-and-intent.md` | what is this supposed to be, for whom, with what promises — every source of intent reconciled |
| `01-codebase-map.md` (+ `.json`) | every module: purpose, size, category, dependencies, health |
| `02-pipeline-trace.md` | what actually happens, function by function, when `unfold()` runs on a transformer and a diffusion model; where the time goes |
| `03-reader-coverage-matrix.md` | every mechanism family × the reader that proves it × the syntax shapes it accepts × known refusals; the supported set vs the HF ecosystem |
| `04-consumers-and-surface.md` | IR, op-graph, renderers, JSON, params: what each produces; the real user journey through the HTML |
| `05-verification-system.md` | every net, gate, script, corpus witness, receipt: what it proves and does not |
| `06-history-and-docs.md` | the full timeline, every documentation plane, what is current vs stale, the decision log |
| `07-alternatives-and-physics.md` | what best-in-class would be; static analysis vs instantiation vs tracing, measured on real models |
| `08-findings-register.md` | every prior finding, accepted / voided / revised with evidence |
| `09-judgment.md` | the judgment, after all of the above |
| `10-post-u-plan.md` | the plan to execute after the U plan |
| `11-sibling-codebases.md` | the rest of the `Understand/` umbrella — Flowy, unfold-npm, the HF Space, the older explainer apps — what each is, how it relates to the engine, and what product it implies |

Rules for every pass: read-only on the repository (code, tests, corpus,
galleries, manifests); render only into the scratchpad; cite `file:line`;
mark every unverified statement as such; end with a **Loose ends** section.
- `12-execution-order.md` — **v2.4, binding** (2026-09-02): three-axis reconciliation contract, per-question authority matrix incl. value + tying rows, product boundary (base construction incl. factory remap in scope; deployment layer out), S0–S12 with lawful S7 escape, first-principles bets, and the measured SD3.5 today-vs-v2.2 table (§7).nciled with the counter-review): settled decisions, the 9-row reconciliation contract, S0–S12 with shadow-mode oracle before any authority cutover and responsibility-by-responsibility deletion. `07`/`08`/`10` carry amendment notes.
- `13-hard-models-and-relations.md` — the 2026-09-02 probe of DeepSeek-V4 (mHC 4-stream residual, hash-MoE), Gemma3n (AltUp, KV sharing 18→20…), Gemma4, LongCat, Qwen3-Next, Jamba, Nemotron-H, DeepSeek-V3.2, gpt-oss, Llama-4: what the instance+trace substrate exposed, the consensus, and the relation axis.
- `14-confirmation-checklist.md` — **read first after context loss**: intent paragraph, confirmations owed by Soumil, per-step code checks with poisons, model-wise checks, and the recachable state.
- `15-executor-brief.md` — **FINAL** instruction for the implementing agent: laws, the S0–S12 table with done-when ids, completion-sheet format, stop conditions, rulings log, and the reviewer protocol.
