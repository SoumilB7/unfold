# S8 status sheet — STOPPED, not complete

step: S8 — UNet family; B5 recorded by Soumil; first witness SDXL; S7 remains DONE.

tree: `bb48a31e336c1721f0a111862bd081acf390bb25` (isolated WIP branch `codex/s8-unet-stop-review`, based on `83140f1`).

receipt: `S8-class-lookup-stop`; path: `unfold-pkg/verification/receipts/S8-class-lookup-stop/`. This is a stop receipt, not an acceptance receipt. The original branch retains the uncommitted candidate. No push, release, or re-bless occurred. The four unrelated untracked documents were excluded from the snapshot.

## Stop condition and counterexample

[U11 §9](../../unfold-pkg/docs/U11_UNET_EXECUTION_PLAN.md) requires stopping when “the one ProgramIndex cannot express a required source address and a new neutral observation record would be needed.” This explicit condition applies to the skip-input proof below. Positive facts may still be drawn without exhaustive execution knowledge.

In actual SDXL up-stage source, the saved residual argument is popped, optionally transformed by `apply_freeu`, then passed to `torch.cat`. The concat-to-resnet call edge and root saved-bank-to-stage argument are independently positive evidence. Closing the intervening input lineage requires resolving the optional branch or proving its transformation. Its guard calls `getattr(self, name, None)` for four attributes. Absence from the instance attribute inventory alone does not establish the default: Python lookup may find a class attribute.

`ProgramIndex.ClassRecord.body_assigns` records direct class-body assignments, but not conditional class-body assignments or a complete class-namespace coverage witness. A closure cannot treat that partial list as a proof of absence. No new observation record or class-lookup authority has been implemented.

The receipt's `source-boundary.json` contains exact installed-source excerpts and their file hash. Its counterexample constructs two classes whose only source change is `if True` versus `if False` around a class-level scalar assignment. Both have identical constructed module populations and no corresponding instance field; neither has an indexed conditional member declaration. One execution observes a multiplication. This disproves instance-only lookup closure. The other trace's lack of multiplication is not used as a static negative proof. Source hashes differ, as they should; this does not assert identical evidence records.

## Bounded prerequisite for owner review — not implemented

Consider a neutral index extension recording class-body assignment addresses under their guards and whether the supported class namespace is completely accounted for. A reader could combine these with the exact resolved class/MRO and existing constructor evidence to prove a particular lookup or default. Conditional assignments must be represented or explicitly prevent closure. Descriptors, decorators, custom lookup and namespace mutation remain unresolved unless the specific lookup is closed. A runtime class-namespace witness is another possible route requiring its own authority contract; this sheet adopts neither.

Required poisons for an adopted route: conditional class member changes lookup; inherited member defeats instance-only absence; unresolved descriptor/custom lookup prevents default proof; simple closed absence permits the default. Do not add a FreeU/model-name exception, infer inactivity from a trace, or move this debt out of S8 to pass its exit. Owner disposition is required under U11 §9 before implementation resumes.

## Checks

| C-8 / user exit | Status | Evidence and limit |
|---|---|---|
| Existing fact layer and ModelIR; exact runtime class bindings | PASS for candidate routing | `unet_runtime.py`, `runtime_source.py`, `unet_cutover.py`, ordinary facts/IR; final proof review remains open. |
| Fourteen readers with production callers or deletion | FAIL as completed exit | Orchestrator calls existing readers; final per-reader authority/deletion inventory unfinished. |
| Old path unreachable in normal production | PASS for focused routing controls | Private scoped comparison flag; direct old-author rejection and production routing tested. Full differential not run. |
| Down/mid/up and cross-attention proven/chipped/guarded | PASS for ordinary artifact presence | Seven stages; cross-attention role remains chipped. Context argument provenance is not promoted to a Q/K/V mechanism proof. Visual acceptance open. |
| Skip concats and conditioning connected | FAIL | Positive fragments render; saved-bank-to-concat input crosses unresolved FreeU branch. Full conditioning/cell route closure remains open. |
| Real shape-derived banner | PASS for ordinary SDXL | 2,567,463,684 denoiser parameters and seven stages, not full-pipeline parameters or transformer-layer counts. |
| Family facts qualified; S7 unstamped-fact debt closed | FAIL | Eleven typed fact keys exist; family matrix not regenerated or closure demonstrated. Placement never qualifies facts. |
| All UNet witnesses differential; zero unexplained | FAIL | Full delta enumeration/re-proof report not performed. No arbiter approval or re-bless. |
| Family execution closure | FAIL | Accepted 240 unresolved SDXL occurrences not closed. Candidate supplies no observations, leaving its 1,930 execution rows class 1. Accepted S7 positive observations must be lawfully integrated. |
| Three full implementation→connection→fact→overview→drill→numbers traces | FAIL | Not completed. Ordinary artifacts alone are insufficient. |
| Six HTML conditions, including same-scratch-source perturbation | FAIL | Ordinary completed; other conditions and unchanged-copy HTML byte identity unperformed. |
| Actual HTML decision page | PASS for delivery | [Decision page](../12-design/S8/index.html), latest ordinary page and baseline. Browser unavailable; pixels not inspected. |
| Deletion close behind | FAIL | Old author quarantined; no legacy authority file/deletion unit removed yet. |
| Isolated serial broad gate, preservation and fingerprints | FAIL as broad exit | Focused snapshot check passed; no broad gate, full preservation verdict or Linux CI. |

## Six input conditions

1. **Ordinary: PASS for artifact generation.** Input, facts, inventory, IR and HTML persisted in `ordinary/`. Static/runtime root source hashes both `052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26`. HTML hash `46c26639e85e5ccb28fa42db976003c1b74543d372d1e139697fb9a7cdbb24da`; wiring checker returned no problems. This is not a scratch-perturbation result.
2. **Sparse valid config: FAIL / not run to HTML.** Labelled declaration-default support exists; before/after witness absent.
3. **Misleading field: FAIL / not run to HTML.** Focused routing poison does not replace field-flagging demonstration.
4. **Equivalent rewrite: FAIL / not run to HTML.**
5. **Computation change and unchanged-copy control: FAIL / not run on SDXL.** Generic source override and synthetic construction/hash controls exist. They do not establish required SDXL fact/drawing/numbers changes or HTML byte identity.
6. **Missing evidence: FAIL / not run to HTML.**

## Deltas, coverage and approval

- **SDXL partial observed delta:** hollow baseline to constructed stages, module drill cards, positive FFN/connection fragments and shape counts. Causes: candidate runtime construction, selected source readers and shape-value proofs. This is not an exhaustive semantic/pixel inventory; unenumerated output deltas remain blocking. Actual before/after HTML is linked on the decision page.
- **SD-v1-4:** an earlier exploratory probe produced 709 modules, 16 FFNs and 859,520,964 denoiser parameters. It was not rerun on this immutable candidate; no final verdict is claimed.
- **Additional corpus/unseen UNets:** exhaustive witness enumeration and differential remain open. No zero-unexplained claim.
- **Arbiter approval:** none. Scratch/candidate outputs never blessed.
- **Coverage:** ordinary SDXL denominator 1,930 constructed modules. Candidate facts include 70 FFN mechanisms, 60 external-context connections and 17 residual-cell arithmetic entries. These overlap and are not coverage percentages or proof of complete cells. Proven/flagged/silent per witness and deltas versus S7 are not established; silent=0 is not claimed. Accepted S7 artifacts are unchanged.

## Verification and poisons

Committed-tree focused lane: **104 passed, 2 deselected**; tracked-file fingerprints identical before/after (`d8295cc669f0dc7a5b2223f2e73645320b3ba0dab4763991c28102046a674764`). Command and output are in `focused/result.json` and `focused/pytest.log`. This was the only test lane running in the isolated worktree. It is not the broad acceptance bracket.

Controls cover source/checkpoint mismatch; existence proof used for connection; exact primitive identity; placement separate from fact qualification; source override wrong hash, unused, preloaded and duplicate import addresses; possible member replacement; and legacy flag routing. Separate deliberately failing pre-fix red transcripts were not captured for this stop receipt; that completion requirement remains open. See receipt `poisons.md` for exact tests and limits.

The class-lookup counterexample's assertions passed; raw sources, inventories and observations are in `counterexample/`, with `reproduce.py`. It exposes a missing proof boundary, not an accepted reader. Ordinary HTML was generated on the working candidate; `source-manifest.json` matches the isolated snapshot. Only the later focused lane is a committed-tree test.

## Deletion, growth and open debt

- Production lines added/deleted: **2,428 / 22** (`model_unfolder/` and `physics/` versus `83140f1`), see `growth.json`. Snapshot total including tests/scripts/packaging: **3,103 / 28**.
- Old production authority entry paths quarantined: **1**. Legacy authority files deleted: **0**. Deletion responsibility remains S8.
- Temporary comparison flags added/retired: **1 / 0**. Generic one-time source override capability: **1**, not a per-model hook.
- Debt eliminated: no accepted family-matrix count. Class-1 execution/fact findings are not declared closed; observation integration remains S8 work.
- Runtime dependency/packaging changes need clean-wheel verification. Final proof-summary completeness, constructor defaults versus applied values, and visual review remain open.

Owner/arbiter: decide the bounded class-lookup prerequisite under U11 §9. Executor/S8: after disposition, finish it and all failed exit rows, deletion, witness deltas, three traces, six conditions, matrix and broad bracket. Obtain Soumil's explicit output-delta approval before re-bless. Other families' S7 closure continues separately.

**S8 is STOPPED, not DONE. This is a status report under 15 §4.**
