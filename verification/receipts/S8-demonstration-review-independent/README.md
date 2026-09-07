# Independent S8 demonstration/report audit

Reviewed scripts at immutable `4637ef4b77a8139326a8643b9f7eb68ab9d0f93c`, `/private/tmp/unfold-s8-ports-review`. Scope: source inspection and a report-only probe against persisted ordinary artifacts. No model runs, pytest, production edits, or output acceptance.

The generator is a useful experiment harness. It generates actual HTML, records static/runtime/scratch root hashes, raises on a same-source mismatch, preserves changed source bytes and a source diff, and refuses production-source drift during a run. The reporter explicitly emits `review_required`, `blessed: false`, and `named_reproof_candidate`; these are appropriately not accepted semantic proofs. Its individual PASS checks nevertheless have the following concrete gaps.

## Reproduced report gaps

1. **Actual artifact bytes are not rechecked.** `scripts/report_s8_demonstration.py:156–160` loads JSON sidecars without reading `page.html`. Line 171 treats a nonempty stored hash as proof of actual HTML; lines 198 and 218 use stored visual/hash observations. Removing the page from a temporary copied case still yields `actual HTML generated: PASS`. An unchanged case can likewise receive byte-identity PASS with no page bytes present. This does not show the original generator failed to render; it shows the report cannot detect stale, missing or mixed review artifacts. Minimal closure: hash the actual pages and regenerate or validate their observations before reporting those checks.

2. **The report trusts a same-source boolean even when its displayed hashes disagree.** Line 215 accepts `same_source is True`. The probe assigns three different hashes and retains the boolean; the displayed same-source check still passes. The generator itself correctly compares all three hashes at `scripts/demonstrate_s8_unet.py:326–331`. Preserve that positive enforcement, and have the report also compare the saved hashes to each other and to its archived scratch source bytes. No new model run is needed to verify artifact integrity.

3. **A missing canonical fact-to-block link is not a trace gap.** `claim_traces()` selects a block by occurrence at lines 348–360, but does not require the trace's fact key in that block's `source_fact_keys`. Clearing every block's fact citations produces the same `chain_gaps` as before for all three traces. These remain `REVIEW_REQUIRED`, so this is an incomplete linkage diagnostic rather than a false final acceptance. Add the exact fact-to-block citation check; review the drill's actual mechanism correspondence separately.

4. **The fixed semantic selection will omit a new primary-state fact.** `SEMANTIC_FACTS` at lines 19–24 is a closed list. The probe changes an additional `root.denoiser.primary_state_routes` value while retaining the same IR and page, and the reported semantic facts remain equal. That name is a prospective example, not a claim about a fact already present at this checkpoint. Full IR deltas are still retained and never automatically blessed; however, that does not make the condition's semantic-equivalence PASS cover an excluded projected fact. Include all projected family mechanism/value/relation facts with explicitly documented metadata-only exclusions, or update the list with a coverage assertion when the primary-route fact lands.

Exact probe sources and outputs are in `reproduce_report.py` and `results.json`. Inputs are the saved d3a4d61 ordinary artifacts, whose schema is compatible with the checked report. That older run predates per-fact archive maps, so its pre-existing archive checks fail; the probe makes no claim that the entire condition passes. It establishes the individual false PASS labels and the unchanged trace-gap list. The complete newer archive implementation is assessed by source inspection, not by pretending those historical artifacts already contain it.

## What the six conditions currently establish

| Condition | Positive implementation | Remaining acceptance obligation |
| --- | --- | --- |
| Ordinary | A real `Diagram.to_html()` result and shape-derived totals are persisted. | Inspect actual page bytes, full stage connections and the claimed card/drill content. |
| Sparse valid config | The harness omits only supplied values equal to supported source defaults; it refuses an empty omission set. The report checks class-default fact provenance and exact omitted key/value text on the denoiser card, plus selected semantics/structure/SVG preservation. | Include every newly projected semantic fact; validate actual HTML observations. This is a bounded valid sparse witness, not a claim that all arbitrary omissions preserve construction. |
| Misleading field | The harness inserts previously absent `hidden_act=relu`; the report requires nearby visible non-consumption language and unchanged selected semantics/structure/SVG. | Confirm the flag describes that field and no excluded new mechanism fact changes. A prose-window check alone is not a consumer-authority proof. |
| Equivalent rewrite | The harness rewrites actual local-name AST sites in scratch root source; static and runtime root hashes must match it. Selected semantics/structure/SVG must stay equal. | Ensure new projected facts participate, and inspect any retained full IR delta. Evidence-address normalization is explicitly stated. |
| Real computation change plus unchanged control | A unique constructor source site increases a real repeated transformer count. The report requires changed facts, drawing and total parameters, and finds each added FFN's block, drill SVG, subtree parameter text and new parameter paths. The unchanged copy compares the saved HTML hashes. | Verify actual bytes, then match the specific added computation to its actual drill and evidence. Existence of any SVG on a card does not independently prove the new mechanism is what it draws. All deltas remain owner-review candidates. |
| Missing evidence | The harness removes the exact previously observed nested static dependency. The report requires fewer FFN proofs and a visible limitation on each affected occurrence's card. | Confirm that the missing proof actually limits the corresponding drawing. A lower fact count plus a missing-evidence chip does not by itself reject a stale confident FFN drill still shown beside the chip. The report does not currently assert that negative correspondence. |

## Three claim traces

The chosen traces—FFN computation, cell-member connections, and a spatial operation—are relevant. They expose canonical fact values, typed proof summaries, archived-source links, the overview stage, actual card/drill presence and shape-backed numbers. They do not automatically re-prove source semantics, and correctly say so. Completion still needs the sheet's human-readable per-claim implementation → connection → fact → overview → drill → numbers explanation and reviewer inspection of the exact source/diagram correspondence. Merely obtaining three JSON rows or empty `chain_gaps` is insufficient, particularly until the omitted citation check is closed.

This follows the user's S8 item 5 and `z-docs/10-full-research/14-confirmation-checklist.md:157–158`; it does not expand the demonstration into a standing per-model framework.

## Reproduce

From `/private/tmp/unfold-s8-ports-review`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-demonstration-review-independent/reproduce_report.py > /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-demonstration-review-independent/results.json
```

Exit 0. Temporary review copies only; original artifacts and the immutable checkout remain untouched.
