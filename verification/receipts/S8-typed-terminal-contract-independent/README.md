# Independent typed terminal contract review

ACCEPT within the frozen15-file source scope, based on source and test-body inspection. This is not a claim that independent tests, new model runs, final output equivalence or broad gates passed. No production edits, models or pytest were run. Source-pins.json identifies exact working-tree bytes on base4fdc741; source-snapshot.json.gz preserves them. Executor may release the freeze.

## ConstructionSummary

The optional frozen record in ir.py:489 has required fields, nonnegative exact-int quantities (bool is rejected), nonempty scope/citations, and no numerical defaults. ModelIR reconstructs a supplied dictionary through this constructor, rejects extra/missing keys by the constructor signature and rejects a different object type; to_dict omits the record when absent. Existing unrelated dictionary shapes remain unchanged.

The producer in evidence/construction_summary.py:26 requires actual EvidenceFact instances with exact ledger IDs, code_proven status, expected claim kind and exact existing proof class. It invokes validate_fact_claim; serialized-looking objects, matching citation strings or a value-only record are insufficient. It joins the three facts by their complete inventory values, verifies every constructed stage address is in the population, and requires exact shape/population address sets. Shape totals are re-derived by the existing identity-aware shape proof, including shared parameter aliases. The record adds neither active-path quantities nor mechanism assertions.

The stage proof previously caches a mutable value. This new boundary reconstructs that existing proof afresh from its graph/bindings and validates the fact against the fresh derivation. Mutating both the fact's shared stage list and the old cached comparison value therefore does not qualify a changed stage count. Shape and population proof values already derive freshly. This is a bounded revalidation of the existing proof contract, not an alternative source reader.

The reverse audit recomputes the qualified record and checks all quantities, scope and citation keys. It is called before the successful cutover returns (unet_cutover.py:87) and by projection_claims_from_product (reconciliation.py:955). Thus the helper is integrated into production and the family projection audit; merely adding its unused definition would not have sufficed. Unsupported/missing or altered summaries are rejected by that audit. Params and the header read only the projected IR record; they no longer read shapes/stages from extras or reinterpret source/config. Unknown active/embed/output values stay None.

## ComponentEntry

The second optional frozen record (ir.py:518) validates nonempty root/title/subtitle, unique nonempty input IDs distinct from root, rejects a scalar-string ID sequence and normalizes JSON lists into tuples. ModelIR dictionary construction and to_dict preserve the record; absence is omitted.

blocks.component_entry_for_handoffs authors the interface when explicit component-presence evidence is incomplete. The successful UNet projector uses the same scoped handoffs that restrict visible supplied components and its actual root formal IDs. Full supplied pipelines omit the record. The limited source/instance fallback uses the opaque render plus the same handoff-based component entry in _parse_projected_denoiser; it does not invent root input IDs. Existing supplied partial components remain independent cards. No class-name or source lookup is introduced in consumers.

expanded.__init__ passes the typed entry explicitly to build_sampling_loop; a component entry prevents a fabricated sampling-loop JSON structure. HTML reads root/input IDs from the same projected entry and uses its title/subtitle. The component-only header now says DENOISER COMPONENT instead of SAMPLING LOOP/Denoiser applied iteratively. That is a named user-visible correction which still needs final actual-output evidence and delta review; it is not silently identical output.

## Census and debt

The source pins add exactly the two reviewed author identities (typed-ledger writer and existing ModelIR projector) and two owner-authorized ModelIR field vocabulary/identity rows. The new fields do not create a parallel IR. The four added raw-extras read groups are addressed at their consumers instead of new debt fingerprint rows: parameters/summary quantities use ConstructionSummary, and both component-scope consumers use ComponentEntry. structural_debt.py is unchanged in this reviewed worktree. No raw-consumer allowlist growth or scanner exclusion was introduced.

Inspected tests cover shared-alias count, strict roundtrip/absence/bool counts, changed quantities/citations, unqualified records, cross-inventory join, cached stage payload mutation, competing extras, component-only/partial/full/opaque paths, invalid entry IDs, and the old sampling recurrence header negative. These test bodies support the stated intended checks; executor owns their execution and the full live census result.

Remaining evidence is operational: exact focused/structural results, actual final component and pipeline pages, per-output schema/header causes, refreshed coverage/matrix linkage, and preservation/broad results. This receipt grants no baseline blessing or S8 DONE status.
