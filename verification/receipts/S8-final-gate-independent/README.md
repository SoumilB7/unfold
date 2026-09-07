# Independent final-gate review — 5227e9c

Bounded source/receipt review only. No pytest, model construction, production edits, commits, or blessing. `review.json` pins the inspected files and HEAD. The matrix/coverage/broad lanes were pending or active; this is not a claim that they passed.

Verdict: the inspected coordinator preserves its failure boundary. No concrete coordinator path was found that marks a skipped or failed lane green. Final S8 acceptance remains pending the actual results and Soumil's output approval.

## Exact approval boundary

`tests/preservation_expected_manifest.json` is byte-identical to accepted S7 at83140f1. `test_expected_witness_zero_drift_zero_skip` invokes fresh production comparison for each of the29 recorded witnesses (`tests/test_preservation.py:188`, `test_support/preservation.py:357`). Every recorded surface and the complete ordered view list are compared. The known S8 output changes therefore remain expected preapproval comparison failures; this review does not predict that they are the only possible failures. Every actual failure must be attributed from its output before calling it expected.

The coordinator never blesses or allows preservation drift. Keep a failed preapproval run failed, preserve its full evidence, finish unaffected lanes, and stop at the review sheet. The approval boundary is immediately before changing expected output surfaces/signatures/manifests or blessed gallery artifacts to accept these outputs. S8 law5 requires an independent persisted verdict and Soumil's approval of the enumerated per-witness causes before re-bless. Owner technical disposition is not Soumil's approval. After any approved baseline update, verification of the newly committed artifact tree is still required. A sheet with a FAIL or missing artifact remains NOT DONE (§15.4).

## What the coordinator actually guarantees

- `scripts/verify_commit.py:280` runs each phase serially, collecting every lane result even after a lane failure. All four preflight lanes run. Only a fully successful preflight starts the full remainder and preservation (`:356–362`). A preflight failure records both heavy lanes missing. Within the heavy phase, full-suite failure does not skip preservation.
- Every lane runs on a detached named commit, with tree and external-artifact hashes before/after. Missing external artifacts fail staging; copy hashes must match. Exceptions yield return125, not PASS.
- The full remainder excludes exactly the dedicated preservation and four authority files. The file bracket rejects empty/duplicate global collections, partitions every collected node into one batch, and compares each batch's actual collection to its exact scheduled set. It rejects missing/failed batches or changed schedule bytes.
- Overall success additionally requires unchanged coordinator and original external-artifact fingerprints (`verify_commit.py:413–422`). A consumer must not report PASS merely because all individual lane rows say passed; these global comparisons and the process exit remain authoritative.
- The static lane checks only the selected commit's immediate-parent diff (`:174–181`). If the selected final commit contains artifacts only, a green static lane does not mean all historical S8 production edits were linted by that invocation. Report its actual scope; existing prior focused/static receipts can be cited separately.

## Freshness and completeness boundaries

The final lane manifest correctly declares a derived39-target matrix:26 preserved original artifacts (including a separate exact PixArt address replay), one SDXL saved-evidence final-source reconciliation, and12 fresh final-source targets. It explicitly says this was not a fresh39-model run. The resume script asserts source pins, preserved bytes, target sets, exact replay premises, and invokes unchanged `generate_s7_shadow.py --check` afterward. Preserve the referenced premise/replay artifacts in-repo with the final receipt; a temporary path alone is insufficient.

`generate_s7_shadow.py --check` validates the complete declared dependency surface and all39 stored artifact sets, hashes, nonempty denominators, findings and cross-file recipe/relation consistency. It does not run the39 models or recompute every historical proof. Freshness of preserved rows therefore also depends on the separately reviewed carry-forward premises; a green check cannot replace or relabel that lineage.

By contrast, `coverage.py --check` really calls the44-input generator and compares the complete resulting manifest. Its observer wrapper calls the original Diagram renderer and returns its bytes, only capturing SDXL/SD14 outputs. The final result still needs the exact29+15 set, silent0, per-model prior/current values and actual public parse/render capture comparison. The broad coordinator does not itself establish the six C8 conditions, three claim traces, independent differential dispositions, or user approval; retain their separate pinned receipts.

## Required final broad evidence

1. Exact verified commit including final matrix/coverage artifacts, exact coordinator command/flags, coordinator and bracket source pins, environment, serial lane schedule, actual final exit/status.
2. All six lane logs/results; full global collection, bracket schedule and every batch log; collection count and dedicated-lane/remainder ownership. No missing lane or test batch accepted.
3. Every lane tree and external-artifact before/after fingerprint, original external-artifact before/after fingerprint, and coordinator before/after fingerprint. Copy temporary logs/schedules into the committed receipt.
4. Per-witness preservation findings, distinguishing individually evidenced pending-approval output deltas from crashes, new authority failures or unexplained drift. Preserve original red output. Do not rewrite a red coordinator result as a qualified green result.
5. Matrix per-target generation lineage, successful check receipt, preserved-original hashes and source/context/address replay premises; completed44-input generation and live check, per-model coverage deltas and final SDXL/SD14 public render equality.
6. Links to final actual pages, incremental output ledgers, all C8 cases/traces, family execution preservation, deletion/growth ledger, and independent dispositions. Explicitly record that output blessing is pending until Soumil approves it.
