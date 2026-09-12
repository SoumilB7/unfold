# S10-1 — UNet authority deletion and measured latency

Status: implementation, independent source review, corrected census checks and 52/52 preservation complete; owner approval of the exact source stamp, deferred S7 artifact checks, standalone commit and final unit acceptance remain pending. This receipt does not claim S10-1 DONE and does not bless any output.

The accepted instance/source projection is the only UNet production path. The old structural adapter, differential flag, five legacy renderer entrypoints, five quarantined raw-source readers and their dead parser were deleted. No replacement mechanism was added. An incomplete construction still produces its typed limited result.

## Exact production parity

Eight witnesses were run in fresh processes on isolated `fff99e22` and deletion-only trees: SDXL (corpus), SD-v1-4 (unseen), published Kandinsky, the explicit IF default-head control, the two original invalid head inputs, Llama and PixArt. All **64 raw surface comparisons were byte-identical**: input, full IR, structural IR, ledgers, HTML, expanded JSON, parameters, and typed failures. The two invalid configurations remain specifically `ConstructionFailed` at `construct`, naming `num_attention_heads`; unrelated environment failures cannot count as preservation.

Raw before/after payloads, logs, experiment namespace, source fingerprints and the exact helper are in `artifact-map.json` under `deletion/parity/`. These are experiment outputs, never blessed examples. The actual new SDXL/SD14 pages are presented at the root workspace's `z-docs/12-design/S9-unet/index.html`.

## Verification

- Changed-file static checks passed.
- 84 focused deletion, routing, quarantine, debt and consumer controls passed.
- The authority lane first passed 43/44; the remaining test correctly required removing the stale four-broad-exception baseline for the deleted parser. The corrected exception file passed all four tests. All original outcomes are retained; the full authority lane was not unnecessarily repeated.
- 45 latency contract and artifact-link controls passed, including downward arithmetic and upward-ceiling poisons.
- All 12 fresh-process latency samples passed the existing runner and cold/warm cache/import checks.
- Executable-symbol census and no-fallback import grep passed with zero hits. The retained `block_views/unet.py` is the source-backed replacement renderer, not the deleted adapter.
- Every scoped lane and the parity campaign retained identical before/after source/artifact fingerprints. No preservation baseline, example or active S7 artifact was changed.

The first static/focused attempts exposed ordinary deletion bookkeeping: an unused local, two extinct structural-surface keys and three extinct extras-writer debt rows. Their failure logs are retained. No gate or mechanism comparator was weakened.

## S2 latency ratchet

| Target | Cold median | Warm median |
|---|---:|---:|
| SDXL | 28.366631 s | 8.095881 s |
| SD-v1-4 | 26.493520 s | 8.002396 s |

Three samples per mode and target; every sample retained, including SDXL's 33.809460 s cold sample. The 30 / 9 median targets were met on this host. The S2 method gives `ceil(1.3 × slowest median)` = **37 / 11 s**, ratcheted down from 43 / 13. The recorder checks each new gate against the previous accepted gate before writing. The schema permits downward computed budgets under the original ceiling so historical receipts remain checkable.

These are measured costs, not a claim that deleting dormant code caused every observed improvement. Host load, full timing boundaries, cache evidence and actual tool/source hashes remain linked to the original samples. The measured tree is explicitly `fff99e22` plus the archived deletion patch and source pins. Only public package import and HTML rendering remain outside the library budget, per the existing S2/S8.1 contract.

## Concrete approval delta — not installed

Only `verification/s7/matrix.json` needs freshness renewal for this unit:

- Current SHA256: `47f372e4629711fd51676d3005a522cdd10d6ad2f31a0dd4d12c1360490014d1`
- Candidate SHA256: `efe7d97e2504c38d3679e05521f556068284fe638ad9cb83c646cdd196dd8b2c`
- Eleven source entries change, including three deleted source files. Every non-source matrix field, all 39 model rows, and all 117 model/observation/relation payloads remain byte-identical.

The unchanged generator produced the source-hash map on the isolated deletion tree. Exact old/new candidate bytes and each changed source/hash are archived under `deletion/s7-stamp-candidate/`. The earlier source-stamp verdict applies only to the superseded candidate. Independent review accepts the corrected census and revised source stamp; see `closure/independent-census-fix-review/verdict.json`. Soumil's approval is still required before the guarded installation. The active S7 source stamp therefore remains intentionally stale against these uncommitted source changes; no broad-green claim is made.

## Growth and open items

Production: **36 lines added / 2,204 deleted**. Two legacy modules, five legacy renderer entrypoints, five quarantined readers, one raw parse site, three extras-writer debt rows, four renderer debt rows, and four broad exception catches removed. No temporary bridge added; the differential bridge retired. Unresolved configuration accounting remains explicit.

The former legacy-only tests were removed with their dead targets. The exact-source topology rename tests, cell mechanism controls, nested FFN/attention tests and the current production occurrence/card tests remain. The deletion routing poison additionally proves limited construction cannot revive a legacy authority.

Open: owner-approved guarded S7 source renewal, deferred S7 artifact checks, standalone commit and final unit acceptance; S9-owned mechanism/config accounting; S11 N1/N2 module/cache layering and the two local diagnostic fingerprint exporters. No S9 behavior is included in the isolated S10 candidate.

## Broad-bracket correction and revised candidate

The original isolated bracket collected 4,766 tests in 28 batches: 27 batches passed; batch 012 failed only `test_no_new_raw_structural_extras_write_grows_silently`. The live canonical UNet projection still transports `render` and `unet` extras. Removing legacy writer debt did not eliminate that transport. The static scanner previously missed inline `ModelIR(extras={...})`.

The correction detects those inline writes and dictionary spreads, pins the two actual `project_unet` transport writers and their exact remaining consumers, and records their debt. No IR, renderer, fact or parameter producer changed. The corrected isolated tree passed all **62 targeted census, registry, exception and quarantine tests** in 127.56 s; lane duration 135.616 s. Source fingerprint `3137964b9140d1485eee9b583d28b252c5ff7c1390a2b7fc1bb786b388258005` and artifact fingerprint `5aed5480485a877c5393ec4eb97be56cf2709327baa35ede0da54884029bd97e` are unchanged across that lane.

The original red bracket, corrected source manifest/patch, full targeted logs and revised stamp candidate are retained under artifact-map entries `deletion/broad/`, `deletion/census-fix/` and `deletion/s7-stamp-candidate-v2/`. Earlier parity and latency measurements remain tied to their original exact source pins; this metadata-only correction does not rewrite those measurements. The dedicated preservation lane passed **52/52** in 308.65 s (309.820 s supervised), with the same source/artifact fingerprints unchanged; the deferred S7 artifact file requires owner-approved stamp installation. No whole-unit green claim.

The old source-stamp candidate `a08d366d018626ab87b69230db977c46a117634aa98bbd8392830bf586d1d490` is superseded, never installed. The current candidate is `efe7d97e2504c38d3679e05521f556068284fe638ad9cb83c646cdd196dd8b2c`; all 39 rows, 117 payloads and non-source fields are still unchanged. Independent verdict accepts the revised candidate; Soumil approval remains pending.

Preservation closure: corrected isolated candidate `c05c1911…` passed all 52 preservation checks; no baseline was changed. Independent census/stamp review is accepted. Soumil has been asked to approve current source-stamp candidate `efe7d97e…`; installation and the exact S7 artifact checks remain pending.

Uninstalled proposal validation: the unchanged S7 `check(output=...)` passes on a scratch copy containing the revised source stamp and byte-identical payloads (8.661 s supervised, same source/artifact fingerprints unchanged). Active `verification/s7/matrix.json` remains byte-identical to HEAD. This validates the proposal without approving or installing it. Full active S7 artifact tests remain deferred until owner approval.
