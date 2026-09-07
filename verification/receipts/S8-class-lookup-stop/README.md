# S8 class-lookup stop receipt

**STOPPED / NOT DONE. No output was blessed.** Candidate tree: `bb48a31e336c1721f0a111862bd081acf390bb25`, isolated branch `codex/s8-unet-stop-review`, base `83140f1`.

The complete status sheet is mirrored in [status-sheet.md](status-sheet.md). The live sheet and actual SDXL decision page are outside git at workspace `z-docs/11-execution/S8.md` and `z-docs/12-design/S8/index.html`, following the existing location ruling.

The required saved-bank-to-concat input proof crosses an optional FreeU branch. Instance-field absence does not prove Python class lookup takes its default. Current ClassRecord lacks conditional class-body member addresses/complete namespace coverage. U11 §9 requires stopping before adding a new neutral observation record. No such extension has been implemented. The sheet proposes a bounded prerequisite for owner disposition.

## Evidence

- [source-boundary.json](source-boundary.json): actual installed up-stage FreeU/concat source excerpts with line numbers and file hash.
- [counterexample/result.json](counterexample/result.json): two classes with identical constructed populations and no instance scalar field, but different observed arithmetic due to a conditional class member. Sources, inventory and observation records are adjacent. Trace absence is never used as static negative proof.
- [ordinary/result.json](ordinary/result.json): actual full production-candidate SDXL parse/render, 1,930 constructed modules, seven stages, 2,567,463,684 denoiser parameters. Root static/runtime source hashes agree. Wiring checker returned no problems. Adjacent gzip files contain unmodified HTML, IR, facts and inventory; config is `input.json`.
- [focused/result.json](focused/result.json), [focused/pytest.log](focused/pytest.log): isolated committed-tree focused run, **104 passed / 2 deselected**, tracked fingerprints identical. Not a broad gate or preservation verdict.
- [poisons.md](poisons.md): focused controls and missing red-transcript qualification.
- [source-manifest.json](source-manifest.json): SHA-256 of the 41 candidate source/test/script/packaging changes, matched to the isolated snapshot before testing.
- [growth.json](growth.json): production growth and outstanding deletion.
- `artifact-manifest.json`: byte hashes for receipt contents, excluding itself.

Ordinary rendering happened on the working candidate; the source manifest matches the later isolated snapshot. It was not a committed-tree render. The synthetic counterexample was also run before the snapshot. The focused tests were run on the immutable commit. The receipt was added after that lane finished.

## Reproduce the counterexample

From package root, choose a fresh output directory (script refuses an existing directory):

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 verification/receipts/S8-class-lookup-stop/reproduce.py --output /tmp/s8-class-lookup-reproduction
```

Uses the existing network-disabled inventory/observation subprocesses. The source override is the generic S8 one-time capability, not a model hook. Synthetic success is not the missing SDXL computation-change demonstration.

The focused command and exact file fingerprint are recorded in `focused/result.json`. Required full differentials, all six HTML conditions, three complete traces, family fact/execution closure, deletion unit, packaging check, broad gate, preservation and Linux CI remain open. No complete per-witness coverage or zero-unexplained result is claimed. No push occurred.
