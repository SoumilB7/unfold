# Guarded staged reproof — 710da8e

All 19 guarded staged bless operations passed. Each operation performed exactly one mandatory actual offline reproduction and returned the original report through its observer. The recorded source, external-input, and execution-input pin checks all passed. No live baseline was changed during this phase.

This lossless archive contains all 19 completed result directories and exact per-model logs, plus the exact executed `bless_one.py`, approval overlay, gallery input manifest, selected external-input manifest, and 19 independent approval records. `summary.json` derives its per-model status and counts from the actual result and pin-check records. The ongoing expected-output generation directory, log, and changing staged outputs were excluded.

`artifact-map.json` maps 232 complete logical paths to separately stored streams by appending `.gzip`. Raw and stored sizes and SHA-256 hashes are recorded. Every stream was decompressed and compared byte for byte with the released original. All source files remained unchanged during packaging. Gallery inputs are pinned in each operation's original input record and separately archived in `S8-approved-current-gallery-710da8e`.

Verify or restore from a clean checkout:

```sh
python3 verify_restore.py
python3 verify_restore.py --out /private/tmp/s8-guarded-reproof-restored
```

The restoration destination must not exist. Actual per-model reports and results restore under `results/`, stdout/stderr logs under `logs/`, and exact execution inputs under `execution-inputs/`. Historical absolute paths remain unchanged as provenance; they are not needed to verify or restore the archive.

This receipt records the completed staged reproof phase. It does not claim completion of later expected-output generation, installation, or final verification.
