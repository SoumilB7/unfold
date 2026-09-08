# Completed workers2 verification — c207550

**COMPLETED_PASS.** The fresh whole coordinator exited zero on commit `c2075501e4b0cf7e97cc004c3d913011e64e3632`, with all six lanes passing and no missing lanes. All 26 full-test batches completed: **4,445 passed, zero failed, 14 skipped, 2 xfailed** (4,461 outcomes). Preservation passed all 52 tests in 558.62 seconds of pytest time.

The full lane used two workers, while the separately scheduled focused, authority, and preservation lanes retained their recorded worker counts of one, three, and four. Outer elapsed time was 5,899.555 seconds; the inner coordinator recorded 5,899.203 seconds. Exact lane-wrapper and pytest durations remain separately available in the raw receipts and logs.

The coordinator, source-artifact, and every lane's before/after fingerprints compare equal. The selected external-input maps cover 59 installed sources and one cached task config and also compare equal; this is not an exhaustive external package/cache census.

All 44 streams preserve the complete outer wrapper, inner coordinator, 26 bracket logs and schedule, and five execution-preparation files. The archived runner bytes match the actual command record's runner SHA. The archived manifest with the runner's explicit `--workers 2` addition expands to the exact actual command; the archived selected-input union matches the actual before/after maps. Individual input JSON byte hashes are archival pins, not invented separately persisted runtime hashes.

`summary.json` derives the full-test totals directly from the 26 terminal batch summaries. The earlier red whole run and isolated diagnostics remain separate, unchanged evidence. This passing run does not retrospectively prove the original failures' cause.

Each complete logical path has `.gzip` appended. Injective mapping, every compressed/raw size and SHA-256, decompressed byte equality against the released sources, and full restoration were verified. No original run outputs were modified.

From a clean checkout:

```sh
python3 verify_restore.py
python3 verify_restore.py --out /private/tmp/s8-workers2-full-restored
```

The destination must not exist. Raw records restore into `outer/`, `inner/`, `bracket/`, and `execution-preparation/` namespaces. Historical absolute paths are retained as provenance and are not needed for archive verification.
