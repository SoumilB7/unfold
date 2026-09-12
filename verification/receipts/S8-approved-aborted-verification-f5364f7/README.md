# Aborted verification — f5364f7

**ABORTED_AFTER_KNOWN_FOCUSED_FAILURE.** The focused lane completed with 510 passed and one failed: the S7 matrix freshness check detected 19 approved corpus review-metadata fixture hash changes. The exact source delta and full failure log are preserved. The owner authorized termination before the remaining expensive work.

Authority checks were interrupted and remain partial. Full tests and preservation were not run. No completed inner coordinator receipt exists, and no global fingerprint or authority verdict is inferred. The static command exited zero. The selected external-input records cover 59 installed sources and one cached task config and compare exactly before/finally; this is explicitly narrower than a completed global gate.

The outer receipt records `KeyboardInterrupt()`, 451.399 seconds elapsed, and a null exit-code field. The executor separately reported launcher exit 130; `summary.json` attributes that value rather than rewriting the original receipt. The retained abort census confirms all six owned PIDs were gone.

All 13 released files from the outer wrapper, partial inner coordinator, abort census, static check, and exact source diagnosis are preserved losslessly. Every complete logical path has `.gzip` appended; no extension replacement or collision is possible within the verified injective map. Sizes, raw/stored SHA-256 hashes, and decompressed byte comparisons pass. Original files remained unchanged.

From a clean checkout:

```sh
python3 verify_restore.py
python3 verify_restore.py --out /private/tmp/s8-aborted-verification-restored
```

The destination must not exist. Restoration preserves the original logs and state under separate `outer/`, `inner-partial/`, `abort/`, `static/`, and `diagnosis/` directories. This is an aborted-run receipt, not a combined or completed verification result.
