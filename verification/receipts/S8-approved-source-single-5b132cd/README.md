# Isolated source-freshness retry — 5b132cd

The previously failed S7 matrix freshness target passed: **1 passed in 11.34 seconds**. The wrapper recorded 14.186 seconds and exit zero. Its exact source/input before/after maps and selected 60 external-input maps compare equal. This is one isolated retry, not a completed full verification gate.

The actual command records base commit `5b132cd75052196fcba9ba0902e424b46283c90e` with only corrected `verification/s7/matrix.json` source metadata in the isolated checkout. The exact matrix input SHA-256 is `4b5696bcef94f08da5c95fce8b8d0f0091b50eb285e219a8673c89d72b36accd`. The separate source-refresh receipt and independent review document the 19 metadata hash changes; this archive does not infer product changes or combine earlier test results.

All seven original run files, the exact executed `run_single.py`, and the exact matrix input are stored as nine lossless streams. Each complete logical path has `.gzip` appended. Unique mapping, compressed/raw sizes and hashes, decompressed byte comparisons, and source stability were verified.

From a clean checkout:

```sh
python3 verify_restore.py
python3 verify_restore.py --out /private/tmp/s8-source-single-restored
```

The restoration destination must not exist. Original commands, logs, and pin maps restore under `run/`; the executed script and matrix restore under `executed/` and `input/`.
