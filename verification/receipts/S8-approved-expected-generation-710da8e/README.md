# Actual expected-output generation — 710da8e

The unchanged canonical writer generated the expected manifest for all 29 witnesses in 1,275.505 seconds. The actual output SHA-256 is `84f0df7ade2f311adefbbb14f8ac0732b941f98d6110723f6554530a1d14b390`. Recorded production-source, corpus/gallery, external-input, and tool-input equality checks all passed. This phase did not mutate the live baseline or regenerate the historical baseline.

The 13 losslessly archived streams preserve the complete released generation directory, exact stdout/stderr log including warnings, and the executed `build_expected.py`, approval overlay, and selected external-input manifest. Original before/finally records are retained exactly; the corpus/gallery before snapshot and final equality result remain separate as authored.

`artifact-map.json` appends `.gzip` to each complete logical path without replacing extensions. Every stored and decompressed size/hash was verified, every raw stream compared byte for byte with its released source, and all source bytes remained unchanged during packaging. `summary.json` contains the actual result and pin checks without inferring later installation or verification status.

From a clean checkout:

```sh
python3 verify_restore.py
python3 verify_restore.py --out /private/tmp/s8-expected-generation-restored
```

The destination must not exist. The exact generated manifest restores under `generation/`, the full log under `logs/`, and executed inputs under `execution-inputs/`. Historical absolute paths are preserved as provenance; verification and restoration do not require those paths.
