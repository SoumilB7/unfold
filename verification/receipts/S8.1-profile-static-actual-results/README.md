# S8.1 retained profiling and static diagnostics

This receipt archives only the 26 expressly released profiling, child-profile/phase, static-focused/replay, metadata, preparation, and independent-review directories listed in `inventory.json`, plus their 12 existing adjacent logs. It contains **248 streams**, **178,734,388 original bytes**, and **12,398,120 compressed bytes**. All 248 streams were restored to separate files and checked by SHA-256 and byte count. Exact restored membership across 50 directories and the still-retained original membership/bytes were verified after packaging.

`artifact-map.json` SHA-256: `77026faed6b75b5b9061945d0e2f25a1f047f5257cb9718677e8750737a51c2d`.

Storage appends `.gzip` to each complete logical filename, including existing suffixes, using serial gzip level 3 and zero timestamps. Every storage path is unique. No named directory was missing; there were no symlinks. Profiles and pickles were treated strictly as opaque archival bytes: they were never executed, loaded with a profile reader, or unpickled. No external file referenced by a manifest was imported into this archive. Candidate/worktree directories and the separately archived value-hash, reader, and cache records were excluded.

`outcomes.json` retains literal status/error/return-code and qualification fields from the original result JSON. The history remains separate:

- The child cProfile and lightweight child-phase attempts retain their failed status and failed inventory-parity assertion.
- The initial static-focused and metadata-focused attempts retain return code 1; their corrected versions retain their actual return code 0.
- The metadata child result retains its successful inventory parity and unchanged pin checks.
- Baseline profiling, historical-inventory static replays, and the hash-count diagnostic retain their original artifacts, counters, source/input identities, and stated measurement limitations. Historical-inventory replay is not fresh model construction.

This is an archive-integrity result, not a reinterpretation of those outcomes. Profiling and observer overhead, partial phase snapshots, overlapping cumulative times, and the distinction between reaped child CPU and intrinsic worker timing remain in the original records. No latency median, budget acceptance, fresh model result, historical failure cause, or performance improvement is inferred by packaging.

`archive.py` is the exact executed packaging script. `result.json` pins it and records restoration/source-equality checks. Locate an artifact's `stored` path in the map and decompress that stream to inspect its exact original bytes; the expected restored hash and size are recorded alongside it. All original temporary outputs remain untouched.
