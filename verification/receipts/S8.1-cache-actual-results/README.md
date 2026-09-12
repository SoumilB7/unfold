# S8.1 actual cache, observation, and public diagnostics

This receipt preserves **2,055 original streams** from the 31 expressly released diagnostic/preparation/cache directories, including 15 adjacent logs and the source snapshots already present inside those directories. There are **261,327,132 original bytes**, stored in **42,482,943 gzip bytes**. All 2,055 streams were restored to separate files and verified by SHA-256 and byte count. Exact file/directory membership and still-retained original bytes were checked again after packaging.

`artifact-map.json` SHA-256: `9537f7e6c9117601ce22a1f48fb783bbd2fa234e89d178448577c4fb9bafbef9`.

The mapping appends `.gzip` to each complete logical filename; it never removes an existing suffix. Every original filename, including any already compressed file, therefore has a distinct storage address. Compression used gzip level 3 with a zero timestamp. `inventory.json` and the map retain all 136 directories, including the explicitly empty `public-entries-v5` cache. No enumerated source was missing; there were no symlinks. Paths mentioned inside historical manifests were **not** imported as extra streams.

`outcomes.json` extracts literal status/error/cache-disposition fields from the retained actual records; full results and logs remain archived. The attempts remain separate:

| Actual attempt | Recorded disposition |
| --- | --- |
| Cache cold v1, v2 | Successful inventory parity; dependencies ineligible for storage, so these remain cache misses. |
| Cache cold v3 / warm v3 | Successful inventory parity; actual stored historical capture followed by its actual cache hit. |
| Cache unit v2 | Original exit 1 preserved. |
| Cache units v3, v4, v5; reader unit v1 | Original exit 0 results preserved. |
| Import probe v1 | Original `getset_descriptor` JSON-serialization failure preserved. |
| Import probe v2 | Recorded error is null; exact diagnostics preserved. |
| Observation cold / warm v4 | Successful actual cold observation stored, followed by historical cache reuse. |
| Public hit v3 | Original assertion failure preserved; actual diagnostics show a miss and fresh worker, not the intended hit. |
| Public hit v4 | Original “preparation identity must already be populated” failure preserved. |
| Public hit v5 | Original “actual public identity not populated” failure preserved. |
| Public cold v5 | Original assertion failure and actual worker outcome preserved; its cache directory is empty. |

These are individual diagnostics, not the completed three-pair latency campaign or proof that a latency budget passed. A historical cache hit is not new live mechanism evidence. This archive does not reconstruct the cause of any earlier run, rerun a model, render a page, execute tests, or alter a baseline.

`archive.py` is the exact executed packaging program. `result.json` records its hash, complete restoration checks, original-source equality, and archive-only duration. Existing original outputs remain untouched. To inspect any artifact, locate its `stored` path in the map and decompress that `.gzip` stream; the map gives the expected restored SHA-256 and size.
