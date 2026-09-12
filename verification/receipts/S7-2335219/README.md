# S7 final cross-platform closure receipt

- verified commit: `2335219d22f68c2c8452ebe9fc39c9b7ff71ef8d`
- coordinator run: `17253e6fff`
- Linux quality run: `34016241824`
- result: **PASS; every local lane green and fingerprint-identical; every Linux
  quality step green**
- product delta: **none**

| lane | result |
|---|---:|
| collection | 4,230 node ids; every runnable test assigned exactly once |
| focused S5/S7/Linux | 237 passed |
| U2 authority | 44 passed |
| full | all 24 fresh-process batches passed |
| preservation | 52 passed; 29/29 witnesses byte-identical |
| static | clean; five changed Python files |
| Linux namespace | 39/39 live shadows matched |
| support denominator | 29+15; no silent failures |
| reviewed examples | byte-current |

The first local attempt used ten intra-lane workers and one bounded relation
observer timed out after five minutes under contention.  The exact test passed in
7.36 seconds alone.  The final receipt used four workers; the formerly failing
batch passed in 116.8 seconds and the complete bracket passed.  The failed run is
not cited as evidence.

Linux exposed two successive portability defects before the final green run:
absolute static source paths in the persisted source seal/receipts, then host
environment values and the process-local source-index fingerprint inside relation
plan identity.  Both were fixed at their producers.  No semantic payload field
was broadly ignored: persisted static provenance is now content hash + line, and
relation observation identity still covers every execution fact while replacing
only source location and the declared host environment fields.

The machine record is `receipt.json`; exact compressed local lane logs are in
`lanes/`.  The independent Linux receipt is GitHub Actions run `34016241824`.
