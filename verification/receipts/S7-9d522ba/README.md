# S7 R4 projection/fact-separation receipt

- verified implementation commit: `9d522ba20740e1aa7549309931c6ed69e45e9197`
- coordinator run: `4c8f0ebd44`
- result: **PASS; every local lane green and fingerprint-identical**
- product delta: **none**
- S7 status: **awaiting owner acceptance; B5 and S8 remain locked**

| lane | result |
|---|---:|
| collection | 4,232 node ids; every runnable test assigned exactly once |
| focused S7 + kernel | 284 passed |
| U2 authority | 44 passed |
| full | all 24 bounded fresh-process batches passed |
| preservation | 52 passed; 29/29 witnesses byte-identical |
| static | clean; four changed Python files |

R4 restores the product's authority over the projection axis: a cited block or
render event keeps an occurrence `rendered` or `grouped(parent)`. Unsupported
fact proofs are not erased or promoted. Each is a separate blocking
`investigation_missing / claim_proof_unstamped` finding owned by S9 reader
migration.

The regenerated 39-model denominator contains 44,088 occurrences: 3,051
rendered, 4,740 grouped, 1,700 exact non-architectural containers, and 34,597
projection-unresolved. It retains 710 qualified fact citations and exposes
16,324 unstamped fact proofs. The three reason-class totals are 37,185
investigation-missing, 34,717 structure-unaccounted, and zero
mechanism-unresolved, for 71,902 blocking findings. These totals are recomputed
from each per-model artifact and were not used as classification targets.

Class 3 is now constructible only from a real non-resolved `ReaderResult` bound
to the exact cited fact and one of that fact's declared reader ids. The joiner no
longer manufactures investigation receipts. The receipt's exact compressed lane
logs are under `lanes/`; the machine record is `receipt.json`.
