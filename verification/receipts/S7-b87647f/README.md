# S7 v2.6 committed-tree receipt

- verified correction chain: `bd3690f2a5c2d6da7583bd8979050530a8de5acc`
  then `b87647f1a6fe381ab2e52983fa086adf67f15721`
- coordinator run: `040e43b266`
- result: **PASS; every lane green and every tree/artifact fingerprint identical**
- product delta: **none**
- production consumers of `physics`: **zero**

| lane | result |
|---|---:|
| collection | 4,125 tests |
| focused S6/S7 | 233 passed |
| U2 authority | 44 passed |
| exhaustive remainder | every globally collected test ran exactly once; all passed |
| preservation | 52 passed; 29/29 witnesses byte-identical |
| static | pyflakes/diff/isolation checks clean; 4 changed Python files |

The coordinator ran its lanes serially. Bounded concurrency occurred only
inside one active lane. Every lane ran in a detached worktree at the exact
committed correction tree and retained identical tree and blessed-artifact
fingerprints before and after.

The corrected shadow denominator is 29 corpus witnesses plus 10 frozen
`TO_SERVE` models. It contains 44,088 runtime occurrences with no silent drop:

- 26,967 unresolved values are `investigation_missing`;
- 34,717 are `structure_unaccounted`, including HunyuanVideo's 120
  construction conflicts;
- 0 persisted rows are `mechanism_unresolved` because every relation candidate
  in this denominator either has its exact proof or does not produce a row.

The zero class-3 count is not a vacuous contract. Permanent positive and
negative poisons construct a valid investigated mechanism unknown and reject a
class-3 row lacking its typed investigation record or concrete reason. A
recipe observation never authors a mechanism fact. The S7 blocking gate now
blocks reason classes 1 and 2; a valid class 3 remains visible without refusing
the rest of the drawing, exactly as plan v2.6 §1g requires.

Projection is now joined from the existing product: typed facts and canonical
IR blocks cite exact static occurrences, which join to runtime paths; exact
closed `torch.nn` children may join through a drawn parent attribute; exact
container runtime types are non-architectural. Anything else remains
`structure_unaccounted` with the exact reason “no product block or fact cites
this occurrence.” Removing a product block in the poison fixture flips its
occurrence back to that state.

Every one of the 39 targets received a callable-signature-derived execution
attempt. The persisted failed-observation diagnostics normalize only Torch's
volatile date/time/PID prefix; the source location, traceback, and exception
remain intact. This closes a discovered artifact-hash nondeterminism without
changing execution evidence.

The machine record is `receipt.json`; losslessly compressed lane logs are in
`lanes/`; the poison index is `poisons.md`. Full shadow artifacts and their
hashes are under `verification/s7/`.
