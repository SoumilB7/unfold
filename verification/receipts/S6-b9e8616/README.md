# S6 committed-tree receipt

- verified implementation: `b9e861675873452157d7bed6594b0dd1be9d5ef2`
- coordinator run: `3770d90320`
- result: **PASS; every lane green and every tree/artifact fingerprint identical**
- product delta: **none**
- production consumers of `physics`: **zero**

| lane | result |
|---|---:|
| collection | 4,084 tests |
| focused S6 + kernel | 192 passed |
| U2 authority | 44 passed |
| exhaustive non-preservation remainder | every globally collected test ran exactly once; all passed |
| preservation | 52 passed; 29/29 witnesses byte-identical |
| static | 6 changed Python files; pyflakes and diff checks clean |

The coordinator ran its six lanes strictly serially. Bounded concurrency was
used only inside an active lane. Each lane ran in its own detached worktree at
the committed implementation and retained identical tree and blessed-artifact
fingerprints before and after.

The frozen pilot set contains eight typed inventories and eleven named
observations. SD3.5 block 0 records exactly one SDPA, 16 adds, and 16
multiplies. PixArt records the requested `Transformer2DModel` factory and the
actual `PixArtTransformer2DModel` runtime class. Qwen2-VL's data-dependent full
vision path and DBRX's incompatible whole-model route remain typed failures;
each also has a positive exact sub-recipe. No failed recipe is omitted or
silently converted to a known mechanism.

The machine record is `receipt.json`; losslessly compressed original lane logs
are in `lanes/`. The poison index is `poisons.md`. Pilot artifacts and their
content hashes are under `verification/s6/pilots/`.
