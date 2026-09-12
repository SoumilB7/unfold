# S7 committed-tree receipt

- verified implementation: `b969f342cb344fd5d285197f41f2c49b9d62337d`
- verified matrix tree: `1e99c3aba31a2721628df090216db3556673235b`
- coordinator run: `49c8143aaa`
- result: **PASS; every lane green and every tree/artifact fingerprint identical**
- product delta: **none**
- production consumers of `physics`: **zero**

| lane | result |
|---|---:|
| collection | 4,112 tests |
| focused S6/S7 | 220 passed |
| U2 authority | 44 passed |
| exhaustive non-preservation remainder | every globally collected test ran exactly once; all passed |
| preservation | 52 passed; 29/29 witnesses byte-identical |
| full-range supplemental static/authority | pyflakes clean; identity/taint/exception ratchets 47 passed |

The coordinator ran its lanes serially. Bounded concurrency occurred only
inside one active lane. Every lane ran in a detached worktree at the exact
committed matrix tree and retained identical tree and blessed-artifact
fingerprints before and after.

The shadow denominator is exactly 29 corpus witnesses plus 10 frozen
`TO_SERVE` models. It contains 44,088 runtime occurrences with no silent
drop. The matrix intentionally retains 120 construction conflicts, 34,249
execution-unresolved axes, and 44,088 projection-unresolved axes. These are
blocking findings, not omitted rows or inferred `non_architectural` labels.

The relation controls are exact: Gemma-2 has parameter tying; SD3.5 has no
relation row; Gemma3n has a four-stream residual, ten exact cross-layer reuse
edges, parameter tying and a per-layer side-input bank; DeepSeek-V4 has a
four-stream residual and the `model.hc_head` rank-collapse side head. Nemotron
has 52 members in three encounter-order groups and no invented cycle.

The machine record is `receipt.json`; losslessly compressed lane logs are in
`lanes/`; the poison index is `poisons.md`. Full shadow artifacts and hashes
are under `verification/s7/`.
