# S7 v2.6.2 closure receipt

- verified implementation commit: `36f90a64b58f9b57056477e29e3f3892661cc2a8`
- verified static-fix commit: `77d5b6af25ce12cc6815c17dfdd2418435817c2c`
- coordinator run: `e7c7fc406f`
- result: **PASS; every lane green and every tree/artifact fingerprint identical**
- product delta: **none**
- production consumers of `physics`: **zero**

| lane | result |
|---|---:|
| collection | 4,132 tests |
| focused S6/S7 | 240 passed |
| U2 authority | 44 passed |
| exhaustive remainder | every globally collected test ran exactly once; all passed |
| preservation | 52 passed; 29/29 witnesses byte-identical |
| static | pyflakes/diff checks clean; 2 changed Python files in the verified commit |

The coordinator ran lanes serially. Bounded concurrency occurred only inside one
active lane. Every lane ran in a detached worktree at exact commit `77d5b6a`.

## Claim-kind closure

Every fact cited by the persisted projection axis now carries exactly one of the
five closed claim kinds: `existence`, `connection`, `applied_function`, `value`,
or `relation`. The S7 bridge contains 33 mechanism-keyed declarations covering
17,034 occurrence-level citations:

- existence: 604
- connection: 6,493
- applied function: 3,217
- value: 1,200
- relation: 5,520

The bridge is a transition, not a second fact registry: every declaration must
name a registered fact, duplicate rows fail construction, and every projected
fact missing a declaration blocks. Evidence strength is checked per cited fact:
constructor support cannot certify connection, and config-only support cannot
certify an applied function. S9 moves the declaration onto each reader-authored
fact and deletes this bridge.

## Recipe closure

All 39 targets have a persisted typed recipe decision:

- recipe resolution: 37 `ok`, 2 typed `failed`
- execution: 22 `ok`, 17 typed `failed`
- retries: exactly 2, both from the closed grouped-matmul dtype error to bf16
- `no_recipe_attempted`: 0 throughout the reconciliation denominator

PixArt-Sigma and Qwen3-Omni have no exact resolved `forward` on the selected
owner and therefore record `ConfigurationFailed(recipe_resolution)` rather than
claiming a recipe was resolved. DeepSeek-V3 reads checkpoint `dtype=bfloat16`
and succeeds directly. GPT-OSS and Qwen3-VL have an explicitly present null
checkpoint dtype: their first float32 probes hit the exact reviewed grouped-mm
error and their sole bf16 retries succeed. The checkpoint record stays null;
only execution dtype changes.

Recipe dimensions come from the supported config-resolution path, including
class defaults and named bounded calculations. Six conditioning-width spellings
(`cross_attention_dim`, `caption_projection_dim`, `text_embed_dim`,
`cap_feat_dim`, `context_in_dim`, `text_dim`) are one generic semantic alias
vocabulary, never model or class dispatch. An unresolved callable or input
meaning yields a typed resolution failure.

The machine record is `receipt.json`; losslessly compressed lane logs are in
`lanes/`; the poison index is `poisons.md`.
