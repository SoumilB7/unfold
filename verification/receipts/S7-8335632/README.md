# S7 v2.6.2 final local receipt

- verified implementation commit: `8335632efba1856055225d6e26d54135cb61972f`
- coordinator run: `67e090fd8d`
- result: **PASS; every lane green and every tree/artifact fingerprint identical**
- product delta: **none**
- production consumers of `physics`: **zero**
- Linux namespace lane: **pending push**

| lane | result |
|---|---:|
| collection | 4,211 node ids; 4,115 runnable tests in the bounded full bracket |
| focused S6/S7 | 319 passed |
| U2 authority | 44 passed |
| full | all 24 fresh-process batches passed; 4,115 tests ran exactly once |
| preservation | 52 passed; 29/29 witnesses byte-identical |
| static | clean across 20 changed Python files |

The coordinator ran lanes serially. Bounded concurrency occurred only inside the
active full or preservation lane. Every lane ran in a detached worktree at the
exact implementation commit.

## What this closes

- Projection claim strength originates at the reader-authored fact. A config
  value can prove only an exact value; constructor occurrence can prove only
  existence. Consumer requirements cannot upgrade either proof.
- The 39-model matrix explicitly separates facts with no claim declaration from
  facts whose declaration exists but whose typed proof is still missing. The
  result is intentionally conservative: 710 citations are qualified and 16,324
  remain visible as unqualified S9 work, rather than being blessed by a central
  fact-name table.
- All 39 targets received one signature-derived recipe attempt. Execution is 22
  `ok` and 17 typed `failed`; `no_recipe_attempted` is zero.
- GPT-OSS and Qwen3-VL alone receive exactly one causal grouped-matmul retry from
  float32 to bf16. Their checkpoint dtype remains explicitly present and null;
  the execution retry never rewrites deployment evidence.
- Relation probing is model-, family-, class- and field-role blind. It selects
  exact ordered containers from construction and successful execution evidence.
  Custom relation meaning still requires exact source-bound runtime operations.
- Hard controls remain separated: Gemma3n has 30/30 output-reaching stream
  contractions; DeepSeek-V4 has 43/43; DeepSeek-V3 has zero; SD3.5 has none.
- HunyuanVideo's 120 construction conflicts remain visible and blocking.

## Exact final denominator

| surface | count |
|---|---:|
| runtime occurrences | 44,088 |
| rendered | 1,864 |
| grouped | 1,674 |
| exact non-architectural containers | 1,700 |
| projection unresolved | 38,850 |
| execution unresolved | 20,861 |
| investigation missing | 20,861 |
| structure unaccounted | 34,717 |
| mechanism unresolved | 4,253 |
| relation rows | 21 |
| blocking findings | 55,578 |

These numbers are a disagreement inventory, not a progress score. A row moving
from falsely qualified to explicit unresolved is an honesty improvement even if
the visible unresolved count grows.

The machine record is `receipt.json`; losslessly compressed lane logs are in
`lanes/`; the adversarial coverage index is `poisons.md`.
