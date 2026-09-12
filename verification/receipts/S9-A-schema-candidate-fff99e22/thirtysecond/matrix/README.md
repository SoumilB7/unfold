# S7 shadow disagreement matrix

This is a 29-corpus + 10-TO_SERVE observation denominator. It is not a production dispatch table. Runtime names are addresses only; custom mechanism meanings require exact resolved source and existing facts.

Every occurrence has construction, execution and projection axes. `no_recipe_attempted` identifies our missing probe; `unobserved_no_static_proof` identifies an attempted recipe that did not prove this occurrence. Under v2.6, every unresolved value is classified as `investigation_missing`, `structure_unaccounted`, or `mechanism_unresolved`; the last is legal only with its typed reader-result exhaustion bound to the exact claim. A rendered or grouped occurrence may still carry blocking fact-level `claim_proof_unstamped` findings owned by S9; drawing an occurrence does not qualify those facts for cutover. S7 does not relabel an execution observation as a known mechanism. Full per-occurrence tables are the deterministic gzip JSON files under `models/`.

Recipe status is reported per target. Checkpoint dtype (including absence/null) is kept separate from the execution dtype; a recorded bf16 retry never rewrites deployment evidence.

| cohort | model | recipe | checkpoint dtype | execution dtype | retry | occurrences | construction conflicts | no recipe | attempted-unobserved | rendered | grouped | containers | projection unresolved | unstamped fact proofs | investigation missing | structure unaccounted | mechanism unresolved | relations |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| corpus | AuraFlow-v0.3 | ok→failed | None (absent) | float32 | 0 | 1101 | 0 | 0 | 759 | 38 | 0 | 38 | 1025 | 0 | 759 | 1025 | 0 | none |
| corpus | bloom | ok→ok | None (present) | float32 | 0 | 777 | 0 | 0 | 1 | 2 | 210 | 1 | 564 | 141 | 142 | 564 | 0 | param_share |
| corpus | CogVideoX-5b | ok→failed | None (absent) | float32 | 0 | 1448 | 0 | 0 | 1067 | 43 | 0 | 85 | 1320 | 0 | 1067 | 1320 | 0 | none |
| corpus | dbrx-base | ok→failed | None (absent) | float32 | 0 | 527 | 0 | 0 | 524 | 2 | 241 | 1 | 283 | 0 | 524 | 283 | 0 | none |
| corpus | DeepSeek-V3 | ok→ok | bfloat16 (present) | bfloat16 | 0 | 1215 | 0 | 0 | 1 | 307 | 307 | 1 | 600 | 61 | 62 | 600 | 0 | none |
| corpus | FLUX.2-dev | ok→failed | None (absent) | float32 | 0 | 706 | 0 | 0 | 702 | 60 | 72 | 10 | 564 | 0 | 702 | 564 | 0 | none |
| corpus | FluxTransformer2DModel | ok→failed | None (absent) | float32 | 0 | 1321 | 0 | 0 | 1279 | 58 | 19 | 59 | 1185 | 0 | 1279 | 1185 | 0 | none |
| corpus | gemma-2-2b-it | ok→ok | bfloat16 (present) | bfloat16 | 0 | 397 | 0 | 0 | 1 | 132 | 236 | 1 | 28 | 26 | 27 | 28 | 0 | param_share |
| corpus | GLM-4.5 | ok→ok | bfloat16 (present) | bfloat16 | 0 | 1743 | 0 | 0 | 1 | 370 | 830 | 1 | 542 | 92 | 93 | 542 | 0 | none |
| corpus | gpt-oss-20b | ok→ok | None (present) | bfloat16 | 1 | 271 | 0 | 0 | 1 | 26 | 218 | 1 | 26 | 24 | 25 | 26 | 0 | none |
| corpus | granite-3.0-8b-instruct | ok→ok | bfloat16 (present) | bfloat16 | 0 | 527 | 0 | 0 | 1 | 202 | 282 | 1 | 42 | 40 | 41 | 42 | 0 | param_share |
| corpus | HunyuanVideo | ok→failed | None (absent) | float32 | 0 | 1927 | 120 | 0 | 1399 | 25 | 3 | 67 | 1832 | 0 | 1399 | 1952 | 0 | none |
| corpus | llama-7b | ok→ok | float16 (present) | float16 | 0 | 423 | 0 | 0 | 1 | 162 | 226 | 1 | 34 | 32 | 33 | 34 | 0 | none |
| corpus | LTX-Video | ok→failed | None (absent) | float32 | 0 | 777 | 0 | 0 | 775 | 29 | 0 | 85 | 663 | 0 | 775 | 663 | 0 | none |
| corpus | Lumina-Image-2.0 | ok→failed | None (absent) | float32 | 0 | 946 | 0 | 0 | 674 | 3 | 0 | 34 | 909 | 0 | 674 | 909 | 0 | none |
| corpus | mochi-1-preview | ok→failed | None (absent) | float32 | 0 | 2416 | 0 | 0 | 2410 | 1 | 0 | 144 | 2271 | 0 | 2410 | 2271 | 0 | none |
| corpus | musicgen-small | ok→failed | None (absent) | float32 | 0 | 969 | 0 | 0 | 336 | 1 | 0 | 63 | 905 | 1 | 337 | 905 | 0 | param_share |
| corpus | OLMo-2-1124-7B | ok→ok | float32 (present) | float32 | 0 | 487 | 0 | 0 | 1 | 162 | 290 | 1 | 34 | 32 | 33 | 34 | 0 | none |
| corpus | PixArt-Sigma-XL-2-1024-MS | failed→failed | None (absent) | float32 | 0 | 1310 | 0 | 0 | 85 | 0 | 0 | 85 | 1225 | 0 | 85 | 1225 | 0 | none |
| corpus | prxpixel-t2i | ok→ok | None (absent) | float32 | 0 | 447 | 0 | 0 | 25 | 26 | 48 | 26 | 347 | 0 | 25 | 347 | 0 | none |
| corpus | Qwen-Image | ok→failed | None (absent) | float32 | 0 | 2479 | 0 | 0 | 2297 | 61 | 0 | 301 | 2117 | 0 | 2297 | 2117 | 0 | none |
| corpus | Qwen2-VL-7B-Instruct | ok→ok | bfloat16 (present) | bfloat16 | 0 | 731 | 0 | 0 | 328 | 30 | 198 | 3 | 500 | 28 | 356 | 500 | 0 | none |
| corpus | Qwen3.5-27B text component | ok→ok | None (present) | float32 | 0 | 1063 | 0 | 0 | 49 | 274 | 337 | 1 | 451 | 16 | 65 | 451 | 0 | none |
| corpus | Qwen3-8B | ok→ok | bfloat16 (present) | bfloat16 | 0 | 547 | 0 | 0 | 1 | 182 | 326 | 1 | 38 | 36 | 37 | 38 | 0 | none |
| corpus | Sana_1600M_1024px_diffusers | ok→failed | None (absent) | float32 | 0 | 905 | 0 | 0 | 461 | 22 | 20 | 41 | 822 | 0 | 461 | 822 | 0 | none |
| corpus | stable-diffusion-3.5-large | ok→failed | None (absent) | float32 | 0 | 1689 | 0 | 0 | 1421 | 1 | 0 | 114 | 1574 | 0 | 1421 | 1574 | 0 | none |
| corpus | stable-diffusion-xl-base-1.0 | ok→failed | None (absent) | float32 | 0 | 3551 | 0 | 0 | 239 | 1928 | 0 | 2 | 1621 | 0 | 239 | 1621 | 0 | none |
| corpus | stablelm-2-1_6b | ok→ok | float16 (present) | float16 | 0 | 343 | 0 | 0 | 1 | 122 | 121 | 1 | 99 | 24 | 25 | 99 | 0 | none |
| corpus | Wan2.2-T2V-A14B-Diffusers | ok→failed | None (absent) | float32 | 0 | 1140 | 0 | 0 | 1138 | 1 | 0 | 121 | 1018 | 0 | 1138 | 1018 | 0 | none |
| to_serve | CohereLabs/c4ai-command-a-03-2025 | ok→ok | bfloat16 (present) | bfloat16 | 0 | 791 | 0 | 0 | 1 | 322 | 385 | 1 | 83 | 64 | 65 | 83 | 0 | param_share |
| to_serve | deepseek-ai/DeepSeek-Coder-V2-Lite-Base | ok→ok | bfloat16 (present) | bfloat16 | 0 | 489 | 0 | 0 | 27 | 83 | 136 | 1 | 269 | 0 | 27 | 269 | 0 | none |
| to_serve | deepseek-ai/DeepSeek-V4-Flash | ok→ok | bfloat16 (present) | bfloat16 | 0 | 1502 | 0 | 0 | 104 | 88 | 86 | 1 | 1327 | 0 | 104 | 1327 | 0 | multi_stream_residual, side_head |
| to_serve | google/gemma-3n-E2B | ok→ok | bfloat16 (present) | bfloat16 | 0 | 2896 | 0 | 0 | 1951 | 94 | 0 | 141 | 2661 | 0 | 1951 | 2661 | 0 | activation_reuse, multi_stream_residual, param_share, per_layer_side_input |
| to_serve | ai21labs/Jamba-v0.1 | ok→ok | bfloat16 (present) | bfloat16 | 0 | 546 | 0 | 0 | 1 | 1 | 0 | 1 | 544 | 0 | 1 | 544 | 0 | none |
| to_serve | LiquidAI/LFM2-1.2B | ok→ok | bfloat16 (present) | bfloat16 | 0 | 201 | 0 | 0 | 1 | 8 | 96 | 1 | 96 | 6 | 7 | 96 | 0 | param_share |
| to_serve | nvidia/Nemotron-H-8B-Base-8K | ok→ok | bfloat16 (present) | bfloat16 | 0 | 370 | 0 | 0 | 1 | 2 | 53 | 1 | 314 | 0 | 1 | 314 | 0 | none |
| to_serve | Qwen/Qwen3.5-27B | ok→ok | None (present) | float32 | 0 | 1345 | 0 | 0 | 330 | 194 | 0 | 2 | 1149 | 0 | 330 | 1149 | 0 | none |
| to_serve | Qwen/Qwen3-Omni-30B-A3B-Instruct | failed→failed | bfloat16 (present) | bfloat16 | 0 | 2144 | 0 | 0 | 2144 | 1 | 0 | 21 | 2122 | 0 | 2144 | 2122 | 0 | none |
| to_serve | Qwen/Qwen3-VL-235B-A22B-Instruct | ok→ok | None (present) | bfloat16 | 1 | 1621 | 0 | 0 | 298 | 2 | 0 | 3 | 1616 | 0 | 298 | 1616 | 0 | none |
