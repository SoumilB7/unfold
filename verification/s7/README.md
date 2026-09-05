# S7 shadow disagreement matrix

This is a 29-corpus + 10-TO_SERVE observation denominator. It is not a production dispatch table. Runtime names are addresses only; custom mechanism meanings require exact resolved source and existing facts.

Every occurrence has construction, execution and projection axes. `no_recipe_attempted` identifies our missing probe; `unobserved_no_static_proof` identifies an attempted recipe that did not prove this occurrence. Under v2.6, every unresolved value is classified as `investigation_missing`, `structure_unaccounted`, or `mechanism_unresolved`; the last is legal only with its typed investigation receipt and concrete reason. S7 does not relabel an execution observation as a known mechanism. Full per-occurrence tables are the deterministic gzip JSON files under `models/`.

Recipe status is reported per target. Checkpoint dtype (including absence/null) is kept separate from the execution dtype; a recorded bf16 retry never rewrites deployment evidence.

| cohort | model | recipe | checkpoint dtype | execution dtype | retry | occurrences | construction conflicts | no recipe | attempted-unobserved | rendered | grouped | containers | projection unresolved | investigation missing | structure unaccounted | mechanism unresolved | relations |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| corpus | AuraFlow-v0.3 | ok→failed | None (absent) | float32 | 0 | 1101 | 0 | 0 | 759 | 0 | 0 | 38 | 1063 | 759 | 1025 | 38 | none |
| corpus | bloom | ok→ok | None (present) | float32 | 0 | 777 | 0 | 0 | 2 | 1 | 0 | 1 | 775 | 2 | 564 | 211 | param_share |
| corpus | CogVideoX-5b | ok→failed | None (absent) | float32 | 0 | 1448 | 0 | 0 | 1067 | 0 | 0 | 85 | 1363 | 1067 | 1320 | 43 | none |
| corpus | dbrx-base | ok→failed | None (absent) | float32 | 0 | 527 | 0 | 0 | 524 | 1 | 0 | 1 | 525 | 524 | 283 | 242 | none |
| corpus | DeepSeek-V3 | ok→ok | bfloat16 (present) | bfloat16 | 0 | 1215 | 0 | 0 | 2 | 245 | 0 | 1 | 969 | 2 | 600 | 369 | none |
| corpus | FLUX.2-dev | ok→failed | None (absent) | float32 | 0 | 706 | 0 | 0 | 702 | 0 | 0 | 10 | 696 | 702 | 564 | 132 | none |
| corpus | FluxTransformer2DModel | ok→failed | None (absent) | float32 | 0 | 1321 | 0 | 0 | 1279 | 0 | 0 | 59 | 1262 | 1279 | 1185 | 77 | none |
| corpus | gemma-2-2b-it | ok→ok | bfloat16 (present) | bfloat16 | 0 | 397 | 0 | 0 | 2 | 79 | 104 | 1 | 213 | 2 | 28 | 185 | param_share |
| corpus | GLM-4.5 | ok→ok | bfloat16 (present) | bfloat16 | 0 | 1743 | 0 | 0 | 2 | 277 | 368 | 1 | 1097 | 2 | 542 | 555 | none |
| corpus | gpt-oss-20b | ok→ok | None (present) | bfloat16 | 1 | 271 | 0 | 0 | 2 | 1 | 96 | 1 | 173 | 2 | 26 | 147 | none |
| corpus | granite-3.0-8b-instruct | ok→ok | bfloat16 (present) | bfloat16 | 0 | 527 | 0 | 0 | 2 | 121 | 160 | 1 | 245 | 2 | 42 | 203 | param_share |
| corpus | HunyuanVideo | ok→failed | None (absent) | float32 | 0 | 1927 | 120 | 0 | 1399 | 2 | 0 | 67 | 1858 | 1399 | 1952 | 26 | none |
| corpus | llama-7b | ok→ok | float16 (present) | float16 | 0 | 423 | 0 | 0 | 2 | 97 | 128 | 1 | 197 | 2 | 34 | 163 | none |
| corpus | LTX-Video | ok→failed | None (absent) | float32 | 0 | 777 | 0 | 0 | 775 | 0 | 0 | 85 | 692 | 775 | 663 | 29 | none |
| corpus | Lumina-Image-2.0 | ok→failed | None (absent) | float32 | 0 | 946 | 0 | 0 | 674 | 0 | 0 | 34 | 912 | 674 | 909 | 3 | none |
| corpus | mochi-1-preview | ok→failed | None (absent) | float32 | 0 | 2416 | 0 | 0 | 2410 | 0 | 0 | 144 | 2272 | 2410 | 2271 | 1 | none |
| corpus | musicgen-small | ok→failed | None (absent) | float32 | 0 | 969 | 0 | 0 | 337 | 0 | 0 | 63 | 906 | 337 | 905 | 1 | param_share |
| corpus | OLMo-2-1124-7B | ok→ok | float32 (present) | float32 | 0 | 487 | 0 | 0 | 2 | 97 | 128 | 1 | 261 | 2 | 34 | 227 | none |
| corpus | PixArt-Sigma-XL-2-1024-MS | failed→failed | None (absent) | float32 | 0 | 1310 | 0 | 0 | 86 | 0 | 0 | 85 | 1225 | 86 | 1225 | 0 | none |
| corpus | prxpixel-t2i | ok→ok | None (absent) | float32 | 0 | 447 | 0 | 0 | 26 | 0 | 0 | 26 | 421 | 26 | 347 | 74 | none |
| corpus | Qwen-Image | ok→failed | None (absent) | float32 | 0 | 2479 | 0 | 0 | 2297 | 0 | 0 | 301 | 2178 | 2297 | 2117 | 61 | none |
| corpus | Qwen2-VL-7B-Instruct | ok→ok | bfloat16 (present) | bfloat16 | 0 | 731 | 0 | 0 | 329 | 1 | 112 | 3 | 615 | 329 | 500 | 115 | none |
| corpus | Qwen3.5-27B text component | ok→ok | None (present) | float32 | 0 | 1063 | 0 | 0 | 50 | 193 | 64 | 1 | 805 | 50 | 451 | 354 | none |
| corpus | Qwen3-8B | ok→ok | bfloat16 (present) | bfloat16 | 0 | 547 | 0 | 0 | 2 | 109 | 144 | 1 | 293 | 2 | 38 | 255 | none |
| corpus | Sana_1600M_1024px_diffusers | ok→failed | None (absent) | float32 | 0 | 905 | 0 | 0 | 461 | 0 | 0 | 41 | 864 | 461 | 822 | 42 | none |
| corpus | stable-diffusion-3.5-large | ok→failed | None (absent) | float32 | 0 | 1689 | 0 | 0 | 1421 | 0 | 0 | 114 | 1575 | 1421 | 1574 | 1 | none |
| corpus | stable-diffusion-xl-base-1.0 | ok→failed | None (absent) | float32 | 0 | 3551 | 0 | 0 | 240 | 0 | 0 | 239 | 3312 | 240 | 3312 | 0 | none |
| corpus | stablelm-2-1_6b | ok→ok | float16 (present) | float16 | 0 | 343 | 0 | 0 | 2 | 73 | 96 | 1 | 173 | 2 | 99 | 74 | none |
| corpus | Wan2.2-T2V-A14B-Diffusers | ok→failed | None (absent) | float32 | 0 | 1140 | 0 | 0 | 1138 | 0 | 0 | 121 | 1019 | 1138 | 1018 | 1 | none |
| to_serve | CohereLabs/c4ai-command-a-03-2025 | ok→ok | bfloat16 (present) | bfloat16 | 0 | 791 | 0 | 0 | 2 | 193 | 256 | 1 | 341 | 2 | 83 | 258 | param_share |
| to_serve | deepseek-ai/DeepSeek-Coder-V2-Lite-Base | ok→ok | bfloat16 (present) | bfloat16 | 0 | 489 | 0 | 0 | 28 | 82 | 0 | 1 | 406 | 28 | 269 | 137 | none |
| to_serve | deepseek-ai/DeepSeek-V4-Flash | ok→ok | bfloat16 (present) | bfloat16 | 0 | 1502 | 0 | 0 | 105 | 1 | 0 | 1 | 1500 | 105 | 1413 | 87 | multi_stream_residual, side_head |
| to_serve | google/gemma-3n-E2B | ok→ok | bfloat16 (present) | bfloat16 | 0 | 2896 | 0 | 0 | 1952 | 93 | 0 | 141 | 2662 | 1952 | 2661 | 1 | activation_reuse, multi_stream_residual, param_share, per_layer_side_input |
| to_serve | ai21labs/Jamba-v0.1 | ok→ok | bfloat16 (present) | bfloat16 | 0 | 546 | 0 | 0 | 2 | 1 | 0 | 1 | 544 | 2 | 544 | 0 | none |
| to_serve | LiquidAI/LFM2-1.2B | ok→ok | bfloat16 (present) | bfloat16 | 0 | 201 | 0 | 0 | 2 | 1 | 18 | 1 | 181 | 2 | 96 | 85 | param_share |
| to_serve | nvidia/Nemotron-H-8B-Base-8K | ok→ok | bfloat16 (present) | bfloat16 | 0 | 370 | 0 | 0 | 2 | 1 | 0 | 1 | 368 | 2 | 314 | 54 | none |
| to_serve | Qwen/Qwen3.5-27B | ok→ok | None (present) | float32 | 0 | 1345 | 0 | 0 | 331 | 193 | 0 | 2 | 1150 | 331 | 1149 | 1 | none |
| to_serve | Qwen/Qwen3-Omni-30B-A3B-Instruct | failed→failed | bfloat16 (present) | bfloat16 | 0 | 2144 | 0 | 0 | 2144 | 1 | 0 | 21 | 2122 | 2144 | 2122 | 0 | none |
| to_serve | Qwen/Qwen3-VL-235B-A22B-Instruct | ok→ok | None (present) | bfloat16 | 1 | 1621 | 0 | 0 | 299 | 1 | 0 | 3 | 1617 | 299 | 1616 | 1 | none |
