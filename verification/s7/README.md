# S7 shadow disagreement matrix

This is a 29-corpus + 10-TO_SERVE observation denominator. It is not a production dispatch table. Runtime names are addresses only; custom mechanism meanings require exact resolved source and existing facts.

Every occurrence has construction, execution and projection axes. `no_recipe_attempted` identifies our missing probe; `unobserved_no_static_proof` identifies an attempted recipe that did not prove this occurrence. Under v2.6, every unresolved value is classified as `investigation_missing`, `structure_unaccounted`, or `mechanism_unresolved`; the last is legal only with its typed investigation receipt and concrete reason. S7 does not relabel an execution observation as a known mechanism. Full per-occurrence tables are the deterministic gzip JSON files under `models/`.

| cohort | model | occurrences | construction conflicts | no recipe | attempted-unobserved | rendered | grouped | containers | projection unresolved | investigation missing | structure unaccounted | mechanism unresolved | relations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| corpus | AuraFlow-v0.3 | 1101 | 0 | 0 | 759 | 38 | 0 | 38 | 1025 | 759 | 1025 | 0 | none |
| corpus | bloom | 777 | 0 | 0 | 2 | 2 | 210 | 1 | 564 | 2 | 564 | 0 | param_share |
| corpus | CogVideoX-5b | 1448 | 0 | 0 | 1067 | 43 | 0 | 85 | 1320 | 1067 | 1320 | 0 | none |
| corpus | dbrx-base | 527 | 0 | 0 | 524 | 2 | 241 | 1 | 283 | 524 | 283 | 0 | none |
| corpus | DeepSeek-V3 | 1215 | 0 | 0 | 2 | 307 | 307 | 1 | 600 | 2 | 600 | 0 | none |
| corpus | FLUX.2-dev | 706 | 0 | 0 | 702 | 60 | 72 | 10 | 564 | 702 | 564 | 0 | none |
| corpus | FluxTransformer2DModel | 1321 | 0 | 0 | 1279 | 58 | 19 | 59 | 1185 | 1279 | 1185 | 0 | none |
| corpus | gemma-2-2b-it | 397 | 0 | 0 | 2 | 132 | 236 | 1 | 28 | 2 | 28 | 0 | param_share |
| corpus | GLM-4.5 | 1743 | 0 | 0 | 1743 | 370 | 830 | 1 | 542 | 1743 | 542 | 0 | none |
| corpus | gpt-oss-20b | 271 | 0 | 0 | 271 | 26 | 218 | 1 | 26 | 271 | 26 | 0 | none |
| corpus | granite-3.0-8b-instruct | 527 | 0 | 0 | 2 | 202 | 282 | 1 | 42 | 2 | 42 | 0 | param_share |
| corpus | HunyuanVideo | 1927 | 120 | 0 | 1399 | 25 | 3 | 67 | 1832 | 1399 | 1952 | 0 | none |
| corpus | llama-7b | 423 | 0 | 0 | 2 | 162 | 226 | 1 | 34 | 2 | 34 | 0 | none |
| corpus | LTX-Video | 777 | 0 | 0 | 775 | 29 | 0 | 85 | 663 | 775 | 663 | 0 | none |
| corpus | Lumina-Image-2.0 | 946 | 0 | 0 | 674 | 3 | 0 | 34 | 909 | 674 | 909 | 0 | none |
| corpus | mochi-1-preview | 2416 | 0 | 0 | 2410 | 1 | 0 | 144 | 2271 | 2410 | 2271 | 0 | none |
| corpus | musicgen-small | 969 | 0 | 0 | 337 | 1 | 0 | 63 | 905 | 337 | 905 | 0 | param_share |
| corpus | OLMo-2-1124-7B | 487 | 0 | 0 | 2 | 162 | 290 | 1 | 34 | 2 | 34 | 0 | none |
| corpus | PixArt-Sigma-XL-2-1024-MS | 1310 | 0 | 0 | 86 | 0 | 0 | 85 | 1225 | 86 | 1225 | 0 | none |
| corpus | prxpixel-t2i | 447 | 0 | 0 | 447 | 26 | 48 | 26 | 347 | 447 | 347 | 0 | none |
| corpus | Qwen-Image | 2479 | 0 | 0 | 2297 | 61 | 0 | 301 | 2117 | 2297 | 2117 | 0 | none |
| corpus | Qwen2-VL-7B-Instruct | 731 | 0 | 0 | 329 | 30 | 198 | 3 | 500 | 329 | 500 | 0 | none |
| corpus | Qwen3.5-27B text component | 1063 | 0 | 0 | 50 | 274 | 337 | 1 | 451 | 50 | 451 | 0 | none |
| corpus | Qwen3-8B | 547 | 0 | 0 | 2 | 182 | 326 | 1 | 38 | 2 | 38 | 0 | none |
| corpus | Sana_1600M_1024px_diffusers | 905 | 0 | 0 | 461 | 22 | 20 | 41 | 822 | 461 | 822 | 0 | none |
| corpus | stable-diffusion-3.5-large | 1689 | 0 | 0 | 1421 | 1 | 0 | 114 | 1574 | 1421 | 1574 | 0 | none |
| corpus | stable-diffusion-xl-base-1.0 | 3551 | 0 | 0 | 240 | 0 | 0 | 239 | 3312 | 240 | 3312 | 0 | none |
| corpus | stablelm-2-1_6b | 343 | 0 | 0 | 2 | 122 | 121 | 1 | 99 | 2 | 99 | 0 | none |
| corpus | Wan2.2-T2V-A14B-Diffusers | 1140 | 0 | 0 | 1138 | 1 | 0 | 121 | 1018 | 1138 | 1018 | 0 | none |
| to_serve | CohereLabs/c4ai-command-a-03-2025 | 791 | 0 | 0 | 2 | 322 | 385 | 1 | 83 | 2 | 83 | 0 | param_share |
| to_serve | deepseek-ai/DeepSeek-Coder-V2-Lite-Base | 489 | 0 | 0 | 489 | 83 | 136 | 1 | 269 | 489 | 269 | 0 | none |
| to_serve | deepseek-ai/DeepSeek-V4-Flash | 1502 | 0 | 0 | 1453 | 2 | 86 | 1 | 1413 | 1453 | 1413 | 0 | multi_stream_residual, side_head |
| to_serve | google/gemma-3n-E2B | 2896 | 0 | 0 | 1952 | 94 | 0 | 141 | 2661 | 1952 | 2661 | 0 | activation_reuse, multi_stream_residual, param_share, per_layer_side_input |
| to_serve | ai21labs/Jamba-v0.1 | 546 | 0 | 0 | 546 | 1 | 0 | 1 | 544 | 546 | 544 | 0 | none |
| to_serve | LiquidAI/LFM2-1.2B | 201 | 0 | 0 | 2 | 8 | 96 | 1 | 96 | 2 | 96 | 0 | param_share |
| to_serve | nvidia/Nemotron-H-8B-Base-8K | 370 | 0 | 0 | 2 | 2 | 53 | 1 | 314 | 2 | 314 | 0 | none |
| to_serve | Qwen/Qwen3.5-27B | 1345 | 0 | 0 | 331 | 194 | 0 | 2 | 1149 | 331 | 1149 | 0 | none |
| to_serve | Qwen/Qwen3-Omni-30B-A3B-Instruct | 2144 | 0 | 0 | 2144 | 1 | 0 | 21 | 2122 | 2144 | 2122 | 0 | none |
| to_serve | Qwen/Qwen3-VL-235B-A22B-Instruct | 1621 | 0 | 0 | 1621 | 2 | 0 | 3 | 1616 | 1621 | 1616 | 0 | none |
