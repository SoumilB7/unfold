# S7 shadow disagreement matrix

This is a 29-corpus + 10-TO_SERVE observation denominator. It is not a production dispatch table. Runtime names are addresses only; custom mechanism meanings require exact resolved source and existing facts.

Every occurrence has construction, execution and projection axes. Unresolved axes are intentionally blocking; S7 does not relabel them as non-architectural to improve a score. Full per-occurrence tables are the deterministic gzip JSON files under `models/`.

| cohort | model | occurrences | construction conflicts | execution unresolved | projection unresolved | relations |
|---|---|---:|---:|---:|---:|---|
| corpus | AuraFlow-v0.3 | 1101 | 0 | 759 | 1101 | none |
| corpus | bloom | 777 | 0 | 777 | 777 | param_share |
| corpus | CogVideoX-5b | 1448 | 0 | 1067 | 1448 | none |
| corpus | dbrx-base | 527 | 0 | 524 | 527 | none |
| corpus | DeepSeek-V3 | 1215 | 0 | 2 | 1215 | none |
| corpus | FLUX.2-dev | 706 | 0 | 702 | 706 | none |
| corpus | FluxTransformer2DModel | 1321 | 0 | 1279 | 1321 | none |
| corpus | gemma-2-2b-it | 397 | 0 | 397 | 397 | param_share |
| corpus | GLM-4.5 | 1743 | 0 | 1743 | 1743 | none |
| corpus | gpt-oss-20b | 271 | 0 | 271 | 271 | none |
| corpus | granite-3.0-8b-instruct | 527 | 0 | 527 | 527 | param_share |
| corpus | HunyuanVideo | 1927 | 120 | 1399 | 1927 | none |
| corpus | llama-7b | 423 | 0 | 2 | 423 | none |
| corpus | LTX-Video | 777 | 0 | 775 | 777 | none |
| corpus | Lumina-Image-2.0 | 946 | 0 | 674 | 946 | none |
| corpus | mochi-1-preview | 2416 | 0 | 2410 | 2416 | none |
| corpus | musicgen-small | 969 | 0 | 337 | 969 | param_share |
| corpus | OLMo-2-1124-7B | 487 | 0 | 487 | 487 | none |
| corpus | PixArt-Sigma-XL-2-1024-MS | 1310 | 0 | 86 | 1310 | none |
| corpus | prxpixel-t2i | 447 | 0 | 447 | 447 | none |
| corpus | Qwen-Image | 2479 | 0 | 2297 | 2479 | none |
| corpus | Qwen2-VL-7B-Instruct | 731 | 0 | 329 | 731 | none |
| corpus | Qwen3.5-27B text component | 1063 | 0 | 1063 | 1063 | none |
| corpus | Qwen3-8B | 547 | 0 | 547 | 547 | none |
| corpus | Sana_1600M_1024px_diffusers | 905 | 0 | 461 | 905 | none |
| corpus | stable-diffusion-3.5-large | 1689 | 0 | 1421 | 1689 | none |
| corpus | stable-diffusion-xl-base-1.0 | 3551 | 0 | 240 | 3551 | none |
| corpus | stablelm-2-1_6b | 343 | 0 | 343 | 343 | none |
| corpus | Wan2.2-T2V-A14B-Diffusers | 1140 | 0 | 1138 | 1140 | none |
| to_serve | CohereLabs/c4ai-command-a-03-2025 | 791 | 0 | 775 | 791 | param_share |
| to_serve | deepseek-ai/DeepSeek-Coder-V2-Lite-Base | 489 | 0 | 489 | 489 | none |
| to_serve | deepseek-ai/DeepSeek-V4-Flash | 1502 | 0 | 1453 | 1502 | multi_stream_residual, side_head |
| to_serve | google/gemma-3n-E2B | 2896 | 0 | 2801 | 2896 | activation_reuse, multi_stream_residual, param_share, per_layer_side_input |
| to_serve | ai21labs/Jamba-v0.1 | 546 | 0 | 546 | 546 | none |
| to_serve | LiquidAI/LFM2-1.2B | 201 | 0 | 201 | 201 | param_share |
| to_serve | nvidia/Nemotron-H-8B-Base-8K | 370 | 0 | 370 | 370 | none |
| to_serve | Qwen/Qwen3.5-27B | 1345 | 0 | 1345 | 1345 | none |
| to_serve | Qwen/Qwen3-Omni-30B-A3B-Instruct | 2144 | 0 | 2144 | 2144 | none |
| to_serve | Qwen/Qwen3-VL-235B-A22B-Instruct | 1621 | 0 | 1621 | 1621 | none |
