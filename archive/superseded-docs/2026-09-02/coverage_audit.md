# Coverage audit

_Generated 2026-07-04 by `scripts/coverage_audit.py`._

- Models attempted: **49**
- Parsed: **35**  ·  Gated/inaccessible: **13**  ·  Errored: **1**
- Distinct unparsed config fields: **0**

## Config fields we don't parse

Architectural-looking keys present in configs that no parser code reads. Sorted by how many models carry them.

| Field | # models | Models |
| --- | --- | --- |

## Partial-config reasons

_None — every parsed model produced a complete structure._

## Per-model detail

| Model | model_type | status | unparsed fields |
| --- | --- | --- | --- |
| meta-llama/Llama-2-7b-hf | — | gated | — |
| meta-llama/Llama-2-70b-hf | — | gated | — |
| codellama/CodeLlama-34b-hf | llama | ok | — |
| meta-llama/Meta-Llama-3-8B | — | gated | — |
| meta-llama/Llama-3.1-8B | — | gated | — |
| meta-llama/Llama-3.2-1B | — | gated | — |
| meta-llama/Llama-3.2-11B-Vision | — | gated | — |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | — | gated | — |
| meta-llama/Llama-4-Maverick-17B-128E-Instruct | — | gated | — |
| mistralai/Mistral-7B-v0.1 | mistral | ok | — |
| mistralai/Mixtral-8x7B-v0.1 | mixtral | ok | — |
| mistralai/Mixtral-8x22B-v0.1 | mixtral | ok | — |
| mistralai/Mistral-Small-24B-Instruct-2501 | mistral | ok | — |
| mistralai/Pixtral-12B-2409 | — | http_404 | — |
| mistralai/Ministral-8B-Instruct-2410 | mistral | ok | — |
| mistralai/Magistral-Small-2506 | mistral | ok | — |
| Qwen/Qwen2.5-72B | qwen2 | ok | — |
| Qwen/Qwen2-VL-7B-Instruct | qwen2_vl | ok | — |
| Qwen/QwQ-32B | qwen2 | ok | — |
| Qwen/Qwen3-0.6B | qwen3 | ok | — |
| Qwen/Qwen3-8B | qwen3 | ok | — |
| Qwen/Qwen3-30B-A3B | qwen3_moe | ok | — |
| Qwen/Qwen3-235B-A22B | qwen3_moe | ok | — |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | qwen3_moe | ok | — |
| Qwen/Qwen3-VL-235B-A22B-Instruct | qwen3_vl_moe | ok | — |
| Qwen/Qwen3-Omni-30B-A3B-Instruct | qwen3_omni_moe | ok | — |
| google/gemma-7b | — | gated | — |
| google/gemma-2-27b | — | gated | — |
| google/gemma-3-4b-it | — | gated | — |
| google/gemma-3-27b-it | — | gated | — |
| google/recurrentgemma-2b | — | gated | — |
| deepseek-ai/deepseek-llm-67b-chat | llama | ok | — |
| deepseek-ai/DeepSeek-V2 | deepseek_v2 | ok | — |
| deepseek-ai/DeepSeek-V3 | deepseek_v3 | ok | — |
| deepseek-ai/DeepSeek-R1 | deepseek_v3 | ok | — |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | qwen2 | ok | — |
| deepseek-ai/DeepSeek-V3.1-Terminus | deepseek_v3 | ok | — |
| deepseek-ai/DeepSeek-V3.2-Exp | deepseek_v32 | ok | — |
| openai/gpt-oss-20b | gpt_oss | ok | — |
| openai/gpt-oss-120b | gpt_oss | ok | — |
| moonshotai/Kimi-K2-Instruct | kimi_k2 | ok | — |
| moonshotai/Kimi-K2-Thinking | kimi_k2 | ok | — |
| zai-org/GLM-4.5 | glm4_moe | ok | — |
| zai-org/GLM-4.6 | glm4_moe | ok | — |
| microsoft/phi-2 | phi | ok | — |
| microsoft/Phi-3-mini-4k-instruct | phi3 | ok | — |
| microsoft/Phi-3.5-MoE-instruct | phimoe | ok | — |
| microsoft/phi-4 | phi3 | ok | — |
| microsoft/phi-4-multimodal-instruct | phi4mm | ok | — |

## Not audited

- **meta-llama/Llama-2-7b-hf** — gated (needs HF token)
- **meta-llama/Llama-2-70b-hf** — gated (needs HF token)
- **meta-llama/Meta-Llama-3-8B** — gated (needs HF token)
- **meta-llama/Llama-3.1-8B** — gated (needs HF token)
- **meta-llama/Llama-3.2-1B** — gated (needs HF token)
- **meta-llama/Llama-3.2-11B-Vision** — gated (needs HF token)
- **meta-llama/Llama-4-Scout-17B-16E-Instruct** — gated (needs HF token)
- **meta-llama/Llama-4-Maverick-17B-128E-Instruct** — gated (needs HF token)
- **google/gemma-7b** — gated (needs HF token)
- **google/gemma-2-27b** — gated (needs HF token)
- **google/gemma-3-4b-it** — gated (needs HF token)
- **google/gemma-3-27b-it** — gated (needs HF token)
- **google/recurrentgemma-2b** — gated (needs HF token)
- **mistralai/Pixtral-12B-2409** — http_404: HTTP Error 404: Not Found
