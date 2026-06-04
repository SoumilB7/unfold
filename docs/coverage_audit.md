# Coverage audit

_Generated 2026-06-01 by `scripts/coverage_audit.py`._

- Models attempted: **49**
- Parsed: **35**  ·  Gated/inaccessible: **13**  ·  Errored: **1**
- Distinct unparsed config fields: **35**

## Config fields we don't parse

Architectural-looking keys present in configs that no parser code reads. Sorted by how many models carry them.

| Field | # models | Models |
| --- | --- | --- |
| `norm_topk_prob` | 14 | Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-Coder-30B-A3B-Instruct, Qwen3-VL-235B-A22B-Instruct, Qwen3-Omni-30B-A3B-Instruct, DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, Kimi-K2-Instruct, Kimi-K2-Thinking, GLM-4.5, GLM-4.6 |
| `n_group` | 9 | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, Kimi-K2-Instruct, Kimi-K2-Thinking, GLM-4.5, GLM-4.6 |
| `output_router_logits` | 9 | Mixtral-8x7B-v0.1, Mixtral-8x22B-v0.1, Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-Coder-30B-A3B-Instruct, Qwen3-Omni-30B-A3B-Instruct, gpt-oss-20b, gpt-oss-120b, Phi-3.5-MoE-instruct |
| `routed_scaling_factor` | 9 | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, Kimi-K2-Instruct, Kimi-K2-Thinking, GLM-4.5, GLM-4.6 |
| `router_aux_loss_coef` | 9 | Mixtral-8x7B-v0.1, Mixtral-8x22B-v0.1, Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-Coder-30B-A3B-Instruct, Qwen3-Omni-30B-A3B-Instruct, gpt-oss-20b, gpt-oss-120b, Phi-3.5-MoE-instruct |
| `topk_group` | 9 | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, Kimi-K2-Instruct, Kimi-K2-Thinking, GLM-4.5, GLM-4.6 |
| `quantization_config` | 8 | DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, gpt-oss-20b, gpt-oss-120b, Kimi-K2-Instruct, Kimi-K2-Thinking |
| `qk_nope_head_dim` | 7 | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, Kimi-K2-Instruct, Kimi-K2-Thinking |
| `qk_rope_head_dim` | 7 | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, Kimi-K2-Instruct, Kimi-K2-Thinking |
| `scoring_func` | 7 | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, Kimi-K2-Instruct, Kimi-K2-Thinking |
| `topk_method` | 7 | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, Kimi-K2-Instruct, Kimi-K2-Thinking |
| `v_head_dim` | 7 | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, Kimi-K2-Instruct, Kimi-K2-Thinking |
| `original_max_position_embeddings` | 4 | Phi-3-mini-4k-instruct, Phi-3.5-MoE-instruct, phi-4, phi-4-multimodal-instruct |
| `aux_loss_alpha` | 3 | DeepSeek-V2, Kimi-K2-Instruct, Kimi-K2-Thinking |
| `seq_aux` | 3 | DeepSeek-V2, Kimi-K2-Instruct, Kimi-K2-Thinking |
| `initial_context_length` | 2 | gpt-oss-20b, gpt-oss-120b |
| `lm_head_bias` | 2 | Phi-3.5-MoE-instruct, phi-4-multimodal-instruct |
| `swiglu_limit` | 2 | gpt-oss-20b, gpt-oss-120b |
| `attn_module_list_cfg` | 1 | DeepSeek-V3.1-Terminus |
| `audio_processor` | 1 | phi-4-multimodal-instruct |
| `code2wav_config` | 1 | Qwen3-Omni-30B-A3B-Instruct |
| `embd_layer` | 1 | phi-4-multimodal-instruct |
| `enable_audio_output` | 1 | Qwen3-Omni-30B-A3B-Instruct |
| `full_attn_mod` | 1 | phi-4-multimodal-instruct |
| `index_head_dim` | 1 | DeepSeek-V3.2-Exp |
| `index_n_heads` | 1 | DeepSeek-V3.2-Exp |
| `index_topk` | 1 | DeepSeek-V3.2-Exp |
| `input_jitter_noise` | 1 | Phi-3.5-MoE-instruct |
| `interpolate_factor` | 1 | phi-4-multimodal-instruct |
| `mlp_bias` | 1 | phi-4-multimodal-instruct |
| `router_jitter_noise` | 1 | Phi-3.5-MoE-instruct |
| `shared_expert_intermediate_size` | 1 | Qwen3-Omni-30B-A3B-Instruct |
| `speech_lora` | 1 | phi-4-multimodal-instruct |
| `talker_config` | 1 | Qwen3-Omni-30B-A3B-Instruct |
| `vision_lora` | 1 | phi-4-multimodal-instruct |

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
| mistralai/Mixtral-8x7B-v0.1 | mixtral | ok | `output_router_logits`, `router_aux_loss_coef` |
| mistralai/Mixtral-8x22B-v0.1 | mixtral | ok | `output_router_logits`, `router_aux_loss_coef` |
| mistralai/Mistral-Small-24B-Instruct-2501 | mistral | ok | — |
| mistralai/Pixtral-12B-2409 | — | http_404 | — |
| mistralai/Ministral-8B-Instruct-2410 | mistral | ok | — |
| mistralai/Magistral-Small-2506 | mistral | ok | — |
| Qwen/Qwen2.5-72B | qwen2 | ok | — |
| Qwen/Qwen2-VL-7B-Instruct | qwen2_vl | ok | — |
| Qwen/QwQ-32B | qwen2 | ok | — |
| Qwen/Qwen3-0.6B | qwen3 | ok | — |
| Qwen/Qwen3-8B | qwen3 | ok | — |
| Qwen/Qwen3-30B-A3B | qwen3_moe | ok | `norm_topk_prob`, `output_router_logits`, `router_aux_loss_coef` |
| Qwen/Qwen3-235B-A22B | qwen3_moe | ok | `norm_topk_prob`, `output_router_logits`, `router_aux_loss_coef` |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | qwen3_moe | ok | `norm_topk_prob`, `output_router_logits`, `router_aux_loss_coef` |
| Qwen/Qwen3-VL-235B-A22B-Instruct | qwen3_vl_moe | ok | `norm_topk_prob` |
| Qwen/Qwen3-Omni-30B-A3B-Instruct | qwen3_omni_moe | ok | `code2wav_config`, `enable_audio_output`, `norm_topk_prob`, `output_router_logits`, `router_aux_loss_coef`, `shared_expert_intermediate_size`, `talker_config` |
| google/gemma-7b | — | gated | — |
| google/gemma-2-27b | — | gated | — |
| google/gemma-3-4b-it | — | gated | — |
| google/gemma-3-27b-it | — | gated | — |
| google/recurrentgemma-2b | — | gated | — |
| deepseek-ai/deepseek-llm-67b-chat | llama | ok | — |
| deepseek-ai/DeepSeek-V2 | deepseek_v2 | ok | `aux_loss_alpha`, `n_group`, `norm_topk_prob`, `qk_nope_head_dim`, `qk_rope_head_dim`, `routed_scaling_factor`, `scoring_func`, `seq_aux`, `topk_group`, `topk_method`, `v_head_dim` |
| deepseek-ai/DeepSeek-V3 | deepseek_v3 | ok | `n_group`, `norm_topk_prob`, `qk_nope_head_dim`, `qk_rope_head_dim`, `quantization_config`, `routed_scaling_factor`, `scoring_func`, `topk_group`, `topk_method`, `v_head_dim` |
| deepseek-ai/DeepSeek-R1 | deepseek_v3 | ok | `n_group`, `norm_topk_prob`, `qk_nope_head_dim`, `qk_rope_head_dim`, `quantization_config`, `routed_scaling_factor`, `scoring_func`, `topk_group`, `topk_method`, `v_head_dim` |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | qwen2 | ok | — |
| deepseek-ai/DeepSeek-V3.1-Terminus | deepseek_v3 | ok | `attn_module_list_cfg`, `n_group`, `norm_topk_prob`, `qk_nope_head_dim`, `qk_rope_head_dim`, `quantization_config`, `routed_scaling_factor`, `scoring_func`, `topk_group`, `topk_method`, `v_head_dim` |
| deepseek-ai/DeepSeek-V3.2-Exp | deepseek_v32 | ok | `index_head_dim`, `index_n_heads`, `index_topk`, `n_group`, `norm_topk_prob`, `qk_nope_head_dim`, `qk_rope_head_dim`, `quantization_config`, `routed_scaling_factor`, `scoring_func`, `topk_group`, `topk_method`, `v_head_dim` |
| openai/gpt-oss-20b | gpt_oss | ok | `initial_context_length`, `output_router_logits`, `quantization_config`, `router_aux_loss_coef`, `swiglu_limit` |
| openai/gpt-oss-120b | gpt_oss | ok | `initial_context_length`, `output_router_logits`, `quantization_config`, `router_aux_loss_coef`, `swiglu_limit` |
| moonshotai/Kimi-K2-Instruct | kimi_k2 | ok | `aux_loss_alpha`, `n_group`, `norm_topk_prob`, `qk_nope_head_dim`, `qk_rope_head_dim`, `quantization_config`, `routed_scaling_factor`, `scoring_func`, `seq_aux`, `topk_group`, `topk_method`, `v_head_dim` |
| moonshotai/Kimi-K2-Thinking | kimi_k2 | ok | `aux_loss_alpha`, `n_group`, `norm_topk_prob`, `qk_nope_head_dim`, `qk_rope_head_dim`, `quantization_config`, `routed_scaling_factor`, `scoring_func`, `seq_aux`, `topk_group`, `topk_method`, `v_head_dim` |
| zai-org/GLM-4.5 | glm4_moe | ok | `n_group`, `norm_topk_prob`, `routed_scaling_factor`, `topk_group` |
| zai-org/GLM-4.6 | glm4_moe | ok | `n_group`, `norm_topk_prob`, `routed_scaling_factor`, `topk_group` |
| microsoft/phi-2 | phi | ok | — |
| microsoft/Phi-3-mini-4k-instruct | phi3 | ok | `original_max_position_embeddings` |
| microsoft/Phi-3.5-MoE-instruct | phimoe | ok | `input_jitter_noise`, `lm_head_bias`, `original_max_position_embeddings`, `output_router_logits`, `router_aux_loss_coef`, `router_jitter_noise` |
| microsoft/phi-4 | phi3 | ok | `original_max_position_embeddings` |
| microsoft/phi-4-multimodal-instruct | phi4mm | ok | `audio_processor`, `embd_layer`, `full_attn_mod`, `interpolate_factor`, `lm_head_bias`, `mlp_bias`, `original_max_position_embeddings`, `speech_lora`, `vision_lora` |

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
