# Serve audit

_Generated 2026-07-04 by `scripts/serve_audit.py` (render every catalogued model end-to-end)._

- Audited: **105**  ·  rendered clean: **92**  ·  schema problems: **0**  ·  coupling problems: **0**  ·  errored: **7**  ·  gated: **12**
- Catalogue ids seen: **111** (gated skipped up front: **6**)

## Needs attention

### mistralai/Pixtral-12B-2409 — load_error
- error: `ModelNotFoundError: Couldn't find or recognize 'mistralai/Pixtral-12B-2409'. Check the model id; if it's a newly released architecture, update transformers (`pip install -U transformers`) — your installed version may not know it yet.`

### Qwen/Qwen3-Omni-30B-A3B-Instruct — load_error
- error: `ConfigParseError: Loaded 'Qwen/Qwen3-Omni-30B-A3B-Instruct' but couldn't parse a usable model — no transformer layers were found (missing num_hidden_layers and all known aliases, or this isn't a decoder transformer config). If it's a brand-`

### tencent/HunyuanVideo-1.5 — load_error
- error: `ConfigParseError: Loaded 'tencent/HunyuanVideo-1.5' but couldn't parse a usable model — no transformer layers were found (missing num_hidden_layers and all known aliases, or this isn't a decoder transformer config). If it's a brand-new arch`

### tencent/HunyuanVideo-I2V — load_error
- error: `ModelNotFoundError: 'tencent/HunyuanVideo-I2V' has no unfoldable model config — its config.json is a stub (keys: Name) with no architecture, model_type, or diffusers pipeline (no model_index.json). This is typically an ORIGINAL-RELEASE repo`

### h94/IP-Adapter — load_error
- error: `ModelNotFoundError: Couldn't find or recognize 'h94/IP-Adapter'. Check the model id; if it's a newly released architecture, update transformers (`pip install -U transformers`) — your installed version may not know it yet.`

### h94/IP-Adapter-FaceID — load_error
- error: `ModelNotFoundError: Couldn't find or recognize 'h94/IP-Adapter-FaceID'. Check the model id; if it's a newly released architecture, update transformers (`pip install -U transformers`) — your installed version may not know it yet.`

### latent-consistency/lcm-lora-sdxl — load_error
- error: `ModelNotFoundError: Couldn't find or recognize 'latent-consistency/lcm-lora-sdxl'. Check the model id; if it's a newly released architecture, update transformers (`pip install -U transformers`) — your installed version may not know it yet.`


## Partial-config warnings (⚠)

| Model | warnings |
| --- | --- |
| moonshotai/Kimi-K2-Instruct | Modeling source is unavailable; the positional scheme remains unknown. |
| moonshotai/Kimi-K2-Thinking | Modeling source is unavailable; the positional scheme remains unknown. |
| THUDM/glm-4-9b-chat | Modeling source is unavailable; the positional scheme remains unknown. |
| microsoft/phi-4-multimodal-instruct | Modeling source is unavailable; the positional scheme remains unknown. |
| allenai/OLMo-7B | Modeling source is unavailable; the positional scheme remains unknown. |
| nvidia/Nemotron-H-8B-Base-8K | Modeling source is present but the configured positional scheme is unresolved.; Config layer_types contains unrecognized value 'mamba' — treated as causal.; Config layer_types contains unrecognized value 'mlp' — treated as causal. |
| inclusionAI/Ling-lite | Modeling source is unavailable; the positional scheme remains unknown. |
| inclusionAI/Ling-plus | Modeling source is unavailable; the positional scheme remains unknown. |
| LiquidAI/LFM2-1.2B | Config layer_types contains unrecognized value 'conv' — treated as causal. |
| MiniMaxAI/MiniMax-Text-01 | Modeling source is unavailable; the positional scheme remains unknown. |
| Snowflake/snowflake-arctic-instruct | Modeling source is unavailable; the positional scheme remains unknown. |
| ai21labs/Jamba-v0.1 | Modeling source is present but the configured positional scheme is unresolved.; Config layer_types contains unrecognized value 'mamba' — treated as causal. |
| baichuan-inc/Baichuan2-13B-Base | Modeling source is unavailable; the positional scheme remains unknown. |
| internlm/internlm2_5-7b | Modeling source is unavailable; the positional scheme remains unknown. |
| tencent/HunyuanImage-3.0 | Modeling source is unavailable; the positional scheme remains unknown. |
| kandinsky-community/kandinsky-3 | This UNet config declares no down_block_types/up_block_types — per-stage attention placement is defined in the model code, not the config, so the denoiser is shown as a convolutional U skeleton with text cross-attention (dim 4096) entering at code-defined stages. |

## Per-model detail

| Model | arch | status | layers | ⚠ | ⓘ | schema | coupling | unparsed | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huggyllama/llama-7b | LlamaForCausalLM | ok | 32 | — | — | — | — | — | 8.5 |
| codellama/CodeLlama-34b-hf | LlamaForCausalLM | ok | 48 | — | — | — | — | — | 0.4 |
| mistralai/Mistral-7B-v0.1 | MistralForCausalLM | ok | 32 | — | — | — | — | — | 0.4 |
| mistralai/Mixtral-8x7B-v0.1 | MixtralForCausalLM | ok | 32 | — | — | — | — | — | 0.5 |
| mistralai/Mistral-Small-24B-Instruct-2501 | MistralForCausalLM | ok | 40 | — | — | — | — | — | 0.4 |
| mistralai/Pixtral-12B-2409 | — | load_error | — | — | — | — | — | — | 0.8 |
| Qwen/Qwen2.5-72B | Qwen2ForCausalLM | ok | 80 | — | — | — | — | — | 0.5 |
| Qwen/QwQ-32B | Qwen2ForCausalLM | ok | 64 | — | — | — | — | — | 0.4 |
| Qwen/Qwen3-0.6B | Qwen3ForCausalLM | ok | 28 | — | — | — | — | — | 0.4 |
| Qwen/Qwen3-8B | Qwen3ForCausalLM | ok | 36 | — | — | — | — | — | 0.4 |
| Qwen/Qwen3-30B-A3B | Qwen3MoeForCausalLM | ok | 48 | — | — | — | — | — | 0.4 |
| Qwen/Qwen3-235B-A22B | Qwen3MoeForCausalLM | ok | 94 | — | — | — | — | — | 0.4 |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | Qwen3MoeForCausalLM | ok | 94 | — | — | — | — | — | 0.4 |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | Qwen3MoeForCausalLM | ok | 48 | — | — | — | — | — | 0.8 |
| Qwen/Qwen2-VL-7B-Instruct | Qwen2VLForConditionalGeneration | ok | 28 | — | — | — | — | — | 1.1 |
| Qwen/Qwen3-VL-235B-A22B-Instruct | Qwen3VLMoeForConditionalGeneration | ok | 94 | — | — | — | — | — | 1.0 |
| Qwen/Qwen3-Omni-30B-A3B-Instruct | — | load_error | — | — | — | — | — | — | 1.1 |
| google/gemma-2-27b | Gemma2ForCausalLM | ok | 46 | — | — | — | — | `sliding_window_size` | 0.7 |
| google/gemma-3-27b-it | Gemma3ForConditionalGeneration | ok | 62 | — | — | — | — | `_sliding_window_pattern`, `boi_token_index`, `eoi_token_index` | 0.8 |
| google/diffusiongemma-26B-A4B-it | DiffusionGemmaForBlockDiffusion | ok | 30 | — | — | — | — | — | 1.2 |
| deepseek-ai/deepseek-llm-67b-chat | LlamaForCausalLM | ok | 95 | — | — | — | — | — | 0.4 |
| deepseek-ai/DeepSeek-V2 | DeepseekV2ForCausalLM | ok | 60 | — | — | — | — | — | 0.4 |
| deepseek-ai/DeepSeek-V3 | DeepseekV3ForCausalLM | ok | 61 | — | — | — | — | — | 0.5 |
| deepseek-ai/DeepSeek-R1 | DeepseekV3ForCausalLM | ok | 61 | — | — | — | — | — | 0.5 |
| deepseek-ai/DeepSeek-V3.1-Terminus | DeepseekV3ForCausalLM | ok | 61 | — | — | — | — | `attn_module_list_cfg` | 0.5 |
| deepseek-ai/DeepSeek-V3.2-Exp | DeepseekV32ForCausalLM | ok | 61 | — | — | — | — | `mlp_layer_types` | 0.5 |
| openai/gpt-oss-20b | GptOssForCausalLM | ok | 24 | — | — | — | — | — | 0.4 |
| openai/gpt-oss-120b | GptOssForCausalLM | ok | 36 | — | — | — | — | — | 0.4 |
| moonshotai/Kimi-K2-Instruct | DeepseekV3ForCausalLM | ok | 61 | 1 | — | — | — | — | 0.9 |
| moonshotai/Kimi-K2-Thinking | DeepseekV3ForCausalLM | ok | 61 | 1 | — | — | — | — | 1.0 |
| THUDM/glm-4-9b-chat | ChatGLMModel | ok | 40 | 1 | — | — | — | `add_bias_linear`, `add_qkv_bias`, `apply_query_key_layer_scaling`, `attn_implementation`, `bias_dropout_fusion`, `fp32_residual_connection`, `layernorm_epsilon`, `multi_query_attention`, `original_rope`, `post_layer_norm`, `rmsnorm`, `rope_ratio` | 1.7 |
| zai-org/GLM-4.5 | Glm4MoeForCausalLM | ok | 92 | — | — | — | — | — | 0.5 |
| zai-org/GLM-4.6 | Glm4MoeForCausalLM | ok | 92 | — | — | — | — | — | 0.4 |
| microsoft/phi-2 | PhiForCausalLM | ok | 32 | — | — | — | — | — | 0.4 |
| microsoft/Phi-3-mini-4k-instruct | Phi3ForCausalLM | ok | 32 | — | — | — | — | `original_max_position_embeddings` | 0.4 |
| microsoft/phi-4 | Phi3ForCausalLM | ok | 40 | — | — | — | — | `original_max_position_embeddings` | 0.4 |
| microsoft/Phi-3.5-MoE-instruct | PhiMoEForCausalLM | ok | 32 | — | — | — | — | `lm_head_bias` | 0.6 |
| microsoft/phi-4-multimodal-instruct | Phi4MMForCausalLM | ok | 32 | 1 | — | — | — | `audio_processor`, `embd_layer`, `full_attn_mod`, `interpolate_factor`, `lm_head_bias`, `original_max_position_embeddings`, `speech_lora`, `vision_lora` | 1.6 |
| tiiuae/falcon-7b | FalconForCausalLM | ok | 32 | — | — | — | — | `bias`, `new_decoder_architecture` | 0.5 |
| tiiuae/falcon-40b | FalconForCausalLM | ok | 60 | — | — | — | — | `bias`, `new_decoder_architecture` | 0.4 |
| tiiuae/Falcon3-10B-Base | LlamaForCausalLM | ok | 40 | — | — | — | — | — | 0.4 |
| databricks/dbrx-base | — | gated | — | — | — | — | — | — | 0.2 |
| 01-ai/Yi-34B | LlamaForCausalLM | ok | 60 | — | — | — | — | — | 0.4 |
| 01-ai/Yi-1.5-34B | LlamaForCausalLM | ok | 60 | — | — | — | — | — | 0.4 |
| allenai/OLMo-7B | OLMoForCausalLM | ok | 32 | 1 | — | — | — | `alibi_bias_max`, `attention_layer_norm`, `attention_layer_norm_with_affine`, `bias_for_layer_norm`, `block_group_size`, `block_type`, `embedding_dropout`, `embedding_size`, `flash_attention`, `include_bias`, `init_device`, `init_fn`, `init_std`, `layer_norm_type`, `layer_norm_with_affine`, `mlp_hidden_size`, `multi_query_attention`, `precision`, `residual_dropout`, `rope`, `rope_full_precision`, `scale_logits`, `weight_tying` | 0.9 |
| allenai/OLMo-2-1124-13B | Olmo2ForCausalLM | ok | 40 | — | — | — | — | — | 0.4 |
| CohereForAI/c4ai-command-r-v01 | — | gated | — | — | — | — | — | — | 0.3 |
| ibm-granite/granite-3.3-8b-instruct | GraniteForCausalLM | ok | 40 | — | — | — | — | `attention_multiplier`, `embedding_multiplier`, `logits_scaling`, `residual_multiplier` | 0.4 |
| LGAI-EXAONE/EXAONE-4.0-32B | Exaone4ForCausalLM | ok | 64 | — | — | — | — | — | 0.4 |
| nvidia/Nemotron-H-8B-Base-8K | NemotronHForCausalLM | ok | 52 | 3 | — | — | — | `attention_head_dim`, `chunk_size`, `conv_kernel`, `expand`, `layers_block_type`, `mamba_head_dim`, `mamba_hidden_act`, `mamba_num_heads`, `mamba_proj_bias`, `mamba_ssm_cache_dtype`, `mlp_hidden_act`, `moe_shared_expert_intermediate_size`, `moe_shared_expert_overlap`, `mtp_layers_block_type`, `n_groups`, `num_logits_to_keep`, `rescale_prenorm_residual`, `residual_in_fp32`, `ssm_state_size`, `time_step_floor`, `time_step_limit`, `time_step_max`, `time_step_min`, `time_step_rank`, `use_bias`, `use_conv_bias`, `use_mamba_kernels` | 0.5 |
| nvidia/Llama-3_1-Nemotron-51B | — | gated | — | — | — | — | — | — | 0.3 |
| ByteDance-Seed/Seed-OSS-36B-Instruct | SeedOssForCausalLM | ok | 64 | — | — | — | — | `attention_out_bias`, `residual_dropout` | 0.4 |
| inclusionAI/Ling-lite | BailingMoeForCausalLM | ok | 28 | 1 | — | — | — | `embedding_dropout`, `norm_head`, `norm_softmax`, `output_dropout`, `use_bias`, `use_qkv_bias` | 0.8 |
| inclusionAI/Ling-plus | BailingMoeForCausalLM | ok | 88 | 1 | — | — | — | `embedding_dropout`, `norm_head`, `norm_softmax`, `output_dropout`, `use_bias`, `use_qkv_bias` | 1.0 |
| LiquidAI/LFM2-1.2B | Lfm2ForCausalLM | ok | 16 | 1 | — | — | — | `block_auto_adjust_ff_dim`, `block_dim`, `block_ffn_dim_multiplier`, `block_mlp_init_scale`, `block_multiple_of`, `block_norm_eps`, `block_out_init_scale`, `block_use_swiglu`, `block_use_xavier_init`, `conv_L_cache`, `conv_bias`, `conv_dim`, `conv_dim_out`, `conv_use_xavier_init`, `full_attn_idxs`, `use_pos_enc` | 0.4 |
| MiniMaxAI/MiniMax-Text-01 | MiniMaxText01ForCausalLM | ok | 80 | 1 | — | — | — | `attn_type_list`, `layernorm_full_attention_alpha`, `layernorm_full_attention_beta`, `layernorm_linear_attention_alpha`, `layernorm_linear_attention_beta`, `layernorm_mlp_alpha`, `layernorm_mlp_beta`, `postnorm`, `shared_intermediate_size`, `shared_moe_mode` | 0.9 |
| Snowflake/snowflake-arctic-instruct | ArcticForCausalLM | ok | 35 | 1 | — | — | — | `enable_expert_tensor_parallelism`, `moe_eval_capacity_factor`, `moe_layer_frequency`, `moe_min_capacity`, `moe_token_dropping`, `moe_train_capacity_factor`, `parallel_attn_mlp_res`, `use_residual` | 0.9 |
| ai21labs/Jamba-v0.1 | JambaForCausalLM | ok | 32 | 2 | — | — | — | `attn_layer_offset`, `attn_layer_period`, `expert_layer_offset`, `expert_layer_period`, `mamba_conv_bias`, `mamba_d_conv`, `mamba_d_state`, `mamba_dt_rank`, `mamba_expand`, `mamba_proj_bias`, `num_logits_to_keep`, `use_mamba_kernels` | 0.5 |
| stabilityai/stablelm-2-12b | StableLmForCausalLM | ok | 40 | — | — | — | — | `rotary_scaling_factor`, `use_norm_bias`, `use_qkv_bias` | 0.4 |
| HuggingFaceTB/SmolLM2-1.7B | LlamaForCausalLM | ok | 24 | — | — | — | — | — | 0.4 |
| baichuan-inc/Baichuan2-13B-Base | BaichuanForCausalLM | ok | 40 | 1 | — | — | — | `_from_model_config`, `gradient_checkpointing`, `model_max_length` | 0.9 |
| internlm/internlm2_5-7b | InternLM2ForCausalLM | ok | 32 | 1 | — | — | — | `attn_implementation`, `bias` | 0.9 |
| EleutherAI/gpt-neox-20b | GPTNeoXForCausalLM | ok | 44 | — | — | — | — | `attention_probs_dropout_prob`, `hidden_dropout_prob` | 0.5 |
| EleutherAI/gpt-j-6b | GPTJForCausalLM | ok | 28 | — | — | — | — | `gradient_checkpointing`, `rotary`, `scale_attn_weights`, `summary_first_dropout`, `summary_proj_to_labels`, `summary_type`, `summary_use_proj` | 0.5 |
| EleutherAI/pythia-12b | GPTNeoXForCausalLM | ok | 36 | — | — | — | — | — | 0.6 |
| bigscience/bloom | BloomForCausalLM | ok | 70 | — | — | — | — | — | 0.5 |
| facebook/opt-66b | OPTForCausalLM | ok | 64 | — | — | — | — | `_remove_final_layer_norm`, `activation_dropout`, `do_layer_norm_before`, `enable_bias`, `init_std`, `layer_norm_elementwise_affine`, `layerdrop`, `word_embed_proj_dim` | 0.4 |
| mosaicml/mpt-7b | — | gated | — | — | — | — | — | — | 0.2 |
| black-forest-labs/FLUX.1-dev | FluxTransformer2DModel | ok | 57 | — | — | — | — | — | 4.8 |
| black-forest-labs/FLUX.2-dev | Flux2Transformer2DModel | ok | 56 | — | — | — | — | `multimodal_projector_bias` | 2.9 |
| stabilityai/stable-diffusion-3-medium-diffusers | — | gated | — | — | — | — | — | — | 0.3 |
| Qwen/Qwen-Image | QwenImageTransformer2DModel | ok | 60 | — | — | — | — | — | 3.9 |
| Qwen/Qwen-Image-Edit | QwenImageTransformer2DModel | ok | 60 | — | — | — | — | — | 3.0 |
| fal/AuraFlow-v0.3 | AuraFlowTransformer2DModel | ok | 36 | — | — | — | — | — | 3.1 |
| HiDream-ai/HiDream-I1-Full | HiDreamImageTransformer2DModel | ok | 48 | — | — | — | — | — | 4.4 |
| Alpha-VLLM/Lumina-Image-2.0 | Lumina2Transformer2DModel | ok | 26 | — | — | — | — | — | 2.6 |
| Alpha-VLLM/Lumina-Next-SFT-diffusers | LuminaNextDiT2DModel | ok | 24 | — | — | — | — | — | 2.5 |
| shuttleai/shuttle-3-diffusion | FluxTransformer2DModel | ok | 57 | — | — | — | — | — | 3.4 |
| OmniGen2/OmniGen2 | OmniGen2Transformer2DModel | ok | 32 | — | — | — | — | — | 1.8 |
| PixArt-alpha/PixArt-XL-2-1024-MS | Transformer2DModel | ok | 28 | — | — | — | — | — | 2.9 |
| PixArt-alpha/PixArt-Sigma-XL-2-1024-MS | Transformer2DModel | ok | 28 | — | — | — | — | — | 2.9 |
| Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers | HunyuanDiT2DModel | ok | 40 | — | — | — | — | `attention_probs_dropout_prob`, `directionality`, `hidden_dropout_prob`, `pooler_fc_size`, `pooler_num_attention_heads`, `pooler_num_fc_layers`, `pooler_size_per_head`, `pooler_type`, `position_embedding_type`, `type_vocab_size` | 3.6 |
| Efficient-Large-Model/Sana_1600M_1024px_diffusers | SanaTransformer2DModel | ok | 20 | — | — | — | — | — | 2.7 |
| THUDM/CogView3-Plus-3B | CogView3PlusTransformer2DModel | ok | 30 | — | — | — | — | — | 4.6 |
| THUDM/CogView4-6B | CogView4Transformer2DModel | ok | 28 | — | — | — | — | — | 4.3 |
| tencent/HunyuanImage-3.0 | HunyuanImage3ForCausalMM | ok | 32 | 1 | — | — | — | `add_classification_head`, `anyres_pooling_size`, `anyres_vit_two_views`, `attention_head_dim`, `cla_share_factor`, `class_num`, `cond_image_type`, `cond_token_attn_type`, `dense_list`, `group_limited_greedy`, `im_end_id`, `im_newline_id`, `im_start_id`, `mask_init_id`, `model_version`, `moe_drop_tokens`, `moe_layer_num_skipped`, `moe_random_routing_dropped_token`, `moe_topk`, `num_media_embeds`, `num_shared_expert`, `pad_id`, `pool_type`, `position_embedding_xdrope`, `rope_type`, `skip_cls_token`, `text_end_id`, `text_start_id`, `use_cla`, `use_mixed_mlp_moe`, `use_mla`, `use_rotary_pos_emb`, `vae_autocast_dtype`, `vae_downsample_factor`, `vae_dtype`, `vae_type`, `video_end_id`, `video_start_id`, `vit`, `vit_add_patchemb_bias`, `vit_aligner`, `vit_downsample_factor`, `vit_input_resolution`, `vit_mapping_type`, `vit_norm_type`, `vit_patch`, `vit_processor`, `vit_remove_prenorm`, `vit_token`, `vit_type`, `vit_used_rms_norm` | 0.9 |
| Wan-AI/Wan2.1-T2V-1.3B-Diffusers | WanTransformer3DModel | ok | 30 | — | — | — | — | — | 3.3 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | WanTransformer3DModel | ok | 30 | — | — | — | — | — | 2.8 |
| hunyuanvideo-community/HunyuanVideo | HunyuanVideoTransformer3DModel | ok | 60 | — | — | — | — | — | 3.5 |
| tencent/HunyuanVideo-1.5 | — | load_error | — | — | — | — | — | — | 1.1 |
| tencent/HunyuanVideo-I2V | — | load_error | — | — | — | — | — | — | 0.8 |
| THUDM/CogVideoX-5b | CogVideoXTransformer3DModel | ok | 42 | — | — | — | — | — | 5.1 |
| Lightricks/LTX-Video | LTXVideoTransformer3DModel | ok | 28 | — | — | — | — | — | 2.8 |
| genmo/mochi-1-preview | MochiTransformer3DModel | ok | 48 | — | — | — | — | — | 3.2 |
| rhymes-ai/Allegro | AllegroTransformer3DModel | ok | 32 | — | — | — | — | — | 3.1 |
| stable-diffusion-v1-5/stable-diffusion-v1-5 | UNet2DConditionModel | ok | 0 | — | — | — | — | — | 2.4 |
| stabilityai/stable-diffusion-xl-base-1.0 | UNet2DConditionModel | ok | 0 | — | — | — | — | — | 2.6 |
| kandinsky-community/kandinsky-3 | Kandinsky3UNet | ok | 0 | 1 | — | — | — | — | 2.5 |
| stabilityai/stable-video-diffusion-img2vid-xt | UNetSpatioTemporalConditionModel | ok | 0 | — | — | — | — | — | 1.9 |
| stabilityai/stable-audio-open-1.0 | — | gated | — | — | — | — | — | — | 0.3 |
| diffusers/controlnet-canny-sdxl-1.0 | ControlNetModel | ok | 0 | — | — | — | — | — | 1.1 |
| xinsir/controlnet-union-sdxl-1.0 | ControlNetModel | ok | 0 | — | — | — | — | — | 0.9 |
| h94/IP-Adapter | — | load_error | — | — | — | — | — | — | 0.9 |
| h94/IP-Adapter-FaceID | — | load_error | — | — | — | — | — | — | 0.9 |
| latent-consistency/lcm-lora-sdxl | — | load_error | — | — | — | — | — | — | 0.8 |
