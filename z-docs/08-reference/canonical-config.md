# Canonical config reference — the standpoint of what fields a config should have

> **Status: `#TODO` — not consumed by any code yet.** This reference is the intended
> standpoint for the (not-yet-built) [`true_config()`](../01-product/true-config.md)
> capability. The field set is now **harvested from real declarations**, not authored
> from memory, but it is not yet wired to anything.

## Evidence base

Every field below is recorded from a real declaration on this machine. Nothing is
invented, and no value is illustrative:

| Source | What it supplies | Extent |
|---|---|---|
| Installed `transformers` config classes | each family's declared field set + defaults | 37 families harvested (656 registered) |
| Installed `diffusers` model/scheduler `__init__` signatures | the diffusers config schema | 21 classes |
| Blessed corpus fixtures | real serialized checkpoint values | 25 checkpoints, 229 distinct fields |

Field annotations below read `[tag] — n families, e.g. …` where `n` is the count of
harvested families that declare the field. A field declared by exactly one family is
marked as such; it is still legitimate, but it is a dialect, not a norm.

## What this is

A curated **superset** of the config fields that carry architectural meaning across
the supported mechanism surface. It is the *menu*, not a *loadable model*: no real
checkpoint uses all of these at once (a model is dense or MoE, standard-attention or
MLA, transformer or diffusion). It exists so that:

- `true_config()` fills only the fields a given model's code actually reads, drawn
  from this vocabulary — never inventing a field outside it;
- a code-read field that is *not* here is a signal to extend this reference
  (curated growth), not a licence to guess.

Each field is tagged by role:

- `[value]` — a checkpoint geometry/number the code reads; stays config, never converts to code.
- `[enum]` — a declared semantic choice the source dispatches on.
- `[schedule]` — a per-layer list/gate; the *shape* is code-proven, the *value* stays config.
- `[scale]` — a multiplier/softcap applied in the math.
- `[token]` / `[runtime]` / `[address]` — non-structural bookkeeping (kept, but never drives architecture).

Foreign input formats may use different spellings. They are recognized from
file layout plus required structural keys and normalized through scope-qualified
aliases. A local spelling cannot be promoted into the global alias set when it
collides elsewhere, and input-format recognition cannot manufacture
`model_type`, `architectures`, or mechanism facts.

## Three corrections the harvest forced

Recorded because they falsify plausible assumptions, including ones this document
previously asserted:

1. **RoPE is not a top-level pair.** In the installed version, 28 families declare a
   single `rope_parameters` object; `rope_theta`/`rope_scaling` are not top-level
   class fields. The real DeepSeek-V3 checkpoint serializes `rope_parameters`
   (carrying `beta_fast`/`beta_slow`/`factor`/`mscale`) and has **no** `rope_scaling`.
2. **Some "code-only" structural facts *are* declared fields in some families.**
   Gating, norm placement, and norm multiplicity are code-proven *in general*, but
   `is_gated_act` (T5/UMT5), `apply_residual_connection_post_layernorm` (BLOOM), and
   `num_ln_in_parallel_attn` (Falcon) are genuine config fields. Where a family
   declares one, it is legitimate config; the code still decides, and the field is
   the checkpoint's selection.
3. **A field can be checkpoint-real yet never a class default.** `scoring_func` and
   `topk_method` appear in the real DeepSeek-V3 config but are declared by almost no
   config class — which is exactly why a checkpoint that omits them cannot be read
   from config alone. Class-default harvesting alone would have missed them; real
   checkpoints alone would have missed unserialized defaults. Both sources are
   required.

**Excluded on evidence, not assumption:** no harvested family declares a field for
attention-kind label, fused Q/K/V or gate-up storage, or the router's score
transform as a *structural* choice. Those remain code-proven and live in the diagram
and facts, never here.

---

# Part I — Transformer decoder family

## §1 Source address and identity — locates code, never a fact
```jsonc
{
  "model_type": "llama",                    // [address]
  "architectures": ["LlamaForCausalLM"],    // [address]
  "auto_map": {},                           // [address] remote-code pointer
  "_repo_id": null,                         // [address] loader provenance stamp
  "_name_or_path": ""                       // [address] exporter-baked; outranked by _repo_id
}
```

## §2 Stack geometry
```jsonc
{
  "hidden_size": 4096,               // [value] 28 families
  "num_hidden_layers": 32,           // [value] 27 families
  "num_attention_heads": 32,         // [value] 27 families
  "intermediate_size": 11008,        // [value] 26 families
  "max_position_embeddings": 2048,   // [value] 27 families
  "vocab_size": 32000,               // [value] 34 families
  "tie_word_embeddings": false,      // [value] 36 families — the most universal field harvested

  // legacy GPT-2 lineage spellings (codegen, gpt2, gptj, bloom)
  "n_layer": 2, "n_head": 8, "n_positions": 2048, "n_inner": null,
  // T5/DBRX lineage
  "d_model": 512, "num_heads": 8
}
```

## §3 Vocabulary and special tokens
```jsonc
{
  "bos_token_id": 1,    // [token] 32 families
  "eos_token_id": 2,    // [token] 34 families
  "pad_token_id": null, // [token] 33 families
  "sep_token_id": null  // [token]
}
```

## §4 Attention — head layout and projection bias
```jsonc
{
  "num_key_value_heads": 32,   // [value] 25 families — GQA; == heads is MHA, 1 is MQA
  "head_dim": 128,             // [value] 13 families
  "attention_bias": false,     // [value] 18 families
  "attention_dropout": 0.0,    // [value] 28 families
  "qkv_bias": true,            // [value] qwen2_moe
  "use_qkv_bias": false,       // [value] stablelm
  "multi_query": true,         // [enum]  falcon — pairs with num_kv_heads
  "num_kv_heads": 71,          // [value] falcon
  "clip_qkv": null,            // [value] olmoe
  "lm_head_bias": false        // [value] phimoe
}
```

## §5 Attention — score scaling and logit softcapping
```jsonc
{
  "attention_multiplier": 1.0,        // [scale] granite, granitemoe
  "query_pre_attn_scalar": 256,       // [scale] gemma2, gemma3_text
  "attn_logit_softcapping": 50.0,     // [scale] gemma2, gemma3_text
  "final_logit_softcapping": 30.0,    // [scale] gemma2, gemma3_text
  "attn_scale": 0.1,                  // [scale] llama4_text — temperature tuning
  "attn_temperature_tuning": true,    // [enum]  llama4_text
  "floor_scale": 8192,                // [value] llama4_text
  "scale_attn_weights": true,         // [enum]  gpt2
  "scale_attn_by_inverse_layer_idx": false, // [enum] gpt2
  "reorder_and_upcast_attn": false    // [runtime] gpt2
}
```

## §6 Attention — context windowing and per-layer attention type
```jsonc
{
  "sliding_window": null,             // [value]    13 families — the window SIZE
  "use_sliding_window": false,        // [enum]     qwen2_moe, qwen3, qwen3_moe
  "max_window_layers": 28,            // [value]    qwen2_moe, qwen3
  "layer_types": ["full_attention"],  // [schedule]  9 families — the authoritative per-layer list
  "_sliding_window_pattern": 6,       // [schedule] gemma3_text — note the private underscore prefix
  "attention_chunk_size": 8192,       // [value]    llama4_text — chunked/local span
  "use_bidirectional_attention": null // [enum]     gemma2, gemma3_text
}
```

## §7 Attention — query/key normalization
```jsonc
{
  "qk_layernorm": false,  // [enum] persimmon, stablelm
  "use_qk_norm": true     // [enum] glm4_moe, llama4_text — the second real spelling
}
```

## §8 Attention — latent compression (MLA)
```jsonc
{
  "q_lora_rank": 1536,     // [value] deepseek_v2/v3
  "kv_lora_rank": 512,     // [value] deepseek_v2/v3 — the MLA signal
  "qk_rope_head_dim": 64,  // [value] deepseek_v2/v3
  "qk_nope_head_dim": 128, // [value] deepseek_v2/v3
  "qk_head_dim": 192,      // [value] deepseek_v3
  "v_head_dim": 128        // [value] deepseek_v2/v3
}
```

## §9 Attention — linear and hybrid mixer sizing
```jsonc
{
  "linear_num_key_heads": 16,    // [value] qwen3_next
  "linear_num_value_heads": 32,  // [value] qwen3_next
  "linear_key_head_dim": 128,    // [value] qwen3_next
  "linear_value_head_dim": 128,  // [value] qwen3_next
  "full_attn_alpha_factor": 1,   // [scale] minimax
  "full_attn_beta_factor": 1,    // [scale] minimax
  "linear_attn_alpha_factor": 1, // [scale] minimax
  "linear_attn_beta_factor": 1   // [scale] minimax
}
```

## §10 Positional encoding
```jsonc
{
  // the installed home for RoPE — one object, 28 families
  "rope_parameters": {
    "rope_type": "default",   // [enum]  default | linear | dynamic | yarn | llama3 | longrope
    "rope_theta": 10000.0,    // [value] base frequency
    "factor": 40,             // [value] extension factor
    "beta_fast": 32,          // [value] yarn
    "beta_slow": 1,           // [value] yarn
    "mscale": 1.0,            // [value] yarn/deepseek
    "original_max_position_embeddings": 4096 // [value]
  },

  "partial_rotary_factor": 0.5,  // [value]    glm4_moe, persimmon, qwen3_next, stablelm
  "rotary_dim": 64,              // [value]    codegen, gptj — the legacy spelling
  "rope_interleave": true,       // [enum]     deepseek_v3
  "no_rope_layers": [1,1,1,0],   // [schedule] llama4_text — the authoritative NoPE list
  "no_rope_layer_interval": 4,   // [schedule] llama4_text — cadence fallback
  "original_max_position_embeddings": 4096   // [value] phi3
}
```

## §11 Feed-forward — width and activation
```jsonc
{
  "hidden_act": "silu",              // [enum]  24 families
  "hidden_activation": "gelu_pytorch_tanh", // [enum] gemma2, gemma3_text
  "activation_function": "gelu_new", // [enum]  codegen, gpt2, gptj
  "activation": "gelu",              // [enum]  falcon
  "dense_act_fn": "relu",            // [enum]  t5, umt5
  "feed_forward_proj": "relu",       // [enum]  t5, umt5 — may carry a "gated-" prefix
  "is_gated_act": false,             // [enum]  t5, umt5 — gating DECLARED in config
  "mlp_bias": false,                 // [value] deepseek_v2, granite, llama
  "intermediate_size_mlp": 16384,    // [value] llama4_text — dense-MLP width alias
  "ffn_hidden_size": 18176,          // [value] falcon
  "n_inner": null                    // [value] codegen, gpt2, gptj — null ⇒ code derives 4×hidden
}
```

## §12 Mixture of experts — expert capacity
```jsonc
{
  "num_experts_per_tok": 8,             // [value] 13 families — top-k
  "num_local_experts": 16,              // [value]  7 families (gpt_oss, llama4, mixtral, …)
  "n_routed_experts": 256,              // [value]  3 families (deepseek_v2/v3, glm4_moe)
  "num_experts": 60,                    // [value]  3 families (olmoe, qwen2_moe, qwen3_next)
  "moe_intermediate_size": 2048,        // [value]  6 families
  "n_shared_experts": 1,                // [value]  deepseek_v2/v3, glm4_moe
  "shared_expert_intermediate_size": 5632 // [value] qwen2_moe, qwen3_next
}
```
Three spellings for one concept (`num_local_experts` / `n_routed_experts` /
`num_experts`) is the normal case, not an anomaly; alias vocabulary resolves them.

## §13 Mixture of experts — layer schedule
```jsonc
{
  "moe_layers": [0,1,2],            // [schedule] llama4_text — explicit MoE index list
  "first_k_dense_replace": 3,       // [schedule] deepseek_v2/v3, glm4_moe — dense prefix
  "decoder_sparse_step": 1,         // [schedule] qwen2_moe, qwen3_moe, qwen3_next — cadence
  "interleave_moe_layer_step": 1,   // [schedule] llama4_text
  "mlp_only_layers": []             // [schedule] qwen2_moe, qwen3_moe, qwen3_next — exclusion
}
```

## §14 Mixture of experts — routing and selection
```jsonc
{
  "norm_topk_prob": true,        // [value]  7 families
  "n_group": 8,                  // [value]  deepseek_v2/v3, glm4_moe — group routing
  "topk_group": 4,               // [value]  deepseek_v2/v3, glm4_moe
  "routed_scaling_factor": 2.5,  // [scale]  deepseek_v2/v3, glm4_moe
  "topk_method": "greedy",       // [enum]   deepseek_v2 only as a class default
  "scoring_func": "sigmoid",     // [enum]   checkpoint-serialized (DeepSeek-V3); NOT a class default
  "router_jitter_noise": 0.0,    // [runtime] llama4_text, minimax, mixtral, …
  "input_jitter_noise": 0.0,     // [runtime] phimoe
  "router_aux_loss_coef": 0.001, // [runtime] 10 families
  "output_router_logits": false  // [runtime] 11 families
}
```
`scoring_func`/`topk_method` are the standing proof that config alone is insufficient:
a family that copies the routing *code* without the *strings* is unreadable from
config, so the score transform is code-proven and these fields only confirm.

## §15 Normalization — epsilon and declared topology
```jsonc
{
  "rms_norm_eps": 1e-06,      // [value] 21 families
  "layer_norm_epsilon": 1e-05,// [value]  7 families (bloom, codegen, falcon, gpt2, gptj, t5, …)
  "layer_norm_eps": 1e-05,    // [value]  4 families (cohere2, gpt_neox, persimmon, stablelm)
  "norm_epsilon": 1e-05,      // [value]  starcoder2
  "apply_residual_connection_post_layernorm": false, // [enum] bloom — placement DECLARED
  "num_ln_in_parallel_attn": null                    // [value] falcon — norm multiplicity DECLARED
}
```
The epsilon *spelling* never decides the norm kind — the norm's arithmetic does
(families exist that construct `nn.LayerNorm` while carrying `rms_norm_eps`).

## §16 Residual topology
```jsonc
{
  "use_parallel_residual": false, // [enum]  gpt_neox, stablelm
  "parallel_attn": true,          // [enum]  falcon
  "residual_dropout": 0.0,        // [runtime] starcoder2
  "hidden_dropout": 0.0,          // [runtime] bloom, gpt_neox, stablelm, …
  "resid_pdrop": 0.0,             // [runtime] dbrx, gptj, phi3
  "embd_pdrop": 0.0,              // [runtime] codegen, gptj, phi3
  "attn_pdrop": 0.0               // [runtime] codegen, gpt2, gptj
}
```

## §17 Stream and logit multipliers
```jsonc
{
  "embedding_multiplier": 1.0, // [scale] granite, granitemoe
  "residual_multiplier": 1.0,  // [scale] granite, granitemoe
  "logits_scaling": 1.0,       // [scale] granite, granitemoe
  "mlp_alpha_factor": 1,       // [scale] minimax
  "mlp_beta_factor": 1         // [scale] minimax
}
```

## §18 Encoder–decoder and relative attention bias
```jsonc
{
  "is_decoder": false,                  // [enum]  gpt_neox, t5, umt5
  "is_encoder_decoder": false,          // [enum]
  "add_cross_attention": false,         // [enum]  gpt2
  "relative_attention_num_buckets": 32, // [value] t5, umt5
  "relative_attention_max_distance": 128,// [value] t5, umt5
  "initializer_factor": 1.0             // [runtime] t5, umt5
}
```

## §19 Component composition — multimodal sub-configs
```jsonc
{
  "text_config":   { "…": "a complete member of Part I, in its own scope" },
  "vision_config": { "…": "…" },
  "audio_config":  { "…": "…" },
  "thinker_config": { "…": "wrapper spelling that hides the multimodal host" }
}
```
Each sub-config is completed independently under its own component source. Wrapper
spellings that hide the language model or the modality host resolve through the
shared wrapper vocabulary; nesting can repeat (`thinker_config.text_config`).

## §20 Runtime and training bookkeeping — never architectural
```jsonc
{
  "use_cache": true,          // [runtime] 34 families
  "initializer_range": 0.02,  // [runtime] 33 families
  "dtype": "bfloat16",        // [runtime]
  "pretraining_tp": 1,        // [runtime] bloom, deepseek_v3, llama
  "transformers_version": "5.x", // [runtime]
  "cache_implementation": null   // [runtime]
}
```

---

# Part II — Diffusion family (diffusers idiom)

Structure here is code-proven: stage kinds, autoencoder latent kind, and sampler
mechanism are **not** config fields. The config carries geometry, declared enums, and
component addresses. Block-type string lists are **addresses** that locate stage
classes, never structural facts.

## §21 Pipeline component index (`model_index.json`)
```jsonc
{
  "_class_name": "StableDiffusionXLPipeline", // [address]
  "_diffusers_version": "0.x",                // [runtime]
  "unet":         ["diffusers", "UNet2DConditionModel"],  // [address]
  "transformer":  ["diffusers", "SD3Transformer2DModel"], // [address]
  "vae":          ["diffusers", "AutoencoderKL"],         // [address]
  "scheduler":    ["diffusers", "EulerDiscreteScheduler"],// [address]
  "text_encoder": ["transformers", "CLIPTextModel"]       // [address]
}
```

## §22 UNet denoiser — geometry and stage composition
```jsonc
{
  "sample_size": 128,                           // [value]
  "in_channels": 4, "out_channels": 4,          // [value]
  "block_out_channels": [320, 640, 1280],       // [value]
  "layers_per_block": 2,                        // [value]
  "transformer_layers_per_block": 1,            // [value]
  "reverse_transformer_layers_per_block": null, // [value]
  "down_block_types": ["…"],                    // [address] stage classes
  "up_block_types": ["…"],                      // [address]
  "mid_block_type": "UNetMidBlock2DCrossAttn",  // [address]
  "attention_head_dim": 64,                     // [value]
  "num_attention_heads": null,                  // [value]
  "norm_num_groups": 32, "norm_eps": 1e-05,     // [value] GroupNorm
  "act_fn": "silu",                             // [enum]
  "resnet_time_scale_shift": "default",         // [enum]
  "resnet_out_scale_factor": 1.0,               // [scale]
  "conv_in_kernel": 3, "conv_out_kernel": 3,    // [value]
  "downsample_padding": 1,                      // [value]
  "mid_block_scale_factor": 1,                  // [scale]
  "upcast_attention": false,                    // [runtime]
  "use_linear_projection": false,               // [enum]
  "dropout": 0.0                                // [runtime]
}
```

## §23 UNet denoiser — conditioning and time embedding
```jsonc
{
  "cross_attention_dim": 2048,          // [value]
  "encoder_hid_dim": null,              // [value]
  "encoder_hid_dim_type": null,         // [enum]  conditioning kind (text_proj | image_proj | …)
  "addition_embed_type": null,          // [enum]  added-conditioning dialect
  "addition_embed_type_num_heads": 64,  // [value]
  "addition_time_embed_dim": null,      // [value]
  "projection_class_embeddings_input_dim": null, // [value]
  "class_embed_type": null,             // [enum]
  "num_class_embeds": null,             // [value]
  "class_embeddings_concat": false,     // [enum]
  "time_embedding_type": "positional",  // [enum]
  "time_embedding_dim": null,           // [value]
  "time_embedding_act_fn": null,        // [enum]
  "time_cond_proj_dim": null,           // [value]
  "timestep_post_act": null,            // [enum]
  "flip_sin_to_cos": true, "freq_shift": 0, // [value]
  "only_cross_attention": false,        // [enum]
  "mid_block_only_cross_attention": null,// [enum]
  "cross_attention_norm": null,         // [enum]
  "dual_cross_attention": false,        // [enum]
  "attention_type": "default"           // [enum]
}
```

## §24 DiT denoiser — geometry and patching
```jsonc
{
  "num_layers": 19,             // [value] 5 of 6 image-DiT classes
  "num_attention_heads": 24,    // [value] all 6
  "attention_head_dim": 64,     // [value] 5 of 6
  "in_channels": 16,            // [value] all 6
  "out_channels": null,         // [value] all 6
  "patch_size": 2,              // [value] all 6
  "sample_size": 128,           // [value] 5 of 6
  "norm_eps": 1e-06,            // [value]
  "norm_elementwise_affine": false, // [enum]
  "mlp_ratio": 2.5,             // [value] sana
  "qk_norm": "rms_norm",        // [enum]  sd3, sana — DECLARED q/k norm
  "dual_attention_layers": [],  // [schedule] sd3
  "axes_dims_rope": [16,56,56], // [value] flux — per-axis RoPE dims
  "axes_dim_rope": [16,56,56],  // [value] lumina2 — the sibling spelling
  "axes_lens": [300,512,512]    // [value] lumina2
}
```

## §25 DiT denoiser — conditioning and joint attention
```jsonc
{
  "joint_attention_dim": 4096,     // [value] sd3, flux, auraflow
  "pooled_projection_dim": 2048,   // [value] sd3, flux
  "caption_projection_dim": 1152,  // [value] sd3, auraflow
  "caption_channels": 4096,        // [value] pixart, sana
  "cross_attention_dim": 1152,     // [value] pixart, sana
  "cross_attention_head_dim": 64,  // [value] sana
  "guidance_embeds": false,        // [enum]  flux, sana
  "guidance_embeds_scale": 0.1,    // [scale] sana
  "attention_bias": true,          // [value] pixart, sana
  "pos_embed_max_size": 96,        // [value] sd3, auraflow
  "interpolation_scale": null,     // [value] pixart, sana
  "cap_feat_dim": 2304             // [value] lumina2
}
```

## §26 Autoencoder — latent space and stage composition
```jsonc
{
  "in_channels": 3, "out_channels": 3,       // [value]
  "latent_channels": 4,                      // [value] KL family
  "z_dim": 16,                               // [value] wan
  "block_out_channels": [128,256,512,512],   // [value]
  "down_block_types": ["…"], "up_block_types": ["…"], // [address]
  "layers_per_block": 2,                     // [value]
  "norm_num_groups": 32, "norm_eps": 1e-06,  // [value]
  "act_fn": "silu",                          // [enum]
  "scaling_factor": 0.18215,                 // [scale]
  "shift_factor": null,                      // [scale]
  "latents_mean": null, "latents_std": null, // [value]
  "force_upcast": true,                      // [runtime]
  "mid_block_add_attention": true,           // [enum]
  "use_quant_conv": true,                    // [enum]
  "use_post_quant_conv": true,               // [enum]
  "sample_size": 1024,                       // [value]

  // DC-AE dialect (sana)
  "encoder_block_types": ["…"], "decoder_block_types": ["…"], // [address]
  "encoder_qkv_multiscales": [], "decoder_qkv_multiscales": [], // [value]
  "downsample_block_type": "conv", "upsample_block_type": "interpolate", // [enum]

  // temporal/3-D dialect (cogvideox, wan)
  "temporal_compression_ratio": 4,   // [value]
  "temperal_downsample": [false],    // [schedule] (upstream spelling preserved)
  "scale_factor_spatial": 8, "scale_factor_temporal": 4, // [value]
  "sample_height": 480, "sample_width": 720 // [value]
}
```

## §27 Sampler — timestep schedule and update rule
```jsonc
{
  "num_train_timesteps": 1000,          // [value] all
  "prediction_type": "epsilon",         // [enum]  epsilon | v_prediction | sample
  "beta_schedule": "linear",            // [enum]
  "beta_start": 0.0001, "beta_end": 0.02, // [value]
  "trained_betas": null,                // [value]
  "timestep_spacing": "leading",        // [enum]
  "timestep_type": "discrete",          // [enum]
  "steps_offset": 0,                    // [value]
  "rescale_betas_zero_snr": false,      // [enum]
  "set_alpha_to_one": false,            // [enum]

  // sigma parameterizations
  "sigma_min": null, "sigma_max": null, // [value]
  "use_karras_sigmas": false, "use_exponential_sigmas": false,
  "use_beta_sigmas": false, "use_flow_sigmas": false, // [enum]
  "interpolation_type": "linear",       // [enum]
  "final_sigmas_type": "zero",          // [enum]
  "invert_sigmas": false,               // [enum]

  // multistep / predictor-corrector solvers
  "solver_order": 2,                    // [value] dpm++, unipc
  "algorithm_type": "dpmsolver++",      // [enum]  dpm++
  "solver_type": "midpoint",            // [enum]
  "solver_p": null,                     // [enum]  unipc
  "predict_x0": true,                   // [enum]  unipc
  "disable_corrector": [],              // [schedule] unipc
  "lower_order_final": true,            // [enum]
  "euler_at_final": false,              // [enum]
  "lambda_min_clipped": -inf,           // [value] dpm++
  "variance_type": null,                // [enum]

  // thresholding
  "thresholding": false,                // [enum]
  "dynamic_thresholding_ratio": 0.995,  // [value]
  "sample_max_value": 1.0,              // [value]
  "clip_sample": false, "clip_sample_range": 1.0, // [value]

  // flow-matching shift (sd3/flux lineage)
  "shift": 1.0, "flow_shift": null, "shift_terminal": null, // [value]
  "use_dynamic_shifting": false,        // [enum]
  "base_shift": 0.5, "max_shift": 1.15, // [value]
  "base_image_seq_len": 256, "max_image_seq_len": 4096, // [value]
  "time_shift_type": "exponential",     // [enum]
  "stochastic_sampling": false,         // [enum]

  // consistency (lcm)
  "original_inference_steps": 50,       // [value]
  "timestep_scaling": 10.0              // [scale]
}
```

---

## How `true_config()` uses this reference

1. Traverse the model's source to find every config field the code reads (the demand set).
2. Keep only fields that appear in this reference; surface any code-read field missing
   here as a reference-extension signal rather than emitting it silently.
3. Resolve each field's value: serialized config → class default → code-derived
   default → unresolved.
4. Assemble in the model's native spellings and nesting, recursing sub-configs.
5. Emit the completed config; carry provenance and the config-versus-true diff in a
   separate companion, never inside the config.

The result is a config that could sit in the repository as the one the checkpoint
should have shipped: complete over what the code reads, minimal beyond it.

## Known limits of this reference

- 37 of 656 registered families harvested. The remainder are unaudited; the namelist
  was chosen for mechanism coverage, not popularity.
- Class defaults and real checkpoints disagree in both directions (correction 3
  above). Both sources are required, and a third — the code's actual reads — is what
  `true_config()` adds.
- Diffusers schemas are `__init__` signatures, which are the config contract but are
  version-bound; the installed versions are `transformers 5.12.1` / `diffusers 0.38.0`.
