# Net-1 migration-debt census (authoritative, generated)

Regenerate with `python3 scripts/census.py`; `--check` fails when this
file is stale. Never hand-edit — the generator is the producer.

Source: the parser's own owner-scoped OCCURRENCE-EXACT view
(`config_access.accessed_unconsumed_exact`: component + exact dotted path +
actual spelling) across the committed corpus.

## What this list is

A **discovery list**, not a list of receipts owed. No row carries a unit
assignment yet: an owner is not a unit (`root` alone spans U6 attention,
U7 FFN/norm and U8 position/mask; vision is mainly U9; denoiser U10/U11;
vae U12; scheduler U13), so assignment is per OCCURRENCE/mechanism and
belongs to U2.2.

Every row must receive
exactly ONE disposition (U2.2): structural+already projected · structural
but not projected · geometry-only · display-only · address/source-selection
only · non-architectural metadata · unused/phantom (delete the read or fix
ownership) · ambiguous/unsupported (preserve unknown). Only
mechanism-driving rows need new interpretation code.

## The law

- Migration is claimed at exact **(owner, mechanism)** scope — never an adapter or file.
- A claim is valid only when every present read in scope has an exact path and owner,
  and each read is consumed, scoped-ignored, or precisely classified.
- Ambiguities remain blocking regardless of claims.
- Net 1 blocks each claimed scope immediately (`config_migration_claims`).
- Net 2 (`config_consumed_unreceipted`) joins exact occurrence -> exact fact target ->
  exact render RECEIPT, and validates the receipt's value/status fingerprint against
  the fingerprint recorded AT the consumption plus the registered surface/kind policy.
  Coverage is owner/mechanism-SCOPED (`projection_coverage.receipted_scopes`), never a
  global flag: inside a receipted scope a missing or invalid receipt BLOCKS
  unconditionally; every other scope stays this advisory census.
- Rows below are VISIBLE debt: the count must shrink only as rows are genuinely
  resolved. Mass registration is debt-laundering and is rejected.

## Live claims (source-to-target bound; observed and matched corpus-wide)

| scope | claimed by | bindings (path -> target) | observed | target-matched |
|---|---|---|---|---|
| root.video/projector_out_width | COR-4 | vision_config.hidden_size -> root.video.projector_out_features | 2 | 1 |
| root.vision/encoder_width | COR-5 | vision_config.embed_dim -> root.vision.hidden_size; vision_config.vision_hidden_size -> root.vision.hidden_size; vision_config.width -> root.vision.hidden_size; vision_config.hidden_size -> root.vision.hidden_size | 6 | 2 |
| root.vision/projector_out_width | COR-4 | vision_config.hidden_size -> root.vision.projector_out_features | 2 | 1 |

## Owners with NO consumption at all (audit_incomplete)

- `root.denoiser` — 1 witnesses (e.g. ['stable-diffusion-xl-base-1-0']); **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)
- `root.scheduler` — 15 witnesses (e.g. ['auraflow-v0-3', 'cogvideox-5b', 'flux-2-dev', 'fluxtransformer2dmodel']); **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

## Standing accessed-but-unconsumed occurrences: 281

Format: `exact.dotted.path (witness count)`, with `(as spelling)` when the
supplying alias differs. The row key is the FULL occurrence, so two paths
sharing a canonical leaf are two rows.

### `root` — 52 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

`apply_residual_connection_post_layernorm` (1w), `attention_bias` (7w), `attn_logit_softcapping` (1w), `final_logit_softcapping` (1w), `first_k_dense_replace` (2w), `image_token_id` (1w), `is_encoder_decoder` (10w), `kv_lora_rank` (1w), `layer_norm_eps` (1w), `layer_norm_epsilon` (1w), `layer_types` (4w), `max_window_layers` (2w), `mlp_bias` (1w), `moe_layer_freq` (1w), `n_group` (2w), `norm_topk_prob` (2w), `num_hidden_layers` (1w), `num_nextn_predict_layers` (2w), `partial_rotary_factor` (2w), `q_lora_rank` (1w), `qk_layernorm` (1w), `qk_nope_head_dim` (1w), `qk_rope_head_dim` (1w), `query_pre_attn_scalar` (1w), `rms_norm_eps` (8w), `rope_parameters` (9w), `rope_parameters.factor` (2w), `rope_parameters.original_max_position_embeddings` (2w), `rope_parameters.rope_theta` (8w), `rope_parameters.rope_type` (8w), `rope_parameters.type` (1w), `routed_scaling_factor` (2w), `scoring_func` (1w), `swiglu_limit` (1w), `text_config` (1w), `text_config.rope_parameters` (1w), `text_config.rope_parameters.mrope_section` (1w), `text_config.rope_parameters.rope_theta` (1w), `text_config.rope_parameters.rope_type` (1w), `text_config.rope_parameters.type` (1w), `topk_group` (2w), `topk_method` (1w), `use_bidirectional_attention` (1w), `use_parallel_residual` (1w), `use_qk_norm` (1w), `use_qkv_bias` (1w), `use_sliding_window` (2w), `v_head_dim` (1w), `video_token_id` (1w), `vision_config` (1w), `vision_end_token_id` (1w), `vision_start_token_id` (1w)

### `root.denoiser` — 101 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

`_repo_id` (14w), `act_fn` (1w), `activation_fn` (4w), `added_kv_proj_dim` (1w), `addition_embed_type` (1w), `addition_embed_type_num_heads` (1w), `addition_time_embed_dim` (1w), `attention_bias` (4w), `attention_head_dim` (1w), `attention_out_bias` (1w), `attention_type` (1w), `axes_dim` (1w), `axes_dim_rope` (1w), `axes_dims_rope` (3w), `axes_lens` (1w), `block_out_channels` (1w), `bottleneck_size` (1w), `boundary_ratio` (1w), `cap_feat_dim` (1w), `caption_channels` (3w), `caption_projection_dim` (2w), `center_input_sample` (1w), `class_embeddings_concat` (1w), `context_in_dim` (1w), `conv_in_kernel` (1w), `conv_out_kernel` (1w), `cross_attention_dim` (4w), `cross_attention_head_dim` (1w), `cross_attn_norm` (1w), `default_sample_size` (1w), `double_self_attention` (1w), `down_block_types` (1w), `downsample_padding` (1w), `dual_cross_attention` (1w), `encoder_hid_dim` (1w), `encoder_hid_dim_type` (1w), `eps` (2w), `ffn_dim_multiplier` (1w), `flip_sin_to_cos` (2w), `freq_dim` (1w), `freq_shift` (2w), `guidance_embeds` (3w), `image_dim` (1w), `in_channels` (1w), `interpolation_scale` (1w), `joint_attention_dim` (5w), `layers_per_block` (1w), `max_text_seq_length` (1w), `mid_block_scale_factor` (1w), `mid_block_type` (1w), `mlp_ratio` (4w), `multiple_of` (1w), `norm_elementwise_affine` (4w), `norm_type` (1w), `num_attention_heads` (1w), `num_cross_attention_heads` (1w), `num_embeds_ada_norm` (1w), `num_refiner_layers` (2w), `only_cross_attention` (2w), `out_channels` (13w), `pooled_projection_dim` (5w), `pos_embed_max_size` (2w), `pos_embed_seq_len` (1w), `projection_class_embeddings_input_dim` (1w), `qk_norm` (5w), `resnet_out_scale_factor` (1w), `resnet_skip_time_act` (1w), `resnet_time_scale_shift` (1w), `resolution_embeds` (1w), `rope_axes_dim` (1w), `rope_max_seq_len` (1w), `rope_theta` (2w), `sample_frames` (1w), `sample_height` (1w), `sample_size` (6w), `sample_width` (1w), `spatial_interpolation_scale` (1w), `temporal_compression_ratio` (1w), `temporal_interpolation_scale` (1w), `text_dim` (1w), `text_embed_dim` (3w), `text_encoder` (15w), `text_encoder_2` (4w), `text_encoder_3` (1w), `theta` (1w), `time_cond_proj_dim` (1w), `time_embed_dim` (2w), `time_embedding_act_fn` (1w), `time_embedding_type` (1w), `time_factor` (1w), `time_max_period` (1w), `timestep_activation_fn` (1w), `timestep_guidance_channels` (1w), `timestep_post_act` (1w), `transformer_2` (1w), `transformer_layers_per_block` (1w), `up_block_types` (1w), `upcast_attention` (2w), `use_additional_conditions` (1w), `use_linear_projection` (2w), `use_rotary_positional_embeddings` (1w)

### `root.scheduler` — 7 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

`_scheduler_config.beta_schedule` (5w), `_scheduler_config.num_train_timesteps` (15w), `_scheduler_config.prediction_type` (5w), `_scheduler_config.shift` (10w), `_scheduler_config.timestep_spacing` (5w), `_scheduler_config.use_dynamic_shifting` (7w), `scheduler` (15w)

### `root.text_encoder` — 34 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

`_text_encoder_configs.text_encoder.attn_logit_softcapping` (2w), `_text_encoder_configs.text_encoder.feed_forward_proj` (6w), `_text_encoder_configs.text_encoder.final_logit_softcapping` (2w), `_text_encoder_configs.text_encoder.image_token_id` (1w), `_text_encoder_configs.text_encoder.image_token_index` (1w), `_text_encoder_configs.text_encoder.is_decoder` (6w), `_text_encoder_configs.text_encoder.is_encoder_decoder` (14w), `_text_encoder_configs.text_encoder.is_gated_act` (6w), `_text_encoder_configs.text_encoder.layer_norm_eps` (2w), `_text_encoder_configs.text_encoder.layer_norm_epsilon` (6w), `_text_encoder_configs.text_encoder.layer_types` (3w), `_text_encoder_configs.text_encoder.max_window_layers` (1w), `_text_encoder_configs.text_encoder.num_hidden_layers` (9w), `_text_encoder_configs.text_encoder.query_pre_attn_scalar` (2w), `_text_encoder_configs.text_encoder.rms_norm_eps` (6w), `_text_encoder_configs.text_encoder.rope_parameters` (6w), `_text_encoder_configs.text_encoder.text_config` (2w), `_text_encoder_configs.text_encoder.use_bidirectional_attention` (2w), `_text_encoder_configs.text_encoder.use_sliding_window` (1w), `_text_encoder_configs.text_encoder.video_token_id` (1w), `_text_encoder_configs.text_encoder.vision_config` (2w), `_text_encoder_configs.text_encoder.vision_end_token_id` (1w), `_text_encoder_configs.text_encoder.vision_start_token_id` (1w), `attention_bias` (4w), `mlp_bias` (1w), `rope_parameters` (4w), `rope_parameters.mrope_section` (1w), `rope_parameters.rope_theta` (4w), `rope_parameters.rope_type` (4w), `text_config.rope_parameters` (2w), `text_config.rope_parameters.mrope_section` (1w), `text_config.rope_parameters.rope_theta` (2w), `text_config.rope_parameters.rope_type` (2w), `text_config.rope_parameters.type` (1w)

### `root.text_encoder.vision` — 24 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

`vision_config.depth` (1w), `vision_config.fullatt_block_indexes` (1w), `vision_config.hidden_act` (2w), `vision_config.image_size` (1w), `vision_config.image_token_id` (1w), `vision_config.image_token_index` (1w), `vision_config.in_channels` (1w), `vision_config.in_chans` (1w), `vision_config.intermediate_size` (2w), `vision_config.num_attention_heads` (1w), `vision_config.num_channels` (1w), `vision_config.num_heads` (1w), `vision_config.num_hidden_layers` (2w), `vision_config.patch_size` (2w), `vision_config.projector_hidden_act` (1w), `vision_config.rope_parameters` (1w), `vision_config.spatial_merge_size` (1w), `vision_config.temporal_patch_size` (1w), `vision_config.tokens_per_second` (1w), `vision_config.video_token_id` (1w), `vision_config.vision_config` (2w), `vision_config.vision_end_token_id` (1w), `vision_config.vision_feature_layer` (1w), `vision_config.vision_start_token_id` (1w)

### `root.text_encoder_2` — 5 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

`_text_encoder_configs.text_encoder_2.feed_forward_proj` (1w), `_text_encoder_configs.text_encoder_2.is_encoder_decoder` (3w), `_text_encoder_configs.text_encoder_2.is_gated_act` (1w), `_text_encoder_configs.text_encoder_2.layer_norm_eps` (3w), `_text_encoder_configs.text_encoder_2.num_hidden_layers` (3w)

### `root.text_encoder_3` — 5 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

`_text_encoder_configs.text_encoder_3.feed_forward_proj` (1w), `_text_encoder_configs.text_encoder_3.is_decoder` (1w), `_text_encoder_configs.text_encoder_3.is_encoder_decoder` (1w), `_text_encoder_configs.text_encoder_3.is_gated_act` (1w), `_text_encoder_configs.text_encoder_3.layer_norm_epsilon` (1w)

### `root.vae` — 38 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

`_vae_config.add_attention_block` (1w), `_vae_config.attn_scales` (2w), `_vae_config.base_dim` (2w), `_vae_config.batch_norm_eps` (1w), `_vae_config.batch_norm_momentum` (1w), `_vae_config.decoder_act_fns` (1w), `_vae_config.decoder_block_types` (1w), `_vae_config.decoder_causal` (1w), `_vae_config.decoder_layers_per_block` (1w), `_vae_config.decoder_norm_types` (1w), `_vae_config.decoder_qkv_multiscales` (1w), `_vae_config.dim_mult` (2w), `_vae_config.down_block_types` (8w), `_vae_config.downsample_block_type` (1w), `_vae_config.encoder_block_out_channels` (2w), `_vae_config.encoder_block_types` (1w), `_vae_config.encoder_causal` (1w), `_vae_config.encoder_layers_per_block` (1w), `_vae_config.encoder_qkv_multiscales` (1w), `_vae_config.invert_scale_latents` (1w), `_vae_config.latents_mean` (7w), `_vae_config.latents_std` (7w), `_vae_config.mid_block_add_attention` (4w), `_vae_config.norm_num_groups` (8w), `_vae_config.out_channels` (11w), `_vae_config.resnet_norm_eps` (1w), `_vae_config.sample_size` (6w), `_vae_config.scaling_factor` (11w), `_vae_config.shift_factor` (4w), `_vae_config.spatial_compression_ratio` (1w), `_vae_config.spatial_expansions` (1w), `_vae_config.spatio_temporal_scaling` (1w), `_vae_config.temperal_downsample` (2w), `_vae_config.temporal_expansions` (1w), `_vae_config.up_block_types` (8w), `_vae_config.upsample_block_type` (1w), `_vae_config.use_post_quant_conv` (5w), `_vae_config.use_quant_conv` (5w)

### `root.vision` — 15 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

`vision_config.depth` (1w), `vision_config.hidden_act` (1w), `vision_config.image_token_id` (1w), `vision_config.in_channels` (1w), `vision_config.in_chans` (1w), `vision_config.mlp_ratio` (1w), `vision_config.num_heads` (1w), `vision_config.num_hidden_layers` (1w), `vision_config.patch_size` (1w), `vision_config.spatial_merge_size` (1w), `vision_config.temporal_patch_size` (1w), `vision_config.video_token_id` (1w), `vision_config.vision_config` (1w), `vision_config.vision_end_token_id` (1w), `vision_config.vision_start_token_id` (1w)

