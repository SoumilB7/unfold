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

## Standing accessed-but-unconsumed occurrences: 227

Format: `exact.dotted.path (witness count)`, with `(as spelling)` when the
supplying alias differs. The row key is the FULL occurrence, so two paths
sharing a canonical leaf are two rows.

Paths are relative to each owner's DOCUMENT (named per section below):
that keeps the key host-independent, so a claim binding matches the same
mechanism whether a model is parsed standalone or embedded in a pipeline.
Prefix a row with its document to address the value in the witness file.

### `root` — 52 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

Paths relative to: the top-level document

`apply_residual_connection_post_layernorm` (1w), `attention_bias` (7w), `attn_logit_softcapping` (1w), `final_logit_softcapping` (1w), `first_k_dense_replace` (2w), `image_token_id` (1w), `is_encoder_decoder` (10w), `kv_lora_rank` (1w), `layer_norm_eps` (1w), `layer_norm_epsilon` (1w), `layer_types` (3w), `max_window_layers` (1w), `mlp_bias` (1w), `moe_layer_freq` (1w), `n_group` (2w), `norm_topk_prob` (2w), `num_nextn_predict_layers` (2w), `partial_rotary_factor` (2w), `q_lora_rank` (1w), `qk_layernorm` (1w), `qk_nope_head_dim` (1w), `qk_rope_head_dim` (1w), `query_pre_attn_scalar` (1w), `rms_norm_eps` (7w), `rope_parameters` (8w), `rope_parameters.factor` (2w), `rope_parameters.original_max_position_embeddings` (2w), `rope_parameters.partial_rotary_factor` (2w), `rope_parameters.rope_theta` (8w), `rope_parameters.rope_type` (8w), `rope_parameters.type` (1w), `routed_scaling_factor` (2w), `scoring_func` (1w), `swiglu_limit` (1w), `text_config` (1w), `text_config.rope_parameters` (1w), `text_config.rope_parameters.mrope_section` (1w), `text_config.rope_parameters.rope_theta` (1w), `text_config.rope_parameters.rope_type` (1w), `text_config.rope_parameters.type` (1w), `topk_group` (2w), `topk_method` (1w), `use_bidirectional_attention` (1w), `use_parallel_residual` (1w), `use_qk_norm` (1w), `use_qkv_bias` (1w), `use_sliding_window` (1w), `v_head_dim` (1w), `video_token_id` (1w), `vision_config` (1w), `vision_end_token_id` (1w), `vision_start_token_id` (1w)

### `root.denoiser` — 100 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

Paths relative to: the top-level document

`act_fn` (1w), `activation_fn` (4w), `added_kv_proj_dim` (1w), `addition_embed_type` (1w), `addition_embed_type_num_heads` (1w), `addition_time_embed_dim` (1w), `attention_bias` (4w), `attention_head_dim` (1w), `attention_out_bias` (1w), `attention_type` (1w), `axes_dim` (1w), `axes_dim_rope` (1w), `axes_dims_rope` (3w), `axes_lens` (1w), `block_out_channels` (1w), `bottleneck_size` (1w), `boundary_ratio` (1w), `cap_feat_dim` (1w), `caption_channels` (3w), `caption_projection_dim` (2w), `center_input_sample` (1w), `class_embeddings_concat` (1w), `context_in_dim` (1w), `conv_in_kernel` (1w), `conv_out_kernel` (1w), `cross_attention_dim` (4w), `cross_attention_head_dim` (1w), `cross_attn_norm` (1w), `default_sample_size` (1w), `double_self_attention` (1w), `down_block_types` (1w), `downsample_padding` (1w), `dual_cross_attention` (1w), `encoder_hid_dim` (1w), `encoder_hid_dim_type` (1w), `eps` (2w), `ffn_dim_multiplier` (1w), `flip_sin_to_cos` (2w), `freq_dim` (1w), `freq_shift` (2w), `guidance_embeds` (3w), `image_dim` (1w), `in_channels` (1w), `interpolation_scale` (1w), `joint_attention_dim` (5w), `layers_per_block` (1w), `max_text_seq_length` (1w), `mid_block_scale_factor` (1w), `mid_block_type` (1w), `mlp_ratio` (4w), `multiple_of` (1w), `norm_elementwise_affine` (4w), `norm_type` (1w), `num_attention_heads` (1w), `num_cross_attention_heads` (1w), `num_embeds_ada_norm` (1w), `num_refiner_layers` (2w), `only_cross_attention` (2w), `out_channels` (13w), `pooled_projection_dim` (5w), `pos_embed_max_size` (2w), `pos_embed_seq_len` (1w), `projection_class_embeddings_input_dim` (1w), `qk_norm` (5w), `resnet_out_scale_factor` (1w), `resnet_skip_time_act` (1w), `resnet_time_scale_shift` (1w), `resolution_embeds` (1w), `rope_axes_dim` (1w), `rope_max_seq_len` (1w), `rope_theta` (2w), `sample_frames` (1w), `sample_height` (1w), `sample_size` (6w), `sample_width` (1w), `spatial_interpolation_scale` (1w), `temporal_compression_ratio` (1w), `temporal_interpolation_scale` (1w), `text_dim` (1w), `text_embed_dim` (3w), `text_encoder` (15w), `text_encoder_2` (4w), `text_encoder_3` (1w), `theta` (1w), `time_cond_proj_dim` (1w), `time_embed_dim` (2w), `time_embedding_act_fn` (1w), `time_embedding_type` (1w), `time_factor` (1w), `time_max_period` (1w), `timestep_activation_fn` (1w), `timestep_guidance_channels` (1w), `timestep_post_act` (1w), `transformer_2` (1w), `transformer_layers_per_block` (1w), `up_block_types` (1w), `upcast_attention` (2w), `use_additional_conditions` (1w), `use_linear_projection` (2w), `use_rotary_positional_embeddings` (1w)

### `root.scheduler` — 1 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

Paths relative to: the top-level document

`scheduler` (15w)

### `root.text_encoder` — 27 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

Paths relative to: `_text_encoder_configs.text_encoder`

`attention_bias` (4w), `attn_logit_softcapping` (2w), `feed_forward_proj` (6w), `final_logit_softcapping` (2w), `image_token_id` (1w), `image_token_index` (1w), `is_decoder` (3w), `is_encoder_decoder` (12w), `is_gated_act` (6w), `layer_norm_eps` (2w), `layer_norm_epsilon` (6w), `layer_types` (2w), `mlp_bias` (1w), `query_pre_attn_scalar` (2w), `rms_norm_eps` (4w), `rope_parameters` (3w), `rope_parameters.rope_theta` (3w), `rope_parameters.rope_type` (3w), `text_config` (2w), `text_config.rope_parameters` (1w), `text_config.rope_parameters.rope_theta` (1w), `text_config.rope_parameters.rope_type` (1w), `use_bidirectional_attention` (2w), `video_token_id` (1w), `vision_config` (2w), `vision_end_token_id` (1w), `vision_start_token_id` (1w)

### `root.text_encoder.vision` — 25 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

Paths relative to: `_text_encoder_configs.text_encoder`

`image_token_id` (1w), `image_token_index` (1w), `projector_hidden_act` (1w), `text_config.num_hidden_layers` (2w), `video_token_id` (1w), `vision_config` (2w), `vision_config.depth` (1w), `vision_config.fullatt_block_indexes` (1w), `vision_config.hidden_act` (2w), `vision_config.image_size` (1w), `vision_config.in_channels` (1w), `vision_config.in_chans` (1w), `vision_config.intermediate_size` (2w), `vision_config.num_attention_heads` (1w), `vision_config.num_channels` (1w), `vision_config.num_heads` (1w), `vision_config.num_hidden_layers` (1w), `vision_config.patch_size` (2w), `vision_config.rope_parameters` (1w), `vision_config.spatial_merge_size` (1w), `vision_config.temporal_patch_size` (1w), `vision_config.tokens_per_second` (1w), `vision_end_token_id` (1w), `vision_feature_layer` (1w), `vision_start_token_id` (1w)

### `root.text_encoder_2` — 2 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

Paths relative to: `_text_encoder_configs.text_encoder_2`

`is_encoder_decoder` (3w), `layer_norm_eps` (3w)

### `root.text_encoder_3` — 5 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

Paths relative to: `_text_encoder_configs.text_encoder_3`

`feed_forward_proj` (1w), `is_decoder` (1w), `is_encoder_decoder` (1w), `is_gated_act` (1w), `layer_norm_epsilon` (1w)

### `root.vision` — 15 rows — **UNCLASSIFIED** (U2.2 assigns per occurrence/mechanism — an owner spans several units)

Paths relative to: the top-level document

`image_token_id` (1w), `text_config.num_hidden_layers` (1w), `video_token_id` (1w), `vision_config` (1w), `vision_config.depth` (1w), `vision_config.hidden_act` (1w), `vision_config.in_channels` (1w), `vision_config.in_chans` (1w), `vision_config.mlp_ratio` (1w), `vision_config.num_heads` (1w), `vision_config.patch_size` (1w), `vision_config.spatial_merge_size` (1w), `vision_config.temporal_patch_size` (1w), `vision_end_token_id` (1w), `vision_start_token_id` (1w)

## Reads whose LOCATION is unknown: 15

NOT classifiable, and NOT part of the census above: the read is real
and the value is real, but the reader touched a nested object without
naming which, so the ledger recorded an honest bare leaf. Asking for a
disposition here would be asking to classify a location nobody
established.

A **producer** backlog (U2.2b): each shrinks where its READER names the
object it read (`wrapper_path` / `config_container(obj=)`) — never by a
census filter, and never by deciding what an unlocatable row means.

- `root` — 5: `layer_types` (1w), `max_window_layers` (1w), `rms_norm_eps` (1w), `rope_parameters` (1w), `use_sliding_window` (1w)
- `root.text_encoder` — 8: `layer_norm_eps` (2w), `layer_norm_epsilon` (6w), `layer_types` (1w), `max_window_layers` (1w), `rms_norm_eps` (6w), `rope_parameters` (2w), `text_config` (2w), `use_sliding_window` (1w)
- `root.text_encoder_2` — 1: `layer_norm_eps` (3w)
- `root.text_encoder_3` — 1: `layer_norm_epsilon` (1w)

## Reads whose ORIGIN is unknown: 80

BLOCKING debt, and NOT part of the census above. The document these
were read from was never prepared, so nobody can say whether the
checkpoint declared them or a config class supplied them. Unestablished
is not a synonym for declared — letting it default into the checkpoint
census is what made the class's words look like the file's.

These are a few lost DOCUMENT BOUNDARIES multiplied across many reads,
not one problem per row: they collapse when preparation is centralized
(one prepared document per boundary), not by classifying them.

- `root` — 5: `layer_types` (1w), `max_window_layers` (1w), `rms_norm_eps` (1w), `rope_parameters` (1w), `use_sliding_window` (1w)
- `root.scheduler` — 6: `_scheduler_config.beta_schedule` (5w), `_scheduler_config.num_train_timesteps` (15w), `_scheduler_config.prediction_type` (5w), `_scheduler_config.shift` (10w), `_scheduler_config.timestep_spacing` (5w), `_scheduler_config.use_dynamic_shifting` (7w)
- `root.text_encoder` — 15: `hidden_act` (1w), `hidden_size` (1w), `intermediate_size` (1w), `layer_norm_eps` (2w), `layer_norm_epsilon` (6w), `layer_types` (1w), `max_position_embeddings` (1w), `max_window_layers` (1w), `num_attention_heads` (1w), `num_hidden_layers` (1w), `rms_norm_eps` (6w), `rope_parameters` (2w), `text_config` (2w), `use_sliding_window` (1w) …
- `root.text_encoder_2` — 9: `d_ff` (1w), `d_model` (1w), `dense_act_fn` (1w), `feed_forward_proj` (1w), `is_gated_act` (1w), `layer_norm_eps` (3w), `num_heads` (1w), `num_layers` (1w), `vocab_size` (1w)
- `root.text_encoder_3` — 1: `layer_norm_epsilon` (1w)
- `root.vae` — 44: `_vae_config.add_attention_block` (1w), `_vae_config.attn_scales` (2w), `_vae_config.base_dim` (2w), `_vae_config.batch_norm_eps` (1w), `_vae_config.batch_norm_momentum` (1w), `_vae_config.block_out_channels` (10w), `_vae_config.decoder_act_fns` (1w), `_vae_config.decoder_block_out_channels` (2w), `_vae_config.decoder_block_types` (1w), `_vae_config.decoder_causal` (1w), `_vae_config.decoder_layers_per_block` (1w), `_vae_config.decoder_norm_types` (1w), `_vae_config.decoder_qkv_multiscales` (1w), `_vae_config.dim_mult` (2w) …

## Fields the CHECKPOINT never declared: 17

The installed config class supplied these (located by `model_type` —
identity-as-ADDRESS, which is lawful). They are excluded from the
checkpoint census because they are not the checkpoint's words, and
listed here because they are real and often STRUCTURAL: a
class-supplied `layer_types` IS a mask schedule. The open question for
each is not "what does this declaration mean" but "may the class decide
this, and does the fact it authors say so".

- `root.denoiser` — 3: `_repo_id` [loader_metadata] (14w), `_text_encoder_configs` [loader_metadata] (15w), `_vae_config` [loader_metadata] (14w)
- `root.scheduler` — 1: `_scheduler_config` [loader_metadata] (15w)
- `root.text_encoder` — 12: `_name_or_path` [class_default] (2w), `is_decoder` [class_default] (3w), `is_encoder_decoder` [class_default] (2w), `rope_parameters` [class_default] (1w), `rope_parameters.mrope_section` [class_default] (1w), `rope_parameters.rope_theta` [class_default] (1w), `rope_parameters.rope_type` [class_default] (1w), `text_config.rope_parameters` [class_default] (1w), `text_config.rope_parameters.mrope_section` [class_default] (1w), `text_config.rope_parameters.rope_theta` [class_default] (1w), `text_config.rope_parameters.rope_type` [class_default] (1w), `text_config.rope_parameters.type` [class_default] (1w)
- `root.vae` — 1: `_vae_config` [loader_metadata] (14w)

