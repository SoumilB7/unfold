# COR-5 Net-1 migration-debt census (authoritative)

Generated 2026-07-14 on the COR-5 tree (parent `08d2bfe`) from the 25-witness
corpus — the parser's own owner-scoped `config_access.accessed_unconsumed`
view, exactly what turns blocking for a scope the moment a `MigrationClaim`
covers it.

## The law (COR-5 §10, as refined)

- Migration is claimed at exact **(owner, mechanism)** scope — never an adapter or file.
- A claim is valid only when every present read in scope has an exact path and owner,
  and each read is consumed, scoped-ignored, or precisely classified.
- Ambiguities remain blocking regardless of claims.
- Poison tests prove an empty declaration cannot pass vacuously
  (`tests/test_projection_audit.py::test_cor5_*`).
- Net 1 blocks each claimed scope immediately (`config_migration_claims`).
- Unclaimed rows below remain VISIBLE advisory debt — this file is that debt, exact
  and assigned; it must SHRINK as vertical units consume their reads and claim.
- Net 2 (`config_consumed_unprojected`) independently verifies projection and blocks
  exactly where a parse declares `projection_receipts_available=True` (U2 lands receipts).

## Live claims (all observed and violation-free corpus-wide)

| scope | claimed by | paths | observed events (corpus) |
|---|---|---|---|
| root.video/projector_out_width | COR-4 | vision_config.hidden_size | 2 |
| root.vision/encoder_width | COR-5 | vision_config.embed_dim, vision_config.vision_hidden_size, vision_config.width, vision_config.hidden_size | 18 |
| root.vision/projector_out_width | COR-4 | vision_config.hidden_size | 14 |

## Owners with NO consumption at all (audit_incomplete)

- `root.denoiser` — 1 witnesses (e.g. ['stable-diffusion-xl-base-1-0']); assigned **H7** (diffusion denoiser vertical (DiT/UNet fact families))
- `root.scheduler` — 15 witnesses (e.g. ['auraflow-v0-3', 'cogvideox-5b', 'flux-2-dev', 'fluxtransformer2dmodel']); assigned **H7** (sampling/scheduler block)

## Standing accessed-but-unconsumed rows: 250

Format: `field (witness count)`. Row key is (owner, canonical); the exact dotted
paths per row are in the event ledger of any carrying witness.

### `root` — 47 rows — assigned **H8** (transformer tail — branch-driving flags, rope internals, MoE routing, sliding-window schedules)

`apply_residual_connection_post_layernorm` (1w), `attention_bias` (8w), `attn_logit_softcapping` (1w), `factor` (2w), `final_logit_softcapping` (1w), `first_k_dense_replace` (2w), `image_token_id` (1w), `is_encoder_decoder` (10w), `kv_lora_rank` (1w), `layer_norm_eps` (1w), `layer_norm_epsilon` (1w), `layer_types` (4w), `max_window_layers` (2w), `mlp_bias` (1w), `moe_layer_freq` (1w), `mrope_section` (1w), `n_group` (2w), `norm_topk_prob` (2w), `num_nextn_predict_layers` (2w), `original_max_position_embeddings` (2w), `partial_rotary_factor` (2w), `q_lora_rank` (1w), `qk_layernorm` (1w), `qk_nope_head_dim` (1w), `qk_rope_head_dim` (1w), `query_pre_attn_scalar` (1w), `rms_norm_eps` (8w), `rope_parameters` (9w), `rope_scaling` (9w), `rope_theta` (9w), `rope_type` (9w), `routed_scaling_factor` (2w), `scoring_func` (1w), `swiglu_limit` (1w), `text_config` (1w), `topk_group` (2w), `topk_method` (1w), `type` (2w), `use_bidirectional_attention` (1w), `use_parallel_residual` (1w), `use_qk_norm` (1w), `use_sliding_window` (2w), `v_head_dim` (1w), `video_token_id` (1w), `vision_config` (1w), `vision_end_token_id` (1w), `vision_start_token_id` (1w)

### `root.denoiser` — 95 rows — assigned **H7** (diffusion denoiser vertical (DiT/UNet fact families))

`_repo_id` (14w), `act_fn` (1w), `added_kv_proj_dim` (1w), `addition_embed_type` (1w), `addition_embed_type_num_heads` (1w), `addition_time_embed_dim` (1w), `attention_bias` (4w), `attention_head_dim` (1w), `attention_out_bias` (1w), `attention_type` (1w), `axes_dims_rope` (6w), `axes_lens` (1w), `block_out_channels` (1w), `bottleneck_size` (1w), `boundary_ratio` (1w), `caption_input_dim` (3w), `caption_projection_dim` (2w), `center_input_sample` (1w), `class_embeddings_concat` (1w), `conv_in_kernel` (1w), `conv_out_kernel` (1w), `cross_attention_dim` (5w), `cross_attention_head_dim` (1w), `cross_attn_norm` (1w), `default_sample_size` (1w), `double_self_attention` (1w), `down_block_types` (1w), `downsample_padding` (1w), `dual_cross_attention` (1w), `encoder_hid_dim` (1w), `encoder_hid_dim_type` (1w), `eps` (2w), `ffn_dim_multiplier` (1w), `flip_sin_to_cos` (2w), `freq_dim` (1w), `freq_shift` (2w), `guidance_embeds` (3w), `hidden_act` (4w), `image_dim` (1w), `in_channels` (1w), `joint_attention_dim` (5w), `kv_join_dim` (1w), `layers_per_block` (1w), `max_text_seq_length` (1w), `mid_block_scale_factor` (1w), `mid_block_type` (1w), `mlp_ratio` (4w), `multiple_of` (1w), `norm_elementwise_affine` (4w), `norm_type` (1w), `num_attention_heads` (1w), `num_cross_attention_heads` (1w), `num_embeds_ada_norm` (1w), `num_refiner_layers` (2w), `only_cross_attention` (2w), `out_channels` (13w), `pooled_projection_dim` (5w), `pos_embed_max_size` (3w), `pos_embed_seq_len` (1w), `projection_class_embeddings_input_dim` (1w), `qk_norm` (5w), `resnet_out_scale_factor` (1w), `resnet_skip_time_act` (1w), `resnet_time_scale_shift` (1w), `resolution_embeds` (1w), `rope_max_seq_len` (1w), `rope_theta` (2w), `sample_frames` (1w), `sample_height` (1w), `sample_size` (6w), `sample_width` (1w), `spatial_interpolation_scale` (1w), `temporal_compression_ratio` (1w), `temporal_interpolation_scale` (1w), `text_embed_dim` (4w), `text_encoder` (15w), `text_encoder_2` (4w), `text_encoder_3` (1w), `theta` (1w), `time_cond_proj_dim` (1w), `time_embed_dim` (2w), `time_embedding_act_fn` (1w), `time_embedding_type` (1w), `time_factor` (1w), `time_max_period` (1w), `timestep_activation_fn` (1w), `timestep_guidance_channels` (1w), `timestep_post_act` (1w), `transformer_2` (1w), `transformer_layers_per_block` (1w), `up_block_types` (1w), `upcast_attention` (2w), `use_additional_conditions` (1w), `use_linear_projection` (2w), `use_rotary_positional_embeddings` (1w)

### `root.scheduler` — 7 rows — assigned **H7** (sampling/scheduler block)

`beta_schedule` (5w), `num_train_timesteps` (15w), `prediction_type` (5w), `scheduler` (15w), `shift` (10w), `timestep_spacing` (5w), `use_dynamic_shifting` (7w)

### `root.text_encoder` — 29 rows — assigned **H8** (supporting text-encoder tower flags)

`attention_bias` (4w), `attn_logit_softcapping` (2w), `feed_forward_proj` (6w), `final_logit_softcapping` (2w), `image_token_id` (1w), `image_token_index` (1w), `is_decoder` (6w), `is_encoder_decoder` (14w), `is_gated_act` (6w), `layer_norm_eps` (2w), `layer_norm_epsilon` (6w), `layer_types` (3w), `max_window_layers` (1w), `mlp_bias` (1w), `mrope_section` (2w), `query_pre_attn_scalar` (2w), `rms_norm_eps` (6w), `rope_parameters` (6w), `rope_scaling` (6w), `rope_theta` (6w), `rope_type` (6w), `text_config` (2w), `type` (1w), `use_bidirectional_attention` (2w), `use_sliding_window` (1w), `video_token_id` (1w), `vision_config` (2w), `vision_end_token_id` (1w), `vision_start_token_id` (1w)

### `root.text_encoder_2` — 4 rows — assigned **H8** (supporting text-encoder tower flags)

`feed_forward_proj` (1w), `is_encoder_decoder` (3w), `is_gated_act` (1w), `layer_norm_eps` (3w)

### `root.text_encoder_3` — 5 rows — assigned **H8** (supporting text-encoder tower flags)

`feed_forward_proj` (1w), `is_decoder` (1w), `is_encoder_decoder` (1w), `is_gated_act` (1w), `layer_norm_epsilon` (1w)

### `root.vae` — 38 rows — assigned **H7** (VAE geometry/IO block (source-derived in the UNet/VAE unit))

`add_attention_block` (1w), `attn_scales` (2w), `base_dim` (2w), `batch_norm_eps` (1w), `batch_norm_momentum` (1w), `decoder_act_fns` (1w), `decoder_block_types` (1w), `decoder_causal` (1w), `decoder_layers_per_block` (1w), `decoder_norm_types` (1w), `decoder_qkv_multiscales` (1w), `dim_mult` (2w), `down_block_types` (8w), `downsample_block_type` (1w), `encoder_block_out_channels` (2w), `encoder_block_types` (1w), `encoder_causal` (1w), `encoder_layers_per_block` (1w), `encoder_qkv_multiscales` (1w), `invert_scale_latents` (1w), `latents_mean` (7w), `latents_std` (7w), `mid_block_add_attention` (4w), `norm_num_groups` (8w), `out_channels` (11w), `resnet_norm_eps` (1w), `sample_size` (6w), `scaling_factor` (11w), `shift_factor` (4w), `spatial_compression_ratio` (1w), `spatial_expansions` (1w), `spatio_temporal_scaling` (1w), `temperal_downsample` (2w), `temporal_expansions` (1w), `up_block_types` (8w), `upsample_block_type` (1w), `use_post_quant_conv` (5w), `use_quant_conv` (5w)

### `root.vision` — 25 rows — assigned **H8** (vision tower + grid-stream fields (modality tail))

`depth` (2w), `fullatt_block_indexes` (1w), `hidden_act` (3w), `image_size` (1w), `image_token_id` (2w), `image_token_index` (1w), `in_channels` (2w), `in_chans` (2w), `intermediate_size` (2w), `mlp_ratio` (1w), `num_attention_heads` (1w), `num_channels` (1w), `num_heads` (2w), `num_hidden_layers` (3w), `patch_size` (3w), `projector_hidden_act` (1w), `rope_parameters` (1w), `spatial_merge_size` (2w), `temporal_patch_size` (2w), `tokens_per_second` (1w), `video_token_id` (2w), `vision_config` (3w), `vision_end_token_id` (2w), `vision_feature_layer` (1w), `vision_start_token_id` (2w)

## Regeneration

```bash
# from unfold-pkg/ — rebuilds this census from the live corpus
python3 /dev/stdin <<'PY'
import json, pathlib
from collections import defaultdict
import model_unfolder as mu
rows = defaultdict(set); inc = defaultdict(set)
for p in sorted(pathlib.Path('tests/sable_test_corpus').glob('*.json')):
    ca = (mu.unfold(json.loads(p.read_text())['config']).to_ir()
          .get('extras') or {}).get('config_access') or {}
    for e in ca.get('accessed_unconsumed') or []: rows[e].add(p.stem)
    for o in ca.get('audit_incomplete') or []: inc[o].add(p.stem)
print(len(rows), 'rows;', len(inc), 'incomplete owners')
PY
```
