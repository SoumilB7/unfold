# Blocks supported

Every clickable thing in a diagram is a **block**. There are two layers:

1. **IR blocks** — emitted by the adapter into `LayerSpec.blocks` / `extras`. Each
   carries `id`, `role`, `kind`, label/title/description, and (when it can be
   expanded) a `view` + `children`. Defined under
   `model_unfolder/adapters/...`.
2. **Detail views** ("opening architectures") — the SVG drawn when a block is
   clicked. Routed by a block's `view` key through the single recursive router in
   [block_views/registry.py](../model_unfolder/renderers/html/block_views/registry.py).
   Defined under `model_unfolder/renderers/html/block_views/...`.


---

## 1. Model-level blocks (decoder bookends)

| Block | id | Defined in |
|---|---|---|
| Tokenized text | `tok_text` | model_unfolder/adapters/transformer/blocks/model.py |
| Token Embedding layer | `embed` | model_unfolder/adapters/transformer/blocks/model.py |
| Final RMSNorm | `final_rms` | model_unfolder/adapters/transformer/blocks/model.py |
| Linear output layer (LM head) | `lm_head` | model_unfolder/adapters/transformer/blocks/model.py |

## 2. Decoder-layer topology blocks

| Block | id | Defined in |
|---|---|---|
| Pre-attention norm (RMSNorm/LayerNorm) | `rms1` | model_unfolder/adapters/transformer/blocks/layers.py |
| Attention | `attn` | model_unfolder/adapters/transformer/blocks/layers.py |
| Residual add (post-attention) | `add1` | model_unfolder/adapters/transformer/blocks/layers.py |
| Pre-FFN norm | `rms2` | model_unfolder/adapters/transformer/blocks/layers.py |
| Feed-Forward / MoE | `ffn` | model_unfolder/adapters/transformer/blocks/layers.py |
| Residual add (post-FFN) | `add2` | model_unfolder/adapters/transformer/blocks/layers.py |
| Shared pre-block norm (parallel residual) | `rms1` | model_unfolder/adapters/transformer/blocks/layers.py |
| Residual add (parallel, combined) | `add1` | model_unfolder/adapters/transformer/blocks/layers.py |

## 3. Attention child blocks (inside `attn`, selected by `attention.kind`)

Dispatched in `attention_child_blocks(...)`.

### SDPA — MHA / GQA / MQA
| Block | id | Defined in |
|---|---|---|
| Query projection | `q_proj` | model_unfolder/adapters/transformer/blocks/attention.py |
| Key projection | `k_proj` | model_unfolder/adapters/transformer/blocks/attention.py |
| Value projection | `v_proj` | model_unfolder/adapters/transformer/blocks/attention.py |
| Scaled / grouped / multi-query scores | `scaled_scores` (`qkv_dot`) | model_unfolder/adapters/transformer/blocks/attention.py |
| Softmax weights | `attn_softmax` | model_unfolder/adapters/transformer/blocks/attention.py |
| Apply values | `attn_apply_v` | model_unfolder/adapters/transformer/blocks/attention.py |
| Concatenate heads | `concat_heads` | model_unfolder/adapters/transformer/blocks/attention.py |
| Output projection | `o_proj` | model_unfolder/adapters/transformer/blocks/attention.py |

### MLA (Multi-head Latent Attention)
| Block | id | Defined in |
|---|---|---|
| Query path (group) | `mla_query_path` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Q projection | `mla_q` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Q noPE slice | `mla_q_nope` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Q RoPE slice | `mla_q_rope` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Apply RoPE to query | `mla_q_rope_apply` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Final MLA query (concat) | `mla_q_concat` | model_unfolder/adapters/transformer/blocks/attention.py |
| KV cache path (group) | `mla_kv_path` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ K/V latent compression | `mla_kv_down` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Stored latent cache c_t | `mla_cache` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ K/V head expansion | `mla_kv_up` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Latent key content | `mla_k_nope` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Key positional slice | `mla_k_rope` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Apply RoPE to key | `mla_k_rope_apply` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Composed MLA key (concat) | `mla_k_merge` | model_unfolder/adapters/transformer/blocks/attention.py |
| ↳ Latent value heads | `mla_v` | model_unfolder/adapters/transformer/blocks/attention.py |
| Latent scores | `scaled_scores` | model_unfolder/adapters/transformer/blocks/attention.py |
| Softmax / Apply V / Concat heads / Output proj | `attn_softmax`, `attn_apply_v`, `concat_heads`, `o_proj` | model_unfolder/adapters/transformer/blocks/attention.py |

### SSM (Mamba / Mamba-2 / Jamba / Zamba / Falcon-H1)
| Block | id | Defined in |
|---|---|---|
| SSM input projection | `ssm_in_proj` | model_unfolder/adapters/transformer/blocks/attention.py |
| Short convolution | `ssm_conv` | model_unfolder/adapters/transformer/blocks/attention.py |
| Selective state-space scan | `ssm_scan` | model_unfolder/adapters/transformer/blocks/attention.py |
| SSM gate | `ssm_gate` | model_unfolder/adapters/transformer/blocks/attention.py |
| SSM output projection | `ssm_out_proj` | model_unfolder/adapters/transformer/blocks/attention.py |

### Recurrent / LRU (RecurrentGemma)
| Block | id | Defined in |
|---|---|---|
| LRU input projection | `lru_in_proj` | model_unfolder/adapters/transformer/blocks/attention.py |
| Linear recurrent state | `lru_state` | model_unfolder/adapters/transformer/blocks/attention.py |
| Recurrent gate | `lru_gate` | model_unfolder/adapters/transformer/blocks/attention.py |
| LRU output projection | `lru_out_proj` | model_unfolder/adapters/transformer/blocks/attention.py |

### RWKV
| Block | id | Defined in |
|---|---|---|
| Receptance gate | `rwkv_receptance` | model_unfolder/adapters/transformer/blocks/attention.py |
| RWKV key projection | `rwkv_key` | model_unfolder/adapters/transformer/blocks/attention.py |
| RWKV value projection | `rwkv_value` | model_unfolder/adapters/transformer/blocks/attention.py |
| Time-decay recurrence | `rwkv_time_mix` | model_unfolder/adapters/transformer/blocks/attention.py |
| RWKV output projection | `rwkv_out` | model_unfolder/adapters/transformer/blocks/attention.py |

### Linear attention (MiniMax lightning, etc.)
| Block | id | Defined in |
|---|---|---|
| Query / Key / Value projection | `q_proj`, `k_proj`, `v_proj` | model_unfolder/adapters/transformer/blocks/attention.py |
| Kernel feature map | `kernel_map` | model_unfolder/adapters/transformer/blocks/attention.py |
| Linear attention mix | `linear_mix` | model_unfolder/adapters/transformer/blocks/attention.py |
| Output projection | `o_proj` | model_unfolder/adapters/transformer/blocks/attention.py |

## 4. FFN child blocks (inside `ffn`, selected by `ffn.kind`/`ffn.gated`)

| Block | id | Defined in |
|---|---|---|
| **Dense** input projection | `up_proj` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| Dense activation | `silu` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| Dense output projection | `down_proj` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| **Gated** gate projection | `gate_proj` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| Gated up projection | `up_proj` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| Gated activation | `silu` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| Gated element-wise multiply | `mul` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| Gated down projection | `down_proj` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| **MoE** router | `router` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| MoE expert FFN (×N, with shared) | `expert_1`/`expert_k`/`expert_kp1`/`expert_n` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| MoE weighted sum | `add_moe` | model_unfolder/adapters/transformer/blocks/feed_forward.py |
| ↳ Expert gate/act/up/mul/down | `expert_gate_proj`, `expert_act`, `expert_up_proj`, `expert_mul`, `expert_down_proj` | model_unfolder/adapters/transformer/blocks/feed_forward.py |

## 5. Per-Layer Embedding (PLE side pathway — Gemma 3n / 4)

| Block | id | Defined in |
|---|---|---|
| Per-Layer Embeddings | `ple` | model_unfolder/adapters/transformer/special_parts/per_layer_embedding.py |
| ↳ Per-layer input gate | `ple_gate` | model_unfolder/adapters/transformer/special_parts/per_layer_embedding.py |
| ↳ PLE activation | `ple_act` | model_unfolder/adapters/transformer/special_parts/per_layer_embedding.py |
| ↳ Per-layer gate multiply | `ple_mul` | model_unfolder/adapters/transformer/special_parts/per_layer_embedding.py |
| ↳ Per-layer input vector | `per_layer_input` | model_unfolder/adapters/transformer/special_parts/per_layer_embedding.py |
| ↳ Per-layer projection | `ple_proj` | model_unfolder/adapters/transformer/special_parts/per_layer_embedding.py |
| ↳ Post-PLE norm | `ple_norm` | model_unfolder/adapters/transformer/special_parts/per_layer_embedding.py |
| Residual add (PLE) | `add3` | model_unfolder/adapters/transformer/special_parts/per_layer_embedding.py |
| External pathway: lookup / proj / combine | `ple_lookup`, `ple_proj_in`, `ple_combine` | model_unfolder/adapters/transformer/special_parts/per_layer_embedding.py |

## 6. Cross-attention side state (mllama / Llama-4 vision conditioning)

| Block | id | Defined in |
|---|---|---|
| Projected image states | `cross_attention_states` | model_unfolder/adapters/transformer/parser.py |

## 6b. Multi-Token Prediction (MTP) head (DeepSeek-V3 style)

Detected from `num_nextn_predict_layers` / `num_mtp_layers`; drawn as a stacked-card glyph above `lm_head`.

| Block | id | Defined in |
|---|---|---|
| MTP head (stack) | `mtp` | model_unfolder/adapters/transformer/blocks/model.py |
| ↳ Hidden-state norm | `mtp_hnorm` | model_unfolder/adapters/transformer/blocks/model.py |
| ↳ Next-token embedding | `mtp_emb` | model_unfolder/adapters/transformer/blocks/model.py |
| ↳ Embedding norm | `mtp_enorm` | model_unfolder/adapters/transformer/blocks/model.py |
| ↳ Concatenate | `mtp_concat` | model_unfolder/adapters/transformer/blocks/model.py |
| ↳ Projection (eh_proj) | `mtp_proj` | model_unfolder/adapters/transformer/blocks/model.py |
| ↳ Transformer block | `mtp_block` | model_unfolder/adapters/transformer/blocks/model.py |
| ↳ Shared output head | `mtp_head` | model_unfolder/adapters/transformer/blocks/model.py |

(Drawing: `_draw_mtp_head` in model_unfolder/renderers/html/views.py)

## 7. Multimodal model-level blocks (clickable pathways)

Built in `_MODALITY_BLOCK_SPECS`; structural path data comes from the modality builders.

| Block | id / view | Defined in |
|---|---|---|
| Vision → tokens | `vision_path` | model_unfolder/renderers/html/metadata_modalities.py |
| Audio → tokens | `audio_path` | model_unfolder/renderers/html/metadata_modalities.py |
| Video → grid | `video_path` | model_unfolder/renderers/html/metadata_modalities.py |
| Multimodal fusion / Vision cross-attention | `multimodal_fusion` | model_unfolder/renderers/html/metadata_modalities.py |
| Vision: patch embedding (sub-block) | `vision_patch_embedding` | model_unfolder/renderers/html/metadata_modalities.py |
| Vision: encoder (sub-block) | `vision_encoder` | model_unfolder/renderers/html/metadata_modalities.py |
| Vision: self-attention (sub-block) | `vision_self_attention` | model_unfolder/renderers/html/metadata_modalities.py |
| Vision: MLP (sub-block) | `vision_mlp` | model_unfolder/renderers/html/metadata_modalities.py |

### Modality path stages (structural data, `Stage` envelope)

| Stage source | Defined in |
|---|---|
| Vision path stages (input/tiling/patch/encoder/reduction/projector/tokens) | model_unfolder/adapters/transformer/special_parts/modalities/vision.py |
| Audio path stages | model_unfolder/adapters/transformer/special_parts/modalities/audio.py |
| Video companion path | model_unfolder/adapters/transformer/special_parts/modalities/vision.py |
| Fusion path (cross-attn / unified-stream / prefix / placeholder) | model_unfolder/adapters/transformer/special_parts/modalities/fusion.py |
| Stage / assemble_path envelope | model_unfolder/adapters/transformer/special_parts/modalities/schema.py |

---

## Detail views ("opening architectures")

When a block with a `view` is clicked, the single recursive router draws the expansion.
Routing table: [block_views/registry.py](../model_unfolder/renderers/html/block_views/registry.py).

| `view` | Builder | Defined in |
|---|---|---|
| `attention` (router) | `build_attention_view` → dispatch by kind | model_unfolder/renderers/html/block_views/attention.py |
| ↳ MHA / SDPA (default) | `build` | model_unfolder/renderers/html/block_views/attention_types/multi_head.py |
| ↳ GQA | `build` | model_unfolder/renderers/html/block_views/attention_types/grouped_query.py |
| ↳ MQA | `build` | model_unfolder/renderers/html/block_views/attention_types/multi_query.py |
| ↳ MLA | `build` | model_unfolder/renderers/html/block_views/attention_types/latent.py |
| ↳ SSM | `build_ssm` | model_unfolder/renderers/html/block_views/attention_types/state_space.py |
| ↳ Recurrent / LRU | `build_recurrent` | model_unfolder/renderers/html/block_views/attention_types/state_space.py |
| ↳ RWKV | `build` | model_unfolder/renderers/html/block_views/attention_types/rwkv.py |
| ↳ Linear attention | `build` | model_unfolder/renderers/html/block_views/attention_types/linear.py |
| `mla_query_path` | `build_query_path_view` | model_unfolder/renderers/html/block_views/attention_types/latent.py |
| `mla_kv_cache_path` | `build_kv_cache_view` | model_unfolder/renderers/html/block_views/attention_types/latent.py |
| `gated_ffn` | `build_ffn_view` | model_unfolder/renderers/html/block_views/feed_forward.py |
| `dense_ffn` | `build_dense_ffn_view` | model_unfolder/renderers/html/block_views/feed_forward.py |
| `moe` | `build_moe_view` | model_unfolder/renderers/html/block_views/mixture_of_experts.py |
| `moe_expert` | `build_moe_expert_view` | model_unfolder/renderers/html/block_views/mixture_of_experts.py |
| `per_layer_embedding` | `build_per_layer_embedding_view` | model_unfolder/renderers/html/block_views/per_layer_embedding.py |
| `vision_path` | `build_vision_path_view` | model_unfolder/renderers/html/block_views/modality_views/vision.py |
| `audio_path` | `build_audio_path_view` | model_unfolder/renderers/html/block_views/modality_views/audio.py |
| `video_path` | `build_video_path_view` | model_unfolder/renderers/html/block_views/modality_views/video.py |
| `multimodal_fusion` (placeholder) | `build_multimodal_fusion_view` | model_unfolder/renderers/html/block_views/modality_views/fusion_placeholder.py |
| ↳ fusion: cross-attention | `build_cross_attention_fusion_view` | model_unfolder/renderers/html/block_views/modality_views/fusion_cross_attention.py |
| ↳ fusion: unified grid stream | `build_unified_stream_view` | model_unfolder/renderers/html/block_views/modality_views/fusion_grid.py |
| `vision_patch_embedding` | `build_patch_embedding_view` | model_unfolder/renderers/html/block_views/modality_views/vision_details.py |
| `vision_encoder` | `build_vision_encoder_view` | model_unfolder/renderers/html/block_views/modality_views/vision_details.py |
| `vision_self_attention` | `build_vision_self_attention_view` | model_unfolder/renderers/html/block_views/modality_views/vision_details.py |
| `vision_mlp` | `build_vision_mlp_view` | model_unfolder/renderers/html/block_views/modality_views/vision_details.py |
| `mtp_head` | `build_mtp_head_view` | model_unfolder/renderers/html/block_views/mtp_head.py |

---

## Block `kind`s (rendering glyphs)

The architecture view picks a glyph/slot from `kind`:
`source`, `embedding`, `norm`, `attention`, `ffn`, `linear`, `activation`,
`gate_mul`, `residual_add`, `ple`, `output`, plus modality `fusion` and the
modality pathway kinds. Roles (`input`, `embedding`, `norm`, `attention`,
`ffn`, `residual`, `gate`, `ple`, `output`, `vision`) drive tooltips and
inspect cards. See the contract in
[blocks/__init__.py](../model_unfolder/adapters/transformer/blocks/__init__.py).
