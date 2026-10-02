# Strict capture benchmark — measured results

Repos with a verdict: **1944**  |  **FULL** 1083  **OUT** 470  **PARTIAL** 270  **FAIL** 104  **FULL_LEFTOVERS** 17

Runnable 1474: FULL 73.5%  |  architecture-complete (FULL + FULL_LEFTOVERS) 74.6%

FULL = every shipped weight used, every module executed, closed dataflow, no opaque ops, identical structure for two inputs. FULL_LEFTOVERS = everything the model runs is captured; the only gap is shipped tensors the library itself certifies as unused by this model and that no available runtime executes (listed per result). Architecture-complete = FULL + FULL_LEFTOVERS.

## By category

| category | repos | FULL | FULL_LEFTOVERS | PARTIAL | FAIL | OUT | FULL % of runnable | architecture-complete % |
|---|---|---|---|---|---|---|---|---|
| text_decoder | 436 | 300 | 4 | 37 | 8 | 87 | 86% | 87% |
| image_diffusion | 332 | 140 | 0 | 83 | 26 | 83 | 56% | 56% |
| encoder_only | 221 | 156 | 3 | 26 | 13 | 23 | 79% | 80% |
| vision_language | 174 | 95 | 3 | 15 | 8 | 53 | 79% | 81% |
| uncategorized | 151 | 72 | 1 | 34 | 19 | 25 | 57% | 58% |
| vision_backbone | 144 | 112 | 0 | 5 | 13 | 14 | 86% | 86% |
| video_diffusion | 91 | 27 | 0 | 13 | 0 | 51 | 68% | 68% |
| detection_segmentation | 89 | 47 | 1 | 11 | 8 | 22 | 70% | 72% |
| speech_recognition | 82 | 54 | 0 | 10 | 0 | 18 | 84% | 84% |
| audio_generation | 69 | 7 | 0 | 15 | 0 | 47 | 32% | 32% |
| encoder_decoder | 57 | 43 | 3 | 5 | 2 | 4 | 81% | 87% |
| dense_vision_other | 34 | 9 | 0 | 12 | 3 | 10 | 38% | 38% |
| omni_any_to_any | 26 | 4 | 2 | 4 | 3 | 13 | 31% | 46% |
| audio_classification | 20 | 9 | 0 | 0 | 0 | 11 | 100% | 100% |
| time_series | 18 | 8 | 0 | 0 | 1 | 9 | 89% | 89% |

## Why PARTIAL (which strict checks failed)

- T1_weights: 260
- T2_modules: 167
- T5_stable_structure: 43
- T3_closed_dataflow: 7

## Why FAIL / OUT

- **FAIL**: forward failed (60), pipeline could not be assembled (23), build failed (7), StrictDataclassFieldValidationError (4), crash/OOM (3), repo incomplete (component files missing) (2), ZeroDivisionError (1), KeyError (1), StrictDataclassClassValidationError (1), timeout (1), AttributeError (1)
- **OUT**: class not in installed library (222), gated (92), GGUF/MLX/adapter pointer (53), remote code only (46), gguf architecture not supported by installed libra (25), MLX-quantized checkpoint (packed U32 + scales/bias (11), weights only in a converted format the library can (10), the library's GGUF loader cannot read this file (7), components come from a library that is not install (4)

## By subcategory

| category / subcategory | repos | FULL | PARTIAL | FAIL | OUT |
|---|---|---|---|---|---|
| audio_classification / audio_classification | 20 | 9 | 0 | 0 | 11 |
| audio_generation / audio_lm | 26 | 4 | 4 | 0 | 18 |
| audio_generation / music_generation | 5 | 1 | 4 | 0 | 0 |
| audio_generation / neural_codec | 1 | 1 | 0 | 0 | 0 |
| audio_generation / tts | 37 | 1 | 7 | 0 | 29 |
| dense_vision_other / depth_estimation | 14 | 2 | 10 | 0 | 2 |
| dense_vision_other / other_dense | 20 | 7 | 2 | 3 | 8 |
| detection_segmentation / detr_style | 30 | 25 | 2 | 1 | 2 |
| detection_segmentation / mask2former | 15 | 12 | 0 | 2 | 1 |
| detection_segmentation / other | 33 | 10 | 6 | 0 | 16 |
| detection_segmentation / sam | 11 | 0 | 3 | 5 | 3 |
| encoder_decoder / bart_style | 16 | 15 | 0 | 0 | 1 |
| encoder_decoder / other | 7 | 4 | 1 | 0 | 2 |
| encoder_decoder / t5_style | 15 | 8 | 1 | 2 | 1 |
| encoder_decoder / translation | 19 | 16 | 3 | 0 | 0 |
| encoder_only / bert_family | 84 | 51 | 19 | 7 | 5 |
| encoder_only / embeddings | 137 | 105 | 7 | 6 | 18 |
| image_diffusion / auraflow | 2 | 2 | 0 | 0 | 0 |
| image_diffusion / dit_pixart_style | 4 | 1 | 1 | 1 | 1 |
| image_diffusion / flux_style | 20 | 16 | 2 | 0 | 2 |
| image_diffusion / hunyuandit | 1 | 1 | 0 | 0 | 0 |
| image_diffusion / mmdit | 6 | 3 | 2 | 0 | 1 |
| image_diffusion / other | 103 | 41 | 31 | 14 | 17 |
| image_diffusion / qwen_image | 7 | 0 | 5 | 1 | 1 |
| image_diffusion / sana | 4 | 3 | 0 | 0 | 1 |
| image_diffusion / unet_sd | 185 | 73 | 42 | 10 | 60 |
| omni_any_to_any / any_to_any | 26 | 4 | 4 | 3 | 13 |
| speech_recognition / conformer | 1 | 0 | 1 | 0 | 0 |
| speech_recognition / other | 37 | 12 | 8 | 0 | 17 |
| speech_recognition / wav2vec2_hubert | 38 | 37 | 1 | 0 | 0 |
| speech_recognition / whisper_style | 6 | 5 | 0 | 0 | 1 |
| text_decoder / dense | 290 | 187 | 17 | 3 | 79 |
| text_decoder / hybrid_ssm_attention | 37 | 30 | 6 | 1 | 0 |
| text_decoder / linear_recurrent_attention | 1 | 1 | 0 | 0 | 0 |
| text_decoder / mla | 17 | 7 | 8 | 0 | 2 |
| text_decoder / moe_grouped_hash_routing | 23 | 18 | 2 | 1 | 2 |
| text_decoder / moe_shared_experts | 12 | 8 | 3 | 0 | 1 |
| text_decoder / mtp | 4 | 3 | 1 | 0 | 0 |
| text_decoder / sliding_local_global | 52 | 46 | 0 | 3 | 3 |
| time_series / forecasting | 18 | 8 | 0 | 1 | 9 |
| uncategorized / unclassified | 151 | 72 | 34 | 19 | 25 |
| video_diffusion / cogvideox | 4 | 3 | 1 | 0 | 0 |
| video_diffusion / hunyuanvideo | 4 | 4 | 0 | 0 | 0 |
| video_diffusion / ltx | 5 | 3 | 1 | 0 | 1 |
| video_diffusion / mochi | 1 | 0 | 1 | 0 | 0 |
| video_diffusion / other | 65 | 12 | 5 | 0 | 48 |
| video_diffusion / wan | 12 | 5 | 5 | 0 | 2 |
| vision_backbone / clip_siglip_dual_encoder | 23 | 21 | 0 | 0 | 2 |
| vision_backbone / convnext_swin | 14 | 14 | 0 | 0 | 0 |
| vision_backbone / dino_ssl | 8 | 4 | 0 | 0 | 4 |
| vision_backbone / other | 51 | 31 | 5 | 12 | 3 |
| vision_backbone / vit | 48 | 42 | 0 | 1 | 5 |
| vision_language / cross_attention_perceiver | 4 | 2 | 2 | 0 | 0 |
| vision_language / native_multimodal | 23 | 19 | 0 | 1 | 2 |
| vision_language / other | 133 | 60 | 13 | 7 | 51 |
| vision_language / projector_mlp | 12 | 12 | 0 | 0 | 0 |
| vision_language / video_llm | 2 | 2 | 0 | 0 | 0 |

Cost per repo: median 32s, p90 88s; peak RAM median 0.54 GB, max 5.58 GB