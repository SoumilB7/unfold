# SHOULD SUPPORT coverage audit

This note maps the requested major OSS models onto the parser/rendering
vocabulary in `model_unfolder`. The goal is not just "can we read the config",
but whether the rendered architecture tells the truth when the model changes
inside a transformer block, swaps the block type, or wraps the text model in a
multimodal shell.

## Current vocabulary

The IR already has the right high-level split:

- Attention kinds: `mha`, `gqa`, `mqa`, `mla`, `ssm`, `recurrent`, `linear`,
  `rwkv`.
- FFN kinds: `dense`, `moe`.
- Per-layer changes: attention mask/window, KV source layer, QK norm,
  no-RoPE, shared layer, dense-vs-MoE phase changes.
- Cross-layer edges: currently KV sharing.

The dedicated transformer family adapters currently cover:

- `llama`: Llama and Llama 4 text-ish configs.
- `phi`: Phi-2, Phi-3/Phi-4-style configs, and Phi-MoE field shapes.
- `yi`: Yi Llama-like configs.
- `olmo`: OLMo, OLMo-2, and OLMoE configs.
- `dbrx`: Databricks DBRX nested GQA + MoE configs.
- `mistral`: Mistral, Mixtral, Mistral3/Pixtral wrappers.
- `qwen`: Qwen2/Qwen3, Qwen MoE, Qwen hybrid linear-attention configs.
- `gemma`: Gemma 1/CodeGemma, Gemma 2, Gemma 3, Gemma 4/Gemma4n-style
  PLE and KV sharing.
- `deepseek`: DeepSeek V2/V3/Kimi-style MLA and DeepSeek MoE.
- `falcon`: Falcon classic and Falcon-H1 SSM-ish configs.
- `cohere`: Command-R / Command-R+ / Cohere2 QK-norm GQA.
- `recurrent_gemma`: Griffin/RecurrentGemma LRU + local attention.
- plus Jamba/Zamba/Minimax/RWKV/GPT-NeoX and a generic fallback.

The fallback is useful as a smoke detector, but if a family has custom nested
config fields or a nonstandard block, it should get a real adapter.

## Highest-risk gaps first

| Area | Why it matters | Current status | Needed action |
| --- | --- | --- | --- |
| DBRX | MoE config is nested under `ffn_config`; attention KV heads live under `attn_config`. Fallback aliases would miss this shape. | Covered by `families/dbrx.py`. | Later: tune any DBRX-specific renderer wording if needed. |
| Phi-2 | `model_type="phi"`, MHA, `gelu_new`, partial rotary, layernorm-ish Phi topology. | Covered by `families/phi.py`. | Later: verify Phi multimodal wrapper details. |
| Gemma 1 / CodeGemma | `model_type="gemma"` dense Gemma decoder. | Covered by `families/gemma/gemma1.py`. | Later: verify exact CodeGemma variants. |
| Qwen2-VL | Text config is wrapped in a multimodal config with vision/projector pieces. | Current Qwen adapter likely misses `qwen2_vl` unless fallback unwraps it. | Add wrapper handling in Qwen adapter; render text stack plus "vision wrapper ignored" warning. |
| Llama 3.2 Vision / Llama 4 multimodal | Text model lives under wrapper; Llama 4 adds MoE, QK norm, chunked attention, vision tower. | Llama 4 text fields are partly covered; wrapper handling should be explicit. | Add clean text-config unwrap in Llama adapter and expose multimodal wrapper metadata. |
| Falcon3 | Newer Falcon family may not use classic Falcon config shape. | Risky. | Verify config; add `falcon3` branch if fields differ. |
| Yi | Llama-like, but may use `model_type="yi"` rather than `llama`. | Covered by `families/yi.py`. | Later: verify exact Yi-1.5 variants. |
| OLMo 1 / OLMoE | OLMo 1 and OLMoE use distinct family names and MoE fields. | Covered by `families/olmo.py`. | Later: refine OLMo norm/activation metadata from real configs. |
| Older DeepSeek LLM/Coder | Pre-V2 DeepSeek models are not `deepseek_v2/v3`; they are Llama-like. | Gap/approx. | Add `deepseek_legacy` alias or route to Llama-like parser. |
| Sarvam 2B / Sarvam 105B | Sarvam-M is Mistral-like; the others need config verification. | Unknown until config read. | Add only aliases backed by real config fields. |

## Meta Llama family

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `huggyllama/llama-7b`, `13b`, `30b`, `65b` | Llama 1 decoder: MHA, RMSNorm, SwiGLU dense FFN, RoPE, causal mask. | No special topology. Big variants only scale dims/layers. | `llama` adapter should cover if config says `llama`. |
| `meta-llama/Llama-2-7b-hf`, `13b-hf`, chat variants | Llama 2 7B/13B: MHA + dense SwiGLU. | No special topology. | `llama`. |
| `meta-llama/Llama-2-70b-hf`, chat | Llama 2 70B: GQA + dense SwiGLU. | GQA KV sharing view should be correct. | `llama`. |
| `codellama/CodeLlama-7b-hf`, `13b-hf` | Code Llama small: Llama-like MHA dense decoder. | Long context/RoPE scaling should be metadata, not topology. | `llama`. |
| `codellama/CodeLlama-34b-hf`, `70b-hf` | Llama-like, larger variants generally use GQA. | Same as Llama 2 70B: GQA + long-context metadata. | `llama`. |
| `meta-llama/Meta-Llama-3-8B`, `70B` | Llama 3: GQA, dense SwiGLU, RMSNorm, RoPE. | Better GQA internals and KV head grouping matter. | `llama`. |
| `meta-llama/Llama-3.1-8B`, `70B`, `405B` | Llama 3.1: GQA, dense SwiGLU, larger rope scaling/context. | Need show rope/context metadata without changing topology. | `llama`. |
| `meta-llama/Llama-3.2-1B`, `3B` | Text-only small Llama 3.2: GQA/dense decoder. | Straightforward. | `llama`. |
| `meta-llama/Llama-3.2-11B-Vision`, `90B-Vision` | Multimodal wrapper with vision tower/projector plus Llama text decoder. | The text stack is not enough if we claim full model coverage; render should show wrapper and vision path as top-level metadata. | Needs explicit multimodal unwrap/metadata. |
| `meta-llama/Llama-4-Scout`, `Maverick` | Llama 4-style wrapper: GQA, MoE experts, QK norm, possible chunked attention, no-RoPE/iRoPE metadata, vision tower. | This is a real custom option: MoE block, QK-norm attention, text_config unwrap, and vision shell. | Partially in `llama`; should be hardened. |

Design note: Llama-family block arrangement stays the usual pre-norm residual
decoder unless Llama 4 marks no-RoPE/chunked attention/MoE. The block type
changes only at attention kind (`mha`/`gqa`) or FFN kind (`dense`/`moe`).

## Mistral family

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `mistralai/Mistral-7B-v0.1`, `Mistral-7B-Instruct-v0.3` | GQA decoder, dense SwiGLU FFN, RMSNorm, sliding-window attention in classic configs. | Alternating/global vs all-sliding needs correct mask labels. | `mistral`. |
| `mistralai/Mixtral-8x7B-*`, `Mixtral-8x22B-*` | GQA + sparse MoE FFN, usually top-2 experts. | MoE details must open to router -> selected experts -> expert FFN. | `mistral`. |
| `mistralai/Mistral-Small-24B-Instruct-2501`, `Mistral-Large-Instruct-2407`, `Codestral-22B-v0.1` | Mistral-style GQA dense decoder variants. | Mostly config dimensions/context. | `mistral`, verify exact model_type. |
| `mistralai/Pixtral-12B-2409` | Multimodal Mistral/Pixtral wrapper with image side plus text decoder. | Text stack should render, but full model needs vision wrapper. | `mistral` wrapper partially. |
| `mistralai/Devstral-Small-2505` | Mistral-small derivative, likely GQA dense. | Verify exact config; probably no custom block. | likely `mistral`. |

Design note: Mistral challenges are mostly layer masks and MoE, not exotic
attention math.

## Qwen family

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `Qwen/Qwen1.5-72B`, `Qwen/Qwen2-72B`, `Qwen/Qwen2.5-72B`, `Qwen/Qwen2.5-Coder-32B`, `Qwen/QwQ-32B` | Qwen decoder: GQA, dense gated FFN, RMSNorm/RoPE. | Qwen-specific rope/context metadata; sometimes QK norm. | `qwen`. |
| `Qwen/Qwen3-32B` | Qwen3 dense GQA decoder. | Qwen3 metadata; check QK norm/rope settings. | `qwen`. |
| `Qwen/Qwen3-235B-A22B` | Qwen3 MoE: GQA plus 128 experts, top-8 active, expert hidden size separate from dense hidden. | MoE renderer must show active experts and shared/router metadata cleanly. | `qwen`. |
| `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct` | Multimodal wrapper with Qwen2 text decoder and vision tower. | Needs wrapper unwrapping; full render should not pretend it is text-only. | Gap: add Qwen-VL unwrap. |

Design note: Qwen MoE is not "just dense FFN with experts"; expert hidden size,
top-k, shared experts, and dense-only layers should be explicit when present.

## Google Gemma family

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `google/gemma-2b`, `google/gemma-7b`, `google/codegemma-7b-it` | Gemma 1-style dense decoder, GQA/MQA-ish depending variant, GeGLU/GELU-style gated FFN. | Straight dense Gemma path. | `gemma1`. |
| `google/gemma-2-9b`, `google/gemma-2-27b` | Gemma 2: GQA, dense gated FFN, alternating local/global attention, logit/attention softcapping. | Softcapping and query scalar are custom metadata. | `gemma2`. |
| `google/gemma-3-1b-it`, `4b-it`, `12b-it`, `27b-it` | Gemma 3 text/multimodal wrapper: GQA, dense FFN, sliding/global layer types. | Multimodal wrapper and text_config handling; 1B may be text-only. | `gemma3`. |
| `google/recurrentgemma-2b` | Griffin/RecurrentGemma: recurrent LRU layers plus local MQA attention layers. | This is a block-type change, not just attention variant. Needs recurrent block view. | `recurrent_gemma`. |

Design note: Gemma 4/Gemma3n-style PLE and KV sharing are already modeled in
the code, but Gemma 1 is missing.

## DeepSeek family

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `deepseek-ai/deepseek-llm-67b-chat`, `deepseek-ai/deepseek-coder-33b-instruct` | Older DeepSeek Llama-like dense transformer, not MLA. | Model type is likely not `deepseek_v2/v3`; do not route to MLA. | Gap: add legacy DeepSeek alias/adapter. |
| `deepseek-ai/DeepSeek-V2`, `DeepSeek-V2.5` | MLA attention plus dense/MoE phase depending config. | MLA needs nested query/KV latent views; MoE phase changes. | `deepseek`. |
| `deepseek-ai/DeepSeek-V3`, `DeepSeek-R1` | MLA + MoE: KV LoRA rank, Q LoRA rank, noPE/RoPE split, routed/shared experts, MTP. | This is one of the hardest render targets: MLA subviews and MoE subviews must both be good. | `deepseek`. |
| `DeepSeek-R1-Distill-Llama-8B`, `DeepSeek-R1-Distill-Qwen-32B` | Distilled into Llama/Qwen architectures; ordinary GQA/dense text decoders. | Just route by actual config family. | `llama` / `qwen`. |

Design note: DeepSeek V2/V3/R1 are the reason `mla` must be a first-class
attention type rather than a "GQA with labels".

## Phi family

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `microsoft/phi-2` | Phi decoder: MHA, GELU-new dense FFN, partial RoPE, Phi-specific naming. | Dedicated non-gated FFN route. | `phi`. |
| `microsoft/Phi-3-mini-4k-instruct`, `Phi-3-medium-128k-instruct`, `Phi-3.5-mini-instruct`, `phi-4`, `phi-4-mini-instruct` | Phi-3/Phi-4-ish decoder configs, generally GQA/MQA/dense, long-context rope metadata. | Dedicated Phi route; avoids pretending these are Llama. | `phi`. |
| `microsoft/Phi-3.5-MoE-instruct` | Phi MoE variant. | Needs MoE fields validated; likely not identical to Mixtral. | Needs explicit check. |
| `microsoft/phi-4-multimodal-instruct` | Multimodal Phi wrapper plus text decoder. | Wrapper metadata; audio/vision paths should be surfaced. | Gap unless fallback unwraps enough. |

## Falcon

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `tiiuae/falcon-7b`, `falcon-40b`, `falcon-180B` | Falcon decoder: MQA/GQA/MHA depending size, dense GELU FFN, layernorm, often parallel attention/residual. | Parallel topology is custom; norm kind differs from Llama. | `falcon`. |
| `tiiuae/Falcon3-7B-Instruct` | New Falcon3 line; exact config should be verified. | May not share classic Falcon fields. | Risk: add `falcon3` branch if needed. |

## DBRX

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `databricks/dbrx-base`, `databricks/dbrx-instruct` | Decoder-only transformer with GQA and fine-grained MoE: nested `attn_config`/`ffn_config`, 16 experts, top-4 active in public config. | Dedicated parser reads nested attention/FFN config. MoE should show top-4 and inactive expert pool. | `dbrx`. |

DBRX is now handled explicitly because it is common and the config shape is
clearly distinct.

## Yi

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `01-ai/Yi-34B`, `01-ai/Yi-1.5-34B` | Llama-like dense decoder, GQA/MHA depending exact config. | Simple Llama-like topology with Yi family identity preserved. | `yi`. |

## OLMo

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `allenai/OLMo-7B` | OLMo 1 decoder, dense FFN, MHA/GQA depending release, different naming/norm defaults. | Uses OLMo aliases like `d_model`, `n_layers`, `mlp_ratio`. | `olmo`. |
| `allenai/OLMo-2-1124-7B` | OLMo 2-style decoder. | QK norm / norm details should be accurate. | `olmo`. |
| `allenai/OLMoE-1B-7B-0924` | OLMoE MoE decoder. | MoE route with top-k expert metadata. | `olmo`. |

## Command-R

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `CohereForAI/c4ai-command-r-v01`, `c4ai-command-r-plus` | Cohere GQA decoder with per-head Q/K normalization. Command-R7B/Cohere2 can include sliding/global patterns. | QK norm should show in attention internals, not just text metadata. | `cohere`. |

## Sarvam

| Models | What they are made of | Architecture challenges | Current path |
| --- | --- | --- | --- |
| `sarvamai/sarvam-m` | Public config is `MistralForCausalLM`, GQA dense decoder, model_type `mistral`. | No custom topology; Sarvam branding only. | `mistral`. |
| `sarvamai/sarvam-2b-v0.5`, `sarvamai/sarvam-105b` | Need exact configs before promising. They may be Mistral/Llama-like or custom MoE. | Treat as unknown until config-backed. | Verify, then alias or add adapter. |

## Custom options the renderer should be ready for

These are the actual feature knobs that will decide whether a model needs a
custom block view:

- `attention.kind`: MHA/GQA/MQA/MLA/SSM/recurrent/linear/RWKV.
- `attention.mask`: causal vs sliding vs global vs chunked.
- `attention.qk_norm`: Cohere/Llama4-style Q/K norm before score math.
- `attention.no_rope`: Llama4 iRoPE/no-RoPE layers.
- `attention.kv_source_layer`: Gemma4/YOCO-style cross-layer KV reuse.
- `ffn.kind`: dense vs MoE.
- MoE fields: total experts, top-k active, shared experts, dense-only layers,
  first dense replacement, expert hidden size.
- Multimodal wrappers: `text_config`, `vision_config`, projector metadata,
  image token indices.
- Special side paths: PLE/per-layer input vectors, MTP heads, recurrent state,
  SSM state, chunked attention cache.

## Suggested implementation order

1. Harden multimodal wrapper handling for Llama 3.2 Vision, Llama 4, Qwen2-VL,
   Gemma 3, Pixtral, and Phi-4 multimodal.
2. Tighten existing advanced views: MLA, MoE recursive detail, QK norm, no-RoPE,
   sliding/global layer map.
3. Verify Sarvam 2B/105B and Falcon3 configs, then add aliases or dedicated
   adapters only where the configs prove they are needed.

## Source spot checks

Representative public config/model-card checks used while making this matrix:

- [Qwen3-235B-A22B config](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/4baae73c9c67ae2a578b7a13c27036adc76aed44/config.json):
  `model_type="qwen3_moe"`, 128 experts, top-8, 64 Q heads
  and 4 KV heads.
- [DeepSeek-V3 config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/2dd55ceadbf93fe45fd377b17765ec12e947e6c3/config.json):
  `model_type="deepseek_v3"`, MLA fields (`kv_lora_rank`,
  `q_lora_rank`, RoPE/noPE split), 256 routed experts, top-8, shared expert,
  and MTP.
- [DBRX docs/model card](https://huggingface.co/docs/transformers/model_doc/dbrx)
  and [public config mirror](https://huggingface.co/nicoboss/dbrx-base/blob/main/config.json):
  `model_type="dbrx"`, nested `attn_config` and `ffn_config`, 48 Q heads,
  8 KV heads, 16 experts, top-4.
- [Llama 4 Scout-style config](https://huggingface.co/mlx-community/meta-llama-Llama-4-Scout-17B-16E-fp16/blob/main/config.json):
  multimodal wrapper with `text_config`, 16 local experts, GQA, QK norm,
  long context, and vision config.
- [Gemma 3 27B-style config](https://huggingface.co/callgg/gemma-3-27b-it-bf16/blob/main/config.json):
  wrapper with `text_config` and `vision_config`.
- [Phi-2 config](https://huggingface.co/microsoft/phi-2/blob/main/config.json):
  `model_type="phi"`, MHA, GELU-new, partial rotary.
- [Sarvam-M config](https://huggingface.co/sarvamai/sarvam-m/blob/b5f6ae5ff366c0fc4cb88911e8a1afd21a516c74/config.json):
  public config is `MistralForCausalLM` / `model_type="mistral"`.
