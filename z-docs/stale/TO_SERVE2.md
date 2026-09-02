# The Big Open-Weight Model Atlas — LLMs & Diffusion Models, Grouped by Family

> **What this is:** Every *notable* open-weight **transformer LLM** family and every *notable* open-weight **diffusion (image/video) model** family, grouped by lab/family, with Hugging Face IDs and a one-line architecture note per release.
>
> **What "version" means here:** each official *size* and each official *variant* (Base / Instruct / Chat / Reasoning / Coder / Math / VL, etc.) released by the original org. This is the meaningful "weight version" granularity.
>
> **Deliberately excluded (per your request):** finetunes & community merges of the above, community re-quantizations (GGUF / AWQ / GPTQ / FP8 re-uploads by third parties), and the architectures you named — **State-Space Models (Mamba etc.), JEPA, pure CNNs, and LSTMs/RNNs (incl. RWKV)**. Those are catalogued in the **Excluded Appendix** at the bottom so you can see they weren't *missed*.
>
> **Honesty note:** HF hosts *hundreds of thousands* of repos; a literal "every model on Earth" list is impossible (99%+ are finetunes/quants/merges). This is the exhaustive **base-family** map. Entries dated after **Jan 2026** (my knowledge cutoff) were pulled from live search and are flagged **⚠️verify-ID** where the exact repo string may still be shifting.

---

## LEGEND — architecture one-liner shorthand
- **Dense decoder** = standard autoregressive dense Transformer (GPT-style).
- **MoE** = sparse Mixture-of-Experts decoder (only some experts active per token).
- **Enc-dec** = encoder–decoder Transformer (T5-style).
- **Enc-only** = encoder-only Transformer (BERT-style, non-generative).
- **VLM** = vision-language multimodal Transformer.
- **U-Net LDM** = latent-diffusion with a convolutional U-Net denoiser.
- **DiT / MMDiT** = Diffusion Transformer / Multimodal Diffusion Transformer denoiser.
- **RF** = rectified-flow / flow-matching training objective.
- **Hybrid†** = transformer *plus* a linear-attention/SSM component (kept in-scope because the transformer dominates; flagged so you know).

---

# PART A — LARGE LANGUAGE MODELS (transformer-based)

`| QN: mapping every open-weight transformer-LLM family → its official HF release IDs + arch note`

## A1 · Meta — LLaMA / OPT / Galactica
`| QN: Meta's Llama lineage v1→v4 plus its pre-Llama open decoders`

| HF ID | Architecture |
|---|---|
| `huggyllama/llama-7b` · `-13b` · `-30b` · `-65b` | LLaMA 1 dense decoder (community re-host; Meta didn't post v1 on HF) |
| `meta-llama/Llama-2-7b-hf` · `-13b-hf` · `-70b-hf` (+ `-chat-hf`) | Llama 2 dense decoder (GQA on 70B) |
| `meta-llama/CodeLlama-7b-hf` · `-13b` · `-34b` · `-70b` (+ `-Python`, `-Instruct`) | Code Llama — Llama 2 continued-pretrained on code |
| `meta-llama/Meta-Llama-3-8B` · `-70B` (+ `-Instruct`) | Llama 3 dense decoder, 128k-vocab tokenizer |
| `meta-llama/Llama-3.1-8B` · `-70B` · `-405B` (+ `-Instruct`) | Llama 3.1 dense decoder, 128k context |
| `meta-llama/Llama-3.2-1B` · `-3B` (+ `-Instruct`) | Llama 3.2 small dense decoders |
| `meta-llama/Llama-3.2-11B-Vision` · `-90B-Vision` (+ `-Instruct`) | Llama 3.2 VLM (cross-attn image adapter) |
| `meta-llama/Llama-3.3-70B-Instruct` | Llama 3.3 dense decoder (instruct-only refresh) |
| `meta-llama/Llama-4-Scout-17B-16E` (+ `-Instruct`) | Llama 4 MoE, 17B active / 16 experts, native-multimodal early fusion |
| `meta-llama/Llama-4-Maverick-17B-128E` (+ `-Instruct`) | Llama 4 MoE, 17B active / 128 experts, VLM |
| *Llama 4 Behemoth (288B active / 16E)* | Teacher MoE — **previewed, weights not released** |
| `facebook/opt-125m` … `facebook/opt-66b` | OPT dense decoder (2022 GPT-3 replication) |
| `facebook/galactica-125m` … `-120b` | Galactica dense decoder (science corpus) |
| `facebook/MobileLLM-125M` … `-1.5B` | MobileLLM — deep-thin on-device dense decoder |

## A2 · Mistral AI
`| QN: Mistral's dense 7B line, Mixtral MoEs, and the 2024–25 Small/Nemo/Codestral/Magistral variants`

| HF ID | Architecture |
|---|---|
| `mistralai/Mistral-7B-v0.1` · `-v0.2` · `-v0.3` (+ `-Instruct-*`) | Dense decoder w/ sliding-window attention |
| `mistralai/Mixtral-8x7B-v0.1` (+ `-Instruct`) | Sparse MoE, 8 experts, 2 active |
| `mistralai/Mixtral-8x22B-v0.1` (+ `-Instruct`) | Sparse MoE, larger |
| `mistralai/Mistral-Nemo-Base-2407` (+ `-Instruct-2407`) | 12B dense decoder (w/ NVIDIA) |
| `mistralai/Mistral-Small-24B-Base-2501` (+ `-Instruct-2501`) | 24B dense decoder |
| `mistralai/Mistral-Small-3.1-24B-Base-2503` (+ `-Instruct`) | 24B dense VLM |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24B dense refresh |
| `mistralai/Codestral-22B-v0.1` | Dense code decoder |
| `mistralai/Mathstral-7B-v0.1` | Dense math decoder |
| `mistralai/Ministral-8B-Instruct-2410` | Dense edge decoder |
| `mistralai/Pixtral-12B-2409` · `Pixtral-Large-Instruct-2411` | VLM (vision encoder + Mistral decoder) |
| `mistralai/Mistral-Large-Instruct-2407` · `-2411` | 123B dense decoder (research license) |
| `mistralai/Magistral-Small-2506` (+ later) | Dense reasoning decoder |
| `mistralai/Devstral-Small-2505` | Dense agentic-coding decoder |
| `mistralai/Mistral-Small-3` / Mistral 3 Large (2026) ⚠️verify-ID | Adopted DeepSeek-V3-style coarse MoE |

## A3 · Alibaba — Qwen
`| QN: the whole Qwen tree v1→v3.5, all dense+MoE sizes and the Coder/Math/VL/Omni specialists`

| HF ID | Architecture |
|---|---|
| `Qwen/Qwen-1_8B` · `-7B` · `-14B` · `-72B` (+ `-Chat`) | Qwen 1 dense decoder |
| `Qwen/Qwen1.5-0.5B` … `-110B` (+ `-Chat`) | Qwen 1.5 dense decoder |
| `Qwen/Qwen1.5-MoE-A2.7B` | First Qwen MoE |
| `Qwen/Qwen2-0.5B` · `-1.5B` · `-7B` · `-72B` | Qwen 2 dense decoder |
| `Qwen/Qwen2-57B-A14B` | Qwen 2 MoE |
| `Qwen/Qwen2.5-0.5B` · `-1.5B` · `-3B` · `-7B` · `-14B` · `-32B` · `-72B` (+ `-Instruct`) | Qwen 2.5 dense decoder |
| `Qwen/Qwen2.5-Coder-*` · `Qwen2.5-Math-*` · `Qwen2.5-VL-*` · `Qwen2.5-1M` · `Qwen2.5-Omni-7B` | Coder / Math / VLM / long-ctx / any-to-any specialists |
| `Qwen/QwQ-32B` · `Qwen/QVQ-72B-Preview` | Dense reasoning / visual-reasoning decoders |
| `Qwen/Qwen3-0.6B` · `-1.7B` · `-4B` · `-8B` · `-14B` · `-32B` | Qwen 3 dense decoder, hybrid think/non-think |
| `Qwen/Qwen3-30B-A3B` · `Qwen/Qwen3-235B-A22B` (+ `-Instruct-2507`, `-Thinking-2507`) | Qwen 3 MoE (128 experts, 8 active) |
| `Qwen/Qwen3-Coder-*` · `Qwen3-VL-*` · `Qwen3-Omni-*` · `Qwen3-Next-*` (Gated-DeltaNet **Hybrid†**) | Qwen 3 specialists |
| `Qwen/Qwen3.5-*` (Feb 2026) ⚠️verify-ID | Qwen 3.5 open-weights; uses Gated DeltaNet linear-attn **Hybrid†** |

## A4 · DeepSeek
`| QN: DeepSeek's dense v1, the V2/V3/V4 MoE line, R1 reasoning, and Coder/Math/VL specialists`

| HF ID | Architecture |
|---|---|
| `deepseek-ai/deepseek-llm-7b-base` · `-67b-base` (+ `-chat`) | Dense decoder |
| `deepseek-ai/deepseek-moe-16b-base` (+ `-chat`) | Early fine-grained MoE |
| `deepseek-ai/DeepSeek-V2` · `DeepSeek-V2-Lite` · `DeepSeek-V2.5` | MoE w/ Multi-head Latent Attention (MLA) |
| `deepseek-ai/DeepSeek-V3` · `-V3-0324` · `-V3.1` · `-V3.2-Exp` · `-V3.2` | 671B MoE / 37B active, MLA + aux-loss-free routing (V3.2 adds Sparse Attention) |
| `deepseek-ai/DeepSeek-R1` · `-R1-Zero` · `-R1-0528` | RL-reasoning MoE (built on V3 base) |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-*` · `-Llama-*` | Dense distills (reasoning traces → Qwen/Llama) |
| `deepseek-ai/deepseek-coder-1.3b/6.7b/33b-base` (+ `-instruct`) · `DeepSeek-Coder-V2` | Code decoder / code MoE |
| `deepseek-ai/deepseek-math-7b-base` (+ `-instruct`/`-rl`) | Math dense decoder |
| `deepseek-ai/deepseek-vl-*` · `deepseek-ai/deepseek-vl2` | VLM (VL2 is MoE) |
| `deepseek-ai/DeepSeek-V4-Pro` · `-V4-Flash` (2026) ⚠️verify-ID | New arch: 1.6T/49B (Pro) & 284B/13B (Flash) MoE, hybrid compressed attention |

## A5 · Google — BERT / T5 / Gemma
`| QN: Google's open encoders, enc-dec T5 family, and the Gemma 1→4 decoders incl. the e4b question`

| HF ID | Architecture |
|---|---|
| `google-bert/bert-base-uncased` (+ large, multilingual, `roberta`-adjacent) | Enc-only (non-generative; borderline "LLM") |
| `google-t5/t5-small` … `-11b` · `google/flan-t5-*` · `google/mt5-*` · `google/byt5-*` | Enc-dec |
| `google/ul2` · `google/flan-ul2` | Enc-dec (mixture-of-denoisers) |
| `google/switch-base-*` · `switch-c-2048` | Enc-dec **MoE** (Switch Transformer) |
| `google/gemma-2b` · `-7b` (+ `-it`) | Gemma 1 dense decoder |
| `google/gemma-2-2b` · `-9b` · `-27b` (+ `-it`) | Gemma 2 dense decoder (local/global attn) |
| `google/gemma-3-1b-it` · `-4b` · `-12b` · `-27b` (+ `-pt`) | Gemma 3 dense VLM |
| `google/gemma-3n-E2B` · `google/gemma-3n-E4B` | Gemma 3n — "effective" 2B/4B via per-layer embeddings (mobile) |
| `google/codegemma-*` · `google/paligemma-*` · `google/paligemma2-*` | Code decoder / VLM |
| `google/shieldgemma-*` · `google/datagemma-*` | Safety / retrieval-grounded variants |
| **Gemma 4** — e.g. `google/gemma-4-*`, incl. **`gemma-4-e4b`** & Gemma-4-31B (2026) ⚠️verify-ID | Yes, Gemma 4 exists; `e4b` = the ~4B *effective-param* mobile variant (same E2B/E4B naming Gemma 3n used). Exact repo string not yet in my training data — **confirm on huggingface.co/google before scripting against it.** |

*RecurrentGemma (`google/recurrentgemma-2b`/`-9b`) uses the **Griffin** linear-recurrent hybrid — see Excluded Appendix.*

## A6 · Microsoft — Phi
`| QN: the Phi "small-but-strong" line v1→v4 plus older DialoGPT/Orca`

| HF ID | Architecture |
|---|---|
| `microsoft/phi-1` · `microsoft/phi-1_5` · `microsoft/phi-2` | Dense decoder (textbook-quality data) |
| `microsoft/Phi-3-mini-4k-instruct` (+ `-128k`) · `-small-*` · `-medium-*` | Dense decoder |
| `microsoft/Phi-3.5-mini-instruct` · `Phi-3.5-MoE-instruct` · `Phi-3.5-vision-instruct` | Dense / MoE / VLM |
| `microsoft/phi-4` · `Phi-4-mini-instruct` · `Phi-4-multimodal-instruct` | Dense decoder / VLM |
| `microsoft/Phi-4-reasoning` · `-reasoning-plus` · `-mini-reasoning` | Dense reasoning decoder |
| `microsoft/DialoGPT-*` · `microsoft/Orca-2-7b`/`-13b` | Legacy dialogue / reasoning-distill dense |

## A7 · EleutherAI
`| QN: the early fully-open GPT replications that seeded the ecosystem`

| HF ID | Architecture |
|---|---|
| `EleutherAI/gpt-neo-125M` · `-1.3B` · `-2.7B` | Dense decoder |
| `EleutherAI/gpt-j-6b` | Dense decoder (parallel attn/FF) |
| `EleutherAI/gpt-neox-20b` | Dense decoder |
| `EleutherAI/pythia-14m` … `-12b` (+ `-deduped`) | Dense decoder suite (interpretability) |

## A8 · BigScience
`| QN: the multilingual community-trained BLOOM family`

| HF ID | Architecture |
|---|---|
| `bigscience/bloom` (176B) · `bloom-560m` … `-7b1` | Dense decoder, 46 languages (ALiBi) |
| `bigscience/bloomz-*` · `bigscience/mt0-*` | Multitask-finetuned (base arch same) |

## A9 · TII (Abu Dhabi) — Falcon
`| QN: Falcon dense giants + the Falcon 3 refresh (Mamba/H1 hybrids noted separately)`

| HF ID | Architecture |
|---|---|
| `tiiuae/falcon-7b` · `-40b` · `-180B` (+ `-instruct`) | Dense decoder (multiquery attn) |
| `tiiuae/falcon-11B` (Falcon 2) | Dense decoder |
| `tiiuae/Falcon3-1B/3B/7B/10B-Base` (+ `-Instruct`) | Falcon 3 dense decoder |
| `tiiuae/Falcon-H1-*` ⚠️ | **Hybrid†** attention+SSM (see Appendix) |
| `tiiuae/falcon-mamba-7b` | Pure SSM — **excluded** (see Appendix) |

## A10 · 01.AI — Yi
`| QN: the Yi bilingual dense line + Coder/VL specialists`

| HF ID | Architecture |
|---|---|
| `01-ai/Yi-6B` · `-9B` · `-34B` (+ `-Chat`, `-200K`) | Dense decoder (Llama-arch) |
| `01-ai/Yi-1.5-6B` · `-9B` · `-34B` (+ `-Chat`) | Dense decoder refresh |
| `01-ai/Yi-Coder-1.5B`/`-9B` · `01-ai/Yi-VL-*` | Code decoder / VLM |

## A11 · Cohere (Cohere Labs / C4AI)
`| QN: Command-R RAG/agent models + the multilingual Aya family`

| HF ID | Architecture |
|---|---|
| `CohereForAI/c4ai-command-r-v01` (35B) · `c4ai-command-r-plus` (104B) | Dense decoder, RAG/tool-tuned |
| `CohereForAI/c4ai-command-r-08-2024` · `-plus-08-2024` · `c4ai-command-r7b-12-2024` | Dense decoder refresh |
| `CohereLabs/c4ai-command-a-03-2025` | 111B dense decoder (agentic flagship) |
| `CohereForAI/aya-101` · `aya-23-8B`/`-35B` · `aya-expanse-*` · `aya-vision-*` | Multilingual dense / VLM |

## A12 · Databricks
`| QN: Dolly (early instruct) and the DBRX MoE`

| HF ID | Architecture |
|---|---|
| `databricks/dolly-v2-3b` · `-7b` · `-12b` | Dense decoder (Pythia-based) |
| `databricks/dbrx-base` · `dbrx-instruct` | Fine-grained MoE (16 experts, 4 active) |

## A13 · Stability AI (text)
`| QN: the StableLM decoders and Stable Code`

| HF ID | Architecture |
|---|---|
| `stabilityai/stablelm-base-alpha-*` · `stablelm-tuned-alpha-*` | Dense decoder |
| `stabilityai/stablelm-3b-4e1t` · `stablelm-2-1_6b` · `stablelm-2-12b` (+ `-zephyr`,`-chat`) | Dense decoder |
| `stabilityai/stable-code-3b` · `stable-code-instruct-3b` | Code dense decoder |

## A14 · MosaicML
`| QN: the MPT commercial-friendly decoders`

| HF ID | Architecture |
|---|---|
| `mosaicml/mpt-7b` · `mpt-30b` (+ `-instruct`, `-chat`, `-storywriter`) | Dense decoder (ALiBi, FlashAttn) |

## A15 · Together AI
`| QN: the early open RedPajama replication`

| HF ID | Architecture |
|---|---|
| `togethercomputer/RedPajama-INCITE-7B-Base` · `-3B-Base` (+ `-Instruct`,`-Chat`) | Dense decoder |

## A16 · NVIDIA — Nemotron / Minitron
`| QN: Nemotron v3/v4, the Llama-Nemotron reasoning distills, Minitron pruning, and the v3 hybrid MoEs`

| HF ID | Architecture |
|---|---|
| `nvidia/nemotron-3-8b-base-4k` (+ chat) | Dense decoder (enterprise) |
| `nvidia/Nemotron-4-340B-Base` · `-Instruct` · `-Reward` | Dense decoder (synthetic-data generator) |
| `nvidia/Minitron-4B-Base` · `-8B-Base` · `nvidia/Nemotron-4-Minitron-*` | Pruned+distilled dense decoder |
| `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` | Llama-3.1 reasoning-tuned dense |
| `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` · `Llama-3_3-Nemotron-Super-49B-v1(.5)` · `Nemotron-Nano-8B` | NAS-optimized dense reasoning decoders |
| `nvidia/Nemotron-H-*` | **Hybrid†** Mamba-2 + Transformer |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` · `-Super-120B-A12B` · *Ultra-550B-A55B* (2026) ⚠️verify-ID | **Hybrid†** LatentMoE: interleaved Mamba-2 + MoE + few attention layers |

## A17 · xAI
`| QN: Grok open-weight drops`

| HF ID | Architecture |
|---|---|
| `xai-org/grok-1` | 314B MoE (8 experts, 2 active) |
| Grok-2 open-weights (2025) ⚠️verify-ID | MoE (released later as open weights) |

## A18 · OpenAI (open weights)
`| QN: the two open OpenAI decoders — GPT-2 and the 2025 gpt-oss MoEs`

| HF ID | Architecture |
|---|---|
| `openai-community/gpt2` · `-medium` · `-large` · `-xl` | Dense decoder (the original) |
| `openai/gpt-oss-120b` · `openai/gpt-oss-20b` | MoE reasoning decoder, MXFP4 experts, attention sinks |
| `openai/gpt-oss-safeguard-120b` · `-20b` | Safety-classifier variant (built on gpt-oss) |

## A19 · BigCode
`| QN: the open code-LLM line (Santa/Star/StarCoder2)`

| HF ID | Architecture |
|---|---|
| `bigcode/santacoder` · `bigcode/starcoderbase` · `bigcode/starcoder` | Code dense decoder (MQA, FIM) |
| `bigcode/starcoder2-3b` · `-7b` · `-15b` | Code dense decoder (GQA, sliding window) |

## A20 · Salesforce
`| QN: the CodeGen / CodeT5 code-model line`

| HF ID | Architecture |
|---|---|
| `Salesforce/codegen-350M-*` … `-16B-*` · `codegen2-*` · `codegen25-*` | Code dense decoder |
| `Salesforce/codet5-*` · `Salesforce/codet5p-*` | Code enc-dec |
| `Salesforce/xgen-7b-*` | Dense decoder (long ctx) |

## A21 · Allen AI (Ai2) — OLMo
`| QN: the fully-open (data+code+weights) OLMo line + OLMoE + Molmo VLM`

| HF ID | Architecture |
|---|---|
| `allenai/OLMo-1B` · `OLMo-7B` (+ `-0424`,`-0724`,`-Instruct`) | Fully-open dense decoder |
| `allenai/OLMo-2-1124-7B` · `-13B` · `OLMo-2-0325-32B` · `OLMo-2-0425-1B` (+ `-Instruct`) | OLMo 2 dense decoder |
| `allenai/OLMoE-1B-7B-0924` (+ `-Instruct`) | Fully-open **MoE** |
| `allenai/Molmo-7B-D` · `-7B-O` · `-72B` · `MolmoE-1B` | VLM (open) |
| `allenai/Llama-3.1-Tulu-3-*` | Post-training recipe on Llama (open) |

## A22 · Baichuan
`| QN: the Baichuan bilingual dense line`

| HF ID | Architecture |
|---|---|
| `baichuan-inc/Baichuan-7B` · `Baichuan-13B-Base` (+ `-Chat`) | Dense decoder |
| `baichuan-inc/Baichuan2-7B-Base` · `-13B-Base` (+ `-Chat`) | Dense decoder |

## A23 · InternLM (Shanghai AI Lab)
`| QN: InternLM decoders v1→v3 + InternVL multimodal`

| HF ID | Architecture |
|---|---|
| `internlm/internlm-7b` · `-20b` | Dense decoder |
| `internlm/internlm2-7b` · `-20b` · `internlm/internlm2_5-1_8b`/`-7b`/`-20b` | Dense decoder |
| `internlm/internlm3-8b-instruct` | Dense decoder |
| `OpenGVLab/InternVL-*` · `InternVL2-*` · `InternVL2_5-*` · `InternVL3-*` | VLM |

## A24 · Zhipu AI / THUDM — GLM
`| QN: ChatGLM → GLM-4 → the GLM-4.5/4.6/5 MoE flagships + CogVLM/CodeGeeX`

| HF ID | Architecture |
|---|---|
| `THUDM/chatglm-6b` · `chatglm2-6b` · `chatglm3-6b` | Prefix-LM / GLM dense |
| `THUDM/glm-4-9b` · `glm-4-9b-chat` · `glm-4v-9b` | Dense decoder / VLM |
| `THUDM/GLM-4-32B-0414` · `GLM-Z1-32B-0414` | Dense decoder / reasoning |
| `zai-org/GLM-4.5` · `GLM-4.5-Air` · `GLM-4.6` (+ `-V`) | MoE (MLA); flagship agent/coding |
| `zai-org/GLM-5` · `GLM-5.1` / `GLM-5.2` (2026) ⚠️verify-ID | ~744B MoE, MLA + Sparse Attention |
| `THUDM/cogvlm-*` · `cogagent-*` · `THUDM/codegeex4-*` | VLM / GUI agent / code |

## A25 · MiniMax
`| QN: the MiniMax lightning-attention hybrid flagships`

| HF ID | Architecture |
|---|---|
| `MiniMaxAI/MiniMax-Text-01` · `MiniMax-VL-01` | **Hybrid†** lightning (linear) + softmax attn MoE |
| `MiniMaxAI/MiniMax-M1` · `MiniMax-M2` (+ M2.x 2026 ⚠️verify-ID) | MoE reasoning/agentic (M2.5 reverted to plain GQA) |

## A26 · Moonshot AI — Kimi
`| QN: the Kimi K2 trillion-param MoE line + Moonlight + Kimi-VL`

| HF ID | Architecture |
|---|---|
| `moonshotai/Moonlight-16B-A3B` (+ `-Instruct`) | MoE (Muon-trained) |
| `moonshotai/Kimi-K2-Base` · `Kimi-K2-Instruct` | 1T MoE / 32B active (MLA) |
| `moonshotai/Kimi-K2-Thinking` · `Kimi-K2.5` · `Kimi-K2.6` (2026) ⚠️verify-ID | MoE reasoning + MoonViT vision |
| `moonshotai/Kimi-VL-A3B-*` | MoE VLM |

## A27 · Tencent — Hunyuan (LLM)
`| QN: Hunyuan's open MoE + dense decoders (image/video are in Part B)`

| HF ID | Architecture |
|---|---|
| `tencent/Hunyuan-A52B-Instruct` (Hunyuan-Large) | MoE, 52B active |
| `tencent/Hunyuan-7B-Instruct` · `tencent/Hunyuan-A13B-Instruct` | Dense / MoE decoder |

## A28 · Apple
`| QN: Apple's open on-device decoders`

| HF ID | Architecture |
|---|---|
| `apple/OpenELM-270M` · `-450M` · `-1_1B` · `-3B` (+ `-Instruct`) | Layer-wise-scaled dense decoder |
| `apple/DCLM-7B` · `apple/DCLM-Baseline-7B` | Dense decoder (data-curation benchmark) |

## A29 · IBM — Granite
`| QN: Granite 3.x dense+MoE, Granite-Code, and the Granite 4 Mamba hybrid`

| HF ID | Architecture |
|---|---|
| `ibm-granite/granite-3.0-2b-base`/`-8b-base` (+ `.1`,`.2`,`.3`, `-instruct`) | Dense decoder |
| `ibm-granite/granite-3.0-1b-a400m` · `-3b-a800m` | Small **MoE** |
| `ibm-granite/granite-3b-code-base` · `-8b`/`-20b`/`-34b-code` | Code dense decoder |
| `ibm-granite/granite-guardian-*` · `granite-vision-*` | Safety / VLM |
| `ibm-granite/granite-4.0-*` (2025) | **Hybrid†** Mamba-2 + Transformer MoE |

## A30 · Snowflake · Upstage · LG · others (Western/regional)
`| QN: the remaining notable non-Chinese labs — Arctic, SOLAR, EXAONE, SmolLM, and friends`

| HF ID | Architecture |
|---|---|
| `Snowflake/snowflake-arctic-base` · `snowflake-arctic-instruct` | Dense+MoE hybrid-parallel |
| `upstage/SOLAR-10.7B-v1.0` (+ `-Instruct`) | Dense decoder (depth-up-scaling) |
| `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct` · `EXAONE-3.5-2.4B`/`-7.8B`/`-32B` · `EXAONE-Deep-*` · `EXAONE-4.0-*` | Dense decoder (KR/EN) |
| `HuggingFaceTB/SmolLM-135M`/`-360M`/`-1.7B` · `SmolLM2-*` · `SmolLM3-3B` | Small dense decoder (fully open) |
| `h2oai/h2o-danube3-*` · `h2o-danube2-*` | Small dense decoder |
| `Nexusflow/`, `NousResearch/` Hermes | Mostly *finetunes* → out of scope (noted) |
| `sarvamai/sarvam-1` · `sarvam-m` | Indic dense decoder |
| `arcee-ai/AFM-4.5B` · Arcee "Trinity" (2026) ⚠️verify-ID | Dense / DeepSeek-style MoE |
| `AMD-OLMo-*` · `amd/Instella-3B` | Fully-open dense decoder (AMD) |
| `utter-project/EuroLLM-*` · `occiglot/*` · `openGPT-X/Teuken-7B-*` | European multilingual dense |
| `Aleph-Alpha/Pharia-1-LLM-7B-*` | Dense decoder (EU) |
| `LumiOpen/Poro-34B` · `Viking-*` | Nordic multilingual dense |

## A31 · Other Chinese labs (long tail but notable)
`| QN: the broader Chinese open-weight ecosystem beyond Qwen/DeepSeek/GLM/Kimi`

| HF ID | Architecture |
|---|---|
| `baidu/ERNIE-4.5-0.3B` … `-21B-A3B` … `-300B-A47B` (+ `-Base`, `-VL`, `-Paddle`/`-PT`) | Dense + MoE (some VLM); Apache-2.0 (2025) |
| `inclusionAI/Ling-lite`/`-plus` · `Ling-2.x` · `Bailing-*` (Ant Group) | MoE (Ling-2.5 uses linear-attn **Hybrid†**) |
| `XiaomiMiMo/MiMo-7B-*` · MiMo-V2.5-Pro (2026) ⚠️verify-ID | Dense / large MoE reasoning |
| `stepfun-ai/step-*` · Step-3.5 Flash (2026) ⚠️verify-ID | MoE (MTP) |
| `Skywork/Skywork-13B-*` · `Skywork-MoE-*` · `Skywork-OR1-*` | Dense / MoE / reasoning |
| `IEITYuan/Yuan2-*` | Dense decoder |
| `OrionStarAI/Orion-14B-*` · `xverse/XVERSE-*` | Dense decoder |
| `TeleAI/TeleChat2-*` (China Telecom) | Dense / MoE |
| `Nanbeige/Nanbeige-*` (+ 4.1 2026) | Dense / MoE (plain GQA) |
| `rednote-hilab/dots.llm1` (Xiaohongshu) | MoE |
| `openPangu-*` (Huawei) ⚠️verify-ID | Dense / MoE |
| `internlm/`, `Tencent/Hunyuan-MT-*` (translation) | specialist decoders |


---

# PART B — DIFFUSION / GENERATIVE VISUAL MODELS

`| QN: mapping every open-weight diffusion (image, video, audio) family → its official HF release IDs + denoiser type`

## B1 · Image — Stability AI (Stable Diffusion lineage)
`| QN: the whole SD tree from v1 U-Nets through the SD3.5 MMDiTs + Cascade`

| HF ID | Architecture |
|---|---|
| `CompVis/stable-diffusion-v1-1` … `-v1-4` | U-Net LDM (CLIP text enc) |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` (was `runwayml/…`) | U-Net LDM |
| `stabilityai/stable-diffusion-2` · `-2-1` · `-2-base` | U-Net LDM (OpenCLIP) |
| `stabilityai/stable-diffusion-xl-base-1.0` · `-refiner-1.0` | U-Net LDM (dual text enc, larger) |
| `stabilityai/sdxl-turbo` | Distilled SDXL (adversarial, 1–4 step) |
| `stabilityai/stable-cascade` | Würstchen-v3 cascaded latent (small latent space) |
| `stabilityai/stable-diffusion-3-medium` | MMDiT + RF |
| `stabilityai/stable-diffusion-3.5-large` · `-large-turbo` · `-medium` | MMDiT + RF |

## B2 · Image — Black Forest Labs (FLUX)
`| QN: the FLUX rectified-flow DiT family incl. Kontext/Krea and FLUX.2`

| HF ID | Architecture |
|---|---|
| `black-forest-labs/FLUX.1-dev` · `FLUX.1-schnell` | RF DiT (12B); schnell = timestep-distilled |
| `black-forest-labs/FLUX.1-Kontext-dev` | RF DiT, in-context image editing |
| `black-forest-labs/FLUX.1-Krea-dev` | RF DiT, photographic-aesthetic tune |
| `black-forest-labs/FLUX.1-Fill-dev` · `-Canny-dev` · `-Depth-dev` · `-Redux-dev` | Inpaint / ControlNet-style / variation |
| `black-forest-labs/FLUX.2-dev` · `FLUX.2-klein` (2026) ⚠️verify-ID | Next-gen RF DiT (klein = small ~4B) |

## B3 · Image — other major families
`| QN: every other notable open image generator — DeepFloyd, Kandinsky, PixArt, Sana, HiDream, Lumina, Kolors, OmniGen, Qwen-Image, HunyuanImage, Z-Image, CogView, plus SDXL distills`

| HF ID | Architecture |
|---|---|
| `DeepFloyd/IF-I-XL-v1.0` (+ `IF-II`, `IF-III`) | Pixel-space **cascaded** diffusion (T5 text) |
| `kandinsky-community/kandinsky-2-1` · `-2-2` · `kandinsky-3` | U-Net LDM (image-prior) |
| `ai-forever/Kandinsky-5.0-*` (2025/26) ⚠️verify-ID | DiT (latest Kandinsky) |
| `playgroundai/playground-v2-1024px-aesthetic` · `playground-v2.5-1024px-aesthetic` | SDXL-arch U-Net (aesthetic-tuned) |
| `PixArt-alpha/PixArt-XL-2-1024-MS` · `PixArt-alpha/PixArt-Sigma-XL-2-*` | DiT (T5 text enc) |
| `Efficient-Large-Model/Sana_1600M_1024px` (+ `Sana_600M`, Sana-1.5) | Linear-attention DiT (deep-compression AE), NVIDIA |
| `HiDream-ai/HiDream-I1-Full` · `-Dev` · `-Fast` | Sparse **MoE DiT** (17B) |
| `Alpha-VLLM/Lumina-Next-SFT` · `Alpha-VLLM/Lumina-Image-2.0` · `Lumina-T2X` | Flow-based DiT (Next-DiT) |
| `Kwai-Kolors/Kolors` | U-Net LDM (GLM text enc, bilingual) |
| `Shitao/OmniGen-v1` · `OmniGen2/OmniGen2` | Unified any-to-image diffusion transformer |
| `Qwen/Qwen-Image` · `Qwen/Qwen-Image-Edit`(`-2509`) · `Qwen-Image-2.0`/`-2512` ⚠️verify-ID | 20B MMDiT (text rendering); Edit = instruction editing |
| `tencent/HunyuanDiT` (`Tencent-Hunyuan/HunyuanDiT`) | DiT (bilingual, cross-attn) |
| `tencent/HunyuanImage-2.1` · `HunyuanImage-3.0` (2025) | 17B DiT / **80B MoE** autoregressive-multimodal image model |
| `Tongyi-MAI/Z-Image-Turbo` (2025) ⚠️verify-ID | Compact ~6B DiT, few-step, bilingual |
| `THUDM/CogView3-Plus-3B` · `THUDM/CogView4-6B` | DiT (T2I) |
| `segmind/SSD-1B` | Distilled/pruned SDXL U-Net |
| `warp-ai/wuerstchen` | Cascaded latent (Würstchen v2) |
| `ByteDance/SDXL-Lightning` · `ByteDance/Hyper-SD` | Step-distilled SDXL adapters |
| `SimianLuo/LCM_Dreamshaper_v7` (+ `latent-consistency/*`) | Latent Consistency Model (few-step distill) |
| `MeissonFlow/Meissonic` | Masked-token (non-autoregressive) T2I |
| `zai-org/GLM-Image` (2026) ⚠️verify-ID | DiT (strong typography) |

## B4 · Video — open text/image-to-video
`| QN: every notable open video generator — Wan, HunyuanVideo, CogVideoX, Mochi, LTX, Cosmos, Open-Sora, SVD, AnimateDiff, and more`

| HF ID | Architecture |
|---|---|
| `Wan-AI/Wan2.1-T2V-1.3B` · `-14B` · `Wan2.1-I2V-*` | DiT video (3D-VAE), Alibaba |
| `Wan-AI/Wan2.2-T2V-A14B` · `-I2V-A14B` · `Wan2.2-TI2V-5B` · `-S2V-14B` · `-Animate-14B` | **MoE DiT** video (A14B) + dense 5B |
| `tencent/HunyuanVideo` · `HunyuanVideo-I2V` · `HunyuanVideo-1.5` | 13B DiT video (dual→single stream, LLM text enc) |
| `THUDM/CogVideoX-2b` · `-5b` · `CogVideoX1.5-5B` (+ `-I2V`) | Expert-Transformer DiT + 3D-VAE |
| `genmo/mochi-1-preview` | 10B Asymmetric DiT (AsymmDiT), Apache-2.0 |
| `Lightricks/LTX-Video` (+ LTX-2 / LTX-2.3 2026 ⚠️verify-ID) | Fast DiT video (real-time; LTX-2 adds synced audio) |
| `nvidia/Cosmos-1.0-Diffusion-*` · `nvidia/Cosmos-Predict2-*` | World-model DiT (Physical AI) |
| `hpcai-tech/Open-Sora-v2` (+ Open-Sora 1.x) | Open DiT video (full training pipeline) |
| `Skywork/SkyReels-V1` · `SkyReels-V2-*` | Human-centric / infinite-length DiT video |
| `rhymes-ai/Allegro` | DiT video |
| `stabilityai/stable-video-diffusion-img2vid-xt` (+ `-1-1`) | U-Net video diffusion (image-to-video) |
| `guoyww/animatediff-motion-adapter-*` | Motion module over SD U-Net |
| `damo-vilab/text-to-video-ms-1.7b` | Early U-Net T2V (ModelScope) |
| `rain1011/pyramid-flow-*` | Pyramidal-flow DiT video |
| `stepfun-ai/stepvideo-t2v` | 30B DiT video |

## B5 · Audio / music diffusion (bonus — same generative family)
`| QN: the open diffusion-based audio generators, in case "diffusion models" includes sound`

| HF ID | Architecture |
|---|---|
| `stabilityai/stable-audio-open-1.0` (+ `-small`) | Latent audio diffusion (DiT) |
| `facebook/musicgen-*` · `facebook/audiogen-*` | *Autoregressive* (not diffusion) — noted for completeness |


---

# APPENDIX — EXCLUDED ARCHITECTURES (listed so you know they weren't missed)

`| QN: the families I deliberately left OUT of Parts A/B because they're state-space, RNN/linear, JEPA, CNN, or LSTM — per your exclusion rules`

## X1 · State-Space Models (SSM / Mamba) — excluded
| HF ID | Note |
|---|---|
| `state-spaces/mamba-130m` … `-2.8b` · `mamba2-*` | Pure selective SSM |
| `mistralai/Mamba-Codestral-7B-v0.1` | Codestral Mamba (SSM) |
| `tiiuae/falcon-mamba-7b` | Falcon Mamba (SSM) |
| `Zyphra/Zamba2-*` · `Zamba-7B` | Mostly-SSM hybrid |
| `nvidia/Hymba-1.5B-*` | Attn+SSM hybrid (SSM-heavy) |
| `ibm-ai-platform/Bamba-*` | Mamba2 hybrid |

## X2 · RNN / linear-attention (RWKV & kin) — excluded
| HF ID | Note |
|---|---|
| `RWKV/rwkv-4-*` · `rwkv-5-*` (Eagle) · `rwkv-6-*` (Finch) · RWKV-7 (Goose) | Attention-free RNN / linear |
| `fla-hub/*` (RWKV/GLA/RetNet variants) | Linear-attention RNNs |

## X3 · JEPA — excluded
| HF ID | Note |
|---|---|
| `facebook/ijepa_*` | Image JEPA (self-supervised, non-generative) |
| `facebook/vjepa2-*` | Video JEPA 2 (world model) |

## X4 · CNN & LSTM — excluded
Classic ConvNets (ResNet, EfficientNet, ConvNeXt) and LSTM/GRU seq models (incl. xLSTM `NX-AI/xLSTM-7b`) are out of scope by your rules.

## X5 · Hybrids kept IN-scope but flagged
These appear in Parts A/B with a **Hybrid†** tag because a Transformer still does most of the work, but they contain linear-attn/SSM blocks: **Nemotron-H / Nemotron-3**, **IBM Granite 4**, **MiniMax-01/M-series**, **Qwen3-Next / Qwen3.5** (Gated DeltaNet), **Falcon-H1**, **Ling-2.x**. `ai21labs/AI21-Jamba-*` is a Mamba-Transformer hybrid — borderline; I list it here rather than in-scope since ~half its blocks are SSM.

---

# HOW TO USE THIS / CAVEATS
`| QN: telling you exactly what's solid vs. what to double-check before you rely on an ID`

1. **IDs without ⚠️ are ones I'm confident are the correct canonical repo strings** (established families, HF org conventions I know).
2. **⚠️verify-ID = released after my Jan-2026 cutoff**, pulled from live web search. The *model exists*, but confirm the exact repo path on `huggingface.co` before scripting (orgs sometimes rename, e.g. `THUDM` → `zai-org`, `CohereForAI` → `CohereLabs`).
3. **"All weight versions"** = I listed every official *size* + *variant* (Base/Instruct/Chat/Reasoning/Coder/Math/VL/Turbo/distilled). I did **not** list third-party GGUF/AWQ/GPTQ/FP8 re-uploads or finetunes — those number in the hundreds of thousands and you asked to exclude them.
4. **Your `gemma-4-e4b` question:** yes — Gemma 4 shipped in 2026 and uses the same `E2B`/`E4B` "effective-parameter" naming Google introduced with Gemma 3n, so an `E4B` (~4B-effective mobile) variant is expected. Treat the exact repo string as ⚠️verify-ID until you see it live under `google/`.
5. Fastest way to enumerate a single org's *current* full list: the HF org page (e.g. `huggingface.co/Qwen`, `/mistralai`, `/black-forest-labs`) or the API `https://huggingface.co/api/models?author=<org>`.

*Compiled from model knowledge through Jan 2026 + live search on ~7 verification queries (Jul 2026). Family count ≈ 60+ LLM families and ≈ 30+ diffusion families.*