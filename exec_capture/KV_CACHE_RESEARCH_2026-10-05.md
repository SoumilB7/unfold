# KV cache: does the execution route detect it for every family? (2026-10-05)

Question (Soumil): "will this open cache for every one of them". Research only, no exec_to_ir change yet.

## Method
32 families (smallest runnable repo per family from results_v2), two tools in harness_v2/:
- `cache_probe.py`: one run with use_cache=True; what is stored, who produced it.
- `cache_probe2.py`: two-step decode. Step 1 prefill → cache. Step 2 = one new token with that cache fed
  back; every stored tensor is a named input, and each step-2 attention lists which stored slots reach its
  score / weights·V matmul through shape/concat ops only. That is the predicate "the cache is read back",
  not a proxy. Rotation evidence = cos/sin ops between the projection and the stored key.

## Ground truth (probe2)
| group | families | what the run shows |
|---|---|---|
| plain decoder | Llama-3.2, Qwen3, OLMo-2, Cohere, Pythia, GPT-2, Falcon-7B, BLOOM, OPT | every layer writes K,V and reads its own back |
| sliding window | Phi-3, e5-Mistral, Gemma-2, TranslateGemma | stored = slice of the concat attention read; step 2 reads it back |
| KV sharing | Gemma-3n E2B (30 layers, 20 caches) | layers 20–29 write nothing; they read layer 18 (sliding) / 19 (full) cache |
| MoE | Qwen1.5-MoE, Mixtral, GPT-OSS, DeepSeek-V2-Lite, DeepSeek-R1 | as plain decoder |
| MLA | DeepSeek-V2-Lite, R1 | HF eager caches DECOMPRESSED per-head K (h×192) / V (h×128), not the latent |
| enc-dec | T5, BART | two caches: self (grows) + cross; step-2 cross-attention runs only q,o (K/V reused) |
| hybrid | Qwen3-Next, Jamba, Zamba, Nemotron-H, Falcon-H1, Granite-hybrid, LFM2 | K/V only on attention layers; conv/recurrent states never read by attention |
| no KV cache | BERT (encoder), Mamba (cache_params, not past_key_values) | nothing returned |
| bidirectional | EmbeddingGemma | cache mechanically read back, but mask is bidirectional → a decode cache is meaningless |

KV cache written and read back: 30/32; none for BERT and Mamba (correct).

## What exec_to_ir (commit 1dd1554d) draws today
Cache drawn for 8/32: Llama, Qwen3, OLMo-2, Cohere, OPT, Qwen1.5-MoE, Falcon-H1, tiny Gemma-3n.
Root causes of the misses (each reproduced):
1. **Single-pass ancestry read check**: sliding layers store a slice downstream of what attention read →
   read=False (Phi-3, Gemma-2, TranslateGemma, Gemma-3n E2B).
2. **Role by weight equality** (w==wk / w==wv): fused QKV (Pythia, GPT-2, Falcon, Phi-3) and MLA
   (kv_b_proj makes both) give only "key" → value missing → not drawn.
3. **"after RoPE" = neg+cat in the chain**: the cache's own concat is in the chain, so it is a proxy
   (Cohere: exec_to_ir rope=False yet cache after="rope").
4. **Layer 0 copied to all layers**: tiny Gemma-3n layers 2–3 drawn as writing their own cache; they read
   layers 0–1's.
5. **Enc-dec**: verifies on the encoder's layer 0 → nothing found.
Side findings (not cache): RoPE missed for Cohere and DeepSeek; TranslateGemma's 29 sliding + 5 full drawn
"34 of 34 identical"; Qwen1.5-MoE and Falcon-H1 not refused although scope is dense decoders; 12/32 crash
(MoE/hybrid/embedding/encoder recognition).

## Proposed fix (not implemented)
Two-step decode as the cache proof (probe2's predicate); role = read by score matmul (K) vs weights·V (V);
rotation by cos/sin evidence; per-layer cache facts (own write / reads layer N / none); enc-dec self+cross
caches; state slots never drawn as K/V; cache drawn only with an observed causal mask.
Open decisions: MLA (HF decompressed vs vLLM latent), checkpoints declaring use_cache=false (e5-Mistral).
