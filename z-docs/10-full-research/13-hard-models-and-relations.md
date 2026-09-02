# Hard models and the relation axis — probe of 2026-09-02

Question from Soumil: do newer models (DeepSeek's multi-layer connections,
hybrids, shared KV) void the "multiplier × tower" transformer design, and is
the plan accounting for them? Answer: **yes they void the naive picture, and
the plan as of v2.2 did not account for relations between occurrences.**
v2.3 adds the relation axis (`12` §1f). Everything below was measured on this
machine (torch 2.9.1, transformers 5.12.1, default configs shrunk to 6
layers unless noted, bf16, meta device, FakeTensor forward on 8 tokens).
Script: `scratchpad/demo/hard_models.py` (copy in §5).

## §1 What each architecture actually does (from the installed source)

| model | construct | where in source | breaks which assumption |
|---|---|---|---|
| **DeepSeek-V4** | **mHC, manifold-constrained hyper-connections**: the residual is `hc_mult` parallel streams `[B,S,hc_mult,D]` through the whole stack; each sublayer has a `DeepseekV4HyperConnection` computing `pre` (stream collapse), `post` (output placement), `comb` (H×H mixer, Sinkhorn-projected to doubly-stochastic); `hc_head` collapses at the end. Plus **hash-MoE** (expert choice by a frozen `tid2eid[input_ids]` table for the first `mlp_layer_types=="hash_moe"` layers), CSA/HCA compressed attention with an indexer. | `modeling_deepseek_v4.py:895-935, 1105-1170, 1272-1337` | one residual line; "router = learned"; attention = one KV path |
| **Gemma3n** | **AltUp** (4 residual streams `[4,B,S,D]`, predict/correct per layer), **LAuReL** block, **per-layer inputs** (a pre-stack embedding projected and sliced per layer), **KV sharing** (last `num_kv_shared_layers` layers reuse K/V from the last non-shared layer of the same type) | `modeling_gemma3n.py:997-1050, 1630-1665` | one residual line; layer input = previous layer output only; per-layer KV |
| **Gemma4** | KV sharing via `shared_kv_states[layer_type]`, per-layer inputs, double-wide MLP only on KV-shared layers, tied embeddings | `modeling_gemma4.py:1071-1073, 1252, 1270` | per-layer KV; homogeneous MLP width |
| **LongCat-Flash** | each logical layer = 2 attention sublayers + 2 MLPs + 1 **shortcut MoE** whose output is added after the *second* sublayer; MTP weights ignored on load | `modeling_longcat_flash.py:453-530` | one attention + one FFN per layer; residual = sum of adjacent sublayer |
| **Qwen3-Next** | hybrid: `linear_attention` (gated DeltaNet) vs `full_attention` by `layer_types`, MoE every layer; MTP ignored | `modeling_qwen3_next.py:824, 971` | homogeneous mixer |
| **Jamba** | attention / Mamba alternation by `layers_block_type`, MoE on some | `modeling_jamba.py:669-682` | homogeneous mixer |
| **Nemotron-H** | aperiodic `hybrid_override_pattern` string: each block is one of Mamba2 / MoE / Attention / MLP | `modeling_nemotron_h.py` | periodic schedule |
| **DeepSeek-V3.2** | 3 dense + MoE layers; **DSA indexer** inside each attention selecting top-k tokens, its own RoPE variant | `modeling_deepseek_v32.py:166-244, 349-438` | attention = one score path |
| **gpt-oss** | attention **sinks** as a learned per-head parameter concatenated to logits; sliding/full alternation | `modeling_gpt_oss.py:267-345` | softmax over keys only |
| **Llama-4** | `layer_types` select the mask per layer (chunked vs full, NoPE layers), MoE interleaved | `modeling_llama4.py:574` | homogeneous mask/position |

## §2 What the instance + trace substrate exposed (measured)

| model | instance: layers → distinct shapes; modules beside the stack | trace: hidden shape at layer boundary | trace: cross-layer edges (origin i → consumed j, residual chain excluded) | typed failure |
|---|---|---|---|---|
| DeepSeek-V4 | 6 → 1 (`attn_hc`, `ffn_hc`, attn, MoE, 2 norms); beside: `rotary_emb`, **`hc_head`** | **`(1, 8, 4, 4096)` → multi-stream, n = 4** | 0 (mHC is multi-stream, not cross-layer) | none (bf16 required for `grouped_mm`) |
| Gemma3n (35 layers, unshrunk) | 35 → 1 (`altup`, `laurel`, per-layer gate/projection, 5 norms); beside: `per_layer_model_projection`, `altup_projections`, `altup_unembed_projections`; **`lm_head` tied** | **`(4, 1, 8, 2048)` → multi-stream, n = 4** | **45 edge kinds: layer 18 → layers 20…34** (KV sharing) | none |
| Gemma4 | 6 → 1; beside: `per_layer_model_projection`; tied | — | — | `KeyError('full_attention')`: my probe set `num_kv_shared_layers=2` on a 6-layer slice with no earlier full-attention layer → an honest typed failure of the *recipe*, not the model |
| Qwen3-Next | 6 → **2** (layers 0,1,2,4,5 linear-attn; layer 3 full-attn) | `(1,8,2048)` | 0 | none (gated DeltaNet torch path traced) |
| Jamba | 6 → **2** (even Mamba, odd attention) | `(1,8,4096)` | 0 | none (Mamba slow path traced) |
| Nemotron-H | 4 → **4** (Mamba2 / MoE / Attention / MLP, one each) | `(1,8,4096)` | 0 | none |
| DeepSeek-V3.2 | 6 → **2** (0–2 dense, 3–5 MoE) | `(1,8,7168)` | 0 | none (indexer top-k traced) |
| gpt-oss | 6 → 1 | `(1,8,2880)` | 0 | none |
| Llama-4 | 6 → 1 | `(1,8,5120)` | 0 | none |
| LongCat-Flash | — | — | — | probe's stack finder failed (its `self_attn`/`mlps` are 2-element ModuleLists *inside* the layer); structure known from source §1 |

Also observed on every model: pre-stack tensors consumed inside layers ≥ 1
(rotary cos/sin, the causal mask, Gemma's per-layer inputs as `(1,8,256)` /
`(1,8,512)` slices). These are the `per_layer_side_input` relation for Gemma
and ordinary global inputs elsewhere; the detector must distinguish them by
whether the tensor is sliced per layer index.

## §3 Consensus: what the plan must add, and what it already had

1. **A relation axis** (`12` §1f) with typed kinds: `param_share`,
   `activation_reuse`, `multi_stream_residual(n)`, `per_layer_side_input`,
   `intra_layer_shortcut`, `layer_reuse`, `conditional_skip`, `side_head`.
   Detectors are cheap and already demonstrated: parameter identity; tensor
   lineage across layer hooks; hidden-state rank at the layer boundary;
   pre-stack tensor consumption; non-layer modules beside the stack.
2. **The IR already has the seam**: `ModelIR.cross_layer_edges` with
   `CrossLayerEdge(kind, from_layer, to_layer, shared)` (`ir.py:390`), produced
   today only for `kv_share` from the source reader
   `evidence/kv_sharing_schedule.py` and drawn in `renderers/html/views.py:1271`.
   Generalise this field; do not add a parallel one.
3. **Heterogeneous stacks were already representable**: per-layer `layers`
   list, `distinct_layer_groups` (collapse by signature in encounter order),
   `detect_layer_period` returning `None` for aperiodic schedules. The
   heterogeneity found by the instance (Qwen3-Next 5:1, Jamba, Nemotron-H's
   string, DeepSeek 3+58) lands in existing structure. What must be confirmed
   is that the *renderer* draws an aperiodic schedule truthfully rather than
   forcing a cycle (`14` C-7).
4. **Multi-stream residual is the one thing the current IR cannot say.** A
   `Layer` has one residual; DeepSeek-V4 and Gemma3n have four with learned
   mixers per sublayer. This needs a first-class `residual_streams` fact on
   the stack plus per-sublayer mixer occurrences, drawn as a bus. Until it
   exists, both models must render a visible `relation_unresolved` chip on
   the residual, never a single line.
5. **Recipes need dtype.** MoE `grouped_mm` refuses fp32 under FakeTensor;
   bf16 is part of the recipe record (`12` §1c).
6. **Data-dependent control flow was not the blocker the counter-review
   feared** for this set: learned top-k, hash routing, indexer top-k, Mamba's
   scan and gated DeltaNet all traced under FakeTensor in ≤ 0.6 s. The
   remaining risk is models whose forward branches on tensor *values*
   (early-exit, dynamic depth); those get a typed `execution_unresolved`.
7. **Side modules beside the stack are the new normal** (`hc_head`,
   `per_layer_model_projection`, `altup_*`, MTP). The denominator law makes
   silent omission impossible; the relation axis gives them a meaning.

## §4 What this does *not* settle

- How to **draw** a 4-stream residual with Sinkhorn mixing so a learner
  reads it correctly. That is S12 design work and a Her Eyes question.
- `layer_reuse` and `conditional_skip` were not probed (no installed
  looped/MoD model in this environment). The detector design is stated; the
  first witness must be added to the corpus when one is available.
- LongCat's intra-layer shortcut was read from source, not traced; the probe
  needs a layer finder that accepts nested sublayer ModuleLists.

## §5 Probe script

`scratchpad/demo/hard_models.py` — meta build from the default config class,
signature grouping of the largest decoder ModuleList, parameter-identity
groups, then a FakeTensor forward with layer pre/post hooks setting the
"current layer" and a `TorchFunctionMode` that tags every produced tensor
with its origin layer and flags consumption in a later layer (residual chain
output excluded) and pre-stack tensors consumed in layers ≥ 1. Copy this
into `physics/` at S6; it is the seed of the relation detectors.
