# LLM connection-fidelity audit (Sable · code-truth)

*How far off are our diagram's **connections** (the dataflow arrows / residual
topology / norm placement / merges) from the real HuggingFace `forward()`?*

Method (Gate A.6, both directions): read the actual `DecoderLayer.forward()` in
the **installed `transformers` 5.12** source (ground truth, not guessed), extract
its connection topology, and diff against the block wiring our parser emits
(`decoder_layer_blocks` / `parallel_decoder_layer_blocks`). Focus is *connections*,
not dims. Internal attention/FFN/MoE wiring (Q/K/V, gate/up/down, router→experts→
sum, MLA query∥kv, DSA indexer) is projected from the canonical op-graph and was
verified separately; this audit is the **per-layer macro-wiring**.

## Verdict

The **standard pre-norm, two-residual** topology — which covers the large majority
of the catalogue and nearly every most-served model — is **connection-exact**.
The gaps are concentrated in **three families with distinctive norm/residual
topologies** that the generic builder flattens, plus one parallel-residual nuance.

| Family (forward source) | Source topology | Our wiring | Fidelity |
|---|---|---|---|
| **Llama** 1–4, Code Llama | pre-norm; `r=h; h=ln1(h); h=attn(h); h=r+h; r=h; h=ln2(h); h=mlp(h); h=r+h` | identical | ✅ **exact** |
| **Mistral / Mixtral / Ministral / Magistral** | same | identical | ✅ exact |
| **Qwen2 / 2.5 / QwQ / Qwen3 / Qwen3-MoE** | same | identical | ✅ exact |
| **DeepSeek-V2/V3/R1/V3.1/V3.2, Kimi-K2** | same (layer level) | identical | ✅ exact |
| **GLM-4-MoE, DBRX, Yi, Phi-3/4, Phi-3.5-MoE** | same | identical | ✅ exact |
| **GPT-OSS** | same (mlp returns router scores; sinks are attn-internal) | identical | ✅ exact (sinks = Tier-3, see below) |
| **Phi-3** | same + `resid_*_dropout` on each branch | identical | ✅ exact (dropout = inference no-op, correctly omitted) |
| **Granite** | same + `h = r + h * residual_multiplier` (scalar on each branch) | identical | ✅ connections exact · ⚠ missing the Tier-3 `residual_multiplier` scalar |
| **GPT-NeoX (parallel)** | parallel: `mlp(post_ln(h)) + attn(in_ln(h)) + h` — **two** pre-norms | parallel, **one** shared norm | ⚠ topology right (parallel + combined ⊕), collapses 2 norms→1 (correct for GPT-J/Falcon, under-counts NeoX) — **still open** (minor) |
| **Gemma-2 / Gemma-3** | **sandwich**: `r + post_attn_ln(attn(in_ln(h)))` then `r + post_ffn_ln(mlp(pre_ffn_ln(h)))` — **four** norms | sandwich, four norms | ✅ **FIXED** (`norm_placement=double`) |
| **OLMo-2** | **post-norm**: `r + post_attn_ln(attn(h))` then `r + post_ffn_ln(mlp(h))` — norm on the **output** | post-norm: `attn → norm → ⊕` | ✅ **FIXED** (`norm_placement=post`) |
| **Cohere / Command-R/A** | **parallel**: one `in_ln(h)` feeds both attn and mlp; `r + attn + mlp` (single ⊕) | parallel, one norm, one combined ⊕ | ✅ **FIXED** (data-driven parallel detection) |

## Resolution (this pass)

`decoder_layer_blocks` now honours `norm_placement` ∈ `{pre, post, double}` (emitting
the post-attention / post-FFN / sandwich norm nodes and the correct ⊕ taps), and the
parser reads per-family topology from data — `everchanging/transformer/layer_topology.yaml`
(`gemma2/gemma3 → double`, `olmo2 → post`, `cohere/cohere2 → parallel`). All three
were re-verified against the `forward()` source **and** rendered (pixels inspected);
`test_layer_norm_placement_matches_source_topology` pins them. Adding a future family
with one of these topologies is now a single YAML row — no code change.

**Still open (minor, by choice):** GPT-NeoX's two-norm parallel (we draw one shared
norm — accurate for GPT-J/Falcon), and Granite's `residual_multiplier` scalar (Tier-3
annotation, connections already exact).

---

## Recursive (drill-depth) conformance — every block to its leaves

The above is the layer macro-wiring. Going **recursively into every drill-down view,
to the description-only leaves**, and conforming each against its HF sub-module
`forward()` (captured from the *actual graph the renderer builds*, not memory):

| Drill view | Connections drawn | Source forward | Fidelity |
|---|---|---|---|
| **Attention (SDPA)** | `q/k → RoPE → QK^T/√d`, `v → ⊙`, → softmax → ⊙ → concat → o_proj; K/V cache **ports** | `LlamaAttention.forward` (`apply_rotary_pos_emb(q,k)`, `cache.update`, sdpa, o_proj) | ✅ exact *(after RoPE fix below)* |
| **MLA query path** | `q-proj → split(nope,rope) → RoPE(rope) → concat` | `DeepseekV3Attention` q path | ✅ exact |
| **MLA KV path** | `down → latent cache → up → split(k_nope,v)`; RoPE-key branches **pre-cache** from `down`; merge | `DeepseekV3Attention` kv path (`k_pe` from `kv_a_proj`, pre-expansion) | ✅ exact |
| **DSA indexer** | `hidden → indexer proj → index scores → top-k → selected keys` (third path into scores) | `DeepseekV32` lightning indexer | ✅ exact |
| **FFN (gated)** | `gate → act ∥ up → ⊙ → down` | `LlamaMLP` (`down(act(gate(x))*up(x))`) | ✅ exact |
| **MoE** | `router → [experts] ∥ shared(from input) → ⊕` | `DeepseekV3MoE` (`moe(x) + shared(x)`) | ✅ exact |
| **MoE router gate** | adapts: Mixtral/Qwen3-MoE `gate→softmax→top-k`; DeepSeek `gate→sigmoid→[+bias]→group-limit→top-k→[norm]→[×scale]` | `MixtralSparseMoeBlock` / `DeepseekV3TopkRouter` | ✅ exact (config-driven) |
| **MoE expert** | `gate → act ∥ up → ⊙ → down` | expert `MLP` | ✅ exact |

### The one drill-depth fix (this pass)
**RoPE application was missing in the standard SDPA attention drill.** Every RoPE
family's `forward` runs `apply_rotary_pos_emb(q, k)` between the projections and the
scores; our drill drew `q,k → scores` with no RoPE node (only the MLA drill showed
it). Fixed: `_sdpa_region` now draws an `apply RoPE (Q)` / `apply RoPE (K)` node on
the Q and K lanes, **gated** on `rope` (a new `AttentionSpec.rope`, set false for
ALiBi/learned families — BLOOM/MPT/GPT-2/OPT — via `layer_topology.yaml` `no_rope`)
and on per-layer NoPE (Llama-4). Verified: Qwen3 draws it, BLOOM does not; rendered
and pinned (`test_attention_lanes_and_spine_are_derived_from_edges`,
`test_non_rope_family_omits_the_rope_step`).

### Remaining Tier-3 notes at drill depth (properties, not mis-wired connections)
- **GPT-OSS attention sinks** — a learned per-head logit added at the softmax; not
  drawn (a Tier-3 property of the scores).
- **GPT-OSS gate order** — gpt-oss softmaxes *after* top-k (vs Mixtral's softmax-then-
  top-k); our gate shows the same node set, generic order.
- **MLA LoRA norms** (`q_a_layernorm` / `kv_a_layernorm`) — folded into the projection
  node labels rather than separate boxes.

**Recursive verdict:** every drill view's connections now reconcile with the HF
sub-module `forward()` in both directions, down to the description-only leaves; the
only un-drawn items are Tier-3 *properties* (sinks, a folded LoRA norm), not
mis-wired or missing dataflow.

### FFN / MoE-expert variant depth (the leaf wiring is NOT one-size-fits-all)

Run per family (the internals genuinely differ), conformed against each MLP `forward`:

| Variant | Family | Our leaf wiring | Source | |
|---|---|---|---|---|
| **Dense** (no gate) | Phi-2, GPT-2, OPT, GPT-NeoX, BLOOM | `up → act → down` | `PhiMLP` (`fc2(act(fc1(x)))`) | ✅ |
| **Gated SwiGLU** | Llama/Mistral/Qwen/DeepSeek | `gate → act ∥ up → ⊙ → down` | `LlamaMLP` | ✅ |
| **GeGLU** (gated, GELU) | Gemma-2/3 | same gated shape, **activation node = GELU** | `Gemma2MLP` | ✅ (activation correctly named per `hidden_act`: SiLU/GELU/ReLU/GEGLU) |
| **Fused `gate_up_proj`** | Phi-3, gpt-oss | drawn as the **functional** `gate ∥ up` split | code fuses into one matrix + chunk | ✅ functionally faithful; fusing is a code-packing detail **not derivable from config**, so the two-path view is the honest config-level abstraction (noted) |
| **Clamped SwiGLU expert** | gpt-oss | gated shape; clamp shown as a `clamped ±7` chip | `GptOssExperts` (`(up+1)·gate·σ(αgate)`, + gate_up/down **bias**) | ✅ shape exact; the clamp/`+1`/σ-gate formula and expert bias are Tier-3 properties (clamp surfaced) |

So the FFN/MoE leaf decomposition **is** differentiated where the config can tell us
(dense vs gated, and the activation function), and the remaining per-family differences
(fused packing, the gpt-oss glu formula, expert bias) are code-only details surfaced as
Tier-3 where present — not connection mis-wirings. Per-family re-derivation is now the
standing rule (CLAUDE.md Gate A.7 / B.1).

## The three red-flags (code → structure: things the `forward()` does that the diagram omits/mis-wires)

1. **Gemma sandwich norm — `models/gemma2/modeling_gemma2.py` `Gemma2DecoderLayer.forward`.**
   The code applies `post_attention_layernorm` to the *attention output* and
   `post_feedforward_layernorm` to the *MLP output* (before each residual add), on
   top of `input_layernorm` / `pre_feedforward_layernorm`. We draw only the two
   input norms, so two real norm nodes and their wiring are missing. *(We already
   model exactly this for DiffusionGemma via `diffusion_gemma_layer_blocks`; the
   generic path doesn't reuse it.)* Affects `gemma`, `gemma2`, `gemma3`.

2. **OLMo-2 post-norm — `models/olmo2/modeling_olmo2.py` `Olmo2DecoderLayer.forward`.**
   `self_attn` runs on the *raw* hidden state and `post_attention_layernorm` is
   applied to its *output*; same for the MLP. Our diagram shows `norm → attn`; the
   truth is `attn → norm`. The arrow direction through the norm is reversed.

3. **Cohere parallel residual — `models/cohere/modeling_cohere.py` `CohereDecoderLayer.forward`.**
   `hidden_states = residual + hidden_states_attention + hidden_states_mlp`, where
   **both** the attention and the MLP read the *same* `input_layernorm(h)`. We draw
   it as two sequential sub-blocks with two separate norms and two adds. The whole
   residual topology is wrong (sequential vs parallel). Cohere has **no**
   `use_parallel_residual` config flag, so our flag-based detector misses it.

## Root cause (one place)

`parser.py` hardcodes `norm_placement = "pre"` (line 224) and only ever emits
`decoder_layer_blocks` (sequential pre-norm) unless the `use_parallel_residual`/
`parallel_attn` **config flag** is set. There is no per-family detection for
sandwich norm (Gemma), post-norm (OLMo-2), or flag-less architectural parallelism
(Cohere). The IR already carries `norm_placement` (`pre`/`post`/`double`), but
neither the parser sets it for these families nor does `decoder_layer_blocks`
honor it.

## Fix shape (additive, when you want it)

- **Gemma**: route `gemma/gemma2/gemma3` to a sandwich builder (the
  `diffusion_gemma_layer_blocks` post-norm pattern, generalized — add `post_attn_ln`
  + `post_ffn_ln` quiet norm nodes between each sublayer and its `⊕`).
- **OLMo-2**: detect post-norm (model_type `olmo2`) → set `norm_placement="post"` and
  have the layer builder emit `attn → norm → ⊕` (norm on the output side).
- **Cohere**: treat `cohere`/`cohere2` as architecturally parallel (no flag) → route
  to `parallel_decoder_layer_blocks`.
- **GPT-NeoX**: let the parallel builder optionally carry **two** input norms
  (attn-norm + mlp-norm) when the family uses them, vs one shared (GPT-J/Falcon).

All four are macro-wiring fixes in the parser + the layer builders; the op-graph,
renderer, and the standard need no change. Each should be verified by reading the
`forward()` (above) and rendering the layer.
