# Optimization & Modularization — rolling the catalogue onto the standard

*This is the capstone of the standardization pass: what the block-worthiness
standard (`docs/BLOCK_STANDARD.md`) demanded, what was changed to meet it across
**all** transformer and diffusion families, the evidence it works, and the
roadmap for what's left.*

The thesis from `BLOCK_STANDARD.md` is one sentence: **a block is a thing a
researcher would draw on a whiteboard and give a name; everything else is an
arrow or a footnote.** The work below takes that from a rule applied to one model
(DiffusionGemma) to a property of the whole engine, and removes the seams that
let the rule be violated silently.

Every change here is **additive** — a tag, a YAML row, a config read, a new view —
never a rewrite, and each is verified by *rendering and inspecting* the output and
by a pinned test, per the working nature.

---

## 1. What the standard demanded, and the seams that blocked it

The standard says residual `⊕`, gate `×`, splits and concats are **Tier-2
connectors** (glyphs, no card); scalars and properties are **Tier-3 annotations**
(chips/captions); only named computation is a **Tier-1 block**. Before this pass:

| Seam | Symptom | Why it blocked the standard |
|------|---------|------------------------------|
| `static` / `branch_side` not in the `Block` schema | the paradigm's own keys were treated as **typos** by `validate_block_tree` | you could not roll a model onto Tier-2 without the validator faulting it (latent — no paradigm model was in the test corpus, so CI was green on a broken contract) |
| every standard decoder layer drew residual adds as **clickable Tier-1 boxes** | Llama/Mistral/Qwen/… rendered the `⊕` as a big card-bearing box | the over-blocking disease the standard exists to cure, in the most-used path |
| the MoE gate was an opaque `Router` box | modern routing (sigmoid, group-limit, bias, scale) lived only in card prose | the *diagram* showed plain top-k — wrong for 9 frontier MoEs |
| the always-on **shared expert** was absent from the MoE diagram | `out = shared(x) + Σ gᵢ·eᵢ(x)` drew only the routed sum | a Tier-1 computation the code does was missing |
| `layer_types` label vocabulary was **hardcoded** in the parser | `attention`, `deepseek_sparse_attention` "treated as causal" with a warning | violated the "config vocab lives in `everchanging/` YAML" rule, and mislabeled real attention layers |
| training/loss + merged T5 encoder fields counted as **unparsed** | the coverage/serve audits flagged ~20 non-architectural keys | noise drowned the *real* dropped-architecture signals |
| a wide/off-centre graph **note** clipped at the canvas edge | the router caption was cut off | `fit_svg` never measured note width |

---

## 2. The changes (this pass)

### 2.1 Schema: bless the paradigm keys — `block_schema.py`
`static: bool` and `branch_side: str` are now first-class `Block` keys with the
Gate-C semantics documented inline. The schema test corpus gained a
`block_paradigm` entry (DiffusionGemma) so the Tier-2 connector / inline-branch
keys are **validated on every run** — the contract can't silently rot again.
*(Closed a latent bug: DiffusionGemma's tree produced 10 false "unknown key" faults.)*

### 2.2 The connector sweep: standard decoder layers → Tier-2 — `blocks/layers.py`
`decoder_layer_blocks` and `parallel_decoder_layer_blocks` now tag `add1` / `add2`
`static: True`. Every standard-decoder family (Llama, Mistral, Qwen, Phi, Yi,
DBRX, Cohere, OLMo, Granite, GPT-J/NeoX/Falcon parallel-residual, …) inherits it
at once. The residual `⊕` is now a quiet glyph on its loop, and the two boxes that
carry the architecture — attention and FFN — are what the eye lands on.

> **Before:** `RMSNorm · [Attention] · [⊕ Residual add] · RMSNorm · [Feed-Forward] · [⊕ Residual add]` — 4 boxes + 2 big add boxes.
> **After:** `RMSNorm · [Attention] · ⊕ · RMSNorm · [Feed-Forward] · ⊕` — the adds are glyphs on their loops. Verified by rendering Llama-2-70B (GQA) and inspecting the pixels.

### 2.3 The MoE gate: show the real router + the shared expert
- **`block_views/moe_router.py` (new):** the `Router` box now drills into a
  **config-driven gate pipeline** — `Gate (Linear→scores) → score (sigmoid│softmax)
  → [group-limit: keep g of N groups] → top-k → [renormalize] → [×scale]` — with
  the aux-loss-free (`noaux_tc`) **bias entering the *selection* step from the
  side** and a caption noting the weights use the raw scores. Every bracketed step
  appears **only when the config declares it**, so Mixtral collapses to
  `softmax → top-2`, Qwen3-MoE adds `renormalize`, and DeepSeek-V3 shows the full
  grouped, bias-corrected, ×2.5 pipeline. (Verified across all three configs.)
- **`mixture_of_experts.py`:** the always-on **shared expert** is now drawn as a
  Tier-1 lane that **taps the block input** (bypassing the router) and joins the
  weighted-sum `⊕`; the sum itself is a Tier-2 `static` connector.
- A new engine glyph kind `select` (routing selection step) was added to `KIND`.

### 2.4 Two architectural features that were silently dropped
- **DeepSeek-V3.2 DSA** (`ir.py` + `parser.py` + `labels.py` + `opgraph.py` +
  `block_views/dsa_indexer.py`): the lightning indexer is now a **Tier-1
  drill-down sub-block** — a *third path* into the attention scores
  (`hidden → indexer projections (index_n_heads × index_head_dim) → index scores
  → keep top-index_topk → selected keys`), clickable, alongside an attention-card
  annotation. Added to the canonical `_mla_region` and **strictly gated on
  `index_n_heads`**, so the other six MLA models (V3 / R1 / V3.1 / Kimi×2) are
  untouched. (V3.2's unparsed list collapsed from 7 fields to just
  `quantization_config`.)
- **gpt-oss clamped SwiGLU** (`swiglu_limit`): read into `FFNSpec.activation_clip`
  and shown as a `clamped ±7` chip on the FFN.

### 2.7 Robustness: parameter estimation never crashes — `params.py`
A `_as_count` coercion makes `_ffn_params` / `_attn_params` tolerate list-valued
or `None` counts (a heterogeneous MoE schedule), degrading to an approximate
number instead of raising. *Closed a real crash:* `tencent/HunyuanImage-3.0`
declares list-valued expert counts and previously died with a `TypeError`
mid-render; it now renders (≈79.8 B params, schema/coupling clean).

### 2.5 Modularization: config vocabulary out of code
- **`everchanging/transformer/layer_types.yaml` (new):** the per-layer
  attention-type label groups (`full` / `sliding` / `compressed_sparse` /
  `heavily_compressed`), formerly four hardcoded `set` literals in the parser, are
  now editable data. Added `attention` (fixes Nemotron-H's attention layers being
  mislabeled) and `deepseek_sparse_attention` to `full`.
- **`everchanging/transformer/ignored_fields.yaml`:** the MoE training/aux-loss
  knobs (`router_aux_loss_coef`, `seq_aux`, jitter, …) and the T5/CLIP encoder
  internals merged in from diffusion component configs (`d_kv`,
  `relative_attention_*`, `feed_forward_proj`, …) are now quieted, so the audits
  surface **only real architectural gaps**.

### 2.6 Engine: notes no longer clip — `graph_engine.py`
The downstream `note` registers its full horizontal extent (not a bare point), so
`fit_svg` grows the canvas to contain it. Fixes the clipped router caption and any
future wide/off-centre note across **every** view.

---

## 3. Evidence

- **Tests:** `python3 -m pytest -q` → **all green**, including new pins:
  `test_moe_gate_view_is_config_driven_and_shared_expert_drawn`,
  `test_dsa_indexer_and_clamped_swiglu_are_surfaced`,
  `test_layer_type_labels_externalized_and_cover_modern_spellings`, and the
  `block_paradigm` schema-corpus entry.
- **Click-coupling:** `validate_click_coupling` clean on every rendered model.
- **Serve audit** (`scripts/serve_audit.py`, render every catalogued model
  end-to-end → `docs/serve_audit.md`): **88 rendered clean · 0 schema problems ·
  0 coupling problems** across both adapters (5 errored = newest unsupported
  formats; 13 gated). The unparsed-field noise collapsed and the partial-config
  list now shows only genuine items. Concrete before → after:
  | Model | unparsed before | unparsed after |
  |---|---|---|
  | FLUX.1-dev | 9 (T5 encoder internals) | **none** |
  | Qwen3-235B-A22B | 2 (router training flags) | **none** |
  | gpt-oss-20b | 5 | **2** (`swiglu_limit` now drawn) |
  | DeepSeek-V3.2-Exp | 7 | **4** (the 3 `index_*` DSA fields now drawn) |

  Partial-config warnings dropped from DeepSeek-V3.2 (DSA recognized) and lost
  Nemotron-H's bogus `attention` warning; what remains is only the honest
  out-of-scope signal (`mamba`/`mlp`/`conv` hybrid layers).
- **Pixels:** Llama architecture, the DeepSeek-V3 MoE view, and the router gate
  drill were each rendered to PNG and inspected.

---

## 4. Roadmap — what's left, with the shape of the fix

Ordered by value. None are blocking; each is additive.

1. ~~**DSA indexer as a drill-down sub-block**~~ — **DONE** (§2.4): now a Tier-1
   sub-block on V3.2's attention, strictly gated, with its own drill-down view.
2. **Transformer pale-when-unapproved guard.** Diffusion renders an unblessed
   `diffusion_stage` pale; the transformer taxonomy (`transformer/typing.yaml`) is
   blessed but the renderer doesn't yet draw transformer blocks pale when their
   stage is unknown. Wiring it makes a mistyped/new transformer stage as visible
   as it already is on the diffusion side. (`block_schema.py` notes the hook.)
3. **Loaders for the newest formats** (from the serve audit's 5 errors):
   `mistralai/Pixtral-12B-2409` (consolidated, non-`transformers` layout),
   `Qwen3-Omni` (text model nested under `thinker_config`), and the Tencent
   Hunyuan-Video new releases (invalid root JSON / new component layout). *(The
   `HunyuanImage-3.0` `TypeError` crash from this list is now fixed — see §2.7;
   it renders. The remaining four are graceful typed load errors, not crashes.)*
4. ~~**M-RoPE surfacing**~~ — **DONE**: `rope_scaling.mrope_section` is read and
   shown as a Tier-3 chip (`M-RoPE 16/24/24`) + description on Qwen-VL attention.
5. **Scope note — hybrid SSM.** Nemotron-H / Jamba / LFM2 / Falcon-H1 interleave
   Mamba/conv layers with attention. Per the standing scope (LLMs + diffusion, no
   SSM/Mamba), those layers stay out; the audit's `mamba`/`mlp`/`conv`
   "treated as causal" warnings are the honest signal that we draw the attention
   layers and not the state-space ones.

---

## 5. How to extend without re-reading this

- **New family →** run the **Sable** procedure (`BLOCK_STANDARD.md` §6): render
  first, sort each element into a tier, tag (`static` / `branch_side` / `facts` /
  `layer_annotations`), bless any new stage in `everchanging/`, conform both
  directions against the HF `forward()`, re-render, keep the gates green.
- **New config dialect →** add the spelling to the relevant
  `everchanging/<adapter>/*.yaml` (aliases, layer_types, ignored_fields) — no code
  change.
- **New catalogue model →** add the id to `toserve.md`; `scripts/serve_audit.py`
  picks it up automatically (gated repos are marked `🔒` or self-detected at 401).
