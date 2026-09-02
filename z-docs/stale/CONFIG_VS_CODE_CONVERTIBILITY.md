# CONFIG ↔ CODE: contributions, partial-config handling, and convertibility

*What config.json supplies vs what the modeling source supplies inside `unfold()`,
how code covers progressively-partial configs, and — field by field — what can be
fully converted to code-authoritative vs what must remain a checkpoint value.*

*Written 2026-07-12. Analysis doc; companion to `DISTRIBUTION_OF_INVOCATION.md`,
`MODEL_DISTRIBUTIONS.md`, and `unfold-pkg/docs/EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md`.*

---

## 0. The governing law

> **Code proves that a field *controls* a structural branch or expression; the
> checkpoint config supplies the *value* selected for this checkpoint.**
> (EVIDENCE_ARCHITECTURE_HARDENING_PLAN §1.2, binding.)

Neither replaces the other. Config is **checkpoint truth**; code (read as AST,
never executed) is **structure truth**. Identity (`model_type`, `architectures`,
`auto_map`, `_repo_id`) is an **address** to locate source — never a fact.

---

## 1. Who supplies what

| Question | Answered by | Why | Status tier |
|---|---|---|---|
| *How big?* — hidden_size, layers, heads, kv_heads, intermediate_size, head_dim | **config** | Geometry is a checkpoint choice; `nn.Linear(config.hidden_size,…)` has no literal | `config_declared` (via `consume()`) |
| *Which enum?* — activation, prediction type, declared conditioning kind | **config**, when source proves the dispatch reads it | A public semantic declaration | `config_declared` / `code_and_config` |
| *What shape?* — gated vs dense, norm placement, RoPE applied?, fused storage?, QK-norm?, which layers MoE, router fn, per-projection bias | **code (AST)** | Config is **silent** about all of these | `code_proven` |
| *Which value for a code-named gate?* — `if layer_idx in config.moe_layers`, `bias=config.attention_bias` | **code names the field → config supplies the value** | Code decides *shape*, config the *selection* | `code_and_config` |
| *Where is the source?* | **config identity** | Identity is an address, never a fact | (resolution only) |

Code path: geometry via
[`consume()`](unfold-pkg/model_unfolder/adapters/transformer/parser.py#L621);
structure via the always-on `_code_*` readers (the activation ladder at
[lines 650–692](unfold-pkg/model_unfolder/adapters/transformer/parser.py#L650) is
the whole doctrine in one function). Every fact records *which channel decided it*
via the [`FactLedger` status enum](unfold-pkg/model_unfolder/evidence/context.py#L25).

---

## 2. Three stages of partial config, and how code handles each

The tri-state law holds throughout: **proven / honest-unknown / never-a-wrong-guess.**

### Stage 1 — config is *structurally silent* (complete geometry, no structure fields)
The everyday case. Config declares dimensions but nothing about gating, norm
placement, RoPE, fused storage.
- **Code handles it:** ~a dozen always-on `_code_*` readers read the AST
  ([`_code_ffn_activation`](unfold-pkg/model_unfolder/adapters/transformer/parser.py#L670),
  FFN gating/storage, norm-from-math, placement-from-dataflow, fused QKV, QK-norm,
  MoE schedule, router, partial-rotary, bias).
- **Example:** Phi-3 → code proves *dense* (not the old "modern LLM = gated" default).
- **Status:** `code_proven` (+ `config_declared` geometry, or `code_and_config`).

### Stage 2 — config is *bare / un-hydrated* (even declared defaults missing)
A raw `config.json` from a registry-predating / remote-code repo that never saw its
config class, so class defaults upstream didn't serialize are absent (Gemma-2's
`sliding_window_pattern`; GPT-J's `n_inner: null`).
- **Code handles it, three ways:**
  1. **Class-default hydration** —
     [`_hydrate_config_class_defaults`](unfold-pkg/model_unfolder/parser.py#L470)
     re-runs the dict through `AutoConfig.for_model`; raw keys win, `_`-stamps
     survive, parse stays name-blind (defaults arrive as data). Exposed as the
     [`class_defaults` tier](unfold-pkg/model_unfolder/evidence/context.py#L126).
  2. **Derived expressions from source** —
     [`_code_intermediate_size`](unfold-pkg/model_unfolder/adapters/transformer/parser.py#L646)
     evaluates GPT-J's `n_inner=None → 4×n_embd` AST ternary; ChatGLM's RoPE
     fraction exists *only* in code.
  3. **Fall to the Stage-1 code readers.**
- **Status:** `class_default` / `derived` / `code_proven`. (GPT-J param 2.29B → 6.05B, with provenance.)

### Stage 3 — config present but *source unavailable* (`oracle_missing`)
Config resolves but no modeling `.py` can be located. Code has nothing to read.
- **Code handles it by honest degradation:** the unknown tier is pre-decided —
  [`_unknown_status = "ambiguous" if _source_present else "oracle_missing"`](unfold-pkg/model_unfolder/adapters/transformer/parser.py#L600).
  Every `_code_*` reader returns `None`, so config-declarable facts stay declared
  (weaker `config_declared`), code-only facts become `unknown` (pale/static), and
  the op-graph ends in an **opaque node** — never a conventional default.
- **Key distinction:** `oracle_missing` (no source — legitimate) ≠ `ambiguous`
  (source present but reader failed — **blocking**). A detector failure is never
  permission to assert a fact.

### The cross-cutting edge — config presence contradicts code (a *dead flag*)
Qwen2 hardcodes `bias=True`, so config `attention_bias=False` is dead → **code wins**;
Llama *gates* on `config.attention_bias`, so there the flag is honored. Same law,
opposite outcomes, decided by *what the code does with the field*, never presence alone.

---

## 3. The convertibility test

> **Does the source *contain* the fact, or only a *slot that reads a number from config*?**

- Source **enacts a structure** (builds a gated MLP, applies a sliding mask, gates a
  per-layer class on a field it *names*) → **convertible to code-authoritative.**
- Source is `nn.Linear(config.hidden_size, …)` — a slot with no literal → **not
  convertible; a genuine checkpoint value.** (*Values can't shift, shapes can.*)

**Worked example — Llama-4-Scout (`llama4_text`):**

| Config field | Fact | Bucket | Convertible? | How code proves it |
|---|---|---|---|---|
| hidden_size, layers, heads, kv_heads, intermediate_size(_mlp), num_local_experts, num_experts_per_tok, rope_theta, vocab | geometry / counts | C | ❌ | code reads them — pure values |
| hidden_act + gating | gated SwiGLU | A | ✅ done | forward: 2 linears + chunk + multiply |
| `moe_layers` (list) | which layers MoE | A | ✅ done (shape) | `layer_idx in config.moe_layers` → structural "builds experts"; list stays config |
| `no_rope_layers` | NoPE schedule | A | ✅ done | reads the per-layer list |
| `use_qk_norm` | QK-norm | A | ✅ done | constructed+applied, composite `and self.use_rope` |
| attention bias | `+bias` | A | ✅ done | reads `nn.Linear(bias=…)` |
| (Llama4TextExperts bmm) | fused expert storage | A | ✅ done | matmul-with-weight → `linear`; fused SwiGLU drill |
| `attention_chunk_size` | chunked attention | B | ✅ **not done** | code applies a chunked mask — currently unread |
| `attn_scale`/`attn_temperature_tuning`/`floor_scale` | attn temperature tuning | B | ✅ **not done** | code enacts the schedule — currently unread |

---

## 4. Full comparison + one-liner conversion plan

**The headline:** transformer structure is **essentially already fully
code-authoritative** (S1/S2/S3 + honesty-wave arcs closed it). The remaining
config→code frontier is almost entirely **diffusion (H7)** + two Llama-4 attention
facts. Geometry never converts.

**Column-3 legend:** ✅ code-authoritative today · ⚠ partly config / render-deferred · 🔴 still config / class-name-table / terminal-default.

### A. Pure geometry — stays config forever (non-convertible residue)

| Fact | Config supplies | Code status | Plan → full code |
|---|---|---|---|
| hidden_size, layers, heads, kv_heads, intermediate_size, head_dim, vocab, max_pos | the number | n/a | **N/A** — `nn.Linear(config.X)`, checkpoint truth |
| rope_theta, rms_norm_eps, window/chunk *sizes*, expert *counts*, n_group/topk_group, routed_scale | the number | n/a | **N/A** — checkpoint value the code reads |

### B. Transformer structure — already converted (config residual = value only)

| Fact | Config supplies (residual) | Code status | Plan / note |
|---|---|---|---|
| FFN gated-vs-dense | — | ✅ | done — 2-linears+chunk+multiply |
| FFN activation | act name (`code_and_config`) | ✅ | done — FFN forward / ACT2FN dispatch |
| Norm kind (RMS/LayerNorm) | eps value | ✅ | done — from norm math, math-above-eps |
| Norm placement (pre/post/sandwich/parallel) | — | ✅ | done — forward dataflow |
| Fused QKV / gate_up / expert storage | — | ✅ | done — construction shape + chunk/split |
| QK-norm | gate value when gated | ✅ | done — constructed+applied, per-layer |
| Partial-rotary split | rope_dim value | ✅ | done — rotary slice width from code |
| MoE-vs-dense schedule | `moe_layers` / threshold value | ✅ | done — structural experts + per-layer gate |
| Router scoring / aux-bias / sparsemixer | n_group/topk/renorm | ✅ | done — routing closure, not `scoring_func` string |
| Attention bias (per-projection) | bool when code gates on config | ✅ | done — reads `nn.Linear(bias=…)` |
| Sliding-vs-full schedule | window size + pattern value | ✅ | done + regression-locked (`layer_types`) |
| NoPE / cross-attn layer schedule | the list value | ✅ | done — `no_rope_layers` / `cross_attention_layers` |
| Attention sinks, score-scaling (main path) | scale constant | ✅ | done — structural Parameter-into-scores; B1 wired |
| Softcapping, MLP bias, MLA cross-check | the cap/bool value | ✅ | done (honesty waves) |

### C. Transformer structure — convertible, still pending (small)

| Fact | Config supplies (residual) | Code status | Plan → full code |
|---|---|---|---|
| Chunked / local attention (Llama-4) | `attention_chunk_size` | 🔴 unread | read the chunked-mask construction; size stays config |
| Attn temperature tuning (Llama-4) | `attn_scale`/`floor_scale` | 🔴 unread | read the temperature schedule from forward; values stay config |
| Parallel 2-norm render (NeoX/Falcon) | — | ⚠ fact✅ render-deferred | build the lane/tap 2-box layout to draw the wired fact |
| Hybrid mixer schedule (attn-vs-linear) | `layer_types` list | ⚠ IN-scope part | reuse per-layer primitive; pure-Mamba stays C1/out |

### D. Diffusion structure — the real remaining frontier (H7)

Currently leans on class-name YAML tables (`unet_blocks.yaml`, `vae_classes.yaml`,
`schedulers.yaml`) + terminal defaults (Transformer2D / KL / Euler) — the exact
identity debt the hardening plan deletes.

| Fact | Config supplies (residual) | Code status | Plan → full code (H7.x) |
|---|---|---|---|
| Conditioning kind (text/image) | `encoder_hid_dim_type`/`addition_embed_type` enum | 🔴 config-only | H7.1 — bind enum → constructed projector via source; unknown → opaque |
| UNet stage construction / mid presence | block-type strings (address only) | 🔴 class-name + default | H7.2 — resolve exact down/mid/up stage classes; negative-mid needs complete inspection |
| UNet cell internals (ResNet/attn/FFN/norm) | — | 🔴 Transformer2D default | H7.3 — per-stage-owner evidence; delete `unet_blocks.yaml` + 2D default |
| Temporal per-stage (TemporalResnet/AlphaBlender) | — | 🔴 model-level stamp | H7.4 — prove each stage; frames-axis can't stamp every stage |
| VAE latent kind (KL/VQ), conv-dim, causality | — | 🔴 `vae_classes.yaml` | H7.5 — read exact VAE code; delete table; unknown ≠ KL |
| Scheduler mechanism (history/predictor-corrector/consistency) | coefficients + prediction type | 🔴 `schedulers.yaml` | H7.6 — read `step()` for state/history/noise; delete table; unknown ≠ Euler |
| DiT block-norm kind (AdaLN vs …) | — | ⚠ reader exists, Wave-C wiring | verify `diffusion_norm_from_classes` on `_dit_norm_kind` fallback |

---

## 5.5 `true_config()` — materialize the complete config (proposed)

Everything above says: config is silent about structure, and code recovers it. The
natural artifact of that is a function that **hands the recovery back as a config** —

> **`true_config(x)`** returns the config-shaped, **provenance-carrying** dict of the
> fully-resolved architecture: the checkpoint's declared values **plus** every
> structural fact the code proves (each tagged with the channel that decided it —
> `config_declared` / `code_proven` / `code_and_config` / `class_default` /
> `derived`), with every still-unproven fact listed honestly under `_unresolved`
> (never guessed). "Everything a config should have had," and nothing it shouldn't.

Design principles (they fall straight out of the project's laws):

- **A projection, not new acquisition** — it serializes the already-parsed
  `ModelIR` + `FactLedger` (`ir.extras["fact_provenance"]`); it never recomputes a
  fact (Law 2 "one declaration, N projections"; G-8 "one author").
- **Provenance never dropped** — a flat merge that hides the channel would be the
  defaults-as-facts sin; every field carries `{status, source}`.
- **Unknown stays unknown** — holes go in `_unresolved` with the reason
  (`oracle_missing` vs `ambiguous`), never filled (Law 5).
- **Values vs shapes** — geometry keeps checkpoint values; a code-named gate
  (`moe_layers`) shows the shape (code) *and* the value (config).
- **It is the campaign's coverage meter** — today it is partial and *says so*
  (`_coverage.unattributed`, `_unresolved`); it becomes complete exactly when the
  config→code registry closes (H2/H10). A `diff=True` mode gives the 3-way
  {declared / added-by-code / still-unknown} split — the live companion to the §4
  buckets. A round-trip (`unfold(true_config(x))` reproduces the diagram) is both a
  metamorphic test and a portable source-free spec.

**Status:** design only. Full spec + build steps: `unfold-pkg/docs/TRUE_CONFIG_PLAN.md`
(committed on `audio-composite-support` as a separate DESIGN-ONLY commit — no code).

## 6. One-sentence conclusion

**Config's permanent job shrinks to exactly two things — geometry numbers, and the
*selected value* of a code-named gate** (the `moe_layers` list, the
`first_k_dense_replace` count); **everything structural on the transformer side is
already code-authoritative**, and **the only large body of "still config,
convertible" work left is the diffusion vertical (H7)** — conditioning, UNet
stages/cells, temporal, VAE, scheduler — each converting by the same recipe:
*resolve the exact owner's callable, read the mechanism from its `forward()`/`__init__`,
delete the class-name table + terminal default, keep the checkpoint value as
`config_declared`.*
