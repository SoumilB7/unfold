# Execution order v2.3 — binding (2026-09-02)

v2.3 adds the **relation axis** (§1f) after the hard-model probe in `13-hard-models-and-relations.md`. Supersedes v1 (big-bang cutover; stale F4/F2c) and v2.1 (single-disposition
reconciliation; bet 2 overstated; remapping mis-scoped; unlawful time-box
escape). Every correction in v2.2 was tested live on SD3.5, PixArt-Σ and
Gemma-2 (§7) before being accepted. Both reviews approve the direction.

---

## §0 The settled decisions

| decision | ruling |
|---|---|
| dependencies | torch, transformers, diffusers and whatever a model needs are acceptable and may be executed. |
| remote custom code | supported, in a bounded subprocess: network off, timeout, memory cap, version record, explicit provenance. Isolation, not prohibition. |
| authority | per question (§1b), never a vote; conflicts are typed blocking findings. |
| tier contract | tier 1 structure complete inside the declared support set (missing = blocking); tier 2 mechanism honest (unknown legitimate, visible); tier 3 values exact and sourced. Pending Soumil's one-line ratification. |
| no single author | none of instance / trace / source / config authors the diagram; a typed reconciliation layer (§1) feeds **the existing canonical fact layer and IR**; the instance tree is never a second structure description beside the IR. |
| U11 | F4/F2c (`9b3cb7b`, `9a4e1e5`) kept and pushed. G/H/I frozen on the old substrate; resumed as G′/H/I over the reconciled authority. |
| deletion | responsibility by responsibility, after a proven replacement and corpus + out-of-corpus parity. Never in the replacement's unit. |
| identity | still address only. `if class_name == "Llama…"` remains forbidden. Exact **runtime type ∈ the closed `torch.nn` set** is definitional, not identity: it is the object, not a name. A user class merely *named* `LayerNorm` is custom. |

## §1 The reconciliation contract — three orthogonal axes

A single disposition enum was wrong: a module is simultaneously constructed,
executed under some recipe, and drawn somewhere. Forcing one value would have
made `reconciliation.py` invent precedence while being written. Every
constructed occurrence gets **exactly one value on each applicable axis**.

**Construction axis** (authority: meta instance inventory)
- `eager_constructed` — present after `__init__`
- `lazy_observed` — absent after `__init__`, appeared during a recipe forward
  (first-call caches, `Lazy*` modules)
- `construction_conflict` — instance and static owner graph disagree on class,
  count or shape → **blocking**
- `not_constructed(guard)` — a source-declared child whose constructor guard
  fired false for this checkpoint (SD3.5 `attn2 = None`); a construction-level
  negative, supporting evidence for tier-2 negatives

**Execution axis** (authority: recipe-qualified trace for positives; static
closure for reachability and negatives)
- `observed(recipe_ids)` — positively executed
- `statically_reachable` — not observed under the recipes run, proven source
  path exists
- `proven_inactive` — complete static negative proof under closure
- `execution_unresolved` — none provable, **visible chip**

**Projection axis** (authority: canonical facts / block-worthiness rule)
- `rendered` — its own block or cell
- `grouped(parent)` — drawn inside a parent block (projections, norms inside
  attention, `Dropout`, containers), rule cited
- `non_architectural(reason)` — helper/utility, reason required, never silent
- `projection_unresolved` — **visible chip**

Laws: a missing trace call is never negative evidence; `named_modules()`
plus lazily observed modules is the denominator and a silent omission is a
gate failure; provenance is two-keyed (instance path for construction, static
occurrence / config path for meaning); a `rendered`-tier element with an
unresolved axis inside the declared support set is a blocking finding.

## §1b The authority matrix (precedence per question)

| question | primary authority | supporting |
|---|---|---|
| Which class did the framework select? | framework factory / instantiated object | source resolver + exact library revision |
| What modules were constructed? | meta instance inventory | constructor source + checkpoint metadata |
| What ran under a given mode? | recipe-qualified trace (module hooks **and** a `TorchFunctionMode` for functional ops) | source control flow |
| What could run on unobserved alternatives? | static source analysis under closure | additional traces |
| What does a **custom** mechanism compute, its alternatives, its negatives? | exact static mechanism reader, starting from the exact class object | trace / export conformance |
| What does a **closed framework primitive** compute? | its exact runtime type (`torch.nn.LayerNorm` *is* LayerNorm) | source of the call site |
| Is fused vs split storage used? | instantiated parameter shapes | source assignment |
| What raw value did the checkpoint declare? | config document (path provenance) | — |
| What class default applies when absent? | exact class source / default declaration | instance confirms |
| What derived runtime value was computed (`head_dim = hidden // heads`)? | constructor expression + exact operand provenance; instance confirms | — |
| What parameter shape is expected? | instantiated parameter inventory | — |
| What shape was actually stored? | checkpoint tensor metadata | — |
| Is a parameter shared (weight tying)? | **parameter object identity on the instance** — a tier-1 structural relation | config declaration, source assignment, tensor metadata |
| What gets drawn? | canonical typed facts / structural IR | receipts against all authorities |
| What may a renderer decide? | presentation only | never architecture |

Corrected law: *static source is the primary authority for custom mechanism
semantics, alternatives and negative proofs; closed framework primitive
types, parameter shapes and positive execution observations are primary for
the questions they definitionally answer.*

## §1c Records every instance/trace result must carry

Package name + version for every library in the resolved class's MRO; exact
source-file hash of each modeling file; config hash; resolved class (module,
qualname); constructor or factory used; build flags; and for observations the
**recipe**: input modality, train/eval state, cache state, encoder/decoder
mode, conditioning present, library version, flags. Recipes are chosen from
callable signatures and capabilities, never from family names. Value
provenance is never erased by the instance (§1b value rows).

## §1d What the product draws

**The declared base architecture: the object constructed from the checkpoint
config and the resolved modeling code, before deployment-specific
transformation.**
- **In scope:** framework factory / class remapping during base construction
  (PixArt `Transformer2DModel` → `PixArtTransformer2DModel`), guarded
  constructor branches, weight tying.
- **Deployment layer, out of scope until the base ships:** post-weight-load
  mutation, quantization replacement, LoRA/adapters, device sharding,
  loader hooks. Represented later from checkpoint metadata or an actual load.

## §1f The relation axis (edges between occurrences)

The three axes are per-occurrence. Modern stacks also carry **relations**
between occurrences that a "×N identical layers on one residual line" diagram
cannot express (`13` §2, all measured). Every relation gets one typed row:

| relation kind | example (measured) | primary detector | supporting |
|---|---|---|---|
| `param_share` | Gemma-2/3n/4 `lm_head.weight is embed_tokens.weight` | parameter object identity (instance) | config `tie_word_embeddings`, source assignment |
| `activation_reuse(from→to, what)` | Gemma3n KV sharing: tensors born in layer 18 consumed in layers 20–34 | tensor lineage across layer hooks (trace) | source reader `kv_sharing_schedule` (already produces `CrossLayerEdge(kind="kv_share")`) |
| `multi_stream_residual(n)` | DeepSeek-V4 mHC `[B,S,4,D]`; Gemma3n AltUp `[4,B,S,D]` | hidden-state rank/shape at the layer boundary (trace) + per-layer mixer modules (`attn_hc`, `ffn_hc`, `altup`) (instance) | source reader for the mixing algebra (Sinkhorn `comb`, `pre`/`post`) |
| `per_layer_side_input` | Gemma3n/4 `per_layer_inputs` sliced per layer from a pre-stack projection | pre-stack tensor consumed inside layer ≥1 (trace) + `per_layer_model_projection` (instance) | source |
| `intra_layer_shortcut` | LongCat-Flash: shortcut MoE output added after the second sublayer | source reader (residual algebra) | trace add-lineage |
| `layer_reuse` (looped/universal) | not yet probed | same module object invoked at ≥2 stack positions (trace) | source loop |
| `conditional_skip` | not yet probed (MoD-style) | trace under recipe + source guard | — |
| `side_head` | DeepSeek-V4 `hc_head`, MTP heads (`model.mtp.*` ignored on load) | instance (non-layer module beside the stack) | source |

Laws: the IR already carries `ModelIR.cross_layer_edges` (`ir.py:390`);
the relation axis generalises that field rather than adding a second one; a
relation detected by trace or instance with no source explanation is
`relation_unresolved` and **visible**; a non-layer module beside the stack
with no relation row and no projection disposition is a denominator failure.
Heterogeneous stacks (Qwen3-Next 5:1, Jamba, Nemotron-H's aperiodic
M/MoE/A/MLP string, DeepSeek-V3.2 3+58, Llama-4 `layer_types`, gpt-oss
sliding/full) are already representable: the IR holds a per-layer list,
`distinct_layer_groups` collapses by signature in encounter order, and
`detect_layer_period` returns `None` for aperiodic schedules (`ir.py:468-500`).

## §1e Isolation (engineering, not policy)

Subprocess; timeout; memory ceiling; network off by default; stdout/stderr
captured; deterministic environment record; typed failure; **no fallback to a
familiar structure**. Remote custom code opt-in under the same mechanism.

---

## §2 The order

### S0 — hygiene (day 1)
Rotate + scrub the two HF tokens (`finalize.ipynb` cell 1, `done/tryrun.ipynb`
cell 18); notebook secrets scan in pre-commit; version-control `z-docs/` and
`.claude/PROTOCOL.md` (curate `done/`; exclude secrets and generated
artifacts); commit-body hook; archive the 12 superseded docs.
**Exit:** token grep empty; empty commit bodies rejected; z-docs tracked.

### S1 — close U11-F (days 1–2, the running agent)
Finish the committed-tree coordinator on `9b3cb7b`/`9a4e1e5`; if green, push;
set the §10 tracker row to F4/F2c DONE with the receipt id. **Do not start
G/H/I.**
**Exit:** upstream tip contains `9a4e1e5`; tracker row correct.

### S2 — latency (days 1–3)
`_SourceWalker._seg` once-per-file line split; memoise
`decoder_block_candidates_for_config`, `resolve_component_root`,
`everchanging.load`.
**Exit (two budgets):** (a) `_seg` proven linear and output-identical on the
5 profiled targets (fingerprints unchanged); (b) ProgramIndex construction
*after imports* and end-to-end cold latency each measured on a quiet machine
and recorded as the baseline; budgets are set from that baseline, not
asserted in advance. Library import (measured 6.5 s for torch + diffusers +
transformers) is outside our budget.

### S3 — consumer honesty (days 2–6)
Remove the verified unknown→known sites (`graph.py:105`, `op_render.py:287`,
`metadata_modalities.py:494/668/897`), the label sniff (`labels.py:787`), the
invented literals (`blocks/model.py:120,165,268`), the phantom expert boxes
(`feed_forward.py:582-608`), the fixed sliding-window depiction, and the
placeholder text tower for bare diffusion configs (live today: a plain
`unfold(dict)` on SD3.5 returns `stack.kind = decoder_only, num_layers = 0`
with `tok_text / embed / final_rms / lm_head` blocks); add the visible
zero-layer warning. Extend `consumer_firewall` with a blocking "no semantic
default resolves an unknown" rule.
**Exit:** rule blocking with poison test; preservation delta enumerated per
witness and approved before re-bless; every removed site yields a chip.

### S4 — recall and unknown-rate gates, no authority change (weeks 1–2)
Directed recall ratchet (proven → unresolved requires a named re-proof;
labels in the fixture signature); no waiver without a chip; flip
`config_accessed_unprojected` / `asserted_facts` to blocking; wire
`census.py --check`; `zero_asserted_census` must not swallow exceptions;
receipts in-repo; `bless` requires a persisted verdict from a reviewer that
did not implement the change; per-model `proven / flagged / silent`
published, silent must be 0.
**Exit:** each gate has a poison test that fires; `coverage.json` committed.

### S5 — ship v0.3.0, an **honesty/reliability** release (end of week 2)
Not a completeness release. README publishes: the supported corpus; known
incomplete structure (SD3.5 / PixArt denoiser opaque if still so); the
proven / flagged / silent counts from S4; one version string. Space unpinned
and redeployed with typed refusals visible; `examples/` regenerated at HEAD
with chips; `deepseek-v3.html` is DeepSeek.
**Exit:** `pip install model-unfolder==0.3.0` renders the 29 witnesses; the
Space renders 5 live; README limitations match `coverage.json`.

### S6 — instance-oracle pilot, no production consumer (weeks 2–4)
`physics/instance_inventory.py`: meta construction in a bounded subprocess
(§1e); typed tree = path, class, origin module, MRO, children, repetition
groups by child signature, parameter shapes, parameter aliasing groups,
`__init__`-set attributes, guarded children set to `None`; §1c records; typed
failures, never a fallback.
`physics/execution_observation.py`: FakeTensor forward under a named recipe;
module pre-hooks **plus** a `TorchFunctionMode` log (this sees SDPA, silu,
gelu, chunk, cat, layer_norm and the residual `add` / gate `mul` arithmetic
that module hooks miss — verified §7); positive evidence only; typed failure
on data-dependent models; lazily constructed modules recorded.
Pilot: Llama, DeepSeek-V3, Qwen2-VL (text + vision), SD3.5, PixArt, SDXL,
MusicGen, DBRX.
**Exit:** all 8 produce a typed inventory or typed failure; version-pinned
receipt per model; zero production imports of the new modules.

### S7 — reconciliation layer + shadow comparison (weeks 3–6, boxed)
Implement §1/§1b as `evidence/reconciliation.py`: joins inventory,
observation, static owner graph, config document into the three-axis table
with two-keyed provenance, feeding the **existing** fact layer (no parallel
IR). Static closure starts **from the exact runtime-selected class** rather
than a guessed bundle root, and remains demand-driven for helpers, functional
calls, decorators, processors and closures outside the class file. Run in
**shadow** on all 29 witnesses + 10 out-of-corpus `TO_SERVE.md` models.
Publish the disagreement matrix per model. Add a Sable net that blocks on any
occurrence with an unresolved axis. **No pixel changes.**
**Exit (3-week box):** every occurrence on 39 models has all axes assigned;
matrix committed and read by Soumil; first cutover family named.
**Lawful escape:** if the box expires, S8 may begin **only** for a family
whose complete occurrence denominator is closed, whose three-axis table has
zero unclassified occurrences, and whose matrix is approved. Unclassified
occurrences in other families stay visible work and cannot be used by that
cutover.

### S8 — first family cutover with differential testing (weeks 6–9)
One family (expected UNet, so G′ has a home) to the reconciled authority; old
path kept behind a flag for exact differential testing; every difference is a
named re-proof or a blocking finding. U11-G′ = thin projection of the
three-axis table; D/E/F readers re-pointed at exact cell classes; each of the
14 `unet_*` readers gains a production caller or is deleted.
**Exit:** SDXL renders full down/mid/up stages with cross-attention proven,
chipped or `not_constructed(guard)`, never manufactured; differential report
has zero unexplained differences; old path unreached.

### S9 — DiT / text families + restoration (weeks 8–12, ships v0.4.0)
Rival-invocation pruning with the consumed config value; readers on exact
instances (closure from the exact class: mixin lane, processors,
`canonical_import`, partial cell, heterogeneous `FeedForward.net`, two-lane
FFN); the lost-detail restoration list (`10` §3-4) with an owner each;
negatives proven under closure; `Conv1D`, slice-unpack, `GELU(tanh)` honest.
**Exit:** diffusion tier-1 100 % on the 15 diffusion witnesses (SD3.5 ×38 with
the 37/1 split, PixArt real class); SD3.5 FFN chip filled; DeepSeek score
scale correct; the 10 out-of-corpus models ≥ 90 % proven, 0 silent.

### S10 — delete replaced responsibilities (rolling from S8, each its own unit)
Name the responsibility, its replacement, show corpus + out-of-corpus parity,
delete, receipt. `adapters/diffusor/unet.py` structural authorship and the
renderer's UNet graph authorship (U11-H) first.
**Exit per unit:** parity receipt + deletion in one commit; no fallback
reachable.

### S11 — U12–U15 on the hybrid (weeks 10–14, ships v0.5.0)
VAE and scheduler skeletons from the inventory + mechanism readers; U14
firewall rows → 0; U15 YAML → syntax only, dead authorities deleted,
duplicate helpers unified, `parse()` split observe → bind → interpret →
project.
**Exit:** empty firewall register with anti-vacuity poison; one unknown
vocabulary.

### S12 — the learner product (∥ from S4, design-led, ships v1.0)
One-page spec as acceptance test; Her Eyes blocking per bless; `unfold-npm`
renderer-only over engine JSON; Space serves cached JSON in seconds.

---

## §3 The verification law (every step)

1. Broad gate on the committed tree in an isolated worktree; receipt in-repo.
2. Every output delta enumerated per witness with its evidence-level cause,
   approved by Soumil **before** re-bless; superseded signature recorded.
3. Coverage denominator regenerated; proven count never drops without a
   named re-proof.
4. Reconciliation net: zero unresolved axes on any occurrence (from S7 on).
5. Commit body non-empty by hook; shipping steps carry tag + PyPI + Space.

## §4 Message to the U11 agent

> Finish the coordinator on F4/F2c (`9b3cb7b`, `9a4e1e5`); if green, push and
> set the §10 tracker row to DONE with the receipt id. Do not start G/H/I.
> G resumes as G′ over the reconciled three-axis authority
> (`z-docs/10-full-research/12-execution-order.md` S8). Your D/E/F readers
> are kept and re-pointed at exact cell classes. Do not add another
> structure-authoring reader.

## §5 Still Soumil's
1. Ratify the tier contract (one line).
2. Approve the one-page learner spec when written (S12 start).
3. Read the S7 disagreement matrix and name the first cutover family.
S0–S6 need none of these.

---

## §6 First-principles bets (what is provable; what would falsify it)

| # | bet | confidence | why provable | falsifier |
|---|---|---|---|---|
| 1 | For a fixed (config, library revision, code), the eagerly constructed module tree is a deterministic function; meta instantiation enumerates it exactly. | ~99 % | construction is code execution on a fixed input | `__init__` reading environment/clock/filesystem; lazily built modules. Both are typed axis values (`lazy_observed`), never fallbacks. |
| 2 | The exact class object dissolves **class and owner discovery** across the bundle boundary: runtime factory remapping, mixins, the initial class/MRO source, the runtime-selected processor. It does **not** supply mechanism closure: imported helpers, functional calls, decorators, dynamically selected callables and out-of-file closures still need demand-driven static closure, which now *starts from* the exact class instead of a guessed bundle root. | ~99 % for discovery; closure remains a separate system | the object is the resolution; verified §7 (processor `JointAttnProcessor2_0` and `_chunked_feed_forward` resolved by object; `gate_msa.unsqueeze` etc. are tensor-local and need the static reader) | dynamically generated classes without retrievable source → typed `source_unavailable`. |
| 3 | Static source is the primary authority for **custom** mechanism semantics, alternatives and negatives; "never executes" is a property of program text that no finite trace set can establish. Closed framework primitives, parameter shapes and positive observations are primary for what they definitionally answer. | 100 % (logic) | existential vs universal claims | none. |
| 4 | Negative proofs are valid only under a **closure** premise; closure is supplied by the demand-driven static closure system of bet 2's second half, not by the class object alone. | 100 % | undecidability of all-paths analysis | none. |
| 5 | Trace is positive evidence only. | 100 % | recipe-specific by construction | none. |
| 6 | Config is the authority for raw declared values; derived values need constructor expression + operand provenance, with the instance confirming; shapes are the instance's. | 100 % | provenance is where a number came from | none. |
| 7 | Reconciliation without per-question precedence becomes a fifth judgment engine; the matrix must be code and the axes orthogonal. | ~95 % | the FFN/position/projection divergences | if a matrix-enforced single table still diverges across consumers, the matrix is wrong, not the principle. |
| 8 | The instance inventory (plus lazily observed modules) is the first independent recall denominator the project has had. | ~95 % | every existing net reads the parser's own AST | a shared blind spot with the static graph on a module built lazily *and* never exercised by any recipe → surfaces as `execution_unresolved`, still visible. |
| 9 | Shadow mode without a time box does not reach production. | ~90 % | 14 `unet_*` readers, zero callers | none needed if S7 closes early. |
| 10 | Which static construction code to delete, in what order. | **open** | decided by the S7 matrix | receipts. |
| 11 | What a learner needs to see. | **not provable by an engine** | product judgment | the one-page spec + Her Eyes. |

S0–S6 rest only on bets 1–6. S7 tests 7–8. Bet 10 is placed only after S7.

---

## §7 What would have been achieved — SD3.5 Large, measured 2026-09-02

(Hard-model edge cases — multi-stream residual, KV sharing, hybrids — are in
`13-hard-models-and-relations.md`; the acceptance checks per step are in
`14-confirmation-checklist.md`.)

Same checkpoint config from `tests/sable_test_corpus/`. Left column is
today's corpus route (`_coerce → ParseContext.build → config_to_ir`, 42 s
with code inspection). Right column is the v2.2 authority on this machine
(torch 2.9.1, diffusers 0.38.0), scripts `scratchpad/demo/meta_demo.py`,
`v22_test.py`.

| element | today (static route) | v2.2 (authority that proved it) |
|---|---|---|
| denoiser stack | one opaque box: "Repeated denoiser structure unresolved", `num_layers` absent, 169 leaf fields total | 38 `JointTransformerBlock`, 8.06 B params, 1,456 modules (**instance**, 2.3 s) |
| per-layer heterogeneity | none | layers 0–36 have `ff_context` + AdaLN-Zero context; layer 37 has `AdaLayerNormContinuous` and no `ff_context` = `context_pre_only` branch fired (**instance**) |
| block children | none | norm1, norm1_context, attn, norm2, ff, norm2_context, ff_context with every parameter shape (**instance**) |
| attention internals | none | to_q/k/v + add_q/k/v_proj 2432×2432 with bias; `norm_q/norm_k/norm_added_q/k` = diffusers `RMSNorm` weight[64] ⇒ **QK-norm present, head_dim 64** (**instance shapes**; mechanism from `normalization.py:510` via **static reader on the exact class**) |
| attention op order | none | to_q → to_k → to_v → norm_q → norm_k → add_q → add_k → add_v → norm_added_q/k → cat ×3 → `scaled_dot_product_attention` → to_add_out (**trace**, 0.07 s, recipe: batch 1, 16 image + 8 text tokens, eval, joint text) |
| runtime-selected processor | not reachable (out-of-bundle) | `JointAttnProcessor2_0` @ `attention_processor.py:1422`; its `__call__` closure (14 `attn.*` calls, `F.scaled_dot_product_attention`, `torch.cat`) (**exact class object → static closure**) |
| FFN | none (the grey chip Soumil refused) | 2432 → 9728 → 2432, ×4, `GELU(approximate="tanh")` (**instance shapes + exact primitive attributes**); `_chunked_feed_forward` helper resolved by object to `attention.py` |
| AdaLN-Zero | none | `norm1.linear` out = 14592 = 6 × 2432 ⇒ six modulation params; silu → linear → chunk → layer_norm observed (**instance + trace**) |
| residual and gating arithmetic | none | 16 `add`, 16 `mul` events seen by the function mode; module hooks alone would have missed them (**trace**) |
| dual attention (`attn2`) | n/a | `dual_attention_layers` unset ⇒ constructor set `attn2 = None`; `not_constructed(guard)`; trace shows no attn2 call (**instance negative + trace corroboration**; static reader sees only `if self.use_dual_attention`) |
| framework primitives | n/a | `nn.Linear/SiLU/LayerNorm/ModuleList` classified by exact runtime type; `AdaLayerNormZero`, `Attention`, `RMSNorm`, `FeedForward` routed to source (**§1b split**) |
| weight tying (control: Gemma-2 / Llama) | boolean from config | Gemma-2 `lm_head.weight is embed_tokens.weight → True`, one aliased group; Llama `False`, none (**parameter identity**, tier 1) |
| class remap (control: PixArt-Σ) | bundle root `transformer_2d.py`, wrong file | `_class_name: Transformer2DModel` constructs `PixArtTransformer2DModel` @ `pixart_transformer_2d.py` (**factory**, in scope by §1d) |

What v2.2 still does **not** claim from this run: the residual *equation*
(which add feeds which), the exact modulation algebra, and why the
attention is joint rather than cross. Those remain the static reader's on the
exact class, per bet 3, and are the S7/S9 work.
