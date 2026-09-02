# U3-A2 — Execution-Flow Observation Prerequisite: Corpus Reconnaissance

**Status: RECON REPORT ONLY — no implementation.** Held for Codex review. This
report establishes that a new *neutral* observation kernel is required before any
execution-order graph can be built, and proposes the smallest additions.

## Method
For each of the 26 witnesses in `tests/sable_test_corpus/*.json` the real pipeline
resolved a **target model-stage forward**: `resolve_component_root` → (when the
root resolved) `resolve_declared_model_stage`; the target class is B1's model-stage
child when B1 resolved (10 transformers), else the D0 root class (15 diffusion
`self_or_root(failed)`, 1 `self_or_root(absent)` musicgen). Its `forward` (or
`__call__`) body was parsed and classified against 11 execution-flow patterns.
Script: `scratchpad/a2_recon.py`.

## Per-witness pattern presence (counts)
| witness | target | case | patterns present |
|---|---|---|---|
| auraflow-v0-3 | AuraFlowTransformer2DModel | root(failed) | P1×9 P3×2 P6×2 P7 P8×3 P9 P10 |
| bloom | BloomModel | model-stage | P1×4 P3 P6 P7 P8 P9 |
| cogvideox-5b | CogVideoXTransformer3DModel | root(failed) | P1×10 P3 P6×3 P7 P8×3 P9 |
| deepseek-v3 | DeepseekV3Model | model-stage | P1×3 P4 P6 P7 |
| flux-2-dev | Flux2Transformer2DModel | root(failed) | P1×15 P3×2 P6×5 P7 P8×2 P9 P10 |
| fluxtransformer2dmodel | FluxTransformer2DModel | root(failed) | P1×10 P3×2 P6×3 P7 P8×4 P9 P10 |
| gemma-2-2b-it | Gemma2Model | model-stage | P1×3 P4 P6 P7 |
| glm-4-5 | Glm4MoeModel | model-stage | P1×3 P4 P6 P7 |
| gpt-oss-20b | GptOssModel | model-stage | P1×3 P3 P6 P7 |
| hunyuanvideo | HunyuanVideoTransformer3DModel | root(failed) | P1×8 P3×4 P6×2 P7 P8×7 P9 P10 |
| llama-7b | LlamaModel | model-stage | P1×3 P4 P6 P7 |
| ltx-video | LTXVideoTransformer3DModel | root(failed) | P1×7 P3 P4 P6 P7 P8×2 P9 |
| lumina-image-2-0 | Lumina2Transformer2DModel | root(failed) | P1×5 P3×3 P6 P7 P8×3 P9 |
| mochi-1-preview | MochiTransformer3DModel | root(failed) | P1×6 P3 P6 P7 P8×4 P9 |
| musicgen-small | MusicgenForConditionalGeneration | root(absent) | P1×4 P6×3 P7 P8 |
| olmo-2-1124-7b | Olmo2Model | model-stage | P1×3 P4 P6 P7 |
| pixart-sigma-xl-2-1024-ms | Transformer2DModel | root(failed) | P1×7 P3 **P5** P6×7 P7 P8×4 P9 |
| prxpixel-t2i | PRXTransformer2DModel | root(failed) | P1×6 P3 P6 P7 P8 P9 |
| qwen-image | QwenImageTransformer2DModel | root(failed) | P1×9 P3 P6 P7 P8×4 P9 P10 |
| qwen2-vl-7b-instruct | Qwen2VLModel | model-stage | P1×7 P6×6 P7 P8×2 P10 |
| qwen3-8b | Qwen3Model | model-stage | P1×3 P4 P6 P7 |
| sana-1600m-1024px-diffusers | SanaTransformer2DModel | root(failed) | P1×8 P3×2 P6×3 P7 P8×4 P9 P10 |
| stable-diffusion-3-5-large | SD3Transformer2DModel | root(failed) | P1×7 P3 P6×2 P7 P8×4 P9 |
| stable-diffusion-xl-base-1-0 | UNet2DConditionModel | root(failed) | P1×13 P3×2 P6×6 P7 P8×3 P9 P10 |
| stablelm-2-1-6b | StableLmModel | model-stage | P1×3 P3 P6 P7 |
| wan2-2-t2v-a14b-diffusers | WanTransformer3DModel | root(failed) | P1×6 P3×2 P6 P7 P8×5 P9 |

## Prevalence
| pattern | count | note |
|---|---|---|
| P1 direct addressed child call | **26/26** | universal |
| P2 exact `Sequential` call | 0/26 | model-stage forwards never run a top-level `Sequential` |
| P3 `ModuleList` iteration (`for x in self.layers`) | **18/26** | the layer-stack loop |
| P4 indexed/subscripted access (`self.layers[i]`) | 7/26 | mostly LLMs |
| P5 helper-delegated stack | 1/26 | pixart |
| P6 branch-conditional child call | **26/26** | universal (grad-checkpoint / optional paths) |
| P7 alias/reassignment chain | **26/26** | universal (`hidden` threaded/rebound) |
| P8 tuple/list unpacking | **18/26** | attention returns tuples |
| P9 early return(s) | 16/26 | multiple exit paths |
| P10 repeated invocation of same child | 8/26 | same field called ≥2× |
| P11 unsupported/dynamic | 0/26 | **under-counted — see caveat** |

## Capability matrix — can CURRENT records prove it?
Records available today: `CallObservation` (`lexical_order`+`guard`+callee+receiver),
`ControlRecord` (branch/loop+controlling), `DataflowObservation` (producer→consumer
op, unlabelled), `ConstructionSite`/`ContainerElementsRecord`, `GuardStep`.
Verdicts: ✅ provable · 🟡 partial · ❌ missing.

| pattern | invocation identity | def→use binding | container-iter binding | guard/path | happens-before | negative completeness |
|---|---|---|---|---|---|---|
| P1 direct call | 🟡 (self.field known; no InvocationId) | 🟡 | – | ✅ guard | 🟡 (only via def-use) | ❌ |
| P2 Sequential | 🟡 | – | ❌ (framework protocol not encoded) | ✅ | ❌ | ❌ |
| P3 ModuleList iter | ❌ (`layer` loop-var unbound to elements) | 🟡 loop-carried | ❌ **core gap** | ✅ | ❌ | ❌ |
| P4 indexed | ❌ (subscript→element unbound) | 🟡 | ❌ | ✅ | ❌ | ❌ |
| P5 helper-deleg | 🟡 (needs transitive invocation) | 🟡 | ❌ | ✅ | ❌ | ❌ |
| P6 branch-cond | 🟡 | 🟡 | – | ✅ **covered** | 🟡 | ❌ |
| P7 alias/reassign | – | ❌ (no exact-target SSA) **core gap** | – | 🟡 | ❌ | ❌ |
| P8 unpacking | – | ❌ (no tuple-target assignment record) | – | 🟡 | ❌ | ❌ |
| P9 early return | – | 🟡 | – | 🟡 | ❌ | ❌ (no return control record) |
| P10 repeat same | ❌ (no CallSiteId) | 🟡 | – | ✅ | ❌ | ❌ |
| P11 unsupported | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ (invisible, not typed) |

**Reading:** `guard/path` is the ONE broadly-covered capability. `invocation
identity` is partial (direct calls only). `def→use binding`, `container-iteration
binding`, `happens-before`, and `negative completeness` are largely **missing** —
and they gate the universal/majority patterns (P1, P3, P6, P7, P8). Therefore an
execution-order graph cannot be built on the current substrate.

## Conclusion — a new NEUTRAL kernel is required
The gaps cluster on five capabilities that the dominant patterns need. The
smallest neutral observation additions (observation-only, no roles, no order
claims baked in):

1. **`CallSiteId`** — exact, stable identity for each call site (fixes P10; anchors
   every invocation). `= (enclosing_callable, span, ordinal)` mirroring
   `ConstructionSiteId`.
2. **`InvocationObservation`** — one invocation node: exact callable + `CallSiteId`
   + receiver + resolved callee target + guard path. Replaces lexical
   `CallObservation` as the ordering unit (P1/P3/P4/P5/P10).
3. **Assignment/definition records with EXACT targets** — SSA-like def records
   including tuple/list unpack targets, so a value can be threaded through
   reassignment and unpacking (P7/P8) to yield exact def→use edges.
4. **Loop target + iterable + body relationship** — binds `for layer in
   self.layers` to the container's elements (P3) and enables the *exact container
   iteration* proof kind; subscript→element binding for P4.
5. **Return/branch control-flow records** — exit paths and branch alternatives tied
   to the flow (P6/P9), the basis for negative completeness.
6. **Typed unsupported regions** — explicitly mark forms not observable (dynamic
   dispatch, `getattr`-call, dynamic subscript), so "these are ALL the invocations"
   is *provable-or-explicitly-incomplete* rather than silently assumed (P11).

## Eventual order-graph contract (for the LATER unit, not now)
- `InvocationId = exact callable + exact call site (CallSiteId) + addressed owner
  occurrence`.
- `HappensBeforeEdge(source, target, proof_kind, spans, guard_path)`.
- Allowed `proof_kind` initially: **exact def-use dependency**; **exact framework
  execution protocol** (e.g. a correctly bound `torch.nn.Sequential`); **exact
  container iteration** (only once the loop target/iterable binding is proven).
- **`ModuleDict` never implies execution. `ModuleList` construction order alone
  never implies execution.**
- Output: proven edges; conditional edges; **unresolved pairs**; unsupported
  regions; completeness status. **Never call an unproven pair unordered.**

## Caveats (honest limits of this recon)
- **P11 = 0 is under-counted.** The classifier flagged only `getattr(self,…)()` and
  non-Name/non-Constant subscript keys. Real forwards contain more unobservable
  forms (dynamic dispatch, comprehension-built calls, cross-module helper hops);
  the true count is ≥ shown. This is itself the argument for **typed unsupported
  regions** — the kernel must make the unobserved *visible*, not zero.
- **P2 = 0 is real at this altitude** (top-level model-stage forward) but Sequential
  is used deeper (projectors/patch pipelines); the framework-protocol proof kind is
  still needed for those later.
- Diffusion targets are the D0 **root** class (B1 doesn't resolve a `base_model_prefix`
  stage there); their forwards are the transformer-block loops, which is the
  relevant ordering surface.
- Classification is *presence* (does the pattern occur), not per-invocation
  coverage; the kernel design above is driven by which capabilities the present
  patterns require, not by raw counts.

---

# Appendix A — Occurrence-Exact Frozen Worklist (26 witnesses)

Generated from the **Phase-2 execution records** (`program_index.py`) over the
model-stage forward of each witness: the B1 model-stage occurrence where B1
resolves (10 transformers), else the D0 root occurrence — which for diffusion is
**explicitly authorized by the diffusion adapter, never inferred from a B1
failure** (the `D0 root (authorized)` basis denotes that authorization). Columns:
exact execution owner · owner-selection basis · exact callable · direct call
sites · bindings (with target shapes) · loops `total(for-resolvable)` where
*for-resolvable* = for-loops whose iterable is a directly-bound container
attribute · returns · control transfers · unsupported execution regions ·
no-unsupported-region flag (necessary, NOT sufficient, for negative completeness).

| witness | owner | basis | callable | sites | binds (shapes) | loops(for-res) | rets | transfers | unsupported | no-unsup-region |
|---|---|---|---|---|---|---|---|---|---|---|
| auraflow-v0-3 | AuraFlowTransformer2DModel | D0 root (authorized) | forward | 27 | 22 {tuple:3, name:19} | 2(0) | 2 | {return:2} | {} | yes |
| bloom | BloomModel | B1 model-stage | forward | 15 | 23 {name:22, tuple:1} | 1(0) | 2 | {raise:1, return:2} | {} | yes |
| cogvideox-5b | CogVideoXTransformer3DModel | D0 root (authorized) | forward | 25 | 25 {tuple:3, name:22} | 1(0) | 2 | {return:2} | {} | yes |
| deepseek-v3 | DeepseekV3Model | B1 model-stage | forward | 11 | 10 {name:10} | 1(0) | 1 | {raise:1, return:1} | {} | yes |
| flux-2-dev | Flux2Transformer2DModel | D0 root (authorized) | forward | 36 | 40 {name:35, attribute:1, subscript:2, tuple:2} | 2(0) | 4 | {return:4} | {} | yes |
| fluxtransformer2dmodel | FluxTransformer2DModel | D0 root (authorized) | forward | 33 | 24 {name:20, tuple:4} | 2(0) | 2 | {return:2} | {} | yes |
| gemma-2-2b-it | Gemma2Model | B1 model-stage | forward | 14 | 12 {name:12} | 1(0) | 1 | {raise:1, return:1} | {} | yes |
| glm-4-5 | Glm4MoeModel | B1 model-stage | forward | 11 | 10 {name:10} | 1(0) | 1 | {raise:1, return:1} | {} | yes |
| gpt-oss-20b | GptOssModel | B1 model-stage | forward | 14 | 12 {name:12} | 1(0) | 1 | {raise:1, return:1} | {} | yes |
| hunyuanvideo | HunyuanVideoTransformer3DModel | D0 root (authorized) | forward | 25 | 29 {tuple:7, name:22} | 4(4) | 2 | {return:2} | {} | yes |
| llama-7b | LlamaModel | B1 model-stage | forward | 11 | 10 {name:10} | 1(0) | 1 | {raise:1, return:1} | {} | yes |
| ltx-video | LTXVideoTransformer3DModel | D0 root (authorized) | forward | 20 | 17 {name:15, tuple:2} | 1(1) | 2 | {return:2} | {} | yes |
| lumina-image-2-0 | Lumina2Transformer2DModel | D0 root (authorized) | forward | 25 | 20 {tuple:3, name:14, subscript:3} | 5(3) | 2 | {return:2} | {} | yes |
| mochi-1-preview | MochiTransformer3DModel | D0 root (authorized) | forward | 17 | 16 {tuple:4, name:12} | 1(0) | 2 | {return:2} | {} | yes |
| musicgen-small | MusicgenForConditionalGeneration | D0 root (authorized) | forward | 15 | 20 {name:15, subscript:4, tuple:1} | 1(0) | 1 | {raise:1, return:1} | {} | yes |
| olmo-2-1124-7b | Olmo2Model | B1 model-stage | forward | 11 | 10 {name:10} | 1(0) | 1 | {raise:1, return:1} | {} | yes |
| pixart-sigma-xl-2-1024-ms | Transformer2DModel | D0 root (authorized) | forward | 16 | 15 {name:11, tuple:4} | 1(1) | 2 | {return:2} | {} | yes |
| prxpixel-t2i | PRXTransformer2DModel | D0 root (authorized) | forward | 12 | 11 {name:10, tuple:1} | 1(1) | 2 | {return:2} | {} | yes |
| qwen-image | QwenImageTransformer2DModel | D0 root (authorized) | forward | 30 | 25 {name:20, tuple:4, subscript:1} | 1(0) | 2 | {return:2} | {} | yes |
| qwen2-vl-7b-instruct | Qwen2VLModel | B1 model-stage | forward | 15 | 11 {name:9, tuple:2} | 0(0) | 1 | {return:1} | {} | yes |
| qwen3-8b | Qwen3Model | B1 model-stage | forward | 14 | 13 {name:12, subscript:1} | 1(0) | 1 | {raise:1, return:1} | {} | yes |
| sana-1600m-1024px-diffusers | SanaTransformer2DModel | D0 root (authorized) | forward | 23 | 22 {name:18, tuple:4} | 2(0) | 2 | {return:2} | {} | yes |
| stable-diffusion-3-5-large | SD3Transformer2DModel | D0 root (authorized) | forward | 19 | 19 {tuple:4, name:15} | 1(0) | 2 | {return:2} | {} | yes |
| stable-diffusion-xl-base-1-0 | UNet2DConditionModel | D0 root (authorized) | forward | 42 | 52 {name:47, tuple:3, subscript:2} | 4(1) | 2 | {break:1, return:2} | {} | yes |
| stablelm-2-1-6b | StableLmModel | B1 model-stage | forward | 11 | 10 {name:10} | 1(1) | 1 | {raise:1, return:1} | {} | yes |
| wan2-2-t2v-a14b-diffusers | WanTransformer3DModel | D0 root (authorized) | forward | 33 | 28 {tuple:5, name:23} | 2(2) | 2 | {return:2} | {} | yes |

Aggregate control transfers: **26 return · 10 raise · 1 break**.

## The four separated capabilities (per observed construct)
| capability | verdict here | where it is proven |
|---|---|---|
| observed syntax | **YES** — the Phase-2 record exists | ProgramIndex (this phase) |
| address/binding resolved | only where an OwnerGraph child / import binding exists | Phase 3 (addressed invocation resolver) |
| execution relation proven | **NO anywhere** | Phase 4 (execution-flow resolver) |
| negative completeness proven | **NO** — "no unsupported region" is necessary, not sufficient | Phase 4 (needs all dispatch classified) |

## Corrections to the body report (frozen)
1. **Returns are OBSERVED today** (`ReturnObservation` + `ControlTransferObservation`), correcting the earlier "❌": every return is recorded with its guard path. What remains missing is **guard/control-flow COMPLETENESS** — proving which returns dominate a path.
2. **Tuple/list targets are OBSERVED** (`BindingObservation.targets` preserves the exact destructuring structurally). Still missing: **resolving the destructuring** (binding `a,b` to the call's outputs) and **versioned binding** — both Phase 4.
3. **`UnsupportedExecutionRegion` exists** (try/with/async-with/match/await) but is **incomplete for execution flow**: dynamic `getattr(...)()`, dynamic-subscript callees and comprehension-built calls are only partially surfaced. This incompleteness is deliberate and labelled — see (6).
4. **Guard coverage is PARTIAL, not complete**: branch/loop guards ride the guard path, but the container-**assignment** guard is not carried (the U3-B2 gap) and per-path guard completeness is unproven.
5. **The diffusion execution root is EXPLICITLY AUTHORIZED by the diffusion adapter** — never inferred from a B1 failure. The `D0 root (authorized)` basis records that explicit authorization; the resolver must be *given* the owner, it never selects one.
6. **P11 is NOT a certified zero.** At the model-stage forward level the EXACT observed unsupported set is `{try, with, async_with, match, await}` — 0 in most top-level forwards because these live in helpers / generation utils — PLUS a **labelled LOWER BOUND**: dynamic `getattr`/subscript dispatch and comprehension-built calls occur and are not yet fully classified, so the true unsupported count is a lower bound, never a misleading zero.

## Corpus signal that shapes Phase 3/4 (frozen)
- **Container iteration split**: diffusion forwards iterate a directly-bound container (`for block in self.blocks:` → for-resolvable > 0), whereas **LLM forwards iterate a SLICE** (`for layer in self.layers[: n]:` → the iterable is a `subscript`, for-resolvable = 0). Exact loop/container binding is therefore harder for LLMs; the Phase-3 resolver must leave sliced-iterable loops **unresolved**, never guessed.
- **Tuple-unpacking is pervasive** (attention returns), so exact destructuring def-use is the highest-value Phase-4 capability.
- **Early transfers block universal ordering**: 10 witnesses `raise` (LLM input validation), 1 `break` (SDXL) — these force conditional/partial results, never a certified complete graph.
