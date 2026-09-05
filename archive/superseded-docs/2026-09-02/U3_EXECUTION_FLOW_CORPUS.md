# U3 — Execution-Flow: an Open, Conservative Local-Relation Substrate (26 witnesses)

`resolve_execution_flow` run over each witness's execution owner (B1 model-stage
occurrence, else the diffusion-adapter-authorized D0 root). **No model/family
exception was added.**

**This is NOT a whole-callable execution-completeness result.** There is no CFG
coverage unit yet, so every callable result is `partial` / open. The `proven` and
`conditional` edges are valid LOCAL happens-before relations; `unsupported_regions`
and `loops` are PUBLISHED coverage gaps that are explicitly **non-exhaustive** —
they do not prove what remains uncovered. There is no `resolved`/`complete` status
and no closed-world `coverage` certificate anywhere in this campaign.

| witness | owner | status | addressed | templates | proven | conditional | unresolved(rel) | gaps: unsupp |
|---|---|---|---|---|---|---|---|---|
| auraflow-v0-3 | AuraFlowTransformer2DModel | partial | 2 | 0 | 0 | 0 | 1 | 2 |
| bloom | BloomModel | partial | 0 | 0 | 0 | 0 | 0 | 1 |
| cogvideox-5b | CogVideoXTransformer3DModel | partial | 0 | 0 | 0 | 0 | 0 | 1 |
| deepseek-v3 | DeepseekV3Model | partial | 2 | 0 | 0 | 0 | 1 | 1 |
| flux-2-dev | Flux2Transformer2DModel | partial | 9 | 0 | 3 | 2 | 4 | 2 |
| fluxtransformer2dmodel | FluxTransformer2DModel | partial | 1 | 0 | 0 | 0 | 0 | 3 |
| gemma-2-2b-it | Gemma2Model | partial | 3 | 0 | 0 | 0 | 3 | 1 |
| glm-4-5 | Glm4MoeModel | partial | 2 | 0 | 0 | 0 | 1 | 1 |
| gpt-oss-20b | GptOssModel | partial | 2 | 0 | 0 | 0 | 1 | 1 |
| hunyuanvideo | HunyuanVideoTransformer3DModel | partial | 4 | 0 | 0 | 0 | 0 | 1 |
| llama-7b | LlamaModel | partial | 2 | 0 | 0 | 0 | 1 | 1 |
| ltx-video | LTXVideoTransformer3DModel | partial | 1 | 1 | 0 | 1 | 0 | 1 |
| lumina-image-2-0 | Lumina2Transformer2DModel | partial | 2 | 3 | 0 | 8 | 1 | 1 |
| mochi-1-preview | MochiTransformer3DModel | partial | 1 | 0 | 0 | 0 | 3 | 1 |
| musicgen-small | MusicgenForConditionalGeneration | partial | 0 | 0 | 0 | 0 | 0 | 0 |
| olmo-2-1124-7b | Olmo2Model | partial | 2 | 0 | 0 | 0 | 1 | 1 |
| pixart-sigma-xl-2-1024-ms | Transformer2DModel | partial | 0 | 0 | 0 | 0 | 0 | 1 |
| prxpixel-t2i | PRXTransformer2DModel | partial | 2 | 1 | 0 | 1 | 1 | 1 |
| qwen-image | QwenImageTransformer2DModel | partial | 2 | 0 | 0 | 0 | 1 | 5 |
| qwen2-vl-7b-instruct | Qwen2VLModel | partial | 1 | 0 | 0 | 0 | 0 | 0 |
| qwen3-8b | Qwen3Model | partial | 2 | 0 | 0 | 0 | 1 | 1 |
| sana-1600m-1024px-diffusers | SanaTransformer2DModel | partial | 1 | 0 | 0 | 0 | 1 | 5 |
| stable-diffusion-3-5-large | SD3Transformer2DModel | partial | 0 | 0 | 0 | 0 | 0 | 1 |
| stable-diffusion-xl-base-1-0 | UNet2DConditionModel | partial | 0 | 0 | 0 | 0 | 0 | 7 |
| stablelm-2-1-6b | StableLmModel | partial | 1 | 1 | 0 | 1 | 1 | 1 |
| wan2-2-t2v-a14b-diffusers | WanTransformer3DModel | partial | 2 | 1 | 0 | 3 | 1 | 1 |

**Aggregate: 26 partial (all open) · 3 proven edges · 16 conditional edges · 7 repeated-invocation templates.** Plus per-forward published coverage gaps (unsupported regions above; loops carried in `res.loops`, not shown).

## Reading this substrate
- **Every result is `partial`/open.** The substrate proves LOCAL relations, never a
  complete callable ordering. `musicgen`/`qwen2-vl` have 0 published unsupported
  regions but are still `partial` — absence of a published gap is NOT a completeness
  claim (the gap list is non-exhaustive).
- **Local proven edges (3)** are exact straight-line versioned def-use (flux 3);
  **conditional edges (16)** are guarded def-use inside a proven branch (lumina 8,
  wan 3, flux 2, ltx/prx/stablelm 1). **Templates (7)** are loop-body-span-bound
  single-uniquely-proven-element containers.
- **Coverage gaps are published, non-exhaustive.** Unsupported regions surface
  IfExp/BoolOp/comprehension/lambda/chained-comparison/try/with/match/await/unknown
  statements (SDXL 7, qwen-image/sana 5, fluxtransformer 3, …). Loops are published
  in `res.loops`. This list does not prove exhaustiveness.
- **Unresolved relations** carry typed reasons: `transformed_reaching_definition`
  (producers preserved through a transform, never erased), `ambiguous_reaching_
  definition` (conflicting branches), `unresolved_alias_reaching_definition` (a
  failed/ambiguous alias source, preserved on the target).

## Consumption boundary
- **Do NOT migrate any reader that requires whole-callable completeness** — this
  substrate does not provide it.
- A reader needing only a **particular positively-proven LOCAL relation** (e.g. a
  specific proven `f -> g` def-use edge) may be proposed as its own separate unit.
- A genuine CFG coverage unit is a prerequisite for any completeness claim; until it
  exists, `resolved`/`complete`/closed-world coverage are intentionally absent.

## Per-pattern parity with the frozen A2 report
| A2 pattern | prevalence | realization |
|---|---|---|
| P1 direct child call | 26/26 | addressed where a proven `self.<field>` graph child exists |
| P3 ModuleList iteration | 18/26 | template only for a directly-bound single-uniquely-proven-element container (7) |
| P4 indexed access | 7/26 | `indexed_access_unproven` |
| P6 branch-conditional | 26/26 | conditional edges within a proven branch |
| P7 alias/reassignment | 26/26 | most-specific dominating def-use; transforms/aliases preserved as typed unresolved |
| P8 tuple unpacking | 18/26 | exact destructure; through-complex → unresolved |
| P9 early return | 16/26 | a coverage gap; result stays partial |
| P11 unsupported | non-exhaustive | published per-forward as coverage gaps |

## Recurring-error guardrail proofs (verified)
1. No identity reconstructed (CallSiteId.of derives; occurrences from the graph; ContainerAddress.record authoritative).
2. Round-trip to authority — edges carry producer+consumer spans (typed, source-consistent); nodes belong to the resolved callable; templates round-trip to their LoopObservation; `call_site == CallSiteId.of(call)`.
3. No self-certification — resolvers consume ProgramIndex/OwnerGraph/B2.
4. No global-name architecture selection — owner-scoped; loops bind by body-span.
5. No config/YAML-selected owner/mechanism — owner explicit; diffusion root adapter-authorized.
6. No unknown-kind fallthrough — typed buckets + published gaps; unresolved never `unordered`; unknown statements published, not silently assumed.
7. Closed status DTOs — unique census + exact partition; absent carries no call-site/graph payload; failed payload defined; node uniqueness; mutually-exclusive statuses; no completeness/coverage claim exists to over-assert.
8. Seam-attacking poisons — records + Phase-3 + Phase-4 poisons attack index↔resolver, resolver↔graph, resolver↔inventory, def-use↔ordering.
9. Corpus parity = compatibility only — soundness rests on the synthetic counterexamples; this table is a local-relation lower bound.
