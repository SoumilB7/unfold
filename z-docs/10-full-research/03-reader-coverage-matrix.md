# 03 — Reader coverage matrix

*Question:* for every architectural mechanism the product needs to draw, which reader proves it, from which exact code shapes, and what does it refuse — and how much of the Hugging Face ecosystem (and of the author's own target catalogue) do today's protocols actually accept?

Pass date 2026-09-02, tree `a00ae48`+ (branch `audio-composite-support`). Read-only; nothing rendered; nothing executed except `python3 -c` counts against the installed `transformers 5.12.1` / `diffusers 0.38.0` and one import of the package's own registries. Every statement not backed by a `file:line` or a counted number is marked **unverified**. Paths are relative to `unfold-pkg/` unless they start with `z-docs/`, `done/` or `/Library/...`.

---

## 0. What "reader" means here, and the numbers

The evidence layer is `model_unfolder/evidence/` — 131 modules, 96,898 lines (wc). The transformer parser is `adapters/transformer/parser.py` (4,996 lines); the diffusion parser `adapters/diffusor/parser.py` (1,036) + `unet.py` (971) + `projection_ir.py` (1,428) + `config_binding.py` (774).

A **reader** is a function that takes the one immutable `ProgramIndex` (`evidence/program_index.py:2483`) plus an already-resolved *owner occurrence* and returns a `ReaderResult[T]` (`evidence/reader_result.py:122`) whose status is one of `resolved | incomplete | ambiguous | absent | failed` and whose failures use the closed vocabulary `missing_source | parse_failure | unsupported_syntax | dynamic_dispatch | external_unavailable | unresolved_import | out_of_owner | incomplete_graph | conflict` (`reader_result.py:23-33`). A reader **never** collapses absence into `False`; a failure is non-consumable. The parser joins a resolved reader value to the U1 config-occurrence ledger (`evidence/config_access.py:1113 resolve`) and only then writes a spec field and a typed fact.

The *address rails* every mechanism reader sits on (all address-only, no mechanism semantics):

| rail | entry | what it proves |
|---|---|---|
| D0 component root | `component_owner.resolve_component_root` `evidence/component_owner.py:1531` | the declared architecture class → one exact class occurrence |
| B1 model stage | `resolve_declared_model_stage` `component_owner.py:1780` | the root's framework-declared stage occurrence (closed protocol, `component_owner.py:1605`) |
| B2 containers | `container_inventory.resolve_container_inventory` `container_inventory.py:246` | every `ModuleList/Sequential/ModuleDict` record owned by that exact occurrence; rivals typed, never picked |
| F2 repeated child | `repeated_child.resolve_repeated_child` `repeated_child.py:135` + `execution_flow.py:305/869` | the loop-bound element of **one cited homogeneous container** invoked in the stage forward |
| block path | `decoder_block.decoder_block_path_for_config` `decoder_block.py:426` | config path → root → stage → repeated child, shared by every decoder mechanism |
| nested config root | `config_scoped_owner.resolve_config_constructed_root` `config_scoped_owner.py:222` | accepts only direct field construction or `child = Child._from_config(config.child); self.slot = child`; anything else is `unsupported_config_construction` (`:312`) |
| return delegation | `delegated_stage.resolve_return_delegated_child` `delegated_stage.py:44` | only `return self.child(...)` or `r = self.child(...); return r` |
| output stage | `output_repeated_stage.resolve_output_repeated_stage` `output_repeated_stage.py:191` | child whose repeated container reaches a structured return |

Consequences the whole matrix inherits: **heterogeneous per-layer containers** (a `ModuleList` whose elements are built from different classes) fail F2 ("a template resolves only a single-element (homogeneous) container", `execution_flow.py:196`) unless a dedicated schedule reader (`layer_selector`, `cross_attention_replacement`) re-derives the selection; **ModuleDict never implies execution** (`execution_flow.py:2143`); **lexical order is never execution order**; every whole-forward claim is `incomplete` because U3 exposes no CFG coverage certificate (`diffusion_root.py:1845-1849`).

Counted facts used below:

| count | value | how |
|---|---|---|
| registered facts (`evidence/registry.py:251 REGISTRY`) | **62** (16 `diffusion_*`, 46 transformer/model) | import + len |
| facts with a projection route (receipt-validated) | 30 | `FactDefinition.projection_routes` |
| qualification rules (`evidence/qualification.py:53 QUALIFICATION_MATRIX`) | **17**, all `ir_scope="transformer_decoder"` | read |
| structural-debt rows (`evidence/structural_debt.py`) | **279** | import; grouped in §4 |
| quarantined legacy semantic readers still called in production | **5**, all UNet (`docs/U3_CURRENT_READER_INVENTORY.md:28-34`) | read |
| block views (`renderers/html/block_views/registry.py:177 VIEW_REGISTRY`) | 34 keys | read |
| blessed corpus witnesses (`tests/sable_test_corpus/*.json`) | **29** = 14 transformer + 15 diffusion | ls |
| installed `transformers` model types (`CONFIG_MAPPING_NAMES`) | **656** | python |
| installed torch modeling files (`models/*/modeling_*.py`, excl. tf/flax) | **466** in 484 model dirs | python |
| installed diffusers `*Transformer*Model` classes / UNet classes / autoencoder classes / scheduler files | **52** / 19 (6 top-level `UNet*Model`) / 28 (24 real VAEs) / 53 | python |

---

## 1. Mechanism inventory — the three sources merged

**(a) Fact registry** (`evidence/registry.py:251-1300`): the 62 keys, each with owner patterns, allowed statuses, `unknown_policy` and `notes` that name the proving unit. Their allowed statuses are the honest statement of what still comes from config: e.g. `mask` allows `config_declared`/`class_default` (`registry.py:652`), `mechanism` allows `class_default`/`ambiguous`/`oracle_missing` but *not* `code_proven` (`:609`), `tie_word_embeddings` is `class_default|config_declared` only (`:1142`), `sinks`/`cached`/`gated`/`output_projection`/`output_gate`/`embedding_norm_kind`/`position_addition`/`diffusion_root_topology`/`diffusion_stream_relation` are `code_proven`-only.

**(b) IR vocabulary** (`ir.py:16-238`): `AttentionSpec` carries 40 fields — of which `compress_ratio`, `index_topk`, `index_n_heads`, `index_head_dim` (DeepSeek-V3.2 DSA, `ir.py:94-97`), `mrope_section` (`:98`), `shared` (Zamba, `:79`), `no_rope` (`:80`), `rope_3d` (`:81`), `variant` (`:116`) and `cross_kv_source` (`:90`) have **no registered fact**; `FFNSpec` has `conv_glu` in its kind vocabulary (`:127`); `LayerSpec` carries `norm_placement ∈ pre|post|double` and `residual_topology` (`:224-231`); `ModelIR.cross_layer_edges` (`:419`) carries `kv_share` edges. Diffusion projection DTOs live in `adapters/diffusor/schema.py:65-297` and are *passive* over U10 evidence ("cannot independently assert MHA, a gated FFN, a stream role, or a temporal mechanism", `schema.py:1-13`).

**(c) View registry** (`block_views/registry.py:177-233`): `attention`, `ffn|moe|gated_ffn|dense_ffn`, `moe_router`, `topk_selection`, `dsa_indexer`, `scheduler_step`, `per_layer_embedding`, `vision_path|audio_path|video_path|conditioning_path`, `audio_encoder|video_encoder|vision_encoder`, `multimodal_fusion`, `mtp_head|mtp_transformer_block`, `self_conditioning`, `vae_decoder|vae_decoder_block`, `text_encoder`, `unet|unet_stage|unet_resnet|unet_transformer|encoded_text_concat`, `tower`, `ops`, `mla_query_path|mla_kv_cache_path`, `moe_expert`. The adapter child-block tables also still define `ssm`, `rwkv`, `linear` attention drills (`adapters/transformer/blocks/attention.py:97-100`) — see §4 for why they are unreachable.

Merged, the product needs to draw ~95 mechanisms in 15 families. The matrix follows.

---

## 2. The matrix

Column key — **Gate**: `Q` = row in the blocking qualification matrix (`qualification.py:53`); `R` = fact has a projection route validated by receipts (`registry.py`/`receipts.py:215`); `C` = registry census only (registered key, no route/rule); `—` = not a fact. **Witnesses**: corpus slug or real-source test controls (from grepping `tests/test_*.py`; e.g. OPT is the substrate control for the ProgramIndex/execution-flow suites, GPT-2/T5/BERT are real-source controls even though not blessed). "unverified" marks protocol behaviour I inferred from a failure string but did not confirm by running.

### 2.1 Component / pipeline

| mechanism | reader | accepted shapes | refusals (verbatim) | gate | witnesses |
|---|---|---|---|---|---|
| root component & nested component inventory | `component_inventory.resolve_component_inventory` `:111` | a component is `active` only with an exact config-scope construction **and** field installation; otherwise `declared_unused` | "an unresolved D0 root cannot be active" (`:100`) | — | llava, mllama, qwen2-vl, gemma4 (tests) |
| composite slots (MusicGen `decoder=main`, diffusers `text_encoder_N`, `transformer_2`) | config address only: `everchanging/transformer/composite_slots.yaml`, `everchanging/evidence/ledger_ignores.yaml:45-60` | slot key selects a sub-document | — | — | musicgen, flux, wan2-2 |
| repeated stages under a component | `component_stages.resolve_component_stages` `:114` | any exact owner positively invoking a repeated container; **no execution order** | — | — | qwen2_vl, idefics |
| diffusion pipeline handoffs | `adapters/diffusor/parser.py:522 _projected_pipeline_handoffs` | config | — | — | all 15 diffusion witnesses |

### 2.2 Model bookends

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| embedding-stage norm | `embedding_bookend.embedding_stage_norm_evidence` `:48` / `read_embedding_stage_norm` `:91` | one **unguarded** norm invocation with a def-use edge into the repeated child, both owned by the model stage; norm kind via `primitive_semantics` | "the exact model-stage forward is absent" (`:136`); positive-only, absence never fabricates (registry `:1161`) | C | bloom, gemma-2, llama |
| final norm | `final_bookend.final_stage_norm_evidence` `:39` | repeated child → norm → **every** primary return; bounded lineage alternatives | "lineage path alternatives exceed the bounded proof capacity" (`:332`); "the model-stage return has no exact primary hidden-state slot" (`:314`) | C | all 14 LLM witnesses |
| weight tying (manual) | `weight_tying.manual_weight_tying_for_path` `:168` | unguarded root `__init__` assignment `lm_head.weight = embed.weight` between an external Linear returned by root forward and an external Embedding feeding the repeated block | config-gated ties stay unknown → fact `tie_word_embeddings` is `config_declared`/`class_default` only (registry `:1142`) | C | llama |
| learned/fixed position add before the stack | `position_absolute.decoder_learned_absolute_position_for_path` `:100`; `position_fixed` `:111`; `position_table` `:54` (direct `.weight` add); `component_position` `:80` (multi-axis lookup in a nested stage) | coordinate origin (§2.6) → `Embedding` → addition → repeated child | "no code-proven coordinate value feeds an embedding primitive" (`position_absolute.py:183`); "a coordinate embedding was not unconditionally added into the stream reaching the exact repeated child" (`:212`) | Q+R (`position_addition`) | GPT-2 (learned), OPT poison (cumsum refused), bloom/llama negatives |
| embedding input scale (Gemma `√d`), logits divisor (Cohere `logit_scale`), final logit softcap | **no reader** — config reads carried as structural debt (`structural_debt.py` rows "the model-stage embedding-input scale", "the language-head logits divisor", `extras:softcap`, unit U14) | config | — | — | gemma-2 |
| LM head block itself | **no reader** (drawn unconditionally by `adapters/transformer/blocks/model.py`, `docs/blocks_supported.md:22-25`) | — | — | — | — |

### 2.3 Layer stack and per-layer schedules

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| layer count | B2 count citation (`container_inventory.py:1365-1368`) + `mixer_schedule.block_layer_index_transport` `:466` | `range(<exact config path>)` comprehension whose target becomes the block's `__init__` formal | "block comprehension is not a direct range" (`mixer_schedule.py:497`); "layer comprehension is not exact builtin range" (`:510`); "layer count is not exact positive config evidence" (`:519`) | — | all |
| per-layer construction selection (which class at index *i*) | `layer_selector.resolve_layer_selector` `:788` / `resolve_layer_field_schedule` `:657` | guarded construction sites whose guards evaluate from config operands at the exact index; candidate census derived inside the module | unsupported syntax, dynamic values, rival live sites → typed uncertainty | — | qwen3-5, gemma-2, recurrent (synthetic) |
| mixer schedule (attention vs gated-delta per layer) | `mixer_schedule.decoder_mixer_schedule_for_path` `:257` | index transport + selected child + **invoked** by block forward + a U6 mechanism (`ordinary_attention` or `gated_delta` only) | "no exact U6 mixer mechanism is available at this block" (`:385`); "not every layer selects and invokes one proven mixer" (`:446`) | Q+R | qwen3-5-27b-text, llama, qwen3 |
| FFN schedule (dense/MoE per layer) | `ffn_schedule.decoder_ffn_schedule_for_path` `:275` | same index transport; each layer's selected construction has a positive U7 ordinary **or** routed proof; uniform-repetition transport for homogeneous stacks | "not every layer selects and invokes one proven FFN mechanism" (`:538`); "one construction has rival mechanism proofs" (`:383`) | Q+R | deepseek-v3, glm-4-5, gpt-oss, dbrx, granite |
| head-geometry schedule | `attention_geometry.decoder_attention_geometry_schedule_for_path` `:256` | per-layer constructor fields that reach Q/K/V reshape and K/V-repeat sites | "the grouping field is not an exact query/KV quotient" (`:425`); "the exact grouping field does not reach both K/V repeat paths" (`:435`) | Q | gemma3, gemma4 (tests) |
| mask schedule + geometry | `attention_mask.decoder_attention_mask_execution_for_path` `:1315` (composes `:607`, `:879`, `:1055`, `:1184`, `:1277`) | **only** import-resolved framework builders — table `_MASK_PROTOCOLS` `attention_mask.py:69-82`: `create_causal_mask`, `create_bidirectional_mask`, `create_sliding_window_causal_mask`, `create_chunked_causal_mask`, `create_bidirectional_sliding_window_mask`; geometry fields `sliding_window` / `attention_chunk_size` from the builder's exact stage-config actual (`:85-95`); per-layer selection by exact `enumerate` index over a config list, or the Qwen3/Gemma-2 `causal_mask_mapping[layer.attention_type]` dict (`modeling_qwen3.py:413-418` is the witnessed shape); uniform schedule over the exact container count | "a block mask formal retains rival or conditional builders" (`:870`); "no exact mask formal reaches the selected self-attention score lane" (`:1023`); "the exact mask builder does not receive the stage config object" (`:1356`); local same-name function never qualifies (`:10-13`); runtime-supplied mask maps are an explicit boundary (`docs/U6_U7_U8_QUALIFICATION_MATRIX.md:238`) | Q+R | gemma-2, gpt-oss, llama, bloom, T5, BERT, GPT-2, qwen3 |
| mask direction fallback when no builder is proven | `decoderness.declared_decoderness` `:28` + `everchanging/transformer/decoderness.yaml` | `is_decoder`, `*ForCausalLM`/`LMHeadModel` suffix, `*ForConditionalGeneration` without `is_encoder_decoder`, composite `decoder=main` | none of these → typed unknown, never "causal" (`decoderness.yaml:19-21`) | C (`mask` allows `config_declared`) | musicgen |
| position schedule (which layers rotate) | `position_schedule.decoder_position_application_schedule_for_path` `:199` | `range(count) → comprehension target → block __init__ formal → attention __init__ formal → exact forward guard → proved rotation` (`:6-8`) | "attention index crosses an unproved owner hop" (`:393`); "layer {i} guard is not exactly evaluable" (`:271`); inactive ≠ NoPE (`:10-11`, registry `:691-694`) | Q+R | llama (uniform); synthetic alternating |
| QK-norm schedule | `qk_norm_schedule.decoder_qk_norm_schedule_for_path` `:89` | U6 proof joined to the mixer schedule; config gates may disable, never create | "QK normalization requires an exact per-layer mixer occurrence" (`:138`) | Q+R | qwen3, olmo-2, gemma3, stablelm |
| cross-layer KV sharing | `kv_sharing_schedule.decoder_kv_sharing_schedule_for_path` `:120` | one attention forward both **reads and writes** one shared mapping; constructor selectors resolve one earlier producer per sharing layer | "multiple ordinary-attention occurrences implement KV reuse" (`:150`); "KV read/write selector fields are not complete for every layer" (`:181`) | Q+R | gemma3, gemma4 (tests) |
| cross-attention schedule — additive (self + cross in every block) | `cross_attention_schedule.decoder_cross_attention_all_layers_for_path` `:101` | one block constructs **two unconditional** occurrences of the same attention class; local dataflow proves an optional formal feeds K/V but not Q; one call leaves it at literal `None` | "one attention construction called twice is not a dual module shape" (`:160`); "both attention modules must be constructed unconditionally" (`:209`) | Q+R | musicgen |
| cross-attention schedule — replacement (heterogeneous stack) | `cross_attention_replacement.decoder_replacement_cross_attention_schedule_for_path` `:136` | stage invokes one heterogeneous container; per-index selector; inside each selected block Q descends from one formal, K and V from another | "the attention compute path has no unique owner-side Q/K/V call" (`:314`); "the three attention affine inputs do not prove self or Q-vs-K/V lineage" (`:396`) | Q+R | synthetic; mllama (component tests) |

### 2.4 Cell topology

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| norm placement + residual topology (+ residual scale / gate) | `cell_topology.decoder_cell_topology_for_path` `:635` | two complete residual equations: **sequential** (attn merge → FFN input → FFN merge) or **parallel** (same input, one merge with both) (`:665-667`); residual may live in the block or inside the addressed child's forward plus same-file helpers; learned scaling field = `ResidualGateProof` `:68`; guarded topologies need an exact source-bound selector | "cell path has rival unguarded mechanism calls" (`:430`); "guarded cell alternatives lack an exact source-bound selector" (`:461`); "parallel and sequential equations both reach return" (`:1609`); "parallel branches use more than two distinct input norms" (`:926`); "no exact normalization boundary is proven" (`:1638`) | C (`norm_placement`, `residual_topology`, `residual_scale`, `parallel_norm_count`) | bloom, dbrx, falcon, gemma-2 (sandwich), gpt-oss, gptj, llama, olmo-2 (post), opt, qwen3, qwen3-5, stablelm |
| norm kind | `decoder_norm.decoder_norm_kind_for_path` `:185` + `primitive_semantics` | external registry `_EXTERNAL_PRIMITIVES` `primitive_semantics.py:29-43` (`torch.nn.{Embedding,GroupNorm,LayerNorm,RMSNorm}` + functional); custom norms classified from implementation ops: mean-subtract → `layernorm`, rms → `rmsnorm` (`:222-228`) | "positive norm calls do not prove absence of opaque calls" (`decoder_norm.py:451`); "external constructor {x} has no registered primitive protocol" (`primitive_semantics.py:90`) | C (`norm_kind`, unknown → generic node) | bloom, deepseek, gemma-2, glm-4-5, gpt-oss, llama, musicgen, qwen2-vl |
| parallel-branch input norms | `parallel_norm.exact_branch_census_at_block` `:172` / `exact_norm_sources_at_block` `:376` | exact def-use edge into each branch callee's first non-receiver formal | "an inline FFN has no addressed branch-input call" (`:205`) | C | falcon, gptj, dbrx |

### 2.5 Attention

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| attention-compute child (is this child attention at all) | `attention_child.attention_child_evidence` `:377` / `_positive_census` `:222` | `scaled_dot_product_attention` call, **or** a dot product and a softmax in the same exact callable (`:14-17`); follows `ALL_ATTENTION_FUNCTIONS`/`get_interface(..., eager_attention_forward)` only through the exact binding (`:19-22`) | "attention-compute child presence depends on an unresolved exact construction guard" (`:324`); anything without a softmax (linear/lightning attention, SSM, RG-LRU, RWKV) is simply not attention → no lane | — | 14 LLM witnesses |
| diffusers attention lane (container + injected processor) | `attention_lane.attention_lane_positive_census` `:556` | framework `Attention` constructor or a class using the processor mixin joined to the block's construction+call; injected processor kept only if its `forward/__call__` proves compute; public `heads/kv_heads/dim_head` give geometry only (`:9-21`) | "exact external construction has no indexed implementation or approved framework protocol" (`:639`) | — | 15 diffusion witnesses |
| Q/K/V storage | `attention_storage.decoder_attention_projection_storage_for_path` `:205` / `_for_child_evidence` `:440` | **split** = three distinct, unconditionally constructed affine occurrences reaching the compute entry (`:546-555`); **fused_qkv** = one affine occurrence → a tensor-method `.split(...)`/`.chunk(...)` with ≥3 flat unpack targets (`:557`, closed protocol `_proves_tensor_lane_unpack` `:950-975`); affine = `torch.nn.Linear` or a wrapper whose first base is a registered affine primitive (`affine.py:24-30`; `transformers.Conv1D` accepted — GPT-2) | "Q/K/V storage is not an exact three-producer split or one-producer three-lane unpack (producers=…, conditional=…, chained=…)" (`:562-567`); "the exact attention entry has no code-proven affine producers" (`:530`). **Not accepted:** subscript-slice unpacks (`qkv[..., :n]`, Falcon `qkv[:, :, :, :-2]`), `.view(...).unbind(...)`, einops `rearrange`, `torch.split(x, …)` functional form (unverified), conditional construction | C (`projection_mode`, unknown → pale) | bloom, dbrx, gpt-2, llama, qwen3, stablelm, olmo-2, deepseek, musicgen, gemma-2, qwen2-vl |
| storage/mechanism at a literal-registry dispatch site (`ATTENTION_CLASSES[config._attn_implementation]`) | `dispatch_selection.resolve_dispatch_candidates` `:239`; `dispatch_attention_storage` `:148`; `dispatch_attention_mechanism` `:149` | `self.<field>(...)` whose construction target is a literal-registry subscript; every candidate proves the same storage / selector-controlled singleton-K/V | "dispatch construction requires self.<field>(...)" (`dispatch_selection.py:264`); "the dispatch registry contains a non-literal key" (`:303`); "selector-controlled singleton K/V requires a fused producer" (`dispatch_attention_mechanism.py:174`) | C | falcon |
| mechanism kind MHA/GQA | `attention.attention_head_binding_at_block` `:2191` (public `decoder_attention_mechanism_for_path` `:1782`) | three Linear output widths share one per-head factor plus exact config count paths; fused: packed width + reshape + head-count relation | "fused QKV storage lacks one exact packed-width, reshape and head-count relation" (`:2265`); "split Q/K/V widths do not prove one shared factor plus exact head-count config bindings" (`:2399`) | Q (`head_geometry`) | llama, bloom, dbrx, qwen3, gpt-2 |
| MQA (selector-controlled single K/V head) | `attention.multi_query_attention_binding_at_block` `:2815` | `selector → one-K/V → projection → split → attention` dataflow | "the exact primary attention occurrence lacks a complete selector→one-K/V→projection→split→compute proof" (`:2854`) | Q | falcon (multi_query=True) |
| MLA | `attention.latent_attention_binding_at_block` `:3039` | compressed-KV / expanded-K/V projection pair + split + live flow; every auxiliary width independently bound (`docs/U6_U7_U8_QUALIFICATION_MATRIX.md:66`) | "no exact compressed-KV/expanded-KV projection pair and split dataflow binds the latent attention dimensions" (`:3102`) | Q | deepseek-v3 |
| head dim | `attention_geometry.decoder_attention_head_geometry_for_path` `:725` | evaluates the exact shared factor through the owner constructor; records only operands used | "the exact shared Q/K/V factor is not evaluable" (`:822`); "the exact head expression carries conflicting config premises" (`:830`) | Q | llama, bloom, gemma3, gemma4, deepseek |
| gated-delta (Qwen3-Next/3.5 recurrent mixer) geometry | `attention.gated_delta_geometry_at_occurrence` `:1929` | exact Q/K/V split + reshapes, Q/K repeat ratio, `Conv1d` kernel, sigmoid/softplus recurrent terminals (registry `:995`) | "no exact split/reshape/repeat/conv/recurrent geometry protocol was proven under the decoder block" (`:1901`) | Q+R | qwen3-5-27b-text |
| QK-norm | `qk_norm.decoder_qk_norm_evidence_for_path` `:121` | norm construction classified; **two** application sites reach the Q and K score operands; each input descends from an exact Linear; guards resolve to config paths (`:5-9`) | "Q and K do not reach two distinct exact norm applications" (`:266`); "a live norm result feeds another exact Linear application before the score and is therefore an intermediate/latent norm" (`:302`); "Q/K norm guard is not an exact config-field predicate" (`:737`) | R (`qk_norm`) | qwen3, olmo-2, glm-4-5, gemma3, stablelm |
| projection bias | `projection_bias.decoder_attention_bias_for_path` `:214` / `decoder_ffn_bias_for_path` `:295` | **`torch.nn.Linear` only**: omitted kw → `True`, literal bool, `bias=config.<field>` → exact path | "projection bias is only proven for exact torch Linear protocol" (`:390`); "the exact bias expression is not a source-only boolean" (`:411`); non-uniform → `"mixed"` pattern (`:128`) | C (`bias`) | bloom, deepseek, gemma-2, glm-4-5, gpt-oss, llama, qwen2-vl, qwen3 |
| score scaling | `attention.decoder_attention_score_scaling_for_path` `:946` | scaled (`*scaling`/`/√d`) or raw QK^T straight into softmax (`:3781`: "True=scale, False=neutral, None=unsupported") | "the exact score-to-softmax path is not a supported scaled or raw-score protocol" (`:1216`) — e.g. per-position temperature (Llama-4 `attn_temperature_tuning`), DeBERTa scale factor: unverified | C (`scores_scale`) | T5 (raw), llama, bloom, flux |
| logit softcap | `attention.attention_logit_softcap_at_block` `:1254` | config-guarded `score/cap → tanh → *cap` before one softmax | "the exact score path does not prove one config-bound guarded divide/tanh/multiply softcap protocol" (`:1348`) | R | gemma-2 |
| attention sinks | `attention_sinks.decoder_attention_sinks_for_path` `:111` | `nn.Parameter` on the child → child passed to compute callable → `torch.cat` with scores → softmax (`:8-14`) | "no exact learned Parameter joins scores before softmax" (`:209`) | C (`sinks`, code_proven only) | gpt-oss |
| QKV clip | `attention.attention_qkv_clip_at_block` `:1384` | fused QKV projection → config-bound `clamp` → compute | "the selected attention does not have one exact fused QKV lane" (`:1409`) | R | dbrx |
| KV cache | `attention.attention_cache_at_block` `:1504` | two projected lanes update one callable parameter; both replacements reach compute; only `parameter is not None` guard (`:3460`) | positive-only; unmatched source stays unknown (registry `:923`) | R | bloom, deepseek, llama, qwen3, stablelm |
| output projection | `attention_output.decoder_attention_output_projection_for_path` `:123` | attention-value terminal → one unique separately-constructed Linear in the same owner | "the attention-value terminal does not reach one unique exact Linear output projection" (`:289`) | R | all LLM witnesses |
| output gate | `attention.AttentionOutputGateBinding` `:444` | query-lane split → sigmoid → multiply attention result → output projection | config flags never author (registry `:970`) | R | qwen3-5 (gated attention) |
| score-side additives (neutral) | `attention_score_additives.decoder_attention_score_additives_for_path` `:288` | `Tensor.baddbmm` with exact finite non-zero `beta` (source/config); explicit `score + operand` / `+=` | "baddbmm beta is not an exact finite source/config value" (`:353`); implicit framework default refused (`:11-13`) | — | bloom, T5, flux, llama |
| attention input interface / invocation role / container interface (diffusers Q-primary, K/V-context) | `attention_input_interface` `:138`, `attention_invocation_role` `:243`, `attention_container_interface` `:176` | one exact SDPA call; Q from one formal, K and V from the same other formal; `context is None → context = primary` fallback | "a conditional role requires both exact alternatives" (`attention_invocation_role.py:218`) | — | opt (control), diffusion witnesses |
| DSA lightning indexer (`index_topk/index_n_heads/index_head_dim`), compressed sparse attention (`compress_ratio`) | **no reader** — config-declared `AttentionSpec` fields (`ir.py:94-97`); `dsa_indexer` view exists | config | — | — | (DeepSeek-V3.2 parsed in `docs/coverage_audit.md`) |
| M-RoPE section | `multiaxis_position.multimodal_multiaxis_position_result` `:80` proves only the ≥3-axis `torch.stack` route; the partition/base/width operands are U9 debt rows (`structural_debt.py` "the multimodal rotary coordinate partition" etc.) | fusion wrapper passes `position_ids=` to one constructed child; value closure reaches a ≥3-axis stack | "no exact fusion wrapper constructs multi-axis position ids" (`:130`) | — | qwen2-vl |

### 2.6 Position

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| coordinate origin (shared prerequisite) | `position_coordinate.coordinate_origin` `:48` | `torch.arange` with 1–3 **scalar** bounds; scalar = numeric literal, scalar parameter, `tensor.shape[<int>]`, one lane of `a, b, n = t.size()`, `x + y` of scalars, `cache.get_seq_length() if cache is not None else 0` with the exact parameter/default/guard/zero-arg protocol (`:118-170`, `docs/U6_U7_U8_QUALIFICATION_MATRIX.md:190-201`); `arange ± scalar`; approved shape-only wrappers `_COORDINATE_WRAPPERS`; `None`-defaulted `arange` (`defaulted_arange`) | **refused**: `cumsum` (OPT, `:` doc §5), `arange + tensor`, an attribute offset such as `positions + self.offset` (BART) or `self.embed_positions.num_embeddings` as a bound (Whisper) — `_scalar_offset` handles only `constant|name|subscript(shape)|binop|ifexp` (`:118-170`), a buffer slice `self.position_ids[:, a:b]` (BERT/CLIP), arbitrary tensors/method names | — | GPT-2, OPT poison, synthetic |
| Q/K rotation application | `position_application.decoder_qk_half_turn_application_for_path` `:150` | one local helper returning two lanes whose algebra is one of `split_half_turn` (rotate_half over halves), `chunk_pair` (`torch.chunk`), `interleaved_pair` (`x[..., ::2]`/`[1::2]`, GPT-J/Cohere), `complex_pair` (`view_as_complex`→multiply→`view_as_real`, Llama-4) (`:6-9`, `:83-84`, `:361-391`); outputs reach the proven Q/K score operands | "no exact two-lane position-rotation→query/key path was proven" (`:285`); positive only, never NoPE (`:12-13`) | Q (`position_schedule`) | llama, bloom(neg), deepseek, gemma2, gpt_oss, olmo, qwen3 |
| separate per-lane rotary (vision towers) | `separate_rotary.read_separate_qk_rotary` `:72` | Q and K each depend on one call to the same indexed callable with identical non-tensor args; closure reaches the half-turn protocol | "Q/K producer lineage is not exact" (`:108`) | — | synthetic |
| rotation factors (is it *positional*) | `position_factors.decoder_position_trig_factors_for_path` `:185` / `_complex_factors_for_path` `:267` | producer forward returns cos/sin of **one shared phase** = stored owner state × explicitly bound coordinate input, with an exact coordinate origin; complex variant = unit-complex phase | "producer return is not ordered cosine/sine" (`:613`); "shared phase is not stored-state × coordinate-input math" (`:623`); "phase input has no exact coordinate origin" (`:634`). A precomputed sinusoid table gathered by index (GPT-J `_get_embed_positions`) does not fit: unverified | R (`rope_theta`, `rope_initialization`) | same |
| rotation geometry (full vs partial rotary) | `position_geometry.decoder_position_application_geometry_for_path` `:144` | full = complete Q/K lanes enter; partial = exact two-part split before, exact concat after with untouched complement preserved | "Q/K do not share one exact split-and-recombine geometry" (`:214`); "the exact applied Q/K width is not consistently evaluable" (`:235`) — a `torch.chunk(q, int(1/self.partial_rotary_factor))` count (RecurrentGemma) is probably unevaluable: unverified | Q | stablelm (partial by slice), deepseek, llama, qwen3 |
| frequency initialization (θ, rope type) | `position_initialization.decoder_position_frequency_initialization_for_path` `:202` | phase state ← `self.<buffer>` ← lane 0 of one local initializer; helper's first return `1/(base**exp)`; base = exact config path / framework-normalized (`rope_config_normalization.py:82`) / literal; **or** one callable selected from an imported literal registry (`ROPE_INIT_FUNCTIONS`) with an exact selector and complete operand set | "the enacted initializer is neither local nor an exact registry entry" (`:489`); "selected registry callable has no exact config-parameter mapping" (`:622`); "initializer operand {x} has unknown provenance" (`:670`) | R | dbrx, deepseek, gemma2, gpt-oss (YaRN), granite (legacy→runtime map), llama, olmo, qwen3, stablelm |
| ALiBi | `position_linear_bias.decoder_alibi_score_bias_for_path` `:169` | enabled `baddbmm` receiver on the score lane **and** receiver traces to a producer returning `head-slopes × cumulative coordinate` | "the exact producer is not slope times cumulative coordinate" (`:319`); "the producer wrapper is not one direct local helper call" (`:310`). An ALiBi added explicitly (`scores + alibi`, MPT/Baichuan) does not satisfy the baddbmm join: unverified | Q (`position_schedule`) | bloom |
| learned relative bias (T5 buckets) | `position_relative_bias.decoder_relative_position_bias_for_path` `:210` | relative coords → bucket callable → learned `Embedding`; table owner by exact first-index selection | "table ownership is not exact first-index selection" (`:572`); "repeated construction is not one direct index" (`:533`) | Q | T5 (FLUX/SD3 text encoders) |
| NoPE | not a fact: an inactive schedule slot is *unknown* (`position_schedule.py:10-11`; registry `:691-694`) | — | — | — | — |
| 3-D / axial RoPE in DiTs | `diffusion_block` reuses the half-turn reader per lane; `rope_3d` chip is `AttentionSpec` only (`ir.py:81`) — positive rotation, factor provenance separate | — | — | `diffusion_attention_position_application` route | wan, hunyuanvideo, cogvideox, mochi, ltx |

### 2.7 FFN

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| ordinary FFN mechanism (dense / gated / fused gate-up) | `ffn_mechanism.decoder_ffn_mechanism_for_path` `:754` / `ffn_mechanism_at_block` `:514` | projection constructions + local value flow in **one** invoked child or inline on the block; modes `dense | split | fused_gate_up` (`:113`), split protocols `chunk|split|tensor_split` (`:59`); activation via `activation_semantics` (`ACT2FN[config.x]`, `get_activation`, direct primitive — `everchanging/evidence/program_index_vocab.yaml:15-21`); exhaustive construction alternatives must agree (`EquivalentFFNMechanism` `:192`); config-selected nested branch (`ConfigSelectedFFNMechanism` `:260`) | "no exact invoked child or inline block has a proven ordinary two/three-projection FFN dataflow" (`:626`); repeated sequential execution of one child is not an FFN (`:551-556`); an FFN split across **two** children (BERT `intermediate`/`output`, ViT) has no single owner with 2–3 projections: unverified | C (`activation`, `gated`, `projection_mode`) + Q (`intermediate_size`) | bloom, deepseek, falcon, glm-4-5, gpt-oss, llama, musicgen, qwen2-vl, stablelm, T5 |
| fused input transform (diffusers `GEGLU`/`SwiGLU` containers) | `ffn_input_transform.fused_input_projection_transform_at_symbol` `:130` | `affine(x) → chunk(2) → value * activation(gate)` on every return path | "return paths prove different input transforms" (`:176`) | — | flux, sd3 |
| composite container FFN (diffusers `FeedForward.net`) | `selected_composite_ffn.selected_composite_ffn_mechanism` `:251` | selector token → guarded local construction → `append` exactly once → later affine output → direct state-carrying loop | "container has an unclassified executed element" (`:477`); "container execution loop is not exact" (`:520`) | — | opt (control), diffusion witnesses |
| intermediate width | `ffn_width.decoder_ffn_intermediate_width_for_path` `:75` | input dimension of the proven output projection; every input projection emits the same width | "width evaluation does not choose among conditional FFN owners" (`:119`); "the exact output-projection input width is not evaluable" (`:140`) | Q | GPT-2, GPT-J, CodeGen, bloom, llama, musicgen, T5 |
| conv-GLU (Sana) | drawn via the diffusion FFN projection (`FFNSpec.kind="conv_glu"`, `ir.py:127`); the proving reader is not separately identifiable from the diffusion composition — **unverified** which reader authors it | — | — | `diffusion_ffn_mechanism` route | sana |

### 2.8 MoE

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| router policy | `router.decoder_router_selection_for_path` `:430` | reachable callable containing the selection op **`topk`** (`_SELECTION_PROTOCOL` `:60`) or the `sparse_mixer` kind (`:217`); score transform `softmax|sigmoid` (`:57-58`); grouped routing = one group-level `topk` reaching a mask with `group_score_kind ∈ top1_max|top2_sum` and exact `n_group`/`topk_group` paths (`:249-256`, `:1053`); renorm/scale operands | "no exact invoked decoder-block closure proves a router selection" (`:398`); "no exact routed-expert storage anchors this selection policy" (`:381`); routers without `topk` (argmax/one-hot top-1 Switch, expert-choice, hash) are not provable: unverified | Q (`routing_policy`) | dbrx, deepseek-v3, glm-4-5, gpt-oss, mixtral |
| routed expert storage | `expert_storage.decoder_routed_expert_storage_for_path` `:323` | **fused**: a ≥3-D `torch.nn.Parameter` used under an expert loop → two lanes split/interleaved → multiply → feeds a different stacked Parameter (`:8-13`); **split**: three repeated Parameters on one child + per-expert selection of all three in the parent loop + gate/up→act→down flow (`:15-19`) | "no exact invoked variant proves fused or split expert storage" (`:248`). A `ModuleList` of `nn.Linear`/MLP modules per expert (pre-5.x transformers, most `trust_remote_code` MoEs) matches neither shape: unverified | C (`expert_projection_mode`) | deepseek-v3, glm-4-5, gpt-oss, dbrx (flattened) |
| expert width | `expert_width.decoder_expert_intermediate_width_for_path` `:126` | fused param dimension containing one literal two-lane factor (`2*w` / `w*2`) whose remainder equals a down-param dimension | "split/flattened expert storage does not prove a per-expert width" (`:150`, DBRX) | Q | deepseek-v3, glm-4-5, gpt-oss |
| shared experts | `expert_width.decoder_shared_expert_count_for_path` `:212` | one ordinary FFN added to routed output whose width is `per-expert width × <config count>` | "the shared FFN width is not a two-factor product" (`:283`); a shared expert declared with its own width field (Qwen2-MoE `shared_expert_intermediate_size`) or with a sigmoid gate: unverified | Q | deepseek-v3, glm-4-5 |
| expert activation formula (α, clamps, up-offset) | `expert_storage.ExpertActivationEvidence` `:75` | operands on the exact gate lane | — | C | gpt-oss |
| top-k selection drill, expert drill | views `topk_selection`, `moe_expert` project the router/storage facts | — | — | — | — |

### 2.9 Multimodal (transformer wrappers)

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| tower mechanisms (attention/FFN/norm/position per component) | `component_tower.recursive_component_mechanisms` `:493`; `repeated_stage.repeated_stage_mechanisms_at_owner` `:81` | U6/U7/U8 readers re-run on each active component's repeated child | "this exact attention variant has no ordinary shared-factor proof" (`component_tower.py:755`) | — | llama, mllama, qwen2-vl |
| fusion (where modality tokens enter the decoder) | `fusion.fusion_result` `:134` | normalized expressions on the exact root graph (masked-scatter / index-put style routes: not enumerated in the docstring — unverified list) | "fusion may be hidden in an unsupported/unresolved execution path" (`:170`); "a competing fusion path remains unsupported/unresolved" (`:193`) | — | gemma4, llama, llava, mllama, paligemma, qwen2_vl |
| projector / merger | `projector_lineage.projector_lineage_result` `:186` → `projector_chain.read_projector_operation_chain` `:109` → `projector_width.projector_width_evidence` `:48`; perceiver-style `repeated_projector` `:198/339` | backwards from the fusion operand to the terminal affine; ops only via exact import protocols, `Sequential` elements, OwnerGraph calls; learned query seed for resamplers | "the return path contains a guarded/rival/unresolved binding" (`projector_chain.py:190`); "a return-producing container is not one exact Sequential" (`:300`) | R (`projector_in/out_features`) | gemma4, idefics, llama, llava, mistral, mllama, opt, paligemma, qwen2_vl |
| frontend ops before a stage (patch embed, etc.) | `stage_operations.stage_operation_inventory_at_owner` `:128`; `component_operations` `:129/351/482` | registered operations whose result reaches a repeated call | "no supported operation has a positive path to the repeated stage" (`:207`); absence never negative | — | qwen2_vl |
| feature selection (`hidden_states[-2]`, CLS drop, concat) | `wrapper_features.wrapper_feature_selection_result` `:93` | indexing / concatenation / token slice on an exactly constructed component's output | "no exact component-output feature operation is proven" (`:137`) | — | llama, llava, mllama |
| vision-tower numeric operands (patch size, merge size, hidden width, depth, head count, FFN expansion/activation) | **no reader** — 9 `root.vision` U14 + 4 `root.text_encoder.vision` U9 config_read debt rows | config | — | — | qwen2-vl |
| audio composite (MusicGen encoder/decoder, K codebooks) | `codebook_streams.decoder_codebook_streams_for_path` `:184` (input bank + output bank, `ModuleList` → comprehension → aggregate); conditioning encoder facts are 7 `root.conditioning` U14 config rows | — | "a codebook count path requires the container's exact citation" (`:114`) | Q+R (`codebook_streams`) | musicgen |

### 2.10 Diffusion transformers (DiT / MMDiT)

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| root topology | `diffusion_root.read_diffusion_root_topology` `:406` | `repeated_stack` (root forward iterates a root-owned container and invokes the loop element) or `u_shaped` (two loops joined by an accumulated bypass) (`:9-19`); result always `incomplete` | "no exact root-owned container invocation is proven" (`:439`); "the exact root class has no indexed forward" (`:430`) | R (`diffusion_root_topology`) | 15 diffusion witnesses + sdxl (u_shaped) |
| stack inventory (depth, variant) | `diffusion_stack.read_diffusion_stack_inventory` `:620` | one container address + one symbolic element occurrence; comprehension = one template, never N; guarded rival containers selected only by checkpoint evidence or `register_to_config` literal default (`config_registration.py:185`) | "{n} container/traversal candidates remain unresolved" (`:758`) | R (`diffusion_stack_depth`, `_variant`) | all |
| block lanes (attention/FFN/norm/QK-norm/rotation per lane) | `diffusion_block.read_diffusion_block_facts` `:380` | composes U6/U7/U8 readers per lane; gated-delta lanes kept separate; half-turn ≠ RoPE until factor proof | "{n} U10-B stack candidates stay opaque" (`:429`) | R (`diffusion_attention_*`, `diffusion_ffn_mechanism`, `diffusion_norm_mechanism`) | all |
| stream relations (dual/single/joined) | `diffusion_stream.read_diffusion_stream_graph` `:1344` | `state` = formal with lineage to a return; `context` = non-returned formal supplying K/V of a proven lane; joined-input = exact framework concat | "lane input has rival or unsupported reaching definitions" (`:1147`) | R (`diffusion_stream_relation`) | flux, sd3, hunyuanvideo, qwen-image, auraflow |
| conditioning applications | `diffusion_conditioning.read_diffusion_conditioning_graph` `:482` | exactly three shapes: `norm_modulation`, `bare_gate`, `gate_in_norm` (`:6-13`); no name/config/dimension creates a claim | "positive conditioning applications do not prove whole-block absence" (`:511`). Additive conditioning (`hidden + temb`), cross-attention-only conditioning, and timestep/guidance *modality naming* are outside (`:14-16`, U10-E) | R (`diffusion_conditioning_applications`) | all |
| bookends (patchify → stack → unpatchify) | `diffusion_bookends.read_diffusion_bookends` `:435` | registered op → exact state argument of the repeated call; registered op consuming the repeated call → root return; 3-D primitives only on those routes | "positive local bookend routes do not prove whole-root completeness" (`:561`); temporal *declarations* (patch tuples, `num_frames`) never accepted (`:22-30`) | R (`diffusion_bookend_operations`, `_geometry`) | all |
| companion denoisers (Wan 2.2 experts) | `diffusion_companion.read_diffusion_companions` `:200` | each slot resolved through its own D0 graph; strongest result `same_source_contract`, never equivalence | "source profiles are positive/open-world and do not prove instantiated equivalence" (`:249`) | — | wan2-2, flux (tests) |
| temporal axis (video) | `denoiser.denoiser_temporal_axis` `:39` | an exact marker name (`everchanging` vocab, default `num_frames`) observed in the owner's forward | "the forward contains an unsupported expression; absence is unprovable" (`:83`) | — | 11 denoisers (tests) |
| denoiser chips still config-drawn | **no reader** — 80 `root.denoiser` U15 config_read rows (`structural_debt.py`), e.g. "the DiT FFN inner-width derivation", "the cross-attention sublayer's own head geometry", "the denoiser Q/K-normalization fact", "the denoiser rotary-application fact", "the conditioning card" | config | — | — | — |

### 2.11 UNet

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| stage construction / selection / operands | `unet_stage_construction.read_unet_stage_construction` `:660`; `unet_stage_selection.read_unet_stage_selection` `:500`; `unet_stage_operands` `:428`; `unet_stage_constructor_operands` `:270`; `unet_selected_stage_children` `:522`; `unet_selected_constructor` `:656` | append tied to one producer call in the root constructor; factory (`get_down_block`) expanded through the exact called import (`import_source.py:199`); checkpoint list → loop → factory formal → guarded return | "positive exact construction inventory; config branch selection and whole-callable coverage remain open" (`:732`); "one or more stage positions have no unique live return" (`unet_stage_selection.py:580`) | — (no `unet_*` fact registered) | sdxl, sd1.5 (tests) |
| stage execution (mid block) | `unet_stage_execution.read_unet_stage_execution` `:261` | root fields invoked textually inside the U10 inter-loop interval | "the exact U10 proof does not expose exactly two route stages" (`:279`) | — | sdxl |
| cells (ResNet/attention/sampler) | `unet_stage_cells.read_unet_stage_cells` `:536`; `unet_cell_mechanism.read_unet_cell_mechanisms` `:767`; `unet_nested_mechanism` `:498`; `unet_selected_spatial_operations` `:740` | container iteration incl. sliced/reversed/enumerate/zip; ops `GroupNorm/Conv2d/Conv3d/Dropout`; two-branch additive return (÷ expr); side-input injection; samplers = registered primitive on the returned value | "U11-D2 local mechanisms cannot prove a temporal axis" (`:368`); "positive exact cell mechanisms; temporal axes and CFG remain open" (`:786`) | — | sdxl, sd1.5 |
| cross-attention source route | `unet_attention_source.read_unet_runtime_attention_sources` `:722` + `unet_root_preprocess` `:475` | non-optional root formal → contiguous formal route → K/V; explicitly *not yet* a `cross_attention` boolean (`:6-11`) | — | — | sdxl |
| **what the UNet drawing actually uses today** | `adapters/diffusor/unet.py:parse_unet` (config: `block_out_channels`, `down_block_types`, …) **plus the 5 quarantined legacy readers** `patterns.unet_{transformer_ffn_activation,mid_block_present,code_attention_placement,stage_attn_cell,stage_temporal}_from_files` called from `adapters/diffusor/parser.py:57,81,101,113` (`docs/U3_CURRENT_READER_INVENTORY.md:28-34`); 34 `root.denoiser` U11 config_read debt rows + `extras:unet` | — | — | — | sdxl |

### 2.12 VAE, scheduler, encoders

| mechanism | reader | notes |
|---|---|---|
| VAE geometry (latent channels, scaling, patchify, temporal, decoder attention chip) | **no reader** — `adapters/diffusor/parser.py:725 _vae_geom` reads `_vae_config` via `config_access.resolve` under `container_scoped(("_vae_config",))` (`:78` of that region); 33 `root.vae` U12 debt rows | config only |
| scheduler / sampling loop | **no reader** — `_scheduler_geom` `parser.py:647`: `num_train_timesteps`, `prediction_type`, `shift`, `use_dynamic_shifting`, `beta_schedule`, `timestep_spacing` consumed under mechanism `sampling_loop`; flow-matching flag from class-name markers `_FLOW_MATCHING_MARKERS` (identity-as-display) | config only; `scheduler_step` view |
| pipeline text encoders (T5/CLIP/Llama/Qwen2.5-VL) | recursion through the transformer adapter (`submodel.py`, parity net) | T5 relative bias / no-scaling and CLIP tower are the witnessed encoder shapes |

### 2.13 MTP, PLE, codebooks, block diffusion

| mechanism | reader | accepted shapes | refusals | gate | witnesses |
|---|---|---|---|---|---|
| MTP modules | `mtp.decoder_mtp_construction_for_path` `:102` | `norm(hidden) ‖ norm(shared embedding) → concat → projection → exact constructed block → exact shared output head` (`:8-13`) | "model-stage address is unresolved" (`:117`) | R | deepseek-v3, glm-4-5 |
| per-layer embedding (Gemma-3n PLE) | `per_layer_side_input.decoder_per_layer_side_input_for_path` `:71` | a repeated-call argument indexed by the loop index → block formal → gate→act→multiply→projection→norm chain added into state | "decoder block address is unresolved" (`:90`) | Q+R | gemma3 (tests) |
| codebook streams | see §2.9 | | | Q+R | musicgen |
| block diffusion (DiffusionGemma canvas, self-conditioning, dense‖MoE) | **no reader** — `extras:block_diffusion`, `extras:block_diffusion.canvas_length` U14 debt; `self_conditioning` view | config | — | — | — |

---

## 3. Which of this is actually gated

The 17 qualification rules (`qualification.py:53-140`) are the only *value-exact* instance gates and they cover: head geometry (+schedule), gated-delta geometry, mask schedule, position schedule, position addition, mixer schedule, QK-norm schedule, KV-sharing schedule, cross-attention schedule, intermediate size, FFN schedule, expert width, routing policy, shared-expert count, codebook streams, PLE pathway. They apply **only** to `transformer_decoder` IR and are skipped for any IR whose `extras.render.family == "diffusion"` (`qualification.py:200-205`). Diffusion facts are gated through projection routes + receipts instead (`receipts.py:215 join_obligation_receipts`), and the UNet has **no registered fact at all** — its drawing is governed by structural-debt rows, not by a reader.

Facts registered but with neither a rule nor a route (census-only, so a wrong value cannot be caught at the instance level): `activation`, `gated`, `expert_projection_mode`, `expert_activation_formula`, `bias`, `mechanism`, `mask` (the uniform-direction fact), `sinks`, `norm_kind`, `norm_placement`, `residual_topology`, `parallel_norm_count`, `residual_scale`, `scores_scale`, `projection_mode`, `tie_word_embeddings`, `embedding_norm_kind`, `final_norm_kind` (`registry.py` rows with `routes=0`).

---

## 4. Mechanisms with no reader (drawn from config, legacy, or not drawn)

From the structural-debt register (279 rows, grouped):

| owner | rows | unit | what is still config/legacy |
|---|---|---|---|
| `root.denoiser` (DiT) | 80 | U15 | FFN inner-width derivation (+rounding quantum), cross-attention sublayer head geometry, QK-norm fact, projection-bias fact, rotary application/geometry facts, conditioning card chips, "an exact denoiser source/config operand" |
| `root.denoiser` (UNet) | 34 | U11 | stage-channel ladder, per-stage ResNet counts, per-stage Transformer2D depth, attention placement (down/up), bottleneck stage, timestep-embedding mechanism/width/activation, ResNet norm/residual-scale/time-scale-shift, added-cond (SDXL text_time), class conditioning, dual cross-attention selector, input/output conv kernels, downsampling padding |
| `root.vae` | 33 | U12 | every VAE chip |
| `consumer.*` (renderer 37, conformance 29, json 15, params 2) | 83 | U14 | terminal consumers still reading raw `extras`/config |
| `root.vision`, `root.text_encoder.vision` | 13 | U14/U9 | patch size, merge size, hidden width, depth, head count, FFN expansion/activation, projector out-width chip, tower position table |
| `root` (multimodal rotary) | 11 | U9 | M-RoPE partition/base/width/initializer selector, interleaved application |
| `root.conditioning` (MusicGen T5) | 7 | U14 | encoder attention geometry, FFN protocol/width/activation, repetition, vocabulary |
| `root` / `model` misc | 9 | U14/U5 | `extras:block_diffusion`, `extras:softcap`, `extras:modalities`, embedding-input scale, logits divisor, `extras:render` (presentation) |

Additionally, **drawn kinds with no producer**: the `ssm`, `rwkv`, `linear` attention drill tables (`adapters/transformer/blocks/attention.py:97-100`) are unreachable — the parser assigns `attn_kind` only from a U6 binding, a scheduled kind, or `gated_delta` (`adapters/transformer/parser.py:4008-4023`), and no production module emits those strings (grep). RG-LRU (RecurrentGemma), Mamba/Mamba-2 (Jamba, Falcon-H1, Granite-4.0-H, Nemotron-H, Bamba, Zamba2), RWKV, and MiniMax lightning attention therefore have **no reader and no drawing**; on a hybrid stack the mixer schedule fails as a whole ("not every layer selects and invokes one proven mixer", `mixer_schedule.py:446`), so even the attention layers of a hybrid lose their per-layer schedule. `patterns.py:57-59` states the scope limit explicitly.

Also without a reader: DSA indexer / compressed sparse attention (config `AttentionSpec` fields), Llama-4 attention temperature scaling (unsupported score transform), Gemma-3n AltUp/MatFormer/laurel (not even a spec field), LoRA adapter towers (Phi-4-multimodal), Switch-style top-1 argmax routing (no `topk`), expert-choice routing, the LM head block, the embedding block, the Gemma embedding scale, Cohere logit scale, and every VAE/scheduler value.

---

## 5. Ecosystem denominator

### 5.1 Installed ecosystem

`transformers 5.12.1`: **656** `model_type`s, **466** torch modeling files. `diffusers 0.38.0`: **52** `*Transformer*Model` denoiser classes (incl. `AceStepTransformer1DModel`, `StableAudioDiTModel`, `LTX2VideoTransformer3DModel`, `Flux2Transformer2DModel`, `HunyuanVideo15Transformer3DModel`, `NucleusMoEImageTransformer2DModel`, `ZImageTransformer2DModel`…), **6** top-level UNets (`UNet1DModel`, `UNet2DModel`, `UNet2DConditionModel`, `UNet3DConditionModel`, `UNetMotionModel`, `UNetSpatioTemporalConditionModel`), **24** autoencoders, **53** schedulers. The corpus exercises 14/656 transformer types (2.1%) and 12/52 denoiser classes + `UNet2DConditionModel` (25%).

### 5.2 Stratified sample — 25 transformers model types not in the corpus, classified by reading the source

Strata: 5 classic decoders, 6 modern llama-likes, 5 MoE/hybrid, 2 recurrent, 4 encoders/enc-dec, 3 vision/audio/VLM. Cells: **A** accepted by a protocol above (source shape matches a witnessed shape); **A?** matches the protocol text but unwitnessed; **R** refused (reason); **N** no reader / out of scope; **—** not applicable. Source lines are from `/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/transformers/models/<m>/modeling_<m>.py`.

| model_type | attention storage / mechanism | position | mask | FFN | MoE | cell | end-to-end decoder path |
|---|---|---|---|---|---|---|---|
| gpt2 | A — Conv1D + `.split(self.split_size, dim=2)` (`:185`) | A — learned, `arange + past_seen_tokens` (`:581`) | A — `create_causal_mask` (`:591`) | A — Conv1D c_fc/c_proj | — | A seq | **A** (real-source control) |
| gptj | A — split q/k/v | A? rotation `rotate_every_two` = interleaved_pair (`:57-67`); **R?** factors: precomputed sinusoid table gathered via `_get_embed_positions` (`:190`) not "stored state × coordinate"; partial rotary by slice (stablelm-shaped) A | A (`:505`) | A (ffn_width witness) | — | A parallel (witness) | **A/partial** (θ likely unknown) |
| gpt_neox | A? — fused `query_key_value` → `.view(...)` → `qkv.chunk(3, dim=-1)` (`:214-215`) = one-producer three-lane unpack | A? rotate_half + `rotary_ndims` slices | A (`:355`) | A dense_h_to_4h | — | A? `use_parallel_residual` config guard | **A?** |
| opt | A split | **R** — `torch.cumsum(attention_mask)` (`:65,351`) | A (`:356`) | A fc1/fc2 | — | A? `do_layer_norm_before` guard | **partial** (position unknown) |
| falcon | **split** — `_split_heads` uses subscript slices `qkv[:, :, :, :-2]` (`:273-275`) → storage R; MQA path (`multi_query=True`, falcon-7b) A via `multi_query_attention_binding_at_block` (witness); `new_decoder_architecture` GQA (40b/180b) **R** | A rotate_half; ALiBi checkpoints A (bloom-shaped `baddbmm`) — unverified for Falcon's `build_alibi_tensor` cumsum (`:191`) | A (`:768`) | A | — | A parallel (witness) | **A for 7b, R for 40b/180b** |
| mistral | A split | A rotate_half + local init | **R?** — `mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask; causal_mask = mask_function(...)` (`:372-373`): builder aliased through a config-conditional expression — not the witnessed dict shape; maps to "retains rival or conditional builders" (`attention_mask.py:870`) unless the ifexp is evaluated (unverified) | A gated | — | A seq | **partial** (mask unknown → typed) |
| mixtral | A split | A | R? same alias (`:467`) | A (dense absent; experts fused) | A — `gate_up_proj` 3-D Parameter + `.chunk(2)` (`:70-92`), `topk`, softmax | A | **partial** (mask) |
| phi3 | **R** — fused `qkv_proj` + `qkv[..., :query_pos]` slices (`:239`) → storage unknown → mechanism unknown | A? partial rotary slices + cat (`:200-204`); LongRoPE registry init A? | R? alias (`:403`) | A — `gate_up_proj` + `chunk(2)` = `fused_gate_up` (`ffn_mechanism.py:113`) | — | A seq | **R** (attention kind None) |
| cohere | A split | A — interleaved `x[..., ::2]` (`:188`) = interleaved_pair | A (`:423`) | A gated | — | A parallel `residual + attn + mlp` (`:355`) | **A?** |
| gemma3 | A split, QK-norm A (witness in schedule tests) | A rotation; **?** position schedule: `position_embeddings[self.config.layer_types[i]]` dict subscript at the stage loop (`:571`) is not the `range→formal→guard` transport | A — `causal_mask_mapping` dict by `layer_types` (`:557-558`, Gemma-2-witnessed shape) | A gated | — | A sandwich (Gemma-2 witness) | **A/partial** (local/global θ per layer unverified) |
| llama4 | A split; **R** score scaling — `attn_temperature_tuning` is an unsupported score transform (unverified) | A? `complex_pair` (`:250-253`); NoPE via `use_rope` guard: **?** | A — `create_chunked_causal_mask` in mapping (`:562-563`) | A gated | A? fused `gate_up_proj` Parameter + `chunk(2)` (`:63-82`), `topk` sigmoid (`:150`); shared expert width = same field → shared-expert reader wants a `width × count` product: **R?** | A seq | **partial** |
| olmoe | A split, QK-norm A (olmo witness) | A | A (`:490`) | — | A fused (`:310-332`) | A | **A?** |
| qwen3_moe | A split, QK-norm A | A | R? alias (`:496`) | — | A fused (`:223-245`) | A | **partial** (mask) |
| granitemoe | A split | A | A (`:512`) | — | A? `GraniteMoeParallelExperts` 3-D Parameter + `input_linear` chunk(2) (`:257`), `topk` | A + residual_multiplier (granite witness) | **A?** |
| jamba | attention layers A split; Mamba layers **N** | A | A (`:719`) | A/moe fused | A fused (`:502-524`) | A | **R** — `ModuleList` of two decoder-layer classes (heterogeneous) fails F2; mixer schedule cannot prove Mamba layers |
| minimax | softmax layers A; lightning layers **N** (no softmax → not attention) | A | R? alias | — | A fused (`:483-505`), `topk` (`:468`) | **R?** `residual*alpha + h*beta` learned/config scaling (`:577-581`) | **R** (mixer schedule) |
| mamba2 | **N** | — | — | — | — | — | **R** (no attention child, no decoder block mechanism) |
| recurrent_gemma | attention layers A; RG-LRU **N** | **R?** partial rotary `torch.chunk(q, int(1/self.partial_rotary_factor))` (`:223`) count unevaluable | A — `create_sliding_window_causal_mask` (`:663`) | A gated (gelu) | — | A | **R** (mixer schedule: recurrent layers unproven) |
| bert | A split (query/key/value Linear) | **R** — `self.position_ids[:, past:past+seq]` buffer slice (`:88`) | A — `create_bidirectional_mask`/`create_causal_mask` by `is_decoder` (`:701-708`, witnessed) | **R?** — FFN split across `BertIntermediate`/`BertOutput` children (no single 2–3-projection owner) | — | A? post-norm inside child forward | **partial** |
| t5 | A split, raw scores A (witness) | A relative bias (witness) | A (`:691-708`) | A wi/wo, wi_0/wi_1 | — | A pre-RMS | **A** (encoder witnessed; decoder cross-attn A?) |
| deberta_v2 | A split | **R** — disentangled c2p/p2c bias (`rel_embeddings`, `:93-94`) is neither bucket→Embedding nor baddbmm | **R** — custom mask (`get_attention_mask`, no framework builder) | R? two-child FFN | — | A? post-norm | **R/partial** |
| bart | A split | **R** — `BartLearnedPositionalEmbedding` adds `self.offset` attribute (`:92`) | A (`:527,634`) | A fc1/fc2 | — | A post-norm | **partial**; decoder cross-attn A? (MusicGen-shaped) |
| vit | A split | **R** — `nn.Parameter` position added directly (`:84`), not `Embedding.weight` | A bidirectional (`:376`) | R? two-child FFN | — | A pre-LN | **partial** (only as an embedded tower) |
| whisper | A split | **R** — encoder `embed_positions(arange(self.embed_positions.num_embeddings))` attribute bound (`:622`); decoder custom `WhisperPositionalEmbedding` | A (`:765`) | A fc1/fc2 | — | A | **partial** |
| llava | wrapper: fusion A, projector A (witnesses in tests) | (backbone) | (backbone) | (backbone) | — | — | **A** for the wrapper; backbone per its own row; CLIP tower position **R** (buffer slice) |

**Acceptance per protocol** (denominator = models where the mechanism exists; A and A? counted as accepted; R? counted as refused):

| protocol | accepted | denominator | rate | refusals in the sample |
|---|---|---|---|---|
| attention storage (split / fused unpack) | 20 | 23 | 87% | phi3 (slice), falcon-new-arch (slice), jamba (address) |
| attention mechanism kind reaching the spec (needs storage + head binding + mixer schedule) | 15 | 23 | 65% | + minimax, recurrent_gemma (hybrid schedule), llama4 partial |
| mask (framework builder table) | 15 | 22 | 68% | mistral, mixtral, qwen3_moe, phi3, minimax (config-conditional alias), deberta (custom) |
| position — rotary application | 14 | 15 | 93% | recurrent_gemma geometry |
| position — rotary factors/θ | 12 | 15 | 80% | gptj (table gather), gemma3 per-layer (unverified), recurrent_gemma |
| position — learned/fixed absolute | 1 (gpt2) | 7 | 14% | opt (cumsum), bert (buffer slice), deberta, bart (attribute offset), vit (Parameter add), whisper (attribute bound) |
| position — bias families (ALiBi / relative) | 1 (t5) | 2 | 50% | deberta (disentangled) |
| FFN ordinary mechanism | 19 | 22 | 86% | bert, deberta, vit (two-child FFN, unverified) |
| MoE storage + router | 8 | 8 | 100% (all 5.x fused-Parameter experts) | shared-expert count for llama4 R? |
| cell topology | 21 | 22 | 95% | minimax (α/β scaling) |
| end-to-end decoder path yields a drawable, source-proven stack | 8 A + 9 partial | 24 | 33% clean / 71% with typed unknowns | 7 R: phi3, jamba, minimax, mamba2, recurrent_gemma, deberta, falcon-40b |

### 5.3 The author's declared target catalogue

Sources: `done/TO_SERVE.md` (447 lines; **306** unique HF ids over **243** table rows: 135 text, 24 VLM, 37 image, 21 video, 26 audio), `unfold-pkg/previews/old/toserve.md` (June-2026 canonical list, **118** ids), `unfold-pkg/previews/old/toserve_model.md` (**165** ids, should-support + nice-to-have), `scripts/coverage_audit.py` MODELS (49 attempted, 35 parsed, `docs/coverage_audit.md:5-7`) and `scripts/dit_coverage.py`. Union of ids: **433**. Below, each TO_SERVE row-group is mapped to its installed `model_type` / diffusers class and given the §5.2 verdict by protocol shape (**A** clean, **P** partial: core proven with typed unknowns, **R** core mechanism or address refused, **N** out of scope / no reader / no HF source). Rows marked † were opened in §5.2 or are corpus witnesses; the rest are classified by family shape and are **unverified**.

**Part 1 — text (135 rows)**

| family (rows) | model_type / class | verdict | why |
|---|---|---|---|
| Llama 1/2/3/3.1/3.2-text/3.3/CodeLlama/Guard/Tülu/Yi/Yi-1.5/DeepSeek-LLM/Coder/Math/TinyLlama/SOLAR/Hermes/Sarvam-1/SmolLM-2/XGen/Falcon-3/Nemotron-Llama/Minitron (≈30) | `llama` † | A | corpus witness llama-7b |
| Llama 3.2 Vision (2) | `mllama` | P | replacement cross-attn + tower readers witnessed in tests; vision numeric chips config |
| Llama 4 Scout/Maverick (2) | `llama4` † | P | temperature scaling R, shared-expert width R?, NoPE schedule ? |
| OPT, Galactica (2) | `opt` † | P | cumsum positions R |
| Mistral 7B/NeMo/Small/Large/Ministral/Codestral/Mathstral/Devstral/Magistral/sarvam-m (≈10) | `mistral` † | P | mask alias R? |
| Mixtral 8×7B/8×22B (2) | `mixtral` † | P | same |
| Mistral Small 3.1+/Pixtral (3) | `mistral3`/`pixtral` + llava wrapper | P | vision 2-D rope in tower unverified |
| Qwen 1 (1) | remote `qwen` | N/? | hub-source path only (`sources.py:14`) |
| Qwen1.5/2/2.5/Coder/Math/QwQ/QVQ-text (≈8) | `qwen2` | P | bias True A; mask alias R? |
| Qwen1.5-MoE / Qwen2-57B-A14B (2) | `qwen2_moe` | P | shared expert with own width field + sigmoid gate R? |
| Qwen3 dense (6) | `qwen3` † | A | witness |
| Qwen3 MoE/Coder-480B (3) | `qwen3_moe` † | P | mask alias R? |
| Qwen3-Next, Qwen3.5 (2) | `qwen3_next`, `qwen3_5` † | A | gated-delta witness qwen3-5-27b-text |
| Qwen3-Omni, Qwen2-Audio (2) | `qwen3_omni_moe`, `qwen2_audio` | P | composite audio partially shipped (PR #14); talker/code2wav unverified |
| DeepSeek-MoE-16B, Coder-V2 (2) | remote / `deepseek_v2` | R? | remote `ModuleList` experts |
| DeepSeek-V2/V2.5 (2) | `deepseek_v2` | P | MLA A; expert storage shape unverified |
| DeepSeek-V3/V3-0324/R1/R1-0528/V3.1/Prover/Math-V2 (7) | `deepseek_v3` † | A | witness deepseek-v3 |
| R1-Distill (2) | `qwen2`/`llama` | A/P | |
| DeepSeek-V3.2/Speciale (2) | `deepseek_v32` | P | DSA indexer no reader |
| DeepSeek-V4 Flash/Pro (2) | not installed | N | remote; mHC + compressed sparse attention unmodelled |
| Janus/Janus-Pro (1) | `janus` | N | AR image generation unmodelled |
| Gemma 1/CodeGemma/ShieldGemma (3) | `gemma`/`gemma2` | A | embedding scale is config debt |
| Gemma 2 (1) | `gemma2` † | A | witness |
| Gemma 3 (1) | `gemma3` † | P | per-layer rope θ transport ? |
| Gemma 3n (1) | `gemma3n` | P | PLE + KV-share A; AltUp/MatFormer N |
| RecurrentGemma (1) | `recurrent_gemma` † | R | RG-LRU N → mixer schedule fails |
| PaliGemma 2 (1) | `paligemma` | P | fusion/projector witnessed in tests |
| T5/Flan-T5/UL2/ByT5/mT5/T0/mT0 (5) | `t5`,`umt5`,`mt5` † | P | encoder A; decoder additive cross-attn A? |
| Switch (1) | `switch_transformers` | R | top-1 router without `topk` |
| Phi-1/1.5/2 (3) | `phi` | P | parallel residual + partial rotary (phi in tests) |
| Phi-3/3.5-mini/4/4-mini (5) | `phi3` † | R | fused-QKV slice → attention unknown |
| Phi-3.5-MoE (1) | `phimoe` | P | `sparse_mixer` router A; mask alias R? |
| Phi-3.5-vision / Phi-4-multimodal / reasoning (3) | `phi3`, `phi4_multimodal` | R/N | LoRA adapters no reader |
| Falcon 7B/2 (2) | `falcon` † | A | MQA protocol |
| Falcon 40B/180B (2) | `falcon` † | R | new-arch slices |
| Falcon-H1 (1) | `falcon_h1` | R | SSM-parallel hybrid |
| ChatGLM/2/3, CodeGeeX4 (2) | remote `chatglm` | R | control fixture: "geometry withheld" (`docs/U6_U7_U8_QUALIFICATION_MATRIX.md:127`) |
| GLM-4-9B (1) | `glm`/`glm4` | P | partial rotary slices A?, fused gate_up A |
| GLM-4V / 4.5V / 4.1V (2) | `glm4v`, `glm4v_moe` | P | |
| GLM-4.5/Air/4.6/4.7 (3) | `glm4_moe` † | A | witness glm-4-5 |
| GLM-5 line (1) | `glm_moe_dsa` | P | DSA no reader |
| Kimi-K2/Thinking/K2.5/K2.6/VL/Audio (5) | not installed (`kimi_k2` parsed by config in `coverage_audit.md` but no `kimi` modeling dir) | R? | remote DeepSeek-style `ModuleList` experts |
| MiniMax-Text-01/VL-01/M1 (3) | `minimax` † | R | lightning attention N |
| MiniMax-M2/M2.5 (1) | `minimax_m2` | A? | standard GQA + MoE (unopened) |
| Hunyuan-Large/A13B/dense (3) | `hunyuan_v1_moe`, `hunyuan_v1_dense` | P | CLA KV-sharing + shared expert unverified |
| ERNIE 4.5 dense/MoE/VL (3) | `ernie4_5`, `ernie4_5_moe` | P | heterogeneous multimodal experts unverified |
| StepFun Step-3 / Step-Audio (2) | remote | N | |
| InternLM 2/2.5/3, XComposer (3) | remote `internlm2` | R | `wqkv` + einops `rearrange` |
| Baichuan 2 / Omni (2) | remote | R | `W_pack` + `torch.split` (unverified), explicit-add ALiBi R |
| Cohere Command-R/R+/R7B/A, Aya, Aya-Vision (5) | `cohere`, `cohere2`, `aya_vision` † | A/P | interleaved rope A; layer_types sliding (cohere2) A? |
| OLMo/OLMo-2/OLMo-3/OLMoE/Molmo (4) | `olmo`, `olmo2` †, `olmo3`, `olmoe` † | A | witness olmo-2 |
| GPT-Neo (1) | `gpt_neo` | P | local attention via `torch.tril` buffer mask R (no framework builder) |
| GPT-J, GPT-NeoX, Pythia, RedPajama (4) | `gptj` †, `gpt_neox` † | P/A? | θ from table gather R? (gptj) |
| BLOOM/BLOOMZ (2) | `bloom` † | A | witness |
| MPT (2) | `mpt` | P | `Wqkv.chunk(3)` A; explicit-add ALiBi R |
| StableLM/Zephyr (1) | `stablelm` † | A | witness |
| DBRX (1) | `dbrx` † | A | witness (expert width withheld by design) |
| StarCoder2 (1) | `starcoder2` | P | mask alias R? |
| SantaCoder/StarCoder (1) | `gpt_bigcode` | P | MQA control: counts proven, `head_dim=None` (`docs/U6_U7_U8_QUALIFICATION_MATRIX.md:125`) |
| CodeGen (1) | `codegen` | P | `qkv_proj` mp_num reshape/permute + `torch.split` R?; ffn_width witnessed |
| Grok-1/2 (1) | remote | N | |
| Nemotron 3/4/H, Nemotron-3-Nano (2) | `nemotron`, `nemotron_h` | A?/R | Nemotron-H SSM hybrid R |
| Granite 3.x (1) | `granite` † | A | witness |
| Granite 4.0-H (1) | `granitemoehybrid` | R | Mamba2 blocks |
| OpenELM, DCLM (2) | remote / `llama`-like | ?/A | |
| EXAONE 3/3.5/4 (1) | `exaone`, `exaone4` | A? | layer_types local/global + QK-norm (unopened) |
| SmolLM3 (1) | `smollm3` | P | NoPE every 4th layer via `no_rope_layers` list guard ? |
| Jais (1), Teuken (1), regional (1), Danube (1), Arcee (1), Krutrim (1) | `jais2`/remote, remote, various, `llama`, `arcee`, remote | ?/A | |
| Sarvam-30b (finalize target) | unknown | ? | |

Part-1 tally (135 rows): **A ≈ 55**, **P ≈ 52**, **R ≈ 20**, **N ≈ 8** → ~41% clean, ~79% drawable with typed unknowns, ~21% refused or unmodelled.

**Part 2 — VLMs (24 rows)**: Qwen2-VL † A (witness); Qwen2.5-VL, Qwen3-VL, Qwen2.5-Omni P (window attention in tower / deepstack unverified); LLaVA-1.5 † A (wrapper witnessed in tests), LLaVA-NeXT/OneVision P (feature-selection reader covers `hidden_states[-2]` slice; AnyRes pooling unverified); Idefics2/3, SmolVLM P (perceiver via `repeated_projector`, tests); PaliGemma P; InternVL, Aria, Emu3, Chameleon, Fuyu, Kosmos-2, Florence-2, GOT-OCR2 P/? (installed, unopened); MiniCPM-V, CogVLM, Ovis, NVLM, DeepSeek-VL2, Moondream, DeepSeek-OCR, Qwen-VL N (remote code). Tally: A 2, P ≈ 12, N ≈ 10.

**Part 3 — image (37 rows)**: SD3/3.5 †, FLUX.1 dev/schnell/Krea/Kontext/Fill †, FLUX.2 †, Qwen-Image family †, PixArt α/Σ †, Sana †, Lumina-Image-2 †, AuraFlow † → **A** (12 rows, corpus witnesses); HunyuanDiT, Lumina-Next, CogView3/4, HiDream, Chroma, HunyuanImage-2.1, OmniGen → **A?** (in `scripts/dit_coverage.py`, "ALL CLEAN" per project memory, not re-verified here; 7 rows); SD1.4/1.5/2.x/SDXL †/Turbo/Lightning/Playground/Kolors/SSD-1B/DeepFloyd/Kandinsky-2.2/Marigold/Riffusion → **P** (11 rows: `UNet2DConditionModel` drawn from config + quarantined legacy readers, §2.11); Stable Cascade, Würstchen, Kandinsky 3 → **R** (`is_unet` by signature → skeleton + "code-defined placement" note, `adapters/diffusor/unet.py:44-58`); LDM, unCLIP/Karlo → P/?; HunyuanImage-3.0, aMUSEd, Janus/Show-o/Emu3, OmniGen2, BLIP-Diffusion/ControlNet → **N** (5). Tally: A 12, A? 7, P 13, R 3, N 5 → 51% clean-or-audited.

**Part 4 — video (21 rows)**: Wan 2.1/2.2 †, HunyuanVideo †, HunyuanVideo-I2V, LTX-Video †, CogVideoX 2B/5B/I2V/1.5 †, Mochi † → **A** (8); HunyuanVideo-1.5, LTX-2, SkyReels-V2, Allegro, Latte → **A?** (installed classes, unverified; 5); SVD → **P** (temporal declarations refused by `diffusion_bookends.py:22-30`; "SVD 3D-test open" in the run_77 ledger); ModelScope/Zeroscope (`UNet3DConditionModel`), AnimateDiff (`UNetMotionModel`) → **P?** (UNet config path; temporal cells unproven); Wan 2.5, Open-Sora, Open-Sora-Plan, Pyramid Flow, VideoCrafter, CogVideoX-Fun → **N** (6). Tally: A 8, A? 5, P 3, N 6.

**Part 5 — audio (26 rows)**: MusicGen/AudioGen † → **A** (2); Stable Audio Open (`StableAudioDiTModel`, 1-D audio DiT shipped in PR #14), ACE-Step, Dia, CSM, Moshi, Orpheus (`llama`), SeamlessM4T, Bark → **P/?** (installed, unopened; 8); AudioLDM-2, Tango, Riffusion → **P** (UNet path); Spark-TTS (`qwen2` + codec) P; everything else (MAGNeT, XTTS, Kokoro, F5-TTS, Chatterbox, CosyVoice, Fish, VibeVoice, IndexTTS, Higgs, MMS-TTS/VITS, MegaTTS…) → **N** (13). Tally: A 2, P ≈ 11, N 13.

**Catalogue acceptance (243 rows):** A/A? ≈ 96 (40%), P ≈ 91 (37%), R ≈ 23 (9%), N ≈ 33 (14%). Against the narrower `toserve.md` canonical list (118 ids, LLM Part A + diffusion Part B), the picture is better because it was written with the corpus in view: of its 41 LLM member rows ≈ 18 A, 17 P, 6 R (Phi-3/4, Falcon-40b, RecurrentGemma, Jamba, MiniMax, Nemotron-H); of its 24 diffusion rows ≈ 17 A/A?, 5 P, 2 N (HunyuanImage-3.0, OmniGen2).

### 5.4 `finalize.ipynb` — the author's own manual targets (18 ids, 24 cells)

| id | mapped | verdict | why |
|---|---|---|---|
| `Wan-AI/Wan2.2-TI2V-5B` | `WanTransformer3DModel` (single expert, 5B, `patch_size (1,2,2)`) | A? | Wan2.2-T2V-A14B is the witness; TI2V single-denoiser variant unverified |
| `hunyuanvideo-community/HunyuanVideo` | `HunyuanVideoTransformer3DModel` | A | witness |
| `unsloth/gemma-2-9b-it` | `gemma2` | A | witness gemma-2-2b-it (same source) |
| `google/recurrentgemma-9b` | `recurrent_gemma` | **R** | RG-LRU no reader → mixer schedule fails; attention layers only |
| `google/diffusiongemma-26B-A4B-it` | `diffusion_gemma` (installed) | **R/config** | block-diffusion canvas, dense‖MoE parallel FFN, self-conditioning all `extras:block_diffusion` debt; no reader |
| `google/gemma-4-e4b` | `gemma4`? (tests mention gemma4 as smoke control) | P | "custom config with plausible counts only → geometry withheld" (`docs/U6_U7_U8_QUALIFICATION_MATRIX.md:127`) |
| `EleutherAI/gpt-neox-20b` | `gpt_neox` † | A? | fused chunk(3) |
| `Qwen/Qwen2-VL-7B-Instruct` | `qwen2_vl` | A | witness |
| `codellama/CodeLlama-7b-hf` | `llama` | A | |
| `deepseek-ai/DeepSeek-R1`, `DeepSeek-V3.1-Base` | `deepseek_v3` | A | witness |
| `deepseek-ai/DeepSeek-V4-Flash` | not installed | **N** | remote only; new attention family |
| `ideogram-ai/ideogram-4-nf4` | unknown class | **N** | not a diffusers/transformers architecture I can map (unverified) |
| `meta-llama/Llama-3.2-11B-Vision` | `mllama` | P | |
| `mistral-community/pixtral-12b` | `llava` wrapper + `pixtral` tower | P | tower 2-D rope unverified |
| `openai/gpt-oss-120b` | `gpt_oss` | A | witness gpt-oss-20b |
| `sarvamai/sarvam-30b` | unknown (`sarvam-m` is mistral; 30b unverified) | ? | |
| `stabilityai/stable-diffusion-xl-base-1.0` | `UNet2DConditionModel` | P | witness, but config+legacy-drawn (§2.11) |

7 of 18 the author personally tests are clean-by-protocol; 3 are refused or unmodelled by design (RecurrentGemma, DiffusionGemma, DeepSeek-V4); the rest are partial.

---

## 6. The ten most recurring refusal shapes, ranked by how many ecosystem models they block

Ranking is by families in the catalogue + installed ecosystem that exhibit the shape (my count from §5; a real census would need a scripted AST scan over all 466 files, which this pass did not run).

| # | refusal shape | reader / string | blocks (examples) | est. families |
|---|---|---|---|---|
| 1 | **config-conditional mask-builder alias** `mask_function = create_causal_mask if config.sliding_window is None else create_sliding_window_causal_mask` | `attention_mask.py:870` "retains rival or conditional builders" (unverified whether the ifexp is reduced) | mistral, mixtral, qwen2, qwen2_moe, qwen3_moe, phi3, phimoe, minimax, starcoder2, granitemoe-shared?, every Mistral-derived remote model | ≥ 12 installed types, ~35 catalogue rows |
| 2 | **non-attention token mixer in the stack** (Mamba/Mamba-2, RG-LRU, RWKV, lightning/linear attention, short-conv) | no attention child (`attention_child.py:14-17`); "not every layer selects and invokes one proven mixer" (`mixer_schedule.py:446`) | jamba, mamba/mamba2, falcon_h1, granitemoehybrid, nemotron_h, bamba, zamba2, recurrent_gemma, minimax, lfm2, rwkv*, qwen3_next? (gated-delta is the one recurrent mixer with a reader) | ≥ 12 types, ~15 rows |
| 3 | **fused-QKV unpacked by subscript slices / reshape-permute** (`qkv[..., :n]`, `qkv[:, :, :, :-2]`, einops) | `attention_storage.py:562` "not an exact three-producer split or one-producer three-lane unpack" | phi3 family, falcon new-arch, codegen, internlm2, chatglm, baichuan (`W_pack` — unverified), moshi?, bark? | ≥ 6 types, ~15 rows |
| 4 | **learned/fixed absolute positions not fed by a bare `torch.arange` + scalar** (cumsum, buffer slice `self.position_ids[:, a:b]`, attribute offset, `.weight` gather, `nn.Parameter` add) | `position_coordinate.py:118-170`; `position_absolute.py:183` | opt, bert/roberta/electra/…, clip/siglip/vit/dinov2 towers, bart/mbart/blenderbot, whisper, deberta, xlm-r, gpt_neo? | ≥ 20 types (all classic encoders and towers) |
| 5 | **heterogeneous per-layer container** (`ModuleList` of different layer classes) | `execution_flow.py:196` "a template resolves only a single-element (homogeneous) container" | jamba, zamba2, bamba, falcon_h1, mllama (handled by `cross_attention_replacement`), granitemoehybrid | ≥ 6 types |
| 6 | **`trust_remote_code`-only families** — no installed modeling source; hub download path exists (`sources.py:14`) but the remote code uses shapes 2/3/7 | `missing_source` / shapes above | Qwen-1, ChatGLM, InternLM, Baichuan, Kimi, DeepSeek-V2-remote/V4, MiniCPM-V, CogVLM, Ovis, Step, Grok | ≥ 15 catalogue families (~30 rows) |
| 7 | **`ModuleList`-of-modules experts** (pre-5.x transformers, most remote MoEs) | `expert_storage.py:8-19` requires a ≥3-D Parameter or three repeated Parameters on one child | remote DeepSeek/Kimi/Qwen-MoE code; older installed versions | ~8 catalogue rows |
| 8 | **explicit-add ALiBi / non-bucket relative bias / disentangled bias** | `position_linear_bias.py:319`, `position_relative_bias.py:572`, `attention_score_additives.py` (neutral only) | mpt, baichuan-13b, jais, falcon-alibi (`cumsum` producer), deberta_v2/v3, xlnet | ≥ 6 types |
| 9 | **unsupported score transform** (per-position temperature, logit-scale attribute, custom scale factor) | `attention.py:1216` | llama4, deberta, cohere `logit_scale` on head (debt), gpt_neo (no scaling — raw is supported) | ≥ 3 types |
| 10 | **FFN split across two child modules / two-stage post-norm blocks** (BERT `intermediate`/`output` idiom) | `ffn_mechanism.py:626` (unverified) | bert, roberta, deberta, electra, vit, dinov2, wav2vec2-style encoders | ≥ 10 encoder types (only matters when drawn as towers) |

Diffusion-side recurring refusals (not ranked against the above because the population is smaller): conv-U denoisers that are not `UNet2DConditionModel` (`Kandinsky3UNet`, `StableCascadeUNet`, `UNet3DConditionModel`, `UNetMotionModel`, `UNetSpatioTemporalConditionModel`) fall to the `is_unet` skeleton; additive conditioning outside the three `diffusion_conditioning` shapes; temporal declarations vs operations (`diffusion_bookends.py:22-30`).

---

## 7. Loose ends

1. **Nothing in §5 was rendered.** Every A/P/R is a static classification of the modeling source against the reader protocols; the corpus witnesses are the only rendered ground truth. A scripted pass that runs `unfold()` over the 25-model sample (and over `dit_coverage.py`'s list) and diffs the reader statuses against this table is the obvious next step — and would turn the `?` cells into facts.
2. **Refusal shape #1 is the single biggest unknown.** Whether `attention_mask` reduces `A if config.x is None else B` through the exact config guard decides the mask verdict for the entire Mistral/Qwen2 lineage. `test_attention_mask.py` (39 tests) has no Mistral/Qwen2 control; the Qwen3 control uses the dict shape (`modeling_qwen3.py:413-418`). One test would settle it.
3. **Hybrid stacks are all-or-nothing today.** Because `mixer_schedule` requires *every* layer to carry a proven mechanism, a Jamba/Nemotron-H/Falcon-H1/RecurrentGemma stack loses even its attention layers' schedule. The explicit scope note in `patterns.py:57-59` predates the schedule readers; `toserve.md` promises "draw attention layers" for Jamba — that promise is not met by the current address rails (F2 homogeneity).
4. **The UNet is still not reader-drawn.** All eleven `unet_*` readers exist and are test-covered on SDXL/SD1.5, yet `adapters/diffusor/parser.py:57-113` still calls the five quarantined `patterns.unet_*_from_files` readers and the drawing flows through `extras:unet` (34 U11 debt rows). No `unet_*` fact is registered, so no receipt or qualification gate touches the UNet drawing.
5. **VAE and scheduler have no evidence layer at all** (33 + config rows); `scheduler_flow_matching` is derived from class-name markers, which is identity-as-display at best.
6. **Diffusion attention chips are only half migrated**: 80 `root.denoiser` U15 rows say QK-norm, rotary application/geometry, projection bias and cross-attention head geometry still reach the DiT card from config even though the U6/U8 readers run in `diffusion_block`.
7. **Positional coverage is asymmetric**: rotary is well-proven (93% application acceptance in the sample), learned/fixed absolute is nearly unproven (1/7). Since every encoder tower (CLIP/SigLIP/ViT) uses the refused shapes, the multimodal towers' position rows are all config or unknown.
8. **`IR` fields with no fact**: `compress_ratio`, `index_*`, `mrope_section`, `shared`, `no_rope`, `rope_3d`, `variant`, `cross_kv_source` are written by the parser without registry governance (`ir.py:79-116`) — the structural-writes census (`structural_writes.py`) pins them but nothing gates their values.
9. **Model-type spellings I could not map**: `ideogram-ai/ideogram-4-nf4`, `sarvamai/sarvam-30b`, `google/gemma-4-e4b` (the tests call it a "smoke control" whose geometry is withheld). The finalize notebook targets three architectures the product cannot draw by design (RecurrentGemma, DiffusionGemma, DeepSeek-V4).
10. **`torch.split(x, …)` functional form** and `.unbind()` as a three-lane unpack are unverified against `_proves_tensor_lane_unpack` (`attention_storage.py:950-975`); MiniMax's softmax lanes and Baichuan's `W_pack` depend on it.
11. **Witness concentration**: of 29 blessed witnesses, the LLM half is 14 models covering 13 model types; Falcon, GPT-2, T5, BERT, OPT are real-source *test* controls but not blessed galleries, so their end-to-end drawings are not pixel-locked. The 466-file ecosystem has never been scanned by any script in the repo (`coverage_audit.py` parses 49 configs; `dit_coverage.py` ~30 repos) — the acceptance rates above are the first estimate of the denominator and should be treated as such.
