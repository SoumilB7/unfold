# 02 — Pipeline trace: what happens when `unfold()` runs, and where the time goes

Written 2026-09-02 against `unfold-pkg` tree `9b3cb7b` ("feat(u11): prove root
preprocessing lineage"; 12 uncommitted paths untouched). Environment: Python
3.12.10, transformers 5.12.1, diffusers 0.38.0, macOS. All measurements are
wall-clock on this machine; scratch (profiles, dumps, scripts) lives under
`scratchpad/r02/` (`profile_run.py`, `fact_dump.py`, `sd35_failure.py`,
`sd35_reason.py`, `sd35_encoder.py`, `*.prof`, `*.timing.json`, `*.facts.json`).
Line numbers are for the tree above. Statements not verified by running are
marked **(unverified)**.

Subjects: `tests/sable_test_corpus/llama-7b.json` (LlamaForCausalLM, 32 layers),
`stable-diffusion-3-5-large.json` (SD3Transformer2DModel pipeline index + 3
fetched encoder configs), `qwen2-vl-7b-instruct.json` (composite text+vision).

Headline: `unfold()` does **no rendering** — it parses to IR and estimates params;
HTML/JSON are lazy on `Diagram.to_html()` / `to_json()`. Warm, the parse is
1.1 s (Llama), 3.2–3.7 s (SD3.5), 5.6–9.6 s (Qwen2-VL); render is 30–70 ms.
Cold (fresh process) the same calls take **25 s / 96 s / 114 s**, and 62–85 % of
that is one quadratic bug in the AST walker (`ast.get_source_segment` per
node) whose result is cached only in-process.

---

## 1. Ordered call chain — common entry

| # | call | consumes → produces | file:line |
|---|---|---|---|
| 1 | `unfold(cfg_or_id, token, inspect_code=False, code_source="local", return_json=False)` | anything HF-shaped → `Diagram` (or expanded JSON) | `model_unfolder/__init__.py:54-98` |
| 2 | `config_to_ir(cfg_or_id, ...)` | → `ModelIR` | `parser.py:23` |
| 3 | `_coerce_prepared` → `_raw_input` → `prepare_document(raw, loader_keys=LOADER_STAMPS, merge=False)` | dict / id / PretrainedConfig → `PreparedDocument` (document = the raw dict in shadow mode; `checkpoint` snapshot; `class_overlay`; `provenance` map). For a dict with `model_type` this calls **`AutoConfig.for_model`** on a deep clone — the first such call in a process imports transformers (+torch, +sklearn via `import_utils.packages_distributions`) | `parser.py:480-498`, `evidence/document.py:246-338` |
| 4 | `ParseContext.build(cfg, source="local", token)` | → `ParseContext` = `source_bundle` + `class_defaults_by_path` + `declared_decoderness`; empty `facts`/`config_access` ledgers, lazy `_program_index` | `evidence/context.py:249-265` |
| 4a | `_installed_config_defaults_by_path` → walks every nested dict (depth ≤5) and calls `_installed_config_defaults` → **`prepare_document` again** per dict carrying `model_type` | class overlay per document path | `context.py:268-323` |
| 4b | `resolve_source_files(target, source="local")` | config → `SourceBundle(files, component_files, supporting_files, component_model_types, component_architectures, import_roots, pipeline_components, warnings)` | `evidence/sources.py:14-86` |
| 5 | `find_adapter(cfg)` — `custom.ADAPTERS` (empty) → `diffusor.parser.matches` → `transformer.parser.matches` (always True) | → adapter module | `adapters/__init__.py:11-15`, `adapters/diffusor/parser.py:255-278`, `adapters/transformer/parser.py:867-868` |
| 6 | root `DocumentBinding("root", (), prepared)` registered; enter `capture_events` + `owner_scope("root")` + `bound_document` + `capture_facts`; set `active_parse_context` | ledgers armed | `parser.py:91-104` |
| 7 | `adapter.parse(cfg, context=parse_context)` | → `ModelIR` (see §2 / §3) | `parser.py:102` |
| 8 | post-parse accounting: `source_provenance`, unread/pending-debt join (`_config_debug.unparsed_fields`, `pending_projection_paths`, `pending_classification_paths`), `config_ambiguity`, `config_access` (accessed/consumed/absent_default/obligations/migration_claims/…), fold spec `asserted` tags into the `FactLedger`, `fact_provenance`, `config_consumed` | `ir.extras[...]` | `parser.py:105-393` |
| 9 | `_ensure_parsable` (layers, or `extras.unet`, or diffusion render with `opaque_layer_block`), `_debug_validate_blocks` (off), optional `_attach_code_evidence` (only with `inspect_code=True`) | raise or return IR | `parser.py:394-398`, `412-445` |
| 10 | `Diagram(ir)` → `estimate_params(ir)` | params dict | `diagram.py:20-27`, `params.py:115` |
| 11 (lazy) | `to_html()` → `_html` → `RenderContext(theme, fact_rows=extras.fact_provenance)` → `render_document(self.to_ir(), mount_id)` → `render_fragment` → family switch (`diffusion` → `views_diffusion.render_diffusion_fragment`; else `_render_fragment_body`) → `_make_info` → per group `_block_lookup` / `_meta_for` → `_build_architecture_view` (SVG from `spec.blocks`) → `_build_inspect_cards` / `_build_nested_inspect_panels` (block views call `opgraph.attention_region` / `ffn_region` / `ops_region` then `graph_engine.render_graph`) | HTML string, cached per `standalone` | `diagram.py:156-181`, `renderers/html/document.py:14-38`, `metadata.py:26,96-123`, `views.py:112`, `block_views/attention.py:55,261`, `block_views/feed_forward.py:24,38`, `opgraph.py:140,467,1188`, `graph_engine.py:64` |
| 12 (lazy) | `to_json()` → `expanded.build_expanded(ir, params)` | expanded schema dict | `diagram.py:49-59`, `expanded/__init__.py:46` |

Notes on the entry: `ParseContext.build` at `parser.py:57` is unconditional even when
the caller passed `parse_context=None`, so every by-dict call pays both step 3 and 4a
hydrations; the `identity_address` decorator on `prepare_document` /
`_installed_config_defaults` is a marker only (`evidence/identity_roles.py:38-46`).

## 2. Transformer subject (Llama-7B) — inside `adapters/transformer/parser.py::parse`

`parse` is a single function, lines **876–4644** (3,769 lines; the file is 4,996
lines — the task brief's 3,762 figure is stale). Its section markers:

| lines | section | what it consumes / notes | where it notes facts |
|---|---|---|---|
| 876-1013 | setup: `debug.reset()`, `_note_fact` / `_note_bound_attention_fact` / `_note_typed_fact` closures | `context.facts` (FactLedger), `_source_present` | — |
| 1014-1175 | document/scope helpers: `_replace_component`, `_nested_scope`, `_scoped` (alias resolution via `config_access.resolve` with `_ALIASES` from `everchanging/transformer/aliases.yaml`), alias-group classification | config document, class defaults by path | ambiguity events |
| 1176-1330 | the config accessor: `get` (inspect, may discard), `consume` (flows into a fact: `res.consume(fact_owner, fact_key)`), `_consume_code_bound_path` (consume only the exact path the source reader named), `_resolve_exact_config_path` (no alias search; class-default fallback) | every config read goes through `config_access.resolve` → `ConfigAccessLedger` events | consumption obligations |
| 1347-1370 | first reader: `_ffn_mechanism_result` (**this is what triggers `context.program_index()` — the whole index build happens here on the cold path**) | ProgramIndex + `decoder_ffn_mechanism_for_path` | `decoder.ffn.{gated,projection_mode,activation}` |
| 1592-1617 | U8-C mask execution authority (`attention_mask.decoder_attention_mask_execution_for_path`) | | `decoder.attention.mask`, `mask_schedule` |
| 1618-1922 | U8-B positional operation authority: `position_schedule`, `position_absolute`, `position_fixed`, `position_linear_bias`, `position_relative_bias`, `position_initialization` | | `decoder.attention.position_schedule`, `rope_theta`, `rope_initialization` |
| 1923-2066 | mixer schedule, attention geometry, FFN schedule | `mixer_schedule`, `attention_geometry`, `ffn_schedule` | `mixer_schedule`, `ffn_schedule` |
| 2067-2416 | mask mechanisms, cell topology (`_cell_topology_result`), output projection | `attention_mask`, `cell_topology`, `attention_output` | `decoder.layer.{norm_placement,residual_topology,parallel_norm_count}` |
| 2417-2859 | **Attention shape**: `_attention_mechanism_result`, head geometry, gated-delta geometry | `attention`, `attention_geometry` | `decoder.attention.{mechanism,head_geometry}` |
| 2860-2982 | Position encoding + QK-norm (`qk_norm_schedule`) | | |
| 2983-3031 | Cross-layer K/V reuse (`kv_sharing_schedule`) | | |
| 3032-3373 | Q/K/V/O bias (`projection_bias`), score scaling, logit softcap, qkv clip, cache | | `bias`, `scores_scale`, `cached` |
| 3374-3502 | Layer topology (norm kind, residual scale) | `decoder_norm` | `norm_kind` |
| 3503-3780 | MoE (`expert_width`, `expert_storage`, `router`, shared experts) | | |
| 3781-3852 | per-layer side input (`per_layer_side_input`) | | |
| 3853-3981 | cross-attention decoder layers (`cross_attention_replacement`, `cross_attention_schedule`) | | |
| 3982-4543 | **Walk the layer stack**: `for i in range(num_layers)` assembles `AttentionSpec`/`FFNSpec` per layer from the schedules computed above, then `assembly.decoder_layer(...)`; modality towers (`component_tower.recursive_component_mechanisms`, `multiaxis_position`, `projector`, `wrapper_features` — all via `context.cached_reader_result`), codebook streams, MTP | schedules | spec-level `asserted` tags |
| 4544-4644 | block diffusion (masked text LMs) + `ModelIR(...)` return | | |

Reader invocation order and helper wrappers (each wrapper memoizes through
`context.cached_reader_result(name, config_path, factory)`, `context.py:238-247`):
`_cell_topology_result`→`cell_topology.decoder_cell_topology_for_path` (99-224);
`_decoder_norm_result`→`decoder_norm.decoder_norm_kind_for_path` (225-242);
`_ffn_mechanism_result`→`ffn_mechanism.decoder_ffn_mechanism_for_path` (243-292);
`_code_final_norm`→`final_bookend.final_stage_norm_evidence` (308-321);
`_attention_storage_result`→`attention_storage.decoder_attention_projection_storage_for_path` (322-339);
`_attention_mechanism_result`→`attention.decoder_attention_mechanism_for_path` (340-356);
`_attention_head_geometry_result`→`attention_geometry` (357-373); gated-delta (374-389);
`_projection_bias_result` (399-427); output projection (438-454); `_router_result` (455-512);
score scaling / softcap / qkv-clip / cache (513-580, all in `evidence/attention.py`);
sinks (594-620); cross-attention schedule (621-637); FFN width (638-653); expert storage/width (654-711).
The 27 cached reader keys for Llama are listed in §6.

Owner resolution: every reader that needs the decoder block starts from
`decoder_block.decoder_block_path_for_config` (`evidence/decoder_block.py:426`) →
`decoder_block_candidates_for_config` (:463) → **`component_owner.resolve_component_root(index, bundle, "root")`**
(`component_owner.py:1531`) → `resolve_owner_graph` (:356) → `_Resolver.build`
(:509). None of these are memoized — see §5.

## 3. Diffusion subject (SD3.5-large) — `adapters/diffusor/parser.py::parse`

| # | call | consumes → produces | file:line |
|---|---|---|---|
| D1 | `matches(cfg)`: `_class_name="SD3Transformer2DModel"` → `sources._installed_diffusers_model_class_file(cls)` (lru) is not None → True | routes before the catch-all | `diffusor/parser.py:255-278` |
| D2 | source resolution (step 4b) goes `_installed_transformers_bundle` (no files: root has no `model_type`) → `_installed_diffusers_bundle` → root file `diffusers/models/transformers/transformer_sd3.py`; for each `_text_encoder_configs.*` slot calls `_installed_transformers_bundle(enc_cfg)` and prefixes its component keys; `import_roots["root"]=(diffusers root,)` | bundle with 4 components, 3 distinct modeling files + 6 supporting | `sources.py:33-41`, `529-607`, `614-631` |
| D3 | `parse` (decorated `@owner_scoped("root.denoiser")`, `ROOT_COMPONENT="root.denoiser"`) → `_shadow_diffusion_root_topology(context)` → `context.program_index()` (**index build**) + `_shadow_diffusion_root_resolution` → `component_owner.resolve_component_root(index, bundle, "root")` → `diffusion_root.read_diffusion_root_topology(index, root)` | `ReaderResult[DiffusionRootTopology]` kind=`repeated_stack`, status **incomplete** | `diffusor/parser.py:617-637`, `284-311`, `evidence/diffusion_root.py:406` |
| D4 | not `u_shaped` → `_parse_projected_denoiser(cfg, arch, context, _bound_diffusion_source_projection(context, cfg))` | | `diffusor/parser.py:642-644` |
| D5 | `_bound_diffusion_source_projection` → `config_binding.bind_diffusion_source_projection(index, root, binding, topology, companions)` → internally the lru-cached `_source_only_diffusion_stack_and_blocks(index, root)` = `diffusion_stack.read_diffusion_stack_inventory` + `diffusion_block.read_diffusion_block_facts`; `_shadow_diffusion_stream_and_conditioning`; `_shadow_diffusion_bookends`; `_shadow_diffusion_companions` | `BoundDiffusionSourceProjection` (has_value=True, status **incomplete**, 10 typed failures, **0 operands**) | `diffusor/parser.py:483-519`, `312-339`, `config_binding.py:680` |
| D6 | `_parse_projected_denoiser`: `bound_result.has_value` → `projection_ir.project_diffusion_ir(...)` → `projection.layers == []`, `templates == []`; `conditioning_proven` computed from bookend applications; `_projected_pipeline_handoffs(cfg, context, ...)` | | `diffusor/parser.py:562-596`, `projection_ir.py:1206` |
| D7 | `_projected_pipeline_handoffs` → `_text_encoder_specs(cfg, context)`: for each of `text_encoder`, `_2`, `_3`: `prepare_document(sub)` → `DocumentBinding(f"root.{key}", ("_text_encoder_configs", key), prepared)` → `owner_scope` + `bound_document` → `encoder_panel.normalize_encoder_config(doc, context=slot_parse_context(...), binding)` → **`adapters.transformer.parser.parse(doc, context=slot_ctx)`** (a full recursive transformer parse with its own `ParseContext`, own `FactLedger`, own lazily-built `ProgramIndex` on the re-rooted sub-bundle) → `_project_encoder_spec` → `submodel_spec(ir, altitude="tower")`; then `_vae_geom`, `_scheduler_geom` | `text_encoder_specs` list (+ `vae`, scheduler geom) | `diffusor/parser.py:522-542`, `910-993`, `encoder_panel.py:27-80`, `evidence/context.py:334-443` |
| D8 | `bound_result.has_value` is True (an empty projection is still a value), so the **projected** branch runs: `diffusion_projected_render_spec(projection, handoffs)` (`blocks.py:191-246`) — with 0 templates it emits `projected_layers=None`, `model_blocks=[]` and an `opaque_layer_block` titled "Repeated denoiser structure unresolved"; the render key set (`family, layout, loop_blocks, loop_edges, loop_region, model_blocks, opaque_layer_block, theme`) is identical to the `diffusion_opaque_render_spec` fallback (`blocks.py:249-285`), so the two paths are indistinguishable by keys. `warnings.extend(projection.unresolved)` adds nothing (empty) | `ModelIR(layers=[], hidden_size=0, extras={"render": …})`, **no IR warning** | `diffusor/parser.py:570-596, 598-614` |
| D9 | `_ensure_parsable` accepts the zero-layer IR through the U10-F3 exemption (`family=="diffusion"` and `model_blocks` and `opaque_layer_block`) | | `parser.py:429-434` |

Qwen2-VL (composite transformer) follows §2 with `config_path=("text_config",)`
for decoder readers, plus the modality tower branch at `transformer/parser.py:4278-4300`
(`recursive_component_mechanisms` over `vision_config`, 2.25 s warm).

## 4. Stage-timing table

Measured by wall-clock wrappers around the named functions (`profile_run.py`), three
calls in one process: **cold** = first call, **warm** = second, **warm2** = third.
Times in seconds. `unfold` = `config_to_ir` + `Diagram()`; `to_html` / `to_json` are
the lazy consumers called afterwards. Package import itself: 0.33–0.39 s.

| stage | Llama cold | Llama warm | SD3.5 cold | SD3.5 warm | Qwen2-VL cold | Qwen2-VL warm |
|---|---|---|---|---|---|---|
| `_coerce_prepared` (root hydration) | **8.38** | 0.002 | 0.004 (no root `model_type`) | 0.001 | **9.79** | 0.018 |
| `_installed_config_defaults_by_path` | 0.002 | 0.002 | **12.03** (first transformers import lands here) | 0.005 | 0.020 | 0.022 |
| `resolve_source_files` | 0.012 | 0.013 | 2.87 (includes `import diffusers` 2.36) | 0.033 | 0.063 | 0.075 |
| `build_program_index` (calls) | **15.59** (1) | 0.005 (1) | **77.75** (4) | 0.019 (4) | **97.59** (1) | 0.010 (1) |
| `_observe_source` (calls / lru misses) | 15.57 (4/4) | 0.0001 (4/0) | 77.59 (19/15) | 0.0007 (19/0) | 97.53 (12/12) | 0.001 (12/0) |
| adapter `parse` total | 16.71 | **1.107** | 80.45 (of which 3 encoder sub-parses 29.40) | **3.13** (encoders 2.94) | 104.05 | **5.14** (warm2 9.05) |
| `estimate_params` | 0.0004 | 0.0004 | 0.0 | 0.0 | 0.0008 | 0.0004 |
| **`unfold()` total** | **25.12** | **1.14** | **95.76** | **3.68** (warm2 3.15) | **114.35** | **5.65** (warm2 9.64) |
| `to_html()` (bytes) | 0.039 (55,139) | 0.040 | 0.030 (159,466) | 0.031 | 0.068 (106,271) | 0.044 |
| `to_json()` | 0.045 | 0.045 | 0.000 (0 layers) | 0.000 | 0.034 | 0.030 |

Three cold facts worth stating plainly:

1. Rendering is negligible (≤70 ms). Everything is in the evidence layer.
2. The parse-proper (readers + assembly) is 1.1 s / 0.2 s denoiser + 2.9 s encoders /
   5–9 s. Cold adds 24 s / 92 s / 109 s of index build + library import that is
   **not persisted across processes**.
3. Qwen warm2 (9.6 s) > warm (5.6 s) reproduced once; not investigated **(unverified
   cause; GC or hash-seed dependent set ordering are candidates)**.

## 5. Where the time goes — profile findings

### 5.1 Cold: two costs dominate

**(a) The AST walker re-splits the whole file for every expression node.**
`_SourceWalker._seg` (`evidence/program_index.py:2355-2361`) calls
`ast.get_source_segment(self.text, node)` for every `_expr` node; CPython's
`get_source_segment` calls `_splitlines_no_ff(source)` on the **entire file** each
time (`ast.py:308`). Llama cold: 17,149 `get_source_segment` calls, 14.14 s of
15.10 s inside `_splitlines_no_ff`; SD3.5: 76,588 calls, 69.45 s; Qwen (12 sources)
~97 s. That is 62 % (Llama), 81 % (SD3.5), 85 % (Qwen) of the cold `unfold()`. Cost
is O(nodes × file lines) per file; `modeling_qwen2_vl.py` and `modeling_t5.py` are
long files with many nodes, hence the super-linear blow-up.

**(b) The first `AutoConfig.for_model` imports transformers + torch (+ sklearn).**
Wrapper: 8.4 s (Llama `_coerce_prepared`), 12.0 s (SD3.5 — lands in
`_installed_config_defaults_by_path` because the root has no `model_type`), 9.8 s
(Qwen). Isolated measurement: `import transformers; from transformers import
AutoConfig` 3.68 s; `AutoConfig.for_model("llama", …)` 3.23 s, and **torch is
imported during `for_model`**. cProfile shows `transformers/__init__` 3.1–5.2 s and
`sklearn/__init__` 1.9–2.9 s (pulled in by `transformers.utils.import_utils` →
`importlib.metadata.packages_distributions`, 1.85–4.4 s). Note: cProfile attributes
`prepare_document`/`for_model` rows as `prim=0, cum=0.00` (a profiler artifact —
**the wrapper wall-clock and the isolated timing are the authority here**).

Top-25 cumulative (Llama cold, `llama-7b.cold.top60.txt`, condensed):
`parse` 16.71 → `cached_reader_result` 15.85 → `_ffn_mechanism_result` 15.61 →
`context.program_index` 15.60 → `build_program_index` 15.59 → `_observe_source` 15.57
→ `_SourceWalker.run` 15.56 → `_callable` 15.37 → `_expr` 15.36 (17,310 calls) →
`_seg` 15.23 → `ast.get_source_segment` 15.13 → `_splitlines_no_ff` 15.10 (14.14 tottime)
→ `_visit_stmts`/`_stmt` 14.90 → `_class` 12.43 → `_assign` 8.24 → `_walk_expr` 4.83
→ `_record_call` 3.34; import chain: `transformers/__init__` 3.10, `sklearn` 1.90,
`importlib.metadata.packages_distributions` 1.85, torch `_ops.fallthrough` 11.0 (recursive
cumulative, nested under the import).

### 5.2 Warm: the readers, and how often they repeat

Llama warm (`llama-7b.warm.top60.txt`; parse = 1.107 s):

| cumulative | calls | function | why it repeats |
|---|---|---|---|
| 0.526 | 1 | `position_schedule.decoder_position_application_schedule_for_path` (:199) | loops **`for layer_index in range(transport.layer_count)`** (:238-244) calling the full `decoder_qk_half_turn_application_for_path` once per layer (32 calls, 0.48 s), each re-resolving the block path |
| 0.409 | 59 | `attention_child.attention_child_positive_census` (:222) | called by `attention_child_evidence` (54), `mixer_schedule` (3), `attention_mask` (2) — no memo |
| 0.325 | 79 | `decoder_block.decoder_block_candidates_for_config` (:463) | called by 8 distinct readers (`decoder_block_path_for_config` 59, score additives 4, mixer 3, mask ×2 readers, ffn schedule, alibi, score scaling); each re-runs **`resolve_component_root`** (83 calls, 0.243 s) → `resolve_owner_graph` → `_Resolver.build` |
| 0.306 | 6,313 | `construction_calls.resolve_import_reference` (:330) | per-site import resolution, no memo |
| 0.145 | 337 | `execution_flow.resolve_addressed_invocations` (:305) | |
| 0.086 | 1 | `attention_mask.decoder_attention_mask_execution_for_path` | |
| 0.070 | 8 | `everchanging.load` (YAML `safe_load`) | `load_composite_slots` ×6 per parse, `load_program_index_vocab`, `load_decoderness` — **YAML is re-read and re-parsed on every call; there is no cache in `everchanging/__init__.py:32-39`** |
| 0.067 | 96 | `ir.layer_signature` | signature per layer for grouping |
| 0.120 tottime | 225,962 | `builtins.hash` | frozen dataclass hashing of `SourceId`/`SymbolId`/spans |

SD3.5 warm (3.13 s): `_text_encoder_specs` 3.09 → 3× `transformer.parse` 2.94 (so the
denoiser branch proper is ≈0.2 s); `decoder_block_candidates_for_config` **149 calls,
1.53 s**; `everchanging.load` **72 calls, 0.71 s** — 64 of them `load_composite_slots`
called from `debug.component_prefix_owners` (`transformer/debug.py:115`) ← `parser.py:170
_unread_path_owner`, i.e. the post-parse **config audit re-reads YAML once per unread
path**; `resolve_component_root` 163 calls; `position_schedule` 3 calls 0.57 s.

Qwen2-VL warm (5.14 s): `decoder_block_candidates_for_config` 123 calls 2.31 s;
`position_schedule` 2.29 s (56 per-layer calls); `component_tower.recursive_component_mechanisms`
2.25 s (vision tower: `_position_results` 1.49 s); `resolve_owner_graph` **266 calls**
1.93 s; `resolve_import_reference` 12,466 calls 1.65 s; YAML 42 loads 0.44 s.

### 5.3 What is cached where (and what is not)

| cache | key | scope | evidence |
|---|---|---|---|
| `program_index._observe_source` `@lru_cache(128)` | `(SourceId(path, content fingerprint, component_key), text, vocab key, factory key)` | **process** | `program_index.py:2773-2810`; the file is still read+hashed every build (:2864-2884). Cross-process: **nothing persisted** — every fresh interpreter (CLI, notebook kernel, each pytest process, each Sable run) re-pays §5.1(a). |
| `ParseContext._program_index` / `component_inventory` | one per context | call | `context.py:221-236` |
| `ParseContext.reader_results` via `cached_reader_result(name, config_path)` | `(reader, path)` | call | `context.py:238-247`; 27 keys for Llama, 31 Qwen, 4 root keys SD3.5 |
| `ProgramIndex._address_index` `@cached_property` | per index | call | `program_index.py:2523` |
| `adapters/diffusor/parser.py` U10-C process caches `@lru_cache(64)` on `_source_only_diffusion_stack_and_blocks`, `_source_only_diffusion_stream_and_conditioning`, `_source_only_diffusion_bookends` | `(ProgramIndex, root)` — hashes the immutable index (fingerprint-bearing) | process | `:312-339, 368, 417` |
| `sources._transformers_file_for_class` (128), `_installed_diffusers_model_class_file` (64), `_diffusers_class_file` (64) | class name | process | `sources.py:309, 425, 453` |
| `structural_debt` (256), `transitive` (256), `forward_ops` (256), `patterns` (128), `vision` (128) lru caches | path-keyed | process | none of `transitive/forward_ops/patterns/vision/conformance` appear in any warm `unfold` profile — they are Sable/legacy-only paths (verified: `ast.parse` fires once in warm Qwen, 14 ms, from `sources._architecture_from_config_class`) |
| `Diagram._ir_cache`, `_html_cache[standalone]`, `_json_cache` | per Diagram | object | `diagram.py:25-27` |
| **not cached**: `everchanging.load` (YAML), `resolve_component_root` / `resolve_owner_graph`, `decoder_block_candidates_for_config`, `attention_child_positive_census`, `resolve_import_reference`, `prepare_document` (called ≥2× per document per parse) | | | |

### 5.4 Repeated work, quantified

- **Same source walked more than once, by design.** `build_program_index` indexes a
  shared file "once PER owning component" (`program_index.py:2821-2823`) and the lru
  key includes `component_key` (:2784-2785). Qwen: `modeling_qwen2_vl.py` + 3
  supporting files × 3 components = **12 walks of 4 distinct files** (97.5 s cold).
  SD3.5: the root index has 10 nodes; each of the 3 slot contexts re-roots its
  component to `"root"` (`context.py:357-366`), so the slot index's `SourceId` differs
  from the root index's → **15 walks of 6 distinct files** (19 observations, 4 hits).
- **Same reader per layer.** `position_schedule` (32/28/12 per-layer calls);
  `decoder_block_candidates_for_config` ×79/149/123; `resolve_component_root` ×83/163/137
  per parse.
- **Same document hydrated twice.** Root: `_coerce_prepared` (`parser.py:53`) and
  `_installed_config_defaults` (`context.py:287`); each nested dict with `model_type`
  again in the walk; each encoder slot again at `diffusor/parser.py:962`. Only the
  first pays the import; later ones are ms.
- **YAML per call**: 8 / 72 / 42 loads per parse.

### 5.5 The three biggest latency levers (in order)

1. **Fix `_seg`** — split lines once per file and slice by `(lineno, col_offset,
   end_lineno, end_col_offset)` instead of `ast.get_source_segment` (`program_index.py:2355-2361`).
   Expected: removes ~14 s of 15.6 s (Llama), ~69 of 78 s (SD3.5), ~90 of 98 s (Qwen)
   from cold. Pure optimization: `source_segment` text is unchanged. **(expected, unverified)**
2. **Persist the observation slice across processes** (or at least share walks across
   component keys and slot re-rooting). The cache key is already content-addressed
   (`SourceId.content_fingerprint`, vocab keys, factory names), so a disk cache keyed by
   `(canonical_path, fingerprint, vocab_key, factory_key)` with the component key
   applied on load would make cold ≈ warm + library import. Remaining cold floor after
   lever 1 is then the transformers/torch import (~7–10 s) — lever 3.
3. **Avoid importing torch to hydrate a config**: `AutoConfig.for_model` is the only
   reason torch loads (`document.py:306`). Options: hydrate via the config class's
   `__init__` signature without importing `transformers` top-level **(unverified
   feasibility)**, or lazily hydrate only when a reader asks for a class default.
   Warm-path levers (smaller, but multiply for corpus/Sable runs): memoize
   `resolve_component_root`/`decoder_block_candidates_for_config` per
   `(index, bundle, config_path, allow_root_stage)`; hoist the per-layer loop in
   `position_schedule` to one graph resolution + per-layer selector evaluation; cache
   `everchanging.load` by file mtime.

## 6. Fact inventories (what the evidence layer produces)

### 6.1 Llama-7B (`llama-7b.facts.json`)

Source bundle: `root` → `transformers/models/llama/modeling_llama.py`; supporting
`configuration_llama.py`, `modeling_rope_utils.py`, `configuration_utils.py`; no
warnings. ProgramIndex: 4 source nodes, 15 classes, 100 callables, 27 construction
sites, 0 parse failures. `class_defaults_by_path` = `{(): 1 field}`;
`declared_decoderness = "architectures[LlamaForCausalLM]"`.

Reader results (27 cached keys, all at `config_path=()`): resolved — `decoder.ffn.mechanism`,
`ffn.intermediate_width`, `attention.position.rope_schedule`, `position.frequency_initialization`,
`attention.mixer_schedule`, `ffn.schedule`, `layer.norm_kind`, `layer.cell_topology`,
`attention.projection_storage`, `attention.output_projection`, `attention.mechanism`,
`attention.head_geometry`, `attention.projection_bias`, `attention.score_scaling`,
`attention.cached`, `model.final_norm_kind`; absent — `position.fixed_absolute`,
`position.alibi`, `position.relative_bias`, `mtp_modules`; failed (typed) —
`position.learned_absolute` (incomplete_graph "no code-proven coordinate value feeds an
embedding primitive"), `gated_delta_geometry` (unsupported_syntax), `attention.sinks`
(incomplete_graph "no unconditional learned Parameter"), `logit_softcap`, `qkv_clip`
(unsupported_syntax), `ordinary_ffn.projection_bias` (unsupported_syntax "the exact bias
expression is not a source-only boolean" — yet `decoder.attention.bias=False` resolved via
the code-bound `mlp_bias`/`attention_bias` path), `ffn.expert_intermediate_width`.

Fact ledger (`extras.fact_provenance`, 24 rows):

| owner.fact | value | status | source |
|---|---|---|---|
| decoder.attention.bias | False | code_and_config | decoder_attention_bias_for_path |
| decoder.attention.cached | True | code_proven | decoder_attention_cache_for_path |
| decoder.attention.head_geometry | mha 32/32/128 | code_and_config | decoder_attention_head_geometry_for_path |
| decoder.attention.mask / mask_schedule | causal ×32 | code_and_config | decoder_attention_mask_execution_for_path |
| decoder.attention.mechanism | mha | code_and_config | decoder_attention_mechanism_for_path |
| decoder.attention.mixer_schedule | ordinary_attention ×32 | code_and_config | decoder_mixer_schedule_for_path |
| decoder.attention.output_projection | True | code_proven | decoder_attention_output_projection_for_path |
| decoder.attention.position_schedule | rope / qk_rotation ×32 | code_and_config | decoder_position_application_schedule_for_path |
| decoder.attention.projection_mode | split_qkv | code_proven | decoder_attention_projection_storage_for_path |
| decoder.attention.rope_initialization | local_default `LlamaRotaryEmbedding.compute_default_rope_parameters` | code_and_config | position_frequency_initialization |
| decoder.attention.rope_theta | 10000.0 | code_and_config | position_frequency_initialization |
| decoder.attention.scores_scale | sqrt(head_dim) | code_proven | decoder_attention_score_scaling_for_path |
| decoder.ffn.activation | silu | code_and_config | decoder_ffn_mechanism_for_path:hidden_act |
| decoder.ffn.ffn_schedule | dense ×32 | code_and_config | decoder_ffn_schedule_for_path |
| decoder.ffn.gated | True | code_proven | decoder_ffn_mechanism_for_path |
| decoder.ffn.intermediate_size | 11008 | code_and_config | decoder_ffn_intermediate_width_for_path |
| decoder.ffn.projection_mode | split | code_proven | decoder_ffn_mechanism_for_path |
| decoder.layer.norm_kind | rmsnorm | code_proven | decoder_norm_kind_for_path |
| decoder.layer.norm_placement | pre | code_proven | decoder_cell_topology_for_path |
| decoder.layer.parallel_norm_count | None | **ambiguous** | — (recorded at `transformer/parser.py:3440-3441` whenever the cell topology carries no parallel count; for a sequential cell this is a mislabel of "not applicable" as "ambiguous", and it is what produces the Llama warning "parallel norm count — modeling source is present but unresolved") |
| decoder.layer.residual_topology | sequential | code_proven | decoder_cell_topology_for_path |
| model.final_norm_kind | rmsnorm | code_proven | final_stage_norm_evidence |
| model.tie_word_embeddings | False | config_declared | tie_word_embeddings |

Tier census: 12 code_and_config, 10 code_proven, 1 config_declared, 1 ambiguous,
0 asserted, 0 class_default, 0 oracle_missing.

Config access (`extras.config_access`, owner `root`): **accessed (present) 20** —
`_name_or_path, architectures, attention_bias, head_dim, hidden_act, hidden_size,
intermediate_size, is_encoder_decoder, max_position_embeddings, mlp_bias, model_type,
num_attention_heads, num_hidden_layers, num_key_value_heads, rms_norm_eps,
rope_parameters, rope_theta, rope_type, tie_word_embeddings, vocab_size`;
**consumed 16** (the same minus `_name_or_path, architectures, model_type, rope_parameters`);
**absent_default 29** premises (e.g. `attn_logit_softcapping, clip_qkv, num_experts,
sliding_window, swiglu_limit, query_pre_attn_scalar, …`); `accessed_unconsumed` []; 22
projection obligations (20 `unreceipted`, 2 `pending`); `consumed_unprojected` 14;
migration claims 3 with 0 violations; no `non_checkpoint`, `unestablished_provenance`,
`accessed_unresolved_path`, `audit_incomplete`, or `config_ambiguity`.
`config_audit.unread` = [] (the 35-key fixture is fully classified: 21 accessed, the
rest are in the ignore vocabulary). `config_consumed` = the 16 above.
IR warnings: one (the parallel-norm-count line).

### 6.2 SD3.5-large — root (`root.denoiser`) and `root.text_encoder`

Source bundle: 4 components (`root` → `transformer_sd3.py`; `text_encoder`/`_2` →
`modeling_clip.py` (+`configuration_clip.py`, `configuration_utils.py`);
`text_encoder_3` → `modeling_t5.py` (+`configuration_t5.py`, `configuration_utils.py`));
architectures `SD3Transformer2DModel / CLIPTextModelWithProjection ×2 / T5EncoderModel`;
`import_roots["root"]` set. Root ProgramIndex: 10 nodes, 61 classes, 293 callables,
164 sites, 0 parse failures. `class_defaults_by_path` has only the 3 encoder paths
(root has no `model_type`); `declared_decoderness=None`.

Root reader results (4 keys): `root.denoiser.component_root` resolved;
`root.denoiser.topology` **incomplete** (has_value, kind=`repeated_stack`);
`root.denoiser.companions` absent; `root.denoiser.bound_source_projection` **incomplete**
(has_value, 0 operands, 10 failures — listed in §7).

Root fact ledger: **1 row** — `root.denoiser.diffusion_root_topology = repeated_stack |
code_proven | "exact diffusion source occurrence"`. `projection_receipts` = 1.

Config access by owner (accessed / consumed): `root.denoiser` 8 / **0** (all 8 are
addresses: `_class_name, _repo_id, _scheduler_config, _text_encoder_configs, _vae_config,
text_encoder, text_encoder_2, text_encoder_3`); `root.text_encoder` 11 / 6;
`root.text_encoder_2` 11 / 6; `root.text_encoder_3` 18 / 13; `root.vae` 16 / 14;
`root.scheduler` 3 / 1. `absent_default` 102. `non_checkpoint`: 5 loader-metadata reads
(`_repo_id, _scheduler_config, _text_encoder_configs, _vae_config` on `root.denoiser`;
`_vae_config` on `root.vae`). `document_roots`: encoders at
`["_text_encoder_configs", "text_encoder*"]`, others `[]`. 44 obligations, all
`unreceipted`. `config_audit`: `unread` [], `pending_projection`
`[_vae_config.act_fn, _vae_config.in_channels]`, `pending_classification` = **all 12
denoiser structural fields** (`attention_head_dim, caption_projection_dim, in_channels,
joint_attention_dim, num_attention_heads, num_layers, out_channels, patch_size,
pooled_projection_dim, pos_embed_max_size, qk_norm, sample_size`) + `_vae_config.sample_size`.
IR warnings: **none** (the incomplete projection is silent at IR level; the notes carry
only the U12/U13 handoff sentence). Render: `family=diffusion, layout=dit_pipeline,
opaque_layer_block{id: denoiser_structure_unresolved}`.

`root.text_encoder` (CLIP-L; standalone slot replay `sd35_encoder.py`, slot context
`component_namespace="root.text_encoder"`, bundle re-rooted to `modeling_clip.py`,
class overlay 1 field, decoderness None). Sub-parse → 12 layers, hidden 768. Slot
fact ledger, **16 rows**: code_proven — `attention.bias=True, output_projection=True,
projection_mode=split_qkv, scores_scale=sqrt(head_dim), ffn.gated=False,
ffn.projection_mode=dense, layer.norm_kind=layernorm, norm_placement=pre,
residual_topology=sequential`; code_and_config — `ffn.ffn_schedule=dense×12`;
**ambiguous** — `attention.mask=unknown, attention.mechanism=None, ffn.activation=None,
layer.parallel_norm_count=None, model.final_norm_kind=None, model.tie_word_embeddings=None`.
Slot reader failures: `ffn.intermediate_width` (unsupported_syntax "output-projection
input width is not evaluable"), `position.rope_schedule` and `mixer_schedule`
(incomplete_graph "comprehension index has no exact block formal"), `attention.mechanism`
(incomplete_graph — the MLA-shaped failure string), `attention.cached`, `final_norm_kind`
(incomplete_graph "local_or_free_name_call"). Ledger: 57 events; consumed
`hidden_size, is_encoder_decoder, layer_norm_eps, max_position_embeddings,
num_hidden_layers, vocab_size`; accessed-present adds `_name_or_path, architectures,
hidden_act, intermediate_size, model_type`.

Two consequences visible only from this trace: (i) the slot's 16 facts live on the
slot `ParseContext.facts` and are **not** merged into `ir.extras.fact_provenance`
(root has 1 row) — they reach the page only through `submodel_spec` inside
`text_encoder_specs` → loop blocks (`blocks.py:1037-1038, 1102-1103`), so the
projection-audit net cannot see encoder facts; (ii) `normalize_encoder_config` wraps
the sub-parse in `except Exception: return {}` (`encoder_panel.py:79-80`), so an
encoder parse crash degrades silently to a generic encoder box.

### 6.3 Qwen2-VL (summary)

Bundle: 3 components (`root`, `text_config`, `vision_config`) all → `modeling_qwen2_vl.py`
(+3 supporting each); index 12 nodes, 72 classes, 411 callables. 31 reader keys (most at
`text_config`); 21 facts (9 code_proven, 9 code_and_config, 1 config_declared, 2 ambiguous:
`final_norm_kind`, `parallel_norm_count`), owners include `root.vision`/`root.video`
projector widths. Notable failures: `position.rope_schedule@text_config`
(incomplete_graph "no layer positively proves the application" — Qwen's M-RoPE is not
proven, hence the IR warning "exact positional operation is unresolved"),
`ffn.schedule` ("FFN construction selector is not complete and exact"),
`model.final_norm_kind` ("field_is_not_a_constructed_child; local_or_free_name_call;
non_owner_call"), `root.wrapper_features` (unsupported_syntax). Config: accessed 28 /
consumed 14; `pending_projection` 14 paths (all `vision_config.*` and
`text_config.rope_parameters.*`), `pending_classification`
`text_config.is_encoder_decoder, vision_config.is_encoder_decoder`.

## 7. Failure trace — where SD3.5's denoiser stops producing structure

Walk (`sd35_failure.out.txt`, `sd35_reason.out.txt`):

1. `resolve_component_root(index, bundle, "root")` → **resolved**, declared
   `SD3Transformer2DModel`, `address_resolved=True` (`component_owner.py:1531`).
   The owner graph's root node has **0 resolved children and 7 `UnresolvedChild`
   rows** (`pos_embed, time_text_embed, context_embedder, transformer_blocks,
   norm_out, proj_out, …` — every child class is imported from another diffusers
   module) **(unverified: the seven field names, truncated in the dump)**.
2. `read_diffusion_root_topology` → **incomplete**, kind `repeated_stack`, one
   `RepeatedRootStage`, failure "positive local topology proof; whole-callable coverage
   is open" (`diffusion_root.py:406`). The loop over `self.transformer_blocks` in
   `forward` (`transformer_sd3.py:301`) is found locally.
3. `read_diffusion_stack_inventory` (`diffusion_stack.py:620`) → `visit(root)` →
   `resolve_container_inventory` → `resolve_addressed_invocations` (`execution_flow.py:305`).
   The `transformer_blocks` container has exactly one element site
   (`transformer_sd3.py:155-163`, the `[JointTransformerBlock(...) for i in range(num_layers)]`
   comprehension), but `_unique_element_template` (`execution_flow.py:656-665`) returns
   None because `site.candidates[0].symbol is None`: **`JointTransformerBlock` is not an
   indexed symbol** — it is defined in `diffusers/models/attention.py:580`, imported at
   `transformer_sd3.py:23`, and that file is not in the bundle (`component_files["root"]`
   is `transformer_sd3.py` only; "classes named JointTransformerBlock in index: []").
   → `UnresolvedInvocation(..., "heterogeneous_or_unresolved_container_elements")`
   (`execution_flow.py:441-443`) → `_unresolved_candidate(owner, "transformer_blocks",
   "heterogeneous_or_unresolved_container_elements", …)` (`diffusion_stack.py:724-725`).
   Result: **0 stacks, 1 unresolved**, failures "U3 supplies positive local execution
   relations, not whole-forward coverage" + "1 container/traversal candidates remain
   unresolved".
4. `read_diffusion_block_facts` (`_source_only_diffusion_stack_and_blocks`,
   `diffusor/parser.py:339`) has no stack rows → adds "1 U10-B stack candidates stay
   opaque". `_attention_lane` (`diffusion_block.py:276-322`) is **never reached** for this
   model; its `else` branch at `:301-311` ("framework attention-container proof does not
   establish internal projection, Q/K, geometry, scaling, or position facts"), which
   fails every lane whose child is a `FrameworkAttentionLaneEvidence` (protocols
   `framework_container | indexed_framework_container | source_mixin_delegate`,
   `attention_lane.py:171-174`), is the **second** gate and would apply only after the
   block class were indexed.
5. streams / conditioning / bookends each return `incomplete` with their local-proof
   failure strings (`"local stream proofs do not establish whole-forward completeness"`,
   `"positive conditioning applications do not prove whole-block absence"`,
   `"positive local bookend routes do not prove whole-root completeness"`).
6. `bind_diffusion_source_projection` → `BoundDiffusionSourceProjection` with **0
   operands**, status incomplete, 10 failures (the 7 above plus "U10-F1 is an open-world
   source projection; config operands are unbound" and "U10-F2 binds exact operands but
   production consumption waits for F3"). `project_diffusion_ir` → 0 layers, 0 templates,
   `unresolved=()`.
7. `_parse_projected_denoiser` takes the `has_value` branch (the object has a value even
   though it is empty), appends no warning (`projection.unresolved` is empty), and the IR
   ships with `layers=[]`, `hidden_size=0`, the opaque render, and **no IR warning** —
   the user-facing signal is only the "Denoiser internals unresolved" block.

Verdict on the two known causes: **cause 1 (block class outside the bundle) is
confirmed and is the sole active cause** — `build_program_index` is only ever called
with `external_nodes=()` (`context.py:227`); `bundle.import_roots` is consumed by
`import_source.py:245, 389` to *prove* import references, never to *expand* the index,
and the "demand expansion" mentioned at `attention_lane.py:507-510` has no producer in
the tree (grep: no caller passes `external_nodes`). **Cause 2 (`diffusion_block.py:299-310`
refusing framework/mixin lanes) is real but latent** — it cannot fire until cause 1 is
fixed, and when it does, an SD3 `JointTransformerBlock` (`Attention(... processor=
JointAttnProcessor2_0())`, `attention.py:626-632`) would land on the `source_mixin_delegate`
/ `framework_container` protocol and be refused at `:301-311`.

## 8. Loose ends

- The cProfile rows for `prepare_document` / `AutoConfig.for_model` show `prim=0,
  cum=0.00` in all three cold profiles; the hydration cost is established by wrappers
  and an isolated timing, not by cProfile. Cause of the artifact not investigated.
- Qwen2-VL warm2 (9.6 s) vs warm (5.6 s): single observation, cause unknown.
- The seven `UnresolvedChild.field` names for SD3's root were truncated in the dump.
- `resolve_source_files` for SD3.5 cold (2.87 s) includes `import diffusers` (2.36 s)
  inside `_installed_diffusers_model_class_file`; not itemized further.
- `hash` (226k–982k calls per parse) is the top tottime primitive warm; which frozen
  dataclass dominates was not measured.
- Fact provenance for encoder sub-parses (16 rows for CLIP-L) never reaches
  `ir.extras.fact_provenance`; whether the projection-audit net has a separate channel
  for `submodel_spec` facts was not checked here (belongs to 05-verification-system).
- `_ensure_parsable`'s U10-F3 exemption plus the silent `has_value` branch means a
  diffusion parse can return a structure-less IR with zero warnings; whether Sable's
  nets catch this for SD3.5 (it is a blessed corpus witness) is for 05.
- Multimodal subject was profiled and fact-dumped but not call-chain-traced line by
  line beyond the `4278-4300` tower branch.
- Not measured: `unfold("org/model")` by id (hub fetch + `_load_diffusion_config` path,
  `parser.py:521-547`), `inspect_code=True` (`_attach_code_evidence`), `to_png`/`save_images`.
