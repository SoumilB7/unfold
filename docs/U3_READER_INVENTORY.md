# U3 Reader Inventory — every code-evidence reader as of `audio-composite-support` @ 1d0c72b

All paths relative to `/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/model_unfolder/`. Line numbers verified against the working tree. Block format per reader: **(1)** inputs, **(2)** extracts, **(3)** target selection, **(4)** failure shape, **(5)** duplicated low-level extraction, **(6)** callers + owner info the caller already has. 37 `ast.parse` call sites exist under `model_unfolder/`; 34 read MODEL source (in scope), 3 parse the package's own source (self-audit guards, out of ProgramIndex scope).

---

## evidence/ast_scanner.py (106 lines)

**`scan_python_files(files)` :17** — (1) raw path iterable (callers pass `bundle.files` flat). (2) per-class `ClassEvidence`: self-assigned field names + lines, call-name Counter + first line, `config.X` attribute reads (`config_refs`). (3) none — every ClassDef in every file. (4) unparseable file silently skipped (bare `continue` :24-25); no per-file failure record. (5) its own self-assign walk (`_collect_self_*` :68-81) — twin of `forward_ops._field_types`. (6) callers: `inspector.inspect_model_code` :27; `patterns.decoder_attention_sinks_from_files` :2392; `conformance.check_fact_conformance` MLA cross-check :492-497; diffusor parser (`diffusion_norm_from_classes` input). Callers hold the full bundle incl. `component_files` but pass flat files.

**`_call_name(node)` :83** — the shared call-name normalizer: factory classmethods (`X._from_config` → `X`, vocabulary `load_constructor_classmethods()`) and `CLASSES[key](...)` subscript → registry base name. Imported by forward_ops, transitive, position, vision, audio, projector, fusion, conformance, patterns, identity_guard. This is already ONE shared rail — keep as index primitive.

---

## evidence/forward_ops.py (629 lines)

**`extract_forward_ops(files, *, component="root")` :34** — (1) raw paths; `component` is a caller-supplied stamp only. (2) per-class `ForwardOps`: op-kind presence set with positive-config-gate context (`_forward_op_occurrences` :190, `_positive_gate_fields` :255), ctor field→class map (`_field_types` :433 — handles Call, IfExp branch pick, `ACT2FN[...]` Subscript, `self.f = <local ctor var>` hop), branch-aware `module_list_elems` :487 (default-branch pick for gated construction), `signature_tokens` (all call names + fields), `forward_params` :308, `init_class_refs` :292 (every class constructed in `__init__` incl. nested kwargs), `gated_op_kinds`, init-local fn folding (`_init_local_fn_bindings` :151), container classes (init-only, helper-method folding `_init_helper_methods` :179), BLOOM residual-threading rule :207-214, fused-projection matmul rule (`_binop_op_kind` :354, `_projection_operand` :379 — "proj"/"weight" NAME substring on operands). (3) none — all classes; consumers pick. Role typing via `_role_of` :396 = case-insensitive class-name-substring table (`type_roles` YAML) — the central name-nominates mechanism. (4) parse failure → `{}` from `_parse_file` :60-64 (bare); class without forward+init → dropped silently. (5) own parse cache `_parse_file` (lru 256, keyed path+mtime :52-70) — cache #1 of 4. (6) callers: conformance (all four check_* nets), patterns (≥8 readers re-call it per invocation), diffusor parser (`unet_mid_block_present`, `denoiser_block_timestep_conditioning`, rope/attn/ffn-kind readers), ffn.py. Callers already have `component_files[owner]` + `component_architectures`.

**`unclassified_call_tokens(files)` :551** — loud-miss net. (1) raw paths, (2) bare call tokens not in any vocabulary, (3) all forward()s, (4) `{}`; **own uncached `ast.parse` :600** (parse #2 of the same files during a sable run). (6) caller: `tests/test_loud_miss.py` + sable.

Shared helpers exported to 8+ modules: `_method` :419, `_self_field` :426, `_field_types` :433, `_module_list_elems` :487, `_init_class_refs` :292, `_role_of` :396, `_call_op_kind` :325, `_binop_op_kind` :354, `_forward_params` :308.

---

## evidence/transitive.py (611 lines)

**`build_registry(files, *, component="root")` :77** — (1) raw paths. (2) per-callable `CallableInfo`: direct op_kinds + call_tokens (`_scan_body` :464), folded-init `field_types` (`_folded_init_bodies` :352 — init + transitively-called self-methods), `field_type_candidates`/`field_type_dispatch` from module-level literal dict registries (`_module_class_maps` :376, `_field_class_map_candidates` :400), `self_field_calls`, `iter_field_calls` (`_iter_self_field` :502), `sub_module_classes` (`_init_sub_modules` :528 — literal/comprehension/append/`add_module`/local-list-wrap), `init_class_refs` + diffusers class-attr processor refs (`_class_attr_processor_refs` :426 — `_default_processor_cls`/`_available_processors`), `var_fn_bindings` (`_bound_free_fn` :511 — `REGISTRY.get(key, default_fn)` dispatch), free functions, forward-helper method folding (:305-331). (3) none — all classes + top-level functions. (4) parse failure → `{}` (`_parse_file` :212-217, bare). (5) parse cache #2 (lru 256, path+mtime); its own ctor walk PARALLEL to forward_ops' (both parse the same file into different caches). (6) callers: conformance (`check_nested_conformance` :758/:779/:845, position, vision, audio, projector, fusion, ffn, stacks, patterns (`unet_*` BFS readers), diffusor parser directly.

**`resolve_architecture_anchor(registry, declared)` :89** — structural root recovery when the declared class was renamed upstream: unique construction root else keep declared. (4) may return None/stale name silently. (6) projector :24, fusion :20, conformance `_reachable_forward_ops` :1562.

**`transitive_closure(start, registry, vocab, *, extra_class_refs, max_nodes=400)` :115** — BFS to `(op_kinds, signature_tokens)`; terminals: attention-compute tokens → curated pair, rope/cache markers → semantic-only, declared library helpers; follows field classes, dispatch (`eager` preferred :151), iterated ModuleLists, var-fn bindings, injected processor classes; `attn.to_q` cross-class token resolution via ROOT's field_types :176-178. (4) missing start → empty frozensets (bare). (6) conformance drills/composites, position `_rope_mechanism`/`_relative_bias_mechanism`, vision `layer_facts_from_block`, ffn (indirect).

---

## evidence/inspector.py (28 lines)

**`inspect_model_code(target, *, source, token)` :13** — resolve bundle → `scan_python_files(bundle.files)` FLAT → `patterns.infer_code_evidence`. (3) no component scoping at all (blends every tower's classes into one findings list). (4) typed `CodeEvidence` with warnings. (6) callers: `parser.py:388-390` (optional `inspect_code=True` attach path → `validate_ir_with_evidence`), package `__init__` public API.

---

## evidence/patterns.py (3,567 lines; 20 `ast.parse` sites)

Tier A — ClassEvidence-consuming chip detectors (no own parse): **`infer_code_evidence` :41** + 13 `_detect_*` (:115-351). Class gating by NAME substring: `_is_attention_class` :112, `_has_dense_ffn` :367 (requires "mlp" in class name), `_has_moe` :380 (name OR fields), `_interesting_classes` :399 (name filter, cap 48). Feed `CodeFinding` chips + `validate.py`. Failure: empty findings.

Tier B — `*_from_files` readers (each takes raw file paths; most re-parse). One line each — (2)(3)(4) inline, ⚑ = same-role-union/best-candidate site, ● = bare failure:

- **`diffusion_norm_from_classes` :434** — from ClassEvidence calls; picks `max(used, key=count)` ⚑ vote; ● None.
- **`_parse_defs` :543** — parse cache #3 (lru 128, tuple(files), NO mtime key — stale-cache hazard vs caches 1/2).
- **`_shared_ffn_defs` :523** — parses an EXTERNAL library module (`diffusers.models.attention` located via import) — the index must model out-of-bundle library files; broad `except Exception` :538 (one of the 2 pinned patterns.py/sources.py broad excepts).
- **`diffusion_ffn_activation_from_files` :556** — blocks = class name `.endswith("block")` ⚑; most-common vote over kwarg/structural hits ⚑; inline `_standalone_act_fns` :614 fallback; ● None.
- **`diffusion_axes_dims_rope_from_files` :641** — any FunctionDef with arg default `axes_dims_rope`/`axes_dim`, first hit ⚑; ● None.
- **`diffusion_rope_from_files` :684** — ANY ForwardOps in files matching rotary markers ⚑ (whole-file union); ● False.
- **`diffusion_attn_kind_from_files` :699 / `diffusion_ffn_kind_from_files` :715** — union of ALL classes' init_class_refs across all files ⚑; ● None.
- **`diffusion_qk_norm_from_files` :742** — whole-file walk, 4 spellings, `Counter.most_common` ⚑; ● None.
- **`diffusion_cross_qk_norm_from_files` :804** — cross site found via wiring_roles text params; per-site, unanimous-else-None; ● None.
- **`unet_transformer_ffn_activation_from_files` :879** — anchored BFS from declared block-type strings (identity-as-address, lawful), unanimous; ● None.
- **`unet_stage_temporal_from_files` :949** — BFS + marker substrings "conv3d"/"alphablender" (hardcoded, not YAML); ● None (positive-witness only).
- **`unet_stage_attn_cell_from_files` :986** — BFS role scan; ● None.
- **`unet_code_attention_placement_from_files` :1038** — architecture-addressed `_class_node`; ● None.
- **`unet_mid_block_present_from_files` :1078** — "mid"/"down"/"up"/"bottleneck" NAME substrings on fields/calls; ● None.
- **`diffusion_single_stream_fusion_from_files` :1143** — "single" in field name selects the elem class; ● None.
- **`diffusion_gate_via_norm_from_files` :1192** — class NAME must contain "Modulated"+("Norm"|"RMS") ⚑ name-based; ● False.
- **`decoder_layer_topology_from_files` :1244** — via `_find_decoder_layer`; `_linearize_forward` :3288 (post-order role stream) + `_classify_topology` :3343; ● None.
- **`_find_decoder_layer` :1290** — THE shared anchor: structural roles (attention+ffn / attention+norm) + KV-cache forward-param tiebreak (`_DECODER_CACHE_PARAMS` :1285), else **first candidate** `(caching or candidates)[0]` :1326 ⚑ broad best-candidate. Used by 10 readers.
- **`layer_class_count_from_files` :1330** — count of layer-shaped classes; ● 0.
- **`decoder_norm_kind_from_files` :1354** — norm class-name substring rms/layernorm; ● None.
- **`norm_kind_from_files_math` :1440** + **`_norm_math_verdict` :1378** — math classification (mean-sub vs pow/rsqrt), base-class recursion depth 3, torch-primitive names as leaves; unanimous-else-None; ● None.
- **`decoder_ffn_gated_from_files` :1485** — MLP forward + ctor shape; MoE one-hop member recursion; inline fc1/fc2 branch :1566-1592; ● None.
- **`decoder_ffn_activation_from_files` :1595** — activation class-name substring; ● None.
- **`ffn_activation_dispatch_field_from_files` :1631** — `ACT2FN[config.X]`/`get_activation(config.X)` → field name (code_and_config tier); ● None.
- **`denoiser_temporal_axis_from_files` :1704** — architecture-addressed forward name scan vs `temporal_forward_markers`; ● None.
- **`attention_fused_qkv_from_files` :1737** — ALL attention-role classes across all files ⚑ (unanimous-else-None); fused-name set `_QKV_PROJ_NAMES` :2563; ● None.
- **QK-norm rail :1769-2001** — `_attention_qk_norm` :1860: guard-tracked ctor walk + forward taint dataflow (latent-norm exclusion, ≥2 live sites), gate atoms = config fields the code reads (typed `QKNormCodeEvidence`/`QKNormGateAtom` — the U3 model answer-shape); `_QK_UNRESOLVED` sentinel → None (honest).
- **`_ConfigExprEvaluator` :2004** — evaluates `__init__` arithmetic against cfg; **reads cfg values directly** (`cfg_get` :2016) outside the config-access funnel.
- **`decoder_rope_dim_from_files` :2081** — every class w/ "rotary" call substring across `_parse_defs` ⚑; single-value-else-None; cfg values via evaluator.
- **`decoder_router_evidence_from_files` :2216** — union of route-role classes + `_has_moe`-shaped classes ⚑, minus framework-container NAME SUFFIXES `_CONTAINER_SUFFIXES` :2144; free-fn one-hop; typed `RouterCodeEvidence`; ● None.
- **`decoder_intermediate_size_from_files` :2337** — layer ternary → FFN-class ternary → widened-Linear; refuses rival FFN fields :2372-2373; cfg values via evaluator; ● None.
- **`decoder_attention_sinks_from_files` :2383** — `scan_python_files` again, ANY class ⚑; ● False.
- **`decoder_qk_norm_from_files` :2398** — decoder-layer-anchored, unanimous-else-None; ● None.
- **`_class_builds_experts` :2442** — "expert" substring anywhere in class body incl. string constants; ● None (unknown class).
- **`_MoEGateEvaluator` :2469** — per-layer gate eval; direct cfg reads.
- **`decoder_parallel_norm_count_from_files` :2597** — dataflow (norm-var → attn/ffn arg); >2 norm fields → None (Falcon); ● None.
- **`decoder_attention_bias_from_files` :2662 / `decoder_mlp_bias_from_files` :2707** — Linear ctor `bias=` (literal / `config.X` resolved via `_linear_bias_value` :2570 — direct cfg read); unanimous-else-None; ● None.
- **`lm_head_tying_from_files` :2757** — top-level `x.weight = y.weight` w/ field-name marker vocab `_HEAD_FIELD_MARKERS`/`_EMBED_FIELD_MARKERS` :2753-2754 (in-code, not YAML); ● None.
- **`decoder_moe_schedule_from_files` :2818** — layer-init gate walk + ternary; per-layer bool list; any doubt → None.
- **`embedding_stage_norm_from_files` :2953** — every class, FIRST match ⚑; order-aware embed-var lineage; ● None.
- **`expert_fused_gate_up_from_files` :3030** — any class w/ `*gate_up*` field referenced in forward, first hit ⚑; ● None.
- **`attention_score_scaling_from_files` :3071** — role-wide union over all files (deliberate: T5 ModuleList sublayers) ⚑; unanimous-else-None; ● None.
- **`attention_causality_from_files` :3132** — EVERY class, EVERY method, EVERY file ⚑ global union; decoderness gates statically evaluated **against direct cfg reads** (`_flag` :3167 w/ `_FLAG_DEFAULTS` is_decoder/add_cross_attention :3165); cross-input pruning; `is_causal=True` literal; ● None.
- **`decoder_cross_attention_all_layers_from_files` :3386** — all classes, first match, cross-field vocab from composite_slots; ● None.
- **`decoder_codebook_streams_from_files` :3442** — ModuleList-of-Embedding/Linear + forward sum/stack proof; tri-state dict.
- **`denoiser_block_timestep_conditioning_from_files` :3526** — architecture-addressed, block = attention-role ctor, unanimous; ● None.

(6) callers: transformer parser (via `_source_files` = **flat `context.source_bundle.files`**, adapters/transformer/parser.py:119-129 — ~20 call sites :143-395 rely on `_find_decoder_layer` to re-derive text ownership the caller already holds in `component_files`); diffusor parser (via its `_source_files` = `component_files["root"]`, adapters/diffusor/parser.py:103-118 — proper owner scoping); conformance `_check_storage_facts` :655-663, `_check_component_storage_facts` :614.

---

## evidence/stacks.py (280 lines)

**`secondary_stacks_from_files(files, architecture)` :64** — (1) raw paths + architecture string. (2) block-shaped ModuleList stacks NOT the root stack: owner/field/block class, `count_field` traced hop-by-hop through the ctor chain (`_resolve_count` :157, `_range_expr` :187, `_constructor_arg` :212), `lane_param` (`_lane_param` :243), `entry_projection`, ctor kwargs (`_block_ctor_kwargs` :121). (3) BFS from architecture (address); block-shaped = constructs attention/ffn-role field; when several block elems, `sorted(blocks)[0]` :107 ⚑. (4) ● `[]` (arch missing / nothing found — indistinguishable). (5) uses `build_registry` + `vision._class_node` re-parse (cache #4). (6) callers: diffusor parser (refiner stacks), tests/test_stacks.py; caller passes root component files.

---

## evidence/ffn.py (123 lines)

**`ffn_structure_evidence(files, *, expected_gated, component, architecture)` :17** — (1) raw paths + optional architecture + config-expected gating. (2) FFN storage: dense/split/fused_gate_up, gated, owner class/line. (3) whole-registry candidate scan by op signature (activation present, no attention/route, ≥2 called linear fields, QKV-lane exclusion :49-55); `_reachable_classes` :96 anchor only when architecture is given, else whole files ⚑; multiple candidates accepted only when profile-identical, then **shortest name** :87 pick. (4) TYPED `FFNStructureEvidence` proven/ambiguous/oracle_missing — the model failure shape U3 wants everywhere. (5) builds BOTH `extract_forward_ops` AND `build_registry` over the same files. (6) callers: transformer parser :196, conformance `_storage_problems_for_spec` :572-598.

---

## evidence/position.py (466 lines)

**`decoder_positional_evidence(target, *, source, bundle)` :32** — (1) config target + bundle; files via `conformance._component_source(bundle, "text")` :54. (2) typed `PositionalEvidence`/`PositionalMechanism` (kind + application altitude + symbols + line): rope (closure token markers, `_rope_mechanism` :250), relative bias :265, alibi (`_ALIBI_CALL_MARKERS` :27 — in-code vocab), learned/fixed absolute w/ add-dataflow proof (`_absolute_embedding_mechanism` :306; `_POSITION_FIELD_MARKERS` :22 in-code; fixed-vs-learned partly by class NAME "sinus"/"fixed" `_position_field_is_fixed` :415 ⚑ name-based). (3) blocks via conformance helpers (architecture-anchored + domain name-markers), attention candidates via role classes → dispatch-config selection (`_configured_attention_classes` :220) → `_outermost_candidates` :199 (nested-kernel drop — existing rival-owner logic); model-stage classes :170. (4) typed proven/ambiguous/oracle_missing; internal helpers bare-None. (5) **re-parses the source file per class for line numbers** — `_class_forward` :372, `_call_line` :382 (uncached `ast.parse` ×2). **(8) raw config walks:** `_config_value` :444-450 / `_config_scopes` :453-466 (hardcoded wrapper list `text_config/language_config/llm_config/text_model_config/thinker_config`, 4 deep) — keys read: `alibi` :84 (Falcon arbitration: config-True demands a proven alibi path), `layer_types` :117 (hybrid linear+full ⇒ proven rope+none mix), `_attn_implementation` :231 (dispatch selection, default "eager"), `rotary_dim`/`rotary_pct`/`partial_rotary_factor` :434-441 (`_effective_rotary_is_zero` ⇒ proven "none"). None of these route through the U1 ConfigAccessLedger — unlogged config consumption inside an evidence reader. (6) callers: transformer parser :447-449 (result → position mechanisms + rope flag + `to_dict` into IR), conformance `check_fact_conformance` :509-532 (symmetric net — same function, so parser and net cannot diverge).

---

## evidence/sources.py (676 lines) — (7) source-resolution flow

`ParseContext.build(target)` (context.py:192-206) → `resolve_source_files` :14 → ladder: `_path_bundle` :89 → `_installed_transformers_bundle` :106 (walk `_component_configs` :257 — `*_config` suffix unconditional + composite-slot names gated on child `model_type`; per component: `model_type_to_module_name` registry dir :333, `modeling*.py` glob, arch from `architectures[0]` / `MODEL_MAPPING_NAMES` :315 / `config_class` declaration `_architecture_from_config_class` :526 **[ast.parse :547 — reads ownership declaration + constructed-set to pick the live owner; 2 live owners → None honest]**) → `_installed_diffusers_bundle` :415 (`_class_name` regex class-def scan over all diffusers models/*.py; pipeline text-encoder slots recursively qualified through the SAME transformers resolver, prefixed `text_encoder.…` :456-468) → `_installed_diffusers_arch_bundle` :394 → remote-code hub rail gated on config-declared `auto_map` (`_declares_remote_code` :509) → `_hub_bundle` :578 (cache-first snapshot). Outputs `SourceBundle` (models.py:400-424): flat `files` + `component_files` (qualified dotted paths) + `component_model_types` + `component_architectures` + `pipeline_components` + warnings.

**Identity as ADDRESS (lawful):** model_type→installed dir; `architectures[0]`→class-def regex (`_transformers_file_for_class` :242, unique-match-only); `_class_name`→diffusers scan (`_diffusers_class_file` :370); `auto_map`→hub; `AutoConfig.for_model` class defaults (`_installed_config_defaults` context.py:210, `@identity_address`-decorated); `declared_decoderness` (decoderness.py — config declarations only, no source read).
**Name COMPLETES resolution (U3 step-2 flags):** `_looks_like_diffusion_class` :354-363 — `dit_class_markers` + "UNet"/"Transformer" substring gates whether the diffusers bundle is attempted AT ALL (a denoiser class without a marker never gets its installed source); `_model_type` :625 silently substitutes `text_config.model_type` for the root's (owner blur at the address tier).
**`slot_parse_context` (context.py:234-300)** — derives a slot's ParseContext from the parent's ALREADY-resolved bundle: re-roots `component_files`/`types`/`architectures` under the slot (`text_encoder.text_config` → `text_config`), rebuilds flat files from the subtree, re-derives class_defaults/decoderness from the bundle's own per-component records. Correct one-resolution law; the re-rooted bundle is exactly the shape a per-owner ProgramIndex view must serve. Ambient `active_parse_context` ContextVar :308.

(4) failure shape: warnings tuples on the bundle (good), but component gaps are warnings-only; `_transformers_file_for_class` returns None on 0 or 2+ matches indistinguishably.

---

## evidence/vision.py (1,085 lines)

**`vision_tower_evidence(target, *, source, bundle)` :27** — (1) bundle (or re-resolve); files = `_component_source(bundle,"vision")`. (2) typed `VisionTowerEvidence`: patch pipeline as ordered `SourceOp`s (`_patch_ops` :450, `_owner_patch_flow_ops` :482 — derived-value dataflow with hardcoded boundary field markers :519-522, `_ordered_patch_callable` :577, `.T` attribute transpose :651-664, plumbing-run collapse `_collapse_plumbing_runs` :408), model-stage position kind :725, per-block `VisionLayerEvidence` variants incl. per-instance gating (`_configured_block_instances` :1028 — `is_gated` kwarg/positional). (3) blocks: `_component_block_classes` + `_domain_block_classes` (name-marker domain filter ⚑) + op-kind filter :71-72; root fallback via `_assigned_component_class` :124 (constructor consumes `config.vision_config` — dataflow, single-candidate-else-""); owner `_vision_owner` :343 — architecture else **`max(candidates, key=len(field_types))`** :372 ⚑ best-candidate. (4) typed proven/ambiguous/oracle_missing; internal helpers bare. (5) parse cache #4 `_parsed_classes` :1072 (lru 128, NO mtime); `_calls_in_execution_order` :989 copy 1/3; `_dedupe_ops` :980 copy 1/3.

**`layer_facts_from_block(block, registry, vocab, ctor_kwargs)` :189** — THE shared per-block typed-fact reader (audio + diffusor secondary stacks ride it): norm kind (excl. tuple-unpacked gate producers `_tuple_unpacked_self_calls` :323), dataflow norm placement :743, closure-proven FFN gating, gate liveness pruning by construction-site falsy literals (`_instance_gate_dead` :272), gate source/activation, lane norms construction-site-aware (`_lane_norm_fact` :916, `_lane_norm_gate_param` :820, `_ctor_kwarg_truthy` :860), projection modes (field-name heuristics :772-798), rope marker, attention kind = "linearatt" substring :789 ⚑, `standard_cell`. Returns plain dict (● un-typed booleans; `ffn_gated` may be None). (6) callers: this file, audio.py :99, diffusor parser (refiners), conformance `_check_vision_facts` field-by-field diff :156-203.

---

## evidence/audio.py (624 lines)

**`audio_tower_evidence(target, *, source, bundle)` :31** — (1) bundle; files = `_component_source(bundle,"audio")`. (2) typed `AudioTowerEvidence`: front-end ordered ops + position kind/application (`_owner_frontend_ops` :173, buffer-add detection `_position_add_line` :585 regex `position|pos_embed`), post ops, output projection takeover `_take_output_projection` :452 ("output"/"proj" field substrings), per-block `AudioLayerEvidence` with **SSA-like `_flow_ops` graph** :275 (env var→op-id edges, `inputs` tuples — the richest dataflow extraction in the codebase), reachable callables :425, `repeat_field` from comprehension `config.X` :466, `layer_facts` via vision's shared reader. (3) `_is_audio_cell` :112 (structural); owner `_tower_owner` :119 — architecture else **max(field count)** :146 ⚑. (4) typed proven/ambiguous/oracle_missing. (5) own UNCACHED `_class_node` :615 (ast.parse per call — every `_flow_ops`/`_split_repeated_region` call re-parses the file); `_calls_in_execution_order` copy 2/3; `_dedupe` copy 2/3; label maps `_label_for`/`_shape_label` partially duplicate vision/projector label logic. (6) callers: transformer parser :1789 (modalities/audio.py builder consumes the result), conformance `_check_audio_facts` :262.

---

## evidence/projector.py (831 lines)

**`projector_evidence(target, *, source, bundle)` :18** — (1) bundle; **uses `bundle.files` FLAT** :23 (not component_files) + `resolve_architecture_anchor`. (2) typed `ProjectorEvidence`: ordered connector ops (`_callable_ops` :82 — norm/linear/conv-flavor-from-`groups=` :556/embedding/attention/Sequential unroll :746/perceiver loop :134-151/gated-MLP exact-graph recognizer `_gated_mlp_ops` :617), learned queries (`_has_reachable_parameter` :686), kind derivation :708, **COR-4 width bindings** :213-482: `_config_param_chains` :319 walks construction sites root→child propagating exact config dotted prefixes (conflict ⇒ drop, never guess) — **this is the prototype of U3's ComponentOwner/config-prefix resolution**; `_bind_expr` :272 → config_bound/code_bound/derived/unavailable. (3) `_reachable_projectors` :61 — field-name markers (`projector|merger|connector|resampler`; `*_projection` gated on modality word or wrapper proof `_is_projection_wrapper` :506); pick = `sorted(..., key=(depth, field_rank, name))[0]` :33-35 ⚑ best-candidate. (4) typed proven/ambiguous/oracle_missing. (5) own UNCACHED `_class_node` :822; `_calls_in_order` copy 3/3; `_dedupe` copy 3/3; `_factory_fields` :730 (`_from_*` prefix) duplicates ast_scanner's factory vocabulary rail with a looser rule. **Direct cfg reads outside funnel:** `_activation_value` :761 (`projector_hidden_act|hidden_act|hidden_activation`), `_resampler_depth` :773. (6) callers: transformer parser :1821, audio.py :81 (audio connector fallback), conformance `_check_projector_facts` :206; modalities/vision.py `apply_projector_evidence` :102 consumes the RESULT (widths through the evented accessor).

---

## evidence/fusion.py (219 lines)

**`fusion_evidence(target, *, source, bundle)` :14** — (1) `bundle.files` FLAT + anchor. (2) typed `FusionEvidence`: masked_scatter routes, cross-attention route (param + ownership proof :137), encoder_hidden_states seq2seq route :155, prefix-concat routes; grid positions. (3) `_reachable(root)` wrappers by depth; equal-depth signature disagreement ⇒ ambiguous :31-36 (rival-owner handling already present). Modality classification `_modality` :176 = substring of `ast.unparse(call)` TEXT ("video"/"audio"/"image|vision|pixel") ⚑ string-level; `"vision_model" in source_text` :143, `"image_grid_thw" in source_text` :78 — unparse-text matching the index should replace with real symbol queries. (4) typed proven/ambiguous/oracle_missing. (5) own UNCACHED `_class_node` :210; own `_method` :202 (copy of forward_ops'). (6) callers: transformer parser :1809, conformance `_check_fusion_facts` :321; modalities/fusion.py `apply_fusion_evidence` :71 consumes the result.

---

## evidence/conformance.py (1,851 lines) — chief consumer + own readers

Entry points (all take `target` + `ir`/`render_log` + bundle): **`check_model_conformance` :118** (per layer-group op-set diff; `_component_source(bundle,"text")` → `_augment_diffusion_files` → `extract_forward_ops`); **`check_wiring_conformance` :383** (forward_params vs drawn rails; direct cfg read `add_cross_attention` :433); **`check_fact_conformance` :440** (storage/bookend/MLA/position/NoPE/linear-attn; calls scan_python_files, patterns storage readers, position, then the four tower checks :559-562); **`check_nested_conformance` :713** (per-drill transitive closure; **render-log events carry exact `component`/`source_owner`/`source_file` ownership** :765-805 and event-bound registries :845-852 — the exact-owner signal U3 should generalize).

Own extraction/selection machinery: `_block_classes` :872 (structural); `_component_block_classes` :881 — BFS from architecture, **fallback = ALL block classes in files** :892/:917 ⚑; `_init_helper_block_classes` :920 (**ast.parse :929** uncached); `_resolve_drill_closure` :955 (replaced same-role union with owner-field resolution + profile-equality; `_view_class_candidates` :1017 view↔class NAME markers ⚑ with `or candidates` fallback); `_inline_ffn_closure` :980 (excludes non-text by NAME markers ⚑); `_resolve_selection_closure` :1032; `_direct_role_classes` :1058 (role field markers, `matched or fallback` union ⚑); `_domain_block_classes` :1091 (component_class_markers NAME filter, `selected or blocks` ⚑); `_expert_classes` :1105 ("expert"/"shared" field substrings); `_unique_candidate_closure` :1127 (identical-closure-else-None — the rival-owner pattern); `_constructor_envs` :1171 (**ast.parse :1179**), `_selected_init_refs` :1214 (**ast.parse :1219**), `_eval_static_condition` :1252; `resolve_view_code` :1485 — tier-1 own ModuleLists (exact), tier-2 anchored, override map, tier-3 NAME suffix (`DecoderLayer|TransformerBlock`) + single-stream markers + **shortest-name pick** :1554 ⚑; `_reachable_forward_ops` :1557; `_is_block_class` :1634 (structural where exact, suffix elsewhere); `_component_source` :1658 (domain → dotted-path segment match, `_TEXT_WRAPPERS` :1654 duplicated vocab, deepest-path pick; pipeline-slot exclusion :1676-1681); dormancy: `_op_is_dormant` :1702 / `_branch_inactive` :1717 / **raw config walks `_config_field_value` :1727, `_config_scopes` :1736, `_as_mapping` :1748** (second copy of the wrapper-walk, also outside the ledger); `_augment_diffusion_files` :1775 + `_imported_model_files` :1802 (**ast.parse :1818**; import-closure = the index's module/alias layer, currently bespoke). (6) caller: sable.py :371-474 (all four nets, blocking); diffusor parser imports `_augment_diffusion_files` directly.

---

## Package-source parsers (NOT model-source readers — exclude from ProgramIndex, keep)

- identity_guard.py `scan_identity_source` :269 (**ast.parse :272**) — scans model_unfolder's OWN code for identity debt.
- structural_writes.py `_scan_raw` :266 (**ast.parse :272**) — package structural-write census.
- structural_debt.py `_module_symbols` :869 (**ast.parse :874**) — package symbol existence.
- preview.py — regex over rendered HTML only (:98, :211). Not a source reader.
- adapters/transformer/special_parts/modalities/* — pure consumers of evidence results + evented `config_access`; no AST use (verified: only vision.py:226-228 records typed facts, accessors/detect/builder import config_access).

---

## Duplication clusters (name each; U3 deletes after parity)

1. **Four parallel parse caches + ≥10 uncached parse sites over the same files**: forward_ops `_parse_file` :59 (mtime), transitive `_parse_file` :212 (mtime), patterns `_parse_defs` :542 (NO mtime), vision `_parsed_classes` :1072 (NO mtime); uncached: audio `_class_node` :615, fusion `_class_node` :210, projector `_class_node` :822, position `_class_forward` :372 + `_call_line` :382, conformance :929/:1179/:1219/:1818, forward_ops `unclassified_call_tokens` :600, sources `_architecture_from_config_class` :547, patterns ~16 inline per-reader parse loops.
2. **Ctor-assign / self-field walking**: forward_ops `_field_types`; transitive folded-init; ast_scanner `_collect_self_*`; patterns (`_class_ffn_shape` :505, `_attention_qk_norm.walk_init` :1884, `decoder_moe_schedule.walk` :2862, `decoder_cross_attention._self_fields` :3419, `expert_fused_gate_up` :3056, `lm_head_tying` :2787…); vision `_instance_gate_dead` :293; projector `_self_assigns` :439 + `_factory_fields` :730.
3. **ModuleList element extraction ×3**: forward_ops `_module_list_elems` (branch-aware), transitive `_init_sub_modules` (append-aware), patterns `_modulelist_of` :3460; conformance `_init_helper_block_classes` re-wraps #1.
4. **Construction-graph BFS reachability (~14 copies)**: ffn `_reachable_classes` :96; position `_model_stage_classes` :170; vision `_vision_owner.reaches` :350 / `_class_reaches` :1006 / `_assigned_component_class` :140; audio `_tower_owner.reaches` :126 / `_reachable_callable_evidence` :430; projector `_reachable_projectors` :61 / `_reaches_role` :668 / `_has_reachable_parameter` :686 / `_config_param_chains` :319; fusion `_reachable` :40; conformance `_component_block_classes` :894 / `_reachable_forward_ops` :1557; patterns unet BFS ×3 :904/:966/:1006; stacks paths walk :72; transitive `resolve_architecture_anchor` constructed-set :102.
5. **`config.X` expression readers**: ast_scanner config_refs; forward_ops `_positive_gate_fields`; patterns `_config_field` :1645, `_qk_config_atom` :1812, `_ConfigExprEvaluator.cfg_field` :2025, `_MoEGateEvaluator._cfg_get` :2486, `_linear_bias_value` :2581; stacks `_resolve_count` attr case :169; audio `_repeat_field` :479; projector `_attr_path` :307.
6. **Raw config-scope walks bypassing the U1 ledger**: position `_config_scopes` :453; conformance `_config_scopes` :1736; patterns evaluators + `attention_causality._flag` :3167 + `_cfg_num_layers` :2942; projector `_activation_value`/`_resampler_depth`; sources `_model_type` text_config fallback :629. Wrapper vocab (`text_config`… `thinker_config`) hardcoded in ≥2 places.
7. **(path,name)→ClassDef lookup**: vision cached; audio/fusion/projector uncached; inline `next(... ast.walk ...)` in position ×2, conformance ×3.
8. **Norm-kind labeling from class name**: vision `_norm_label` :967, projector `_norm_label` :795, audio `_label_for` :508, patterns `decoder_norm_kind` :1371, `diffusion_norm.base_kind` :448, `_qk_norm_type` :731.
9. **Ctor param-default extraction**: patterns :771/:928, vision `_ctor_kwarg_truthy` :869 + `_init_param_default_truthy` :897, conformance `_literal_param_defaults` :1201, stacks `_constructor_arg` :212.
10. **Execution-order Call visitor + SourceOp dedupe**: three identical copies each (vision :989/:980, audio :546/:564, projector :808/:799); `_mtime` duplicated (forward_ops :52, transitive :205); `_method` re-implemented in fusion :202.

---

## Deletion candidates (same-role-union / broad-best-candidate — U3 step 4)

- patterns.py:684, :699, :715 (whole-file union of ForwardOps / init_class_refs)
- patterns.py:602, :610, :801 (most-common VOTE resolution)
- patterns.py:1326 (`_find_decoder_layer` first-candidate fallback)
- patterns.py:1737-1765, :3071-3129, :3132-3285, :2383-2395, :2757-2803, :2953-3027, :3030-3068 (role-wide / any-class / first-hit scans)
- patterns.py:2245-2250 (router union incl. `_has_moe` NAME test; `_CONTAINER_SUFFIXES` :2144 name exclusion)
- patterns.py:1211 (`"Modulated" in node.name` gate), :573 (`.endswith("block")`), :367-377 (`_has_dense_ffn` "mlp"-in-name), :415 (`_position_field_is_fixed` "sinus"/"fixed" class name)
- vision.py:369-374 (`max(field_types)` owner pick); audio.py:145-147 (same)
- projector.py:33-35 (sorted-first candidate pick)
- conformance.py:892+:917 (ALL-blocks fallback), :1087 (`matched or fallback`), :1025+:1029 (`selected or candidates`), :1102 (`selected or blocks`), :1548-1554 (name-suffix tier + shortest-name pick — keep only as the documented no-oracle tier or delete)
- ffn.py:31-70 unanchored whole-registry scan (when architecture absent), :87 shortest-name pick
- sources.py:354-363 (`_looks_like_diffusion_class` marker gate completing the WHERE-to-look decision)
- fusion.py:176-183 + :64/:78/:143 (unparse-text substring matching)
- transformer parser `_source_files` (adapters/transformer/parser.py:119-129) — flat-files feeding; owner is already in `context.source_bundle.component_files`.

Bare-failure replacement targets (→ ReaderResult): every patterns `*_from_files` (None/False/0/{}/[]); `extract_forward_ops`/`build_registry` `{}`-on-parse-error; `transitive_closure` empty-frozenset; stacks `[]`; `scan_python_files` skip; `resolve_view_code` None; all `_class_node` Nones. Already-typed shapes to converge on: `PositionalEvidence`, `FFNStructureEvidence`, `VisionTowerEvidence`, `AudioTowerEvidence`, `ProjectorEvidence`, `FusionEvidence`, `QKNormCodeEvidence`, `RouterCodeEvidence`, `ConformanceProblem("unresolved")`.

---

## ProgramIndex schema union (every distinct extraction the index must serve)

- files/modules + provenance (bundle component key; external-library flag — `_shared_ffn_defs` parses `diffusers.models.attention` outside the bundle); parse failures/unsupported syntax as records, not skips; DETERMINISTIC CONTENT-FINGERPRINT identity (Soumil boundary 3 — mtime keying is described in this document only as the LEGACY behavior of the caches being deleted).
- import graph + aliases; called-name-filtered import closure within a package `models/` root (conformance `_imported_model_files`).
- classes: bases (for norm-math base recursion), class-body assigns (`config_class`, `config:` annotation, `_default_processor_cls`, `_available_processors`, top-level `is_causal` literals), spans.
- module-level literal dict registries (ATTENTION_CLASSES maps) with key→class dispatch.
- methods/functions: forward/`__call__`/`__init__` + helper-fold closures (init-delegated builders, forward self-method folding, init-local fn bindings), params w/ literal defaults, spans; free functions; top-level-only vs nested statement position.
- ctor fields: field→assigned EXPRESSION (Call/IfExp/Subscript-ACT2FN/Name-hop/local-ctor-var), guard context (if-branch chain incl. else-marker), every construction SITE w/ literal+default-resolved kwargs and positional-mapped args, factory-classmethod resolution, `_from_*` factories.
- ModuleList/Sequential/ModuleDict elements (comprehension/literal/append/`add_module`/local-list-wrap) + `range()` count expression + per-site ctor kwargs; `register_buffer` name/value calls.
- calls: bare tokens, `self.<field>` calls, iterated fields (`for m in self.net` / enumerate), var→fn dispatch bindings, execution-order sequence w/ line numbers (SourceOp spine), call args/keywords (residual threading, qk_norm kwargs, `bias=`, `groups=`, `activation_fn=`, width args).
- attribute reads/writes: `.T`, `.weight`-tying chains, tuple-unpacked producer detection, subscript/slice reads.
- config-path reads: `config.X` / `self.config.X` / `cfg.X`, per-layer subscript `config.X[layer_idx]`, evaluable init arithmetic over config (ConfigExprEvaluator forms) — surfaced as PATHS for the ledgered consumer, never values.
- control structure: positive-truthiness gate fields, static-evaluable decoderness/ctor-env branches, dead-branch pruning by construction-site literals, loops/comprehensions with controlling expressions, pre-loop statement ordering (embedding-stage lineage).
- dataflow: op-kind presence + gated occurrences; binop classification with operand-name evidence (projection matmul, mean/pow/rsqrt norm math); taint chains (QK-lane, embed-var lineage, norm-feeds-branch, derived-from-patch); SSA flow graph w/ input edges (audio `_flow_ops`).
- construction-graph: reachability closures, unique-root anchor, owner chains (class,field) hop lists, config-prefix propagation per constructor arg (projector `_config_param_chains` — the ComponentOwner seed), rival/conflict records instead of silent drops.
- string constants where they are declarations (`register_buffer("e_score_correction_bias",…)`, block-type strings); NO raw `ast.unparse` text matching in the target state.

---

## Tests pinning current behavior

`tests/test_code_evidence.py` (3,514 lines, 128 tests — patterns readers, position matrix :416-503, sources address rails :955-1057, QK-norm states :1227-1392, storage :790, rope-dim :1466); `test_conformance.py` (1,432 lines, ~60 tests — extractor/topology/norm-kind/dormancy :904-986, closures :1066-1095, resolver ownership :119-237, diffusion readers :349-645); `test_ffn_evidence.py`, `test_audio_evidence.py`, `test_projector_evidence.py`, `test_fusion_evidence.py`, `test_stacks.py`, `test_loud_miss.py` (token ratchet + oracle floor), `test_reader_exceptions.py` (H5 broad-except ratchet w/ per-module baseline — evidence/patterns.py:1, evidence/sources.py:2 pinned; evidence total ≤7), `test_h9_frontier.py` (metamorphic archetypes), `test_tower.py`, `test_submodel_parity.py` (embedded ≡ standalone through `slot_parse_context` — parity-sensitive to any resolution change), `test_metamorphic.py`, `test_sable.py` + `tests/sable_test_corpus/*.json` fixtures (bloom, gemma-2, glm-4-5, gpt-oss, deepseek-v3, llama, flux/flux-2, hunyuanvideo, cogvideox, mochi, ltx, lumina-2, auraflow… + `galleries/` blessed baselines) — these pin exact reader OUTPUT (op sets, positions, storage) and will catch any index-migration drift; `preservation_baseline` + manifests pin whole-render bytes.

---

## Summary

1. **~60 reader functions** across 12 evidence modules consume model source; patterns.py alone holds ~40 `*_from_files` readers with 20 of the repo's 37 `ast.parse` sites; 34 sites read model source, 3 are package self-audit (identity_guard/structural_writes/structural_debt — out of U3 scope).
2. **4 competing parse caches** (2 mtime-keyed, 2 not) plus ≥10 uncached per-call parses mean one sable run parses the same modeling file up to ~8 ways; the duplication clusters are: parse caches, ctor-assign walks, ModuleList extraction ×3, ~14 BFS reachability copies, 6+ config-expression readers, 3 raw config-scope walks outside the U1 ledger, 3 identical execution-order visitors.
3. **~24 same-role-union / broad-best-candidate sites** flagged for deletion (worst: patterns' whole-file unions/votes, `_find_decoder_layer[0]`, vision/audio `max(field-count)` owner picks, conformance ALL-blocks and `or`-fallback unions, `resolve_view_code` tier-3 shortest-name).
4. Source resolution is already one-shot and identity-as-address (`ParseContext.build` → sources ladder → `SourceBundle.component_files` → `slot_parse_context` re-rooting); the two places a NAME completes resolution are `sources._looks_like_diffusion_class` :354 (gates the diffusers search) and the domain/view NAME-marker filters in conformance (`_domain_block_classes`/`_view_class_candidates`) — plus transformer parser's `_source_files` :119 which throws the owner away by passing flat `bundle.files` to every patterns reader.
5. `position.py`'s raw walks read exactly `alibi`, `layer_types`, `_attn_implementation`, `rotary_dim`, `rotary_pct`, `partial_rotary_factor` through a hardcoded 5-wrapper scope chain (:444-466), feeding the ALiBi-vs-RoPE arbitration, zero-rotary "none" proof, hybrid mixed-schedule proof, and dispatch-class selection — all unlogged by the config-access ledger (conformance :1727-1746 and projector :761-788 are the two sibling offenders).
6. Failure shapes: 8 typed evidence dataclasses already exist (Positional, FFNStructure, VisionTower, AudioTower, Projector, Fusion, QKNormCode, RouterCode — plus ConformanceProblem as an additional typed failure surface) (Positional/FFNStructure/VisionTower/AudioTower/Projector/Fusion/QKNorm + RouterCodeEvidence) — everything else fails as bare None/False/empty; the projector width-binding rail (`_config_param_chains` projector.py:319) is the closest existing prototype of U3's ComponentOwner + config-prefix resolution and should seed the index design. `# UNVERIFIED:` exact runtime parse-count per sable run (static count of sites, not profiled); everything else above was read directly from source.

---

## U3-E owner-access classification (2026-07-21 — exhaustive; U3-E DONE, EMPTY)

Every `*_from_files` reader in `evidence/patterns.py` classified by owner-access
pattern, to test Section-6 rule 2 (exact owner already available) + rule 5
(a corpus positive AND negative) + rule 6 (answerable from current observations).
"Owner already available" today means the **root** (`resolve_component_root`);
no lawful boundary yet identifies any child (model-stage or nested) owner.

- **ROOT-ONLY (owner = the root class's own fields/forward):** ONLY
  `unet_mid_block_present`. It reads the root's own `__init__` fields + its own
  `forward` (down/up/mid call names). ✗ rule 5: SDXL (UNet2DConditionModel) is
  the corpus's ONLY UNet → a positive witness but no negative UNet witness.
  (`denoiser_temporal_axis` was the other root-only reader; migrated + deleted in
  U3-D1.)
- **WHOLE-FILE / per-class UNION (`ops.values()` / `ast.walk` over all classes):**
  diffusion_ffn_activation, diffusion_axes_dims_rope, diffusion_rope,
  diffusion_attn_kind, diffusion_ffn_kind, diffusion_qk_norm,
  diffusion_cross_qk_norm, unet_code_attention_placement,
  diffusion_single_stream_fusion, diffusion_gate_via_norm,
  decoder_layer_topology, layer_class_count, decoder_norm_kind,
  decoder_ffn_gated, decoder_ffn_activation, ffn_activation_dispatch_field,
  attention_fused_qkv, decoder_rope_dim, decoder_router_evidence,
  decoder_intermediate_size, decoder_attention_sinks, decoder_qk_norm,
  decoder_parallel_norm_count, decoder_attention_bias, decoder_mlp_bias,
  lm_head_tying, decoder_moe_schedule, embedding_stage_norm,
  expert_fused_gate_up, attention_score_scaling, attention_causality,
  decoder_cross_attention_all_layers, decoder_codebook_streams. Their faithful
  migration needs an exact non-root owner (attention/ffn/model-stage/expert),
  for which no lawful boundary exists yet.
- **CHILD / BLOCK TRAVERSE (needs a block/stage occupancy the root points to):**
  denoiser_block_timestep_conditioning; and the block-typed
  unet_transformer_ffn_activation, unet_stage_temporal, unet_stage_attn_cell
  (take `block_type`). → U3-F (decoder-block / stage occurrence discovery).

**Result:** no reader satisfies rule 2 ∧ rule 5 ∧ rule 6 simultaneously → U3-E is
empty. The next lawful boundary is U3-B1 (a declared model-stage address
resolver); nested-mechanism owners are U3-F. Selection rules are NOT weakened.
