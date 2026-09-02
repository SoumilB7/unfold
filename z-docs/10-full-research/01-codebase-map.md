# 01 — Codebase map: every module, its purpose, dependencies and health

Pass r01 of `z-docs/10-full-research/` (read-only; scratch in `scratchpad/r01/`).
Tree: `a00ae48`+ on `audio-composite-support`, inspected 2026-09-02.
Companion data: `01-codebase-map.json` (one record per module and per YAML file:
path, lines, first docstring sentence, function/class/dataclass counts, longest
function, in-package fan-out, fan-in, category, health flags; plus the dead-code,
duplicate-helper, status-vocabulary and import-cycle registers).

**Method.** A read-only Python `ast` walk over every `.py` under
`unfold-pkg/model_unfolder/` (267 files, 133,052 lines) and every
YAML under `everchanging/` (18 files). Fan-in/out are *module-level import
edges resolved to files* (relative imports resolved against the package tree; a
`from .pkg import submodule` edge points at the submodule file). "Lazy" means a
function-local import. Dead-code candidates are top-level definitions whose name
has **zero** `Name`/`Attribute`/`import` references anywhere in the package,
then word-grepped against `tests/`, `scripts/`, `test_support/`, `examples/`.
Categories follow the closed set in the brief; every assignment is a judgement
from the module's own docstring plus the U3 quarantine register
(`unfold-pkg/docs/U3_CURRENT_READER_INVENTORY.md`,
`model_unfolder/evidence/legacy_reader_quarantine.py:97-183`). Nothing here ran
the test suite. Statements I could not verify mechanically are marked
*(unverified)*.

## 1. Category summary

| category | files | lines | share | functions | classes | what it is |
|---|---:|---:|---:|---:|---:|---|
| reader | 88 | 60,177 | 45.2% | 1,882 | 276 | one exact-owner mechanism proof per module (`evidence/attention*.py`, `position_*.py`, `unet_*.py`, `diffusion_*.py`, …) |
| substrate | 46 | 27,334 | 20.5% | 873 | 200 | `ProgramIndex`, owner graph, execution flow, sources/context/document, typed facts/results, config-access ledger, IR + op-graph DTOs |
| adapter | 36 | 18,230 | 13.7% | 439 | 14 | the two config→IR parsers and their block/assembly helpers (`adapters/transformer`, `adapters/diffusor`, `parser.py`, `submodel.py`) |
| consumer | 73 | 16,537 | 12.4% | 458 | 14 | HTML/SVG renderers, expanded JSON, params, labels, conformance nets |
| gate | 8 | 5,445 | 4.1% | 194 | 17 | scanners of our own code and register enforcers (identity guard, consumer firewall, structural debt/writes, quarantine, receipts, projection audit) |
| legacy | 5 | 3,082 | 2.3% | 127 | 3 | quarantined pre-ProgramIndex parsers/readers with an assigned deletion unit (U11/U14) |
| product | 10 | 1,792 | 1.3% | 65 | 8 | public API, `Diagram`, `sable`, lint, errors, preview |
| vocab | 1 (+18 YAML) | 455 (+1,318 YAML) | 0.3% | 27 | 0 | `everchanging/` loader + the YAML it loads |
| **total** | **267** | **133,052** | 100% | **4,065** | **532** (497 dataclasses) | |

Two numbers frame everything else: **65.7 % of the code is the evidence layer**
(reader + substrate), and the **two parsers plus their helpers are 13.7 %**, of
which one function — `adapters/transformer/parser.py:876 parse()` — is 3,762
lines, i.e. 2.8 % of the whole package in a single body.

## 2. Per-category module inventory

Columns: fan-in = number of package modules importing this one; fan-out =
number of package modules it imports. Flags cite `file:line`. Full per-module
records (including the import lists) are in the JSON.

### product (10 files, 1,792 lines)

| path | lines | purpose | fan-in | fan-out | health flags |
|---|---:|---|---:|---:|---|
| `__init__.py` | 102 | model_unfolder — turn any HuggingFace transformer into a clear architecture diagram. | 0 | 8 | — |
| `diagram.py` | 195 | Diagram — the renderable object. | 2 | 6 | — |
| `errors.py` | 55 | Public exception types raised by model-unfolder. | 5 | 0 | — |
| `evidence/__init__.py` | 54 | Static source-code evidence for model topology validation. | 3 | 9 | import-cycle:SCC#6 (lazy-broken; see cycles table) |
| `evidence/inspector.py` | 28 | Public inspection entry points for static code evidence. | 1 | 4 | — |
| `html_renderer.py` | 5 | Compatibility wrapper for the HTML/SVG renderer backend. | 1 | 1 | — |
| `input_formats.py` | 100 | Structural input-format normalization. | 1 | 2 | — |
| `lint.py` | 177 | Label lint — cheap, mechanical guards on the TEXT a block draws. | 2 | 1 | — |
| `preview.py` | 221 | Render the baked diagram to PNG images — pixels as a first-class oracle. | 2 | 0 | — |
| `sable.py` | 855 | Sable — the one-command quality harness for a model. | 1 | 17 | god-function:sable=301L (sable.py:360); lazy-imports:37 function-local in-package imports (first sable.py:216) |

### adapter (36 files, 18,230 lines)

| path | lines | purpose | fan-in | fan-out | health flags |
|---|---:|---|---:|---:|---|
| `adapters/__init__.py` | 15 | Adapter registry. | 2 | 3 | — |
| `adapters/custom/__init__.py` | 8 | Project-local and experimental adapters. | 1 | 0 | — |
| `adapters/diffusor/__init__.py` | 11 | Diffusor-family adapters. | 1 | 1 | — |
| `adapters/diffusor/blocks.py` | 1393 | Model-level block declarations for diffusion (DiT/MMDiT) pipelines. | 2 | 8 | spine:adapter->consumer labels.py (adapters/diffusor/blocks.py:19); spine:adapter->consumer labels.py (adapters/diffusor/blocks.py:19) |
| `adapters/diffusor/compound.py` | 131 | Typed compound-stage facts for diffusion architectures. | 2 | 0 | — |
| `adapters/diffusor/config_binding.py` | 774 | U10-F2 — exact checkpoint operands for the diffusion source projection. | 2 | 20 | — |
| `adapters/diffusor/loader.py` | 152 | Load a diffusion pipeline's denoiser config by HuggingFace model ID. | 1 | 2 | spine:adapter->product errors.py (adapters/diffusor/loader.py:61,lazy); import-cycle:SCC#6 (lazy-broken; see cycles table) |
| `adapters/diffusor/parser.py` | 1036 | Diffusion adapter routing and passive projection. | 1 | 27 | spine:adapter->consumer encoder_panel.py (adapters/diffusor/parser.py:1034); spine:adapter->consumer conformance.py (adapters/diffusor/parser.py:56,lazy); spine:adapter->consumer conformance.py (adapters/diffusor/parser.py:112,lazy); lazy-imports:37 function-local in-package imports (first adapters/diffusor/parser.py:267) |
| `adapters/diffusor/projection_ir.py` | 1428 | U10-F3 — the sole diffusion source/config projection into typed IR. | 2 | 12 | lazy-imports:12 function-local in-package imports (first adapters/diffusor/projection_ir.py:609) |
| `adapters/diffusor/schema.py` | 471 | U10-F1 — closed, source-only diffusion projection schema. | 3 | 12 | — |
| `adapters/diffusor/unet.py` | 971 | UNet (UNet2DConditionModel) diffusion denoisers — SD1.5 / SD2 / SDXL / Kandinsky. | 1 | 8 | spine:adapter->consumer labels.py (adapters/diffusor/unet.py:757,lazy) |
| `adapters/transformer/__init__.py` | 8 | Transformer-LLM adapter. | 1 | 1 | — |
| `adapters/transformer/assembly.py` | 158 | Assembly helpers for transformer-family adapters. | 2 | 2 | — |
| `adapters/transformer/blocks/__init__.py` | 54 | Reusable transformer block descriptions for renderers. | 1 | 5 | spine:adapter->consumer labels.py (adapters/transformer/blocks/__init__.py:32); spine:adapter->consumer labels.py (adapters/transformer/blocks/__init__.py:32) |
| `adapters/transformer/blocks/attention.py` | 1087 | Reusable attention-family child block declarations. | 4 | 4 | god-function:_sdpa_detailed_child_blocks=325L (adapters/transformer/blocks/attention.py:223); spine:adapter->consumer labels.py (adapters/transformer/blocks/attention.py:7); import-cycle:SCC#3 (lazy-broken; see cycles table) |
| `adapters/transformer/blocks/feed_forward.py` | 637 | Reusable FFN-family child block declarations. | 4 | 6 | spine:adapter->consumer labels.py (adapters/transformer/blocks/feed_forward.py:7); spine:adapter->consumer labels.py (adapters/transformer/blocks/feed_forward.py:7); spine:adapter->consumer labels.py (adapters/transformer/blocks/feed_forward.py:491,lazy); spine:adapter->consumer mixture_of_experts.py (adapters/transformer/blocks/feed_forward.py:493,lazy); import-cycle:SCC#3 (lazy-broken; see cycles table) |
| `adapters/transformer/blocks/layers.py` | 498 | Reusable decoder-layer topology declarations. | 1 | 6 | spine:adapter->consumer labels.py (adapters/transformer/blocks/layers.py:8); spine:adapter->consumer labels.py (adapters/transformer/blocks/layers.py:8); spine:adapter->consumer labels.py (adapters/transformer/blocks/layers.py:8); spine:adapter->consumer labels.py (adapters/transformer/blocks/layers.py:8); spine:adapter->consumer labels.py (adapters/transformer/blocks/layers.py:8); spine:adapter->consumer labels.py (adapters/transformer/blocks/layers.py:450,lazy); spine:adapter->consumer labels.py (adapters/transformer/blocks/layers.py:450,lazy) |
| `adapters/transformer/blocks/model.py` | 434 | Model-level transformer block declarations. | 2 | 2 | — |
| `adapters/transformer/common.py` | 101 | Shared helpers for transformer-family config adapters. | 11 | 2 | import-cycle:SCC#2 (lazy-broken; see cycles table) |
| `adapters/transformer/debug.py` | 345 | Centralized parse-time debugging — one switch for all diagnostics. | 3 | 3 | import-cycle:SCC#2 (lazy-broken; see cycles table); dead:_value_at@97 |
| `adapters/transformer/parser.py` | 4996 | The transformer-LLM parser — the only adapter. | 2 | 54 | god-module:4996L (adapters/transformer/parser.py:1); god-function:parse=3762L (adapters/transformer/parser.py:876); lazy-imports:67 function-local in-package imports (first adapters/transformer/parser.py:95); dead:_source_files@86,_code_expert_storage@703,_unwrap_text@808,_norm_kind_evidence@4865 |
| `adapters/transformer/special_parts/__init__.py` | 2 | Reusable transformer layer parts. | 0 | 0 | — |
| `adapters/transformer/special_parts/modalities/__init__.py` | 6 | Multimodal pathway extraction public API. | 1 | 1 | — |
| `adapters/transformer/special_parts/modalities/accessors.py` | 57 | Small config access helpers for multimodal extraction. | 5 | 1 | import-cycle:SCC#2 (lazy-broken; see cycles table) |
| `adapters/transformer/special_parts/modalities/audio.py` | 33 | Audio modality address shell. | 1 | 1 | import-cycle:SCC#2 (lazy-broken; see cycles table) |
| `adapters/transformer/special_parts/modalities/builder.py` | 132 | Top-level multimodal extras assembly. | 1 | 5 | — |
| `adapters/transformer/special_parts/modalities/conditioning.py` | 54 | Conditioning-component address shell. | 1 | 2 | import-cycle:SCC#2 (lazy-broken; see cycles table) |
| `adapters/transformer/special_parts/modalities/detect.py` | 59 | Config-declared multimodal feature detection. | 2 | 1 | import-cycle:SCC#2 (lazy-broken; see cycles table) |
| `adapters/transformer/special_parts/modalities/evidence_projection.py` | 548 | Pure U9 typed-evidence projection into modality path specifications. | 1 | 4 | — |
| `adapters/transformer/special_parts/modalities/fusion.py` | 213 | Model-level modality fusion extraction. | 2 | 2 | — |
| `adapters/transformer/special_parts/modalities/registry.py` | 106 | Modality registry — the single place that enumerates input modalities. | 3 | 5 | import-cycle:SCC#2 (lazy-broken; see cycles table) |
| `adapters/transformer/special_parts/modalities/schema.py` | 214 | Tiny payload builders for multimodal extras. | 5 | 1 | import-cycle:SCC#2 (lazy-broken; see cycles table) |
| `adapters/transformer/special_parts/modalities/vision.py` | 344 | Vision and video modality path extraction. | 2 | 6 | import-cycle:SCC#2 (lazy-broken; see cycles table) |
| `adapters/transformer/special_parts/per_layer_embedding.py` | 220 | Reusable Per-Layer Embedding (PLE) transformer part. | 1 | 2 | spine:adapter->consumer labels.py (adapters/transformer/special_parts/per_layer_embedding.py:10) |
| `parser.py` | 875 | Parse a HuggingFace config (or model ID) into our IR. | 4 | 15 | god-function:config_to_ir=376L (parser.py:23); spine:adapter->product errors.py (parser.py:8); spine:adapter->product errors.py (parser.py:8); spine:adapter->product errors.py (parser.py:8); spine:adapter->product errors.py (parser.py:8); spine:adapter->product __init__.py (parser.py:449,lazy); spine:adapter->product __init__.py (parser.py:449,lazy); spine:adapter->product input_formats.py (parser.py:819,lazy); lazy-imports:19 function-local in-package imports (first parser.py:74); import-cycle:SCC#6 (lazy-broken; see cycles table) |
| `submodel.py` | 659 | First-class embedded sub-models — ONE recursive projection for every supporting tower. | 4 | 6 | spine:adapter->consumer labels.py (submodel.py:301,lazy); spine:adapter->consumer labels.py (submodel.py:368,lazy); spine:adapter->consumer labels.py (submodel.py:408,lazy); spine:adapter->consumer labels.py (submodel.py:408,lazy); spine:adapter->consumer labels.py (submodel.py:408,lazy); spine:adapter->consumer labels.py (submodel.py:513,lazy); spine:adapter->consumer labels.py (submodel.py:513,lazy); lazy-imports:16 function-local in-package imports (first submodel.py:178); import-cycle:SCC#3 (lazy-broken; see cycles table) |

### substrate (46 files, 27,334 lines)

| path | lines | purpose | fan-in | fan-out | health flags |
|---|---:|---|---:|---:|---|
| `block_schema.py` | 400 | Schema + validation for render blocks — the dict tree the diagram draws. | 11 | 2 | spine:substrate->consumer registry.py (block_schema.py:378,lazy); import-cycle:SCC#3 (lazy-broken; see cycles table) |
| `evidence/activation_semantics.py` | 24 | Shared, syntax-level activation protocol vocabulary. | 4 | 0 | — |
| `evidence/affine.py` | 186 | Closed construction protocols for affine projection storage. | 9 | 2 | — |
| `evidence/arbitration.py` | 278 | Evidence arbitration — ranking candidates for ONE mechanism by their origin. | 0 | 2 | zero-fan-in (evidence/arbitration.py:1) |
| `evidence/call_arguments.py` | 262 | Neutral exact call-argument to callee-formal binding. | 6 | 4 | — |
| `evidence/component_inventory.py` | 205 | U9-A — exact inventory of root-owned source components. | 3 | 4 | — |
| `evidence/component_owner.py` | 2155 | U3-B — exact construction-occurrence ownership over ``ProgramIndex``. | 98 | 2 | god-module:2155L (evidence/component_owner.py:1); dup-helpers:6 same-named small helpers also defined elsewhere (_span_before@36, _self_attribute_chain@1214, _plain_name@1210, _attribute_chain@1372) |
| `evidence/component_stages.py` | 220 | Exact repeated-stage inventory below one active component occurrence. | 1 | 5 | — |
| `evidence/config_access.py` | 1546 | H3 (restart) — the owner-scoped config-access event ledger (plan §16.5). | 16 | 3 | god-module:1546L (evidence/config_access.py:1); spine:substrate->gate receipts.py (evidence/config_access.py:960,lazy); import-cycle:SCC#1 (lazy-broken; see cycles table) |
| `evidence/config_guard.py` | 464 | Exact evaluation of source guards whose operands bind to config paths. | 15 | 4 | spine:substrate->reader attention.py (evidence/config_guard.py:13); dup-helpers:4 same-named small helpers also defined elsewhere (_span_before@178, _self_field@437, _span_key@172, _span_within@183) |
| `evidence/config_registration.py` | 304 | Exact framework registration of constructor parameters as config fields. | 5 | 4 | — |
| `evidence/config_scoped_owner.py` | 945 | Resolve a selected nested config scope to its exact constructed model root. | 3 | 4 | dup-helpers:5 same-named small helpers also defined elsewhere (_span_between@867, _failed@935, _expr_nodes@512, _candidate_sort_key@915) |
| `evidence/construction_arguments.py` | 257 | Neutral exact construction-argument to ``__init__``-formal binding. | 4 | 2 | — |
| `evidence/construction_calls.py` | 435 | U3-F3a — exact construction-call addresses, including external primitives. | 59 | 2 | — |
| `evidence/constructor_condition.py` | 403 | Select an exact call argument using constructor-proven field values. | 4 | 5 | dup-helpers:3 same-named small helpers also defined elsewhere (_expression_names@349, _failed@392, _self_fields@334) |
| `evidence/constructor_fields.py` | 428 | Exact instance-field values derived from constructor-formal transport. | 1 | 4 | dup-helpers:3 same-named small helpers also defined elsewhere (_span_before@415, _names@397, _target_contains_name@409) |
| `evidence/constructor_values.py` | 524 | Exact literal constructor values across explicitly supplied owner routes. | 7 | 5 | — |
| `evidence/container_inventory.py` | 403 | U3-B2 — neutral container address inventory. | 35 | 2 | — |
| `evidence/context.py` | 452 | One immutable source-resolution context shared by one model parse. | 15 | 10 | spine:substrate->reader decoderness.py (evidence/context.py:392,lazy); spine:substrate->reader decoderness.py (evidence/context.py:257,lazy); lazy-imports:11 function-local in-package imports (first evidence/context.py:286); import-cycle:SCC#1 (lazy-broken; see cycles table) |
| `evidence/decoder_block.py` | 563 | U3-F — one exact path from a selected config scope to its decoder block. | 33 | 10 | — |
| `evidence/decoder_stage.py` | 111 | Exact selected transformer stage before any repeated-child interpretation. | 2 | 5 | — |
| `evidence/delegated_stage.py` | 195 | Exact return-delegation address evidence for nested model stages. | 1 | 5 | dup-helpers:4 same-named small helpers also defined elsewhere (_span_before@185, _forward@128, _same_expression@168, _single_name_target@177) |
| `evidence/dispatch_selection.py` | 405 | Exact code-registry + config-value construction selection. | 2 | 4 | — |
| `evidence/document.py` | 386 | The ONE document-preparation primitive. | 8 | 2 | import-cycle:SCC#1 (lazy-broken; see cycles table) |
| `evidence/execution_flow.py` | 1237 | U3 Phase 3+4 — addressed invocation resolver and conservative versioned def-use execution-flow resolver. | 35 | 4 | dead:_span_sort@1218 |
| `evidence/expression_eval.py` | 725 | Exact, owner-scoped evaluation of structural source expressions. | 24 | 1 | — |
| `evidence/expression_value.py` | 225 | Evaluate one exact owner expression from code and typed config premises. | 2 | 3 | spine:substrate->reader attention.py (evidence/expression_value.py:14); dup-helpers:5 same-named small helpers also defined elsewhere (_span_before@207, _self_field@176, _name_is_shadowed@182, _freeze@213) |
| `evidence/facts.py` | 331 | H1 evidence primitives: typed facts with owner, completeness, source spans, premises, and the negative-proof law (har… | 6 | 1 | import-cycle:SCC#1 (lazy-broken; see cycles table) |
| `evidence/framework_config.py` | 1316 | Exact framework-owned ``self.config`` address evidence. | 13 | 4 | — |
| `evidence/framework_operations.py` | 145 | Neutral exact-import protocols for supported framework operations. | 2 | 2 | — |
| `evidence/identity_roles.py` | 58 | Typed address/display markers for lawful identity use (plan §16.2). | 5 | 0 | — |
| `evidence/import_source.py` | 447 | U11-A — demand-driven exact called-import source expansion. | 6 | 2 | — |
| `evidence/invocation_source.py` | 407 | Exact formal-to-formal source routes across addressed Python calls. | 2 | 5 | spine:substrate->reader diffusion_stream.py (evidence/invocation_source.py:15); dup-helpers:4 same-named small helpers also defined elsewhere (_failed@396, _bind_call@38, _span_within@71, _ordinary_params@32) |
| `evidence/layer_selector.py` | 1069 | U8-A — exact, mechanism-neutral per-layer construction selection. | 5 | 5 | spine:substrate->reader attention.py (evidence/layer_selector.py:20); dup-helpers:7 same-named small helpers also defined elsewhere (_span_before@1015, _self_field@926, _span_contains@1021, _span_key@1009) |
| `evidence/models.py` | 410 | Data containers for static model-code evidence. | 66 | 0 | — |
| `evidence/output_repeated_stage.py` | 613 | Exact output-lineage address proof for a nested repeated stage. | 2 | 6 | dup-helpers:4 same-named small helpers also defined elsewhere (_span_before@419, _self_field@525, _span_key@426, _within@500) |
| `evidence/primitive_semantics.py` | 466 | U3-F3b — primitive semantics from exact construction + implementation evidence. | 9 | 3 | dup-helpers:3 same-named small helpers also defined elsewhere (_self_field@379, _name_is_shadowed@357, _resolved_call_target@348) |
| `evidence/program_index.py` | 2929 | U3 — the ONE raw program index (master plan §20.6; docs/U3_RUNBOOK.md). | 122 | 1 | god-module:2929L (evidence/program_index.py:1); lawful-parse:ast.parse (evidence/program_index.py:2790); dup-helpers:9 same-named small helpers also defined elsewhere (_self_field@1799, _params@1527, _span_after@2466, _site@2020) |
| `evidence/qualification.py` | 394 | Owner-qualified, value-exact structural projection matrix. | 1 | 0 | — |
| `evidence/reader_result.py` | 262 | U3-C — typed results for one exact owner occurrence. | 99 | 2 | — |
| `evidence/registry.py` | 1502 | H2 — the closed fact registry (hardening plan §3.4). | 8 | 3 | god-module:1502L (evidence/registry.py:1); import-cycle:SCC#1 (lazy-broken; see cycles table) |
| `evidence/repeated_child.py` | 266 | U3-F2 — exact repeated-child occurrence boundary. | 9 | 4 | — |
| `evidence/self_method_return.py` | 297 | Neutral exact transport across one indexed ``self.method(...)`` return. | 2 | 3 | — |
| `evidence/sources.py` | 822 | Source discovery for static Hugging Face modeling-code inspection. | 7 | 2 | lawful-parse:ast.parse (evidence/sources.py:693) |
| `ir.py` | 612 | Intermediate representation shared by evidence and every projection. | 19 | 0 | — |
| `opgraph.py` | 1250 | The canonical operation graph — one structural source, no rendering, no config parsing. | 12 | 0 | — |

### reader (88 files, 60,177 lines)

| path | lines | purpose | fan-in | fan-out | health flags |
|---|---:|---|---:|---:|---|
| `evidence/attention.py` | 5324 | Owner-qualified attention mechanism evidence. | 21 | 11 | god-module:5324L (evidence/attention.py:1); dead:_latest_unconditional_binding@2783; dup-helpers:12 same-named small helpers also defined elsewhere (_is_negative_one@3906, _expr_contains_span@3955, _expr_contains_name@3965, _span_before@4005) |
| `evidence/attention_child.py` | 1138 | U3-F5a — positively prove an attention-compute child of one exact block. | 24 | 7 | dead:attention_compute_proof_for_symbol@532; dup-helpers:10 same-named small helpers also defined elsewhere (_span_before@842, _span_sort_key@1119, _target_names@799, _self_field@1111) |
| `evidence/attention_container_interface.py` | 446 | Unanimous code proof for a container's default attention input interface. | 2 | 5 | dup-helpers:5 same-named small helpers also defined elsewhere (_span_before@425, _self_field@395, _target_is_name@403, _span_key@432) |
| `evidence/attention_geometry.py` | 857 | Exact-owner attention head-dimension evidence. | 7 | 13 | dup-helpers:7 same-named small helpers also defined elsewhere (_expr_contains_span@711, _expr_contains_name@702, _span_before@720, _span_key@846) |
| `evidence/attention_input_interface.py` | 520 | Code-proven primary/context input interface for one attention implementation. | 1 | 6 | dup-helpers:7 same-named small helpers also defined elsewhere (_span_before@500, _target_names@489, _target_is_name@496, _span_key@507) |
| `evidence/attention_invocation_role.py` | 500 | Occurrence-qualified self vs contextual-input attention call evidence. | 1 | 8 | dup-helpers:3 same-named small helpers also defined elsewhere (_failed@490, _unique_values@431, _bind_call@465) |
| `evidence/attention_lane.py` | 691 | Positive attention lanes, including exact processor-delegation protocols. | 7 | 10 | — |
| `evidence/attention_mask.py` | 1930 | Exact framework-mask construction reaching an exact repeated block. | 1 | 17 | god-module:1930L (evidence/attention_mask.py:1); dup-helpers:11 same-named small helpers also defined elsewhere (_expr_contains_span@1809, _target_names@1825, _expression_names@1856, _span_key@1895) |
| `evidence/attention_operands.py` | 306 | Exact, role-neutral attention score operand evidence. | 3 | 6 | — |
| `evidence/attention_output.py` | 520 | Exact attention output-projection evidence. | 2 | 9 | dup-helpers:7 same-named small helpers also defined elsewhere (_expr_contains_span@468, _span_before@492, _target_names@481, _self_field@508) |
| `evidence/attention_score_additives.py` | 569 | Exact score-side additive application evidence. | 3 | 7 | dup-helpers:3 same-named small helpers also defined elsewhere (_expr_contains_span@504, _target_names@484, _span_key@515) |
| `evidence/attention_sinks.py` | 426 | Exact-owner learned attention-sink evidence. | 1 | 7 | dup-helpers:7 same-named small helpers also defined elsewhere (_span_before@414, _target_names@378, _exact_call_target@388, _span_key@402) |
| `evidence/attention_storage.py` | 1175 | U3-F5b — exact Q/K/V projection-storage evidence. | 28 | 12 | import-cycle:SCC#4 (lazy-broken; see cycles table); dup-helpers:6 same-named small helpers also defined elsewhere (_span_before@1136, _span_sort_key@1147, _target_names@1117, _self_field@1128) |
| `evidence/cell_topology.py` | 1972 | Exact-owner decoder cell topology from positive residual equations. | 5 | 13 | god-module:1972L (evidence/cell_topology.py:1); dup-helpers:5 same-named small helpers also defined elsewhere (_span_key@1946, _field_config_path@951, _expr_nodes@1316, _position@1932) |
| `evidence/codebook_streams.py` | 408 | U3-G — exact-owner multi-codebook aggregation evidence. | 1 | 7 | — |
| `evidence/component_operations.py` | 635 | U9-D/E — exact stage-boundary and repeated-block operation projections. | 2 | 10 | dup-helpers:6 same-named small helpers also defined elsewhere (_self_field@616, _span_key@608, _contains@596, _before@604) |
| `evidence/component_position.py` | 387 | Exact learned multi-axis position lookup before a nested repeated stage. | 1 | 7 | dup-helpers:4 same-named small helpers also defined elsewhere (_span_key@381, _before@374, _self_method@258, _parameter_origins@322) |
| `evidence/component_tower.py` | 893 | U9-D2 — recursive mechanism facts for exact active component towers. | 3 | 25 | — |
| `evidence/cross_attention_replacement.py` | 433 | Exact per-layer replacement cross-attention schedule. | 3 | 12 | — |
| `evidence/cross_attention_schedule.py` | 331 | Exact dual-attention construction evidence for every decoder block. | 3 | 7 | — |
| `evidence/decoder_norm.py` | 519 | Exact-owner decoder-block normalization primitive evidence. | 6 | 11 | dead:norm_invocations_in_graph@276 |
| `evidence/decoderness.py` | 62 | Config-declared decoder-ness (U2 mask default-kill). | 2 | 1 | — |
| `evidence/denoiser.py` | 95 | U3-D1 — the denoiser temporal-axis reader, bound to one exact owner. | 1 | 4 | — |
| `evidence/diffusion_block.py` | 455 | U10-C — canonical mechanism facts for exact diffusion block occurrences. | 6 | 18 | — |
| `evidence/diffusion_bookends.py` | 577 | U10-E — exact diffusion-root bookend and 3-D operation evidence. | 4 | 8 | dup-helpers:7 same-named small helpers also defined elsewhere (_target_names@77, _self_field@302, _expression_names@86, _span_key@58) |
| `evidence/diffusion_companion.py` | 263 | U10-E — independent companion-denoiser source comparison. | 3 | 10 | — |
| `evidence/diffusion_conditioning.py` | 529 | U10-D — exact conditioning applications in diffusion blocks. | 5 | 5 | — |
| `evidence/diffusion_root.py` | 476 | U10-A — exact diffusion-root topology evidence. | 6 | 4 | dup-helpers:9 same-named small helpers also defined elsewhere (_target_names@106, _self_field@66, _names@95, _forward@254) |
| `evidence/diffusion_stack.py` | 773 | U10-B — occurrence-exact diffusion repeated-stack inventory. | 5 | 8 | dup-helpers:5 same-named small helpers also defined elsewhere (_target_names@89, _self_field@77, _span_key@61, _within@68) |
| `evidence/diffusion_stream.py` | 1409 | U10-D — exact local stream relations for diffusion block occurrences. | 11 | 11 | dup-helpers:4 same-named small helpers also defined elsewhere (_span_key@49, _before@53, _forward@914, _guard_prefix@60) |
| `evidence/dispatch_attention_mechanism.py` | 737 | Candidate-equivalent attention mechanism evidence at literal dispatch sites. | 1 | 5 | dup-helpers:4 same-named small helpers also defined elsewhere (_dependency_closure@672, _target_names@436, _self_field@723, _walk_expr@660) |
| `evidence/dispatch_attention_storage.py` | 454 | U3-F5d — candidate-equivalent attention storage at dispatch sites. | 2 | 6 | import-cycle:SCC#4 (lazy-broken; see cycles table); dup-helpers:4 same-named small helpers also defined elsewhere (_calls_exact_super_init@363, _is_super_init_call@377, _self_field@426, _exact_call_target@415) |
| `evidence/embedding_bookend.py` | 243 | U3-F4 — exact embedding-stage normalization evidence. | 2 | 9 | — |
| `evidence/expert_storage.py` | 1556 | Exact-address, positive-only routed-expert storage evidence. | 6 | 8 | god-module:1556L (evidence/expert_storage.py:1); dead:_field_referenced_in_spans@1132; dup-helpers:11 same-named small helpers also defined elsewhere (_target_names@1471, _self_field@1499, _callable_expressions@1455, _span_key@1528) |
| `evidence/expert_width.py` | 619 | Exact fused routed-expert intermediate-width evidence. | 1 | 10 | dup-helpers:6 same-named small helpers also defined elsewhere (_dependency_closure@537, _self_field@529, _span_key@551, _same_expression@598) |
| `evidence/ffn_input_transform.py` | 493 | Positive code proof for one FFN input projection transform. | 1 | 6 | dup-helpers:5 same-named small helpers also defined elsewhere (_target_names@461, _self_field@436, _span_key@484, _call_at_span@430) |
| `evidence/ffn_mechanism.py` | 1639 | Exact-owner dense feed-forward mechanism evidence. | 15 | 11 | god-module:1639L (evidence/ffn_mechanism.py:1); dup-helpers:8 same-named small helpers also defined elsewhere (_dependency_closure@1465, _self_field@1595, _span_key@1617, _span_within@1602) |
| `evidence/ffn_schedule.py` | 758 | U8-E — occurrence-exact dense/routed-FFN placement. | 1 | 12 | — |
| `evidence/ffn_width.py` | 233 | Exact-owner FFN intermediate-width evidence. | 1 | 8 | — |
| `evidence/final_bookend.py` | 421 | Exact model-stage final-normalization evidence. | 1 | 8 | dup-helpers:6 same-named small helpers also defined elsewhere (_span_sort_key@417, _target_names@373, _span_key@391, _position@386) |
| `evidence/fusion.py` | 654 | U9-B — exact owner-bound wrapper fusion evidence. | 6 | 6 | dup-helpers:7 same-named small helpers also defined elsewhere (_target_names@612, _self_field@575, _names@605, _forward@557) |
| `evidence/kv_sharing_schedule.py` | 448 | U8-E exact cross-layer K/V reuse schedule. | 1 | 7 | dup-helpers:5 same-named small helpers also defined elsewhere (_expression_contains_name@422, _self_field@398, _plain_name@394, _guard_value@329) |
| `evidence/mixer_schedule.py` | 565 | Exact per-layer placement of already-proven decoder mixer mechanisms. | 5 | 14 | — |
| `evidence/mtp.py` | 492 | Exact repeated auxiliary next-token predictor evidence. | 1 | 6 | dup-helpers:6 same-named small helpers also defined elsewhere (_target_names@435, _self_field@475, _failed@157, _contains_name@483) |
| `evidence/multiaxis_position.py` | 305 | U9-E — exact multimodal multi-axis position-construction route. | 2 | 6 | dup-helpers:3 same-named small helpers also defined elsewhere (_target_names@249, _before@285, _unique_calls@292) |
| `evidence/parallel_norm.py` | 451 | Exact input-normalization evidence for parallel decoder branches. | 2 | 12 | — |
| `evidence/per_layer_side_input.py` | 409 | Exact per-layer side-input injection evidence. | 1 | 4 | dup-helpers:7 same-named small helpers also defined elsewhere (_target_names@192, _self_field@392, _failed@148, _contains_name@400) |
| `evidence/position_absolute.py` | 319 | Exact model-stage learned-absolute position application evidence. | 3 | 11 | — |
| `evidence/position_application.py` | 1316 | U8-B — exact Q/K position-rotation application evidence. | 5 | 9 | dead:_occurrence_key@1297; dup-helpers:8 same-named small helpers also defined elsewhere (_is_negative_one@963, _span_before@1290, _expression_names@500, _parameter_origins@1266) |
| `evidence/position_coordinate.py` | 365 | Neutral, exact position-coordinate origin evidence. | 3 | 2 | dup-helpers:3 same-named small helpers also defined elsewhere (_span_before@351, _span_key@360, _resolved_call_target@345) |
| `evidence/position_factors.py` | 815 | U8-B2 — exact position-derived rotation-factor provenance. | 2 | 12 | dup-helpers:5 same-named small helpers also defined elsewhere (_span_before@782, _span_sort_key@789, _forward_failure@801, _target_paths@583) |
| `evidence/position_fixed.py` | 527 | Exact fixed-sinusoidal pre-stack position evidence. | 2 | 11 | dup-helpers:7 same-named small helpers also defined elsewhere (_expr_contains_span@496, _expr_contains_name@487, _target_names@457, _self_field@466) |
| `evidence/position_geometry.py` | 710 | Exact applied Q/K position-rotation geometry. | 1 | 12 | dup-helpers:11 same-named small helpers also defined elsewhere (_span_before@653, _span_strict_before@673, _span_key@659, _exact_target@647) |
| `evidence/position_initialization.py` | 960 | Exact initialization of the stored frequency base used by applied RoPE. | 1 | 10 | dup-helpers:7 same-named small helpers also defined elsewhere (_span_before@937, _span_key@944, _failed@949, _selected_value@891) |
| `evidence/position_linear_bias.py` | 698 | Exact linear-coordinate attention-bias evidence (ALiBi mechanism). | 2 | 10 | dup-helpers:10 same-named small helpers also defined elsewhere (_expr_contains_span@669, _expr_contains_name@657, _target_names@636, _self_field@661) |
| `evidence/position_relative_bias.py` | 753 | Exact learned relative-coordinate attention-bias evidence. | 2 | 8 | dup-helpers:7 same-named small helpers also defined elsewhere (_expr_contains_span@727, _target_names@638, _self_field@643, _span_key@743) |
| `evidence/position_schedule.py` | 511 | Exact per-layer selection of one already-proven position application. | 3 | 13 | dup-helpers:4 same-named small helpers also defined elsewhere (_selected_value@457, _freeze@478, _name_shadowed@449, _forward_failure@497) |
| `evidence/position_table.py` | 170 | Exact direct absolute-position table application inside a repeated stage. | 1 | 7 | — |
| `evidence/projection_bias.py` | 475 | Exact projection-bias evidence for decoder attention and ordinary FFNs. | 1 | 11 | dead:_merge_result@450 |
| `evidence/projector.py` | 475 | Exact multimodal projector/merger evidence from qualified HF source. | 3 | 13 | — |
| `evidence/projector_chain.py` | 781 | U9-C — exact owner-qualified callable operation chains. | 7 | 9 | dup-helpers:8 same-named small helpers also defined elsewhere (_self_field@752, _expression_names@577, _span_key@739, _before@728) |
| `evidence/projector_lineage.py` | 1039 | U9-C — projector selection by exact producer lineage. | 3 | 11 | dup-helpers:9 same-named small helpers also defined elsewhere (_self_field@1013, _expr_key@1021, _span_key@1032, _contains@971) |
| `evidence/projector_width.py` | 399 | U9-C — exact affine width operands for one proven projector lineage. | 1 | 5 | dup-helpers:5 same-named small helpers also defined elsewhere (_self_field@382, _span_key@395, _before@389, _occurrence_prefix@377) |
| `evidence/qk_norm.py` | 846 | U3-F — Q/K normalization from one exact attention occurrence. | 5 | 9 | dup-helpers:6 same-named small helpers also defined elsewhere (_span_before@821, _self_field@792, _binding_target_names@555, _exact_target@803) |
| `evidence/qk_norm_schedule.py` | 240 | U8-E — exact per-layer Q/K-normalization placement. | 1 | 8 | — |
| `evidence/repeated_projector.py` | 722 | U9-D — exact affine-prefix -> repeated-stage projector pipeline. | 1 | 13 | dup-helpers:8 same-named small helpers also defined elsewhere (_span_before@687, _target_names@664, _span_key@680, _attribute_chain@642) |
| `evidence/repeated_stage.py` | 225 | U9-D — reusable mechanisms for one exact repeated component stage. | 1 | 10 | — |
| `evidence/rope_config_normalization.py` | 299 | Exact source proof for the framework's legacy-to-runtime RoPE config map. | 1 | 3 | dup-helpers:3 same-named small helpers also defined elsewhere (_spans@284, _self_method@220, _present@279) |
| `evidence/router.py` | 1920 | Exact-owner MoE router-selection evidence. | 1 | 10 | god-module:1920L (evidence/router.py:1); dup-helpers:17 same-named small helpers also defined elsewhere (_expr_contains_span@1822, _span_before@1872, _dependency_closure@1843, _target_names@1540) |
| `evidence/selected_composite_ffn.py` | 643 | Exact selector-to-composite-FFN proof for framework container modules. | 0 | 9 | zero-fan-in (evidence/selected_composite_ffn.py:1); dup-helpers:4 same-named small helpers also defined elsewhere (_self_field@615, _span_key@632, _within@624, _call_at_span@609) |
| `evidence/separate_rotary.py` | 240 | Exact separate-call Q/K rotary application evidence. | 2 | 6 | — |
| `evidence/stage_operations.py` | 250 | U9-E — positive code-proven operation routes into a repeated stage. | 1 | 7 | — |
| `evidence/unet_attention_source.py` | 791 | U11-F2 — exact runtime external-source routes to U-Net attention lanes. | 0 | 16 | zero-fan-in (evidence/unet_attention_source.py:1); dup-helpers:3 same-named small helpers also defined elsewhere (_forward@73, _construction_span@128, _name@211) |
| `evidence/unet_cell_mechanism.py` | 806 | U11-D2 — exact, occurrence-qualified U-Net child mechanisms. | 2 | 8 | dup-helpers:7 same-named small helpers also defined elsewhere (_target_names@83, _expression_names@92, _span_key@64, _before@71) |
| `evidence/unet_nested_mechanism.py` | 545 | U11-E1 — exact nested mechanism inventory for U-Net cell occurrences. | 1 | 12 | — |
| `evidence/unet_root_preprocess.py` | 516 | U11-F4 — exact root self-helper source preprocessing. | 1 | 10 | dup-helpers:4 same-named small helpers also defined elsewhere (_before@59, _self_method@71, _name@66, _guard_state@273) |
| `evidence/unet_selected_child_execution.py` | 526 | U11-F3d exact execution of selected-stage children. | 3 | 6 | dup-helpers:9 same-named small helpers also defined elsewhere (_self_field@124, _call_key@229, _walk@114, _within@161) |
| `evidence/unet_selected_constructor.py` | 817 | Occurrence-exact constructor environments for selected U-Net children. | 3 | 5 | dead:constructor_guard_evidence@712 |
| `evidence/unet_selected_spatial.py` | 783 | U11-F3d spatial effects of positively executed selected-stage children. | 0 | 7 | zero-fan-in (evidence/unet_selected_spatial.py:1); dup-helpers:8 same-named small helpers also defined elsewhere (_target_names@450, _expression_names@459, _span_key@60, _walk@471) |
| `evidence/unet_selected_stage_children.py` | 559 | U11-F3c neutral child-construction population for selected stages. | 1 | 8 | dup-helpers:4 same-named small helpers also defined elsewhere (_within@58, _source_order@53, _construction_span@107, _guard_state@111) |
| `evidence/unet_stage_cells.py` | 671 | U11-D1 — neutral child-invocation inventory for exact U-Net stages. | 4 | 7 | dup-helpers:4 same-named small helpers also defined elsewhere (_target_names@73, _self_field@66, _before@88, _within@58) |
| `evidence/unet_stage_construction.py` | 833 | U11-B — exact U-shaped repeated-stage construction inventory. | 4 | 7 | dup-helpers:3 same-named small helpers also defined elsewhere (_target_names@90, _self_field@82, _within@73) |
| `evidence/unet_stage_constructor_operands.py` | 308 | U11-F3b exact values entering one selected stage constructor. | 2 | 5 | dup-helpers:3 same-named small helpers also defined elsewhere (_ordinary@34, _actual_map@39, _exact_value_equal@58) |
| `evidence/unet_stage_execution.py` | 388 | U11-C — partial, exact U-shaped stage-execution evidence. | 1 | 7 | — |
| `evidence/unet_stage_operands.py` | 479 | U11-F3 neutral values supplied to one selected stage-factory call. | 1 | 7 | dup-helpers:6 same-named small helpers also defined elsewhere (_span_key@83, _actuals@57, _within@88, _simple_target@76) |
| `evidence/unet_stage_selection.py` | 597 | U11-F1 — occurrence-exact selection of repeated U-Net stages. | 5 | 8 | dup-helpers:3 same-named small helpers also defined elsewhere (_target_names@72, _within@63, _ordinary_params@83) |
| `evidence/weight_tying.py` | 427 | U3-G — exact manual output-head/input-embedding weight tying. | 1 | 8 | — |
| `evidence/wrapper_features.py` | 337 | U9-E2b — positive wrapper feature-selection evidence. | 2 | 6 | dup-helpers:8 same-named small helpers also defined elsewhere (_span_before@295, _target_names@308, _span_key@303, _names@315) |

### consumer (73 files, 16,537 lines)

| path | lines | purpose | fan-in | fan-out | health flags |
|---|---:|---|---:|---:|---|
| `encoder_panel.py` | 139 | The ONE encoder-panel round-trip — adapter-neutral. | 1 | 6 | spine:consumer->adapter parser.py (encoder_panel.py:47,lazy); spine:consumer->adapter submodel.py (encoder_panel.py:125,lazy); lazy-imports:10 function-local in-package imports (first encoder_panel.py:23) |
| `evidence/conformance.py` | 2106 | Op-conformance diff: does the rendered diagram match the HF ``forward()``? | 3 | 17 | god-module:2106L (evidence/conformance.py:1); spine:consumer->legacy ast_scanner.py (evidence/conformance.py:30); spine:consumer->legacy forward_ops.py (evidence/conformance.py:31); spine:consumer->legacy forward_ops.py (evidence/conformance.py:31); spine:consumer->legacy forward_ops.py (evidence/conformance.py:31); spine:consumer->legacy forward_ops.py (evidence/conformance.py:31); spine:consumer->legacy transitive.py (evidence/conformance.py:34); spine:consumer->legacy transitive.py (evidence/conformance.py:34); spine:consumer->legacy transitive.py (evidence/conformance.py:34); spine:consumer->reader projector.py (evidence/conformance.py:166,lazy); spine:consumer->reader projector.py (evidence/conformance.py:166,lazy); spine:consumer->reader component_tower.py (evidence/conformance.py:233,lazy); spine:consumer->reader fusion.py (evidence/conformance.py:356,lazy); spine:consumer->reader multiaxis_position.py (evidence/conformance.py:373,lazy); spine:consumer->reader attention_storage.py (evidence/conformance.py:685,lazy); spine:consumer->reader expert_storage.py (evidence/conformance.py:689,lazy); spine:consumer->reader ffn_mechanism.py (evidence/conformance.py:690,lazy); spine:consumer->reader attention_storage.py (evidence/conformance.py:764,lazy); spine:consumer->reader expert_storage.py (evidence/conformance.py:767,lazy); spine:consumer->reader ffn_mechanism.py (evidence/conformance.py:811,lazy); spine:consumer->reader embedding_bookend.py (evidence/conformance.py:845,lazy); spine:consumer->legacy transitive.py (evidence/conformance.py:1767,lazy); spine:consumer->legacy ast_scanner.py (evidence/conformance.py:561,lazy); spine:consumer->legacy patterns.py (evidence/conformance.py:562,lazy); spine:consumer->legacy forward_ops.py (evidence/conformance.py:563,lazy); spine:consumer->legacy forward_ops.py (evidence/conformance.py:1890,lazy); lazy-imports:23 function-local in-package imports (first evidence/conformance.py:166); import-cycle:SCC#5 (lazy-broken; see cycles table); legacy-parse:ast.parse (evidence/conformance.py:1127); legacy-parse:ast.parse (evidence/conformance.py:1417); legacy-parse:ast.parse (evidence/conformance.py:1377); quarantined:deletion unit U14 (docs/U3_CURRENT_READER_INVENTORY.md) |
| `evidence/validate.py` | 70 | Validation between config-derived IR and static code evidence. | 1 | 2 | — |
| `expanded/__init__.py` | 98 | Structured JSON view of an unfolded model. | 1 | 10 | — |
| `expanded/attention.py` | 185 | Attention spec + operation graph. | 1 | 4 | — |
| `expanded/block_graph.py` | 41 | Block-level DAG: one node per IR block, sequential edges. | 1 | 1 | — |
| `expanded/code_evidence.py` | 63 | Normalise the IR's optional code-evidence section into the JSON view. | 1 | 1 | — |
| `expanded/ffn.py` | 93 | FFN spec + operation graph (dense / gated / MoE). | 1 | 4 | — |
| `expanded/grouping.py` | 60 | Layer grouping by structural signature. | 2 | 1 | — |
| `expanded/layer_group.py` | 46 | Assemble one layer group from its sub-modules. | 1 | 6 | — |
| `expanded/loop.py` | 79 | Project the declared diffusion sampling loop onto the JSON schema. | 1 | 1 | — |
| `expanded/modalities.py` | 74 | Structured multimodal input pathways for expanded JSON. | 1 | 1 | — |
| `expanded/norms.py` | 14 | Norm spec for one layer. | 1 | 0 | — |
| `expanded/ops.py` | 41 | Shared building blocks for the operation graphs. | 3 | 1 | — |
| `expanded/pathways.py` | 29 | External pathways: PLE and similar side-channel constructions. | 1 | 1 | — |
| `expanded/region.py` | 73 | Project a canonical :class:`~..opgraph.Region` onto the JSON node/edge schema. | 2 | 2 | — |
| `expanded/residual.py` | 36 | Residual topology: sequential vs parallel residual, plus the add nodes. | 1 | 1 | — |
| `expanded/sections.py` | 274 | Top-level sections: model identity, dimensions, parameters, io. | 1 | 2 | — |
| `expanded/stack.py` | 59 | The decoder stack viewed three ways. | 1 | 2 | — |
| `expanded/utils.py` | 49 | Tiny shared helpers used across the expanded/ package. | 12 | 0 | — |
| `labels.py` | 976 | Renderer-agnostic vocabulary for talking about transformer specs. | 18 | 2 | — |
| `params.py` | 311 | Rough parameter-count estimation from an IR. | 4 | 1 | — |
| `renderers/__init__.py` | 1 | Rendering backends for model-unfolder diagrams. | 0 | 0 | — |
| `renderers/html/__init__.py` | 5 | HTML/SVG rendering backend. | 1 | 1 | — |
| `renderers/html/block_views/__init__.py` | 25 | Reusable rich block detail views for the HTML renderer. | 3 | 6 | — |
| `renderers/html/block_views/attention.py` | 499 | Attention detail views — projections of the ONE canonical attention region. | 2 | 7 | — |
| `renderers/html/block_views/block_facts.py` | 34 | Block-local semantic facts for detail renderers. | 4 | 0 | — |
| `renderers/html/block_views/declared_ops.py` | 54 | The declared-ops view — ANY card-declared op chain, rendered by the ONE engine. | 1 | 5 | — |
| `renderers/html/block_views/dsa_indexer.py` | 35 | Detail view for DeepSeek-V3.2's DSA "lightning indexer". | 1 | 2 | — |
| `renderers/html/block_views/feed_forward.py` | 47 | The feed-forward view — a projection of the ONE canonical FFN op-graph. | 2 | 5 | — |
| `renderers/html/block_views/mixture_of_experts.py` | 119 | Detail SVGs for mixture-of-experts blocks. | 3 | 6 | — |
| `renderers/html/block_views/modalities.py` | 16 | Compatibility facade for multimodal detail SVG renderers. | 2 | 1 | — |
| `renderers/html/block_views/modality_views/__init__.py` | 15 | Multimodal detail SVG renderers. | 1 | 4 | — |
| `renderers/html/block_views/modality_views/audio.py` | 233 | Audio pathway detail SVGs. | 2 | 6 | — |
| `renderers/html/block_views/modality_views/common.py` | 80 | Shared helpers for multimodal detail SVGs. | 5 | 2 | — |
| `renderers/html/block_views/modality_views/conditioning.py` | 26 | Conditioning-encoder pathway detail SVGs (seq2seq composites). | 1 | 1 | — |
| `renderers/html/block_views/modality_views/fusion_cross_attention.py` | 53 | Cross-attention modality fusion detail SVG. | 1 | 3 | — |
| `renderers/html/block_views/modality_views/fusion_grid.py` | 149 | Qwen-style unified multimodal stream detail SVG. | 1 | 4 | — |
| `renderers/html/block_views/modality_views/fusion_placeholder.py` | 222 | Placeholder-replacement modality fusion detail SVG. | 1 | 8 | — |
| `renderers/html/block_views/modality_views/fusion_prefix.py` | 25 | Source-proven prefix-concatenation modality fusion view. | 1 | 2 | — |
| `renderers/html/block_views/modality_views/video.py` | 33 | Video pathway detail SVGs. | 2 | 3 | — |
| `renderers/html/block_views/modality_views/vision.py` | 74 | Vision pathway detail SVG. | 2 | 3 | — |
| `renderers/html/block_views/modality_views/vision_details.py` | 83 | Vision encoder tower — THE one tower projection. | 2 | 4 | — |
| `renderers/html/block_views/moe_router.py` | 164 | Detail view for the MoE router — the gate that turns a token into expert weights. | 1 | 4 | — |
| `renderers/html/block_views/mtp_head.py` | 130 | Detail SVG for the Multi-Token Prediction (MTP) head stack. | 1 | 6 | import-cycle:SCC#3 (lazy-broken; see cycles table) |
| `renderers/html/block_views/per_layer_embedding.py` | 185 | Detail SVG for reusable Per-Layer Embedding blocks. | 2 | 4 | import-cycle:SCC#3 (lazy-broken; see cycles table) |
| `renderers/html/block_views/registry.py` | 230 | Central recursive router for block detail views. | 2 | 21 | import-cycle:SCC#3 (lazy-broken; see cycles table) |
| `renderers/html/block_views/scheduler_step.py` | 43 | Detail view for one scheduler/sampler step (z_t → z_{t-1}). | 1 | 2 | — |
| `renderers/html/block_views/self_conditioning.py` | 46 | Detail view for DiffusionGemma's self-conditioning block. | 1 | 2 | — |
| `renderers/html/block_views/text_encoder.py` | 155 | Detail view for a diffusion text encoder (CLIP / T5 / …). | 1 | 3 | — |
| `renderers/html/block_views/unet.py` | 547 | Detail SVG for a UNet diffusion denoiser (UNet2DConditionModel). | 1 | 6 | import-cycle:SCC#3 (lazy-broken; see cycles table) |
| `renderers/html/block_views/vae.py` | 111 | Detail SVG for the diffusion VAE decoder (AutoencoderKL). | 1 | 3 | import-cycle:SCC#3 (lazy-broken; see cycles table) |
| `renderers/html/cards.py` | 239 | Inspect-card HTML for architecture block clicks. | 2 | 2 | — |
| `renderers/html/document.py` | 241 | Top-level HTML document and fragment rendering. | 1 | 10 | — |
| `renderers/html/evidence.py` | 114 | Render static code-evidence summaries. | 2 | 1 | — |
| `renderers/html/graph.py` | 325 | A declarative node-graph for block diagrams — the single source of shape. | 10 | 0 | — |
| `renderers/html/graph_engine.py` | 671 | Lay out a :class:`~.graph.Graph` to SVG — the one engine every view uses. | 17 | 5 | — |
| `renderers/html/interactions.py` | 130 | Inline browser behavior for the HTML renderer. | 2 | 0 | — |
| `renderers/html/metadata.py` | 280 | Layer grouping, per-block tooltip metadata, and architecture badges. | 7 | 4 | import-cycle:SCC#3 (lazy-broken; see cycles table) |
| `renderers/html/metadata_modalities.py` | 1400 | Multimodal block metadata and detail-card children. | 1 | 4 | spine:consumer->adapter submodel.py (renderers/html/metadata_modalities.py:472,lazy); spine:consumer->adapter submodel.py (renderers/html/metadata_modalities.py:867,lazy); spine:consumer->adapter submodel.py (renderers/html/metadata_modalities.py:703,lazy); import-cycle:SCC#3 (lazy-broken; see cycles table) |
| `renderers/html/op_render.py` | 287 | Project a canonical :class:`~...opgraph.Region` onto a render :class:`~.graph.Graph`. | 4 | 3 | — |
| `renderers/html/patch_grid.py` | 92 | Canonical patch-grid facts -> display strings. | 1 | 1 | dead:grid_title@12,grid_subtitle@28 |
| `renderers/html/render_context.py` | 160 | Call-local rendering state: theme, diagnostics and provenance events. | 7 | 1 | dead:ensure_render_context@143,release_render_context@151 |
| `renderers/html/sections.py` | 133 | Reusable HTML sections and header fragments. | 2 | 2 | — |
| `renderers/html/stack_view.py` | 201 | Self-sizing vertical-stack canvas for block drill-down views. | 11 | 2 | — |
| `renderers/html/styles.py` | 418 | Scoped CSS for rendered HTML fragments. | 2 | 2 | god-function:_style=396L (renderers/html/styles.py:8) |
| `renderers/html/svg.py` | 632 | Low-level SVG primitives and routing helpers. | 12 | 3 | — |
| `renderers/html/theme.py` | 111 | Shared visual constants for the HTML/SVG renderer. | 15 | 1 | — |
| `renderers/html/tower.py` | 273 | ONE backbone for every transformer tower. | 7 | 3 | — |
| `renderers/html/utils.py` | 36 | Escaping and formatting helpers for renderer modules. | 12 | 0 | — |
| `renderers/html/views.py` | 1432 | Top-level SVG views for architecture and layer maps. | 2 | 9 | god-function:_build_architecture_view=600L (renderers/html/views.py:112) |
| `renderers/html/views_diffusion.py` | 860 | Diffusion rendering: the sampling loop is the hero; the DiT denoiser drills out of it. | 1 | 12 | — |
| `renderers/html/views_modalities.py` | 147 | Top-level multimodal architecture scaffolds. | 1 | 3 | — |

### gate (8 files, 5,445 lines)

| path | lines | purpose | fan-in | fan-out | health flags |
|---|---:|---|---:|---:|---|
| `evidence/claims_audit.py` | 188 | Migration-claim validation and corpus coverage — ONE shared implementation. | 1 | 0 | — |
| `evidence/consumer_firewall.py` | 746 | Static dependency and structural-read firewall for architecture consumers. | 1 | 0 | lawful-parse:ast.parse (evidence/consumer_firewall.py:516); dup-helpers:3 same-named small helpers also defined elsewhere (_call_name@152, _config_path@228, _import_target@457) |
| `evidence/identity_guard.py` | 926 | Report identity-derived architectural decisions before they can hide. | 1 | 4 | import-cycle:SCC#6 (lazy-broken; see cycles table); lawful-parse:ast.parse (evidence/identity_guard.py:282); dead:violation_snapshot@916 |
| `evidence/legacy_reader_quarantine.py` | 581 | Blocking inventory of surviving pre-ProgramIndex semantic readers. | 0 | 0 | lawful-parse:ast.parse (evidence/legacy_reader_quarantine.py:395); lawful-parse:ast.parse (evidence/legacy_reader_quarantine.py:417); lawful-parse:ast.parse (evidence/legacy_reader_quarantine.py:469); lawful-parse:ast.parse (evidence/legacy_reader_quarantine.py:508); zero-fan-in (evidence/legacy_reader_quarantine.py:1); dup-helpers:4 same-named small helpers also defined elsewhere (_scope@199, _scope@313, _remember_alias@230, _remember_alias@353) |
| `evidence/receipts.py` | 437 | U2-R5 projection receipts — the render half of the fact/projection contract. | 9 | 1 | import-cycle:SCC#1 (lazy-broken; see cycles table); dead:is_receipted_scope@187 |
| `evidence/structural_debt.py` | 1273 | U2-R6 — the ONE structural-debt register (§R6). | 2 | 3 | lawful-parse:ast.parse (evidence/structural_debt.py:1047); dead:_drawn@183 |
| `evidence/structural_writes.py` | 1097 | H2 Part B — the StructuralWrite census across every author surface (§16.4). | 2 | 1 | lawful-parse:ast.parse (evidence/structural_writes.py:281) |
| `renderers/html/fact_projection.py` | 197 | U2 P4 — the fact-projection witness channel (projection-audit net #13). | 6 | 0 | — |

### legacy (5 files, 3,082 lines)

| path | lines | purpose | fan-in | fan-out | health flags |
|---|---:|---|---:|---:|---|
| `evidence/ast_scanner.py` | 106 | AST scanner for modeling source files. | 5 | 2 | legacy-parse:ast.parse (evidence/ast_scanner.py:23); quarantined:deletion unit U14 (docs/U3_CURRENT_READER_INVENTORY.md) |
| `evidence/forward_ops.py` | 629 | Extract the coarse op-kind PRESENCE-SET a class's ``forward()`` performs. | 5 | 3 | legacy-parse:ast.parse (evidence/forward_ops.py:62); legacy-parse:ast.parse (evidence/forward_ops.py:600); quarantined:deletion unit U14 (docs/U3_CURRENT_READER_INVENTORY.md); dup-helpers:3 same-named small helpers also defined elsewhere (_self_field@426, _mtime@52, _parse_file@60) |
| `evidence/patterns.py` | 766 | Family-agnostic structural pattern inference from static code evidence. | 3 | 5 | lazy-imports:13 function-local in-package imports (first evidence/patterns.py:325); import-cycle:SCC#5 (lazy-broken; see cycles table); legacy-parse:ast.parse (evidence/patterns.py:416); quarantined:deletion unit U11 (docs/U3_CURRENT_READER_INVENTORY.md); dead:_ConfigExprEvaluator@689 |
| `evidence/transitive.py` | 611 | Transitive op-set closure for the RECURSIVE drill-conformance net. | 3 | 2 | legacy-parse:ast.parse (evidence/transitive.py:215); quarantined:deletion unit U14 (docs/U3_CURRENT_READER_INVENTORY.md) |
| `evidence/vision.py` | 970 | Component-qualified vision tower evidence from the real HF source. | 1 | 8 | spine:legacy->consumer conformance.py (evidence/vision.py:35,lazy); import-cycle:SCC#5 (lazy-broken; see cycles table); legacy-parse:ast.parse (evidence/vision.py:960); quarantined:deletion unit U14 (docs/U3_CURRENT_READER_INVENTORY.md); dead:_vision_owner@174,_model_position_kind@556,_vision_mechanism_root@616,_vision_attention_kind@627,_configured_block_instances@913 |

### vocab (19 files, 1,773 lines)

| path | lines | purpose | fan-in | fan-out | health flags |
|---|---:|---|---:|---:|---|
| `everchanging/__init__.py` | 455 | User-editable config vocabulary — *data, not code*. | 16 | 0 | — |
| `everchanging/conformance/abstractions.yaml` | 54 | Deliberate-abstraction allow-list for the op-conformance diff. | 1 | 0 | — |
| `everchanging/conformance/conformance_map.yaml` | 18 | View <-> backing HF class.method — GENUINE EXCEPTIONS ONLY. | 1 | 0 | — |
| `everchanging/conformance/fact_markers.yaml` | 41 | Vocabulary for FACT-conformance (evidence/conformance.py::check_fact_conformance) | 1 | 0 | — |
| `everchanging/conformance/op_tokens.yaml` | 51 | Code-token (the call name in a forward() body) -> canonical op-kind. | 1 | 0 | — |
| `everchanging/conformance/transitive.yaml` | 188 | Transitive-delegation vocabulary for the RECURSIVE drill-conformance net | 1 | 0 | — |
| `everchanging/conformance/type_roles.yaml` | 29 | Constructed-class-name SUBSTRING -> the canonical op-role of a `self.<field>(…)` | 1 | 0 | — |
| `everchanging/conformance/wiring_roles.yaml` | 21 | Wiring-conformance vocabulary: a drawn SIDE-INPUT (conditioning rail) must | 1 | 0 | — |
| `everchanging/diffusor/aliases.yaml` | 91 | Diffusion (DiT / MMDiT) field aliases — canonical field name -> the config key | 1 | 0 | — |
| `everchanging/diffusor/conditioning.yaml` | 68 | U11 compatibility-only UNet conditioning display vocabulary. | 1 | 0 | — |
| `everchanging/diffusor/text_encoders.yaml` | 14 | Diffusers text-encoder class name -> friendly family label. | 1 | 0 | — |
| `everchanging/diffusor/typing.yaml` | 125 | Diffusion block typing — the APPROVED static type for diffusion diagrams. | 1 | 0 | — |
| `everchanging/evidence/ledger_ignores.yaml` | 97 | Ledger-side scoped-ignore vocabulary (U2-R7). | 1 | 0 | — |
| `everchanging/evidence/program_index_vocab.yaml` | 27 | Syntactic vocabulary for the U3 ProgramIndex walker (evidence/program_index.py). | 1 | 0 | — |
| `everchanging/transformer/aliases.yaml` | 75 | Field aliases — canonical field name -> the config key spellings the parser | 1 | 0 | — |
| `everchanging/transformer/composite_slots.yaml` | 92 | Declared component SLOTS of composite / seq2seq wrapper configs. | 1 | 0 | — |
| `everchanging/transformer/decoderness.yaml` | 27 | Config-declared DECODER-NESS evidence (U2 mask default-kill). | 1 | 0 | — |
| `everchanging/transformer/ignored_fields.yaml` | 232 | Config fields that are NOT architectural — the debug diagnostic skips these | 1 | 0 | — |
| `everchanging/transformer/typing.yaml` | 68 | Transformer block typing — the APPROVED static type for transformer diagrams. | 1 | 0 | — |

## 3. Dependency-direction check (the one-way spine)

The intended spine is `vocab → substrate → reader → adapter → consumer → product`,
with gates allowed to read anything. Checking every import edge against that
(edge matrix in `scratchpad/r01/` and the JSON) gives **54 distinct import
statements that cross the spine backwards** (75 imported names). Grouped:

**Q1 — does any consumer import adapters or evidence readers?**

* No *renderer* or *expanded/* module imports an evidence **reader**. The only
  consumer→reader edges are inside `evidence/conformance.py` itself (11 lazy
  imports: `conformance.py:166,233,356,373,685,689,690,764,767,811,845`),
  which is by design — the conformance net compares readers against the drawing.
* Renderers **do** reach back into the adapter layer for the recursive
  sub-model projection: `renderers/html/metadata_modalities.py:472,703,867` →
  `submodel.py` (lazy), and `encoder_panel.py:47` → `adapters/transformer/parser.py`,
  `encoder_panel.py:125` → `submodel.py` (lazy). `submodel.py` is the seam where
  adapter and renderer meet (it builds render blocks *and* re-parses encoder
  configs), so these are the "adapter-in-renderer" wrong-way edges.
* `block_schema.py:378` (substrate DTO) imports
  `renderers/html/block_views/registry.py` to learn the legal view names — the
  schema depends on the renderer's registry instead of the reverse.

**Q2 — does any reader import a renderer?** **No.** Zero reader→consumer and
zero reader→adapter edges (208 reader→reader, 563 reader→substrate, 2 reader→vocab).
The reader layer is clean in this respect.

**Q3 — adapters importing presentation.** 17 import statements in the adapter
layer pull `labels.py` (renderer-agnostic *display strings*) or a renderer
module into the config→IR path — the "presentation in IR" wrong:
`adapters/transformer/blocks/layers.py:8`, `blocks/attention.py:7`,
`blocks/feed_forward.py:7,491`, `blocks/__init__.py:32`,
`special_parts/per_layer_embedding.py:10`, `adapters/diffusor/blocks.py:19`,
`adapters/diffusor/unet.py:757`, `submodel.py:301,368,408,513`; and
`blocks/feed_forward.py:493` imports `renderers/html/block_views/mixture_of_experts._EXPERT_IDS`
(an adapter reading a private renderer constant). `adapters/diffusor/parser.py:56,112`
imports `evidence.conformance._augment_diffusion_files` (a private helper of a
consumer) and `adapters/diffusor/parser.py:1034` imports `encoder_panel.py`.

**Q4 — substrate reaching up into readers.** `evidence/config_guard.py:13`,
`evidence/expression_value.py:14`, `evidence/layer_selector.py:20` all import
`exact_config_path_for_expression` from `evidence/attention.py` — a substrate
utility that lives inside the 5,324-line attention reader; and
`evidence/invocation_source.py:15` imports `local_lineage_at_callable` from the
`diffusion_stream.py` reader. `evidence/context.py:257,392` lazily imports the
`decoderness` reader; `evidence/config_access.py:960` lazily imports the
`receipts` gate. These four misfiled helpers are the reason a "substrate-only"
import of `config_guard` transitively loads all of `attention.py`.

**Q5 — legacy reaching into consumers.** `evidence/vision.py:35` imports
`conformance._direct_role_classes` (lazy) — the quarantined vision reader
depends on the conformance net's private helper, and `conformance.py:561-563`
depends back on `ast_scanner`, `patterns`, `forward_ops`. That is cycle #5 below.

**Import cycles.** Over *all* edges there are 6 strongly-connected components
(35 modules); over *eager-only* edges there are **none** — every cycle is
broken by function-local imports. There are **379 function-local in-package
imports** in total; `adapters/transformer/parser.py` alone has 67
(first at `adapters/transformer/parser.py:95`; the rest are inside `parse()`) and `adapters/diffusor/parser.py` 37,
`sable.py` 37, `evidence/conformance.py` 23, `parser.py` 19, `submodel.py` 16.

| # | size | members (short paths) | what binds it |
|---|---:|---|---|
| 1 | 6 | `evidence/config_access, context, document, facts, receipts, registry` | the fact ledger ↔ config-access ↔ receipts contract (`config_access.py:960`, `context.py:257`, `receipts.py` ↔ `registry.py`) |
| 2 | 9 | `adapters/transformer/common, debug` + `special_parts/modalities/*` | `common.py`↔`debug.py` and the modality registry reaching `debug` |
| 3 | 11 | `adapters/transformer/blocks/attention, feed_forward`, `block_schema`, `renderers/html/block_views/{mtp_head, per_layer_embedding, registry, unet, vae}`, `renderers/html/metadata, metadata_modalities`, `submodel` | adapter blocks ↔ renderer registry ↔ submodel (Q1/Q3 above) |
| 4 | 2 | `evidence/attention_storage ↔ dispatch_attention_storage` | mutual lazy import (`attention_storage.py` lazy → dispatch) |
| 5 | 3 | `evidence/conformance ↔ patterns ↔ vision` | legacy parsers and the conformance net share private helpers |
| 6 | 4 | `adapters/diffusor/loader, evidence/__init__, evidence/identity_guard, parser` | `parser.py:449` lazily imports the evidence package; `identity_guard` lazily imports `parser` |

## 4. God functions (> 300 lines) and god modules (> 1,500 lines)

| function | lines | site |
|---|---:|---|
| `parse` | 3,762 | `adapters/transformer/parser.py:876` — the entire transformer config→IR projection in one body; 67 function-local imports; section comments (`# ---- U8-C …`, `# ---- Attention shape ----`) stand in for functions |
| `_build_architecture_view` | 600 | `renderers/html/views.py:112` |
| `_style` | 396 | `renderers/html/styles.py:8` (a CSS string; benign) |
| `config_to_ir` | 376 | `parser.py:23` — loading/retry/hydration/evidence attachment interleaved |
| `_sdpa_detailed_child_blocks` | 325 | `adapters/transformer/blocks/attention.py:223` |
| `sable` | 301 | `sable.py:360` — the harness driver; every net inline |

| module | lines | category | note |
|---|---:|---|---|
| `evidence/attention.py` | 5,324 | reader | 121 functions, 11 dataclasses; hosts the substrate helper `exact_config_path_for_expression` used by 3 substrate modules (§3 Q4) and 4 same-named small helpers duplicated elsewhere |
| `adapters/transformer/parser.py` | 4,996 | adapter | 75 % of it is `parse()` |
| `evidence/program_index.py` | 2,929 | substrate | 41 dataclasses + `ProgramIndex` (`:2483`) + the walker; fan-in 122 — the true centre of the package |
| `evidence/component_owner.py` | 2,155 | substrate | owner graph + declared model-stage resolution; fan-in 98 |
| `evidence/conformance.py` | 2,106 | consumer | 4 legacy `ast.parse` sites (`:1127,1377,1417` + `_constructor_envs`); imports 17 modules, 23 lazily |
| `evidence/cell_topology.py` | 1,972 | reader | |
| `evidence/attention_mask.py` | 1,930 | reader | 11 dataclasses; longest fn 270 lines |
| `evidence/router.py` | 1,920 | reader | |
| `evidence/ffn_mechanism.py` | 1,639 | reader | |
| `evidence/expert_storage.py` | 1,556 | reader | |
| `evidence/config_access.py` | 1,546 | substrate | the H3 config-access event ledger; part of cycle #1 |
| `evidence/registry.py` | 1,502 | substrate | the closed fact registry — mostly data rows (15 functions in 1,502 lines) |

## 5. Dead-code candidates

Definition: top-level function/class with zero in-package references (names,
attributes, or imports), not in the public `__all__`
(`model_unfolder/__init__.py:26`). Word-grep over every `.py` in `tests/`,
`scripts/`, `test_support/`, `examples/` then splits them.

**5a. No caller anywhere in code (26).** Deletable unless a docs plan claims them;
`docs/U3_READER_INVENTORY.md` mentions the `vision.py` ones as historical.

| site | name | kind | lines | outside `.py` hits |
|---|---|---|---:|---|
| `adapters/transformer/debug.py:97` | `_value_at` | function | 11 | — |
| `adapters/transformer/parser.py:86` | `_source_files` | function | 11 | — |
| `adapters/transformer/parser.py:703` | `_code_expert_storage` | function | 7 | — |
| `adapters/transformer/parser.py:808` | `_unwrap_text` | function | 3 | — |
| `adapters/transformer/parser.py:4865` | `_norm_kind_evidence` | function | 23 | — |
| `evidence/attention.py:2783` | `_latest_unconditional_binding` | function | 13 | — |
| `evidence/attention_child.py:532` | `attention_compute_proof_for_symbol` | function | 17 | — |
| `evidence/decoder_norm.py:276` | `norm_invocations_in_graph` | function | 24 | — |
| `evidence/execution_flow.py:1218` | `_span_sort` | function | 2 | — |
| `evidence/expert_storage.py:1132` | `_field_referenced_in_spans` | function | 8 | — |
| `evidence/identity_guard.py:916` | `violation_snapshot` | function | 3 | — |
| `evidence/patterns.py:689` | `_ConfigExprEvaluator` | class | 75 | — |
| `evidence/position_application.py:1297` | `_occurrence_key` | function | 3 | — |
| `evidence/projection_bias.py:450` | `_merge_result` | function | 10 | — |
| `evidence/receipts.py:187` | `is_receipted_scope` | function | 4 | — |
| `evidence/structural_debt.py:183` | `_drawn` | function | 7 | — |
| `evidence/unet_selected_constructor.py:712` | `constructor_guard_evidence` | function | 22 | — |
| `evidence/vision.py:174` | `_vision_owner` | function | 32 | — |
| `evidence/vision.py:556` | `_model_position_kind` | function | 16 | — |
| `evidence/vision.py:616` | `_vision_mechanism_root` | function | 9 | — |
| `evidence/vision.py:627` | `_vision_attention_kind` | function | 32 | — |
| `evidence/vision.py:913` | `_configured_block_instances` | function | 42 | — |
| `renderers/html/patch_grid.py:12` | `grid_title` | function | 14 | — |
| `renderers/html/patch_grid.py:28` | `grid_subtitle` | function | 15 | — |
| `renderers/html/render_context.py:143` | `ensure_render_context` | function | 6 | — |
| `renderers/html/render_context.py:151` | `release_render_context` | function | 4 | — |

Notes: `adapters/transformer/parser.py:703 _code_expert_storage`, `:808 _unwrap_text`,
`:4865 _norm_kind_evidence` and `:86 _source_files` are pre-ProgramIndex helpers
the 3,762-line `parse()` no longer calls. `evidence/vision.py:174,556,616,627,913`
are the deleted "U9 vision authority" left in place. `renderers/html/render_context.py:143,151`
(`ensure_/release_render_context`) and `patch_grid.py:12,28` are renderer entry
points nothing renders through.

**5b. Reachable only from tests (54).** These are *built but unwired* —
the more important finding, because several are whole units:

| site | name | kind | lines | first test hits |
|---|---|---|---:|---|
| `adapters/diffusor/parser.py:460` | `_shadow_diffusion_source_projection` | function | 21 | tests/test_diffusion_schema.py |
| `adapters/transformer/assembly.py:87` | `single_stream_decoder_layer` | function | 33 | tests/test_u4_d_cell_topology.py |
| `adapters/transformer/debug.py:83` | `bound_fields` | function | 6 | tests/test_config_intents.py |
| `adapters/transformer/debug.py:91` | `consumed_fields` | function | 4 | tests/test_config_intents.py |
| `encoder_panel.py:12` | `hydrate_encoder_config_facts` | function | 13 | tests/test_config_paths.py |
| `evidence/arbitration.py:178` | `arbitrate` | function | 50 | tests/test_arbitration.py, tests/test_attention_mechanism.py |
| `evidence/arbitration.py:230` | `verdict_to_fact` | function | 43 | tests/test_arbitration.py |
| `evidence/attention.py:917` | `decoder_attention_head_binding_for_path` | function | 27 | tests/test_attention_mechanism.py |
| `evidence/attention_mask.py:1277` | `decoder_attention_mask_geometry_for_path` | function | 36 | tests/test_attention_mask.py |
| `evidence/attention_storage.py:172` | `decoder_attention_projection_storage_mode_evidence` | function | 31 | tests/test_attention_storage.py |
| `evidence/claims_audit.py:149` | `audit_claim_coverage` | function | 37 | tests/test_projection_audit.py |
| `evidence/config_access.py:1059` | `resolve_priority` | function | 52 | tests/test_config_paths.py |
| `evidence/config_access.py:1373` | `present_spelling` | function | 16 | tests/test_config_paths.py |
| `evidence/decoder_norm.py:302` | `norm_preserving_invocations_in_graph` | function | 10 | tests/test_decoder_norm.py |
| `evidence/dispatch_selection.py:173` | `resolve_dispatch_construction` | function | 64 | tests/test_dispatch_selection.py |
| `evidence/document.py:370` | `checkpoint_provenance` | function | 11 | tests/test_diffusion_companion.py |
| `evidence/facts.py:317` | `migrated_legacy_debt` | function | 7 | tests/test_evidence_facts.py |
| `evidence/forward_ops.py:551` | `unclassified_call_tokens` | function | 79 | tests/test_loud_miss.py |
| `evidence/identity_guard.py:459` | `scan_declared_class_vocabulary` | function | 5 | tests/test_identity_guard.py |
| `evidence/identity_guard.py:466` | `scan_display_vocabulary` | function | 6 | tests/test_identity_guard.py |
| `evidence/identity_guard.py:501` | `scan_fact_provenance_identity` | function | 23 | tests/test_h4_taint.py |
| `evidence/identity_roles.py:49` | `identity_display` | function | 7 | tests/test_authority_probes.py, tests/test_h4_taint.py |
| `evidence/legacy_reader_quarantine.py:517` | `legacy_reader_quarantine_problems` | function | 47 | scripts/generate_u3_reader_inventory.py, tests/test_legacy_reader_quarantine.py |
| `evidence/patterns.py:412` | `_parse_defs` | function | 11 | tests/test_legacy_reader_quarantine.py |
| `evidence/position_initialization.py:202` | `decoder_position_frequency_initialization_for_path` | function | 19 | tests/test_position_initialization.py |
| `evidence/program_index.py:2813` | `clear_program_index_source_cache` | function | 3 | tests/test_program_index.py |
| `evidence/projector_chain.py:109` | `read_projector_operation_chain` | function | 64 | tests/test_projector_chain.py |
| `evidence/reader_result.py:229` | `ambiguity_from_conflicts` | function | 27 | tests/test_reader_result.py |
| `evidence/registry.py:1317` | `census_problems` | function | 28 | tests/test_fact_registry.py |
| `evidence/registry.py:1483` | `new_raw_structural_extras` | function | 13 | tests/test_fact_registry.py |
| `evidence/selected_composite_ffn.py:251` | `selected_composite_ffn_mechanism` | function | 31 | tests/test_selected_composite_ffn.py, tests/test_unet_nested_mechanism.py |
| `evidence/structural_debt.py:999` | `drawn_leaf_is_lawful` | function | 12 | tests/test_projection_obligations.py, tests/test_u2_r9_final_corrections.py |
| `evidence/structural_debt.py:1013` | `drawn_unledgered_names` | function | 7 | tests/test_h8_transformer.py, tests/test_projection_obligations.py |
| `evidence/structural_debt.py:1022` | `debt_keys` | function | 5 | tests/test_structural_debt.py |
| `evidence/structural_debt.py:1029` | `debt_targets` | function | 3 | tests/test_fact_registry.py, tests/test_structural_writes.py |
| `evidence/structural_writes.py:518` | `new_structural_writes` | function | 7 | tests/test_structural_writes.py |
| `evidence/structural_writes.py:527` | `stale_surface_entries` | function | 5 | tests/test_structural_writes.py |
| `evidence/structural_writes.py:537` | `runtime_structural_targets` | function | 15 | tests/test_structural_writes.py |
| `evidence/structural_writes.py:574` | `runtime_structural_surface` | function | 38 | tests/test_u2_r4_structural_multiset.py |
| `evidence/unet_attention_source.py:722` | `read_unet_runtime_attention_sources` | function | 60 | tests/test_unet_nested_mechanism.py |
| `evidence/unet_cell_mechanism.py:767` | `read_unet_cell_mechanisms` | function | 25 | tests/test_unet_cell_mechanism.py, tests/test_unet_nested_mechanism.py |
| `evidence/unet_nested_mechanism.py:498` | `read_unet_nested_mechanisms` | function | 34 | tests/test_unet_nested_mechanism.py |
| `evidence/unet_root_preprocess.py:475` | `read_unet_root_preprocessing` | function | 32 | tests/test_unet_nested_mechanism.py, tests/test_unet_root_preprocess.py |
| `evidence/unet_selected_child_execution.py:486` | `read_unet_selected_child_execution` | function | 34 | tests/test_unet_nested_mechanism.py, tests/test_unet_selected_child_execution.py |
| `evidence/unet_selected_spatial.py:740` | `read_unet_selected_spatial_operations` | function | 37 | tests/test_unet_selected_child_execution.py |
| `evidence/unet_selected_stage_children.py:522` | `read_unet_selected_stage_children` | function | 31 | tests/test_unet_nested_mechanism.py, tests/test_unet_selected_child_execution.py |
| `evidence/unet_stage_cells.py:536` | `read_unet_stage_cells` | function | 124 | tests/test_unet_cell_mechanism.py, tests/test_unet_nested_mechanism.py |
| `evidence/unet_stage_construction.py:660` | `read_unet_stage_construction` | function | 79 | tests/test_unet_cell_mechanism.py, tests/test_unet_nested_mechanism.py |
| `evidence/unet_stage_constructor_operands.py:270` | `read_unet_selected_stage_constructor_operands` | function | 30 | tests/test_unet_nested_mechanism.py, tests/test_unet_selected_child_execution.py |
| `evidence/unet_stage_execution.py:261` | `read_unet_stage_execution` | function | 116 | tests/test_unet_cell_mechanism.py, tests/test_unet_nested_mechanism.py |
| `evidence/unet_stage_operands.py:428` | `read_unet_selected_stage_operands` | function | 46 | tests/test_unet_nested_mechanism.py, tests/test_unet_selected_child_execution.py |
| `evidence/unet_stage_selection.py:500` | `read_unet_stage_selection` | function | 90 | tests/test_unet_nested_mechanism.py, tests/test_unet_root_preprocess.py |
| `evidence/vision.py:18` | `layer_facts_from_block` | function | 83 | tests/test_code_evidence.py |
| `parser.py:781` | `_hydrate_config_class_defaults` | function | 29 | tests/test_code_evidence.py |

Three clusters stand out:

1. **All fourteen U11 `evidence/unet_*` readers** (8,619 lines across
   14 modules; entry points `read_unet_stage_selection`
   (`unet_stage_selection.py:500`), `read_unet_stage_construction`
   (`unet_stage_construction.py:660`), `read_unet_cell_mechanisms`
   (`unet_cell_mechanism.py:767`), …) have **no production caller**. A grep for
   `read_unet_` outside `evidence/unet_*` returns nothing. Meanwhile
   `adapters/diffusor/parser.py:56-122` still calls the five quarantined
   `patterns.py` `*_from_files` readers. The quarantine register
   (`legacy_reader_quarantine.py:97-125`) says U11 "registers the exact
   owner-qualified fact and deletes" them — the readers exist, the deletion has
   not happened, and the diagram is still drawn from the legacy path.
2. **`evidence/arbitration.py`** (278 lines; `arbitrate` `:178`, `verdict_to_fact` `:230`)
   has zero fan-in — the evidence-ranking substrate is not used by any parser.
3. **`evidence/selected_composite_ffn.py`** (643 lines, `:251`) and
   `evidence/dispatch_selection.resolve_dispatch_construction` (`:173`) are
   test-only.

The gate-side "test-only" rows (`structural_debt.py:999-1029`,
`structural_writes.py:518-574`, `identity_guard.py:459-501`, `registry.py:1317,1483`,
`claims_audit.py:149`) are expected: gates are exercised by the test suite, not by
`unfold()`.

## 6. Duplicate helpers across `evidence/*`

129 private helper names are defined in two or more evidence modules
(functions ≤ 60 lines). Top entries, with how many *distinct bodies* the copies
have (identical AST body ⇒ pure copy-paste; distinct ⇒ drifted copies of the
same idea):

| helper | defs | distinct bodies | first sites |
|---|---:|---:|---|
| `_self_field` | 47 | 26 | attention.py:5059, attention_child.py:1111, attention_container_interface.py:395, attention_lane.py:438, attention_output.py:508 … |
| `_span_key` | 42 | 14 | attention_container_interface.py:432, attention_geometry.py:846, attention_input_interface.py:507, attention_lane.py:678, attention_mask.py:1895 … |
| `_target_names` | 30 | 19 | attention.py:5033, attention_child.py:799, attention_input_interface.py:489, attention_mask.py:1825, attention_output.py:481 … |
| `_span_before` | 26 | 13 | attention.py:4005, attention_child.py:842, attention_container_interface.py:425, attention_geometry.py:720, attention_input_interface.py:500 … |
| `_failed` | 12 | 7 | attention_container_interface.py:437, attention_input_interface.py:512, attention_invocation_role.py:490, attention_mask.py:1872, config_scoped_owner.py:935 … |
| `_within` | 12 | 6 | diffusion_root.py:50, diffusion_stack.py:68, execution_flow.py:668, output_repeated_stage.py:500, selected_composite_ffn.py:624 … |
| `_before` | 11 | 5 | component_operations.py:604, component_position.py:374, diffusion_bookends.py:63, diffusion_stream.py:53, multiaxis_position.py:285 … |
| `_expression_names` | 10 | 10 | attention_child.py:814, attention_mask.py:1856, attention_output.py:455, constructor_condition.py:349, diffusion_bookends.py:86 … |
| `_span_within` | 10 | 7 | cell_topology.py:1936, config_guard.py:183, container_inventory.py:69, expert_storage.py:1518, ffn_mechanism.py:1602 … |
| `_expr_contains_span` | 9 | 6 | attention.py:3955, attention_geometry.py:711, attention_mask.py:1809, attention_output.py:468, attention_score_additives.py:504 … |
| `_span_sort_key` | 6 | 5 | attention.py:4012, attention_child.py:1119, attention_storage.py:1147, embedding_bookend.py:229, final_bookend.py:417 … |
| `_dependency_closure` | 6 | 4 | attention.py:5011, dispatch_attention_mechanism.py:672, expert_width.py:537, ffn_mechanism.py:1465, router.py:1843 … |
| `_attribute_chain` | 6 | 4 | attention_geometry.py:600, attention_mask.py:1834, attention_sinks.py:356, component_owner.py:1372, framework_config.py:1273 … |
| `_names` | 6 | 5 | attention_input_interface.py:480, constructor_fields.py:397, diffusion_root.py:95, expert_storage.py:1486, fusion.py:605 … |
| `_forward` | 5 | 5 | delegated_stage.py:128, diffusion_root.py:254, diffusion_stream.py:914, fusion.py:557, unet_attention_source.py:73 |
| `_walk` | 5 | 2 | diffusion_bookends.py:108, diffusion_conditioning.py:41, expert_width.py:557, unet_selected_child_execution.py:114, unet_selected_spatial.py:471 |
| `_forward_failure` | 5 | 3 | position_application.py:1302, position_factors.py:801, position_geometry.py:700, position_schedule.py:497, qk_norm.py:830 |
| `_expr_contains_name` | 4 | 4 | attention.py:3965, attention_geometry.py:702, position_fixed.py:487, position_linear_bias.py:657 |
| `_exact_call_target` | 4 | 4 | attention_child.py:1076, attention_output.py:369, attention_sinks.py:388, dispatch_attention_storage.py:415 |
| `_plain_name` | 4 | 4 | attention_geometry.py:595, component_owner.py:1210, kv_sharing_schedule.py:394, layer_selector.py:922 |
| `_self_method` | 4 | 4 | component_position.py:258, projector_lineage.py:838, rope_config_normalization.py:220, unet_root_preprocess.py:71 |
| `_guard_prefix` | 4 | 3 | diffusion_stream.py:60, projector_chain.py:735, projector_lineage.py:967, wrapper_features.py:291 |

Specifically named in the brief:

* `_parse_file` — 2 defs, 2 bodies: `evidence/forward_ops.py:60` and
  `evidence/transitive.py:213`, each with its own identical `_mtime` cache key
  (`forward_ops.py:52` ≡ `transitive.py:205`). Both are registered legacy parse
  sites (deletion unit U14).
* `_class_node` — 1 def (`evidence/vision.py:966`), itself legacy.
* `_forward` — 5 defs, 5 bodies (`delegated_stage.py:128`, `diffusion_root.py:254`,
  `diffusion_stream.py:914`, `fusion.py:557`, `unet_attention_source.py:73`):
  five spellings of "find the `forward` callable of this symbol in the index".
* `_self_field` — **47 defs, 26 distinct bodies**, including a method on
  `ProgramIndex` itself (`program_index.py:1799`). "Is this expression
  `self.<field>`?" is re-implemented in 47 places; 21 of them are byte-identical
  to another.
* `_span_key` (42/14), `_target_names` (30/19), `_span_before` (26/13),
  `_span_within` (10/7), `_expr_contains_span` (9/6): the span-ordering algebra
  that `program_index.SourceSpan` (`:110`) should own is copied per reader.

The pattern is uniform: every reader was written as a self-contained file with
its own tail of 5–15 tiny helpers, and nobody lifted them into
`program_index`/`expression_eval`. This is the structural cause of the reader
layer's 60k lines.

## 7. The vocabulary of unknown

There is not one status vocabulary but **three typed ones plus a tail of ad-hoc
strings**:

1. **Fact statuses** — `evidence/context.py:29 FACT_STATUSES` (9) + `legacy_asserted`
   (`facts.py:33`): `code_proven, config_declared, class_default, code_and_config,
   derived, ambiguous, oracle_missing, asserted, unknown`. Failure statuses are
   `ambiguous | oracle_missing | unknown` (`facts.py:48`); only
   `code_proven | code_and_config` carry the negative-proof obligation
   (`facts.py:67`). Renderers owe a witness only for
   `code_proven, config_declared, class_default, code_and_config`
   (`renderers/html/fact_projection.py:24`).
2. **ReaderResult statuses** — `evidence/reader_result.py:23`: `resolved, incomplete,
   ambiguous, absent, failed`; completeness `complete | partial | none` (`:24`);
   failure kinds (`:25-35`) `missing_source, parse_failure, unsupported_syntax,
   dynamic_dispatch, external_unavailable, unresolved_import, out_of_owner,
   incomplete_graph, conflict`; provenance kinds `source, config, code_and_config,
   derived, external` (`:36`). Note the *second* failure-kind vocabulary in
   `facts.py:44` (`unsupported_syntax, unresolved_import, ambiguous_ownership,
   source_missing, reader_error`) overlaps but does not equal this one
   (`source_missing` vs `missing_source`).
3. **Per-dataclass local vocabularies** declared only in comments or module
   constants: `partial` (`call_arguments.py:88`), `active | inactive`
   (`projector_lineage.py:121`), `active, declared_unused, ambiguous, failed,
   unavailable` (`component_inventory.py:23`), `constructed | guard_absent`
   (`unet_selected_stage_children.py:37`), `resolved | incomplete | absent | failed`
   (`component_stages.py:77`), `unresolved` as a *state* (`layer_selector.py:779`),
   `unledgered` as an explicit non-status for the receipt join (`receipts.py:126`),
   `unavailable` as a reader-missing sentinel in the parser
   (`adapters/transformer/parser.py:4693`), and the op-graph's own
   `mechanism_unresolved | gating_unresolved | storage_unresolved | unsupported`
   (`opgraph.py:170-352`). `proven` appears as an ad-hoc dict status in adapter
   payloads (`adapters/diffusor/blocks.py:1128`,
   `special_parts/modalities/fusion.py:73`, `modalities/vision.py:108`).

Occurrence count of every status literal found in `status=` keywords,
`status` assignments/comparisons and `*STATUS*` constants (package-wide):

| string | count | example sites |
|---|---:|---|
| `resolved` | 573 | adapters/diffusor/config_binding.py:709, adapters/diffusor/config_binding.py:301 |
| `ambiguous` | 101 | adapters/transformer/parser.py:974, adapters/transformer/parser.py:4606 |
| `class_default` | 67 | adapters/diffusor/projection_ir.py:614, adapters/diffusor/projection_ir.py:1285 |
| `failed` | 61 | evidence/attention_child.py:252, evidence/attention_lane.py:588 |
| `absent` | 50 | evidence/attention_child.py:257, evidence/attention_mask.py:1331 |
| `code_and_config` | 49 | adapters/diffusor/projection_ir.py:529, adapters/diffusor/projection_ir.py:717 |
| `incomplete` | 41 | adapters/transformer/special_parts/modalities/evidence_projection.py:282, adapters/transformer/special_parts/modalities/evidence_projection.py:288 |
| `partial` | 36 | evidence/attention_mask.py:1439, evidence/call_arguments.py:242 |
| `code_proven` | 26 | adapters/diffusor/projection_ir.py:614, adapters/diffusor/projection_ir.py:643 |
| `derived` | 13 | evidence/arbitration.py:68, evidence/arbitration.py:263 |
| `active` | 8 | evidence/component_inventory.py:23, evidence/component_inventory.py:54 |
| `proven` | 6 | adapters/diffusor/blocks.py:1128, adapters/transformer/special_parts/modalities/fusion.py:73 |
| `oracle_missing` | 4 | adapters/transformer/parser.py:974, adapters/transformer/parser.py:4606 |
| `unavailable` | 4 | adapters/transformer/parser.py:4693, adapters/transformer/parser.py:4693 |
| `config_declared` | 4 | adapters/transformer/parser.py:1544, evidence/arbitration.py:68 |
| `unknown` | 4 | adapters/transformer/parser.py:4606, evidence/arbitration.py:252 |
| `asserted` | 4 | evidence/arbitration.py:68, evidence/context.py:29 |
| `legacy_asserted` | 4 | evidence/facts.py:33, evidence/facts.py:296 |
| `constructed` | 4 | evidence/unet_selected_child_execution.py:312, evidence/unet_selected_child_execution.py:429 |
| `status` | 3 | adapters/transformer/parser.py:4693, evidence/receipts.py:126 |
| `guard_absent` | 3 | evidence/unet_selected_child_execution.py:107, evidence/unet_selected_stage_children.py:37 |
| `inactive` | 2 | evidence/projector_lineage.py:128, evidence/projector_lineage.py:547 |
| `declared_unused` | 1 | evidence/component_inventory.py:23 |
| `unresolved` | 1 | evidence/layer_selector.py:779 |
| `unledgered` | 1 | evidence/receipts.py:126 |
| `mechanism_unresolved` | 1 | opgraph.py:352 |
| `gating_unresolved` | 1 | opgraph.py:352 |
| `mechanism` | 1 | opgraph.py:352 |
| `unsupported` | 1 | opgraph.py:173 |

(`status` ×3 and `mechanism` ×1 are artefacts of the collector matching
`.get("status")`/`reason == "mechanism"` and are not statuses.)

Reading: `resolved` dominates (573) because readers construct successes
explicitly; the honest-failure side is spread over **nine** spellings
(`ambiguous, failed, absent, incomplete, partial, oracle_missing, unknown,
unavailable, unresolved`) whose meaning depends on which dataclass you are in.
The projection to the drawing collapses all of them to "pale" through
`FACT_STATUSES`, so nothing in the HTML distinguishes "source missing" from
"source present, syntax unsupported".

## 8. What a reader needs to know to navigate — 15 modules in reading order

1. `model_unfolder/__init__.py` (102) — `unfold()` = `config_to_ir` → `Diagram`; the whole public surface is 21 names (`:26`).
2. `parser.py` (875) — `config_to_ir` (`:23`): id/dict/config coercion, HF loading with retries, class-default hydration, `find_adapter`, evidence attachment (`:448`).
3. `adapters/__init__.py` (15) — `find_adapter`: transformer first, diffusor second; there are only two.
4. `evidence/sources.py` (822) — how modeling files are found (installed transformers/diffusers, hub, path); the `SourceBundle`.
5. `evidence/document.py` (386) + `evidence/context.py` (452) — `prepare_document` and `ParseContext`: the one immutable per-parse context every reader receives; the `FactLedger` lives here.
6. `evidence/program_index.py` (2,929) — the raw program index: every dataclass a reader consumes (`SourceSpan:110`, `ExprNode:238`, `CallableRecord:318`, `ConstructionSite:542`, `ProgramIndex:2483`, `build_program_index:2818`). Fan-in 122.
7. `evidence/component_owner.py` (2,155) — occurrence ownership over the index: `OwnerOccurrenceId:45`, `OwnerGraph:222`, `resolve_component_root:1531`, `resolve_declared_model_stage:1780`.
8. `evidence/reader_result.py` (262) + `evidence/facts.py` (331) + `evidence/registry.py` (1,502) — the typed result contract, the fact laws (I-3/I-5/I-9), and the closed key registry.
9. `evidence/config_access.py` (1,546) — the owner-scoped config-access ledger: how a config read becomes a consumed, receipted fact.
10. `evidence/execution_flow.py` (1,237) + `evidence/expression_eval.py` (725) — addressed invocation and def-use resolution; owner-scoped expression evaluation. Most readers are thin over these two.
11. `evidence/attention.py` (5,324) — read as *the* example reader: dataclasses first (`:141-783`), then `attention_head_binding_at_block`; every other reader follows its shape.
12. `adapters/transformer/parser.py` (4,996) — the `parse()` body in section order (`:876` onward: identity → text unwrap → attention shape → U8 mask → U8-B position → U8-D storage → FFN/MoE → norms → composites → warnings).
13. `ir.py` (612) + `opgraph.py` (1,250) — what the parser produces and the canonical op regions the renderers draw.
14. `renderers/html/document.py` (241) → `views.py` (1,432) → `block_views/registry.py` (230) → `graph_engine.py` (671) — HTML document, architecture view, click-through detail router, the one SVG layout engine.
15. `sable.py` (855) — the harness: parse → render → nets (`:519-615` for the U2 nets) → gallery → bless/check_regression. Read with `.claude/PROTOCOL.md` *(not re-verified in this pass)*.

## 9. Health verdict per category

**Substrate (46 files, 27.3k lines) — sound core, blurred edges.** The centre
(`program_index`, `component_owner`, `execution_flow`, `reader_result`, `facts`)
is coherent, typed to the point of 200 dataclasses, and is what every reader
imports (fan-in 122/98/35/99). Its edges are not clean: four substrate modules
import helpers that were left inside readers (§3 Q4), `block_schema` imports the
renderer registry (`block_schema.py:378`), and the fact ledger / config-access /
receipts trio forms a 6-module cycle broken only by lazy imports. `arbitration.py`
is unused. `registry.py` is 1,502 lines of which ~95 % are data rows that arguably
belong in `everchanging/` by the project's own "vocabulary is YAML" rule
(*judgement, not measured*).

**Reader (88 files, 60.2k lines) — correct direction, wrong economics.** No
reader imports a renderer or adapter; every one returns a typed `ReaderResult`
and cites spans. But the layer is 45 % of the codebase because each reader
re-implements its own 5–15 span/expression helpers (129 duplicated names, `_self_field`
×47), eight readers exceed 1,500 lines, and one (`attention.py`) is 5,324 lines
with substrate-grade helpers trapped inside it. Fourteen U11 UNet readers
(8,619 lines) are complete, tested and **not called by the product**.

**Adapter (36 files, 18.2k lines) — the worst-shaped code in the package.**
`parse()` at `adapters/transformer/parser.py:876` is 3,762 lines with 67
function-local imports; `adapters/diffusor/parser.py` has 37 more and still
routes UNet facts through the quarantined `patterns.py` readers (`:56-122`).
Adapters import display vocabulary (`labels.py`, 17 statements) and a private
renderer constant (`blocks/feed_forward.py:493`), so the IR is built with
presentation in hand. Four dead pre-ProgramIndex helpers remain in the
transformer parser. The block declaration helpers (`blocks/attention.py`,
`feed_forward.py`, `layers.py`) are reasonable in isolation but sit inside cycle #3
with the renderer registry.

**Consumer (73 files, 16.5k lines) — small, layered, one 600-line view.** The
renderer is well factored around `graph.py` → `graph_engine.py` → views, and the
`expanded/` JSON package is tidy (16 files, largest 274 lines). Problems:
`_build_architecture_view` (`views.py:112`, 600 lines) and
`metadata_modalities.py` (1,400 lines) carry the multimodal complexity that the
adapter did not model; `metadata_modalities.py` and `encoder_panel.py` reach back
into `submodel.py`/the transformer parser (§3 Q1). `evidence/conformance.py`
(2,106 lines) is a consumer by role but is entangled with every legacy parser and
is the biggest single obstacle to deleting them (cycle #5, 4 legacy `ast.parse`
sites).

**Gate (8 files, 5.4k lines) — heavy, and the only category that scans itself.**
Every gate is a lawful `ast.parse` site per the register; they are the reason
the quarantine holds. Cost: 5.4k lines of self-inspection and a `structural_debt.py`
register whose helpers exist only for tests (`:999-1029`). No wrong-way imports
except `config_access.py:960` lazily pulling `receipts`.

**Legacy (5 files, 3.1k lines) — smaller than its footprint.** Only 2.3 % of
lines, but it is still on the *production* path of every UNet diagram
(`adapters/diffusor/parser.py:56-122`) and of every conformance run
(`conformance.py:30-34`). The quarantine register is precise (9 parse sites,
5 readers, deletion units U11/U14) and mechanically enforced; the replacement
(U11 readers) is written but unwired.

**Product (10 files, 1.8k lines) — thin and fine,** apart from `sable()` being a
301-line function with 37 lazy imports; `html_renderer.py` is a 5-line
compatibility shim.

**Vocab (1 loader + 18 YAML, 1.8k lines) — healthy and honoured.** Every YAML
has a header explaining its contract; the loader is the only consumer; 16
modules import it. The rule "vocabulary in YAML, never in code" is upheld for
parsing aliases but not for the fact registry (`registry.py`) or the op-graph's
status strings.

## 10. Loose ends

* **U11 is half-landed.** 14 readers / 8,619 lines with no production caller;
  the diffusor adapter still draws from `patterns.py`. Whether the U11 readers'
  outputs already match the legacy readers on the corpus is *not verified here*
  (pass 03/05 should check).
* **`arbitration.py` and `selected_composite_ffn.py`** — unwired; are they
  planned consumers or abandoned? Not determinable from code.
* **The `ast.parse` census differs by one.** The inventory doc says 20 exact
  evidence-layer sites; this walk found 19 literal `ast.parse(` calls
  (`scratchpad/r01/raw.json`). Either one site uses an alias/`compile` or the doc
  counts `sources.py:693` twice — *unverified*.
* **`exact_config_path_for_expression` in `attention.py`** is imported by three
  substrate modules; moving it is a mechanical fix that would cut the largest
  reader's fan-in from 21 to ~17 *(estimate)*.
* **Two failure-kind vocabularies** (`facts.py:44` vs `reader_result.py:25`) are
  not reconciled; the map does not say which one the drawing sees.
* **`registry.py` as data-in-code** contradicts the everchanging rule; whether the
  register is meant to be code-locked (a gate) or vocabulary is a doctrine
  question for pass 00/06.
* **Cycle #3 (adapter blocks ↔ renderer registry ↔ submodel)** cannot be broken
  without deciding who owns `submodel.py`; it is categorised `adapter` here but
  half of its imports are renderer-side.
* **Health flags only cover what AST can see.** No runtime profiling, no test
  execution, no check of the `.claude/worktrees/verify-*` copies that the grep
  surfaced (they are stale duplicates of the package and should not be searched
  by future passes).
* Per-file fan-in counts treat `from .pkg import name` as an edge to the
  package `__init__` when `name` is not a submodule; a handful of edges to
  `evidence/__init__.py` (fan-in 12) may therefore stand for a specific module.
