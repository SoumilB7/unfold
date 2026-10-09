# exec_capture cleanup inventory (branch exec-capture/cleanup, from 9364a97b)

Scope: every file PR #15 adds (`git diff --numstat origin/main...9364a97b`: 149 files, 146,399 lines, all under
`exec_capture/`). Goal: remove what is redundant without changing what the harness grades. Every row below says what
the file is, who reads it (grep over the whole branch), its class, and the action.

Classes: KEEP-CODE (code on a live path) · KEEP-INPUT (data the harness reads at run time) · GENERATED (reproduced
by a script in the branch) · DUPLICATE (copy of something else) · LOG/DUMP · RESEARCH-ONLY (tools on no grading
path) · DEAD-CODE (unreachable parts inside live modules, listed per module in section 3). Prose files fit none of
these, so they carry an eighth label, DOC.

Status of this file: written before the cleanup (the plan), then updated with what was done and how it was
verified (section 6).

## 1. Summary by class (before the cleanup)

| class | files | lines | action |
|---|---|---|---|
| KEEP-INPUT: catalog/index.json | 1 | 64,773 | reformat to one row per line (same `json.load` content) |
| KEEP-INPUT + evidence: results_v2/_witness/vllm/*.json | 84 | 33,880 | keep; slimming proposed, not done (section 5) |
| GENERATED: results_v2/summary.json, partials.json | 2 | 36,185 | keep the content, one row per line; writers changed to match |
| GENERATED, kept as is: SUMMARY.md, PARTIALS.md, POINTERS.md, catalog/coverage.md | 4 | 713 | keep |
| KEEP-CODE: harness_v2 (every module except the two cache probes) | 20 | 4,860 | keep; dead code and duplicated steps removed (section 3) |
| RESEARCH-ONLY: harness_v2/cache_probe.py, cache_probe2.py | 2 | 280 | delete (research note now points at 9364a97b) |
| RESEARCH-ONLY, cited evidence: docs/evidence/probes/*.py (minus the two copies), deepseek-walkthrough/*.py, v_old/v_new.json | 25 | 695 | keep |
| DUPLICATE, cited evidence: docs/evidence/probes/common.py, lowcost.py | 2 | 453 | needs owner decision (keep recommended) |
| LOG/DUMP, cited evidence: docs/evidence/audit-pods/D-full-suite.log | 1 | 3,313 | needs owner decision (slim recommended) |
| DOC: README.md, KV_CACHE_RESEARCH_2026-10-05.md, docs/benchmark.md, audit-pods A/E, catalog/taxonomy.yaml, results_v2/fail_classes.json | 7 | 1,237 | keep (docs/benchmark.md: owner decision) |
| config: exec_capture/.gitignore | 1 | 10 | keep |

## 2. Every file

Consumers were found with `git grep` over the branch. "CLI" = run by hand; the README §8 lists the commands.

### 2.1 catalog/ and results_v2/ (data)

| file | lines | purpose | read by | class | action (line saving) |
|---|---|---|---|---|---|
| catalog/index.json | 64,773 | the 1,988-row benchmark catalog | run_batch.py `json.load(--catalog)`; fields read: `repo, category, subcategory, library_name, quant_format, diffusers_pipeline_class, negative_control, base_model`, and copied into each result's `catalog`: `sources, pipeline_tag, mechanism_tags, total_params, gated, remote_code`. Never read: `architectures, class_in_installed_library, classification_depth, downloads, dtypes, has_safetensors, inputs_needed, model_type, notes` | KEEP-INPUT | reformat one row per line: 64,773 → 1,990 (−62,783). The writer, scripts/build_catalog.py, lives in model-benchmark/ and is not in the branch |
| catalog/taxonomy.yaml | 562 | category/mechanism taxonomy the catalog was built from | build_catalog.py (not in branch); README §4 and docs/benchmark.md cite it; no harness code reads it | DOC | keep |
| catalog/coverage.md | 214 | coverage report of the catalog build | written by build_catalog.py (not in branch); README §4 cites it | GENERATED (generator outside the branch) | keep |
| results_v2/summary.json | 20,314 | one row per measured repo: verdict, failed checks, reason, secs, peak GB | written by aggregate.py; read by nothing in the branch (resolve_pointers.py skips it) | GENERATED | keep content, one row per line: 20,314 → 1,946 (−18,368); aggregate.py writes that format. Not deleted: its input, the per-model result tree, is local-only (gitignored), so this file is the only committed per-repo verdict record |
| results_v2/SUMMARY.md | 103 | headline table | aggregate.py; README §4 cites it | GENERATED | keep |
| results_v2/partials.json | 15,871 | cause list for every PARTIAL | written by partials.py; read by nothing | GENERATED | keep content, one row per line: 15,871 → 272 (−15,599); partials.py writes that format |
| results_v2/PARTIALS.md | 316 | PARTIAL causes table | partials.py; README §4 cites it | GENERATED | keep |
| results_v2/POINTERS.md | 80 | GGUF/MLX/LoRA copies linked to their base model | resolve_pointers.py (also needs baselines/2026-09-28-top-downloads/other_libs/mech_a_gguf.json, not in branch) | GENERATED (one input outside the branch) | keep |
| results_v2/fail_classes.json | 213 | hand classification of 211 FAIL reasons (2026-09-29) | no script in the branch or in model-benchmark writes it; README §5 cites its counts | DOC (measurement record) | keep; it predates the 2026-10-02 numbers (FAIL is now 104) |
| results_v2/_witness/vllm/*.json (84 files, appendix A) | 33,880 | vLLM execution witness per MTP repo | written by mtp_witness_run.py; read by worker_v2.py T1 (fields `closed, executed_checkpoint_keys, executed_by, vllm_class, notes, vllm_mtp_arch` and whether `vllm_mtp_arch` is present, `registry_error`); README §6H cites the measured counts | KEEP-INPUT + evidence | keep; slimming proposed in section 5 |

### 2.2 harness_v2/ (code)

| file | lines | purpose | read by | class | action |
|---|---|---|---|---|---|
| run_batch.py | 197 | runner: one subprocess per repo, memory/disk guards, result file | CLI | KEEP-CODE | 1 dead local removed |
| worker_v2.py | 1,065 | transformers worker: build, passes, modes, T1–T5, verdict | run_batch.py (non-diffusion rows) | KEEP-CODE | dead import/name removed; 4 duplicated steps folded (section 3) |
| worker_v2_diffusion.py | 526 | diffusers worker, per component | run_batch.py (diffusers rows) | KEEP-CODE | unused import removed |
| inputs.py | 426 | input builder from forward signature + repo processor | worker_v2.py (`build_passes`, now also `mask_positions`), exec_to_ir.py | KEEP-CODE | 2 dead lines removed |
| synth.py | 185 | input synthesizer by argument name | worker_v2.py, worker_v2_diffusion.py | KEEP-CODE | unused import + local removed |
| dag.py | 311 | DagRecorder + analyse | worker_v2, worker_v2_diffusion, vllm_witness, exec_to_ir, docs/evidence/probes/equiv_dag.py | KEEP-CODE | none |
| lowcost.py | 194 | zero_storage, HEAVY/BIAS_FIRST, header readers | worker_v2, worker_v2_diffusion, dag, vllm_witness, mtp_witness_run, exec_to_ir | KEEP-CODE | `Recorder`, `fetch_cached` removed (section 3) |
| noweights.py | 48 | no-weight-download guard | worker_v2, worker_v2_diffusion, vllm_witness, exec_to_ir | KEEP-CODE | none |
| common.py | 339 | checkpoint split, name matcher, library rename/ignore | worker_v2, worker_v2_diffusion, mtp_witness_run (`ckpt_split`) | KEEP-CODE | 9 unreferenced helpers removed; `buffer_names` added (section 3) |
| libload.py | 188 | library authority (from_pretrained on sparse stubs) | worker_v2 (`Authority`) | KEEP-CODE | `library_load_report` removed |
| gguf_parts.py | 223 | GGUF header parts list | worker_v2 | KEEP-CODE | none |
| vllm_witness.py | 260 | vLLM MTP witness build/run | mtp_witness_run.py | KEEP-CODE (runs only in .venv_vllm) | none: it cannot be re-run here, so it is not touched (findings in section 3) |
| mtp_witness_run.py | 91 | writes results_v2/_witness/vllm/<repo>.json | CLI (.venv_vllm) | KEEP-CODE | none (same reason) |
| exec_to_ir.py | 451 | execution recording → unfold ModelIR → renderer | CLI; imported only by cache_probe*.py | RESEARCH-ONLY (product prototype, no grading path) | keep: it is the product direction, not redundant |
| cache_probe.py | 111 | KV-cache research probe 1 | KV_CACHE_RESEARCH note only | RESEARCH-ONLY | delete (−111); note points at 9364a97b |
| cache_probe2.py | 169 | KV-cache research probe 2 | KV_CACHE_RESEARCH note only | RESEARCH-ONLY | delete (−169) |
| aggregate.py | 71 | → SUMMARY.md + summary.json | CLI | KEEP-CODE | unused import removed; writes one row per line |
| partials.py | 122 | → PARTIALS.md + partials.json | CLI | KEEP-CODE | writes one row per line |
| resolve_pointers.py | 36 | → POINTERS.md, adds `pointer` to pointer results | CLI | KEEP-CODE | none |
| bulk_roots.py | 79 | → results_v2/_bulk_features.json (not committed) | CLI (cwd must be exec_capture/) | KEEP-CODE | unused import removed |
| merge_rerun.py | 29 | merge rerun dirs into results_v2 | CLI | KEEP-CODE | none |
| compare_rerun.py | 19 | verdict transitions of a rerun | CLI | KEEP-CODE | none |

### 2.3 docs/ and top level

| file | lines | purpose | read by | class | action |
|---|---|---|---|---|---|
| README.md | 130 | track state, method, numbers, resume steps | humans | DOC | keep; regeneration commands added to §8 |
| KV_CACHE_RESEARCH_2026-10-05.md | 48 | KV-cache research findings | humans | DOC | keep; now names the commit holding the probes |
| .gitignore | 10 | keeps caches, venvs, per-model results out of git | git | config | keep |
| docs/benchmark.md | 114 | byte-identical copy of model-benchmark/README.md (outside the branch): catalog axes and fields (useful), plus the v1 `harness/` runner, `worker_zs.py`, `baselines/` (none of which is in the branch) | humans; nothing cites it | DOC (copy of a file outside the branch) | needs owner decision |
| docs/evidence/audit-pods/A-intent-history.md | 114 | audit report | README §1 cites it | DOC | keep |
| docs/evidence/audit-pods/E-product-output.md | 56 | product-output truth check | README §1 | DOC | keep |
| docs/evidence/audit-pods/D-full-suite.log | 3,313 | raw pytest output of the unfold-pkg suite (94 failed / 5,769 passed); lines 86–3217 are failure tracebacks | README §1 cites "D full-suite log" | LOG/DUMP (cited evidence) | needs owner decision |
| docs/evidence/deepseek-walkthrough/{step0_config, step1_husk, step2_export, step2b_details, step3_headers, step4_tool, partial_config, wrong_config}.py | 11+42+40+30+49+34+31+37 = 274 | DeepSeek-V3 walkthrough scripts | README §2, §10 | RESEARCH-ONLY (cited) | keep |
| docs/evidence/probes/{alt_probe, diff_probe, equiv, equiv_bf16, equiv_dag, hub_coverage, hub_unknown, meta_probe, persistent, profile_baseline, profile_cached, profile_lowcost, trace_probe, version_probe, vocab_probe}.py | 57+26+26+26+18+29+25+20+18+38+18+34+31+31+22 = 419 | the probes behind README §3 | README §10 | RESEARCH-ONLY (cited) | keep |
| docs/evidence/probes/v_old.json, v_new.json | 1+1 | version_probe outputs | README §3 (transformers 5.12.1 → 5.17.0) | RESEARCH-ONLY (cited) | keep |
| docs/evidence/probes/common.py | 332 | older copy of harness_v2/common.py (`library_ignored` reads class attributes only; `split_library_ignored` has no slot check) | the probe scripts import it via `sys.path.insert(0, ".")` | DUPLICATE (older copy; diff in section 4) | needs owner decision |
| docs/evidence/probes/lowcost.py | 121 | older copy of harness_v2/lowcost.py (`fetch_headers(workers=48)`, no `.bin` reader; still has `Recorder`, `fetch_cached`) | equiv.py, equiv_bf16.py, persistent.py, profile_cached.py, profile_lowcost.py via `sys.path.insert(0, ".")` | DUPLICATE (older copy) | needs owner decision |

## 3. Dead code and redundant steps inside live modules

Evidence is static (no caller in the branch, `git grep -w <name>`), plus `pyflakes`. A fields-present census of the
1,944 results in model-benchmark/results_v2 was also taken, but it is weak evidence for the CURRENT code: only 223
results carry `t1_authority` (the rest were written by earlier harness versions), so the census is used only to
show that paths are live, never to prove a path dead.

### 3.1 Removed (no caller anywhere in the branch)

| module | removed | why it is dead |
|---|---|---|
| common.py | `StepTimeout`, `time_limit`, `run_step`, `husk_tensors`, `skip_weight_init`, `DEPTH_FIELDS`, `truncate_depth`, `_SHARED_ZERO`, `zero_storage` (+ imports `signal, time, traceback, contextlib`) | no importer: worker_v2/worker_v2_diffusion import only `ckpt_split, match, library_rename, split_library_ignored, looks_like_stored_buffer, top_groups, numel, INT_DTYPES`; mtp_witness_run only `ckpt_split`. Every worker takes `zero_storage` from lowcost.py. The probe scripts use their own copy (docs/evidence/probes/common.py) |
| lowcost.py | `Recorder` (superseded by dag.DagRecorder), `fetch_cached` (superseded by the worker's own header cache), imports `TorchDispatchMode, tree_flatten, tree_map`, local `import json` | only the probe scripts use these names, and they import their own copy, docs/evidence/probes/lowcost.py |
| libload.py | `library_load_report()` | no caller; worker_v2 uses `Authority.report` directly |
| run_batch.py | `R_mem = round(fp, 2)` | assigned, never read |
| inputs.py | `encdec = ...`, `import types` | assigned/imported, never read |
| synth.py | `import itertools`, `prm = sig[p]` | never read |
| worker_v2.py | `import math`, `as shape_err` | never read |
| worker_v2_diffusion.py | `analyse` in the dag import | never called |
| aggregate.py, bulk_roots.py | `import re`, `import os` | never used |

### 3.2 Duplicated steps folded (same computation, one copy kept)

| where | duplicate | kept |
|---|---|---|
| worker_v2.py §2 build and `_fresh()` | the zero-storage build (eager attention, float32, `(TypeError, ValueError)` fallback) was written twice | `_fresh()`, now defined before §2 and used for the main build too |
| worker_v2.py `_unused_params()` | identical to `_unused_now()` (`{n for n, _ in m.named_parameters()} - used_params`) | `_unused_now()` |
| worker_v2.py training pass `bool_masked_pos` | the same patch-count formula as `inputs.mask_positions(kw, cfg)` (same patch size, tubelet, 2-D fallback, `arange(n) % 2 == 0`) | `inputs.mask_positions` |
| worker_v2.py T1 `model_buffers` and common.split_library_ignored `buf_names` | the same buffer-name set (named_buffers + `_non_persistent_buffers_set`) built twice | new `common.buffer_names(model)` |

### 3.3 Redundancies found and left alone (folding them would change behaviour or cannot be re-run here)

- Class-candidate enumeration over the auto mappings is written twice in worker_v2.py (library selection and matcher
  selection). They differ: the library one skips `*CONFIG*` mappings and `*Config` names; the matcher one does not
  (it builds `BertConfig`-style names, which fail inside `_uncovered_numel` and are dropped). Merging changes which
  candidates are tried.
- Config walks: `_cap` (recursive over `dir`), `_kern` (recursive over `vars`), `_pos_lengths` and the long-input
  `_scan` (one level over `dir`), `synth._subconfigs` (one level, dict-aware), `inputs._cfg_get` (fixed list of
  sub-configs). Different depth and source, so different results on nested configs; not merged.
- Stored-buffer predicates: `common.looks_like_stored_buffer`, the inline formula in `split_library_ignored`
  (adds buffer names), `worker_v2._stored_constant` (needs N in the model's position lengths), `libload.stored_buffer`
  (integer table matched by name part and size). Four different rules.
- `build_passes` is called twice per model (main and variant), and each call loads the repo's processor components
  again from the Hub. Sharing them would hand the variant pass a processor whose size limits the main pass
  changed and restored.
- run_batch.py builds the base-model job in two near-identical blocks (GGUF rows set `quant_format: None` as well).
- worker_v2.py and worker_v2_diffusion.py each define `_last_resort`, `save`, `fail` (same text).
- vllm_witness.py / mtp_witness_run.py: `import json` and `analyse` (vllm_witness) and `import torch`
  (mtp_witness_run) are unused, and mtp_witness_run's comment names `apply_witness.py`, which no longer exists (the
  closure is decided in worker_v2.py). Not edited: they run only in .venv_vllm with the vLLM source, so the
  regression run cannot cover them.

### 3.4 Superseded paths that are NOT dead (when the old path still runs)

- Name matcher (`common.match`) as T1 authority: runs whenever there is no usable library load report. That is (a)
  .bin-only repos (`libload` mirrors `.safetensors` only: `{"error": "no safetensors weight files"}`), (b) GGUF repos
  (the library-authority block is skipped when `GGUF_RAW` is set), (c) the library load raises or times out (meta
  load not possible, e.g. an `aten::equal` tie check, quantization formats the loader cannot run on meta). With a
  report, the matcher still runs: its `class_lacks` keys name the library's silent drops, and `matcher_view` is
  stored for comparison. Census: 223 results with `t1_authority = library_load_report`; matcher-only rows in every
  category.
- Matcher class selection (`_uncovered_numel` → `class_selected_by_weights`): runs when headers exist and
  `_auth_sel` is False, i.e. cases (a) and (c) above. GGUF repos get no class selection at all (their parts list is
  only built after the model, so `T` is empty at that point). 58 results carry `class_selected_by_weights`.
- `fetch_bin_headers` / `_bin_index`: the .bin ground truth (134 results with `ground_truth_source =
  pytorch_bin_index`; the diffusion worker uses `_bin_index` for .bin components).
- Env knobs read by the worker but never set by run_batch: `BENCH_STAGE_S` (150), `BENCH_LIBLOAD_S` (1200),
  `BENCH_TOKEN_BUDGET` (inputs.py, 1024). Operator overrides; kept.

## 4. DUPLICATE evidence

`diff docs/evidence/probes/common.py harness_v2/common.py` (before the cleanup): the probe copy's
`library_ignored` reads only class attributes (`pats = []`), returns `[re.compile(p) for p in pats]` without
de-duplication, and `split_library_ignored` has no "key has a slot in the built model" exclusion.
`diff docs/evidence/probes/lowcost.py harness_v2/lowcost.py`: the probe copy has `fetch_headers(repo, workers=48)`
and lacks `_STORAGE_DTYPE`, `_bin_index`, `fetch_bin_headers`. So these are older versions, kept because the probe
scripts import them exactly as they ran.

## 5. Proposals (not done)

- Witness JSONs (33,880 lines, 84 files): the worker reads 7 fields. The bulk is four key lists:
  `executed_checkpoint_keys` 13,885 lines, `mtp_executed` 12,211, `mtp_not_loaded` 3,968, `op_kinds` 2,423.
  `mtp_executed` equals the `mtp.*` / `.mtp.` subset of `executed_checkpoint_keys` in 42 of the 43 files that have
  it; the exception is Qwen3-Next-80B, whose file has only `mtp_executed` (older schema, see bug 2). Options:
  (a) write each file compact (one line, identical `json.load` content: 33,880 → 84 lines), changing
  mtp_witness_run.py's `indent=1` to match; (b) also drop `mtp_executed` where it is derivable.
- catalog/index.json: 9 of 23 fields per row are never read by the harness (2.1). Keep them (they document how
  each repo entered the catalog) unless the owner wants a slimmer input.

## 6. What was done (filled in after the cleanup)

See the cleanup report on the PR for the commit list; per-commit verification is recorded below.

| commit | change | verification |
|---|---|---|
| 134ee217 | removed cache_probe.py / cache_probe2.py (research only, nothing imports them) | grep: no importer; research note points to commit 29ed4b6e |
| 6a12c8fe | catalog/index.json one row per line (64,773 → 1,990 lines) | `json.load` equal before/after (1,988 rows) |
| 96516199 | summary.json, partials.json one row per line; aggregate.py / partials.py write that form | `json.load` equal (1,944 / 270 rows) |
| 9ce6ad9e, 6707f4db | helpers nothing calls removed from common.py, lowcost.py, libload.py | grep over the branch and model-benchmark: no live caller |
| e0e1a2cd | unused imports / never-read locals (pyflakes) | stdlib or same-module imports only, no side effects |
| ce212ff9 | four duplicated steps folded into one copy each | 33+3 control repos run before/after: identical except a temp-dir path in one error message (same noise between two pre-cleanup runs); outputs in model-benchmark/_reruns/cleanup/ |

## 7. Bugs noticed (reported, not fixed: this is a cleanup)

1. worker_v2.py header cache: on a cache hit `ground_truth_source` is never set (it is set only on a miss). So a
   result's fields depend on cache state (993 of the live results have no `ground_truth_source`), and for a .bin-only
   repo with cached headers the check `ground_truth_source in (None, "safetensors")` passes, so the library authority
   runs anyway, mirrors nothing and records `library_load_report_error: "not run: no safetensors weight files"` plus a
   `library_class_selection` block, which a cache miss would not record. Seen in the control run: facebook/opt-125m,
   tiny-random-SwitchTransformers. The verdict goes through the matcher either way.
2. results_v2/_witness/vllm/Qwen__Qwen3-Next-80B-A3B-Instruct.json is in an older schema: `closed: true` but no
   `executed_checkpoint_keys` and no `vllm_mtp_arch`. worker_v2 reads `executed_checkpoint_keys or []`, so this
   witness covers 0 keys (`executed_by_witness_partial`) and the repo cannot reach FULL_LEFTOVERS, although README
   §6H reports Qwen3-Next 1,553/1,553 executed. Re-running mtp_witness_run.py for it would fix the record.
3. `library_load_report_error` can carry a temp-dir path (e.g. `OSError: /var/folders/...`), so that field differs
   from run to run for the same repo.
4. bulk_roots.py uses `RES = "results_v2"` relative to the working directory, unlike the other report scripts
   (relative to the script); run from anywhere else it reads nothing.
5. README §1 and §10 cite `evidence/...`; in this folder the files are under `docs/evidence/...`.
6. mtp_witness_run.py's comment names `apply_witness.py`, which does not exist.
7. results_v2/fail_classes.json has no generator and counts 211 FAILs (2026-09-29); the current results have 104.
8. aggregate.py divides by the runnable count in the headline line; with zero runnable rows it raises.

## Appendix A. Witness files

All 84 are KEEP-INPUT + evidence (written by mtp_witness_run.py, read by worker_v2.py T1). Action for every one: keep.

| file | lines | closed | executed_checkpoint_keys | vllm_mtp_arch | read by worker as |
|---|---|---|---|---|---|
| `CMSManhattan__JiRackDeltaNet_27b.json` | 10 | False | (absent) | Qwen3_5MTP | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `FacebookAI__xlm-roberta-base.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `Inferact__Qwen3.8-27B-NVFP4.json` | 109 | True | 17 | (absent) | covers declared-unbuilt keys |
| `OBLITERATUS__Ornith-1.5-9B-OBLITERATED.json` | 109 | True | 17 | (absent) | covers declared-unbuilt keys |
| `PaddlePaddle__PaddleOCR-VL-1.5.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `PaddlePaddle__PaddleOCR-VL-1.6.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `QuantTrio__Qwen3.5-9B-AWQ.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `QuantTrio__Qwen3.6-35B-A3B-AWQ.json` | 1664 | True | 787 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `Qwen__Qwen3-Next-80B-A3B-Instruct.json` | 1641 | True | (absent) | (absent) | closed but NO executed_checkpoint_keys (older schema): covers nothing |
| `Qwen__Qwen3.5-0.8B.json` | 108 | True | 16 | (absent) | covers declared-unbuilt keys |
| `Qwen__Qwen3.5-122B-A10B-FP8.json` | 2440 | True | 787 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.5-27B.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.5-2B.json` | 109 | True | 16 | Qwen3_5MTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.5-35B-A3B-FP8.json` | 2440 | True | 787 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.5-35B-A3B.json` | 1663 | True | 787 | (absent) | covers declared-unbuilt keys |
| `Qwen__Qwen3.5-397B-A17B-FP8.json` | 4744 | True | 1555 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.5-4B.json` | 109 | True | 16 | Qwen3_5MTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.5-9B.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.6-27B-FP8.json` | 118 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.6-27B.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.6-35B-A3B-FP8.json` | 2440 | True | 787 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.6-35B-A3B.json` | 132 | True | 21 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.8-27B-FP8.json` | 118 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `Qwen__Qwen3.8-27B.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `RadixArk__Qwen3.8-27B-NVFP4.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `RedHatAI__Qwen3.6-35B-A3B-NVFP4.json` | 132 | True | 21 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `apodex__Apodex-1.1-mini.json` | 1664 | True | 787 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `apple__DepthPro-hf.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `bosonai__higgs-tts-2-3b-base.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `cyankiwi__Qwen3.6-27B-AWQ-INT4.json` | 117 | False | 1 | Qwen3_5MTP | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `cyankiwi__Qwen3.8-27B-AWQ-INT4.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `datalab-to__chandra-ocr-2.json` | 109 | True | 16 | Qwen3_5MTP | covers declared-unbuilt keys |
| `datalab-to__surya-ocr-2.json` | 109 | True | 16 | Qwen3_5MTP | covers declared-unbuilt keys |
| `deepseek-ai__DeepSeek-R1.json` | 878 | True | 789 | DeepSeekMTPModel | covers declared-unbuilt keys |
| `deepseek-ai__DeepSeek-V3-0324.json` | 877 | True | 789 | (absent) | covers declared-unbuilt keys |
| `deepseek-ai__DeepSeek-V3.1.json` | 878 | True | 789 | DeepSeekMTPModel | covers declared-unbuilt keys |
| `deepseek-ai__DeepSeek-V3.2.json` | 10 | False | (absent) | DeepseekV32MTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `deepseek-ai__DeepSeek-V3.json` | 878 | True | 789 | DeepSeekMTPModel | covers declared-unbuilt keys |
| `deepseek-ai__DeepSeek-V4-Flash-0731.json` | 10 | False | (absent) | DeepSeekV4MTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `deepseek-ai__DeepSeek-V4-Flash-DSpark.json` | 10 | False | (absent) | DeepSeekV4MTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `deepseek-ai__DeepSeek-V4-Flash-Vision-Exp.json` | 10 | False | (absent) | DeepSeekV4MTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `deepseek-ai__DeepSeek-V4-Flash.json` | 10 | False | (absent) | DeepSeekV4MTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `deepseek-ai__DeepSeek-V4-Pro.json` | 10 | False | (absent) | DeepSeekV4MTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `distilbert__distilgpt2.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `efwkjn__cohere-asr-ja-v0.1.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `efwkjn__cohere-asr-ja.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `empero-ai__Qwen3.8-2B.json` | 109 | True | 16 | Qwen3_5MTP | covers declared-unbuilt keys |
| `empero-ai__Qwen3.8-4B.json` | 109 | True | 16 | Qwen3_5MTP | covers declared-unbuilt keys |
| `empero-ai__Qwen3.8-9B.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `facebook__esm2_t33_650M_UR50D.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `facebook__esm2_t6_8M_UR50D.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `facebook__sapiens2-pose-1b.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `facebook__sapiens2-seg-0.4b.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `google-t5__t5-base.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `google-t5__t5-large.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `google-t5__t5-small.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `google__gemma-3n-E2B-it.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `google__gemma-4-E2B-it.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `google__gemma-4-E4B-it.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `google__gemma-4-E4B.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `microsoft__deberta-v3-base.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `microsoft__deberta-v3-large.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `microsoft__deberta-v3-small.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `microsoft__mdeberta-v3-base.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `nvidia__GLM-5.2-NVFP4.json` | 10 | False | (absent) | DeepseekV32MTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `nvidia__NVIDIA-Nemotron-3-Super-120B-A12B-BF16.json` | 10 | False | (absent) | NemotronHMTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `nvidia__NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4.json` | 10 | False | (absent) | NemotronHMTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.json` | 10 | False | (absent) | NemotronHMTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.json` | 10 | False | (absent) | NemotronHMTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `nvidia__Qwen3.5-122B-A10B-NVFP4.json` | 1664 | True | 787 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `nvidia__Qwen3.6-27B-NVFP4.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `nvidia__Qwen3.6-35B-A3B-NVFP4.json` | 132 | True | 21 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `nvidia__Qwen3.8-27B-NVFP4.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `openai-community__gpt2-large.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `openai-community__gpt2.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `ornith-ai__Ornith-1.5-35B-A3B-NVFP4.json` | 1664 | True | 787 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `ornith-ai__Ornith-1.5-35B-A3B.json` | 1664 | True | 787 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `ornith-ai__Ornith-1.5-397B-FP8.json` | 3200 | True | 1555 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `principled-intelligence__gemma-4-E2B-it-text-only.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `sshleifer__tiny-gpt2.json` | 10 | False | (absent) | None | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `unsloth__Qwen3.6-35B-A3B-NVFP4.json` | 132 | True | 21 | Qwen3_5MoeMTP | covers declared-unbuilt keys |
| `unsloth__Qwen3.8-27B-NVFP4.json` | 110 | True | 17 | Qwen3_5MTP | covers declared-unbuilt keys |
| `zai-org__GLM-4.7-Flash.json` | 10 | False | (absent) | Glm4MoeLiteMTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
| `zai-org__GLM-5.2-FP8.json` | 10 | False | (absent) | DeepseekV32MTPModel | not closed: only vllm_mtp_arch/registry_error used (FULL_LEFTOVERS eligibility) |
