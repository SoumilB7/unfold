# model-benchmark

An idealized benchmark catalog + harness for measuring how well an architecture-extraction
mechanism (specifically: build a HF model with the library's own class using zero-storage
weights, run one real forward pass, record every op and which weights it touched, and check that
against the checkpoint's safetensors headers) generalizes across **every kind of model that
exists**, not just the top-downloaded ones.

This exists because `model-unfolder` (the sibling project, `../unfold-pkg/`) turns any Hugging
Face model into an architecture diagram, and a prior benchmark run (preserved in `baselines/`)
found the winning extraction mechanism but only measured it against the top-500-downloaded
models. The owner asked for a benchmark organized by model **category** first, so future runs have
a real denominator: "did we cover every officially-supported architecture class, not just the
popular ones."

## The three axes (`catalog/taxonomy.yaml` -> `scripts/build_catalog.py`)

1. **Architecture-class coverage** -- every `model_type` registered in the installed
   `transformers`' `CONFIG_MAPPING` (656 of them), and every `diffusers` pipeline class (303 of
   them), each matched to its top-2 most-downloaded public Hub repos, verified live by reading
   that repo's actual `config.json` / `model_index.json` (not by name/tag guessing). A class with
   no matching repo is recorded as **uncovered** -- a finding, not an error.
2. **Popularity** -- top-N most-downloaded repos per Hub `pipeline_tag`, for the 23 tasks listed
   in `taxonomy.yaml`'s `popularity_axis` (text-generation 300, image-text-to-text 100, ... down to
   time-series-forecasting 10). Anchors the catalog to what people actually run.
3. **Mechanism witnesses** -- for every mechanism feature named in the taxonomy (MLA, MTP,
   MoE shared/grouped routing, hybrid SSM/attention, sliding window, softcapping, dual-stream DiT
   transition, 3D-RoPE, RVQ codebooks, ...), at least 2 real repos verified live to actually declare
   the specific config field that proves it (never trusted from memory/identity).

Every row in `catalog/index.json` carries which axis/axes included it and why (`sources`), plus
`category`/`subcategory`, `pipeline_tag`, `library_name`, `architectures`/`model_type` (or
`diffusers_pipeline_class`), `total_params` + `dtypes` (from safetensors metadata), `quant_format`,
`gated`/`remote_code`/`has_safetensors`/`class_in_installed_library`, `inputs_needed`, `downloads`,
`base_model`, `mechanism_tags`, `negative_control` (gguf_only / remote_code_only / gated /
adapter_only_lora / broken_or_partial_config), and `classification_depth` (`deep` for
text-generation rows where the full `config.json` was fetched to detect mechanism fields; `coarse`
everywhere else, using `pipeline_tag` + `config.model_type` keyword membership only).

## Taxonomy summary (`catalog/taxonomy.yaml`)

18 top-level categories (text_decoder with 11 mechanism subcategories; encoder_only;
encoder_decoder; vision_language; omni_any_to_any; speech_recognition; audio_generation;
audio_classification; vision_backbone; detection_segmentation; dense_vision_other;
image_diffusion; video_diffusion; audio_diffusion; vae_autoencoder; time_series;
speculative_decoding_draft; plus `uncategorized` as an explicit safety-net bucket -- a non-zero
count there is a taxonomy/classifier gap to fix, not a normal outcome). Each subcategory states a
one-line definition, the `inputs_needed` to actually exercise it, and its `mechanism_features` with
the exact `evidence_fields` a detector should read. Also defines the three stress axes (scale
bands tiny->1T, quantization formats fp8/nvfp4/mxfp4/gptq/awq/bnb/compressed-tensors/gguf/mlx,
and five negative-control kinds).

## Rebuilding the catalog

```
cd model-benchmark
python3 scripts/build_catalog.py            # full run (network calls, ~1 minute; resumable via catalog/_cache/*.json)
python3 scripts/build_catalog.py --fast     # smoke-test the SCRIPT itself: shrinks popularity N's,
                                             # truncates the fallback search lists to 15/20 items
```
Delete `catalog/_cache/*.json` to force any phase to redo its live HF calls. The script never
downloads model weights; the only non-metadata HTTP calls are small `config.json` /
`model_index.json` / `1_Pooling/config.json` text fetches for mechanism-witness verification and
for the ~285 `text-generation` rows deep-classified for mla/mtp/moe/hybrid/sliding/softcap (see
"Deep text-decoder classification" in `catalog/coverage.md`) -- done via direct HTTP, never through
`huggingface_hub`'s local cache.

## Running the harness by category

`harness/run_all.py` is the original `BENCH/run_all.py` adapted to read `catalog/index.json`
instead of a fixed `models.json`, and to route by category/library instead of a hardcoded
text/vlm/diffusion split. **No measurement logic changed** -- `common.py` / `worker_zs.py` /
`worker_text.py` / `worker_diffusion.py` are byte-for-byte the mechanisms from `BENCH`.

```
cd harness
python3 run_all.py --mechanism zs --category text_decoder --subcategory mla --workers 4
python3 run_all.py --mechanism classic --category vision_language --repo llava-hf/llava-1.5-7b-hf
python3 run_all.py --mechanism diffusion --category image_diffusion --subcategory flux_style
```
`--mechanism` is `zs` (the winning mechanism, `worker_zs.py`), `classic` (the classic mechanisms,
`worker_text.py`), or `diffusion` (`worker_diffusion.py`, only applies to `diffusers` rows).
Results land at `results/<category>/<subcategory>/<mechanism>__<repo>.json`, resumable (a job
whose output already has `"done": true` is skipped and reported `cached`), with the same
per-job timeouts as the original BENCH (1800s text/vlm, 2400s diffusion, overridable with
`--timeout`).

## What is not yet runnable, and why

`run_all.py` will not invent measurement logic for an input shape the workers were not built for.
It **skips** (with a printed reason) any row in: `encoder_only` (no causal-LM head), `vision_backbone`
/ `dense_vision_other` (plain image classifiers -- the workers' image path is built for VLM
chat-template pairing, not a bare `ViTImageProcessor`), `detection_segmentation` (no box/mask
ground truth to score against), `time_series` (no numeric context-window builder),
`speculative_decoding_draft` (needs the target model's hidden states, not just `input_ids`), and
the `vision_language.video_llm` subcategory (no real video tensor input). `--mechanism classic`
additionally skips audio-only categories (`worker_text.py` has no audio forward path; only
`worker_zs.py` does). The full list with reasons is in `catalog/taxonomy.yaml`
(`harness_input_paths.not_implemented`) and echoed in `catalog/coverage.md`.

`results/SMOKE.md` documents an 8-job smoke run across 6 tiny models (text_decoder dense + MoE,
encoder_decoder, vision_language, speech_recognition, image_diffusion) proving the routing and
resumability work end to end, run strictly sequentially (`--workers 1`), one process at a time.

## Baselines

`baselines/2026-09-28-top-downloads/` is the durable copy of the prior top-500-downloads benchmark
run (the one that found the winning mechanism): `models.json` (the 500-row fixed denominator used
then), `REPORT.md` (the measured mechanism comparison), `other_libs/summary.md` (GGUF / MLX / ONNX
/ vLLM-registry alternative-mechanism results), and `results/` (every per-model JSON, 1043 files).
`harness/` in this repo is that same code (`common.py`, `worker_*.py`, `run_all.py`, `report.py`,
`select_models.py`), copied verbatim except for `run_all.py`'s catalog-routing rewrite described
above. This is the ONLY durable copy -- the original lived in a session scratchpad that will be
cleaned up.
