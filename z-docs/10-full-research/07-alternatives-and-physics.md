# 07 — Alternatives and physics: what best-in-class looks like, measured

Pass written 2026-09-02 against `unfold-pkg` HEAD `9b3cb7b` (branch
`audio-composite-support`), Python 3.12.10, torch 2.9.1, transformers 5.12.1,
diffusers 0.38.0, safetensors 0.8.0, huggingface_hub 1.10.1 (macOS, CPU only).
Read-only on the repository; every prototype, log and JSON lives under
`scratchpad/r07/` (paths below are relative to
`/private/tmp/claude-501/-Users-soumil-Code-Projects-Understand-llmvisualizer/a92a5cee-b94f-4931-a83f-631ec44bdf84/scratchpad/r07/`).
Tier vocabulary is the one `00-goal-and-intent.md:165-169` records: **tier 1
structure**, **tier 2 mechanism**, **tier 3 values**.

The question: what should the *authority* be for each tier, given that the
product promise is "the true, complete structure of any supported HF model,
never guessing, never silently omitting" (`README.md:9-12`), and given that the
project today answers *every* tier by static AST over one source bundle per
component (`evidence/sources.py:107-122`, `parser.py:396`)?

Short answer, argued below with numbers: **tier 1 belongs to instantiation
(meta device) plus a FakeTensor forward trace; tier 2 belongs to the existing
static forward readers, re-pointed at exact module instances; tier 3 belongs
to the config document plus the instance attributes `__init__` computed from
it.** Static AST stays the only source of *provenance to source lines* and of
*config-key binding*, which nothing in the landscape provides. The project's
current cold latency (17–85 s) is not a cost of static analysis at all; it is
one quadratic call site.

---

## 1. Landscape — who else draws architectures, and where their truth comes from

Web survey delegated to a research subagent (URLs are its citations; I
verified locally only the torch.fx / torch.export / meta-device rows, marked ✔).
Everything else in this table is **as reported by the subagent, unverified by
me** unless marked.

| tool | truth source | needs to run | coverage | honesty: unresolved / ×N / provenance | interactivity | last active |
|---|---|---|---|---|---|---|
| torchview (github.com/mert-kurttutan/torchview) | real forward via `__torch_function__` tensor subclass; `device='meta'` supported | instantiated module + input; forward runs | any nn.Module | shows only the path the sample input took; `roll=True` folds ×N; no source-line provenance | static graphviz | 2026-02 |
| torchinfo / torchsummary (github.com/TylerYep/torchinfo) | forward hooks during a forward pass | instantiated module (+input for shapes) | any nn.Module, *module level only* | silently omits functional ops (SDPA, RoPE math, residual adds); no ×N fold, no provenance | text table | 2026-08 |
| Netron (netron.app) | a serialized file: ONNX / TorchScript / torch.export / safetensors … | file only | exported formats only; a safetensors file shows a flat tensor list, no graph | shows whatever the file holds; ×N already unrolled; no Python-source provenance | click node → properties | 2026-09 |
| torch.fx `symbolic_trace` ✔ | Proxy execution of `forward` | instantiated module, no tensors | any module **without** data-dependent control flow | loud failure on control flow; loops unrolled; `record_stack_traces` optional and fragile | none itself | PyTorch 2.13 |
| torch.export ✔ | FakeTensor trace on example inputs → ATen graph | example inputs; module may be built under `FakeTensorMode` | any exportable module; data-dependent `if` needs `torch.cond` | loud failure; loops unrolled; **guaranteed `node.meta["stack_trace"]` (file+line) and `nn_module_stack`** — strongest provenance surveyed | none; feeds Netron / Model Explorer | PyTorch 2.13 |
| TensorBoard `add_graph` | `torch.jit.trace` of a forward pass | model + real input | any traceable module | *silently* specialises control flow; ×N unrolled; name scopes only | expand/collapse scopes | PyTorch 2.13 |
| Google Model Explorer (github.com/google-ai-edge/model-explorer) | an `ExportedProgram` (or TFLite/MLIR) | `torch.export` must succeed | PyTorch via export only; a diffusers UNet failed in issue #64 | groups ops by originating `nn.Module` instance; ×N unrolled per instance; stack_trace display unverified | hierarchical expand/collapse, search | 2026-09 |
| HF transformers docs / model cards | hand-drawn figure copied from the paper (e.g. BLIP-2); Llama page has no diagram; config docs link fields to `modeling_llama.py#L…` | nothing | per-model, inconsistent | no fidelity guarantee vs code; provenance exists for *config docs*, not for a diagram | static | 2026 |
| bertviz | attention tensors from a real forward | weights + forward | HF encoders/decoders | activations, not architecture | interactive heads | 2026-01 |
| Polo Club Transformer Explainer; Bycroft LLM Visualization | hand-designed GPT-2 geometry | browser | one family | fixed layout; no provenance | rich | 2026 |
| torchlens (github.com/johnmarktaylor91/torchlens) | Python-level wrapping of every torch op during a *real* forward; "No meta tensors" | real inputs + weights | any module incl. control flow, loops, SSMs | records everything that ran; rolled ×N view; **file+line per op** (`code_context`) | graphviz PDF | 2026-08 |
| torchexplorer, hiddenlayer, W&B watch, pytorchviz, Keras `plot_model`, onnx-modifier, netscope | autograd / jit trace / functional-model connectivity / ONNX / prototxt | forward+backward or a file | narrow; hiddenlayer and netscope dead | no provenance | varies | 2020–2026 |

What the survey shows (subagent's synthesis, which my prototypes corroborate):

- Drawing *without weights and without a forward pass* is done today only by
  file/config viewers (Netron, Keras, netscope) and hand-drawn explainers.
  None derives truth from PyTorch source.
- Drawing *without weights but with a forward* is done by torchview on meta,
  and by torch.export under `FakeTensorMode` (both confirmed below).
- Provenance to source lines exists in exactly two places: `torch.export`
  metadata (measured: 2135 of 2434 Llama graph nodes carry `stack_trace`,
  `llama.r2.json`) and torchlens (needs real weights).
- **No surveyed tool links a diagram node to a config key.** No surveyed tool
  reports "could not resolve" *inside* the diagram. torchinfo and jit-trace
  tools omit or specialise silently; fx/export fail loudly instead.
- ×N folding exists only in torchview (`roll`) and torchlens.

So the two things this project uniquely does — **config-path binding on every
value** (`evidence/config_access.py`, `everchanging/*/aliases.yaml`) and
**visible unknown chips with reasons** (`00-goal-and-intent.md:167`) — are
genuinely absent from the field. They are worth protecting. What the field
does better is *enumeration of what exists*: every tool that instantiates the
module gets the true module tree for free, and the project is the only one
that reconstructs it by reading Python.

---

## 2. Physics prototypes — measured on the five models (+ Qwen3.5-27B and T5-small for the named control-flow cases)

Scripts: `proto_meta.py` (round 1), `proto_round2.py` (round 2),
`time_unfold.py`. Configs from `unfold-pkg/tests/sable_test_corpus/*.json`
`["config"]`; diffusers classes from `from_config` on the same dict (component
and `_`-prefixed keys stripped). Attention forced to `eager`. No weights, no
network, no GPU.

### 2.1 Summary table

Times are wall seconds on this laptop. "cold" = first call in a fresh
process **after** `import torch`/`transformers` (those are 1.0–1.7 s and
1.5–2.0 s respectively, `*.r2.json:import_*_s`); the round-1 numbers that
include the lazy first import of the model class are given in brackets.

| model | a. meta instantiate cold [incl. lazy class import] / warm | tree: modules / depth / param tensors / params | b. hook trace, meta inputs | b. hook trace, FakeTensorMode | b. `torch.fx` | b. `torch.export` (meta) | d. `unfold()` cold / warm (`inspect_code=True`) |
|---|---|---|---|---|---|---|---|
| Llama-7B | 0.33 s [10.7 s] / 0.02 s | 423 / 5 / 291 / 6.738 B | **FAIL** `Tensor.item()` at `masking_utils.py:783` | **OK** 422 calls, 0.29 s | FAIL `co_varnames is too small` (py3.12 + HF decorators); HF's own `transformers.utils.fx` no longer exists in v5 | **OK** 2.3 s, 2434 nodes, 2135 with `stack_trace` | 16.8 s / 0.39 s |
| DeepSeek-V3 | 0.45 s [10.2 s] / 0.04 s | 1215 / 6 / 909 / 671.0 B | FAIL (same) | **OK** 1214 calls, 1.04 s — 58×`DeepseekV3MoE` + 58×`TopkRouter` + 58×`NaiveMoe` + 61×`MLP` executed | FAIL (same) | **OK** 9.9 s, 11 721 nodes | 22.2 s / 1.28 s |
| Qwen2-VL-7B (text path) | fake 0.21 s [8.4 s] | 703 / 6 / 730 / 8.291 B (both towers) | **OK** 371 calls, 0.12 s (this model's mask path never hits `.item()`) | **OK** 371 calls, 0.33 s | FAIL (same) | FAIL round 1 only because `use_cache` was stripped → `DynamicCache` in output; not re-run | 85.2 s / 2.00 s |
| Qwen2-VL vision tower | real CPU instantiate 12.9 s (676 M params, random init) | 32 × `Qwen2VLVisionBlock{norm1, attn: VisionAttention{qkv, proj}, norm2, mlp: VisionMlp}`, `PatchEmbed(Conv3d)`, `VisionRotaryEmbedding`, `PatchMerger` | meta pixels + CPU grid: FAIL device mismatch; all-meta: FAIL `.tolist()` at `vision_utils.py:74` | FAIL `DynamicOutputShapeException: aten.repeat_interleave` | — | real CPU: FAIL `GuardOnDataDependentSymNode` at `vision_utils.py:75` | (same row) |
| **real CPU hook trace** of the vision tower | — | — | **OK** 330 calls, 0.31 s; class histogram `qwen2vl.r2.json` | — | — | — | — |
| Qwen3.5-27B (bonus; `layer_types` 48 linear / 16 full) | 0.53 s / 0.04 s | 1014 modules called | FAIL (same) | **OK** 1014 calls, 4.26 s; per-layer class + mask kwarg recorded | — | **OK** 45.5 s, 51 383 nodes | (not in the 5) |
| PixArt-Σ | `Transformer2DModel.from_config` 0.04 s [9.1 s] → **returns `PixArtTransformer2DModel`** | 663 / 6 / 603 / 0.611 B | **OK** 578 calls, 0.19 s | — | — | **OK** 1.8 s, 2410 nodes | 37.1 s / 0.69 s |
| SD3.5-large | 0.16 s [10.9 s] / 0.06 s | 1456 / 6 / 1226 / 8.057 B | **OK** 1342 calls, 0.40 s | — | FAIL `TraceError: control flow` | **OK** 4.3 s, 6017 nodes (38 SDPA, 536 linear, 152 layer_norm, 152 rms via pow/mean/rsqrt) | 56.9 s / 1.06 s |
| T5-small (bonus, real CPU, 60 M) | — | 266 module calls | **OK** 0.04 s | — | — | **OK** 1.1 s, 1042 nodes | — |

Logs: `<target>.log`, `<target>.r2.log`; structured results `<target>.json`,
`<target>.r2.json`; param shapes `<target>.param_shapes.json`; full call traces
`<target>.*.trace.json`; `time_unfold.json`.

### 2.2 (a) Meta-device instantiation — what it yields for free

Every one of the five targets constructs on `torch.device("meta")` in
0.04–0.53 s once the library is imported (0.02–0.06 s on a warm second
build). The sibling audit's "0.02–0.3 s" (`first-principles-judgment.md:50`,
register row A6) is therefore **true after import and false for the first
call in a process**: the first `from_config` pays 8–11 s of lazy
`transformers`/`diffusers`/`torch` sub-imports (`llama.log` vs `llama.r2.log`).
A long-lived server (the Space is one gradio process, `hf/app.py:117,126`)
pays it once.

What the instance gives, with no forward and no weights:

- **The complete module tree with real names and classes** —
  `model.layers.0.self_attn.q_a_layernorm: DeepseekV3RMSNorm` and
  `kv_a_proj_with_mqa` simply exist (`deepseek.log` line 3). The ×N is the
  `ModuleList` length. Depth is 5–6 for all five.
- **Every parameter shape** (`*.param_shapes.json`): DeepSeek's fused experts
  are `gate_up_proj [256, 4096, 7168]`, `down_proj [256, 7168, 2048]` — the
  expert-axis question the DBRX finding raises (`08-findings-register.md`,
  register A6 text) is answered by the tensor shape, not by a header.
- **Which guarded branch `__init__` took**: DeepSeek `first_k_dense_replace=3`
  → 3×`DeepseekV3MLP` + 58×`DeepseekV3MoE` (`deepseek.log:3`); SD3.5
  `context_pre_only` = `[False]*37 + [True]` (`sd35.log:3`); PixArt children
  `{pos_embed, transformer_blocks, norm_out, proj_out, adaln_single,
  caption_projection}` + `scale_shift_table` parameter, attention processors
  `AttnProcessor2_0` (`pixart.r2.log`); Qwen2-VL `visual.blocks[0].attn` is
  `VisionAttention{qkv, proj}` with `_attn_implementation='eager'`
  (`qwen2vl.log:3`).
- **One correction the static path cannot make**: for
  `_class_name: Transformer2DModel` with `norm_type: ada_norm_single`,
  diffusers 0.38 never builds `Transformer2DModel` at all — `from_config`
  remaps to `PixArtTransformer2DModel`
  (`diffusers/models/model_loading_utils.py:55-60` `_CLASS_REMAPPING_DICT`,
  applied at `modeling_utils.py:2115`). The project resolves the source bundle
  to `transformers/transformer_2d.py` (verified:
  `_diffusers_class_file("Transformer2DModel")` and `resolve_source_files(cfg)`
  both return `transformer_2d.py`; `sources.py:432-450`). The static path reads
  a class the runtime does not construct. The two classes are structurally
  similar, so the drawn diagram may still be right — but its provenance points
  at the wrong file. (Loose end L1: I did not diff the two module trees; my
  direct-ctor probe of the legacy class failed on my own kwarg filter, see L1.)

What it cannot yield: anything about `forward` — op order, residual taps,
where the mask or rotation is applied, gating algebra. It also does not tell
you which constructed modules are *reached*; that is the trace's job.

### 2.3 (b) Forward capture — what runs, what resolves, what does not

**Plain meta tensors are not enough for transformers LLMs.** Llama, DeepSeek
and Qwen3.5 die three modules in at
`masking_utils.py:783`
`if not is_tracing(packed_sequence_mask) and (packed_sequence_mask[:, -1] == 0).all():`
because `.all()` needs `.item()`. The guard is the fix: `is_tracing()`
(`utils/import_utils.py:1527-1537`) returns True for a FakeTensor. Building the
model and inputs under `FakeTensorMode(allow_non_fake_inputs=True)` makes all
three run to completion (`*.r2.log b_hook_fake`), 0.3–4.3 s. Diffusers models
have no such guard and run on bare meta.

What the FakeTensor hook trace resolves — the exact cases the brief names as
static-analysis failures:

- **Qwen3.5 mask ternary** (`modeling_qwen3_5.py:1201`
  `layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask`):
  the trace records, per layer, the module class and the `attention_mask`
  kwarg it received — 48×`Qwen3_5GatedDeltaNet` (`linear_attn`, mask `None`)
  and 16×`Qwen3_5Attention` (`self_attn`, mask `(1,1,8,8)`) at layers
  3, 7, 11, …, 63 (`qwen35.r2.json:qwen35_layer_attn`). This is the
  per-layer schedule the project's `mixer_schedule.py`/`position_schedule.py`
  (565 + 511 lines) reconstruct statically.
- **T5 loop-carried bias** (`modeling_t5.py:297-327`): only
  `encoder.block.0.layer.0.SelfAttention` and `decoder.block.0.layer.0.SelfAttention`
  own `relative_attention_bias`; block 0 receives `position_bias=None`, blocks
  1–5 receive `(1, 8, 6, 6)` — the carried tensor, visible as a kwarg
  (`t5.r2.json:t5_position_bias_kwarg_per_attention`). Register row B6 ("T5
  g1 silent negative") is answerable from a 0.04 s trace.
- **DeepSeek MoE dispatch**: under FakeTensorMode the router and the fused
  `NaiveMoe` execute for all 58 MoE layers (one harmless `data_ptr()`
  deprecation warning from `integrations/moe.py:284`). No `ShapeEnv` was
  needed.
- **PixArt guarded inits** (`transformer_2d.py:168-173`): moot — see 2.2; the
  constructed class has no `is_input_*` flags at all, and the trace runs
  (578 calls).
- **Qwen2-VL `VisionAttention`** (`modeling_qwen2_vl.py:354-400`): the vision
  tower is *not* traceable on meta or fake tensors — `get_vision_position_ids`
  does `grid_thw.tolist()` (`vision_utils.py:74`) and `repeat_interleave` on
  data (`DynamicOutputShapeException`). It **is** traceable on real CPU tensors
  with the real 7B vision config (676 M random params, 12.9 s to build, 0.31 s
  to trace 330 module calls). The attention dispatch itself
  (`ALL_ATTENTION_FUNCTIONS.get_interface`, `:388`) is resolved by the trace
  only as "the module ran"; the *function* chosen is a functional call and
  appears in `torch.export` (which fails here) or in the static reader.

What the trace cannot see, measured:

- Branches not taken by the chosen inputs. A text-only `input_ids` call never
  enters `visual` (371 calls, none in `model.visual`, `qwen2vl.log`). One input
  recipe per modality is required, and the recipe is a *choice* that must be
  recorded with the result.
- Data-dependent code: `.tolist()`, `.item()`, `nonzero`, `repeat_interleave`
  with tensor repeats. Two of seven targets hit it (Qwen2-VL vision; Llama-class
  masks until FakeTensorMode). Real-CPU tracing of one sub-tower is the
  fallback and is cheap when the tower is < 1 B params.
- Functional ops between modules (residual add, RoPE math, SDPA) — hooks see
  modules only; `torch.export` sees ops.

**`torch.fx.symbolic_trace` is dead for this purpose**: every HF target fails
before tracing starts (`ValueError: code: co_varnames is too small` — fx's
code-object patching against the v5 forward decorators on Python 3.12), and
transformers 5 removed its own `HFTracer` (`ImportError: cannot import name
'fx' from 'transformers.utils'`). SD3.5 fails on control flow.

**`torch.export` on meta works for 5 of 7** once `use_cache=False` is passed
(2–46 s, graph sizes 1 k–51 k nodes), and carries what nothing else does:
`nn_module_stack` = `[('', 'transformers.models.llama.modeling_llama.LlamaForCausalLM'), ('model', …LlamaModel), ('model.embed_tokens', torch.nn.modules.sparse.Embedding)]`
and a `stack_trace` ending in `modeling_llama.py, line 389, in forward:
inputs_embeds = self.embed_tokens(input_ids)` on 88 % of nodes
(`llama.r2.json`). That is module path ↔ source line ↔ op, machine-readable.
Fails on the Qwen2-VL vision tower (data-dependent, `vision_utils.py:75`).
Cost grows with unrolled depth (Qwen3.5 64 layers → 45 s); it is a deep
witness for conformance, not the per-request path.

### 2.4 (c) Checkpoint headers

The local HF cache holds **0 `.safetensors` weight files** and 2
`model.safetensors.index.json` files (gemma-3n-E2B-it; PixArt-LCM text
encoder); the corpus targets have `config.json` only. An index file gives
tensor *names* and a total byte size, **no shapes** (gemma-3n: 1556 keys, e.g.
`model.audio_tower.conformer.0.attention.attn.per_dim_scale`; verified). Shapes
need the safetensors header; `HfApi.get_safetensors_metadata` fetches it by
HTTP range request (verified in its source) — a network read, **skipped by
this pass's rule**. Consequence for DBRX expert axes: unverified here; but
2.2 shows the same question is answered offline by the meta instance
(`experts.mlp.w1` shape on the instantiated `DbrxExperts`) — the header is a
*third witness for shapes*, not the only one.

### 2.5 (d) Static AST — the project's current physics, timed and profiled

`time_unfold.json`, fresh subprocess per row, `return_json=True`:

| model | cold (`inspect_code` False / True) | warm second call | HTML render | JSON nodes / bytes | torch imported as side effect |
|---|---|---|---|---|---|
| llama-7b | 22.6 / 16.8 s | 0.54 / 0.39 s | 1.0 s, 55 KB | 663–763 / 10–13 KB | yes |
| deepseek-v3 | 19.9 / 22.2 s | 0.92 / 1.28 s | 1.9–2.7 s, 154 KB | 1754–1946 / 29–35 KB | yes |
| qwen2-vl-7b | 81.7 / 85.2 s | 1.92 / 2.00 s | 3.9–4.0 s, 106 KB | 1096–1243 / 26–31 KB | yes |
| pixart-sigma | 41.3 / 37.1 s | 0.63 / 0.69 s | 1.3 s, 121 KB | 172–202 / 3–4 KB | yes |
| sd3.5-large | 55.9 / 56.9 s | 0.97 / 1.06 s | 2.0–2.1 s, 159 KB | 196–258 / 3–5 KB | yes |

Three facts about that cold time:

1. **It is one quadratic call site, not "static analysis".** cProfile of a cold
   Llama `unfold()` (20.5 s): 18.1 s cumulative in `ast.get_source_segment`,
   17 149 calls from `ProgramIndex._seg`
   (`evidence/program_index.py:2356-2362`); CPython's
   `_splitlines_no_ff` re-splits the whole file per call (16.9 s tottime). A
   precomputed line table makes this O(node). The remaining ~4 s is `torch`
   being imported through `transformers` (`torch/_ops.py`, `_jit_internal`
   frames in the profile). Register row A7 ("cold 30–100 s") is real but
   mis-attributed if read as inherent.
2. **`inspect_code` is not a switch on the evidence layer.** The Space calls
   `unfold(model_id, token=token)` (`hf/app.py:117`) with the default
   `inspect_code=False` (`__init__.py:58`), and the profile shows the full
   `ProgramIndex` build regardless (`parser.py:396` only gates the *report*).
   So production already runs all of `evidence/`.
3. **It runs without torch** — verified by poisoning `sys.modules["torch"]`:
   `unfold(llama, inspect_code=True)` returns in 17.5 s with `code_evidence`
   present (transformers prints "PyTorch was not found" and still serves
   `AutoConfig` and its own `__file__`, which is all `sources.py:112-122`
   needs; transformers 5.12.1's pip metadata lists no torch requirement). The
   `dependencies = []` claim (`pyproject.toml:15`) is honest for the AST path.

What static AST uniquely provides, and no prototype above replaces:

- **Provenance to source lines for every fact**, the `SourceSpan`
  (`program_index.py:2349-2354`), on the *construction* side as well as the
  forward side. Instantiation gives the class of an instance;
  `inspect.getsourcelines(type(m))` gives the class span; but *which line
  assigned `self.q_a_layernorm`*, and *under which `if`*, is static.
- **Config-path binding**: that `self.num_heads` came from
  `config.text_config.num_attention_heads` via a constructor formal
  (`evidence/config_access.py`, `construction_arguments.py`,
  `constructor_fields.py`). No trace carries this; the instance holds only the
  *value*.
- **Provable negatives at the mechanism level** ("no QK-norm in this
  attention's forward") from the forward AST — a trace shows what ran, not
  that nothing else could.
- **Forward semantics without any input recipe**, and without executing
  anything — the only route for remote-code repos the user has not trusted.

---

## 3. Hybrid design — the right physics per tier

### 3.1 Authorities

| tier | authority | second witness | provenance kept how |
|---|---|---|---|
| **1 — structure** (components, stacks, ×N, block classes, sub-blocks, parameter shapes, constructed branches, *reached* modules, per-layer schedules) | **meta-device instance** (`from_config` under `torch.device("meta")`, `attn_implementation="eager"`) enumerated by `named_modules()` / `named_parameters()`; **FakeTensor hook trace** for reachability, call order and per-call kwargs (mask / bias / cache presence) | `torch.export` graph (`nn_module_stack`) where it succeeds; safetensors header shapes when a checkpoint is present (network-gated) | module path → `type(m)` → `inspect.getsourcefile/lines` → the static `ProgramIndex` span of that class; the *construction line* of each child comes from the existing constructor readers scoped to that one class |
| **2 — mechanism** (op order, residual taps, position application, gating algebra, routing, masks, norms) | **the existing static forward readers** (`evidence/attention*.py`, `position_*.py`, `ffn_mechanism.py`, `router.py`, `cell_topology.py`, …) run over the `forward` of the **exact class of the exact instance** | the trace arbitrates *which branch executed* (kwarg present/absent, which child ran); `torch.export` op histogram for conformance (`aten.scaled_dot_product_attention` ×38 on SD3.5 is a checkable count) | unchanged: `SourceSpan` per fact; unknown chips with reasons stay exactly as the doctrine says (`00-goal-and-intent.md:167`) |
| **3 — values** | the **config document** (path-bound, `config_access.py`) **plus instance attributes** after `__init__` arithmetic (`head_dim`, `num_key_value_groups`, `scaling`) — the instance is the arithmetic already done | parameter shapes (tier 1) must agree with the values; disagreement is a loud finding | config path for the input; class + line of the assignment for the derived value |

### 3.2 How the honesty doctrine survives

- **Instantiation enumerates; it never classifies a model.** The identity law
  (`feedback-detect-from-evidence-never-identity`) concerns *model* identity
  selecting structure. `named_modules()` is not identity: it is the object
  the checkpoint's own code built from the checkpoint's own config. Reading
  `type(m).__name__` to *label* a box is display, which the identity roles
  already permit (`evidence/identity_roles.py`). Reading
  `isinstance(m, nn.LayerNorm)` to say "LayerNorm" is a primitive's
  definition, which the audit already argued should be lawful
  (`first-principles-judgment.md:3.2`).
- **Unknown stays confined to tier 2**, as agreed. A module that exists but
  whose forward the readers cannot prove is drawn (tier 1, from the instance)
  with a mechanism chip — the exact split `00-goal-and-intent.md:165-167`
  asks for, and one the current single-physics design cannot make: today an
  unproven *construction* becomes a pale or missing box.
- **The trace never asserts a negative.** "Not reached with this input
  recipe" is recorded with the recipe; "does not exist" comes only from the
  instance; "never applied in `forward`" comes only from the static reader.
- **Every tier-1 element gets a recall gate for free**: `named_modules()` is
  the denominator. Drawn ∪ chipped must equal the instance's reached set —
  this is the "no waiver without a chip" reverse-conformance the audit wants
  (`first-principles-judgment.md:5.3`), and it is a set difference, not a
  reader.

### 3.3 Trust and remote code

- **Installed `transformers`/`diffusers` are already executed** by the
  project: `parser.py:511` imports `AutoConfig`, which imports the library,
  which imports torch when present (measured: `torch_imported: true` on every
  `unfold()` row). Running `LlamaForCausalLM.__init__` from the same wheel is
  not a new trust boundary. Only 1 of 466 installed modeling files reaches for
  weights or the network inside `__init__` (`timm_backbone/modeling_timm_backbone.py:54`
  `timm.create_model`), and 0 of 142 diffusers model files (AST scan,
  this pass).
- **Remote code** (`auto_map` / `trust_remote_code`) is a policy switch:
  default off → tier 1 falls back to the static constructor readers with a
  visible "structure not verified by construction" chip; opt-in → instantiate
  in a subprocess with network disabled and a wall-clock cap. The project's
  current stance "remote code is read, never executed" (`parser.py:489`)
  becomes the default, not the only mode.
- **Reproducibility**: the meta instance is deterministic (no RNG affects
  structure; `_init_weights` is skipped on meta); the trace is deterministic
  for a fixed (library version, config, input recipe, `attn_implementation`).
  Record all four in the receipt, as `hash_signature` records the source today.

### 3.4 Latency that results (per request, warm process)

`import torch`+`transformers` ≈ 3 s once per process; instantiate 0.03–0.5 s;
FakeTensor trace 0.3–4 s (Qwen3.5 the worst at 4.3 s); static tier-2 readers
over the reached classes — today's warm figure, 0.4–2.0 s, minus the
ownership substrate they no longer need; `torch.export` 2–46 s **optional**,
run by Sable as a conformance witness, not by the page. Order of magnitude:
**1–5 s warm per model**, versus 0.4–2.0 s warm today *plus* a 17–85 s cold
penalty that, independently of this design, should be a ~1 s fix
(`program_index.py:2359`).

### 3.5 What happens to `evidence/` (147 files, 95 837 lines = 72 % of the 133 052-line package)

Grouped by file name into the three questions each file answers
(`evidence_files.txt`; grouping is mine, by docstring, and is an estimate):

| bucket | files | lines | fate |
|---|---|---|---|
| **A. construction / ownership** — `program_index.py` (2929), `component_owner.py` (2155), `config_access.py` (1546), `layer_selector.py`, `constructor_*`, `construction_*`, `config_scoped_owner.py`, `container_inventory.py`, `execution_flow.py`, `structural_writes.py`, `structural_debt.py`, `identity_guard.py`, `expert_storage.py`, `attention_storage.py`, `attention_child.py`, `attention_geometry.py`, all `unet_*`, `diffusion_stack.py`, `diffusion_root.py`, `projector_*`, `*_schedule.py`, `dispatch_*`, `fusion.py`, `mtp.py`, `weight_tying.py` … | 89 | 54 639 | **mostly deleted**: the instance answers "what exists, where, ×how many, what shape, which branch, in what order". **Kept and shrunk**: `program_index.py` (as the span/provenance index only — no ownership resolution), `config_access.py` + `construction_arguments.py` + `constructor_fields.py` (config-path binding, now scoped to one class per instance), `sources.py`. Estimate: 54.6 k → **8–12 k** |
| **B. forward mechanism** — `attention.py` (5324), `attention_mask.py`, `cell_topology.py`, `router.py`, `ffn_mechanism.py`, `position_*.py`, `qk_norm.py`, `decoder_norm.py`, `diffusion_stream/conditioning/block/bookends.py`, `vision.py`, `forward_ops.py`, `transitive.py`, `patterns.py` … | 41 | 32 439 | **kept, re-pointed**: input becomes `(class, forward AST, instance attrs, trace kwargs)` instead of `(guessed owner occurrence)`. The "external container", "bundle boundary" and `mro_incomplete` refusals (register A1–A5) disappear because the class is known, not searched for. Expect a modest shrink from deleted owner-resolution preambles: 32 k → **25–30 k** |
| **C. verification / plumbing** — `conformance.py` (2106), `registry.py`, `facts.py`, `receipts.py`, `consumer_firewall.py`, `claims_audit.py`, `legacy_reader_quarantine.py`, `document.py`, `context.py`, `models.py` … | 17 | 8 759 | **kept**; conformance gains the trace and the export op histogram as inputs; `legacy_reader_quarantine.py` and `structural_debt.py`'s "external" classes of debt become empty |

Plus **new**: ~1–2 k lines — `instantiate.py` (meta/fake build, input recipes
per modality, subprocess + trust gate), `trace.py` (hooks, kwarg capture,
reached-set), `provenance.py` (module path ↔ class span ↔ config path). Net
estimate **≈ 45–55 k lines of evidence** (from 96 k), with the biggest single
deletions being the ownership substrate U3 built (`component_owner.py`,
`layer_selector.py`, `config_scoped_owner.py`, `execution_flow.py`, the seven
`unet_selected_*`/`unet_stage_*` readers). Confidence in the bucket sizes:
moderate (names and docstrings, not a dependency cut); in the direction: high.

---

## 4. Risks and counter-arguments, honestly

1. **"Model code is not executed."** It already is: `import transformers`
   executes the library, and `torch` gets imported on every request today.
   `__init__` on meta allocates no memory and reads no weights (all five
   targets, up to 671 B parameters, in < 0.6 s, `param_on_meta: true`). The
   real objection is *remote code*, and that is a switch (3.3), not a reason
   to reconstruct `nn.ModuleList` by parsing Python.
2. **Version coupling.** Identical for both physics: static reads the
   installed source; instantiation runs it. The remap in 2.2 shows the
   installed *runtime* can differ from the installed *source you chose to
   read* — coupling to the runtime is the safer side.
3. **`__init__`s that need weights or network.** Measured: 1/466 transformers,
   0/142 diffusers. Custom-code repos are unmeasured (L3).
4. **Data-dependent forwards.** Two of seven targets needed a workaround
   (FakeTensorMode for the HF mask guard; real-CPU tracing for Qwen2-VL
   vision). Real-CPU tracing needs the tower to fit (676 M params = 2.7 GB
   fp32 here; a 5 B vision tower would not on a small Space). Fallback: the
   static readers, with a "not traced" chip — the current physics, honestly
   labelled.
5. **Input recipes are a hidden identity table.** "Feed `input_ids` for
   causal LMs, `hidden_states`+`encoder_hidden_states`+`timestep` for DiTs" is
   per-family knowledge. It must live in `everchanging/` YAML keyed by
   *signature shape* (which formals `forward` declares — readable from the
   instance), never by `model_type`. Recipes can be wrong silently (a branch
   never entered looks like "absent"); mitigated by 3.2 — the trace never
   asserts absence.
6. **`torch.fx` is not an option** (2.3) and `torch.export` is slow and
   fragile on data-dependent towers; treat it as a Sable witness only.
7. **Determinism of the `.item()`-style guards**: `is_tracing()` behaviour is
   a transformers-internal contract (`import_utils.py:1527`); a future version
   could route differently. The mitigation is the same receipt discipline as
   today's `hash_signature`.
8. **The sibling claim "0.02–0.3 s"** understates first-call cost by ~10 s of
   lazy imports; per-process, not per-request. Stated in 2.2.
9. **Sunk cost is real**: bucket A is the product of U3–U11 and several vet
   rounds. Deleting it is the point (the audit's obligation, `first-principles-judgment.md:5.1`),
   but the deletion must be paired with a recall gate so that no verified
   detail is lost again (register B4). The instance is that gate's
   denominator.
10. **Bucket estimates are by name.** A real dependency cut could move
    5–10 k lines between A and B.

---

## 5. Recommendation

Adopt the split in §3: **tier 1 from the meta instance + FakeTensor trace,
tier 2 from the existing static forward readers pointed at exact classes,
tier 3 from config paths + instance attributes**, with `torch.export` as an
optional Sable witness and safetensors headers as an optional third shape
witness. Before any of that, fix `program_index.py:2359` (line-table instead
of `get_source_segment`) — it alone removes most of the 17–85 s cold time and
is independent of the design decision.

Confidence: **high (≈0.85)** that instantiation + trace is the correct
authority for tier 1 and that static readers remain correct for tier 2 — the
five targets plus the two named control-flow cases all resolved, and the one
class remap (PixArt) went the instance's way. **Moderate (≈0.6)** on the
evidence-layer size estimate and on the claim that the ownership substrate can
be deleted wholesale without losing config-path provenance on constructor
arguments; that needs a dependency cut, not a docstring grouping. **Low**
confidence on custom-code repos, which were not measured.

---

## Loose ends

- **L1** PixArt legacy vs modern class: my direct `Transformer2DModel(**kw)`
  probe failed on *my* kwarg filter (`__init__.__code__.co_varnames` of the
  `@register_to_config` wrapper), so I did not diff the legacy and modern
  module trees. The remap itself is verified (`model_loading_utils.py:55-60`).
- **L2** Qwen2-VL full-model export and a text-plus-image FakeTensor recipe
  were not attempted; the vision tower was traced standalone on real CPU.
- **L3** No custom-code (`trust_remote_code`) repo was instantiated; the
  1/466 network-in-`__init__` figure covers installed transformers only.
- **L4** DBRX expert axes via safetensors header: skipped (network). The
  meta-instance route was not run for DBRX either (not in the five).
- **L5** The evidence-layer bucket assignment (`evidence_files.txt`) is by
  file name and docstring; `02-pipeline-trace.md` / `01-codebase-map.json`
  dependency data should replace it.
- **L6** `unfold()` warm timings were taken while the round-1 prototype ran
  in another process; cold numbers were taken alone. Re-run on a quiet
  machine before quoting to the decimal.
- **L7** Landscape rows are the delegated subagent's web findings; I verified
  only the torch.fx / torch.export / meta-device claims against the local
  install. Model Explorer's display of `stack_trace`, torchinfo-on-meta, and
  all "last active" dates are as reported.
- **L8** The `is_tracing()`-guard workaround (FakeTensorMode) is a
  transformers-internal contract; its stability across 5.x was not checked.
- **L9** Register rows A1–A7 and B6 are addressed above but not yet marked
  accepted/voided; that is `08-findings-register.md`'s job.

---

## Amendment 2026-09-02

The conclusion "the instance should become tier-1 authority and ≈55k lines
can mostly be deleted" is **overreach**; the experiment stands, the authority
decision does not. Additional limits recorded from the counter-review:
`named_modules()` includes constructed-but-unused modules; hooks do not see
functional ops (RoPE, residual adds, masks, SDPA, reshapes); tracing covers
only the recipe's paths; class-span provenance is not construction-site or
config-path provenance; "every module drawn" conflicts with block-worthiness.
Hence the reconciliation contract in `12` §1.
