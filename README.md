# unfold

> Turn any HuggingFace transformer into a clear, interactive architecture diagram — inline in Jupyter.

`unfold` reads a model's HuggingFace config and renders an architecture diagram you can actually read. It auto-detects MLA vs GQA vs MHA, dense vs MoE, sliding-window patterns, and the dense → MoE phase change in models like DeepSeek-V3. It estimates parameter counts (with active/total split for MoE).

```python
from unfold import unfold
unfold("moonshotai/Kimi-K2-Instruct")
```

That's it — the diagram renders inline in the notebook.

---

## Install

```bash
pip install -e .
pip install transformers   # optional, only needed to load by model ID
```

## Usage

```python
from unfold import unfold

# By model ID (needs `transformers` installed)
unfold("meta-llama/Meta-Llama-3-8B")

# From a HF config object
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
unfold(cfg)

# From a raw config.json dict (no `transformers` install required)
import json
unfold(json.load(open("config.json")))
```

Outside notebooks:

```python
diagram = unfold(cfg)
diagram.save("model.html")     # standalone interactive HTML
diagram.save("model.json")     # the IR
diagram.param_count()          # {'total': ..., 'active': ..., 'per_layer': [...]}
diagram.to_ir()                # the full IR dict
```

## What you get

The diagram has four views, switchable via tabs:

| View | What it shows |
|------|---------------|
| **Architecture** | The repeating transformer block (norm → attention → norm → FFN, with residuals). Color-coded by role. Click any block to inspect its dimensions. |
| **MoE** | When the model is sparse — router → top-k experts → weighted sum. With a prominent active-per-token callout (e.g. `8/384 · 2.1%`). |
| **FFN / Expert** | The gated SwiGLU/GeGLU block (gate × activation × up → down). |
| **Layer map** | Heterogeneous models (DeepSeek-V3, Gemma) get a stripe-chart showing which layer types appear where. |

A stats banner above the diagram shows layers, hidden size, vocab, context length, and parameter count.

For MoE models, the count splits as `total (active)` — e.g. DeepSeek-V3 reports `~675B (~41.6B active)`.

## Supported architectures

| Family | Adapter | Notes |
|---|---|---|
| DeepSeek-V2 / V3 / Kimi K2 | [adapters/deepseek.py](unfold/adapters/deepseek.py) | MLA + dense → MoE phase change |
| Llama / Mistral / Qwen2 / Qwen3 / Phi-3 | [adapters/llama.py](unfold/adapters/llama.py) | GQA / MQA / MHA dense |
| Gemma 2 / 3 | [adapters/llama.py](unfold/adapters/llama.py) | sliding-window pattern detection |

## Architecture

```
HF config  →  parser  →  IR (ModelIR dataclass)  →  renderer.js  →  inline SVG
```

Three layers, sharply separated:

- **Parser** ([unfold/adapters/](unfold/adapters/)) — per-architecture Python adapters. Each adapter has `matches(cfg)` and `parse(cfg) -> ModelIR`.
- **IR** ([unfold/ir.py](unfold/ir.py)) — dataclasses describing a model as a list of `LayerSpec`s, each carrying its own `AttentionSpec` and `FFNSpec`. Layer-aware so heterogeneous architectures (DeepSeek-V3 dense+MoE, Gemma sliding+full) map onto the same shape.
- **Renderer** ([unfold/static/renderer.js](unfold/static/renderer.js)) — vanilla JS, no build step. Consumes IR JSON, produces SVG.
- **Glue** ([unfold/diagram.py](unfold/diagram.py)) — `Diagram` class. Implements `_repr_html_()`, `save()`, `to_html()`.

## Adding a new architecture

When the parser hits an unsupported config it raises `ValueError`. To add support:

1. Create `unfold/adapters/yourmodel.py` with `matches(cfg)` and `parse(cfg) -> ModelIR`.
2. Register it in [unfold/adapters/__init__.py](unfold/adapters/__init__.py) (most-specific first).
3. Add a smoke test using a real config dict in [tests/test_smoke.py](tests/test_smoke.py).

Use the `_g(cfg, "field_name", default)` helper to read fields — it works for both HF config objects and plain dicts.

## Heterogeneous layers

Each `LayerSpec` carries its own `AttentionSpec` and `FFNSpec`:

- DeepSeek-V3: first 3 layers dense, rest MoE — handled via `first_k_dense_replace`.
- Gemma sliding-window: alternating `mask="sliding"` and `mask="causal"` per `sliding_window_pattern` or `layer_types`.
- Cross-layer KV sharing (Gemma 3n / YOCO / CLA): `attention.kv_source_layer` + `ModelIR.cross_layer_edges`. (IR ready; renderer doesn't visualize yet.)

The "Layer map" view detects repeating patterns via signature-based RLE (`ModelIR.layer_groups()`) and shows colored stripes per layer type.

## Tests

```bash
python tests/test_smoke.py
```

Should print:

```
Kimi K2 OK  — ~1.03T total / 34.9B active
DeepSeek-V3 phase change OK  — ~675B total / 41.6B active
Llama-3 OK  — ~8.03B params

All smoke tests passed.
```

(Param counts are estimates — they don't model every implementation detail, just give the right order of magnitude.)

## Roadmap

Currently v0.2:

- [x] DeepSeek-V3 / Kimi K2 (MLA + MoE)
- [x] Llama / Mistral / Qwen / Phi
- [x] Gemma sliding-window
- [x] Param-count estimation with active/total for MoE
- [x] Layer map for heterogeneous architectures
- [ ] Cross-layer KV-sharing arrows (Gemma 3n / YOCO / CLA)
- [ ] Mamba / SSM block type
- [ ] Layer slider (jump to layer L_i and inspect that layer specifically)
- [ ] Diff view (compare two models side-by-side)
- [ ] PNG export

## License

[Apache 2.0](LICENSE).
