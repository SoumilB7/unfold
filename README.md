# MODEL UNFOLDER

> your one click model unfolder

[![PyPI](https://img.shields.io/pypi/v/model-unfolder)](https://pypi.org/project/model-unfolder/)

```python
from model_unfolder import unfold
unfold("meta-llama/Meta-Llama-3-8B")
```

<p align="center">
  <a href="examples/llama-3-8b.html">
    <img src="examples/images/llama-3-8b.png" width="540" alt="Meta-Llama-3-8B architecture diagram">
  </a>
</p>

---

## Install

```bash
pip install model-unfolder

# for local development
pip install -e .
pip install transformers   # only required to load by model ID
```

## Three ways to call it

```python
from model_unfolder import unfold

# 1) by HuggingFace model ID — only config.json is downloaded, never weights
unfold("meta-llama/Meta-Llama-3-8B")
unfold("deepseek-ai/DeepSeek-V3")

# 2) from a transformers AutoConfig
from transformers import AutoConfig
unfold(AutoConfig.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True))

# 3) from a raw config.json dict — no transformers install needed
import json
unfold(json.load(open("config.json")))
```

## Built on `transformers`

Pass a model ID and `unfold` calls `transformers.AutoConfig.from_pretrained(model_id)` under the hood ([parser.py](model_unfolder/parser.py)). It only retries with `trust_remote_code=True` when Transformers says the config requires remote code.

## Auth-token from your environment

Gated models (Llama-3, Mistral, Gemma, …) need a HuggingFace token. `unfold` reuses whatever `transformers` / `huggingface_hub` already see:

```bash
# Either set an env var
export HF_TOKEN="hf_xxxxxxxx"            # also accepted: HUGGING_FACE_HUB_TOKEN

# or use the CLI cache (persists across sessions)
huggingface-cli login

# or load a .env in your notebook
# >>> from dotenv import load_dotenv; load_dotenv()
```

No extra config in `model_unfolder` itself.

## Save / export

```python
diagram = unfold(cfg)
diagram.save("model.html")   # standalone interactive HTML
diagram.save("model.json")   # IR (no rendering)
diagram.param_count()        # {"total": ..., "active": ..., "per_layer": [...]}
diagram.to_ir()              # full IR dict
```

Param estimates are close to published numbers — DeepSeek-V3 reports `~675B (~41B active)`, Llama-3-8B reports `8.03B`.

## Models supported

### Transformers

| Family | Models |
|---|---|
| DeepSeek | DeepSeek-V2, DeepSeek-V3, Kimi K2 |
| Llama | Llama 3 / 3.1 / 3.2 / 3.3 |
| Mistral | Mistral 7B, Mixtral 8x7B / 8x22B, Mistral Medium 3.5 |
| Qwen | Qwen2 / 2.5 (dense), Qwen2-MoE, Qwen3 (dense), Qwen3-MoE |
| Gemma | Gemma 3, Gemma 4 (31B, E2B, E4B) |

### Diffusors

Coming soon.

### Custom

Drop a request in issues.


## License

[Apache 2.0](LICENSE).
