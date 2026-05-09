# unfold

> One line of Python turns a HuggingFace transformer config into a clear, interactive architecture diagram. Inline in Jupyter.

```python
from unfold import unfold
unfold("mistralai/Mistral-7B-v0.3")
```

<p align="center">
<svg width="540" viewBox="0 0 720 920" role="img" xmlns="http://www.w3.org/2000/svg"><title>Mistral-7B-v0.3 architecture</title><defs><marker id="uf-readme-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs><rect x="40" y="26" width="640" height="868" rx="18" ry="18" fill="#E1F5EE"/><rect x="110" y="200" width="500" height="490" rx="18" ry="18" fill="#9FE1CB"/><g><title>Tokenized text: shape [batch, seq_len]</title><rect x="250" y="820" width="220" height="44" rx="11" ry="11" fill="#0F6E56" stroke="#0E5C48" stroke-width="0.6"/><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="17" x="360" y="842">Tokenized text</text></g><g><title>Token embedding: 32,768 x 4,096</title><rect x="230" y="752" width="260" height="44" rx="11" ry="11" fill="#0F6E56" stroke="#0E5C48" stroke-width="0.6"/><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="17" x="360" y="774">Token Embedding layer</text></g><g><title>RMSNorm; dim 4,096</title><rect x="280" y="600" width="160" height="36" rx="11" ry="11" fill="#0F6E56" stroke="#0E5C48" stroke-width="0.6"/><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="16" x="360" y="618">RMSNorm</text></g><g><title>Grouped-query attention; 32 Q / 8 KV heads</title><rect x="245" y="505" width="230" height="60" rx="11" ry="11" fill="#0F6E56" stroke="#0E5C48" stroke-width="0.6"/><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="17" x="360" y="525">Grouped-Query</text><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="17" x="360" y="545">Attention</text></g><g><title>Residual add</title><circle cx="360" cy="470" r="14" fill="#0F6E56" stroke="#0E5C48" stroke-width="0.6"/><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="22" x="360" y="471">+</text></g><g><title>RMSNorm; dim 4,096</title><rect x="280" y="395" width="160" height="36" rx="11" ry="11" fill="#0F6E56" stroke="#0E5C48" stroke-width="0.6"/><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="16" x="360" y="413">RMSNorm</text></g><g><title>Feed-forward (gated SiLU); hidden 14,336</title><rect x="280" y="310" width="160" height="44" rx="11" ry="11" fill="#0F6E56" stroke="#0E5C48" stroke-width="0.6"/><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="17" x="360" y="332">Feed-Forward</text></g><g><title>Residual add</title><circle cx="360" cy="275" r="14" fill="#0F6E56" stroke="#0E5C48" stroke-width="0.6"/><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="22" x="360" y="276">+</text></g><g><title>Final RMSNorm</title><rect x="270" y="140" width="180" height="36" rx="11" ry="11" fill="#0F6E56" stroke="#0E5C48" stroke-width="0.6"/><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="16" x="360" y="158">Final RMSNorm</text></g><g><title>LM head: 4,096 -> 32,768</title><rect x="230" y="70" width="260" height="44" rx="11" ry="11" fill="#0F6E56" stroke="#0E5C48" stroke-width="0.6"/><text text-anchor="middle" dominant-baseline="central" fill="#FFFFFF" font-family="cursive" font-size="17" x="360" y="92">Linear output layer</text></g><line x1="360" y1="820" x2="360" y2="802" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" marker-end="url(#uf-readme-arrow)" fill="none"/><line x1="360" y1="752" x2="360" y2="642" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" marker-end="url(#uf-readme-arrow)" fill="none"/><line x1="360" y1="600" x2="360" y2="571" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" marker-end="url(#uf-readme-arrow)" fill="none"/><line x1="360" y1="505" x2="360" y2="490" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" marker-end="url(#uf-readme-arrow)" fill="none"/><line x1="360" y1="456" x2="360" y2="437" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" marker-end="url(#uf-readme-arrow)" fill="none"/><line x1="360" y1="395" x2="360" y2="360" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" marker-end="url(#uf-readme-arrow)" fill="none"/><line x1="360" y1="310" x2="360" y2="295" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" marker-end="url(#uf-readme-arrow)" fill="none"/><line x1="360" y1="261" x2="360" y2="182" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" marker-end="url(#uf-readme-arrow)" fill="none"/><line x1="360" y1="140" x2="360" y2="120" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" marker-end="url(#uf-readme-arrow)" fill="none"/><line x1="360" y1="70" x2="360" y2="38" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" marker-end="url(#uf-readme-arrow)" fill="none"/><path d="M 440 618 L 570 618 Q 582 618 582 606 L 582 482 Q 582 470 570 470 L 380 470" fill="none" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#uf-readme-arrow)"/><path d="M 440 413 L 570 413 Q 582 413 582 401 L 582 287 Q 582 275 570 275 L 380 275" fill="none" stroke="#0F6E56" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#uf-readme-arrow)"/><rect x="532" y="212" width="66" height="26" rx="13" ry="13" fill="rgba(255,255,255,0.65)" stroke="#B6DDCB" stroke-width="0.5"/><text text-anchor="middle" dominant-baseline="central" fill="#04342C" font-family="cursive" font-size="20" x="565" y="225">× 32</text></svg>
</p>

<sup><i>Static preview of the architecture diagram. The live render adds badges, a stats grid, click-to-inspect for every block, and (for heterogeneous models like DeepSeek-V3) a layer-type toggle.</i></sup>

---

## Install

```bash
pip install -e .
pip install transformers   # only required to load by model ID
```

## Three ways to call it

```python
from unfold import unfold

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

Pass a model ID and `unfold` calls `transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)` under the hood ([parser.py](unfold/parser.py)). Anything `AutoConfig` can resolve — public, private, gated, or `trust_remote_code` — works here.

## Auth — token from your environment

Gated models (Llama-3, Mistral, Gemma, …) need a HuggingFace token. `unfold` reuses whatever `transformers` / `huggingface_hub` already see:

```bash
# Either set an env var
export HF_TOKEN="hf_xxxxxxxx"            # also accepted: HUGGING_FACE_HUB_TOKEN

# or use the CLI cache (persists across sessions)
huggingface-cli login

# or load a .env in your notebook
# >>> from dotenv import load_dotenv; load_dotenv()
```

No extra config in `unfold` itself.

## Save / export

```python
diagram = unfold(cfg)
diagram.save("model.html")   # standalone interactive HTML
diagram.save("model.json")   # IR (no rendering)
diagram.param_count()        # {"total": ..., "active": ..., "per_layer": [...]}
diagram.to_ir()              # full IR dict
```

Param estimates are close to published numbers — DeepSeek-V3 reports `~675B (~41B active)`, Llama-3-8B reports `8.03B`.

## Live demos

Open in any browser to interact (click blocks, expand sub-blocks, toggle layer types):

| Model | Highlights | Demo |
|---|---|---|
| Llama-3-8B | GQA + dense baseline | [examples/llama-3-8b.html](examples/llama-3-8b.html) |
| Mistral-7B-v0.3 | GQA + dense, 32k context | [examples/mistral-7b-v0.3.html](examples/mistral-7b-v0.3.html) |
| DeepSeek-V3 | MLA + dense → MoE phase change | [examples/deepseek-v3.html](examples/deepseek-v3.html) |
| Kimi K2 | MLA + 384-expert MoE, ~1T params | [examples/kimi-k2.html](examples/kimi-k2.html) |

## Supported architectures

| Family | Adapter | Notes |
|---|---|---|
| DeepSeek-V2 / V3 / Kimi K2 | [adapters/deepseek.py](unfold/adapters/deepseek.py) | MLA + dense → MoE phase change |
| Llama / Mistral / Qwen2 / Qwen3 / Phi-3 | [adapters/llama.py](unfold/adapters/llama.py) | GQA / MQA / MHA + dense FFN |
| Gemma 2 / 3 | [adapters/llama.py](unfold/adapters/llama.py) | sliding-window pattern detection |

Adding a new architecture: write `matches(cfg)` and `parse(cfg) -> ModelIR` in a new adapter, register it in [adapters/\_\_init\_\_.py](unfold/adapters/__init__.py).

## License

[Apache 2.0](LICENSE).
