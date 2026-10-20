import collections, warnings; warnings.filterwarnings("ignore")
import transformers, diffusers
from huggingface_hub import HfApi
api = HfApi()
for tag, n in [("text-generation", 300), ("image-text-to-text", 100), ("text-to-image", 100)]:
    c = collections.Counter(); ex = collections.defaultdict(list)
    for m in api.list_models(pipeline_tag=tag, sort="downloads", limit=n, expand=["config", "tags", "library_name", "cardData"]):
        cfg = m.config or {}; tags = set(m.tags or [])
        archs = cfg.get("architectures") or []
        if tag == "text-to-image":
            cls = (cfg.get("diffusers") or {}).get("_class_name") if isinstance(cfg.get("diffusers"), dict) else None
            ok = bool(cls) and hasattr(diffusers, cls); archs = [cls] if cls else []
        else: ok = bool(archs) and all(hasattr(transformers, a) for a in archs)
        if ok or cfg.get("auto_map"): continue
        base = any(t.startswith("base_model:") for t in tags)
        if "gguf" in tags or m.library_name in ("gguf", "ggml", "llama.cpp"): k = "GGUF/ggml re-packaging of another model"
        elif "lora" in tags or m.library_name == "peft" or any(t.startswith("base_model:adapter:") for t in tags): k = "LoRA/adapter on a base model"
        elif archs: k = "architecture class not in installed library (newer version or other lib)"
        elif m.library_name in ("mlx", "onnx", "openvino", "ctranslate2", "mlc-llm", "exllamav2"): k = f"other runtime format ({m.library_name})"
        else: k = "other / unclassified"
        c[k] += 1; ex[k].append((m.id, archs[:1], base))
    print(f"\n== '{tag}': the unrecognized bucket")
    for k, v in c.most_common(): print(f"  {v:3d}  {k:70s} e.g. {[e[0] for e in ex[k][:2]]}{' ' + str([e[1] for e in ex[k][:3]]) if 'class' in k else ''}")
    print(f"       of these, with a 'base_model' link on the Hub: {sum(1 for k in ex for e in ex[k] if e[2])}")
print("\ninstalled: transformers", transformers.__version__, "| diffusers", diffusers.__version__)
