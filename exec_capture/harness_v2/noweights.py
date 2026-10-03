"""Hard guard: this process may NEVER download model weights.
Any download of a weight file through huggingface_hub raises. Header reads (HTTP range requests) are unaffected."""
import re
WEIGHT = re.compile(r"\.(safetensors|bin|pt|pth|ckpt|gguf|onnx|onnx_data|msgpack|h5|tflite|mlmodel|npz|pkl)$", re.I)


class WeightDownloadBlocked(RuntimeError):
    pass


def install():
    import huggingface_hub, huggingface_hub.file_download as fd
    orig = fd.hf_hub_download
    def guarded(repo_id, filename, *a, **k):
        name = filename if isinstance(filename, str) else str(filename)
        sub = k.get("subfolder") or ""
        if WEIGHT.search(name):
            raise WeightDownloadBlocked(f"blocked weight download: {repo_id}/{sub + '/' if sub else ''}{name}")
        return orig(repo_id, filename, *a, **k)
    fd.hf_hub_download = guarded
    huggingface_hub.hf_hub_download = guarded
    try:
        import transformers.utils.hub as th
        if hasattr(th, "hf_hub_download"): th.hf_hub_download = guarded
    except Exception:
        pass
    try:
        import diffusers.utils.hub_utils as dh
        if hasattr(dh, "hf_hub_download"): dh.hf_hub_download = guarded
    except Exception:
        pass
    # snapshot_download is also how transformers fetches TOKENIZER files: keep it, but never let a weight file through
    BLOCK = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf", "*.onnx", "*.onnx_data", "*.msgpack",
             "*.h5", "*.tflite", "*.mlmodel", "*.npz", "*.pkl"]
    import huggingface_hub._snapshot_download as sd
    orig_snap = sd.snapshot_download
    def filtered_snapshot(*a, **k):
        ig = k.get("ignore_patterns") or []
        ig = [ig] if isinstance(ig, str) else list(ig)
        k["ignore_patterns"] = ig + BLOCK
        return orig_snap(*a, **k)
    sd.snapshot_download = filtered_snapshot
    huggingface_hub.snapshot_download = filtered_snapshot
    try:
        import huggingface_hub.hf_api as ha
        if hasattr(ha, "snapshot_download"): ha.snapshot_download = filtered_snapshot
    except Exception:
        pass
