"""Phase-by-phase cost of the CURRENT zero-storage mechanism: time, peak RAM, HF cache growth."""
import sys, time, os, resource, json, shutil, subprocess
T0 = time.time()
def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
def cache_mb():
    out = subprocess.run(["du", "-sk", os.path.expanduser("~/.cache/huggingface/hub")], capture_output=True, text=True).stdout.split()
    return int(out[0]) / 1024 if out else 0
repo, mode = sys.argv[1], sys.argv[2]
c0 = cache_mb(); phases = []
def mark(name, t): phases.append((name, round(time.time() - t, 2), round(peak(), 2)))
t = time.time(); import torch, transformers; from huggingface_hub import HfApi; mark("imports", t)
sys.path.insert(0, "."); from common import zero_storage
from torch.utils._python_dispatch import TorchDispatchMode
t = time.time(); cfg = transformers.AutoConfig.from_pretrained(repo); mark("config", t)
t = time.time(); meta = HfApi().get_safetensors_metadata(repo); n = sum(len(f.tensors) for f in meta.files_metadata.values()); mark(f"headers ({len(meta.files_metadata)} files, {n} tensors)", t)
cls = getattr(transformers, cfg.architectures[0])
t = time.time()
with torch.device("meta"): mm = cls._from_config(cfg)
sd = {k: tuple(v.shape) for k, v in mm.state_dict().items()}; del mm; mark("meta husk for matching", t)
t = time.time()
with zero_storage():
    m = cls._from_config(cfg, dtype=torch.float32)
m.eval(); mark("zero-storage build", t)
if mode == "image":
    from PIL import Image
    proc = transformers.AutoProcessor.from_pretrained(repo)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    kw = dict(proc(images=Image.new("RGB", (224, 224)), text=proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), return_tensors="pt"))
    kw = {k: (v.float() if v.is_floating_point() else v) for k, v in kw.items()}
else:
    kw = {"input_ids": torch.zeros((1, 16), dtype=torch.long)}
class Rec(TorchDispatchMode):
    def __torch_dispatch__(s, f, ty, a=(), k=None): return f(*a, **(k or {}))
t = time.time()
with torch.no_grad(), Rec():
    m(**kw, use_cache=False)
mark("real forward", t)
print(json.dumps({"repo": repo, "total_s": round(time.time() - T0, 1), "peak_gb": round(peak(), 2), "hf_cache_growth_mb": round(cache_mb() - c0, 1), "phases": phases}))
