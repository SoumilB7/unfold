import sys, time, resource, os, json
T0 = time.time(); peak = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
import torch, transformers; t_imp = time.time() - T0
sys.path.insert(0, "."); from lowcost import zero_storage, Recorder, fetch_cached
from transformers import CONFIG_MAPPING
repo = sys.argv[1]
t = time.time(); cfgd, T, src = fetch_cached(repo); t_meta = time.time() - t
cfg = CONFIG_MAPPING[cfgd["model_type"]](**{k: v for k, v in cfgd.items() if k != "model_type"})
cls = getattr(transformers, cfgd["architectures"][0])
t = time.time()
with zero_storage(): m = cls._from_config(cfg, dtype=torch.bfloat16)
m.eval(); t_build = time.time() - t
ids = {id(p): n for n, p in m.named_parameters(remove_duplicate=False)}
r = Recorder(ids); t = time.time()
with torch.no_grad(), r: m(input_ids=torch.zeros((1, 16), dtype=torch.long), use_cache=False)
t_fwd = time.time() - t
hsize = sum(os.path.getsize(os.path.join("hdrcache", f)) for f in os.listdir("hdrcache") if f.startswith(repo.replace("/", "__")))
print(f"{repo}: config+headers from {src}: {t_meta:.2f}s | imports {t_imp:.2f}s | build {t_build:.2f}s | forward {t_fwd:.2f}s | total {time.time()-T0:.1f}s | peak {peak()} GB | cached metadata on disk {hsize/1e6:.2f} MB")
