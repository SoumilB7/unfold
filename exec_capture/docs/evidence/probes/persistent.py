import sys, time, resource, gc
T0 = time.time(); peak = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
import torch, transformers
sys.path.insert(0, "."); from lowcost import zero_storage, Recorder, fetch_cached
from transformers import CONFIG_MAPPING
print(f"process start + imports: {time.time()-T0:.2f}s, peak {peak()} GB")
for repo in sys.argv[1:]:
    t = time.time(); cfgd, T, src = fetch_cached(repo); tm = time.time() - t
    cfg = CONFIG_MAPPING[cfgd["model_type"]](**{k: v for k, v in cfgd.items() if k != "model_type"})
    cls = getattr(transformers, cfgd["architectures"][0])
    t = time.time()
    with zero_storage(): m = cls._from_config(cfg, dtype=torch.bfloat16)
    m.eval(); ids = {id(p): n for n, p in m.named_parameters(remove_duplicate=False)}; r = Recorder(ids)
    with torch.no_grad(), r: m(input_ids=torch.zeros((1, 16), dtype=torch.long), use_cache=False)
    P = dict(m.named_parameters()); cov = sum(P[n].numel() for n in r.used if n in P) / sum(p.numel() for p in P.values())
    print(f"  {repo:36s} metadata ({src}) {tm:5.2f}s | build+forward {time.time()-t:5.2f}s | params {sum(p.numel() for p in P.values())/1e9:7.1f}B used {cov:.4f} | peak so far {peak()} GB")
    del m, P, r, T; gc.collect()
print(f"all done in {time.time()-T0:.1f}s")
