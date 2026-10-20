import time, collections, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM
for mid in sys.argv[1:]:
    t=time.time()
    cfg=AutoConfig.from_pretrained(mid)
    with torch.device("meta"):
        m=AutoModelForCausalLM.from_config(cfg)
    n=sum(p.numel() for p in m.parameters())
    print(f"\n== {mid}: built in {time.time()-t:.1f}s, params={n:,}")
    layers=None
    for name,mod in m.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod)>4 and name.endswith("layers"): layers=mod; break
    sig=collections.Counter()
    for i,l in enumerate(layers):
        kids=tuple(f"{k}:{type(v).__name__}" for k,v in l.named_children())
        sig[kids]+=1
    for k,c in sig.items(): print(f"  {c}x layer children -> {k}")
    # linear shapes of first layer
    for k,v in layers[0].named_modules():
        if isinstance(v, torch.nn.Linear): print(f"    L0 {k}: {v.in_features}->{v.out_features}")
