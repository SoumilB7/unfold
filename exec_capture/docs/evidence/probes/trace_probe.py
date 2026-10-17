import sys, time, torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoConfig, AutoModelForCausalLM
stack=[]; log=[]
class Rec(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out=func(*args, **(kwargs or {}))
        if stack and "self_attn" in stack[-1] and ".layers.0." in stack[-1]+".":
            name=str(func.overloadpacket.__name__)
            extra=""
            if name in ("mul","div") and any(isinstance(a,(int,float)) for a in args): extra=" const="+str([a for a in args if isinstance(a,(int,float))])
            if name in ("_scaled_dot_product_flash_attention_for_cpu","scaled_dot_product_attention","_scaled_dot_product_efficient_attention") : extra=" kw="+str({k:v for k,v in (kwargs or {}).items() if k in('scale','is_causal')})
            log.append(f"{stack[-1]}: {name}{extra}")
        return out
def pre(n):  return lambda m,a: (stack.append(n), None)[1]
def post(n): return lambda m,a,o: (stack.pop(), None)[1]
for mid, over in [("google/gemma-2-9b",{"num_hidden_layers":2}),("LiquidAI/LFM2-1.2B",{}),("deepseek-ai/DeepSeek-V3",{"num_hidden_layers":4})]:
    cfg=AutoConfig.from_pretrained(mid, **over); t=time.time(); log.clear()
    try: cfg._attn_implementation="eager"
    except Exception: pass
    with torch.device("meta"): m=AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    for n,mod in m.named_modules(): mod.register_forward_pre_hook(pre(n)); mod.register_forward_hook(post(n))
    try:
        with Rec(): m(input_ids=torch.zeros((1,8),dtype=torch.long,device="meta"))
        print(f"\n== {mid}: meta forward OK in {time.time()-t:.1f}s; layer0 attention ops:")
        seen=[]; 
        for l in log:
            if l not in seen[-1:]: seen.append(l)
        print("   "+"\n   ".join(seen[:40]))
    except Exception as e:
        import traceback; traceback.print_exc(limit=-4); break
