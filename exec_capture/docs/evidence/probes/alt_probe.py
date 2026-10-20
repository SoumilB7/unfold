import time, torch, collections
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import HfApi

# A) full dataflow graph via torch.export (edges + owning module for every op)
cfg=AutoConfig.from_pretrained("google/gemma-2-9b", num_hidden_layers=1)
with torch.device("meta"): m=AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
m.eval(); t=time.time()
try:
    ep=torch.export.export(m, (), {"input_ids":torch.zeros((1,8),dtype=torch.long,device="meta"),"use_cache":False}, strict=False)
    g=ep.graph; nodes=[n for n in g.nodes if n.op=="call_function"]
    print(f"A) torch.export OK in {time.time()-t:.1f}s: {len(nodes)} ops with explicit input edges")
    for n in nodes:
        stack=n.meta.get("nn_module_stack",{})
        owner=list(stack.values())[-1][0] if stack else "?"
        if "self_attn" in owner and n.target.__name__.split('.')[0] in ("mul","div","tanh","_softmax","softmax","matmul","bmm"):
            ins=[a.name if hasattr(a,'name') else a for a in n.args]
            print(f"   {owner:32s} {str(n.target.__name__):18s} inputs={ins}")
except Exception as e:
    print("A) torch.export FAILED:", type(e).__name__, str(e)[:300])

# B) checkpoint headers only (no weights downloaded)
t=time.time()
try:
    meta=HfApi().get_safetensors_metadata("LiquidAI/LFM2-1.2B")
    shapes={}
    for f in meta.files_metadata.values():
        for k,v in f.tensors.items(): shapes[k]=v.shape
    total=sum(__import__('math').prod(s) for s in shapes.values())
    kinds=collections.Counter()
    for k in shapes:
        parts=k.split('.')
        if 'layers' in parts:
            i=parts.index('layers'); kinds[(int(parts[i+1]), parts[i+2])]+=0
    per=collections.defaultdict(set)
    for (li,kind) in kinds: per[li].add(kind)
    sched=["attn" if "self_attn" in per[i] else ("conv" if "conv" in per[i] else "?") for i in sorted(per)]
    print(f"B) safetensors headers in {time.time()-t:.1f}s: {len(shapes)} tensors, {total:,} params; schedule={sched}")
except Exception as e:
    print("B) headers FAILED:", type(e).__name__, str(e)[:200])

# C) config perturbation: which ops does a field control? (no source reading)
from torch.utils._python_dispatch import TorchDispatchMode
class Rec(TorchDispatchMode):
    def __init__(s): super().__init__(); s.c=[]
    def __torch_dispatch__(s,f,types,args=(),kw=None):
        c=[a for a in args if isinstance(a,float)]
        if c: s.c.append((f.overloadpacket.__name__,c[0]))
        return f(*args,**(kw or {}))
def consts(**ov):
    c=AutoConfig.from_pretrained("google/gemma-2-9b", num_hidden_layers=1, **ov)
    with torch.device("meta"): mm=AutoModelForCausalLM.from_config(c, attn_implementation="eager")
    r=Rec()
    with r: mm(input_ids=torch.zeros((1,8),dtype=torch.long,device="meta"))
    return r.c
base=consts(); pert=consts(query_pre_attn_scalar=64, attn_logit_softcapping=20.0)
print("C) perturbation diff:", [(a,b) for a,b in zip(base,pert) if a!=b])
