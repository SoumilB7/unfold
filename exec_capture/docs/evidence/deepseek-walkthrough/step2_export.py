import time, collections, torch, warnings; warnings.filterwarnings("ignore")
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")   # the plain reference path, no fused kernel
model.eval()
t = time.time()
ep = torch.export.export(model, (), {"input_ids": torch.zeros((1, 8), dtype=torch.long, device="meta"), "use_cache": False}, strict=False)
nodes = [n for n in ep.graph.nodes if n.op == "call_function"]
print(f"exported full 61-layer model in {time.time()-t:.1f}s -> {len(nodes)} recorded operations")
torch.export.save(ep, "deepseek_v3.pt2"); import os; print(f"saved as a file: deepseek_v3.pt2 ({os.path.getsize('deepseek_v3.pt2')/1e6:.1f} MB)")

def owner(n):
    s = n.meta.get("nn_module_stack", {})
    return list(s.values())[-1][0] if s else "(top)"
def shape(n):
    v = n.meta.get("val"); return tuple(v.shape) if hasattr(v, "shape") else "-"
def op(n): return str(n.target).replace("aten.", "").replace(".default", "").replace(".Tensor", "")
def arg(a):
    if isinstance(a, torch.fx.Node): return a.name
    if isinstance(a, (list, tuple)): return "[" + ",".join(arg(x) for x in a) + "]"
    return repr(a)
SKIP = {"_assert_tensor_metadata", "alias", "detach_", "contiguous"}
def show(title, pred, limit=200):
    print(f"\n--- {title} ---")
    k = 0
    for n in nodes:
        if not pred(owner(n)) or op(n).split(".")[0] in SKIP: continue
        o = owner(n).replace("model.layers.", "L")
        print(f"  {n.name:22s} = {op(n):22s}({', '.join(arg(a) for a in n.args)[:70]:70s}) -> {str(shape(n)):22s} [{o}]")
        k += 1
        if k >= limit: break

show("layer 0 attention: every operation, in order", lambda o: o.startswith("model.layers.0.self_attn"))
show("layer 3 router (gate): every operation", lambda o: o.startswith("model.layers.3.mlp.gate"))
ex = [n for n in nodes if owner(n).startswith("model.layers.3.mlp.experts")]
print(f"\n--- layer 3 routed experts: {len(ex)} ops; kinds = {dict(collections.Counter(op(n).split('.')[0] for n in ex).most_common(8))}")
show("layer 3 MoE combine (outside gate/experts/shared)", lambda o: o == "model.layers.3.mlp")
print("\nconstants multiplied into attention scores, per layer:",
      sorted({(a) for n in nodes if owner(n).endswith("self_attn") and op(n)=="mul" for a in n.args if isinstance(a, float)}))
