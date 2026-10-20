import torch, collections, warnings; warnings.filterwarnings("ignore")
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")
with torch.device("meta"): model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
model.eval()
ep = torch.export.export(model, (), {"input_ids": torch.zeros((1, 8), dtype=torch.long, device="meta"), "use_cache": False}, strict=False)
nodes = [n for n in ep.graph.nodes if n.op == "call_function"]
def owner(n):
    s = n.meta.get("nn_module_stack", {}); return list(s.values())[-1][0] if s else "(top)"
def shape(n):
    v = n.meta.get("val"); return tuple(v.shape) if hasattr(v, "shape") else "-"
def op(n): return str(n.target).replace("aten.", "").replace(".default", "").replace(".Tensor", "")
def arg(a):
    if isinstance(a, torch.fx.Node): return a.name
    if isinstance(a, (list, tuple)): return "[" + ",".join(arg(x) for x in a) + "]"
    return repr(a)
SKIP = {"_assert_tensor_metadata", "alias", "detach_", "contiguous"}
def show(title, pred):
    print(f"\n--- {title} ---")
    for n in nodes:
        if pred(owner(n)) and op(n).split(".")[0] not in SKIP:
            print(f"  {n.name:18s} = {op(n):18s}({', '.join(arg(a) for a in n.args)[:78]:78s}) -> {str(shape(n)):16s} [{owner(n).replace('model.layers.','L')}]")
show("model entry: embedding (+ anything before layer 0)", lambda o: o in ("model.embed_tokens",))
show("layer 0 wiring (ops owned by the layer itself = residual adds)", lambda o: o == "model.layers.0")
show("layer 0 input_layernorm", lambda o: o == "model.layers.0.input_layernorm")
show("layer 0 dense MLP (gate/up/down + activation)", lambda o: o.startswith("model.layers.0.mlp"))
show("layer 3 shared expert", lambda o: o.startswith("model.layers.3.mlp.shared_experts"))
show("layer 3 routed experts (all ops)", lambda o: o.startswith("model.layers.3.mlp.experts"))
show("model exit: final norm + lm_head", lambda o: o in ("model.norm", "lm_head"))
print("\nactivation ops anywhere in the model:", dict(collections.Counter(op(n).split('.')[0] for n in nodes if op(n).split('.')[0] in ("silu","gelu","relu","sigmoid","tanh","softplus","exp"))))
