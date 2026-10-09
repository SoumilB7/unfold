import time, resource, collections, torch, transformers, warnings; warnings.filterwarnings("ignore")
from transformers import AutoConfig, AutoModelForCausalLM
print("transformers", transformers.__version__, "| torch", torch.__version__)
cfg = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")
t = time.time()
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg)
print(f"built {type(model).__name__} in {time.time()-t:.2f}s; parameter devices = {set(p.device.type for p in model.parameters())}; "
      f"process RAM peak = {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
total = sum(p.numel() for p in model.parameters())
print(f"total parameters = {total:,}")

print("\n[top level]")
for name, mod in model.named_children(): print(f"  {name}: {type(mod).__name__}")
for name, mod in model.model.named_children():
    extra = f" x{len(mod)}" if isinstance(mod, torch.nn.ModuleList) else ""
    print(f"  model.{name}: {type(mod).__name__}{extra}")
print(f"  model.embed_tokens.weight {tuple(model.model.embed_tokens.weight.shape)} | lm_head.weight {tuple(model.lm_head.weight.shape)} | model.norm.weight {tuple(model.model.norm.weight.shape)}")

print("\n[layer schedule: children of each of the", len(model.model.layers), "layers]")
sched = collections.Counter()
for i, layer in enumerate(model.model.layers):
    sched[tuple(f"{k}:{type(v).__name__}" for k, v in layer.named_children())] += 1
for sig, n in sched.items(): print(f"  {n} layers -> {sig}")
first = [i for i,l in enumerate(model.model.layers) if type(l.mlp).__name__.endswith("MoE")][0]
print(f"  first MoE layer index = {first}")

def dump(prefix, mod):
    for n, p in mod.named_parameters(recurse=False): print(f"    {prefix} param {n} {tuple(p.shape)}")
    for n, b in mod.named_buffers(recurse=False): print(f"    {prefix} buffer {n} {tuple(b.shape)}")

for li in (0, first):
    L = model.model.layers[li]
    print(f"\n[layer {li}: every submodule with its own weights]")
    for n, m in L.named_modules():
        own = list(m.named_parameters(recurse=False)) + list(m.named_buffers(recurse=False))
        if not own: continue
        desc = f"{type(m).__name__}"
        if isinstance(m, torch.nn.Linear): desc += f" {m.in_features}->{m.out_features} bias={m.bias is not None}"
        shapes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in own)
        print(f"  {n or '(layer)':38s} {desc:32s} {shapes}")
    print(f"  layer {li} parameters = {sum(p.numel() for p in L.parameters()):,}")
