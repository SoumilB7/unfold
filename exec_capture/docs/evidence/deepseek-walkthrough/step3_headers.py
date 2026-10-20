import time, re, math, collections, json, torch, warnings; warnings.filterwarnings("ignore")
from huggingface_hub import HfApi
t = time.time()
meta = HfApi().get_safetensors_metadata("deepseek-ai/DeepSeek-V3")
T = {}
for fname, fm in meta.files_metadata.items():
    for name, info in fm.tensors.items(): T[name] = (tuple(info.shape), info.dtype)
print(f"read headers of {len(meta.files_metadata)} weight files in {time.time()-t:.1f}s -> {len(T)} tensors (no weights downloaded)")
print("dtypes:", dict(collections.Counter(d for _, d in T.values())))
scale = {k for k in T if k.endswith("_scale_inv")}
print(f"quantization scale tensors (*_scale_inv): {len(scale)}  e.g. {sorted(scale)[0]} {T[sorted(scale)[0]]}")
layers = sorted({int(m.group(1)) for k in T for m in [re.match(r"model\.layers\.(\d+)\.", k)] if m})
print(f"layer indices in checkpoint: {layers[0]}..{layers[-1]}  ({len(layers)} layers)")

print("\n[layer 61: every tensor except per-expert copies and scales]")
for k in sorted(T):
    if k.startswith("model.layers.61.") and ".experts." not in k and k not in scale: print(f"  {k:62s} {str(T[k][0]):16s} {T[k][1]}")
n_exp61 = len({k.split('.experts.')[1].split('.')[0] for k in T if k.startswith('model.layers.61.mlp.experts.')})
print(f"  (+ {n_exp61} routed experts in layer 61)")

# Compare with the husk (Step 1), tensor by tensor
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")
with torch.device("meta"): model = AutoModelForCausalLM.from_config(cfg)
H = {n: tuple(p.shape) for n, p in list(model.named_parameters()) + list(model.named_buffers()) if "rotary" not in n}
fused = collections.defaultdict(int)       # checkpoint stores 256 experts separately; the husk stores them fused
exact = mismatch = 0; ckpt_only = []
for k, (shp, dt) in T.items():
    if k in scale: continue
    m = re.match(r"(model\.layers\.\d+\.mlp\.experts)\.\d+\.(gate|up|down)_proj\.weight", k)
    if m:
        fused[m.group(1) + (".down_proj" if m.group(2) == "down" else ".gate_up_proj")] += math.prod(shp); continue
    if k in H: exact += (H[k] == shp); mismatch += (H[k] != shp)
    else: ckpt_only.append(k)
fused_ok = sum(1 for k, n in fused.items() if k in H and math.prod(H[k]) == n)
fused_ck_only = [k for k in fused if k not in H]
husk_only = [k for k in H if k not in T and k not in fused]
print(f"\n[husk vs checkpoint]")
print(f"  tensors identical in name+shape        : {exact}")
print(f"  shape mismatches                       : {mismatch}")
print(f"  fused expert tensors matching exactly  : {fused_ok} of {sum(1 for k in fused if k in H)}")
print(f"  in checkpoint but NOT in the husk      : {len(ckpt_only) + len(fused_ck_only)}  -> all under: {sorted({'.'.join(k.split('.')[:3]) for k in ckpt_only + fused_ck_only})}")
print(f"  in the husk but NOT in checkpoint      : {len(husk_only)} {husk_only[:5]}")
ck_params = sum(math.prod(s) for k, (s, d) in T.items() if k not in scale)
l61 = sum(math.prod(s) for k, (s, d) in T.items() if k not in scale and k.startswith("model.layers.61."))
print(f"\n  checkpoint parameters (excluding scales): {ck_params:,}")
print(f"  of which layer 61                       : {l61:,}")
print(f"  checkpoint minus layer 61               : {ck_params - l61:,}")
print(f"  husk (Step 1)                           : {sum(p.numel() for p in model.parameters()):,}")
