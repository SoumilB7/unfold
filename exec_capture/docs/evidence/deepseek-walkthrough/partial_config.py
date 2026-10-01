import json, re, math, collections, torch, warnings; warnings.filterwarnings("ignore")
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoConfig, AutoModelForCausalLM, DeepseekV3Config
T = {}
for fm in HfApi().get_safetensors_metadata("deepseek-ai/DeepSeek-V3").files_metadata.values():
    for n, i in fm.tensors.items():
        if not n.endswith("_scale_inv"): T[n] = tuple(i.shape)
def check(label, cfgdict):
    cfg = DeepseekV3Config(**cfgdict)
    with torch.device("meta"): m = AutoModelForCausalLM.from_config(cfg)
    H = {n: tuple(p.shape) for n, p in list(m.named_parameters()) + list(m.named_buffers()) if "rotary" not in n}
    fused = collections.defaultdict(int); bad = []; missing = []
    for k, s in T.items():
        if k.startswith("model.layers.61."): continue
        mm = re.match(r"(model\.layers\.\d+\.mlp\.experts)\.\d+\.(gate|up|down)_proj\.weight", k)
        if mm: fused[mm.group(1) + (".down_proj" if mm.group(2) == "down" else ".gate_up_proj")] += math.prod(s); continue
        if k not in H: missing.append(k)
        elif H[k] != s: bad.append((k, H[k], s))
    bad += [(k, H.get(k), n) for k, n in fused.items() if k not in H or math.prod(H[k]) != n]
    extra = [k for k in H if k not in T and k not in fused]
    verdict = "MATCHES the real weights" if not (bad or missing or extra) else f"CAUGHT: {len(bad)} wrong shapes, {len(missing)} tensors the build lacks, {len(extra)} the weights lack"
    print(f"{label:55s} -> {verdict}")
    for k, h, c in bad[:3]: print(f"      e.g. {k}: built {h}, real weights {c}")
    for k in missing[:2]: print(f"      e.g. missing in build: {k}")
    for k in extra[:2]: print(f"      e.g. not in weights: {k}")
full = json.load(open(hf_hub_download("deepseek-ai/DeepSeek-V3", "config.json")))
check("real hub config (43 fields)", full)
for drop in (["moe_intermediate_size"], ["q_lora_rank"], ["first_k_dense_replace", "n_routed_experts"], ["kv_lora_rank", "qk_rope_head_dim", "v_head_dim"]):
    part = {k: v for k, v in full.items() if k not in drop}
    check(f"partial config, dropped {drop}", part)
check("only 5 fields: hidden_size, layers, heads, vocab, arch", {k: full[k] for k in ["hidden_size","num_hidden_layers","num_attention_heads","vocab_size","architectures"]})
