import json, re, math, collections, torch, warnings; warnings.filterwarnings("ignore")
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoConfig, AutoModelForCausalLM, CONFIG_MAPPING
def headers(repo):
    T = {}
    for fm in HfApi().get_safetensors_metadata(repo).files_metadata.values():
        for n, i in fm.tensors.items():
            if not n.endswith("_scale_inv"): T[n] = tuple(i.shape)
    return T
def check(repo, T, label, cfgdict):
    cfg = CONFIG_MAPPING[cfgdict["model_type"]](**cfgdict)
    with torch.device("meta"): m = AutoModelForCausalLM.from_config(cfg)
    H = {n: tuple(p.shape) for n, p in list(m.named_parameters()) + list(m.named_buffers()) if "rotary" not in n and "inv_freq" not in n}
    fused = collections.defaultdict(int); bad = []; missing = []
    for k, s in T.items():
        if re.match(r"model\.layers\.61\.", k) and "DeepSeek" in repo: continue
        mm = re.match(r"(model\.layers\.\d+\.mlp\.experts)\.\d+\.(gate|up|down)_proj\.weight", k)
        if mm and mm.group(1) + ".down_proj" in H: fused[mm.group(1) + (".down_proj" if mm.group(2) == "down" else ".gate_up_proj")] += math.prod(s); continue
        if k not in H: missing.append(k)
        elif H[k] != s: bad.append((k, H[k], s))
    bad += [(k, H.get(k), f"{n:,} values") for k, n in fused.items() if math.prod(H[k]) != n]
    extra = [k for k in H if k not in T and k not in fused and not (k == "lm_head.weight" and getattr(cfg, "tie_word_embeddings", False))]
    ok = not (bad or missing or extra)
    print(f"  {label:58s} -> {'MATCHES real weights' if ok else f'CAUGHT: {len(bad)} wrong shapes, {len(missing)} missing, {len(extra)} extra'}")
    for k, h, c in bad[:2]: print(f"        {k}: built {h} vs real {c}")
    for k in (missing[:1] + extra[:1]): print(f"        {k}")
print("DeepSeek-V3, deliberately WRONG values:")
T = headers("deepseek-ai/DeepSeek-V3"); full = json.load(open(hf_hub_download("deepseek-ai/DeepSeek-V3", "config.json")))
check("deepseek", T, "moe_intermediate_size 2048 -> 1408", {**full, "moe_intermediate_size": 1408})
check("deepseek", T, "n_routed_experts 256 -> 160", {**full, "n_routed_experts": 160})
check("deepseek", T, "first_k_dense_replace 3 -> 1", {**full, "first_k_dense_replace": 1})
check("deepseek", T, "q_lora_rank 1536 -> None (no low-rank query)", {**full, "q_lora_rank": None})
print("Qwen3-8B, fields DROPPED (class defaults differ from this checkpoint):")
T = headers("Qwen/Qwen3-8B"); full = json.load(open(hf_hub_download("Qwen/Qwen3-8B", "config.json")))
check("qwen", T, "real hub config", full)
for drop in (["intermediate_size"], ["num_key_value_heads"], ["head_dim"], ["tie_word_embeddings"]):
    check("qwen", T, f"dropped {drop}", {k: v for k, v in full.items() if k not in drop})
