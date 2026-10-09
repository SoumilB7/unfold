import torch, time, collections, warnings; warnings.filterwarnings("ignore")
from transformers import AutoConfig, AutoModelForCausalLM
M=[("meta-llama/Meta-Llama-3-8B",1),("Qwen/Qwen3-8B",1),("allenai/OLMo-2-1124-7B",1),("mistralai/Mixtral-8x7B-v0.1",1),
   ("microsoft/Phi-3-mini-4k-instruct",1),("google/gemma-3-1b-pt",1),("google/gemma-2-9b",1),("HuggingFaceTB/SmolLM3-3B",1),
   ("ibm-granite/granite-3.3-2b-base",1),("deepseek-ai/DeepSeek-V3",4),("LiquidAI/LFM2-1.2B",3),("state-spaces/mamba-130m-hf",1),
   ("Qwen/Qwen3-30B-A3B",1),("tiiuae/falcon-mamba-7b",1)]
vocab=collections.Counter(); per={}
for mid,L in M:
    t=time.time()
    try:
        ov={"num_hidden_layers":L}
        if "DeepSeek" in mid: ov["first_k_dense_replace"]=3
        cfg=AutoConfig.from_pretrained(mid,**ov)
        with torch.device("meta"): m=AutoModelForCausalLM.from_config(cfg, attn_implementation="eager") if not "mamba" in mid else AutoModelForCausalLM.from_config(cfg)
        ep=torch.export.export(m,(),{"input_ids":torch.zeros((1,8),dtype=torch.long,device="meta"),"use_cache":False},strict=False)
        ops={str(n.target).replace("aten.","").split(".")[0] for n in ep.graph.nodes if n.op=="call_function"}
        new=ops-set(vocab); vocab.update(ops); per[mid]=len(ops)
        print(f"OK   {mid:38s} {len(ops):3d} distinct ops  {time.time()-t:4.1f}s  new to vocab: {sorted(new)[:12]}{'...' if len(new)>12 else ''}")
    except Exception as e:
        print(f"FAIL {mid:38s} {type(e).__name__}: {str(e)[:140]}")
print(f"\nUNION of distinct ops across {len(per)} models: {len(vocab)}")
print(sorted(vocab))
