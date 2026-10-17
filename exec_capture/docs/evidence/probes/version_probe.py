import json, sys, time, collections, torch, warnings
warnings.filterwarnings("ignore")
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
MODELS=["meta-llama/Meta-Llama-3-8B","Qwen/Qwen3-8B","allenai/OLMo-2-1124-7B","mistralai/Mixtral-8x7B-v0.1","microsoft/Phi-3-mini-4k-instruct","google/gemma-3-1b-pt","HuggingFaceTB/SmolLM3-3B","ibm-granite/granite-3.3-2b-base"]
out={"transformers":transformers.__version__,"tool":{},"exec":{}}
from model_unfolder import unfold
for mid in MODELS:
    try:
        d=unfold(mid); ir=d.ir
        kinds=collections.Counter(str(getattr(l.attention,'kind',None)) for l in ir.layers)
        p=None
        try:
            from model_unfolder import estimate_params; e=estimate_params(ir); p=getattr(e,'total',None) or (e.get('total') if isinstance(e,dict) else None)
        except Exception as ex: p=f"params-err {type(ex).__name__}"
        out["tool"][mid]={"ok":True,"layers":len(ir.layers),"attn":dict(kinds),"params":p,"warnings":len(getattr(ir,'warnings',[]) or [])}
    except Exception as e:
        out["tool"][mid]={"ok":False,"err":f"{type(e).__name__}: {str(e)[:120]}"}
    try:
        cfg=AutoConfig.from_pretrained(mid)
        with torch.device("meta"): m=AutoModelForCausalLM.from_config(cfg)
        n=sum(x.numel() for x in m.parameters())
        c2=AutoConfig.from_pretrained(mid, num_hidden_layers=1)
        with torch.device("meta"): m1=AutoModelForCausalLM.from_config(c2, attn_implementation="eager")
        ep=torch.export.export(m1,(),{"input_ids":torch.zeros((1,8),dtype=torch.long,device="meta"),"use_cache":False},strict=False)
        ops=[n_.target.__name__.split('.')[0] for n_ in ep.graph.nodes if n_.op=="call_function"]
        consts=[a for n_ in ep.graph.nodes if n_.op=="call_function" for a in n_.args if isinstance(a,float)]
        out["exec"][mid]={"ok":True,"params":n,"n_ops":len(ops),"ops_sig":collections.Counter(ops).most_common(6),"consts":consts[:6]}
    except Exception as e:
        out["exec"][mid]={"ok":False,"err":f"{type(e).__name__}: {str(e)[:120]}"}
print(json.dumps(out))
