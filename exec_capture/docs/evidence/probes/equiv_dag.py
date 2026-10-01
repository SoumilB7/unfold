import sys, torch, transformers
sys.path.insert(0, "/Users/soumil/Code/Projects/Understand/llmvisualizer/model-benchmark/harness_v2")
from lowcost import zero_storage
from dag import DagRecorder, math_attention, clear_library_caches
repo = sys.argv[1]
cfg = transformers.AutoConfig.from_pretrained(repo); cls = getattr(transformers, cfg.architectures[0])
with zero_storage():
    m = cls._from_config(cfg, attn_implementation="eager", dtype=torch.bfloat16 if "A3B" in repo else torch.float32)
m.eval(); res = {}
ids = torch.tensor([[5, 17, 99, 3, 42, 7, 256, 11, 13, 1, 2, 3]])
for skip in (False, True):
    clear_library_caches(); r = DagRecorder(m, {"input_ids": ids}, skip=skip)
    with torch.no_grad(), math_attention(), r:
        out = m(input_ids=ids, use_cache=False).logits
    r.remove_hooks()
    res[skip] = ([(n.module, n.func) for n in r.nodes], r.used_params, out.clone(), r.skipped)
a, b = res[False], res[True]
print(f"{repo}: identical op sequence {a[0]==b[0]} ({len(a[0])} ops) | identical weights used {a[1]==b[1]} | identical logits {torch.equal(a[2], b[2])} | skipped {b[3]}")
