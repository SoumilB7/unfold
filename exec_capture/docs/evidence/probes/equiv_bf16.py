import sys, torch, transformers, hashlib
sys.path.insert(0, "."); import lowcost
from lowcost import zero_storage, Recorder
repo = sys.argv[1]
cfg = transformers.AutoConfig.from_pretrained(repo); cls = getattr(transformers, cfg.architectures[0])
with zero_storage():
    m = cls._from_config(cfg, dtype=torch.bfloat16)
m.eval()
ids = {id(p): n for n, p in list(m.named_parameters(remove_duplicate=False)) + list(m.named_buffers(remove_duplicate=False))}
traces = {}
for skip in (False, True):
    class TraceRec(Recorder):
        def __init__(s, ids, skip): super().__init__(ids, skip); s.trace = []; s.outs = []
        def __torch_dispatch__(s, func, types, args=(), kwargs=None):
            out = super().__torch_dispatch__(func, types, args, kwargs)
            s.trace.append(str(func))
            o = out[0] if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor) else out
            if isinstance(o, torch.Tensor): s.outs.append((tuple(o.shape), str(o.dtype), bool(o.float().abs().sum().item() == 0) if o.numel() and o.is_floating_point() else None))
            return out
    r = TraceRec(ids, skip)
    with torch.no_grad(), r:
        logits = m(input_ids=torch.zeros((1, 8), dtype=torch.long), use_cache=False).logits
    traces[skip] = (r.trace, r.outs, r.used, logits.clone(), r.skipped)
a, b = traces[False], traces[True]
print(f"{repo}: ops {len(a[0])} vs {len(b[0])} | identical op sequence: {a[0] == b[0]} | identical output shapes/dtypes/zero-ness: {a[1] == b[1]} | "
      f"identical weights used: {a[2] == b[2]} | identical final logits: {torch.equal(a[3], b[3])} | arithmetic skipped: {b[4]}")
