"""Research probe: what does each model's K/V cache actually store, who writes it, who reads it?

Runs the model once with use_cache=True under the recorder (zero-storage weights, real tokenized prompt) and
reports, from the recording only:
  - the cache classes the library used (container + per-layer classes)
  - every tensor stored in the returned cache: its slot (attribute path), shape, the recorded op that produced
    it, the module that ran that op, and the stage it was taken at (after RoPE / after a norm / straight from a
    projection / other)
  - for every attention op (softmax) in the run, which stored tensors it read (directly or through ops)
Nothing is changed in the model or the builder.  Usage: python cache_probe.py <repo> <out.json>
"""
import os, re, sys, json, collections
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import noweights; noweights.install()
from lowcost import zero_storage
from dag import DagRecorder, math_attention, clear_library_caches
from inputs import build_passes
from exec_to_ir import Rec, MATMULS, SOFTMAX


def main(repo, out_path):
    import transformers
    R0 = {"repo": repo}
    cfg = transformers.AutoConfig.from_pretrained(repo)
    arch = (cfg.architectures or [None])[0]
    cls = getattr(transformers, arch, None)
    R0["class"] = arch
    with zero_storage():
        try:
            model = cls._from_config(cfg, attn_implementation="eager", dtype=torch.float32)
        except (TypeError, ValueError):
            model = cls._from_config(cfg, dtype=torch.float32)
    model.eval()
    passes, _, _ = build_passes(repo, model, cfg)
    label, kw = passes[0]
    R0["pass"] = label
    clear_library_caches()
    rec = DagRecorder(model, kw)
    try:
        with torch.no_grad(), math_attention(), rec:
            out = model(**kw, use_cache=True)
    except TypeError:
        with torch.no_grad(), math_attention(), rec:
            out = model(**kw)
    rec.remove_hooks()
    pkv = getattr(out, "past_key_values", None)
    R0["cache_container"] = type(pkv).__name__ if pkv is not None else None
    if pkv is None:
        R0["note"] = "no cache returned"
        json.dump(R0, open(out_path, "w"), indent=1); return
    # stored tensors with their slot paths
    stored, seen = [], set()
    layer_classes = collections.Counter()
    def walk(o, path, depth=0):
        if depth > 7 or id(o) in seen: return
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.dim() >= 2 and o.numel() > 0: stored.append((path, o))
            return
        if isinstance(o, (list, tuple)):
            for i, x in enumerate(o): walk(x, f"{path}[{i}]", depth + 1)
        elif isinstance(o, dict):
            for k, x in o.items(): walk(x, f"{path}.{k}", depth + 1)
        elif hasattr(o, "__dict__") and not isinstance(o, (torch.nn.Module,)):
            if depth > 0: layer_classes[type(o).__name__] += 1
            for k, x in vars(o).items(): walk(x, f"{path}.{k}", depth + 1)
    walk(pkv, "cache")
    R0["layer_classes"] = dict(layer_classes)
    R = Rec(rec, model); N = R.n
    is_w = lambda j: N[j].func in MATMULS and bool(R.weights(j))
    # attention reads: ancestors of every softmax (and of the matmul right after it, for values)
    sms = [j for j, nd in enumerate(N) if nd.func in SOFTMAX]
    anc = {}
    for s in sms:
        a = set(R.back(s, lambda j: False, 4000)[1])
        after = next((c for c in range(s + 1, min(s + 40, len(N))) if N[c].func in ("bmm", "matmul") and not R.weights(c)
                      and s in set(R.back(c, lambda j: False, 60)[1])), None)
        av = set(R.back(after, lambda j: False, 4000)[1]) if after is not None else set()
        anc[s] = (a, av, N[s].module)
    rows = []
    for path, t in stored:
        p = rec.producer.get(id(t))
        row = {"slot": path, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
        if p is None:
            row["producer"] = None                       # not produced by a recorded op (preallocated, untouched)
        else:
            hit, chain = R.back(p, is_w)
            funcs = {N[j].func for j in chain}
            stage = ("rope" if {"neg", "cat"} <= funcs or "view_as_complex" in funcs else
                     "norm" if "rsqrt" in funcs or "native_layer_norm" in funcs else
                     "projection" if hit is not None else "other")
            readers = sorted({anc[s][2] for s in sms if p in anc[s][0] or p in anc[s][1]})
            row.update({"producer": p, "producer_op": N[p].func, "producer_module": N[p].module,
                        "weight": (R.weights(hit)[0] if hit is not None else None), "stage": stage,
                        "read_by": readers})
        rows.append(row)
    R0["stored"] = rows
    R0["attention_ops"] = len(sms)
    json.dump(R0, open(out_path, "w"), indent=1, default=str)


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        json.dump({"repo": sys.argv[1], "error": f"{type(e).__name__}: {str(e)[:300]}",
                   "where": f"{tb[-1].filename.split('site-packages/')[-1]}:{tb[-1].lineno}" if tb else ""},
                  open(sys.argv[2], "w"), indent=1)
