import sys, time, resource, json, concurrent.futures as cf
T0 = time.time()
def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
repo, mode = sys.argv[1], sys.argv[2]; phases = []
def mark(n, t): phases.append((n, round(time.time() - t, 2), round(peak(), 2)))
t = time.time(); import torch, transformers; mark("imports", t)
sys.path.insert(0, "."); from lowcost import zero_storage, Recorder, fetch_headers
t = time.time()
with cf.ThreadPoolExecutor(2) as ex:
    fc = ex.submit(transformers.AutoConfig.from_pretrained, repo); fh = ex.submit(fetch_headers, repo)
    cfg = fc.result(); T, nf = fh.result()
mark(f"config + headers in parallel ({nf} files, {len(T)} tensors)", t)
cls = getattr(transformers, cfg.architectures[0])
t = time.time()
with zero_storage():
    m = cls._from_config(cfg, dtype=torch.bfloat16)
m.eval(); mark("zero-storage build", t)
if mode == "image":
    from PIL import Image
    proc = transformers.AutoProcessor.from_pretrained(repo)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    kw = dict(proc(images=Image.new("RGB", (224, 224)), text=proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), return_tensors="pt"))
    kw = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in kw.items()}
else:
    kw = {"input_ids": torch.zeros((1, 16), dtype=torch.long)}
ids = {id(p): n for n, p in list(m.named_parameters(remove_duplicate=False)) + list(m.named_buffers(remove_duplicate=False))}
rec = Recorder(ids, skip=True)
t = time.time()
with torch.no_grad(), rec:
    m(**kw, use_cache=False)
mark("real forward (zero-skip)", t)
P = dict(m.named_parameters())
print("RESULT " + json.dumps({"repo": repo, "total_s": round(time.time() - T0, 1), "peak_gb": round(peak(), 2), "ops": rec.ops, "skipped": rec.skipped,
                  "params_used_frac": round(sum(P[n].numel() for n in rec.used if n in P) / sum(p.numel() for p in P.values()), 6), "phases": phases}))
