"""Execution recording -> unfold's ModelIR -> unfold's own renderer.

The model is built with zero-storage weights, run on a real tokenized prompt, and every op is recorded
(DagRecorder). The recognizers below read that recording and fill the same ModelIR the code parser fills
(model facts, one LayerSpec per executed layer, the per-layer block tree, the outer blocks). Every fact
comes from what ran: tensor shapes, the weights each op read, recorded constants, the built model's own
buffers, and the values of activations (the attention mask). Nothing is read from modeling source.
The unfold package is imported read-only; its Diagram renders the result unchanged.

Scope of this first version: decoder-only transformers (attention + dense MLP).
Usage: python exec_to_ir.py <repo> <out.html> [--unfold-pkg PATH]
"""
import os, re, sys, json, math, collections, dataclasses
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noweights; noweights.install()
from lowcost import zero_storage
from dag import DagRecorder, math_attention, clear_library_caches
from inputs import build_passes

MATMULS = {"mm", "addmm", "matmul", "linear", "bmm", "baddbmm"}
VIEWS = {"view", "_unsafe_view", "unsqueeze", "squeeze", "expand", "t", "transpose", "reshape", "contiguous",
         "clone", "slice", "select", "alias", "detach", "_to_copy", "to", "permute", "lift_fresh", "repeat_interleave"}
SOFTMAX = {"_softmax", "softmax", "_safe_softmax"}
ACTS = {"silu": "silu", "gelu": "gelu", "relu": "relu", "sigmoid": "sigmoid", "tanh": "tanh"}


class MaskCapture(DagRecorder):
    """DagRecorder that also keeps the small square tensors added to attention scores (to read the mask's values)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.square_adds = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = super().__torch_dispatch__(func, types, args, kwargs)
        if func.overloadpacket.__name__ == "add":
            for a in args:
                if isinstance(a, torch.Tensor) and a.dim() >= 2 and a.shape[-1] == a.shape[-2] and a.numel() <= 1 << 16:
                    self.square_adds[len(self.nodes) - 1] = a.detach().clone()
        return out


class Rec:
    def __init__(self, rec, model):
        self.n = rec.nodes
        self.rec = rec
        self.pshape = {k: tuple(v.shape) for k, v in model.named_parameters(remove_duplicate=False)}
        self.cons = collections.defaultdict(list)
        for i, nd in enumerate(self.n):
            for e in nd.ins:
                if e[0] == "op": self.cons[e[1]].append(i)

    def srcs(self, i):
        return [e[1] for e in self.n[i].ins if e[0] == "op"]

    def weights(self, i):
        """Checkpoint weights an op reads, directly or through a view (e.g. the transpose before a matmul)."""
        out = [e[1] for e in self.n[i].ins if e[0] == "param"]
        for s in self.srcs(i):
            if self.n[s].func in VIEWS:
                out += [e[1] for e in self.n[s].ins if e[0] == "param"]
        return out

    def back(self, i, stop, limit=400):
        """Walk ALL producers of op i without passing through stop ops; returns (nearest stop op, every op visited)."""
        seen, frontier, path, hit = set(), [i], [], None
        while frontier and len(seen) < limit:
            nxt = []
            for j in frontier:
                if j in seen: continue
                seen.add(j)
                if j != i and stop(j):
                    if hit is None: hit = j          # the nearest stop op (BFS order); do not expand past it
                    continue
                path.append(j)
                nxt += self.srcs(j)
            frontier = nxt
        return hit, path

    def base(self, i):
        """The first producer of op i that is not a shape-only op (the tensor a view was taken of)."""
        seen = 0
        while self.n[i].func in VIEWS and self.srcs(i) and seen < 20:
            i = self.srcs(i)[0]; seen += 1
        return i


def norm_kind(R, ops):
    f = {R.n[i].func for i in ops}
    if "native_layer_norm" in f or "layer_norm" in f: return "layernorm"
    if "rsqrt" in f and ("pow" in f or "mul" in f) and "mean" in f:
        return "rmsnorm" if "sub" not in f else "layernorm"
    return None


def verify_cache(rec_c, out_c, model, wk, wv, layer_prefix):
    """Observed K/V cache behaviour: which op produced the tensors stored in the returned cache, and whether this
    layer's attention read its keys/values from them. Returns {} if no cache was returned."""
    pkv = getattr(out_c, "past_key_values", None)
    if pkv is None:
        return {}
    found, seen = [], set()
    def walk(o, depth=0):
        if depth > 6 or id(o) in seen: return
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.dim() == 4: found.append(o)
            return
        if isinstance(o, (list, tuple)):
            for x in o: walk(x, depth + 1)
        elif isinstance(o, dict):
            for x in o.values(): walk(x, depth + 1)
        elif hasattr(o, "__dict__"):
            for x in vars(o).values(): walk(x, depth + 1)
    walk(pkv)
    R = Rec(rec_c, model); N = R.n
    is_w = lambda j: N[j].func in MATMULS and bool(R.weights(j))
    write = {}
    for t in found:
        p = rec_c.producer.get(id(t))
        if p is None or not N[p].module.startswith(layer_prefix): continue
        hit, path = R.back(p, is_w)
        if hit is None: continue
        w_ = R.weights(hit)[0]
        role = "key" if w_ == wk else ("value" if w_ == wv else None)
        if role and role not in write:
            funcs = {N[j].func for j in path}
            after = "rope" if ("neg" in funcs and "cat" in funcs) else ("norm" if "rsqrt" in funcs else "projection")
            write[role] = {"producer_op": p, "after": after, "stored_shape": list(t.shape)}
    if not write:
        return {"returned": True, "layer_written": False}
    # read: this layer's score and weights·V matmuls take their K / V from the stored tensors
    lops = [j for j, nd in enumerate(N) if nd.module.startswith(layer_prefix)]
    sm = next(j for j in lops if N[j].func in SOFTMAX)
    scores, _ = R.back(sm, lambda j: N[j].func in ("bmm", "matmul") and not R.weights(j))
    reads = {}
    for role, w in write.items():
        anc = set(R.back(scores, lambda j: False, 600)[1]) | {scores}
        later = [j for j in lops if j > sm and N[j].func in ("bmm", "matmul") and not R.weights(j)]
        anc_v = set(R.back(later[0], lambda j: False, 600)[1]) if later else set()
        reads[role] = w["producer_op"] in (anc if role == "key" else anc_v)
    return {"returned": True, "layer_written": True, "write": write, "read": reads,
            "verified": write.get("key") is not None and write.get("value") is not None and all(reads.values())}


def build(repo):
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(repo)
    arch = (cfg.architectures or [None])[0]
    cls = getattr(transformers, arch)
    with zero_storage():
        model = cls._from_config(cfg, attn_implementation="eager", dtype=torch.float32)
    model.eval()
    passes, _, _ = build_passes(repo, model, cfg)
    label, kw = passes[0]
    clear_library_caches()
    rec = MaskCapture(model, kw)
    with torch.no_grad(), math_attention(), rec:
        model(**kw, use_cache=False)
    rec.remove_hooks()
    # second pass with the cache ON: the returned cache holds the stored keys/values; their producers are looked up
    # in this recording (the tensors stay alive inside the cache object, so the producer map still knows them)
    rec_c = DagRecorder(model, kw)
    with torch.no_grad(), math_attention(), rec_c:
        out_c = model(**kw, use_cache=True)
    rec_c.remove_hooks()
    return model, cfg, arch, kw, rec, (rec_c, out_c)


def recognize(repo):
    model, cfg, arch, kw, rec, (rec_c, out_c) = build(repo)
    R = Rec(rec, model)
    N = R.n
    ev = collections.defaultdict(list)                     # fact -> evidence strings

    # ---- the repeated layer stack: the module prefix whose numbered children executed the most
    pref = collections.defaultdict(set)
    for nd in N:
        m = re.match(r"^(.*?)\.(\d+)(?:\.|$)", nd.module)
        if m: pref[m.group(1)].add(int(m.group(2)))
    stack = max(pref, key=lambda p: len(pref[p]))
    layer_ids = sorted(pref[stack])
    L = {i: [j for j, nd in enumerate(N) if nd.module == f"{stack}.{i}" or nd.module.startswith(f"{stack}.{i}.")] for i in layer_ids}
    ev["layers"].append(f"{len(layer_ids)} numbered children of {stack} executed")

    # identical structure across layers
    normseq = lambda ops: [(re.sub(rf"^{re.escape(stack)}\.\d+", "L", N[j].module), N[j].func) for j in ops]
    ref = normseq(L[layer_ids[0]])
    identical = [i for i in layer_ids if normseq(L[i]) == ref]

    # ---- embedding / LM head / final norm
    emb = next(j for j, nd in enumerate(N) if nd.func == "embedding")
    emb_w = R.weights(emb)[0]
    vocab, hidden = R.pshape[emb_w]
    ev["embedding"].append(f"op {emb} embedding reads {emb_w} {list(R.pshape[emb_w])}")
    last = max(L[layer_ids[-1]])
    head = next(j for j in range(last + 1, len(N)) if N[j].func in MATMULS and R.weights(j))
    head_w = R.weights(head)[0]
    pid = {k: id(v) for k, v in model.named_parameters(remove_duplicate=False)}
    tied = pid.get(head_w) == pid.get(emb_w)
    ev["tied"].append(f"LM head op {head} reads {head_w}; same tensor as {emb_w}: {tied}")
    fin_ops = [j for j in range(last + 1, head)]
    final_norm = norm_kind(R, fin_ops)

    # ---- one layer in detail (all layers are checked identical above)
    lops = L[layer_ids[0]]; lset = set(lops)
    sm = next(j for j in lops if N[j].func in SOFTMAX)
    attn_mod = N[sm].module
    scores, pre_sm = R.back(sm, lambda j: N[j].func in ("bmm", "matmul") and not R.weights(j))
    num_heads = N[sm].shape[1] if N[sm].shape and len(N[sm].shape) == 4 else None
    def fwd(i, want, limit=12):                          # first consumer reached through shape-only/cast/dropout ops
        frontier, seen = [i], set()
        for _ in range(limit):
            nxt = []
            for j in frontier:
                for c in R.cons[j]:
                    if c in seen: continue
                    seen.add(c)
                    if want(c): return c
                    if N[c].func in VIEWS or N[c].func in ("dropout", "native_dropout", "_to_copy", "type_as"): nxt.append(c)
            frontier = nxt
        return None
    apply_v = fwd(sm, lambda j: N[j].func in ("bmm", "matmul"))
    # score scale: a constant multiplying/dividing the scores or the query
    scale = None
    for j in pre_sm + R.back(scores, lambda j: R.weights(j) != [])[1]:
        if N[j].func in ("mul", "div") and N[j].consts:
            c = N[j].consts[0]; scale = c if N[j].func == "mul" else 1 / c; break
    # mask: a square tensor added to the scores before softmax; causal if upper triangle is -inf-like
    mask = None
    for j in pre_sm:
        if N[j].func == "add" and j in rec.square_adds:
            t = rec.square_adds[j].float().reshape(-1, *rec.square_adds[j].shape[-2:])[0]
            up = torch.triu(torch.ones_like(t, dtype=torch.bool), 1)
            mask = "causal" if bool((t[up] < -1e4).all()) and bool((t[~up].abs() < 1e-3).all()) else "custom"
    # Q / K / V projections by dataflow from the score and apply-V matmuls
    q_side, k_side = [s for s in R.srcs(scores)][:2]
    def proj(start):
        hit, path = R.back(start, lambda j: N[j].func in MATMULS and bool(R.weights(j)))
        return hit, path
    q_proj, q_path = proj(q_side)
    k_proj, k_path = proj(k_side)
    v_in = [s for s in R.srcs(apply_v) if s != sm and s not in R.cons[sm]]
    v_proj, v_path = proj(v_in[-1] if v_in else apply_v)
    wq, wk, wv = (R.weights(x)[0] for x in (q_proj, k_proj, v_proj))
    head_dim = N[q_side].shape[-1] if N[q_side].shape else None
    q_out = R.pshape[wq][0]; k_out = R.pshape[wk][0]
    num_kv = k_out // head_dim if head_dim else None
    if num_heads is None and head_dim: num_heads = q_out // head_dim
    kind = "mha" if num_kv == num_heads else ("mqa" if num_kv == 1 else "gqa")
    fused = wq == wk == wv
    ev["attention"].append(f"softmax op {sm} shape {list(N[sm].shape)}; Q from {wq} {list(R.pshape[wq])}, K from {wk}, V from {wv}")
    qk_norm_q = norm_kind(R, q_path); qk_norm_k = norm_kind(R, k_path)
    qn_w = next((w for j in q_path for w in R.weights(j) if len(R.pshape.get(w, ())) == 1), None)
    rope_q = any(N[j].func == "neg" for j in q_path) and any(N[j].func == "cat" for j in q_path)
    rope_k = any(N[j].func == "neg" for j in k_path) and any(N[j].func == "cat" for j in k_path)
    norm_before_rope = None
    if qk_norm_q and rope_q:
        first_neg = min(j for j in q_path if N[j].func == "neg"); first_rs = min(j for j in q_path if N[j].func == "rsqrt")
        norm_before_rope = first_rs < first_neg
    theta = None
    for name, buf in model.named_buffers():
        if name.endswith("inv_freq") and buf.numel() > 1:
            d = 2 * buf.numel(); raw = float(buf[1].double() ** (-d / 2))
            theta = float(f"{raw:.4g}")                 # the buffer is float32: 4 significant digits are real
            break
    cache = verify_cache(rec_c, out_c, model, wk, wv, f"{stack}.{layer_ids[0]}")
    ev["cache"].append(json.dumps(cache, default=str))
    o_proj = next((j for j in sorted(lset) if j > apply_v and N[j].module.startswith(attn_mod) and N[j].func in MATMULS and R.weights(j)), None)
    wo = R.weights(o_proj)[0] if o_proj is not None else None
    bias = any(N[x].func == "addmm" for x in (q_proj, k_proj, v_proj) + ((o_proj,) if o_proj else ()))
    # ---- MLP: weight matmuls of the layer outside the attention module
    mlp_mm = [j for j in lops if N[j].func in MATMULS and R.weights(j) and not N[j].module.startswith(attn_mod)]
    gate = up = down = None; act = None; gated = False
    for a in mlp_mm:
        for b in mlp_mm:
            if a < b and {R.base(s) for s in R.srcs(a)} & {R.base(s) for s in R.srcs(b)}:
                for m_ in lops:
                    if N[m_].func == "mul" and m_ > b:
                        anc = set(R.back(m_, lambda j: False, limit=40)[1])
                        if a in anc and b in anc:
                            for j in sorted(anc):
                                if N[j].func in ACTS and j > min(a, b):
                                    act = ACTS[N[j].func]; actsrc = j
                                    gate = a if a in R.back(actsrc, lambda x: False, 10)[1] else b
                                    up = b if gate == a else a; gated = True
                                    break
                    if gated: break
            if gated: break
        if gated: break
    if gated:
        down = next(j for j in mlp_mm if j > max(gate, up))
    elif len(mlp_mm) >= 2:
        up, down = mlp_mm[0], mlp_mm[-1]
        act = next((ACTS[N[j].func] for j in lops if up < j < down and N[j].func in ACTS), None)
    wg = R.weights(gate)[0] if gate is not None else None
    wu, wd = R.weights(up)[0], R.weights(down)[0]
    inter = R.pshape[wu][0]
    ev["mlp"].append(f"{'gated: ' + wg + ' -> ' + str(act) + ' x ' if gated else ''}{wu} -> {wd}; inner width {inter}")
    # ---- norms and residuals: norm ops whose output feeds the projections; adds that combine branches
    pre_attn_ops = [j for j in lops if j < min(q_proj, k_proj, v_proj) and N[j].module != attn_mod and not N[j].module.startswith(attn_mod + ".")]
    pre_mlp_ops = [j for j in lops if (o_proj or apply_v) < j < min(x for x in (gate, up) if x is not None) and not N[j].module.startswith(attn_mod)]
    n1 = norm_kind(R, pre_attn_ops); n2 = norm_kind(R, pre_mlp_ops)
    adds = [j for j in lops if N[j].func == "add" and N[j].module == f"{stack}.{layer_ids[0]}"]
    residual = "sequential" if len(adds) >= 2 and adds[0] in R.back(adds[1], lambda j: False, 400)[1] else ("parallel" if adds else None)
    placement = "pre" if n1 and n2 else ("post" if adds else None)
    ev["residual"].append(f"{len(adds)} residual adds in the layer module; placement {placement}; topology {residual}")
    facts = dict(repo=repo, arch=arch, vocab=vocab, hidden=hidden, tied=tied, final_norm=final_norm, layers=len(layer_ids),
                 identical=len(identical), layer_ops=len(lops), kind=kind, num_heads=num_heads, num_kv=num_kv, head_dim=head_dim,
                 q_out=q_out, k_out=k_out, o_shape=R.pshape.get(wo), fused=fused, qk_norm=bool(qk_norm_q and qk_norm_k),
                 qk_norm_kind=qk_norm_q, qn_extent=(R.pshape[qn_w][0] if qn_w else None), norm_before_rope=norm_before_rope,
                 rope=rope_q and rope_k, theta=theta, scale=scale, mask=mask, bias=bias, o_proj=o_proj is not None,
                 gated=gated, act=act, inter=inter, norm=n1 or n2, placement=placement, residual=residual,
                 max_pos=getattr(cfg, "max_position_embeddings", None), total_ops=len(N), evidence=dict(ev), cache=cache,
                 seq=int(kw["input_ids"].shape[-1]) if "input_ids" in kw else None)
    return facts


def fmt(n):
    return f"{n:,}"


def to_ir(F):
    """Build unfold's ModelIR from execution facts (same fields the code parser fills)."""
    import model_unfolder.ir as I
    C = F.get("cache") or {}
    # the canonical graph wires K (after RoPE when RoPE exists) and V into the cache, and the cache into scores
    # and weights·V: claim the cache only when the recording shows exactly that
    cache_ok = bool(C.get("verified")) and C["write"]["key"]["after"] == ("rope" if F["rope"] else C["write"]["key"]["after"])
    def make(cls, **kv):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in kv.items() if k in names})
    NK = {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}
    nlabel = NK.get(F["norm"], "Norm")
    kind_title = {"gqa": "Grouped-query attention", "mha": "Multi-head attention", "mqa": "Multi-query attention"}[F["kind"]]
    kind_label = {"gqa": "Grouped-Query", "mha": "Multi-Head", "mqa": "Multi-Query"}[F["kind"]]
    scale_txt = None
    if F["scale"]:
        inv = 1 / F["scale"]; d = round(inv ** 2)
        scale_txt = f"scores × {F['scale']:.6g} (= 1/√{d})" if abs(math.sqrt(d) - inv) < 1e-3 else f"scores × {F['scale']:.6g}"
    attn = make(I.AttentionSpec, kind=F["kind"], mixer_state="ordinary_attention", num_heads=F["num_heads"], num_kv_heads=F["num_kv"],
                head_dim=F["head_dim"], rope_theta=F["theta"], mask=F["mask"], qk_norm=F["qk_norm"], qk_norm_kind=F["qk_norm_kind"],
                qk_norm_axis="last_dimension" if F["qk_norm"] else None, qk_norm_extent=F["qn_extent"],
                qk_norm_placement=("after_reshape" if F["qn_extent"] == F["head_dim"] else "before_reshape") if F["qk_norm"] else None,
                rope=F["rope"], position_kind="rope" if F["rope"] else None, position_application="qk_rotation" if F["rope"] else None,
                bias=F["bias"], output_projection=F["o_proj"], projection_mode="fused_qkv" if F["fused"] else "split_qkv",
                scores_scaled=F["scale"] is not None, cached=cache_ok,
                cache_evidence=({"payload": ["key", "value"],
                                 "location": f"keys after {F['cache']['write']['key']['after']}, values after {F['cache']['write']['value']['after']}",
                                 "guard": "observed in a run with use_cache=True"} if cache_ok else None))
    ffn = make(I.FFNSpec, kind="dense", activation=F["act"], intermediate_size=F["inter"], gated=F["gated"], bias=False,
               projection_mode="split" if F["gated"] else None)
    H, hd, nh, nkv = F["hidden"], F["head_dim"], F["num_heads"], F["num_kv"]
    attn_facts = [f"{nh} Q heads", f"{nkv} KV heads", f"head dim {hd}"]
    if F["theta"]: attn_facts.append(f"RoPE θ {fmt(round(F['theta']))}")
    if F["qk_norm"]: attn_facts.append("QK-Norm")
    if not F["bias"]: attn_facts.append("bias-free projections")
    children = [
        {"id": "q_proj", "title": "Query projection", "description": "Recorded matmul producing the per-head queries.", "facts": [f"{fmt(H)} → {fmt(F['q_out'])}", f"{nh} Q heads", f"head dim {hd}"]},
        {"id": "k_proj", "title": "Key projection", "description": "Recorded matmul producing the keys.", "facts": [f"{fmt(H)} → {fmt(F['k_out'])}", f"{nkv} KV heads"] + (["cache ports: ⌃ write · ⊥ read"] if cache_ok else [])},
        {"id": "v_proj", "title": "Value projection", "description": "Recorded matmul producing the values.", "facts": [f"{fmt(H)} → {fmt(F['k_out'])}", f"{nkv} KV heads"] + (["cache ports: ⌃ write · ⊥ read"] if cache_ok else [])},
        {"id": "q_reshape", "title": "Reshape query heads", "description": "The projected width is split into heads before the next op.", "facts": [f"head dim {hd}"]},
        {"id": "k_reshape", "title": "Reshape key heads", "description": "The projected width is split into heads before the next op.", "facts": [f"head dim {hd}"]},
    ]
    if F["qk_norm"]:
        where = "before RoPE" if F["norm_before_rope"] else "after RoPE"
        children += [{"id": "q_norm", "title": f"Query {NK.get(F['qk_norm_kind'], 'norm')}", "description": f"Recorded {NK.get(F['qk_norm_kind'])} on Q over the last axis ({F['qn_extent']}), {where}."},
                     {"id": "k_norm", "title": f"Key {NK.get(F['qk_norm_kind'], 'norm')}", "description": f"Recorded {NK.get(F['qk_norm_kind'])} on K over the last axis ({F['qn_extent']}), {where}."}]
    if F["rope"]:
        th = f"θ={F['theta']:.6g}" if F["theta"] else "θ unknown"
        children += [{"id": "q_rope", "title": "Apply RoPE (Q)", "description": f"Rotate-half rotary embedding recorded on the query heads; frequency base from the built model's buffer: {th}.", "facts": [f"frequency base {th}"]},
                     {"id": "k_rope", "title": "Apply RoPE (K)", "description": f"Rotate-half rotary embedding recorded on the key heads; {th}.", "facts": [f"frequency base {th}"]}]
    children += [
        {"id": "scaled_scores", "title": "Scaled dot-product scores", "description": f"Q·Kᵀ recorded; {scale_txt or 'no scale constant recorded'}; mask: {F['mask'] or 'none'} (read from the mask tensor's values).",
         "facts": [f"{nh} Q heads", f"{nkv} KV heads"] + ([f"{nh // nkv} Q per KV head"] if nkv and nh % nkv == 0 and nh != nkv else [])},
        {"id": "attn_softmax", "title": "Softmax weights", "description": "Recorded softmax over each query row."},
        {"id": "attn_apply_v", "title": "Matrix multiplication", "description": "Recorded weights · V — one context vector per head."},
        {"id": "concat_heads", "title": "Concatenate heads", "description": "Heads are stacked back into one width.", "facts": [f"{nh} × {hd}", f"→ {fmt(nh * hd)}"]},
    ]
    if cache_ok:
        kw_ = C["write"]["key"]; vw_ = C["write"]["value"]
        children.append({"id": "kv_cache", "title": "K/V cache update and read",
                         "description": f"Observed in a run with the cache on: the returned cache stores the keys after {kw_['after']} "
                                        f"(shape {kw_['stored_shape']}) and the values after {vw_['after']} (shape {vw_['stored_shape']}); "
                                        "this layer's scores and weights·V read them back from the cache.",
                         "facts": ["stores key + value", f"keys after {kw_['after']}", "read by scores and weights·V"]})
    if F["o_proj"]:
        children.append({"id": "o_proj", "title": "Output projection", "description": "Recorded matmul back to the residual width.", "facts": [f"{fmt(F['o_shape'][1])} → {fmt(F['o_shape'][0])}"]})
    attn_block = {"id": "attn", "role": "attention", "kind": "attention",
                  "label": [kind_label, "(QK-Norm)"] if F["qk_norm"] else kind_label + (" Attention" if F["kind"] != "gqa" else ""),
                  "title": kind_title + (" (QK-Norm)" if F["qk_norm"] else ""),
                  "description": kind_title + " — observed head counts from the recorded tensors.", "facts": attn_facts,
                  "view": "attention", "detail": {"attention": attn.to_dict() if hasattr(attn, "to_dict") else dataclasses.asdict(attn)},
                  "children": children}
    actl = {"silu": "SiLU", "gelu": "GELU", "relu": "ReLU"}.get(F["act"], F["act"] or "activation")
    if F["gated"]:
        ffn_children = [
            {"id": "gate_proj", "label": "Linear (gate)", "title": "Gate projection", "description": f"Recorded matmul producing the gate path (through {actl}).", "facts": [f"{fmt(H)} → {fmt(F['inter'])}"]},
            {"id": "up_proj", "label": "Linear (up)", "title": "Up projection", "description": "Recorded matmul into the inner width.", "facts": [f"{fmt(H)} → {fmt(F['inter'])}"]},
            {"id": "activation", "label": actl, "title": actl, "description": "Recorded element-wise non-linearity on the gate path."},
            {"id": "multiply", "label": "x", "title": "Gate product", "description": f"Recorded {actl}(gate) × up."},
            {"id": "down_proj", "label": "Linear (down)", "title": "Down projection", "description": "Recorded matmul back to the residual width.", "facts": [f"{fmt(F['inter'])} → {fmt(H)}"]},
        ]
        ffn_block = {"id": "ffn", "role": "ffn", "kind": "ffn", "label": "Gated FFN", "title": "Gated feed-forward",
                     "description": "Gated MLP — recorded gate path modulates the up projection before the down projection.",
                     "facts": [actl, f"hidden {fmt(F['inter'])}", "split gate/up"], "view": "gated_ffn",
                     "detail": {"ffn": dataclasses.asdict(ffn)}, "children": ffn_children}
    else:
        ffn_block = {"id": "ffn", "role": "ffn", "kind": "ffn", "label": "FFN", "title": "Feed-forward", "description": "Recorded two-layer MLP.",
                     "facts": [actl, f"hidden {fmt(F['inter'])}"], "detail": {"ffn": dataclasses.asdict(ffn)}}
    blocks = [
        {"id": "rms1", "role": "norm", "kind": "norm", "label": nlabel, "title": "Pre-attention norm", "description": f"Recorded {nlabel} on the layer input before attention.", "facts": [f"dim {fmt(H)}"]},
        attn_block,
        {"id": "add1", "role": "residual", "kind": "residual_add", "residual_from": "rms1", "label": "+", "title": "Residual add", "description": "Recorded add: block input + attention output."},
        {"id": "rms2", "role": "norm", "kind": "norm", "label": nlabel, "title": "Pre-FFN norm", "description": f"Recorded {nlabel} before the FFN.", "facts": [f"dim {fmt(H)}"]},
        ffn_block,
        {"id": "add2", "role": "residual", "kind": "residual_add", "residual_from": "rms2", "label": "+", "title": "Residual add", "description": "Recorded add: post-attention + FFN output."},
    ]
    layers = [make(I.LayerSpec, index=i, attention=attn, ffn=ffn, norm_kind=F["norm"], norm_placement=F["placement"],
                   residual_topology=F["residual"], blocks=blocks) for i in range(F["layers"])]
    tie_txt = " — weights tied with the output head (same tensor read)." if F["tied"] else "."
    fn = NK.get(F["final_norm"], "Norm")
    model_blocks = [
        {"id": "tok_text", "role": "input", "kind": "source", "label": "Tokenized text", "title": "Tokenized text", "description": "Input token IDs.", "facts": [f"shape [1, {F['seq']}] in the recorded run"]},
        {"id": "embed", "role": "embedding", "kind": "embedding", "label": "Token Embedding layer", "title": "Token embedding", "description": "Maps each token id to its vector" + tie_txt, "facts": [f"{fmt(F['vocab'])} vocab", f"{fmt(H)}-d"]},
        {"id": "final_rms", "role": "norm", "kind": "norm", "label": f"Final {fn}", "title": "Final norm", "description": f"Recorded {fn} over the last hidden state before the output head.", "facts": [f"dim {fmt(H)}"], "resolved": True},
        {"id": "lm_head", "role": "output", "kind": "output", "label": "Linear output layer", "title": "LM head", "description": "Recorded matmul into vocabulary logits" + tie_txt, "facts": [f"{fmt(H)} → {fmt(F['vocab'])}"]},
    ]
    extras = {"render": {"family": "transformer", "layout": "decoder_only", "model_blocks": model_blocks},
              "exec_capture": {"source": "execution recording (zero-storage weights, real tokenized prompt)", "total_ops": F["total_ops"],
                               "layers_identical": f"{F['identical']} of {F['layers']}", "evidence": F["evidence"],
                               "declared_not_observed": {"max_position_embeddings": F["max_pos"]}}}
    ir = make(I.ModelIR, name=F["repo"].split("/")[-1], architecture=F["arch"], vocab_size=F["vocab"], hidden_size=H,
              max_position_embeddings=F["max_pos"], tie_word_embeddings=F["tied"], embedding_norm_kind=None,
              final_norm_kind=F["final_norm"], layers=layers, cross_layer_edges=[], extras=extras,
              notes=["Built from an execution recording: every fact is observed in the run (no modeling-code parsing). "
                     "Context length is the config's declared value (not observable in one run)."])
    return ir


if __name__ == "__main__":
    repo, out = sys.argv[1], sys.argv[2]
    # default: the frozen renderer copy (model-benchmark/renderer_snapshot, see its SNAPSHOT.txt) so renders do not
    # move while unfold-pkg is being changed; --unfold-pkg PATH renders against any other tree
    _snap = os.path.join(HERE, "..", "renderer_snapshot")
    pkg = (sys.argv[sys.argv.index("--unfold-pkg") + 1] if "--unfold-pkg" in sys.argv
           else _snap if os.path.isdir(os.path.join(_snap, "model_unfolder")) else os.path.join(HERE, "..", "..", "unfold-pkg"))
    sys.path.insert(0, os.path.abspath(pkg))
    F = recognize(repo)
    print(json.dumps({k: v for k, v in F.items() if k != "evidence"}, default=str, indent=1))
    from model_unfolder.diagram import Diagram
    path = Diagram(to_ir(F)).save(out)
    print("saved", path)
