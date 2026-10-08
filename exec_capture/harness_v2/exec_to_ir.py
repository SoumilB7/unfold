"""Execution recording -> unfold's ModelIR -> unfold's own renderer.

The model is built with zero-storage weights, run on a real tokenized prompt, and every op is recorded
(DagRecorder). The recognizers below read that recording and fill the same ModelIR the code parser fills
(model facts, one LayerSpec per executed layer, the per-layer block tree, the outer blocks). Every fact
comes from what ran: tensor shapes, the weights each op read, recorded constants, the built model's own
buffers, and the values of activations (the attention mask). Nothing is read from modeling source.
The unfold package is imported read-only; its Diagram renders the result unchanged.

Scope of this first version: decoder-only transformers (attention + dense MLP).
Usage: python exec_to_ir.py <repo> <out.html> [--bundle PATH | --captures DIR [--refresh]] [--unfold-pkg PATH]
The recording is the capture bundle harness v2 wrote for the run it graded (worker_v2.py with BENCH_EMIT_CAPTURE):
one execution, one truth. Without --bundle the bundle is taken from --captures (default ../captures) and produced
there by running the worker once if missing.
"""
import os, re, sys, json, math, collections, dataclasses
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capture

MATMULS = {"mm", "addmm", "matmul", "linear", "bmm", "baddbmm"}
VIEWS = {"view", "_unsafe_view", "unsqueeze", "squeeze", "expand", "t", "transpose", "reshape", "contiguous",
         "clone", "slice", "select", "alias", "detach", "_to_copy", "to", "permute", "lift_fresh", "repeat_interleave"}
SOFTMAX = {"_softmax", "softmax", "_safe_softmax"}
ACTS = {"silu": "silu", "gelu": "gelu", "relu": "relu", "sigmoid": "sigmoid", "tanh": "tanh"}
# ops that move or retype values without computing new ones (never need a drawn block of their own)
TRANSPARENT = VIEWS | {"dropout", "native_dropout", "type_as", "copy", "copy_", "lift_fresh_copy", "cat_identity"}
NORM_OPS = {"pow", "mean", "add", "rsqrt", "mul", "sub", "var", "var_mean", "native_layer_norm", "layer_norm", "div", "sqrt"}
ROPE_OPS = {"neg", "cat", "mul", "add", "slice", "select", "stack", "flatten", "view_as_complex", "view_as_real", "chunk", "split"}
ROUTING = {"topk", "argmax", "sort", "argsort", "max"}          # the worker's own router evidence
ROPE_STRUCT = {"neg", "cat", "slice", "select", "stack", "flatten", "chunk", "split", "unbind", "view_as_complex",
               "view_as_real"}                                   # rotate-half / complex-rotation plumbing
STAT = {"rsqrt", "native_layer_norm", "layer_norm", "var", "var_mean", "sqrt"}
TRIG = {"cos", "sin", "polar"}


class Rec:
    def __init__(self, nodes, pshape):
        self.n = nodes
        self.pshape = pshape
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
        return sorted(out, key=lambda w: len(self.pshape.get(w, ())) < 2)    # matrices before biases (stable)

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


class NotDrawn(Exception):
    """The recording shows something this adapter cannot draw truthfully yet; the evidence says what."""


def norm_kind(R, ops):
    f = {R.n[i].func for i in ops}
    if "native_layer_norm" in f or "layer_norm" in f: return "layernorm"
    if "rsqrt" in f and ("pow" in f or "mul" in f) and "mean" in f:
        return "rmsnorm" if "sub" not in f else "layernorm"
    return None


def verify_cache(cache_run, pshape, wk, wv, layer_prefix):
    """Observed K/V cache behaviour: which op produced the tensors stored in the returned cache, and whether this
    layer's attention read its keys/values from them. Returns {} if no cache was returned. `cache_run` is the
    bundle's cache-on run of the same pass (its op graph + every stored 4-D tensor with its producing op)."""
    if not cache_run or not cache_run.get("cache"):
        return {}
    found = cache_run["cache"]["slots"]
    R = Rec(capture.nodes_of(cache_run["graph"]), pshape); N = R.n
    is_w = lambda j: N[j].func in MATMULS and bool(R.weights(j))
    write = {}
    for t in found:
        p = t["producer"]
        if p is None or not N[p].module.startswith(layer_prefix): continue
        hit, path = R.back(p, is_w)
        if hit is None: continue
        w_ = R.weights(hit)[0]
        role = "key" if w_ == wk else ("value" if w_ == wv else None)
        if role and role not in write:
            funcs = {N[j].func for j in path}
            after = "rope" if ("neg" in funcs and "cat" in funcs) else ("norm" if "rsqrt" in funcs else "projection")
            write[role] = {"producer_op": p, "after": after, "stored_shape": list(t["shape"])}
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


CAPTURES = os.path.join(HERE, "..", "captures")


def bundle_for(repo, captures=CAPTURES, refresh=False):
    """The capture bundle of the run harness v2 graded for this repo; runs the worker once (emitting) if missing."""
    p = capture.path_for(captures, repo)
    if refresh or not os.path.exists(p):
        import subprocess, tempfile, shutil
        hf = tempfile.mkdtemp(prefix="hfc_ir_")
        res = os.path.join(captures, repo.replace("/", "__") + ".result.json")
        env = dict(os.environ, BENCH_EMIT_CAPTURE=captures, HF_HUB_CACHE=hf, TOKENIZERS_PARALLELISM="false",
                   BENCH_HEADER_CACHE=os.environ.get("BENCH_HEADER_CACHE", os.path.join(HERE, "..", "cache_v2", "headers")))
        try:
            subprocess.run([sys.executable, os.path.join(HERE, "worker_v2.py"), repo, res], env=env, check=False)
        finally:
            shutil.rmtree(hf, ignore_errors=True)
        if not os.path.exists(p):
            raise RuntimeError(f"the worker wrote no capture bundle for {repo} (see {res})")
    return capture.load(p)


def recognize(bundle):
    """Facts from a capture bundle, or NotDrawn with the evidence when the recording is outside what this adapter
    draws truthfully (it never crashes and never draws a guess)."""
    try:
        return _recognize(bundle)
    except NotDrawn:
        raise
    except Exception as e:
        import traceback
        fr = traceback.extract_tb(e.__traceback__)[-1]
        raise NotDrawn(f"the recognizer could not follow this recording ({type(e).__name__}: {str(e)[:120]} at line "
                       f"{fr.lineno}): its structure is outside the supported decoder-only dense pattern") from e


def _recognize(bundle):
    """Facts from a capture bundle: the first main pass harness v2 ran (and graded), the executed class."""
    repo, arch, cfg = bundle["repo"], bundle["class"], bundle.get("config") or {}
    if not bundle.get("pass_order") or bundle["pass_order"][0] not in bundle.get("passes", {}):
        if bundle.get("verdict") in ("OUT", "FAIL"):
            raise NotDrawn(f"the model did not run (verdict {bundle.get('verdict')}: {str(bundle.get('reason') or '')[:160]})")
        raise NotDrawn("no main pass was recorded (the run was graded on synthesized or sub-model passes only)")
    label = bundle["pass_order"][0]
    P = bundle["passes"][label]
    if P.get("squeezed_retry"):
        raise NotDrawn(f"pass '{label}' needed the 5-D pixel retry: its recording also holds the failed attempt")
    pshape = {k: tuple(v) for k, v in bundle["model"]["param_shapes"].items()}
    R = Rec(capture.nodes_of(P["graph"]), pshape)
    N = R.n
    for j, nd in enumerate(N):                  # cat with an empty operand (e.g. full-width rotary's empty pass-through)
        if nd.func == "cat":
            shp = [N[e[1]].shape if e[0] == "op" else (e[1] if e[0] == "external" else None) for e in nd.ins]
            if len(shp) == 2 and any(x is not None and len(x) and x[-1] == 0 for x in shp):
                nd.func = "cat_identity"
    square_adds = capture.square_tensors(P["squares"])
    ev = collections.defaultdict(list)                     # fact -> evidence strings

    # ---- the repeated layer stack: the module prefix whose numbered children executed the most
    pref = collections.defaultdict(set)
    for nd in N:
        m = re.match(r"^(.*?)\.(\d+)(?:\.|$)", nd.module)
        if m: pref[m.group(1)].add(int(m.group(2)))
    if not pref:
        raise NotDrawn("no repeated numbered layer stack executed")
    stacks = sorted(p for p in pref if len(pref[p]) >= 2)
    if len(stacks) > 1:
        raise NotDrawn(f"more than one repeated numbered module stack executed ({', '.join(stacks)}): only a single "
                       "layer stack is drawn yet")
    stack = max(pref, key=lambda p: len(pref[p]))
    pass_stacks = dict(bundle.get("pass_stacks") or {})
    for l2, P2 in bundle["passes"].items():                 # bundles without per-pass stacks: the kept passes
        if l2 not in pass_stacks:
            pass_stacks[l2] = sorted({mm_.group(1) for func, mod, *_ in P2["graph"]["nodes"]
                                      for mm_ in [re.match(r"^(.*?)\.(\d+)(?:\.|$)", P2["graph"]["strings"][mod])] if mm_})
    for l2, st2 in pass_stacks.items():
        if l2 == label: continue
        extra = sorted(set(st2) - set(pref))
        if extra:
            raise NotDrawn(f"the graded run also executed {', '.join(extra)} (pass '{l2}'), which a drawing of pass "
                           f"'{label}' would omit: multi-input models are not drawn yet")
    layer_ids = sorted(pref[stack])
    L = {i: [j for j, nd in enumerate(N) if nd.module == f"{stack}.{i}" or nd.module.startswith(f"{stack}.{i}.")] for i in layer_ids}
    routing = sorted({N[j].func for i in layer_ids for j in L[i] if N[j].func in ROUTING})
    if routing:
        raise NotDrawn(f"data-dependent routing inside the layers ({', '.join(routing)} ops): mixture-of-experts "
                       "layers are not drawn yet")
    ev["layers"].append(f"{len(layer_ids)} numbered children of {stack} executed")

    # identical structure across layers
    def normseq(ops):
        nm = lambda x: re.sub(rf"^{re.escape(stack)}\.\d+", "L", x)
        return [(nm(N[j].module), N[j].func, tuple(N[j].consts),
                 tuple(sorted(nm(e[1]) for e in N[j].ins if e[0] in ("param", "buffer")))) for j in ops]
    ref = normseq(L[layer_ids[0]])
    identical = [i for i in layer_ids if normseq(L[i]) == ref]
    if len(identical) < len(layer_ids):
        odd = [i for i in layer_ids if i not in identical]
        raise NotDrawn(f"layers differ: {len(identical)} of {len(layer_ids)} run layer 0's ops, constants and weights "
                       f"(others: {odd[:6]}{'...' if len(odd) > 6 else ''}); per-layer drawing is not implemented yet")
    # the RoPE config gate: only default and llama3 frequencies (llama3 rescales long wavelengths only; the drawn base
    # is checked against the buffer itself below); every other scaling type also changes what the drawing shows
    rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    rope_cfg = [v for v in ([rs] + [x for x in rs.values() if isinstance(x, dict)]) if isinstance(v, dict)]
    rtypes = {v.get("rope_type") or v.get("type") for v in rope_cfg} - {None, "default"}
    if rtypes - {"llama3"}:
        raise NotDrawn(f"RoPE with scaling type {sorted(rtypes)} (config): its frequencies are not drawn yet")
    rope_note = next((f"llama3 frequency scaling declared (factor {v.get('factor')}, original context "
                      f"{v.get('original_max_position_embeddings')}): long wavelengths rescaled"
                      for v in rope_cfg if (v.get("rope_type") or v.get("type")) == "llama3"), None)


    # ---- embedding / LM head / final norm
    emb = next(j for j, nd in enumerate(N) if nd.func == "embedding")
    emb_w = R.weights(emb)[0]
    vocab, hidden = R.pshape[emb_w]
    ev["embedding"].append(f"op {emb} embedding reads {emb_w} {list(R.pshape[emb_w])}")
    # the hidden-state path: every op downstream of the token embedding (positions / masks are side inputs)
    hid, frontier = {emb}, [emb]
    while frontier:
        nxt = []
        for j in frontier:
            for c in R.cons[j]:
                if c not in hid: hid.add(c); nxt.append(c)
        frontier = nxt
    def _off(j, k):
        """Is tensor operand k of op j off the hidden-state path (a mask, a table, a parameter, an input)?"""
        e = N[j].ins[k] if k < len(N[j].ins) else None
        return e is not None and not (e[0] == "op" and e[1] in hid)
    def _off_ops(j):
        return [e[1] for e in N[j].ins if e[0] == "op" and e[1] not in hid]
    def _rot_operands(j):
        """Off-path operands of op j computed from cos / sin (the rotation tables)."""
        return [x for x in _off_ops(j) if any(N[y].func in TRIG for y in R.back(x, lambda z: False, 200)[1] + [x])]
    def _param_side(e):
        """A norm weight / bias: a 1-D parameter, or an off-path value computed only from parameters (e.g. 1 + w)."""
        if e[0] == "param": return len(R.pshape.get(e[1], ())) == 1
        if e[0] == "op" and e[1] not in hid:
            anc = R.back(e[1], lambda z: False, 50)[1] + [e[1]]
            kinds = {x[0] for y in anc for x in N[y].ins}
            return "param" in kinds and not kinds & {"input", "external", "buffer"}
        return False
    def _norm_match(region):
        """(kind, ops) if `region` is exactly one normalization, else (None, set()).
        RMSNorm: [x^2 -> mean -> + eps] -> rsqrt -> x * r -> [* weight];  LayerNorm: native_layer_norm, or the manual
        mean / center / variance / + eps / rsqrt|sqrt -> scale -> [* weight] -> [+ bias]. Order is checked: before the
        statistic only the statistic's own ops, one scaling op, then at most one weight and one bias (1-D parameters)."""
        region = set(region)
        unit = {j for j in region if N[j].func in ("mul", "div") and N[j].consts and all(c in (1, 1.0) for c in N[j].consts)
                and not _off_ops(j) and not any(e[0] == "param" for e in N[j].ins)}          # x 1.0 changes nothing
        core = [j for j in region if N[j].func not in TRANSPARENT and j not in unit]
        if not core: return None, set()
        if len(core) == 1 and N[core[0]].func in ("native_layer_norm", "layer_norm"):
            return "layernorm", region
        stat = [j for j in core if N[j].func in ("rsqrt", "sqrt")]
        if len(stat) != 1: return None, set()
        r = stat[0]
        def anc(j):                                     # region ancestors of j
            got, fr = set(), [j]
            while fr:
                nx = []
                for x in fr:
                    for y in R.srcs(x):
                        if y in region and y not in got: got.add(y); nx.append(y)
                fr = nx
            return got
        before = anc(r)
        centered = False
        for j in before:
            nd = N[j]; f = nd.func
            if f in TRANSPARENT or f in ("mean", "var", "var_mean"): continue
            if f == "pow" and list(nd.consts) in ([2], [2.0]): continue
            if f == "add" and nd.consts and 0 < abs(nd.consts[0]) <= 1e-2 and len(R.srcs(j)) == 1: continue
            if f == "sub" and any(N[x].func == "mean" for x in R.srcs(j)): centered = True; continue
            return None, set()
        # the scaling op: x * rsqrt(..) or x / sqrt(..), its other operand from before the statistic (x or x - mean)
        # the centering may be recomputed for the scaling operand (x - mean again, e.g. Cohere's LayerNorm)
        recentered = {j for j in core if j not in before and N[j].func == "sub" and centered
                      and any(N[x].func == "mean" for x in R.srcs(j))}
        after = [j for j in core if j != r and j not in before and j not in recentered]
        from_r = lambda x: x == r or (N[x].func in TRANSPARENT and r in anc(x))          # the statistic (or a cast)
        scl = [j for j in after if N[j].func in ("mul", "div") and not N[j].consts
               and not any(e[0] == "param" for e in N[j].ins)
               and any(from_r(x) for x in R.srcs(j))
               and any(x != r and r not in anc(x) for x in R.srcs(j))]                   # times the (centered) input
        if len(scl) != 1: return None, set()
        rest = [j for j in after if j != scl[0]]
        wmul = [j for j in rest if N[j].func == "mul" and any(_param_side(e) for e in N[j].ins) and not N[j].consts]
        badd = [j for j in rest if N[j].func == "add" and any(_param_side(e) for e in N[j].ins) and not N[j].consts]
        if len(wmul) > 1 or len(badd) > 1 or len(wmul) + len(badd) != len(rest): return None, set()
        if badd and wmul and not (wmul[0] in anc(badd[0])): return None, set()      # weight before bias
        if any(scl[0] not in anc(j) for j in rest): return None, set()                # both after the scaling
        if badd and not centered: return None, set()          # an RMSNorm with a bias has no drawn form
        region = region | unit
        return ("layernorm" if centered else "rmsnorm"), region

    def _norm_accept(region):
        return _norm_match(region)[1]

    def _norm_desc(region):
        """What the layout message says about a norm slot: its kind, 'none', or a normalization with extra ops."""
        k_, acc = _norm_match(region)
        if k_: return k_
        if any(N[j].func in STAT for j in region):
            extra = sorted({N[j].func for j in region if N[j].func not in TRANSPARENT | STAT | {"mean", "pow", "var"}})
            return f"a normalization with extra ops ({', '.join(extra) or 'order'})"
        return "none"

    def _region(target, lo):
        """Hidden-path ops after `lo` that feed `target` (walking back along the hidden-state path)."""
        got, frontier = set(), [target]
        while frontier:
            nxt = []
            for j in frontier:
                for s_ in R.srcs(j):
                    if s_ in hid and s_ > lo and s_ not in got: got.add(s_); nxt.append(s_)
            frontier = nxt
        return got

    def _thru(x):
        """The value an op stands for, looking through shape-only / cast ops."""
        seen = 0
        while N[x].func in TRANSPARENT and R.srcs(x) and seen < 20:
            x = R.srcs(x)[0]; seen += 1
        return x

    def _rope_match(ops):
        """The ops of exactly one rotation in `ops`: x*cos + rot(x)*sin with rot(x) = cat/stack(-half, half) of the
        same x, or one complex multiply by a polar table. None if the rotation ops are anything else."""
        ops = set(ops)
        trig = [j for j in ops if N[j].func == "mul" and _rot_operands(j)]
        hadd = [j for j in ops if N[j].func in ("add", "sub") and len(R.srcs(j)) == 2 and all(x in hid for x in R.srcs(j))]
        if any(any(e[0] == "param" for e in N[j].ins) for j in trig): return None
        cplx = {j for j in ops if N[j].func in ("view_as_complex", "view_as_real")}
        if len(trig) == 1 and cplx and not hadd:
            return {trig[0]} | cplx
        if len(trig) != 2 or len(hadd) != 1: return None
        A = hadd[0]
        if {_thru(x) for x in R.srcs(A)} != set(trig) and set(R.srcs(A)) != set(trig): return None
        hidden_in = {}
        for m_ in trig:
            hs = [x for x in R.srcs(m_) if x in hid]
            if len(hs) != 1: return None
            hidden_in[m_] = _thru(hs[0])
        rot = [m_ for m_ in trig if N[hidden_in[m_]].func in ("cat", "stack", "flatten")]
        if len(rot) != 1: return None
        x_m = [m_ for m_ in trig if m_ not in rot][0]; x = R.base(hidden_in[x_m])
        rc = hidden_in[rot[0]]
        if N[rc].func == "flatten": rc = _thru(R.srcs(rc)[0])
        parts = [_thru(y) for y in R.srcs(rc)]
        negs = [y for y in parts if N[y].func == "neg"]
        if len(parts) != 2 or len(negs) != 1: return None
        halves = [R.base(R.srcs(negs[0])[0])] + [R.base(y) for y in parts if y not in negs]
        if any(h != x for h in halves): return None                 # both halves of the same rotated tensor
        keep = {A, rc, negs[0], *trig, hidden_in[rot[0]]}
        return keep | {j for j in ops if N[j].func in TRANSPARENT}

    last = max(L[layer_ids[-1]])
    head = next((j for j in range(last + 1, len(N)) if N[j].func in MATMULS and R.weights(j)), None)
    if head is None:
        raise NotDrawn("no output projection runs after the layers (an encoder / embedding model): only models "
                       "ending in an output head are drawn yet")
    head_w = R.weights(head)[0]
    tied = head_w == emb_w or any(head_w in g and emb_w in g for g in bundle["model"]["tie_groups"])
    ev["tied"].append(f"LM head op {head} reads {head_w}; same tensor as {emb_w}: {tied}")
    fin_ops = [j for j in range(last + 1, head)]
    final_norm = _norm_match(_region(head, last) & set(fin_ops))[0]

    # ---- one layer in detail (all layers are checked identical above)
    lops = L[layer_ids[0]]; lset = set(lops)
    sm = next((j for j in lops if N[j].func in SOFTMAX), None)
    if sm is None:
        raise NotDrawn("layer 0 runs no softmax attention (a state-space / linear-attention mixer): not drawn yet")
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
    for j in sorted(set(pre_sm) | set(R.back(scores, lambda j: R.weights(j) != [])[1])):
        if N[j].func in ("mul", "div") and N[j].consts and not [e for e in N[j].ins if e[0] == "op"][1:]:
            c = N[j].consts[0]; f = c if N[j].func == "mul" else 1 / c
            scale = f if scale is None else scale * f
    # mask: the score addend from off the hidden-state path, as captured; causal if its upper triangle is -inf-like
    sq_ids = capture.square_ids(P["squares"])
    def _mask_adds(pre):
        out = []
        for j in pre:
            if j not in hid or N[j].func != "add": continue
            per = square_adds.get(j, {})
            if -1 in per:                                    # older bundle: the last square operand only
                out.append((j, -1)); continue
            ks = [k for k in per if _off(j, k)]
            if ks: out.append((j, ks[0]))
        return out
    madds = _mask_adds(pre_sm)
    mask = None
    if len(madds) == 1:
        j_, k_ = madds[0]
        t = square_adds[j_][k_]
        up = torch.triu(torch.ones_like(t, dtype=torch.bool), 1)
        mask = "causal" if bool((t[up] <= -1e4).all()) and bool((t[~up].abs() < 1e-3).all()) else "custom"
    def _mask_id(i):
        smi = next((j for j in L[i] if N[j].func in SOFTMAX), None)
        if smi is None: return None
        ms = _mask_adds(R.back(smi, lambda j: N[j].func in ("bmm", "matmul") and not R.weights(j))[1])
        return tuple(sq_ids[j][k] for j, k in ms)
    masks_seen = {_mask_id(i) for i in layer_ids}
    if len(masks_seen) > 1:
        raise NotDrawn(f"the attention masks differ between layers ({len(masks_seen)} distinct, e.g. sliding-window "
                       "and full layers): per-layer masks are not drawn yet")
    if mask is None and not madds:
        inplace = sorted({N[j].func for j in pre_sm if j in hid and N[j].func in ("add_", "masked_fill", "masked_fill_", "where", "mul_")})
        if inplace:
            raise NotDrawn(f"the mask is applied with {', '.join(inplace)} on the scores, so its values are not captured: "
                           "the mask is unknown")
        # a broadcast [.., 1, S] addend from off the hidden path is a padding mask: every query sees every key
        pad = [j for j in pre_sm if j in hid and N[j].func == "add"
               for x in _off_ops(j) if N[x].shape and len(N[x].shape) >= 2 and N[x].shape[-2] == 1]
        if pad:
            raise NotDrawn("the scores get a broadcast padding mask (every query attends to every key: bidirectional "
                           "encoder attention): only causal decoders are drawn yet")
        if not any(j in hid and N[j].func in ("add", "add_", "masked_fill", "masked_fill_", "where") for j in pre_sm):
            raise NotDrawn("nothing masks the scores (every query attends to every key: unmasked, bidirectional "
                           "attention): only causal decoders are drawn yet")
    if mask is None:
        raise NotDrawn(f"{'no' if not madds else len(madds)} additive mask operand(s) found on the scores (masked_fill / "
                       "where / in-place add, extra score biases, or a sequence longer than the captured side): the "
                       "mask is unknown")
    if mask == "custom":
        raise NotDrawn("the attention mask added to the scores is not causal (bidirectional, sliding or custom): "
                       "only causal masks are drawn yet")
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
    if fused:
        raise NotDrawn(f"Q, K and V come from one fused projection ({wq}): its split into Q/K/V widths is not drawn yet")
    if R.pshape[wq][-1] != hidden and R.pshape[wq][0] == hidden:
        raise NotDrawn(f"projection weights are stored [in, out] ({wq} {list(R.pshape[wq])}): that layout is not drawn yet")
    if head_dim is None or num_kv is None or num_heads is None:
        raise NotDrawn("heads / head width / K-V heads not observable from the recorded shapes")
    ev["attention"].append(f"softmax op {sm} shape {list(N[sm].shape)}; Q from {wq} {list(R.pshape[wq])}, K from {wk}, V from {wv}")
    qk_norm_q = norm_kind(R, q_path); qk_norm_k = norm_kind(R, k_path)
    qn_w = next((w for j in q_path for w in R.weights(j) if len(R.pshape.get(w, ())) == 1), None)
    rope_q = any(j in hid and N[j].func == "mul" and _rot_operands(j) for j in q_path)
    rope_k = any(j in hid and N[j].func == "mul" and _rot_operands(j) for j in k_path)
    norm_before_rope = None
    if qk_norm_q and rope_q:
        first_rot = min(j for j in q_path if j in hid and N[j].func == "mul" and _rot_operands(j))
        first_rs = min((j for j in q_path if N[j].func in ("rsqrt", "native_layer_norm")), default=None)
        norm_before_rope = first_rs is not None and first_rs < first_rot
    if qk_norm_q and rope_q and norm_before_rope is False:
        raise NotDrawn("QK-norm runs after the rotation: the drawing has no way to show that order yet")
    theta = None
    if rope_q:
        rot_muls = [j for j in q_path if j in hid and N[j].func == "mul" and _rot_operands(j)]
        if any(N[j].func == "view_as_complex" for j in q_path):
            raise NotDrawn("the rotation is a complex multiply (view_as_complex x polar table): that RoPE form is not "
                           "drawn yet")
        if len(rot_muls) > 2:
            # the DeepSeek interleave: cat([x1*cos - x2*sin, x2*cos + x1*sin]) -> 4 multiplies, one add, one sub, one cat
            subs = [j for j in q_path if j in hid and N[j].func == "sub" and set(R.srcs(j)) <= set(rot_muls)]
            adds_ = [j for j in q_path if j in hid and N[j].func == "add" and set(R.srcs(j)) <= set(rot_muls)]
            form = (" (the interleaved pairwise form, e.g. DeepSeek)" if len(rot_muls) == 4 and len(subs) == 1
                    and len(adds_) == 1 else "")
            raise NotDrawn(f"{len(rot_muls)} cos/sin multiplies on the query (rotate-half has 2){form}: that rotation "
                           "is not drawn yet")
        # coverage measured on the rotated hidden tensor (the table's own width can differ, e.g. halves)
        widths = {N[h].shape[-1] for j in rot_muls for h in R.srcs(j) if h in hid and N[h].shape}
        if widths and widths != {head_dim}:
            raise NotDrawn(f"rotary embedding covers {sorted(widths)} of {head_dim} head dimensions (partial rotary): "
                           "not drawn yet")
        bufs = bundle["model"]["computed_buffers"]
        def _theta_bufs(i):
            """The rotation buffers feeding layer i's query-path rotation."""
            muls = [j for j in L[i] if j in hid and N[j].func == "mul" and _rot_operands(j)]
            return sorted({e[1] for j in muls for x in _rot_operands(j)
                           for y in R.back(x, lambda z: False, 200)[1] + [x] for e in N[y].ins if e[0] == "buffer"})
        per_layer = {tuple(_theta_bufs(i)) for i in layer_ids}
        if len(per_layer) > 1:
            raise NotDrawn(f"layers rotate with different frequency tables ({sorted(per_layer)}): per-layer RoPE is "
                           "not drawn yet")
        names = [n_ for n_ in _theta_bufs(layer_ids[0]) if n_ in bufs and len(bufs[n_]["values"]) > 2]
        if len(names) == 1:
            vals = bufs[names[0]]["values"]
            # a plain RoPE table is geometric from 1: v0 = 1 and v2 = v1^2 (scaled tables fail this)
            if abs(vals[0] - 1) > 1e-6 or abs(vals[2] - vals[1] ** 2) > 1e-4 * abs(vals[2]):
                raise NotDrawn(f"the rotation frequencies in {names[0]} are not a plain geometric series from 1 "
                               "(scaled RoPE): not drawn yet")
            d = 2 * len(vals); raw = float(vals[1] ** (-d / 2))
            theta = float(f"{raw:.4g}")                 # the buffer is float32: 4 significant digits are real
            ev["rope"].append(f"frequency base from {names[0]} (the buffer feeding every layer's rotation)")
    cache = verify_cache(bundle.get("cache_run"), pshape, wk, wv, f"{stack}.{layer_ids[0]}")
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
    act_op = next((j for j in lops if N[j].func in ACTS and up is not None and down is not None and min(x for x in (gate, up) if x is not None) < j < down), None)
    if act == "gelu" and act_op is not None and "tanh" in (P["graph"].get("op_strings") or {}).get(str(act_op), []):
        act = "gelu_pytorch_tanh"
    wg = R.weights(gate)[0] if gate is not None else None
    wu, wd = R.weights(up)[0], R.weights(down)[0]
    inter = R.pshape[wu][0]
    ev["mlp"].append(f"{'gated: ' + wg + ' -> ' + str(act) + ' x ' if gated else ''}{wu} -> {wd}; inner width {inter}")
    # ---- norms and residuals: norm ops whose output feeds the projections; adds that combine branches
    pre_attn_ops = [j for j in lops if j < min(q_proj, k_proj, v_proj) and N[j].module != attn_mod and not N[j].module.startswith(attn_mod + ".")]
    pre_mlp_ops = [j for j in lops if (o_proj or apply_v) < j < min(x for x in (gate, up) if x is not None) and not N[j].module.startswith(attn_mod)]
    # residual adds: in the layer module itself, combining two hidden-path values (residual + branch output)
    adds = [j for j in lops if N[j].func == "add" and N[j].module == f"{stack}.{layer_ids[0]}"
            and len(N[j].ins) == 2 and all(e[0] == "op" and e[1] in hid for e in N[j].ins) and not N[j].consts]
    n1 = _norm_match(_region(q_proj, min(lops) - 1) & set(pre_attn_ops))[0]
    n2 = _norm_match(_region(min(x for x in (gate, up) if x is not None), adds[0] if adds else (o_proj or apply_v))
                     & set(pre_mlp_ops))[0]
    residual = "sequential" if len(adds) >= 2 and adds[0] in R.back(adds[1], lambda j: False, 400)[1] else ("parallel" if adds else None)
    placement = "pre" if n1 and n2 else ("post" if adds else None)
    ev["residual"].append(f"{len(adds)} residual adds in the layer module; placement {placement}; topology {residual}")

    # FFN bias: an addmm on up/down, or a 1-D weight added right after them
    ffn_bias = any(N[x].func == "addmm" for x in (gate, up, down) if x is not None)
    # ---- the op-accounting gate: every op on the hidden-state path is explained by a block this drawing shows;
    #      anything left over (a mixer branch, a softcap, a multiplier, a position embedding, ...) means NotDrawn
    q_hid, k_hid, v_hid = (set(x) & hid for x in (q_path, k_path, v_path))
    rope = rope_q and rope_k; qkn = bool(qk_norm_q and qk_norm_k)
    def _scalar_only(j):
        """A multiply / divide by a Python scalar and nothing else from off the hidden-state path."""
        return N[j].func in ("mul", "div") and bool(N[j].consts) and not _off_ops(j) and not any(e[0] == "param" for e in N[j].ins)
    def _proj_path_bad(ops):
        ops = set(ops)
        trig = [j for j in ops if N[j].func == "mul" and _rot_operands(j)]
        first = min(trig) if trig else None
        explained_ = set()
        if qkn:
            pre = {j for j in ops if first is None or j < first}
            kind_, acc = _norm_match(pre)
            if kind_ is None:
                return sorted(j for j in pre if N[j].func not in TRANSPARENT and not _scalar_only(j))
            explained_ |= acc
        if rope:
            m_ = _rope_match(ops - explained_)
            if m_ is None:
                return sorted(j for j in ops - explained_ if N[j].func not in TRANSPARENT and not _scalar_only(j))
            explained_ |= m_
        return [j for j in ops if j not in explained_ and N[j].func not in TRANSPARENT and not _scalar_only(j)]
    explained = {emb, head, q_proj, k_proj, v_proj, scores, sm, apply_v} | set(adds)
    explained |= {x for x in (o_proj, gate, up, down, act_op) if x is not None}
    if gated:
        explained |= {j for j in lops if N[j].func == "mul" and j > max(gate, up) and j < down
                      and not N[j].consts and not _off_ops(j) and not any(e[0] == "param" for e in N[j].ins)}
    # the two layer norms and the final norm: exactly a normalization between its input and the projection it feeds
    mlp_in = min(x for x in (gate, up) if x is not None)
    for tgt, lo, pool in ((q_proj, min(lops) - 1, pre_attn_ops), (mlp_in, adds[0] if adds else (o_proj or apply_v), pre_mlp_ops), (head, last, fin_ops)):
        explained |= _norm_accept(_region(tgt, lo) & set(pool))
    # before softmax: scalar scales and the one mask add only
    explained |= {j for j in pre_sm if j in hid and (N[j].func in TRANSPARENT or _scalar_only(j))}
    explained |= {j for j, _ in madds}
    bad = []
    bad += [("query path", j) for j in _proj_path_bad(q_hid)] + [("key path", j) for j in _proj_path_bad(k_hid)]
    bad += [("value path", j) for j in v_hid if N[j].func not in TRANSPARENT]
    explained |= q_hid | k_hid | v_hid
    first_layer_op, last_layer_op = min(L[layer_ids[0]]), last
    layer_mods = tuple(f"{stack}.{i}" for i in layer_ids)
    for j in sorted(hid):
        nd = N[j]
        if j in explained or nd.func in TRANSPARENT:
            continue
        if _scalar_only(j) and all(c in (1, 1.0) for c in nd.consts):
            continue                                   # a multiplier of exactly 1 changes nothing
        in_layer0 = j in lset
        in_other_layer = (not in_layer0) and any(nd.module == lm or nd.module.startswith(lm + ".") for lm in layer_mods)
        if in_other_layer:
            continue                                   # other layers run layer 0's op sequence (checked above)
        where_ = "layer 0" if in_layer0 else ("before the layers" if j < first_layer_op else
                                              ("after the layers" if j > last_layer_op else "between layers"))
        if j > head and nd.func in ("_to_copy", "float"):
            continue
        bad.append((where_, j))
    if bad:
        def _ordinary(j):                      # a norm's own op (x^2, mean, eps add, rsqrt, scaling, weight)
            nd = N[j]; c = list(nd.consts)
            if nd.func in ("mean", "rsqrt", "sqrt", "var", "sub"): return True
            if nd.func == "pow": return c in ([2], [2.0])
            if nd.func == "add": return (len(c) == 1 and abs(c[0]) <= 1e-2) or (not c and any(e[0] == "param" for e in nd.ins))
            if nd.func == "mul": return not c
            return False
        bad.sort(key=lambda wj: (_ordinary(wj[1]), wj[1]))      # the unusual ops first (e.g. a x3 after a norm)
        ex = ", ".join(f"{w} op {j} {N[j].func} in {N[j].module or '(root)'}" for w, j in bad[:5])
        raise NotDrawn(f"{len(bad)} recorded ops on the hidden-state path belong to no block this drawing shows "
                       f"(e.g. {ex}): drawing it would omit or misstate them")
    if not (n1 and n2) or placement != "pre" or residual != "sequential" or len(adds) != 2:
        d1 = _norm_desc(_region(q_proj, min(lops) - 1) & set(pre_attn_ops))
        d2 = _norm_desc(_region(mlp_in, adds[0] if adds else (o_proj or apply_v)) & set(pre_mlp_ops))
        raise NotDrawn(f"the layer is not pre-norm sequential (observed: attention-input norm {d1}, MLP-input norm "
                       f"{d2}, {len(adds)} residual add(s) combining two hidden values): only pre-norm sequential "
                       "layers are drawn yet")
    if final_norm is None:
        raise NotDrawn("no final norm observed before the output head: that model ending is not drawn yet")
    win = cfg.get("sliding_window")
    lt = cfg.get("layer_types")
    marks_sliding = (not lt) or any("sliding" in str(t) for t in lt)
    if isinstance(win, int) and win > 0 and cfg.get("use_sliding_window") is not False and marks_sliding:
        mp = bundle.get("max_position_embeddings") or cfg.get("max_position_embeddings")
        if not mp or win < mp:
            raise NotDrawn(f"the config declares a {win}-token sliding window, longer than this run's sequence, so "
                           "the window is invisible in the recording: sliding attention is not drawn yet")
    facts = dict(repo=repo, arch=arch, vocab=vocab, hidden=hidden, tied=tied, final_norm=final_norm, layers=len(layer_ids),
                 identical=len(identical), layer_ops=len(lops), kind=kind, num_heads=num_heads, num_kv=num_kv, head_dim=head_dim,
                 q_out=q_out, k_out=k_out, o_shape=R.pshape.get(wo), fused=fused, qk_norm=bool(qk_norm_q and qk_norm_k),
                 qk_norm_kind=qk_norm_q, qn_extent=(R.pshape[qn_w][0] if qn_w else None), norm_before_rope=norm_before_rope,
                 rope=rope_q and rope_k, theta=theta, scale=scale, mask=mask, bias=bias, ffn_bias=ffn_bias, o_proj=o_proj is not None,
                 gated=gated, act=act, inter=inter, norm=n1 or n2, placement=placement, residual=residual,
                 max_pos=(bundle.get("max_position_embeddings") if bundle.get("max_position_embeddings") is not None else cfg.get("max_position_embeddings")), total_ops=len(N), evidence=dict(ev), cache=cache,
                 seq=P["inputs"]["input_ids"][-1] if isinstance(P["inputs"].get("input_ids"), list) else None,
                 graded={k: bundle.get(k) for k in ("verdict", "failed_checks", "reason", "checks", "decisions", "leftovers")},
                 pass_label=label,
                 rope_note=(rope_note + ("; the drawn base comes from the unscaled high-frequency entries of the "
                                         "rotation buffer" if theta is not None else "; the base is not measurable "
                                         "from this run's rotation buffer")) if rope_note else None)
    return facts


def fmt(n):
    return f"{n:,}"


def to_ir(F):
    """Build unfold's ModelIR from execution facts (same fields the code parser fills)."""
    import model_unfolder.ir as I
    G = F.get("graded") or {}
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
    ffn = make(I.FFNSpec, kind="dense", activation=F["act"], intermediate_size=F["inter"], gated=F["gated"], bias=F.get("ffn_bias", False),
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
              notes=["Built from an execution recording: every fact is observed in the run harness v2 graded "
                     f"(verdict {G.get('verdict')}, pass '{F.get('pass_label')}'; no modeling-code parsing). "
                     "Context length is the config's declared value (not observable in one run)."]
                    + ([f"Benchmark verdict FULL_LEFTOVERS: only tensors the library itself declares unused were not "
                        f"executed ({', '.join(map(str, (G.get('leftovers') or {}).get('library_declared_unbuilt') or [])) or 'see the result'})."]
                       if G.get("verdict") == "FULL_LEFTOVERS" else [])
                    + ([F["rope_note"]] if F.get("rope_note") else []),
              warnings=([] if G.get("verdict") in ("FULL", "FULL_LEFTOVERS") else
                        [f"Unresolved evidence — benchmark verdict {G.get('verdict')}: checks failed "
                         f"{', '.join(G.get('failed_checks') or []) or 'n/a'}"
                         + (f" ({G.get('reason')})" if G.get('reason') else "")
                         + ". Drawn from what ran; parts that did not run are not shown."]))
    return ir


if __name__ == "__main__":
    repo, out = sys.argv[1], sys.argv[2]
    # default: the frozen renderer copy (model-benchmark/renderer_snapshot, see its SNAPSHOT.txt) so renders do not
    # move while unfold-pkg is being changed; --unfold-pkg PATH renders against any other tree
    _snap = os.path.join(HERE, "..", "renderer_snapshot")
    pkg = (sys.argv[sys.argv.index("--unfold-pkg") + 1] if "--unfold-pkg" in sys.argv
           else _snap if os.path.isdir(os.path.join(_snap, "model_unfolder")) else os.path.join(HERE, "..", "..", "unfold-pkg"))
    sys.path.insert(0, os.path.abspath(pkg))
    if "--bundle" in sys.argv:
        B = capture.load(sys.argv[sys.argv.index("--bundle") + 1])
    else:
        B = bundle_for(repo, sys.argv[sys.argv.index("--captures") + 1] if "--captures" in sys.argv else CAPTURES,
                       refresh="--refresh" in sys.argv)
    try:
        F = recognize(B)
    except NotDrawn as e:
        side = os.path.splitext(out)[0] + ".not_drawn.json"
        json.dump({"repo": B.get("repo"), "not_drawn": str(e), "verdict": B.get("verdict"),
                   "failed_checks": B.get("failed_checks"), "class": B.get("class")}, open(side, "w"), indent=1)
        print("NOT DRAWN:", e); print("written", side); sys.exit(2)
    print(json.dumps({k: v for k, v in F.items() if k != "evidence"}, default=str, indent=1))
    from model_unfolder.diagram import Diagram
    try:
        ir = to_ir(F)
    except Exception as e:
        side = os.path.splitext(out)[0] + ".not_drawn.json"
        json.dump({"repo": B.get("repo"), "not_drawn": f"the IR could not be built from the facts ({type(e).__name__}: {str(e)[:160]})",
                   "verdict": B.get("verdict")}, open(side, "w"), indent=1)
        print("NOT DRAWN: IR build failed:", e); print("written", side); sys.exit(2)
    path = Diagram(ir).save(out)
    print("saved", path)
