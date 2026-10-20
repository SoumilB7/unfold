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
TRANSPARENT = VIEWS | {"dropout", "native_dropout", "type_as", "copy", "copy_", "lift_fresh_copy", "cat_identity", "slice_empty"}
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


IDX_OPS = {"view", "_unsafe_view", "reshape", "transpose", "permute", "split", "split_with_sizes", "chunk", "slice", "select", "unsqueeze",
           "squeeze", "expand", "contiguous", "clone", "alias", "detach", "t", "unbind"}       # move values, compute none


def _apply_idx(func, x, c):
    """One recorded shape-only / slicing op applied to an index tensor (its recorded constants as arguments)."""
    if func in ("view", "_unsafe_view", "reshape"): return x.reshape([int(v) for v in c])
    if func == "transpose": return x.transpose(int(c[0]), int(c[1]))
    if func == "permute": return x.permute([int(v) for v in c])
    if func == "split": return torch.split(x, c[0], dim=int(c[1]) if len(c) > 1 else 0)
    if func == "split_with_sizes":                       # [size, size, ..., dim]
        return torch.split(x, [int(v) for v in c[:-1]], dim=int(c[-1]))
    if func == "chunk": return torch.chunk(x, int(c[0]), dim=int(c[1]) if len(c) > 1 else 0)
    if func == "unbind": return torch.unbind(x, int(c[0]) if c else 0)
    if func == "slice":
        dim, st, en = (list(c) + [0, 0, None])[:3]; step = c[3] if len(c) > 3 else 1
        ix = [slice(None)] * x.dim(); ix[int(dim)] = slice(st, en, step)
        return x[tuple(ix)]
    if func == "select": return x.select(int(c[0]), int(c[1]))
    if func == "unsqueeze": return x.unsqueeze(int(c[0]))
    if func == "squeeze": return x.squeeze(int(c[0])) if c else x.squeeze()
    if func == "expand": return x.expand([int(v) for v in c])
    if func == "t": return x.t()
    if func in ("contiguous", "clone", "alias", "detach"): return x
    raise ValueError(func)


def _port(e):
    return e[2] if len(e) > 2 and e[2] is not None else 0


def fused_qkv_split(N, mm, role_paths, head_dim, B, S):
    """The Q / K / V split of ONE fused projection, read from the recording: the shape-only / slicing ops after the
    matmul are replayed on an index tensor (value = token * F + output channel), so every query / key / value head
    element is traced to the exact output channel (and token) it came from. role_paths: {"q"|"k"|"v": ops between the
    role's consumer and the matmul}. Returns the channel map per role, the layout it forms, and the replayed ops; raises
    NotDrawn if the split is not exact or not one of the drawn layouts."""
    shp = N[mm].shape
    if shp is None: raise NotDrawn("the fused projection's output shape was not recorded")
    Fw, T = shp[-1], 1
    for d in shp[:-1]: T *= d
    if T != B * S: raise NotDrawn(f"the fused projection runs on {T} rows, not the {B}x{S} tokens")
    vals = {(mm, 0): (torch.arange(T)[:, None] * Fw + torch.arange(Fw)[None, :]).reshape(*shp)}
    last = max(max(p) for p in role_paths.values() if p)
    for j in range(mm + 1, last + 1):
        nd = N[j]
        if nd.func not in IDX_OPS or len(nd.ins) != 1 or nd.ins[0][0] != "op": continue
        key = (nd.ins[0][1], _port(nd.ins[0]))
        if key not in vals: continue
        try:
            out = _apply_idx(nd.func, vals[key], list(nd.consts))
        except Exception as e:
            raise NotDrawn(f"the fused projection's split op {j} {nd.func} {list(nd.consts)} could not be replayed "
                           f"({type(e).__name__})")
        outs = list(out) if isinstance(out, (list, tuple)) else [out]
        if nd.shape is not None and tuple(outs[0].shape) != tuple(nd.shape):
            raise NotDrawn(f"replaying the fused projection's split op {j} {nd.func} gives shape "
                           f"{list(outs[0].shape)}, the recording {list(nd.shape)}")
        for p, o in enumerate(outs): vals[(j, p)] = o
    tok_want = (torch.arange(B)[:, None, None, None] * S + torch.arange(S)[None, None, :, None])
    maps, keys, ops = {}, {}, set()
    for r in ("q", "k", "v"):
        multi = ("split", "split_with_sizes", "chunk", "unbind")       # a multi-output op: only the port a role reads
        cand = sorted({k for j in role_paths[r] for k in (([(j, 0)] if N[j].func not in multi else []) +
                       [(e[1], _port(e)) for e in N[j].ins if e[0] == "op"])
                       if k in vals and vals[k].dim() == 4 and vals[k].shape[0] == B and vals[k].shape[2] == S
                       and vals[k].shape[3] == head_dim})
        if not cand:
            raise NotDrawn(f"no [batch, heads, tokens, {head_dim}] {r.upper()} tensor is split from the fused projection "
                           "by shape-only ops")
        k = cand[0]; V = vals[k]
        if not bool((V // Fw == tok_want).all()):
            raise NotDrawn(f"the fused projection's {r.upper()} split moves values across tokens")
        ch = V % Fw
        if not bool((ch == ch[:1, :, :1, :]).all()):
            raise NotDrawn(f"the fused projection's {r.upper()} channels differ between tokens")
        maps[r] = ch[0, :, 0, :]
        keys[r] = k
        # the replayed ops from the matmul to this role tensor
        fr = [k[0]]
        while fr:
            x = fr.pop()
            if x == mm or x in ops: continue
            ops.add(x)
            fr += [e[1] for e in N[x].ins if e[0] == "op"]
    allch = torch.cat([maps[r].reshape(-1) for r in "qkv"])
    if not torch.equal(allch.sort().values, torch.arange(Fw)):
        raise NotDrawn(f"the fused projection's {Fw} output channels are not each used exactly once by Q, K and V")
    nh, nkv, hd = maps["q"].shape[0], maps["k"].shape[0], head_dim
    if maps["v"].shape[0] != nkv or nh % nkv:
        raise NotDrawn(f"fused projection: {nh} query heads, {nkv} key heads, {maps['v'].shape[0]} value heads")
    ar = lambda n: torch.arange(n * hd).reshape(n, hd)
    G = nh // nkv
    if (torch.equal(maps["q"], ar(nh)) and torch.equal(maps["k"], nh * hd + ar(nkv))
            and torch.equal(maps["v"], (nh + nkv) * hd + ar(nkv))):
        layout = "qkv_blocks"
    else:
        grp = (G + 2) * hd
        gq = torch.stack([g * grp + i * hd + torch.arange(hd) for g in range(nkv) for i in range(G)])
        gk = torch.stack([g * grp + G * hd + torch.arange(hd) for g in range(nkv)])
        gv = torch.stack([g * grp + (G + 1) * hd + torch.arange(hd) for g in range(nkv)])
        if torch.equal(maps["q"], gq) and torch.equal(maps["k"], gk) and torch.equal(maps["v"], gv):
            layout = "grouped"
        else:
            raise NotDrawn("the fused projection's output channels split into Q/K/V in a layout that is neither "
                           "[Q | K | V] blocks nor per-K/V-group [q x G | k | v] blocks")
    return {"layout": layout, "widths": (nh * hd, nkv * hd, nkv * hd), "heads": (nh, nkv), "keys": keys,
            "ops": ops, "group": G, "out": Fw}


_DT = {k_: v_ for n_, v_ in (("bool", torch.bool), ("int64", torch.int64), ("int32", torch.int32), ("float32", torch.float32),
                              ("float64", torch.float64), ("long", torch.int64)) for k_ in (n_, "torch." + n_)}


def replay_mask_condition(N, cond, S_rec, S_new, B=1):
    """Re-run the recorded boolean mask construction feeding `cond` at sequence length S_new: the recorded aranges /
    expands of length S_rec are re-sized, every comparison constant (e.g. a window) is kept, the attention-mask input is
    all ones (no padding, the drawn regime). Returns the bool tensor, or raises NotDrawn on an op it cannot replay."""
    anc, fr = set(), [cond]
    while fr:
        x = fr.pop()
        if x in anc: continue
        anc.add(x); fr += [e[1] for e in N[x].ins if e[0] == "op"]
    vals = {}
    def arg(e):
        if e[0] == "op": return vals[(e[1], _port(e))]
        if e[0] == "input" and "attention_mask" in str(e[1]): return torch.ones(B, S_new, dtype=torch.int64)
        raise NotDrawn(f"the mask construction reads {e[0]} {str(e[1])[:40]}: it cannot be replayed")
    rs = lambda v: S_new if v == S_rec else v
    for j in sorted(anc):
        nd = N[j]; f = nd.func; c = list(nd.consts); a = [arg(e) for e in nd.ins]
        dt = _DT.get(str(nd.outs[0][1])) if nd.outs else None
        if f == "arange":
            v = torch.arange(int(c[0]), rs(int(c[1]))) if len(c) == 2 else torch.arange(rs(int(c[0])))
        elif f in ("add", "sub") and len(a) == 2 and not c:
            v = a[0] + a[1] if f == "add" else a[0] - a[1]
        elif f in ("add", "sub") and len(a) == 1 and len(c) == 1:
            v = a[0] + c[0] if f == "add" else a[0] - c[0]
        elif f == "unsqueeze": v = a[0].unsqueeze(int(c[0]))
        elif f in ("le", "lt", "gt", "ge", "eq", "ne"):
            v = getattr(torch, f)(a[0], a[1] if len(a) > 1 else c[0])
        elif f in ("bitwise_and", "logical_and"): v = a[0] & a[1]
        elif f in ("bitwise_or", "logical_or"): v = a[0] | a[1]
        elif f in ("bitwise_not", "logical_not"): v = ~a[0]
        elif f == "new_ones": v = torch.ones([rs(int(x)) for x in c] if c else (), dtype=dt or torch.bool)
        elif f == "index": v = a[0][tuple(a[1:])]
        elif f == "expand": v = a[0].expand([rs(int(x)) for x in c])
        elif f in ("_to_copy", "to"): v = a[0].to(dt) if dt is not None else a[0]
        elif f in ("alias", "clone", "contiguous", "detach"): v = a[0]
        elif f in ("view", "reshape", "_unsafe_view"): v = a[0].reshape([rs(int(x)) for x in c])
        else:
            raise NotDrawn(f"the mask construction op {f} {c[:4]} cannot be replayed")
        vals[(j, 0)] = v
    return vals[(cond, 0)]


def mask_kind_of(Mb):
    """('causal', None) or ('sliding', w) if the [S, S] bool mask is exactly that band; else None."""
    S = Mb.shape[-1]
    Mb = Mb.reshape(-1, S, S)
    if not bool((Mb == Mb[:1]).all()): return None
    Mb = Mb[0]
    i = torch.arange(S)[:, None]; j = torch.arange(S)[None, :]
    if torch.equal(Mb, j <= i): return ("causal", None)
    w = int(Mb[-1].sum())
    if 0 < w < S and torch.equal(Mb, (j <= i) & (j > i - w)): return ("sliding", w)
    return None


def fused_pair_split(N, mm, keys, B, S):
    """The two halves of ONE fused projection (e.g. gate | up) read by `keys` ({name: (op, port)}): the shape-only ops
    after the matmul are replayed on an index tensor (token * F + channel); each key must keep every token in place and
    read the same channels for every token. Returns ({name: channel LongTensor}, replayed ops)."""
    shp = N[mm].shape
    if shp is None: raise NotDrawn("the fused projection's output shape was not recorded")
    Fw, T = shp[-1], 1
    for d in shp[:-1]: T *= d
    if T != B * S: raise NotDrawn(f"the fused projection runs on {T} rows, not the {B}x{S} tokens")
    vals = {(mm, 0): (torch.arange(T)[:, None] * Fw + torch.arange(Fw)[None, :]).reshape(*shp)}
    last = max(k[0] for k in keys.values())
    for j in range(mm + 1, last + 1):
        nd = N[j]
        if nd.func not in IDX_OPS or len(nd.ins) != 1 or nd.ins[0][0] != "op": continue
        key = (nd.ins[0][1], _port(nd.ins[0]))
        if key not in vals: continue
        try:
            out = _apply_idx(nd.func, vals[key], list(nd.consts))
        except Exception as e:
            raise NotDrawn(f"the fused projection's split op {j} {nd.func} could not be replayed ({type(e).__name__})")
        outs = list(out) if isinstance(out, (list, tuple)) else [out]
        for p_, o in enumerate(outs): vals[(j, p_)] = o
    chans, ops = {}, set()
    for nm_, k in keys.items():
        if k not in vals: raise NotDrawn(f"the fused projection's {nm_} half is not reached by shape-only ops")
        V = vals[k].reshape(T, -1)
        if not bool((V // Fw == torch.arange(T)[:, None]).all()):
            raise NotDrawn(f"the fused projection's {nm_} half moves values across tokens")
        ch = V % Fw
        if not bool((ch == ch[:1]).all()): raise NotDrawn(f"the fused projection's {nm_} channels differ between tokens")
        chans[nm_] = ch[0]
        fr = [k[0]]
        while fr:
            x = fr.pop()
            if x == mm or x in ops: continue
            ops.add(x); fr += [e[1] for e in N[x].ins if e[0] == "op"]
    return chans, ops


def verify_cache(cache_run, pshape, wk, wv, layer_prefix, fused_head_dim=None):
    """Observed K/V cache behaviour: which op produced the tensors stored in the returned cache, and whether this
    layer's attention read its keys/values from them. Returns {} if no cache was returned. `cache_run` is the
    bundle's cache-on run of the same pass (its op graph + every stored 4-D tensor with its producing op)."""
    if not cache_run or not cache_run.get("cache"):
        return {}
    try:
        _cn, _cmap = capture.fold_groups(capture.nodes_of(cache_run["graph"]),       # F10: decomposed ops drawn as one call
                                         cache_run["graph"].get("f09_assembled"))     # (F09: assembled builds by class)
    except ValueError as e:
        raise NotDrawn(f"cache run: {e}")
    found = [dict(t, producer=_cmap.get(t["producer"])) if t.get("producer") is not None else t
             for t in cache_run["cache"]["slots"]]
    R = Rec(_cn, pshape); N = R.n
    is_w = lambda j: N[j].func in MATMULS and bool(R.weights(j))
    fused_keys = None
    if wk == wv and fused_head_dim:                 # one fused projection: K / V told apart by the recorded split
        try:
            lops_ = [j for j, nd in enumerate(N) if nd.module.startswith(layer_prefix)]
            sm_ = next(j for j in lops_ if N[j].func in SOFTMAX)
            sc_, _ = R.back(sm_, lambda j: N[j].func in ("bmm", "matmul") and not R.weights(j))
            av_ = next(j for j in lops_ if j > sm_ and N[j].func in ("bmm", "matmul") and not R.weights(j))
            vin_ = [x for x in R.srcs(av_) if x != sm_ and x not in R.cons[sm_]]
            paths_ = {"q": R.back(R.srcs(sc_)[0], is_w), "k": R.back(R.srcs(sc_)[1], is_w),
                      "v": R.back(vin_[-1] if vin_ else av_, is_w)}
            Bs, Ss = N[sm_].shape[0], N[sm_].shape[-2]
            fused_keys = fused_qkv_split(N, paths_["k"][0], {r: set(v_[1]) for r, v_ in paths_.items()},
                                         fused_head_dim, Bs, Ss)["keys"]
        except (NotDrawn, StopIteration, IndexError, TypeError):
            fused_keys = {}
    write = {}
    for t in found:
        p = t["producer"]
        if p is None or not N[p].module.startswith(layer_prefix): continue
        hit, path = R.back(p, is_w)
        if hit is None: continue
        w_ = R.weights(hit)[0]
        if fused_keys is not None and w_ == wk:
            cons_ = {(e[1], _port(e)) for x in path + [p] for e in N[x].ins if e[0] == "op"} | {(x, 0) for x in path + [p]}
            role = ("key" if fused_keys.get("k") in cons_ else ("value" if fused_keys.get("v") in cons_ else None))
        else:
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


def render_id():
    """Content hash of the drawing code (this file + the renderer copy's snapshot record): separate from grading_id."""
    import hashlib
    h = hashlib.sha256(open(os.path.abspath(__file__), "rb").read())
    snap = os.path.join(HERE, "..", "renderer_snapshot", "SNAPSHOT.txt")
    if os.path.exists(snap): h.update(open(snap, "rb").read())
    return h.hexdigest()[:12]


def recognize(bundle):
    """Facts from a capture bundle, or NotDrawn with the evidence when the recording is outside what this adapter
    draws truthfully (it never crashes and never draws a guess)."""
    try:
        F = _recognize(bundle)
        F["provenance"] = {"revision": bundle.get("revision"), "grading_id": bundle.get("grading_id"),
                           "render_id": render_id()}
        return F
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
    # F10: the drawing view folds every op recorded as its decomposition back into the one call the program made
    try:
        _nodes, _remap = capture.fold_groups(capture.nodes_of(P["graph"]), P["graph"].get("f09_assembled"))
    except ValueError as e:
        raise NotDrawn(str(e))
    P = dict(P, squares=capture.remap_squares(P["squares"], _remap),
             graph=dict(P["graph"], op_strings={str(_remap[int(j)]): v for j, v in (P["graph"].get("op_strings") or {}).items()
                                                if int(j) in _remap}))
    R = Rec(_nodes, pshape)
    N = R.n
    for j, nd in enumerate(N):                  # cat with an empty operand (e.g. full-width rotary's empty pass-through)
        if nd.func == "cat":
            shp = [N[e[1]].shape if e[0] == "op" else (e[1] if e[0] == "external" else None) for e in nd.ins]
            if len(shp) == 2 and any(x is not None and len(x) and x[-1] == 0 for x in shp):
                nd.func = "cat_identity"
    _empty = {j for j, nd in enumerate(N) if nd.func == "slice" and nd.shape is not None and len(nd.shape) and nd.shape[-1] == 0}
    if _empty:
        _ec = collections.defaultdict(list)
        for k, n2 in enumerate(N):
            for e in n2.ins:
                if e[0] == "op" and e[1] in _empty: _ec[e[1]].append(k)
        for j in _empty:
            if _ec[j] and all(N[c].func == "cat_identity" for c in _ec[j]):
                N[j].func = "slice_empty"                # moves nothing: an empty tensor joined by a cat that keeps the other
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
    def _own_name(w, module):
        """A tied tensor has several names; the recorder keeps one. Prefer the name inside the reading module."""
        grp = next((g for g in bundle["model"]["tie_groups"] if w in g), None)
        if grp:
            own = [n_ for n_ in grp if module and n_.startswith(module + ".")]
            if own: return own[0]
        return w
    emb_w = _own_name(R.weights(emb)[0], N[emb].module)
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

    def _moves_only(j):
        """A shape-only op that keeps every value in place. A slice / select / narrow that drops part of the tensor
        (x[:, :1], one token) is NOT shape-only: it changes which values flow on."""
        nd = N[j]
        if nd.func not in TRANSPARENT: return False
        if nd.func in ("slice", "select", "narrow"):
            src = R.srcs(j)
            return bool(src) and nd.shape is not None and N[src[0]].shape is not None and tuple(nd.shape) == tuple(N[src[0]].shape)
        return True
    # identical layers: the same ops, constants, string arguments, weights AND wiring. Each operand is named by its
    # producer: an op of the same layer (by position), the one hidden state entering the layer, or the very same
    # shared off-path tensor (e.g. one mask, one rotation table); a layer reading a different value is not identical
    strs_ = P["graph"].get("op_strings") or {}
    def normseq(i, loose=False):
        nm = lambda x: re.sub(rf"^{re.escape(stack)}\.{i}(?=\.|$)", "L", x)
        local = {j: k for k, j in enumerate(L[i])}
        def edge(e):
            if e[0] != "op": return (e[0], nm(e[1]) if isinstance(e[1], str) else e[1])
            if e[1] in local: return ("local", local[e[1]])
            return ("hidden_in",) if e[1] in hid else (("shared",) if loose else ("shared", e[1]))
        return [(nm(N[j].module), N[j].func, tuple(N[j].consts), tuple(edge(e) for e in N[j].ins), N[j].shape,
                 tuple(tuple(R.pshape.get(e[1], ())) for e in N[j].ins if e[0] == "param"),
                 tuple(strs_.get(str(j), []))) for j in L[i]]
    ref = normseq(layer_ids[0], loose=True)
    identical = [i for i in layer_ids if normseq(i, loose=True) == ref]
    # layers run the same ops, constants, weights and wiring; they may read DIFFERENT shared off-path tensors (a
    # sliding vs a full mask, a local vs a global rotation table): one kind per distinct set (checked below)
    _strict = {i: repr(normseq(i)) for i in layer_ids}
    kind_reps = list(dict.fromkeys(_strict[i] for i in layer_ids))
    layer_kind = [kind_reps.index(_strict[i]) for i in layer_ids]
    kind_rep_layer = [layer_ids[layer_kind.index(k)] for k in range(len(kind_reps))]
    if len(identical) < len(layer_ids):
        odd = [i for i in layer_ids if i not in identical]
        raise NotDrawn(f"layers differ: {len(identical)} of {len(layer_ids)} run layer 0's ops, constants, weights and "
                       f"wiring (others: {odd[:6]}{'...' if len(odd) > 6 else ''}); per-layer drawing is not implemented yet")
    # the layers form one chain: each reads exactly one hidden state, the one the previous layer hands on
    def _layer_io(i):
        s_ = set(L[i])
        ins_ = {e[1] for j in L[i] for e in N[j].ins if e[0] == "op" and e[1] in hid and e[1] not in s_}
        outs = {j for j in L[i] if j in hid and any(c not in s_ for c in R.cons[j])}
        return ins_, outs
    layer_in, prev_out = None, None
    for i in layer_ids:
        ins_, outs = _layer_io(i)
        if len(ins_) != 1 or len(outs) != 1 or (prev_out is not None and ins_ != prev_out):
            raise NotDrawn(f"layer {i} is not chained to the previous one through a single hidden state (hidden inputs "
                           f"{sorted(ins_)[:4]}, outputs {sorted(outs)[:4]}): only a plain layer stack is drawn yet")
        if layer_in is None: layer_in = next(iter(ins_))
        prev_out = outs
    stack_out = next(iter(prev_out))
    def _thru0(x):
        """Look through shape-only ops (no slicing) to the value an operand stands for."""
        seen_ = 0
        while N[x].func in ("view", "_unsafe_view", "reshape", "contiguous", "clone", "alias", "detach", "_to_copy",
                            "to", "type_as") and R.srcs(x) and seen_ < 20:
            x = R.srcs(x)[0]; seen_ += 1
        return x
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
        core = [j for j in region if not _moves_only(j) and j not in unit]
        if not core: return None, set()
        # a normalization reads exactly one hidden value: its statistic and its scaling see the same input
        ext = {_thru(x) for j in region for x in R.srcs(j) if x in hid and x not in region}
        if len(ext) != 1: return None, set()
        xin = next(iter(ext))
        def _last_axis(j, dims):                         # reduces / normalizes over the last axis only
            rank = len(N[R.srcs(j)[0]].shape or ()) if R.srcs(j) else 0
            return len(dims) == 1 and isinstance(dims[0], int) and dims[0] in (-1, rank - 1)
        if len(core) == 1 and N[core[0]].func in ("native_layer_norm", "layer_norm"):
            c_ = list(N[core[0]].consts)                  # [normalized_shape..., eps]: one dim, the last
            src_ = R.srcs(core[0])
            if not src_ or len(c_) != 2 or N[src_[0]].shape is None or c_[0] != N[src_[0]].shape[-1]:
                return None, set()
            return "layernorm", region
        stat = [j for j in core if N[j].func in ("rsqrt", "sqrt")]
        if len(stat) != 1: return None, set()
        r = stat[0]
        # the statistic is exactly [rsqrt|sqrt]( mean_last( (x | x - mean_last(x)) ^ 2 ) [+ eps] )
        def _one(j):
            s_ = [_thru(x) for x in R.srcs(j)]
            return s_[0] if len(s_) == 1 else None
        y = _one(r)
        if y is not None and N[y].func == "add" and N[y].consts: y = _one(y)
        if y is None or N[y].func != "mean" or not _last_axis(y, list(N[y].consts)): return None, set()
        y = _one(y)
        if y is None or N[y].func != "pow" or list(N[y].consts) not in ([2], [2.0]): return None, set()
        y = _one(y)
        def _centered_of(s):                              # s = x - mean_last(x) of the norm's own input
            if s is None or N[s].func != "sub" or N[s].consts: return False
            a_ = [_thru(x) for x in R.srcs(s)]
            return (len(a_) == 2 and a_[0] == xin and N[a_[1]].func == "mean" and _last_axis(a_[1], list(N[a_[1]].consts))
                    and _one(a_[1]) == xin)
        if y != xin and not _centered_of(y): return None, set()
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
            if f in TRANSPARENT or f == "mean": continue           # var / var_mean: correction unchecked, not drawn
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
        # x * rsqrt(stat) or x / sqrt(stat); the other operand is the norm's input (or that input, centered)
        if (N[r].func, N[scl[0]].func) not in (("rsqrt", "mul"), ("sqrt", "div")): return None, set()
        if N[scl[0]].func == "div" and not from_r(R.srcs(scl[0])[-1]): return None, set()
        other = [_thru(x) for x in R.srcs(scl[0]) if not from_r(x)]
        if len(other) != 1 or not (other[0] == xin if not centered else _centered_of(other[0])): return None, set()
        rest = [j for j in after if j != scl[0]]
        wmul = [j for j in rest if N[j].func == "mul" and any(_param_side(e) for e in N[j].ins) and not N[j].consts]
        badd = [j for j in rest if N[j].func == "add" and any(_param_side(e) for e in N[j].ins) and not N[j].consts]
        if len(wmul) > 1 or len(badd) > 1 or len(wmul) + len(badd) != len(rest): return None, set()
        if badd and wmul and not (wmul[0] in anc(badd[0])): return None, set()      # weight before bias
        if any(scl[0] not in anc(j) for j in rest): return None, set()                # both after the scaling
        # the affine ops form one chain: scaling -> [x weight] -> [+ bias], each reading the previous value
        prev_ = scl[0]
        for j in wmul + badd:
            hs_ = [_thru(x) for x in R.srcs(j) if x in hid]
            if hs_ != [prev_]: return None, set()
            prev_ = j
        if badd and not centered: return None, set()          # an RMSNorm with a bias has no drawn form
        region = region | unit
        return ("layernorm" if centered else "rmsnorm"), region

    def _norm_accept(region):
        return _norm_match(region)[1]

    def _norm_info(region):
        """{kind, eps, weight} of the one normalization in `region` (or None): eps is the constant the recorded
        statistic adds (0.0 when none is added); weight is 'w' (read directly), '1+w' (the weight plus exactly 1,
        computed from the parameter alone) or 'none'. Any other weight transform raises NotDrawn."""
        kind_, acc = _norm_match(region)
        if not kind_: return None
        ln = [j for j in acc if N[j].func in ("native_layer_norm", "layer_norm")]
        if ln:
            ps = [e for e in N[ln[0]].ins if e[0] == "param"]
            return {"kind": kind_, "eps": float(list(N[ln[0]].consts)[1]), "weight": "w" if ps else "none"}
        stat = next(j for j in acc if N[j].func in ("rsqrt", "sqrt"))
        y = _thru(R.srcs(stat)[0]); eps = 0.0
        if N[y].func == "add" and N[y].consts: eps = float(N[y].consts[0])
        wmul = [j for j in acc if N[j].func == "mul" and not N[j].consts and any(_param_side(e) for e in N[j].ins)]
        if not wmul: return {"kind": kind_, "eps": eps, "weight": "none"}
        e = next(e for e in N[wmul[0]].ins if _param_side(e))
        if e[0] == "param": return {"kind": kind_, "eps": eps, "weight": "w"}
        anc = set(R.back(e[1], lambda z: False, 50)[1]) | {e[1]}
        core = [x for x in anc if N[x].func not in ("_to_copy", "to", "type_as", "alias", "clone", "detach")]
        if (len(core) == 1 and N[core[0]].func == "add" and list(N[core[0]].consts) in ([1], [1.0])
                and [e2[0] for e2 in N[core[0]].ins] == ["param"]):
            return {"kind": kind_, "eps": eps, "weight": "1+w"}
        raise NotDrawn(f"a {kind_} weight enters through {sorted(N[x].func for x in core)} {[list(N[x].consts) for x in core]}: "
                       "only the weight itself (w) or 1 + w is drawn")

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
        while _moves_only(x) and R.srcs(x) and seen < 20:
            x = R.srcs(x)[0]; seen += 1
        return x

    SHAPE_ONLY = {"view", "_unsafe_view", "unsqueeze", "expand", "reshape", "contiguous", "clone", "alias", "detach",
                  "_to_copy", "to", "type_as", "lift_fresh_copy", "copy", "transpose"}       # no slicing / reordering
    def _walk(x):
        seen = 0
        while N[x].func in SHAPE_ONLY and R.srcs(x) and seen < 30:
            x = R.srcs(x)[0]; seen += 1
        return x
    def _last_dim(j, d):
        return isinstance(d, int) and N[j].shape is not None and d in (-1, len(N[j].shape) - 1)

    def _rope_table(x):
        """(role, buffer, angle op, width) if x is cos or sin of the rotate-half angle table cat(f, f) on the last axis,
        f = positions 0..S-1 times the frequency buffer, scaled by exactly 1; None for anything else."""
        x = _walk(x); hops = 0
        while (N[x].func in ("mul", "div") and N[x].consts and all(c in (1, 1.0) for c in N[x].consts)
               and len(N[x].ins) == 1 and len(R.srcs(x)) == 1 and hops < 4):
            x = _walk(R.srcs(x)[0]); hops += 1                    # the attention scaling, exactly 1 (default, llama3)
        if N[x].func not in ("cos", "sin") or len(N[x].ins) != 1 or len(R.srcs(x)) != 1: return None
        role, c = N[x].func, _walk(R.srcs(x)[0])
        if (N[c].func != "cat" or len(N[c].ins) != 2 or N[c].ins[0] != N[c].ins[1] or N[c].ins[0][0] != "op"
                or not _last_dim(c, N[c].consts[0] if N[c].consts else 0)):
            return None                                           # rotate-half layout: the half-table twice, side by side
        f = _walk(N[c].ins[0][1])
        if N[f].func not in ("bmm", "matmul", "mm", "mul") or len(N[f].ins) != 2 or any(e[0] != "op" for e in N[f].ins):
            return None
        buf = pos = None
        for e in N[f].ins:
            y = _walk(e[1])
            bs = [e2[1] for e2 in N[y].ins if e2[0] == "buffer"]
            if N[y].func in SHAPE_ONLY and len(N[y].ins) == 1 and bs:
                buf = bs[0]; continue
            while N[y].func == "add" and list(N[y].consts) in ([0], [0.0]) and len(R.srcs(y)) == 1:
                y = _walk(R.srcs(y)[0])                           # + 0 (no cached positions)
            S_ = N[emb].shape[-2] if N[emb].shape and len(N[emb].shape) >= 2 else None       # the run's token count
            if (N[y].func == "arange" and not N[y].ins and S_ and list(N[y].consts) in ([S_], [0, S_])
                    and N[f].shape is not None and S_ in tuple(N[f].shape)[-2:]):
                pos = y
        bvals = (bundle["model"].get("computed_buffers") or {}).get(buf, {}).get("values") if buf else None
        if buf is None or pos is None or not bvals or N[c].shape[-1] != 2 * len(bvals): return None
        return role, buf, c, N[c].shape[-1]

    def _rope_match(ops):
        """The ops of exactly one rotate-half rotation in `ops`: x*cos + cat(-x[..., d/2:], x[..., :d/2])*sin, with
        cos / sin of one angle table (see _rope_table). None if the rotation ops are anything else (complex, interleaved,
        swapped tables, wrong half, wrong axis, subtraction, ...)."""
        ops = set(ops)
        trig = [j for j in ops if N[j].func == "mul" and _rot_operands(j)]
        if len(trig) != 2: return None
        adds_ = [j for j in ops if N[j].func == "add" and not N[j].consts and len(N[j].ins) == 2
                 and sorted(_thru(x) for x in R.srcs(j)) == sorted(trig)]
        if len(adds_) != 1: return None
        A = adds_[0]
        tab, hin = {}, {}
        for m_ in trig:
            hs = [x for x in R.srcs(m_) if x in hid]; offs = [x for x in R.srcs(m_) if x not in hid]
            if len(N[m_].ins) != 2 or N[m_].consts or len(hs) != 1 or len(offs) != 1: return None
            tab[m_] = _rope_table(offs[0]); hin[m_] = hs[0]
            if tab[m_] is None: return None
        if {t[0] for t in tab.values()} != {"cos", "sin"} or len({t[2] for t in tab.values()}) != 1: return None
        mc = next(m_ for m_ in trig if tab[m_][0] == "cos"); ms = next(m_ for m_ in trig if tab[m_][0] == "sin")
        x = _walk(hin[mc])                                       # x * cos
        rc = _walk(hin[ms])                                      # rotate_half(x) * sin
        d = tab[mc][3]; h = d // 2
        if N[rc].func != "cat" or len(N[rc].ins) != 2 or any(e[0] != "op" for e in N[rc].ins): return None
        if not _last_dim(rc, N[rc].consts[0] if N[rc].consts else 0) or N[rc].shape[-1] != d: return None
        neg, keep_half = (_walk(e[1]) for e in N[rc].ins)
        if N[neg].func != "neg" or len(R.srcs(neg)) != 1: return None
        def _half(sl, start, end_ok):
            nd = N[sl]
            if nd.func != "slice" or len(nd.consts) < 3 or len(R.srcs(sl)) != 1: return False
            dim, st, en = nd.consts[:3]; step = nd.consts[3] if len(nd.consts) > 3 else 1
            rank = len(N[R.srcs(sl)[0]].shape or ())
            return (dim in (-1, rank - 1) and st == start and step == 1 and end_ok(en) and _walk(R.srcs(sl)[0]) == x
                    and N[R.srcs(sl)[0]].shape == N[hin[mc]].shape)          # the same tensor x*cos reads
        second = _walk(R.srcs(neg)[0])
        if not (_half(second, h, lambda en: en >= d) and _half(keep_half, 0, lambda en: en == h)): return None
        keep = {A, rc, neg, second, keep_half, *trig}
        return keep | {j for j in ops if _moves_only(j)}

    first_layer_op = min(L[layer_ids[0]])
    embed_scale = None
    for j in sorted(hid):
        if j >= first_layer_op or N[j].func != "mul": continue
        srcs_ = [e for e in N[j].ins if not (e[0] == "op" and _thru0(e[1]) == emb)]
        if len(N[j].ins) == 2 and len(srcs_) == 1 and srcs_[0][0] == "buffer":
            bv = ((bundle["model"].get("computed_buffers") or {}).get(srcs_[0][1]) or {}).get("values")
            if bv and len(bv) == 1: embed_scale = {"op": j, "value": float(bv[0]), "source": f"buffer {srcs_[0][1]}"}
        elif len(N[j].ins) == 1 and len(N[j].consts) == 1:
            embed_scale = {"op": j, "value": float(N[j].consts[0]), "source": "constant"}
        if embed_scale:
            ev["embedding_scale"].append(f"op {j} multiplies the token embedding by {embed_scale['value']:.8g} "
                                         f"({embed_scale['source']})")
            break
    pos_emb = None
    S_tok = N[emb].shape[-2] if N[emb].shape and len(N[emb].shape) >= 2 else None
    for j in sorted(hid):
        if j >= first_layer_op or N[j].func != "add" or N[j].consts or len(N[j].ins) != 2: continue
        srcs_ = [e[1] for e in N[j].ins if e[0] == "op"]
        if len(srcs_) != 2 or emb not in {_thru0(x) for x in srcs_}: continue
        other = [x for x in srcs_ if _thru0(x) != emb]
        if len(other) != 1 or other[0] in hid: continue
        lk = _thru0(other[0])
        if N[lk].func != "embedding" or len(R.weights(lk)) != 1: continue
        tbl = R.weights(lk)[0]
        idx = [e[1] for e in N[lk].ins if e[0] == "op"]
        if len(idx) != 1 or len(R.pshape.get(tbl, ())) != 2 or R.pshape[tbl][1] != hidden: continue
        y = idx[0]; hops = 0
        while hops < 12 and (N[y].func in ("view", "unsqueeze", "expand", "reshape", "_to_copy", "to", "alias", "clone")
                             or (N[y].func == "add" and list(N[y].consts) in ([0], [0.0]))) and len(R.srcs(y)) == 1:
            y = R.srcs(y)[0]; hops += 1
        if N[y].func == "arange" and not N[y].ins and S_tok and list(N[y].consts) in ([S_tok], [0, S_tok]):
            pos_emb = {"op": j, "lookup": lk, "table": tbl, "shape": list(R.pshape[tbl])}
            ev["position"].append(f"op {lk} embedding reads {tbl} {list(R.pshape[tbl])} at positions 0..{S_tok - 1}; "
                                  f"op {j} adds it to the token embedding")
            break
    last = max(L[layer_ids[-1]])
    head = next((j for j in range(last + 1, len(N)) if N[j].func in MATMULS and R.weights(j)), None)
    if head is None:
        raise NotDrawn("no output projection runs after the layers (an encoder / embedding model): only models "
                       "ending in an output head are drawn yet")
    head_w = R.weights(head)[0]
    head_w = _own_name(head_w, N[head].module)
    ties = bundle["model"]["tie_groups"]
    grp = next((g for g in ties if head_w in g and emb_w in g), None)
    shipped_groups = bundle["model"].get("tie_groups_shipped")
    shipped = (None if shipped_groups is None else
               next((sg for g_, sg in zip(ties, shipped_groups) if g_ == grp), []) if grp else [])
    if grp is None:
        tied = False
        ev["tied"].append(f"LM head op {head} reads {head_w}, a different tensor from {emb_w}: not tied")
    elif shipped is not None and len(shipped) == 1:
        tied = True
        ev["tied"].append(f"LM head op {head} reads the embedding's tensor ({emb_w}); the checkpoint ships only "
                          f"{shipped or 'neither name'}, so the head is the embedding")
    else:
        # the model built from the config ties them, but the checkpoint ships both: the library ties them only if
        # their values are equal, which a run without the weights cannot check
        tied = None
        both = bool(shipped) and len(shipped) >= 2
        ev["tied"].append(f"the built model ties {head_w} to {emb_w} (config); " +
                          ("the checkpoint ships both tensors, equality not checked" if both else
                           "the checkpoint's tensor list is not recorded, so whether it ships one or both is unknown"))
    tie_both_shipped = bool(grp) and bool(shipped) and len(shipped) >= 2
    fin_ops = [j for j in range(last + 1, head)]
    final_norm = _norm_match(_region(head, last) & set(fin_ops))[0]

    # ---- one layer in detail (all layers are checked identical above)
    lops = L[layer_ids[0]]; lset = set(lops)
    sm = next((j for j in lops if N[j].func in SOFTMAX), None)
    if sm is None:
        raise NotDrawn("layer 0 runs no softmax attention (a state-space / linear-attention mixer): not drawn yet")
    if not (N[sm].consts and N[sm].shape is not None and N[sm].consts[0] in (-1, len(N[sm].shape) - 1)):
        raise NotDrawn(f"the attention softmax normalizes over axis {list(N[sm].consts)[:1]}, not the key axis: not drawn")
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
    def _softcap_at(t_):
        """div(x, c) -> tanh -> mul(., c) around tanh op t_ (the same constant c): (c, {ops}) or None."""
        d_ = R.srcs(t_)
        if len(d_) != 1 or N[d_[0]].func != "div" or len(N[d_[0]].consts) != 1 or len(N[d_[0]].ins) != 1: return None
        m_ = [x for x in R.cons[t_] if N[x].func == "mul" and len(N[x].consts) == 1 and len(N[x].ins) == 1]
        if len(R.cons[t_]) != 1 or len(m_) != 1 or float(N[m_[0]].consts[0]) != float(N[d_[0]].consts[0]): return None
        return float(N[d_[0]].consts[0]), {d_[0], t_, m_[0]}
    attn_softcap = None; softcap_ops = set()
    for j in pre_sm:
        if j in hid and N[j].func == "tanh":
            sc_ = _softcap_at(j)
            if sc_ is None or attn_softcap is not None:
                raise NotDrawn("a tanh on the attention scores that is not exactly tanh(x / c) * c: not drawn")
            attn_softcap, softcap_ops = sc_
            ev["softcap"].append(f"scores soft-capped: tanh(x / {attn_softcap:g}) * {attn_softcap:g} (ops {sorted(softcap_ops)})")
    # score scale: a constant multiplying/dividing the scores or the query
    scale = None
    for j in sorted(set(pre_sm) | set(R.back(scores, lambda j: R.weights(j) != [])[1])):
        if j in softcap_ops: continue
        if j in hid and N[j].func in ("mul", "div") and N[j].consts and not [e for e in N[j].ins if e[0] == "op"][1:]:
            c = N[j].consts[0]; f = c if N[j].func == "mul" else 1 / c
            scale = f if scale is None else scale * f
    # mask: the score addend from off the hidden-state path, as captured; causal if its upper triangle is -inf-like
    sq_ids = capture.square_ids(P["squares"])
    def _mask_adds(pre):
        out = []
        for j in pre:
            if j not in hid or N[j].func != "add" or N[j].consts: continue   # add(..., alpha=c) scales the mask
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
    S_rec = N[emb].shape[-2] if N[emb].shape and len(N[emb].shape) >= 2 else None
    def _mask_of(i):
        """('causal', None) or ('sliding', w) for layer i: its mask operand's recorded boolean construction, replayed at
        a length beyond every comparison constant it uses; the captured values at the recorded length must agree."""
        smi = next((j for j in L[i] if N[j].func in SOFTMAX), None)
        ms = _mask_adds(R.back(smi, lambda j: N[j].func in ("bmm", "matmul") and not R.weights(j))[1]) if smi is not None else []
        if len(ms) != 1: return None
        j_, k_ = ms[0]
        e_ = N[j_].ins[k_]
        if e_[0] != "op" or N[e_[1]].func != "where" or not N[e_[1]].ins or N[e_[1]].ins[0][0] != "op":
            raise NotDrawn(f"layer {i}'s mask is not a where(condition, 0, min) of a recorded boolean construction")
        cond = N[e_[1]].ins[0][1]
        ancs, fr = set(), [cond]
        while fr:
            x = fr.pop()
            if x in ancs: continue
            ancs.add(x); fr += [e[1] for e in N[x].ins if e[0] == "op"]
        cmax = max([abs(float(c)) for x in ancs if N[x].func in ("add", "sub", "le", "lt", "gt", "ge", "eq", "ne")
                    for c in N[x].consts if isinstance(c, (int, float)) and not isinstance(c, bool)] + [0])
        S_new = int(max(64, 2 * cmax + 2))
        if S_new > 20000:
            raise NotDrawn(f"the mask construction compares against {cmax:g}: too long to replay")
        Bb = N[e_[1]].shape[0] if N[e_[1]].shape else 1
        at_rec = replay_mask_condition(N, cond, S_rec, S_rec, Bb)
        if at_rec.dim() > 2:
            sl_ = at_rec.reshape(-1, at_rec.shape[-2], at_rec.shape[-1])
            if not bool((sl_ == sl_[:1]).all()):
                raise NotDrawn(f"layer {i}'s mask construction differs across heads or batch rows (a per-head / per-row "
                               "mask): only one [S, S] mask shared by every head is drawn")
            at_rec = sl_[0]
        t = square_adds[j_][k_]
        allowed = at_rec.reshape(t.shape) if at_rec.numel() == t.numel() else at_rec.expand(t.shape)
        if not (bool((t[allowed].abs() < 1e-3).all()) and bool((t[~allowed] <= -1e4).all())):
            raise NotDrawn(f"layer {i}'s captured mask values disagree with its replayed construction")
        mk = mask_kind_of(replay_mask_condition(N, cond, S_rec, S_new, Bb))
        if mk is None:
            raise NotDrawn(f"layer {i}'s replayed mask (length {S_new}) is neither causal nor one causal sliding window")
        return mk
    kind_masks = [_mask_of(r_) for r_ in kind_rep_layer]
    softcap_order = None
    if attn_softcap is not None and madds:
        pre_set = set(pre_sm)
        def _anc(x0):
            a_, fr_ = set(), [x0]
            while fr_:
                x = fr_.pop()
                for e in N[x].ins:
                    if e[0] == "op" and e[1] in pre_set and e[1] not in a_: a_.add(e[1]); fr_.append(e[1])
            return a_
        cap_div = next(x for x in softcap_ops if N[x].func == "div"); cap_mul = next(x for x in softcap_ops if N[x].func == "mul")
        if all(cap_mul in _anc(j) for j, _ in madds): softcap_order = "before"
        elif all(j in _anc(cap_div) for j, _ in madds): softcap_order = "after"
        else:
            raise NotDrawn("the attention soft-cap and the mask add are not one after the other on the scores: not drawn")
        ev["softcap"].append(f"the soft-cap is applied {softcap_order} the mask is added (dataflow of ops "
                             f"{sorted(softcap_ops)} and mask add {[j for j, _ in madds]})")
    if None in kind_masks:
        raise NotDrawn("a layer kind has no single additive mask operand on its scores: the mask is unknown")
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
    if mask == "custom":                       # a window shorter than the recorded run: the replay decides
        if kind_masks[layer_kind[0]][0] != "sliding":
            raise NotDrawn("the attention mask added to the scores is not causal (bidirectional, sliding or custom): "
                           "only causal masks are drawn yet")
        mask = "causal"
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
    def _w_layout(j, w):
        """How matmul op j reads its weight w: 'out_in' (x @ W^T: through a transpose, or linear) or 'in_out' (x @ W:
        read directly as the right operand, e.g. Conv1D); None if neither is recorded."""
        nd = N[j]
        if nd.func == "linear": return "out_in"
        if nd.func not in ("mm", "addmm", "matmul") or not nd.ins: return None
        rhs = nd.ins[-1]
        if rhs[0] == "param": return "in_out" if rhs[1] == w and len(R.pshape.get(w, ())) == 2 else None
        if rhs[0] == "op":
            t_ = N[rhs[1]]
            if len(t_.ins) == 1 and t_.ins[0][0] == "param" and t_.ins[0][1] == w and len(R.pshape.get(w, ())) == 2:
                if t_.func == "t": return "out_in"
                if t_.func in ("transpose", "permute") and sorted(int(c) % 2 for c in list(t_.consts)[:2]) == [0, 1]:
                    return "out_in"
        return None
    lays = {_w_layout(j, w) for j, w in ((q_proj, wq), (k_proj, wk), (v_proj, wv))}
    if None in lays or len(lays) != 1:
        raise NotDrawn(f"the Q/K/V projections read their weights in no single recorded layout ({sorted(map(str, lays))})")
    w_layout = next(iter(lays))
    odim = (lambda w: R.pshape[w][0]) if w_layout == "out_in" else (lambda w: R.pshape[w][-1])
    fused = wq == wk == wv
    split = None
    if fused:
        if not (q_proj == k_proj == v_proj):
            raise NotDrawn(f"one weight ({wq}) is read by several Q/K/V matmuls: that split is not drawn")
        split = fused_qkv_split(N, q_proj, {"q": set(q_path), "k": set(k_path), "v": set(v_path)}, head_dim,
                                N[sm].shape[0], N[sm].shape[-2])
        if split["out"] != odim(wq):
            raise NotDrawn(f"the fused projection's output ({split['out']}) is not its weight's out width ({odim(wq)})")
        q_out, k_out = split["widths"][:2]
        ev["qkv_split"].append(f"fused projection {wq} split by the recorded shape-only ops into "
                               f"{split['layout']} (Q {q_out}, K {k_out}, V {split['widths'][2]})")
    else:
        q_out = odim(wq); k_out = odim(wk)
    num_kv = k_out // head_dim if head_dim else None
    if num_heads is None and head_dim: num_heads = q_out // head_dim
    kind = "mha" if num_kv == num_heads else ("mqa" if num_kv == 1 else "gqa")
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
    if bool(qk_norm_q) != bool(qk_norm_k) or qk_norm_q != qk_norm_k:
        raise NotDrawn(f"the query is normalized by {qk_norm_q} and the key by {qk_norm_k}: one QK-norm kind is drawn, so "
                       "this is not drawn")
    if qk_norm_q and rope_q and norm_before_rope is False:
        raise NotDrawn("QK-norm runs after the rotation: the drawing has no way to show that order yet")
    theta = None
    kind_thetas = [None] * len(kind_rep_layer)
    rot_dim = head_dim
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
        if widths and widths != {head_dim} and len(widths) == 1 and 0 < min(widths) < head_dim and min(widths) % 2 == 0:
            rot_dim = min(widths)                        # partial rotary: proven by _partial_match below
        elif widths and widths != {head_dim}:
            raise NotDrawn(f"rotary embedding covers {sorted(widths)} of {head_dim} head dimensions (partial rotary): "
                           "not drawn yet")
        bufs = bundle["model"]["computed_buffers"]
        def _theta_bufs(i):
            """The rotation buffers feeding layer i's query-path rotation."""
            muls = [j for j in L[i] if j in hid and N[j].func == "mul" and _rot_operands(j)]
            return sorted({e[1] for j in muls for x in _rot_operands(j)
                           for y in R.back(x, lambda z: False, 200)[1] + [x] for e in N[y].ins if e[0] == "buffer"})
        def _theta_of(i_layer):
            """(theta, buffer) of layer i's rotation: every frequency of its one recorded table must follow from the
            base (default series, or the declared llama3 rescale of it)."""
            # a frequency table needs entry 0 (must be 1) and entry 1 (gives the base): 2 entries suffice (head_dim 4)
            fed = [n_ for n_ in _theta_bufs(i_layer) if n_ in bufs]
            names = [n_ for n_ in fed if len(bufs[n_]["values"]) >= 2]
            if fed and not names:
                raise NotDrawn(f"the buffers feeding the rotation ({fed[:3]}) hold one value each: one entry cannot show the "
                               "frequency base, so the rotation is not drawn")
            if len(names) != 1:
                raise NotDrawn(f"the rotation reads {len(names)} recorded frequency tables ({names[:3]}): its "
                               "frequencies cannot be checked, so the rotation is not drawn")
            if names:
                vals = bufs[names[0]]["values"]; n_ = len(vals)
                # EVERY frequency: theta^(-i/n) (default), or the llama3 piecewise rescale of that series; the base comes
                # from entry 1 (a high frequency, unscaled by llama3), every other entry must follow from it
                raw = float(vals[1] ** (-n_)) if n_ > 1 and vals[1] > 0 else None
                if raw is None or abs(vals[0] - 1) > 1e-6:
                    raise NotDrawn(f"the rotation frequencies in {names[0]} do not start at 1: not drawn")
                l3 = next((v for v in rope_cfg if (v.get("rope_type") or v.get("type")) == "llama3"), None)
                if l3:
                    fac, lo_f, hi_f = float(l3["factor"]), float(l3["low_freq_factor"]), float(l3["high_freq_factor"])
                    old_ctx = float(l3["original_max_position_embeddings"])
                    lo_wl, hi_wl = old_ctx / lo_f, old_ctx / hi_f
                    def _l3(f_):                                 # transformers' _compute_llama3_parameters, per entry
                        wl = 2 * math.pi / f_
                        if wl < hi_wl: return f_
                        if wl > lo_wl: return f_ / fac
                        sm_ = (old_ctx / wl - lo_f) / (hi_f - lo_f)
                        return (1 - sm_) * f_ / fac + sm_ * f_
                def _want(base_):
                    w_ = [base_ ** (-i / n_) for i in range(n_)]
                    return [_l3(f_) for f_ in w_] if l3 else w_
                def _freq_off(base_):
                    w_ = _want(base_)
                    return [i for i in range(n_) if abs(vals[i] - w_[i]) > 1e-4 * abs(w_[i])], w_
                # candidate bases: the one entry 1 implies (unscaled in every real llama3 table), else the declared one;
                # a base is accepted only if EVERY entry follows from it, so the buffer itself proves it
                decl = next((v.get("rope_theta") for v in rope_cfg if v.get("rope_theta")), None) or cfg.get("rope_theta")
                off, want = _freq_off(raw)
                if off and decl and not _freq_off(float(decl))[0]:
                    raw = float(decl); off, want = [], _want(raw)
                if off:
                    raise NotDrawn(f"rotation frequency {off[0]} of {n_} in {names[0]} is {vals[off[0]]:.6g}, not the "
                                   f"{'llama3-scaled ' if l3 else ''}base-{raw:.4g} value {want[off[0]]:.6g} "
                                   f"({len(off)} entries differ): not drawn")
                return float(f"{raw:.4g}"), names[0]          # the buffer is float32: 4 significant digits are real
            return None, None
        theta, tbuf = _theta_of(layer_ids[0])
        kind_thetas = [_theta_of(r_)[0] for r_ in kind_rep_layer]
        if theta is not None:
            ev["rope"].append(f"frequency base from {tbuf} (the buffer feeding every layer's rotation)")
    cache = verify_cache(bundle.get("cache_run"), pshape, wk, wv, f"{stack}.{layer_ids[0]}",
                         fused_head_dim=head_dim if fused else None)
    ev["cache"].append(json.dumps(cache, default=str))
    o_proj = next((j for j in sorted(lset) if j > apply_v and N[j].module.startswith(attn_mod) and N[j].func in MATMULS and R.weights(j)), None)
    wo = R.weights(o_proj)[0] if o_proj is not None else None
    v_out = split["widths"][2] if fused else odim(wv)
    if v_out != k_out:
        raise NotDrawn(f"the value projection is {v_out} wide and the key projection {k_out}: different K/V widths "
                       "are not drawn yet")
    bias = any(N[x].func == "addmm" for x in (q_proj, k_proj, v_proj))          # Q/K/V projection bias
    o_bias = o_proj is not None and N[o_proj].func == "addmm"
    def _gelu_tanh_formula(up_, down_):
        """0.5*x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3))) with x = up's output, every constant exact, the
        result read by down. Returns {"out", "ops"} or None."""
        def c1(j, f, v):
            return N[j].func == f and len(N[j].consts) == 1 and abs(float(N[j].consts[0]) - v) <= 1e-12 * max(1, abs(v)) \
                and len(R.srcs(j)) == 1 and len(N[j].ins) == 1
        x = _thru(up_)
        hin_d = [_thru(y) for y in R.srcs(down_) if y in hid]
        if len(hin_d) != 1: return None
        out = hin_d[0]
        if N[out].func != "mul" or N[out].consts or len(R.srcs(out)) != 2: return None
        a, b = R.srcs(out)
        for m1, a2 in ((a, b), (b, a)):
            if not (c1(m1, "mul", 0.5) and _thru(R.srcs(m1)[0]) == x and c1(a2, "add", 1.0)): continue
            t = R.srcs(a2)[0]
            if N[t].func != "tanh" or len(R.srcs(t)) != 1: continue
            m3 = R.srcs(t)[0]
            if not c1(m3, "mul", math.sqrt(2 / math.pi)): continue
            a1 = R.srcs(m3)[0]
            if N[a1].func != "add" or N[a1].consts or len(R.srcs(a1)) != 2: continue
            for xx, m2 in (R.srcs(a1), R.srcs(a1)[::-1]):
                if _thru(xx) != x or not c1(m2, "mul", 0.044715): continue
                pw = R.srcs(m2)[0]
                if c1(pw, "pow", 3.0) and _thru(R.srcs(pw)[0]) == x:
                    return {"out": out, "ops": {out, m1, a2, t, m3, a1, m2, pw}}
        return None
    # ---- MLP: weight matmuls of the layer outside the attention module
    mlp_mm = [j for j in lops if N[j].func in MATMULS and R.weights(j) and not N[j].module.startswith(attn_mod)]
    gate = up = down = None; act = None; gated = False; gate_up = None
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
    if gated and gate_up is None:
        down = next(j for j in mlp_mm if j > max(gate, up))
    if not gated and len(mlp_mm) == 2:
        # one fused gate/up matmul: down reads act(half_a) x half_b, both halves split from the same matmul output
        u0, d0 = mlp_mm
        hd_ = [_thru(y) for y in R.srcs(d0) if y in hid]
        if len(hd_) == 1 and N[hd_[0]].func == "mul" and not N[hd_[0]].consts and len(N[hd_[0]].ins) == 2:
            prod_ = hd_[0]
            for ea, eb in (tuple(N[prod_].ins), tuple(N[prod_].ins)[::-1]):
                if ea[0] != "op" or eb[0] != "op": continue
                pa = _thru(ea[1])
                if N[pa].func not in ACTS or len(N[pa].ins) != 1 or N[pa].ins[0][0] != "op": continue
                if u0 not in set(R.back(pa, lambda z: False, 60)[1]) or u0 not in set(R.back(eb[1], lambda z: False, 60)[1]):
                    continue
                kg = (N[pa].ins[0][1], _port(N[pa].ins[0])); ku = (eb[1], _port(eb))
                Bt, St = N[emb].shape[0], N[emb].shape[-2]
                chans, gu_ops = fused_pair_split(N, u0, {"gate": kg, "up": ku}, Bt, St)
                I_ = chans["gate"].numel(); Fw_ = N[u0].shape[-1]
                if chans["up"].numel() != I_ or 2 * I_ != Fw_:
                    raise NotDrawn(f"the fused gate/up projection splits {Fw_} channels into {I_} + {chans['up'].numel()}")
                if torch.equal(chans["gate"], torch.arange(I_)) and torch.equal(chans["up"], I_ + torch.arange(I_)):
                    lay_ = "gate_up"
                elif torch.equal(chans["up"], torch.arange(I_)) and torch.equal(chans["gate"], I_ + torch.arange(I_)):
                    lay_ = "up_gate"
                else:
                    raise NotDrawn("the fused gate/up projection's halves are not two contiguous blocks")
                gate_up = {"layout": lay_, "inter": I_, "ops": gu_ops, "prod": prod_, "g_edge": kg, "u_edge": ku}
                gate = up = u0; down = d0; act = ACTS[N[pa].func]; gated = True
                ev["mlp_split"].append(f"fused gate/up projection split by the recorded ops into "
                                       f"{'[gate | up]' if lay_ == 'gate_up' else '[up | gate]'} of {I_} each")
                break
    if gate_up is None and not gated and len(mlp_mm) >= 2:
        up, down = mlp_mm[0], mlp_mm[-1]
        act = next((ACTS[N[j].func] for j in lops if up < j < down and N[j].func in ACTS), None)
    act_op = next((j for j in lops if N[j].func in ACTS and up is not None and down is not None and min(x for x in (gate, up) if x is not None) < j < down), None)
    if act == "gelu" and act_op is not None and "tanh" in (P["graph"].get("op_strings") or {}).get(str(act_op), []):
        act = "gelu_pytorch_tanh"
    act_form = None
    if not gated and up is not None and down is not None:
        act_form = _gelu_tanh_formula(up, down)
        if act_form:
            act, act_op = "gelu_pytorch_tanh", act_form["out"]
            ev["activation"].append(f"activation recorded as the tanh-GELU formula (ops {sorted(act_form['ops'])})")
    wg = R.weights(gate)[0] if gate is not None else None
    wu, wd = R.weights(up)[0], R.weights(down)[0]
    for j_, w_ in ((o_proj, wo), (gate, wg), (up, wu), (down, wd)):
        if j_ is not None and _w_layout(j_, w_) != w_layout:
            raise NotDrawn(f"{w_} is read as {_w_layout(j_, w_)}, the Q/K/V weights as {w_layout}: mixed weight "
                           "layouts are not drawn")
    inter = odim(wu) if gate_up is None else gate_up["inter"]
    ev["mlp"].append(f"{'gated: ' + wg + ' -> ' + str(act) + ' x ' if gated else ''}{wu} -> {wd}; inner width {inter}")
    # ---- norms and residuals: norm ops whose output feeds the projections; adds that combine branches
    pre_attn_ops = [j for j in lops if j < min(q_proj, k_proj, v_proj) and N[j].module != attn_mod and not N[j].module.startswith(attn_mod + ".")]
    pre_mlp_ops = [j for j in lops if (o_proj or apply_v) < j < min(x for x in (gate, up) if x is not None) and not N[j].module.startswith(attn_mod)]
    # residual adds: in the layer module itself, combining two hidden-path values (residual + branch output)
    adds = [j for j in lops if N[j].func == "add" and N[j].module == f"{stack}.{layer_ids[0]}"
            and len(N[j].ins) == 2 and all(e[0] == "op" and e[1] in hid for e in N[j].ins) and not N[j].consts]
    n1 = _norm_match(_region(q_proj, min(lops) - 1) & set(pre_attn_ops))[0]
    mlp_in0 = min(x for x in (gate, up) if x is not None)
    o_out0 = o_proj if o_proj is not None else apply_v
    # parallel: the MLP branch does not read the attention output (both branches read the layer input)
    parallel = bool(adds) and o_out0 not in set(R.back(mlp_in0, lambda j: False, 4000)[1])
    mlp_lo = (min(lops) - 1) if parallel else (adds[0] if adds else (o_proj or apply_v))
    n2 = _norm_match(_region(mlp_in0, mlp_lo) & set(pre_mlp_ops))[0]
    parallel_norms = None
    if parallel:
        shared = n2 is None and sorted({_thru(x) for x in R.srcs(mlp_in0) if x in hid}) == \
            sorted({_thru(x) for x in R.srcs(q_proj) if x in hid})
        parallel_norms = 1 if shared else (2 if n2 else None)
        if shared: n2 = n1
        residual = "parallel"
        placement = "pre" if n1 and parallel_norms else None
    a_out_reg = m_out_reg = None
    if not parallel:
        residual = "sequential" if len(adds) >= 2 and adds[0] in R.back(adds[1], lambda j: False, 400)[1] else ("parallel" if adds else None)
        def _out_norm(add_j, out_op):
            """The hidden-path ops between a sublayer's output op and the residual add it feeds, when they are one
            normalization (post / sandwich placement); else None."""
            br = [x for x in R.srcs(add_j) if x == out_op or out_op in set(R.back(x, lambda z: False, 400)[1])]
            if len(br) != 1 or _thru(br[0]) == out_op: return None
            reg = {x for x in R.back(br[0], lambda z: z == out_op, 400)[1] if x in hid and x > out_op}
            return reg if _norm_match(reg)[0] else None
        if len(adds) == 2 and residual == "sequential":
            a_out_reg, m_out_reg = _out_norm(adds[0], o_out0), _out_norm(adds[1], down)
        has_out = bool(a_out_reg) and bool(m_out_reg)
        placement = ("pre" if n1 and n2 and not (a_out_reg or m_out_reg) else
                     "double" if n1 and n2 and has_out else
                     "post" if not n1 and not n2 and has_out else None)
    ev["residual"].append(f"{len(adds)} residual adds in the layer module; placement {placement}; topology {residual}")

    # FFN bias: an addmm on up/down, or a 1-D weight added right after them
    ffn_bias = any(N[x].func == "addmm" for x in (gate, up, down) if x is not None)
    # ---- the op-accounting gate: every op on the hidden-state path is explained by a block this drawing shows;
    #      anything left over (a mixer branch, a softcap, a multiplier, a position embedding, ...) means NotDrawn
    q_hid, k_hid, v_hid = (set(x) & hid - (split["ops"] if fused else set()) for x in (q_path, k_path, v_path))
    rope = rope_q and rope_k; qkn = bool(qk_norm_q and qk_norm_k)
    def _scalar_only(j):
        """A multiply / divide by a Python scalar and nothing else from off the hidden-state path."""
        return N[j].func in ("mul", "div") and bool(N[j].consts) and not _off_ops(j) and not any(e[0] == "param" for e in N[j].ins)
    def _partial_match(ops, m_):
        """Partial rotary: the rotated tensor is x[..., :r] of one full-width head tensor T and the result is
        cat(rotated, T[..., r:]) on the last axis. Returns those three ops, or None."""
        A = [j for j in m_ if N[j].func == "add"]
        if len(A) != 1: return None
        A = A[0]
        rins = {_walk(h) for j in m_ if N[j].func == "mul" for h in R.srcs(j) if h in hid}
        S1 = [x for x in rins if N[x].func == "slice"]
        if len(S1) != 1: return None
        S1 = S1[0]; c1 = list(N[S1].consts)
        rank = len(N[R.srcs(S1)[0]].shape or ()) if R.srcs(S1) else 0
        if not (len(c1) >= 3 and c1[0] in (-1, rank - 1) and c1[1] == 0 and c1[2] == rot_dim and (c1[3:] or [1])[0] == 1):
            return None
        T = N[S1].ins[0]                                 # the edge (op, port): the full-width head tensor
        if T[0] != "op": return None
        S2 = [j for j in ops if N[j].func == "slice" and j != S1 and N[j].ins and N[j].ins[0] == T
              and list(N[j].consts)[:2] in ([c1[0], rot_dim],) and N[j].consts[2] >= head_dim]
        if len(S2) != 1: return None
        S2 = S2[0]
        C = [j for j in ops if N[j].func == "cat" and len(N[j].ins) == 2 and _last_dim(j, N[j].consts[0] if N[j].consts else 0)
             and [_thru(e[1]) if e[0] == "op" else None for e in N[j].ins] == [A, S2]
             and N[j].shape is not None and N[j].shape[-1] == head_dim]
        if len(C) != 1: return None
        return {S1, S2, C[0]}

    def _proj_path_bad(ops):
        ops = set(ops)
        trig = [j for j in ops if N[j].func == "mul" and _rot_operands(j)]
        first = min(trig) if trig else None
        explained_ = set()
        if qkn:
            pre = {j for j in ops if first is None or j < first}
            kind_, acc = _norm_match(pre)
            if kind_ is None:
                return sorted(j for j in ops if not _moves_only(j) and not _scalar_only(j))
            explained_ |= acc
        if rope:
            m_ = _rope_match(ops - explained_)
            if m_ is None:
                raise NotDrawn("the rotation on the query/key path is not exactly x*cos + rotate_half(x)*sin of one "
                               "position x frequency table (rotate-half halves, cos/sin roles, last axis, positions "
                               "0..S-1, scale 1): that rotation is not drawn")
            explained_ |= m_
            if rot_dim != head_dim:
                p_ = _partial_match(ops, m_)
                if p_ is None:
                    raise NotDrawn(f"the rotation covers {rot_dim} of {head_dim} head dimensions but is not exactly "
                                   f"cat(rotate(x[..., :{rot_dim}]), x[..., {rot_dim}:]) of one head tensor: not drawn")
                explained_ |= p_
        return [j for j in ops if j not in explained_ and not _moves_only(j) and not _scalar_only(j)]
    explained = {emb, head, q_proj, k_proj, v_proj, scores, sm, apply_v} | set(adds) | (split["ops"] if fused else set())
    if pos_emb: explained.add(pos_emb["op"])
    explained |= act_form["ops"] if act_form else set()
    explained |= {x for x in (o_proj, gate, up, down, act_op) if x is not None}
    if gated:
        explained |= {j for j in lops if N[j].func == "mul" and j > max(gate, up) and j < down
                      and not N[j].consts and not _off_ops(j) and not any(e[0] == "param" for e in N[j].ins)}
    # the two layer norms and the final norm: exactly a normalization between its input and the projection it feeds
    mlp_in = min(x for x in (gate, up) if x is not None)
    for tgt, lo, pool in ((q_proj, min(lops) - 1, pre_attn_ops), (mlp_in, mlp_lo, pre_mlp_ops), (head, last, fin_ops)):
        explained |= _norm_accept(_region(tgt, lo) & set(pool))
    for reg_ in (a_out_reg, m_out_reg):
        if reg_: explained |= _norm_accept(reg_)
    if embed_scale: explained.add(embed_scale["op"])
    # before softmax: scalar scales and the one mask add only
    explained |= {j for j in pre_sm if j in hid and (N[j].func in TRANSPARENT or _scalar_only(j))}
    explained |= {j for j, _ in madds}
    explained |= softcap_ops
    if gate_up: explained |= gate_up["ops"]
    final_softcap = None
    after_ = [j for j in sorted(hid) if j > head and N[j].func == "tanh"]
    if after_:
        sc_ = _softcap_at(after_[0])
        if sc_ is None or len(after_) > 1:
            raise NotDrawn("a tanh on the logits that is not exactly tanh(x / c) * c: not drawn")
        final_softcap = sc_[0]; explained |= sc_[1]
        ev["softcap"].append(f"logits soft-capped: tanh(x / {final_softcap:g}) * {final_softcap:g}")
    bad = []
    bad += [("query path", j) for j in _proj_path_bad(q_hid)] + [("key path", j) for j in _proj_path_bad(k_hid)]
    bad += [("value path", j) for j in v_hid if not _moves_only(j)]
    explained |= q_hid | k_hid | v_hid
    first_layer_op, last_layer_op = min(L[layer_ids[0]]), last
    layer_mods = tuple(f"{stack}.{i}" for i in layer_ids)
    for j in sorted(hid):
        nd = N[j]
        if j in explained or _moves_only(j):
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
    accounting = {"hidden": len(hid), "explained": len(hid) - len(bad), "unexplained": len(bad),
                  "layer0_ops": len([j for j in hid if j in lset]), "layers_covered_by_identity": len(layer_ids) - 1}
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
    def _sum_leaves(j):                              # the terms a chain of residual adds sums
        return (_sum_leaves(_thru(R.srcs(j)[0])) + _sum_leaves(_thru(R.srcs(j)[1]))) if j in adds else [_thru(j)]
    lay_out = next(iter(_layer_io(layer_ids[0])[1]))
    if parallel:
        if not n1 or parallel_norms not in (1, 2) or len(adds) != 2 or placement != "pre":
            raise NotDrawn(f"parallel attention + MLP branches, but the norms are not exactly one shared or two separate "
                           f"pre-branch norms ({n1}, {n2}, {len(adds)} adds): not drawn")
        if sorted(_sum_leaves(lay_out)) != sorted([_thru(layer_in), _thru(o_out0), _thru(down)]):
            raise NotDrawn("the parallel layer's output is not layer input + attention output + MLP output: not drawn")
    elif placement not in ("pre", "post", "double") or residual != "sequential" or len(adds) != 2:
        d1 = _norm_desc(_region(q_proj, min(lops) - 1) & set(pre_attn_ops))
        d2 = _norm_desc(_region(mlp_in, adds[0] if adds else (o_proj or apply_v)) & set(pre_mlp_ops))
        raise NotDrawn(f"the layer is not a drawn sequential placement (observed: attention-input norm {d1}, MLP-input "
                       f"norm {d2}, attention-output norm {'yes' if a_out_reg else 'no'}, MLP-output norm "
                       f"{'yes' if m_out_reg else 'no'}, {len(adds)} residual add(s)): pre, post (sublayer output) and "
                       "sandwich placements are drawn")
    regs_ = [r_ for r_ in ((_region(q_proj, min(lops) - 1) & set(pre_attn_ops)) if n1 else None,
                           (_region(mlp_in, mlp_lo) & set(pre_mlp_ops)) if (n2 and not (parallel and parallel_norms == 1)) else None,
                           a_out_reg, m_out_reg) if r_]
    infos_ = [_norm_info(r_) for r_ in regs_]
    if len({(i_["kind"], i_["eps"], i_["weight"]) for i_ in infos_}) != 1:
        raise NotDrawn(f"the layer's norms differ ({[(i_['kind'], i_['eps'], i_['weight']) for i_ in infos_]}): one norm "
                       "kind / eps / weight form is drawn per layer")
    norm_info = infos_[0]
    n1 = n2 = norm_info["kind"]
    if final_norm is None:
        raise NotDrawn("no final norm observed before the output head: that model ending is not drawn yet")
    final_info = _norm_info(_region(head, last) & set(fin_ops))
    qk_info = None
    if qkn:
        qi_ = [_norm_info({j for j in set(p_) & hid if (min([x for x in p_ if x in hid and N[x].func == "mul" and _rot_operands(x)] or [10 ** 9]) > j)})
               for p_ in (q_path, k_path)]
        if None in qi_ or qi_[0] != qi_[1]:
            raise NotDrawn(f"the query and key norms differ in eps / weight form ({qi_}): not drawn")
        qk_info = qi_[0]
    # ---- wiring: every arrow the drawing shows is a recorded dataflow edge of layer 0 (the other layers are wired
    #      identically, checked above), looking only through shape-only ops
    def _hin(j):
        return sorted({_thru(x) for x in R.srcs(j) if x in hid})
    def _feeds(region):
        return sorted({_thru(x) for j in region for x in R.srcs(j) if x in hid and x not in region})
    reg1 = _region(q_proj, min(lops) - 1) & set(pre_attn_ops)
    reg2 = _region(mlp_in, mlp_lo) & set(pre_mlp_ops)
    regf = _region(head, last) & set(fin_ops)
    o_out = o_proj if o_proj is not None else apply_v
    v_src = _hin(v_proj)
    a_res = _thru(next(x for x in R.srcs(adds[0]) if a_out_reg and x in a_out_reg)) if a_out_reg else _thru(o_out)
    m_res = _thru(next(x for x in R.srcs(adds[1]) if m_out_reg and x in m_out_reg)) if m_out_reg else _thru(down)
    arrows = [
        ("attention-input norm <- layer input", _feeds(reg1) == [_thru(layer_in)]) if placement != "post" else
        ("Q, K, V projections <- the layer input", v_src == [_thru(layer_in)]),
        ("Q, K, V projections <- one normed input", _hin(q_proj) == _hin(k_proj) == v_src and len(v_src) == 1
         and (v_src[0] in reg1 if placement != "post" else True)),
        ("weighted sum <- softmax, V", _hin(apply_v) == sorted({_thru(sm), _thru(split["keys"]["v"][0] if fused else v_proj)})),
        ("final norm <- last layer output", _feeds(regf) == [_thru(stack_out)]),
    ]
    if not parallel:
        arrows[3:3] = [
            ("first residual add <- layer input + attention output", _hin(adds[0]) == sorted({_thru(layer_in), a_res})),
            ("MLP-input norm <- first residual sum", _feeds(reg2) == [adds[0]]) if placement != "post" else
            ("MLP projections <- first residual sum", _hin(up) == [adds[0]]),
            ("MLP projections <- one normed input", len({tuple(_hin(x)) for x in (gate, up) if x is not None}) == 1
             and len(_hin(up)) == 1 and (_hin(up)[0] in reg2 if placement != "post" else True)),
            ("second residual add <- first sum + MLP output", _hin(adds[1]) == sorted({adds[0], m_res})),
        ]
        for nm_, reg_, src_ in (("attention", a_out_reg, o_out), ("MLP", m_out_reg, down)):
            if reg_: arrows.append((f"{nm_}-output norm <- {nm_} output", _feeds(reg_) == [_thru(src_)]))
    elif parallel_norms == 2:
        arrows += [("MLP-input norm <- layer input", _feeds(reg2) == [_thru(layer_in)]),
                   ("MLP projections <- one normed input", len({tuple(_hin(x)) for x in (gate, up) if x is not None}) == 1
                    and len(_hin(up)) == 1 and _hin(up)[0] in reg2),
                   ("layer output = layer input + attention output + MLP output",
                    sorted(_sum_leaves(lay_out)) == sorted([_thru(layer_in), _thru(o_out), _thru(down)]))]
    else:
        arrows += [("MLP projections <- the attention's normed input", len({tuple(_hin(x)) for x in (gate, up) if x is not None}) == 1
                    and _hin(up) == v_src),
                   ("layer output = layer input + attention output + MLP output",
                    sorted(_sum_leaves(lay_out)) == sorted([_thru(layer_in), _thru(o_out), _thru(down)]))]
    if o_proj is not None:
        arrows.append(("output projection <- weighted sum", _hin(o_proj) == [apply_v]))
    if gated:
        prod = _hin(down)
        if gate_up:
            arrows.append(("down projection <- act(gate half) x up half", prod == [gate_up["prod"]]
                           and _hin(prod[0]) == sorted({act_op, _thru(gate_up["u_edge"][0])})
                           and _hin(act_op) == [_thru(gate_up["g_edge"][0])]))
        else:
            arrows.append(("down projection <- act(gate) x up", len(prod) == 1 and N[prod[0]].func == "mul"
                           and _hin(prod[0]) == sorted({act_op, up}) and act_op is not None and _hin(act_op) == [gate]))
    else:
        arrows.append(("down projection <- act(up)", act_op is not None and _hin(down) == [act_op]
                       and (_hin(act_op) == [up] if not act_form else True)))
    scaled_mm = [x for x in (q_proj, k_proj, v_proj, o_proj, gate, up, down, head)
                 if x is not None and any(c not in (1, 1.0) for c in N[x].consts)]
    arrows.append(("projections are plain matmuls (no alpha / beta multipliers)", not scaled_mm))
    miswired = [a_ for a_, ok in arrows if not ok]
    if miswired:
        raise NotDrawn(f"the recorded dataflow does not match the drawn arrows ({'; '.join(miswired)}): not drawn")
    ev["layers"].append(f"wiring: {len(arrows)} drawn arrows match recorded dataflow edges; layers chained through one "
                        "hidden state each")
    win = cfg.get("sliding_window")
    lt = cfg.get("layer_types")
    marks_sliding = (not lt) or any("sliding" in str(t) for t in lt)
    if isinstance(win, int) and win > 0 and cfg.get("use_sliding_window") is not False and marks_sliding \
            and not any(m_[0] == "sliding" for m_ in kind_masks):
        mp = bundle.get("max_position_embeddings") or cfg.get("max_position_embeddings")
        if not mp or win < mp:
            raise NotDrawn(f"the config declares a {win}-token sliding window but no layer's recorded mask construction "
                           "contains one: the window cannot be drawn from what ran")
    # layer kinds differ ONLY in the shared tensors their mask / rotation read (checked op by op)
    seq0 = normseq(kind_rep_layer[0])
    for k_, r_ in enumerate(kind_rep_layer[1:], 1):
        for p_, (a_, b_) in enumerate(zip(seq0, normseq(r_))):
            if a_ == b_: continue
            j_ = L[r_][p_]
            for e in N[j_].ins:
                if e[0] != "op" or e[1] in hid or e[1] in set(L[r_]): continue
                x_ = e[1]
                trig_ = any(N[y].func in TRIG for y in R.back(x_, lambda z: False, 300)[1] + [x_])
                if not (trig_ or N[x_].func == "where"):
                    raise NotDrawn(f"layer {r_} differs from layer {kind_rep_layer[0]} in op {j_} {N[j_].func} reading "
                                   f"{N[x_].func} (not a mask or rotation table): per-layer differences of that kind are "
                                   "not drawn")
    kinds = []
    for k_ in range(len(kind_rep_layer)):
        kinds.append({"mask": kind_masks[k_][0], "window": kind_masks[k_][1], "theta": kind_thetas[k_]})
    kind_list = list(dict.fromkeys(repr(k_) for k_ in kinds))         # identical facts -> one drawn kind
    kind_of_layer = [kind_list.index(repr(kinds[layer_kind[n_]])) for n_ in range(len(layer_ids))]
    kinds = [eval(k_) for k_ in kind_list]
    if len(kinds) > 1 or kinds[0]["window"]:
        ev["layers"].append("layer kinds: " + ", ".join(
            f"{chr(65 + n_)} = {k_['mask']}{' window ' + str(k_['window']) if k_['window'] else ''}"
            f"{', RoPE theta ' + format(k_['theta'], 'g') if k_['theta'] else ''}" for n_, k_ in enumerate(kinds)))
    facts = dict(repo=repo, arch=arch, vocab=vocab, hidden=hidden, tied=tied, final_norm=final_norm, layers=len(layer_ids),
                 identical=len(identical), layer_ops=len(lops), kind=kind, num_heads=num_heads, num_kv=num_kv, head_dim=head_dim,
                 q_out=q_out, k_out=k_out, o_shape=((R.pshape.get(wo) if w_layout == "out_in" else tuple(reversed(R.pshape[wo])))
                                                    if wo else None), fused=fused,
                 qkv_layout=split["layout"] if fused else None, qkv_group=split["group"] if fused else None,
                 qkv_width=split["out"] if fused else None, w_layout=w_layout,
                 rot_dim=(rot_dim if rope_q and rot_dim != head_dim else None),
                 parallel_norms=parallel_norms, pos_emb=pos_emb, head_bias=N[head].func == "addmm",
                 gate_up=({"layout": gate_up["layout"], "inter": gate_up["inter"]} if gate_up else None),
                 layer_kinds=kinds, kind_of_layer=kind_of_layer, attn_softcap=attn_softcap, final_softcap=final_softcap,
                 attn_softcap_order=softcap_order,
                 norm_eps=norm_info["eps"], norm_weight=norm_info["weight"], final_eps=final_info["eps"],
                 final_weight=final_info["weight"], qk_eps=qk_info["eps"] if qk_info else None,
                 qk_weight=qk_info["weight"] if qk_info else None, embed_scale=embed_scale,
                 act_form="tanh_formula" if act_form else None, qk_norm=bool(qk_norm_q and qk_norm_k),
                 qk_norm_kind=qk_norm_q, qn_extent=(R.pshape[qn_w][0] if qn_w else None), norm_before_rope=norm_before_rope,
                 rope=rope_q and rope_k, theta=theta, scale=scale, mask=mask, bias=bias, ffn_bias=ffn_bias, o_proj=o_proj is not None,
                 gated=gated, act=act, inter=inter, norm=n1, placement=placement, residual=residual, v_out=v_out,
                 o_bias=o_bias,
                 max_pos=(bundle.get("max_position_embeddings") if bundle.get("max_position_embeddings") is not None else cfg.get("max_position_embeddings")), total_ops=len(N), evidence=dict(ev), cache=cache,
                 seq=P["inputs"]["input_ids"][-1] if isinstance(P["inputs"].get("input_ids"), list) else None,
                 graded={k: bundle.get(k) for k in ("verdict", "failed_checks", "reason", "checks", "decisions", "leftovers", "unknowns")},
                 pass_label=label, accounting=accounting, tie_both_shipped=tie_both_shipped,
                 config_view={k: cfg.get(k) for k in ("hidden_size", "vocab_size", "num_hidden_layers",
                              "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
                              "tie_word_embeddings", "hidden_act", "rope_theta")}
                             | {"rope_theta_params": ((cfg.get("rope_parameters") or {}).get("rope_theta")
                                                      if isinstance(cfg.get("rope_parameters"), dict) else None)},
                 rope_note=(rope_note + ("; the drawn base comes from the unscaled high-frequency entries of the "
                                         "rotation buffer" if theta is not None else "; the base is not measurable "
                                         "from this run's rotation buffer")) if rope_note else None)
    return facts


def fmt(n):
    return f"{n:,}"


def _scorecard(F):
    """Per-fact provenance for the IR's fact ledger (status execution_observed / config_declared) and the visible
    EXECUTION EVIDENCE section: graded verdict and checks, op accounting, every drawn fact with its evidence and its
    agreement with the config (the config is only checked against, never the source of an observed fact)."""
    from model_unfolder.evidence.context import FactLedger, FACT_STATUSES
    if "execution_observed" not in FACT_STATUSES:
        raise RuntimeError("this renderer's IR has no 'execution_observed' fact status: render against "
                           "model-benchmark/renderer_snapshot (see its SNAPSHOT_PATCHES.md) or apply that patch")
    E, C, G = F.get("evidence") or {}, F.get("config_view") or {}, F.get("graded") or {}
    first = lambda k: (E.get(k) or [""])[0]
    theta_cfg = C.get("rope_theta") if C.get("rope_theta") is not None else C.get("rope_theta_params")
    hd_cfg = C.get("head_dim") or ((C["hidden_size"] // C["num_attention_heads"])
                                   if C.get("hidden_size") and C.get("num_attention_heads") else None)
    ACT_CFG = {"silu": "silu", "swish": "silu", "gelu": "gelu", "relu": "relu", "gelu_pytorch_tanh": "gelu_pytorch_tanh",
               "gelu_new": "gelu_pytorch_tanh", "gelu_fast": "gelu_pytorch_tanh", "sigmoid": "sigmoid", "tanh": "tanh"}
    act_cfg = ACT_CFG.get(C.get("hidden_act"))                 # config names outside the map are not compared
    cache = F.get("cache") or {}
    rows = [  # owner, fact, label, value, status, evidence, config value (None: nothing to check against)
        ("model", "vocab_size", "vocabulary", F["vocab"], "execution_observed", first("embedding"), C.get("vocab_size")),
        ("model", "hidden_size", "hidden width", F["hidden"], "execution_observed", first("embedding"), C.get("hidden_size")),
        ("model", "num_hidden_layers", "layers", F["layers"], "execution_observed", first("layers"), C.get("num_hidden_layers")),
        (("model", "tie_word_embeddings", "tied output head", F["tied"], "execution_observed", first("tied"),
          C.get("tie_word_embeddings")) if F["tied"] is not None else
         ("model", "tie_word_embeddings", "tied output head",
          "config ties; checkpoint ships both (unchecked)" if F.get("tie_both_shipped") else
          "config ties; shipped tensors not recorded (unverified)", "config_declared", first("tied"), None)),
        ("model", "final_norm_kind", "final norm", F["final_norm"], "execution_observed",
         "norm ops between the last layer and the head", None),
        ("decoder.attention", "bias", "Q/K/V bias", F["bias"], "execution_observed", "addmm on the Q/K/V projections", None),
        ("decoder.attention", "num_heads", "query heads", F["num_heads"], "execution_observed", first("attention"), C.get("num_attention_heads")),
        ("decoder.attention", "num_kv_heads", "K/V heads", F["num_kv"], "execution_observed", first("attention"), C.get("num_key_value_heads")),
        ("decoder.attention", "head_dim", "head width", F["head_dim"], "execution_observed", "query reshape shape", hd_cfg),
        ("decoder.attention", "scores_scale", "score scale", round(F["scale"], 6) if F["scale"] else None,
         "execution_observed", "product of the scalar multipliers on the score path", None),
        ("decoder.attention", "mask", "mask", "sliding" if F.get("window") else F["mask"], "execution_observed",
         "captured additive mask operand on the scores", None),
        ("decoder.attention", "rope_theta", "RoPE base", F["theta"], "execution_observed" if F["theta"] else "unknown",
         first("rope") or "no measurable rotation buffer", theta_cfg),
        ("decoder.attention", "qk_norm", "QK-norm", F["qk_norm"], "execution_observed", "norm ops on the query/key paths", None),
        (("decoder.attention", "cached", "K/V cache", bool(cache.get("verified")), "execution_observed",
          "separate cache-on run of the drawn pass: stored K/V produced by this layer's projections and read by its "
          "scores", None) if cache else
         ("decoder.attention", "cached", "K/V cache", None, "unknown", "the cache-on run is missing or failed", None)),
        ("decoder.attention", "o_bias", "output projection bias", F.get("o_bias"), "execution_observed",
         "addmm on the output projection", None),
        ("decoder.ffn", "intermediate_size", "FFN width", F["inter"], "execution_observed", first("mlp"), C.get("intermediate_size")),
        ("decoder.ffn", "activation", "activation", F["act"], "execution_observed", first("mlp"), act_cfg),
        ("decoder.ffn", "gated", "gated FFN", F["gated"], "execution_observed", first("mlp"), None),
        ("decoder.layer", "norm_kind", "layer norms", F["norm"], "execution_observed", first("residual"), None),
        ("decoder.layer", "residual_topology", "residual", F["residual"], "execution_observed", first("residual"), None),
        ("model", "max_position_embeddings", "context length", F["max_pos"], "config_declared",
         "config.max_position_embeddings (not observable in one run)", None),
    ]
    rows += [("decoder.layer", "norm_eps", "layer norm eps", F["norm_eps"], "execution_observed",
              "the constant the recorded norm statistic adds", None),
             ("decoder.layer", "norm_weight", "layer norm weight", NORM_WEIGHT_TEXT[F["norm_weight"]], "execution_observed",
              "how the recorded norm reads its weight parameter", None),
             ("model", "final_norm_eps", "final norm eps", F["final_eps"], "execution_observed",
              "the constant the recorded final-norm statistic adds", None),
             ("model", "final_norm_weight", "final norm weight", NORM_WEIGHT_TEXT[F["final_weight"]], "execution_observed",
              "how the recorded final norm reads its weight parameter", None)]
    if F.get("qk_eps") is not None:
        rows += [("decoder.attention", "qk_norm_eps", "QK-norm eps", F["qk_eps"], "execution_observed",
                  "the constant the recorded QK-norm statistic adds", None),
                 ("decoder.attention", "qk_norm_weight", "QK-norm weight", NORM_WEIGHT_TEXT[F["qk_weight"]], "execution_observed",
                  "how the recorded QK-norm reads its weight parameter", None)]
    if F.get("layer_kinds") and (len(F["layer_kinds"]) > 1 or F["layer_kinds"][0]["window"]):
        rle_, desc_ = layer_pattern_text(F)
        rows += [("decoder.layer", "layer_pattern", "layer pattern", rle_, "execution_observed",
                  "layers grouped by the shared mask / rotation tensors they read; each kind's mask replayed", None),
                 ("decoder.layer", "layer_kinds", "layer kinds", desc_, "execution_observed", first("layers"), None)]
    if F.get("attn_softcap"):
        rows.append(("decoder.attention", "logit_softcap", "attention logit soft-cap", F["attn_softcap"],
                     "execution_observed", first("softcap"), None))
    if F.get("final_softcap"):
        rows.append(("model", "final_logit_softcap", "final logit soft-cap", F["final_softcap"], "execution_observed",
                     "div / tanh / mul on the logits", None))
    if F.get("gate_up"):
        rows.append(("decoder.ffn", "gate_up_split", "fused gate/up split", gate_up_text(F), "execution_observed",
                     first("mlp_split"), None))
    if F.get("embed_scale"):
        rows.append(("model", "embedding_scale", "embedding scale", F["embed_scale"]["value"], "execution_observed",
                     first("embedding_scale"), None))
    if F.get("fused"):
        rows.append(("decoder.attention", "qkv_split", "fused Q/K/V split", qkv_split_text(F), "execution_observed",
                     first("qkv_split"), None))
    if F.get("w_layout") == "in_out":
        rows.append(("decoder.layer", "weight_layout", "projection weights", "stored [in, out]", "execution_observed",
                     "every layer projection reads its weight directly as the right matmul operand (x @ W)", None))
    if F.get("rot_dim"):
        rows.append(("decoder.attention", "rotary_dim", "rotary dims", F["rot_dim"], "execution_observed",
                     "the rotated slice x[..., :r] and the pass-through x[..., r:] joined by one cat", None))
    if F.get("parallel_norms"):
        rows.append(("decoder.layer", "parallel_norm_count", "parallel branch norms", F["parallel_norms"],
                     "execution_observed", "norm ops between the layer input and the attention / MLP projections", None))
    if F.get("pos_emb"):
        rows.append(("model", "position_embedding", "position embedding", "learned absolute", "execution_observed",
                     first("position"), None))
    if F.get("head_bias"):
        rows.append(("model", "lm_head_bias", "LM head bias", True, "execution_observed", "addmm on the LM head", None))
    L, facts = FactLedger(), []
    for owner, fact, label, value, status, src, cfg_value in rows:
        L.record(owner, fact, value, status, src or None)
        check = None
        if status == "execution_observed" and cfg_value is not None and value is not None:
            num = isinstance(value, float) and isinstance(cfg_value, (int, float)) and not isinstance(cfg_value, bool)
            same = abs(value - cfg_value) <= 1e-3 * abs(cfg_value) if num else value == cfg_value
            check = "agrees" if same else f"says {cfg_value}"
        facts.append({"key": f"{owner}.{fact}", "label": label, "value": value, "status": status, "source": src,
                      "config_check": check})
    card = {"verdict": G.get("verdict"), "checks": G.get("checks") or {}, "pass": F.get("pass_label"),
            "total_ops": F.get("total_ops"), "op_accounting": F.get("accounting") or {}, "facts": facts,
            "source": "execution recording (zero-storage weights, real tokenized prompt)"}
    return L.to_dict(), card


NORM_WEIGHT_TEXT = {"w": "weight w", "1+w": "weight 1 + w", "none": "no weight"}


def norm_chips(eps, weight):
    """The page chips of a norm's eps and weight form (the oracle's page_reader parses exactly these)."""
    return [f"eps {eps:g}", NORM_WEIGHT_TEXT[weight]]


def gate_up_text(F):
    I_ = fmt(F["gate_up"]["inter"])
    return f"[gate {I_} | up {I_}]" if F["gate_up"]["layout"] == "gate_up" else f"[up {I_} | gate {I_}]"


def qkv_split_text(F):
    """The page text of a fused Q/K/V split (the oracle's page_reader parses exactly this)."""
    if F.get("qkv_layout") == "qkv_blocks":
        return f"[Q {fmt(F['q_out'])} | K {fmt(F['k_out'])} | V {fmt(F['v_out'])}]"
    return (f"per K/V group [q × {F['qkv_group']} | k | v] × {F['num_kv']}, head dim {F['head_dim']}")


def layer_pattern_text(F):
    """('A×5 B×1 ...', 'A = ...; B = ...') of a mixed / windowed layer stack (the page text page_reader parses)."""
    ks, ko = F["layer_kinds"], F["kind_of_layer"]
    runs, prev = [], None
    for k in ko:
        if runs and runs[-1][0] == k: runs[-1][1] += 1
        else: runs.append([k, 1])
    rle = " ".join(f"{chr(65 + k)}×{n}" for k, n in runs)
    desc = "; ".join(f"{chr(65 + n)} = " + (f"sliding window {k['window']}" if k["window"] else "causal")
                     + (f", RoPE θ {k['theta']:g}" if k["theta"] else "") for n, k in enumerate(ks))
    return rle, desc


def to_ir(F):
    """Build unfold's ModelIR from execution facts. A stack of several layer kinds (sliding / full masks, local / global
    rotation bases) is built kind by kind with the same code and assembled in the recorded per-layer order."""
    kinds = F.get("layer_kinds") or [{"mask": "causal", "window": None, "theta": F["theta"]}]
    if len(kinds) == 1 and not kinds[0]["window"]:
        return _to_ir_core(F)
    irs = [_to_ir_core(dict(F, theta=k["theta"], window=k["window"])) for k in kinds]
    base = irs[F["kind_of_layer"][0]]
    base.layers = [dataclasses.replace(irs[k].layers[0], index=i) for i, k in enumerate(F["kind_of_layer"])]
    return base


def _to_ir_core(F):
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
    attn = make(I.AttentionSpec, rope_dim=F.get("rot_dim"), kind=F["kind"], mixer_state="ordinary_attention", num_heads=F["num_heads"], num_kv_heads=F["num_kv"],
                head_dim=F["head_dim"], rope_theta=F["theta"], mask="sliding" if F.get("window") else F["mask"],
                window_size=F.get("window"), qk_norm=F["qk_norm"], qk_norm_kind=F["qk_norm_kind"],
                qk_norm_axis="last_dimension" if F["qk_norm"] else None, qk_norm_extent=F["qn_extent"],
                qk_norm_placement=("after_reshape" if F["qn_extent"] == F["head_dim"] else "before_reshape") if F["qk_norm"] else None,
                rope=F["rope"], position_kind="rope" if F["rope"] else None, position_application="qk_rotation" if F["rope"] else None,
                bias=F["bias"], output_projection=F["o_proj"], projection_mode="fused_qkv" if F["fused"] else "split_qkv",
                scores_scaled=F["scale"] is not None, cached=(cache_ok if C else None),    # no cache run: unresolved
                cache_evidence=({"payload": ["key", "value"],
                                 "location": f"keys after {F['cache']['write']['key']['after']}, values after {F['cache']['write']['value']['after']}",
                                 "guard": "observed in a run with use_cache=True"} if cache_ok else None))
    ffn = make(I.FFNSpec, kind="dense", activation=F["act"], intermediate_size=F["inter"], gated=F["gated"], bias=F.get("ffn_bias", False),
               projection_mode=("fused_gate_up" if F.get("gate_up") else "split") if F["gated"] else None)
    H, hd, nh, nkv = F["hidden"], F["head_dim"], F["num_heads"], F["num_kv"]
    attn_facts = [f"{nh} Q heads", f"{nkv} KV heads", f"head dim {hd}"]
    if F["theta"]: attn_facts.append(f"RoPE θ {fmt(round(F['theta']))}")
    if F["qk_norm"]: attn_facts.append("QK-Norm")
    if not F["bias"] and not F.get("o_bias"): attn_facts.append("bias-free projections")
    elif F["bias"] and not F.get("o_bias"): attn_facts.append("bias on Q/K/V, none on the output projection")
    elif F.get("o_bias") and not F["bias"]: attn_facts.append("bias on the output projection only")
    children = ([
        {"id": "qkv_proj", "title": "Fused QKV projection",
         "description": f"One recorded matmul produces Q, K and V; its {fmt(F['qkv_width'])} output channels split as "
                        f"{qkv_split_text(F)} (traced from the recorded split ops).",
         "facts": [f"{fmt(H)} → {fmt(F['qkv_width'])}", f"split {qkv_split_text(F)}"]}] if F.get("fused") else []) + [
        {"id": "q_proj", "title": "Query projection", "description": "Recorded matmul producing the per-head queries.", "facts": [f"{fmt(H)} → {fmt(F['q_out'])}", f"{nh} Q heads", f"head dim {hd}"]},
        {"id": "k_proj", "title": "Key projection", "description": "Recorded matmul producing the keys.", "facts": [f"{fmt(H)} → {fmt(F['k_out'])}", f"{nkv} KV heads"] + (["cache ports: ⌃ write · ⊥ read"] if cache_ok else [])},
        {"id": "v_proj", "title": "Value projection", "description": "Recorded matmul producing the values.", "facts": [f"{fmt(H)} → {fmt(F['v_out'])}", f"{nkv} KV heads"] + (["cache ports: ⌃ write · ⊥ read"] if cache_ok else [])},
        {"id": "q_reshape", "title": "Reshape query heads", "description": "The projected width is split into heads before the next op.", "facts": [f"head dim {hd}"]},
        {"id": "k_reshape", "title": "Reshape key heads", "description": "The projected width is split into heads before the next op.", "facts": [f"head dim {hd}"]},
    ]
    if F.get("fused"):
        children = [c for c in children if c["id"] not in ("q_proj", "k_proj", "v_proj")]
    if F.get("w_layout") == "in_out": attn_facts.append("weights stored [in, out]")
    if F.get("rot_dim"): attn_facts.append(f"partial RoPE {F['rot_dim']}/{hd}")
    if F.get("window"): attn_facts.append(f"sliding window {F['window']}")
    if F.get("attn_softcap"): attn_facts.append(f"logit soft-cap {F['attn_softcap']:g}")
    if F["qk_norm"]:
        where = "before RoPE" if F["norm_before_rope"] else "after RoPE"
        children += [{"id": "q_norm", "title": f"Query {NK.get(F['qk_norm_kind'], 'norm')}", "description": f"Recorded {NK.get(F['qk_norm_kind'])} on Q over the last axis ({F['qn_extent']}), {where}.", "facts": norm_chips(F["qk_eps"], F["qk_weight"]) if F.get("qk_eps") is not None else []},
                     {"id": "k_norm", "title": f"Key {NK.get(F['qk_norm_kind'], 'norm')}", "description": f"Recorded {NK.get(F['qk_norm_kind'])} on K over the last axis ({F['qn_extent']}), {where}.", "facts": norm_chips(F["qk_eps"], F["qk_weight"]) if F.get("qk_eps") is not None else []}]
    if F["rope"]:
        th = f"θ={F['theta']:.6g}" if F["theta"] else "θ unknown"
        part = (f" Rotates the first {F['rot_dim']} of {hd} head dims; the other {hd - F['rot_dim']} pass through."
                if F.get("rot_dim") else "")
        pf = [f"rotary dims {F['rot_dim']} of {hd}"] if F.get("rot_dim") else []
        children += [{"id": "q_rope", "title": "Apply RoPE (Q)", "description": f"Rotate-half rotary embedding recorded on the query heads; frequency base from the built model's buffer: {th}." + part, "facts": [f"frequency base {th}"] + pf},
                     {"id": "k_rope", "title": "Apply RoPE (K)", "description": f"Rotate-half rotary embedding recorded on the key heads; {th}." + part, "facts": [f"frequency base {th}"] + pf}]
    children += [
        {"id": "scaled_scores", "title": "Scaled dot-product scores", "description": f"Q·Kᵀ recorded; {scale_txt or 'no scale constant recorded'}"
         + (f"; scores soft-capped tanh(x / {F['attn_softcap']:g}) × {F['attn_softcap']:g} {F.get('attn_softcap_order')} the "
            "mask is added" if F.get("attn_softcap") else "")
         + (f"; mask: sliding window {F['window']} (causal; window read from the replayed mask construction)." if F.get("window")
            else f"; mask: {F['mask'] or 'none'} (read from the mask tensor's values)."),
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
        if F.get("gate_up"):
            gu_ = gate_up_text(F)
            ffn_children = [{"id": "gate_up_proj", "label": "Linear (gate|up)", "title": "Fused gate/up projection",
                             "description": f"One recorded matmul produces the gate and up halves, split as {gu_} (traced "
                                            "from the recorded split ops).",
                             "facts": [f"{fmt(H)} → {fmt(2 * F['inter'])}", f"split {gu_}"]}] + \
                ffn_children[2:]
        ffn_block = {"id": "ffn", "role": "ffn", "kind": "ffn", "label": "Gated FFN", "title": "Gated feed-forward",
                     "description": "Gated MLP — recorded gate path modulates the up projection before the down projection.",
                     "facts": [actl, f"hidden {fmt(F['inter'])}", "split gate/up"], "view": "gated_ffn",
                     "detail": {"ffn": dataclasses.asdict(ffn)}, "children": ffn_children}
    else:
        ffn_block = {"id": "ffn", "role": "ffn", "kind": "ffn", "label": "FFN", "title": "Feed-forward", "description": "Recorded two-layer MLP.",
                     "facts": [actl, f"hidden {fmt(F['inter'])}"], "detail": {"ffn": dataclasses.asdict(ffn)}}
    if F.get("w_layout") == "in_out": ffn_block["facts"] = ffn_block["facts"] + ["weights stored [in, out]"]
    blocks = [
        {"id": "rms1", "role": "norm", "kind": "norm", "label": nlabel, "title": "Pre-attention norm", "description": f"Recorded {nlabel} on the layer input before attention.", "facts": [f"dim {fmt(H)}"] + norm_chips(F["norm_eps"], F["norm_weight"])},
        attn_block,
        {"id": "add1", "role": "residual", "kind": "residual_add", "residual_from": "rms1", "label": "+", "title": "Residual add", "description": "Recorded add: block input + attention output."},
        {"id": "rms2", "role": "norm", "kind": "norm", "label": nlabel, "title": "Pre-FFN norm", "description": f"Recorded {nlabel} before the FFN.", "facts": [f"dim {fmt(H)}"] + norm_chips(F["norm_eps"], F["norm_weight"])},
        ffn_block,
        {"id": "add2", "role": "residual", "kind": "residual_add", "residual_from": "rms2", "label": "+", "title": "Residual add", "description": "Recorded add: post-attention + FFN output."},
    ]
    if F.get("placement") in ("post", "double"):     # norms on the sublayer OUTPUT (post: OLMo-2; double: sandwich)
        nf_ = [f"dim {fmt(H)}"] + norm_chips(F["norm_eps"], F["norm_weight"])
        post_a = {"id": "post_attn_ln", "role": "norm", "kind": "norm", "label": nlabel, "title": "Post-attention norm",
                  "description": f"Recorded {nlabel} on the attention OUTPUT before the residual add.", "facts": nf_}
        post_f = {"id": "post_ffn_ln", "role": "norm", "kind": "norm", "label": nlabel, "title": "Post-FFN norm",
                  "description": f"Recorded {nlabel} on the FFN OUTPUT before the residual add.", "facts": nf_}
        dbl = F["placement"] == "double"
        add1_ = {"id": "add1", "role": "residual", "kind": "residual_add", "residual_from": "rms1" if dbl else "attn",
                 "label": "+", "title": "Residual add", "description": "Recorded add: block input + normed attention output."}
        add2_ = {"id": "add2", "role": "residual", "kind": "residual_add", "residual_from": "rms2" if dbl else "ffn",
                 "label": "+", "title": "Residual add", "description": "Recorded add: post-attention + normed FFN output."}
        blocks = ([blocks[0], attn_block, post_a, add1_, blocks[3], ffn_block, post_f, add2_] if dbl else
                  [attn_block, post_a, add1_, ffn_block, post_f, add2_])
    if F.get("parallel_norms"):                      # parallel branches (the renderer's GPT-NeoX / GPT-J lane layout)
        two = F["parallel_norms"] == 2
        blocks = [
            {"id": "rms1", "role": "norm", "kind": "norm", "label": nlabel,
             "title": "Pre-block norms (separate)" if two else "Pre-block norm (shared)",
             "description": (f"Recorded {nlabel} on the layer input before attention; a second, separate {nlabel} on the "
                             "layer input before the FFN." if two else
                             f"Recorded {nlabel} on the layer input, feeding both attention and the FFN."),
             "facts": [f"dim {fmt(H)}"] + norm_chips(F["norm_eps"], F["norm_weight"])},
            attn_block,
            {"id": "add1", "role": "residual", "kind": "residual_add", "residual_from": "rms1", "label": "+",
             "title": "Residual add (parallel)", "description": "Recorded adds: layer input + attention output + FFN output."},
            dict(ffn_block, lane="left", tap_from="attn", feeds="add1", side_align="tap", w=220),
        ]
    layers = [make(I.LayerSpec, index=i, attention=attn, ffn=ffn, norm_kind=F["norm"], norm_placement=F["placement"],
                   residual_topology=F["residual"], blocks=blocks,
                   **({"parallel_norm_count": F["parallel_norms"]} if F.get("parallel_norms") else {}))
              for i in range(F["layers"])]
    tie_txt = (" — weights tied with the output head (same tensor read; the checkpoint ships one of them)."
               if F["tied"] is True else
               (" — tied to the output head in the model built from the config, but the checkpoint ships both "
                "tensors: the library ties them only if their values are equal, which was not checked."
                if F.get("tie_both_shipped") else
                " — tied to the output head in the model built from the config; whether the checkpoint ships one "
                "or both tensors is not recorded, so the tie is unverified.")
               if F["tied"] is None else ".")
    fn = NK.get(F["final_norm"], "Norm")
    model_blocks = [
        {"id": "tok_text", "role": "input", "kind": "source", "label": "Tokenized text", "title": "Tokenized text", "description": "Input token IDs.", "facts": [f"shape [1, {F['seq']}] in the recorded run"]},
        {"id": "embed", "role": "embedding", "kind": "embedding", "label": "Token Embedding layer", "title": "Token embedding", "description": "Maps each token id to its vector" + tie_txt, "facts": [f"{fmt(F['vocab'])} vocab", f"{fmt(H)}-d"]},
        {"id": "final_rms", "role": "norm", "kind": "norm", "label": f"Final {fn}", "title": "Final norm", "description": f"Recorded {fn} over the last hidden state before the output head.", "facts": [f"dim {fmt(H)}"] + norm_chips(F["final_eps"], F["final_weight"]), "resolved": True},
        {"id": "lm_head", "role": "output", "kind": "output", "label": "Linear output layer", "title": "LM head", "description": "Recorded matmul into vocabulary logits" + tie_txt, "facts": [f"{fmt(H)} → {fmt(F['vocab'])}"]},
    ]
    if F.get("head_bias"):
        model_blocks[-1]["facts"] = model_blocks[-1]["facts"] + ["with bias"]
    if F.get("final_softcap"):
        model_blocks[-1]["facts"] = model_blocks[-1]["facts"] + [f"logit soft-cap {F['final_softcap']:g}"]
    if F.get("embed_scale"):
        model_blocks[1]["facts"] = model_blocks[1]["facts"] + [f"scaled × {F['embed_scale']['value']:.8g}"]
    if F.get("pos_emb"):
        model_blocks.extend([
            {"id": "position_ids", "role": "input", "kind": "source", "label": "Position IDs", "title": "Position indices",
             "description": "Sequence positions 0..S-1 (recorded arange) used to look up learned positional vectors."},
            {"id": "position_embed", "role": "embedding", "kind": "embedding", "label": "Learned Position Embedding",
             "title": "Learned Position Embedding",
             "description": "Looks up one learned positional vector for each sequence position.",
             "facts": [f"{fmt(F['pos_emb']['shape'][0])} positions", f"{fmt(F['pos_emb']['shape'][1])}-d"]},
            {"id": "position_add", "role": "residual", "kind": "residual_add", "label": "+", "title": "Token + position embedding",
             "detail": {"position_kind": "learned_absolute", "position_application": "embedding_add"},
             "description": "Adds the learned positional vector to the token embedding before the decoder stack."},
        ])
    extras = {"render": {"family": "transformer", "layout": "decoder_only", "model_blocks": model_blocks},
              "exec_capture": {"source": "execution recording (zero-storage weights, real tokenized prompt)", "total_ops": F["total_ops"],
                               "layers_identical": f"{F['identical']} of {F['layers']}", "evidence": F["evidence"],
                               "declared_not_observed": {"max_position_embeddings": F["max_pos"]}}}
    extras["fact_provenance"], extras["execution_evidence"] = _scorecard(F)
    ir = make(I.ModelIR, name=F["repo"].split("/")[-1], architecture=F["arch"], vocab_size=F["vocab"], hidden_size=H,
              max_position_embeddings=F["max_pos"], tie_word_embeddings=F["tied"], embedding_norm_kind=None,
              **({"embedding_scale": F["embed_scale"]["value"]} if F.get("embed_scale") else {}),
              final_norm_kind=F["final_norm"], layers=layers, cross_layer_edges=[], extras=extras,
              notes=["Built from an execution recording: every fact is observed in the run harness v2 graded "
                     f"(verdict {G.get('verdict')}, pass '{F.get('pass_label')}'; no modeling-code parsing). "
                     "Context length is the config's declared value (not observable in one run)."]
                    + ([f"Benchmark verdict FULL_LEFTOVERS: the only shipped tensors not executed are "
                        + "; ".join(x for x in [
                            (f"tensors the library itself declares unused ({', '.join(map(str, (G.get('leftovers') or {}).get('library_declared_unbuilt') or []))})"
                             if (G.get('leftovers') or {}).get('library_declared_unbuilt') else ""),
                            (f"{len((G.get('leftovers') or {}).get('buffers_never_read_in_any_mode') or [])} stored buffers measured never "
                             "read in any run mode (e.g. training statistics; not declared by the library)"
                             if (G.get('leftovers') or {}).get('buffers_never_read_in_any_mode') else "")] if x) + "."]
                       if G.get("verdict") == "FULL_LEFTOVERS" else [])
                    + ([F["rope_note"]] if F.get("rope_note") else []),
              warnings=([f"Executed class is arbitrary among equally fitting library classes "
                         f"({', '.join((G.get('decisions') or {})['class_ambiguous'].get('tied') or [])}): no checkpoint "
                         f"evidence chooses between them; only what every one of them supports was run."]
                        if (G.get("decisions") or {}).get("class_ambiguous") else []) +
                       ([] if G.get("verdict") in ("FULL", "FULL_LEFTOVERS") else
                        [f"Benchmark verdict FULL_UNVERIFIED: every observed check passed, but not everything could be "
                         f"verified: " + "; ".join(f"{u.get('what')}" + (f" ({u.get('count')}: {', '.join(map(str, (u.get('items') or [])[:3]))})"
                                                                     if u.get('count') else "") for u in (G.get('unknowns') or []))
                         + ". Drawn from what ran."] if G.get("verdict") == "FULL_UNVERIFIED" else
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
    try:
        if "--bundle" in sys.argv:
            B = capture.load(sys.argv[sys.argv.index("--bundle") + 1])
        else:
            B = bundle_for(repo, sys.argv[sys.argv.index("--captures") + 1] if "--captures" in sys.argv else CAPTURES,
                           refresh="--refresh" in sys.argv)
    except ValueError as e:                      # a stale or foreign bundle: say so, never draw from it
        print("NOT DRAWN:", e); sys.exit(2)
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
