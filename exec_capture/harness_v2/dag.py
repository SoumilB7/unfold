"""Full dataflow recorder for one real forward pass (zero-storage weights + exact zero-skip).

Every tensor is mapped to the op that produced it; every op records its input edges
(op -> op, param, buffer, model input, or EXTERNAL = appeared from outside the recording),
the module it ran inside, and its scalar constants. Nothing is inferred from names or source text.
"""
import re, math, cmath, numbers, weakref, collections
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map
import lowcost
from lowcost import ZERO_PTRS, ZERO_REG
aten = torch.ops.aten

OPAQUE = re.compile(r"(scaled_dot_product|flash|_efficient_attention|cudnn|_fused|fused_|mkldnn_rnn|_cufft)")
CONTROL_SINKS = {"_local_scalar_dense", "is_nonzero", "equal", "item"}


# ---- F07: the zero-skip returns an op's EXACT result without arithmetic, one rule per op (dossier
#      model-benchmark/_reruns/p2a1_f07). Z(t) = known all-zero (registered zero storage, or by value); F(t) = no inf/nan.
COW_EVENTS = []          # forward-pass writes that gave a whole zero-storage weight its own storage (F07-1)
NONFINITE_EVENTS = []    # ops that turned finite inputs into NaN / inf (zero-weight artefact, F07-3)
ZSW_EVENTS = []          # forward-pass writes through a view of shared zero storage (refused; a G2 unknown)


class ZeroStorageWrite(RuntimeError):
    """A forward-pass write into shared zero storage through a view / alias that cannot be given its own storage."""


def is_reg(t):
    return isinstance(t, torch.Tensor) and t.numel() > 1 and t.untyped_storage().data_ptr() in ZERO_REG


def known_zero(t):
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return False
    if is_reg(t):
        return True
    return t.numel() > 1 and not bool(torch.any(t))


def finite(t):
    if t is None or not isinstance(t, torch.Tensor) or not (t.is_floating_point() or t.is_complex()) or is_reg(t):
        return True
    return bool(torch.isfinite(t).all())


def product_zero(a, b):
    """a contracted with b is exactly zero: one factor is all-zero and the other has no inf / nan (0 * inf = nan)."""
    return (known_zero(a) and finite(b)) or (known_zero(b) and finite(a))


def _meta(func, args, kwargs):
    """The real op's output metadata (shape, dtype, strides), or None when even shape inference rejects the call: the
    caller then executes it for real (so the user's error is raised, never replaced by zeros)."""
    m = tree_map(lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x, (args, kwargs))
    try:
        return func(*m[0], **m[1])
    except RuntimeError:
        if func is not aten._grouped_mm.default:  # only grouped expert matmul's meta kernel needs bf16
            return None
        orig = next((a.dtype for a in tree_flatten(args)[0] if isinstance(a, torch.Tensor) and a.is_floating_point()), torch.float32)
        mb = tree_map(lambda x: x.to("meta", torch.bfloat16) if isinstance(x, torch.Tensor) and x.is_floating_point() else
                      (x.to("meta") if isinstance(x, torch.Tensor) else x), (args, kwargs))
        try:
            o = func(*mb[0], **mb[1])
        except RuntimeError:
            return None
        return o.to(orig) if o.is_floating_point() else o


def _zeros_like_meta(mo):
    return torch.empty_strided(mo.shape, mo.stride(), dtype=mo.dtype).zero_()       # the real op's strides


def _filled_like_meta(mo, value):
    return torch.empty_strided(mo.shape, mo.stride(), dtype=mo.dtype).copy_(value.expand(mo.shape))


SKIP_OPS = {aten.mm.default, aten.bmm.default, aten._grouped_mm.default, aten.addmm.default, aten.baddbmm.default,
            aten.embedding.default, aten.convolution.default}


def _one_dtype(args, kwargs):
    """All floating / complex tensor operands share one dtype (a mixed-dtype call raises in real torch: never skip it)."""
    return len({a.dtype for a in tree_flatten((args, kwargs))[0]
                if isinstance(a, torch.Tensor) and (a.is_floating_point() or a.is_complex())}) <= 1


def exact_zero_result(func, args, kwargs):
    """The op's exact result when a zero operand determines it, else None (= execute normally)."""
    if not _one_dtype(args, kwargs):
        return None
    r = _exact_zero_result(func, args, kwargs)
    return r


def _exact_zero_result(func, args, kwargs):
    A = list(args) + [None] * 4
    if func in (aten.mm.default, aten.bmm.default):
        if not product_zero(A[0], A[1]):
            return None
        mo = _meta(func, args, kwargs)
        return None if mo is None else _zeros_like_meta(mo)
    if func is aten._grouped_mm.default:
        if kwargs.get("bias", A[3]) is not None or not product_zero(A[0], A[1]):
            return None
        mo = _meta(func, args, kwargs)
        return None if mo is None else _zeros_like_meta(mo)
    if func in (aten.addmm.default, aten.baddbmm.default):
        c, a, b = A[0], A[1], A[2]
        beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
        if not isinstance(alpha, numbers.Number) or not cmath.isfinite(alpha) or not isinstance(beta, numbers.Number):
            return None
        if not (c.is_floating_point() or c.is_complex()) and not float(beta).is_integer():
            return None                           # integer bias x fractional beta: torch's own rounding (execute)
        if not product_zero(a, b):
            return None                           # a zero bias alone never zeroes the product
        mo = _meta(func, args, kwargs)
        if mo is None:
            return None
        # beta == 0 ignores c (even nan); + 0.0 gives torch's sign of zero (-0.0 bias -> +0.0, as x*0 + c*beta does)
        return _zeros_like_meta(mo) if beta == 0 else _filled_like_meta(mo, c * beta + 0.0)
    if func is aten.embedding.default:
        w, idx = A[0], A[1]
        if not is_reg(w):
            return None                           # registered zero tables only (a -0.0 table by value keeps its sign)
        if idx.is_floating_point() or idx.is_complex():
            return None                           # the real op rejects non-integer indices
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= w.shape[0]):
            return None                           # the real op raises
        mo = _meta(func, args, kwargs)
        return None if mo is None else _zeros_like_meta(mo)
    if func is aten.convolution.default:
        x, w, b = A[0], A[1], A[2]
        if not product_zero(x, w):
            return None
        mo = _meta(func, args, kwargs)
        if mo is None:
            return None
        return _zeros_like_meta(mo) if b is None else _filled_like_meta(mo, b.reshape((1, -1) + (1,) * (mo.dim() - 2)) + 0.0)
    return None


def zero_storage_findings(models=()):
    """(G2 unknowns, notes) from this process's zero-storage execution (F07 rulings, Soumil 2026-10-13): non-finite
    values created from finite inputs; non-finite values held by the built models; build-time writes that could not be
    applied (over the size cap, or through an alias of storage no Parameter owns); writes refused through a view; zero
    storage changed outside the recorder. Copy-on-write and alias writes into Parameters (zero by design) are notes."""
    unk = []
    held = []
    pptrs = set()
    for m in models:
        for n_, p_ in m.named_parameters(remove_duplicate=False):
            pptrs.add(p_.untyped_storage().data_ptr())
        for n_, t_ in list(m.named_parameters(remove_duplicate=False)) + list(m.named_buffers(remove_duplicate=False)):
            # NaN is never a legitimate stored value; +-inf is (e.g. Gemma-4 clippable-linear clamp-bound buffers the
            # library's own init sets), so only NaN held by the build is a zero-weight artefact (learned params are reset)
            if (t_.is_floating_point() or t_.is_complex()) and t_.numel() and not is_reg(t_) and bool(torch.isnan(t_).any()):
                held.append(n_)
    if held:
        unk.append({"predicate": "G2", "what": "the zero-storage build holds NaN values a real model does not",
                    "items": held[:20], "count": len(held)})
    alias_unowned = [d for d, p in lowcost.BUILD_ALIAS_WRITES if p not in pptrs] if models else []
    if alias_unowned:
        unk.append({"predicate": "G2", "what": "build-time writes through an alias of zero storage no Parameter owns (a "
                    "computed buffer left as zeros)", "items": alias_unowned[:20], "count": len(alias_unowned)})
    if ZSW_EVENTS:
        unk.append({"predicate": "G2", "what": "forward-pass writes through a view of shared zero storage were refused "
                    "(the pass could not run faithfully)", "items": ZSW_EVENTS[:20], "count": len(ZSW_EVENTS)})
    if NONFINITE_EVENTS:
        unk.append({"predicate": "G2", "what": "the zero-weight execution produced non-finite values from finite inputs "
                    "(a real model never computes these): values downstream differ from the checkpoint's",
                    "items": sorted(set(NONFINITE_EVENTS))[:20], "count": len(NONFINITE_EVENTS)})
    if lowcost.BUILD_UNAPPLIED:
        unk.append({"predicate": "G2", "what": "build-time writes into zero storage could not be applied (computed values "
                    "left as zeros)", "items": lowcost.BUILD_UNAPPLIED[:20], "count": len(lowcost.BUILD_UNAPPLIED)})
    if not lowcost.registry_intact():
        unk.append({"predicate": "G2", "what": "shared zero storage was changed outside the recorder: weights sharing it "
                    "were not zero", "items": [], "count": 0})
    notes = {"copy_on_write": COW_EVENTS[:20], "build_alias_writes_left_zero": [d for d, _ in lowcost.BUILD_ALIAS_WRITES][:20],
             "learned_params_reset_to_zero": sorted(set(lowcost.BUILD_PARAMS_RESET))[:20]}
    return unk, {k: v for k, v in notes.items() if v}


ALLOC_OPS = {"empty", "empty_like", "new_empty", "new_empty_strided", "empty_strided", "empty_permuted"}
METADATA_INPLACE = {"unsqueeze_", "squeeze_", "t_", "transpose_", "as_strided_", "detach_", "swapdims_", "swapaxes_",
                    "requires_grad_"}                # change a view's metadata, never its values


def _write_targets(func, args, kwargs):
    if func.overloadpacket.__name__ in METADATA_INPLACE:
        return []
    out = []
    for i, sa in enumerate(func._schema.arguments):
        if sa.alias_info is not None and sa.alias_info.is_write:
            tgt = args[i] if i < len(args) else kwargs.get(sa.name)
            out += [t for t in (tgt if isinstance(tgt, (list, tuple)) else [tgt]) if isinstance(t, torch.Tensor)]
    return out


class Node:
    __slots__ = ("func", "module", "ins", "consts", "n_out", "shape")

    def __init__(self, func, module, ins, consts, n_out, shape):
        self.func, self.module, self.ins, self.consts, self.n_out, self.shape = func, module, ins, consts, n_out, shape


class DagRecorder(TorchDispatchMode):
    def __init__(self, model, inputs: dict, skip=True):
        super().__init__()
        self.skip = skip
        self.param_names = {id(t): n for n, t in model.named_parameters(remove_duplicate=False)}
        self.buffer_names = {id(t): n for n, t in model.named_buffers(remove_duplicate=False)}
        self.input_names = {id(v): k for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        self.const_names = {}            # tensors built at construction and stored as plain module attributes
        for mn, mod in model.named_modules():
            for an, av in vars(mod).items():
                if isinstance(av, torch.Tensor) and not isinstance(av, torch.nn.Parameter) and an not in ("_parameters", "_buffers"):
                    self.const_names.setdefault(id(av), f"{mn}.{an}" if mn else an)
        self.producer = {}          # id(tensor) -> node index (entry removed when the tensor dies)
        self.nodes = []
        self.used_params, self.used_buffers, self.used_inputs = set(), set(), set()
        self.external = []          # (shape, dtype, module) for tensors of unknown origin
        self.skipped = 0
        self.stack = ["(root)"]
        self._hooks = []
        for name, mod in model.named_modules():
            n = name or "(root)"
            self._hooks.append(mod.register_forward_pre_hook(lambda m, a, n=n: (self.stack.append(n), None)[1]))
            self._hooks.append(mod.register_forward_hook(lambda m, a, o, n=n: (self._pop(n), None)[1]))
        self.called = set()

    def _pop(self, n):
        if self.stack and self.stack[-1] == n:
            self.stack.pop()
            self.called.add(n)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    ZERO_KEEP = {"mul_", "div_", "zero_", "neg_", "relu_", "abs_", "clamp_", "clamp_min_", "clamp_max_", "tanh_", "sin_",
                 "sqrt_", "fill_", "masked_fill_", "embedding_renorm_"}     # (zero rows have norm 0: renorm keeps them)

    def _zero_preserving_inplace(self, func, args, kwargs):
        name = func.overloadpacket.__name__
        if name not in self.ZERO_KEEP or not args or not isinstance(args[0], torch.Tensor) or args[0].numel() <= 1:
            return False
        if args[0].untyped_storage().data_ptr() not in ZERO_REG:
            return False
        others = list(args[1:]) + list(kwargs.values())
        def val(x):
            return x.item() if isinstance(x, torch.Tensor) and x.numel() == 1 else x
        if name in ("fill_", "masked_fill_"):
            return all(val(x) == 0 for x in others[-1:] if not isinstance(val(x), torch.Tensor))
        if name == "mul_":
            return all(not isinstance(val(x), torch.Tensor) and math.isfinite(val(x)) or
                       isinstance(x, torch.Tensor) and bool(torch.isfinite(x).all()) for x in others)
        if name == "div_":
            x = others[0] if others else None
            if isinstance(val(x), torch.Tensor):
                return bool(torch.isfinite(x).all()) and bool((x != 0).all())
            return x is not None and val(x) != 0 and math.isfinite(val(x))
        if name in ("clamp_", "clamp_min_", "clamp_max_"):
            if any(isinstance(val(x), torch.Tensor) for x in others):
                return False                            # tensor bounds: not decided here (execute)
            lo = val(others[0]) if others else None
            hi = val(others[1]) if len(others) > 1 else None
            if name == "clamp_max_": lo, hi = None, lo
            return (lo is None or lo <= 0) and (hi is None or hi >= 0)
        return True                                     # zero_, neg_, relu_, abs_, tanh_, sin_, sqrt_ keep zero

    def _track(self, t, idx):
        k = id(t)
        self.producer[k] = idx
        try:
            weakref.finalize(t, self.producer.pop, k, None)
        except TypeError:
            pass

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        mod = self.stack[-1]
        ins, consts, reads_weight = [], [], False
        p_read, b_read = [], []          # credited only once the op has completed (F13: an op that raises reads nothing)
        for a in tree_flatten((args, kwargs))[0]:
            if isinstance(a, torch.Tensor):
                k = id(a)
                if k in self.producer:
                    ins.append(("op", self.producer[k]))
                elif k in self.param_names:
                    ins.append(("param", self.param_names[k])); p_read.append(self.param_names[k]); reads_weight = True
                elif k in self.buffer_names:
                    ins.append(("buffer", self.buffer_names[k])); b_read.append(self.buffer_names[k]); reads_weight = True
                elif k in self.input_names:
                    ins.append(("input", self.input_names[k])); self.used_inputs.add(self.input_names[k])
                elif k in self.const_names:
                    ins.append(("module_constant", self.const_names[k]))
                else:
                    ins.append(("external", tuple(a.shape), str(a.dtype)))
                    self.external.append((tuple(a.shape), str(a.dtype), mod, str(func)))
            elif isinstance(a, (int, float, bool)) and not isinstance(a, bool):
                consts.append(a)
        try:
            targets = _write_targets(func, args, kwargs)
        except Exception:
            targets = []
        out = None
        reg_targets = [t for t in targets if is_reg(t)]
        if reg_targets:
            if self._zero_preserving_inplace(func, args, kwargs):
                # an in-place op on zero-storage weights whose result is provably still zero (e.g. RWKV rescales weights
                # with div_ at inference): skip the write (the storage is shared), keep the op in the graph
                self.skipped += 1
                out = args[0]
            else:
                for t in reg_targets:             # F07-1 copy-on-write: a whole weight gets its own storage
                    if id(t) in self.param_names or id(t) in self.buffer_names:
                        torch.Tensor.set_(t, torch.zeros(t.shape, dtype=t.dtype))
                        COW_EVENTS.append(f"{func.overloadpacket.__name__} into {self.param_names.get(id(t)) or self.buffer_names.get(id(t))}")
                    else:
                        ZSW_EVENTS.append(f"{func.overloadpacket.__name__} in {mod}")
                        raise ZeroStorageWrite(f"{func} writes into shared zero storage through a view in {mod}")
        elif self.skip and func in SKIP_OPS:
            out = exact_zero_result(func, args, kwargs)
            if out is not None:
                self.skipped += 1
        if out is None:
            out = func(*args, **kwargs)
            # F07-3: non-finite values created from finite inputs (a zero-weight artefact a real model never computes):
            # NaN by any op (e.g. 0/0 normalising a zero embedding); inf only by an op reading a weight (an attention
            # mask legitimately writes -inf). Registered zero storage cannot hold either (registry_intact covers it).
            fo = [] if func.overloadpacket.__name__ in ALLOC_OPS else [   # (allocation = uninitialised memory, no value)
                o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor) and o.is_floating_point() and o.numel()
                and not is_reg(o)]
            bad = any(bool(torch.isnan(o).any()) for o in fo) or (reads_weight and any(bool(torch.isinf(o).any()) for o in fo))
            # a literal non-finite argument excuses only its own kind (masked_fill(-inf) may write inf; it never makes NaN)
            lits = [c for c in tree_flatten((args, kwargs))[0] if isinstance(c, float) and not math.isfinite(c)]
            bad = (not any(math.isnan(c) for c in lits) and any(bool(torch.isnan(o).any()) for o in fo)) or (
                reads_weight and not any(math.isinf(c) for c in lits) and any(bool(torch.isinf(o).any()) for o in fo))
            if bad and all(finite(a) for a in tree_flatten((args, kwargs))[0] if isinstance(a, torch.Tensor)):
                NONFINITE_EVENTS.append(f"{func.overloadpacket.__name__} in {mod}")
        self.used_params.update(p_read); self.used_buffers.update(b_read)
        outs = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
        idx = len(self.nodes)
        self.nodes.append(Node(str(func.overloadpacket.__name__), mod, ins, consts, len(outs),
                               tuple(outs[0].shape) if outs else None))
        for o in outs:
            self._track(o, idx)
        for t in targets:                         # in-place / out= mutation: the argument now carries this op's value
            self._track(t, idx)
        return out

    def output_nodes(self, outputs):
        return {self.producer[id(t)] for t in tree_flatten(outputs)[0]
                if isinstance(t, torch.Tensor) and id(t) in self.producer}


def layer_of(module_path):
    """(container_path, index) for the innermost repeated-block index in a module path, else None."""
    parts = module_path.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return (".".join(parts[:i]), int(parts[i]))
    return None


def analyse(rec: DagRecorder, outputs):
    """Reachability, closure, opacity, layer patterns, cross-layer edges."""
    nodes = rec.nodes
    # backward reachability from model outputs and from control sinks (values that steered Python control flow)
    roots = set(rec.output_nodes(outputs)) | {i for i, n in enumerate(nodes) if n.func in CONTROL_SINKS}
    reach, stack = set(), list(roots)
    while stack:
        i = stack.pop()
        if i in reach:
            continue
        reach.add(i)
        for e in nodes[i].ins:
            if e[0] == "op":
                stack.append(e[1])
    dead = [i for i in range(len(nodes)) if i not in reach]
    opaque = collections.Counter(n.func for n in nodes if OPAQUE.search(n.func))
    big = lambda e: (1 if not e[0] else __import__("math").prod(e[0])) > 1
    python_mediated = [e for e in rec.external if big(e) and "lift_fresh" in e[3]]      # torch.tensor(<python list>) inside forward
    ext_nonscalar = [e for e in rec.external if big(e) and "lift_fresh" not in e[3]]
    # layer signatures and cross-layer edges
    sig = collections.defaultdict(list)
    for n in nodes:
        lo = layer_of(n.module)
        if lo:
            rel = n.module[len(lo[0]) + 1 + len(str(lo[1])):]
            sig[lo].append((re.sub(r"\.\d+", ".N", rel), n.func))
    containers = collections.defaultdict(dict)
    for (c, i), s in sig.items():
        containers[c][i] = hash(tuple(s))
    patterns = {}
    for c, d in containers.items():
        order = [d[i] for i in sorted(d)]
        labels, seen = [], {}
        for h in order:
            seen.setdefault(h, chr(ord("A") + len(seen)) if len(seen) < 26 else "?")
            labels.append(seen[h])
        runs, prev, cnt = [], None, 0
        for l in labels:
            if l == prev:
                cnt += 1
            else:
                if prev: runs.append(f"{prev}x{cnt}" if cnt > 1 else prev)
                prev, cnt = l, 1
        if prev: runs.append(f"{prev}x{cnt}" if cnt > 1 else prev)
        patterns[c] = {"layers": len(order), "distinct_layer_types": len(seen), "pattern": " ".join(runs)[:300]}
    cross = collections.Counter()
    into_layers = collections.Counter()
    for n in nodes:
        lc = layer_of(n.module)
        for e in n.ins:
            if e[0] != "op":
                continue
            lp = layer_of(nodes[e[1]].module)
            if lc and lp and lc[0] == lp[0] and lc[1] - lp[1] > 1:
                cross[f"{lc[0]}: layer i -> layer i+{lc[1]-lp[1]}"] += 1
            if lc and not lp and nodes[e[1]].module != "(root)":
                into_layers[f"{nodes[e[1]].module} -> {lc[0]}.*"] += 1
    acts = sorted({n.func for n in nodes if n.func in {"silu", "gelu", "relu", "tanh", "sigmoid", "softplus", "mish", "elu",
                                                          "leaky_relu", "hardswish", "hardsigmoid", "relu6", "glu", "softmax",
                                                          "_softmax", "log_softmax", "_log_softmax", "erf", "exp"}})
    return {"ops": len(nodes), "zero_skipped": rec.skipped,
            "dead_ops": len(dead), "dead_op_funcs": dict(collections.Counter(nodes[i].func for i in dead).most_common(6)),
            "dead_op_modules": dict(collections.Counter(nodes[i].module for i in dead).most_common(4)),
            "external_nonscalar": len(ext_nonscalar),
            "python_mediated_values": len(python_mediated),
            "python_mediated_examples": [list(map(str, e)) for e in python_mediated[:3]],
            "external_examples": [list(map(str, e)) for e in ext_nonscalar[:5]],
            "opaque_ops": dict(opaque),
            "layer_patterns": patterns,
            "cross_layer_edges": dict(cross.most_common(8)),
            "non_layer_into_layers": dict(into_layers.most_common(8)),
            "activation_like_ops": acts,
            "structure_signature": hash(tuple((re.sub(r"\.\d+", ".N", n.module), n.func) for n in nodes))}


def tied_alias_modules(model, called_names):
    """Modules never called whose every parameter is the same tensor object as a parameter of a called module."""
    mods = {n or "(root)": m for n, m in model.named_modules(remove_duplicate=False)}
    live = set()
    for n in called_names:
        m = mods.get(n)
        if m is not None:
            live |= {id(p) for p in m.parameters(recurse=False)}
    out = set()
    for n, m in mods.items():
        ps = list(m.parameters(recurse=True))
        if n not in called_names and ps and all(id(p) in live for p in ps):
            out.add(n)
    return out


import contextlib as _ctx


@_ctx.contextmanager
def math_attention():
    """Force PyTorch's decomposed math attention so no fused attention kernel hides its internals."""
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.MATH]):
            yield
    except ImportError:
        yield



def clear_library_caches():
    """Clear every functools cache that belongs to transformers/diffusers code, so each recorded pass computes
    everything itself (models cache position tables / anchors across calls)."""
    import gc, functools
    n = 0
    for o in gc.get_objects():
        if isinstance(o, functools._lru_cache_wrapper):
            mod = getattr(o, "__module__", "") or getattr(getattr(o, "__wrapped__", None), "__module__", "") or ""
            if mod.startswith(("transformers", "diffusers")):
                try:
                    o.cache_clear(); n += 1
                except Exception:
                    pass
    return n
