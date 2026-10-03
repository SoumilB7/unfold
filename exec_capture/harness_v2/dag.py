"""Full dataflow recorder for one real forward pass (zero-storage weights + exact zero-skip).

Every tensor is mapped to the op that produced it; every op records its input edges
(op -> op, param, buffer, model input, or EXTERNAL = appeared from outside the recording),
the module it ran inside, and its scalar constants. Nothing is inferred from names or source text.
"""
import re, math, weakref, collections
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map
from lowcost import ZERO_PTRS, HEAVY, BIAS_FIRST

OPAQUE = re.compile(r"(scaled_dot_product|flash|_efficient_attention|cudnn|_fused|fused_|mkldnn_rnn|_cufft)")
CONTROL_SINKS = {"_local_scalar_dense", "is_nonzero", "equal", "item"}


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
        self.zero_act = set()        # storage pointers of activations known to be exactly zero (outputs of skipped ops)
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
                 "sqrt_", "fill_", "masked_fill_"}

    def _zero_preserving_inplace(self, func, args, kwargs):
        name = func.overloadpacket.__name__
        if name not in self.ZERO_KEEP or not args or not isinstance(args[0], torch.Tensor) or args[0].numel() <= 1:
            return False
        if args[0].untyped_storage().data_ptr() not in ZERO_PTRS:
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
        ins, consts, zero_operand = [], [], False
        for a in tree_flatten((args, kwargs))[0]:
            if isinstance(a, torch.Tensor):
                k = id(a)
                if k in self.producer:
                    ins.append(("op", self.producer[k]))
                elif k in self.param_names:
                    ins.append(("param", self.param_names[k])); self.used_params.add(self.param_names[k])
                elif k in self.buffer_names:
                    ins.append(("buffer", self.buffer_names[k])); self.used_buffers.add(self.buffer_names[k])
                elif k in self.input_names:
                    ins.append(("input", self.input_names[k])); self.used_inputs.add(self.input_names[k])
                elif k in self.const_names:
                    ins.append(("module_constant", self.const_names[k]))
                else:
                    ins.append(("external", tuple(a.shape), str(a.dtype)))
                    self.external.append((tuple(a.shape), str(a.dtype), mod, str(func)))
                if a.numel() > 1:
                    sp = a.untyped_storage().data_ptr()
                    if sp in ZERO_PTRS or sp in self.zero_act:
                        zero_operand = True
            elif isinstance(a, (int, float, bool)) and not isinstance(a, bool):
                consts.append(a)
        if self.skip and not zero_operand and func in HEAVY:
            # value check: an activation operand that is entirely zero makes the result exactly zero,
            # provided every other tensor operand is finite (0 * inf would really be NaN)
            ts = [a for a in tree_flatten((args, kwargs))[0] if isinstance(a, torch.Tensor) and a.is_floating_point()]
            zs = [a for a in ts if a.numel() > 4096 and not bool(torch.any(a))]
            if zs and all(bool(torch.isfinite(a).all()) for a in ts if all(a is not z for z in zs)):
                zero_operand = True
        if self.skip and zero_operand and func in HEAVY:
            self.skipped += 1
            m = tree_map(lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x, (args, kwargs))
            try:
                meta_out = func(*m[0], **m[1])
            except RuntimeError:
                # some kernels (e.g. grouped expert matmul) only accept bf16 even for shape inference:
                # infer the shape in bf16, return exact zeros in the original dtype
                orig = next((a.dtype for a in tree_flatten(args)[0] if isinstance(a, torch.Tensor) and a.is_floating_point()), torch.float32)
                mb = tree_map(lambda x: x.to("meta", torch.bfloat16) if isinstance(x, torch.Tensor) and x.is_floating_point() else
                              (x.to("meta") if isinstance(x, torch.Tensor) else x), (args, kwargs))
                meta_out = tree_map(lambda o: o.to(orig) if isinstance(o, torch.Tensor) and o.is_floating_point() else o, func(*mb[0], **mb[1]))
            first_zero = args[0].numel() > 1 and (args[0].untyped_storage().data_ptr() in ZERO_PTRS or args[0].untyped_storage().data_ptr() in self.zero_act)
            if func in BIAS_FIRST and not first_zero:
                out = (args[0] * kwargs.get("beta", 1)).expand(meta_out.shape).contiguous()
            else:
                out = tree_map(lambda o: torch.zeros(o.shape, dtype=o.dtype) if isinstance(o, torch.Tensor) else o, meta_out)
                for o in tree_flatten(out)[0]:
                    if isinstance(o, torch.Tensor) and o.numel() > 1:
                        p = o.untyped_storage().data_ptr(); self.zero_act.add(p)
                        try:
                            weakref.finalize(o, self.zero_act.discard, p)
                        except TypeError:
                            pass
        elif self._zero_preserving_inplace(func, args, kwargs):
            # an in-place op on zero-storage weights whose result is provably still zero (e.g. RWKV rescales weights
            # with div_ at inference): skip the write (the storage is shared and read-only), keep the op in the graph
            self.skipped += 1
            out = args[0]
        else:
            out = func(*args, **kwargs)
        outs = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
        idx = len(self.nodes)
        self.nodes.append(Node(str(func.overloadpacket.__name__), mod, ins, consts, len(outs),
                               tuple(outs[0].shape) if outs else None))
        for o in outs:
            self._track(o, idx)
        # in-place / out= mutation: the mutated argument now carries this op's value
        try:
            for i, sa in enumerate(func._schema.arguments):
                if sa.alias_info is not None and sa.alias_info.is_write:
                    tgt = args[i] if i < len(args) else kwargs.get(sa.name)
                    if isinstance(tgt, torch.Tensor):
                        self._track(tgt, idx)
                        self.zero_act.discard(tgt.untyped_storage().data_ptr())      # written in place: no longer known-zero
        except Exception:
            pass
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
