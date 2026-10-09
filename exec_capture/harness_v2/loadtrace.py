"""How the library filled each tensor it did NOT load from the checkpoint (G1 load provenance, INTENT.md).
A TorchFunctionMode observes every torch call inside the library's own post-load phase (_finalize_model_loading: move
missing keys, init, tie) and classifies each state-dict tensor from what was written into it: deterministic init
(ones/zeros/constant), computed (a copy of a tensor computed in the window, e.g. a sinusoid table), random (an RNG fill,
also through a temporary), uninitialized (no write at all), unproven (a copy from a tensor of unknown provenance).
No per-model tables and no source parsing. Ported unchanged from the 2026-10-09 research probe, validated against real
CPU loads under two seeds (0 disagreements on 14 fixtures; model-benchmark/_reruns/g1_research)."""
import collections
import torch
from torch.overrides import TorchFunctionMode

RANDOM_INIT = {"normal_", "uniform_", "trunc_normal_", "xavier_uniform_", "xavier_normal_", "kaiming_uniform_",
               "kaiming_normal_", "orthogonal_", "sparse_"}
DET_INIT = {"zeros_", "ones_", "constant_", "eye_", "dirac_"}
RANDOM_INPLACE = {"normal_", "uniform_", "random_", "exponential_", "geometric_", "cauchy_", "log_normal_", "bernoulli_"}
DET_INPLACE_FULL = {"zero_", "fill_"}
RANDOM_CREATE = {"randn", "rand", "randint", "randperm", "normal", "bernoulli", "multinomial", "poisson", "rand_like",
                 "randn_like", "randint_like"}


def _root(t):
    while getattr(t, "_base", None) is not None:
        t = t._base
    return t


class InitTracer(TorchFunctionMode):
    """Observes every torch call during the library's post-load phase (move missing keys, init, tie)."""
    def __init__(self):
        super().__init__()
        self.keep = []                       # strong refs: ids stay unique
        self.made = {}                       # id(tensor) -> "clean" | "random"   (produced inside the window)
        self.alias = {}                      # id(view/data alias) -> root tensor
        self.writes = collections.defaultdict(list)   # id(root) -> [(fn, full, kind)]
        self.unmapped_inplace = collections.Counter()

    def _r(self, t):
        t2 = self.alias.get(id(t))
        return _root(t2 if t2 is not None else t)

    def _taint(self, x):
        """'random' if x is random-derived, 'clean' if made in-window from clean inputs, else 'external'."""
        if not isinstance(x, torch.Tensor):
            return "clean"
        s = self.made.get(id(x)) or self.made.get(id(self._r(x)))
        w = self.writes.get(id(self._r(x))) or self.writes.get(id(x)) or []
        if s == "random" or any(k == "random" for _, _, k in w):
            return "random"                             # random dominates: an in-place RNG fill of a clean temporary
        if any(k in ("external_copy", "unknown_init") for _, _, k in w):
            return "external"
        if s or w:
            return "clean"
        return "external"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", str(func))
        mod = getattr(func, "__module__", "") or ""
        if name == "__set__" and getattr(getattr(func, "__self__", None), "__name__", "") == "data" \
                and len(args) >= 2 and isinstance(args[0], torch.Tensor):
            out = func(*args, **kwargs)
            ts = self._taint(args[1])
            self.writes[id(self._r(args[0]))].append(("data.__set__", True, {"random": "random", "clean": "computed", "external": "external_copy"}[ts]))
            self.keep.append(self._r(args[0]))
            return out
        if name == "__get__":                          # property access, e.g. Tensor.data
            pname = getattr(getattr(func, "__self__", None), "__name__", "")
            out = func(*args, **kwargs)
            if pname == "data" and isinstance(out, torch.Tensor) and args and isinstance(args[0], torch.Tensor):
                self.alias[id(out)] = self._r(args[0]); self.keep.append(out)
            return out
        tens_args = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)]
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, (list, tuple)):
                tens_args += [b for b in a if isinstance(b, torch.Tensor)]
        is_init = mod == "torch.nn.init"
        inplace = is_init or (name.endswith("_") and not name.startswith("_")) or name in ("copy_", "__setitem__")
        out = func(*args, **kwargs)
        tgt = args[0] if args else kwargs.get("tensor", kwargs.get("self"))   # torch.nn.init dispatches by keyword
        if inplace and isinstance(tgt, torch.Tensor):
            root = self._r(tgt)
            full = tgt.numel() == root.numel()
            if is_init or name in RANDOM_INIT or name in DET_INIT:     # init fns also arrive re-wrapped (module != torch.nn.init)
                kind = "random" if name in RANDOM_INIT else "det" if name in DET_INIT else "unknown_init"
            elif name in RANDOM_INPLACE:
                kind = "random"
            elif name in DET_INPLACE_FULL:
                kind = "det"
            elif name in ("copy_", "__setitem__"):
                if name == "__setitem__":
                    src = args[2] if len(args) > 2 else kwargs.get("value")
                    idx = args[1] if len(args) > 1 else None
                    full = full and (idx is Ellipsis or (isinstance(idx, slice) and idx == slice(None)))
                else:
                    src = args[1] if len(args) > 1 else kwargs.get("src", kwargs.get("value", kwargs.get("other")))
                ts = self._taint(src)
                kind = {"random": "random", "clean": "computed", "external": "external_copy"}[ts]
            else:                                       # arithmetic in place: inherits its inputs' taint
                ts = {self._taint(a) for a in tens_args[1:]}
                kind = "random" if "random" in ts else "external_copy" if "external" in ts else "inplace_clean"
            self.writes[id(root)].append((f"{mod.split('.')[-1] + '.' if is_init else ''}{name}", full, kind))
            self.keep.append(root)
            return out
        # out-of-place: outputs get the taint of their inputs (random creation ops are random)
        ts = {self._taint(a) for a in tens_args}
        st = "random" if (name in RANDOM_CREATE or "random" in ts) else "clean" if "external" not in ts else None
        outs = out if isinstance(out, (list, tuple)) else [out]
        for o in outs:
            if isinstance(o, torch.Tensor):
                self.keep.append(o)
                if st:
                    self.made[id(o)] = st
                elif tens_args and getattr(o, "_base", None) is not None:
                    pass                                # a view of an external tensor: resolved through _base
                if getattr(o, "_base", None) is None and tens_args and name in ("__getitem__", "view", "narrow",
                                                                                 "select", "detach", "view_as"):
                    self.alias[id(o)] = self._r(tens_args[0])
        return out


def classify(writes):
    """Final state of one tensor from its ordered writes inside the post-load window."""
    if not writes:
        return "UNINITIALIZED", []
    state, fns = None, []
    for fn, full, kind in writes:
        fns.append(fn)
        if kind == "random":
            state = "RANDOM"                            # random even if partial: values depend on the RNG
        elif kind == "inplace_clean" and state is not None:
            pass                                        # x.mul_(c) keeps whatever x was (random stays random)
        elif kind in ("det", "computed", "inplace_clean"):
            if full:
                state = "DETERMINISTIC" if kind != "computed" else "COMPUTED"
            elif state is None:
                state = "PARTIAL_ONLY"
        elif kind in ("external_copy", "unknown_init"):
            if full or state is None:
                state = "UNPROVEN"
    if state is None or state == "PARTIAL_ONLY":
        state = "UNINITIALIZED"
    return state, fns
