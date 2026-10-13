"""Initialization provenance of a library load (INTENT G1; EXECUTION_PLAN P2a.2 / FULL-repair F03; dossier
model-benchmark/_reruns/p2a2_f03; rulings z-docs/alternative-approach/DECISIONS_2026-10-13_F03.md): ordered,
per-element-coverage provenance of every tensor the library writes while finalising a load.

Each tensor root carries the SET of element origins it may contain (a bitmask), either uniform for the whole root
(an int) or per element (a CPU uint8 label tensor, materialised only when a write covers part of a root of at most
LIMIT elements). Writes are applied in order:
  overwrite (fill / random fill / copy / setitem / .data =)  -> covered elements := the source's origins
  data-dependent coverage (masked_*, index_*, scatter*, put_, setitem with a tensor index, fill_diagonal_)
                                                           -> covered elements |= the source's origins (old ones stay)
  in-place arithmetic (mul_, div_, add_, clamp_, log_, ...) -> covered elements |= the operands' origins, and a
                                                              pre-window (checkpoint) value becomes 'modified'
Allocation is not initialization: empty / empty_like / new_empty / empty_strided make UNINIT elements.
A tensor that existed before the window and was never written keeps PRE (checkpoint if the loader set it,
uninitialized for a missing target, unknown when used as a source).
Final class: RANDOM if any element may be random; else UNINITIALIZED if any may be allocation garbage; else UNPROVEN if
any may come from an unknown source; else CKPT_DERIVED (copied from an UNMODIFIED loaded tensor: certified, F03-2) /
COMPUTED / DETERMINISTIC. Loaded targets: LOADED only if every element is still the untouched checkpoint value."""
import collections
import torch
from torch.overrides import TorchFunctionMode

D, C, R, X, U, P, M, K = 1, 2, 4, 8, 16, 32, 64, 128   # constant, computed, random, unknown, uninitialized, pre-window,
                                                      # pre-window value modified, derived from a LOADED (checkpoint) tensor
LIMIT = 1 << 24                                     # per-element labels only up to this many elements (else uniform union)

RANDOM_INIT = {"normal_", "uniform_", "trunc_normal_", "xavier_uniform_", "xavier_normal_", "kaiming_uniform_",
               "kaiming_normal_", "orthogonal_", "sparse_"}
DET_INIT = {"zeros_", "ones_", "constant_", "eye_", "dirac_"}
RANDOM_INPLACE = {"normal_", "uniform_", "random_", "exponential_", "geometric_", "cauchy_", "log_normal_", "bernoulli_"}
DET_INPLACE_FULL = {"zero_", "fill_"}
RANDOM_CREATE = {"randn", "rand", "randint", "randperm", "normal", "bernoulli", "multinomial", "poisson", "rand_like",
                 "randn_like", "randint_like"}
ALLOC = {"empty", "empty_like", "new_empty", "empty_strided", "new_empty_strided"}
CONST_LIKE = {"zeros_like", "ones_like", "full_like", "new_zeros", "new_ones", "new_full", "new_tensor"}
NO_VALUE = {"requires_grad_", "detach_", "retain_grad", "share_memory_", "record_stream", "pin_memory"}
META_PERMUTE = {"t_", "transpose_", "squeeze_", "unsqueeze_", "swapaxes_", "swapdims_", "as_strided_"}
DATA_DEP = {"masked_fill_", "masked_scatter_", "index_fill_", "index_copy_", "index_add_", "index_put_", "index_reduce_",
            "scatter_", "scatter_add_", "scatter_reduce_", "put_", "fill_diagonal_", "masked_fill", "index_put"}
VIEWS = ("__getitem__", "view", "narrow", "select", "detach", "view_as", "reshape", "flatten", "unflatten", "t", "transpose",
         "permute", "expand", "expand_as", "squeeze", "unsqueeze", "as_strided", "diagonal", "unfold", "split", "chunk",
         "unbind", "movedim", "swapaxes", "swapdims", "T", "mT", "real", "imag", "view_as_real", "view_as_complex")


def _root(t):
    while getattr(t, "_base", None) is not None:
        t = t._base
    return t


def _has_tensor(idx):
    if isinstance(idx, torch.Tensor):
        return True
    if isinstance(idx, (list, tuple)):
        return any(_has_tensor(i) for i in idx)
    return False


def _bits_class(b, loaded=False):
    if loaded:
        if b == P:
            return "LOADED"
        if b & R:
            return "RANDOM"
        if b & U:
            return "UNINITIALIZED"
        if b & X:
            return "UNPROVEN"
        return "LOADED_MODIFIED" if b & (P | M) else "LOADED_OVERWRITTEN"
    if b & R:
        return "RANDOM"
    if b & (U | P | M):                 # a missing target's pre-window content is the loader's empty allocation
        return "UNINITIALIZED"
    if b & X:
        return "UNPROVEN"
    if b & K:                           # certified only as an exact copy of untouched loaded values (F03-2)
        return "CKPT_DERIVED" if b == K else "UNPROVEN"
    if b & C:
        return "COMPUTED"
    return "DETERMINISTIC"


class InitTracer(TorchFunctionMode):
    def __init__(self, loaded=None):
        super().__init__()
        self.loaded = loaded or {}                       # {name: object the loader set from the checkpoint} (set_ids)
        self.loaded_ids = {id(o) for o in self.loaded.values() if o is not None}
        self.same = {}                                  # id(out-of-place identity result) -> id(root of its input)
        self.same_ver = {}                              # id(identity result) -> the root's write count when it was made
        self.keep = []
        self.made = {}                                  # id(out-of-place result) -> origin bits
        self.alias = {}                                 # id(.data alias / view without _base) -> root
        self.lab = {}                                   # id(root) -> int bits | uint8 label tensor
        self.writes = collections.defaultdict(list)    # id(root) -> [(fn, full, kind)]  (report only)

    LEGACY = ("FloatTensor", "DoubleTensor", "HalfTensor", "BFloat16Tensor", "LongTensor", "IntTensor", "ShortTensor",
              "CharTensor", "ByteTensor", "BoolTensor")

    def __enter__(self):
        """Legacy type constructors (torch.LongTensor(list)) and nn.Parameter(t) do not dispatch a torch function: wrap
        them for the window so their results carry an origin (isinstance against them keeps working)."""
        tr, self._saved = self, {}
        class _LegacyMeta(type):
            def __instancecheck__(cls, x):
                return isinstance(x, cls._orig)
            def __call__(cls, *a, **k):
                return tr._legacy(cls._orig, a, k)
        for n in self.LEGACY:
            c = getattr(torch, n, None)
            if c is not None:
                self._saved[n] = c
                setattr(torch, n, _LegacyMeta(n, (), {"_orig": c}))
        P_ = torch.nn.Parameter
        self._saved_pnew = P_.__dict__.get("__new__")
        orig_new = P_.__new__
        def pnew(cls, data=None, requires_grad=True):
            o = orig_new(cls, data, requires_grad)
            if isinstance(data, torch.Tensor):            # a re-wrap shares data's storage: the same origin
                tr.alias[id(o)] = tr._r(data); tr.keep.append(o)
            return o
        P_.__new__ = staticmethod(pnew)
        return super().__enter__()

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            for n, c in self._saved.items():
                setattr(torch, n, c)
            if self._saved_pnew is not None:
                torch.nn.Parameter.__new__ = self._saved_pnew

    def _legacy(self, orig, a, k):
        o = orig(*a, **k)
        if a and isinstance(a[0], torch.Tensor):
            self.alias[id(o)] = self._r(a[0])
        else:
            if a and type(a[0]).__module__ == "numpy":
                st = X                                    # numpy data: its origin is unseen
            elif a and isinstance(a[0], (list, tuple)):
                st = D                                    # Python values
            elif not a or o.numel() == 0:
                st = D                                    # empty
            else:
                st = U                                    # sizes only: uninitialised memory
            self.made[id(o)] = st
        self.keep.append(o)
        return o

    def _r(self, t):
        """The storage owner: follows view bases AND .data aliases (a view of a .data alias, e.g. w.data[i], is w)."""
        seen = 0
        while seen < 16:
            t2 = self.alias.get(id(t))
            t = _root(t2 if t2 is not None else t)
            if self.alias.get(id(t)) is None:
                return t
            seen += 1
        return t

    # ---- origin of a value used as a SOURCE (pre-window content is unknown provenance) ----
    def _src_bits(self, x):
        if not isinstance(x, torch.Tensor):
            return D
        r = self._r(x)
        if id(r) in self.lab:
            b = self._region_bits(r, x)
        elif id(x) in self.made:
            b = self.made[id(x)]
        elif id(r) in self.made:
            b = self.made[id(r)]
        else:
            b = P
        if b & (P | M):        # pre-window content: checkpoint-derived only from a WHOLLY untouched loaded tensor (F03-2)
            lab = self.lab.get(id(r))
            untouched = lab is None or (isinstance(lab, int) and lab == P)
            b = (b & ~(P | M)) | (K if (self._is_loaded(r) and not b & M and untouched) else X)
        return b

    def _is_loaded(self, r):
        return id(r) in self.loaded_ids

    def _identity_of(self, src, root):
        """src holds exactly root's own current values (an identity op of the whole root)."""
        if not (isinstance(src, torch.Tensor) and tuple(src.shape) == tuple(root.shape)):
            return False
        if self.same.get(id(src)) == id(root):
            return self.same_ver.get(id(src)) == len(self.writes.get(id(root), ()))   # stale after a later write
        return self._r(src) is root and src.stride() == root.stride() and src.storage_offset() == root.storage_offset()

    def _region_bits(self, root, view):
        L = self.lab[id(root)]
        if isinstance(L, int):
            return L
        v = self._label_view(L, root, view) if view.dtype == root.dtype else None
        return self._all_bits(L if v is None else v)

    def _label_view(self, L, root, view):
        try:
            return torch.as_strided(L, tuple(view.size()), tuple(view.stride()), view.storage_offset() - root.storage_offset())
        except Exception:
            return None

    def _init_bits(self, root):
        if id(root) in self.lab:
            return self.lab[id(root)]
        return self.made.get(id(root), P)

    def _apply(self, root, tgt, idx, new, mode):
        """mode: 'over' (covered := new), 'union' (covered |= new), 'arith' (covered |= new, P -> P|M)."""
        cur = self._init_bits(root)
        whole = tgt.numel() == root.numel() and tgt.dtype == root.dtype and idx is None
        if mode == "over" and whole:
            self.lab[id(root)] = new
            return
        precise = (root.numel() <= LIMIT and root.is_contiguous() and tgt.dtype == root.dtype
                   and not (idx is not None and _has_tensor(idx)))
        if not precise:                                   # coverage unknown or too large: sound union
            add = new | (M if (mode == "arith" and (cur if isinstance(cur, int) else 0) & P) else 0)
            base = cur if isinstance(cur, int) else int(self._all_bits(cur))
            self.lab[id(root)] = base | add
            return
        L = cur if isinstance(cur, torch.Tensor) else torch.full((root.numel(),), cur, dtype=torch.uint8, device="cpu")
        v = self._label_view(L, root, tgt)
        if v is None:                                     # coverage not mappable: sound union
            self.lab[id(root)] = self._all_bits(L) | new | (M if mode == "arith" else 0)
            return
        if idx is not None:
            v = v[idx]
            if v.numel() == 0:
                self.lab[id(root)] = L
                return
        if mode == "over":
            v.fill_(new)
        else:
            touched = v | new
            if mode == "arith":
                touched = torch.where((v & P) != 0, touched | M, touched)
            v.copy_(touched)
        first = L[0]
        self.lab[id(root)] = int(first) if bool((L == first).all()) else L

    def _all_bits(self, L):
        out = 0
        for b in L.unique().tolist():
            out |= int(b)
        return out

    def bits_of(self, t):
        r = self._r(t)
        v = self._init_bits(r)
        return v if isinstance(v, int) else self._all_bits(v)

    def final_class(self, t, loaded=False):
        return _bits_class(self.bits_of(t), loaded)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", str(func))
        mod = getattr(func, "__module__", "") or ""
        owner = getattr(getattr(func, "__self__", None), "__name__", "")
        if name == "__set__" and owner == "data" and len(args) >= 2 and isinstance(args[0], torch.Tensor):
            root = self._r(args[0])
            ident = self._identity_of(args[1], root)       # e.g. _adjust_bias: bias.data = pad(bias.data, (0, 0))
            src = self._src_bits(args[1]) if not ident else None
            out = func(*args, **kwargs)
            self.writes[id(root)].append(("data.__set__", True, "over"))
            if not ident:
                self.lab[id(root)] = src
            self.keep += [root, args[1]]
            return out
        if name == "__get__":
            out = func(*args, **kwargs)
            if owner == "data" and isinstance(out, torch.Tensor) and args and isinstance(args[0], torch.Tensor):
                self.alias[id(out)] = self._r(args[0]); self.keep.append(out)
            return out
        flat = list(args) + list(kwargs.values())
        tens_args = [a for a in flat if isinstance(a, torch.Tensor)]
        for a in flat:
            if isinstance(a, (list, tuple)):
                tens_args += [b for b in a if isinstance(b, torch.Tensor)]
        is_init = mod == "torch.nn.init"
        inplace = is_init or (name.endswith("_") and not name.startswith("_")) or name in ("copy_", "__setitem__")
        tgt = args[0] if args else kwargs.get("tensor", kwargs.get("self"))
        if inplace and isinstance(tgt, torch.Tensor) and name in NO_VALUE:
            return func(*args, **kwargs)
        # source origins are read BEFORE the call (a self-referencing write reads the old value)
        if inplace and isinstance(tgt, torch.Tensor):
            others = [a for a in tens_args if a is not tgt]
            src_b = 0
            for a in others:
                src_b |= self._src_bits(a)
        out = func(*args, **kwargs)
        if inplace and isinstance(tgt, torch.Tensor):
            root = self._r(tgt)
            full = tgt.numel() == root.numel()
            idx = None
            if name in RANDOM_INIT or name in RANDOM_INPLACE:
                mode, new, kind = "over", R, "random"
            elif name in DET_INIT or name in DET_INPLACE_FULL:
                mode, new, kind = "over", D, "det"
            elif is_init:
                mode, new, kind = "over", X, "unknown_init"
            elif name in META_PERMUTE:
                cur = self._init_bits(root)
                self.lab[id(root)] = cur if isinstance(cur, int) else self._all_bits(cur)
                return out
            elif name == "copy_":
                if full and self._identity_of(args[1] if len(args) > 1 else None, root):
                    self.writes[id(root)].append(("copy_", True, "identity"))
                    return out
                mode, new, kind = "over", (src_b or D), "copy"
            elif name == "__setitem__":
                idx = args[1] if len(args) > 1 else None
                val = args[2] if len(args) > 2 else kwargs.get("value")
                new = self._src_bits(val)
                mode, kind = ("union" if _has_tensor(idx) else "over"), "setitem"
                if idx is Ellipsis or (isinstance(idx, slice) and idx == slice(None)):
                    idx = None
            elif name in ("set_",):
                mode, new, kind = "over", (src_b or U), "set_"
            elif name == "resize_":
                mode, new, kind = "union", U, "resize_"
            elif name in DATA_DEP:
                mode, new, kind = ("arith" if name in ("index_add_", "scatter_add_", "scatter_reduce_", "index_reduce_")
                                   else "union"), (src_b or D), "data_dependent"
            else:
                mode, new, kind = "arith", (src_b or D), "arith"
            if new & D and mode == "arith":
                new = (new & ~D) | (D if name in ("clamp_", "clamp_min_", "clamp_max_") else C)
            if new & K and mode in ("arith", "union"):   # derived from loaded values, not a copy of them (F03-2)
                new = (new & ~K) | X
            self.writes[id(root)].append((f"{'init.' if is_init else ''}{name}", full and idx is None, kind))
            self._apply(root, tgt, idx, new, mode)
            self.keep.append(root)
            return out
        # out-of-place
        if name in ALLOC:
            st = U
        elif name in RANDOM_CREATE:
            st = R
        elif name in CONST_LIKE:
            st = D
        elif not tens_args:                                   # factory from Python values (zeros/ones/arange/tensor/...),
            st = X if any(type(a).__module__ == "numpy" for a in flat) else D   # but numpy data's origin is unseen
        else:
            st = 0
            for a in tens_args:
                st |= self._src_bits(a)
            if st & D:
                st = (st & ~D) | C
        dst = kwargs.get("out")
        if isinstance(dst, torch.Tensor):                     # torch.fn(..., out=t) overwrites t
            st_out = 0
            for a in tens_args:
                if a is not dst:
                    st_out |= self._src_bits(a)
            st_out = st if name in ALLOC or name in RANDOM_CREATE or not st_out else ((st_out & ~D) | C if st_out & D else st_out)
            if st_out & K:
                st_out = (st_out & ~K) | X
            root = self._r(dst)
            self.writes[id(root)].append((f"{name}(out=)", dst.numel() == root.numel(), "over"))
            self._apply(root, dst, None, st_out or D, "over")
            self.keep.append(root)
            return out
        outs = out if isinstance(out, (list, tuple)) else [out]
        for o in outs:
            if isinstance(o, torch.Tensor):
                self.keep.append(o)
                in_roots = {id(self._r(a)) for a in tens_args}
                # a view of an INPUT (named view op, or a _base that is an input root); a view of the op's own internal
                # result (repeat_interleave, einsum, tensordot) is a new value
                is_view = tens_args and (name in VIEWS or (getattr(o, "_base", None) is not None and id(_root(o)) in in_roots))
                if is_view and getattr(o, "_base", None) is None:
                    self.alias[id(o)] = self._r(tens_args[0])
                if not is_view:
                    src0 = tens_args[0] if tens_args else None
                    ident = src0 is not None and o.numel() == src0.numel() and (
                        name in ("clone", "contiguous", "detach", "to", "type", "float") and o.dtype == src0.dtype
                        or name == "pad" and all(int(w) == 0 for w in (args[1] if len(args) > 1 else kwargs.get("pad", (1,)))))
                    st_o = st if ident or not st & K else (st & ~K) | X      # only identities keep checkpoint-derived
                    self.made[id(o)] = st_o
                    if getattr(o, "_base", None) is not None:
                        self.made[id(_root(o))] = st_o; self.keep.append(_root(o))
                    if ident:
                        r0 = self._r(src0)
                        self.same[id(o)] = self.same.get(id(src0), id(r0))
                        self.same_ver[id(o)] = self.same_ver.get(id(src0), len(self.writes.get(id(r0), ())))
        return out
