"""Low-cost version of the winning mechanism.
- zero-storage weights (registry of zero storages); small tensors are real zeros (deterministic)
- the zero-skip itself (dag.exact_zero_result) returns an op's exact result without arithmetic when a zero operand
  provably zeroes it; every registered zero storage is kept alive (ZERO_REG) so its address is never reused (F07)
- checkpoint headers (safetensors in parallel; legacy .bin through a restricted unpickler)"""
import contextlib, weakref, math, torch, concurrent.futures as cf
aten = torch.ops.aten
ZERO_PTRS = set()
ZERO_REG = {}            # storage data_ptr -> the registered zero tensor itself: kept alive, so a freed address can never
                         # be reused by a live nonzero tensor and mistaken for a known zero (F07)
_SHARED = {}
_HANDED = {}             # id -> weakref of each tensor object zero_storage handed out (identity, never tensor ==)
BUILD_UNAPPLIED = []     # build-time writes into zero storage that could not be applied (a G2 unknown, F07-2)
BUILD_ALIAS_WRITES = []  # build-time writes through an alias / view of zero storage left as zeros (learned weights)
MATERIALISE_MAX = 1 << 24


def _register(t):
    p = t.untyped_storage().data_ptr()
    ZERO_PTRS.add(p)
    ZERO_REG.setdefault(p, t)


BUILD_PARAMS_RESET = []  # learned parameters the build left non-finite, reset to zero (F07-3)


def normalise_learned(model):
    """The method executes learned parameters as zeros (the checkpoint supplies their values): a parameter an init left
    non-finite on zero storage (e.g. A_log = log(0) = -inf) is reset to zero and recorded (F07-3). Returns the model."""
    for n, p in model.named_parameters(remove_duplicate=False):
        if ((p.is_floating_point() or p.is_complex()) and p.numel() and p.untyped_storage().data_ptr() not in ZERO_REG
                and not bool(torch.isfinite(p).all())):
            with torch.no_grad():
                torch.Tensor.zero_(p)
            BUILD_PARAMS_RESET.append(n)
    return model


def _zero_view(shape, dt):
    """A registered all-zero tensor of this shape / dtype without its memory: >= 3-D shapes share a 2-D zero buffer
    (real inner strides for kernels that need them), smaller ones expand a 0-d zero."""
    shape = tuple(int(s) for s in shape)
    if len(shape) >= 3:
        key = (shape[-2], shape[-1], dt)
        if key not in _SHARED:
            _SHARED[key] = torch.zeros(key[0], key[1], dtype=dt); _register(_SHARED[key])
        return _SHARED[key].expand(*shape)
    base = torch.zeros((), dtype=dt); _register(base)
    return base.expand(*shape)


def uniform_float32(model):
    """The executed build is one dtype, float32, even where a sub-config declares another (a composite model builds
    sub-models in their sub-config dtype, so a bf16 head would meet float32 activations): zero-storage tensors become
    float32 zero views (no memory), small real tensors are cast. Tied tensors (one object) are converted once."""
    for mod in model.modules():
        for d in (mod._parameters, mod._buffers):
            for k, t in list(d.items()):
                if t is None or not t.is_floating_point() or t.dtype == torch.float32:
                    continue
                if t.numel() > 1 and t.untyped_storage().data_ptr() in ZERO_REG:
                    new = _zero_view(t.shape, torch.float32)
                else:
                    new = t.detach().to(torch.float32)
                if isinstance(t, torch.nn.Parameter):
                    t.data = new
                else:
                    d[k] = new
    return model


def registry_intact():
    """Every registered zero storage still holds only zeros (a write that bypassed the recorder, e.g. through numpy,
    would make every tensor sharing it wrong)."""
    return all(not bool(torch.any(t)) for t in ZERO_REG.values())


def _keeps_zero(n, a, k):
    """A build-time in-place write that leaves an all-zero tensor all-zero (so the no-op is exact)."""
    def val(x):
        return x.item() if isinstance(x, torch.Tensor) and x.numel() == 1 else x
    if n in ("zero_", "erfinv_", "sqrt_"):
        return True
    if n == "fill_":
        return bool(a) and not isinstance(val(a[0]), torch.Tensor) and val(a[0]) == 0
    if n in ("masked_fill_", "index_fill_"):
        return bool(a) and not isinstance(val(a[-1]), torch.Tensor) and val(a[-1]) == 0
    if n == "__setitem__":
        return len(a) > 1 and not isinstance(val(a[1]), torch.Tensor) and val(a[1]) == 0
    if n in ("mul_", "div_"):
        x = val(a[0]) if a else None
        if isinstance(x, torch.Tensor) or x is None:
            return False
        return math.isfinite(x) and (n == "mul_" or x != 0)
    if n == "clamp_":
        lo, hi = (val(a[0]) if a else k.get("min")), (val(a[1]) if len(a) > 1 else k.get("max"))
        if isinstance(lo, torch.Tensor) or isinstance(hi, torch.Tensor):
            return False
        return (lo is None or lo <= 0) and (hi is None or hi >= 0)
    return False


@contextlib.contextmanager
def zero_storage(min_numel=4096):
    real_empty = torch.empty
    def fake_empty(*size, **kw):
        shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
        n = 1
        for s in shape: n *= int(s)
        dt = kw.get("dtype") or torch.get_default_dtype()
        if kw.get("device") not in (None, "cpu", torch.device("cpu")) or kw.get("out") is not None:
            return real_empty(*size, **kw)
        if n >= min_numel:
            t = _zero_view(shape, dt); _HANDED[id(t)] = weakref.ref(t)
            return t
        return torch.zeros(*shape, dtype=dt)              # small tensors: real zeros, deterministic
    names = ["uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_",
             "xavier_normal_", "orthogonal_", "sparse_", "constant_", "ones_", "zeros_", "eye_", "dirac_"]
    init_saved = {n: getattr(torch.nn.init, n) for n in names if hasattr(torch.nn.init, n)}
    tsaved = {n: getattr(torch.Tensor, n) for n in ("uniform_", "normal_", "copy_", "fill_", "zero_", "mul_", "add_", "sub_",
                                                   "div_", "clamp_", "erfinv_", "exp_", "log_", "sqrt_", "masked_fill_",
                                                   "index_fill_", "scatter_", "__setitem__")}
    def zv(t): return t.dim() > 0 and t.numel() > 1 and t.untyped_storage().data_ptr() in ZERO_PTRS
    def guard(n):
        o = tsaved[n]
        def f(self, *a, **k):
            if n in ("uniform_", "normal_"):              # random init: weights are zero by design, so a tensor the
                if not zv(self) and self.numel():         # legacy constructors left uninitialised is zeroed
                    with torch.no_grad():
                        tsaved["zero_"](self)
                return self
            if not zv(self):
                return o(self, *a, **k)
            if isinstance(self, torch.nn.Parameter) or _keeps_zero(n, a, k):
                return self                               # learned weights stay zero (the checkpoint supplies them)
            r_ = _HANDED.get(id(self))
            if r_ is not None and r_() is self:           # a computed buffer / temporary: give it real storage (F07-2)
                if self.numel() > MATERIALISE_MAX:
                    BUILD_UNAPPLIED.append(f"{n} into a {tuple(self.shape)} zero-storage tensor (over the size cap)")
                    return self
                torch.Tensor.set_(self, torch.zeros(self.shape, dtype=self.dtype))
                return o(self, *a, **k)
            BUILD_ALIAS_WRITES.append((f"{n} through an alias of a {tuple(self.shape)} zero-storage tensor",
                                       self.untyped_storage().data_ptr()))
            return self                                   # (an alias of a learned weight: stays zero by design)
        return f
    try:
        import transformers.initialization as ti
        ti_saved = dict(ti.TORCH_INIT_FUNCTIONS)
        _RND = ("uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_",
                "xavier_normal_", "orthogonal_", "sparse_")
        def _ti_noop(k):
            def f(t, *a, **kk):
                if k in _RND and isinstance(t, torch.Tensor) and t.numel() and not (
                        t.dim() > 0 and t.numel() > 1 and t.untyped_storage().data_ptr() in ZERO_PTRS):
                    with torch.no_grad():
                        torch.Tensor.zero_(t)
                return t
            return f
        for k in ti_saved: ti.TORCH_INIT_FUNCTIONS[k] = _ti_noop(k)
    except Exception:
        ti, ti_saved = None, {}
    torch.empty = fake_empty
    RANDOM_INIT = {"uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_",
                   "xavier_normal_", "orthogonal_", "sparse_"}
    def init_noop(name):
        def f(t, *a, **k):                                # weights are zero by design: a random init zeroes a tensor
            if name in RANDOM_INIT and isinstance(t, torch.Tensor) and not zv(t) and t.numel():
                with torch.no_grad():
                    tsaved["zero_"](t)                    # (legacy constructors leave uninitialised memory)
            return t
        return f
    for n in init_saved: setattr(torch.nn.init, n, init_noop(n))
    for n in tsaved: setattr(torch.Tensor, n, guard(n))
    try:
        yield
    finally:
        torch.empty = real_empty
        for n, f in init_saved.items(): setattr(torch.nn.init, n, f)
        for n, f in tsaved.items(): setattr(torch.Tensor, n, f)
        if ti is not None: ti.TORCH_INIT_FUNCTIONS.update(ti_saved)




def fetch_headers(repo, workers=8):
    """All safetensors headers of a repo, fetched in parallel (range requests; nothing cached on disk)."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = [f for f in api.list_repo_files(repo) if f.endswith(".safetensors") and "/" not in f]
    T = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for md in ex.map(lambda f: api.parse_safetensors_file_metadata(repo, f), files):
            for n, i in md.tensors.items(): T[n] = (tuple(i.shape), i.dtype)
    return T, len(files)


# ---------------------------------------------------------------------------------------------------------------
# Ground truth for repos that ship only PyTorch .bin files: read the tensor index (names, shapes, dtypes) from the
# pickle inside the zip archive with a RESTRICTED unpickler (only tensor-rebuild functions allowed; nothing else can
# run), fetching only the needed byte ranges over HTTP.
_STORAGE_DTYPE = {"FloatStorage": "F32", "HalfStorage": "F16", "BFloat16Storage": "BF16", "DoubleStorage": "F64",
                  "LongStorage": "I64", "IntStorage": "I32", "ShortStorage": "I16", "CharStorage": "I8",
                  "ByteStorage": "U8", "BoolStorage": "BOOL", "UntypedStorage": "?"}


def _bin_index(fileobj):
    import pickle, zipfile, collections as _c
    class _Storage:
        def __init__(self, name): self.name = name
    class _T:
        __slots__ = ("shape", "dtype")
        def __init__(self, shape, dtype): self.shape, self.dtype = tuple(shape), dtype
    def rebuild(storage, offset, size, stride, *a, **k):
        return _T(size, storage)
    def rebuild_param(data, *a, **k):
        return data
    def rebuild_from_type_v2(func, new_type, args, state):
        # func was itself resolved through find_class, so it can only be one of the allowed rebuild functions
        if func not in (rebuild, rebuild_param):
            raise pickle.UnpicklingError("blocked rebuild function")
        return func(*args)
    allowed = {("torch._tensor", "_rebuild_from_type_v2"): rebuild_from_type_v2, ("torch._utils", "_rebuild_from_type_v2"): rebuild_from_type_v2,
               ("torch", "Tensor"): object, ("torch.nn.parameter", "Parameter"): object,
               ("torch._utils", "_rebuild_tensor_v2"): rebuild, ("torch._utils", "_rebuild_tensor"): rebuild,
               ("torch._utils", "_rebuild_parameter"): rebuild_param, ("collections", "OrderedDict"): _c.OrderedDict,
               ("torch", "Size"): tuple}
    class RU(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in allowed:
                return allowed[(module, name)]
            if module == "torch" and name.endswith("Storage"):
                return _Storage(name)
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        def persistent_load(self, pid):
            st = pid[1] if isinstance(pid, tuple) and len(pid) > 1 else None
            return _STORAGE_DTYPE.get(getattr(st, "name", ""), "?")
    try:
        zf = zipfile.ZipFile(fileobj)
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        member = zf.open(pkl)
        member._expected_crc = None          # torch's own zip reader does not verify CRCs (old writers stored bad ones)
        obj = RU(member).load()
    except zipfile.BadZipFile:
        # legacy (pre-zip) format: pickled magic number, protocol, sys_info, then the object itself
        fileobj.seek(0)
        u = RU(fileobj)
        for _ in range(3):
            u.load()
        obj = u.load()
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    return {k: (v.shape, v.dtype) for k, v in obj.items() if isinstance(v, _T)}


def fetch_bin_headers(repo, workers=4):
    from huggingface_hub import HfApi, HfFileSystem
    files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".bin") and "/" not in f
             and ("pytorch_model" in f or f.startswith("model"))]
    fs = HfFileSystem()
    T = {}
    def one(f):
        import noweights
        with fs.open(f"{noweights.ref(repo)}/{f}", "rb", block_size=1 << 20) as fh:     # the pinned revision
            return _bin_index(fh)
    with cf.ThreadPoolExecutor(workers) as ex:
        for d in ex.map(one, files):
            T.update(d)
    return T, len(files)
