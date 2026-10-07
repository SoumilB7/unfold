"""Low-cost version of the winning mechanism.
- zero-storage weights (registry of zero storages); small tensors are real zeros (deterministic)
- HEAVY / BIAS_FIRST: the op sets dag.DagRecorder zero-skips (a heavy op with a zero-weight operand returns its exact
  result, zeros or the bias, without arithmetic; the op and the weights it touched are still recorded)
- checkpoint headers (safetensors in parallel; legacy .bin through a restricted unpickler)"""
import contextlib, torch, concurrent.futures as cf
aten = torch.ops.aten
ZERO_PTRS = set()
_SHARED = {}


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
            if len(shape) >= 3:
                key = (int(shape[-2]), int(shape[-1]), dt)
                if key not in _SHARED:
                    _SHARED[key] = torch.zeros(key[0], key[1], dtype=dt); ZERO_PTRS.add(_SHARED[key].untyped_storage().data_ptr())
                return _SHARED[key].expand(*shape)
            base = torch.zeros((), dtype=dt); ZERO_PTRS.add(base.untyped_storage().data_ptr())
            return base.expand(*shape)
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
            if n in ("uniform_", "normal_") or zv(self): return self
            return o(self, *a, **k)
        return f
    try:
        import transformers.initialization as ti
        ti_saved = dict(ti.TORCH_INIT_FUNCTIONS)
        for k in ti_saved: ti.TORCH_INIT_FUNCTIONS[k] = lambda t, *a, **kk: t
    except Exception:
        ti, ti_saved = None, {}
    torch.empty = fake_empty
    for n in init_saved: setattr(torch.nn.init, n, lambda t, *a, **k: t)
    for n in tsaved: setattr(torch.Tensor, n, guard(n))
    try:
        yield
    finally:
        torch.empty = real_empty
        for n, f in init_saved.items(): setattr(torch.nn.init, n, f)
        for n, f in tsaved.items(): setattr(torch.Tensor, n, f)
        if ti is not None: ti.TORCH_INIT_FUNCTIONS.update(ti_saved)


HEAVY = {aten.mm.default, aten.addmm.default, aten.bmm.default, aten.baddbmm.default, aten.embedding.default,
         aten.convolution.default, aten._grouped_mm.default, aten.matmul.default, aten.linear.default}
BIAS_FIRST = {aten.addmm.default, aten.baddbmm.default}


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
        with fs.open(f"{repo}/{f}", "rb", block_size=1 << 20) as fh:
            return _bin_index(fh)
    with cf.ThreadPoolExecutor(workers) as ex:
        for d in ex.map(one, files):
            T.update(d)
    return T, len(files)
