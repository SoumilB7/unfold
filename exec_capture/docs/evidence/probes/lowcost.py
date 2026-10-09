"""Low-cost version of the winning mechanism.
- zero-storage weights (registry of zero storages); small tensors are real zeros (deterministic)
- ZeroSkip: any heavy op with a zero-weight operand returns its exact result (zeros, or the bias) without arithmetic;
  the op and the weights it touched are still recorded
- headers fetched in parallel, concurrently with config"""
import contextlib, torch, concurrent.futures as cf
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map
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


class Recorder(TorchDispatchMode):
    """Records every op and every parameter it touches; skips arithmetic that is known to be exactly zero."""
    def __init__(self, ids, skip=True):
        super().__init__(); self.ids = ids; self.used = set(); self.ops = 0; self.skipped = 0; self.skip = skip
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat = tree_flatten((args, kwargs))[0]
        zero_operand = False
        for a in flat:
            if isinstance(a, torch.Tensor):
                n = self.ids.get(id(a))
                if n is not None: self.used.add(n)
                if a.numel() > 1 and a.untyped_storage().data_ptr() in ZERO_PTRS: zero_operand = True
        self.ops += 1
        if self.skip and zero_operand and func in HEAVY:
            self.skipped += 1
            m = tree_map(lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x, (args, kwargs))
            out = func(*m[0], **m[1])
            if func in BIAS_FIRST and not (args[0].numel() > 1 and args[0].untyped_storage().data_ptr() in ZERO_PTRS):
                beta = kwargs.get("beta", 1)
                return (args[0] * beta).expand(out.shape).contiguous()
            return tree_map(lambda o: torch.zeros(o.shape, dtype=o.dtype) if isinstance(o, torch.Tensor) else o, out)
        return func(*args, **kwargs)


def fetch_headers(repo, workers=48):
    """All safetensors headers of a repo, fetched in parallel (range requests; nothing cached on disk)."""
    import json
    from huggingface_hub import HfApi
    api = HfApi()
    files = [f for f in api.list_repo_files(repo) if f.endswith(".safetensors") and "/" not in f]
    T = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for md in ex.map(lambda f: api.parse_safetensors_file_metadata(repo, f), files):
            for n, i in md.tensors.items(): T[n] = (tuple(i.shape), i.dtype)
    return T, len(files)


def fetch_cached(repo, cache_dir="hdrcache"):
    """Config + header table cached per repo revision (tiny JSON). Returns (config_dict, tensors, source)."""
    import os, json
    from huggingface_hub import HfApi, hf_hub_download
    os.makedirs(cache_dir, exist_ok=True)
    sha = HfApi().model_info(repo).sha
    p = os.path.join(cache_dir, repo.replace("/", "__") + f"@{sha[:12]}.json")
    if os.path.exists(p):
        d = json.load(open(p)); return d["config"], {k: (tuple(v[0]), v[1]) for k, v in d["tensors"].items()}, "cache"
    with cf.ThreadPoolExecutor(2) as ex:
        fc = ex.submit(lambda: json.load(open(hf_hub_download(repo, "config.json", revision=sha))))
        fh = ex.submit(fetch_headers, repo)
        cfgd = fc.result(); T, _ = fh.result()
    json.dump({"config": cfgd, "tensors": {k: [list(s), d] for k, (s, d) in T.items()}}, open(p, "w"))
    return cfgd, T, "network"
