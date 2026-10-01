"""STRICT capture check for one diffusers pipeline repo. Usage: worker_v2_diffusion.py <repo> <out.json>
Same verdict ladder and checks as worker_v2.py, applied per weighted component; the repo is FULL only if every
weighted component is FULL. The pipeline's own __call__ runs for real (zero-storage weights, exact zero-skip);
entry points the generation path never uses (e.g. VAE.encode for text-to-image) get their own pass."""
import sys, os, json, re, time, inspect, resource, traceback, collections, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error"); os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import noweights; noweights.install()
torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "2")))
import transformers, diffusers
from huggingface_hub import HfApi, hf_hub_download
from lowcost import zero_storage
from common import ckpt_split, match, library_rename, split_library_ignored, looks_like_stored_buffer, top_groups, numel
from dag import DagRecorder, analyse, tied_alias_modules, math_attention, clear_library_caches

repo, out_path = sys.argv[1], sys.argv[2]
HDR_CACHE = os.environ.get("BENCH_HEADER_CACHE", os.path.join(HERE, "..", "cache_v2", "headers"))
R = {"repo": repo, "diffusers": diffusers.__version__, "transformers": transformers.__version__, "components": {}}
T0 = time.time()


def _last_resort(etype, e, tb):
    import traceback as _tb
    R["verdict"] = R.get("verdict") or "FAIL"
    R["reason"] = R.get("reason") or f"unhandled {etype.__name__}: {str(e)[:250]}"
    fr = _tb.extract_tb(tb)
    R["where"] = R.get("where") or (f"{fr[-1].filename.split('site-packages/')[-1]}:{fr[-1].lineno}" if fr else "")
    try: save()
    except Exception: pass


sys.excepthook = _last_resort; api = HfApi()


def save():
    R["secs"] = round(time.time() - T0, 1)
    R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(R, open(out_path, "w"), indent=1, default=str)


def fail(v, why, where=""):
    R["verdict"] = v; R["reason"] = why[:400]; R["where"] = where; save(); sys.exit(0)


def resolve(lib, cname):
    """The class a model_index entry names, found the way diffusers' own loader finds it: the library itself, then the
    pipeline package diffusers.pipelines.<lib>, then its deprecated home, then that package's submodules."""
    import importlib, pkgutil
    for mod in ((diffusers,) if lib == "diffusers" else (transformers,) if lib == "transformers" else ()):
        if hasattr(mod, cname): return getattr(mod, cname)
        if cname.endswith("FeatureExtractor") and hasattr(mod, cname[: -len("FeatureExtractor")] + "ImageProcessor"):
            return getattr(mod, cname[: -len("FeatureExtractor")] + "ImageProcessor")      # renamed in transformers 5
    for path in (f"diffusers.pipelines.{lib}", f"diffusers.pipelines.deprecated.{lib}"):
        try:
            pkg = importlib.import_module(path)
        except Exception:
            continue
        if hasattr(pkg, cname):
            return getattr(pkg, cname)
        for sub in pkgutil.iter_modules(getattr(pkg, "__path__", [])):
            try:
                sm = importlib.import_module(f"{path}.{sub.name}")
            except Exception:
                continue
            if hasattr(sm, cname):
                return getattr(sm, cname)
    return None


try:
    idx = json.load(open(hf_hub_download(repo, "model_index.json")))
    sha = api.model_info(repo).sha
    files = api.list_repo_files(repo)
except Exception as e:
    s = f"{type(e).__name__}: {e}"
    fail("OUT" if ("gated" in s.lower() or "403" in s or "404" in s) else "FAIL", ("gated or missing: " if "403" in s or "404" in s else "") + s[:300])
pcls = getattr(diffusers, idx.get("_class_name", ""), None)
if pcls is None:
    fail("OUT", f"pipeline class not in installed diffusers: {idx.get('_class_name')}")
R["pipeline"] = pcls.__name__


def headers(sub):
    os.makedirs(HDR_CACHE, exist_ok=True)
    hp = os.path.join(HDR_CACHE, repo.replace("/", "__") + f"__{sub}@{sha[:12]}.json")
    if os.path.exists(hp):
        cached = json.load(open(hp))
        if cached:                                  # an empty table is never trusted as ground truth
            return {k: (tuple(v[0]), v[1]) for k, v in cached.items()}
    at = (lambda f: f.count("/") == 0) if sub == "" else (lambda f: f.startswith(sub + "/") and f.count("/") == 1)
    fs = [f for f in files if at(f) and f.endswith(".safetensors")]
    groups = collections.defaultdict(list)
    for f in fs:
        b = re.sub(r"-\d{5}-of-\d{5}$", "", f.split("/")[-1][: -len(".safetensors")])
        groups[b.split(".")[1] if "." in b else ""].append(f)
    pick = next((groups[p] for p in ("", "fp32", "bf16", "fp16") if p in groups), next(iter(groups.values()), []))
    T = {}
    for f in pick:
        for n, i in api.parse_safetensors_file_metadata(repo, f).tensors.items():
            T[n] = (tuple(i.shape), i.dtype)
    if not T:        # legacy .bin component weights: read the tensor index with the restricted unpickler (no code run)
        from lowcost import _bin_index
        from huggingface_hub import HfFileSystem
        fsys = HfFileSystem()
        bins = [f for f in files if at(f) and f.endswith(".bin") and "training_args" not in f]
        full = [f for f in bins if ".fp16." not in f] or bins
        for f in full:
            with fsys.open(f"{repo}/{f}", "rb", block_size=1 << 20) as fh:
                T.update(_bin_index(fh))
    if T:
        json.dump({k: [list(s), d] for k, (s, d) in T.items()}, open(hp, "w"))
    return T


objs, models = {}, {}
for name, spec in idx.items():
    if not (isinstance(spec, list) and len(spec) == 2) or spec[0] is None or spec[1] is None:
        continue
    lib, cname = spec
    C = R["components"].setdefault(name, {"class": f"{lib}.{cname}"})
    cls = resolve(lib, cname)
    sub = name if any(f.startswith(name + "/") for f in files) else None
    if sub is None:
        C["from_repo_root"] = True                   # old single-folder layout
    try:
        if cls is not None and inspect.isclass(cls) and issubclass(cls, diffusers.ModelMixin):
            with zero_storage():
                m = cls.from_config(cls.load_config(repo, subfolder=sub))
            models[name] = m.float().eval()
        elif cls is not None and inspect.isclass(cls) and issubclass(cls, transformers.PreTrainedModel):
            # the class the index names, with ITS config type (diffusers loads getattr(lib, class).from_pretrained)
            cc = getattr(cls, "config_class", None)
            try:
                cfg = cc.from_pretrained(repo, subfolder=sub) if cc is not None else transformers.AutoConfig.from_pretrained(repo, subfolder=sub)
            except Exception:
                cfg = transformers.AutoConfig.from_pretrained(repo, subfolder=sub)
            with zero_storage():
                try:
                    m = cls._from_config(cfg, attn_implementation="eager", dtype=torch.float32)
                except (TypeError, ValueError):
                    m = cls._from_config(cfg, dtype=torch.float32)
            models[name] = m.eval()
        elif cls is not None:
            objs[name] = cls.from_pretrained(repo, subfolder=sub)
        else:
            for auto in (transformers.AutoImageProcessor, transformers.AutoTokenizer, transformers.AutoProcessor):
                try:
                    objs[name] = auto.from_pretrained(repo, subfolder=sub); break
                except Exception:
                    pass
            if name not in objs:
                C["error"] = f"class {lib}.{cname} not in installed libraries"
    except Exception as e:
        C["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    if name in models:
        try:
            C["headers"] = len(headers(name if sub else ""))
        except Exception as e:
            C["headers_error"] = f"{type(e).__name__}: {str(e)[:150]}"
save()
container = torch.nn.ModuleDict(models)

# ---- the pipeline's own __call__, recorded end to end ------------------------------------------------------
def run_pipeline(height, width, prompt):
    kwargs = {}
    params = inspect.signature(pcls.__init__).parameters
    for k in params:
        if k == "self": continue
        if k in models: kwargs[k] = models[k]
        elif k in objs: kwargs[k] = objs[k]
        elif params[k].default is None or k in ("safety_checker", "feature_extractor", "image_encoder"): kwargs[k] = None
        elif k == "requires_safety_checker": kwargs[k] = False
    rec = DagRecorder(container, {})
    try:
        pipe = pcls(**kwargs)
    except Exception as e:
        rec.remove_hooks()
        return rec, None, f"pipeline could not be assembled: {type(e).__name__}: {str(e)[:200]}", {}, {}
    rec.remove_hooks()                              # each call attempt below records with its own recorder
    sig = inspect.signature(pcls.__call__).parameters
    ck = {"prompt": prompt, "num_inference_steps": 2}
    if "height" in sig: ck.update(height=height, width=width)
    # smallest valid call (chosen by parameter name, never by model): the architecture does not depend on size
    for p in sig:
        if "resolution_binning" in p or p in ("_auto_resize", "auto_resize"): ck[p] = False
    if "num_frames" in sig: ck["num_frames"] = 5
    if "max_sequence_length" in sig: ck["max_sequence_length"] = 64
    if "prior_num_inference_steps" in sig: ck["prior_num_inference_steps"] = 2
    from PIL import Image
    TENSORISH = ("embed", "latent")                 # names of tensor slots: never given a picture
    for p in sig:                                   # every image-typed slot of __call__ gets a real image
        if p in ("ip_adapter_image", "ip_adapter_image_embeds") or any(t in p for t in TENSORISH): continue
        if "mask" in p and ("image" in p or p == "mask"): ck[p] = Image.new("L", (width, height), 255)
        elif p in ("image", "init_image", "control_image", "control_images", "image_reference", "reference_image", "cond_image"):
            ck[p] = Image.new("RGB", (width, height), (120, 60, 30))
    # required arguments of __call__ (no default): fill by NAME (never by pipeline name)
    frames = [Image.new("RGB", (width, height), (120 + 10 * i, 60, 30)) for i in range(5)]
    for p, prm in sig.items():
        if p in ("self", "kwargs", "args") or p in ck or prm.default is not inspect.Parameter.empty:
            continue
        n = p.lower()
        if any(t in n for t in TENSORISH):
            ck[p] = ("__EMBED__", p)                 # resolved below: a zero tensor, sizes tried from component configs
        elif "mask" in n: ck[p] = Image.new("L", (width, height), 255)
        elif "video" in n or "frames" in n or n in ("pose_video", "face_video", "reference_video"): ck[p] = frames
        elif "image" in n or n in ("reference", "ref", "cond", "control"): ck[p] = Image.new("RGB", (width, height), (120, 60, 30))
        elif "class" in n and "label" in n: ck[p] = [0]
        elif "prompt" in n and not n.startswith(("num_", "max_", "min_")): ck[p] = prompt
        elif "resolution" in n: ck[p] = max(height, width)
        elif "audio" in n: ck[p] = __import__("numpy").zeros(16000, dtype="float32")
    for p in sig:                                   # optional but data-carrying slots some pipelines check at runtime
        n = p.lower()
        if p not in ck and n.endswith("_boxes"): ck[p] = [[0.1, 0.1, 0.5, 0.5]]
        if p not in ck and n.endswith("_phrases"): ck[p] = ["a cat"]
        if p not in ck and n in ("video", "processing_resolution"):
            ck[p] = frames if n == "video" else max(height, width)
        if p not in ck and n.endswith("_prompt") and "negative" not in n and not n.startswith(("num_", "max_", "min_")):
            ck[p] = prompt
    embed_dims = []                                 # candidate sizes for required embedding tensors: the components' own config
    for m_ in models.values():
        c_ = getattr(m_, "config", None)
        for k_ in ("encoder_hid_dim", "cross_attention_dim", "projection_dim", "image_embed_dim", "embedding_dim", "hidden_size"):
            v_ = getattr(c_, k_, None) if c_ is not None else None
            if isinstance(v_, int) and v_ not in embed_dims: embed_dims.append(v_)
    embed_dims = embed_dims or [768]
    def _variant(kind, dim=None):
        def conv(v):
            if isinstance(v, Image.Image):
                if kind == "list": return [v]
                if kind == "nested": return [[v, v]]
                if kind == "rgba": return v.convert("RGBA")
                if kind == "gray": return v.convert("L")
            if isinstance(v, list) and v and isinstance(v[0], Image.Image) and kind in ("rgba", "gray"):
                return [conv(x) for x in v]
            return v
        out = {}
        drop = set()
        if kind == "novideo": drop = {k_ for k_ in ck if "video" in k_ and sig[k_].default is not inspect.Parameter.empty}
        if kind == "noimage": drop = {k_ for k_ in ck if "image" in k_ and sig[k_].default is not inspect.Parameter.empty}
        if kind == "deflen": drop = {"max_sequence_length"}
        if kind == "deflen_frames": drop = {"max_sequence_length", "num_frames"}
        for k_, v_ in ck.items():
            if k_ in drop: continue
            if isinstance(v_, tuple) and v_ and v_[0] == "__EMBED__":
                out[k_] = torch.zeros(1, dim or embed_dims[0])
            else:
                out[k_] = conv(v_)
        return out
    best = None
    ladder = [("base", None)] + [("base", d) for d in embed_dims[1:4]]
    tried = []
    while ladder:
        kind, dim = ladder.pop(0)
        tried.append(f"{kind}:{dim}" if dim else kind)
        res = _attempt(pipe, _variant(kind, dim), sig)
        if best is None or len(res[0].nodes) > len(best[0].nodes): best = res
        err = res[2] or ""
        if not err: break
        if "has no len()" in err and not any(t.startswith("list") for t in tried): ladder = [("list", dim), ("nested", dim)] + ladder
        elif ("channel" in err.lower()) and not any(t.startswith("rgba") for t in tried): ladder = [("rgba", dim), ("gray", dim)] + ladder
        elif ("simultaneously" in err or "only one of" in err.lower()) and "novideo" not in tried:
            ladder = [("novideo", dim), ("noimage", dim)] + ladder
        elif "must match the size" in err and "deflen" not in tried:
            ladder = [("deflen", dim), ("deflen_frames", dim)] + ladder
        elif not any(isinstance(v, tuple) for v in ck.values()) and not ladder: break
        if len(tried) >= 7: break
    R.setdefault("pipeline_input_variants_tried", tried)
    R.setdefault("pipeline_call_args", {k: (type(v).__name__ if not isinstance(v, tuple) else "embed-zeros")
                                        for k, v in ck.items() if k in sig})
    return best


def _attempt(pipe, ck, sig):
    rec = DagRecorder(container, {})
    clear_library_caches()
    hooks, calls = [], collections.Counter()
    for cn, m in models.items():                         # a component's call arguments are its inputs
        def pre(mod, args, kw, cn=cn):
            calls[cn] += 1
            for t in list(args) + list(kw.values()):
                if isinstance(t, torch.Tensor) and id(t) not in rec.producer:
                    rec.input_names[id(t)] = f"{cn}.arg"
        hooks.append(m.register_forward_pre_hook(pre, with_kwargs=True))
    steps = collections.Counter(); sched = objs.get("scheduler"); o = None
    if sched is not None and hasattr(sched, "step"):
        o = sched.step
        def counted(*a, **k): steps["scheduler.step"] += 1; return o(*a, **k)
        sched.step = counted
    # nested text generation inside a pipeline (prompt enhancers, captioners): with zero weights no end token is ever
    # produced, so cap the length. Length is data, not architecture: every module still runs.
    from transformers.generation.utils import GenerationMixin
    _orig_gen = GenerationMixin.generate
    def _capped(self, *a, **k):
        k["max_new_tokens"] = min(int(k.get("max_new_tokens") or 2), 2)
        k.pop("max_length", None)
        if "min_new_tokens" in k: k["min_new_tokens"] = min(int(k["min_new_tokens"] or 0), 2)
        return _orig_gen(self, *a, **k)
    GenerationMixin.generate = _capped
    stopped, out = None, None
    try:
        with torch.no_grad(), math_attention(), rec:
            out = pipe(**{k: v for k, v in ck.items() if k in sig})
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        lib = [f for f in tb if "site-packages/diffusers/" in f.filename or "site-packages/transformers/" in f.filename]
        f = lib[-1] if lib else tb[-1]
        stopped = f"{type(e).__name__}: {str(e)[:160]} @ {f.filename.split('site-packages/')[-1]}:{f.lineno}"
    finally:
        GenerationMixin.generate = _orig_gen
        if o is not None: sched.step = o
        for h in hooks: h.remove()
        rec.remove_hooks()
    return rec, out, stopped, dict(calls), dict(steps)


rec_a, out_a, stop_a, calls_a, steps_a = run_pipeline(128, 128, "a photo of a cat")
R["pipeline_call"] = {"stopped_at": stop_a, "component_calls": calls_a, "scheduler": steps_a}
save()
rec_b, out_b, stop_b, _, _ = run_pipeline(128, 128, "a watercolor painting of a lighthouse at dusk, soft light")
extra = {}
for cn, m in models.items():                               # entry points the generation path never used
    # (whether or not the pipeline call reached this component: a call that stopped early never touched it)
    if callable(getattr(m, "encode", None)) and any(n.startswith("encoder") for n, _ in m.named_modules()) and not any(
            n.module.startswith(cn + ".encoder") for n in rec_a.nodes):
        ch = getattr(m.config, "in_channels", None) or getattr(m.config, "audio_channels", None) or 3
        errs = []
        for shape in ((1, ch, 128, 128), (1, ch, 8192), (1, ch, 5, 128, 128), (1, ch, 9, 128, 128)):  # image, audio, video
            r = DagRecorder(container, {})
            try:
                with torch.no_grad(), math_attention(), r:
                    m.encode(torch.zeros(*shape))
                extra[cn] = r; R["components"][cn]["encode_input_shape"] = list(shape); break
            except Exception as e:
                errs.append(f"{shape}: {type(e).__name__}: {str(e)[:100]}")
            finally:
                r.remove_hooks()
        if cn not in extra:
            R["components"][cn]["encode_error"] = errs


for cn, m in models.items():
    dec_mods = [n for n, _ in m.named_modules() if n.startswith("decoder") and any(True for _ in _.parameters(recurse=False))]
    if not callable(getattr(m, "decode", None)) or not dec_mods:
        continue
    ran = {c.split(".", 1)[1] for c in rec_a.called if c.startswith(cn + ".") and "." in c}
    if all(d in ran for d in dec_mods):
        continue
    cfg_ = getattr(m, "config", None)
    zc = next((getattr(cfg_, k) for k in ("latent_channels", "z_dim", "z_channels", "embed_dim") if isinstance(getattr(cfg_, k, None), int)), 4)
    errs = []
    for shape in ((1, zc, 3, 16, 16), (1, zc, 16, 16), (1, zc, 64)):        # video (several frames), image, audio
        r = DagRecorder(container, {})
        try:
            with torch.no_grad(), math_attention(), r:
                m.decode(torch.zeros(*shape))
            extra[cn] = extra.get(cn, []) + [r] if isinstance(extra.get(cn), list) else ([extra[cn], r] if cn in extra else [r])
            R["components"][cn]["decode_input_shape"] = list(shape); break
        except Exception as e:
            errs.append(f"{shape}: {type(e).__name__}: {str(e)[:100]}")
        finally:
            r.remove_hooks()
    if "decode_input_shape" not in R["components"][cn]:
        R["components"][cn]["decode_error"] = errs


# direct runs: a component whose own forward never ran in the pipeline call (the call stopped first) is run directly,
# inputs synthesized by argument name with sizes from its own config (synth.py); first candidate that executes is kept,
# a second run with different values gives the structure-stability check
import synth, signal as _sig
direct = {}
def _alarm(*_): raise TimeoutError("direct-run time limit")
_sig.signal(_sig.SIGALRM, _alarm)
for cn, m in models.items():
    if any(c == cn or c.startswith(cn + ".") for c in rec_a.called) or not any(True for _ in m.parameters()):
        continue
    tried, ok = [], None
    for label, kw in synth.candidates(m, limit=12):
        r = DagRecorder(container, synth.flat_tensors(kw, f"{cn}.arg"))     # synthesized tensors are named inputs
        _sig.alarm(240)
        try:
            with torch.no_grad(), math_attention(), r:
                m(**kw)
            ok = (label, r); break
        except Exception as e:
            tried.append(f"{label}: {type(e).__name__}: {str(e)[:90]}")
        finally:
            _sig.alarm(0); r.remove_hooks()
    if ok is None:
        R["components"][cn]["direct_run_error"] = tried[:4]
        continue
    label, r1 = ok
    kw2 = dict(next(kw for lb, kw in synth.candidates(m, limit=12, seed=1) if lb == label))
    r2 = DagRecorder(container, synth.flat_tensors(kw2, f"{cn}.arg"))
    try:
        with torch.no_grad(), math_attention(), r2:
            m(**kw2)
    except Exception:
        r2 = None
    finally:
        if r2 is not None: r2.remove_hooks()
    direct[cn] = (r1, r2)
    extra[cn] = (extra[cn] if isinstance(extra.get(cn), list) else ([extra[cn]] if cn in extra else [])) + [r1]
    R["components"][cn]["direct_run"] = {"inputs": label, "attempts_before": len(tried),
                                        "reason": "the pipeline call never reached this component"}


def norm(nodes, cn):
    return [(re.sub(r"\.\d+", ".N", n.module), n.func) for n in nodes if n.module.split(".")[0] == cn]


verdicts = []
for cn, m in models.items():
    C = R["components"][cn]
    ex_ = extra.get(cn)
    recs = [rec_a] + (ex_ if isinstance(ex_, list) else ([ex_] if ex_ is not None else []))
    used = set().union(*[{p.split(".", 1)[1] for p in r.used_params | r.used_buffers if p.split(".")[0] == cn} for r in recs])
    called = set().union(*[{c.split(".", 1)[1] if "." in c else "(root)" for c in r.called if c.split(".")[0] == cn} for r in recs])
    checks = {}
    try:
        T = headers("" if R["components"][cn].get("from_repo_root") else cn)
    except Exception:
        T = {}
    if T:
        W, A, packed = ckpt_split(T)
        ren = library_rename(m) if isinstance(m, transformers.PreTrainedModel) else None
        bufs, unbuilt = split_library_ignored(m, W, ren) if isinstance(m, transformers.PreTrainedModel) else ([], [])
        W = {k: v for k, v in W.items() if k not in set(bufs) | set(unbuilt)}
        sd = m.state_dict(keep_vars=True); H = {k: tuple(v.shape) for k, v in sd.items()}
        alias = collections.defaultdict(list)
        for k, v in sd.items(): alias[id(v)].append(k)
        matched, un_c, _, _ = match(W, H, alias, packed, rename=ren)
        masks = [c for c in un_c if looks_like_stored_buffer(*W[c])]; un_c = [c for c in un_c if c not in set(masks)]
        for names in alias.values():
            if any(n in used for n in names): used |= set(names)
        unused = [(c, numel(s)) for c, (s, _) in W.items() if c not in set(masks) | set(un_c) and not (
            (h := matched.get(c)) is not None and ((h in used) if not h.startswith(("fused:", "renamed:")) else any(u.startswith(h.split(":", 1)[1] + ".") for u in used)))]
        checks["T1_weights"] = {"pass": not unused and not un_c and not unbuilt, "shipped_weights_unused": top_groups(unused, 4),
                                "shipped_but_not_built_by_model": top_groups([(c, numel(W[c][0])) for c in un_c], 4),
                                "library_declared_unbuilt": len(unbuilt)}
    else:
        checks["T1_weights"] = {"pass": False, "no_ground_truth": True}
    mods = {n or "(root)": mm for n, mm in m.named_modules()}
    drop = {n for n, mm in mods.items() if isinstance(mm, torch.nn.modules.dropout._DropoutNd)}
    leaf = {n for n, mm in mods.items() if type(mm).forward is not torch.nn.Module.forward and n not in drop}
    tied = tied_alias_modules(m, called)
    def exercised(n):
        if n in called or n in tied or n == "(root)": return True
        ps = [pn for pn, _ in mods[n].named_parameters(prefix=n)]
        return bool(ps) and any(p in used for p in ps)
    never_all = sorted(n for n in leaf if not exercised(n))
    never = [n for n in never_all if any(True for _ in mods[n].parameters())]
    C["weightless_never_called_info"] = len(never_all) - len(never)
    checks["T2_modules"] = {"pass": not never, "modules": len(leaf), "never_executed": len(never),
                            "never_executed_groups": dict(collections.Counter(".".join("N" if p.isdigit() else p for p in n.split(".")) for n in never).most_common(6))}
    ext = [e for r in recs for e in r.external if e[2].split(".")[0] == cn and (1 if not e[0] else __import__("math").prod(e[0])) > 1 and "lift_fresh" not in e[3]]
    checks["T3_closed_dataflow"] = {"pass": not ext, "external_nonscalar_tensors": len(ext), "examples": [list(map(str, e)) for e in ext[:4]]}
    opq = collections.Counter(n.func for r in recs for n in r.nodes if n.module.split(".")[0] == cn and __import__("dag").OPAQUE.search(n.func))
    checks["T4_no_opaque_ops"] = {"pass": not opq, "opaque_ops": dict(opq)}
    sa, sb = norm(rec_a.nodes, cn), norm(rec_b.nodes, cn)
    if cn in direct:                                      # structure stability from the two direct runs
        d1, d2 = direct[cn]
        sa = norm(d1.nodes, cn); sb = norm(d2.nodes, cn) if d2 is not None else []
    checks["T5_stable_structure"] = {"pass": bool(sa) and sa == sb, "ops_a": len(sa), "ops_b": len(sb)}
    C["checks"] = checks
    C["failed_checks"] = [k for k, v in checks.items() if not v["pass"]]
    C["verdict"] = "FULL" if not C["failed_checks"] else ("PARTIAL" if used else "FAIL")
    verdicts.append(C["verdict"])
    sub = [n for n in rec_a.nodes if n.module.split(".")[0] == cn]
    C["ops"] = len(sub)
R["stability_run_stopped_at"] = stop_b
if stop_a and stop_a.startswith("pipeline could not be assembled"):
    errs = {cn: C.get("error") for cn, C in R["components"].items() if C.get("error")}
    if any("not in installed libraries" in (e or "") for e in errs.values()):
        verdicts = ["OUT"]; R["reason"] = "components come from a library that is not installed: " + "; ".join(f"{k}: {v[:80]}" for k, v in errs.items())[:350]
    elif any("config.json" in (e or "") or "404" in (e or "") for e in errs.values()):
        verdicts = ["FAIL"]; R["reason"] = "repo incomplete (component files missing): " + "; ".join(f"{k}: {v[:80]}" for k, v in errs.items())[:350]
    else:
        verdicts = ["FAIL"]; R["reason"] = stop_a[:350]
if stop_a and not R.get("reason"):
    R["reason"] = f"pipeline call stopped: {stop_a[:300]}"
R["verdict"] = ("OUT" if verdicts == ["OUT"] else "FULL" if verdicts and all(v == "FULL" for v in verdicts)
                else ("PARTIAL" if any(v in ("PARTIAL", "FULL") for v in verdicts) else "FAIL"))   # FAIL only if nothing was captured
R["failed_components"] = {cn: R["components"][cn]["failed_checks"] for cn in models if R["components"][cn].get("failed_checks")}
save()
