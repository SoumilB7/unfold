"""STRICT capture check for one transformers model. Usage: worker_v2.py <repo> <out.json>

Verdict ladder
  FULL    : ran, and all five checks pass
            T1 every shipped learned weight is used (stored masks/quant bookkeeping excluded; library-declared-unbuilt = fail)
            T2 every module (incl. parameter-free ones) executed in some pass
            T3 closed dataflow: no non-scalar tensor enters from outside the recording
            T4 no opaque fused ops
            T5 identical op structure for two different inputs
  PARTIAL : ran; the failing checks are named
  FAIL    : could not build or run (reason recorded)
  OUT     : excluded by policy (remote code only, gated, no config, class not in installed library)
"""
import sys, os, json, time, resource, traceback, collections, warnings, inspect
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import noweights; noweights.install()
torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "2")))
import transformers
from lowcost import zero_storage, fetch_headers, fetch_bin_headers
from common import ckpt_split, match, library_rename, split_library_ignored, looks_like_stored_buffer, top_groups, numel, INT_DTYPES, buffer_names
from dag import DagRecorder, analyse, tied_alias_modules, math_attention, clear_library_caches
from inputs import build_passes, mask_positions

repo, out_path = sys.argv[1], sys.argv[2]
HDR_CACHE = os.environ.get("BENCH_HEADER_CACHE", os.path.join(HERE, "..", "cache_v2", "headers"))
R = {"repo": repo, "transformers": transformers.__version__, "torch": torch.__version__, "steps": {}}
T0 = time.time()


def _last_resort(etype, e, tb):
    import traceback as _tb
    R["verdict"] = R.get("verdict") or "FAIL"
    R["reason"] = R.get("reason") or f"unhandled {etype.__name__}: {str(e)[:250]}"
    fr = _tb.extract_tb(tb)
    R["where"] = R.get("where") or (f"{fr[-1].filename.split('site-packages/')[-1]}:{fr[-1].lineno}" if fr else "")
    try: save()
    except Exception: pass


sys.excepthook = _last_resort


def save():
    R["secs"] = round(time.time() - T0, 1)
    R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(R, open(out_path, "w"), indent=1, default=str)


def fail(verdict, reason, where=""):
    R["verdict"] = verdict; R["reason"] = reason[:400]; R["where"] = where; save(); sys.exit(0)


def classify_load_error(e):
    s = f"{type(e).__name__}: {e}"
    if "trust_remote_code" in s or "custom code" in s: return "OUT", "remote code only"
    if "gated" in s.lower() or "403" in s or "GatedRepo" in s: return "OUT", "gated (no access)"
    if "does not appear to have a file named config.json" in s or "404" in s: return "OUT", "no config.json"
    if "Unrecognized" in s or "does not recognize" in s: return "OUT", "class not in installed library"
    return "FAIL", s[:300]


# ---- 1. config (build instructions) + headers (ground truth) ------------------------------------------------
# GGUF-only repo: the library loads it with from_pretrained(gguf_file=...), which takes config AND architecture from
# the GGUF header. Same rule here: header range-read into a sparse stub, the library parses it; no tensor data read.
GGUF_RAW = None
try:
    from huggingface_hub import HfApi as _Api
    _files = _Api().list_repo_files(repo)
    if any(f.endswith(".gguf") for f in _files) and not any(f.endswith((".safetensors", ".bin", ".pt", ".pth")) for f in _files):
        import gguf_parts, tempfile
        GGUF_RAW, _kv, _gf, _first = gguf_parts.fetch_gguf_parts(repo)
        import atexit, shutil
        _stub = tempfile.mkdtemp(prefix="gguf_header_stub_")      # sparse header-only file, outside the download cache
        atexit.register(shutil.rmtree, _stub, True)
        try:
            cfg = gguf_parts.library_config(_first, _stub)
        except ValueError as e:
            if "not supported" in str(e): fail("OUT", f"gguf architecture not supported by installed library: {str(e)[:160]}")
            raise
        R["gguf_stub_disk_kb"] = int(os.popen(f"du -sk '{_stub}'").read().split()[0])
        R["gguf_header_bytes_read"] = len(_first[2])
        R["config_source"] = f"gguf header ({_gf[0]}), library gguf_file rule"; R["gguf_files"] = _gf
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as _CLM
        if not getattr(cfg, "architectures", None) and _CLM.get(cfg.model_type):
            cfg.architectures = [_CLM[cfg.model_type]]; R["class_from_library_auto_mapping"] = _CLM[cfg.model_type]
        try:
            _rc = transformers.AutoConfig.from_pretrained(repo, trust_remote_code=False)
            if _rc.model_type != cfg.model_type:
                R["repo_config_disagrees_with_weights"] = {"config.json": _rc.model_type, "gguf header": cfg.model_type}
        except Exception:
            pass
except Exception as e:
    if GGUF_RAW is not None:                 # the library's own GGUF reader cannot read this file: a format limit
        fail("OUT", f"the library's GGUF loader cannot read this file: {type(e).__name__}: {str(e)[:220]}")
    R["steps"]["gguf_probe_error"] = f"{type(e).__name__}: {str(e)[:120]}"
if GGUF_RAW is None:
    try:
        cfg = transformers.AutoConfig.from_pretrained(repo, trust_remote_code=False)
    except Exception as e:
        v, why = classify_load_error(e); fail(v, why)
archs = getattr(cfg, "architectures", None) or []
cls = next((getattr(transformers, a) for a in archs if hasattr(transformers, a)), None)
if cls is None and not archs:
    # no class declared: use the library's own loading rule for this model_type (what AutoModel would build)
    try:
        from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
        nm = MODEL_MAPPING_NAMES.get(cfg.model_type)
        nm = nm[0] if isinstance(nm, (list, tuple)) else nm
        if nm and hasattr(transformers, nm):
            cls = getattr(transformers, nm); R["class_from_library_auto_mapping"] = nm
    except Exception:
        pass
if cls is None:
    fail("OUT", f"class not in installed library: {archs or 'no architectures declared'}")
R["class"] = cls.__name__; R["model_type"] = cfg.model_type
try:
    from huggingface_hub import HfApi
    if GGUF_RAW is not None: raise LookupError("gguf repo: parts list comes from the gguf header after build")
    sha = HfApi().model_info(repo).sha
    os.makedirs(HDR_CACHE, exist_ok=True)
    hp = os.path.join(HDR_CACHE, repo.replace("/", "__") + f"@{sha[:12]}.json")
    cached = json.load(open(hp)) if os.path.exists(hp) else {}
    if cached:
        T = {k: (tuple(v[0]), v[1]) for k, v in cached.items()}
    else:
        T, _ = fetch_headers(repo)
        src = "safetensors"
        if not T:                                  # no safetensors: read the .bin tensor index (restricted, no code run)
            T, _ = fetch_bin_headers(repo); src = "pytorch_bin_index"
        R["ground_truth_source"] = src
        if T: json.dump({k: [list(s), d] for k, (s, d) in T.items()}, open(hp, "w"))
except Exception as e:
    T = {}
    R["steps"]["headers_error"] = f"{type(e).__name__}: {str(e)[:160]}"
R["has_ground_truth"] = bool(T)
if not T and GGUF_RAW is None:
    # weights only in a format the library cannot load (ONNX, OpenVINO IR, NPU/TFLite/CoreML exports): a converted
    # copy, like MLX. Its architecture is the base model's; there is no library loading rule to score it against.
    try:
        _fs = HfApi().list_repo_files(repo)
        _ext = {f.rsplit(".", 1)[-1].lower() for f in _fs if "." in f}
        _conv = sorted(_ext & {"onnx", "onnx_data", "xml", "q4nx", "xclbin", "tflite", "mlmodel", "mlpackage", "engine", "pte", "rknn", "om"})
        _lib = _ext & {"safetensors", "gguf", "pt", "pth", "ckpt", "h5", "msgpack"}
        _ov = "xml" in _conv and any(f.endswith(".bin") and f[:-4] + ".xml" in _fs for f in _fs)
        if _conv and not _lib and (not any(f.endswith(".bin") for f in _fs) or _ov):
            try:
                _cd = HfApi().model_info(repo).card_data
                _base = getattr(_cd, "base_model", None) if _cd is not None else None
            except Exception:
                _base = None
            fail("OUT", f"weights only in a converted format the library cannot load ({', '.join(_conv)}"
                        f"{'; .bin files are OpenVINO weights' if _ov else ''}): architecture is measured on its base model {_base}")
    except SystemExit:
        raise
    except Exception:
        pass
if T and any(k.endswith(".biases") for k in T) and any(d == "U32" for _, d in T.values()):
    try:
        cd = HfApi().model_info(repo).card_data
        base = getattr(cd, "base_model", None) if cd is not None else None
    except Exception:
        base = None
    fail("OUT", f"MLX-quantized checkpoint (packed U32 + scales/biases, not a transformers weight format): architecture is measured on its base model {base}")

# ---- 1b. data-size caps: with zero weights every score ties, so "keep everything above a threshold" keeps
#          everything. Cap unlimited data-dependent counts (not architecture: same modules, same ops).
_caps = []
def _cap(c, path="config"):
    for f in ("max_keypoints", "max_num_keypoints", "max_detections", "num_top_queries_max"):
        v = getattr(c, f, "absent")
        if v in (-1, None):
            setattr(c, f, 64); _caps.append(f"{path}.{f}: {v} -> 64")
    for a in dir(c):
        if a.endswith("_config") and not a.startswith("_"):
            sc = getattr(c, a, None)
            if hasattr(sc, "to_dict") and sc is not c:
                _cap(sc, f"{path}.{a}")
_cap(cfg)
if _caps:
    R["data_size_caps"] = _caps

# ---- 1d. implementation switches: CUDA-only fused kernels cannot run here. The library's own switch
#          (use_*_kernel(s) = False) selects its pure-PyTorch path, as attn_implementation="eager" does for attention.
if not torch.cuda.is_available():
    import re
    def _kern(c, path=""):
        for k, v in list(vars(c).items()):
            if isinstance(v, bool) and v and re.fullmatch(r"use_\w*kernels?", k):
                setattr(c, k, False); R.setdefault("implementation_switches", []).append(f"{path}{k}=False")
            elif hasattr(v, "to_dict") and v is not c:
                _kern(v, f"{path}{k}.")
    _kern(cfg)

# time limits. Optional stages (3a-3d) are bounded so they can never cost the main result: each model call gets a hard
# limit, and no new optional call starts once the job is past its optional budget (the runner kills at
# BENCH_DEADLINE_EPOCH). The library authority (1c, T1) has its own budget, capped by that deadline.
import signal, contextlib
class StageTimeout(Exception):
    pass
ALARMS = [0]                                          # every alarm counts, even one the library swallows
def _on_alarm(sig, frm):
    ALARMS[0] += 1
    raise StageTimeout("stage time limit reached")
signal.signal(signal.SIGALRM, _on_alarm)
STAGE_S = int(os.environ.get("BENCH_STAGE_S", "150")); LIBLOAD_S = int(os.environ.get("BENCH_LIBLOAD_S", "1200")); DEADLINE = float(os.environ.get("BENCH_DEADLINE_EPOCH", "0")); OPTIONAL_BUDGET_S = int(os.environ.get("BENCH_OPTIONAL_BUDGET_S", "600"))
@contextlib.contextmanager
def _limit():
    signal.alarm(STAGE_S)
    try:
        yield
    finally:
        signal.alarm(0)
def _optional_ok(label):
    if time.time() - T0 < OPTIONAL_BUDGET_S:
        return True
    R.setdefault("optional_stages_skipped_time_budget", []).append(label)
    return False

# ---- 1c. class selection by weight evidence: if the declared class does not build every shipped tensor, try every
#          class the library registers for this model_type and keep the one covering the most checkpoint weights.
def _uncovered_numel(klass):
    with torch.device("meta"):
        try:
            mm = klass._from_config(cfg)
        except Exception:
            return None
    W0, _, packed0 = ckpt_split(T)
    ren0 = library_rename(mm)
    bufs0, unb0 = split_library_ignored(mm, W0, ren0)
    W0 = {k: v for k, v in W0.items() if k not in set(bufs0) | set(unb0)}
    H0 = {k: tuple(v.shape) for k, v in mm.state_dict(keep_vars=True).items()}
    al0 = collections.defaultdict(list)
    for k, v in mm.state_dict(keep_vars=True).items(): al0[id(v)].append(k)
    _, unc0, unh0, _ = match(W0, H0, al0, packed0, rename=ren0)
    unc0 = [c for c in unc0 if not looks_like_stored_buffer(*W0[c])]
    return sum(numel(W0[c][0]) for c in unc0), len(unh0)
# The library authority (its own from_pretrained over sparse header stubs) decides when it can run: a class is
# scored by what the library itself says it cannot place (unexpected / mismatched / silently dropped file tensors,
# and parameters it leaves uninitialized). The name matcher below is the fallback when no load report is possible.
def _pos_lengths(c):
    out = set()
    for sc in [c] + [getattr(c, a) for a in dir(c) if a.endswith("_config") and hasattr(getattr(c, a, None), "to_dict")]:
        for f in ("n_positions", "max_position_embeddings", "n_ctx", "max_seq_len", "block_size"):
            v = getattr(sc, f, None)
            if isinstance(v, int) and v > 0: out.add(v)
    return out
_POS = _pos_lengths(cfg)
def _stored_constant(x):
    """A shipped tensor that is a stored table, not a learned weight, from its header alone: an integer/bool table,
    or a float 1x1xNxN matrix whose N is one of the model's own position lengths (a stored attention mask)."""
    s_, d_ = T[x]
    return d_ in INT_DTYPES or d_ == "BOOL" or (len(s_) == 4 and s_[0] == s_[1] == 1 and s_[2] == s_[3] and s_[2] in _POS)
AUTH, _auth_sel, _auth_mirror_failed = None, False, False
def _lib_score(rep):
    """(file side, model side): file tensors the library cannot place (unexpected / mismatched, not integer tables;
    silently dropped), and parameters it leaves uninitialized. None when there is no report."""
    if not rep or "error" in rep: return None
    srcs = rep.get("reported_sources", {})
    def _int_table(k):                                # integer/bool tables and counters (renamed or not)
        ss = srcs.get(k) or [k]
        return all(x in T and _stored_constant(x) for x in ss)
    bad = [k for k in rep["unexpected"] + [m_[0] for m_ in rep["mismatched"]] if not _int_table(k)]
    n_bad = sum(numel(T[x][0]) for k in bad for x in (srcs.get(k) or [k]) if x in T)
    return (n_bad + rep.get("unaccounted_numel", 0), rep.get("missing_param_numel", 0))
if T and R.get("ground_truth_source") in (None, "safetensors") and GGUF_RAW is None:
    _t1c = time.time(); _sel = {"candidates": {}, "complete": False}
    try:
        import libload
        AUTH = libload.Authority(repo)
        import atexit; atexit.register(AUTH.close)
        _A_aux = set(ckpt_split(T)[1])
        def _rep(klass):                              # a report produced while an alarm fired is not trusted/cached
            n0 = ALARMS[0]; r_ = AUTH.report(klass, _A_aux, config=cfg)
            if ALARMS[0] != n0:
                AUTH.cache.pop(klass.__name__, None); raise StageTimeout("alarm during library load")
            return r_
        _left = int(DEADLINE - time.time() - 900) if DEADLINE else LIBLOAD_S
        if _left < 60:
            raise StageTimeout("no time left for library class selection")
        signal.setitimer(signal.ITIMER_REAL, min(LIBLOAD_S, _left), 5)
        try:
            try:
                s0 = _lib_score(_rep(cls))
            except StageTimeout:
                _auth_mirror_failed = AUTH.n is None; raise
            if s0 is not None:
                _auth_sel = True
                if s0 != (0, 0):
                    import transformers.models.auto.modeling_auto as MA
                    cands = set()
                    for nm in dir(MA):
                        mp = getattr(MA, nm)
                        if nm.endswith("_MAPPING_NAMES") and "CONFIG" not in nm and isinstance(mp, dict):
                            v = mp.get(cfg.model_type)
                            for c in ([v] if isinstance(v, str) else list(v) if isinstance(v, (list, tuple)) else []):
                                if hasattr(transformers, c) and not c.endswith("Config"): cands.add(c)
                    scored = []
                    for c in sorted(cands - {cls.__name__}):
                        _n0 = ALARMS[0]
                        try:
                            sc = _lib_score(_rep(getattr(transformers, c)))
                        except StageTimeout:
                            _auth_sel = False; raise          # incomplete: fall back to the matcher's selection
                        except Exception as e:
                            if ALARMS[0] != _n0:              # the library turned our timeout into another error
                                _auth_sel = False; raise StageTimeout("alarm during candidate load")
                            _sel["candidates"][c] = f"error: {type(e).__name__}: {str(e)[:80]}"; continue
                        if sc is not None:
                            scored.append((sc, c)); _sel["candidates"][c] = list(sc)
                        else:
                            _sel["candidates"][c] = "no report"
                    _sel["complete"] = True
                    # switch only on FILE-side evidence: the candidate places strictly more of the shipped tensors,
                    # or fits exactly (nothing unplaced, nothing uninitialized). Parameters missing from the file alone
                    # never decide the class; a tie for best keeps the declared class (recorded as ambiguous).
                    ok = [(sc, c) for sc, c in scored if sc[0] < s0[0] or sc == (0, 0)]
                    if ok:
                        best = min(sc for sc, _ in ok)
                        winners = [c for sc, c in ok if sc == best]
                        if len(winners) == 1:
                            R["class_selected_by_library"] = {"declared": cls.__name__, "selected": winners[0],
                                                              "score_declared": list(s0), "score_selected": list(best)}
                            cls = getattr(transformers, winners[0]); R["class"] = cls.__name__
                        else:
                            _sel["ambiguous"] = winners
                else:
                    _sel["complete"] = True
                _sel["declared_score"] = list(s0)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except Exception as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R["steps"]["library_class_selection_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    _sel["secs"] = round(time.time() - _t1c, 1)
    R["library_class_selection"] = _sel
if T and not _auth_sel:
    base = _uncovered_numel(cls)
    if base and base[0] > 0:
        import transformers.models.auto.modeling_auto as MA
        cands = set()
        for nm in dir(MA):
            mp = getattr(MA, nm)
            if nm.endswith("_MAPPING_NAMES") and isinstance(mp, dict):
                v = mp.get(cfg.model_type)
                for c in ([v] if isinstance(v, str) else list(v) if isinstance(v, (list, tuple)) else []):
                    if hasattr(transformers, c): cands.add(c)
        scored = []
        for c in sorted(cands - {cls.__name__}):
            s = _uncovered_numel(getattr(transformers, c))
            if s is not None: scored.append((s[0], s[1], c))
        if scored:
            best = min(scored)
            if best[0] < base[0]:
                R["class_selected_by_weights"] = {"declared": cls.__name__, "selected": best[2],
                                                  "uncovered_numel_declared": base[0], "uncovered_numel_selected": best[0]}
                cls = getattr(transformers, best[2]); R["class"] = cls.__name__

# ---- 2. build: zero-storage weights ---------------------------------------------------------------------
def _fresh():
    with zero_storage():
        try:
            mm = cls._from_config(cfg, attn_implementation="eager", dtype=torch.float32)
        except (TypeError, ValueError):
            mm = cls._from_config(cfg, dtype=torch.float32)
    return mm.eval()
t = time.time()
try:
    m = _fresh()
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)
    fail("FAIL", f"build: {type(e).__name__}: {str(e)[:250]}", f"{tb[-1].filename.split('site-packages/')[-1]}:{tb[-1].lineno}" if tb else "")
R["steps"]["build_s"] = round(time.time() - t, 2)
R["params"] = sum(p.numel() for p in m.parameters())
if GGUF_RAW is not None:                     # rename GGUF tensors by the library's own GGUF loader map
    try:
        T, _un = gguf_parts.rename_to_library(GGUF_RAW, m)
        R["ground_truth_source"] = "gguf_header"; R["gguf_unmapped_tensors"] = _un[:20]
        R["steps"].pop("headers_error", None)
    except Exception as e:
        R["steps"]["gguf_rename_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    R["has_ground_truth"] = bool(T)

# ---- 3. passes: real inputs from the repo's own processors, chosen by forward() signature --------------
ROOT_NO_FORWARD = type(m).forward is torch.nn.Module.forward        # a wrapper (e.g. omni): only sub-models have forwards
try:
    if ROOT_NO_FORWARD:
        # run every direct sub-model that has its own forward and weights; inputs from ITS signature and ITS config
        passes, passes_b, notes, proc_errs, extra_passes = [], [], [], {}, []
        for cn, child in m.named_children():
            if type(child).forward is torch.nn.Module.forward or not any(True for _ in child.parameters()):
                continue
            ccfg = getattr(child, "config", cfg)
            try:
                p1, n1, e1 = build_passes(repo, child, ccfg)
                child.__dict__.pop("_bench_extra_passes", None)
                p2, _, _ = build_passes(repo, child, ccfg, text_len_variant=True)
            except Exception as e:
                notes.append(f"sub={cn}: input construction {type(e).__name__}: {str(e)[:80]}"); continue
            passes += [(f"sub={cn}|{l}", kw) for l, kw in p1]; passes_b += [(f"sub={cn}|{l}", kw) for l, kw in p2]
            notes += [f"sub={cn}: {x}" for x in n1]; proc_errs.update({f"sub={cn}|{k}": v for k, v in (e1 or {}).items()})
        R["root_has_no_forward"] = True
    else:
        passes, notes, proc_errs = build_passes(repo, m, cfg)
        extra_passes = list(m.__dict__.pop("_bench_extra_passes", []))
        passes_b, _, _ = build_passes(repo, m, cfg, text_len_variant=True)
except Exception as e:
    # the processor could not build inputs: no processor-built passes; the synthesized-input fallback still runs
    passes, passes_b, extra_passes, notes, proc_errs = [], [], [], [f"input construction: {type(e).__name__}: {str(e)[:200]}"], {}
R["input_notes"] = notes; R["processor_errors"] = proc_errs
if not passes:
    notes.append("no processor-built input path for this model's forward(): synthesized inputs will be tried")

used_params, used_buffers, called, pass_res, sigs = set(), set(), set(), {}, {}
S_ = {}
variants = dict(passes_b)
def _schedule():
    for label, kw in passes:
        yield label, kw
    first_ok = next((l for l, _ in passes if pass_res.get(l, {}).get("ok")), None)   # T5 on the first pass that ran
    if first_ok and first_ok in variants:
        yield "stability:" + first_ok, variants[first_ok]
for label, kw in _schedule():
    model_for_pass = _fresh() if label.startswith("stability:") else m
    clear_library_caches()
    rec = DagRecorder(model_for_pass, kw)
    t = time.time()
    try:
        _target = model_for_pass
        if "sub=" in label:                                   # a sub-model pass of a wrapper without forward
            _target = model_for_pass.get_submodule(label.split("sub=", 1)[1].split("|", 1)[0])
        def _call(kw):
            try:
                return _target(**kw, use_cache=False)
            except TypeError as te:
                if "use_cache" not in str(te) and "unexpected keyword" not in str(te): raise
                return _target(**kw)
        with torch.no_grad(), math_attention(), rec:
            try:
                out = _call(kw)
            except (ValueError, RuntimeError):
                squeezed = {k: (v.squeeze(1) if isinstance(v, torch.Tensor) and k.startswith("pixel_values") and v.dim() == 5 and v.shape[1] == 1 else v)
                            for k, v in kw.items()}
                if all(squeezed[k] is kw[k] for k in kw): raise
                for k, v in squeezed.items():
                    if isinstance(v, torch.Tensor): rec.input_names[id(v)] = k
                out = _call(squeezed)
                R.setdefault("input_notes", []).append(f"{label}: processor gave 5-D pixel tensor with singleton group dim; retried squeezed")
        a = analyse(rec, out)
        pass_res[label] = {"ok": True, "secs": round(time.time() - t, 2), **{k: v for k, v in a.items() if k != "structure_signature"}}
        sigs[label] = (a["structure_signature"], [(n.module, n.func) for n in rec.nodes])
        if not label.startswith("stability:") and "router_params" not in S_:
            cands, up, scales = set(), [], set()
            for i, n in enumerate(rec.nodes):
                if n.func not in ("topk", "argmax", "sort", "argsort", "max"):
                    continue
                frontier, depth = [e[1] for e in n.ins if e[0] == "op"], 0
                while frontier and depth < 6:
                    nxt = []
                    for j in frontier:
                        pins = [e[1] for e in rec.nodes[j].ins if e[0] == "param"]
                        if pins:
                            cands.update(pins)              # stop at the first op that reads a weight
                            up.extend(e[1] for e in rec.nodes[j].ins if e[0] == "op")
                        else:
                            nxt += [e[1] for e in rec.nodes[j].ins if e[0] == "op"]
                    frontier, depth = nxt, depth + 1
            # the router's input: the next weighted op upstream (usually a norm). A zero scale there makes every
            # token identical, so its 1-D scale weights are set to ~1 in the router mode.
            pdims = {n: p.dim() for n, p in m.named_parameters()}
            frontier, depth, seen_ = up, 0, set()
            while frontier and depth < 8:
                nxt = []
                for j in frontier:
                    if j in seen_: continue
                    seen_.add(j)
                    pins = [e[1] for e in rec.nodes[j].ins if e[0] == "param"]
                    if pins:
                        scales.update(p for p in pins if pdims.get(p) == 1 and p not in cands)
                    else:
                        nxt += [e[1] for e in rec.nodes[j].ins if e[0] == "op"]
                frontier, depth = nxt, depth + 1
            S_["router_params"] = sorted(cands)
            S_["router_input_scales"] = sorted(scales)
            S_["embedding_params"] = sorted({e[1] for n in rec.nodes if n.func in ("embedding", "embedding_bag") for e in n.ins if e[0] == "param"})
        if not label.startswith("stability:"):
            used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        lib = [f for f in tb if "site-packages/transformers/" in f.filename]
        f = lib[-1] if lib else (tb[-1] if tb else None)
        pass_res[label] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:220]}",
                           "where": f"{f.filename.split('site-packages/')[-1]}:{f.lineno} ({(f.line or '').strip()[:90]})" if f else ""}
    finally:
        rec.remove_hooks()
    save()
# ---- 3a. extra input variants (e.g. a second image resolution) only while weights remain unused ----------
for label, kw in extra_passes:
    if not ({n for n, _ in m.named_parameters()} - used_params) or not _optional_ok(label):
        break
    clear_library_caches()
    rec = DagRecorder(m, kw)
    try:
        with torch.no_grad(), math_attention(), rec, _limit():
            try:
                out = m(**kw, use_cache=False)
            except TypeError:
                out = m(**kw)
        a = analyse(rec, out)
        pass_res[label] = {"ok": True, **{k: v for k, v in a.items() if k != "structure_signature"}}
        used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called
    except Exception as e:
        pass_res[label] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}"}
    finally:
        rec.remove_hooks()

# ---- 3a2. more execution modes (values/inputs only, never structure), only while weights remain unused ------
def _run_mode(label, kw, model=None):
    global used_params, used_buffers, called
    if not _optional_ok(label): return
    mdl = model or m
    clear_library_caches()
    rec = DagRecorder(mdl, {k: v for k, v in kw.items() if isinstance(v, torch.Tensor)})
    try:
        with torch.no_grad(), math_attention(), rec, _limit():
            try:
                out = mdl(**kw, use_cache=False)
            except TypeError:
                out = mdl(**kw)
        a = analyse(rec, out)
        before = len(used_params)
        used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called
        pass_res[label] = {"ok": True, "newly_used_params": len(used_params) - before,
                           **{k: v for k, v in a.items() if k not in ("structure_signature",)}}
    except Exception as e:
        pass_res[label] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}"}
    finally:
        rec.remove_hooks()
def _unused_now():
    return {n for n, _ in m.named_parameters()} - used_params
sig_m = inspect.signature(m.forward).parameters
ok_kw = [kw for l, kw in passes if pass_res.get(l, {}).get("ok")]
base = dict(max(ok_kw, key=len)) if ok_kw else None
if base and _unused_now():
    # (i) optional outputs: every return_*/output_* switch the forward offers, plus a global-attention mask
    flags = {p: True for p, prm in sig_m.items() if p.startswith(("return_", "output_")) and prm.default in (False, None)
             and p not in ("return_dict", "return_loss")}
    if "global_attention_mask" in sig_m and "input_ids" in base:
        g = torch.zeros_like(base["input_ids"]); g[..., 0] = 1; flags["global_attention_mask"] = g
    if flags:
        _run_mode("mode:output_flags", dict(base, **flags))
if base and _unused_now() and "input_ids" in base:
    # (ii) long input: longer than every window/chunk/compress length the config declares
    lens = []
    def _scan(c):
        for k, v in (c.to_dict().items() if hasattr(c, "to_dict") else []):
            if isinstance(v, int) and not isinstance(v, bool) and 2 < v <= 16384 and any(t in k for t in ("window", "compress", "chunk", "block_size", "global", "local", "sliding")):
                lens.append(v)
    _scan(cfg)
    for a_ in dir(cfg):
        if a_.endswith("_config") and hasattr(getattr(cfg, a_, None), "to_dict"): _scan(getattr(cfg, a_))
    L = min(max(lens) * 2 + 8 if lens else 512, 2048)                 # attention memory ~ L^2
    ids = base["input_ids"]
    if ids.shape[-1] < L:
        long_ids = ids.repeat(1, L // ids.shape[-1] + 1)[:, :L]
        kw_long = {k: v for k, v in base.items() if k not in ("attention_mask", "token_type_ids", "position_ids")}
        kw_long["input_ids"] = long_ids
        if "attention_mask" in base: kw_long["attention_mask"] = torch.ones_like(long_ids)
        if "decoder_input_ids" in base and base["decoder_input_ids"].shape[-1] < 64:   # decoder-side paths need tokens too
            d = base["decoder_input_ids"]
            kw_long["decoder_input_ids"] = torch.cat([d[..., :1], long_ids[..., :63].to(d.dtype).expand(d.shape[0], -1)], -1)
            kw_long.pop("decoder_attention_mask", None)
        R["long_input_length"] = L
        _run_mode("mode:long_input", kw_long)
        if pass_res.get("mode:long_input", {}).get("ok"): S_["long_kw"] = kw_long
if base and _unused_now() and S_.get("router_params"):
    # (iii) routers: with zero weights every token is identical and ties, so all go to one expert. Give the
    #       router weights (the weights feeding topk/argmax/sort in the recording) and the token embeddings
    #       (so tokens differ) small random values, and use the longest input so every expert can get tokens.
    hooks = []
    emb_ids = set()
    for pn in S_.get("embedding_params", []):
        try: emb_ids.add(id(m.get_parameter(pn)))
        except Exception: pass
    emb_mods = [mod for _, mod in m.named_modules() if any(id(p) in emb_ids for p in mod.parameters(recurse=False))]
    for emod in emb_mods:                               # every module holding the same table (tied copies too);
        try:                                            # noise on the OUTPUT, no full random table in memory
            hooks.append(emod.register_forward_hook(
                lambda mod, i, o: o + torch.randn(o.shape, generator=torch.Generator().manual_seed(1)).to(o.dtype) * 0.02
                if isinstance(o, torch.Tensor) and o.is_floating_point() else o))
        except Exception:
            pass
    for pn in S_.get("router_input_scales", []):
        try:
            p = m.get_parameter(pn)
            with torch.no_grad():
                p.data = 1 + torch.randn(p.shape, generator=torch.Generator().manual_seed(2)).to(p.dtype) * 0.02
        except Exception:
            pass
    for pn in S_["router_params"]:
        try:
            p = m.get_parameter(pn)
            with torch.no_grad():
                p.data = torch.randn(p.shape, generator=torch.Generator().manual_seed(0)).to(p.dtype) * 0.02
        except Exception:
            pass
    R["router_params_randomised"] = (S_["router_params"] + S_.get("embedding_params", []))[:20]
    _run_mode("mode:router_random", S_.get("long_kw", base))
    for h in hooks: h.remove()

# ---- 3b. generate(): parts that only run inside the generation loop (codecs, multi-stage generators) ------
if hasattr(m, "generate") and (not any(v["ok"] for l, v in pass_res.items() if not l.startswith("stability:")) or _unused_now()) and _optional_ok("generate"):
    ok_kws = [kw for l, kw in passes if pass_res.get(l, {}).get("ok")] or [kw for _, kw in passes]
    gkw = dict(max(ok_kws, key=len)) if ok_kws else {}      # the pass that already combines the most modalities
    gkw = {k: v for k, v in gkw.items() if k not in ("decoder_input_ids", "labels")}
    clear_library_caches()
    rec = DagRecorder(m, gkw)
    t = time.time()
    try:
        with torch.no_grad(), math_attention(), rec, _limit():
            def _gen(kw):
                try:
                    return m.generate(**kw, max_new_tokens=2, min_new_tokens=2, do_sample=False)
                except TypeError as te:
                    if "unexpected keyword" not in str(te): raise
                    return m.generate(**kw)               # e.g. forecasting models: horizon comes from the config
            try:
                out = _gen(gkw)
            except Exception as first_err:
                out, fallback_ok = None, False          # retry with each other working input set
                for kw in sorted(ok_kws, key=len, reverse=True):
                    kw2 = {k: v for k, v in kw.items() if k not in ("decoder_input_ids", "labels")}
                    if kw2 == gkw: continue
                    for k, v in kw2.items():                     # the retry's inputs are model inputs too
                        if isinstance(v, torch.Tensor): rec.input_names[id(v)] = k
                    try:
                        out = _gen(kw2); fallback_ok = True; break
                    except Exception:
                        continue
                if not fallback_ok: raise first_err
        a = analyse(rec, out)
        pass_res["generate"] = {"ok": True, "secs": round(time.time() - t, 2), **{k: v for k, v in a.items() if k != "structure_signature"}}
        used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        f = next((x for x in reversed(tb) if "site-packages/transformers/" in x.filename), tb[-1] if tb else None)
        pass_res["generate"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}",
                                "where": f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" if f else ""}
    finally:
        rec.remove_hooks()
    save()
# ---- 3c. entry points: sub-modules with encode()/decode() that never ran (codecs, VAEs inside a model) ------
def _exercised_prefix(prefix):
    return any(u.startswith(prefix + ".") for u in used_params)
all_kw = {}
for l, kw in passes:
    if pass_res.get(l, {}).get("ok"):
        for k, v in kw.items(): all_kw.setdefault(k, v)
for name, sub in m.named_modules():
    if not name or name.count(".") > 1 or not callable(getattr(sub, "encode", None)) or _exercised_prefix(name):
        continue
    if not any(True for _ in sub.parameters()):
        continue
    esig = inspect.signature(sub.encode).parameters
    ekw = {k: v for k, v in all_kw.items() if k in esig}
    if not ekw or not _optional_ok(f"entry:{name}"):
        continue
    clear_library_caches()
    rec = DagRecorder(m, ekw)
    try:
        with torch.no_grad(), math_attention(), rec, _limit():
            enc = sub.encode(**ekw)
            if callable(getattr(sub, "decode", None)):
                dsig = inspect.signature(sub.decode).parameters
                fields = enc if isinstance(enc, dict) else (vars(enc) if hasattr(enc, "__dict__") else {})
                dkw = {k: v for k, v in dict(fields).items() if k in dsig and v is not None}
                if "padding_mask" in dsig and "padding_mask" in ekw: dkw.setdefault("padding_mask", ekw["padding_mask"])
                if dkw: sub.decode(**dkw)
        a = analyse(rec, enc)
        pass_res[f"entry:{name}"] = {"ok": True, **{k: v for k, v in a.items() if k != "structure_signature"}}
        used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called
    except Exception as e:
        pass_res[f"entry:{name}"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}"}
    finally:
        rec.remove_hooks()
# public methods the model's own classes define (e.g. predict_contacts), run when their module owns weights
# nothing has read yet; discovered by reflection, arguments filled by name from the inputs that already ran
BLOCK_VERBS = ("save", "push", "load", "resize", "set", "tie", "init", "prune", "from", "to", "register", "enable", "disable",
               "freeze", "unfreeze", "gradient", "post_init", "generate", "prepare", "can_", "num_", "get_input_embeddings",
               "get_output_embeddings", "get_decoder", "get_encoder", "train", "eval", "half", "float", "cuda", "cpu",
               "merge", "fuse", "reset", "clear", "update", "add", "delete", "remove", "apply", "quantize", "dequantize", "smart_apply")
ALIAS = {"tokens": "input_ids", "input": "input_ids", "ids": "input_ids", "images": "pixel_values", "image": "pixel_values"}
def _unused_under(prefix):
    return any((n.startswith(prefix + ".") or not prefix) and n not in used_params for n, _ in m.named_parameters())
tried = set()
for name, sub in m.named_modules():
    if name.count(".") > 1 or not _unused_under(name):
        continue
    for cls_ in type(sub).__mro__:
        if not cls_.__module__.startswith("transformers.models."):
            continue
        for meth, fn in vars(cls_).items():
            if meth.startswith("_") or meth in ("forward", "encode", "decode") or meth.startswith(BLOCK_VERBS) or not inspect.isfunction(fn):
                continue
            if (type(sub).__name__, meth) in tried or not _unused_under(name) or not _optional_ok(f"entry:{meth}"):
                continue
            tried.add((type(sub).__name__, meth))
            try:
                msig = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                continue
            mkw, ok = {}, True
            for pn, prm in list(msig.items())[1:]:
                if prm.kind in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD): continue
                src = pn if pn in all_kw else ALIAS.get(pn)
                if src in all_kw: mkw[pn] = all_kw[src]
                elif prm.default is prm.empty: ok = False; break
            if not ok or not mkw:
                continue
            clear_library_caches()
            rec = DagRecorder(m, mkw)
            try:
                with torch.no_grad(), math_attention(), rec, _limit():
                    res = getattr(sub, meth)(**mkw)
                a = analyse(rec, res)
                before = len(used_params)
                used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called
                pass_res[f"entry:{name or '(root)'}.{meth}"] = {"ok": True, "newly_used_params": len(used_params) - before,
                                                                 **{k: v for k, v in a.items() if k != "structure_signature"}}
            except Exception as e:
                pass_res[f"entry:{name or '(root)'}.{meth}"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}"}
            finally:
                rec.remove_hooks()
# ---- 3d. training pass: weights the model builds but only training uses (masking vectors, denoising queries,
#          loss weights, mask tokens). Extra inputs are chosen by what the forward ACCEPTS, never by model name.
sig_f = inspect.signature(m.forward).parameters
P_names = {n for n, _ in m.named_parameters()}
persist_bufs = set(m.state_dict().keys()) - P_names                      # checkpoint-backed buffers (e.g. loss weights)
if ((P_names - used_params) or (persist_bufs - used_buffers)) and any(v["ok"] for v in pass_res.values()):
    ok_kws2 = [kw for l, kw in passes if pass_res.get(l, {}).get("ok")] or [kw for _, kw in passes]
    base_kw = dict(max(ok_kws2, key=len)) if ok_kws2 else {}
    extras = [{}]
    pv = base_kw.get("pixel_values")
    if "bool_masked_pos" in sig_f:
        extras = [{"bool_masked_pos": mask_positions(base_kw, cfg)}]
    label_cands = [None]
    if "labels" in sig_f:
        H_, W_ = (pv.shape[-2], pv.shape[-1]) if pv is not None and pv.dim() >= 4 else (64, 64)
        ids = base_kw.get("input_ids")
        label_cands = [
            [{"class_labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}],
            [{"class_labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "masks": torch.zeros(1, H_, W_)}],
            ids.clone() if ids is not None else None,
            torch.tensor([0]),
            torch.zeros(1, dtype=torch.long),
        ]
    mask_cands = [{}]
    if "mask_labels" in sig_f and "class_labels" in sig_f and pv is not None:
        mask_cands = [{"mask_labels": [torch.zeros(1, pv.shape[-2], pv.shape[-1])], "class_labels": [torch.tensor([0])]}]
    m.train()
    done_train, errs = False, []
    for ex in extras:
        for mk in mask_cands:
            for lab in label_cands:
                kw = dict(base_kw, **ex, **mk)
                if lab is not None: kw["labels"] = lab
                if not _optional_ok("train"): break
                clear_library_caches()
                rec = DagRecorder(m, {k: v for k, v in kw.items() if isinstance(v, torch.Tensor)})
                try:
                    with torch.no_grad(), math_attention(), rec, _limit():
                        out = m(**kw)
                    newly = (rec.used_params - used_params)
                    a = analyse(rec, out)
                    pass_res["train"] = {"ok": True, "extra_inputs": sorted(set(ex) | set(mk) | ({"labels"} if lab is not None else set())),
                                         "newly_used_params": len(newly), "ops": a["ops"],
                                         "note": "training-mode pass: counts toward T1/T2 only (loss/matching code may route through Python)"}
                    used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called
                    done_train = True
                except Exception as e:
                    errs.append(f"{type(e).__name__}: {str(e)[:90]}")
                finally:
                    rec.remove_hooks()
                if done_train: break
            if done_train: break
        if done_train: break
    m.eval()
    if not done_train:
        pass_res["train"] = {"ok": False, "error": " | ".join(errs[:3])}
    save()
R["passes"] = pass_res
ok_passes = [l for l, v in pass_res.items() if v["ok"] and not l.startswith("stability:") and l != "train"]
if not ok_passes and not ROOT_NO_FORWARD:
    # every processor-built pass failed: synthesize the forward's inputs by argument name with sizes from the config
    # (synth.py); the first candidate that executes becomes the pass, a second run with other values gives T5
    import synth
    tried = []
    for label, kw in synth.candidates(m, limit=40):
        clear_library_caches()
        rec = DagRecorder(m, synth.flat_tensors(kw, "synth"))
        try:
            with torch.no_grad(), math_attention(), rec, _limit():
                out = m(**kw)
            a = analyse(rec, out)
            pass_res["synth"] = {"ok": True, "inputs": label, **{k: v for k, v in a.items() if k != "structure_signature"}}
            sigs["synth"] = (a["structure_signature"], [(n.module, n.func) for n in rec.nodes])
            used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called
            passes.append(("synth", kw))
            kw2 = next(k2 for lb, k2 in synth.candidates(m, limit=40, seed=1) if lb == label)
            m2 = _fresh(); rec2 = DagRecorder(m2, synth.flat_tensors(kw2, "synth"))
            try:
                with torch.no_grad(), math_attention(), rec2, _limit():
                    o2 = m2(**kw2)
                sigs["stability:synth"] = (analyse(rec2, o2)["structure_signature"], [(n.module, n.func) for n in rec2.nodes])
                pass_res["stability:synth"] = {"ok": True}
            except Exception as e:
                pass_res["stability:synth"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
            finally:
                rec2.remove_hooks()
            R.setdefault("input_notes", []).append(f"inputs synthesized by argument name ({label}); processor-built passes all failed")
            break
        except Exception as e:
            tried.append(f"{label}: {type(e).__name__}: {str(e)[:90]}")
        finally:
            rec.remove_hooks()
    if "synth" not in pass_res:
        pass_res["synth"] = {"ok": False, "error": " | ".join(tried[:3])}
    R["passes"] = pass_res
    ok_passes = [l for l, v in pass_res.items() if v["ok"] and not l.startswith("stability:") and l != "train"]
if not ok_passes:
    fail("FAIL", "every forward pass failed: " + "; ".join(f"{l}: {v.get('error','')[:120]}" for l, v in pass_res.items()))

# ---- 4. checks ---------------------------------------------------------------------------------------------
checks = {}
# T1 weights
if T:
    W, A, packed = ckpt_split(T)
    ren = library_rename(m)
    bufs, unbuilt = split_library_ignored(m, W, ren)
    W = {k: v for k, v in W.items() if k not in set(bufs) | set(unbuilt)}
    H = {k: tuple(v.shape) for k, v in m.state_dict(keep_vars=True).items()}
    alias = collections.defaultdict(list)
    for k, v in m.state_dict(keep_vars=True).items():
        alias[id(v)].append(k)
    matched, un_c, _, _ = match(W, H, alias, packed, rename=ren)
    model_buffers = buffer_names(m)                         # includes non-persistent buffers (recomputed tables)
    masks = [c for c in un_c if looks_like_stored_buffer(*W[c]) or c in model_buffers or (ren and ren(c) in model_buffers)]
    un_c = [c for c in un_c if c not in set(masks)]
    used = set(used_params) | set(used_buffers)
    for names in alias.values():
        if any(n in used for n in names): used |= set(names)
    tot = sum(numel(W[c][0]) for c in W if c not in set(masks))
    unused = []
    for c, (s, _) in W.items():
        if c in set(masks) or c in set(un_c): continue
        h = matched.get(c)
        ok = h is not None and ((h in used) if not h.startswith(("fused:", "renamed:")) else any(u.startswith(h.split(":", 1)[1] + ".") for u in used))
        if not ok: unused.append((c, numel(s)))
    # ---- execution witness for what the library declares it never builds (e.g. MTP): results_v2/_witness/vllm/<repo>.json
    #      (vLLM executed these tensors under the recorder; written by mtp_witness_run.py). Covered keys count as executed.
    WIT = None
    try:
        _wp = os.path.join(HERE, "..", "results_v2", "_witness", "vllm", repo.replace("/", "__") + ".json")
        WIT = json.load(open(_wp)) if os.path.exists(_wp) else None
    except Exception:
        WIT = None
    witnessed = []
    if WIT and WIT.get("closed") and unbuilt:
        _ex = set(WIT.get("executed_checkpoint_keys") or [])
        witnessed = [k for k in unbuilt if k in _ex]
        if len(witnessed) == len(unbuilt):
            R["executed_by_witness"] = {"runtime": WIT.get("executed_by"), "vllm_class": WIT.get("vllm_class"),
                                        "tensors": len(witnessed), "notes": WIT.get("notes")}
            unbuilt = []
        else:
            R["executed_by_witness_partial"] = {"covered": len(witnessed), "declared_unbuilt": len(unbuilt)}
    # leftover eligibility for declared-unbuilt tensors: no runtime implements them (vLLM's own MTP registry said none)
    no_runtime = bool(WIT) and not WIT.get("vllm_mtp_arch") and not WIT.get("registry_error") and "vllm_mtp_arch" in WIT
    unb = sum(numel(T[k][0]) for k in unbuilt)
    # ---- library authority (safetensors repos): which shipped keys the library consumes, from its own from_pretrained
    #      on sparse header stubs (file choice, renames, merges/splits, ignore lists all the library's). Replaces the
    #      name matcher's "class lacks" and "unused" sets; the matcher's result is kept for comparison.
    LIB = None
    reported_by_library = None                    # None = no library load report (matcher only): nothing certified
    if R.get("ground_truth_source") in (None, "safetensors") and GGUF_RAW is None:
        try:
            import libload
            # its own budget (sharded repos have hundreds of headers to read), not the optional-stage limit
            # never past the runner's kill (its deadline, on its clock): leave 150 s to finish, grade and save.
            # A repeating timer: if the library swallows one alarm (its conversion ops catch every Exception), the
            # next one fires 5 s later, so the deadline holds.
            _left = int(DEADLINE - time.time() - 150) if DEADLINE else LIBLOAD_S
            if _left < 30:
                raise StageTimeout(f"no time left for the library load report ({_left}s)")
            signal.setitimer(signal.ITIMER_REAL, min(LIBLOAD_S, _left), 5)
            try:
                if _auth_mirror_failed:
                    raise StageTimeout("library mirror already timed out in class selection")
                if AUTH is None:
                    AUTH = libload.Authority(repo); import atexit; atexit.register(AUTH.close)
                _n0 = ALARMS[0]
                LIB = AUTH.report(cls, set(A), config=cfg)       # cached when class selection already loaded it
                if ALARMS[0] != _n0:                             # an alarm the library swallowed: report not trusted
                    LIB = None; raise StageTimeout("alarm during library load")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception as e:
            signal.setitimer(signal.ITIMER_REAL, 0)        # a tick that landed inside the finally above left it armed
            R["library_load_report_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        if LIB and "error" in LIB:                 # e.g. no safetensors files: say so instead of looking like a legacy row
            R["library_load_report_error"] = f"not run: {LIB['error']}"
    if not LIB or "error" in LIB:
        R["t1_authority"] = "name matcher (no library load report)"
    if LIB and "error" not in LIB:
        matcher_lacks = list(un_c)
        matcher_view = {"class_lacks": top_groups([(c, numel(W[c][0])) for c in un_c], 5), "unused": top_groups(unused, 5)}
        # the library names reported keys after ITS renames: test each against its checkpoint source(s)
        _srcs = LIB.get("reported_sources", {})
        _src = lambda k: _srcs.get(k) or [k]
        lib_unexp = [k for k in LIB["unexpected"] if not all(x in set(unbuilt) | set(bufs) for x in _src(k))]
        lib_unexp = [k for k in lib_unexp if not all(x in T and _stored_constant(x) for x in _src(k))
                     and k not in model_buffers]
        mism = [m_[0] for m_ in LIB["mismatched"]]
        loaded = {n for n, _ in m.named_parameters()} - set(LIB["missing"])
        used_n = set(used)
        un_loaded = sorted(p for p in loaded if p not in used_n)
        shp = {n: tuple(p.shape) for n, p in m.named_parameters()}
        un_c = lib_unexp + [k for k in mism if k not in lib_unexp]
        reported_by_library = list(un_c)          # "unexpected"/"mismatched": the library says NOT ok for an identical arch
        # silent drops: tensors in the files the library read that land nowhere and are not reported. Named by the
        # matcher's class-lacks keys inside the read files when those explain the size; otherwise an unnamed deficit.
        deficit = LIB.get("unaccounted_numel", 0)
        silent = []
        if deficit:
            readk = set(LIB.get("read_keys", []))
            cand = [c for c in matcher_lacks if c in readk and c not in un_c]
            if cand and abs(sum(numel(T[c][0]) for c in cand) - deficit) <= max(1, deficit // 100):
                silent = cand
            else:
                silent = [f"(unidentified: {deficit:,} numbers dropped silently by the library loader)"]
                T[silent[0]] = ((deficit,), "?")
            un_c = un_c + [c for c in silent if c not in un_c]
        R["library_silent_drops"] = silent[:50]
        unused = [(p, numel(shp[p])) for p in un_loaded]
        R["t1_authority"] = "library_load_report"
        R["library_load_report"] = {"files_read": LIB.get("files_read"), "files_ignored": sorted(set(LIB.get("files_all", [])) - set(LIB.get("files_read", []))),
                                    "unaccounted_numel": LIB.get("unaccounted_numel"), "unexpected": len(LIB["unexpected"]), "missing": LIB["missing"][:20], "mismatched": LIB["mismatched"][:10],
                                    "loaded_params": LIB["loaded_params"], "model_params": LIB["model_params"], "stub_disk_kb": LIB["stub_disk_kb"],
                                    "matcher_view": matcher_view}
        W = dict(W, **{k: T[k] for k in un_c if k in T})
        if LIB.get("read_keys"):
            rk = set(LIB["read_keys"])
            R["checkpoint_numel_in_files_library_reads"] = sum(numel(T[k][0]) for k in rk if k in T)
    checks["T1_weights"] = {
        "pass": not unused and not un_c and not unbuilt,
        # leftovers = ONLY what the library certifies BY DESIGN: keys it declares ignorable (class or built-model list)
        # and keys its own conversion silently drops. Merely reported "unexpected" keys do not qualify (the library
        # itself says "not ok if you expect identical arch"), nor does anything only our matcher saw.
        "leftovers_eligible": (not unused) and bool(un_c or unbuilt) and (not unbuilt or no_runtime)
                              and (not un_c or (reported_by_library is not None and not reported_by_library
                                                and all(c in (R.get("library_silent_drops") or []) for c in un_c)
                                                and not any(str(c).startswith("(unidentified") for c in un_c))),
        "shipped_weights_unused": top_groups(unused, 5), "shipped_weights_unused_numel": sum(n for _, n in unused),
        "shipped_but_not_built_by_model": top_groups([(c, numel((W.get(c) or T.get(c) or ((0,),))[0])) for c in un_c], 5),
        "library_declared_unbuilt_numel": unb, "library_declared_unbuilt_prefixes": sorted({".".join(k.split(".")[:3]) for k in unbuilt})[:5],
        "stored_masks_excluded": len(bufs) + len(masks), "checkpoint_learned_numel": tot + unb}
    # parts list for everything shipped that no run executed: drawn from the checkpoint alone
    # (name + shape + reason), wiring unknown. Grouped by layer pattern; FULL stays strict.
    def _inv(items, reason):
        g = collections.OrderedDict()
        for c in items:
            s = tuple((T.get(c) or W.get(c) or [()])[0])
            key = __import__("re").sub(r"\.\d+(?=\.|$)", ".*", c)
            e = g.setdefault(key, {"pattern": key, "reason": reason, "tensors": 0, "numel": 0, "shape": list(s)})
            e["tensors"] += 1; e["numel"] += numel(s)
        return list(g.values())
    unused_names = [c for c, _ in unused]
    mode_ran = {l for l, v in pass_res.items() if v.get("ok")}
    R["shipped_not_executed"] = (_inv(sorted(unbuilt), "library_declared_unbuilt")
                                 + _inv(sorted(un_c), "not_in_library_class")
                                 + _inv(sorted(unused_names), "built_but_not_executed_in_any_mode"))
    R["shipped_not_executed_modes_tried"] = sorted(mode_ran)
else:
    P = dict(m.named_parameters())
    unused = [(n, p.numel()) for n, p in P.items() if n not in used_params]
    checks["T1_weights"] = {"pass": False, "no_ground_truth": True, "note": "no safetensors headers: only the model's own parameters could be checked",
                            "model_params_unused": top_groups(unused, 5)}
# T2 modules
mods = {n or "(root)": mod for n, mod in m.named_modules()}
dropouts = {n for n, mod in mods.items() if isinstance(mod, torch.nn.modules.dropout._DropoutNd)}   # identity at inference
leafish = {n for n, mod in mods.items() if type(mod).forward is not torch.nn.Module.forward and n not in dropouts}
tied = tied_alias_modules(m, called)
param_used = set(used_params)
def exercised(n):
    if n in called or n in tied: return True
    ps = [pn for pn, _ in mods[n].named_parameters(prefix="" if n == "(root)" else n)]
    return bool(ps) and any(p in param_used for p in ps)          # weights read directly (e.g. .weight of an embedding)
never_all = sorted(n for n in leafish if not exercised(n))
never = [n for n in never_all if any(True for _ in mods[n].parameters())]            # modules WITH weights that never ran
never_paramfree = [n for n in never_all if n not in set(never)]                        # weightless definitions never called
checks["T2_modules"] = {"pass": not never, "modules": len(leafish), "never_executed": len(never), "dropout_modules_excluded": len(dropouts), "weight_tied_alias_modules": len(tied),
                        "never_executed_groups": dict(collections.Counter(".".join("N" if p.isdigit() else p for p in n.split(".")) for n in never).most_common(8)),
                        "weightless_never_called_info": dict(collections.Counter(".".join("N" if p.isdigit() else p for p in n.split(".")) for n in never_paramfree).most_common(6))}
# T3 closure, T4 opacity
ext = sum(v.get("external_nonscalar", 0) for l, v in pass_res.items() if v["ok"] and not l.startswith("stability:"))
pm = sum(v.get("python_mediated_values", 0) for l, v in pass_res.items() if v["ok"] and not l.startswith("stability:"))
checks["T3_closed_dataflow"] = {"pass": ext == 0, "external_nonscalar_tensors": ext, "python_mediated_values_info": pm,
                                "examples": sum((v.get("external_examples", []) for v in pass_res.values() if v["ok"]), [])[:5],
                                "dead_ops_info": {l: v.get("dead_ops") for l, v in pass_res.items() if v["ok"]}}
opq = collections.Counter()
for v in pass_res.values():
    if v["ok"]: opq.update(v.get("opaque_ops", {}))
checks["T4_no_opaque_ops"] = {"pass": not opq, "opaque_ops": dict(opq)}
# T5 stability: first pass vs the same modality with a different input
first = next((l for l, _ in passes if l in sigs), passes[0][0])
if first in sigs and ("stability:" + first) in sigs:
    a, b = sigs[first][1], sigs["stability:" + first][1]
    norm = lambda s: [(__import__("re").sub(r"\.\d+", ".N", mm), f) for mm, f in s]
    na, nb = norm(a), norm(b)
    div = next((i for i in range(min(len(na), len(nb))) if na[i] != nb[i]), None)
    same = na == nb
    checks["T5_stable_structure"] = {"pass": same, "ops_a": len(na), "ops_b": len(nb),
                                     "first_divergence": None if same else {"index": div if div is not None else min(len(na), len(nb)),
                                                                             "a": na[div] if div is not None and div < len(na) else None,
                                                                             "b": nb[div] if div is not None and div < len(nb) else None}}
else:
    checks["T5_stable_structure"] = {"pass": False, "note": "second input could not be run"}
R["checks"] = checks
failed = [k for k, v in checks.items() if not v["pass"]]
R["verdict"] = "FULL" if not failed else "PARTIAL"
_t1 = checks.get("T1_weights", {})
if failed == ["T1_weights"] and _t1.get("leftovers_eligible"):
    # everything the model runs is captured; the only gap is shipped tensors the library itself certifies as unused by
    # this model (declared-ignored / reported unexpected / silently dropped) and that no available runtime executes
    R["verdict"] = "FULL_LEFTOVERS"
    R["leftovers"] = {"library_declared_unbuilt": _t1.get("library_declared_unbuilt_prefixes"),
                      "not_in_library_class": _t1.get("shipped_but_not_built_by_model"),
                      "silently_dropped": R.get("library_silent_drops"),
                      "certification": ("transformers load report" if R.get("t1_authority") == "library_load_report" else "no load report")
                                       + " / declared-ignore list" + ("; vLLM MTP registry: none" if _t1.get("library_declared_unbuilt_numel") else "")}
R["failed_checks"] = failed
R["structure"] = {l: {k: v.get(k) for k in ("layer_patterns", "cross_layer_edges", "non_layer_into_layers", "activation_like_ops", "ops", "zero_skipped")}
                  for l, v in pass_res.items() if v["ok"] and not l.startswith("stability:")}
save()
