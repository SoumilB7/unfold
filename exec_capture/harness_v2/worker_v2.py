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
T0M = time.monotonic()               # job clock (pauses in sleep, as the runner's kill timer does)
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import noweights; noweights.install()
torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "2")))
import transformers
from lowcost import zero_storage, fetch_headers, fetch_bin_headers
import common
from common import grading_id, runtime, ckpt_split, match, library_rename, split_library_ignored, looks_like_stored_buffer, top_groups, numel, INT_DTYPES, buffer_names
from dag import DagRecorder, analyse, tied_alias_modules, math_attention, clear_library_caches
import dag
from inputs import build_passes, mask_positions
import attempts                                   # F13: one recorder per attempt; a failed attempt credits nothing
ATTEMPT_LEDGER = []                               # every attempt of every stage (entry point, inputs, flags, ok / error)

repo, out_path = sys.argv[1], sys.argv[2]
HDR_CACHE = os.environ.get("BENCH_HEADER_CACHE", os.path.join(HERE, "..", "cache_v2", "headers"))
UNKNOWNS = []       # predicates that could not be evaluated (INTENT.md): an otherwise FULL row becomes FULL_UNVERIFIED
R = {"repo": repo, "transformers": transformers.__version__, "torch": torch.__version__, "steps": {}, "grading_id": grading_id(), "runtime": runtime()}
T0 = time.time()
# capture bundle for the renderer (capture.py): opt-in, never part of the graded result
EMIT = os.environ.get("BENCH_EMIT_CAPTURE")
CAP, BUILD_ATTN = {"passes": {}, "exec_kw": {}, "pass_order": []}, None
def _emit():
    if not EMIT or CAP.get("written"): return
    CAP["written"] = True
    try:
        import capture
        g = globals()
        # the checkpoint's own tensor names, also under the library's renames: which tie-group members it ships
        _T, _ren = g.get("T") or {}, g.get("ren")
        shipped = set(_T) | ({_ren(k) for k in _T} if _ren else set())
        CAP["path"] = capture.write(EMIT, repo, R, CAP, g.get("m"), g.get("cfg"), BUILD_ATTN, shipped)
    except Exception as e:
        sys.stderr.write(f"capture bundle not written: {type(e).__name__}: {e}\n")


def _last_resort(etype, e, tb):
    import traceback as _tb
    common.after_grading_exception(R, UNKNOWNS, f"{etype.__name__}: {str(e)[:200]}")   # FULL* never stands after a crash
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


network_error = noweights.network_error         # type-based (noweights.py): an outage is never evidence about a model


def fail(verdict, reason, where=""):
    R["verdict"] = verdict; R["reason"] = reason[:400]; R["where"] = where; save(); _emit(); sys.exit(0)


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
# one checkpoint revision for the whole run (INTENT.md G6): every hub read below uses this commit
try:
    from huggingface_hub import HfApi as _Api0
    R["revision"] = _Api0().model_info(repo).sha
    noweights.pin(repo, R["revision"])
except Exception as e:
    if network_error(e):
        fail("FAIL", f"network: repo revision unavailable ({network_error(e)}: {str(e)[:120]})", "revision")
    R["revision_error"] = f"{type(e).__name__}: {str(e)[:160]}"     # e.g. gated / missing: classified at config load
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
    if network_error(e):
        fail("FAIL", f"network: GGUF probe ({network_error(e)}: {str(e)[:120]})", "gguf probe")
    R["steps"]["gguf_probe_error"] = f"{type(e).__name__}: {str(e)[:120]}"
if GGUF_RAW is None:
    try:
        cfg = transformers.AutoConfig.from_pretrained(repo, trust_remote_code=False)
    except Exception as e:
        if network_error(e):
            fail("FAIL", f"network: config unavailable ({network_error(e)}: {str(e)[:120]})", "config")
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
    sha = R.get("revision") or HfApi().model_info(repo).sha
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
    # a network failure says nothing about the model: never grade a row without its checkpoint's tensor list because
    # the connection dropped (e.g. the Mac slept); FAIL as an environment error so the runner re-runs it
    if network_error(e):
        fail("FAIL", f"network: checkpoint headers unavailable ({network_error(e)}: {str(e)[:120]})", "headers")
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
    except Exception as e:
        if network_error(e):
            fail("FAIL", f"network: repo file list unavailable ({network_error(e)}: {str(e)[:120]})", "file list")
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

# ---- F04: the load hypothesis. The library load (authority) and the executed build start from ONE config snapshot,
#      taken here after the declared switches above and before any build (the library's _from_config writes dtype and
#      attention implementation into the config it is given). The authority loads with the attention implementation
#      the executed build uses (a documented from_pretrained option), so loaded and executed are comparable exactly;
#      the library's default attention is reported beside it (info).
import copy as _copy
CFG_LOAD = _copy.deepcopy(cfg)
_ATTN_CHOICE = {}
def _attn_choice(klass):
    """The attention implementation the executed build of this class uses (the same rule as _fresh: eager when the
    class accepts it, else the library default), from a meta build of the load config snapshot."""
    if klass not in _ATTN_CHOICE:
        try:
            with torch.device("meta"):
                klass._from_config(_copy.deepcopy(CFG_LOAD), attn_implementation="eager", dtype=torch.float32)
            _ATTN_CHOICE[klass] = "eager"
        except (TypeError, ValueError):         # _fresh's own fallback rule (the class does not take eager): the library
            _ATTN_CHOICE[klass] = None          # default. Anything else (e.g. a stage time limit) is raised, never cached
    return _ATTN_CHOICE[klass]
def _lkw(klass):
    a = _attn_choice(klass)
    return {"attn_implementation": a} if a else {}
_A_aux = None

# time limits. Optional stages (3a-3d) are bounded so they can never cost the main result: each model call gets a hard
# limit, and no new optional call starts once the job is past its optional budget (the runner kills at
# BENCH_JOB_TIMEOUT_S, monotonic clock). The library authority (1c, T1) has its own budget, capped by that deadline.
import signal, contextlib
class StageTimeout(Exception):
    pass
ALARMS = [0]                                          # every alarm counts, even one the library swallows
def _on_alarm(sig, frm):
    ALARMS[0] += 1
    raise StageTimeout("stage time limit reached")
signal.signal(signal.SIGALRM, _on_alarm)
STAGE_S = int(os.environ.get("BENCH_STAGE_S", "150")); LIBLOAD_S = int(os.environ.get("BENCH_LIBLOAD_S", "1200")); JOB_TIMEOUT_S = float(os.environ.get("BENCH_JOB_TIMEOUT_S", "0")); DEADLINE = (T0M + JOB_TIMEOUT_S - 10) if JOB_TIMEOUT_S else 0; OPTIONAL_BUDGET_S = int(os.environ.get("BENCH_OPTIONAL_BUDGET_S", "600"))
@contextlib.contextmanager
def _limit():
    signal.alarm(STAGE_S)
    try:
        yield
    finally:
        signal.alarm(0)
def _optional_ok(label):
    if time.monotonic() - T0M < OPTIONAL_BUDGET_S:          # job clock: a sleep does not use up the budget
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
    bufs0, unb0 = split_library_ignored(mm, W0, ren0, QW)
    W0 = {k: v for k, v in W0.items() if k not in set(bufs0) | set(unb0)}
    H0 = {k: tuple(v.shape) for k, v in mm.state_dict(keep_vars=True).items()}
    al0 = collections.defaultdict(list)
    for k, v in mm.state_dict(keep_vars=True).items(): al0[id(v)].append(k)
    _, unc0, unh0, _ = match(W0, H0, al0, packed0, rename=ren0)
    _bn0 = buffer_names(mm)
    unc0 = [c for c in unc0 if not common.stored_constant(c, *W0[c], quantized=QW, pos=_POS,
                                                         buffer_slot=(c in _bn0 or bool(ren0 and ren0(c) in _bn0)))]
    return sum(numel(W0[c][0]) for c in unc0), len(unh0)
# The library authority (its own from_pretrained over sparse header stubs) decides when it can run: a class is
# scored by what the library itself says it cannot place (unexpected / mismatched / silently dropped file tensors,
# and parameters it leaves uninitialized). The name matcher below is the fallback when no load report is possible.
_POS = common.pos_lengths(cfg)
QW = common.quantized_weight_keys(T) if T else set()      # quantized learned weights: never stored tables (P0.6 B4)
def _stored_constant(x, rep=None):
    """A shipped tensor that is a stored table, not a learned weight: THE common rule (common.stored_constant), with the
    library load's own evidence for an integer / bool tensor (F05: dtype alone is not evidence)."""
    ev = ((rep or {}).get("int_table_evidence") or {}).get(x)
    return common.stored_constant(x, *T[x], quantized=QW, pos=_POS, buffer_slot=ev in ("loaded_buffer", "nonpersistent_slot"),
                                  declared=ev == "declared_ignorable")
AUTH, _auth_sel, _auth_mirror_failed = None, False, False
DECLARED_CLS = cls        # the class a user's default load uses: R-REFUSE judges its load (Soumil 2026-10-13)
import class_rules
TIED, TIED_FWD, TIED_SUB = [], [], []   # tied classes (P0.4), their forward parameter names, their sub-module types
def _tie(res):
    """A tie for best with no declared winner (class_rules.resolve): record it, add the G1 unknown (never FULL), and
    keep what the tied classes share, so only passes and inputs every tied class supports run."""
    if not res["record"]:
        return
    TIED[:] = [getattr(transformers, c) for c in res["tied"]]
    TIED_FWD[:] = [set(inspect.signature(k.forward).parameters) for k in TIED]
    TIED_SUB[:] = []
    for k in TIED:                                  # sub-module types (depth <= 1), from a meta build of each class
        try:
            with torch.device("meta"):
                mk = k._from_config(cfg)
            TIED_SUB.append({n: type(x).__name__ for n, x in mk.named_modules() if n and n.count(".") <= 1})
        except Exception:
            TIED_SUB.append({})                     # unknown layout: no sub-module entry is shared
    R["class_ambiguous"] = res["record"]
    UNKNOWNS.append(res["unknown"])
def _skip_tied(what):
    R.setdefault("passes_skipped_class_ambiguous", []).append(what)
def _lib_score(rep):
    """(file side, model side): file tensors the library cannot place (unexpected / mismatched, not integer tables;
    silently dropped), and parameters it leaves uninitialized. None when there is no report."""
    if not rep or "error" in rep: return None
    if not libload.authority(rep)["certifies"]: return None   # a load that certifies nothing decides nothing (P0.5)
    srcs = rep.get("reported_sources", {})
    def _int_table(k):                                # integer/bool tables and counters (renamed or not)
        ss = srcs.get(k) or [k]
        return all(x in T and _stored_constant(x, rep) for x in ss)
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
            n0 = ALARMS[0]; r_ = AUTH.report(klass, _A_aux, config=CFG_LOAD, **_lkw(klass))
            if ALARMS[0] != n0:
                AUTH.forget(klass, _A_aux, CFG_LOAD, **_lkw(klass)); raise StageTimeout("alarm during library load")
            return r_
        _left = int(DEADLINE - time.monotonic() - 900) if DEADLINE else LIBLOAD_S
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
                            _rc = _rep(getattr(transformers, c))
                            _sel.setdefault("candidate_states", {})[c] = (_rc or {}).get("load_state")
                            if libload.refuses(_rc):              # a class the library refuses is never selected
                                _sel["candidates"][c] = "refused by the library"; continue
                            sc = _lib_score(_rc)
                        except StageTimeout:
                            _auth_sel = False; raise          # incomplete: fall back to the matcher's selection
                        except Exception as e:
                            if ALARMS[0] != _n0:              # the library turned our timeout into another error
                                _auth_sel = False; raise StageTimeout("alarm during candidate load")
                            if network_error(e):              # an outage is not the library refusing a candidate
                                raise
                            _sel["candidates"][c] = f"error: {type(e).__name__}: {str(e)[:80]}"; continue
                        if sc is not None:
                            scored.append((sc, c)); _sel["candidates"][c] = list(sc)
                        else:
                            _sel["candidates"][c] = "no report"
                    _sel["complete"] = True
                    signal.setitimer(signal.ITIMER_REAL, 0)   # scoring done: no alarm may split the decision below
                    # switch only on FILE-side evidence: the candidate places strictly more of the shipped tensors, or
                    # fits exactly (nothing unplaced, nothing uninitialized). Parameters missing from the file alone never
                    # decide the class. The declared class among the best is kept; a tie for best with no declared winner
                    # is never decided by name: G1 unknown, shared passes only (class_rules, P0.4)
                    _res = class_rules.resolve("library load report", cls.__name__, scored, class_rules.authority_qualifies(s0))
                    if _res["switched"]:
                        R["class_selected_by_library"] = {"declared": cls.__name__, "selected": _res["selected"],
                                                          "score_declared": list(s0), "score_selected": list(_res["best"])}
                        if _res["tied"]: _sel["ambiguous"] = _res["tied"]
                        _tie(_res)
                        cls = getattr(transformers, _res["selected"]); R["class"] = cls.__name__
                    # candidates that could not be scored are never dropped silently (P0.6 N6)
                    _inc = class_rules.selection_incomplete((_res["best"] if _res["switched"] else s0)[0],
                                                            [f"{c}: {v}" for c, v in _sel["candidates"].items()
                                                             if not isinstance(v, list) and v != "refused by the library"])
                    if _inc: UNKNOWNS.append(_inc)
                else:
                    _sel["complete"] = True
                _sel["declared_score"] = list(s0)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except Exception as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R["steps"]["library_class_selection_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        if not isinstance(e, StageTimeout) and network_error(e):   # lost its connection: not a fact about the model
            fail("FAIL", f"network: library class selection ({network_error(e)}: {str(e)[:120]})", "class selection")
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
                    if hasattr(transformers, c) and not c.endswith("Config"): cands.add(c)   # model classes only
        scored, _unscored = [], []
        for c in sorted(cands - {cls.__name__}):
            s = _uncovered_numel(getattr(transformers, c))
            if s is not None: scored.append(((s[0], s[1]), c))
            else: _unscored.append(f"{c}: could not be built from this config")
        R["class_selection_matcher"] = {"declared_uncovered_numel": base[0], "scored": {c: list(sc) for sc, c in scored},
                                        "unscored": _unscored}
        # a tie for best with no declared winner is never decided by name (class_rules, P0.4)
        _res = class_rules.resolve("weight-name matcher", cls.__name__, scored, class_rules.matcher_qualifies(base[0]))
        if _res["switched"]:
            R["class_selected_by_weights"] = {"declared": cls.__name__, "selected": _res["selected"],
                                              "uncovered_numel_declared": base[0], "uncovered_numel_selected": _res["best"][0]}
            _tie(_res)
            cls = getattr(transformers, _res["selected"]); R["class"] = cls.__name__
        _inc = class_rules.selection_incomplete(_res["best"][0] if _res["switched"] else base[0], _unscored)   # P0.6 N6
        if _inc: UNKNOWNS.append(_inc)

# ---- 2. build: zero-storage weights ---------------------------------------------------------------------
def _fresh():
    global BUILD_ATTN
    with zero_storage():
        try:
            mm = cls._from_config(cfg, attn_implementation="eager", dtype=torch.float32); BUILD_ATTN = "eager"
        except (TypeError, ValueError):
            mm = cls._from_config(cfg, dtype=torch.float32); BUILD_ATTN = "library default"
    _lc = __import__("lowcost")                    # one float32 build (sub-configs may declare bf16); learned weights
    mm = _lc.normalise_learned(_lc.uniform_float32(mm.eval()))  # executed as zeros (F07-3)
    try:
        BUILD_IDS.append(libload.model_identity(mm))    # F04: every build of the executed model is the same model
    except Exception as _e:
        BUILD_IDS.append({"error": f"identity: {type(_e).__name__}: {str(_e)[:120]}"})
    return mm
import libload
BUILD_IDS = []
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
                if network_error(e):
                    fail("FAIL", f"network: input construction ({network_error(e)}: {str(e)[:120]})", "inputs")
                notes.append(f"sub={cn}: input construction {type(e).__name__}: {str(e)[:80]}"); continue
            passes += [(f"sub={cn}|{l}", kw) for l, kw in p1]; passes_b += [(f"sub={cn}|{l}", kw) for l, kw in p2]
            notes += [f"sub={cn}: {x}" for x in n1]; proc_errs.update({f"sub={cn}|{k}": v for k, v in (e1 or {}).items()})
        R["root_has_no_forward"] = True
    else:
        passes, notes, proc_errs = build_passes(repo, m, cfg)
        extra_passes = list(m.__dict__.pop("_bench_extra_passes", []))
        passes_b, _, _ = build_passes(repo, m, cfg, text_len_variant=True)
except Exception as e:
    if network_error(e):                       # an outage is not "this model has no input path"
        fail("FAIL", f"network: input construction ({network_error(e)}: {str(e)[:120]})", "inputs")
    # the processor could not build inputs: no processor-built passes; the synthesized-input fallback still runs
    passes, passes_b, extra_passes, notes, proc_errs = [], [], [], [f"input construction: {type(e).__name__}: {str(e)[:200]}"], {}
R["input_notes"] = notes; R["processor_errors"] = proc_errs
if not passes:
    notes.append("no processor-built input path for this model's forward(): synthesized inputs will be tried")

used_params, used_buffers, called, pass_res, sigs = set(), set(), set(), {}, {}
PASS_READS = []                     # F05: per credited pass (tensors read, modules run): sibling evidence is per pass
def libload_pass(rec):
    return (set(rec.used_params) | set(rec.used_buffers), set(rec.called))
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
    if EMIT and not label.startswith("stability:"):
        import capture
        _recorder = capture.CaptureRecorder
    else:
        _recorder = DagRecorder
    t = time.time()
    rec, _ledger = None, []
    try:
        _target = model_for_pass
        if "sub=" in label:                                   # a sub-model pass of a wrapper without forward
            _target = model_for_pass.get_submodule(label.split("sub=", 1)[1].split("|", 1)[0])
        def _call(kw):
            # each attempt records alone (F13); the retry without use_cache only after a use_cache TypeError
            try:
                return attempts.run(model_for_pass, model_for_pass, kw, lambda: _target(**kw, use_cache=False), _ledger,
                                    {"entry": label, "use_cache": False}, recorder=_recorder)
            except TypeError as te:
                if "use_cache" not in str(te) and "unexpected keyword" not in str(te): raise
                return attempts.run(model_for_pass, model_for_pass, kw, lambda: _target(**kw), _ledger,
                                    {"entry": label}, recorder=_recorder)
        _exec_kw, _retried = kw, False
        try:
            rec, out = _call(kw)
        except (ValueError, RuntimeError):
            squeezed = {k: (v.squeeze(1) if isinstance(v, torch.Tensor) and k.startswith("pixel_values") and v.dim() == 5 and v.shape[1] == 1 else v)
                        for k, v in kw.items()}                # outside every recorder: the worker's op, not the model's
            if all(squeezed[k] is kw[k] for k in kw): raise
            rec, out = _call(squeezed); _exec_kw, _retried = squeezed, True
            R.setdefault("input_notes", []).append(f"{label}: processor gave 5-D pixel tensor with singleton group dim; retried squeezed")
        a = analyse(rec, out)
        pass_res[label] = {"ok": True, "secs": round(time.time() - t, 2), "attempts": _ledger, **{k: v for k, v in a.items() if k != "structure_signature"}}
        _keep = label.startswith("stability:") or not any(not l.startswith("stability:") for l in sigs)   # (T5's two runs)
        sigs[label] = (a["structure_signature"], [(n.module, n.func) for n in rec.nodes], rec.nodes if _keep else None)
        if EMIT and not label.startswith("stability:") and "sub=" not in label:
            try:
                _cp = CAP["passes"]
                _first = next(iter(_cp), None)                 # always kept: the drawn pass
                _other = [l for l in _cp if l != _first]       # at most one more: the pass with the most inputs
                if _first is None or len(_exec_kw) > len(_cp[_other[0] if _other else _first]["inputs"]):
                    for l in _other: _cp.pop(l); CAP["exec_kw"].pop(l, None)
                    _cp[label] = {"graph": capture.graph(rec), "squares": capture.squares(rec),
                                  "inputs": capture.input_shapes(_exec_kw), "squeezed_retry": _retried}
                    CAP["exec_kw"][label] = _exec_kw
                    if "model" not in CAP:          # model facts as they were for the drawn pass (before any mode)
                        CAP["model"] = capture.model_facts(m)
                        CAP["max_position_embeddings"] = getattr(cfg, "max_position_embeddings", None)
                CAP["pass_order"].append(label)
                # every pass's repeated stacks (cheap), so the reader can check what the drawn pass leaves out
                import re
                CAP.setdefault("pass_stacks", {})[label] = sorted({mm_.group(1) for n_ in rec.nodes
                                                                   for mm_ in [re.match(r"^(.*?)\.(\d+)(?:\.|$)", n_.module)] if mm_})
            except Exception as e:
                CAP.setdefault("errors", []).append(f"{label}: {type(e).__name__}: {str(e)[:160]}")
        if not label.startswith("stability:") and "router_params" not in S_:
            cands, up, scales = set(), [], set()
            _bodyn = __import__("dag").body_nodes(rec.nodes)  # F10: the program's own calls, not a decomposition's
            for i, n in enumerate(rec.nodes):
                if n.func not in ("topk", "argmax", "sort", "argsort", "max") or i in _bodyn:
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
            used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called; PASS_READS.append(libload_pass(rec))
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        lib = [f for f in tb if "site-packages/transformers/" in f.filename]
        f = lib[-1] if lib else (tb[-1] if tb else None)
        pass_res[label] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:220]}", "attempts": _ledger,
                           "where": f"{f.filename.split('site-packages/')[-1]}:{f.lineno} ({(f.line or '').strip()[:90]})" if f else ""}
    finally:
        if rec is not None: rec.remove_hooks()
        ATTEMPT_LEDGER.extend(_ledger)
    save()
# ---- 3a. extra input variants (e.g. a second image resolution) only while weights remain unused ----------
for label, kw in extra_passes:
    if not ({n for n, _ in m.named_parameters()} - used_params) or not _optional_ok(label):
        break
    rec, _ledger = None, []
    try:
        with _limit():
            try:
                rec, out = attempts.run(m, m, kw, lambda: m(**kw, use_cache=False), _ledger, {"entry": label, "use_cache": False})
            except TypeError:
                rec, out = attempts.run(m, m, kw, lambda: m(**kw), _ledger, {"entry": label})
        a = analyse(rec, out)
        pass_res[label] = {"ok": True, "attempts": _ledger, **{k: v for k, v in a.items() if k != "structure_signature"}}
        used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called; PASS_READS.append(libload_pass(rec))
    except Exception as e:
        pass_res[label] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}", "attempts": _ledger}
    finally:
        if rec is not None: rec.remove_hooks()
        ATTEMPT_LEDGER.extend(_ledger)

# ---- 3a2. more execution modes (values/inputs only, never structure), only while weights remain unused ------
def _run_mode(label, kw, model=None):
    global used_params, used_buffers, called
    if not _optional_ok(label): return
    mdl = model or m
    tkw = {k: v for k, v in kw.items() if isinstance(v, torch.Tensor)}
    rec, _ledger = None, []
    try:
        with _limit():
            try:
                rec, out = attempts.run(mdl, mdl, tkw, lambda: mdl(**kw, use_cache=False), _ledger, {"entry": label, "use_cache": False}, py_inputs=kw)
            except TypeError:
                rec, out = attempts.run(mdl, mdl, tkw, lambda: mdl(**kw), _ledger, {"entry": label}, py_inputs=kw)
        a = analyse(rec, out)
        before = len(used_params)
        used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called; PASS_READS.append(libload_pass(rec))
        pass_res[label] = {"ok": True, "newly_used_params": len(used_params) - before, "attempts": _ledger,
                           **{k: v for k, v in a.items() if k not in ("structure_signature",)}}
    except Exception as e:
        pass_res[label] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}", "attempts": _ledger}
    finally:
        if rec is not None: rec.remove_hooks()
        ATTEMPT_LEDGER.extend(_ledger)
def _unused_now():
    return {n for n, _ in m.named_parameters()} - used_params
sig_m = inspect.signature(m.forward).parameters
ok_kw = [kw for l, kw in passes if pass_res.get(l, {}).get("ok")]
base = dict(max(ok_kw, key=len)) if ok_kw else None
if base and _unused_now():
    # (i) optional outputs: every return_*/output_* switch the forward offers, plus a global-attention mask
    flags = {p: True for p, prm in sig_m.items() if p.startswith(("return_", "output_")) and prm.default in (False, None)
             and p not in ("return_dict", "return_loss")}
    for p in [p for p in flags if not class_rules.shared_input(TIED_FWD, p)]:
        del flags[p]; _skip_tied(f"mode:output_flags.{p}")             # not every tied class accepts it (P0.4)
    if "global_attention_mask" in sig_m and "input_ids" in base and class_rules.shared_input(TIED_FWD, "global_attention_mask"):
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
    _router_snap = attempts.Snapshot(m)                 # the state before the router weights are randomised (F13)
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
    _lost = _router_snap.restore()                      # F13: later passes run the tested (zero-weight) state again
    ATTEMPT_LEDGER.append({"entry": "mode:router_random (randomised weights put back)", "ok": not _lost,
                           **({"state_not_restored": _lost[:20]} if _lost else {})})

# ---- 3b. generate(): parts that only run inside the generation loop (codecs, multi-stage generators) ------
_gen_wanted = hasattr(m, "generate") and (not any(v["ok"] for l, v in pass_res.items() if not l.startswith("stability:")) or _unused_now())
if _gen_wanted and not class_rules.shared(TIED, "generate"):
    _skip_tied("generate")                                              # not every tied class can generate (P0.4)
if hasattr(m, "generate") and class_rules.shared(TIED, "generate") and (not any(v["ok"] for l, v in pass_res.items() if not l.startswith("stability:")) or _unused_now()) and _optional_ok("generate"):
    ok_kws = [kw for l, kw in passes if pass_res.get(l, {}).get("ok")] or [kw for _, kw in passes]
    gkw = dict(max(ok_kws, key=len)) if ok_kws else {}      # the pass that already combines the most modalities
    gkw = {k: v for k, v in gkw.items() if k not in ("decoder_input_ids", "labels")}
    rec, _ledger = None, []
    t = time.time()
    try:
        with _limit():
            _gen_mode = {}
            def _gen(kw):                               # each attempt records alone (F13)
                # Fable N3: every call asks for greedy (do_sample=False) where the library's generate() takes it; a
                # generate() that takes no such flag (e.g. forecasting models sample their horizon) is recorded as such
                for gk, desc in (({"max_new_tokens": 2, "min_new_tokens": 2, "do_sample": False}, {"max_new_tokens": 2, "deterministic": True}),
                                 ({"do_sample": False}, {"deterministic": True}),
                                 ({}, {"deterministic": False, "why": "generate() takes no do_sample: it may sample"})):
                    try:
                        r_ = attempts.run(m, m, kw, lambda: m.generate(**kw, **gk), _ledger, {"entry": "generate", **desc})
                        _gen_mode.clear(); _gen_mode.update(desc)
                        return r_
                    except TypeError as te:
                        if not gk or "unexpected keyword" not in str(te): raise   # (the last form's error is the call's)
            try:
                rec, out = _gen(gkw)
            except Exception as first_err:
                fallback_ok = False                     # retry with each other working input set
                for kw in sorted(ok_kws, key=len, reverse=True):
                    kw2 = {k: v for k, v in kw.items() if k not in ("decoder_input_ids", "labels")}
                    if kw2 == gkw: continue
                    try:
                        rec, out = _gen(kw2); fallback_ok = True; break
                    except Exception:
                        continue
                if not fallback_ok: raise first_err
        a = analyse(rec, out)
        pass_res["generate"] = {"ok": True, "secs": round(time.time() - t, 2), "attempts": _ledger, **_gen_mode,
                                **{k: v for k, v in a.items() if k != "structure_signature"}}
        used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called; PASS_READS.append(libload_pass(rec))
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        f = next((x for x in reversed(tb) if "site-packages/transformers/" in x.filename), tb[-1] if tb else None)
        pass_res["generate"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "attempts": _ledger,
                                "where": f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" if f else ""}
    finally:
        if rec is not None: rec.remove_hooks()
        ATTEMPT_LEDGER.extend(_ledger)
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
    if not class_rules.entry_allowed(TIED, TIED_SUB, name, "encode"):    # not the same module in every tied class (P0.4)
        _skip_tied(f"entry:{name}"); continue
    if not any(True for _ in sub.parameters()):
        continue
    esig = inspect.signature(sub.encode).parameters
    ekw = {k: v for k, v in all_kw.items() if k in esig}
    if not ekw or not _optional_ok(f"entry:{name}"):
        continue
    rec, _ledger = None, []
    try:
        def _encdec(sub=sub, ekw=ekw):
            enc = sub.encode(**ekw)
            if callable(getattr(sub, "decode", None)):
                dsig = inspect.signature(sub.decode).parameters
                fields = enc if isinstance(enc, dict) else (vars(enc) if hasattr(enc, "__dict__") else {})
                dkw = {k: v for k, v in dict(fields).items() if k in dsig and v is not None}
                if "padding_mask" in dsig and "padding_mask" in ekw: dkw.setdefault("padding_mask", ekw["padding_mask"])
                if dkw: sub.decode(**dkw)
            return enc
        with _limit():
            rec, enc = attempts.run(m, m, ekw, _encdec, _ledger, {"entry": f"entry:{name}"})
        a = analyse(rec, enc)
        pass_res[f"entry:{name}"] = {"ok": True, "attempts": _ledger, **{k: v for k, v in a.items() if k != "structure_signature"}}
        used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called; PASS_READS.append(libload_pass(rec))
    except Exception as e:
        pass_res[f"entry:{name}"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}", "attempts": _ledger}
    finally:
        if rec is not None: rec.remove_hooks()
        ATTEMPT_LEDGER.extend(_ledger)
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
            if not class_rules.entry_allowed(TIED, TIED_SUB, name, meth):   # not the same in every tied class (P0.4)
                tried.add((type(sub).__name__, meth))
                _skip_tied(f"entry:{name or '(root)'}.{meth}"); continue
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
            rec, _ledger = None, []
            try:
                with _limit():
                    rec, res = attempts.run(m, m, mkw, lambda: getattr(sub, meth)(**mkw), _ledger, {"entry": f"entry:{name or '(root)'}.{meth}"})
                a = analyse(rec, res)
                before = len(used_params)
                used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called; PASS_READS.append(libload_pass(rec))
                pass_res[f"entry:{name or '(root)'}.{meth}"] = {"ok": True, "newly_used_params": len(used_params) - before, "attempts": _ledger,
                                                                 **{k: v for k, v in a.items() if k != "structure_signature"}}
            except Exception as e:
                pass_res[f"entry:{name or '(root)'}.{meth}"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}", "attempts": _ledger}
            finally:
                if rec is not None: rec.remove_hooks()
                ATTEMPT_LEDGER.extend(_ledger)
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
    _sh = lambda p: p in sig_f and (class_rules.shared_input(TIED_FWD, p) or _skip_tied(f"train.{p}"))   # P0.4
    if _sh("bool_masked_pos"):
        extras = [{"bool_masked_pos": mask_positions(base_kw, cfg)}]
    label_cands = [None]
    if _sh("labels"):
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
    if _sh("mask_labels") and _sh("class_labels") and pv is not None:
        mask_cands = [{"mask_labels": [torch.zeros(1, pv.shape[-2], pv.shape[-1])], "class_labels": [torch.tensor([0])]}]
    m.train()
    done_train, errs, train_ledger = False, [], []
    def _double(v):
        """The same inputs as a batch of 2 (F05-3: a BatchNorm layer cannot train on one example)."""
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == 1:
            return torch.cat([v, v], 0)
        if isinstance(v, list) and len(v) == 1:
            return v * 2
        return v
    for _bs in (1, 2):                                   # batch 2 only when every batch-1 candidate failed
        if done_train: break
        if _bs == 2:
            base_kw = {k: _double(v) for k, v in base_kw.items()}
            extras = [{k: _double(v) for k, v in ex.items()} for ex in extras]
            mask_cands = [{k: _double(v) for k, v in mk.items()} for mk in mask_cands]
            label_cands = [_double(lab) for lab in label_cands]
        for ex in extras:
            for mk in mask_cands:
                for lab in label_cands:
                    kw = dict(base_kw, **ex, **mk)
                    if lab is not None: kw["labels"] = lab
                    if not _optional_ok("train"): break
                    rec, _ledger = None, []
                    try:
                        with _limit():
                            rec, out = attempts.run(m, m, {k: v for k, v in kw.items() if isinstance(v, torch.Tensor)},
                                                    lambda: m(**kw), _ledger, {"entry": "train", "labels": type(lab).__name__, "batch": _bs}, py_inputs=kw)
                        newly = (rec.used_params - used_params)
                        newly_b = (rec.used_buffers - used_buffers) & persist_bufs   # F10-4: checkpoint buffers it alone reads
                        a = analyse(rec, out)
                        pass_res["train"] = {"ok": True, "extra_inputs": sorted(set(ex) | set(mk) | ({"labels"} if lab is not None else set())),
                                             "newly_used_params": len(newly), "newly_used_buffers": len(newly_b),
                                             "ops": a["ops"], "body_ops": a["body_ops"], "attempts": train_ledger, "batch": _bs,
                                             **{k_: a[k_] for k_ in ("opaque_ops", "opaque_reasons", "opacity_unknown_ops", "opacity_unknown_reasons", "decomposition_disagrees", "decomposed_ops", "decomposed_random_info")},
                                             **__import__("dag").f09_fields(a),                     # (F09 B1: every pass record)
                                             "note": "training-mode pass: counts toward T1/T2 only (loss/matching code may route through Python)"}
                        used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called; PASS_READS.append(libload_pass(rec))
                        done_train = True
                    except Exception as e:
                        errs.append(f"{type(e).__name__}: {str(e)[:90]}")
                    finally:
                        if rec is not None: rec.remove_hooks()
                        ATTEMPT_LEDGER.extend(_ledger); train_ledger.extend(_ledger)
                    if done_train: break
                if done_train: break
            if done_train: break
    m.eval()
    if not done_train:
        pass_res["train"] = {"ok": False, "error": " | ".join(errs[:3]), "attempts": train_ledger}
    save()
R["passes"] = pass_res
ok_passes = [l for l, v in pass_res.items() if v["ok"] and not l.startswith("stability:") and l != "train"]
if not ok_passes and not ROOT_NO_FORWARD:
    # every processor-built pass failed: synthesize the forward's inputs by argument name with sizes from the config
    # (synth.py); the first candidate that executes becomes the pass, a second run with other values gives T5
    import synth
    tried, synth_ledger = [], []
    for label, kw in synth.candidates(m, limit=40):
        rec, _ledger = None, []
        try:
            with _limit():
                rec, out = attempts.run(m, m, synth.flat_tensors(kw, "synth"), lambda: m(**kw), _ledger, {"entry": f"synth:{label}"}, py_inputs=kw)
            a = analyse(rec, out)
            pass_res["synth"] = {"ok": True, "inputs": label, "attempts": synth_ledger, **{k: v for k, v in a.items() if k != "structure_signature"}}
            sigs["synth"] = (a["structure_signature"], [(n.module, n.func) for n in rec.nodes], rec.nodes)
            used_params |= rec.used_params; used_buffers |= rec.used_buffers; called |= rec.called; PASS_READS.append(libload_pass(rec))
            passes.append(("synth", kw))
            kw2 = next(k2 for lb, k2 in synth.candidates(m, limit=40, seed=1) if lb == label)
            m2, rec2, _l2 = _fresh(), None, []
            try:
                with _limit():
                    rec2, o2 = attempts.run(m2, m2, synth.flat_tensors(kw2, "synth"), lambda: m2(**kw2), _l2,
                                            {"entry": f"stability:synth:{label}"}, py_inputs=kw2)
                _a2 = analyse(rec2, o2)
                sigs["stability:synth"] = (_a2["structure_signature"], [(n.module, n.func) for n in rec2.nodes], rec2.nodes)
                pass_res["stability:synth"] = {"ok": True, "attempts": _l2,      # (F10: its ops are classified too)
                                               **{k_: _a2[k_] for k_ in ("opaque_ops", "opaque_reasons", "opacity_unknown_ops", "opacity_unknown_reasons", "decomposition_disagrees", "decomposed_ops", "decomposed_random_info")},
                                               **__import__("dag").f09_fields(_a2)}                 # (F09 B1: every pass record)
            except Exception as e:
                pass_res["stability:synth"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}", "attempts": _l2}
            finally:
                if rec2 is not None: rec2.remove_hooks()
                ATTEMPT_LEDGER.extend(_l2)
            R.setdefault("input_notes", []).append(f"inputs synthesized by argument name ({label}); processor-built passes all failed")
            break
        except Exception as e:
            tried.append(f"{label}: {type(e).__name__}: {str(e)[:90]}")
        finally:
            if rec is not None: rec.remove_hooks()
            ATTEMPT_LEDGER.extend(_ledger); synth_ledger.extend(_ledger)
    if "synth" not in pass_res:
        pass_res["synth"] = {"ok": False, "error": " | ".join(tried[:3]), "attempts": synth_ledger}
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
    bufs, unbuilt = split_library_ignored(m, W, ren, QW)
    W_UNBUILT = {k: W[k] for k in unbuilt}         # shapes under ckpt_split names (a packed key has no raw-name twin in T)
    W = {k: v for k, v in W.items() if k not in set(bufs) | set(unbuilt)}
    H = {k: tuple(v.shape) for k, v in m.state_dict(keep_vars=True).items()}
    alias = collections.defaultdict(list)
    for k, v in m.state_dict(keep_vars=True).items():
        alias[id(v)].append(k)
    matched, un_c, _, _ = match(W, H, alias, packed, rename=ren)
    model_buffers = buffer_names(m)                         # includes non-persistent buffers (recomputed tables)
    _sigM = common.recomputed_signatures(m, loaded=set(matched.values()))   # F05: a hint only (values not compared)
    masks = [c for c in un_c if common.stored_constant(c, *W[c], quantized=QW, pos=_POS,
                                                     buffer_slot=(c in model_buffers or bool(ren and ren(c) in model_buffers)))]
    un_c = [c for c in un_c if c not in set(masks)]
    # F05: integer / bool tables with no evidence of their meaning are not learned weights (no failure) and are never
    # hidden either: a G1 unknown names them (int_tables_unexplained); quantized codes stay learned weights (QW)
    INT_UNEXPLAINED = [c for c in un_c if common.is_int_table(W[c][1], c, QW)]
    un_c = [c for c in un_c if c not in set(INT_UNEXPLAINED)]
    INT_HINTS = {c: "same name and size as a table the model computes itself (values not compared)"
                 for c in INT_UNEXPLAINED if (c.rsplit(".", 1)[-1], numel(W[c][0])) in _sigM}
    used = set(used_params) | set(used_buffers)
    for names in alias.values():
        if any(n in used for n in names): used |= set(names)
    tot = sum(numel(W[c][0]) for c in W if c not in set(masks))
    unused = []
    for c, (s, _) in W.items():
        if c in set(masks) or c in set(un_c) or c in set(INT_UNEXPLAINED): continue   # (F05: an unknown, not a weight)
        h = matched.get(c)
        ok = h is not None and ((h in used) if not h.startswith(("fused:", "renamed:")) else any(u.startswith(h.split(":", 1)[1] + ".") for u in used))
        if not ok: unused.append((c, numel(s)))
    # F05: a checkpoint-loaded persistent buffer (any dtype) unread is judged by THE buffer rule (libload.buffer_gaps):
    # a same-kind sibling read -> a real gap (stays a T1 failure); otherwise an R-BUF unknown, never a learned-weight fail
    import libload as _ll
    unused, MATCHER_BUFFER_UNKNOWN = _ll.matcher_buffer_split(unused, matched, _ll.buffer_groups(m), used, PASS_READS)
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
            # credit is proven only if the witness recording is transparent (G4), shown stable on different inputs (G5a)
            # and bound to this revision and the current witness code (G6, F14); otherwise each gap is an unknown
            _wunk = common.witness_execution_credit(WIT, R.get("revision"))
            R["executed_by_witness"] = {"runtime": WIT.get("executed_by"), "vllm_class": WIT.get("vllm_class"),
                                        "vllm_version": WIT.get("vllm_version"), "vllm_source_id": WIT.get("vllm_source_id"),
                                        "revision": WIT.get("revision"), "producer_id": WIT.get("producer_id"),
                                        "tensors": len(witnessed), "notes": WIT.get("notes"),
                                        "credit": "unknown" if _wunk else "proven",
                                        "opaque_ops": WIT.get("opaque_ops") if "op_full" in WIT else "not recorded (old witness record)",
                                        "external_nonscalar": WIT.get("external_nonscalar")}
            for _p, _why in _wunk:
                UNKNOWNS.append({"predicate": _p, "what": "tensors executed only in the vLLM witness, whose recording: " + _why,
                                 "items": witnessed[:20], "count": len(witnessed)})
            unbuilt = []
        else:
            R["executed_by_witness_partial"] = {"covered": len(witnessed), "declared_unbuilt": len(unbuilt)}
    # leftover eligibility for declared-unbuilt tensors (DB-3 reading B): only tensors a runtime is known to run need
    # witness credit; the library's own declaration certifies the rest; a missing / errored record is an unknown
    # (fully witnessed tensors were credited above, with the G4 transparency check, and removed from `unbuilt`; a
    #  partial witness gives no credit here: a tensor that needs runtime credit and is still listed keeps leftovers out)
    _need, _rt_unk = common.runtime_credit_needed(unbuilt, WIT, R.get("revision"))
    no_runtime = not _need                        # (name kept: no remaining declared-unbuilt tensor needs runtime credit)
    R["runtime_credit"] = {"needed": len(_need)}
    if unbuilt and not _need and not _rt_unk and WIT:                          # F14: the scoped statement that certified them
        R["runtime_credit"]["statement"] = (common.no_mtp_statement(WIT) if not WIT.get("vllm_mtp_arch")
                                            else common.loader_statement(WIT, len(unbuilt)))
    if _rt_unk:
        UNKNOWNS.append({"predicate": _rt_unk[0], "what": _rt_unk[1], "items": sorted(unbuilt)[:20], "count": len(unbuilt)})
    unb = sum(numel(W_UNBUILT[k][0]) for k in unbuilt)
    # ---- library authority (safetensors repos): which shipped keys the library consumes, from its own from_pretrained
    #      on sparse header stubs (file choice, renames, merges/splits, ignore lists all the library's). Replaces the
    #      name matcher's "class lacks" and "unused" sets; the matcher's result is kept for comparison.
    LIB, G1, _g2f = None, None, []                 # (_g2f: F04 failure items; none without a library load)
    reported_by_library = None                    # None = no library load report (matcher only): nothing certified
    if R.get("ground_truth_source") in (None, "safetensors") and GGUF_RAW is None:
        try:
            import libload
            # its own budget (sharded repos have hundreds of headers to read), not the optional-stage limit
            # never past the runner's kill (its deadline, on its clock): leave 150 s to finish, grade and save.
            # A repeating timer: if the library swallows one alarm (its conversion ops catch every Exception), the
            # next one fires 5 s later, so the deadline holds.
            _left = int(DEADLINE - time.monotonic() - 150) if DEADLINE else LIBLOAD_S
            if _left < 30:
                raise StageTimeout(f"no time left for the library load report ({_left}s)")
            signal.setitimer(signal.ITIMER_REAL, min(LIBLOAD_S, _left), 5)
            try:
                if _auth_mirror_failed:
                    raise StageTimeout("library mirror already timed out in class selection")
                if AUTH is None:
                    AUTH = libload.Authority(repo); import atexit; atexit.register(AUTH.close)
                _n0 = ALARMS[0]
                LIB = AUTH.report(cls, set(A), config=CFG_LOAD, **_lkw(cls))   # cached only for the same hypothesis (F04)
                if ALARMS[0] != _n0:                             # an alarm the library swallowed: report not trusted
                    LIB = None; raise StageTimeout("alarm during library load")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception as e:
            signal.setitimer(signal.ITIMER_REAL, 0)        # a tick that landed inside the finally above left it armed
            R["library_load_report_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            if network_error(e):                      # the library load lost its connection: not a fact about the model
                fail("FAIL", f"network: library load report unavailable ({network_error(e)}: {str(e)[:120]})", "library load")
        if LIB and "error" in LIB:                 # e.g. no safetensors files: say so instead of looking like a legacy row
            R["library_load_report_error"] = f"not run: {LIB['error']}"
    # what the library load may certify (P0.5): loaded / refused (forced diagnostic evidence) / quantized codes the
    # installed library cannot interpret (not certified: the matcher path below, with the diagnostic kept)
    _AUTH_ST = libload.authority(LIB) if LIB is not None else {"certifies": False, "state": None, "unknown": None, "fail": {}}
    if LIB and "error" not in LIB:
        R["library_load_state"] = LIB.get("load_state")
        if LIB.get("refusal"):
            R["library_load_refusal"] = LIB["refusal"]
        # R-REFUSE: the user's default load = the declared class's (its report is cached from class selection)
        _decl_rep = LIB if cls is DECLARED_CLS else (AUTH.lookup(DECLARED_CLS, _A_aux, CFG_LOAD, **_lkw(DECLARED_CLS))
                                                     if AUTH is not None and _A_aux is not None else None)
        R["library_refuses_checkpoint"] = bool(_decl_rep) and libload.refuses(_decl_rep)
        if R["library_refuses_checkpoint"]:
            _rk = _decl_rep.get("refusal_unexplained_by_quantization") or (_decl_rep.get("refusal") or {}).get("mismatched") or []
            R["library_refusal_keys"] = {"count": len(_rk), "keys": list(_rk)[:10], "class": DECLARED_CLS.__name__}
    if _AUTH_ST["unknown"]:
        R["library_load_report_error"] = "not certified: " + _AUTH_ST["unknown"]
        R["library_load_diagnostic"] = {"state": _AUTH_ST["state"], "quantized_evidence": LIB.get("quantized_evidence"),
                                        "forced_diagnostic_recovery": LIB.get("forced_diagnostic_recovery"),
                                        "mismatched": LIB.get("mismatched", [])[:10], "unexpected": len(LIB.get("unexpected", [])),
                                        "missing": LIB.get("missing", [])[:20], "unaccounted_numel": LIB.get("unaccounted_numel")}
        LIB = None                                 # evidence kept above; it certifies nothing
    if not LIB or "error" in LIB:
        R["t1_authority"] = "name matcher (no library load report)"
        UNKNOWNS.append({"predicate": "G2", "what": "the executed model is not compared with a library-loaded model (no "
                         "certified library load)", "items": [], "count": 0})
        UNKNOWNS.append({"predicate": "G1", "what": "no library load: which shipped tensors the library consumes, and how "
                         "it fills the rest, is not established (" + (R.get("library_load_report_error") or "no report") + ")",
                         "items": [], "count": 0})
    if LIB and "error" not in LIB:
        matcher_lacks = list(un_c)
        matcher_view = {"class_lacks": top_groups([(c, numel(W[c][0])) for c in un_c], 5), "unused": top_groups(unused, 5)}
        # the library names reported keys after ITS renames: test each against its checkpoint source(s)
        _srcs = LIB.get("reported_sources", {})
        _src = lambda k: _srcs.get(k) or [k]
        lib_unexp = [k for k in LIB["unexpected"] if not all(x in set(unbuilt) | set(bufs) for x in _src(k))]
        lib_unexp = [k for k in lib_unexp if not all(x in T and _stored_constant(x, LIB) for x in _src(k))
                     and k not in model_buffers]
        # F05: unexpected integer / bool tables with no evidence of their meaning: an unknown, never hidden, never a
        # learned-weight failure (quantized codes stay learned weights)
        INT_UNEXPLAINED = [k for k in lib_unexp if all(x in T and common.is_int_table(T[x][1], x, QW) for x in _src(k))]
        lib_unexp = [k for k in lib_unexp if k not in set(INT_UNEXPLAINED)]
        _ev = LIB.get("int_table_evidence") or {}
        INT_HINTS = {k: "same name and size as a table the model computes itself (values not compared)"
                     for k in INT_UNEXPLAINED if any(_ev.get(x) == "recomputed" for x in _src(k))}
        MATCHER_BUFFER_UNKNOWN = []                   # the library path judges buffers in g1_assess
        mism = [m_[0] for m_ in LIB["mismatched"]]
        loaded = libload.loaded_params({n for n, _ in m.named_parameters()}, LIB)   # never-shipped params excluded
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
        # G1/G2 load provenance (INTENT.md): how the library filled every tensor it did not load, and whether the
        # executed model is the loaded one
        _sd = m.state_dict(keep_vars=True)
        _pn = {n for n, _ in m.named_parameters(remove_duplicate=False)}
        _bg = libload.buffer_groups(m)             # every persistent buffer, ANY dtype (F05: dtype is not meaning)
        # F04: names / shapes (and the rest of G2) are compared by g2_compare below when the load has an identity
        _has_id = bool(LIB.get("identity")) and not (LIB.get("identity") or {}).get("error")
        G1 = libload.g1_assess(LIB.get("provenance") or {}, used,
                               ((LIB.get("provenance") or {}).get("state_shapes") or {}) if _has_id
                               else {k: list(t.shape) for k, t in _sd.items()}, _bg, PASS_READS)
        try:
            _EID = libload.model_identity(m)
        except Exception as _e:
            _EID = {"error": f"identity: {type(_e).__name__}: {str(_e)[:120]}", "modules": {}}
        _g2u, _g2f, _g2r = libload.g2_decide(LIB.get("identity"), BUILD_IDS, _EID,
                                             {"attn_implementation": BUILD_ATTN, "dtype": "float32", "zero_storage": True})
        UNKNOWNS.extend(_g2u)
        try:                                         # info: what the library's DEFAULT attention would build instead
            with torch.device("meta"):
                _dm = cls._from_config(_copy.deepcopy(CFG_LOAD))
            _dmods = {n or "(root)": f"{type(x).__module__}.{type(x).__qualname__}" for n, x in _dm.named_modules(remove_duplicate=False)}
            _dd = sorted(n for n in _dmods if _EID["modules"].get(n) != _dmods[n])
            _datt = getattr(_dm.config, "_attn_implementation", None); del _dm
        except Exception as _e:
            _dd, _datt = None, f"not evaluated: {type(_e).__name__}"
        R["g2"] = dict(_g2r, load_hypothesis=LIB.get("hypothesis"),
                       executed_build={"attn_implementation": BUILD_ATTN, "dtype": "float32", "zero_storage": True},
                       library_default_attention={"implementation": _datt, "modules_differ": None if _dd is None else len(_dd),
                                                  "examples": (_dd or [])[:5]})
        UNKNOWNS.extend(G1["unknowns"])
        _prov = LIB.get("provenance") or {}
        R["load_provenance"] = {"quantizer": _prov.get("quantizer"),
                                # F05-4 (R-NORM kept): persistent state the library fills itself (constant / deterministic)
                                # because the checkpoint does not ship it, named so nobody reads it as learned
                                "initialised_by_library_not_shipped": libload.initialised_not_shipped(_prov.get("nonloaded")) or None,
                                "nonloaded": {k: v for k, v in list((_prov.get("nonloaded") or {}).items())[:60]},
                                "nonloaded_classes": dict(collections.Counter(v["class"] for v in (_prov.get("nonloaded") or {}).values())),
                                "missing_silenced_by_ignore": _prov.get("missing_silenced_by_ignore"),
                                "unexpected_silenced_by_ignore": _prov.get("unexpected_silenced_by_ignore"),
                                "files_not_opened": _prov.get("files_not_opened"),
                                "raw_tie_word_embeddings": _prov.get("raw_tie_word_embeddings"),
                                "library_tie_word_embeddings": _prov.get("library_tie_word_embeddings"),
                                "fail": G1["fail"], "leftovers": G1["leftovers"]}
        if LIB.get("tie_value_check_unevaluable"):
            # the library ties shipped-twice weights only if their values are equal: not checkable without the weights
            R["library_tie_value_check_unevaluable"] = {"pairs": LIB["tie_value_check_unevaluable"],
                                                       "shipped_tied_tensors": LIB.get("tie_unchecked_shipped") or []}
        R["library_load_report"] = {"files_read": LIB.get("files_read"), "files_ignored": sorted(set(LIB.get("files_all", [])) - set(LIB.get("files_read", []))),
                                    "unaccounted_numel": LIB.get("unaccounted_numel"), "unexpected": len(LIB["unexpected"]), "missing": LIB["missing"][:20], "mismatched": LIB["mismatched"][:10],
                                    "loaded_params": LIB["loaded_params"], "model_params": LIB["model_params"], "stub_disk_kb": LIB["stub_disk_kb"],
                                    "matcher_view": matcher_view}
        W = dict(W, **{k: T[k] for k in un_c if k in T})
        if LIB.get("read_keys"):
            rk = set(LIB["read_keys"])
            R["checkpoint_numel_in_files_library_reads"] = sum(numel(T[k][0]) for k in rk if k in T)
    if INT_UNEXPLAINED:
        UNKNOWNS.append(common.int_table_unknown(INT_UNEXPLAINED, INT_HINTS))
    if MATCHER_BUFFER_UNKNOWN:
        UNKNOWNS.append({"predicate": "G3", "what": "checkpoint-loaded buffers no recorded run read in the modes tried (not "
                         "proven dead: no supported input is shown to leave them unread)",
                         "items": MATCHER_BUFFER_UNKNOWN[:20], "count": len(MATCHER_BUFFER_UNKNOWN)})
    _g1f = dict((G1 or {}).get("fail") or {}, **(_AUTH_ST.get("fail") or {}))   # + a refusal the codes do not explain
    if _g2f:                                         # F04: the executed model is not the loaded one (observed)
        _g1f["executed_differs_from_loaded"] = _g2f
    _g1l = (G1 or {}).get("leftovers") or {}
    _t1_old = not unused and not un_c and not unbuilt
    checks["T1_weights"] = {
        "pass": _t1_old and not _g1f and not _g1l,
        "load_provenance_fail": _g1f, "buffers_never_read_in_any_mode": _g1l.get("buffers_never_read_in_any_mode", [])[:20],
        # leftovers = ONLY what the library certifies BY DESIGN: keys it declares ignorable (class or built-model list)
        # and keys its own conversion silently drops. Merely reported "unexpected" keys do not qualify (the library
        # itself says "not ok if you expect identical arch"), nor does anything only our matcher saw.
        "leftovers_eligible": not _g1f and (((not unused) and bool(un_c or unbuilt) and (not unbuilt or no_runtime)
                              and (not un_c or (reported_by_library is not None and not reported_by_library
                                                and all(c in (R.get("library_silent_drops") or []) for c in un_c)
                                                and not any(str(c).startswith("(unidentified") for c in un_c))))
                                                or (_t1_old and bool(_g1l))),
        "shipped_weights_unused": top_groups(unused, 5), "shipped_weights_unused_numel": sum(n for _, n in unused),
        "shipped_but_not_built_by_model": top_groups([(c, numel((W.get(c) or T.get(c) or ((0,),))[0])) for c in un_c], 5),
        "library_declared_unbuilt_numel": unb, "library_declared_unbuilt_prefixes": sorted({".".join(k.split(".")[:3]) for k in unbuilt})[:5],
        "stored_masks_excluded": len(bufs) + len(masks), "checkpoint_learned_numel": tot + unb}
    # parts list for everything shipped that no run executed: drawn from the checkpoint alone
    # (name + shape + reason), wiring unknown. Grouped by layer pattern; FULL stays strict.
    def _inv(items, reason):
        g = collections.OrderedDict()
        for c in items:
            s = tuple((T.get(c) or W.get(c) or W_UNBUILT.get(c) or [()])[0])
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
# F09 / G4: a non-scalar tensor built from Python values is credited only as a captured constant (no open input / weight
# escape before it; complete values kept) or by the direct-copy contract (dag.py). Anything else is an UNKNOWN on every
# recording that ran: main, modes, entries, generate, synth AND stability (G5a evidence); the training-mode pass only
# when it alone credits tensors (F10-4). Whether the escaped values came from inputs only or from weights never matters
# here (an input-only origin does not show the computation: G5 never excuses G4).
_pbu, _pbx, _pbc = 0, [], collections.Counter()
for l, v in pass_res.items():
    if not v["ok"]:
        continue
    _n = v.get("python_built_unexplained")
    if _n is None:                       # (Opus B1: an ok pass without an F09 record is NOT shown: never 0)
        _n = max(1, v.get("python_mediated_values") or 0)
        v = dict(v, python_built_unexplained_examples=[{"what": "this pass's record carries no F09 classification"}])
    _pbc.update(v.get("python_built") or {})
    if l == "train" and not ((v.get("newly_used_params") or 0) + (v.get("newly_used_buffers") or 0)):
        continue
    if _n:
        _pbu += _n
        _pbx += [dict(e, pass_label=l) for e in (v.get("python_built_unexplained_examples") or [])][:3]
if _pbu:
    UNKNOWNS.append({"predicate": "G4", "what": "tensors built from Python values after a model input or weight value "
                     "was read into Python (or with values not kept): the recording does not show how they were computed",
                     "items": _pbx[:20], "count": _pbu})
checks["T3_closed_dataflow"] = {"pass": ext == 0, "external_nonscalar_tensors": ext, "python_mediated_values_info": pm,
                                "python_built_info": dict(_pbc),
                                "examples": sum((v.get("external_examples", []) for v in pass_res.values() if v["ok"]), [])[:5],
                                "dead_ops_info": {l: v.get("dead_ops") for l, v in pass_res.items() if v["ok"]}}
# F10 / G4: every credited pass (main, modes, entries, generate, synth, train, stability) on the classifier's verdict per
# recorded node (dag.analyse): an opaque op fails T4; an op without a recorded overload, or a decomposition that did not
# reproduce the op's own values, is an UNKNOWN (never a pass, never a failure)
# The training-mode pass (D6 / G5-1 open: "counts toward weight use only") never fails T4: an opaque op there (e.g. a
# loss kernel) makes only the tensors that pass alone credits an UNKNOWN (D-F10-5 option A); otherwise it is info.
opq, _ounk, _odis, _odec = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
_owhy, _uwhy = {}, {}
for l, v in pass_res.items():
    if v["ok"] and l != "train":
        opq.update(v.get("opaque_ops", {})); _ounk.update(v.get("opacity_unknown_ops", {}))
        _odis.update(v.get("decomposition_disagrees", {})); _odec.update(v.get("decomposed_ops", {}))
        for k_, w_ in (v.get("opaque_reasons") or {}).items(): _owhy.setdefault(k_, w_)
        for k_, w_ in (v.get("opacity_unknown_reasons") or {}).items(): _uwhy.setdefault(k_, w_)
_tr = pass_res.get("train") or {}
# F10-4: what the training-mode pass ALONE credits: parameters AND checkpoint (persistent) buffers no other pass read;
# its opaque ops, ops without an overload and disagreeing decompositions all leave those tensors unshown
_tr_opq = dict(_tr.get("opaque_ops") or {}) if _tr.get("ok") else {}
_tr_unk = dict(_tr.get("opacity_unknown_ops") or {}) if _tr.get("ok") else {}
_tr_dis = dict(_tr.get("decomposition_disagrees") or {}) if _tr.get("ok") else {}
_tr_alone = (_tr.get("newly_used_params") or 0) + (_tr.get("newly_used_buffers") or 0) if _tr.get("ok") else 0
checks["T4_no_opaque_ops"] = {"pass": not opq, "opaque_ops": dict(opq), "opaque_reasons": {k_: _owhy.get(k_) for k_ in opq},
                              "decomposed_ops_info": dict(_odec), "train_pass_opaque_ops_info": _tr_opq,
                              **({"train_pass_unknown_ops_info": dict(_tr_unk, **_tr_dis)} if (_tr_unk or _tr_dis) else {})}
if (_tr_opq or _tr_unk or _tr_dis) and _tr_alone > 0:
    UNKNOWNS.append({"predicate": "G4", "what": "tensors credited only by the training-mode pass (parameters "
                     f"{_tr.get('newly_used_params') or 0}, checkpoint buffers {_tr.get('newly_used_buffers') or 0}), whose "
                     "recording's transparency is not shown: " + ("opaque ops " + str(_tr_opq) if _tr_opq else "")
                     + (" no recorded overload " + str(_tr_unk) if _tr_unk else "")
                     + (" decomposition did not reproduce the op " + str(_tr_dis) if _tr_dis else ""),
                     "items": sorted(set(_tr_opq) | set(_tr_unk) | set(_tr_dis))[:20], "count": _tr_alone})
R["vocabulary_notes"] = __import__("opclass").vocabulary_note()   # (F10-24: rejected vocabulary entries: a note)
if _ounk or _odis:
    UNKNOWNS.append({"predicate": "G4", "what": "ops whose transparency is not shown: "
                     + ("not shown (each with its reason) " + str({k_: _uwhy.get(k_, "no recorded overload") for k_ in _ounk})
                        if _ounk else "")
                     + (" decomposition did not reproduce the op " + str(dict(_odis)) if _odis else ""),
                     "items": sorted(set(_ounk) | set(_odis))[:20], "count": len(set(_ounk) | set(_odis))})
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
    # F06-2 (Soumil 2026-10-13): INFO only, never part of the grade until F12 / P2.4 moves T5 onto the detailed record:
    # do the two runs also agree op by op on overload, arguments (canonical form), producer ports and output shapes?
    try:
        if sigs[first][2] is None or sigs["stability:" + first][2] is None:
            raise ValueError("the detailed record of a T5 run was not kept")
        ka = [dag.node_key(n) for n in sigs[first][2]]
        kb = [dag.node_key(n) for n in sigs["stability:" + first][2]]
        dv = next((i for i in range(min(len(ka), len(kb))) if ka[i] != kb[i]), None if len(ka) == len(kb) else min(len(ka), len(kb)))
        checks["T5_stable_structure"]["info_schema2_same"] = ka == kb
        if ka != kb and dv is not None:
            checks["T5_stable_structure"]["info_schema2_first_divergence"] = {
                "index": dv, "a": repr(ka[dv])[:300] if dv < len(ka) else None, "b": repr(kb[dv])[:300] if dv < len(kb) else None}
    except Exception as _e:                              # (info only: it can never change pass / failed)
        checks["T5_stable_structure"]["info_schema2_same"] = None
        checks["T5_stable_structure"]["info_schema2_note"] = f"not comparable: {str(_e)[:160]}"
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
                      "buffers_never_read_in_any_mode": _t1.get("buffers_never_read_in_any_mode") or None,
                      "certification": ("transformers load report" if R.get("t1_authority") == "library_load_report" else "no load report")
                                       + " / declared-ignore list" + (f"; vLLM witness credit ({R['executed_by_witness'].get('credit')}) for the tensors it runs"
                                                                       if R.get("executed_by_witness") else "")
                                       + ("; " + R["runtime_credit"]["statement"] if (R.get("runtime_credit") or {}).get("statement") else "")
                                       + ("; buffers measured never read in any mode (not declared by the library)"
                                          if _t1.get("buffers_never_read_in_any_mode") else "")}
_unr = attempts.unrestored(ATTEMPT_LEDGER)
if _unr:                                  # F13: an attempt changed state that could not be put back afterwards
    UNKNOWNS.append({"predicate": "G2", "what": "an attempt changed model state that could not be put back: the "
                     "attempts after it may not have run the state being tested", "items": _unr[:20], "count": len(_unr)})
R["attempt_ledger"] = attempts.ledger_record(ATTEMPT_LEDGER)
_zs_unk, _zs_notes = __import__("dag").zero_storage_findings([m] if "m" in globals() and isinstance(globals()["m"], torch.nn.Module) else [])      # F07: zero-weight execution faithfulness (G2)
UNKNOWNS.extend(_zs_unk)
if _zs_notes: R["zero_storage_notes"] = _zs_notes
R["revision_pin"] = dict(noweights.PIN_LOG, outcome=noweights.revision_outcome())
_bad = {r_: o for r_, o in R["revision_pin"]["outcome"].items() if not o["ok"]}
if _bad:                                  # what the cache actually holds: a read went to another revision
    UNKNOWNS.append({"predicate": "G6", "what": "the run's cache holds files from another revision than the pinned one",
                     "items": [f"{r_}: snapshots {o['snapshots']} refs {o['refs']}" for r_, o in _bad.items()], "count": len(_bad)})
if noweights.PIN_LOG["other_revision"]:
    UNKNOWNS.append({"predicate": "G6", "what": "some hub reads named another revision than the pinned commit",
                     "items": noweights.PIN_LOG["other_revision"], "count": len(noweights.PIN_LOG["other_revision"])})
if R.get("revision") is None:
    UNKNOWNS.append({"predicate": "G6", "what": "the checkpoint revision could not be pinned ("
                     + (R.get("revision_error") or "unknown") + ")", "items": [], "count": 0})
if R.get("library_refuses_checkpoint") and R["verdict"] not in ("OUT",):
    # R-REFUSE (Soumil 2026-10-13): the installed library refuses its own default load of this checkpoint: a user
    # cannot run it as shipped (checks stay recorded)
    R["verdict"] = "FAIL"
    _rk = R.get("library_refusal_keys") or {}
    R["reason"] = (f"library refuses this checkpoint ({_rk.get('class')}): {_rk.get('count')} mismatched tensors, e.g. "
                   + ", ".join(_rk.get("keys") or [])[:200])
if R["verdict"] in ("FULL", "FULL_LEFTOVERS") and UNKNOWNS:
    # every observed check passes, but a predicate could not be evaluated: never FULL (INTENT.md, Soumil 2026-10-09)
    R["verdict"] = "FULL_UNVERIFIED"
R["unknowns"] = UNKNOWNS
R["failed_checks"] = failed
R["structure"] = {l: {k: v.get(k) for k in ("layer_patterns", "cross_layer_edges", "non_layer_into_layers", "activation_like_ops", "ops", "zero_skipped")}
                  for l, v in pass_res.items() if v["ok"] and not l.startswith("stability:")}
save()
# ---- capture bundle only (after the graded result is saved): the drawn pass again with the cache on, on a fresh
#      model under a time limit; the stored K/V tensors and the ops that produced them
if EMIT and CAP["passes"]:
    _cl = next(iter(CAP["passes"]))
    try:
        import capture
        _ckw = CAP["exec_kw"][_cl]
        clear_library_caches()
        with _limit():
            _cm = _fresh()
            _rc = capture.CaptureRecorder(_cm, _ckw)
            try:
                with torch.no_grad(), math_attention(), _rc:
                    _oc = _cm(**_ckw, use_cache=True)
                CAP["cache_run"] = {"pass": _cl, "graph": capture.graph(_rc, _oc), "cache": capture.cache_slots(_rc, _oc)}
            finally:
                _rc.remove_hooks()
        del _oc, _cm
    except Exception as e:
        CAP["cache_run"] = {"pass": _cl, "error": f"{type(e).__name__}: {str(e)[:200]}"}
_emit()
