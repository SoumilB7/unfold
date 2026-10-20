"""F13 retry isolation controls (EXECUTION_PLAN P2a.3; dossier model-benchmark/_reruns/p2a3_f13). Pure torch, tiny
modules, no downloads, seconds. Drives the REAL worker statements (extracted by AST from worker_v2.py /
worker_v2_diffusion.py) so a fix is tested where it lives.
    python3 selftest_attempts.py [harness_dir]    # exit 0 = every control as expected
NEGATIVE controls (N*): a failed attempt credits nothing and leaves no state behind -> FAIL on HEAD c67a0c80 (pre-fix
proof). VALID controls (V*): legitimate credit is kept -> PASS on HEAD and after the fix.
N12-N15 + W1-W4 cover Soumil's 2026-10-13 rulings (state put back after EVERY attempt, per-attempt recorded seed, router
weights put back, unrestorable writes -> G2 unknown). HEAD proof: run it against an extracted HEAD tree
(`git archive HEAD exec_capture/harness_v2 | tar -x -C <tmp>` then `python3 selftest_attempts.py <tmp>/exec_capture/harness_v2`)."""
import sys, os, ast, time, traceback, inspect, contextlib, warnings
warnings.filterwarnings("ignore")
H = sys.argv[1] if len(sys.argv) > 1 else (os.path.dirname(os.path.abspath(__file__)) if os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker_v2.py")) else
    "/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-exec-capture/exec_capture/harness_v2")
sys.path.insert(0, H)
import torch
torch.set_num_threads(1)
from dag import DagRecorder, analyse, math_attention, clear_library_caches
import class_rules, diffusion_rules
from inputs import mask_positions
SRC = {f: open(os.path.join(H, f)).read() for f in ("worker_v2.py", "worker_v2_diffusion.py")}
TREE = {f: ast.parse(s) for f, s in SRC.items()}
RES = []


def block(f, start, end=None):
    body, src = TREE[f].body, SRC[f]
    i = next(k for k, s in enumerate(body) if start in ast.get_source_segment(src, s))
    j = i if end is None else next(k for k in range(i, len(body)) if end in ast.get_source_segment(src, body[k]))
    return compile(ast.Module(body=body[i:j + 1], type_ignores=[]), f"<{f}:{body[i].lineno}>", "exec")


def fn(f, name):
    n = next(s for s in TREE[f].body if isinstance(s, ast.FunctionDef) and s.name == name)
    return compile(ast.Module(body=[n], type_ignores=[]), f"<{f}:{n.lineno}>", "exec")


try:
    import attempts as _attempts_mod                  # present once F13 lands (only the copy in the harness under test)
    if os.path.dirname(os.path.abspath(_attempts_mod.__file__)) != os.path.abspath(H):
        _attempts_mod = None
except ImportError:
    _attempts_mod = None


def env_for(m, **kw):
    e = dict(attempts=_attempts_mod, ATTEMPT_LEDGER=[], torch=torch, time=time, traceback=traceback, inspect=inspect, DagRecorder=DagRecorder, analyse=analyse,
             math_attention=math_attention, clear_library_caches=clear_library_caches, class_rules=class_rules,
             mask_positions=mask_positions, m=m, R={"steps": {}}, save=lambda: None, EMIT=None, used_params=set(),
             used_buffers=set(), called=set(), pass_res={}, sigs={}, S_={}, _optional_ok=lambda label: True,
             PASS_READS=[], libload_pass=lambda rec: (set(rec.used_params) | set(rec.used_buffers), set(rec.called)),   # F05 worker globals
             _limit=contextlib.nullcontext, TIED=[], TIED_FWD=[], TIED_SUB=[], _skip_tied=lambda w: None,
             _fresh=lambda: type(m)(), cfg=None, variants={})
    e.update(kw); return e


def check(name, cond, detail=""):
    RES.append((name, bool(cond), detail))


def P(*shape): return torch.nn.Parameter(torch.ones(*shape))
x = torch.ones(2)
MAIN = block("worker_v2.py", "def _schedule():", "for label, kw in _schedule():")
RUN_MODE = fn("worker_v2.py", "_run_mode")
EXTRA = block("worker_v2.py", "for label, kw in extra_passes:")
GEN = block("worker_v2.py", 'if hasattr(m, "generate") and class_rules.shared(TIED, "generate")')
UNUSED = fn("worker_v2.py", "_unused_now")
TRAIN = block("worker_v2.py", "sig_f = inspect.signature(m.forward).parameters", "if ((P_names - used_params)")


# ---- N1 main pass: TypeError naming use_cache raised INSIDE the forward after reads; retry without it succeeds -----
class DeepUseCache(torch.nn.Module):
    def __init__(s): super().__init__(); s.failed = P(2); s.good = P(2); s.cache_proj = torch.nn.Linear(2, 2)
    def forward(s, x, **kw):
        if "use_cache" in kw:
            s.cache_proj(x * s.failed); raise TypeError("Inner.forward() got an unexpected keyword argument 'use_cache'")
        return x + s.good
e = env_for(DeepUseCache(), passes=[("text", {"x": x})]); exec(MAIN, e)
check("N1 main _call retry: failed attempt's reads not credited", e["pass_res"]["text"]["ok"] and e["used_params"] == {"good"},
      sorted(e["used_params"]))
check("N1b main _call retry: failed attempt's modules not credited (T2)", "cache_proj" not in e["called"], sorted(e["called"]))
check("N1c main _call retry: graded graph = the successful call only", e["pass_res"]["text"].get("ops") == 1,
      f"ops={e['pass_res']['text'].get('ops')} dead={e['pass_res']['text'].get('dead_ops')}")


# ---- N2 main pass: 5-D pixel attempt reads a temporal weight then raises; squeezed retry succeeds -----------------
class Video(torch.nn.Module):
    def __init__(s): super().__init__(); s.temporal = torch.nn.Linear(4, 4); s.good = P(4)
    def forward(s, pixel_values):
        if pixel_values.dim() == 5:
            s.temporal(pixel_values); raise RuntimeError("expected 4-D input")
        return pixel_values + s.good
e = env_for(Video(), passes=[("image", {"pixel_values": torch.ones(1, 1, 3, 2, 4)})]); exec(MAIN, e)
check("N2 squeezed retry: 5-D attempt's reads not credited", e["used_params"] == {"good"}, sorted(e["used_params"]))
check("N2b squeezed retry: the worker's own squeeze is not recorded as a model op",
      not any(f == "squeeze" for _, f in e["sigs"].get("image", (None, []))[1]), e["sigs"].get("image", (None, []))[1])


# ---- N10 a failed attempt that writes a buffer in place: the retry sees the restored value ----------------------------
SEEN = []
class Counter(torch.nn.Module):
    def __init__(s): super().__init__(); s.good = P(2); s.register_buffer("steps", torch.zeros(1))
    def forward(s, x, **kw):
        s.steps.add_(1); SEEN.append(float(s.steps))       # e.g. a step counter / running statistic
        if "use_cache" in kw: raise TypeError("Inner.forward() got an unexpected keyword argument 'use_cache'")
        return x * s.steps + s.good
mcn = Counter(); e = env_for(mcn, passes=[("text", {"x": x})]); exec(MAIN, e)
check("N10 failed attempt's in-place buffer write is undone before the retry (and the pass leaves the base state)",
      SEEN == [1.0, 1.0] and float(mcn.steps) == 0.0, (SEEN, float(mcn.steps)))
# ---- N11 every attempt is in the ledger: entry point, inputs, flags, success ----------------------------------------
e = env_for(DeepUseCache(), passes=[("text", {"x": x})]); exec(MAIN, e)
_l = e["pass_res"]["text"].get("attempts") or []
check("N11 the pass records both attempts (failed with use_cache=False, then the credited one)",
      [a.get("ok") for a in _l] == [False, True] and _l[0].get("use_cache") is False and _l[1].get("inputs") == {"x": [2]}, _l)


# ---- N3 / N4 mode pass and extra variant: any TypeError triggers a retry in the same recorder ----------------------
class ModeTE(torch.nn.Module):
    def __init__(s): super().__init__(); s.failed = P(2); s.good = P(2)
    def forward(s, x, use_cache=None, output_attentions=False):
        if use_cache is False:
            x * s.failed; raise TypeError("unsupported operand type(s)")
        return x + s.good
e = env_for(ModeTE()); exec(RUN_MODE, e); e["_run_mode"]("mode:output_flags", {"x": x, "output_attentions": True})
check("N3 _run_mode retry: failed attempt's reads not credited", e["used_params"] == {"good"}, sorted(e["used_params"]))
e = env_for(ModeTE(), extra_passes=[("variant", {"x": x})]); exec(EXTRA, e)
check("N4 extra-variant retry: failed attempt's reads not credited", e["used_params"] == {"good"}, sorted(e["used_params"]))


# ---- N5 generate: first input set fails after reads, fallback input set succeeds ---------------------------------
class Gen(torch.nn.Module):
    def __init__(s): super().__init__(); s.failed = P(2); s.good = P(2); s.never = P(2)
    def forward(s, x, y=None): return x + s.good
    def generate(s, x, y=None, **kw):
        if y is not None:
            x * s.failed; raise ValueError("needs a decoder start token")
        return x + s.good
e = env_for(Gen(), passes=[("t", {"x": x}), ("ty", {"x": x, "y": x})], pass_res={"t": {"ok": True}, "ty": {"ok": True}},
            used_params={"good"})
exec(UNUSED, e); exec(GEN, e)
check("N5 generate fallback: failed input set's reads not credited", e["pass_res"]["generate"]["ok"] and "failed" not in e["used_params"],
      sorted(e["used_params"]))


# ---- N6 diffusion: every ladder attempt failed -> nothing credited from the kept (failed) attempt -----------------
def _dstmt(prefix):                                  # one statement of the diffusion grading loop, by its text
    return next(s for s in ast.walk(TREE["worker_v2_diffusion.py"]) if isinstance(s, ast.stmt)
                and ast.get_source_segment(SRC["worker_v2_diffusion.py"], s).startswith(prefix))
try:
    _okst = [_dstmt("rec_ok = ")]                     # (absent before F13)
except StopIteration:
    _okst = []
USED = compile(ast.Module(body=_okst + [_dstmt("ex_ = extra.get(cn)"), _dstmt("recs = "), _dstmt("used = set().union(")],
                          type_ignores=[]), "<worker_v2_diffusion grading: ex_/recs/used>", "exec")
class Comp(torch.nn.Module):
    def __init__(s): super().__init__(); s.a = torch.nn.Linear(2, 2); s.b = torch.nn.Linear(2, 2)
box = torch.nn.ModuleDict({"transformer": Comp()})
def attempt(read, fail):
    r = DagRecorder(box, {}); err = None
    try:
        with torch.no_grad(), r:
            for n in read: getattr(box["transformer"], n)(torch.ones(2))
            if fail: raise RuntimeError("vae decode failed")
    except RuntimeError as ex:
        err = str(ex)
    r.remove_hooks(); return (r, None, err, {}, {})
best = None
for res in (attempt(["a"], True), attempt(["a", "b"], True)):
    best = diffusion_rules.pick_attempt(best, res)
e = {"rec_a": best[0], "stop_a": best[2], "extra": {}, "cn": "transformer", "diffusion_rules": diffusion_rules}; exec(USED, e)
check("N6 diffusion all attempts failed: the kept failed attempt credits nothing", not e["used"], sorted(e["used"]))


# ---- N7 state: a failed pass caches a value on the module; the next pass must not inherit it -----------------------
class Stateful(torch.nn.Module):
    def __init__(s): super().__init__(); s.proj = torch.nn.Linear(2, 2); s.good = P(2); s._memo = None
    def forward(s, x=None, y=None):
        if s._memo is None: s._memo = s.proj(x)
        if y is not None: raise ValueError("bad pass")
        return s._memo + s.good
e = env_for(Stateful(), passes=[("bad", {"x": x, "y": x}), ("ok", {"x": x})]); exec(MAIN, e)
check("N7 failed pass leaves no state: the next pass records what a fresh model computes",
      {"proj.weight", "proj.bias"} <= e["used_params"], sorted(e["used_params"]))


# ---- N8 train: failed label candidates must not advance BatchNorm statistics (state restored between attempts) ----
BN_SEEN = []
class Train(torch.nn.Module):
    def __init__(s): super().__init__(); s.bn = torch.nn.BatchNorm1d(2); s.good = P(2); s.loss_fail = P(2); s.loss_ok = P(2)
    def forward(s, input_ids, labels=None):
        h = s.bn(input_ids.float().reshape(-1, 2) + 1.0) + s.good; BN_SEEN.append(int(s.bn.num_batches_tracked))
        if isinstance(labels, list): (h * s.loss_fail).sum(); raise ValueError("labels must be a tensor")
        return (h * s.loss_ok).sum() if labels is not None else h
mt = Train()
e = env_for(mt, passes=[("t", {"input_ids": torch.arange(4).reshape(2, 2)})], pass_res={"t": {"ok": True}},
            used_params={"good", "bn.weight", "bn.bias"}, used_buffers={"bn.running_mean", "bn.running_var"})
exec(TRAIN, e)
check("N8 train: every candidate starts from the base BatchNorm state (none inherits a failed one's update)",
      bool(BN_SEEN) and set(BN_SEEN) == {1} and int(mt.bn.num_batches_tracked) == 0, (BN_SEEN, int(mt.bn.num_batches_tracked)))
check("V4 train: failed candidates not credited, the successful one is",
      "loss_fail" not in e["used_params"] and "loss_ok" in e["used_params"] and not mt.training, sorted(e["used_params"]))

# ---- F05-3: a training pass a BatchNorm layer cannot run on ONE example is retried as a batch of 2 ------------------------
class BN1(torch.nn.Module):
    def __init__(s): super().__init__(); s.bn = torch.nn.BatchNorm1d(2); s.head = P(2)
    def forward(s, input_ids, labels=None):
        h = s.bn(input_ids.float().reshape(input_ids.shape[0], -1)[:, :2])          # (batch, 2): batch 1 cannot train
        return (h * s.head).sum() if labels is not None else h
mb1 = BN1()
e = env_for(mb1, passes=[("t", {"input_ids": torch.arange(2).reshape(1, 2)})], pass_res={"t": {"ok": True}}, used_params=set(),
            used_buffers=set())
exec(TRAIN, e)
_tr = e["pass_res"].get("train") or {}
check("F05-3 train: a batch-1 BatchNorm failure is retried as a batch of 2, credited, recorded",
      _tr.get("ok") is True and _tr.get("batch") == 2 and "head" in e["used_params"]
      and any(a_.get("batch") == 1 and not a_.get("ok") for a_ in _tr.get("attempts") or []), _tr)



# ---- N9 recorder: an op that raises (caught inside a successful forward) is not a recorded op: no credit ----------
class Caught(torch.nn.Module):
    def __init__(s): super().__init__(); s.w_bad = P(3, 3); s.good = P(2)
    def forward(s, x):
        try: torch.matmul(x, s.w_bad)                      # shape error inside aten: no node is recorded
        except RuntimeError: pass
        return x + s.good
mc = Caught(); r = DagRecorder(mc, {"x": x})
with torch.no_grad(), r: mc(x)
r.remove_hooks()
check("N9 an op that raised credits nothing (node-less read)", "w_bad" not in r.used_params,
      f"used={sorted(r.used_params)} nodes={[n.func for n in r.nodes]}")


# ---- N12 (F13-3) a SUCCESSFUL pass caches a value on the module; the next pass must still record the real path ------
class Memo(torch.nn.Module):
    def __init__(s): super().__init__(); s.proj = torch.nn.Linear(2, 2); s.good = P(2); s._memo = None
    def forward(s, x=None):
        if s._memo is None: s._memo = s.proj(x)
        return s._memo + s.good
e = env_for(Memo(), passes=[("a", {"x": x}), ("b", {"x": x})]); exec(MAIN, e)
_b_mods = {mod for mod, _ in e["sigs"].get("b", (None, []))[1]}
check("N12 a successful pass's cached value is put back: the next pass records proj itself", any(m_.startswith("proj") for m_ in _b_mods),
      sorted(_b_mods))


# ---- N13 (F13-4) every attempt runs from its own recorded seed (reproducible) ----------------------------------------
DRAWN = []
class Noisy(torch.nn.Module):
    def __init__(s): super().__init__(); s.good = P(2)
    def forward(s, x, **kw):
        DRAWN.append(float(torch.rand(1)))
        if "use_cache" in kw: raise TypeError("Inner.forward() got an unexpected keyword argument 'use_cache'")
        return x + s.good
e = env_for(Noisy(), passes=[("text", {"x": x})]); exec(MAIN, e)
_l = e["pass_res"]["text"].get("attempts") or []
_seeds = [a_.get("seed") for a_ in _l]
def _first_draw(sd):
    torch.manual_seed(sd); return float(torch.rand(1))
check("N13 each attempt records its own seed and draws exactly from it",
      len(_seeds) == 2 and None not in _seeds and _seeds[0] != _seeds[1] and DRAWN == [_first_draw(sd) for sd in _seeds], (_seeds, DRAWN))


# ---- N14 (F13 router) the router mode's randomised weights are put back before later passes --------------------------
class Router(torch.nn.Module):
    def __init__(s): super().__init__(); s.router = torch.nn.Linear(2, 2, bias=False); s.good = P(2); s.extra = P(2)
    def forward(s, x, use_cache=None): return s.router(x) + s.good
mr = Router(); torch.nn.init.zeros_(mr.router.weight); _w0 = mr.router.weight.detach().clone(); _p0 = mr.router.weight.data_ptr()
ROUTER = block("worker_v2.py", 'if base and _unused_now() and S_.get("router_params"):')
e = env_for(mr, base={"x": x}, S_={"router_params": ["router.weight"]}, used_params={"good", "router.weight"})
exec(RUN_MODE, e); exec(UNUSED, e); exec(ROUTER, e)
check("N14 the router mode ran on randomised weights and they are put back afterwards",
      e["pass_res"].get("mode:router_random", {}).get("ok") is True and torch.equal(mr.router.weight.detach(), _w0)
      and mr.router.weight.data_ptr() == _p0, (e["pass_res"].get("mode:router_random"), mr.router.weight.detach().tolist()))


# ---- N15 an in-place write into a parameter's own storage is put back by value ------------------------------------
class WritesParam(torch.nn.Module):
    def __init__(s): super().__init__(); s.good = P(2)
    def forward(s, x, **kw):
        with torch.no_grad(): s.good.mul_(2)
        if "use_cache" in kw: raise TypeError("Inner.forward() got an unexpected keyword argument 'use_cache'")
        return x + s.good
e = env_for(WritesParam(), passes=[("text", {"x": x})]); exec(MAIN, e)
_mw = e["m"]
check("N15 an in-place parameter write by an attempt is put back by value (the retry and later passes see the tested value)",
      torch.equal(_mw.good.detach(), torch.ones(2)) and e["pass_res"]["text"].get("ok") is True, _mw.good.tolist())


# ---- B1 (cross-exam): state the first snapshot missed is put back, or reported -------------------------------------
import types as _types
def _iso(model, body, objects=()):
    led = []
    try:
        with _attempts_mod.isolated([model], led, {"entry": "probe"}, objects=objects):
            body()
    except Exception:
        pass
    return led[-1] if led else {}
if not _attempts_mod:
    for _n in ("B1a", "B1b", "B1c", "B1d", "B1e", "B1f", "B1g", "S2", "R2-S1", "S2b", "S6"):
        check(f"{_n} (no attempts module in the harness under test)", False)
else:
    class Holder(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.w = P(3); s.register_buffer("b", torch.zeros(3)); s.cache = {}; s.lst = [torch.zeros(2)]
            s.config = _types.SimpleNamespace()
    hm = Holder()
    _iso(hm, lambda: (hm.w.data.mul_(3), hm.b.data.add_(1)))
    check("B1a writes through .data (parameter and buffer) are put back by value",
          torch.equal(hm.w.detach(), torch.ones(3)) and torch.equal(hm.b, torch.zeros(3)), (hm.w.tolist(), hm.b.tolist()))
    _iso(hm, lambda: hm.cache.__setitem__("bias", torch.ones(2)))
    check("B1b a dict cache filled by an attempt is emptied again (LeViT attention_bias_cache)", hm.cache == {}, list(hm.cache))
    _iso(hm, lambda: hm.lst[0].add_(5))
    check("B1c a tensor held in a list attribute is put back", torch.equal(hm.lst[0], torch.zeros(2)), hm.lst[0].tolist())
    try:
        from transformers import PretrainedConfig
        hm2 = Holder(); hm2.config = PretrainedConfig(); hm2.config.problem_type = None
        _iso(hm2, lambda: setattr(hm2.config, "problem_type", "single_label_classification"))
        check("B1d a config attribute changed in place is put back", hm2.config.problem_type is None, hm2.config.problem_type)
    except ImportError:
        pass
    hm3 = Holder(); hm3.cache = {"k": {"inner": 1}}
    e_ = _iso(hm3, lambda: hm3.cache["k"].__setitem__("inner", 2))
    check("B1e a nested change it cannot put back is REPORTED (state_not_restored -> G2)", any("cache" in n for n in e_.get("state_not_restored", [])),
          e_.get("state_not_restored"))
    sched = _types.SimpleNamespace(timesteps=[1, 2], _step_index=None)
    _iso(hm, lambda: (setattr(sched, "_step_index", 5), sched.timesteps.append(3)), objects=[sched])
    check("B1f an extra object (scheduler / pipeline) is put back", sched._step_index is None and sched.timesteps == [1, 2],
          (sched._step_index, sched.timesteps))
    LV = Holder(); LV.attention_biases = P(4)
    def _lv_forward(x):
        if "b" not in LV.cache: LV.cache["b"] = LV.attention_biases * 1.0
        return x[:1] + LV.cache["b"][:1]
    LV.forward = _lv_forward
    e = env_for(LV, passes=[("a", {"x": x}), ("b", {"x": x})]); exec(MAIN, e)
    check("B1g LeViT-style dict cache: the second pass reads attention_biases itself (F13-3)",
          any(f in ("mul",) for _, f in e["sigs"].get("b", (None, []))[1]), e["sigs"].get("b", (None, []))[1])

    # ---- S2 a stage time limit cannot cut the restore short; the ledger entry is always written ---------------------
    import signal as _signal
    hb = Holder(); _done = []
    _orig = _attempts_mod.Snapshot.restore
    def _slow(self):
        time.sleep(0.2); out = _orig(self); _done.append(1); return out
    _attempts_mod.Snapshot.restore = _slow
    _old_h = _signal.signal(_signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError("stage limit")))
    led, raised = [], None
    try:
        with _attempts_mod.isolated([hb], led, {"entry": "alarm"}):
            hb.b.add_(1); _signal.setitimer(_signal.ITIMER_REAL, 0.05)
    except TimeoutError as ex:
        raised = ex
    finally:
        _attempts_mod.Snapshot.restore = _orig; _signal.setitimer(_signal.ITIMER_REAL, 0); _signal.signal(_signal.SIGALRM, _old_h)
    check("S2 an alarm during the restore waits for it: state put back, ledger written, the limit still raised",
          _done == [1] and torch.equal(hb.b, torch.zeros(3)) and len(led) == 1 and raised is not None, (_done, hb.b.tolist(), led, raised))

    # ---- R2-S1 copy-on-write by the recorder is put back and NOT reported (a note, never a G2 unknown) ---------------
    import lowcost as _lc
    class Big(torch.nn.Module):
        def __init__(s):
            super().__init__()
            with _lc.zero_storage(): s.w = torch.nn.Parameter(torch.empty(64, 64), requires_grad=False)
        def forward(s, x):
            s.w.add_(1.0); return x.sum() + s.w.sum()
    bm = Big(); _ptr0 = bm.w.data_ptr()
    _rb = DagRecorder(bm, {"x": x}); _lb = []
    with _attempts_mod.isolated([bm], _lb, {"entry": "cow"}), torch.no_grad(), _rb:
        bm(x)
    _rb.remove_hooks()
    check("R2-S1 a recorder copy-on-write is put back (registered storage again) and not reported as unrestored",
          bm.w.data_ptr() == _ptr0 and not _lb[-1].get("state_not_restored") and not bool(bm.w.any()), (_lb[-1], bm.w.data_ptr() == _ptr0))

    # ---- S2b an alarm delivered right after the restore marks the attempt failed ------------------------------------
    _old_h = _signal.signal(_signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError("stage limit")))
    _attempts_mod.Snapshot.restore = _slow; led2 = []
    try:
        with _attempts_mod.isolated([Holder()], led2, {"entry": "alarm2"}):
            _signal.setitimer(_signal.ITIMER_REAL, 0.05)
    except TimeoutError:
        pass
    finally:
        _attempts_mod.Snapshot.restore = _orig; _signal.setitimer(_signal.ITIMER_REAL, 0); _signal.signal(_signal.SIGALRM, _old_h)
    check("S2b a time limit delivered after a completed attempt's restore records the attempt as failed",
          [e_.get("ok") for e_ in led2] == [False], led2)

    # ---- S6 a discarded attempt leaves no recorder events -------------------------------------------------------------
    import dag as _dag
    n0 = len(_dag.NONFINITE_EVENTS)
    def _bad():
        _dag.NONFINITE_EVENTS.append("probe"); raise RuntimeError("x")
    _iso(Holder(), _bad)
    check("S6 a failed attempt's recorder events are dropped", len(_dag.NONFINITE_EVENTS) == n0, _dag.NONFINITE_EVENTS[n0:])

# ---- S3/S4 synth path: every candidate and the stability run are attempts, with their ledger ------------------------
SYNTH = block("worker_v2.py", "if not ok_passes and not ROOT_NO_FORWARD:")
class SynthOnly(torch.nn.Module):
    def __init__(s): super().__init__(); s.good = P(4); s.register_buffer("count", torch.zeros(1))
    def forward(s, hidden_states):
        s.count.add_(1); return hidden_states.flatten()[:4] + s.good
ms = SynthOnly()
e = env_for(ms, ok_passes=[], ROOT_NO_FORWARD=False, passes=[], _fresh=lambda: SynthOnly()); exec(SYNTH, e)
_sl = e["pass_res"].get("synth", {}).get("attempts") or []
_s2 = e["pass_res"].get("stability:synth", {}).get("attempts") or []
check("S3 synth: the pass carries its attempts; S4: the stability run is an attempt too (seeded, in the ledger)",
      e["pass_res"].get("synth", {}).get("ok") is True and bool(_sl) and "seed" in _sl[-1] and bool(_s2) and "seed" in _s2[0]
      and float(ms.count) == 0.0, (e["pass_res"].get("synth"), e["pass_res"].get("stability:synth"), float(ms.count)))

# ---- S1 diffusion credit: the first call is evidence only if it completed (behaviour + every site uses it) -----------
check("S1a completed_call: a stopped call gives no recorder, a completed one gives it",
      getattr(diffusion_rules, "completed_call", lambda r, s_: r)("REC", "RuntimeError") is None
      and getattr(diffusion_rules, "completed_call", lambda r, s_: None)("REC", None) == "REC")
_why = getattr(diffusion_rules, "direct_run_reason")
try:
    _w = _why("unet", {}, "RuntimeError: vae", True)
except TypeError:
    _w = _why("unet", {}, "RuntimeError: vae")
check("S1b the direct-run reason says a component ran inside a crashed call (not 'stopped before reaching')",
      "ran inside a pipeline call that later raised" in _w, _w)
WD, W2 = SRC["worker_v2_diffusion.py"], SRC["worker_v2.py"]
_uses = [ast.get_source_segment(WD, n) for n in ast.walk(TREE["worker_v2_diffusion.py"]) if isinstance(n, ast.Name) and n.id == "rec_a"]
_ruses = sorted({n.lineno for n in ast.walk(TREE["worker_v2_diffusion.py"]) if isinstance(n, ast.Name) and n.id == "rec_a"})
_rtext = [WD.splitlines()[i - 1].strip() for i in _ruses]
check("S1c diffusion: rec_a is read ONLY where it is assigned, turned into rec_ok, the direct-run reason evidence and T5",
      len(_rtext) == 5 and _rtext[0].startswith("rec_a, out_a, stop_a") and _rtext[1].startswith("rec_ok = diffusion_rules.completed_call(rec_a, stop_a)")
      and 'getattr(rec_a, "stopped_in"' in _rtext[2] and "rec_a.called" in _rtext[3] and _rtext[4].startswith("sa, sb = norm(rec_a.nodes"), _rtext)
for _gate in ('for n in (rec_ok.nodes if rec_ok else [])', 'for c in (rec_ok.called if rec_ok else ())',
              'if (rec_ok and any(c == cn or c.startswith(cn + ".") for c in rec_ok.called))',
              'sub = [n for n in (rec_ok.nodes if rec_ok else [])'):
    check(f"S1d diffusion gate reads only the completed call: {_gate[:60]}", _gate in WD)
# _attempt marks a stopped call as a failed attempt in the ledger (behaviour)
_ATT = fn("worker_v2_diffusion.py", "_attempt")
if _attempts_mod:
    _cont = torch.nn.ModuleDict({"u": torch.nn.Linear(2, 2)})
    ea = {"attempts": _attempts_mod, "ATTEMPT_LEDGER": [], "container": _cont, "objs": {},
          "_attempt_call": lambda pipe, ck, sig: ("REC", None, "RuntimeError: stopped", {}, {})}
    exec(_ATT, ea); ea["_attempt"](None, {"prompt": "x"}, {"prompt": None})
    check("S1e diffusion _attempt: a stopped pipeline call is a FAILED attempt in the ledger",
          [e_.get("ok") for e_ in ea["ATTEMPT_LEDGER"]] == [False], ea["ATTEMPT_LEDGER"])
else:
    check("S1e diffusion _attempt (no attempts module in the harness under test)", False)
check("W5 reflected public methods run as attempts", "rec, res = attempts.run(m, m, mkw, lambda: getattr(sub, meth)(**mkw), _ledger," in W2)
check("W6 encode/decode entry points run as attempts", "rec, enc = attempts.run(m, m, ekw, _encdec, _ledger," in W2)
check("W1 diffusion: the pipeline call runs in an isolated state window (with the pipeline and scheduler)",
      'with attempts.isolated([container], ATTEMPT_LEDGER, {"entry": "pipeline"' in WD and "objects=[pipe, objs.get(\"scheduler\")]" in WD)
check("W2 diffusion: encode / decode extras and both direct runs are isolated", WD.count("attempts.isolated([container], ATTEMPT_LEDGER") == 5)
check("W3 diffusion: a stopped first call is never credited", "recs = ([rec_ok] if rec_ok else [])" in WD)
check("W4 both workers add the G2 unknown for unrestored state before the FULL promotion",
      0 <= W2.find("attempts.unrestored(ATTEMPT_LEDGER)") < W2.find('if R["verdict"] in ("FULL", "FULL_LEFTOVERS") and UNKNOWNS:')
      and 0 <= WD.find("attempts.unrestored(ATTEMPT_LEDGER)") < WD.find('if R["verdict"] == "FULL" and UNKNOWNS:'))


# ---- valid controls ----------------------------------------------------------------------------------------------
class Fails(torch.nn.Module):
    def __init__(s): super().__init__(); s.failed = P(2); s.good = P(2)
    def forward(s, x=None, y=None):
        if y is not None: x * s.failed; raise ValueError("bad pass")
        return x + s.good
e = env_for(Fails(), passes=[("bad", {"x": x, "y": x}), ("ok", {"x": x})]); exec(MAIN, e)
check("V1 a failed pass is recorded, not credited; the good pass is credited",
      e["pass_res"]["bad"]["ok"] is False and e["used_params"] == {"good"}, sorted(e["used_params"]))
class Mode(torch.nn.Module):
    def __init__(s): super().__init__(); s.good = P(2); s.attn_extra = P(2)
    def forward(s, x, use_cache=None, output_attentions=False):
        return x + s.good + (s.attn_extra if output_attentions else 0)
e = env_for(Mode(), used_params={"good"}); exec(RUN_MODE, e); e["_run_mode"]("mode:output_flags", {"x": x, "output_attentions": True})
check("V2 a legitimate second mode that reads new weights IS credited",
      "attn_extra" in e["used_params"] and e["pass_res"]["mode:output_flags"].get("newly_used_params") == 1, sorted(e["used_params"]))
best2 = None
for res in (attempt(["a", "b"], True), attempt(["a"], False)):
    best2 = diffusion_rules.pick_attempt(best2, res)
e = {"rec_a": best2[0], "stop_a": best2[2], "extra": {}, "cn": "transformer", "diffusion_rules": diffusion_rules}; exec(USED, e)
check("V3 diffusion: the completed attempt is credited, the failed one is not", e["used"] == {"a.weight", "a.bias"}, sorted(e["used"]))
class Plain(torch.nn.Module):
    def __init__(s): super().__init__(); s.good = P(2); s.lin = torch.nn.Linear(2, 2)
    def forward(s, x, use_cache=None): return s.lin(x) + s.good
e = env_for(Plain(), passes=[("text", {"x": x})]); exec(MAIN, e)
check("V5 a pass with no retry is credited exactly", e["used_params"] == {"good", "lin.weight", "lin.bias"} and "lin" in e["called"],
      sorted(e["used_params"]))

for n, ok, d in RES:
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   [{str(d)[:160]}]"))
neg_fail = [n for n, ok, _ in RES if not ok]
print(f"\n{sum(ok for _, ok, _ in RES)}/{len(RES)} controls as expected")
sys.exit(1 if neg_fail else 0)
