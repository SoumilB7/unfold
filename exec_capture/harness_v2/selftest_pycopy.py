"""F09 / P2.2 controls: tensors built from Python values are credited only by observed evidence, never by matching values.

Usage: python3 selftest_pycopy.py [harness_dir] [--only ID,ID] [--json out.json]
  harness_dir defaults to this file's folder. Run it against the HEAD / worktree tree to show they fail the negatives.

Families:
  N  negative: the tensor must stay a G4 UNKNOWN (python_built_unexplained > 0, no external tensor = never PARTIAL)
  P  positive: proven constants (complete values kept) and identity copies keep credit (no unknown, closed dataflow)
  H  recorder hygiene: hooks put back, the harness's own reads are not escapes, program values unchanged
  W  wiring: the worker / witness / diffusion lines (extracted by AST and executed, or pinned by AST)
"""
import sys, os, re, ast, json, pickle, collections, traceback, types
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
H = os.path.abspath(ARGS[0]) if ARGS else os.path.dirname(os.path.abspath(__file__))
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--json=")), None)
sys.path.insert(0, H)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import dag
torch.set_num_threads(1)

RESULTS = collections.OrderedDict()


def control(cid, desc):
    def deco(f):
        if ONLY and cid not in ONLY:
            return f
        try:
            ok, why = f()
        except Exception as e:
            ok, why = False, f"raised {type(e).__name__}: {str(e)[:160]}"
        RESULTS[cid] = {"ok": bool(ok), "desc": desc, "why": why}
        print(f"{'PASS' if ok else 'FAIL'} {cid:4} {desc[:70]:70} {'' if ok else why}")
        return f
    return deco


def record(m, inputs, call=None, rec_inputs=None, py_inputs=None):
    ri = inputs if rec_inputs is None else rec_inputs
    try:
        rec = dag.DagRecorder(m, ri) if py_inputs is None else dag.DagRecorder(m, ri, py_inputs=py_inputs)
    except TypeError as e:                           # (a tree whose recorder takes no py_inputs)
        raise RuntimeError(f"recorder takes no py_inputs: {e}")
    try:
        with torch.no_grad(), rec:
            y = call() if call else m(**inputs)
        a = dag.analyse(rec, y)
    finally:
        rec.remove_hooks()
    return rec, a, y


def unknown(a):
    """(G4 unknown count, external non-scalar). None = this tree keeps no F09 record (every Python-built tensor credited)."""
    return a.get("python_built_unexplained"), a.get("external_nonscalar")


def want_unknown(m, inputs, **kw):
    rec, a, _ = record(m, inputs, **kw)
    u, ext = unknown(a)
    if u is None:
        return False, f"no F09 record: credited today (T3 pass, python_mediated {a.get('python_mediated_values')})"
    if ext:
        return False, f"external tensor (PARTIAL), not an unknown: {a.get('external_examples')}"
    return u > 0, f"unexplained={u} classes={a.get('python_built')} escapes={a.get('escapes_info')}"


def want_credit(m, inputs, extra=None, **kw):
    rec, a, y = record(m, inputs, **kw)
    u, ext = unknown(a)
    if u is None:
        return False, "no F09 record: credit without evidence"
    if u or ext:
        return False, f"unexplained={u} ext={ext} classes={a.get('python_built')} escapes={a.get('escapes_info')}"
    if extra:
        return extra(rec, a, y)
    return True, f"classes={a.get('python_built')}"


class W(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(()))
        self.cfg = [1.0, 2.0, 3.0, 4.0][:n]


def M(fwd, init=None):
    """A tiny module whose forward is `fwd(self, **inputs)`."""
    cls = type("Ctl", (W,), {"forward": lambda self, **kw: fwd(self, **kw)})
    m = cls()
    if init:
        init(m)
    return m


X4 = lambda: torch.tensor([2.0, 3.0, 5.0, 7.0])

# ------------------------------------------------------------------------------------------------ negatives (unknown)
@control("N01", "Python square of an input escape, inputs 0/1 (F09: matches the escape by value)")
def _():
    return want_unknown(M(lambda s, x: torch.tensor([v * v for v in x.tolist()]) + s.w), {"x": torch.tensor([0.0, 1.0, 1.0, 0.0])})

@control("N02", "Python square of an input escape, asymmetric inputs")
def _():
    return want_unknown(M(lambda s, x: torch.tensor([v * v for v in x.tolist()]) + s.w), {"x": X4()})

def _inplace_sq(s, x):
    v = x.tolist()
    for i in range(len(v)):
        v[i] = v[i] * v[i]
    return torch.tensor(v) + s.w

@control("N03", "in-place int square on 0/1: the SAME list object, cached small ints (identity cannot tell)")
def _():
    return want_unknown(M(_inplace_sq), {"x": torch.tensor([0, 1, 1, 0])})

def _tail(s, x):
    v = x.tolist(); v[-1] *= 2
    return torch.tensor(v) + s.w

@control("N04", "4097-element list, only the last value doubled (F09: the 4096-value cap)")
def _():
    return want_unknown(M(_tail), {"x": torch.arange(4097, dtype=torch.float32)})

@control("N05", "Python relu on non-negative inputs (returns the escaped objects themselves)")
def _():
    return want_unknown(M(lambda s, x: torch.tensor([max(v, 0.0) for v in x.tolist()]) + s.w), {"x": X4()})

@control("N06", "list reversal on a palindromic input")
def _():
    return want_unknown(M(lambda s, x: torch.tensor(x.tolist()[::-1]) + s.w), {"x": torch.tensor([1.0, 2.0, 1.0])})

@control("N07", "weight values through numpy (torch.tensor(w.numpy() * 2))")
def _():
    def init(m):
        m.v = torch.nn.Parameter(torch.tensor([0.5, 0.25, 2.0, 1.0]))
    return want_unknown(M(lambda s, x: torch.tensor(s.v.detach().numpy() * 2) + x, init), {"x": X4()})

@control("N08", "input values through numpy and from_numpy")
def _():
    return want_unknown(M(lambda s, x: torch.from_numpy(x.numpy() * 2) + s.w), {"x": X4()})

@control("N09", "as_tensor of Python arithmetic on an escape")
def _():
    return want_unknown(M(lambda s, x: torch.as_tensor([v + 1 for v in x.tolist()]) + s.w), {"x": X4()})

@control("N10", "new_tensor of Python arithmetic on an escape")
def _():
    return want_unknown(M(lambda s, x: x.new_tensor([v + 1 for v in x.tolist()]) + s.w), {"x": X4()})

@control("N11", "weight bytes read through untyped_storage, then a tensor built from them (ASTRA storage escape)")
def _():
    def init(m):
        m.v = torch.nn.Parameter(torch.tensor([0.5, 0.25, 2.0, 1.0]))
    def f(s, x):
        b = [int(s.v.untyped_storage()[i]) for i in range(4)]
        return torch.tensor(b + [0], dtype=torch.float32)[:4] + x
    return want_unknown(M(f, init), {"x": X4()})

@control("N12", "input values read through repr(), parsed, rebuilt")
def _():
    def f(s, x):
        vals = [float(t) for t in re.findall(r"\d+\.\d+", repr(x))]
        return torch.tensor(vals) + s.w
    return want_unknown(M(f), {"x": torch.tensor([2.5, 3.5, 5.5, 7.5])})

@control("N13", "direct copy spelling with a CALL between (dtype=getattr(...)): contract not shown")
def _():
    return want_unknown(M(lambda s, x: torch.tensor(x.tolist(), dtype=getattr(torch, "float32")) + s.w), {"x": X4()})

def _stored(s, x):
    v = x.tolist()
    return torch.tensor(v) + s.w

@control("N14", "tolist stored in a variable, then copied (unchanged, but no Python code is shown not to touch it)")
def _():
    return want_unknown(M(_stored), {"x": X4()})

@control("N15", "direct spelling that CASTS (float64 escape -> float32 tensor): not an identity copy")
def _():
    return want_unknown(M(lambda s, x: torch.tensor(x.tolist()) + s.w), {"x": torch.tensor([2.0, 3.0, 5.0, 7.0], dtype=torch.float64)})

def _cfg_after_item(s, x):
    if bool((x > 0).any()):              # an input value read into Python (the whisper / encoder-decoder pattern)
        pass
    return torch.tensor(s.cfg) + x

@control("N16", "config list AFTER an input value was read into Python (barrier: not shown independent)")
def _():
    return want_unknown(M(_cfg_after_item), {"x": X4()})

@control("N17", "model input given as a Python list, arithmetic in Python")
def _():
    m = M(lambda s, x, sizes: torch.tensor([v * 2.0 for v in sizes]) + x)
    return want_unknown(m, {"x": X4(), "sizes": [1.0, 2.0, 3.0, 4.0]})

@control("N18", "Tensor.__dlpack__ escape, then a built tensor")
def _():
    def f(s, x):
        cap = x.__dlpack__()
        return torch.tensor([float(v) for v in np.from_dlpack(x)]) + s.w
    return want_unknown(M(f), {"x": X4()})

@control("N19", "a value-set output shape (nonzero), then a tensor built from it")
def _():
    def f(s, x):
        n = (x > 2.5).nonzero().shape[0]
        return torch.tensor([float(n), 1.0, 1.0, 1.0]) + x
    return want_unknown(M(f), {"x": X4()})

@control("N20", "a checkpoint (persistent) buffer read into Python, then a tensor built")
def _():
    def init(m):
        m.register_buffer("scale", torch.tensor([1.0, 2.0, 3.0, 4.0]))
    return want_unknown(M(lambda s, x: torch.tensor([v * 2 for v in s.scale.tolist()]) + x, init), {"x": X4()})

@control("N21", "torch.utils.dlpack.to_dlpack escape, then a built tensor")
def _():
    def f(s, x):
        cap = torch.utils.dlpack.to_dlpack(x)
        y = torch.utils.dlpack.from_dlpack(cap)
        return torch.tensor([1.0, 2.0, 3.0, 4.0]) + x
    return want_unknown(M(f), {"x": X4()})

@control("N22", "pickle (Tensor.__reduce_ex__) escape, then a built tensor")
def _():
    def f(s, x):
        blob = pickle.dumps(x)
        return torch.tensor([float(len(blob) % 7), 1.0, 1.0, 1.0]) + x
    return want_unknown(M(f), {"x": X4()})

@control("N23", "a static constant above the value cap is not evidence (values not kept)")
def _():
    old = dag.PY_CONST_CAP if hasattr(dag, "PY_CONST_CAP") else None
    if old is None:
        return False, "no PY_CONST_CAP: values never kept"
    dag.PY_CONST_CAP = 3
    try:
        return want_unknown(M(lambda s, x: torch.tensor(s.cfg) + x), {"x": X4()})
    finally:
        dag.PY_CONST_CAP = old

@control("N24", "tolist called through a stored bound method (f = x.tolist; torch.tensor(f()))")
def _():
    def f(s, x):
        g = x.tolist
        return torch.tensor(g()) + s.w
    return want_unknown(M(f), {"x": X4()})

@control("N25", "a single-value tensor from an escape is reported (info), not graded (INTENT: non-scalar)")
def _():
    rec, a, _ = record(M(lambda s, x: x * torch.tensor(x.sum().item() ** 2) + s.w), {"x": X4()})
    n = a.get("python_built_scalar_unexplained_info")
    return n == 1 and a.get("python_built_unexplained") == 0, f"scalar info={n} graded={a.get('python_built_unexplained')}"

# --------------------------------------------------------------------------------------------- positives (credited)
def _values_kept(expect):
    def chk(rec, a, y):
        pc = getattr(rec, "pyconst", None)
        if not pc:
            return False, "no complete constant values kept"
        got = [v for (_, _, vals) in pc.values() for v in vals.tolist()]
        return got == expect, f"kept {len(got)} values (first {got[:4]})"
    return chk

@control("P01", "config list: static, its COMPLETE values kept")
def _():
    return want_credit(M(lambda s, x: torch.tensor(s.cfg) + x), {"x": X4()}, _values_kept([1.0, 2.0, 3.0, 4.0]))

@control("P02", "shape numbers: static")
def _():
    return want_credit(M(lambda s, x: torch.tensor([x.shape[0], x.shape[0] * 2, 1, 1], dtype=torch.float32) + x), {"x": X4()})

def _edge_is(kind, name):
    def chk(rec, a, y):
        cp = [(i, p) for i, p in getattr(rec, "pyb", {}).items() if p["class"] == "copy"]
        if len(cp) != 1:
            return False, f"copies {cp}"
        i, p = cp[0]
        e = rec.nodes[i].ins[0]
        sealed = [x["sealed"] for x in rec.escapes]
        return (e[0] == kind and (name is None or e[1] == name) and all(sealed)), f"edge {e} sealed {sealed}"
    return chk

@control("P03", "direct identity copy torch.tensor(x.tolist()): the edge IS the input; escape sealed")
def _():
    return want_credit(M(lambda s, x: torch.tensor(x.tolist()) + s.w), {"x": X4()}, _edge_is("input", "x"))

@control("P04", "direct copy with load-only keyword arguments (dtype, device)")
def _():
    return want_credit(M(lambda s, x: torch.tensor(x.tolist(), dtype=torch.float32, device=x.device) + s.w), {"x": X4()},
                       _edge_is("input", "x"))

@control("P05", "copies made by recorded ops (clone, torch.tensor(tensor)): no Python-built tensor at all")
def _():
    def chk(rec, a, y):
        return not getattr(rec, "pyb", {1: 1}) or not rec.pyb, f"pyb {getattr(rec, 'pyb', None)}"
    return want_credit(M(lambda s, x: torch.tensor(x * 2) + x.clone() + s.w), {"x": X4()}, chk)

@control("P06", "escape of a factory-built tensor (arange from shapes) -> Python arithmetic: static")
def _():
    return want_credit(M(lambda s, x: torch.tensor([v * 2.0 for v in torch.arange(x.shape[0]).tolist()]) + x), {"x": X4()})

def _seal_then_cfg(s, x):
    y = torch.tensor(x.tolist())
    return torch.tensor(s.cfg) + y + s.w

@control("P07", "a sealed direct copy does not poison a later config constant")
def _():
    def chk(rec, a, y):
        return a.get("python_built") == {"copy": 1, "static": 1}, f"classes {a.get('python_built')}"
    return want_credit(M(_seal_then_cfg), {"x": X4()}, chk)

@control("P08", "a 5000-value config constant: ALL values kept (no 16-value snippet, no 4096 cap)")
def _():
    big = [float(i % 97) for i in range(5000)]
    def init(m):
        m.big = big
    return want_credit(M(lambda s, x: torch.tensor(s.big).sum() + x, init), {"x": X4()}, _values_kept(big))

@control("P09", "a non-persistent buffer (built at construction) read into Python: static after")
def _():
    def init(m):
        m.register_buffer("inv", torch.tensor([1.0, 0.5, 0.25, 0.125]), persistent=False)
    return want_credit(M(lambda s, x: torch.tensor([v * 2 for v in s.inv.tolist()]) + x, init), {"x": X4()})

@control("P10", "2-D direct identity copy (nested lists)")
def _():
    return want_credit(M(lambda s, x: torch.tensor(x.tolist()) + s.w), {"x": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
                       _edge_is("input", "x"))

@control("P11", "direct copy of a weight-derived tensor: G4 credit (closed edge); weight use is G5b's question")
def _():
    def init(m):
        m.v = torch.nn.Parameter(torch.tensor([0.5, 0.25, 2.0, 1.0]))
    return want_credit(M(lambda s, x: torch.tensor((s.v * 2).tolist()) + x, init), {"x": X4()}, _edge_is("op", None))

# ------------------------------------------------------------------------------------------------- recorder hygiene
@control("H01", "hooks put back after the recording, also after an exception inside it")
def _():
    names = ("tolist", "numpy", "__array__", "__dlpack__", "__repr__", "__reduce_ex__", "new_tensor")   # (__repr__: never patched)
    before = {n: getattr(torch.Tensor, n) for n in names}
    tb = (torch.tensor, torch.as_tensor, torch.asarray, torch.utils.dlpack.to_dlpack)
    def boom(s, x):
        x.tolist(); raise ValueError("inside")
    try:
        record(M(boom), {"x": X4()})
    except ValueError:
        pass
    after = {n: getattr(torch.Tensor, n) for n in names}
    ta = (torch.tensor, torch.as_tensor, torch.asarray, torch.utils.dlpack.to_dlpack)
    return before == after and tb == ta and getattr(dag.DagRecorder, "_install_hooks", None) is not None, \
        f"restored {before == after and tb == ta}"

@control("H02", "the harness's own value reads (F10 value checks on decomposed ops) are not escapes")
def _():
    def f(s, x):
        y = torch.stack([x, x * 2]).sum(0) + torch.nn.functional.batch_norm(x.reshape(1, 4), torch.zeros(4), torch.ones(4))
        return torch.tensor(s.cfg) + y
    rec, a, _ = record(M(f), {"x": X4()})
    esc = getattr(rec, "escapes", None)
    if esc is None:
        return False, "no escape log"
    return esc == [] and a.get("python_built") == {"static": 1} and sum(getattr(rec, "decomposed", {}).values()) > 0, \
        f"escapes {esc} classes {a.get('python_built')} decomposed {dict(getattr(rec, 'decomposed', {}))}"

@control("H03", "program values unchanged by the hooks (each control's output == a plain run)")
def _():
    cases = [(M(lambda s, x: torch.tensor([v * v for v in x.tolist()]) + s.w), {"x": X4()}),
             (M(lambda s, x: torch.tensor(x.tolist(), dtype=torch.float32) + s.w), {"x": X4()}),
             (M(lambda s, x: torch.from_numpy(x.numpy() * 2) + s.w), {"x": X4()}),
             (M(_seal_then_cfg), {"x": X4()})]
    bad = []
    for m, kw in cases:
        with torch.no_grad():
            plain = m(**kw)
        _, _, y = record(m, kw)
        if not torch.equal(plain, y):
            bad.append(str(m.forward))
    return not bad, f"differ: {bad}"

@control("H04", "an escape channel that could not be watched: static is not shown (only copies stay credited)")
def _():
    m = M(lambda s, x: torch.tensor(s.cfg) + x)
    x = X4()
    rec = dag.DagRecorder(m, {"x": x})
    if not hasattr(rec, "escape_hooks_ok"):
        return False, "no hook-completeness flag"
    try:
        with torch.no_grad(), rec:
            y = m(x=x)
        rec.escape_hooks_ok = False
        a = dag.analyse(rec, y)
    finally:
        rec.remove_hooks()
    return a.get("python_built_unexplained") == 1, f"unexplained {a.get('python_built_unexplained')}"

@control("H06", "a rejected F09-7 vocabulary entry: an escape channel is not watched, so static is not shown")
def _():
    if not hasattr(dag, "F09_VOCAB_ERRORS"):
        return False, "no F09-7 vocabulary"
    old = list(dag.F09_VOCAB_ERRORS)
    dag.F09_VOCAB_ERRORS.append("selftest: a rejected entry")
    try:
        return want_unknown(M(lambda s, x: torch.tensor(s.cfg) + x), {"x": X4()})
    finally:
        dag.F09_VOCAB_ERRORS[:] = old


@control("H07", "the recording is unchanged by F09's bookkeeping: a factory's output is not handed back through an extra detach")
def _():
    rec, a, _ = record(M(lambda s, x: torch.arange(4) + torch.zeros(4) + x + s.w), {"x": X4()})
    f = [n.func for n in rec.nodes]
    return f == ["arange", "zeros", "add", "add", "add"], f"{f}"


_F09_OP = []

def _pylist_op():
    """A custom op whose Python kernel reads its operand's values (it runs INSIDE the recorder's handler)."""
    if not _F09_OP:
        @torch.library.custom_op("f09selftest::pylist", mutates_args=())
        def pylist(x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([v * 1.0 for v in x.tolist()], dtype=x.dtype)
        _F09_OP.append(pylist)
    return _F09_OP[0]

@control("H05", "a value read made inside the recorder's own handler (a Python op kernel) is not a program escape")
def _():
    op = _pylist_op()
    def f(s, x):
        y = op(x)
        return torch.tensor(s.cfg) + y
    rec, a, _ = record(M(f), {"x": X4()})
    esc = getattr(rec, "escapes", None)
    if esc is None:
        return False, "no escape log"
    return esc == [] and a.get("python_built") == {"static": 1}, f"escapes {esc} classes {a.get('python_built')}"

# ------------------------------------------------------------------------------------------- round 2 (dual cross-exam)
from typing import List
import io, copy as _copymod


@torch.jit.script
def _ts_square(y: torch.Tensor) -> torch.Tensor:          # (Fable E18: TorchScript fills torch.tensor in C++)
    v: List[float] = y.tolist()
    return torch.tensor([a * a for a in v])


def _cls_of(rec):
    return sorted(p["class"] for p in getattr(rec, "pyb", {}).values())

# B3 / Fable B1: tensors inside a constructor's data
@control("R01", "B3: torch.tensor([x[0]*2, x[1], x[2], x[3]]) is ASSEMBLED: its edges are those tensors (never a static constant)")
def _():
    def chk(rec, a, y):
        p = [(i, q) for i, q in rec.pyb.items() if q["class"] == "assembled"]
        if len(p) != 1:
            return False, f"classes {_cls_of(rec)}"
        ins = rec.nodes[p[0][0]].ins
        return len(ins) == 4 and all(e[0] == "op" for e in ins), f"edges {ins}"
    return want_credit(M(lambda s, x: torch.tensor([x[0] * 2, x[1], x[2], x[3]]) + s.w), {"x": X4()}, chk)

@control("R02", "B3: as_tensor / new_tensor / legacy torch.Tensor of tensor elements are assembled too (Qwen2-VL [[t,h,w,1]])")
def _():
    def grid(s, x):
        thw = (x > 2.5).long() * 3 + 1
        t, h, w = thw[0], thw[1], thw[2]
        return torch.tensor([[t, h, w, 1]]).float().reshape(4) + torch.as_tensor([x[0], x[1], x[2], x[3]]) + \
            x.new_tensor([x[0], x[1], x[2], x[3]]) + torch.Tensor([x[0], x[1], x[2], x[3]])
    def chk(rec, a, y):
        c = _cls_of(rec)
        return c.count("assembled") == 4 and "static" not in c, f"classes {c}"
    return want_credit(M(grid), {"x": X4()}, chk)

@control("R03", "B3: a Python number from a tainted read mixed into a constructor with tensors -> unknown")
def _():
    return want_unknown(M(lambda s, x: torch.tensor([x.max().item() * 2, x[1], x[2], x[3]]) + s.w), {"x": X4()})

@control("R04", "B3: elements from config-built tensors only (Fable E39) stay credited")
def _():
    return want_credit(M(lambda s, x: torch.tensor([torch.arange(4.0)[i] for i in range(4)]) + x), {"x": X4()})

@control("R05", "B3: a typed legacy constructor (torch.LongTensor([...]): not hookable) is UNWATCHED -> unknown")
def _():
    return want_unknown(M(lambda s, x: torch.LongTensor([1, 2, 3, 4]).float() + x), {"x": X4()})

@control("R06", "B4: torch.tensor(G.tolist()) of a tensor the recording never made is external like a direct use")
def _():
    G = torch.tensor([1.5, -2.0, 3.25, 4.0])
    rec, a, _ = record(M(lambda s, x: torch.tensor(G.tolist()) * x + s.w), {"x": X4()})
    return (a.get("external_nonscalar") or 0) >= 1, f"external {a.get('external_nonscalar')} classes {_cls_of(rec)}"

class _Stale(torch.nn.Module):
    """Replaces 4 tensor attributes and a non-persistent buffer during forward (dynamic-RoPE style): the freed ids are
    handed to the next tensors CPython makes (the raw tensors of the builds)."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(()))
        for i in range(4):
            setattr(self, f"c{i}", torch.zeros(4))
        self.register_buffer("inv", torch.ones(4), persistent=False)
    def forward(self, x):
        out = self.w
        self.register_buffer("inv", x + 1, persistent=False)
        for i in range(4):
            setattr(self, f"c{i}", x * (i + 2))
            out = out + torch.tensor([v * v for v in x.tolist()])
        return out

@control("R07", "B5: replaced attributes / buffers never lend their ids to a later build (20 trials, all 4 builds unknown)")
def _():
    bad = 0
    for _t in range(20):
        rec, a, _ = record(_Stale(), {"x": X4()})
        lf = [n for n in rec.nodes if n.func.startswith("lift_fresh")]
        if any(n.ins[0][0] in ("module_constant", "buffer", "param", "input") for n in lf) or a.get("python_built_unexplained") != 4:
            bad += 1
    return bad == 0, f"{bad}/20 trials misnamed or credited"

# B2: taint through writes into views / bases
def _x7b(s, x):
    buf = torch.zeros(2, 4); buf[0, :3] = x[:3]
    return torch.tensor([a * 2 for a in buf[0].tolist()]) + s.w

def _x7c(s, x):
    padded = torch.zeros(1, 4); padded[0, :2] = (x[:2] > 2.5).float()
    n = (padded != 0).any(dim=0).nonzero().shape[0]
    return torch.tensor([n * 1.0, n * 2.0]) + s.w

def _a4(s, x):
    t = torch.zeros(4); v = t.view(4)
    t.add_(x)
    return torch.tensor([a * a for a in v.tolist()]) + s.w

def _a4c(s, x):                           # read through a NEW tensor computed from the written view's base
    buf = torch.zeros(2, 4); buf[0, :3] = x[:3]
    y = buf * 2
    return torch.tensor([a + 1 for a in y[0].tolist()]) + s.w

@control("R08", "B2: buf[0,:3] = x[:3]; buf[0].tolist() -> unknown (Opus X7b)")
def _():
    return want_unknown(M(_x7b), {"x": X4()})

@control("R09", "B2: slice-written buffer -> nonzero shape -> rebuilt (BLT shape, Opus X7c) -> unknown")
def _():
    return want_unknown(M(_x7c), {"x": X4()})

@control("R10", "B2: a view taken BEFORE an in-place write of its base, then read (Opus A4) -> unknown")
def _():
    return want_unknown(M(_a4), {"x": X4()})

@control("R11", "B2: a tensor computed from a view-written base, then read -> unknown")
def _():
    return want_unknown(M(_a4c), {"x": X4()})

@control("R12", "B2 nearby valid: a buffer written only from config values, then read -> static")
def _():
    def f(s, x):
        buf = torch.zeros(2, 4); buf[0, :3] = torch.arange(3.0)
        return torch.tensor([a * 2 for a in buf[0].tolist()]) + x
    return want_credit(M(f), {"x": X4()})

# B6 / F09-5: builds with a dimension, and tensors assembled from single values
def _a1(s, x):
    out = torch.zeros(4)
    for i, v in enumerate(x.tolist()):
        out[i] = v * v
    return out + s.w

def _a3b(s, x):
    out = torch.zeros(4); v = x.tolist()
    out[0] = v[0] * v[1]
    return out + s.w

@control("R13", "F09-5: out[i] = v*v from an input read (setitem assembly, Opus A1) -> unknown")
def _():
    return want_unknown(M(_a1), {"x": X4()})

@control("R14", "F09-5: stack of 0-d builds from an input read (Opus A2) -> unknown")
def _():
    return want_unknown(M(lambda s, x: torch.stack([torch.tensor(v * v) for v in x.tolist()]) + s.w), {"x": X4()})

@control("R15", "F09-5: cat of [1]-shaped builds (Opus A2b) -> unknown")
def _():
    return want_unknown(M(lambda s, x: torch.cat([torch.tensor([v * v]) for v in x.tolist()]) + s.w), {"x": X4()})

@control("R16", "F09-5: out[0] = v0*v1 (Opus A3b) -> unknown")
def _():
    return want_unknown(M(_a3b), {"x": X4()})

@control("R17", "F09-5: a [1, 1] build from an input value (whisper token pattern, Fable E30) -> unknown")
def _():
    return want_unknown(M(lambda s, x: torch.tensor([[int(x.argmax())]]).float().sum() + x), {"x": X4()})

@control("R18", "F09-5 nearby valid: out[i] = 3.0 config values, an empty [0] placeholder, torch.tensor([x.max()]) (edge)")
def _():
    def f(s, x):
        out = torch.zeros(4)
        for i in range(4):
            out[i] = 3.0                     # config single values placed BEFORE any read: static
        n = int(x.sum().item())              # a tainted read
        e = torch.LongTensor([]).float()     # an empty placeholder after it (unwatched constructor): no value, not graded
        return out + torch.tensor([x.max()]) + e.sum() + x + n * 0
    def chk(rec, a, y):
        return a.get("python_built_scalar_unexplained_info") == 0, f"classes {_cls_of(rec)} info {a.get('python_built_scalar_unexplained_info')}"
    return want_credit(M(f), {"x": X4()}, chk)

def _chain(s, x):
    out = torch.zeros(4); v = x.tolist()
    u = torch.tensor(v[0]) * 2 + 1            # scalar arithmetic on a single value the recording cannot show
    out[0] = u
    return out + s.w

@control("R34", "F09-5: scalar tensor arithmetic on such a value, then placed (out[0] = t*2+1) -> unknown")
def _():
    return want_unknown(M(_chain), {"x": X4()})

@control("R35", "F09-5: torch.tensor([torch.tensor(v*v) ...]) - 0-d builds as constructor leaves -> unknown")
def _():
    return want_unknown(M(lambda s, x: torch.tensor([torch.tensor(v * v) for v in x.tolist()]) + s.w), {"x": X4()})

@control("N23b", "a constructor whose data exceeds the walk cap is not shown, even with tensor elements (never assembled)")
def _():
    old = dag.PY_CONST_CAP
    dag.PY_CONST_CAP = 3
    try:
        return want_unknown(M(lambda s, x: torch.tensor([x[0], x[1], x[2], x[3]]) + s.w), {"x": X4()})
    finally:
        dag.PY_CONST_CAP = old

def _idx_cfg(s, x):
    idx = [0, 2, 3]                          # a Python list index (C++ makes the index tensor: no constructor)
    e = x.reshape(1, 4, 1)
    i = 2
    return x[idx].sum() + e[:, [i], :].sum() + x     # (+ omdet's form: a tuple holding slices and an inline list)

def _idx_read(s, x):
    idx = [int(v) % 4 for v in x.tolist()]   # index values from an input read
    return x[idx] + s.w

class _PropIdx:
    @property
    def idx(self):                           # a property: code runs on access, never run by the recorder
        return [0, 1, 2, 3]

_PROP = _PropIdx()

def _idx_call(s, x):
    return x[_PROP.idx] + s.w                # a property index: not resolvable from the frame (stays unwatched)

@control("R36", "list INDEX of config numbers (x[idx], whisper init_tokens[batch_idx_map] shape) -> static, credited")
def _():
    def chk(rec, a, y):
        return a.get("python_built") == {"static": 2}, f"classes {a.get('python_built')}"
    return want_credit(M(_idx_cfg), {"x": X4()}, chk)

@control("R37", "list INDEX built from an input read -> unknown; a PROPERTY index (not resolvable: never run) -> unwatched (unknown)")
def _():
    r1 = want_unknown(M(_idx_read), {"x": X4()})
    rec, a, _ = record(M(_idx_call), {"x": X4()})
    return r1[0] and _cls_of(rec) == ["unwatched"], f"read {r1}; call-result index {_cls_of(rec)}"

def _bcast(s, x):
    v = x.tolist()
    out = torch.zeros(4); out[:] = torch.tensor(v[0] * 2)     # one value broadcast into every position
    o2 = torch.zeros(4).fill_(torch.tensor(v[1] * 3))
    o3 = torch.zeros(2, 4); o3[torch.zeros(4, dtype=torch.long), torch.arange(4)] = torch.tensor(v[2])   # index_put_, many positions
    o4 = torch.zeros(4); o4.copy_(torch.tensor(v[3]))                # copy_ broadcasting one value
    return out + o2 + o3.sum(0) + o4 + s.w

@control("R38", "F09-5 boundary: ONE such value broadcast into many positions (fill_, out[:] = t, index_put_) stays F09-3 info")
def _():
    rec, a, _ = record(M(_bcast), {"x": X4()})
    return a.get("python_built_unexplained") == 0 and (a.get("python_built_scalar_unexplained_info") or 0) >= 4, \
        f"graded {a.get('python_built_unexplained')} classes {_cls_of(rec)} info {a.get('python_built_scalar_unexplained_info')}"

@control("R19", "F09-5 boundary: a true 0-d value from a read used in arithmetic stays info (F11), never graded")
def _():
    rec, a, _ = record(M(lambda s, x: x * torch.tensor(x.sum().item() ** 2) + s.w), {"x": X4()})
    return a.get("python_built_unexplained") == 0 and a.get("python_built_scalar_unexplained_info") == 1, \
        f"graded {a.get('python_built_unexplained')} info {a.get('python_built_scalar_unexplained_info')}"

@control("R20", "F09-3 (Opus S1): numeric arguments while a tainted read is open are COUNTED (info)")
def _():
    rec, a, _ = record(M(lambda s, x: torch.full((4,), x.sum().item() ** 2) + s.w), {"x": X4()})
    n = a.get("python_scalar_args_while_open_info")
    return bool(n) and a.get("python_built_unexplained") == 0, f"count {n}"

# S2: number data passed in as inputs, on every pass
def _sizes(s, x, sizes):
    return torch.tensor([v * 2.0 for v in sizes]) + x

@control("R21", "S2: a list input invisible to rec_inputs (tensor-only passes) is still data (py_inputs) -> unknown")
def _():
    x = X4()
    return want_unknown(M(_sizes), {"x": x, "sizes": [1.0, 2.0, 3.0, 4.0]}, rec_inputs={"x": x},
                        py_inputs={"x": x, "sizes": [1.0, 2.0, 3.0, 4.0]})

@control("R22", "S2: a dict holding a number list, and a numeric ndarray inside a list, are data -> unknown")
def _():
    m1 = M(lambda s, x, meta: torch.tensor([v * 2.0 for v in meta["sizes"]]) + x)
    m2 = M(lambda s, x, arr: torch.tensor(arr[0]) * 2 + x)
    r1 = want_unknown(m1, {"x": X4(), "meta": {"sizes": [1.0, 2.0, 3.0, 4.0]}})
    r2 = want_unknown(m2, {"x": X4(), "arr": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)]})
    return r1[0] and r2[0], f"dict {r1} ndarray {r2}"

# F09-6: memory with no recorded write
def _load(s, x):
    buf = io.BytesIO(); torch.save(x * 2, buf); buf.seek(0)
    return torch.load(buf, weights_only=True) + s.w

def _from_buffer_cfg(s, x):
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
    st = torch.UntypedStorage.from_buffer(b, dtype=torch.uint8)
    return torch.empty(0, dtype=torch.float32).set_(st, 0, (4,), (1,)) + x

@control("R23", "F09-6: TorchScript torch.tensor(list) = empty + C++ fill, then read (Fable E18) -> unknown")
def _():
    return want_unknown(M(lambda s, x: _ts_square(x) + s.w), {"x": X4()})

@control("R24", "F09-6: torch.load of bytes (Fable E10) and a storage from a raw buffer with no prior read (E11b) -> unknown")
def _():
    r1 = want_unknown(M(_load), {"x": X4()})
    r2 = want_unknown(M(_from_buffer_cfg), {"x": X4()})
    return r1[0] and r2[0], f"load {r1} from_buffer {r2}"

@control("R25", "F09-6 nearby valid: copy.deepcopy(x) and copy.copy(x), then a config constant -> credited (Fable E32 / note)")
def _():
    return want_credit(M(lambda s, x: _copymod.deepcopy(x) + _copymod.copy(x) + torch.tensor(s.cfg)), {"x": X4()})

@control("R26", "F09-6 nearby valid: empty() then a full recorded write (fill_ / copy_ / normal_) then read -> credited")
def _():
    def f(s, x):
        a = torch.empty(4).fill_(1.0); b = torch.empty(4); b.copy_(x); c = torch.empty_like(x).zero_()
        d = torch.zeros_like(torch.empty(4))                 # read for metadata only
        buf = torch.empty(4); buf[:2].fill_(1.0); buf[2:].fill_(2.0)   # views of fresh memory, then written
        return a + b + c + d + buf + torch.tensor(s.cfg)
    return want_credit(M(f), {"x": X4()})

# F09-7: untagged value -> shape operations
def _tsplit(s, x):
    idx = (x > 2.5).long().cumsum(0)[:1] + 1
    parts = torch.tensor_split(torch.arange(8.0), idx)
    return torch.tensor([float(p.shape[0]) for p in parts] + [0.0, 0.0]) + s.w

def _pack(s, x):
    lengths = (x > 2.5).long().sum().reshape(1).clamp(min=1)
    packed = torch.nn.utils.rnn.pack_padded_sequence(torch.arange(4.0).reshape(1, 4, 1), lengths, batch_first=True)
    return torch.tensor([float(packed.data.shape[0]), 1.0, 1.0, 1.0]) + s.w

@control("R27", "F09-7: tensor_split with a tensor of split points (CompositeImplicit C++ read, Fable E13) -> unknown")
def _():
    return want_unknown(M(_tsplit), {"x": X4()})

@control("R28", "F09-7: pack_padded_sequence lengths from input values (Fable E15) -> unknown")
def _():
    return want_unknown(M(_pack), {"x": X4()})

# S4: the four surviving mutants of round 1 (Opus MX1 / MX2 / MX3 / MX10)
def _helper_rev(l):
    l.reverse()
    return torch.tensor(l)

class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer("stat", torch.tensor([1.0, 2.0, 3.0]))

class _Outer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.zeros(())); self.inner = _Inner()
    def forward(self, x):
        return torch.tensor([v * v for v in self.inner.stat.tolist()]) + x[:3] + self.w

@control("R29", "MX1: a TWO-element unexplained build (the superpoint / omdet / encoder-decoder shape) -> unknown")
def _():
    return want_unknown(M(lambda s, x: torch.tensor([v * v for v in x.tolist()[:2]]) + s.w), {"x": X4()})

@control("R30", "MX2: tolist in the caller, reversed + rebuilt in a helper frame (palindrome) -> unknown")
def _():
    return want_unknown(M(lambda s, x: _helper_rev(x.tolist()) + s.w), {"x": torch.tensor([1.0, 2.0, 2.0, 1.0])})

@control("R31", "MX3: a boolean-mask index sets a shape, rebuilt -> unknown")
def _():
    def f(s, x):
        n = x[x > 3].shape[0]
        return torch.tensor([n * 1.0, n * 2.0]) + s.w
    return want_unknown(M(f), {"x": X4()})

@control("R32", "MX10: a NESTED module's checkpoint buffer read into Python -> unknown")
def _():
    return want_unknown(_Outer(), {"x": X4()})

class _Mut:
    def __init__(self, x): self.x = x
    @property
    def dt(self):
        self.x.add_(1.0)                     # a recorded write between tolist and the constructor (Fable C01)
        return torch.float32

@control("R33", "MX6: the copy guard refuses a direct copy whose source changed between tolist and the constructor")
def _():
    def f(s, x):
        o = _Mut(x)
        return torch.tensor(x.tolist(), dtype=o.dt) + s.w
    x = X4()
    rec, a, _ = record(M(f), {"x": x})
    return "copy" not in _cls_of(rec), f"classes {_cls_of(rec)}"


# ------------------------------------------------------------------------------------------------- worker wiring
def _src(name):
    p = os.path.join(H, name)
    return open(p).read() if os.path.exists(p) else ""

def _worker_block():
    """The worker's F09 statements (from `_pbu, _pbx, _pbc = ...` to the `if _pbu:` that appends the unknown)."""
    tree = ast.parse(_src("worker_v2.py"))
    body = tree.body
    i0 = next((i for i, s in enumerate(body) if isinstance(s, ast.Assign) and "_pbu" in ast.unparse(s.targets[0])), None)
    if i0 is None:
        return None
    i1 = next(i for i in range(i0, len(body)) if isinstance(body[i], ast.If) and ast.unparse(body[i].test) == "_pbu")
    return compile(ast.Module(body=body[i0:i1 + 1], type_ignores=[]), "worker_v2.py:F09", "exec")

def _run_worker(pass_res):
    code = _worker_block()
    if code is None:
        return None
    g = {"pass_res": pass_res, "UNKNOWNS": [], "checks": {"T3_closed_dataflow": {}}, "collections": collections}
    exec(code, g)
    return g["UNKNOWNS"]

def _p(n, **kw):
    return dict({"ok": True, "python_built_unexplained": n, "python_built": {"unexplained": n} if n else {},
                 "python_built_unexplained_examples": [{"module": "m", "shape": [4]}] if n else []}, **kw)

@control("W01", "worker: an input-only unexplained tensor in the main pass -> a G4 unknown (G5 never excuses G4)")
def _():
    u = _run_worker({"text": _p(1)})
    return bool(u) and u[0]["predicate"] == "G4" and u[0]["count"] == 1, f"{u}"

@control("W02", "worker: nothing unexplained -> no unknown")
def _():
    u = _run_worker({"text": _p(0), "stability:text": _p(0)})
    return u == [], f"{u}"

@control("W03", "worker: unexplained only in the STABILITY recording -> unknown")
def _():
    u = _run_worker({"text": _p(0), "stability:text": _p(2)})
    return bool(u) and u[0]["count"] == 2, f"{u}"

@control("W04", "worker: unexplained only in the training pass that alone credits tensors -> unknown")
def _():
    u = _run_worker({"text": _p(0), "train": _p(1, newly_used_params=3)})
    return bool(u), f"{u}"

@control("W05", "worker: unexplained in a training pass that credits nothing alone -> info only")
def _():
    u = _run_worker({"text": _p(0), "train": _p(1, newly_used_params=0, newly_used_buffers=0)})
    return u == [], f"{u}"

@control("W06", "worker: a pass with no F09 record but Python-built tensors -> unknown (nothing credited)")
def _():
    u = _run_worker({"text": {"ok": True, "python_mediated_values": 2}})
    return bool(u) and u[0]["count"] == 2, f"{u}"

@control("W07", "worker: a FAILED pass's recording credits nothing and adds nothing")
def _():
    u = _run_worker({"text": _p(0), "modes": dict(_p(3), ok=False)})
    return u == [], f"{u}"

def _append_only(src):
    """UNKNOWNS is never reassigned after its creation and never shrunk (append / extend only)."""
    t = ast.parse(src)
    bad = []
    for n in ast.walk(t):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            for x in tg:
                s = ast.unparse(x)
                if s.startswith("UNKNOWNS") and not (isinstance(n, ast.Assign) and s == "UNKNOWNS" and ast.unparse(n.value) == "[]"
                                                      and n.col_offset == 0):
                    bad.append(ast.unparse(n)[:80])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and ast.unparse(n.func.value) == "UNKNOWNS" \
                and n.func.attr not in ("append", "extend"):
            bad.append(ast.unparse(n)[:80])
        if isinstance(n, ast.Delete) and "UNKNOWNS" in ast.unparse(n):
            bad.append(ast.unparse(n)[:80])
    inits = [n for n in t.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "UNKNOWNS"]
    return (not bad and len(inits) == 1), bad

@control("W08", "UNKNOWNS append-only in both workers; FULL* + unknowns -> FULL_UNVERIFIED")
def _():
    out = []
    for f in ("worker_v2.py", "worker_v2_diffusion.py"):
        ok, bad = _append_only(_src(f))
        out.append((f, ok, bad))
    conv = re.search(r'if R\["verdict"\] in \("FULL", "FULL_LEFTOVERS"\) and UNKNOWNS:\s*\n(?:\s*#.*\n)*\s*R\["verdict"\] = "FULL_UNVERIFIED"',
                     _src("worker_v2.py")) is not None
    return all(o[1] for o in out) and conv and _worker_block() is not None, f"{out} conversion {conv} F09 block {_worker_block() is not None}"

@control("W09", "witness: transparent requires python_built_unexplained == 0 from BOTH runs (None / absent = not shown)")
def _():
    sys.path.insert(0, H)
    try:
        import importlib, witness_io
        importlib.reload(witness_io)
    except Exception as e:
        return False, f"witness_io: {e}"
    fn = getattr(witness_io, "witness_transparent", None)
    if fn is None:
        return False, "no witness_io.witness_transparent"
    base = {"op_full": ["aten::mm"], "opaque_ops": {}, "opacity_unknown_ops": [], "decomposition_disagrees": [],
            "external_nonscalar": 0}
    res = {n: fn(dict(base, python_built_unexplained=n)) for n in (0, 1, None)}
    res["absent"] = fn(dict(base))
    # the producer's own statements (mtp_witness_run, by AST): the record's value from BOTH recorded runs
    t = ast.parse(_src("mtp_witness_run.py"))
    stmts = None
    for n in ast.walk(t):
        b = getattr(n, "body", None)
        if isinstance(b, list):
            k = next((i for i, x in enumerate(b) if isinstance(x, ast.Assign)
                      and ast.unparse(x.targets[0]) == "R['python_built_unexplained']"), None)
            if k is not None:
                stmts = b[max(0, k - 1):k + 1]
    if stmts is None:
        return False, f"{res}; the producer never sets python_built_unexplained"
    code = compile(ast.Module(body=stmts, type_ignores=[]), "mtp:F09", "exec")
    def prod(a, b):
        g = {"_a": {"python_built_unexplained": a}, "_b": {"python_built_unexplained": b}, "R": {}}
        exec(code, g)
        return g["R"].get("python_built_unexplained")
    both = prod(0, 1) == 1 and prod(1, 0) == 1 and prod(0, 0) == 0 and prod(None, 0) is None and prod(0, None) is None
    return res == {0: True, 1: False, None: False, "absent": False} and both, f"{res} producer sums both runs: {both}"

def _block(fname, first, last_test):
    """Statements of `fname` from the Assign to `first` through the `if <last_test>:` that ends the block (by AST)."""
    t = ast.parse(_src(fname))
    for n in ast.walk(t):
        for fld in ("body", "orelse"):
            stmts = getattr(n, fld, None)
            if not isinstance(stmts, list):
                continue
            i0 = next((i for i, x in enumerate(stmts) if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == first), None)
            if i0 is not None:
                i1 = next(i for i in range(i0, len(stmts)) if isinstance(stmts[i], ast.If) and ast.unparse(stmts[i].test) == last_test)
                return compile(ast.Module(body=stmts[i0:i1 + 1], type_ignores=[]), f"{fname}:{first}", "exec")
    return None

def _fake_rec(entries):
    return types.SimpleNamespace(pyb=dict(enumerate(entries)), escape_hooks_ok=True, external=[])

_UNX = lambda mod: {"class": "unexplained", "numel": 4, "ndim": 1, "module": mod, "shape": [4], "dtype": "torch.float32", "after": None}

@control("W10", "diffusion: a component's unexplained build -> unknown for THAT component, on its STABILITY recordings too")
def _():
    blk = _block("worker_v2_diffusion.py", "graded_recs", "_pbu") or _block("worker_v2_diffusion.py", "_pbu", "_pbu")
    if blk is None:
        return False, "no F09 block in the diffusion worker"
    main = _fake_rec([_UNX("unet.down.0"), dict(_UNX("vae.dec"), **{"class": "static"})])
    stab = _fake_rec([_UNX("vae.dec")])
    direct2 = _fake_rec([_UNX("text_encoder.layers.0")])
    res = {}
    for cn in ("unet", "vae", "text_encoder", "scheduler"):
        g = {"recs": [main], "cn": cn, "UNKNOWNS": [], "rec_b": stab, "stop_b": None,
             "direct": {"text_encoder": (main, direct2)}, "__import__": __import__}
        exec(blk, g)
        res[cn] = [u["count"] for u in g["UNKNOWNS"]]
    return res == {"unet": [1], "vae": [1], "text_encoder": [1], "scheduler": []}, f"{res}"

@control("W11", "diffusion: a build in pipeline code (module (root)) -> one (pipeline) unknown (Opus S3 / Fable S6)")
def _():
    blk = _block("worker_v2_diffusion.py", "_f09_pipe_recs", "_pbp")
    if blk is None:
        return False, "no pipeline bucket in the diffusion worker"
    g = {"rec_ok": _fake_rec([_UNX("(root)"), _UNX("unet.x")]), "rec_b": None, "stop_b": "x", "models": {"unet": 1},
         "UNKNOWNS": []}
    exec(blk, g)
    return [(u["component"], u["count"]) for u in g["UNKNOWNS"]] == [("(pipeline)", 1)], f"{g['UNKNOWNS']}"

def _producer(target):
    t = ast.parse(_src("worker_v2.py"))
    for n in ast.walk(t):
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == f"pass_res[{target!r}]" and isinstance(n.value, ast.Dict):
            keys = [k.value if isinstance(k, ast.Constant) else None for k in n.value.keys]
            if "ok" in keys and getattr(n.value.values[keys.index("ok")], "value", None) is True:
                return n
    return None

@control("W12", "B1: the worker's OWN train and stability:synth records carry the F09 result (executed by AST, Opus x3)")
def _():
    m = M(lambda s, x: torch.tensor([v * v for v in x.tolist()]) + s.w)
    rec, a, _ = record(m, {"x": X4()})
    out = {}
    for target in ("train", "stability:synth"):
        n = _producer(target)
        if n is None:
            return False, f"no producer {target}"
        env = {"ex": {}, "mk": {}, "lab": None, "newly": {"p1", "p2"}, "newly_b": set(), "a": a, "_a2": a, "train_ledger": [],
               "_bs": 1, "_l2": [], "_ledger": [], "t": 0, "time": __import__("time"), "label": "text",
               "f09_fields": getattr(dag, "f09_fields", lambda a: {})}
        r = eval(compile(ast.Expression(n.value), target, "eval"), env)
        g = {"pass_res": {"text": dict(_p(0)), target: r}, "UNKNOWNS": [], "collections": collections}
        code = _worker_block()
        if code is None:
            return False, "no F09 reader block"
        exec(code, g)
        out[target] = [u["count"] for u in g["UNKNOWNS"]]
    return out == {"train": [1], "stability:synth": [1]}, f"{out}"

@control("W13", "B1: every ok pass record of the worker spreads the analyse() result or f09_fields (AST, all producers)")
def _():
    t = ast.parse(_src("worker_v2.py"))
    bad = []
    for n in ast.walk(t):
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]).startswith("pass_res[") and isinstance(n.value, ast.Dict):
            keys = [k.value if isinstance(k, ast.Constant) else None for k in n.value.keys]
            if "ok" not in keys or getattr(n.value.values[keys.index("ok")], "value", None) is not True:
                continue
            spreads = [ast.unparse(v) for k, v in zip(n.value.keys, n.value.values) if k is None]
            if not any("f09_fields(" in x or ".items()" in x and "structure_signature" in x for x in spreads):
                bad.append(f"line {n.lineno}: {ast.unparse(n.targets[0])}")
    return not bad, f"records without F09: {bad}"

@control("W14", "W06b: an ok pass whose record has NO F09 field at all -> unknown (never read as 0)")
def _():
    u = _run_worker({"text": {"ok": True}})
    return bool(u) and u[0]["count"] >= 1, f"{u}"

@control("W15", "S2 wiring: every attempts.run of a filtered input dict passes the ORIGINAL kwargs (py_inputs), both workers")
def _():
    bad = []
    t = ast.parse(_src("worker_v2.py"))
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "attempts.run" and len(n.args) >= 3 \
                and not isinstance(n.args[2], ast.Name) and not any(k.arg == "py_inputs" for k in n.keywords):
            bad.append(f"worker_v2.py:{n.lineno}")
    t = ast.parse(_src("worker_v2_diffusion.py"))
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "DagRecorder" and len(n.args) >= 2 \
                and ast.unparse(n.args[1]) != "{}" and not any(k.arg == "py_inputs" for k in n.keywords):
            bad.append(f"worker_v2_diffusion.py:{n.lineno}")
    fn = next((n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == "_attempt_call"), None)
    pc = fn is not None and "py_inputs" in ast.unparse(fn)
    return not bad and pc, f"missing py_inputs: {bad}; pipeline call passes its kwargs: {pc}"

@control("W16", "F09-13: a failed attempt's open tainted read stays open for EVERY later attempt of the run (its retry AND other passes)")
def _():
    sys.path.insert(0, H)
    try:
        import importlib, attempts
        importlib.reload(attempts)
    except Exception as e:
        return False, f"attempts: {e}"
    if not hasattr(dag, "CARRIED"):
        return False, "no run-wide taint carry (dag.CARRIED)"
    dag.CARRIED.update(n=0, why=None)
    CACHE = {}
    def fwd(s, x):
        if "v" not in CACHE:
            CACHE["v"] = [v * v for v in x.tolist()]
            raise ValueError("first attempt fails after filling a Python cache")
        return torch.tensor(CACHE["v"]) + s.w
    try:
        m = M(fwd); x = X4()
        try:
            attempts.run(m, m, {"x": x}, lambda: m(x=x), [], {"entry": "text"})
        except ValueError:
            pass
        rec, out = attempts.run(m, m, {"x": x}, lambda: m(x=x), [], {"entry": "text"})
        a = dag.analyse(rec, out); rec.remove_hooks()
        m2 = M(lambda s, x: torch.tensor(CACHE["v"]) + x)        # ANOTHER pass reads the same cache
        rec2, out2 = attempts.run(m2, m2, {"x": x}, lambda: m2(x=x), [], {"entry": "stability:text"})
        a2 = dag.analyse(rec2, out2); rec2.remove_hooks()
    finally:
        dag.CARRIED.update(n=0, why=None)
    m3 = M(lambda s, x: torch.tensor(s.cfg) + x)                   # after the reset: a clean run credits again
    rec3, a3, _ = record(m3, {"x": X4()})
    return a["python_built_unexplained"] == 1 and a2["python_built_unexplained"] == 1 and a3["python_built_unexplained"] == 0, \
        f"retry {a['python_built_unexplained']}; other pass {a2['python_built_unexplained']}; clean {a3['python_built_unexplained']}"


def _diff_t4_block():
    """The diffusion worker's own statements from `graded_recs = ...` (before the T3 line) and its T4 lines (after the
    T3 line up to `sa, sb = norm(`), by AST: F10's per-component T4 on every graded recording."""
    src = _src("worker_v2_diffusion.py"); t = ast.parse(src)
    loop = next((x for x in ast.walk(t) if isinstance(x, ast.For) and any(
        ast.get_source_segment(src, b).startswith('checks["T3_closed_dataflow"]') for b in x.body)), None)
    if loop is None:
        return None
    b = loop.body
    g = [x for x in b if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "graded_recs"]
    i = next(k for k, x in enumerate(b) if ast.get_source_segment(src, x).startswith('checks["T3_closed_dataflow"]'))
    j = next(k for k in range(i + 1, len(b)) if ast.get_source_segment(src, b[k]).startswith("sa, sb = norm("))
    return compile(ast.Module(body=g + b[i + 1:j], type_ignores=[]), "<diffusion graded_recs + T4>", "exec"), bool(g)

def _fake_nodes(opaque):
    n = dag.Node("mm", "transformer.blk", [], [], 1, (2, 2), op="aten::mm", opq=None)
    o = dag.Node("secret", "transformer.blk", [], [], 1, (2, 2), op="f09ctl::secret", opq="opaque: f09ctl::secret")
    return types.SimpleNamespace(nodes=[n] + ([o] if opaque else []), returns=None, pyb={}, external=[], escape_hooks_ok=True)

@control("W17", "F10 T4 (diffusion, per component) reads the STABILITY recording too (Opus S3): an opaque op only there fails T4")
def _():
    got = _diff_t4_block()
    if got is None:
        return False, "no diffusion T4 block"
    code, has_g = got
    if not has_g:
        return False, "no graded_recs: T4 reads the credited call only"
    res = {}
    for label, stab, stop in (("stab opaque", _fake_nodes(True), None), ("stab stopped", _fake_nodes(True), "stopped"),
                              ("stab clean", _fake_nodes(False), None)):
        e = {"recs": [_fake_nodes(False)], "rec_b": stab, "stop_b": stop, "direct": {}, "cn": "transformer", "checks": {},
             "UNKNOWNS": [], "collections": collections, "R": {}, "__import__": __import__}
        exec(code, e)
        res[label] = e["checks"]["T4_no_opaque_ops"]["pass"]
    return res == {"stab opaque": False, "stab stopped": True, "stab clean": True}, f"{res}"

@control("W18", "drawing view: an assembled build round-trips through the bundle as lift_fresh_assembled (never a view of its first element); a copy keeps lift_fresh with its source edge")
def _():
    import capture
    m = M(lambda s, x: torch.tensor([x[0] * 2, x[1]]) + torch.tensor(x.tolist())[:2] + s.w)
    x = X4()
    rec = capture.CaptureRecorder(m, {"x": x})
    try:
        with torch.no_grad(), rec:
            y = m(x=x)
    finally:
        rec.remove_hooks()
    f, _ = capture.fold_groups(capture.nodes_of(capture.graph(rec, y)))
    got = [(n.func, [e[0] for e in n.ins]) for n in f if n.func.startswith("lift")]
    return got == [("lift_fresh_assembled", ["op", "op"]), ("lift_fresh", ["input"])], f"{got}"

@control("R40", "the list-index read is the same under a recorder SUBCLASS (CaptureRecorder: EMIT runs): the grade never depends on emitting a bundle")
def _():
    import capture
    res = {}
    for cls in (dag.DagRecorder, capture.CaptureRecorder):
        m = M(_idx_cfg); x = X4()
        rec = cls(m, {"x": x})
        try:
            with torch.no_grad(), rec:
                y = m(x=x)
            a = dag.analyse(rec, y)
        finally:
            rec.remove_hooks()
        res[cls.__name__] = a.get("python_built")
    return res == {"DagRecorder": {"static": 2}, "CaptureRecorder": {"static": 2}}, f"{res}"

def _r36b(s, x):
    s.idxl = [0, 2]
    e = x.reshape(1, 4)
    i = 1
    return (x[s.idxl].sum() + x[range(2)].sum() + x[list(range(3))].sum() + x[[i, i + 1]].sum()
            + x[s.idxl[:]].sum() + e[range(1), ...].sum() + x)        # (esmfold: esmaa[range(B), ...])

@control("R36b", "honest static list indices resolve: plain data attribute, range(n), list(range(n)), [i, i+1], s.idx[:], esmfold's x[range(B), ...]")
def _():
    def chk(rec, a, y):
        c = _cls_of(rec)
        return "unwatched" not in c and set(c) == {"static"} and len(c) == 6, f"classes {c}"
    return want_credit(M(_r36b), {"x": X4()}, chk)

class _DescIdx:
    def __get__(self, obj, typ=None):
        return [0, 1]

class _HasDesc(W):
    di = _DescIdx()
    def forward(self, x):
        return x[self.di] + self.w            # a non-data descriptor: code on access, never run

@control("R37b", "an index read through a descriptor / property stays unwatched (never run by the recorder)")
def _():
    rec, a, _ = record(_HasDesc(), {"x": X4()})
    return _cls_of(rec) == ["unwatched"] and a.get("python_built_unexplained") == 1, f"classes {_cls_of(rec)}"

class _DynWrapped(dag.DagRecorder):
    __torch_dispatch__ = torch._disable_dynamo(dag.DagRecorder.__torch_dispatch__)   # (what torch 2.14 does to every mode)

@control("R42", "R3-S1: a recorder whose __torch_dispatch__ is wrapped by dynamo's disable (torch 2.14) resolves list indices the same")
def _():
    m = M(_idx_cfg); x = X4()
    rec = _DynWrapped(m, {"x": x})
    try:
        with torch.no_grad(), rec:
            y = m(x=x)
        a = dag.analyse(rec, y)
    finally:
        rec.remove_hooks()
    return a.get("python_built") == {"static": 2}, f"classes {a.get('python_built')}"

@control("R43", "R3-S2: an assembled build's edges are in DATA order (torch.tensor([x[0], x[1]*2, x[2]]))")
def _():
    rec, a, _ = record(M(lambda s, x: torch.tensor([x[0], x[1] * 2, x[2]]) + s.w), {"x": X4()})
    lf = [n for n in rec.nodes if n.func.startswith("lift_fresh")]
    if len(lf) != 1:
        return False, f"{[n.func for n in rec.nodes]}"
    src = [(rec.nodes[e[1]].func, rec.nodes[e[1]].args) for e in lf[0].ins if e[0] == "op"]
    want = [rec.nodes[i] for i in range(len(rec.nodes))]
    sel = [i for i, n in enumerate(rec.nodes) if n.func == "select"]
    mul = [i for i, n in enumerate(rec.nodes) if n.func == "mul"]
    order = [e[1] for e in lf[0].ins]
    return len(sel) == 3 and order == [sel[0], mul[0], sel[2]], f"edges {order} selects {sel} mul {mul}"

def _drawn(fn):
    import capture
    m = M(fn); x = X4()
    rec = capture.CaptureRecorder(m, {"x": x})
    try:
        with torch.no_grad(), rec:
            y = m(x=x)
    finally:
        rec.remove_hooks()
    g = capture.graph(rec, y)
    f, _ = capture.fold_groups(capture.nodes_of(g), g.get("f09_assembled"))
    return [n.func for n in f if n.func.startswith("lift")]

@control("W18b", "R3-S3: builds drawn by their CLASS: one op leaf + a number, and parameter leaves, are assembled (never a view of one source)")
def _():
    def init_fn(s, x):
        t = x[0] * 2
        return torch.tensor([t, 1.0]).sum() + torch.tensor([s.w, s.w]).sum() + x
    got = _drawn(init_fn)
    return got == ["lift_fresh_assembled", "lift_fresh_assembled"], f"{got}"

@control("R45", "R3-S4: never-written memory read by a constructor (leaves) or through a direct copy -> F09-6 unknown, never sealed")
def _():
    r1 = want_unknown(M(lambda s, x: torch.tensor([torch.empty(()), torch.empty(())]) + x[:2] + s.w), {"x": X4()})
    def two(s, x):
        b = torch.empty(2)
        return torch.tensor([b[0], b[1]]) + x[:2] + s.w
    r2 = want_unknown(M(two), {"x": X4()})
    r3 = want_unknown(M(lambda s, x: torch.tensor(torch.empty(4).tolist()) + x + s.w), {"x": X4()})
    return r1[0] and r2[0] and r3[0], f"leaves {r1[0]}, view leaves {r2[0]}, copy {r3[0]}"

def _taint_leaf(s, x):
    b = torch.zeros(())
    b.view(1).copy_(x[:1])                                      # b's storage now holds an input value (written via a view)
    t = torch.tensor([b, b])                                    # C++ reads b itself: no recorded op reads b's storage
    n = t.tolist()
    return torch.tensor([v * 2 for v in n]) + s.w

@control("R46", "R3-S4: constructor leaves from a TAINTED storage taint the build (a later read of it is a tainted read)")
def _():
    return want_unknown(M(_taint_leaf), {"x": X4()})

@control("R47", "R3-S5: resize_ is not a write: grown memory read after it -> unknown; a shrink of written memory stays credited")
def _():
    r1 = want_unknown(M(lambda s, x: torch.empty(0).resize_(4) + x + s.w), {"x": X4()})
    r2 = want_unknown(M(lambda s, x: torch.zeros(2).resize_(4) + x + s.w), {"x": X4()})
    r3 = want_credit(M(lambda s, x: torch.zeros(8).resize_(4) + x + s.w), {"x": X4()})
    return r1[0] and r2[0] and r3[0], f"empty+resize {r1}; grown {r2}; shrunk {r3}"

def _overw(s, x):
    a = torch.empty(4); a.put_(torch.arange(4), torch.ones(4))          # accumulate=False: writes, reads nothing
    b = torch.empty(3, 3); b.fill_diagonal_(1.0)                        # (as_strided + fill_)
    c = torch.empty(4); c.index_put_((torch.arange(4),), torch.ones(4))
    d = torch.empty(3, 2); d.index_reduce_(0, torch.arange(3), torch.ones(3, 2), "prod", include_self=False)
    e = torch.empty(3); e.scatter_reduce_(0, torch.arange(3), torch.ones(3), "sum", include_self=False)
    f = torch.empty(1, 3); f.scatter_(1, torch.arange(3).view(1, 3), torch.ones(1, 3))   # no reduce: an overwrite
    return a + b.diagonal().sum() + c + d.sum() + e.sum() + f.sum() + x + s.w

def _accum(s, x):
    c = torch.empty(4); c.index_put_((torch.arange(4),), torch.ones(4), accumulate=True)   # READS the never-written target
    return c + x + s.w

def _reads(s, x):
    f = torch.empty(1, 4); f.scatter_(1, torch.arange(4).view(1, 4), torch.ones(1, 4), reduce="add")   # reads
    d = torch.empty(4, 1); d.index_reduce_(0, torch.arange(4), torch.ones(4, 1), "prod", include_self=True)   # reads
    return f.view(4) + d.view(4) + x + s.w

@control("R48", "schema evidence: put_ / index_put_ (accumulate=False), index_reduce_ / scatter_reduce_ (include_self=False), scatter_ without reduce overwrite; accumulate=True / reduce / include_self=True read")
def _():
    r1 = want_credit(M(_overw), {"x": X4()})
    r2 = want_unknown(M(_accum), {"x": X4()})
    r3 = want_unknown(M(_reads), {"x": X4()})
    return r1[0] and r2[0] and r3[0], f"overwrites {r1}; accumulate {r2}; reduce / include_self {r3}"

@control("R25b", "Fable S-2: copy.copy of an OP OUTPUT (set_ from a recorded storage), then a config constant -> credited")
def _():
    return want_credit(M(lambda s, x: _copymod.copy(x * 2) + torch.tensor(s.cfg)), {"x": X4()})

@control("R02b", "Fable S-3: TUPLE data is walked too: torch.tensor((x[0]*2, x[1], x[2], x[3])) is assembled, never static")
def _():
    def chk(rec, a, y):
        return _cls_of(rec) == ["assembled"], f"classes {_cls_of(rec)}"
    return want_credit(M(lambda s, x: torch.tensor((x[0] * 2, x[1], x[2], x[3])) + s.w), {"x": X4()}, chk)

def _one_elem(s, x):
    v = x.tolist()
    out = torch.zeros(1)
    out.copy_(torch.tensor(v[0] * 2))         # one value into a WHOLE one-element tensor: not assembly
    return out + s.w

@control("R18b", "Fable S-3: a single value written into a whole one-element tensor is NOT assembly (storage-size guard)")
def _():
    rec, a, _ = record(M(_one_elem), {"x": X4()})
    return a.get("python_built_unexplained") == 0 and "assembled_unexplained" not in _cls_of(rec), f"classes {_cls_of(rec)}"

@control("W19", "N4: the witness G4 reason NAMES F09 when python_built_unexplained is the cause (and says 'not classified' when absent)")
def _():
    sys.path.insert(0, H)
    import importlib, common
    importlib.reload(common)
    base = {"op_full": ["aten::mm"], "opaque_ops": {}, "opacity_unknown_ops": [], "decomposition_disagrees": [],
            "external_nonscalar": 0}
    def g4(w):
        try:
            return [r for p, r in common.witness_execution_credit(w, "rev") if p == "G4"]
        except Exception as e:
            return [f"raised {e}"]
    a, b, c = g4(dict(base, python_built_unexplained=2)), g4(dict(base)), g4(dict(base, python_built_unexplained=0))
    ok = any("F09" in r for r in a) and not any("outside the recording" in r for r in a) \
        and any("no F09 classification" in r for r in b) and c == []
    return ok, f"F09 {a}; absent {b}; clean {c}"

def _set_empty(s, x):
    e = torch.empty(4)
    y = torch.empty(0).set_(e.untyped_storage(), 0, (4,), (1,))     # a view of never-written memory, by set_
    return y + x + s.w

def _set_grown(s, x):
    e = torch.empty(0); e.resize_(4)
    y = torch.empty(0).set_(e.untyped_storage(), 0, (4,), (1,))
    return y + x + s.w

def _set_tensor(s, x):
    e = torch.empty(4)
    y = torch.empty(0); y.set_(e)                                    # (set_.source_Tensor: no storage argument)
    return y + x + s.w

@control("R24b", "Fable R4 S1/S4/S5: set_ onto never-written memory (an empty, a grown resize_, a tensor source) stays never-written; set_ onto a written storage is credited")
def _():
    r1 = want_unknown(M(_set_empty), {"x": X4()})
    r2 = want_unknown(M(_set_grown), {"x": X4()})
    r3 = want_unknown(M(_set_tensor), {"x": X4()})
    r4 = want_credit(M(lambda s, x: torch.empty(0).set_((x * 2).untyped_storage(), 0, (4,), (1,)) + torch.tensor(s.cfg)), {"x": X4()})
    return r1[0] and r2[0] and r3[0] and r4[0], f"empty {r1}; grown {r2}; tensor {r3}; written {r4}"

_RUNS = []

class _Evil:
    def __add__(self, o):
        _RUNS.append("Evil.__add__"); return [0, 2]

class _EvilInt(int):
    def __add__(self, o):
        _RUNS.append("int.__add__"); return [0, 2]

class _EvilList(list):
    def __getitem__(self, k):
        _RUNS.append("list.__getitem__"); return list.__getitem__(self, k)
    def __iter__(self):
        _RUNS.append("list.__iter__"); return list.__iter__(self)
    def __reversed__(self):
        _RUNS.append("list.__reversed__"); return list.__reversed__(self)
    def __len__(self):
        _RUNS.append("list.__len__"); return list.__len__(self)

class _Idx:
    def __index__(self):
        _RUNS.append("__index__"); return 2

class _Box:
    def __getitem__(self, k):
        _RUNS.append("Box.__getitem__"); return 1

class _Obj:
    def meth(self):
        _RUNS.append("meth"); return [0, 2]

class _Cfg:
    def __init__(self):
        self.lst = [0, 2]
    def __getattribute__(self, nm):
        _RUNS.append("__getattribute__"); return object.__getattribute__(self, nm)

def _mk():
    _RUNS.append("function"); return [0, 2]

def _side_effects(s, x):
    e, ei, el, ix, o, bx, cf = _Evil(), _EvilInt(1), _EvilList([0, 1]), _Idx(), _Obj(), _Box(), _Cfg()
    seq = [0, 1, 2, 3]
    xx = x.reshape(2, 2)
    a = x[e + 1]                     # program arithmetic (__add__)
    b = x[ei + 1]                    # an int subclass's __add__
    c = x[o.meth()]                  # a method call
    d = x[_mk()]                     # a plain function call
    f = x[[bx[0], 2]]                # a container's __getitem__
    g = x[[el[0], 2]]                # a list subclass's __getitem__
    h = x[seq[ix::1]]                # a slice bound's __index__ (BUILD_SLICE + subscript)
    k = x[seq[ix:]]                  # a slice bound's __index__ (BINARY_SLICE)
    m = x[el]                        # a list subclass as the index (C++ reads it through its own methods)
    n = xx[[0, 1], el]               # ... next to a plain list part
    i = x[cf.lst]                    # a class's own __getattribute__ (the resolver reads the instance dict directly)
    return sum(t.sum() for t in (a, b, c, d, f, g, h, k, m, n, i)) + x + s.w

@control("R50", "the recorder NEVER runs program code: __add__ (also an int subclass's), a method, a function, a container / list-subclass __getitem__ / __iter__ / __reversed__, a slice bound's __index__, a class's __getattribute__ run exactly as often recorded as unrecorded; every index needing such code stays unwatched, the plain one is credited")
def _():
    m = M(_side_effects)
    _RUNS.clear()
    with torch.no_grad():
        m(x=X4())
    plain_runs = sorted(_RUNS)
    _RUNS.clear()
    rec, a, _ = record(m, {"x": X4()})
    runs = sorted(_RUNS)
    cls = _cls_of(rec)
    n_unw = len(cls) - 1
    return runs == plain_runs and cls == ["static"] + ["unwatched"] * n_unw and n_unw >= 10, \
        f"program code runs recorded {runs} vs unrecorded {plain_runs}; classes {cls}"

class _Lie(list):
    """A list subclass whose iteration yields other values (C++ reads a subclass through __iter__)."""
    def __iter__(self):
        return iter(self.src)

def _lie_ctor(s, x):
    L = _Lie([1.0, 2.0, 3.0, 4.0])
    L.src = list(x.unbind(0))        # C++ reads x's elements through __iter__; the raw items are numbers
    return torch.tensor(L) + s.w

@control("R51", "constructor data that is a sequence SUBCLASS is never followed: torch.tensor(L) whose __iter__ yields x's elements is a G4 unknown, never a static constant from L's raw items")
def _():
    return want_unknown(M(_lie_ctor), {"x": X4()})

@control("R39", "F09-9: an unseen build's unknown NAMES its reason (typed legacy constructor; unresolvable list index)")
def _():
    r1, a1, _ = record(M(lambda s, x: torch.LongTensor([1, 2, 3, 4]).float() + x), {"x": X4()})
    r2, a2, _ = record(M(_idx_call), {"x": X4()})
    w1 = [(e.get("after") or {}).get("kind", "") for e in a1.get("python_built_unexplained_examples", [])]
    w2 = [(e.get("after") or {}).get("kind", "") for e in a2.get("python_built_unexplained_examples", [])]
    ok = any("constructor the recorder cannot watch" in w for w in w1) and any("list index" in w for w in w2)
    return ok, f"legacy {w1}; index {w2}"


n_fail = sum(1 for r in RESULTS.values() if not r["ok"])
print(f"\n{len(RESULTS)} controls, {n_fail} failed  (harness: {H})")
if OUT:
    json.dump({"harness": H, "results": RESULTS, "failed": n_fail}, open(OUT, "w"), indent=1)
sys.exit(1 if n_fail else 0)
