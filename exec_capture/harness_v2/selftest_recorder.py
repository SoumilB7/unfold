"""F07 controls for the recorder's zero-skip and zero storage (EXECUTION_PLAN P2a.1; dossier
model-benchmark/_reruns/p2a1_f07; rulings DECISIONS 2026-10-13 F07-1/2/3). Pure torch, tiny shapes, no downloads.
Every NEGATIVE control must equal plain torch arithmetic (or raise exactly what plain torch raises); every POSITIVE
control must still take the skip (cost) AND equal plain arithmetic.
    python3 selftest_recorder.py            # exit 0 = all controls as expected
(pre-fix proof: run this file next to the HEAD-before-F07 dag.py / lowcost.py -> it fails)
"""
import sys, os, gc, math, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore"); os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch, torch.nn.functional as F
import lowcost, dag
Rec, zero_storage = dag.DagRecorder, lowcost.zero_storage
ZeroStorageWrite = getattr(dag, "ZeroStorageWrite", None)
registry_intact = getattr(lowcost, "registry_intact", None)
torch.manual_seed(0); torch.set_num_threads(1)
FAILS, N = [], 64
inf, nan = float("inf"), float("nan")

def Z(*shape, dtype=torch.float32):                  # a registered zero, from the worker's own producer
    with zero_storage(): return torch.empty(*shape, dtype=dtype)
def R(*shape, dtype=torch.float32): return torch.rand(*shape, dtype=dtype) + 0.5
def poke(t, v): t = t.clone(); t.view(-1)[0] = v; return t

class Wrap(torch.nn.Module):
    def __init__(s, fn): super().__init__(); s.fn = fn
    def forward(s, **kw): return s.fn(**kw)

def recorded(fn, kw, skip=True):
    m = Wrap(fn); r = Rec(m, kw, skip=skip)
    try:
        with torch.no_grad(), r: y = m(**kw)
    except Exception as e: y = e
    finally: r.remove_hooks()
    return y, r

def equal(a, b):
    if isinstance(a, Exception) or isinstance(b, Exception): return type(a) is type(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.stride() != b.stride(): return False
    return bool(torch.equal(torch.isnan(a), torch.isnan(b))) and bool(torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)))

def control(name, fn, kw, must_skip=None):
    try:
        with torch.no_grad(): real = fn(**{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()})
    except Exception as e: real = e
    got, r = recorded(fn, kw)
    ok = equal(got, real) and (must_skip is None or (r.skipped > 0) == must_skip)
    print(("PASS " if ok else "FAIL ") + f"{name}  (skipped={r.skipped})")
    if not ok: FAILS.append(name)

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: FAILS.append(name)

# ---------------- NEGATIVE controls: a zero operand that does NOT zero the result -------------------------------
control("N1 addmm: registered-zero bias, nonzero product", lambda c, a, b: torch.addmm(c, a, b), dict(c=Z(4096), a=R(3, 8), b=R(8, 4096)))
control("N2 embedding: registered-zero indices, nonzero table", lambda w, i: F.embedding(i, w), dict(w=R(2, 2) * 3, i=Z(4096, dtype=torch.int64)))
control("N3 mm: zero weight x inf activation = nan", lambda a, b: torch.mm(a, b), dict(a=poke(R(3, N), inf), b=Z(N, N)))
control("N4 conv: zero kernel + bias 3 = 3", lambda x, w, b: F.conv1d(x, w, b), dict(x=R(1, N, 4), w=Z(N, N, 2), b=torch.full((N,), 3.)))
control("N5 conv: zero input (by value) + bias = bias", lambda x, w, b: F.conv2d(x, w, b, padding=1), dict(x=torch.zeros(1, 2, N, N), w=R(3, 2, 3, 3), b=R(3)))
control("N6 conv_transpose: zero kernel + bias = bias", lambda x, w, b: F.conv_transpose1d(x, w, b), dict(x=R(1, N, 4), w=Z(N, N, 2), b=R(N)))
control("N7 addmm: value-zero bias, nonzero product", lambda c, a, b: torch.addmm(c, a, b), dict(c=torch.zeros(5000), a=R(3, 8), b=R(8, 5000)))
control("N8 baddbmm: registered-zero bias, nonzero q,k", lambda c, q, k: torch.baddbmm(c, q, k), dict(c=Z(2, N, N), q=R(2, N, 4), k=R(2, 4, N)))
control("N9 addmm beta=0 ignores a nan bias", lambda c, a, b: torch.addmm(c, a, b, beta=0), dict(c=poke(R(N), nan), a=R(3, N), b=Z(N, N)))
control("N10 addmm alpha=inf x zero product = nan", lambda c, a, b: torch.addmm(c, a, b, alpha=inf), dict(c=R(N), a=R(3, N), b=Z(N, N)))
control("N11 embedding: zero table, out-of-range index raises", lambda w, i: F.embedding(i, w), dict(w=Z(N, N), i=torch.tensor([[1, N + 5]])))
control("N12 bmm: inf x zero = nan", lambda a, b: torch.bmm(a, b), dict(a=poke(R(2, 3, N), inf), b=Z(2, N, N)))
control("N13 conv: registered-zero bias, nonzero x and kernel", lambda x, w, b: F.conv1d(x, w, b), dict(x=R(1, 2, 4), w=R(4096, 2, 1), b=Z(4096)))
xa, Wz, Bm = R(3, N), Z(N, N), R(N, 4)
def foreach(x, W, B):
    y = torch.mm(x, W); torch._foreach_add_([y], 1.0); return torch.mm(y, B)
control("N14 skipped output written by _foreach_add_ is no longer zero", foreach, dict(x=xa, W=Wz, B=Bm))
def npwrite(x, W, B):
    y = torch.mm(x, W); y.numpy()[:] = 1.0; return torch.mm(y, B)
control("N15 skipped output written outside dispatch (numpy) is no longer zero", npwrite, dict(x=xa, W=Wz, B=Bm))
# N16 allocator reuse: freed registered storages must never make a live nonzero tensor 'known zero'
tmp = [Z(N, N + i) for i in range(1000)]; del tmp; gc.collect()
bad = 0
for i in range(4000):
    a = R(1, 2 + i % 22); b = R(a.shape[1], 2)
    got, _ = recorded(lambda a, b: torch.mm(a, b), dict(a=a, b=b))
    bad += not equal(got, a @ b)
check(f"N16 no live tensor inherits a freed registered address ({bad} wrong of 4000)", bad == 0)
# N17 a write into shared zero storage THROUGH A VIEW cannot get its own storage: refused loudly (typed), never silent
E1, E2 = Z(2, N, N), Z(3, N, N)                      # same (N,N) key -> the same shared buffer
def write(x, E1):
    E1[0].add_(1.0); return torch.mm(x, E1[0])
got, _ = recorded(write, dict(x=xa, E1=E1))
check("N17 write into registered storage raises ZeroStorageWrite", ZeroStorageWrite is not None and isinstance(got, ZeroStorageWrite))
check("N17b ...and no other registered tensor changed", not bool(torch.any(E2)) and (registry_intact is None or registry_intact()))
lowcost._SHARED[(N, N, torch.float32)].zero_()                     # (repair, in case HEAD let the write through)
# N18 (F07-1) a forward-pass write into a WHOLE zero-storage weight gets its own storage: plain arithmetic, others intact
class CowM(torch.nn.Module):
    def __init__(s):
        super().__init__()
        with zero_storage():
            s.W = torch.nn.Parameter(torch.empty(3, N, N)); s.V = torch.nn.Parameter(torch.empty(2, N, N))
    def forward(s, x):
        s.W.add_(1.0); return torch.matmul(x, s.W[0])
cm = CowM(); xw = R(3, N)
r = Rec(cm, {"x": xw})
try:
    with torch.no_grad(), r: got = cm(x=xw)
except Exception as e: got = e
finally: r.remove_hooks()
check("N18 whole-weight write: copy-on-write gives plain arithmetic", not isinstance(got, Exception) and torch.allclose(got, xw @ torch.ones(N, N)))
check("N18b ...the other weight sharing the zero buffer is still zero, registry intact", not bool(torch.any(cm.V)) and (registry_intact is None or registry_intact()))
check("N18c ...and the copy-on-write is recorded", bool(getattr(dag, "COW_EVENTS", None)))
# N19 (F07-2) a computed buffer filled during the build gets its real values (not silently left zero)
with zero_storage():
    buf = torch.empty(N, N + 3); buf.copy_(torch.full((N, N + 3), 2.0))
    ex = torch.empty(N, N + 5).exp_()
    keep0 = torch.empty(N, N + 7); keep0.fill_(0.0)
    P_ = torch.nn.Parameter(torch.empty(N, N + 9)); P_.data.fill_(1.0)
check("N19 build-time copy_ into a zero-storage buffer is applied", bool((buf == 2.0).all()))
check("N19b build-time exp_ on a zero-storage temporary is applied (= 1)", bool((ex == 1.0).all()))
check("N19c a zero-preserving fill stays registered zero storage", keep0.untyped_storage().data_ptr() in lowcost.ZERO_PTRS)
check("N19d a learned Parameter stays zero by design (the checkpoint supplies it)", not bool(torch.any(P_)))
# N20 (F07-3) an op reading a weight that turns finite inputs into NaN is recorded (zero-weight artefact)
class NanM(torch.nn.Module):
    def __init__(s):
        super().__init__()
        with zero_storage(): s.w = torch.nn.Parameter(torch.empty(N, N))
    def forward(s, x):
        return torch.mm(x, s.w / s.w.norm())                # 0 / 0 = nan: weight_norm-style artefact
nm = NanM(); before = len(getattr(dag, "NONFINITE_EVENTS", []))
r = Rec(nm, {"x": R(2, N)})
with torch.no_grad(), r: nm(x=R(2, N))
r.remove_hooks()
check("N20 NaN created from finite inputs by an op reading a weight is recorded", len(getattr(dag, "NONFINITE_EVENTS", [])) > before)
# N21 a write that bypasses the recorder (numpy) into shared zero storage is detected by the registry check
victim = Z(N, N + 11); victim.numpy()[0, 0] = 5.0
check("N21 registry_intact detects a write outside dispatch", registry_intact is not None and not registry_intact())
victim.numpy()[0, 0] = 0.0

# ---- v2 controls (F07 cross-exam p2a1_xexam) ----
# B2: a call real torch rejects must not be certified as running (no skip, raise exactly what torch raises)
control("B2 mm fp32 x bf16 zero weight raises like torch", lambda a, b: torch.mm(a, b), dict(a=R(3, N), b=Z(N, N, dtype=torch.bfloat16)))
control("B2b addmm with mixed dtypes raises like torch", lambda c, a, b: torch.addmm(c, a, b), dict(c=R(N), a=R(3, N), b=Z(N, N, dtype=torch.bfloat16)))
control("B2c embedding with float indices raises like torch", lambda w, i: F.embedding(i, w), dict(w=Z(N, N), i=torch.tensor([[1.0, 2.0]])))
# N2: the sign of zero is preserved (bias -0.0; beta < 0 with +0 bias)
def signs(fn, kw):
    real = fn(**{k: v.clone() for k, v in kw.items()}); got, _ = recorded(fn, kw)
    return not isinstance(got, Exception) and torch.equal(torch.signbit(got), torch.signbit(real))
check("N2 addmm bias -0.0 keeps its sign", signs(lambda c, a, b: torch.addmm(c, a, b), dict(c=torch.full((N,), -0.0), a=R(3, N), b=Z(N, N))))
check("N2b addmm beta=-1 with +0 bias gives torch's sign", signs(lambda c, a, b: torch.addmm(c, a, b, beta=-1), dict(c=torch.zeros(N), a=R(3, N), b=Z(N, N))))
# S1: NaN created from finite inputs by an op that reads NO weight (CLIP-style 0/0 normalisation) is recorded
class NormM(torch.nn.Module):
    def __init__(s):
        super().__init__()
        with zero_storage(): s.w = torch.nn.Parameter(torch.empty(N, N))
    def forward(s, x):
        e = torch.mm(x, s.w); return e / e.norm(dim=-1, keepdim=True)
nm2 = NormM(); before = len(getattr(dag, "NONFINITE_EVENTS", []))
r = Rec(nm2, {"x": R(2, N)})
with torch.no_grad(), r: nm2(x=R(2, N))
r.remove_hooks()
check("S1 NaN from an op reading no weight (0/0 of activations) is recorded", len(getattr(dag, "NONFINITE_EVENTS", [])) > before)
# S2: a legacy-constructor parameter (torch.Tensor(n): uninitialised memory) whose random init is a no-op is zeroed
junk = torch.full((N, 3), 7.0); junk2 = torch.full((N,), 7.0); del junk, junk2     # leave garbage to be reused
with zero_storage():
    leg = torch.nn.Parameter(torch.Tensor(N, 3)); torch.nn.init.normal_(leg); leg2 = torch.nn.Parameter(torch.Tensor(N)); leg2.data.uniform_()
check("S2 random init under zero storage leaves no uninitialised memory (legacy constructors)", not bool(torch.any(leg)) and not bool(torch.any(leg2)))
# S2b: the build itself holding NaN / inf is reported (post-build scan)
fz = getattr(dag, "zero_storage_findings", None)
class BadBuf(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.register_buffer("b", torch.full((5,), float("nan")))
bb = BadBuf()
check("S2b a build holding NaN is reported as a G2 unknown", fz is not None and any("NaN" in u["what"] for u in fz([bb])[0]))
# S3: metadata-only in-place ops are not writes (no copy-on-write, no refusal)
def meta_inplace(x, E):
    v = E[0]; v.unsqueeze_(0); return torch.mm(x, v[0])
control("S3 unsqueeze_ on a view of zero storage is not a write", meta_inplace, dict(x=xa, E=Z(2, N, N)))
# N1: a build-time write through an alias of zero storage that belongs to no Parameter is a G2 unknown
with zero_storage():
    cb = torch.empty(N, N + 13); cb[:, 0::2].copy_(torch.ones(N, (N + 13 + 1) // 2))
class HoldBuf(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.register_buffer("cb", cb)
check("N1 an alias write into a computed buffer left zero is a G2 unknown", fz is not None and any("alias" in u["what"] for u in fz([HoldBuf()])[0]))

# ---- v3 controls (F07 gate: false positives seen on the control set) ----
# V3a: an allocation (empty_like: uninitialised memory, may hold NaN) is never a non-finite "computation"
class AllocM(torch.nn.Module):
    def forward(s, x):
        g = torch.full((4096,), float("nan")); del g      # leave NaN garbage for the allocator to reuse
        return torch.empty_like(x)
am = AllocM(); before = len(getattr(dag, "NONFINITE_EVENTS", []))
r = Rec(am, {"x": torch.zeros(4096)})
with torch.no_grad(), r: am(x=torch.zeros(4096))
r.remove_hooks()
check("V3a allocation (empty_like) output is never a non-finite event", len(getattr(dag, "NONFINITE_EVENTS", [])) == before)
# V3b: a learned Parameter the build left non-finite (A_log = log(0) = -inf) is reset to zero (the method executes
#      learned weights as zeros; the checkpoint supplies their values) and recorded; the findings do not flag it
class ALogM(torch.nn.Module):
    def __init__(s):
        super().__init__()
        A = torch.empty(8).uniform_(0, 16); s.A_log = torch.nn.Parameter(torch.log(A))
with zero_storage(): alm = ALogM()
nl = getattr(lowcost, "normalise_learned", None)
if nl: nl(alm)
check("V3b a build-left -inf learned parameter is reset to zero", bool(torch.isfinite(alm.A_log).all()) and not bool(torch.any(alm.A_log)))
check("V3c ...recorded, and not reported as a G2 unknown", bool(getattr(lowcost, "BUILD_PARAMS_RESET", None)) and fz is not None
      and not any("non-finite" in u["what"] for u in fz([alm])[0] if "A_log" in str(u["items"])))

# V4: the executed build is one dtype (float32), even when a sub-config declares bf16 (composite models such as
#     Qwen2.5-Omni build sub-models in their sub-config dtype): zero-storage weights become float32 zero views
#     (no memory), small real tensors are cast; nothing materialises
class Composite(torch.nn.Module):
    def __init__(s):
        super().__init__()
        with zero_storage():
            s.big = torch.nn.Linear(N, N * 2, dtype=torch.bfloat16); s.small = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
            s.register_buffer("tab", torch.empty(N, N + 15, dtype=torch.bfloat16))
cp = Composite()
uf = getattr(lowcost, "uniform_float32", None)
if uf: uf(cp)
dts = {t.dtype for t in list(cp.parameters()) + list(cp.buffers())}
check("V4 after the build every floating param / buffer is float32", dts == {torch.float32})
check("V4b ...zero-storage weights stay registered zero views (nothing materialised)",
      getattr(dag, "is_reg", lambda t: False)(cp.big.weight) and cp.big.weight.untyped_storage().nbytes() <= 4 * N * N and not bool(torch.any(cp.big.weight)))

# V5 (F07 xexam2 F1): +-inf sentinel buffers the library itself sets (Gemma-4 clippable-linear clamp bounds) are not a
#     zero-weight artefact: no G2; a NaN held by the build still is (S2b)
class ClampBuf(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.register_buffer("input_min", torch.tensor(float("-inf"))); s.register_buffer("input_max", torch.tensor(float("inf")))
check("V5 library +-inf sentinel buffers are not reported as a G2 unknown",
      fz is not None and not any("build holds" in u["what"] or "non-finite values (NaN" in u["what"] for u in fz([ClampBuf()])[0]))
# V6 (F07 xexam2 F2): a literal inf excuses only inf: zero weights x literal inf = NaN is still recorded
class LitM(torch.nn.Module):
    def __init__(s):
        super().__init__()
        with zero_storage(): s.w = torch.nn.Parameter(torch.empty(N, N))
    def forward(s, x):
        return torch.mul(torch.mm(x, s.w), float("inf"))     # 0 * inf = NaN: a zero-weight artefact
lm = LitM(); before = len(getattr(dag, "NONFINITE_EVENTS", []))
r = Rec(lm, {"x": R(2, N)})
with torch.no_grad(), r: lm(x=R(2, N))
r.remove_hooks()
check("V6 NaN from zero weights times a literal inf is recorded (a literal excuses only its own kind)", len(getattr(dag, "NONFINITE_EVENTS", [])) > before)
class MaskM(torch.nn.Module):
    def forward(s, x):
        return torch.softmax(x.masked_fill(torch.ones(4, 4).triu(1).bool(), float("-inf")), -1)
mm_ = MaskM(); before = len(getattr(dag, "NONFINITE_EVENTS", []))
r = Rec(mm_, {"x": R(4, 4)})
with torch.no_grad(), r: mm_(x=R(4, 4))
r.remove_hooks()
check("V6b a causal -inf mask then softmax is clean (no event)", len(getattr(dag, "NONFINITE_EVENTS", [])) == before)

# ---------------- POSITIVE controls: the sound skip is still taken and exact -------------------------------------
control("P10 zero-by-value activation x weight is skipped (value detection)", lambda a, b: torch.mm(a, b), dict(a=torch.zeros(3, N), b=R(N, N)), must_skip=True)
control("P1 mm(x, zero W) skipped", lambda a, b: torch.mm(a, b), dict(a=R(3, N), b=Z(N, N)), must_skip=True)
control("P2 addmm(b, x, zero W) = b, skipped", lambda c, a, b: torch.addmm(c, a, b), dict(c=R(N), a=R(3, N), b=Z(N, N)), must_skip=True)
control("P2b addmm beta=.3 bf16 bitwise", lambda c, a, b: torch.addmm(c, a, b, beta=.3), dict(c=R(N, dtype=torch.bfloat16), a=R(3, N, dtype=torch.bfloat16), b=Z(N, N, dtype=torch.bfloat16)), must_skip=True)
control("P3 embedding(zero W, idx) skipped", lambda w, i: F.embedding(i, w), dict(w=Z(N, N), i=torch.tensor([[1, 2, 3]])), must_skip=True)
control("P4 conv(x, zero W, b) = b, skipped", lambda x, w, b: F.conv1d(x, w, b), dict(x=R(1, N, 4), w=Z(N, N, 2), b=R(N)), must_skip=True)
control("P4b conv channels_last keeps real strides", lambda x, w: F.conv2d(x, w), dict(x=R(1, 8, 6, 6).to(memory_format=torch.channels_last), w=Z(N, 8, 3, 3)), must_skip=True)
control("P5 baddbmm(alibi, zero q, k) = beta*alibi, skipped", lambda c, q, k: torch.baddbmm(c, q, k, alpha=.5), dict(c=R(2, 1, 5), q=Z(2, 5, 1024), k=R(2, 1024, 5)), must_skip=True)
def keep(x, E):
    E[0].div_(2.0); return torch.mm(x, E[0])
control("P6 zero-preserving in-place on registered storage skipped, still zero", keep, dict(x=xa, E=Z(2, N, N)), must_skip=True)
def inval(x, W, B):
    y = torch.mm(x, W); y.add_(1); return torch.mm(y, B)
control("P7 skipped output written by add_ is no longer zero", inval, dict(x=xa, W=Wz, B=Bm))
control("P9 embedding max_norm on registered table (renorm is zero-preserving, not refused)", lambda w, i: F.embedding(i, w, max_norm=1.0), dict(w=Z(N, N), i=torch.tensor([[1, 2]])), must_skip=True)
try:
    import transformers as T
    for name, cfg in [("LlamaForCausalLM", T.LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=False, max_position_embeddings=32)),
                      ("BertModel", T.BertConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32))]:
        with zero_storage(): m = getattr(T, name)._from_config(cfg, attn_implementation="eager", dtype=torch.float32).eval()
        kw = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        res = {}
        for skip in (True, False):
            r = Rec(m, kw, skip=skip)
            with torch.no_grad(), dag.math_attention(), r: y = m(**kw, **({"use_cache": False} if name.startswith("Llama") else {}))
            r.remove_hooks(); res[skip] = (y[0], [n.func for n in r.nodes], r.skipped)
        check(f"P8 {name}: skip forward == plain forward, same op list, skips taken ({res[True][2]})",
              torch.equal(res[True][0], res[False][0]) and res[True][1] == res[False][1] and res[True][2] >= 7)
except ImportError:
    print("SKIP P8 (transformers not installed)")
print(f"\n{len(FAILS)} failing control(s)" + (": " + "; ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
