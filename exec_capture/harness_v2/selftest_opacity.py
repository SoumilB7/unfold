"""F10 / INTENT G4 controls: operation transparency fails closed. Each control records a tiny module under the
harness recorder (dag.DagRecorder, the zero-skip on, math attention as attempts.run uses) and grades T4 from
dag.analyse exactly as the workers do: FAIL = an opaque op; UNKNOWN = transparency not shown (no recorded overload,
or a decomposition that does not reproduce the op); PASS otherwise.

  NEG controls must NOT be PASS (an opaque / custom / fused / subclass / overridden computation never earns
      transparent credit).
  VALID controls must be PASS (plain supported ops; an op PyTorch decomposes, recorded as its decomposition and
      drawn as one node), or exactly the stated non-failure (a packet-only record is UNKNOWN, never FAIL).

    python3 selftest_opacity.py [--harness DIR] [--report]   # exit 0 = every control as expected
--harness runs the same controls against another harness copy (e.g. HEAD) to show which controls it gets wrong.
"""
import os, sys, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
H = sys.argv[sys.argv.index("--harness") + 1] if "--harness" in sys.argv else HERE
sys.path.insert(0, os.path.abspath(H))
import torch
import dag
from dag import DagRecorder, analyse, math_attention

torch.manual_seed(0)
F10 = hasattr(dag.Node, "__slots__") and "body" in dag.Node.__slots__     # the harness under test has F10


def t4(a):
    if a.get("opaque_ops"):
        return "FAIL"
    if a.get("opacity_unknown_ops") or a.get("decomposition_disagrees"):
        return "UNKNOWN"
    return "PASS"


def record(mod, inputs, math=True, **rk):
    """(recorder, analysis) of mod(**inputs), recorded like attempts.run (no_grad, math attention, zero-skip on)."""
    rec = DagRecorder(mod, inputs, **rk)
    try:
        if math:
            with torch.no_grad(), math_attention(), rec:
                out = mod(**inputs)
        else:
            with torch.no_grad(), rec:
                out = mod(**inputs)
    finally:
        rec.remove_hooks()
    return rec, analyse(rec, out)


class Fn(torch.nn.Module):
    def __init__(self, fn, d=8, n=6):
        super().__init__()
        self.fn = fn
        self.w = torch.nn.Parameter(torch.randn(n, d))

    def forward(self, x):
        return self.fn(self, x)


X = torch.randn(2, 5, 8)

# ---- custom ops (torch.library): a Python implementation the recorder cannot see into ----------------------------
_LIB = torch.library.Library("f10ctl", "FRAGMENT")


@torch.library.custom_op("f10ctl::secret", mutates_args=())
def _secret(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x @ w.t()) * 3.0                                   # a hidden projection + activation


@torch.library.custom_op("f10ctl::add", mutates_args=())
def _cadd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:           # named "add", computes a product
    return x * y


class DequantWeight(torch.Tensor):
    """A weight whose own __torch_dispatch__ computes (dequantizes) before every op: work outside the recording."""
    @staticmethod
    def __new__(cls, q, scale):
        r = torch.Tensor._make_wrapper_subclass(cls, q.shape, dtype=torch.float32)
        r.q, r.scale = q, scale
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        unwrap = lambda a: a.q.float() * a.scale if isinstance(a, DequantWeight) else a
        return func(*[unwrap(a) for a in args], **{k: unwrap(v) for k, v in (kwargs or {}).items()})


class SubW(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("wq", torch.randint(-8, 8, (6, 8), dtype=torch.int8))
        self.wd = DequantWeight(self.wq, 0.1)

    def forward(self, x):
        return torch.mm(x.reshape(-1, 8), self.wd.t())


class Int8Pack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("wq", torch.randint(-8, 8, (6, 8), dtype=torch.int8))
        self.register_buffer("sc", torch.ones(6))

    def forward(self, x):
        return torch.ops.aten._weight_int8pack_mm(x.reshape(-1, 8), self.wq, self.sc)


def _dyn_quant():
    torch.backends.quantized.engine = [e for e in torch.backends.quantized.supported_engines if e != "none"][0]
    return torch.ao.nn.quantized.dynamic.Linear(8, 6)


class Grouped(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(2, 8, 8))

    def forward(self, x):
        a = x.reshape(-1, 8)[:8].contiguous()
        return torch._grouped_mm(a, self.w, offs=torch.tensor([4, 8], dtype=torch.int32, device=a.device))


class Attn(torch.nn.Module):
    """Eager attention (the plain supported ops) or torch's sdpa."""
    def __init__(self, sdpa=False):
        super().__init__()
        self.q = torch.nn.Linear(8, 8); self.k = torch.nn.Linear(8, 8); self.v = torch.nn.Linear(8, 8)
        self.sdpa = sdpa

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.sdpa:                                        # [B, heads, S, d] (the 4-D form fused kernels take)
            sp = lambda t: t.reshape(t.shape[0], t.shape[1], 2, 4).transpose(1, 2)
            a = torch.nn.functional.scaled_dot_product_attention(sp(q), sp(k), sp(v)).transpose(1, 2).reshape(x.shape)
        else:
            a = torch.softmax(q @ k.transpose(-1, -2) / 8 ** 0.5, -1) @ v
        return torch.nn.functional.gelu(a + x)


class Overridden(torch.nn.Module):
    """aten::roll with its CPU kernel replaced (a library overriding an aten kernel): the name says roll."""
    def forward(self, x):
        return torch.roll(x, 1, -1) + 1


class Mutating(torch.nn.Module):
    """In-place ops PyTorch decomposes (masked_fill_ with a tensor value, triu_): the write is the decomposition's."""
    def forward(self, x):
        y = x.clone()
        y.masked_fill_(y > 0, torch.tensor(0.5))
        s = (y[0] @ y[0].transpose(-1, -2)).clone()
        s.triu_()
        return y.sum() + s.sum()


class StackNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = torch.nn.LayerNorm(8)

    def forward(self, x):
        return torch.stack([self.n(x), x], 0).sum(0) * x.norm(dim=-1, keepdim=True)


def legacy_rec(func_names):
    """A schema-1 recording (packet names only, as captures before F06 hold them), read back through capture."""
    import capture
    g = {"schema": 1, "strings": ["(root)"] + list(func_names),
         "nodes": [[k + 1, 0, ([["o", k - 1]] if k else []), [], 1, [2]] for k in range(len(func_names))]}
    nodes = capture.nodes_of(g)

    class R:
        pass
    r = R(); r.nodes = nodes; r.external = []; r.skipped = 0
    r.output_nodes = lambda out: {len(nodes) - 1}
    r.return_structure = lambda out: []
    return r


def run_legacy(names):
    r = legacy_rec(names)
    return analyse(r, None)


CONTROLS = []          # (id, kind NEG/VALID, expected, description, thunk -> analysis)


def C(cid, kind, expect, desc):
    def deco(fn):
        CONTROLS.append((cid, kind, expect, desc, fn)); return fn
    return deco


@C("N1", "NEG", {"FAIL", "UNKNOWN"}, "torch.library custom op with a Python implementation (hidden matmul + tanh)")
def _n1():
    return record(Fn(lambda s, x: torch.ops.f10ctl.secret(x, s.w)), {"x": X})[1]


@C("N2", "NEG", {"FAIL", "UNKNOWN"}, "custom op NAMED add (namespace f10ctl) computing a product")
def _n2():
    return record(Fn(lambda s, x: torch.ops.f10ctl.add(x, x)), {"x": X})[1]


@C("N3", "NEG", {"FAIL", "UNKNOWN"}, "nn.MultiheadAttention fast path: aten::_native_multi_head_attention (one fused op)")
def _n3():
    m = torch.nn.MultiheadAttention(8, 2, batch_first=True).eval()
    return record(Fn(lambda s, x: m(x, x, x, need_weights=False)[0]), {"x": X})[1]


@C("N4", "NEG", {"FAIL", "UNKNOWN"}, "nn.TransformerEncoderLayer fast path: aten::_transformer_encoder_layer_fwd")
def _n4():
    m = torch.nn.TransformerEncoderLayer(8, 2, 16, batch_first=True).eval()
    return record(Fn(lambda s, x: m(x)), {"x": X})[1]


def _without_review(name, thunk):
    """Run a control with one reviewed entry removed (and the classifier cache cleared): the gate must refuse again."""
    import opclass
    rev = getattr(opclass, "REVIEWED", None)
    rev = rev if isinstance(rev, dict) else {}
    cache = getattr(opclass, "_CACHE", None)
    saved = rev.pop(name, None)
    if cache is not None: cache.clear()
    try:
        return thunk()
    finally:
        if saved is not None:
            rev[name] = saved
        if cache is not None: cache.clear()


@C("V-R1", "VALID", {"PASS"}, "aten::_grouped_mm, REVIEWED (Soumil 2026-10-14, F10-2 = G4-1): transparent")
def _v_r1():
    return record(Grouped(), {"x": torch.randn(2, 5, 8)})[1]


@C("N5", "NEG", {"FAIL", "UNKNOWN"}, "aten::_grouped_mm with its review entry removed: fails closed")
def _n5():
    return _without_review("aten::_grouped_mm", lambda: record(Grouped(), {"x": torch.randn(2, 5, 8)})[1])


@C("N6", "NEG", {"FAIL", "UNKNOWN"}, "quantized aten kernel aten::_weight_int8pack_mm")
def _n6():
    return record(Int8Pack(), {"x": X})[1]


@C("N7", "NEG", {"FAIL", "UNKNOWN"}, "quantized::linear_dynamic (non-aten quantized namespace)")
def _n7():
    q = _dyn_quant()
    return record(Fn(lambda s, x: q(x.reshape(-1, 8))), {"x": X})[1]


@C("N8", "NEG", {"FAIL", "UNKNOWN"}, "tensor-subclass weight dequantizing in its own __torch_dispatch__ (op name: aten::mm)")
def _n8():
    return record(SubW(), {"x": X})[1]


def _bicubic_aa():
    return record(Fn(lambda s, x: torch.nn.functional.interpolate(x.reshape(1, 2, 5, 8), size=(3, 4), mode="bicubic",
                                                                  antialias=True)), {"x": X})[1]


@C("V-R2", "VALID", {"PASS"}, "anti-aliased bicubic resize, REVIEWED (Soumil 2026-10-14, F10-2): transparent")
def _v_r2():
    return _bicubic_aa()


@C("N9", "NEG", {"FAIL", "UNKNOWN"}, "the anti-aliased bicubic resize with its review entry removed: no evidence, no decomposition -> fails")
def _n9():
    return _without_review("aten::_upsample_bicubic2d_aa", _bicubic_aa)


@C("N10", "NEG", {"FAIL", "UNKNOWN"}, "a prims:: op called directly (non-aten namespace)")
def _n10():
    return record(Fn(lambda s, x: torch.ops.prims.sin(x) + 1), {"x": X})[1]


@C("N11", "NEG", {"FAIL", "UNKNOWN"}, "aten::roll whose CPU kernel a library replaced (the decomposition disagrees)")
def _n11():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("roll", lambda x, shifts, dims=(): x * 2.0, "CPU")
    try:
        return record(Overridden(), {"x": X})[1]
    finally:
        lib._destroy()


@C("N12", "NEG", {"FAIL", "UNKNOWN"}, "packet-only (schema-1) record whose 'add' was a custom op: cannot be called transparent")
def _n12():
    return run_legacy(["add", "add"])


@C("V1", "VALID", {"PASS"}, "plain supported ops: linear / add / bmm / softmax / gelu (eager attention)")
def _v1():
    return record(Attn(sdpa=False), {"x": X})[1]


@C("V2", "VALID", {"PASS"}, "sdpa under the math backend: aten::_safe_softmax recorded as its decomposition")
def _v2():
    rec, a = record(Attn(sdpa=True), {"x": X})
    if F10:
        assert a["decomposed_ops"].get("aten::_safe_softmax"), a["decomposed_ops"]
    return a


@C("V3", "VALID", {"PASS"}, "sdpa WITHOUT math backend (flash-for-cpu kernel): recorded as its decomposition")
def _v3():
    rec, a = record(Attn(sdpa=True), {"x": X}, math=False)
    if F10:
        assert any("scaled_dot_product" in k for k in a["decomposed_ops"]), a["decomposed_ops"]
    return a


@C("V4", "VALID", {"PASS"}, "stack / full-reduction sum / vector norm (non-core, decomposable) + layer_norm (core)")
def _v4():
    return record(StackNorm(), {"x": X})[1]


@C("V5", "VALID", {"PASS"}, "in-place decomposable ops (masked_fill_.Tensor, triu_): recorded as their decomposition (on copies; the op writes)")
def _v5():
    rec, a = record(Fn(lambda s, x: Mutating()(x)), {"x": X})
    if F10:
        assert a["decomposed_ops"], a
    return a


@C("V6", "VALID", {"UNKNOWN"}, "packet-only (schema-1) record of a plain add: UNKNOWN (re-record), never FAIL")
def _v6():
    return run_legacy(["add", "mm", "add"])


# =====================================================================================================================
# F10 round 2 (dual cross-exam 2026-10-14: Opus crossexam/, Fable crossexam_fable/; rulings F10-5..F10-8). Each control
# below FAILS on the round-1 worktree (model-benchmark/_reruns/p21_f10/impl_r2/r1) unless it is marked as a mutant
# pin (it then refuses a named surviving mutant of the round-1 reports). CHECK controls assert a property directly:
# expect TRUE.
# =====================================================================================================================
import ast, collections, contextlib, copy, inspect, tempfile, textwrap, time, traceback
import lowcost


def CHECK(ok, detail=""):
    return ("CHECK", bool(ok), detail)


def _zero_build(build, min_numel=16):
    """A model built as the worker builds it: zero storage (registered shared zero buffers above min_numel), float32,
    learned values normalised (the F07 zero-skip then fires exactly as in a real row)."""
    with lowcost.zero_storage(min_numel=min_numel):
        m = build()
    lowcost.uniform_float32(m)
    lowcost.normalise_learned(m)
    return m.eval()


def _reg(t):
    return t.untyped_storage().data_ptr() in lowcost.ZERO_REG


def _skipped_decomposed(rec, a, op):
    """(the op was recorded as its decomposition, the recorder zero-skipped something)"""
    return bool(a.get("decomposed_ops", {}).get(op)) and rec.skipped > 0


def _also(a, cond, detail):
    """The gate's verdict first (a refusing harness shows its own T4 verdict); when it passes, the extra property must
    hold too, else the control reports FALSE."""
    if t4(a) != "PASS" or cond():
        return a
    return CHECK(False, detail)


# ---- B1: a decomposable op the F07 zero-skip / zero-preserving in-place path short-circuits ------------------------
class BloomishZ(torch.nn.Module):
    """alibi.baddbmm(q, k^T) with q, k from a projection whose weight is REGISTERED zero storage (the harness build)."""
    def __init__(self):
        super().__init__()
        self.qk = torch.nn.Linear(8, 16, bias=False)

    def forward(self, x):
        q, k = self.qk(x).split(8, -1)
        alibi = torch.arange(5.0).expand(2, 5, 5).contiguous()
        return torch.softmax(alibi.baddbmm(q, k.transpose(1, 2), beta=1.0, alpha=0.3), -1)


@C("Z1", "VALID", {"PASS"}, "B1: BLOOM-style baddbmm on registered zero storage (zero-skipped): recorded as its decomposition, exact value kept")
def _z1():
    m = _zero_build(BloomishZ)
    assert _reg(m.qk.weight), "the control needs registered zero storage"
    x = torch.randn(2, 5, 8)
    rec, a = record(m, {"x": x})
    return _also(a, lambda: _skipped_decomposed(rec, a, "aten::baddbmm"), (a.get("decomposed_ops"), rec.skipped))


@C("Z2", "VALID", {"PASS"}, "B1: real tiny BloomForCausalLM (transformers code), zero-storage build: baddbmm recorded as its decomposition")
def _z2():
    from transformers import BloomConfig, BloomForCausalLM
    cfg = BloomConfig(vocab_size=64, hidden_size=16, n_layer=1, n_head=2)
    cfg._attn_implementation = "eager"
    m = _zero_build(lambda: BloomForCausalLM(cfg))
    rec, a = record(m, {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])})
    return _also(a, lambda: _skipped_decomposed(rec, a, "aten::baddbmm"), (a.get("decomposed_ops"), rec.skipped))


@C("Z3", "VALID", {"PASS"}, "B1: RecurrentGemma-style torch.baddbmm(bias, act, W) with W registered zero (+ the real tiny model)")
def _z3():
    class RG(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.w = torch.nn.Parameter(torch.empty(2, 4, 4)); s.b = torch.nn.Parameter(torch.empty(2, 4))
        def forward(s, x):
            return torch.baddbmm(s.b[:, None, :], x.reshape(2, -1, 4), s.w)
    m = _zero_build(RG)
    rec, a = record(m, {"x": torch.randn(2, 3, 4)})
    from transformers import RecurrentGemmaConfig, RecurrentGemmaForCausalLM
    cfg = RecurrentGemmaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                               num_key_value_heads=1, lru_width=16, attention_window_size=8, conv1d_width=2,
                               block_types=["recurrent", "attention"])
    cfg._attn_implementation = "eager"
    m2 = _zero_build(lambda: RecurrentGemmaForCausalLM(cfg))
    rec2, a2 = record(m2, {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])})
    out = dict(a)
    for k in ("opaque_ops", "opacity_unknown_ops", "decomposition_disagrees"):
        out[k] = dict(collections.Counter(a.get(k) or {}) + collections.Counter(a2.get(k) or {}))
    return _also(out, lambda: _skipped_decomposed(rec, a, "aten::baddbmm") and a2.get("decomposed_ops", {}).get("aten::baddbmm"),
                 (a.get("decomposed_ops"), a2.get("decomposed_ops")))


class ZeroKeep(torch.nn.Module):
    """zero_ / fill_(tensor 0) / masked_fill_(mask, tensor 0) on a REGISTERED zero weight: the zero-preserving in-place
    path (F07) skips the write (shared storage, no copy-on-write)."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(6, 8), requires_grad=False)

    def forward(self, x):
        self.w.zero_()
        self.w.fill_(torch.tensor(0.0))
        self.w.masked_fill_(torch.ones(6, 8, dtype=torch.bool), torch.tensor(0.0))
        return x @ self.w.t()


@C("Z4", "VALID", {"PASS"}, "B1: zero_ / fill_.Tensor / masked_fill_.Tensor on registered zero storage: decomposition recorded, storage still shared (no copy-on-write)")
def _z4():
    import dag as _d
    m = _zero_build(ZeroKeep)
    assert _reg(m.w), "the control needs registered zero storage"
    n_cow = len(_d.COW_EVENTS)
    rec, a = record(m, {"x": torch.randn(2, 8)})
    dec = a.get("decomposed_ops", {})
    return _also(a, lambda: all(dec.get(o) for o in ("aten::zero_", "aten::fill_.Tensor", "aten::masked_fill_.Tensor"))
                 and _reg(m.w) and len(_d.COW_EVENTS) == n_cow and not bool(m.w.any()),
                 {"decomposed": dec, "still_registered": _reg(m.w), "cow": _d.COW_EVENTS[n_cow:]})


@C("Z5", "NEG", {"FAIL"}, "B1 twin: a zero-skipped recording never hides an opaque namespace (custom op reading a registered zero weight)")
def _z5():
    m = _zero_build(lambda: Fn(lambda s, x: torch.ops.f10ctl.secret(x, s.w) + (x @ s.w.t())))
    return record(m, {"x": X})[1]


# ---- B2 / F10-5: prims inside a recorded PyTorch decomposition are transparent; a direct prims call is not ---------
def _ctl_fn(fn, x=None, **rk):
    return record(Fn(fn), {"x": X if x is None else x}, **rk)[1]


@C("P1", "VALID", {"PASS"}, "F10-5: torch.var_mean (its decomposition emits prims::var/sum/div)")
def _p1():
    return _ctl_fn(lambda s, x: sum(torch.var_mean(x, -1)))


@C("P2", "VALID", {"PASS"}, "F10-5: torch.std_mean")
def _p2():
    return _ctl_fn(lambda s, x: sum(torch.std_mean(x, -1)))


@C("P3", "VALID", {"PASS"}, "F10-5: nn.InstanceNorm1d in eval (native_batch_norm(training=True) -> var_mean -> prims)")
def _p3():
    m = torch.nn.InstanceNorm1d(5).eval()
    return _ctl_fn(lambda s, x: m(x))


@C("P4", "VALID", {"PASS"}, "F10-5: torch.cumprod (prims::prod inside)")
def _p4():
    return _ctl_fn(lambda s, x: torch.cumprod(x, -1))


@C("P5", "VALID", {"PASS"}, "F10-5: in-place float floor division x //= 2 (floor_divide_ -> 15 prims kinds)")
def _p5():
    def f(s, x):
        y = x.clone(); y //= 2
        return y
    return _ctl_fn(f)


@C("P6", "VALID", {"PASS"}, "F10-5: F.batch_norm(training=True) (BiT weight standardisation, timm StdConv)")
def _p6():
    return _ctl_fn(lambda s, x: torch.nn.functional.batch_norm(x.reshape(1, 10, 8), None, None, training=True, momentum=0.0, eps=1e-5))


@C("P7", "VALID", {"PASS"}, "F10-5 / F10-6: out= overloads (cat.out: variant of core cat; tril.out: decomposed with prims)")
def _p7():
    def f(s, x):
        o1 = torch.empty(4, 5, 8); torch.cat([x, x], 0, out=o1)
        o2 = torch.empty(2, 5, 8); torch.tril(x, out=o2)
        return o1.sum() + o2.sum()
    return _ctl_fn(f)


@C("P8", "VALID", {"PASS"}, "F10-5 / F10-7: uniform_ / normal_ in a forward (prims::uniform / prims::normal inside; random)")
def _p8():
    def f(s, x):
        return x + torch.empty_like(x).uniform_() + torch.empty_like(x).normal_()
    rec, a = record(Fn(f), {"x": X})
    return a


_ISIN_N = 200


def _isin_big():
    """isin of 200 values in a 1000-value list: PyTorch's decomposition sorts and binary-searches (searchsorted)."""
    def f(s, x):
        v = (x.reshape(-1)[:_ISIN_N] * 100).round().long()
        return torch.isin(v, torch.arange(1000)).sum() + x.sum()
    return record(Fn(f), {"x": torch.randn(2, 20, 8)})[1]


@C("P9", "NEG", {"FAIL"}, "an opaque op INSIDE a recorded decomposition: isin -> searchsorted with its review removed (refuses mutants O, P/M4)")
def _p9():
    a = _without_review("aten::searchsorted.Tensor", _isin_big)
    return a


@C("P10", "NEG", {"FAIL"}, "a decomposition that calls a custom op (the decomposition table entry replaced): the body op is graded (refuses P/M4)")
def _p10():
    import opclass
    dec = getattr(opclass, "decompositions", None)
    key = torch.ops.aten.cumprod.default
    table = dec() if dec else {}
    saved = table.get(key)
    table[key] = lambda self, dim, dtype=None: torch.ops.f10ctl.cumprod_secret(self, dim)
    try:
        return _ctl_fn(lambda s, x: torch.cumprod(x, -1))
    finally:
        if saved is not None:
            table[key] = saved


@torch.library.custom_op("f10ctl::cumprod_secret", mutates_args=())
def _cumprod_secret(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.cumprod(x, dim)


@C("P11", "VALID", {"PASS"}, "isin -> searchsorted (reviewed): the decomposition's inner op is classified and passes")
def _p11():
    return _isin_big()


# ---- F10-6: out= / in-place variants by schema; reviewed real kernels; the loader refuses bad entries ---------------
@C("S1", "VALID", {"PASS"}, "F10-6: in-place metadata variants t_ / squeeze_ / swapaxes_ (variants of the views t / squeeze / swapaxes)")
def _s1():
    def f(s, x):
        y = x[0].clone(); y.t_(); y.unsqueeze_(0); y.squeeze_(); y.swapaxes_(0, 1)
        return y
    return _ctl_fn(f)


@C("S2", "VALID", {"PASS"}, "F10-6: out= variant of a core op (mm.out) and of a reviewed op (sort.values_stable)")
def _s2():
    def f(s, x):
        a = x.reshape(-1, 8); o = torch.empty(10, 10); torch.mm(a, a.t(), out=o)
        v, i = torch.empty(10, 8), torch.empty(10, 8, dtype=torch.long); torch.sort(a, stable=True, out=(v, i))
        return o.sum() + v.sum()
    return _ctl_fn(f)


@C("S3", "VALID", {"PASS"}, "F10-6 reviewed kernels: FFT (FNet fftn + irfft), sort / argsort(stable), unique, masked_select, bincount, median, kthvalue, mode")
def _s3():
    def f(s, x):
        r = torch.fft.fftn(x, dim=(1, 2)).real.sum() + torch.fft.irfft(torch.fft.rfft(x, dim=-1), n=8, dim=-1).sum()
        r = r + torch.sort(x, dim=-1, stable=True)[0].sum() + torch.argsort(x, dim=-1, stable=True).sum()
        r = r + torch.unique(x.round()).sum() + torch.masked_select(x, x > 0).sum()
        r = r + torch.bincount((x.abs() * 3).long().reshape(-1)).sum() + x.median() + x.median(-1)[0].sum()
        r = r + x.kthvalue(2, -1)[0].sum() + x.round().mode(-1)[0].sum()
        return r
    return _ctl_fn(f)


@C("S4", "VALID", {"PASS"}, "F10-6 reviewed kernels: adaptive_max_pool2d, nn.Embedding(max_norm) (embedding_renorm_), EmbeddingBag, addbmm")
def _s4():
    emb = torch.nn.Embedding(10, 8, max_norm=1.0)
    bag = torch.nn.EmbeddingBag(10, 8, mode="mean")
    def f(s, x):
        r = torch.nn.functional.adaptive_max_pool2d(x.reshape(1, 2, 5, 8), (2, 3)).sum()
        ids = torch.tensor([[1, 2, 3]])
        r = r + emb(ids).sum() + bag(ids).sum()
        b = x.reshape(2, 5, 8)
        return r + torch.addbmm(torch.zeros(5, 5), b, b.transpose(1, 2)).sum()
    return _ctl_fn(f)


def _aten_frag():
    """A library op registered into the aten namespace whose schema returns an alias but WRITES an argument (no aten
    op has this today; a library or a future version can define one): not a view."""
    if not hasattr(torch.ops.aten, "f10ctl_vw"):
        lib = torch.library.Library("aten", "FRAGMENT")
        lib.define("f10ctl_vw(Tensor(a) self, Tensor(b!) other) -> Tensor(a)")
        def impl(self, other):
            other.add_(1.0)
            return self.view(self.shape)
        lib.impl("f10ctl_vw", impl, "CPU")
        _aten_frag.lib = lib
    return torch.ops.aten.f10ctl_vw


@C("S5", "NEG", {"FAIL"}, "an aten-namespace op returning an alias while writing an argument is not a view (refuses Q)")
def _s5():
    op = _aten_frag()
    def f(s, x):
        y = x.clone()
        return op(x, y) + y
    return _ctl_fn(f)


@C("S6", "NEG", {"UNKNOWN"}, "in-place op whose functional twin is decomposable (masked_fill_.Tensor), kernel replaced: value-checked, never credited by an in-place rule (refuses E/M10)")
def _s6():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("masked_fill_.Tensor", lambda self, mask, value: self.mul_(2.0), "CPU")
    try:
        def f(s, x):
            y = x.clone(); y.masked_fill_(y > 0, torch.tensor(0.5))
            return y
        return _ctl_fn(f)
    finally:
        lib._destroy()


def _yaml_refused(entries, over=None, extra=None):
    import opclass
    load = getattr(opclass, "_load_reviewed", None)
    if load is None:
        return False, "no loader"
    import yaml
    doc = {"reviewed": entries}
    if over is not None:
        doc["reviewed_over_decomposition"] = over
    doc.update(extra or {})
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(doc, fh)
    try:
        load(fh.name)
        return False, "accepted"
    except ValueError as e:
        return True, str(e)[:120]
    finally:
        os.unlink(fh.name)


@C("S7", "CHECK", {"TRUE"}, "F10-6/F10-10 loader refuses: typo overload, packet name, non-aten, no reason, a decomposable entry in `reviewed`, an evidence entry, a variant of another entry; and in reviewed_over_decomposition a non-decomposable / evidence op, a missing / extra / empty field, a typo, a key in both sections, an unknown section (refuses D)")
def _s7():
    good = {"aten::_grouped_mm": "x"}
    cases = {"typo": {"aten::_grouped_mm.default": "x"}, "packet": {"_grouped_mm": "x"}, "non-aten": {"f10ctl::secret": "x"},
             "no reason": {"aten::_grouped_mm": ""}, "decomposable": dict(good, **{"aten::_unsafe_view": "x"}),
             "evidence": dict(good, **{"aten::mm": "x"}),
             "variant of another entry": {"aten::bernoulli.p": "x", "aten::bernoulli_.float": "y"}}
    res = {k: _yaml_refused(v) for k, v in cases.items()}
    # F10-10: the second section takes ONLY decomposable overloads, each with exactly {reason, decomposition_unreliable}
    ok_over = {"reason": "r", "decomposition_unreliable": "d"}
    over = {"over: not decomposable": {"aten::_grouped_mm": ok_over}, "over: evidence op": {"aten::mm": ok_over},
            "over: missing field": {"aten::upsample_bilinear2d": {"reason": "r"}},
            "over: extra field": {"aten::upsample_bilinear2d": dict(ok_over, why="w")},
            "over: empty field": {"aten::upsample_bilinear2d": dict(ok_over, decomposition_unreliable=" ")},
            "over: typo": {"aten::upsample_bilinear2d.default": ok_over}}
    res.update({k: _yaml_refused({}, v) for k, v in over.items()})
    res["over and reviewed both"] = _yaml_refused({"aten::upsample_bilinear2d": "x"}, {"aten::upsample_bilinear2d": ok_over})
    res["unknown section"] = _yaml_refused({}, None, {"reviewed_overides": {}})
    ok_good = not _yaml_refused(good)[0] and not _yaml_refused(good, {"aten::upsample_bilinear2d": ok_over})[0]
    return CHECK(all(r for r, _ in res.values()) and ok_good, {k: v for k, v in res.items()})


@C("S8", "CHECK", {"TRUE"}, "F10-6 / F10-24: the shipped YAML loads STRICTLY in the main torch (no rejected entry: VOCAB_ERRORS empty, so the note is absent); every key is the exact name of a real overload; _unsafe_view / detach_ / transpose_ are not listed")
def _s8():
    import opclass
    rev = getattr(opclass, "REVIEWED", {})
    names = [(k, getattr(opclass, "resolve", lambda n: None)(k)) for k in rev]
    bad = [k for k, o in names if o is None or o.name() != k]
    gone = [k for k in ("aten::_unsafe_view", "aten::detach_", "aten::transpose_") if k in rev]
    try:
        strict = opclass._load_reviewed(opclass.VOCAB) == rev
    except ValueError as e:
        strict = f"refused: {e}"
    return CHECK(rev and not bad and not gone and strict is True and not opclass.VOCAB_ERRORS,
                 {"unresolved": bad, "should_be_gone": gone, "strict": strict, "errors": list(opclass.VOCAB_ERRORS)})


# ---- F10-7: random calls; the flash kernel with dropout 0 is value-checked; output structure ----------------------
class Flash(torch.nn.Module):
    def __init__(self, direct=False):
        super().__init__(); self.direct = direct

    def forward(self, x):
        q = x.reshape(2, 5, 2, 4).transpose(1, 2).contiguous()
        if self.direct:                               # a direct caller that READS logsumexp (output 1)
            o, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(q, q, q)
            return o.sum() + lse.sum(), lse
        return torch.nn.functional.scaled_dot_product_attention(q, q, q)


@C("R1", "NEG", {"UNKNOWN", "FAIL"}, "F10-7 (Opus S1): the flash-for-cpu kernel REPLACED (q*100), dropout_p == 0 in eval: value-checked, not passed")
def _r1():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("_scaled_dot_product_flash_attention_for_cpu",
             lambda q, k, v, dropout_p=0.0, is_causal=False, attn_mask=None, scale=None:
             (q * 100.0, torch.zeros(q.shape[:-1])), "CPU")
    try:
        return record(Flash(), {"x": X}, math=False)[1]
    finally:
        lib._destroy()


@C("R2", "VALID", {"PASS"}, "F10-7: train-mode dropout (bernoulli_.float, a variant of reviewed bernoulli.p) and torch.bernoulli")
def _r2():
    drop = torch.nn.Dropout(0.3).train()
    return _ctl_fn(lambda s, x: drop(x) + torch.bernoulli(torch.full_like(x, 0.5)))


@C("R3", "CHECK", {"TRUE"}, "F10-7: a random call is recorded as random (not value-checked); a seeded op called with dropout_p == 0 is value-checked (refuses M11)")
def _r3():
    rec, a = record(Fn(lambda s, x: x + torch.empty_like(x).uniform_()), {"x": X})
    rnd = dict(a.get("decomposed_random_info") or {})
    rec2, _ = record(Flash(), {"x": X}, math=False)
    fl = [n.body[3] for n in rec2.nodes if getattr(n, "body", None) is not None and "flash" in (n.op or "")]
    return CHECK(rnd.get("aten::uniform_") and fl == [True], {"random_info": rnd, "flash_agree": fl})


@C("R4", "CHECK", {"TRUE"}, "F10-7: a random op's values are the op's own: the program draws exactly what plain torch draws, and the stream continues as plain torch's (refuses WM20)")
def _r4():
    # (exponential_, the Gumbel-noise idiom: PyTorch's decomposition draws a DIFFERENT amount from the stream than the
    #  kernel, so only restoring the op's own stream state keeps every later draw the program's)
    mod = Fn(lambda s, x: x + torch.empty_like(x).uniform_() * torch.empty_like(x).normal_() + torch.bernoulli(torch.full_like(x, 0.5))
             + torch.empty_like(x).exponential_())
    torch.manual_seed(5)
    with torch.no_grad():
        plain = mod(x=X)
    nxt_plain = torch.rand(3)
    torch.manual_seed(5)
    rec = DagRecorder(mod, {"x": X})
    try:
        with torch.no_grad(), math_attention(), rec:
            got = mod(x=X)
    finally:
        rec.remove_hooks()
    nxt = torch.rand(3)
    return CHECK(torch.equal(plain, got) and torch.equal(nxt, nxt_plain), {"same_values": torch.equal(plain, got), "same_stream": torch.equal(nxt, nxt_plain)})


@C("R5", "NEG", {"UNKNOWN"}, "F10-7 (Opus p15): a caller reads the flash kernel's logsumexp, which the decomposition does not reproduce (different structure): UNKNOWN, and the program gets the real [B,H,S]")
def _r5():
    m = Flash(direct=True)
    rec = DagRecorder(m, {"x": X})
    try:
        with torch.no_grad(), rec:
            out = m(x=X)
    finally:
        rec.remove_hooks()
    a = analyse(rec, out)
    if tuple(out[1].shape) != (2, 2, 5):            # the program must receive the op's own logsumexp [B, H, S]
        return CHECK(False, {"program_logsumexp_shape": tuple(out[1].shape), "T4": t4(a)})
    return a


# ---- F10-8: the precision-scaled tolerance law ----------------------------------------------------------------------
def _resize(mode, src, dst):
    torch.manual_seed(0)
    x = torch.randn(1, 3, *src)
    return record(Fn(lambda s, x_: torch.nn.functional.interpolate(x_, size=dst, mode=mode, align_corners=False)), {"x": x})[1]


@C("T1", "VALID", {"PASS"}, "F10-8/F10-10: bilinear + bicubic resize 17x23->40x50 and 24x24->14x14 (PASS; since F10-10 the plain resizes are reviewed, recorded whole)")
def _t1():
    out = {"opaque_ops": {}, "opacity_unknown_ops": {}, "decomposition_disagrees": {}, "decomposed_ops": {}}
    for mode in ("bilinear", "bicubic"):
        for src, dst in (((17, 23), (40, 50)), ((24, 24), (14, 14))):
            a = _resize(mode, src, dst)
            for k in ("opaque_ops", "opacity_unknown_ops", "decomposition_disagrees", "decomposed_ops"):
                out[k].update(a.get(k) or {})
    return out


def _roll_skewed(rel):
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("roll", lambda x, shifts, dims=(): torch.cat([x[..., -1:], x[..., :-1]], -1) * (1.0 + rel), "CPU")
    try:
        return record(Fn(lambda s, x: torch.roll(x, 1, -1) + 1), {"x": X})[1]
    finally:
        lib._destroy()


@C("T2", "NEG", {"UNKNOWN"}, "F10-9: a kernel that disagrees with its decomposition by 1e-4 relative (beyond 64 eps per element): UNKNOWN")
def _t2():
    return _roll_skewed(1e-4)


@C("T3", "LAW", {"PASS"}, "F10-9 law width: a 1e-6 relative difference (within 64 eps x |ref_i|) counts as reproduced")
def _t3():
    return _roll_skewed(1e-6)


# ---- values are the op's own; S2 BatchNorm running statistics; ops count; decomposition that raises ---------------
@C("H1", "CHECK", {"TRUE"}, "the program's values are the op's own, not the decomposition's (aten::roll kernel replaced by x*2: refuses H)")
def _h1():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("roll", lambda x, shifts, dims=(): x * 2.0, "CPU")
    try:
        m = Overridden()
        with torch.no_grad():
            plain = m(X)
        rec = DagRecorder(m, {"x": X})
        try:
            with torch.no_grad(), math_attention(), rec:
                got = m(X)
        finally:
            rec.remove_hooks()
        return CHECK(torch.equal(plain, got), float((plain - got).abs().max()))
    finally:
        lib._destroy()


@C("B1", "CHECK", {"TRUE"}, "Opus S2: BatchNorm in training updates its running statistics ONCE (as plain torch), not by the decomposition and the op")
def _b1():
    torch.manual_seed(1)
    bn = torch.nn.BatchNorm1d(4, momentum=0.5).train()
    bn2 = copy.deepcopy(bn)
    x = torch.randn(8, 4) * 3 + 2
    with torch.no_grad():
        bn2(x)
    record(Fn(lambda s, x_: bn(x_)), {"x": x})
    ok = torch.equal(bn.running_mean, bn2.running_mean) and torch.equal(bn.running_var, bn2.running_var) \
        and int(bn.num_batches_tracked) == int(bn2.num_batches_tracked)
    return CHECK(ok, {"recorded": bn.running_mean.tolist(), "plain": bn2.running_mean.tolist()})


@C("C1", "CHECK", {"TRUE"}, "Fable S6: `ops` counts the program's calls (equal to the undecomposed recording); body_ops counts decomposition nodes")
def _c1():
    m = StackNorm()
    ra, aa = record(m, {"x": X})
    rb, ab = record(m, {"x": X}, decompose=False) if F10 else (ra, aa)
    return CHECK(aa["ops"] == len(rb.nodes) and aa.get("body_ops", 0) == len(ra.nodes) - len(rb.nodes) and aa.get("body_ops", 0) > 0,
                 {"ops": aa["ops"], "undecomposed": len(rb.nodes), "body_ops": aa.get("body_ops")})


@C("F1", "NEG", {"UNKNOWN"}, "Fable N5: a decomposition that raises does not fail the pass: the op runs itself, stays one node, G4 unknown")
def _f1():
    import opclass
    dec = getattr(opclass, "decompositions", None)
    table = dec() if dec else {}
    key = torch.ops.aten.stack.default
    saved = table.get(key)
    def boom(*a, **k):
        raise RuntimeError("decomposition failed (control)")
    table[key] = boom
    try:
        m = StackNorm()
        with torch.no_grad():
            plain = m(X)
        rec = DagRecorder(m, {"x": X})
        try:
            with torch.no_grad(), math_attention(), rec:
                out = m(X)
        finally:
            rec.remove_hooks()
        assert torch.equal(plain, out), "the program's values must be the op's own"
        return analyse(rec, out)
    finally:
        if saved is not None:
            table[key] = saved


@C("F2", "CHECK", {"TRUE"}, "Fable N5 (drawing bundle): a decomposition that raises leaves no capture data keyed by its discarded node indices")
def _f2():
    import opclass, capture
    table = opclass.decompositions() if hasattr(opclass, "decompositions") else {}
    key = torch.ops.aten.stack.default
    saved = table.get(key)
    def boom(tensors, dim=0):
        torch.nn.functional.gelu(tensors[0][..., :5], approximate="tanh")       # (a string argument: op_strings)
        sq = tensors[0][0, :5, :5]
        sq + sq                                                                  # (a square add: square_adds)
        raise RuntimeError("decomposition failed (control)")
    table[key] = boom
    try:
        m = StackNorm()
        rec = capture.CaptureRecorder(m, {"x": X})
        try:
            with torch.no_grad(), math_attention(), rec:
                out = m(X)
        finally:
            rec.remove_hooks()
        stale = {"op_strings": {k: rec.nodes[k].func if k < len(rec.nodes) else None for k in rec.op_strings},
                 "square_adds": {k: rec.nodes[k].func if k < len(rec.nodes) else None for k in rec.square_adds}}
        return CHECK(not rec.op_strings and not rec.square_adds, stale)
    finally:
        if saved is not None:
            table[key] = saved


# ---- graph schema / drawing guards -----------------------------------------------------------------------------------
@C("K1", "NEG", {"UNKNOWN"}, "a schema-2 graph (pre-F10, no opq column) read back is UNKNOWN, never transparent by omission (refuses K/M6)")
def _k1():
    import capture
    rec, _ = record(Attn(sdpa=False), {"x": X})
    g = capture.graph(rec)
    g = dict(g, schema=2, nodes=[row[:10] for row in g["nodes"]])
    r = legacy_rec([])
    r.nodes = capture.nodes_of(g)
    r.output_nodes = lambda out: {len(r.nodes) - 1}
    return analyse(r, None)


@C("L1", "CHECK", {"TRUE"}, "fold_groups refuses a visible node that reads a node hidden inside a decomposition group (refuses L)")
def _l1():
    import capture
    rec, _ = record(StackNorm(), {"x": X})
    nodes = list(rec.nodes)
    g = next((i for i, n in enumerate(nodes) if getattr(n, "body", None) is not None and n.body[1] > n.body[0]), None)
    if g is None:
        return CHECK(False, "no decomposition group recorded")
    hid = nodes[g].body[0]
    last = nodes[-1]
    nodes[-1] = type(last)(last.func, last.module, list(last.ins) + [("op", hid, 0)], last.consts, last.n_out, last.shape,
                           last.op, last.args, last.outs, last.cargs, last.opq, last.body)
    try:
        capture.fold_groups(nodes)
        return CHECK(False, "folded without refusing")
    except ValueError as e:
        return CHECK(True, str(e)[:100])


@C("O1", "CHECK", {"TRUE"}, "every recorded decomposition has a body recorded under the recorder, and its outputs come from that body (refuses O)")
def _o1():
    bad = []
    for name, mod in (("sdpa", Attn(sdpa=True)), ("stack/norm", StackNorm())):
        rec, a = record(mod, {"x": X})
        for i, n in enumerate(rec.nodes):
            b = getattr(n, "body", None)
            if b is None:
                continue
            if not (b[1] > b[0] and b[2] and all(e[0] == "op" and b[0] <= e[1] < b[1] for e in b[2])):
                bad.append((name, n.op, b[:3]))
    return CHECK(F10 and not bad, bad[:4])


@C("O2", "CHECK", {"TRUE"}, "a decomposition's body reads the PROGRAM's tensors (its private operand copies are never external; closure T3 unchanged)")
def _o2():
    bad = []
    for name, mod in (("sdpa", Attn(sdpa=True)), ("stack/norm", StackNorm()), ("in-place", Fn(lambda s, x: Mutating()(x))),
                      ("batch_norm train", Fn(lambda s, x: torch.nn.functional.batch_norm(x.reshape(1, 10, 8), None, None, training=True)))):
        rec, a = record(mod, {"x": X})
        inner = __import__("dag").body_nodes(rec.nodes) if F10 else set()
        ext = [(name, rec.nodes[i].op) for i in inner for e in rec.nodes[i].ins if e[0] == "external"]
        if a.get("external_nonscalar") or ext or not inner:
            bad.append((name, a.get("external_nonscalar"), ext[:2], len(inner)))
    return CHECK(not bad, bad)


# ---- worker wiring (Fable S5 / Opus B3): the worker's OWN T4 lines, extracted by AST, on fake pass results ---------
def _wsrc(f):
    return open(os.path.join(H, f)).read()


def _top_between(f, after, before):
    src = _wsrc(f); body = ast.parse(src).body
    i = next(k for k, s in enumerate(body) if ast.get_source_segment(src, s).startswith(after))
    j = next(k for k in range(i + 1, len(body)) if ast.get_source_segment(src, body[k]).startswith(before))
    return compile(ast.Module(body=body[i + 1:j], type_ignores=[]), f"<{f}:{body[i + 1].lineno}-{body[j - 1].end_lineno}>", "exec")


def _t4_worker(pass_res):
    code = _top_between("worker_v2.py", 'checks["T3_closed_dataflow"]', "first = next(")
    e = {"pass_res": pass_res, "checks": {}, "UNKNOWNS": [], "collections": collections, "R": {}}
    exec(code, e)
    _LAST_ROW.clear(); _LAST_ROW.update(e["R"])
    return e["checks"]["T4_no_opaque_ops"], [u for u in e["UNKNOWNS"] if u.get("predicate") == "G4"]


_LAST_ROW = {}          # (the row fields the last worker T4 block set: F10-24 vocabulary_notes)


def _p(**k):
    return dict({"ok": True, "opaque_ops": {}, "opacity_unknown_ops": {}, "decomposition_disagrees": {}, "decomposed_ops": {}}, **k)


@C("W1", "CHECK", {"TRUE"}, "worker T4: an opaque op in the main pass fails T4")
def _w1():
    c, u = _t4_worker({"text": _p(opaque_ops={"f10ctl::secret": 1})})
    return CHECK(c["pass"] is False and not u, (c, u))


@C("W2", "CHECK", {"TRUE"}, "worker T4: an op without an overload / a disagreeing decomposition is a G4 unknown, T4 passes")
def _w2():
    c1, u1 = _t4_worker({"text": _p(opacity_unknown_ops={"add": 2})})
    c2, u2 = _t4_worker({"text": _p(decomposition_disagrees={"aten::roll": 1})})
    return CHECK(c1["pass"] and len(u1) == 1 and c2["pass"] and len(u2) == 1, (u1, u2))


@C("W3", "CHECK", {"TRUE"}, "worker T4: generate and stability:synth are graded (an opaque op there fails T4)")
def _w3():
    c1, _ = _t4_worker({"text": _p(), "generate": _p(opaque_ops={"aten::_fft_c2c": 1})})
    c2, _ = _t4_worker({"text": _p(), "stability:synth": _p(opaque_ops={"f10ctl::secret": 1})})
    c3, u3 = _t4_worker({"text": _p(), "generate": _p(decomposition_disagrees={"aten::roll": 1})})
    return CHECK(c1["pass"] is False and c2["pass"] is False and len(u3) == 1, (c1, c2, u3))


@C("W4", "CHECK", {"TRUE"}, "worker T4: a failed pass is not graded")
def _w4():
    c, u = _t4_worker({"text": _p(), "mode:x": dict(_p(opaque_ops={"f10ctl::secret": 1}), ok=False)})
    return CHECK(c["pass"] and not u, (c, u))


@C("W5", "CHECK", {"TRUE"}, "F10-4 train rule: never a T4 failure; info when the train pass credits nothing alone")
def _w5():
    c, u = _t4_worker({"text": _p(), "train": _p(opaque_ops={"aten::_ctc_loss": 1}, newly_used_params=0, newly_used_buffers=0)})
    return CHECK(c["pass"] and not u, (c, u))


@C("W6", "CHECK", {"TRUE"}, "F10-4 train rule: a G4 unknown when the train pass alone credits parameters")
def _w6():
    c, u = _t4_worker({"text": _p(), "train": _p(opaque_ops={"aten::_ctc_loss": 1}, newly_used_params=3)})
    return CHECK(c["pass"] and len(u) == 1, (c, u))


@C("W7", "CHECK", {"TRUE"}, "F10-4 train rule (Opus S3 / Fable S2): a G4 unknown when the train pass alone credits checkpoint BUFFERS")
def _w7():
    c, u = _t4_worker({"text": _p(), "train": _p(opaque_ops={"aten::_ctc_loss": 1}, newly_used_params=0, newly_used_buffers=2)})
    return CHECK(c["pass"] and len(u) == 1, (c, u))


@C("W8", "CHECK", {"TRUE"}, "F10-4 train rule (Opus S3): the train pass's unknown ops and disagreeing decompositions are kept (G4 unknown when it credits alone)")
def _w8():
    c1, u1 = _t4_worker({"text": _p(), "train": _p(opacity_unknown_ops={"add": 1}, newly_used_params=2)})
    c2, u2 = _t4_worker({"text": _p(), "train": _p(decomposition_disagrees={"aten::roll": 1}, newly_used_params=2)})
    return CHECK(c1["pass"] and len(u1) == 1 and c2["pass"] and len(u2) == 1, (u1, u2))


def _diffusion_t4(recs, cn="transformer"):
    src = _wsrc("worker_v2_diffusion.py"); tree = ast.parse(src)
    loop = next(s for s in ast.walk(tree) if isinstance(s, ast.For) and any(
        ast.get_source_segment(src, b).startswith('checks["T3_closed_dataflow"]') for b in s.body))
    b = loop.body
    i = next(k for k, s in enumerate(b) if ast.get_source_segment(src, s).startswith('checks["T3_closed_dataflow"]'))
    j = next(k for k in range(i + 1, len(b)) if ast.get_source_segment(src, b[k]).startswith("sa, sb = norm("))
    e = {"recs": recs, "cn": cn, "checks": {}, "UNKNOWNS": [], "collections": collections, "R": {}}
    exec(compile(ast.Module(body=b[i + 1:j], type_ignores=[]), "<diffusion T4>", "exec"), e)
    _LAST_ROW.clear(); _LAST_ROW.update(e["R"])
    return e["checks"]["T4_no_opaque_ops"], [u for u in e["UNKNOWNS"] if u.get("predicate") == "G4"]


class Pipe(torch.nn.Module):
    def __init__(self, t_fn, v_fn):
        super().__init__()
        self.transformer = Fn(t_fn); self.vae = Fn(v_fn)

    def forward(self, x):
        return self.vae(self.transformer(x))


def _pipe_rec(t_fn, v_fn=lambda s, x: x + 1):
    m = Pipe(t_fn, v_fn)
    return record(m, {"x": X})[0]


@C("W9", "CHECK", {"TRUE"}, "diffusion T4 (per component): an opaque op in the component fails; another component's does not; an op inside a decomposition is graded")
def _w9():
    c1, _ = _diffusion_t4([_pipe_rec(lambda s, x: torch.ops.f10ctl.secret(x, s.w))])
    c2, _ = _diffusion_t4([_pipe_rec(lambda s, x: x * 2, lambda s, x: torch.ops.f10ctl.secret(x, s.w))])
    rec = _without_review("aten::searchsorted.Tensor", lambda: _pipe_rec(
        lambda s, x: x + torch.isin((x.reshape(-1)[:_ISIN_N] * 100).round().long(), torch.arange(1000)).sum()))
    c3, _ = _diffusion_t4([rec])
    return CHECK(c1["pass"] is False and c2["pass"] is True and c3["pass"] is False, (c1, c2, c3))


@C("W10", "CHECK", {"TRUE"}, "diffusion T4: a disagreeing decomposition in the component is a G4 unknown")
def _w10():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("roll", lambda x, shifts, dims=(): x * 2.0, "CPU")
    try:
        rec = _pipe_rec(lambda s, x: torch.roll(x, 1, -1))
    finally:
        lib._destroy()
    c, u = _diffusion_t4([rec])
    return CHECK(c["pass"] and len(u) == 1, (c, u))


# ---- the worker's train block (AST, as selftest_attempts runs it) + its T4 lines: buffers credited by train alone ----
def _env_for(m, **kw):
    import attempts as _att
    import class_rules
    from inputs import mask_positions
    e = dict(attempts=_att, ATTEMPT_LEDGER=[], torch=torch, time=time, traceback=traceback, inspect=inspect,
             DagRecorder=DagRecorder, analyse=analyse, math_attention=math_attention, class_rules=class_rules,
             mask_positions=mask_positions, m=m, R={"steps": {}}, save=lambda: None, EMIT=None, used_params=set(),
             used_buffers=set(), called=set(), pass_res={}, sigs={}, S_={}, _optional_ok=lambda label: True,
             PASS_READS=[], libload_pass=lambda rec: (set(rec.used_params) | set(rec.used_buffers), set(rec.called)),
             _limit=contextlib.nullcontext, TIED=[], TIED_FWD=[], TIED_SUB=[], _skip_tied=lambda w: None,
             _fresh=lambda: type(m)(), cfg=None, variants={}, checks={}, UNKNOWNS=[], collections=collections)
    e.update(kw); return e


def _worker_block(start, end=None):
    src = _wsrc("worker_v2.py"); body = ast.parse(src).body
    i = next(k for k, s in enumerate(body) if start in ast.get_source_segment(src, s))
    j = i if end is None else next(k for k in range(i, len(body)) if end in ast.get_source_segment(src, body[k]))
    return compile(ast.Module(body=body[i:j + 1], type_ignores=[]), f"<worker_v2.py:{body[i].lineno}>", "exec")


class TrainBuf(torch.nn.Module):
    """A loss weight buffer (persistent, shipped) read only by the training-mode pass, through an opaque op."""
    def __init__(s):
        super().__init__(); s.w = torch.nn.Parameter(torch.ones(4, 4)); s.register_buffer("loss_w", torch.ones(4))

    def forward(s, input_ids, labels=None):
        h = input_ids.float() @ s.w
        if s.training and labels is not None:
            return torch.ops.f10ctl.secret(h * s.loss_w, s.w).sum()
        return h


@C("W11", "CHECK", {"TRUE"}, "F10-4 through the worker's own train block + T4 lines: a checkpoint buffer only the train pass reads, behind an opaque op -> G4 unknown")
def _w11():
    m = TrainBuf().eval()
    TRAIN = _worker_block("sig_f = inspect.signature(m.forward).parameters", "if ((P_names - used_params)")
    e = _env_for(m, passes=[("t", {"input_ids": torch.ones(1, 4)})], pass_res={"t": _p()}, used_params={"w"}, used_buffers=set())
    exec(TRAIN, e)
    tr = e["pass_res"].get("train") or {}
    exec(_top_between("worker_v2.py", 'checks["T3_closed_dataflow"]', "first = next("), e)
    u = [x for x in e["UNKNOWNS"] if x.get("predicate") == "G4"]
    return CHECK(tr.get("ok") and tr.get("opaque_ops") and e["checks"]["T4_no_opaque_ops"]["pass"] and len(u) == 1,
                 {"train": {k: tr.get(k) for k in ("ok", "opaque_ops", "newly_used_params", "newly_used_buffers", "error")}, "unknowns": u})


# ---- Fable N3: generate deterministic where the library allows; recorded when not ----------------------------------
GEN_SEEN = []           # (module-level: attempts.run puts a model's own attributes back after every attempt, F13)


class GenNoMax(torch.nn.Module):
    """generate() without max_new_tokens / min_new_tokens (e.g. a forecasting head): takes do_sample."""
    def __init__(s):
        super().__init__(); s.w = torch.nn.Parameter(torch.ones(2))

    def forward(s, x):
        return x * s.w

    def generate(s, x, do_sample=True):
        GEN_SEEN.append(do_sample)
        return x * s.w


class GenNoSample(GenNoMax):
    def generate(s, x):
        GEN_SEEN.append("no-flag")
        return x * s.w


def _gen_run(m):
    UNUSED = compile(ast.Module(body=[n for n in ast.parse(_wsrc("worker_v2.py")).body if isinstance(n, ast.FunctionDef)
                                      and n.name == "_unused_now"], type_ignores=[]), "<_unused_now>", "exec")
    GEN = _worker_block('if hasattr(m, "generate") and class_rules.shared(TIED, "generate")')
    e = _env_for(m, passes=[("t", {"x": torch.ones(2)})], pass_res={})
    exec(UNUSED, e); exec(GEN, e)
    return e["pass_res"].get("generate") or {}


@C("G1", "CHECK", {"TRUE"}, "Fable N3: a generate() without max_new_tokens is still called with do_sample=False (recorded deterministic)")
def _g1():
    GEN_SEEN.clear(); m = GenNoMax(); g = _gen_run(m)
    return CHECK(g.get("ok") and GEN_SEEN == [False] and g.get("deterministic") is True, {"seen": list(GEN_SEEN), "gen": {k: g.get(k) for k in ("ok", "deterministic", "error")}})


@C("G2", "CHECK", {"TRUE"}, "Fable N3: a generate() that takes no do_sample is recorded as possibly sampling")
def _g2():
    GEN_SEEN.clear(); m = GenNoSample(); g = _gen_run(m)
    return CHECK(g.get("ok") and g.get("deterministic") is False, {"seen": list(GEN_SEEN), "gen": {k: g.get(k) for k in ("ok", "deterministic", "why", "error")}})


# ---- Fable N1: the witness reports the recorder's reason (with tensor types), not the bare name's -----------------
@C("M1", "CHECK", {"TRUE"}, "Fable N1: mtp_witness_run reports a subclass-dispatched op (aten::t on the dequantizing weight) with the recorder's reason (tensor subclass), never the bare name's ('core')")
def _m1():
    import opclass
    src = _wsrc("mtp_witness_run.py"); tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "one")
    b = fn.body
    i = next(k for k, s in enumerate(b) if ast.get_source_segment(src, s).startswith('R["op_full"]'))
    j = next(k for k, s in enumerate(b) if ast.get_source_segment(src, s).startswith('R["transparent"]'))

    class WRec(DagRecorder):
        def __init__(s, *a, **k):
            super().__init__(*a, **k); s.op_full = []
        def __torch_dispatch__(s, func, types, args=(), kwargs=None):
            o = super().__torch_dispatch__(func, types, args, kwargs); s.op_full.append(func.name()); return o
    m = SubW()
    recs = []
    for _ in range(2):
        r = WRec(m, {"x": X})
        try:
            with torch.no_grad(), r:
                o = m(X)
        finally:
            r.remove_hooks()
        recs.append((r, o))
    e = {"R": {}, "rec_a": recs[0][0], "out_a": recs[0][1], "rec_b": recs[1][0], "out_b": recs[1][1], "opclass": opclass,
         "analyse": analyse, "_pin_log": lambda: {}}
    exec(compile(ast.Module(body=b[i:j + 1], type_ignores=[]), "<mtp_witness_run.one>", "exec"), e)
    oo = e["R"].get("opaque_ops") or {}
    return CHECK(oo and all("subclass" in (w or "") for w in oo.values()) and e["R"].get("transparent") is False, {"opaque_ops": oo})



# =====================================================================================================================
# F10 round 3 (re-check of round 2 by Opus crossexam/r2 + Fable crossexam_fable/r2; rulings F10-9..F10-12). Each control
# FAILS on the round-2 worktree (impl_r3/r2) unless marked as a mutant pin (it then refuses a named round-2 survivor).
# =====================================================================================================================
import subprocess, signal as _signal


def _without_reviews(names, thunk):
    """_without_review for several entries at once."""
    if not names:
        return thunk()
    return _without_review(names[0], lambda: _without_reviews(names[1:], thunk))


_RESIZES = (("bilinear", (224, 224), (384, 384)), ("bicubic", (236, 236), (436, 436)), ("linear", (249,), (382,)))


def _resize_sweep():
    """The reviewers' resize sizes whose decompositions differ from the kernel by more than reordering (crossexam r2
    p22/p23), plus nearest (decomposed; agrees exactly)."""
    out = {"opaque_ops": {}, "opacity_unknown_ops": {}, "decomposition_disagrees": {}, "decomposed_ops": {}}
    for mode, src, dst in _RESIZES + (("nearest", (224, 224), (384, 384)),):
        torch.manual_seed(0)
        x = torch.randn(1, 3 if len(src) == 2 else 4, *src) * 2
        a = record(Fn(lambda s, x_: torch.nn.functional.interpolate(x_, size=dst, mode=mode, **({} if mode == "nearest" else {"align_corners": False}))),
                   {"x": x})[1]
        for k in out:
            out[k].update(a.get(k) or {})
    return out


_RESIZE_REVIEWS = ["aten::upsample_bilinear2d", "aten::upsample_bicubic2d", "aten::upsample_bicubic2d.vec",
                   "aten::upsample_linear1d", "aten::upsample_linear1d.vec"]


@C("T4", "VALID", {"PASS"}, "F10-10: plain bilinear / bicubic / linear resizes at sizes where PyTorch's decomposition rounds differently (224->384, 236->436, 249->382): reviewed, recorded whole; nearest stays decomposed")
def _t4():
    a = _resize_sweep()
    dec = a.get("decomposed_ops") or {}
    return _also(a, lambda: "aten::upsample_nearest2d" in dec and not any(k in dec for k in _RESIZE_REVIEWS), dec)


@C("T5", "NEG", {"UNKNOWN"}, "F10-10: the same sweep with the resize review entries removed: the decompositions disagree (UNKNOWN again)")
def _t5():
    return _without_reviews(_RESIZE_REVIEWS, _resize_sweep)


@C("T6", "NEG", {"UNKNOWN"}, "F10-9 (Fable R2-S1): a _safe_softmax kernel that scales every probability below 1e-5 by 0.7 on a 20 000-wide row is caught")
def _t6():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 20000) * 3
    lib = torch.library.Library("aten", "IMPL")
    def bad(x, dim, dtype=None):
        p = torch.softmax(x, dim)
        return torch.where(p < 1e-5, p * 0.7, p)
    lib.impl("_safe_softmax", bad, "CPU")
    try:
        return record(Fn(lambda s, x: torch.ops.aten._safe_softmax(x, -1)), {"x": q}, math=False)[1]
    finally:
        lib._destroy()


@C("T7", "VALID", {"PASS"}, "F10-9: 24 ordinary decomposed ops at large sizes (the Opus r2 sweep: cancellation sums, var_mean, logsumexp, sdpa, baddbmm K=512, norms, ...) still agree")
def _t7():
    F_ = torch.nn.functional
    torch.manual_seed(0)
    big = torch.randn(64, 4096); img = torch.randn(1, 3, 224, 224) * 2; seq = torch.randn(1, 12, 512, 64)
    cases = [(lambda x: x.sum(), big), (lambda x: (x - x.mean(-1, keepdim=True)).sum(), big), (lambda x: x.norm(dim=-1), big),
             (lambda x: torch.logsumexp(x * 10, -1), big), (lambda x: torch.var_mean(x, -1)[0], big), (lambda x: x.std(-1), big),
             (lambda x: F_.scaled_dot_product_attention(x, x, x, attn_mask=torch.ones(512, 512, dtype=torch.bool).tril()), seq),
             (lambda x: torch.baddbmm(torch.randn(8, 64, 64), x.reshape(8, 64, -1)[:, :, :512], x.reshape(8, 512, -1)[:, :, :64], beta=0.5, alpha=0.125), big),
             (lambda x: F_.glu(x, -1), big), (lambda x: torch.linalg.vector_norm(x, 3, -1), big),
             (lambda x: torch.addmv(torch.randn(64), x, torch.randn(4096)), big), (lambda x: torch.mv(x, torch.randn(4096)), big),
             (lambda x: torch.cumprod(1 + x[:, :64] * 0.01, -1), big), (lambda x: F_.normalize(x, dim=-1) * 3, big),
             (lambda x: F_.hardswish(x * 4), big), (lambda x: F_.batch_norm(x, None, None, training=True), big),
             (lambda x: F_.instance_norm(x.reshape(1, 64, 4096)), big), (lambda x: torch.cdist(x[:32, :256], x[32:, :256]), big),
             (lambda x: F_.unfold(x, 16, stride=16), img), (lambda x: torch.roll(x, 7, -1), big),
             (lambda x: F_.layer_norm(x, (4096,)), big), (lambda x: torch.softmax(x * 3, -1), big),
             (lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6), big)]
    out = {"opaque_ops": {}, "opacity_unknown_ops": {}, "decomposition_disagrees": {}, "decomposed_ops": {}}
    for f, x in cases:
        a = record(Fn(lambda s, x_, f=f: f(x_)), {"x": x}, skip=False)[1]
        for k in out:
            out[k].update(a.get(k) or {})
    a = record(Fn(lambda s, x_: F_.scaled_dot_product_attention(x_, x_, x_, is_causal=True)), {"x": seq}, math=False)[1]
    for k in out:
        out[k].update(a.get(k) or {})
    return out


@C("T8", "CHECK", {"TRUE"}, "F10-9/15/16/21/22 pinned: TOL_ULPS 64, SLACK_ULPS 8, ROBUST_RATIO 1e6, BUCKET_EPS 128, REPEAT_SHARE 0.01, CORE_QUANTILE 0.25, LEVEL_SIGNIFICANCE 1e-6, COPY_CAP 256 MB, OUTPUT_CAP 1 GB; F10-20 coupling SLACK_ULPS x eps(float32) x ROBUST_RATIO < 1 (an entry just under the ratio gets less slack than the typical value; false at half precision, hence F10-20). F10-21 deleted the F10-13 sentinel threshold (a repeated finfo.min mask is a repeated position, a lone one a ratio outlier): SENTINEL_FRACTION must be gone")
def _t8():
    import dag as _d
    want = {"TOL_ULPS": 64, "SLACK_ULPS": 8, "ROBUST_RATIO": 1e6, "BUCKET_EPS": 128, "REPEAT_SHARE": 0.01, "CORE_QUANTILE": 0.25,
            "LEVEL_SIGNIFICANCE": 1e-6,
            "COPY_CAP": 256 << 20, "OUTPUT_CAP": 1 << 30}
    detail = {k: getattr(_d, k, None) for k in want}
    coupling = (getattr(_d, "SLACK_ULPS", 0) or 0) * torch.finfo(torch.float32).eps * (getattr(_d, "ROBUST_RATIO", 0) or 0)
    detail["SLACK_ULPS x eps32 x ROBUST_RATIO"] = coupling
    detail["SENTINEL_FRACTION present"] = hasattr(_d, "SENTINEL_FRACTION")
    return CHECK(all(detail[k] == v for k, v in want.items()) and 0 < coupling < 1 and not hasattr(_d, "SENTINEL_FRACTION"), detail)


@C("T9", "NEG", {"UNKNOWN"}, "F10-9: non-finite values must match exactly (a roll kernel that returns -inf where its decomposition gives +inf: refuses X4)")
def _t9():
    lib = torch.library.Library("aten", "IMPL")
    def bad(x, shifts, dims=()):
        r = torch.cat([x[..., -1:], x[..., :-1]], -1)
        return torch.where(torch.isposinf(r), -r, r)
    lib.impl("roll", bad, "CPU")
    try:
        x = X.clone(); x[0, 0, 3] = float("inf")
        return record(Fn(lambda s, x_: torch.roll(x_, 1, -1)), {"x": x})[1]
    finally:
        lib._destroy()


# ---- F10-11: a view-only decomposition is transparent by that evidence (no copies, no value check) -----------------
def _count_copies(thunk):
    import dag as _d
    n = [0]
    saved = _d._layout_copy
    def counting(*a, **k):
        n[0] += 1
        return saved(*a, **k)
    _d._layout_copy = counting
    try:
        return thunk(), n[0]
    finally:
        _d._layout_copy = saved


@C("U1", "VALID", {"PASS"}, "F10-11: aten::_unsafe_view (decomposition = aten::view only) is transparent with no body, no copy, no value check")
def _u1():
    def run():
        return record(Fn(lambda s, x: torch.ops.aten._unsafe_view(x * 1, [10, 8])), {"x": X})
    (rec, a), nc = _count_copies(run)
    n = [n_ for n_ in rec.nodes if n_.op == "aten::_unsafe_view"]
    return _also(a, lambda: n and all(n_.body is None and n_.opq is None for n_ in n) and nc == 0 and "aten::_unsafe_view" not in (a.get("decomposed_ops") or {}),
                 {"copies": nc, "decomposed": a.get("decomposed_ops")})


@C("U2", "NEG", {"UNKNOWN"}, "F10-11: a decomposition with arithmetic (the _unsafe_view entry replaced by (x*2).view) is not caught by the view rule: recorded, disagrees")
def _u2():
    import opclass
    table = opclass.decompositions(); key = torch.ops.aten._unsafe_view.default; saved = table.get(key)
    vo = getattr(opclass, "_VIEW_ONLY", {}); vo_saved = dict(vo); vo.clear()
    table[key] = lambda self, size: (self * 2).view(size)
    try:
        return record(Fn(lambda s, x: torch.ops.aten._unsafe_view(x * 1, [10, 8])), {"x": X})[1]
    finally:
        table[key] = saved; vo.clear(); vo.update(vo_saved)


# ---- F10-12: copies of zero storage are shared zero views; other copies capped; unannotated zero-storage writes --------
_Q1 = r"""
import sys, resource; sys.path.insert(0, sys.argv[1])
import torch, lowcost, dag
from dag import DagRecorder, analyse
w = lowcost._zero_view((256, 1024, 1024), torch.float32); lowcost._register(w)
class Z(torch.nn.Module):
    def forward(self, x):
        w.zero_(); return x.sum()
m = Z(); r = DagRecorder(m, {"x": torch.randn(2)}); b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with torch.no_grad(), r: o = m(torch.randn(2))
r.remove_hooks(); a = analyse(r, o)
d = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - b) / 2**20
print("RESULT", int(d), bool(a["decomposed_ops"].get("aten::zero_")), not a["decomposition_disagrees"], not a["opacity_unknown_ops"],
      lowcost.registry_intact(), w.untyped_storage().data_ptr() in lowcost.ZERO_REG)
"""


@C("Z6", "CHECK", {"TRUE"}, "F10-12 (Fable R2-S3 q1): zero_ on a (256,1024,1024) registered zero view is recorded as its decomposition with < 64 MB extra memory (was +1.5 GB)")
def _z6():
    r = subprocess.run([sys.executable, "-c", _Q1, os.path.abspath(H)], capture_output=True, text=True, timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
    if line is None:
        return CHECK(False, r.stderr[-300:])
    f = line.split()
    return CHECK(int(f[1]) < 64 and all(x == "True" for x in f[2:]), line)


@C("Z7", "NEG", {"UNKNOWN"}, "F10-12: a private copy over the cap is not made: the op is recorded whole and is a NAMED G4 unknown (cap lowered to 1 KB for the control)")
def _z7():
    import dag as _d
    saved = getattr(_d, "COPY_CAP", None)
    _d.COPY_CAP = 1024
    try:
        def f(s, x):
            y = torch.randn(64, 64); y.masked_fill_(y > 0, torch.tensor(0.5))
            return y
        a = record(Fn(f), {"x": X})[1]
    finally:
        if saved is not None:
            _d.COPY_CAP = saved
    why = (a.get("opacity_unknown_reasons") or {}).get("aten::masked_fill_.Tensor", "")
    return a if "cap" in why else CHECK(False, {"reasons": a.get("opacity_unknown_reasons"), "T4": t4(a)})


@C("Z8", "CHECK", {"TRUE"}, "F10-12 (Opus N-R2-1): BatchNorm training whose running stats are REGISTERED zero storage: the decomposition never writes the program's tensors; stats equal plain torch; registry intact")
def _z8():
    def mk():
        bn = torch.nn.BatchNorm1d(6, momentum=0.5)
        bn._buffers["running_mean"] = lowcost._zero_view((6,), torch.float32)
        bn._buffers["running_var"] = lowcost._zero_view((6,), torch.float32) + 0     # (a real buffer: var starts at 0 here)
        return bn.train()
    torch.manual_seed(1); xb = torch.randn(3, 6) * 3 + 5
    plain = torch.nn.BatchNorm1d(6, momentum=0.5).train()
    plain.running_var.zero_()
    with torch.no_grad():
        plain(xb)
    bn = mk()
    rec, a = record(bn, {"input": xb})                # (the BatchNorm module itself: its buffers are the model's own)
    ok = torch.equal(bn.running_mean, plain.running_mean) and torch.equal(bn.running_var, plain.running_var) and lowcost.registry_intact()
    return CHECK(ok and t4(a) == "PASS", {"recorded": bn.running_mean.tolist()[:2], "plain": plain.running_mean.tolist()[:2],
                                          "registry_intact": lowcost.registry_intact(), "T4": t4(a)})


# ---- round-2 mutation survivors -------------------------------------------------------------------------------------
@C("B2", "NEG", {"UNKNOWN"}, "Fable R2M3 / Opus X3: a BatchNorm kernel with the right output but a wrong running-stat write: the decomposition's copy is compared, UNKNOWN")
def _b2():
    lib = torch.library.Library("aten", "IMPL")
    def bad(inp, weight, bias, rm, rv, training, momentum, eps):
        out = torch.ops.aten._native_batch_norm_legit(inp, weight, bias, rm, rv, training, momentum, eps)
        rm.add_(1.0)
        return out
    lib.impl("native_batch_norm", bad, "CPU")
    try:
        bn = torch.nn.BatchNorm1d(8, momentum=0.5).train()
        return record(Fn(lambda s, x: bn(x.reshape(-1, 8))), {"x": X})[1]
    finally:
        lib._destroy()


@C("B3", "NEG", {"UNKNOWN"}, "a kernel that changes an operand its decomposition only reads (the softmax input scaled in place): the operand fingerprint catches it")
def _b3():
    lib = torch.library.Library("aten", "IMPL")
    def bad(x, dim, dtype=None):
        p = torch.softmax(x, dim)
        x.mul_(1.5)
        return p
    lib.impl("_safe_softmax", bad, "CPU")
    try:
        return record(Fn(lambda s, x: torch.ops.aten._safe_softmax(x * 1, -1)), {"x": X}, math=False)[1]
    finally:
        lib._destroy()


@C("P12", "NEG", {"FAIL"}, "Fable R2M5 / Opus X11: a direct prims::div AFTER a decomposition that emitted prims::div (var_mean) stays opaque")
def _p12():
    return _ctl_fn(lambda s, x: torch.var_mean(x, -1)[0].sum() + torch.ops.prims.div(x, x).sum())


def _aten_two():
    """An aten-namespace in-place op writing TWO arguments whose functional twin (same arguments) is pointwise."""
    if not hasattr(torch.ops.aten, "f10ctl_two_"):
        lib = torch.library.Library("aten", "FRAGMENT")
        lib.define("f10ctl_two(Tensor self, Tensor other) -> Tensor", tags=(torch.Tag.pointwise,))
        lib.define("f10ctl_two_(Tensor(a!) self, Tensor(b!) other) -> Tensor(a!)")
        lib.impl("f10ctl_two", lambda a, b: a + b, "CPU")
        def two_(a, b):
            b.mul_(3.0); a.add_(b)
            return a
        lib.impl("f10ctl_two_", two_, "CPU")
        _aten_two.lib = lib
    return torch.ops.aten.f10ctl_two_


@C("S9", "NEG", {"FAIL"}, "Fable R2M11: an in-place op that writes two arguments is not an in-place variant of its (transparent) functional twin")
def _s9():
    op = _aten_two()
    def f(s, x):
        y, z = x.clone(), x.clone()
        return op(y, z) + z
    return _ctl_fn(f)


def _aten_outw():
    """An aten-namespace `.out` overload that ALSO writes a positional argument (a hidden side effect); its functional
    twin (same arguments without the written ones) is pointwise."""
    if not hasattr(torch.ops.aten, "f10ctl_o"):
        lib = torch.library.Library("aten", "FRAGMENT")
        lib.define("f10ctl_o(Tensor self) -> Tensor", tags=(torch.Tag.pointwise,))
        lib.define("f10ctl_o.out(Tensor self, Tensor(a!) scratch, *, Tensor(b!) out) -> Tensor(b!)")
        lib.impl("f10ctl_o", lambda a: a * 2, "CPU")
        def o_out(a, scratch, out):
            scratch.mul_(5.0); out.copy_(a * 2)
            return out
        lib.impl("f10ctl_o.out", o_out, "CPU")
        _aten_outw.lib = lib
    return torch.ops.aten.f10ctl_o.out


@C("S10", "NEG", {"FAIL"}, "Opus X6: an out= overload that also writes a POSITIONAL argument is not an out= variant of its functional twin")
def _s10():
    op = _aten_outw()
    def f(s, x):
        sc, o = x.clone(), torch.empty_like(x)
        return op(x, sc, out=o) + sc
    return _ctl_fn(f)


_EXT = torch.randn(4, 4)


@C("F3", "CHECK", {"TRUE"}, "Fable R2M15 / Opus X5: a decomposition that reads an outside tensor and makes NaN, then raises, leaves no external edge and no non-finite event")
def _f3():
    import opclass, dag as _d
    table = opclass.decompositions(); key = torch.ops.aten.stack.default; saved = table.get(key)
    def boom(tensors, dim=0):
        (tensors[0][..., :4] @ _EXT).sum()                        # an external tensor read
        z = tensors[0] * 0; z / z                                   # NaN from finite values
        raise RuntimeError("decomposition failed (control)")
    table[key] = boom
    n_nf = len(_d.NONFINITE_EVENTS)
    try:
        rec, a = record(StackNorm(), {"x": X})
    finally:
        table[key] = saved
    return CHECK(t4(a) == "UNKNOWN" and a.get("external_nonscalar") == 0 and not rec.external and len(_d.NONFINITE_EVENTS) == n_nf,
                 {"external": a.get("external_nonscalar"), "nonfinite_new": _d.NONFINITE_EVENTS[n_nf:][:3], "T4": t4(a)})


@C("N13", "CHECK", {"TRUE"}, "Fable R2M17: the classifier checks EVERY operand type, not only the first (Parameter first, a dispatching subclass second)")
def _n13():
    import opclass
    k, why = opclass.decide("aten::mm", (torch.nn.Parameter, DequantWeight))
    return CHECK(k == "opaque" and "subclass" in why, (k, why))


class DispParam(torch.nn.Parameter):
    """A Parameter subclass with its own __torch_dispatch__ (it doubles every result outside the recording)."""
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return torch._C._disabled_torch_dispatch_impl(func, types, args, kwargs) * 2


@C("N14", "NEG", {"FAIL"}, "Opus X14: a Parameter SUBCLASS with its own __torch_dispatch__ is opaque (only nn.Parameter itself is exempt)")
def _n14():
    class M(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.w = DispParam(torch.randn(5, 8))
        def forward(s, x):
            return x + s.w
    return record(M(), {"x": X})[1]


@C("W12", "CHECK", {"TRUE"}, "Opus X10: diffusion T4 on a recording without a returned structure counts an unreproduced output as read (flash logsumexp returned): G4 unknown")
def _w12():
    def t_fn(s, x):
        q = x.reshape(2, 5, 2, 4).transpose(1, 2).contiguous()
        return torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(q, q, q)
    class P2(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.transformer = Fn(t_fn)
        def forward(s, x):
            o, lse = s.transformer(x)
            return o, lse
    m = P2()
    rec = DagRecorder(m, {"x": X})                  # (no analyse(): the returned structure stays unknown, as in diffusion)
    try:
        with torch.no_grad(), rec:
            m(X)
    finally:
        rec.remove_hooks()
    c, u = _diffusion_t4([rec])
    return CHECK(getattr(rec, "returns", None) is None and c["pass"] and len(u) == 1, (c, u))


# ---- Fable R2-N1 / F10-24: a vocabulary torch rejects degrades per entry (a visible NOTE on the row, never an unknown) -
_Y1 = r"""
import sys, os, shutil, tempfile
src = sys.argv[1]; d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "everchanging"))
shutil.copy(os.path.join(src, "opclass.py"), d)
open(os.path.join(d, "everchanging", "opacity_reviewed.yaml"), "w").write(
    'reviewed:\n  "aten::_grouped_mm": "kept"\n  "aten::_grouped_mm.default": "a typo"\n  "aten::_unsafe_view": "now decomposable"\n')
sys.path.insert(0, d)
import opclass
vn = opclass.vocabulary_note()
print("RESULT", sorted(opclass.REVIEWED), len(opclass.VOCAB_ERRORS), bool(vn and vn.get("count") == 2 and "predicate" not in vn),
      opclass.decide("aten::_unsafe_view")[0])
"""


@C("Y1", "CHECK", {"TRUE"}, "Fable R2-N1 / F10-24: the import survives a vocabulary with rejected entries (a typo, i.e. not a real overload; a decomposable entry); they credit nothing (_unsafe_view is graded by its evidence: decompose); a NOTE (no predicate) names both")
def _y1():
    r = subprocess.run([sys.executable, "-c", _Y1, os.path.abspath(H)], capture_output=True, text=True, timeout=120)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
    return CHECK(line == "RESULT ['aten::_grouped_mm'] 2 True decompose", line or r.stderr[-300:])


@C("Y2", "CHECK", {"TRUE"}, "F10-24 (supersedes Fable R2-N1's whole-row unknown): with a rejected entry for an op that did not run, the worker's and the diffusion worker's own T4 lines add NO unknown and set the row's vocabulary_notes")
def _y2():
    import opclass
    saved = list(getattr(opclass, "VOCAB_ERRORS", []))
    if not hasattr(opclass, "VOCAB_ERRORS"):
        return CHECK(False, "no VOCAB_ERRORS")
    opclass.VOCAB_ERRORS[:] = ["'aten::x' is not the exact name of a real overload"]
    try:
        c, u = _t4_worker({"text": _p()}); n1 = dict(_LAST_ROW)
        c2, u2 = _diffusion_t4([_pipe_rec(lambda s, x: x * 2)]); n2 = dict(_LAST_ROW)
    finally:
        opclass.VOCAB_ERRORS[:] = saved
    note = lambda r: (r.get("vocabulary_notes") or {}).get("count") == 1 and "predicate" not in (r.get("vocabulary_notes") or {"predicate": 1})
    return CHECK(c["pass"] and not u and c2["pass"] and not u2 and note(n1) and note(n2), (u, u2, n1, n2))


@C("Y3", "CHECK", {"TRUE"}, "F10-24 fail-closed: a rejected entry whose op RUNS and is opaque in this torch (aten::searchsorted.Tensor inside isin's decomposition, its review dropped) is graded on its own evidence: T4 FAILS naming the op, and the row carries the note")
def _y3():
    import opclass
    saved = list(opclass.VOCAB_ERRORS)
    opclass.VOCAB_ERRORS[:] = ["'aten::searchsorted.Tensor' rejected by this torch (simulated)"]
    try:
        a = _without_review("aten::searchsorted.Tensor", lambda: _ctl_fn(
            lambda s, x: x + torch.isin((x.reshape(-1)[:_ISIN_N] * 100).round().long(), torch.arange(1000)).sum()))
        c, u = _t4_worker({"text": dict(a, ok=True)}); n = dict(_LAST_ROW)
    finally:
        opclass.VOCAB_ERRORS[:] = saved
    return CHECK(c["pass"] is False and "aten::searchsorted.Tensor" in c["opaque_ops"] and (n.get("vocabulary_notes") or {}).get("count") == 1,
                 (c.get("opaque_ops"), u, n))


@C("Y4", "CHECK", {"TRUE"}, "F10-24: a rejected 'redundant' entry whose op runs and is transparent / decomposable by evidence in this torch (the vLLM venv's _fft_c2r is core, adaptive_max_pool2d decomposes): the ops are graded by that evidence, no unknown, T4 passes, the row carries the note (aten::add.Tensor core and aten::var_mean decomposed stand in)")
def _y4():
    import opclass
    saved = list(opclass.VOCAB_ERRORS)
    opclass.VOCAB_ERRORS[:] = ["'aten::add.Tensor' needs no review (it is transparent: core)",
                               "'aten::var_mean.correction' is not opaque here (it is decompose)"]
    try:
        a = _ctl_fn(lambda s, x: torch.var_mean(x + 1.0, -1)[0])
        c, u = _t4_worker({"text": dict(a, ok=True)}); n = dict(_LAST_ROW)
    finally:
        opclass.VOCAB_ERRORS[:] = saved
    return CHECK(c["pass"] and not u and (n.get("vocabulary_notes") or {}).get("count") == 2
                 and "aten::var_mean.correction" in (c.get("decomposed_ops_info") or {}), (c, u, n))


@C("Y5", "CHECK", {"TRUE"}, "F10-24 witness side: a witness record whose recording is clean but whose venv rejected vocabulary entries (stored transparent False, as the 43 records made with torch 2.14.1) is transparent by the ONE rule the producer and the grader share (witness_io.witness_transparent), and the grader's credit raises no G4 for it; an opaque op in the record still does")
def _y5():
    import witness_io, common
    w = {"op_full": ["aten::mm"], "opaque_ops": {}, "opacity_unknown_ops": [], "decomposition_disagrees": [], "external_nonscalar": 0,
         "vocabulary_errors": ["'aten::_fft_c2r' needs no review (it is transparent: core)"], "transparent": False}
    preds = lambda r: sorted({p for p, _ in common.witness_execution_credit(r, "r" * 40)})
    got = {"rule": witness_io.witness_transparent(w), "credit G4": "G4" in preds(w),
           "opaque op: rule": witness_io.witness_transparent(dict(w, opaque_ops={"f10ctl::secret": "opaque"})),
           "opaque op: credit G4": "G4" in preds(dict(w, opaque_ops={"f10ctl::secret": "opaque"}))}
    return CHECK(got == {"rule": True, "credit G4": False, "opaque op: rule": False, "opaque op: credit G4": True}, got)


# ---- Fable R2-N2: a stage time limit interrupts a long decomposition promptly; the recorder's state is put back -------
class _Limit(Exception):
    pass


@C("A1", "CHECK", {"TRUE"}, "Fable R2-N2 / Opus N-R3-3: a stage limit firing inside a 3 s decomposition is delivered within 1 s; its nodes, external edges, non-finite events, private copies and random draws are all put back")
def _a1():
    import opclass, dag as _d
    table = opclass.decompositions(); key = torch.ops.aten.stack.default; saved = table.get(key)
    seen = {}
    def slow(tensors, dim=0):
        seen.update(nodes=len(rec.nodes), ext=len(rec.external), nf=len(_d.NONFINITE_EVENTS), rng=torch.get_rng_state())
        (tensors[0][..., :4] @ _EXT).sum()                  # nodes + an external edge
        z = tensors[0] * 0; z / z                           # a non-finite event
        torch.rand(100)                                      # random draws
        t0 = time.time()
        while time.time() - t0 < 3.0:
            time.sleep(0.01)
        return torch.cat([t.unsqueeze(dim) for t in tensors], dim)
    table[key] = slow
    old = _signal.signal(_signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Limit()))
    m = StackNorm(); rec = DagRecorder(m, {"x": X}); got, t0 = None, time.time()
    try:
        _signal.setitimer(_signal.ITIMER_REAL, 0.2)
        with torch.no_grad(), math_attention(), rec:
            m(X)
    except _Limit:
        got = "limit"
    except BaseException as e:
        got = type(e).__name__
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        _signal.signal(_signal.SIGALRM, old)
        rec.remove_hooks()
        table[key] = saved
    dt = time.time() - t0
    clean = {"nodes": len(rec.nodes) == seen.get("nodes"), "external": len(rec.external) == seen.get("ext"),
             "nonfinite": len(_d.NONFINITE_EVENTS) == seen.get("nf"), "private": not getattr(rec, "_private", {}),
             "rng": "rng" in seen and torch.equal(torch.get_rng_state(), seen["rng"]),
             "depth": getattr(rec, "_dec_depth", 0) == 0, "alias": not getattr(rec, "_alias", {})}
    return CHECK(got == "limit" and dt < 1.0 and all(clean.values()), {"raised": got, "secs": round(dt, 2), **clean})



# =====================================================================================================================
# F10 round 4 (re-check of round 3 by Opus crossexam/r3 + Fable crossexam_fable/r3; rulings F10-13, F10-14). Each control
# FAILS on the round-3 worktree (impl_r4/r3) unless marked as a mutant pin (it then refuses a named round-3 survivor).
# =====================================================================================================================
_FMIN = torch.finfo(torch.float32).min


@C("T10", "CHECK", {"TRUE"}, "F10-13 law: a finfo.min mask with its zeros tampered to 12345, masked scores +1000, and a sentinel changed to -inf all disagree; the untampered sentinel mask and masked scores agree")
def _t10():
    import dag as _d
    torch.manual_seed(0)
    S = 64
    m_in = torch.full((S, S), _FMIN); m_in[0, 0] = 0.0
    true = torch.triu(m_in, 1)
    tam = true.clone(); tam[torch.isfinite(tam) & (tam == 0)] = 12345.0
    scores = torch.randn(S, S) + torch.triu(torch.full((S, S), _FMIN), 1)
    t2 = scores.clone(); t2[scores > _FMIN / 2] += 1000.0
    t3 = true.clone(); t3[0, 5] = float("-inf")
    near = scores.clone(); near[scores > _FMIN / 2] *= (1 + 1e-6)          # within the law on the ordinary entries
    res = {"tampered zeros": _d._same_values(tam, true), "scores +1000": _d._same_values(t2, scores),
           "sentinel -> -inf": _d._same_values(t3, true), "untampered mask": _d._same_values(true.clone(), true),
           "scores within the law": _d._same_values(near, scores)}
    return CHECK(res == {"tampered zeros": False, "scores +1000": False, "sentinel -> -inf": False, "untampered mask": True,
                         "scores within the law": True}, res)


def _triu_tamper(x, diagonal=0):
    """A replaced triu kernel: correct above the diagonal, 12345 where it must write 0 (no triu call inside)."""
    r_ = torch.arange(x.shape[-2]).unsqueeze(-1); c_ = torch.arange(x.shape[-1]).unsqueeze(0)
    return torch.where(c_ - r_ >= diagonal, x, torch.full_like(x, 12345.0))


@C("T11", "NEG", {"UNKNOWN"}, "F10-13 (Fable r1_sentinel_slack): a replaced triu kernel writing 12345 where 0 belongs, on a finfo.min causal mask (the HF idiom), is caught")
def _t11():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("triu", _triu_tamper, "CPU")
    try:
        return _ctl_fn(lambda s, x: torch.triu(torch.full((16, 16), _FMIN), 1) + x[0, :, :1].sum())
    finally:
        lib._destroy()


@C("T12", "NEG", {"UNKNOWN"}, "F10-13 (Opus p30 A): a replaced where.ScalarOther kernel that doubles the kept scores, other=finfo.min, is caught")
def _t12():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("where.ScalarOther", lambda cond, self, other: torch.ops.aten.where.self(cond, self * 2.0, torch.full_like(self, other)), "CPU")
    try:
        return _ctl_fn(lambda s, x: torch.where(x > 0, x, _FMIN))
    finally:
        lib._destroy()


@C("T13", "NEG", {"UNKNOWN"}, "F10-13 (Opus p30 A): a replaced masked_fill.Tensor kernel that doubles the kept scores, value=finfo.min, is caught")
def _t13():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("masked_fill.Tensor", lambda self, mask, value: torch.ops.aten.where.self(mask, value.to(self.dtype), self * 2.0), "CPU")
    try:
        return _ctl_fn(lambda s, x: x.masked_fill(x < 0, torch.tensor(_FMIN)))
    finally:
        lib._destroy()


@C("T14", "VALID", {"PASS"}, "F10-13: ordinary finfo.min masks still agree (causal mask full(finfo.min).triu(1), where(.., finfo.min), masked_fill(.., finfo.min), softmax over them)")
def _t14():
    def f(s, x):
        sc = x[0] @ x[0].transpose(-1, -2)
        cm = torch.triu(torch.full(sc.shape, _FMIN), 1)
        a = torch.softmax(sc + cm, -1)
        b = torch.where(sc > 0, sc, _FMIN)
        c = sc.masked_fill(sc < 0, torch.tensor(_FMIN))
        return a.sum() + torch.softmax(b, -1).sum() + torch.softmax(c, -1).sum() + torch.tril(torch.full(sc.shape, _FMIN), -1).sum()
    return _ctl_fn(f)


_TRI = (((12, 16, 38), (18, 59, 69)), ((8, 35, 34), (6, 41, 73)), ((7, 10, 38), (13, 72, 29)))
_TRI_REVIEWS = ["aten::upsample_trilinear3d", "aten::upsample_trilinear3d.vec"]


def _tri_sweep():
    """trilinear sizes whose decomposition rounds source coordinates differently (crossexam r3 p31), + nearest 3-D."""
    out = {"opaque_ops": {}, "opacity_unknown_ops": {}, "decomposition_disagrees": {}, "decomposed_ops": {}}
    for src, dst in _TRI:
        for mode in ("trilinear", "nearest"):
            torch.manual_seed(0)
            x = torch.randn(1, 2, *src) * 2
            a = record(Fn(lambda s, x_: torch.nn.functional.interpolate(x_, size=dst, mode=mode, **({} if mode == "nearest" else {"align_corners": False}))),
                       {"x": x})[1]
            for k in out:
                out[k].update(a.get(k) or {})
    return out


@C("T15", "VALID", {"PASS"}, "F10-14: trilinear 3-D resizes at sizes where the decomposition rounds differently pass, recorded whole; nearest 3-D stays decomposed")
def _t15():
    a = _tri_sweep()
    dec = a.get("decomposed_ops") or {}
    return _also(a, lambda: "aten::upsample_nearest3d" in dec and not any(k in dec for k in _TRI_REVIEWS), dec)


@C("T16", "NEG", {"UNKNOWN"}, "F10-14: the trilinear sweep with its review entries removed is UNKNOWN again")
def _t16():
    return _without_reviews(_TRI_REVIEWS, _tri_sweep)


# ---- Fable R3-S1: the view-only cache key is the whole call ----------------------------------------------------------
@C("U3", "CHECK", {"TRUE"}, "Fable R3-S1: channel_shuffle groups=1 is view-only; the following groups=2 call (same tensor) is recorded as its decomposition, not credited by the cache")
def _u3():
    import opclass
    getattr(opclass, "_VIEW_ONLY", {}).clear()
    def f(s, x):
        return torch.nn.functional.channel_shuffle(x, 1).sum() + torch.nn.functional.channel_shuffle(x, 2).sum()
    rec, a = record(Fn(f), {"x": torch.randn(1, 4, 3, 5)})
    cs = [(n.body is not None, n.opq) for n in rec.nodes if n.op == "aten::channel_shuffle"]
    return CHECK(cs == [(False, None), (True, None)] and t4(a) == "PASS", cs)


@C("U4", "CHECK", {"TRUE"}, "Fable R3-S1: native_channel_shuffle groups=1 on a contiguous 6x6 is view-only; its transpose (same shape, other strides) is not credited by that cache entry")
def _u4():
    import opclass
    getattr(opclass, "_VIEW_ONLY", {}).clear()
    def f(s, x):
        y = x.reshape(1, 4, 6, 6)
        return torch.ops.aten.native_channel_shuffle(y, 1).sum() + torch.ops.aten.native_channel_shuffle(y.transpose(2, 3), 1).sum()
    rec, a = record(Fn(f), {"x": torch.randn(144)})
    # the first call is view-only (no body, transparent); the second is NOT credited by the first call's cache entry:
    # PyTorch's decomposition is recorded, or (here: it calls view on a non-viewable tensor) it raises -> a G4 unknown
    cs = [(n.body is not None, n.opq) for n in rec.nodes if n.op == "aten::native_channel_shuffle"]
    ok = len(cs) == 2 and cs[0] == (False, None) and (cs[1][0] or str(cs[1][1]).startswith("unknown:"))
    return CHECK(ok, cs)


# ---- Fable R3-S2: a stage limit inside any `except Exception` bookkeeping reaches the caller --------------------------
def _limit_inside(arm):
    """Run a small recording with a raising SIGALRM handler and `arm` (a context manager) raising the signal from inside
    one bookkeeping function; -> whether the limit reached the caller."""
    old = _signal.signal(_signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Limit()))
    got = None
    try:
        with arm():
            m = Fn(lambda s, x: torch.roll(x, 1, -1) * 2 + torch.bernoulli(torch.full_like(x, 0.5)))
            rec = DagRecorder(m, {"x": X})
            try:
                with torch.no_grad(), math_attention(), rec:
                    m(X)
            finally:
                rec.remove_hooks()
    except _Limit:
        got = True
    except BaseException as e:
        got = type(e).__name__
    finally:
        _signal.signal(_signal.SIGALRM, old)
    return got is True


def _once(obj, attr, when=lambda *a, **k: True):
    """Patch obj.attr so that its FIRST qualifying call raises SIGALRM inside it, then behaves as before."""
    @contextlib.contextmanager
    def cm():
        orig = getattr(obj, attr); fired = []
        def wrapped(*a, **k):
            if not fired and when(*a, **k):
                fired.append(1); _signal.raise_signal(_signal.SIGALRM)
            return orig(*a, **k)
        setattr(obj, attr, wrapped)
        try:
            yield
        finally:
            setattr(obj, attr, orig)
    return cm


@contextlib.contextmanager
def _arm_view_trace():
    """The signal fires INSIDE view_only_decomposition's META trace (roll's decomposition, on meta operands)."""
    import opclass
    getattr(opclass, "_VIEW_ONLY", {}).clear()
    table = opclass.decompositions(); key = torch.ops.aten.roll.default; orig = table[key]; fired = []
    def dec(x, shifts, dims=()):
        if x.device.type == "meta" and not fired:
            fired.append(1); _signal.raise_signal(_signal.SIGALRM)
        return orig(x, shifts, dims)
    table[key] = dec
    try:
        yield
    finally:
        table[key] = orig


@C("A2", "CHECK", {"TRUE"}, "Fable R3-S2: a stage limit raised inside the view-only trace, bind_args, _write_targets, a probe copy (_layout_copy) or _fingerprint reaches the caller (never swallowed)")
def _a2():
    import dag as _d
    res = {"view-only trace": _limit_inside(_arm_view_trace),
           "bind_args": _limit_inside(_once(_d, "_schema_of")),
           "_write_targets": _limit_inside(_once(_d, "_write_targets")),
           "_layout_copy (random probe)": _limit_inside(_once(torch, "empty_strided")),
           "_fingerprint": _limit_inside(_once(torch.Tensor, "view", lambda self, *a, **k: bool(a) and isinstance(a[0], torch.dtype)))}
    return CHECK(all(res.values()), res)


# ---- Fable R3-S3: the decomposition's own outputs count toward the cap ------------------------------------------------
@C("Z9", "NEG", {"UNKNOWN"}, "F10-16: a decomposed op whose OUTPUTS exceed the output cap (lowered to 64 bytes for the control; 1 GB in a row) is recorded whole, a named unknown")
def _z9():
    import dag as _d
    name = "OUTPUT_CAP" if hasattr(_d, "OUTPUT_CAP") else "COPY_CAP"
    saved = getattr(_d, name, None); setattr(_d, name, 64)
    try:
        a = record(Fn(lambda s, x: torch.ops.aten._safe_softmax(x, -1)), {"x": X}, math=False)[1]
    finally:
        if saved is not None:
            setattr(_d, name, saved)
    why = (a.get("opacity_unknown_reasons") or {}).get("aten::_safe_softmax", "")
    return a if "cap" in why else CHECK(False, {"reasons": a.get("opacity_unknown_reasons"), "T4": t4(a)})


_Q4 = r"""
import sys, resource; sys.path.insert(0, sys.argv[1])
import torch, dag
from dag import DagRecorder, analyse
m = torch.nn.Module()
q, k, v = (torch.randn(1, 24, 2048, 64) for _ in range(3))
f = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
with dag.math_attention(): f()
b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
r = DagRecorder(m, {})
with torch.no_grad(), dag.math_attention(), r: o = f()
r.remove_hooks(); a = analyse(r, o)
print("RESULT", int((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - b) / 2**20), bool(a.get("decomposed_ops", {}).get("aten::_safe_softmax")),
      not a.get("decomposition_disagrees"), not a.get("opacity_unknown_ops"))
"""


@C("Z10", "CHECK", {"TRUE"}, "F10-16: a 384 MB attention softmax (24 heads x 2048 tokens, starcoder2-3b-like) is decomposed and value-checked chunk by chunk, with extra memory under 3x the output (an unchunked compare needs ~4.8x)")
def _z10():
    r = subprocess.run([sys.executable, "-c", _Q4, os.path.abspath(H)], capture_output=True, text=True, timeout=900)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
    if line is None:
        return CHECK(False, r.stderr[-300:])
    f = line.split()
    # (r5 measures +462..+883 MB: the decomposition's own 384 MB output lives beside the op's; the unchunked float64
    #  compare reads +1835 MB (impl_r5/receipts); maxrss is noisy under load, so the bound sits between the two)
    return CHECK(int(f[1]) < 1152 and all(x == "True" for x in f[2:]), line)


# ---- seeded ops: over the cap; a raising decomposition ----------------------------------------------------------------
@C("R6", "NEG", {"UNKNOWN"}, "Fable R3M14: a seeded op whose two-seed probe copies exceed the copy cap (lowered below them) is a named unknown, never credited as random without the probe")
def _r6():
    import dag as _d
    saved = getattr(_d, "COPY_CAP", None)
    _d.COPY_CAP = 64 if hasattr(_d, "OUTPUT_CAP") else X.numel() * 4 * 3 // 2       # (r5: probe copies 320 B > 64 B; outputs separately capped)
    try:
        a = record(Fn(lambda s, x: torch.bernoulli(torch.full_like(x, 0.5))), {"x": X})[1]
    finally:
        if saved is not None:
            _d.COPY_CAP = saved
    why = (a.get("opacity_unknown_reasons") or {}).get("aten::bernoulli", "")
    return a if ("cap" in why and not a.get("decomposed_random_info")) else CHECK(False, {"reasons": a.get("opacity_unknown_reasons"), "random": a.get("decomposed_random_info")})


@C("R7", "CHECK", {"TRUE"}, "Opus Y4: a seeded op whose decomposition draws, then raises: the op runs from its own start, the program's values and stream equal plain torch")
def _r7():
    import opclass
    table = opclass.decompositions(); key = torch.ops.aten.bernoulli.default; saved = table[key]
    def boom(x, *a, **k):
        torch.rand(1000)                                   # draws from the stream, then fails
        raise RuntimeError("decomposition failed (control)")
    mod = Fn(lambda s, x: x + torch.bernoulli(torch.full_like(x, 0.5)))
    torch.manual_seed(7)
    with torch.no_grad():
        plain = mod(x=X)
    nxt_plain = torch.rand(3)
    table[key] = boom
    try:
        torch.manual_seed(7)
        rec = DagRecorder(mod, {"x": X})
        try:
            with torch.no_grad(), math_attention(), rec:
                got = mod(x=X)
        finally:
            rec.remove_hooks()
        nxt = torch.rand(3)
    finally:
        table[key] = saved
    return CHECK(torch.equal(plain, got) and torch.equal(nxt, nxt_plain), {"values": torch.equal(plain, got), "stream": torch.equal(nxt, nxt_plain)})


# ---- the write guard protects every active frame (Opus Y3) -----------------------------------------------------------
_NEST = {}


@C("Z11", "CHECK", {"TRUE"}, "Opus Y3: a nested decomposition that writes a program tensor it was not passed (one the OUTER decomposition reads) is stopped: the program's values equal plain torch")
def _z11():
    import opclass
    table = opclass.decompositions(); ks, kr = torch.ops.aten.stack.default, torch.ops.aten.roll.default
    ss, sr = table[ks], table[kr]
    def stack_dec(tensors, dim=0):
        _NEST["victim"] = tensors[1]
        r = torch.roll(tensors[0], 1, -1)              # a nested decomposed op
        return ss([r * 0 + tensors[0], tensors[1]], dim)
    def roll_dec(x, shifts, dims=()):
        v = _NEST.get("victim")
        if v is not None:
            v.mul_(2.0)                                # writes a tensor of the OUTER call it was never passed
        return sr(x, shifts, dims)
    m = Fn(lambda s, x: torch.stack([x * 1, x * 3], 0).sum(0))
    with torch.no_grad():
        plain = m(x=X)
    table[ks], table[kr] = stack_dec, roll_dec
    try:
        rec = DagRecorder(m, {"x": X})
        try:
            with torch.no_grad(), math_attention(), rec:
                got = m(x=X)
        finally:
            rec.remove_hooks()
    finally:
        table[ks], table[kr] = ss, sr; _NEST.clear()
    return CHECK(torch.equal(plain, got), float((plain - got).abs().max()))


# ---- position-weighted fingerprint (Opus N-R3-1) ----------------------------------------------------------------------
def _roll_mutating(mut):
    lib = torch.library.Library("aten", "IMPL")
    def bad(self, shifts, dims=()):
        with torch._C._DisableTorchDispatch():
            r = torch.cat([self[..., -1:], self[..., :-1]], -1).clone()
            mut(self)
        return r
    lib.impl("roll", bad, "CPU")
    try:
        def f(s, x):
            y = x * 1.0
            return torch.roll(y, 1, -1) + y
        return _ctl_fn(f)
    finally:
        lib._destroy()


@C("B4", "NEG", {"UNKNOWN"}, "Opus N-R3-1: a roll kernel that REVERSES its read-only input in place (same multiset of values, right output) is caught")
def _b4():
    return _roll_mutating(lambda t: t.copy_(t.flip(-1)))


@C("B5", "NEG", {"UNKNOWN"}, "Opus N-R3-1: a roll kernel that randomly PERMUTES its read-only input's elements in place is caught")
def _b5():
    g = torch.Generator().manual_seed(3)
    return _roll_mutating(lambda t: t.view(-1).copy_(t.view(-1)[torch.randperm(t.numel(), generator=g)].clone()))



# =====================================================================================================================
# F10 round 5 (re-check of round 4 by Opus crossexam/r4 + Fable crossexam_fable/r4; rulings F10-15, F10-16). Each control
# FAILS on the round-4 worktree (impl_r5/r4) unless marked as a mutant pin (it then refuses a named round-4 mutant).
# =====================================================================================================================
def _where_doubling(fill):
    """where.ScalarOther with its kernel replaced to DOUBLE the kept scores (Opus p40 case 1)."""
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("where.ScalarOther", lambda c, s_, o: torch.ops.aten.where.self(c, s_ * 2.0, torch.full_like(s_, o)), "CPU")
    try:
        return _ctl_fn(lambda s, x: torch.where(x > 0, x, fill))
    finally:
        lib._destroy()


def _merge(*analyses):
    out = {"opaque_ops": {}, "opacity_unknown_ops": {}, "decomposition_disagrees": {}, "decomposed_ops": {}}
    for a in analyses:
        for k in out:
            out[k].update(a.get(k) or {})
    return out


@C("T17", "NEG", {"UNKNOWN"}, "F10-15 (Opus p40 case 1): a where.ScalarOther kernel doubling the kept scores, with a -1e9 fill and with a finfo.min/8 fill, is caught")
def _t17():
    a1, a2 = _where_doubling(-1e9), _where_doubling(_FMIN / 8)
    return _merge(a1, a2) if t4(a1) == "UNKNOWN" and t4(a2) == "UNKNOWN" else CHECK(False, {"-1e9": t4(a1), "finfo.min/8": t4(a2)})


@C("T18", "NEG", {"UNKNOWN"}, "F10-15 (Fable R4-N1) / F10-21/22: a replaced triu kernel writing 0.5 where 0 belongs on a -1e9 mask is caught: the fill is a repeated position that never sets S, so the 0.5 written into the decomposition's zeros meets the slack of S = 0.5 and fails")
def _t18():
    lib = torch.library.Library("aten", "IMPL")
    def bad(x, diagonal=0):
        r_ = torch.arange(x.shape[-2]).unsqueeze(-1); c_ = torch.arange(x.shape[-1]).unsqueeze(0)
        return torch.where(c_ - r_ >= diagonal, x, torch.full_like(x, 0.5))
    lib.impl("triu", bad, "CPU")
    try:
        return _ctl_fn(lambda s, x: torch.triu(torch.full((16, 16), -1e9), 1) + x[0, :, :1].sum())
    finally:
        lib._destroy()


@C("T19", "VALID", {"PASS"}, "F10-15 (Opus p40 case 2): arithmetic producing sentinel / fill-range values still agrees: baddbmm with a mask (beta 1, 0.75), logsumexp(scores+mask), sums / stacks of scaled masks, tril, sdpa float mask, nearest upsample of a mask, -1e9 variants, bf16")
def _t19():
    F_ = torch.nn.functional
    mn = _FMIN
    tri = torch.ones(6, 6, dtype=torch.bool).triu(1)
    outs = []
    for fill in (mn, -1e9):
        m = torch.zeros(1, 1, 6, 6).masked_fill(tri, fill)
        cases = [lambda x, m=m: torch.baddbmm(m[0].expand(2, 6, 6), x.reshape(2, 6, 2), x.reshape(2, 6, 2).transpose(1, 2)),
                 lambda x, m=m: torch.baddbmm(m[0].expand(2, 6, 6), x.reshape(2, 6, 2), x.reshape(2, 6, 2).transpose(1, 2), beta=0.75),
                 lambda x, m=m: torch.logsumexp(m[0, 0] + x.reshape(6, 4)[:, :1], -1),
                 lambda x, m=m: (m[0, 0] * 0.4).sum(-1),
                 lambda x, m=m: torch.stack([m[0, 0], m[0, 0] * 0.6]),
                 lambda x, m=m: torch.tril(m[0, 0]),
                 lambda x, m=m: F_.scaled_dot_product_attention(x.reshape(1, 2, 6, 2), x.reshape(1, 2, 6, 2), x.reshape(1, 2, 6, 2), attn_mask=m),
                 lambda x, m=m: F_.interpolate(m, scale_factor=2)]
        for f in cases:
            outs.append(record(Fn(lambda s, x_, f=f: f(x_)), {"x": torch.randn(4, 6)})[1])
    xb = torch.randn(4, 6).bfloat16()
    outs.append(record(Fn(lambda s, x_: torch.where(x_ > 0, x_, torch.finfo(torch.bfloat16).min)), {"x": xb})[1])
    return _merge(*outs)


@C("T20", "VALID", {"PASS"}, "F10-15/17 wide-range honest outputs agree: mv with a 1e5-scaled row (the decomposition differs by ulps there) and peaky softmaxes (logit std 10 / 20). (Since F10-17 excluded entries are held to the relative law, so this no longer refuses a smaller ratio by behaviour; T8 pins 1e6)")
def _t20():
    torch.manual_seed(0)
    x = torch.randn(64, 4096); x[3] *= 1e5
    v = torch.randn(4096)
    a1 = record(Fn(lambda s, x_: torch.mv(x_, v)), {"x": x}, skip=False)[1]
    a2 = record(Fn(lambda s, x_: torch.ops.aten._safe_softmax(x_ * 10, -1) + torch.ops.aten._safe_softmax(x_ * 20, -1)),
                {"x": torch.randn(4, 4096)}, math=False)[1]
    return _merge(a1, a2)


@C("T21", "NEG", {"UNKNOWN"}, "F10-15 threshold from above: a -1e7 fill (1e7 x the O(1) kept scores) is excluded and exact: the doubling kernel is caught (a 1e8 ratio would hide it)")
def _t21():
    return _where_doubling(-1e7)


@C("T22", "NEG", {"UNKNOWN"}, "Fable R4-S2 / R4M1 / F10-21/22: a tril kernel writing finfo.min where 0 belongs on ordinary data is caught (the ratio exclusion is decided on ref: the finfo.min entry never sets S and fails the law with the scores' slack; kept, it would set S itself and fail: T22B pins the ref side at the threshold)")
def _t22():
    lib = torch.library.Library("aten", "IMPL")
    def bad(x, diagonal=0):
        r_ = torch.arange(x.shape[-2]).unsqueeze(-1); c_ = torch.arange(x.shape[-1]).unsqueeze(0)
        return torch.where(c_ - r_ <= diagonal, x, torch.full_like(x, _FMIN))
    lib.impl("tril", bad, "CPU")
    try:
        return _ctl_fn(lambda s, x: torch.tril(x[0] @ x[0].transpose(-1, -2)))
    finally:
        lib._destroy()


@C("T23", "NEG", {"UNKNOWN"}, "Fable R4-S2 by behaviour (F10-21/22 update): kept values ~1e33 beside a 93% 0.6 x finfo.min fill (NOT 1e6 x typical): the fill is a repeated position and never sets S (the F10-13 sentinel threshold that used to keep it out is deleted): a kernel scaling the kept values by 1.01 is caught")
def _t23():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("where.ScalarOther", lambda c, s_, o: torch.ops.aten.where.self(c, s_ * 1.01, torch.full_like(s_, o)), "CPU")
    try:
        return _ctl_fn(lambda s, x: torch.where(x > 1.5, x * 1e33, _FMIN * 0.6))     # (~93% fill, kept ~1e33)
    finally:
        lib._destroy()


@C("T24", "CHECK", {"TRUE"}, "F10-16: chunked comparison gives the chunk-free verdicts (chunk lowered to 7): the scale is global across chunks, exclusions and non-finite checks per chunk")
def _t24():
    import dag as _d
    if not hasattr(_d, "COMPARE_CHUNK"):
        return CHECK(False, "no chunked comparison")
    torch.manual_seed(0)
    ref = torch.randn(64) * 0.01; ref[60] = 3.0                         # the scale (3.0) sits in the last chunk
    near = ref.clone(); near[2] += 8 * torch.finfo(torch.float32).eps * 2.5   # within 8 eps x 3.0 only via the GLOBAL scale
    far = ref.clone(); far[2] += 1e-3
    m = ref.clone(); m[10] = _FMIN; m2 = m.clone(); m2[11] += 1e-3
    nn_ = ref.clone(); nn_[5] = float("nan"); nn2 = nn_.clone(); nn2[5] = 0.0
    cases = {"near (global scale)": (near, ref, True), "far": (far, ref, False), "sentinel ok": (m.clone(), m, True),
             "tamper beside sentinel": (m2, m, False), "nan kept": (nn_.clone(), nn_, True), "nan lost": (nn2, nn_, False)}
    saved = _d.COMPARE_CHUNK; _d.COMPARE_CHUNK = 7
    try:
        got = {k: _d._same_values(a, r) for k, (a, r, _) in cases.items()}
    finally:
        _d.COMPARE_CHUNK = saved
    return CHECK(all(got[k] == want for k, (_, _, want) in cases.items()), got)


@C("B6", "CHECK", {"TRUE"}, "Fable R4-S1: the single-pass fingerprint (contiguous) equals the marginal fingerprint of the same values (same strength), chunk boundaries included")
def _b6():
    import dag as _d
    torch.manual_seed(0)
    t = torch.randn(3, 5, 7, 11)
    saved = getattr(_d, "COMPARE_CHUNK", None)
    try:
        if saved is not None:
            _d.COMPARE_CHUNK = 13
        single = _d._fingerprint(t)
        marginal = _d._fingerprint(t.transpose(0, 1).contiguous().transpose(0, 1))   # (same values, strided: marginals)
    finally:
        if saved is not None:
            _d.COMPARE_CHUNK = saved
    return CHECK(single == marginal and single is not None and t.is_contiguous(), {"single": single, "marginal": marginal})



@C("R8", "CHECK", {"TRUE"}, "Fable R4M3: a seeded op whose OUTPUTS exceed the output cap (lowered) is a named unknown before the two-seed probe runs (the probe would run the too-large call twice more)")
def _r8():
    import dag as _d
    if not hasattr(_d, "OUTPUT_CAP"):
        return CHECK(False, "no output cap")
    saved, probes = _d.OUTPUT_CAP, []
    orig = DagRecorder._random_call
    def counting(self, *a, **k):
        probes.append(1); return orig(self, *a, **k)
    DagRecorder._random_call = counting; _d.OUTPUT_CAP = 64
    try:
        a = record(Fn(lambda s, x: torch.bernoulli(torch.full_like(x, 0.5))), {"x": X})[1]
    finally:
        DagRecorder._random_call = orig; _d.OUTPUT_CAP = saved
    why = (a.get("opacity_unknown_reasons") or {}).get("aten::bernoulli", "")
    return CHECK("output cap" in why and not probes, {"reason": why[:90], "probe_runs": len(probes)})


@C("S11", "VALID", {"PASS"}, "Fable R4M5: an output the decomposition reproduces with another DTYPE is 'not reproduced' (unrep): unknown only if the program reads it; here it is unread (var_mean's mean, decomposition replaced to return it in float64)")
def _s11():
    import opclass
    table = opclass.decompositions(); key = torch.ops.aten.var_mean.correction; saved = table[key]
    def dec(x, dim=None, *, correction=None, keepdim=False):
        v, m = saved(x, dim, correction=correction, keepdim=keepdim)
        return v, m.double()
    table[key] = dec
    try:
        return _ctl_fn(lambda s, x: torch.var_mean(x, -1)[0])
    finally:
        table[key] = saved


@C("T25", "NEG", {"UNKNOWN"}, "F10-15 / F10-21: 45% -1e9 fill, 40% exact zeros, 15% kept (by position), a kernel doubling the kept values is caught: the op's value never enters the typical magnitude, and the decomposition's zeros never count in it (a typical of 0 would disable the ratio)")
def _t25():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("where.ScalarOther", lambda c, s_, o: torch.ops.aten.where.self(c, s_ * 2.0, torch.full_like(s_, o)), "CPU")
    try:
        def f(s, x):
            # by position (no reliance on random draws): 45% -1e9 fill (the mode), 40% exact zeros, 15% kept (nonzero): the
            # zeros are the majority of the NON-mode entries, so a median over them would be 0 and disable the rule
            i = torch.arange(x.numel()).reshape(x.shape)
            r = torch.where(i >= int(0.85 * x.numel()), x.abs() + 0.5, torch.zeros_like(x))
            return torch.where(i >= int(0.45 * x.numel()), r, -1e9)
        return _ctl_fn(f)
    finally:
        lib._destroy()



# =====================================================================================================================
# F10 round 6 (re-check of round 5 by Opus crossexam/R5_REPORT.md + Fable crossexam_fable/R5_REPORT.md; rulings F10-17
# relative law on excluded entries, F10-18 skip the most frequent value). Each control FAILS on the round-5 worktree
# (impl_r6/r5) unless marked as a mutant pin.
# =====================================================================================================================
def _rec_any(f, x=None, math=True):
    return record(Fn(lambda s, x_: f(x_)), {"x": torch.zeros(1) if x is None else x}, math=math, skip=False)[1]


@C("RL1", "VALID", {"PASS"}, "F10-17 (Opus SR5-1 p51/p52, Fable R5-S1): honest wide-scaled outputs agree: cumprod ~1.5 over 120 steps and cumprod(full(1.5),100), var_mean / std / batch_norm(train) with channels 1e3..1e7 apart, linalg_vector_norm and addmv with a 1e7 row")
def _rl1():
    F_ = torch.nn.functional
    def chans(spread, rows=64, n=512):
        t = torch.randn(rows, n); t[0] *= spread; return t
    torch.manual_seed(0)
    cases = [lambda x: torch.var_mean(chans(1e3), -1)[0], lambda x: torch.var_mean(chans(3e3), -1)[0],
             lambda x: chans(1e4).std(-1),
             lambda x: F_.batch_norm(chans(1e4, 64, 16).t().contiguous(), None, None, training=True),
             lambda x: chans(1e7).norm(dim=-1),
             lambda x: torch.cumprod(torch.full((3, 120), 1.5) + torch.rand(3, 120) * 0.01, -1),
             lambda x: torch.addmv(torch.randn(64), chans(1e7, 64, 128), torch.randn(128), beta=0.5, alpha=1.7),
             lambda x: torch.cumprod(torch.full((4, 100), 1.5), -1)]
    for big in (1e4, 1e5):                                   # (channel spreads 1e6 / 1e7, p52)
        cases.append(lambda x, big=big: torch.var_mean(torch.cat([torch.randn(1, 512) * big, torch.randn(63, 512) * 1e-2]), -1)[0])
        cases.append(lambda x, big=big: F_.batch_norm(torch.cat([torch.randn(16, 1) * big, torch.randn(16, 63) * 1e-2], 1), None, None, training=True))
    outs = []
    for f in cases:
        torch.manual_seed(0)
        outs.append(_rec_any(f))
    return _merge(*outs)


@C("RL2", "CHECK", {"TRUE"}, "F10-17/22 law width on excluded entries (64 eps x |ref| plus the slack of the NON-excluded scale, here the O(1) scores): a -1e9 fill moved by +1.6e4 (~2.1 x the bound) and Fable's tamper set (finfo.min zeros->12345, fill halved, -1e9 x0.5 / +1e4, -1e7 doubled, kept +0.5 beside -1e9) disagree; a -1e9 fill moved by +4e3 (inside the bound) agrees")
def _rl2():
    import dag as _d
    torch.manual_seed(0)
    S = 128; sc = torch.randn(S, S); m10 = torch.rand(S, S) < 0.1
    fill = lambda v: torch.where(m10, torch.tensor(v), sc)
    tam = {"finfo.min zeros->12345": (torch.triu(torch.full((S, S), _FMIN), 1), lambda r: r.masked_fill(r == 0, 12345.0)),
           "finfo.min fill halved": (torch.where(torch.rand(S, S) < 0.5, torch.tensor(_FMIN), sc), lambda r: torch.where(r == _FMIN, r / 2, r)),
           "-1e9 fill x0.5": (fill(-1e9), lambda r: torch.where(r == -1e9, r * 0.5, r)),
           "-1e9 fill +1e4": (fill(-1e9), lambda r: torch.where(r == -1e9, r + 1e4, r)),
           "-1e9 fill +1.6e4 (2.1x the bound)": (fill(-1e9), lambda r: torch.where(r == -1e9, r + 1.6e4, r)),
           "-1e7 fill doubled": (fill(-1e7), lambda r: torch.where(r == -1e7, r * 2, r)),
           "kept +0.5 beside -1e9": (fill(-1e9), lambda r: torch.where(r == -1e9, r, r + 0.5))}
    got = {k: _d._same_values(t(r), r) for k, (r, t) in tam.items()}
    r = fill(-1e9)
    got["-1e9 fill +4e3 (inside the bound)"] = _d._same_values(torch.where(r == -1e9, r + 4e3, r), r)
    want = {k: False for k in tam}; want["-1e9 fill +4e3 (inside the bound)"] = True
    return CHECK(got == want, got)


def _windows_doubling(w, fill):
    L = 64
    idx = torch.arange(L)
    win = (idx[None, :] <= idx[:, None]) & (idx[:, None] - idx[None, :] < w)
    torch.manual_seed(0)
    s_ = torch.randn(1, 4, L, L)
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("where.ScalarOther", lambda c, a, o: torch.ops.aten.where.self(c, a * 2.0, torch.full_like(a, o)), "CPU")
    try:
        return record(Fn(lambda s, x_: torch.where(win, x_, fill)), {"x": s_})[1]
    finally:
        lib._destroy()


@C("RL3", "NEG", {"UNKNOWN"}, "F10-18/F10-21 (Opus SR5-2): a where.ScalarOther kernel doubling the kept scores on [1,4,64,64] sliding windows 16 (~80% filled) and 4 (~95% filled), with -1e9 and finfo.min/8 fills, is caught: the fill is a repeated position (never in S) and typical is the scores' (the plain median is the fill)")
def _rl3():
    res = {(w, f): _windows_doubling(w, v) for w in (16, 4) for f, v in (("-1e9", -1e9), ("finfo.min/8", _FMIN / 8))}
    if all(t4(a) == "UNKNOWN" for a in res.values()):
        return _merge(*res.values())
    return CHECK(False, {str(k): t4(a) for k, a in res.items()})


@C("RL4", "NEG", {"UNKNOWN"}, "F10-18/F10-21 (Fable R5-N1): where(90% / 50% masked, scores, -1e9) with a kernel adding 0.5 to the kept scores, and triu(full(-1e9)) with its zeros written as 50, are caught (the fill is a repeated position: never in S)")
def _rl4():
    torch.manual_seed(0)
    sc = torch.randn(64, 64)
    res = {}
    for frac in (0.9, 0.5):
        msk = torch.rand(64, 64) >= frac
        lib = torch.library.Library("aten", "IMPL")
        lib.impl("where.ScalarOther", lambda c, a, o: torch.ops.aten.where.self(c, a + 0.5, torch.full_like(a, o)), "CPU")
        try:
            res[f"where {int(frac*100)}% masked, kept +0.5"] = record(Fn(lambda s, x_, msk=msk: torch.where(msk, x_, -1e9)), {"x": sc})[1]
        finally:
            lib._destroy()
    lib = torch.library.Library("aten", "IMPL")
    def bad(x, diagonal=0):
        r_ = torch.arange(x.shape[-2]).unsqueeze(-1); c_ = torch.arange(x.shape[-1]).unsqueeze(0)
        return torch.where(c_ - r_ >= diagonal, x, torch.full_like(x, 50.0))
    lib.impl("triu", bad, "CPU")
    try:
        res["triu(full(-1e9)) zeros -> 50"] = _ctl_fn(lambda s, x: torch.triu(torch.full((16, 16), -1e9), 1) + x[0, :, :1].sum())
    finally:
        lib._destroy()
    if all(t4(a) == "UNKNOWN" for a in res.values()):
        return _merge(*res.values())
    return CHECK(False, {k: t4(a) for k, a in res.items()})


@C("RL5", "VALID", {"PASS"}, "F10-21/22 causes no false unknown on honest outputs with repeated values that are NOT fills (their positions never set the slack scale): hardtanh saturating ~80%, clamp-heavy var_mean, batch_norm(train) with 80% identical features, normalize / var_mean / vector_norm of 78-86%-identical rows (inexact decompositions), mostly-ones cumprod, relu-heavy mv / addmv, one-hot-ish softmax, honest sdpa math with 95%-masked bool and -1e9 sliding windows, logsumexp of 90%-identical rows")
def _rl5():
    F_ = torch.nn.functional
    torch.manual_seed(0)
    x = torch.randn(64, 512)
    idx = torch.arange(64)
    win = (idx[None, :] <= idx[:, None]) & (idx[:, None] - idx[None, :] < 4)
    q = torch.randn(1, 4, 64, 16)
    cases = [(lambda x: F_.hardtanh(x * 3 + 2.5), True), (lambda x: torch.var_mean(torch.clamp(x * 3 + 2.5, max=1.0), -1)[0], True),
             (lambda x: F_.batch_norm(torch.where(torch.rand(64, 512) < 0.8, torch.tensor(2.0), x), None, None, training=True), True),
             (lambda x: F_.normalize(torch.where(torch.rand(64, 1) < 0.8, torch.ones(64, 512), x), dim=-1) * 3, True),
             (lambda x: torch.cumprod(torch.where(torch.rand(4, 120) < 0.9, torch.tensor(1.0), torch.full((4, 120), 1.5)), -1), True),
             (lambda x: torch.mv(torch.relu(x - 1.5), torch.randn(512)), True),
             (lambda x: torch.addmv(torch.randn(64), torch.relu(x - 1.5), torch.randn(512)), True),
             (lambda x: torch.ops.aten._safe_softmax(x * 30, -1), False),
             (lambda x: torch.var_mean(torch.where(torch.rand(64, 1) < 0.8, torch.linspace(0, 1, 512).expand(64, 512), x), -1)[0], True),
             (lambda x: F_.scaled_dot_product_attention(q, q, q, attn_mask=win), True),
             (lambda x: F_.scaled_dot_product_attention(q, q, q, attn_mask=torch.where(win, 0.0, -1e9)), True),
             (lambda x: torch.logsumexp(torch.where(torch.rand(64, 1) < 0.9, torch.ones(64, 512), x), -1), True),
             (lambda x: torch.linalg.vector_norm(torch.where(torch.rand(64, 1) < 0.8, torch.full((64, 512), 0.3), x), dim=-1), True)]
    outs = []
    for f, math in cases:
        torch.manual_seed(1)
        outs.append(_rec_any(f, x, math))
    return _merge(*outs)



# =====================================================================================================================
# F10 round 7 (Opus crossexam/R6b_REPORT.md J1-J6; rulings F10-19 "posb" typical magnitude, F10-20 half precision).
# The r6-tree result of each control is in impl_r7/IMPL_REPORT.md.
# =====================================================================================================================
_EPS32 = torch.finfo(torch.float32).eps


def _win(w, L=64):
    i = torch.arange(L)
    return (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < w)


def _where_kernel(kernel, fn, x):
    """Record fn(x) with where.ScalarOther's KERNEL replaced by kernel(c, s, o) (the op's own value = ref)."""
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("where.ScalarOther", kernel, "CPU")
    try:
        return record(Fn(lambda s, x_: fn(x_)), {"x": x})[1]
    finally:
        lib._destroy()


def _where_decomposition(dec, kernel, fn, x):
    """Record fn(x) with where.ScalarOther's DECOMPOSITION replaced by dec (the decomposition's value = a) and,
    when kernel is given, its kernel replaced too."""
    import opclass
    table = opclass.decompositions(); key = torch.ops.aten.where.ScalarOther; saved = table[key]
    table[key] = dec
    try:
        return _where_kernel(kernel, fn, x) if kernel is not None else record(Fn(lambda s, x_: fn(x_)), {"x": x})[1]
    finally:
        table[key] = saved


def _all_unknown(res):
    if all(t4(a) == "UNKNOWN" for a in res.values()):
        return _merge(*res.values())
    return CHECK(False, {k: t4(a) for k, a in res.items()})


@C("J1", "NEG", {"UNKNOWN"}, "F10-19/F10-21 (Opus R6b J1): sliding windows 16 / 4 with a -1e9 fill that the kernel JITTERS by up to 60 eps relative (inside the F10-17 law) and doubles the kept scores: caught (the statistic and the repeated positions come from the decomposition, whose fill is exact)")
def _j1():
    torch.manual_seed(0)
    sc = torch.randn(1, 4, 64, 64)
    def kernel(c, s_, o):
        g = torch.Generator().manual_seed(1)
        jit = torch.full_like(s_, o) * (1 + _EPS32 * torch.randint(-60, 61, s_.shape, generator=g).to(s_.dtype))
        return torch.ops.aten.where.self(c, s_ * 2.0, jit)
    return _all_unknown({f"window{w}": _where_kernel(kernel, lambda x, w=w: torch.where(_win(w), x, -1e9), sc) for w in (16, 4)})


@C("J2", "NEG", {"UNKNOWN"}, "F10-19/F10-21 (Opus R6b J2): the DECOMPOSITION's fill scattered over 127 values within 63 eps relative (each < 1% of the entries; ref's fill exact), window 16, kernel doubling the kept scores: caught, because 128-eps buckets merge the scattered fill into one repeated bucket (refuses P2, repeats counted on exact values)")
def _j2():
    torch.manual_seed(0)
    sc = torch.randn(1, 4, 64, 64)
    k = torch.randint(-63, 64, sc.shape, generator=torch.Generator().manual_seed(2)).float()
    def dec(c, s_, o):
        return torch.where(c, s_, torch.full_like(s_, o) * (1 + _EPS32 * k))
    kernel = lambda c, s_, o: torch.ops.aten.where.self(c, s_ * 2.0, torch.full_like(s_, o))
    return _where_decomposition(dec, kernel, lambda x: torch.where(_win(16), x, -1e9), sc)


@C("J3", "NEG", {"UNKNOWN"}, "F10-19/F10-21/22 (Opus R6b J3): scores on a 9-level grid (k x 0.25) beside a window-16 / causal -1e9 fill, kernel negating the kept scores: caught. Every nonzero value of the decomposition is repeated: none sets S, so the slack is that of the zeros (0) and the negation fails the relative part")
def _j3():
    torch.manual_seed(0)
    g9 = torch.randint(-4, 5, (1, 4, 64, 64)).float() * 0.25
    kernel = lambda c, s_, o: torch.ops.aten.where.self(c, -s_, torch.full_like(s_, o))
    return _all_unknown({lab: _where_kernel(kernel, lambda x, w=w: torch.where(_win(w), x, -1e9), g9) for lab, w in (("window16", 16), ("causal", 64))})


@C("J4", "NEG", {"UNKNOWN"}, "F10-19/F10-21 (Opus R6b J4): two fills of 30% each, by position (a -1e9 mask fill and a -1e8 padding the input carries) beside 40% scores, kernel doubling the scores only: caught (both fills are repeated positions; a 50% repeat threshold is refused by T8 / J6 / SR7-3)")
def _j4():
    torch.manual_seed(0)
    n = 4 * 64 * 64
    i = torch.arange(n).reshape(1, 4, 64, 64) % 10
    x = torch.where((i >= 3) & (i < 6), torch.tensor(-1e8), torch.randn(1, 4, 64, 64))
    kernel = lambda c, s_, o: torch.ops.aten.where.self(c, torch.where(s_.abs() < 1e6, s_ * 2.0, s_), torch.full_like(s_, o))
    return _where_kernel(kernel, lambda x_: torch.where(i >= 3, x_, -1e9), x)


@C("J5", "VALID", {"PASS"}, "F10-19/F10-21 (Opus R6b J5, p62): a clamp-like floor of 1e-12 on 60% / 90% of the entries beside scores ~0.05, the decomposition off by an honest absolute error of half the slack (0.5 x 8 eps x S) on the scores: agrees. The floor is a repeated value of the decomposition, so typical comes from the scores (refuses any min with a plain median, which is the floor)")
def _j5():
    out = []
    for frac in (0.6, 0.9):
        n = 4000; k = int(n * (1 - frac))
        torch.manual_seed(0)
        y = torch.randn(n) * 0.05
        keep = torch.arange(n) < k
        S = float(y[:k].abs().max())
        noise = 0.5 * 8 * _EPS32 * S * (torch.rand(n) * 2 - 1) * keep
        import opclass
        saved = opclass.decompositions()[torch.ops.aten.where.ScalarOther]
        # (an exact CPU kernel, so the call reaches the recorder as where.ScalarOther rather than its implicit where.self)
        exact = lambda c, s_, o: torch.ops.aten.where.self(c, s_, torch.full_like(s_, o))
        out.append(_where_decomposition(lambda c, s_, o, noise=noise: saved(c, s_ + noise, o), exact,
                                        lambda x, keep=keep: torch.where(keep, x, 1e-12), y))
    if not all((a.get("decomposed_ops") or {}).get("aten::where.ScalarOther") for a in out):   # (the law really ran)
        return CHECK(False, [a.get("decomposed_ops") for a in out])
    return _merge(*out)


@C("J6", "NEG", {"UNKNOWN"}, "F10-19/F10-21 (Opus R6b J6 = Fable R6-S1): a window-16 -1e9 fill (47%) plus a -1e8 key padding in every row (37.5%), kernel tampering the kept scores (x2 / +0.5 / := 0.25): caught. Both fills are repeated positions (never in S)")
def _j6():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 64, 64); pad = torch.arange(64) >= 40; x[..., pad] = -1e8
    keep = _win(16) | pad[None, :]          # (the padding holds -1e8 in every row: 37.5% padding, 47% -1e9 fill)
    res = {}
    for lab, t in (("x2", lambda s_: s_ * 2.0), ("+0.5", lambda s_: s_ + 0.5), (":= 0.25", lambda s_: torch.full_like(s_, 0.25))):
        kernel = lambda c, s_, o, t=t: torch.ops.aten.where.self(c, torch.where(s_.abs() < 1e6, t(s_), s_), torch.full_like(s_, o))
        res[lab] = _where_kernel(kernel, lambda x_: torch.where(keep, x_, -1e9), x)
    return _all_unknown(res)


@C("J7", "NEG", {"UNKNOWN"}, "F10-21/22: every nonzero value of the decomposition repeated (a constant score block of -1500, as a zero-weight recording produces, beside a causal -1e9 fill), kernel moving the block by +900: caught. Neither the block nor the fill sets S; with the fill in S (r6) the slack 954 would hide it")
def _j7():
    kernel = lambda c, s_, o: torch.ops.aten.where.self(c, s_ + 900.0, torch.full_like(s_, o))
    return _where_kernel(kernel, lambda x: torch.where(_win(64), x, -1e9), torch.full((1, 4, 64, 64), -1500.0))


@C("J8", "NEG", {"UNKNOWN"}, "F10-19/F10-21: six fills of 12% each (-1e9 mask fill, -2e9..-6e9 carried by the input), the decomposition scattering each over 127 values within 63 eps relative, kernel doubling the 28% scores: caught (J9 pins the 128-eps width by behaviour)")
def _j8():
    torch.manual_seed(0)
    i = torch.arange(4 * 64 * 64).reshape(1, 4, 64, 64) % 25
    x = torch.randn(1, 4, 64, 64)
    for k in range(2, 7):
        x = torch.where((i >= 3 * (k - 1)) & (i < 3 * k), torch.tensor(-1e9 * k), x)
    j = torch.randint(-63, 64, x.shape, generator=torch.Generator().manual_seed(3)).float()
    def dec(c, s_, o):
        big = s_.abs() >= 1e6
        return torch.where(c, torch.where(big, s_ * (1 + _EPS32 * j), s_), torch.full_like(s_, o) * (1 + _EPS32 * j))
    kernel = lambda c, s_, o: torch.ops.aten.where.self(c, torch.where(s_.abs() < 1e6, s_ * 2.0, s_), torch.full_like(s_, o))
    return _where_decomposition(dec, kernel, lambda x_: torch.where(i >= 3, x_, -1e9), x)


_HALF_HONEST = {"logsumexp": lambda x: torch.logsumexp(x, -1),
                "batch_norm(train)": lambda x: torch.nn.functional.batch_norm(x, None, None, training=True)}
_HALF_EXACT = {"var_mean": lambda x: torch.var_mean(x, -1)[0], "std": lambda x: x.std(-1), "mv": lambda x: torch.mv(x, x[0]),
               "linalg_vector_norm": lambda x: torch.linalg.vector_norm(x, dim=-1), "triu": lambda x: torch.triu(x),
               "cumprod": lambda x: torch.cumprod(1 + x * 0.01, -1)}


def _half_runs(cases, dtypes):
    out = {}
    for dt in dtypes:
        for k, f in cases.items():
            torch.manual_seed(0)
            out[(str(dt).split(".")[-1], k)] = _rec_any(f, torch.randn(16, 64).to(dt))
    return out


@C("HP1", "CHECK", {"TRUE"}, "F10-20: honest bf16 and fp16 decompositions that are not bit-equal (logsumexp, batch_norm(train): a few ulps apart) are each a NAMED G4 unknown whose reason names half precision, the op recorded whole (no decomposition kept: node for node the recording made without decomposition)")
def _hp1():
    got = {}
    for dt in (torch.bfloat16, torch.float16):
        for k, f in _HALF_HONEST.items():
            recs = {}
            for dec in (True, False):
                torch.manual_seed(0)
                recs[dec] = record(Fn(lambda s, x_, f=f: f(x_)), {"x": torch.randn(16, 64).to(dt)}, skip=False, decompose=dec)
            rec, a = recs[True]
            why = " ".join((a.get("opacity_unknown_reasons") or {}).values())
            got[f"{str(dt).split('.')[-1]} {k}"] = {"t4": t4(a), "half reason": "half-precision" in why,
                                                   "decomposed": bool(a.get("decomposed_ops")),
                                                   "nodes": len(rec.nodes), "nodes undecomposed": len(recs[False][0].nodes)}
    # (the decomposition's own nodes are discarded: the recording is node for node the undecomposed one)
    return CHECK(all(v["t4"] == "UNKNOWN" and v["half reason"] and not v["decomposed"] and v["nodes"] == v["nodes undecomposed"]
                     for v in got.values()), got)


@C("HP2", "VALID", {"PASS"}, "F10-20: bit-equal half-precision decompositions still pass (var_mean, std, mv, vector_norm, triu, cumprod in bf16 and fp16, and var_mean / std of an input holding a NaN: the same NaN bits are bit-equal; torch.complex of float16 halves, a complex32 output, decided by its bits; each recorded as its decomposition)")
def _hp2():
    res = _half_runs(_HALF_EXACT, (torch.bfloat16, torch.float16))
    for dt in (torch.bfloat16, torch.float16):
        x = torch.randn(16, 64).to(dt); x[3, 5] = float("nan")
        res[(str(dt), "var_mean nan")] = _rec_any(lambda x_: torch.var_mean(x_, -1)[0], x)
        res[(str(dt), "std nan")] = _rec_any(lambda x_: x_.std(-1), x)
    # torch.complex of two float16 halves: a complex32 output, bit-equal (r6: the recorder raised, equal_cpu is not
    # implemented for ComplexHalf)
    res[("float16", "complex")] = _rec_any(lambda x_: torch.complex(x_, x_ * 0.5), torch.randn(16, 64).half())
    if not all(a.get("decomposed_ops") for a in res.values()):
        return CHECK(False, {f"{d} {k}": a.get("decomposed_ops") for (d, k), a in res.items()})
    return _merge(*res.values())


class _BNHalf(torch.nn.Module):
    def __init__(self, dt):
        super().__init__()
        self.register_buffer("rm", torch.full((8,), 0.25, dtype=dt)); self.register_buffer("rv", torch.ones(8, dtype=dt))

    def forward(self, x):
        return torch.nn.functional.batch_norm(x, self.rm, self.rv, training=True)


@C("HP4", "CHECK", {"TRUE"}, "F10-20 on WRITTEN operands: batch_norm(train) with half-precision running statistics, its decomposition replaced to leave the running mean 1 ulp off (outputs bit-equal: a constant input) is the named half-precision unknown in bf16 and fp16; the honest decomposition passes")
def _hp4():
    import opclass
    table = opclass.decompositions(); key = torch.ops.aten.native_batch_norm.default; saved = table[key]
    def bumped(*a, **k):
        r = saved(*a, **k)
        a[3].mul_(1.0078125)                     # (one bf16 ulp; a few fp16 ulps)
        return r
    got = {}
    for dt in (torch.bfloat16, torch.float16):
        for lab, dec in (("honest", saved), ("running mean off", bumped)):
            table[key] = dec
            try:
                a = record(_BNHalf(dt), {"x": torch.full((16, 8), 0.5, dtype=dt)}, skip=False)[1]
            finally:
                table[key] = saved
            why = " ".join((a.get("opacity_unknown_reasons") or {}).values())
            got[f"{str(dt).split('.')[-1]} {lab}"] = (t4(a), "half-precision" in why)
    want = {k: (("UNKNOWN", True) if k.endswith("off") else ("PASS", False)) for k in got}
    return CHECK(got == want, got)


@C("HP3", "VALID", {"PASS"}, "F10-20 leaves float32 unaffected: the HP1 ops (logsumexp, batch_norm(train)) and the HP2 ops in float32 agree by the F10-9 law and are recorded as their decompositions")
def _hp3():
    res = _half_runs({**_HALF_HONEST, **_HALF_EXACT}, (torch.float32,))
    if not all(a.get("decomposed_ops") for a in res.values()):
        return CHECK(False, {f"{d} {k}": a.get("decomposed_ops") for (d, k), a in res.items()})
    return _merge(*res.values())



# =====================================================================================================================
# F10 round 8 (rulings F10-21: typical read from the decomposition only, repeated positions held to the relative law
# and never in the slack scale, a quartile filter on the non-repeated set; the sentinel rule deleted). Constructions from
# Opus crossexam/R7_REPORT.md (SR7-1..3, p91-p93) and Fable crossexam_fable/R7b_REPORT.md (x1, x3, x4). The r7-tree
# result of each control is in impl_r8/IMPL_REPORT.md.
# =====================================================================================================================
def _where_tamper(kernel_value, fn, x):
    """Record fn(x) (a torch.where(mask, x, fill) call) with where.ScalarOther's kernel returning
    where(c, kernel_value(s), o): the decomposition gives the honest value, the op's own value is tampered."""
    return _where_kernel(lambda c, s_, o: torch.ops.aten.where.self(c, kernel_value(s_), torch.full_like(s_, o)), fn, x)


def _small_fill(frac, n=20000):
    """A 1-D mask keeping all but the first frac x n positions (the fill positions)."""
    keep = torch.ones(n, dtype=torch.bool); keep[:int(frac * n)] = False
    return keep


@C("SR7-1", "NEG", {"UNKNOWN"}, "F10-21 (Opus SR7-1): a SMALL fill (0.3 % / 0.9 %, -1e9 and finfo.min/8, too small to repeat) beside N(0,1) scores, kernel writing kept := sign(kept) (a repeated constant in ref only): caught. The statistic reads the decomposition (refuses positions read from ref, whose repeated +-1 would leave the fill alone to set typical), and the fill is a ratio outlier (refuses the ratio removal)")
def _sr71():
    res = {}
    for frac in (0.003, 0.009):
        for lab, fill in (("-1e9", -1e9), ("finfo.min/8", _FMIN / 8)):
            torch.manual_seed(0)
            keep = _small_fill(frac)
            res[f"{frac*100:.1f}% {lab}"] = _where_tamper(torch.sign, lambda x, keep=keep, fill=fill: torch.where(keep, x, fill), torch.randn(20000))
    return _all_unknown(res)


def _grid(n=20000, seed=0):
    torch.manual_seed(seed)
    return torch.randint(-4, 5, (n,)).float() * 0.25


@C("SR7-2", "NEG", {"UNKNOWN"}, "F10-21/22 (Opus SR7-2): honest 9-level grid scores beside a < 1 % -1e9 fill (0.2 % / 0.8 %), kernel kept := -kept / x3 / := 0, and grid + 0.8 % fill + 0.6 % continuous with the grid negated: caught. The grid levels are repeated positions that never set S, and the grid has two or more significant levels, so the level cap makes the non-repeated fill an outlier too (refuses removing the level cap: the fill would set S = 1e9, slack ~950)")
def _sr72():
    res = {}
    for frac in (0.002, 0.008):
        keep = _small_fill(frac)
        for lab, t in (("-kept", lambda s_: -s_), ("x3", lambda s_: s_ * 3), (":= 0", lambda s_: torch.zeros_like(s_))):
            res[f"{frac*100:.1f}% {lab}"] = _where_tamper(t, lambda x, keep=keep: torch.where(keep, x, -1e9), _grid())
    g = _grid(); torch.manual_seed(1); g[160:280] = torch.randn(120)          # 0.6 % continuous after a 0.8 % fill
    on_grid = lambda s_: (s_ * 4 == torch.round(s_ * 4))
    keep = _small_fill(0.008)
    res["0.8% + 0.6% continuous: grid := -grid"] = _where_tamper(lambda s_: torch.where(on_grid(s_), -s_, s_), lambda x: torch.where(keep, x, -1e9), g)
    return _all_unknown(res)


@C("SR7-3", "NEG", {"UNKNOWN"}, "F10-21 (Opus SR7-3): a repeated value never sets the slack scale S: a causal -1e5 fill beside N(0,1) scores (1e5 < 1e6 x typical), kernel x1.01 / +0.05; a constant -1500 block (zero-weight scores) + causal -1e9, kernel +300; the block + ONE padded key (1.6 %), +300: caught (with the fill in S the slack is 8 eps x F, up to 0.95 x typical)")
def _sr73():
    torch.manual_seed(0)
    res = {}
    sc = torch.randn(1, 4, 64, 64)
    for lab, t in (("x1.01", lambda s_: s_ * 1.01), ("+0.05", lambda s_: s_ + 0.05)):
        res[f"-1e5 causal {lab}"] = _where_tamper(t, lambda x: torch.where(_win(64), x, -1e5), sc)
    blk = torch.full((1, 4, 64, 64), -1500.0)
    res["-1500 block + causal -1e9: +300"] = _where_tamper(lambda s_: s_ + 300.0, lambda x: torch.where(_win(64), x, -1e9), blk)
    pad = torch.arange(64) < 63
    res["-1500 block + 1 padded key: +300"] = _where_tamper(lambda s_: s_ + 300.0, lambda x: torch.where(pad, x, -1e9), blk)
    return _all_unknown(res)


@C("Z12", "NEG", {"UNKNOWN"}, "F10-21 zero family (impl_r7 open item 1 / Opus SR7-4): -1e9 and finfo.min/8 fills on 90 % and 50 % of the entries, the kept scores holding 30 % honest zeros, kernel kept := 0 and kept := -0.0: caught. The op's value never enters the statistic (in r7 ref was zero wherever the decomposition was not repeated, so the fill set the scale)")
def _z12():
    res = {}
    for frac in (0.9, 0.5):
        for lab, fill in (("-1e9", -1e9), ("finfo.min/8", _FMIN / 8)):
            torch.manual_seed(0)
            x = torch.randn(64, 64); x[torch.rand(64, 64) < 0.3] = 0.0
            keep = torch.rand(64, 64) >= frac
            for zl, t in ((":= 0", torch.zeros_like), (":= -0.0", lambda s_: -torch.zeros_like(s_))):
                res[f"{int(frac*100)}% {lab} {zl}"] = _where_tamper(t, lambda x_, keep=keep, fill=fill: torch.where(keep, x_, fill), x)
    return _all_unknown(res)


@C("GB1", "NEG", {"UNKNOWN"}, "F10-21 quartile filter (Fable R7b x3 guard band): grid scores (repeated) + 0.3 % continuous scores + a 0.8 % -1e9 fill (the non-repeated set is 1.1 %, dominated by the fill), kernel doubling the continuous scores: caught; and its neighbours (continuous / fill 0.2 / 1.2, 0.5 / 0.4, 0.9 / 0, 1.1 / 0, 0 / 0.9, 2.0 / 0.5 %; the continuous doubled, the grid doubled where there is none). The fill is above 1e6 x the non-repeated set's 25th percentile and does not set typical (refuses the filter's removal and an anchor at the minimum is refused by SR6F)")
def _gb1():
    res = {}
    on_grid = lambda s_: (s_ * 4 == torch.round(s_ * 4))
    for cont, fill in ((0.003, 0.008), (0.002, 0.012), (0.005, 0.004), (0.009, 0.0), (0.011, 0.0), (0.0, 0.009), (0.02, 0.005)):
        n = 20000; g = _grid(n); nf, nc = int(fill * n), int(cont * n)
        torch.manual_seed(2); g[nf:nf + nc] = torch.randn(nc) + 0.1           # (continuous: never on the 0.25 grid)
        keep = torch.ones(n, dtype=torch.bool); keep[:nf] = False
        t = (lambda s_: torch.where(on_grid(s_), s_, s_ * 2)) if nc else (lambda s_: s_ * 2)
        res[f"cont {cont*100:.1f}% fill {fill*100:.1f}%"] = _where_tamper(t, lambda x, keep=keep: torch.where(keep, x, -1e9), g)
    return _all_unknown(res)


@C("SG1", "CHECK", {"TRUE"}, "F10-21 one sign convention: a -1e9 fill (90 %) and a -1500 constant block are repeated positions (R) at every position, by the same magnitude convention the statistic uses; +1e9 and -1e9 share a bucket. A mask keyed on the signed value while the statistic keys |a| (Fable's first B: repS silently inert) fails this")
def _sg1():
    import dag as _d
    torch.manual_seed(0)
    a = torch.where(torch.rand(4096) < 0.9, torch.tensor(-1e9), torch.randn(4096))
    b = torch.where(torch.rand(4096) < 0.5, torch.tensor(-1500.0), torch.tensor(-1e9))
    got = {}
    for lab, t in (("-1e9 fill", a), ("-1500 block + -1e9", b)):
        keys = _d._robust_typical(t)[1]
        r = _d._repeated(t.to(torch.float64), keys, torch.finfo(t.dtype).eps)
        big = t.abs() >= 1000
        got[lab] = bool(r[big].all()) and not bool(r[~big].any())
    keys = _d._robust_typical(a)[1]
    got["+1e9 in the -1e9 bucket"] = bool(_d._repeated(torch.tensor([1e9, -1e9], dtype=torch.float64), keys, torch.finfo(torch.float32).eps).all())
    return CHECK(all(got.values()), got)


@C("SR6F", "CHECK", {"TRUE"}, "F10-21/22, no tiny anchor and no one-level cap (Fable SR6-1, R8 e): a 1e-12 floor on 40 / 60 / 90 / 98.5 / 99.2 / 99.9 % of the entries beside values in [1e-3, 0.1] and [1e-2, 0.1], the decomposition off by an honest 0.5 x slack absolute error on the values: agrees (a filter anchored at the minimum, or any min with the floor, excludes every value and fails them)")
def _sr6f():
    import dag as _d
    torch.manual_seed(0)
    eps = torch.finfo(torch.float32).eps
    got = {}
    for frac in (0.4, 0.6, 0.9, 0.985, 0.992, 0.999):
        for lo in (1e-3, 1e-2):
            sz = 128 if frac < 0.95 else 256
            vals = torch.rand(sz, sz) * (0.1 - lo) + lo
            ref = torch.where(torch.rand(sz, sz) < frac, torch.tensor(1e-12), vals)
            a = torch.where(ref == 1e-12, ref, ref + (torch.rand(sz, sz) * 2 - 1) * 0.5 * 8 * eps * 0.1)
            got[f"{int(frac*100)}% [{lo:g},0.1]"] = _d._same_values(a, ref)
    return CHECK(all(got.values()), got)


def _honest_ops(ops, x=None):
    outs, bad = [], {}
    for k, f in ops.items():
        torch.manual_seed(0)
        a = _rec_any(lambda x_, f=f: f(), x)
        outs.append(a)
        if t4(a) != "PASS":
            bad[k] = t4(a)
    return CHECK(False, bad) if bad else _merge(*outs)


@C("HV1", "VALID", {"PASS"}, "F10-21 honest decomposed ops with repeated values (Fable R7b x1, 21 ops): relu-heavy mv, quantized hardswish, one-hot _safe_softmax, cumprod of 1s and 2s, floor_divide(arange, 3), stacked ids + scores, hardtanh 80 %, batch_norm(train) 80 % identical, var_mean / normalize / logsumexp of identical rows, cumprod(1.5, 100), sdpa math with a 95 % -1e9 mask, triu scores + -1e9, a 1e-6 fill, a clamp floor + glu, saturated mish, grid + 1 % tiny residues, constant 3.0 via triu, baddbmm with a -1e9 bias, a -1500 block beside -1e9: all agree")
def _hv1():
    F_ = torch.nn.functional
    H = {
     "relu-heavy mv": lambda: torch.mv(F_.relu(torch.randn(256, 64)), torch.randn(64)),
     "quantized hardswish": lambda: F_.hardswish(torch.round(torch.randn(64, 256) * 4) / 4),
     "one-hot _safe_softmax x40": lambda: torch.ops.aten._safe_softmax(torch.randn(16, 1024) * 40, -1),
     "cumprod 1s with 2s": lambda: torch.cumprod(torch.where(torch.arange(4096.).reshape(4, 1024) % 97 == 0, torch.tensor(2.0), torch.tensor(1.0)), -1),
     "floor_divide(arange,3)": lambda: torch.floor_divide(torch.arange(4096.).reshape(4, 1024), 3.0),
     "stack ids+scores": lambda: torch.stack([torch.arange(1024.).repeat(4).reshape(4, 1024), torch.randn(4, 1024)], 0),
     "hardtanh 80%": lambda: F_.hardtanh(torch.randn(64, 256) * 5),
     "batch_norm train 80% identical": lambda: F_.batch_norm(torch.cat([torch.full((8, 32, 4, 4), 3.0), torch.randn(2, 32, 4, 4)]), None, None, training=True),
     "var_mean identical rows": lambda: torch.var_mean(torch.where(torch.rand(64, 256) < 0.8, torch.tensor(2.0), torch.randn(64, 256)), -1)[0],
     "normalize identical rows": lambda: F_.normalize(torch.where(torch.rand(64, 256) < 0.8, torch.tensor(2.0), torch.randn(64, 256)), dim=-1),
     "cumprod(1.5,100)": lambda: torch.cumprod(torch.full((4, 100), 1.5), -1),
     "sdpa-math 95% -1e9": lambda: F_.scaled_dot_product_attention(torch.randn(1, 4, 128, 32), torch.randn(1, 4, 128, 32), torch.randn(1, 4, 128, 32), attn_mask=torch.where(torch.rand(128, 128) < 0.95, torch.tensor(-1e9), torch.tensor(0.0))),
     "logsumexp identical rows": lambda: torch.logsumexp(torch.where(torch.rand(64, 1024) < 0.9, torch.tensor(1.0), torch.randn(64, 1024) * 3), -1),
     "triu scores + -1e9": lambda: torch.triu(torch.randn(128, 128) + torch.triu(torch.full((128, 128), -1e9), 1), 0),
     "where small fill 1e-6": lambda: torch.where(torch.rand(128, 128) < 0.5, torch.tensor(1e-6), torch.randn(128, 128)) + torch.triu(torch.zeros(128, 128)),
     "clamp-floor 60% + glu": lambda: F_.glu(torch.cat([torch.clamp(torch.where(torch.rand(64, 128) < 0.6, torch.tensor(-1.0), torch.rand(64, 128) * 0.1), min=1e-12), torch.zeros(64, 128)], -1)),
     "mish saturated": lambda: F_.mish(torch.randn(64, 256) * 20),
     "grid 99% + 1% tiny residues": lambda: torch.where(torch.rand(128, 128) < 0.99, torch.round(torch.randn(128, 128) * 4) / 4, torch.randn(128, 128) * 1e-9) + torch.triu(torch.zeros(128, 128)),
     "constant 3.0 via triu": lambda: torch.triu(torch.full((64, 64), 3.0), 0),
     "baddbmm -1e9 bias q*30": lambda: torch.baddbmm(torch.triu(torch.full((1, 128, 128), -1e9), 1), torch.randn(1, 128, 8) * 30, torch.randn(1, 8, 128)),
     "-1500 block + -1e9": lambda: torch.where(torch.rand(64, 64) < 0.5, torch.tensor(-1500.0), torch.tensor(-1e9)) + torch.triu(torch.zeros(64, 64)),
    }
    return _honest_ops(H)


@C("HV2", "VALID", {"PASS"}, "F10-21/22 honest repeated values made by arithmetic, which never set S but receive its slack (Opus R7 p93 [repS]): var_mean / std of 78 % constant rows, constant rows + a per-row offset, var_mean / vector_norm / logsumexp / mse_loss of 78 % rows of 0.1, var_mean / std of rows 1e4 +- 1e-3 (cancellation), cumprod of 1.1, baddbmm of constant operands: all agree")
def _hv2():
    F_ = torch.nn.functional
    def x1():
        x = torch.randn(256, 64); x[:200] = 3.7; return x
    def x3():
        x = torch.full((256, 64), 0.1); x[200:] = torch.randn(56, 64); return x
    def x4():
        x = torch.full((256, 64), 1e4); x[:, ::2] += 1e-3; x[200:] = torch.randn(56, 64) * 1e4; return x
    H = {
     "var_mean 78% constant rows": lambda: torch.var_mean(x1(), -1)[0], "std 78% constant rows": lambda: x1().std(-1),
     "var_mean constant rows + offset": lambda: torch.var_mean(x1() + 1e-3 * torch.randn(256, 1), -1)[0],
     "var_mean rows of 0.1": lambda: torch.var_mean(x3(), -1)[0], "vector_norm rows of 0.1": lambda: torch.linalg.vector_norm(x3(), dim=-1),
     "logsumexp rows of 0.1": lambda: torch.logsumexp(x3(), -1), "mse_loss none rows of 0.1": lambda: F_.mse_loss(x3(), torch.zeros(256, 64), reduction="none"),
     "var_mean 1e4 +- 1e-3": lambda: torch.var_mean(x4(), -1)[0], "std 1e4 +- 1e-3": lambda: x4().std(-1),
     "cumprod 1.1": lambda: torch.cumprod(torch.full((64, 30), 1.1), -1),
     "baddbmm constant": lambda: torch.baddbmm(torch.full((8, 16, 16), 0.3), torch.full((8, 16, 4), 0.7), torch.full((8, 4, 16), 0.2), beta=0.5, alpha=1.3),
    }
    return _honest_ops(H)


@C("RC1", "CHECK", {"TRUE"}, "F10-21 chunk invariance: repeated positions are decided per chunk with the SAMPLE's global bucket keys, so the verdicts with the chunk lowered to 7 equal the default ones (SR7-3 -1e5 fill x1.01 and the -1500 block +300 disagree, their honest forms agree, a grid + 0.8 % fill := -kept disagrees). Keys counted per chunk would find no repeat in 7 entries and let the fill set S")
def _rc1():
    import dag as _d
    torch.manual_seed(0)
    m = _win(64).expand(1, 4, 64, 64); sc = torch.randn(1, 4, 64, 64)
    h = torch.where(m, sc, torch.tensor(-1e5)); blk = torch.where(m, torch.tensor(-1500.0), torch.tensor(-1e9))
    g = _grid(); g[:160] = -1e9
    cases = {"-1e5 x1.01": (h, torch.where(m, h * 1.01, h), False), "-1e5 honest": (h, h + 0.0 * h, True),
             "block +300": (blk, torch.where(m, blk + 300, blk), False), "grid := -kept": (g, torch.where(g == -1e9, g, -g), False)}
    out = {}
    for chunk in (None, 7):
        saved = _d.COMPARE_CHUNK
        if chunk:
            _d.COMPARE_CHUNK = chunk
        try:
            out[chunk or "default"] = {k: _d._same_values(a, r) for k, (a, r, _) in cases.items()}
        finally:
            _d.COMPARE_CHUNK = saved
    want = {k: w for k, (_, _, w) in cases.items()}
    return CHECK(out["default"] == want and out[7] == want, out)



@C("GB2", "NEG", {"UNKNOWN"}, "F10-21 the typical set never holds zeros: grid scores WITHOUT a zero level + 0.5 % exact zeros + 0.3 % continuous scores + a 0.8 % -1e9 fill, kernel doubling the continuous scores: caught. Zeros counted in the non-repeated set would be its 25th percentile, typical 0, no ratio exclusion, and the fill would set S")
def _gb2():
    n = 20000; g = _grid(n); g[g == 0] = 0.125
    torch.manual_seed(3); nf, nz, nc = 160, 100, 60
    g[nf:nf + nz] = 0.0; g[nf + nz:nf + nz + nc] = torch.randn(nc) + 0.1
    keep = torch.ones(n, dtype=torch.bool); keep[:nf] = False
    on_grid = lambda s_: (s_ * 8 == torch.round(s_ * 8))
    return _where_tamper(lambda s_: torch.where(on_grid(s_), s_, s_ * 2), lambda x: torch.where(keep, x, -1e9), g)


@C("SR6G", "CHECK", {"TRUE"}, "F10-21 the quartile, not the minimum: N(0,1) scores with a sliver of distinct tiny values (12 entries 1e-12..1e-11, not repeated), the decomposition off by an honest 0.5 x slack absolute error on the scores: agrees (an anchor at the minimum makes typical ~1e-12, excludes every score and fails them on the relative law alone)")
def _sr6g():
    import dag as _d
    torch.manual_seed(0)
    eps = torch.finfo(torch.float32).eps
    ref = torch.randn(20000); ref[:12] = torch.linspace(1e-12, 1e-11, 12)
    S = float(ref.abs().max())
    a = ref.clone(); a[12:] += (torch.rand(19988) * 2 - 1) * 0.5 * 8 * eps * S
    return CHECK(_d._same_values(a, ref), {"typical": _d._robust_typical(a)[0]})


@C("FB1", "CHECK", {"TRUE"}, "F10-21 fallback = the smallest repeated level: 3 x 2^20 entries (sampled with stride 3), every SAMPLED nonzero value repeated (-1e9 fill 75 %, grid levels 0.25..1), one lone -1e8 at an unsampled position, the kernel writing 50 into the decomposition's zeros: caught (typical 0.25 excludes the lone -1e8; a fallback at the plain median, 1e9, would let it set S = 1e8 and the slack 95 hide the 50). The honest value agrees")
def _fb1():
    import dag as _d
    torch.manual_seed(0)
    n = 3 << 20
    a = torch.where(torch.rand(n) < 0.75, torch.tensor(-1e9), torch.randint(-4, 5, (n,)).float() * 0.25)
    a[1] = -1e8                                  # (stride 3 samples 0, 3, 6, ...: index 1 is never sampled)
    ref = torch.where(a == 0, torch.tensor(50.0), a)
    return CHECK(not _d._same_values(a, ref) and _d._same_values(a, a + 0.0 * a) and _d._robust_typical(a)[0] == 0.25,
                 {"typical": _d._robust_typical(a)[0]})


@C("T22B", "CHECK", {"TRUE"}, "F10-21 the ratio exclusion is decided on ref (T22 at the threshold): N(0,1) scores and three honest values that the decomposition puts 30 eps BELOW 1e6 x typical and the kernel 30 eps ABOVE it (inside the relative law), the kernel adding 0.5 to every score: caught (excluded on ref, S stays the scores' scale). Deciding on the decomposition's value keeps them, S ~ 1e6 x typical, the slack 0.95 x typical hides the +0.5")
def _t22b():
    import dag as _d
    torch.manual_seed(0)
    eps = torch.finfo(torch.float32).eps
    a = torch.randn(20000); a[:3] = 1e9                    # (placeholder: above 1e6 x the quartile, so not in typical)
    out = 1e6 * _d._robust_typical(a)[0]
    a[:3] = out * (1 - 30 * eps)
    ref = a.clone(); ref[:3] = out * (1 + 30 * eps); ref[3:] += 0.5
    o2 = 1e6 * _d._robust_typical(a)[0]
    return CHECK(float(a[0]) <= o2 < float(ref[0]) and not _d._same_values(a, ref),
                 {"a": float(a[0]), "out": o2, "ref": float(ref[0])})



@C("J9", "CHECK", {"TRUE"}, "F10-21 bucket width 128 eps by behaviour: grid scores beside a 10 % -1e9 fill that the decomposition scatters over 127 values within 63 eps (the kernel's fill exact), the kernel writing 50 into the decomposition's zeros: caught. 128-eps buckets keep the scattered fill one repeated set that never sets S; 8-eps buckets split it into ~16 buckets of < 1 %, the fill becomes the whole non-repeated set, sets typical and S = 1e9, and the slack 954 hides the 50. The honest value agrees")
def _j9():
    import dag as _d
    eps = torch.finfo(torch.float32).eps
    g = _grid(); nf = 2000
    j = torch.randint(-63, 64, (nf,), generator=torch.Generator().manual_seed(4)).float()
    a = g.clone(); a[:nf] = -1e9 * (1 + eps * j)
    ref = g.clone(); ref[:nf] = -1e9
    tam = torch.where(a == 0, torch.tensor(50.0), ref)
    return CHECK(_d._same_values(a, ref) and not _d._same_values(a, tam), {"honest": _d._same_values(a, ref), "zeros := 50": _d._same_values(a, tam)})



# =====================================================================================================================
# F10 round-8 closure (ruling F10-22 after Fable's final check, crossexam_fable/R8_REPORT.md B1): excluded entries never
# set S but still receive the slack; the level cap when the decomposition has two or more significant repeated levels.
# =====================================================================================================================
def _dominant(shape, frac, sigma, seed=0):
    """1.0 on frac of the entries, 1.0 + N(1, sigma) noise elsewhere (Fable z3: a dominant identical level per channel)."""
    torch.manual_seed(seed)
    x = 1.0 + sigma * torch.randn(shape)
    return torch.where(torch.rand(shape) < frac, torch.tensor(1.0), x)


@C("BN1", "VALID", {"PASS"}, "F10-22 (Fable R8 B1): honest batch_norm(train) [10,32,8,8] with 80 % of each channel = 1.0 and the rest N(1, sigma) (sigma 1.0 / 0.1 / 0.03; 90 % with 0.3) and instance_norm [2,4,32,32] with 80 % / 95 % = 1.0 + N(1, 0.1) agree, each recorded as its decomposition: the normalised dominant level is a repeated value made by cancellation whose honest error is absolute (eps x invstd), so a repeated position must receive the slack; and the synthetic level 1e-3 / 1e-4 (60 %) with a 5e-8 absolute error agrees (law). Refuses 'repeated positions get the relative law alone' (the r8 law)")
def _b1():
    import dag as _d
    F_ = torch.nn.functional
    cases = {f"bn 80% sigma {s_}": (lambda s_=s_: F_.batch_norm(_dominant((10, 32, 8, 8), 0.8, s_), None, None, training=True)) for s_ in (1.0, 0.1, 0.03)}
    cases["bn 90% sigma 0.3"] = lambda: F_.batch_norm(_dominant((10, 32, 8, 8), 0.9, 0.3), None, None, training=True)
    for fr in (0.8, 0.95):
        cases[f"instance_norm {int(fr*100)}% sigma 0.1"] = lambda fr=fr: F_.instance_norm(_dominant((2, 4, 32, 32), fr, 0.1))
    outs, bad = [], {}
    for k, f in cases.items():
        a = _rec_any(lambda x_, f=f: f())
        outs.append(a)
        if t4(a) != "PASS" or not (a.get("decomposed_ops") or {}).get("aten::native_batch_norm"):
            bad[k] = (t4(a), a.get("decomposed_ops"))
    torch.manual_seed(0)
    for lvl in (1e-3, 1e-4):
        vals = torch.rand(128, 128) * 0.1 + 1e-3
        ref = torch.where(torch.rand(128, 128) < 0.6, torch.tensor(lvl), vals)
        a_ = ref + (torch.rand(128, 128) * 2 - 1) * 5e-8
        if not _d._same_values(a_, ref):
            bad[f"synthetic level {lvl:g}"] = "disagrees"
    return CHECK(False, bad) if bad else _merge(*outs)


@C("LK1", "NEG", {"UNKNOWN"}, "F10-22 level cap (impl_r8 known gap K1, closed): 9-level grid scores WITH a zero level beside a 0.2 % / 0.8 % -1e9 fill, kernel writing 0.5 / 50 / 900 into the decomposition's zero positions: caught. The grid has two or more significant repeated levels, so the fill (above 1e6 x 0.25) is an outlier, never typical, never in S (refuses removing the level cap)")
def _k1():
    res = {}
    for frac in (0.002, 0.008):
        keep = _small_fill(frac)
        for c in (0.5, 50.0, 900.0):
            res[f"{frac*100:.1f}% zeros -> {c:g}"] = _where_tamper(lambda s_, c=c: torch.where(s_ == 0, torch.tensor(c), s_), lambda x, keep=keep: torch.where(keep, x, -1e9), _grid())
    return _all_unknown(res)


@C("LS1", "CHECK", {"TRUE"}, "F10-22 level significance (1e-6 x the largest repeated level): a 1e-12 clamp floor (60 %) plus a saturation level 0.1 (20 %) beside values in [1e-3, 0.1), the decomposition off by an honest 0.5 x slack absolute error: agrees (counting the 1e-12 floor as a level would cap at 1e-6 and take every value out of S); and grid levels 1e-3 and 1 with a zero level beside a 0.5 % -1e9 fill, the kernel writing 0.5 into the zeros: caught (a significance above 1e-3 would leave one level and no cap)")
def _ls1():
    import dag as _d
    torch.manual_seed(0)
    eps = torch.finfo(torch.float32).eps
    u = torch.rand(256, 256)
    ref = torch.where(u < 0.6, torch.tensor(1e-12), torch.where(u < 0.8, torch.tensor(0.1), torch.rand(256, 256) * 0.099 + 1e-3))
    a = torch.where(u < 0.8, ref, ref + (torch.rand(256, 256) * 2 - 1) * 0.5 * 8 * eps * 0.1)
    g = torch.tensor([0.0, 1e-3, -1e-3, 1.0, -1.0])[torch.randint(0, 5, (20000,))]; g[:100] = -1e9
    tam = torch.where(g == 0, torch.tensor(0.5), g)
    got = {"floor + 0.1 level honest": _d._same_values(a, ref), "levels 1e-3 / 1 + fill: zeros -> 0.5": _d._same_values(g, tam)}
    return CHECK(got["floor + 0.1 level honest"] and not got["levels 1e-3 / 1 + fill: zeros -> 0.5"], got)



@C("GB3", "NEG", {"UNKNOWN"}, "F10-21 quartile filter with ONE repeated level (no level cap): a single level 0.25 on 98.9 % of the entries, 0.3 % continuous scores and a 0.8 % -1e9 fill, kernel doubling the continuous scores: caught. The fill is above 1e6 x the non-repeated set's 25th percentile and does not set typical (refuses removing the filter, which the level cap covers only when there are two levels)")
def _gb3():
    n = 20000
    x = torch.full((n,), 0.25); nf, nc = 160, 60
    torch.manual_seed(5); x[nf:nf + nc] = torch.randn(nc) + 0.1
    keep = torch.ones(n, dtype=torch.bool); keep[:nf] = False
    return _where_tamper(lambda s_: torch.where(s_ == 0.25, s_, s_ * 2), lambda x_: torch.where(keep, x_, -1e9), x)


@C("LC1", "CHECK", {"TRUE"}, "F10-22 level cap, each part: (a) it bounds S: levels +-1e-3 / +-2e-3 (69.5 %), N(0,1) scores (30 %), a 0.5 % -1e4 fill (under 1e6 x typical, above 1e6 x 1e-3), kernel +5e-3 on the scores: caught; (b) it bounds typical: levels 1 / 2 (98.9 %), 0.2 % scores ~1e-3, a 0.9 % -1e8 fill and one honest 1e4, kernel +5e-3 on the scores: caught; (c) it is 1e6 x the SMALLEST significant level: levels +-1e-3 / +-1 with a zero level and a 0.5 % -1e5 fill, kernel writing 5e-3 into the zeros: caught. Every honest form agrees")
def _lc1():
    import dag as _d
    torch.manual_seed(6)
    n = 20000; got = {}
    a = torch.tensor([1e-3, -1e-3, 2e-3, -2e-3])[torch.randint(0, 4, (n,))]
    a[:6000] = torch.randn(6000); a[6000:6100] = -1e4
    t = a.clone(); t[:6000] += 5e-3
    got["(a) honest"], got["(a) scores +5e-3"] = _d._same_values(a, a + 0.0 * a), _d._same_values(a, t)
    b = torch.tensor([1.0, 2.0])[torch.randint(0, 2, (n,))]
    b[:40] = torch.rand(40) * 1e-3 + 1e-3; b[40:220] = -1e8; b[220] = 1e4
    t = b.clone(); t[:40] += 5e-3
    got["(b) honest"], got["(b) scores +5e-3"] = _d._same_values(b, b + 0.0 * b), _d._same_values(b, t)
    c = torch.tensor([0.0, 1e-3, -1e-3, 1.0, -1.0])[torch.randint(0, 5, (n,))]; c[:100] = -1e5
    t = torch.where(c == 0, torch.tensor(5e-3), c)
    got["(c) honest"], got["(c) zeros -> 5e-3"] = _d._same_values(c, c + 0.0 * c), _d._same_values(c, t)
    return CHECK(all(v for k, v in got.items() if k.endswith("honest")) and not any(v for k, v in got.items() if not k.endswith("honest")), got)



@C("F23", "CHECK", {"TRUE"}, "F10-23 (close-out): BatchNorm2d in training mode and InstanceNorm2d on non-centred low-variance channels (75 % of each channel = 100.0, the rest 100.0 + N(0, 0.1); Opus p105/p106) are a NAMED G4 unknown: recorded as the decomposition of aten::native_batch_norm, which disagrees with the kernel (two-pass variance vs Welford: 2e-4 absolute, a real difference), never a pass and never a crash. A tolerance widened for norm ops (x1e3) or a silent pass would turn it into false transparency")
def _f23():
    def chans(shape):
        torch.manual_seed(2)
        x = torch.full(shape, 100.0)
        x[..., :2, :] += torch.randn(*shape[:-2], 2, shape[-1]) * 0.1
        return x
    got = {}
    for lab, mod in (("BatchNorm2d train", torch.nn.BatchNorm2d(16).train()), ("InstanceNorm2d", torch.nn.InstanceNorm2d(16))):
        try:
            a = _rec_any(lambda x_, mod=mod: mod(x_), chans((8, 16, 8, 8)))
            got[lab] = {"t4": t4(a), "disagrees": sorted(a.get("decomposition_disagrees") or {}),
                        "decomposed": "aten::native_batch_norm" in (a.get("decomposed_ops") or {})}
        except Exception as e:
            got[lab] = {"raised": f"{type(e).__name__}: {str(e)[:80]}"}
    ok = all(v.get("t4") == "UNKNOWN" and "aten::native_batch_norm" in v.get("disagrees", []) and v.get("decomposed") for v in got.values())
    return CHECK(ok, got)



@C("SP1", "CHECK", {"TRUE"}, "F10 close-out single-pass check: a decomposition value that is NaN or +-inf where the op's value is finite disagrees (alone, beside a -1e9 fill, and in the second of two chunks with the chunk lowered to 7), and one entry over the bound by 1 % of the slack disagrees while one at 99 % of it agrees: the one-pass excess equals the per-entry law")
def _sp1():
    import dag as _d
    torch.manual_seed(0)
    ref = torch.randn(64); ref[:20] = -1e9
    got = {}
    for lab, v in (("nan", float("nan")), ("+inf", float("inf")), ("-inf", float("-inf"))):
        for pos in (3, 40):
            a = ref.clone(); a[pos] = v
            got[f"{lab} at {pos}"] = _d._same_values(a, ref)
    saved = _d.COMPARE_CHUNK; _d.COMPARE_CHUNK = 7
    try:
        a = ref.clone(); a[60] = float("nan"); got["nan in a later chunk (chunk 7)"] = _d._same_values(a, ref)
    finally:
        _d.COMPARE_CHUNK = saved
    r64 = ref.double(); eps = torch.finfo(torch.float64).eps          # (float64: the bound is exact to the last ulp)
    S = float(r64[20:].abs().max()); j = 30
    for frac in (1.01, 0.99):
        a = r64.clone(); a[j] += 64 * eps * abs(float(r64[j])) + frac * 8 * eps * S
        got[f"one entry at {frac:g} x the bound"] = _d._same_values(a, r64)
    want = {k: False for k in got}; want["one entry at 0.99 x the bound"] = True
    return CHECK(got == want, got)



def drawing_parity():
    """A decomposed recording, folded for drawing, equals the same call recorded without decomposition, node for
    node (func, module, operands, constants, shape, overload, arguments, outputs)."""
    import capture
    out = {}
    for name, mod in (("sdpa attention", Attn(sdpa=True)), ("stack/norm", StackNorm()), ("in-place", Fn(lambda s, x: Mutating()(x)))):
        ra, _ = record(mod, {"x": X}, decompose=True)
        rb, _ = record(mod, {"x": X}, decompose=False)
        g = capture.nodes_of(capture.graph(ra))
        fa, _m = capture.fold_groups(g)
        key = lambda n: (n.func, n.module, tuple(map(tuple, n.ins)), tuple(n.consts), n.shape, n.op, repr(n.args), repr(n.outs))
        same = [key(n) for n in fa] == [key(n) for n in rb.nodes]
        out[name] = {"decomposed_nodes": len(ra.nodes), "folded": len(fa), "undecomposed": len(rb.nodes), "same": same}
    return out


def drawing_namespace():
    """What the drawing's matchers read for a custom op named add: the packet ('add'), or the qualified op."""
    rec, _ = record(Fn(lambda s, x: torch.ops.f10ctl.add(x, x)), {"x": X})
    n = rec.nodes[-1]
    r = {"recorded func": n.func, "op": getattr(n, "op", None)}
    if F10:
        import capture
        r["drawing view func"] = capture.fold_groups(capture.nodes_of(capture.graph(rec)))[0][-1].func
    return r


if __name__ == "__main__":
    res, bad = {}, []
    for cid, kind, expect, desc, fn in CONTROLS:
        try:
            a = fn()
            if isinstance(a, tuple) and a and a[0] == "CHECK":
                got, detail = ("TRUE" if a[1] else "FALSE"), a[2]
            else:
                got = t4(a)
                detail = {k: a.get(k) for k in ("opaque_ops", "opacity_unknown_ops", "decomposition_disagrees", "decomposed_ops") if k in a}
        except Exception as e:
            got, detail = "ERROR", f"{type(e).__name__}: {str(e)[:200]}"
        ok = got in expect
        res[cid] = {"kind": kind, "desc": desc, "expect": sorted(expect), "got": got, "ok": ok, "detail": detail}
        print(f"{'ok ' if ok else 'BAD'} {cid:4} {kind:5} expect {'/'.join(sorted(expect)):12} got {got:8} {desc}", flush=True)
        if not ok:
            bad.append(cid)
            print("      ", json.dumps(detail, default=str)[:400])
    if F10:
        dp = drawing_parity()
        print("drawing parity (folded decomposed == undecomposed):", json.dumps(dp))
        if not all(v["same"] for v in dp.values()):
            bad.append("drawing_parity")
        res["drawing_parity"] = dp
    ns = drawing_namespace()
    print("custom op named add, as the drawing reads it:", ns)
    res["drawing_namespace"] = ns
    if F10 and ns.get("drawing view func") == "add":
        bad.append("drawing_namespace")
    if "--report" in sys.argv:
        print(json.dumps(res, indent=1, default=str))
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out:
        json.dump(res, open(out, "w"), indent=1, default=str)
    print("ALL AS EXPECTED" if not bad else f"NOT AS EXPECTED: {bad}")
    sys.exit(1 if bad else 0)
