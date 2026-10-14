"""Full dataflow recorder for one real forward pass (zero-storage weights + exact zero-skip).

Every tensor is mapped to the op that produced it; every op records its input edges
(op -> op, param, buffer, model input, or EXTERNAL = appeared from outside the recording),
the module it ran inside, and its scalar constants. Nothing is inferred from names or source text.
"""
import re, math, cmath, numbers, weakref, collections, contextlib, signal
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map
import lowcost
from lowcost import ZERO_PTRS, ZERO_REG
aten = torch.ops.aten

OPAQUE = re.compile(r"(scaled_dot_product|flash|_efficient_attention|cudnn|_fused|fused_|mkldnn_rnn|_cufft)")
CONTROL_SINKS = {"_local_scalar_dense", "is_nonzero", "equal", "item"}


# ---- F07: the zero-skip returns an op's EXACT result without arithmetic, one rule per op (dossier
#      model-benchmark/_reruns/p2a1_f07). Z(t) = known all-zero (registered zero storage, or by value); F(t) = no inf/nan.
COW_EVENTS = []          # forward-pass writes that gave a whole zero-storage weight its own storage (F07-1)
NONFINITE_EVENTS = []    # ops that turned finite inputs into NaN / inf (zero-weight artefact, F07-3)
ZSW_EVENTS = []          # forward-pass writes through a view of shared zero storage (refused; a G2 unknown)


class ZeroStorageWrite(RuntimeError):
    """A forward-pass write into shared zero storage through a view / alias that cannot be given its own storage."""


def is_reg(t):
    return isinstance(t, torch.Tensor) and t.numel() > 1 and t.untyped_storage().data_ptr() in ZERO_REG


def known_zero(t):
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return False
    if is_reg(t):
        return True
    return t.numel() > 1 and not bool(torch.any(t))


def finite(t):
    if t is None or not isinstance(t, torch.Tensor) or not (t.is_floating_point() or t.is_complex()) or is_reg(t):
        return True
    return bool(torch.isfinite(t).all())


def product_zero(a, b):
    """a contracted with b is exactly zero: one factor is all-zero and the other has no inf / nan (0 * inf = nan)."""
    return (known_zero(a) and finite(b)) or (known_zero(b) and finite(a))


def _meta(func, args, kwargs):
    """The real op's output metadata (shape, dtype, strides), or None when even shape inference rejects the call: the
    caller then executes it for real (so the user's error is raised, never replaced by zeros)."""
    m = tree_map(lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x, (args, kwargs))
    try:
        return func(*m[0], **m[1])
    except RuntimeError:
        if func is not aten._grouped_mm.default:  # only grouped expert matmul's meta kernel needs bf16
            return None
        orig = next((a.dtype for a in tree_flatten(args)[0] if isinstance(a, torch.Tensor) and a.is_floating_point()), torch.float32)
        mb = tree_map(lambda x: x.to("meta", torch.bfloat16) if isinstance(x, torch.Tensor) and x.is_floating_point() else
                      (x.to("meta") if isinstance(x, torch.Tensor) else x), (args, kwargs))
        try:
            o = func(*mb[0], **mb[1])
        except RuntimeError:
            return None
        return o.to(orig) if o.is_floating_point() else o


def _zeros_like_meta(mo):
    return torch.empty_strided(mo.shape, mo.stride(), dtype=mo.dtype).zero_()       # the real op's strides


def _filled_like_meta(mo, value):
    return torch.empty_strided(mo.shape, mo.stride(), dtype=mo.dtype).copy_(value.expand(mo.shape))


SKIP_OPS = {aten.mm.default, aten.bmm.default, aten._grouped_mm.default, aten.addmm.default, aten.baddbmm.default,
            aten.embedding.default, aten.convolution.default}


def _one_dtype(args, kwargs):
    """All floating / complex tensor operands share one dtype (a mixed-dtype call raises in real torch: never skip it)."""
    return len({a.dtype for a in tree_flatten((args, kwargs))[0]
                if isinstance(a, torch.Tensor) and (a.is_floating_point() or a.is_complex())}) <= 1


def exact_zero_result(func, args, kwargs):
    """The op's exact result when a zero operand determines it, else None (= execute normally)."""
    if not _one_dtype(args, kwargs):
        return None
    r = _exact_zero_result(func, args, kwargs)
    return r


def _exact_zero_result(func, args, kwargs):
    A = list(args) + [None] * 4
    if func in (aten.mm.default, aten.bmm.default):
        if not product_zero(A[0], A[1]):
            return None
        mo = _meta(func, args, kwargs)
        return None if mo is None else _zeros_like_meta(mo)
    if func is aten._grouped_mm.default:
        if kwargs.get("bias", A[3]) is not None or not product_zero(A[0], A[1]):
            return None
        mo = _meta(func, args, kwargs)
        return None if mo is None else _zeros_like_meta(mo)
    if func in (aten.addmm.default, aten.baddbmm.default):
        c, a, b = A[0], A[1], A[2]
        beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
        if not isinstance(alpha, numbers.Number) or not cmath.isfinite(alpha) or not isinstance(beta, numbers.Number):
            return None
        if not (c.is_floating_point() or c.is_complex()) and not float(beta).is_integer():
            return None                           # integer bias x fractional beta: torch's own rounding (execute)
        if not product_zero(a, b):
            return None                           # a zero bias alone never zeroes the product
        mo = _meta(func, args, kwargs)
        if mo is None:
            return None
        # beta == 0 ignores c (even nan); + 0.0 gives torch's sign of zero (-0.0 bias -> +0.0, as x*0 + c*beta does)
        return _zeros_like_meta(mo) if beta == 0 else _filled_like_meta(mo, c * beta + 0.0)
    if func is aten.embedding.default:
        w, idx = A[0], A[1]
        if not is_reg(w):
            return None                           # registered zero tables only (a -0.0 table by value keeps its sign)
        if idx.is_floating_point() or idx.is_complex():
            return None                           # the real op rejects non-integer indices
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= w.shape[0]):
            return None                           # the real op raises
        mo = _meta(func, args, kwargs)
        return None if mo is None else _zeros_like_meta(mo)
    if func is aten.convolution.default:
        x, w, b = A[0], A[1], A[2]
        if not product_zero(x, w):
            return None
        mo = _meta(func, args, kwargs)
        if mo is None:
            return None
        return _zeros_like_meta(mo) if b is None else _filled_like_meta(mo, b.reshape((1, -1) + (1,) * (mo.dim() - 2)) + 0.0)
    return None


def zero_storage_findings(models=()):
    """(G2 unknowns, notes) from this process's zero-storage execution (F07 rulings, Soumil 2026-10-13): non-finite
    values created from finite inputs; non-finite values held by the built models; build-time writes that could not be
    applied (over the size cap, or through an alias of storage no Parameter owns); writes refused through a view; zero
    storage changed outside the recorder. Copy-on-write and alias writes into Parameters (zero by design) are notes."""
    unk = []
    held = []
    pptrs = set()
    for m in models:
        for n_, p_ in m.named_parameters(remove_duplicate=False):
            pptrs.add(p_.untyped_storage().data_ptr())
        for n_, t_ in list(m.named_parameters(remove_duplicate=False)) + list(m.named_buffers(remove_duplicate=False)):
            # NaN is never a legitimate stored value; +-inf is (e.g. Gemma-4 clippable-linear clamp-bound buffers the
            # library's own init sets), so only NaN held by the build is a zero-weight artefact (learned params are reset)
            if (t_.is_floating_point() or t_.is_complex()) and t_.numel() and not is_reg(t_) and bool(torch.isnan(t_).any()):
                held.append(n_)
    if held:
        unk.append({"predicate": "G2", "what": "the zero-storage build holds NaN values a real model does not",
                    "items": held[:20], "count": len(held)})
    alias_unowned = [d for d, p in lowcost.BUILD_ALIAS_WRITES if p not in pptrs] if models else []
    if alias_unowned:
        unk.append({"predicate": "G2", "what": "build-time writes through an alias of zero storage no Parameter owns (a "
                    "computed buffer left as zeros)", "items": alias_unowned[:20], "count": len(alias_unowned)})
    if ZSW_EVENTS:
        unk.append({"predicate": "G2", "what": "forward-pass writes through a view of shared zero storage were refused "
                    "(the pass could not run faithfully)", "items": ZSW_EVENTS[:20], "count": len(ZSW_EVENTS)})
    if NONFINITE_EVENTS:
        unk.append({"predicate": "G2", "what": "the zero-weight execution produced non-finite values from finite inputs "
                    "(a real model never computes these): values downstream differ from the checkpoint's",
                    "items": sorted(set(NONFINITE_EVENTS))[:20], "count": len(NONFINITE_EVENTS)})
    if lowcost.BUILD_UNAPPLIED:
        unk.append({"predicate": "G2", "what": "build-time writes into zero storage could not be applied (computed values "
                    "left as zeros)", "items": lowcost.BUILD_UNAPPLIED[:20], "count": len(lowcost.BUILD_UNAPPLIED)})
    if not lowcost.registry_intact():
        unk.append({"predicate": "G2", "what": "shared zero storage was changed outside the recorder: weights sharing it "
                    "were not zero", "items": [], "count": 0})
    notes = {"copy_on_write": COW_EVENTS[:20], "build_alias_writes_left_zero": [d for d, _ in lowcost.BUILD_ALIAS_WRITES][:20],
             "learned_params_reset_to_zero": sorted(set(lowcost.BUILD_PARAMS_RESET))[:20]}
    return unk, {k: v for k, v in notes.items() if v}


ALLOC_OPS = {"empty", "empty_like", "new_empty", "new_empty_strided", "empty_strided", "empty_permuted"}
METADATA_INPLACE = {"unsqueeze_", "squeeze_", "t_", "transpose_", "as_strided_", "detach_", "swapdims_", "swapaxes_",
                    "requires_grad_"}                # change a view's metadata, never its values


def _write_targets(func, args, kwargs):
    if func.overloadpacket.__name__ in METADATA_INPLACE:
        return []
    out = []
    for i, sa in enumerate(func._schema.arguments):
        if sa.alias_info is not None and sa.alias_info.is_write:
            tgt = args[i] if i < len(args) else kwargs.get(sa.name)
            out += [t for t in (tgt if isinstance(tgt, (list, tuple)) else [tgt]) if isinstance(t, torch.Tensor)]
    return out


GRAPH_SCHEMA = 2   # F06: graph schema version. 1 = legacy (packet name, flattened untyped constants, no ports / returns)


class Node:
    """One recorded op. Legacy fields (schema 1, kept unchanged for the drawing adapter): func = overload packet name,
    ins = tensor operands in flatten order, consts = flattened int/float constants (Booleans / None / strings / dtypes
    dropped), n_out = number of tensor outputs, shape = shape of output 0.
    Schema-2 fields (F06; None = not recorded = UNKNOWN, never a default):
      op   : qualified overload, func.name() (namespace::name[.overload]);
      args : the call bound to the op's own schema: ((arg name, typed value), ...) in schema order, with an argument
             omitted exactly when its value equals the schema's declared default (type and value); a tensor is
             {"t": k}, k = its position in `ins`; None when the call could not be bound;
      outs : ((shape, dtype), ...) per output port (the position among the op's tensor outputs);
    and every ("op", i) edge in `ins` carries the producer's output port: ("op", i, port).
      cargs: the CANONICAL form of args (Soumil 2026-10-13, F06-1 "equivalent by meaning"), or None when it equals
             args: a form folded only when running the op both ways on the same random operands gives the same
             result (canonical_args). `args` stays the exact evidence; comparisons (node_key) use the canonical form."""
    __slots__ = ("func", "module", "ins", "consts", "n_out", "shape", "op", "args", "outs", "cargs")

    def __init__(self, func, module, ins, consts, n_out, shape, op=None, args=None, outs=None, cargs=None):
        self.func, self.module, self.ins, self.consts, self.n_out, self.shape = func, module, ins, consts, n_out, shape
        self.op, self.args, self.outs, self.cargs = op, args, outs, cargs


# ---- F06 typed argument encoding (JSON-native, so the in-memory form IS the serialised form) ----------------------
_DT = {}


def _dtype_name(d):
    n = _DT.get(d)
    if n is None:
        n = _DT[d] = str(d).replace("torch.", "")
    return n


class Unencodable(TypeError):
    pass


def enc(v, tref):
    """A call argument as a JSON-native typed value. Python's own types survive JSON (true / 1 / 1.0 stay distinct);
    everything else is tagged. tref(tensor) -> its {"t": k} reference."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else {"float": repr(v)}
    if isinstance(v, complex):
        return {"complex": [repr(v.real), repr(v.imag)]}
    if isinstance(v, torch.Tensor):
        return tref(v)
    if isinstance(v, (list, tuple)):
        return [enc(x, tref) for x in v]
    if isinstance(v, torch.dtype):
        return {"dtype": _dtype_name(v)}
    if isinstance(v, torch.device):
        return {"device": str(v)}
    if isinstance(v, torch.layout):
        return {"layout": str(v).replace("torch.", "")}
    if isinstance(v, torch.memory_format):
        return {"memory_format": str(v).replace("torch.", "")}
    if isinstance(v, torch.Generator):
        return {"generator": str(v.device)}           # which generator stream: values, not structure (seed not kept)
    if isinstance(v, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        raise Unencodable("symbolic value")
    raise Unencodable(type(v).__name__)


def canon(v):
    """A hashable, TYPE-EXACT key of an encoded value (True != 1 != 1.0; -0.0 != 0.0; nan == nan)."""
    if v is None:
        return ("n",)
    if isinstance(v, bool):
        return ("b", v)
    if isinstance(v, int):
        return ("i", v)
    if isinstance(v, float):
        return ("f", repr(v))
    if isinstance(v, str):
        return ("s", v)
    if isinstance(v, list):
        return ("l",) + tuple(canon(x) for x in v)
    if isinstance(v, dict):
        return ("d",) + tuple((k, canon(x)) for k, x in sorted(v.items()))
    raise Unencodable(type(v).__name__)


_SCHEMA = {}


def _schema_of(func):
    """Per overload: qualified name and [(arg name, kwarg_only, canonical declared default or a no-default marker)]."""
    sc = _SCHEMA.get(func)
    if sc is None:
        spec = []
        for a in func._schema.arguments:
            d = _NODEFAULT
            if a.has_default_value():
                try:
                    d = canon(enc(a.default_value, lambda t: (_ for _ in ()).throw(Unencodable("tensor default"))))
                except Unencodable:
                    d = _NODEFAULT
            spec.append((a.name, a.kwarg_only, d))
        sc = _SCHEMA[func] = (func.name(), spec)
    return sc


_NODEFAULT = ("<no default>",)


def bind_args(func, args, kwargs, ins_ids):
    """The call bound to the op's schema (see Node.args). ins_ids: id() of the tensors of `ins` in order (a tensor
    passed twice appears twice; each occurrence is matched in flatten order). None when the call cannot be bound."""
    try:
        name, spec = _schema_of(func)
        pos = collections.defaultdict(list)
        for k, i in enumerate(ins_ids):
            pos[i].append(k)
        cursor = collections.Counter()
        def tref(t):
            ks = pos.get(id(t))
            j = cursor[id(t)]
            if not ks or j >= len(ks):
                raise Unencodable("tensor not among the operands")
            cursor[id(t)] += 1
            return {"t": ks[j]}
        npos = len(args)
        if npos > len(spec) or any(k not in {n for n, _, _ in spec} for k in kwargs):
            return None
        out = []
        for j, (an, kwonly, dflt) in enumerate(spec):
            if j < npos and not kwonly:
                v, given = args[j], True
            elif an in kwargs:
                v, given = kwargs[an], True
            else:
                if dflt is _NODEFAULT:
                    return None                       # a required argument the call did not pass: cannot bind
                continue                              # absent = the declared default (the dispatcher strips it)
            e = enc(v, tref)
            if given and dflt is not _NODEFAULT and canon(e) == dflt:
                continue                              # explicitly passed, equal (type and value) to the default
            out.append((an, e))
        return tuple(out)
    except (Unencodable, RuntimeError, AttributeError):
        return None


# ---- F06-1 (Soumil 2026-10-13): forms that compute the same thing compare equal ("equivalent by meaning") -------------
# Never decided by an operator table: each candidate form (a negative dim written non-negatively, an integer scalar
# written as a float, an explicit argument equal in value to the declared default) is accepted only when the op, run on
# the SAME random operands both ways (operand shapes and dtypes as recorded), gives the same outputs. A call whose
# operands are too large, or that is random / fails / differs, keeps its exact form (distinct, never folded).
CANON_DRAWS = 3                  # at least; more when an operand carries few random bits per draw (bool, tiny tensors)
CANON_BITS = 64                  # random bits each operand must carry across the draws
CANON_OUT_MAX = 1 << 16          # a probe whose output would be larger than this (a size argument) is not run: no fold
_SMALL = (2, 3, 5, 7, 11, 13, 17)
_EQ_CACHE = {}
# only operators whose implementation is pure computation are replayed by a probe (a Python custom op may have side
# effects); other namespaces never fold. Why the dim rule is sound whatever the draws: `others` always contains every
# dim value the operands could name, so it contains the TRUE meaning of the negative dim; for a wrong candidate that
# value runs and matches form A, so the fold is refused (the draw count matters only for number / default folds).
PROBE_NAMESPACES = {"aten", "prims"}


@contextlib.contextmanager
def _alarm_blocked():
    """A stage time limit (SIGALRM) never runs inside the block (a probe would swallow it as 'not equivalent'; a restore
    would be cut short): while inside, the alarm only records that it fired, and the original handler runs right after.
    Swapping the Python handler (not only masking the signal) also covers an alarm that arrived just before the block
    (Python runs a signal's handler at the next bytecode, possibly already inside)."""
    fired = []
    try:
        old = signal.signal(signal.SIGALRM, lambda *a: fired.append(a))
    except (ValueError, OSError, TypeError):              # (not the main thread: no handler can be installed)
        old = None
    try:
        yield
    finally:
        if old is not None:
            signal.signal(signal.SIGALRM, old)
            if fired:
                if callable(old):
                    old(*fired[0])
                elif old == signal.SIG_DFL:
                    signal.raise_signal(signal.SIGALRM)


def _is_number_param(a):
    """A parameter that takes a Python number: `number` (Scalar), optional number, or a Tensor parameter given a Python
    number (a wrapped scalar)."""
    t = a.type
    if isinstance(t, torch._C.OptionalType):
        t = t.getElementType()
    return isinstance(t, (torch._C.NumberType, torch._C.TensorType))


def _shape_map(tensors):
    """Stand-in sizes that keep the call's size PATTERN (which sizes are 0 / 1 / equal to each other) but not the sizes
    themselves: the fold decision depends on ranks, dtypes, arguments and that pattern, never on the input length."""
    sizes = sorted({d for t in tensors for d in t.shape if d > 1})
    if len(sizes) > len(_SMALL):
        return None
    m = {d: _SMALL[k] for k, d in enumerate(sizes)}
    m[0], m[1] = 0, 1
    return m


def _call_key(func, args, kwargs):
    """Everything about a call except tensor VALUES: the op, every non-tensor argument (type and value), each tensor's
    stand-in shape and dtype, and which operands are the same tensor."""
    flat, spec = tree_flatten((args, kwargs))
    sm = _shape_map([x for x in flat if isinstance(x, torch.Tensor)]) or {}
    alias, parts = {}, []
    for x in flat:
        if isinstance(x, torch.Tensor):
            parts.append(("T", alias.setdefault(id(x), len(alias)), tuple(sm.get(d, d) for d in x.shape), str(x.dtype)))
        else:
            parts.append(("V", type(x).__name__, repr(x)))
    return (func, str(spec), tuple(parts))


def _probe(args, kwargs, sm, draw):
    """Random operands with the stand-in shapes and the recorded dtypes (one draw; aliases stay aliases). Integers are
    wide, signed and mostly distinct (an op that needs them as indices fails: no fold)."""
    g = torch.Generator().manual_seed(1000 + draw)
    made = {}
    def mk(t):
        if id(t) in made:
            return made[id(t)]
        shape = tuple(sm[d] for d in t.shape)
        if t.is_complex():
            r = torch.complex(torch.randn(shape, generator=g) * 3, torch.randn(shape, generator=g) * 3).to(t.dtype)
        elif t.is_floating_point():
            r = (torch.randn(shape, generator=g, dtype=torch.float64) * 3 + 0.37).to(t.dtype)
        elif t.dtype == torch.bool:
            r = torch.randint(0, 2, shape, generator=g).bool()
        else:
            info = torch.iinfo(t.dtype)
            r = torch.randint(max(info.min, -1000), min(info.max, 1000), shape, generator=g).to(t.dtype)
        made[id(t)] = r
        return r
    return tree_map(lambda v: mk(v) if isinstance(v, torch.Tensor) else v, (args, kwargs))


def _bits(t):
    t = t.detach().contiguous().reshape(-1)
    if t.is_complex():
        t = torch.view_as_real(t).contiguous().reshape(-1)
    return t.view(torch.uint8) if t.numel() else t


def _same_outputs(a, b):
    """Bitwise-identical outputs (dtype, shape and every bit; -0.0 != 0.0)."""
    fa, fb = tree_flatten(a)[0], tree_flatten(b)[0]
    if len(fa) != len(fb):
        return False
    for x, y in zip(fa, fb):
        if isinstance(x, torch.Tensor) != isinstance(y, torch.Tensor):
            return False
        if isinstance(x, torch.Tensor):
            if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(_bits(x), _bits(y)):
                return False
        elif type(x) is not type(y) or x != y:
            return False
    return True


def _draws(tensors, sm):
    """Enough draws that every operand carries CANON_BITS random bits in total (a bool or a 2-element operand carries
    only a few bits per draw, so a few draws can coincide by chance)."""
    def bits(t):
        n = 1
        for d in t.shape:
            n *= sm[d]
        per = 1 if t.dtype == torch.bool else (8 if t.dtype in (torch.uint8, torch.int8) else (10 if not (t.is_floating_point() or t.is_complex()) else 16))
        return max(1, n * per)
    return min(256, max(CANON_DRAWS, -(-CANON_BITS // min(bits(t) for t in tensors))))


def _outputs(func, form, sm, n):
    return [func(*a, **k) for a, k in (_probe(form[0], form[1], sm, d) for d in range(n))]


def _small_enough(func, form, sm):
    """The probe's outputs fit CANON_OUT_MAX (checked on meta tensors: a size argument keeps its real value)."""
    a, k = tree_map(lambda v: torch.empty(tuple(sm[d] for d in v.shape), dtype=v.dtype, device="meta")
                    if isinstance(v, torch.Tensor) else v, (form[0], form[1]))
    out = func(*a, **k)
    return all(o.numel() <= CANON_OUT_MAX for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor))


def _equivalent(func, form_a, form_b, others=(), need_other=True):
    """form_a and form_b (each (args, kwargs)) compute the same thing: on every draw their outputs are bitwise equal (a
    random op's two runs differ, so it never folds), and the probe is INFORMATIVE: each other legal value of the argument that
    runs gives a different output on some draw, and at least one does when any exist (a probe that cannot tell values
    apart proves nothing). need_other=False (dims): when no other value of the range is legal, the candidate is the only
    legal value, so it names the same thing. Random ops are never probed (they would consume the global RNG). Cached on
    the full call keys."""
    key = (_call_key(func, *form_a), _call_key(func, *form_b), tuple(_call_key(func, *o) for o in others))
    if key in _EQ_CACHE:
        return _EQ_CACHE[key]
    ok = False
    rng = torch.get_rng_state()
    try:
        ts = [t for t in tree_flatten(form_a)[0] if isinstance(t, torch.Tensor)]
        sm = _shape_map(ts)
        seeded = getattr(torch.Tag, "nondeterministic_seeded", None)
        if (ts and sm is not None and all(t.device.type == "cpu" and not t.is_meta for t in ts)
                and (not isinstance(func, torch._ops.OpOverload) or func.namespace in PROBE_NAMESPACES)
                and not (seeded is not None and seeded in getattr(func, "tags", ())) and _small_enough(func, form_a, sm)):
            n = _draws(ts, sm)
            oa, ob = _outputs(func, form_a, sm, n), _outputs(func, form_b, sm, n)
            ok = all(_same_outputs(x, y) for x, y in zip(oa, ob))    # (a random op differs between its two runs: no fold)
            ran = 0
            for o in others:
                if not ok:
                    break
                try:
                    oo = _outputs(func, o, sm, n)
                except Exception:
                    continue                              # an illegal other value tells nothing
                ran += 1
                if all(_same_outputs(x, y) for x, y in zip(oa, oo)):
                    ok = False                            # this value is indistinguishable too: the probe is blind
            if ok and need_other and not ran:
                ok = False                                # nothing showed the probe can tell values apart
    except Exception:
        ok = False
    finally:
        torch.set_rng_state(rng)                          # probing never moves the global RNG
    _EQ_CACHE[key] = ok
    return ok


def canonical_args(func, args, kwargs, bound, ins_ids):
    """The canonical form of a bound call (Node.cargs), or None when nothing folds. Candidates per argument, in order:
    a negative dim -> dim + rank of an operand (or rank + 1), an omitted negative default dim -> the same, an int for a
    number parameter -> float (only when every tensor operand is floating: an integer operand changes the computation
    dtype), then an explicit value equal in value to the declared default -> omitted. Each fold is accepted only by
    _equivalent, with the argument's other legal values as the informativeness check."""
    if bound is None:
        return None
    try:
        name, spec = _schema_of(func)
        sch = {a.name: a for a in func._schema.arguments}
        live = {}
        for j, (an, kwonly, dflt) in enumerate(spec):
            if j < len(args) and not kwonly:
                live[an] = ("pos", j)
            elif an in kwargs:
                live[an] = ("kw", an)
        tens = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        ranks = sorted({t.dim() for t in tens})
        # a number may be written as a float only when no INTEGER tensor takes part (it would change the computation
        # dtype): at least one floating operand, the rest floating or bool (bool never decides the dtype against a float)
        all_float = (any(t.is_floating_point() or t.is_complex() for t in tens)
                     and all(t.is_floating_point() or t.is_complex() or t.dtype == torch.bool for t in tens))
        cur_a, cur_k = list(args), dict(kwargs)
        def setv(an, v, a_=None, k_=None):
            a2, k2 = list(cur_a if a_ is None else a_), dict(cur_k if k_ is None else k_)
            w, j = live[an]
            if w == "pos":
                a2[j] = v
            else:
                k2[j] = v
            return a2, k2
        def dim_fold(an, v):
            """v (negative) -> the first c = v + n (n = an operand rank or rank + 1) that computes the same."""
            nonlocal cur_a, cur_k
            top = max(ranks) + 2                          # every dim value any operand could name (rank + 1 included)
            for n in sorted({r for r in ranks} | {r + 1 for r in ranks}):
                c = v + n
                if c < 0:
                    continue
                form_b = setv(an, c)
                others = [setv(an, d) for d in range(top) if d != c]
                if _equivalent(func, (cur_a, cur_k), form_b, others, need_other=False):
                    cur_a, cur_k = form_b
                    return True
            return False
        changed = False
        for an, kwonly, dflt in spec:
            a_s = sch[an]
            if an not in live:
                d0 = a_s.default_value if a_s.has_default_value() else None
                if "dim" in an and isinstance(d0, int) and not isinstance(d0, bool) and d0 < 0:
                    live[an] = ("kw", an); cur_k = dict(cur_k, **{an: d0})       # the omitted default, written out
                    if dim_fold(an, d0):
                        changed = True
                    else:
                        del cur_k[an]; live.pop(an)
                continue
            w, j = live[an]
            v = cur_a[j] if w == "pos" else cur_k[j]
            if "dim" in an and isinstance(v, int) and not isinstance(v, bool) and v < 0:
                changed |= dim_fold(an, v)
            elif ("dim" in an and isinstance(v, (list, tuple)) and v and all(isinstance(d, int) and not isinstance(d, bool) for d in v)
                  and any(d < 0 for d in v)):
                for r in ranks:
                    c = type(v)(d + r if d < 0 else d for d in v)
                    if min(c) < 0 or len(set(c)) != len(c):
                        continue
                    form_b = setv(an, c)
                    top = max(ranks) + 2                  # each element replaced by every other dim value, in turn
                    others = [setv(an, type(v)(c[:i] + type(c)([d]) + c[i + 1:])) for i in range(len(c)) for d in range(top) if d not in c]
                    if _equivalent(func, (cur_a, cur_k), form_b, others, need_other=False):
                        cur_a, cur_k = form_b; changed = True
                        break
            elif isinstance(v, int) and not isinstance(v, bool) and all_float and _is_number_param(a_s):
                form_b = setv(an, float(v))
                if _equivalent(func, (cur_a, cur_k), form_b, [setv(an, float(v) + 1.0)]):
                    cur_a, cur_k = form_b; changed = True
            # an explicit value equal IN VALUE (not type) to the declared default: omitted when that computes the same
            v = cur_a[j] if w == "pos" else cur_k[j]
            d0 = a_s.default_value if a_s.has_default_value() else None
            removable = w == "kw" or j == len(cur_a) - 1
            if (removable and isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(d0, (int, float))
                    and not isinstance(d0, bool) and type(v) is not type(d0) and float(v) == float(d0)):
                a2, k2 = list(cur_a), dict(cur_k)
                if w == "kw":
                    del k2[an]
                else:
                    a2.pop()
                if _equivalent(func, (cur_a, cur_k), (a2, k2), [setv(an, type(v)(float(v) + 1))]):
                    cur_a, cur_k, changed = a2, k2, True
                    live.pop(an)
        if not changed:
            return None
        return bind_args(func, tuple(cur_a), cur_k, ins_ids)
    except Exception:
        return None


class DagRecorder(TorchDispatchMode):
    def __init__(self, model, inputs: dict, skip=True):
        super().__init__()
        self.skip = skip
        self.param_names = {id(t): n for n, t in model.named_parameters(remove_duplicate=False)}
        self.buffer_names = {id(t): n for n, t in model.named_buffers(remove_duplicate=False)}
        self.input_names = {id(v): k for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        self.const_names = {}            # tensors built at construction and stored as plain module attributes
        for mn, mod in model.named_modules():
            for an, av in vars(mod).items():
                if isinstance(av, torch.Tensor) and not isinstance(av, torch.nn.Parameter) and an not in ("_parameters", "_buffers"):
                    self.const_names.setdefault(id(av), f"{mn}.{an}" if mn else an)
        self.producer = {}          # id(tensor) -> node index (entry removed when the tensor dies)
        self.port = {}              # id(tensor) -> output port of that node (F06)
        self.ext_id = {}            # id(tensor) -> ordinal of an external tensor (F06: identity, not just shape)
        self.returns = None         # the model call's returned structure (return_structure), set by analyse()
        self.nodes = []
        self.used_params, self.used_buffers, self.used_inputs = set(), set(), set()
        self.external = []          # (shape, dtype, module) for tensors of unknown origin
        self.skipped = 0
        self.stack = ["(root)"]
        self._hooks = []
        for name, mod in model.named_modules():
            n = name or "(root)"
            self._hooks.append(mod.register_forward_pre_hook(lambda m, a, n=n: (self.stack.append(n), None)[1]))
            self._hooks.append(mod.register_forward_hook(lambda m, a, o, n=n: (self._pop(n), None)[1]))
        self.called = set()

    def _pop(self, n):
        if self.stack and self.stack[-1] == n:
            self.stack.pop()
            self.called.add(n)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    ZERO_KEEP = {"mul_", "div_", "zero_", "neg_", "relu_", "abs_", "clamp_", "clamp_min_", "clamp_max_", "tanh_", "sin_",
                 "sqrt_", "fill_", "masked_fill_", "embedding_renorm_"}     # (zero rows have norm 0: renorm keeps them)

    def _zero_preserving_inplace(self, func, args, kwargs):
        name = func.overloadpacket.__name__
        if name not in self.ZERO_KEEP or not args or not isinstance(args[0], torch.Tensor) or args[0].numel() <= 1:
            return False
        if args[0].untyped_storage().data_ptr() not in ZERO_REG:
            return False
        others = list(args[1:]) + list(kwargs.values())
        def val(x):
            return x.item() if isinstance(x, torch.Tensor) and x.numel() == 1 else x
        if name in ("fill_", "masked_fill_"):
            return all(val(x) == 0 for x in others[-1:] if not isinstance(val(x), torch.Tensor))
        if name == "mul_":
            return all(not isinstance(val(x), torch.Tensor) and math.isfinite(val(x)) or
                       isinstance(x, torch.Tensor) and bool(torch.isfinite(x).all()) for x in others)
        if name == "div_":
            x = others[0] if others else None
            if isinstance(val(x), torch.Tensor):
                return bool(torch.isfinite(x).all()) and bool((x != 0).all())
            return x is not None and val(x) != 0 and math.isfinite(val(x))
        if name in ("clamp_", "clamp_min_", "clamp_max_"):
            if any(isinstance(val(x), torch.Tensor) for x in others):
                return False                            # tensor bounds: not decided here (execute)
            lo = val(others[0]) if others else None
            hi = val(others[1]) if len(others) > 1 else None
            if name == "clamp_max_": lo, hi = None, lo
            return (lo is None or lo <= 0) and (hi is None or hi >= 0)
        return True                                     # zero_, neg_, relu_, abs_, tanh_, sin_, sqrt_ keep zero

    def _forget(self, k):
        self.producer.pop(k, None); self.port.pop(k, None)

    def _track(self, t, idx, port=0):
        k = id(t)
        self.producer[k] = idx
        self.port[k] = port
        try:
            weakref.finalize(t, self._forget, k)
        except TypeError:
            pass

    def _ext(self, t):
        k = id(t)
        if k not in self.ext_id:
            self.ext_id[k] = len(self.ext_id)
            try:
                weakref.finalize(t, self.ext_id.pop, k, None)
            except TypeError:
                pass
        return self.ext_id[k]

    def ref(self, t):
        """The recorder's reference of a tensor (the same tuple an op operand edge uses)."""
        k = id(t)
        if k in self.producer:
            return ("op", self.producer[k], self.port.get(k))
        if k in self.param_names:
            return ("param", self.param_names[k])
        if k in self.buffer_names:
            return ("buffer", self.buffer_names[k])
        if k in self.input_names:
            return ("input", self.input_names[k])
        if k in self.const_names:
            return ("module_constant", self.const_names[k])
        return ("external", tuple(t.shape), str(t.dtype), self._ext(t))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        mod = self.stack[-1]
        ins, consts, reads_weight = [], [], False
        p_read, b_read = [], []          # credited only once the op has completed (F13: an op that raises reads nothing)
        ins_ids = []
        for a in tree_flatten((args, kwargs))[0]:
            if isinstance(a, torch.Tensor):
                k = id(a)
                ins_ids.append(k)
                if k in self.producer:
                    ins.append(("op", self.producer[k], self.port.get(k)))
                elif k in self.param_names:
                    ins.append(("param", self.param_names[k])); p_read.append(self.param_names[k]); reads_weight = True
                elif k in self.buffer_names:
                    ins.append(("buffer", self.buffer_names[k])); b_read.append(self.buffer_names[k]); reads_weight = True
                elif k in self.input_names:
                    ins.append(("input", self.input_names[k])); self.used_inputs.add(self.input_names[k])
                elif k in self.const_names:
                    ins.append(("module_constant", self.const_names[k]))
                else:
                    ins.append(("external", tuple(a.shape), str(a.dtype), self._ext(a)))
                    self.external.append((tuple(a.shape), str(a.dtype), mod, str(func)))
            elif isinstance(a, (int, float, bool)) and not isinstance(a, bool):
                consts.append(a)
        bound = bind_args(func, args, kwargs, ins_ids)      # before execution: an in-place op changes nothing here
        with _alarm_blocked():                            # (a stage limit is delivered after the probes, never swallowed)
            cbound = canonical_args(func, args, kwargs, bound, ins_ids)   # F06-1: the canonical form (None = same as bound)
        try:
            targets = _write_targets(func, args, kwargs)
        except Exception:
            targets = []
        out = None
        reg_targets = [t for t in targets if is_reg(t)]
        if reg_targets:
            if self._zero_preserving_inplace(func, args, kwargs):
                # an in-place op on zero-storage weights whose result is provably still zero (e.g. RWKV rescales weights
                # with div_ at inference): skip the write (the storage is shared), keep the op in the graph
                self.skipped += 1
                out = args[0]
            else:
                for t in reg_targets:             # F07-1 copy-on-write: a whole weight gets its own storage
                    if id(t) in self.param_names or id(t) in self.buffer_names:
                        torch.Tensor.set_(t, torch.zeros(t.shape, dtype=t.dtype))
                        COW_EVENTS.append(f"{func.overloadpacket.__name__} into {self.param_names.get(id(t)) or self.buffer_names.get(id(t))}")
                    else:
                        ZSW_EVENTS.append(f"{func.overloadpacket.__name__} in {mod}")
                        raise ZeroStorageWrite(f"{func} writes into shared zero storage through a view in {mod}")
        elif self.skip and func in SKIP_OPS:
            out = exact_zero_result(func, args, kwargs)
            if out is not None:
                self.skipped += 1
        if out is None:
            out = func(*args, **kwargs)
            # F07-3: non-finite values created from finite inputs (a zero-weight artefact a real model never computes):
            # NaN by any op (e.g. 0/0 normalising a zero embedding); inf only by an op reading a weight (an attention
            # mask legitimately writes -inf). Registered zero storage cannot hold either (registry_intact covers it).
            fo = [] if func.overloadpacket.__name__ in ALLOC_OPS else [   # (allocation = uninitialised memory, no value)
                o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor) and o.is_floating_point() and o.numel()
                and not is_reg(o)]
            bad = any(bool(torch.isnan(o).any()) for o in fo) or (reads_weight and any(bool(torch.isinf(o).any()) for o in fo))
            # a literal non-finite argument excuses only its own kind (masked_fill(-inf) may write inf; it never makes NaN)
            lits = [c for c in tree_flatten((args, kwargs))[0] if isinstance(c, float) and not math.isfinite(c)]
            bad = (not any(math.isnan(c) for c in lits) and any(bool(torch.isnan(o).any()) for o in fo)) or (
                reads_weight and not any(math.isinf(c) for c in lits) and any(bool(torch.isinf(o).any()) for o in fo))
            if bad and all(finite(a) for a in tree_flatten((args, kwargs))[0] if isinstance(a, torch.Tensor)):
                NONFINITE_EVENTS.append(f"{func.overloadpacket.__name__} in {mod}")
        self.used_params.update(p_read); self.used_buffers.update(b_read)
        outs = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
        idx = len(self.nodes)
        self.nodes.append(Node(str(func.overloadpacket.__name__), mod, ins, consts, len(outs),
                               tuple(outs[0].shape) if outs else None, _schema_of(func)[0], bound,
                               tuple((tuple(o.shape), _dtype_name(o.dtype)) for o in outs), cbound))
        for p_, o in enumerate(outs):
            self._track(o, idx, p_)
        for t in targets:                         # in-place / out= mutation: the argument now carries this op's value
            p_ = next((j for j, o in enumerate(outs) if o is t), None)
            self._track(t, idx, p_ if p_ is not None else "w:" + next(
                (sa.name for i_, sa in enumerate(func._schema.arguments) if sa.alias_info is not None and sa.alias_info.is_write
                 and ((args[i_] if i_ < len(args) else kwargs.get(sa.name)) is t)), "?"))
        return out

    def return_structure(self, outputs, max_depth=8):
        """F06: the complete returned structure of the model call: [(path, ref)] in walk order. path = list of keys /
        attribute names / indices; ref = the recorder's reference of a tensor leaf (op output port, parameter, buffer,
        input, module constant or external) - so a returned input or parameter, and one tensor returned under two
        names, are visible; ["container", type name, size] for a container; ["value", typed value] for a scalar /
        None / string leaf; ["opaque", type name] for an object not walked (depth limit, no attributes)."""
        out, seen = [], set()
        def walk(o, path, depth):
            if isinstance(o, torch.Tensor):
                out.append((path, self.ref(o))); return
            if o is None or isinstance(o, (bool, int, float, complex, str, torch.dtype, torch.device)):
                out.append((path, ("value", enc(o, None)))); return
            if depth >= max_depth or id(o) in seen:
                out.append((path, ("opaque", type(o).__name__, "repeat" if id(o) in seen else "depth"))); return
            seen.add(id(o))
            if isinstance(o, dict):                   # ModelOutput is a dict: its fields in order
                out.append((path, ("container", type(o).__name__, len(o))))
                for k, v in o.items():
                    walk(v, path + [str(k)], depth + 1)
            elif isinstance(o, (list, tuple)):
                out.append((path, ("container", type(o).__name__, len(o))))
                for i, v in enumerate(o):
                    walk(v, path + [i], depth + 1)
            elif hasattr(o, "__dict__") and not isinstance(o, torch.nn.Module):
                d = vars(o)
                out.append((path, ("container", type(o).__name__, len(d))))
                for k, v in d.items():
                    walk(v, path + ["." + k], depth + 1)
            else:
                out.append((path, ("opaque", type(o).__name__, "no attributes")))
        walk(outputs, [], 0)
        return out

    def output_nodes(self, outputs):
        return {self.producer[id(t)] for t in tree_flatten(outputs)[0]
                if isinstance(t, torch.Tensor) and id(t) in self.producer}


def layer_of(module_path):
    """(container_path, index) for the innermost repeated-block index in a module path, else None."""
    parts = module_path.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return (".".join(parts[:i]), int(parts[i]))
    return None


def analyse(rec: DagRecorder, outputs):
    """Reachability, closure, opacity, layer patterns, cross-layer edges."""
    nodes = rec.nodes
    try:
        rec.returns = rec.return_structure(outputs)      # F06: kept on the recorder (capture bundle; later T5)
    except Exception as e:                                # never fatal: an unrecorded structure is UNKNOWN (None)
        rec.returns = None
        rec.returns_error = f"{type(e).__name__}: {str(e)[:120]}"
    # backward reachability from model outputs and from control sinks (values that steered Python control flow)
    roots = set(rec.output_nodes(outputs)) | {i for i, n in enumerate(nodes) if n.func in CONTROL_SINKS}
    reach, stack = set(), list(roots)
    while stack:
        i = stack.pop()
        if i in reach:
            continue
        reach.add(i)
        for e in nodes[i].ins:
            if e[0] == "op":
                stack.append(e[1])
    dead = [i for i in range(len(nodes)) if i not in reach]
    opaque = collections.Counter(n.func for n in nodes if OPAQUE.search(n.func))
    big = lambda e: (1 if not e[0] else __import__("math").prod(e[0])) > 1
    python_mediated = [e for e in rec.external if big(e) and "lift_fresh" in e[3]]      # torch.tensor(<python list>) inside forward
    ext_nonscalar = [e for e in rec.external if big(e) and "lift_fresh" not in e[3]]
    # layer signatures and cross-layer edges
    sig = collections.defaultdict(list)
    for n in nodes:
        lo = layer_of(n.module)
        if lo:
            rel = n.module[len(lo[0]) + 1 + len(str(lo[1])):]
            sig[lo].append((re.sub(r"\.\d+", ".N", rel), n.func))
    containers = collections.defaultdict(dict)
    for (c, i), s in sig.items():
        containers[c][i] = hash(tuple(s))
    patterns = {}
    for c, d in containers.items():
        order = [d[i] for i in sorted(d)]
        labels, seen = [], {}
        for h in order:
            seen.setdefault(h, chr(ord("A") + len(seen)) if len(seen) < 26 else "?")
            labels.append(seen[h])
        runs, prev, cnt = [], None, 0
        for l in labels:
            if l == prev:
                cnt += 1
            else:
                if prev: runs.append(f"{prev}x{cnt}" if cnt > 1 else prev)
                prev, cnt = l, 1
        if prev: runs.append(f"{prev}x{cnt}" if cnt > 1 else prev)
        patterns[c] = {"layers": len(order), "distinct_layer_types": len(seen), "pattern": " ".join(runs)[:300]}
    cross = collections.Counter()
    into_layers = collections.Counter()
    for n in nodes:
        lc = layer_of(n.module)
        for e in n.ins:
            if e[0] != "op":
                continue
            lp = layer_of(nodes[e[1]].module)
            if lc and lp and lc[0] == lp[0] and lc[1] - lp[1] > 1:
                cross[f"{lc[0]}: layer i -> layer i+{lc[1]-lp[1]}"] += 1
            if lc and not lp and nodes[e[1]].module != "(root)":
                into_layers[f"{nodes[e[1]].module} -> {lc[0]}.*"] += 1
    acts = sorted({n.func for n in nodes if n.func in {"silu", "gelu", "relu", "tanh", "sigmoid", "softplus", "mish", "elu",
                                                          "leaky_relu", "hardswish", "hardsigmoid", "relu6", "glu", "softmax",
                                                          "_softmax", "log_softmax", "_log_softmax", "erf", "exp"}})
    return {"ops": len(nodes), "zero_skipped": rec.skipped,
            "dead_ops": len(dead), "dead_op_funcs": dict(collections.Counter(nodes[i].func for i in dead).most_common(6)),
            "dead_op_modules": dict(collections.Counter(nodes[i].module for i in dead).most_common(4)),
            "external_nonscalar": len(ext_nonscalar),
            "python_mediated_values": len(python_mediated),
            "python_mediated_examples": [list(map(str, e)) for e in python_mediated[:3]],
            "external_examples": [list(map(str, e)) for e in ext_nonscalar[:5]],
            "opaque_ops": dict(opaque),
            "layer_patterns": patterns,
            "cross_layer_edges": dict(cross.most_common(8)),
            "non_layer_into_layers": dict(into_layers.most_common(8)),
            "activation_like_ops": acts,
            "graph_schema": GRAPH_SCHEMA,
            "unbound_calls": sum(1 for n in nodes if n.args is None),            # F06: calls not bound to a schema
            "unbound_call_ops": dict(collections.Counter(n.func for n in nodes if n.args is None).most_common(6)),
            "returns_recorded": rec.returns is not None,
            "structure_signature": hash(tuple((re.sub(r"\.\d+", ".N", n.module), n.func) for n in nodes))}


def tied_alias_modules(model, called_names):
    """Modules never called whose every parameter is the same tensor object as a parameter of a called module."""
    mods = {n or "(root)": m for n, m in model.named_modules(remove_duplicate=False)}
    live = set()
    for n in called_names:
        m = mods.get(n)
        if m is not None:
            live |= {id(p) for p in m.parameters(recurse=False)}
    out = set()
    for n, m in mods.items():
        ps = list(m.parameters(recurse=True))
        if n not in called_names and ps and all(id(p) in live for p in ps):
            out.add(n)
    return out


import contextlib as _ctx


@_ctx.contextmanager
def math_attention():
    """Force PyTorch's decomposed math attention so no fused attention kernel hides its internals."""
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.MATH]):
            yield
    except ImportError:
        yield



def clear_library_caches():
    """Clear every functools cache that belongs to transformers/diffusers code, so each recorded pass computes
    everything itself (models cache position tables / anchors across calls)."""
    import gc, functools
    n = 0
    for o in gc.get_objects():
        if isinstance(o, functools._lru_cache_wrapper):
            mod = getattr(o, "__module__", "") or getattr(getattr(o, "__wrapped__", None), "__module__", "") or ""
            if mod.startswith(("transformers", "diffusers")):
                try:
                    o.cache_clear(); n += 1
                except Exception:
                    pass
    return n


def node_key(n, edge_of=None):
    """F06 canonical identity of one recorded op (schema 2): qualified overload, module, schema-bound typed arguments
    with each tensor replaced by its edge (kind, producer index or name, output port), and every output's shape and
    dtype. Raises ValueError when the node lacks schema-2 evidence (a schema-1 node, an unbound call, an unknown port):
    an older or incomplete recording can never compare equal by omission."""
    if n.op is None or n.args is None or n.outs is None:
        raise ValueError(f"node {n.func} has no schema-2 record (op / args / outs unknown)")
    edge_of = edge_of or (lambda e: tuple(e))
    for e in n.ins:
        if e[0] == "op" and (len(e) < 3 or e[2] is None):
            raise ValueError(f"node {n.func} reads an op output whose port is unknown")
    def sub(v):
        if isinstance(v, dict) and set(v) == {"t"}:
            return ("T",) + tuple(edge_of(n.ins[v["t"]]))
        if isinstance(v, list):
            return ("l",) + tuple(sub(x) for x in v)
        return canon(v)
    args = n.cargs if n.cargs is not None else n.args          # F06-1: forms that compute the same thing compare equal
    return (n.op, n.module, tuple((a, sub(v)) for a, v in args), tuple((tuple(sh), dt) for sh, dt in n.outs))


def returns_key(returns):
    if returns is None:
        raise ValueError("the returned structure was not recorded (UNKNOWN)")
    def c(r):
        if r[0] == "value":
            return ("value", canon(r[1]))
        return tuple(tuple(x) if isinstance(x, list) else x for x in r)
    return tuple((tuple(p), c(r)) for p, r in returns)
