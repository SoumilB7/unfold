"""Fail-closed op transparency classifier (INTENT.md G4; F10). One rule for every credited recording (main passes,
modes, generate, train, synth, diffusion components, vLLM witness). It reads a QUALIFIED overload (an OpOverload, or
its full name "namespace::name[.overload]"), never a packet name, and returns one of:

  transparent  the op's computation is fixed by its name + recorded arguments, by registry evidence:
                 core       aten overload tagged torch.Tag.core (Core ATen IR, PyTorch's decomposition target)
                 pointwise  aten overload tagged torch.Tag.pointwise
                 view       aten overload whose schema returns an alias of an input and writes nothing
                 factory    aten tensor constructor / *_like constructor (reads metadata only)
                 variant    F10-6 (Soumil 2026-10-14): the in-place (`op_`) or out= variant of a transparent aten
                            overload, by SCHEMA: the same packet family, the same arguments (names, types, keyword-only
                            flags; alias annotations aside), only the result written into a given tensor
                 prim       F10-5: a prims:: op EMITTED BY a PyTorch reference decomposition the recorder is recording
                            (PyTorch's documented primitives); a prims:: op the program calls directly stays opaque
                 reviewed   an overload listed with a reason in everchanging/opacity_reviewed.yaml (owner decision)
  decompose    an aten overload PyTorch itself decomposes (torch._decomp.decomposition_table or
               core_aten_decompositions()): the recorder records it AS that decomposition, grouped under one node
  opaque       anything else: a non-aten namespace (torch.library custom ops, quantized::, vllm_ir::, a prims:: op
               called directly, ...), an aten overload with no evidence, or any op handled by a tensor subclass's own
               __torch_dispatch__
  unknown      the record holds no overload (a schema-1 packet name): never guessed, never failed
"""
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = os.path.join(HERE, "everchanging", "opacity_reviewed.yaml")

try:
    from torch._subclasses.fake_tensor import _like_tensor_constructors as _LIKE
except Exception:
    _LIKE = set()
try:
    from torch._subclasses.fake_tensor import _is_tensor_constructor as _is_ctor
except Exception:
    _is_ctor = lambda f: False

_DEC = None


def decompositions():
    """PyTorch's own decomposition registry: the general table plus the core-ATen set."""
    global _DEC
    if _DEC is None:
        from torch._decomp import decomposition_table, core_aten_decompositions
        _DEC = dict(decomposition_table)
        _DEC.update(core_aten_decompositions())
    return _DEC


def resolve(name):
    """'ns::op[.overload]' -> OpOverload, or None. A packet name (no '::') never resolves."""
    if not isinstance(name, str) or "::" not in name:
        return None
    ns, rest = name.split("::", 1)
    pk, _, ov = rest.partition(".")
    try:
        return getattr(getattr(getattr(torch.ops, ns), pk), ov or "default")
    except Exception:
        return None


def _writes(a):
    return a.alias_info is not None and a.alias_info.is_write


def _arg_key(args):
    """An argument list without alias annotations: (name, type, keyword-only) per argument."""
    return tuple((a.name, str(a.type), bool(a.kwarg_only)) for a in args)


def functional_twin(op):
    """F10-6: the functional aten overload `op` is a variant of, by schema, or None.
    in-place: packet `name_`, the first argument (self) written and nothing else; the twin is an overload of packet
              `name` with the same argument list (alias annotations aside).
    out=    : the written arguments are all keyword-only (the out tensors); the twin is an overload of the SAME packet
              whose argument list equals this one without them."""
    s = op._schema
    w = [a for a in s.arguments if _writes(a)]
    if not w:
        return None
    pk = op.overloadpacket.__name__
    if pk.endswith("_") and not pk.startswith("_") and len(w) == 1 and s.arguments and s.arguments[0] is w[0] \
            and not w[0].kwarg_only:
        base, key = getattr(torch.ops.aten, pk[:-1], None), _arg_key(s.arguments)
    elif all(a.kwarg_only for a in w):
        base, key = op.overloadpacket, _arg_key([a for a in s.arguments if not _writes(a)])
    else:
        return None
    if not isinstance(base, torch._ops.OpOverloadPacket):
        return None
    for o in base.overloads():
        t = getattr(base, o, None)
        if t is not None and t is not op and not any(_writes(a) for a in t._schema.arguments) \
                and _arg_key(t._schema.arguments) == key:
            return t
    return None


def _evidence(op, reviewed):
    """Registry evidence class of one aten overload, or None."""
    tags = set(op.tags)
    if torch.Tag.core in tags:
        return "core"
    if torch.Tag.pointwise in tags:
        return "pointwise"
    s = op._schema
    if (any(r.alias_info is not None for r in s.returns)
            and all(not r.alias_info.is_write for r in s.returns if r.alias_info is not None)
            and not any(_writes(a) for a in s.arguments)):
        return "view"
    if op in _LIKE:
        return "factory"
    try:
        if _is_ctor(op):
            return "factory"
    except Exception:
        pass
    tw = functional_twin(op)
    if tw is not None:
        k, why = _decide(tw, reviewed)
        if k == "transparent":
            return f"variant of {tw.name()} ({why})"
    return None


def _decide(op, reviewed):
    if op.namespace != "aten":
        return "opaque", f"non-aten namespace {op.namespace}"
    ev = _evidence(op, reviewed)
    if ev is not None:
        return "transparent", ev
    if op.name() in reviewed:
        return "transparent", "reviewed: " + reviewed[op.name()]
    if op in decompositions():
        return "decompose", "PyTorch reference decomposition"
    return "opaque", "no registry evidence (not core/pointwise/view/factory/variant/reviewed) and no decomposition"


SECTIONS = ("reviewed", "reviewed_over_decomposition")
OVER_FIELDS = ("reason", "decomposition_unreliable")


def _load_reviewed(path=VOCAB, strict=True):
    """The owner-reviewed overloads -> {overload: reason}. Two sections:
      reviewed                     overload: reason. The overload must be opaque without the review: neither
                                   transparent by evidence (or as a variant of another entry) nor decomposable.
      reviewed_over_decomposition  overload: {reason, decomposition_unreliable}. F10-10: the overload MUST be
                                   decomposable, and the entry must say why PyTorch's decomposition does not reproduce
                                   the kernel (it is then recorded whole, not value-checked). Exactly these two fields.
    Every key must be the exact op.name() of a real aten overload, in one section only, with non-empty strings.
    strict=True: the first problem raises ValueError (selftests, edits). strict=False (the import): every rejected
    entry is left out and reported -> (reviewed, errors); a rejected entry credits nothing, and every graded row and
    witness record then carries a visible NOTE naming it (vocabulary_note), never an unknown (F10-24, Soumil
    2026-10-17, supersedes Fable R2-N1's whole-row unknown): every op is graded on its own evidence, so an op that runs
    and needed the dropped review is still opaque / unknown by itself, and a torch upgrade that invalidates an entry
    (the vLLM venv's torch rejects entries the main torch needs) neither stops the harness nor taints ops that never
    ran."""
    import yaml
    errors = []

    def bad(msg):
        if strict:
            raise ValueError("opacity_reviewed.yaml: " + msg)
        errors.append(msg)
    try:
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
    except Exception as e:
        if strict:
            raise ValueError(f"opacity_reviewed.yaml: unreadable ({type(e).__name__}: {e})")
        return {}, [f"unreadable ({type(e).__name__}: {str(e)[:120]})"]
    if not isinstance(d, dict):
        bad("the file must be a mapping of sections"); d = {}
    for sec in d:
        if sec not in SECTIONS:
            bad(f"unknown section {sec!r} (sections: {', '.join(SECTIONS)})")
    plain, over = d.get("reviewed") or {}, d.get("reviewed_over_decomposition") or {}
    if not isinstance(plain, dict):
        bad("'reviewed' must be a mapping overload -> reason"); plain = {}
    if not isinstance(over, dict):
        bad("'reviewed_over_decomposition' must be a mapping overload -> {reason, decomposition_unreliable}"); over = {}

    def real(k):
        if not isinstance(k, str) or "::" not in k:
            bad(f"{k!r} must be a qualified overload"); return None
        op = resolve(k)
        if op is None or op.name() != k:
            bad(f"{k!r} is not the exact name of a real overload" + (f" (it resolves to {op.name()!r})" if op is not None else ""))
            return None
        if op.namespace != "aten":
            bad(f"{k!r} is not an aten overload"); return None
        return op
    cand = {}                                         # overload name -> (op, reason, section)
    for k, why in plain.items():
        op = real(k)
        if op is None:
            continue
        if not isinstance(why, str) or not why.strip():
            bad(f"{k!r} needs a reason"); continue
        cand[k] = (op, why, "reviewed")
    for k, ent in over.items():
        op = real(k)
        if op is None:
            continue
        if k in cand:
            bad(f"{k!r} is listed in both sections"); cand.pop(k); continue
        if not isinstance(ent, dict) or set(ent) != set(OVER_FIELDS) or not all(isinstance(ent[f], str) and ent[f].strip() for f in OVER_FIELDS):
            bad(f"{k!r} (reviewed_over_decomposition) needs exactly {', '.join(OVER_FIELDS)}, each a non-empty string"); continue
        cand[k] = (op, f"{ent['reason']} (decomposition not used: {ent['decomposition_unreliable']})", "reviewed_over_decomposition")
    out = {}
    for k, (op, why, sec) in cand.items():
        others = {x: c[1] for x, c in cand.items() if x != k}
        kind, ev = _decide(op, others if sec == "reviewed" else {})     # (by evidence + the OTHER entries)
        if sec == "reviewed" and kind != "opaque":
            bad(f"{k!r} needs no review (it is {kind}: {ev})" + (": an entry for a decomposable op belongs in "
                "reviewed_over_decomposition, with why its decomposition is unreliable" if kind == "decompose" else ""))
            continue
        if sec == "reviewed_over_decomposition" and kind != "decompose":
            bad(f"{k!r} is not decomposable by PyTorch (it is {kind}: {ev}): " + ("it needs no review" if kind == "transparent"
                else "list it under reviewed"))
            continue
        out[k] = why
    return out if strict else (out, errors)


REVIEWED, VOCAB_ERRORS = _load_reviewed(VOCAB, strict=False)


def vocabulary_note():
    """F10-24: the NOTE every graded row / witness record carries while this torch rejects vocabulary entries, else
    None. A note, not an unknown: a rejected entry credits nothing, and every op is graded on its own evidence."""
    if not VOCAB_ERRORS:
        return None
    return {"note": "the reviewed-op vocabulary (everchanging/opacity_reviewed.yaml) has entries this torch rejects; they "
            "credit nothing (the ops they name, if they run, are graded on their own evidence)",
            "torch": torch.__version__, "items": list(VOCAB_ERRORS)[:20], "count": len(VOCAB_ERRORS)}


def _pure_view(op):
    """A schema that returns an alias of an input and writes nothing (view / reshape-as-view / expand / transpose ...)."""
    s = op._schema
    return (op.namespace in ("aten", "prims") and bool(s.returns) and all(r.alias_info is not None and not r.alias_info.is_write
            for r in s.returns) and not any(_writes(a) for a in s.arguments))


_VIEW_ONLY = {}


def view_only_decomposition(op, args, kwargs):
    """F10-11 (Soumil 2026-10-14): PyTorch's decomposition of `op` is only views (no arithmetic, no allocation): every op
    it executes is a pure-view schema AND each of its outputs shares its input's storage (it is a view / reshape of it). Derived FROM the decomposition
    by running it once on meta copies of the call's operands (no data, no copies of real memory), cached per overload and
    WHOLE call (shapes, strides, offsets, dtypes, every scalar argument): such an op is transparent by that evidence,
    with no copies and no value check. A decomposition that cannot run on meta, or runs any other op, is not view-only."""
    from torch.utils._pytree import tree_flatten, tree_map
    from torch.utils._python_dispatch import TorchDispatchMode
    leaves, spec = tree_flatten((args, kwargs))
    # Fable R3-S1: the WHOLE call is the key (every tensor's shape, strides, offset and dtype; every other argument's
    # value): a decomposition can branch on any of them (channel_shuffle groups=1 is views only, groups=2 copies)
    key = (op, str(spec), tuple((tuple(x.shape), tuple(x.stride()), x.storage_offset(), x.dtype) if isinstance(x, torch.Tensor)
                                else (type(x).__name__, repr(x)) for x in leaves))
    r = _VIEW_ONLY.get(key)
    if r is not None:
        return r
    ran = []

    class _Log(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, a=(), k=None):
            ran.append(func)
            return func(*a, **(k or {}))
    try:
        ma, mk = tree_map(lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x, (args, kwargs))
        with _Log():
            out = decompositions()[op](*ma, **mk)
        # every op a pure-view schema AND every output really aliases an input (an alias-annotated op such as reshape
        # may copy: channel_shuffle groups>1 reshapes a transposed tensor into a NEW one, a permutation, not a view)
        ins = {x.untyped_storage()._cdata for x in tree_flatten((ma, mk))[0] if isinstance(x, torch.Tensor)}
        outs = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
        r = all(_pure_view(f) for f in ran) and all(o.untyped_storage()._cdata in ins for o in outs)
    except Exception:
        r = False
    _VIEW_ONLY[key] = r
    return r


_CACHE = {}


def _subclass(types):
    """A tensor subclass that runs its own __torch_dispatch__ (its computation happens outside the recording)."""
    for t in types or ():
        if t is torch.Tensor or t is torch.nn.Parameter:
            continue
        if getattr(t, "__torch_dispatch__", None) is not getattr(torch.Tensor, "__torch_dispatch__", None):
            return t
    return None


def decide(op, types=(), in_decomposition=False):
    """-> (kind, evidence); kind in {"transparent", "decompose", "opaque", "unknown"}. in_decomposition: the call was
    emitted by a PyTorch reference decomposition the recorder is recording (F10-5: its prims:: ops are transparent)."""
    sub = _subclass(types)
    if sub is not None:
        return "opaque", f"tensor subclass {sub.__name__} handles the op in its own __torch_dispatch__"
    if isinstance(op, str):
        if "::" not in op:
            return "unknown", "packet name only: the overload was not recorded"
        o = resolve(op)
        if o is None:
            ns = op.split("::", 1)[0]
            return ("opaque", f"non-aten namespace {ns}") if ns != "aten" else ("opaque", "unknown aten overload")
        return decide(o, (), in_decomposition)
    if in_decomposition and op.namespace == "prims":
        return "transparent", "prim: a PyTorch primitive emitted by a recorded reference decomposition (F10-5)"
    r = _CACHE.get(op)
    if r is None:
        r = _CACHE[op] = _decide(op, REVIEWED)
    return r


def classify(name):
    """Compatibility form -> (transparent: bool, evidence or reason). A decomposable op is NOT transparent by itself:
    only its recorded decomposition is (the recorder's group node says whether it was recorded)."""
    kind, why = decide(name)
    return kind == "transparent", (why if kind != "decompose" else "decomposable: record its decomposition")


if __name__ == "__main__":
    import sys
    for n in sys.argv[1:]:
        print(n, decide(n))
