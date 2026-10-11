"""Fail-closed op transparency classifier (INTENT.md G4; research 2026-10-10, model-benchmark/_reruns/g4_research). An op is TRANSPARENT only by one of these evidence classes,
all read from PyTorch's own op registry (tags / schema / FakeTensor constructor sets), never from the op's name:
  core        aten op overload tagged torch.Tag.core (the Core ATen IR opset: PyTorch's decomposition target)
  view        aten op overload whose schema returns an alias of its input and writes nothing (no arithmetic)
  inplace     aten in-place/out overload whose functional counterpart (same name minus trailing "_") is core/pointwise
  factory     aten tensor constructor (no tensor inputs) or *_like/new_* constructor (reads only metadata)
  pointwise   aten overload tagged torch.Tag.pointwise (an elementwise scalar function fully fixed by name + args)
  reviewed    aten op on the explicit REVIEWED list below (each entry with a reason); everything else is OPAQUE
Anything outside the aten namespace (custom torch.library ops, vllm_ir::, _C_cache_ops::, ...) is OPAQUE.
Classification works on a full name ("aten::addmm" / "aten::add.Tensor") or a bare packet name ("addmm", every
overload of the packet must be transparent for the packet to count as transparent)."""
import torch

# ops that are not core / view / pointwise / factory but whose semantics are a fixed, fully-specified computation
# visible in the recording's arguments. Each needs a reason; the list is a DECISION for the owner, not inferred.
REVIEWED = {
    "lift_fresh": "materialises a Python-built tensor (T3 judges its values separately)",
    "_local_scalar_dense": "reads one value into Python (a value escape, recorded as a sink)",
    "_to_copy": "dtype/device copy", "to": "dtype/device copy", "copy_": "copy values into a tensor",
    "clone": "copy", "detach": "no arithmetic", "detach_": "no arithmetic", "alias": "no arithmetic",
    "_unsafe_view": "view", "unbind": "views", "split": "views", "split_with_sizes": "views",
    "matmul": "CompositeImplicit matmul (decomposes before dispatch; seen only in vLLM witness)",
    "linear": "x @ W^T + b (seen only if CompositeImplicit is bypassed)",
    # single fixed-semantics primitives with no PyTorch reference decomposition (seen on real recordings, 2026-10-09)
    "max": "full reduction max (max.dim is core)", "min": "full reduction min (min.dim is core)", "_is_all_true": "all() of a bool tensor (assert/control)",
    "repeat_interleave": "repeat by counts (data-dependent shape: also a value escape)", "cummax": "running max scan",
    "cummin": "running min scan", "unique_consecutive": "run-length unique (data-dependent shape)",
    "histc": "histogram (MoE token counts)", "transpose_": "in-place metadata transpose",
}
# NOT reviewed (owner decision): _grouped_mm (MoE grouped GEMM; offsets are a recorded input edge)

try:
    from torch._subclasses.fake_tensor import _like_tensor_constructors as _LIKE
except Exception:
    _LIKE = set()
try:
    from torch._subclasses.fake_tensor import _is_tensor_constructor as _is_ctor
except Exception:
    _is_ctor = lambda f: False


def _overloads(name):
    ns, rest = (name.split("::", 1) + [None])[:2] if "::" in name else ("aten", name)
    if ns != "aten":
        return ns, None
    pk_name, _, ov = rest.partition(".")
    if "::" in name and not ov:
        ov = "default"                       # func.name() omits ".default": a full name always means one overload
    pk = getattr(torch.ops.aten, pk_name, None)
    if pk is None:
        return ns, []
    ovs = [ov] if ov else pk.overloads()
    out = []
    for o in ovs:
        try:
            out.append(getattr(pk, o or "default"))
        except Exception:
            pass
    return ns, out


def _cls_one(op):
    tags = set(op.tags)
    if torch.Tag.core in tags: return "core"
    if torch.Tag.pointwise in tags: return "pointwise"
    s = op._schema
    if s.returns and all(r.alias_info is not None and not r.alias_info.is_write for r in s.returns if r.alias_info is not None) \
            and any(r.alias_info is not None for r in s.returns) and not any(a.alias_info is not None and a.alias_info.is_write for a in s.arguments):
        return "view"
    if op in _LIKE or op.overloadpacket in {o.overloadpacket for o in _LIKE}:
        return "factory"
    try:
        if _is_ctor(op): return "factory"
    except Exception:
        pass
    nm = op.overloadpacket.__name__
    if nm.endswith("_") and not nm.startswith("_"):
        base = getattr(torch.ops.aten, nm[:-1], None)
        if base is not None:
            for o in base.overloads():
                b = getattr(base, o)
                if torch.Tag.core in b.tags or torch.Tag.pointwise in b.tags:
                    return "inplace"
    return None


def classify(name):
    """-> (transparent: bool, evidence class or reason)"""
    ns, ovs = _overloads(name)
    if ns != "aten":
        return False, f"non-aten namespace {ns}"
    if not ovs:
        return False, "unknown aten op"
    pk = name.split("::")[-1].split(".")[0]
    classes = [_cls_one(o) for o in ovs]
    if all(c is not None for c in classes):
        return True, "/".join(sorted(set(classes)))
    # packet-level: any transparent overload actually used is unknown when only the packet name is recorded
    if pk in REVIEWED:
        return True, "reviewed: " + REVIEWED[pk]
    if any(c is not None for c in classes) and "." not in name.split("::")[-1]:
        good = sorted({c for c in classes if c})
        return False, f"packet only partly transparent ({good}); record the overload"
    return False, "opaque: not core/view/pointwise/inplace/factory/reviewed"


if __name__ == "__main__":
    import sys, json
    for n in sys.argv[1:]:
        print(n, classify(n))
