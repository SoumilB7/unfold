"""F06 control matrix: the recorded graph must keep every distinction it claims to describe, through serialisation.

    python3 selftest_graph.py [harness_dir]        # default: this file's directory; exit 0 = all as expected

Every NEG pair computes two DIFFERENT things (asserted: the outputs differ, bitwise); the recording of each, written and
read back as a capture bundle would be, must not compare equal. Every VALID pair computes the same thing in equivalent
forms and must compare equal. Every UNKNOWN control feeds an older or incomplete recording to the schema-2 comparison,
which must refuse it (never compare it equal by omission). INFO rows are reported, never failed (owner decisions).

The comparison is the strongest the tree under test offers: schema 2 = dag.node_key per node + dag.returns_key of the
returned structure, after a JSON round trip; a tree without them (HEAD before F06) = the full serialised graph as a
reader loads it (capture.graph -> json -> json.loads, op_strings included) + the analyse() output roots.
Pure torch, tiny modules; no downloads.
"""
import os, sys, json, math, gzip, tempfile
HD = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HD)
import torch
import dag, capture

V2 = hasattr(dag, "node_key") and hasattr(capture, "returns_of")
FAILS, N = [], 0


def check(name, got, want):
    global N
    N += 1
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        FAILS.append(name)


class Fn(torch.nn.Module):
    def __init__(self, f, params=None):
        super().__init__()
        self.f = f
        for k, v in (params or {}).items():
            if v.is_floating_point():
                self.register_parameter(k, torch.nn.Parameter(v))
            else:
                self.register_buffer(k, v)

    def forward(self, x):
        return self.f(self, x)


def record(mod, x):
    rec = capture.CaptureRecorder(mod, {"x": x}, skip=False)
    try:
        with torch.no_grad(), rec:
            y = mod(x)
    finally:
        rec.remove_hooks()
    dag.analyse(rec, y)
    return rec, y


def fingerprint(rec, y):
    """What a reader of the bundle can compare."""
    g = json.loads(json.dumps(capture.graph(rec), default=str))
    if V2:
        return (tuple(dag.node_key(n) for n in capture.nodes_of(g)), dag.returns_key(capture.returns_of(g)))
    return (g, sorted(rec.output_nodes(y)))


def leaves(y):
    from torch.utils._pytree import tree_flatten
    return [t for t in tree_flatten(y)[0] if isinstance(t, torch.Tensor)]


def bits(t):
    t = t.detach()
    if t.is_complex():
        t = torch.view_as_real(t)
    return (str(t.dtype), tuple(t.shape), t.contiguous().reshape(-1).view(torch.uint8).tolist() if t.numel() else [])


def differ(ya, yb):
    if type(ya) is not type(yb):
        return True
    if isinstance(ya, dict) and list(ya) != list(yb):
        return True
    a, b = leaves(ya), leaves(yb)
    return len(a) != len(b) or any(bits(p) != bits(q) for p, q in zip(a, b))


def pair(fa, fb, x, params=None):
    ma, mb = Fn(fa, params), Fn(fb, params)
    ra, ya = record(ma, x.clone())
    rb, yb = record(mb, x.clone())
    return fingerprint(ra, ya) == fingerprint(rb, yb), differ(ya, yb)


def neg(name, fa, fb, x, params=None):
    same, d = pair(fa, fb, x, params)
    if not d:
        check(f"NEG {name}: the two programs really differ", d, True)
        return
    check(f"NEG {name}", "distinguished" if not same else "COLLIDES", "distinguished")


def valid(name, fa, fb, x, params=None, expect_same_output=True):
    same, d = pair(fa, fb, x, params)
    if expect_same_output and d:
        check(f"VALID {name}: the two programs compute the same thing", not d, True)
        return
    check(f"VALID {name}", "equal" if same else "DIFFERS", "equal")


def info(name, fa, fb, x, params=None):
    same, d = pair(fa, fb, x, params)
    print(f"info {name}: outputs {'differ' if d else 'equal'}, recordings {'equal' if same else 'differ'}")


m33 = torch.tensor([[0.3, -1.2, 2.0], [1.5, 0.1, -0.7], [0.9, 2.2, -0.4]])
x10 = torch.arange(10.) * 0.37 - 1.2
xi = torch.tensor([[3, 1, 2], [7, 5, 4], [6, 0, 8]])
W = {"w1": torch.tensor([[1., 2., 0.], [0., 1., 3.], [2., 0., 1.]]), "w2": torch.tensor([[0., 1., 1.], [1., 0., 2.], [3., 1., 0.]])}
G1, G2 = torch.tensor([1., 2., 3.]), torch.tensor([5., 7., 11.])      # tensors from outside the recording (external)

print(f"harness: {HD}  (schema-2 comparison: {V2})")
# ---- 1. producer output ports
neg("unbind: output 0 vs output 1 feeds the add", lambda s, x: x.unbind(0)[0] + 1, lambda s, x: x.unbind(0)[1] + 1, m33)
neg("split: chunk 0 vs chunk 2", lambda s, x: x.split(1)[0] * 2, lambda s, x: x.split(1)[2] * 2, m33)
neg("max(dim): values vs indices (same shape)", lambda s, x: torch.max(x, 1)[0] * 1, lambda s, x: torch.max(x, 1)[1] * 1, m33)
neg("unbind: a - b vs b - a (both edges from one op)", lambda s, x: (lambda u: u[0] - u[1])(x.unbind(0)),
    lambda s, x: (lambda u: u[1] - u[0])(x.unbind(0)), m33)
neg("topk: values vs indices into the next op", lambda s, x: torch.topk(x, 3).values + 0, lambda s, x: torch.topk(x, 3).indices + 0, x10)
# ---- 2. arguments: keyword roles, Booleans, None placeholders, overloads, nesting, types
neg("addmm alpha=2 vs beta=2", lambda s, x: torch.addmm(x, x, x, alpha=2), lambda s, x: torch.addmm(x, x, x, beta=2), m33)
neg("clamp(min=.5) vs clamp(max=.5)", lambda s, x: torch.clamp(x, min=.5), lambda s, x: torch.clamp(x, max=.5), m33)
neg("sort(dim=0) ascending vs descending", lambda s, x: torch.sort(x, dim=0).values,
    lambda s, x: torch.sort(x, dim=0, descending=True).values, x10)
neg("topk(dim=-1) sorted=False vs largest=False", lambda s, x: torch.topk(x, 3, -1, True, False).values,
    lambda s, x: torch.topk(x, 3, -1, largest=False).values, x10)
neg("x**2 vs 2**x (pow overloads)", lambda s, x: x ** 2, lambda s, x: 2 ** x, m33)
neg("x[idx] vs x[:, idx] (nested None path)", lambda s, x: x[s.idx], lambda s, x: x[:, s.idx], m33,
    {"idx": torch.tensor([2, 0, 1])})
neg("to(float16) vs to(bfloat16)", lambda s, x: x.to(torch.float16), lambda s, x: x.to(torch.bfloat16), m33)
neg("mul by 1j vs 2j (complex scalar)", lambda s, x: x * 1j, lambda s, x: x * 2j, m33)
neg("integer tensor * 2 vs * 2.0 (scalar type)", lambda s, x: x * 2, lambda s, x: x * 2.0, xi)
neg("mul by 0.0 vs -0.0 (signed zero)", lambda s, x: x * 0.0, lambda s, x: x * -0.0, m33)
neg("masked_fill -inf vs nan", lambda s, x: x.masked_fill(x > 1, float("-inf")), lambda s, x: x.masked_fill(x > 1, float("nan")), m33)
neg("div rounding floor vs trunc", lambda s, x: torch.div(x, 2, rounding_mode="floor"), lambda s, x: torch.div(x, 2, rounding_mode="trunc"), m33 * 3)
neg("external: G1 read twice vs G1 then G2", lambda s, x: x + G1 + G1, lambda s, x: x + G1 + G2, m33)
# ---- 3. the model call's returned structure
neg("returned dict fields swapped", lambda s, x: (lambda a, b: {"logits": a, "hidden": b})(x + 1, x * 2),
    lambda s, x: (lambda a, b: {"logits": b, "hidden": a})(x + 1, x * 2), m33)
neg("returned tuple order swapped", lambda s, x: (lambda a, b: (a, b))(x + 1, x * 2), lambda s, x: (lambda a, b: (b, a))(x + 1, x * 2), m33)
neg("input returned directly", lambda s, x: (x + 1, x), lambda s, x: (x + 1,), m33)
neg("parameter returned directly", lambda s, x: (x @ s.w1, s.w2), lambda s, x: (x @ s.w1,), m33, dict(W))
neg("one tensor returned under two names", lambda s, x: (lambda y: (y, y))(x + 1), lambda s, x: (x + 1,), m33)
neg("returned container type tuple vs list", lambda s, x: (lambda a, b: (a, b))(x + 1, x * 2), lambda s, x: (lambda a, b: [a, b])(x + 1, x * 2), m33)
neg("returned input vs returned parameter in the same slot", lambda s, x: (x + 1, x), lambda s, x: (x + 1, s.w1), m33, dict(W))


def input_dtype_pair():
    f = lambda s, x: torch.softmax(x * 2, -1)
    ra, ya = record(Fn(f), m33.clone())
    rb, yb = record(Fn(f), m33.double())
    return fingerprint(ra, ya) == fingerprint(rb, yb), differ(ya, yb)


_s, _d = input_dtype_pair()
check("NEG the same ops on a float32 vs float64 input (only output dtypes differ)", "distinguished" if (_d and not _s) else ("COLLIDES" if _d else "no difference"), "distinguished")
# ---- 4. equivalent forms and unchanged graphs
valid("addmm() vs addmm(beta=1, alpha=1)", lambda s, x: torch.addmm(x, x, x), lambda s, x: torch.addmm(x, x, x, beta=1, alpha=1), m33)
valid("gelu() vs gelu(approximate='none')", lambda s, x: torch.nn.functional.gelu(x), lambda s, x: torch.nn.functional.gelu(x, approximate="none"), m33)
valid("sum(1) vs sum(1, keepdim=False)", lambda s, x: x.sum(1), lambda s, x: x.sum(1, keepdim=False), m33)
valid("softmax(-1) vs softmax(dim=-1)", lambda s, x: x.softmax(-1), lambda s, x: x.softmax(dim=-1), m33)
valid("sort() vs sort(dim=-1)", lambda s, x: torch.sort(x).values, lambda s, x: torch.sort(x, dim=-1).values, x10)
valid("the same program twice", lambda s, x: torch.addmm(x, x, x).softmax(-1).unbind(0)[1], lambda s, x: torch.addmm(x, x, x).softmax(-1).unbind(0)[1], m33)
valid("NaN argument recorded twice (typed key, not float ==)", lambda s, x: x.masked_fill(x > 1, float("nan")), lambda s, x: x.masked_fill(x > 1, float("nan")), m33)
valid("in-place op output read downstream", lambda s, x: (lambda y: y.add_(1) * 2)(x + 1), lambda s, x: (lambda y: y.add_(1) * 2)(x + 1), m33)


def same_program_other_values():
    f = lambda s, x: (torch.topk(x @ s.w1, 2).indices.float() + x.unbind(0)[2].sum(), x.max(1))
    ra, ya = record(Fn(f, dict(W)), m33.clone())
    rb, yb = record(Fn(f, dict(W)), (m33 * -3 + 1).clone())
    return fingerprint(ra, ya) == fingerprint(rb, yb)


check("VALID the same program on other input values (structure, not values)", "equal" if same_program_other_values() else "DIFFERS", "equal")


def roundtrip_exact():
    """A tiny Llama recorded, serialised and read back: every node and the returned structure survive exactly."""
    import transformers
    c = transformers.LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    m = transformers.LlamaForCausalLM._from_config(c, attn_implementation="eager", dtype=torch.float32).eval()
    kw = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    rec = capture.CaptureRecorder(m, kw, skip=False)
    try:
        with torch.no_grad(), dag.math_attention(), rec:
            y = m(**kw, use_cache=False)
    finally:
        rec.remove_hooks()
    dag.analyse(rec, y)
    g = json.loads(json.dumps(capture.graph(rec), default=str))
    back = capture.nodes_of(g)
    if not V2:
        return all((a.func, a.module, [tuple(e) for e in a.ins], list(a.consts), a.n_out, a.shape) ==
                   (b.func, b.module, [tuple(e) for e in b.ins], list(b.consts), b.n_out, b.shape) for a, b in zip(rec.nodes, back)) \
            and False                                         # schema 1 keeps no returned structure to compare
    return [dag.node_key(n) for n in rec.nodes] == [dag.node_key(n) for n in back] and \
        dag.returns_key(rec.returns) == dag.returns_key(capture.returns_of(g)) and all(n.args is not None for n in rec.nodes)


check("ROUNDTRIP tiny Llama: every node, port, argument and the returned structure survive the round trip", roundtrip_exact(), True)


# ---- 5. older or incomplete recordings never satisfy the schema-2 comparison
def refused(fn):
    """'refused' only for the deliberate refusal (ValueError); a tree without the schema-2 check cannot refuse."""
    try:
        fn()
    except ValueError:
        return "refused"
    except (AttributeError, TypeError) as e:
        return f"NO SCHEMA-2 CHECK ({type(e).__name__})"
    return "ACCEPTED"


def old_bundle():
    """A bundle in the pre-F06 format (/2, schema-1 graph rows)."""
    rec, y = record(Fn(lambda s, x: x.unbind(0)[1] + 1), m33.clone())
    g1 = {"strings": [], "nodes": [], "op_strings": {}}
    S = {}
    def s_(v):
        if v not in S: S[v] = len(g1["strings"]); g1["strings"].append(v)
        return S[v]
    for n in rec.nodes:
        ins = [["o", e[1]] if e[0] == "op" else (["x", list(e[1]), e[2]] if e[0] == "external" else [capture.KIND[e[0]], s_(e[1])])
               for e in n.ins]
        g1["nodes"].append([s_(n.func), s_(n.module), ins, list(n.consts), n.n_out, list(n.shape) if n.shape else None])
    b = {"format": "exec-capture-bundle/2", "repo": "selftest/old", "passes": {"p": {"graph": g1}}, "pass_order": ["p"]}
    p = os.path.join(tempfile.mkdtemp(prefix="f06_"), "old.capture.json.gz")
    with gzip.open(p, "wt") as fh:
        json.dump(b, fh)
    return p


def stronger_check_on_old_bundle():
    capture.load(old_bundle())                                # default load: a pre-F06 tree takes /2 as current


def old_bundle_with_legacy_flag():
    b = capture.load(old_bundle(), allow_legacy=True)
    assert b["graph_schema"] == 1
    [dag.node_key(n) for n in capture.nodes_of(b["passes"]["p"]["graph"])]


def returns_unrecorded():
    rec = capture.CaptureRecorder(Fn(lambda s, x: x + 1), {"x": m33}, skip=False)
    try:
        with torch.no_grad(), rec:
            rec.model_out = Fn(lambda s, x: x + 1)(m33)
    finally:
        rec.remove_hooks()
    g = json.loads(json.dumps(capture.graph(rec), default=str))      # no analyse(), no outputs: returns unknown
    dag.returns_key(capture.returns_of(g))


def unbound_call():
    dag.node_key(dag.Node("addmm", "(root)", [("op", 0, 0)], [], 1, (3, 3), "aten::addmm", None, (((3, 3), "float32"),)))


def v1_edges_port_unknown():
    b = capture.load(old_bundle(), allow_legacy=True)
    ports = {e[2] if len(e) > 2 else "absent" for n in capture.nodes_of(b["passes"]["p"]["graph"]) for e in n.ins if e[0] == "op"}
    if ports != {None}:
        return f"ports read as {sorted(map(str, ports))}"
    raise ValueError("unknown")


def port_unknown():
    dag.node_key(dag.Node("add", "(root)", [("op", 0, None)], [1], 1, (3,), "aten::add.Tensor", (("self", {"t": 0}), ("other", 1)), (((3,), "float32"),)))


check("UNKNOWN a pre-F06 bundle (format /2) is refused by the default load", refused(stronger_check_on_old_bundle), "refused")
check("UNKNOWN a pre-F06 bundle loaded with allow_legacy still cannot be compared", refused(old_bundle_with_legacy_flag), "refused")
check("UNKNOWN a recording whose returned structure was not recorded cannot be compared", refused(returns_unrecorded), "refused")
check("UNKNOWN an op call that could not be bound to its schema cannot be compared", refused(unbound_call), "refused")
check("UNKNOWN an edge whose producer port is unknown cannot be compared", refused(port_unknown), "refused")
check("UNKNOWN a schema-1 op edge reads back with port None (unknown), never port 0", refused(v1_edges_port_unknown), "refused")

# ---- 6. Soumil's D1 ruling (2026-10-13, F06-1 "also by meaning"): forms that compute the same thing compare equal;
#      a form is folded only when running the op both ways on the same random operands gives the same result.
def rule(name, fa, fb, x, want, params=None):
    same, d = pair(fa, fb, x, params)
    print(f"     (outputs {'differ' if d else 'equal'})", end=" ")
    check(f"RULE-D1 {name}", "equal" if same else "kept distinct", want)


rule("addmm() vs addmm(alpha=1.0): the default written as a float computes the same", lambda s, x: torch.addmm(x, x, x),
     lambda s, x: torch.addmm(x, x, x, alpha=1.0), m33, "equal")
rule("sort(dim=0) vs sort(dim=-1) on a 1-D tensor: the same dim", lambda s, x: torch.sort(x, dim=0).values,
     lambda s, x: torch.sort(x, dim=-1).values, x10, "equal")
rule("softmax(dim=-1) vs softmax(dim=1) on a 2-D tensor: the same dim", lambda s, x: torch.softmax(x, dim=-1),
     lambda s, x: torch.softmax(x, dim=1), m33, "equal")
rule("unsqueeze(-1) vs unsqueeze(2) on a 2-D tensor: the same position (rank + 1)", lambda s, x: x.unsqueeze(-1) * 2,
     lambda s, x: x.unsqueeze(2) * 2, m33, "equal")
rule("x + 1 vs x + 1.0 on a float tensor: the same result", lambda s, x: x + 1, lambda s, x: x + 1.0, m33, "equal")
# ... and only those: forms that compute something else stay distinct even when shapes collide
X22 = torch.tensor([[0.3, -1.2], [1.5, 0.1]])
rule("stack(dim=-1) vs stack(dim=1) of 2x2 tensors (same output shape, transposed values)", lambda s, x: torch.stack([x, x * 2], dim=-1),
     lambda s, x: torch.stack([x, x * 2], dim=1), X22, "kept distinct")
rule("x * 2 vs x * 2.0 on an INTEGER tensor (int vs float result)", lambda s, x: x * 2, lambda s, x: x * 2.0, xi, "kept distinct")
rule("sum(dim=0) vs sum(dim=-1) on a square matrix (different dims)", lambda s, x: x.sum(dim=0), lambda s, x: x.sum(dim=-1), m33, "kept distinct")
rule("x ** 2 vs x ** 2.0 on a float tensor (a number parameter)", lambda s, x: x ** 2, lambda s, x: x ** 2.0, m33, "equal")
rule("clamp(min=0) vs clamp(min=0.0) on a float tensor", lambda s, x: x.clamp(min=0), lambda s, x: x.clamp(min=0.0), m33, "equal")
# cross-exam B2: random-operand coincidences must not fold
I2 = torch.tensor([5, 9])
rule("stack(dim=-1) vs stack(dim=0) of 1-D INTEGER tensors ([5,9]: transposed layouts)", lambda s, x: torch.stack([x, x * 2], dim=-1),
     lambda s, x: torch.stack([x, x * 2], dim=0), I2, "kept distinct")
rule("index_fill(-1) vs index_fill(0) on a square matrix (an index covering every row)", lambda s, x: x.index_fill(-1, torch.tensor([0, 2]), 7.),
     lambda s, x: x.index_fill(0, torch.tensor([0, 2]), 7.), m33, "kept distinct")
XI = torch.tensor([16777216, 3, 7])
rule("eq(int64 x, 16777217) vs eq(x, 16777217.0): an integer operand changes the computation dtype", lambda s, x: torch.eq(x, 16777217),
     lambda s, x: torch.eq(x, 16777217.0), XI, "kept distinct")

rule("sum(-1) vs sum(0) on a 1-D tensor (a 0-d output folds)", lambda s, x: x.sum(-1), lambda s, x: x.sum(0), x10, "equal")
B2_ = torch.tensor([False, True])
rule("stack(dim=-1) vs stack(dim=0) of 1-D BOOL tensors (1 random bit per element)", lambda s, x: torch.stack([x, ~x], dim=-1),
     lambda s, x: torch.stack([x, ~x], dim=0), B2_, "kept distinct")
rule("stack(dim=-1) vs stack(dim=1) of 1x2 BOOL tensors", lambda s, x: torch.stack([x[None], ~x[None]], dim=-1),
     lambda s, x: torch.stack([x[None], ~x[None]], dim=1), B2_, "kept distinct")
rule("masked_fill(m, 3) vs 3.0 on an INTEGER tensor: never written as a float", lambda s, x: x.masked_fill(x > 4, 3),
     lambda s, x: x.masked_fill(x > 4, 3.0), xi, "kept distinct")
rule("masked_fill(m, 0) vs 0.0 on a float tensor (a bool mask does not decide the dtype)", lambda s, x: x.masked_fill(x > 0, 0),
     lambda s, x: x.masked_fill(x > 0, 0.0), m33, "equal")

# ---- 6b. the fold machinery itself (cross-exam B1 / S3 mutants) -----------------------------------------------------
if hasattr(dag, "_equivalent"):
    dag._EQ_CACHE.clear()
    t_ = torch.tensor([[1., 2.], [3., 4.]]); idx_ = torch.tensor([0]); src_ = torch.tensor([[1.], [1.]])
    aten_ = torch.ops.aten
    def _rec_key(f, x):
        r_, y_ = record(Fn(f), x)
        return [dag.node_key(n) for n in r_.nodes]
    k_fresh = _rec_key(lambda s, x: x.index_add(-1, torch.tensor([0]), torch.ones(2, 1)), t_)
    _rec_key(lambda s, x: x.index_add(-1, torch.tensor([0]), torch.ones(2, 1), alpha=0), t_)          # an earlier call
    k_after = _rec_key(lambda s, x: x.index_add(-1, torch.tensor([0]), torch.ones(2, 1)), t_)
    check("B1 a call's canonical key does not depend on what ran earlier in the process (cache keyed on the full call)", k_fresh == k_after, True)
    check("S3 bitwise: the same values in another dtype are not the same output",
          dag._same_outputs(torch.tensor([1., 2.]), torch.tensor([1., 2.], dtype=torch.float64)), False)
    check("S3 bitwise: integers 1e6 vs 1e6+1 are not the same output (no tolerance)",
          dag._same_outputs(torch.tensor([1000000]), torch.tensor([1000001])), False)
    check("S3 bitwise: -0.0 vs 0.0 are not the same output", dag._same_outputs(torch.tensor([-0.0]), torch.tensor([0.0])), False)
    check("S3 bitwise: float values 1e-7 apart are not the same output",
          dag._same_outputs(torch.tensor([1.0]), torch.tensor([1.0 + 1e-6])), False)
    import random as _r
    check("S3 a non-deterministic op never folds (the form is run twice)",
          dag._equivalent(lambda x: x + _r.random(), ((t_,), {}), ((t_,), {})), False)
    check("S3 an uninformative probe never folds (another legal value gives the same 1-d output)",
          dag._equivalent(lambda x, d: x * 1 + 0 * d, ((t_, 1), {}), ((t_, 1.0), {}), [((t_, 2.0), {})]), False)
    check("S3 ... while an informative one does (sanity: the control rejects for the right reason)",
          dag._equivalent(lambda x, d: x * d, ((t_, 1), {}), ((t_, 1.0), {}), [((t_, 2.0), {})]), True)
    check("R2-S1 a 0-d output compares (bitwise view of a scalar tensor)", dag._same_outputs(torch.tensor(2.5), torch.tensor(2.5)), True)
    check("S3 the call key keeps the size PATTERN (3x4 vs 4x4 differ) ...",
          dag._call_key(aten_.add.Tensor, (torch.ones(3, 4), torch.ones(3, 4)), {}) != dag._call_key(aten_.add.Tensor, (torch.ones(4, 4), torch.ones(4, 4)), {}), True)
    check("S2 ... but not the sizes (3x4 vs 30x40 are one stand-in): a fold never depends on the input length",
          dag._call_key(aten_.add.Tensor, (torch.ones(3, 4), torch.ones(3, 4)), {}) == dag._call_key(aten_.add.Tensor, (torch.ones(30, 40), torch.ones(30, 40)), {}), True)
    # a custom op (registered here) to steer the probe: f(x, v, dim, s) = (x * s).sum(dim) + v * 0, and g ignores dim / s
    _lib = torch.library.Library("selftestgraph", "DEF")
    _lib.define("f(Tensor x, Tensor v, int dim, float s) -> Tensor")
    _lib.impl("f", lambda x, v, dim, s: (x * s).sum(dim) + v * 0, "CompositeExplicitAutograd")
    _lib.define("g(Tensor x, int dim, Scalar s) -> Tensor")
    _lib.impl("g", lambda x, dim, s: x * 1.0, "CompositeExplicitAutograd")
    _lib.define("h(Tensor a, Tensor b, int dim) -> Tensor")
    _lib.impl("h", lambda a, b, dim: (a * 2 - b).sum(dim), "CompositeExplicitAutograd")
    import time as _time
    _lib.define("slow(Tensor x, int dim) -> Tensor")
    _lib.impl("slow", lambda x, dim: (_time.sleep(0.02), torch.softmax(x, dim))[1], "CompositeExplicitAutograd")
    _CALLS = {"seeded": 0}
    def _seeded_impl(x, dim):
        _CALLS["seeded"] += 1
        return torch.softmax(x, dim)
    _lib.define("seeded(Tensor x, int dim) -> Tensor", tags=(torch.Tag.nondeterministic_seeded,))
    _lib.impl("seeded", _seeded_impl, "CompositeExplicitAutograd")
    _lib.define("rng(Tensor x, int dim) -> Tensor")
    _lib.impl("rng", lambda x, dim: torch.softmax(x, dim) + 0 * torch.rand(()), "CompositeExplicitAutograd")
    _lib.define("only1(Tensor x, Scalar s) -> Tensor")
    def _only1(x, s):
        if s != 1:
            raise ValueError("s must be 1")
        return x * 1.0
    _lib.impl("only1", _only1, "CompositeExplicitAutograd")
    dag.PROBE_NAMESPACES.add("selftestgraph")          # (the controls' own pure ops may be replayed by a probe)
    opf, opg, oph = torch.ops.selftestgraph.f.default, torch.ops.selftestgraph.g.default, torch.ops.selftestgraph.h.default
    x33_, v3_ = torch.randn(3, 3), torch.randn(3)
    def _ckey(f_, *a):
        r_ = dag.DagRecorder(Fn(lambda s, x: f_(*a)), {"x": a[0]}, skip=False)
        with torch.no_grad(), r_:
            f_(*a)
        r_.remove_hooks()
        return [dag.node_key(n) for n in r_.nodes]
    dag._EQ_CACHE.clear()
    _ckey(opf, x33_, v3_, -1, 0.0)                     # s = 0: every dim gives zeros (not foldable, cached)
    check("R6 the cache key holds the scalar arguments: after an s=0 call, f(dim=-1, s=1) is still not keyed like dim 0",
          _ckey(opf, x33_, v3_, -1, 1.0) != _ckey(opf, x33_, v3_, 0, 1.0), True)
    check("R10 a dim the op ignores never folds (every other dim gives the same output: blind probe)",
          [n.cargs for n in record(Fn(lambda s, x: opg(x, -1, 2.0)), x33_)[0].nodes], [None])
    check("R13 a number the op ignores is never written as a float (blind probe)",
          [n.cargs is None or all(a_ != "s" or isinstance(v_, int) for a_, v_ in n.cargs) for n in record(Fn(lambda s, x: opg(x, 0, 2)), x33_)[0].nodes], [True])
    dag._EQ_CACHE.clear()
    _ka = _ckey(oph, x33_, x33_, -1)                   # the same tensor twice: a*2 - a = a, informative
    _kb = _ckey(oph, x33_, torch.randn(3, 3), -1)
    check("R7 the cache key holds the alias pattern (h(a, a) and h(a, b) are proven separately; both fold dim -1)",
          dag._call_key(oph, (x33_, x33_, -1), {}) != dag._call_key(oph, (x33_, torch.randn(3, 3), -1), {}), True)
    _sz = torch.ops.aten.new_full.default
    check("R2-S2 a probe whose output would be huge (a size argument) is not run",
          dag._small_enough(_sz, ((torch.ones(3), [8192, 8192], 0), {}), {3: 2, 1: 1, 0: 0}), False)
    _rs0 = torch.get_rng_state()
    record(Fn(lambda s, x: torch.ops.aten.rrelu.default(x, 0, 1, True)), m33)
    _rs1 = torch.get_rng_state(); torch.set_rng_state(_rs0)
    torch.ops.aten.rrelu.default(m33, 0, 1, True)
    check("R2-S3 probing never moves the global RNG (recording a random op leaves it where a plain run does)",
          torch.equal(_rs1, torch.get_rng_state()), True)
    # R3-B1 a stage time limit firing during a probe is delivered, never swallowed
    import signal as _sg
    dag._EQ_CACHE.clear()
    _old = _sg.signal(_sg.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError("stage limit")))
    _got = None
    try:
        _sg.setitimer(_sg.ITIMER_REAL, 0.05)
        record(Fn(lambda s, x: torch.ops.selftestgraph.slow.default(x, -1)), x33_)
        _sg.setitimer(_sg.ITIMER_REAL, 0); _time.sleep(0.01)
    except TimeoutError as ex:
        _got = ex
    finally:
        _sg.setitimer(_sg.ITIMER_REAL, 0); _sg.signal(_sg.SIGALRM, _old)
    check("R3-B1 a stage time limit that fires while a fold is being probed still reaches the worker", _got is not None, True)
    _CALLS["seeded"] = 0
    record(Fn(lambda s, x: torch.ops.selftestgraph.seeded.default(x, -1)), x33_)
    check("R18 an op tagged nondeterministic_seeded is never replayed by a probe (it ran exactly once)", _CALLS["seeded"], 1)
    dag._EQ_CACHE.clear()
    _r0 = torch.get_rng_state()
    record(Fn(lambda s, x: torch.ops.selftestgraph.rng.default(x, -1)), x33_)
    _r1 = torch.get_rng_state(); torch.set_rng_state(_r0); torch.ops.selftestgraph.rng.default(x33_, -1)
    check("R19 an UNTAGGED op that draws from the global RNG: probing still leaves the RNG where a plain run does",
          torch.equal(_r1, torch.get_rng_state()), True)
    dag._EQ_CACHE.clear()
    check("R2/R17 a number whose other values all raise is never written as a float (nothing showed the probe can tell values apart)",
          [n.cargs for n in record(Fn(lambda s, x: torch.ops.selftestgraph.only1.default(x, 1)), x33_)[0].nodes], [None])
    _l2 = torch.library.Library("selftestgraphx", "DEF")
    _l2.define("sx(Tensor x, int dim) -> Tensor"); _l2.impl("sx", lambda x, dim: torch.softmax(x, dim), "CompositeExplicitAutograd")
    check("R3-S2 an op outside the probed namespaces (a custom op that may have side effects) is never replayed: no fold",
          [n.cargs for n in record(Fn(lambda s, x: torch.ops.selftestgraphx.sx.default(x, -1)), x33_)[0].nodes], [None])
    big = torch.randn(64, 2048)
    kb = [n.cargs for n in record(Fn(lambda s, x: torch.softmax(x, dim=-1)), big)[0].nodes]
    ks = [n.cargs for n in record(Fn(lambda s, x: torch.softmax(x, dim=-1)), big[:2, :8])[0].nodes]
    check("S2 softmax(dim=-1) folds the same way at 64x2048 (over 65k elements) as at 2x8", kb == ks and kb[0] is not None, True)
else:
    for n_ in ("B1", "S3 dtype", "S3 int", "S3 -0", "S3 float", "S3 nondeterministic", "S3 uninformative", "S3 informative", "R2-S1",
               "S3 key pattern", "S2 key size", "R6", "R10", "R13", "R7", "R2-S2", "R2-S3", "R3-B1", "R18", "R19", "R2/R17",
               "R3-S2", "S2 big"):
        check(f"{n_} (no fold machinery in the harness under test)", False, True)

# ---- 7. owner rulings F06-4 / D5 --------------------------------------------------------------------------------------
class RetP(torch.nn.Module):
    def __init__(s): super().__init__(); s.w = torch.nn.Parameter(torch.ones(3)); s.unused = torch.nn.Parameter(torch.ones(3))
    def forward(s, x): return {"y": x * s.w, "raw": s.unused}
_rm = RetP(); _rr, _ry = record(_rm, torch.ones(3))
_ret = getattr(_rr, "returns", None) or []
check("F06-4 a parameter the model only RETURNS is recorded in the returned structure",
      any(r[0] == "param" and r[1] == "unused" for _, r in _ret) if V2 else False, True)
check("F06-4 ... and earns no coverage credit (used_params excludes it)", "unused" not in _rr.used_params and "w" in _rr.used_params, True)
import ast as _ast
_reads = []
import common as _common
for _f in sorted(set(_common.GRADING_FILES) | {"attempts.py"}):
    _p = os.path.join(HD, _f)
    if os.path.exists(_p):
        if _f == "capture.py":
            continue                                      # (the bundle writer serialises it FOR the drawing)
        _reads += [f"{_f}:{n_.lineno}" for n_ in _ast.walk(_ast.parse(open(_p).read()))
                   if isinstance(n_, _ast.Attribute) and n_.attr == "consts" and isinstance(n_.ctx, _ast.Load)]
check("D5 no grading file reads the legacy flattened constants (Node.consts is drawing-only)", _reads, [])

# ---- 8. worker T5 info field (F06-2): present, info only ------------------------------------------------------------
_keep_st = [n_ for n_ in __import__("ast").walk(__import__("ast").parse(open(os.path.join(HD, "worker_v2.py")).read()))
            if isinstance(n_, __import__("ast").Assign) and any(getattr(t_, "id", None) == "_keep" for t_ in n_.targets)]
if _keep_st:
    _ks = __import__("ast").Module(body=[_keep_st[0]], type_ignores=[])
    _kept = {}
    for _lb, _sg in (("text", {}), ("image", {"text": 1}), ("stability:text", {"text": 1, "image": 1})):
        _e = {"label": _lb, "sigs": _sg, "any": any}
        exec(compile(_ks, "<_keep>", "exec"), _e); _kept[_lb] = _e["_keep"]
    check("R16 the worker keeps the detailed record for exactly the runs T5 compares (first main pass + stability)",
          _kept, {"text": True, "image": False, "stability:text": True})
    _W2s = open(os.path.join(HD, "worker_v2.py")).read()
    check("R16 ... and stores it where T5 reads it (main passes by _keep; synth and its stability run always)",
          ('rec.nodes if _keep else None)' in _W2s, _W2s.count('[(n.module, n.func) for n in rec.nodes], rec.nodes)'),
           '[(n.module, n.func) for n in rec2.nodes], rec2.nodes)' in _W2s), (True, 1, True))
else:
    check("R16 the worker keeps the detailed record for T5's runs", False, True)

_W2 = open(os.path.join(HD, "worker_v2.py")).read()
_tree = __import__("ast").parse(_W2)
_t5 = next((n_ for n_ in _tree.body if isinstance(n_, __import__("ast").If) and "T5_stable_structure" in (__import__("ast").get_source_segment(_W2, n_) or "")
            and "stability:" in (__import__("ast").get_source_segment(_W2, n_) or "")), None)
if _t5 is not None and "info_schema2_same" in __import__("ast").get_source_segment(_W2, _t5):
    ra_, _ = record(Fn(lambda s, x: torch.softmax(x, -1) * 2), m33); rb_, _ = record(Fn(lambda s, x: torch.softmax(x, -1) * 3), m33)
    env_ = {"__import__": __import__, "sigs": {"text": (None, [(n.module, n.func) for n in ra_.nodes], ra_.nodes),
                                               "stability:text": (None, [(n.module, n.func) for n in rb_.nodes], rb_.nodes)},
            "first": "text", "checks": {}, "dag": dag, "next": next, "range": range, "min": min, "len": len, "repr": repr, "str": str}
    exec(compile(__import__("ast").Module(body=[_t5], type_ignores=[]), "<T5 block>", "exec"), env_)
    _c = env_["checks"]["T5_stable_structure"]
    check("F06-2 T5 keeps its legacy pass and adds an info-only schema-2 field (here: same ops, different scalar -> differs)",
          (_c["pass"], _c.get("info_schema2_same")), (True, False))
    env_["sigs"]["stability:text"] = (None, env_["sigs"]["stability:text"][1], [object()])   # an unexpected record
    exec(compile(__import__("ast").Module(body=[_t5], type_ignores=[]), "<T5 block>", "exec"), env_)
    _c = env_["checks"]["T5_stable_structure"]
    check("R15 any error in the info block (not only ValueError) leaves T5 as it was", (_c["pass"], _c.get("info_schema2_same")), (True, None))
    env_["sigs"]["stability:text"] = (None, env_["sigs"]["stability:text"][1], None)
    exec(compile(__import__("ast").Module(body=[_t5], type_ignores=[]), "<T5 block>", "exec"), env_)
    _c = env_["checks"]["T5_stable_structure"]
    check("F06-2 a missing detailed record gives info None, never a failed T5", (_c["pass"], _c.get("info_schema2_same")), (True, None))
else:
    check("F06-2 worker T5 info field present", False, True)

print(f"{len(FAILS)} unexpected of {N}" + (": " + "; ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
