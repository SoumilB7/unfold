"""Capture bundle: what worker_v2 executed and graded, written for the renderer adapter (exec_to_ir.py), so the
diagram is drawn from the same run the benchmark certified instead of a second, independent run.

Opt-in (BENCH_EMIT_CAPTURE=<dir>); grading is unchanged. A bundle holds no weight values:
  - the op graph of the main passes that ran (func, module, input edges, scalar constants, output shape)
  - parameter shapes and tie groups (names sharing one tensor), non-persistent buffers computed at build
    (e.g. RoPE inv_freq)
  - the small square activations added to attention scores (the mask's values)
  - the stored K/V cache tensors of a cache-on run of the first pass (slot path, shape, producing op)
  - the run's decisions (executed class, config after the worker's switches, build call) and its verdict
"""
import base64, collections, gzip, hashlib, json, os
import torch
from dag import DagRecorder, Node

FORMAT = "exec-capture-bundle/1"
SQUARE_MAX = 1 << 16          # numel limit of a square activation kept from an add (as exec_to_ir's MaskCapture)
SQUARE_SIDE_MAX = 256         # stored slice is [S, S]; larger sides are not kept
BUF_MAX = 4096                # non-persistent buffers up to this size are stored with their values
KIND = {"op": "o", "param": "p", "buffer": "b", "input": "i", "module_constant": "c", "external": "x"}
UNKIND = {v: k for k, v in KIND.items()}


class CaptureRecorder(DagRecorder):
    """DagRecorder that also keeps the small square tensors added to attention scores (to read the mask's values).
    The recording itself (nodes, producers, used parameters) is exactly DagRecorder's."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.square_adds = {}
        self.op_strings = {}          # node index -> string arguments (e.g. gelu's approximate="tanh"), dropped by DagRecorder

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = super().__torch_dispatch__(func, types, args, kwargs)
        strs = [a for a in list(args) + list((kwargs or {}).values()) if isinstance(a, str)]
        if strs:
            self.op_strings[len(self.nodes) - 1] = strs
        if func.overloadpacket.__name__ == "add":
            # every square operand, keyed by its position among the op's tensor inputs (the node's `ins` order), so the
            # reader can tell the mask (an operand off the hidden-state path) from the scores
            ts = [a for a in args if isinstance(a, torch.Tensor)]
            sq = {k: a.detach().clone() for k, a in enumerate(ts)
                  if a.dim() >= 2 and a.shape[-1] == a.shape[-2] and a.numel() <= SQUARE_MAX}
            if sq:
                self.square_adds[len(self.nodes) - 1] = sq
        return out


def graph(rec):
    """A recording as interned strings + compact nodes: [func, module, ins, consts, n_out, shape]."""
    strs, sidx = [], {}
    def s(x):
        if x not in sidx:
            sidx[x] = len(strs); strs.append(x)
        return sidx[x]
    nodes = []
    for n in rec.nodes:
        ins = []
        for e in n.ins:
            k = KIND[e[0]]
            if k == "o": ins.append([k, e[1]])
            elif k == "x": ins.append([k, list(e[1]), e[2]])
            else: ins.append([k, s(e[1])])
        nodes.append([s(n.func), s(n.module), ins, list(n.consts), n.n_out, list(n.shape) if n.shape is not None else None])
    return {"strings": strs, "nodes": nodes,
            "op_strings": {str(j): v for j, v in getattr(rec, "op_strings", {}).items()}}


def nodes_of(g):
    """Inverse of graph(): dag.Node objects with the recorder's own edge tuples."""
    S = g["strings"]; out = []
    for func, mod, ins, consts, n_out, shape in g["nodes"]:
        e2 = []
        for e in ins:
            k = UNKIND[e[0]]
            if k == "op": e2.append(("op", e[1]))
            elif k == "external": e2.append(("external", tuple(e[1]), e[2]))
            else: e2.append((k, S[e[1]]))
        out.append(Node(S[func], S[mod], e2, consts, n_out, tuple(shape) if shape is not None else None))
    return out


def squares(rec):
    """Square activations added in the run: node index -> first [S, S] slice (float32), deduplicated by content."""
    by_node, tensors = {}, {}
    for j, per in getattr(rec, "square_adds", {}).items():
        for k, t in per.items():
            a = t.float().reshape(-1, *t.shape[-2:])[0].contiguous()
            if a.shape[-1] > SQUARE_SIDE_MAX:
                continue
            raw = a.numpy().tobytes(); h = hashlib.sha1(raw).hexdigest()[:16]
            tensors.setdefault(h, {"shape": list(a.shape), "f32": base64.b64encode(raw).decode()})
            by_node.setdefault(str(j), {})[str(k)] = h
    return {"by_node": by_node, "tensors": tensors, "format": "per-operand"}


def square_tensors(sq):
    """Inverse of squares(): node index -> {operand position: torch tensor [S, S]} (operand -1: older bundles, which
    kept only the last square operand)."""
    import numpy as np
    T = {h: torch.from_numpy(np.frombuffer(base64.b64decode(v["f32"]), dtype=np.float32).reshape(v["shape"]).copy())
         for h, v in sq["tensors"].items()}
    out = {}
    for j, v in sq["by_node"].items():
        out[int(j)] = {int(k): T[h] for k, h in v.items()} if isinstance(v, dict) else {-1: T[v]}
    return out


def square_ids(sq):
    """node index -> {operand position: content id} (same keys as square_tensors)."""
    return {int(j): ({int(k): h for k, h in v.items()} if isinstance(v, dict) else {-1: v}) for j, v in sq["by_node"].items()}


def cache_slots(rec_c, out):
    """Every 4-D tensor left in the returned cache, in walk order: slot path, shape, the op that produced it."""
    pkv = getattr(out, "past_key_values", None)
    if pkv is None:
        return None
    found, seen = [], set()
    def walk(o, path, depth=0):                       # same traversal (and order) as exec_to_ir's verify_cache
        if depth > 6 or id(o) in seen: return
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.dim() == 4: found.append((path, o))
            return
        if isinstance(o, (list, tuple)):
            for i, x in enumerate(o): walk(x, f"{path}[{i}]", depth + 1)
        elif isinstance(o, dict):
            for k, x in o.items(): walk(x, f"{path}.{k}", depth + 1)
        elif hasattr(o, "__dict__"):
            for k, x in vars(o).items(): walk(x, f"{path}.{k}", depth + 1)
    walk(pkv, "cache")
    return {"container": type(pkv).__name__,
            "slots": [{"path": p, "shape": list(t.shape), "producer": rec_c.producer.get(id(t))} for p, t in found]}


def model_facts(m):
    """Parameter shapes, tie groups, and the non-persistent buffers computed at build (with values when small)."""
    shapes, groups = {}, collections.defaultdict(list)
    for n, p in m.named_parameters(remove_duplicate=False):
        shapes[n] = list(p.shape); groups[id(p)].append(n)
    nonpersistent = set()
    for mn, mod in m.named_modules():
        for bn in getattr(mod, "_non_persistent_buffers_set", set()):
            nonpersistent.add(f"{mn}.{bn}" if mn else bn)
    bufs = {}
    for n, b in m.named_buffers(remove_duplicate=False):
        if n in nonpersistent and b.is_floating_point() and 0 < b.numel() <= BUF_MAX and b.device.type != "meta":
            bufs[n] = {"shape": list(b.shape), "values": b.detach().double().flatten().tolist()}
    return {"param_shapes": shapes, "tie_groups": sorted(sorted(v) for v in groups.values() if len(v) > 1),
            "computed_buffers": bufs}


def input_shapes(kw):
    return {k: (list(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__) for k, v in kw.items()}


def path_for(out_dir, repo):
    return os.path.join(out_dir, repo.replace("/", "__") + ".capture.json.gz")


def write(out_dir, repo, R, cap, model, cfg, build_attn, shipped=None):
    """The bundle: the first pass that ran and the pass with the most inputs (if another one), the cache-on run,
    model facts, decisions and verdict."""
    passes = cap.get("passes", {})          # the worker keeps the first pass and the one with the most inputs
    try:
        config = cfg.to_dict() if cfg is not None else None
    except Exception as e:
        config = {"error": f"{type(e).__name__}: {e}"}
    import transformers
    facts = cap.get("model") or (model_facts(model) if model is not None else None)
    if facts is not None and shipped is not None:
        # per tie group, the names the checkpoint itself ships: two or more = the model built from config ties them,
        # but the library ties them only if their values are equal (which a zero-weight run cannot check)
        facts = dict(facts, tie_groups_shipped=[[n for n in g if n in shipped] for g in facts["tie_groups"]])
    bundle = {
        "format": FORMAT, "repo": repo, "transformers": transformers.__version__, "torch": torch.__version__,
        "verdict": R.get("verdict"), "reason": R.get("reason"), "failed_checks": R.get("failed_checks"),
        "leftovers": R.get("leftovers"),
        "checks": {k: v.get("pass") for k, v in (R.get("checks") or {}).items()},
        "class": R.get("class"), "model_type": R.get("model_type"),
        "decisions": {k: R.get(k) for k in ("class_selected_by_library", "class_selected_by_weights",
                                            "class_from_library_auto_mapping", "implementation_switches",
                                            "data_size_caps")},
        "build_attn_implementation": build_attn, "config": config,
        "max_position_embeddings": cap.get("max_position_embeddings"),
        "model": facts,
        "passes": passes, "pass_order": cap.get("pass_order", list(passes)), "pass_stacks": cap.get("pass_stacks"),
        "cache_run": cap.get("cache_run"), "capture_errors": cap.get("errors"),
    }
    os.makedirs(out_dir, exist_ok=True)
    p = path_for(out_dir, repo); tmp = p + ".tmp"
    with gzip.open(tmp, "wt") as fh:
        json.dump(bundle, fh, default=str)
    os.replace(tmp, p)
    return p


def load(path):
    with gzip.open(path, "rt") as fh:
        b = json.load(fh)
    if b.get("format") != FORMAT:
        raise ValueError(f"not a capture bundle ({b.get('format')})")
    return b
