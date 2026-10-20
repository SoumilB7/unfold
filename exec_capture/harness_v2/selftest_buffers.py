"""F05 controls: buffers and integer tables are judged by MEANING (evidence), never by dtype or by the modes tried.
(FULL-repair F05 / EXECUTION_PLAN P2b.3; R-BUF 2026-10-13 retained; dossier model-benchmark/_reruns/p2b3_f05.)

Each control is a tiny randomly-initialised library model saved as a local checkpoint (no downloads), graded through
the worker's own T1/G1/G3 code path: libload.Authority's library load + provenance + per-key integer-table evidence,
the zero-storage DagRecorder execution, libload.g1_assess with the buffer inventory, and the worker's unexpected-key
rule. The same file runs against HEAD (no F05 functions: it reproduces HEAD's worker logic) so every NEG control is
shown to FAIL there.

    python3 selftest_buffers.py [name-substring ...]          # exit 0 = every control as expected
Acceptance (F05): an optional inference mode that reads a buffer prevents an unsupported dead-state claim; a relevant
integer table cannot disappear; a proven counter / index does not become a false learned-weight failure."""
import os, sys, json, tempfile, shutil, collections, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch, transformers
from safetensors.torch import save_file
import libload, lowcost, common
from dag import DagRecorder, math_attention

SEL = sys.argv[1:]
F05 = hasattr(libload, "buffer_groups")          # HEAD has none of the F05 rules: its worker logic is reproduced below
bad = 0


def check(name, got, want):
    global bad
    ok = got == want
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (want {want!r})"), flush=True)


def save(d, name, model, edit=None):
    p = os.path.join(d, name); os.makedirs(p, exist_ok=True)
    model.config.save_pretrained(p)
    sd = {k: v.contiguous().clone() for k, v in model.state_dict().items()}
    if edit: sd = edit(sd)
    save_file(sd, os.path.join(p, "model.safetensors"))
    return p


LLAMA = dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
             num_key_value_heads=1, max_position_embeddings=32, tie_word_embeddings=False)


class OptIntTable(transformers.LlamaForCausalLM):
    """A persistent INTEGER table (shipped, loaded) read only in an optional documented mode (use_table=True)."""
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("table", torch.arange(8, dtype=torch.long) % config.vocab_size)
        self.post_init()
    def forward(self, input_ids=None, use_table=False, **kw):
        out = super().forward(input_ids=input_ids, **kw)
        if use_table:
            out.logits = out.logits + self.model.embed_tokens(self.table).sum()
        return out


class OptFloatTable(transformers.LlamaForCausalLM):
    """The P0.3 floating twin: a persistent FLOAT table read only in the optional mode."""
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("ftable", torch.ones(8))
        self.post_init()
    def forward(self, input_ids=None, use_table=False, **kw):
        out = super().forward(input_ids=input_ids, **kw)
        if use_table:
            out.logits = out.logits + self.ftable.sum()
        return out


class HashGate(torch.nn.Module):
    """A token -> expert routing table (DeepSeek-V4 hash-routing style): persistent integer state from the checkpoint."""
    def __init__(self, vocab):
        super().__init__()
        self.register_buffer("tid2eid", torch.zeros(vocab, 2, dtype=torch.long))


class SiblingGates(transformers.LlamaForCausalLM):
    """Two routing tables of the same kind; the forward reads only gate 0's (gate 1's is a real gap)."""
    def __init__(self, config):
        super().__init__(config)
        self.gates = torch.nn.ModuleList([HashGate(config.vocab_size) for _ in range(2)])
        self.post_init()
    def forward(self, input_ids=None, **kw):
        out = super().forward(input_ids=input_ids, **kw)
        out.logits = out.logits + self.model.embed_tokens(self.gates[0].tid2eid[input_ids]).sum()
        return out


class DeclaresGate(transformers.LlamaForCausalLM):
    """The library DECLARES a shipped routing table ignorable (class b: library-declared unused)."""
    _keys_to_ignore_on_load_unexpected = [r"gate\.tid2eid"]


def grade(path, cls, kw, train_kw=None):
    """The worker's T1 (library path) + G1/G3 path on one local checkpoint."""
    cfg = transformers.AutoConfig.from_pretrained(path)
    A = libload.Authority(path); A.mirror()
    T = {k: v for h in A.headers.values() for k, v in h.items()}
    aux = set(common.ckpt_split(T)[1])
    LIB = A.report(cls, aux, config=cfg)
    with lowcost.zero_storage():
        m = cls._from_config(cfg, attn_implementation="eager", dtype=torch.float32).eval()
    used, passes = set(), []
    for mode, kws in (("eval", kw), ("train", train_kw)):
        if kws is None:
            continue
        m.train(mode == "train")
        rec = DagRecorder(m, kws)
        try:
            with (torch.no_grad() if mode == "eval" else torch.enable_grad()), math_attention(), rec:
                m(**kws)
        finally:
            rec.remove_hooks()
        used |= set(rec.used_params) | set(rec.used_buffers)
        passes.append((set(rec.used_params) | set(rec.used_buffers), set(rec.called)))
    m.eval()
    sd = m.state_dict(keep_vars=True)
    alias = collections.defaultdict(list)
    for k, v in sd.items(): alias[id(v)].append(k)
    for names in alias.values():
        if any(n in used for n in names): used |= set(names)
    QW, POS = common.quantized_weight_keys(T), common.pos_lengths(cfg)
    model_buffers = common.buffer_names(m)
    srcs = LIB.get("reported_sources", {})
    _src = lambda k: srcs.get(k) or [k]
    unknowns = []
    if F05:
        ev = LIB.get("int_table_evidence") or {}
        def sc(x):
            e = ev.get(x)
            return common.stored_constant(x, *T[x], quantized=QW, pos=POS, buffer_slot=e in ("loaded_buffer", "nonpersistent_slot"),
                                          declared=e == "declared_ignorable")
    else:                                         # HEAD worker_v2.py: _stored_constant(x) (any integer is a table)
        sc = lambda x: common.stored_constant(x, *T[x], quantized=QW, pos=POS)
    lib_unexp = [k for k in LIB["unexpected"] if not all(x in T and sc(x) for x in _src(k)) and k not in model_buffers]
    if F05:
        intu = [k for k in lib_unexp if all(x in T and common.is_int_table(T[x][1], x, QW) for x in _src(k))]
        lib_unexp = [k for k in lib_unexp if k not in set(intu)]
        if intu:
            unknowns.append(common.int_table_unknown(intu))
        bg = libload.buffer_groups(m)
    else:                                         # HEAD worker_v2.py:1134: FLOATING persistent buffers only
        pn = {n for n, _ in m.named_parameters(remove_duplicate=False)}
        bg = {k: (type(m.get_submodule(k.rpartition(".")[0]) if "." in k else m).__name__, k.rpartition(".")[2])
              for k, t in sd.items() if k not in pn and t.is_floating_point()}
    try:                                          # per-pass sibling evidence, as the worker passes it (F05)
        G = libload.g1_assess(LIB["provenance"], used, {k: list(t.shape) for k, t in sd.items()}, bg, passes)
    except TypeError:                             # HEAD: no per-pass argument
        G = libload.g1_assess(LIB["provenance"], used, {k: list(t.shape) for k, t in sd.items()}, bg)
    unknowns += G["unknowns"]
    loaded = libload.loaded_params({n for n, _ in m.named_parameters()}, LIB)
    unused = sorted(p for p in loaded if p not in used)
    v = "PARTIAL" if (G["fail"] or lib_unexp or LIB["mismatched"] or unused) else "FULL"
    if v == "FULL" and unknowns:
        v = "FULL_UNVERIFIED"
    return v, {"fail": G["fail"], "unexpected": lib_unexp, "unused": unused[:4],
               "unknowns": [u["what"][:90] for u in unknowns], "int_evidence": (LIB.get("int_table_evidence") or {})}


def build(d):
    torch.manual_seed(0)
    C = {}
    ids = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    cfg = transformers.LlamaConfig(**LLAMA)
    m = transformers.LlamaForCausalLM(cfg)
    L = transformers.LlamaForCausalLM
    BUF = "loaded buffers no recorded run read"          # (HEAD wording: "floating buffers no recorded run read")
    INTU = "shipped integer / bool tables the model does not load"
    # ---- acceptance 1: an optional inference mode that reads a buffer prevents an unsupported dead-state claim ----
    oi = OptIntTable(cfg)
    C["N1 integer table read only in an optional mode, mode not tried -> unknown (HEAD: escapes, FULL)"] = (
        save(d, "N1", oi), OptIntTable, ids, None, "FULL_UNVERIFIED+" + BUF)
    C["V1 the same table with its optional mode tried -> FULL"] = (save(d, "V1", oi), OptIntTable, dict(ids, use_table=True), None, "FULL")
    of = OptFloatTable(cfg)
    C["V2 floating twin (P0.3 / R-BUF retained): optional mode not tried -> unknown"] = (
        save(d, "V2", of), OptFloatTable, ids, None, "FULL_UNVERIFIED+buffers no recorded run read")
    sg = SiblingGates(cfg)
    C["N3 routing table unread while a same-kind sibling is read -> real gap (HEAD: escapes, FULL)"] = (
        save(d, "N3", sg), SiblingGates, ids, None, "PARTIAL:buffers_unread_while_siblings_read")
    # ---- acceptance 2: a relevant integer table cannot disappear ----
    tid = torch.randint(0, 2, (64, 2), dtype=torch.long)
    C["N2 shipped routing table the class does not build, undeclared -> named unknown (HEAD: hidden, FULL)"] = (
        save(d, "N2", m, lambda sd: dict(sd, **{"model.layers.0.mlp.gate.tid2eid": tid})), L, ids, None, "FULL_UNVERIFIED+" + INTU)
    C["N4 U8 1x1x7x7 table, 7 not a position length, undeclared -> named unknown (HEAD: hidden, FULL)"] = (
        save(d, "N4", m, lambda sd: dict(sd, **{"model.layers.0.self_attn.bias": torch.ones(1, 1, 7, 7, dtype=torch.uint8)})),
        L, ids, None, "FULL_UNVERIFIED+" + INTU)
    C["V3 the same routing table, library DECLARES it ignorable (class b) -> FULL"] = (
        save(d, "V3", DeclaresGate(cfg), lambda sd: dict(sd, **{"model.layers.0.mlp.gate.tid2eid": tid})), DeclaresGate, ids, None, "FULL")
    C["V4 U8 1x1xNxN causal mask, N a position length (C7 shape rule) -> FULL"] = (
        save(d, "V4", m, lambda sd: dict(sd, **{"model.layers.0.self_attn.bias": torch.tril(torch.ones(1, 1, 32, 32, dtype=torch.uint8))})),
        L, ids, None, "FULL")
    nc = transformers.GPTNeoConfig(vocab_size=64, hidden_size=16, num_layers=2, num_heads=2, max_position_embeddings=32,
                                   attention_types=[[["global", "local"], 1]], window_size=4, intermediate_size=32,
                                   tie_word_embeddings=False)
    neo = transformers.GPTNeoForCausalLM(nc)
    def neo_masks(sd):                        # old GPT-Neo checkpoints ship the masks at the NON-PERSISTENT slot
        sd = dict(sd)
        for i in range(2):
            sd[f"transformer.h.{i}.attn.attention.bias"] = torch.tril(torch.ones(1, 1, 32, 32, dtype=torch.bool))
        return sd
    C["V5 bool masks at the library's non-persistent slot (library declares them derived) -> FULL"] = (
        save(d, "V5", neo, neo_masks), transformers.GPTNeoForCausalLM, ids, None, "FULL")
    # ---- acceptance 3: a proven counter / index does not become a false learned-weight failure ----
    r = transformers.ResNetForImageClassification(transformers.ResNetConfig(embedding_size=8, hidden_sizes=[8, 16], depths=[1, 1], num_labels=3))
    px = {"pixel_values": torch.randn(1, 3, 32, 32)}
    trn = dict(px, labels=torch.tensor([0]))
    C["V6 BN counters shipped; the worker's training pass increments them -> FULL"] = (
        save(d, "V6", r), transformers.ResNetForImageClassification, px, trn, "FULL")
    C["V7 BN counters shipped, never observed (no training pass) -> unknown, never PARTIAL (HEAD: FULL)"] = (
        save(d, "V7", r), transformers.ResNetForImageClassification, px, None, "FULL_UNVERIFIED+" + BUF)
    C["V8 a 0-d integer counter shipped where the class has no slot (FrozenBN style) -> unknown, never PARTIAL (HEAD: FULL)"] = (
        save(d, "V8", r, lambda sd: dict(sd, **{"resnet.embedder.extra_norm.num_batches_tracked": torch.tensor(7)})),
        transformers.ResNetForImageClassification, px, trn, "FULL_UNVERIFIED+" + INTU)
    C["V9 BN counters NOT shipped (library initialises them deterministically) -> FULL"] = (
        save(d, "V9", r, lambda sd: {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}),
        transformers.ResNetForImageClassification, px, None, "FULL")
    C["V10 clean llama (no buffers, no integer tables) -> FULL"] = (save(d, "V10", m), L, ids, None, "FULL")
    return C


def run_models():
    global bad
    d = tempfile.mkdtemp(prefix="selftest_buffers_")
    try:
        for name, (path, cls, kw, train_kw, want) in build(d).items():
            if SEL and not any(s.lower() in name.lower() for s in SEL): continue
            try:
                got, why = grade(path, cls, kw, train_kw)
            except Exception as e:
                import traceback
                got, why = "ERROR", f"{type(e).__name__}: {str(e)[:200]} {traceback.format_exc()[-300:]}"
            want_v, _, need = want.partition("+")
            want_v, _, want_why = want_v.partition(":")
            w = why if isinstance(why, dict) else {}
            ok = (got == want_v and (not want_why or want_why in (w.get("fail") or {}))
                  and (not need or any(need in u for u in w.get("unknowns") or [])))
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})")
                  + f"  {json.dumps({k: v for k, v in why.items() if v}, default=str)[:300] if isinstance(why, dict) else why}", flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_rules():
    """The shared rules themselves (the worker, the diffusion worker and class selection all call these)."""
    qw = set()
    def sc(k, s, d, **ev):
        return common.stored_constant(k, s, d, quantized=qw, pos={32}, **ev)
    check("R1 an undeclared I64 routing table with no slot is NOT a stored table (HEAD: hidden by dtype)",
          sc("layers.0.ffn.gate.tid2eid", (129280, 6), "I64"), False)
    check("R2 the same table the library declares ignorable is stored", sc("layers.0.ffn.gate.tid2eid", (129280, 6), "I64", declared=True), True)
    check("R3 an integer table landing in a buffer slot is stored state", sc("blocks.0.relative_position_index", (49, 49), "I64", buffer_slot=True), True)
    check("R4 a U8 1x1xNxN mask, N a position length, is stored (C7 shape rule)", sc("h.0.attn.bias", (1, 1, 32, 32), "U8"), True)
    check("R5 a U8 1x1x7x7 table, 7 not a position length, is NOT stored (HEAD: hidden by dtype)", sc("h.0.attn.bias", (1, 1, 7, 7), "U8"), False)
    check("R6 a 0-d I64 counter with no slot and no declaration is NOT stored (HEAD: hidden by dtype)", sc("x.num_batches_tracked", (), "I64"), False)
    raw = {"l.0.w1.weight": ((32, 16), "I8"), "l.0.w1.weight_scale": ((32, 1), "F32")}
    q = common.quantized_weight_keys(raw)
    check("R7 I8 codes with a scale sibling stay a learned weight (B4 unchanged)",
          common.stored_constant("l.0.w1.weight", (32, 16), "I8", quantized=q, declared=True), False)
    check("R8 a learned 0-d float stays strict-unless-declared (C7 unchanged)", sc("text_model.logit_scale", (), "F32"), False)
    gaps = getattr(libload, "buffer_gaps", None)
    bg = {"g.0.tid2eid": ("HashGate", "tid2eid"), "g.1.tid2eid": ("HashGate", "tid2eid"), "bn.num_batches_tracked": ("BatchNorm2d", "num_batches_tracked")}
    if gaps:
        f_, u_ = gaps(bg, set(bg), {"g.0.tid2eid"})
    else:                                          # HEAD: integer buffers never reach the rule (floating filter)
        f_, u_ = [], []
    check("R9 integer sibling gap is a failure, a never-read counter an unknown (any dtype)", (f_, u_),
          (["g.1.tid2eid"], ["bn.num_batches_tracked"]))
    # per-pass sibling evidence (F05 residual found by probes/counter_sibling_residual.py): eval runs bn1 + bn2 and reads
    # no counter; the TRAINING pass runs only bn1 and increments its counter
    bgc = {"bn1.num_batches_tracked": ("BatchNorm2d", "num_batches_tracked"), "bn2.num_batches_tracked": ("BatchNorm2d", "num_batches_tracked")}
    ev_pass = ({"bn1.weight", "bn2.weight", "bn1.running_mean", "bn2.running_mean"}, {"bn1", "bn2"})
    tr_pass = ({"bn1.weight", "bn1.num_batches_tracked"}, {"bn1"})
    rd = ev_pass[0] | tr_pass[0]
    if gaps:
        try:
            got = gaps(bgc, set(bgc), rd, [ev_pass, tr_pass])
        except TypeError:                          # (the first prototype had no per-pass evidence)
            got = gaps(bgc, set(bgc), rd)
    else:
        got = ([], [])
    check("R11 a counter only the training pass touches, in a module that pass skips (inference runs it): unknown, "
          "never a failure", got, ([], ["bn2.num_batches_tracked"]))
    bge = {f"q.layers.{i}.codebook.embed": ("Codebook", "embed") for i in range(2)}
    p1 = ({"q.layers.0.codebook.embed"}, {"q", "q.layers.0"})
    check("R12 a codebook whose module never ran while its sibling's was read stays a failure (EnCodec C11 shape)",
          gaps(bge, set(bge), p1[0], [p1]) if gaps else None, (["q.layers.1.codebook.embed"], []))
    p2 = ({"q.layers.0.codebook.embed"}, {"q", "q.layers.0", "q.layers.1.codebook"})
    check("R13 a module that ran in the same pass where its sibling read the buffer, without reading its own: failure",
          gaps(bge, set(bge), p2[0], [p2]) if gaps else None, (["q.layers.1.codebook.embed"], []))
    split = getattr(libload, "matcher_buffer_split", None)
    # (names outside witness_io.AUX: HEAD's matcher drops num_batches_tracked / position_ids BY NAME before T1)
    unused = [("gate.step_count", 1), ("q.codebook.embed_avg", 32), ("head.weight", 12)]
    matched = {c: c for c, _ in unused}
    bgm = {"gate.step_count": ("Gate", "step_count"), "q.codebook.embed_avg": ("Codebook", "embed_avg")}
    if split:
        keep, unk = split(unused, matched, bgm, set())
    else:                                          # HEAD matcher path: every unread matched tensor is an unused weight
        keep, unk = unused, []
    check("R10 matcher path: unread loaded buffers (integer counter, EMA float) are R-BUF unknowns, not unread learned "
          "weights (HEAD: T1 failure)", (keep, sorted(unk)), ([("head.weight", 12)], ["gate.step_count", "q.codebook.embed_avg"]))


def run_wiring():
    """The worker call sites really use the F05 rules (a reverted site fails here)."""
    W2 = open(os.path.join(HERE, "worker_v2.py")).read()
    WD = open(os.path.join(HERE, "worker_v2_diffusion.py")).read()
    LL = open(os.path.join(HERE, "libload.py")).read()
    wiring = {
        "worker_v2: G3 buffer inventory is libload.buffer_groups (no floating filter)": "_bg = libload.buffer_groups(m)" in W2
            and "for k, t in _sd.items() if k not in _pn and t.is_floating_point()}" not in W2,
        "worker_v2: unexpected integer tables use the library's per-key evidence": "_stored_constant(x, LIB) for x in _src(k)" in W2,
        "worker_v2: class scoring uses the library's per-key evidence": "_stored_constant(x, rep) for x in ss" in W2,
        "worker_v2: unexplained integer tables become the G1 unknown": "common.int_table_unknown(INT_UNEXPLAINED" in W2
            and W2.find("common.int_table_unknown(INT_UNEXPLAINED") < W2.find('if R["verdict"] in ("FULL", "FULL_LEFTOVERS") and UNKNOWNS:'),
        "worker_v2: matcher path judges buffers by THE buffer rule": "_ll.matcher_buffer_split(unused, matched" in W2,
        "worker_v2_diffusion: masks take slot evidence; unexplained tables -> unknown": "buffer_slot=(c in _bnD" in WD and "common.int_table_unknown(" in WD,
        "libload: g1_assess uses buffer_gaps": "gap_fail, gap_unknown = buffer_gaps(" in LL,
        "worker_v2: per-pass reads reach g1_assess and the matcher split": W2.count("PASS_READS.append(libload_pass(rec))") == 8
            and "_bg, PASS_READS)" in W2 and "used, PASS_READS)" in W2,
        "libload: the load report carries int_table_evidence": '"int_table_evidence": int_ev' in LL,
        "worker: F05-3 the training pass retries as a batch of 2 only after every batch-1 candidate failed":
            "for _bs in (1, 2):" in W2 and "if done_train: break" in W2 and "def _double(v):" in W2,
    }
    for k, v in wiring.items():
        check(f"W {k}", v, True)


def run_worker_lines():
    """The worker's OWN F05 lines, executed (cross-exam S1: a mirror of the logic cannot catch a change to the worker)."""
    import textwrap, ast, collections, transformers, common
    from common import ckpt_split, match, library_rename, split_library_ignored, numel, buffer_names
    W2 = open(os.path.join(HERE, "worker_v2.py")).read(); WD = open(os.path.join(HERE, "worker_v2_diffusion.py")).read()
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                   num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32)
    m = transformers.LlamaForCausalLM(cfg)
    T = {k: (tuple(v.shape), "F32") for k, v in m.state_dict().items()}
    key = "model.layers.0.mlp.gate.tid2eid"
    T[key] = ((64, 2), "I64")                                  # shipped, no slot in the class, undeclared
    used_params = {n for n, _ in m.named_parameters()}
    env = dict(globals(), torch=torch, transformers=transformers, common=common, collections=collections, ckpt_split=ckpt_split,
               match=match, library_rename=library_rename, split_library_ignored=split_library_ignored, numel=numel,
               buffer_names=buffer_names, m=m, T=T, cfg=cfg, used_params=set(used_params), used_buffers=set(),
               PASS_READS=[(set(used_params), {n for n, _ in m.named_modules()})],
               QW=common.quantized_weight_keys(T), _POS=common.pos_lengths(cfg))
    if F05 and "INT_UNEXPLAINED" in W2:
        a = W2.index("    W, A, packed = ckpt_split(T)")
        b = W2.index("\n", W2.index("    #      (vLLM executed these tensors under the recorder"))
        g = dict(env); exec(textwrap.dedent(W2[a:b].rsplit("\n", 2)[0]), g)
        check("B1 matcher path (worker's own lines): an unexplained integer table is named, never an unused learned weight",
              (g.get("INT_UNEXPLAINED"), [c for c, _ in g["unused"]], g["un_c"]), ([key], [], []))
        a = WD.index("        W, A, packed = ckpt_split(T)"); b = WD.index('        checks["T1_weights"] = {"pass": not unused')
        g = dict(env, used=set(used_params), UNKNOWNS=[], cn="transformer", POS=env["_POS"]); exec(textwrap.dedent(WD[a:b]), g)
        check("B1 diffusion component (its own lines): the table is a named unknown, never an unused learned weight",
              ([u["items"] for u in g["UNKNOWNS"]], [c for c, _ in g["unused"]]), ([[f"transformer:{key}"]], []))
    else:
        check("B1 worker lines with the F05 rule", False, True)
    T_ = ast.parse(W2)
    fn = lambda name: next((n for n in T_.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    sc, lp = fn("_stored_constant"), fn("libload_pass")
    if sc is not None and lp is not None and F05:
        T2 = {"x.tid2eid": ((64, 2), "I64"), "text_model.logit_scale": ((), "F32"), "bn.idx": ((8,), "I64")}
        g = {"common": common, "T": T2, "QW": set(), "_POS": set()}
        exec(compile(ast.Module(body=[sc], type_ignores=[]), "<_stored_constant>", "exec"), g)
        st = g["_stored_constant"]
        check("M6/M10 worker _stored_constant: an int table with NO evidence is not 'stored'; with the library's evidence it is; "
              "a learned 0-d float is never stored",
              (st("x.tid2eid", {"int_table_evidence": {}}), st("bn.idx", {"int_table_evidence": {"bn.idx": "loaded_buffer"}}),
               st("text_model.logit_scale", {})), (False, True, False))
        g = {}; exec(compile(ast.Module(body=[lp], type_ignores=[]), "<libload_pass>", "exec"), g)
        R_ = type("R", (), {"used_params": {"a.w"}, "used_buffers": {"a.b"}, "called": {"a", "(root)"}})()
        check("M8 worker libload_pass: (tensors read, modules run) of a pass", g["libload_pass"](R_), ({"a.w", "a.b"}, {"a", "(root)"}))
    else:
        check("M6/M8 worker _stored_constant / libload_pass present", False, True)
    if hasattr(libload, "initialised_not_shipped"):
        nl = {"det": {"class": "INIT_DETERMINISTIC", "kind": "buffer"}, "rnd": {"class": "INIT_RANDOM", "kind": "param"},
              "np": {"class": "INIT_DETERMINISTIC", "kind": "buffer_nonpersistent"},
              "shipped": {"class": "INIT_DETERMINISTIC", "kind": "buffer", "set_by_loader": True}}
        check("M9/S2 F05-4 label: only persistent state the library fills itself (never random, derived or loader-set keys)",
              libload.initialised_not_shipped(nl), ["det"])
    else:
        check("M9 initialised_not_shipped present", False, True)
    if hasattr(common, "int_table_unknown"):
        u = common.int_table_unknown(["b.t", "a.t"])
        check("M11 the integer-table unknown names its keys", (u["predicate"], u["items"], u["count"]), ("G1", ["a.t", "b.t"], 2))
    if F05:
        check("W worker: initialised_not_shipped is the row's F05-4 field",
              '"initialised_by_library_not_shipped": libload.initialised_not_shipped(_prov.get("nonloaded")) or None' in W2, True)


if __name__ == "__main__":
    print(f"(F05 rules present: {F05})")
    if not SEL or "rules" in SEL:
        run_rules()
    if not SEL or "wiring" in SEL:
        run_wiring()
    if not SEL or "worker" in SEL:
        run_worker_lines()
    run_models()
    print(f"\n{bad} unexpected")
    sys.exit(1 if bad else 0)
