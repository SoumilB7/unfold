"""P4 functional oracle (Track D): is the DRAWING functionally right?

For one capture bundle (the recording harness v2 graded; the same object exec_to_ir draws from):
  1. page facts: exec_to_ir.recognize -> to_ir -> the ModelIR the HTML renders; every fact the reference uses is read
     back from that IR (layer specs, block descriptions, the EXECUTION EVIDENCE scorecard, notes). Nothing is read from
     the recognizer's private dict, so a fact the page does not state cannot help the reference.
  2. the REAL model: the executed library class (bundle["class"]) built from the bundle's config with random seeded
     weights (no weight download, no remote code), float64, eager attention. Layers are shrunk to `n_layers` (the page
     claims every layer runs layer 0's ops; that claim is the recognizer's, the oracle checks the layers it builds).
  3. an independent reference written here in plain torch ONLY from the page facts; it reads the real model's tensors
     through the page's parameter mapping (which tensor each drawn op reads: the evidence strings on the page for
     Q/K/V/gate/up/down/embedding; the recording's reader ops for o_proj / norms / biases / head).
  4. both run on the same random tokens (batch 2, positions 0..S-1, causal); compared stage by stage with the real
     model's own stage inputs (so a failure names the stage) and end to end.
Facts the page does NOT state but the reference needs are taken from the config and listed per row as `undrawn`
(norm eps; llama3 low/high frequency factors). Conventions the page implies but does not spell out are fixed here and
listed in CONVENTIONS (they are what a reader of the page would assume).

Verdicts: PASS | FAIL(stage: facts) | NOT_CHECKABLE(reason) | NOT_DRAWN(recognizer reason).
    python3 oracle.py <bundle.capture.json.gz> [--layers 2] [--seeds 0,1] [--seq 24] [--dtype float64]
"""
import os, re, sys, json, math, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RENDERER = "/Users/soumil/Code/Projects/Understand/llmvisualizer/model-benchmark/renderer_snapshot"
if RENDERER not in sys.path:
    sys.path.insert(0, RENDERER)
import torch
import capture, exec_to_ir

TOL = 1e-4          # max |real - ref| / max |ref| per stage. float64 everywhere, but the library computes RoPE tables, softmax
                    # and RMSNorm in fp32: measured floor <= 1.3e-6 per stage, 5.4e-6 end to end (SmolLM2, S=257)
# the eps seeds: every non-norm parameter x 1e-3 so each norm's input is small and its stated eps matters. Seed 2 also
# shrinks the embedding (the layer and final norms see a small residual stream); seed 3 keeps the embedding at full scale
# (projections of an O(1) stream are ~1e-3, so a QK-norm's mean square is comparable to its eps and the scores stay O(1))
# seed 4 shrinks ONLY the embedding (the residual stream) and keeps every projection at full scale: a post / sandwich
# output norm then sees a sublayer output whose mean square is comparable to eps, while its normalized output is O(1)
# TEMPORARY (Soumil's Q5 ruling, 2026-10-19): only the end-to-end composite of a run longer than a stated window is
# gated at TOL_LONG_E2E, with the measured cause stated on the row; every stage keeps TOL. Measured on Gemma-3-1b
# (6 layers, 528 tokens, 262k vocabulary): end to end 1.71e-4, every stage <= 2.1e-5. Diagnosis: the library's
# float32 RMSNorm cast is NOT the cause (patched to float64: 1.66e-4); the library's float32 RoPE tables ARE (patched
# to float64 tables: 9.2e-6, every stage <= 1.0e-6). Mirroring float32 tables in the reference only reached 1.27e-4
# (the library also builds inv_freq itself in float32), so the reference stays float64 and the margin is stated.
# The same cause at 257 tokens: end to end 1.06e-4 as is, 8.9e-6 with float64 tables -- so every run longer than
# LONG_S = 128 tokens is a "long run" for the end-to-end composite (stages keep TOL at every length).
LONG_S, TOL_LONG_E2E = 128, 3e-4
LONG_RUN_CAUSE = ("the library's float32 RoPE angle tables at long positions (Q5 diagnosis on Gemma-3-1b: as is 1.06e-4 "
                  "at 257 tokens / 1.71e-4 at 528; with float64 tables 8.9e-6 / 9.2e-6)")
EPS_SEED_SCALE, EPS_SEEDS = 1e-3, {2: "all", 3: "projections", 4: "embedding"}
CONVENTIONS = [
    "GQA: query head h reads K/V head h // (heads / kv_heads) (contiguous groups; the page says 'N Q per KV head')",
    "RoPE: rotate-half on the full head width, inv_freq[i] = theta^(-2i/head_dim), positions 0..S-1 (the page says "
    "'Rotate-half rotary embedding', 'frequency base theta')",
    "causal mask: query i sees keys 0..i; softmax over the key axis",
    "RMSNorm: x * rsqrt(mean(x^2, -1) + eps) * w;  LayerNorm: (x - mean) / sqrt(biased var + eps) * w + b",
    "projections: y = x @ W^T (+ b) with W stored [out, in] (the page's 'H -> out' widths)",
]
ACTS = {"silu": torch.nn.functional.silu, "gelu": lambda x: torch.nn.functional.gelu(x),
        "gelu_pytorch_tanh": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        "relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}


class NotCheckable(Exception):
    pass


# --------------------------------------------------------------------------------------------- 1. the page's facts
def page_facts(ir):
    """(P, issues): every fact the reference uses, read from the rendered IR -- the spec fields AND the display text the
    reader sees, which must agree (page_reader)."""
    import page_reader
    return page_reader.read_page(ir)


# ---------------------------------------------------------------------------------- 3a. the page's parameter mapping
def param_map(bundle, P):
    """Which checkpoint tensor each drawn op reads. Q/K/V/gate/up/down/embedding: the names the page's evidence states.
    The rest (output projection, norm weights / biases, projection biases, LM head): the tensor the recorded op in that
    drawn position read (the recording the drawing was made from)."""
    ev = P["src"]                                                    # the EXECUTION EVIDENCE card's source column
    mq = re.search(r"Q from (\S+) \[.*?K from (\S+), V from (\S+)$", ev.get("decoder.attention.num_heads", ""))
    me = re.search(r"embedding reads (\S+) \[", ev.get("model.vocab_size", ""))
    mg = re.match(r"(?:gated: (\S+) -> \S+ x )?(\S+) -> (\S+);", ev.get("decoder.ffn.intermediate_size", ""))
    if not (mq and me and mg):
        raise NotCheckable("tensor mapping: the page's evidence card does not name the projection tensors")
    wq, wk, wv = mq.groups(); wg, wu, wd = mg.groups(); emb = me.group(1)
    G = bundle["passes"][bundle["pass_order"][0]]["graph"]
    N, _ = capture.fold_groups(capture.nodes_of(G), G.get("f09_assembled"))
    pshape = {k: tuple(v) for k, v in bundle["model"]["param_shapes"].items()}
    R = exec_to_ir.Rec(N, pshape)
    readers = collections.defaultdict(list)
    for j in range(len(N)):
        for w in R.weights(j):
            readers[w].append(j)
    rd = lambda w: readers[w][0]
    rdm = lambda w: next(j for j in readers[w] if N[j].func in exec_to_ir.MATMULS)     # the matmul reading a matrix
    lp = re.match(r"^(.*?\.\d+)\.", wq).group(1)                     # layer 0's module prefix
    stack = lp.rsplit(".", 1)[0]
    q_op, k_op, v_op, u_op, d_op = rdm(wq), rdm(wk), rdm(wv), rdm(wu), rdm(wd)
    g_op = rdm(wg) if wg else None
    mlp_in = min(x for x in (g_op, u_op) if x is not None)
    sm = next(j for j in range(min(q_op, k_op), len(N)) if N[j].func in exec_to_ir.SOFTMAX)
    scores, _ = R.back(sm, lambda j: N[j].func in ("bmm", "matmul") and not R.weights(j))
    apply_v = next(j for j in range(sm + 1, len(N)) if N[j].func in ("bmm", "matmul") and not R.weights(j))
    o_op = next((j for j in range(apply_v + 1, mlp_in) if N[j].func in exec_to_ir.MATMULS and R.weights(j)
                 and len(pshape.get(R.weights(j)[0], ())) == 2), None)
    bias_of = lambda j: (R.weights(j)[1] if j is not None and len(R.weights(j)) > 1 else None)
    mp = re.search(r"embedding reads (\S+) \[", ev.get("model.position_embedding", ""))
    M = {"emb": emb, "q": wq, "k": wk, "v": wv, "gate": wg, "up": wu, "down": wd, "pos": mp.group(1) if mp else None,
         "o": R.weights(o_op)[0] if o_op is not None else None,
         "q_b": bias_of(q_op), "k_b": bias_of(k_op), "v_b": bias_of(v_op), "o_b": bias_of(o_op),
         "gate_b": bias_of(g_op), "up_b": bias_of(u_op), "down_b": bias_of(d_op)}
    used = {v for v in M.values() if v}
    q_anc = set(R.back(R.srcs(scores)[0], lambda j: False, 2000)[1])
    k_anc = set(R.back(R.srcs(scores)[1], lambda j: False, 2000)[1])
    last_layer_op = max(j for j, nd in enumerate(N) if re.match(rf"^{re.escape(stack)}\.\d+(\.|$)", nd.module))
    head = next(j for j in range(len(N) - 1, last_layer_op, -1) if N[j].func in exec_to_ir.MATMULS and R.weights(j))
    M["head"] = R.weights(head)[0]
    M["head_b"] = bias_of(head)
    # the norms between the attention output and the MLP input, and after the MLP output, are named by the page's
    # placement: pre -> [n2]; post (sublayer output) -> [no1] and [no2]; double (sandwich) -> [no1, n2] and [no2]
    mid_roles = {"pre": ["n2"], "post": ["no1"], "double": ["no1", "n2"]}.get(P.get("placement"), ["n2"])
    mid_seen = []
    for w in sorted(readers, key=rd):                                 # 1-D tensors by where their reader sits
        if w in used or len(pshape.get(w, ())) != 1: continue
        r = rd(w)
        ws = [x for x in R.weights(r) if len(pshape.get(x, ())) == 1]
        # a bias is ADDED to a hidden value; a weight is multiplied in (directly, or as 1 + w computed from it alone)
        hidden_add = N[r].func == "add" and any(e[0] == "op" for e in N[r].ins)
        slot = "b" if hidden_add or ws.index(w) > 0 else "w"
        if w.startswith(lp + "."):
            if r < min(q_op, k_op, v_op): role = "n1"
            elif r < sm and r in q_anc: role = "qn"
            elif r < sm and r in k_anc: role = "kn"
            elif (o_op if o_op is not None else apply_v) < r < mlp_in:
                if slot == "w": mid_seen.append(w)
                k_ = len(mid_seen) - 1 if slot == "w" else max(len(mid_seen) - 1, 0)
                if k_ >= len(mid_roles): continue
                role = mid_roles[k_]
            elif r > d_op: role = "no2"
            else: continue
        elif last_layer_op < r < head: role = "nf"
        else: continue
        M.setdefault(f"{role}_{slot}", w)
    M["_layer_prefix"], M["_stack"] = lp, stack
    return M


# ------------------------------------------------------------------------------------------ 2. the real model
def build_real(bundle, n_layers, dtype=torch.float64):
    import transformers
    cls = getattr(transformers, bundle["class"], None)
    if cls is None:
        raise NotCheckable(f"executed class {bundle['class']} is not in the installed transformers")
    conf = cls.config_class.from_dict(dict(bundle["config"]))
    n0 = getattr(conf, "num_hidden_layers", None)
    if n0 and n0 > n_layers:
        conf.num_hidden_layers = n_layers
        if isinstance(getattr(conf, "layer_types", None), list): conf.layer_types = conf.layer_types[:n_layers]
    return cls._from_config(conf, attn_implementation="eager", dtype=dtype).eval()


def config_layers(bundle):
    """The layer count of the model the library builds from this config (through the config class's own names)."""
    import transformers
    cls = getattr(transformers, bundle["class"], None)
    return getattr(cls.config_class.from_dict(dict(bundle["config"])), "num_hidden_layers", None) if cls else None


def randomize(model, M, seed, small=1.0, mode="all", emb_div=1.0):
    """Seeded random values for every parameter (asymmetric: norm weights ~ 1 +- 0.3, biases ~ N(0, 0.3), matrices
    scaled by 1/sqrt(fan_in), embedding N(0, 1)). Shared tensors get one value. `small` < 1 scales every non-norm
    parameter down, so every norm (layer, final, QK) sees small inputs and its stated eps matters (the eps seed)."""
    g = torch.Generator().manual_seed(seed)
    norm_w = {v for k, v in M.items() if not k.startswith("_") and v and k.endswith("_w")}
    seen = set()
    named = dict(model.named_parameters(remove_duplicate=False))
    emb_id = id(named[M["emb"]]) if M["emb"] in named else None      # by identity: a tied head is the same tensor
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(remove_duplicate=False)):
            if id(p) in seen: continue
            seen.add(id(p))
            r = torch.randn(p.shape, generator=g, dtype=torch.float64)
            if p.dim() >= 2:
                # the eps seeds also divide by the page's embedding scale (Gemma x sqrt(H)), so the stream stays small
                s_emb = small / emb_div if mode in ("all", "embedding") else 1.0
                s_mat = small if mode in ("all", "projections") else 1.0
                r = r * s_emb if id(p) == emb_id else r / math.sqrt(p.shape[-1]) * s_mat
            else:
                r = 1 + 0.3 * r if any(_same_role(name, w) for w in norm_w) else \
                    0.3 * r * (small if mode in ("all", "projections") else 1.0)
            p.copy_(r.to(p.dtype))


def _strip(n):
    return re.sub(r"\.\d+\.", ".#.", n)


def _same_role(a, b):
    return _strip(a) == _strip(b)


# ------------------------------------------------------------------------------------- 3b. the reference (page only)
class Reference:
    """The function the page describes, in plain torch. P = page facts, W(role, layer) -> tensor."""

    def __init__(self, P, W, eps, l3_extra):
        self.P, self.W, self.eps, self.l3_extra = P, W, eps, l3_extra

    def norm(self, kind, x, w, b, eps, wform="w"):
        if wform == "none": w = None                              # the page's stated weight form decides
        elif w is not None and wform == "1+w": w = 1 + w
        if kind == "rmsnorm":
            if b is not None: raise NotCheckable("an RMSNorm with a bias has no drawn form")
            y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            return y * w if w is not None else y
        if kind == "layernorm":
            mu = x.mean(-1, keepdim=True); var = (x - mu).pow(2).mean(-1, keepdim=True)
            y = (x - mu) / torch.sqrt(var + eps)
            y = y * w if w is not None else y
            return y + b if b is not None else y
        raise NotCheckable(f"norm kind {kind!r} has no reference")

    def lin(self, x, w, b=None):
        """A layer projection as the page states its weight layout: x @ W^T ([out, in]) or x @ W ([in, out])."""
        y = x @ (w if self.P.get("w_layout") == "in_out" else w.T)
        return y + b if b is not None else y

    def qkv(self, h, i):
        """Q, K, V widths [B, S, heads x hd] from separate projections or the page's fused split."""
        P, W = self.P, self.W
        if not P.get("fused"):
            return tuple(self.lin(h, W(r, i), W(r + "_b", i) if P["qkv_bias"] else None) for r in ("q", "k", "v"))
        y = self.lin(h, W("q", i), W("q_b", i) if P["qkv_bias"] else None)
        lay, a, b, c = P["qkv_split"]
        B, S, _ = y.shape
        if lay == "qkv_blocks":
            return y[..., :a], y[..., a:a + b], y[..., a + b:a + b + c]
        G, n, d = a, b, c                                  # per K/V group: [q x G | k | v], head dim d
        g = y.reshape(B, S, n, (G + 2) * d)
        return (g[..., :G * d].reshape(B, S, n * G * d), g[..., G * d:(G + 1) * d].reshape(B, S, n * d),
                g[..., (G + 1) * d:].reshape(B, S, n * d))

    def kind(self, i):
        """The page's kind of layer i: its mask, window and rotation base (one kind unless the page states a pattern)."""
        P = self.P
        if P.get("kinds"): return P["kinds"][P["kind_of_layer"][i]]
        return {"mask": P["mask"], "window": P.get("window"), "theta": P["theta"]}

    def inv_freq(self, theta=None):
        P = self.P; d = P.get("rot_dim") or P["head_dim"]
        theta = P["theta"] if theta is None else theta
        f = [theta ** (-(2 * i) / d) for i in range(d // 2)]
        if P["llama3"]:
            fac, ctx = P["llama3"]["factor"], P["llama3"]["original_max_position_embeddings"]
            lo, hi = self.l3_extra["low_freq_factor"], self.l3_extra["high_freq_factor"]
            lo_wl, hi_wl = ctx / lo, ctx / hi
            def sc(f_):
                wl = 2 * math.pi / f_
                if wl < hi_wl: return f_
                if wl > lo_wl: return f_ / fac
                s = (ctx / wl - lo) / (hi - lo)
                return (1 - s) * f_ / fac + s * f_
            f = [sc(x) for x in f]
        return torch.tensor(f, dtype=torch.float64)

    def rope(self, x, S, theta=None):
        r = self.P.get("rot_dim")
        if r:                                              # partial: rotate the first r dims, pass the rest through
            return torch.cat([self._rope_full(x[..., :r], S, theta), x[..., r:]], -1)
        return self._rope_full(x, S, theta)

    def _rope_full(self, x, S, theta=None):
        pos = torch.arange(S, dtype=torch.float64)
        ang = pos[:, None] * self.inv_freq(theta)[None, :]
        ang = torch.cat([ang, ang], -1)                               # [S, d]
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        h = x.shape[-1] // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], -1)
        return x * cos + rot * sin

    def qk_norm(self, x, role, i):
        P = self.P
        return self.norm(P["qk_norm_kind"], x, self.W(role + "_w", i), self.W(role + "_b", i), P["qk_eps"], P["qk_weight"])

    def attention_probs(self, h, i):
        """(probabilities [B, nh, S, S], v [B, nkv, S, hd]) from the attention input h."""
        P, W = self.P, self.W
        B, S, _ = h.shape
        nh, nkv, hd = P["num_heads"], P["num_kv"], P["head_dim"]
        q, k, v = self.qkv(h, i)
        per_head = P["qk_norm"] and P["qk_norm_extent"] == hd
        if P["qk_norm"] and not per_head:                             # a norm over the whole projected width
            q, k = self.qk_norm(q, "qn", i), self.qk_norm(k, "kn", i)
        q = q.view(B, S, nh, hd).transpose(1, 2)
        k = k.view(B, S, nkv, hd).transpose(1, 2)
        v = v.view(B, S, nkv, hd).transpose(1, 2)
        if per_head and P["qk_norm_before_rope"]:
            q, k = self.qk_norm(q, "qn", i), self.qk_norm(k, "kn", i)
        if P["rope"]:
            if P["rope_form"] != "rotate_half" or not self.kind(i)["theta"]:
                raise NotCheckable(f"the page's rotation ({P['rope_form']}, theta {P['theta']}) has no reference")
            q, k = self.rope(q, S, self.kind(i)["theta"]), self.rope(k, S, self.kind(i)["theta"])
        if per_head and P["qk_norm_before_rope"] is False:
            q, k = self.qk_norm(q, "qn", i), self.qk_norm(k, "kn", i)
        g = nh // nkv
        kr = k[:, self._group_index()]                                # [B, nh, S, hd]
        s = q @ kr.transpose(-1, -2)
        if P["scale"] is not None: s = s * P["scale"]
        kd = self.kind(i)
        ii = torch.arange(S)[:, None]; jj = torch.arange(S)[None, :]
        if kd["mask"] == "sliding" and kd["window"]:        # causal band of the stated width
            allowed = (jj <= ii) & (jj > ii - kd["window"])
        elif kd["mask"] == "causal":
            allowed = jj <= ii
        else:
            raise NotCheckable(f"mask {P['mask']!r} has no reference")
        c_ = P.get("attn_softcap")                            # the page's attention-logit soft-cap, in its stated order
        if c_ and P.get("attn_softcap_order") == "before":
            s = torch.tanh(s / c_) * c_
        s = s.masked_fill(~allowed, float("-inf"))
        if c_ and P.get("attn_softcap_order") == "after":     # a masked score becomes -c, not -inf
            s = torch.tanh(s / c_) * c_
        elif c_ and P.get("attn_softcap_order") != "before":
            raise NotCheckable(f"soft-cap order {P.get('attn_softcap_order')!r} has no reference")
        return torch.softmax(s, -1), v

    def _group_index(self):
        P = self.P; nh, nkv = P["num_heads"], P["num_kv"]
        g = nh // nkv
        return torch.tensor([h // g for h in range(nh)]) if P.get("gqa_grouping", "contiguous") == "contiguous" else \
            torch.tensor([h % nkv for h in range(nh)])

    def attention_apply(self, probs, v, i):
        P, W = self.P, self.W
        B, nh, S, _ = probs.shape
        o = probs @ v[:, self._group_index()]
        o = o.transpose(1, 2).reshape(B, S, nh * P["head_dim"])
        if P["o_proj"]:
            o = self.lin(o, W("o", i), W("o_b", i) if P["o_bias"] else None)
        return o

    def attention(self, h, i):
        p, v = self.attention_probs(h, i)
        return self.attention_apply(p, v, i)

    def mlp(self, h, i):
        P, W = self.P, self.W
        act = ACTS.get(P["act"])
        if act is None: raise NotCheckable(f"activation {P['act']!r} has no reference")
        bb = lambda r: W(r + "_b", i) if P["ffn_bias"] else None
        if P["gated"] and P.get("gate_up"):                  # one fused projection, split as the page states
            y = self.lin(h, W("gate", i), bb("gate")); I_ = P["gate_up"][1]
            g_, u_ = (y[..., :I_], y[..., I_:2 * I_]) if P["gate_up"][0] == "gate_up" else (y[..., I_:2 * I_], y[..., :I_])
            z = act(g_) * u_
        elif P["gated"]:
            z = act(self.lin(h, W("gate", i), bb("gate"))) * self.lin(h, W("up", i), bb("up"))
        else:
            z = act(self.lin(h, W("up", i), bb("up")))
        return self.lin(z, W("down", i), bb("down"))

    def _ln(self, r, x, i):
        return self.norm(self.P["norm"], x, self.W(r + "_w", i), self.W(r + "_b", i), self.P["norm_eps"], self.P["norm_weight"])
    def n1(self, x, i): return self._ln("n1", x, i)
    def n2(self, x, i): return self._ln("n2", x, i)
    def no1(self, x, i): return self._ln("no1", x, i) if self.P["placement"] in ("post", "double") else x
    def no2(self, x, i): return self._ln("no2", x, i) if self.P["placement"] in ("post", "double") else x

    # stage predictions from the REAL model's stage inputs (so a mismatch names the stage)
    # placements (IR meaning): pre r + sub(norm(h)); post r + norm(sub(h)); double r + norm_o(sub(norm_i(h)))
    def attn_input(self, x, i):
        return self.n1(x, i) if self.P["placement"] in ("pre", "double") else x

    def mlp_input(self, x, a, i):
        if self.P["residual"] == "parallel":              # both branches read the layer input
            return self.n2(x, i) if self.P["parallel_norms"] == 2 else self.n1(x, i)
        h = x + self.no1(a, i)
        return self.n2(h, i) if self.P["placement"] in ("pre", "double") else h

    def layer_output(self, x, a, m, i):
        if self.P["residual"] == "parallel": return x + a + m
        return x + self.no1(a, i) + self.no2(m, i)

    def layer(self, x, i):
        if self.P["residual"] not in ("sequential", "parallel") or (
                self.P["residual"] == "parallel" and self.P["parallel_norms"] not in (1, 2)) or \
                self.P["placement"] not in ("pre", "post", "double"):
            raise NotCheckable(f"residual topology {self.P['residual']!r} has no reference")
        a = self.attention(self.attn_input(x, i), i)
        return self.layer_output(x, a, self.mlp(self.mlp_input(x, a, i), i), i)

    def embed(self, ids):
        x = self.W("emb", 0)[ids]
        if self.P.get("embed_scale") is not None: x = x * self.P["embed_scale"]      # the page's embedding scale
        if self.P.get("pos_emb"):                          # learned absolute positions 0..S-1, added
            x = x + self.W("pos", 0)[torch.arange(ids.shape[-1])]
        return x

    def final(self, x):
        P = self.P
        h = self.norm(P["final_norm"], x, self.W("nf_w", 0), self.W("nf_b", 0), P["final_eps"], P["final_weight"])
        head = self.W("emb", 0) if P["tied"] in (True, None) else self.W("head", 0)
        y = h @ head.T
        if P.get("head_bias"): y = y + self.W("head_b", 0)
        if P.get("final_softcap"):                            # the page's final-logit soft-cap
            y = torch.tanh(y / P["final_softcap"]) * P["final_softcap"]
        return y

    def model(self, ids, n_layers):
        x = self.embed(ids)
        for i in range(n_layers): x = self.layer(x, i)
        return self.final(x)


# ------------------------------------------------------------------------------------------ 4. compare
STAGE_FACTS = {
    "embedding": "token embedding (table lookup; no scale / position add drawn)",
    "attention input": "norm placement + attention-input norm kind",
    "attention probabilities": "heads / K-V heads / head_dim / GQA grouping / QK-norm / RoPE (form, theta) / score scale / mask",
    "attention output": "value path / GQA grouping of V / output projection (+bias)",
    "MLP input": "first residual add + MLP-input norm (placement, kind)",
    "MLP output": "gated / activation / FFN width / FFN bias",
    "layer output": "residual wiring (second add) / placement",
    "final norm + head": "final norm kind / tied head",
    "end to end": "every drawn fact composed",
    "end to end (long run)": "every drawn fact composed, on the run longer than a stated window",
}


def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def _owner(model, tensor_name):
    return model.get_submodule(tensor_name.rsplit(".", 1)[0]) if "." in tensor_name else model


def _common_module(model, names):
    parts = [n.split(".")[:-1] for n in names if n]
    pre = []
    for seg in zip(*parts):
        if len(set(seg)) != 1: break
        pre.append(seg[0])
    return ".".join(pre)


def rel(a, b):
    a, b = a.double(), b.double()
    if a.shape != b.shape: return float("inf")
    den = b.abs().max().item() or 1.0
    return (a - b).abs().max().item() / den


def check(model, P, M, eps, l3_extra, n_layers, seeds=(0, 1), S=24, B=2, randomize_weights=True, tol=TOL):
    """Run the real model and the page reference; per-stage relative max error, worst over seeds."""
    names = dict(model.named_parameters(remove_duplicate=False))
    lp, stack = M["_layer_prefix"], M["_stack"]
    def W(role, i):
        n = M.get(role)
        if not n: return None
        n_i = re.sub(rf"^{re.escape(lp)}\.", f"{stack}.{i}.", n) if n.startswith(lp + ".") else n
        if n_i not in names:
            raise NotCheckable(f"tensor {n_i} (page role {role}) is not in the real model built from the config")
        return names[n_i]
    attn_mod = _common_module(model, [M["q"], M["k"], M["v"], M.get("o")])
    mlp_mod = _common_module(model, [M.get("gate"), M["up"], M["down"]])
    attn_rel, mlp_rel = attn_mod[len(lp) + 1:], mlp_mod[len(lp) + 1:]
    worst = collections.defaultdict(float)
    errors = {}
    ref = Reference(P, W, eps, l3_extra)
    for seed in seeds:
        if randomize_weights:
            randomize(model, M, 1000 + seed, small=EPS_SEED_SCALE if seed in EPS_SEEDS else 1.0,
                      mode=EPS_SEEDS.get(seed, "all"), emb_div=abs(P.get("embed_scale") or 1.0))
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, min(P["vocab"], W("emb", 0).shape[0]), (B, S), generator=g)
        cap = collections.defaultdict(dict)
        hooks = []
        for i in range(n_layers):
            lay = model.get_submodule(f"{stack}.{i}")
            am, mm = lay.get_submodule(attn_rel), lay.get_submodule(mlp_rel)
            def mk(key, i=i):
                def h(mod, args, kwargs, out):
                    xin = args[0] if args else (kwargs.get("hidden_states") if "hidden_states" in kwargs else next(iter(kwargs.values())))
                    cap[i][key] = (xin.detach(), out)
                return h
            hooks += [lay.register_forward_hook(mk("layer"), with_kwargs=True),
                      am.register_forward_hook(mk("attn"), with_kwargs=True),
                      mm.register_forward_hook(mk("mlp"), with_kwargs=True)]
        try:
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=False)
        finally:
            for h in hooks: h.remove()
        logits = out.logits if hasattr(out, "logits") else _first(out)
        with torch.no_grad():
            st = {}
            def stage(key, real_fn, ref_fn):
                """One comparison; a page whose facts cannot even be applied to the real tensors (shape mismatch)
                is a mismatch too (inf), with the error kept."""
                try:
                    st[key] = rel(real_fn(), ref_fn())
                except NotCheckable:
                    raise
                except (RuntimeError, IndexError, ValueError, AttributeError, TypeError) as e:
                    st[key] = float("inf"); errors.setdefault(key, f"{type(e).__name__}: {str(e)[:160]}")
            stage("embedding", lambda: cap[0]["layer"][0], lambda: ref.embed(ids))
            for i in range(n_layers):
                x, lo = cap[i]["layer"][0], _first(cap[i]["layer"][1])
                ain, aout = cap[i]["attn"][0], cap[i]["attn"][1]
                a = _first(aout)
                min_, m = cap[i]["mlp"][0], _first(cap[i]["mlp"][1])
                sfx = f" [layer {i}]"
                stage("attention input" + sfx, lambda: ain, lambda: ref.attn_input(x, i))
                probs_real = aout[1] if isinstance(aout, (tuple, list)) and len(aout) > 1 and torch.is_tensor(aout[1]) \
                    and aout[1].dim() == 4 else None
                if probs_real is not None:
                    stage("attention probabilities" + sfx, lambda: probs_real, lambda: ref.attention_probs(ain, i)[0])
                stage("attention output" + sfx, lambda: a, lambda: ref.attention(ain, i))
                stage("MLP input" + sfx, lambda: min_, lambda: ref.mlp_input(x, a, i))
                stage("MLP output" + sfx, lambda: m, lambda: ref.mlp(min_, i))
                stage("layer output" + sfx, lambda: lo, lambda: ref.layer_output(x, a, m, i))
            last = _first(cap[n_layers - 1]["layer"][1])
            stage("final norm + head", lambda: logits, lambda: ref.final(last))
            stage("end to end" if S <= LONG_S else "end to end (long run)", lambda: logits, lambda: ref.model(ids, n_layers))
        for k, v in st.items():
            worst[k] = max(worst[k], v)
    fails = [k for k, v in worst.items() if not (v <= tol)]
    order = list(worst)
    first = min(fails, key=order.index) if fails else None
    return {"stages": dict(worst), "fail_stages": fails, "first_fail": first,
            "first_fail_facts": STAGE_FACTS[re.sub(r" \[layer \d+\]$", "", first)] if first else None,
            "end_to_end": worst["end to end"], "errors": errors, "tol": tol, "seeds": list(seeds), "S": S, "B": B}


def undrawn_inputs(cfg, P):
    """Values the reference needs that the page does not state (taken from the config, listed on the row). Norm eps is
    on the page (group 2), so only the llama3 low / high frequency factors remain."""
    eps = None
    l3 = None
    if P["llama3"]:
        rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
        l3 = {k: float(rs[k]) for k in ("low_freq_factor", "high_freq_factor")}
    und = {}
    if l3: und.update({f"llama3 {k}": v for k, v in l3.items()})
    return eps, l3, und


ROLE_FACT = {"emb": "vocab / hidden", "q": "query heads x head_dim", "k": "K/V heads x head_dim", "v": "K/V heads x head_dim",
             "o": "output projection width", "gate": "FFN width", "up": "FFN width", "down": "FFN width",
             "n1_w": "attention-input norm", "n2_w": "MLP-input norm (placement / parallel norm count)",
             "no1_w": "attention-output norm (placement)", "no2_w": "MLP-output norm (placement)",
             "nf_w": "final norm", "qn_w": "QK-norm extent", "kn_w": "QK-norm extent",
             "head": "vocab / hidden", "pos": "position table", "q_b": "Q/K/V bias width", "head_b": "LM head bias"}


def static_checks(P, M, model, cfg):
    """Page facts vs the real model's tensors (no run needed): sizes, biases, tie, layer count. [(verdict, fact, msg)]"""
    out = []
    names = dict(model.named_parameters(remove_duplicate=False))
    T = lambda r: names.get(M.get(r)) if M.get(r) else None
    H, V, nh, nkv, hd, I = P["hidden"], P["vocab"], P["num_heads"], P["num_kv"], P["head_dim"], P["inter"]
    mat = (lambda o, i: (o, i)) if P.get("w_layout") != "in_out" else (lambda o, i: (i, o))   # the page's layout
    want = {"emb": (V, H), "up": mat(I, H), "down": mat(H, I)}
    pl = P["placement"]
    lnorms = (["n1_w"] if P["residual"] == "parallel" or pl in ("pre", "double") else []) + \
        (["n2_w"] if (P["residual"] != "parallel" and pl in ("pre", "double")) or
         (P["residual"] == "parallel" and P.get("parallel_norms") == 2) else []) + \
        (["no1_w", "no2_w"] if P["residual"] != "parallel" and pl in ("post", "double") else [])
    if P.get("norm_weight") != "none": want.update({r: (H,) for r in lnorms})
    if P.get("final_weight") != "none": want["nf_w"] = (H,)
    # a norm drawn with NO weight on a model whose norm reads one is a wrong page (the weight would be silently dropped)
    for form_, roles_, fact_ in ((P.get("norm_weight"), lnorms, "layer norm weight form"),
                                 (P.get("final_weight"), ["nf_w"], "final norm weight form"),
                                 (P.get("qk_weight") if P["qk_norm"] else None, ["qn_w", "kn_w"], "QK-norm weight form")):
        if form_ == "none":
            have_ = [M[r] for r in roles_ if T(r) is not None]
            if have_:
                out.append(("FAIL", fact_, f"the page draws the norm with no weight; the model's norm reads {', '.join(have_)}"))
    if P.get("fused"):
        want["q"] = mat((nh + 2 * nkv) * hd, H)
        if P["qkv_bias"]: want["q_b"] = ((nh + 2 * nkv) * hd,)
    else:
        want.update(q=mat(nh * hd, H), k=mat(nkv * hd, H), v=mat(nkv * hd, H))
    if P["o_proj"]: want["o"] = mat(H, nh * hd)
    if P["gated"] and P.get("gate_up"): want["gate"] = want["up"] = mat(2 * I, H)
    elif P["gated"]: want["gate"] = mat(I, H)
    if P.get("pos_emb"): want["pos"] = (P.get("pos_rows"), H)
    if P.get("head_bias"): want["head_b"] = (V,)
    if P["qk_norm"] and P.get("qk_weight") != "none": want.update(qn_w=(P["qk_norm_extent"],), kn_w=(P["qk_norm_extent"],))
    if P["tied"] is False: want["head"] = (V, H)
    for r, shp in want.items():
        t = T(r)
        if t is None:
            out.append(("FAIL", ROLE_FACT[r], f"the page draws {r} but the model has no tensor in that position"))
        elif tuple(t.shape) != tuple(shp):
            out.append(("FAIL", ROLE_FACT[r], f"the page's facts imply {r} of shape {list(shp)}; the model's tensor "
                                              f"{M[r]} is {list(t.shape)}"))
    for fact, roles, drawn in (("Q/K/V bias", ("q_b", "k_b", "v_b"), P["qkv_bias"]),
                               ("output projection bias", ("o_b",), P["o_bias"]),
                               ("FFN bias", ("gate_b", "up_b", "down_b"), P["ffn_bias"])):
        have = [r for r in roles if T(r) is not None]
        need_all = fact != "FFN bias"
        if drawn and (not have or (need_all and len(have) != len(roles))):
            out.append(("FAIL", fact, f"the page draws a {fact}; the model has {have or 'no'} bias tensors there"))
        if not drawn and have:
            out.append(("FAIL", fact, f"the page draws no {fact}; the model has {', '.join(M[r] for r in have)}"))
    e, h = T("emb"), T("head")
    same = e is not None and h is not None and (h is e or h.data_ptr() == e.data_ptr())
    if P["tied"] in (True, None) and not same:
        out.append(("FAIL", "tied head", f"the page says the head is the embedding; the model's head reads {M.get('head')}"))
    if P["tied"] is False and same:
        out.append(("FAIL", "tied head", "the page says the head is a separate tensor; the model's head reads the embedding"))
    for r, kind_ in (("n1", P["norm"]), ("n2", P["norm"]), ("no1", P["norm"]), ("no2", P["norm"]), ("nf", P["final_norm"])):
        if kind_ == "rmsnorm" and T(r + "_b") is not None:
            out.append(("FAIL", "norm kind", f"the page draws an RMSNorm (no bias); the model's norm adds {M[r + '_b']}"))
    if T("head_b") is not None and not P.get("head_bias"):
        out.append(("FAIL", "LM head bias", f"the page draws no LM head bias; the model's head adds {M['head_b']}"))
    if P.get("fused") and (M.get("q") != M.get("k") or M.get("q") != M.get("v")):
        out.append(("FAIL", "fused Q/K/V", "the page draws one fused projection; the recording maps Q / K / V to different tensors"))
    n_cfg = cfg.get("num_hidden_layers")
    if n_cfg is not None and P["layers"] != n_cfg:
        out.append(("FAIL", "layer count", f"the page draws {P['layers']} layers; the model built from the config has {n_cfg}"))
    return out


def affordable_layers(cfg, budget_bytes=4.5e9):
    """All layers when the float64 model fits the budget, else 2 (the page's identical-layer claim covers the rest).
    `cfg`: a dict, or a bundle (its config is read through the library's config class: n_embd / n_layer / ...)."""
    if isinstance(cfg, dict) and "config" in cfg and "class" in cfg:
        import transformers
        cls = getattr(transformers, cfg["class"], None)
        c_ = cls.config_class.from_dict(dict(cfg["config"])) if cls else None
        g_ = lambda k: getattr(c_, k, None) if c_ is not None else None
        H, V, L = g_("hidden_size") or 0, g_("vocab_size") or 0, g_("num_hidden_layers") or 0
        I = g_("intermediate_size") or g_("n_inner") or 4 * H
    else:
        H, V, I, L = (cfg.get(k) or 0 for k in ("hidden_size", "vocab_size", "intermediate_size", "num_hidden_layers"))
    per = 4 * H * H + 3 * H * I
    full = (2 * V * H + L * per) * 8
    return L if L and full <= budget_bytes else min(2, L or 2)


def config_window(B):
    """The sliding window the library would build from this config (through its config class), else None."""
    import transformers
    cls = getattr(transformers, B["class"], None)
    if cls is None: return None
    c_ = cls.config_class.from_dict(dict(B["config"]))
    w = getattr(c_, "sliding_window", None)
    if not isinstance(w, int) or w <= 0 or getattr(c_, "use_sliding_window", None) is False: return None
    lt = getattr(c_, "layer_types", None)
    if lt and not any("sliding" in str(t) for t in lt): return None
    return w


_SIG_CACHE = {}
def layer_signatures(B, stack_path):
    """(signatures, why): one per layer of the FULL library model, built on the meta device (no memory): every scalar
    attribute the library holds on the layer and its submodules (e.g. an attention module's sliding window / layer type,
    which also selects its rotation table), keyed by submodule path. An attribute equal to the layer's own index in every
    layer (an index, not a configuration) is left out. Library evidence, independent of the recognizer."""
    key = (id(B), stack_path)
    if key in _SIG_CACHE: return _SIG_CACHE[key][1]
    import transformers
    try:
        cls = getattr(transformers, B["class"])
        conf = cls.config_class.from_dict(dict(B["config"]))
        with torch.device("meta"):
            m = cls._from_config(conf, attn_implementation="eager")
        stack = m.get_submodule(stack_path)                 # the page's layer stack (its parameter mapping)
        raw = []
        for layer in stack:
            d = {}
            for n, sub in layer.named_modules():
                for k, v in vars(sub).items():
                    if k.startswith("_") or k == "training": continue
                    if v is None or isinstance(v, (bool, int, float, str)): d[(n, k)] = v
            raw.append(d)
        keys = set().union(*raw)
        idx = {k for k in keys if all(r.get(k) == i for i, r in enumerate(raw))}
        sigs = [tuple(sorted(((n + "." if n else "") + k, repr(r.get((n, k), "<absent>"))) for n, k in keys - idx))
                for r in raw]
        out = (sigs, "")
    except Exception as e:
        out = (None, f"{type(e).__name__}: {str(e)[:120]}")
    _SIG_CACHE[key] = (B, out)
    return out


def unbuilt_layer_checks(B, P, nl, stack_path):
    """The page's kind (mask, window, rotation base) of every layer the oracle does not build must be the kind of a
    BUILT (run-checked) layer the library configures identically; a layer configured unlike every built one is
    NOT_CHECKABLE (named), never PASS."""
    n = P["layers"] or 0
    if nl >= n: return []
    sigs, why = layer_signatures(B, stack_path)
    fact = "layer kind of unbuilt layers (mask / window / RoPE base)"
    if sigs is None or len(sigs) != n:
        return [("NOT_CHECKABLE", fact, f"layers {nl}..{n - 1} are not built and the library's per-layer configuration "
                                        f"could not be read ({why or f'{0 if sigs is None else len(sigs)} layers'})")]
    kinds = P.get("kinds") or [{"mask": P["mask"], "window": P.get("window"), "theta": P["theta"]}]
    kol = P.get("kind_of_layer") or [0] * n
    pk = lambda i: (kinds[kol[i]]["mask"], kinds[kol[i]].get("window"), kinds[kol[i]]["theta"])
    desc = lambda t: (f"sliding window {t[1]}" if t[0] == "sliding" else t[0]) + (f", RoPE base {t[2]:g}" if t[2] else "")
    out, unmatched = [], []
    for i in range(nl, n):
        rep_ = next((j for j in range(nl) if sigs[j] == sigs[i]), None)
        if rep_ is None: unmatched.append(i); continue
        if pk(i) != pk(rep_):
            out.append(("FAIL", "layer kind", f"layer {i} is drawn as {desc(pk(i))}; the library configures it exactly like "
                                              f"built layer {rep_}, which is checked as {desc(pk(rep_))}"))
            break
    if unmatched and not out:
        ex = sorted(set(sigs[unmatched[0]]) - set(sigs[0]))[:3]
        out.append(("NOT_CHECKABLE", fact, f"layers {unmatched[:6]}{'...' if len(unmatched) > 6 else ''} are configured "
                                           f"unlike every built layer (e.g. {ex}) and are not built here"))
    return out


def model_dtype(B, n_layers, budget=4.5e9):
    """float64 when the model with n_layers fits the budget, else float32 (stated on the row); None if neither fits."""
    import transformers
    cls = getattr(transformers, B["class"], None)
    c_ = cls.config_class.from_dict(dict(B["config"])) if cls else None
    g_ = lambda k: getattr(c_, k, None) if c_ is not None else None
    H, V = g_("hidden_size") or 0, g_("vocab_size") or 0
    I = g_("intermediate_size") or g_("n_inner") or 4 * H
    tied = bool(g_("tie_word_embeddings"))
    n_par = V * H * (1 if tied else 2) + n_layers * (4 * H * H + 3 * H * I)
    for dt, b in ((torch.float64, 8), (torch.float32, 4)):
        if n_par * b <= budget * (1 if b == 8 else 1.2): return dt, ""
    return None, f"{n_par / 1e9:.1f}B parameters do not fit even in float32"


def long_run_ok(P, S, budget=4e9):
    """A batch-1 run of S tokens: the attention probabilities (heads x S^2, a few copies) and the logits fit."""
    nh, V = P["num_heads"] or 1, P["vocab"] or 1
    return S <= 4200 and (6 * nh * S * S * 8 + 3 * S * V * 8) <= budget


def run_bundle(B, n_layers=None, seeds=(0, 1, 2, 3, 4), S=24, model=None, randomize_weights=True, f_edit=None, ir_edit=None):
    """One row: NOT_DRAWN / NOT_CHECKABLE / PASS / FAIL. `model`: an already-built real model (the mutant harness);
    `f_edit` edits the recognizer facts before to_ir (a self-consistent wrong page), `ir_edit` edits the rendered IR."""
    row = {"repo": B.get("repo"), "class": B.get("class"), "unchecked_facts": None}
    try:
        F = exec_to_ir.recognize(B)
    except exec_to_ir.NotDrawn as e:
        row.update(verdict="NOT_DRAWN", reason=str(e)); return row
    import page_reader
    row["unchecked_facts"] = page_reader.UNCHECKED
    cfg = B.get("config") or {}
    res, issues = None, []
    try:
        if f_edit: f_edit(F)
        ir = exec_to_ir.to_ir(F)
        if ir_edit: ir_edit(ir)
        P, issues = page_facts(ir)
        if P is None:
            row.update(verdict=f"FAIL(page: {issues[0][1]})", issues=issues); return row
        row["page_facts"] = {k: v for k, v in P.items() if k != "src"}
        M = param_map(B, P)
        row["param_map"] = {k: v for k, v in M.items() if v and not k.startswith("_")}
        eps, l3, und = undrawn_inputs(cfg, P)
        row["undrawn_inputs"] = und
        cfg = dict(cfg, num_hidden_layers=config_layers(B))
        # every layer KIND the page states must run: build at least up to the first layer of each kind
        kinds_ = P.get("kinds") or [{"mask": P["mask"], "window": P.get("window"), "theta": P["theta"]}]
        need_ = (max(P["kind_of_layer"].index(k) for k in range(len(kinds_))) + 1) if P.get("kinds") else 1
        if model is None:
            # ... and the first layer of every distinct library per-layer configuration (meta-device signatures)
            sigs_, _w = layer_signatures(B, M["_stack"])
            need_lib = (max(sigs_.index(x) for x in set(sigs_)) + 1) if sigs_ else 1
            n_build = max(n_layers or affordable_layers(B), min(max(need_, need_lib), P["layers"]))
            dt_, why_ = model_dtype(B, n_build)
            row["dtype"] = str(dt_).replace("torch.", "")
            if dt_ is None:
                raise NotCheckable(f"memory: {why_}")
            model = build_real(B, n_build, dtype=dt_)
        # a window must be crossed: one run longer than the widest stated window (when memory allows)
        wins_ = [k["window"] for k in kinds_ if k.get("window")]
        S_list = list(S) if isinstance(S, (tuple, list)) else [S]
        if wins_:
            S_long = max(wins_) + 16
            if long_run_ok(P, S_long):
                S_list.append(S_long)
            else:
                issues = issues + [("NOT_CHECKABLE", "sliding window", f"the stated window {max(wins_)} needs a run longer "
                                    f"than {max(wins_)} tokens, too large for this machine (proven on a constructed model "
                                    "with a small window instead)")]
            row["window_runs"] = [x for x in S_list if x > max(wins_)]
        else:
            # the page states no window: a window the library would apply beyond every checked length is invisible
            # to the runs, so a "full causal" page on a windowed model cannot PASS (the config only refuses a PASS here;
            # it never supplies a drawn fact)
            cw_ = config_window(B)
            s_max = max(S) if isinstance(S, (tuple, list)) else S
            if cw_ and cw_ >= s_max and long_run_ok(P, cw_ + 16):
                S_list.append(cw_ + 16)                              # the config picks a run LENGTH only: cross it
                row["window_runs"] = [cw_ + 16]
            elif cw_ and cw_ >= s_max:
                issues = issues + [("NOT_CHECKABLE", "sliding window", f"the page states full causal attention; the "
                                    f"model's config declares a {cw_}-token window, longer than every checked run "
                                    f"({s_max}): full vs windowed attention is not distinguishable here")]
        nl = len(model.get_submodule(M["_stack"]))
        row["real_layers_built"], row["drawn_layers"] = nl, P["layers"]
        issues = issues + static_checks(P, M, model, cfg) + unbuilt_layer_checks(B, P, nl, M["_stack"])
        size_fail = any(v == "FAIL" and f in set(ROLE_FACT.values()) for v, f, _ in issues)
        if not size_fail:                                            # wrong sizes: the reference cannot run
            for S_ in S_list:                                        # worst stage error kept
                r_ = check(model, P, M, eps, l3, nl, seeds=seeds if S_ <= 300 else seeds[:1] + tuple(x for x in seeds if x in EPS_SEEDS),
                           S=S_, B=2 if S_ <= 300 else 1, randomize_weights=randomize_weights)
                if res is None:
                    res = r_; res["S"] = [S_]
                else:
                    for k, v in r_["stages"].items(): res["stages"][k] = max(res["stages"].get(k, 0.0), v)
                    res["errors"].update({k: v for k, v in r_["errors"].items() if k not in res["errors"]})
                    res["S"].append(S_)
            order = list(res["stages"])
            res["fail_stages"] = [k for k, v in res["stages"].items()
                                  if not (v <= (TOL_LONG_E2E if k == "end to end (long run)" else res["tol"]))]
            if "end to end (long run)" in res["stages"]:
                res["long_run_note"] = (f"end to end (long run) {res['stages']['end to end (long run)']:.2e}, tol "
                                        f"{TOL_LONG_E2E:g} (temporary, measured), stages tol {res['tol']:g}; cause: "
                                        + LONG_RUN_CAUSE)
            res["first_fail"] = min(res["fail_stages"], key=order.index) if res["fail_stages"] else None
            res["first_fail_facts"] = (STAGE_FACTS[re.sub(r" \[layer \d+\]$", "", res["first_fail"])]
                                       if res["first_fail"] else None)
            res["end_to_end"] = res["stages"]["end to end"]
            row.update(res)
    except NotCheckable as e:                                         # named; a page FAIL still takes precedence
        issues = issues + [("NOT_CHECKABLE", str(e).split(":")[0], str(e))]
        res = None
    row["issues"] = issues
    fails = [x for x in issues if x[0] == "FAIL"]
    nck = [x for x in issues if x[0] == "NOT_CHECKABLE"]
    if fails:
        row["verdict"] = f"FAIL(page: {fails[0][1]})"; row["reason"] = fails[0][2]
    elif res and res["fail_stages"]:
        row["verdict"] = f"FAIL({res['first_fail']})"
    elif nck:
        row["verdict"] = f"NOT_CHECKABLE({nck[0][1]})"; row["reason"] = nck[0][2]
    else:
        row["verdict"] = "PASS"
    return row


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    B = capture.load(a[0])
    r = run_bundle(B, n_layers=opt("--layers", 0) or None, S=opt("--seq", 24),
                   seeds=tuple(int(x) for x in opt("--seeds", "0,1").split(",")))
    print(json.dumps(r, indent=1, default=str))
