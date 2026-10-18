"""Read what a drawn page STATES (P4 oracle, Track D): the facts in the spec fields AND in the text the reader sees.

exec_to_ir.to_ir writes every fact twice: once into the typed spec (LayerSpec / AttentionSpec / FFNSpec / ModelIR fields,
which the renderer turns into some text of its own) and once as free display text (block labels, titles, descriptions
and fact chips, model blocks, the EXECUTION EVIDENCE card, notes). The oracle checks the spec functionally, so the text
must say the same thing: every display string is parsed here by a fixed grammar and compared with the spec.

read_page(ir) -> (P, issues)
  P       the page facts (from the spec) the reference is built from
  issues  [(verdict, fact, message)]: "FAIL" = the page contradicts itself (text vs spec, block order vs topology, layer
          i vs layer 0, card vs spec); "NOT_CHECKABLE" = the page shows text this grammar cannot read (named), or a
          value outside the oracle's vocabulary for a fact it checks.
UNCHECKED lists the page facts the oracle does not check at all (shown on every row, never silently).
"""
import math, re

UNCHECKED = ["K/V cache (decode with a cache is not run)", "context length (max_position_embeddings: config-declared)",
             "graded verdict / evidence notes", "norm eps (not stated on the page)"]
NK = {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}
NK_R = {v: k for k, v in NK.items()}
ACT_LABEL = {"silu": "SiLU", "gelu": "GELU", "relu": "ReLU"}
ACT_R = {v: k for k, v in ACT_LABEL.items()}
KIND_TITLE = {"gqa": "Grouped-query attention", "mha": "Multi-head attention", "mqa": "Multi-query attention"}
KIND_LABEL = {"gqa": "Grouped-Query", "mha": "Multi-Head", "mqa": "Multi-Query"}
TIE_TXT = {
    " — weights tied with the output head (same tensor read; the checkpoint ships one of them).": True,
    " — tied to the output head in the model built from the config, but the checkpoint ships both tensors: the library "
    "ties them only if their values are equal, which was not checked.": "unverified",
    " — tied to the output head in the model built from the config; whether the checkpoint ships one or both tensors "
    "is not recorded, so the tie is unverified.": "unverified",
    ".": False}
# the layer topologies this oracle has a reference for, as the drawn block sequence (id, kind, residual_from)
TOPOLOGY = {
    ("pre", "sequential"): [("rms1", "norm", None), ("attn", "attention", None), ("add1", "residual_add", "rms1"),
                            ("rms2", "norm", None), ("ffn", "ffn", None), ("add2", "residual_add", "rms2")],
    ("post", "sequential"): [("attn", "attention", None), ("add1", "residual_add", "attn"), ("rms1", "norm", None),
                             ("ffn", "ffn", None), ("add2", "residual_add", "ffn"), ("rms2", "norm", None)],
    ("pre", "parallel"): [("rms1", "norm", None), ("attn", "attention", None), ("add1", "residual_add", "rms1"),
                          ("ffn", "ffn", None)],
}
_N = r"(\d[\d,]*)"


def _int(s):
    return int(s.replace(",", ""))


def _same_detail(a, b):
    """A block's detail dict states the same spec (empty values -- None / [] / False -- count as one)."""
    if not isinstance(a, dict) or not isinstance(b, dict): return False
    return all((a.get(k) == b.get(k)) or (not a.get(k) and not b.get(k)) for k in set(a) | set(b))


def spec_facts(d):
    """The facts in the typed spec fields of layer 0 / the model (what the reference is built from)."""
    L0 = d["layers"][0]
    a, f = L0["attention"], L0["ffn"]
    card = {x["key"]: x for x in d["extras"]["execution_evidence"]["facts"]}
    l3 = None
    for n in d.get("notes") or []:
        mm = re.search(r"llama3 frequency scaling declared \(factor ([\d.e+]+), original context (\d+)\)", n)
        if mm: l3 = {"factor": float(mm.group(1)), "original_max_position_embeddings": int(mm.group(2))}
    return dict(
        vocab=d["vocab_size"], hidden=d["hidden_size"], layers=len(d["layers"]), tied=d["tie_word_embeddings"],
        final_norm=d["final_norm_kind"], norm=L0["norm_kind"], placement=L0["norm_placement"],
        residual=L0["residual_topology"], kind=a["kind"], num_heads=a["num_heads"], num_kv=a["num_kv_heads"],
        head_dim=a["head_dim"], qk_norm=bool(a.get("qk_norm")), qk_norm_kind=a.get("qk_norm_kind"),
        qk_norm_extent=a.get("qk_norm_extent"), rope=bool(a.get("rope")), theta=a.get("rope_theta"), llama3=l3,
        mask=a.get("mask"), qkv_bias=bool(a.get("bias")), o_proj=bool(a.get("output_projection")),
        scores_scaled=bool(a.get("scores_scaled")), gated=bool(f.get("gated")), act=f.get("activation"),
        inter=f.get("intermediate_size"), ffn_bias=bool(f.get("bias")),
        fused=a.get("projection_mode") == "fused_qkv", rot_dim=a.get("rope_dim"),
        parallel_norms=L0.get("parallel_norm_count"),
        # from the text / card only (no spec field): filled below
        qkv_split=None, w_layout="out_in", pos_emb=False, head_bias=False, pos_rows=None, qkv_width=None,
        # facts only the card / text carry (no spec field): taken from the card, checked against the text below
        o_bias=(card.get("decoder.attention.o_bias") or {}).get("value"),
        scale=(card.get("decoder.attention.scores_scale") or {}).get("value"),
        # the tensor names the page shows (EXECUTION EVIDENCE card sources)
        src={k: (v.get("source") or "") for k, v in card.items()})


def read_page(ir):
    d = ir.to_dict() if hasattr(ir, "to_dict") else ir
    issues = []
    fail = lambda fact, msg: issues.append(("FAIL", fact, msg))
    unread = lambda fact, msg: issues.append(("NOT_CHECKABLE", fact, msg))
    if not d.get("layers"):
        return None, [("FAIL", "layers", "the page draws no layer")]
    P = spec_facts(d)
    seen = []                                         # (fact, value, where) read from the text

    def saw(fact, value, where):
        seen.append((fact, value, where))

    # ---- every layer is the same as layer 0 (the page draws one layer and repeats it)
    strip = lambda L: {k: v for k, v in L.items() if k != "index"}
    for i, L in enumerate(d["layers"][1:], 1):
        if strip(L) != strip(d["layers"][0]):
            keys = sorted(k for k in strip(L) if strip(L)[k] != strip(d["layers"][0]).get(k))
            fail("layers identical", f"layer {i} differs from layer 0 on the page ({', '.join(keys)})")
            break
    L0 = d["layers"][0]
    blocks = L0.get("blocks") or []
    # ---- the drawn block sequence is one topology the oracle has a reference for, and it is the spec's topology
    seq = [(b.get("id"), b.get("kind"), b.get("residual_from")) for b in blocks]
    drawn_topo = next((t for t, s in TOPOLOGY.items() if s == seq), None)
    if drawn_topo is None:
        if (P["placement"], P["residual"]) in TOPOLOGY:
            fail("block order", f"the layer's block sequence {[s[0] for s in seq]} is not the "
                                f"{P['placement']}-norm {P['residual']} order the layer spec states")
        else:
            unread("block order", f"the layer's block sequence {[s[0] for s in seq]} has no reference topology")
    elif drawn_topo != (P["placement"], P["residual"]):
        fail("block order", f"the blocks are drawn {drawn_topo[0]}-norm {drawn_topo[1]}, the layer spec says "
                            f"{P['placement']}-norm {P['residual']}")
    # ---- vocabulary of the values the reference supports (a value outside it is named, never passed)
    for fact, ok in (("norm", P["norm"] in NK), ("final_norm", P["final_norm"] in NK), ("mask", P["mask"] == "causal"),
                     ("act", P["act"] in ("silu", "gelu", "gelu_pytorch_tanh", "relu", "sigmoid", "tanh")),
                     ("kind", P["kind"] in KIND_TITLE),
                     ("qk_norm_kind", (not P["qk_norm"]) or P["qk_norm_kind"] in NK),
                     ("placement / residual", (P["placement"], P["residual"]) in TOPOLOGY)):
        if not ok:
            unread(fact, f"the page states {fact} = {P.get(fact, (P['placement'], P['residual']))!r}: the oracle has no "
                         "reference for that value")
    # kind is a function of the head counts
    nh, nkv = P["num_heads"], P["num_kv"]
    if nh and nkv:
        want = "mha" if nkv == nh else ("mqa" if nkv == 1 else "gqa")
        if P["kind"] != want:
            fail("attention kind", f"the page says {P['kind']} with {nh} query / {nkv} K-V heads ({want})")

    # ---- the text, block by block
    H = P["hidden"]
    def chips(where, items, readers, free=()):
        for s in items or []:
            if s in free: continue
            for rx, fn in readers:
                m = re.fullmatch(rx, s)
                if m:
                    fn(m); break
            else:
                unread(where, f"{where} shows {s!r}, which the oracle cannot read")
    def width(role_in, role_out, where):
        return (rf"{_N} → {_N}", lambda m: (saw(role_in, _int(m.group(1)), where), saw(role_out, _int(m.group(2)), where)))
    def text(where, s, readers, free=()):
        chips(where, [s] if s is not None else [], readers, free)

    for b in blocks:
        bid = b.get("id"); w = f"block {bid}"
        if b.get("kind") == "norm":
            text(w + " label", b.get("label"), [(r"(RMSNorm|LayerNorm)", lambda m: saw("norm", NK_R[m.group(1)], w))])
            chips(w, b.get("facts"), [(rf"dim {_N}", lambda m: saw("hidden", _int(m.group(1)), w))])
            text(w + " description", b.get("description"),
                 [(r"Recorded (RMSNorm|LayerNorm) on the layer input before attention\.", lambda m: saw("norm", NK_R[m.group(1)], w)),
                  (r"Recorded (RMSNorm|LayerNorm) before the FFN\.", lambda m: saw("norm", NK_R[m.group(1)], w)),
                  (r"Recorded (RMSNorm|LayerNorm) on the layer input before attention; a second, separate (RMSNorm|LayerNorm) on "
                   r"the layer input before the FFN\.", lambda m: (saw("norm", NK_R[m.group(1)], w), saw("norm", NK_R[m.group(2)], w),
                                                                  saw("parallel_norms", 2, w))),
                  (r"Recorded (RMSNorm|LayerNorm) on the layer input, feeding both attention and the FFN\.",
                   lambda m: (saw("norm", NK_R[m.group(1)], w), saw("parallel_norms", 1, w)))])
            text(w + " title", b.get("title"), [(r"Pre-attention norm|Pre-FFN norm", lambda m: saw("parallel_norms", None, w)),
                                                (r"Pre-block norms \(separate\)", lambda m: saw("parallel_norms", 2, w)),
                                                (r"Pre-block norm \(shared\)", lambda m: saw("parallel_norms", 1, w))])
        elif bid == "attn":
            lab = b.get("label")
            labs = lab if isinstance(lab, list) else [lab]
            for x in labs:
                text(w + " label", x, [(r"(Grouped-Query|Multi-Head|Multi-Query)(?: Attention)?",
                                        lambda m: saw("kind", {v: k for k, v in KIND_LABEL.items()}[m.group(1)], w)),
                                       (r"\(QK-Norm\)", lambda m: saw("qk_norm", True, w))])
            if "(QK-Norm)" not in labs: saw("qk_norm", False, w + " label")
            text(w + " title", b.get("title"), [(r"(Grouped-query|Multi-head|Multi-query) attention( \(QK-Norm\))?",
                                                 lambda m: (saw("kind", {"Grouped-query": "gqa", "Multi-head": "mha", "Multi-query": "mqa"}[m.group(1)], w),
                                                            saw("qk_norm", bool(m.group(2)), w)))])
            text(w + " description", b.get("description"),
                 [(r"(Grouped-query|Multi-head|Multi-query) attention — observed head counts from the recorded tensors\.",
                   lambda m: saw("kind", {"Grouped-query": "gqa", "Multi-head": "mha", "Multi-query": "mqa"}[m.group(1)], w))])
            bias_chip = []
            chips(w, b.get("facts"), [
                (rf"{_N} Q heads", lambda m: saw("num_heads", _int(m.group(1)), w)),
                (rf"{_N} KV heads", lambda m: saw("num_kv", _int(m.group(1)), w)),
                (rf"head dim {_N}", lambda m: saw("head_dim", _int(m.group(1)), w)),
                (rf"RoPE θ {_N}", lambda m: saw("theta~int", _int(m.group(1)), w)),
                (r"QK-Norm", lambda m: saw("qk_norm", True, w)),
                (r"bias-free projections", lambda m: bias_chip.append((False, False))),
                (r"bias on Q/K/V, none on the output projection", lambda m: bias_chip.append((True, False))),
                (r"bias on the output projection only", lambda m: bias_chip.append((False, True))),
                (r"weights stored \[in, out\]", lambda m: saw("w_layout", "in_out", w)),
                (rf"partial RoPE {_N}/{_N}", lambda m: (saw("rot_dim", _int(m.group(1)), w), saw("head_dim", _int(m.group(2)), w)))])
            if "weights stored [in, out]" not in (b.get("facts") or []): saw("w_layout", "out_in", w + " (no layout chip)")
            if not any(re.fullmatch(r"partial RoPE .*", x) for x in b.get("facts") or []): saw("rot_dim", None, w + " (no partial chip)")
            qb, ob = bias_chip[0] if bias_chip else (True, True)      # no chip = both biased (to_ir's convention)
            saw("qkv_bias", qb, w + " bias chip"); saw("o_bias", ob, w + " bias chip")
            if not any(re.fullmatch(r"RoPE θ .*", s) for s in b.get("facts") or []) and P["rope"] and P["theta"]:
                fail("rope_theta", "the attention block's chips state no RoPE θ while the spec has one")
            if ("QK-Norm" in (b.get("facts") or [])) != P["qk_norm"]:
                fail("qk_norm", "the attention block's QK-Norm chip disagrees with the spec")
            if not _same_detail((b.get("detail") or {}).get("attention"), L0["attention"]):
                fail("attention detail", "the attention block's detail differs from the layer's attention spec")
            kids = {c.get("id"): c for c in b.get("children") or []}
            want_kids = ({"qkv_proj"} if P["fused"] else {"q_proj", "k_proj", "v_proj"}) | {"q_reshape", "k_reshape", "scaled_scores", "attn_softmax",
                         "attn_apply_v", "concat_heads"} | ({"q_norm", "k_norm"} if P["qk_norm"] else set()) \
                | ({"q_rope", "k_rope"} if P["rope"] else set()) | ({"o_proj"} if P["o_proj"] else set())
            have = set(kids) - {"kv_cache"}
            if have != want_kids:
                fail("attention sub-blocks", f"drawn sub-blocks {sorted(have ^ want_kids)} disagree with the spec "
                                             "(QK-norm / RoPE / output projection)")
            for cid, c in kids.items():
                cw = f"{w}.{cid}"
                F = c.get("facts")
                if cid == "q_proj":
                    chips(cw, F, [width("hidden", "q_out", cw), (rf"{_N} Q heads", lambda m: saw("num_heads", _int(m.group(1)), cw)),
                                  (rf"head dim {_N}", lambda m: saw("head_dim", _int(m.group(1)), cw))])
                elif cid in ("k_proj", "v_proj"):
                    r = cid[0]
                    chips(cw, F, [width("hidden", f"{r}_out", cw), (rf"{_N} KV heads", lambda m: saw("num_kv", _int(m.group(1)), cw))],
                          free=("cache ports: ⌃ write · ⊥ read",))
                elif cid in ("q_reshape", "k_reshape"):
                    chips(cw, F, [(rf"head dim {_N}", lambda m: saw("head_dim", _int(m.group(1)), cw))])
                elif cid in ("q_norm", "k_norm"):
                    text(cw + " title", c.get("title"), [(r"(?:Query|Key) (RMSNorm|LayerNorm)", lambda m: saw("qk_norm_kind", NK_R[m.group(1)], cw))])
                    text(cw, c.get("description"), [(r"Recorded (RMSNorm|LayerNorm) on [QK] over the last axis \((\d+)\), (before|after) RoPE\.",
                                                     lambda m: (saw("qk_norm_kind", NK_R[m.group(1)], cw), saw("qk_norm_extent", int(m.group(2)), cw),
                                                                saw("qk_norm_before_rope", m.group(3) == "before", cw)))])
                elif cid in ("q_rope", "k_rope"):
                    chips(cw, F, [(r"frequency base θ=([\d.e+-]+)", lambda m: saw("theta~6g", float(m.group(1)), cw)),
                                  (rf"rotary dims {_N} of {_N}", lambda m: (saw("rot_dim", _int(m.group(1)), cw), saw("head_dim", _int(m.group(2)), cw)))])
                    part = r"(?: Rotates the first (\d+) of (\d+) head dims; the other (\d+) pass through\.)?"
                    def _rd(m, cw=cw):
                        saw("theta~6g", float(m.group(1)), cw)
                        if m.group(2):
                            saw("rot_dim", int(m.group(2)), cw); saw("head_dim", int(m.group(3)), cw)
                            if int(m.group(2)) + int(m.group(4)) != int(m.group(3)): fail("rot_dim", f"{cw}: rotated + passed != head dims")
                        else:
                            saw("rot_dim", None, cw)
                    text(cw, c.get("description"), [
                        (r"Rotate-half rotary embedding recorded on the query heads; frequency base from the built model's buffer: θ=([\d.e+-]+)\." + part, _rd),
                        (r"Rotate-half rotary embedding recorded on the key heads; θ=([\d.e+-]+)\." + part, _rd)])
                elif cid == "qkv_proj":
                    sp = (r"(\[Q (?P<q>[\d,]+) \| K (?P<k>[\d,]+) \| V (?P<v>[\d,]+)\]"
                          r"|per K/V group \[q × (?P<g>\d+) \| k \| v\] × (?P<n>\d+), head dim (?P<d>\d+))")
                    def _sp(m, cw=cw):
                        if m.group("q"):
                            saw("qkv_split", ("qkv_blocks", _int(m.group("q")), _int(m.group("k")), _int(m.group("v"))), cw)
                        else:
                            saw("qkv_split", ("grouped", int(m.group("g")), int(m.group("n")), int(m.group("d"))), cw)
                    chips(cw, F, [(rf"{_N} → {_N}", lambda m: (saw("hidden", _int(m.group(1)), cw), saw("qkv_width", _int(m.group(2)), cw))),
                                  ("split " + sp, _sp)])
                    text(cw + " title", c.get("title"), [(r"Fused QKV projection", lambda m: None)])
                    text(cw, c.get("description"), [(rf"One recorded matmul produces Q, K and V; its {_N} output channels split as "
                                                     + sp + r" \(traced from the recorded split ops\)\.",
                                                     lambda m: (saw("qkv_width", _int(m.group(1)), cw), _sp(m)))])
                elif cid == "scaled_scores":
                    def _sc(m):
                        x = m.group(1)
                        if x is None: saw("scale", None, cw)
                        else:
                            saw("scale~6g", float(x), cw)
                            if m.group(2): saw("scale~exact", 1 / math.sqrt(int(m.group(2))), cw)
                        saw("mask", None if m.group(3) == "none" else m.group(3), cw)
                    text(cw, c.get("description"), [
                        (r"Q·Kᵀ recorded; (?:scores × ([\d.e+-]+)(?: \(= 1/√(\d+)\))?|no scale constant recorded); "
                         r"mask: (\w+) \(read from the mask tensor's values\)\.", _sc)])
                    chips(cw, F, [(rf"{_N} Q heads", lambda m: saw("num_heads", _int(m.group(1)), cw)),
                                  (rf"{_N} KV heads", lambda m: saw("num_kv", _int(m.group(1)), cw)),
                                  (rf"{_N} Q per KV head", lambda m: saw("q_per_kv", _int(m.group(1)), cw))])
                elif cid == "concat_heads":
                    chips(cw, F, [(rf"{_N} × {_N}", lambda m: (saw("num_heads", _int(m.group(1)), cw), saw("head_dim", _int(m.group(2)), cw))),
                                  (rf"→ {_N}", lambda m: saw("q_out", _int(m.group(1)), cw))])
                elif cid == "o_proj":
                    chips(cw, F, [width("o_in", "hidden", cw)])
                elif cid in ("attn_softmax", "attn_apply_v", "kv_cache"):
                    pass                                   # no checked fact (kv_cache: UNCHECKED)
                else:
                    unread(cw, f"sub-block {cid!r} is outside the oracle's grammar")
        elif bid == "ffn":
            actl = lambda s: ACT_R.get(s, s)
            text(w + " label", b.get("label"), [(r"Gated FFN", lambda m: saw("gated", True, w)), (r"FFN", lambda m: saw("gated", False, w))])
            text(w + " title", b.get("title"), [(r"Gated feed-forward", lambda m: saw("gated", True, w)), (r"Feed-forward", lambda m: saw("gated", False, w))])
            text(w + " description", b.get("description"), [
                (r"Gated MLP — recorded gate path modulates the up projection before the down projection\.", lambda m: saw("gated", True, w)),
                (r"Recorded two-layer MLP\.", lambda m: saw("gated", False, w))])
            chips(w, b.get("facts"), [(rf"hidden {_N}", lambda m: saw("inter", _int(m.group(1)), w)),
                                      (r"split gate/up", lambda m: saw("gated", True, w)),
                                      (r"weights stored \[in, out\]", lambda m: saw("w_layout", "in_out", w)),
                                      (r"(SiLU|GELU|ReLU|gelu_pytorch_tanh|sigmoid|tanh)", lambda m: saw("act", actl(m.group(1)), w))])
            if not _same_detail((b.get("detail") or {}).get("ffn"), L0["ffn"]):
                fail("ffn detail", "the FFN block's detail differs from the layer's FFN spec")
            kids = {c.get("id"): c for c in b.get("children") or []}
            if P["gated"] and set(kids) != {"gate_proj", "up_proj", "activation", "multiply", "down_proj"}:
                fail("ffn sub-blocks", f"gated FFN drawn with sub-blocks {sorted(kids)}")
            if not P["gated"] and kids:
                fail("ffn sub-blocks", f"dense FFN drawn with sub-blocks {sorted(kids)}")
            for cid, c in kids.items():
                cw = f"{w}.{cid}"
                if cid in ("gate_proj", "up_proj"):
                    chips(cw, c.get("facts"), [width("hidden", "inter", cw)])
                    if cid == "gate_proj":
                        text(cw, c.get("description"), [(r"Recorded matmul producing the gate path \(through (.+)\)\.", lambda m: saw("act", actl(m.group(1)), cw))])
                elif cid == "down_proj":
                    chips(cw, c.get("facts"), [width("inter", "hidden", cw)])
                elif cid == "activation":
                    text(cw + " label", c.get("label"), [(r"(.+)", lambda m: saw("act", actl(m.group(1)), cw))])
                    text(cw + " title", c.get("title"), [(r"(.+)", lambda m: saw("act", actl(m.group(1)), cw))])
                elif cid == "multiply":
                    text(cw, c.get("description"), [(r"Recorded (.+)\(gate\) × up\.", lambda m: saw("act", actl(m.group(1)), cw))])
                else:
                    unread(cw, f"FFN sub-block {cid!r} is outside the oracle's grammar")
        elif bid in ("add1", "add2"):
            text(w + " description", b.get("description"), [
                (r"Recorded add: block input \+ attention output\.", lambda m: saw("residual", "sequential", w)),
                (r"Recorded add: post-attention \+ FFN output\.", lambda m: saw("residual", "sequential", w)),
                (r"Recorded adds: layer input \+ attention output \+ FFN output\.", lambda m: saw("residual", "parallel", w))])
        else:
            unread(w, f"layer block {bid!r} is outside the oracle's grammar")

    # ---- the model blocks
    mb = {b.get("id"): b for b in ((d.get("extras") or {}).get("render") or {}).get("model_blocks") or []}
    base_mb, pos_mb = {"tok_text", "embed", "final_rms", "lm_head"}, {"position_ids", "position_embed", "position_add"}
    if set(mb) not in (base_mb, base_mb | pos_mb):
        unread("model blocks", f"model blocks {sorted(mb)}: the oracle reads tok_text / embed / final_rms / lm_head "
                               "(+ position_ids / position_embed / position_add)")
    saw("pos_emb", pos_mb <= set(mb), "model blocks")
    if pos_mb <= set(mb):
        pe = mb["position_embed"]
        text("position_embed label", pe.get("label"), [(r"Learned Position Embedding", lambda m: None)])
        chips("position_embed", pe.get("facts"), [(rf"{_N} positions", lambda m: saw("pos_rows", _int(m.group(1)), "position_embed")),
                                                  (rf"{_N}-d", lambda m: saw("hidden", _int(m.group(1)), "position_embed"))])
        if ((mb["position_add"].get("detail") or {}).get("position_kind") != "learned_absolute"
                or (mb["position_add"].get("detail") or {}).get("position_application") != "embedding_add"):
            fail("position embedding", "the position add's detail is not a learned absolute embedding add")
    if "lm_head" in mb:
        saw("head_bias", "with bias" in (mb["lm_head"].get("facts") or []), "lm_head")
    def tie_of(desc, prefix, where):
        if desc is None or not desc.startswith(prefix):
            unread(where, f"{where} description {desc!r} is outside the grammar"); return
        t = desc[len(prefix):]
        if t not in TIE_TXT: unread(where, f"{where} tie text {t!r} is outside the grammar"); return
        saw("tied", TIE_TXT[t], where)
    if "embed" in mb:
        chips("embed", mb["embed"].get("facts"), [(rf"{_N} vocab", lambda m: saw("vocab", _int(m.group(1)), "embed")),
                                                 (rf"{_N}-d", lambda m: saw("hidden", _int(m.group(1)), "embed"))])
        tie_of(mb["embed"].get("description"), "Maps each token id to its vector", "embed")
    if "lm_head" in mb:
        chips("lm_head", mb["lm_head"].get("facts"), [width("hidden", "vocab", "lm_head")], free=("with bias",))
        tie_of(mb["lm_head"].get("description"), "Recorded matmul into vocabulary logits", "lm_head")
    if "final_rms" in mb:
        text("final_rms label", mb["final_rms"].get("label"), [(r"Final (RMSNorm|LayerNorm)", lambda m: saw("final_norm", NK_R[m.group(1)], "final_rms"))])
        chips("final_rms", mb["final_rms"].get("facts"), [(rf"dim {_N}", lambda m: saw("hidden", _int(m.group(1)), "final_rms"))])
        text("final_rms description", mb["final_rms"].get("description"),
             [(r"Recorded (RMSNorm|LayerNorm) over the last hidden state before the output head\.", lambda m: saw("final_norm", NK_R[m.group(1)], "final_rms"))])

    # ---- the EXECUTION EVIDENCE card
    CARD = {"model.vocab_size": "vocab", "model.hidden_size": "hidden", "model.num_hidden_layers": "layers",
            "model.final_norm_kind": "final_norm", "decoder.attention.bias": "qkv_bias", "decoder.attention.num_heads": "num_heads",
            "decoder.attention.num_kv_heads": "num_kv", "decoder.attention.head_dim": "head_dim", "decoder.attention.mask": "mask",
            "decoder.attention.qk_norm": "qk_norm", "decoder.attention.o_bias": "o_bias",
            "decoder.ffn.intermediate_size": "inter", "decoder.ffn.activation": "act", "decoder.ffn.gated": "gated",
            "decoder.layer.norm_kind": "norm", "decoder.layer.residual_topology": "residual"}
    for x in d["extras"]["execution_evidence"]["facts"]:
        k, v = x["key"], x["value"]
        if k in CARD: saw(CARD[k], v, f"card {k}")
        elif k == "decoder.attention.rope_theta": saw("theta~card", v, f"card {k}")
        elif k == "decoder.attention.scores_scale": saw("scale~card", v, f"card {k}")
        elif k == "model.tie_word_embeddings":
            if isinstance(v, str) and ("unverified" in v or "unchecked" in v): v = "unverified"
            saw("tied", v, f"card {k}")
        elif k == "decoder.attention.qkv_split":
            m = re.fullmatch(r"\[Q ([\d,]+) \| K ([\d,]+) \| V ([\d,]+)\]|per K/V group \[q × (\d+) \| k \| v\] × (\d+), head dim (\d+)", str(v))
            if not m: unread(f"card {k}", f"the card's Q/K/V split {v!r} is outside the grammar")
            elif m.group(1): saw("qkv_split", ("qkv_blocks", _int(m.group(1)), _int(m.group(2)), _int(m.group(3))), f"card {k}")
            else: saw("qkv_split", ("grouped", int(m.group(4)), int(m.group(5)), int(m.group(6))), f"card {k}")
        elif k == "decoder.layer.weight_layout":
            saw("w_layout", "in_out" if v == "stored [in, out]" else v, f"card {k}")
        elif k == "decoder.attention.rotary_dim": saw("rot_dim", v, f"card {k}")
        elif k == "decoder.layer.parallel_norm_count": saw("parallel_norms", v, f"card {k}")
        elif k == "model.position_embedding": saw("pos_emb", v == "learned absolute", f"card {k}")
        elif k == "model.lm_head_bias": saw("head_bias", v is True, f"card {k}")
        elif k in ("decoder.attention.cached", "model.max_position_embeddings"):
            pass                                           # UNCHECKED
        else:
            unread(f"card {k}", f"the evidence card shows {k} = {v!r}, which the oracle does not read")

    # ---- every text fact must equal the spec
    tied_spec = "unverified" if P["tied"] is None else P["tied"]
    derived = {"q_out": nh * P["head_dim"] if nh and P["head_dim"] else None,
               "k_out": nkv * P["head_dim"] if nkv and P["head_dim"] else None,
               "v_out": nkv * P["head_dim"] if nkv and P["head_dim"] else None,
               "o_in": nh * P["head_dim"] if nh and P["head_dim"] else None,
               "q_per_kv": nh // nkv if nh and nkv else None, "tied": tied_spec}
    def want(fact):
        if fact in derived: return derived[fact]
        return P.get(fact)
    # facts only the text / card state (no spec field): every mention must agree; the agreed value is the page fact
    TEXT_ONLY = ("qkv_split", "w_layout", "pos_emb", "head_bias", "pos_rows", "qkv_width")
    for f_ in TEXT_ONLY:
        vs = {v for x, v, _ in seen if x == f_}
        if len(vs) > 1: fail(f_, f"the page states {f_} as {sorted(map(str, vs))}")
        elif vs: P[f_] = next(iter(vs))
    seen[:] = [x for x in seen if x[0] not in TEXT_ONLY]
    if P["fused"] != (P["qkv_split"] is not None):
        fail("qkv split", f"the spec says {'a fused' if P['fused'] else 'separate'} Q/K/V projection(s); the text "
                          f"{'states no split' if P['qkv_split'] is None else 'states a fused split'}")
    elif P["qkv_split"] and nh and nkv and P["head_dim"]:
        hd_ = P["head_dim"]
        want_sp = ((nh * hd_, nkv * hd_, nkv * hd_) if P["qkv_split"][0] == "qkv_blocks" else (nh // nkv, nkv, hd_))
        if tuple(P["qkv_split"][1:]) != want_sp or (P.get("qkv_width") not in (None, (nh + 2 * nkv) * hd_)):
            fail("qkv split", f"the stated split {P['qkv_split']} / width {P.get('qkv_width')} disagrees with {nh} Q / "
                              f"{nkv} K-V heads of {hd_}")
    if P["rot_dim"] is not None and not (P["rope"] and 0 < P["rot_dim"] < (P["head_dim"] or 0)):
        fail("rot_dim", f"partial rotary {P['rot_dim']} of {P['head_dim']} head dims")
    ex = [v for f_, v, _ in seen if f_ == "scale~exact"]      # an exact form in the text is the stated value
    if ex and P["scale"] is not None and abs(ex[0] - P["scale"]) <= 5.01e-7: P["scale"] = ex[0]
    for fact, v, where in seen:
        if fact == "qk_norm_before_rope":
            continue                                       # text-only fact: read below
        if fact == "theta~int":
            ok = P["theta"] is not None and abs(v - round(P["theta"])) < 0.5
        elif fact in ("theta~6g", "theta~card"):
            ok = (v is None and P["theta"] is None) or (v is not None and P["theta"] is not None
                                                       and abs(v - P["theta"]) <= 1e-5 * abs(P["theta"]))
        elif fact == "scale~card":                       # the card rounds the scale to 6 decimals (to_ir)
            ok = (v is None and P["scale"] is None) or (v is not None and P["scale"] is not None
                                                       and abs(v - P["scale"]) <= 5.01e-7)
        elif fact in ("scale~6g", "scale~exact"):
            ok = (v is None and P["scale"] is None) or (v is not None and P["scale"] is not None
                                                       and abs(v - P["scale"]) <= 1e-5 * abs(P["scale"]))
        elif fact == "scale":
            ok = P["scale"] is None
        else:
            ok = v == want(fact)
        if not ok:
            fail(fact.split("~")[0], f"{where} shows {fact.split('~')[0]} = {v!r}, the spec says "
                                     f"{want(fact.split('~')[0]) if fact.split('~')[0] != 'theta' else P['theta']!r}")
    if not P["scores_scaled"] and any(f_.startswith("scale~") for f_, _, _ in seen):
        fail("scale", "the spec says the scores are not scaled; the page text shows a scale")
    if not P["scores_scaled"]: P["scale"] = None
    if P["scale"] is None and P["scores_scaled"]:
        fail("scale", "the spec says the scores are scaled but no scale value is shown")
    # facts only the text states (no spec field): the reference uses what the text says
    rb = {v for f_, v, _ in seen if f_ == "qk_norm_before_rope"}
    if len(rb) > 1: fail("qk_norm order", "the Q and K norm descriptions disagree on the order with RoPE")
    P["qk_norm_before_rope"] = (next(iter(rb)) if rb else None) if P["qk_norm"] else None
    P["rope_form"] = "rotate_half" if P["rope"] else None          # the only rotation text the grammar reads
    return P, issues
