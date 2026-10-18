"""Refusal controls on REAL rows (P4 oracle): wrong pages from the two independent checks of the oracle, 2026-10-18.

  E. Opus's cases (model-benchmark/_reruns/d1_oracle/check_opus): the recognizer facts or the rendered IR are edited so
     the page states something wrong (sizes, biases, tie, layer count, kind, block order, per-layer differences, display
     text); plus a drawn vocabulary larger than the real one, and exact-vs-tanh GELU on a tanh-GELU model.
  F. Fable's probes (model-benchmark/_reruns/d1_oracle/check_fable/probe.py): IR mutations on 4 PASS rows. Fable's two
     evidence-swap probes edited extras.exec_capture.evidence, which the page does not render; here the swap is applied
     to the EXECUTION EVIDENCE card's source column, the place the page shows (and the oracle reads) the tensor names.
Every wrong page must FAIL; the controls must PASS.
    python3 selftest_oracle_pages.py <captures-dir> [--json OUT]       # exit 0 = as expected
"""
import os, sys, json, copy, time, dataclasses
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch, capture, exec_to_ir, oracle

CAP = sys.argv[1]
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
load = lambda slug: capture.load(os.path.join(CAP, slug + ".capture.json.gz"))
rows, bad = [], []


def run(tag, B, model, want, S=24, seeds=(0,), **edits):
    t0 = time.time()
    try:
        r = oracle.run_bundle(B, model=model, S=S, seeds=seeds, **edits)
    except Exception as e:
        r = {"verdict": f"ORACLE_CRASH({type(e).__name__}: {str(e)[:120]})"}
    v = r["verdict"]
    got = "PASS" if v == "PASS" else ("FAIL" if v.startswith("FAIL") else v)
    ok = got == want
    if not ok: bad.append((tag, v))
    rows.append({"case": tag, "repo": B["repo"], "want": want, "verdict": v, "reason": r.get("reason"),
                 "end_to_end": r.get("end_to_end"), "first_fail": r.get("first_fail")})
    e2e = r.get("end_to_end")
    print(f"{'ok  ' if ok else 'BAD '} [{tag}] want {want}: {v}{'' if e2e is None else f'  e2e {e2e:.1e}'}"
          f"{'  -- ' + str(r.get('reason'))[:130] if r.get('reason') else ''}  ({time.time() - t0:.0f}s)", flush=True)


def F_(**kv):
    return {"f_edit": lambda F: F.update(kv)}


def IR_(fn):
    return {"ir_edit": fn}


def card_set(ir, key, value):
    for f in ir.extras["execution_evidence"]["facts"]:
        if f["key"] == key: f["value"] = value; return
    raise KeyError(key)


def card_src_swap(key, a, b):
    def e(ir):
        for f in ir.extras["execution_evidence"]["facts"]:
            if f["key"] == key:
                f["source"] = f["source"].replace(a, "@@").replace(b, a).replace("@@", b)
    return e


def kid(ir, cid):
    for b in ir.layers[0].blocks:
        for c in b.get("children", []) or []:
            if c["id"] == cid: return c
    raise KeyError(cid)


# ------------------------------------------------------------------------------------------- E. Opus
def opus():
    B = load("HuggingFaceTB__SmolLM2-135M"); m = oracle.build_real(B, 2)
    run("E smol control", B, m, "PASS")
    run("E smol hidden 640", B, m, "FAIL", **F_(hidden=640))
    run("E smol vocab 32000", B, m, "FAIL", **F_(vocab=32000))
    run("E smol vocab 60000 (larger than real)", B, m, "FAIL", **F_(vocab=60000))
    run("E smol layers 12", B, m, "FAIL", **F_(layers=12))
    run("E smol FFN width 2048", B, m, "FAIL", **F_(inter=2048))
    run("E smol untied (model ties)", B, m, "FAIL", **F_(tied=False))
    run("E smol Q/K/V bias added", B, m, "FAIL", **F_(bias=True))
    run("E smol o_proj bias added", B, m, "FAIL", **F_(o_bias=True))
    run("E smol all biases added", B, m, "FAIL", **F_(bias=True, o_bias=True, ffn_bias=True))
    run("E smol kind mha (9 Q / 3 KV)", B, m, "FAIL", **F_(kind="mha"))
    run("E smol K/V width 384", B, m, "FAIL", **F_(k_out=384, v_out=384))
    def blk_swap(ir):
        for L in ir.layers:
            b = L.blocks; L.blocks = [b[3], b[4], b[5], b[0], b[1], b[2]]
    run("E smol block order FFN first", B, m, "FAIL", **IR_(blk_swap))
    def lay_diff(ir):
        for L in ir.layers[2:]:
            L.attention = dataclasses.replace(L.attention, rope_theta=10000.0)
            L.ffn = dataclasses.replace(L.ffn, activation="gelu", intermediate_size=3072)
    run("E smol layers >= 2 differ (theta / GELU / FFN 3072)", B, m, "FAIL", **IR_(lay_diff))
    def disp(ir):
        for L in ir.layers:
            for b in L.blocks:
                if b["id"] == "attn":
                    b["facts"] = [f.replace("RoPE θ 100,000", "RoPE θ 10,000") for f in b["facts"]]
                    for c in b["children"]:
                        if c["id"] in ("q_rope", "k_rope"):
                            c["facts"] = ["frequency base θ=10000"]; c["description"] = c["description"].replace("θ=100000", "θ=10000")
                if b["id"] == "ffn": b["facts"] = ["GELU", "hidden 1,536", "split gate/up"]
        card_set(ir, "decoder.attention.rope_theta", 10000.0); card_set(ir, "decoder.ffn.activation", "gelu")
    run("E smol display text theta 10,000 + GELU (spec unchanged)", B, m, "FAIL", **IR_(disp))
    run("E smol theta 100500", B, m, "FAIL", S=(24, 257), **F_(theta=100500.0))
    run("E smol QK-norm added", B, m, "FAIL", **F_(qk_norm=True, qk_norm_kind="rmsnorm", qn_extent=64, norm_before_rope=True))
    del m
    B2 = copy.deepcopy(B); B2["config"]["hidden_act"] = "gelu_pytorch_tanh"
    m2 = oracle.build_real(B2, 2)
    run("E smol(tanh-GELU model) page gelu_pytorch_tanh (control)", B2, m2, "PASS", S=(24, 257), **F_(act="gelu_pytorch_tanh"))
    run("E smol(tanh-GELU model) page exact GELU", B2, m2, "FAIL", S=(24, 257), **F_(act="gelu"))
    del m2
    Q = load("Qwen__Qwen2.5-0.5B"); mq = oracle.build_real(Q, 2)
    run("E qwen2.5 control", Q, mq, "PASS")
    run("E qwen2.5 o_proj bias added", Q, mq, "FAIL", **F_(o_bias=True))
    run("E qwen2.5 tie flipped", Q, mq, "FAIL", f_edit=lambda F: F.update(tied=not F["tied"]))


# ------------------------------------------------------------------------------------------- F. Fable
def m_untied(ir): ir.tie_word_embeddings = False
def m_tied(ir): ir.tie_word_embeddings = True
def m_qkv_bias_added(ir): ir.layers[0].attention.bias = True
def m_o_bias_added(ir): card_set(ir, "decoder.attention.o_bias", True)
def m_ffn_bias_added(ir): ir.layers[0].ffn.bias = True
def m_inter(ir): ir.layers[0].ffn.intermediate_size = 4096
def m_hidden(ir): ir.hidden_size = 1234
def m_vocab_small(ir): ir.vocab_size = ir.vocab_size // 2
def m_layers_3(ir): ir.layers = ir.layers[:3]
def m_kind_mha(ir): ir.layers[0].attention.kind = "mha"
def m_kind_mqa(ir): ir.layers[0].attention.kind = "mqa"
def m_block_order(ir):
    b = list(ir.layers[0].blocks)
    ia = next(i for i, x in enumerate(b) if x["id"] == "attn"); im = next(i for i, x in enumerate(b) if x["id"] == "ffn")
    b[ia], b[im] = b[im], b[ia]; ir.layers[0].blocks = b
def m_qk_after_rope(ir):
    c = kid(ir, "q_norm"); c["description"] = c["description"].replace("before RoPE", "after RoPE")
def m_qk_extent_whole(ir): ir.layers[0].attention.qk_norm_extent = ir.hidden_size
def m_qk_layernorm(ir): ir.layers[0].attention.qk_norm_kind = "layernorm"
def m_not_scaled(ir): ir.layers[0].attention.scores_scaled = False
def m_scale_card_only(ir):
    c = kid(ir, "scaled_scores"); c["description"] = c["description"].split(" (= ")[0] + c["description"][c["description"].index(";", c["description"].index("scores ×")):]
def m_theta_1pct(ir): ir.layers[0].attention.rope_theta = ir.layers[0].attention.rope_theta * 1.01
def m_theta_01pct(ir): ir.layers[0].attention.rope_theta = ir.layers[0].attention.rope_theta * 1.001
def m_mask_bidir(ir): ir.layers[0].attention.mask = "bidirectional"
def m_no_oproj(ir): ir.layers[0].attention.output_projection = False
def m_kv_mha(ir): ir.layers[0].attention.num_kv_heads = ir.layers[0].attention.num_heads
def m_residual_parallel(ir): ir.layers[0].residual_topology = "parallel"
FABLE = {
    "tied head drawn as UNTIED (model ties)": (m_untied, lambda P: P["tied"] is True, "FAIL"),
    "unverified tie drawn as UNTIED": (m_untied, lambda P: P["tied"] is None, "FAIL"),
    "untied head drawn as TIED": (m_tied, lambda P: P["tied"] is False, "FAIL"),
    "Q/K/V bias ADDED": (m_qkv_bias_added, lambda P: not P["qkv_bias"], "FAIL"),
    "o_proj bias ADDED": (m_o_bias_added, lambda P: not P["o_bias"], "FAIL"),
    "FFN bias ADDED": (m_ffn_bias_added, lambda P: not P["ffn_bias"], "FAIL"),
    "FFN width drawn 4096": (m_inter, lambda P: P["inter"] != 4096, "FAIL"),
    "hidden width drawn 1234": (m_hidden, lambda P: True, "FAIL"),
    "vocab drawn half": (m_vocab_small, lambda P: True, "FAIL"),
    "layer count drawn 3": (m_layers_3, lambda P: P["layers"] > 3, "FAIL"),
    "attention kind drawn mha (GQA model)": (m_kind_mha, lambda P: P["kind"] == "gqa", "FAIL"),
    "attention kind drawn mqa (GQA model)": (m_kind_mqa, lambda P: P["kind"] == "gqa", "FAIL"),
    "blocks drawn MLP before attention": (m_block_order, lambda P: True, "FAIL"),
    "QK-norm drawn after RoPE": (m_qk_after_rope, lambda P: P["qk_norm"], "FAIL"),
    "QK-norm extent drawn whole width": (m_qk_extent_whole, lambda P: P["qk_norm"], "FAIL"),
    "QK-norm kind drawn LayerNorm": (m_qk_layernorm, lambda P: P["qk_norm"], "FAIL"),
    "scores drawn unscaled": (m_not_scaled, lambda P: P["scale"] is not None, "FAIL"),
    "scale only on the 6-decimal card (sanity)": (m_scale_card_only, lambda P: P["scale"] is not None, "PASS"),
    "RoPE theta x1.01": (m_theta_1pct, lambda P: P["rope"], "FAIL"),
    "RoPE theta x1.001": (m_theta_01pct, lambda P: P["rope"], "FAIL"),
    "mask drawn bidirectional": (m_mask_bidir, lambda P: True, "FAIL"),
    "output projection drawn absent": (m_no_oproj, lambda P: P["o_proj"], "FAIL"),
    "KV heads drawn = Q heads": (m_kv_mha, lambda P: P["num_kv"] != P["num_heads"], "FAIL"),
    "residual drawn parallel": (m_residual_parallel, lambda P: True, "FAIL"),
    "card: Q/K tensors swapped": (card_src_swap("decoder.attention.num_heads", "q_proj.weight", "k_proj.weight"), lambda P: True, "FAIL"),
    "card: gate/up tensors swapped": (card_src_swap("decoder.ffn.intermediate_size", "gate_proj.weight", "up_proj.weight"), lambda P: P["gated"], "FAIL"),
}
LONG = {"RoPE theta x1.01", "RoPE theta x1.001", "layer count drawn 3"}


def fable():
    for slug in ("HuggingFaceTB__SmolLM2-135M", "Qwen__Qwen2.5-0.5B", "Qwen__Qwen3-0.6B", "TinyLlama__TinyLlama-1.1B-Chat-v1.0"):
        B = load(slug); m = oracle.build_real(B, 2)
        base = oracle.run_bundle(B, model=m, S=24, seeds=(0,))
        run(f"F {slug} control", B, m, "PASS")
        for name, (mut, test, want) in FABLE.items():
            try:
                if not test(base["page_facts"]): continue
            except Exception:
                continue
            run(f"F {slug} {name}", B, m, want, S=(24, 257) if name in LONG else 24, **IR_(mut))
        del m


# ------------------------------------------------------------------------------------------- G. Opus, group 2 (norms)
def opus_group2():
    def U(**kv): return {"f_edit": lambda F: F.update(kv)}
    def Mf(k, f): return {"f_edit": lambda F: F.update({k: f(F[k])})}
    common = [("eps x1.1", Mf("norm_eps", lambda e: e * 1.1)), ("eps x10", Mf("norm_eps", lambda e: e * 10)),
              ("eps /10", Mf("norm_eps", lambda e: e / 10)), ("final eps x10", Mf("final_eps", lambda e: e * 10)),
              ("final eps x1.1", Mf("final_eps", lambda e: e * 1.1)), ("weight none", U(norm_weight="none")),
              ("final weight none", U(final_weight="none")),
              ("weight flip", {"f_edit": lambda F: F.update(norm_weight="1+w" if F["norm_weight"] == "w" else "w")}),
              ("final weight flip", {"f_edit": lambda F: F.update(final_weight="1+w" if F["final_weight"] == "w" else "w")}),
              ("embedding scale added", U(embed_scale={"op": -1, "value": 24.0, "source": "constant"}))]
    qk = [("QK eps x10", Mf("qk_eps", lambda e: e * 10)), ("QK eps x1.1", Mf("qk_eps", lambda e: e * 1.1)),
          ("QK weight none", U(qk_weight="none")), ("QK weight 1+w", U(qk_weight="1+w"))]
    pl = lambda p_: ("placement " + p_, U(placement=p_))
    G = {"HuggingFaceTB__SmolLM2-135M": common + [pl("post"), pl("double")],
         "Qwen__Qwen3-0.6B": common + qk + [pl("double")],
         "openai-community__gpt2": common + [pl("post")],
         "allenai__OLMo-2-0425-1B": common + qk + [pl("pre"), pl("double")]}
    for slug, cases in G.items():
        B = load(slug); m = oracle.build_real(B, 2)
        run(f"G {slug} control", B, m, "PASS", S=(24, 257), seeds=(0, 1, 2, 3, 4))
        for tag, ed in cases:
            run(f"G {slug} {tag}", B, m, "FAIL", S=(24, 257), seeds=(0, 2, 3, 4), **ed)
        del m


if __name__ == "__main__":
    opus()
    fable()
    opus_group2()
    print(f"\n{len(rows) - len(bad)}/{len(rows)} as expected; unexpected: {bad}")
    if OUT: json.dump(rows, open(OUT, "w"), indent=1, default=str)
    sys.exit(1 if bad else 0)
