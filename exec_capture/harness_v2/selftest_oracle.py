"""The oracle's own control (P4, Track D): it must REFUSE drawings known to be wrong.

  A. the 8 KNOWN_OPEN mutants of selftest_mutants.py: the recognizer DRAWS each of them (a false drawing); the oracle
     must FAIL every one (the real model runs the mutant code, the reference runs the page).
  B. hand-made wrong page facts on correct drawings: wrong RoPE theta, pre/post-norm swapped, GQA grouping tiled,
     wrong head split, wrong activation, wrong score scale, wrong norm kind, QK-norm dropped, Q/K/V bias dropped: each
     must FAIL. Applied to the tiny controls here; run_families.py applies them to real family rows too.
  C. controls: the unchanged tiny Llama / Qwen3 / MHA models must PASS (else the oracle refuses everything).
  D. (information only) every other selftest_mutants mutant: what the recognizer did and what the oracle says.

    python3 selftest_oracle.py [--json OUT] [--all]      # exit 0 = A all FAIL, B all FAIL, C all PASS
"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = sys.argv; sys.argv = [_argv[0]]
import selftest_mutants as SM            # its module body only defines mutants (main() runs under __main__)
sys.argv = _argv
import torch, transformers, capture, exec_to_ir, oracle


def mutant_row(name, n_seeds=2, **edits):
    patch, bkw = SM.MUT[name]
    SM.restore()
    torch.manual_seed(0)
    fam = bkw.get("fam", "llama")
    S = bkw.get("S", 12)
    kw_ = dict(vocab_size=32, hidden_size=64, intermediate_size=96, num_hidden_layers=bkw.get("layers", 3),
               num_attention_heads=bkw.get("heads", 4), num_key_value_heads=bkw.get("kv", 2), head_dim=bkw.get("hd", 16),
               max_position_embeddings=128, **({"rope_parameters": bkw["rope"]} if bkw.get("rope") else {}))
    c = (transformers.LlamaConfig if fam == "llama" else transformers.Qwen3Config)(**kw_)
    cls = transformers.LlamaForCausalLM if fam == "llama" else transformers.Qwen3ForCausalLM
    m = cls._from_config(c, attn_implementation="eager", dtype=torch.float32).eval()
    try:
        if patch: patch(m)
        # the recording + bundle exactly as selftest_mutants.bundle_of makes it (S tokens, ids 1..S)
        kw = {"input_ids": torch.tensor([list(range(1, bkw.get("S", 5) + 1))]), "use_cache": False}
        r = capture.CaptureRecorder(m, kw, skip=False)
        try:
            with torch.no_grad(), r:
                m(**kw)
        finally:
            r.remove_hooks()
        b = {"format": capture.FORMAT, "repo": f"selftest/{name}", "class": type(m).__name__, "config": c.to_dict(),
             "model": capture.model_facts(m),
             "passes": {"text": {"graph": capture.graph(r), "squares": capture.squares(r), "inputs": capture.input_shapes(kw)}},
             "pass_order": ["text"], "verdict": "FULL", "checks": {}, "max_position_embeddings": 128}
        b["model"]["tie_groups_shipped"] = []
        m.double()                                     # the patch stays active: the real model runs the mutant code
        row = oracle.run_bundle(b, model=m, S=S, seeds=tuple(range(n_seeds)), **edits)
    finally:
        SM.restore()
    row["mutant"] = name
    return row


# ---------------------------------------------------------------------- hand-made wrong page facts
# Each is a self-consistent wrong PAGE: the recognizer facts are edited before to_ir (f), or the rendered IR is edited
# so that spec AND text say the wrong thing (ir). The functional check must catch it, not only the page reader.
def _theta(F): F["theta"] = F["theta"] * 10
def _post(ir):                                      # the layer drawn post-norm: placement + block order + skips
    import dataclasses
    for L in ir.layers:
        b = {x["id"]: dict(x) for x in L.blocks}
        b["add1"]["residual_from"] = "attn"; b["add2"]["residual_from"] = "ffn"
        L.blocks = [b["attn"], b["add1"], b["rms1"], b["ffn"], b["add2"], b["rms2"]]
        L.norm_placement = "post"
def _split(F): F["num_heads"] *= 2; F["num_kv"] *= 2; F["head_dim"] //= 2
def _act(F): F["act"] = "gelu" if F["act"] != "gelu" else "silu"
def _scale(F): F["scale"] = 1.0 / F["head_dim"]
def _normkind(F): F["norm"] = "layernorm" if F["norm"] == "rmsnorm" else "rmsnorm"
def _final(F): F["final_norm"] = "layernorm" if F["final_norm"] == "rmsnorm" else "rmsnorm"
def _noqk(F): F.update(qk_norm=False, qk_norm_kind=None, qn_extent=None, norm_before_rope=None)
def _nobias(F): F["bias"] = False
def _nol3(F): F["rope_note"] = None
WRONG = {   # name: (kind "f" | "ir", edit, applies-to test on the correct page's facts)
    "RoPE theta x10": ("f", _theta, lambda P: P["rope"]),
    "pre-norm drawn as post-norm": ("ir", _post, lambda P: True),
    "head split wrong (2x heads, half head_dim)": ("f", _split, lambda P: True),
    "activation swapped (silu <-> gelu)": ("f", _act, lambda P: True),
    "score scale 1/d instead of 1/sqrt(d)": ("f", _scale, lambda P: P["scale"] is not None),
    "layer norm kind swapped (RMS <-> Layer)": ("f", _normkind, lambda P: True),
    "final norm kind swapped": ("f", _final, lambda P: True),
    "QK-norm dropped": ("f", _noqk, lambda P: P["qk_norm"]),
    "Q/K/V bias dropped": ("f", _nobias, lambda P: P["qkv_bias"]),
    "llama3 frequency rescale dropped": ("f", _nol3, lambda P: P["llama3"]),
}


def edit_kw(kind, fn):
    return {"f_edit": fn} if kind == "f" else {"ir_edit": fn}


def _no_oproj(m):                                     # a model whose attention has no output projection (MHA: widths fit)
    for lay in m.model.layers: lay.self_attn.o_proj = torch.nn.Identity()
SM.MUT["NO-O_PROJ llama MHA (control: drawn without one)"] = (_no_oproj, {"kv": 4})
def _draw_oproj(ir):                                  # the wrong page: an output projection drawn anyway (spec + text)
    import dataclasses
    for L in ir.layers:
        L.attention = dataclasses.replace(L.attention, output_projection=True)
        for b in L.blocks:
            if b["id"] == "attn":
                b["children"] = list(b["children"]) + [{"id": "o_proj", "title": "Output projection",
                    "description": "Recorded matmul back to the residual width.", "facts": ["64 → 64"]}]
                b["detail"] = {"attention": L.attention.to_dict() if hasattr(L.attention, "to_dict") else dataclasses.asdict(L.attention)}


def applicable(P, test):
    try:
        return bool(test(P))
    except Exception:
        return False


def main():
    out = {"A_known_open": [], "B_wrong_facts": [], "C_controls": [], "D_other_mutants": []}
    bad = []
    def show(tag, row, want):
        v = row["verdict"]
        got = "PASS" if v == "PASS" else ("FAIL" if v.startswith("FAIL") else v)
        ok = got == want
        if not ok: bad.append((tag, row.get("mutant"), v))
        e2e = row.get("end_to_end")
        print(f"{'ok  ' if ok else 'BAD '} [{tag}] {row.get('mutant')}{' / ' + row['wrong_fact'] if row.get('wrong_fact') else ''}: "
              f"{v}{'' if e2e is None else f'  (end-to-end rel err {e2e:.2e})'}"
              f"{'  facts: ' + row['first_fail_facts'] if row.get('first_fail_facts') else ''}"
              f"{'  reason: ' + row['reason'][:140] if row.get('reason') else ''}", flush=True)
    for name in [n for n in SM.MUT if n.startswith("CONTROL")]:
        r = mutant_row(name); out["C_controls"].append(r); show("C control", r, "PASS")
    for name in sorted(SM.KNOWN_OPEN):
        r = mutant_row(name); out["A_known_open"].append(r)
        if r["verdict"] == "NOT_DRAWN":
            print(f"note [A] {name}: the recognizer now refuses it ({r['reason'][:100]})")
        show("A KNOWN_OPEN", r, "FAIL")
    for ctl in ("CONTROL llama", "CONTROL qwen3", "CONTROL llama MHA S=16"):
        base = mutant_row(ctl)
        for wname, (kind, edit, test) in WRONG.items():
            if not applicable(base.get("page_facts", {}), test): continue
            r = mutant_row(ctl, **edit_kw(kind, edit)); r["wrong_fact"] = wname
            out["B_wrong_facts"].append(r)
            # a self-consistent wrong page: the FUNCTIONAL check must refuse it (not only the page reader)
            if r["verdict"].startswith("FAIL(page"): r["verdict"] = "PAGE_INCONSISTENT(" + r["verdict"] + ")"
            show("B wrong fact", r, "FAIL")
    r = mutant_row("NO-O_PROJ llama MHA (control: drawn without one)"); out["C_controls"].append(r); show("C control", r, "PASS")
    r = mutant_row("NO-O_PROJ llama MHA (control: drawn without one)", ir_edit=_draw_oproj)
    r["wrong_fact"] = "output projection drawn on a model without one"; out["B_wrong_facts"].append(r)
    show("B wrong fact (static)", r, "FAIL")
    if "--all" in sys.argv:
        for name in SM.MUT:
            if name.startswith(("CONTROL", "NO-O_PROJ")) or name in SM.KNOWN_OPEN: continue
            r = mutant_row(name); r["benign"] = name in SM.BENIGN; out["D_other_mutants"].append(r)
            print(f"info [D {'BENIGN' if name in SM.BENIGN else 'refused-by-recognizer expected'}] {name}: {r['verdict']}"
                  f"{'' if r.get('end_to_end') is None else ' (e2e %.2e)' % r['end_to_end']}"
                  f"{'  facts: ' + r['first_fail_facts'] if r.get('first_fail_facts') else ''}"
                  f"{'  ' + r['reason'][:110] if r.get('reason') else ''}", flush=True)
    print(f"\n{len(bad)} unexpected: {bad}")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=str)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
