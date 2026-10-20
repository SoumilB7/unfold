"""Constructed-model controls for family group 3+4 (sliding windows, mixed layer kinds, soft-capping, fused gate/up).

The real rows with 2047 / 4096-token windows cannot run a sequence longer than their window on this machine, so each
fact kind is ALSO proven on a tiny model built from the library's own config class with a small window (6): the model
is recorded exactly as the worker records it, drawn by exec_to_ir, and checked by the oracle with a run longer than the
window. Each control must be drawn and PASS; each wrong page (one per new fact kind) must FAIL.
    python3 selftest_oracle_families.py [--json OUT]        # exit 0 = as expected
"""
import os, sys, json, copy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch, transformers, capture, exec_to_ir, oracle
import transformers.masking_utils as MU
from transformers.models.gemma2 import modeling_gemma2 as G2

W = 6                                                  # the constructed window
BASE = dict(vocab_size=64, hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=256)
MODELS = {
    "mistral (uniform sliding)": (transformers.MistralConfig, transformers.MistralForCausalLM,
                                  dict(num_hidden_layers=2, sliding_window=W)),
    "phi3 (sliding + fused QKV + fused gate/up)": (transformers.Phi3Config, transformers.Phi3ForCausalLM,
                                                   dict(num_hidden_layers=2, sliding_window=W, pad_token_id=0)),
    "gemma2 (sliding / full + soft-caps)": (transformers.Gemma2Config, transformers.Gemma2ForCausalLM,
                                            dict(num_hidden_layers=4, sliding_window=W, attn_logit_softcapping=50.0,
                                                 final_logit_softcapping=30.0, query_pre_attn_scalar=16,
                                                 layer_types=["sliding_attention", "full_attention"] * 2)),
    "gemma3 (5 sliding : 1 full, local / global theta, QK-norm 1+w)": (
        transformers.Gemma3TextConfig, transformers.Gemma3ForCausalLM,
        dict(num_hidden_layers=6, sliding_window=W, query_pre_attn_scalar=16,
             layer_types=["sliding_attention"] * 5 + ["full_attention"],
             rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                              "full_attention": {"rope_type": "default", "rope_theta": 1000000.0}})),
    # Opus g3 review: 12 layers, two sliding / full cycles; also run SHRUNK (2 layers asked: the oracle builds up to the
    # first layer of each kind, 6, so layers 6..11 are checked only by the meta-device per-layer configuration)
    "gemma3 x12 (A5 B1 A5 B1)": (
        transformers.Gemma3TextConfig, transformers.Gemma3ForCausalLM,
        dict(num_hidden_layers=12, sliding_window=W, query_pre_attn_scalar=16,
             layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 2,
             rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                              "full_attention": {"rope_type": "default", "rope_theta": 1000000.0}})),
    # soft-cap c = 5: the order relative to the mask add changes the function (at c = 50 it does not, measurably)
    "gemma2 c=5 (soft-cap before the mask)": (transformers.Gemma2Config, transformers.Gemma2ForCausalLM,
                                              dict(num_hidden_layers=2, sliding_window=W, attn_logit_softcapping=5.0,
                                                   final_logit_softcapping=None, query_pre_attn_scalar=16,
                                                   layer_types=["sliding_attention", "full_attention"])),
    "gemma2 c=5 (soft-cap AFTER the mask, patched eager)": (transformers.Gemma2Config, transformers.Gemma2ForCausalLM,
                                              dict(num_hidden_layers=2, sliding_window=W, attn_logit_softcapping=5.0,
                                                   final_logit_softcapping=None, query_pre_attn_scalar=16,
                                                   layer_types=["sliding_attention", "full_attention"])),
    # Opus note 3: a window written as a COMPARISON constant ((q - kv) < 80), no add / sub constant: the replay must
    # run longer than 80 (the old rule replayed at 64 and saw full causal)
    "mistral window 80 as a comparison constant (patched overlay)": (
        transformers.MistralConfig, transformers.MistralForCausalLM, dict(num_hidden_layers=2, sliding_window=80)),
    # Opus note: a soft-cap whose float prints long (50 * 1.1 = 55.00000000000001; the chip shows 55): read as one value
    "gemma2 c=50*1.1 (soft-cap float formatting)": (transformers.Gemma2Config, transformers.Gemma2ForCausalLM,
                                                    dict(num_hidden_layers=2, sliding_window=W, attn_logit_softcapping=50 * 1.1,
                                                         final_logit_softcapping=30 * 1.1, query_pre_attn_scalar=16,
                                                         layer_types=["sliding_attention", "full_attention"])),
    # Fable note 2: a TRUE per-head window (head 0: 6, the others: 4) as one 4-D where inside eager: a named refusal
    "mistral per-head window (patched eager)": (transformers.MistralConfig, transformers.MistralForCausalLM,
                                                dict(num_hidden_layers=2, sliding_window=W)),
}
EXPECT_REFUSAL = {"mistral per-head window (patched eager)": "differs across heads"}

# library patches a constructed model runs under (record AND oracle run): its function is the patched one
_ORIG_EAGER, _ORIG_OVERLAY = G2.eager_attention_forward, MU.sliding_window_overlay
def _eager_cap_after(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kw):
    ks, vs = G2.repeat_kv(key, module.num_key_value_groups), G2.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None: w = w + attention_mask
    if softcap is not None: w = torch.tanh(w / softcap) * softcap
    w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(w, vs).transpose(1, 2).contiguous(), w
from transformers.models.mistral import modeling_mistral as MMis
_ORIG_MIS_EAGER = MMis.eager_attention_forward
def _eager_per_head(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    ks, vs = MMis.repeat_kv(key, module.num_key_value_groups), MMis.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, ks.transpose(2, 3)) * scaling; S_ = w.shape[-1]
    h = torch.arange(w.shape[1])[None, :, None, None]
    q = torch.arange(S_)[None, None, :, None]; kv = torch.arange(S_)[None, None, None, :]
    w = w + torch.where((kv <= q) & (kv > q - W) & ((h == 0) | (kv > q - 4)), 0.0, torch.finfo(w.dtype).min)
    w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(w, vs).transpose(1, 2).contiguous(), w
def _overlay_cmp(sw):
    return lambda b, h, q, kv: (q - kv) < sw
PATCH = {"gemma2 c=5 (soft-cap AFTER the mask, patched eager)": lambda on: setattr(G2, "eager_attention_forward", _eager_cap_after if on else _ORIG_EAGER),
         "mistral window 80 as a comparison constant (patched overlay)": lambda on: setattr(MU, "sliding_window_overlay", _overlay_cmp if on else _ORIG_OVERLAY),
         "mistral per-head window (patched eager)": lambda on: setattr(MMis, "eager_attention_forward", _eager_per_head if on else _ORIG_MIS_EAGER)}


def record(name, S=5):
    cfg_cls, cls, extra = MODELS[name]
    torch.manual_seed(0)
    c = cfg_cls(**dict(BASE, **extra))
    m = cls._from_config(c, attn_implementation="eager", dtype=torch.float32).eval()
    kw = {"input_ids": torch.tensor([list(range(1, S + 1))]), "attention_mask": torch.ones(1, S, dtype=torch.long),
          "use_cache": False}
    r = capture.CaptureRecorder(m, kw, skip=False)
    try:
        with torch.no_grad(), r:
            m(**kw)
    finally:
        r.remove_hooks()
    b = {"format": capture.FORMAT, "repo": f"constructed/{name}", "class": type(m).__name__, "config": c.to_dict(),
         "model": capture.model_facts(m),
         "passes": {"text": {"graph": capture.graph(r), "squares": capture.squares(r), "inputs": capture.input_shapes(kw)}},
         "pass_order": ["text"], "verdict": "FULL", "checks": {}, "max_position_embeddings": 256}
    b["model"]["tie_groups_shipped"] = []
    return b, m.double()


def _kinds(F, fn):
    F["layer_kinds"] = [fn(dict(k)) for k in F["layer_kinds"]]
def w_plus1(F): _kinds(F, lambda k: dict(k, window=k["window"] + 1) if k["window"] else k)
def w_minus1(F): _kinds(F, lambda k: dict(k, window=k["window"] - 1) if k["window"] else k)
def w_dropped(F): _kinds(F, lambda k: dict(k, mask="causal", window=None) if k["window"] else k)
def pattern_rolled(F): F["kind_of_layer"] = F["kind_of_layer"][1:] + F["kind_of_layer"][:1]
def theta_swapped(F):
    th = [k["theta"] for k in F["layer_kinds"]]
    F["layer_kinds"] = [dict(k, theta=th[(n + 1) % len(th)]) for n, k in enumerate(F["layer_kinds"])]
def acap_x2(F): F["attn_softcap"] *= 2
def acap_drop(F): F["attn_softcap"] = None
def fcap_x2(F): F["final_softcap"] *= 2
def fcap_drop(F): F["final_softcap"] = None
def gu_swap(F): F["gate_up"] = dict(F["gate_up"], layout="up_gate" if F["gate_up"]["layout"] == "gate_up" else "gate_up")
def late_swap(F):            # the last layer and the last layer of another kind trade kinds (late: unbuilt on shrunk rows)
    k = list(F["kind_of_layer"]); i2 = len(k) - 1; i1 = max(i for i in range(i2) if k[i] != k[i2])
    k[i1], k[i2] = k[i2], k[i1]; F["kind_of_layer"] = k
def swap_11_20(F):           # Opus g3 real-row case: layer 11 and layer 20 trade kinds
    k = list(F["kind_of_layer"]); k[11], k[20] = k[20], k[11]; F["kind_of_layer"] = k
def cap_order_flip(F): F["attn_softcap_order"] = "after" if F["attn_softcap_order"] == "before" else "before"
WRONG = {   # name: (edit, applies-to test on the recognizer facts)
    "sliding window +1": (w_plus1, lambda F: any(k["window"] for k in F["layer_kinds"])),
    "sliding window -1": (w_minus1, lambda F: any(k["window"] for k in F["layer_kinds"])),
    "sliding window drawn as full causal": (w_dropped, lambda F: any(k["window"] for k in F["layer_kinds"])),
    "layer pattern shifted by one layer": (pattern_rolled, lambda F: len(F["layer_kinds"]) > 1),
    "per-kind RoPE bases swapped": (theta_swapped, lambda F: len({k["theta"] for k in F["layer_kinds"]}) > 1),
    "attention soft-cap x2": (acap_x2, lambda F: F["attn_softcap"]),
    "attention soft-cap dropped": (acap_drop, lambda F: F["attn_softcap"]),
    "final soft-cap x2": (fcap_x2, lambda F: F["final_softcap"]),
    "final soft-cap dropped": (fcap_drop, lambda F: F["final_softcap"]),
    "fused gate/up halves swapped": (gu_swap, lambda F: F.get("gate_up")),
    "last layers' kinds swapped": (late_swap, lambda F: len(F["layer_kinds"]) > 1),
    "layers 11 and 20 kinds swapped (Opus g3)": (swap_11_20, lambda F: len(F["layer_kinds"]) > 1 and len(F["kind_of_layer"]) > 20
                                                  and F["kind_of_layer"][11] != F["kind_of_layer"][20]),
}
# constructed only: at a real model's c = 50 the two orders are the same function to e^-50 (no run can tell them apart)
WRONG_CONSTRUCTED = {"soft-cap order flipped (before <-> after the mask)": (cap_order_flip, lambda F: F["attn_softcap"] and F["attn_softcap"] <= 10)}
def kindB_text_order_flipped(F):   # page TEXT only: kind B's scores text states the other order (kind A's unchanged)
    def ed(ir):
        import copy
        for i, k in enumerate(F["kind_of_layer"]):
            if k != 1: continue
            L = ir.layers[i]; object.__setattr__(L, "blocks", copy.deepcopy(L.blocks))
            for b_ in L.blocks:
                for c in b_.get("children") or []:
                    if c.get("id") == "scaled_scores":
                        o = F["attn_softcap_order"]; c["description"] = c["description"].replace(
                            f"{o} the mask", f"{'after' if o == 'before' else 'before'} the mask")
    return ed
IR_WRONG = {"kind B's text alone states the other soft-cap order": (kindB_text_order_flipped,
                                                                    lambda F: F["attn_softcap"] and len(F["layer_kinds"]) > 1)}
def lay8_full(F): k = list(F["kind_of_layer"]); k[8] = 1; F["kind_of_layer"] = k
def lay11_sliding(F): F["kind_of_layer"] = F["kind_of_layer"][:11] + [0]
SHRUNK = {"gemma3 x12 (A5 B1 A5 B1)": [("control [shrunk to 2 asked]", None, "PASS"),
                                       ("layer 11 drawn sliding [shrunk]", lay11_sliding, "FAIL"),
                                       ("layer 8 drawn full [shrunk]", lay8_full, "FAIL")]}


def main():
    rows, bad = [], []
    def show(tag, r, want):
        v = r["verdict"]; got = "PASS" if v == "PASS" else ("FAIL" if v.startswith("FAIL") else v)  # NOT_DRAWN exact
        ok = got == want
        if not ok: bad.append((tag, v))
        rows.append({"case": tag, "want": want, "verdict": v, "reason": r.get("reason"), "end_to_end": r.get("end_to_end"),
                     "S": r.get("S"), "window_runs": r.get("window_runs")})
        e2e = r.get("end_to_end")
        print(f"{'ok  ' if ok else 'BAD '} [{tag}] want {want}: {v}{'' if e2e is None else f'  e2e {e2e:.1e}'}"
              f"  S={r.get('S')}{'  -- ' + str(r.get('reason'))[:120] if r.get('reason') else ''}", flush=True)
    only = sys.argv[sys.argv.index("--only") + 1].split("|") if "--only" in sys.argv else None
    for name in MODELS:
        if only and name not in only: continue
        if name in PATCH: PATCH[name](True)
        try:
            b, m = record(name)
            try:
                F = exec_to_ir.recognize(b)
            except exec_to_ir.NotDrawn as e:
                if name in EXPECT_REFUSAL:            # a named refusal is the expected verdict
                    show(name + " refused, named", {"verdict": "NOT_DRAWN" if EXPECT_REFUSAL[name] in str(e) else
                                                    f"NOT_DRAWN(other reason)", "reason": str(e)}, "NOT_DRAWN")
                else:
                    show(name + " (drawn?)", {"verdict": f"NOT_DRAWN: {e}"}, "PASS")
                continue
            if name in EXPECT_REFUSAL:
                show(name + " refused, named", {"verdict": "DRAWN"}, "NOT_DRAWN"); continue
            print(f"     {name}: kinds {F['layer_kinds']} pattern {F['kind_of_layer']} softcaps {F['attn_softcap']}/"
                  f"{F['final_softcap']} order {F.get('attn_softcap_order')} gate/up {F['gate_up']} qk {F['qk_weight']}")
            if name in PATCH and "comparison" in name:   # the control exercises the replay-length rule
                want_w = MODELS[name][2]["sliding_window"]
                ok_w = [k["window"] for k in F["layer_kinds"]] == [want_w]
                show(name + f" drawn window = {want_w} (no add/sub constant reaches it)",
                     {"verdict": "PASS" if ok_w else f"FAIL(drawn {F['layer_kinds']})"}, "PASS")
            show(name + " control", oracle.run_bundle(b, model=m, S=(24,)), "PASS")
            for wn, (ed, test) in list(WRONG.items()) + list(WRONG_CONSTRUCTED.items()):
                if not test(F): continue
                show(f"{name} / {wn}", oracle.run_bundle(b, model=m, S=(24,), seeds=(0, 2, 3, 4), f_edit=ed), "FAIL")
            for wn, (mk, test) in IR_WRONG.items():
                if not test(F): continue
                show(f"{name} / {wn}", oracle.run_bundle(b, model=m, S=(24,), seeds=(0, 2, 3, 4), ir_edit=mk(F)), "FAIL")
            for tag, ed, want in SHRUNK.get(name, []):
                r = oracle.run_bundle(b, n_layers=2, S=(24,), seeds=(0, 2, 3, 4), f_edit=ed)
                show(f"{name} / {tag} (built {r.get('real_layers_built')})", r, want)
        finally:
            if name in PATCH: PATCH[name](False)
    print(f"\n{len(rows) - len(bad)}/{len(rows)} as expected; unexpected: {bad}")
    if "--json" in sys.argv:
        json.dump(rows, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=str)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
