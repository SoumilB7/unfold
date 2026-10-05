"""Research probe 2: is the stored cache actually READ BACK?  (two-step decode, recording only)

Step 1 (prefill) runs the prompt with use_cache=True under the recorder: every tensor left in the returned cache
gets its slot path, its producer op, the weight it was projected by, and the evidence of positional rotation on
its path (a multiply whose other operand derives from cos/sin ops — the rotation itself, whatever code shape it
takes: rotate_half, chunk-and-subtract, complex multiply).
Step 2 (decode) feeds ONE new token with past_key_values = that cache, under a fresh recorder that knows every
stored tensor as a named input "cache:<slot>". For each attention in step 2 (score matmul feeding a softmax, and
the weights·V matmul after it) it lists which stored slots reach the matmul's operands through shape/concat ops
only (transport: no arithmetic in between). It also lists the projection matmuls each attention module ran in
step 2 (a cross-attention whose K/V come from the cache does not re-project them).
Nothing is changed in the model or the builder.  Usage: python cache_probe2.py <repo> <out.json>
"""
import os, sys, json, collections
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import noweights; noweights.install()
from lowcost import zero_storage
from dag import DagRecorder, math_attention, clear_library_caches
from inputs import build_passes
from exec_to_ir import Rec, MATMULS, SOFTMAX, VIEWS

TRANSPORT = VIEWS | {"cat", "repeat", "index_select", "index", "copy", "copy_", "new_empty", "empty_like"}


def kernel_switches(cfg):
    for sc in [cfg] + [getattr(cfg, a) for a in dir(cfg) if a.endswith("_config") and not a.startswith("_")
                       and hasattr(getattr(cfg, a, None), "to_dict")]:
        for a in list(vars(sc)):
            if a.startswith("use_") and "kernel" in a and getattr(sc, a) is True:
                setattr(sc, a, False)


def stored_tensors(pkv):
    out, seen = [], set()
    def walk(o, path, depth=0):
        if depth > 7 or id(o) in seen: return
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.dim() >= 2 and o.numel() > 0: out.append((path, o))
            return
        if isinstance(o, (list, tuple)):
            for i, x in enumerate(o): walk(x, f"{path}[{i}]", depth + 1)
        elif isinstance(o, dict):
            for k, x in o.items(): walk(x, f"{path}.{k}", depth + 1)
        elif hasattr(o, "__dict__") and not isinstance(o, torch.nn.Module):
            for k, x in vars(o).items(): walk(x, f"{path}.{k}", depth + 1)
    walk(pkv, "cache")
    return out


def rotation_evidence(R, chain):
    """cos/sin (or polar / complex multiply) ops between the projection and the stored tensor: a position-dependent
    rotation, whatever code shape it takes (rotate_half, chunk-and-subtract, complex multiply)."""
    return any(R.n[j].func in ("cos", "sin", "polar", "view_as_complex") for j in chain)


def transport_sources(R, rec, i):
    """Named inputs and ops reaching op i's operands through transport ops only."""
    N = R.n
    slots, seen, frontier = set(), set(), [i]
    while frontier:
        nxt = []
        for j in frontier:
            for e in N[j].ins:
                if e[0] == "input" and str(e[1]).startswith("cache:"): slots.add(e[1][6:])
                elif e[0] == "op" and e[1] not in seen and N[e[1]].func in TRANSPORT:
                    seen.add(e[1]); nxt.append(e[1])
        frontier = nxt
    return slots


def main(repo, out_path):
    import transformers
    R0 = {"repo": repo}
    cfg = transformers.AutoConfig.from_pretrained(repo)
    kernel_switches(cfg)
    arch = (cfg.architectures or [None])[0]
    cls = getattr(transformers, arch, None)
    R0["class"] = arch
    with zero_storage():
        try:
            model = cls._from_config(cfg, attn_implementation="eager", dtype=torch.float32)
        except (TypeError, ValueError):
            model = cls._from_config(cfg, dtype=torch.float32)
    model.eval()
    passes, _, _ = build_passes(repo, model, cfg)
    label, kw = passes[0]
    R0["pass"] = label
    clear_library_caches()
    rec1 = DagRecorder(model, kw)
    with torch.no_grad(), math_attention(), rec1:
        out1 = model(**kw, use_cache=True)
    rec1.remove_hooks()
    pkv = getattr(out1, "past_key_values", None)
    R0["cache_container"] = type(pkv).__name__ if pkv is not None else None
    R0["config_use_cache"] = getattr(cfg.get_text_config(), "use_cache", None)
    if pkv is None:
        R0["note"] = "no past_key_values returned"
        json.dump(R0, open(out_path, "w"), indent=1); return
    R1 = Rec(rec1, model); N1 = R1.n
    is_w = lambda j: N1[j].func in MATMULS and bool(R1.weights(j))
    stored = stored_tensors(pkv)
    rows = {}
    for path, t in stored:
        p = rec1.producer.get(id(t))
        row = {"shape": list(t.shape)}
        if p is not None:
            hit, chain = R1.back(p, is_w, 1500)
            ws = [w for w in R1.weights(hit) if w.endswith("weight")] if hit is not None else []
            row.update({"producer_op": N1[p].func, "module": N1[p].module, "weight": ws[0] if ws else None,
                        "rotated": rotation_evidence(R1, set(chain)),
                        "normed": any(N1[j].func in ("rsqrt", "native_layer_norm") for j in chain)})
        rows[path] = row
    R0["stored"] = rows
    # ---- step 2: one new token, cache fed back
    ids = kw.get("input_ids")
    if ids is None:
        R0["step2"] = "no input_ids (not a token decoder)"
        json.dump(R0, open(out_path, "w"), indent=1, default=str); return
    kw2 = {k: v for k, v in kw.items() if k not in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                                                      "token_type_ids", "position_ids", "cache_position")}
    kw2["input_ids"] = ids[:, -1:]
    if "attention_mask" in kw:
        kw2["attention_mask"] = torch.cat([kw["attention_mask"], kw["attention_mask"][:, -1:]], dim=1)
    if "decoder_input_ids" in kw:                      # encoder-decoder: the decoder takes the step, encoder reused
        kw2 = {"encoder_outputs": (out1.encoder_last_hidden_state,), "decoder_input_ids": kw["decoder_input_ids"][:, -1:]}
        if "attention_mask" in kw: kw2["attention_mask"] = kw["attention_mask"]
    names2 = {f"cache:{p}": t for p, t in stored}
    rec2 = DagRecorder(model, dict(kw2, **names2))
    try:
        with torch.no_grad(), math_attention(), rec2:
            model(**kw2, past_key_values=pkv, use_cache=True)
    except Exception as e:
        R0["step2_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        json.dump(R0, open(out_path, "w"), indent=1, default=str); return
    rec2.remove_hooks()
    R2 = Rec(rec2, model); N2 = R2.n
    att = []
    for s, nd in enumerate(N2):
        if nd.func not in SOFTMAX: continue
        # nearest matmul upstream of the softmax; attention only if it multiplies two activations (a router's nearest
        # matmul reads a weight, so a router softmax is not taken for attention)
        sc, _ = R2.back(s, lambda j: N2[j].func in MATMULS or N2[j].func == "einsum", 40)
        if sc is None or R2.weights(sc): continue
        av = next((c for c in range(s + 1, min(s + 60, len(N2))) if (N2[c].func in MATMULS or N2[c].func == "einsum")
                   and not R2.weights(c) and s in set(R2.back(c, lambda j: False, 80)[1])), None)
        mod = nd.module
        projs = sorted({w for j, n2 in enumerate(N2) if n2.module.startswith(mod) and n2.func in MATMULS
                        for w in R2.weights(j) if w.endswith("weight")})
        att.append({"module": mod, "key_slots": sorted(transport_sources(R2, rec2, sc)),
                    "value_slots": sorted(transport_sources(R2, rec2, av)) if av is not None else None,
                    "projections_run_in_step2": projs})
    R0["step2_attention"] = att
    used = set()
    for a in att: used |= set(a["key_slots"]) | set(a["value_slots"] or [])
    R0["stored_never_read_in_step2"] = sorted(p for p in rows if p not in used)
    json.dump(R0, open(out_path, "w"), indent=1, default=str)


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        json.dump({"repo": sys.argv[1], "error": f"{type(e).__name__}: {str(e)[:300]}",
                   "where": f"{tb[-1].filename.split('site-packages/')[-1]}:{tb[-1].lineno}" if tb else ""},
                  open(sys.argv[2], "w"), indent=1)
