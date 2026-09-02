"""Baseline generation audit at HEAD: what the product generates per model vs the contract.
usage: gen_audit.py <out_dir> <name>=<corpus:slug | id:Org/Repo> ...
"""
import os, sys, json, time, traceback, collections, pathlib
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import logging; logging.disable(logging.WARNING)
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
CORPUS = pathlib.Path("/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/tests/sable_test_corpus")
sys.path.insert(0, "/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg")
UNK = {None, "unknown", "unresolved", "ambiguous", "incomplete"}
LAYER_KEYS = ["attention.kind", "attention.mixer_state", "attention.num_heads", "attention.num_kv_heads", "attention.head_dim",
              "attention.rope_theta", "attention.position_kind", "attention.position_application", "attention.qk_norm", "attention.mask",
              "attention.score_scaling", "ffn.kind", "ffn.activation", "ffn.gated", "ffn.intermediate_size", "ffn.num_experts",
              "ffn.routing", "norm_kind", "norm_placement", "residual_topology"]
def get(d, path):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict): return None
        cur = cur.get(p)
    return cur
def sig(layer):
    return tuple((k, json.dumps(get(layer, k), default=str)) for k in LAYER_KEYS)
def summarize_ir(ir):
    layers = ir.get("layers") or []
    groups = collections.OrderedDict()
    for L in layers: groups.setdefault(sig(L), []).append(L.get("index"))
    unk = collections.Counter(); total = 0
    for L in layers:
        for k in LAYER_KEYS:
            v = get(L, k); total += 1
            if v is None or (isinstance(v, str) and v.lower() in {"unknown","unresolved","ambiguous","incomplete"}): unk[k] += 1
    rd = (ir.get("extras") or {}).get("render") or {}
    blocks = rd.get("model_blocks") or []
    unresolved_blocks = [b.get("id") for b in blocks if "unresolved" in json.dumps(b).lower()]
    return {
        "architecture": ir.get("architecture"), "family": rd.get("family"), "num_layers": len(layers),
        "distinct_layer_groups": [{"indices": idx if len(idx) <= 8 else [idx[0], "...", idx[-1], f"n={len(idx)}"],
                                   "facts": {k: json.loads(v) for k, v in s}} for s, idx in groups.items()][:6],
        "layer_fact_slots": total, "unknown_slots": sum(unk.values()), "unknown_by_key": dict(unk),
        "opaque_layer_block": (rd.get("opaque_layer_block") or {}).get("title"),
        "model_blocks": [b.get("id") for b in blocks], "unresolved_model_blocks": unresolved_blocks,
        "loop_blocks": [b.get("id") for b in (rd.get("loop_blocks") or [])],
        "cross_layer_edges": ir.get("cross_layer_edges") or [], "warnings": ir.get("warnings") or [],
        "tie_word_embeddings": ir.get("tie_word_embeddings"), "hidden_size": ir.get("hidden_size"),
        "components": sorted({k for k in (ir.get("extras") or {}).keys() if k.startswith("component")}),
        "layer0_blocks": [b.get("kind") or b.get("id") for b in ((layers[0].get("blocks") if layers else None) or [])][:12],
    }
def instance_inventory(cfg, name):
    import torch, torch.nn as nn
    t = time.perf_counter()
    try:
        with torch.device("meta"):
            if isinstance(cfg, dict) and "_class_name" in cfg:
                import diffusers
                cls = getattr(diffusers, cfg["_class_name"]); model = cls.from_config({k: v for k, v in cfg.items() if not k.startswith("_")})
            else:
                from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
                d = OUT / "_cfg" / name; d.mkdir(parents=True, exist_ok=True)
                c = dict(cfg); c.pop("quantization_config", None); c.pop("auto_map", None)
                json.dump(c, open(d / "config.json", "w"))
                conf = AutoConfig.from_pretrained(d)
                import transformers
                model = None; errs = []
                archs = list(getattr(conf, "architectures", None) or c.get("architectures") or [])
                for builder in ([lambda: getattr(transformers, a)._from_config(conf, attn_implementation="eager") for a in archs if hasattr(transformers, a)]
                                + [lambda: AutoModelForCausalLM.from_config(conf, attn_implementation="eager"), lambda: AutoModel.from_config(conf, attn_implementation="eager")]):
                    try: model = builder(); break
                    except Exception as e: errs.append(f"{type(e).__name__}: {str(e)[:120]}")
                if model is None: raise RuntimeError(" | ".join(errs)[:400])
        def s(m): return tuple(sorted((n, type(c).__name__) for n, c in m.named_children()))
        cands = [(n, c) for n, c in model.named_modules() if isinstance(c, nn.ModuleList) and len(c) >= 2 and not any(isinstance(l, (nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout)) for l in c)]
        stacks = []
        for n, ml in cands:
            if any(n.startswith(p + ".") for p, _ in cands if p != n): continue  # top-level stacks only
            g = collections.OrderedDict()
            for i, l in enumerate(ml): g.setdefault(s(l), []).append(i)
            stacks.append({"path": n, "len": len(ml), "element": type(ml[0]).__name__, "distinct": len(g),
                           "groups": [{"n": len(v), "first": v[0], "children": [f"{a}={b}" for a, b in k]} for k, v in list(g.items())[:6]]})
        al = collections.defaultdict(list)
        for pn, p in model.named_parameters(remove_duplicate=False): al[id(p)].append(pn)
        return {"ok": True, "class": type(model).__name__, "module": type(model).__module__, "seconds": round(time.perf_counter() - t, 2),
                "params_B": round(sum(p.numel() for p in model.parameters()) / 1e9, 3), "modules": sum(1 for _ in model.modules()),
                "stacks": stacks[:8], "shared_params": [v for v in al.values() if len(v) > 1][:4]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "seconds": round(time.perf_counter() - t, 2)}
def run(name, spec):
    rec = {"name": name, "spec": spec, "started": time.strftime("%H:%M:%S")}
    kind, _, ref = spec.partition(":")
    if kind == "corpus":
        fx = json.load(open(CORPUS / f"{ref}.json")); cfg = fx["config"]; rec["model"] = fx.get("model"); target = cfg
        rec["blessed_views"] = len(fx.get("hash_signature") or [])
    else:
        target = ref; cfg = None
    from model_unfolder import sable as sable_fn
    from model_unfolder.parser import _coerce, config_to_ir
    from model_unfolder.diagram import Diagram
    from model_unfolder.evidence.context import ParseContext
    t = time.perf_counter()
    try:
        c = _coerce(target, token=None); ctx = ParseContext.build(c, source="local", token=None)
        ir = Diagram(config_to_ir(c, parse_context=ctx)).to_ir()
        rec["parse_seconds"] = round(time.perf_counter() - t, 1); rec["ir"] = summarize_ir(ir)
        if cfg is None: cfg = dict(c) if isinstance(c, dict) else (c.to_dict() if hasattr(c, "to_dict") else None)
    except Exception as e:
        rec["parse_error"] = f"{type(e).__name__}: {str(e)[:400]}"; rec["parse_seconds"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    try:
        rep = sable_fn(target, source="local", render_images=False)
        rec["sable_seconds"] = round(time.perf_counter() - t, 1)
        rec["sable"] = {"oracle": rep.oracle, "views": len(rep.view_hashes), "view_labels": [v[0] for v in rep.view_hashes][:40],
                        "checks_total": len(rep.checks), "checks_failing": [{"name": ch.name, "blocking": ch.blocking, "findings": ch.findings[:4]} for ch in rep.checks if ch.findings],
                        "checks_passing": [ch.name for ch in rep.checks if not ch.findings]}
    except Exception as e:
        rec["sable_error"] = f"{type(e).__name__}: {str(e)[:400]}"; rec["sable_seconds"] = round(time.perf_counter() - t, 1)
    if isinstance(cfg, dict): rec["instance"] = instance_inventory(cfg, name)
    else: rec["instance"] = {"ok": False, "error": "no config dict available"}
    # first shadow comparison: layers
    try:
        ir_n = rec["ir"]["num_layers"]; st = rec["instance"].get("stacks") or []
        main = max(st, key=lambda x: x["len"]) if st else None
        rec["shadow"] = {"ir_layers": ir_n, "instance_main_stack": main and {"path": main["path"], "len": main["len"], "distinct": main["distinct"]},
                         "ir_distinct_groups": len(rec["ir"]["distinct_layer_groups"]),
                         "layer_count_agrees": (main is not None and ir_n == main["len"]),
                         "tying_agrees": (rec["ir"].get("tie_word_embeddings") is None) or (bool(rec["instance"].get("shared_params")) == bool(rec["ir"].get("tie_word_embeddings")))}
    except Exception as e:
        rec["shadow"] = {"error": str(e)[:200]}
    rec["finished"] = time.strftime("%H:%M:%S")
    (OUT / "models" / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"[{rec['finished']}] {name}: layers={rec.get('ir',{}).get('num_layers')} unknown={rec.get('ir',{}).get('unknown_slots')}/{rec.get('ir',{}).get('layer_fact_slots')} views={rec.get('sable',{}).get('views')} failing={len(rec.get('sable',{}).get('checks_failing',[]))} inst={rec['instance'].get('ok')} agree={rec.get('shadow',{}).get('layer_count_agrees')} err={rec.get('parse_error') or rec.get('sable_error') or ''}", flush=True)
for arg in sys.argv[2:]:
    name, _, spec = arg.partition("=")
    try: run(name, spec)
    except Exception:
        print(f"{name}: CRASH {traceback.format_exc().splitlines()[-1]}", flush=True)
