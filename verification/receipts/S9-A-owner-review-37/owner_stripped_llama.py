"""Owner's stripped-config witness on frozen37: Llama-7B config reduced to five keys."""
import json, os, sys, time
tree = "/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/.claude/worktrees/verify-s9-a-thirtyseventh"
sys.path.insert(0, tree); os.chdir(tree)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  UNFOLD_EVIDENCE_CACHE_DIR="/private/tmp/unfold-s9a-thirtyseventh/owner-sparse-llama-cache")
import model_unfolder
full = json.load(open("tests/sable_test_corpus/llama-7b.json"))["config"]
keep = ("architectures", "model_type", "hidden_size", "num_hidden_layers", "vocab_size")
sparse = {k: full[k] for k in keep if k in full}
out = {}
for name, cfg in (("full", full), ("stripped", sparse)):
    t = time.perf_counter(); d = model_unfolder.unfold(cfg); dt = time.perf_counter() - t
    ir = d.ir if isinstance(d.ir, dict) else d.ir.to_dict()
    rows = (ir.get("extras") or {}).get("fact_provenance") or {}
    layers = ir.get("num_layers") or (ir.get("stack") or {}).get("num_layers")
    html = d.to_html()
    import re
    body = re.sub(r"<style.*?</style>", "", html, flags=re.S)
    chips = {k: len(re.findall(r'class="[^"]*\buf-chip-' + k + r'\b', body)) for k in ("unresolved", "class_default", "symbolic", "pending_design")}
    nodes = len(re.findall(r'class="[^"]*\buf-node\b', body)); cards = len(re.findall(r"uf-card-detail", body))
    keys = ("model.hidden_size", "decoder.attention.head_geometry", "decoder.ffn.activation", "decoder.ffn.intermediate_size", "decoder.attention.mask", "decoder.layer.norm_kind")
    facts = {k: {kk: rows[k].get(kk) for kk in ("status", "value")} for k in keys if k in rows}
    statuses = {}
    for k, r in rows.items(): statuses[r.get("status")] = statuses.get(r.get("status"), 0) + 1
    out[name] = dict(seconds=round(dt, 2), layers=layers, facts=facts, status_counts=statuses, chips=chips, nodes=nodes, cards=cards, html_bytes=len(html), warnings=len(getattr(d, "warnings", []) or []))
    open(f"/private/tmp/unfold-s9a-thirtyseventh/owner-sparse-llama-{name}.html", "w").write(html)
out["structure_equal"] = out["full"]["nodes"] == out["stripped"]["nodes"] and out["full"]["cards"] == out["stripped"]["cards"] and out["full"]["layers"] == out["stripped"]["layers"]
json.dump(out, open("/private/tmp/unfold-s9a-thirtyseventh/owner-sparse-llama.json", "w"), indent=1)
print(json.dumps(out, indent=1))
