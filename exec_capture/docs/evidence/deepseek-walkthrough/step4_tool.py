import json, sys, re, warnings; warnings.filterwarnings("ignore")
from model_unfolder import unfold, estimate_params
src = sys.argv[1]
arg = json.load(open(src))["config"] if src.endswith(".json") else src
try:
    d = unfold(arg)
except Exception as e:
    print(f"  unfold({'fixture dict' if src.endswith('.json') else repr(src)}) -> CRASH {type(e).__name__}: {e}"); sys.exit()
html = d.to_html(); js = json.dumps(d.to_json() if hasattr(d, "to_json") else d.ir.to_dict() if hasattr(d.ir,"to_dict") else {}, default=str)
e = estimate_params(d.ir); tot = e["total"] if isinstance(e, dict) else getattr(e, "total", None)
ir = d.ir
kinds = {}
for l in ir.layers:
    f = getattr(l, "ffn", None); k = getattr(f, "kind", None) if f else None
    kinds[str(k)] = kinds.get(str(k), 0) + 1
print(f"  layers drawn: {len(ir.layers)} | FFN kinds per layer: {kinds} | params total: {tot:,}" if tot else f"  layers {len(ir.layers)} params {tot}")
checks = {
  "q path 7168->1536->24576":            ["1536", "24576"],
  "kv_a 576 (=512 latent + 64 rope)":    ["576"],
  "kv_b 512->32768":                     ["32768"],
  "o_proj 16384->7168":                  ["16384"],
  "dense FFN width 18432":               ["18432", "18,432"],
  "expert width 2048":                   ["2048", "2,048"],
  "score scale 0.13523...":              ["0.1352"],
  "router sigmoid":                      ["sigmoid"],
  "group top-k (8 groups, top 4)":       ["topk_group", "top-4 group", "4 groups", "topk group"],
  "routed scaling 2.5":                  ["2.5"],
  "correction bias":                     ["e_score_correction_bias", "correction bias"],
  "shared expert":                       ["shared expert", "shared_experts", "Shared expert"],
  "MTP / layer 61":                      ["MTP", "Multi-Token", "nextn", "eh_proj", "layers.61"],
}
for name, pats in checks.items():
    hit = [p for p in pats if p in html or p in js]
    print(f"    {'FOUND ' if hit else 'absent'}  {name:36s} {hit[:2]}")
