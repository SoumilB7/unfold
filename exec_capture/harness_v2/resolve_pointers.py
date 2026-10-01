"""Link every OUT pointer (GGUF / MLX / LoRA copy) to its base model's measured verdict and, for GGUF, to the
header-level size comparison (GGUF tensor elements vs base checkpoint). Writes results_v2/POINTERS.md."""
import json, glob, os, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results_v2")
res = {}
for f in glob.glob(os.path.join(RES, "**", "*.json"), recursive=True):
    if "/_" in f.replace(RES, "") or f.endswith(("summary.json",)): continue
    try:
        R = json.load(open(f)); res[R["repo"]] = (f, R)
    except Exception:
        pass
gg = {}
gp = os.path.join(ROOT, "baselines", "2026-09-28-top-downloads", "other_libs", "mech_a_gguf.json")
if os.path.exists(gp):
    for r in json.load(open(gp)).get("results", []):
        gg[r["repo"]] = r
rows, stats = [], collections.Counter()
for repo, (f, R) in res.items():
    if R.get("verdict") != "OUT" or "measured on its base model" not in (R.get("reason") or ""): continue
    base = R["reason"].split("base model")[-1].strip()
    base = None if base in ("None", "") else base
    bv = res.get(base, (None, {}))[1].get("verdict") if base else None
    g = gg.get(repo, {})
    ratio = g.get("ratio_gguf_over_base")
    R["pointer"] = {"base_model": base, "base_verdict": bv, "gguf_over_base_elements": ratio,
                    "gguf_has_mmproj": g.get("has_mmproj")}
    json.dump(R, open(f, "w"), indent=1, default=str)
    stats[("base measured: " + (bv or "not in results")) if base else "no base declared"] += 1
    rows.append((repo, base, bv, ratio))
lines = ["# Pointer repos (GGUF / MLX / LoRA copies) resolved to their base model", "",
         "Counts: " + ", ".join(f"{k}: {v}" for k, v in stats.most_common()), "",
         "| copy | base model | base verdict | GGUF/base elements |", "|---|---|---|---|"]
lines += [f"| {a} | {b} | {c} | {'' if d is None else f'{d:.3f}'} |" for a, b, c, d in sorted(rows)]
open(os.path.join(RES, "POINTERS.md"), "w").write("\n".join(lines))
print("\n".join(lines[:4]))
