"""Summarise results_v2/**.json into results_v2/SUMMARY.md (+ summary.json). Every number is counted from result files."""
import json, glob, os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_v2")
rows = []
for f in glob.glob(os.path.join(RES, "**", "*.json"), recursive=True):
    if os.path.basename(f).startswith(("summary", "_")) or "/_" in f.replace(RES, ""): continue
    try:
        R = json.load(open(f))
    except Exception:
        continue
    if not isinstance(R, dict) or "repo" not in R:
        continue
    cat = (R.get("catalog") or {}).get("category") or f.split(os.sep)[-3]
    sub = (R.get("catalog") or {}).get("subcategory") or f.split(os.sep)[-2]
    fails = R.get("failed_checks") or sorted({c for v in (R.get("failed_components") or {}).values() for c in v})
    rows.append({"repo": R["repo"], "category": cat, "subcategory": sub, "verdict": R.get("verdict", "?"),
                 "failed_checks": fails, "reason": (R.get("reason") or "")[:160], "secs": R.get("job_secs") or R.get("secs"),
                 "peak_gb": R.get("peak_rss_gb"), "harness_id": R.get("harness_id")})


def reason_class(r):
    s = r["reason"]
    for pat, name in [("remote code", "remote code only"), ("gated", "gated"), ("no config", "no config.json"),
                      ("not in installed", "class not in installed library"), ("repo: architecture is measured on its base", "GGUF/MLX/adapter pointer"),
                      ("timeout", "timeout"), ("no input path", "no input path"), ("input construction", "input construction"),
                      ("every forward pass failed", "forward failed"), ("build:", "build failed"), ("out of memory", "crash/OOM")]:
        if pat in s: return name
    return s.split(":")[0][:50] or "unknown"


V = collections.Counter(r["verdict"] for r in rows)
HV = collections.Counter(r["harness_id"] or "unstamped (graded before 2026-10-09)" for r in rows)
if "--strict" in sys.argv and len(HV) > 1:
    sys.exit(f"refusing: rows graded by {len(HV)} harness versions {dict(HV)}; regrade so every row has one id")
lines = ["# Strict capture benchmark — measured results", "",
         "Graded by harness version(s): " + ", ".join(f"`{k}` {v}" for k, v in HV.most_common())
         + ("  **(MIXED: these numbers combine different grading code)**" if len(HV) > 1 else ""), "",
         f"Repos with a verdict: **{len(rows)}**  |  " + "  ".join(f"**{k}** {v}" for k, v in V.most_common()), "",
         (lambda run: f"Runnable {run}: FULL {100*V['FULL']/run:.1f}%  |  architecture-complete (FULL + FULL_LEFTOVERS) "
                      f"{100*(V['FULL']+V['FULL_LEFTOVERS'])/run:.1f}%  |  FULL_UNVERIFIED {100*V['FULL_UNVERIFIED']/run:.1f}%")(
             V['FULL'] + V['FULL_LEFTOVERS'] + V['FULL_UNVERIFIED'] + V['PARTIAL'] + V['FAIL']), "",
         "FULL = every shipped weight used, every module executed, closed dataflow, no opaque ops, identical structure for two inputs. "
         "FULL_LEFTOVERS = everything the model runs is captured; the only gap is shipped tensors the library itself certifies as unused "
         "by this model and that no available runtime executes, or stored buffers measured never read in any run mode (both listed per result). Architecture-complete = FULL + FULL_LEFTOVERS. "
         "FULL_UNVERIFIED = every observed check passes, but some predicate could not be evaluated (e.g. no library load, a tie "
         "shipped twice); the unknowns are listed per result and it is never counted as FULL.", "",
         "## By category", "", "| category | repos | FULL | FULL_LEFTOVERS | FULL_UNVERIFIED | PARTIAL | FAIL | OUT | FULL % of runnable | architecture-complete % |",
         "|---|---|---|---|---|---|---|---|---|---|"]
bycat = collections.defaultdict(list)
for r in rows: bycat[r["category"]].append(r)
for c, rs in sorted(bycat.items(), key=lambda x: -len(x[1])):
    cc = collections.Counter(r["verdict"] for r in rs)
    runnable = cc["FULL"] + cc["FULL_LEFTOVERS"] + cc["FULL_UNVERIFIED"] + cc["PARTIAL"] + cc["FAIL"]
    lines.append(f"| {c} | {len(rs)} | {cc['FULL']} | {cc['FULL_LEFTOVERS']} | {cc['FULL_UNVERIFIED']} | {cc['PARTIAL']} | {cc['FAIL']} | {cc['OUT']} | "
                 f"{100*cc['FULL']/runnable:.0f}% | {100*(cc['FULL']+cc['FULL_LEFTOVERS'])/runnable:.0f}% |"
                 if runnable else f"| {c} | {len(rs)} | 0 | 0 | 0 | 0 | 0 | {cc['OUT']} | – | – |")
lines += ["", "## Why PARTIAL (which strict checks failed)", ""]
pc = collections.Counter(c for r in rows if r["verdict"] == "PARTIAL" for c in r["failed_checks"])
lines += [f"- {k}: {v}" for k, v in pc.most_common()]
lines += ["", "## Why FAIL / OUT", ""]
for v in ("FAIL", "OUT"):
    rc = collections.Counter(reason_class(r) for r in rows if r["verdict"] == v)
    lines.append(f"- **{v}**: " + ", ".join(f"{k} ({n})" for k, n in rc.most_common(12)))
lines += ["", "## By subcategory", "", "| category / subcategory | repos | FULL | FULL_LEFTOVERS | FULL_UNVERIFIED | PARTIAL | FAIL | OUT |",
          "|---|---|---|---|---|---|---|---|"]
bysub = collections.defaultdict(list)
for r in rows: bysub[(r["category"], r["subcategory"])].append(r)
for (c, s), rs in sorted(bysub.items()):
    cc = collections.Counter(r["verdict"] for r in rs)
    lines.append(f"| {c} / {s} | {len(rs)} | {cc['FULL']} | {cc['FULL_LEFTOVERS']} | {cc['FULL_UNVERIFIED']} | {cc['PARTIAL']} | {cc['FAIL']} | {cc['OUT']} |")
sec = [r["secs"] for r in rows if isinstance(r["secs"], (int, float))]
gb = [r["peak_gb"] for r in rows if isinstance(r["peak_gb"], (int, float))]
if sec:
    sec.sort(); gb.sort()
    lines += ["", f"Cost per repo: median {sec[len(sec)//2]:.0f}s, p90 {sec[int(len(sec)*.9)]:.0f}s; peak RAM median {gb[len(gb)//2] if gb else 0:.2f} GB, max {max(gb) if gb else 0:.2f} GB"]
os.makedirs(RES, exist_ok=True)
open(os.path.join(RES, "SUMMARY.md"), "w").write("\n".join(lines))
with open(os.path.join(RES, "summary.json"), "w") as fh:                  # one row per line (same JSON content)
    fh.write("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n")
print("\n".join(lines[:40]))
