"""Compare a re-run folder with results_v2: verdict transitions + what the new modes contributed."""
import json, glob, os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results_v2")
trans = collections.Counter(); rows = []; mode_gain = collections.Counter()
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
        try: N = json.load(open(f))
        except Exception: continue
        if not isinstance(N, dict) or not N.get("verdict"): continue
        rel = os.path.relpath(f, d); o = os.path.join(RES, rel)
        O = json.load(open(o)) if os.path.exists(o) else {}
        a, b = O.get("verdict", "NEW"), N["verdict"]; trans[(a, b)] += 1
        for l, p in (N.get("passes") or {}).items():
            if l.startswith(("mode:", "entry:", "image+text")) and p.get("ok") and (p.get("newly_used_params") or 0) > 0:
                mode_gain[l.split(".")[-1] if l.startswith("entry:") else l] += 1
        if a != b: rows.append((a, b, N["repo"], (N.get("reason") or O.get("reason") or "")[:90]))
for (a, b), n in sorted(trans.items(), key=lambda x: -x[1]): print(f"{a:8} -> {b:8} {n}")
print("modes that added weights:", dict(mode_gain))
for r in sorted(rows): print(" ", *r)
