"""Run the P4 oracle over every capture bundle in a folder (one row per repo) + hand-made wrong page facts on a few
real rows. Writes <out>/results.json. One model at a time (float64, layers shrunk to --layers).
    python3 run_families.py <captures-dir> <out-dir> [--layers 2] [--wrong REPO,REPO,...]
"""
import os, sys, json, glob, gc, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch, capture, oracle
from selftest_oracle import WRONG, applicable, edit_kw

a = sys.argv[1:]
cap_dir, out_dir = a[0], a[1]
NL = int(a[a.index("--layers") + 1]) if "--layers" in a else 0      # 0: every layer when it fits (oracle.affordable_layers)
WRONG_ON = a[a.index("--wrong") + 1].split(",") if "--wrong" in a else []
ONLY = a[a.index("--only") + 1].split(",") if "--only" in a else None
os.makedirs(out_dir, exist_ok=True)
outp = os.path.join(out_dir, "results.json")
res = json.load(open(outp)) if os.path.exists(outp) else {"rows": {}, "wrong_facts": {}}
for p in sorted(glob.glob(os.path.join(cap_dir, "*.capture.json.gz"))):
    B = capture.load(p)
    repo = B["repo"]
    if ONLY and repo not in ONLY: continue
    t0 = time.time()
    try:
        row = oracle.run_bundle(B, n_layers=NL or None, S=(24, 257), seeds=(0, 1, 2, 3, 4))
    except Exception as e:                           # an oracle crash is reported as such, never as a verdict
        import traceback; traceback.print_exc()
        row = {"repo": repo, "verdict": "ORACLE_ERROR", "reason": f"{type(e).__name__}: {str(e)[:300]}"}
    row["bundle"] = os.path.basename(p); row["verdict_graded"] = B.get("verdict"); row["seconds"] = round(time.time() - t0, 1)
    res["rows"][repo] = row
    print(f"{repo}: {row['verdict']}  e2e={row.get('end_to_end')}  {row.get('first_fail_facts') or row.get('reason', '')[:150]}",
          flush=True)
    # wrong pages on a PASS row, or on a row whose only open item is a window too long to cross here
    only_window = row["verdict"].startswith("NOT_CHECKABLE(sliding window") and not row.get("fail_stages")
    if repo in WRONG_ON and (row["verdict"] == "PASS" or only_window):
        import exec_to_ir
        from selftest_oracle_families import WRONG as FAM_WRONG
        F0 = exec_to_ir.recognize(B)
        n_ = row.get("real_layers_built") or 2
        dt_ = {"float64": oracle.torch.float64, "float32": oracle.torch.float32}.get(row.get("dtype"), oracle.torch.float64)
        model = oracle.build_real(B, n_, dtype=dt_)
        cases = [(w, k, e, t, "page") for w, (k, e, t, *_f) in WRONG.items()] + \
                [(w, "f", e, t, "facts") for w, (e, t) in FAM_WRONG.items()]
        for wname, kind, edit, test, on in cases:
            if on == "page" and not applicable(row["page_facts"], test): continue
            if on == "facts" and not applicable(F0, test): continue
            try:
                r = oracle.run_bundle(B, model=model, S=(24, 257), seeds=(0, 2, 3, 4), **edit_kw(kind, edit))
            except Exception as e:                   # a crash is reported as such, never as a refusal
                r = {"verdict": f"ORACLE_CRASH({type(e).__name__}: {str(e)[:120]})"}
            r = {k: r.get(k) for k in ("verdict", "reason", "first_fail", "first_fail_facts", "end_to_end", "errors", "issues")}
            res["wrong_facts"].setdefault(repo, {})[wname] = r
            print(f"   wrong fact [{wname}]: {r['verdict']}  e2e={r['end_to_end']}", flush=True)
        del model
    json.dump(res, open(outp, "w"), indent=1, default=str)
    gc.collect()
