"""Merge re-run result folders into results_v2: old file -> results_v2/_superseded_<tag>/, newest re-run file in place.
Usage: python merge_rerun.py <tag> <dir> [<dir> ...]   (later dirs / newer files win)"""
import json, os, shutil, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results_v2")
tag, dirs = sys.argv[1], sys.argv[2:]
best = {}
for d in dirs:
    for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
        try:
            R = json.load(open(f))
        except Exception:
            continue
        if not isinstance(R, dict) or not R.get("verdict"):
            continue
        rel = os.path.relpath(f, d)
        if rel not in best or os.path.getmtime(f) > os.path.getmtime(best[rel]):
            best[rel] = f
moved = new = 0
for rel, f in best.items():
    dst = os.path.join(RES, rel)
    if os.path.exists(dst):
        sup = os.path.join(RES, f"_superseded_{tag}", rel)
        os.makedirs(os.path.dirname(sup), exist_ok=True)
        shutil.move(dst, sup); moved += 1
    else:
        new += 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(f, dst)
print(f"merged {len(best)} results ({moved} replaced, {new} new) -> {RES}; old copies in _superseded_{tag}/")
