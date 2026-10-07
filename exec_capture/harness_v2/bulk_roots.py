"""Bulk root-cause mining over every non-FULL result: one feature vector per model, clustered by mechanism."""
import json, glob, re, collections
RES = "results_v2"
rows = []
for f in glob.glob(f"{RES}/**/*.json", recursive=True):
    if "/_" in f.replace(RES, ""): continue
    try: R = json.load(open(f))
    except Exception: continue
    if not isinstance(R, dict) or R.get("verdict") not in ("PARTIAL", "FAIL"): continue
    rows.append(R)

def norm(s): return re.sub(r"'[^']*'", "'X'", re.sub(r"\d+", "N", s or ""))

def where_file(w):
    m = re.search(r"(transformers|diffusers|torch)/([\w/]+?)\.py", w or "")
    if not m: return ""
    p = m.group(2)
    p = re.sub(r"models/[\w]+/modeling_[\w]+", "models/*/modeling", p)
    p = re.sub(r"pipelines/[\w]+/pipeline_[\w]+", "pipelines/*/pipeline", p)
    p = re.sub(r"models/transformers/transformer_[\w]+", "models/transformers/*", p)
    return f"{m.group(1)}/{p}"

def features(R):
    fs = []
    diff = R.get("components") is not None
    passes = R.get("passes") or {}
    errs = [(l, p.get("error") or "", p.get("where") or "") for l, p in passes.items() if not p.get("ok") and not l.startswith(("stability", "train", "generate", "entry"))]
    pc = (R.get("pipeline_call") or {}).get("stopped_at") or ""
    if pc:
        et = pc.split(":")[0]; wf = where_file(pc)
        fs.append(("pipeline_stop", et, wf, norm(pc.split("@")[0])[:70]))
    for l, e, w in errs:
        fs.append(("pass_error:" + l.split(":")[0], e.split(":")[0], where_file(w) or where_file(e), norm(e)[:70]))
    if R.get("verdict") == "FAIL" and not errs and not pc:
        fs.append(("fail_reason", (R.get("reason") or "").split(":")[0][:40], "", norm(R.get("reason"))[:70]))
    ck = R.get("checks") or {}
    comps = {cn: C.get("checks") or {} for cn, C in (R.get("components") or {}).items() if C.get("checks") and C.get("verdict") != "FULL"}
    for cn, cks in ([("", ck)] if not diff else list(comps.items())):
        t1 = cks.get("T1_weights", {})
        if t1 and not t1.get("pass", True):
            if t1.get("no_ground_truth"): fs.append(("T1", "no_ground_truth", "", ""))
            if t1.get("library_declared_unbuilt_numel") or t1.get("library_declared_unbuilt"): fs.append(("T1", "declared_unbuilt", "", ""))
            if t1.get("shipped_but_not_built_by_model"): fs.append(("T1", "ckpt_part_not_in_class", "", ""))
            if t1.get("shipped_weights_unused"): fs.append(("T1", "shipped_unused", "", ""))
        t2 = cks.get("T2_modules", {})
        if t2 and not t2.get("pass", True):
            for g in list((t2.get("never_executed_groups") or {}))[:2]:
                leaf = [p for p in g.split(".") if p not in ("N",)]
                fs.append(("T2_never_ran", (leaf[-1] if leaf else g)[:30], (cn + ":" if cn else "") + ".".join(leaf[:2])[:40], ""))
        for k in ("T3_closed_dataflow", "T4_no_opaque_ops", "T5_stable_structure"):
            v = cks.get(k, {})
            if v and not v.get("pass", True): fs.append((k[:2], "note" if v.get("note") else "diff", "", ""))
    return fs

ftab = collections.Counter(); ex = collections.defaultdict(list); per_model = {}
for R in rows:
    fs = features(R); per_model[R["repo"]] = fs
    for f in set((a, b, c) for a, b, c, d in fs):
        ftab[f] += 1; ex[f].append(R["repo"])

print(f"non-FULL models: {len(rows)}  (PARTIAL {sum(r['verdict']=='PARTIAL' for r in rows)}, FAIL {sum(r['verdict']=='FAIL' for r in rows)})\n")
print("== error type x library location (pass errors / pipeline stops) ==")
loc = collections.Counter(); lex = collections.defaultdict(list)
for R in rows:
    for a, b, c, d in per_model[R["repo"]]:
        if a.startswith(("pass_error", "pipeline_stop", "fail_reason")):
            loc[(b, c)] += 1; lex[(b, c)].append(d)
for (et, wf), n in loc.most_common(25):
    print(f"{n:4d} {et[:28]:28s} {wf[:50]:50s} | {collections.Counter(lex[(et, wf)]).most_common(1)[0][0][:80]}")
print("\n== which modality pass fails, and how ==")
mp = collections.Counter((a, b) for R in rows for a, b, c, d in per_model[R["repo"]] if a.startswith("pass_error"))
for k, n in mp.most_common(14): print(f"{n:4d} {k}")
print("\n== checks ==")
ck = collections.Counter((a, b) for R in rows for a, b, c, d in set(per_model[R["repo"]]) if a in ("T1", "T3", "T4", "T5"))
for k, n in ck.most_common(): print(f"{n:4d} {k}")
print("\n== module kinds that never ran (leaf names) ==")
nr = collections.Counter(b for R in rows for a, b, c, d in set(per_model[R["repo"]]) if a == "T2_never_ran")
print(nr.most_common(30))
json.dump(per_model, open(f"{RES}/_bulk_features.json", "w"), indent=0)
