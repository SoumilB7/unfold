"""Precise cause for every PARTIAL result, from its recorded evidence only. Writes results_v2/PARTIALS.md + partials.json.

Cause taxonomy (per failed check; a model can have several):
  T1 mtp_or_declared_unbuilt  REAL  weights the library declares it does not build (_keys_to_ignore_on_load_unexpected)
  T1 ckpt_part_class_lacks    REAL  checkpoint tensors with no parameter in any registered class (checkpoint/code mismatch)
  T1 no_ground_truth          GAP   weights only in a format the harness cannot read (ONNX, custom)
  T1 unused_weight_module_ran REAL  the owning module executed but never read this weight (dead/optional weight)
  T1 unused_path_not_run      GAP?  the owning module never executed: a path no input reached (see T2 + input notes)
  T2 path_not_run             GAP?  weighted modules never executed
  T3 external_tensor          REAL  a tensor enters from outside PyTorch (e.g. numpy-built attention pattern)
  T5 input_dependent          REAL  op structure differs for two same-shape inputs (early exit, routing loops)
  pipeline_component          (diffusion) per-component version of the above
"""
import json, glob, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); RES = os.path.join(ROOT, "results_v2")


def pat(n):
    return re.sub(r"\.\d+(?=\.|$)", ".*", n)


def classify_checks(checks, passes=None, notes=None):
    out = []
    t1 = checks.get("T1_weights", {})
    t2 = checks.get("T2_modules", {})
    never = set((t2.get("never_executed_groups") or {}).keys())
    if t1 and not t1.get("pass", True):
        if t1.get("no_ground_truth"):
            out.append(("T1", "no_ground_truth", "GAP", "weights not in safetensors/.bin", 0))
        if t1.get("library_declared_unbuilt_numel") or t1.get("library_declared_unbuilt"):
            n = t1.get("library_declared_unbuilt_numel") or 0
            out.append(("T1", "mtp_or_declared_unbuilt", "REAL", ", ".join(t1.get("library_declared_unbuilt_prefixes") or [])[:80], n))
        for g, n in (t1.get("shipped_but_not_built_by_model") or []):
            out.append(("T1", "ckpt_part_class_lacks", "REAL", g, n))
        for g, n in (t1.get("shipped_weights_unused") or []):
            owner = ".".join(g.split(".")[:-1]) if g.split(".")[-1] in ("*",) else g
            owner_n = owner.replace(".*", ".N")
            ran = not any(owner_n.startswith(ng) or ng.startswith(owner_n) for ng in never)
            out.append(("T1", "unused_weight_module_ran" if ran else "unused_path_not_run", "REAL" if ran else "GAP?", g, n))
    if t2 and not t2.get("pass", True):
        for g, c in list((t2.get("never_executed_groups") or {}).items())[:4]:
            out.append(("T2", "path_not_run", "GAP?", g, c))
    t3 = checks.get("T3_closed_dataflow", {})
    if t3 and not t3.get("pass", True):
        ex = (t3.get("examples") or [[""]])[0]
        out.append(("T3", "external_tensor", "REAL", " ".join(map(str, ex))[:90], t3.get("external_nonscalar_tensors", 0)))
    t5 = checks.get("T5_stable_structure", {})
    if t5 and not t5.get("pass", True):
        if t5.get("note"):
            out.append(("T5", "second_input_not_run", "GAP", t5["note"], 0))
        else:
            fd = t5.get("first_divergence") or {}
            out.append(("T5", "input_dependent", "REAL", f"{fd.get('a')} vs {fd.get('b')}"[:90], 0))
    for k in ("T4_no_opaque_ops",):
        v = checks.get(k, {})
        if v and not v.get("pass", True):
            out.append(("T4", "opaque_op", "REAL", str(v.get("opaque_ops"))[:80], 0))
    return out


rows = []
for f in glob.glob(os.path.join(RES, "**", "*.json"), recursive=True):
    if "/_" in f.replace(RES, "") or os.path.basename(f).startswith(("summary", "partials")):
        continue
    try:
        R = json.load(open(f))
    except Exception:
        continue
    if not isinstance(R, dict) or R.get("verdict") != "PARTIAL":
        continue
    cat = (R.get("catalog") or {}).get("category") or f.split(os.sep)[-3]
    causes = []
    if R.get("components") is not None:
        for cn, C in R["components"].items():
            if C.get("checks") and C.get("verdict") != "FULL":
                causes += [(c[0], c[1], c[2], f"{cn}: {c[3]}", c[4]) for c in classify_checks(C["checks"])]
    else:
        causes = classify_checks(R.get("checks") or {}, R.get("passes"), R.get("input_notes"))
        # a modality pass that errored means its tower never got input: unused weights there are a harness gap
        mod_fail = {l: (v.get("error") or "")[:70] for l, v in (R.get("passes") or {}).items()
                    if not v.get("ok") and l.split(":")[0] in ("image", "audio", "video", "time_series")}
        sig_fail = [p for p in (R.get("processor_errors") or {}) if p in ("processor_error", "image_processor_error")]
        if mod_fail:
            causes = [(c[0], c[1] if c[1] not in ("unused_weight_module_ran",) else "unused_modality_input_failed",
                       "GAP" if c[1] in ("unused_weight_module_ran", "unused_path_not_run", "path_not_run") else c[2], c[3], c[4]) for c in causes]
            causes.append(("input", "modality_pass_failed", "GAP", "; ".join(f"{k}: {v}" for k, v in mod_fail.items())[:120], 0))
    failed_passes = {l: (p.get("error") or "")[:90] for l, p in (R.get("passes") or {}).items() if not p.get("ok")}
    rows.append({"repo": R["repo"], "category": cat, "class": R.get("class") or R.get("pipeline"), "causes": causes,
                 "failed_passes": failed_passes, "input_notes": (R.get("input_notes") or [])[:3]})

# model-level verdict: REAL if every cause is REAL; GAP if any cause is GAP; else UNCLEAR (GAP?)
def level(r):
    ks = {c[2] for c in r["causes"]}
    return "REAL" if ks == {"REAL"} else ("GAP" if "GAP" in ks else ("UNCLEAR" if ks else "NO-EVIDENCE"))


for r in rows:
    r["level"] = level(r)
with open(os.path.join(RES, "partials.json"), "w") as fh:                 # one row per line (same JSON content)
    fh.write("[\n" + ",\n".join(json.dumps(r, default=str) for r in rows) + "\n]\n")

L = ["# PARTIAL results — precise causes (measured from result files)", "",
     f"PARTIAL models: **{len(rows)}**  |  " + "  ".join(f"**{k}** {v}" for k, v in collections.Counter(r['level'] for r in rows).most_common()), "",
     "REAL = checkpoint/library limit (the method is right to refuse FULL); GAP = harness could not read or reach it; "
     "UNCLEAR = a weighted path never ran and the reason is not yet proven (usually a missing input).", "",
     "## Cause frequency (models affected)", "", "| check | cause | class | models |", "|---|---|---|---|"]
cf = collections.Counter()
for r in rows:
    for c in {(c[0], c[1], c[2]) for c in r["causes"]}:
        cf[c] += 1
L += [f"| {a} | {b} | {c} | {n} |" for (a, b, c), n in cf.most_common()]
L += ["", "## By category", "", "| category | PARTIAL | REAL | GAP | UNCLEAR |", "|---|---|---|---|---|"]
bc = collections.defaultdict(collections.Counter)
for r in rows: bc[r["category"]][r["level"]] += 1
L += [f"| {k} | {sum(v.values())} | {v['REAL']} | {v['GAP']} | {v['UNCLEAR']} |" for k, v in sorted(bc.items(), key=lambda x: -sum(x[1].values()))]
L += ["", "## Every PARTIAL", "", "| repo | category | level | causes (check:cause:where:numel) | failed passes |", "|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (r["level"], r["category"], r["repo"])):
    cs = "; ".join(f"{c[0]}:{c[1]}:{c[3]}:{c[4]:,}" if isinstance(c[4], int) else f"{c[0]}:{c[1]}:{c[3]}" for c in r["causes"][:4]).replace("|", "/")
    fp = "; ".join(f"{k}: {v}" for k, v in r["failed_passes"].items())[:140].replace("|", "/")
    L.append(f"| {r['repo']} | {r['category']} | {r['level']} | {cs[:300]} | {fp} |")
open(os.path.join(RES, "PARTIALS.md"), "w").write("\n".join(L))
print("\n".join(L[:40]))
