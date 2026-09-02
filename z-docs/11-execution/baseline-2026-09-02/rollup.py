"""Roll the per-model audit JSONs into the baseline table with an applicability rule for unknown slots."""
import json, pathlib, sys, collections
OUT = pathlib.Path("/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/z-docs/11-execution/baseline-2026-09-02")
rows = []
def applicable(key, facts):
    pk = (facts.get("attention.position_kind") or "")
    if key == "attention.rope_theta": return str(pk).startswith("rope") or pk in ("rope", "partial_rope", "mrope", "3d_rope")
    if key in ("ffn.num_experts", "ffn.routing"): return facts.get("ffn.kind") == "moe"
    if key.startswith("attention.") and facts.get("attention.mixer_state") not in (None, "ordinary_attention", "unknown"): return key in ("attention.kind", "attention.mixer_state")
    return True
UNK = (None, "unknown", "unresolved", "ambiguous", "incomplete")
for p in sorted(OUT.glob("models/*.json")):
    d = json.load(open(p)); ir = d.get("ir") or {}; sb = d.get("sable") or {}; inst = d.get("instance") or {}; sh = d.get("shadow") or {}
    # recompute unknown rate with applicability over the distinct groups (weighted by group size)
    tot = unk = 0; unk_keys = collections.Counter()
    for g in ir.get("distinct_layer_groups") or []:
        n = next((int(str(x).split("=")[1]) for x in g["indices"] if isinstance(x, str) and x.startswith("n=")), len(g["indices"]))
        for k, v in g["facts"].items():
            if not applicable(k, g["facts"]): continue
            tot += n
            if v in UNK: unk += n; unk_keys[k] += n
    CORE = ("attention.kind","attention.num_heads","attention.head_dim","attention.position_kind","attention.mask","ffn.kind","ffn.activation","norm_kind","norm_placement")
    ctot = cunk = 0
    for g in ir.get("distinct_layer_groups") or []:
        n = next((int(str(x).split("=")[1]) for x in g["indices"] if isinstance(x, str) and x.startswith("n=")), len(g["indices"]))
        for k in CORE:
            if k in g["facts"] and applicable(k, g["facts"]):
                ctot += n
                if g["facts"][k] in UNK: cunk += n
    stacks = inst.get("stacks") or []
    stack_sum = sum(x["len"] for x in stacks) if stacks else None
    stack_max = max((x["len"] for x in stacks), default=None)
    agree = (ir.get("num_layers") in ({stack_sum} | {x["len"] for x in stacks})) if stacks and ir.get("num_layers") else None
    fails = sb.get("checks_failing") or []
    rows.append({
        "name": d["name"], "set": "C" if d["spec"].startswith("id:") else ("B" if (ir.get("family") == "diffusion") else "A"),
        "family": ir.get("family"), "arch": ir.get("architecture") or "", "layers": ir.get("num_layers"), "groups": len(ir.get("distinct_layer_groups") or []),
        "unk": f"{unk}/{tot}" if tot else "-", "unk_pct": round(100 * unk / tot, 1) if tot else None, "unk_keys": ", ".join(f"{k.split('.')[-1]}×{v}" for k, v in unk_keys.most_common(4)),
        "opaque": ir.get("opaque_layer_block") or "", "unres_blocks": ",".join(ir.get("unresolved_model_blocks") or []), "edges": len(ir.get("cross_layer_edges") or []),
        "views": sb.get("views"), "blessed": d.get("blessed_views"), "fail_blocking": sum(1 for f in fails if f.get("blocking")), "fail_adv": sum(1 for f in fails if not f.get("blocking")),
        "fail_names": "; ".join(f["name"] for f in fails)[:120],
        "inst": inst.get("class") if inst.get("ok") else f"FAIL {inst.get('error','')[:60]}", "inst_stack": (sh.get("instance_main_stack") or {}).get("len"), "inst_distinct": (sh.get("instance_main_stack") or {}).get("distinct"),
        "agree": agree, "core_unk": f"{cunk}/{ctot}" if ctot else "-", "core_pct": round(100*cunk/ctot,1) if ctot else None, "stacks": "+".join(str(x["len"]) for x in stacks) or "-", "tying": sh.get("tying_agrees"), "shared": len(inst.get("shared_params") or []),
        "t_parse": d.get("parse_seconds"), "t_sable": d.get("sable_seconds"), "err": (d.get("parse_error") or d.get("sable_error") or "")[:100],
    })
print(json.dumps(rows, indent=1))
md = ["| set | model | family | layers IR / instance stacks | groups (IR / inst) | core unknown | all unknown | top unknown keys | opaque / unresolved blocks | edges | views (now / blessed) | sable failing (block/adv) | instance | agree | tying | parse s | sable s | error |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (r["set"], r["name"])):
    md.append(f"| {r['set']} | {r['name']} | {r['family']} | {r['layers']} / {r['stacks']} | {r['groups']} / {r['inst_distinct']} | {r['core_unk']} ({r['core_pct']}%) | {r['unk']} ({r['unk_pct']}%) | {r['unk_keys']} | {r['opaque']} {r['unres_blocks']} | {r['edges']} | {r['views']} / {r['blessed']} | {r['fail_blocking']}/{r['fail_adv']} {r['fail_names']} | {r['inst']} | {r['agree']} | {r['tying']} | {r['t_parse']} | {r['t_sable']} | {r['err']} |")
(OUT / "_table.md").write_text("\n".join(md) + "\n")
print("table rows:", len(rows), file=sys.stderr)
