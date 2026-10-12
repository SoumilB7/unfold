"""Negative controls for the diffusion worker's pipeline-level rules (diffusion_rules.py; EXECUTION_PLAN P0.1).
Each control states the rule it protects. `--legacy` runs the same controls against the pre-P0.1 logic (copied verbatim
from worker_v2_diffusion.py at 8b5c90a4) to show they catch it.

    python3 selftest_diffusion.py [--legacy]      # exit 0 = every control as expected
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusion_rules as DR


class _Rec:
    def __init__(self, n): self.nodes = [None] * n


def legacy_overall(v, stop_a, stop_b):                     # 8b5c90a4 worker_v2_diffusion.py:545-546 (stop_b ignored)
    verdict = ("OUT" if v == ["OUT"] else "FULL" if v and all(x == "FULL" for x in v)
               else ("PARTIAL" if any(x in ("PARTIAL", "FULL") for x in v) else "FAIL"))
    return verdict, []


def legacy_pick(best, res):                                # 8b5c90a4 worker_v2_diffusion.py run_pipeline
    return res if best is None or len(res[0].nodes) > len(best[0].nodes) else best


# --- the P0.1 v2 logic (before the v3 cross-exam fixes), copied verbatim from the worker, for the v3 controls ---
def v2_component_verdict(failed, not_evaluated, used):
    return "FULL" if not failed else ("PARTIAL" if used else "FAIL")


def v2_t5(sa, sb, direct, second_direct_ran, stop_a, stop_b, graded_a, graded_b):
    ok = direct or (not stop_a and not stop_b)
    return {"pass": (bool(sa) and sa == sb) if ok else None}


def v2_extend(err, tried, ladder, dim, has_steps, has_embed_slot):
    if "has no len()" in err and not any(t.startswith("list") for t in tried): return [("list", dim), ("nested", dim)] + ladder, False
    elif ("channel" in err.lower()) and not any(t.startswith("rgba") for t in tried): return [("rgba", dim), ("gray", dim)] + ladder, False
    elif ("simultaneously" in err or "only one of" in err.lower()) and "novideo" not in tried:
        return [("novideo", dim), ("noimage", dim)] + ladder, False
    elif "must match the size" in err and "deflen" not in tried:
        return [("deflen", dim), ("deflen_frames", dim)] + ladder, False
    elif not ladder and has_steps and not any(t.startswith("steps") for t in tried):
        return [("steps4", dim), ("steps10", dim)], False
    elif not has_embed_slot and not ladder: return ladder, True
    return ladder, False


def v2_initial(embed_dims, has_embed_slot):
    return [("base", None)] + [("base", d) for d in embed_dims[1:4]]


def run_ladder(initial, extend, oracle, has_steps=True, has_embed=True, cap=9):
    """The worker's ladder loop over an error oracle (attempt label "kind" / "kind:dim" -> error or None)."""
    ladder, tried = list(initial), []
    while ladder:
        kind, dim = ladder.pop(0)
        tried.append(f"{kind}:{dim}" if dim else kind)
        err = oracle(tried[-1])
        if not err: break
        ladder, stop = extend(err, tried, ladder, dim, has_steps, has_embed)
        if stop or len(tried) >= cap: break
    return tried


def main():
    legacy = "--legacy" in sys.argv
    overall = legacy_overall if legacy else DR.overall_verdict
    pick = legacy_pick if legacy else DR.pick_attempt
    bad = 0
    def check(name, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}" + ("" if ok else f" (want {want!r})"))
    # 1. every component perfect (direct runs), but the graded pipeline call failed -> never FULL
    check("components FULL, call a stopped -> not FULL", overall(["FULL", "FULL"], "ValueError @ scheduling_pndm.py:217", None)[0], "PARTIAL")
    # 2. the first call completed, the stability call failed -> never FULL
    check("call a ok, call b stopped -> not FULL", overall(["FULL"], None, "RuntimeError @ x.py:1")[0], "PARTIAL")
    # 3. an invalid candidate (more ops, failed) then a valid one (fewer ops, completed) -> the valid one is graded
    failed = (_Rec(500), None, "TypeError: bad candidate", {}, {})
    good = (_Rec(300), "out", None, {}, {})
    check("invalid candidate then valid one -> graded on the valid one", pick(pick(None, failed), good)[2], None)
    # 4. assembly failure keeps its verdict (OUT / FAIL set by the assembly rule), never FULL
    check("assembly failure (OUT list) stays OUT", overall(["OUT"], "pipeline could not be assembled: X", None)[0], "OUT")
    check("assembly failure (FAIL list) stays FAIL", overall(["FAIL"], "pipeline could not be assembled: X", None)[0], "FAIL")
    # 5. a successful pipeline with a failed component check -> PARTIAL (the gate must not hide component failures)
    check("call completed, a component PARTIAL -> PARTIAL", overall(["FULL", "PARTIAL"], None, None)[0], "PARTIAL")
    # positive control: everything completed and passed -> FULL (the gate must not block a clean row)
    check("clean row -> FULL", overall(["FULL", "FULL"], None, None)[0], "FULL")
    # the pipeline failure is named as a failed check, not only a reason string
    if not legacy:
        check("failed call is a named check", overall(["FULL"], "E", "E")[1], ["pipeline_call_completed", "pipeline_stability_call_completed"])
        check("T5 from two truncated pipeline runs is not evaluable", DR.pipeline_t5_evaluable("E", "E"), False)
        check("T5 from two completed pipeline runs is evaluable", DR.pipeline_t5_evaluable(None, None), True)
    if not legacy:
        # the worker's ending (finalize) and component check split are what the worker actually calls
        R = {"components": {"unet": {"failed_checks": []}, "vae": {"failed_checks": ["T2_modules"]}}}
        DR.finalize(R, ["FULL", "PARTIAL"], None, "RuntimeError @ x.py:1", ["unet", "vae"])
        check("finalize: only the stability call stops -> PARTIAL", R["verdict"], "PARTIAL")
        check("finalize: pipeline entry names the stability call", R["failed_components"].get("(pipeline)"),
              ["pipeline_stability_call_completed"])
        check("finalize: component failures kept", R["failed_components"].get("vae"), ["T2_modules"])
        R2 = {"components": {"unet": {"failed_checks": []}}}
        DR.finalize(R2, ["FULL"], None, None, ["unet"])
        check("finalize: clean -> FULL, no pipeline entry", (R2["verdict"], "(pipeline)" in R2["failed_components"]), ("FULL", False))
        R3 = {"components": {}}
        DR.finalize(R3, ["FULL"], "E @ a", None, [])
        check("finalize: first call stops -> reason names it", R3["reason"].startswith("pipeline call stopped"), True)
        f, ne = DR.component_checks({"T1_weights": {"pass": True}, "T5_stable_structure": {"pass": None}, "T2_modules": {"pass": False}})
        check("component: T5 not evaluated is not a failure", (f, ne), (["T2_modules"], ["T5_stable_structure"]))
    # --- v3 controls (cross-exam p01b_xexam F1-F5); --legacy runs them against the v2 logic above ---
    cv = v2_component_verdict if legacy else DR.component_verdict
    t5 = v2_t5 if legacy else DR.t5_check
    ext = v2_extend if legacy else DR.extend_ladder
    ini = v2_initial if legacy else DR.initial_ladder
    # F1: a component whose check could not be evaluated is never FULL, and neither is the row
    check("F1 component with a not-evaluated check -> FULL_UNVERIFIED", cv([], ["T5_stable_structure"], True), "FULL_UNVERIFIED")
    if not legacy:
        check("F1 row of FULL + FULL_UNVERIFIED components -> FULL_UNVERIFIED", overall(["FULL", "FULL_UNVERIFIED"], None, None)[0], "FULL_UNVERIFIED")
        check("F1 FULL_UNVERIFIED components + a stopped call -> PARTIAL", overall(["FULL_UNVERIFIED"], "E", None)[0], "PARTIAL")
        check("F1 FULL_UNVERIFIED + PARTIAL component -> PARTIAL", overall(["FULL_UNVERIFIED", "PARTIAL"], None, None)[0], "PARTIAL")
    A = [("x", "add")] * 3
    # F2: the second direct run did not complete -> not evaluated (was: False, read as a REAL input-dependence)
    check("F2 second direct run failed -> T5 None", t5(A, [], True, False, None, None, "base", "base")["pass"], None)
    # F2: no ops in either completed call -> nothing to compare -> None (was: False)
    check("F2 no ops in either run -> T5 None", t5([], [], False, True, None, None, "base", "base")["pass"], None)
    # F3: the two calls graded different variants -> not comparable -> None (was: compared, False)
    check("F3 calls graded different variants -> T5 None", t5(A, A[:2], False, True, None, None, "base", "steps4")["pass"], None)
    # positive / refusing controls: comparable runs still decide T5
    check("T5 comparable, same structure -> True", t5(A, A, False, True, None, None, "base", "base")["pass"], True)
    check("T5 comparable, different structure -> False", t5(A, A[:2], False, True, None, None, "base", "base")["pass"], False)
    check("T5 direct runs differ -> False", t5(A, A[:2], True, True, "E", "E", None, None)["pass"], False)
    # F4: a guard keyed on the bare kind must not re-add a family when attempts carry a dim (was: deflen:1024 x7,
    # steps never tried). Oracle: base fails otherwise; from base:1024 on, a size mismatch, until steps4 completes.
    K = lambda t: t.split(":")[0]
    orc = lambda t: (None if K(t) == "steps4" else "ValueError: other" if t == "base"
                     else "RuntimeError: The size of tensor a (4) must match the size of tensor b")
    tr = run_ladder(ini([768, 1024], True), ext, orc)
    check("F4 size-mismatch ladder reaches steps4", tr[-1].split(":")[0], "steps4")
    check("F4 no variant family repeated", len(tr) == len(set(tr)), True)
    orc2 = lambda t: (None if K(t) == "steps4" else "ValueError: other" if t == "base"
                      else "ValueError: only one of video or image may be given")
    tr2 = run_ladder(ini([768, 1024], True), ext, orc2)
    check("F4 video/image ladder reaches steps4", tr2[-1].split(":")[0], "steps4")
    # F5: no embedding slot -> base:<dim> would repeat the identical call; it is not tried
    check("F5 no embedding slot -> one base attempt", ini([768, 1024], False), [("base", None)])
    # ---- P0.6 F7: a call that stops INSIDE a component's method reached that component; the direct-run reason says
    #      what the evidence shows (never "never reached" when the traceback runs through it) ----
    import torch
    class Normalizer(torch.nn.Module):
        def scale(self, x):
            return x @ torch.zeros(3, 2)                     # raises: width mismatch, inside the component
    comps = {"image_normalizer": Normalizer(), "unet": torch.nn.Linear(2, 2)}
    def pipe(x):
        return comps["image_normalizer"].scale(x)
    try:
        pipe(torch.zeros(1, 4))
    except Exception as e:
        tb = e.__traceback__
    walk = getattr(DR, "components_on_traceback", None)
    si = walk(tb, comps) if walk and not legacy else {}
    why = (lambda cn, si, stop: DR.direct_run_reason(cn, si, stop)) if hasattr(DR, "direct_run_reason") and not legacy \
        else (lambda cn, si, stop: "the pipeline call never reached this component")     # ce9a8db4 wording
    check("F7 the stopping component is found on the traceback", "image_normalizer" in si, True)
    check("F7 an untouched component is not", "unet" in si, False)
    check("F7 reason for the stopping component is not 'never reached'",
          "never reached" in why("image_normalizer", si, "RuntimeError @ x.py:1"), False)
    check("F7 reason for a component the call completed without calling",
          "completed" in why("unet", {}, None), True)
    print(f"\n{bad} unexpected" + (" (legacy logic: expected to fail)" if legacy else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
