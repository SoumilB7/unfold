"""Pipeline-level grading rules of the diffusion worker, kept pure so they can be tested without a hub repo
(INTENT.md; EXECUTION_PLAN_2026-10-12.md P0.1).

Rule: an unsuccessful graded pipeline call (the first call OR the stability call) never receives an overall FULL.
Component results are kept and reported separately; they cannot certify a pipeline whose own call did not complete.
"""


FULLISH = ("FULL", "FULL_UNVERIFIED")
MAX_ATTEMPTS = 9                                  # pipeline call attempts per call (bounded ladder)


def pick_attempt(best, res):
    """The graded attempt among the call variants tried. An attempt is (recorder, output, stopped, calls, steps);
    `stopped` is None when the call completed. A completed attempt is THE graded call; among failed attempts the one
    that recorded the most ops is kept (as evidence of how far the call got)."""
    if not res[2]:
        return res
    if best is None or (best[2] and len(res[0].nodes) > len(best[0].nodes)):
        return res
    return best


def overall_verdict(component_verdicts, stop_a, stop_b):
    """(verdict, pipeline_failed_checks). component_verdicts: the per-component verdicts (or ["OUT"] / ["FAIL"] set by
    the assembly rule). stop_a / stop_b: None when the first / stability call completed, else the stop reason."""
    v = component_verdicts
    if v == ["OUT"]:
        verdict = "OUT"
    elif v and all(x in FULLISH for x in v):      # a component with a not-evaluated check is never FULL (INTENT.md)
        verdict = "FULL" if all(x == "FULL" for x in v) else "FULL_UNVERIFIED"
    else:
        verdict = "PARTIAL" if any(x in ("PARTIAL",) + FULLISH for x in v) else "FAIL"
    failed = []
    if stop_a:
        failed.append("pipeline_call_completed")
    if stop_b:
        failed.append("pipeline_stability_call_completed")
    if failed and verdict in FULLISH:
        verdict = "PARTIAL"                     # observed: the graded pipeline call did not complete
    return verdict, failed


def component_verdict(failed, not_evaluated, used):
    """A component's verdict: an observed failure -> PARTIAL (FAIL if the component never ran); otherwise a check
    that could not be evaluated -> FULL_UNVERIFIED (never FULL); otherwise FULL."""
    if failed:
        return "PARTIAL" if used else "FAIL"
    return "FULL_UNVERIFIED" if not_evaluated else "FULL"


def t5_check(sa, sb, direct, second_direct_ran, stop_a, stop_b, graded_a, graded_b):
    """T5 (structure stability) of one component from its two runs (sa, sb: normalised op sequences).
    pass True/False only when the two runs are comparable evidence; otherwise pass None with the reason:
      - direct runs: the second direct run did not complete;
      - pipeline runs: a call stopped (truncated runs compare equal without proving anything), or the two calls
        graded different input variants (different step counts / input forms are not the same call);
      - the component recorded no ops in either run (nothing to compare; T2 reports it)."""
    note = None
    if direct:
        if not second_direct_ran:
            note = "second direct run did not complete: stability not evaluated"
    elif not pipeline_t5_evaluable(stop_a, stop_b):
        note = "pipeline call(s) stopped: stability not evaluated from truncated runs"
    elif graded_a != graded_b:
        note = f"the two pipeline calls graded different input variants ({graded_a} vs {graded_b}): not comparable"
    if note is None and not sa and not sb:
        note = "the component recorded no ops in either run: stability not evaluated (see T2)"
    out = {"pass": None if note else (sa == sb), "ops_a": len(sa), "ops_b": len(sb)}
    if note:
        out["note"] = note
    return out


def initial_ladder(embed_dims, has_embed_slot):
    """The first call variants: the base call, plus one per candidate embedding size only when the call has an
    embedding-tensor slot (otherwise base:<dim> would repeat the identical call)."""
    return [("base", None)] + ([("base", d) for d in embed_dims[1:4]] if has_embed_slot else [])


def kind_of(tried_label):
    return tried_label.split(":")[0]


def extend_ladder(err, tried, ladder, dim, has_steps, has_embed_slot):
    """Next variants after a failed attempt, chosen from the error text and parameter names only (never the model).
    tried: labels "kind" or "kind:dim". Returns (ladder, stop). Each variant family is added at most once."""
    kinds = {kind_of(t) for t in tried}
    if "has no len()" in err and "list" not in kinds:
        return [("list", dim), ("nested", dim)] + ladder, False
    if "channel" in err.lower() and "rgba" not in kinds:
        return [("rgba", dim), ("gray", dim)] + ladder, False
    if ("simultaneously" in err or "only one of" in err.lower()) and "novideo" not in kinds:
        return [("novideo", dim), ("noimage", dim)] + ladder, False
    if "must match the size" in err and "deflen" not in kinds:
        return [("deflen", dim), ("deflen_frames", dim)] + ladder, False
    if not ladder and has_steps and not kinds & {"steps4", "steps10"}:
        return [("steps4", dim), ("steps10", dim)], False   # bounded: the call may fail only because 2 steps is too few
    if not ladder and not has_embed_slot:
        return ladder, True
    return ladder, False


def pipeline_t5_evaluable(stop_a, stop_b):
    """Structure stability read from the two pipeline calls counts only if both calls completed: two runs truncated at
    the same point compare equal without proving anything."""
    return not stop_a and not stop_b


def finalize(R, component_verdicts, stop_a, stop_b, component_names):
    """The diffusion worker's pipeline-level ending, in one testable place: stop reasons, the overall verdict, the
    completion record and the pipeline entry in failed_components. Component entries are left as they are."""
    if stop_a and not R.get("reason"):
        R["reason"] = f"pipeline call stopped: {stop_a[:300]}"
    if stop_b and not R.get("reason"):
        R["reason"] = f"pipeline stability call stopped: {stop_b[:300]}"
    R["verdict"], failed = overall_verdict(component_verdicts, stop_a, stop_b)
    R["pipeline_call_completed"] = {"first": not stop_a, "stability": not stop_b}
    comps = R.get("components") or {}
    R["failed_components"] = {cn: comps[cn]["failed_checks"] for cn in component_names
                              if (comps.get(cn) or {}).get("failed_checks")}
    if failed:
        R["failed_components"]["(pipeline)"] = failed
        R["pipeline_stop"] = {"first": stop_a, "stability": stop_b}
    return R


def component_checks(checks):
    """(failed, not_evaluated) check names of one component: pass False is a failure, pass None is not evaluated."""
    return ([k for k, v in checks.items() if v.get("pass") is False],
            [k for k, v in checks.items() if v.get("pass") is None])
