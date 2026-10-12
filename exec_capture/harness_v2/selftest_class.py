"""Negative controls for the class tie-break (class_rules.py; EXECUTION_PLAN P0.4). `--legacy` runs the same controls
against the logic at ce9a8db4 (worker_v2.py authority path ~358-368, matcher path ~391-399) to show they catch it.

    python3 selftest_class.py [--legacy]      # exit 0 = every control as expected
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import class_rules as CR


def legacy_authority(declared, scored, qualifies):         # ce9a8db4: a tie keeps the declared class, no unknown
    ok = [(sc, c) for sc, c in scored if qualifies(sc)]
    if ok:
        best = min(sc for sc, _ in ok)
        winners = [c for sc, c in ok if sc == best]
        if len(winners) == 1:
            return {"selected": winners[0], "tied": winners, "ambiguous": False}
        return {"selected": declared, "tied": winners, "ambiguous": False}
    return {"selected": declared, "tied": [], "ambiguous": False}


def legacy_matcher(declared, scored, qualifies):           # ce9a8db4: min over (numel, n, NAME) -> name order decides
    flat = sorted((sc[0], sc[1], c) for sc, c in scored)
    if flat and qualifies((flat[0][0], flat[0][1])):
        return {"selected": flat[0][2], "tied": [flat[0][2]], "ambiguous": False}
    return {"selected": declared, "tied": [], "ambiguous": False}


def legacy_resolve(path, declared, scored, qualifies):   # ce9a8db4: no unknown, no record
    ch = legacy_matcher(declared, scored, qualifies) if path == "matcher" else legacy_authority(declared, scored, qualifies)
    return {"selected": ch["selected"], "unknown": None, "record": None}


def legacy_entry_allowed(tied, subtypes, name, meth):     # P0.4 v1: only root methods were checked
    return True if name else all(hasattr(k, meth) for k in tied)


def legacy_shared_input(tied_fwd, name):                 # P0.4 v1: inputs were never checked
    return True


class _K:                                                  # a stand-in class with chosen capabilities
    def __init__(self, *attrs):
        for a in attrs: setattr(self, a, True)


def main():
    legacy = "--legacy" in sys.argv
    bad = 0
    def check(name, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}" + ("" if ok else f" (want {want!r})"))
    # the qualifiers are the worker's own (class_rules.authority_qualifies / matcher_qualifies), not re-implemented
    paths = [("authority", legacy_authority if legacy else CR.choose, CR.authority_qualifies),
             ("matcher", legacy_matcher if legacy else CR.choose, lambda s0: CR.matcher_qualifies(s0[0]))]
    for pname, choose, q in paths:
        # data2vec-text-base: declared Data2VecTextModel leaves the shipped lm_head unplaced; MaskedLM and CausalLM both
        # place everything -> a tie with no declared winner -> ambiguous (never silently chosen by name)
        s0 = (51801, 0)
        r = choose("Data2VecTextModel", [((0, 0), "Data2VecTextForCausalLM"), ((0, 0), "Data2VecTextForMaskedLM"),
                                         ((51801, 5), "Data2VecTextForSequenceClassification")], q(s0))
        check(f"{pname}: two-way tie, no declared winner -> ambiguous", r["ambiguous"], True)
        check(f"{pname}: tie lists both tied classes", sorted(r["tied"]), ["Data2VecTextForCausalLM", "Data2VecTextForMaskedLM"])
        # name order must not decide: the same tie with the names swapped gives the same verdict
        r2 = choose("Model", [((0, 0), "ZForMaskedLM"), ((0, 0), "AForCausalLM")], q(s0))
        check(f"{pname}: tie is ambiguous whatever the names", r2["ambiguous"], True)
        # xlm-roberta-base: declared MaskedLM ties CausalLM -> the declared class is among the best -> kept, no unknown
        s0x = (590592, 0)
        rx = choose("XLMRobertaForMaskedLM", [((590592, 0), "XLMRobertaForCausalLM"), ((842130, 0), "XLMRobertaModel")], q(s0x))
        check(f"{pname}: declared among the best -> declared kept", (rx["selected"], rx["ambiguous"]), ("XLMRobertaForMaskedLM", False))
        # one strictly better candidate -> selected, not ambiguous
        r1 = choose("BertModel", [((0, 0), "BertForPreTraining"), ((100, 0), "BertForMaskedLM")], q((500, 0)))
        check(f"{pname}: single best candidate -> selected", (r1["selected"], r1["ambiguous"]), ("BertForPreTraining", False))
    # shared passes: a pass runs only if every tied class supports it
    if not legacy:
        check("shared: generate on CausalLM only -> not shared", CR.shared([_K("generate"), _K()], "generate"), False)
        check("shared: generate on both -> shared", CR.shared([_K("generate"), _K("generate")], "generate"), True)
        check("shared: no tie -> everything allowed", CR.shared([], "generate"), True)
    # --- v2 (cross-exam p04_xexam S1/S2): the worker's whole decision and the shared-pass gates ---
    resolve = legacy_resolve if legacy else CR.resolve
    for pname, q in (("authority", CR.authority_qualifies((51801, 0))), ("matcher", CR.matcher_qualifies(51801))):
        r = resolve(pname, "Data2VecTextModel", [((0, 0), "Data2VecTextForCausalLM"), ((0, 0), "Data2VecTextForMaskedLM")], q)
        check(f"S2 {pname}: resolve on a tie -> G1 unknown", (r.get("unknown") or {}).get("predicate"), "G1")
        check(f"S2 {pname}: resolve on a tie -> record names the class run", (r.get("record") or {}).get("ran"), r["selected"])
        r1 = resolve(pname, "BertModel", [((0, 0), "BertForPreTraining"), ((100, 0), "BertForMaskedLM")], q)
        check(f"S2 {pname}: resolve single best -> no unknown", (r1["selected"], r1.get("unknown")), ("BertForPreTraining", None))
    ea = legacy_entry_allowed if legacy else CR.entry_allowed
    A, B = _K("predict_x"), _K()
    check("S1 root method only one tied class has -> skipped", ea([A, B], [{}, {}], "", "predict_x"), False)
    check("S1 root method every tied class has -> runs", ea([A, A], [{}, {}], "", "predict_x"), True)
    check("S1 sub-module of different types across the tie -> skipped",
          ea([A, B], [{"lm_head": "CausalLMHead"}, {"lm_head": "MaskedLMHead"}], "lm_head", "predict_x"), False)
    check("S1 sub-module missing in one tied class -> skipped", ea([A, B], [{"pooler": "Pooler"}, {}], "pooler", "f"), False)
    check("S1 sub-module of the same type in every tied class -> runs",
          ea([A, B], [{"encoder": "Enc"}, {"encoder": "Enc"}], "encoder", "f"), True)
    check("S1 no tie -> every entry runs", ea([], [], "lm_head", "f"), True)
    si = legacy_shared_input if legacy else CR.shared_input
    check("S1 input only one tied forward accepts -> not used", si([{"input_ids", "labels"}, {"input_ids"}], "labels"), False)
    check("S1 input every tied forward accepts -> used", si([{"labels"}, {"labels", "x"}], "labels"), True)
    check("S1 no tie -> every input used", si([], "labels"), True)
    # --- P0.6 N6: a candidate that could not be scored is never dropped silently; it matters (an unknown) only when the
    #     selected class still leaves shipped tensors unplaced ---
    si = getattr(CR, "selection_incomplete", None)
    inc = (lambda sel_file_side, unscored: si(sel_file_side, unscored)) if si and not legacy else (lambda a, b: None)
    check("N6 selected leaves tensors unplaced + an unscored candidate -> G1 unknown",
          (inc(590592, ["XForCausalLM: error: AttributeError"]) or {}).get("predicate"), "G1")
    check("N6 selected fits exactly (file side 0) + an unscored candidate -> no unknown", inc(0, ["XForCausalLM: no report"]), None)
    check("N6 selected leaves tensors unplaced, every candidate scored -> no unknown", inc(590592, []), None)
    print(f"\n{bad} unexpected" + (" (legacy logic: expected to fail)" if legacy else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
