"""Which library class executes a checkpoint when the declared class does not place every shipped tensor
(INTENT.md G1; EXECUTION_PLAN_2026-10-12 P0.4). Pure, so it is tested without a hub repo.

Rule: switch away from the declared class only on file-side evidence (a candidate places strictly more of the shipped
tensors, or fits exactly). The single best candidate is selected. When several candidates tie for best, no evidence
chooses between them: the row gets a `class_ambiguous` unknown (never FULL), one tied class runs (a fixed order,
recorded as arbitrary), and only the passes EVERY tied class supports run, so nothing class-specific is decided by
name order.
"""


def choose(declared, scored, qualifies):
    """declared: the declared class name. scored: [(score, class_name)] for the other candidates, scores compared as
    tuples: [0] = shipped (file-side) tensor numel the class cannot place, [1] = model-side size (parameters the file
    does not fill); lower is better, compared as a tuple (pre-P0.4 ordering, unchanged). qualifies(score): the
    file-side evidence that the candidate beats the declared class. Returns {"selected", "tied", "ambiguous"}."""
    ok = [(sc, c) for sc, c in scored if qualifies(sc)]
    if not ok:
        return {"selected": declared, "tied": [], "ambiguous": False}
    best = min(sc for sc, _ in ok)
    winners = sorted(c for sc, c in ok if sc == best)
    return {"selected": winners[0], "tied": winners, "ambiguous": len(winners) > 1}


def shared(tied_classes, attr):
    """True when every tied class has `attr` (a pass or entry point it supports); True when there is no tie."""
    return all(hasattr(k, attr) for k in tied_classes)


def authority_qualifies(s0):
    """Library-authority path: a candidate beats the declared score s0 when it places strictly more shipped tensors,
    or fits exactly (nothing unplaced, nothing uninitialized)."""
    return lambda sc: sc[0] < s0[0] or tuple(sc) == (0, 0)


def matcher_qualifies(base0):
    """Weight-name matcher path: a candidate beats the declared class when strictly fewer shipped numel are uncovered."""
    return lambda sc: sc[0] < base0


def resolve(path, declared, scored, qualifies):
    """The worker's whole class decision for one path, pure. Returns
      selected: the class name to run; switched: selected != declared; best: its score (None when not switched);
      tied: tied class names (empty unless ambiguous); record: R["class_ambiguous"] or None; unknown: the G1 unknown
      or None."""
    ch = choose(declared, scored, qualifies)
    switched = ch["selected"] != declared
    best = dict((c, sc) for sc, c in scored)[ch["selected"]] if switched else None
    out = {"selected": ch["selected"], "switched": switched, "best": best, "tied": [], "record": None, "unknown": None}
    if ch["ambiguous"]:
        out["tied"] = list(ch["tied"])
        out["record"] = {"path": path, "declared": declared, "tied": list(ch["tied"]), "ran": ch["selected"],
                         "scores": {c: list(best) for c in ch["tied"]},
                         "note": "no file evidence chooses between the tied classes; the class run is the first in a "
                                 "fixed order (arbitrary) and only passes and inputs every tied class supports run"}
        out["unknown"] = {"predicate": "G1", "what": f"which library class loads this checkpoint is ambiguous ({path})",
                          "items": list(ch["tied"]), "count": len(ch["tied"])}
    return out


def shared_input(tied_forward_params, name):
    """An input / flag is used only if every tied class's forward accepts it (True when there is no tie).
    tied_forward_params: one set of forward parameter names per tied class."""
    return all(name in ps for ps in tied_forward_params)


def entry_allowed(tied_classes, tied_subtypes, module_name, meth):
    """A reflected entry method (or encode/decode) on `module_name` ("" = the root) runs only when it is the same thing
    in every tied class: at the root, every tied class has the method; on a sub-module, every tied class has a
    sub-module of the same type there. True when there is no tie. tied_subtypes: one {module_name: type name} per
    tied class."""
    if not tied_classes:
        return True
    if not module_name:
        return shared(tied_classes, meth)
    types = {st.get(module_name) for st in tied_subtypes}
    return len(types) == 1 and None not in types
