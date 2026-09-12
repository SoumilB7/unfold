# Eleventh A5 correction, bounded independent source review

RETURN for one remaining direct reflective-mutation form. No runtime or output approval.

The two-file correction closes the reported local alias setattr/delattr, name-to-field alias chain, immediate getter-to-field/subscript storage, reflected selected getter, and parameter/local/enclosing-callable shadowing cases. Ten actual source poisons are present. Earlier A3/S1/A4/P1 verdict scopes are unchanged.

Remaining concrete source case: `requested = "nn"; setattr(getattr(torch, requested), "ReLU", replacement)`. The getter's result is supplied directly to the builtin mutator. `_ancestor_read_has_local_mutation` discovers an Expr origin but it has no assignment targets and hence no aliases; `if not aliases: continue` skips the origin. The earlier direct-target scan only inspects statement assignment targets. No local alias Name exists for the later alias-access loop to visit.

Extend the bounded exact-expression use check to a getter expression supplied directly as the first argument of the already-enumerated reflective mutators/accessors (at least setattr/delattr). Add that source poison. This is explicit captured syntax, not a requirement for arbitrary external call-effect analysis. Root and executor have the exact case. Prior findings remain preserved, not relabelled.
