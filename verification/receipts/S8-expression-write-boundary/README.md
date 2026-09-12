# Expression-local write refusal

Independent9fe review found a stale sibling operand in an if-expression branch: a walrus write inside the selected tuple could change a later operand, but that operand still resolved to the old formal. The local reader now refuses any expression containing such a named write before routing its siblings. This covers the exact if-expression counterexample and analogous call/tuple actuals without implementing evaluation-order interpretation. Source conditional expressions without writes retain both routes.

49 primary/route/layout checks passed in0.31seconds, including three expression-local write controls. No architecture or renderer change is claimed for ordinary SDXL from this refusal. Final models must still run against the frozen corrected source.

The ordinary9fe process succeeded and archived all12projected facts, but its report correctly stopped before dependent controls: insertion-order block traversal differed after sorted-key JSON serialization. The separate mapping-order receipt fixes only mapping traversal and preserves list-order sensitivity; original artifacts remain unchanged in S8-final-9fe0868-report-stop.

No outputs blessed or pushed.
