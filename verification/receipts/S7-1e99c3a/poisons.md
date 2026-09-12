# S7 poison index

The permanent S7 poisons live in `tests/test_s7_reconciliation.py`,
`tests/test_s7_relation_observation.py`, `tests/test_s7_relation_source.py`,
`tests/test_s7_artifacts.py`, plus the Linux boundary control in
`tests/test_s6_physics.py`. A green poison test means the injected false claim
was rejected or remained visibly unresolved.

1. Unknown authority or axis vocabulary is rejected by the closed enums.
2. Two projection values for one occurrence block; they are never voted or
   merged.
3. Instance count 38 versus static count 0 becomes a blocking construction
   conflict while preserving all 38 instance rows.
4. An empty or incomplete occurrence denominator fails anti-vacuously.
5. Config-document hash drift and foreign successful-observation provenance
   both block the join.
6. Runtime class identity cannot stand in for static meaning provenance.
7. A trace alias carrying two addresses cannot certify either address.
8. A custom runtime relation without an exact source proof remains
   `relation_unresolved`; a mismatched class/source proof cannot clear it.
9. Parameter identity is accepted only for the definitionally exact
   `param_share` relation.
10. A relation endpoint outside the occurrence denominator is rejected.
11. A projection fact key not backed by the carried typed fact is rejected.
12. Duck-typed authority objects and self-grouping projection claims are
    rejected.
13. Publishing any unresolved table in canonical IR activates a blocking
    `reconciliation_axes` Sable check.
14. A non-container relation address, forged lineage, empty stack, or false
    recipe library version yields a typed failure.
15. A recurrent-state source proof disappears when the recombination return
    disappears.
16. A post-stack collapse proof disappears when the head executes before the
    stack or does not reach the return.
17. Relation proof spans must carry the exact class source fingerprint.
18. Missing, stale, source-drifted, or hash-drifted shadow artifacts make the
    committed matrix gate red.
19. The hard relation set and exact Gemma3n reuse edges are pinned.
20. Nemotron's 52-member aperiodic schedule must retain its three exact groups
    and `period=None`.
21. Linux CI must enter a real OS network namespace; audit-hook-only mode is
    explicitly rejected as non-equivalent.
22. Any `physics` import from adapters or renderers remains forbidden.
