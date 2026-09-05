# S7 v2.6 poison index

The permanent S7 poisons live in `tests/test_s7_reconciliation.py`,
`tests/test_s7_relation_observation.py`, `tests/test_s7_relation_source.py`,
`tests/test_s7_artifacts.py`, plus the Linux boundary control in
`tests/test_s6_physics.py`.

1. Unknown authority and axis vocabularies are rejected.
2. Every unresolved construction, execution, projection, or relation value
   requires exactly one of the three §1g reason classes.
3. `mechanism_unresolved` without a typed recipe/reader/closure investigation
   and one closed concrete reason is rejected.
4. A completed investigation cannot be attached to
   `investigation_missing` or `structure_unaccounted` to blur the classes.
5. A valid investigated mechanism unknown remains visible but does not trigger
   S7's class-1/class-2 blocking gate.
6. A serialized class-3 lookalike with a missing, malformed, or extra-field
   investigation record is rejected by the persisted-artifact gate.
7. An observed execution axis carries no mechanism or fact field; a trace
   cannot promote itself into semantic understanding.
8. Every per-model matrix class count is independently re-derived from the
   gzip occurrence tables and must match.
9. Two projection values for one occurrence block; they are never voted or
   merged.
10. Removing one exact product block from a scratch IR flips the corresponding
    primitive occurrence to `projection_unresolved` /
    `structure_unaccounted`.
11. Instance count 38 versus static count 0 becomes a blocking construction
    conflict while preserving all 38 instance rows.
12. An empty or incomplete occurrence denominator fails anti-vacuously.
13. Config-document hash drift and foreign successful-observation provenance
    both block the join.
14. Runtime class identity cannot stand in for static meaning provenance.
15. A trace alias carrying two addresses cannot certify either address.
16. A custom runtime relation without an exact source proof remains an
    investigated `relation_unresolved`; a mismatched proof cannot clear it.
17. Parameter identity is accepted only for the definitionally exact
    `param_share` relation.
18. A relation endpoint outside the occurrence denominator is rejected.
19. A projection fact key not backed by the carried typed fact is rejected.
20. Duck-typed authority objects and self-grouping projection claims are
    rejected.
21. Publishing a class-1/class-2 unresolved table activates the blocking
    `reconciliation_axes` Sable check.
22. Relation address, lineage, stack, recipe-version, and source-span forgeries
    remain rejected.
23. The hard relation set, exact Gemma3n reuse edges, and Nemotron's aperiodic
    52-member schedule remain pinned.
24. Missing, stale, source-drifted, hash-drifted, or reason-count-drifted
    shadow artifacts make the checker red.
25. Two Torch diagnostics differing only in date/time/PID serialize identically;
    a changed exception still changes the payload.
26. Linux CI must enter a real OS network namespace; audit-hook-only mode is
    explicitly rejected as non-equivalent.
27. Any `physics` import from adapters or renderers remains forbidden.
