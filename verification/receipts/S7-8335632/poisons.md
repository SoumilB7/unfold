# S7 v2.6.2 final poison index

The final correction is pinned at the authority seams, not only at helper-level
happy paths.

## Claim authority

1. A constructor/existence proof offered to a connection consumer is rejected.
2. A config/value proof offered to an applied-function or relation consumer is
   rejected.
3. A reader-declared value fact with exact checkpoint evidence qualifies without
   a fact-name allowlist.
4. A fact with a declaration but no typed proof remains separately visible from
   a fact with no declaration.
5. A forged prepared-document token, equal-looking foreign document, wrong path,
   wrong checkpoint value, wrong status, wrong reader, or wrong owner is rejected.
6. A constructor site not identical to the authoritative ProgramIndex object is
   rejected.
7. Removing one product block from a scratch IR flips the cited occurrence back
   to `projection_unresolved`.

## Recipe and retry authority

8. All 39 targets have a persisted resolution and an attempted recipe or typed
   resolution failure; an empty attempt set cannot pass.
9. Checkpoint dtype and recipe execution dtype are independent closed records.
10. Wrong operator, wrong final dtype error, a detached generic dtype message, or
    a trailing unrelated diagnostic cannot trigger a retry.
11. The accepted grouped-matmul diagnostic permits exactly one bf16 retry and
    changes no other recipe input.
12. Direct checkpoint bf16, absent dtype, explicit-null dtype, and ambiguous dtype
    remain distinct controls.

## Relation authority

13. Container selection uses exact runtime type plus one ordered invocation of
    every direct member; partial, repeated, heterogeneous, rival, or out-of-order
    containers stay typed unresolved.
14. `ModuleDict` and arbitrary similarly-shaped classes never acquire ordered
    execution semantics.
15. `x + x`, wrong-axis matmul, discarded matmul, mixed-plus-unmixed, opposing
    mixes, in-place cancellation, and zero-scaled mixes cannot prove a
    multi-stream residual relation.
16. Arbitrary tensor multipliers carry unknown coefficients; they are never
    silently treated as one.
17. Matrix-contraction rows must round-trip to the exact layer, output lineage,
    source fingerprint, and source line.
18. Gemma3n/DeepSeek-V4 positive controls and DeepSeek-V3/SD3.5 negative controls
    are checked from the persisted artifacts.

## Isolation and closure

19. A constructor network attempt is refused; timeout and memory-cap failures are
    typed.
20. Linux attestation requires the OS network namespace and `ENETUNREACH`; the
    Python audit hook is not accepted as equivalent.
21. Every unresolved axis carries exactly one v2.6 reason class; class 3 requires
    a real investigation receipt and concrete reason.
22. Physics remains shadow-only: adapters and renderers have no production import
    or consumer.
23. Preservation is 52/52 byte-identical and every coordinator lane verifies an
    unchanged tree and artifact fingerprint.
