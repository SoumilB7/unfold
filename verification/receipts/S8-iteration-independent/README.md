# Independent iteration integration — ACCEPT within scope

Reviewed immutable commit 45604499d44b3131a75777da1df32ebac6da20f3 at /private/tmp/unfold-s8-invocation-review. Source hashes are recorded in results.json and fingerprints.json; all six reviewed source files match after replay. Iteration reader f189e12531b06fc916703fb45882510e81aa1cba6406ee7428e36252e3266412; production caller f6a1d4c1b179eb81de2f4315f7c885be5a03be648bf845b846d19648dc882bad.

Command:

    python3 verification/receipts/S8-iteration-independent/replay.py /private/tmp/unfold-s8-invocation-review

The script lives in the primary checkout's receipt and imports production code from the explicit immutable checkout. It runs seven existing control functions directly, without pytest, then tiny synthetic inventory fixtures and a static replay of fresh v2 saved SDXL evidence. No full model construction, model forward execution, production edits, commits or blessings.

## Production stability is computed

Unlike the earlier reader-only replay, this replay calls read_root_invocations. Its root/body parent closure must succeed; the caller then computes _member_stays_bound at the loop-end boundary and supplies that result to read_iteration_binding. Instrumentation delegates to the original functions and records their actual results; it never substitutes True or False.

A synthetic mutating self helper replaces the container before iteration. Production rejects its body closure and emits no iteration targets. Rejection happens before the per-loop reader, which is the correct earlier boundary. The ordinary synthetic case proceeds with computed member stability True. The existing missing-parent-stability control also verifies direct reader refusal when the premise is absent.

## Exact slots, source calls and conditions

The ordinary tiny constructed fixture uses the same child object in two ModuleList slots plus None. Production preserves stages.0 and stages.1 as distinct occurrence addresses, preserves the None slot, and assigns no callable target to None. Its bound call keeps the source guard and a per_iteration condition carrying exact slot indices/paths.

Fresh v2 saved actual SDXL inventory/source replay independently computes positive parent stability for:

- down_blocks loop at source line 1138: three recorded slots, two guarded source call sites, six conditional call/slot pairs;
- up_blocks loop at line 1198: three recorded slots, two guarded source call sites, six conditional call/slot pairs.

Every returned target references an actual call record in the indexed callable and its occurrence path equals the selected recorded slot's path. Source guards remain attached. All current installed source bytes match the saved source hashes before replay. No class-name or source-segment matching chooses the loop in this production replay.

The reader establishes a possible target when that slot is selected and that source call is reached. The call/slot combinations do not prove every iteration executes, that both alternative calls execute, or an observed order between slots. Empty and None slots remain explicit; no execution closure is inferred. Production conditions preserve the lookup and inherited helper exclusions, while the per-call records retain source guards.

## Seven controls

All seven existing test functions PASS, including their multiple cases: direct/enumerated selection and None; separate canonical length witness; container alias/escape/write and dynamic lookup refusal; local/nested/import/definition rebinding and after-loop refusal; missing parent/iterator and shadowed enumerate refusal; empty container and conditional lookup; boolean guards without hidden container effects.

No concrete authority defect was established within this bounded iteration integration review. Actual HTML placement, visible condition presentation, source-to-fact projection and full S8 exit remain separate owner reviews. This acceptance does not bless outputs.
