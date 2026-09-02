# Problem and failure reporting

## Required problem report

When a problem is found, report:

1. the visible false, missing, or inconsistent claim;
2. the exact model component and owner;
3. the evidence that exists;
4. the evidence that is absent, ambiguous, or unreadable;
5. the first pipeline boundary where truth is lost;
6. every affected projection and equivalent model class;
7. whether the defect is acquisition, binding, interpretation, normalization,
   projection, parameter, verification, or presentation;
8. the smallest general capability required;
9. expected regression risk and witnesses;
10. whether a user decision is needed.
11. whether the proposed correction changes the producer or merely relaxes a
    downstream check.
12. the old path that will be deleted.

## What not to do

- Do not add a family/model special case.
- Do not choose the first alias.
- Do not translate unknown to zero or a familiar default.
- Do not add an unused read to clear an audit.
- Do not solve only in labels, cards, or one renderer.
- Do not infer an individual owner from a union of sibling evidence.
- Do not catch every exception and return absence.
- Do not refresh expected artifacts before explaining the delta.
- Do not describe a partial conversion as the final architecture.
- Do not infer applicability from whether the renderer drew a node.
- Do not turn a known false consumption into permanent advisory debt.
- Do not call a change principled until a poison independently breaks the
  producer and the consumer.

## Correct stop behavior

If the current evidence cannot justify the architecture, stop the structural
claim—not the investigation. Surface the typed state, affected owner, source
failure or ambiguity, and the missing interpreter capability. Unknown is the
correct product response until that capability is implemented.

If a new invariant reveals that the real correction exceeds the promised
commit, report the expanded producer defect and pause the commit. Do not preserve
the schedule by weakening the invariant.
