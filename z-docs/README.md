# Model Unfolder documentation

This is the current product and engineering context for `model-unfolder`.
It describes the product we are building, the boundaries the code must enforce,
the current implementation state, and the finite route to release.

The shortest accurate description is:

> Model Unfolder is a source-grounded architecture explainer. It supports a
> bounded, measured set of common mechanisms and renders unresolved mechanisms
> honestly instead of guessing a familiar structure.

## Reading order

1. [Start here](00-start-here/README.md)
2. [Product contract](01-product/README.md)
3. [System architecture](02-architecture/README.md)
4. [Evidence architecture](03-evidence/README.md)
5. [Domain adapters](04-domains/README.md)
6. [Verification](05-verification/README.md)
7. [Development protocol](06-development/README.md)
8. [Current state](07-current-state/README.md)
9. [Reference](08-reference/README.md)
10. [Unit verdicts U0–U10](09-unit-verdicts/README.md)
11. [Full research](10-full-research/README.md) — the whole project understood before it is judged: goal, codebase map, pipeline trace, reader coverage, surfaces, verification, history, alternatives; then the findings register, judgment and the post-U plan (2026-09-01→) — dated audit of every completed unit against the goal (2026-08-31)

## Document authority

- Chapters `00` through `06` describe durable intent and engineering rules.
- Chapter `07` is the only live status/tracker area. It must be rewritten when
  state changes; dated execution bookkeeping does not accumulate there.
- Chapter `08` records the code evidence used to validate these docs.
- Chapter `09` is a dated audit; it judges completed units against the goal
  and never overrides `07`.
- Implementation plans under `unfold-pkg/docs/` may supply detailed worklists,
  but they never override the one-way evidence law or current status here.
- `z-docs/stale/` is a quarantined archive. It is excluded from operational
  reading, status, terminology, and implementation authority.

## What does not belong in active documentation

The active chapters exclude dated session narratives, model-by-model war
stories, old test counts, superseded trackers, stale line numbers,
agent-specific motivation, and completion claims unsupported by the current
tree. See the
[documentation boundary](08-reference/documentation-boundary.md).
