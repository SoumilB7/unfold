# How to use these docs

## For a new contributor

Read the doctrine, public product contract, pipeline, evidence model, and change
protocol before editing a parser or renderer. Use the code map to locate the
current implementation rather than relying on a dated line number from an old
plan.

## For a bug

Start from the false or missing user-visible claim and trace backward, but fix
the earliest false author rather than weakening the downstream check:

```text
rendered claim
  -> projection receipt
  -> typed fact or geometry target
  -> interpretation
  -> owner binding
  -> exact config occurrence / source span
```

Stop at the first broken boundary. Fix that shared boundary, then test the
reported model, an equivalent model, a counterexample, missing source, partial
source, and an alias collision.

A renderer, card, receipt, audit, or expected artifact may never decide whether
an upstream component or mechanism was applicable. Applicability comes from
source-bound ownership and reachability. If the real correction exceeds the
current change, report that dependency instead of creating an advisory escape.

## For current status

Read only [the current-state section](../07-current-state/README.md). Dated
completion stories elsewhere are historical evidence, not current status.

## Updating these docs

- Durable laws change rarely and require an explicit architectural decision.
- Code maps and verification ledgers change whenever implementation moves.
- Current state is rewritten, not appended.
- Old counts, commit diaries, and dated execution labels do not enter durable
  chapters.
- Every implementation claim must link to a current code symbol or test.
