# Architectural change protocol

## Before code

1. State the mechanism without naming the reported model.
2. Name the exact owner and component path.
3. Identify the current source/config observations and their completeness.
4. Trace every consumer: IR/spec, op graph, cards, expanded views, renderers,
   parameters, audits, fixtures, and visual witnesses.
5. Freeze preservation evidence without re-blessing.
6. Choose positive, equivalent, collision, partial-source, missing-source, and
   unrelated preservation witnesses.
7. Predict every expected output delta.
8. State the earliest current author of the wrong claim.
9. State why applicability is active, inactive, or unresolved without using the
   renderer's output as evidence.

## Implementation stages

```text
observe -> bind -> interpret -> normalize -> project -> verify -> delete old path
```

Do not collapse these steps into one model helper. Compare a replacement with
the ship path when required, cut over narrowly, and remove the superseded
authority path in the same mechanism change. A temporary bridge requires an
exact owner, reason, removal condition, and explicit user decision.

## Producer-first failure rule

When a new check fails, do not first change the check. Trace the complete chain
and repair the earliest invalid author. The following are prohibited as a
default response:

- changing blocking to advisory;
- conditioning an obligation on whether a downstream node was drawn;
- adding an allowlist, ignore, or indefinite pending row;
- broadening an exact owner/path/mechanism match;
- retaining a known false consumption and adding a second diagnostic for it.

If the producer correction exceeds the current change, stop and report the
dependency. A green gate is not more important than evidence direction.

## Definition of a mechanism-level fix

- no model/repository identity selects structure;
- source/config roles follow the authority boundary;
- ownership is exact;
- ambiguity and unavailable source remain explicit;
- every consumer uses the same fact;
- the reported model passes;
- an equivalent implementation passes unchanged;
- a counterexample does not overgeneralize;
- an active component with its projection deliberately removed fails;
- an inactive candidate with similar config fields produces no positive fact;
- preservation and visual inspection show no unexplained breakage.

## Review questions

- What exact evidence proves the claim?
- Could a sibling component with the same field clear this obligation?
- What happens when aliases disagree?
- What happens when source is missing or syntax is unsupported?
- Does any renderer or parameter path recompute the fact?
- Did a YAML/data table gain architectural authority?
- Was the old fallback actually removed?
- Which current view changed, and why?
- Did any downstream artifact decide upstream applicability?
- Did the change shrink production authority or only add another rail?

## Commit boundary

One high-conflict substrate or mechanism change per reviewed commit. Do not mix
unrelated tests or future work into the same change. Record exact checks and the
unchanged tree fingerprint. Also report production lines added/deleted, old
authority paths removed, and temporary bridge growth.
