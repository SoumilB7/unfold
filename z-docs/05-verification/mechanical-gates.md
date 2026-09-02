# Mechanical gates

## Sable

`model_unfolder/sable.py::sable` parses with the same source-resolved ship path,
renders all baked graphs in a call-local render context, and currently runs
checks covering:

- click coupling and unique references;
- dangling connectors and dotted-arrow/boundary rules;
- unread config fields;
- op, wiring, fact, and nested conformance when source is reachable;
- label lint and evidence ambiguity;
- asserted-fact and zero-asserted censuses;
- fact projection audit;
- accessed/consumed config projection diagnostics;
- config audit incompleteness and alias ambiguity.

Do not document Sable as a fixed check count. Checks and blocking coverage
states evolve; `sable.py` is authoritative.

## Conformance

Conformance compares the diagram with source evidence in both directions:

- missing code-proven operations;
- fabricated drawn operations;
- wiring/dataflow disagreement;
- semantic facts that op-presence cannot distinguish;
- transitive leaf-drill closure.

If the source oracle is missing, the report must say conformance was degraded.

## Identity and structural guards

- `evidence/identity_guard.py` detects identity-derived structural decisions and
  maintains a fingerprinted manifest for narrowly lawful vocabulary.
- `evidence/structural_writes.py` inventories structural author surfaces.
- static tests enforce dependency/firewall and no-growth rules.
- metamorphic tests exercise rename, collision, missing-source,
  partial-source, and equivalent-control relations.

## Anti-vacuity

Every guard needs a poison case that fails for the intended reason. A green test
without a firing negative control is not proof that the rule is wired to the
production path.

## Test isolation

Shared fixtures belong in the importable `test_support` package. Tests must not
import other test modules. Each audit file must pass alone and in the grouped
gate.
