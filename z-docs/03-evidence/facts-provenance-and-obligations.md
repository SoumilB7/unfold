# Facts, provenance, and obligations

## Typed facts

`model_unfolder/evidence/facts.py::EvidenceFact` carries:

- mechanism-scoped key;
- exact owner;
- value;
- epistemic status;
- inspection completeness;
- structured source spans;
- exact config paths;
- premise keys for derived facts;
- typed failure context and a human reason.

Raw and bound observations are distinct types and cannot substitute for an
architectural fact.

## Closed registry

`model_unfolder/evidence/registry.py` registers allowed fact keys, owners,
statuses, value types, projection surfaces, unknown policy, parameter use, and
negative-proof requirements. A new fact requires a conscious registry change;
using a different representation must not bypass the structural-write census.

## Two symmetric obligations

Every structural fact creates two questions:

1. Evidence-to-projection: where is this established fact shown or consumed?
2. Projection-to-evidence: what evidence supports this drawn or estimated
   claim?

The first prevents read-but-never-drawn facts. The second prevents fabrication.

The obligations are not allowed to define their own applicability. An active
scope comes from source-bound component/mechanism evidence. Omitting a node in a
renderer cannot erase an obligation.

## Config projection obligations

`ConfigOccurrenceKey`, `ProjectionTarget`, and `ProjectionObligation` connect
one exact config occurrence to one exact target. Joining by a bare field name or
an approximate owner pair is forbidden.

## Structural write census

`model_unfolder/evidence/structural_writes.py` inventories structural authors
across facts, specs, extras, op graphs, cards, and parameters. It is a conversion
guard: new representations are not permission to escape the registry.

## Current implementation boundary

Typed facts and the closed registry are implemented, but not every spec,
`extras` leaf, renderer claim, or parameter assumption is authored by a native
typed fact. The correct direction is one mechanism at a time: measure, compare,
cut over, delete the superseded path, then prove equivalence and
counterexamples.

The current working tree adds typed projection receipts and a real-consumer
pilot for modality projector output width. It also corrects recursive component
namespacing so nested encoder modalities do not become top-level owners. This
work remains under full unchanged-tree verification and is not documented as a
completed release boundary yet.
