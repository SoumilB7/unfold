# End-to-end pipeline

## Implemented flow

```text
input
  -> config/repository coercion
  -> one call-local ParseContext
  -> source bundle + class/config declarations
  -> transformer or diffusion adapter
  -> config occurrences + static code observations
  -> owner binding and typed fact/temporary compatibility recording
  -> ModelIR
  -> structural blocks/op graphs/expanded schema
  -> HTML, cards, drills, JSON, parameters, render events
  -> receipts, mechanical checks, preservation, and visual review
```

## Input coercion

`model_unfolder/parser.py` owns the public loading ladder. It handles plain
objects and model IDs, raw configuration retrieval, diffusion repository
layouts, alternate parameter formats, refusal classification, and final adapter
selection.

## Parse context

`model_unfolder/evidence/context.py::ParseContext` is the call-local state
carrier. It currently owns:

- the resolved `SourceBundle`;
- component-qualified registry caches;
- the fact ledger;
- the config-access ledger;
- installed config-class defaults;
- declared decoder role;
- the ambient component namespace used by recursive component parses.

No parser or render state should depend on mutable cross-model globals.

## Adapter boundary

The root parser selects the transformer or diffusion adapter. Adapters may
coordinate domain-specific readers, but they must not bypass the evidence
contract. The adapter's job is to assemble facts and semantic structure, not to
invent a complete view from ecosystem conventions.

## Structural and projection boundary

`ModelIR` remains the main parser/consumer contract. Expanded schema builders,
HTML renderers, and parameter estimation consume it. Some facts are native typed
facts while other structural fields and `extras` remain temporary compatibility
surfaces; the structural-write census and fact registry make that boundary
visible.

The target is not to make `ModelIR` another interpreter. Mechanism meaning is
decided before projection, and every consumer reads the same canonical
fact/region.

## Error boundary

Load and parse failures should become typed, actionable errors or typed evidence
failure states. A broad reader exception that returns `None` may not silently
convert a tool failure into architectural absence.

## Applicability boundary

A component or mechanism is active only when ownership and reachability are
established upstream. A renderer does not decide applicability by drawing or
omitting a node. The allowed states are:

- active: facts may be consumed and required projections owe receipts;
- inactive/dormant: no positive structural fact is consumed;
- unresolved: the output is partial or opaque with a typed reason.
