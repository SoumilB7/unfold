# Shared infrastructure

## Why it is needed

The original implementation grew from a practical config-first visualizer.
Adapters read common fields, selected familiar structures, assembled view-ready
dictionaries, and renderers filled remaining gaps with conventional labels and
shapes. This worked for known happy paths but distributed architectural
authority across parsers, YAML, specs, renderers, JSON, and parameter formulas.

The same FFN or projector could therefore be identified differently depending
on whether it appeared in the main decoder, a vision tower, a text encoder, or a
diffusion component. Correct output could be coincidental: several independent
assumptions happened to agree.

## What shared means

Shared infrastructure does not mean one giant interpreter. It means one
implementation for each universal responsibility:

```text
source resolution       locate exact component files
program index            record syntax without architectural judgment
owner binding            identify constructed/reachable component instances
mechanism interpreters   turn bound behavior into typed facts
semantic normalization   represent the mechanism independently of presentation
projection consumers     render/serialize/count the same fact
receipts and checks       prove both evidence-to-output and output-to-evidence
```

Domain adapters still understand ecosystem packaging: pipeline slots, wrapper
configs, component addresses, and domain-specific entry points. They do not get
separate definitions of attention, FFN, projector, normalization, or ownership.

## The original failure mode

A generic component reader could inspect a nested encoder without preserving
its ambient ownership namespace. A second context-less parse could then report a
nested encoder's vision projector as the pipeline's top-level `root.vision`.
Downstream code saw a valid-looking width and consumed it even though that owner
did not exist in the rendered pipeline.

The correct shared fix is to carry the namespace through recursive component
binding and parse each component once. It is not to exempt the missing renderer
receipt. This protects every nested text, vision, audio, VAE, scheduler, and
secondary component using the same rule.

## What exists now

- call-local parse and render contexts;
- exact config occurrences with owner, path, alias, value state, intent, and
  mechanism;
- source bundles with component files;
- typed facts, provenance, completeness, and a closed registry;
- structural-write, identity, conformance, and preservation checks;
- canonical IR/op regions for part of the architecture;
- render events and an in-progress typed receipt channel.

## What is still being consolidated

- overlapping source scanners into one owner-bound program index;
- parser-local interpretation into reusable mechanism readers;
- temporary fact/spec/extras representations into canonical facts and regions;
- hand-maintained projection witnesses into actual consumer-emitted receipts;
- JSON, parameter, card, and conformance inference into pure fact consumers;
- semantic YAML and conventional fallbacks into source-bound evidence or honest
  unknowns.

## The scaling test

Shared infrastructure is successful when an FFN with equivalent code receives
the same fact and drill regardless of its containing model, and when a new
checkpoint changes only values. If supporting an equivalent implementation
requires a model-name branch, copied renderer, or new family table, the shared
boundary is incomplete.
