# Current implementation state

Verified against the workspace on 2026-07-16. This file describes the current
tree without dated execution labels or historical completion stories.

## Committed foundation

The current branch head provides:

- reproducible clean-checkout preservation for 25 representative models;
- call-local parse, config-access, and render state;
- exact config occurrences carrying owner, dotted path, actual alias, value
  state, intent, mechanism, and target;
- ambiguity handling that refuses first-alias selection;
- unknown-safe core geometry and incomplete parameter results;
- source bundles and component-qualified source provenance;
- typed evidence facts, completeness, premises, source spans, and a closed fact
  registry;
- owner-and-mechanism consumption claims with binding-level witness coverage;
- identity, structural-write, conformance, preservation, isolation, and
  anti-vacuity checks;
- source-authoritative modality projector width for the covered paths.

These foundations improve traceability and prevent several classes of silent
cross-component errors. They do not mean every mechanism is source-derived or
every consumer is a pure projection.

## Working tree under verification

The current uncommitted change adds the first typed projection-receipt slice:

- receipts are emitted by the real declared-operations drill consumer;
- each receipt identifies fact owner, mechanism, surface, node path,
  projection kind, and a value/status hash;
- receipt coverage is scoped by owner and mechanism instead of one global flag;
- the mechanical audit joins exact occurrence to fact target to receipt;
- reverse-fabrication checks reject receipts without registered evidence;
- modality projector output width is the pilot, with no intended pixel change.

The same working tree removes the old model-named params-format resource and
its model-named loader path. Foreign `params.json` ingestion now:

- recognizes a layered-transformer input dialect from file layout and the
  required `dim`, `n_layers`, and `n_heads` keys;
- resolves its text and vision spellings through scope-qualified entries in the
  shared alias vocabulary;
- keeps `dim` out of the global `hidden_size` aliases, where it would collide
  with other component meanings;
- refuses unequal duplicate declarations;
- preserves repository identity only as provenance; and
- does not manufacture `model_type`, an architecture class, or a mechanism.

Consequently, a source-less foreign config can still contribute checkpoint
geometry, but mechanism facts remain unresolved instead of borrowing an
installed model-family implementation.

The same working tree also removes the project-owned Transformers
model-type-to-source-directory table. Installed Transformers source resolution
now follows the library's own auto-configuration module registry, accepts only
directories that exist in the installed package, and refuses suffix-parent
guessing. Unknown or rebranded config types can still resolve through exact
declared-class source lookup, but they no longer get a project-maintained
family directory mapping.

The new receipt check exposed an upstream ownership defect in a nested diffusion
text encoder. The correct working-tree repair now:

- removes the duplicate context-less text-encoder sub-parse;
- derives encoder names and geometry from one namespaced parse;
- carries `ParseContext.component_namespace` through recursive slots;
- attributes a nested modality to `<parent namespace>.<modality>` rather than
  falsely promoting it to the pipeline's top-level modality owner.

This is the producer-first correction. Receipt enforcement is not excused based
on whether a renderer happened to draw a modality.

The preservation manifest is being regenerated because evidence surfaces and
check names changed. Until the unchanged-tree isolated, grouped, full-suite,
preservation, and visual checks finish, this working-tree slice is **under
verification**, not complete.

## Current architectural limitations

- the key-level `facts_projected` channel and the typed receipt channel overlap;
- only a small receipt scope has a real-consumer cutover;
- source readers still overlap instead of consuming one program index;
- transformer and diffusion parsers still combine acquisition,
  interpretation, normalization, and view preparation;
- many specs and `extras` leaves are temporary structural authors;
- cards, expanded JSON, parameters, and conformance still contain independent
  assumptions;
- semantic config/YAML authority remains in schedules, conditioning, diffusion
  config chips, component markers, and conventional defaults;
- UNet, VAE, scheduler, and several multimodal/heterogeneous mechanisms are not
  fully owner-bound fact-to-receipt paths.

## Debt measurement

The last committed corpus census recorded 267 exact accessed-but-unconsumed
occurrences and two owners with no consumption. That number is a worklist, not
267 required interpreters. Every occurrence must be classified as a bound
checkpoint value, address/declaration, display value, exact non-architectural
field, unresolved mechanism input, or unsupported input. Only mechanism-driving
occurrences require new interpretation code.

The census must be regenerated after the namespace and receipt change lands;
the old number must not be presented as current afterward.

## Product readiness

The package already renders a broad set of transformer, multimodal, audio, and
diffusion configurations. Its remaining conversion is about trust and
generality: removing coincidental correctness from config/family/default paths
and making every supported mechanism share one evidence and projection route.

The product can release without universal model coverage once the declared
support corpus has no known confidently wrong claims and out-of-scope mechanisms
degrade honestly.
