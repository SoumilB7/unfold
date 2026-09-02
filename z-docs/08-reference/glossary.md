# Glossary

**Address identity** — a model/class/repository identifier used only to locate
source or produce a label.

**Architectural fact** — a typed claim about a mechanism or geometry owned by a
specific component.

**Applicability** — whether a source-proven component or mechanism belongs to a
specific semantic output. Applicability is established before rendering; the
absence of a drawn element is not proof that a fact was inapplicable.

**Binding** — attaching an observation or config occurrence to the exact
component/source owner it describes. Binding is not consumption.

**Completeness** — whether an inspection can prove only presence or also absence
within its scope.

**Config occurrence** — one exact component path, config path, actual spelling,
canonical field, and value state.

**Consumption** — using an occurrence/fact to decide a named architectural or
geometry target.

**Evidence graph** — owner-qualified typed facts plus their observations,
premises, completeness, provenance, and failures.

**Identity debt** — any flow in which model/class/repository identity reaches a
structural sink rather than merely an address/display sink.

**Oracle missing** — modeling source was unavailable. It is not proof that a
mechanism is absent.

**Projection** — a consumer-specific representation of established semantic
structure, such as a block, card, JSON node, or parameter formula.

**Projection receipt** — structured evidence that a specific fact/target was
consumed by a projection.

**Producer** — the earliest stage that creates ownership, reachability,
mechanism or fact meaning. If a downstream check exposes a false producer, fix
the producer rather than weakening the check.

**Phantom consumption** — a config occurrence attached to a component or target
that the actual source structure does not own. It is a producer/ownership bug,
not a harmless missing-render condition.

**Raw program index** — source syntax indexed without assigning architectural
meaning.

**Semantic graph** — normalized architecture independent of a particular view.

**Structural sink** — an IR/spec field, opgraph node, block/card claim, renderer
structure, or parameter assumption that communicates architecture.

**Unknown** — evidence does not justify a value. It is distinct from false,
missing config, and explicit null.
