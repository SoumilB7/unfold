# Source acquisition and static analysis

## Source resolution

`model_unfolder/evidence/sources.py::resolve_source_files` resolves modeling
source without executing it. Its current routes include:

- explicit local files/directories;
- installed Transformers source;
- installed Diffusers source;
- component and wrapper source bundles;
- repository source for declared remote-code models;
- Hub source in the supported modes.

Identity is lawful here as an address: a declared architecture/class can locate
the file that defines it. Structural decisions happen only after that source is
read.

For installed Transformers packages, source addressing delegates config-type
normalization to the installed library's own auto-configuration registry and
then checks that the named module directory exists. The project must not carry
its own model-type-to-directory table, and it must not guess a parent source
package by chopping suffixes such as `_text` or `_vision`. If the installed
registry cannot resolve the component, the source remains unresolved unless an
exact declared-class route later finds it.

## Parse once, reuse by owner

`ParseContext` owns the source bundle and component-qualified registry caches.
Readers should consume this context rather than independently resolving files.
Recursive component parses must preserve an ambient ownership namespace and
must not repeat the same sub-parse merely to derive a label or summary. Repeated
source resolution or context-less component parsing is both expensive and a
correctness risk.

## Static readers

The evidence package includes readers for:

- literal forward operations;
- transitive callable closure;
- construction and dataflow patterns;
- FFN, positional, vision, audio, projector, fusion, and secondary-stack
  evidence;
- model, wiring, fact, and nested conformance.

These readers parse text/AST. They do not instantiate model classes or load
weights.

## Reader result law

A reader must return evidence plus completeness/failure state. Important
outcomes include:

- found and complete;
- found with presence-only or partial completeness;
- ambiguous ownership or interpretation;
- unsupported syntax;
- missing source;
- reader error.

A reader exception must not become `False` or “mechanism absent.” The current
tree still contains a ratcheted population of broad exception handlers; new
ones are forbidden and existing ones are scheduled for removal.

## Extraction architecture direction

`forward_ops.py`, `transitive.py`, and specialized readers still contain some
overlapping extraction and interpretation responsibilities. The binding
direction is a shared raw program index, an owner-bound program graph, and
typed interpreters. Until that conversion lands, do not build another parallel
scanner.
