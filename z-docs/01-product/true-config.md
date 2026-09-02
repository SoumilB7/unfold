# True config — the complete config a checkpoint should have shipped

> **Status: `#TODO` — not yet implemented.** This is a plan only. Nothing here
> exists in the code today; `true_config()` is to be built later.

## What it is

`true_config(x)` reconstructs the config a model *should* have carried: the
checkpoint's own field spellings and nesting, made complete and correct by filling
in every field the modeling code reads but the shipped `config.json` left implicit.

The output is a config, not a report. Diffed against the repository's own
`config.json`, every added line is a field that legitimately belongs there —
resolved from a class default, a code-derived default expression, or a corrected
value — never an invented structural annotation and never provenance clutter.

This is the fourth projection of the one architecture declaration, beside the SVG
diagram, the JSON schema, and the cards. It is a serialization of already-resolved
evidence, not a second interpreter: it never recomputes a fact.

## The field-belonging rule

A field belongs in a true config when, and only when, the modeling code reads it
from config.

- Included: `hidden_size`, `n_inner`, `sliding_window_pattern`, `rope_theta`,
  `attention_bias`, `qk_layernorm`, `moe_layers` — every config value the
  architecture consults.
- Excluded: gating, norm placement, attention kind, fused storage. These are not
  config fields. A real config does not carry a `gated` key; those facts come from
  the modeling code that the model type selects, and they remain in the diagram and
  facts, not in the config.

This rule is what keeps the output a config rather than a bloated dictionary. The
field set is **complete** (nothing the code needs is missing) and **minimal**
(nothing the code does not read is added).

The set of legitimate field names is bounded by the [canonical config
reference](../08-reference/canonical-config.md): a curated superset of the config
fields that carry architectural meaning across the supported mechanism surface. A
true config is the intersection of that reference with the fields this model's code
actually reads, populated with resolved values. A code-read field absent from the
reference is surfaced and flagged rather than silently emitted; it is a signal to
extend the reference, not to guess.

## How each field is resolved

For every field the code demands, the value is resolved through the existing
evidence ladder, in order:

1. the checkpoint's serialized config value;
2. the installed config class default (the hydration channel);
3. a code-derived default expression, when the modeling source computes the value
   itself (for example a feed-forward width defined as a multiple of the hidden
   size when the explicit field is absent);
4. unresolved — never a guess.

Sub-configurations (`text_config`, `vision_config`, `audio_config`, and similar)
recurse through the same stages under their own component source, so a composite
model produces a composite true config with each component completed in its own
scope.

## Honesty boundaries

- A field the code reads but nothing can resolve is not fabricated. It is omitted
  from the config and reported as an unresolved demand — a real gap, made visible.
- Structural facts without a config field are excluded, not invented.
- A dead flag — a config value the code ignores — is normalized to the value the
  code actually enacts; the original is recorded outside the config.
- When source is unavailable, the true config is the present config plus an explicit
  note that completeness could not be verified. It never fills the unknown fields
  from convention.
- Provenance and the config-versus-true diff live in a separate companion, never
  inside the config object.
- When ingestion normalizes a foreign config dialect, the companion retains the
  exact original path and spelling for every canonical field. The normalizer may
  not manufacture a model type or class merely to make source lookup succeed.

## Why this is the natural capstone of the config-to-code direction

The product's direction is that structure is proven from source and config supplies
selected values. A true config is that direction expressed as an artifact: it states
exactly which config fields the architecture depends on, and their true values,
having discovered the complete set by traversing the code rather than trusting the
checkpoint to be complete.

Its completeness is bounded by how much of the architecture the evidence layer
already resolves. Where a demanded field cannot yet be attributed, the companion
says so. As mechanism coverage closes, the unresolved set shrinks to nothing, so the
true config doubles as a coverage measure for the whole direction.

## Delivery shape

- a projection that reads the resolved model representation and its evidence, with
  no source re-resolution and no fact recomputation;
- a config-shaped assembler that keeps the model's native field spellings and
  nesting and recurses sub-configs;
- the demand scan that enumerates every config field the source reads, per
  component and per scope;
- a separate companion carrying provenance and the config-versus-true diff;
- verification: a demanded field is resolved or explicitly unresolved; re-running
  the product on a true config reproduces the same architecture signature; no field
  the code never reads appears.
