# Transformer adapter

## Entry and assembly

- `model_unfolder/adapters/transformer/parser.py` resolves transformer
  declarations and evidence into layer/model specs.
- `model_unfolder/adapters/transformer/assembly.py` and the files under
  `adapters/transformer/blocks/` construct shared block structure.
- `model_unfolder/expanded/` and the HTML block views project deeper details.

## What may come from config

Checkpoint geometry, layer counts, exact declared schedules/enums, and selected
values may come from configuration after exact ownership and semantic binding.
Aliases normalize syntax only.

## What must come from code or typed mechanism evidence

Examples include FFN gating/storage, attention implementation/storage,
positional application, norm math and placement, construction topology,
secondary stacks, and the meaning of a config-controlled branch.

## Heterogeneous layers

Layer schedules are instances of a shared mechanism. A list, repeating pattern,
interval, or construction loop must normalize to per-layer semantic cells. A
new model name is never a schedule mechanism.

## Unknown behavior

The adapter now preserves unknown core geometry instead of substituting zero.
Activation alias conflicts are resolved once and remain ambiguous. Every new
field consumer must preserve this behavior through layer assembly, expanded
views, rendering, and parameter estimation.

## Remaining seams

- large parts of the parser still interleave acquisition, interpretation,
  normalization, and projection preparation;
- transitional spec fields and extras coexist with typed facts;
- schedule and role vocabulary still carries transitional config authority;
- all config consumers have not yet reached blocking exact projection receipts.
