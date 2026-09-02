# Multimodal components

## Component model

Transformer-hosted modalities are constructed through:

- `special_parts/modalities/accessors.py`;
- `registry.py` and `builder.py`;
- `vision.py`, `audio.py`, `conditioning.py`, and `fusion.py`;
- the specialized evidence readers in `model_unfolder/evidence/`.

The registry identifies component slots and builders. It must not become a
family-to-architecture table.

## Exact scope

Each nested config read carries both its semantic owner and its exact wrapper
path. Examples include `vision_config.*`, `_text_encoder_configs.<slot>.*`,
`_vae_config.*`, and `_scheduler_config.*`. Compatibility leaf labels are not
truth keys.

## Vision and audio towers

Tower norm, FFN, attention, patch/feature geometry, and repeated-cell structure
must come from the resolved component source plus its owned checkpoint values.
The class name may help locate or label the component but cannot select its
internal architecture.

## Projectors

Projector input/output width and operation sequence belong to the resolved
projector component. A generic language/vision `hidden_size` is not a safe
substitute. If source cannot uniquely bind the projector field or constructor
expression, the projector claim stays unknown.

## Fusion

Fusion is determined by structural flow: prefix/concatenation, cross-attention,
joint streams, projections, and conditioning edges. A model-family hint may not
select the fusion diagram.

## Current implementation status

Exact container scoping and source-authoritative projector width binding are
implemented. The active working tree further carries the recursive component
namespace through diffusion text-encoder parsing and nested transformer
modalities. A vision component inside a text encoder is therefore owned beneath
that encoder rather than being misreported as a pipeline-level vision tower.

The same working tree pilots typed projection receipts for projector output
width. That pilot is not current-state truth until its unchanged-tree
verification finishes. See `07-current-state/implementation-state.md` for the
live boundary between committed behavior and work under verification.
