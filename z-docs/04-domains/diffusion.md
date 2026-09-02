# Diffusion adapter

## Entry and assembly

- `model_unfolder/adapters/diffusor/loader.py` and the root parser load pipeline
  and component configurations.
- `model_unfolder/adapters/diffusor/parser.py` assembles pipeline, denoiser,
  scheduler, VAE, text-encoder, conditioning, and geometry structure.
- `blocks.py`, `compound.py`, and `unet.py` build diffusion-specific structure.
- diffusion HTML and expanded projections consume that structure.

## Ownership

Pipeline root, denoiser, VAE, scheduler, and each text encoder are separate
owners. Identical leaf names do not merge. Nested component config paths must
remain exact from loader through projection audit.

## Source versus config

Configuration supplies pipeline composition and selected numeric/enum values.
Installed Diffusers source proves component construction, transformer/UNet/VAE
mechanisms, temporal behavior, factories, and the meaning of component fields.

## Unknown behavior

Geometry and activation conflicts must remain unknown rather than becoming
zero or a retried alias. Repeated `_inspect` calls may not weaken a previously
recorded ambiguity.

## Remaining seams

- `diffusor/config_facts.yaml` can still create card chips directly from config;
- conditioning and typing YAML still contain structural/presentation mappings;
- the diffusion parser still mixes evidence interpretation and view topology;
- VAE, scheduler, and temporal fact families are not all native typed facts
  with complete projection receipts.
