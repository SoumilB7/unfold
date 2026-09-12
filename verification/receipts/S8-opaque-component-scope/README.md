# Opaque UNet fallback component scope — RETURN

The corrected successful UNet projection filters absent outer components. Its failed-construction/source path still calls `_parse_projected_denoiser` with a failed result, which uses `diffusion_opaque_render_spec`. That producer does not apply the supplied component-presence evidence.

The bounded replay supplies all three component-presence flags as false and no encoder, scheduler or VAE document. It produces `noise`, `timestep`, `text_encoder`, `latent`, `denoiser`, `scheduler`, `vae_decode` and `image` cards, and an expanded sampling loop. The actual overview SVG contains scheduler, VAE and noise labels. `component_scope` is absent. This is a direct producer/render check, not a full model or missing-source campaign.

Returned to executor: apply the same component-scope rule to the UNet failure result while preserving its typed denoiser limitation. A failure to establish the denoiser must not manufacture its surrounding pipeline. This closes the same defect found by the extra models; it requires no new mechanism reader.

Exact source bytes, hashes, result and SVG are retained. Sources remained identical during the check. Run from `unfold-pkg` with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 verification/receipts/S8-opaque-component-scope/replay.py`. The replay rejects changed source; save any corrected result separately. No production edits, model construction, pytest or blessing.
