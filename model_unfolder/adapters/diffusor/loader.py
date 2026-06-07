"""Load a diffusion pipeline's denoiser config by HuggingFace model ID.

Diffusers repos don't carry a transformers-style root ``config.json`` with a
``model_type``, so ``AutoConfig`` can't read them (it raises "Unrecognized model
... should have a model_type key").  Instead the pipeline is described by
``model_index.json`` at the repo root, and each component (transformer,
text_encoder, vae, ...) lives in its own subfolder with its own ``config.json``.

This downloads ``model_index.json`` + the denoiser component's ``config.json``
and merges the component class names back in, so the diffusor parser can both
detect the DiT backbone and name the surrounding pipeline (text encoders, VAE).
Only files are downloaded — no model code is executed.
"""
from __future__ import annotations

import json
from typing import Any

#: Pipeline component keys that may hold the denoiser, in preference order.
#: UNet (``"unet"``) is intentionally absent — UNet diffusion isn't supported yet.
_DENOISER_KEYS = ("transformer",)


def load_diffusion_config_by_id(model_id: str, token: Any = None) -> dict | None:
    """Return a merged denoiser config dict, or ``None`` if not a supported
    diffusion repo (no ``model_index.json``, or no transformer denoiser)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        # Can't tell whether this is a diffusion repo; let the caller fall back
        # (its raw-config path raises the proper missing-dependency error).
        return None

    index = _download_json(hf_hub_download, model_id, "model_index.json", token)
    if not isinstance(index, dict):
        return None

    denoiser_key = next(
        (k for k in _DENOISER_KEYS if isinstance(index.get(k), (list, tuple))),
        None,
    )
    if denoiser_key is None:
        return None  # e.g. a UNet pipeline — not supported here

    cfg = _download_json(
        hf_hub_download, model_id, "config.json", token, subfolder=denoiser_key
    )
    if not isinstance(cfg, dict):
        return None

    # Merge the pipeline component wiring (text_encoder, vae, scheduler, ...) into
    # the denoiser config so the parser can name the surrounding stages — without
    # clobbering the denoiser's own ``_class_name`` (used for detection + arch
    # name).  Underscore keys (``_class_name``, ``_diffusers_version``) are the
    # pipeline's, not the denoiser's, so keep them out.
    for key, val in index.items():
        if key.startswith("_"):
            continue
        cfg.setdefault(key, val)

    # Pull the scheduler's own config.json so the loop can show real values
    # (num_train_timesteps, shift) rather than a placeholder. Best-effort.
    if isinstance(index.get("scheduler"), (list, tuple)):
        sched = _download_json(
            hf_hub_download, model_id, "scheduler_config.json", token, subfolder="scheduler"
        ) or _download_json(
            hf_hub_download, model_id, "config.json", token, subfolder="scheduler"
        )
        if isinstance(sched, dict):
            cfg.setdefault("_scheduler_config", sched)

    # Pull the VAE's config so the VAE-decoder view shows real channels/stages.
    if isinstance(index.get("vae"), (list, tuple)):
        vae = _download_json(hf_hub_download, model_id, "config.json", token, subfolder="vae")
        if isinstance(vae, dict):
            cfg.setdefault("_vae_config", vae)

    # Pull each text encoder's own config so the encoder view can show real
    # depth/width/heads instead of a schematic "× N layers".  Best-effort.
    enc_cfgs: dict[str, Any] = {}
    for key in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        if not isinstance(index.get(key), (list, tuple)):
            continue
        ec = _download_json(hf_hub_download, model_id, "config.json", token, subfolder=key)
        if isinstance(ec, dict):
            enc_cfgs[key] = ec
    if enc_cfgs:
        cfg.setdefault("_text_encoder_configs", enc_cfgs)

    cfg.setdefault("_pipeline_class_name", index.get("_class_name"))
    cfg.setdefault("_name_or_path", model_id)
    # The repo id IS the model tag the user typed — used for the display name
    # (the component's own _name_or_path is just ".../transformer").
    cfg["_repo_id"] = model_id
    return cfg


def _download_json(hf_hub_download, model_id, filename, token, subfolder=None):
    kwargs: dict = {"repo_id": model_id, "filename": filename}
    if subfolder:
        kwargs["subfolder"] = subfolder
    if token is not None:
        kwargs["token"] = token
    try:
        path = hf_hub_download(**kwargs)
    except Exception:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
