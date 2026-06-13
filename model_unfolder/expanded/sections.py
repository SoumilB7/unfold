"""Top-level sections: model identity, dimensions, parameters, io."""
from __future__ import annotations

from typing import Any

from ..params import humanize
from .utils import drop_none, shape


def build_model(raw: dict, evidence: dict | None) -> dict[str, Any]:
    provenance = ((evidence or {}).get("provenance")) or {}
    return drop_none({
        "name":         raw.get("name"),
        "architecture": raw.get("architecture"),
        "model_id":     provenance.get("model_id"),
        "model_type":   provenance.get("model_type"),
    })


def _is_diffusion(raw: dict) -> bool:
    return ((raw.get("extras") or {}).get("render") or {}).get("family") == "diffusion"


def build_dimensions(raw: dict) -> dict[str, Any]:
    # A denoiser has no token vocabulary and no tied LM head — those IR fields
    # exist only to keep the param estimate honest and must NOT leak here as if
    # the model had word embeddings.  Report the DiT's latent geometry instead.
    if _is_diffusion(raw):
        diff = (raw.get("extras") or {}).get("diffusion") or {}
        return drop_none({
            "hidden_size": raw.get("hidden_size"),
            "in_channels": diff.get("in_channels"),
            "patch_size":  diff.get("patch_size"),
        })
    return drop_none({
        "vocab_size":          raw.get("vocab_size"),
        "hidden_size":         raw.get("hidden_size"),
        "context_length":      raw.get("max_position_embeddings"),
        "tie_word_embeddings": raw.get("tie_word_embeddings"),
    })


def build_parameters(params: dict) -> dict[str, Any]:
    return {
        "total":  {"value": params["total"],  "human": humanize(params["total"])},
        "active": {"value": params["active"], "human": humanize(params["active"])},
        "sparse": bool(params["is_sparse"]),
    }


def build_io(raw: dict) -> dict[str, Any]:
    """Tokens → embedding → stack → final norm → LM head (a denoiser's latent I/O
    is different — see :func:`_diffusion_io`)."""
    if _is_diffusion(raw):
        return _diffusion_io(raw)
    hidden = raw.get("hidden_size")
    vocab  = raw.get("vocab_size")
    fusion = (((raw.get("extras") or {}).get("modalities") or {}).get("fusion") or {})

    out: dict[str, Any] = {
        "input": {
            "kind":  "token_ids",
            "shape": ["batch", "sequence"],
            "trace": {"ir_path": "input"},
        },
        "token_embedding": drop_none({
            "operation":     "embedding_lookup",
            "vocab_size":    vocab,
            "embedding_dim": hidden,
            "weight_shape":  shape(vocab, hidden),
            "output_width":  hidden,
            "trace":         {"ir_path": "extras.render.model_blocks.embed"},
        }),
    }
    if fusion:
        out["stack_input"] = drop_none({
            "kind":          (fusion.get("output") or {}).get("kind"),
            "width":         (fusion.get("output") or {}).get("width") or hidden,
            "source":        "modalities.fusion",
            "trace":         {"ir_path": "extras.modalities.fusion"},
        })
    out["final_norm"] = drop_none({
        "operation":        "norm",
        "kind":             _final_norm_kind(raw),
        "normalized_shape": hidden,
        "trace":            {"ir_path": "extras.render.model_blocks.final_rms"},
    })
    out["lm_head"] = drop_none({
        "operation":               "linear",
        "in_features":             hidden,
        "out_features":            vocab,
        "weight_shape":            shape(vocab, hidden),
        "tied_to_token_embedding": bool(raw.get("tie_word_embeddings")),
        "trace":                   {"ir_path": "extras.render.model_blocks.lm_head"},
    })
    return out


def _diffusion_io(raw: dict) -> dict[str, Any]:
    """A denoiser's I/O is a latent, not tokens: a noisy latent is patchified to
    the hidden width, the stack runs, then it is unpatchified back to a
    noise/velocity prediction in latent space.  No vocabulary, no LM head — the
    same bookend nodes the LLM path traces, told honestly for a DiT."""
    hidden = raw.get("hidden_size")
    diff = (raw.get("extras") or {}).get("diffusion") or {}
    ch = diff.get("in_channels")
    return {
        "input": drop_none({
            "kind":     "noisy_latent",
            "channels": ch,
            "shape":    ["batch", ch or "channels", "height", "width"],
            "trace":    {"ir_path": "input"},
        }),
        "patchify": drop_none({
            "operation":    "linear",
            "out_features": hidden,
            "output_width": hidden,
            "trace":        {"ir_path": "extras.render.model_blocks.embed"},
        }),
        "final_norm": drop_none({
            "operation":        "norm",
            "kind":             _final_norm_kind(raw),
            "normalized_shape": hidden,
            "trace":            {"ir_path": "extras.render.model_blocks.final_rms"},
        }),
        "output": drop_none({
            "operation":   "linear",
            "kind":        "noise_prediction",
            "in_features": hidden,
            "trace":       {"ir_path": "extras.render.model_blocks.lm_head"},
        }),
    }


def _final_norm_kind(raw: dict) -> str | None:
    layers = raw.get("layers") or []
    return layers[-1].get("norm_kind") if layers else None
