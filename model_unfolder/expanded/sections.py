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


def build_dimensions(raw: dict) -> dict[str, Any]:
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
    """Tokens → embedding → stack → final norm → LM head."""
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


def _final_norm_kind(raw: dict) -> str | None:
    layers = raw.get("layers") or []
    return layers[-1].get("norm_kind") if layers else None
