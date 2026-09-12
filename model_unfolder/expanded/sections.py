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


def _is_projected_diffusion(raw: dict) -> bool:
    """The source-projected denoiser path, identified by structure only."""
    extras = raw.get("extras") or {}
    return _is_diffusion(raw) and "unet" not in extras


def build_dimensions(raw: dict) -> dict[str, Any]:
    # A denoiser has no token vocabulary and no tied LM head — those IR fields
    # exist only to keep the param estimate honest and must NOT leak here as if
    # the model had word embeddings.  Report the DiT's latent geometry instead.
    if _is_diffusion(raw):
        extras = raw.get("extras") or {}
        if _is_projected_diffusion(raw):
            # ModelIR uses zero as the legacy parameter-estimator sentinel.  In
            # expanded architecture JSON it would falsely mean a proven
            # zero-width denoiser, so omit it unless F3 projected a real width.
            hidden = raw.get("hidden_size")
            if hidden == 0:
                hidden = None
            return drop_none({"hidden_size": hidden})
        diff = extras.get("diffusion") or {}
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
    out = {
        "total":  {"value": params["total"],  "human": humanize(params["total"])},
        "active": {"value": params["active"], "human": humanize(params["active"])},
        "sparse": bool(params["is_sparse"]),
    }
    # Parameter estimates may deliberately retain a deterministic convention
    # while a mechanism is unresolved.  The raw IR and HTML already disclose
    # those conventions; expanded JSON must carry the same qualification or it
    # would present the identical number as a stronger architectural claim.
    if params.get("assumptions"):
        out["assumptions"] = list(params["assumptions"])
    if params.get("incomplete"):
        out["incomplete"] = params["incomplete"]
    return out


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
    # Model-stage positional addition is projected from the canonical layer
    # specs plus the canonical model-block graph.  The retired raw
    # ``extras.position_encoding`` envelope was a second structural authority.
    model_blocks = tuple(
        block for block in
        (((raw.get("extras") or {}).get("render") or {})
         .get("model_blocks") or [])
        if isinstance(block, dict))
    model_block_ids = {block.get("id") for block in model_blocks}
    position_add = next(
        (block for block in model_blocks if block.get("id") == "position_add"),
        None)
    position_add_detail = (position_add or {}).get("detail") or {}
    if (position_add_detail.get("position_kind")
            in {"learned_absolute", "fixed_absolute"}
            and position_add_detail.get("position_application")
            == "embedding_add"
            and {"position_ids", "position_embed", "position_add"}
            <= model_block_ids):
        out["position_ids"] = {
            "kind": "position_ids",
            "shape": ["batch", "sequence"],
            "trace": {"ir_path": "extras.render.model_blocks.position_ids"},
        }
        out["position_embedding"] = drop_none({
            "operation": "embedding_lookup",
            "embedding_dim": hidden,
            "output_width": hidden,
            "trace": {"ir_path": "extras.render.model_blocks.position_embed"},
        })
        out["position_add"] = {
            "operation": "elementwise_add",
            "inputs": ["token_embedding", "position_embedding"],
            "trace": {"ir_path": "extras.render.model_blocks.position_add"},
        }
    if fusion:
        fusion_output = fusion.get("output") or {}
        fusion_resolved = fusion.get("kind") != "code_defined_fusion"
        out["stack_input"] = drop_none({
            "kind":          fusion_output.get("kind"),
            # The decoder width is a lawful operand only after wrapper source
            # proves a concrete fusion mechanism.  An opaque declared lane
            # cannot manufacture a hidden-width stack-input fact.
            "width":         fusion_output.get("width") or (
                hidden if fusion_resolved else None),
            "source":        "modalities.fusion",
            "trace":         {"ir_path": "extras.modalities.fusion"},
        })
    final_kind = _final_norm_kind(raw)
    if final_kind:
        out["final_norm"] = {
            "operation": "norm",
            "kind": final_kind,
            "normalized_shape": hidden,
            "trace": {"ir_path": "final_norm_kind"},
        }
    else:
        out["final_stage"] = {
            "status": "unresolved",
            "trace": {"ir_path": "final_norm_kind"},
        }
    lm_head = drop_none({
        "operation":               "linear",
        "in_features":             hidden,
        "out_features":            vocab,
        "weight_shape":            shape(vocab, hidden),
        # U2 tri-state: a bool() here would fabricate an "untied" claim.
        "tied_to_token_embedding": (bool(raw["tie_word_embeddings"])
                                    if raw.get("tie_word_embeddings") is not None
                                    else None),
        "trace":                   {"ir_path": "extras.render.model_blocks.lm_head"},
    })
    # ``drop_none`` intentionally removes ordinary absent fields, but this one
    # is a typed architectural unknown and must remain explicit in machine JSON.
    if raw.get("tie_word_embeddings") is None:
        lm_head["tied_to_token_embedding"] = None
    out["lm_head"] = lm_head
    return out


def _diffusion_io(raw: dict) -> dict[str, Any]:
    """A denoiser's I/O is a latent, not tokens: a noisy latent is patchified to
    the hidden width, the stack runs, then it is unpatchified back to a
    noise/velocity prediction in latent space.  No vocabulary, no LM head — the
    same bookend nodes the LLM path traces, told honestly for a DiT."""
    if _is_projected_diffusion(raw):
        return _projected_diffusion_io(raw)

    # U11 compatibility handoff: the legacy UNet surface is intentionally not
    # dismantled by U10.  Its replacement is owned by the later U11 unit.
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
        "final_stage": (
            {
                "operation": "norm",
                "kind": _final_norm_kind(raw),
                "normalized_shape": hidden,
                "trace": {"ir_path": "final_norm_kind"},
            }
            if _final_norm_kind(raw) else
            {
                "status": "unresolved",
                "trace": {"ir_path": "final_norm_kind"},
            }
        ),
        "output": drop_none({
            "operation":   "linear",
            "kind":        "noise_prediction",
            "in_features": hidden,
            "trace":       {"ir_path": "extras.render.model_blocks.lm_head"},
        }),
    }


def _projected_diffusion_io(raw: dict) -> dict[str, Any]:
    """U10-F3 machine I/O from the same adapter-authored boundary DTO.

    No label is interpreted here.  The structured ``detail.operations`` list
    is copied only when its owning block is explicitly resolved; otherwise the
    boundary remains an explicit unknown.  Output media domain is deliberately
    absent because U12, not the denoiser source, owns that fact.
    """
    render = ((raw.get("extras") or {}).get("render") or {})
    blocks = {
        block.get("id"): block
        for block in render.get("model_blocks") or ()
        if isinstance(block, dict) and block.get("id")
    }

    def transform(node_id: str, trace: str) -> dict[str, Any]:
        block = blocks.get(node_id) or {}
        operations = tuple((block.get("detail") or {}).get("operations") or ())
        if block.get("resolved") is True and operations:
            return {
                "operations": list(operations),
                "trace": {"ir_path": trace},
            }
        return {
            "status": "unresolved",
            "trace": {"ir_path": trace},
        }

    return {
        "input": {
            "kind": "denoiser_state",
            "trace": {"ir_path": "extras.render.model_blocks.tok_text"},
        },
        "input_transform": transform(
            "embed", "extras.render.model_blocks.embed"),
        "output_transform": transform(
            "final_rms", "extras.render.model_blocks.final_rms"),
        "output": {
            "kind": "denoiser_state",
            "domain": None,
            "trace": {"ir_path": "extras.render.model_blocks.lm_head"},
        },
    }


def _final_norm_kind(raw: dict) -> str | None:
    value = raw.get("final_norm_kind")
    return value if value not in {None, "", "unknown"} else None
