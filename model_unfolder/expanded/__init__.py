"""Structured JSON view of an unfolded model.

This package emits **data, not prose** — every field is a number, name,
shape, ref, or graph.  The visual renderer keeps labels/descriptions;
this view keeps the bones an LLM (or another tool) can re-trace back
into the parsed IR and the static code-evidence scan.

Module split (one concern per file):

* :mod:`grouping`       — bucket layers by structural signature
* :mod:`sections`       — top-level model / dimensions / parameters / io
* :mod:`stack`          — the decoder stack as a for-loop
* :mod:`attention`      — attention spec + operation graph
* :mod:`ffn`            — FFN spec + operation graph
* :mod:`norms`          — norm spec
* :mod:`residual`       — residual topology
* :mod:`block_graph`    — generic block-level DAG (one node per IR block)
* :mod:`layer_group`    — assemble all of the above for one group
* :mod:`pathways`       — external pathways (PLE etc.)
* :mod:`code_evidence`  — normalise the optional code-evidence section
* :mod:`utils`          — drop-none, shape, index-range helpers

Single entry point: :func:`build_expanded`.
"""
from __future__ import annotations

from typing import Any

from ..ir import ModelIR
from ..params import estimate_params

from .grouping import group_layers
from .sections import build_model, build_dimensions, build_parameters, build_io
from .stack import build_stack
from .layer_group import build_layer_group
from .pathways import build_external_pathways
from .code_evidence import normalise_code_evidence


SCHEMA_VERSION = "3.0"


def build_expanded(ir: ModelIR, params: dict | None = None) -> dict:
    """Return a structured architecture JSON object."""
    if params is None:
        params = estimate_params(ir)

    raw      = ir.to_dict()
    layers   = raw.get("layers") or []
    extras   = raw.get("extras") or {}
    groups   = group_layers(layers)
    evidence = normalise_code_evidence(extras.get("code_evidence"))

    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "format":         "model_unfolder.expanded",
        "model":          build_model(raw, evidence),
        "dimensions":     build_dimensions(raw),
        "parameters":     build_parameters(params),
        "io":             build_io(raw),
        "stack":          build_stack(layers, groups),
        "layer_groups":   [build_layer_group(g, raw, evidence) for g in groups],
    }

    external = list(build_external_pathways(extras))
    if external:
        out["external_pathways"] = external

    edges = raw.get("cross_layer_edges") or []
    if edges:
        out["cross_layer_edges"] = edges

    if evidence:
        out["code_evidence"] = evidence

    warnings = raw.get("warnings") or []
    if warnings:
        out["warnings"] = warnings

    return out


__all__ = ["build_expanded", "SCHEMA_VERSION"]
