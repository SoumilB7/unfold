"""Tiny payload builders for multimodal extras.

Every modality path has the same envelope — a handful of named sections
(``input``/``embedding``/``encoder``/``projector``/``tokens``) plus a parallel
``pipeline`` of the same steps and a ``trace``.  Historically each path wrote
the section dict *and* the matching pipeline step by hand, so the two drifted
and every modality re-implemented the envelope.  ``Stage`` + ``assemble_path``
make a path one ordered list of stages; the section view and the pipeline view
are derived from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .accessors import drop_none


def multimodal_payload(inputs: dict[str, Any], fusion: dict[str, Any]) -> dict:
    """Wrap modality inputs and fusion into the extras payload shape."""
    return {"modalities": {"inputs": inputs, "fusion": fusion}}


def pipeline_step(step_id: str, operation: str, kind: str, **values: Any) -> dict:
    """Build one semantic pipeline step without display text."""
    return drop_none({"id": step_id, "operation": operation, "kind": kind, **values})


@dataclass
class Stage:
    """One stage of a modality path: a named section and its pipeline step.

    ``section_fields`` populate the section payload (e.g. ``encoder``);
    ``step_fields`` populate the pipeline step.  When ``step_fields`` is None
    the section fields are reused — most stages keep them in sync, the few that
    differ (an ``input`` section that carries sizes the pipeline step omits)
    pass an explicit ``step_fields``.
    """
    section: str
    step_id: str
    operation: str
    kind: str
    section_fields: dict = field(default_factory=dict)
    step_fields: dict | None = None


def assemble_path(path_kind: str, stages: list[Stage], trace_config_paths: list[str]) -> dict:
    """Build a modality path envelope from ordered stages.

    The section view and the ``pipeline`` view are both derived from ``stages``,
    so they can never drift, and every modality shares one envelope shape.
    """
    sections: dict[str, Any] = {}
    pipeline: list[dict] = []
    for stage in stages:
        sections[stage.section] = drop_none({"kind": stage.kind, **stage.section_fields})
        step_fields = stage.section_fields if stage.step_fields is None else stage.step_fields
        pipeline.append(pipeline_step(stage.step_id, stage.operation, stage.kind, **step_fields))
    return drop_none({
        "kind": path_kind,
        **sections,
        "pipeline": pipeline,
        "trace": {"config_paths": trace_config_paths},
    })

