"""Vision and video modality path extraction."""
from __future__ import annotations

from typing import Any

from .....evidence import config_access as _config_access
from .accessors import as_int
from .schema import Stage, assemble_path


def apply_projector_evidence(payload: dict | None, evidence, cfg: Any = None,
                             owner_namespace: str = "root") -> dict | None:
    """Project destination-qualified connector records into their exact paths.

    The card, op drill, path label, and fact-conformance net all read this same
    object.  Config supplies a dimension operand only after source proves the
    exact constructor boundary that consumes it; an unbound number never
    becomes projector architecture.  Latent/input geometry remains separate
    and never fabricates callable order when source evidence is absent.

    COR-4 (§9): the projector's ``out_features`` is written HERE and only here,
    from the evidence's source-bound width — a config path proven at the
    construction site (read back through the evented accessor so the numeric
    premise stays a logged config read) or a literal from the source.  Unproven
    or unbindable widths stay absent: no language-width or family fallback.
    """
    if not payload:
        return payload
    modalities = (payload.get("modalities") or {}).get("inputs") or {}
    # The geometry shell creates a connector node so the path remains
    # navigable, but its kind/ops/width are not architectural evidence.  Clear
    # those claims for every lane before applying destination-owned records;
    # otherwise a source-missing lane silently keeps the legacy config answer.
    for name, path in modalities.items():
        if isinstance(path, dict):
            _clear_projector_claim(path)
    if evidence is None:
        return payload
    if getattr(evidence, "status", "") in {"failed", "ambiguous"}:
        detail = "; ".join(
            getattr(failure, "detail", "")
            for failure in getattr(evidence, "failures", ())
            if getattr(failure, "detail", ""))
        for path in modalities.values():
            if not isinstance(path, dict):
                continue
            # A failure may qualify an already-declared side block; it must
            # never manufacture a projector merely so there is somewhere to
            # display the finding.  When no projector block exists, the
            # parser's model-level warning remains the visible receipt.
            projector = path.get("projector")
            if not isinstance(projector, dict):
                continue
            projector["kind"] = "code_defined_projector"
            projector["evidence_unresolved"] = True
            projector["evidence_failure"] = detail or getattr(
                evidence, "status", "unresolved")
        return payload
    from .....evidence.reader_result import ReaderResult
    reader_result = evidence if isinstance(evidence, ReaderResult) else None
    records = _projector_records(evidence)
    for record in records:
        destinations = tuple(getattr(record, "modalities", ()) or ())
        # Compatibility-only legacy evidence predates destination ownership.
        # Production U9 results always carry destinations.
        if not destinations:
            destinations = ("vision", "video")
        evidence_dict = record.to_dict()
        for name in destinations:
            path = modalities.get(name)
            if not isinstance(path, dict):
                continue
            _apply_one_projector(
                path, name, record, evidence_dict, cfg, owner_namespace, reader_result=reader_result)
    return payload


def _clear_projector_claim(path: dict) -> None:
    projector = path.get("projector")
    if isinstance(projector, dict):
        for key in (
                "activation", "kind", "learned_queries", "ops",
                "in_features", "out_features", "profile", "source_class",
                "source_component", "source_evidence", "source_field",
                "source_owner"):
            projector.pop(key, None)
        projector["kind"] = "code_defined_projector"
    for step in path.get("pipeline") or ():
        if step.get("id") not in {
                "projector", "video_projector", "audio_projector",
                "conditioning_projector"}:
            continue
        for key in (
                "activation", "learned_queries", "ops",
                "in_features", "out_features"):
            step.pop(key, None)
        step["kind"] = "code_defined_projector"


def _projector_records(evidence):
    if getattr(evidence, "status", "") in {"resolved", "incomplete"}:
        inventory = getattr(evidence, "value", None)
        return tuple(getattr(inventory, "projectors", ()) or ())
    inventory = getattr(evidence, "projectors", None)
    if inventory is not None:
        return tuple(inventory)
    return (evidence,) if hasattr(evidence, "to_dict") else ()


def _apply_one_projector(path, name, evidence, evidence_dict, cfg,
                         owner_namespace, *, reader_result=None):
    projector = path.get("projector") or {}
    projector.pop("profile", None)
    bound_in, _in_fact_status = _bound_projector_width(
        evidence, cfg, owner=f"{owner_namespace}.{name}", lane="in", reader_result=reader_result)
    bound_out, out_fact_status = _bound_out_width(
        evidence, cfg, owner=f"{owner_namespace}.{name}", reader_result=reader_result)
    if bound_in is not None:
        projector["in_features"] = bound_in
    if bound_out is not None:
        _insert_after(projector, "in_features", "out_features", bound_out)
        tokens = path.get("tokens")
        if isinstance(tokens, dict):
            tokens["width"] = bound_out
    projector["source_evidence"] = evidence_dict
    projector["source_owner"] = evidence.owner_class
    projector["source_component"] = evidence.component
    projector["source_class"] = evidence.projector_class
    projector["source_field"] = evidence.field_name
    if evidence.status == "proven":
        projector["kind"] = evidence.kind
        projector["ops"] = [op.to_dict() for op in evidence.ops]
        projector["learned_queries"] = evidence.learned_queries
    else:
        projector["kind"] = "code_defined_projector"
        projector.pop("ops", None)
        projector.pop("learned_queries", None)
    for step in path.get("pipeline") or []:
        if step.get("id") not in {
                "projector", "video_projector", "audio_projector",
                "conditioning_projector"}:
            continue
        if bound_in is not None:
            step["in_features"] = bound_in
        if bound_out is not None:
            _insert_after(step, "in_features", "out_features", bound_out)
        step["kind"] = projector["kind"]
        if evidence.status == "proven":
            step["ops"] = projector["ops"]
        else:
            step.pop("ops", None)
    if bound_out is not None:
        pipeline = path.get("pipeline") or ()
        if pipeline:
            pipeline[-1]["width"] = bound_out


def _bound_out_width(evidence, cfg: Any, *, owner: str, reader_result=None) \
        -> tuple[int | None, str | None]:
    """Resolve source-bound output width and fact status, or two unknowns.

    ``code_bound`` carries its literal; ``config_bound`` names an exact dotted
    path from the ROOT config, which is read through the evented accessor under
    that exact container so ownership and the logged read agree.  ``derived``
    and ``unavailable`` resolve nothing — the drawing stays honestly unknown.
    """
    if evidence is None or getattr(evidence, "status", "") != "proven":
        return None, None
    source = getattr(evidence, "out_width_source", "unavailable")
    if source == "code_bound":
        # a literal in the projector source — code alone proves it.  R5-vet:
        # a pure-code width still needs a TYPED FACT (and hence a receipt);
        # it merely has no config obligation.
        width = as_int(getattr(evidence, "out_width_value", None))
        if width is not None:
            _record_projector_fact(owner, width, "code_proven", None, evidence, reader_result=reader_result)
        return width, "code_proven"
    if source != "config_bound" or cfg is None:
        return None, None
    parts = tuple(getattr(evidence, "out_width_path", ()) or ())
    if not parts:
        return None, None
    from .....evidence.reader_claims import resolve_reader_operand
    resolution = resolve_reader_operand(cfg, parts, component=owner)
    if resolution is None or resolution.ambiguous or not (
            resolution.present or resolution.source_kind == "class_default"):
        return None, None
    # U2-R5: ONE author for the evidence status.  The projector SOURCE proves it
    # consumes this exact config value, so the status is ``code_and_config`` —
    # never ``config_declared`` merely because a number exists.  The SAME status
    # goes to the consumption (whose fingerprint Net 2 checks), to the typed
    # FACT (where the validator's expected hash originates), and — via the
    # returned pair — to the drawn card's descriptor, so no later surface can
    # re-derive it differently.
    fact_status = ("class_default" if resolution.source_kind == "class_default"
                   or resolution.provenance == "class_default" else "code_and_config")
    width = as_int(resolution.consume(
        fact_owner=owner, fact_key="projector_out_features",
        mechanism="projector_out_width", status=fact_status))
    if width is not None:
        _record_projector_fact(owner, width, fact_status, None if fact_status == "class_default" else ".".join(parts),
                               evidence, reader_result=reader_result)
    return width, fact_status


def _bound_projector_width(evidence, cfg: Any, *, owner: str, lane: str, reader_result=None):
    """Resolve one source-proven projector boundary width."""
    if lane not in {"in", "out"}:
        raise ValueError("projector width lane is in|out")
    if lane == "out":
        return _bound_out_width(evidence, cfg, owner=owner, reader_result=reader_result)
    if evidence is None or getattr(evidence, "status", "") != "proven":
        return None, None
    source = getattr(evidence, "in_width_source", "unavailable")
    if source == "code_bound":
        width = as_int(getattr(evidence, "in_width_value", None))
        if width is not None:
            _record_projector_fact(
                owner, width, "code_proven", None, evidence,
                fact_key="projector_in_features", reader_result=reader_result)
        return width, "code_proven"
    if source != "config_bound" or cfg is None:
        return None, None
    parts = tuple(getattr(evidence, "in_width_path", ()) or ())
    if not parts:
        return None, None
    from .....evidence.reader_claims import resolve_reader_operand
    resolution = resolve_reader_operand(cfg, parts, component=owner)
    if resolution is None or resolution.ambiguous or not (
            resolution.present or resolution.source_kind == "class_default"):
        return None, None
    status = ("class_default" if resolution.source_kind == "class_default"
              or resolution.provenance == "class_default" else "code_and_config")
    width = as_int(resolution.consume(
        fact_owner=owner, fact_key="projector_in_features",
        mechanism="projector_in_width", status=status))
    if width is not None:
        _record_projector_fact(
            owner, width, status, None if status == "class_default" else ".".join(parts), evidence,
            fact_key="projector_in_features", reader_result=reader_result)
    return width, status


def _record_projector_fact(owner: str, width: int, status: str,
                           config_path: "str | None", evidence, *,
                           fact_key: str = "projector_out_features", reader_result=None) -> None:
    """Record the typed projector-width fact — the ONE author for both tiers.

    R5-vet: ``code_and_config`` must substantiate BOTH halves — the exact config
    path AND the exact projector source span (file/line/class from the
    construction-site evidence); a config path alone does not substantiate the
    "code" half.  ``code_proven`` cites the span alone.

    U2-R5 pilot scope: recorded ONLY for the owners the registry migrates
    (root.vision / root.video).  An EMBEDDED tower's owner
    (root.text_encoder.vision) is deliberately unmigrated: its consumption
    obligation stays on the advisory census as exact R6 debt, and attempting
    the typed write would (rightly) be rejected by the closed-world registry —
    a rejection that, swallowed by the modality try/except, once silently
    dropped the WHOLE projector-evidence application for embedded VL encoders.
    The registry decides, deterministically."""
    from .....evidence.context import active_facts
    from .....evidence.facts import EvidenceFact, SourceSpan
    from .....evidence.registry import REGISTRY
    facts = active_facts()
    definition = REGISTRY.get(fact_key)
    if facts is None or definition is None \
            or owner not in definition.owner_patterns:
        return
    span = SourceSpan(
        component=str(getattr(evidence, "component", "") or ""),
        class_name=getattr(evidence, "projector_class", None),
        file=getattr(evidence, "source_file", None),
        line=getattr(evidence, "line", None))
    source_label = (f"{span.file}:{span.line}" if span.file else
                    str(span.class_name or "projector source"))
    fact = EvidenceFact(
        key=fact_key, owner=owner, value=width,
        status=status, completeness="complete",
        config_paths=(config_path,) if config_path else (),
        source_spans=(span,),
        legacy_source=(f"{config_path} + {source_label}" if config_path
                       else source_label),
        reason=("projector input width" if fact_key == "projector_in_features"
                else "projector output width")
               + " bound at the construction site the source proves")
    if reader_result is not None and reader_result.claim_witness is not None:
        from .....evidence.reader_claims import qualify_reader_fact
        if not any(item is evidence for item in reader_result.value.projectors):
            raise ValueError("projector leaf is not a member of its actual reader result")
        document = _config_access.current_prepared_document.get()
        if document is not None:
            fact = qualify_reader_fact(fact, reader_result, reader_result.claim_witness.index, document)
    facts.record_typed(fact)


def _insert_after(target: dict, anchor: str, key: str, value: Any) -> None:
    """Insert ``key`` directly after ``anchor`` preserving all other order —
    the overlay must not reorder card fields the build already wrote."""
    if anchor not in target:
        target[key] = value
        return
    items = list(target.items())
    target.clear()
    for existing_key, existing_value in items:
        target[existing_key] = existing_value
        if existing_key == anchor:
            target[key] = value




def vision_path(_cfg: Any, _text_cfg: Any, _vision_cfg: Any,
                _text_hidden_size: int) -> dict:
    """Build an opaque, address-only lane for a declared visual component.

    The checkpoint may establish that a vision component exists.  It cannot
    establish patching, encoder, connector, token-route, geometry, or position
    mechanisms.  Those fields are projected later from exact U9 source facts;
    this shell exists only so those facts have stable structural destinations.
    """
    shape = ["batch", "images", "channels", "height", "width"]
    return assemble_path(
        "code_defined_modality_path",
        [
            Stage("input", "image_pixels", "input", "image_pixels",
                  {"shape": shape}),
            Stage("embedding", "patch_embedding", "unknown",
                  "code_defined_embedding", {}),
            Stage("encoder", "vision_encoder", "unknown",
                  "code_defined_encoder", {}),
            Stage("projector", "projector", "unknown",
                  "code_defined_projector", {}),
            Stage("tokens", "vision_tokens", "unknown",
                  "code_defined_tokens", {}),
        ],
        [],
    )


def video_path(_cfg: Any, _vision_cfg: Any, _text_hidden_size: int) -> dict:
    """Build an opaque, address-only video lane over a declared component."""
    shape = ["batch", "videos", "frames", "channels", "height", "width"]
    return assemble_path(
        "code_defined_modality_path",
        [
            Stage("input", "video_frames", "input", "video_frames",
                  {"shape": shape}),
            Stage("embedding", "video_patch_embedding", "unknown",
                  "code_defined_embedding", {}),
            Stage("encoder", "video_encoder", "unknown",
                  "code_defined_encoder", {}),
            Stage("projector", "video_projector", "unknown",
                  "code_defined_projector", {}),
            Stage("tokens", "video_tokens", "unknown",
                  "code_defined_tokens", {}),
        ],
        [],
    )


__all__ = [
    "video_path",
    "vision_path",
]
