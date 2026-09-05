"""Structural input-format normalization.

This module translates foreign configuration syntax into the canonical field
spellings consumed by adapters.  A format may be recognized only from its file
layout and required keys.  Model, class, and repository names are forbidden as
format selectors, and normalization may not assert mechanism semantics.
"""
from __future__ import annotations

from typing import Any

from .errors import ConfigParseError
from .everchanging import load_input_format_aliases


_LAYERED_PARAMS_FORMAT = "layered_transformer_params"
_MISSING = object()


def _selected_value(source: dict, canonical: str, spellings: list[str]):
    """Resolve one format-local alias without first-hit ambiguity.

    The format vocabulary is scoped, so these aliases do not compete with
    fields in any other component.  Multiple equal declarations are harmless;
    unequal declarations are malformed input and must not be silently ordered.
    """
    names = list(dict.fromkeys([canonical, *spellings]))
    present = [(name, source[name]) for name in names if name in source]
    if not present:
        return _MISSING
    first_name, first_value = present[0]
    for other_name, other_value in present[1:]:
        try:
            equal = first_value == other_value
            if not isinstance(equal, bool):
                equal = bool(equal)
        except (TypeError, ValueError) as exc:
            raise ConfigParseError(
                f"Cannot compare input-format aliases {first_name!r} and "
                f"{other_name!r} for {canonical!r}: {exc}"
            ) from exc
        if not equal:
            raise ConfigParseError(
                f"Conflicting input-format aliases for {canonical!r}: "
                f"{first_name}={first_value!r}, {other_name}={other_value!r}"
            )
    return first_value


def _normalize_scope(source: dict, rules: dict[str, list[str]], scope: str) -> dict:
    prefix = f"{scope}."
    normalized: dict[str, Any] = {}
    for key, spellings in rules.items():
        if not key.startswith(prefix):
            continue
        canonical = key[len(prefix):]
        if canonical == "container":
            continue
        value = _selected_value(source, canonical, spellings)
        if value is not _MISSING:
            normalized[canonical] = value
    return normalized


def normalize_params_json(params: Any) -> dict | None:
    """Normalize the supported layered-transformer ``params.json`` dialect.

    Recognition is structural: the document must contain every declared
    required key.  The result contains checkpoint geometry/declarations only;
    it deliberately supplies no ``model_type`` or architecture identity.
    """
    if not isinstance(params, dict):
        return None
    rules = load_input_format_aliases(_LAYERED_PARAMS_FORMAT)
    required = rules.get("required") or []
    if not required or any(name not in params for name in required):
        return None

    normalized = _normalize_scope(params, rules, "text")
    container_names = rules.get("vision.container") or []
    containers = [(name, params[name]) for name in container_names if name in params]
    if len(containers) > 1:
        first_name, first_value = containers[0]
        for other_name, other_value in containers[1:]:
            if first_value != other_value:
                raise ConfigParseError(
                    "Conflicting vision containers in params.json: "
                    f"{first_name!r} and {other_name!r}"
                )
    if containers:
        container_name, vision = containers[0]
        if not isinstance(vision, dict):
            raise ConfigParseError(
                f"Input-format container {container_name!r} must be an object"
            )
        normalized["vision_config"] = _normalize_scope(vision, rules, "vision")
    return normalized


__all__ = ["normalize_params_json"]
