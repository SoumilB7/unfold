"""Escaping and formatting helpers for renderer modules."""
from __future__ import annotations

from html import escape
from typing import Any


def _fmt_int(value: Any) -> str:
    if value is None:
        return "?"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _num(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _html(value: Any) -> str:
    return escape(str(value), quote=False)


def _attr(value: Any) -> str:
    return escape(str(value), quote=True)


def facts_html(facts) -> str:
    """The one chips row every inspect card uses for its numeric/spec facts."""
    if not facts:
        return ""
    chips = "".join(f'<span class="uf-fact">{_html(f)}</span>' for f in facts if f)
    return f'<div class="uf-card-facts">{chips}</div>' if chips else ""
