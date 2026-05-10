"""Gemma family adapter — dispatches to version-specific sub-adapters.

Sub-adapters (in priority order):
  gemma4  — Gemma 4 (31B, E2B, E4B, 26B-A4B)   model_type: gemma4 / gemma4_text
  gemma3  — Gemma 3 / 3n (1B–27B, E2B, E4B)     model_type: gemma3 / gemma3n
"""
from __future__ import annotations

from typing import Any

from . import gemma3, gemma4

_SUB_ADAPTERS = [gemma4, gemma3]


def matches(cfg: Any) -> bool:
    return any(a.matches(cfg) for a in _SUB_ADAPTERS)


def parse(cfg: Any) -> Any:
    for a in _SUB_ADAPTERS:
        if a.matches(cfg):
            return a.parse(cfg)
    # Should not be reached since matches() already confirmed one hit
    return gemma3.parse(cfg)
