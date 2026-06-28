"""One immutable source-resolution context shared by one model parse.

Architectural parsing used to resolve the same Transformers/Diffusers source in
each detector independently, and Sable resolved it again for every conformance
net.  Besides wasted work, that made a name-blind parse impossible: scrubbing
``model_type`` also removed the address needed to rediscover source.

``ParseContext`` separates those phases.  Identity may be used once to locate
the source bundle; every architectural detector then consumes that already
resolved bundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import SourceBundle
from .sources import resolve_source_files


@dataclass
class ParseContext:
    """Call-local evidence state for one parse/conformance run."""

    source_bundle: SourceBundle
    source: str = "local"
    # Later evidence units cache component-qualified AST registries here.  The
    # cache is call-local: no model or concurrent render can contaminate it.
    registries: dict[tuple[str, tuple[str, ...]], dict] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        target: Any,
        *,
        source: str = "local",
        token: Any = None,
    ) -> "ParseContext":
        return cls(
            source_bundle=resolve_source_files(target, source=source, token=token),
            source=source,
        )


__all__ = ["ParseContext"]
