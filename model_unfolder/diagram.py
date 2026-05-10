"""Diagram — the renderable object.

Implements ``_repr_html_`` so it auto-renders inline in Jupyter (like
``matplotlib`` or a ``pandas`` DataFrame). Outside notebooks, call
``.save(path)`` to write a portable HTML file.
"""
from __future__ import annotations
import json
import os
import uuid
from .ir import ModelIR
from .html_renderer import render_document, render_fragment
from .params import estimate_params, humanize


class Diagram:
    """A renderable diagram of a transformer architecture."""

    def __init__(self, ir: ModelIR):
        self.ir = ir
        self._mount_id = f"uf-{uuid.uuid4().hex[:10]}"
        self._params = estimate_params(ir)
        self._ir_cache: dict | None = None
        self._html_cache: dict[bool, str] = {}

    def to_ir(self) -> dict:
        """Return the underlying IR (plus param estimates) as a plain dict."""
        if self._ir_cache is not None:
            return self._ir_cache

        d = self.ir.to_dict()
        p = self._params
        d["params"] = {
            "total": p["total"],
            "active": p["active"],
            "total_h": humanize(p["total"]),
            "active_h": humanize(p["active"]),
            "is_sparse": p["is_sparse"],
        }
        self._ir_cache = d
        return d

    def param_count(self) -> dict:
        """Return parameter-count estimates: total / active / per-layer breakdown."""
        return self._params

    @property
    def warnings(self) -> list[str]:
        """Adapter-emitted warnings — unknown model types, unrecognised layer types, etc."""
        return list(self.ir.warnings)

    def _repr_html_(self) -> str:
        """Jupyter calls this; returned HTML string is rendered inline."""
        return self._html(standalone=False)

    def to_html(self, standalone: bool = True) -> str:
        """Return the diagram as an HTML string.

        Parameters
        ----------
        standalone : bool
            If True (default), wraps the diagram in a full HTML document.
            If False, returns a fragment usable for embedding (Jupyter mode).
        """
        return self._html(standalone=standalone)

    def save(self, path: str) -> str:
        """Save the diagram to disk.

        - ``.html`` — interactive standalone document
        - ``.json`` — the underlying IR (no rendering)
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".html":
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_html(standalone=True))
        elif ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_ir(), f, indent=2)
        else:
            raise ValueError(
                f"Unsupported extension {ext!r}. Use .html or .json."
            )
        return path

    def _html(self, standalone: bool) -> str:
        if standalone not in self._html_cache:
            if standalone:
                self._html_cache[standalone] = render_document(self.to_ir(), self._mount_id)
            else:
                self._html_cache[standalone] = render_fragment(self.to_ir(), self._mount_id)
        return self._html_cache[standalone]

    def __repr__(self) -> str:
        s = (
            f"<Diagram {self.ir.name!r} · {self.ir.num_layers} layers · "
            f"~{humanize(self._params['total'])} params"
            + (f" ({humanize(self._params['active'])} active)" if self._params['is_sparse'] else "")
            + ">"
        )
        if self.ir.warnings:
            s += "\n" + "\n".join(f"  ⚠ {w}" for w in self.ir.warnings)
        return s
