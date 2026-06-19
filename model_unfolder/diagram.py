"""Diagram — the renderable object.

Implements ``_repr_html_`` so it auto-renders inline in Jupyter (like
``matplotlib`` or a ``pandas`` DataFrame). Outside notebooks, call
``.save(path)`` to write a portable HTML file.
"""
from __future__ import annotations
import json
import os
import uuid
from .expanded import build_expanded
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
        self._json_cache: dict | None = None

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

    def to_json(self) -> dict:
        """Return the traceable expanded architecture schema as a dict.

        Unlike :meth:`to_ir`, this view avoids renderer labels/descriptions and
        emits stable JSON objects for dimensions, layer groups, projections,
        operation graphs, cache behavior, and trace paths into the parsed IR or
        static code-evidence scan.
        """
        if self._json_cache is None:
            self._json_cache = build_expanded(self.ir, self._params)
        return self._json_cache

    def to_json_string(self, indent: int = 2) -> str:
        """Return the expanded JSON as a formatted string."""
        return json.dumps(self.to_json(), indent=indent, default=str)

    def param_count(self) -> dict:
        """Return parameter-count estimates: total / active / per-layer breakdown."""
        return self._params

    @property
    def warnings(self) -> list[str]:
        """Adapter-emitted warnings — unknown model types, unrecognised layer types, etc."""
        return list(self.ir.warnings)

    def wiring_problems(self) -> list[str]:
        """Dable's dangling-connector flag — first-class, like click-coupling.

        Re-renders every baked graph and returns one message per connector
        (⊕ / × / ⊙) drawn with a missing input (empty list = clean). Treat a
        non-empty result as a build-blocking bug, not a warning."""
        from .renderers.html.graph_engine import reset_wiring_log, drain_wiring_log
        reset_wiring_log()
        self._html_cache.pop(True, None)         # force a fresh render so the detector runs
        self.to_html(standalone=True)
        return drain_wiring_log()

    def to_png(self, path: str, *, scale: float = 2.0, background: str = "white") -> str:
        """Render the top architecture view to a PNG image (needs ``rsvg-convert``)."""
        from .preview import architecture_svg, svg_to_png
        return svg_to_png(architecture_svg(self.to_html(standalone=True)), path,
                          scale=scale, background=background)

    def save_images(self, outdir: str, *, scale: float = 2.0, background: str = "white") -> list[str]:
        """Render the architecture view AND every drill view to PNGs in *outdir*.

        Pixels are the only oracle that catches a dangling connector or an
        unclickable block, so this is the norm for verifying output — one image
        per baked view, named by the block/view it belongs to."""
        from .preview import render_images
        return render_images(self, outdir, scale=scale, background=background)

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
        - ``.json`` — expanded architecture schema (no rendering)
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".html":
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_html(standalone=True))
        elif ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_json_string(indent=2))
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
