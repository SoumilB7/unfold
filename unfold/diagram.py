"""TransformerDiagram — the renderable object.

Implements ``_repr_html_`` so it auto-renders inline in Jupyter (like
``matplotlib`` or a ``pandas`` DataFrame). Outside notebooks, call
``.save(path)`` to write a portable HTML file.
"""
from __future__ import annotations
import json
import os
import uuid
from importlib import resources
from .ir import ModelIR


def _load_renderer_js() -> str:
    pkg = "transformer_viz.static"
    return resources.files(pkg).joinpath("renderer.js").read_text(encoding="utf-8")


class TransformerDiagram:
    """A renderable diagram of a transformer architecture."""

    def __init__(self, ir: ModelIR):
        self.ir = ir
        self._mount_id = f"tv-{uuid.uuid4().hex[:10]}"

    def to_ir(self) -> dict:
        """Return the underlying IR as a plain dict."""
        return self.ir.to_dict()

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
        ir_json = json.dumps(self.to_ir())
        renderer_js = _load_renderer_js()
        mount_id = self._mount_id

        body = f"""
<div id="{mount_id}" style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#04342C;"></div>
<script>
(function() {{
  function init() {{
    if (!window.TransformerViz) {{
{renderer_js}
    }}
    var mount = document.getElementById("{mount_id}");
    if (mount) window.TransformerViz.render({ir_json}, mount);
  }}
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", init);
  }} else {{
    init();
  }}
}})();
</script>
"""
        if not standalone:
            return body

        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{self.ir.name} — architecture</title>
<style>
  body {{ margin: 0; padding: 24px; background: #FAFAF7; }}
  .tv-frame {{ max-width: 760px; margin: 0 auto; }}
</style>
</head>
<body>
  <div class="tv-frame">{body}</div>
</body>
</html>
"""

    def __repr__(self) -> str:
        return f"<TransformerDiagram name={self.ir.name!r} layers={self.ir.num_layers}>"
