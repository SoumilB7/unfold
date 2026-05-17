"""Public inspection entry points for static code evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .ast_scanner import scan_python_files
from .models import CodeEvidence
from .patterns import infer_code_evidence
from .sources import resolve_source_files


def inspect_model_code(target: Any, *, source: str = "local", token: Any = None) -> CodeEvidence:
    """Inspect HF-style model code without importing or executing it.

    ``target`` can be a config dict/object, a model id, or a local Python
    file/directory.  By default, the scanner uses the installed Transformers
    package.  ``source`` may also be a local file/directory path. Use
    ``source="hub"`` only when you explicitly want to download repository
    source files; the downloader filters for ``*.py``/JSON files and ignores
    weight formats.
    """
    if source not in {"local", "path", "hub", "auto"} and Path(source).exists():
        target = source
        source = "path"
    bundle = resolve_source_files(target, source=source, token=token)
    classes = scan_python_files(bundle.files)
    return infer_code_evidence(bundle, classes)
