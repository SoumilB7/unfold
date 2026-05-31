"""User-editable config vocabulary — *data, not code*.

Files in this folder are loaded at runtime so new config dialects can be
supported by editing YAML, with the parser code left untouched.

* ``aliases.yaml`` — canonical field name -> list of config key spellings the
  parser will try, in order.  Add a new spelling for an existing field, or a
  whole new field, by editing that file.

Loading prefers PyYAML when installed; otherwise a tiny built-in reader handles
the flow-style (``key: [a, b, c]``) format the shipped file uses, so the package
keeps working with no third-party dependency.
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent


def load_aliases() -> dict[str, list[str]]:
    """Load the field-alias table from ``aliases.yaml``."""
    text = (_DIR / "aliases.yaml").read_text(encoding="utf-8")
    try:
        import yaml  # optional; not a hard dependency
    except ImportError:
        return _parse_flow_yaml(text)
    return yaml.safe_load(text) or {}


def _parse_flow_yaml(text: str) -> dict[str, list[str]]:
    """Minimal reader for ``key: [a, b, c]`` / ``key:`` + ``- item`` blocks.

    Handles ``#`` comments and quoted scalars — enough for the alias file when
    PyYAML is absent.  Not a general YAML parser.
    """
    data: dict = {}
    current = None
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0] if " #" in raw else raw
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current is not None:
            data[current].append(_unquote(stripped[2:]))
            continue
        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        key, val = key.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            data[key] = [_unquote(x) for x in val[1:-1].split(",") if x.strip()]
            current = None
        elif val:
            data[key] = _unquote(val)
            current = None
        else:
            data[key] = []
            current = key
    return data


def _unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


__all__ = ["load_aliases"]
