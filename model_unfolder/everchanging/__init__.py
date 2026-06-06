"""User-editable config vocabulary — *data, not code*.

Files in this folder are loaded at runtime so new config dialects can be
supported by editing YAML, with the parser code left untouched.  Anything that
is config-variable vocabulary — aliases, ignore lists, word types — lives here
as YAML, never hardcoded in the parser or renderers.

* ``aliases.yaml`` — canonical field name -> list of config key spellings the
  parser tries, in order.  Add a spelling, or a whole new field.
* ``ignored_fields.yaml`` — config keys / suffixes that are *not* architectural,
  skipped by the unparsed-field diagnostic.

Loading prefers PyYAML when installed; otherwise a tiny built-in reader handles
both the flow-style (``key: [a, b, c]``) and block-style (``key:`` + ``- item``)
formats the shipped files use, so the package keeps working with no third-party
dependency.
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent


def _load(filename: str) -> dict:
    """Load a YAML data file from this folder (PyYAML if present, else built-in)."""
    text = (_DIR / filename).read_text(encoding="utf-8")
    try:
        import yaml  # optional; not a hard dependency
    except ImportError:
        return _parse_flow_yaml(text)
    return yaml.safe_load(text) or {}


def load_aliases() -> dict[str, list[str]]:
    """Load the transformer field-alias table from ``aliases.yaml``."""
    return _load("aliases.yaml")


def load_diffusion_aliases() -> dict[str, list[str]]:
    """Load the diffusion (DiT/MMDiT) field-alias table from ``diffusion_aliases.yaml``."""
    return _load("diffusion_aliases.yaml")


def load_ignored_fields() -> dict[str, list[str]]:
    """Load non-architectural config keys/suffixes from ``ignored_fields.yaml``."""
    data = _load("ignored_fields.yaml")
    return {"keys": data.get("keys") or [], "suffixes": data.get("suffixes") or []}


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


__all__ = ["load_aliases", "load_diffusion_aliases", "load_ignored_fields"]
