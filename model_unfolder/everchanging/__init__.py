"""User-editable config vocabulary — *data, not code*.

Files in this folder are loaded at runtime so new config dialects can be
supported by editing JSON, with the parser code left untouched.

* ``aliases.json`` — canonical field name -> list of config key spellings the
  parser will try, in order.  Add a new spelling for an existing field, or a
  whole new field, by editing that file.
"""
from __future__ import annotations

import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def load_aliases() -> dict[str, list[str]]:
    """Load the field-alias table from ``aliases.json``."""
    with open(_DIR / "aliases.json", encoding="utf-8") as f:
        return json.load(f)


__all__ = ["load_aliases"]
