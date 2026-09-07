"""Exact local source substitution for the one-time S8 demonstration.

The override changes an import address, never architecture semantics. It runs
only inside the existing isolated worker, before the addressed module imports.
No installed file is changed and no cached bytecode can substitute other bytes.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys


@dataclass(frozen=True)
class SourceOverride:
    module: str
    path: str
    sha256: str

    def __post_init__(self):
        if (not isinstance(self.module, str) or "." not in self.module
                or any(not part.isidentifier() for part in self.module.split("."))):
            raise ValueError("source override requires an exact module address")
        path = Path(self.path)
        if not path.is_absolute() or path.suffix != ".py" or path.name == "__init__.py":
            raise ValueError("source override requires an absolute Python module file")
        if (len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("source override requires the expected SHA-256")

    def read_verified(self) -> bytes:
        source = Path(self.path).read_bytes()
        if hashlib.sha256(source).hexdigest() != self.sha256:
            raise ValueError("scratch source fingerprint differs from requested bytes")
        return source


class _ExactSourceLoader(importlib.abc.Loader):
    def __init__(self, override, loaded):
        self.override = override
        self.loaded = loaded

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = self.override.read_verified()
        exec(compile(source, self.override.path, "exec", dont_inherit=True), module.__dict__)
        self.loaded.add(self.override.module)


class _ExactSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, overrides):
        self.overrides = {row.module: row for row in overrides}
        self.loaded = set()

    def find_spec(self, fullname, path=None, target=None):
        override = self.overrides.get(fullname)
        if override is None:
            return None
        return importlib.util.spec_from_file_location(
            fullname, override.path,
            loader=_ExactSourceLoader(override, self.loaded))


@contextmanager
def source_overrides(overrides):
    """Refuse stale imports, unused overrides, or bytes changed during import."""
    if not overrides:
        yield
        return
    if any(row.module in sys.modules for row in overrides):
        raise ValueError("scratch source module was already imported")
    finder = _ExactSourceFinder(overrides)
    for row in overrides:
        row.read_verified()
    sys.meta_path.insert(0, finder)
    try:
        yield
        if finder.loaded != set(finder.overrides):
            raise ValueError("scratch source override was not examined by construction")
        for row in overrides:
            row.read_verified()
    finally:
        sys.meta_path.remove(finder)
