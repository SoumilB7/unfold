"""U11-A demand-driven imported-source address boundary."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from model_unfolder.evidence.import_source import (
    CalledImportSourceResolution,
    resolve_called_import_source,
)
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index


def _write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _fixture(tmp_path, root_source, files):
    package = tmp_path / "pkg"
    _write(package / "__init__.py", "")
    root = _write(package / "root.py", root_source)
    for name, source in files.items():
        _write(package / name, source)
    bundle = SourceBundle(
        source="test", component_files={"root": (root,)},
        import_roots={"root": (SourceImportRoot("pkg", str(package.resolve())),)},
    )
    index = build_program_index(bundle)
    call = next(item for item in index.calls
                if item.callee.kind in {"name", "attribute"}
                and item.callee.source_segment.endswith("make"))
    return bundle, index, call


def test_relative_called_import_adds_exact_source_and_symbol(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from .factory import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {"factory.py": "def make(): return object()"},
    )
    result = resolve_called_import_source(index, bundle, "root", call)
    assert result.status == "resolved"
    assert result.imported_symbol.qualified_name == "make"
    assert result.source_node.source_id.component_key == "root"
    assert len(result.index.source_nodes) == len(index.source_nodes) + 1


def test_import_alias_resolves_without_using_alias_as_semantics(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from .factory import build as make
        class Root:
            def __init__(self): self.child = make()
        """,
        {"factory.py": "def build(): return object()"},
    )
    result = resolve_called_import_source(index, bundle, "root", call)
    assert result.status == "resolved"
    assert result.imported_symbol.qualified_name == "build"


def test_same_name_in_unimported_file_cannot_enter(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from .chosen import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {
            "chosen.py": "def make(): return 'chosen'",
            "decoy.py": "def make(): return 'decoy'",
        },
    )
    result = resolve_called_import_source(index, bundle, "root", call)
    assert result.status == "resolved"
    paths = {node.source_id.canonical_path for node in result.index.source_nodes}
    assert any(path.endswith("chosen.py") for path in paths)
    assert not any(path.endswith("decoy.py") for path in paths)


def test_uncalled_import_does_not_expand_the_index(tmp_path):
    package = tmp_path / "pkg"
    _write(package / "__init__.py", "")
    root = _write(package / "root.py", """
        from .factory import make
        class Root:
            def forward(self, value): return value.relu()
    """)
    _write(package / "factory.py", "def make(): return object()")
    bundle = SourceBundle(
        source="test", component_files={"root": (root,)},
        import_roots={"root": (SourceImportRoot("pkg", str(package.resolve())),)},
    )
    index = build_program_index(bundle)
    assert len(index.source_nodes) == 1
    assert not any(node.source_id.canonical_path.endswith("factory.py")
                   for node in index.source_nodes)


def test_broken_imported_source_is_typed_failure(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from .factory import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {"factory.py": "def make(:\n    pass"},
    )
    result = resolve_called_import_source(index, bundle, "root", call)
    assert result.status == "failed"
    assert result.failure_kind == "syntax_error"


def test_external_called_import_stays_typed_incomplete(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from elsewhere.factory import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {},
    )
    result = resolve_called_import_source(index, bundle, "root", call)
    assert result.status == "incomplete"
    assert result.failure_kind == "external_import"


def test_guarded_import_cannot_be_promoted_to_one_source(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        if FLAG:
            from .factory import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {"factory.py": "def make(): return object()"},
    )
    result = resolve_called_import_source(index, bundle, "root", call)
    assert result.status == "incomplete"
    assert result.failure_kind == "guarded_import"


def test_rival_import_bindings_are_preserved_as_ambiguity(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from .one import make
        from .two import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {"one.py": "def make(): return 1", "two.py": "def make(): return 2"},
    )
    result = resolve_called_import_source(index, bundle, "root", call)
    assert result.status == "ambiguous"
    assert len(result.rival_bindings) == 2


def test_cross_component_call_is_rejected(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from .factory import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {"factory.py": "def make(): return object()"},
    )
    with pytest.raises(ValueError, match="requested component"):
        resolve_called_import_source(index, bundle, "root.vision", call)


def test_content_change_with_preserved_mtime_changes_imported_identity(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from .factory import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {"factory.py": "def make(): return 1"},
    )
    first = resolve_called_import_source(index, bundle, "root", call)
    path = tmp_path / "pkg" / "factory.py"
    stamp = path.stat()
    path.write_text("def make(): return 2", encoding="utf-8")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    second = resolve_called_import_source(index, bundle, "root", call)
    assert first.status == second.status == "resolved"
    assert first.source_node.source_id.content_fingerprint != \
        second.source_node.source_id.content_fingerprint


def test_resolution_dto_rejects_forged_resolved_payload(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from .factory import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {"factory.py": "def make(): return object()"},
    )
    with pytest.raises(ValueError, match="source, symbol and index"):
        CalledImportSourceResolution(
            "resolved", call, "root", binding_chain=(index.imports[0],))


def test_diffusers_source_bundle_declares_exact_package_root(tmp_path, monkeypatch):
    from model_unfolder.evidence import sources

    model = _write(
        tmp_path / "diffusers" / "models" / "toy.py",
        "class NeutralArchitecture: pass",
    )
    monkeypatch.setattr(
        sources, "_installed_diffusers_model_class_file",
        lambda _name: model)
    bundle = sources._installed_diffusers_bundle(
        {"_class_name": "NeutralArchitecture"})
    assert bundle.component_files == {"root": (model,)}
    assert bundle.import_roots["root"] == (
        SourceImportRoot("diffusers", str((tmp_path / "diffusers").resolve())),)


def test_exact_reexport_chain_resolves_without_name_search(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from .api import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {
            "api.py": "from .impl import build as make",
            "impl.py": "def build(): return object()",
            "decoy.py": "def make(): return 'wrong'",
        },
    )
    result = resolve_called_import_source(index, bundle, "root", call)
    assert result.status == "resolved"
    assert result.imported_symbol.qualified_name == "build"
    assert len(result.binding_chain) == 2
    assert [Path(node.source_id.canonical_path).name
            for node in result.source_chain] == ["api.py", "impl.py"]
    assert not any(node.source_id.canonical_path.endswith("decoy.py")
                   for node in result.index.source_nodes)


def test_reexport_cycle_is_typed_incomplete(tmp_path):
    bundle, index, call = _fixture(
        tmp_path,
        """
        from .one import make
        class Root:
            def __init__(self): self.child = make()
        """,
        {
            "one.py": "from .two import make",
            "two.py": "from .one import make",
        },
    )
    result = resolve_called_import_source(index, bundle, "root", call)
    assert result.status == "incomplete"
    assert result.failure_kind == "reexport_cycle"


def test_real_sdxl_factory_enters_one_index_on_exact_demand():
    import diffusers

    package = Path(diffusers.__file__).resolve().parent
    source = package / "models" / "unets" / "unet_2d_condition.py"
    bundle = SourceBundle(
        source="test", component_files={"root": (str(source),)},
        import_roots={"root": (SourceImportRoot(
            "diffusers", str(package)),)},
    )
    index = build_program_index(bundle)
    call = next(item for item in index.calls
                if item.callee.kind == "name"
                and item.callee.name == "get_down_block")
    result = resolve_called_import_source(
        index, bundle, "root", call)
    assert result.status == "resolved"
    assert result.imported_symbol.qualified_name == "get_down_block"
    assert result.source_node.source_id.canonical_path.endswith(
        "diffusers/models/unets/unet_2d_blocks.py")
    assert len(result.index.source_nodes) == len(index.source_nodes) + 1
