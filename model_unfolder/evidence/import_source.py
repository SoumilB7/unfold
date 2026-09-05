"""U11-A — demand-driven exact called-import source expansion.

The baseline ProgramIndex remains cheap and contains the model bundle.  A
domain reader that reaches one exact imported constructor/factory call may ask
this boundary to add that ONE imported source to a new immutable ProgramIndex.
The source observation is cached by ProgramIndex's content-addressed observer;
there is no whole-package walk and no second semantic parser.

This module resolves addresses only.  Import/module/symbol spellings never
classify an architectural mechanism, and an uncalled import cannot enter.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

from .models import SourceBundle, SourceImportRoot
from .program_index import (
    CallObservation,
    ImportRecord,
    ProgramIndex,
    SourceFileNode,
    SymbolId,
    _aggregate_fingerprint,
    build_program_index,
)


def _binding_root(expr) -> str | None:
    current = expr
    while current is not None and current.kind == "attribute" \
            and current.children:
        current = current.children[0]
    return current.name if current is not None and current.kind == "name" else None


def _module_file(root: Path, parts: tuple[str, ...]) -> Path | None:
    if not parts:
        candidate = root / "__init__.py"
        return candidate if candidate.is_file() else None
    direct = root.joinpath(*parts).with_suffix(".py")
    if direct.is_file():
        return direct
    package = root.joinpath(*parts) / "__init__.py"
    return package if package.is_file() else None


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_path(record: ImportRecord, root: SourceImportRoot) \
        -> tuple[Path | None, str]:
    package_root = Path(root.path).resolve()
    source_path = Path(record.source.canonical_path).resolve()
    if not _path_within(source_path, package_root):
        return None, "source_outside_import_root"

    target = record.target
    if target.startswith("."):
        dots = len(target) - len(target.lstrip("."))
        suffix = target[dots:]
        base = list(source_path.relative_to(package_root)
                    .with_suffix("").parts[:-1])
        ascend = dots - 1
        if ascend > len(base):
            return None, "path_escape"
        if ascend:
            base = base[:-ascend]
        parts = tuple((*base, *(p for p in suffix.split(".") if p)))
    else:
        package = tuple(root.package.split("."))
        target_parts = tuple(p for p in target.split(".") if p)
        if target_parts[:len(package)] != package:
            return None, "external_import"
        parts = target_parts[len(package):]

    # ImportFrom includes its imported symbol in target.  Plain import names
    # the module.  Test the exact module then its one parent form—never search.
    for candidate in ((parts, parts[:-1]) if parts else (parts,)):
        path = _module_file(package_root, candidate)
        if path is not None:
            if not _path_within(path, package_root):
                return None, "path_escape"
            return path.resolve(), ""
    return None, "unresolved_module"


def _merge_index(base: ProgramIndex, added: ProgramIndex) -> ProgramIndex:
    updates = {}
    for field in fields(ProgramIndex):
        if field.name in {"bundle_source", "fingerprint"}:
            continue
        left = getattr(base, field.name)
        right = getattr(added, field.name)
        if not isinstance(left, tuple) or not isinstance(right, tuple):
            raise TypeError("ProgramIndex observation surfaces are immutable tuples")
        merged = list(left)
        for item in right:
            if item not in merged:
                merged.append(item)
        updates[field.name] = tuple(merged)
    source_ids = [node.source_id for node in updates["source_nodes"]]
    source_ids.extend(item.source for item in updates["parse_failures"])
    updates["fingerprint"] = _aggregate_fingerprint(source_ids)
    return replace(base, **updates)


@dataclass(frozen=True)
class CalledImportSourceResolution:
    status: str  # resolved | ambiguous | incomplete | failed
    call: CallObservation
    component: str
    binding_chain: tuple[ImportRecord, ...] = ()
    rival_bindings: tuple[ImportRecord, ...] = ()
    source_chain: tuple[SourceFileNode, ...] = ()
    source_node: SourceFileNode | None = None
    imported_symbol: SymbolId | None = None
    index: ProgramIndex | None = None
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "ambiguous", "incomplete", "failed"}:
            raise ValueError(f"unknown called-import status {self.status!r}")
        if not isinstance(self.call, CallObservation) or not self.component:
            raise TypeError("called-import results retain exact call + component")
        if any(not isinstance(item, ImportRecord)
               for item in (*self.binding_chain, *self.rival_bindings)):
            raise TypeError("called-import provenance uses exact ImportRecords")
        if any(not isinstance(item, SourceFileNode) for item in self.source_chain):
            raise TypeError("called-import source chains use SourceFileNodes")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("a failure detail requires a typed failure kind")
        call_root = _binding_root(self.call.callee)
        if self.binding_chain:
            if self.binding_chain[0].source != self.call.enclosing_callable.source \
                    or self.binding_chain[0].alias != call_root:
                raise ValueError("the import chain begins at the exact called binding")
            if len(self.source_chain) > len(self.binding_chain) \
                    or len(self.binding_chain) - len(self.source_chain) > 1:
                raise ValueError("an import chain has at most one unresolved edge")
            for offset, binding in enumerate(
                    self.binding_chain[1:len(self.source_chain) + 1], start=1):
                if binding.source != self.source_chain[offset - 1].source_id:
                    raise ValueError("each re-export binding belongs to the prior source")
        if any(node.source_id.component_key != self.component
               for node in self.source_chain):
            raise ValueError("the entire imported source chain is component-qualified")
        if self.rival_bindings:
            if len({(item.source, item.alias)
                    for item in self.rival_bindings}) != 1:
                raise ValueError("rival imports compete for one exact lexical binding")
            expected_source = (self.source_chain[-1].source_id
                               if self.source_chain
                               else self.call.enclosing_callable.source)
            expected_alias = (self.binding_chain[-1].target.lstrip(".")
                              .split(".")[-1]
                              if self.binding_chain else call_root)
            if any(item.source != expected_source or item.alias != expected_alias
                   for item in self.rival_bindings):
                raise ValueError("rival imports occur at the exact unresolved edge")
        if self.status == "resolved":
            if not self.binding_chain or self.rival_bindings \
                    or not self.source_chain or self.source_node is None \
                    or self.imported_symbol is None \
                    or not isinstance(self.index, ProgramIndex):
                raise ValueError("resolved import carries its exact chain, source, symbol and index")
            if len(self.binding_chain) != len(self.source_chain):
                raise ValueError("every resolved import edge has an exact source")
            if self.failure_kind:
                raise ValueError("resolved import carries no failure")
            if self.source_node.source_id.component_key != self.component \
                    or self.imported_symbol.source != self.source_node.source_id \
                    or self.source_chain[-1] != self.source_node:
                raise ValueError("resolved import remains component/source qualified")
            if self.imported_symbol.qualified_name != \
                    self.binding_chain[-1].target.lstrip(".").split(".")[-1]:
                raise ValueError("the final imported symbol closes the exact target")
            if self.source_node not in self.index.source_nodes \
                    or self.index.class_by_symbol(self.imported_symbol) is None \
                    and self.index.callable_by_symbol(self.imported_symbol) is None:
                raise ValueError("the expanded index contains the source and symbol")
        elif self.status == "ambiguous":
            if len(self.rival_bindings) < 2 or any((self.source_node,
                                                    self.imported_symbol,
                                                    self.index)):
                raise ValueError("ambiguous import preserves rivals only")
        else:
            if not self.failure_kind or self.source_node is not None \
                    or self.imported_symbol is not None or self.index is not None:
                raise ValueError("incomplete/failed import carries a typed failure only")


def resolve_called_import_source(
        index: ProgramIndex,
        bundle: SourceBundle,
        component: str,
        call: CallObservation,
        ) -> CalledImportSourceResolution:
    """Add the exact source addressed by one called import binding.

    The call must already belong to ``index``.  Module-level and same-callable
    local bindings are considered; guarded/rival bindings are preserved as
    ambiguity. Exact module-level re-exports are followed as a bounded address
    chain; arbitrary calls inside the imported file are not traversed.
    """
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("called-import resolution requires ProgramIndex + SourceBundle")
    if not isinstance(call, CallObservation) \
            or call not in index.calls_in(call.enclosing_callable):
        raise ValueError("called-import resolution requires an indexed exact call")
    if call.enclosing_callable.source.component_key != component:
        raise ValueError("the call belongs to the requested component")
    alias = _binding_root(call.callee)
    if alias is None:
        return CalledImportSourceResolution(
            "incomplete", call, component,
            failure_kind="dynamic_call_target",
            failure_detail="the call target has no exact lexical root")
    bindings = tuple(item for item in index.imports
                     if item.source == call.enclosing_callable.source
                     and item.alias == alias
                     and item.enclosing_callable in {
                         None, call.enclosing_callable})
    if not bindings:
        return CalledImportSourceResolution(
            "incomplete", call, component,
            failure_kind="unbound_import",
            failure_detail="the called alias has no exact import binding")
    if len(bindings) > 1:
        return CalledImportSourceResolution(
            "ambiguous", call, component, rival_bindings=bindings)
    if bindings[0].guard:
        return CalledImportSourceResolution(
            "incomplete", call, component, binding_chain=bindings,
            failure_kind="guarded_import",
            failure_detail="the import binding is conditional")

    binding = bindings[0]
    roots = tuple(root for root in bundle.import_roots.get(component, ())
                  if _path_within(Path(binding.source.canonical_path),
                                  Path(root.path)))
    if not roots:
        return CalledImportSourceResolution(
            "incomplete", call, component, binding_chain=(binding,),
            failure_kind="import_root_unavailable",
            failure_detail="the component declares no enclosing import root")
    resolved: list[Path] = []
    failures: list[str] = []
    for root in roots:
        path, failure = _resolve_path(binding, root)
        if path is not None:
            resolved.append(path)
        elif failure:
            failures.append(failure)
    resolved = list(dict.fromkeys(resolved))
    if len(resolved) > 1:
        return CalledImportSourceResolution(
            "incomplete", call, component, binding_chain=(binding,),
            failure_kind="ambiguous_import_root",
            failure_detail="more than one declared root resolves the import")
    if not resolved:
        kind = failures[0] if failures else "unresolved_module"
        return CalledImportSourceResolution(
            "incomplete", call, component, binding_chain=(binding,),
            failure_kind=kind,
            failure_detail="the exact called import did not resolve to one source")

    expanded = index
    chain = [binding]
    source_chain = []
    seen_paths = set()
    current_path = resolved[0]
    for _hop in range(8):
        if current_path in seen_paths:
            return CalledImportSourceResolution(
                "incomplete", call, component, binding_chain=tuple(chain),
                source_chain=tuple(source_chain),
                failure_kind="reexport_cycle",
                failure_detail="the exact re-export chain cycles")
        seen_paths.add(current_path)
        existing = next((node for node in expanded.source_nodes
                         if node.source_id.component_key == component
                         and Path(node.source_id.canonical_path).resolve()
                         == current_path), None)
        if existing is None:
            added = build_program_index(SourceBundle(
                source=bundle.source,
                component_files={component: (str(current_path),)},
            ))
            if added.parse_failures:
                failure = added.parse_failures[0]
                return CalledImportSourceResolution(
                    "failed", call, component, binding_chain=tuple(chain),
                    source_chain=tuple(source_chain),
                    failure_kind=failure.kind, failure_detail=failure.detail)
            expanded = _merge_index(expanded, added)
            source_node = added.source_nodes[0]
        else:
            source_node = existing
        source_chain.append(source_node)

        imported_name = chain[-1].target.lstrip(".").split(".")[-1]
        symbols = tuple(
            item.symbol for item in (*expanded.classes, *expanded.callables)
            if item.symbol.source == source_node.source_id
            and item.symbol.qualified_name == imported_name)
        if len(symbols) == 1:
            return CalledImportSourceResolution(
                "resolved", call, component, binding_chain=tuple(chain),
                source_chain=tuple(source_chain), source_node=source_node,
                imported_symbol=symbols[0], index=expanded)
        if len(symbols) > 1:
            return CalledImportSourceResolution(
                "incomplete", call, component, binding_chain=tuple(chain),
                source_chain=tuple(source_chain),
                failure_kind="imported_symbol_ambiguous",
                failure_detail="the imported source defines rival target symbols")

        reexports = tuple(item for item in expanded.imports
                          if item.source == source_node.source_id
                          and item.enclosing_callable is None
                          and item.alias == imported_name)
        if len(reexports) > 1:
            return CalledImportSourceResolution(
                "ambiguous", call, component, binding_chain=tuple(chain),
                rival_bindings=reexports, source_chain=tuple(source_chain))
        if not reexports:
            return CalledImportSourceResolution(
                "incomplete", call, component, binding_chain=tuple(chain),
                source_chain=tuple(source_chain),
                failure_kind="imported_symbol_absent",
                failure_detail="the exact source neither defines nor re-exports the target")
        if reexports[0].guard:
            return CalledImportSourceResolution(
                "incomplete", call, component,
                binding_chain=tuple((*chain, reexports[0])),
                source_chain=tuple(source_chain),
                failure_kind="guarded_reexport",
                failure_detail="the exact re-export is conditional")
        chain.append(reexports[0])
        next_paths = []
        next_failures = []
        for root in roots:
            path, failure = _resolve_path(reexports[0], root)
            if path is not None:
                next_paths.append(path)
            elif failure:
                next_failures.append(failure)
        next_paths = list(dict.fromkeys(next_paths))
        if len(next_paths) != 1:
            return CalledImportSourceResolution(
                "incomplete", call, component, binding_chain=tuple(chain),
                source_chain=tuple(source_chain),
                failure_kind=("ambiguous_import_root" if len(next_paths) > 1
                              else (next_failures[0] if next_failures
                                    else "unresolved_module")),
                failure_detail="the exact re-export did not resolve to one source")
        current_path = next_paths[0]
    return CalledImportSourceResolution(
        "incomplete", call, component, binding_chain=tuple(chain),
        source_chain=tuple(source_chain), failure_kind="reexport_depth_limit",
        failure_detail="the exact re-export chain exceeds eight sources")


def canonical_called_import_target(
        bundle: SourceBundle,
        resolution: CalledImportSourceResolution,
        ) -> CanonicalCalledImportTarget | None:
    """Return the unique package-qualified address of a resolved import.

    A relative import spelling has no global meaning.  This address-only join
    uses the exact resolved source node and the component's declared import
    root to produce ``package.module.Symbol``.  It does not classify that
    address as any architectural mechanism.
    """
    if not isinstance(bundle, SourceBundle) \
            or not isinstance(resolution, CalledImportSourceResolution):
        raise TypeError("canonical import target requires bundle + resolution")
    if resolution.status != "resolved":
        return None
    source = Path(resolution.source_node.source_id.canonical_path).resolve()
    matches = []
    for root in bundle.import_roots.get(resolution.component, ()):
        root_path = Path(root.path).resolve()
        if not _path_within(source, root_path):
            continue
        relative = source.relative_to(root_path)
        module_parts = (relative.parts[:-1] if relative.name == "__init__.py"
                        else (*relative.parts[:-1], relative.stem))
        module = ".".join((root.package, *module_parts))
        matches.append(CanonicalCalledImportTarget(
            bundle, resolution, root,
            f"{module}.{resolution.imported_symbol.qualified_name}"))
    unique = {}
    for item in matches:
        key = (item.import_root.package, item.import_root.path,
               item.qualified_target)
        unique.setdefault(key, item)
    return next(iter(unique.values())) if len(unique) == 1 else None


@dataclass(frozen=True)
class CanonicalCalledImportTarget:
    """Self-verifying package address for one exact called-import proof."""

    bundle: SourceBundle
    resolution: CalledImportSourceResolution
    import_root: SourceImportRoot
    qualified_target: str

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, SourceBundle) \
                or not isinstance(self.resolution, CalledImportSourceResolution) \
                or not isinstance(self.import_root, SourceImportRoot) \
                or not self.qualified_target:
            raise TypeError("canonical import evidence is fully typed")
        if self.resolution.status != "resolved" \
                or self.import_root not in self.bundle.import_roots.get(
                    self.resolution.component, ()):
            raise ValueError("canonical import evidence belongs to a resolved declared root")
        source = Path(
            self.resolution.source_node.source_id.canonical_path).resolve()
        root_path = Path(self.import_root.path).resolve()
        if not _path_within(source, root_path):
            raise ValueError("the resolved source belongs to the declared root")
        relative = source.relative_to(root_path)
        module_parts = (relative.parts[:-1] if relative.name == "__init__.py"
                        else (*relative.parts[:-1], relative.stem))
        expected = ".".join((
            self.import_root.package, *module_parts,
            self.resolution.imported_symbol.qualified_name))
        if self.qualified_target != expected:
            raise ValueError("the canonical target derives from root + source + symbol")


__all__ = [
    "CalledImportSourceResolution",
    "CanonicalCalledImportTarget",
    "canonical_called_import_target",
    "resolve_called_import_source",
]
