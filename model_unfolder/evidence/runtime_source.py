"""Bind source-reader addresses to the reconciled instance denominator.

This is an evidence address lookup, not another architecture graph. Runtime
construction chooses the object; exact source bytes choose the class callable
that readers may investigate. Neither a familiar name nor a trace supplies its
mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from .program_index import ProgramIndex, SymbolId
from .reconciliation import ReconciliationTable


@dataclass(frozen=True)
class RuntimeSourceBindings:
    table: ReconciliationTable
    inventory: object = field(repr=False, compare=False)
    index: ProgramIndex = field(repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.table, ReconciliationTable):
            raise TypeError("runtime bindings require the reconciled denominator")
        if not isinstance(self.index, ProgramIndex):
            raise TypeError("runtime bindings require the authoritative ProgramIndex")
        if (type(self.inventory).__module__, type(self.inventory).__qualname__) != (
                "physics.instance_inventory", "InstanceInventory"):
            raise TypeError("runtime bindings require the exact instance inventory")
        if self.inventory.provenance.config_sha256 != self.table.config_sha256:
            raise ValueError("runtime bindings must share checkpoint provenance")
        rows = {row.provenance.instance_path: row for row in self.table.occurrences}
        for module in self.inventory.modules:
            row = rows.get(module.path)
            if row is None or row.provenance.runtime_class is None:
                raise ValueError("instance occurrence is absent from reconciliation")
            cls = row.provenance.runtime_class
            if (cls.module, cls.qualname) != (module.class_ref.module, module.class_ref.qualname):
                raise ValueError("runtime class differs from reconciled occurrence")

    @cached_property
    def _rows(self):
        return {row.provenance.instance_path: row for row in self.table.occurrences}

    @cached_property
    def _modules(self):
        return {row.path: row for row in self.inventory.modules}

    @cached_property
    def _symbols(self):
        result = {}
        for record in self.index.classes:
            key = (record.symbol.qualified_name, record.symbol.source.content_fingerprint)
            result.setdefault(key, []).append(record.symbol)
        return result

    def symbol_at(self, path: str) -> SymbolId | None:
        """An exact indexed class, or no source binding; never a class-name vote."""
        row = self._rows.get(path)
        if row is None or row.construction.kind not in {"eager_constructed", "lazy_observed"}:
            return None
        module = self._modules.get(path)
        if module is None:
            return None
        hashes = {source.sha256 for source in self.inventory.provenance.source_files
                  if source.module == module.class_ref.module}
        candidates = tuple(symbol for fingerprint in hashes
                           for symbol in self._symbols.get((module.class_ref.qualname, fingerprint), ())
                           if symbol.source.component_key == "root")
        return candidates[0] if len(candidates) == 1 else None

    def direct_members(self, field_path: str, *, repeated: bool) -> tuple[str, ...]:
        """Instance addresses under an exact source-established field."""
        if not repeated:
            return (field_path,) if field_path in self._modules else ()
        container = self._modules.get(field_path)
        if container is None:
            return ()
        return tuple(f"{field_path}.{child}" for child in container.children)

    def matching_members(self, field_path: str, symbol: SymbolId, *, repeated: bool) -> tuple[str, ...]:
        return tuple(path for path in self.direct_members(field_path, repeated=repeated)
                     if self.symbol_at(path) == symbol)


@dataclass(frozen=True)
class RuntimePrimitiveClaimProof:
    """An exact framework type proves its operation, never its caller's wiring."""

    fact_id: str
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)

    claim_kind = "applied_function"
    proof_kind = "exact_framework_runtime_type"
    reader_symbols = ("evidence.primitive_semantics.read_runtime_primitives",)

    def __post_init__(self):
        if self.fact_id != "root.denoiser.runtime_primitives" or not isinstance(self.bindings, RuntimeSourceBindings):
            raise ValueError("primitive claims require the exact reconciled framework types")

    @property
    def value(self):
        from .primitive_semantics import runtime_primitive_definition
        rows = {}
        for module in self.bindings.inventory.modules:
            if "forward" in module.init_attributes or any(
                    module.init_attributes.get(key) for key in ("_forward_hooks", "_forward_pre_hooks")):
                # The canonical type does not certify an instance-specific
                # forward replacement or hook's computation.
                continue
            meaning = self.bindings._rows[module.path].provenance.meaning.framework_primitive
            definition = runtime_primitive_definition(module.class_ref)
            if meaning is not None and definition is not None:
                kind, label, function = definition
                rows[module.path] = {"kind": kind, "label": label, "function": function}
        return rows

    def summary(self):
        from .claim_evidence import ClaimProofSummary
        from .receipts import value_status_hash
        return ClaimProofSummary(
            self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols,
            tuple(sorted({f"runtime-types:{value_status_hash(self.value, 'code_proven')}",
                          *(f"source:{row.module}:{row.sha256}" for row in
                            self.bindings.inventory.provenance.source_files)})),
            document_fingerprints=(self.bindings.table.config_sha256,))
