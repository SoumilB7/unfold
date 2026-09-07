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

    def forward_is_unmodified(self, path: str) -> bool:
        """Recorded instance overrides/hooks defeat class-forward authority."""
        module = self._modules.get(path)
        if module is None:
            return False
        attrs = module.init_attributes
        return ("forward" not in attrs and not attrs.get("_forward_hooks")
                and not attrs.get("_forward_pre_hooks") and not attrs.get("_compiled_call_impl"))

    def route_forwards_unmodified(self, path: str) -> bool:
        parts = path.split(".") if path else []
        return all(self.forward_is_unmodified(".".join(parts[:offset]))
                   for offset in range(len(parts) + 1))

    def primitive_at(self, path: str) -> str | None:
        """Worker identity plus an unchanged invoked route, never an address vote."""
        module = self._modules.get(path)
        if module is None or not self.route_forwards_unmodified(path):
            return None
        from .primitive_semantics import runtime_primitive_definition
        witness = module.framework_primitive
        if runtime_primitive_definition(witness) is None:
            return None
        # This is a consistency check on the two worker channels, not positive
        # authority from a spelling. A replaced occurrence cannot retain an
        # earlier primitive's witness after the reconciliation row is rebuilt.
        addresses = {
            "linear": ("linear", "Linear"),
            "conv1d": ("conv", "Conv1d"), "conv2d": ("conv", "Conv2d"),
            "conv3d": ("conv", "Conv3d"), "group_norm": ("normalization", "GroupNorm"),
            "layer_norm": ("normalization", "LayerNorm"), "rms_norm": ("normalization", "RMSNorm"),
            "silu": ("activation", "SiLU"), "gelu": ("activation", "GELU"),
            "relu": ("activation", "ReLU"), "dropout": ("dropout", "Dropout"),
        }
        module_name, qualname = addresses[witness.key]
        if (module.class_ref.module, module.class_ref.qualname) != (
                "torch.nn.modules." + module_name, qualname):
            return None
        return witness.key

    def construction_members(self, population, construction) -> tuple[str, ...]:
        """Join a selected constructor occurrence to its exact runtime slot."""
        stage = population.stage.occurrence_id.parent_field
        if population.selected.position is not None:
            stage += f".{population.selected.position}"
        field_path = f"{stage}.{population.field}"
        if construction not in population.present_constructions:
            return ()
        if population.storage_kind == "direct":
            return (field_path,) if field_path in self._modules else ()
        record = population.container_record
        members = self.direct_members(field_path, repeated=True)
        if record is None or len(record.elements) != len(members):
            # A symbolic template with varying constructor operands needs its
            # own iteration-to-slot proof, never same-class broadcasting.
            return ()
        positions = [number for number, site in enumerate(record.elements)
                     if construction.site == site]
        return (members[positions[0]],) if len(positions) == 1 else ()


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

    @cached_property
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
            definition = (runtime_primitive_definition(module.framework_primitive)
                          if self.bindings.primitive_at(module.path) is not None else None)
            if meaning is not None and definition is not None:
                kind, label, function = definition
                rows[module.path] = {"kind": kind, "label": label, "function": function,
                                     "method_source_sha256": module.framework_primitive.forward_source_sha256}
        return rows

    def summary(self):
        return self._summary

    @cached_property
    def _summary(self):
        from .claim_evidence import ClaimProofSummary
        from .receipts import value_status_hash
        return ClaimProofSummary(
            self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols,
            tuple(sorted({f"runtime-types:{value_status_hash(self.value, 'code_proven')}",
                          *(f"source:{row.module}:{row.sha256}" for row in
                            self.bindings.inventory.provenance.source_files)})),
            document_fingerprints=(self.bindings.table.config_sha256,))
