"""Existence proof for the reconciled constructed module population."""
from dataclasses import dataclass, field

from .claim_evidence import ClaimProofSummary
from .program_index import portable_source_index_fingerprint
from .receipts import value_status_hash
from .runtime_source import RuntimeSourceBindings


@dataclass(frozen=True)
class InstancePopulationClaimProof:
    fact_id: str
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)

    claim_kind = "existence"
    proof_kind = "constructed_module_population"
    reader_symbols = ("evidence.instance_population_claim.read_instance_population",)

    def __post_init__(self):
        if self.fact_id != "root.denoiser.constructed_modules":
            raise ValueError("module population proof belongs to its exact existence fact")
        if not isinstance(self.bindings, RuntimeSourceBindings):
            raise TypeError("module existence requires the reconciled instance")

    @property
    def value(self):
        return {module.path: {
            "class_module": module.class_ref.module,
            "class_name": module.class_ref.qualname,
            "children": list(module.children),
        } for module in self.bindings.inventory.modules}

    def summary(self):
        provenance = self.bindings.inventory.provenance
        return ClaimProofSummary(
            self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols,
            tuple(sorted({f"checkpoint:{provenance.config_sha256}",
                          f"population:{value_status_hash(self.value, 'code_proven')}",
                          *(f"source:{source.module}:{source.sha256}"
                            for source in provenance.source_files)})),
            document_fingerprints=(provenance.config_sha256,))


def read_instance_population(bindings):
    from .facts import EvidenceFact

    proof = InstancePopulationClaimProof("root.denoiser.constructed_modules", bindings)
    return EvidenceFact(
        key="constructed_modules", owner="root.denoiser", value=proof.value,
        status="code_proven", completeness="complete", claim_kind=proof.claim_kind,
        claim_readers=proof.reader_symbols, claim_evidence=proof)


@dataclass(frozen=True)
class ConstructorDefaultsClaimProof:
    """A default declaration is a value, never a mechanism or checkpoint fact."""
    fact_id: str
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)
    document: object = field(repr=False, compare=False)

    claim_kind = "value"
    proof_kind = "omitted_constructor_parameter_default_declarations"
    reader_symbols = ("evidence.instance_population_claim.read_constructor_defaults",)

    def __post_init__(self):
        from .reconciliation import _sha256
        from .document import PreparedDocument
        if self.fact_id != "root.denoiser.declared_constructor_defaults":
            raise ValueError("default proof belongs to its exact value fact")
        if not isinstance(self.bindings, RuntimeSourceBindings) or not isinstance(self.document, PreparedDocument):
            raise TypeError("default declarations require exact source and checkpoint evidence")
        if _sha256(self.document.checkpoint) != self.bindings.table.config_sha256:
            raise ValueError("default declarations belong to another checkpoint")

    def declarations(self):
        from .reconciliation import _sha256
        from .expression_eval import ConfigExpressionEvaluator
        from .program_index import SymbolId
        if _sha256(self.document.checkpoint) != self.bindings.table.config_sha256:
            raise ValueError("default declaration checkpoint changed after qualification")
        root = self.bindings.symbol_at("")
        if root is None:
            return ()
        constructor = self.bindings.index.callable_by_symbol(SymbolId(root.source, root.qualified_name + ".__init__"))
        if constructor is None:
            return ()
        evaluator = ConfigExpressionEvaluator((), {}, {}, allow_control_literals=True)
        return tuple((param, value) for param in constructor.params
                     if param.has_default and param.default is not None
                     and param.name not in self.document.checkpoint
                     and (value := evaluator.expression(param.default)) is not None)

    @property
    def value(self):
        return {param.name: {"value": value.value, "provenance": "class_default",
                             "checkpoint": "omitted"}
                for param, value in self.declarations()}

    def summary(self):
        refs = tuple(sorted({f"sha256:{param.default.span.source.content_fingerprint}:"
                             f"{param.default.span.line}:{param.default.span.col}"
                             for param, _ in self.declarations()}))
        return ClaimProofSummary(self.fact_id, self.claim_kind, self.proof_kind,
                                 self.reader_symbols, refs,
                                 document_fingerprints=(self.bindings.table.config_sha256,),
                                 index_fingerprints=(portable_source_index_fingerprint(self.bindings.index),))


def read_constructor_defaults(bindings, document):
    from .facts import EvidenceFact
    proof = ConstructorDefaultsClaimProof("root.denoiser.declared_constructor_defaults", bindings, document)
    value = proof.value
    if not value:
        return None
    return EvidenceFact(key="declared_constructor_defaults", owner="root.denoiser",
                        value=value, status="class_default", completeness="presence_only",
                        claim_kind=proof.claim_kind, claim_readers=proof.reader_symbols,
                        claim_evidence=proof)
