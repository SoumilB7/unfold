"""Value proof for exact constructed parameter shapes, including aliases."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .claim_evidence import ClaimProofSummary
from .receipts import value_status_hash
from .runtime_source import RuntimeSourceBindings


@dataclass(frozen=True)
class InstanceShapeClaimProof:
    """Shapes prove counts and dimensions; they cannot prove a connection."""

    fact_id: str
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)

    claim_kind = "value"
    proof_kind = "constructed_parameter_shapes"
    reader_symbols = ("evidence.instance_shape_claim.read_instance_shapes",)

    def __post_init__(self):
        if self.fact_id != "root.denoiser.constructed_parameter_shapes":
            raise ValueError("shape proof belongs to the denoiser shape fact")
        if not isinstance(self.bindings, RuntimeSourceBindings):
            raise TypeError("shape proof requires the reconciled instance")

    @property
    def value(self):
        inventory = self.bindings.inventory
        parameters = {
            f"{module.path}.{parameter.name}".lstrip("."): parameter
            for module in inventory.modules for parameter in module.parameters}
        representative = {name: group.names[0]
                          for group in inventory.parameter_aliases for name in group.names}
        unique = {representative.get(name, name): parameter
                  for name, parameter in parameters.items()}
        # Alias equivalence is only useful if all aliases agree on the shape.
        if any(parameters[name].shape != parameters[representative[name]].shape
               for name in representative):
            raise ValueError("shared parameter identities disagree on shape")
        identities_by_module = {module.path: set() for module in inventory.modules}
        for module in inventory.modules:
            ancestors = [""]
            parts = module.path.split(".") if module.path else []
            ancestors.extend(".".join(parts[:end]) for end in range(1, len(parts) + 1))
            for parameter in module.parameters:
                name = f"{module.path}.{parameter.name}".lstrip(".")
                identity = representative.get(name, name)
                for ancestor in ancestors:
                    identities_by_module[ancestor].add(identity)
        return {
            "scope": "denoiser",
            "total": sum(math.prod(parameter.shape) for parameter in unique.values()),
            "parameterized_modules": sum(bool(module.parameters) for module in inventory.modules),
            "by_module": {path: sum(math.prod(unique[name].shape) for name in identities)
                          for path, identities in identities_by_module.items()},
            "parameters": {name: {"shape": list(parameter.shape),
                                  "dtype": parameter.dtype,
                                  "identity": representative.get(name, name)}
                           for name, parameter in sorted(parameters.items())},
        }

    def summary(self):
        inventory = self.bindings.inventory
        return ClaimProofSummary(
            self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols,
            tuple(sorted({
                f"checkpoint:{inventory.provenance.config_sha256}",
                f"shape-value:{value_status_hash(self.value, 'code_proven')}",
                *(f"source:{source.module}:{source.sha256}"
                  for source in inventory.provenance.source_files),
            })),
            document_fingerprints=(inventory.provenance.config_sha256,))


def read_instance_shapes(bindings):
    """Declare exactly the values extracted from the constructed denominator."""
    from .facts import EvidenceFact

    proof = InstanceShapeClaimProof("root.denoiser.constructed_parameter_shapes", bindings)
    return EvidenceFact(
        key="constructed_parameter_shapes", owner="root.denoiser",
        value=proof.value, status="code_proven", completeness="complete",
        claim_kind=proof.claim_kind, claim_readers=proof.reader_symbols,
        claim_evidence=proof)
