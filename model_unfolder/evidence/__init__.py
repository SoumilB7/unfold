"""Static source-code evidence for model topology validation."""
from .conformance import (
    ConformanceProblem,
    check_fact_conformance,
    check_model_conformance,
    check_nested_conformance,
    check_wiring_conformance,
)
from .forward_ops import extract_forward_ops
from .inspector import inspect_model_code
from .models import (
    ClassEvidence,
    CodeEvidence,
    CodeFinding,
    ForwardOps,
    PositionalEvidence,
    PositionalMechanism,
    SourceBundle,
)
from .position import decoder_positional_evidence
from .validate import validate_ir_with_evidence

__all__ = [
    "inspect_model_code",
    "extract_forward_ops",
    "check_model_conformance",
    "check_wiring_conformance",
    "check_fact_conformance",
    "check_nested_conformance",
    "validate_ir_with_evidence",
    "ConformanceProblem",
    "ClassEvidence",
    "CodeEvidence",
    "CodeFinding",
    "ForwardOps",
    "PositionalEvidence",
    "PositionalMechanism",
    "SourceBundle",
    "decoder_positional_evidence",
]
