"""Static source-code evidence for model topology validation."""
from .conformance import (
    ConformanceProblem,
    check_fact_conformance,
    check_model_conformance,
    check_nested_conformance,
    check_wiring_conformance,
)
from .forward_ops import extract_forward_ops
from .context import ParseContext
from .inspector import inspect_model_code
from .identity_guard import (
    IdentityViolation,
    NameBlindResult,
    name_blind_diff,
    scan_identity_debt,
    scrub_semantic_identity,
)
from .models import (
    AudioCallableEvidence,
    AudioLayerEvidence,
    AudioTowerEvidence,
    ClassEvidence,
    CodeEvidence,
    CodeFinding,
    ForwardOps,
    PositionalEvidence,
    PositionalMechanism,
    SourceOp,
    SourceBundle,
    VisionLayerEvidence,
    VisionTowerEvidence,
)
from .position import decoder_positional_evidence
from .vision import vision_tower_evidence
from .projector import projector_evidence
from .fusion import fusion_evidence
from .audio import audio_tower_evidence
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
    "AudioCallableEvidence",
    "AudioLayerEvidence",
    "AudioTowerEvidence",
    "CodeEvidence",
    "CodeFinding",
    "ForwardOps",
    "PositionalEvidence",
    "PositionalMechanism",
    "SourceBundle",
    "SourceOp",
    "VisionLayerEvidence",
    "VisionTowerEvidence",
    "ParseContext",
    "decoder_positional_evidence",
    "vision_tower_evidence",
    "projector_evidence",
    "fusion_evidence",
    "audio_tower_evidence",
    "IdentityViolation",
    "NameBlindResult",
    "name_blind_diff",
    "scan_identity_debt",
    "scrub_semantic_identity",
]
