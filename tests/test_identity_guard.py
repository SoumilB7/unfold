"""The identity guard, BLOCKING: debt is zero and stays zero.

Unit 9 endpoint.  Every identity-to-structure mechanism the static net detects
is now either eradicated (debt == []) or consciously reclassified as DECLARED
class vocabulary — two lawful categories pinned exactly below so a new table or
use site is a reviewed act, never an accident:

* code-shape markers classify a class ALREADY RESOLVED from the model's own
  ``__init__`` evidence (conformance registry) — same category as forward_ops
  type-roles (RMSNorm -> norm);
* declared-component markers read a diffusers config's OWN ``_class_name``
  constructor declaration (like ``architectures[0]``), never a per-model table.
"""
from __future__ import annotations

from collections import Counter

from model_unfolder.evidence.identity_guard import (
    name_blind_diff,
    scan_declared_class_vocabulary,
    scan_identity_debt,
    scan_identity_source,
    scan_identity_yaml_source,
)


def test_identity_debt_is_zero_and_blocking():
    """No identity-derived architectural decision anywhere in production code.
    This is the Unit 9 flip: the guard is no longer report-only — ANY new
    identity branch / helper / family-fact table fails this test."""
    assert scan_identity_debt() == []


# The multiset of (file, kind, detail) -> count for DECLARED vocabulary,
# deliberately NOT keyed on line number (line pins train blind bumping).  A new
# runtime access bumps a count; a new table/file adds a key — both require a
# conscious edit HERE with the category justification.
EXPECTED_DECLARED_VOCABULARY = {
    ("model_unfolder/adapters/diffusor/parser.py",
     "declared_class_vocabulary",
     "runtime access to declared class vocabulary 'dit_class_markers' "
     "(declared-component: reads the config's own _class_name declaration)"): 1,
    ("model_unfolder/adapters/diffusor/parser.py",
     "declared_class_vocabulary",
     "runtime access to declared class vocabulary 'scheduler_flow_matching_markers' "
     "(declared-component: reads the config's own _class_name declaration)"): 1,
    ("model_unfolder/evidence/conformance.py",
     "declared_class_vocabulary",
     "runtime access to declared class vocabulary 'component_class_markers' "
     "(code-shape: classifies a class resolved from init evidence)"): 2,
    ("model_unfolder/evidence/conformance.py",
     "declared_class_vocabulary",
     "runtime access to declared class vocabulary 'drill_class_markers' "
     "(code-shape: classifies a class resolved from init evidence)"): 1,
    ("model_unfolder/evidence/conformance.py",
     "declared_class_vocabulary",
     "runtime access to declared class vocabulary 'processor_markers' "
     "(code-shape: classifies a class resolved from init evidence)"): 1,
    ("model_unfolder/evidence/conformance.py",
     "declared_class_vocabulary",
     "runtime access to declared class vocabulary 'single_stream_class_markers' "
     "(code-shape: classifies a class resolved from init evidence)"): 1,
    ("model_unfolder/evidence/sources.py",
     "declared_class_vocabulary",
     "runtime access to declared class vocabulary 'dit_class_markers' "
     "(declared-component: reads the config's own _class_name declaration)"): 1,
    ("model_unfolder/everchanging/conformance/conformance_map.yaml",
     "declared_vocabulary_table",
     "populated declared class vocabulary 'single_stream_class_markers' (code-shape)"): 1,
    ("model_unfolder/everchanging/conformance/transitive.yaml",
     "declared_vocabulary_table",
     "populated declared class vocabulary 'component_class_markers' (code-shape)"): 1,
    ("model_unfolder/everchanging/conformance/transitive.yaml",
     "declared_vocabulary_table",
     "populated declared class vocabulary 'drill_class_markers' (code-shape)"): 1,
    ("model_unfolder/everchanging/conformance/transitive.yaml",
     "declared_vocabulary_table",
     "populated declared class vocabulary 'processor_markers' (code-shape)"): 1,
    ("model_unfolder/everchanging/diffusor/typing.yaml",
     "declared_vocabulary_table",
     "populated declared class vocabulary 'dit_class_markers' (declared-component)"): 1,
    ("model_unfolder/everchanging/diffusor/typing.yaml",
     "declared_vocabulary_table",
     "populated declared class vocabulary 'scheduler_flow_matching_markers' "
     "(declared-component)"): 1,
}


def test_declared_class_vocabulary_is_pinned_exactly():
    actual = Counter((item.path, item.kind, item.detail)
                     for item in scan_declared_class_vocabulary())
    assert dict(actual) == EXPECTED_DECLARED_VOCABULARY


def test_static_guard_negative_controls_cover_all_three_identity_mechanisms():
    direct = scan_identity_source(
        "def parse(model_type):\n"
        "    if model_type == 'pixtral':\n"
        "        return {'norm_kind': 'RMSNorm'}\n"
    )
    assert any(item.kind == "identity_branch" for item in direct)

    profile = scan_identity_source(
        "def card(profile):\n"
        "    profile_title = {'qwen': 'Qwen merger'}.get(profile)\n"
        "    return {'title': profile_title}\n"
    )
    assert any(item.kind == "identity_profile" for item in profile)

    table = scan_identity_yaml_source(
        "norm_kind:\n  pixtral: RMSNorm\n  siglip: LayerNorm\n"
    )
    assert any(item.kind == "identity_table" for item in table)


def test_static_guard_catches_class_name_domain_substring_inside_evidence():
    findings = scan_identity_source(
        "def choose(block_class):\n"
        "    if 'vision' in block_class.lower():\n"
        "        return {'kind': 'vision_encoder'}\n",
        path="model_unfolder/evidence/new_detector.py",
    )
    assert any(item.kind == "class_identity_branch" for item in findings)


def test_declared_vocabulary_is_still_detected_never_invisible():
    """Reclassification must not blind the scanner: a declared-table access and
    a populated declared table are still FINDINGS (of the declared kind, pinned
    above), and a family-keyed fact table remains identity DEBT."""
    access = scan_identity_source("ops = vocab['processor_markers']\n")
    assert any(item.kind == "declared_class_vocabulary" for item in access)

    table = scan_identity_yaml_source("drill_class_markers:\n  - gated-delta=DeltaNet\n")
    assert any(item.kind == "declared_vocabulary_table" for item in table)
    assert not any(item.kind == "identity_table" for item in table)

    debt_table = scan_identity_yaml_source("ffn_activation_fn:\n  pixtral: silu\n")
    assert any(item.kind == "identity_table" for item in debt_table)


def test_name_blind_guard_preserves_vision_structure_with_pre_resolved_source():
    from tests.test_declared_ops import PIXTRAL_STYLE

    result = name_blind_diff(PIXTRAL_STYLE)
    assert result.structural_equal
    assert result.changed_paths == ()


def test_name_blind_guard_preserves_source_address_and_clean_decoder_structure():
    from transformers import AutoConfig

    result = name_blind_diff(AutoConfig.for_model("llama").to_dict())
    assert result.structural_equal
    assert result.changed_paths == ()


def test_name_blind_guard_over_blessed_corpus():
    """BLOCKING corpus net, STRICT: every blessed fixture — LLM and diffusion —
    must parse structurally IDENTICAL with all semantic identity scrubbed.
    (The former text-encoder containment tolerance is gone: the encoder
    sub-parse now inherits the root's pre-resolved component bundle via
    ``_slot_context``, so a scrubbed sub-config loses no address.)"""
    from model_unfolder.sable import load_corpus

    corpus = load_corpus()
    assert corpus, "no blessed fixtures — the corpus lock is gone"
    for fname, fix in corpus:
        result = name_blind_diff(fix["config"])
        assert result.structural_equal, \
            f"{fname}: name-blind structural drift: {result.changed_paths[:6]}"
