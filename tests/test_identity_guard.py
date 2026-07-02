"""Negative controls and the temporary report-only identity debt snapshot."""
from __future__ import annotations

from collections import Counter

from model_unfolder.evidence.identity_guard import (
    name_blind_diff,
    scan_identity_debt,
    scan_identity_source,
    scan_identity_yaml_source,
)


# The pin is the MULTISET of (file, kind, detail) -> count, deliberately NOT
# keyed on line number.  Pinning by line made every edit that inserts a line
# above a debt item a false failure that trained us to bump numbers blindly.
# A count-per-(file, kind, detail) is immune to line shifts yet still trips on a
# genuinely new identity usage: a new call of an existing kind bumps the count,
# a new kind/file adds a key.  Runtime triage keeps line numbers (each finding
# still carries .line); only this regression comparison drops them.
EXPECTED_REPORT_ONLY_DEBT = {
    ("model_unfolder/adapters/diffusor/blocks.py",
     "class_identity_branch", "class-name/domain substring controls an architectural branch"): 3,
    ("model_unfolder/adapters/diffusor/parser.py",
     "class_identity_branch", "class-name/domain substring controls an architectural branch"): 1,
    ("model_unfolder/adapters/diffusor/parser.py",
     "class_marker_table", "runtime access to class-name marker vocabulary 'dit_class_markers'"): 1,
    ("model_unfolder/adapters/diffusor/parser.py",
     "class_marker_table", "runtime access to class-name marker vocabulary 'scheduler_flow_matching_markers'"): 1,
    ("model_unfolder/adapters/transformer/parser.py",
     "identity_branch", "identity-derived predicate controls a branch"): 1,
    ("model_unfolder/evidence/conformance.py",
     "class_marker_table", "runtime access to class-name marker vocabulary 'component_class_markers'"): 2,
    ("model_unfolder/evidence/conformance.py",
     "class_marker_table", "runtime access to class-name marker vocabulary 'drill_class_markers'"): 1,
    ("model_unfolder/evidence/conformance.py",
     "class_marker_table", "runtime access to class-name marker vocabulary 'processor_markers'"): 2,
    ("model_unfolder/evidence/conformance.py",
     "class_marker_table", "runtime access to class-name marker vocabulary 'single_stream_class_markers'"): 1,
    ("model_unfolder/evidence/sources.py",
     "class_marker_table", "runtime access to class-name marker vocabulary 'dit_class_markers'"): 1,
    ("model_unfolder/everchanging/conformance/conformance_map.yaml",
     "identity_table", "populated class-name marker table 'single_stream_class_markers' can select architecture"): 1,
    ("model_unfolder/everchanging/conformance/transitive.yaml",
     "identity_table", "populated class-name marker table 'component_class_markers' can select architecture"): 1,
    ("model_unfolder/everchanging/conformance/transitive.yaml",
     "identity_table", "populated class-name marker table 'drill_class_markers' can select architecture"): 1,
    ("model_unfolder/everchanging/conformance/transitive.yaml",
     "identity_table", "populated class-name marker table 'processor_markers' can select architecture"): 1,
    ("model_unfolder/everchanging/diffusor/typing.yaml",
     "identity_table", "populated class-name marker table 'dit_class_markers' can select architecture"): 1,
    ("model_unfolder/everchanging/diffusor/typing.yaml",
     "identity_table", "populated class-name marker table 'scheduler_flow_matching_markers' can select architecture"): 1,
}


def test_report_only_identity_debt_is_pinned_and_cannot_grow():
    actual = Counter((item.path, item.kind, item.detail) for item in scan_identity_debt())
    assert dict(actual) == EXPECTED_REPORT_ONLY_DEBT


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
