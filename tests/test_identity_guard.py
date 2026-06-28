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
    ("model_unfolder/adapters/diffusor/parser.py",
     "identity_branch", "identity-derived predicate controls a branch"): 3,
    ("model_unfolder/adapters/diffusor/parser.py",
     "identity_helper", "runtime call to _class_default()"): 9,
    ("model_unfolder/adapters/transformer/parser.py",
     "identity_branch", "identity-derived predicate controls a branch"): 1,
    ("model_unfolder/adapters/transformer/special_parts/modalities/audio.py",
     "identity_helper", "runtime call to audio_family_hint()"): 1,
    ("model_unfolder/adapters/transformer/special_parts/modalities/detect.py",
     "identity_branch", "identity-derived predicate controls a branch"): 1,
    ("model_unfolder/adapters/transformer/special_parts/modalities/detect.py",
     "identity_helper", "runtime call to model_family_hint()"): 6,
    ("model_unfolder/adapters/transformer/special_parts/modalities/vision.py",
     "identity_branch", "identity-derived predicate controls a branch"): 1,
    ("model_unfolder/adapters/transformer/special_parts/modalities/vision.py",
     "identity_helper", "runtime call to model_family_hint()"): 1,
    ("model_unfolder/evidence/sources.py",
     "identity_helper", "runtime call to _guess_model_type_from_id()"): 1,
    ("model_unfolder/renderers/html/metadata_modalities.py",
     "identity_profile", "family profile selects rendered metadata"): 1,
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
