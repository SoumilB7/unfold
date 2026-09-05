"""U10-E independent companion-denoiser comparison controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.diffusion_companion import (
    CompanionDenoiserComparison,
    read_diffusion_companions,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


SOURCE = """
import torch
from torch import nn
from torch.nn import functional as F
class Mixer:
    def __init__(self, config):
        self.q = nn.Linear(config.width, config.width)
        self.k = nn.Linear(config.width, config.width)
        self.v = nn.Linear(config.width, config.width)
    def forward(self, x):
        return F.scaled_dot_product_attention(self.q(x), self.k(x), self.v(x))
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def forward(self, x):
        x = x + self.mix(x)
        return x
class Root:
    def __init__(self, config):
        self.units = nn.ModuleList([Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for unit in self.units:
            x = unit(x)
        return x
"""


DIFFERENT = SOURCE.replace(
    "x = x + self.mix(x)",
    "left = self.mix(x)\n        right = self.mix(x)\n        x = x + left + right")


def _write(tmp_path, name, source):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(root_file, companion_file=None, *, companion=True):
    component_files = {"root": (root_file,)}
    architectures = {"root": "Root"}
    companions = ()
    if companion:
        component_files["candidate"] = (() if companion_file is None
                                         else (companion_file,))
        architectures["candidate"] = "Root"
        companions = ("candidate",)
    files = tuple(dict.fromkeys((root_file,) + (
        (companion_file,) if companion_file is not None else ())))
    return SourceBundle(
        source="test", files=files, architecture="Root",
        component_files=component_files,
        component_architectures=architectures,
        companion_components=companions)


def _read(bundle):
    index = build_program_index(bundle)
    return read_diffusion_companions(index, bundle)


def test_same_source_is_only_same_source_contract_not_architecture_equivalence(tmp_path):
    path = _write(tmp_path, "same.py", SOURCE)
    result = _read(_bundle(path, path))
    comparison = result.require_value().comparisons[0]
    assert comparison.relation == "same_source_contract"
    assert comparison.architecture_equivalent is None
    assert comparison.primary.root.component_key == "root"
    assert comparison.companion.root.component_key == "candidate"


def test_different_source_with_matching_partial_evidence_is_not_equivalence(tmp_path):
    first = _write(tmp_path, "first.py", SOURCE)
    # Comment changes content identity while preserving all observed structure.
    second = _write(tmp_path, "second.py", "# separate implementation\n" + SOURCE)
    comparison = _read(_bundle(first, second)).require_value().comparisons[0]
    assert comparison.relation == "matching_partial_evidence"
    assert comparison.architecture_equivalent is None


def test_different_positive_structure_is_retained_even_if_dimensions_could_match(tmp_path):
    first = _write(tmp_path, "first.py", SOURCE)
    second = _write(tmp_path, "second.py", DIFFERENT)
    comparison = _read(_bundle(first, second)).require_value().comparisons[0]
    assert comparison.relation == "different_positive_evidence"
    assert comparison.primary.signature != comparison.companion.signature


def test_missing_companion_source_is_visible_unresolved(tmp_path):
    first = _write(tmp_path, "first.py", SOURCE)
    result = _read(_bundle(first, None))
    comparison = result.require_value().comparisons[0]
    assert comparison.relation == "unresolved"
    assert comparison.failure_kind == "missing_source"
    assert comparison.architecture_equivalent is None


def test_no_companion_address_is_absent_not_an_equivalence_claim(tmp_path):
    first = _write(tmp_path, "first.py", SOURCE)
    result = _read(_bundle(first, companion=False))
    assert result.status == "absent"


def test_parser_shadow_publishes_one_cached_companion_result(tmp_path):
    from model_unfolder.adapters.diffusor.parser import (
        _shadow_diffusion_companions,
    )
    from model_unfolder.evidence.context import ParseContext

    path = _write(tmp_path, "root.py", SOURCE)
    context = ParseContext(source_bundle=_bundle(path, companion=False))
    first = _shadow_diffusion_companions(context)
    second = _shadow_diffusion_companions(context)
    assert first is second
    assert first.status == "absent"
    assert context.reader_results[("root.denoiser.companions", ())] is first


def test_slot_spelling_and_dimensions_are_not_inputs_to_comparison(tmp_path):
    first = _write(tmp_path, "first.py", SOURCE)
    second = _write(tmp_path, "second.py", "# distinct\n" + SOURCE)
    original = _bundle(first, second)
    renamed = replace(
        original,
        component_files={"root": (first,), "opaque_slot": (second,)},
        component_architectures={"root": "Root", "opaque_slot": "Root"},
        companion_components=("opaque_slot",))
    a = _read(original).require_value().comparisons[0]
    b = _read(renamed).require_value().comparisons[0]
    assert a.relation == b.relation == "matching_partial_evidence"
    assert a.primary.signature == b.primary.signature
    assert a.companion.signature == b.companion.signature


def test_profiles_are_resolved_independently_not_reused_from_primary(tmp_path):
    first = _write(tmp_path, "first.py", SOURCE)
    second = _write(tmp_path, "second.py", DIFFERENT)
    comparison = _read(_bundle(first, second)).require_value().comparisons[0]
    assert comparison.primary is not comparison.companion
    assert comparison.primary.root.graph is not comparison.companion.root.graph
    assert comparison.primary.spans != comparison.companion.spans


def test_comparison_dto_rejects_false_same_source_claim(tmp_path):
    first = _write(tmp_path, "first.py", SOURCE)
    second = _write(tmp_path, "second.py", "# distinct\n" + SOURCE)
    comparison = _read(_bundle(first, second)).require_value().comparisons[0]
    with pytest.raises(ValueError, match="source"):
        CompanionDenoiserComparison(
            comparison.component_key, "same_source_contract",
            comparison.primary, comparison.companion)


def test_companion_component_cannot_be_root(tmp_path):
    first = _write(tmp_path, "first.py", SOURCE)
    comparison = _read(_bundle(first, first)).require_value().comparisons[0]
    with pytest.raises(ValueError, match="non-root"):
        replace(comparison, component_key="root")


def test_bundle_cannot_name_root_or_duplicate_companion_addresses(tmp_path):
    first = _write(tmp_path, "first.py", SOURCE)
    base = _bundle(first, first)
    for components in (("root",), ("candidate", "candidate")):
        result = _read(replace(base, companion_components=components))
        assert result.status == "failed"
        assert result.failures[0].kind == "conflict"


def test_loader_fetches_exact_duplicate_denoiser_component_independently(
        tmp_path, monkeypatch):
    import json
    import huggingface_hub
    from model_unfolder.adapters.diffusor.loader import load_diffusion_config_by_id

    pipeline = {
        "_class_name": "Pipeline",
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "opaque_companion": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    primary = {"_class_name": "FluxTransformer2DModel", "num_layers": 2}
    companion = {"_class_name": "FluxTransformer2DModel", "num_layers": 3}

    def fake_download(repo_id, filename, subfolder=None, token=None):
        if filename == "model_index.json" and subfolder is None:
            value = pipeline
        elif filename == "config.json" and subfolder == "transformer":
            value = primary
        elif filename == "config.json" and subfolder == "opaque_companion":
            value = companion
        else:
            raise FileNotFoundError
        path = tmp_path / f"{subfolder or 'root'}-{filename}"
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    loaded = load_diffusion_config_by_id("example/model")
    assert loaded["_companion_denoiser_configs"] == {
        "opaque_companion": companion}
    # A different pipeline component declaration is not promoted by its key.
    assert "vae" not in loaded["_companion_denoiser_configs"]


def test_source_bundle_keeps_companion_separate_from_encoder_slots():
    from model_unfolder.evidence.sources import resolve_source_files

    cfg = {
        "_class_name": "FluxTransformer2DModel",
        "_companion_denoiser_configs": {
            "opaque_companion": {"_class_name": "FluxTransformer2DModel"}},
    }
    bundle = resolve_source_files(cfg)
    assert bundle.companion_components == ("opaque_companion",)
    assert bundle.pipeline_components == ()
    assert bundle.component_architectures["opaque_companion"] \
        == "FluxTransformer2DModel"
    assert bundle.component_files["opaque_companion"]


def test_companion_config_envelope_is_explicit_loader_metadata():
    from model_unfolder.evidence.document import (
        LOADER_STAMPS,
        checkpoint_provenance,
    )

    key = "_companion_denoiser_configs"
    assert key in LOADER_STAMPS
    provenance = checkpoint_provenance({
        key: {"candidate": {"_class_name": "Root"}}})
    assert provenance[key] == "loader_metadata"
