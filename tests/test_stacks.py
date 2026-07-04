"""The GENERAL secondary-stack detector (evidence/stacks.py).

Pins the general behaviors, never one model: nested constructor-chain count
resolution, direct range(count) binding, lane extraction from plain and
enumerate() loops and from delegated calls, and the negative control (a plain
LLM yields exactly its root stack)."""
from __future__ import annotations

import json
import pathlib

import pytest

from model_unfolder.evidence.stacks import secondary_stacks_from_files


def _diffusion_stacks(slug: str, architecture: str):
    from model_unfolder.evidence.conformance import (
        _augment_diffusion_files,
        _component_source,
    )
    from model_unfolder.evidence.sources import resolve_source_files

    fixture = pathlib.Path(__file__).parent / "sable_corpus" / f"{slug}.json"
    config = json.loads(fixture.read_text())["config"]
    bundle = resolve_source_files(config, source="local")
    _component, files = _component_source(bundle, "text")
    if not files:
        pytest.skip("modeling source not installed")
    return secondary_stacks_from_files(_augment_diffusion_files(files), architecture)


def test_nested_constructor_chain_resolves_count_field_and_text_lane():
    """HunyuanVideo's refiner ModuleList lives two constructor hops below the
    architecture; the count must trace ``range(num_layers)`` -> kwarg
    ``num_layers=num_refiner_layers`` -> the config field, and the lane is the
    forward argument fed to the top of the chain."""
    stacks = _diffusion_stacks("hunyuanvideo", "HunyuanVideoTransformer3DModel")
    by_field = {s.field_name: s for s in stacks}
    refiner = by_field["refiner_blocks"]
    assert refiner.count_field == "num_refiner_layers"
    assert refiner.lane_param == "encoder_hidden_states"
    assert refiner.block_class == "HunyuanVideoIndividualTokenRefinerBlock"
    # Root stacks are detected too (the CONSUMER excludes them by count field);
    # the fused single-stream block counts as block-shaped by field roles.
    assert by_field["transformer_blocks"].count_field == "num_layers"
    assert by_field["single_transformer_blocks"].count_field == "num_single_layers"


def test_same_block_class_stacks_distinguish_by_count_field_and_lane():
    """Lumina2 reuses ONE block class for the root stack and both refiners —
    exclusion by class would erase the refiners; count field + lane are the
    distinguishing facts."""
    stacks = _diffusion_stacks("lumina-image-2-0", "Lumina2Transformer2DModel")
    by_field = {s.field_name: s for s in stacks}
    assert by_field["layers"].count_field == "num_layers"
    assert by_field["context_refiner"].count_field == "num_refiner_layers"
    assert by_field["context_refiner"].lane_param == "encoder_hidden_states"
    assert by_field["noise_refiner"].count_field == "num_refiner_layers"
    assert by_field["noise_refiner"].lane_param == "hidden_states"


def test_plain_llm_negative_control_yields_exactly_the_root_stack():
    import transformers

    root = pathlib.Path(transformers.__file__).parent / "models/llama"
    stacks = secondary_stacks_from_files(
        sorted(root.glob("modeling_*.py")), "LlamaForCausalLM")
    assert [(s.owner_class, s.field_name, s.count_field) for s in stacks] == [
        ("LlamaModel", "layers", "num_hidden_layers"),
    ]
