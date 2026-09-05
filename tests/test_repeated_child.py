"""U3-F2 — exact repeated-child occurrence boundary poisons."""
from __future__ import annotations

from dataclasses import replace
import json
import pathlib
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.container_inventory import resolve_container_inventory
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.repeated_child import (
    RepeatedChildResolution,
    resolve_repeated_child,
)


_MODEL = """
    class Block:
        def __init__(self, config): pass
        def forward(self, x): return x
    class Other:
        def __init__(self, config): pass
        def forward(self, x): return x
    class BaseModel:
        def __init__(self, config):
            self.layers = ModuleList([Block(config) for _ in range(config.n)])
            self.others = ModuleList([Other(config) for _ in range(config.m)])
        def forward(self, x):
        # BODY
    class Wrapper:
        base_model_prefix = "model"
        def __init__(self, config):
            self.model = BaseModel(config)
"""


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(paths, arch="Wrapper"):
    return SourceBundle(
        source="local",
        files=tuple(paths),
        component_files={"root": tuple(paths)},
        component_architectures={"root": arch},
    )


def _pipeline(tmp_path, body, source=_MODEL):
    path = _write(tmp_path, "model.py", source.replace("        # BODY", body))
    bundle = _bundle((path,))
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    result = resolve_repeated_child(index, root, stage, inventory)
    return index, root, stage, inventory, result


@pytest.mark.parametrize("body,kind", [
    ("""
            for layer in self.layers:
                x = layer(x)
            return x""", "direct"),
    ("""
            for layer in self.layers[: 2]:
                x = layer(x)
            return x""", "sliced"),
    ("""
            for index, layer in enumerate(self.layers):
                x = layer(x)
            return x""", "enumerated"),
    ("""
            for index, layer in enumerate(self.layers[: 2]):
                x = layer(x)
            return x""", "enumerated_sliced"),
])
def test_supported_iteration_shapes_resolve_one_exact_graph_child(
        tmp_path, body, kind):
    index, root, stage, inventory, result = _pipeline(tmp_path, body)
    assert result.status == "resolved"
    assert result.child_symbol.qualified_name == "Block"
    assert root.graph.node_for(result.child_occurrence).symbol == result.child_symbol
    (proof,) = result.proofs
    assert proof.template.iteration_kind == kind
    assert proof.child_occurrence == result.child_occurrence
    assert proof.child_occurrence.sites[-1] == proof.template.element_template.site_id


def test_two_calls_through_one_template_do_not_fabricate_two_children(tmp_path):
    _, _, _, _, result = _pipeline(tmp_path, """
            for layer in self.layers:
                x = layer(x)
                x = layer(x)
            return x""")
    assert result.status == "resolved"
    assert len(result.proofs) == 2
    assert {proof.child_occurrence for proof in result.proofs} == {
        result.child_occurrence}


def test_two_executed_containers_are_exact_ambiguous_occurrences(tmp_path):
    _, root, stage, _, result = _pipeline(tmp_path, """
            for layer in self.layers:
                x = layer(x)
            for other in self.others:
                x = other(x)
            return x""")
    assert result.status == "ambiguous"
    assert len({proof.child_occurrence for proof in result.rivals}) == 2
    assert {proof.child_symbol.qualified_name for proof in result.rivals} == {
        "Block", "Other"}
    assert all(root.graph.node_for(proof.child_occurrence) is not None
               for proof in result.rivals)


def test_same_child_class_at_two_sites_stays_occurrence_ambiguous(tmp_path):
    source = _MODEL.replace(
        "self.others = ModuleList([Other(config) for _ in range(config.m)])",
        "self.others = ModuleList([Block(config) for _ in range(config.m)])")
    _, _, _, _, result = _pipeline(tmp_path, """
            for layer in self.layers:
                x = layer(x)
            for other in self.others:
                x = other(x)
            return x""", source)
    assert result.status == "ambiguous"
    assert {proof.child_symbol.qualified_name for proof in result.rivals} == {"Block"}
    assert len({proof.child_occurrence for proof in result.rivals}) == 2


def test_no_positive_repeated_invocation_is_incomplete_not_absent(tmp_path):
    _, _, _, _, result = _pipeline(tmp_path, "            return x")
    assert result.status == "incomplete"
    assert result.incomplete_reasons
    assert "absent" not in result.status


def test_shadowed_enumerate_remains_incomplete(tmp_path):
    source = _MODEL.replace(
        "    class Block:",
        "    enumerate = object()\n    class Block:")
    _, _, _, _, result = _pipeline(tmp_path, """
            for index, layer in enumerate(self.layers):
                x = layer(x)
            return x""", source)
    assert result.status == "incomplete"


def test_heterogeneous_container_does_not_pick_the_first_element(tmp_path):
    source = _MODEL.replace(
        "ModuleList([Block(config) for _ in range(config.n)])",
        "ModuleList([Block(config), Other(config)])")
    _, _, _, _, result = _pipeline(tmp_path, """
            for layer in self.layers:
                x = layer(x)
            return x""", source)
    assert result.status == "incomplete"
    assert any("heterogeneous" in reason for reason in result.incomplete_reasons)


def test_inventory_from_another_owner_is_a_typed_failure(tmp_path):
    index, root, stage, inventory, result = _pipeline(
        tmp_path, "            return x")
    root_inventory = resolve_container_inventory(
        index, root, root.graph.root.occurrence)
    failed = resolve_repeated_child(index, root, stage, root_inventory)
    assert failed.status == "failed"
    assert failed.failure_kind == "inventory_mismatch"
    with pytest.raises(TypeError):
        resolve_repeated_child(index, root, stage, object())


def test_broken_component_file_cannot_be_bypassed(tmp_path):
    good = _write(tmp_path, "good.py", _MODEL.replace(
        "        # BODY", "            return x"))
    broken = _write(tmp_path, "broken.py", "class Broken(:\n")
    bundle = _bundle((good, broken))
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "failed"
    with pytest.raises(ValueError):
        resolve_declared_model_stage(index, root)


def test_resolution_dto_rejects_laundered_status_payloads(tmp_path):
    _, _, _, _, result = _pipeline(tmp_path, """
            for layer in self.layers:
                x = layer(x)
            return x""")
    with pytest.raises(ValueError):
        replace(result, status="resolved", proofs=())
    with pytest.raises(ValueError):
        replace(result, status="incomplete", incomplete_reasons=())
    with pytest.raises(ValueError):
        RepeatedChildResolution(
            "failed", result.model_stage, failure_kind="guessed")


def test_complete_renaming_changes_no_address_law(tmp_path):
    renamed = (_MODEL
        .replace("Block", "Unit")
        .replace("Other", "Alternate")
        .replace("BaseModel", "Core")
        .replace("Wrapper", "Shell")
        .replace('base_model_prefix = "model"', 'base_model_prefix = "engine"')
        .replace("self.model =", "self.engine =")
        .replace("self.layers", "self.units")
        .replace("for layer in self.units", "for item in self.units")
        .replace("x = layer(x)", "x = item(x)"))
    path = _write(tmp_path, "renamed.py", renamed.replace("        # BODY", """
            for item in self.units[: 2]:
                x = item(x)
            return x"""))
    bundle = _bundle((path,), arch="Shell")
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    result = resolve_repeated_child(index, root, stage, inventory)
    assert result.status == "resolved"
    assert result.proofs[0].template.iteration_kind == "sliced"


@pytest.mark.parametrize("slug,kind,expected_symbol", [
    ("bloom", "enumerated", "BloomBlock"),
    ("deepseek-v3", "sliced", "DeepseekV3DecoderLayer"),
    ("gemma-2-2b-it", "enumerated_sliced", "Gemma2DecoderLayer"),
    ("glm-4-5", "sliced", "Glm4MoeDecoderLayer"),
    ("gpt-oss-20b", "enumerated", "GptOssDecoderLayer"),
    ("llama-7b", "sliced", "LlamaDecoderLayer"),
    ("olmo-2-1124-7b", "sliced", "Olmo2DecoderLayer"),
    ("qwen3-8b", "enumerated_sliced", "Qwen3DecoderLayer"),
    ("stablelm-2-1-6b", "direct", "StableLmDecoderLayer"),
])
def test_real_transformer_model_stages_resolve_exact_repeated_children(
        slug, kind, expected_symbol):
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    data = json.loads((corpus / f"{slug}.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    result = resolve_repeated_child(index, root, stage, inventory)
    assert result.status == "resolved"
    assert result.child_symbol.qualified_name == expected_symbol
    assert result.proofs[0].template.iteration_kind == kind
    assert root.graph.node_for(result.child_occurrence) is not None


def test_qwen2_vl_nested_text_stack_is_not_laundered_into_the_b1_stage():
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    data = json.loads((corpus / "qwen2-vl-7b-instruct.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    result = resolve_repeated_child(index, root, stage, inventory)
    assert result.status == "incomplete"
    assert result.child_occurrence is None
