"""U3-F shared selected-config -> exact decoder-block address controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.decoder_block import (
    decoder_block_path_at_root,
    decoder_block_path_for_config,
)
from model_unfolder.evidence.models import SourceBundle


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _path(slug, config_path):
    config = json.loads((_CORPUS / f"{slug}.json").read_text())["config"]
    context = ParseContext.build(config)
    return decoder_block_path_for_config(
        context.program_index(), context.source_bundle, config_path,
        allow_root_stage=True)


@pytest.mark.parametrize(("slug", "config_path", "block_symbol"), [
    ("bloom", (), "BloomBlock"),
    ("musicgen-small", ("decoder",), "MusicgenDecoderLayer"),
    ("qwen2-vl-7b-instruct", ("text_config",), "Qwen2VLDecoderLayer"),
])
def test_real_selected_config_reaches_its_exact_decoder_block(
        slug, config_path, block_symbol):
    result = _path(slug, config_path)
    assert result.status == "resolved", result.failures
    value = result.value
    node = value.component_root.graph.node_for(value.block_occurrence)
    assert node is not None
    assert node.symbol.qualified_name == block_symbol
    assert value.config_path == config_path
    assert value.repeated_child.status == "resolved"
    if config_path:
        assert value.address_spans
        assert any("config-scope construction" in item.detail
                   for item in result.provenance)


def test_nested_decoder_path_cannot_be_forged_as_its_stage_or_root():
    result = _path("musicgen-small", ("decoder",))
    assert result.status == "resolved"
    value = result.value
    with pytest.raises(ValueError):
        replace(value, stage_occurrence=value.block_occurrence)
    with pytest.raises(ValueError):
        replace(
            value,
            repeated_child=replace(
                value.repeated_child,
                model_stage=value.block_occurrence))
    with pytest.raises(ValueError):
        replace(value, config_path=("foreign",))
    with pytest.raises(ValueError):
        replace(value, address_spans=())


def test_declared_root_path_cannot_be_forged_as_nested():
    result = _path("bloom", ())
    assert result.status == "resolved"
    with pytest.raises(ValueError):
        replace(result.value, config_path=("decoder",))
    with pytest.raises(ValueError):
        replace(
            result.value,
            address_spans=(result.value.repeated_child.proofs[0].template.call.span,))


def test_delegation_depth_is_bounded_by_graph_cycles_not_magic_number(
        tmp_path):
    chain = []
    depth = 10
    for index in range(depth):
        child = f"Stage{index + 1}" if index + 1 < depth else "Stack"
        chain.append(
            f"class Stage{index}:\n"
            f"    def __init__(self, config): self.child = {child}(config)\n"
            "    def forward(self, x): return self.child(x)\n"
        )
    source = (
        "from torch import nn\n"
        "class Block:\n"
        "    def __init__(self, config): pass\n"
        "    def forward(self, x): return x\n"
        "class Stack:\n"
        "    def __init__(self, config):\n"
        "        self.items = nn.ModuleList("
        "[Block(config) for _ in range(config.layers)])\n"
        "    def forward(self, x):\n"
        "        for item in self.items:\n"
        "            x = item(x)\n"
        "        return x\n"
        + "".join(reversed(chain))
        + "class Wrapper:\n"
        "    base_model_prefix = 'model'\n"
        "    def __init__(self, config): self.model = Stage0(config)\n"
    )
    path = tmp_path / "model.py"
    path.write_text(source, encoding="utf-8")
    bundle = SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"

    result = decoder_block_path_at_root(
        index, root, allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert len(result.value.stage_occurrence.sites) == depth + 1
    block = result.value.component_root.graph.node_for(
        result.value.block_occurrence)
    assert block is not None
    assert block.symbol.qualified_name == "Block"


def test_exact_return_delegation_beats_unrelated_repeated_container_rivals(
        tmp_path):
    source = """
from torch import nn
class Noise:
    def __init__(self, config): pass
    def forward(self, x): return x
class Block:
    def __init__(self, config): pass
    def forward(self, x): return x
class Core:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class Stage:
    def __init__(self, config):
        self.left = nn.ModuleList(
            [Noise(config) for _ in range(config.layers)])
        self.right = nn.ModuleList(
            [Noise(config) for _ in range(config.layers)])
        self.core = Core(config)
    def forward(self, x):
        return self.core(x)
class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Stage(config)
"""
    path = tmp_path / "model.py"
    path.write_text(source, encoding="utf-8")
    bundle = SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"

    result = decoder_block_path_at_root(
        index, root, allow_root_stage=True)
    assert result.status == "resolved", result.failures
    block = result.value.component_root.graph.node_for(
        result.value.block_occurrence)
    assert block is not None
    assert block.symbol.qualified_name == "Block"


def test_unconstructed_config_path_never_falls_back_to_root_decoder():
    result = _path("qwen2-vl-7b-instruct", ("does_not_exist",))
    assert result.status in {"absent", "failed"}
    assert result.status != "resolved"


def test_decoder_path_rejects_a_non_exact_config_path_shape():
    config = json.loads(
        (_CORPUS / "bloom.json").read_text())["config"]
    context = ParseContext.build(config)
    with pytest.raises(TypeError):
        decoder_block_path_for_config(
            context.program_index(), context.source_bundle,
            ["decoder"], allow_root_stage=True)
