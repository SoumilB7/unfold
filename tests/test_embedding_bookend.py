"""U3-F4 — exact pre-stack normalization reader controls."""
from __future__ import annotations

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
from model_unfolder.evidence.embedding_bookend import (
    embedding_stage_norm_evidence,
    read_embedding_stage_norm,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.repeated_child import resolve_repeated_child


_SOURCE = """
    import torch
    from torch import nn

    class Unit:
        def __init__(self, config): pass
        def forward(self, x): return x

    class CustomNorm:
        def __init__(self, config):
            self.eps = config.eps
        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + self.eps)

    class Core:
        def __init__(self, config):
            self.embedding = nn.Embedding(config.vocab, config.hidden)
            self.entry = CustomNorm(config)
            self.units = nn.ModuleList(
                [Unit(config) for _ in range(config.layers)])
        def forward(self, token_ids, inputs_embeds=None):
            if inputs_embeds is None:
                inputs_embeds = self.embedding(token_ids)
            hidden = self.entry(inputs_embeds)
            for unit in self.units:
                hidden = unit(hidden)
            return hidden

    class Shell:
        base_model_prefix = "core"
        def __init__(self, config):
            self.core = Core(config)
"""


def _write(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _pipeline(tmp_path, source=_SOURCE):
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="local",
        files=(path,),
        component_files={"root": (path,)},
        component_architectures={"root": "Shell"},
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    repeated = resolve_repeated_child(index, root, stage, inventory)
    return bundle, index, root, stage, inventory, repeated


def test_custom_rms_norm_feeding_exact_repeated_child_resolves(tmp_path):
    bundle, index, root, stage, inventory, repeated = _pipeline(tmp_path)
    result = read_embedding_stage_norm(
        index, root, stage.occurrence, inventory, repeated)
    assert result.status == "resolved"
    assert result.value == "RMSNorm"
    assert result.owner == stage.occurrence
    assert result.provenance[0].spans
    assert embedding_stage_norm_evidence(index, bundle).value == "RMSNorm"


def test_external_layernorm_protocol_feeding_stack_resolves(tmp_path):
    source = _SOURCE.replace(
        "self.entry = CustomNorm(config)",
        "self.entry = nn.LayerNorm(config.hidden)",
    )
    bundle, index, *_ = _pipeline(tmp_path, source)
    result = embedding_stage_norm_evidence(index, bundle)
    assert result.status == "resolved"
    assert result.value == "LayerNorm"


def test_explicit_transformer_authorization_supports_a_bare_model_root(tmp_path):
    source = _SOURCE.replace(
        """    class Shell:
        base_model_prefix = "core"
        def __init__(self, config):
            self.core = Core(config)
""",
        "",
    )
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="local", files=(path,), component_files={"root": (path,)},
        component_architectures={"root": "Core"})
    index = pi.build_program_index(bundle)
    refused = embedding_stage_norm_evidence(index, bundle)
    accepted = embedding_stage_norm_evidence(
        index, bundle, allow_root_stage=True)
    assert refused.status == "failed"
    assert accepted.status == "resolved"
    assert accepted.value == "RMSNorm"


def test_final_norm_cannot_be_reversed_into_an_entry_bookend(tmp_path):
    source = _SOURCE.replace(
        "            hidden = self.entry(inputs_embeds)\n",
        "",
    ).replace(
        "            return hidden",
        "            return self.entry(hidden)",
    )
    bundle, index, *_ = _pipeline(tmp_path, source)
    result = embedding_stage_norm_evidence(index, bundle)
    assert result.status == "failed"
    assert "feeds the repeated child" in result.failures[0].detail


def test_guarded_norm_does_not_author_an_unconditional_diagram_block(tmp_path):
    source = _SOURCE.replace(
        "            hidden = self.entry(inputs_embeds)",
        """            hidden = inputs_embeds
            if token_ids is not None:
                hidden = self.entry(hidden)""",
    )
    bundle, index, *_ = _pipeline(tmp_path, source)
    result = embedding_stage_norm_evidence(index, bundle)
    assert result.status == "failed"
    assert "guarded" in result.failures[0].detail


def test_two_repeated_child_occurrences_are_ambiguity_not_a_picked_stack(tmp_path):
    source = _SOURCE.replace(
        """            for unit in self.units:
                hidden = unit(hidden)
            return hidden""",
        """            for unit in self.units:
                hidden = unit(hidden)
            for unit in self.other_units:
                hidden = unit(hidden)
            return hidden""",
    ).replace(
        """            self.units = nn.ModuleList(
                [Unit(config) for _ in range(config.layers)])""",
        """            self.units = nn.ModuleList(
                [Unit(config) for _ in range(config.layers)])
            self.other_units = nn.ModuleList(
                [Unit(config) for _ in range(config.layers)])""",
    )
    _, index, root, stage, inventory, repeated = _pipeline(tmp_path, source)
    assert repeated.status == "ambiguous"
    result = read_embedding_stage_norm(
        index, root, stage.occurrence, inventory, repeated)
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_cross_owner_inventory_cannot_clear_the_reader(tmp_path):
    _, index, root, stage, inventory, repeated = _pipeline(tmp_path)
    root_inventory = resolve_container_inventory(
        index, root, root.graph.root.occurrence)
    result = read_embedding_stage_norm(
        index, root, stage.occurrence, root_inventory, repeated)
    assert result.status == "failed"
    assert result.failures[0].kind == "out_of_owner"


@pytest.mark.parametrize(("slug", "expected"), [
    ("bloom", "LayerNorm"),
    ("llama-7b", None),
    ("gemma-2-2b-it", None),
])
def test_real_transformer_bookend_controls(slug, expected):
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    data = json.loads((corpus / f"{slug}.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    result = embedding_stage_norm_evidence(
        context.program_index(), context.source_bundle,
        allow_root_stage=True)
    if expected is None:
        assert result.status != "resolved"
    else:
        assert result.status == "resolved"
        assert result.value == expected
