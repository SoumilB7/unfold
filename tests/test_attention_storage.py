"""U3-F5b — exact attention Q/K/V storage controls."""
from __future__ import annotations

from dataclasses import replace
import json
import pathlib
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.attention_storage import (
    attention_projection_storage_evidence,
    decoder_attention_projection_storage_evidence,
    decoder_attention_projection_storage_mode_evidence,
)
from model_unfolder.evidence.construction_calls import ConstructionOccurrenceId
from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.container_inventory import resolve_container_inventory
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.repeated_child import resolve_repeated_child


_SOURCE = """
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Compute:
        def __init__(self, config):
            self.a = nn.Linear(config.hidden, config.hidden)
            self.b = nn.Linear(config.hidden, config.hidden)
            self.c = nn.Linear(config.hidden, config.hidden)
            self.out = nn.Linear(config.hidden, config.hidden)
        def forward(self, x):
            one = self.a(x)
            two = self.b(x)
            three = self.c(x)
            weights = F.softmax(torch.matmul(one, two), dim=-1)
            mixed = torch.matmul(weights, three)
            return self.out(mixed)

    class Other:
        def __init__(self, config):
            self.up = nn.Linear(config.hidden, config.wide)
            self.down = nn.Linear(config.wide, config.hidden)
        def forward(self, x):
            return self.down(F.gelu(self.up(x)))

    class Cell:
        def __init__(self, config):
            self.left = Compute(config)
            self.right = Other(config)
        def forward(self, x):
            x = self.left(x)
            return self.right(x)

    class Core:
        def __init__(self, config):
            self.items = nn.ModuleList(
                [Cell(config) for _ in range(config.layers)])
        def forward(self, x):
            for item in self.items:
                x = item(x)
            return x

    class Wrapper:
        base_model_prefix = "core"
        def __init__(self, config):
            self.core = Core(config)
"""


def _pipeline(tmp_path, source=_SOURCE):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    repeated = resolve_repeated_child(index, root, stage, inventory)
    assert repeated.status == "resolved"
    return index, root, repeated


def test_three_exact_linear_producers_feeding_compute_are_split(tmp_path):
    index, root, repeated = _pipeline(tmp_path)
    result = attention_projection_storage_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "resolved"
    assert result.value.mode == "split"
    assert len(result.value.projections) == 3


def test_one_linear_feeding_three_lane_unpack_is_fused(tmp_path):
    source = _SOURCE.replace(
        """            self.a = nn.Linear(config.hidden, config.hidden)
            self.b = nn.Linear(config.hidden, config.hidden)
            self.c = nn.Linear(config.hidden, config.hidden)""",
        "            self.packed = nn.Linear(config.hidden, config.hidden * 3)",
    ).replace(
        """        def forward(self, x):
            one = self.a(x)
            two = self.b(x)
            three = self.c(x)""",
        """        def unpack(self, packed):
            return packed, packed, packed
        def forward(self, x):
            packed = self.packed(x)
            one, two, three = self.unpack(packed)""",
    )
    index, root, repeated = _pipeline(tmp_path, source)
    result = attention_projection_storage_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "resolved"
    assert result.value.mode == "fused_qkv"
    assert len(result.value.projections) == 1


def test_helper_that_ignores_packed_projection_cannot_claim_fused(tmp_path):
    source = _SOURCE.replace(
        """            self.a = nn.Linear(config.hidden, config.hidden)
            self.b = nn.Linear(config.hidden, config.hidden)
            self.c = nn.Linear(config.hidden, config.hidden)""",
        "            self.packed = nn.Linear(config.hidden, config.hidden * 3)",
    ).replace(
        """        def forward(self, x):
            one = self.a(x)
            two = self.b(x)
            three = self.c(x)""",
        """        def unpack(self, packed):
            return 1, 2, 3
        def forward(self, x):
            packed = self.packed(x)
            one, two, three = self.unpack(packed)""",
    )
    index, root, repeated = _pipeline(tmp_path, source)
    result = attention_projection_storage_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "failed"


def test_chained_low_rank_projections_are_not_three_split_qkv_lanes(tmp_path):
    source = _SOURCE.replace(
        "            two = self.b(x)",
        "            two = self.b(one)",
    )
    index, root, repeated = _pipeline(tmp_path, source)
    result = attention_projection_storage_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "failed"


def test_three_linears_not_reaching_compute_do_not_prove_split(tmp_path):
    source = _SOURCE.replace(
        "            weights = F.softmax(torch.matmul(one, two), dim=-1)",
        """            one = x
            weights = F.softmax(torch.matmul(one, two), dim=-1)""",
    )
    index, root, repeated = _pipeline(tmp_path, source)
    result = attention_projection_storage_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "failed"


def test_guarded_projection_definition_cannot_prove_split(tmp_path):
    source = _SOURCE.replace(
        "            one = self.a(x)",
        """            if self.training:
                one = self.a(x)
            else:
                one = x""",
    )
    index, root, repeated = _pipeline(tmp_path, source)
    result = attention_projection_storage_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "failed"


def test_unrelated_linear_before_compute_does_not_change_qkv_storage(tmp_path):
    source = _SOURCE.replace(
        "            self.out = nn.Linear(config.hidden, config.hidden)",
        """            self.extra = nn.Linear(config.hidden, config.hidden)
            self.out = nn.Linear(config.hidden, config.hidden)""",
    ).replace(
        "            one = self.a(x)",
        """            ignored = self.extra(x)
            one = self.a(x)""",
    )
    index, root, repeated = _pipeline(tmp_path, source)
    result = attention_projection_storage_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "resolved"
    assert result.value.mode == "split"


def test_familiar_projection_field_names_without_compute_are_not_evidence(tmp_path):
    source = _SOURCE.replace(
        "            weights = F.softmax(torch.matmul(one, two), dim=-1)",
        "            weights = one + two",
    ).replace(
        "            mixed = torch.matmul(weights, three)",
        "            mixed = weights + three",
    )
    index, root, repeated = _pipeline(tmp_path, source)
    result = attention_projection_storage_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "failed"


def test_cross_owner_block_is_not_reused(tmp_path):
    index, root, repeated = _pipeline(tmp_path)
    result = attention_projection_storage_evidence(
        index, root, root.graph.root.occurrence)
    assert result.status == "failed"
    with pytest.raises(TypeError):
        attention_projection_storage_evidence(index, root, object())


def test_storage_dto_rejects_projection_from_another_owner(tmp_path):
    index, root, repeated = _pipeline(tmp_path)
    result = attention_projection_storage_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "resolved"
    first, *rest = result.value.projections
    forged = ConstructionOccurrenceId(
        root.graph.root.occurrence, first.site)
    with pytest.raises(ValueError):
        replace(result.value, projections=(forged, *rest))


def test_dispatch_selected_attention_is_not_silently_picked():
    """Falcon's block constructs attention through a config-keyed class map.

    F5b must not pick one registry candidate, nor use the familiar Falcon
    spelling.  A later candidate-equivalence/config-address boundary must prove
    the selected implementation before production cutover.
    """
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    context = ParseContext.build(
        _coerce(AutoConfig.for_model("falcon").to_dict()))
    result = decoder_attention_projection_storage_evidence(
        context.program_index(), context.source_bundle,
        allow_root_stage=True)
    assert result.status == "failed"
    assert "unresolved constructed-child calls" in result.failures[0].detail
    mode = decoder_attention_projection_storage_mode_evidence(
        context.program_index(), context.source_bundle,
        allow_root_stage=True)
    assert mode.status == "resolved", mode.failures
    assert mode.value == "fused_qkv"


@pytest.mark.parametrize(("slug", "expected"), [
    ("bloom", "fused_qkv"),
    ("gemma-2-2b-it", "split"),
    ("gpt-oss-20b", "split"),
    ("llama-7b", "split"),
    ("olmo-2-1124-7b", "split"),
    ("qwen3-8b", "split"),
    ("stablelm-2-1-6b", "split"),
    ("deepseek-v3", None),
])
def test_real_decoder_storage_examples(slug, expected):
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    data = json.loads((corpus / f"{slug}.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    repeated = resolve_repeated_child(index, root, stage, inventory)
    result = attention_projection_storage_evidence(
        index, root, repeated.child_occurrence)
    if expected is None:
        assert result.status != "resolved"
    else:
        assert result.status == "resolved", result.failures
        assert result.value.mode == expected


@pytest.mark.parametrize(("slug", "expected"), [
    ("bloom", "fused_qkv"),
    ("llama-7b", "split"),
    ("deepseek-v3", None),
])
def test_high_level_storage_mode_preserves_direct_model_controls(
        slug, expected):
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    data = json.loads((corpus / f"{slug}.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    result = decoder_attention_projection_storage_mode_evidence(
        context.program_index(), context.source_bundle,
        allow_root_stage=True)
    if expected is None:
        assert result.status != "resolved"
    else:
        assert result.status == "resolved", result.failures
        assert result.value == expected
