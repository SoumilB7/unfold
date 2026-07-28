"""U3-F5b — exact attention Q/K/V storage controls."""
from __future__ import annotations

from dataclasses import replace
import json
import pathlib
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.attention_storage import (
    attention_projection_storage_mode_at_root,
    attention_projection_storage_evidence,
    decoder_attention_projection_storage_evidence,
    decoder_attention_projection_storage_for_path,
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


def test_two_unguarded_attention_children_never_become_unanimous_storage(
        tmp_path):
    source = _SOURCE.replace(
        "            self.right = Other(config)",
        "            self.right = Compute(config)",
    )
    index, root, _ = _pipeline(tmp_path, source)
    result = attention_projection_storage_mode_at_root(
        index, root, allow_root_stage=False)
    assert result.status == "ambiguous"


def test_one_unguarded_attention_child_is_not_laundered_by_guarded_secondary(
        tmp_path):
    source = _SOURCE.replace(
        "            self.right = Other(config)",
        "            self.right = Compute(config)",
    ).replace(
        """        def forward(self, x):
            x = self.left(x)
            return self.right(x)""",
        """        def forward(self, x, side=None):
            x = self.left(x)
            if side is not None:
                x = self.right(x)
            return x""",
    )
    index, root, _ = _pipeline(tmp_path, source)
    result = attention_projection_storage_mode_at_root(
        index, root, allow_root_stage=False)
    assert result.status == "resolved"
    assert result.value == "split"
    assert any("uniquely unguarded" in item.detail
               for item in result.provenance)


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


def test_guarded_projection_definition_without_reaching_proof_stays_unknown(
        tmp_path):
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


def test_guarded_projection_construction_cannot_prove_split(tmp_path):
    source = _SOURCE.replace(
        "            self.a = nn.Linear(config.hidden, config.hidden)",
        """            if config.use_a:
                self.a = nn.Linear(config.hidden, config.hidden)""",
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


def test_musicgen_nested_decoder_storage_uses_exact_constructed_scope():
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    data = json.loads((corpus / "musicgen-small.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    result = decoder_attention_projection_storage_for_path(
        context.program_index(), context.source_bundle, ("decoder",),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value == "split"
    assert any("config-scope construction" in item.detail
               for item in result.provenance)
    assert any("forward's unconditional return" in item.detail
               for item in result.provenance)
    assert any("uniquely unguarded" in item.detail
               for item in result.provenance)


@pytest.mark.parametrize(("slug", "path", "value", "status"), [
    ("musicgen-small", ("decoder",), "split_qkv", "code_proven"),
    ("bloom", (), "fused_qkv", "code_proven"),
    ("llama-7b", (), "split_qkv", "code_proven"),
    # Low-rank/chained projection dataflow is deliberately outside the current
    # proof.  It must stay conventional/visible debt, never become false code
    # certainty merely because the model is in the same broad decoder family.
    ("deepseek-v3", (), None, "ambiguous"),
    # The wrapper's text_config is a distinct config/owner scope: its exact
    # direct field construction resolves independently of the sibling vision
    # tower, whose fused QKV storage must not contaminate this split decoder.
    ("qwen2-vl-7b-instruct", ("text_config",), "split_qkv", "code_proven"),
])
def test_parser_projection_fact_consumes_the_exact_path_reader(
        slug, path, value, status):
    from model_unfolder import config_to_ir
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    data = json.loads((corpus / f"{slug}.json").read_text())
    cfg = _coerce(data["config"])
    context = ParseContext.build(cfg)
    config_to_ir(cfg, parse_context=context)

    assert context.selected_config_paths["transformer.main"] == path
    fact = context.facts.records["decoder.attention.projection_mode"]
    assert (fact.value, fact.status) == (value, status)
    if status == "code_proven":
        assert fact.source == \
            "decoder_attention_projection_storage_for_path"
    else:
        assert fact.source is None


def test_parser_and_conformance_share_one_exact_storage_result(monkeypatch):
    from model_unfolder import config_to_ir
    from model_unfolder.evidence import attention_storage as storage_module
    from model_unfolder.evidence.conformance import check_fact_conformance
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    data = json.loads((corpus / "musicgen-small.json").read_text())
    cfg = _coerce(data["config"])
    context = ParseContext.build(cfg)
    real_reader = storage_module.decoder_attention_projection_storage_for_path
    calls = []

    def counted_reader(*args, **kwargs):
        calls.append((args[2], kwargs.get("allow_root_stage")))
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(
        storage_module, "decoder_attention_projection_storage_for_path",
        counted_reader)
    ir = config_to_ir(cfg, parse_context=context)
    key = ("decoder.attention.projection_storage", ("decoder",))
    parsed_result = context.reader_results[key]
    decoder_calls = [call for call in calls if call[0] == ("decoder",)]
    assert decoder_calls == [(("decoder",), True)]

    problems = check_fact_conformance(
        cfg, ir.to_dict(), bundle=context.source_bundle,
        program_index=context.program_index(), parse_context=context)
    assert not [problem for problem in problems
                if problem.kind == "wrong_storage"]
    assert context.reader_results[key] is parsed_result
    assert [call for call in calls if call[0] == ("decoder",)] == decoder_calls


def test_standalone_conformance_never_assumes_root_for_a_nested_decoder():
    from model_unfolder.evidence.conformance import (
        _storage_config_path_for_conformance,
    )
    from model_unfolder.evidence.models import SourceBundle

    root_only = SourceBundle(
        source="local", files=("root.py",),
        component_files={"root": ("root.py",)})
    assert _storage_config_path_for_conformance(
        root_only, "root") == ()

    nested = SourceBundle(
        source="local", files=("root.py", "text.py"),
        component_files={
            "root": ("root.py",),
            "thinker_config.text_config": ("text.py",),
        })
    assert _storage_config_path_for_conformance(
        nested, "thinker_config.text_config") \
        == ("thinker_config", "text_config")
    assert _storage_config_path_for_conformance(
        nested, "root") is None

    class Context:
        selected_config_paths = {}

    # A context that has not run the parser carries no selection receipt.  It
    # must abstain rather than silently upgrading that absence to root.
    assert _storage_config_path_for_conformance(
        root_only, "root", Context()) is None


def test_parser_falcon_dispatch_equivalence_is_code_proven():
    from transformers import AutoConfig
    from model_unfolder import config_to_ir
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    cfg = _coerce(AutoConfig.for_model("falcon").to_dict())
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    fact = context.facts.records["decoder.attention.projection_mode"]
    assert (fact.value, fact.status, fact.source) == (
        "fused_qkv", "code_proven",
        "decoder_attention_projection_storage_for_path")
    assert ir.layers[0].attention.projection_mode == "fused_qkv"
