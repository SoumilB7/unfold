"""U8 neutral cross-owner call-argument binding controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.call_arguments import (
    bind_addressed_invocation,
    bind_repeated_child_call,
)
from model_unfolder.evidence.container_inventory import resolve_container_inventory
from model_unfolder.evidence.decoder_block import decoder_block_path_for_config
from model_unfolder.evidence.execution_flow import resolve_addressed_invocations
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


_SOURCE = """
from torch import nn

class Unit(nn.Module):
    def __init__(self, config):
        self.inner = Leaf(config)

    def forward(self, first, second=None):
        return self.inner(first, modifier=second)

class Leaf(nn.Module):
    def __init__(self, config):
        self.weight = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, value, modifier=None):
        return self.weight(value)

class Stage(nn.Module):
    def __init__(self, config):
        self.items = nn.ModuleList(
            [Unit(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden, auxiliary):
        for item in self.items:
            hidden = item(hidden, second=auxiliary)
        return hidden

class Wrapper(nn.Module):
    base_model_prefix = "model"

    def __init__(self, config):
        self.model = Stage(config)
"""


def _setup(tmp_path, source=_SOURCE):
    path = tmp_path / "modeling_calls.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = build_program_index(bundle)
    block = decoder_block_path_for_config(
        index, bundle, (), allow_root_stage=True)
    assert block.status == "resolved"
    return index, bundle, block.value


def test_repeated_call_binds_exact_positional_and_keyword_formals(tmp_path):
    index, _bundle, block = _setup(tmp_path)
    result = bind_repeated_child_call(
        index, block.component_root, block.repeated_child.proofs[0])
    assert result.status == "resolved"
    assert result.for_formal("first").actual.source_segment == "hidden"
    assert result.for_formal("second").actual.source_segment == "auxiliary"
    assert result.for_formal("self") is None


def test_formal_and_local_renaming_does_not_change_binding_law(tmp_path):
    source = (_SOURCE
              .replace("first, second=None", "alpha, beta=None")
              .replace("self.inner(first, modifier=second)",
                       "self.inner(alpha, modifier=beta)")
              .replace("item(hidden, second=auxiliary)",
                       "item(hidden, beta=auxiliary)"))
    index, _bundle, block = _setup(tmp_path, source)
    result = bind_repeated_child_call(
        index, block.component_root, block.repeated_child.proofs[0])
    assert result.status == "resolved"
    assert result.for_formal("alpha").actual.source_segment == "hidden"
    assert result.for_formal("beta").actual.source_segment == "auxiliary"


def test_expanded_kwargs_is_partial_but_exact_bindings_survive(tmp_path):
    source = _SOURCE.replace(
        "hidden = item(hidden, second=auxiliary)",
        "hidden = item(hidden, second=auxiliary, **kwargs)").replace(
            "def forward(self, hidden, auxiliary):",
            "def forward(self, hidden, auxiliary, **kwargs):")
    index, _bundle, block = _setup(tmp_path, source)
    result = bind_repeated_child_call(
        index, block.component_root, block.repeated_child.proofs[0])
    assert result.status == "partial"
    assert result.unresolved == ("expanded_kwargs",)
    assert result.for_formal("second").actual.source_segment == "auxiliary"


def test_duplicate_formal_binding_is_a_typed_failure(tmp_path):
    source = _SOURCE.replace(
        "item(hidden, second=auxiliary)",
        "item(hidden, first=auxiliary, second=auxiliary)")
    index, _bundle, block = _setup(tmp_path, source)
    result = bind_repeated_child_call(
        index, block.component_root, block.repeated_child.proofs[0])
    assert result.status == "failed"
    assert result.failure_kind == "duplicate_argument"


def test_wrong_program_index_cannot_reuse_a_foreign_owner_graph(tmp_path):
    index, _bundle, block = _setup(tmp_path)
    other_path = tmp_path / "other"
    other_path.mkdir()
    other_index, _other_bundle, _other_block = _setup(
        other_path, _SOURCE + "\n# distinct content identity\n")
    result = bind_repeated_child_call(
        other_index, block.component_root, block.repeated_child.proofs[0])
    assert result.status == "failed"
    assert result.failure_kind == "index_mismatch"


def test_direct_child_binding_uses_the_exact_addressed_invocation(tmp_path):
    index, _bundle, block = _setup(tmp_path)
    root = block.component_root
    inventory = resolve_container_inventory(
        index, root, block.block_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, block.block_occurrence, inventory)
    direct = next(item for item in invocations.addressed
                  if item.call.callee.source_segment == "self.inner")
    result = bind_addressed_invocation(index, root, direct)
    assert result.status == "resolved"
    assert result.for_formal("value").actual.source_segment == "first"
    assert result.for_formal("modifier").actual.source_segment == "second"


def test_result_closure_rejects_foreign_root_and_callable(tmp_path):
    index, _bundle, block = _setup(tmp_path)
    result = bind_repeated_child_call(
        index, block.component_root, block.repeated_child.proofs[0])
    binding = result.bindings[0]
    with pytest.raises(ValueError):
        replace(result, callee_symbol=block.stage_occurrence.root)
    with pytest.raises(ValueError):
        replace(binding, callee_callable=block.stage_occurrence.root)
    with pytest.raises(ValueError):
        replace(binding, actual=result.bindings[1].actual)


def test_real_llama_factor_formals_bind_across_both_owner_calls():
    from transformers import AutoConfig
    from model_unfolder.evidence.attention_child import attention_child_evidence
    from model_unfolder.evidence.context import ParseContext

    context = ParseContext.build(AutoConfig.for_model("llama"))
    index = context.program_index()
    block = decoder_block_path_for_config(
        index, context.source_bundle, (), allow_root_stage=True).value
    attention = attention_child_evidence(
        index, block.component_root, block.block_occurrence).value
    attention_args = bind_addressed_invocation(
        index, block.component_root, attention.invocation)
    block_args = bind_repeated_child_call(
        index, block.component_root, block.repeated_child.proofs[0])
    assert attention_args.status == "partial"  # explicit **kwargs remains visible
    assert block_args.status == "partial"
    assert attention_args.for_formal(
        "position_embeddings").actual.source_segment == "position_embeddings"
    assert block_args.for_formal(
        "position_embeddings").actual.source_segment == "position_embeddings"
