"""U6 complete-dispatch attention mechanism controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.attention import bind_attention_mechanism
from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.container_inventory import resolve_container_inventory
from model_unfolder.evidence.dispatch_attention_mechanism import (
    EquivalentDispatchMultiQueryBinding,
    dispatch_multi_query_attention_binding_at_block,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.repeated_child import resolve_repeated_child


_SOURCE = """
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Projection(nn.Linear):
        pass

    class Eager(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.width = config.hidden
            self.red = config.query_groups
            self.blue = config.use_one_kv
            self.green = config.alternate_layout
            self.unit = self.width // self.red
            self.kv = config.shared_groups if (self.green or not self.blue) else 1
            self.packed = Projection(self.width, self.width + 2 * self.unit)
            self.out = Projection(self.width, self.width)
        def partition(self, packed):
            if self.green:
                shaped = packed.view(-1, self.red, 3, self.unit)
                return shaped[..., 0, :], shaped[..., 1, :], shaped[..., 2, :]
            elif not self.blue:
                shaped = packed.view(-1, self.red, 3, self.unit)
                return shaped[..., 0, :], shaped[..., 1, :], shaped[..., 2, :]
            else:
                packed = packed.view(-1, self.red + 2, self.unit)
                return packed[..., :-2, :], packed[..., [-2], :], packed[..., [-1], :]
        def forward(self, x):
            packed = self.packed(x)
            q, k, v = self.partition(packed)
            mixed = F.scaled_dot_product_attention(q, k, v)
            return self.out(mixed)

    class Flash(Eager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    CHOICES = {"eager": Eager, "flash": Flash}

    class Cell(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.branch = CHOICES[config.implementation](config)
        def forward(self, x):
            return self.branch(x)

    class Core(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.items = nn.ModuleList(
                [Cell(config) for _ in range(config.layers)])
        def forward(self, x):
            for item in self.items:
                x = item(x)
            return x

    class Wrapper(nn.Module):
        base_model_prefix = "core"
        def __init__(self, config):
            super().__init__()
            self.core = Core(config)
"""


def _pipeline(tmp_path, source=_SOURCE):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"})
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    repeated = resolve_repeated_child(index, root, stage, inventory)
    return index, root, repeated.child_occurrence


def test_complete_dispatch_census_proves_one_mqa_protocol(tmp_path):
    index, root, block = _pipeline(tmp_path)
    result = dispatch_multi_query_attention_binding_at_block(
        index, root, block)
    assert result.status == "resolved", result.failures
    value = result.value
    assert isinstance(value, EquivalentDispatchMultiQueryBinding)
    assert (value.num_heads_path, value.selector_path,
            value.alternate_architecture_path) == (
                ("query_groups",), ("use_one_kv",),
                ("alternate_layout",))
    assert len(value.proofs) == 2
    assert not hasattr(value, "attention_occurrence")


def test_dispatch_binding_requires_true_selector_and_false_alternate(tmp_path):
    index, root, block = _pipeline(tmp_path)
    binding = dispatch_multi_query_attention_binding_at_block(
        index, root, block).value
    good = {
        binding.num_heads_path: 16,
        binding.selector_path: True,
        binding.alternate_architecture_path: False,
        ("shared_groups",): 16,
    }
    bound = bind_attention_mechanism(binding, good)
    assert (bound.kind, bound.num_heads, bound.num_kv_heads) == ("mqa", 16, 1)
    assert ("shared_groups",) not in dict(bound.premises)
    assert bind_attention_mechanism(
        binding, {**good, binding.selector_path: False}) is None
    assert bind_attention_mechanism(
        binding, {**good, binding.alternate_architecture_path: True}) is None


def test_flag_without_singleton_split_is_powerless_for_every_candidate(tmp_path):
    source = _SOURCE.replace(
        "packed = packed.view(-1, self.red + 2, self.unit)",
        "packed = packed.view(-1, self.red, 3, self.unit)")
    index, root, block = _pipeline(tmp_path, source)
    result = dispatch_multi_query_attention_binding_at_block(
        index, root, block)
    assert result.status == "failed"


def test_singleton_helper_that_does_not_feed_attention_is_powerless(tmp_path):
    source = _SOURCE.replace(
        "q, k, v = self.partition(packed)\n            mixed = F.scaled_dot_product_attention(q, k, v)",
        "unused_q, unused_k, unused_v = self.partition(packed)\n"
        "            q, k, v = packed, packed, packed\n"
        "            mixed = F.scaled_dot_product_attention(q, k, v)")
    index, root, block = _pipeline(tmp_path, source)
    result = dispatch_multi_query_attention_binding_at_block(
        index, root, block)
    assert result.status == "failed"


def test_rebound_varargs_cannot_certify_inherited_config_paths(tmp_path):
    source = _SOURCE.replace(
        "def __init__(self, *args, **kwargs):\n            super().__init__(*args, **kwargs)",
        "def __init__(self, *args, **kwargs):\n"
        "            args = ()\n"
        "            super().__init__(*args, **kwargs)")
    index, root, block = _pipeline(tmp_path, source)
    result = dispatch_multi_query_attention_binding_at_block(
        index, root, block)
    assert result.status == "failed"


def test_one_non_equivalent_registry_candidate_blocks_the_union(tmp_path):
    source = _SOURCE.replace(
        "class Flash(Eager):\n        def __init__(self, *args, **kwargs):\n            super().__init__(*args, **kwargs)",
        "class Flash(Eager):\n        def __init__(self, config):\n            super().__init__(config)\n            self.blue = config.other_selector")
    index, root, block = _pipeline(tmp_path, source)
    result = dispatch_multi_query_attention_binding_at_block(
        index, root, block)
    assert result.status in {"failed", "ambiguous"}


def test_result_closure_rejects_missing_candidate_and_path_laundering(
        tmp_path):
    index, root, block = _pipeline(tmp_path)
    value = dispatch_multi_query_attention_binding_at_block(
        index, root, block).value
    with pytest.raises(ValueError):
        replace(value, proofs=value.proofs[:1])
    with pytest.raises(ValueError):
        replace(value, selector_path=value.num_heads_path)
    with pytest.raises(TypeError):
        replace(value, block_occurrence=object())


def test_real_falcon_uses_code_bound_selector_not_class_default_kv_count():
    from transformers import AutoConfig
    from model_unfolder import config_to_ir
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    config = _coerce(AutoConfig.for_model("falcon").to_dict())
    context = ParseContext.build(config)
    # The public parser route must use the same dispatch binding and must not
    # let FalconConfig's class-supplied num_kv_heads=71 outrank multi_query=True.
    ir = config_to_ir(config, parse_context=context)
    assert ir.layers[0].attention.kind == "mqa"
    assert ir.layers[0].attention.num_heads == 71
    assert ir.layers[0].attention.num_kv_heads == 1
    fact = context.facts.typed["decoder.attention.mechanism"]
    assert fact.status == "code_and_config"
    assert set(fact.config_paths) == {
        "num_attention_heads", "multi_query", "new_decoder_architecture"}
